"""Reviewed, train-isolated three-port chains; not a live carrier timetable.

Terminal aliases are explicit evidence annotations pinned to the exact source
target. No route is inferred merely because three ports exist in a registry.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Annotated, Any

from pydantic import BaseModel, ConfigDict, StringConstraints, model_validator

from document_ocr.hashing import canonical_json_bytes, sha256_bytes
from document_ocr.synthesis.country_registry import CountryRegistry
from document_ocr.synthesis.routes import Locode, RouteLocation

Text = Annotated[str, StringConstraints(min_length=1)]


class ReviewedPort(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)
    source_name: Text
    locode: Locode
    evidence: Text


class TransshipmentObservation(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)
    source_document_id: Text
    source_target_sha256: Annotated[str, StringConstraints(pattern=r"^[0-9a-f]{64}$")]
    port_of_loading: ReviewedPort
    transshipment_port: ReviewedPort
    port_of_discharge: ReviewedPort

    @model_validator(mode="after")
    def distinct_ports(self) -> TransshipmentObservation:
        codes = (
            self.port_of_loading.locode,
            self.transshipment_port.locode,
            self.port_of_discharge.locode,
        )
        if len(set(codes)) != 3 or codes[0][:2] == codes[2][:2]:
            raise ValueError(
                "transshipment observations require three distinct international ports"
            )
        return self


@dataclass(frozen=True, slots=True)
class TransshipmentChain:
    source_document_id: str
    loading: RouteLocation
    transshipment: RouteLocation
    discharge: RouteLocation


def reviewed_chains(
    observations: Sequence[TransshipmentObservation],
    *,
    source_targets: Mapping[str, Mapping[str, Any]],
    fit_document_ids: Sequence[str],
    maritime_ports: Mapping[str, RouteLocation],
    countries: CountryRegistry,
) -> tuple[TransshipmentChain, ...]:
    fit_ids = set(fit_document_ids)
    seen: set[str] = set()
    result = []
    for observation in observations:
        sid = observation.source_document_id
        if sid not in fit_ids or sid in seen:
            raise ValueError("transshipment evidence must be unique and inside the train-only fit")
        seen.add(sid)
        target = source_targets[sid]
        if sha256_bytes(canonical_json_bytes(target)) != observation.source_target_sha256:
            raise ValueError(f"transshipment evidence target hash differs: {sid}")
        route = target["documentPatch"].get("route", {})
        ports = []
        for role, reviewed in (
            ("portOfLoading", observation.port_of_loading),
            ("transshipmentPort", observation.transshipment_port),
            ("portOfDischarge", observation.port_of_discharge),
        ):
            source = route.get(role, {})
            if source.get("name") != reviewed.source_name:
                raise ValueError(f"transshipment evidence differs from printed role: {sid}:{role}")
            port = maritime_ports.get(reviewed.locode)
            if port is None:
                raise ValueError(
                    f"transshipment port lacks pinned maritime support: {reviewed.locode}"
                )
            country = source.get("country")
            if country is not None and countries.resolve(country) != port.country_code:
                raise ValueError(f"transshipment port contradicts printed country: {sid}:{role}")
            ports.append(port)
        result.append(TransshipmentChain(sid, *ports))
    return tuple(sorted(result, key=lambda row: row.source_document_id))
