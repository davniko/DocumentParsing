"""Private capacity contracts for printed, identifier-unassigned equipment mixes."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from decimal import Decimal
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, model_validator

from document_ocr.hashing import sha256_bytes
from document_ocr.label_schemas.bill_of_lading_v5 import (
    CONTAINER_SIZE_CATEGORIES,
    CONTAINER_TYPE_CATEGORIES,
)
from document_ocr.synthesis.transport_capacity import (
    TransportCapacityLimits,
    classify_equipment,
    equipment_capacity,
)


class EquipmentMixMember(BaseModel):
    """A printed count of one canonical size/type, without identifier ownership."""

    model_config = ConfigDict(extra="forbid", frozen=True)
    count: int = Field(gt=0, strict=True)
    sizeCategory: str
    typeCategory: str

    @model_validator(mode="after")
    def canonical_pair(self) -> EquipmentMixMember:
        if self.sizeCategory not in CONTAINER_SIZE_CATEGORIES:
            raise ValueError("equipment mix requires a canonical size category")
        if self.typeCategory not in CONTAINER_TYPE_CATEGORIES:
            raise ValueError("equipment mix requires a canonical type category")
        return self


class PrintedEquipmentMix(BaseModel):
    """Source-certified equipment multiset; never creates public container labels."""

    model_config = ConfigDict(extra="forbid", frozen=True)
    source_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    source_text_literals: tuple[str, ...] = Field(min_length=1)
    members: tuple[EquipmentMixMember, ...] = Field(min_length=1)

    @model_validator(mode="after")
    def distinct_members(self) -> PrintedEquipmentMix:
        pairs = [(m.sizeCategory, m.typeCategory) for m in self.members]
        if len(set(pairs)) != len(pairs):
            raise ValueError("equipment mix must consolidate repeated canonical pairs")
        if any(not text.strip() for text in self.source_text_literals):
            raise ValueError("equipment mix requires nonempty source evidence")
        return self

    def validate_source(self, text: str, containers: Sequence[Mapping[str, Any]]) -> None:
        if sha256_bytes(text.encode()) != self.source_sha256:
            raise ValueError("equipment mix source hash differs")
        if any(text.count(literal) != 1 for literal in self.source_text_literals):
            raise ValueError("equipment mix evidence is missing or not uniquely owned")
        if sum(member.count for member in self.members) != len(containers):
            raise ValueError("equipment mix count differs from retained container identifiers")
        # This contract solves unassigned mixes, not independently classified rows.
        # Such rows must retain their own capacity/identity contract instead.
        if any("sizeCategory" in row or "typeCategory" in row for row in containers):
            raise ValueError("unassigned equipment mix conflicts with public equipment categories")

    def capacities(self, limits: TransportCapacityLimits) -> list[dict[str, str | None]]:
        result = []
        for member in self.members:
            pair = member.model_dump(exclude={"count"})
            if classify_equipment(pair) in {"unclassified", "unsupported_reefer"}:
                raise ValueError("equipment mix lacks a supported canonical capacity contract")
            capacity = equipment_capacity(pair, limits)
            result.extend(
                {
                    "payloadKg": str(capacity.payload_kg),
                    "volumeM3": str(capacity.volume_m3) if capacity.volume_m3 is not None else None,
                }
                for _ in range(member.count)
            )
        return result


def capacity_assignment(
    loads: Sequence[Mapping[str, Decimal]], capacities: Sequence[Mapping[str, Any]]
) -> tuple[int, ...] | None:
    """Find a joint payload/cube matching, not an identifier classification.

    Augmenting-path matching is polynomial, rather than enumerating permutations.
    A witness proves only that at least one assignment fits the printed multiset.
    """
    if len(loads) != len(capacities) or not loads:
        raise ValueError("equipment mix requires one load per capacity")
    limits = []
    for capacity in capacities:
        payload = Decimal(str(capacity["payloadKg"]))
        volume = capacity["volumeM3"]
        volume = Decimal(str(volume)) if volume is not None else None
        if (
            not payload.is_finite()
            or payload <= 0
            or (volume is not None and (not volume.is_finite() or volume <= 0))
        ):
            raise ValueError("equipment mix capacities must be finite and positive")
        limits.append((payload, volume))
    edges = []
    for load in loads:
        if set(load) - {"grossWeight", "netWeight", "volume"}:
            raise ValueError("equipment mix load contains an unknown physical dimension")
        if any(not value.is_finite() or value < 0 for value in load.values()):
            raise ValueError("equipment mix loads must be finite and nonnegative")
        mass = max(load.get("grossWeight", Decimal(0)), load.get("netWeight", Decimal(0)))
        edges.append(
            [
                index
                for index, (payload, volume) in enumerate(limits)
                if mass <= payload and (volume is None or load.get("volume", 0) <= volume)
            ]
        )
    matched = [-1] * len(capacities)

    def augment(row: int, visited: set[int]) -> bool:
        for index in edges[row]:
            if index in visited:
                continue
            visited.add(index)
            if matched[index] < 0 or augment(matched[index], visited):
                matched[index] = row
                return True
        return False

    for row in sorted(range(len(loads)), key=lambda index: len(edges[index])):
        if not augment(row, set()):
            return None
    assignment = [-1] * len(loads)
    for index, row in enumerate(matched):
        assignment[row] = index
    return tuple(assignment)
