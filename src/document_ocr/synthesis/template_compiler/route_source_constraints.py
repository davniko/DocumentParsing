"""Source-certified route and freight restrictions for immutable printed context.

These certificates constrain *sampling support* before a scenario is drawn and
verify the sampled result afterward. They cannot repair an incompatible OCR
after rendering or silently keep a source value in place of a requested change.
"""

from __future__ import annotations

import csv
import io
import re
import unicodedata
import zipfile
from collections.abc import Mapping
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Annotated, Any, Literal

from pydantic import BaseModel, ConfigDict, Field, StringConstraints

from document_ocr.hashing import canonical_json_bytes, sha256_bytes, sha256_file
from document_ocr.synthesis.country_registry import CountryRegistry
from document_ocr.synthesis.shipment_scenarios import (
    ScenarioSupport,
    WeightedCategory,
    normalize_location_name,
)

from . import complete_targets as targets
from .party_address_roles import PartyAddressRoleCertificate, required_immutable_admin1

_STRICT = ConfigDict(extra="forbid", frozen=True, strict=True)
_SHA = Annotated[str, StringConstraints(pattern=r"^[0-9a-f]{64}$")]
_COUNTRY = Annotated[str, StringConstraints(pattern=r"^[A-Z]{2}$")]
_LEGAL = re.compile(
    rb"Container detention tariffs and conditions applicable for the port of loading,"
    rb"[\s\S]{0,256}?https://www\.maersk\.com/local-information/[^\s]*/"
    rb"(?P<country>[A-Za-z-]+)/export",
    re.I,
)


class SourceSpan(BaseModel):
    model_config = _STRICT

    byte_start: Annotated[int, Field(ge=0)]
    byte_end: Annotated[int, Field(gt=0)]
    source_text: Annotated[str, StringConstraints(min_length=1)]


class RouteAdmissibilityCertificate(BaseModel):
    """Reviewed source ownership of one country-specific loading policy."""

    model_config = _STRICT

    schema_version: Literal[1]
    source_document_id: str
    source_sha256: _SHA
    source_label_sha256: _SHA
    template_sha256: _SHA
    commercial_origin_country_code: _COUNTRY
    loading_country_code: _COUNTRY
    shipper_country_code: _COUNTRY
    freight_arrangement: Literal["prepaid", "collect"]
    source_loading_port_locode: Annotated[str, StringConstraints(pattern=r"^[A-Z]{2}[A-Z0-9]{3}$")]
    loading_country_clause: SourceSpan


@dataclass(frozen=True, slots=True)
class Admin1Membership:
    """Pinned GeoNames/UN-LOCODE crosswalk, keyed by actual location identities."""

    geoname_admin1: Mapping[int, str]
    subdivision_admin1: Mapping[tuple[str, str], str]
    admin1_names_by_code: Mapping[str, str]

    def region_for_locality(self, locality: Mapping[str, Any]) -> tuple[str, str]:
        """Resolve one sampled locality to its pinned admin1 code and canonical name."""

        country = locality.get("countryCode")
        city_id = locality.get("geonameId")
        subdivision = locality.get("subdivisionCode")
        source = locality.get("source")
        if not isinstance(country, str) or re.fullmatch(r"[A-Z]{2}", country) is None:
            raise ValueError("sampled locality has no valid country identity")
        city_mode = source == "geonames_population_weighted"
        endpoint_mode = source in {
            "observed_port",
            "registry_port_exploration",
            "registry_port_no_observed_support",
            "endpoint",
            "train_observed_transshipment_chain",
        }
        if city_mode:
            if not isinstance(city_id, int) or isinstance(city_id, bool) or subdivision is not None:
                raise ValueError("sampled GeoNames locality has conflicting identities")
            code = self.geoname_admin1.get(city_id)
        elif endpoint_mode:
            if city_id is not None or not isinstance(subdivision, str) or not subdivision:
                raise ValueError("sampled endpoint lacks an unambiguous subdivision")
            code = self.subdivision_admin1.get((country, subdivision))
        else:
            raise ValueError("sampled locality has an unsupported identity source")
        if code is None or not code.startswith(country + "."):
            raise ValueError("sampled locality has no matching pinned admin1 membership")
        name = self.admin1_names_by_code.get(code)
        if name is None:
            raise ValueError("sampled locality admin1 has no pinned canonical name")
        return code, name


def _name(value: str) -> str:
    value = unicodedata.normalize("NFKD", value).encode("ascii", "ignore").decode()
    return " ".join(re.findall(r"[a-z0-9]+", value.casefold()))


def load_admin1_membership(
    *,
    cities_archive: Path,
    cities_sha256: str,
    admin1_codes: Path,
    admin1_sha256: str,
    unlocode_archive: Path,
    unlocode_sha256: str,
) -> Admin1Membership:
    """Join source-pinned populated places and port subdivisions by named admin1."""

    for path, expected in (
        (cities_archive, cities_sha256),
        (admin1_codes, admin1_sha256),
        (unlocode_archive, unlocode_sha256),
    ):
        if sha256_file(path) != expected:
            raise ValueError(f"administrative-area dependency hash differs: {path}")
    admin1_names: dict[tuple[str, str], str] = {}
    admin1_display_names: dict[str, str] = {}
    by_name: dict[tuple[str, str], set[str]] = {}
    for line in admin1_codes.read_text().splitlines():
        parts = line.split("\t")
        if len(parts) != 4 or "." not in parts[0]:
            raise ValueError("GeoNames admin1 row has an invalid shape")
        country, _ = parts[0].split(".", 1)
        if not parts[2] or (
            parts[0] in admin1_display_names and admin1_display_names[parts[0]] != parts[2]
        ):
            raise ValueError("GeoNames admin1 has an absent or conflicting canonical name")
        admin1_display_names[parts[0]] = parts[2]
        admin1_names[(country, parts[0])] = _name(parts[1])
        by_name.setdefault((country, _name(parts[1])), set()).add(parts[0])
        by_name.setdefault((country, _name(parts[2])), set()).add(parts[0])
    city_membership: dict[int, str] = {}
    with zipfile.ZipFile(cities_archive) as archive:
        city_rows = archive.read("cities15000.txt").decode("utf-8").splitlines()
    for row in city_rows:
        parts = row.split("\t")
        if len(parts) < 11:
            raise ValueError("GeoNames city row has no country/admin1 columns")
        admin1 = parts[8] + "." + parts[10]
        if (parts[8], admin1) not in admin1_names:
            continue
        geoname_id = int(parts[0])
        if geoname_id in city_membership:
            raise ValueError("GeoNames city identifier repeats in pinned archive")
        city_membership[geoname_id] = admin1
    subdivision_membership: dict[tuple[str, str], str] = {}
    with zipfile.ZipFile(unlocode_archive) as archive:
        subdivision_rows = csv.reader(
            io.StringIO(archive.read("release/csv/SubdivisionCodes.csv").decode("utf-8"))
        )
        for subdivision_row in subdivision_rows:
            if len(subdivision_row) < 3:
                raise ValueError("UN/LOCODE subdivision row has an invalid shape")
            country, subdivision, name = subdivision_row[:3]
            matches = by_name.get((country, _name(name)), set())
            if len(matches) == 1:
                key = (country, subdivision)
                resolved = next(iter(matches))
                if key in subdivision_membership and subdivision_membership[key] != resolved:
                    raise ValueError("UN/LOCODE subdivision has conflicting GeoNames identities")
                subdivision_membership[key] = resolved
    return Admin1Membership(city_membership, subdivision_membership, admin1_display_names)


def requires_certificate(source: bytes) -> bool:
    """Prevent country-specific loading clauses from escaping unconditioned sampling."""

    return bool(_LEGAL.search(source))


def validate_certificate(
    certificate: RouteAdmissibilityCertificate,
    *,
    source: targets.SourceTemplate,
    support: ScenarioSupport,
    countries: CountryRegistry,
) -> None:
    """Pin every route restriction to printed source and registry identities."""

    if (
        certificate.source_document_id != source.document_id
        or certificate.source_sha256 != sha256_bytes(source.source)
        or certificate.source_label_sha256
        != sha256_bytes(canonical_json_bytes(source.source_target))
        or certificate.template_sha256 != source.original_template_sha256
    ):
        raise ValueError("route admissibility certificate has a different source preimage")
    span = certificate.loading_country_clause
    if source.source[span.byte_start : span.byte_end] != span.source_text.encode("utf-8"):
        raise ValueError("route legal clause differs from the exact printed source span")
    clauses = tuple(_LEGAL.finditer(source.source))
    if len(clauses) != 1 or (
        clauses[0].start() != span.byte_start or clauses[0].end() != span.byte_end
    ):
        raise ValueError("country-specific loading clause is absent or ambiguous")
    country_name = clauses[0]["country"].decode("ascii")
    if (
        countries.require(country_name, field="source loading-policy URL")
        != certificate.loading_country_code
    ):
        raise ValueError("loading-country claim contradicts printed legal-policy URL")
    shipper = source.source_target["documentPatch"]["parties"]["shipper"]
    if (
        countries.require(shipper["country"], field="source shipper")
        != certificate.shipper_country_code
    ):
        raise ValueError("source shipper country contradicts route certificate")
    if certificate.commercial_origin_country_code != certificate.shipper_country_code:
        raise ValueError("source-certified shipper must own commercial-origin country")
    printed_country = tuple(
        slot
        for binding in source.template.bindings
        if "documentPatch.parties.shipper.country" in binding.target_paths
        for slot in binding.occurrences
    )
    if not printed_country or any(
        countries.require(slot.source_text, field="printed shipper country")
        != certificate.shipper_country_code
        for slot in printed_country
    ):
        raise ValueError("source shipper country lacks a matching printed binding")
    port_name = source.source_target["documentPatch"]["route"]["portOfLoading"]["name"]
    ports = tuple(
        row
        for row in support.maritime_by_country.get(certificate.loading_country_code, ())
        if row.locode == certificate.source_loading_port_locode
        and normalize_location_name(row.name) == normalize_location_name(port_name)
    )
    if len(ports) != 1:
        raise ValueError("source loading port has no unique pinned UN/LOCODE country owner")
    loading_slots = tuple(
        slot
        for binding in source.template.bindings
        if "documentPatch.route.portOfLoading.name" in binding.target_paths
        for slot in binding.occurrences
    )
    if not loading_slots or any(
        normalize_location_name(slot.source_text) != normalize_location_name(port_name)
        or source.source[slot.byte_start : slot.byte_end] != slot.source_text.encode("utf-8")
        for slot in loading_slots
    ):
        raise ValueError("source loading port lacks matching printed route ownership")
    freight = source.source_target["documentPatch"]["freight"]["paymentArrangement"]
    if freight != certificate.freight_arrangement:
        raise ValueError("source freight label contradicts reviewed arrangement")
    payment_slots = tuple(
        slot
        for binding in source.template.bindings
        if "documentPatch.freight.paymentArrangement" in binding.target_paths
        for slot in binding.occurrences
    )
    if not payment_slots or any(slot.source_text.casefold() != freight for slot in payment_slots):
        raise ValueError("source freight arrangement lacks exact printed ownership")
    if not all(
        source.source[slot.byte_start : slot.byte_end] == slot.source_text.encode()
        for slot in payment_slots
    ):
        raise ValueError("printed freight slots differ from source bytes")


def _admin1_constraint(
    party_certificate: PartyAddressRoleCertificate | None,
) -> tuple[str, str] | None:
    if party_certificate is None:
        return None
    matching = [
        group
        for group in party_certificate.groups
        if group.address_target_path == "documentPatch.parties.shipper.address"
    ]
    if len(matching) != 1:
        raise ValueError("shipper address role is absent or ambiguous")
    return required_immutable_admin1(party_certificate, matching[0].group_key)


def condition_support(
    support: ScenarioSupport,
    certificate: RouteAdmissibilityCertificate,
    *,
    party_certificate: PartyAddressRoleCertificate | None = None,
    admin1: Admin1Membership | None = None,
) -> ScenarioSupport:
    """Restrict only source-proven choices; preserve the pinned priors within them."""

    origin = certificate.commercial_origin_country_code
    loading = certificate.loading_country_code
    observed = tuple(row for row in support.observed_export_countries if row.value == origin)
    registry = tuple(
        row for row in support.maritime_registry_export_countries if row.value == origin
    )
    loading_rows = tuple(
        row for row in support.physical_loading_countries_by_origin[origin] if row.value == loading
    )
    if len(observed) != 1 or registry != (WeightedCategory(origin, 1),) or not loading_rows:
        raise ValueError("source-restricted origin/loading lacks supported route prior")
    payment_present = False
    freight_rows = tuple(
        row
        for row in support.freight_arrangements_by_payment_presence[payment_present]
        if row.value == certificate.freight_arrangement
    )
    if len(freight_rows) != 1:
        raise ValueError("source-restricted freight lacks support at the template topology")
    localities = dict(support.localities_by_country)
    maritime = dict(support.maritime_by_country)
    observed_ports = dict(support.observed_loading_ports)
    fixed_admin1 = _admin1_constraint(party_certificate)
    if fixed_admin1 is not None:
        if admin1 is None or fixed_admin1[0] != certificate.shipper_country_code:
            raise ValueError("source-fixed shipper region has no matching pinned registry")
        country, region = fixed_admin1
        eligible_cities = tuple(
            row
            for row in localities[country]
            if admin1.geoname_admin1.get(row.geoname_id) == region
        )
        eligible_ports = tuple(
            row
            for row in maritime[country]
            if admin1.subdivision_admin1.get((country, row.subdivision_code or "")) == region
        )
        if not eligible_cities or not eligible_ports:
            raise ValueError("source-fixed shipper region lacks city/endpoint port support")
        localities[country] = eligible_cities
        maritime[country] = eligible_ports
        port_codes = {row.locode for row in eligible_ports}
        observed_ports[country] = tuple(
            row for row in observed_ports.get(country, ()) if row.value in port_codes
        )
    return replace(
        support,
        observed_export_countries=observed,
        maritime_registry_export_countries=registry,
        physical_loading_countries_by_origin={origin: loading_rows},
        loading_country_methods_by_origin={
            origin: support.loading_country_methods_by_origin[origin]
        },
        trade_destinations_by_origin={origin: support.trade_destinations_by_origin[origin]},
        freight_arrangements_by_payment_presence={
            **support.freight_arrangements_by_payment_presence,
            payment_present: freight_rows,
        },
        localities_by_country=localities,
        maritime_by_country=maritime,
        observed_loading_ports=observed_ports,
    )


def validate_scenario(
    certificate: RouteAdmissibilityCertificate,
    scenario: Mapping[str, Any],
    *,
    party_certificate: PartyAddressRoleCertificate | None = None,
    admin1: Admin1Membership | None = None,
) -> None:
    """Independently check all country, freight, and immutable-region assertions."""

    if (
        scenario["commercialOriginCountryCode"] != certificate.commercial_origin_country_code
        or scenario["loadingPort"]["countryCode"] != certificate.loading_country_code
        or scenario["freight"]["arrangement"] != certificate.freight_arrangement
    ):
        raise ValueError("sampled route or freight violates source-certified context")
    shippers = [
        row
        for row in scenario["partyLocalities"]
        if row["role"] == "shipper" and row["occurrence"] == 0
    ]
    if (
        len(shippers) != 1
        or shippers[0]["conditioningCountryCode"] != certificate.shipper_country_code
    ):
        raise ValueError("sampled shipper country violates source-certified context")
    fixed_admin1 = _admin1_constraint(party_certificate)
    if fixed_admin1 is None:
        return
    if admin1 is None:
        raise ValueError("source-fixed shipper region has no pinned registry")
    locality = shippers[0]["locality"]
    if locality is None or locality["countryCode"] != fixed_admin1[0]:
        raise ValueError("sampled shipper locality violates immutable source region")
    actual, _ = admin1.region_for_locality(locality)
    if actual != fixed_admin1[1]:
        raise ValueError("sampled shipper locality is outside immutable source admin1")
