"""Commodity-specific non-bulk packaging support for synthetic DG scenarios.

This is a semantic synthesis envelope, not a shipment-compliance certificate.
49 CFR 173.201-203 and 173.211-213 distinguish ordinary liquid/solid
packaging from special articles, gases and explosives. A broad hazard category
alone cannot make that distinction. Material specifications, special provisions
and operational approvals are deliberately not invented in the training text.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from decimal import ROUND_CEILING, Decimal
from typing import Any, Literal

from document_ocr.synthesis.dangerous_goods_registry import (
    DangerousGoodsHmtRecord,
    LoadedDangerousGoodsRegistry,
)
from document_ocr.synthesis.generators import DeterministicStream


@dataclass(frozen=True)
class PackagingProfile:
    section: str
    state: Literal["liquid", "solid"]
    packing_group: str


# Official section scopes: https://www.phmsa.dot.gov/regulations/title49/part/173
_PROFILES = {
    "201": PackagingProfile("173.201", "liquid", "I"),
    "202": PackagingProfile("173.202", "liquid", "II"),
    "203": PackagingProfile("173.203", "liquid", "III"),
    "211": PackagingProfile("173.211", "solid", "I"),
    "212": PackagingProfile("173.212", "solid", "II"),
    "213": PackagingProfile("173.213", "solid", "III"),
}


def profile(record: DangerousGoodsHmtRecord) -> PackagingProfile:
    value = _PROFILES.get(record.source_fields.get("Non-bulk Packaging", ""))
    if (
        value is None
        or record.packing_group_code != value.packing_group
        or record.exact_hazard_class[0] in "127"
        or not record.maritime_eligible
        or record.nos_entry
        or record.technical_name_required
    ):
        raise ValueError("DG identity needs a specialized packaging/technical-name contract")
    return value


def supported_registry(registry: LoadedDangerousGoodsRegistry) -> LoadedDangerousGoodsRegistry:
    records = []
    for record in registry.hmt_records:
        try:
            profile(record)
        except ValueError:
            continue  # Explicit sampling envelope, never a substituted source identity.
        records.append(record)
    if not records:
        raise ValueError("DG registry has no supported non-bulk chemical identities")
    ids = {r.record_id for r in records}
    links = tuple(
        link
        for link in registry.ecics_links
        if link.disposition == "eligible_unique_maritime_hmt" and link.hmt_record_id in ids
    )
    return LoadedDangerousGoodsRegistry(
        tuple(records), links, registry.hmt_sha256, registry.ecics_sha256
    )


def sample_packages(
    *,
    records: Sequence[DangerousGoodsHmtRecord],
    group: Mapping[str, Any],
    packages: Sequence[Mapping[str, Any]],
    stream: DeterministicStream,
) -> tuple[tuple[str, ...], dict[str, Any]]:
    profiles = tuple(profile(r) for r in records)
    if not profiles:
        raise ValueError("DG packaging requires an exact commodity identity")
    if len(records) != 1:
        raise ValueError("multiple DG identities need explicit separate-package allocation")
    if not packages:
        return (), dict(basis="hmt_non_bulk_chemical_v1", sections=[p.section for p in profiles])
    if any("quantity" not in p for p in packages):
        raise ValueError("DG packaging requires explicit package quantities")
    count = sum(Decimal(str(p["quantity"])) for p in packages)
    if count <= 0:
        raise ValueError("DG package count must be positive")
    # Conservative synthetic envelope for BOTH mass and volume. 400 kg / 450 L
    # are the solid non-bulk upper bounds in 49 CFR 171.8, not typical weights.
    # Gross weight/cargo cube overestimate content; do not infer a chemical density.
    required_units = 1
    if "grossWeight" in group or "netWeight" in group:
        mass = group.get("grossWeight", group.get("netWeight"))
        assert mass is not None
        factor = {"kilogram": "1", "metric_tonne": "1000", "pound": ".45359237"}[mass["unit"]]
        required_units = max(
            required_units,
            int(
                (Decimal(str(mass["value"])) * Decimal(factor) / count / 400).to_integral_value(
                    rounding=ROUND_CEILING
                )
            ),
        )
    if "volume" in group:
        volume = group["volume"]
        factor = {"cubic_metre": "1"}[volume["unit"]]
        required_units = max(
            required_units,
            int(
                (
                    Decimal(str(volume["value"])) * Decimal(factor) / count / Decimal(".45")
                ).to_integral_value(rounding=ROUND_CEILING)
            ),
        )
    # The source quantity counts packages, not an inferred number of overpacks.
    # Converting hundreds of drums into hundreds of pallets would preserve mass
    # arithmetic while inventing an unsupported, often impossible topology.
    pallet_topology = all(p["typeCategory"] == "PACKAGE_PALLET" for p in packages)
    if required_units > 1 and not pallet_topology:
        raise ValueError("DG inner-package mass requires an explicit source overpack topology")
    categories = (
        ("PACKAGE_PALLET",)
        if pallet_topology
        else ("PACKAGE_DRUM", "PACKAGE_BOX", "PACKAGE_CARTON")
    )
    signature = tuple(
        categories[stream.derive(str(i)).randbelow(len(categories))] for i in range(len(packages))
    )
    return signature, dict(
        basis="hmt_non_bulk_chemical_v1",
        sections=[p.section for p in profiles],
        state=profiles[0].state,
        packagingScope=(
            "generic outer packaging with compatible inner receptacles; pallets are overpacks"
        ),
        minimumInnerUnitsPerOverpack=required_units,
        complianceScope="semantic synthesis only; no material-specific transport approval asserted",
    )
