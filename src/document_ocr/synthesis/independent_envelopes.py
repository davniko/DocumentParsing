"""Source-independent physical envelopes for the reduced package vocabulary.

The bounds in this module are synthetic-data planning limits, not packing
approvals and not facts recovered from a source document.  Every lookup is
closed over the reviewed package vocabulary.  Context-sensitive counting units
must be supplied explicitly; package tokens never fill in missing authority.
"""

from __future__ import annotations

import copy
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from decimal import Decimal
from functools import lru_cache
from types import MappingProxyType
from typing import Any

from document_ocr.synthesis.package_aliases import PACKAGE_ALIASES
from document_ocr.synthesis.tank_equipment import TANK_COUNTED_PACKAGE_TYPES

DEFAULT_ROW_PAYLOAD_KG = Decimal("16000")
DEFAULT_ROW_VOLUME_M3 = Decimal("24")

PACKAGE_CATEGORY_FAMILIES: Mapping[str, str] = MappingProxyType(
    {
        "PACKAGE_BAG": "BAG",
        "PACKAGE_BAG_JUMBO": "BAG",
        "PACKAGE_BAG_PAPER": "BAG",
        "PACKAGE_BAG_PAPER_MULTI_WALL": "BAG",
        "PACKAGE_BAG_PLASTIC": "BAG",
        "PACKAGE_BAG_SUPER_BULK": "BAG",
        "PACKAGE_BALE": "BALE",
        "PACKAGE_BALE_NON_COMPRESSED": "BALE",
        "PACKAGE_BIN": "BIN",
        "PACKAGE_BLOCK": "BLOCK",
        "PACKAGE_BOX": "BOX",
        "PACKAGE_BOX_FIBREBOARD": "BOX",
        "PACKAGE_BOX_WOODEN_NATURAL_WOOD_ORDINARY": "BOX",
        "PACKAGE_BULK_LIQUID": "BULK_LIQUID",
        "PACKAGE_BULK_SOLID_FINE_PARTICLES_POWDERS": "BULK_POWDER",
        "PACKAGE_BULK_SOLID_GRANULAR_PARTICLES_GRAINS": "BULK_GRAIN",
        "PACKAGE_BUNDLE": "BUNDLE",
        "PACKAGE_CARTON": "CARTON",
        "PACKAGE_CASE": "CASE",
        "PACKAGE_CASE_WOODEN": "CASE",
        "PACKAGE_COIL": "COIL",
        "PACKAGE_CRATE": "CRATE",
        "PACKAGE_CRATE_WOODEN": "CRATE",
        "PACKAGE_DRUM": "DRUM",
        "PACKAGE_DRUM_FIBRE": "DRUM",
        "PACKAGE_DRUM_PLASTIC": "DRUM",
        "PACKAGE_DRUM_STEEL": "DRUM",
        "PACKAGE_FLEXIBAG": "FLEXIBAG",
        "PACKAGE_FLEXITANK": "FLEXITANK",
        "PACKAGE_INTERMEDIATE_BULK_CONTAINER": "IBC",
        "PACKAGE_LOT": "SET",
        "PACKAGE_OCTABIN": "OCTABIN",
        "PACKAGE_PACKAGE": "PACKAGE",
        "PACKAGE_PALLET": "PALLET",
        "PACKAGE_PALLET_WOODEN": "PALLET",
        "PACKAGE_PARCEL": "PARCEL",
        "PACKAGE_PIECE": "SET",
        "PACKAGE_RACK": "RACK",
        "PACKAGE_REEL": "REEL",
        "PACKAGE_ROLL": "ROLL",
        "PACKAGE_SACK": "BAG",
        "PACKAGE_SET": "SET",
        "PACKAGE_SKID": "SKID",
        "PACKAGE_SLAB": "SLAB",
        "PACKAGE_UNIT": "SET",
        "PACKAGE_UNPACKED_OR_UNPACKAGED": "UNPACKED",
        "PACKAGE_UNPACKED_OR_UNPACKAGED_SINGLE_UNIT": "UNPACKED_SINGLE_UNIT",
        "PACKAGE_VEHICLE": "VEHICLE",
    }
)

PACKAGE_COUNTING_CONTEXTS: Mapping[str, frozenset[str]] = MappingProxyType(
    {
        **{category: frozenset() for category in PACKAGE_CATEGORY_FAMILIES},
        "PACKAGE_BULK_LIQUID": frozenset({"bulk_loading_lot"}),
        "PACKAGE_BULK_SOLID_FINE_PARTICLES_POWDERS": frozenset({"bulk_loading_lot"}),
        "PACKAGE_BULK_SOLID_GRANULAR_PARTICLES_GRAINS": frozenset({"bulk_loading_lot"}),
        "PACKAGE_FLEXIBAG": frozenset({"non_dg_liquid_system"}),
        "PACKAGE_FLEXITANK": frozenset({"non_dg_liquid_system"}),
        "PACKAGE_VEHICLE": frozenset({"vehicle"}),
        "PACKAGE_UNPACKED_OR_UNPACKAGED": frozenset({"vehicle", "discrete_article"}),
        "PACKAGE_UNPACKED_OR_UNPACKAGED_SINGLE_UNIT": frozenset({"vehicle", "discrete_article"}),
    }
)

# family -> (minimum net kg, maximum net kg, tare kg, occupied cubic metres,
# authority kind).  These conservative scales are intentionally disclosed
# synthetic planning bounds.  They do not encode commodity compatibility.
_FAMILY_ENVELOPES: Mapping[str, tuple[int, int, int, str, str]] = MappingProxyType(
    {
        "PACKAGE": (10, 30, 1, "0.08", "established_independent_planning_scale"),
        "CARTON": (8, 24, 1, "0.08", "established_independent_planning_scale"),
        "BOX": (12, 40, 2, "0.10", "established_independent_planning_scale"),
        "CRATE": (60, 180, 15, "0.45", "established_independent_planning_scale"),
        "CASE": (60, 180, 10, "0.45", "established_independent_planning_scale"),
        "PALLET": (200, 650, 25, "1.10", "established_independent_planning_scale"),
        "BAG": (20, 50, 1, "0.08", "established_independent_planning_scale"),
        "BALE": (100, 220, 5, "0.60", "established_independent_planning_scale"),
        "DRUM": (120, 200, 12, "0.28", "established_independent_planning_scale"),
        "IBC": (600, 900, 50, "1.20", "established_independent_planning_scale"),
        "BUNDLE": (40, 120, 2, "0.35", "established_independent_planning_scale"),
        "ROLL": (40, 100, 3, "0.35", "established_independent_planning_scale"),
        "REEL": (80, 180, 8, "0.45", "established_independent_planning_scale"),
        "SET": (12, 40, 3, "0.20", "established_independent_planning_scale"),
        "BIN": (80, 240, 10, "0.60", "disclosed_flat48_probe_scale"),
        "BLOCK": (80, 240, 0, "0.40", "disclosed_flat48_probe_scale"),
        "BULK_LIQUID": (8000, 12000, 0, "16.0", "disclosed_flat48_probe_scale"),
        "BULK_POWDER": (8000, 12000, 0, "16.0", "disclosed_flat48_probe_scale"),
        "BULK_GRAIN": (8000, 12000, 0, "16.0", "disclosed_flat48_probe_scale"),
        "COIL": (100, 350, 5, "0.50", "disclosed_flat48_probe_scale"),
        "FLEXIBAG": (11000, 15000, 80, "20.0", "bounded_non_dg_liquid_system"),
        "FLEXITANK": (11000, 15000, 80, "20.0", "bounded_non_dg_liquid_system"),
        "OCTABIN": (400, 700, 20, "1.30", "disclosed_flat48_probe_scale"),
        "PARCEL": (3, 20, 1, "0.05", "disclosed_flat48_probe_scale"),
        "RACK": (150, 500, 30, "1.50", "disclosed_flat48_probe_scale"),
        "SKID": (200, 650, 25, "1.10", "disclosed_flat48_probe_scale"),
        "SLAB": (80, 240, 0, "0.40", "disclosed_flat48_probe_scale"),
    }
)

_COUNTED_UNIT_ENVELOPES: Mapping[str, tuple[int, int, int, str, str]] = MappingProxyType(
    {
        "vehicle": (500, 1800, 0, "12.0", "bounded_vehicle_unit_domain"),
        "discrete_article": (100, 600, 0, "1.50", "bounded_discrete_article_domain"),
        "tank_loading_lot": (8000, 15000, 0, "20.0", "bounded_installed_tank_contents_lot"),
    }
)


@dataclass(frozen=True, slots=True)
class PackageEnvelope:
    """A source-independent physical planning contract for one counted unit."""

    category: str
    family: str
    counted_unit_context: str
    net_kg: tuple[int, int]
    tare_kg: int
    cube_m3: Decimal
    authority: str
    cube_range_m3: tuple[Decimal, Decimal] | None = None

    def catalog_envelope(self) -> dict[str, Any]:
        return {
            "legacyCategory": self.category,
            "netKg": list(self.net_kg),
            "tareKg": self.tare_kg,
            "cubeM3": str(self.cube_m3),
        }

    def receipt(self) -> dict[str, Any]:
        result = {
            "category": self.category,
            "family": self.family,
            "countedUnitContext": self.counted_unit_context,
            "envelope": self.catalog_envelope(),
            "authority": self.authority,
            "sourceCargoAccess": False,
            "physicalApproval": False,
            "publicCountOrAllocationInferred": False,
        }
        if self.cube_range_m3 is not None:
            result["volumeRangeM3"] = [str(value) for value in self.cube_range_m3]
        return result


def package_categories() -> tuple[str, ...]:
    """Return the exact closed category inventory in deterministic order."""

    return tuple(sorted(PACKAGE_CATEGORY_FAMILIES))


def package_family(category: str) -> str:
    try:
        return PACKAGE_CATEGORY_FAMILIES[category]
    except (KeyError, TypeError) as exc:
        raise ValueError(f"outside the reviewed flat48 numeric scope: {category!r}") from exc


@lru_cache(maxsize=64)
def package_envelope(category: str, *, counted_unit_context: str | None = None) -> PackageEnvelope:
    """Resolve one envelope without inferring an ambiguous counting unit."""

    family = package_family(category)
    allowed_contexts = PACKAGE_COUNTING_CONTEXTS[category]
    if allowed_contexts and counted_unit_context not in allowed_contexts:
        raise ValueError(
            f"{category} requires counted_unit_context from {sorted(allowed_contexts, key=str)}; "
            f"received {counted_unit_context!r}"
        )
    tank_lot = category in TANK_COUNTED_PACKAGE_TYPES and counted_unit_context == "tank_loading_lot"
    if not allowed_contexts and counted_unit_context is not None and not tank_lot:
        raise ValueError(f"ordinary package {category} does not accept a counting context")
    context = counted_unit_context or "declared_package"
    values = (
        _COUNTED_UNIT_ENVELOPES[context]
        if context in _COUNTED_UNIT_ENVELOPES
        else _FAMILY_ENVELOPES[family]
    )
    low, high, tare, cube_text, authority = values
    cube = Decimal(cube_text)
    if (
        low <= 0
        or high < low
        or tare < 0
        or not cube.is_finite()
        or cube <= 0
        or Decimal(high + tare) > DEFAULT_ROW_PAYLOAD_KG
        or cube > DEFAULT_ROW_VOLUME_M3
    ):
        raise ValueError(f"invalid independent numeric envelope for {category}")
    return PackageEnvelope(
        category,
        family,
        context,
        (low, high),
        tare,
        cube,
        authority,
        (Decimal("14"), Decimal("20")) if context == "tank_loading_lot" else None,
    )


def maximum_units_per_row(
    envelope: PackageEnvelope,
    *,
    unit_net_kg: int | Decimal,
    unit_cube_m3: Decimal | None = None,
    payload_kg: Decimal = DEFAULT_ROW_PAYLOAD_KG,
    volume_m3: Decimal | None = DEFAULT_ROW_VOLUME_M3,
) -> int:
    """Return a conservative whole-unit capacity for explicit row limits."""

    net = Decimal(unit_net_kg)
    payload = Decimal(payload_kg)
    if not net.is_finite() or not payload.is_finite() or net <= 0 or payload <= 0:
        raise ValueError("positive finite unit net and payload are required")
    if not Decimal(envelope.net_kg[0]) <= net <= Decimal(envelope.net_kg[1]):
        raise ValueError("unit net is outside its independent package envelope")
    by_mass = int(payload // (net + Decimal(envelope.tare_kg)))
    if volume_m3 is None:
        by_volume = by_mass
    else:
        volume = Decimal(volume_m3)
        if not volume.is_finite() or volume <= 0:
            raise ValueError("row volume must be positive and finite when supplied")
        cube = envelope.cube_m3 if unit_cube_m3 is None else Decimal(unit_cube_m3)
        low_cube, high_cube = envelope.cube_range_m3 or (envelope.cube_m3, envelope.cube_m3)
        if not cube.is_finite() or not low_cube <= cube <= high_cube:
            raise ValueError("unit cube is outside its independent package envelope")
        by_volume = int(volume // cube)
    maximum = min(by_mass, by_volume)
    if maximum < 1:
        raise ValueError("one counted unit does not fit the explicit row capacity")
    return maximum


def configure_sampler(
    sampler: Any,
    *,
    categories: Iterable[str],
    category_contexts: Mapping[str, str] | None = None,
) -> tuple[dict[str, Any], ...]:
    """Install validated envelope families on one sampler instance only."""

    selected = tuple(categories)
    if not selected or len(selected) != len(set(selected)):
        raise ValueError("configuration requires nonempty distinct selected categories")
    contexts = dict(category_contexts or {})
    if set(contexts) - set(selected):
        raise ValueError("counted-unit contexts reference unselected package categories")
    contracts = tuple(
        package_envelope(category, counted_unit_context=contexts.get(category))
        for category in selected
    )
    if not isinstance(sampler.catalog, dict) or not isinstance(
        sampler.catalog.get("packageEnvelopes"), dict
    ):
        raise ValueError("sampler has no mutable packageEnvelopes catalogue")
    catalog = copy.deepcopy(sampler.catalog)
    by_family: dict[str, PackageEnvelope] = {}
    for contract in contracts:
        previous = by_family.get(contract.family)
        if previous is not None:
            if previous.category != contract.category:
                raise ValueError("one legacy sampler family cannot encode multiple flat categories")
            if (
                previous.net_kg,
                previous.tare_kg,
                previous.cube_m3,
            ) != (contract.net_kg, contract.tare_kg, contract.cube_m3):
                raise ValueError("selected categories disagree about their shared numeric envelope")
        by_family[contract.family] = contract
        catalog["packageEnvelopes"][contract.family] = contract.catalog_envelope()
    sampler.catalog = catalog
    return tuple(contract.receipt() for contract in contracts)


if set(PACKAGE_CATEGORY_FAMILIES) != set(PACKAGE_ALIASES):
    raise RuntimeError("independent envelope inventory differs from package alias vocabulary")
if len(PACKAGE_CATEGORY_FAMILIES) != 48:
    raise RuntimeError("independent envelope inventory must contain exactly 48 categories")
