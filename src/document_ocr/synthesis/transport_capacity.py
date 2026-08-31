"""Type-aware transport-capacity gates for structured B/L synthesis.

The task labels preserve carrier-written equipment descriptions rather than a
normalized equipment code.  Free-text shorthands are therefore never mapped to
equipment families here.  Classification is restricted to an exact four-byte
ISO 6346 size/type code present in ``typeCode`` or, when that field is absent,
an exact code-only ``typeDescription``.  An unrecognized container remains
explicit and receives the configured absolute equipment ceiling; it is never
silently treated as a standard dry box.

The default values are configured by the run.  The reference configuration is
based on Maersk's published dry-equipment cargo limits (20 standard, 40
standard, 40 high cube, and 45 high cube) and its published 47,300 kg flat-rack
maximum.  They are upper-bound safety gates, not loading recommendations.
"""

from __future__ import annotations

import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from decimal import Decimal
from typing import Any, Literal, Protocol, cast

EquipmentFamily = Literal[
    "twenty_standard",
    "forty_standard",
    "forty_high_cube",
    "forty_five_high_cube",
    "out_of_gauge",
    "unclassified",
]

_MASS_TO_KG = {
    "kilogram": Decimal("1"),
    "metric_tonne": Decimal("1000"),
    "pound": Decimal("0.45359237"),
}
_VOLUME_TO_M3 = {"cubic_metre": Decimal("1")}


@dataclass(frozen=True, slots=True)
class EquipmentCapacity:
    """One classified container's conservative payload and volume ceilings."""

    family: EquipmentFamily
    payload_kg: Decimal
    volume_m3: Decimal | None


@dataclass(frozen=True, slots=True)
class TransportCapacityLimits:
    """Versioned numeric limits supplied by strict run configuration."""

    policy: Literal["source_type_aware_maersk_upper_bounds_v1"]
    published_reference_margin_fraction: Decimal
    twenty_standard_payload_kg: Decimal
    twenty_standard_volume_m3: Decimal
    forty_standard_payload_kg: Decimal
    forty_standard_volume_m3: Decimal
    forty_high_cube_payload_kg: Decimal
    forty_high_cube_volume_m3: Decimal
    forty_five_high_cube_payload_kg: Decimal
    forty_five_high_cube_volume_m3: Decimal
    out_of_gauge_payload_kg: Decimal
    unclassified_payload_kg: Decimal
    unclassified_volume_m3: Decimal

    def __post_init__(self) -> None:
        values = (
            self.twenty_standard_payload_kg,
            self.twenty_standard_volume_m3,
            self.forty_standard_payload_kg,
            self.forty_standard_volume_m3,
            self.forty_high_cube_payload_kg,
            self.forty_high_cube_volume_m3,
            self.forty_five_high_cube_payload_kg,
            self.forty_five_high_cube_volume_m3,
            self.out_of_gauge_payload_kg,
            self.unclassified_payload_kg,
            self.unclassified_volume_m3,
        )
        if any(not value.is_finite() or value <= 0 for value in values):
            raise ValueError("transport-capacity limits must be finite and positive")
        if not self.published_reference_margin_fraction.is_finite() or not Decimal(
            0
        ) <= self.published_reference_margin_fraction <= Decimal("0.10"):
            raise ValueError("published reference margin must be between zero and 0.10")


class TransportCapacitySettings(Protocol):
    policy: Literal["source_type_aware_maersk_upper_bounds_v1"]
    published_reference_margin_fraction: float
    twenty_standard_payload_kg: float
    twenty_standard_volume_m3: float
    forty_standard_payload_kg: float
    forty_standard_volume_m3: float
    forty_high_cube_payload_kg: float
    forty_high_cube_volume_m3: float
    forty_five_high_cube_payload_kg: float
    forty_five_high_cube_volume_m3: float
    out_of_gauge_payload_kg: float
    unclassified_payload_kg: float
    unclassified_volume_m3: float


def capacity_limits(settings: TransportCapacitySettings) -> TransportCapacityLimits:
    """Freeze validated configuration floats into exact decimal arithmetic."""

    return TransportCapacityLimits(
        policy=settings.policy,
        published_reference_margin_fraction=Decimal(
            str(settings.published_reference_margin_fraction)
        ),
        twenty_standard_payload_kg=Decimal(str(settings.twenty_standard_payload_kg)),
        twenty_standard_volume_m3=Decimal(str(settings.twenty_standard_volume_m3)),
        forty_standard_payload_kg=Decimal(str(settings.forty_standard_payload_kg)),
        forty_standard_volume_m3=Decimal(str(settings.forty_standard_volume_m3)),
        forty_high_cube_payload_kg=Decimal(str(settings.forty_high_cube_payload_kg)),
        forty_high_cube_volume_m3=Decimal(str(settings.forty_high_cube_volume_m3)),
        forty_five_high_cube_payload_kg=Decimal(str(settings.forty_five_high_cube_payload_kg)),
        forty_five_high_cube_volume_m3=Decimal(str(settings.forty_five_high_cube_volume_m3)),
        out_of_gauge_payload_kg=Decimal(str(settings.out_of_gauge_payload_kg)),
        unclassified_payload_kg=Decimal(str(settings.unclassified_payload_kg)),
        unclassified_volume_m3=Decimal(str(settings.unclassified_volume_m3)),
    )


@dataclass(frozen=True, slots=True)
class TransportCapacityReceipt:
    """Complete, JSON-safe validation result for one task target."""

    policy: str
    container_count: int
    equipment_families: tuple[str, ...]
    payload_capacity_kg: Decimal | None
    volume_capacity_m3: Decimal | None
    gross_weight_kg: Decimal | None
    net_weight_kg: Decimal | None
    volume_m3: Decimal | None
    gross_payload_utilization: Decimal | None
    net_payload_utilization: Decimal | None
    volume_utilization: Decimal | None
    violations: tuple[str, ...]

    @property
    def valid(self) -> bool:
        return not self.violations

    def to_dict(self) -> dict[str, Any]:
        def number(value: Decimal | None) -> float | None:
            return float(value) if value is not None else None

        return {
            "policy": self.policy,
            "container_count": self.container_count,
            "equipment_families": list(self.equipment_families),
            "payload_capacity_kg": number(self.payload_capacity_kg),
            "volume_capacity_m3": number(self.volume_capacity_m3),
            "gross_weight_kg": number(self.gross_weight_kg),
            "net_weight_kg": number(self.net_weight_kg),
            "volume_m3": number(self.volume_m3),
            "gross_payload_utilization": number(self.gross_payload_utilization),
            "net_payload_utilization": number(self.net_payload_utilization),
            "volume_utilization": number(self.volume_utilization),
            "violations": list(self.violations),
            "valid": self.valid,
        }


# Conservative subset of ISO 6346:2022 detailed type-code characters published
# by BIC. Group codes (for example GP) are intentionally left unclassified;
# accepting fewer codes only loosens this safety gate, whereas accepting an
# invalid shorthand could apply the wrong equipment ceiling.
_ISO_DETAILED_SIZE_TYPE_CODE = re.compile(r"^[A-Z0-9]{2}[GVBSRHUPKNA][0-9ABDGJMVWXY]$")


def _exact_iso_size_type_code(container: Mapping[str, Any]) -> str | None:
    type_code = container.get("typeCode")
    if type_code is not None:
        if not isinstance(type_code, str):
            return None
        candidate = type_code.strip().upper()
        return candidate if _ISO_DETAILED_SIZE_TYPE_CODE.fullmatch(candidate) else None
    description = container.get("typeDescription")
    if not isinstance(description, str):
        return None
    candidate = description.strip().upper()
    return candidate if _ISO_DETAILED_SIZE_TYPE_CODE.fullmatch(candidate) else None


def classify_equipment(container: Mapping[str, Any]) -> EquipmentFamily:
    """Classify only exact ISO 6346 size/type codes; free text stays unclassified."""

    code = _exact_iso_size_type_code(container)
    if code is None:
        return "unclassified"
    length_code, height_code, type_code = code[0], code[1], code[2]
    if type_code in {"P", "U"}:
        return "out_of_gauge"
    if length_code == "L" and height_code == "5":
        return "forty_five_high_cube"
    if length_code == "2" and height_code in {"0", "2"}:
        return "twenty_standard"
    if length_code == "4" and height_code == "5":
        return "forty_high_cube"
    if length_code == "4" and height_code in {"0", "2"}:
        return "forty_standard"
    return "unclassified"


def equipment_capacity(
    container: Mapping[str, Any], limits: TransportCapacityLimits
) -> EquipmentCapacity:
    family = classify_equipment(container)
    values: dict[EquipmentFamily, tuple[Decimal, Decimal | None]] = {
        "twenty_standard": (
            limits.twenty_standard_payload_kg,
            limits.twenty_standard_volume_m3,
        ),
        "forty_standard": (
            limits.forty_standard_payload_kg,
            limits.forty_standard_volume_m3,
        ),
        "forty_high_cube": (
            limits.forty_high_cube_payload_kg,
            limits.forty_high_cube_volume_m3,
        ),
        "forty_five_high_cube": (
            limits.forty_five_high_cube_payload_kg,
            limits.forty_five_high_cube_volume_m3,
        ),
        # Out-of-gauge equipment has no meaningful enclosed-volume ceiling.
        "out_of_gauge": (limits.out_of_gauge_payload_kg, None),
        "unclassified": (
            limits.unclassified_payload_kg,
            limits.unclassified_volume_m3,
        ),
    }
    payload, volume = values[family]
    if family in {
        "twenty_standard",
        "forty_standard",
        "forty_high_cube",
        "forty_five_high_cube",
    }:
        multiplier = Decimal(1) + limits.published_reference_margin_fraction
        payload *= multiplier
        volume = cast(Decimal, volume) * multiplier
    return EquipmentCapacity(family=family, payload_kg=payload, volume_m3=volume)


def _canonical_measure(
    measure: Mapping[str, Any] | None,
    *,
    factors: Mapping[str, Decimal],
    kind: str,
) -> Decimal | None:
    if measure is None:
        return None
    value = measure.get("value")
    unit = measure.get("unit")
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not isinstance(unit, str):
        raise ValueError(f"{kind} measure is not a numeric value/unit pair")
    try:
        factor = factors[unit]
    except KeyError as error:
        raise ValueError(f"unsupported {kind} unit: {unit!r}") from error
    canonical = Decimal(str(value)) * factor
    if not canonical.is_finite() or canonical <= 0:
        raise ValueError(f"{kind} measure must be finite and positive")
    return canonical


def group_measure_totals(target: Mapping[str, Any]) -> dict[str, dict[str, Decimal | None]]:
    """Return canonical mass/volume totals by cargo-group identity."""

    patch = cast(Mapping[str, Any], target["documentPatch"])
    output: dict[str, dict[str, Decimal | None]] = {}
    for group in cast(Sequence[Mapping[str, Any]], patch.get("cargoGroups") or ()):
        group_id = cast(str, group["groupId"])
        if group_id in output:
            raise ValueError(f"duplicate cargo group: {group_id}")
        output[group_id] = {
            "gross": _canonical_measure(
                cast(Mapping[str, Any] | None, group.get("grossWeight")),
                factors=_MASS_TO_KG,
                kind="mass",
            ),
            "net": _canonical_measure(
                cast(Mapping[str, Any] | None, group.get("netWeight")),
                factors=_MASS_TO_KG,
                kind="mass",
            ),
            "volume": _canonical_measure(
                cast(Mapping[str, Any] | None, group.get("volume")),
                factors=_VOLUME_TO_M3,
                kind="volume",
            ),
        }
    return output


def _sum_present(values: Sequence[Decimal | None]) -> Decimal | None:
    present = [value for value in values if value is not None]
    return sum(present, start=Decimal(0)) if present else None


def document_capacity_receipt(
    target: Mapping[str, Any], limits: TransportCapacityLimits
) -> TransportCapacityReceipt:
    """Validate aggregate cargo measures against the source equipment topology."""

    patch = cast(Mapping[str, Any], target["documentPatch"])
    containers = cast(Sequence[Mapping[str, Any]], patch.get("containers") or ())
    capacities = tuple(equipment_capacity(row, limits) for row in containers)
    payload_capacity = (
        sum((row.payload_kg for row in capacities), start=Decimal(0)) if capacities else None
    )
    volume_capacity = (
        sum((cast(Decimal, row.volume_m3) for row in capacities), start=Decimal(0))
        if capacities and all(row.volume_m3 is not None for row in capacities)
        else None
    )
    group_totals = group_measure_totals(target)
    gross = _sum_present([row["gross"] for row in group_totals.values()])
    net = _sum_present([row["net"] for row in group_totals.values()])
    volume = _sum_present([row["volume"] for row in group_totals.values()])
    violations: list[str] = []
    if gross is not None and net is not None and gross < net:
        violations.append("document_gross_weight_below_net_weight")
    if payload_capacity is not None:
        if gross is not None and gross > payload_capacity:
            violations.append("document_gross_weight_exceeds_container_payload")
        if net is not None and net > payload_capacity:
            violations.append("document_net_weight_exceeds_container_payload")
    if volume_capacity is not None and volume is not None and volume > volume_capacity:
        violations.append("document_volume_exceeds_container_capacity")

    def utilization(value: Decimal | None, capacity: Decimal | None) -> Decimal | None:
        return value / capacity if value is not None and capacity is not None else None

    return TransportCapacityReceipt(
        policy=limits.policy,
        container_count=len(containers),
        equipment_families=tuple(row.family for row in capacities),
        payload_capacity_kg=payload_capacity,
        volume_capacity_m3=volume_capacity,
        gross_weight_kg=gross,
        net_weight_kg=net,
        volume_m3=volume,
        gross_payload_utilization=utilization(gross, payload_capacity),
        net_payload_utilization=utilization(net, payload_capacity),
        volume_utilization=utilization(volume, volume_capacity),
        violations=tuple(violations),
    )


def numeric_fit_envelope_violations(
    target: Mapping[str, Any], limits: TransportCapacityLimits
) -> tuple[str, ...]:
    """Return only severe absolute-capacity defects that may contaminate numeric fitting.

    Fleet-specific payload variation and printed equipment aliases make the
    type-aware policy deliberately stricter than the model-fit quarantine.
    Numeric fitting therefore uses the configured maximum known equipment
    envelope.  Generated targets and selectable templates still have to pass
    the stricter type-aware receipt.
    """

    patch = cast(Mapping[str, Any], target["documentPatch"])
    containers = cast(Sequence[Mapping[str, Any]], patch.get("containers") or ())
    if not containers:
        return ()
    totals = group_measure_totals(target)
    gross = _sum_present([row["gross"] for row in totals.values()])
    net = _sum_present([row["net"] for row in totals.values()])
    volume = _sum_present([row["volume"] for row in totals.values()])
    payload_cap = limits.unclassified_payload_kg * len(containers)
    has_out_of_gauge = any(classify_equipment(row) == "out_of_gauge" for row in containers)
    volume_cap = None if has_out_of_gauge else limits.unclassified_volume_m3 * len(containers)
    violations: list[str] = []
    if gross is not None and gross > payload_cap:
        violations.append("gross_weight_exceeds_absolute_equipment_envelope")
    if net is not None and net > payload_cap:
        violations.append("net_weight_exceeds_absolute_equipment_envelope")
    if volume is not None and volume_cap is not None and volume > volume_cap:
        violations.append("volume_exceeds_absolute_equipment_envelope")
    return tuple(violations)


def _allocation_container_numbers(target: Mapping[str, Any], group_id: str) -> tuple[str, ...]:
    patch = cast(Mapping[str, Any], target["documentPatch"])
    values = {
        cast(str, allocation["containerNumber"])
        for row in cast(Sequence[Mapping[str, Any]], patch.get("cargoAllocationGroups") or ())
        if row["groupId"] == group_id
        for allocation in cast(Sequence[Mapping[str, Any]], row.get("allocations") or ())
        if isinstance(allocation.get("containerNumber"), str)
    }
    return tuple(sorted(values))


def group_capacity_budgets(
    target: Mapping[str, Any], limits: TransportCapacityLimits
) -> dict[str, dict[str, Decimal | None]]:
    """Allocate document capacity by source measure share and explicit container links.

    Source-share budgets add exactly to the document capacity, which prevents
    independently sampled cargo groups from jointly overloading the shipment.
    An explicit group-to-container allocation can only tighten that budget.
    """

    patch = cast(Mapping[str, Any], target["documentPatch"])
    containers = cast(Sequence[Mapping[str, Any]], patch.get("containers") or ())
    by_number = {
        cast(str, row["containerNumber"]): equipment_capacity(row, limits) for row in containers
    }
    receipt = document_capacity_receipt(target, limits)
    measures = group_measure_totals(target)
    document_totals = {
        name: _sum_present([row[name] for row in measures.values()])
        for name in ("gross", "net", "volume")
    }
    document_caps = {
        "gross": receipt.payload_capacity_kg,
        "net": receipt.payload_capacity_kg,
        "volume": receipt.volume_capacity_m3,
    }
    output: dict[str, dict[str, Decimal | None]] = {}
    for group_id, group_measures in measures.items():
        linked = _allocation_container_numbers(target, group_id)
        unknown = sorted(set(linked) - set(by_number))
        if unknown:
            raise ValueError(
                f"cargo allocation references unknown containers for {group_id}: {unknown}"
            )
        linked_caps = [by_number[value] for value in linked]
        group_budgets: dict[str, Decimal | None] = {}
        for name in ("gross", "net", "volume"):
            measure = group_measures[name]
            total = document_totals[name]
            capacity = document_caps[name]
            share_budget = (
                capacity * measure / total
                if capacity is not None and measure is not None and total is not None
                else None
            )
            if not linked_caps:
                allocation_budget = None
            elif name == "volume":
                allocation_budget = (
                    sum(
                        (cast(Decimal, row.volume_m3) for row in linked_caps),
                        start=Decimal(0),
                    )
                    if all(row.volume_m3 is not None for row in linked_caps)
                    else None
                )
            else:
                allocation_budget = sum((row.payload_kg for row in linked_caps), start=Decimal(0))
            present = [value for value in (share_budget, allocation_budget) if value is not None]
            group_budgets[name] = min(present) if present else None
        output[group_id] = group_budgets
    return output
