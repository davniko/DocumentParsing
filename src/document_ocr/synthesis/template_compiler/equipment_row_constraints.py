"""Capacity constraints from certified cargo measurements and audited unit choices.

The container-number and numeric byte spans establish ownership; searching for
digits near a container would misread seal numbers and repeated cargo prose.
These private observations do not add fields to the extraction target.
An explicitly unit-bearing total in a single-container, single-cargo document
has the same physical owner even when OCR puts it on a separate line.
Missing units may have explicitly contracted synthetic-only SI units. These do
not assert units for the original shipment or enter OCR/extraction labels.
"""

from __future__ import annotations

import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from decimal import Decimal
from typing import Any, Literal

from document_ocr.synthesis.transport_capacity import TransportCapacityLimits, equipment_capacity

from .complete_targets import SourceTemplate
from .measurement_columns import ordered_column_unit
from .numeric_auxiliary import NumericContract, PreparedNumeric, resolve


@dataclass(frozen=True)
class RowMeasurement:
    binding_key: str
    container_indices: tuple[int, ...]
    dimension: Literal["mass", "volume"]
    unit_factor: Decimal
    unit_evidence: Literal["printed", "synthetic_private"] = "printed"


@dataclass(frozen=True)
class RowLoad:
    observation: RowMeasurement
    value: Decimal


_UNITS = {
    "KG": ("mass", Decimal(1)),
    "KGS": ("mass", Decimal(1)),
    "KGM": ("mass", Decimal(1)),
    "KILOGRAMS": ("mass", Decimal(1)),
    "MT": ("mass", Decimal(1000)),
    "LB": ("mass", Decimal("0.45359237")),
    "LBS": ("mass", Decimal("0.45359237")),
    "LBR": ("mass", Decimal("0.45359237")),
    "CBM": ("volume", Decimal(1)),
    "MTQ": ("volume", Decimal(1)),
    "M3": ("volume", Decimal(1)),
    "M³": ("volume", Decimal(1)),
    "CBF": ("volume", Decimal("0.3048") ** 3),
    "FT3": ("volume", Decimal("0.3048") ** 3),
    "FTQ": ("volume", Decimal("0.3048") ** 3),
    "CFT": ("volume", Decimal("0.3048") ** 3),
}
_UNIT_NAMES = r"KILOGRAMS|KGS?|KGM|MTQ|MT|LBS?|LBR|CBM|M3|M³|CBF|CFT|FT3|FTQ"
_UNIT = re.compile(r"^\s*(" + _UNIT_NAMES + r")\b", re.I)
_EMBEDDED_UNIT = re.compile(r"[+\-]?[\d.,\s]+(" + _UNIT_NAMES + r")\s*", re.I)
_PREFIX_UNIT = re.compile(r"\b(" + _UNIT_NAMES + r")\s*[\])]*\s*:\s*$", re.I)
_OTHER_PREFIX_UNIT = re.compile(
    r"\b(?:SQM|M2|M²|SQFT|FT2|FT²|GAL(?:LONS?)?|LIT(?:ER|RE)S?|ML|CM|MM|MET(?:ER|RE)S?)\s*[:(\[\])]*\s*$",
    re.I,
)
_NEXT_FIELD = re.compile(r"^[^\W\d][^\d:\n]*:", re.UNICODE)
_TARGET_UNITS = {
    "kilogram": ("mass", Decimal(1)),
    "metric_tonne": ("mass", Decimal(1000)),
    "pound": ("mass", Decimal("0.45359237")),
    "cubic_metre": ("volume", Decimal(1)),
}
_QUOTE_UNITS = re.compile(r"(?<!\w)(" + _UNIT_NAMES + r")(?!\w)", re.I)


def _printed_quote_unit(contract: NumericContract, source: bytes) -> tuple[str, Decimal] | None:
    quote = contract.printed_unit_quote
    if quote is None:
        return None
    if quote.encode() not in source or _QUOTE_UNITS.fullmatch(quote.strip()):
        raise ValueError("printed unit quote requires an exact contextual source occurrence")
    expected = "mass" if contract.role in {"cargo_mass", "tare"} else "volume"
    units = {_UNITS[m[1].upper()] for m in _QUOTE_UNITS.finditer(quote)}
    candidates = {(d, f) for d, f in units if d == expected}
    if len(candidates) != 1:
        raise ValueError("printed unit quote does not prove one unit for its measurement dimension")
    return candidates.pop()


def _unresolved_notation(surface: str, before: str, after: str) -> bool:
    suffix = after.strip().lstrip("|/").strip()
    return bool(
        _OTHER_PREFIX_UNIT.search(before)
        or re.fullmatch(r"[+\-]?[\d.,\s]+", surface) is None
        or (suffix and suffix[0].isalpha() and not _NEXT_FIELD.match(suffix))
    )


def _validate_target_unit(
    contract: NumericContract, target: Mapping[str, Any], dimension: str, factor: Decimal
) -> None:
    """A numeric equality alone cannot prove a kg/tonne conversion.

    For example, both 25,925 MT and a 25925 kg label contain the same digits,
    but the former is a decimal comma. Check units before accepting that source
    arithmetic or conditioning any equipment on it.
    """
    if contract.mode not in {"target_sum", "target_share", "target_average"}:
        return
    for path in contract.target_paths:
        if not re.search(r"\.(?:grossWeight|netWeight|volume)\.value$", path):
            continue  # This is not a typed measurement dependency.
        measure = resolve(target, path.removesuffix(".value"))
        source_dimension, source_factor = _TARGET_UNITS[measure["unit"]]
        if source_dimension != dimension or Decimal(contract.multiplier) * factor != source_factor:
            raise ValueError("numeric dependency conversion contradicts the printed unit")


def compile_rows(
    source: SourceTemplate,
    contracts: Mapping[str, NumericContract],
    *,
    include_tare: bool = False,
) -> tuple[RowMeasurement, ...]:
    # Tare uses the same byte/unit/owner proof, but its equipment consumer must
    # request it explicitly: empty-container mass is not a cargo capacity load.
    roles = (
        {"cargo_mass", "cargo_volume", "tare"} if include_tare else {"cargo_mass", "cargo_volume"}
    )
    physical = {k: v for k, v in contracts.items() if v.role in roles}
    if not physical:
        return ()
    raw = source.source
    patch = source.target["documentPatch"]
    singleton_cargo = (
        len(patch.get("containers", ())) == 1 and len(patch.get("cargoGroups", ())) == 1
    )
    owners: dict[tuple[int, int], list[tuple[int, int, frozenset[int]]]] = {}

    def line_span(start: int, end: int) -> tuple[int, int] | None:
        left = raw.rfind(b"\n", 0, start) + 1
        right = raw.find(b"\n", start)
        if right == -1:
            right = len(raw)
        return (left, right) if end <= right else None

    for binding in source.template.bindings:
        paths = [
            re.fullmatch(r"documentPatch\.containers\[(\d+)\]\.containerNumber", p)
            for p in binding.target_paths
        ]
        # One printed identity commonly owns both the equipment row and its
        # cargo-allocation reference. The allocation alias is not another row.
        if not any(paths) or any(not p.endswith(".containerNumber") for p in binding.target_paths):
            continue
        indices = {int(m[1]) for m in paths if m is not None}
        for slot in binding.occurrences:
            span = line_span(slot.byte_start, slot.byte_end)
            if span is not None:
                owners.setdefault(span, []).append(
                    (slot.byte_start, slot.byte_end, frozenset(indices))
                )
    result: set[RowMeasurement] = set()
    for binding in source.template.bindings:
        if binding.logical_key not in physical:
            continue
        quoted_unit = _printed_quote_unit(physical[binding.logical_key], raw)
        # A reviewed physical-row dependency can span OCR table columns or
        # lines. This is an explicit compilation assertion, never proximity
        # inference from a group name or equal numbers. Multiple row owners
        # mean equal per-row values (not an aggregate); every row must have a
        # printed occurrence, allowing complete repeated copies of the table.
        dependencies = getattr(binding, "dependency_paths", ())
        aggregate = dependencies == ("documentPatch.containers",)
        if "documentPatch.containers" in dependencies and not aggregate:
            raise ValueError("aggregate measurement cannot mix inventory and row ownership")
        if aggregate and (binding.target_paths or not patch.get("containers")):
            raise ValueError(
                "aggregate measurement requires a nonempty source-only equipment scope"
            )
        row_paths = [re.fullmatch(r"documentPatch\.containers\[(\d+)\]", p) for p in dependencies]
        explicit_indices: frozenset[int] | None = None
        if any(row_paths):
            if binding.target_paths or not all(row_paths):
                raise ValueError("physical row dependency mixes container and target ownership")
            explicit_indices = frozenset(int(m[1]) for m in row_paths if m is not None)
            if (
                len(explicit_indices) != len(dependencies)
                or max(explicit_indices) >= len(patch.get("containers", ()))
                or len(binding.occurrences) % len(explicit_indices)
            ):
                raise ValueError("physical row dependency lacks a complete printed container scope")
        for slot in binding.occurrences:
            span = line_span(slot.byte_start, slot.byte_end)
            if span is None:
                if (
                    explicit_indices is not None
                    or aggregate
                    or physical[binding.logical_key].synthetic_unit is not None
                ):
                    raise ValueError(
                        "physical row dependency needs a single-line measurement surface"
                    )
                continue  # A table-column/aggregate needs its own ownership contract.
            before = raw[span[0] : slot.byte_start].decode()
            after = raw[slot.byte_end : span[1]].decode()
            column_unit = ordered_column_unit(raw, byte_end=slot.byte_end, surface=slot.source_text)
            unit = (
                _UNIT.fullmatch(column_unit)
                if column_unit is not None
                else (
                    _EMBEDDED_UNIT.fullmatch(slot.source_text)
                    or _UNIT.match(after)
                    or _PREFIX_UNIT.search(before)
                )
            )
            contract = physical[binding.logical_key]
            private = contract.synthetic_unit
            evidence: Literal["printed", "synthetic_private"] = "printed"
            if private is not None:
                if unit is not None or _unresolved_notation(slot.source_text, before, after):
                    raise ValueError(
                        "private unit cannot replace printed or unresolved measurement notation"
                    )
                dimension, factor = _TARGET_UNITS[private]
                evidence = "synthetic_private"
            elif unit is not None:
                dimension, factor = _UNITS[unit[1].upper()]
                if quoted_unit is not None and quoted_unit != (dimension, factor):
                    raise ValueError("printed unit quote contradicts the local measurement unit")
            elif quoted_unit is not None:
                if _unresolved_notation(slot.source_text, before, after):
                    raise ValueError(
                        "printed unit quote cannot override unresolved measurement notation"
                    )
                dimension, factor = quoted_unit
            else:
                if explicit_indices is not None or aggregate:
                    raise ValueError("physical row dependency lacks a printed measurement unit")
                continue
            _validate_target_unit(contract, source.target, dimension, factor)
            expected: Literal["mass", "volume"] = (
                "mass" if contract.role in {"cargo_mass", "tare"} else "volume"
            )
            if aggregate:
                if dimension != expected:
                    raise ValueError("aggregate measurement unit contradicts its measurement role")
                if raw[slot.byte_start : slot.byte_end].decode() != slot.source_text:
                    raise ValueError("numeric cargo total does not match certified source bytes")
                result.add(
                    RowMeasurement(
                        binding.logical_key,
                        tuple(range(len(patch["containers"]))),
                        expected,
                        factor,
                        evidence,
                    )
                )
                continue
            preceding = [entry for entry in owners.get(span, ()) if entry[1] <= slot.byte_start]
            if explicit_indices is not None:
                last_end = max((entry[1] for entry in preceding), default=span[0])
                if not singleton_cargo and re.search(
                    rb"\b(?:TOTAL|SUBTOTAL|GRAND\s+TOTAL)\b", raw[last_end : slot.byte_start], re.I
                ):
                    raise ValueError("declared cargo total needs aggregate equipment scope")
                if preceding:
                    local = frozenset().union(
                        *(entry[2] for entry in preceding if entry[1] == last_end)
                    )
                    if not local <= explicit_indices:
                        raise ValueError(
                            "physical row dependency contradicts printed container identity"
                        )
                expected_dimension: Literal["mass", "volume"] = (
                    "mass"
                    if physical[binding.logical_key].role in {"cargo_mass", "tare"}
                    else "volume"
                )
                if dimension != expected_dimension:
                    raise ValueError("physical row dependency contradicts printed measurement unit")
                if raw[slot.byte_start : slot.byte_end].decode() != slot.source_text:
                    raise ValueError("numeric container row does not match certified source bytes")
                result.update(
                    RowMeasurement(binding.logical_key, (i,), expected_dimension, factor, evidence)
                    for i in explicit_indices
                )
                continue
            if not preceding:
                if not singleton_cargo:
                    if private is not None:
                        raise ValueError(
                            "private unit does not establish a physical measurement owner"
                        )
                    continue
                owner_indices = frozenset({0})
                last_end = span[0]
            else:
                last_end = max(entry[1] for entry in preceding)
                owner_indices = frozenset().union(
                    *(entry[2] for entry in preceding if entry[1] == last_end)
                )
            if len(owner_indices) != 1:
                raise ValueError("numeric container row has ambiguous equipment ownership")
            if not singleton_cargo and re.search(
                rb"\b(?:TOTAL|SUBTOTAL|GRAND\s+TOTAL)\b", raw[last_end : slot.byte_start], re.I
            ):
                raise ValueError("same-line cargo total needs aggregate equipment scope")
            if dimension != expected:
                raise ValueError("numeric container row unit contradicts its measurement role")
            if raw[slot.byte_start : slot.byte_end].decode() != slot.source_text:
                raise ValueError("numeric container row does not match certified source bytes")
            result.add(
                RowMeasurement(
                    binding.logical_key, (next(iter(owner_indices)),), expected, factor, evidence
                )
            )
    return tuple(sorted(result, key=lambda r: (r.container_indices, r.binding_key, r.unit_factor)))


def prepare_loads(
    observations: Sequence[RowMeasurement], prepared: Mapping[str, PreparedNumeric]
) -> tuple[RowLoad, ...]:
    result = []
    for observation in observations:
        value = Decimal(prepared[observation.binding_key].value) * observation.unit_factor
        if not value.is_finite() or value < 0:
            raise ValueError("container row measurement must be finite and nonnegative")
        result.append(RowLoad(observation, value))
    return tuple(result)


def fits(
    load: RowLoad, containers: Sequence[Mapping[str, Any]], limits: TransportCapacityLimits
) -> bool:
    if len(containers) != len(load.observation.container_indices):
        raise ValueError("physical measurement capacity scope differs from its equipment owners")
    total = Decimal(0)
    for container in containers:
        capacity = equipment_capacity(container, limits)
        ceiling = (
            capacity.payload_kg if load.observation.dimension == "mass" else capacity.volume_m3
        )
        # Open/out-of-gauge equipment has no enclosed volume ceiling in this policy.
        if ceiling is None:
            return True
        total += ceiling
    return load.value <= total


def condition_weights(
    loads: Sequence[RowLoad],
    groups: Sequence[tuple[int, ...]],
    domains: Sequence[Mapping[str, int]],
    limits: TransportCapacityLimits,
) -> tuple[dict[str, int], ...]:
    """Remove impossible draws without assigning aggregate cargo equally to rows.

    For an aggregate, the other groups contribute their maximum remaining
    capacity. This is a feasibility bound, not an allocation or an acceptance
    test: the actual sampled fleet is checked again before publication.
    """
    capacities = {}
    for pair in {pair for domain in domains for pair in domain}:
        size, kind = pair.split("|")
        cap = equipment_capacity(dict(sizeCategory=size, typeCategory=kind), limits)
        capacities[pair] = {"mass": cap.payload_kg, "volume": cap.volume_m3}
    owner_groups = {i: g for g, indices in enumerate(groups) for i in indices}
    if sum(map(len, groups)) != len(owner_groups) or len(groups) != len(domains):
        raise ValueError("physical equipment groups must cover disjoint owners and domains")
    contributions = []
    for load in loads:
        counts: dict[int, int] = {}
        for i in load.observation.container_indices:
            counts[owner_groups[i]] = counts.get(owner_groups[i], 0) + 1
        dimension = load.observation.dimension
        maxima = {}
        for g, count in counts.items():
            values = [capacities[p][dimension] for p in domains[g]]
            maxima[g] = None if None in values else max(v for v in values if v is not None) * count
        contributions.append((load, counts, maxima))
    result = []
    for g, domain in enumerate(domains):
        accepted = {}
        relevant = [
            (load, counts[g], [v for other, v in maxima.items() if other != g])
            for load, counts, maxima in contributions
            if g in counts
        ]
        for pair, weight in domain.items():
            for load, count, others in relevant:
                ceiling = capacities[pair][load.observation.dimension]
                if ceiling is not None and None not in others:
                    total = ceiling * count + sum(v for v in others if v is not None)
                    if load.value > total:
                        break
            else:
                accepted[pair] = weight
        if not accepted:
            raise ValueError(
                "printed container rows have no configured capacity-compatible equipment"
            )
        result.append(accepted)
    return tuple(result)


def validate(
    loads: Sequence[RowLoad],
    containers: Sequence[Mapping[str, Any]],
    limits: TransportCapacityLimits,
) -> list[dict[str, Any]]:
    receipts = []
    for load in loads:
        observation = load.observation
        if not fits(load, [containers[i] for i in observation.container_indices], limits):
            raise ValueError(
                "printed container row exceeds equipment capacity: " + observation.binding_key
            )
        receipts.append(
            dict(
                bindingKey=observation.binding_key,
                containerIndices=list(observation.container_indices),
                dimension=observation.dimension,
                value=str(load.value),
                sourceUnitFactor=str(observation.unit_factor),
                unit="kilogram" if observation.dimension == "mass" else "cubic_metre",
                unitEvidence=observation.unit_evidence,
            )
        )
    return receipts


def validate_prepared(
    receipts: Sequence[Mapping[str, Any]], prepared: Mapping[str, PreparedNumeric]
) -> None:
    """Prove the values about to render are those used for the equipment draw."""
    for receipt in receipts:
        generated = prepared[receipt["bindingKey"]]
        private_unit = generated.contract.synthetic_unit
        if private_unit is not None and (
            receipt["unitEvidence"] != "synthetic_private"
            or receipt["unit"] != private_unit
            or Decimal(receipt["sourceUnitFactor"]) != 1
        ):
            raise ValueError("private measurement unit changed after equipment sampling")
        if private_unit is None and receipt["unitEvidence"] == "synthetic_private":
            raise ValueError("private measurement unit lost its generation contract")
        value = Decimal(generated.value) * Decimal(receipt["sourceUnitFactor"])
        if value != Decimal(receipt["value"]):
            raise ValueError("printed container measurement changed after equipment sampling")
