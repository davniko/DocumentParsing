"""Source-proven numeric dependency contracts, outside the training schema.

Unowned numeric surfaces must never be randomized digit by digit. A one-time
semantic interpretation states whether a scalar is target-derived, a shipment
total scaled with the scenario, or a deliberately fixed non-identity fact.
Host validation proves source arithmetic and every subsequent realization.
"""

from __future__ import annotations

import re
from collections.abc import Mapping, Sequence
from decimal import ROUND_FLOOR, ROUND_HALF_UP, Decimal, InvalidOperation
from fractions import Fraction
from itertools import pairwise
from math import gcd, lcm
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from document_ocr.synthesis.rendering import (
    _NUMBER,
    _numeric_interpretations,
    render_number_surface,
)

from .generation_contract import leaves
from .host import _MEASUREMENT_UNIT_SURFACES
from .models import CertifiedSemanticTemplate, SemanticBinding
from .semantic_plan import disposition_by_binding


class NumericContract(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)
    mode: Literal[
        "target_sum",
        "target_share",
        "target_average",
        "target_converted",
        "target_equation_count",
        "binding_sum",
        "unit_product",
        "source_scaled",
        "source_fixed",
        "sampled_equipment_tare",
        "surface_fixed",
        "review_required",
    ]
    role: Literal[
        "cargo_mass",
        "cargo_volume",
        "cargo_quantity",
        "tare",
        "per_unit_measurement",
        "density",
        "temperature",
        "commercial",
        "operational",
        "unknown",
        "metadata",
        "dimensions",
    ]
    source_value: str = Field(min_length=1)
    target_paths: list[str]
    dependency_bindings: list[str] = Field(default_factory=list)
    multiplier: Literal["1", "1000", "0.001", "kg_to_lb", "lb_to_kg", "cbm_to_cbf", "cbf_to_cbm"]
    divisor: int = Field(ge=1)
    reason: str
    synthetic_unit: Literal["kilogram", "cubic_metre"] | None = None
    printed_unit_quote: str | None = Field(default=None, min_length=1)

    @model_validator(mode="after")
    def private_unit_is_a_synthetic_physical_choice(self) -> NumericContract:
        if self.printed_unit_quote is not None and (
            self.synthetic_unit is not None
            or self.role not in {"cargo_mass", "cargo_volume", "tare"}
        ):
            raise ValueError(
                "printed unit evidence requires cargo mass/volume or equipment tare "
                "and cannot be a private unit choice"
            )
        if self.synthetic_unit is not None and (
            self.mode not in (
                {"source_fixed", "sampled_equipment_tare"}
                if self.role == "tare"
                else {"source_scaled", "binding_sum"}
            )
            or self.target_paths
            or (self.dependency_bindings and self.mode != "binding_sum")
            or self.multiplier != "1"
            or self.divisor != 1
            or self.role not in {
                "kilogram": {"cargo_mass", "tare"},
                "cubic_metre": {"cargo_volume"},
            }[self.synthetic_unit]
        ):
            raise ValueError(
                "private unit requires a source-only physical measurement of the same dimension"
            )
        return self


class PreparedNumeric(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)
    contract: NumericContract
    value: str
    scenario_scale: str


def measurement_unit_binding(binding: SemanticBinding) -> bool:
    return bool(
        not binding.target_paths
        # Unit spellings overlap with geographic codes (LB = Lebanon, MT = Malta).
        # Surface matching must not override the binding's semantic kind.
        and binding.value_kind
        in {
            "decimal_measurement",
            "temperature",
            "other_text",
            "operational_text",
            "commercial_text",
            "cargo_text",
        }
        and binding.occurrences
        and all(
            any(
                "".join(c for c in slot.source_text.casefold() if c.isalnum()) in aliases
                for aliases in (
                    *_MEASUREMENT_UNIT_SURFACES.values(),
                    frozenset({"cbf", "cft", "ft3", "ft³", "ftq", "cubicfeet"}),
                )
            )
            for slot in binding.occurrences
        )
    )


def numeric_bindings(template: CertifiedSemanticTemplate) -> tuple[SemanticBinding, ...]:
    dispositions = disposition_by_binding(template.auxiliary_semantic_plan)
    range_members = {
        key
        for constraint in template.coherence_constraints
        if constraint.kind
        in {"inclusive_range_cardinality", "aggregate_inclusive_range_cardinality"}
        for key in constraint.member_logical_keys
    }
    return tuple(
        b
        for b in template.bindings
        if not b.target_paths
        and not measurement_unit_binding(b)
        and b.derivation is None
        and (
            (b.value_kind == "package" and b.logical_key not in range_members)
            or (
                b.value_kind in {"integer", "decimal_measurement", "temperature"}
                and all(s.render_policy == "numeric_surface" for s in b.occurrences)
            )
        )
        and any(any(c.isdigit() for c in s.source_text) for s in b.occurrences)
        and (
            b.value_kind == "package"
            or b.logical_key not in dispositions
            or dispositions[b.logical_key].disposition
            not in {"composite_number", "document_sequence", "stable_vocabulary", "entity_member"}
        )
    )


def numeric_payload(
    *, source: bytes, target: Mapping[str, Any], bindings: Sequence[SemanticBinding]
) -> dict[str, Any]:
    return {
        "sourceTarget": target,
        "numericTargetLeaves": numeric_target_leaves(target),
        "sourceText": source.decode(),
        "numericBindings": [
            {
                "key": f"numeric_{i:04d}",
                "logicalKey": b.logical_key,
                "groupKind": b.group_kind,
                "groupKey": b.group_key,
                "valueKind": b.value_kind,
                "sourceSurfaces": [s.source_text for s in b.occurrences],
                "dependencyPaths": list(b.dependency_paths),
                "dependencyBindings": list(b.dependency_bindings),
            }
            for i, b in enumerate(bindings)
        ],
    }


def numeric_target_leaves(target: Mapping[str, Any]) -> dict[str, int | float]:
    return {
        path: value
        for path, value in leaves(target).items()
        if path.startswith("documentPatch.")
        and isinstance(value, (int, float))
        and not isinstance(value, bool)
    }


def _number(value: Any) -> Decimal:
    if isinstance(value, bool):
        raise ValueError("boolean is not a numeric dependency")
    try:
        number = Decimal(str(value))
    except InvalidOperation as error:
        raise ValueError("numeric dependency is not a decimal") from error
    if not number.is_finite():
        raise ValueError("numeric dependency is not finite")
    return number


def resolve(target: Mapping[str, Any], path: str) -> Any:
    if not path.startswith("documentPatch."):
        raise ValueError("numeric dependency is outside documentPatch")
    parts = re.findall(r"([A-Za-z][A-Za-z0-9]*)|\[([0-9]+)\]", path)
    reconstructed = ""
    current: Any = target
    for name, index in parts:
        reconstructed += ("." if reconstructed else "") + name if name else f"[{index}]"
        current = current[name] if name else current[int(index)]
    if reconstructed != path:
        raise ValueError("invalid numeric dependency path")
    return current


def quantum(binding: SemanticBinding, old: Decimal) -> Decimal:
    return max(surface_quantum(slot.source_text, old) for slot in binding.occurrences)


def _word_count_surface(text: str) -> tuple[int, int, int] | None:
    """Closed package counts only; incidental number words are not numeric facts."""
    from .descendant import _number_word_value

    match = re.fullmatch(
        r"\s*(?P<number>[A-Za-z]+(?:[ -]+[A-Za-z]+)*)\s+"
        r"(?:PALLETS?|CARTONS?|PACKAGES?|CRATES?|DRUMS?|BUNDLES?|CASES?|"
        r"PIECES?|ROLLS?|SACKS?|BAGS?|BOX(?:ES)?)(?:\s+ONLY)?\s*",
        text,
        re.I,
    )
    if match is None:
        return None
    value = _number_word_value(match["number"])
    if value is None:
        return None
    return value, match.start("number"), match.end("number")


def render_numeric_surface(text: str, old: Decimal, new: Decimal) -> str:
    word_count = _word_count_surface(text)
    if word_count is None:
        return render_number_surface(text, old, new)
    from .descendant import _case_like, _number_to_words, _pluralize_number_word_noun

    value, start, end = word_count
    if old != value or new < 0 or new != new.to_integral_value():
        raise ValueError("package number words disagree with the integral numeric contract")
    if old == new:
        return text
    return (
        text[:start]
        + _case_like(text[start:end], _number_to_words(int(new)))
        + _pluralize_number_word_noun(text[end:], number=int(new))
    )


def surface_quantum(text: str, old: Decimal) -> Decimal:
    word_count = _word_count_surface(text)
    if word_count is not None:
        if old != word_count[0]:
            raise ValueError("package number words disagree with the numeric source value")
        return Decimal(1)
    options = {
        (match.start(), match.end(), precision)
        for match in _NUMBER.finditer(text)
        for value, decimal, grouping, precision in _numeric_interpretations(match.group())
        if value == old
    }
    if len(options) != 1:
        raise ValueError("numeric source grammar is ambiguous")
    if render_number_surface(text, old, old) != text:
        raise ValueError("numeric contract does not reproduce source")
    return Decimal(1).scaleb(-next(iter(options))[2])


def validate_contract(
    binding: SemanticBinding,
    contract: NumericContract,
    source_target: Mapping[str, Any],
    *,
    source_template: CertifiedSemanticTemplate | None = None,
) -> None:
    if not contract.reason.strip():
        raise ValueError("numeric dependency explanation missing")
    if binding.value_kind == "package" and contract.role != "cargo_quantity":
        raise ValueError("numbered package surfaces require an explicit cargo quantity contract")
    if contract.mode == "review_required":
        raise ValueError(
            f"numeric semantic review required: {binding.logical_key}: {contract.reason}"
        )
    if contract.mode == "target_equation_count":
        from . import package_equations

        if (
            contract.role != "cargo_quantity"
            or contract.multiplier != "1"
            or contract.divisor != 1
            or contract.dependency_bindings
            or len(contract.target_paths) != 1
        ):
            raise ValueError("package-equation contract has incompatible numeric metadata")
        observed = package_equations.binding_value(
            binding, source_target, source_target, contract.target_paths[0]
        )
        if _number(contract.source_value) != observed:
            raise ValueError("package-equation contract source quantity differs")
        quantum(binding, _number(contract.source_value))
        return
    if contract.mode == "binding_sum":
        if (
            not contract.dependency_bindings
            or len(set(contract.dependency_bindings)) != len(contract.dependency_bindings)
            or binding.logical_key in contract.dependency_bindings
            or contract.target_paths
            or contract.multiplier != "1"
            or contract.divisor != 1
            or contract.role not in {"cargo_quantity", "cargo_mass", "cargo_volume"}
        ):
            raise ValueError("binding sum requires distinct other same-dimension cargo bindings")
        quantum(binding, _number(contract.source_value))
        return
    if contract.dependency_bindings:
        raise ValueError("only binding_sum may declare auxiliary numeric dependencies")
    if contract.mode == "surface_fixed":
        if contract.role not in {
            "metadata",
            "dimensions",
            "commercial",
            "operational",
            "temperature",
        }:
            raise ValueError("fixed surface cannot hide a mutable cargo quantity or identity")
        if contract.source_value != binding.occurrences[0].source_text:
            raise ValueError("fixed-surface contract does not identify its exact source text")
        if contract.target_paths or contract.multiplier != "1" or contract.divisor != 1:
            raise ValueError("fixed surface cannot hide a numeric target dependency")
        return
    old = _number(contract.source_value)
    step = quantum(binding, old)
    if contract.mode == "sampled_equipment_tare":
        if (
            contract.role != "tare"
            or old <= 0
            or contract.target_paths
            or contract.multiplier != "1"
            or contract.divisor != 1
        ):
            raise ValueError("sampled equipment tare requires a positive source-only tare")
        return
    if contract.mode == "target_converted":
        if source_template is None or len(contract.target_paths) != 1 or contract.divisor != 1:
            raise ValueError("converted measurement needs its source template and one target leaf")
        path = contract.target_paths[0]
        volume = contract.multiplier in {"cbm_to_cbf", "cbf_to_cbm"}
        pattern = r"\.volume\.value$" if volume else r"\.(?:grossWeight|netWeight)\.value$"
        if not re.search(pattern, path) or contract.role != (
            "cargo_volume" if volume else "cargo_mass"
        ):
            raise ValueError("unit conversion requires the corresponding typed cargo measure")
        factor, unit = conversion_factor(contract.multiplier)
        if resolve(source_target, path.removesuffix("value") + "unit") != unit:
            raise ValueError("mass conversion disagrees with the target unit")
        original = _number(resolve(source_target, path))
        precisions = [
            precision
            for candidate in source_template.bindings
            if path in candidate.target_paths
            or path.removesuffix(".value") in candidate.target_paths
            for slot in candidate.occurrences
            for match in _NUMBER.finditer(slot.source_text)
            for value, _, _, precision in _numeric_interpretations(match.group())
            if value == original
        ]
        if not precisions:
            raise ValueError("source conversion lacks a proved target-side printed precision")
        target_step = Decimal(1).scaleb(-max(precisions))
        # Two rounded printed measurements are consistent only if their rounding
        # intervals overlap after the exact unit conversion. No arbitrary epsilon.
        if abs(original * factor - old) > (step + target_step * factor) / 2:
            raise ValueError("source unit-conversion rounding intervals do not overlap")
        return
    multiplier = _number(contract.multiplier)
    if multiplier not in {Decimal(1), Decimal(1000), Decimal("0.001")}:
        raise ValueError("numeric contract conversion is not identity or kg/tonne SI conversion")
    if contract.mode != "target_average" and contract.divisor != 1:
        raise ValueError("only equal-partition contracts may specify a divisor")
    if contract.mode in {"target_sum", "target_share", "target_average", "unit_product"}:
        if not contract.target_paths or len(set(contract.target_paths)) != len(
            contract.target_paths
        ):
            raise ValueError("target-derived numeric contract needs unique paths")
        if any(
            isinstance(resolve(source_target, p), bool)
            or not isinstance(resolve(source_target, p), (int, float))
            for p in contract.target_paths
        ):
            raise ValueError(
                "numeric dependencies must be typed numeric leaves, not identifiers or prose"
            )
        if contract.mode == "unit_product":
            if contract.role != "per_unit_measurement" or len(contract.target_paths) != 2:
                raise ValueError("unit-product needs quantity and measurement dependencies")
            count, measurement = (_number(resolve(source_target, p)) for p in contract.target_paths)
            if count <= 0 or measurement != count * old * multiplier:
                raise ValueError("source unit-product arithmetic is false")
            return
        total = (
            sum((_number(resolve(source_target, p)) for p in contract.target_paths), Decimal(0))
            * multiplier
        )
        if contract.mode == "target_average":
            count_support = {
                len(binding.occurrences),
                len(source_target.get("documentPatch", {}).get("containers", [])),
            }
            if contract.divisor not in count_support or contract.divisor < 2:
                raise ValueError("equal partition lacks one source occurrence per constituent")
            if contract.role == "cargo_quantity":
                if (
                    len(contract.target_paths) != 1
                    or not re.fullmatch(
                        r"documentPatch\.cargoPackages\[\d+\]\.quantity",
                        contract.target_paths[0],
                    )
                    or multiplier != 1
                    or old != old.to_integral_value()
                    or step != 1
                ):
                    raise ValueError("equal package partition requires one integer package total")
            elif any(
                not re.search(r"\.(?:grossWeight|netWeight|volume)\.value$", p)
                for p in contract.target_paths
            ):
                raise ValueError("equal partition requires a shipment measurement total")
            total /= contract.divisor
        if (
            (contract.mode == "target_sum" and total != old)
            or (contract.mode == "target_share" and not 0 <= old <= total)
            or (contract.mode == "target_average" and total != old)
        ):
            raise ValueError(
                f"numeric source dependency is false: {binding.logical_key}: {total} != {old}"
            )
    elif contract.target_paths or multiplier != 1:
        raise ValueError(
            "non-derived numeric contract cannot hide target dependencies or conversion"
        )
    if contract.mode == "source_scaled" and contract.role not in {
        "cargo_mass",
        "cargo_volume",
        "cargo_quantity",
    }:
        raise ValueError("only shipment totals can scale with the cargo scenario")
    if contract.mode == "source_fixed" and contract.role in {
        "cargo_mass",
        "cargo_volume",
        "cargo_quantity",
        "unknown",
    }:
        raise ValueError("shipment totals cannot be silently frozen")


def _allocate_equal_precision(
    keys: Sequence[str], weights: Sequence[Decimal], total: Decimal, step: Decimal
) -> dict[str, Decimal]:
    if total / step != (total / step).to_integral_value():
        raise ValueError("target share total exceeds printed precision")
    units = int(total / step)
    if units < sum(v > 0 for v in weights):
        raise ValueError("target total cannot preserve printed nonzero constituents")
    weight_total = sum(weights)
    quotas = [Decimal(units) * value / weight_total for value in weights]
    counts = [max(1 if old > 0 else 0, int(q)) for old, q in zip(weights, quotas, strict=True)]
    while sum(counts) > units:
        eligible = [i for i, n in enumerate(counts) if n > (1 if weights[i] > 0 else 0)]
        selected = max(eligible, key=lambda i: (Decimal(counts[i]) - quotas[i], keys[i]))
        counts[selected] -= 1
    while sum(counts) < units:
        eligible = [i for i, value in enumerate(weights) if value > 0]
        selected = max(eligible, key=lambda i: (quotas[i] - counts[i], keys[i]))
        counts[selected] += 1
    return {key: Decimal(n) * step for key, n in zip(keys, counts, strict=True)}


def _allocate_mixed_precision(
    keys: Sequence[str], weights: Sequence[Decimal], steps: Sequence[Decimal], total: Decimal
) -> dict[str, Decimal]:
    """Allocate coarse printed rows first, reserving exact finer-row minima.

    Decimal surface quanta form a divisibility chain. The finest nonzero tier
    receives the remainder, so no mass/volume is lost by independently rounding
    rows with different displayed precision. Zero source rows remain zero.
    """
    active = [i for i, value in enumerate(weights) if value > 0]
    tiers = sorted({steps[i] for i in active}, reverse=True)
    if any(a % b for a, b in pairwise(tiers)):
        raise ValueError("target-share precision is not a decimal divisibility chain")
    if total % tiers[-1]:
        raise ValueError("target share total exceeds printed precision")
    if total < sum(steps[i] for i in active):
        raise ValueError("target total cannot preserve printed nonzero constituents")
    result = {key: Decimal(0) for key in keys}
    remaining = total
    original_total = sum(weights)
    for step in tiers:
        members = [i for i in active if steps[i] == step]
        finer = [i for i in active if steps[i] < step]
        if finer:
            ideal = total * sum(weights[i] for i in members) / original_total
            upper = ((remaining - sum(steps[i] for i in finer)) / step).to_integral_value(
                rounding=ROUND_FLOOR
            )
            units = min(
                upper,
                max(
                    Decimal(len(members)), (ideal / step).to_integral_value(rounding=ROUND_HALF_UP)
                ),
            )
            budget = units * step
        else:
            budget = remaining
        result.update(
            _allocate_equal_precision(
                [keys[i] for i in members], [weights[i] for i in members], budget, step
            )
        )
        remaining -= budget
    if remaining or sum(result.values()) != total:
        raise ValueError("target-share allocation did not conserve its total")
    return result


def prepare(
    bindings: Sequence[SemanticBinding],
    contracts: Mapping[str, NumericContract],
    *,
    source_target: Mapping[str, Any],
    target: Mapping[str, Any],
    scale: Decimal,
    source_template: CertifiedSemanticTemplate | None = None,
    equipment_tare_values: Mapping[str, Decimal] | None = None,
) -> dict[str, PreparedNumeric]:
    if set(contracts) != {b.logical_key for b in bindings}:
        raise ValueError("numeric contracts do not cover every unowned numeric binding")
    sampled_tares = {
        key for key, contract in contracts.items() if contract.mode == "sampled_equipment_tare"
    }
    if equipment_tare_values is not None and set(equipment_tare_values) != sampled_tares:
        raise ValueError("sampled equipment tare values must cover exactly the declared bindings")
    # The identity replay validates the observed source, before any scenario exists.
    # Every non-identity scenario must supply the physical sampler's values; absence
    # can never silently restore the source tare during synthesis.
    source_replay = scale == 1 and target == source_target
    if sampled_tares and equipment_tare_values is None and not source_replay:
        raise ValueError("sampled equipment tare values are required for a generated scenario")
    from . import package_equations

    # A proved equation is stronger than an independently scaled unowned count.
    # Record the derived contract explicitly; never change a frozen target.
    contracts = dict(contracts)
    for binding in bindings:
        contract = contracts[binding.logical_key]
        if contract.mode == "source_scaled" and contract.role == "cargo_quantity":
            path = package_equations.binding_dependency(binding, source_target)
            if path is not None:
                contracts[binding.logical_key] = contract.model_copy(
                    update={
                        "mode": "target_equation_count",
                        "target_paths": [path],
                        "reason": (
                            "Exact same-unit source count owned by the complete target package "
                            "equation; its topology and inner totals are solved together."
                        ),
                    }
                )
    independent_counts = {
        key: _number(contract.source_value)
        for key, contract in contracts.items()
        if contract.mode == "source_scaled" and contract.role == "cargo_quantity"
    }
    if len(independent_counts) >= 3:
        combined = sum(independent_counts.values())
        for key, value in independent_counts.items():
            if value > 0 and combined - value == value:
                raise ValueError(
                    "source-only quantity aggregate requires an explicit binding_sum "
                    f"contract: {key}"
                )
    prepared = {}
    shares: dict[tuple[tuple[str, ...], str], list[SemanticBinding]] = {}
    for binding in bindings:
        contract = contracts[binding.logical_key]
        if contract.mode == "target_share":
            shares.setdefault(
                (tuple(sorted(contract.target_paths)), contract.multiplier), []
            ).append(binding)
    share_values = {}
    for (paths, multiplier), members in shares.items():
        source_total = sum(
            (_number(resolve(source_target, p)) for p in paths), Decimal(0)
        ) * _number(multiplier)
        old_values = [_number(contracts[b.logical_key].source_value) for b in members]
        if sum(old_values) != source_total or source_total <= 0:
            raise ValueError(
                "target-share constituents do not exactly account for their source total"
            )
        steps = [quantum(b, v) for b, v in zip(members, old_values, strict=True)]
        total = sum((_number(resolve(target, p)) for p in paths), Decimal(0)) * _number(multiplier)
        keys = [b.logical_key for b in members]
        share_values.update(
            _allocate_equal_precision(keys, old_values, total, steps[0])
            if len(set(steps)) == 1
            else _allocate_mixed_precision(keys, old_values, steps, total)
        )
    for binding in bindings:
        contract = contracts[binding.logical_key]
        validate_contract(binding, contract, source_target, source_template=source_template)
        if contract.mode == "binding_sum":
            continue
        if contract.mode == "surface_fixed":
            prepared[binding.logical_key] = PreparedNumeric(
                contract=contract, value=contract.source_value, scenario_scale=str(scale)
            )
            continue
        old = _number(contract.source_value)
        step = quantum(binding, old)
        if contract.mode == "sampled_equipment_tare":
            value = (
                old
                if equipment_tare_values is None
                else _number(equipment_tare_values[binding.logical_key])
            )
            if value <= 0 or value.quantize(step) != value:
                raise ValueError(
                    "sampled equipment tare must be positive and exactly representable"
                )
        elif contract.mode == "target_equation_count":
            value = Decimal(
                package_equations.binding_value(
                    binding, source_target, target, contract.target_paths[0]
                )
            )
        elif contract.mode == "target_converted":
            path = contract.target_paths[0]
            original, current = (_number(resolve(doc, path)) for doc in (source_target, target))
            value = (
                old
                if current == original
                else (current * conversion_factor(contract.multiplier)[0]).quantize(
                    step, rounding=ROUND_HALF_UP
                )
            )
        elif contract.mode == "target_share":
            value = share_values[binding.logical_key]
        elif contract.mode in {"target_sum", "target_average"}:
            value = (
                sum((_number(resolve(target, p)) for p in contract.target_paths), Decimal(0))
                * _number(contract.multiplier)
                / contract.divisor
            )
            if value.quantize(step) != value:
                raise ValueError("target-derived quantity exceeds numeric source precision")
        elif contract.mode == "source_scaled":
            value = (old * scale).quantize(step, rounding=ROUND_HALF_UP)
            if (binding.value_kind == "integer" or contract.role == "cargo_quantity") and old > 0:
                value = max(Decimal(1), value)
        elif contract.mode == "unit_product":
            expected_total = derived_target_values({binding.logical_key: contract}, target)
            if any(_number(resolve(target, p)) != v for p, v in expected_total.items()):
                raise ValueError("generated unit-product arithmetic is false")
            value = old
        else:
            value = old
        prepared[binding.logical_key] = PreparedNumeric(
            contract=contract, value=str(value), scenario_scale=str(scale)
        )
    pending = {b.logical_key: b for b in bindings if contracts[b.logical_key].mode == "binding_sum"}
    for key in pending:
        contract = contracts[key]
        if any(dep not in contracts for dep in contract.dependency_bindings):
            raise ValueError("numeric binding sum references an unknown binding")
        if any(contracts[dep].role != contract.role for dep in contract.dependency_bindings):
            raise ValueError("numeric binding sum mixes incompatible quantity roles")
        if contract.role in {"cargo_mass", "cargo_volume"}:
            if source_template is None:
                raise ValueError("physical binding sum requires the source template unit evidence")
            _validate_physical_sum_units(key, contract, contracts)
        if sum(
            _number(contracts[dep].source_value) for dep in contract.dependency_bindings
        ) != _number(contract.source_value):
            raise ValueError("numeric binding sum source arithmetic is false")
    while pending:
        ready = [
            key
            for key in pending
            if all(dep in prepared for dep in contracts[key].dependency_bindings)
        ]
        if not ready:
            raise ValueError("numeric binding sum dependencies contain a cycle")
        for key in ready:
            binding = pending.pop(key)
            contract = contracts[key]
            value = sum(
                (_number(prepared[dep].value) for dep in contract.dependency_bindings), Decimal(0)
            )
            step = quantum(binding, _number(contract.source_value))
            if value.quantize(step) != value:
                raise ValueError("numeric binding sum exceeds printed precision")
            prepared[key] = PreparedNumeric(
                contract=contract, value=str(value), scenario_scale=str(scale)
            )
    return prepared


def _validate_physical_sum_units(
    key: str,
    contract: NumericContract,
    contracts: Mapping[str, NumericContract],
) -> None:
    """Check dimensional consistency; physical-row compilation proves source quotes."""
    from .equipment_row_constraints import _QUOTE_UNITS, _UNITS

    # Byte templates intentionally contain the original text only as span evidence;
    # require explicit contextual quotes/private units for a dimensional sum. The
    # regular physical-row compiler independently checks these quotes against the
    # actual source before the scenario is accepted.
    keys = (key, *contract.dependency_bindings)
    units: set[tuple[str, Decimal]] = set()
    for item in keys:
        member = contracts[item]
        if member.synthetic_unit is not None:
            units.add(("mass" if member.synthetic_unit == "kilogram" else "volume", Decimal(1)))
        elif member.printed_unit_quote is not None:
            expected = "mass" if member.role == "cargo_mass" else "volume"
            evidence = {
                _UNITS[m[1].upper()] for m in _QUOTE_UNITS.finditer(member.printed_unit_quote)
            }
            evidence = {value for value in evidence if value[0] == expected}
            if len(evidence) != 1:
                raise ValueError("physical binding sum quote must declare one compatible unit")
            units.add(evidence.pop())
        else:
            raise ValueError("physical binding sum needs explicit unit evidence for every member")
    if len(units) != 1:
        raise ValueError("physical binding sum mixes different measurement units")


def conversion_factor(name: str) -> tuple[Decimal, str]:
    # NIST Handbook 44 Appendix C: one avoirdupois pound is exactly 0.45359237 kg.
    if name == "kg_to_lb":
        return Decimal(1) / Decimal("0.45359237"), "kilogram"
    if name == "lb_to_kg":
        return Decimal("0.45359237"), "pound"
    # The international foot is exactly 0.3048 metres; cube the length factor.
    if name == "cbm_to_cbf":
        return Decimal(1) / Decimal("0.3048") ** 3, "cubic_metre"
    if name == "cbf_to_cbm":
        return Decimal("0.3048") ** 3, "cubic_foot"
    raise ValueError("unsupported mass conversion")


def derived_target_values(
    contracts: Mapping[str, NumericContract], target: Mapping[str, Any]
) -> dict[str, Decimal]:
    """Generate constrained totals before the target is accepted or frozen."""
    values: dict[str, Decimal] = {}
    for contract in contracts.values():
        if contract.mode != "unit_product":
            continue
        count_path, measure_path = contract.target_paths
        value = (
            _number(resolve(target, count_path))
            * _number(contract.source_value)
            * _number(contract.multiplier)
        )
        if measure_path in values and values[measure_path] != value:
            raise ValueError("unit-product constraints disagree on generated measurement")
        values[measure_path] = value
    return values


def generated_measurements(
    bindings: Sequence[SemanticBinding],
    contracts: Mapping[str, NumericContract],
    target: Mapping[str, Any],
) -> dict[str, Decimal]:
    """Honor every printed measurement precision before committing the scenario."""
    steps: dict[str, list[Decimal]] = {}
    for binding in bindings:
        contract = contracts[binding.logical_key]
        if contract.mode not in {"target_sum", "target_share", "target_average"}:
            continue
        step = (
            quantum(binding, _number(contract.source_value))
            * contract.divisor
            / _number(contract.multiplier)
        )
        for path in contract.target_paths:
            if re.search(r"\.(?:grossWeight|netWeight|volume)\.value$", path):
                steps.setdefault(path, []).append(step)
    values = {}
    for path, required in steps.items():
        scale = Decimal(10) ** max(0, max(-int(v.as_tuple().exponent) for v in required))
        step = Decimal(lcm(*(int(v * scale) for v in required))) / scale
        current = _number(resolve(target, path))
        values[path] = (
            max(step, (current / step).to_integral_value(rounding=ROUND_FLOOR) * step)
            if current
            else Decimal(0)
        )
    for path, value in derived_target_values(contracts, target).items():
        if any(value % step for step in steps.get(path, [])):
            raise ValueError("unit-product total cannot fit all printed measurement surfaces")
        values[path] = value
    return values


def equal_partition_steps(
    contracts: Mapping[str, NumericContract], template: CertifiedSemanticTemplate
) -> dict[str, Decimal]:
    """Total increments whose equal constituents fit every printed precision.

    A pair of rows printed to 0.001 requires a total increment of 0.002, not
    merely three decimal places. SI conversions apply before intersecting the
    lattices. Multiple dependent totals conservatively use the same lattice,
    so their sum also remains exactly divisible without post-generation repair.
    """
    averages = {
        k: c
        for k, c in contracts.items()
        if c.mode == "target_average" and c.role != "cargo_quantity"
    }
    if not averages:
        return {}
    by_key = {b.logical_key: b for b in template.bindings if b.logical_key in averages}
    steps: dict[str, list[Fraction]] = {}
    for key, contract in averages.items():
        step = (
            quantum(by_key[key], _number(contract.source_value))
            * contract.divisor
            / _number(contract.multiplier)
        )
        for path in contract.target_paths:
            steps.setdefault(path, []).append(Fraction(step))
    return {
        path: Decimal(lcm(*(s.numerator for s in values)))
        / Decimal(gcd(*(s.denominator for s in values)))
        for path, values in steps.items()
    }


def minimum_quantities(contracts: Mapping[str, NumericContract]) -> dict[str, int]:
    groups: dict[str, int] = {}
    for contract in contracts.values():
        if (
            contract.mode == "target_share"
            and contract.role == "cargo_quantity"
            and len(contract.target_paths) == 1
            and contract.multiplier == "1"
        ):
            path = contract.target_paths[0]
            if re.fullmatch(
                r"documentPatch\.(?:cargoPackages\[\d+\]\.quantity|cargoAllocationGroups\[\d+\]\.allocations\[\d+\]\.packageQuantity)",
                path,
            ):
                groups[path] = groups.get(path, 0) + int(_number(contract.source_value) > 0)
    return groups


def quantity_multiples(
    contracts: Mapping[str, NumericContract],
    source_target: Mapping[str, Any],
    template: CertifiedSemanticTemplate,
) -> dict[str, int]:
    """Solve count divisibility before drawing equal rows or per-unit mass totals."""
    result: dict[str, int] = {}
    for key, contract in contracts.items():
        if contract.mode == "target_average" and contract.role == "cargo_quantity":
            binding = next(b for b in template.bindings if b.logical_key == key)
            validate_contract(binding, contract, source_target, source_template=template)
            path = contract.target_paths[0]
            result[path] = lcm(result.get(path, 1), contract.divisor)
    for contract in contracts.values():
        if contract.mode != "unit_product":
            continue
        count_path, measure_path = contract.target_paths
        if not re.fullmatch(r"documentPatch\.cargoPackages\[\d+\]\.quantity", count_path):
            raise ValueError("unit-product count must identify a structured package quantity")
        old = _number(resolve(source_target, measure_path))
        steps = [quantum(b, old) for b in template.bindings if measure_path in b.target_paths]
        by_key = {b.logical_key: b for b in template.bindings}
        steps.extend(
            quantum(by_key[key], _number(dependency.source_value))
            * dependency.divisor
            / _number(dependency.multiplier)
            for key, dependency in contracts.items()
            if dependency.mode in {"target_sum", "target_share", "target_average"}
            and measure_path in dependency.target_paths
        )
        if not steps:
            raise ValueError("unit-product measurement lacks proved printed total precision")
        factor = _number(contract.source_value) * _number(contract.multiplier)
        multiple = lcm(*(Fraction(factor / step).denominator for step in steps))
        result[count_path] = lcm(result.get(count_path, 1), multiple)
    return result


def render_prepared(
    bindings: Sequence[SemanticBinding],
    prepared: Mapping[str, PreparedNumeric],
    *,
    source_target: Mapping[str, Any],
    target: Mapping[str, Any],
    source_template: CertifiedSemanticTemplate | None = None,
    equipment_tare_values: Mapping[str, Decimal] | None = None,
) -> dict[str, dict[str, str]]:
    if not bindings and not prepared:
        return {}
    scales = {row.scenario_scale for row in prepared.values()}
    if len(scales) != 1:
        raise ValueError("numeric auxiliaries disagree on scenario scale")
    expected = prepare(
        bindings,
        {key: row.contract for key, row in prepared.items()},
        source_target=source_target,
        target=target,
        scale=_number(next(iter(scales))),
        source_template=source_template,
        equipment_tare_values=equipment_tare_values,
    )
    if expected != prepared:
        raise ValueError("prepared numeric auxiliary differs from its dependency contract")
    return {
        binding.logical_key: {
            slot.slot_id: slot.source_text
            if prepared[binding.logical_key].contract.mode == "surface_fixed"
            else render_numeric_surface(
                slot.source_text,
                _number(prepared[binding.logical_key].contract.source_value),
                _number(prepared[binding.logical_key].value),
            )
            for slot in binding.occurrences
        }
        for binding in bindings
    }
