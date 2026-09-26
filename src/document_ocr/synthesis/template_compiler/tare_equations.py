"""Source-proved cargo gross = net + printed equipment tare equations."""

from __future__ import annotations

import re
from collections.abc import Mapping
from dataclasses import dataclass
from decimal import Decimal
from typing import Any

from .complete_targets import SourceTemplate
from .numeric_auxiliary import NumericContract

_EQUATION = re.compile(r"gross\s+weight\s*\(\s*tare\s*\+\s*nett?\s*\)", re.I)


@dataclass(frozen=True, slots=True)
class TareEquation:
    group_id: str
    tare_kilograms: Decimal
    binding_key: str


def _proved_source_only_row_equations(
    source: SourceTemplate,
    contracts: Mapping[str, NumericContract],
    tares: list[tuple[str, NumericContract]],
) -> bool:
    """Recognize independently owned container-row sums, not cargo totals."""
    by_key = {binding.logical_key: binding for binding in source.template.bindings}
    owned_gross: set[str] = set()
    owned_net: set[str] = set()
    for tare_key, tare_contract in tares:
        if tare_contract.mode != "source_fixed" or tare_contract.target_paths:
            return False
        gross = [
            binding
            for binding in source.template.bindings
            if binding.derivation == "sum_decimal_values"
            and tare_key in binding.dependency_bindings
        ]
        if len(gross) != 1 or len(gross[0].dependency_bindings) != 2:
            return False
        gross_binding = gross[0]
        net_key = next(key for key in gross_binding.dependency_bindings if key != tare_key)
        net_contract = contracts.get(net_key)
        if (
            net_key in owned_net
            or gross_binding.logical_key in owned_gross
            or net_contract is None
            or net_contract.role != "cargo_mass"
            or net_contract.target_paths
            or net_key not in by_key
            or not gross_binding.occurrences
        ):
            return False
        try:
            total = Decimal(net_contract.source_value) + Decimal(tare_contract.source_value)
            exact = all(
                Decimal(slot.source_text) == total
                and source.source[slot.byte_start : slot.byte_end].decode("utf-8")
                == slot.source_text
                for slot in gross_binding.occurrences
            )
        except (ArithmeticError, UnicodeDecodeError):
            return False
        if not exact:
            return False
        owned_net.add(net_key)
        owned_gross.add(gross_binding.logical_key)
    return len(owned_gross) == len(tares)


def compile_equations(
    source: SourceTemplate, contracts: Mapping[str, NumericContract]
) -> tuple[TareEquation, ...]:
    """Require complete printed ownership before accepting a mass equation.

    An unfilled form caption is not an equation. The current observed grammar
    has one cargo group; multiple groups need explicit tare-to-group ownership.
    """
    if _EQUATION.search(source.source.decode("utf-8")) is None:
        return ()
    groups = source.target["documentPatch"].get("cargoGroups") or []
    tares = [(key, contract) for key, contract in contracts.items() if contract.role == "tare"]
    complete = [g for g in groups if "netWeight" in g and "grossWeight" in g]
    if not tares and not complete:
        return ()  # Printed caption, but no filled mass facts.
    if not complete and tares and _proved_source_only_row_equations(source, contracts, tares):
        return ()  # The row-level gross derivations already enforce each equation.
    if len(groups) != 1 or len(complete) != 1 or len(tares) != 1:
        raise ValueError("printed gross/net/tare equation lacks one complete cargo owner")
    group = complete[0]
    key, contract = tares[0]
    if contract.mode != "source_fixed" or contract.target_paths:
        raise ValueError("printed tare equation needs a fixed source-only tare")
    if any(group[name].get("unit") != "kilogram" for name in ("netWeight", "grossWeight")):
        raise ValueError("printed tare equation has incompatible target mass units")
    binding = next((b for b in source.template.bindings if b.logical_key == key), None)
    if binding is None or not binding.occurrences:
        raise ValueError("printed tare equation has no certified tare binding")
    tare = Decimal(contract.source_value)
    if tare <= 0 or any(
        Decimal(slot.source_text) != tare
        or source.source[slot.byte_start : slot.byte_end].decode("utf-8") != slot.source_text
        for slot in binding.occurrences
    ):
        raise ValueError("printed tare equation binding differs from source tare")
    net = Decimal(str(group["netWeight"]["value"]))
    gross = Decimal(str(group["grossWeight"]["value"]))
    if net <= 0 or net + tare != gross:
        raise ValueError("source gross weight does not equal printed net plus tare")
    return (TareEquation(group["groupId"], tare, key),)


def apply_equations(target: dict[str, Any], equations: tuple[TareEquation, ...]) -> None:
    groups = {group["groupId"]: group for group in target["documentPatch"].get("cargoGroups", [])}
    for equation in equations:
        group = groups.get(equation.group_id)
        if group is None or "netWeight" not in group or "grossWeight" not in group:
            raise ValueError("sampled cargo lost a printed mass-equation owner")
        if any(group[name].get("unit") != "kilogram" for name in ("netWeight", "grossWeight")):
            raise ValueError("sampled mass equation changed kilograms")
        net = Decimal(str(group["netWeight"]["value"]))
        if net <= 0:
            raise ValueError("sampled net weight is not positive")
        group["grossWeight"]["value"] = float(net + equation.tare_kilograms)


def validate_equations(target: Mapping[str, Any], equations: tuple[TareEquation, ...]) -> None:
    groups = {group["groupId"]: group for group in target["documentPatch"].get("cargoGroups", [])}
    for equation in equations:
        group = groups.get(equation.group_id)
        if group is None or "netWeight" not in group or "grossWeight" not in group:
            raise ValueError("printed mass equation is absent from final target")
        net = Decimal(str(group["netWeight"]["value"]))
        gross = Decimal(str(group["grossWeight"]["value"]))
        if (
            group["netWeight"].get("unit") != "kilogram"
            or group["grossWeight"].get("unit") != "kilogram"
            or net + equation.tare_kilograms != gross
        ):
            raise ValueError("final gross weight differs from net plus printed tare")
