"""Printed package-unit mass lower bounds without inventing extraction totals.

An explicitly owned ``400 X 25 KG BAGS`` row proves 10,000 kg of stated package
mass. It does not prove that an omitted total is a net-weight label. Reviewed
package ownership, an exact quantity binding and a local printed mass unit are
required; neither numeric proximity nor a model's role name supplies evidence.
"""

from __future__ import annotations

import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from decimal import Decimal
from typing import Any

from .complete_targets import SourceTemplate
from .numeric_auxiliary import NumericContract


@dataclass(frozen=True)
class PackageUnitLoad:
    binding_key: str
    package_index: int
    group_id: str
    mass_kg: Decimal


_OWNER = re.compile(r"documentPatch\.cargoPackages\[(\d+)\]")
_COUNT = re.compile(r"(?<![\w.,])(?P<count>\d+(?:,\d{3})*)[ \t]*[X\u00d7*][ \t]*$", re.I)
_MASS = re.compile(
    r"(?P<mass>\d+(?:[.,]\d+)?)[ \t]*(?P<unit>KGS?|KGM|KILOGRAMS?|LBS?|POUNDS?|MT|TONNES?)\b", re.I
)
_PACK = re.compile(
    r"[ \t]*(?:NET[ \t]+)?(?P<pack>BAGS?|CARTONS?|BOX(?:ES)?|DRUMS?|SACKS?|BALES?|PAILS?)\b", re.I
)
_CATEGORIES = {
    "BAG": "PACKAGE_BAG",
    "BAGS": "PACKAGE_BAG",
    "CARTON": "PACKAGE_CARTON",
    "CARTONS": "PACKAGE_CARTON",
    "BOX": "PACKAGE_BOX",
    "BOXES": "PACKAGE_BOX",
    "DRUM": "PACKAGE_DRUM",
    "DRUMS": "PACKAGE_DRUM",
    "SACK": "PACKAGE_SACK",
    "SACKS": "PACKAGE_SACK",
    "BALE": "PACKAGE_BALE",
    "BALES": "PACKAGE_BALE",
    "PAIL": "PACKAGE_PAIL",
    "PAILS": "PACKAGE_PAIL",
}
_FACTORS = {"kilogram": Decimal(1), "metric_tonne": Decimal(1000), "pound": Decimal("0.45359237")}


def compile_loads(
    source: SourceTemplate, contracts: Mapping[str, NumericContract]
) -> tuple[PackageUnitLoad, ...]:
    """Compile only explicitly declared package-mass scopes; other roles stay separate."""
    if not any(c.role == "per_unit_measurement" for c in contracts.values()):
        return ()
    patch = source.target["documentPatch"]
    packages = patch.get("cargoPackages", ())
    groups = {g["groupId"] for g in patch.get("cargoGroups", ())}
    result: dict[int, PackageUnitLoad] = {}
    for binding in source.template.bindings:
        contract = contracts.get(binding.logical_key)
        if contract is None or contract.role != "per_unit_measurement":
            continue
        dependencies = binding.dependency_paths
        if not any(_OWNER.fullmatch(p) for p in dependencies):
            continue
        if len(dependencies) != 1 or (owner := _OWNER.fullmatch(dependencies[0])) is None:
            raise ValueError("package unit mass requires one reviewed package owner")
        index = int(owner[1])
        if index >= len(packages):
            raise ValueError("package unit mass owner is outside source package inventory")
        package = packages[index]
        if (
            package["groupId"] not in groups
            or type(package.get("quantity")) is not int
            or package["quantity"] <= 0
        ):
            raise ValueError("package unit mass needs a positive quantity and known cargo group")
        if (
            contract.mode != "source_fixed"
            or contract.target_paths
            or contract.dependency_bindings
            or contract.multiplier != "1"
            or contract.divisor != 1
            or binding.target_paths
        ):
            raise ValueError("package unit mass requires a fixed source-only unit value")
        observed = Decimal(contract.source_value)
        if not observed.is_finite() or observed <= 0:
            raise ValueError("package unit mass must be finite and positive")
        masses = set()
        for slot in binding.occurrences:
            if source.source[slot.byte_start : slot.byte_end].decode() != slot.source_text:
                raise ValueError("package unit mass slot differs from certified source bytes")
            left = source.source.rfind(b"\n", 0, slot.byte_start) + 1
            right = source.source.find(b"\n", slot.byte_end)
            if right < 0:
                right = len(source.source)
            prefix = source.source[left : slot.byte_start].decode()
            count = _COUNT.search(prefix)
            if count is None or int(count["count"].replace(",", "")) != package["quantity"]:
                raise ValueError("package unit mass lacks an exact printed quantity-times-mass row")
            count_start = left + len(prefix[: count.start("count")].encode())
            count_end = left + len(prefix[: count.end("count")].encode())
            path = f"documentPatch.cargoPackages[{index}].quantity"
            if not any(
                path in candidate.target_paths
                and any(
                    s.byte_start == count_start and s.byte_end == count_end
                    for s in candidate.occurrences
                )
                for candidate in source.template.bindings
            ):
                raise ValueError("package unit mass count is not owned by the declared package")
            suffix = source.source[slot.byte_start : right].decode()
            mass = _MASS.match(suffix)
            if mass is None or Decimal(mass["mass"].replace(",", ".")) != observed:
                raise ValueError("package unit mass lacks the same printed number and mass unit")
            if slot.byte_end > slot.byte_start + len(suffix[: mass.end()].encode()):
                raise ValueError("package unit mass slot includes unrelated text")
            noun = _PACK.match(suffix, mass.end())
            if noun is None or package.get("typeCategory") != _CATEGORIES[noun["pack"].upper()]:
                raise ValueError("package unit mass noun contradicts the observed package category")
            unit = mass["unit"].upper()
            factor = (
                Decimal(1)
                if unit.startswith("K")
                else Decimal("0.45359237")
                if unit.startswith(("L", "P"))
                else Decimal(1000)
            )
            masses.add(observed * factor)
        if len(masses) != 1:
            raise ValueError("repeated package unit mass observations disagree or are absent")
        load = PackageUnitLoad(binding.logical_key, index, package["groupId"], masses.pop())
        previous = result.get(index)
        if previous is not None and previous.mass_kg != load.mass_kg:
            raise ValueError("package unit mass contracts disagree on one package row")
        # Repeated statements of one package mass are evidence aliases, not extra load.
        result.setdefault(index, load)
    loads = tuple(result[i] for i in sorted(result))
    validate_gross(lower_bounds(loads, source.target), source.target)
    return loads


def lower_bounds(loads: Sequence[PackageUnitLoad], target: Mapping[str, Any]) -> dict[str, Decimal]:
    """Return group mass floors, not netWeight labels; never mutate the target."""
    packages = target["documentPatch"].get("cargoPackages", ())
    result: dict[str, Decimal] = {}
    seen = set()
    for load in loads:
        if load.package_index in seen:
            raise ValueError("package unit load duplicates a package row")
        seen.add(load.package_index)
        if load.package_index >= len(packages):
            raise ValueError("generated package inventory omits a unit-mass owner")
        package = packages[load.package_index]
        count = package.get("quantity")
        if type(count) is not int or count <= 0 or package["groupId"] != load.group_id:
            raise ValueError(
                "generated package unit load has invalid quantity or changed cargo ownership"
            )
        result[load.group_id] = (
            result.get(load.group_id, Decimal(0)) + Decimal(count) * load.mass_kg
        )
    return result


def validate_gross(bounds: Mapping[str, Decimal], target: Mapping[str, Any]) -> None:
    """Check observed gross mass when present; caller enforces private capacity too."""
    groups = {g["groupId"]: g for g in target["documentPatch"].get("cargoGroups", ())}
    for group_id, floor in bounds.items():
        if group_id not in groups:
            raise ValueError("package unit mass has no generated cargo group")
        gross = groups[group_id].get("grossWeight")
        if gross is not None:
            mass = Decimal(str(gross["value"])) * _FACTORS[gross["unit"]]
            if not mass.is_finite() or mass < floor:
                raise ValueError("cargo gross weight is below explicitly printed package-unit mass")
