"""Source-proved anonymous fleets, kept outside observable extraction labels.

A printed complete count/type receipt can constrain a shipment even when no
container IDs are printed. It is not permission to invent extraction labels or
to disregard other equipment statements. Unresolved scopes fail explicitly.
"""

from __future__ import annotations

import re
from collections.abc import Mapping
from copy import deepcopy
from dataclasses import dataclass, replace
from decimal import Decimal
from typing import Any

from . import complete_targets as targets
from . import equipment_receipts, unit_package_loads
from . import numeric_auxiliary as numeric


@dataclass(frozen=True)
class AnonymousInventory:
    source: targets.SourceTemplate
    physical_source: targets.SourceTemplate
    binding_keys: tuple[str, ...]
    row_pairs: tuple[str | None, ...]
    private_measurements: tuple[tuple[str, str], ...]

    @property
    def equipment_pairs(self) -> tuple[str, ...]:
        return tuple(dict.fromkeys(pair for pair in self.row_pairs if pair is not None))

    def proposed(self, target: Mapping[str, Any], *, sample_id: str, seed: int) -> dict[str, Any]:
        result = deepcopy(dict(target))
        patch = result["documentPatch"]
        if patch.get("containers"):
            raise ValueError("anonymous inventory cannot overwrite labelled containers")
        patch["containers"] = deepcopy(self.physical_source.target["documentPatch"]["containers"])
        bindings = {b.logical_key: b for b in self.source.template.bindings}
        for key, field in self.private_measurements:
            if field in patch["cargoGroups"][0]:
                raise ValueError("anonymous physical measurement cannot overwrite a label")
            observed = self.physical_source.target["documentPatch"]["cargoGroups"][0][field]
            old = Decimal(str(observed["value"]))
            value = (old * targets.scenario_scale(sample_id, seed)).quantize(
                numeric.quantum(bindings[key], old)
            )
            patch["cargoGroups"][0][field] = {"unit": observed["unit"], "value": float(value)}
        return result

    def observable(self, target: dict[str, Any]) -> dict[str, Any]:
        """Remove only the privately introduced facts, recording every removed fact."""
        patch = target["documentPatch"]
        containers = patch.pop("containers")
        if "containers" in self.source.target["documentPatch"]:
            patch["containers"] = []
        measurements = {
            field: patch["cargoGroups"][0].pop(field) for _, field in self.private_measurements
        }
        return dict(
            sourceEquipmentBindings=self.binding_keys,
            printedEquipmentRetained=True,
            retentionReason="printed_inventory_without_individual_extraction_identifiers",
            privateEquipment=containers,
            privateCargoMeasurements=measurements,
            labelVisibilityUnchanged=True,
        )


def _observation(surface: str) -> tuple[int | None, equipment_receipts._EquipmentShape | None]:
    """Parse complete receipts or explicit count/type components without inferring either."""
    text = surface.strip()
    tail = re.search(equipment_receipts._TAIL + "$", text, re.I)
    assert tail is not None
    bare = text[: tail.start()].strip()
    multiplier = re.fullmatch(r"(0*\d+)\s*[xX*]", bare)
    if multiplier is not None:
        # Explicitly owned split inventory multipliers are counts, not a type
        # called '2 x'. Their separate type evidence is checked by the caller.
        return int(multiplier[1]), None
    # Stripping the noun from "1 CTNR" leaves just a count. A bare integer
    # without the printed equipment noun is not an equipment observation.
    if re.fullmatch(r"0*\d+", bare) and re.search(equipment_receipts._NOUN, text, re.I):
        return int(bare), None
    for pattern in (
        equipment_receipts._REVERSE,
        equipment_receipts._COMPACT_REVERSE,
        equipment_receipts._FORWARD,
    ):
        term = pattern.fullmatch(bare)
        if term is not None:
            if (
                pattern is equipment_receipts._FORWARD
                and term["count"] in {"20", "40", "45"}
                and term["join"].isspace()
                and not term["equipment"][0].isdigit()
            ):
                # "40 RH" is one type component, never forty RH containers.
                return None, equipment_receipts._semantic(bare)
            try:
                shape = equipment_receipts._semantic(term["equipment"])
            except ValueError:
                # A dimension such as "40 RH" can resemble the forward
                # count grammar. It remains a type component, not count 40.
                if pattern is equipment_receipts._FORWARD:
                    break
                raise
            return int(term["count"]), shape
    return None, equipment_receipts._semantic(bare)


def _inventory_rows(source: targets.SourceTemplate, receipts: list[Any]) -> list[dict[str, Any]]:
    """Join only components sharing a compiler-reviewed inventory scope.

    Separate groups are not additive or repeated by default. The compiler must
    express that relationship before they can define one private inventory.
    """
    groups = {getattr(binding, "group_key", None) for binding in receipts}
    if len(groups) != 1:
        raise ValueError("anonymous equipment components lack one reviewed inventory scope")
    counts: set[int] = set()
    shapes: set[equipment_receipts._EquipmentShape] = set()
    surfaces: list[str] = []
    bare_counts = False
    keys = {binding.logical_key for binding in receipts}
    for binding in receipts:
        if getattr(binding, "dependency_paths", ()) not in ((), ("documentPatch.containers",)):
            raise ValueError("anonymous equipment has a contradictory inventory dependency")
        for slot in binding.occurrences:
            if source.source[slot.byte_start : slot.byte_end].decode() != slot.source_text:
                raise ValueError("anonymous equipment source evidence is stale")
        if getattr(binding, "derivation", None) == "container_package_count":
            if not binding.dependency_bindings or not set(binding.dependency_bindings) <= keys:
                raise ValueError("anonymous count summary lacks its reviewed equipment owner")
            continue
        for slot in binding.occurrences:
            if (
                re.fullmatch(r"[0-9]+", slot.source_text.strip())
                and getattr(slot, "render_policy", None) == "numeric_surface"
                and binding.group_kind == "equipment"
                and len(receipts) > 1
            ):
                # A reviewed equipment/numeric component joined to a separate
                # type in the same scope is a count, not an arbitrary number
                # discovered by scanning OCR. Never invent a printed noun.
                counts.add(int(slot.source_text.strip()))
                bare_counts = True
                surfaces.append(slot.source_text)
                continue
            count, shape = _observation(slot.source_text)
            if count is not None:
                counts.add(count)
            if shape is not None:
                shapes.add(shape)
            surfaces.append(slot.source_text)
    if len(counts) != 1 or next(iter(counts)) <= 0:
        raise ValueError("anonymous equipment lacks one explicit positive inventory count")
    if bare_counts and not shapes:
        raise ValueError("bare equipment count requires a reviewed separate type component")
    count = next(iter(counts))
    # Every non-null predicate must agree, including explicit length versus
    # complete dimension/category. Unknown predicates remain private choices.
    size = {shape.size for shape in shapes if shape.size is not None}
    kind = {shape.kind for shape in shapes if shape.kind is not None}
    length = {shape.length for shape in shapes if shape.length is not None}
    if any(len(values) > 1 for values in (size, kind, length)):
        raise ValueError("anonymous repeated equipment inventories disagree")
    row: dict[str, Any] = {}
    if size and kind:
        row["sizeCategory"] = next(iter(size))
        row["typeCategory"] = next(iter(kind))
    elif size or kind:
        raise ValueError("anonymous equipment partial shape lacks a representable dimension")
    if length and not size:
        row["typeDescription"] = next(iter(length)) + " FT"
    elif surfaces and shapes and not size:
        # Height/type-only predicates cannot yet be represented by the existing
        # private equipment sampler without fabricating a missing dimension.
        raise ValueError("anonymous equipment partial shape lacks a representable dimension")
    if any(not equipment_receipts._compatible(row, shape) for shape in shapes):
        raise ValueError("anonymous repeated equipment inventories disagree")
    return [dict(row, containerNumber=f"private-equipment-{i}") for i in range(count)]


def compile_inventory(
    source: targets.SourceTemplate,
    contracts: Mapping[str, numeric.NumericContract],
) -> AnonymousInventory | None:
    patch = source.target["documentPatch"]
    if patch.get("containers"):
        return None
    receipts = [
        b
        for b in source.template.bindings
        if b.value_kind == "equipment" and not b.target_paths and b.render_mode != "carrier_static"
    ]
    if not receipts:
        return None
    if not patch.get("cargoGroups") or patch.get("cargoAllocationGroups"):
        raise ValueError("anonymous equipment requires cargo owners and no observed allocations")
    containers = _inventory_rows(source, receipts)
    physical = deepcopy(source.target)
    physical["documentPatch"]["containers"] = containers
    changes = {
        binding.logical_key: binding.model_copy(
            update={
                "render_mode": "deterministic_derived",
                "derivation": "equipment_receipt",
                "dependency_paths": ("documentPatch.containers",),
                "dependency_bindings": (),
            }
        )
        for binding in receipts
    }
    fields = []
    package_mass_keys = {
        load.binding_key for load in unit_package_loads.compile_loads(source, contracts)
    }
    for binding in numeric.numeric_bindings(source.template):
        contract = contracts[binding.logical_key]
        if binding.logical_key in package_mass_keys:
            continue  # Explicit package owners are checked by the joint cargo sampler.
        if contract.synthetic_unit is None:
            if contract.role in {
                "cargo_mass",
                "cargo_volume",
                "per_unit_measurement",
                "dimensions",
                "density",
            } and contract.mode in {"source_scaled", "source_fixed", "surface_fixed"}:
                raise ValueError(
                    "anonymous equipment has an unowned source-only physical measurement"
                )
            continue
        if len(patch["cargoGroups"]) != 1:
            raise ValueError("anonymous private measurement requires one cargo owner")
        # A unit choice is not ownership. Only an immediately printed explicit
        # GROSS WEIGHT / VOLUME caption and one cargo group establish this scope.
        owners = set()
        for slot in binding.occurrences:
            prefix = source.source[: slot.byte_start].decode()
            heading = re.search(
                r"(?:^|\n)[ \t]*(GROSS[ \t]+WEIGHT|VOLUME)[ \t]*:?[ \t]*\n[ \t\r\n]*$", prefix, re.I
            )
            if heading is None:
                raise ValueError(
                    "private anonymous measurement lacks a unique printed cargo heading"
                )
            owners.add("grossWeight" if heading[1].upper().startswith("GROSS") else "volume")
        if len(owners) != 1:
            raise ValueError("private anonymous measurement has conflicting printed owners")
        field = owners.pop()
        expected_unit = "kilogram" if field == "grossWeight" else "cubic_metre"
        if (
            contract.synthetic_unit != expected_unit
            or field in physical["documentPatch"]["cargoGroups"][0]
        ):
            raise ValueError("private anonymous measurement contradicts its physical cargo owner")
        physical["documentPatch"]["cargoGroups"][0][field] = {
            "value": float(Decimal(contract.source_value)),
            "unit": contract.synthetic_unit,
        }
        changes[binding.logical_key] = binding.model_copy(
            update={"dependency_paths": ("documentPatch.containers",)}
        )
        fields.append((binding.logical_key, field))
    template = source.template.model_copy(
        update={
            "bindings": tuple(changes.get(b.logical_key, b) for b in source.template.bindings),
        }
    )
    return AnonymousInventory(
        source,
        replace(source, target=physical, template=template),
        tuple(binding.logical_key for binding in receipts),
        tuple(
            row["sizeCategory"] + "|" + row["typeCategory"]
            if "sizeCategory" in row and "typeCategory" in row
            else None
            for row in containers
        ),
        tuple(fields),
    )
