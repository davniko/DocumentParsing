"""Registry-owned DG facts and deterministic rendering outside training labels.

The extraction schema intentionally stores broad hazard categories. Exact printed
classes, packing groups and shipping names belong to the scenario, not an LLM's
guess from those broad labels. Every generated tuple is retained as a whole.
"""

from __future__ import annotations

import re
from collections.abc import Mapping
from typing import TYPE_CHECKING, Annotated, Any, Literal

from pydantic import BaseModel, ConfigDict, Field

from document_ocr.synthesis.dangerous_goods_registry import DangerousGoodsHmtRecord

from .models import CertifiedSemanticTemplate, NonEmptyText, SemanticBinding

if TYPE_CHECKING:
    from .descendant import BindingOutput


def validate_un_references(text: str, expected: set[str]) -> None:
    mentioned = set(re.findall(r"\bUN\s*(?:NO\.?|NUMBER)?\s*[:#-]?\s*(\d{4})(?!\d)", text, re.I))
    if mentioned - expected:
        raise ValueError("text contains a UN number outside its sampled DG facts")


class DangerousGoodsFact(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)
    target_path: NonEmptyText
    record: DangerousGoodsHmtRecord

    def validate_target(self, target: Mapping[str, Any]) -> None:
        from .descendant import _resolve_path

        value = _resolve_path(target, self.target_path)
        expected = {
            "unNumber": self.record.un_number,
            "hazardCategory": self.record.hazard_category,
            "subsidiaryHazardCategories": list(self.record.subsidiary_hazard_categories),
            "packingGroupCategory": self.record.packing_group_category,
        }
        if not isinstance(value, Mapping):
            raise ValueError("DG scenario path does not identify a declaration")
        for key, actual in value.items():
            if key == "flashPoint":
                raise ValueError("new DG flashpoints require a formulation-property contract")
            if key not in expected or actual != expected[key]:
                raise ValueError(f"target DG declaration differs from its registry tuple: {key}")
        if not self.record.maritime_eligible:
            raise ValueError("DG scenario is not maritime eligible")


DgField = Literal[
    "un_number", "primary_class", "subsidiary_class", "packing_group", "shipping_name"
]


class DangerousGoodsSurface(BaseModel):
    """A reviewed binding to one regulatory tuple, possibly shared by cargo groups."""

    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)
    logical_key: NonEmptyText
    target_path: NonEmptyText
    shared_target_paths: tuple[NonEmptyText, ...] = ()
    field: DgField
    subsidiary_index: Annotated[int, Field(ge=0)] | None = None


_DECLARATION = re.compile(r"(documentPatch\.cargoGroups\[\d+\]\.dangerousGoods\[\d+\])\.")
_CLASS = re.compile(r"(?<![A-Za-z0-9.])([1-9](?:\.[1-6])?[A-Z]?)(?![A-Za-z0-9.])")
_PACKING = re.compile(
    r"^(?P<prefix>\s*(?:(?:P\.?G\.?|GR\.?|PACKING\s+GROUP)\s*[:.]?\s*)?)(?P<value>III|II|I|[123])(?P<suffix>\s*)$",
    re.IGNORECASE,
)
_CAPTION = re.compile(r"\s*(?:UN|CLASS|P\.?\s*G\.?|PACKING\s+GROUP)\s*[:.]?\s*", re.I)
DERIVATIONS: Mapping[str, DgField] = {
    "sampled_dg_un_number": "un_number",
    "sampled_dg_primary_class": "primary_class",
    "sampled_dg_packing_group": "packing_group",
    "sampled_dg_shipping_name": "shipping_name",
}
_OWNER = re.compile(r"documentPatch\.cargoGroups\[\d+\]\.dangerousGoods\[\d+\]")
_EXPLICIT_UN_LINE = re.compile(r"(?im)^\s*UN\s*(?:NUMBER|NO\.?)\s*[:#-]?\s*([0-9]{4})\b")
_UN_SURFACE = re.compile(r"\s*(?:UN\s*[:#-]?\s*)?\d{4}\s*", re.I)
_CLASS_SURFACE = re.compile(r"\s*(?:(?:CLASS|CL\.?)\s*[:.]?\s*)?[1-9](?:\.[1-6])?[A-Z]?\s*", re.I)
_OTHER_PROPERTY = re.compile(
    r"\b(?:FLASH\s*POINT|CLASS|UN\s*(?:NO\.?|NUMBER)?\s*[:#-]?\s*\d{4}|"
    r"P\.?G\.?\s*[:.]?\s*(?:III|II|I)\b|NET\s+WEIGHT|GROSS\s+WEIGHT)\b",
    re.I,
)


def validate_explicit_un_source_coverage(raw: str, source_target: Mapping[str, Any]) -> None:
    """Fail closed when an affirmative printed UN declaration is absent from labels.

    Form captions, negative DG statements and generic legal text without a
    concrete UN number are not positive declarations. A concrete UN outside the
    current target needs source-role review rather than silent source-only use.
    """

    printed = set(_EXPLICIT_UN_LINE.findall(raw))
    if not printed:
        return
    target = {
        dangerous["unNumber"]
        for group in source_target["documentPatch"].get("cargoGroups", [])
        for dangerous in group.get("dangerousGoods", [])
        if dangerous.get("unNumber") is not None
    }
    if missing := printed - target:
        raise ValueError(
            "explicit printed DG UN number lacks a task target: " + ", ".join(sorted(missing))
        )


def reviewed_shared_draft_contract(
    raw: str,
    source_target: Mapping[str, Any],
    draft: Any,
    occurrences: tuple[Any, ...],
) -> bool:
    """Prove a repeated multi-target DG property has an explicit shared tuple.

    This is a compiler acceptance contract, not a guess from equal target values.
    The shared group annotation, repeated equal property surfaces, equal full DG
    source records, and repeated printed UN declarations are all required.
    """

    if (
        draft.group_kind != "dangerous_goods"
        or draft.value_kind != "dangerous_goods"
        or not draft.group_key.startswith("dangerous_goods:shared:")
        or len(draft.target_paths) < 2
        or len(occurrences) != len(draft.target_paths)
        or not raw
    ):
        return False
    owner_paths: list[str] = []
    leaves: set[str] = set()
    for path in draft.target_paths:
        match = _DECLARATION.match(path)
        if match is None:
            return False
        owner_paths.append(match.group(1))
        leaves.add(path[len(match.group(1)) + 1 :])
    if len(set(owner_paths)) != len(owner_paths) or len(leaves) != 1:
        return False
    if len({row.source_text.strip().casefold() for row in occurrences}) != 1:
        return False
    groups = source_target["documentPatch"].get("cargoGroups", [])
    declarations = []
    for path in owner_paths:
        indices = tuple(int(value) for value in re.findall(r"\[([0-9]+)\]", path))
        if len(indices) != 2:
            return False
        gi, di = indices
        try:
            declarations.append(groups[gi]["dangerousGoods"][di])
        except (IndexError, KeyError, TypeError):
            return False
    if any(value != declarations[0] for value in declarations[1:]):
        return False
    un_number = declarations[0].get("unNumber")
    return isinstance(un_number, str) and _EXPLICIT_UN_LINE.findall(raw) == [un_number] * len(
        owner_paths
    )


def explicit_surface(binding: Any, declaration_paths: set[str]) -> DangerousGoodsSurface:
    """Typed private fact ownership; group labels are not declaration identifiers.

    Compilation/critic review must prove the semantic span. These invariants
    prevent an annotation from selecting a missing tuple or swallowing another
    property's value. No additional extraction fields are required or created.
    """
    field = DERIVATIONS.get(binding.derivation)
    if (
        field is None
        or binding.render_mode != "deterministic_derived"
        or binding.group_kind != "dangerous_goods"
        or binding.value_kind != "dangerous_goods"
        or binding.target_paths
        or binding.dependency_bindings
        or len(binding.dependency_paths) != 1
        or _OWNER.fullmatch(binding.dependency_paths[0]) is None
        or binding.dependency_paths[0] not in declaration_paths
    ):
        raise ValueError(
            "explicit DG fact requires one existing declaration owner and a typed surface"
        )
    occurrences = getattr(binding, "occurrences", (binding,))
    if not occurrences:
        raise ValueError("explicit DG fact has no printed occurrence")
    grammars = {
        "un_number": _UN_SURFACE,
        "primary_class": _CLASS_SURFACE,
        "packing_group": _PACKING,
    }
    for slot in occurrences:
        if field in grammars:
            valid = grammars[field].fullmatch(slot.source_text) is not None
        else:
            valid = bool(re.search(r"[A-Za-z]", slot.source_text)) and not _OTHER_PROPERTY.search(
                slot.source_text
            )
        if not valid:
            raise ValueError(
                "explicit DG fact surface mixes properties or lacks its declared grammar"
            )
    return DangerousGoodsSurface(
        logical_key=binding.logical_key, target_path=binding.dependency_paths[0], field=field
    )


def compile_surfaces(template: CertifiedSemanticTemplate) -> tuple[DangerousGoodsSurface, ...]:
    """Resolve explicit target owners and unambiguous typed auxiliary groups.

    Unknown source-only properties are a review result, never copied into a new
    chemical scenario. No document identities or proximity heuristics are used.
    """
    groups: dict[str, set[str]] = {}
    for binding in template.bindings:
        for path in binding.target_paths:
            if match := _DECLARATION.match(path):
                groups.setdefault(binding.group_key, set()).add(match.group(1))
    declarations = set().union(*groups.values()) if groups else set()
    output = []
    for binding in template.bindings:
        if getattr(binding, "derivation", None) in DERIVATIONS:
            output.append(explicit_surface(binding, declarations))
            continue
        # A compiler may explicitly inventory a column caption in its DG group.
        # Only a closed, value-free literal grammar is invariant across chemicals;
        # static declarations such as 'CLASS 3' must still receive a fact owner.
        if (
            binding.render_mode == "literal_static"
            and not binding.target_paths
            and not binding.dependency_paths
            and not binding.dependency_bindings
            and all(_CAPTION.fullmatch(s.source_text) for s in binding.occurrences)
        ):
            continue
        owned = [p for p in binding.target_paths if _DECLARATION.match(p)]
        if not owned and binding.group_kind != "dangerous_goods":
            continue
        if owned:
            if len(owned) != len(binding.target_paths):
                raise ValueError("composite DG facts require an explicit surface contract")
            declaration = _DECLARATION.match(owned[0])
            assert declaration is not None
            owner = declaration.group(1)
            leaf = owned[0][len(owner) + 1 :]
            shared_owners: list[str] = []
            for path in owned[1:]:
                shared = _DECLARATION.match(path)
                assert shared is not None
                if path[len(shared.group(1)) + 1 :] != leaf:
                    raise ValueError("shared DG surface combines unlike regulatory properties")
                shared_owners.append(shared.group(1))
            if shared_owners and (
                binding.target_relationship != "shared_value_equality"
                or len({owner, *shared_owners}) != len(owned)
            ):
                raise ValueError("shared DG surface lacks equal distinct declaration owners")
            field: DgField
            index = None
            if leaf == "unNumber":
                field = "un_number"
            elif leaf == "hazardCategory":
                field = "primary_class"
            elif leaf in {"subsidiaryHazardCategory", "subsidiaryHazardCategories[0]"}:
                field, index = "subsidiary_class", 0
            elif leaf in {"packingGroupCategory", "flashPoint.packingGroupCategory"}:
                field = "packing_group"
            else:
                raise ValueError(f"DG property requires a typed realization contract: {path}")
        else:
            owners = groups.get(binding.group_key, set())
            if len(owners) != 1:
                raise ValueError(
                    f"source-only DG fact has no unique declaration owner: {binding.logical_key}"
                )
            owner = next(iter(owners))
            if all(_PACKING.fullmatch(s.source_text) for s in binding.occurrences):
                field = "packing_group"
            else:
                # Free prose cannot be typed merely because its logical key says
                # 'description'. It may include chemical qualifiers or properties.
                raise ValueError(
                    f"source-only DG fact needs compilation review: {binding.logical_key}"
                )
            index = None
            shared_owners = []
        output.append(
            DangerousGoodsSurface(
                logical_key=binding.logical_key,
                target_path=owner,
                shared_target_paths=tuple(shared_owners),
                field=field,
                subsidiary_index=index,
            )
        )
    return tuple(output)


def shared_declaration_representatives(
    template: CertifiedSemanticTemplate, source_target: Mapping[str, Any]
) -> dict[str, str]:
    """Return only explicitly co-bound DG declarations that must share one tuple.

    Equal source values alone never imply a shared synthetic identity. The
    compiler must co-bind the same property paths under shared-value equality.
    """

    groups: list[frozenset[str]] = []
    fields: dict[frozenset[str], set[DgField]] = {}
    for surface in compile_surfaces(template):
        if not surface.shared_target_paths:
            continue
        owners = frozenset((surface.target_path, *surface.shared_target_paths))
        if any(owners & previous and owners != previous for previous in groups):
            raise ValueError("overlapping DG shared-declaration contracts differ")
        if owners not in fields:
            groups.append(owners)
        fields.setdefault(owners, set()).add(surface.field)
    result: dict[str, str] = {}
    for owners in groups:
        if "un_number" not in fields[owners]:
            raise ValueError("shared DG declarations require an owned UN-number surface")
        source_values = []
        cargo_groups = source_target["documentPatch"]["cargoGroups"]
        for path in owners:
            match = _OWNER.fullmatch(path)
            assert match is not None
            group_index, declaration_index = (
                int(value) for value in re.findall(r"\[([0-9]+)\]", path)
            )
            source_values.append(cargo_groups[group_index]["dangerousGoods"][declaration_index])
        if any(value != source_values[0] for value in source_values[1:]):
            raise ValueError("shared DG source declarations are not the same regulatory tuple")
        representative = min(owners)
        result.update(dict.fromkeys(owners, representative))
    return result


def _render_field(
    binding: SemanticBinding, contract: DangerousGoodsSurface, fact: DangerousGoodsFact
) -> BindingOutput:
    from .descendant import BindingOutput, _layout_like_source

    record = fact.record
    if contract.field == "un_number":
        value = record.un_number
    elif contract.field == "primary_class":
        value = record.exact_hazard_class
    elif contract.field == "subsidiary_class":
        if contract.subsidiary_index is None or contract.subsidiary_index >= len(
            record.exact_subsidiary_hazards
        ):
            raise ValueError("DG tuple lacks a printed subsidiary hazard")
        value = record.exact_subsidiary_hazards[contract.subsidiary_index]
    elif contract.field == "packing_group":
        if record.packing_group_code is None:
            raise ValueError("DG tuple lacks a printed packing group")
        value = record.packing_group_code
    else:
        value = record.proper_shipping_name
    replacements = {}
    for slot in binding.occurrences:
        source = slot.source_text
        if contract.field == "un_number":
            matches = list(re.finditer(r"(?<![0-9])[0-9]{4}(?![0-9])", source))
            if len(matches) != 1:
                raise ValueError("DG UN surface does not contain one complete four-digit identity")
            match = matches[0]
            replacement = source[: match.start()] + value + source[match.end() :]
        elif contract.field in {"primary_class", "subsidiary_class"}:
            matches = list(_CLASS.finditer(source))
            if len(matches) != 1:
                raise ValueError("DG class surface does not contain one complete class")
            match = matches[0]
            replacement = source[: match.start()] + value + source[match.end() :]
        elif contract.field == "packing_group":
            packing_match = _PACKING.fullmatch(source)
            if packing_match is None:
                raise ValueError("DG packing-group source grammar is unresolved")
            printed = (
                str({"I": 1, "II": 2, "III": 3}[value])
                if packing_match["value"].isdigit()
                else value
            )
            replacement = packing_match["prefix"] + printed + packing_match["suffix"]
        else:
            replacement = _layout_like_source(source, value)
        replacements[slot.slot_id] = replacement
    return BindingOutput(replacements=replacements, canonical_value=value)


def render_facts(
    *,
    source: bytes,
    template: CertifiedSemanticTemplate,
    target: Mapping[str, Any],
    facts: tuple[DangerousGoodsFact, ...],
) -> dict[str, BindingOutput]:
    from .descendant import _validate_binding_format

    by_path = {f.target_path: f for f in facts}
    if len(by_path) != len(facts):
        raise ValueError("duplicate DG scenario declarations")
    expected_paths = {
        f"documentPatch.cargoGroups[{gi}].dangerousGoods[{di}]"
        for gi, group in enumerate(target["documentPatch"].get("cargoGroups", []))
        for di, _ in enumerate(group.get("dangerousGoods", []))
    }
    if set(by_path) != expected_paths:
        raise ValueError("DG registry facts do not cover target declarations exactly")
    surfaces = compile_surfaces(template)
    for fact in facts:
        fact.validate_target(target)
    bindings = {b.logical_key: b for b in template.bindings}
    if len({s.logical_key for s in surfaces}) != len(surfaces):
        raise ValueError("duplicate DG surface owners")
    result = {}
    for surface in surfaces:
        if surface.target_path not in by_path or surface.logical_key not in bindings:
            raise ValueError("DG surface is outside its scenario or template")
        for shared_path in surface.shared_target_paths:
            if (
                shared_path not in by_path
                or by_path[shared_path].record.record_id
                != by_path[surface.target_path].record.record_id
            ):
                raise ValueError("shared DG labels must use one complete regulatory tuple")
        binding = bindings[surface.logical_key]
        output = _render_field(binding, surface, by_path[surface.target_path])
        _validate_binding_format(source=source, template=template.byte_template, output=output)
        result[surface.logical_key] = output
    return result
