"""Declared package-count authority, noun aliases and contained packing.

A reviewed template chooses the declared unit. Contained packing can then scale
with it without becoming a public package row or inventing a rounded count.
"""

from __future__ import annotations

import re
from collections.abc import Sequence
from copy import deepcopy
from dataclasses import asdict, dataclass
from decimal import Decimal, InvalidOperation
from fractions import Fraction
from math import lcm
from random import Random
from typing import TYPE_CHECKING

from document_ocr.synthesis.curated import _number_style, digest, flat
from document_ocr.synthesis.package_aliases import PackageAliasMode, select_package_alias
from document_ocr.synthesis.template_compiler.descendant import _number_to_words

if TYPE_CHECKING:
    from document_ocr.synthesis.curated_physical import PhysicalRenderPlan
    from document_ocr.synthesis.curated_templates import SamplingBlueprint


def validate_contained_quantity_sampling(
    blueprint: SamplingBlueprint, *, multiple: int, fixed: int | None
) -> None:
    """Prove all configured draws can render whole contained quantities."""
    specs = [
        recipe["contained_quantity"]
        for recipe in blueprint.ownership_data.get("surfaces", {}).values()
        if "contained_quantity" in recipe
    ]
    if not specs:
        return
    source = flat(blueprint.target)
    required = 1
    for spec in specs:
        required = lcm(
            required,
            Fraction(spec["source_quantity"], source[spec["quantity_path"]]).denominator,
        )
    if (fixed if fixed is not None else multiple) % required:
        raise ValueError(f"quantity sampling must be divisible by {required} for contained packing")


def contained_quantity_surfaces(
    blueprint: SamplingBlueprint, target: dict, key: str, recipe: dict
) -> list[str]:
    """Scale a private count by the exact reviewed source accounting ratio."""
    if set(recipe) != {"quantity_path", "source_quantity", "source_unit"}:
        raise ValueError(f"invalid contained quantity recipe: {key}")
    unit = recipe["source_unit"]
    if not isinstance(unit, str) or not unit.strip() or unit != unit.strip():
        raise ValueError(f"contained quantity requires a printed source unit: {key}")
    path, inner = recipe["quantity_path"], recipe["source_quantity"]
    if not isinstance(path, str) or not re.fullmatch(
        r"documentPatch\.goodsItemDetails\[\d+\]\.numberAndTypeOfPackages\[\d+\]\.packageQuantity",
        path,
    ):
        raise ValueError(f"contained quantity requires declared package path: {key}")
    old, new = flat(blueprint.target).get(path), flat(target).get(path)
    if any(type(value) is not int or value <= 0 for value in (old, new, inner)):
        raise ValueError(f"contained quantity requires positive integer counts: {key}")
    binding = blueprint.historical_bindings[key]
    if binding.get("target_paths"):
        raise ValueError(f"contained count cannot own public shipment targets: {key}")
    count = Fraction(new * inner, old)
    if count.denominator != 1:
        raise ValueError(f"contained quantity is fractional; revise quantity multiple: {key}")
    result = []
    source_bytes = blueprint.source.encode("utf-8")
    for occurrence in binding["occurrences"]:
        source = occurrence["source_text"]
        suffix = source_bytes[occurrence["byte_end"] :].decode("utf-8")
        if not re.match(r"\s*" + re.escape(unit) + r"(?!\w)", suffix):
            raise ValueError(f"contained unit does not follow its source count: {key}")
        if source.casefold() == _number_to_words(inner).casefold():
            value = _number_to_words(count.numerator)
            if source.isupper():
                value = value.upper()
            elif source.istitle():
                value = value.title()
            elif source[:1].isupper() and source[1:].islower():
                value = value.capitalize()
            elif not source.islower():
                raise ValueError(f"unsupported contained-count word casing: {key}")
            result.append(value)
        else:
            # This also rejects a mismatched original count; the declaration is
            # a source certificate, not permission to overwrite arbitrary text.
            result.append(_number_style(source, str(inner), str(count.numerator)))
    if not result:
        raise ValueError(f"contained quantity has no source occurrences: {key}")
    return result


def contained_packing_context(blueprint: SamplingBlueprint, target: dict) -> list[dict]:
    """Give generation and review the same certified private packing as rendering."""
    result = []
    for key, surface in blueprint.ownership_data.get("surfaces", {}).items():
        if "contained_quantity" not in surface:
            continue
        recipe = surface["contained_quantity"]
        contained_quantity_surfaces(blueprint, target, key, recipe)
        path = recipe["quantity_path"]
        declared = flat(target)[path]
        count = Fraction(declared * recipe["source_quantity"], flat(blueprint.target)[path])
        result.append(
            {
                "declared_package_path": path,
                "declared_quantity": declared,
                "contained_quantity": count.numerator,
                "printed_contained_unit": recipe["source_unit"],
            }
        )
    return result


_QUANTITY = re.compile(
    r"^documentPatch\.goodsItemDetails\[\d+\]\.(?:numberAndTypeOfPackages|splitGoodsPlacement)\[\d+\]\.packageQuantity$"
)
_CATEGORY = re.compile(
    r"^documentPatch\.goodsItemDetails\[\d+\]\.numberAndTypeOfPackages\[\d+\]\.typeCategory$"
)
_DIRECT_EXPRESSION = re.compile(r"^\{([^{}]+)\}$")


@dataclass(frozen=True, slots=True)
class PackageCountFact:
    """One source-pinned occurrence and the exact host-owned canonical count."""

    owner: str
    occurrence: int
    byte_start: int
    byte_end: int
    source_text_sha256: str
    source_sha256: str
    target_sha256: str
    quantity: int
    target_paths: tuple[str, ...]
    basis: str

    def receipt(self) -> dict:
        result = asdict(self)
        result["target_paths"] = list(self.target_paths)
        return result


def _positive_integer(value, role: str) -> int:
    if isinstance(value, bool) or not isinstance(value, (str, int, Decimal)):
        raise ValueError(f"{role} must be a positive integer")
    try:
        number = Decimal(value)
    except InvalidOperation as error:
        raise ValueError(f"{role} must be a positive integer") from error
    if not number.is_finite() or number <= 0 or number != number.to_integral_value():
        raise ValueError(f"{role} must be a positive integer")
    return int(number)


def _verify_region(source_bytes: bytes, region) -> None:
    if not 0 <= region.start < region.end <= len(source_bytes) or source_bytes[
        region.start : region.end
    ] != region.source.encode("utf-8"):
        raise ValueError("count/package occurrence differs from its pinned source span")


def _count_authority(blueprint, physical_plan, region, leaves, variable_kinds):
    if region.curated_key is not None:
        key = region.curated_key
        if variable_kinds.get(key) != "count":
            raise ValueError("package quantity is not owned by a count variable")
        paths = tuple(
            binding.path
            for binding in blueprint.contract.targets
            if binding.expression == "{" + key + "}" and _QUANTITY.fullmatch(binding.path)
        )
        if not paths:
            raise ValueError("curated count lacks an exact package-count expression")
        if key not in physical_plan.variable_values:
            raise ValueError("physical plan omitted its authoritative count variable")
        quantity = _positive_integer(physical_plan.variable_values[key], "physical count")
        basis = "curated_count_expression"
    else:
        recipe = blueprint.ownership_data.get("surfaces", {}).get(region.key, {})
        expression = recipe.get("expression")
        match = _DIRECT_EXPRESSION.fullmatch(expression) if isinstance(expression, str) else None
        if match is not None and _QUANTITY.fullmatch(match[1]):
            paths = (match[1],)
            basis = "declared_count_expression"
        else:
            paths = tuple(region.target_paths)
            if not paths or not all(_QUANTITY.fullmatch(path) for path in paths):
                raise ValueError("historical count has unresolved non-count ownership")
            basis = "unambiguous_historical_count_owner"
        quantities = {_positive_integer(leaves[path], "target count") for path in paths}
        if len(quantities) != 1:
            raise ValueError("historical count has conflicting canonical owners")
        quantity = quantities.pop()
    if any(_positive_integer(leaves[path], "target count") != quantity for path in paths):
        raise ValueError("physical count differs from its authoritative target expression")
    return quantity, paths, basis


def _scalar_package_count_owner(blueprint: SamplingBlueprint, region) -> bool:
    """Keep complete count/type composites under their physical surface recipe."""
    if not any(_QUANTITY.fullmatch(path) for path in region.target_paths):
        return False
    if region.curated_key is not None or all(
        _QUANTITY.fullmatch(path) for path in region.target_paths
    ):
        return True
    expression = blueprint.ownership_data.get("surfaces", {}).get(region.key, {}).get("expression")
    match = _DIRECT_EXPRESSION.fullmatch(expression) if isinstance(expression, str) else None
    return match is not None and _QUANTITY.fullmatch(match[1]) is not None


def compile_package_count_context(
    blueprint: SamplingBlueprint, physical_plan: PhysicalRenderPlan
) -> tuple[PackageCountFact, ...]:
    """Compile count provenance from exact expressions or unambiguous owners.

    No source-ID/name heuristics or presentation-token parsing. Physical values
    must equal the authoritative target path, so an erroneous planner value is
    not trusted merely because it came from a planner.
    """
    leaves = flat(physical_plan.target)
    source_bytes = blueprint.source.encode("utf-8")
    source_sha256 = digest(source_bytes)
    target_sha256 = digest(physical_plan.target)
    variable_kinds = {variable.key: variable.kind for variable in blueprint.contract.variables}
    result = []
    identities = set()
    for region in blueprint.regions:
        if not _scalar_package_count_owner(blueprint, region):
            continue
        _verify_region(source_bytes, region)
        identity = (region.key, region.occurrence)
        if identity in identities:
            raise ValueError("duplicate package-count occurrence identity")
        identities.add(identity)
        quantity, paths, basis = _count_authority(
            blueprint, physical_plan, region, leaves, variable_kinds
        )
        result.append(
            PackageCountFact(
                region.key,
                region.occurrence,
                region.start,
                region.end,
                digest(region.source.encode("utf-8")),
                source_sha256,
                target_sha256,
                quantity,
                paths,
                basis,
            )
        )
    return tuple(result)


def project_package_aliases_with_context(
    blueprint: SamplingBlueprint,
    physical_plan: PhysicalRenderPlan,
    facts: Sequence[PackageCountFact],
    *,
    mode: PackageAliasMode,
    rng: Random,
) -> dict[str, str | list[str]]:
    """Project nouns from canonical occurrence facts; preserve every count."""
    leaves = flat(physical_plan.target)
    source_bytes = blueprint.source.encode("utf-8")
    source_sha256, target_sha256 = digest(source_bytes), digest(physical_plan.target)
    regions = sorted(blueprint.regions, key=lambda region: region.start)
    region_by_id = {(region.key, region.occurrence): region for region in regions}
    variable_kinds = {variable.key: variable.kind for variable in blueprint.contract.variables}
    expected_ids = {
        (region.key, region.occurrence)
        for region in regions
        if _scalar_package_count_owner(blueprint, region)
    }
    facts_by_id = {}
    for fact in facts:
        if not isinstance(fact, PackageCountFact) or any(
            type(value) is not int for value in (fact.occurrence, fact.byte_start, fact.byte_end)
        ):
            raise ValueError("package-count context must use typed integer occurrence/span fields")
        identity = (fact.owner, fact.occurrence)
        if identity in facts_by_id:
            raise ValueError("duplicate package-count context identity")
        if identity not in region_by_id:
            raise ValueError("count context refers to an inactive owner")
        region = region_by_id[identity]
        if (
            fact.source_sha256 != source_sha256
            or fact.target_sha256 != target_sha256
            or (fact.byte_start, fact.byte_end) != (region.start, region.end)
            or fact.source_text_sha256 != digest(region.source.encode("utf-8"))
        ):
            raise ValueError("count context source/target/occurrence provenance is stale")
        _verify_region(source_bytes, region)
        authoritative_quantity, authoritative_paths, authoritative_basis = _count_authority(
            blueprint, physical_plan, region, leaves, variable_kinds
        )
        if (fact.quantity, fact.target_paths, fact.basis) != (
            authoritative_quantity,
            authoritative_paths,
            authoritative_basis,
        ):
            raise ValueError("count context differs from its compiled authoritative expression")
        if not fact.target_paths or any(
            not _QUANTITY.fullmatch(path)
            or _positive_integer(leaves[path], "target count") != fact.quantity
            for path in fact.target_paths
        ):
            raise ValueError("count context differs from its authoritative target quantity")
        _positive_integer(fact.quantity, "context count")
        facts_by_id[identity] = fact
    if facts_by_id.keys() != expected_ids:
        raise ValueError("package-count context coverage differs from active count owners")
    groups = {}
    for index, region in enumerate(regions):
        if (
            region.curated_key is not None
            or not region.target_paths
            or not all(_CATEGORY.fullmatch(path) for path in region.target_paths)
        ):
            continue
        _verify_region(source_bytes, region)
        # Explicit template surface recipes already own the noun's
        # presentation. Optional alias sampling is not a second owner and must
        # neither replace their value nor create a conflicting replacement.
        if region.key in blueprint.ownership_data.get("surfaces", {}):
            continue
        categories = {leaves[path] for path in region.target_paths}
        if len(categories) != 1:
            raise ValueError("shared package noun has conflicting categories")
        counts = []
        for neighbour in regions[max(0, index - 1) : index] + regions[index + 1 : index + 2]:
            gap = (
                source_bytes[neighbour.end : region.start]
                if neighbour.end <= region.start
                else source_bytes[region.end : neighbour.start]
            )
            if not re.fullmatch(rb"[ \t]*", gap):
                continue
            if any(_QUANTITY.fullmatch(path) for path in neighbour.target_paths):
                identity = (neighbour.key, neighbour.occurrence)
                if identity not in facts_by_id:
                    raise ValueError(
                        "adjacent package-count occurrence lacks authoritative context"
                    )
                counts.append(facts_by_id[identity].quantity)
        if counts:
            if len(set(counts)) != 1:
                raise ValueError("package noun has conflicting adjacent count contexts")
            quantity = counts[0]
        else:
            quantities = {
                _positive_integer(
                    leaves[path.rsplit(".", 1)[0] + ".packageQuantity"], "declared count"
                )
                for path in region.target_paths
            }
            if len(quantities) != 1:
                raise ValueError("shared package noun has conflicting declared counts")
            quantity = quantities.pop()
        groups.setdefault(region.key, []).append((region, categories.pop(), quantity))
    surfaces = deepcopy(dict(physical_plan.surface_values))
    for key, owners in groups.items():
        originals = blueprint.historical_bindings[key]["occurrences"]
        values = surfaces.get(key, [occurrence["source_text"] for occurrence in originals])
        if not isinstance(values, list) or len(values) != len(originals):
            raise ValueError("package alias occurrence count differs from its binding")
        selected = {}
        for region, category, quantity in owners:
            identity = (category, quantity)
            if mode == "source" or identity not in selected:
                selected[identity] = select_package_alias(
                    category, quantity, region.source, mode=mode, rng=rng
                )
            leading = region.source[: len(region.source) - len(region.source.lstrip())]
            trailing = region.source[len(region.source.rstrip()) :]
            values[region.occurrence] = leading + selected[identity] + trailing
        surfaces[key] = values
    return surfaces
