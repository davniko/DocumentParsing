"""Exact source-certified contained counts, separate from shipment accounting.

A reviewed template chooses the declared unit. Contained packing can then scale
with it without becoming a public package row or inventing a rounded count.
"""

from __future__ import annotations

import re
from fractions import Fraction
from math import lcm
from typing import TYPE_CHECKING

from document_ocr.synthesis.curated import _number_style, flat
from document_ocr.synthesis.template_compiler.descendant import _number_to_words

if TYPE_CHECKING:
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
        if not re.match(r"\s+" + re.escape(unit) + r"(?!\w)", suffix):
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
