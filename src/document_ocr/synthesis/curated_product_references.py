"""Explicit source-only short commodity references in joint goods wording.

A detached Marks name is not a second extraction description. Its reviewed
owner links to one existing goods item; the model writes that short reference
together with the complete description, rather than a host truncating prose.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING

from document_ocr.synthesis.curated_wording import WordingField, WordingRequest, wording_api_keys

if TYPE_CHECKING:
    from document_ocr.synthesis.curated_templates import SamplingBlueprint


@dataclass(frozen=True, slots=True)
class ProductReference:
    key: str
    goods_index: int
    example: str
    product_keys: tuple[str, ...]


def product_references(blueprint: SamplingBlueprint) -> tuple[ProductReference, ...]:
    """Require one unambiguous goods link and complete non-target text ownership."""
    declarations = blueprint.ownership_data.get("surfaces", {})
    selected = [
        (key, recipe) for key, recipe in declarations.items() if "product_reference" in recipe
    ]
    if not selected:
        return ()
    goods = blueprint.target["documentPatch"].get("goodsItemDetails", [])
    variables = {variable.key: variable for variable in blueprint.contract.variables}
    references = []
    for key, recipe in selected:
        link = recipe["product_reference"]
        if (
            set(recipe) != {"product_reference"}
            or not isinstance(link, dict)
            or set(link) != {"goods_index"}
            or type(link["goods_index"]) is not int
            or not 0 <= link["goods_index"] < len(goods)
        ):
            raise ValueError(f"product reference requires one existing goods_index: {key}")
        index = link["goods_index"]
        path = f"documentPatch.goodsItemDetails[{index}].description"
        targets = [binding for binding in blueprint.contract.targets if binding.path == path]
        if len(targets) != 1 or not isinstance(goods[index].get("description"), str):
            raise ValueError(f"product reference has no unique current goods description: {key}")
        product_keys = tuple(
            variable.key
            for variable in blueprint.contract.variables
            if variable.kind == "product" and "{" + variable.key + "}" in targets[0].expression
        )
        if not product_keys:
            raise ValueError(f"product reference needs its goods description lexical owner: {key}")
        regions = [region for region in blueprint.regions if region.key == key]
        historical = blueprint.historical_bindings.get(key)
        if (
            key in variables
            or not regions
            or historical is None
            or historical.get("value_kind") != "other_text"
            or historical.get("target_paths")
            or any(region.curated_key is not None or region.target_paths for region in regions)
            or len(regions) != len(historical["occurrences"])
        ):
            raise ValueError(f"product reference requires a complete source-only text owner: {key}")
        examples = {region.source for region in regions}
        if len(examples) != 1:
            raise ValueError(f"product reference has ambiguous source-name occurrences: {key}")
        example = examples.pop()
        validate_reference_text(example, key=key)
        references.append(ProductReference(key, index, example, product_keys))
    return tuple(references)


def validate_reference_text(value: str, *, key: str) -> None:
    """A short detached reference is one nonempty physical line, never a block."""
    if not isinstance(value, str) or not value.strip() or "\n" in value or "\r" in value:
        raise ValueError(f"product reference requires nonempty single-line commodity text: {key}")


def product_reference_fields(
    blueprint: SamplingBlueprint,
    *,
    description_fields: Sequence[WordingField],
    host_products: dict[str, str],
) -> tuple[WordingField, ...]:
    references = product_references(blueprint)
    if not references:
        return ()
    aliases = (
        wording_api_keys(WordingRequest("source-only-references", "", tuple(description_fields)))
        if description_fields
        else {}
    )
    fields = []
    for reference in references:
        if any(key not in host_products and key not in aliases for key in reference.product_keys):
            raise ValueError(
                f"product reference lacks its requested main product field: {reference.key}"
            )
        description = " + ".join(
            repr(host_products[key]) if key in host_products else aliases[key]
            for key in reference.product_keys
        )
        fields.append(
            WordingField(
                reference.key,
                "goods reference",
                reference.example,
                f"Short commodity-name reference for goods item {reference.goods_index + 1}, "
                f"the SAME current product as main description components {description}. "
                "Write these jointly and keep their commodity identities consistent. "
                "The old reference illustrates only the short naming scope, not an identity "
                "to preserve. Output a short name on one line, not a copy of the complete "
                "description/specification block. No packaging, package material, quantities, "
                "fill/capacity, shipment weights, HS codes or regulatory declarations. "
                "An intrinsic product material, brand or model may be part of its identity. "
                "This source-only reference has no extraction target.",
            )
        )
    return tuple(fields)


def product_reference_baselines(blueprint: SamplingBlueprint) -> dict[str, str]:
    """Explicit source examples for the CPU ownership preflight, not generation."""
    return {reference.key: reference.example for reference in product_references(blueprint)}


def product_reference_surface(
    blueprint: SamplingBlueprint, key: str, lexical_values: dict[str, str] | None
) -> str:
    if key not in {reference.key for reference in product_references(blueprint)}:
        raise ValueError(f"unreviewed source-only product reference: {key}")
    if lexical_values is None or key not in lexical_values:
        raise ValueError(f"missing generated source-only product reference: {key}")
    value = lexical_values[key]
    validate_reference_text(value, key=key)
    return value
