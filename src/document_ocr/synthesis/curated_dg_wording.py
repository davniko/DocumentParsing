"""DG facts constrain commercial prose without replacing its lexical generation."""

from __future__ import annotations

import yaml


def validate_product_ownership(blueprint) -> None:
    """Reject incomplete multi-entry presentation contracts before paid generation."""
    goods = blueprint.target["documentPatch"].get("goodsItemDetails", [])
    declarations = [entry for item in goods for entry in item.get("dangerousGoods", [])]
    if len(declarations) < 2:
        return
    products = [v for v in blueprint.contract.variables if v.kind == "product"]
    if any(v.dg_indices is None for v in products):
        raise ValueError("multi-DG product regions require explicit declaration ownership")
    if {index for v in products for index in v.dg_indices} != set(range(len(declarations))):
        raise ValueError("multi-DG product ownership must cover exactly the source declarations")


def regulatory_facts(scenario) -> list[dict]:
    """Validate declaration order before any wording or numeric owner uses it."""
    facts = scenario.provenance.get("dgPrintedFacts")
    entries = facts if isinstance(facts, list) else [facts]
    declarations = [
        entry
        for goods in scenario.cargo.get("goodsItemDetails", [])
        for entry in goods.get("dangerousGoods", [])
    ]
    if (
        not declarations
        or len(entries) != len(declarations)
        or any(
            not isinstance(entry, dict)
            or not isinstance(entry.get("properShippingName"), str)
            or not entry["properShippingName"].strip()
            or entry.get("unNumber") != declaration.get("unNumber")
            for entry, declaration in zip(entries, declarations, strict=True)
        )
    ):
        raise ValueError("DG wording requires a sampled registry identity matching its UN facts")
    return entries


def proper_shipping_name(scenario, dg_index: int | None = None) -> str:
    """Read one explicitly owned identity; multi-DG callers must name its index."""
    entries = regulatory_facts(scenario)
    if dg_index is None:
        if len(entries) != 1:
            raise ValueError("multi-DG proper-shipping-name owner requires a declaration index")
        dg_index = 0
    if type(dg_index) is not int or not 0 <= dg_index < len(entries):
        raise ValueError("DG identity index is outside the sampled declarations")
    return entries[dg_index]["properShippingName"]


def dg_wording_context(scenario, *, cargo: dict | None = None) -> str:
    """Keep product identity authoritative and printed field presence unchanged."""
    regulatory_facts(scenario)
    return (
        "\nSAMPLED DG PRODUCT FACTS (compatibility context, not text to copy verbatim)\n"
        + yaml.safe_dump(
            {
                "registryIdentity": scenario.provenance["dgPrintedFacts"],
                "declarations": [
                    entry
                    for goods in scenario.cargo.get("goodsItemDetails", [])
                    for entry in goods.get("dangerousGoods", [])
                ],
            },
            sort_keys=False,
            allow_unicode=True,
        )
        + "Write new commercial wording for the sampled product, matching the source's "
        "descriptive depth and details. Retain the sampled chemical identity and compatible "
        "physical form or vehicle propulsion; do not substitute another substance or formulation. "
        "Separately owned proper shipping names, UN numbers, classes, packing groups, subsidiary "
        "hazards and flash points are rendered by the host only where the template prints them. "
        "Do not add those declarations to the product region. They constrain meaning, not wording."
        "\nDECLARED PACKAGE COUNTS (compatibility context, not additional description)\n"
        + yaml.safe_dump(
            [
                goods.get("numberAndTypeOfPackages", [])
                for goods in (scenario.cargo if cargo is None else cargo).get(
                    "goodsItemDetails", []
                )
            ],
            sort_keys=False,
        )
        + "The source is a presentation example for a different shipment. Where its pattern "
        "includes a count at the declared package level, use the supplied count and unit. "
        "Preserve source-like intrinsic capacities and specifications coherently. Shipment "
        "weight/volume totals and total captions belong to separate host fields, not generated "
        "product wording. Product state/formulation must agree with the sampled DG facts, "
        "and numeric specifications for one product must agree with each other."
    )
