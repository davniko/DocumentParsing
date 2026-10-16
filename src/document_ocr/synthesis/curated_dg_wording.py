"""DG facts constrain commercial prose without replacing its lexical generation."""

from __future__ import annotations

import yaml


def proper_shipping_name(scenario) -> str:
    """Read the sampled registry identity, never a generated description alias."""
    facts = scenario.provenance.get("dgPrintedFacts")
    value = facts.get("properShippingName") if isinstance(facts, dict) else None
    declarations = [
        entry
        for goods in scenario.cargo.get("goodsItemDetails", [])
        for entry in goods.get("dangerousGoods", [])
    ]
    if (
        not isinstance(value, str)
        or not value.strip()
        or not declarations
        or {entry.get("unNumber") for entry in declarations} != {facts.get("unNumber")}
    ):
        raise ValueError("DG wording requires a sampled registry identity matching its UN facts")
    return value


def dg_wording_context(scenario, *, cargo: dict | None = None) -> str:
    """Keep product identity authoritative and printed field presence unchanged."""
    proper_shipping_name(scenario)
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
