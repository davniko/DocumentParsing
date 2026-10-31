"""Compile exact direct-count declaration owners without guessing cargo scope."""

from __future__ import annotations

import re
from itertools import pairwise

PREFIX = "documentPatch.goodsItemDetails[0].numberAndTypeOfPackages[0]"
ALLOCATION = re.compile(
    r"documentPatch\.goodsItemDetails\[0\]\.splitGoodsPlacement\[(\d+)\]\.packageQuantity"
)


def quantity_scope(blueprint, region):
    paths = {
        binding.path
        for binding in blueprint.contract.targets
        if region.curated_key is not None and binding.expression == "{" + region.curated_key + "}"
    }
    if region.curated_key is None and region.kind == "integer":
        binding = blueprint.historical_bindings.get(region.key, {})
        if binding.get("group_kind") == "package" and binding.get("value_kind") == "integer":
            paths = set(region.target_paths)
    if PREFIX + ".packageQuantity" in paths:
        return "shipment"
    indices = {int(m[1]) for path in paths if (m := ALLOCATION.fullmatch(path))}
    return next(iter(indices)) if len(indices) == 1 else None


def compile_declarations(blueprint):
    """Prefer complete count/noun phrases; never consume intervening fields.

    Quantity-only columns are not automatically expanded here. Their width and
    source presentation require a reviewed contract rather than text adjacency.
    """
    raw = blueprint.source.encode()
    quantities = [
        (region, scope)
        for region in blueprint.regions
        if (scope := quantity_scope(blueprint, region)) is not None
    ]
    found = []
    for noun in blueprint.regions:
        if noun.target_paths != (PREFIX + ".typeCategory",):
            continue
        choices = []
        for number, scope in quantities:
            start, end = min(number.start, noun.start), max(number.end, noun.end)
            owners = sorted(
                [r for r in blueprint.regions if r.start < end and start < r.end],
                key=lambda r: r.start,
            )
            if any(r is not noun and quantity_scope(blueprint, r) != scope for r in owners):
                continue
            gaps = [raw[a.end : b.start] for a, b in pairwise(owners)]
            if any(not re.fullmatch(rb"[\s()xX]*", gap) or gap.count(b"\n") > 1 for gap in gaps):
                continue
            choices.append((start, end, scope, owners))
        if not choices:
            continue
        # A serial-range endpoint on the previous line is not another repetition
        # inside this declaration. Prefer complete same-line count/noun ownership;
        # still include parenthesized word/digit repetitions on that line.
        start, end, scope, owners = max(
            choices,
            key=lambda value: (b"\n" not in raw[value[0] : value[1]], value[1] - value[0]),
        )
        found.append(
            {
                "owners": [[r.key, r.occurrence] for r in owners],
                "start": start,
                "end": end,
                "source": raw[start:end].decode(),
                "scope": scope,
            }
        )
    local_phrases = [r for r in found if "\n" not in r["source"]]
    # PACKAGES\n(TWENTY PACKAGES) contains two noun occurrences but only the
    # latter owns TWENTY. Do not make the preceding noun steal that count.
    found = [
        r
        for r in found
        if "\n" not in r["source"]
        or not any(
            r["start"] < other["end"] and other["start"] < r["end"] for other in local_phrases
        )
    ]
    used = set()
    for row in found:
        owners = {tuple(owner) for owner in row["owners"]}
        if used & owners:
            raise ValueError("ambiguous repeated declaration ownership: " + blueprint.document_id)
        used.update(owners)
    shipment = [r for r in found if r["scope"] == "shipment"]
    if shipment:
        return "shipment", shipment
    local = [r for r in found if type(r["scope"]) is int]
    containers = len(blueprint.target["documentPatch"].get("containerInformation", []))
    if not local or {r["scope"] for r in local} != set(range(containers)):
        raise ValueError("no complete owned declaration projection: " + blueprint.document_id)
    return "container_occurrences", local
