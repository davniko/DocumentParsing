"""Compose owned private packing presentations for disjoint public groups.

Complete clauses remain group-specific. Bare total-measurement cells are summed;
they must never contain a list of per-group values or a fabricated average unit.
The existing numeric renderer still owns each inner/support relationship.
"""

from __future__ import annotations

from copy import deepcopy
from decimal import Decimal as D

from document_ocr.synthesis.curated_measurements import format_measure
from document_ocr.synthesis.independent_numeric import NUMERIC_REGION_PREFIX
from document_ocr.synthesis.independent_wording import META, phrase_surfaces


def _fact(ledger, group):
    matches = [
        scope["facts"][group]
        for scope in [ledger, *ledger.get("scopeLedgers", {}).values()]
        if group in scope["facts"]
    ]
    if not matches or any(value != matches[0] for value in matches):
        raise ValueError("private component fact does not have a unique scoped authority")
    return matches[0]


def combine(blueprint, certificate, ledger, children, original):
    """Project complete child ledgers into the same exact source owners."""
    result = deepcopy(original)
    result.update(
        countedCategory="PACKAGE_PACKAGE",
        publicQuantity=ledger.total,
        relation="DISJOINT_GROUPS",
        relatedCategory=None,
        relatedQuantity=None,
        unitNetKg=None,
        unitGrossKg=None,
        netKg=ledger.totals()["netKg"],
        grossKg=ledger.totals()["grossKg"],
        facts={"declaredGroupLedgers": deepcopy(children)},
        surfaces={},
        wordingFacts=[],
        wordingAppends={},
        presentationDecisions=[],
    )
    result.pop("scopeLedgers", None)
    for index, region in enumerate(certificate["regions"]):
        key = f"{NUMERIC_REGION_PREFIX}:{index}"
        active = [(i, child) for i, child in enumerate(children) if key in child["surfaces"]]
        if not active:
            raise ValueError("an owned private occurrence has no positive group membership")
        role = region["role"]
        scope = region.get("scope", {"kind": "shipment"})
        if scope["kind"] == "shared_container":
            raise ValueError("shared container facts require the explicit shared-unit contract")
        quantities = ledger.counts if scope["kind"] == "shipment" else ledger.matrix[scope["index"]]
        if role in {"private_measurement", "container_net_mass", "component_measurement"}:
            fact = "netKg" if role == "container_net_mass" else region["fact"]
            attr = {
                "netKg": "net_kg",
                "grossKg": "gross_kg",
                "declaredUntypedMassKg": "gross_kg",
                "volumeM3": "volume_m3",
            }.get(fact)
            if attr is None:
                raise ValueError("bare per-unit cell needs a reviewed multi-group presentation")
            amount = D(0)
            for i, child in active:
                count = quantities[i]
                if role == "component_measurement":
                    count = _fact(child, region["factGroup"])["quantities"][
                        region["componentIndex"]
                    ]
                amount += count * getattr(ledger.groups[i], attr)
            baseline = region["baselineKg"] if role == "container_net_mass" else region["baseline"]
            text = format_measure(region["source_text"], D(str(baseline)), amount)
        elif role == "net_statement":
            amount = sum(quantities[i] * ledger.groups[i].net_kg for i, _ in active)
            text = f"N.W. {amount.normalize():f} KG."
        else:
            terms = []
            for _, child in active:
                values = child["surfaces"][key]
                if len(values) != 1:
                    raise ValueError("complete group owner requires one literal surface")
                if values[0]:
                    terms.append(values[0])
            # Whole clauses retain their own counts/units. A component-package
            # cell instead lists the additive outer portions of that Marks lot.
            separator = " + " if role == "component_package" else "\n"
            text = separator.join(terms)
        result["surfaces"][key] = [text]
    for child in children:
        result["wordingFacts"].extend(child["wordingFacts"])
        for key, value in child["wordingAppends"].items():
            previous = result["wordingAppends"].get(key)
            result["wordingAppends"][key] = value if previous is None else previous + " " + value
        result["presentationDecisions"].extend(child["presentationDecisions"])
    additional = phrase_surfaces(blueprint) if META in blueprint.ownership_data else {}
    if set(additional) & set(result["surfaces"]):
        raise ValueError("wording phrase owner overlaps complete private group owner")
    result["surfaces"].update(additional)
    return result


def build(campaign, blueprint, ledger, draws, authorities, original, *, variant):
    certificate = campaign.numeric.certificate(blueprint.document_id)
    if not certificate["regions"] and not certificate["wordingOwners"]:
        # No packing relationship was printed; there is nothing to reinterpret
        # or expand. Source-only linguistic phrases retain their own authority.
        return deepcopy(original)
    children = []
    noun = campaign.package_presentations.noun_renderer(
        blueprint.document_id, campaign.config["seed"], variant
    )
    for i, (draw, authority) in enumerate(zip(draws, authorities, strict=True)):
        child = campaign.numeric.build_ledger(
            blueprint,
            draw,
            authority["plan"],
            f"{campaign.config['seed']}:declared-group:{i}",
            variant,
            noun_renderer=noun,
            allow_empty_group_scopes=True,
        )
        children.append(child)
    return combine(blueprint, certificate, ledger, children, original)
