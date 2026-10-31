"""Compose disjoint package groups with one authoritative physical ledger."""

from __future__ import annotations

import copy
import random
from decimal import Decimal as D
from itertools import pairwise

from document_ocr.synthesis.curated import digest
from document_ocr.synthesis.curated_scenarios import CandidateIncompatibility
from document_ocr.synthesis.declared_package_rendering import public_rows
from document_ocr.synthesis.declared_packages import CountedPackage, DeclaredPackages, sample_groups

MEASURES = (("netWeight", "netKg"), ("grossWeight", "grossKg"), ("volume", "volumeM3"))
FACTORS = {"kilogram": D(1), "metric_tonne": D(1000), "pound": D("0.45359237"), "cubic_metre": D(1)}


def _unit(draw: dict, *, shared=False) -> CountedPackage:
    facts = draw["physical"]
    return CountedPackage(
        draw["internalCategory"],
        D(facts["unitNetKg"]),
        D(facts["unitGrossKg"]),
        D(facts["unitVolumeM3"]),
        1 if shared else draw["provenance"]["numericEnvelope"]["quantityMultiple"],
    )


def _partition(draw: dict, count: int, capacities, seed, key) -> DeclaredPackages:
    counts = [r["packageQuantity"] for r in draw["allocations"]] or [
        draw["package"]["packageQuantity"]
    ]
    total = sum(counts)
    if total < count:
        raise CandidateIncompatibility("not enough counted units for requested disjoint groups")
    rng = random.Random(digest([seed, key, "declared-partition"]))
    cuts = [0, *sorted(rng.sample(range(1, total), count - 1)), total]
    rows, offset = [], 0
    for quantity in counts:
        end = offset + quantity
        rows.append(tuple(max(0, min(end, b) - max(offset, a)) for a, b in pairwise(cuts)))
        offset = end
    return DeclaredPackages(
        tuple(_unit(draw, shared=True) for _ in range(count)), tuple(rows), capacities
    )


def _resize(draw: dict, group: CountedPackage, counts: tuple[int, ...]) -> dict:
    result = copy.deepcopy(draw)
    quantity = sum(counts)
    result["package"]["packageQuantity"] = quantity
    facts = group.facts(quantity)
    for field, key in MEASURES:
        result["physical"][field]["value"] = facts[key]
    if result["allocations"]:
        result["allocations"] = [
            {"row": i, "packageQuantity": n, **group.facts(n)} for i, n in enumerate(counts)
        ]
    return result


def composition_capacities(draw, contract, count):
    rows = copy.deepcopy(draw["provenance"]["rowCapacities"])
    policy = contract.get("uncontainerizedBudget")
    if policy is not None:
        if policy != "per_group" or draw["equipment"] or len(rows) != 1:
            raise ValueError("per-group planning budget requires uncontainerized shipment")
        # No carrying equipment/capacity is printed. The single-row budget is
        # a sampling bound, not a vessel rating. Reserve that same explicit
        # budget for each disjoint group; never enlarge actual hardware limits.
        rows = [{key: str(D(value) * count) for key, value in rows[0].items()}]
    return tuple((D(c["payloadKg"]), D(c["volumeM3"])) for c in rows), rows


def component_minima(certificate, load_count):
    shipment, loads = 1, [1] * load_count
    for region in certificate["regions"]:
        if region["role"] not in {
            "component_package",
            "component_measurement",
            "subset_component_packing",
        }:
            continue
        count = region["componentCount"]
        if type(count) is not int or count < 1:
            raise ValueError("component count must be a positive integer")
        scope = region.get("scope", {"kind": "shipment"})
        if scope["kind"] == "shipment":
            shipment = max(shipment, count)
        elif scope["kind"] == "container" and 0 <= scope["index"] < load_count:
            loads[scope["index"]] = max(loads[scope["index"]], count)
        else:
            raise ValueError("component minimum has unsupported source scope")
    return shipment, loads


def expand(campaign, result, authority, contract, request, *, attempt, equipment_domain):
    """Return a new plan; never alter a previously prepared single-row scenario."""
    blueprint, original, sample_id, target = result
    old_draw = original.provenance["independentDraw"]
    capacity_rows = old_draw["provenance"]["rowCapacities"]
    count = authority["count"]
    capacities, composed_capacity_rows = composition_capacities(old_draw, contract, count)
    key = f"{blueprint.document_id}:{original.variant}:{attempt}"
    if authority["mode"] == "shared_units":
        ledger = _partition(old_draw, count, capacities, campaign.config["seed"], key)
        private = copy.deepcopy(original.provenance["freshPrivatePacking"])
        group_draws = []
    elif authority["mode"] == "independent_groups":
        children = authority["groups"]
        draws = []
        for i, child in enumerate(children):
            frozen = child["request"]
            draw = campaign.sampler.draw(
                request=frozen,
                seed=campaign.config["seed"],
                draw_key=f"{key}:group:{i}",
                container_count=len(old_draw["equipment"]),
                selected_category=frozen["selectedCategory"],
                counted_unit_context=frozen["countedUnitContext"],
                quantity_multiple=1,
                equipment_slots=old_draw["equipment"],
                equipment_domain=equipment_domain,
                row_capacities=capacity_rows,
                unit_profile_precision=frozen["unitProfilePrecision"],
            )
            draws.append(draw)
        units = tuple(_unit(draw) for draw in draws)
        minimum, load_minima = component_minima(
            campaign.numeric.certificate(blueprint.document_id), len(capacities)
        )
        ledger = sample_groups(
            units,
            capacities,
            seed=campaign.config["seed"],
            key=key,
            utilization=tuple(
                D(str(value)) for value in campaign.config["declared_packages"]["utilization"]
            ),
            minimum_group_count=minimum,
            minimum_load_counts=load_minima,
            exclusive_groups=[
                i
                for i, draw in enumerate(draws)
                if old_draw["equipment"]
                and draw["provenance"]["countedUnitContext"]
                in {"non_dg_liquid_system", "tank_loading_lot"}
            ],
        )
        group_draws = [
            _resize(draw, unit, tuple(row[i] for row in ledger.matrix))
            for i, (draw, unit) in enumerate(zip(draws, units, strict=True))
        ]
        from document_ocr.synthesis.declared_package_private import build

        private = build(
            campaign,
            blueprint,
            ledger,
            group_draws,
            children,
            original.provenance["freshPrivatePacking"],
            variant=original.variant,
        )
    else:
        raise ValueError("unrecognized declared-group planning authority")
    rows = public_rows(ledger, contract["projection"])
    families = {
        campaign.package_presentations.vocabulary.entries[g.category]["family"]
        for g in ledger.groups
    }
    aggregate = next(iter(families)) if len(families) == 1 else "PACKAGE_PACKAGE"
    target = copy.deepcopy(target)
    cargo = copy.deepcopy(original.cargo)
    totals = ledger.totals()
    for destination in (target["documentPatch"], cargo):
        goods = destination["goodsItemDetails"][0]
        goods["numberAndTypeOfPackages"] = copy.deepcopy(rows)
        for field, fact in MEASURES:
            if field in goods:
                goods[field]["value"] = float(D(totals[fact]) / FACTORS[goods[field]["unit"]])
        if "splitGoodsPlacement" in goods:
            identifiers = [
                c["equipmentIdentifier"] for c in destination.get("containerInformation", [])
            ]
            goods["splitGoodsPlacement"] = ledger.project_placements(
                goods["splitGoodsPlacement"], identifiers
            )
    provenance = copy.deepcopy(original.provenance)
    provenance.pop("registryLoadProfile")
    provenance["commodityPhysicalForm"] = old_draw["profile"]["form"]
    provenance["declaredPackages"] = ledger.receipt()
    provenance["declaredPackageContract"] = copy.deepcopy(contract)
    provenance["declaredPackageAuthoritySha256"] = digest(authority)
    provenance["declaredPackageGroupDraws"] = group_draws
    provenance["explicitPhysicalFacts"] = {
        field: {"value": totals[fact], "unit": "cubic_metre" if field == "volume" else "kilogram"}
        for field, fact in MEASURES
    }
    for row, facts in zip(
        provenance["physicalRows"], ledger.row_facts() if old_draw["equipment"] else [], strict=True
    ):
        row.update(
            quantity=facts["packageQuantity"],
            shareNumerator=facts["packageQuantity"],
            shareDenominator=ledger.total,
        )
    draw = copy.deepcopy(old_draw)
    draw["package"] = {
        "packageQuantity": ledger.total,
        "packageTypeFamily": aggregate.removeprefix("PACKAGE_"),
        "packageMaterial": None,
    }
    draw["internalCategory"] = aggregate
    draw["physical"] = copy.deepcopy(provenance["explicitPhysicalFacts"])
    draw["declaredPackages"] = ledger.receipt()
    draw["allocations"] = ledger.row_facts() if old_draw["equipment"] else []
    draw["provenance"]["method"] = "independent_declared_package_groups_v1"
    draw["provenance"]["rowCapacities"] = composed_capacity_rows
    if contract.get("uncontainerizedBudget"):
        draw["provenance"]["uncontainerizedPlanningBudget"] = {
            "basis": "per_declared_group_not_hardware_capacity",
            "groupCount": count,
            "singleGroupBudget": capacity_rows,
        }
    provenance["independentDraw"] = draw
    private["publicQuantity"] = ledger.total
    if authority["mode"] == "independent_groups":
        private["countedCategory"] = aggregate
        # No unique physical unit exists for heterogeneous disjoint groups.
        private["unitNetKg"] = private["unitGrossKg"] = None
        private["netKg"], private["grossKg"] = totals["netKg"], totals["grossKg"]
    provenance["freshPrivatePacking"] = private
    scenario = original.model_copy(update={"cargo": cargo, "provenance": provenance})
    return blueprint, scenario, sample_id, target
