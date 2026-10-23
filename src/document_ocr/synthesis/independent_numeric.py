"""Reviewed, source-independent private numeric rendering.

The public target remains owned by the ordinary synthesis pipeline.  This
module consumes an explicitly supplied, reviewed certificate inventory and
owns only complete private OCR regions: repeated counts, package relations,
unit masses, ranges, capacities and partial/container-scoped presentations.
Missing authority always fails closed.
"""

from __future__ import annotations

import copy
import hashlib
import json
import random
import re
from collections.abc import Mapping, Sequence
from dataclasses import replace
from decimal import ROUND_FLOOR, ROUND_HALF_UP, Decimal
from itertools import pairwise
from pathlib import Path
from typing import Any, cast

from pydantic import BaseModel, ConfigDict, Field

from document_ocr.synthesis.curated import assign, flat
from document_ocr.synthesis.curated_descriptions import validate_description_regions
from document_ocr.synthesis.curated_measurements import format_measure
from document_ocr.synthesis.curated_scenarios import CandidateIncompatibility, FilePin
from document_ocr.synthesis.curated_templates import RenderRegion, SamplingBlueprint
from document_ocr.synthesis.independent_cargo_presentation import CargoPresentation
from document_ocr.synthesis.independent_envelopes import PACKAGE_COUNTING_CONTEXTS, package_envelope
from document_ocr.synthesis.package_aliases import PACKAGE_ALIASES

NUMERIC_LEDGER_OWNER = "independentNumericLedger"
NUMERIC_REGION_PREFIX = "held-numeric"
PREPARED_SUPPORT_KEY = "preparedDiscreteArticleSupport"
PRIVATE_PARTITION_KEY = "sharedPrivateBatchPartition"
POUNDS_TO_KG = Decimal("0.45359237")

SUPPORTED_REGION_ROLES = frozenset(
    {
        "three_level_chain",
        "packing_clause",
        "unit_net_prefix",
        "unit_mass_suffix",
        "support_mass_breakdown",
        "component_mass_pair",
        "count_net_statement",
        "net_statement",
        "unit_mass_pair",
        "inclusive_range",
        "receptacle_statement",
        "container_net_mass",
        "on_support_with_unit_net",
        "private_ranges",
        "private_measurement",
        "shared_container_range",
        "dual_unit_net_statement",
        "support_statement",
        "mixed_package_equation",
        "component_package",
        "component_measurement",
        "mixed_package_ranges",
    }
)
_COMPOSITION_ROLES = frozenset(
    {
        "three_level_chain",
        "packing_clause",
        "receptacle_statement",
        "support_mass_breakdown",
        "component_mass_pair",
        "on_support_with_unit_net",
        "support_statement",
    }
)
_RELATIONS = frozenset({"NONE", "CONTAINS", "SUPPORTS", "ON_SUPPORT"})


class SourceReadinessHold(BaseModel):
    """A separately reviewed nonnumeric source blocker, not a numeric failure."""

    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)
    category: str = Field(pattern=r"^[a-z][a-z0-9_]*$")
    reason: str = Field(min_length=1, pattern=r"\S")
    evidence: FilePin


def _digest(value: Any) -> str:
    raw = value if isinstance(value, bytes) else json.dumps(value, sort_keys=True).encode()
    return hashlib.sha256(raw).hexdigest()


def _decimal(value: Any) -> Decimal:
    result = Decimal(str(value))
    if not result.is_finite() or result < 0:
        raise ValueError("private numeric facts must be finite nonnegative decimals")
    return result


def _number(value: Any) -> str:
    return format(_decimal(value).normalize(), "f")


def _noun(category: str, quantity: int) -> str:
    if category not in PACKAGE_ALIASES or type(quantity) is not int or quantity < 1:
        raise ValueError("known package category and positive integer count required")
    return PACKAGE_ALIASES[category].variants(quantity)[0]


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def project_handling_targets(
    blueprint: SamplingBlueprint, target: dict[str, Any], ledger: Mapping[str, Any]
) -> tuple[dict[str, Any], list[dict[str, str]]]:
    """Project explicitly certified complete handling text, never numeric targets."""
    certificate = blueprint.ownership_data[NUMERIC_LEDGER_OWNER]["certificate"]
    changes = []
    projected = target
    seen = set()
    for index, declaration in enumerate(certificate["regions"]):
        path = declaration.get("handlingTargetPath")
        if path is None:
            continue
        region = next(
            (r for r in blueprint.regions if r.key == f"{NUMERIC_REGION_PREFIX}:{index}"), None
        )
        if (
            region is None
            or path in seen
            or path not in region.target_paths
            or path not in flat(target)
        ):
            raise ValueError("handling projection requires one existing exact target owner")
        seen.add(path)
        values = ledger["surfaces"][region.key]
        if len(values) != 1:
            raise ValueError("complete handling projection requires one rendered occurrence")
        value = " ".join((values[0] + declaration.get("handlingTargetSuffix", "")).upper().split())
        before = flat(target)[path]
        if not isinstance(before, str) or not value:
            raise ValueError("handling projection requires nonempty string values")
        if projected is target:
            projected = copy.deepcopy(target)
        assign(projected, path, value)
        changes.append({"path": path, "before": before, "after": value})
    return projected, changes


def _has_composition(regions: Sequence[Mapping[str, Any]]) -> bool:
    return any(
        region["role"] in _COMPOSITION_ROLES
        or (region["role"] == "private_ranges" and region["countRole"] == "related")
        for region in regions
    )


def support_budget(
    category: str,
    budget_kg: Decimal,
    maximum_count: int,
    rng: random.Random,
    *,
    fixed_count: int | None = None,
) -> dict[str, Any]:
    """Draw support units from an independent tare, never a source ratio."""

    authority = package_envelope(category)
    tare = Decimal(authority.tare_kg)
    if tare <= 0:
        raise ValueError("support category needs a positive independent unit-tare envelope")
    feasible = min(maximum_count, int(budget_kg // tare))
    count = fixed_count
    if count is None:
        if feasible < 1:
            raise CandidateIncompatibility(
                "private support tare exceeds sampled packing headroom: "
                f"budget={_number(budget_kg)}kg; one {category}={_number(tare)}kg"
            )
        count = rng.randint(1, feasible)
    elif count < 1 or count > feasible:
        raise CandidateIncompatibility(
            "counted support tare exceeds sampled packing headroom: "
            f"{count}*{_number(tare)}kg > {_number(budget_kg)}kg"
        )
    return {
        "category": category,
        "quantity": count,
        "unitTareKg": _number(tare),
        "totalTareKg": _number(count * tare),
        "packingBudgetKg": _number(budget_kg),
        "authority": {**authority.receipt(), "unitTareKg": _number(tare)},
    }


def replace_complete_regions(
    blueprint: SamplingBlueprint, declarations: Sequence[Mapping[str, Any]]
) -> tuple[SamplingBlueprint, list[dict[str, Any]]]:
    """Replace only exactly certified complete regions and transfer ownership."""

    raw = blueprint.source.encode()
    regions = list(blueprint.regions)
    bindings = copy.deepcopy(dict(blueprint.historical_bindings))
    ownership = copy.deepcopy(dict(blueprint.ownership_data))
    transfers: list[dict[str, Any]] = []
    spans: list[tuple[int, int]] = []
    for index, declaration in enumerate(declarations):
        start = declaration.get("byte_start")
        end = declaration.get("byte_end")
        quote = declaration.get("source_text")
        if type(start) is not int or type(end) is not int or not isinstance(quote, str):
            raise ValueError("complete numeric region needs byte offsets and source text")
        if end <= start or raw[start:end] != quote.encode():
            raise ValueError("complete numeric region differs from certified source bytes")
        if any(start < other_end and other_start < end for other_start, other_end in spans):
            raise ValueError("complete numeric certificates overlap each other")
        spans.append((start, end))
        overlap = [region for region in regions if region.start < end and start < region.end]
        if any(region.start < start or region.end > end for region in overlap):
            raise ValueError("complete numeric certificate partially intersects an owner")
        if any(region.kind == "product" for region in overlap):
            raise ValueError("complete numeric certificate may not swallow a product owner")
        paths = tuple(sorted({path for region in overlap for path in region.target_paths}))
        if any(".dangerousGoods" in path or ".hsCodes" in path for path in paths):
            raise ValueError("packing certificate would swallow regulated/HS extraction fields")
        handling_paths = [path for path in paths if ".handlingInstructions[" in path]
        if handling_paths and handling_paths != [declaration.get("handlingTargetPath")]:
            raise ValueError("complete numeric handling owner requires explicit target projection")
        if declaration.get("handlingTargetPath") is not None and not handling_paths:
            raise ValueError("handling projection must retain its actual source target owner")
        key = f"{NUMERIC_REGION_PREFIX}:{index}"
        removed = {(region.key, region.occurrence) for region in overlap}
        regions = [region for region in regions if (region.key, region.occurrence) not in removed]
        active = {region.key for region in regions}
        retired: list[dict[str, Any]] = []
        for old_key in {region.key for region in overlap}:
            if old_key in bindings:
                if old_key not in active:
                    del bindings[old_key]
                else:
                    occurrences = bindings[old_key]["occurrences"]
                    kept = [
                        old_index
                        for old_index in range(len(occurrences))
                        if (old_key, old_index) not in removed
                    ]
                    occurrence_map = {old: new for new, old in enumerate(kept)}
                    bindings[old_key]["occurrences"] = [occurrences[old] for old in kept]
                    regions = [
                        replace(region, occurrence=occurrence_map[region.occurrence])
                        if region.key == old_key
                        else region
                        for region in regions
                    ]
            if old_key in active:
                continue
            for stage in (
                "surfaces",
                "delete",
                "constants",
                "identifier_privacy",
                "measure_owners",
            ):
                stage_values = ownership.get(stage, {})
                if old_key in stage_values:
                    retired.append(
                        {"owner": old_key, "stage": stage, "value": stage_values.pop(old_key)}
                    )
            omissions = ownership.get("omit", [])
            retired.extend(
                {"owner": old_key, "stage": "omit", "value": item}
                for item in omissions
                if item["key"] == old_key
            )
            if omissions:
                ownership["omit"] = [item for item in omissions if item["key"] != old_key]
        bindings[key] = {
            "logical_key": key,
            "value_kind": "other_text",
            "target_paths": list(paths),
            "occurrences": [{"byte_start": start, "byte_end": end, "source_text": quote}],
        }
        allow_empty = (
            declaration["role"] == "private_ranges"
            and declaration.get("countRole") == "related"
            and declaration.get("optionalWithoutRelatedUnit") is True
        )
        regions.append(
            RenderRegion(
                key,
                0,
                start,
                end,
                quote,
                paths,
                "auxiliary",
                allow_empty=allow_empty,
            )
        )
        transfers.append(
            {
                "owner": key,
                "role": declaration["role"],
                "targetPaths": list(paths),
                "displacedOwners": [
                    {"key": region.key, "occurrence": region.occurrence} for region in overlap
                ],
                "retiredMetadata": retired,
            }
        )
    result = replace(
        blueprint,
        historical_bindings=bindings,
        regions=tuple(sorted(regions, key=lambda region: (region.start, region.end))),
        ownership_data=ownership,
    )
    validate_description_regions(result.description_blocks, cast(Any, result.regions))
    return result, transfers


def _apply_certificate(
    blueprint: SamplingBlueprint, certificate: Mapping[str, Any]
) -> SamplingBlueprint:
    if _digest(blueprint.source.encode()) != certificate["sourceSha256"]:
        raise ValueError("numeric source certificate hash changed")
    if _digest(blueprint.contract.model_dump(mode="json")) != certificate["sourceContractSha256"]:
        raise ValueError("numeric source contract pin changed")
    if _digest(blueprint.target) != certificate["sourcePublicTargetSha256"]:
        raise ValueError("numeric source public-target pin changed")
    if certificate["reviewStatus"] != "reviewed":
        raise ValueError("numeric source certificate was not reviewed")
    original_target = copy.deepcopy(blueprint.target)
    for owner in certificate.get("wordingOwners", []):
        matches = [
            region
            for region in blueprint.regions
            if region.kind == "product"
            and region.curated_key == owner["key"]
            and region.start == owner["byte_start"]
            and region.end == owner["byte_end"]
            and region.source == owner["source_text"]
        ]
        if len(matches) != 1:
            raise ValueError("complete current product owner differs from reviewed pin")
    result, transfers = replace_complete_regions(blueprint, certificate["regions"])
    ownership = copy.deepcopy(dict(result.ownership_data))
    reconciled: list[dict[str, Any]] = []
    regions: list[RenderRegion] = []
    complete_products = {owner["key"] for owner in certificate.get("wordingOwners", [])}
    for region in result.regions:
        if region.kind != "product" or region.curated_key not in complete_products:
            regions.append(region)
            continue
        retained = tuple(path for path in region.target_paths if path.endswith(".description"))
        removed = set(region.target_paths) - set(retained)
        for path in removed:
            if not any(
                other is not region and path in other.target_paths for other in result.regions
            ):
                raise ValueError("product-only reconciliation would lose a public owner")
        if removed:
            reconciled.append({"productOwner": region.key, "removedPublicRoles": sorted(removed)})
        regions.append(replace(region, target_paths=retained))
    result = replace(result, regions=tuple(regions))
    remaining = {
        key
        for key, recipe in ownership.get("surfaces", {}).items()
        if "contained_quantity" in recipe
    }
    if remaining:
        raise ValueError("source-ratio private count recipes remain: " + str(sorted(remaining)))
    ownership[NUMERIC_LEDGER_OWNER] = {
        "certificate": copy.deepcopy(certificate),
        "transfers": transfers,
        "productPublicRoleReconciliation": reconciled,
    }
    ownership["freshPrivatePacking"] = {
        "sourceSha256": certificate["sourceSha256"],
        "reviewStatus": "reviewed",
        "regions": [],
        "holdReasons": [],
        "supersededBy": NUMERIC_LEDGER_OWNER,
    }
    result = replace(result, ownership_data=ownership)
    if result.target != original_target:
        raise AssertionError("numeric ownership changed public source targets")
    return result


def _prepare_draw(
    draw: Mapping[str, Any],
    plan: Mapping[str, Any],
    request: Mapping[str, Any],
    certificate: Mapping[str, Any],
) -> dict[str, Any]:
    """Compose explicit support mass or declare a mass-only bulk partition."""

    provenance = draw.get("provenance")
    if not isinstance(provenance, Mapping):
        raise ValueError("numeric draw requires provenance")
    _require(
        PREPARED_SUPPORT_KEY not in provenance and PRIVATE_PARTITION_KEY not in provenance,
        "composition draw was already prepared",
    )
    counted = package_envelope(
        draw["internalCategory"], counted_unit_context=request["countedUnitContext"]
    )
    support_needed = (
        plan["privatePackingRelation"] == "ON_SUPPORT"
        and counted.counted_unit_context == "discrete_article"
        and counted.tare_kg == 0
        and "composition" not in provenance.get("numericEnvelope", {})
    )
    bulk_batches = counted.counted_unit_context == "bulk_loading_lot" and any(
        region.get("scope", {}).get("kind") == "shared_container"
        and region.get("fact") == "partitionNetKg"
        for region in certificate["regions"]
    )
    result = copy.deepcopy(dict(draw))
    if not support_needed and not bulk_batches:
        return result
    if bulk_batches:
        result["provenance"][PRIVATE_PARTITION_KEY] = {
            "mode": "positive_mass_kg",
            "countedUnitContext": counted.counted_unit_context,
            "basis": "Source batch fields declare masses, not per-batch package counts.",
            "publicPackageCountsInferred": False,
        }
    if support_needed:
        related = plan["relatedPackageType"]
        support = package_envelope(related)
        tare = Decimal(support.tare_kg)
        _require(
            support.family in {"PALLET", "SKID", "RACK"} and tare > 0,
            "discrete articles need a positive known support envelope",
        )
        quantity = result["package"]["packageQuantity"]
        _require(type(quantity) is int and quantity > 0, "positive article count required")
        net = _decimal(result["physical"]["unitNetKg"])
        old_gross = _decimal(result["physical"]["unitGrossKg"])
        _require(net > 0 and old_gross == net, "zero-tare article has unexplained headroom")
        new_gross = net + tare
        prepared = {
            "version": 1,
            "countedCategory": result["internalCategory"],
            "countedQuantity": quantity,
            "relatedCategory": related,
            "category": related,
            "quantity": quantity,
            "unitTareKg": _number(tare),
            "totalTareKg": _number(tare * quantity),
            "packingBudgetKg": _number(tare * quantity),
            "unitNetKg": _number(net),
            "beforeUnitGrossKg": _number(old_gross),
            "afterUnitGrossKg": _number(new_gross),
            "countBasis": "one_independent_envelope_support_per_counted_discrete_article",
            "authority": {**support.receipt(), "unitTareKg": _number(tare)},
            "sourceCargoAccess": False,
            "sourceCountRatioUsed": False,
        }
        result["physical"]["unitGrossKg"] = _number(new_gross)
        result["physical"]["grossWeight"] = {
            "value": _number(new_gross * quantity),
            "unit": "kilogram",
        }
        for row in result["allocations"]:
            row["grossKg"] = _number(new_gross * row["packageQuantity"])
        result["provenance"][PREPARED_SUPPORT_KEY] = prepared
    return result


def _consume_support(
    draw: Mapping[str, Any], plan: Mapping[str, Any], quantity: int
) -> dict[str, Any]:
    prepared = draw["provenance"][PREPARED_SUPPORT_KEY]
    _require(
        plan["privatePackingRelation"] == "ON_SUPPORT"
        and plan["relatedPackageType"] == prepared["relatedCategory"],
        "prepared support relation/category changed",
    )
    _require(
        quantity == prepared["countedQuantity"] == prepared["quantity"],
        "prepared support needs its exact counted scope",
    )
    _require(
        draw["internalCategory"] == prepared["countedCategory"],
        "prepared support counted category changed",
    )
    tare = _decimal(prepared["unitTareKg"])
    net = _decimal(prepared["unitNetKg"])
    _require(
        _decimal(draw["physical"]["unitNetKg"]) == net
        and _decimal(draw["physical"]["unitGrossKg"]) == net + tare,
        "prepared support unit masses changed",
    )
    _require(
        _decimal(prepared["beforeUnitGrossKg"]) == net
        and _decimal(prepared["afterUnitGrossKg"]) == net + tare
        and _decimal(prepared["totalTareKg"]) == tare * quantity
        and _decimal(prepared["packingBudgetKg"]) == tare * quantity,
        "prepared support mass does not conserve",
    )
    _require(
        _decimal(draw["physical"]["grossWeight"]["value"]) == quantity * (net + tare),
        "prepared support global gross changed",
    )
    for row in draw["allocations"]:
        _require(
            _decimal(row["grossKg"]) == row["packageQuantity"] * (net + tare),
            "prepared support row gross changed",
        )
    _require(
        prepared["sourceCargoAccess"] is False and prepared["sourceCountRatioUsed"] is False,
        "prepared support used source cargo",
    )
    return copy.deepcopy(prepared)


def _positive_mass_parts(
    total_kg: Any, regions: Sequence[Mapping[str, Any]], rng: random.Random
) -> list[Decimal]:
    total = _decimal(total_kg)
    _require(total > 0 and bool(regions), "positive batch mass and explicit fields required")
    _require(
        all(
            region["fact"] == "partitionNetKg" and region.get("unit", "kilogram") == "kilogram"
            for region in regions
        ),
        "batch fields need an explicit kilogram fact contract",
    )
    baselines = [_decimal(region["baseline"]) for region in regions]
    _require(all(value > 0 for value in baselines), "positive source mass precision required")
    quantum = max(Decimal(1).scaleb(cast(int, value.as_tuple().exponent)) for value in baselines)
    _require(total % quantum == 0, "batch mass does not fit its source precision")
    count = max(region["partitionIndex"] for region in regions) + 1
    _require(
        count == 2 and {region["partitionIndex"] for region in regions} == {0, 1},
        "reviewed batch presentation requires two explicit mass components",
    )
    units = int(total / quantum)
    _require(units >= count, "positive batch masses do not fit source precision")
    first = rng.randint(1, units - 1)
    return [quantum * first, total - quantum * first]


def _validate_composed_capacity(
    draw: Mapping[str, Any], physical_rows: Sequence[Mapping[str, Any]]
) -> None:
    if PREPARED_SUPPORT_KEY not in draw["provenance"]:
        return
    _require(len(physical_rows) == len(draw["allocations"]), "capacity row count differs")
    for allocation, physical in zip(draw["allocations"], physical_rows, strict=True):
        _require(
            allocation["packageQuantity"] == physical["quantity"],
            "capacity row quantity differs",
        )
        capacity = physical["capacity"]
        payload = _decimal(capacity["payloadKg"])
        _require(payload > 0, "positive explicit hardware payload required")
        if _decimal(allocation["grossKg"]) > payload:
            raise CandidateIncompatibility("composed load exceeds selected equipment payload")
        if capacity["volumeM3"] is not None:
            cube = _decimal(capacity["volumeM3"])
            _require(cube > 0, "positive explicit enclosed hardware cube required")
            if _decimal(allocation["volumeM3"]) > cube:
                raise CandidateIncompatibility("composed load exceeds selected enclosed volume")


def _inclusive_range(
    quantity: int,
    rng: random.Random,
    *,
    prefix_pattern: str = "A0A",
    digits: int = 6,
) -> dict[str, Any]:
    if type(quantity) is not int or quantity < 1 or digits < 1 or quantity > 9 * 10 ** (digits - 1):
        raise ValueError("inclusive range cannot fit quantity in its digit width")
    alphabet = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
    prefix = "".join(
        str(rng.randrange(10)) if character.isdigit() else rng.choice(alphabet)
        for character in prefix_pattern
    )
    start = rng.randint(10 ** (digits - 1), 10**digits - quantity)
    return {
        "prefix": prefix,
        "start": start,
        "end": start + quantity - 1,
        "quantity": quantity,
        "digits": digits,
    }


def _typed_relation(plan: Mapping[str, Any], category: str) -> tuple[str, str | None]:
    relation = plan["privatePackingRelation"]
    related = plan["relatedPackageType"]
    if relation == "MIXED":
        from document_ocr.synthesis.independent_accounting import MixedPackingPlan

        _require(category == "PACKAGE_PACKAGE", "mixed accounting counts generic packages")
        MixedPackingPlan.model_validate(
            {
                "unknown": False,
                "supportPackageType": plan["supportPackageType"],
                "relatedPackageType": related,
                "reason": "Verified native mixed accounting",
            }
        )
        return relation, related
    if relation not in _RELATIONS:
        raise ValueError("unsupported private packing relation")
    if (relation == "NONE") != (related is None):
        raise ValueError("private relation/category presence mismatch")
    if related is not None:
        if not isinstance(related, str):
            raise ValueError("related package category must be a string")
        _noun(related, 1)
        if PACKAGE_COUNTING_CONTEXTS[related]:
            raise ValueError("related units must be countable packages, not loading contexts")
    if "PALLET" in category and relation == "CONTAINS":
        raise ValueError("an open counted pallet supports rather than encloses packing")
    return relation, related


def _composition(relation: str, counted: str, related: str | None) -> str:
    if relation in {"NONE", "MIXED"}:
        return counted
    if related is None:
        raise AssertionError("nonempty relation needs related package wording")
    if relation == "SUPPORTS":
        return f"{related} ON {counted}"
    if relation == "ON_SUPPORT":
        return f"{counted} ON {related}"
    return f"{related} IN {counted}"


def _build_scope_ledger(
    blueprint: SamplingBlueprint,
    draw: Mapping[str, Any],
    typed_plan: Mapping[str, Any],
    seed: int | str,
    variant: int | str,
    *,
    nominal_capacity_litres: Decimal | str | int | None = None,
    capacity_authority: str | None = None,
    composition_facts: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Render current facts into one certificate scope without source ratios."""

    certificate = blueprint.ownership_data[NUMERIC_LEDGER_OWNER]["certificate"]
    target_before = _digest(blueprint.target)
    category = draw["internalCategory"]
    quantity = draw["package"]["packageQuantity"]
    if not isinstance(category, str) or type(quantity) is not int or quantity < 1:
        raise ValueError("numeric draw requires category and positive package quantity")
    counted = f"{quantity} {_noun(category, quantity)}"
    relation, related_category = _typed_relation(typed_plan, category)
    # Keep the reviewed deterministic namespace so production promotion does
    # not change previously validated facts for the same source/seed/variant.
    rng = random.Random(_digest([seed, blueprint.document_id, variant, "held-private-ledger"]))
    unit_net = _decimal(draw["physical"]["unitNetKg"])
    unit_gross = _decimal(draw["physical"]["unitGrossKg"])
    if unit_net <= 0 or unit_gross < unit_net:
        raise ValueError("positive unit net and gross >= net required")
    net = quantity * unit_net
    gross = quantity * unit_gross
    support: dict[str, Any] | None = None
    from document_ocr.synthesis.independent_composed_profiles import scope_composition

    native_composition = (
        scope_composition(draw, typed_plan, quantity) if relation != "NONE" else None
    )
    if native_composition is not None:
        if composition_facts is not None and (
            composition_facts["relatedQuantity"] != native_composition["relatedQuantity"]
        ):
            raise ValueError("scoped composition disagrees with complete group authority")
        composition_facts = native_composition
    if composition_facts is not None:
        related_quantity = composition_facts["relatedQuantity"]
        support = composition_facts.get("supportEnvelope")
    elif relation == "ON_SUPPORT":
        if related_category is None:
            raise AssertionError("support relation lacks a related category")
        support = (
            _consume_support(draw, typed_plan, quantity)
            if PREPARED_SUPPORT_KEY in draw["provenance"]
            else support_budget(related_category, gross - net, quantity, rng)
        )
        related_quantity = support["quantity"]
    elif relation in {"SUPPORTS", "CONTAINS"}:
        related_quantity = quantity * rng.randint(2, 6)
    else:
        related_quantity = None
    if relation == "SUPPORTS" and composition_facts is None:
        support = support_budget(
            category,
            gross - net,
            quantity,
            rng,
            fixed_count=quantity,
        )
    related = (
        f"{related_quantity} {_noun(related_category, related_quantity)}"
        if related_quantity is not None and related_category is not None
        else None
    )
    composition = _composition(relation, counted, related)
    result: dict[str, Any] = {
        "version": 1,
        "sourceDocumentId": blueprint.document_id,
        "certificateSha256": _digest(certificate),
        "countedCategory": category,
        "publicQuantity": quantity,
        "relation": relation,
        "relatedCategory": related_category,
        "relatedQuantity": related_quantity,
        "unitNetKg": _number(unit_net),
        "unitGrossKg": _number(unit_gross),
        "netKg": _number(net),
        "grossKg": _number(gross),
        "facts": {},
        "surfaces": {},
        "wordingFacts": [],
        "wordingAppends": {},
        "presentationDecisions": [],
    }
    if support is not None:
        result["facts"]["supportEnvelope"] = support
    if (nominal_capacity_litres is None) != (capacity_authority is None):
        raise ValueError("nominal capacity and explicit authority must be supplied together")
    capacity = None if nominal_capacity_litres is None else _decimal(nominal_capacity_litres)
    if capacity is not None and capacity <= 0:
        raise ValueError("nominal capacity must be positive")
    if capacity is not None and (
        not isinstance(capacity_authority, str) or not capacity_authority.strip()
    ):
        raise ValueError("nominal capacity requires a nonempty explicit authority")
    if capacity is not None and not any(
        region["role"] == "receptacle_statement" for region in certificate["regions"]
    ):
        raise ValueError("certificate has no owned nominal-capacity presentation")

    for index, item in enumerate(certificate["regions"]):
        role = item["role"]
        text: str
        if role in {"mixed_package_equation", "mixed_package_ranges"}:
            _require(relation == "MIXED", "mixed equation needs native component types")
            group = item.get("factGroup", "mixedAccounting")
            if group not in result["facts"]:
                if native_composition is not None:
                    result["facts"][group] = native_composition["mixed"]
                else:
                    if quantity < 2:
                        raise CandidateIncompatibility("mixed accounting needs two counted units")
                    support = support_budget(
                        typed_plan["supportPackageType"], gross - net, quantity - 1, rng
                    )
                    outer = support["quantity"]
                    packed = outer * rng.randint(2, 6)
                    loose = quantity - outer
                    result["facts"][group] = {
                        "supportQuantity": outer,
                        "packedQuantity": packed,
                        "looseQuantity": loose,
                        "totalRelatedQuantity": packed + loose,
                        "declaredQuantity": quantity,
                        "supportEnvelope": support,
                        "sourceCountRatioUsed": False,
                    }
            mixed = result["facts"][group]
            outer, packed, loose = (
                mixed["supportQuantity"],
                mixed["packedQuantity"],
                mixed["looseQuantity"],
            )
            total_related = mixed["totalRelatedQuantity"]
            _require(
                outer + loose == quantity and packed + loose == total_related,
                "mixed accounting conservation failed",
            )
            if role == "mixed_package_equation":
                text = (
                    f"{counted}={outer} {_noun(typed_plan['supportPackageType'], outer)}/"
                    f"{packed} {_noun(related_category, packed)}+"
                    f"{loose} {_noun(related_category, loose)}="
                    f"{total_related} {_noun(related_category, total_related)}"
                )
            else:
                text = (
                    f"{_noun(related_category, 1)} NOS.:1-{total_related}\n"
                    f"{_noun(typed_plan['supportPackageType'], 1)} NOS.:1-{outer}; "
                    f"LOOSE:{loose} {_noun(related_category, loose)}"
                )
            result["relatedQuantity"] = total_related
        elif role in {"component_package", "component_measurement"}:
            group = item["factGroup"]
            parts = item["componentCount"]
            if group not in result["facts"]:
                if quantity < parts:
                    raise CandidateIncompatibility("component counts need positive parts")
                boundaries = [0, *sorted(rng.sample(range(1, quantity), parts - 1)), quantity]
                values = [right - left for left, right in pairwise(boundaries)]
                result["facts"][group] = {
                    "quantities": values,
                    "total": quantity,
                    "sourceCountRatioUsed": False,
                }
            fact = result["facts"][group]
            _require(
                len(fact["quantities"]) == parts and sum(fact["quantities"]) == quantity,
                "component count ownership disagrees",
            )
            count = fact["quantities"][item["componentIndex"]]
            if role == "component_package":
                text = f"{count} {_noun(category, count)}"
            else:
                unit = {
                    "grossKg": unit_gross,
                    "netKg": unit_net,
                    "volumeM3": _decimal(draw["physical"]["unitVolumeM3"]),
                }[item["fact"]]
                text = format_measure(item["source_text"], _decimal(item["baseline"]), unit * count)
        elif role == "three_level_chain":
            if relation in {"SUPPORTS", "CONTAINS"} and related_category in {
                "PACKAGE_CARTON",
                "PACKAGE_BOX",
                "PACKAGE_BOX_FIBREBOARD",
            }:
                if related_quantity is None:
                    raise AssertionError("three-level relation has no inner quantity")
                pieces = related_quantity * rng.randint(2, 100)
                text = f"QTY:{counted}={related}={pieces} PIECES"
                result["facts"]["threeLevels"] = {
                    "counted": quantity,
                    "inner": related_quantity,
                    "pieces": pieces,
                    "oldRatioUsed": False,
                }
            else:
                text = "QTY:" + composition
                result["presentationDecisions"].append(
                    {
                        "owner": f"{NUMERIC_REGION_PREFIX}:{index}",
                        "choice": "two_or_one_level_presentation",
                        "reason": "Current typed plan does not request a third piece layer.",
                    }
                )
        elif role == "packing_clause":
            repeats = item.get("repeats", 1)
            if type(repeats) is not int or repeats < 1:
                raise ValueError("repeated packing clause needs positive cardinality")
            text = (
                item.get("prefix", "") + "\n".join([composition] * repeats) + item.get("suffix", "")
            )
        elif role == "unit_net_prefix":
            text = f"{counted} X {_number(unit_net)} KG NET EACH OF "
        elif role == "unit_mass_suffix":
            text = (
                f", {_noun(category, quantity)}, WITH {_number(unit_net)} KG NET AND "
                f"{_number(unit_gross)} KG GROSS PER {_noun(category, 1)}."
            )
        elif role in {"support_mass_breakdown", "component_mass_pair"}:
            surplus = gross - net
            has_support = relation in {"SUPPORTS", "ON_SUPPORT"}
            if role == "component_mass_pair" or has_support:
                if has_support:
                    if support is None:
                        raise AssertionError("support relation lacks mass authority")
                    support_mass = _decimal(support["totalTareKg"])
                else:
                    cents = int((surplus * 100).to_integral_value(rounding=ROUND_FLOOR))
                    if cents < 2:
                        raise CandidateIncompatibility(
                            "gross/net headroom cannot express positive private components"
                        )
                    support_mass = Decimal(rng.randint(1, max(1, cents // 2))) / 100
                cargo_gross = gross - support_mass
                if cargo_gross < net:
                    raise AssertionError("component split exceeds gross/net headroom")
                result["facts"][role] = {
                    "cargoGrossKg": _number(cargo_gross),
                    "supportOrDunnageGrossKg": _number(support_mass),
                    "shippingGrossKg": _number(gross),
                    "conserved": _number(cargo_gross + support_mass) == _number(gross),
                }
                if role == "component_mass_pair":
                    text = (
                        f"PACKAGED CARGO GROSS: {_number(cargo_gross)} KG\n"
                        f"NON-CARGO PACKING / DUNNAGE GROSS: {_number(support_mass)} KG"
                    )
                else:
                    supports = counted if relation == "SUPPORTS" else related
                    text = (
                        f"TOTAL NET WEIGHT: {_number(net)} KG\n"
                        f"TOTAL GROSS WITHOUT SUPPORTS: {_number(cargo_gross)} KG\n"
                        f"TOTAL SUPPORTS: {supports}\n"
                        f"TOTAL SUPPORTS WEIGHT: {_number(support_mass)} KG\n"
                        f"TOTAL SHIPPING GROSS WITH SUPPORTS: {_number(gross)} KG"
                    )
            else:
                text = (
                    f"TOTAL NET WEIGHT: {_number(net)} KG\n"
                    f"TOTAL SHIPPING GROSS WEIGHT: {_number(gross)} KG"
                )
                result["presentationDecisions"].append(
                    {
                        "owner": f"{NUMERIC_REGION_PREFIX}:{index}",
                        "choice": "no_separate_support_mass_statement",
                        "reason": "Current plan has no support layer; "
                        "no source pallet tare retained.",
                    }
                )
        elif role == "count_net_statement":
            text = f"{counted},\nNET WEIGHT {_number(net)} KG"
        elif role == "net_statement":
            text = f"N.W. {_number(net)} KG."
        elif role == "unit_mass_pair":
            text = (
                f"GROSS WT / {_noun(category, 1)}: {_number(unit_gross)} KG\n"
                f"NET WT / {_noun(category, 1)}: {_number(unit_net)} KG"
            )
        elif role == "dual_unit_net_statement":
            pounds = (unit_net / POUNDS_TO_KG).quantize(Decimal("0.01"), rounding=ROUND_HALF_UP)
            text = (
                item.get("prefix", "")
                + f"NET CONTENT PER {_noun(category, 1)}: {_number(unit_net)} KG ({pounds} LB),"
                + item.get("suffix", "")
            )
            result["facts"]["explicitDualUnitNet"] = {
                "kg": _number(unit_net),
                "lb": str(pounds),
                "kgPerPound": str(POUNDS_TO_KG),
                "rounding": "half_up_2_decimal_lb",
            }
            result["presentationDecisions"].append(
                {
                    "owner": f"{NUMERIC_REGION_PREFIX}:{index}",
                    "choice": "fresh_explicit_per_counted_unit_net",
                    "reason": "Intentional complete-phrase replacement; no inference that the "
                    "original bare dual-unit phrase proved a per-unit fact.",
                }
            )
        elif role == "support_statement":
            text = (
                composition
                if relation in {"SUPPORTS", "ON_SUPPORT"}
                else "NO SUPPORT UNITS DECLARED"
            )
            result["presentationDecisions"].append(
                {
                    "owner": f"{NUMERIC_REGION_PREFIX}:{index}",
                    "choice": "current_typed_support_declaration",
                    "reason": "Complete previous negative/zero-support statement is owned, "
                    "not left as a contradictory scalar beside a new private plan.",
                }
            )
        elif role == "private_measurement":
            text = format_measure(
                item["source_text"], _decimal(item["baseline"]), _decimal(item["sampledValue"])
            )
            result["facts"][item["factGroup"]] = item["sampledValue"]
        elif role == "shared_container_range":
            text = f"{item['rangePrefix']}1 TO {item['rangePrefix']}{quantity}"
            result["facts"][item["factGroup"]] = {
                "start": 1,
                "end": quantity,
                "quantity": quantity,
            }
        elif role == "container_net_mass":
            text = format_measure(item["source_text"], _decimal(item["baselineKg"]), net)
            result["facts"][item["factGroup"]] = {"netKg": _number(net)}
        elif role == "on_support_with_unit_net":
            lines = [f"TOTAL {counted} ONLY"]
            if relation == "ON_SUPPORT":
                lines += [f"ON {related}", str(related)]
            elif relation != "NONE":
                lines.append(composition)
            lines.append(f"[{counted} X {_number(unit_net)} KG]")
            text = "\n".join(lines)
        elif role == "private_ranges":
            count = quantity if item["countRole"] == "counted" else related_quantity
            if count is None and item.get("optionalWithoutRelatedUnit") is True:
                result["surfaces"][f"{NUMERIC_REGION_PREFIX}:{index}"] = [""]
                result["presentationDecisions"].append(
                    {
                        "owner": f"{NUMERIC_REGION_PREFIX}:{index}",
                        "choice": "reviewed_optional_private_range_omission",
                        "reason": "Current typed presentation has no related unit to number.",
                    }
                )
                continue
            if count is None:
                raise ValueError("private range requests an absent related-unit count")
            parts = item["parts"]
            if type(parts) is not int or parts < 1:
                raise ValueError("private range partition cardinality must be positive")
            if count < parts:
                raise CandidateIncompatibility("private range count cannot fill positive lots")
            digits = item["digits"]
            if count > 9 * 10 ** (digits - 1):
                raise CandidateIncompatibility("private range exceeds printed digit width")
            group = item["factGroup"]
            if group not in result["facts"]:
                cuts = sorted(rng.sample(range(1, count), parts - 1))
                boundaries = [0, *cuts, count]
                widths = [right - left for left, right in pairwise(boundaries)]
                common = _inclusive_range(
                    count,
                    rng,
                    prefix_pattern=item["prefixPattern"],
                    digits=digits,
                )
                cursor = common["start"]
                rows = []
                for width in widths:
                    rows.append({"start": cursor, "end": cursor + width - 1, "quantity": width})
                    cursor += width
                result["facts"][group] = {
                    "prefix": common["prefix"],
                    "rows": rows,
                    "quantity": count,
                    "digits": digits,
                    "prefixPattern": item["prefixPattern"],
                    "publicContainerAllocationsNotInferred": True,
                }
            fact = result["facts"][group]
            expected = (parts, count, digits, item["prefixPattern"])
            observed = (
                len(fact["rows"]),
                fact["quantity"],
                fact["digits"],
                fact["prefixPattern"],
            )
            if observed != expected:
                raise ValueError("repeated private range group has conflicting declarations")
            rendered = item.get("renderPart")
            if rendered is not None and (type(rendered) is not int or not 0 <= rendered < parts):
                raise ValueError("private range selected part is outside its partition")
            range_category = category if item["countRole"] == "counted" else related_category
            if range_category is None:
                raise AssertionError("related range category is absent")
            lines = []
            for row_index, row in enumerate(fact["rows"]):
                if rendered is not None and row_index != rendered:
                    continue
                lines.append(
                    f"{_noun(range_category, 1)} NOS.: {fact['prefix']}"
                    f"{row['start']:0{digits}d} - {fact['prefix']}{row['end']:0{digits}d}"
                )
                if item.get("lotCaptions"):
                    lines.append(f"LOT NO.{row_index + 1}")
            text = "\n".join(lines)
        elif role == "inclusive_range":
            digits = item["digits"]
            if quantity > 9 * 10 ** (digits - 1):
                raise CandidateIncompatibility("inclusive range exceeds printed digit width")
            fact = _inclusive_range(
                quantity,
                rng,
                prefix_pattern=item["prefixPattern"],
                digits=digits,
            )
            result["facts"]["inclusiveRange"] = fact
            text = (
                f"{_noun(category, 1)} NOS. {fact['prefix']}{fact['start']:0{digits}d} - "
                f"{fact['prefix']}{fact['end']:0{digits}d}{item.get('suffix', '')}"
            )
        elif role == "receptacle_statement":
            text = composition
            if capacity is not None:
                text += f", NOMINAL CAPACITY {_number(capacity)} LITRES PER {_noun(category, 1)}"
                result["facts"]["nominalCapacity"] = {
                    "litres": _number(capacity),
                    "authority": capacity_authority,
                }
            else:
                result["presentationDecisions"].append(
                    {
                        "owner": f"{NUMERIC_REGION_PREFIX}:{index}",
                        "choice": "reviewed_capacity_and_approval_free_statement",
                        "reason": "No nominal receptacle-capacity or approval authority supplied; "
                        "old source-only claim intentionally not preserved.",
                    }
                )
            text += "\n"
        else:
            raise ValueError("unsupported numeric certificate role: " + str(role))
        result["surfaces"][f"{NUMERIC_REGION_PREFIX}:{index}"] = [text]

    for owner in certificate.get("wordingOwners", []):
        append = owner.get("append")
        if append == "dual_unit_net":
            pounds = (unit_net / POUNDS_TO_KG).quantize(Decimal("0.01"), rounding=ROUND_HALF_UP)
            clause = f"NET CONTENT PER {_noun(category, 1)}: {_number(unit_net)} KG ({pounds} LB)."
            result["wordingAppends"][owner["key"]] = clause
            result["facts"]["dualUnitNet"] = {
                "kg": _number(unit_net),
                "lb": str(pounds),
                "kgPerPound": str(POUNDS_TO_KG),
                "rounding": "half_up_2_decimal_lb",
            }
        elif append == "count_unit_net":
            result["wordingAppends"][owner["key"]] = (
                f"{composition}; {_number(unit_net)} KG NET PER {_noun(category, 1)}."
            )
        elif append is not None:
            raise ValueError("unknown complete-product projection kind")
        result["wordingFacts"].append(
            {
                "key": owner["key"],
                "completeOldProductRegenerated": True,
                "countedUnit": category,
                "rule": "Do not preserve old counts, capacities, approval codes or packing "
                "materials. Host supplies current facts separately.",
            }
        )
    target_after = _digest(blueprint.target)
    if target_after != target_before:
        raise AssertionError("private numeric ledger mutated public target")
    result["publicAllocationVisibilityUnchanged"] = True
    result["invariants"] = {
        "positiveCount": quantity > 0,
        "grossAtLeastNet": gross >= net,
        "publicTargetMutation": False,
        "publicTargetBeforeSha256": target_before,
        "publicTargetAfterSha256": target_after,
        "nativeReasonConsumed": False,
        "sourceRatiosConsumed": False,
    }
    return result


def project_wording(
    values: Mapping[str, str], ledger: Mapping[str, Any]
) -> tuple[dict[str, str], dict[str, Any]]:
    """Project every authoritative clause exactly once and return a receipt."""

    result = copy.deepcopy(dict(values))
    actions: dict[str, dict[str, Any]] = {}
    appended: dict[str, str] = {}
    retained: dict[str, str] = {}
    for key, clause in ledger["wordingAppends"].items():
        if key not in result or not isinstance(result[key], str) or not result[key].strip():
            raise ValueError("complete goods wording required before numeric projection")
        count = result[key].count(clause)
        if count > 1:
            raise ValueError("authoritative numeric clause occurs more than once")
        if count == 0:
            result[key] = result[key].rstrip() + "\n" + clause
            appended[key] = clause
            action = "appended_authoritative_clause"
        else:
            retained[key] = clause
            action = "already_present_exact_authoritative_clause"
        if result[key].count(clause) != 1:
            raise AssertionError("numeric projection did not produce exactly one clause")
        actions[key] = {
            "action": action,
            "exactOccurrencesBefore": count,
            "exactOccurrencesAfter": 1,
        }
    if set(result) != set(values):
        raise AssertionError("numeric wording projection introduced a lexical field")
    receipt = {
        "originalValuesSha256": _digest(values),
        "projectedValuesSha256": _digest(result),
        "ownedClauses": copy.deepcopy(ledger["wordingAppends"]),
        "actions": actions,
        "appendedClauses": appended,
        "retainedClauses": retained,
        "unchangedOtherFields": all(
            result[key] == value
            for key, value in values.items()
            if key not in ledger["wordingAppends"]
        ),
    }
    return result, receipt


def wording_context(ledger: Mapping[str, Any]) -> str:
    facts = {
        key: ledger[key]
        for key in (
            "countedCategory",
            "publicQuantity",
            "relation",
            "relatedCategory",
            "relatedQuantity",
            "wordingFacts",
            "wordingAppends",
        )
    }
    return (
        "\nHOST-OWNED PRIVATE NUMERIC FACTS\n"
        + json.dumps(facts, sort_keys=True)
        + "\nWrite goods identity/specifications only. Do not copy source package counts, "
        "per-unit fill masses, capacities, approval marks or support counts. "
        "The host appends the exact supplied numeric clauses; do not duplicate them. "
        "Do not contradict the sampled counted package or material."
    )


class IndependentNumericCertificates:
    """Immutable complete-region authority loaded from an explicit runtime path."""

    def __init__(
        self,
        *,
        path: Path,
        file_sha256: str,
        certificates: Mapping[str, Mapping[str, Any]],
    ) -> None:
        self.path = path
        self.file_sha256 = file_sha256
        self._certificates: dict[str, dict[str, Any]] = {
            source_id: copy.deepcopy(dict(certificate))
            for source_id, certificate in certificates.items()
        }

    @classmethod
    def from_path(
        cls, path: Path, *, expected_sha256: str | None = None
    ) -> IndependentNumericCertificates:
        """Load and validate one exact v2 certificate inventory."""

        resolved = path.resolve(strict=True)
        payload = resolved.read_bytes()
        file_sha256 = hashlib.sha256(payload).hexdigest()
        if expected_sha256 is not None and file_sha256 != expected_sha256:
            raise ValueError("numeric certificate inventory SHA-256 mismatch")
        value = json.loads(payload)
        if not isinstance(value, dict) or value.get("version") != 2:
            raise ValueError("numeric certificate inventory requires version 2")
        selected = value.get("selectedSources")
        sources = value.get("sources")
        if (
            not isinstance(selected, list)
            or not selected
            or any(not isinstance(source_id, str) or not source_id for source_id in selected)
            or len(selected) != len(set(selected))
        ):
            raise ValueError("certificate inventory requires unique selected source IDs")
        if not isinstance(sources, dict) or set(selected) != set(sources):
            raise ValueError("certificate coverage differs from selected source cohort")
        for source_id, certificate in sources.items():
            cls._validate_certificate(source_id, certificate)
        return cls(path=resolved, file_sha256=file_sha256, certificates=sources)

    @staticmethod
    def _validate_certificate(source_id: str, certificate: Any) -> None:
        if not isinstance(certificate, dict) or certificate.get("reviewStatus") != "reviewed":
            raise ValueError(f"complete source review required: {source_id}")
        required = (
            "sourceSha256",
            "sourceContractSha256",
            "sourcePublicTargetSha256",
            "sourceContainerCount",
            "sourceAllocationPresence",
            "regions",
            "wordingOwners",
        )
        missing = [key for key in required if key not in certificate]
        if missing:
            raise ValueError(f"incomplete certificate {source_id}: missing {missing}")
        for pin in required[:3]:
            value = certificate[pin]
            if not isinstance(value, str) or re.fullmatch(r"[0-9a-f]{64}", value) is None:
                raise ValueError(f"certificate {source_id} has invalid {pin}")
        if (
            type(certificate["sourceContainerCount"]) is not int
            or certificate["sourceContainerCount"] < 0
        ):
            raise ValueError(f"certificate {source_id} has invalid container count")
        if not isinstance(certificate["sourceAllocationPresence"], list) or any(
            type(item) is not bool for item in certificate["sourceAllocationPresence"]
        ):
            raise ValueError(f"certificate {source_id} has invalid allocation mask")
        if not isinstance(certificate["regions"], list) or not isinstance(
            certificate["wordingOwners"], list
        ):
            raise ValueError(f"certificate {source_id} has invalid ownership collections")
        constraints = certificate.get("constraints", [])
        if (
            "requiredPhysicalAuthority" in certificate
            and certificate["requiredPhysicalAuthority"]
            != "native_goods_conditioned_unit_profile_v1"
        ):
            raise ValueError("unknown required physical authority")
        if "unitProfilePrecision" in certificate:
            precision = certificate["unitProfilePrecision"]
            if not isinstance(precision, dict) or set(precision) != {"massKg", "volumeM3"}:
                raise ValueError("unit profile precision requires massKg and volumeM3")
            for value in precision.values():
                if not isinstance(value, str):
                    raise ValueError("unit profile precision must use decimal strings")
                quantum = Decimal(value)
                if (
                    not quantum.is_finite()
                    or quantum <= 0
                    or quantum.normalize().as_tuple().digits != (1,)
                ):
                    raise ValueError("unit profile precision must be a positive power of ten")
        if "cargoPresentation" in certificate:
            CargoPresentation.model_validate(certificate["cargoPresentation"])
        if not isinstance(constraints, list) or any(
            not isinstance(constraint, dict) or not isinstance(constraint.get("kind"), str)
            for constraint in constraints
        ):
            raise ValueError(f"certificate {source_id} has invalid constraints")
        held = certificate.get("holdReasons", [])
        if not isinstance(held, list) or any(not isinstance(reason, str) for reason in held):
            raise ValueError(f"certificate {source_id} has invalid hold reasons")
        readiness = certificate.get("sourceReadinessHolds", [])
        if not isinstance(readiness, list):
            raise ValueError(f"certificate {source_id} has invalid source readiness holds")
        for hold in readiness:
            reviewed = SourceReadinessHold.model_validate(hold)
            evidence = Path(reviewed.evidence.path)
            if not reviewed.evidence.path or evidence.is_absolute() or ".." in evidence.parts:
                raise ValueError("source readiness evidence must be project-relative")
        for region in certificate["regions"]:
            if not isinstance(region, dict):
                raise ValueError(f"certificate {source_id} has invalid region")
            start = region.get("byte_start")
            end = region.get("byte_end")
            text = region.get("source_text")
            if (
                type(start) is not int
                or type(end) is not int
                or start < 0
                or end <= start
                or not isinstance(text, str)
                or not text
                or end - start != len(text.encode())
            ):
                raise ValueError(f"certificate {source_id} has invalid region span")
            role = region.get("role")
            if role not in SUPPORTED_REGION_ROLES and not held:
                raise ValueError(f"unsupported complete-region role: {role}")
            scope = region.get("scope", {"kind": "shipment"})
            if not isinstance(scope, dict) or scope.get("kind") not in {
                "shipment",
                "container",
                "shared_container",
            }:
                raise ValueError("complete numeric region has unknown scope")
            if scope["kind"] == "container" and (
                type(scope.get("index")) is not int
                or not 0 <= scope["index"] < certificate["sourceContainerCount"]
            ):
                raise ValueError("complete numeric region has invalid container row")
            if scope["kind"] == "shared_container" and not any(
                constraint.get("kind") == "equal_container_loads"
                for constraint in certificate.get("constraints", [])
            ):
                raise ValueError("shared facts require an equal-load certificate")
            IndependentNumericCertificates._validate_role(source_id, region)
        component_groups: dict[tuple[str, str], list[dict[str, Any]]] = {}
        for region in certificate["regions"]:
            if region["role"] in {"component_package", "component_measurement"}:
                key = (
                    json.dumps(region.get("scope", {"kind": "shipment"}), sort_keys=True),
                    region["factGroup"],
                )
                component_groups.setdefault(key, []).append(region)
        for regions in component_groups.values():
            counts = {region["componentCount"] for region in regions}
            declared = [r for r in regions if r["role"] == "component_package"]
            if len(counts) != 1 or {region["componentIndex"] for region in declared} != set(
                range(next(iter(counts)))
            ):
                raise ValueError("additive component certificate must own every component")
        mixed = [r for r in certificate["regions"] if r["role"] == "mixed_package_equation"]
        if mixed and (
            not any(c["kind"] == "mixed_package_conservation" for c in constraints)
            or any(r.get("scope", {"kind": "shipment"}) != {"kind": "shipment"} for r in mixed)
            or len({r.get("factGroup", "mixedAccounting") for r in mixed}) != 1
        ):
            raise ValueError("mixed equation needs one shipment accounting authority")
        for owner in certificate["wordingOwners"]:
            if not isinstance(owner, dict):
                raise ValueError(f"certificate {source_id} has invalid wording owner")
            if (
                not isinstance(owner.get("key"), str)
                or not owner["key"]
                or type(owner.get("byte_start")) is not int
                or type(owner.get("byte_end")) is not int
                or owner["byte_start"] < 0
                or owner["byte_end"] <= owner["byte_start"]
                or not isinstance(owner.get("source_text"), str)
                or owner["byte_end"] - owner["byte_start"] != len(owner["source_text"].encode())
                or owner.get("append") not in {None, "dual_unit_net", "count_unit_net"}
            ):
                raise ValueError(f"certificate {source_id} has invalid wording owner")

    @staticmethod
    def _validate_role(source_id: str, region: Mapping[str, Any]) -> None:
        """Validate fields consumed by each executable complete-region role."""

        role = region["role"]
        if "handlingTargetPath" in region and (
            role != "packing_clause"
            or not isinstance(region["handlingTargetPath"], str)
            or re.fullmatch(
                r"documentPatch\.goodsItemDetails\[\d+\]\.handlingInstructions\[\d+\]",
                region["handlingTargetPath"],
            )
            is None
            or not isinstance(region.get("handlingTargetSuffix", ""), str)
        ):
            raise ValueError("complete handling projection has an invalid target declaration")
        if "handlingTargetSuffix" in region and "handlingTargetPath" not in region:
            raise ValueError("handling suffix requires an explicit handling target")
        optional_without_related = region.get("optionalWithoutRelatedUnit")
        if "optionalWithoutRelatedUnit" in region and type(optional_without_related) is not bool:
            raise ValueError(f"certificate {source_id} has invalid optional-region flag")
        if optional_without_related is True and not (
            role == "private_ranges" and region.get("countRole") == "related"
        ):
            raise ValueError(f"certificate {source_id} has invalid optional-region role")
        if role == "packing_clause":
            repeats = region.get("repeats", 1)
            if (
                type(repeats) is not int
                or repeats < 1
                or any(not isinstance(region.get(key, ""), str) for key in ("prefix", "suffix"))
            ):
                raise ValueError(f"certificate {source_id} has invalid packing clause")
        elif role in {"private_measurement", "container_net_mass"}:
            baseline_key = "baselineKg" if role == "container_net_mass" else "baseline"
            if baseline_key not in region:
                raise ValueError(f"certificate {source_id} measurement lacks {baseline_key}")
            _decimal(region[baseline_key])
            if role == "private_measurement" and (
                not isinstance(region.get("fact"), str)
                or not isinstance(region.get("factGroup"), str)
            ):
                raise ValueError(f"certificate {source_id} has invalid private measurement")
            if region.get("fact") == "partitionNetKg" and (
                not isinstance(region.get("partitionGroup"), str)
                or type(region.get("partitionIndex")) is not int
                or region["partitionIndex"] not in (0, 1)
            ):
                raise ValueError(f"certificate {source_id} has invalid mass partition")
        elif role in {"inclusive_range", "private_ranges"}:
            if (
                type(region.get("digits")) is not int
                or region["digits"] < 1
                or not isinstance(region.get("prefixPattern"), str)
            ):
                raise ValueError(f"certificate {source_id} has invalid private range")
            if role == "private_ranges" and (
                region.get("countRole") not in {"counted", "related"}
                or type(region.get("parts")) is not int
                or region["parts"] < 1
                or not isinstance(region.get("factGroup"), str)
            ):
                raise ValueError(f"certificate {source_id} has invalid partitioned range")
        elif role == "shared_container_range" and (
            not isinstance(region.get("rangePrefix"), str)
            or not isinstance(region.get("factGroup"), str)
        ):
            raise ValueError(f"certificate {source_id} has invalid shared range")
        elif role in {"component_package", "component_measurement"} and (
            not isinstance(region.get("factGroup"), str)
            or not region["factGroup"]
            or type(region.get("componentCount")) is not int
            or region["componentCount"] < 2
            or type(region.get("componentIndex")) is not int
            or not 0 <= region["componentIndex"] < region["componentCount"]
        ):
            raise ValueError(f"certificate {source_id} has invalid additive component")
        if role == "component_measurement":
            if region.get("fact") not in {"grossKg", "netKg", "volumeM3"}:
                raise ValueError("component measure must have an explicit mass/volume role")
            _decimal(region["baseline"])

    def certificates(self) -> dict[str, dict[str, Any]]:
        return copy.deepcopy(self._certificates)

    def certificate(self, source_id: str) -> dict[str, Any]:
        self.has_certificate(source_id)
        return copy.deepcopy(dict(self._certificates[source_id]))

    def source_readiness_holds(self, source_id: str) -> tuple[SourceReadinessHold, ...]:
        """Return nonnumeric blockers without altering numeric role readiness."""
        self.has_certificate(source_id)
        return tuple(
            SourceReadinessHold.model_validate(hold)
            for hold in self._certificates[source_id].get("sourceReadinessHolds", [])
        )

    def has_certificate(self, source_id: str) -> bool:
        if source_id not in self._certificates:
            raise ValueError(f"source lacks a complete reviewed numeric certificate: {source_id}")
        return True

    def allowed_private_relations(self, source_id: str) -> tuple[str, ...]:
        self.has_certificate(source_id)
        certificate = self._certificates[source_id]
        if certificate.get("holdReasons"):
            return ()
        if any(r["role"] == "mixed_package_equation" for r in certificate["regions"]):
            return ("MIXED",)
        visible = any(
            region["role"] in _COMPOSITION_ROLES for region in certificate["regions"]
        ) or any(owner.get("append") == "count_unit_net" for owner in certificate["wordingOwners"])
        allowed: tuple[str, ...] = (
            ("NONE", "CONTAINS", "SUPPORTS", "ON_SUPPORT") if visible else ("NONE",)
        )
        if any(
            region["role"] == "private_ranges"
            and region["countRole"] == "related"
            and not region.get("optionalWithoutRelatedUnit")
            for region in certificate["regions"]
        ):
            allowed = tuple(relation for relation in allowed if relation != "NONE")
        return allowed

    def apply_blueprint(
        self, blueprint: SamplingBlueprint, source_id: str | None = None
    ) -> SamplingBlueprint:
        source = source_id or blueprint.document_id
        self.has_certificate(source)
        certificate = self._certificates[source]
        if certificate.get("holdReasons"):
            raise ValueError(
                "source numeric certificate held: " + "; ".join(certificate["holdReasons"])
            )
        patch = blueprint.target["documentPatch"]
        mask = [
            "packageQuantity" in row
            for goods in patch.get("goodsItemDetails", [])
            for row in goods.get("splitGoodsPlacement", [])
        ]
        if (
            mask != certificate["sourceAllocationPresence"]
            or len(patch.get("containerInformation", [])) != certificate["sourceContainerCount"]
        ):
            raise ValueError("source allocation visibility differs from certificate")
        result = _apply_certificate(blueprint, certificate)
        ownership = copy.deepcopy(dict(result.ownership_data))
        bindings = copy.deepcopy(dict(result.historical_bindings))
        ownership["freshPrivatePacking"]["constraints"] = copy.deepcopy(
            certificate.get("constraints", [])
        )
        for index, region in enumerate(certificate["regions"]):
            if region["role"] != "container_net_mass":
                continue
            key = f"{NUMERIC_REGION_PREFIX}:{index}"
            if region.get("scope", {}).get("kind") != "container":
                raise ValueError("container-net mass requires explicit container scope")
            bindings[key]["value_kind"] = "decimal_measurement"
            ownership.setdefault("measure_owners", {})[key] = {
                "measure": "netWeight",
                "row": region["scope"]["index"],
                "unit": "kilogram",
                "baseline": region["baselineKg"],
            }
        return replace(result, historical_bindings=bindings, ownership_data=ownership)

    def prepare_draw(
        self,
        blueprint: SamplingBlueprint,
        draw: Mapping[str, Any],
        typed_plan: Mapping[str, Any],
        request: Mapping[str, Any],
    ) -> dict[str, Any]:
        certificate = blueprint.ownership_data[NUMERIC_LEDGER_OWNER]["certificate"]
        if certificate != self._certificates.get(blueprint.document_id):
            raise ValueError("draw certificate differs from inventory authority")
        return _prepare_draw(draw, typed_plan, request, certificate)

    def build_ledger(
        self,
        blueprint: SamplingBlueprint,
        draw: Mapping[str, Any],
        typed_plan: Mapping[str, Any],
        seed: int | str,
        variant: int | str,
        *,
        nominal_capacity_litres: Decimal | str | int | None = None,
        capacity_authority: str | None = None,
    ) -> dict[str, Any]:
        """Build shipment and child-scope receipts from current sampled facts."""

        certificate = blueprint.ownership_data[NUMERIC_LEDGER_OWNER]["certificate"]
        self.has_certificate(blueprint.document_id)
        relation = typed_plan["privatePackingRelation"]
        if relation not in self.allowed_private_relations(blueprint.document_id):
            raise ValueError("private relation has no complete visible presentation owner")
        if certificate != self._certificates[blueprint.document_id]:
            raise ValueError("blueprint certificate differs from inventory authority")
        if (
            not certificate["regions"]
            and not certificate["wordingOwners"]
            and (relation != "NONE" or typed_plan["relatedPackageType"] is not None)
        ):
            raise ValueError("private relationship has no certified presentation owner")
        runtime_regions = copy.deepcopy(certificate["regions"])
        for region in runtime_regions:
            if (
                region["role"] != "private_measurement"
                or region.get("scope", {}).get("kind") == "shared_container"
            ):
                continue
            scope = region.get("scope", {"kind": "shipment"})
            quantity = (
                draw["package"]["packageQuantity"]
                if scope["kind"] == "shipment"
                else draw["allocations"][scope["index"]]["packageQuantity"]
            )
            unit_net = _decimal(draw["physical"]["unitNetKg"])
            unit_gross = _decimal(draw["physical"]["unitGrossKg"])
            known = {
                "unitNetKg": unit_net,
                "unitGrossKg": unit_gross,
                "netKg": quantity * unit_net,
                "grossKg": quantity * unit_gross,
                "declaredUntypedMassKg": quantity * unit_gross,
            }
            if region["fact"] not in known:
                raise ValueError("unsupported explicit private measurement fact")
            region["sampledValue"] = _number(known[region["fact"]])

        shared = [
            region
            for region in runtime_regions
            if region.get("scope", {}).get("kind") == "shared_container"
        ]
        shared_facts: dict[str, Any] = {}
        if shared:
            rows = draw["allocations"]
            if not rows or len({row["packageQuantity"] for row in rows}) != 1:
                raise CandidateIncompatibility("shared facts require certified equal loads")
            row_quantity = rows[0]["packageQuantity"]
            mass_partition = draw.get("provenance", {}).get(PRIVATE_PARTITION_KEY)
            if mass_partition is not None and (
                mass_partition["mode"] != "positive_mass_kg"
                or mass_partition["countedUnitContext"] != "bulk_loading_lot"
                or mass_partition["publicPackageCountsInferred"] is not False
            ):
                raise ValueError("invalid explicit bulk batch mass authority")
            if row_quantity < 2 and mass_partition is None:
                raise CandidateIncompatibility(
                    "two private lot components require two units per load"
                )
            for field, unit in (
                ("netKg", "unitNetKg"),
                ("grossKg", "unitGrossKg"),
                ("volumeM3", "unitVolumeM3"),
            ):
                if any(
                    _decimal(row[field]) != row_quantity * _decimal(draw["physical"][unit])
                    for row in rows
                ):
                    raise ValueError("shared row facts do not conserve current draw")
            if sum(row["packageQuantity"] for row in rows) != draw["package"]["packageQuantity"]:
                raise ValueError("shared counts do not conserve shipment quantity")
            values = {
                "rowGrossKg": rows[0]["grossKg"],
                "rowVolumeM3": rows[0]["volumeM3"],
                "unitGrossKg": draw["physical"]["unitGrossKg"],
                "unitNetKg": draw["physical"]["unitNetKg"],
            }
            partitions: dict[str, list[int] | list[Decimal]] = {}
            for region in shared:
                if region["role"] == "shared_container_range":
                    continue
                if region["role"] != "private_measurement":
                    raise ValueError("unknown shared-container fact role")
                fact = region["fact"]
                if fact == "partitionNetKg":
                    group = region["partitionGroup"]
                    if group not in partitions:
                        rng = random.Random(
                            _digest([seed, blueprint.document_id, variant, "private-lot", group])
                        )
                        if mass_partition is None:
                            first = rng.randint(1, row_quantity - 1)
                            partitions[group] = [first, row_quantity - first]
                        else:
                            fields = [
                                item for item in shared if item.get("partitionGroup") == group
                            ]
                            partitions[group] = _positive_mass_parts(rows[0]["netKg"], fields, rng)
                    part = region["partitionIndex"]
                    if type(part) is not int or part not in (0, 1):
                        raise ValueError("two-lot partition needs component index 0 or 1")
                    sampled = partitions[group][part]
                    region["sampledValue"] = _number(
                        sampled
                        if mass_partition is not None
                        else sampled * _decimal(draw["physical"]["unitNetKg"])
                    )
                elif fact in values:
                    region["sampledValue"] = values[fact]
                else:
                    raise ValueError("unknown explicit shared-container fact")
            shared_facts = {
                "rowQuantity": row_quantity,
                "scalarFacts": values,
                **(
                    {"positiveLotCountPartitions": partitions}
                    if mass_partition is None
                    else {
                        "positiveLotMassPartitionsKg": {
                            name: [_number(value) for value in parts]
                            for name, parts in partitions.items()
                        }
                    }
                ),
                "oldSourceRatiosUsed": False,
                "publicAllocationsInferred": False,
            }

        shipment_group = ("shipment", None, "shipment")
        composition_rows: dict[int, dict[str, Any]] = {}
        child_composition = any(
            region.get("scope", {}).get("kind") == "container" and _has_composition([region])
            for region in runtime_regions
        )
        if child_composition:
            for row_index in range(len(draw["allocations"])):
                local_draw = self._container_draw(draw, row_index)
                local_certificate = {**certificate, "regions": [], "wordingOwners": []}
                local_blueprint = replace(
                    blueprint,
                    ownership_data={
                        **blueprint.ownership_data,
                        NUMERIC_LEDGER_OWNER: {"certificate": local_certificate},
                    },
                )
                composition_rows[row_index] = _build_scope_ledger(
                    local_blueprint,
                    local_draw,
                    typed_plan,
                    _digest([seed, "container-composition", row_index]),
                    variant,
                )
        groups: dict[tuple[str, int | None, str], list[tuple[int, dict[str, Any]]]] = {}
        for index, region in enumerate(runtime_regions):
            scope = region.get("scope", {"kind": "shipment"})
            # A composition-bearing shipment region must consume the shipment's
            # typed packing plan. Its factGroup identifies correlated private
            # facts, not an independently allocatable child scope.
            group = (
                shipment_group
                if scope["kind"] == "shipment"
                and (
                    _has_composition([region])
                    or region["role"] in {"mixed_package_equation", "mixed_package_ranges"}
                )
                else (
                    scope["kind"],
                    scope.get("index"),
                    region.get("factGroup", "shipment"),
                )
            )
            groups.setdefault(group, []).append((index, region))
        groups.setdefault(shipment_group, [])
        complete: dict[str, Any] | None = None
        scope_ledgers: dict[str, dict[str, Any]] = {}
        for group, regions in groups.items():
            kind, row_index, _ = group
            # Only packageQuantity is scoped below. Copy that branch and keep
            # the immutable sampled facts shared; copying the whole draw is a
            # measurable hot-path cost and provides no additional isolation.
            local_draw = {**draw, "package": dict(draw["package"])}
            if kind == "container":
                if row_index is None:
                    raise ValueError("container scope is missing its row index")
                local_draw = self._container_draw(draw, row_index)
            elif kind == "shared_container":
                local_draw["package"]["packageQuantity"] = shared_facts["rowQuantity"]
            local_certificate = dict(certificate)
            local_certificate["regions"] = [region for _, region in regions]
            local_certificate["wordingOwners"] = (
                certificate["wordingOwners"] if group == shipment_group else []
            )
            local_ownership = {
                **blueprint.ownership_data,
                NUMERIC_LEDGER_OWNER: {
                    **blueprint.ownership_data[NUMERIC_LEDGER_OWNER],
                    "certificate": local_certificate,
                },
            }
            local_blueprint = replace(blueprint, ownership_data=local_ownership)
            local_seed: int | str = seed if group == shipment_group else _digest([seed, group])
            local_plan = dict(typed_plan)
            composition_facts = None
            if child_composition and group == shipment_group:
                quantities = [value["relatedQuantity"] for value in composition_rows.values()]
                supports = [
                    value["facts"].get("supportEnvelope") for value in composition_rows.values()
                ]
                support = None
                if any(value is not None for value in supports):
                    _require(
                        all(value is not None for value in supports),
                        "incomplete container support authority",
                    )
                    support = {
                        **supports[0],
                        "quantity": sum(v["quantity"] for v in supports),
                        "totalTareKg": _number(sum(_decimal(v["totalTareKg"]) for v in supports)),
                        "packingBudgetKg": _number(
                            sum(_decimal(v["packingBudgetKg"]) for v in supports)
                        ),
                    }
                composition_facts = {
                    "relatedQuantity": None
                    if all(q is None for q in quantities)
                    else sum(quantities),
                    "supportEnvelope": support,
                }
            if group != shipment_group:
                if _has_composition(local_certificate["regions"]):
                    _require(kind == "container", "composition requires explicit container scope")
                    row_ledger = composition_rows[row_index]
                    composition_facts = {
                        "relatedQuantity": row_ledger["relatedQuantity"],
                        "supportEnvelope": row_ledger["facts"].get("supportEnvelope"),
                    }
                else:
                    local_plan = {"privatePackingRelation": "NONE", "relatedPackageType": None}
            ledger = _build_scope_ledger(
                local_blueprint,
                local_draw,
                local_plan,
                local_seed,
                variant,
                nominal_capacity_litres=nominal_capacity_litres,
                capacity_authority=capacity_authority,
                composition_facts=composition_facts,
            )
            ledger["surfaces"] = {
                f"{NUMERIC_REGION_PREFIX}:{global_index}": ledger["surfaces"][
                    f"{NUMERIC_REGION_PREFIX}:{local_index}"
                ]
                for local_index, (global_index, _) in enumerate(regions)
            }
            if group == shipment_group:
                complete = ledger
            scope_ledgers[str(group)] = ledger
        if complete is None:
            raise AssertionError("shipment ledger was not produced")
        complete["scopeLedgers"] = {
            key: value for key, value in scope_ledgers.items() if key != str(shipment_group)
        }
        for child in complete["scopeLedgers"].values():
            if set(complete["surfaces"]) & set(child["surfaces"]):
                raise AssertionError("private scope surfaces overlap")
            complete["surfaces"].update(child["surfaces"])

        batches: dict[str, list[tuple[int, dict[str, Any]]]] = {}
        for index, region in enumerate(certificate["regions"]):
            if "rangeBatch" not in region:
                continue
            if region["role"] != "inclusive_range" or region["scope"]["kind"] != "container":
                raise ValueError("shared range batches require container inclusive ranges")
            batches.setdefault(region["rangeBatch"], []).append((index, region))
        for batch, regions in batches.items():
            patterns = {(region["prefixPattern"], region["digits"]) for _, region in regions}
            if len(patterns) != 1:
                raise ValueError("range batch has conflicting prefix/digit contracts")
            counts = {
                region["scope"]["index"]: draw["allocations"][region["scope"]["index"]][
                    "packageQuantity"
                ]
                for _, region in regions
            }
            quantity = sum(counts.values())
            pattern, digits = next(iter(patterns))
            if quantity > 9 * 10 ** (digits - 1):
                raise CandidateIncompatibility("shared range exceeds printed digit width")
            rng = random.Random(
                _digest([seed, blueprint.document_id, variant, "range-batch", batch])
            )
            common = _inclusive_range(quantity, rng, prefix_pattern=pattern, digits=digits)
            cursor = common["start"]
            facts: dict[str, dict[str, Any]] = {}
            for row_index, count in counts.items():
                facts[str(row_index)] = {
                    **common,
                    "start": cursor,
                    "end": cursor + count - 1,
                    "quantity": count,
                }
                cursor += count
            for index, region in regions:
                row_index = region["scope"]["index"]
                fact = facts[str(row_index)]
                surface = (
                    f"{_noun(draw['internalCategory'], 1)} NOS. {fact['prefix']}"
                    f"{fact['start']:0{digits}d} - {fact['prefix']}{fact['end']:0{digits}d}"
                )
                key = f"{NUMERIC_REGION_PREFIX}:{index}"
                complete["surfaces"][key] = [surface]
                scope_key = str(("container", row_index, region.get("factGroup", "shipment")))
                scope = complete["scopeLedgers"][scope_key]
                scope["surfaces"][key] = [surface]
                scope["facts"]["inclusiveRange"] = fact
            complete["facts"]["rangeBatch:" + batch] = {
                "quantity": quantity,
                "rows": facts,
                "sourceEndpointOrderOrRatioUsed": False,
            }
        complete["certificateSha256"] = _digest(certificate)
        complete["certificateFileSha256"] = self.file_sha256
        if shared_facts:
            complete["facts"]["sharedContainerFacts"] = shared_facts
        return complete

    @staticmethod
    def _container_draw(draw: Mapping[str, Any], index: int) -> dict[str, Any]:
        allocation = draw["allocations"][index]
        quantity = allocation["packageQuantity"]
        result = {
            **draw,
            "package": {**draw["package"], "packageQuantity": quantity},
            "physical": {
                **draw["physical"],
                "grossWeight": {"value": allocation["grossKg"], "unit": "kilogram"},
            },
            "allocations": [allocation],
            "provenance": dict(draw["provenance"]),
        }
        if PREPARED_SUPPORT_KEY in result["provenance"]:
            prepared = dict(result["provenance"][PREPARED_SUPPORT_KEY])
            prepared.update(
                countedQuantity=quantity,
                quantity=quantity,
                totalTareKg=_number(_decimal(prepared["unitTareKg"]) * quantity),
                packingBudgetKg=_number(_decimal(prepared["unitTareKg"]) * quantity),
            )
            result["provenance"][PREPARED_SUPPORT_KEY] = prepared
        return result

    @staticmethod
    def validate_composed_capacity(
        draw: Mapping[str, Any], physical_rows: Sequence[Mapping[str, Any]]
    ) -> None:
        _validate_composed_capacity(draw, physical_rows)

    @staticmethod
    def project_wording(
        values: Mapping[str, str], ledger: Mapping[str, Any]
    ) -> tuple[dict[str, str], dict[str, Any]]:
        return project_wording(values, ledger)

    @staticmethod
    def wording_context(ledger: Mapping[str, Any]) -> str:
        return wording_context(ledger)
