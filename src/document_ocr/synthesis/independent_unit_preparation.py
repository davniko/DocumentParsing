"""Prepare native physical profiles for selected independent campaigns.

This is a separate preparation stage: changing physical authority cannot alter
the earlier goods/package decisions or silently invalidate their wire receipts.
Nested and mixed accounting use complete-group facts before freezing totals.
"""

from __future__ import annotations

import copy
import time
from collections.abc import Collection, Mapping
from pathlib import Path
from typing import Any

from document_ocr.atomic import atomic_publish_json
from document_ocr.synthesis import independent_article_profiles as articles
from document_ocr.synthesis import independent_composed_profiles as composed
from document_ocr.synthesis.independent_campaign import IndependentCampaign, unit_profile_capacities
from document_ocr.synthesis.independent_dg_context import physical_form
from document_ocr.synthesis.independent_preparation import (
    _complete_groups,
    _evidence,
    _load_pin,
    _load_source_context,
    _native_batch,
    _publish,
    _runnable_config,
)
from document_ocr.synthesis.independent_unit_profiles import (
    STAGE,
    SYSTEM,
    UnitProfileRequest,
    source_unit_profile_precision,
    unit_profile_output_type,
    unit_profile_prompt,
)


def minimum_groups_per_load(certificate: dict[str, Any], relation: str) -> int:
    """Positive owned component rows constrain units before estimating mass.

    A fixed-outer CONTAINS/SUPPORTS group has one public counted unit. Mixed
    and ON_SUPPORT group cardinalities are proposed later, so their native
    topology and complete-plan checks remain authoritative.
    """
    if relation in {"MIXED", "ON_SUPPORT"}:
        return 1
    loads = max(1, certificate.get("sourceContainerCount", 0))
    minimum = 1
    for region in certificate.get("regions", []):
        if region["role"] not in {"component_package", "component_measurement"}:
            continue
        parts = region["componentCount"]
        if region.get("scope", {}).get("kind", "shipment") == "shipment":
            parts = (parts + loads - 1) // loads
        minimum = max(minimum, parts)
    return minimum


async def prepare_unit_profiles(
    root: Path,
    config: dict[str, Any],
    *,
    source_ids: Collection[str] | None = None,
    retry_rate_limited: bool = False,
    retry_invalid_output: bool = False,
    retry_failed_receipt_sha256s: Collection[str] = (),
    review_feedback: Mapping[str, str] | None = None,
) -> dict[str, Any]:
    """Publish all-or-nothing source groups with immutable received evidence."""
    context = _load_source_context(root, config)
    campaign = context.source
    campaign.calls.retry_rate_limited = retry_rate_limited
    campaign.calls.retry_invalid_output = retry_invalid_output
    campaign.calls.retry_failed_receipt_sha256s = frozenset(retry_failed_receipt_sha256s)
    requests = copy.deepcopy(_load_pin(root, config["independent"]["requests"]))
    plans = _load_pin(root, config["independent"]["private_plans"])
    selected = set(campaign.source_ids if source_ids is None else source_ids)
    if not selected or not selected <= set(campaign.source_ids):
        raise ValueError("unit-profile source scope must be nonempty and selected")
    jobs, topology_jobs, inventory = {}, {}, {}
    if review_feedback is not None and not set(review_feedback) <= {
        key for key, row in requests.items() if row["sourceDocumentId"] in selected
    }:
        raise ValueError("numeric review feedback must belong to the selected request scope")
    # Preflight ALL branches before spending. Unsupported compositions are
    # errors, not silent uses of the fixed-family planner in a variable cohort.
    for key, frozen in requests.items():
        if frozen["sourceDocumentId"] not in selected:
            inventory[key] = {
                "sourceDocumentId": frozen["sourceDocumentId"],
                "variant": frozen["variant"],
                "status": "ready",
            }
            continue
        if any(
            k in frozen
            for k in (
                "nativeUnitProfileEvidence",
                "nativeComposedProfileEvidence",
                "nativeArticleProfileEvidence",
            )
        ):
            raise ValueError("unit profiles are already prepared")
        sid = frozen["sourceDocumentId"]
        # Preflight all precision owners before any paid request. Explicit
        # reviewed precision remains authoritative where it already exists.
        precision = context.certificates[sid].get("unitProfilePrecision")
        if precision is None:
            precision = source_unit_profile_precision(
                campaign.blueprint(sid), context.certificates[sid]["regions"]
            )
        frozen["unitProfilePrecision"] = precision
        capacities = unit_profile_capacities(frozen["equipmentSlots"], campaign.catalog.limits)
        minimum_groups = minimum_groups_per_load(
            context.certificates[sid], plans[key]["privatePackingRelation"]
        )
        if frozen.get("cargoPresentation", {}).get("articleGroups") is not None:
            # The native example estimates the whole explicit manifest, not
            # each component. Reserving each component again would divide the
            # carrying capacity twice.
            minimum_groups = 1
        values = dict(
            goods=frozen["goods"],
            regime=frozen["regime"],
            selectedCategory=frozen["selectedCategory"],
            countedUnitContext=frozen["countedUnitContext"],
            equipment=frozen["equipmentSlots"],
            payloadKg=float(min(float(row["payloadKg"]) for row in capacities)) / minimum_groups,
            volumeBudgetM3=float(min(float(row["volumeM3"]) for row in capacities))
            / minimum_groups,
            volumeBasis=(
                "selected_hardware_capacity"
                if frozen["equipmentSlots"]
                and all(
                    row["typeCategory"] not in {"OPEN_TOP", "PLATFORM"}
                    for row in frozen["equipmentSlots"]
                )
                else "explicit_private_planning_budget_not_enclosed_hardware_rating"
            ),
            dangerousGoods=frozen.get("dangerousGoods"),
            physicalState=(
                physical_form(frozen, context.sampler.hmt_states)
                if frozen["regime"] == "dg"
                else None
            ),
        )
        article_groups = frozen.get("cargoPresentation", {}).get("articleGroups")
        if article_groups is not None:
            values["articleGroups"] = article_groups
        if minimum_groups > 1:
            values["volumeBasis"] += (
                f"; per-group ceiling reserves {minimum_groups} counted groups for "
                "positive source-owned component rows, not extra hardware"
            )
        if review_feedback and key in review_feedback:
            values["reviewFeedback"] = review_feedback[key]
        is_composed = plans[key]["privatePackingRelation"] != "NONE"
        if is_composed:
            request = composed.TopologyRequest(
                **values, privatePacking=composed.packing_roles(plans[key])
            )
            model = composed.topology_output_type(plans[key]["privatePackingRelation"])
            topology_jobs[key] = (
                composed.TOPOLOGY_STAGE,
                model,
                composed.TOPOLOGY_SYSTEM,
                composed.topology_prompt(request),
                model.model_validate,
                request.model_dump(mode="json"),
            )
        else:
            request = UnitProfileRequest(**values)
            is_article = request.articleGroups is not None
            model = (
                articles.article_output_type(request)
                if is_article
                else unit_profile_output_type(request)
            )
            jobs[key] = (
                articles.STAGE if is_article else STAGE,
                model,
                articles.SYSTEM if is_article else SYSTEM,
                articles.article_prompt(request) if is_article else unit_profile_prompt(request),
                (lambda value, req=request: articles.normalize_article_profile(req, value))
                if is_article
                else model.model_validate,
                request.model_dump(mode="json"),
            )
        inventory[key] = {
            "sourceDocumentId": frozen["sourceDocumentId"],
            "variant": frozen["variant"],
            "status": "scheduled",
        }
    # The wire calls are content-addressed and replayed, but an invocation's
    # elapsed times and retry outcomes are not immutable request identities.
    # Keep each invocation journal, as ordinary native preparation already does.
    directory = campaign.output / "unit-profiles" / str(time.time_ns())
    if topology_jobs:
        topology_results = await _native_batch(
            campaign.calls, topology_jobs, directory / "topologies"
        )
        for key, result in topology_results.items():
            if result["error"] is not None:
                inventory[key].update(status="held_native_topology", reason=result["error"])
                continue
            if result["output"]["topology"] is None:
                inventory[key].update(
                    status="held_unknown_topology", reason=result["output"]["basis"]
                )
                continue
            topology_request = composed.TopologyRequest.model_validate(topology_jobs[key][-1])
            requests[key]["nativeCompositionTopologyEvidence"] = _evidence(topology_request, result)
            request = composed.ComposedRequest(
                **topology_request.model_dump(mode="json"),
                topology=result["output"]["topology"],
            )
            model = composed.composed_output_type(request)
            jobs[key] = (
                composed.STAGE,
                model,
                composed.SYSTEM,
                composed.composed_prompt(request),
                lambda value, req=request: composed.validate_composed(req, value),
                request.model_dump(mode="json"),
            )
    results = await _native_batch(campaign.calls, jobs, directory)
    for key, result in results.items():
        if result["error"] is not None:
            inventory[key].update(status="held_native_profile", reason=result["error"])
        elif result["output"]["unknown"]:
            inventory[key].update(status="held_unknown_profile", reason=result["output"]["basis"])
        else:
            is_composed = plans[key]["privatePackingRelation"] != "NONE"
            request = (
                composed.ComposedRequest if is_composed else UnitProfileRequest
            ).model_validate(jobs[key][-1])
            evidence_key = (
                "nativeComposedProfileEvidence"
                if is_composed
                else "nativeArticleProfileEvidence"
                if request.articleGroups is not None
                else "nativeUnitProfileEvidence"
            )
            requests[key][evidence_key] = _evidence(request, result)
            inventory[key]["status"] = "ready"
    groups = _complete_groups(campaign.variant_counts, inventory)
    runnable = _runnable_config(root, config, groups, requests, plans, directory)
    if runnable is not None:
        combined = IndependentCampaign(root, runnable)
        for sid, count in groups.items():
            for variant in range(1, count + 1):
                key = f"{sid}:{variant}"
                try:
                    combined.plan(sid, variant)
                except Exception as exception:
                    inventory[key].update(
                        status="held_combined_plan",
                        reason=f"{type(exception).__name__}: {exception}",
                    )
        groups = _complete_groups(campaign.variant_counts, inventory)
        runnable = _runnable_config(root, config, groups, requests, plans, directory)
    report = {
        "inventory": inventory,
        "readySources": len(groups),
        "readySlots": sum(groups.values()),
        "plannedSlots": len(requests),
        "config": runnable,
        "preparedConfig": None
        if runnable is None
        else _publish(root, directory / "prepared-config.json", runnable),
        "datasetMutation": False,
        "spentAfterUsd": str(campaign.calls.spent),
    }
    atomic_publish_json(directory / "summary.json", report)
    return report
