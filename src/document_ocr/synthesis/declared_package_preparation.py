"""Native physical authorities for disjoint public package groups.

Goods, equipment and DG declarations remain frozen. A group may select another
received plausible package alternative, but never an unapproved category or a
source commodity donor. Every additional call retains ordinary Runner evidence.
"""

from __future__ import annotations

import copy
import time
from pathlib import Path

from document_ocr.synthesis.curated import digest
from document_ocr.synthesis.independent_accounting import (
    MIXED_PACKING_SYSTEM,
    MixedPackingPlan,
    MixedPackingRequest,
    mixed_packing_facts,
    mixed_packing_prompt,
)
from document_ocr.synthesis.independent_packages import (
    PackageChoice,
    PackageChoiceRequest,
    validate_package_choice,
    validate_shared_native_choice,
)
from document_ocr.synthesis.independent_relationships import (
    PRIVATE_PACKING_SYSTEM,
    PrivatePackingRequest,
    private_packing_facts,
    private_packing_output_type,
    private_packing_prompt,
    validate_private_packing_plan,
)
from document_ocr.synthesis.independent_unit_profiles import (
    STAGE,
    SYSTEM,
    UnitProfileRequest,
    unit_profile_output_type,
    unit_profile_prompt,
    verify_profile_evidence,
)


def selection(policy: dict, sid: str, variant: int) -> int:
    values = policy["group_counts"]
    if (
        not isinstance(values, list)
        or not values
        or any(type(v) is not int or v < 1 for v in values)
    ):
        raise ValueError("declared package policy requires positive explicit group counts")
    per_source = policy.get("source_group_counts", {})
    if sid in per_source:
        values = per_source[sid]
        if (
            not isinstance(values, list)
            or not values
            or any(type(v) is not int or v < 1 for v in values)
        ):
            raise ValueError("source package group counts must be positive integers")
    return values[(variant - 1) % len(values)]


def _private_request(
    campaign, request: dict, plan: dict
) -> PrivatePackingRequest | MixedPackingRequest | None:
    # A native NONE decision is specific to its counted category. Only a source
    # certified as having no private relationship can reuse NONE for a new type.
    if plan["origin"] == "reviewed_no_private_relation":
        return None
    context = dict(request["nativeChoiceEvidence"]["request"])
    context.pop("requiredCategory", None)
    if plan["privatePackingRelation"] == "MIXED":
        return MixedPackingRequest.model_validate(
            {**context, "countedCategory": request["selectedCategory"]}
        )
    return PrivatePackingRequest.model_validate(
        {
            **context,
            "countedCategory": request["selectedCategory"],
            "allowedRelations": list(
                campaign.numeric.allowed_private_relations(request["sourceDocumentId"])
            ),
        }
    )


async def prepare(root: Path, config: dict) -> dict:
    """Expand a completely prepared campaign, retaining explicit source holds."""
    from document_ocr.synthesis import independent_composed_profiles as composed
    from document_ocr.synthesis.independent_campaign import IndependentCampaign
    from document_ocr.synthesis.independent_preparation import (
        _complete_groups,
        _evidence,
        _load_pin,
        _native_batch,
        _publish,
        _runnable_config,
        _source_pins,
        _verify_pins,
    )

    base_config = copy.deepcopy(config)
    base_config.pop("declared_packages")
    campaign = IndependentCampaign(root, base_config)
    pins = _source_pins(root, config, campaign.source_ids)
    policy = config["declared_packages"]
    contracts = _load_pin(root, policy["contracts"])["sources"]
    directory = campaign.output / "declared-package-preparation" / str(time.time_ns())
    requests = copy.deepcopy(campaign.requests)
    plans = copy.deepcopy(campaign.private_plans)
    inventory, selected = {}, {}
    for key, frozen in requests.items():
        sid, variant = frozen["sourceDocumentId"], frozen["variant"]
        count = selection(policy, sid, variant)
        inventory[key] = {"sourceDocumentId": sid, "variant": variant, "status": "ready"}
        if count == 1:
            continue
        if "declaredPackageGroups" in frozen:
            raise ValueError("declared package groups are already prepared")
        contract = contracts[sid]
        mode = contract["planning"]
        if mode not in {"independent_groups", "shared_units"}:
            raise ValueError("source has no certified multi-row planning mode")
        if mode == "shared_units":
            frozen["declaredPackageGroups"] = {"mode": mode, "count": count}
            continue
        answer = validate_package_choice(
            PackageChoiceRequest.model_validate(frozen["nativeChoiceEvidence"]["request"]),
            frozen["nativeChoiceEvidence"]["output"],
        )
        alternatives = sorted(answer.package_types, key=lambda c: digest([config["seed"], key, c]))
        if not alternatives:
            raise ValueError("UNKNOWN package alternatives cannot become declared groups")
        # The frozen first group already has native physical authority. Reuse
        # that exact evidence rather than paying to propose and then discard it.
        alternatives = [
            frozen["selectedCategory"],
            *(value for value in alternatives if value != frozen["selectedCategory"]),
        ]
        if plans[key]["privatePackingRelation"] == "MIXED":
            # Each declared group keeps the source's generic accounting unit.
            # Its support and loose-package types are chosen independently by
            # the mixed-accounting call, not substituted for the public total.
            alternatives = [frozen["selectedCategory"]]
        selected[key] = []
        for index in range(count):
            child = copy.deepcopy(frozen)
            if index:
                for field in (
                    "nativeUnitProfileEvidence",
                    "nativeComposedProfileEvidence",
                    "nativeCompositionTopologyEvidence",
                    "nativeArticleProfileEvidence",
                ):
                    child.pop(field, None)
            category = alternatives[index % len(alternatives)]
            child.update(
                selectedCategory=category, countedUnitContext=answer.counting_contexts.get(category)
            )
            selected[key].append({"request": child, "plan": copy.deepcopy(plans[key])})

    for index in range(1, max((len(v) for v in selected.values()), default=0)):
        jobs, private_requests = {}, {}
        for key, children in selected.items():
            if index >= len(children) or inventory[key]["status"] != "ready":
                continue
            child = children[index]
            request, plan = child["request"], child["plan"]
            native = _private_request(campaign, request, plan)
            if native is None:
                continue
            if isinstance(native, MixedPackingRequest):
                private_requests[key] = native
                jobs[key] = (
                    "independent-mixed-accounting",
                    MixedPackingPlan,
                    MIXED_PACKING_SYSTEM,
                    mixed_packing_prompt(native, hs_codes=campaign.sampler.phrases),
                    MixedPackingPlan.model_validate,
                    native.model_dump(mode="json"),
                )
                continue
            model = private_packing_output_type(native)
            private_requests[key] = native
            jobs[key] = (
                "independent-private-packing",
                model,
                PRIVATE_PACKING_SYSTEM,
                private_packing_prompt(native, hs_codes=campaign.sampler.phrases),
                lambda value, req=native: validate_private_packing_plan(req, value),
                native.model_dump(mode="json"),
            )
        results = await _native_batch(campaign.calls, jobs, directory / str(index) / "private")
        for key, result in results.items():
            if result["error"] is not None:
                inventory[key].update(status="held_declared_private_plan", reason=result["error"])
                continue
            native = private_requests[key]
            mixed = isinstance(native, MixedPackingRequest)
            answer = (
                MixedPackingPlan.model_validate(result["output"])
                if mixed
                else validate_private_packing_plan(native, result["output"])
            )
            if answer.unknown:
                inventory[key].update(status="held_declared_private_plan", reason="UNKNOWN")
                continue
            selected[key][index]["plan"] = {
                **(mixed_packing_facts(answer) if mixed else private_packing_facts(answer)),
                "origin": "native_mixed_accounting" if mixed else "native_typed_private_plan",
                "nativePlanPacketSha256": result["packetSha256"],
                "nativePlanEvidence": _evidence(native, result),
            }

        topologies, jobs = {}, {}
        for key, children in selected.items():
            if index >= len(children) or inventory[key]["status"] != "ready":
                continue
            frozen = requests[key]
            child = children[index]
            request, plan = child["request"], child["plan"]
            evidence = frozen.get(
                "nativeUnitProfileEvidence", frozen.get("nativeComposedProfileEvidence")
            )
            if evidence is None:
                raise ValueError(
                    "independent declared groups require native unit/composed profiles"
                )
            values = {
                k: v
                for k, v in evidence["request"].items()
                if k not in {"topology", "privatePacking", "packingLevels"}
            }
            values.update(
                selectedCategory=request["selectedCategory"],
                countedUnitContext=request["countedUnitContext"],
                volumeBasis=values["volumeBasis"]
                + f"; this unit belongs to one of {len(children)} disjoint public groups; "
                "joint membership and total capacity are checked by the host",
            )
            if plan["privatePackingRelation"] == "NONE":
                native = UnitProfileRequest.model_validate(values)
                model = unit_profile_output_type(native)
                jobs[key] = (
                    STAGE,
                    model,
                    SYSTEM,
                    unit_profile_prompt(native),
                    model.model_validate,
                    native.model_dump(mode="json"),
                )
            else:
                from document_ocr.synthesis.independent_numeric import (
                    packing_contents_kind,
                    required_packing_levels,
                )

                native = composed.TopologyRequest(
                    **values,
                    privatePacking=composed.packing_roles(plan),
                    packingLevels=required_packing_levels(
                        campaign.numeric.certificate(request["sourceDocumentId"])
                    ),
                    innerContentsKind=packing_contents_kind(
                        campaign.numeric.certificate(request["sourceDocumentId"])
                    ),
                )
                model = composed.topology_output_type(
                    plan["privatePackingRelation"], native.packingLevels, native.innerContentsKind
                )
                topologies[key] = (
                    composed.TOPOLOGY_STAGE,
                    model,
                    composed.TOPOLOGY_SYSTEM,
                    composed.topology_prompt(native),
                    model.model_validate,
                    native.model_dump(mode="json"),
                )
        results = await _native_batch(
            campaign.calls, topologies, directory / str(index) / "topology"
        )
        for key, result in results.items():
            if result["error"] is not None or result["output"]["topology"] is None:
                inventory[key].update(
                    status="held_declared_topology", reason=result["error"] or "UNKNOWN"
                )
                continue
            request = composed.TopologyRequest.model_validate(topologies[key][-1])
            selected[key][index]["request"]["nativeCompositionTopologyEvidence"] = _evidence(
                request, result
            )
            native = composed.ComposedRequest(
                **request.model_dump(mode="json"), topology=result["output"]["topology"]
            )
            model = composed.composed_output_type(native)
            jobs[key] = (
                composed.STAGE,
                model,
                composed.SYSTEM,
                composed.composed_prompt(native),
                lambda value, req=native: composed.validate_composed(req, value),
                native.model_dump(mode="json"),
            )
        results = await _native_batch(campaign.calls, jobs, directory / str(index) / "profiles")
        for key, result in results.items():
            if result["error"] is not None or result["output"]["unknown"]:
                inventory[key].update(
                    status="held_declared_profile", reason=result["error"] or "UNKNOWN"
                )
                continue
            is_composed = selected[key][index]["plan"]["privatePackingRelation"] != "NONE"
            native = (
                composed.ComposedRequest if is_composed else UnitProfileRequest
            ).model_validate(jobs[key][-1])
            selected[key][index]["request"][
                "nativeComposedProfileEvidence" if is_composed else "nativeUnitProfileEvidence"
            ] = _evidence(native, result)
    for key, children in selected.items():
        if inventory[key]["status"] == "ready":
            requests[key]["declaredPackageGroups"] = {
                "mode": "independent_groups",
                "count": len(children),
                "groups": children,
            }
    groups = _complete_groups(campaign.variant_counts, inventory)
    runnable = _runnable_config(root, config, groups, requests, plans, directory)
    if runnable is not None:
        combined = IndependentCampaign(root, runnable)
        for sid, count in groups.items():
            for variant in range(1, count + 1):
                try:
                    combined.preflight([combined.plan(sid, variant)])
                except (ValueError, KeyError, AssertionError) as error:
                    inventory[f"{sid}:{variant}"].update(
                        status="held_declared_combined_plan",
                        reason=f"{type(error).__name__}: {error}",
                    )
        groups = _complete_groups(campaign.variant_counts, inventory)
        runnable = _runnable_config(root, config, groups, requests, plans, directory)
    _verify_pins(pins)
    pin = None if runnable is None else _publish(root, directory / "prepared-config.json", runnable)
    report = {
        "config": runnable,
        "preparedConfig": pin,
        "inventory": inventory,
        "readySources": len(groups),
        "readySlots": sum(groups.values()),
        "spentAfterUsd": str(campaign.calls.spent),
    }
    _publish(root, directory / "summary.json", report)
    return report


def verify(campaign, frozen: dict, authority: dict, choice: PackageChoice) -> None:
    """Verify every received group proposal against the same frozen shipment."""
    from document_ocr.synthesis.independent_composed_profiles import verify_composed_evidence
    from document_ocr.synthesis.independent_relationships import verify_native_plan_evidence

    groups = authority["groups"]
    if authority["count"] != len(groups) or len(groups) < 2:
        raise ValueError("declared group evidence count differs")
    for child in groups:
        request, plan = child["request"], child["plan"]
        for key in (
            "goods",
            "regime",
            "equipmentSlots",
            "dangerousGoods",
            "sourceDocumentId",
            "variant",
        ):
            if request[key] != frozen[key]:
                raise ValueError("declared group changed frozen " + key)
        campaign.sampler.validate_request(request)
        validate_shared_native_choice(frozen, request, choice)
        verify_native_plan_evidence(
            campaign.root,
            request,
            plan,
            provider=campaign.calls.config["provider"],
            hs_codes=campaign.sampler.phrases,
            allowed_relations=campaign.numeric.allowed_private_relations(
                frozen["sourceDocumentId"]
            ),
            source_certificate_sha256=digest(
                campaign.numeric.certificate(frozen["sourceDocumentId"])
            ),
        )
        if plan["privatePackingRelation"] == "NONE":
            verify_profile_evidence(
                campaign.root,
                request,
                provider=campaign.calls.config["provider"],
                physical_states=campaign.sampler.hmt_states,
            )
        else:
            verify_composed_evidence(
                campaign.root,
                request,
                plan,
                provider=campaign.calls.config["provider"],
                physical_states=campaign.sampler.hmt_states,
            )
