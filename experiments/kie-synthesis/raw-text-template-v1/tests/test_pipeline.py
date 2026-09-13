from __future__ import annotations

import asyncio
import json
from datetime import UTC, datetime
from decimal import Decimal
from pathlib import Path
from typing import Any

import pytest
from document_ocr.hashing import sha256_bytes

from raw_text_template_experiment import pipeline as pipeline_module
from raw_text_template_experiment.agents import empty_usage
from raw_text_template_experiment.host import (
    SpanDraft,
    anchor_drafts,
    inventory_binding_id,
    risk_candidates,
)
from raw_text_template_experiment.models import (
    AgentStageArtifact,
    CarrierAssessment,
    CompilerAgentOutput,
    CriticAgentOutput,
    ExtractionCaseResult,
)
from raw_text_template_experiment.pipeline import (
    _compiler_host_rejection,
    _compiler_payload,
    _critic_launch_decision,
    _critic_payload,
    _critic_review_candidates,
    _critic_review_history,
    _critic_stagnation_reason,
    _draft_inventory,
    _extract_case,
    _latest_compiler_revision_context,
    _literal_review_line_ids,
    _replay_checkpoint_case,
    _restore_or_replay_resume_state,
    _restore_state_checkpoint,
    _state_checkpoint,
    load_config,
    run_extraction,
)

_EXPERIMENT_ROOT = Path(__file__).resolve().parents[1]


def test_cost_threshold_allows_exactly_one_required_state_confirmation() -> None:
    allowed, used = _critic_launch_decision(
        spent=Decimal("0.35"),
        threshold=Decimal("0.35"),
        has_unconfirmed_state=True,
        post_threshold_confirmation_used=False,
    )
    assert (allowed, used) == (True, True)
    assert _critic_launch_decision(
        spent=Decimal("0.36"),
        threshold=Decimal("0.35"),
        has_unconfirmed_state=True,
        post_threshold_confirmation_used=used,
    ) == (False, True)
    assert _critic_launch_decision(
        spent=Decimal("0.35"),
        threshold=Decimal("0.35"),
        has_unconfirmed_state=False,
        post_threshold_confirmation_used=False,
    ) == (False, False)


def test_critic_stagnation_requires_two_consecutive_identical_host_rejections() -> None:
    passed = CriticAgentOutput.model_validate(
        {
            "verdict": "pass",
            "findings": (),
            "additional_bindings": (),
            "rationale": "Fixture output.",
        }
    )

    def rejected(pass_number: int, message: str) -> AgentStageArtifact:
        return _stage("critic", pass_number, passed).model_copy(
            update={
                "status": "host_rejected",
                "error_type": "ValueError",
                "error_message": message,
            }
        )

    first = rejected(1, "functional no-op")
    assert _critic_stagnation_reason((first,)) is None
    assert _critic_stagnation_reason((first, rejected(2, "different defect"))) is None
    reason = _critic_stagnation_reason((first, rejected(2, "functional no-op")))
    assert reason is not None
    assert "two consecutive identical" in reason


def test_critic_review_history_carries_progress_but_no_stale_findings() -> None:
    revision = CriticAgentOutput.model_validate(
        {
            "verdict": "revise",
            "findings": (
                {
                    "finding_kind": "incorrect_static_classification",
                    "line_ids": ("L00003",),
                    "evidence": "VAT: 12345678",
                    "explanation": "The VAT identifier must be regenerated.",
                },
            ),
            "remove_inventory_binding_ids": (),
            "additional_bindings": (),
            "rationale": "Fixture revision.",
        }
    )
    accepted = _stage("critic", 1, revision)
    rejected = _stage("critic", 2, revision).model_copy(
        update={
            "status": "host_rejected",
            "error_type": "ValueError",
            "error_message": "critic transaction is a functional no-op",
        }
    )

    ledger = _critic_review_history((accepted, rejected))

    assert ledger == {
        "completedStages": 2,
        "stageStatusCounts": {"host_rejected": 1, "success": 1},
        "successfulVerdictCounts": {"revise": 1},
        "appliedRevisionCount": 1,
        "currentStateRevision": 1,
        "latestStage": {
            "passNumber": 2,
            "hostStatus": "host_rejected",
            "providerVerdict": "revise",
            "stateChanged": False,
        },
        "carriedFindings": (),
        "interpretation": ledger["interpretation"],
    }
    serialized = json.dumps(ledger)
    assert "VAT: 12345678" not in serialized
    assert "The VAT identifier must be regenerated" not in serialized
    assert "functional no-op" not in serialized


def test_state_checkpoint_round_trips_exact_host_materialization() -> None:
    body = "Acme Ocean Lines Ltd.\nB/L No. ACME123456\n"
    raw = "--- PAGE 1 ---\n" + body
    document_id = "doc_checkpoint"
    carrier = "Acme Ocean Lines Ltd."
    source_target = {
        "documentPatch": {
            "billOfLadingNumber": "ACME123456",
            "parties": {"carrier": {"name": carrier}},
        }
    }

    def anchor(value: str, path: str, anchor_id: str) -> dict[str, Any]:
        start = body.index(value)
        return {
            "anchor_id": anchor_id,
            "document_id": document_id,
            "patchable": True,
            "page_number": 1,
            "page_start": start,
            "page_end": start + len(value),
            "raw_value": value,
            "target_value": value,
            "relation_target_path": path,
            "role_path": path,
            "surface_family": "text",
        }

    drafts = anchor_drafts(
        raw=raw,
        document_id=document_id,
        anchors=(
            anchor(carrier, "documentPatch.parties.carrier.name", "anchor_carrier"),
            anchor(
                "ACME123456",
                "documentPatch.billOfLadingNumber",
                "anchor_bill_number",
            ),
        ),
    )
    assessment = CarrierAssessment.model_validate(
        {
            "canonical_name": carrier,
            "aliases": (),
            "evidence_occurrences": (
                {
                    "line_start": "L00002",
                    "line_end": "L00002",
                    "source_text": carrier,
                    "occurrence_index": 0,
                },
            ),
            "source": "source_label_confirmed_by_ocr",
            "rationale": "The pinned carrier label is printed exactly.",
        }
    )
    checkpoint = _state_checkpoint(
        document_id=document_id,
        raw=raw,
        source_target=source_target,
        drafts=drafts,
        assessment=assessment,
        semantic_only_target_facts=(),
        critic_outputs=(),
    )

    restored, restored_assessment, semantic_only, reviews = _restore_state_checkpoint(
        checkpoint_payload=checkpoint.model_dump_json().encode("utf-8"),
        document_id=document_id,
        raw=raw,
        source_target=source_target,
    )

    assert restored == drafts
    assert restored_assessment == assessment
    assert semantic_only == ()
    assert reviews == []
    with pytest.raises(ValueError, match="source hash differs"):
        _restore_state_checkpoint(
            checkpoint_payload=checkpoint.model_dump_json().encode("utf-8"),
            document_id=document_id,
            raw=raw + "tampered",
            source_target=source_target,
        )


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("config_name", "phase"),
    (("development30.yaml", "development30"), ("transfer200.yaml", "transfer200")),
)
async def test_downstream_provider_launch_is_fail_closed_before_loading_inputs(
    config_name: str, phase: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    config_path = _EXPERIMENT_ROOT / "configs" / config_name
    config = load_config(config_path)
    locked = config.model_copy(
        update={
            "workflow": config.workflow.model_copy(update={"provider_launch_authorized": False})
        }
    )
    monkeypatch.setattr(pipeline_module, "load_config", lambda _path: locked)
    monkeypatch.setattr(
        pipeline_module,
        "_config_inputs",
        lambda *_args, **_kwargs: pytest.fail("locked run loaded extraction inputs"),
    )
    with pytest.raises(ValueError, match=f"provider launch is not authorized for {phase}"):
        await run_extraction(config_path)


def test_compiler_payload_separates_shared_equality_from_semantic_review() -> None:
    body = "Reference: SAME123456\n"
    raw = "--- PAGE 1 ---\n" + body
    value = "SAME123456"
    start = body.index(value)
    paths = ("documentPatch.first", "documentPatch.second")
    rows = tuple(
        {
            "anchor_id": f"anchor_{index}",
            "document_id": "doc_relationship",
            "patchable": True,
            "page_number": 1,
            "page_start": start,
            "page_end": start + len(value),
            "raw_value": value,
            "target_value": value,
            "relation_target_path": path,
            "role_path": path,
            "surface_family": "text",
        }
        for index, path in enumerate(paths)
    )
    anchors = anchor_drafts(raw=raw, document_id="doc_relationship", anchors=rows)
    feature = {
        "document_type": "bill_of_lading",
        "carrier_family": "FIXTURE",
        "template_proxy_id": "template_fixture",
        "page_count": 1,
        "container_count": 0,
        "goods_group_count": 0,
        "package_fact_count": 0,
        "dangerous_goods_count": 0,
        "temperature_setting_count": 0,
    }
    equal = {
        "documentPatch": {
            "first": value,
            "second": value,
            "negotiability": "non_negotiable",
        }
    }
    payload = _compiler_payload(
        document_id="doc_relationship",
        raw=raw,
        source_target=equal,
        feature=feature,
        anchors=anchors,
        risks=risk_candidates(raw, ()),
        prior_error=None,
        prior_output=None,
        prior_drafts=None,
        agent_contract_protocol="legacy_v1",
    )
    assert payload["anchorBindingsWithSharedEqualityConstraint"] == ["anchor_binding_0001"]
    assert payload["anchorBindingsWithCrossFactEqualityAmbiguity"] == ["anchor_binding_0001"]
    assert payload["anchorBindings"][0]["independentTargetFactComponents"] == (
        ("documentPatch.first",),
        ("documentPatch.second",),
    )
    assert payload["anchorBindingsRequiringSemanticReview"] == []
    assert "documentPatch.first" in payload["allowedTargetPaths"]
    assert "documentPatch.negotiability" in payload["allowedTargetPaths"]
    assert [row["targetPath"] for row in payload["semanticOnlyTargetFacts"]] == [
        "documentPatch.negotiability"
    ]
    reference_payload = _compiler_payload(
        document_id="doc_relationship",
        raw=raw,
        source_target=equal,
        feature=feature,
        anchors=anchors,
        risks=risk_candidates(raw, ()),
        prior_error=None,
        prior_output=None,
        prior_drafts=None,
        agent_contract_protocol="reference_compact_v3",
    )
    assert reference_payload["occurrenceCandidates"] == {
        "columns": (
            "occurrenceId",
            "lineStart",
            "lineEnd",
            "sourceText",
            "occurrenceIndex",
        ),
        "rows": (("compiler_occurrence_00001", "L00002", "L00002", value, 0),),
    }

    unequal = {
        "documentPatch": {
            "first": value,
            "second": "OTHER123456",
            "negotiability": "non_negotiable",
        }
    }
    payload = _compiler_payload(
        document_id="doc_relationship",
        raw=raw,
        source_target=unequal,
        feature=feature,
        anchors=anchors,
        risks=risk_candidates(raw, ()),
        prior_error=None,
        prior_output=None,
        prior_drafts=None,
        agent_contract_protocol="legacy_v1",
    )
    assert payload["anchorBindingsWithSharedEqualityConstraint"] == []
    assert payload["anchorBindingsWithCrossFactEqualityAmbiguity"] == []
    assert payload["anchorBindingsRequiringSemanticReview"] == ["anchor_binding_0001"]

    prior = CompilerAgentOutput.model_validate(
        {
            "carrier": {
                "canonical_name": "Fixture Carrier",
                "aliases": (),
                "evidence_occurrences": (
                    {
                        "line_start": "L00002",
                        "line_end": "L00002",
                        "source_text": value,
                        "occurrence_index": 0,
                    },
                ),
                "source": "source_label_confirmed_by_ocr",
                "rationale": "Payload preservation fixture.",
            },
            "anchor_overrides": (),
            "bindings": (),
            "unresolved": (),
            "all_shipment_dependent_surfaces_accounted_for": True,
        }
    )
    revised = _compiler_payload(
        document_id="doc_relationship",
        raw=raw,
        source_target=unequal,
        feature=feature,
        anchors=anchors,
        risks=risk_candidates(raw, ()),
        prior_error="Repair the defective binding.",
        prior_output=prior,
        prior_drafts=anchors,
        agent_contract_protocol="legacy_v1",
    )
    assert revised["previousCandidateOutput"] == prior.model_dump(mode="json")
    assert revised["previousCandidateBindingInventory"][0]["occurrences"][0]["sourceText"] == value
    assert "preserving every unaffected" in revised["repairInstruction"]


def test_literal_review_line_ids_cover_only_meaningful_unmasked_lines() -> None:
    masked = (
        "L00001 | --- PAGE 1 ---\n"
        "L00002 | CAPTION ⟦binding_0001⟧\n"
        "L00003 | ⟦binding_0002⟧\n"
        "L00004 |   \n"
        "L00005 | Ref: 12345\n"
    )

    assert _literal_review_line_ids(masked) == ("L00002", "L00005")
    with pytest.raises(ValueError, match="outside the numbered-line contract"):
        _literal_review_line_ids("not numbered")


def test_critic_inventory_groups_physical_occurrences_by_logical_binding() -> None:
    body = "SAME123456\ncaption\nSAME123456\n"
    raw = "--- PAGE 1 ---\n" + body
    value = "SAME123456"
    path = "documentPatch.billOfLadingNumber"
    offsets = (body.index(value), body.rindex(value))
    anchors = anchor_drafts(
        raw=raw,
        document_id="doc_repeated_inventory",
        anchors=tuple(
            {
                "anchor_id": f"anchor_{index}",
                "document_id": "doc_repeated_inventory",
                "patchable": True,
                "page_number": 1,
                "page_start": offset,
                "page_end": offset + len(value),
                "raw_value": value,
                "target_value": value,
                "relation_target_path": path,
                "role_path": path,
                "surface_family": "text",
            }
            for index, offset in enumerate(offsets)
        ),
    )
    inventory = _draft_inventory(raw, anchors, {"documentPatch": {"billOfLadingNumber": value}})
    assert len(inventory) == 1
    assert inventory[0]["logicalKey"] == f"anchor:{path}"
    assert [row["lineStart"] for row in inventory[0]["occurrences"]] == ["L00002", "L00004"]
    critic_payload = _critic_payload(
        document_id="doc_repeated_inventory",
        raw=raw,
        source_target={"documentPatch": {"billOfLadingNumber": value}},
        feature={
            "document_type": "bill_of_lading",
            "carrier_family": "FIXTURE",
            "template_proxy_id": "template_fixture",
        },
        drafts=anchors,
        risks=risk_candidates(raw, ()),
        prior_error=None,
    )
    assert critic_payload["allowedTargetPaths"] == ("documentPatch.billOfLadingNumber",)
    assert critic_payload["requiredTargetCoBindings"] == ()
    assert "inventoryBindingId" not in inventory[0]
    assert critic_payload["allowedRemovalLogicalKeys"] == (anchors[0].logical_key,)


def test_critic_review_candidates_include_short_typed_integer_repeats() -> None:
    body = "1542 PACKAGE\nSUMMARY 1542\n"
    raw = "--- PAGE 1 ---\n" + body
    value = "1542"
    second = body.rindex(value)
    drafts = anchor_drafts(
        raw=raw,
        document_id="doc_short_integer_repeat",
        anchors=(
            {
                "anchor_id": "anchor_quantity",
                "document_id": "doc_short_integer_repeat",
                "patchable": True,
                "page_number": 1,
                "page_start": second,
                "page_end": second + len(value),
                "raw_value": value,
                "target_value": 1542,
                "relation_target_path": "documentPatch.cargoPackages[0].quantity",
                "role_path": "documentPatch.cargoPackages[0].quantity",
                "surface_family": "integer",
            },
        ),
    )

    candidates = _critic_review_candidates(raw=raw, drafts=drafts)

    assert any(
        row["kind"] == "unowned_exact_repeat"
        and row["sourceTexts"] == (value,)
        and row["lineIds"] == ("L00002",)
        for row in candidates
    )
    digit_start = raw.rindex("2")
    embedded_digit = SpanDraft(
        draft_id="embedded_digit",
        logical_key="agent:short_integer",
        render_mode="deterministic_auxiliary",
        value_kind="integer",
        group_kind="package",
        group_key="package:short",
        target_paths=(),
        derivation=None,
        dependency_paths=(),
        dependency_bindings=(),
        char_start=digit_start,
        char_end=digit_start + 1,
        source_text="2",
        evidence_origin="audited_source_auxiliary",
        render_policy="numeric_surface",
        rationale="Fixture embedded digit.",
    )
    assert not any(
        row["sourceTexts"] == ("2",)
        for row in _critic_review_candidates(raw=raw, drafts=(embedded_digit,))
    )


def test_inventory_exposes_exact_same_line_occurrence_index() -> None:
    body = "MEASUREMENT / PACKAGE QUANTITY\n43.200M3/20\n"
    raw = "--- PAGE 1 ---\n" + body
    value = "20"
    start = body.rindex(value)
    path = "documentPatch.cargoPackages[0].quantity"
    drafts = anchor_drafts(
        raw=raw,
        document_id="doc_same_line_occurrence",
        anchors=(
            {
                "anchor_id": "anchor_explicit_quantity",
                "document_id": "doc_same_line_occurrence",
                "patchable": True,
                "page_number": 1,
                "page_start": start,
                "page_end": start + len(value),
                "raw_value": value,
                "target_value": 20,
                "relation_target_path": path,
                "role_path": path,
                "surface_family": "integer",
            },
        ),
    )

    inventory = _draft_inventory(
        raw,
        drafts,
        {
            "documentPatch": {
                "cargoPackages": [{"groupId": "g1", "packageId": "p1", "quantity": 20}]
            }
        },
    )

    assert inventory[0]["occurrences"] == (
        {
            "sourceBindingId": "anchor_binding_0001",
            "lineStart": "L00003",
            "lineEnd": "L00003",
            "sourceText": "20",
            "occurrenceIndex": 1,
            "exactMatchCount": 2,
            "exactMatchCandidates": (
                {
                    "occurrenceIndex": 0,
                    "rangeRelativeCharStart": 3,
                    "rangeRelativeCharEnd": 5,
                    "leftContext": "43.",
                    "rightContext": "0M3/20",
                    "selected": False,
                },
                {
                    "occurrenceIndex": 1,
                    "rangeRelativeCharStart": 9,
                    "rangeRelativeCharEnd": 11,
                    "leftContext": "43.200M3/",
                    "rightContext": "",
                    "selected": True,
                },
            ),
        },
    )


def test_inventory_exposes_unmasked_context_for_caption_substring_duplicate() -> None:
    body = "SHIPPER COUNTRY CODE: DE\n"
    raw = "--- PAGE 1 ---\n" + body
    value = "DE"
    value_start = body.rindex(value)
    path = "documentPatch.parties.shipper.country"
    drafts = anchor_drafts(
        raw=raw,
        document_id="doc_country_code_duplicate",
        anchors=(
            {
                "anchor_id": "anchor_country_code",
                "document_id": "doc_country_code_duplicate",
                "patchable": True,
                "page_number": 1,
                "page_start": value_start,
                "page_end": value_start + len(value),
                "raw_value": value,
                "target_value": value,
                "relation_target_path": path,
                "role_path": path,
                "surface_family": "text",
            },
        ),
    )

    inventory = _draft_inventory(
        raw,
        drafts,
        {"documentPatch": {"parties": {"shipper": {"country": value}}}},
    )
    occurrence = inventory[0]["occurrences"][0]

    assert occurrence["occurrenceIndex"] == 1
    assert occurrence["exactMatchCount"] == 2
    assert occurrence["exactMatchCandidates"][0]["leftContext"] == "SHIPPER COUNTRY CO"
    assert occurrence["exactMatchCandidates"][1]["leftContext"] == ("SHIPPER COUNTRY CODE: ")
    assert [row["selected"] for row in occurrence["exactMatchCandidates"]] == [False, True]


def _stage(role: str, pass_number: int, output: Any) -> AgentStageArtifact:
    now = datetime.now(UTC)
    return AgentStageArtifact.model_validate(
        {
            "role": role,
            "pass_number": pass_number,
            "started_at": now,
            "completed_at": now,
            "duration_seconds": 0.0,
            "system_prompt_sha256": "0" * 64,
            "user_prompt_sha256": "1" * 64,
            "output_schema_sha256": "2" * 64,
            "status": "success",
            "output": output.model_dump(mode="json"),
            "error_type": None,
            "error_message": None,
            "messages": [],
            "usage": empty_usage(),
        }
    )


def test_compiler_revision_context_survives_a_later_provider_error() -> None:
    candidate = CompilerAgentOutput.model_validate(
        {
            "carrier": {
                "canonical_name": "Example Carrier Ltd",
                "aliases": (),
                "evidence_occurrences": (
                    {
                        "line_start": "L00002",
                        "line_end": "L00002",
                        "source_text": "Example Carrier Ltd",
                        "occurrence_index": 0,
                    },
                ),
                "source": "source_label_confirmed_by_ocr",
                "rationale": "Exact carrier evidence.",
            },
            "anchor_overrides": (),
            "bindings": (),
            "unresolved": (),
            "all_shipment_dependent_surfaces_accounted_for": True,
        }
    )
    rejected = _stage("compiler", 2, candidate).model_copy(
        update={
            "status": "host_rejected",
            "error_type": "ValueError",
            "error_message": "repair this exact target ownership defect",
        }
    )
    provider_error = _stage("compiler", 3, candidate).model_copy(
        update={
            "status": "provider_error",
            "output": None,
            "error_type": "UnexpectedModelBehavior",
            "error_message": "structured output retries exhausted",
        }
    )

    context = _latest_compiler_revision_context(
        drafts=None,
        prior_output=candidate,
        compiler_stages=(rejected, provider_error),
        resume_replay_warning=None,
    )

    assert context == (
        "Host rejected the compiler output: repair this exact target ownership defect"
    )
    assert (
        _latest_compiler_revision_context(
            drafts=(),
            prior_output=candidate,
            compiler_stages=(rejected, provider_error),
            resume_replay_warning=None,
        )
        is None
    )


def test_compiler_host_rejection_aggregates_independent_topology_and_realization_defects() -> None:
    body = "Example Carrier Ltd\nPORTX\nPORTX\n40HQ\n"
    raw = "--- PAGE 1 ---\n" + body
    source_target = {
        "documentPatch": {
            "parties": {"carrier": {"name": "Example Carrier Ltd"}},
            "route": {
                "placeOfDelivery": {"name": "PORTX"},
                "portOfDischarge": {"name": "PORTX"},
            },
            "containers": [{"typeDescription": "40HQ"}],
        }
    }
    anchors = anchor_drafts(
        raw=raw,
        document_id="doc_aggregate_rejection",
        anchors=(
            {
                "anchor_id": "anchor_carrier",
                "document_id": "doc_aggregate_rejection",
                "patchable": True,
                "page_number": 1,
                "page_start": body.index("Example Carrier Ltd"),
                "page_end": body.index("Example Carrier Ltd") + len("Example Carrier Ltd"),
                "raw_value": "Example Carrier Ltd",
                "relation_target_path": "documentPatch.parties.carrier.name",
                "role_path": "documentPatch.parties.carrier.name",
                "surface_family": "text",
            },
        ),
    )
    candidate = CompilerAgentOutput.model_validate(
        {
            "carrier": {
                "canonical_name": "Example Carrier Ltd",
                "aliases": (),
                "evidence_occurrences": (
                    {
                        "line_start": "L00002",
                        "line_end": "L00002",
                        "source_text": "Example Carrier Ltd",
                        "occurrence_index": 0,
                    },
                ),
                "source": "source_label_confirmed_by_ocr",
                "rationale": "Exact carrier evidence.",
            },
            "anchor_overrides": (
                {
                    "anchor_binding_id": anchors[0].draft_id,
                    "rationale": "Fixture intentionally omits replacement ownership.",
                },
            ),
            "bindings": (
                {
                    "logical_key": "route:collapsed",
                    "render_mode": "target_binding",
                    "value_kind": "location",
                    "group_kind": "route",
                    "group_key": "route",
                    "target_paths": (
                        "documentPatch.route.placeOfDelivery.name",
                        "documentPatch.route.portOfDischarge.name",
                    ),
                    "occurrences": (
                        {
                            "line_start": "L00003",
                            "line_end": "L00003",
                            "source_text": "PORTX",
                            "occurrence_index": 0,
                        },
                        {
                            "line_start": "L00004",
                            "line_end": "L00004",
                            "source_text": "PORTX",
                            "occurrence_index": 0,
                        },
                    ),
                    "rationale": "Invalidly collapsed independent route facts.",
                },
                {
                    "logical_key": "equipment:source_only",
                    "render_mode": "deterministic_auxiliary",
                    "value_kind": "equipment",
                    "group_kind": "equipment",
                    "group_key": "equipment:all",
                    "target_paths": (),
                    "occurrences": (
                        {
                            "line_start": "L00005",
                            "line_end": "L00005",
                            "source_text": "40HQ",
                            "occurrence_index": 0,
                        },
                    ),
                    "rationale": "Invalidly source-only equipment type.",
                },
            ),
            "unresolved": (),
            "all_shipment_dependent_surfaces_accounted_for": True,
        }
    )

    rejection = _compiler_host_rejection(
        output=candidate,
        raw=raw,
        source_target=source_target,
        anchors=anchors,
        primary_error=ValueError("template spans overlap:: fixture overlap"),
    )
    message = str(rejection)

    assert "complete candidate validation: template spans overlap:: fixture overlap" in message
    assert "anchor replacement integration" in message
    assert "lacks replacement target ownership" in message
    assert "repeated binding aggregates independently mutable target facts" in message
    assert "owns an equipment-type token as source-only text" in message


class _FakeRuntime:
    def __init__(self, compiler: CompilerAgentOutput, critics: tuple[CriticAgentOutput, ...]):
        self.compiler_output = compiler
        self.critic_outputs = list(critics)
        self.compiler_calls = 0
        self.critic_calls = 0
        self.critic_prior_errors: list[str | None] = []
        self.critic_payloads: list[dict[str, Any]] = []

    async def compiler(self, **kwargs: Any):
        self.compiler_calls += 1
        return self.compiler_output, _stage("compiler", kwargs["pass_number"], self.compiler_output)

    async def critic(self, **kwargs: Any):
        self.critic_calls += 1
        self.critic_payloads.append(kwargs["payload"])
        self.critic_prior_errors.append(kwargs["payload"].get("priorHostRejection"))
        output = self.critic_outputs.pop(0)
        return output, _stage("critic", kwargs["pass_number"], output)


class _FakeStagedRun:
    def __init__(self, root: Path):
        self.stage_root = root
        self.published: dict[str, Any] = {}

    def publish_bytes(self, relative_path: str, payload: bytes) -> None:
        self.published[relative_path] = payload

    def publish_json(self, relative_path: str, payload: Any) -> None:
        self.published[relative_path] = payload


@pytest.mark.asyncio
async def test_false_pass_and_host_rejected_addition_are_repaired_inside_critic_loop(
    tmp_path: Path,
) -> None:
    body = "Acme Ocean Lines Ltd.\nVAT: 12345678\n"
    raw = "--- PAGE 1 ---\n" + body
    carrier = "Acme Ocean Lines Ltd."
    carrier_start = body.index(carrier)
    anchor_rows = (
        {
            "anchor_id": "anchor_carrier",
            "document_id": "doc_critic_repair",
            "patchable": True,
            "page_number": 1,
            "page_start": carrier_start,
            "page_end": carrier_start + len(carrier),
            "raw_value": carrier,
            "target_value": carrier,
            "relation_target_path": "documentPatch.parties.carrier.name",
            "role_path": "documentPatch.parties.carrier.name",
            "surface_family": "text",
        },
    )
    source_target = {
        "schemaVersion": "3.0.0-experimental",
        "documentPatch": {"parties": {"carrier": {"name": carrier}}},
    }
    source = {
        "documentId": "doc_critic_repair",
        "joinedRawText": raw,
        "joinedRawTextSha256": sha256_bytes(raw.encode("utf-8")),
        "target": source_target,
    }
    feature = {
        "document_type": "bill_of_lading",
        "template_proxy_id": "template_fixture",
        "page_count": 1,
        "ocr_lines": len(raw.splitlines()),
        "ocr_characters": len(raw),
        "container_count": 0,
        "seal_count": 0,
        "goods_group_count": 0,
        "package_fact_count": 0,
        "allocation_group_count": 0,
        "allocation_row_count": 0,
        "dangerous_goods_count": 0,
        "temperature_setting_count": 0,
        "additional_information_group_count": 0,
        "marks_group_count": 0,
        "target_leaf_paths": ["parties.carrier.name"],
        "carrier_family": "ACME",
    }
    assessment = {
        "canonical_name": carrier,
        "aliases": (),
        "evidence_occurrences": (
            {
                "line_start": "L00002",
                "line_end": "L00002",
                "source_text": carrier,
                "occurrence_index": 0,
            },
        ),
        "source": "source_label_confirmed_by_ocr",
        "rationale": "Exact carrier evidence.",
    }
    compiler = CompilerAgentOutput.model_validate(
        {
            "carrier": assessment,
            "anchor_overrides": (),
            "bindings": (),
            "unresolved": (),
            "all_shipment_dependent_surfaces_accounted_for": True,
        }
    )

    def revision(*, invalid_overlap: bool) -> CriticAgentOutput:
        return CriticAgentOutput.model_validate(
            {
                "verdict": "revise",
                "findings": (
                    {
                        "finding_kind": "unowned_private_or_auxiliary_fact",
                        "line_ids": ("L00003",),
                        "evidence": "VAT: 12345678",
                        "explanation": "The VAT identifier must be regenerated.",
                    },
                ),
                "additional_bindings": (
                    {
                        "logical_key": "customs:vat",
                        "render_mode": "deterministic_auxiliary",
                        "value_kind": "identifier",
                        "group_kind": "customs",
                        "group_key": "customs:vat",
                        "target_paths": (),
                        "derivation": None,
                        "dependency_paths": (),
                        "dependency_bindings": (),
                        "occurrences": (
                            {
                                "line_start": "L00002" if invalid_overlap else "L00003",
                                "line_end": "L00002" if invalid_overlap else "L00003",
                                "source_text": carrier if invalid_overlap else "12345678",
                                "occurrence_index": 0,
                            },
                        ),
                        "rationale": "Source-only VAT identifier.",
                    },
                ),
                "rationale": "One literal VAT value remains.",
            }
        )

    passed = CriticAgentOutput.model_validate(
        {
            "verdict": "pass",
            "findings": (),
            "additional_bindings": (),
            "rationale": "Every shipment-specific surface is owned.",
        }
    )
    runtime = _FakeRuntime(
        compiler,
        (
            passed,
            revision(invalid_overlap=True),
            revision(invalid_overlap=False),
            passed,
        ),
    )
    config = load_config(_EXPERIMENT_ROOT / "configs" / "canary1.yaml")
    config = config.model_copy(
        update={
            "resume_from": None,
            "workflow": config.workflow.model_copy(
                update={
                    "max_critic_passes": 4,
                    "additional_call_launch_threshold_usd_per_document": Decimal("1"),
                }
            ),
        }
    )
    initial_staged = _FakeStagedRun(tmp_path)
    result, template, _masked = await _extract_case(
        ordinal=1,
        document_id="doc_critic_repair",
        source=source,
        feature=feature,
        anchor_rows=anchor_rows,
        runtime=runtime,
        compiler_prompt="compiler",
        critic_prompt="critic",
        config=config,
        staged=initial_staged,
        document_limiter=asyncio.Semaphore(1),
    )
    assert result.status == "certified", result.rejection_reasons
    assert template is not None
    assert runtime.compiler_calls == 1
    assert runtime.critic_calls == 4
    assert runtime.critic_prior_errors[0] is None
    assert "risk_" in (runtime.critic_prior_errors[1] or "")
    assert "outside its cited findings" in (runtime.critic_prior_errors[2] or "")
    assert runtime.critic_prior_errors[3] is None
    assert result.critic_stages[0].status == "host_rejected"
    assert result.critic_stages[1].status == "host_rejected"
    ledger = runtime.critic_payloads[-1]["priorReviewLedger"]
    assert ledger["stageStatusCounts"] == {"host_rejected": 2, "success": 1}
    assert ledger["successfulVerdictCounts"] == {"revise": 1}
    assert ledger["appliedRevisionCount"] == 1
    assert ledger["carriedFindings"] == ()

    for relative_path in (
        "cases/doc_critic_repair/source.txt",
        "cases/doc_critic_repair/source-label.json",
        "cases/doc_critic_repair/template.json",
        "cases/doc_critic_repair/state-checkpoint.json",
    ):
        destination = tmp_path / relative_path
        destination.parent.mkdir(parents=True, exist_ok=True)
        payload = initial_staged.published[relative_path]
        destination.write_bytes(
            payload
            if isinstance(payload, bytes)
            else json.dumps(payload, ensure_ascii=False).encode("utf-8")
        )

    offline_runtime = _FakeRuntime(compiler, ())
    offline_config = config.model_copy(
        update={
            "workflow": config.workflow.model_copy(
                update={"certified_resume_policy": "current_host_recertify"}
            )
        }
    )
    offline_result, offline_template, _offline_masked = await _extract_case(
        ordinal=1,
        document_id="doc_critic_repair",
        source=source,
        feature=feature,
        anchor_rows=anchor_rows,
        runtime=offline_runtime,
        compiler_prompt="changed compiler",
        critic_prompt="changed critic",
        config=offline_config,
        staged=_FakeStagedRun(tmp_path / "offline-recertification"),
        document_limiter=asyncio.Semaphore(1),
        resume_run_root=tmp_path,
        resume_result=result,
        resume_contract_match=False,
    )
    assert offline_result.status == "certified"
    assert offline_template is not None
    assert offline_runtime.compiler_calls == 0
    assert offline_runtime.critic_calls == 0
    assert offline_result.resume_mode == "current_host_recertification"
    assert offline_result.resume_contract_match is False
    assert offline_result.resume_replay_warning is not None
    assert "without a provider request" in offline_result.resume_replay_warning

    checkpoint = ExtractionCaseResult.model_validate(
        {
            "document_id": "doc_critic_repair",
            "status": "rejected",
            "rejection_reasons": ("critic exhausted passes after local binding patch",),
            "template_sha256": None,
            "compiler_stages": (_stage("compiler", 1, compiler),),
            "critic_stages": (_stage("critic", 1, revision(invalid_overlap=False)),),
            "risk_candidates": risk_candidates(raw, ()),
            "elapsed_seconds": 2.5,
        }
    )
    continuation_runtime = _FakeRuntime(compiler, (passed,))
    continuation_config = config.model_copy(
        update={"workflow": config.workflow.model_copy(update={"max_critic_passes": 2})}
    )
    continued, continued_template, _continued_masked = await _extract_case(
        ordinal=1,
        document_id="doc_critic_repair",
        source=source,
        feature=feature,
        anchor_rows=anchor_rows,
        runtime=continuation_runtime,
        compiler_prompt="compiler",
        critic_prompt="critic",
        config=continuation_config,
        staged=_FakeStagedRun(tmp_path / "continuation"),
        document_limiter=asyncio.Semaphore(1),
        resume_run_root=tmp_path / "prior-run",
        resume_result=checkpoint,
    )
    assert continued.status == "certified"
    assert continued_template is not None
    assert continuation_runtime.compiler_calls == 0
    assert continuation_runtime.critic_calls == 1
    assert continued.resumed_from_run == "prior-run"
    assert continued.resumed_prior_elapsed_seconds == 2.5
    assert continued.resumed_prior_compiler_stages == 1
    assert continued.resumed_prior_critic_stages == 1
    assert continued.elapsed_seconds >= 2.5

    recertification_runtime = _FakeRuntime(compiler, (passed,))
    recertification_config = config.model_copy(
        update={"workflow": config.workflow.model_copy(update={"max_critic_passes": 1})}
    )
    recertified, recertified_template, _recertified_masked = await _extract_case(
        ordinal=1,
        document_id="doc_critic_repair",
        source=source,
        feature=feature,
        anchor_rows=anchor_rows,
        runtime=recertification_runtime,
        compiler_prompt="compiler",
        critic_prompt="critic",
        config=recertification_config,
        staged=_FakeStagedRun(tmp_path / "recertification"),
        document_limiter=asyncio.Semaphore(1),
        resume_run_root=tmp_path / "prior-certified",
        resume_result=result,
        resume_contract_match=False,
    )
    assert recertified.status == "certified", recertified.rejection_reasons
    assert recertified_template is not None
    assert recertification_runtime.compiler_calls == 0
    assert recertification_runtime.critic_calls == 1
    assert recertified.critic_stages[-1].pass_number == 5
    assert recertified.resume_contract_match is False
    chained_drafts, _chained_assessment, _chained_semantic, _chained_reviews, _warning = (
        _replay_checkpoint_case(
            result=recertified,
            raw=raw,
            source_target=source_target,
            anchors=anchor_drafts(
                raw=raw,
                document_id="doc_critic_repair",
                anchors=anchor_rows,
            ),
            risks=risk_candidates(raw, ()),
        )
    )
    assert chained_drafts is not None

    legacy_host_rejected_stage = _stage("compiler", 9, compiler).model_copy(
        update={
            "status": "host_rejected",
            "error_type": "ValueError",
            "error_message": "legacy deterministic host rejected this candidate",
        }
    )
    legacy_host_rejected_result = checkpoint.model_copy(
        update={
            "compiler_stages": (legacy_host_rejected_stage,),
            "critic_stages": (),
        }
    )
    (
        recovered_drafts,
        recovered_assessment,
        recovered_semantic,
        recovered_reviews,
        recovered_warning,
    ) = _replay_checkpoint_case(
        result=legacy_host_rejected_result,
        raw=raw,
        source_target=source_target,
        anchors=anchor_drafts(
            raw=raw,
            document_id="doc_critic_repair",
            anchors=anchor_rows,
        ),
        risks=risk_candidates(raw, ()),
    )
    assert recovered_drafts is not None
    assert recovered_assessment is not None
    assert recovered_semantic == ()
    assert recovered_reviews == []
    assert recovered_warning is not None
    assert "accepted the latest previously host-rejected compiler candidate" in (recovered_warning)
    assert "pass 9" in recovered_warning

    formerly_rejected_review = _stage("critic", 7, revision(invalid_overlap=False)).model_copy(
        update={
            "status": "host_rejected",
            "error_type": "ValueError",
            "error_message": "older host rejected this otherwise valid transaction",
        }
    )
    formerly_rejected_result = checkpoint.model_copy(
        update={"critic_stages": (formerly_rejected_review,)}
    )
    (
        replayed_drafts,
        _replayed_assessment,
        _replayed_semantic,
        replayed_reviews,
        replayed_warning,
    ) = _replay_checkpoint_case(
        result=formerly_rejected_result,
        raw=raw,
        source_target=source_target,
        anchors=anchor_drafts(
            raw=raw,
            document_id="doc_critic_repair",
            anchors=anchor_rows,
        ),
        risks=risk_candidates(raw, ()),
    )
    assert replayed_drafts is not None
    assert any(draft.logical_key == "agent:customs:vat" for draft in replayed_drafts)
    assert replayed_reviews == [revision(invalid_overlap=False)]
    assert replayed_warning is not None
    assert "formerly host-rejected critic transactions at passes 7" in replayed_warning

    stale_revision = CriticAgentOutput.model_validate(
        {
            "verdict": "revise",
            "findings": (
                {
                    "finding_kind": "incorrect_semantic_owner",
                    "line_ids": ("L00003",),
                    "evidence": "VAT: 12345678",
                    "explanation": "Fixture stale inventory edit handle.",
                },
            ),
            "remove_inventory_binding_ids": ("inventory_binding_0123456789abcdef",),
            "additional_bindings": (),
            "rationale": "The prior host used a different inventory identity.",
        }
    )
    stale_checkpoint = checkpoint.model_copy(
        update={"critic_stages": (_stage("critic", 1, stale_revision),)}
    )
    replayed, replayed_assessment, semantic_only, replayed_reviews, warning = (
        _replay_checkpoint_case(
            result=stale_checkpoint,
            raw=raw,
            source_target=source_target,
            anchors=anchor_drafts(
                raw=raw,
                document_id="doc_critic_repair",
                anchors=anchor_rows,
            ),
            risks=risk_candidates(raw, ()),
        )
    )
    assert replayed is not None
    assert replayed_assessment is not None
    assert semantic_only == ()
    assert replayed_reviews == []
    assert warning is not None
    assert "skipped incompatible prior critic pass 1" in warning

    fallback_drafts, fallback_assessment, _semantic, _reviews, fallback_warning = (
        _restore_or_replay_resume_state(
            checkpoint_payload=b"{}",
            result=checkpoint,
            document_id="doc_critic_repair",
            raw=raw,
            source_target=source_target,
            anchors=anchor_drafts(
                raw=raw,
                document_id="doc_critic_repair",
                anchors=anchor_rows,
            ),
            risks=risk_candidates(raw, ()),
            resume_contract_match=False,
        )
    )
    assert fallback_drafts is not None
    assert fallback_assessment is not None
    assert fallback_warning is not None
    assert "rejected the prior exact state checkpoint" in fallback_warning


@pytest.mark.asyncio
async def test_critic_can_reclassify_false_anchor_as_audited_unprinted_target(
    tmp_path: Path,
) -> None:
    body = "Acme Ocean Lines Ltd.\nPACKAGE SUMMARY\n"
    raw = "--- PAGE 1 ---\n" + body
    carrier = "Acme Ocean Lines Ltd."
    quantity_path = "documentPatch.cargoPackages[0].quantity"
    anchor_rows = (
        {
            "anchor_id": "anchor_carrier",
            "document_id": "doc_semantic_only",
            "patchable": True,
            "page_number": 1,
            "page_start": body.index(carrier),
            "page_end": body.index(carrier) + len(carrier),
            "raw_value": carrier,
            "target_value": carrier,
            "relation_target_path": "documentPatch.parties.carrier.name",
            "role_path": "documentPatch.parties.carrier.name",
            "surface_family": "text",
        },
        {
            "anchor_id": "anchor_false_quantity",
            "document_id": "doc_semantic_only",
            "patchable": True,
            "page_number": 1,
            "page_start": body.index("PACKAGE"),
            "page_end": body.index("PACKAGE") + len("PACKAGE"),
            "raw_value": "PACKAGE",
            "target_value": 96,
            "relation_target_path": quantity_path,
            "role_path": quantity_path,
            "surface_family": "integer",
        },
    )
    source_target = {
        "schemaVersion": "3.0.0-experimental",
        "documentPatch": {
            "parties": {"carrier": {"name": carrier}},
            "cargoGroups": [{"groupId": "g1"}],
            "cargoPackages": [{"packageId": "p1", "groupId": "g1", "quantity": 96}],
            "cargoAllocationGroups": [],
        },
    }
    source = {
        "documentId": "doc_semantic_only",
        "joinedRawText": raw,
        "joinedRawTextSha256": sha256_bytes(raw.encode("utf-8")),
        "target": source_target,
    }
    feature = {
        "document_type": "bill_of_lading",
        "template_proxy_id": "template_fixture",
        "page_count": 1,
        "ocr_lines": len(raw.splitlines()),
        "ocr_characters": len(raw),
        "container_count": 0,
        "seal_count": 0,
        "goods_group_count": 1,
        "package_fact_count": 1,
        "allocation_group_count": 0,
        "allocation_row_count": 0,
        "dangerous_goods_count": 0,
        "temperature_setting_count": 0,
        "additional_information_group_count": 0,
        "marks_group_count": 0,
        "target_leaf_paths": ["parties.carrier.name", "cargoPackages[0].quantity"],
        "carrier_family": "ACME",
    }
    compiler = CompilerAgentOutput.model_validate(
        {
            "carrier": {
                "canonical_name": carrier,
                "aliases": (),
                "evidence_occurrences": (
                    {
                        "line_start": "L00002",
                        "line_end": "L00002",
                        "source_text": carrier,
                        "occurrence_index": 0,
                    },
                ),
                "source": "source_label_confirmed_by_ocr",
                "rationale": "Exact carrier evidence.",
            },
            "anchor_overrides": (),
            "bindings": (),
            "unresolved": (),
            "all_shipment_dependent_surfaces_accounted_for": True,
            "semantic_only_target_facts": (),
        }
    )
    anchor_inventory = anchor_drafts(
        raw=raw,
        document_id="doc_semantic_only",
        anchors=anchor_rows,
    )
    quantity_owner = next(row for row in anchor_inventory if quantity_path in row.target_paths)
    reclassify = CriticAgentOutput.model_validate(
        {
            "verdict": "revise",
            "findings": (
                {
                    "finding_kind": "incorrect_semantic_owner",
                    "line_ids": ("L00003",),
                    "evidence": "PACKAGE SUMMARY",
                    "explanation": "The caption does not print the labeled package quantity.",
                },
            ),
            "remove_inventory_binding_ids": (inventory_binding_id(quantity_owner.logical_key),),
            "additional_bindings": (),
            "semantic_only_target_facts": (
                {
                    "target_path": quantity_path,
                    "rationale": (
                        "The source label has a package quantity, but this OCR has only a generic "
                        "package-summary caption and no distinct quantity surface."
                    ),
                },
            ),
            "rationale": "Remove the false caption anchor and preserve the unprinted target fact.",
        }
    )
    passed = CriticAgentOutput.model_validate(
        {
            "verdict": "pass",
            "findings": (),
            "additional_bindings": (),
            "rationale": "The remaining literal text is structural.",
        }
    )
    runtime = _FakeRuntime(compiler, (reclassify, passed))
    config = load_config(_EXPERIMENT_ROOT / "configs" / "canary1.yaml")
    config = config.model_copy(
        update={
            "resume_from": None,
            "workflow": config.workflow.model_copy(
                update={
                    "max_critic_passes": 2,
                    "additional_call_launch_threshold_usd_per_document": Decimal("1"),
                }
            ),
        }
    )
    result, template, _masked = await _extract_case(
        ordinal=1,
        document_id="doc_semantic_only",
        source=source,
        feature=feature,
        anchor_rows=anchor_rows,
        runtime=runtime,
        compiler_prompt="compiler",
        critic_prompt="critic",
        config=config,
        staged=_FakeStagedRun(tmp_path),
        document_limiter=asyncio.Semaphore(1),
    )
    assert result.status == "certified", result.rejection_reasons
    assert template is not None
    assert template.schema_version == 4
    assert template.semantic_only_target_facts[0].target_path == quantity_path
    assert template.semantic_only_target_facts[0].source_value == 96
    assert template.semantic_only_target_facts[0].provenance == "critic_audited_unprinted"
    assert all(quantity_path not in binding.target_paths for binding in template.bindings)

    compiler_classifies = CompilerAgentOutput.model_validate(
        {
            **compiler.model_dump(mode="python"),
            "anchor_overrides": (
                {
                    "anchor_binding_id": quantity_owner.draft_id,
                    "rationale": "The generic caption is not a quantity surface.",
                },
            ),
            "semantic_only_target_facts": (
                {
                    "target_path": quantity_path,
                    "rationale": "A whole-document search found no printed package quantity.",
                },
            ),
        }
    )
    compiler_runtime = _FakeRuntime(compiler_classifies, (passed,))
    compiler_result, compiler_template, _compiler_masked = await _extract_case(
        ordinal=1,
        document_id="doc_semantic_only",
        source=source,
        feature=feature,
        anchor_rows=anchor_rows,
        runtime=compiler_runtime,
        compiler_prompt="compiler",
        critic_prompt="critic",
        config=config,
        staged=_FakeStagedRun(tmp_path / "compiler-classified"),
        document_limiter=asyncio.Semaphore(1),
    )
    assert compiler_result.status == "certified", compiler_result.rejection_reasons
    assert compiler_template is not None
    assert compiler_template.semantic_only_target_facts[0].provenance == (
        "compiler_audited_unprinted"
    )
    assert compiler_runtime.critic_calls == 1
