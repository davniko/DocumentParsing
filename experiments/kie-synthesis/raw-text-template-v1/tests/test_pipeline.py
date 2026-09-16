from __future__ import annotations

import asyncio
import json
import re
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
    SemanticOnlyTargetFact,
)
from raw_text_template_experiment.pipeline import (
    _apply_validated_critic_review,
    _compiler_host_rejection,
    _compiler_occurrence_candidates,
    _compiler_payload,
    _compiler_stagnation_reason,
    _critic_addressable_target_paths,
    _critic_launch_decision,
    _critic_payload,
    _critic_review_candidates,
    _critic_review_history,
    _critic_stagnation_reason,
    _draft_inventory,
    _expected_extraction_artifacts,
    _extract_case,
    _latest_compiler_revision_context,
    _literal_review_line_ids,
    _preview_critic_review,
    _replay_checkpoint_case,
    _require_exact_resume_prefix,
    _restore_or_replay_resume_state,
    _restore_state_checkpoint,
    _resume_contract,
    _state_checkpoint,
    _validated_compiler_state,
    load_config,
    run_extraction,
)

_EXPERIMENT_ROOT = Path(__file__).resolve().parents[1]


def test_resume_contract_is_json_round_trip_stable() -> None:
    contract = _resume_contract(_EXPERIMENT_ROOT / "configs/transfer200.yaml")

    assert json.loads(json.dumps(contract)) == contract


def test_resume_accepts_only_an_exact_nonempty_selection_prefix() -> None:
    first = (1, "doc_first", "a" * 64)
    second = (2, "doc_second", "b" * 64)
    current = (first, second)

    _require_exact_resume_prefix(prior_identity=(first,), current_identity=current)
    _require_exact_resume_prefix(prior_identity=current, current_identity=current)

    for invalid in ((), (second,), (first, (2, "doc_changed", "b" * 64))):
        with pytest.raises(ValueError, match="exact prefix"):
            _require_exact_resume_prefix(
                prior_identity=invalid,
                current_identity=current,
            )


def test_expected_extraction_artifacts_include_relational_locality_receipt(
    tmp_path: Path,
) -> None:
    result = ExtractionCaseResult.model_validate(
        {
            "document_id": "doc_relational_receipt",
            "status": "rejected",
            "rejection_reasons": ("bounded rejection",),
            "template_sha256": None,
            "compiler_stages": (),
            "critic_stages": (),
            "risk_candidates": (),
            "elapsed_seconds": 0.0,
        }
    )
    case_root = tmp_path / "cases" / result.document_id
    case_root.mkdir(parents=True)
    (case_root / "anchor-locality.json").write_text("{}", encoding="utf-8")
    (case_root / "state-checkpoint.json").write_text("{}", encoding="utf-8")

    expected = _expected_extraction_artifacts(
        stage_root=tmp_path,
        results=(result,),
        compiler_repair_prompt=None,
        critic_audit_prompt=None,
        critic_plan_prompt=None,
    )

    assert f"cases/{result.document_id}/anchor-locality.json" in expected
    assert f"cases/{result.document_id}/state-checkpoint.json" in expected


def test_critic_target_vocabulary_excludes_non_renderable_structural_rows() -> None:
    source_target = {
        "documentPatch": {
            "containers": [{"containerNumber": "ABCU1234567"}],
            "cargoPackages": [{"packageId": "p1", "quantity": 4, "marks": ["LOT-1"]}],
            "cargoGroups": [{"groupId": "g1", "description": "WIDGETS"}],
        }
    }

    paths = _critic_addressable_target_paths(source_target=source_target, drafts=())

    assert "documentPatch.containers" in paths
    assert "documentPatch.cargoPackages" in paths
    assert "documentPatch.containers[0].containerNumber" in paths
    assert "documentPatch.cargoPackages[0].marks[0]" in paths
    assert "documentPatch.containers[0]" not in paths
    assert "documentPatch.cargoPackages[0].marks" not in paths
    assert "documentPatch.cargoGroups" not in paths


def test_compiler_boundary_validation_uses_post_override_state() -> None:
    raw = "--- PAGE 1 ---\nTEST CARRIER\nCONTAINER: TGBU9219649\n"
    container_path = "documentPatch.containers[0].containerNumber"
    quantity_path = "documentPatch.cargoPackages[0].quantity"
    source_target = {
        "documentPatch": {
            "containers": [{"containerNumber": "TGBU9219649"}],
            "cargoPackages": [{"packageId": "pkg-0", "quantity": 96}],
        }
    }
    body = raw.split("\n", 1)[1]
    container_start = body.index("TGBU9219649")
    quantity_start = body.index("96")
    anchors = anchor_drafts(
        raw=raw,
        document_id="doc_post_override_boundary",
        anchors=(
            {
                "anchor_id": "container",
                "document_id": "doc_post_override_boundary",
                "patchable": True,
                "page_number": 1,
                "page_start": container_start,
                "page_end": container_start + len("TGBU9219649"),
                "raw_value": "TGBU9219649",
                "target_value": "TGBU9219649",
                "relation_target_path": container_path,
                "role_path": container_path,
                "surface_family": "text",
            },
            {
                "anchor_id": "quantity",
                "document_id": "doc_post_override_boundary",
                "patchable": True,
                "page_number": 1,
                "page_start": quantity_start,
                "page_end": quantity_start + len("96"),
                "raw_value": "96",
                "target_value": 96,
                "relation_target_path": quantity_path,
                "role_path": quantity_path,
                "surface_family": "integer",
            },
        ),
    )
    quantity_owner = next(row for row in anchors if quantity_path in row.target_paths)
    output = CompilerAgentOutput.model_validate(
        {
            "carrier": {
                "canonical_name": "TEST CARRIER",
                "aliases": (),
                "evidence_occurrences": (
                    {
                        "line_start": "L00002",
                        "line_end": "L00002",
                        "source_text": "TEST CARRIER",
                        "occurrence_index": 0,
                    },
                ),
                "source": "ocr_resolved_missing_source_label",
                "rationale": "Fixture carrier evidence.",
            },
            "anchor_overrides": (
                {
                    "anchor_binding_id": quantity_owner.draft_id,
                    "rationale": "The numeric substring is not an independent printed quantity.",
                },
            ),
            "bindings": (
                {
                    "logical_key": "carrier:identity",
                    "render_mode": "carrier_static",
                    "value_kind": "organization",
                    "group_kind": "carrier",
                    "group_key": "carrier:identity",
                    "target_paths": (),
                    "occurrences": (
                        {
                            "line_start": "L00002",
                            "line_end": "L00002",
                            "source_text": "TEST CARRIER",
                            "occurrence_index": 0,
                        },
                    ),
                    "rationale": "Exact carrier identity evidence.",
                },
                {
                    "logical_key": "anchor:" + container_path,
                    "render_mode": "target_binding",
                    "value_kind": "identifier",
                    "group_kind": "equipment",
                    "group_key": "container:0",
                    "target_paths": (container_path,),
                    "occurrences": (
                        {
                            "line_start": "L00003",
                            "line_end": "L00003",
                            "source_text": "TGBU9219649",
                            "occurrence_index": 0,
                        },
                    ),
                    "rationale": "Preserve the complete container identifier only.",
                },
            ),
            "unresolved": (),
            "all_shipment_dependent_surfaces_accounted_for": True,
            "semantic_only_target_facts": (
                {
                    "target_path": quantity_path,
                    "rationale": "No token-bounded quantity is printed.",
                },
            ),
        }
    )

    drafts, _assessment, semantic_only = _validated_compiler_state(
        output=output,
        raw=raw,
        source_target=source_target,
        anchors=anchors,
    )

    assert all(quantity_path not in draft.target_paths for draft in drafts)
    assert {fact.target_path for fact in semantic_only} == {quantity_path}

    rejection = _compiler_host_rejection(
        output=output,
        raw=raw,
        source_target=source_target,
        anchors=anchors,
        primary_error=ValueError("fixture downstream defect"),
    )
    assert "binding token boundaries" not in str(rejection)


def test_compiler_state_discards_semantic_only_claim_refuted_by_pinned_cobinding() -> None:
    raw = "--- PAGE 1 ---\nTEST CARRIER\n96\n96\n"
    carrier_path = "documentPatch.parties.carrier.name"
    quantity_paths = {
        index: (
            f"documentPatch.cargoAllocationGroups[{index}].allocations[0].packageQuantity",
            f"documentPatch.cargoPackages[{index}].quantity",
        )
        for index in range(2)
    }
    source_target = {
        "documentPatch": {
            "parties": {"carrier": {"name": "TEST CARRIER"}},
            "cargoPackages": [
                {"groupId": "g1", "packageId": "p1", "quantity": 96},
                {"groupId": "g2", "packageId": "p2", "quantity": 96},
            ],
            "cargoAllocationGroups": [
                {
                    "groupId": "g1",
                    "coverage": "one_to_one_package_allocations",
                    "packageIds": ["p1"],
                    "allocations": [{"packageId": "p1", "packageQuantity": 96}],
                },
                {
                    "groupId": "g2",
                    "coverage": "one_to_one_package_allocations",
                    "packageIds": ["p2"],
                    "allocations": [{"packageId": "p2", "packageQuantity": 96}],
                },
            ],
        }
    }

    def draft(
        *,
        draft_id: str,
        text: str,
        occurrence: int,
        target_paths: tuple[str, ...],
        mode: str = "target_binding",
    ) -> SpanDraft:
        starts = [match.start() for match in re.finditer(re.escape(text), raw)]
        start = starts[occurrence]
        return SpanDraft(
            draft_id=draft_id,
            logical_key="anchor:" + "|".join(target_paths),
            render_mode=mode,
            value_kind="organization" if mode == "carrier_static" else "integer",
            group_kind="carrier" if mode == "carrier_static" else "package",
            group_key="party:carrier:0" if mode == "carrier_static" else "package:fixture",
            target_paths=target_paths,
            derivation=None,
            dependency_paths=(),
            dependency_bindings=(),
            char_start=start,
            char_end=start + len(text),
            source_text=text,
            evidence_origin="accepted_label_evidence",
            render_policy="natural_text" if mode == "carrier_static" else "numeric_surface",
            rationale="Pinned fixture evidence.",
        )

    carrier = draft(
        draft_id="anchor_binding_0001",
        text="TEST CARRIER",
        occurrence=0,
        target_paths=(carrier_path,),
        mode="carrier_static",
    )
    pinned_quantity = draft(
        draft_id="anchor_binding_0002",
        text="96",
        occurrence=1,
        target_paths=quantity_paths[1],
    )
    output = CompilerAgentOutput.model_validate(
        {
            "carrier": {
                "canonical_name": "TEST CARRIER",
                "aliases": (),
                "evidence_occurrences": (
                    {
                        "line_start": "L00002",
                        "line_end": "L00002",
                        "source_text": "TEST CARRIER",
                        "occurrence_index": 0,
                    },
                ),
                "source": "source_label_confirmed_by_ocr",
                "rationale": "Exact carrier evidence.",
            },
            "anchor_overrides": (
                {
                    "anchor_binding_id": pinned_quantity.draft_id,
                    "rationale": "Fixture wrongly declares this pinned occurrence unprinted.",
                },
            ),
            "bindings": (
                {
                    "logical_key": "package-zero",
                    "render_mode": "target_binding",
                    "value_kind": "integer",
                    "group_kind": "package",
                    "group_key": "package:0",
                    "target_paths": quantity_paths[0],
                    "derivation": None,
                    "dependency_paths": (),
                    "occurrences": (
                        {
                            "line_start": "L00003",
                            "line_end": "L00003",
                            "source_text": "96",
                            "occurrence_index": 0,
                        },
                        {
                            "line_start": "L00004",
                            "line_end": "L00004",
                            "source_text": "96",
                            "occurrence_index": 0,
                        },
                    ),
                    "rationale": "Fixture competing equal-valued package owner.",
                },
            ),
            "unresolved": (),
            "all_shipment_dependent_surfaces_accounted_for": True,
            "semantic_only_target_facts": (
                {
                    "target_path": quantity_paths[1][1],
                    "rationale": "Fixture incorrectly claims no printed quantity.",
                },
            ),
        }
    )

    drafts, _assessment, semantic_only = _validated_compiler_state(
        output=output,
        raw=raw,
        source_target=source_target,
        anchors=(carrier, pinned_quantity),
    )

    quantity_owners = {
        (row.char_start, row.target_paths) for row in drafts if "quantity" in row.logical_key
    }
    assert quantity_owners == {
        (raw.index("96"), quantity_paths[0]),
        (raw.rindex("96"), quantity_paths[1]),
    }
    assert quantity_paths[1][1] not in {fact.target_path for fact in semantic_only}


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


def test_compiler_stagnation_requires_two_consecutive_identical_host_rejections() -> None:
    output = CompilerAgentOutput.model_validate(
        {
            "carrier": {
                "canonical_name": "Example Carrier Ltd",
                "aliases": (),
                "evidence_occurrences": (
                    {
                        "line_start": "L00001",
                        "line_end": "L00001",
                        "source_text": "Example Carrier Ltd",
                        "occurrence_index": 0,
                    },
                ),
                "source": "source_label_confirmed_by_ocr",
                "rationale": "Fixture carrier.",
            },
            "anchor_overrides": (),
            "bindings": (),
            "all_shipment_dependent_surfaces_accounted_for": True,
            "unresolved": (),
            "semantic_only_target_facts": (),
        }
    )

    def rejected(pass_number: int, message: str) -> AgentStageArtifact:
        return _stage("compiler", pass_number, output).model_copy(
            update={
                "status": "host_rejected",
                "error_type": "ValueError",
                "error_message": message,
            }
        )

    first = rejected(1, "overlap A")
    assert _compiler_stagnation_reason((first,)) is None
    assert _compiler_stagnation_reason((first, rejected(2, "overlap B"))) is None
    reason = _compiler_stagnation_reason((first, rejected(2, "overlap A")))
    assert reason is not None
    assert "two consecutive identical" in reason


def test_critic_preview_rejects_transaction_that_exposes_deterministic_risk() -> None:
    raw = "--- PAGE 1 ---\nAcme Ocean Lines Ltd.\nVAT 12345678\n"
    carrier = "Acme Ocean Lines Ltd."
    carrier_start = raw.index(carrier)
    vat = "12345678"
    vat_start = raw.index(vat)
    carrier_draft = SpanDraft(
        draft_id="carrier",
        logical_key="anchor:documentPatch.parties.carrier.name",
        render_mode="carrier_static",
        value_kind="organization",
        group_kind="carrier",
        group_key="party:carrier:0",
        target_paths=("documentPatch.parties.carrier.name",),
        derivation=None,
        dependency_paths=(),
        dependency_bindings=(),
        char_start=carrier_start,
        char_end=carrier_start + len(carrier),
        source_text=carrier,
        evidence_origin="accepted_label_evidence",
        render_policy="natural_text",
        rationale="Fixture carrier.",
    )
    vat_draft = SpanDraft(
        draft_id="vat",
        logical_key="agent:customs:vat",
        render_mode="deterministic_auxiliary",
        value_kind="identifier",
        group_kind="customs",
        group_key="customs:vat",
        target_paths=(),
        derivation=None,
        dependency_paths=(),
        dependency_bindings=(),
        char_start=vat_start,
        char_end=vat_start + len(vat),
        source_text=vat,
        evidence_origin="audited_source_auxiliary",
        render_policy="opaque_identifier",
        rationale="Fixture VAT.",
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
            "rationale": "Exact fixture carrier.",
        }
    )
    review = CriticAgentOutput.model_validate(
        {
            "verdict": "revise",
            "findings": (
                {
                    "finding_kind": "incorrect_semantic_owner",
                    "line_ids": ("L00003",),
                    "evidence": vat,
                    "explanation": "Fixture removes the only risk owner.",
                },
            ),
            "remove_inventory_binding_ids": (inventory_binding_id(vat_draft.logical_key),),
            "additional_bindings": (),
            "rationale": "Intentionally invalid fixture transaction.",
        }
    )

    with pytest.raises(ValueError, match="leaves deterministic risk candidates unowned"):
        _preview_critic_review(
            review,
            raw=raw,
            drafts=(carrier_draft, vat_draft),
            source_target={"documentPatch": {"parties": {"carrier": {"name": carrier}}}},
            assessment=assessment,
            semantic_only_target_facts=(),
        )


def test_critic_revision_rejects_noop_after_complete_host_normalization() -> None:
    raw = (
        "--- PAGE 1 ---\n"
        "Acme Ocean Lines Ltd.\n"
        "AAAU1234567\n"
        "1X40'HQ CONTAINER\n"
        "CARGO\n"
        "40HQ\n"
        "BBBU1234567\n"
    )
    carrier = "Acme Ocean Lines Ltd."
    type_path = "documentPatch.containers[0].typeDescription"
    source_target = {
        "documentPatch": {
            "parties": {"carrier": {"name": carrier}},
            "containers": [
                {"containerNumber": "AAAU1234567", "typeDescription": "1X40'HQ CONTAINER"},
                {"containerNumber": "BBBU1234567"},
            ],
        }
    }

    def draft(
        *,
        draft_id: str,
        text: str,
        logical_key: str,
        target_paths: tuple[str, ...],
        render_mode: str = "target_binding",
        group_key: str,
    ) -> SpanDraft:
        start = raw.index(text)
        return SpanDraft(
            draft_id=draft_id,
            logical_key=logical_key,
            render_mode=render_mode,
            value_kind="organization" if text == carrier else "equipment",
            group_kind="carrier" if text == carrier else "equipment",
            group_key=group_key,
            target_paths=target_paths,
            derivation=None,
            dependency_paths=(),
            dependency_bindings=(),
            char_start=start,
            char_end=start + len(text),
            source_text=text,
            evidence_origin="host_verified_agent_proposal",
            render_policy="natural_text" if text == carrier else "opaque_identifier",
            rationale="Complete-normalization no-op fixture.",
        )

    carrier_draft = draft(
        draft_id="carrier",
        text=carrier,
        logical_key="anchor:documentPatch.parties.carrier.name",
        target_paths=("documentPatch.parties.carrier.name",),
        render_mode="carrier_static",
        group_key="party:carrier:0",
    )
    number_0_path = "documentPatch.containers[0].containerNumber"
    number_1_path = "documentPatch.containers[1].containerNumber"
    number_0 = draft(
        draft_id="number_0",
        text="AAAU1234567",
        logical_key="anchor:" + number_0_path,
        target_paths=(number_0_path,),
        group_key="container:0",
    )
    full_type = draft(
        draft_id="full_type",
        text="1X40'HQ CONTAINER",
        logical_key="anchor:" + type_path,
        target_paths=(type_path,),
        group_key="container:0",
    )
    number_1 = draft(
        draft_id="number_1",
        text="BBBU1234567",
        logical_key="anchor:" + number_1_path,
        target_paths=(number_1_path,),
        group_key="container:1",
    )
    compact_auxiliary = draft(
        draft_id="compact_auxiliary",
        text="40HQ",
        logical_key="agent:host_localized_container_1_type_token",
        target_paths=(),
        render_mode="deterministic_auxiliary",
        group_key="container:1",
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
            "rationale": "Exact fixture carrier.",
        }
    )
    review = CriticAgentOutput.model_validate(
        {
            "verdict": "revise",
            "findings": (
                {
                    "finding_kind": "incorrect_semantic_owner",
                    "line_ids": ("L00006", "L00007"),
                    "evidence": "40HQ",
                    "explanation": "Incorrectly proposes moving the compact token backward.",
                },
            ),
            "remove_inventory_binding_ids": (inventory_binding_id(compact_auxiliary.logical_key),),
            "additional_bindings": (
                {
                    "logical_key": full_type.logical_key,
                    "render_mode": "target_binding",
                    "value_kind": "equipment",
                    "group_kind": "equipment",
                    "group_key": "container:0",
                    "target_paths": (type_path,),
                    "occurrences": (
                        {
                            "line_start": "L00007",
                            "line_end": "L00007",
                            "source_text": "BBBU1234567",
                            "occurrence_index": 0,
                        },
                    ),
                    "rationale": "Incorrect candidate-reference reassignment fixture.",
                },
            ),
            "rationale": "Fixture critic revision.",
        }
    )

    with pytest.raises(ValueError, match="complete host normalization"):
        _apply_validated_critic_review(
            review=review,
            raw=raw,
            drafts=(carrier_draft, number_0, full_type, compact_auxiliary, number_1),
            source_target=source_target,
            assessment=assessment,
            semantic_only_target_facts=(),
        )


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
        coherence_constraints=(),
        critic_outputs=(),
    )

    restored, restored_assessment, semantic_only, coherence, reviews = _restore_state_checkpoint(
        checkpoint_payload=checkpoint.model_dump_json().encode("utf-8"),
        document_id=document_id,
        raw=raw,
        source_target=source_target,
    )

    assert restored == drafts
    assert restored_assessment == assessment
    assert semantic_only == ()
    assert coherence == ()
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
        "rows": (("cand_L00002_00001", "L00002", "L00002", value, 0),),
    }
    relational_reference_payload = _compiler_payload(
        document_id="doc_relationship",
        raw=raw,
        source_target=equal,
        feature=feature,
        anchors=anchors,
        risks=risk_candidates(raw, ()),
        prior_error=None,
        prior_output=None,
        prior_drafts=None,
        agent_contract_protocol="relational_reference_compact_v9",
    )
    assert (
        relational_reference_payload["occurrenceCandidates"]
        == reference_payload["occurrenceCandidates"]
    )
    hybrid_reference_payload = _compiler_payload(
        document_id="doc_relationship",
        raw=raw,
        source_target=equal,
        feature=feature,
        anchors=anchors,
        risks=risk_candidates(raw, ()),
        prior_error=None,
        prior_output=None,
        prior_drafts=None,
        agent_contract_protocol="hybrid_reference_partitioned_v10",
    )
    assert hybrid_reference_payload == relational_reference_payload
    staged_payload = _compiler_payload(
        document_id="doc_relationship",
        raw=raw,
        source_target=equal,
        feature=feature,
        anchors=anchors,
        risks=risk_candidates(raw, ()),
        prior_error=None,
        prior_output=None,
        prior_drafts=None,
        agent_contract_protocol="staged_local_v4",
    )
    assert staged_payload["occurrenceCandidates"] == reference_payload["occurrenceCandidates"]
    faceted_payload = _compiler_payload(
        document_id="doc_relationship",
        raw=raw,
        source_target=equal,
        feature=feature,
        anchors=anchors,
        risks=risk_candidates(raw, ()),
        prior_error=None,
        prior_output=None,
        prior_drafts=None,
        agent_contract_protocol="faceted_staged_local_v5",
    )
    assert faceted_payload["occurrenceCandidates"] == reference_payload["occurrenceCandidates"]
    partitioned_payload = _compiler_payload(
        document_id="doc_relationship",
        raw=raw,
        source_target=equal,
        feature=feature,
        anchors=anchors,
        risks=risk_candidates(raw, ()),
        prior_error=None,
        prior_output=None,
        prior_drafts=None,
        agent_contract_protocol="partitioned_staged_local_v6",
    )
    assert partitioned_payload["occurrenceCandidates"] == reference_payload["occurrenceCandidates"]
    assert "allowedTargetPaths" not in partitioned_payload
    assert "anchorBindings" not in partitioned_payload
    assert "riskCandidates" not in partitioned_payload
    assert partitioned_payload["targetPathTable"]["rows"] == (
        (0, "documentPatch.first"),
        (1, "documentPatch.second"),
        (2, "documentPatch.negotiability"),
    )
    assert partitioned_payload["anchorBindingTable"]["rows"][0][6] == (0, 1)
    assert partitioned_payload["anchorOccurrenceTable"]["rows"] == (
        ("anchor_binding_0001", "L00002", "L00002", value),
    )
    v8_payload = _compiler_payload(
        document_id="doc_relationship",
        raw=raw,
        source_target=equal,
        feature=feature,
        anchors=anchors,
        risks=risk_candidates(raw, ()),
        prior_error=None,
        prior_output=None,
        prior_drafts=None,
        agent_contract_protocol="candidate_first_staged_local_v8",
    )
    assert v8_payload["anchorBindingsAuthorizedForInitialReview"] == ("anchor_binding_0001",)

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
    staged_revised = _compiler_payload(
        document_id="doc_relationship",
        raw=raw,
        source_target=unequal,
        feature=feature,
        anchors=anchors,
        risks=risk_candidates(raw, ()),
        prior_error="Repair the defective binding.",
        prior_output=prior,
        prior_drafts=anchors,
        agent_contract_protocol="staged_local_v4",
    )
    assert staged_revised["previousCandidateBindingInventory"]
    assert "previousCandidateMaskedTemplate" not in staged_revised
    faceted_revised = _compiler_payload(
        document_id="doc_relationship",
        raw=raw,
        source_target=unequal,
        feature=feature,
        anchors=anchors,
        risks=risk_candidates(raw, ()),
        prior_error="Repair the defective binding.",
        prior_output=prior,
        prior_drafts=anchors,
        agent_contract_protocol="faceted_staged_local_v5",
    )
    assert faceted_revised["previousCandidateBindingInventory"]
    assert "previousCandidateMaskedTemplate" not in faceted_revised


def test_compiler_candidates_exclude_embedded_alphanumeric_matches_in_all_protocols() -> None:
    body = "CONTAINER: TGBU9219649\nPACKAGE: 96\n"
    raw = "--- PAGE 1 ---\n" + body
    container = "TGBU9219649"
    quantity = "96"
    anchors = anchor_drafts(
        raw=raw,
        document_id="doc_embedded_quantity",
        anchors=(
            {
                "anchor_id": "container",
                "document_id": "doc_embedded_quantity",
                "patchable": True,
                "page_number": 1,
                "page_start": body.index(container),
                "page_end": body.index(container) + len(container),
                "raw_value": container,
                "target_value": container,
                "relation_target_path": "documentPatch.containers[0].containerNumber",
                "role_path": "documentPatch.containers[0].containerNumber",
                "surface_family": "identifier",
            },
            {
                "anchor_id": "quantity",
                "document_id": "doc_embedded_quantity",
                "patchable": True,
                "page_number": 1,
                "page_start": body.rindex(quantity),
                "page_end": body.rindex(quantity) + len(quantity),
                "raw_value": quantity,
                "target_value": 96,
                "relation_target_path": "documentPatch.cargoPackages[0].quantity",
                "role_path": "documentPatch.cargoPackages[0].quantity",
                "surface_family": "integer",
            },
        ),
    )
    source_target = {
        "documentPatch": {
            "containers": [{"containerNumber": container}],
            "cargoPackages": [{"packageId": "package_0", "quantity": 96}],
        }
    }
    feature = {
        "document_type": "bill_of_lading",
        "carrier_family": "FIXTURE",
        "template_proxy_id": "template_fixture",
        "page_count": 1,
        "container_count": 1,
        "goods_group_count": 0,
        "package_fact_count": 1,
        "dangerous_goods_count": 0,
        "temperature_setting_count": 0,
    }

    v7_payload = _compiler_payload(
        document_id="doc_embedded_quantity",
        raw=raw,
        source_target=source_target,
        feature=feature,
        anchors=anchors,
        risks=risk_candidates(raw, ()),
        prior_error=None,
        prior_output=None,
        prior_drafts=None,
        agent_contract_protocol="candidate_first_staged_local_v7",
    )
    v8_payload = _compiler_payload(
        document_id="doc_embedded_quantity",
        raw=raw,
        source_target=source_target,
        feature=feature,
        anchors=anchors,
        risks=risk_candidates(raw, ()),
        prior_error=None,
        prior_output=None,
        prior_drafts=None,
        agent_contract_protocol="candidate_first_staged_local_v8",
    )
    columns = tuple(v8_payload["occurrenceCandidates"]["columns"])

    def quantity_lines(payload: dict[str, Any]) -> tuple[str, ...]:
        return tuple(
            row[columns.index("lineStart")]
            for row in payload["occurrenceCandidates"]["rows"]
            if row[columns.index("sourceText")] == quantity
        )

    assert quantity_lines(v7_payload) == ("L00003",)
    assert quantity_lines(v8_payload) == ("L00003",)


def test_v8_compiler_candidates_exclude_token_bounded_partial_retained_anchor_matches() -> None:
    body = "REFERENCE: REF/EGYPT/2024\nCOUNTRY: EGYPT\n"
    raw = "--- PAGE 1 ---\n" + body
    reference = "REF/EGYPT/2024"
    country = "EGYPT"
    anchors = anchor_drafts(
        raw=raw,
        document_id="doc_partial_anchor",
        anchors=(
            {
                "anchor_id": "reference",
                "document_id": "doc_partial_anchor",
                "patchable": True,
                "page_number": 1,
                "page_start": body.index(reference),
                "page_end": body.index(reference) + len(reference),
                "raw_value": reference,
                "target_value": reference,
                "relation_target_path": "documentPatch.reference",
                "role_path": "documentPatch.reference",
                "surface_family": "identifier",
            },
        ),
    )
    source_target = {"documentPatch": {"reference": reference, "country": country}}
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

    def payload(protocol: str) -> dict[str, Any]:
        return _compiler_payload(
            document_id="doc_partial_anchor",
            raw=raw,
            source_target=source_target,
            feature=feature,
            anchors=anchors,
            risks=risk_candidates(raw, ()),
            prior_error=None,
            prior_output=None,
            prior_drafts=None,
            agent_contract_protocol=protocol,
        )

    v7_payload = payload("candidate_first_staged_local_v7")
    v8_payload = payload("candidate_first_staged_local_v8")
    columns = tuple(v8_payload["occurrenceCandidates"]["columns"])

    def country_lines(candidate_payload: dict[str, Any]) -> tuple[str, ...]:
        return tuple(
            row[columns.index("lineStart")]
            for row in candidate_payload["occurrenceCandidates"]["rows"]
            if row[columns.index("sourceText")] == country
        )

    assert country_lines(v7_payload) == ("L00002", "L00003")
    assert country_lines(v8_payload) == ("L00003",)


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


def test_critic_review_candidates_do_not_surface_incidental_single_digit_repeats() -> None:
    raw = "--- PAGE 1 ---\n2 PACKAGES\nCLAUSE 2\nPAGE 2\n"
    start = raw.index("2 PACKAGES")
    owner = SpanDraft(
        draft_id="package_quantity",
        logical_key="anchor:documentPatch.cargoPackages[0].quantity",
        render_mode="target_binding",
        value_kind="integer",
        group_kind="package",
        group_key="package:0",
        target_paths=("documentPatch.cargoPackages[0].quantity",),
        derivation=None,
        dependency_paths=(),
        dependency_bindings=(),
        char_start=start,
        char_end=start + 1,
        source_text="2",
        evidence_origin="accepted_label_evidence",
        render_policy="numeric_surface",
        rationale="Fixture package quantity.",
    )

    assert not any(
        row["kind"] == "unowned_exact_repeat" and row["sourceTexts"] == ("2",)
        for row in _critic_review_candidates(raw=raw, drafts=(owner,))
    )


def test_critic_review_candidates_ignore_conditional_negotiability_boilerplate() -> None:
    raw = (
        "--- PAGE 1 ---\n"
        "NON-NEGOTIABLE\n"
        "IF THIS IS A NON-NEGOTIABLE (straight) bill of lading, one original is sufficient.\n"
    )
    start = raw.index("NON-NEGOTIABLE")
    owner = SpanDraft(
        draft_id="negotiability",
        logical_key="anchor:documentPatch.negotiability",
        render_mode="target_binding",
        value_kind="enum",
        group_kind="document",
        group_key="document:negotiability",
        target_paths=("documentPatch.negotiability",),
        derivation=None,
        dependency_paths=(),
        dependency_bindings=(),
        char_start=start,
        char_end=start + len("NON-NEGOTIABLE"),
        source_text="NON-NEGOTIABLE",
        evidence_origin="accepted_label_evidence",
        render_policy="categorical_surface",
        rationale="Fixture selected status.",
    )

    candidates = _critic_review_candidates(raw=raw, drafts=(owner,))

    assert not any(
        row["kind"] == "unowned_exact_repeat" and row["sourceTexts"] == ("NON-NEGOTIABLE",)
        for row in candidates
    )


def test_critic_review_candidates_group_repeats_and_reject_word_substrings() -> None:
    raw = "--- PAGE 1 ---\nCOUNTRY: EGYPT\nEGYPT\nEGYPT\nEGYPTIAN IMPORTER\n"
    source_text = "EGYPT"
    start = raw.index(source_text)
    draft = SpanDraft(
        draft_id="country",
        logical_key="agent:origin_country",
        render_mode="deterministic_auxiliary",
        value_kind="location",
        group_kind="customs",
        group_key="customs:origin",
        target_paths=(),
        derivation=None,
        dependency_paths=(),
        dependency_bindings=(),
        char_start=start,
        char_end=start + len(source_text),
        source_text=source_text,
        evidence_origin="host_verified_agent_proposal",
        render_policy="natural_text",
        rationale="Fixture country.",
    )

    repeats = [
        row
        for row in _critic_review_candidates(raw=raw, drafts=(draft,))
        if row["kind"] == "unowned_exact_repeat"
    ]

    assert len(repeats) == 1
    assert repeats[0]["lineIds"] == ("L00003", "L00004")


def test_critic_review_candidates_surface_repeated_unowned_phrases() -> None:
    raw = "--- PAGE 1 ---\nCONSIGNEE CITY\nSHEBIN EL KOM\nNOTIFY CITY\nSHEBIN EL KOM\n"

    candidates = _critic_review_candidates(raw=raw, drafts=())

    repeated = next(
        row
        for row in candidates
        if row["kind"] == "unowned_repeated_literal_surface"
        and row["sourceTexts"] == ("SHEBIN EL KOM",)
    )
    assert repeated["lineIds"] == ("L00003", "L00005")


def test_critic_review_candidates_surface_operational_status_and_static_contract() -> None:
    raw = "--- PAGE 1 ---\nSIGNED LOCAL CARRIER AGENT\nSHIPPED ON BOARD VESSEL X\n"
    static_surface = "LOCAL CARRIER AGENT"
    start = raw.index(static_surface)
    draft = SpanDraft(
        draft_id="local_agent",
        logical_key="agent:local_carrier_agent",
        render_mode="carrier_static",
        value_kind="organization",
        group_kind="party",
        group_key="party:local_agent",
        target_paths=(),
        derivation=None,
        dependency_paths=(),
        dependency_bindings=(),
        char_start=start,
        char_end=start + len(static_surface),
        source_text=static_surface,
        evidence_origin="host_verified_agent_proposal",
        render_policy="natural_text",
        rationale="Fixture static classification.",
    )

    candidates = _critic_review_candidates(raw=raw, drafts=(draft,))

    static_review = next(
        row for row in candidates if row["kind"] == "carrier_static_contract_review"
    )
    assert static_review["logicalKeys"] == ("agent:local_carrier_agent",)
    assert static_review["details"]["occurrenceContexts"] == (
        "--- PAGE 1 ---\nSIGNED LOCAL CARRIER AGENT",
    )
    status = next(
        row
        for row in candidates
        if row["kind"] == "domain_vocabulary_surface_review"
        and row["details"]["semanticHint"] == "operational_status"
    )
    assert status["lineIds"] == ("L00003",)
    assert status["details"]["relationshipToVerify"] == (
        "Determine whether this phrase is a selected shipment status or an unselected form "
        "caption; only a selected status requires ownership."
    )


def test_critic_review_candidates_expose_every_viable_compact_equipment_owner() -> None:
    raw = "--- PAGE 1 ---\nCMAU1111111 40HQ\nCMAU2222222 40HQ\nSUMMARY 40HQ\n"

    def equipment_draft(*, occurrence: int, group: int) -> SpanDraft:
        start = [match.start() for match in re.finditer("40HQ", raw)][occurrence]
        return SpanDraft(
            draft_id=f"equipment_{group}",
            logical_key=f"agent:container:{group}:type",
            render_mode="deterministic_derived",
            value_kind="equipment",
            group_kind="equipment",
            group_key=f"container:{group}",
            target_paths=(f"documentPatch.containers[{group}]",),
            derivation="equipment_receipt",
            dependency_paths=(f"documentPatch.containers[{group}].typeDescription",),
            dependency_bindings=(),
            char_start=start,
            char_end=start + 4,
            source_text="40HQ",
            evidence_origin="host_verified_agent_proposal",
            render_policy="equipment_surface",
            rationale="Fixture compact equipment receipt.",
        )

    candidates = _critic_review_candidates(
        raw=raw,
        drafts=(
            equipment_draft(occurrence=0, group=0),
            equipment_draft(occurrence=1, group=1),
        ),
    )

    repeat = next(
        row
        for row in candidates
        if row["kind"] == "unowned_exact_repeat" and row["sourceTexts"] == ("40HQ",)
    )
    assert repeat["lineIds"] == ("L00004",)
    assert repeat["logicalKeys"] == (
        "agent:container:0:type",
        "agent:container:1:type",
    )
    assert {row["groupKey"] for row in repeat["details"]["possibleOwnerContexts"]} == {
        "container:0",
        "container:1",
    }


def test_critic_review_candidates_retrieve_semantic_only_legal_evidence() -> None:
    raw = (
        "--- PAGE 1 ---\n"
        "In witness whereof 0 original bills of lading have been signed, one of which being "
        "accomplished, the other(s) to be void.\n"
    )
    semantic_only = SemanticOnlyTargetFact.model_validate(
        {
            "target_path": "documentPatch.negotiability",
            "source_value": "non_negotiable",
            "provenance": "host_document_semantic",
            "rationale": "Document classification requires physical evidence review.",
        }
    )

    candidates = _critic_review_candidates(
        raw=raw,
        drafts=(),
        semantic_only_target_facts=(semantic_only,),
    )

    review = next(row for row in candidates if row["kind"] == "semantic_only_evidence_review")
    assert review["lineIds"] == ("L00002",)
    assert review["details"]["targetPath"] == "documentPatch.negotiability"
    assert "original bills of lading" in review["sourceTexts"][0]


def test_critic_review_candidates_surface_agent_residual_boundaries() -> None:
    raw = "--- PAGE 1 ---\nSHEBIN EL KOM 32111-22 *\nEGYPT\n"
    start = raw.index("SHEBIN")
    surface = "SHEBIN EL KOM 32111-22 *"
    draft = SpanDraft(
        draft_id="notify_address",
        logical_key="agent:party:notify:block",
        render_mode="agent_residual",
        value_kind="address",
        group_kind="party",
        group_key="party:notify:0",
        target_paths=(
            "documentPatch.parties.notifyParties[0].address",
            "documentPatch.parties.notifyParties[0].city",
        ),
        derivation=None,
        dependency_paths=(),
        dependency_bindings=(),
        char_start=start,
        char_end=start + len(surface),
        source_text=surface,
        evidence_origin="host_verified_agent_proposal",
        render_policy="natural_text",
        rationale="Fixture residual.",
    )
    source_target = {
        "documentPatch": {
            "parties": {
                "notifyParties": [
                    {
                        "address": "SHEBIN EL KOM 32111-22",
                        "city": "SHEBIN EL KOM",
                    }
                ]
            }
        }
    }

    candidates = _critic_review_candidates(
        raw=raw,
        drafts=(draft,),
        source_target=source_target,
    )

    review = next(row for row in candidates if row["kind"] == "agent_residual_contract_review")
    assert review["logicalKeys"] == ("agent:party:notify:block",)
    assert review["sourceTexts"] == (surface,)


def test_critic_review_candidates_specialize_joint_original_bill_status() -> None:
    raw = "--- PAGE 1 ---\nConsigned to order of\nNumber of Original FBL's\n3/THREE\n"
    surfaces = ("to order of", "3/THREE")
    starts = tuple(raw.index(surface) for surface in surfaces)
    drafts = tuple(
        SpanDraft(
            draft_id=f"status_{index}",
            logical_key="agent:negotiability:original_count",
            render_mode="agent_residual",
            value_kind="other_text",
            group_kind="document",
            group_key="document",
            target_paths=("documentPatch.negotiability",),
            derivation=None,
            dependency_paths=(),
            dependency_bindings=(),
            char_start=start,
            char_end=start + len(surface),
            source_text=surface,
            evidence_origin="host_verified_agent_proposal",
            render_policy="natural_text",
            rationale="Joint selected status fixture.",
        )
        for index, (surface, start) in enumerate(zip(surfaces, starts, strict=True))
    )

    candidates = _critic_review_candidates(
        raw=raw,
        drafts=drafts,
        source_target={"documentPatch": {"negotiability": "negotiable"}},
    )

    review = next(row for row in candidates if row["kind"] == "agent_residual_contract_review")
    assert "one joint document-status contract" in review["details"]["relationshipToVerify"]
    assert "Do not require either occurrence" in review["details"]["relationshipToVerify"]


def test_critic_review_candidates_group_unowned_standalone_document_status() -> None:
    raw = "--- PAGE 1 ---\nORIGINAL BILL OF LADING\nORIGINAL\nORIGINAL\nNON-NEGOTIABLE\n"

    candidates = _critic_review_candidates(raw=raw, drafts=())
    status_rows = {
        row["sourceTexts"][0]: row
        for row in candidates
        if row["kind"] == "unowned_standalone_document_status"
    }

    assert set(status_rows) == {"ORIGINAL", "NON-NEGOTIABLE"}
    assert status_rows["ORIGINAL"]["lineIds"] == ("L00003", "L00004")
    assert all("ORIGINAL BILL OF LADING" not in row["sourceTexts"] for row in candidates)


def test_critic_review_candidates_do_not_extend_original_mark_into_form_title() -> None:
    raw = "--- PAGE 1 ---\nORIGINAL BILL OF LADING\nORIGINAL\nORIGINAL\n"
    starts = tuple(match.start() for match in re.finditer(r"(?m)^ORIGINAL$", raw))
    drafts = tuple(
        SpanDraft(
            draft_id=f"original_{index}",
            logical_key="agent:document:original_mark",
            render_mode="deterministic_auxiliary",
            value_kind="other_text",
            group_kind="document",
            group_key="document:original_status",
            target_paths=(),
            derivation=None,
            dependency_paths=(),
            dependency_bindings=(),
            char_start=start,
            char_end=start + len("ORIGINAL"),
            source_text="ORIGINAL",
            evidence_origin="host_verified_agent_proposal",
            render_policy="natural_text",
            rationale="Selected standalone mark fixture.",
        )
        for index, start in enumerate(starts)
    )

    candidates = _critic_review_candidates(raw=raw, drafts=drafts)

    assert not any(
        row["kind"] == "unowned_exact_repeat" and row["sourceTexts"] == ("ORIGINAL",)
        for row in candidates
    )


def test_critic_review_candidates_group_selected_domain_vocabulary() -> None:
    raw = (
        "--- PAGE 1 ---\n"
        "1 CONT. 40 REEFER CONTAINER SLAC*\n"
        "TYPE: VAT NUMBER\n"
        "21978,000 KGM\n"
        "*SLAC = Shipper's Load, Stow, Weight and Count\n"
    )

    candidates = _critic_review_candidates(raw=raw, drafts=())
    domain_rows = {
        (row["details"]["semanticHint"], row["sourceTexts"][0]): row
        for row in candidates
        if row["kind"] == "domain_vocabulary_surface_review"
    }

    assert set(domain_rows) == {
        ("identifier_type", "TYPE: VAT NUMBER"),
        ("measurement_unit", "KGM"),
        ("operational_qualifier", "SLAC*"),
    }
    assert domain_rows[("operational_qualifier", "SLAC*")]["lineIds"] == ("L00002",)


def test_critic_review_candidates_flag_package_valued_total() -> None:
    raw = "--- PAGE 1 ---\n========================= 3084 PACKAGE KGM\n"
    source_text = "3084 PACKAGE"
    start = raw.index(source_text)
    draft = SpanDraft(
        draft_id="package_total",
        logical_key="agent:total_package_quantity",
        render_mode="deterministic_auxiliary",
        value_kind="package",
        group_kind="package",
        group_key="package:total",
        target_paths=(),
        derivation=None,
        dependency_paths=(),
        dependency_bindings=(),
        char_start=start,
        char_end=start + len(source_text),
        source_text=source_text,
        evidence_origin="host_verified_agent_proposal",
        render_policy="natural_text",
        rationale="Fixture package total.",
    )

    candidates = _critic_review_candidates(raw=raw, drafts=(draft,))

    totals = [row for row in candidates if row["kind"] == "potential_calculated_total_derivation"]
    assert len(totals) == 1
    assert totals[0]["sourceTexts"] == ("3084 PACKAGE",)


def test_critic_review_candidates_force_first_pass_derivation_checks() -> None:
    raw = (
        "--- PAGE 1 ---\n"
        "COUNTRY: POLAND\n"
        "COUNTRY CODE: PL\n"
        "WEIGHT: 21978,000\n"
        "TOTAL WEIGHT: 43956,000\n"
        "1 X 20' STD CONTAINER\n"
    )

    def draft(
        text: str,
        *,
        logical_key: str,
        render_mode: str,
        value_kind: str,
        group_kind: str,
        group_key: str,
        target_paths: tuple[str, ...] = (),
    ) -> SpanDraft:
        start = raw.index(text)
        return SpanDraft(
            draft_id=logical_key,
            logical_key=logical_key,
            render_mode=render_mode,
            value_kind=value_kind,
            group_kind=group_kind,
            group_key=group_key,
            target_paths=target_paths,
            derivation=None,
            dependency_paths=(),
            dependency_bindings=(),
            char_start=start,
            char_end=start + len(text),
            source_text=text,
            evidence_origin="host_verified_agent_proposal",
            render_policy="natural_text",
            rationale="Fixture binding.",
        )

    drafts = (
        draft(
            "POLAND",
            logical_key="agent:foreign_exporter_country",
            render_mode="deterministic_auxiliary",
            value_kind="location",
            group_kind="customs",
            group_key="customs:foreign_exporter",
        ),
        draft(
            "PL",
            logical_key="agent:foreign_exporter_country_code",
            render_mode="deterministic_auxiliary",
            value_kind="identifier",
            group_kind="customs",
            group_key="customs:foreign_exporter",
        ),
        draft(
            "21978,000",
            logical_key="agent:gross_weight_21978",
            render_mode="deterministic_auxiliary",
            value_kind="decimal_measurement",
            group_kind="cargo",
            group_key="cargo:weight",
        ),
        draft(
            "43956,000",
            logical_key="agent:total_weight_43956",
            render_mode="deterministic_auxiliary",
            value_kind="decimal_measurement",
            group_kind="cargo",
            group_key="cargo:weight-total",
        ),
        draft(
            "1 X 20' STD CONTAINER",
            logical_key="agent:container:0:receipt",
            render_mode="target_binding",
            value_kind="equipment",
            group_kind="equipment",
            group_key="container:0",
            target_paths=("documentPatch.containers[0]",),
        ),
    )

    candidates = _critic_review_candidates(raw=raw, drafts=drafts)
    by_kind = {row["kind"]: row for row in candidates}

    assert by_kind["potential_country_code_derivation"]["logicalKeys"] == (
        "agent:foreign_exporter_country_code",
        "agent:foreign_exporter_country",
    )
    assert by_kind["potential_country_code_derivation"]["sourceTexts"] == (
        "PL",
        "POLAND",
    )
    assert by_kind["potential_country_code_derivation"]["details"][
        "possibleDependencyLogicalKeys"
    ] == ("agent:foreign_exporter_country",)
    assert by_kind["potential_country_code_derivation"]["details"]["suggestedDerivations"] == (
        "country_code",
    )
    assert by_kind["potential_calculated_total_derivation"]["sourceTexts"] == ("43956,000",)
    assert by_kind["potential_equipment_receipt_derivation"]["sourceTexts"] == (
        "1 X 20' STD CONTAINER",
    )


def test_critic_review_candidates_do_not_flag_direct_formatted_scalar_as_derivation() -> None:
    raw = "--- PAGE 1 ---\nHS CODE: 3214.10.0010\n"
    text = "3214.10.0010"
    start = raw.index(text)
    draft = SpanDraft(
        draft_id="hs",
        logical_key="anchor:documentPatch.cargoGroups[0].hsCodes[0]",
        render_mode="target_binding",
        value_kind="identifier",
        group_kind="cargo",
        group_key="cargo:0",
        target_paths=("documentPatch.cargoGroups[0].hsCodes[0]",),
        derivation=None,
        dependency_paths=(),
        dependency_bindings=(),
        char_start=start,
        char_end=start + len(text),
        source_text=text,
        evidence_origin="host_verified_agent_proposal",
        render_policy="opaque_identifier",
        rationale="Host-proven identifier formatting projection.",
    )

    candidates = _critic_review_candidates(raw=raw, drafts=(draft,))

    assert not any(row["kind"].startswith("potential_") for row in candidates)


def test_critic_review_candidates_expose_partial_and_framed_target_surfaces() -> None:
    raw = "--- PAGE 1 ---\nCONSIGNEE ADDRESS\nAL-BAHER\nST.\nFAYOUM\nTERMS: SEA FREIGHT PREPAID\n"
    source_target = {
        "documentPatch": {
            "parties": {"consignee": {"address": "AL-BAHER ST., FAYOUM"}},
            "paymentArrangement": "PREPAID",
        }
    }

    def target_draft(
        source_text: str,
        *,
        logical_key: str,
        target_path: str,
        value_kind: str,
        start: int | None = None,
    ) -> SpanDraft:
        char_start = raw.index(source_text) if start is None else start
        return SpanDraft(
            draft_id=logical_key,
            logical_key=logical_key,
            render_mode="target_binding",
            value_kind=value_kind,
            group_kind="fixture",
            group_key="fixture:0",
            target_paths=(target_path,),
            derivation=None,
            dependency_paths=(),
            dependency_bindings=(),
            char_start=char_start,
            char_end=char_start + len(source_text),
            source_text=source_text,
            evidence_origin="host_verified_agent_proposal",
            render_policy="natural_text",
            rationale="Fixture target surface.",
        )

    drafts = (
        target_draft(
            "AL-BAHER\nST.",
            logical_key="anchor:consignee_address",
            target_path="documentPatch.parties.consignee.address",
            value_kind="address",
        ),
        target_draft(
            "FREIGHT PREPAID",
            logical_key="anchor:payment_arrangement",
            target_path="documentPatch.paymentArrangement",
            value_kind="commercial_text",
        ),
    )

    candidates = _critic_review_candidates(
        raw=raw,
        drafts=drafts,
        source_target=source_target,
    )
    by_kind = {row["kind"]: row for row in candidates}

    assert by_kind["nearby_omitted_target_tokens"]["logicalKeys"] == ("anchor:consignee_address",)
    assert by_kind["nearby_omitted_target_tokens"]["lineIds"] == ("L00003", "L00005")
    omitted = by_kind["nearby_omitted_target_tokens"]["details"]["occurrences"][0]
    assert omitted["omittedTargetSuffixTokens"] == ("fayoum",)
    assert omitted["unownedLineIds"] == ("L00005",)
    assert "FAYOUM" in omitted["nearbyUnownedText"]
    assert by_kind["target_binding_alphanumeric_frame"]["logicalKeys"] == (
        "anchor:payment_arrangement",
    )
    frame = by_kind["target_binding_alphanumeric_frame"]["details"]["occurrences"][0]
    assert frame["literalPrefix"] == "FREIGHT "
    assert frame["literalSuffix"] == ""
    assert "SEA" in frame["nearbyUnownedText"]


def test_repeated_literal_candidate_links_normalized_existing_owner() -> None:
    raw = "--- PAGE 1 ---\nFCL / FCL\n/FCL/FCL/\n/FCL/FCL/\n"
    start = raw.index("FCL / FCL")
    owner = SpanDraft(
        draft_id="movement",
        logical_key="agent:movement_type",
        render_mode="deterministic_auxiliary",
        value_kind="operational_text",
        group_kind="transport",
        group_key="transport:movement",
        target_paths=(),
        derivation=None,
        dependency_paths=(),
        dependency_bindings=(),
        char_start=start,
        char_end=start + len("FCL / FCL"),
        source_text="FCL / FCL",
        evidence_origin="host_verified_agent_proposal",
        render_policy="token_projection",
        rationale="Fixture movement owner.",
    )

    candidates = _critic_review_candidates(raw=raw, drafts=(owner,))
    repeated = next(row for row in candidates if row["kind"] == "unowned_repeated_literal_surface")

    assert repeated["logicalKeys"] == ("agent:movement_type",)
    assert repeated["lineIds"] == ("L00003", "L00004")
    assert repeated["details"]["possibleNormalizedOwnerContexts"][0]["logicalKey"] == (
        "agent:movement_type"
    )


def test_country_code_candidate_carries_same_group_dependency_context() -> None:
    raw = "--- PAGE 1 ---\nCOUNTRY: POLAND\nCOUNTRY CODE: PL\n"
    country_start = raw.index("POLAND")
    code_start = raw.rindex("PL")

    def location_draft(
        *, draft_id: str, logical_key: str, start: int, source_text: str
    ) -> SpanDraft:
        return SpanDraft(
            draft_id=draft_id,
            logical_key=logical_key,
            render_mode="deterministic_auxiliary",
            value_kind="location",
            group_kind="party",
            group_key="party:foreign_exporter:country",
            target_paths=(),
            derivation=None,
            dependency_paths=(),
            dependency_bindings=(),
            char_start=start,
            char_end=start + len(source_text),
            source_text=source_text,
            evidence_origin="host_verified_agent_proposal",
            render_policy="natural_text",
            rationale="Fixture location.",
        )

    candidates = _critic_review_candidates(
        raw=raw,
        drafts=(
            location_draft(
                draft_id="country",
                logical_key="agent:foreign_exporter_country",
                start=country_start,
                source_text="POLAND",
            ),
            location_draft(
                draft_id="code",
                logical_key="agent:foreign_exporter_country_code",
                start=code_start,
                source_text="PL",
            ),
        ),
    )
    candidate = next(
        row for row in candidates if row["kind"] == "potential_country_code_derivation"
    )

    assert candidate["logicalKeys"] == (
        "agent:foreign_exporter_country_code",
        "agent:foreign_exporter_country",
    )
    assert candidate["lineIds"] == ("L00002", "L00003")
    assert candidate["sourceTexts"] == ("PL", "POLAND")
    assert candidate["details"]["possibleDependencyLogicalKeys"] == (
        "agent:foreign_exporter_country",
    )


def test_critic_review_candidates_question_equal_dates_in_different_contexts() -> None:
    raw = "--- PAGE 1 ---\nSHIPPED ON 2026-04-03\nISSUED ON 2026-04-03\n"
    source_text = "2026-04-03"
    first_start = raw.index(source_text)
    second_start = raw.index(source_text, first_start + 1)
    target_path = "documentPatch.shipmentDate"
    source_target = {"documentPatch": {"shipmentDate": source_text}}

    def date_draft(draft_id: str, char_start: int) -> SpanDraft:
        return SpanDraft(
            draft_id=draft_id,
            logical_key="anchor:shipment_date",
            render_mode="target_binding",
            value_kind="date",
            group_kind="transport",
            group_key="transport:dates",
            target_paths=(target_path,),
            derivation=None,
            dependency_paths=(),
            dependency_bindings=(),
            char_start=char_start,
            char_end=char_start + len(source_text),
            source_text=source_text,
            evidence_origin="host_verified_agent_proposal",
            render_policy="date_surface",
            rationale="Fixture repeated date.",
        )

    candidates = _critic_review_candidates(
        raw=raw,
        drafts=(date_draft("date_1", first_start), date_draft("date_2", second_start)),
        source_target=source_target,
    )

    context_reviews = [
        row for row in candidates if row["kind"] == "repeated_binding_context_review"
    ]
    assert len(context_reviews) == 1
    assert context_reviews[0]["lineIds"] == ("L00002", "L00003")
    assert context_reviews[0]["targetPaths"] == (target_path,)


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


def test_inventory_preserves_overlapping_exact_occurrence_index() -> None:
    raw = "--- PAGE 1 ---\nPG III, (25C.C.C.)\n"
    source_text = "C.C."
    first_start = raw.index(source_text)
    second_start = raw.index(source_text, first_start + 1)
    draft = SpanDraft(
        draft_id="agent_binding_flash_point_method",
        logical_key="agent:dangerous_goods:flash_point_method",
        render_mode="deterministic_auxiliary",
        value_kind="dangerous_goods",
        group_kind="dangerous_goods",
        group_key="dangerous_goods:0",
        target_paths=(),
        derivation=None,
        dependency_paths=(),
        dependency_bindings=(),
        char_start=second_start,
        char_end=second_start + len(source_text),
        source_text=source_text,
        evidence_origin="host_verified_agent_proposal",
        render_policy="exact_surface",
        rationale="Fixture overlapping occurrence.",
    )

    inventory = _draft_inventory(raw, (draft,), {"documentPatch": {}})
    occurrence = inventory[0]["occurrences"][0]

    assert occurrence["occurrenceIndex"] == 1
    assert occurrence["exactMatchCount"] == 2
    assert [row["rangeRelativeCharStart"] for row in occurrence["exactMatchCandidates"]] == [
        first_start - raw.index("PG III"),
        second_start - raw.index("PG III"),
    ]
    assert [row["selected"] for row in occurrence["exactMatchCandidates"]] == [False, True]


def test_compiler_candidate_preserves_overlapping_exact_occurrence_index() -> None:
    raw = "--- PAGE 1 ---\nPG III, (25C.C.C.)\n"
    surface = "C.C."

    candidates = _compiler_occurrence_candidates(
        raw=raw,
        source_target={"documentPatch": {}},
        anchors=(),
        risks=(
            type(
                "Risk",
                (),
                {"source_text": surface},
            )(),
        ),
    )

    selected = tuple(row for row in candidates if row["sourceText"] == surface)
    # The leading match ends inside the larger token and is deliberately not exposed. The safe
    # trailing match must nevertheless retain its true overlapping occurrence index.
    assert tuple(row["occurrenceIndex"] for row in selected) == (1,)


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


def test_rejected_candidate_inventory_exposes_invalid_target_path() -> None:
    raw = "--- PAGE 1 ---\nHS 2401\n"
    invalid_path = "documentPatch.cargoGroups[0].hsCodes[1]"
    draft = SpanDraft(
        draft_id="invalid_path_draft",
        logical_key="anchor:" + invalid_path,
        render_mode="target_binding",
        value_kind="identifier",
        group_kind="cargo",
        group_key="cargo:0",
        target_paths=(invalid_path,),
        derivation=None,
        dependency_paths=(),
        dependency_bindings=(),
        char_start=18,
        char_end=22,
        source_text="2401",
        evidence_origin="host_verified_agent_proposal",
        render_policy="identifier_exact",
        rationale="Fixture invalid indexed path.",
    )
    source_target = {"documentPatch": {"cargoGroups": [{"hsCodes": ["2401"]}]}}

    with pytest.raises(ValueError, match="target path index does not exist"):
        _draft_inventory(raw, (draft,), source_target)

    inventory = _draft_inventory(
        raw,
        (draft,),
        source_target,
        rejected_candidate=True,
    )
    assert inventory[0]["targetRelationship"] == "invalid_target_path"
    assert inventory[0]["independentTargetFactComponents"] == ((invalid_path,),)
    assert inventory[0]["targetPathValidationError"] == (
        "target path index does not exist in source label: " + invalid_path
    )


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


def _provider_error_stage(role: str, pass_number: int) -> AgentStageArtifact:
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
            "status": "provider_error",
            "output": None,
            "error_type": "UnexpectedModelBehavior",
            "error_message": "Exceeded maximum output retries (fixture)",
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


def test_compiler_host_rejection_survives_overlapping_equipment_diagnostics() -> None:
    raw = "--- PAGE 1 ---\nABCU1234567 96 7025157\n"
    source_target = {
        "documentPatch": {
            "containers": [
                {
                    "containerNumber": "ABCU1234567",
                    "sealNumbers": ["7025157"],
                }
            ]
        }
    }
    candidate = CompilerAgentOutput.model_validate(
        {
            "carrier": {
                "canonical_name": "Example Carrier",
                "aliases": (),
                "evidence_occurrences": (
                    {
                        "line_start": "L00002",
                        "line_end": "L00002",
                        "source_text": "ABCU1234567",
                        "occurrence_index": 0,
                    },
                ),
                "source": "source_label_confirmed_by_ocr",
                "rationale": "Fixture carrier evidence is not evaluated by this diagnostic.",
            },
            "anchor_overrides": (),
            "bindings": (
                {
                    "logical_key": "anchor:documentPatch.containers[0].containerNumber",
                    "render_mode": "target_binding",
                    "value_kind": "equipment",
                    "group_kind": "equipment",
                    "group_key": "container:0",
                    "target_paths": ("documentPatch.containers[0].containerNumber",),
                    "occurrences": (
                        {
                            "line_start": "L00002",
                            "line_end": "L00002",
                            "source_text": "ABCU1234567",
                            "occurrence_index": 0,
                        },
                    ),
                    "rationale": "Exact container number.",
                },
                {
                    "logical_key": "agent:container_row",
                    "render_mode": "deterministic_auxiliary",
                    "value_kind": "equipment",
                    "group_kind": "equipment",
                    "group_key": "container:0",
                    "target_paths": (),
                    "occurrences": (
                        {
                            "line_start": "L00002",
                            "line_end": "L00002",
                            "source_text": "ABCU1234567 96 7025157",
                            "occurrence_index": 0,
                        },
                    ),
                    "rationale": "Intentionally overlapping row proposal.",
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
        anchors=(),
        primary_error=ValueError("template spans overlap:: fixture overlap"),
    )

    assert "complete candidate validation: template spans overlap:: fixture overlap" in str(
        rejection
    )
    assert "binding locality normalization" in str(rejection)
    assert "template spans overlap" in str(rejection)


def test_compiler_host_rejection_checks_topology_after_semantic_normalization() -> None:
    raw = "--- PAGE 1 ---\nTEMP +1 C\nSETPOINT +1,0 C\n"
    unit_path = "documentPatch.containers[0].temperatureSetpoint.unit"
    value_path = "documentPatch.containers[0].temperatureSetpoint.value"
    source_target = {
        "documentPatch": {
            "containers": [{"temperatureSetpoint": {"unit": "celsius", "value": 1.0}}]
        }
    }
    candidate = CompilerAgentOutput.model_validate(
        {
            "carrier": {
                "canonical_name": "Example Carrier",
                "aliases": (),
                "evidence_occurrences": (
                    {
                        "line_start": "L00002",
                        "line_end": "L00002",
                        "source_text": "TEMP",
                        "occurrence_index": 0,
                    },
                ),
                "source": "source_label_confirmed_by_ocr",
                "rationale": "Carrier evidence is outside this diagnostic.",
            },
            "anchor_overrides": (),
            "bindings": (
                {
                    "logical_key": "container:0:temperature_setpoint",
                    "render_mode": "target_binding",
                    "value_kind": "temperature",
                    "group_kind": "equipment",
                    "group_key": "container:0",
                    "target_paths": (unit_path, value_path),
                    "occurrences": (
                        {
                            "line_start": "L00002",
                            "line_end": "L00002",
                            "source_text": "+1 C",
                            "occurrence_index": 0,
                        },
                        {
                            "line_start": "L00003",
                            "line_end": "L00003",
                            "source_text": "+1,0 C",
                            "occurrence_index": 0,
                        },
                    ),
                    "rationale": "Repeated formatted temperature setpoint.",
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
        anchors=(),
        primary_error=ValueError("unrelated host defect"),
    )

    assert "complete candidate validation: unrelated host defect" in str(rejection)
    assert "binding target topology" not in str(rejection)


class _FakeRuntime:
    def __init__(self, compiler: CompilerAgentOutput, critics: tuple[CriticAgentOutput, ...]):
        self.compiler_output = compiler
        self.critic_outputs = list(critics)
        self.compiler_calls = 0
        self.critic_calls = 0
        self.critic_prior_errors: list[str | None] = []
        self.critic_payloads: list[dict[str, Any]] = []
        self.critic_candidate_only: list[bool] = []

    async def compiler(self, **kwargs: Any):
        self.compiler_calls += 1
        return self.compiler_output, _stage("compiler", kwargs["pass_number"], self.compiler_output)

    async def critic(self, **kwargs: Any):
        self.critic_calls += 1
        self.critic_candidate_only.append(bool(kwargs.get("candidate_only", False)))
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
async def test_source_integrity_contradiction_is_reviewed_before_provider_call(
    tmp_path: Path,
) -> None:
    raw = (
        "--- PAGE 1 ---\n"
        "CARRIER'S RECEIPT: Total number of containers received by Carrier\n"
        "(FOUR) CONTAINER(S) ONLY\n"
    )
    source_target = {
        "schemaVersion": "3.0.0-experimental",
        "documentPatch": {
            "containers": [
                {"containerNumber": "AAAA0000000"},
                {"containerNumber": "BBBB0000000"},
                {"containerNumber": "CCCC0000000"},
            ]
        },
    }
    source = {
        "documentId": "doc_source_integrity_review",
        "joinedRawText": raw,
        "joinedRawTextSha256": sha256_bytes(raw.encode("utf-8")),
        "target": source_target,
    }

    class NoCallRuntime:
        compiler_calls = 0
        critic_calls = 0

        async def compiler(self, **_kwargs: Any):
            self.compiler_calls += 1
            pytest.fail("source integrity review must precede compiler launch")

        async def critic(self, **_kwargs: Any):
            self.critic_calls += 1
            pytest.fail("source integrity review must precede critic launch")

    runtime = NoCallRuntime()
    staged = _FakeStagedRun(tmp_path)
    config = load_config(_EXPERIMENT_ROOT / "configs" / "canary1.yaml").model_copy(
        update={"resume_from": None}
    )

    result, template, masked = await _extract_case(
        ordinal=1,
        document_id="doc_source_integrity_review",
        source=source,
        feature={},
        anchor_rows=(),
        runtime=runtime,  # type: ignore[arg-type]
        compiler_prompt="compiler",
        critic_prompt="critic",
        config=config,
        staged=staged,  # type: ignore[arg-type]
        document_limiter=asyncio.Semaphore(1),
    )

    assert result.status == "review_required"
    assert result.resume_mode == "source_integrity_review"
    assert result.compiler_stages == ()
    assert result.critic_stages == ()
    assert "4_vs_3" in result.rejection_reasons[0]
    assert runtime.compiler_calls == 0
    assert runtime.critic_calls == 0
    assert template is None
    assert masked == raw
    assert "cases/doc_source_integrity_review/template.json" not in staged.published


class _RecoveringRuntime:
    def __init__(
        self,
        compilers: tuple[CompilerAgentOutput | None, ...],
        critics: tuple[CriticAgentOutput | None, ...],
    ) -> None:
        self.compilers = list(compilers)
        self.critics = list(critics)
        self.compiler_calls = 0
        self.critic_calls = 0

    async def compiler(self, **kwargs: Any):
        self.compiler_calls += 1
        output = self.compilers.pop(0)
        stage = (
            _stage("compiler", kwargs["pass_number"], output)
            if output is not None
            else _provider_error_stage("compiler", kwargs["pass_number"])
        )
        return output, stage

    async def critic(self, **kwargs: Any):
        self.critic_calls += 1
        output = self.critics.pop(0)
        stage = (
            _stage("critic", kwargs["pass_number"], output)
            if output is not None
            else _provider_error_stage("critic", kwargs["pass_number"])
        )
        return output, stage


@pytest.mark.asyncio
async def test_provider_output_failure_retries_within_configured_stage_budgets(
    tmp_path: Path,
) -> None:
    carrier = "Acme Ocean Lines Ltd."
    heading = "BILL OF LADING"
    raw = f"--- PAGE 1 ---\n{heading}\n{carrier}\n"
    source_target = {
        "schemaVersion": "3.0.0-experimental",
        "documentPatch": {"parties": {"carrier": {"name": carrier}}},
    }
    source = {
        "documentId": "doc_provider_recovery",
        "joinedRawText": raw,
        "joinedRawTextSha256": sha256_bytes(raw.encode("utf-8")),
        "target": source_target,
    }
    anchor_rows = (
        {
            "anchor_id": "anchor_carrier",
            "document_id": "doc_provider_recovery",
            "patchable": True,
            "page_number": 1,
            "page_start": len(heading) + 1,
            "page_end": len(heading) + 1 + len(carrier),
            "raw_value": carrier,
            "target_value": carrier,
            "relation_target_path": "documentPatch.parties.carrier.name",
            "role_path": "documentPatch.parties.carrier.name",
            "surface_family": "text",
        },
    )
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
    compiler = CompilerAgentOutput.model_validate(
        {
            "carrier": {
                "canonical_name": carrier,
                "aliases": (),
                "evidence_occurrences": (
                    {
                        "line_start": "L00003",
                        "line_end": "L00003",
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
        }
    )
    passed = CriticAgentOutput.model_validate(
        {
            "verdict": "pass",
            "findings": (),
            "additional_bindings": (),
            "rationale": "Every source surface is correctly classified.",
        }
    )
    runtime = _RecoveringRuntime((None, compiler), (None, passed))
    base = load_config(_EXPERIMENT_ROOT / "configs" / "canary1.yaml")
    config = base.model_copy(
        update={
            "resume_from": None,
            "workflow": base.workflow.model_copy(
                update={
                    "max_compiler_passes": 2,
                    "max_critic_passes": 2,
                    "additional_call_launch_threshold_usd_per_document": Decimal("1"),
                }
            ),
        }
    )

    result, template, _masked = await _extract_case(
        ordinal=1,
        document_id="doc_provider_recovery",
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
    assert runtime.compiler_calls == 2
    assert runtime.critic_calls == 2
    assert [stage.status for stage in result.compiler_stages] == ["provider_error", "success"]
    assert [stage.status for stage in result.critic_stages] == ["provider_error", "success"]


@pytest.mark.asyncio
async def test_candidate_prepass_pass_still_requires_distinct_full_critic_pass(
    tmp_path: Path,
) -> None:
    carrier = "Acme Ocean Lines Ltd."
    heading = "BILL OF LADING"
    raw = f"--- PAGE 1 ---\n{heading}\n{carrier}\n"
    source_target = {
        "schemaVersion": "3.0.0-experimental",
        "documentPatch": {"parties": {"carrier": {"name": carrier}}},
    }
    source = {
        "documentId": "doc_candidate_prepass",
        "joinedRawText": raw,
        "joinedRawTextSha256": sha256_bytes(raw.encode("utf-8")),
        "target": source_target,
    }
    anchor_rows = (
        {
            "anchor_id": "anchor_carrier",
            "document_id": "doc_candidate_prepass",
            "patchable": True,
            "page_number": 1,
            "page_start": len(heading) + 1,
            "page_end": len(heading) + 1 + len(carrier),
            "raw_value": carrier,
            "target_value": carrier,
            "relation_target_path": "documentPatch.parties.carrier.name",
            "role_path": "documentPatch.parties.carrier.name",
            "surface_family": "text",
        },
    )
    feature = {
        "document_type": "bill_of_lading",
        "template_proxy_id": "template_candidate_prepass",
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
    carrier_occurrence = {
        "line_start": "L00003",
        "line_end": "L00003",
        "source_text": carrier,
        "occurrence_index": 0,
    }
    compiler = CompilerAgentOutput.model_validate(
        {
            "carrier": {
                "canonical_name": carrier,
                "aliases": (),
                "evidence_occurrences": (carrier_occurrence,),
                "source": "source_label_confirmed_by_ocr",
                "rationale": "Exact carrier evidence.",
            },
            "anchor_overrides": (),
            "bindings": (),
            "unresolved": (),
            "all_shipment_dependent_surfaces_accounted_for": True,
        }
    )
    passed = CriticAgentOutput.model_validate(
        {
            "verdict": "pass",
            "findings": (),
            "additional_bindings": (),
            "rationale": "The assigned scope is valid.",
        }
    )
    runtime = _FakeRuntime(compiler, (passed, passed))
    base = load_config(_EXPERIMENT_ROOT / "configs" / "canary1.yaml")
    config = base.model_copy(
        update={
            "resume_from": None,
            "workflow": base.workflow.model_copy(
                update={
                    "agent_contract_protocol": "candidate_first_staged_local_v7",
                    "max_critic_passes": 2,
                    "additional_call_launch_threshold_usd_per_document": Decimal("1"),
                }
            ),
        }
    )

    result, template, _masked = await _extract_case(
        ordinal=1,
        document_id="doc_candidate_prepass",
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
    assert runtime.critic_calls == 2
    assert runtime.critic_candidate_only == [True, False]
    assert len(result.critic_stages) == 2


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
                    "max_critic_passes": 3,
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
    assert runtime.critic_calls == config.workflow.max_critic_passes + 1
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
    (
        chained_drafts,
        _chained_assessment,
        _chained_semantic,
        _chained_coherence,
        _chained_reviews,
        _warning,
    ) = _replay_checkpoint_case(
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
        recovered_coherence,
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
    assert recovered_coherence == ()
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
        _replayed_coherence,
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
    replayed, replayed_assessment, semantic_only, replayed_coherence, replayed_reviews, warning = (
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
    assert replayed_coherence == ()
    assert replayed_reviews == []
    assert warning is not None
    assert "skipped incompatible prior critic pass 1" in warning

    (
        fallback_drafts,
        fallback_assessment,
        _semantic,
        _coherence,
        _reviews,
        fallback_warning,
    ) = _restore_or_replay_resume_state(
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
    assert template.schema_version == 5
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
