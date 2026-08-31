from __future__ import annotations

import json
from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path

import pytest
from pydantic import ValidationError

from document_ocr.hashing import canonical_json_bytes, sha256_bytes
from document_ocr.synthesis.run_context import ScenarioSourceRecord, SynthesisRunContext
from document_ocr.synthesis.scenario_state import (
    ScenarioChange,
    ScenarioStageProposal,
    ScenarioState,
    advance_scenario_state,
    build_scenario_identity,
    deterministic_scenario_id,
)
from document_ocr.synthesis.task_adapter import BILL_OF_LADING_TASK_ADAPTER

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SOURCE = (
    PROJECT_ROOT / "artifacts/kie-training/datasets/"
    "mpci-bl-combined1157-task-facing-package-categories-v2/records.jsonl"
)


def _source() -> ScenarioSourceRecord:
    row = json.loads(SOURCE.read_text(encoding="utf-8").splitlines()[0])
    return ScenarioSourceRecord.from_target(
        document_id=row["documentId"],
        template_id="template_test",
        target=row["target"],
        source_raw_text_sha256=row["joinedRawTextSha256"],
    )


def _context() -> SynthesisRunContext:
    return SynthesisRunContext(
        task_adapter=BILL_OF_LADING_TASK_ADAPTER,
        scenario_namespace="unit-test-v1",
        source_records=(_source(),),
    )


def _bill_number_change(
    state: ScenarioState,
    *,
    stage_id: str = "identifiers",
    new_value: str = "SYNTHETIC-BL-1",
) -> tuple[dict[str, object], ScenarioChange]:
    target = deepcopy(state.target)
    patch = target["documentPatch"]
    assert isinstance(patch, dict)
    old_value = patch["billOfLadingNumber"]
    patch["billOfLadingNumber"] = new_value
    return target, ScenarioChange.model_validate(
        {
            "target_path": "documentPatch.billOfLadingNumber",
            "role_path": "documentPatch.billOfLadingNumber",
            "stage_id": stage_id,
            "change_kind": "identifier",
            "rendering_kind": "exact_scalar",
            "old_present": True,
            "old_value": old_value,
            "new_present": True,
            "new_value": new_value,
            "method": "unit_test_identifier_v1",
            "coupling_group": "document_identifiers",
        },
        strict=True,
    )


def _proposal(
    state: ScenarioState,
    *,
    stage_id: str = "identifiers",
    new_value: str = "SYNTHETIC-BL-1",
) -> ScenarioStageProposal:
    target, change = _bill_number_change(state, stage_id=stage_id, new_value=new_value)
    target_hash = sha256_bytes(canonical_json_bytes(target))
    return ScenarioStageProposal.model_validate(
        {
            "schema_version": 1,
            "stage_id": stage_id,
            "contract_id": f"{stage_id}-v1",
            "input_state_sha256": state.content_sha256(),
            "target": target,
            "target_sha256": target_hash,
            "changes": (change.model_dump(mode="python"),),
            "provenance": (),
        },
        strict=True,
    )


def test_scenario_identity_is_deterministic_and_stage_independent() -> None:
    values = {
        "task": "bill_of_lading_relation_explicit_v3",
        "scenario_namespace": "pilot-v1",
        "base_document_id": "doc_1",
        "template_id": "template_1",
        "seed": 17,
        "variant_index": 2,
    }

    first = deterministic_scenario_id(**values)
    second = build_scenario_identity(**values)

    assert first == second.scenario_id
    assert first.startswith("syn_")
    assert len(first) == 68
    assert deterministic_scenario_id(**{**values, "variant_index": 3}) != first


def test_bill_of_lading_task_adapter_validates_real_target_and_inverse() -> None:
    source = _source()

    canonical = BILL_OF_LADING_TASK_ADAPTER.validate_target(
        document_id=source.document_id,
        target=source.target,
    )

    assert canonical == source.target
    assert BILL_OF_LADING_TASK_ADAPTER.receipt.table_order[0] == "documents"
    assert BILL_OF_LADING_TASK_ADAPTER.receipt.table_order[-1] == "dangerous_goods"


def test_run_context_is_deterministic_and_returns_isolated_target_copy() -> None:
    context = _context()
    repeated = _context()
    source = context.source_record(context.document_ids[0])
    source.target["schemaVersion"] = "mutated"

    pristine = context.source_record(context.document_ids[0])

    assert context.receipt == repeated.receipt
    assert pristine.target["schemaVersion"] == "3.0.0-experimental"
    assert (
        context.initial_state(base_document_id=pristine.document_id, seed=9, variant_index=0).target
        == pristine.target
    )


def test_stage_transition_hashes_and_exact_ledger_are_chained() -> None:
    context = _context()
    state = context.initial_state(base_document_id=context.document_ids[0], seed=9, variant_index=0)
    proposal = _proposal(state)

    result = context.advance(state=state, proposal=proposal)

    assert result.input_state_sha256 == state.content_sha256()
    assert result.output_state_sha256 == result.output_state.content_sha256()
    assert result.output_state.identity == state.identity
    assert result.output_state.stage_receipts[0].change_paths == (
        "documentPatch.billOfLadingNumber",
    )
    assert result.output_state.target_sha256 == proposal.target_sha256


def test_stage_transition_rejects_unledgered_target_change() -> None:
    context = _context()
    state = context.initial_state(base_document_id=context.document_ids[0], seed=9, variant_index=0)
    proposal = _proposal(state)
    altered = deepcopy(proposal.target)
    patch = altered["documentPatch"]
    assert isinstance(patch, dict)
    patch["masterBillOfLadingNumber"] = "UNLEDGERED"
    invalid = ScenarioStageProposal.model_validate(
        {
            **proposal.model_dump(mode="python"),
            "target": altered,
            "target_sha256": sha256_bytes(canonical_json_bytes(altered)),
        },
        strict=True,
    )

    with pytest.raises(ValueError, match="change ledger mismatch"):
        advance_scenario_state(state=state, proposal=invalid)


def test_stage_transition_rejects_changed_path_owned_by_prior_stage() -> None:
    context = _context()
    initial = context.initial_state(
        base_document_id=context.document_ids[0], seed=9, variant_index=0
    )
    first = context.advance(state=initial, proposal=_proposal(initial)).output_state
    second = _proposal(first, stage_id="second-stage", new_value="SYNTHETIC-BL-2")

    with pytest.raises(ValueError, match="attempts to overwrite changed paths"):
        advance_scenario_state(state=first, proposal=second)


def test_stage_proposal_rejects_duplicate_paths() -> None:
    context = _context()
    state = context.initial_state(base_document_id=context.document_ids[0], seed=9, variant_index=0)
    proposal = _proposal(state)

    with pytest.raises(ValidationError, match="change paths must be unique"):
        ScenarioStageProposal.model_validate(
            {
                **proposal.model_dump(mode="python"),
                "changes": (*proposal.changes, *proposal.changes),
            },
            strict=True,
        )


def test_state_detects_nested_target_mutation() -> None:
    context = _context()
    state = context.initial_state(base_document_id=context.document_ids[0], seed=9, variant_index=0)
    state.target["schemaVersion"] = "mutated"

    with pytest.raises(ValueError, match="target SHA-256"):
        state.content_sha256()


@dataclass(frozen=True)
class _DummyStage:
    stage_id: str = "identifiers"
    contract_id: str = "identifiers-v1"

    def propose(
        self, *, context: SynthesisRunContext, state: ScenarioState
    ) -> ScenarioStageProposal:
        assert context.scenario_namespace == "unit-test-v1"
        return _proposal(state)


def test_run_context_executes_typed_stage_contract() -> None:
    context = _context()
    state = context.initial_state(base_document_id=context.document_ids[0], seed=9, variant_index=0)

    result = context.execute_stage(stage=_DummyStage(), state=state)

    assert result.stage_receipt.contract_id == "identifiers-v1"
