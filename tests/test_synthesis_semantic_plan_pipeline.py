from __future__ import annotations

import json
from copy import deepcopy
from pathlib import Path

import pytest

from document_ocr.hashing import sha256_file
from document_ocr.synthesis.run_context import ScenarioSourceRecord, SynthesisRunContext
from document_ocr.synthesis.run_safety import StagedArtifactRun, StagedRunError
from document_ocr.synthesis.scenario_state import StageProvenanceReceipt
from document_ocr.synthesis.semantic_plan_pipeline import (
    SemanticPlanPipelineError,
    _changed_paths,
    _set_existing_leaf,
    _validate_committed_dependency,
    compose_semantic_targets,
)
from document_ocr.synthesis.task_adapter import BILL_OF_LADING_TASK_ADAPTER

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SOURCE = (
    PROJECT_ROOT / "artifacts/kie-training/datasets/"
    "mpci-bl-combined1157-task-facing-package-categories-v2/records.jsonl"
)


def _fixture() -> tuple[SynthesisRunContext, dict[str, object], str]:
    row = json.loads(SOURCE.read_text(encoding="utf-8").splitlines()[0])
    target = row["target"]
    source = ScenarioSourceRecord.from_target(
        document_id=row["documentId"],
        template_id="template_semantic_test",
        target=target,
        source_raw_text_sha256=row["joinedRawTextSha256"],
    )
    return (
        SynthesisRunContext(
            task_adapter=BILL_OF_LADING_TASK_ADAPTER,
            scenario_namespace="semantic-test-v1",
            source_records=(source,),
        ),
        target,
        row["documentId"],
    )


def _provenance(name: str) -> tuple[StageProvenanceReceipt, ...]:
    return (StageProvenanceReceipt(name=name, sha256="a" * 64),)


def test_semantic_composition_applies_disjoint_stages_and_retains_exact_history() -> None:
    context, source, document_id = _fixture()
    structured = deepcopy(source)
    structured["documentPatch"]["billOfLadingNumber"] = "SYNTH-BL-9001"  # type: ignore[index]
    controlled = deepcopy(source)
    controlled["documentPatch"]["route"]["portOfLoading"]["name"] = "SYNTH PORT"  # type: ignore[index]
    plan = {
        "changes": [
            {
                "target_path": "documentPatch.billOfLadingNumber",
                "family": "document_identifier",
                "method": "shape_preserving_identifier_v1",
                "coupling_group": "document_identifiers",
            }
        ]
    }

    state = compose_semantic_targets(
        context=context,
        base_document_id=document_id,
        variant_index=0,
        seed=5,
        structured_target=structured,
        structured_plan=plan,
        controlled_target=controlled,
        structured_provenance=_provenance("structured-test"),
        controlled_provenance=_provenance("controlled-test"),
    )

    assert state.target["documentPatch"]["billOfLadingNumber"] == "SYNTH-BL-9001"
    assert state.target["documentPatch"]["route"]["portOfLoading"]["name"] == "SYNTH PORT"
    assert tuple(row.stage_id for row in state.stage_receipts) == (
        "structured",
        "controlled-semantics",
    )
    assert len(state.changes) == 2


def test_semantic_composition_rejects_overlapping_stage_ownership() -> None:
    context, source, document_id = _fixture()
    structured = deepcopy(source)
    controlled = deepcopy(source)
    structured["documentPatch"]["billOfLadingNumber"] = "SYNTH-A"  # type: ignore[index]
    controlled["documentPatch"]["billOfLadingNumber"] = "SYNTH-B"  # type: ignore[index]
    plan = {
        "changes": [
            {
                "target_path": "documentPatch.billOfLadingNumber",
                "family": "document_identifier",
                "method": "shape_preserving_identifier_v1",
                "coupling_group": "document_identifiers",
            }
        ]
    }

    with pytest.raises(SemanticPlanPipelineError, match="overlap change ownership"):
        compose_semantic_targets(
            context=context,
            base_document_id=document_id,
            variant_index=0,
            seed=5,
            structured_target=structured,
            structured_plan=plan,
            controlled_target=controlled,
            structured_provenance=_provenance("structured-test"),
            controlled_provenance=_provenance("controlled-test"),
        )


def test_leaf_patch_is_exact_and_presence_preserving() -> None:
    target = {"a": [{"b": "old"}]}

    _set_existing_leaf(target, "a[0].b", "new")

    assert target == {"a": [{"b": "new"}]}
    assert _changed_paths({"a": [{"b": "old"}]}, target) == ("a[0].b",)
    with pytest.raises(SemanticPlanPipelineError, match="absent"):
        _set_existing_leaf(target, "a[1].b", "bad")


def test_committed_dependency_uses_logical_transaction_identity(tmp_path: Path) -> None:
    transaction = "f" * 64
    run = StagedArtifactRun(
        output_parent=tmp_path,
        run_name="dependency-run",
        transaction_sha256=transaction,
    )
    run.publish_json("artifact.json", {"value": 1})
    run.commit(expected_artifacts=("artifact.json",), metadata={})
    root = tmp_path / "dependency-run"

    _validate_committed_dependency(
        root=root,
        commit_sha256=sha256_file(root / "_COMMIT.json"),
        transaction_sha256=transaction,
        label="test dependency",
    )

    with pytest.raises(StagedRunError, match="different transaction"):
        _validate_committed_dependency(
            root=root,
            commit_sha256=sha256_file(root / "_COMMIT.json"),
            transaction_sha256="e" * 64,
            label="test dependency",
        )
