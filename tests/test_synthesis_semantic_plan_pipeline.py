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


def test_semantic_composition_assigns_vessel_registry_change_to_controlled_stage() -> None:
    context, source, document_id = _fixture()
    vessel = source["documentPatch"]["transport"].get("vesselName")  # type: ignore[index,union-attr]
    if vessel is None:
        pytest.skip("semantic fixture has no vessel-name leaf")
    controlled = deepcopy(source)
    controlled["documentPatch"]["transport"]["vesselName"] = "PUBLIC REGISTRY VESSEL"  # type: ignore[index]

    state = compose_semantic_targets(
        context=context,
        base_document_id=document_id,
        variant_index=0,
        seed=5,
        structured_target=deepcopy(source),
        structured_plan={"changes": []},
        controlled_target=controlled,
        structured_provenance=_provenance("structured-test"),
        controlled_provenance=_provenance("controlled-test"),
    )

    change = state.changes[0]
    assert change.target_path == "documentPatch.transport.vesselName"
    assert change.stage_id == "controlled-semantics"
    assert change.change_kind == "identifier"
    assert change.method == "public_cargo_vessel_registry_uniform_v1"
    assert change.coupling_group == "transport"


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


def test_semantic_composition_ledgers_a_package_representation_change() -> None:
    row = next(
        json.loads(line)
        for line in SOURCE.read_text(encoding="utf-8").splitlines()
        if '"typeDescription":"SETS"' in line
    )
    source = row["target"]
    document_id = row["documentId"]
    context = SynthesisRunContext(
        task_adapter=BILL_OF_LADING_TASK_ADAPTER,
        scenario_namespace="semantic-package-representation-test-v1",
        source_records=(
            ScenarioSourceRecord.from_target(
                document_id=document_id,
                template_id="template_package_representation_test",
                target=source,
                source_raw_text_sha256=row["joinedRawTextSha256"],
            ),
        ),
    )
    controlled = deepcopy(source)
    package = controlled["documentPatch"]["cargoPackages"][0]
    assert package.pop("typeDescription") == "SETS"
    package["typeCategory"] = "PACKAGE_SET"

    state = compose_semantic_targets(
        context=context,
        base_document_id=document_id,
        variant_index=0,
        seed=5,
        structured_target=deepcopy(source),
        structured_plan={"changes": []},
        controlled_target=controlled,
        structured_provenance=_provenance("structured-test"),
        controlled_provenance=_provenance("controlled-test"),
    )

    output_package = state.target["documentPatch"]["cargoPackages"][0]
    assert output_package == {
        **{key: value for key, value in package.items()},
    }
    changes = {change.target_path: change for change in state.changes}
    added = changes["documentPatch.cargoPackages[0].typeCategory"]
    removed = changes["documentPatch.cargoPackages[0].typeDescription"]
    assert (added.old_present, added.new_present, added.new_value) == (
        False,
        True,
        "PACKAGE_SET",
    )
    assert (removed.old_present, removed.old_value, removed.new_present) == (
        True,
        "SETS",
        False,
    )


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
