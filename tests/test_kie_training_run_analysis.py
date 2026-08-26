from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from types import ModuleType

import pytest

from document_ocr.training.tasks import canonical_json, get_training_task


def _load_analysis_module() -> ModuleType:
    path = Path(__file__).parents[1] / "tools" / "analyze_kie_training_run.py"
    spec = importlib.util.spec_from_file_location("kie_training_run_analysis", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"could not load analysis module from {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


_ANALYSIS = _load_analysis_module()
DatasetRecord = _ANALYSIS.DatasetRecord
_index_insensitive_counts = _ANALYSIS._index_insensitive_counts
_epoch_zero_baseline = _ANALYSIS._epoch_zero_baseline
_metric_counts = _ANALYSIS._metric_counts
_normalize_path = _ANALYSIS._normalize_path
_reference_identity_index = _ANALYSIS._reference_identity_index
_resolved_checkpoint_path = _ANALYSIS._resolved_checkpoint_path
_schema_failure = _ANALYSIS._schema_failure


def _record(document_id: str, bill_number: str):
    return DatasetRecord(
        document_id=document_id,
        cohort="test",
        split="validation",
        source_path="validation.jsonl",
        raw_text="--- PAGE 1 ---\ntext",
        raw_text_sha256="0" * 64,
        target={
            "schemaVersion": "2.0.0",
            "documentPatch": {"billOfLadingNumber": bill_number},
        },
        page_count=1,
    )


def test_reference_identity_index_recovers_unique_validation_targets() -> None:
    task = get_training_task("bill_of_lading_semantic_v2")
    records = {"doc_a": _record("doc_a", "BL-A"), "doc_b": _record("doc_b", "BL-B")}

    index = _reference_identity_index(records, task)

    reference = canonical_json(task.canonicalize(records["doc_b"].target))
    assert index[reference] == "doc_b"


def test_reference_identity_index_rejects_ambiguous_duplicate_targets() -> None:
    task = get_training_task("bill_of_lading_semantic_v2")
    records = {"doc_a": _record("doc_a", "BL-A"), "doc_b": _record("doc_b", "BL-A")}

    with pytest.raises(ValueError, match="identity recovery is ambiguous"):
        _reference_identity_index(records, task)


def test_index_insensitive_metric_exposes_list_alignment_penalty() -> None:
    predicted = {
        ("$.documentPatch.goodsItems[0].packages[0].type", '"BAGS"'),
        ("$.documentPatch.goodsItems[1].packages[0].type", '"CARTONS"'),
    }
    reference = {
        ("$.documentPatch.goodsItems[0].packages[0].type", '"CARTONS"'),
        ("$.documentPatch.goodsItems[1].packages[0].type", '"BAGS"'),
    }

    strict = _metric_counts(predicted, reference)
    index_insensitive = _index_insensitive_counts(predicted, reference)

    assert strict.f1 == 0.0
    assert index_insensitive.f1 == 1.0
    assert _normalize_path("$.documentPatch.goodsItems[4].packages[2].type") == (
        "$.documentPatch.goodsItems[].packages[].type"
    )


def test_schema_failure_distinguishes_json_schema_and_valid_output() -> None:
    task = get_training_task("bill_of_lading_semantic_v2")
    invalid_json = '{"schemaVersion":"2.0.0"'
    invalid_schema = (
        '{"documentPatch":{"billOfLadingNumber":" BL-A"},"schemaVersion":"2.0.0"}'
    )
    valid = '{"documentPatch":{"billOfLadingNumber":"BL-A"},"schemaVersion":"2.0.0"}'

    assert _schema_failure(invalid_json, task)[0] == "invalid_json"
    assert _schema_failure(invalid_schema, task)[0] == "schema_validation_error"
    assert _schema_failure(valid, task) == ("none", "", [])


def test_resolved_checkpoint_path_confines_resolution_to_run(tmp_path: Path) -> None:
    checkpoint = tmp_path / "checkpoints" / "checkpoint-450"
    checkpoint.mkdir(parents=True)

    assert (
        _resolved_checkpoint_path(tmp_path, "/workspace/old/run/checkpoints/checkpoint-450")
        == checkpoint
    )
    with pytest.raises(ValueError, match="unexpected best checkpoint path"):
        _resolved_checkpoint_path(tmp_path, "/workspace/old/run/final-adapter")


def test_epoch_zero_baseline_requires_explicit_adapter_disabled_marker() -> None:
    environment = {
        "source_code": [
            {"module": "document_ocr.training.runtime", "sha256": "a" * 64}
        ]
    }

    certified = _epoch_zero_baseline(
        [{"epoch": 0, "eval_is_base_model": 1.0}], environment
    )
    unmarked = _epoch_zero_baseline([{"epoch": 0}], environment)

    assert certified["certified_base_model"] is True
    assert "adapter-disabled" in certified["reason"]
    assert unmarked["certified_base_model"] is False
