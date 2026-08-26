from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path
from types import ModuleType

from document_ocr.training.metrics import assess_prediction
from document_ocr.training.tasks import get_training_task


def _load_comparison_module() -> ModuleType:
    path = Path(__file__).parents[1] / "tools" / "compare_kie_training_runs.py"
    spec = importlib.util.spec_from_file_location("kie_training_run_comparison", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"could not load comparison module from {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


_COMPARISON = _load_comparison_module()
_bootstrap_rows = _COMPARISON._bootstrap_rows
_one_to_one_value_errors = _COMPARISON._one_to_one_value_errors
_repeated_ngram_fraction = _COMPARISON._repeated_ngram_fraction


def _assessment(predicted_patch: dict, reference_patch: dict):
    task = get_training_task("bill_of_lading_semantic_v2")
    predicted = json.dumps(
        {"schemaVersion": "2.0.0", "documentPatch": predicted_patch},
        separators=(",", ":"),
    )
    reference = json.dumps(
        {"schemaVersion": "2.0.0", "documentPatch": reference_patch},
        separators=(",", ":"),
    )
    return assess_prediction(predicted, reference, task)


def test_one_to_one_error_matching_never_reuses_prediction() -> None:
    assessment = _assessment(
        {"goodsItems": [{"description": "ALPHA"}]},
        {"goodsItems": [{"description": "ALPHA"}, {"description": "BETA"}]},
    )

    rows = _one_to_one_value_errors(
        assessment,
        ("$.documentPatch.goodsItems[].description",),
        "ALPHA BETA",
    )

    assert [row["error_kind"] for row in rows].count("strict_exact") == 1
    assert [row["error_kind"] for row in rows].count("omission") == 1
    assert [row["error_kind"] for row in rows].count("wrong_value") == 0


def test_one_to_one_error_matching_detects_correct_value_at_wrong_index() -> None:
    assessment = _assessment(
        {
            "goodsItems": [
                {"description": "BETA"},
                {"description": "ALPHA"},
            ]
        },
        {
            "goodsItems": [
                {"description": "ALPHA"},
                {"description": "BETA"},
            ]
        },
    )

    rows = _one_to_one_value_errors(
        assessment,
        ("$.documentPatch.goodsItems[].description",),
        "ALPHA BETA",
    )

    assert [row["error_kind"] for row in rows].count("correct_value_wrong_index") == 2
    assert all(row["error_kind"] != "wrong_value" for row in rows)


def test_repetition_fraction_separates_loop_from_short_unique_text() -> None:
    loop = " ".join(["container ABCD1234567 seal XYZ"] * 20)
    unique = "container ABCD1234567 seal XYZ package quantity 4 drums"

    assert _repeated_ngram_fraction(loop) > 0.8
    assert _repeated_ngram_fraction(unique) == 0.0


def test_paired_bootstrap_is_deterministic_and_document_resampled() -> None:
    rows = [
        {
            "previous_true_positive": 2,
            "previous_predicted": 4,
            "previous_reference": 4,
            "current_true_positive": 3,
            "current_predicted": 4,
            "current_reference": 4,
            "current_aligned_core_true_positive": 3,
            "current_aligned_core_predicted": 4,
            "current_aligned_core_reference": 4,
            "document_f1_delta": 0.25,
            "previous_json_valid": 1,
            "current_json_valid": 1,
            "previous_schema_valid": 1,
            "current_schema_valid": 1,
        },
        {
            "previous_true_positive": 3,
            "previous_predicted": 4,
            "previous_reference": 4,
            "current_true_positive": 2,
            "current_predicted": 4,
            "current_reference": 4,
            "current_aligned_core_true_positive": 2,
            "current_aligned_core_predicted": 4,
            "current_aligned_core_reference": 4,
            "document_f1_delta": -0.25,
            "previous_json_valid": 1,
            "current_json_valid": 0,
            "previous_schema_valid": 1,
            "current_schema_valid": 0,
        },
    ]

    first = _bootstrap_rows(rows)
    second = _bootstrap_rows(rows)

    assert first == second
    assert {row["resampling_unit"] for row in first} == {"paired_validation_document"}
    assert {row["bootstrap_replicates"] for row in first} == {10_000}
