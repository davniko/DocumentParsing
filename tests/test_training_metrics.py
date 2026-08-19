from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np

from document_ocr.training.metrics import make_compute_metrics, structured_metrics
from document_ocr.training.runtime import _predict_and_publish
from document_ocr.training.tasks import canonical_json, get_training_task


def _target(number: str) -> str:
    return canonical_json(
        {
            "schemaVersion": "2.0.0",
            "documentPatch": {"billOfLadingNumber": number},
        }
    )


def test_structured_metrics_distinguish_json_schema_and_exactness() -> None:
    task = get_training_task("bill_of_lading_semantic_v2")
    reference = _target("ABC")

    metrics, assessments = structured_metrics(
        [reference, _target("WRONG"), "not json"],
        [reference, reference, reference],
        task,
    )

    assert metrics["json_valid"] == 2 / 3
    assert metrics["schema_valid"] == 2 / 3
    assert metrics["canonical_exact_match"] == 1 / 3
    # schemaVersion is structural metadata, not an extracted field-value success.
    assert metrics["field_value_accuracy"] == 1 / 3
    assert metrics["field_value_precision"] == 1 / 2
    assert metrics["field_value_recall"] == 1 / 3
    assert metrics["field_value_f1"] == 0.4
    assert assessments[0].canonical_exact_match
    assert not assessments[2].json_valid


def test_field_value_metrics_penalize_missing_extra_and_wrong_values() -> None:
    task = get_training_task("bill_of_lading_semantic_v2")
    reference = canonical_json(
        {
            "schemaVersion": "2.0.0",
            "documentPatch": {
                "billOfLadingNumber": "ABC",
                "originalBillOfLadingNumber": "ORIGINAL",
            },
        }
    )
    prediction = canonical_json(
        {
            "schemaVersion": "2.0.0",
            "documentPatch": {
                "billOfLadingNumber": "ABC",
                "masterBillOfLadingNumber": "EXTRA",
            },
        }
    )

    metrics, _ = structured_metrics([prediction], [reference], task)

    assert metrics["field_value_accuracy"] == 1 / 3
    assert metrics["field_value_precision"] == 1 / 2
    assert metrics["field_value_recall"] == 1 / 2
    assert metrics["field_value_f1"] == 1 / 2


def test_field_value_metrics_require_the_complete_scalar_value() -> None:
    task = get_training_task("bill_of_lading_semantic_v2")

    metrics, _ = structured_metrics([_target("ABX")], [_target("ABC")], task)

    assert metrics["field_value_accuracy"] == 0.0
    assert metrics["field_value_precision"] == 0.0
    assert metrics["field_value_recall"] == 0.0
    assert metrics["field_value_f1"] == 0.0


def test_field_value_metrics_preserve_partial_credit_when_schema_is_invalid() -> None:
    task = get_training_task("bill_of_lading_semantic_v2")
    prediction = canonical_json(
        {
            "schemaVersion": "WRONG",
            "documentPatch": {"billOfLadingNumber": "ABC"},
        }
    )

    metrics, _ = structured_metrics([prediction], [_target("ABC")], task)

    assert metrics["json_valid"] == 1.0
    assert metrics["schema_valid"] == 0.0
    assert metrics["canonical_exact_match"] == 0.0
    assert metrics["field_value_accuracy"] == 1.0
    assert metrics["field_value_precision"] == 1.0
    assert metrics["field_value_recall"] == 1.0
    assert metrics["field_value_f1"] == 1.0


def test_trainer_metric_callback_publishes_exact_field_value_metrics() -> None:
    class StubTokenizer:
        pad_token_id: int | None = 0
        eos_token_id: int | None = 2

        def batch_decode(self, sequences: Any, **_: Any) -> list[str]:
            number = "ABC" if int(sequences[0][0]) in {1, 2} else "WRONG"
            return [_target(number)]

    compute_metrics = make_compute_metrics(
        StubTokenizer(), get_training_task("bill_of_lading_semantic_v2")
    )

    metrics = compute_metrics(
        SimpleNamespace(
            predictions=np.asarray([[1, 2, 0]], dtype=np.int64),
            label_ids=np.asarray([[2]], dtype=np.int64),
        )
    )

    assert metrics == {
        "json_valid": 1.0,
        "schema_valid": 1.0,
        "canonical_exact_match": 1.0,
        "field_value_accuracy": 1.0,
        "field_value_precision": 1.0,
        "field_value_recall": 1.0,
        "field_value_f1": 1.0,
        "generated_tokens_mean": 2.0,
        "generated_tokens_max": 2.0,
        "generation_eos_reached_fraction": 1.0,
    }


def test_best_model_prediction_is_scored_and_persisted_in_one_pass(tmp_path: Path) -> None:
    class StubTokenizer:
        pad_token_id = 0

        def batch_decode(self, sequences: Any, **_: Any) -> list[str]:
            return [_target("ABC") for _ in sequences]

    expected = {
        "json_valid": 1.0,
        "schema_valid": 1.0,
        "canonical_exact_match": 1.0,
        "field_value_accuracy": 1.0,
        "field_value_precision": 1.0,
        "field_value_recall": 1.0,
        "field_value_f1": 1.0,
    }

    class StubTrainer:
        calls = 0

        def predict(self, dataset: Any, metric_key_prefix: str) -> Any:
            self.calls += 1
            return SimpleNamespace(
                predictions=np.asarray([[1, 2, 0]], dtype=np.int64),
                label_ids=np.asarray([[1, 2, -100]], dtype=np.int64),
                metrics={
                    **{f"{metric_key_prefix}_{name}": value for name, value in expected.items()},
                    f"{metric_key_prefix}_loss": 0.1,
                },
            )

    trainer = StubTrainer()
    metrics, prediction_path = _predict_and_publish(
        trainer=trainer,
        tokenizer=StubTokenizer(),
        task=get_training_task("bill_of_lading_semantic_v2"),
        dataset={"document_id": ["doc_one"]},
        split="validation",
        metric_key_prefix="eval",
        predictions_dir=tmp_path / "predictions",
    )

    assert trainer.calls == 1
    assert metrics["eval_field_value_f1"] == 1.0
    row = json.loads(prediction_path.read_text(encoding="utf-8"))
    assert row["document_id"] == "doc_one"
    assert row["json_valid"] is True
