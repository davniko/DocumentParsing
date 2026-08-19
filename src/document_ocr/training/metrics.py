"""Structured generation metrics for sparse KIE decoder targets."""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any, Protocol, cast

from document_ocr.training.tasks import TrainingTask, canonical_json


class DecoderTokenizer(Protocol):
    pad_token_id: int | None
    eos_token_id: int | None

    def batch_decode(self, sequences: Any, **kwargs: Any) -> list[str]: ...


@dataclass(frozen=True, slots=True)
class PredictionAssessment:
    generated_text: str
    reference_text: str
    json_valid: bool
    schema_valid: bool
    canonical_exact_match: bool
    predicted_field_values: frozenset[tuple[str, str]]
    reference_field_values: frozenset[tuple[str, str]]

    def to_dict(self) -> dict[str, Any]:
        return {
            "generated_text": self.generated_text,
            "reference_text": self.reference_text,
            "json_valid": self.json_valid,
            "schema_valid": self.schema_valid,
            "canonical_exact_match": self.canonical_exact_match,
        }


def _field_value_items(value: Any, path: str = "$.documentPatch") -> set[tuple[str, str]]:
    if isinstance(value, dict):
        field_values: set[tuple[str, str]] = set()
        for key in sorted(value):
            field_values.update(_field_value_items(value[key], f"{path}.{key}"))
        return field_values
    if isinstance(value, list):
        field_values = set()
        for index, child in enumerate(value):
            field_values.update(_field_value_items(child, f"{path}[{index}]"))
        return field_values
    return {
        (
            path,
            json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")),
        )
    }


def _document_patch(value: dict[str, Any]) -> dict[str, Any]:
    patch = value.get("documentPatch")
    if not isinstance(patch, dict):
        raise ValueError("canonical decoder target must contain a documentPatch object")
    return patch


def _optional_document_patch(value: dict[str, Any] | None) -> dict[str, Any] | None:
    if value is None:
        return None
    patch = value.get("documentPatch")
    return patch if isinstance(patch, dict) else None


def assess_prediction(
    generated_text: str,
    reference_text: str,
    task: TrainingTask,
) -> PredictionAssessment:
    """Parse, schema-check, and compare one generated sparse target."""

    reference_value = json.loads(reference_text)
    if not isinstance(reference_value, dict):
        raise ValueError("reference decoder target must be a JSON object")
    reference = task.canonicalize(cast(dict[str, Any], reference_value))
    reference_canonical = canonical_json(reference)

    try:
        parsed = json.loads(generated_text)
    except json.JSONDecodeError:
        parsed = None
    json_valid = isinstance(parsed, dict)
    parsed_object = cast(dict[str, Any], parsed) if json_valid else None
    predicted: dict[str, Any] | None = None
    if parsed_object is not None:
        try:
            predicted = task.canonicalize(parsed_object)
        except ValueError:
            predicted = None
    schema_valid = predicted is not None
    generated_canonical = canonical_json(predicted) if predicted is not None else None
    # Exact field-value scoring remains informative when one unrelated field makes an
    # otherwise parseable prediction schema-invalid. Schema validity is reported separately.
    predicted_patch = _optional_document_patch(
        predicted if predicted is not None else parsed_object
    )
    return PredictionAssessment(
        generated_text=generated_text,
        reference_text=reference_canonical,
        json_valid=json_valid,
        schema_valid=schema_valid,
        canonical_exact_match=generated_canonical == reference_canonical,
        predicted_field_values=(
            frozenset(_field_value_items(predicted_patch))
            if predicted_patch is not None
            else frozenset()
        ),
        reference_field_values=frozenset(_field_value_items(_document_patch(reference))),
    )


def structured_metrics(
    generated_texts: Sequence[str],
    reference_texts: Sequence[str],
    task: TrainingTask,
) -> tuple[dict[str, float], list[PredictionAssessment]]:
    """Compute document validity/exactness and micro exact field-value metrics."""

    if len(generated_texts) != len(reference_texts):
        raise ValueError("generated and reference sequence counts differ")
    if not generated_texts:
        raise ValueError("structured metrics require at least one prediction")
    assessments = [
        assess_prediction(generated, reference, task)
        for generated, reference in zip(generated_texts, reference_texts, strict=True)
    ]
    true_positive = sum(
        len(item.predicted_field_values & item.reference_field_values) for item in assessments
    )
    predicted_total = sum(len(item.predicted_field_values) for item in assessments)
    reference_total = sum(len(item.reference_field_values) for item in assessments)
    compared_field_total = sum(
        len(
            {path for path, _ in item.predicted_field_values}
            | {path for path, _ in item.reference_field_values}
        )
        for item in assessments
    )
    accuracy = true_positive / compared_field_total if compared_field_total else 1.0
    precision = true_positive / predicted_total if predicted_total else 0.0
    recall = true_positive / reference_total if reference_total else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    count = len(assessments)
    return (
        {
            "json_valid": sum(item.json_valid for item in assessments) / count,
            "schema_valid": sum(item.schema_valid for item in assessments) / count,
            "canonical_exact_match": (
                sum(item.canonical_exact_match for item in assessments) / count
            ),
            "field_value_accuracy": accuracy,
            "field_value_precision": precision,
            "field_value_recall": recall,
            "field_value_f1": f1,
        },
        assessments,
    )


def make_compute_metrics(tokenizer: DecoderTokenizer, task: TrainingTask) -> Any:
    """Create the Transformers Trainer metric callback without retaining model inputs."""

    def compute_metrics(evaluation_prediction: Any) -> Mapping[str, float]:
        predictions = evaluation_prediction.predictions
        if isinstance(predictions, tuple):
            predictions = predictions[0]
        labels = evaluation_prediction.label_ids
        pad_token_id = tokenizer.pad_token_id
        if pad_token_id is None:
            raise ValueError("tokenizer must define pad_token_id for generated evaluation")
        eos_token_id = tokenizer.eos_token_id
        if eos_token_id is None:
            raise ValueError("tokenizer must define eos_token_id for generated evaluation")
        # NumPy is already a Trainer dependency; copy so the Trainer-owned array is never mutated.
        labels_for_decode = labels.copy()
        labels_for_decode[labels_for_decode == -100] = pad_token_id
        generated_texts = tokenizer.batch_decode(predictions, skip_special_tokens=True)
        reference_texts = tokenizer.batch_decode(labels_for_decode, skip_special_tokens=True)
        metrics, _ = structured_metrics(generated_texts, reference_texts, task)
        generated_token_counts = (predictions != pad_token_id).sum(axis=1)
        eos_reached = (predictions == eos_token_id).any(axis=1)
        metrics.update(
            {
                "generated_tokens_mean": float(generated_token_counts.mean()),
                "generated_tokens_max": float(generated_token_counts.max()),
                "generation_eos_reached_fraction": float(eos_reached.mean()),
            }
        )
        return metrics

    return compute_metrics
