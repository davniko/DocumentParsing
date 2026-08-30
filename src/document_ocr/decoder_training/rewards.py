"""Deterministic, schema-gated rewards shared with generated evaluation."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any, cast

from document_ocr.decoder_training.completions import ThinkingMode, split_completion
from document_ocr.training.metrics import assess_prediction
from document_ocr.training.tasks import TrainingTask


def completion_text(value: Any) -> str:
    """Extract one TRL completion without accepting ambiguous output shapes."""

    if isinstance(value, str):
        return value
    if isinstance(value, list) and len(value) == 1 and isinstance(value[0], dict):
        message = value[0]
        if message.get("role") == "assistant" and isinstance(message.get("content"), str):
            return cast(str, message["content"])
    raise ValueError("completion must be a string or one assistant message")


def schema_gated_field_f1(
    completions: Sequence[Any],
    reference_target: Sequence[str],
    *,
    task: TrainingTask,
    thinking: ThinkingMode,
    **_: Any,
) -> list[float]:
    """Score only a boundary-valid, schema-valid final answer with exact leaf F1."""

    if len(completions) != len(reference_target):
        raise ValueError("completion/reference counts differ")
    rewards: list[float] = []
    for raw_completion, reference in zip(completions, reference_target, strict=True):
        parts = split_completion(completion_text(raw_completion), thinking=thinking)
        if parts.final_answer is None:
            rewards.append(0.0)
            continue
        try:
            assessment = assess_prediction(parts.final_answer, reference, task)
        except ValueError:
            rewards.append(0.0)
            continue
        if not assessment.schema_valid:
            rewards.append(0.0)
            continue
        true_positive = len(assessment.predicted_field_values & assessment.reference_field_values)
        predicted = len(assessment.predicted_field_values)
        expected = len(assessment.reference_field_values)
        precision = true_positive / predicted if predicted else 0.0
        recall = true_positive / expected if expected else 0.0
        rewards.append(
            2.0 * precision * recall / (precision + recall) if precision + recall else 0.0
        )
    return rewards
