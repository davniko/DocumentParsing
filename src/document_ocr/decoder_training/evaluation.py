"""Unconstrained greedy generation evaluation for decoder-only KIE models."""

from __future__ import annotations

import json
import time
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from document_ocr.atomic import atomic_publish_bytes, atomic_publish_json
from document_ocr.decoder_training.completions import ThinkingMode, split_completion
from document_ocr.training.metrics import structured_metrics
from document_ocr.training.tasks import TrainingTask


def _model_device(model: Any) -> Any:
    try:
        return next(model.parameters()).device
    except StopIteration as error:
        raise ValueError("model has no parameters") from error


def generated_evaluation(
    *,
    model: Any,
    tokenizer: Any,
    dataset: Any,
    task: TrainingTask,
    batch_size: int,
    max_new_tokens: int,
    num_beams: int,
    thinking: ThinkingMode,
    output_path: Path | None,
) -> dict[str, float]:
    """Generate exact completions without a JSON grammar and publish a full audit trail."""

    try:
        import torch
    except ImportError as error:
        raise RuntimeError("generated evaluation requires the decoder environment") from error
    if batch_size <= 0 or max_new_tokens <= 0 or num_beams <= 0:
        raise ValueError("generation sizes must be positive")
    if len(dataset) == 0:
        raise ValueError("generated evaluation requires at least one record")
    device = _model_device(model)
    final_answers: list[str] = []
    reference_texts: list[str] = []
    published_rows: list[dict[str, Any]] = []
    generated_token_counts: list[int] = []
    eos_reached: list[bool] = []
    reasoning_boundary_valid: list[bool] = []
    reasoning_token_counts: list[int] = []
    started = time.perf_counter()
    was_training = bool(model.training)
    model.eval()
    try:
        for start in range(0, len(dataset), batch_size):
            batch = dataset[start : start + batch_size]
            prompts = list(batch["prompt"])
            encoded = tokenizer(
                prompts,
                add_special_tokens=False,
                padding=True,
                truncation=False,
                return_tensors="pt",
            )
            if not isinstance(encoded, Mapping) or "input_ids" not in encoded:
                raise ValueError("tokenizer did not return batched input IDs")
            model_inputs = {name: value.to(device) for name, value in encoded.items()}
            prompt_width = int(model_inputs["input_ids"].shape[1])
            with torch.inference_mode():
                generated = model.generate(
                    **model_inputs,
                    do_sample=False,
                    num_beams=num_beams,
                    max_new_tokens=max_new_tokens,
                    use_cache=True,
                    pad_token_id=tokenizer.pad_token_id,
                    eos_token_id=tokenizer.eos_token_id,
                )
            continuation_ids = generated[:, prompt_width:]
            texts = tokenizer.batch_decode(continuation_ids, skip_special_tokens=True)
            counts = [int(row.ne(tokenizer.pad_token_id).sum().item()) for row in continuation_ids]
            reached = [
                bool(row.eq(tokenizer.eos_token_id).any().item()) for row in continuation_ids
            ]
            for offset, (text, count, ended) in enumerate(zip(texts, counts, reached, strict=True)):
                index = start + offset
                reference = str(batch["reference_target"][offset])
                document_id = str(batch["document_id"][offset])
                parts = split_completion(text, thinking=thinking)
                final_answers.append(parts.final_answer or "")
                reference_texts.append(reference)
                generated_token_counts.append(count)
                eos_reached.append(ended)
                reasoning_boundary_valid.append(parts.boundary_valid)
                reasoning_ids = (
                    tokenizer(
                        parts.reasoning_trace,
                        add_special_tokens=False,
                        truncation=False,
                        padding=False,
                    )["input_ids"]
                    if parts.reasoning_trace is not None
                    else []
                )
                reasoning_token_counts.append(len(reasoning_ids))
                published_rows.append(
                    {
                        "documentId": document_id,
                        "generatedText": parts.raw_text,
                        "reasoningTrace": parts.reasoning_trace,
                        "finalAnswer": parts.final_answer,
                        "reasoningBoundaryValid": parts.boundary_valid,
                        "reasoningBoundaryError": parts.boundary_error,
                        "referenceTarget": reference,
                        "generatedTokens": count,
                        "eosReached": ended,
                        "datasetIndex": index,
                    }
                )
    finally:
        if was_training:
            model.train()
    elapsed = time.perf_counter() - started
    metrics, assessments = structured_metrics(final_answers, reference_texts, task)
    metrics.update(
        {
            "generated_tokens_mean": sum(generated_token_counts) / len(generated_token_counts),
            "generated_tokens_max": float(max(generated_token_counts)),
            "generation_eos_reached_fraction": sum(eos_reached) / len(eos_reached),
            "generation_runtime_seconds": elapsed,
            "generation_samples_per_second": len(dataset) / elapsed,
        }
    )
    if thinking == "enabled":
        metrics.update(
            {
                "reasoning_boundary_valid_fraction": (
                    sum(reasoning_boundary_valid) / len(reasoning_boundary_valid)
                ),
                "reasoning_tokens_mean": (
                    sum(reasoning_token_counts) / len(reasoning_token_counts)
                ),
                "reasoning_tokens_max": float(max(reasoning_token_counts)),
            }
        )
    if output_path is not None:
        payload = b"".join(
            json.dumps(
                {**row, "assessment": assessment.to_dict()},
                ensure_ascii=False,
                sort_keys=True,
                separators=(",", ":"),
            ).encode("utf-8")
            + b"\n"
            for row, assessment in zip(published_rows, assessments, strict=True)
        )
        atomic_publish_bytes(output_path, payload)
        atomic_publish_json(output_path.with_suffix(".metrics.json"), metrics)
    return metrics
