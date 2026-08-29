#!/usr/bin/env python3
"""Evaluate one retained KIE checkpoint without resuming or mutating training."""

from __future__ import annotations

import argparse
import json
import re
import shutil
import tempfile
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from document_ocr.atomic import atomic_write_json
from document_ocr.hashing import sha256_file
from document_ocr.training.collator import MetadataStrippingCollator
from document_ocr.training.config import TrainingConfig, load_training_config
from document_ocr.training.data import prepare_datasets
from document_ocr.training.metrics import make_compute_metrics
from document_ocr.training.prediction import SamplerAwarePredictionMixin
from document_ocr.training.prompting import load_prompt
from document_ocr.training.runtime import (
    _configure_cuda_allocator,
    _predict_and_publish,
    build_training_arguments,
    load_lora_model,
    load_tokenizer,
    validate_training_hardware,
)
from document_ocr.training.tasks import load_training_task

CHECKPOINT_PATTERN = re.compile(r"checkpoint-(\d+)")


def _real_directory(path: Path, description: str) -> Path:
    if path.is_symlink():
        raise ValueError(f"{description} must not be a symbolic link: {path}")
    resolved = path.resolve(strict=True)
    if not resolved.is_dir():
        raise ValueError(f"{description} is not a directory: {resolved}")
    return resolved


def _resolve_checkpoint(run_dir: Path, checkpoint: Path) -> tuple[Path, int]:
    resolved_run = _real_directory(run_dir, "run directory")
    resolved_checkpoint = _real_directory(checkpoint, "checkpoint")
    expected_parent = (resolved_run / "checkpoints").resolve(strict=True)
    if resolved_checkpoint.parent != expected_parent:
        raise ValueError("checkpoint must be a direct child of the run checkpoint directory")
    match = CHECKPOINT_PATTERN.fullmatch(resolved_checkpoint.name)
    if match is None:
        raise ValueError(f"unexpected checkpoint directory name: {resolved_checkpoint.name}")
    step = int(match.group(1))
    state = json.loads((resolved_checkpoint / "trainer_state.json").read_text(encoding="utf-8"))
    if int(state["global_step"]) != step:
        raise ValueError("checkpoint directory step differs from trainer_state.global_step")
    return resolved_checkpoint, step


def _expected_adapter_config(config: TrainingConfig) -> dict[str, Any]:
    return {
        "base_model_name_or_path": config.model.name_or_path,
        "bias": config.peft.bias,
        "ensure_weight_tying": config.peft.ensure_weight_tying,
        "lora_alpha": config.peft.alpha,
        "lora_dropout": config.peft.dropout,
        "modules_to_save": config.peft.modules_to_save or None,
        "r": config.peft.rank,
        "target_modules": config.peft.target_modules_regex,
        "task_type": "SEQ_2_SEQ_LM",
        "use_rslora": config.peft.use_rslora,
    }


def _validate_adapter_contract(adapter_dir: Path, config: TrainingConfig) -> dict[str, Any]:
    if adapter_dir.is_symlink() or not adapter_dir.is_dir():
        raise ValueError(f"checkpoint adapter directory is invalid: {adapter_dir}")
    config_path = adapter_dir / "adapter_config.json"
    weights_path = adapter_dir / "adapter_model.safetensors"
    if config_path.is_symlink() or weights_path.is_symlink():
        raise ValueError("checkpoint adapter artifacts must not be symbolic links")
    recorded = json.loads(config_path.read_text(encoding="utf-8"))
    expected = _expected_adapter_config(config)
    mismatches = {
        key: {"expected": expected_value, "recorded": recorded.get(key)}
        for key, expected_value in expected.items()
        if recorded.get(key) != expected_value
    }
    if mismatches:
        raise ValueError(f"checkpoint adapter contract differs from run config: {mismatches}")
    if not weights_path.is_file():
        raise ValueError(f"checkpoint adapter weights are absent: {weights_path}")
    return {
        "adapter_config_path": str(config_path),
        "adapter_config_sha256": sha256_file(config_path),
        "adapter_weights_path": str(weights_path),
        "adapter_weights_sha256": sha256_file(weights_path),
        "adapter_weights_bytes": weights_path.stat().st_size,
    }


def _load_checkpoint_weights(model: Any, adapter_dir: Path, adapter_name: str) -> None:
    from peft.utils.save_and_load import load_peft_weights, set_peft_model_state_dict

    state = load_peft_weights(str(adapter_dir), device="cpu")
    result = set_peft_model_state_dict(model, state, adapter_name=adapter_name)
    if result.unexpected_keys:
        raise RuntimeError(
            "checkpoint adapter produced unexpected state keys: "
            + ", ".join(result.unexpected_keys[:20])
        )
    model.requires_grad_(False)
    model.eval()


def evaluate_checkpoint(
    *,
    project_root: Path,
    run_dir: Path,
    checkpoint: Path,
    output_dir: Path,
) -> dict[str, Any]:
    """Publish predictions from one immutable retained checkpoint."""

    resolved_project = _real_directory(project_root, "project root")
    resolved_run = _real_directory(run_dir, "run directory")
    resolved_checkpoint, checkpoint_step = _resolve_checkpoint(resolved_run, checkpoint)
    resolved_output = output_dir.resolve()
    if resolved_output == resolved_run or resolved_run in resolved_output.parents:
        raise ValueError("checkpoint evaluation output must be outside the training run")
    if resolved_output.exists():
        raise ValueError(f"checkpoint evaluation output already exists: {resolved_output}")
    resolved_output.parent.mkdir(parents=True, exist_ok=True)

    config_path = resolved_run / "config.yaml"
    config = load_training_config(config_path)
    task = load_training_task(resolved_project, config)
    prompt = load_prompt(resolved_project, config.prompt, task)
    adapter_dir = resolved_checkpoint / config.peft.adapter_name
    adapter_provenance = _validate_adapter_contract(adapter_dir, config)

    _configure_cuda_allocator(config)
    hardware = validate_training_hardware(config)
    tokenizer = load_tokenizer(config)
    prepared = prepare_datasets(
        project_root=resolved_project,
        config=config,
        prompt=prompt,
        task=task,
        tokenizer=tokenizer,
    )
    model, model_facts = load_lora_model(config)
    _load_checkpoint_weights(model, adapter_dir, config.peft.adapter_name)

    from transformers import DataCollatorForSeq2Seq, Seq2SeqTrainer, set_seed

    set_seed(config.optimization.seed, deterministic=config.optimization.full_determinism)
    temporary = Path(
        tempfile.mkdtemp(prefix=f".{resolved_output.name}.", dir=resolved_output.parent)
    )
    try:
        arguments = build_training_arguments(config, temporary / "trainer-work")
        collator = MetadataStrippingCollator(
            DataCollatorForSeq2Seq(
                tokenizer=tokenizer,
                model=model,
                padding="longest",
                label_pad_token_id=-100,
                pad_to_multiple_of=config.dataset.preprocessing.pad_to_multiple_of,
                return_tensors="pt",
            )
        )

        class SamplerAwareSeq2SeqTrainer(SamplerAwarePredictionMixin, Seq2SeqTrainer):
            pass

        validation = prepared.datasets[config.evaluation.split]
        trainer = SamplerAwareSeq2SeqTrainer(
            model=model,
            args=arguments,
            data_collator=collator,
            eval_dataset=validation,
            processing_class=tokenizer,
            compute_metrics=make_compute_metrics(tokenizer, task),
        )
        started = time.perf_counter()
        metrics, prediction_path = _predict_and_publish(
            trainer=trainer,
            tokenizer=tokenizer,
            task=task,
            dataset=validation,
            split=config.evaluation.split,
            metric_key_prefix="eval",
            predictions_dir=temporary / "predictions",
        )
        elapsed = time.perf_counter() - started
        checkpoint_state_path = resolved_checkpoint / "trainer_state.json"
        checkpoint_state = json.loads(checkpoint_state_path.read_text(encoding="utf-8"))
        manifest = {
            "schema_version": 1,
            "status": "complete",
            "created_at": datetime.now(UTC).isoformat(),
            "source_run_id": config.run.run_id,
            "source_run_dir": str(resolved_run),
            "source_config_path": str(config_path),
            "source_config_sha256": sha256_file(config_path),
            "checkpoint_path": str(resolved_checkpoint),
            "checkpoint_step": checkpoint_step,
            "checkpoint_epoch": checkpoint_state["epoch"],
            "checkpoint_is_recorded_best": Path(str(checkpoint_state["best_model_checkpoint"])).name
            == resolved_checkpoint.name,
            "checkpoint_best_metric": checkpoint_state["best_metric"],
            "checkpoint_state_sha256": sha256_file(checkpoint_state_path),
            "adapter": adapter_provenance,
            "dataset_cache_identity": prepared.cache_identity,
            "validation_records": len(validation),
            "prediction_path": str(prediction_path.relative_to(temporary)),
            "prediction_sha256": sha256_file(prediction_path),
            "evaluation_seconds": elapsed,
            "metrics": metrics,
            "hardware": hardware,
            "model": model_facts,
        }
        atomic_write_json(temporary / "metrics.json", metrics)
        atomic_write_json(temporary / "manifest.json", manifest)
        temporary.rename(resolved_output)
    except BaseException:
        shutil.rmtree(temporary, ignore_errors=True)
        raise
    return manifest


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, required=True)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    arguments = _parse_args()
    manifest = evaluate_checkpoint(
        project_root=arguments.project_root,
        run_dir=arguments.run_dir,
        checkpoint=arguments.checkpoint,
        output_dir=arguments.output_dir,
    )
    print(json.dumps(manifest, ensure_ascii=False, sort_keys=True, separators=(",", ":")))


if __name__ == "__main__":
    main()
