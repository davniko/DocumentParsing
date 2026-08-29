"""Explicit Unsloth/TRL execution path for SFT and schema-rewarded GRPO."""

from __future__ import annotations

import importlib.metadata
import json
import os
import platform
import sys
from pathlib import Path
from typing import Any, cast

from document_ocr.atomic import atomic_publish_json
from document_ocr.decoder_training.config import DecoderTrainingConfig
from document_ocr.decoder_training.data import (
    inspect_decoder_dataset,
    load_task_and_prompt,
    project_for_trl,
)
from document_ocr.decoder_training.evaluation import generated_evaluation
from document_ocr.decoder_training.rewards import schema_gated_field_f1
from document_ocr.hashing import canonical_json_sha256, sha256_file

_PINNED_RUNTIME = {
    "unsloth": "2026.8.22",
    "trl": "0.24.0",
    "transformers": "5.5.0",
}


def _require_pinned_runtime() -> dict[str, str]:
    versions: dict[str, str] = {}
    for package, expected in _PINNED_RUNTIME.items():
        actual = importlib.metadata.version(package)
        if actual != expected:
            raise RuntimeError(f"decoder runtime requires {package}=={expected}, found {actual}")
        versions[package] = actual
    for package in ("torch", "datasets", "peft", "accelerate", "mlflow-skinny"):
        versions[package] = importlib.metadata.version(package)
    return versions


def _run_dir(project_root: Path, config: DecoderTrainingConfig) -> Path:
    root = Path(config.run.output_dir)
    output = root if root.is_absolute() else project_root / root
    resolved_parent = output.resolve()
    resolved_parent.mkdir(parents=True, exist_ok=True)
    if resolved_parent.is_symlink() or not resolved_parent.is_dir():
        raise ValueError(f"decoder output root must be a real directory: {resolved_parent}")
    return resolved_parent / config.run.run_id


def _token(config: DecoderTrainingConfig) -> str | None:
    name = config.model.token_env
    if name is None:
        return None
    token = os.environ.get(name)
    if not token:
        raise ValueError(f"required model token environment variable is empty: {name}")
    return token


def _grpo_initialization_path(project_root: Path, config: DecoderTrainingConfig) -> Path | None:
    method = config.grpo
    if method is None:
        return None
    unresolved = Path(method.initialize_from)
    path = unresolved if unresolved.is_absolute() else project_root / unresolved
    if path.is_symlink():
        raise ValueError(f"GRPO initialization path must not be a symbolic link: {path}")
    try:
        resolved = path.resolve(strict=True)
    except FileNotFoundError as error:
        raise ValueError(f"GRPO initialization path does not exist: {path}") from error
    adapter_config = resolved / "adapter_config.json"
    if not resolved.is_dir() or not adapter_config.is_file() or adapter_config.is_symlink():
        raise ValueError(
            "GRPO initialization must be an adapter directory with adapter_config.json"
        )
    value = json.loads(adapter_config.read_bytes())
    if not isinstance(value, dict):
        raise ValueError("GRPO adapter_config.json must contain an object")
    expected = {
        "r": config.adapter.rank,
        "lora_alpha": config.adapter.alpha,
        "lora_dropout": config.adapter.dropout,
        "bias": config.adapter.bias,
        "use_rslora": config.adapter.use_rslora,
    }
    mismatches = {
        key: {"expected": expected_value, "found": value.get(key)}
        for key, expected_value in expected.items()
        if value.get(key) != expected_value
    }
    if mismatches:
        raise ValueError(f"GRPO adapter configuration differs from YAML: {mismatches}")
    return resolved


def _configure_environment(config: DecoderTrainingConfig) -> None:
    configured = os.environ.get("PYTORCH_ALLOC_CONF")
    if configured is not None and configured != config.runtime.cuda_allocator_conf:
        raise ValueError("PYTORCH_ALLOC_CONF differs from the immutable decoder configuration")
    os.environ["PYTORCH_ALLOC_CONF"] = config.runtime.cuda_allocator_conf
    mlflow = config.logging.mlflow
    os.environ["MLFLOW_TRACKING_URI"] = mlflow.tracking_uri
    os.environ["MLFLOW_EXPERIMENT_NAME"] = mlflow.experiment_name
    os.environ["MLFLOW_RUN_NAME"] = mlflow.run_name
    os.environ["HF_HUB_DISABLE_TELEMETRY"] = "1"
    if mlflow.resume_run_id is not None:
        os.environ["MLFLOW_RUN_ID"] = mlflow.resume_run_id


def _prepare_lora_model(
    *,
    fast_language_model: Any,
    model: Any,
    config: DecoderTrainingConfig,
    initialization_path: Path | None,
) -> Any:
    """Attach LoRA for SFT, or preserve the already-loaded SFT adapter for GRPO."""

    if initialization_path is not None:
        return model
    return fast_language_model.get_peft_model(
        model,
        r=config.adapter.rank,
        target_modules=config.adapter.target_modules,
        lora_alpha=config.adapter.alpha,
        lora_dropout=config.adapter.dropout,
        bias=config.adapter.bias,
        use_gradient_checkpointing="unsloth",
        random_state=config.run.seed,
        use_rslora=config.adapter.use_rslora,
        modules_to_save=config.adapter.modules_to_save or None,
        loftq_config=None,
    )


def _trainer_common(config: DecoderTrainingConfig, run_dir: Path) -> dict[str, Any]:
    optimization = config.optimization
    evaluation = config.evaluation
    checkpoint = config.checkpoint
    dataloader = config.dataloader
    return {
        "output_dir": str(run_dir / "checkpoints"),
        "do_train": True,
        "do_eval": evaluation.strategy != "no",
        "eval_strategy": evaluation.strategy,
        "eval_steps": evaluation.steps,
        "per_device_train_batch_size": optimization.per_device_train_batch_size,
        "per_device_eval_batch_size": optimization.per_device_eval_batch_size,
        "gradient_accumulation_steps": optimization.gradient_accumulation_steps,
        "learning_rate": optimization.learning_rate,
        "num_train_epochs": optimization.num_train_epochs,
        "max_steps": optimization.max_steps,
        "optim": optimization.optimizer,
        "adam_beta1": optimization.adam_beta1,
        "adam_beta2": optimization.adam_beta2,
        "adam_epsilon": optimization.adam_epsilon,
        "weight_decay": optimization.weight_decay,
        "max_grad_norm": optimization.max_grad_norm,
        "lr_scheduler_type": optimization.lr_scheduler_type,
        # Transformers 5 accepts fractional warmup through warmup_steps.
        "warmup_steps": optimization.warmup_ratio,
        "gradient_checkpointing": optimization.gradient_checkpointing,
        "bf16": optimization.bf16,
        "fp16": False,
        "tf32": optimization.tf32,
        "full_determinism": optimization.full_determinism,
        "seed": config.run.seed,
        "data_seed": config.run.seed,
        "torch_compile": config.runtime.torch_compile,
        "torch_compile_backend": config.runtime.torch_compile_backend,
        "torch_compile_mode": config.runtime.torch_compile_mode,
        "dataloader_num_workers": dataloader.num_workers,
        "dataloader_pin_memory": dataloader.pin_memory,
        "dataloader_persistent_workers": dataloader.persistent_workers,
        "dataloader_prefetch_factor": dataloader.prefetch_factor,
        "dataloader_drop_last": dataloader.drop_last,
        "save_strategy": checkpoint.strategy,
        "save_steps": checkpoint.steps,
        "save_total_limit": checkpoint.total_limit,
        "save_only_model": checkpoint.save_only_model,
        "load_best_model_at_end": checkpoint.load_best_model_at_end,
        "metric_for_best_model": checkpoint.metric_for_best_model,
        "greater_is_better": checkpoint.greater_is_better,
        "logging_strategy": config.logging.strategy,
        "logging_steps": config.logging.steps,
        "logging_first_step": config.logging.first_step,
        "disable_tqdm": config.logging.disable_tqdm,
        "report_to": ["mlflow"],
        "run_name": config.logging.mlflow.run_name,
        "eval_on_start": False,
        "include_num_input_tokens_seen": "non_padding",
    }


def _generation_trainer_class(base_class: type[Any]) -> type[Any]:
    class GeneratedEvaluationTrainer(base_class):  # type: ignore[misc]
        _kie_task: Any
        _kie_tokenizer: Any
        _kie_eval_dataset: Any
        _kie_config: DecoderTrainingConfig
        _kie_run_dir: Path

        def evaluate(self, *args: Any, **kwargs: Any) -> dict[str, float]:
            metrics = super().evaluate(*args, **kwargs)
            step = int(self.state.global_step)
            prediction_path = (
                self._kie_run_dir / "predictions" / f"validation-step-{step:08d}.jsonl"
                if self._kie_config.evaluation.write_predictions
                else None
            )
            generated = generated_evaluation(
                model=self.model,
                tokenizer=self._kie_tokenizer,
                dataset=self._kie_eval_dataset,
                task=self._kie_task,
                batch_size=self._kie_config.optimization.per_device_eval_batch_size,
                max_new_tokens=self._kie_config.evaluation.generation_max_new_tokens,
                num_beams=self._kie_config.evaluation.generation_num_beams,
                output_path=prediction_path,
            )
            prefixed = {f"eval_{name}": value for name, value in generated.items()}
            self.log(prefixed)
            metrics.update(prefixed)
            return cast(dict[str, float], metrics)

    return GeneratedEvaluationTrainer


def run_training(
    *, project_root: Path, config_path: Path, config: DecoderTrainingConfig
) -> dict[str, Any]:
    """Load no GPU state until all local data and immutable contracts pass inspection."""

    _configure_environment(config)
    task, prompt = load_task_and_prompt(project_root, config)
    records, inspection = inspect_decoder_dataset(
        project_root=project_root, config=config, task=task, prompt=prompt
    )
    versions = _require_pinned_runtime()
    initialization_path = _grpo_initialization_path(project_root, config)
    try:
        import torch
    except ImportError as error:
        raise RuntimeError("training requires the isolated decoder environment") from error
    if config.runtime.require_cuda and not torch.cuda.is_available():
        raise RuntimeError("decoder training requires CUDA, but CUDA is unavailable")
    try:
        # Unsloth must patch Transformers and TRL before either trainer is imported.
        # isort: off
        from unsloth import FastLanguageModel  # type: ignore[import-not-found]

        from transformers import EarlyStoppingCallback
        from trl import (  # type: ignore[import-not-found]
            GRPOConfig,
            GRPOTrainer,
            SFTConfig,
            SFTTrainer,
        )
        # isort: on
    except ImportError as error:
        raise RuntimeError("training requires the isolated decoder environment") from error
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=(
            str(initialization_path)
            if initialization_path is not None
            else config.model.name_or_path
        ),
        revision=(None if initialization_path is not None else config.model.revision),
        max_seq_length=config.sequence.max_sequence_length,
        dtype=torch.bfloat16,
        load_in_4bit=False,
        load_in_8bit=False,
        load_in_16bit=True,
        full_finetuning=False,
        token=_token(config),
        trust_remote_code=False,
        use_gradient_checkpointing="unsloth",
        fast_inference=False,
        text_only=True,
        local_files_only=config.model.local_files_only,
        random_state=config.run.seed,
        max_lora_rank=config.adapter.rank,
    )
    datasets, token_inspection = project_for_trl(records, tokenizer=tokenizer, config=config)
    run_dir = _run_dir(project_root, config)
    contract = {
        "schemaVersion": 1,
        "configPath": str(config_path),
        "configFileSha256": sha256_file(config_path),
        "resolvedConfigSha256": canonical_json_sha256(config.model_dump(mode="json")),
        "versions": versions,
        "python": sys.version,
        "platform": platform.platform(),
        "dataset": inspection.to_dict(),
        "tokens": token_inspection.to_dict(),
    }
    atomic_publish_json(run_dir / "run-contract.json", contract)
    initial_evaluation: dict[str, float] = {}
    if config.evaluation.on_start:
        initial_evaluation = generated_evaluation(
            model=model,
            tokenizer=tokenizer,
            dataset=datasets["validation"],
            task=task,
            batch_size=config.optimization.per_device_eval_batch_size,
            max_new_tokens=config.evaluation.generation_max_new_tokens,
            num_beams=config.evaluation.generation_num_beams,
            output_path=(
                run_dir
                / "predictions"
                / (
                    "base-validation.jsonl"
                    if initialization_path is None
                    else "initial-adapter-validation.jsonl"
                )
            ),
        )
    model = _prepare_lora_model(
        fast_language_model=FastLanguageModel,
        model=model,
        config=config,
        initialization_path=initialization_path,
    )
    trainable = sorted(name for name, value in model.named_parameters() if value.requires_grad)
    if not trainable:
        raise RuntimeError("LoRA configuration matched no trainable parameters")
    atomic_publish_json(
        run_dir / "trainable-parameters.json",
        {
            "names": trainable,
            "trainable": sum(value.numel() for value in model.parameters() if value.requires_grad),
            "total": sum(value.numel() for value in model.parameters()),
        },
    )
    common = _trainer_common(config, run_dir)
    callbacks: list[Any] = []
    if config.evaluation.early_stopping_patience is not None:
        callbacks.append(
            EarlyStoppingCallback(
                early_stopping_patience=config.evaluation.early_stopping_patience,
                early_stopping_threshold=config.evaluation.early_stopping_threshold,
            )
        )
    if config.method == "sft":
        arguments = SFTConfig(
            **common,
            max_length=config.sequence.max_sequence_length,
            completion_only_loss=True,
            packing=config.sequence.packing,
            eval_packing=False,
            dataset_num_proc=config.sequence.dataset_num_proc,
            eos_token=tokenizer.eos_token,
        )
        trainer_type = _generation_trainer_class(SFTTrainer)
        trainer = trainer_type(
            model=model,
            processing_class=tokenizer,
            args=arguments,
            train_dataset=datasets["train"],
            eval_dataset=datasets["validation"],
            callbacks=callbacks,
        )
    else:
        method = config.grpo
        assert method is not None
        arguments = GRPOConfig(
            **common,
            max_prompt_length=config.sequence.max_prompt_length,
            max_completion_length=method.rollout.max_completion_length,
            num_generations=method.rollout.num_generations,
            temperature=method.rollout.temperature,
            top_p=method.rollout.top_p,
            use_vllm=False,
            loss_type=method.policy_optimization.loss_type,
            scale_rewards=method.policy_optimization.scale_rewards,
            beta=method.policy_optimization.beta,
            num_iterations=method.policy_optimization.num_iterations,
            epsilon=method.policy_optimization.epsilon,
            mask_truncated_completions=True,
        )

        def schema_gated_field_f1_v1(
            completions: Any, reference_target: Any, **kwargs: Any
        ) -> list[float]:
            return schema_gated_field_f1(
                completions,
                reference_target,
                task=task,
                **kwargs,
            )

        trainer_type = _generation_trainer_class(GRPOTrainer)
        trainer = trainer_type(
            model=model,
            processing_class=tokenizer,
            reward_funcs=schema_gated_field_f1_v1,
            args=arguments,
            train_dataset=datasets["train"],
            eval_dataset=datasets["validation"],
            callbacks=callbacks,
        )
    trainer._kie_task = task
    trainer._kie_tokenizer = tokenizer
    trainer._kie_eval_dataset = datasets["validation"]
    trainer._kie_config = config
    trainer._kie_run_dir = run_dir
    if initial_evaluation:
        trainer.log({f"initial_{name}": value for name, value in initial_evaluation.items()})
    result = trainer.train(resume_from_checkpoint=config.checkpoint.resume_from_checkpoint)
    trainer.save_model(str(run_dir / "adapter"))
    tokenizer.save_pretrained(str(run_dir / "adapter"))
    final_metrics = (
        trainer.evaluate(metric_key_prefix="eval") if config.evaluation.run_final else {}
    )
    summary = {
        "runId": config.run.run_id,
        "method": config.method,
        "initialEvaluation": initial_evaluation,
        "trainMetrics": dict(result.metrics),
        "finalEvaluation": final_metrics,
        "runContractSha256": canonical_json_sha256(contract),
    }
    atomic_publish_json(run_dir / "summary.json", summary)
    return summary
