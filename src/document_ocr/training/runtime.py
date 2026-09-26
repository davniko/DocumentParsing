"""Lazy-loaded tokenizer, PEFT model, Trainer, and run-publication runtime."""

from __future__ import annotations

import importlib
import importlib.metadata
import json
import math
import os
import platform
import re
import sys
from collections.abc import Iterable, Mapping, Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, cast

from document_ocr.atomic import (
    atomic_publish_bytes,
    atomic_publish_json,
    atomic_write_bytes,
    read_regular_file_bytes,
)
from document_ocr.hashing import canonical_json_bytes, sha256_bytes, sha256_file
from document_ocr.training.collator import MetadataStrippingCollator
from document_ocr.training.config import TrainingConfig, parse_training_config, resolve_config_path
from document_ocr.training.data import PreparedDatasets, prepare_datasets
from document_ocr.training.metrics import make_compute_metrics, structured_metrics
from document_ocr.training.prediction import SamplerAwarePredictionMixin
from document_ocr.training.prompting import PromptTemplate
from document_ocr.training.schedule_free import ScheduleFreeTrainerMixin
from document_ocr.training.tasks import TrainingTask

_TRAINING_PACKAGES = (
    "accelerate",
    "datasets",
    "mlflow-skinny",
    "nvidia-ml-py",
    "peft",
    "schedulefree",
    "torch",
    "transformers",
)
_PROHIBITED_ADAPTER_PATHS = ("vision_tower", "multi_modal_projector")


def _utc_now() -> str:
    return datetime.now(UTC).isoformat()


def _access_token(config: TrainingConfig) -> str | None:
    variable = config.model.token_env
    if variable is None:
        return None
    token = os.environ.get(variable)
    if token is None or not token.strip():
        raise RuntimeError(
            f"required model-access token is absent from environment variable {variable}"
        )
    return token


def _configure_cuda_allocator(config: TrainingConfig) -> None:
    """Bind PyTorch's process-global allocator before PyTorch is imported."""

    expected = config.runtime.cuda_allocator_conf
    configured = os.environ.get("PYTORCH_ALLOC_CONF")
    if configured is None:
        if "torch" in sys.modules:
            raise RuntimeError(
                "PyTorch was imported before runtime.cuda_allocator_conf was applied"
            )
        os.environ["PYTORCH_ALLOC_CONF"] = expected
        return
    if configured != expected:
        raise RuntimeError(
            "PYTORCH_ALLOC_CONF differs from runtime.cuda_allocator_conf: "
            f"environment={configured!r}, config={expected!r}"
        )


def load_tokenizer(config: TrainingConfig) -> Any:
    """Load only the exact configured tokenizer; this does not load model weights or use CUDA."""

    try:
        from transformers import AutoTokenizer
    except ImportError as error:
        raise RuntimeError("tokenizer loading requires the 'train' dependency group") from error

    tokenizer = AutoTokenizer.from_pretrained(
        config.model.tokenizer_name_or_path,
        revision=config.model.tokenizer_revision,
        token=_access_token(config),
        local_files_only=config.model.local_files_only,
        trust_remote_code=config.model.trust_remote_code,
        use_fast=config.model.tokenizer_use_fast,
    )
    if tokenizer.pad_token_id is None:
        raise RuntimeError(
            "the configured tokenizer has no pad token; implicit fallback is forbidden"
        )
    if tokenizer.eos_token_id is None:
        raise RuntimeError("the configured tokenizer has no EOS token")
    return tokenizer


def validate_training_hardware(config: TrainingConfig) -> dict[str, Any]:
    """Fail before model loading when requested CUDA/precision features are unsupported."""

    try:
        import torch
    except ImportError as error:
        raise RuntimeError("training requires the 'train' dependency group") from error

    allocator_conf = os.environ.get("PYTORCH_ALLOC_CONF")
    if allocator_conf != config.runtime.cuda_allocator_conf:
        raise RuntimeError("configured CUDA allocator was not applied before hardware validation")
    cuda_available = torch.cuda.is_available()
    try:
        world_size = int(os.environ.get("WORLD_SIZE", "1"))
    except ValueError as error:
        raise RuntimeError("WORLD_SIZE must be an integer") from error
    if world_size != config.runtime.distributed_processes:
        raise RuntimeError(
            "runtime process-count mismatch: expected "
            f"{config.runtime.distributed_processes}, found {world_size}"
        )
    if config.runtime.require_cuda and not cuda_available:
        raise RuntimeError("runtime.require_cuda=true but PyTorch cannot access CUDA")
    if config.runtime.bf16:
        if not cuda_available:
            raise RuntimeError("BF16 training requires CUDA in this pipeline")
        if not torch.cuda.is_bf16_supported():
            raise RuntimeError("the active CUDA device does not support BF16 training")
    if config.runtime.fp16 and not cuda_available:
        raise RuntimeError("FP16 training requires CUDA in this pipeline")
    if config.runtime.tf32:
        if not cuda_available:
            raise RuntimeError("TF32 requires CUDA")
        for index in range(torch.cuda.device_count()):
            major, _ = torch.cuda.get_device_capability(index)
            if major < 8:
                raise RuntimeError(f"CUDA device {index} does not support TF32")

    devices = []
    for index in range(torch.cuda.device_count() if cuda_available else 0):
        properties = torch.cuda.get_device_properties(index)
        devices.append(
            {
                "index": index,
                "name": properties.name,
                "compute_capability": list(torch.cuda.get_device_capability(index)),
                "total_memory_bytes": properties.total_memory,
            }
        )
    return {
        "torch_version": torch.__version__,
        "world_size": world_size,
        "cuda_available": cuda_available,
        "cuda_runtime_version": torch.version.cuda,
        "cuda_allocator_backend": torch.cuda.memory.get_allocator_backend(),
        "cuda_allocator_conf": allocator_conf,
        "cudnn_version": (
            torch.backends.cudnn.version() if cuda_available else None  # type: ignore[no-untyped-call]
        ),
        "bf16_supported": torch.cuda.is_bf16_supported() if cuda_available else False,
        "devices": devices,
    }


def _adapter_module_names(model: Any, adapter_name: str) -> set[str]:
    names: set[str] = set()
    prefix = "base_model.model."
    for name, module in model.named_modules():
        lora_a = getattr(module, "lora_A", None)
        lora_embedding_a = getattr(module, "lora_embedding_A", None)
        if (lora_a is not None and adapter_name in lora_a) or (
            lora_embedding_a is not None and adapter_name in lora_embedding_a
        ):
            names.add(name.removeprefix(prefix))
    return names


def load_lora_model(config: TrainingConfig) -> tuple[Any, dict[str, Any]]:
    """Load the exact seq2seq checkpoint, insert LoRA, and prove its module boundary."""

    try:
        import torch
        from peft import EvaConfig, TaskType, get_peft_model
        from peft import LoraConfig as PeftLoraConfig
        from transformers import AutoModelForSeq2SeqLM
    except ImportError as error:
        raise RuntimeError("model construction requires the 'train' dependency group") from error

    dtype = {
        "bfloat16": torch.bfloat16,
        "float16": torch.float16,
        "float32": torch.float32,
    }[config.model.dtype]
    model = AutoModelForSeq2SeqLM.from_pretrained(
        config.model.name_or_path,
        revision=config.model.revision,
        token=_access_token(config),
        local_files_only=config.model.local_files_only,
        trust_remote_code=config.model.trust_remote_code,
        dtype=dtype,
        attn_implementation=config.model.attention_implementation,
        low_cpu_mem_usage=config.model.low_cpu_mem_usage,
        use_safetensors=True,
    )
    is_encoder_decoder = bool(getattr(model.config, "is_encoder_decoder", False))
    if not is_encoder_decoder:
        raise RuntimeError("configured checkpoint is not an encoder-decoder model")
    model_type = getattr(model.config, "model_type", None)
    if model_type != config.model.model_type:
        raise RuntimeError(
            "model type mismatch: expected "
            f"{config.model.model_type!r}, found {model_type!r}"
        )
    model.config.use_cache = config.model.use_cache
    encoder_config = model.config.get_text_config(decoder=False)
    encoder_config.use_cache = config.model.use_cache
    decoder_config = model.config.get_text_config(decoder=True)
    decoder_config.use_cache = config.model.use_cache
    # Some composite encoder-decoder models copy nested configs into their runtime
    # modules during construction. Configure every distinct module-owned cache default,
    # not only the serializable top-level config.
    runtime_cache_configs = {
        id(module_config): module_config
        for module in model.modules()
        if (module_config := getattr(module, "config", None)) is not None
        and hasattr(module_config, "use_cache")
    }
    for module_config in runtime_cache_configs.values():
        module_config.use_cache = config.model.use_cache
    model.generation_config.do_sample = config.evaluation.generation_do_sample
    # Training cache is disabled for gradient checkpointing, while autoregressive
    # evaluation must use its KV cache. Keep those two execution contracts explicit.
    model.generation_config.use_cache = True
    model.generation_config.max_length = config.evaluation.generation_max_length
    model.generation_config.num_beams = config.evaluation.generation_num_beams
    model.generation_config.repetition_penalty = (
        config.evaluation.generation_repetition_penalty
    )
    model.generation_config.no_repeat_ngram_size = (
        config.evaluation.generation_no_repeat_ngram_size
    )
    model.generation_config.length_penalty = config.evaluation.generation_length_penalty
    model.generation_config.early_stopping = config.evaluation.generation_early_stopping
    # The checkpoint ships sampling defaults, but this pipeline deliberately uses
    # deterministic greedy decoding. Clearing them avoids misleading ignored-flag warnings.
    model.generation_config.top_p = None
    model.generation_config.top_k = None

    target_pattern = re.compile(config.peft.target_modules_regex)
    expected_modules = {
        name for name, _ in model.named_modules() if target_pattern.fullmatch(name)
    }
    if not expected_modules:
        raise RuntimeError("LoRA target regex matched no modules in the configured checkpoint")
    prohibited = {
        name
        for name in expected_modules
        if any(part in name for part in _PROHIBITED_ADAPTER_PATHS)
    }
    if prohibited:
        raise RuntimeError(
            f"LoRA target regex reaches prohibited multimodal modules: {sorted(prohibited)}"
        )

    peft_config = PeftLoraConfig(
        task_type=TaskType.SEQ_2_SEQ_LM,
        r=config.peft.rank,
        lora_alpha=config.peft.alpha,
        lora_dropout=config.peft.dropout,
        bias=config.peft.bias,
        use_rslora=config.peft.use_rslora,
        init_lora_weights=config.peft.init_lora_weights,
        eva_config=(
            EvaConfig(
                rho=config.peft.eva.rho, tau=config.peft.eva.tau,
                whiten=config.peft.eva.whiten, use_label_mask=True,
                adjust_scaling_factors=False,
            )
            if config.peft.eva is not None else None
        ),
        target_modules=config.peft.target_modules_regex,
        modules_to_save=config.peft.modules_to_save or None,
        ensure_weight_tying=config.peft.ensure_weight_tying,
    )
    model = get_peft_model(model, peft_config, adapter_name=config.peft.adapter_name)
    adapted_modules = _adapter_module_names(model, config.peft.adapter_name)
    if adapted_modules != expected_modules:
        missing = sorted(expected_modules - adapted_modules)
        unexpected = sorted(adapted_modules - expected_modules)
        raise RuntimeError(
            f"inserted LoRA module set differs from the validated target set; "
            f"missing={missing}, unexpected={unexpected}"
        )
    if any(
        any(part in name for part in _PROHIBITED_ADAPTER_PATHS) for name in adapted_modules
    ):
        raise RuntimeError("an inserted LoRA adapter reaches the vision/projector boundary")

    total_parameters = sum(parameter.numel() for parameter in model.parameters())
    trainable_parameters = sum(
        parameter.numel() for parameter in model.parameters() if parameter.requires_grad
    )
    unexpected_trainable = [
        name
        for name, parameter in model.named_parameters()
        if parameter.requires_grad
        and "lora_" not in name
        and ".modules_to_save." not in name
    ]
    if trainable_parameters == 0:
        raise RuntimeError("PEFT construction produced no trainable parameters")
    if unexpected_trainable:
        raise RuntimeError(
            "base parameters unexpectedly remain trainable: " + ", ".join(unexpected_trainable[:20])
        )
    return model, {
        "model_type": model_type,
        "is_encoder_decoder": is_encoder_decoder,
        "adapter_name": config.peft.adapter_name,
        "adapted_module_count": len(adapted_modules),
        "adapted_modules": sorted(adapted_modules),
        "total_parameters": total_parameters,
        "trainable_parameters": trainable_parameters,
        "trainable_parameter_ratio": trainable_parameters / total_parameters,
        "base_memory_footprint_bytes": model.get_memory_footprint(),
        "encoder_training_use_cache": encoder_config.use_cache,
        "training_use_cache": decoder_config.use_cache,
        "training_cache_config_count": len(runtime_cache_configs),
        "generation_use_cache": model.generation_config.use_cache,
    }


def _run_relative_directory(run_dir: Path, configured: str, description: str) -> Path:
    path = Path(configured)
    if path.is_absolute():
        raise ValueError(f"{description} must be relative to the run directory")
    unresolved = run_dir / path
    current = run_dir
    for part in path.parts:
        current /= part
        if current.is_symlink():
            raise ValueError(f"{description} must not traverse a symbolic link")
    resolved = unresolved.resolve()
    if not resolved.is_relative_to(run_dir.resolve()):
        raise ValueError(f"{description} escapes the run directory")
    return resolved


def build_training_arguments(config: TrainingConfig, run_dir: Path) -> Any:
    """Translate every configured Trainer choice without silent adaptive fallbacks."""

    try:
        from transformers import Seq2SeqTrainingArguments
    except ImportError as error:
        raise RuntimeError("Trainer construction requires the 'train' dependency group") from error

    checkpoints_dir = run_dir / "checkpoints"
    arguments: dict[str, Any] = {
        "output_dir": str(checkpoints_dir),
        "run_name": config.run.run_id,
        "do_train": True,
        "do_eval": (
            config.evaluation.strategy != "no" or config.evaluation.run_final_evaluation
        ),
        "do_predict": bool(config.evaluation.write_predictions_for),
        "num_train_epochs": config.optimization.num_train_epochs,
        "max_steps": config.optimization.max_steps,
        "per_device_train_batch_size": config.optimization.per_device_train_batch_size,
        "per_device_eval_batch_size": config.evaluation.per_device_batch_size,
        "gradient_accumulation_steps": config.optimization.gradient_accumulation_steps,
        "learning_rate": config.optimization.learning_rate,
        "optim": config.optimization.optimizer,
        "optim_args": config.optimization.optimizer_args,
        "adam_beta1": config.optimization.adam_beta1,
        "adam_beta2": config.optimization.adam_beta2,
        "adam_epsilon": config.optimization.adam_epsilon,
        "lr_scheduler_type": config.optimization.lr_scheduler_type,
        "lr_scheduler_kwargs": config.optimization.lr_scheduler_kwargs,
        # Transformers 5 represents a ratio as a float in warmup_steps.
        "warmup_steps": (
            config.optimization.schedule_free_adamw.warmup_steps
            if config.optimization.schedule_free_adamw is not None
            else config.optimization.warmup_ratio
        ),
        "weight_decay": config.optimization.weight_decay,
        "max_grad_norm": config.optimization.max_grad_norm,
        "label_smoothing_factor": config.optimization.label_smoothing_factor,
        "average_tokens_across_devices": config.optimization.average_tokens_across_devices,
        "seed": config.optimization.seed,
        "data_seed": config.optimization.data_seed,
        "full_determinism": config.optimization.full_determinism,
        "train_sampling_strategy": config.optimization.train_sampling_strategy,
        "length_column_name": config.optimization.length_column_name,
        "bf16": config.runtime.bf16,
        "fp16": config.runtime.fp16,
        "tf32": config.runtime.tf32,
        "gradient_checkpointing": config.runtime.gradient_checkpointing,
        "gradient_checkpointing_kwargs": {
            "use_reentrant": config.runtime.gradient_checkpointing_use_reentrant
        },
        "torch_compile": config.runtime.torch_compile,
        "torch_compile_backend": config.runtime.torch_compile_backend,
        "torch_compile_mode": config.runtime.torch_compile_mode,
        "torch_empty_cache_steps": config.runtime.torch_empty_cache_steps,
        "skip_memory_metrics": config.runtime.skip_memory_metrics,
        "use_cache": config.model.use_cache,
        "auto_find_batch_size": False,
        "eval_strategy": config.evaluation.strategy,
        "eval_on_start": config.evaluation.on_start,
        "predict_with_generate": config.evaluation.predict_with_generate,
        "generation_max_length": config.evaluation.generation_max_length,
        "generation_num_beams": config.evaluation.generation_num_beams,
        "eval_accumulation_steps": config.evaluation.accumulation_steps,
        "save_strategy": config.checkpoint.strategy,
        "save_total_limit": config.checkpoint.total_limit,
        "load_best_model_at_end": config.checkpoint.load_best_model_at_end,
        "metric_for_best_model": config.checkpoint.metric_for_best_model,
        "greater_is_better": config.checkpoint.greater_is_better,
        "enable_jit_checkpoint": config.checkpoint.enable_jit_checkpoint,
        "save_only_model": config.checkpoint.save_only_model,
        "save_on_each_node": config.checkpoint.save_on_each_node,
        "restore_callback_states_from_checkpoint": (
            config.checkpoint.restore_callback_states_from_checkpoint
        ),
        "logging_strategy": config.logging.strategy,
        "logging_first_step": config.logging.first_step,
        # Integrations are constructed explicitly so their artifact paths are deterministic.
        "report_to": [],
        "disable_tqdm": config.logging.disable_tqdm,
        "log_level": config.logging.log_level,
        "include_num_input_tokens_seen": config.logging.include_num_input_tokens_seen,
        "logging_nan_inf_filter": config.logging.filter_nan_inf,
        "log_on_each_node": config.logging.on_each_node,
        "dataloader_num_workers": config.dataloader.num_workers,
        "dataloader_pin_memory": config.dataloader.pin_memory,
        "dataloader_persistent_workers": config.dataloader.persistent_workers,
        "dataloader_prefetch_factor": config.dataloader.prefetch_factor,
        "dataloader_drop_last": config.dataloader.drop_last,
        # Prediction identities are recorded from sampler order, so completed batches
        # must be returned in that same order when worker processes are enabled.
        "dataloader_in_order": True,
        # Sampler metadata must survive until LengthGroupedSampler consumes input_length.
        # MetadataStrippingCollator removes it before model.forward.
        "remove_unused_columns": False,
        "label_names": ["labels"],
        "batch_eval_metrics": False,
        "include_for_metrics": [],
        "prediction_loss_only": False,
        "push_to_hub": False,
    }
    if config.evaluation.steps is not None:
        arguments["eval_steps"] = config.evaluation.steps
    if config.checkpoint.steps is not None:
        arguments["save_steps"] = config.checkpoint.steps
    if config.logging.steps is not None:
        arguments["logging_steps"] = config.logging.steps
    return Seq2SeqTrainingArguments(**arguments)


def _structured_log_callback(path: Path) -> tuple[Any, Any]:
    from transformers import TrainerCallback

    path.parent.mkdir(parents=True, exist_ok=True)
    stream = path.open("a", encoding="utf-8", buffering=1)

    class StructuredLogCallback(TrainerCallback):
        @staticmethod
        def _write(event: dict[str, Any]) -> None:
            stream.write(canonical_json_bytes(event).decode("utf-8") + "\n")

        def on_train_begin(
            self, args: Any, state: Any, control: Any, **kwargs: Any
        ) -> None:
            if state.is_world_process_zero:
                self._write(
                    {
                        "timestamp": _utc_now(),
                        "event": "train_begin",
                        "global_step": state.global_step,
                        "max_steps": state.max_steps,
                        "remaining_steps": max(0, state.max_steps - state.global_step),
                    }
                )

        def on_log(
            self,
            args: Any,
            state: Any,
            control: Any,
            logs: Any = None,
            **kwargs: Any,
        ) -> None:
            if not state.is_world_process_zero or logs is None:
                return
            event = {
                "timestamp": _utc_now(),
                "event": "trainer_log",
                "global_step": state.global_step,
                "max_steps": state.max_steps,
                "remaining_steps": max(0, state.max_steps - state.global_step),
                "progress_fraction": (
                    state.global_step / state.max_steps if state.max_steps else 0.0
                ),
                "epoch": state.epoch,
                "metrics": _json_ready(logs),
            }
            self._write(event)

        def on_save(
            self, args: Any, state: Any, control: Any, **kwargs: Any
        ) -> None:
            if state.is_world_process_zero:
                self._write(
                    {
                        "timestamp": _utc_now(),
                        "event": "checkpoint_saved",
                        "global_step": state.global_step,
                        "max_steps": state.max_steps,
                        "remaining_steps": max(0, state.max_steps - state.global_step),
                    }
                )

    return StructuredLogCallback(), stream


def _update_current_log(
    *, state: Any, logs: dict[str, Any], additions: Mapping[str, float]
) -> None:
    if not state.log_history or state.log_history[-1].get("step") != state.global_step:
        raise RuntimeError("Trainer callback received a log outside the current global step")
    logs.update(additions)
    state.log_history[-1].update(additions)


def _base_model_on_start_evaluation_callback(*, enabled: bool) -> Any:
    """Disable PEFT only for a fresh run's epoch-zero baseline evaluation."""

    from transformers import TrainerCallback

    class BaseModelOnStartEvaluationCallback(TrainerCallback):
        def __init__(self) -> None:
            self._adapter_context: Any | None = None

        def on_train_begin(
            self, args: Any, state: Any, control: Any, **kwargs: Any
        ) -> None:
            if not enabled:
                return
            if state.global_step != 0:
                raise RuntimeError("base-model on-start evaluation requires global step zero")
            if self._adapter_context is not None:
                raise RuntimeError("base-model evaluation adapter context is already active")
            model = kwargs.get("model")
            disable_adapter = getattr(model, "disable_adapter", None)
            if not callable(disable_adapter):
                raise RuntimeError("PEFT model does not expose disable_adapter()")
            adapter_context = disable_adapter()
            if not hasattr(adapter_context, "__enter__") or not hasattr(
                adapter_context, "__exit__"
            ):
                raise RuntimeError("disable_adapter() did not return a context manager")
            adapter_context.__enter__()
            self._adapter_context = adapter_context

        def on_log(
            self,
            args: Any,
            state: Any,
            control: Any,
            logs: dict[str, Any] | None = None,
            **kwargs: Any,
        ) -> None:
            if (
                not enabled
                or logs is None
                or "eval_runtime" not in logs
                or state.global_step != 0
            ):
                return
            if self._adapter_context is None:
                raise RuntimeError("epoch-zero evaluation ran with PEFT adapters enabled")
            _update_current_log(
                state=state,
                logs=logs,
                additions={"eval_is_base_model": 1.0},
            )

        def on_evaluate(
            self, args: Any, state: Any, control: Any, **kwargs: Any
        ) -> None:
            if not enabled or state.global_step != 0:
                return
            if self._adapter_context is None:
                raise RuntimeError("base-model evaluation adapter context is not active")
            adapter_context = self._adapter_context
            self._adapter_context = None
            adapter_context.__exit__(None, None, None)

        def on_epoch_begin(
            self, args: Any, state: Any, control: Any, **kwargs: Any
        ) -> None:
            if self._adapter_context is not None:
                raise RuntimeError(
                    "PEFT adapters remained disabled after the on-start evaluation"
                )

    return BaseModelOnStartEvaluationCallback()


def _cumulative_loss_callback() -> Any:
    """Publish the optimizer-step-weighted cumulative training loss."""

    from transformers import TrainerCallback

    class CumulativeLossCallback(TrainerCallback):
        def on_log(
            self,
            args: Any,
            state: Any,
            control: Any,
            logs: dict[str, Any] | None = None,
            **kwargs: Any,
        ) -> None:
            if logs is None or "loss" not in logs:
                return
            weighted_loss = 0.0
            previous_step = 0
            for entry in state.log_history:
                if "loss" not in entry:
                    continue
                step = entry.get("step")
                loss = entry["loss"]
                if isinstance(step, bool) or not isinstance(step, int) or step <= previous_step:
                    raise RuntimeError("training-loss history has non-increasing global steps")
                if isinstance(loss, bool) or not isinstance(loss, (int, float)):
                    raise RuntimeError("training-loss history contains a non-numeric loss")
                weighted_loss += float(loss) * (step - previous_step)
                previous_step = step
            if previous_step != state.global_step:
                raise RuntimeError("training-loss history does not end at the current global step")
            _update_current_log(
                state=state,
                logs=logs,
                additions={"train_cumulative_loss": weighted_loss / previous_step},
            )

    return CumulativeLossCallback()


def _evaluation_memory_cleanup_callback() -> Any:
    """Measure generated evaluation and release its cache before training resumes."""

    import torch
    from transformers import TrainerCallback

    gibibyte = 1024**3

    class EvaluationMemoryCleanupCallback(TrainerCallback):
        def __init__(self) -> None:
            self._before_evaluation: dict[str, float] | None = None

        def _mark_evaluation_start(self, control: Any) -> None:
            if not control.should_evaluate:
                return
            if not torch.cuda.is_available():
                raise RuntimeError("generated evaluation memory tracking requires CUDA")
            if self._before_evaluation is not None:
                raise RuntimeError("a second evaluation began before the first one completed")
            torch.cuda.synchronize()
            free, total = torch.cuda.mem_get_info()
            torch.cuda.reset_peak_memory_stats()
            self._before_evaluation = {
                "eval_cuda_driver_free_before_gib": free / gibibyte,
                "eval_cuda_driver_used_before_gib": (total - free) / gibibyte,
            }

        def on_step_end(
            self, args: Any, state: Any, control: Any, **kwargs: Any
        ) -> Any:
            self._mark_evaluation_start(control)
            return control

        def on_epoch_end(
            self, args: Any, state: Any, control: Any, **kwargs: Any
        ) -> Any:
            self._mark_evaluation_start(control)
            return control

        def on_log(
            self,
            args: Any,
            state: Any,
            control: Any,
            logs: dict[str, Any] | None = None,
            **kwargs: Any,
        ) -> None:
            if logs is None or "eval_runtime" not in logs:
                return
            if not torch.cuda.is_available():
                raise RuntimeError("evaluation memory cleanup requires CUDA")
            torch.cuda.synchronize()
            free_before_cleanup, total = torch.cuda.mem_get_info()
            allocated = torch.cuda.memory_allocated()
            reserved_before = torch.cuda.memory_reserved()
            peak_allocated = torch.cuda.max_memory_allocated()
            peak_reserved = torch.cuda.max_memory_reserved()
            torch.cuda.empty_cache()
            free_after_cleanup, total_after_cleanup = torch.cuda.mem_get_info()
            if total_after_cleanup != total:
                raise RuntimeError("CUDA device memory total changed during evaluation cleanup")
            reserved_after = torch.cuda.memory_reserved()
            before_evaluation = self._before_evaluation or {}
            self._before_evaluation = None
            _update_current_log(
                state=state,
                logs=logs,
                additions={
                    **before_evaluation,
                    "eval_cuda_allocated_gib": allocated / gibibyte,
                    "eval_cuda_peak_allocated_gib": peak_allocated / gibibyte,
                    "eval_cuda_reserved_before_cleanup_gib": reserved_before / gibibyte,
                    "eval_cuda_peak_reserved_gib": peak_reserved / gibibyte,
                    "eval_cuda_reserved_after_cleanup_gib": reserved_after / gibibyte,
                    "eval_cuda_reserved_released_gib": (reserved_before - reserved_after)
                    / gibibyte,
                    "eval_cuda_driver_used_before_cleanup_gib": (
                        total - free_before_cleanup
                    )
                    / gibibyte,
                    "eval_cuda_driver_used_after_cleanup_gib": (
                        total - free_after_cleanup
                    )
                    / gibibyte,
                },
            )

    return EvaluationMemoryCleanupCallback()


def _final_evaluation_policy_callback(run_final_evaluation: bool) -> Any:
    """Make the configured final-evaluation policy override Trainer's implicit one."""

    from transformers import TrainerCallback

    class FinalEvaluationPolicyCallback(TrainerCallback):
        def on_step_end(
            self, args: Any, state: Any, control: Any, **kwargs: Any
        ) -> Any:
            if (
                not run_final_evaluation
                and state.global_step >= state.max_steps
                and args.eval_strategy.value == "steps"
                and state.global_step % state.eval_steps != 0
            ):
                control.should_evaluate = False
            return control

    return FinalEvaluationPolicyCallback()


def _early_stopping_callback(config: TrainingConfig) -> Any | None:
    patience = config.evaluation.early_stopping_patience
    threshold = config.evaluation.early_stopping_threshold
    if patience is None:
        if threshold is not None:
            raise RuntimeError("early-stopping configuration was not normalized")
        return None
    if threshold is None:
        raise RuntimeError("early-stopping configuration was not normalized")

    from transformers import EarlyStoppingCallback

    return EarlyStoppingCallback(
        early_stopping_patience=patience,
        early_stopping_threshold=threshold,
    )


def _evaluation_metrics_at_step(state: Any, step: int) -> dict[str, Any] | None:
    """Return the most recent completed evaluation at one optimizer step."""

    for entry in reversed(state.log_history):
        if entry.get("step") != step or "eval_runtime" not in entry:
            continue
        return {key: value for key, value in entry.items() if key != "step"}
    return None


def _start_mlflow_tracking(
    config: TrainingConfig,
    tags: Mapping[str, str],
) -> tuple[Any, Any, dict[str, str]]:
    """Start the one runtime-owned MLflow run and return its Trainer callback."""

    try:
        mlflow: Any = importlib.import_module("mlflow")
        integration_module: Any = importlib.import_module(
            "transformers.integrations.integration_utils"
        )
    except ImportError as error:
        raise RuntimeError("MLflow tracking requires the 'train' dependency group") from error
    mlflow_client_class: Any = mlflow.MlflowClient
    callback_class: Any = integration_module.MLflowCallback

    if mlflow.active_run() is not None:
        raise RuntimeError("an MLflow run is already active before training startup")

    mlflow_config = config.logging.mlflow
    environment = {
        "HF_MLFLOW_LOG_ARTIFACTS": "TRUE" if mlflow_config.log_artifacts else "FALSE",
        "MLFLOW_EXPERIMENT_NAME": mlflow_config.experiment_name,
        "MLFLOW_FLATTEN_PARAMS": "TRUE" if mlflow_config.flatten_params else "FALSE",
        "MLFLOW_NESTED_RUN": "FALSE",
        "MLFLOW_TAGS": canonical_json_bytes(dict(tags)).decode("utf-8"),
        "MLFLOW_TRACKING_URI": mlflow_config.tracking_uri,
    }
    for name, value in environment.items():
        os.environ[name] = value
    os.environ.pop("MLFLOW_RUN_ID", None)
    if mlflow_config.max_log_params is None:
        os.environ.pop("MLFLOW_MAX_LOG_PARAMS", None)
    else:
        os.environ["MLFLOW_MAX_LOG_PARAMS"] = str(mlflow_config.max_log_params)

    mlflow.set_tracking_uri(mlflow_config.tracking_uri)
    # A read-only request proves the configured server is reachable before work begins.
    mlflow_client_class(tracking_uri=mlflow_config.tracking_uri).search_experiments(
        max_results=1
    )
    mlflow.set_system_metrics_sampling_interval(
        mlflow_config.system_metrics_sampling_interval
    )
    mlflow.set_system_metrics_samples_before_logging(
        mlflow_config.system_metrics_samples_before_logging
    )
    mlflow.set_experiment(mlflow_config.experiment_name)
    if mlflow_config.resume_run_id is None:
        run = mlflow.start_run(
            run_name=config.run.run_id,
            tags=dict(tags),
            log_system_metrics=mlflow_config.system_metrics,
        )
    else:
        run = mlflow.start_run(
            run_id=mlflow_config.resume_run_id,
            log_system_metrics=mlflow_config.system_metrics,
        )
        mlflow.set_tags(dict(tags))
    reference = {
        "tracking_uri": mlflow_config.tracking_uri,
        "experiment_name": mlflow_config.experiment_name,
        "experiment_id": run.info.experiment_id,
        "run_id": run.info.run_id,
    }
    return callback_class(), mlflow, reference


def _finish_mlflow_tracking(
    mlflow_module: Any,
    reference: Mapping[str, str],
    *,
    status: str,
) -> None:
    active_run = mlflow_module.active_run()
    if active_run is None:
        raise RuntimeError("the runtime-owned MLflow run ended unexpectedly")
    if active_run.info.run_id != reference["run_id"]:
        raise RuntimeError("a different MLflow run became active during training")
    mlflow_module.end_run(status=status)
    final_run = mlflow_module.MlflowClient(
        tracking_uri=reference["tracking_uri"]
    ).get_run(reference["run_id"])
    if final_run.info.status != status:
        raise RuntimeError(
            f"MLflow run status mismatch: expected {status}, found {final_run.info.status}"
        )


def _json_ready(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _json_ready(child) for key, child in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_ready(child) for child in value]
    if hasattr(value, "item") and callable(value.item):
        return value.item()
    return value


def _decode_prediction_output(output: Any, tokenizer: Any) -> tuple[list[str], list[str]]:
    predictions = (
        output.predictions[0]
        if isinstance(output.predictions, tuple)
        else output.predictions
    )
    labels = output.label_ids.copy()
    labels[labels == -100] = tokenizer.pad_token_id
    return (
        tokenizer.batch_decode(predictions, skip_special_tokens=True),
        tokenizer.batch_decode(labels, skip_special_tokens=True),
    )


def _write_prediction_rows(
    *,
    path: Path,
    document_ids: Sequence[str],
    generated_texts: Sequence[str],
    reference_texts: Sequence[str],
    task: TrainingTask,
) -> dict[str, float]:
    metrics, assessments = structured_metrics(generated_texts, reference_texts, task)
    if len(document_ids) != len(assessments):
        raise RuntimeError("prediction count differs from the source split record count")
    lines = []
    for document_id, assessment in zip(document_ids, assessments, strict=True):
        row = {"document_id": document_id, **assessment.to_dict()}
        lines.append(canonical_json_bytes(row))
    atomic_write_bytes(path, b"\n".join(lines) + b"\n")
    return metrics


def _predict_and_publish(
    *,
    trainer: Any,
    tokenizer: Any,
    task: TrainingTask,
    dataset: Any,
    split: str,
    metric_key_prefix: str,
    predictions_dir: Path,
) -> tuple[dict[str, Any], Path]:
    ordered_prediction = trainer.predict_with_identities(
        dataset,
        metric_key_prefix=metric_key_prefix,
    )
    output = ordered_prediction.output
    generated, references = _decode_prediction_output(output, tokenizer)
    prediction_path = predictions_dir / f"{split}.jsonl"
    independently_computed = _write_prediction_rows(
        path=prediction_path,
        document_ids=ordered_prediction.document_ids,
        generated_texts=generated,
        reference_texts=references,
        task=task,
    )
    metrics = cast(dict[str, Any], _json_ready(output.metrics))
    for name, expected in independently_computed.items():
        metric_name = f"{metric_key_prefix}_{name}"
        actual = metrics.get(metric_name)
        if not isinstance(actual, (int, float)) or not math.isclose(
            float(actual), expected, rel_tol=1e-12, abs_tol=1e-12
        ):
            raise RuntimeError(
                f"persisted prediction metric {metric_name!r} differs from Trainer output"
            )
    return metrics, prediction_path


def _source_code_identity(project_root: Path) -> list[dict[str, Any]]:
    project_files = [project_root / "pyproject.toml", project_root / "uv.lock"]
    module_names = [
        "document_ocr.atomic",
        "document_ocr.hashing",
        "document_ocr.label_schemas.bill_of_lading",
        "document_ocr.training.cli",
        "document_ocr.training.collator",
        "document_ocr.training.config",
        "document_ocr.training.data",
        "document_ocr.training.eva",
        "document_ocr.training.metrics",
        "document_ocr.training.prediction",
        "document_ocr.training.prompting",
        "document_ocr.training.runtime",
        "document_ocr.training.splitting",
        "document_ocr.training.tasks",
    ]
    results = []
    for path in project_files:
        if not path.is_file() or path.is_symlink():
            raise RuntimeError(f"training source provenance path is not a regular file: {path}")
        results.append(
            {
                "kind": "project_lock_input",
                "path": str(path.relative_to(project_root)),
                "sha256": sha256_file(path),
                "bytes": path.stat().st_size,
            }
        )
    for module_name in module_names:
        module = importlib.import_module(module_name)
        module_file = getattr(module, "__file__", None)
        if module_file is None:
            raise RuntimeError(f"loaded training module has no file identity: {module_name}")
        path = Path(module_file)
        if not path.is_file() or path.is_symlink():
            raise RuntimeError(f"loaded training module is not a regular file: {path}")
        results.append(
            {
                "kind": "loaded_python_module",
                "module": module_name,
                "path": str(path.resolve(strict=True)),
                "sha256": sha256_file(path),
                "bytes": path.stat().st_size,
            }
        )
    return results


def _environment_report(
    project_root: Path,
    hardware: Mapping[str, Any],
    model_facts: Mapping[str, Any],
) -> dict[str, Any]:
    packages = {}
    for package in _TRAINING_PACKAGES:
        try:
            packages[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError as error:
            raise RuntimeError(f"required training package is not installed: {package}") from error
    return {
        "python": sys.version,
        "platform": platform.platform(),
        "packages": packages,
        "hardware": dict(hardware),
        "model": dict(model_facts),
        "source_code": _source_code_identity(project_root),
    }


def _resume_checkpoint(project_root: Path, run_dir: Path, configured: str | None) -> Path | None:
    if configured is None:
        return None
    path = resolve_config_path(project_root, configured).resolve(strict=True)
    if not path.is_dir() or path.is_symlink():
        raise RuntimeError(f"resume checkpoint is not a real directory: {path}")
    if not path.is_relative_to((run_dir / "checkpoints").resolve()):
        raise RuntimeError("resume checkpoint must be inside this run's checkpoint directory")
    return path


def _preflight_run_destination(project_root: Path, config: TrainingConfig) -> None:
    """Reject an unusable run destination before tokenizer/model allocation."""

    output_root = resolve_config_path(project_root, config.run.output_dir).resolve()
    if output_root == Path(output_root.anchor):
        raise RuntimeError("training output root must not be a filesystem root")
    run_dir = output_root / config.run.run_id
    resume = _resume_checkpoint(
        project_root, run_dir, config.checkpoint.resume_from_checkpoint
    )
    if (run_dir / "manifest.json").exists() or (run_dir / "manifest.json").is_symlink():
        raise RuntimeError("completed training run cannot be resumed or overwritten")
    if resume is None and (run_dir.exists() or run_dir.is_symlink()):
        raise RuntimeError(
            "run directory already exists and no explicit resume checkpoint was configured: "
            f"{run_dir}"
        )
    if resume is not None and (not run_dir.is_dir() or run_dir.is_symlink()):
        raise RuntimeError(f"resumed run directory is not a real directory: {run_dir}")


def _prepare_run_directory(
    *,
    project_root: Path,
    config_payload: bytes,
    config: TrainingConfig,
    prompt: PromptTemplate,
    prepared: PreparedDatasets,
    environment: Mapping[str, Any],
) -> tuple[Path, Path | None, list[Path]]:
    output_root = resolve_config_path(project_root, config.run.output_dir).resolve()
    if output_root == Path(output_root.anchor):
        raise RuntimeError("training output root must not be a filesystem root")
    run_dir = output_root / config.run.run_id
    resume = _resume_checkpoint(project_root, run_dir, config.checkpoint.resume_from_checkpoint)
    if resume is None:
        try:
            run_dir.mkdir(parents=True, exist_ok=False)
        except FileExistsError as error:
            raise RuntimeError(
                "run directory already exists and no explicit resume checkpoint was configured: "
                f"{run_dir}"
            ) from error
    elif not run_dir.is_dir() or run_dir.is_symlink():
        raise RuntimeError(f"resumed run directory is not a real directory: {run_dir}")
    if (run_dir / "manifest.json").exists() or (run_dir / "manifest.json").is_symlink():
        raise RuntimeError("completed training run cannot be resumed or overwritten")

    config_artifacts = [run_dir / "config.yaml", run_dir / "resolved-config.json"]
    if resume is None:
        atomic_publish_bytes(run_dir / "config.yaml", config_payload)
        atomic_publish_json(run_dir / "resolved-config.json", config.model_dump(mode="json"))
    else:
        existing_config = TrainingConfig.model_validate_json(
            read_regular_file_bytes(run_dir / "resolved-config.json"), strict=True
        )
        original_config = parse_training_config(read_regular_file_bytes(run_dir / "config.yaml"))
        if original_config != existing_config:
            raise RuntimeError("frozen raw and resolved run configurations disagree")

        def without_resume(value: TrainingConfig) -> TrainingConfig:
            return value.model_copy(
                update={
                    "checkpoint": value.checkpoint.model_copy(
                        update={
                            "resume_from_checkpoint": None,
                            "allow_resume_source_code_drift": False,
                        }
                    ),
                    "logging": value.logging.model_copy(
                        update={
                            "mlflow": value.logging.mlflow.model_copy(
                                update={"resume_run_id": None}
                            )
                        }
                    ),
                }
            )

        if without_resume(existing_config) != without_resume(config):
            raise RuntimeError("resumed run configuration differs from its immutable run contract")
        resume_invocation_id = f"{resume.name}-{sha256_bytes(config_payload)}"
        resume_config_path = (
            run_dir / "resume-invocations" / f"{resume_invocation_id}.yaml"
        )
        atomic_publish_bytes(resume_config_path, config_payload)
        config_artifacts.append(resume_config_path)
    atomic_publish_bytes(run_dir / "prompt.txt", prompt.encoded)
    dataset_report_path = run_dir / "dataset-report.json"
    # Compare the JSON value that is actually persisted, not Python container
    # implementation details. DatasetInspection contains tuples (for example,
    # source_files), which JSON correctly serializes as arrays and loads back as
    # lists on resume.
    prepared_report = cast(dict[str, Any], _json_ready(prepared.report()))
    if resume is None:
        atomic_publish_json(dataset_report_path, prepared_report)
    else:
        try:
            existing_report = json.loads(
                read_regular_file_bytes(dataset_report_path).decode("utf-8")
            )
        except (UnicodeDecodeError, json.JSONDecodeError) as error:
            raise RuntimeError(
                "resumed run dataset report is not valid UTF-8 JSON"
            ) from error
        if existing_report != prepared_report:
            raise RuntimeError(
                "resumed dataset inspection or partition differs from its immutable run report"
            )
    environment_path = run_dir / "environment.json"
    prepared_environment = cast(dict[str, Any], _json_ready(environment))
    if resume is None:
        atomic_publish_json(environment_path, prepared_environment)
    else:
        try:
            existing_environment = json.loads(
                read_regular_file_bytes(environment_path).decode("utf-8")
            )
        except (UnicodeDecodeError, json.JSONDecodeError) as error:
            raise RuntimeError(
                "resumed run environment report is not valid UTF-8 JSON"
            ) from error
        if config.checkpoint.allow_resume_source_code_drift:
            existing_compatible = {
                key: value
                for key, value in existing_environment.items()
                if key != "source_code"
            }
            prepared_compatible = {
                key: value
                for key, value in prepared_environment.items()
                if key != "source_code"
            }
            if existing_compatible != prepared_compatible:
                raise RuntimeError(
                    "resumed environment differs outside the explicitly allowed "
                    "source-code identity"
                )
        elif existing_environment != prepared_environment:
            raise RuntimeError(
                "resumed environment differs from its immutable run report; "
                "source-code-only drift requires explicit acknowledgement"
            )
        resume_environment_path = (
            run_dir
            / "resume-invocations"
            / f"{resume_invocation_id}.environment.json"
        )
        atomic_publish_json(resume_environment_path, prepared_environment)
        config_artifacts.append(resume_environment_path)
    atomic_write_bytes(
        run_dir / "status.json",
        canonical_json_bytes(
            {
                "schema_version": 1,
                "run_id": config.run.run_id,
                "status": "running",
                "at": _utc_now(),
            }
        )
        + b"\n",
    )
    return run_dir, resume, config_artifacts


def _artifact_descriptors(run_dir: Path, paths: Iterable[Path]) -> list[dict[str, Any]]:
    descriptors = []
    for path in sorted(set(paths)):
        if not path.is_file() or path.is_symlink():
            raise RuntimeError(f"published training artifact is not a regular file: {path}")
        resolved = path.resolve(strict=True)
        if not resolved.is_relative_to(run_dir.resolve()):
            raise RuntimeError(f"published training artifact escapes the run directory: {path}")
        descriptors.append(
            {
                "path": str(resolved.relative_to(run_dir.resolve())),
                "sha256": sha256_file(resolved),
                "bytes": resolved.stat().st_size,
            }
        )
    return descriptors


def run_training(
    *,
    project_root: Path,
    config_path: Path,
    config: TrainingConfig,
    prompt: PromptTemplate,
    task: TrainingTask,
) -> dict[str, Any]:
    """Execute one explicit GPU training run and publish its manifest only after completion."""

    _configure_cuda_allocator(config)
    try:
        from transformers import DataCollatorForSeq2Seq, Seq2SeqTrainer, set_seed
    except ImportError as error:
        raise RuntimeError("training requires the 'train' dependency group") from error

    if config_path.is_symlink() or not config_path.is_file():
        raise RuntimeError(f"configuration provenance path is not a regular file: {config_path}")
    config_payload = config_path.read_bytes()
    if parse_training_config(config_payload) != config:
        raise RuntimeError("configuration changed between CLI validation and training startup")
    _preflight_run_destination(project_root, config)
    hardware = validate_training_hardware(config)
    set_seed(config.optimization.seed, deterministic=config.optimization.full_determinism)
    tokenizer = load_tokenizer(config)
    prepared = prepare_datasets(
        project_root=project_root,
        config=config,
        prompt=prompt,
        task=task,
        tokenizer=tokenizer,
    )
    model, model_facts = load_lora_model(config)
    environment = _environment_report(project_root, hardware, model_facts)
    run_dir, resume_checkpoint, config_artifacts = _prepare_run_directory(
        project_root=project_root,
        config_payload=config_payload,
        config=config,
        prompt=prompt,
        prepared=prepared,
        environment=environment,
    )
    structured_log_path = _run_relative_directory(
        run_dir, config.logging.structured_log, "structured_log"
    )
    structured_stream: Any | None = None
    mlflow_module: Any | None = None
    mlflow_reference: dict[str, str] | None = None
    artifact_paths = [
        *config_artifacts,
        run_dir / "prompt.txt",
        run_dir / "dataset-report.json",
        run_dir / "environment.json",
    ]
    all_metrics: dict[str, Any] = {}
    try:
        structured_callback, structured_stream = _structured_log_callback(
            structured_log_path
        )
        generated_tags = {
            "document_ocr.dataset_cache_identity": prepared.cache_identity,
            "document_ocr.model_revision": config.model.revision,
            "document_ocr.run_id": config.run.run_id,
            "document_ocr.task": config.task,
        }
        duplicate_tags = set(generated_tags) & set(config.logging.mlflow.tags)
        if duplicate_tags:
            raise RuntimeError(
                "configured MLflow tags collide with runtime provenance tags: "
                + ", ".join(sorted(duplicate_tags))
            )
        mlflow_callback, mlflow_module, mlflow_reference = _start_mlflow_tracking(
            config,
            {**config.logging.mlflow.tags, **generated_tags},
        )
        mlflow_reference_path = run_dir / "mlflow-run.json"
        if resume_checkpoint is None:
            atomic_publish_json(mlflow_reference_path, mlflow_reference)
        else:
            existing_reference = json.loads(
                read_regular_file_bytes(mlflow_reference_path).decode("utf-8")
            )
            if existing_reference != mlflow_reference:
                raise RuntimeError("resumed MLflow run differs from its immutable run reference")
        artifact_paths.append(mlflow_reference_path)
        # The metric callbacks enrich mutable logs before MLflow and the durable JSONL
        # consume them. Default progress rendering still receives the original payload.
        callbacks = [
            _base_model_on_start_evaluation_callback(
                enabled=config.evaluation.on_start and resume_checkpoint is None
            ),
            _final_evaluation_policy_callback(config.evaluation.run_final_evaluation),
            _cumulative_loss_callback(),
            _evaluation_memory_cleanup_callback(),
            mlflow_callback,
            structured_callback,
        ]
        early_stopping_callback = _early_stopping_callback(config)
        if early_stopping_callback is not None:
            callbacks.insert(1, early_stopping_callback)
        training_arguments = build_training_arguments(config, run_dir)
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
        if config.peft.eva is not None:
            from document_ocr.training.eva import initialize_eva_for_run

            eva_path = run_dir / "eva-initialization.json"
            initialize_eva_for_run(
                path=eva_path, resuming=resume_checkpoint is not None,
                model=model, train_dataset=prepared.datasets["train"], collator=collator,
                peft=config.peft, dataset_identity=prepared.cache_identity,
                device=training_arguments.device,
            )
            artifact_paths.append(eva_path)
        evaluation_dataset = prepared.datasets.get(config.evaluation.split)
        class StandardSeq2SeqTrainer(SamplerAwarePredictionMixin, Seq2SeqTrainer):
            pass

        class ScheduleFreeSeq2SeqTrainer(
            SamplerAwarePredictionMixin, ScheduleFreeTrainerMixin, Seq2SeqTrainer
        ):
            schedule_free_adamw = config.optimization.schedule_free_adamw

        trainer_type = (
            ScheduleFreeSeq2SeqTrainer
            if config.optimization.schedule_free_adamw is not None
            else StandardSeq2SeqTrainer
        )
        trainer = trainer_type(
            model=model,
            args=training_arguments,
            data_collator=collator,
            train_dataset=prepared.datasets["train"],
            eval_dataset=evaluation_dataset,
            processing_class=tokenizer,
            compute_metrics=(
                make_compute_metrics(tokenizer, task)
                if config.evaluation.predict_with_generate and evaluation_dataset is not None
                else None
            ),
            callbacks=callbacks,
        )

        train_result = trainer.train(
            resume_from_checkpoint=str(resume_checkpoint) if resume_checkpoint is not None else None
        )
        train_metrics = _json_ready(train_result.metrics)
        trainer.log_metrics("train", train_metrics)
        trainer.save_metrics("train", train_metrics)
        trainer.save_state()
        all_metrics["train"] = train_metrics

        predictions_dir = run_dir / "predictions"
        published_prediction_splits: set[str] = set()
        if config.evaluation.run_final_evaluation:
            evaluation_split = config.evaluation.split
            if evaluation_split in config.evaluation.write_predictions_for:
                evaluation_metrics, prediction_path = _predict_and_publish(
                    trainer=trainer,
                    tokenizer=tokenizer,
                    task=task,
                    dataset=prepared.datasets[evaluation_split],
                    split=evaluation_split,
                    metric_key_prefix="eval",
                    predictions_dir=predictions_dir,
                )
                trainer.log_metrics("eval", evaluation_metrics)
                artifact_paths.append(prediction_path)
                published_prediction_splits.add(evaluation_split)
            else:
                evaluation_metrics = None
                # Once Trainer reloads the best checkpoint, metrics from the final training
                # step describe a different model and must not be reused.
                if not config.checkpoint.load_best_model_at_end:
                    evaluation_metrics = _evaluation_metrics_at_step(
                        trainer.state, trainer.state.global_step
                    )
                if evaluation_metrics is None:
                    evaluation_metrics = _json_ready(
                        trainer.evaluate(metric_key_prefix="eval")
                    )
                    trainer.log_metrics("eval", evaluation_metrics)
            trainer.save_metrics("eval", evaluation_metrics)
            all_metrics[f"{evaluation_split}_evaluation"] = evaluation_metrics

        for split in config.evaluation.write_predictions_for:
            if split in published_prediction_splits:
                continue
            dataset = prepared.datasets[split]
            metrics, prediction_path = _predict_and_publish(
                trainer=trainer,
                tokenizer=tokenizer,
                task=task,
                dataset=dataset,
                split=split,
                metric_key_prefix=split,
                predictions_dir=predictions_dir,
            )
            trainer.log_metrics(split, metrics)
            trainer.save_metrics(split, metrics)
            all_metrics[split] = metrics
            artifact_paths.append(prediction_path)

        final_adapter_dir = run_dir / "final-adapter"
        trainer.save_model(str(final_adapter_dir))
        artifact_paths.extend(path for path in final_adapter_dir.rglob("*") if path.is_file())
        checkpoints_dir = run_dir / "checkpoints"
        artifact_paths.extend(
            path for path in checkpoints_dir.rglob("*") if path.is_file()
        )
        structured_stream.flush()
        structured_stream.close()
        artifact_paths.append(structured_log_path)
        _finish_mlflow_tracking(mlflow_module, mlflow_reference, status="FINISHED")
        manifest = {
            "schema_version": 1,
            "status": "complete",
            "run_id": config.run.run_id,
            "completed_at": _utc_now(),
            "model_revision": config.model.revision,
            "tokenizer_revision": config.model.tokenizer_revision,
            "task": config.task,
            "dataset_cache_identity": prepared.cache_identity,
            "mlflow": mlflow_reference,
            "metrics": all_metrics,
            "artifacts": _artifact_descriptors(run_dir, artifact_paths),
        }
        atomic_write_bytes(
            run_dir / "status.json",
            canonical_json_bytes(
                {
                    "schema_version": 1,
                    "run_id": config.run.run_id,
                    "status": "artifacts_complete_manifest_pending",
                    "at": _utc_now(),
                }
            )
            + b"\n",
        )
        # This immutable manifest is deliberately the final run mutation.
        atomic_publish_json(run_dir / "manifest.json", manifest)
        return manifest
    except BaseException as error:
        tracking_end_error: BaseException | None = None
        if (
            mlflow_module is not None
            and mlflow_reference is not None
            and mlflow_module.active_run() is not None
        ):
            try:
                _finish_mlflow_tracking(
                    mlflow_module,
                    mlflow_reference,
                    status="FAILED",
                )
            except BaseException as end_error:
                tracking_end_error = end_error
        status = {
            "schema_version": 1,
            "run_id": config.run.run_id,
            "status": "failed",
            "error_type": type(error).__name__,
            "at": _utc_now(),
        }
        if tracking_end_error is not None:
            status["mlflow_end_error_type"] = type(tracking_end_error).__name__
        atomic_write_bytes(
            run_dir / "status.json",
            canonical_json_bytes(status) + b"\n",
        )
        if tracking_end_error is not None:
            raise BaseExceptionGroup(
                "training failed and its MLflow run could not be closed",
                [error, tracking_end_error],
            ) from error
        raise
    finally:
        if structured_stream is not None and not structured_stream.closed:
            structured_stream.close()
