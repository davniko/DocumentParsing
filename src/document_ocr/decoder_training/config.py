"""Strict, method-discriminated configuration for decoder-only KIE training."""

from __future__ import annotations

import re
from collections.abc import Hashable
from pathlib import Path, PurePath
from typing import Annotated, Any, Literal

import yaml
from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    StringConstraints,
    field_validator,
    model_validator,
)
from yaml.constructor import ConstructorError
from yaml.nodes import MappingNode

from document_ocr.training.config import (
    DatasetFieldsConfig,
    DatasetFileConfig,
    PromptConfig,
    RuntimeDatasetPartitionConfig,
    TaskConstraintsConfig,
)

NonEmptyString = Annotated[str, StringConstraints(strip_whitespace=True, min_length=1)]
PositiveInteger = Annotated[int, Field(gt=0)]
NonNegativeInteger = Annotated[int, Field(ge=0)]
PositiveFloat = Annotated[float, Field(gt=0, allow_inf_nan=False)]
UnitFloat = Annotated[float, Field(ge=0.0, le=1.0, allow_inf_nan=False)]
OpenUnitFloat = Annotated[float, Field(gt=0.0, lt=1.0, allow_inf_nan=False)]

_SHA256_PATTERN = re.compile(r"^[0-9a-f]{64}$")
_REVISION_PATTERN = re.compile(r"^[0-9a-f]{40}$")
_RUN_ID_PATTERN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$")
_ENV_PATTERN = re.compile(r"^[A-Z_][A-Z0-9_]{0,127}$")


class _UniqueKeySafeLoader(yaml.SafeLoader):
    def construct_mapping(self, node: MappingNode, deep: bool = False) -> dict[Any, Any]:
        self.flatten_mapping(node)
        mapping: dict[Any, Any] = {}
        for key_node, value_node in node.value:
            key = self.construct_object(key_node, deep=deep)
            if not isinstance(key, Hashable):
                raise ConstructorError(
                    "while constructing a mapping",
                    node.start_mark,
                    "found an unhashable key",
                    key_node.start_mark,
                )
            if key in mapping:
                raise ConstructorError(
                    "while constructing a mapping",
                    node.start_mark,
                    f"found duplicate key {key!r}",
                    key_node.start_mark,
                )
            mapping[key] = self.construct_object(value_node, deep=deep)
        return mapping


class _StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True, allow_inf_nan=False)


def _safe_path(value: str) -> str:
    path = PurePath(value)
    if not value.strip() or "\x00" in value:
        raise ValueError("path must be a non-empty filesystem path")
    if not path.is_absolute() and any(part in {"", ".", ".."} for part in path.parts):
        raise ValueError("relative paths must not contain empty, '.' or '..' components")
    if path.is_absolute() and path == PurePath(path.anchor):
        raise ValueError("path must not be a filesystem root")
    return value


class RunConfig(_StrictModel):
    run_id: NonEmptyString
    output_dir: NonEmptyString
    seed: NonNegativeInteger

    @field_validator("run_id")
    @classmethod
    def safe_run_id(cls, value: str) -> str:
        if not _RUN_ID_PATTERN.fullmatch(value):
            raise ValueError("run_id must be a filesystem-safe identifier")
        return value

    @field_validator("output_dir")
    @classmethod
    def safe_output_dir(cls, value: str) -> str:
        return _safe_path(value)


class ModelConfig(_StrictModel):
    family: Literal["qwen3_5", "qwen3"]
    name_or_path: NonEmptyString
    revision: NonEmptyString
    token_env: NonEmptyString | None
    local_files_only: bool
    trust_remote_code: Literal[False]
    text_only: Literal[True]
    dtype: Literal["bfloat16"]
    load_in_4bit: Literal[False]
    load_in_8bit: Literal[False]
    fast_inference: Literal[False]

    @field_validator("revision")
    @classmethod
    def immutable_revision(cls, value: str) -> str:
        if not _REVISION_PATTERN.fullmatch(value):
            raise ValueError("model revision must be a lowercase 40-character commit")
        return value

    @field_validator("token_env")
    @classmethod
    def safe_token_env(cls, value: str | None) -> str | None:
        if value is not None and not _ENV_PATTERN.fullmatch(value):
            raise ValueError("token_env must name an uppercase environment variable")
        return value

    @model_validator(mode="after")
    def family_matches_model(self) -> ModelConfig:
        lowered = self.name_or_path.lower()
        if self.family == "qwen3_5" and "qwen3.5" not in lowered:
            raise ValueError("family qwen3_5 requires a Qwen3.5 model name")
        if self.family == "qwen3" and ("qwen3" not in lowered or "qwen3.5" in lowered):
            raise ValueError("family qwen3 requires a Qwen3 model name other than Qwen3.5")
        return self


class DatasetConfig(_StrictModel):
    format: Literal["jsonl"]
    source: DatasetFileConfig
    partition: RuntimeDatasetPartitionConfig
    fields: DatasetFieldsConfig
    cache_dir: NonEmptyString

    @field_validator("cache_dir")
    @classmethod
    def safe_cache_dir(cls, value: str) -> str:
        return _safe_path(value)

    @model_validator(mode="after")
    def partition_is_possible(self) -> DatasetConfig:
        if self.fields.input_sha256 is None:
            raise ValueError("decoder runtime partitioning requires fields.input_sha256")
        size = self.partition.validation_size
        if getattr(size, "kind", None) == "records" and size.value >= self.source.records:
            raise ValueError("validation record count must be smaller than source.records")
        return self


class SequenceConfig(_StrictModel):
    thinking: Literal["disabled"]
    max_prompt_length: PositiveInteger
    max_completion_length: PositiveInteger
    max_sequence_length: PositiveInteger
    overflow: Literal["error"]
    append_eos: Literal[True]
    completion_only_loss: Literal[True]
    packing: bool
    dataset_num_proc: PositiveInteger | None

    @model_validator(mode="after")
    def limits_are_consistent(self) -> SequenceConfig:
        if self.max_prompt_length + self.max_completion_length > self.max_sequence_length:
            raise ValueError(
                "max_prompt_length + max_completion_length exceeds max_sequence_length"
            )
        return self


class LoraConfig(_StrictModel):
    kind: Literal["lora"]
    rank: PositiveInteger
    alpha: PositiveInteger
    dropout: Annotated[float, Field(ge=0.0, lt=1.0)]
    bias: Literal["none"]
    use_rslora: bool
    target_modules: Literal["all-linear"] | list[NonEmptyString]
    modules_to_save: list[NonEmptyString]
    gradient_checkpointing: Literal["unsloth"]

    @field_validator("target_modules", "modules_to_save")
    @classmethod
    def unique_module_lists(cls, value: Any) -> Any:
        if isinstance(value, list) and len(value) != len(set(value)):
            raise ValueError("module lists must not contain duplicates")
        return value


class OptimizationConfig(_StrictModel):
    num_train_epochs: PositiveFloat
    max_steps: int
    per_device_train_batch_size: PositiveInteger
    per_device_eval_batch_size: PositiveInteger
    gradient_accumulation_steps: PositiveInteger
    learning_rate: PositiveFloat
    optimizer: Literal["adamw_torch_fused", "adamw_8bit", "adamw_torch"]
    adam_beta1: OpenUnitFloat
    adam_beta2: OpenUnitFloat
    adam_epsilon: PositiveFloat
    weight_decay: Annotated[float, Field(ge=0.0, allow_inf_nan=False)]
    max_grad_norm: Annotated[float, Field(ge=0.0, allow_inf_nan=False)]
    lr_scheduler_type: Literal["linear", "cosine", "cosine_with_restarts", "constant_with_warmup"]
    warmup_ratio: UnitFloat
    gradient_checkpointing: Literal[True]
    bf16: Literal[True]
    tf32: bool
    full_determinism: bool

    @field_validator("max_steps")
    @classmethod
    def valid_max_steps(cls, value: int) -> int:
        if value != -1 and value <= 0:
            raise ValueError("max_steps must be -1 or positive")
        return value


class RuntimeConfig(_StrictModel):
    require_cuda: Literal[True]
    cuda_allocator_conf: Literal["expandable_segments:True", "backend:cudaMallocAsync"]
    torch_compile: bool
    torch_compile_backend: Literal["inductor"] | None
    torch_compile_mode: Literal["default", "reduce-overhead", "max-autotune"] | None

    @model_validator(mode="after")
    def compile_is_complete(self) -> RuntimeConfig:
        complete = self.torch_compile_backend is not None and self.torch_compile_mode is not None
        partial = (self.torch_compile_backend is None) != (self.torch_compile_mode is None)
        if partial or self.torch_compile != complete:
            raise ValueError("torch compile requires both an explicit backend and mode")
        return self


class DataloaderConfig(_StrictModel):
    num_workers: NonNegativeInteger
    pin_memory: bool
    persistent_workers: bool
    prefetch_factor: PositiveInteger | None
    drop_last: bool

    @model_validator(mode="after")
    def workers_are_consistent(self) -> DataloaderConfig:
        if self.num_workers == 0 and (self.persistent_workers or self.prefetch_factor is not None):
            raise ValueError("persistent workers and prefetching require num_workers > 0")
        return self


class EvaluationConfig(_StrictModel):
    strategy: Literal["no", "steps", "epoch"]
    steps: PositiveInteger | None
    on_start: bool
    run_final: bool
    generation_max_new_tokens: PositiveInteger
    generation_do_sample: Literal[False]
    generation_num_beams: PositiveInteger
    write_predictions: bool
    early_stopping_patience: PositiveInteger | None
    early_stopping_threshold: Annotated[float, Field(ge=0.0, allow_inf_nan=False)] | None

    @model_validator(mode="after")
    def strategy_is_complete(self) -> EvaluationConfig:
        if (self.strategy == "steps") != (self.steps is not None):
            raise ValueError("evaluation.steps is required only for strategy='steps'")
        if self.strategy == "no" and self.on_start:
            raise ValueError("on-start evaluation requires an active strategy")
        if (self.early_stopping_patience is None) != (self.early_stopping_threshold is None):
            raise ValueError("both early-stopping fields must be configured together")
        return self


class CheckpointConfig(_StrictModel):
    strategy: Literal["no", "steps", "epoch"]
    steps: PositiveInteger | None
    total_limit: PositiveInteger | None
    resume_from_checkpoint: NonEmptyString | None
    save_only_model: bool
    load_best_model_at_end: bool
    metric_for_best_model: NonEmptyString | None
    greater_is_better: bool | None

    @model_validator(mode="after")
    def strategy_is_complete(self) -> CheckpointConfig:
        if (self.strategy == "steps") != (self.steps is not None):
            raise ValueError("checkpoint.steps is required only for strategy='steps'")
        if self.save_only_model and self.resume_from_checkpoint is not None:
            raise ValueError("save-only-model checkpoints cannot resume training")
        if self.load_best_model_at_end != (self.metric_for_best_model is not None):
            raise ValueError("best-model loading requires metric_for_best_model")
        if self.load_best_model_at_end != (self.greater_is_better is not None):
            raise ValueError("best-model loading requires greater_is_better")
        return self


class MlflowConfig(_StrictModel):
    tracking_uri: NonEmptyString
    experiment_name: NonEmptyString
    run_name: NonEmptyString
    resume_run_id: NonEmptyString | None
    tags: dict[NonEmptyString, NonEmptyString]


class LoggingConfig(_StrictModel):
    strategy: Literal["steps", "epoch"]
    steps: PositiveInteger | None
    first_step: bool
    mlflow: MlflowConfig
    disable_tqdm: bool

    @model_validator(mode="after")
    def strategy_is_complete(self) -> LoggingConfig:
        if (self.strategy == "steps") != (self.steps is not None):
            raise ValueError("logging.steps is required only for strategy='steps'")
        return self


class RewardConfig(_StrictModel):
    policy: Literal["schema_gated_field_f1_v1"]
    invalid_json: Annotated[float, Field(ge=0.0, le=0.0)]
    invalid_schema: Annotated[float, Field(ge=0.0, le=0.0)]


class RolloutConfig(_StrictModel):
    backend: Literal["unsloth"]
    num_generations: PositiveInteger
    max_completion_length: PositiveInteger
    temperature: Annotated[float, Field(gt=0.0, allow_inf_nan=False)]
    top_p: OpenUnitFloat
    mask_truncated_completions: Literal[True]


class PolicyOptimizationConfig(_StrictModel):
    loss_type: Literal["dapo", "dr_grpo", "grpo"]
    scale_rewards: Literal[False, "group", "batch"]
    beta: Annotated[float, Field(ge=0.0, allow_inf_nan=False)]
    num_iterations: PositiveInteger
    epsilon: OpenUnitFloat

    @model_validator(mode="after")
    def dr_grpo_is_unscaled(self) -> PolicyOptimizationConfig:
        if self.loss_type == "dr_grpo" and self.scale_rewards is not False:
            raise ValueError("dr_grpo requires scale_rewards=false in this experiment contract")
        return self


class GrpoConfig(_StrictModel):
    initialize_from: NonEmptyString
    reward: RewardConfig
    rollout: RolloutConfig
    policy_optimization: PolicyOptimizationConfig

    @field_validator("initialize_from")
    @classmethod
    def safe_checkpoint(cls, value: str) -> str:
        return _safe_path(value)


class DecoderTrainingConfig(_StrictModel):
    schema_version: Literal[1]
    task: NonEmptyString
    method: Literal["sft", "grpo"]
    grpo: GrpoConfig | None
    run: RunConfig
    model: ModelConfig
    prompt: PromptConfig
    task_constraints: TaskConstraintsConfig | None
    dataset: DatasetConfig
    sequence: SequenceConfig
    adapter: LoraConfig
    optimization: OptimizationConfig
    runtime: RuntimeConfig
    dataloader: DataloaderConfig
    evaluation: EvaluationConfig
    checkpoint: CheckpointConfig
    logging: LoggingConfig

    @model_validator(mode="after")
    def cross_section_contract(self) -> DecoderTrainingConfig:
        relation_task = self.task == "bill_of_lading_relation_explicit_v3"
        if relation_task != (self.task_constraints is not None):
            raise ValueError("relation-explicit training requires task_constraints")
        if self.evaluation.generation_max_new_tokens != self.sequence.max_completion_length:
            raise ValueError("evaluation generation limit must equal max_completion_length")
        if self.checkpoint.strategy == "steps" and self.evaluation.strategy == "steps":
            assert self.checkpoint.steps is not None and self.evaluation.steps is not None
            if self.checkpoint.steps % self.evaluation.steps:
                raise ValueError("checkpoint steps must be a multiple of evaluation steps")
        if self.checkpoint.load_best_model_at_end:
            if self.evaluation.strategy == "no":
                raise ValueError("best-model loading requires evaluation")
            if self.checkpoint.strategy != self.evaluation.strategy:
                raise ValueError("best-model checkpoint and evaluation strategies must match")
        if (
            self.evaluation.early_stopping_patience is not None
            and not self.checkpoint.load_best_model_at_end
        ):
            raise ValueError("early stopping requires best-model loading")
        resumed = self.checkpoint.resume_from_checkpoint is not None
        if resumed != (self.logging.mlflow.resume_run_id is not None):
            raise ValueError("checkpoint and MLflow resumption must be configured together")
        if (self.method == "grpo") != (self.grpo is not None):
            raise ValueError("grpo settings are required only when method='grpo'")
        if self.grpo is not None:
            if self.grpo.rollout.max_completion_length != self.sequence.max_completion_length:
                raise ValueError("GRPO rollout limit must equal sequence max_completion_length")
            if self.sequence.packing:
                raise ValueError("GRPO does not support SFT sequence packing")
            generations = self.grpo.rollout.num_generations
            if generations < 2:
                raise ValueError("GRPO requires at least two generations")
            global_batch = (
                self.optimization.per_device_train_batch_size
                * self.optimization.gradient_accumulation_steps
            )
            if global_batch % generations:
                raise ValueError(
                    "GRPO physical batch times accumulation must be divisible by num_generations"
                )
            if (
                self.evaluation.strategy != "no"
                and self.optimization.per_device_eval_batch_size % generations
            ):
                raise ValueError(
                    "single-GPU GRPO eval batch must be divisible by num_generations"
                )
        return self


def parse_decoder_training_config(payload: bytes | str) -> DecoderTrainingConfig:
    if isinstance(payload, bytes):
        try:
            payload = payload.decode("utf-8")
        except UnicodeDecodeError as error:
            raise ValueError("decoder training configuration is not valid UTF-8") from error
    value = yaml.load(payload, Loader=_UniqueKeySafeLoader)
    if not isinstance(value, dict):
        raise ValueError("decoder training configuration root must be a mapping")
    return DecoderTrainingConfig.model_validate(value, strict=True)


def load_decoder_training_config(path: Path) -> DecoderTrainingConfig:
    return parse_decoder_training_config(path.read_bytes())
