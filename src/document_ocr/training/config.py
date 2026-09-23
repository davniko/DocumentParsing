"""Strict YAML configuration for KIE sequence-to-sequence training."""

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

NonEmptyString = Annotated[str, StringConstraints(strip_whitespace=True, min_length=1)]
PositiveInteger = Annotated[int, Field(gt=0)]
NonNegativeInteger = Annotated[int, Field(ge=0)]
PositiveFloat = Annotated[float, Field(gt=0)]
OpenUnitFloat = Annotated[float, Field(ge=0.0, lt=1.0)]
NonNegativeFloat = Annotated[float, Field(ge=0.0)]
FiniteFloat = Annotated[float, Field(allow_inf_nan=False)]

_SHA256_PATTERN = re.compile(r"^[0-9a-f]{64}$")
_REVISION_PATTERN = re.compile(r"^[0-9a-f]{40}$")
_RUN_ID_PATTERN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$")
_FIELD_PATTERN = re.compile(r"^[A-Za-z_][A-Za-z0-9_]{0,127}$")
_ENV_PATTERN = re.compile(r"^[A-Z_][A-Z0-9_]{0,127}$")


class _UniqueKeySafeLoader(yaml.SafeLoader):
    """YAML loader that rejects duplicate keys instead of keeping the last value."""

    def construct_mapping(
        self,
        node: MappingNode,
        deep: bool = False,
    ) -> dict[Any, Any]:
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


def _validate_config_path(value: str) -> str:
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

    @field_validator("run_id")
    @classmethod
    def run_id_is_safe(cls, value: str) -> str:
        if not _RUN_ID_PATTERN.fullmatch(value):
            raise ValueError("run_id must be a filesystem-safe identifier")
        return value

    @field_validator("output_dir")
    @classmethod
    def output_path_is_safe(cls, value: str) -> str:
        return _validate_config_path(value)


class ModelConfig(_StrictModel):
    architecture: Literal["seq2seq_lm"]
    model_type: NonEmptyString
    name_or_path: NonEmptyString
    revision: NonEmptyString
    tokenizer_name_or_path: NonEmptyString
    tokenizer_revision: NonEmptyString
    tokenizer_use_fast: bool
    token_env: NonEmptyString | None
    local_files_only: bool
    trust_remote_code: Literal[False]
    dtype: Literal["bfloat16", "float16", "float32"]
    # T5Gemma 2's merged decoder attention currently supports these two
    # Transformers backends. FlashAttention-2 dispatch is explicitly disabled.
    attention_implementation: Literal["sdpa", "eager"]
    low_cpu_mem_usage: bool
    use_cache: Literal[False]

    @field_validator("revision", "tokenizer_revision")
    @classmethod
    def revisions_are_immutable_commits(cls, value: str) -> str:
        if not _REVISION_PATTERN.fullmatch(value):
            raise ValueError("model and tokenizer revisions must be lowercase 40-character commits")
        return value

    @field_validator("token_env")
    @classmethod
    def token_environment_name_is_safe(cls, value: str | None) -> str | None:
        if value is not None and not _ENV_PATTERN.fullmatch(value):
            raise ValueError("token_env must be an uppercase environment-variable name")
        return value


class PromptConfig(_StrictModel):
    path: NonEmptyString
    placeholder: Literal["{{document_text}}"]
    schema_placeholder: Literal["{{output_schema}}"]

    @field_validator("path")
    @classmethod
    def prompt_path_is_safe(cls, value: str) -> str:
        return _validate_config_path(value)


class TaskConstraintsConfig(_StrictModel):
    path: NonEmptyString
    sha256: NonEmptyString

    @field_validator("path")
    @classmethod
    def constraints_path_is_safe(cls, value: str) -> str:
        return _validate_config_path(value)

    @field_validator("sha256")
    @classmethod
    def digest_is_sha256(cls, value: str) -> str:
        if not _SHA256_PATTERN.fullmatch(value):
            raise ValueError("task-constraints sha256 must be a lowercase 64-character SHA-256")
        return value


class DatasetFileConfig(_StrictModel):
    path: NonEmptyString
    sha256: NonEmptyString
    records: PositiveInteger

    @field_validator("path")
    @classmethod
    def dataset_path_is_safe(cls, value: str) -> str:
        return _validate_config_path(value)

    @field_validator("sha256")
    @classmethod
    def digest_is_sha256(cls, value: str) -> str:
        if not _SHA256_PATTERN.fullmatch(value):
            raise ValueError("dataset sha256 must be a lowercase 64-character SHA-256")
        return value


class PinnedArtifactFileConfig(_StrictModel):
    path: NonEmptyString
    sha256: NonEmptyString

    @field_validator("path")
    @classmethod
    def artifact_path_is_safe(cls, value: str) -> str:
        return _validate_config_path(value)

    @field_validator("sha256")
    @classmethod
    def artifact_digest_is_sha256(cls, value: str) -> str:
        if not _SHA256_PATTERN.fullmatch(value):
            raise ValueError("artifact sha256 must be a lowercase 64-character SHA-256")
        return value


class RealV5ValidationContractConfig(_StrictModel):
    records: PositiveInteger
    document_ids_sha256: NonEmptyString

    @field_validator("document_ids_sha256")
    @classmethod
    def validation_ids_digest_is_sha256(cls, value: str) -> str:
        if not _SHA256_PATTERN.fullmatch(value):
            raise ValueError("validation document_ids_sha256 must be lowercase SHA-256")
        return value


class RealV5EquipmentCorrectionConfig(_StrictModel):
    document_id: NonEmptyString
    input_sha256: NonEmptyString
    evidence_text: NonEmptyString
    container_index: NonNegativeInteger
    type_description: NonEmptyString

    @field_validator("input_sha256")
    @classmethod
    def correction_input_digest_is_sha256(cls, value: str) -> str:
        if not _SHA256_PATTERN.fullmatch(value):
            raise ValueError("correction input_sha256 must be lowercase SHA-256")
        return value


class RealV5ProjectionConfig(_StrictModel):
    schema_version: Literal[1]
    projection: Literal["relation_v3_real_split_to_v5_v1"]
    source: DatasetFileConfig
    baseline_dataset_report: PinnedArtifactFileConfig
    validation: RealV5ValidationContractConfig
    corrections: list[RealV5EquipmentCorrectionConfig]
    output_dir: NonEmptyString

    @field_validator("output_dir")
    @classmethod
    def projection_output_path_is_safe(cls, value: str) -> str:
        return _validate_config_path(value)

    @model_validator(mode="after")
    def correction_documents_are_unique(self) -> RealV5ProjectionConfig:
        document_ids = [item.document_id for item in self.corrections]
        if len(document_ids) != len(set(document_ids)):
            raise ValueError("correction document IDs must be unique")
        return self


class RelationConstraintsBuildConfig(_StrictModel):
    schema_version: Literal[1]
    task: Literal[
        "bill_of_lading_relation_explicit_v3",
        "bill_of_lading_relation_explicit_v4",
        "bill_of_lading_relation_explicit_v5",
    ]
    sources: list[DatasetFileConfig] = Field(min_length=1)
    target_field: NonEmptyString
    package_registry: PinnedArtifactFileConfig
    container_registry: PinnedArtifactFileConfig
    output_dir: NonEmptyString

    @field_validator("target_field")
    @classmethod
    def target_field_is_top_level_identifier(cls, value: str) -> str:
        if not _FIELD_PATTERN.fullmatch(value):
            raise ValueError("target_field must be a top-level identifier")
        return value

    @field_validator("output_dir")
    @classmethod
    def constraints_output_path_is_safe(cls, value: str) -> str:
        return _validate_config_path(value)

    @model_validator(mode="after")
    def source_paths_are_unique(self) -> RelationConstraintsBuildConfig:
        paths = [source.path for source in self.sources]
        if len(paths) != len(set(paths)):
            raise ValueError("relation-constraints source paths must be unique")
        return self


class DatasetPartitionFieldsConfig(_StrictModel):
    document_id: NonEmptyString
    input_sha256: NonEmptyString
    target: NonEmptyString

    @field_validator("document_id", "input_sha256", "target")
    @classmethod
    def field_names_are_top_level_identifiers(cls, value: str) -> str:
        if not _FIELD_PATTERN.fullmatch(value):
            raise ValueError("dataset field names must be top-level identifiers")
        return value

    @model_validator(mode="after")
    def fields_are_distinct(self) -> DatasetPartitionFieldsConfig:
        values = (self.document_id, self.input_sha256, self.target)
        if len(set(values)) != len(values):
            raise ValueError("dataset partition field names must be distinct")
        return self


class DatasetPartitionConfig(_StrictModel):
    schema_version: Literal[1]
    source: DatasetFileConfig
    fields: DatasetPartitionFieldsConfig
    algorithm: Literal["seeded_sha256_rank_v1"]
    seed: NonNegativeInteger
    validation_records: PositiveInteger
    coverage_policy: Literal["retain_each_target_leaf_in_train"]
    output_dir: NonEmptyString

    @field_validator("output_dir")
    @classmethod
    def output_path_is_safe(cls, value: str) -> str:
        return _validate_config_path(value)

    @model_validator(mode="after")
    def validation_is_smaller_than_source(self) -> DatasetPartitionConfig:
        if self.validation_records >= self.source.records:
            raise ValueError("validation_records must be smaller than source.records")
        return self


class ValidationRecordsConfig(_StrictModel):
    kind: Literal["records"]
    value: PositiveInteger


class ValidationFractionConfig(_StrictModel):
    kind: Literal["fraction"]
    value: Annotated[float, Field(gt=0.0, lt=1.0)]
    rounding: Literal["half_up"]


ValidationSizeConfig = Annotated[
    ValidationRecordsConfig | ValidationFractionConfig,
    Field(discriminator="kind"),
]


class RuntimeDatasetPartitionConfig(_StrictModel):
    algorithm: Literal["seeded_sha256_rank_v1"]
    seed: NonNegativeInteger
    validation_size: ValidationSizeConfig
    coverage_policy: Literal["retain_each_target_leaf_in_train"]


class DatasetSplitsConfig(_StrictModel):
    train: list[DatasetFileConfig] = Field(min_length=1)
    validation: list[DatasetFileConfig]
    test: list[DatasetFileConfig]

    @model_validator(mode="after")
    def files_are_unique_across_splits(self) -> DatasetSplitsConfig:
        paths = [item.path for split in (self.train, self.validation, self.test) for item in split]
        if len(paths) != len(set(paths)):
            raise ValueError("each dataset file must belong to exactly one split")
        return self


class DatasetFieldsConfig(_StrictModel):
    document_id: NonEmptyString
    input_text: NonEmptyString
    target: NonEmptyString
    input_sha256: NonEmptyString | None

    @field_validator("document_id", "input_text", "target", "input_sha256")
    @classmethod
    def field_names_are_top_level_identifiers(cls, value: str | None) -> str | None:
        if value is not None and not _FIELD_PATTERN.fullmatch(value):
            raise ValueError("dataset field names must be top-level identifiers")
        return value

    @model_validator(mode="after")
    def fields_are_distinct(self) -> DatasetFieldsConfig:
        values = [self.document_id, self.input_text, self.target]
        if self.input_sha256 is not None:
            values.append(self.input_sha256)
        if len(values) != len(set(values)):
            raise ValueError("dataset field names must be distinct")
        return self


class PreprocessingConfig(_StrictModel):
    max_source_length: PositiveInteger
    max_target_length: PositiveInteger
    source_overflow: Literal["error", "truncate"]
    source_add_special_tokens: bool
    num_proc: PositiveInteger | None
    batch_size: PositiveInteger
    load_from_cache_file: bool
    cache_dir: NonEmptyString
    pad_to_multiple_of: PositiveInteger

    @field_validator("cache_dir")
    @classmethod
    def cache_path_is_safe(cls, value: str) -> str:
        return _validate_config_path(value)


class DatasetConfig(_StrictModel):
    format: Literal["jsonl"]
    splits: DatasetSplitsConfig | None = None
    source: DatasetFileConfig | None = None
    partition: RuntimeDatasetPartitionConfig | None = None
    fields: DatasetFieldsConfig
    preprocessing: PreprocessingConfig

    @model_validator(mode="after")
    def exactly_one_dataset_input_mode(self) -> DatasetConfig:
        has_pre_split = self.splits is not None
        has_runtime_source = self.source is not None
        has_runtime_partition = self.partition is not None
        has_complete_runtime_partition = has_runtime_source and has_runtime_partition
        if has_runtime_source != has_runtime_partition:
            raise ValueError(
                "runtime dataset partitioning requires both source and partition"
            )
        if has_pre_split == has_complete_runtime_partition:
            raise ValueError(
                "dataset must configure exactly one input mode: either splits, or both "
                "source and partition"
            )
        if has_complete_runtime_partition:
            if self.source is None or self.partition is None:
                raise AssertionError("complete runtime partition failed to narrow")
            if self.fields.input_sha256 is None:
                raise ValueError(
                    "runtime dataset partitioning requires fields.input_sha256"
                )
            validation_size = self.partition.validation_size
            if (
                isinstance(validation_size, ValidationRecordsConfig)
                and validation_size.value >= self.source.records
            ):
                raise ValueError(
                    "record-based validation size must be smaller than source.records"
                )
        return self

    @property
    def input_mode(self) -> Literal["pre_split", "runtime_partition"]:
        return "pre_split" if self.splits is not None else "runtime_partition"


class EvaInitializationConfig(_StrictModel):
    """Fixed-rank EVA calibration; redistribution is deliberately not enabled."""

    rho: Annotated[float, Field(ge=1.0, le=1.0)]
    tau: Annotated[float, Field(gt=0.0, le=1.0)]
    whiten: bool
    sample_count: PositiveInteger
    batch_size: PositiveInteger
    tokens_per_stream: PositiveInteger
    max_forward_passes: Annotated[int, Field(ge=3)]
    seed: NonNegativeInteger


class LoraConfig(_StrictModel):
    method: Literal["lora"]
    adapter_name: NonEmptyString
    rank: PositiveInteger
    alpha: PositiveInteger
    dropout: Annotated[float, Field(ge=0.0, lt=1.0)]
    bias: Literal["none", "all", "lora_only"]
    use_rslora: bool
    init_lora_weights: bool | Literal["gaussian", "olora", "pissa", "eva"]
    eva: EvaInitializationConfig | None = None
    target_modules_regex: NonEmptyString
    modules_to_save: list[NonEmptyString]
    ensure_weight_tying: bool

    @model_validator(mode="after")
    def eva_contract_is_explicit(self) -> LoraConfig:
        if (self.init_lora_weights == "eva") != (self.eva is not None):
            raise ValueError(
                "EVA initialization requires an explicit peft.eva block, "
                "and only EVA may configure it"
            )
        if self.eva is not None:
            if self.eva.tokens_per_stream < self.rank:
                raise ValueError("EVA tokens_per_stream must be at least the adapter rank")
            if self.eva.batch_size > self.eva.sample_count:
                raise ValueError("EVA batch_size must not exceed sample_count")
        return self

    @field_validator("target_modules_regex")
    @classmethod
    def target_regex_compiles(cls, value: str) -> str:
        try:
            re.compile(value)
        except re.error as error:
            raise ValueError("target_modules_regex is not a valid regular expression") from error
        return value

    @field_validator("modules_to_save")
    @classmethod
    def modules_to_save_are_unique(cls, value: list[str]) -> list[str]:
        if len(value) != len(set(value)):
            raise ValueError("modules_to_save must be unique")
        return value


class OptimizationConfig(_StrictModel):
    num_train_epochs: PositiveFloat
    max_steps: int
    per_device_train_batch_size: PositiveInteger
    gradient_accumulation_steps: PositiveInteger
    learning_rate: PositiveFloat
    optimizer: Literal["adamw_torch_fused", "adamw_torch", "adafactor"]
    optimizer_args: NonEmptyString | None
    adam_beta1: OpenUnitFloat
    adam_beta2: OpenUnitFloat
    adam_epsilon: PositiveFloat
    lr_scheduler_type: Literal[
        "linear",
        "cosine",
        "cosine_with_restarts",
        "constant",
        "constant_with_warmup",
        "polynomial",
        "inverse_sqrt",
    ]
    lr_scheduler_kwargs: dict[NonEmptyString, int | float | str]
    warmup_ratio: OpenUnitFloat
    weight_decay: NonNegativeFloat
    max_grad_norm: NonNegativeFloat
    label_smoothing_factor: OpenUnitFloat
    average_tokens_across_devices: bool
    seed: NonNegativeInteger
    data_seed: NonNegativeInteger
    full_determinism: bool
    train_sampling_strategy: Literal["random", "group_by_length"]
    length_column_name: Literal["input_length"]

    @field_validator("max_steps")
    @classmethod
    def max_steps_is_disabled_or_positive(cls, value: int) -> int:
        if value != -1 and value <= 0:
            raise ValueError("max_steps must be -1 or a positive integer")
        return value


class RuntimeConfig(_StrictModel):
    distributed_processes: Literal[1]
    require_cuda: bool
    cuda_allocator_conf: Literal[
        "expandable_segments:True",
        "backend:cudaMallocAsync",
    ]
    bf16: bool
    fp16: bool
    tf32: bool
    gradient_checkpointing: bool
    gradient_checkpointing_use_reentrant: bool
    torch_compile: bool
    torch_compile_backend: Literal["inductor"] | None
    torch_compile_mode: Literal["default", "reduce-overhead", "max-autotune"] | None
    torch_empty_cache_steps: PositiveInteger | None
    skip_memory_metrics: bool

    @model_validator(mode="after")
    def precision_and_compile_are_consistent(self) -> RuntimeConfig:
        if self.bf16 and self.fp16:
            raise ValueError("bf16 and fp16 are mutually exclusive")
        if self.torch_compile:
            if self.torch_compile_backend is None or self.torch_compile_mode is None:
                raise ValueError("compiled training requires an explicit backend and mode")
        elif self.torch_compile_backend is not None or self.torch_compile_mode is not None:
            raise ValueError("compile backend/mode must be null when torch_compile is false")
        return self


class DataloaderConfig(_StrictModel):
    num_workers: NonNegativeInteger
    pin_memory: bool
    persistent_workers: bool
    prefetch_factor: PositiveInteger | None
    drop_last: bool

    @model_validator(mode="after")
    def worker_options_require_workers(self) -> DataloaderConfig:
        if self.num_workers == 0 and (self.persistent_workers or self.prefetch_factor is not None):
            raise ValueError("persistent workers and prefetching require num_workers > 0")
        return self


class EvaluationConfig(_StrictModel):
    strategy: Literal["no", "steps", "epoch"]
    split: Literal["train", "validation"]
    steps: PositiveInteger | None
    per_device_batch_size: PositiveInteger
    on_start: bool
    predict_with_generate: bool
    generation_do_sample: Literal[False]
    generation_max_length: PositiveInteger
    generation_num_beams: PositiveInteger
    generation_repetition_penalty: PositiveFloat
    generation_no_repeat_ngram_size: NonNegativeInteger
    generation_length_penalty: FiniteFloat
    generation_early_stopping: bool | Literal["never"]
    accumulation_steps: PositiveInteger | None
    early_stopping_patience: PositiveInteger | None
    early_stopping_threshold: NonNegativeFloat | None
    run_final_evaluation: bool
    write_predictions_for: list[Literal["validation", "test"]]

    @field_validator("write_predictions_for")
    @classmethod
    def prediction_splits_are_unique(
        cls, value: list[Literal["validation", "test"]]
    ) -> list[Literal["validation", "test"]]:
        if len(value) != len(set(value)):
            raise ValueError("write_predictions_for must contain unique split names")
        return value

    @model_validator(mode="after")
    def evaluation_steps_match_strategy(self) -> EvaluationConfig:
        if self.strategy == "steps" and self.steps is None:
            raise ValueError("steps evaluation requires evaluation.steps")
        if self.strategy != "steps" and self.steps is not None:
            raise ValueError("evaluation.steps is only valid for strategy='steps'")
        if self.generation_num_beams == 1 and self.generation_early_stopping is not False:
            raise ValueError("generation early stopping requires beam search")
        if (self.early_stopping_patience is None) != (
            self.early_stopping_threshold is None
        ):
            raise ValueError(
                "early_stopping_patience and early_stopping_threshold must be set together"
            )
        return self


class CheckpointConfig(_StrictModel):
    strategy: Literal["no", "steps", "epoch", "best"]
    steps: PositiveInteger | None
    total_limit: PositiveInteger | None
    load_best_model_at_end: bool
    metric_for_best_model: NonEmptyString | None
    greater_is_better: bool | None
    resume_from_checkpoint: NonEmptyString | None
    allow_resume_source_code_drift: bool = False
    enable_jit_checkpoint: bool
    save_only_model: bool
    save_on_each_node: bool
    restore_callback_states_from_checkpoint: bool

    @model_validator(mode="after")
    def checkpoint_steps_match_strategy(self) -> CheckpointConfig:
        if self.strategy == "steps" and self.steps is None:
            raise ValueError("step checkpointing requires checkpoint.steps")
        if self.strategy != "steps" and self.steps is not None:
            raise ValueError("checkpoint.steps is only valid for strategy='steps'")
        if self.load_best_model_at_end and self.metric_for_best_model is None:
            raise ValueError("best-model loading requires metric_for_best_model")
        if not self.load_best_model_at_end and (
            self.metric_for_best_model is not None or self.greater_is_better is not None
        ):
            raise ValueError("best-model fields require load_best_model_at_end=true")
        return self


class MlflowLoggingConfig(_StrictModel):
    tracking_uri: NonEmptyString
    experiment_name: NonEmptyString
    resume_run_id: NonEmptyString | None
    log_artifacts: bool
    flatten_params: bool
    max_log_params: PositiveInteger | None
    system_metrics: bool
    system_metrics_sampling_interval: PositiveInteger
    system_metrics_samples_before_logging: PositiveInteger
    tags: dict[NonEmptyString, NonEmptyString]


class LoggingConfig(_StrictModel):
    strategy: Literal["steps", "epoch"]
    steps: PositiveInteger | None
    first_step: bool
    report_to: list[Literal["mlflow"]] = Field(min_length=1, max_length=1)
    mlflow: MlflowLoggingConfig
    structured_log: NonEmptyString
    disable_tqdm: bool
    log_level: Literal["debug", "info", "warning", "error", "passive"]
    include_num_input_tokens_seen: Literal["no", "all", "non_padding"]
    filter_nan_inf: bool
    on_each_node: bool

    @field_validator("structured_log")
    @classmethod
    def structured_log_path_is_safe(cls, value: str) -> str:
        validated = _validate_config_path(value)
        if PurePath(validated).is_absolute():
            raise ValueError("training log paths must be relative to the run directory")
        return validated

    @field_validator("report_to")
    @classmethod
    def reporters_are_unique(
        cls, value: list[Literal["mlflow"]]
    ) -> list[Literal["mlflow"]]:
        if len(value) != len(set(value)):
            raise ValueError("report_to must contain unique integrations")
        return value

    @model_validator(mode="after")
    def logging_steps_match_strategy(self) -> LoggingConfig:
        if self.strategy == "steps" and self.steps is None:
            raise ValueError("step logging requires logging.steps")
        if self.strategy != "steps" and self.steps is not None:
            raise ValueError("logging.steps is only valid for strategy='steps'")
        return self


class TrainingConfig(_StrictModel):
    schema_version: Literal[1]
    task: NonEmptyString
    objective: Literal["seq2seq_teacher_forcing"]
    run: RunConfig
    model: ModelConfig
    prompt: PromptConfig
    task_constraints: TaskConstraintsConfig | None = None
    dataset: DatasetConfig
    peft: LoraConfig
    optimization: OptimizationConfig
    runtime: RuntimeConfig
    dataloader: DataloaderConfig
    evaluation: EvaluationConfig
    checkpoint: CheckpointConfig
    logging: LoggingConfig

    @model_validator(mode="after")
    def strategies_and_splits_are_consistent(self) -> TrainingConfig:
        relation_explicit = self.task in {
            "bill_of_lading_relation_explicit_v3",
            "bill_of_lading_relation_explicit_v4",
            "bill_of_lading_relation_explicit_v5",
        }
        if relation_explicit != (self.task_constraints is not None):
            raise ValueError(
                "each relation-explicit bill-of-lading task requires a frozen task_constraints "
                "artifact, and other registered tasks must not configure it"
            )
        if self.dataset.splits is None:
            has_validation = True
            has_test = False
        else:
            has_validation = bool(self.dataset.splits.validation)
            has_test = bool(self.dataset.splits.test)
        evaluation_is_active = (
            self.evaluation.strategy != "no" or self.evaluation.run_final_evaluation
        )
        if (
            evaluation_is_active
            and self.evaluation.split == "validation"
            and not has_validation
        ):
            raise ValueError("evaluation split 'validation' requires at least one validation file")
        if self.evaluation.strategy == "no" and self.evaluation.on_start:
            raise ValueError("eval_on_start requires an evaluation strategy")
        for split in self.evaluation.write_predictions_for:
            if split == "validation" and not has_validation:
                raise ValueError("validation predictions require a validation split")
            if split == "test" and not has_test:
                raise ValueError("test predictions require a test split")
        if self.evaluation.write_predictions_for and self.dataloader.drop_last:
            raise ValueError("prediction publication requires dataloader.drop_last=false")
        if self.evaluation.generation_max_length > self.dataset.preprocessing.max_target_length:
            raise ValueError(
                "generation_max_length must not exceed preprocessing.max_target_length"
            )
        if (
            self.evaluation.strategy != "no"
            or self.evaluation.run_final_evaluation
            or self.evaluation.write_predictions_for
        ) and not self.evaluation.predict_with_generate:
            raise ValueError("structured evaluation and prediction require generation")
        if (
            self.evaluation.predict_with_generate
            and not self.runtime.skip_memory_metrics
        ):
            raise ValueError(
                "generated evaluation requires runtime.skip_memory_metrics=true; "
                "the Transformers detailed memory tracker busy-polls host memory and "
                "starves autoregressive decoding"
            )
        if self.checkpoint.load_best_model_at_end:
            if self.evaluation.strategy == "no":
                raise ValueError("best-model loading requires evaluation")
            if self.checkpoint.strategy not in {self.evaluation.strategy, "best"}:
                raise ValueError("save/evaluation strategies must match for best-model loading")
            if (
                self.checkpoint.strategy == "steps"
                and self.checkpoint.steps is not None
                and self.evaluation.steps is not None
                and self.checkpoint.steps % self.evaluation.steps != 0
            ):
                raise ValueError("checkpoint.steps must be a multiple of evaluation.steps")
        if (
            self.evaluation.early_stopping_patience is not None
            and not self.checkpoint.load_best_model_at_end
        ):
            raise ValueError("early stopping requires load_best_model_at_end=true")
        if self.checkpoint.strategy == "best" and self.evaluation.strategy == "no":
            raise ValueError("best checkpointing requires evaluation")
        if self.checkpoint.save_only_model and self.checkpoint.resume_from_checkpoint is not None:
            raise ValueError("save_only_model checkpoints cannot be used for training resumption")
        has_checkpoint_resume = self.checkpoint.resume_from_checkpoint is not None
        has_mlflow_resume = self.logging.mlflow.resume_run_id is not None
        if has_checkpoint_resume != has_mlflow_resume:
            raise ValueError(
                "checkpoint resume and MLflow resume_run_id must be configured together"
            )
        if self.checkpoint.allow_resume_source_code_drift and not has_checkpoint_resume:
            raise ValueError(
                "allow_resume_source_code_drift requires an explicit resume checkpoint"
            )
        if self.model.dtype == "bfloat16" and not self.runtime.bf16:
            raise ValueError("bfloat16 model loading requires runtime.bf16=true")
        if self.model.dtype == "float16" and not self.runtime.fp16:
            raise ValueError("float16 model loading requires runtime.fp16=true")
        if self.model.dtype == "float32" and (self.runtime.bf16 or self.runtime.fp16):
            raise ValueError("float32 model loading is incompatible with mixed-precision flags")
        return self


def _parse_yaml_mapping(payload: bytes | str, description: str) -> dict[str, Any]:
    if isinstance(payload, bytes):
        try:
            raw = payload.decode("utf-8")
        except UnicodeDecodeError as error:
            raise ValueError(f"{description} is not valid UTF-8") from error
    else:
        raw = payload
    value = yaml.load(raw, Loader=_UniqueKeySafeLoader)
    if not isinstance(value, dict):
        raise ValueError(f"{description} root must be a mapping")
    return value


def parse_training_config(payload: bytes | str) -> TrainingConfig:
    """Parse one strict training YAML payload without importing the GPU stack."""

    return TrainingConfig.model_validate(
        _parse_yaml_mapping(payload, "training configuration"), strict=True
    )


def load_training_config(path: Path) -> TrainingConfig:
    """Load one strict training YAML without importing the GPU training stack."""

    return parse_training_config(path.read_bytes())


def load_real_v5_projection_config(path: Path) -> RealV5ProjectionConfig:
    """Load one strict real-data relation-v5 projection declaration."""

    return RealV5ProjectionConfig.model_validate(
        _parse_yaml_mapping(path.read_bytes(), "real v5 projection configuration"),
        strict=True,
    )


def load_relation_constraints_build_config(path: Path) -> RelationConstraintsBuildConfig:
    """Load a strict relation-explicit vocabulary publication declaration."""

    return RelationConstraintsBuildConfig.model_validate(
        _parse_yaml_mapping(path.read_bytes(), "relation constraints build configuration"),
        strict=True,
    )


def parse_dataset_partition_config(payload: bytes | str) -> DatasetPartitionConfig:
    """Parse one strict pre-training dataset partition declaration."""

    return DatasetPartitionConfig.model_validate(
        _parse_yaml_mapping(payload, "dataset partition configuration"), strict=True
    )


def load_dataset_partition_config(path: Path) -> DatasetPartitionConfig:
    """Load one strict pre-training dataset partition declaration."""

    return parse_dataset_partition_config(path.read_bytes())


def resolve_config_path(project_root: Path, value: str) -> Path:
    """Resolve a validated config path relative to the explicit project root."""

    path = Path(value)
    return path if path.is_absolute() else project_root / path
