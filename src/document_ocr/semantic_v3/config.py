"""Strict configuration for reversible semantic-v3 dataset transforms."""

from __future__ import annotations

import os
import re
from pathlib import Path
from typing import Annotated, Literal

from pydantic import ConfigDict, Field, StringConstraints, field_validator, model_validator

from document_ocr.config import load_strict_yaml_mapping
from document_ocr.label_schemas.common import LabelSchemaModel

NonEmptyString = Annotated[str, StringConstraints(strip_whitespace=True, min_length=1)]
PositiveInteger = Annotated[int, Field(gt=0)]
Sha256 = Annotated[str, StringConstraints(pattern=r"^[0-9a-f]{64}$")]
DocumentId = Annotated[str, StringConstraints(pattern=r"^doc_[0-9a-f]{64}$")]
_SAFE_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]*$")


class _ConfigModel(LabelSchemaModel):
    model_config = ConfigDict(extra="forbid", frozen=True, strict=True, allow_inf_nan=False)


def _absolute_safe_path(value: str, field_name: str, *, directory: bool) -> str:
    path = Path(value)
    if not path.is_absolute():
        raise ValueError(f"{field_name} must be absolute")
    normalized = Path(os.path.abspath(value))
    if directory and normalized == Path(normalized.anchor):
        raise ValueError(f"{field_name} must not be the filesystem root")
    return value


class SemanticV3SourceFile(_ConfigModel):
    id: Annotated[str, StringConstraints(pattern=r"^[a-z][a-z0-9_-]*$")]
    split: Literal["train", "validation", "test"]
    path: NonEmptyString
    sha256: Sha256
    records: PositiveInteger

    @field_validator("path")
    @classmethod
    def path_is_absolute_jsonl(cls, value: str) -> str:
        _absolute_safe_path(value, "source path", directory=False)
        if Path(value).suffix != ".jsonl":
            raise ValueError("source path must end in .jsonl")
        return value


class SemanticV3SourceConfig(_ConfigModel):
    expected_documents: PositiveInteger
    files: list[SemanticV3SourceFile] = Field(min_length=1)

    @model_validator(mode="after")
    def inputs_are_unique_and_counts_match(self) -> SemanticV3SourceConfig:
        ids = [row.id for row in self.files]
        paths = [row.path for row in self.files]
        if len(ids) != len(set(ids)) or len(paths) != len(set(paths)):
            raise ValueError("semantic-v3 input IDs and paths must be unique")
        if sum(row.records for row in self.files) != self.expected_documents:
            raise ValueError("semantic-v3 input counts must sum to expected_documents")
        return self


class FrozenArtifactInput(_ConfigModel):
    path: NonEmptyString
    sha256: Sha256

    @field_validator("path")
    @classmethod
    def path_is_absolute_json(cls, value: str) -> str:
        _absolute_safe_path(value, "frozen artifact path", directory=False)
        if Path(value).suffix != ".json":
            raise ValueError("frozen artifact path must end in .json")
        return value


class SemanticV3RegistryInputs(_ConfigModel):
    package_categories: FrozenArtifactInput | None
    container_categories: FrozenArtifactInput | None
    category_assignments: FrozenArtifactInput | None
    relation_assignments: FrozenArtifactInput | None


class RelationExclusion(_ConfigModel):
    document_id: DocumentId
    reason: Literal[
        "ambiguous_duplicate_package_quantity",
        "incomplete_container_allocation_quantities",
    ]


class SemanticV3OutputConfig(_ConfigModel):
    audit_root: NonEmptyString
    dataset_root: NonEmptyString
    dataset_id: NonEmptyString

    @field_validator("audit_root", "dataset_root")
    @classmethod
    def roots_are_absolute_and_safe(cls, value: str) -> str:
        return _absolute_safe_path(value, "semantic-v3 output root", directory=True)

    @field_validator("dataset_id")
    @classmethod
    def dataset_id_is_safe(cls, value: str) -> str:
        if _SAFE_ID.fullmatch(value) is None:
            raise ValueError("dataset_id contains unsupported characters")
        return value


class TrainingConfigSource(_ConfigModel):
    project_root: NonEmptyString
    baseline_path: NonEmptyString
    baseline_sha256: Sha256
    prompt_path: NonEmptyString
    prompt_sha256: Sha256

    @field_validator("project_root")
    @classmethod
    def project_root_is_absolute_and_safe(cls, value: str) -> str:
        return _absolute_safe_path(value, "training project root", directory=True)

    @field_validator("baseline_path", "prompt_path")
    @classmethod
    def inputs_are_absolute_files(cls, value: str) -> str:
        return _absolute_safe_path(value, "training publication input", directory=False)

    @model_validator(mode="after")
    def suffixes_are_expected(self) -> TrainingConfigSource:
        if Path(self.baseline_path).suffix not in {".yaml", ".yml"}:
            raise ValueError("training baseline path must be YAML")
        if Path(self.prompt_path).suffix != ".txt":
            raise ValueError("training prompt path must end in .txt")
        return self


class TrainingTokenizerAuditSource(_ConfigModel):
    path: NonEmptyString
    tokenizer_json_sha256: Sha256
    tokenizer_config_sha256: Sha256

    @field_validator("path")
    @classmethod
    def path_is_absolute_and_safe(cls, value: str) -> str:
        return _absolute_safe_path(value, "tokenizer audit path", directory=True)


class TrainingCapacityConfig(_ConfigModel):
    max_source_length: PositiveInteger
    max_target_length: PositiveInteger
    per_device_train_batch_size: PositiveInteger
    gradient_accumulation_steps: PositiveInteger
    per_device_eval_batch_size: PositiveInteger

    @model_validator(mode="after")
    def lengths_are_aligned(self) -> TrainingCapacityConfig:
        if self.max_source_length % 256 or self.max_target_length % 256:
            raise ValueError("training token ceilings must be multiples of 256")
        return self


class TrainingConfigPublication(_ConfigModel):
    source: TrainingConfigSource
    tokenizer_audit: TrainingTokenizerAuditSource | None = None
    capacity: TrainingCapacityConfig
    output_filename: Literal["training.yaml"]
    run_id: NonEmptyString
    adapter_name: NonEmptyString
    evaluation_every_epochs: PositiveInteger

    @field_validator("run_id", "adapter_name")
    @classmethod
    def identifiers_are_safe(cls, value: str) -> str:
        if _SAFE_ID.fullmatch(value) is None:
            raise ValueError("training run and adapter names must be filesystem safe")
        return value


class SemanticV3TransformConfig(_ConfigModel):
    schema_version: Literal[1]
    transform: Literal["mpci_bl_semantic_v2_to_relation_explicit_v3"]
    source: SemanticV3SourceConfig
    registries: SemanticV3RegistryInputs
    allowed_relation_exclusions: list[RelationExclusion]
    output: SemanticV3OutputConfig
    training_config: TrainingConfigPublication | None = None

    @model_validator(mode="after")
    def exclusions_are_unique(self) -> SemanticV3TransformConfig:
        values = [row.document_id for row in self.allowed_relation_exclusions]
        if len(values) != len(set(values)):
            raise ValueError("allowed relation exclusions must have unique document IDs")
        return self


def load_semantic_v3_config(path: str | Path) -> SemanticV3TransformConfig:
    return SemanticV3TransformConfig.model_validate(load_strict_yaml_mapping(path), strict=True)
