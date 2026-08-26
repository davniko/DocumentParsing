"""Strict configuration for page-aligned GLM-OCR table recognition."""

from __future__ import annotations

import os
import re
from pathlib import Path, PurePosixPath
from typing import Annotated, Literal

from pydantic import ConfigDict, Field, StringConstraints, field_validator, model_validator

from document_ocr.config import TableVllmConfig, load_strict_yaml_mapping
from document_ocr.label_schemas.common import LabelSchemaModel

NonEmptyString = Annotated[str, StringConstraints(strip_whitespace=True, min_length=1)]
PositiveInteger = Annotated[int, Field(gt=0)]
Sha256 = Annotated[str, StringConstraints(pattern=r"^[0-9a-f]{64}$")]
_RUN_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]*$")


class _ConfigModel(LabelSchemaModel):
    model_config = ConfigDict(extra="forbid", frozen=True, strict=True, allow_inf_nan=False)


def _absolute_safe_directory(value: str, field_name: str) -> str:
    path = Path(value)
    if not path.is_absolute():
        raise ValueError(f"{field_name} must be absolute")
    normalized = Path(os.path.abspath(value))
    if normalized == Path(normalized.anchor):
        raise ValueError(f"{field_name} must not be the filesystem root")
    return value


def _safe_relative_file(value: str, suffix: str, field_name: str) -> str:
    path = PurePosixPath(value)
    if path.is_absolute() or any(part in {"", ".", ".."} for part in path.parts):
        raise ValueError(f"{field_name} must be a safe relative POSIX path")
    if path.suffix != suffix:
        raise ValueError(f"{field_name} must end in {suffix}")
    return value


class TableRecordSetConfig(_ConfigModel):
    id: Annotated[str, StringConstraints(pattern=r"^[a-z][a-z0-9_-]*$")]
    root: NonEmptyString
    records_path: NonEmptyString
    records_sha256: Sha256
    records: PositiveInteger

    @field_validator("root")
    @classmethod
    def root_is_absolute(cls, value: str) -> str:
        return _absolute_safe_directory(value, "record-set root")

    @field_validator("records_path")
    @classmethod
    def records_path_is_safe(cls, value: str) -> str:
        return _safe_relative_file(value, ".jsonl", "records_path")


class ExtractionRunConfig(_ConfigModel):
    run_id: NonEmptyString
    root: NonEmptyString

    @field_validator("root")
    @classmethod
    def root_is_absolute(cls, value: str) -> str:
        return _absolute_safe_directory(value, "extraction-run root")


class ValidatedLabelRasterSourceConfig(_ConfigModel):
    type: Literal["validated_label_rasters"]
    expected_documents: PositiveInteger
    expected_pages: PositiveInteger
    record_sets: list[TableRecordSetConfig] = Field(min_length=1)
    extraction_runs: list[ExtractionRunConfig] = Field(min_length=1)

    @model_validator(mode="after")
    def identities_are_unique(self) -> ValidatedLabelRasterSourceConfig:
        record_ids = [row.id for row in self.record_sets]
        run_ids = [row.run_id for row in self.extraction_runs]
        if len(record_ids) != len(set(record_ids)):
            raise ValueError("record-set ids must be unique")
        if len(run_ids) != len(set(run_ids)):
            raise ValueError("extraction-run ids must be unique")
        if sum(row.records for row in self.record_sets) != self.expected_documents:
            raise ValueError("record-set counts must sum to expected_documents")
        return self


class RawExtractionRunConfig(_ConfigModel):
    id: Annotated[str, StringConstraints(pattern=r"^[a-z][a-z0-9_-]*$")]
    run_id: NonEmptyString
    root: NonEmptyString
    config_sha256: Sha256
    pipeline_fingerprint: Sha256
    inventory_sha256: Sha256
    provenance_sha256: Sha256 | None
    manifest_sha256: Sha256 | None
    expected_documents: PositiveInteger
    expected_pages: PositiveInteger

    @field_validator("root")
    @classmethod
    def root_is_absolute(cls, value: str) -> str:
        return _absolute_safe_directory(value, "raw extraction run root")

    @field_validator("run_id")
    @classmethod
    def run_id_is_safe(cls, value: str) -> str:
        if not _RUN_ID.fullmatch(value):
            raise ValueError("raw extraction run_id contains unsupported characters")
        return value


class RawExtractionRasterSourceConfig(_ConfigModel):
    type: Literal["raw_extraction_rasters"]
    expected_documents: PositiveInteger
    expected_pages: PositiveInteger
    extraction_runs: list[RawExtractionRunConfig] = Field(min_length=1)

    @model_validator(mode="after")
    def identities_and_counts_are_exact(self) -> RawExtractionRasterSourceConfig:
        ids = [row.id for row in self.extraction_runs]
        run_ids = [row.run_id for row in self.extraction_runs]
        roots = [os.path.abspath(row.root) for row in self.extraction_runs]
        if len(ids) != len(set(ids)):
            raise ValueError("raw extraction source ids must be unique")
        if len(run_ids) != len(set(run_ids)):
            raise ValueError("raw extraction run_ids must be unique")
        if len(roots) != len(set(roots)):
            raise ValueError("raw extraction run roots must be unique")
        if sum(row.expected_documents for row in self.extraction_runs) != (self.expected_documents):
            raise ValueError("raw extraction run document counts must sum to expected_documents")
        if sum(row.expected_pages for row in self.extraction_runs) != self.expected_pages:
            raise ValueError("raw extraction run page counts must sum to expected_pages")
        return self


TableSourceConfig = Annotated[
    ValidatedLabelRasterSourceConfig | RawExtractionRasterSourceConfig,
    Field(discriminator="type"),
]


class TableOutputConfig(_ConfigModel):
    root: NonEmptyString

    @field_validator("root")
    @classmethod
    def root_is_absolute(cls, value: str) -> str:
        return _absolute_safe_directory(value, "output root")


class TableRunConfig(_ConfigModel):
    run_id: NonEmptyString
    resume: bool
    fail_fast: bool
    max_total_attempts_per_page: PositiveInteger

    @field_validator("run_id")
    @classmethod
    def run_id_is_safe(cls, value: str) -> str:
        if not _RUN_ID.fullmatch(value):
            raise ValueError("run_id contains unsupported characters")
        return value


class TableConcurrencyConfig(_ConfigModel):
    max_active_documents: PositiveInteger
    max_inflight_pages_global: PositiveInteger
    max_inflight_pages_per_document: PositiveInteger

    @model_validator(mode="after")
    def per_document_fits_global(self) -> TableConcurrencyConfig:
        if self.max_inflight_pages_per_document > self.max_inflight_pages_global:
            raise ValueError("max_inflight_pages_per_document must not exceed the global limit")
        return self


class JoinInputConfig(_ConfigModel):
    split: Literal["train", "validation", "test"]
    path: NonEmptyString
    sha256: Sha256
    records: PositiveInteger

    @field_validator("path")
    @classmethod
    def path_is_absolute_jsonl(cls, value: str) -> str:
        if not Path(value).is_absolute() or Path(value).suffix != ".jsonl":
            raise ValueError("join input path must be an absolute .jsonl path")
        return value


class TableJoinConfig(_ConfigModel):
    dataset_id: Annotated[str, StringConstraints(pattern=r"^[A-Za-z0-9][A-Za-z0-9._-]*$")]
    output_root: NonEmptyString
    inputs: list[JoinInputConfig] = Field(min_length=1)

    @field_validator("output_root")
    @classmethod
    def output_root_is_absolute(cls, value: str) -> str:
        return _absolute_safe_directory(value, "join output root")

    @model_validator(mode="after")
    def has_train_and_validation(self) -> TableJoinConfig:
        splits = {row.split for row in self.inputs}
        if not {"train", "validation"}.issubset(splits):
            raise ValueError("join inputs must include train and validation")
        paths = [row.path for row in self.inputs]
        if len(paths) != len(set(paths)):
            raise ValueError("join input paths must be unique")
        return self


class TableViewConfig(_ConfigModel):
    schema_version: Literal[1]
    view_type: Literal["glm_ocr_table_recognition"]
    source: TableSourceConfig
    output: TableOutputConfig
    run: TableRunConfig
    vllm: TableVllmConfig
    concurrency: TableConcurrencyConfig
    join: TableJoinConfig | None = None

    @model_validator(mode="after")
    def source_output_and_join_do_not_overlap(self) -> TableViewConfig:
        output = Path(os.path.abspath(self.output.root))
        joined = Path(os.path.abspath(self.join.output_root)) if self.join is not None else None
        if joined is not None and (
            output == joined or output in joined.parents or joined in output.parents
        ):
            raise ValueError("table output and joined-dataset roots must not overlap")
        if isinstance(self.source, ValidatedLabelRasterSourceConfig):
            for record_set in self.source.record_sets:
                root = Path(os.path.abspath(record_set.root))
                if output == root or output in root.parents or root in output.parents:
                    raise ValueError("table output must not overlap a label record-set root")
                if joined is not None and (
                    joined == root or joined in root.parents or root in joined.parents
                ):
                    raise ValueError("joined output must not overlap a label record-set root")
        for extraction_run in self.source.extraction_runs:
            root = Path(os.path.abspath(extraction_run.root))
            if output == root or output in root.parents or root in output.parents:
                raise ValueError("table output must not overlap an extraction-run root")
            if joined is not None and (
                joined == root or joined in root.parents or root in joined.parents
            ):
                raise ValueError("joined output must not overlap an extraction-run root")
        if self.join is not None and sum(row.records for row in self.join.inputs) != (
            self.source.expected_documents
        ):
            raise ValueError("join input counts must sum to source.expected_documents")
        return self


def load_table_view_config(path: str | Path) -> TableViewConfig:
    return TableViewConfig.model_validate(load_strict_yaml_mapping(path), strict=True)
