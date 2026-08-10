"""Strict configuration contracts for the document OCR extraction pipeline.

The pipeline configuration is intentionally explicit at reproducibility
boundaries.  In particular, a source snapshot, output location, run identity,
and immutable model revision must all be supplied by the caller.
"""

from __future__ import annotations

import os
import re
from collections.abc import Hashable
from pathlib import Path
from typing import Annotated, Any, Literal
from urllib.parse import urlsplit

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

_SHA_40_PATTERN = re.compile(r"^[0-9a-fA-F]{40}$")
_RUN_ID_PATTERN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]*$")
_S3_BUCKET_PATTERN = re.compile(r"^[a-z0-9][a-z0-9.-]{1,61}[a-z0-9]$")


class _UniqueKeySafeLoader(yaml.SafeLoader):
    """Safe YAML loader that rejects ambiguous duplicate mapping keys."""

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


class _StrictConfigModel(BaseModel):
    """Base for YAML-facing models: no coercion and no unknown settings."""

    model_config = ConfigDict(extra="forbid", strict=True, allow_inf_nan=False)


class LocalSourceConfig(_StrictConfigModel):
    """A content-addressed local PDF source snapshot."""

    type: Literal["local"]
    root: NonEmptyString
    include_glob: NonEmptyString
    dataset_version: NonEmptyString
    require_content_sha256: Literal[True] = True

    @field_validator("root")
    @classmethod
    def root_must_be_absolute(cls, value: str) -> str:
        if not Path(value).is_absolute():
            raise ValueError("local source root must be an absolute path")
        return value


class S3SourceConfig(_StrictConfigModel):
    """An S3 source whose objects must be read by exact VersionId."""

    type: Literal["s3"]
    bucket: NonEmptyString
    prefix: NonEmptyString
    include_glob: NonEmptyString
    region: NonEmptyString
    dataset_version: NonEmptyString
    require_object_version_ids: Literal[True] = True

    @field_validator("bucket")
    @classmethod
    def validate_bucket(cls, value: str) -> str:
        if not _S3_BUCKET_PATTERN.fullmatch(value):
            raise ValueError("bucket must be a valid lowercase S3 bucket name")
        if ".." in value or ".-" in value or "-." in value:
            raise ValueError("bucket must be a valid S3 bucket name")
        return value

    @field_validator("prefix")
    @classmethod
    def prefix_is_not_an_absolute_uri(cls, value: str) -> str:
        if value.startswith("s3://"):
            raise ValueError("prefix must be a key prefix, not an s3:// URI")
        return value


SourceConfig = Annotated[
    LocalSourceConfig | S3SourceConfig,
    Field(discriminator="type"),
]


class OutputConfig(_StrictConfigModel):
    """Destination and batching policy for the completed raw OCR dataset."""

    root: NonEmptyString
    parquet_compression: Literal["zstd", "snappy", "none"]
    write_batch_rows: PositiveInteger

    @field_validator("root")
    @classmethod
    def output_root_is_absolute(cls, value: str) -> str:
        if not Path(value).is_absolute():
            raise ValueError("output root must be an absolute local path")
        normalized = Path(os.path.abspath(value))
        if normalized == Path(normalized.anchor):
            raise ValueError("output root must not be the filesystem root")
        return value


class RunConfig(_StrictConfigModel):
    """Identity and failure semantics for one immutable extraction run."""

    run_id: NonEmptyString
    fail_fast: bool
    resume: bool

    @field_validator("run_id")
    @classmethod
    def validate_run_id(cls, value: str) -> str:
        if not _RUN_ID_PATTERN.fullmatch(value):
            raise ValueError("run_id may contain only letters, digits, '.', '_', and '-'")
        return value


class RasterConfig(_StrictConfigModel):
    """Bounded pypdfium2 page rasterization settings."""

    dpi: PositiveInteger
    max_side_pixels: PositiveInteger
    max_pixels: PositiveInteger
    max_pages_per_document: PositiveInteger
    max_pdf_bytes: PositiveInteger
    max_cached_documents_per_process: PositiveInteger
    image_format: Literal["png", "jpeg"]
    jpeg_quality: Annotated[int, Field(ge=1, le=100)] | None = None
    draw_annotations: bool
    reject_xfa: bool

    @model_validator(mode="after")
    def validate_format_specific_quality(self) -> RasterConfig:
        if self.image_format == "jpeg" and self.jpeg_quality is None:
            raise ValueError("jpeg_quality is required when image_format is 'jpeg'")
        if self.image_format == "png" and self.jpeg_quality is not None:
            raise ValueError("jpeg_quality is only valid when image_format is 'jpeg'")
        return self


class SamplingConfig(_StrictConfigModel):
    """Deterministic generation parameters for raw-corpus extraction."""

    temperature: float
    top_p: Annotated[float, Field(gt=0.0, le=1.0)]
    top_k: PositiveInteger
    repetition_penalty: PositiveFloat
    max_tokens: PositiveInteger
    seed: NonNegativeInteger

    @field_validator("temperature")
    @classmethod
    def temperature_must_be_greedy(cls, value: float) -> float:
        if value != 0.0:
            raise ValueError("temperature must be 0 for deterministic raw OCR")
        return value


class RetryConfig(_StrictConfigModel):
    """Explicit bounded exponential retry policy for inference requests."""

    max_attempts: PositiveInteger
    initial_backoff_seconds: PositiveFloat
    max_backoff_seconds: PositiveFloat
    backoff_multiplier: Annotated[float, Field(ge=1.0)]
    jitter_fraction: Annotated[float, Field(ge=0.0, le=1.0)]

    @model_validator(mode="after")
    def maximum_backoff_covers_initial_backoff(self) -> RetryConfig:
        if self.max_backoff_seconds < self.initial_backoff_seconds:
            raise ValueError("max_backoff_seconds must be at least initial_backoff_seconds")
        return self


class SpeculativeDecodingConfig(_StrictConfigModel):
    """GLM-OCR's vLLM multi-token-prediction decoding settings."""

    method: Literal["mtp"]
    num_speculative_tokens: Literal[1]


class VllmConfig(_StrictConfigModel):
    """vLLM server identity and request policy."""

    endpoint: NonEmptyString
    model: NonEmptyString
    served_model_name: NonEmptyString
    revision: NonEmptyString
    engine_version: NonEmptyString
    container_image: NonEmptyString
    max_model_len: PositiveInteger
    max_num_seqs: PositiveInteger
    gpu_memory_utilization: Annotated[float, Field(gt=0.0, le=1.0)]
    prompt: Literal["Text Recognition:"]
    request_timeout_seconds: PositiveFloat
    api_key_env: NonEmptyString
    sampling: SamplingConfig
    retry: RetryConfig
    speculative_decoding: SpeculativeDecodingConfig

    @field_validator("endpoint")
    @classmethod
    def endpoint_must_be_http(cls, value: str) -> str:
        parsed = urlsplit(value)
        if parsed.scheme not in {"http", "https"} or not parsed.netloc:
            raise ValueError("endpoint must be an absolute HTTP(S) URL")
        try:
            hostname = parsed.hostname
            _ = parsed.port
        except ValueError as error:
            raise ValueError("endpoint has an invalid host or port") from error
        if not hostname:
            raise ValueError("endpoint must contain a valid hostname")
        if parsed.username is not None or parsed.password is not None:
            raise ValueError("endpoint must not contain credentials")
        if parsed.query or parsed.fragment:
            raise ValueError("endpoint must not contain a query string or fragment")
        if parsed.path not in {"", "/"}:
            raise ValueError(
                "endpoint must be the base server URL; the client appends the API path"
            )
        return value

    @field_validator("revision")
    @classmethod
    def revision_must_be_an_immutable_commit(cls, value: str) -> str:
        if not _SHA_40_PATTERN.fullmatch(value):
            raise ValueError(
                "revision must be an immutable 40-character Git commit SHA; "
                "branches such as 'main' are not reproducible"
            )
        return value.lower()

    @field_validator("container_image")
    @classmethod
    def container_image_must_be_digest_pinned(cls, value: str) -> str:
        if not re.fullmatch(r"[^\s]+@sha256:[0-9a-f]{64}", value):
            raise ValueError("container_image must include an immutable lowercase @sha256 digest")
        return value

    @model_validator(mode="after")
    def model_context_covers_maximum_output(self) -> VllmConfig:
        if self.max_model_len <= self.sampling.max_tokens:
            raise ValueError(
                "max_model_len must exceed sampling.max_tokens to leave room for "
                "image/prompt tokens"
            )
        return self


class ConcurrencyConfig(_StrictConfigModel):
    """Independent document, rendering, and inference concurrency limits."""

    max_active_documents: PositiveInteger
    renderer_processes: PositiveInteger
    max_inflight_pages_global: PositiveInteger
    max_inflight_pages_per_document: PositiveInteger

    @model_validator(mode="after")
    def per_document_limit_fits_global_limit(self) -> ConcurrencyConfig:
        if self.max_inflight_pages_per_document > self.max_inflight_pages_global:
            raise ValueError(
                "max_inflight_pages_per_document must not exceed max_inflight_pages_global"
            )
        return self


class BenchmarkConcurrencyPoint(_StrictConfigModel):
    """One valid inference-concurrency point in a benchmark sweep."""

    max_inflight_pages_global: PositiveInteger
    max_inflight_pages_per_document: PositiveInteger

    @model_validator(mode="after")
    def per_document_limit_fits_global_limit(self) -> BenchmarkConcurrencyPoint:
        if self.max_inflight_pages_per_document > self.max_inflight_pages_global:
            raise ValueError(
                "benchmark per-document concurrency must not exceed global concurrency"
            )
        return self


class BenchmarkSweepConfig(_StrictConfigModel):
    """Optional, finite set of concurrency points to benchmark later."""

    points: list[BenchmarkConcurrencyPoint] = Field(min_length=1)
    warmup_pages: NonNegativeInteger
    measured_pages: PositiveInteger
    repetitions: PositiveInteger

    @field_validator("points")
    @classmethod
    def points_must_be_unique(
        cls, value: list[BenchmarkConcurrencyPoint]
    ) -> list[BenchmarkConcurrencyPoint]:
        keys = [
            (
                point.max_inflight_pages_global,
                point.max_inflight_pages_per_document,
            )
            for point in value
        ]
        if len(keys) != len(set(keys)):
            raise ValueError("benchmark concurrency points must be unique")
        return value


class PipelineConfig(_StrictConfigModel):
    """Complete versioned contract for one raw page-OCR extraction run."""

    schema_version: Literal[1]
    source: SourceConfig
    output: OutputConfig
    run: RunConfig
    raster: RasterConfig
    vllm: VllmConfig
    concurrency: ConcurrencyConfig
    benchmark: BenchmarkSweepConfig | None = None

    @model_validator(mode="after")
    def local_source_and_output_must_not_overlap(self) -> PipelineConfig:
        if isinstance(self.source, LocalSourceConfig):
            source_root = Path(self.source.root).resolve(strict=False)
            output_root = Path(self.output.root).resolve(strict=False)
            if (
                source_root == output_root
                or source_root in output_root.parents
                or output_root in source_root.parents
            ):
                raise ValueError("local source and output roots must not overlap")
        return self


def load_config(path: str | Path) -> PipelineConfig:
    """Load a YAML file and validate it without type coercion or extra fields."""

    config_path = Path(path)
    with config_path.open("r", encoding="utf-8") as stream:
        raw: Any = yaml.load(stream, Loader=_UniqueKeySafeLoader)
    if not isinstance(raw, dict):
        raise ValueError("pipeline configuration root must be a YAML mapping")
    return PipelineConfig.model_validate(raw, strict=True)
