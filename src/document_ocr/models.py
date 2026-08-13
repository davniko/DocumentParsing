"""Immutable, page-granular provenance and extraction result contracts."""

from __future__ import annotations

import hashlib
import re
from pathlib import PurePosixPath
from typing import Annotated, Literal

from pydantic import (
    AwareDatetime,
    BaseModel,
    ConfigDict,
    Field,
    StringConstraints,
    field_validator,
    model_validator,
)

NonEmptyString = Annotated[str, StringConstraints(strip_whitespace=True, min_length=1)]
PositiveInteger = Annotated[int, Field(gt=0)]
NonNegativeInteger = Annotated[int, Field(ge=0)]
PositiveFloat = Annotated[float, Field(gt=0)]
NonNegativeFloat = Annotated[float, Field(ge=0)]

_SHA256_PATTERN = re.compile(r"^[0-9a-f]{64}$")
_GIT_SHA_PATTERN = re.compile(r"^[0-9a-f]{40}$")


class _FrozenRecord(BaseModel):
    """Strict immutable base shared by records written to durable datasets."""

    model_config = ConfigDict(extra="forbid", frozen=True, strict=True, allow_inf_nan=False)


class SourceObject(_FrozenRecord):
    """A version-pinned PDF object discovered in a local or S3 snapshot.

    Source-specific metadata is kept in flat columns and validated as a
    discriminated contract.  This makes an inventory directly serializable to
    JSON or Parquet without hiding required provenance in a URI.
    """

    document_id: NonEmptyString
    source_type: Literal["local", "s3"]
    source_uri: NonEmptyString
    source_dataset_version: NonEmptyString
    source_object_version: NonEmptyString
    source_size_bytes: PositiveInteger
    source_last_modified: AwareDatetime
    source_sha256: str | None = None

    local_canonical_path: NonEmptyString | None = None
    local_relative_key: NonEmptyString | None = None
    local_device: NonNegativeInteger | None = None
    local_inode: NonNegativeInteger | None = None
    local_mtime_ns: NonNegativeInteger | None = None

    s3_bucket: NonEmptyString | None = None
    s3_key: NonEmptyString | None = None
    s3_version_id: NonEmptyString | None = None
    s3_etag: NonEmptyString | None = None
    s3_checksum_crc32: NonEmptyString | None = None
    s3_checksum_crc32c: NonEmptyString | None = None
    s3_checksum_sha1: NonEmptyString | None = None
    s3_checksum_sha256: NonEmptyString | None = None
    s3_checksum_crc64nvme: NonEmptyString | None = None
    s3_checksum_type: Literal["COMPOSITE", "FULL_OBJECT"] | None = None

    @field_validator("source_sha256")
    @classmethod
    def validate_optional_source_sha256(cls, value: str | None) -> str | None:
        if value is not None and not _SHA256_PATTERN.fullmatch(value):
            raise ValueError("source_sha256 must be a lowercase 64-character SHA-256")
        return value

    @model_validator(mode="after")
    def validate_source_specific_provenance(self) -> SourceObject:
        local_fields = {
            "local_canonical_path": self.local_canonical_path,
            "local_relative_key": self.local_relative_key,
            "local_device": self.local_device,
            "local_inode": self.local_inode,
            "local_mtime_ns": self.local_mtime_ns,
        }
        s3_required_fields = {
            "s3_bucket": self.s3_bucket,
            "s3_key": self.s3_key,
            "s3_version_id": self.s3_version_id,
            "s3_etag": self.s3_etag,
        }
        s3_optional_fields = {
            "s3_checksum_crc32": self.s3_checksum_crc32,
            "s3_checksum_crc32c": self.s3_checksum_crc32c,
            "s3_checksum_sha1": self.s3_checksum_sha1,
            "s3_checksum_sha256": self.s3_checksum_sha256,
            "s3_checksum_crc64nvme": self.s3_checksum_crc64nvme,
            "s3_checksum_type": self.s3_checksum_type,
        }

        if self.source_type == "local":
            missing = [name for name, value in local_fields.items() if value is None]
            if self.source_sha256 is None:
                missing.append("source_sha256")
            if missing:
                raise ValueError(
                    "local source is missing required provenance: " + ", ".join(missing)
                )
            supplied_s3 = [
                name
                for name, value in {**s3_required_fields, **s3_optional_fields}.items()
                if value is not None
            ]
            if supplied_s3:
                raise ValueError(
                    "local source must not contain S3 provenance: " + ", ".join(supplied_s3)
                )
            return self

        missing = [name for name, value in s3_required_fields.items() if value is None]
        if missing:
            raise ValueError("S3 source is missing required provenance: " + ", ".join(missing))
        supplied_local = [name for name, value in local_fields.items() if value is not None]
        if supplied_local:
            raise ValueError(
                "S3 source must not contain local provenance: " + ", ".join(supplied_local)
            )
        if self.s3_version_id != self.source_object_version:
            raise ValueError("s3_version_id must equal source_object_version")
        return self


class PageProvenance(SourceObject):
    """Source and page identity repeated on every output row."""

    schema_version: Literal[1]
    run_id: NonEmptyString
    extraction_id: NonEmptyString
    page_id: NonEmptyString
    config_sha256: str
    pipeline_fingerprint: str
    source_sha256: str
    document_page_count: PositiveInteger
    page_index: NonNegativeInteger
    page_number: PositiveInteger

    @field_validator("config_sha256", "pipeline_fingerprint", "source_sha256")
    @classmethod
    def validate_required_sha256(cls, value: str) -> str:
        if not _SHA256_PATTERN.fullmatch(value):
            raise ValueError("hash must be a lowercase 64-character SHA-256")
        return value

    @model_validator(mode="after")
    def validate_page_position(self) -> PageProvenance:
        if self.page_number != self.page_index + 1:
            raise ValueError("page_number must equal zero-based page_index + 1")
        if self.page_number > self.document_page_count:
            raise ValueError("page_number must not exceed document_page_count")
        return self


class RasterMetadata(_FrozenRecord):
    """Renderer versions, page geometry, output bounds, and render timing."""

    renderer_name: Literal["pypdfium2"]
    renderer_version: NonEmptyString
    pdfium_version: NonEmptyString
    page_width_points: PositiveFloat
    page_height_points: PositiveFloat
    page_rotation_degrees: Literal[0, 90, 180, 270]
    requested_dpi: PositiveInteger
    effective_dpi: PositiveFloat
    render_scale: PositiveFloat
    raster_width_px: PositiveInteger
    raster_height_px: PositiveInteger
    raster_image_format: Literal["png", "jpeg"]
    raster_mime_type: Literal["image/png", "image/jpeg"]
    draw_annotations: bool
    pdf_form_type: Literal["none", "acroform", "xfa_full", "xfa_foreground"]
    raster_size_bytes: PositiveInteger
    raster_sha256: str
    render_duration_ms: NonNegativeFloat

    @field_validator("raster_sha256")
    @classmethod
    def validate_raster_sha256(cls, value: str) -> str:
        if not _SHA256_PATTERN.fullmatch(value):
            raise ValueError("raster_sha256 must be a lowercase 64-character SHA-256")
        return value

    @model_validator(mode="after")
    def mime_type_matches_image_format(self) -> RasterMetadata:
        expected = {
            "png": "image/png",
            "jpeg": "image/jpeg",
        }[self.raster_image_format]
        if self.raster_mime_type != expected:
            raise ValueError("raster_mime_type must match raster_image_format")
        return self


class InferenceMetadata(_FrozenRecord):
    """Exact model/server/request identity for a successful page inference."""

    inference_endpoint: NonEmptyString
    inference_model: NonEmptyString
    inference_served_model_name: NonEmptyString
    inference_model_revision: str
    inference_server_engine: Literal["vllm"]
    inference_server_engine_version: NonEmptyString
    inference_server_base_image: NonEmptyString
    inference_server_build_manifest_sha256: str
    inference_server_contract_sha256: str
    inference_speculative_method: Literal["mtp"]
    inference_num_speculative_tokens: Literal[1]
    inference_prompt: Literal["Text Recognition:"]
    inference_prompt_sha256: str
    inference_temperature: float
    inference_top_p: Annotated[float, Field(gt=0.0, le=1.0)]
    inference_top_k: PositiveInteger
    inference_repetition_penalty: PositiveFloat
    inference_max_tokens: PositiveInteger
    inference_seed: NonNegativeInteger
    inference_request_id: NonEmptyString
    inference_server_request_id: NonEmptyString | None = None
    inference_finish_reason: Literal["stop"]
    inference_prompt_tokens: NonNegativeInteger
    inference_completion_tokens: NonNegativeInteger
    inference_attempt_count: PositiveInteger
    inference_duration_ms: NonNegativeFloat

    @field_validator("inference_model_revision")
    @classmethod
    def validate_model_revision(cls, value: str) -> str:
        if not _GIT_SHA_PATTERN.fullmatch(value):
            raise ValueError("inference_model_revision must be a lowercase 40-character Git SHA")
        return value

    @field_validator(
        "inference_prompt_sha256",
        "inference_server_contract_sha256",
        "inference_server_build_manifest_sha256",
    )
    @classmethod
    def validate_inference_hashes(cls, value: str) -> str:
        if not _SHA256_PATTERN.fullmatch(value):
            raise ValueError("inference hashes must be lowercase 64-character SHA-256 values")
        return value

    @field_validator("inference_temperature")
    @classmethod
    def validate_greedy_temperature(cls, value: float) -> float:
        if value != 0.0:
            raise ValueError("inference_temperature must be 0 for deterministic OCR")
        return value


class PageTiming(_FrozenRecord):
    """End-to-end wall-clock timestamps and non-overlapping timing buckets."""

    extraction_started_at: AwareDatetime
    extraction_completed_at: AwareDatetime
    queue_duration_ms: NonNegativeFloat
    inference_duration_ms: NonNegativeFloat
    persist_duration_ms: NonNegativeFloat
    total_duration_ms: NonNegativeFloat

    @model_validator(mode="after")
    def completion_is_not_before_start(self) -> PageTiming:
        if self.extraction_completed_at < self.extraction_started_at:
            raise ValueError("extraction_completed_at must not precede extraction_started_at")
        return self


class InferenceAttempt(PageProvenance):
    """One flat request-attempt row, including retryable and terminal failures."""

    attempt_number: PositiveInteger
    attempt_started_at: AwareDatetime
    attempt_completed_at: AwareDatetime
    attempt_duration_ms: NonNegativeFloat
    attempt_outcome: Literal["success", "retryable_error", "terminal_error"]
    inference_request_id: NonEmptyString | None = None
    inference_server_request_id: NonEmptyString | None = None
    http_status_code: Annotated[int, Field(ge=100, le=599)] | None = None
    retry_after_seconds: NonNegativeFloat | None = None
    error_type: NonEmptyString | None = None
    error_message: str | None = None

    @model_validator(mode="after")
    def validate_attempt_outcome(self) -> InferenceAttempt:
        if self.attempt_completed_at < self.attempt_started_at:
            raise ValueError("attempt_completed_at must not precede attempt_started_at")
        if self.attempt_outcome == "success":
            if self.inference_request_id is None:
                raise ValueError("successful attempt requires inference_request_id")
            if self.error_type is not None or self.error_message is not None:
                raise ValueError("successful attempt must not contain error fields")
        elif self.error_type is None or self.error_message is None:
            raise ValueError("failed attempt requires error_type and error_message")
        if self.retry_after_seconds is not None and self.attempt_outcome != "retryable_error":
            raise ValueError("retry_after_seconds is only valid for retryable errors")
        return self


class PageExtractionRecord(
    PageProvenance,
    RasterMetadata,
    InferenceMetadata,
    PageTiming,
):
    """One successful raw OCR row; ``raw_ocr_text`` is never normalized."""

    raster_path: NonEmptyString | None
    raw_ocr_text: str
    raw_ocr_text_sha256: str
    raw_response_sha256: str
    raw_response_path: NonEmptyString

    @field_validator("raw_ocr_text_sha256", "raw_response_sha256")
    @classmethod
    def validate_raw_hashes(cls, value: str) -> str:
        if not _SHA256_PATTERN.fullmatch(value):
            raise ValueError("raw hashes must be lowercase 64-character SHA-256 values")
        return value

    @field_validator("raw_response_path")
    @classmethod
    def raw_response_path_is_safe_and_relative(cls, value: str) -> str:
        path = PurePosixPath(value)
        if path.is_absolute() or ".." in path.parts or value == "." or "\\" in value:
            raise ValueError("raw_response_path must be a safe relative POSIX path")
        return value

    @field_validator("raster_path")
    @classmethod
    def raster_path_is_safe_and_relative(cls, value: str | None) -> str | None:
        if value is None:
            return None
        path = PurePosixPath(value)
        if path.is_absolute() or ".." in path.parts or value == "." or "\\" in value:
            raise ValueError("raster_path must be a safe relative POSIX path")
        return value

    @model_validator(mode="after")
    def raw_text_hash_matches_text(self) -> PageExtractionRecord:
        actual = hashlib.sha256(self.raw_ocr_text.encode("utf-8")).hexdigest()
        if self.raw_ocr_text_sha256 != actual:
            raise ValueError("raw_ocr_text_sha256 does not match raw_ocr_text UTF-8 bytes")
        if self.raster_path is not None:
            suffix = ".png" if self.raster_image_format == "png" else ".jpg"
            expected = (
                PurePosixPath("page-images")
                / self.document_id
                / self.page_id
                / f"{self.raster_sha256}{suffix}"
            ).as_posix()
            if self.raster_path != expected:
                raise ValueError("raster_path is not bound to page identity and raster hash")
        return self


class PageExtractionFailure(PageProvenance):
    """One terminal page failure row correlated with separate attempt rows."""

    failure_stage: Literal[
        "pdf_open",
        "render",
        "raster_publish",
        "raster_validate",
        "inference",
        "persist",
    ]
    failed_at: AwareDatetime
    inference_attempt_count: NonNegativeInteger
    retryable: bool
    error_type: NonEmptyString
    error_message: str
    last_inference_request_id: NonEmptyString | None = None

    @model_validator(mode="after")
    def validate_failure_attempt_count(self) -> PageExtractionFailure:
        after_inference = self.failure_stage in {"inference", "persist"}
        if after_inference and self.inference_attempt_count == 0:
            raise ValueError("inference and post-inference persistence failures require an attempt")
        if (self.inference_attempt_count == 0) != (self.last_inference_request_id is None):
            raise ValueError(
                "last_inference_request_id must be present exactly when attempt history exists"
            )
        return self


class DocumentExtractionFailure(SourceObject):
    """A terminal failure before a PDF could expose a complete page inventory."""

    schema_version: Literal[1]
    run_id: NonEmptyString
    config_sha256: str
    pipeline_fingerprint: str
    failure_stage: Literal["materialize", "pdf_open"]
    failed_at: AwareDatetime
    error_type: NonEmptyString
    error_message: str

    @field_validator("config_sha256", "pipeline_fingerprint")
    @classmethod
    def validate_document_failure_hashes(cls, value: str) -> str:
        if not _SHA256_PATTERN.fullmatch(value):
            raise ValueError("hash must be a lowercase 64-character SHA-256")
        return value


class DatasetSummary(_FrozenRecord):
    """Exact completeness counters bound into the published dataset manifest."""

    inventory_documents: NonNegativeInteger
    registered_documents: NonNegativeInteger
    expected_pages: NonNegativeInteger
    complete_documents: NonNegativeInteger
    failed_documents: NonNegativeInteger
    registered_pages: NonNegativeInteger
    successful_pages: NonNegativeInteger
    failed_pages: NonNegativeInteger
    pending_pages: NonNegativeInteger
    attempt_rows: NonNegativeInteger
    audited_successful_pages: NonNegativeInteger


class DatasetFileManifest(_FrozenRecord):
    """One content-addressed Parquet artifact in a completed raw OCR dataset."""

    record_model: Literal["PageExtractionRecord", "InferenceAttempt"]
    rows: NonNegativeInteger
    bytes: PositiveInteger
    sha256: str
    arrow_schema_sha256: str
    raw_response_artifact_count: NonNegativeInteger
    raw_response_bytes: NonNegativeInteger
    raster_artifact_count: NonNegativeInteger
    raster_bytes: NonNegativeInteger
    path: NonEmptyString

    @field_validator("sha256", "arrow_schema_sha256")
    @classmethod
    def validate_file_hashes(cls, value: str) -> str:
        if not _SHA256_PATTERN.fullmatch(value):
            raise ValueError("dataset artifact hashes must be lowercase SHA-256 values")
        return value

    @model_validator(mode="after")
    def validate_content_addressed_path(self) -> DatasetFileManifest:
        stem_by_model = {
            "PageExtractionRecord": "pages",
            "InferenceAttempt": "attempts",
        }
        expected = f"dataset/{stem_by_model[self.record_model]}-{self.sha256[:16]}.parquet"
        path = PurePosixPath(self.path)
        if path.is_absolute() or ".." in path.parts or "\\" in self.path or self.path != expected:
            raise ValueError("dataset file path is not its expected content-addressed path")
        return self


class RawResponseManifest(_FrozenRecord):
    """Aggregate size and count of page-level raw vLLM response artifacts."""

    artifacts: NonNegativeInteger
    bytes: NonNegativeInteger


class PageImageManifest(_FrozenRecord):
    """Aggregate size and count of retained page-raster artifacts."""

    retained: bool
    artifacts: NonNegativeInteger
    bytes: NonNegativeInteger

    @model_validator(mode="after")
    def disabled_retention_has_no_artifacts(self) -> PageImageManifest:
        if not self.retained and (self.artifacts != 0 or self.bytes != 0):
            raise ValueError("disabled page-image retention cannot publish artifacts")
        return self


class DatasetManifest(_FrozenRecord):
    """Typed, manifest-last contract for a complete page extraction dataset."""

    schema_version: Literal[1]
    dataset_kind: Literal["glm-ocr-raw-page-extractions"]
    run_id: NonEmptyString
    config_sha256: str
    pipeline_fingerprint: str
    inventory_sha256: str
    summary: DatasetSummary
    raw_responses: RawResponseManifest
    page_images: PageImageManifest
    files: tuple[DatasetFileManifest, DatasetFileManifest]

    @field_validator("config_sha256", "pipeline_fingerprint", "inventory_sha256")
    @classmethod
    def validate_manifest_hashes(cls, value: str) -> str:
        if not _SHA256_PATTERN.fullmatch(value):
            raise ValueError("manifest identity hashes must be lowercase SHA-256 values")
        return value

    @model_validator(mode="after")
    def validate_file_roles_and_counts(self) -> DatasetManifest:
        expected_models = (
            "PageExtractionRecord",
            "InferenceAttempt",
        )
        if tuple(item.record_model for item in self.files) != expected_models:
            raise ValueError("dataset files must contain pages and attempts in order")
        expected_rows = (
            self.summary.successful_pages,
            self.summary.attempt_rows,
        )
        if tuple(item.rows for item in self.files) != expected_rows:
            raise ValueError("dataset file row counts do not match the completeness summary")
        pages, attempts = self.files
        if (
            pages.raw_response_artifact_count != self.raw_responses.artifacts
            or pages.raw_response_bytes != self.raw_responses.bytes
            or attempts.raw_response_artifact_count != 0
            or attempts.raw_response_bytes != 0
        ):
            raise ValueError("raw response totals do not match page artifact metadata")
        if self.raw_responses.artifacts != self.summary.successful_pages:
            raise ValueError("every successful page must have exactly one raw response artifact")
        if (
            pages.raster_artifact_count != self.page_images.artifacts
            or pages.raster_bytes != self.page_images.bytes
            or attempts.raster_artifact_count != 0
            or attempts.raster_bytes != 0
        ):
            raise ValueError("page image totals do not match page artifact metadata")
        expected_page_images = self.summary.successful_pages if self.page_images.retained else 0
        if self.page_images.artifacts != expected_page_images:
            raise ValueError(
                "page image retention must publish exactly one raster per successful page"
            )
        return self
