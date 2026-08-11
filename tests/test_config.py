from __future__ import annotations

import hashlib
import math
from datetime import UTC, datetime
from typing import Any

import pytest
import yaml
from pydantic import ValidationError

from document_ocr.config import LocalSourceConfig, PipelineConfig, S3SourceConfig, load_config
from document_ocr.models import (
    DocumentExtractionFailure,
    InferenceAttempt,
    PageExtractionFailure,
    PageExtractionRecord,
    SourceObject,
)
from document_ocr.vllm_contract import runtime_contract_payload, runtime_contract_sha256

SHA256 = "a" * 64
OTHER_SHA256 = "b" * 64
MODEL_REVISION = "c" * 40
CONTAINER_IMAGE = (
    "vllm/vllm-openai:v0.26.0@sha256:"
    "ffb2d59b1c059a5bd8d781320c9f5189de8293693b7d95da54befddaa54abf52"
)
SERVER_CONTRACT_SHA256 = runtime_contract_sha256(
    runtime_contract_payload(
        model="zai-org/GLM-OCR",
        served_model_name="glm-ocr",
        model_revision=MODEL_REVISION,
        max_model_len=32768,
        max_num_seqs=16,
        gpu_memory_utilization=0.9,
        generation_config="vllm",
        speculative_method="mtp",
        num_speculative_tokens=1,
        image_limit_per_prompt=1,
        container_image=CONTAINER_IMAGE,
    )
)
NOW = datetime(2026, 8, 5, 10, 0, tzinfo=UTC)
LATER = datetime(2026, 8, 5, 10, 1, tzinfo=UTC)


def valid_config_data() -> dict[str, Any]:
    return {
        "schema_version": 1,
        "source": {
            "type": "local",
            "root": "/data/pdfs",
            "include_glob": "**/*.pdf",
            "dataset_version": "contracts-2026-08-05",
            "require_content_sha256": True,
        },
        "output": {
            "root": "/data/ocr-output",
            "retain_page_images": False,
            "parquet_compression": "zstd",
            "write_batch_rows": 256,
        },
        "run": {
            "run_id": "glm-ocr-20260805T100000Z",
            "fail_fast": False,
            "resume": True,
        },
        "raster": {
            "dpi": 200,
            "max_side_pixels": 4096,
            "max_pixels": 12_000_000,
            "max_pages_per_document": 500,
            "max_pdf_bytes": 1_000_000_000,
            "max_cached_documents_per_process": 2,
            "image_format": "png",
            "jpeg_quality": None,
            "draw_annotations": True,
            "reject_xfa": True,
        },
        "vllm": {
            "endpoint": "http://127.0.0.1:8000",
            "model": "zai-org/GLM-OCR",
            "served_model_name": "glm-ocr",
            "revision": MODEL_REVISION,
            "engine_version": "0.26.0",
            "container_image": CONTAINER_IMAGE,
            "max_model_len": 32768,
            "max_num_seqs": 16,
            "gpu_memory_utilization": 0.9,
            "prompt": "Text Recognition:",
            "request_timeout_seconds": 120.0,
            "api_key_env": "VLLM_API_KEY",
            "sampling": {
                "temperature": 0.0,
                "top_p": 1.0,
                "top_k": 1,
                "repetition_penalty": 1.0,
                "max_tokens": 8192,
                "seed": 0,
            },
            "retry": {
                "max_attempts": 4,
                "initial_backoff_seconds": 0.5,
                "max_backoff_seconds": 8.0,
                "backoff_multiplier": 2.0,
                "jitter_fraction": 0.2,
            },
            "speculative_decoding": {
                "method": "mtp",
                "num_speculative_tokens": 1,
            },
        },
        "concurrency": {
            "max_active_documents": 4,
            "renderer_processes": 4,
            "max_inflight_pages_global": 16,
            "max_inflight_pages_per_document": 4,
        },
    }


def valid_local_source_object_data() -> dict[str, Any]:
    return {
        "document_id": "document-001",
        "source_type": "local",
        "source_uri": "file:///data/pdfs/a.pdf",
        "source_dataset_version": "contracts-2026-08-05",
        "source_object_version": SHA256,
        "source_size_bytes": 1024,
        "source_last_modified": NOW,
        "source_sha256": SHA256,
        "local_canonical_path": "/data/pdfs/a.pdf",
        "local_relative_key": "a.pdf",
        "local_device": 42,
        "local_inode": 1234,
        "local_mtime_ns": 1_754_389_200_000_000_000,
    }


def valid_page_provenance_data() -> dict[str, Any]:
    return {
        **valid_local_source_object_data(),
        "schema_version": 1,
        "run_id": "glm-ocr-20260805T100000Z",
        "extraction_id": "extract-document-001-page-000002",
        "page_id": "document-001:2",
        "config_sha256": OTHER_SHA256,
        "pipeline_fingerprint": "f" * 64,
        "document_page_count": 3,
        "page_index": 1,
        "page_number": 2,
    }


def valid_page_record_data() -> dict[str, Any]:
    return {
        **valid_page_provenance_data(),
        "renderer_name": "pypdfium2",
        "renderer_version": "4.30.0",
        "pdfium_version": "chromium/7000",
        "page_width_points": 612.0,
        "page_height_points": 792.0,
        "page_rotation_degrees": 0,
        "requested_dpi": 200,
        "effective_dpi": 200.0,
        "render_scale": 200.0 / 72.0,
        "raster_width_px": 1700,
        "raster_height_px": 2200,
        "raster_image_format": "png",
        "raster_mime_type": "image/png",
        "draw_annotations": True,
        "pdf_form_type": "none",
        "raster_size_bytes": 400_000,
        "raster_sha256": "d" * 64,
        "render_duration_ms": 31.5,
        "inference_endpoint": "http://127.0.0.1:8000",
        "inference_model": "zai-org/GLM-OCR",
        "inference_served_model_name": "glm-ocr",
        "inference_model_revision": MODEL_REVISION,
        "inference_server_engine": "vllm",
        "inference_server_engine_version": "0.26.0",
        "inference_server_image": (
            "vllm/vllm-openai:v0.26.0@sha256:"
            "ffb2d59b1c059a5bd8d781320c9f5189de8293693b7d95da54befddaa54abf52"
        ),
        "inference_server_contract_sha256": SERVER_CONTRACT_SHA256,
        "inference_speculative_method": "mtp",
        "inference_num_speculative_tokens": 1,
        "inference_prompt": "Text Recognition:",
        "inference_prompt_sha256": (
            "77e8207573e1b8ba857d25db3a5080ae8651d6ddc2ea79007af5ca7958b80017"
        ),
        "inference_temperature": 0.0,
        "inference_top_p": 1.0,
        "inference_top_k": 1,
        "inference_repetition_penalty": 1.0,
        "inference_max_tokens": 8192,
        "inference_seed": 0,
        "inference_request_id": "req-123",
        "inference_server_request_id": "server-req-123",
        "inference_finish_reason": "stop",
        "inference_prompt_tokens": 128,
        "inference_completion_tokens": 256,
        "inference_attempt_count": 1,
        "inference_duration_ms": 225.0,
        "extraction_started_at": NOW,
        "extraction_completed_at": LATER,
        "queue_duration_ms": 2.0,
        "persist_duration_ms": 4.0,
        "total_duration_ms": 262.5,
        "raw_ocr_text": "  Heading\n\nbody text\n",
        "raw_ocr_text_sha256": hashlib.sha256(b"  Heading\n\nbody text\n").hexdigest(),
        "raster_path": None,
        "raw_response_sha256": "1" * 64,
        "raw_response_path": "raw_responses/document-001/page-000002.json",
    }


def test_document_failure_allows_unmaterialized_versioned_s3_source() -> None:
    source = valid_local_source_object_data()
    failure = DocumentExtractionFailure.model_validate(
        {
            **source,
            "schema_version": 1,
            "run_id": "run-1",
            "config_sha256": "b" * 64,
            "pipeline_fingerprint": "c" * 64,
            "failure_stage": "pdf_open",
            "failed_at": NOW,
            "error_type": "PdfOpenError",
            "error_message": "invalid PDF",
        },
        strict=True,
    )

    assert failure.failure_stage == "pdf_open"


def test_persist_failure_requires_prior_inference_attempt() -> None:
    data = valid_page_provenance_data()
    failure = PageExtractionFailure.model_validate(
        {
            **data,
            "failure_stage": "persist",
            "failed_at": NOW,
            "inference_attempt_count": 1,
            "retryable": False,
            "error_type": "OSError",
            "error_message": "disk full",
            "last_inference_request_id": "req-1",
        },
        strict=True,
    )
    assert failure.inference_attempt_count == 1

    with pytest.raises(ValidationError, match="require an attempt"):
        PageExtractionFailure.model_validate(
            {**failure.model_dump(), "inference_attempt_count": 0}, strict=True
        )


def test_valid_local_configuration_is_fully_typed() -> None:
    config = PipelineConfig.model_validate(valid_config_data(), strict=True)

    assert isinstance(config.source, LocalSourceConfig)
    assert config.vllm.revision == MODEL_REVISION
    assert config.vllm.speculative_decoding.method == "mtp"
    assert config.vllm.speculative_decoding.num_speculative_tokens == 1
    assert config.benchmark is None


@pytest.mark.parametrize(
    ("source_root", "output_root"),
    [
        ("/data", "/data"),
        ("/data/pdfs", "/data/pdfs/output"),
        ("/data/output/source", "/data/output"),
        ("/data/pdfs/../pdfs", "/data/pdfs/output"),
    ],
)
def test_local_source_and_output_roots_must_not_overlap(
    source_root: str,
    output_root: str,
) -> None:
    data = valid_config_data()
    data["source"]["root"] = source_root
    data["output"]["root"] = output_root

    with pytest.raises(ValidationError, match="must not overlap"):
        PipelineConfig.model_validate(data, strict=True)


def test_valid_versioned_s3_source() -> None:
    data = valid_config_data()
    data["source"] = {
        "type": "s3",
        "bucket": "document-training-data",
        "prefix": "pdf/contracts/",
        "include_glob": "**/*.pdf",
        "region": "eu-central-1",
        "dataset_version": "contracts-2026-08-05",
        "require_object_version_ids": True,
    }

    config = PipelineConfig.model_validate(data, strict=True)

    assert isinstance(config.source, S3SourceConfig)
    assert config.source.require_object_version_ids is True


def test_load_config_reads_yaml_mapping_without_coercion(tmp_path: Any) -> None:
    path = tmp_path / "pipeline.yaml"
    path.write_text(
        (
            """
schema_version: 1
source:
  type: local
  root: /data/pdfs
  include_glob: '**/*.pdf'
  dataset_version: contracts-v1
  require_content_sha256: true
output:
  root: /data/output
  retain_page_images: false
  parquet_compression: zstd
  write_batch_rows: 64
run: {run_id: run-1, fail_fast: false, resume: true}
raster:
  dpi: 200
  max_side_pixels: 4096
  max_pixels: 12000000
  max_pages_per_document: 500
  max_pdf_bytes: 1000000000
  max_cached_documents_per_process: 2
  image_format: png
  jpeg_quality: null
  draw_annotations: true
  reject_xfa: true
vllm:
  endpoint: http://127.0.0.1:8000
  model: zai-org/GLM-OCR
  served_model_name: glm-ocr
  revision: cccccccccccccccccccccccccccccccccccccccc
  engine_version: 0.26.0
  container_image: """
            + CONTAINER_IMAGE
            + """
  max_model_len: 32768
  max_num_seqs: 16
  gpu_memory_utilization: 0.9
  prompt: 'Text Recognition:'
  request_timeout_seconds: 120.0
  api_key_env: VLLM_API_KEY
  sampling:
    {temperature: 0.0, top_p: 1.0, top_k: 1, repetition_penalty: 1.0,
     max_tokens: 8192, seed: 0}
  retry:
    {max_attempts: 4, initial_backoff_seconds: 0.5, max_backoff_seconds: 8.0,
     backoff_multiplier: 2.0, jitter_fraction: 0.2}
  speculative_decoding: {method: mtp, num_speculative_tokens: 1}
concurrency:
  max_active_documents: 4
  renderer_processes: 4
  max_inflight_pages_global: 16
  max_inflight_pages_per_document: 4
"""
        ).lstrip(),
        encoding="utf-8",
    )

    assert load_config(path).run.run_id == "run-1"


@pytest.mark.parametrize(
    "duplicate",
    [
        "schema_version: 1\nschema_version: 1\n",
        "schema_version: 1\nsource:\n  type: local\n  type: local\n",
    ],
)
def test_load_config_rejects_duplicate_yaml_keys(tmp_path: Any, duplicate: str) -> None:
    path = tmp_path / "duplicate.yaml"
    path.write_text(duplicate, encoding="utf-8")

    with pytest.raises(yaml.constructor.ConstructorError, match="duplicate key"):
        load_config(path)


@pytest.mark.parametrize("document", ["", "- just-a-list\n- not-a-mapping\n"])
def test_load_config_rejects_non_mapping_yaml(tmp_path: Any, document: str) -> None:
    path = tmp_path / "invalid.yaml"
    path.write_text(document, encoding="utf-8")

    with pytest.raises(ValueError, match="YAML mapping"):
        load_config(path)


@pytest.mark.parametrize("missing", ["source", "output", "run", "vllm"])
def test_reproducibility_boundaries_are_required(missing: str) -> None:
    data = valid_config_data()
    del data[missing]

    with pytest.raises(ValidationError):
        PipelineConfig.model_validate(data, strict=True)


def test_model_revision_is_required_and_must_be_commit_sha() -> None:
    data = valid_config_data()
    data["vllm"]["revision"] = "main"

    with pytest.raises(ValidationError, match="immutable 40-character Git commit SHA"):
        PipelineConfig.model_validate(data, strict=True)

    del data["vllm"]["revision"]
    with pytest.raises(ValidationError):
        PipelineConfig.model_validate(data, strict=True)


def test_unknown_fields_are_forbidden_at_every_level() -> None:
    data = valid_config_data()
    data["vllm"]["sampling"]["beam_width"] = 2

    with pytest.raises(ValidationError, match="Extra inputs are not permitted"):
        PipelineConfig.model_validate(data, strict=True)


def test_strict_types_do_not_coerce_yaml_strings() -> None:
    data = valid_config_data()
    data["concurrency"]["renderer_processes"] = "4"

    with pytest.raises(ValidationError, match="valid integer"):
        PipelineConfig.model_validate(data, strict=True)


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("dpi", 0),
        ("max_side_pixels", -1),
        ("max_pixels", 0),
        ("max_pages_per_document", 0),
        ("max_pdf_bytes", 0),
        ("max_cached_documents_per_process", 0),
    ],
)
def test_raster_limits_must_be_positive(field: str, value: int) -> None:
    data = valid_config_data()
    data["raster"][field] = value

    with pytest.raises(ValidationError, match="greater than 0"):
        PipelineConfig.model_validate(data, strict=True)


def test_jpeg_quality_is_required_only_for_jpeg() -> None:
    jpeg = valid_config_data()
    jpeg["raster"]["image_format"] = "jpeg"
    with pytest.raises(ValidationError, match="jpeg_quality is required"):
        PipelineConfig.model_validate(jpeg, strict=True)

    jpeg["raster"]["jpeg_quality"] = 90
    assert PipelineConfig.model_validate(jpeg, strict=True).raster.jpeg_quality == 90

    png = valid_config_data()
    png["raster"]["jpeg_quality"] = 90
    with pytest.raises(ValidationError, match="only valid"):
        PipelineConfig.model_validate(png, strict=True)


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("temperature", 0.1, "temperature must be 0"),
        ("top_p", 0.0, "greater than 0"),
        ("top_p", 1.1, "less than or equal to 1"),
        ("top_k", 0, "greater than 0"),
        ("repetition_penalty", 0.0, "greater than 0"),
    ],
)
def test_sampling_contract_is_bounded_and_deterministic(
    field: str, value: float, message: str
) -> None:
    data = valid_config_data()
    data["vllm"]["sampling"][field] = value

    with pytest.raises(ValidationError, match=message):
        PipelineConfig.model_validate(data, strict=True)


@pytest.mark.parametrize("value", [math.inf, -math.inf, math.nan])
def test_nonfinite_configuration_numbers_are_rejected(value: float) -> None:
    data = valid_config_data()
    data["vllm"]["request_timeout_seconds"] = value

    with pytest.raises(ValidationError, match="finite number"):
        PipelineConfig.model_validate(data, strict=True)


def test_prompt_and_mtp_depth_are_fixed_for_raw_glm_ocr() -> None:
    data = valid_config_data()
    data["vllm"]["prompt"] = "OCR this"
    with pytest.raises(ValidationError):
        PipelineConfig.model_validate(data, strict=True)

    data = valid_config_data()
    data["vllm"]["speculative_decoding"]["num_speculative_tokens"] = 3
    with pytest.raises(ValidationError):
        PipelineConfig.model_validate(data, strict=True)


def test_endpoint_is_base_url_not_completion_path() -> None:
    data = valid_config_data()
    data["vllm"]["endpoint"] = "http://127.0.0.1:8000/v1/chat/completions"

    with pytest.raises(ValidationError, match="base server URL"):
        PipelineConfig.model_validate(data, strict=True)


@pytest.mark.parametrize(
    ("endpoint", "message"),
    [
        ("http://operator:secret@127.0.0.1:8000", "must not contain credentials"),
        ("http://127.0.0.1:99999", "invalid host or port"),
    ],
)
def test_endpoint_rejects_credentials_and_invalid_ports(endpoint: str, message: str) -> None:
    data = valid_config_data()
    data["vllm"]["endpoint"] = endpoint

    with pytest.raises(ValidationError, match=message):
        PipelineConfig.model_validate(data, strict=True)


def test_per_document_inference_cap_cannot_exceed_global_cap() -> None:
    data = valid_config_data()
    data["concurrency"]["max_inflight_pages_per_document"] = 17

    with pytest.raises(ValidationError, match="must not exceed"):
        PipelineConfig.model_validate(data, strict=True)


def test_benchmark_sweep_requires_unique_valid_points() -> None:
    data = valid_config_data()
    data["benchmark"] = {
        "points": [
            {
                "max_inflight_pages_global": 8,
                "max_inflight_pages_per_document": 2,
            },
            {
                "max_inflight_pages_global": 8,
                "max_inflight_pages_per_document": 2,
            },
        ],
        "warmup_pages": 4,
        "measured_pages": 100,
        "repetitions": 3,
    }
    with pytest.raises(ValidationError, match="must be unique"):
        PipelineConfig.model_validate(data, strict=True)

    data["benchmark"]["points"] = [
        {
            "max_inflight_pages_global": 2,
            "max_inflight_pages_per_document": 3,
        }
    ]
    with pytest.raises(ValidationError, match="must not exceed"):
        PipelineConfig.model_validate(data, strict=True)


def test_source_version_guards_cannot_be_disabled() -> None:
    local = valid_config_data()
    local["source"]["require_content_sha256"] = False
    with pytest.raises(ValidationError):
        PipelineConfig.model_validate(local, strict=True)

    s3 = valid_config_data()
    s3["source"] = {
        "type": "s3",
        "bucket": "document-training-data",
        "prefix": "pdf/",
        "include_glob": "*.pdf",
        "region": "eu-central-1",
        "dataset_version": "v1",
        "require_object_version_ids": False,
    }
    with pytest.raises(ValidationError):
        PipelineConfig.model_validate(s3, strict=True)


def test_output_is_an_explicit_absolute_local_path() -> None:
    data = valid_config_data()
    data["output"]["root"] = "s3://bucket/output"

    with pytest.raises(ValidationError, match="absolute local path"):
        PipelineConfig.model_validate(data, strict=True)

    data = valid_config_data()
    data["source"] = {
        "type": "s3",
        "bucket": "document-training-data",
        "prefix": "pdf/",
        "include_glob": "*.pdf",
        "region": "eu-central-1",
        "dataset_version": "v1",
        "require_object_version_ids": True,
    }
    data["output"]["root"] = "/tmp/.."
    with pytest.raises(ValidationError, match="must not be the filesystem root"):
        PipelineConfig.model_validate(data, strict=True)


def test_local_source_and_output_overlap_is_detected_through_symlinks(tmp_path: Any) -> None:
    source = tmp_path / "source"
    nested_output = source / "generated"
    source.mkdir()
    nested_output.mkdir()
    output_alias = tmp_path / "output-alias"
    output_alias.symlink_to(nested_output, target_is_directory=True)
    data = valid_config_data()
    data["source"]["root"] = str(source)
    data["output"]["root"] = str(output_alias)

    with pytest.raises(ValidationError, match="must not overlap"):
        PipelineConfig.model_validate(data, strict=True)


def test_local_source_object_round_trips_and_is_frozen() -> None:
    source = SourceObject.model_validate(valid_local_source_object_data(), strict=True)
    restored = SourceObject.model_validate_json(source.model_dump_json())

    assert restored == source
    with pytest.raises(ValidationError, match="frozen"):
        restored.source_uri = "file:///different.pdf"  # type: ignore[misc]


def test_source_object_rejects_cross_source_or_missing_provenance() -> None:
    local = valid_local_source_object_data()
    local["s3_bucket"] = "not-allowed"
    with pytest.raises(ValidationError, match="must not contain S3 provenance"):
        SourceObject.model_validate(local, strict=True)

    s3 = {
        "document_id": "document-002",
        "source_type": "s3",
        "source_uri": "s3://document-training-data/pdf/a.pdf",
        "source_dataset_version": "v1",
        "source_object_version": "version-123",
        "source_size_bytes": 2048,
        "source_last_modified": NOW,
        "source_sha256": None,
        "s3_bucket": "document-training-data",
        "s3_key": "pdf/a.pdf",
        "s3_version_id": "different-version",
        "s3_etag": '"etag"',
    }
    with pytest.raises(ValidationError, match="must equal"):
        SourceObject.model_validate(s3, strict=True)


def test_s3_checksum_provenance_is_preserved() -> None:
    data = {
        "document_id": "document-002",
        "source_type": "s3",
        "source_uri": "s3://document-training-data/pdf/a.pdf",
        "source_dataset_version": "v1",
        "source_object_version": "version-123",
        "source_size_bytes": 2048,
        "source_last_modified": NOW,
        "source_sha256": None,
        "s3_bucket": "document-training-data",
        "s3_key": "pdf/a.pdf",
        "s3_version_id": "version-123",
        "s3_etag": '"etag"',
        "s3_checksum_crc32": "crc32-base64",
        "s3_checksum_crc32c": "crc32c-base64",
        "s3_checksum_sha1": "sha1-base64",
        "s3_checksum_sha256": "sha256-base64",
        "s3_checksum_crc64nvme": "crc64-base64",
        "s3_checksum_type": "FULL_OBJECT",
    }

    dumped = SourceObject.model_validate(data, strict=True).model_dump(mode="json")

    assert dumped["s3_checksum_crc64nvme"] == "crc64-base64"
    assert dumped["s3_checksum_type"] == "FULL_OBJECT"


def test_page_result_is_flat_frozen_and_preserves_raw_ocr_exactly() -> None:
    data = valid_page_record_data()
    record = PageExtractionRecord.model_validate(data, strict=True)

    dumped = record.model_dump(mode="python")
    assert record.raw_ocr_text == "  Heading\n\nbody text\n"
    assert dumped["source_uri"] == "file:///data/pdfs/a.pdf"
    assert dumped["raster_width_px"] == 1700
    assert dumped["inference_model_revision"] == MODEL_REVISION
    assert dumped["pipeline_fingerprint"] == "f" * 64
    assert all(not isinstance(value, dict) for value in dumped.values())
    with pytest.raises(ValidationError, match="frozen"):
        record.raw_ocr_text = "changed"  # type: ignore[misc]


def test_page_result_rejects_mismatched_text_hash_or_unsafe_response_path() -> None:
    bad_hash = valid_page_record_data()
    bad_hash["raw_ocr_text_sha256"] = "0" * 64
    with pytest.raises(ValidationError, match="does not match"):
        PageExtractionRecord.model_validate(bad_hash, strict=True)

    unsafe_path = valid_page_record_data()
    unsafe_path["raw_response_path"] = "../outside.json"
    with pytest.raises(ValidationError, match="safe relative POSIX path"):
        PageExtractionRecord.model_validate(unsafe_path, strict=True)


def test_raster_mime_type_must_match_encoding() -> None:
    data = valid_page_record_data()
    data["raster_mime_type"] = "image/jpeg"

    with pytest.raises(ValidationError, match="must match"):
        PageExtractionRecord.model_validate(data, strict=True)


def test_page_identity_and_timestamps_are_consistent() -> None:
    bad_page = valid_page_record_data()
    bad_page["page_number"] = 3
    with pytest.raises(ValidationError, match="page_number must equal"):
        PageExtractionRecord.model_validate(bad_page, strict=True)

    bad_time = valid_page_record_data()
    bad_time["extraction_started_at"] = LATER
    bad_time["extraction_completed_at"] = NOW
    with pytest.raises(ValidationError, match="must not precede"):
        PageExtractionRecord.model_validate(bad_time, strict=True)

    nonfinite_timing = valid_page_record_data()
    nonfinite_timing["render_duration_ms"] = math.inf
    with pytest.raises(ValidationError, match="finite number"):
        PageExtractionRecord.model_validate(nonfinite_timing, strict=True)

    bad_finish_reason = valid_page_record_data()
    bad_finish_reason["inference_finish_reason"] = "length"
    with pytest.raises(ValidationError):
        PageExtractionRecord.model_validate(bad_finish_reason, strict=True)


def test_inference_attempt_requires_outcome_specific_fields() -> None:
    attempt = {
        **valid_page_provenance_data(),
        "attempt_number": 1,
        "attempt_started_at": NOW,
        "attempt_completed_at": LATER,
        "attempt_duration_ms": 100.0,
        "attempt_outcome": "retryable_error",
        "inference_request_id": None,
        "http_status_code": 503,
        "retry_after_seconds": 1.0,
        "error_type": "HTTPStatusError",
        "error_message": "service unavailable",
    }
    assert InferenceAttempt.model_validate(attempt, strict=True).http_status_code == 503

    attempt["error_message"] = None
    with pytest.raises(ValidationError, match="requires error_type and error_message"):
        InferenceAttempt.model_validate(attempt, strict=True)


def test_page_failure_attempt_count_matches_last_request_identity() -> None:
    failure = {
        **valid_page_provenance_data(),
        "failure_stage": "render",
        "failed_at": NOW,
        "inference_attempt_count": 0,
        "retryable": False,
        "error_type": "PdfiumError",
        "error_message": "page render failed",
        "last_inference_request_id": None,
    }
    assert PageExtractionFailure.model_validate(failure, strict=True).failure_stage == "render"

    failure["inference_attempt_count"] = 1
    with pytest.raises(ValidationError, match="present exactly when attempt history exists"):
        PageExtractionFailure.model_validate(failure, strict=True)

    failure["last_inference_request_id"] = "prior-request-1"
    assert PageExtractionFailure.model_validate(failure, strict=True).inference_attempt_count == 1


@pytest.mark.parametrize("prior_attempt_count", [0, 3])
def test_raster_validation_failure_allows_prior_attempts_but_no_new_request(
    prior_attempt_count: int,
) -> None:
    failure = {
        **valid_page_provenance_data(),
        "failure_stage": "raster_validate",
        "failed_at": NOW,
        "inference_attempt_count": prior_attempt_count,
        "retryable": False,
        "error_type": "VllmRasterError",
        "error_message": "raster SHA-256 changed before submission",
        "last_inference_request_id": (
            None if prior_attempt_count == 0 else f"prior-request-{prior_attempt_count}"
        ),
    }
    parsed = PageExtractionFailure.model_validate(failure, strict=True)
    assert parsed.inference_attempt_count == prior_attempt_count

    inconsistent_request_id = "request-without-attempt" if prior_attempt_count == 0 else None
    with pytest.raises(ValidationError, match="present exactly when attempt history exists"):
        PageExtractionFailure.model_validate(
            {**failure, "last_inference_request_id": inconsistent_request_id},
            strict=True,
        )
