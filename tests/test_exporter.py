from __future__ import annotations

import hashlib
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

import pyarrow as pa
import pyarrow.parquet as pq
import pytest
from test_config import SERVER_CONTRACT_SHA256, valid_config_data

from document_ocr.atomic import AtomicConflictError
from document_ocr.config import PipelineConfig
from document_ocr.exporter import (
    DatasetPublicationError,
    _validate_raster,
    arrow_schema,
    publish_complete_dataset,
)
from document_ocr.hashing import (
    canonical_json_bytes,
    canonical_json_sha256,
    sha256_bytes,
    sha256_file,
)
from document_ocr.ledger import ExtractionLedger, LedgerConflictError
from document_ocr.models import (
    InferenceAttempt,
    PageExtractionFailure,
    PageExtractionRecord,
    SourceObject,
)

NOW = datetime(2026, 8, 5, 10, 0, tzinfo=UTC)
MODEL_REVISION = "c" * 40
PIPELINE_FINGERPRINT = "b" * 64
INVENTORY_SHA256 = "d" * 64
SOURCE_SHA256 = "e" * 64


def _run_config_data() -> dict[str, Any]:
    raw = {
        **valid_config_data(),
        "run": {"run_id": "run-1", "fail_fast": False, "resume": True},
    }
    return PipelineConfig.model_validate(raw, strict=True).model_dump(mode="json")


CONFIG_SHA256 = canonical_json_sha256(_run_config_data())


def _page_provenance(page_index: int, page_count: int) -> dict[str, Any]:
    return {
        "document_id": "document-001",
        "source_type": "local",
        "source_uri": "file:///data/pdfs/document.pdf",
        "source_dataset_version": "snapshot-v1",
        "source_object_version": SOURCE_SHA256,
        "source_size_bytes": 4096,
        "source_last_modified": NOW,
        "source_sha256": SOURCE_SHA256,
        "local_canonical_path": "/data/pdfs/document.pdf",
        "local_relative_key": "document.pdf",
        "local_device": 42,
        "local_inode": 123,
        "local_mtime_ns": 1_754_389_200_000_000_000,
        "s3_bucket": None,
        "s3_key": None,
        "s3_version_id": None,
        "s3_etag": None,
        "s3_checksum_crc32": None,
        "s3_checksum_crc32c": None,
        "s3_checksum_sha1": None,
        "s3_checksum_sha256": None,
        "s3_checksum_crc64nvme": None,
        "s3_checksum_type": None,
        "schema_version": 1,
        "run_id": "run-1",
        "extraction_id": f"extraction-{page_index}",
        "page_id": f"page-{page_index}",
        "config_sha256": CONFIG_SHA256,
        "pipeline_fingerprint": PIPELINE_FINGERPRINT,
        "document_page_count": page_count,
        "page_index": page_index,
        "page_number": page_index + 1,
    }


def _attempt(page_index: int, page_count: int) -> InferenceAttempt:
    started = NOW + timedelta(seconds=page_index)
    return InferenceAttempt(
        **_page_provenance(page_index, page_count),
        attempt_number=1,
        attempt_started_at=started,
        attempt_completed_at=started + timedelta(milliseconds=25),
        attempt_duration_ms=25.0,
        attempt_outcome="success",
        inference_request_id=f"request-{page_index}",
        inference_server_request_id=f"server-request-{page_index}",
        http_status_code=200,
        retry_after_seconds=None,
        error_type=None,
        error_message=None,
    )


def _inference_failure(
    page_index: int,
    page_count: int,
    *,
    attempt_count: int,
    last_request_id: str,
) -> PageExtractionFailure:
    return PageExtractionFailure.model_validate(
        {
            **_page_provenance(page_index, page_count),
            "failure_stage": "inference",
            "failed_at": NOW,
            "inference_attempt_count": attempt_count,
            "retryable": True,
            "error_type": "HTTPStatusError",
            "error_message": "vLLM returned HTTP 503",
            "last_inference_request_id": last_request_id,
        },
        strict=True,
    )


def _record(page_index: int, page_count: int) -> PageExtractionRecord:
    text = f"  page {page_index + 1}\n\nbody\n"
    raw_response_sha256 = sha256_bytes(_raw_response_payload(page_index))
    started = NOW + timedelta(seconds=page_index)
    return PageExtractionRecord(
        **_page_provenance(page_index, page_count),
        renderer_name="pypdfium2",
        renderer_version="5.12.0",
        pdfium_version="chromium/7000",
        page_width_points=612.0,
        page_height_points=792.0,
        page_rotation_degrees=0,
        requested_dpi=200,
        effective_dpi=200.0,
        render_scale=200.0 / 72.0,
        raster_width_px=1700,
        raster_height_px=2200,
        raster_image_format="png",
        raster_mime_type="image/png",
        draw_annotations=True,
        pdf_form_type="none",
        raster_size_bytes=400_000,
        raster_sha256="f" * 64,
        render_duration_ms=31.5,
        inference_endpoint="http://127.0.0.1:8000",
        inference_model="zai-org/GLM-OCR",
        inference_served_model_name="glm-ocr",
        inference_model_revision=MODEL_REVISION,
        inference_server_engine="vllm",
        inference_server_engine_version="0.26.0",
        inference_server_image=(
            "vllm/vllm-openai:v0.26.0@sha256:"
            "ffb2d59b1c059a5bd8d781320c9f5189de8293693b7d95da54befddaa54abf52"
        ),
        inference_server_contract_sha256=SERVER_CONTRACT_SHA256,
        inference_speculative_method="mtp",
        inference_num_speculative_tokens=1,
        inference_prompt="Text Recognition:",
        inference_prompt_sha256=sha256_bytes(b"Text Recognition:"),
        inference_temperature=0.0,
        inference_top_p=1.0,
        inference_top_k=1,
        inference_repetition_penalty=1.0,
        inference_max_tokens=8192,
        inference_seed=0,
        inference_request_id=f"request-{page_index}",
        inference_server_request_id=f"server-request-{page_index}",
        inference_finish_reason="stop",
        inference_prompt_tokens=128,
        inference_completion_tokens=256,
        inference_attempt_count=1,
        inference_duration_ms=225.0,
        extraction_started_at=started,
        extraction_completed_at=started + timedelta(milliseconds=265),
        queue_duration_ms=4.0,
        persist_duration_ms=4.5,
        total_duration_ms=265.0,
        raster_path=None,
        raw_ocr_text=text,
        raw_ocr_text_sha256=hashlib.sha256(text.encode()).hexdigest(),
        raw_response_sha256=raw_response_sha256,
        raw_response_path=(
            f"raw-responses/document-001/page-{page_index}/{raw_response_sha256}.json"
        ),
    )


def _raw_response_payload(page_index: int) -> bytes:
    return (
        canonical_json_bytes(
            {"id": f"request-{page_index}", "page_index": page_index, "text": "raw response"}
        )
        + b"\n"
    )


def _write_raw_response(run_root: Path, page_index: int, page_count: int = 1) -> Path:
    record = _record(page_index, page_count)
    path = run_root / record.raw_response_path
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(_raw_response_payload(page_index))
    return path


def test_retained_raster_validation_binds_path_size_and_hash(tmp_path: Path) -> None:
    payload = b"\x89PNG\r\n\x1a\nretained raster fixture"
    digest = sha256_bytes(payload)
    raw = _record(0, 1).model_dump(mode="python")
    raster_path = f"page-images/document-001/page-0/{digest}.png"
    record = PageExtractionRecord.model_validate(
        {
            **raw,
            "raster_path": raster_path,
            "raster_sha256": digest,
            "raster_size_bytes": len(payload),
        },
        strict=True,
    )
    target = tmp_path / raster_path
    target.parent.mkdir(parents=True)
    target.write_bytes(payload)

    assert _validate_raster(tmp_path, record) == len(payload)
    target.write_bytes(b"x" * len(payload))
    with pytest.raises(DatasetPublicationError, match="retained raster artifact hash mismatch"):
        _validate_raster(tmp_path, record)


async def _complete_ledger(
    run_root: Path,
    *,
    page_count: int = 3,
    complete: bool = True,
) -> ExtractionLedger:
    ledger = ExtractionLedger(run_root / "state.sqlite3")
    await ledger.open()
    await ledger.initialize_run(
        run_id="run-1",
        config_sha256=CONFIG_SHA256,
        pipeline_fingerprint=PIPELINE_FINGERPRINT,
        config=_run_config_data(),
        inventory_sha256=INVENTORY_SHA256,
        inventory_document_count=1,
        provenance={"source_tree_sha256": "3" * 64},
        resume=False,
    )
    source = {
        key: value
        for key, value in _page_provenance(0, page_count).items()
        if key
        in {
            "document_id",
            "source_type",
            "source_uri",
            "source_dataset_version",
            "source_object_version",
            "source_size_bytes",
            "source_last_modified",
            "source_sha256",
            "local_canonical_path",
            "local_relative_key",
            "local_device",
            "local_inode",
            "local_mtime_ns",
            "s3_bucket",
            "s3_key",
            "s3_version_id",
            "s3_etag",
            "s3_checksum_crc32",
            "s3_checksum_crc32c",
            "s3_checksum_sha1",
            "s3_checksum_sha256",
            "s3_checksum_crc64nvme",
            "s3_checksum_type",
        }
    }
    await ledger.register_document(
        run_id="run-1",
        inventory_source=SourceObject.model_validate(source, strict=True),
        source_sha256=SOURCE_SHA256,
        page_count=page_count,
    )
    await ledger.register_pages(
        run_id="run-1",
        document_id="document-001",
        pages=tuple(
            (f"extraction-{page_index}", f"page-{page_index}", page_index)
            for page_index in range(page_count)
        ),
    )
    if complete:
        for page_index in range(page_count):
            attempt = _attempt(page_index, page_count)
            record = _record(page_index, page_count)
            _write_raw_response(run_root, page_index, page_count)
            await ledger.record_success(result=record, attempts=(attempt,))
    return ledger


@pytest.mark.asyncio
@pytest.mark.parametrize("drift", ["config_hash", "config_run_id"])
async def test_initialize_run_rejects_unbound_configuration(tmp_path: Path, drift: str) -> None:
    config = _run_config_data()
    config_sha256 = CONFIG_SHA256
    if drift == "config_hash":
        config_sha256 = "9" * 64
        message = "config_sha256"
    else:
        config["run"] = {"run_id": "different-run", "fail_fast": False, "resume": True}
        config_sha256 = canonical_json_sha256(config)
        message = "run_id"

    async with ExtractionLedger(tmp_path / "run" / "state.sqlite3") as ledger:
        with pytest.raises(LedgerConflictError, match=message):
            await ledger.initialize_run(
                run_id="run-1",
                config_sha256=config_sha256,
                pipeline_fingerprint=PIPELINE_FINGERPRINT,
                config=config,
                inventory_sha256=INVENTORY_SHA256,
                inventory_document_count=1,
                provenance={"source_tree_sha256": "3" * 64},
                resume=False,
            )


def test_arrow_schemas_preserve_model_order_types_nullability_and_metadata() -> None:
    pages = arrow_schema(PageExtractionRecord)
    attempts = arrow_schema(InferenceAttempt)
    failures = arrow_schema(PageExtractionFailure)

    assert pages.names == list(PageExtractionRecord.model_fields)
    assert attempts.names == list(InferenceAttempt.model_fields)
    assert failures.names == list(PageExtractionFailure.model_fields)
    assert pages.field("raw_ocr_text").type == pa.large_string()
    assert not pages.field("raw_ocr_text").nullable
    assert failures.field("error_message").type == pa.large_string()
    assert attempts.field("error_message").type == pa.large_string()
    assert attempts.field("error_message").nullable
    assert pages.field("source_last_modified").type == pa.timestamp("us", tz="UTC")
    assert pages.field("page_index").type == pa.int64()
    assert pages.field("draw_annotations").type == pa.bool_()
    assert pages.field("s3_version_id").nullable
    assert pages.field("local_canonical_path").type != pa.large_string()
    assert pages.metadata == {
        b"record_model": b"document_ocr.models.PageExtractionRecord",
        b"dataset_schema_version": b"1",
    }


@pytest.mark.asyncio
async def test_streaming_parquet_publication_is_manifest_last_verified_and_idempotent(
    tmp_path: Path,
) -> None:
    run_root = tmp_path / "run"
    ledger = await _complete_ledger(run_root)
    try:
        first = await publish_complete_dataset(
            ledger=ledger,
            run_id="run-1",
            run_root=run_root,
            batch_rows=2,
            compression="zstd",
            retain_page_images=False,
        )

        manifest_bytes = first.manifest_path.read_bytes()
        assert manifest_bytes == canonical_json_bytes(first.manifest) + b"\n"
        assert first.manifest_sha256 == sha256_bytes(manifest_bytes)
        assert first.manifest["summary"]["successful_pages"] == 3
        assert [item["record_model"] for item in first.manifest["files"]] == [
            "PageExtractionRecord",
            "InferenceAttempt",
        ]
        assert [item["rows"] for item in first.manifest["files"]] == [3, 3]
        assert first.manifest["raw_responses"] == {
            "artifacts": 3,
            "bytes": sum(len(_raw_response_payload(index)) for index in range(3)),
        }
        assert first.manifest["page_images"] == {
            "retained": False,
            "artifacts": 0,
            "bytes": 0,
        }
        assert [item["raster_artifact_count"] for item in first.manifest["files"]] == [0, 0]

        for item in first.manifest["files"]:
            path = run_root / item["path"]
            assert path.is_file()
            assert path.stat().st_size == item["bytes"]
            assert sha256_file(path) == item["sha256"]
            assert path.name.endswith(f"-{item['sha256'][:16]}.parquet")
            parquet = pq.ParquetFile(path)
            assert parquet.metadata.num_rows == item["rows"]
            assert (
                sha256_bytes(parquet.schema_arrow.serialize().to_pybytes())
                == item["arrow_schema_sha256"]
            )

        pages_path = run_root / first.manifest["files"][0]["path"]
        pages_file = pq.ParquetFile(pages_path)
        assert pages_file.metadata.num_row_groups == 2
        assert pages_file.schema_arrow.equals(
            arrow_schema(PageExtractionRecord), check_metadata=True
        )
        pages = pages_file.read(columns=["page_index", "raw_ocr_text"]).to_pylist()
        assert pages == [
            {"page_index": 0, "raw_ocr_text": "  page 1\n\nbody\n"},
            {"page_index": 1, "raw_ocr_text": "  page 2\n\nbody\n"},
            {"page_index": 2, "raw_ocr_text": "  page 3\n\nbody\n"},
        ]

        second = await publish_complete_dataset(
            ledger=ledger,
            run_id="run-1",
            run_root=run_root,
            batch_rows=2,
            compression="zstd",
            retain_page_images=False,
        )
        assert second == first
        assert len(list((run_root / "dataset").glob("*.parquet"))) == 2
        assert not list((run_root / "dataset").glob(".staging-*"))

        await ledger.mark_complete(
            run_id="run-1",
            manifest_sha256=first.manifest_sha256,
            manifest=first.manifest,
        )
        await ledger.mark_complete(
            run_id="run-1",
            manifest_sha256=first.manifest_sha256,
            manifest=first.manifest,
        )
    finally:
        await ledger.close()


@pytest.mark.asyncio
async def test_conflicting_manifest_is_rejected_without_overwrite(tmp_path: Path) -> None:
    run_root = tmp_path / "run"
    ledger = await _complete_ledger(run_root, page_count=1)
    try:
        manifest_path = run_root / "dataset" / "manifest.json"
        manifest_path.parent.mkdir(parents=True)
        original = b'{"foreign":"manifest"}\n'
        manifest_path.write_bytes(original)

        with pytest.raises(AtomicConflictError):
            await publish_complete_dataset(
                ledger=ledger,
                run_id="run-1",
                run_root=run_root,
                batch_rows=1,
                compression="none",
                retain_page_images=False,
            )

        assert manifest_path.read_bytes() == original
    finally:
        await ledger.close()


@pytest.mark.asyncio
async def test_ledger_rejects_manifest_digest_or_run_identity_drift(tmp_path: Path) -> None:
    run_root = tmp_path / "run"
    ledger = await _complete_ledger(run_root, page_count=1)
    try:
        published = await publish_complete_dataset(
            ledger=ledger,
            run_id="run-1",
            run_root=run_root,
            batch_rows=1,
            compression="zstd",
            retain_page_images=False,
        )

        with pytest.raises(LedgerConflictError, match="manifest_sha256 does not match"):
            await ledger.mark_complete(
                run_id="run-1",
                manifest_sha256="9" * 64,
                manifest=published.manifest,
            )

        changed_identity = {**published.manifest, "pipeline_fingerprint": "9" * 64}
        changed_digest = sha256_bytes(canonical_json_bytes(changed_identity) + b"\n")
        with pytest.raises(LedgerConflictError, match="does not match the initialized run"):
            await ledger.mark_complete(
                run_id="run-1",
                manifest_sha256=changed_digest,
                manifest=changed_identity,
            )

        assert await ledger.run_contract("run-1") == {
            "run_id": "run-1",
            "config_sha256": CONFIG_SHA256,
            "pipeline_fingerprint": PIPELINE_FINGERPRINT,
            "inventory_sha256": INVENTORY_SHA256,
            "inventory_document_count": 1,
        }
    finally:
        await ledger.close()


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "manifest_failure",
    ["malformed", "summary_drift", "missing_on_disk", "different_on_disk"],
)
async def test_ledger_completion_requires_exact_published_manifest(
    tmp_path: Path, manifest_failure: str
) -> None:
    run_root = tmp_path / "run"
    ledger = await _complete_ledger(run_root, page_count=1)
    try:
        published = await publish_complete_dataset(
            ledger=ledger,
            run_id="run-1",
            run_root=run_root,
            batch_rows=1,
            compression="zstd",
            retain_page_images=False,
        )
        candidate = published.manifest
        candidate_digest = published.manifest_sha256

        if manifest_failure == "malformed":
            candidate = {
                "run_id": "run-1",
                "config_sha256": CONFIG_SHA256,
                "pipeline_fingerprint": PIPELINE_FINGERPRINT,
                "inventory_sha256": INVENTORY_SHA256,
            }
            candidate_digest = sha256_bytes(canonical_json_bytes(candidate) + b"\n")
            published.manifest_path.write_bytes(canonical_json_bytes(candidate) + b"\n")
        elif manifest_failure == "summary_drift":
            candidate = {
                **published.manifest,
                "summary": {**published.manifest["summary"], "successful_pages": 999},
            }
            candidate_digest = sha256_bytes(canonical_json_bytes(candidate) + b"\n")
            published.manifest_path.write_bytes(canonical_json_bytes(candidate) + b"\n")
        elif manifest_failure == "missing_on_disk":
            published.manifest_path.unlink()
        else:
            published.manifest_path.write_bytes(b"{}\n")

        with pytest.raises(LedgerConflictError, match="manifest"):
            await ledger.mark_complete(
                run_id="run-1",
                manifest_sha256=candidate_digest,
                manifest=candidate,
            )

        cursor = await ledger.connection.execute("SELECT status FROM runs WHERE run_id = 'run-1'")
        async with cursor:
            row = await cursor.fetchone()
        assert row is not None and row["status"] == "running"
    finally:
        await ledger.close()


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("record_kind", "field", "value", "message"),
    [
        ("attempt", "config_sha256", "9" * 64, "page provenance conflicts"),
        (
            "attempt",
            "source_uri",
            "file:///data/pdfs/different.pdf",
            "source provenance conflicts",
        ),
        ("result", "pipeline_fingerprint", "9" * 64, "page provenance conflicts"),
        ("result", "source_sha256", "9" * 64, "page provenance conflicts"),
    ],
)
async def test_typed_ledger_writes_reject_registered_provenance_drift(
    tmp_path: Path,
    record_kind: str,
    field: str,
    value: str,
    message: str,
) -> None:
    ledger = await _complete_ledger(tmp_path / "run", page_count=1, complete=False)
    try:
        if record_kind == "attempt":
            values = _attempt(0, 1).model_dump(mode="python")
            values.update(
                {
                    "attempt_number": 1,
                    "attempt_outcome": "retryable_error",
                    "http_status_code": 503,
                    "retry_after_seconds": 0.1,
                    "error_type": "HTTPStatusError",
                    "error_message": "vLLM returned HTTP 503",
                    field: value,
                }
            )
            drifted = InferenceAttempt.model_validate(values, strict=True)
            failure = _inference_failure(
                0,
                1,
                attempt_count=1,
                last_request_id=str(drifted.inference_request_id),
            )
            with pytest.raises(LedgerConflictError, match=message):
                await ledger.record_failure(failure=failure, attempts=(drifted,))
        else:
            values = _record(0, 1).model_dump(mode="python")
            values[field] = value
            drifted_result = PageExtractionRecord.model_validate(values, strict=True)
            with pytest.raises(LedgerConflictError, match=message):
                await ledger.record_success(
                    result=drifted_result,
                    attempts=(_attempt(0, 1),),
                )
    finally:
        await ledger.close()


@pytest.mark.asyncio
async def test_typed_ledger_rejects_noncontiguous_attempt_history(tmp_path: Path) -> None:
    ledger = await _complete_ledger(tmp_path / "run", page_count=1, complete=False)
    try:
        values = _attempt(0, 1).model_dump(mode="python")
        values.update(
            {
                "attempt_number": 3,
                "attempt_outcome": "retryable_error",
                "http_status_code": 503,
                "retry_after_seconds": 0.1,
                "error_type": "HTTPStatusError",
                "error_message": "vLLM returned HTTP 503",
            }
        )
        skipped = InferenceAttempt.model_validate(values, strict=True)
        failure = _inference_failure(
            0,
            1,
            attempt_count=3,
            last_request_id=str(skipped.inference_request_id),
        )

        with pytest.raises(LedgerConflictError, match="attempt numbers must be contiguous"):
            await ledger.record_failure(failure=failure, attempts=(skipped,))
    finally:
        await ledger.close()


@pytest.mark.asyncio
@pytest.mark.parametrize("artifact_failure", ["missing", "hash_mismatch"])
async def test_publication_rejects_missing_or_corrupt_raw_response_artifacts(
    tmp_path: Path, artifact_failure: str
) -> None:
    run_root = tmp_path / "run"
    ledger = await _complete_ledger(run_root, page_count=1)
    raw_response = run_root / _record(0, 1).raw_response_path
    try:
        if artifact_failure == "missing":
            raw_response.unlink()
            message = "raw response artifact is missing"
        else:
            raw_response.write_bytes(b"corrupt response")
            message = "raw response artifact hash mismatch"

        with pytest.raises(DatasetPublicationError, match=message):
            await publish_complete_dataset(
                ledger=ledger,
                run_id="run-1",
                run_root=run_root,
                batch_rows=1,
                compression="zstd",
                retain_page_images=False,
            )

        assert not (run_root / "dataset" / "manifest.json").exists()
    finally:
        await ledger.close()


class _MalformedLedger:
    def __init__(self, malformed_table: str) -> None:
        self.malformed_table = malformed_table

    async def require_complete(self, run_id: str) -> dict[str, int]:
        assert run_id == "run-1"
        return {
            "inventory_documents": 1,
            "registered_documents": 1,
            "expected_pages": 1,
            "complete_documents": 1,
            "failed_documents": 0,
            "registered_pages": 1,
            "successful_pages": 1,
            "failed_pages": 0,
            "pending_pages": 0,
            "attempt_rows": 1,
            "audited_successful_pages": 1,
        }

    async def iter_success_records(self, run_id: str):
        assert run_id == "run-1"
        if self.malformed_table == "pages":
            yield {"schema_version": 1}
        else:
            yield _record(0, 1).model_dump(mode="json")

    async def iter_attempts(self, run_id: str):
        assert run_id == "run-1"
        if self.malformed_table == "attempts":
            yield {"attempt_number": "one"}
        else:
            yield _attempt(0, 1).model_dump(mode="json")

    async def run_contract(self, run_id: str) -> dict[str, Any]:
        assert run_id == "run-1"
        return {
            "run_id": "run-1",
            "config_sha256": CONFIG_SHA256,
            "pipeline_fingerprint": PIPELINE_FINGERPRINT,
            "inventory_sha256": INVENTORY_SHA256,
            "inventory_document_count": 1,
        }


@pytest.mark.asyncio
@pytest.mark.parametrize("malformed_table", ["pages", "attempts"])
async def test_malformed_ledger_rows_abort_before_manifest_publication(
    tmp_path: Path, malformed_table: str
) -> None:
    run_root = tmp_path / "run"
    _write_raw_response(run_root, 0)

    with pytest.raises(DatasetPublicationError, match="ledger row does not satisfy"):
        await publish_complete_dataset(  # type: ignore[arg-type]
            ledger=_MalformedLedger(malformed_table),
            run_id="run-1",
            run_root=run_root,
            batch_rows=1,
            compression="zstd",
            retain_page_images=False,
        )

    assert not (run_root / "dataset" / "manifest.json").exists()


class _TruncatedLedger:
    def __init__(self, record: PageExtractionRecord) -> None:
        self.record = record

    async def require_complete(self, run_id: str) -> dict[str, int]:
        assert run_id == "run-1"
        return {
            "inventory_documents": 1,
            "registered_documents": 1,
            "expected_pages": 2,
            "complete_documents": 1,
            "failed_documents": 0,
            "registered_pages": 2,
            "successful_pages": 2,
            "failed_pages": 0,
            "pending_pages": 0,
            "attempt_rows": 0,
            "audited_successful_pages": 2,
        }

    async def iter_success_records(self, run_id: str):
        assert run_id == "run-1"
        yield self.record.model_dump(mode="json")

    async def iter_attempts(self, run_id: str):
        assert run_id == "run-1"
        if False:
            yield {}

    async def run_contract(self, run_id: str) -> dict[str, Any]:
        assert run_id == "run-1"
        return {
            "run_id": "run-1",
            "config_sha256": CONFIG_SHA256,
            "pipeline_fingerprint": PIPELINE_FINGERPRINT,
            "inventory_sha256": INVENTORY_SHA256,
            "inventory_document_count": 1,
        }


@pytest.mark.asyncio
async def test_manifest_rejects_export_row_counts_that_disagree_with_complete_summary(
    tmp_path: Path,
) -> None:
    run_root = tmp_path / "run"
    ledger = _TruncatedLedger(_record(0, 2))
    _write_raw_response(run_root, 0, 2)

    with pytest.raises(DatasetPublicationError, match="row count"):
        await publish_complete_dataset(  # type: ignore[arg-type]
            ledger=ledger,
            run_id="run-1",
            run_root=run_root,
            batch_rows=1,
            compression="zstd",
            retain_page_images=False,
        )

    assert not (run_root / "dataset" / "manifest.json").exists()
