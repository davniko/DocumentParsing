from __future__ import annotations

import hashlib
from datetime import UTC, datetime, timedelta

import pytest
from test_config import valid_config_data, valid_local_source_object_data, valid_page_record_data

from document_ocr.config import PipelineConfig
from document_ocr.hashing import canonical_json_bytes, canonical_json_sha256
from document_ocr.ledger import ExtractionLedger, IncompleteRunError, LedgerConflictError
from document_ocr.models import (
    DocumentExtractionFailure,
    InferenceAttempt,
    PageExtractionFailure,
    PageExtractionRecord,
    PageProvenance,
    SourceObject,
)


def _source() -> SourceObject:
    return SourceObject.model_validate(valid_local_source_object_data(), strict=True)


def _run_config_data() -> dict[str, object]:
    config = valid_config_data()
    config["run"]["run_id"] = "run-1"
    return PipelineConfig.model_validate(config, strict=True).model_dump(mode="json")


CONFIG_SHA256 = canonical_json_sha256(_run_config_data())


def _record(page_index: int, page_count: int, text: str) -> PageExtractionRecord:
    data = valid_page_record_data()
    data.update(
        {
            "run_id": "run-1",
            "extraction_id": f"extract-{page_index}",
            "page_id": f"page-{page_index}",
            "config_sha256": CONFIG_SHA256,
            "pipeline_fingerprint": "b" * 64,
            "document_page_count": page_count,
            "page_index": page_index,
            "page_number": page_index + 1,
            "raw_ocr_text": text,
            "raw_ocr_text_sha256": hashlib.sha256(text.encode()).hexdigest(),
            "raw_response_sha256": str(page_index + 1) * 64,
            "raw_response_path": (
                f"raw-responses/document-001/page-{page_index}/{str(page_index + 1) * 64}.json"
            ),
            "inference_request_id": "request-1",
            "inference_server_request_id": "server-request-1",
        }
    )
    return PageExtractionRecord.model_validate(data, strict=True)


def _provenance(record: PageExtractionRecord) -> PageProvenance:
    values = {name: getattr(record, name) for name in PageProvenance.model_fields}
    return PageProvenance.model_validate(values, strict=True)


def _attempt(
    record: PageExtractionRecord,
    *,
    number: int = 1,
    outcome: str = "success",
) -> InferenceAttempt:
    started = datetime(2026, 8, 5, 10, 0, tzinfo=UTC) + timedelta(seconds=number)
    values = _provenance(record).model_dump(mode="python")
    values.update(
        {
            "attempt_number": number,
            "attempt_started_at": started,
            "attempt_completed_at": started + timedelta(milliseconds=10),
            "attempt_duration_ms": 10.0,
            "attempt_outcome": outcome,
            "inference_request_id": f"request-{number}",
            "inference_server_request_id": f"server-request-{number}",
            "http_status_code": 200 if outcome == "success" else 503,
            "retry_after_seconds": 0.1 if outcome == "retryable_error" else None,
            "error_type": None if outcome == "success" else "HTTPStatusError",
            "error_message": None if outcome == "success" else "vLLM returned HTTP 503",
        }
    )
    return InferenceAttempt.model_validate(values, strict=True)


def _failure(record: PageExtractionRecord) -> PageExtractionFailure:
    values = _provenance(record).model_dump(mode="python")
    values.update(
        {
            "failure_stage": "inference",
            "failed_at": datetime(2026, 8, 5, 10, 1, tzinfo=UTC),
            "inference_attempt_count": 1,
            "retryable": True,
            "error_type": "HTTPStatusError",
            "error_message": "vLLM returned HTTP 503",
            "last_inference_request_id": "request-1",
        }
    )
    return PageExtractionFailure.model_validate(values, strict=True)


def _persist_failure(record: PageExtractionRecord) -> PageExtractionFailure:
    values = _provenance(record).model_dump(mode="python")
    values.update(
        {
            "failure_stage": "persist",
            "failed_at": datetime(2026, 8, 5, 10, 1, tzinfo=UTC),
            "inference_attempt_count": 1,
            "retryable": False,
            "error_type": "OSError",
            "error_message": "raw response could not be persisted",
            "last_inference_request_id": "request-1",
        }
    )
    return PageExtractionFailure.model_validate(values, strict=True)


async def _initialize(ledger: ExtractionLedger, *, resume: bool = False) -> str:
    return await ledger.initialize_run(
        run_id="run-1",
        config_sha256=CONFIG_SHA256,
        pipeline_fingerprint="b" * 64,
        config=_run_config_data(),
        inventory_sha256="c" * 64,
        inventory_document_count=1,
        provenance={"code": "d" * 64},
        resume=resume,
    )


@pytest.mark.asyncio
async def test_ledger_requires_every_inventory_document_and_page_to_succeed(tmp_path) -> None:
    first = _record(0, 2, "page zero")
    second = _record(1, 2, "page one")
    async with ExtractionLedger(tmp_path / "state.sqlite3") as ledger:
        assert await _initialize(ledger) == "running"
        await ledger.register_document(
            run_id="run-1",
            inventory_source=_source(),
            source_sha256="a" * 64,
            page_count=2,
        )
        await ledger.register_pages(
            run_id="run-1",
            document_id=first.document_id,
            pages=(
                (first.extraction_id, first.page_id, first.page_index),
                (second.extraction_id, second.page_id, second.page_index),
            ),
        )
        await ledger.record_success(result=first, attempts=(_attempt(first),))
        failed_attempt = _attempt(second, outcome="retryable_error")
        await ledger.record_failure(
            failure=_failure(second),
            attempts=(failed_attempt,),
        )

        with pytest.raises(IncompleteRunError):
            await ledger.require_complete("run-1")

        second = second.model_copy(
            update={
                "inference_attempt_count": 2,
                "inference_request_id": "request-2",
                "inference_server_request_id": "server-request-2",
            }
        )
        await ledger.record_success(result=second, attempts=(_attempt(second, number=2),))
        summary = await ledger.require_complete("run-1")
        assert summary["inventory_documents"] == 1
        assert summary["successful_pages"] == 2
        assert summary["attempt_rows"] == 3
        assert summary["audited_successful_pages"] == 2
        records = [record async for record in ledger.iter_success_records("run-1")]
        assert [record["page_index"] for record in records] == [0, 1]


@pytest.mark.asyncio
async def test_document_open_failure_is_explicit_and_blocks_publication(tmp_path) -> None:
    source = _source()
    failure = DocumentExtractionFailure.model_validate(
        {
            **source.model_dump(mode="python"),
            "schema_version": 1,
            "run_id": "run-1",
            "config_sha256": CONFIG_SHA256,
            "pipeline_fingerprint": "b" * 64,
            "failure_stage": "pdf_open",
            "failed_at": datetime(2026, 8, 5, 10, 0, tzinfo=UTC),
            "error_type": "PdfOpenError",
            "error_message": "invalid PDF",
        },
        strict=True,
    )
    async with ExtractionLedger(tmp_path / "state.sqlite3") as ledger:
        await _initialize(ledger)
        await ledger.record_document_failure(failure=failure)

        summary = await ledger.validation_summary("run-1")
        assert summary["registered_documents"] == 1
        assert summary["failed_documents"] == 1
        failures = [item async for item in ledger.iter_document_failures("run-1")]
        assert failures[0]["failure_stage"] == "pdf_open"
        with pytest.raises(IncompleteRunError):
            await ledger.require_complete("run-1")


@pytest.mark.asyncio
async def test_ledger_rejects_resume_drift_and_conflicting_success(tmp_path) -> None:
    record = _record(0, 1, "first")
    async with ExtractionLedger(tmp_path / "state.sqlite3") as ledger:
        await _initialize(ledger)
        with pytest.raises(LedgerConflictError, match="cannot resume with changed"):
            await ledger.initialize_run(
                run_id="run-1",
                config_sha256=CONFIG_SHA256,
                pipeline_fingerprint="9" * 64,
                config=_run_config_data(),
                inventory_sha256="c" * 64,
                inventory_document_count=1,
                provenance={"code": "d" * 64},
                resume=True,
            )

        await ledger.register_document(
            run_id="run-1",
            inventory_source=_source(),
            source_sha256="a" * 64,
            page_count=1,
        )
        await ledger.register_page(
            run_id="run-1",
            extraction_id=record.extraction_id,
            document_id=record.document_id,
            page_id=record.page_id,
            page_index=record.page_index,
        )
        attempt = _attempt(record)
        await ledger.record_success(result=record, attempts=(attempt,))
        await ledger.record_success(result=record, attempts=(attempt,))
        changed = _record(0, 1, "second")
        with pytest.raises(LedgerConflictError, match="conflicting successful OCR"):
            await ledger.record_success(result=changed, attempts=(attempt,))
        failed_again = _failure(record).model_copy(
            update={
                "inference_attempt_count": 2,
                "last_inference_request_id": "request-2",
            }
        )
        with pytest.raises(LedgerConflictError, match="cannot replace successful OCR"):
            await ledger.record_failure(
                failure=failed_again,
                attempts=(_attempt(record, number=2, outcome="retryable_error"),),
            )


@pytest.mark.asyncio
async def test_ledger_rejects_resume_provenance_drift(tmp_path) -> None:
    async with ExtractionLedger(tmp_path / "state.sqlite3") as ledger:
        await _initialize(ledger)

        with pytest.raises(LedgerConflictError, match="runtime provenance"):
            await ledger.initialize_run(
                run_id="run-1",
                config_sha256=CONFIG_SHA256,
                pipeline_fingerprint="b" * 64,
                config=_run_config_data(),
                inventory_sha256="c" * 64,
                inventory_document_count=1,
                provenance={"code": "9" * 64},
                resume=True,
            )


@pytest.mark.asyncio
async def test_attempt_rows_are_immutable_and_identity_checked(tmp_path) -> None:
    record = _record(0, 1, "first")
    attempt = _attempt(record, outcome="retryable_error")
    async with ExtractionLedger(tmp_path / "state.sqlite3") as ledger:
        await _initialize(ledger)
        await ledger.register_document(
            run_id="run-1",
            inventory_source=_source(),
            source_sha256="a" * 64,
            page_count=1,
        )
        await ledger.register_page(
            run_id="run-1",
            extraction_id=record.extraction_id,
            document_id=record.document_id,
            page_id=record.page_id,
            page_index=record.page_index,
        )
        failure = _failure(record)
        await ledger.record_failure(failure=failure, attempts=(attempt,))
        await ledger.record_failure(failure=failure, attempts=(attempt,))
        assert await ledger.next_attempt_number("run-1", record.extraction_id) == 2
        changed = attempt.model_copy(update={"error_message": "different"})
        with pytest.raises(LedgerConflictError, match="attempt conflict"):
            await ledger.record_failure(failure=failure, attempts=(changed,))


@pytest.mark.asyncio
async def test_success_attempt_and_page_result_roll_back_together(
    tmp_path, monkeypatch: pytest.MonkeyPatch
) -> None:
    record = _record(0, 1, "first")
    async with ExtractionLedger(tmp_path / "state.sqlite3") as ledger:
        await _initialize(ledger)
        await ledger.register_document(
            run_id="run-1",
            inventory_source=_source(),
            source_sha256="a" * 64,
            page_count=1,
        )
        await ledger.register_page(
            run_id="run-1",
            extraction_id=record.extraction_id,
            document_id=record.document_id,
            page_id=record.page_id,
            page_index=record.page_index,
        )

        original_refresh = ledger._refresh_document_status

        async def fail_after_page_update(_run_id: str, _extraction_id: str) -> None:
            raise RuntimeError("simulated failure before commit")

        monkeypatch.setattr(ledger, "_refresh_document_status", fail_after_page_update)
        with pytest.raises(RuntimeError, match="simulated failure"):
            await ledger.record_success(result=record, attempts=(_attempt(record),))
        monkeypatch.setattr(ledger, "_refresh_document_status", original_refresh)

        assert not await ledger.is_successful("run-1", record.extraction_id)
        assert await ledger.next_attempt_number("run-1", record.extraction_id) == 1
        summary = await ledger.validation_summary("run-1")
        assert summary["attempt_rows"] == 0
        assert summary["successful_pages"] == 0


@pytest.mark.asyncio
async def test_failure_attempt_and_page_failure_roll_back_together(
    tmp_path, monkeypatch: pytest.MonkeyPatch
) -> None:
    record = _record(0, 1, "first")
    async with ExtractionLedger(tmp_path / "state.sqlite3") as ledger:
        await _initialize(ledger)
        await ledger.register_document(
            run_id="run-1",
            inventory_source=_source(),
            source_sha256="a" * 64,
            page_count=1,
        )
        await ledger.register_page(
            run_id="run-1",
            extraction_id=record.extraction_id,
            document_id=record.document_id,
            page_id=record.page_id,
            page_index=record.page_index,
        )

        original_refresh = ledger._refresh_document_status

        async def fail_after_page_update(_run_id: str, _extraction_id: str) -> None:
            raise RuntimeError("simulated failure before commit")

        monkeypatch.setattr(ledger, "_refresh_document_status", fail_after_page_update)
        with pytest.raises(RuntimeError, match="simulated failure"):
            await ledger.record_failure(
                failure=_failure(record),
                attempts=(_attempt(record, outcome="retryable_error"),),
            )
        monkeypatch.setattr(ledger, "_refresh_document_status", original_refresh)

        assert await ledger.attempt_state("run-1", record.extraction_id) == (0, None)
        summary = await ledger.validation_summary("run-1")
        assert summary["attempt_rows"] == 0
        assert summary["failed_pages"] == 0
        assert summary["pending_pages"] == 1


@pytest.mark.asyncio
async def test_persistence_failure_attempt_is_retained_and_resume_can_succeed(tmp_path) -> None:
    record = _record(0, 1, "first")
    async with ExtractionLedger(tmp_path / "state.sqlite3") as ledger:
        await _initialize(ledger)
        await ledger.register_document(
            run_id="run-1",
            inventory_source=_source(),
            source_sha256="a" * 64,
            page_count=1,
        )
        await ledger.register_page(
            run_id="run-1",
            extraction_id=record.extraction_id,
            document_id=record.document_id,
            page_id=record.page_id,
            page_index=record.page_index,
        )

        await ledger.record_failure(
            failure=_persist_failure(record),
            attempts=(_attempt(record),),
        )
        assert await ledger.attempt_state("run-1", record.extraction_id) == (1, "request-1")

        resumed = record.model_copy(
            update={
                "inference_attempt_count": 2,
                "inference_request_id": "request-2",
                "inference_server_request_id": "server-request-2",
            }
        )
        await ledger.record_success(
            result=resumed,
            attempts=(_attempt(resumed, number=2),),
        )

        summary = await ledger.require_complete("run-1")
        assert summary["attempt_rows"] == 2
        assert summary["audited_successful_pages"] == 1


@pytest.mark.asyncio
async def test_orphaned_success_attempt_is_detected_as_corrupt_state(tmp_path) -> None:
    record = _record(0, 1, "first")
    legacy_success = _attempt(record)
    async with ExtractionLedger(tmp_path / "state.sqlite3") as ledger:
        await _initialize(ledger)
        await ledger.register_document(
            run_id="run-1",
            inventory_source=_source(),
            source_sha256="a" * 64,
            page_count=1,
        )
        await ledger.register_page(
            run_id="run-1",
            extraction_id=record.extraction_id,
            document_id=record.document_id,
            page_id=record.page_id,
            page_index=record.page_index,
        )
        await ledger.connection.execute(
            """
            INSERT INTO attempts (
                run_id, extraction_id, attempt_number, attempt_json, created_at
            ) VALUES (?, ?, ?, ?, ?)
            """,
            (
                legacy_success.run_id,
                legacy_success.extraction_id,
                legacy_success.attempt_number,
                canonical_json_bytes(legacy_success.model_dump(mode="json")).decode("utf-8"),
                datetime.now(UTC).isoformat(),
            ),
        )
        await ledger.connection.commit()

        next_result = record.model_copy(
            update={
                "inference_attempt_count": 2,
                "inference_request_id": "request-2",
                "inference_server_request_id": "server-request-2",
            }
        )
        with pytest.raises(LedgerConflictError, match="orphaned success attempt"):
            await ledger.record_success(
                result=next_result,
                attempts=(_attempt(next_result, number=2),),
            )

        assert not await ledger.is_successful("run-1", record.extraction_id)
        assert await ledger.next_attempt_number("run-1", record.extraction_id) == 2
        summary = await ledger.validation_summary("run-1")
        assert summary["attempt_rows"] == 1
        assert summary["successful_pages"] == 0


@pytest.mark.asyncio
async def test_writable_ledger_never_follows_existing_symlink(tmp_path) -> None:
    destination = tmp_path / "destination.sqlite3"
    destination.write_bytes(b"protected")
    ledger_path = tmp_path / "state.sqlite3"
    ledger_path.symlink_to(destination)

    ledger = ExtractionLedger(ledger_path)
    with pytest.raises(LedgerConflictError, match="canonical regular file"):
        await ledger.open()

    assert destination.read_bytes() == b"protected"
