from __future__ import annotations

import asyncio
import json
import sqlite3
from datetime import UTC, datetime
from io import BytesIO
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
import yaml
from PIL import Image
from pydantic import ValidationError
from test_config import valid_config_data, valid_page_record_data

import document_ocr.table_views.join as join_module
from document_ocr.config import PipelineConfig
from document_ocr.hashing import canonical_json_bytes, canonical_json_sha256, sha256_bytes
from document_ocr.labeling_agents.models import CompactAnnotationDraft
from document_ocr.labeling_agents.orchestrator import build_compact_annotation
from document_ocr.labeling_agents.work_items import AgentWorkItem
from document_ocr.models import PageExtractionFailure, PageExtractionRecord, SourceObject
from document_ocr.sources import freeze_source_inventory
from document_ocr.table_views import cli as table_cli
from document_ocr.table_views import pipeline as pipeline_module
from document_ocr.table_views.cli import main as table_view_main
from document_ocr.table_views.config import TableViewConfig
from document_ocr.table_views.join import publish_joined_dataset, verify_joined_dataset
from document_ocr.table_views.models import (
    JoinedTableDocumentView,
    JoinedTablePage,
    TableInputPage,
    TablePageResult,
)
from document_ocr.table_views.pipeline import TableProgress
from document_ocr.training.config import (
    DatasetFileConfig,
    DatasetSplitsConfig,
    load_training_config,
)
from document_ocr.training.data import inspect_dataset
from document_ocr.training.prompting import load_prompt
from document_ocr.training.tasks import get_training_task


def _vllm() -> dict[str, Any]:
    return {
        "endpoint": "http://127.0.0.1:8000",
        "model": "zai-org/GLM-OCR",
        "served_model_name": "glm-ocr",
        "revision": "a" * 40,
        "engine_version": "0.26.0",
        "container_base_image": "image@sha256:" + "b" * 64,
        "container_build_manifest_sha256": "c" * 64,
        "dtype": "bfloat16",
        "quantization": "none",
        "max_model_len": 32768,
        "max_num_batched_tokens": 16384,
        "max_num_seqs": 16,
        "gpu_memory_utilization": 0.9,
        "prompt": "Table Recognition:",
        "request_timeout_seconds": 300.0,
        "api_key_env": "VLLM_API_KEY",
        "sampling": {
            "temperature": 0.0,
            "top_p": 0.00001,
            "top_k": 1,
            "repetition_penalty": 1.1,
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
        "speculative_decoding": {"method": "mtp", "num_speculative_tokens": 1},
        "repetition_detection": {
            "min_pattern_size": 5,
            "max_pattern_size": 64,
            "min_count": 5,
        },
    }


def test_table_progress_cli_reports_rate_and_eta(
    capsys: pytest.CaptureFixture[str],
) -> None:
    table_cli._progress(
        TableProgress(
            total_pages=10,
            processed_pages=4,
            successful_pages=3,
            failed_pages_this_invocation=1,
            elapsed_seconds=2.0,
            throughput_pages_per_second=2.0,
            eta_seconds=3.0,
        )
    )

    assert json.loads(capsys.readouterr().err) == {
        "elapsed_seconds": 2.0,
        "eta_seconds": 3.0,
        "failed_pages_this_invocation": 1,
        "phase": "table_recognition",
        "processed_pages": 4,
        "remaining_pages": 6,
        "status": "progress",
        "successful_pages": 3,
        "throughput_pages_per_hour": 7200.0,
        "throughput_pages_per_second": 2.0,
        "total_pages": 10,
    }


def _source_row(document_id: str, text: str) -> dict[str, Any]:
    return {
        "documentId": document_id,
        "joinedRawText": text,
        "joinedRawTextSha256": sha256_bytes(text.encode()),
        "target": {"documentPatch": {"documentReference": document_id[-8:]}},
    }


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> str:
    payload = b"".join(canonical_json_bytes(row) + b"\n" for row in rows)
    path.write_bytes(payload)
    return sha256_bytes(payload)


def _config(tmp_path: Path, train: Path, validation: Path) -> TableViewConfig:
    record_root = tmp_path / "records"
    extraction_root = tmp_path / "extraction"
    table_root = tmp_path / "table"
    joined_root = tmp_path / "joined"
    record_root.mkdir()
    extraction_root.mkdir()
    return TableViewConfig.model_validate(
        {
            "schema_version": 1,
            "view_type": "glm_ocr_table_recognition",
            "source": {
                "type": "validated_label_rasters",
                "expected_documents": 2,
                "expected_pages": 2,
                "record_sets": [
                    {
                        "id": "records",
                        "root": str(record_root),
                        "records_path": "training/records.jsonl",
                        "records_sha256": "d" * 64,
                        "records": 2,
                    }
                ],
                "extraction_runs": [{"run_id": "source-v1", "root": str(extraction_root)}],
            },
            "output": {"root": str(table_root)},
            "run": {
                "run_id": "table-v1",
                "resume": True,
                "fail_fast": False,
                "max_total_attempts_per_page": 8,
            },
            "vllm": _vllm(),
            "concurrency": {
                "max_active_documents": 2,
                "max_inflight_pages_global": 2,
                "max_inflight_pages_per_document": 1,
            },
            "join": {
                "dataset_id": "joined-v1",
                "output_root": str(joined_root),
                "inputs": [
                    {
                        "split": "train",
                        "path": str(train),
                        "sha256": sha256_bytes(train.read_bytes()),
                        "records": 1,
                    },
                    {
                        "split": "validation",
                        "path": str(validation),
                        "sha256": sha256_bytes(validation.read_bytes()),
                        "records": 1,
                    },
                ],
            },
        },
        strict=True,
    )


def _png_bytes(color: tuple[int, int, int]) -> bytes:
    stream = BytesIO()
    Image.new("RGB", (16, 12), color=color).save(stream, format="PNG")
    return stream.getvalue()


def _raw_extraction_config(
    tmp_path: Path,
    *,
    failure_stage: str = "inference",
    failed_raster_count: int = 1,
) -> TableViewConfig:
    source_root = tmp_path / "source"
    raw_output = tmp_path / "raw-output"
    source_root.mkdir()
    raw_output.mkdir()
    source_path = source_root / "source.pdf"
    source_bytes = b"%PDF-1.7\nfixture\n"
    source_path.write_bytes(source_bytes)
    source_sha256 = sha256_bytes(source_bytes)
    document_id = "doc_" + "1" * 64
    raw_run_id = "raw-raster-source-v1"
    run_root = raw_output / "runs" / raw_run_id
    run_root.mkdir(parents=True)

    pipeline_data = valid_config_data()
    pipeline_data["source"] = {
        **pipeline_data["source"],
        "root": str(source_root),
    }
    pipeline_data["output"] = {
        **pipeline_data["output"],
        "root": str(raw_output),
        "retain_page_images": True,
    }
    pipeline_data["run"] = {
        "run_id": raw_run_id,
        "fail_fast": False,
        "resume": True,
        "expected_pages": 2,
    }
    pipeline_config = PipelineConfig.model_validate(pipeline_data, strict=True)
    config_payload = pipeline_config.model_dump(mode="json")
    config_sha256 = canonical_json_sha256(config_payload)
    pipeline_fingerprint = "e" * 64
    source = SourceObject.model_validate(
        {
            "document_id": document_id,
            "source_type": "local",
            "source_uri": source_path.as_uri(),
            "source_dataset_version": "raw-raster-fixture-v1",
            "source_object_version": source_sha256,
            "source_size_bytes": len(source_bytes),
            "source_last_modified": datetime(2026, 8, 22, tzinfo=UTC),
            "source_sha256": source_sha256,
            "local_canonical_path": str(source_path),
            "local_relative_key": source_path.name,
            "local_device": 1,
            "local_inode": 2,
            "local_mtime_ns": 3,
        },
        strict=True,
    )
    inventory = freeze_source_inventory(run_root / "inventory.jsonl", [source])
    provenance = {
        "resolved_config_sha256": config_sha256,
        "pipeline_fingerprint": pipeline_fingerprint,
        "inventory_sha256": inventory.sha256,
        "inventory_document_count": 1,
    }
    provenance_bytes = canonical_json_bytes(provenance) + b"\n"
    (run_root / "run-provenance.json").write_bytes(provenance_bytes)

    page_records: list[tuple[str, str, str | None, str | None]] = []
    for page_index in range(2):
        page_id = f"page-{page_index + 1}"
        extraction_id = f"extract-{page_index + 1}"
        raster_bytes = _png_bytes((page_index * 30, 20, 40))
        raster_sha256 = sha256_bytes(raster_bytes)
        relative_raster = Path("page-images") / document_id / page_id / f"{raster_sha256}.png"
        if page_index == 0 or failed_raster_count >= 1:
            raster_path = run_root / relative_raster
            raster_path.parent.mkdir(parents=True, exist_ok=True)
            raster_path.write_bytes(raster_bytes)

        provenance_fields = {
            **source.model_dump(mode="python"),
            "schema_version": 2,
            "run_id": raw_run_id,
            "extraction_id": extraction_id,
            "page_id": page_id,
            "config_sha256": config_sha256,
            "pipeline_fingerprint": pipeline_fingerprint,
            "document_page_count": 2,
            "page_index": page_index,
            "page_number": page_index + 1,
        }
        if page_index == 0:
            raw_record = valid_page_record_data()
            raw_record.update(provenance_fields)
            raw_record.update(
                {
                    "raster_path": relative_raster.as_posix(),
                    "raster_size_bytes": len(raster_bytes),
                    "raster_sha256": raster_sha256,
                    "raw_response_path": (f"raw-responses/{document_id}/{page_id}/{'a' * 64}.json"),
                    "raw_response_sha256": "a" * 64,
                }
            )
            success = PageExtractionRecord.model_validate(raw_record, strict=True)
            page_records.append(
                (
                    extraction_id,
                    "success",
                    canonical_json_bytes(success.model_dump(mode="json")).decode(),
                    None,
                )
            )
        else:
            failure = PageExtractionFailure.model_validate(
                {
                    **provenance_fields,
                    "failure_stage": failure_stage,
                    "failed_at": datetime(2026, 8, 22, 12, tzinfo=UTC),
                    "inference_attempt_count": 1,
                    "retryable": False,
                    "error_type": "FixtureError",
                    "error_message": "fixture failure",
                    "last_inference_request_id": "request-2",
                },
                strict=True,
            )
            page_records.append(
                (
                    extraction_id,
                    "failed",
                    None,
                    canonical_json_bytes(failure.model_dump(mode="json")).decode(),
                )
            )

    if failed_raster_count == 2:
        extra = _png_bytes((99, 88, 77))
        extra_sha256 = sha256_bytes(extra)
        extra_path = run_root / "page-images" / document_id / "page-2" / f"{extra_sha256}.png"
        extra_path.write_bytes(extra)

    connection = sqlite3.connect(run_root / "state.sqlite3")
    connection.executescript(
        """
        CREATE TABLE runs (
            run_id TEXT PRIMARY KEY, config_sha256 TEXT, pipeline_fingerprint TEXT,
            config_json TEXT, inventory_sha256 TEXT, inventory_document_count INTEGER,
            provenance_json TEXT, status TEXT, manifest_sha256 TEXT, manifest_json TEXT
        );
        CREATE TABLE documents (
            run_id TEXT, document_id TEXT, source_json TEXT, source_sha256 TEXT,
            page_count INTEGER, status TEXT, failure_json TEXT
        );
        CREATE TABLE pages (
            run_id TEXT, extraction_id TEXT, document_id TEXT, page_id TEXT,
            page_index INTEGER, status TEXT, result_json TEXT, failure_json TEXT
        );
        """
    )
    connection.execute(
        "INSERT INTO runs VALUES (?, ?, ?, ?, ?, ?, ?, 'failed', NULL, NULL)",
        (
            raw_run_id,
            config_sha256,
            pipeline_fingerprint,
            canonical_json_bytes(config_payload).decode(),
            inventory.sha256,
            1,
            canonical_json_bytes(provenance).decode(),
        ),
    )
    connection.execute(
        "INSERT INTO documents VALUES (?, ?, ?, ?, 2, 'incomplete', NULL)",
        (
            raw_run_id,
            document_id,
            canonical_json_bytes(source.model_dump(mode="json")).decode(),
            source_sha256,
        ),
    )
    for page_index, (extraction_id, status, result_json, failure_json) in enumerate(page_records):
        connection.execute(
            "INSERT INTO pages VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
            (
                raw_run_id,
                extraction_id,
                document_id,
                f"page-{page_index + 1}",
                page_index,
                status,
                result_json,
                failure_json,
            ),
        )
    connection.commit()
    connection.close()

    return TableViewConfig.model_validate(
        {
            "schema_version": 1,
            "view_type": "glm_ocr_table_recognition",
            "source": {
                "type": "raw_extraction_rasters",
                "expected_documents": 1,
                "expected_pages": 2,
                "extraction_runs": [
                    {
                        "id": "raw_fixture",
                        "run_id": raw_run_id,
                        "root": str(run_root),
                        "config_sha256": config_sha256,
                        "pipeline_fingerprint": pipeline_fingerprint,
                        "inventory_sha256": inventory.sha256,
                        "provenance_sha256": None,
                        "manifest_sha256": None,
                        "expected_documents": 1,
                        "expected_pages": 2,
                    }
                ],
            },
            "output": {"root": str(tmp_path / "table-output")},
            "run": {
                "run_id": "table-from-raw-v1",
                "resume": True,
                "fail_fast": False,
                "max_total_attempts_per_page": 8,
            },
            "vllm": _vllm(),
            "concurrency": {
                "max_active_documents": 2,
                "max_inflight_pages_global": 2,
                "max_inflight_pages_per_document": 1,
            },
        },
        strict=True,
    )


def _view(document_id: str, page_id: str, table_text: str) -> JoinedTableDocumentView:
    page = JoinedTablePage.model_validate(
        {
            "pageIndex": 0,
            "pageNumber": 1,
            "pageId": page_id,
            "tableViewId": f"table_{page_id}",
            "rasterSha256": "e" * 64,
            "finishReason": "stop",
            "completionTokens": 10,
            "rawTableText": table_text,
            "rawTableTextSha256": sha256_bytes(table_text.encode()),
            "resultSha256": "f" * 64,
        },
        strict=True,
    )
    joined = f"--- PAGE 1 ---\n{table_text}"
    return JoinedTableDocumentView.model_validate(
        {
            "schemaVersion": 1,
            "viewType": "glm_ocr_table_recognition",
            "runId": "table-v1",
            "documentPageCount": 1,
            "joinedTableText": joined,
            "joinedTableTextSha256": sha256_bytes(joined.encode()),
            "pages": (page,),
        },
        strict=True,
    )


def test_retry_inference_slot_has_a_separate_recovery_ceiling(
    tmp_path: Path,
) -> None:
    train = tmp_path / "train.jsonl"
    validation = tmp_path / "validation.jsonl"
    _write_jsonl(train, [_source_row("doc_" + "1" * 64, "--- PAGE 1 ---\nA")])
    _write_jsonl(validation, [_source_row("doc_" + "2" * 64, "--- PAGE 1 ---\nB")])
    config = _config(tmp_path, train, validation)
    assert pipeline_module._retry_recovery_concurrency(config.concurrency) == 1

    async def probe() -> tuple[int, int]:
        global_limit = asyncio.Semaphore(2)
        retry_limit = asyncio.Semaphore(1)
        active_first = active_retry = 0
        peak_first = peak_retry = 0
        lock = asyncio.Lock()

        async def enter(prior_attempt_count: int) -> None:
            nonlocal active_first, active_retry, peak_first, peak_retry
            async with pipeline_module._inference_slot(
                prior_attempt_count=prior_attempt_count,
                global_limit=global_limit,
                retry_limit=retry_limit,
            ):
                async with lock:
                    if prior_attempt_count:
                        active_retry += 1
                        peak_retry = max(peak_retry, active_retry)
                    else:
                        active_first += 1
                        peak_first = max(peak_first, active_first)
                await asyncio.sleep(0.01)
                async with lock:
                    if prior_attempt_count:
                        active_retry -= 1
                    else:
                        active_first -= 1

        await asyncio.gather(*(enter(0) for _ in range(2)))
        await asyncio.gather(*(enter(1) for _ in range(3)))
        return peak_first, peak_retry

    assert asyncio.run(probe()) == (2, 1)


def test_base_page_ledger_is_loaded_once_per_extraction_run(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    run_root = tmp_path / "source-run"
    run_root.mkdir()
    ledger_path = run_root / "state.sqlite3"
    connection = sqlite3.connect(ledger_path)
    connection.execute(
        "CREATE TABLE pages (run_id TEXT, extraction_id TEXT, status TEXT, result_json TEXT)"
    )
    for page_index in range(2):
        raw_text = f"page {page_index + 1}"
        extraction_id = f"extract-{page_index + 1}"
        result = {
            "schema_version": 1,
            "run_id": "source-v1",
            "extraction_id": extraction_id,
            "document_id": "doc_" + "1" * 64,
            "document_page_count": 2,
            "page_index": page_index,
            "page_number": page_index + 1,
            "page_id": f"page-{page_index + 1}",
            "raw_ocr_text": raw_text,
            "raw_ocr_text_sha256": sha256_bytes(raw_text.encode()),
            "source_sha256": "a" * 64,
            "raster_path": f"rasters/page-{page_index + 1}.png",
            "raster_mime_type": "image/png",
            "raster_size_bytes": 10,
            "raster_sha256": "b" * 64,
        }
        connection.execute(
            "INSERT INTO pages VALUES (?, ?, 'success', ?)",
            ("source-v1", extraction_id, json.dumps(result)),
        )
    connection.commit()
    connection.close()

    original_connect = pipeline_module.sqlite3.connect
    connection_count = 0

    def counted_connect(*args: Any, **kwargs: Any) -> sqlite3.Connection:
        nonlocal connection_count
        connection_count += 1
        return original_connect(*args, **kwargs)

    monkeypatch.setattr(pipeline_module.sqlite3, "connect", counted_connect)
    rows = pipeline_module._load_successful_base_pages(
        run_root=run_root,
        run_id="source-v1",
    )

    assert connection_count == 1
    assert tuple(rows) == ("extract-1", "extract-2")


def test_table_quality_artifacts_flag_incomplete_repetition_without_filtering() -> None:
    results: Any = [
        SimpleNamespace(
            documentId="doc_" + "1" * 64,
            pageId="page-1",
            pageNumber=1,
            tableViewId="table-1",
            finishReason="stop",
            completionTokens=0,
            attemptCount=1,
            rawTableText="",
            rawTableTextSha256="a" * 64,
        ),
        SimpleNamespace(
            documentId="doc_" + "2" * 64,
            pageId="page-2",
            pageNumber=1,
            tableViewId="table-2",
            finishReason="repetition",
            completionTokens=198,
            attemptCount=2,
            rawTableText="<table><tr><td>20 PKG</td>",
            rawTableTextSha256="b" * 64,
        ),
        SimpleNamespace(
            documentId="doc_" + "3" * 64,
            pageId="page-3",
            pageNumber=1,
            tableViewId="table-3",
            finishReason="stop",
            completionTokens=10,
            attemptCount=1,
            rawTableText="<table><tr><td>OK</td></tr></table>",
            rawTableTextSha256="c" * 64,
        ),
    ]
    attempts = [
        {"outcome": "success", "durationMs": 10.0},
        {"outcome": "transport_error", "durationMs": 300000.0},
        {"outcome": "success", "durationMs": 20.0},
    ]

    summary, flags = pipeline_module._table_quality_artifacts(results, attempts)

    assert summary["pages"] == 3
    assert summary["page_output_classes"] == {
        "complete_html_table_markup": 1,
        "empty": 1,
        "incomplete_html_table_markup": 1,
    }
    assert summary["finish_reasons"] == {"repetition": 1, "stop": 2}
    assert summary["pages_with_multiple_attempts"] == 1
    assert flags[0]["reviewReasons"] == [
        "repetition_finish",
        "incomplete_html_table_markup",
    ]


def test_table_input_rejects_page_order_mismatch() -> None:
    value = {
        "schemaVersion": 1,
        "recordSetId": "records",
        "annotationPath": "/tmp/annotation.json",
        "annotationSha256": "a" * 64,
        "documentId": "doc_" + "b" * 64,
        "documentPageCount": 2,
        "pageIndex": 0,
        "pageNumber": 2,
        "pageId": "page-1",
        "baseExtractionId": "extract-1",
        "baseExtractionRunId": "source-v1",
        "baseRawOcrTextSha256": "c" * 64,
        "sourceSha256": "d" * 64,
        "rasterPath": "/tmp/page.png",
        "rasterMimeType": "image/png",
        "rasterSizeBytes": 10,
        "rasterSha256": "e" * 64,
        "tableViewId": "table-1",
    }
    with pytest.raises(ValidationError, match="pageNumber must equal pageIndex"):
        TableInputPage.model_validate(value, strict=True)


def test_table_config_rejects_text_recognition_prompt(tmp_path: Path) -> None:
    train = tmp_path / "train.jsonl"
    validation = tmp_path / "validation.jsonl"
    _write_jsonl(train, [_source_row("doc_" + "1" * 64, "train")])
    _write_jsonl(validation, [_source_row("doc_" + "2" * 64, "validation")])
    config = _config(tmp_path, train, validation).model_dump(mode="python")
    config["vllm"]["prompt"] = "Text Recognition:"
    with pytest.raises(ValidationError, match="Table Recognition"):
        TableViewConfig.model_validate(config, strict=True)


def test_validated_table_source_accepts_dual_cargo_annotation() -> None:
    document_id = "doc_" + "8" * 64
    page_text = "B/L NO: HBL-001"
    joined_text = f"--- PAGE 1 ---\n{page_text}"
    item = AgentWorkItem.model_validate(
        {
            "source": {
                "documentId": document_id,
                "extractionRunId": "source-v1",
                "sourceUri": "file:///fixture.pdf",
                "localCanonicalPath": "/fixture.pdf",
                "sourceSha256": "a" * 64,
                "documentPageCount": 1,
                "joinedRawTextSha256": sha256_bytes(joined_text.encode()),
                "pages": (
                    {
                        "pageIndex": 0,
                        "pageNumber": 1,
                        "pageId": "page-1",
                        "extractionId": "extract-1",
                        "rawOcrTextSha256": sha256_bytes(page_text.encode()),
                        "rawResponsePath": "raw-response.json",
                        "rawResponseSha256": "b" * 64,
                        "rasterPath": "page.png",
                        "rasterSha256": "c" * 64,
                    },
                ),
            },
            "joinedRawText": joined_text,
        },
        strict=True,
    )
    draft = CompactAnnotationDraft.model_validate(
        {
            "decision": "annotation",
            "documentType": "bill_of_lading",
            "relationExplicitLabel": {
                "schemaVersion": "3.0.0-experimental",
                "documentPatch": {"billOfLadingNumber": "HBL-001"},
            },
            "warnings": (),
            "decisionNotes": ("Fixture label.",),
        },
        strict=True,
    )
    annotation = build_compact_annotation(
        item, draft, pdf_grouping_used=False
    ).model_copy(update={"reviewStatus": "validated"})
    payload = canonical_json_bytes(annotation.model_dump(mode="json"))

    parsed = pipeline_module._validated_bill_of_lading_annotation(
        payload, document_id=document_id
    )

    assert pipeline_module._annotation_training_target(parsed) == (
        annotation.relationExplicitLabel.canonical_target()
    )


def test_validated_table_source_rejects_unknown_annotation_schema() -> None:
    payload = canonical_json_bytes({"annotationSchemaVersion": "4.0.0"})

    with pytest.raises(
        pipeline_module.TableViewError, match="unsupported annotation schema version"
    ):
        pipeline_module._validated_bill_of_lading_annotation(
            payload, document_id="doc_" + "9" * 64
        )


def test_raw_extraction_source_accepts_inference_failure_with_retained_raster(
    tmp_path: Path,
) -> None:
    config = _raw_extraction_config(tmp_path)

    pages = pipeline_module.prepare_table_inputs(config)

    assert [(page.pageIndex, page.baseRawOcrTextSha256 is None) for page in pages] == [
        (0, False),
        (1, True),
    ]
    assert all(page.schemaVersion == 1 for page in pages)
    assert all(page.annotationPath.endswith("run-provenance.json") for page in pages)
    assert all(
        page.annotationSha256 == sha256_bytes(Path(page.annotationPath).read_bytes())
        for page in pages
    )
    assert pipeline_module.load_prepared_pages(config) == pages


def test_raw_extraction_source_rejects_noncanonical_provenance(tmp_path: Path) -> None:
    config = _raw_extraction_config(tmp_path)
    source = config.source.extraction_runs[0]
    provenance_path = Path(source.root) / "run-provenance.json"
    value = json.loads(provenance_path.read_text(encoding="utf-8"))
    provenance_path.write_text(json.dumps(value, indent=2), encoding="utf-8")

    with pytest.raises(pipeline_module.TableViewError, match="provenance conflicts"):
        pipeline_module.prepare_table_inputs(config)


@pytest.mark.parametrize(
    ("failure_stage", "failed_raster_count", "message"),
    [
        ("raster_validate", 1, "failed before a trustworthy retained raster"),
        ("inference", 0, "has no retained raster"),
        ("inference", 2, "exactly one retained raster"),
    ],
)
def test_raw_extraction_source_rejects_untrustworthy_or_ambiguous_failed_page(
    tmp_path: Path,
    failure_stage: str,
    failed_raster_count: int,
    message: str,
) -> None:
    config = _raw_extraction_config(
        tmp_path,
        failure_stage=failure_stage,
        failed_raster_count=failed_raster_count,
    )

    with pytest.raises(pipeline_module.TableViewError, match=message):
        pipeline_module.prepare_table_inputs(config)


def test_join_commands_fail_clearly_when_raw_table_view_has_no_join(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    config = _raw_extraction_config(tmp_path)

    with pytest.raises(pipeline_module.TableViewError, match="join is not configured"):
        publish_joined_dataset(config)
    with pytest.raises(pipeline_module.TableViewError, match="join is not configured"):
        verify_joined_dataset(config)

    config_path = tmp_path / "table.yaml"
    config_path.write_text(
        yaml.safe_dump(config.model_dump(mode="json"), sort_keys=False), encoding="utf-8"
    )
    for command in ("join", "verify-join"):
        assert table_view_main([command, "--config", str(config_path)]) == 4
        assert "join is not configured" in capsys.readouterr().err


def test_published_table_loader_accepts_multiline_single_row_summary(
    tmp_path: Path,
) -> None:
    train = tmp_path / "train.jsonl"
    validation = tmp_path / "validation.jsonl"
    train_id = "doc_" + "1" * 64
    validation_id = "doc_" + "2" * 64
    _write_jsonl(train, [_source_row(train_id, "train")])
    _write_jsonl(validation, [_source_row(validation_id, "validation")])
    config = _config(tmp_path, train, validation)
    paths = pipeline_module.table_run_paths(config)

    input_pages: list[TableInputPage] = []
    results: list[TablePageResult] = []
    for index, document_id in enumerate((train_id, validation_id), start=1):
        table_view_id = f"table-{index}"
        page = TableInputPage.model_validate(
            {
                "schemaVersion": 1,
                "recordSetId": "records",
                "annotationPath": str(tmp_path / f"annotation-{index}.json"),
                "annotationSha256": "d" * 64,
                "documentId": document_id,
                "documentPageCount": 1,
                "pageIndex": 0,
                "pageNumber": 1,
                "pageId": f"page-{index}",
                "baseExtractionId": f"extract-{index}",
                "baseExtractionRunId": "source-v1",
                "baseRawOcrTextSha256": "c" * 64,
                "sourceSha256": "a" * 64,
                "rasterPath": str(tmp_path / f"page-{index}.png"),
                "rasterMimeType": "image/png",
                "rasterSizeBytes": 10,
                "rasterSha256": "e" * 64,
                "tableViewId": table_view_id,
            },
            strict=True,
        )
        table_text = f"<table><tr><td>{index}</td></tr></table>"
        result = TablePageResult.model_validate(
            {
                "schemaVersion": 1,
                "tableViewId": table_view_id,
                "inputPageSha256": canonical_json_sha256(page.model_dump(mode="json")),
                "documentId": document_id,
                "documentPageCount": 1,
                "pageIndex": 0,
                "pageNumber": 1,
                "pageId": f"page-{index}",
                "baseExtractionId": f"extract-{index}",
                "baseExtractionRunId": "source-v1",
                "rasterPath": str(tmp_path / f"page-{index}.png"),
                "rasterSha256": "e" * 64,
                "prompt": "Table Recognition:",
                "promptSha256": sha256_bytes(b"Table Recognition:"),
                "model": "zai-org/GLM-OCR",
                "modelRevision": "a" * 40,
                "servedModelName": "glm-ocr",
                "vllmEngineVersion": "0.26.0",
                "serverContractSha256": "f" * 64,
                "responseId": f"response-{index}",
                "inferenceRequestId": f"request-{index}",
                "inferenceServerRequestId": f"server-{index}",
                "finishReason": "stop",
                "promptTokens": 1,
                "completionTokens": 2,
                "totalTokens": 3,
                "attemptCount": 1,
                "inferenceDurationMs": 1.0,
                "rawTableText": table_text,
                "rawTableTextSha256": sha256_bytes(table_text.encode()),
                "rawResponsePath": f"state/raw/response-{index}.json",
                "rawResponseSha256": "a" * 64,
                "attemptPaths": (f"state/attempts/attempt-{index}.json",),
                "attemptSha256s": ("b" * 64,),
            },
            strict=True,
        )
        input_pages.append(page)
        results.append(result)

    input_payload = b"".join(
        canonical_json_bytes(page.model_dump(mode="json")) + b"\n" for page in input_pages
    )
    paths.input_pages.write_bytes(input_payload)
    paths.input_pages_digest.write_text(
        f"{sha256_bytes(input_payload)}  {paths.input_pages.name}\n", encoding="ascii"
    )
    table_payload = b"".join(
        canonical_json_bytes(result.model_dump(mode="json")) + b"\n" for result in results
    )
    attempts_payload = b'{"outcome":"success"}\n{"outcome":"success"}\n'
    quality_summary = {"documents": 2, "pages": 2}
    quality_payload = (json.dumps(quality_summary, indent=2, sort_keys=True) + "\n").encode()
    quality_flags_payload = b""
    artifacts = {
        "table_pages": ("table-pages.jsonl", table_payload, 2),
        "attempts": ("attempts.jsonl", attempts_payload, 2),
        "quality_summary": ("quality-summary.json", quality_payload, 1),
        "quality_flags": ("quality-flags.jsonl", quality_flags_payload, 0),
    }
    files: list[dict[str, Any]] = []
    for kind, (name, payload, row_count) in artifacts.items():
        (paths.dataset_root / name).write_bytes(payload)
        files.append(
            {
                "kind": kind,
                "path": f"dataset/{name}",
                "rows": row_count,
                "bytes": len(payload),
                "sha256": sha256_bytes(payload),
            }
        )
    manifest = {
        "schema_version": 1,
        "run_id": "table-v1",
        "view_type": "glm_ocr_table_recognition",
        "config_sha256": canonical_json_sha256(config.model_dump(mode="json")),
        "input_pages_sha256": sha256_bytes(input_payload),
        "documents": 2,
        "pages": 2,
        "attempts": 2,
        "quality_summary": quality_summary,
        "files": files,
    }
    (paths.dataset_root / "manifest.json").write_bytes(canonical_json_bytes(manifest) + b"\n")

    loaded = join_module._published_results(config)

    assert [row.tableViewId for row in loaded] == ["table-1", "table-2"]


def test_join_clones_rows_without_changing_training_input_or_target(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    train_id = "doc_" + "1" * 64
    validation_id = "doc_" + "2" * 64
    train = tmp_path / "train.jsonl"
    validation = tmp_path / "validation.jsonl"
    train_row = _source_row(train_id, "train raw OCR")
    validation_row = _source_row(validation_id, "validation raw OCR")
    train_row["target"] = {
        "schemaVersion": "2.0.0",
        "documentPatch": {"billOfLadingNumber": "TRAIN-BL-1"},
    }
    validation_row["target"] = {
        "schemaVersion": "2.0.0",
        "documentPatch": {"billOfLadingNumber": "VALIDATION-BL-1"},
    }
    _write_jsonl(train, [train_row])
    _write_jsonl(validation, [validation_row])
    config = _config(tmp_path, train, validation)

    dataset_root = tmp_path / "table-dataset"
    dataset_root.mkdir()
    (dataset_root / "manifest.json").write_text('{"pages":2}\n', encoding="utf-8")
    views = {
        train_id: _view(train_id, "page-train", "| A | B |"),
        validation_id: _view(validation_id, "page-validation", "| C | D |"),
    }
    monkeypatch.setattr(
        join_module, "table_run_paths", lambda _: SimpleNamespace(dataset_root=dataset_root)
    )
    monkeypatch.setattr(join_module, "_document_views", lambda _: views)

    manifest = publish_joined_dataset(config)
    assert manifest == Path(config.join.output_root) / "manifest.json"
    assert verify_joined_dataset(config) == {
        "documents": 2,
        "pages": 2,
        "train_records": 1,
        "validation_records": 1,
        "test_records": 0,
    }
    cloned = json.loads((Path(config.join.output_root) / "train.jsonl").read_text())
    auxiliary = cloned.pop("auxiliaryViews")
    assert cloned == train_row
    assert auxiliary["glmOcrTableRecognition"]["joinedTableText"] == ("--- PAGE 1 ---\n| A | B |")

    project_root = Path(__file__).resolve().parents[1]
    training_config = load_training_config(
        project_root / "configs/training/t5gemma2_270m_lora.pilot106.yaml"
    )
    joined_train = Path(config.join.output_root) / "train.jsonl"
    training_config = training_config.model_copy(
        update={
            "dataset": training_config.dataset.model_copy(
                update={
                    "splits": DatasetSplitsConfig(
                        train=[
                            DatasetFileConfig(
                                path=str(joined_train),
                                sha256=sha256_bytes(joined_train.read_bytes()),
                                records=1,
                            )
                        ],
                        validation=[],
                        test=[],
                    )
                }
            )
        }
    )
    task = get_training_task(training_config.task)
    prompt = load_prompt(project_root, training_config.prompt, task)
    records, _ = inspect_dataset(
        project_root=project_root,
        config=training_config,
        prompt=prompt,
        task=task,
    )
    rendered = records["train"][0].input_text
    assert "train raw OCR" in rendered
    assert "| A | B |" not in rendered
