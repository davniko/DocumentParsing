from __future__ import annotations

import asyncio
import hashlib
import json
import re
from collections import defaultdict
from dataclasses import replace
from datetime import UTC, datetime
from pathlib import Path
from typing import Literal

import pytest
from PIL import Image
from test_config import valid_config_data

import document_ocr.benchmark as benchmark_module
from document_ocr.benchmark import (
    BenchmarkError,
    BenchmarkExecutionError,
    RasterManifestError,
    load_raster_manifest,
    run_benchmark,
)
from document_ocr.client import OcrResponse, RequestAttempt, ServerInfo
from document_ocr.config import PipelineConfig
from document_ocr.hashing import canonical_json_bytes, canonical_json_sha256, sha256_bytes
from document_ocr.vllm_contract import runtime_contract_payload, runtime_contract_sha256

MODEL_REVISION = "c" * 40
CONTAINER_IMAGE = (
    "vllm/vllm-openai:v0.26.0@sha256:"
    "ffb2d59b1c059a5bd8d781320c9f5189de8293693b7d95da54befddaa54abf52"
)
BUILD_MANIFEST_SHA256 = "d" * 64


def _server_info() -> ServerInfo:
    payload = runtime_contract_payload(
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
        container_base_image=CONTAINER_IMAGE,
        container_build_manifest_sha256=BUILD_MANIFEST_SHA256,
    )
    return ServerInfo(
        version="0.26.0",
        models=("glm-ocr",),
        model_repository="zai-org/GLM-OCR",
        served_model_name="glm-ocr",
        model_revision=MODEL_REVISION,
        max_model_len=32768,
        max_num_seqs=16,
        gpu_memory_utilization=0.9,
        generation_config="vllm",
        speculative_method="mtp",
        num_speculative_tokens=1,
        image_limit_per_prompt=1,
        container_base_image=CONTAINER_IMAGE,
        container_build_manifest_sha256=BUILD_MANIFEST_SHA256,
        contract_sha256=runtime_contract_sha256(payload),
    )


def _benchmark_config(
    *,
    points: list[tuple[int, int]] | None = None,
    warmup_pages: int = 2,
    measured_pages: int = 8,
    repetitions: int = 2,
) -> PipelineConfig:
    raw = valid_config_data()
    raw["benchmark"] = {
        "points": [
            {
                "max_inflight_pages_global": global_limit,
                "max_inflight_pages_per_document": document_limit,
            }
            for global_limit, document_limit in (points or [(3, 1), (4, 2)])
        ],
        "warmup_pages": warmup_pages,
        "measured_pages": measured_pages,
        "repetitions": repetitions,
    }
    return PipelineConfig.model_validate(raw, strict=True)


def _make_png(path: Path, color: tuple[int, int, int]) -> str:
    Image.new("RGB", (16, 12), color=color).save(path, format="PNG")
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_manifest(
    path: Path,
    entries: list[dict[str, str]],
) -> None:
    path.write_text(
        "".join(json.dumps(entry, sort_keys=False) + "\n" for entry in entries),
        encoding="utf-8",
    )


def _manifest_entries(tmp_path: Path, count: int = 4) -> list[dict[str, str]]:
    entries: list[dict[str, str]] = []
    for index in range(count):
        raster = tmp_path / f"page-{index}.png"
        raster_sha256 = _make_png(raster, (index * 20, index * 30, index * 40))
        entries.append(
            {
                "page_id": f"page-{index}",
                "document_id": "document-a" if index < 2 else "document-b",
                "raster_path": str(raster),
                "mime_type": "image/png",
                "raster_sha256": raster_sha256,
            }
        )
    return entries


class FakeBenchmarkClient:
    def __init__(
        self,
        entries: list[dict[str, str]],
        *,
        delay_seconds: float = 0.01,
        inconsistent_text: bool = False,
        fail_requests: bool = False,
        mutate_raster: bool = False,
        retries: bool = True,
    ) -> None:
        self.documents = {
            str(Path(entry["raster_path"]).resolve()): entry["document_id"] for entry in entries
        }
        self.pages = {
            str(Path(entry["raster_path"]).resolve()): entry["page_id"] for entry in entries
        }
        self.raster_hashes = {
            str(Path(entry["raster_path"]).resolve()): entry["raster_sha256"] for entry in entries
        }
        self.delay_seconds = delay_seconds
        self.inconsistent_text = inconsistent_text
        self.fail_requests = fail_requests
        self.mutate_raster = mutate_raster
        self.retries = retries
        self.readiness_calls = 0
        self.calls: list[str] = []
        self.page_calls: defaultdict[str, int] = defaultdict(int)
        self.active_by_point: defaultdict[int, int] = defaultdict(int)
        self.max_active_by_point: defaultdict[int, int] = defaultdict(int)
        self.active_by_point_document: defaultdict[tuple[int, str], int] = defaultdict(int)
        self.max_active_by_point_document: defaultdict[tuple[int, str], int] = defaultdict(int)

    async def check_readiness(self) -> ServerInfo:
        self.readiness_calls += 1
        return _server_info()

    async def recognize_page(
        self,
        raster_path: Path,
        *,
        mime_type: Literal["image/png", "image/jpeg"],
        raster_sha256: str,
        request_id: str,
        first_attempt_number: int = 1,
    ) -> OcrResponse:
        assert mime_type == "image/png"
        assert first_attempt_number == 1
        canonical_path = str(await asyncio.to_thread(raster_path.resolve))
        assert raster_sha256 == self.raster_hashes[canonical_path]
        document_id = self.documents[canonical_path]
        page_id = self.pages[canonical_path]
        match = re.search(r"-p(\d{3})-", request_id)
        assert match is not None
        point_index = int(match.group(1))
        point_document = (point_index, document_id)
        self.calls.append(request_id)
        self.active_by_point[point_index] += 1
        self.active_by_point_document[point_document] += 1
        self.max_active_by_point[point_index] = max(
            self.max_active_by_point[point_index], self.active_by_point[point_index]
        )
        self.max_active_by_point_document[point_document] = max(
            self.max_active_by_point_document[point_document],
            self.active_by_point_document[point_document],
        )
        try:
            await asyncio.sleep(self.delay_seconds)
            if self.fail_requests:
                raise RuntimeError("synthetic client failure")
            if self.mutate_raster:
                await asyncio.to_thread(
                    raster_path.write_bytes, b"mutated while request was active"
                )
        finally:
            self.active_by_point[point_index] -= 1
            self.active_by_point_document[point_document] -= 1

        self.page_calls[page_id] += 1
        text = f"OCR text for {page_id}"
        if self.inconsistent_text and self.page_calls[page_id] > 1:
            text += " changed"
        raw_response = json.dumps(
            {"id": request_id, "page_id": page_id, "text": text},
            separators=(",", ":"),
            sort_keys=True,
        ).encode()
        now = datetime.now(UTC)
        attempts: list[RequestAttempt] = []
        if self.retries and page_id.endswith(("1", "3")):
            attempts.append(
                RequestAttempt(
                    attempt_number=1,
                    attempt_started_at=now,
                    attempt_completed_at=now,
                    attempt_duration_ms=0.1,
                    outcome="http_error",
                    retryable=True,
                    inference_request_id=f"{request_id}-a1",
                    http_status_code=503,
                    error_type="HTTPStatusError",
                    error_message="vLLM returned HTTP 503",
                )
            )
        final_attempt_number = len(attempts) + 1
        final_request_id = f"{request_id}-a{final_attempt_number}"
        attempts.append(
            RequestAttempt(
                attempt_number=final_attempt_number,
                attempt_started_at=now,
                attempt_completed_at=now,
                attempt_duration_ms=1.0,
                outcome="success",
                retryable=False,
                inference_request_id=final_request_id,
                inference_server_request_id=f"server-{final_request_id}",
                http_status_code=200,
            )
        )
        return OcrResponse(
            text=text,
            finish_reason="stop",
            request_id=final_request_id,
            response_id=f"response-{final_request_id}",
            response_model="glm-ocr",
            server_request_id=f"server-{final_request_id}",
            prompt_tokens=10,
            completion_tokens=20,
            total_tokens=30,
            raw_response=raw_response,
            raw_response_sha256=sha256_bytes(raw_response),
            attempts=tuple(attempts),
        )


class CrossDocumentStartClient(FakeBenchmarkClient):
    """Require two documents to reach the client before either can complete."""

    def __init__(self, entries: list[dict[str, str]]) -> None:
        super().__init__(entries, retries=False)
        self.started_documents: set[str] = set()
        self.cross_document_start = asyncio.Event()

    async def recognize_page(
        self,
        raster_path: Path,
        *,
        mime_type: Literal["image/png", "image/jpeg"],
        raster_sha256: str,
        request_id: str,
        first_attempt_number: int = 1,
    ) -> OcrResponse:
        canonical_path = str(await asyncio.to_thread(raster_path.resolve))
        self.started_documents.add(self.documents[canonical_path])
        if len(self.started_documents) == 2:
            self.cross_document_start.set()
        await asyncio.wait_for(self.cross_document_start.wait(), timeout=1.0)
        return await super().recognize_page(
            raster_path,
            mime_type=mime_type,
            raster_sha256=raster_sha256,
            request_id=request_id,
            first_attempt_number=first_attempt_number,
        )


class DriftedServerClient(FakeBenchmarkClient):
    async def check_readiness(self) -> ServerInfo:
        self.readiness_calls += 1
        return replace(_server_info(), max_num_seqs=8)


@pytest.mark.asyncio
async def test_benchmark_honors_limits_and_publishes_complete_canonical_report(
    tmp_path: Path,
) -> None:
    entries = _manifest_entries(tmp_path)
    manifest_path = tmp_path / "rasters.jsonl"
    _write_manifest(manifest_path, entries)
    config = _benchmark_config()
    client = FakeBenchmarkClient(entries)
    report_path = tmp_path / "reports" / "benchmark.json"

    published = await run_benchmark(
        config=config,
        raster_manifest_path=manifest_path,
        report_path=report_path,
        client=client,
    )

    assert published.created is True
    assert published.path == report_path
    report_bytes = report_path.read_bytes()
    report = json.loads(report_bytes)
    assert report_bytes == canonical_json_bytes(report) + b"\n"
    assert published.sha256 == hashlib.sha256(report_bytes).hexdigest()
    assert client.readiness_calls == 1
    assert len(client.calls) == 2 * 2 * (2 + 8)

    assert client.max_active_by_point[0] <= 3
    assert client.max_active_by_point[1] <= 4
    assert client.max_active_by_point[0] >= 2
    assert client.max_active_by_point[1] >= 3
    for (point_index, _document_id), maximum in client.max_active_by_point_document.items():
        assert maximum <= (1 if point_index == 0 else 2)

    assert report["resolved_config_sha256"] == canonical_json_sha256(config.model_dump(mode="json"))
    assert report["benchmark_config_sha256"] == canonical_json_sha256(
        config.benchmark.model_dump(mode="json")
    )
    assert report["harness_identity_sha256"] == canonical_json_sha256(report["harness_identity"])
    assert set(report["harness_identity"]["source_files_sha256"]) == {
        "atomic.py",
        "benchmark.py",
        "client.py",
        "config.py",
        "hashing.py",
        "vllm_contract.py",
    }
    assert "document-ocr-pipeline" in report["harness_identity"]["distributions"]
    assert report["server_identity"]["observed"]["max_model_len"] == 32768
    assert report["server_identity_sha256"] == canonical_json_sha256(report["server_identity"])
    loaded = load_raster_manifest(manifest_path)
    assert report["raster_manifest"]["source_file_sha256"] == loaded.source_file_sha256
    assert report["raster_manifest"]["canonical_sha256"] == loaded.canonical_sha256
    assert report["raster_manifest"]["entry_count"] == 4
    assert len(report["repetitions"]) == 4
    assert report["text_consistency"]["page_count_observed"] == 4
    assert list(report["text_consistency"]["text_sha256_by_page_id"]) == [
        "page-0",
        "page-1",
        "page-2",
        "page-3",
    ]

    for repetition in report["repetitions"]:
        assert repetition["warmup"]["pages"] == 2
        measured = repetition["measured"]
        assert measured["pages"] == 8
        assert measured["prompt_tokens"] == 80
        assert measured["completion_tokens"] == 160
        assert measured["total_tokens"] == 240
        assert measured["retries"] == 4
        assert measured["pages_per_second"] > 0
        assert set(measured["request_latency_ms"]) == {
            "minimum",
            "mean",
            "p50",
            "p95",
            "p99",
            "maximum",
        }
        assert [request["sequence_index"] for request in measured["requests"]] == list(range(8))
        assert [request["manifest_index"] for request in measured["requests"]] == [
            2,
            3,
            0,
            1,
            2,
            3,
            0,
            1,
        ]
        prefix = (
            f"bench-{report['benchmark_identity_sha256'][:16]}-"
            f"p{repetition['point_index']:03d}-r{repetition['repetition_index']:03d}-measured"
        )
        assert [request["client_request_id"] for request in measured["requests"]] == [
            f"{prefix}-q{index:08d}" for index in range(8)
        ]


@pytest.mark.asyncio
async def test_grouped_manifest_does_not_block_other_documents_behind_document_cap(
    tmp_path: Path,
) -> None:
    entries = _manifest_entries(tmp_path, count=6)
    for index, entry in enumerate(entries):
        entry["document_id"] = "document-a" if index < 4 else "document-b"
    manifest_path = tmp_path / "grouped-rasters.jsonl"
    _write_manifest(manifest_path, entries)
    client = CrossDocumentStartClient(entries)

    await run_benchmark(
        config=_benchmark_config(points=[(3, 1)], warmup_pages=0, measured_pages=6, repetitions=1),
        raster_manifest_path=manifest_path,
        report_path=tmp_path / "grouped-benchmark.json",
        client=client,
    )

    assert client.started_documents == {"document-a", "document-b"}
    assert client.max_active_by_point[0] >= 2
    assert all(maximum <= 1 for maximum in client.max_active_by_point_document.values())


@pytest.mark.asyncio
async def test_benchmark_rejects_drifted_resolved_server_contract_before_requests(
    tmp_path: Path,
) -> None:
    entries = _manifest_entries(tmp_path, count=1)
    manifest_path = tmp_path / "rasters.jsonl"
    _write_manifest(manifest_path, entries)
    client = DriftedServerClient(entries)
    report_path = tmp_path / "benchmark.json"

    with pytest.raises(BenchmarkExecutionError, match="max_num_seqs"):
        await run_benchmark(
            config=_benchmark_config(
                points=[(1, 1)], warmup_pages=0, measured_pages=1, repetitions=1
            ),
            raster_manifest_path=manifest_path,
            report_path=report_path,
            client=client,
        )

    assert client.calls == []
    assert not report_path.exists()


def test_manifest_rejects_ambiguous_or_incorrect_raster_identity(tmp_path: Path) -> None:
    entries = _manifest_entries(tmp_path, count=2)
    manifest_path = tmp_path / "rasters.jsonl"

    wrong_hash = [dict(entries[0])]
    wrong_hash[0]["raster_sha256"] = "0" * 64
    _write_manifest(manifest_path, wrong_hash)
    with pytest.raises(RasterManifestError, match="SHA-256 does not match"):
        load_raster_manifest(manifest_path)

    duplicate_page = [dict(entries[0]), dict(entries[1])]
    duplicate_page[1]["page_id"] = duplicate_page[0]["page_id"]
    _write_manifest(manifest_path, duplicate_page)
    with pytest.raises(RasterManifestError, match="duplicate page_id"):
        load_raster_manifest(manifest_path)

    wrong_mime = [dict(entries[0])]
    wrong_mime[0]["mime_type"] = "image/jpeg"
    _write_manifest(manifest_path, wrong_mime)
    with pytest.raises(RasterManifestError, match="MIME type"):
        load_raster_manifest(manifest_path)

    relative_path = [dict(entries[0])]
    relative_path[0]["raster_path"] = "page.png"
    _write_manifest(manifest_path, relative_path)
    with pytest.raises(RasterManifestError, match="invalid raster manifest entry"):
        load_raster_manifest(manifest_path)

    manifest_path.write_text(
        '{"page_id":"one","page_id":"two","document_id":"doc",'
        '"raster_path":"/tmp/a.png","mime_type":"image/png",'
        f'"raster_sha256":"{"0" * 64}"}}\n',
        encoding="utf-8",
    )
    with pytest.raises(RasterManifestError, match="invalid raster manifest entry"):
        load_raster_manifest(manifest_path)


@pytest.mark.asyncio
async def test_text_hash_mismatch_aborts_without_publishing_report(tmp_path: Path) -> None:
    entries = _manifest_entries(tmp_path, count=1)
    manifest_path = tmp_path / "rasters.jsonl"
    _write_manifest(manifest_path, entries)
    report_path = tmp_path / "benchmark.json"

    with pytest.raises(BenchmarkExecutionError, match="text hash changed"):
        await run_benchmark(
            config=_benchmark_config(
                points=[(1, 1)], warmup_pages=0, measured_pages=2, repetitions=1
            ),
            raster_manifest_path=manifest_path,
            report_path=report_path,
            client=FakeBenchmarkClient(entries, inconsistent_text=True, retries=False),
        )

    assert not report_path.exists()


@pytest.mark.asyncio
async def test_harness_drift_aborts_without_publishing_report(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    entries = _manifest_entries(tmp_path, count=1)
    manifest_path = tmp_path / "rasters.jsonl"
    _write_manifest(manifest_path, entries)
    report_path = tmp_path / "benchmark.json"
    original_identity = benchmark_module._harness_identity
    calls = 0

    def drifting_identity() -> dict[str, object]:
        nonlocal calls
        calls += 1
        identity = original_identity()
        if calls > 1:
            identity = {**identity, "python_version": "changed-during-benchmark"}
        return identity

    monkeypatch.setattr(benchmark_module, "_harness_identity", drifting_identity)
    with pytest.raises(BenchmarkExecutionError, match="harness changed"):
        await run_benchmark(
            config=_benchmark_config(
                points=[(1, 1)], warmup_pages=0, measured_pages=1, repetitions=1
            ),
            raster_manifest_path=manifest_path,
            report_path=report_path,
            client=FakeBenchmarkClient(entries, retries=False),
        )

    assert calls == 2
    assert not report_path.exists()


@pytest.mark.asyncio
@pytest.mark.parametrize("mode", ["request_failure", "raster_mutation"])
async def test_request_or_inflight_raster_failure_aborts_without_partial_report(
    tmp_path: Path,
    mode: str,
) -> None:
    entries = _manifest_entries(tmp_path, count=1)
    manifest_path = tmp_path / "rasters.jsonl"
    _write_manifest(manifest_path, entries)
    report_path = tmp_path / "benchmark.json"
    client = FakeBenchmarkClient(
        entries,
        fail_requests=mode == "request_failure",
        mutate_raster=mode == "raster_mutation",
        retries=False,
    )

    with pytest.raises(BenchmarkExecutionError):
        await run_benchmark(
            config=_benchmark_config(
                points=[(1, 1)], warmup_pages=0, measured_pages=1, repetitions=1
            ),
            raster_manifest_path=manifest_path,
            report_path=report_path,
            client=client,
        )

    assert not report_path.exists()


@pytest.mark.asyncio
async def test_missing_sweep_fails_before_reading_manifest_or_contacting_client(
    tmp_path: Path,
) -> None:
    config = PipelineConfig.model_validate(valid_config_data(), strict=True)
    client = FakeBenchmarkClient([])

    with pytest.raises(BenchmarkError, match="no benchmark sweep"):
        await run_benchmark(
            config=config,
            raster_manifest_path=tmp_path / "missing.jsonl",
            report_path=tmp_path / "report.json",
            client=client,
        )

    assert client.readiness_calls == 0
