from __future__ import annotations

import asyncio
import json
from concurrent.futures import Executor, Future
from dataclasses import replace
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Literal

import pyarrow.parquet as pq
import pytest
from pypdf import PdfWriter
from test_config import valid_config_data

import document_ocr.pipeline as pipeline_module
from document_ocr.client import OcrResponse, RequestAttempt, ServerInfo, VllmRasterError
from document_ocr.config import PipelineConfig
from document_ocr.hashing import canonical_json_bytes, sha256_bytes, sha256_file
from document_ocr.ledger import ExtractionLedger, IncompleteRunError
from document_ocr.pipeline import (
    PipelineError,
    PipelineProgress,
    require_run_complete,
    run_pipeline,
)
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


class FakeOcrClient:
    def __init__(self, *, reject_readiness: bool = False) -> None:
        self.reject_readiness = reject_readiness
        self.readiness_calls = 0
        self.request_calls = 0
        self.active_requests = 0
        self.maximum_active_requests = 0
        self.raster_paths: list[Path] = []

    async def check_readiness(self) -> ServerInfo:
        self.readiness_calls += 1
        if self.reject_readiness:
            raise AssertionError("a completed resume must not contact vLLM")
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
        raster_bytes = await asyncio.to_thread(raster_path.read_bytes)
        assert raster_bytes.startswith(b"\x89PNG\r\n\x1a\n")
        assert sha256_bytes(raster_bytes) == raster_sha256
        self.raster_paths.append(raster_path)
        page_index = (
            int(raster_path.stem.removeprefix("page-"))
            if raster_path.stem.startswith("page-")
            else self.request_calls
        )
        self.request_calls += 1
        self.active_requests += 1
        self.maximum_active_requests = max(self.maximum_active_requests, self.active_requests)
        try:
            await asyncio.sleep(0.02)
        finally:
            self.active_requests -= 1

        text = f"raw page {page_index + 1}\n"
        raw = canonical_json_bytes(
            {
                "id": f"response-{page_index}",
                "model": "glm-ocr",
                "text": text,
            }
        )
        now = datetime.now(UTC)
        attempt_request_id = f"{request_id}-a{first_attempt_number}"
        attempt = RequestAttempt(
            attempt_number=first_attempt_number,
            attempt_started_at=now,
            attempt_completed_at=now,
            attempt_duration_ms=20.0,
            outcome="success",
            retryable=False,
            inference_request_id=attempt_request_id,
            inference_server_request_id=f"server-{page_index}",
            http_status_code=200,
        )
        return OcrResponse(
            text=text,
            finish_reason="stop",
            request_id=attempt_request_id,
            response_id=f"response-{page_index}",
            response_model="glm-ocr",
            server_request_id=f"server-{page_index}",
            prompt_tokens=100,
            completion_tokens=10,
            total_tokens=110,
            raw_response=raw,
            raw_response_sha256=sha256_bytes(raw),
            attempts=(attempt,),
        )


class RasterRejectingClient(FakeOcrClient):
    async def recognize_page(
        self,
        raster_path: Path,
        *,
        mime_type: Literal["image/png", "image/jpeg"],
        raster_sha256: str,
        request_id: str,
        first_attempt_number: int = 1,
    ) -> OcrResponse:
        del raster_path, mime_type, raster_sha256, request_id, first_attempt_number
        raise VllmRasterError("synthetic raster identity drift")


class InlineExecutor(Executor):
    def submit(self, fn: Any, /, *args: Any, **kwargs: Any) -> Future[Any]:
        future: Future[Any] = Future()
        try:
            future.set_result(fn(*args, **kwargs))
        except BaseException as error:
            future.set_exception(error)
        return future


def _write_pdf(path: Path, page_count: int) -> None:
    writer = PdfWriter()
    for _ in range(page_count):
        writer.add_blank_page(width=72, height=72)
    with path.open("wb") as stream:
        writer.write(stream)


def _pipeline_config(source_root: Path, output_root: Path) -> PipelineConfig:
    data = valid_config_data()
    data["source"] = {
        "type": "local",
        "root": str(source_root),
        "include_glob": "**/*.pdf",
        "dataset_version": "pipeline-test-v1",
        "require_content_sha256": True,
    }
    data["output"]["root"] = str(output_root)
    data["run"] = {
        "run_id": "pipeline-e2e",
        "fail_fast": True,
        "resume": True,
    }
    data["raster"] = {
        "dpi": 72,
        "max_side_pixels": 256,
        "max_pixels": 65_536,
        "max_pages_per_document": 10,
        "max_pdf_bytes": 1_000_000,
        "max_cached_documents_per_process": 1,
        "image_format": "png",
        "jpeg_quality": None,
        "draw_annotations": True,
        "reject_xfa": True,
    }
    data["concurrency"] = {
        "max_active_documents": 1,
        "renderer_processes": 2,
        "max_inflight_pages_global": 2,
        "max_inflight_pages_per_document": 2,
    }
    return PipelineConfig.model_validate(data, strict=True)


@pytest.mark.asyncio
async def test_pipeline_extracts_real_pdf_pages_and_resumes_without_server(
    tmp_path: Path,
) -> None:
    source_root = tmp_path / "source"
    output_root = tmp_path / "output"
    source_root.mkdir()
    _write_pdf(source_root / "four-pages.pdf", page_count=4)
    config = _pipeline_config(source_root, output_root)
    project_root = Path(__file__).parents[1]
    client = FakeOcrClient()
    progress_events: list[PipelineProgress] = []

    result = await run_pipeline(
        project_root=project_root,
        config=config,
        ocr_client=client,
        progress=progress_events.append,
    )

    assert result.resumed_complete_run is False
    assert result.server_info == _server_info()
    assert client.readiness_calls == 1
    assert client.request_calls == 4
    assert client.maximum_active_requests == 2
    assert result.summary["successful_pages"] == 4
    assert result.summary["audited_successful_pages"] == 4
    assert result.summary["attempt_rows"] == 4
    assert progress_events == [
        PipelineProgress(
            phase="checking_server",
            total_documents=1,
            processed_documents=0,
        ),
        PipelineProgress(
            phase="extracting",
            total_documents=1,
            processed_documents=1,
        ),
        PipelineProgress(
            phase="publishing",
            total_documents=1,
            processed_documents=1,
        ),
    ]

    manifest_bytes = result.dataset_manifest.read_bytes()
    manifest = json.loads(manifest_bytes)
    assert manifest_bytes == canonical_json_bytes(manifest) + b"\n"
    assert manifest["dataset_kind"] == "glm-ocr-raw-page-extractions"
    assert manifest["summary"] == result.summary
    assert manifest["raw_responses"]["artifacts"] == 4
    assert [item["rows"] for item in manifest["files"]] == [4, 4]

    run_root = output_root / "runs" / config.run.run_id
    for item in manifest["files"]:
        artifact = run_root / item["path"]
        assert artifact.stat().st_size == item["bytes"]
        assert sha256_file(artifact) == item["sha256"]
    page_artifact = run_root / manifest["files"][0]["path"]
    pages = pq.read_table(
        page_artifact,
        columns=[
            "document_id",
            "document_page_count",
            "page_id",
            "page_index",
            "page_number",
            "raw_ocr_text",
            "inference_server_contract_sha256",
            "raw_response_sha256",
            "raw_response_path",
        ],
    ).to_pylist()
    assert [page["page_index"] for page in pages] == [0, 1, 2, 3]
    assert [page["page_number"] for page in pages] == [1, 2, 3, 4]
    assert {page["document_id"] for page in pages} == {pages[0]["document_id"]}
    assert {page["document_page_count"] for page in pages} == {4}
    assert [page["raw_ocr_text"] for page in pages] == [
        "raw page 1\n",
        "raw page 2\n",
        "raw page 3\n",
        "raw page 4\n",
    ]
    assert {page["inference_server_contract_sha256"] for page in pages} == {
        _server_info().contract_sha256
    }
    for page in pages:
        relative = Path(page["raw_response_path"])
        raw_response = run_root / relative
        assert raw_response.is_file()
        assert relative.parts == (
            "raw-responses",
            page["document_id"],
            page["page_id"],
            f"{page['raw_response_sha256']}.json",
        )
        assert sha256_file(raw_response) == page["raw_response_sha256"]

    completed_client = FakeOcrClient(reject_readiness=True)
    resumed = await run_pipeline(
        project_root=project_root,
        config=config,
        ocr_client=completed_client,
    )
    assert resumed.resumed_complete_run is True
    assert resumed.server_info is None
    assert resumed.dataset_manifest == result.dataset_manifest
    assert resumed.summary == result.summary
    assert completed_client.readiness_calls == 0
    assert completed_client.request_calls == 0

    before_status = {
        path.relative_to(run_root): (path.stat().st_size, path.stat().st_mtime_ns)
        for path in run_root.rglob("*")
        if path.is_file()
    }
    assert await require_run_complete(config) == result.summary
    after_status = {
        path.relative_to(run_root): (path.stat().st_size, path.stat().st_mtime_ns)
        for path in run_root.rglob("*")
        if path.is_file()
    }
    assert after_status == before_status


@pytest.mark.asyncio
async def test_pipeline_retains_content_addressed_page_image_and_exports_pointer(
    tmp_path: Path,
) -> None:
    source_root = tmp_path / "source"
    output_root = tmp_path / "output"
    source_root.mkdir()
    source_pdf = source_root / "one-page.pdf"
    _write_pdf(source_pdf, page_count=1)
    config = _pipeline_config(source_root, output_root)
    config = PipelineConfig.model_validate(
        {
            **config.model_dump(mode="python"),
            "output": {
                **config.output.model_dump(mode="python"),
                "retain_page_images": True,
            },
        },
        strict=True,
    )
    client = FakeOcrClient()

    result = await run_pipeline(
        project_root=Path(__file__).parents[1],
        config=config,
        ocr_client=client,
        executor=InlineExecutor(),
    )

    run_root = output_root / "runs" / config.run.run_id
    manifest = json.loads(result.dataset_manifest.read_bytes())
    assert manifest["page_images"]["retained"] is True
    assert manifest["page_images"]["artifacts"] == 1
    assert manifest["page_images"]["bytes"] > 0
    pages_path = run_root / manifest["files"][0]["path"]
    page = pq.read_table(
        pages_path,
        columns=[
            "local_canonical_path",
            "local_relative_key",
            "source_sha256",
            "raster_path",
            "raster_sha256",
            "raster_size_bytes",
        ],
    ).to_pylist()[0]
    raster_relative = Path(page["raster_path"])
    raster = run_root / raster_relative
    assert raster_relative.parts[:1] == ("page-images",)
    assert raster.is_file()
    assert raster.stat().st_size == page["raster_size_bytes"]
    assert sha256_file(raster) == page["raster_sha256"]
    assert page["local_canonical_path"] == str(source_pdf.resolve())
    assert page["local_relative_key"] == source_pdf.name
    assert page["source_sha256"] == sha256_file(source_pdf)
    assert client.raster_paths == [raster]
    assert not list((run_root / "scratch" / "rasters").rglob("*.png"))


@pytest.mark.asyncio
async def test_status_for_missing_run_does_not_create_output_directories(
    tmp_path: Path,
) -> None:
    source_root = tmp_path / "source"
    source_root.mkdir()
    output_root = tmp_path / "absent-output"
    config = _pipeline_config(source_root, output_root)

    with pytest.raises(IncompleteRunError, match="run ledger does not exist"):
        await require_run_complete(config)

    assert not output_root.exists()


@pytest.mark.asyncio
async def test_raster_identity_failure_is_durable_and_never_creates_dataset(
    tmp_path: Path,
) -> None:
    source_root = tmp_path / "source"
    output_root = tmp_path / "output"
    source_root.mkdir()
    _write_pdf(source_root / "one-page.pdf", page_count=1)
    config = _pipeline_config(source_root, output_root)
    data = config.model_dump(mode="python")
    data["run"]["fail_fast"] = False
    config = PipelineConfig.model_validate(data, strict=True)

    with pytest.raises(IncompleteRunError, match="incomplete"):
        await run_pipeline(
            project_root=Path(__file__).parents[1],
            config=config,
            ocr_client=RasterRejectingClient(),
        )

    run_root = output_root / "runs" / config.run.run_id
    assert not (run_root / "dataset" / "manifest.json").exists()
    async with ExtractionLedger(run_root / "state.sqlite3") as ledger:
        failures = [row async for row in ledger.iter_page_failures(config.run.run_id)]
        attempts = [row async for row in ledger.iter_attempts(config.run.run_id)]
    assert len(failures) == 1
    assert failures[0]["failure_stage"] == "raster_validate"
    assert failures[0]["inference_attempt_count"] == 0
    assert failures[0]["last_inference_request_id"] is None
    assert attempts == []


@pytest.mark.asyncio
async def test_rendered_page_must_match_initially_inspected_source_identity(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source_root = tmp_path / "source"
    output_root = tmp_path / "output"
    source_root.mkdir()
    _write_pdf(source_root / "one-page.pdf", page_count=1)
    config = _pipeline_config(source_root, output_root)
    data = config.model_dump(mode="python")
    data["run"]["fail_fast"] = False
    config = PipelineConfig.model_validate(data, strict=True)
    original_render_page = pipeline_module.render_page

    def return_drifted_document_identity(*args: Any) -> object:
        rendered = original_render_page(*args)
        drifted_document = replace(
            rendered.document,
            source_ctime_ns=rendered.document.source_ctime_ns + 1,
        )
        return replace(rendered, document=drifted_document)

    monkeypatch.setattr(pipeline_module, "render_page", return_drifted_document_identity)
    client = FakeOcrClient()
    with pytest.raises(IncompleteRunError, match="incomplete"):
        await run_pipeline(
            project_root=Path(__file__).parents[1],
            config=config,
            ocr_client=client,
            executor=InlineExecutor(),
        )
    pipeline_module.close_process_document_cache()

    assert client.request_calls == 0
    run_root = output_root / "runs" / config.run.run_id
    async with ExtractionLedger(run_root / "state.sqlite3") as ledger:
        failures = [row async for row in ledger.iter_page_failures(config.run.run_id)]
    assert len(failures) == 1
    assert failures[0]["failure_stage"] == "render"
    assert failures[0]["error_type"] == "SourceChangedError"


@pytest.mark.asyncio
async def test_persistence_failure_retains_attempt_and_resumes_cleanly(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source_root = tmp_path / "source"
    output_root = tmp_path / "output"
    source_root.mkdir()
    _write_pdf(source_root / "one-page.pdf", page_count=1)
    config = _pipeline_config(source_root, output_root)
    data = config.model_dump(mode="python")
    data["run"]["fail_fast"] = False
    config = PipelineConfig.model_validate(data, strict=True)
    original_publish = pipeline_module.atomic_publish_bytes

    def fail_raw_response_publish(_path: Path, _payload: bytes) -> bool:
        raise OSError("synthetic durable storage failure")

    monkeypatch.setattr(pipeline_module, "atomic_publish_bytes", fail_raw_response_publish)
    with pytest.raises(IncompleteRunError, match="incomplete"):
        await run_pipeline(
            project_root=Path(__file__).parents[1],
            config=config,
            ocr_client=FakeOcrClient(),
        )

    run_root = output_root / "runs" / config.run.run_id
    async with ExtractionLedger(run_root / "state.sqlite3") as ledger:
        failures = [row async for row in ledger.iter_page_failures(config.run.run_id)]
        attempts = [row async for row in ledger.iter_attempts(config.run.run_id)]
    assert failures[0]["failure_stage"] == "persist"
    assert failures[0]["inference_attempt_count"] == 1
    assert [attempt["attempt_outcome"] for attempt in attempts] == ["success"]

    monkeypatch.setattr(pipeline_module, "atomic_publish_bytes", original_publish)
    result = await run_pipeline(
        project_root=Path(__file__).parents[1],
        config=config,
        ocr_client=FakeOcrClient(),
    )

    assert result.summary["successful_pages"] == 1
    assert result.summary["attempt_rows"] == 2
    manifest = json.loads(result.dataset_manifest.read_bytes())
    attempts_path = run_root / manifest["files"][1]["path"]
    attempt_rows = pq.read_table(
        attempts_path,
        columns=["attempt_number", "attempt_outcome", "inference_request_id"],
    ).to_pylist()
    assert [(row["attempt_number"], row["attempt_outcome"]) for row in attempt_rows] == [
        (1, "success"),
        (2, "success"),
    ]
    assert attempt_rows[0]["inference_request_id"].endswith("-a1")
    assert attempt_rows[1]["inference_request_id"].endswith("-a2")


@pytest.mark.asyncio
async def test_pipeline_refuses_symlinked_immutable_provenance(tmp_path: Path) -> None:
    source_root = tmp_path / "source"
    output_root = tmp_path / "output"
    source_root.mkdir()
    _write_pdf(source_root / "one-page.pdf", page_count=1)
    config = _pipeline_config(source_root, output_root)
    run_root = output_root / "runs" / config.run.run_id
    run_root.mkdir(parents=True)
    destination = tmp_path / "untrusted-provenance.json"
    destination.write_text("{}", encoding="utf-8")
    (run_root / "run-provenance.json").symlink_to(destination)

    with pytest.raises(PipelineError, match="cannot load immutable JSON artifact"):
        await run_pipeline(
            project_root=Path(__file__).parents[1],
            config=config,
            ocr_client=FakeOcrClient(),
        )

    assert destination.read_text(encoding="utf-8") == "{}"
