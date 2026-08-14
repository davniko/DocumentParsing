"""Bounded, resumable orchestration for page-level GLM-OCR extraction."""

from __future__ import annotations

import asyncio
import atexit
import json
import multiprocessing
import os
import stat
import time
from collections.abc import Callable
from concurrent.futures import Executor, ProcessPoolExecutor
from contextlib import suppress
from dataclasses import dataclass
from datetime import UTC, datetime
from functools import partial
from pathlib import Path
from typing import Any, Literal, Protocol, cast

import boto3

from document_ocr.atomic import (
    ArtifactReadError,
    AtomicConflictError,
    atomic_publish_bytes,
    atomic_publish_json,
    read_regular_file_bytes,
)
from document_ocr.client import (
    OcrResponse,
    ServerInfo,
    VllmClientError,
    VllmOcrClient,
    VllmRasterError,
)
from document_ocr.config import PipelineConfig, S3SourceConfig
from document_ocr.exporter import PublishedDataset, publish_complete_dataset
from document_ocr.hashing import sha256_bytes, stable_id
from document_ocr.ledger import ExtractionLedger
from document_ocr.models import (
    DocumentExtractionFailure,
    PageExtractionFailure,
    PageExtractionRecord,
    PageProvenance,
    SourceObject,
)
from document_ocr.provenance import collect_runtime_provenance
from document_ocr.renderer import (
    PdfInspection,
    PdfRendererError,
    RenderedPage,
    close_process_document_cache,
    inspect_pdf,
    render_page,
)
from document_ocr.renderer import SourceChangedError as RenderSourceChangedError
from document_ocr.sources import (
    FrozenSourceInventory,
    MaterializedSource,
    SourceError,
    discover_sources,
    freeze_source_inventory,
    load_source_inventory,
    materialize_source,
)


class PipelineError(RuntimeError):
    """The extraction run could not satisfy its completeness contract."""


class RasterPublicationError(PipelineError):
    """A retained page image could not be published as an immutable artifact."""


class _OcrClient(Protocol):
    async def check_readiness(self) -> ServerInfo: ...

    async def recognize_page(
        self,
        raster_path: Path,
        *,
        mime_type: Literal["image/png", "image/jpeg"],
        raster_sha256: str,
        request_id: str,
        first_attempt_number: int = 1,
    ) -> OcrResponse: ...


@dataclass(frozen=True, slots=True)
class RunPaths:
    output_root: Path
    run_root: Path
    inventory: Path
    provenance: Path
    ledger: Path
    source_scratch: Path
    raster_scratch: Path
    page_images: Path


@dataclass(frozen=True, slots=True)
class PipelineResult:
    run_id: str
    resumed_complete_run: bool
    summary: dict[str, int]
    dataset_manifest: Path
    server_info: ServerInfo | None


@dataclass(frozen=True, slots=True)
class PipelineProgress:
    phase: Literal["checking_server", "extracting", "publishing"]
    total_documents: int
    processed_documents: int

    def __post_init__(self) -> None:
        if self.total_documents < 0:
            raise ValueError("progress total_documents cannot be negative")
        if not 0 <= self.processed_documents <= self.total_documents:
            raise ValueError("progress processed_documents is outside the document total")

    @property
    def remaining_documents(self) -> int:
        return self.total_documents - self.processed_documents


PipelineProgressCallback = Callable[[PipelineProgress], None]


def _canonical_directory(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    absolute = Path(os.path.abspath(path))
    resolved = path.resolve(strict=True)
    if resolved != absolute:
        raise PipelineError(f"pipeline directories must not traverse symbolic links: {path}")
    return resolved


def run_paths(config: PipelineConfig) -> RunPaths:
    output_root = _canonical_directory(Path(config.output.root))
    runs_root = _canonical_directory(output_root / "runs")
    run_root = _canonical_directory(runs_root / config.run.run_id)
    scratch_root = _canonical_directory(run_root / "scratch")
    source_scratch = _canonical_directory(scratch_root / "sources")
    raster_scratch = _canonical_directory(scratch_root / "rasters")
    return RunPaths(
        output_root=output_root,
        run_root=run_root,
        inventory=run_root / "inventory.jsonl",
        provenance=run_root / "run-provenance.json",
        ledger=run_root / "state.sqlite3",
        source_scratch=source_scratch,
        raster_scratch=raster_scratch,
        page_images=run_root / "page-images",
    )


def _make_s3_client(config: PipelineConfig) -> Any | None:
    if isinstance(config.source, S3SourceConfig):
        return boto3.client("s3", region_name=config.source.region)
    return None


async def prepare_inventory(
    config: PipelineConfig,
    *,
    s3_client: Any | None = None,
) -> FrozenSourceInventory:
    """Create or verify the immutable inventory for the configured run."""

    paths = run_paths(config)
    digest_path = paths.inventory.with_name(paths.inventory.name + ".sha256")
    if paths.inventory.exists() or digest_path.exists():
        return await asyncio.to_thread(load_source_inventory, paths.inventory)
    objects = await asyncio.to_thread(
        partial(
            discover_sources,
            config.source,
            max_pdf_bytes=config.raster.max_pdf_bytes,
            s3_client=s3_client,
        )
    )
    return await asyncio.to_thread(freeze_source_inventory, paths.inventory, objects)


def _load_json_object(path: Path) -> dict[str, Any]:
    try:
        value: Any = json.loads(read_regular_file_bytes(path))
    except (ArtifactReadError, json.JSONDecodeError, UnicodeDecodeError) as error:
        raise PipelineError(f"cannot load immutable JSON artifact {path}") from error
    if not isinstance(value, dict):
        raise PipelineError(f"immutable JSON artifact must contain an object: {path}")
    return cast(dict[str, Any], value)


async def _prepare_provenance(
    *,
    project_root: Path,
    config: PipelineConfig,
    inventory: FrozenSourceInventory,
    paths: RunPaths,
) -> tuple[dict[str, Any], str, str]:
    current, fingerprint = await asyncio.to_thread(collect_runtime_provenance, project_root, config)
    payload = {
        **current,
        "pipeline_fingerprint": fingerprint,
        "inventory_sha256": inventory.sha256,
        "inventory_document_count": len(inventory.objects),
    }
    config_sha256 = cast(str, current["resolved_config_sha256"])
    if paths.provenance.exists():
        stored = await asyncio.to_thread(_load_json_object, paths.provenance)
        expected = {
            "pipeline_fingerprint": fingerprint,
            "resolved_config_sha256": config_sha256,
            "inventory_sha256": inventory.sha256,
            "inventory_document_count": len(inventory.objects),
        }
        if any(stored.get(key) != value for key, value in expected.items()):
            raise PipelineError(
                "run provenance conflicts with the current pipeline, config, or inventory"
            )
        return stored, fingerprint, config_sha256
    await asyncio.to_thread(atomic_publish_json, paths.provenance, payload)
    return payload, fingerprint, config_sha256


def _renderer_worker_initializer() -> None:
    atexit.register(close_process_document_cache)


def _new_process_pool(worker_count: int) -> ProcessPoolExecutor:
    return ProcessPoolExecutor(
        max_workers=worker_count,
        mp_context=multiprocessing.get_context("spawn"),
        initializer=_renderer_worker_initializer,
    )


def _page_provenance(
    *,
    source: SourceObject,
    config: PipelineConfig,
    config_sha256: str,
    pipeline_fingerprint: str,
    page_count: int,
    page_index: int,
) -> PageProvenance:
    if source.source_sha256 is None:
        raise PipelineError("materialized page source has no SHA-256")
    page_id = stable_id(
        "page",
        "document-page-v1",
        source.document_id,
        page_index,
    )
    extraction_id = stable_id(
        "extract",
        "glm-ocr-page-extraction-v1",
        page_id,
        pipeline_fingerprint,
    )
    return PageProvenance.model_validate(
        {
            **source.model_dump(mode="python"),
            "schema_version": 2,
            "run_id": config.run.run_id,
            "extraction_id": extraction_id,
            "page_id": page_id,
            "config_sha256": config_sha256,
            "pipeline_fingerprint": pipeline_fingerprint,
            "document_page_count": page_count,
            "page_index": page_index,
            "page_number": page_index + 1,
        },
        strict=True,
    )


def _failure_message(error: Exception) -> str:
    message = str(error).strip()
    return message[:2000] if message else type(error).__name__


def _document_failure(
    *,
    source: SourceObject,
    config: PipelineConfig,
    config_sha256: str,
    pipeline_fingerprint: str,
    stage: str,
    error: Exception,
) -> DocumentExtractionFailure:
    return DocumentExtractionFailure.model_validate(
        {
            **source.model_dump(mode="python"),
            "schema_version": 2,
            "run_id": config.run.run_id,
            "config_sha256": config_sha256,
            "pipeline_fingerprint": pipeline_fingerprint,
            "failure_stage": stage,
            "failed_at": datetime.now(UTC),
            "error_type": type(error).__name__,
            "error_message": _failure_message(error),
        },
        strict=True,
    )


def _page_failure(
    *,
    provenance: PageProvenance,
    stage: str,
    error: Exception,
    attempt_count: int,
    retryable: bool,
    last_request_id: str | None,
) -> PageExtractionFailure:
    return PageExtractionFailure.model_validate(
        {
            **provenance.model_dump(mode="python"),
            "failure_stage": stage,
            "failed_at": datetime.now(UTC),
            "inference_attempt_count": attempt_count,
            "retryable": retryable,
            "error_type": type(error).__name__,
            "error_message": _failure_message(error),
            "last_inference_request_id": last_request_id,
        },
        strict=True,
    )


def _raw_response_relative_path(provenance: PageProvenance, raw_response_sha256: str) -> Path:
    return (
        Path("raw-responses")
        / provenance.document_id
        / provenance.page_id
        / f"{raw_response_sha256}.json"
    )


def _raster_relative_path(provenance: PageProvenance, rendered: RenderedPage) -> Path:
    suffix = ".png" if rendered.raster.raster_image_format == "png" else ".jpg"
    return (
        Path("page-images")
        / provenance.document_id
        / provenance.page_id
        / f"{rendered.raster.raster_sha256}{suffix}"
    )


def _validate_raster_entry(
    path: Path,
    *,
    expected_size: int,
) -> None:
    required_flags = [name for name in ("O_CLOEXEC", "O_NOFOLLOW") if not hasattr(os, name)]
    if required_flags:
        raise RasterPublicationError(
            "safe retained-raster validation requires operating-system flags: "
            + ", ".join(required_flags)
        )
    try:
        descriptor = os.open(path, os.O_RDONLY | os.O_CLOEXEC | os.O_NOFOLLOW)
    except OSError as error:
        raise RasterPublicationError(f"retained raster cannot be opened safely: {path}") from error
    try:
        details = os.fstat(descriptor)
        if not stat.S_ISREG(details.st_mode):
            raise RasterPublicationError(f"retained raster is not a regular file: {path}")
    finally:
        os.close(descriptor)
    if details.st_size != expected_size:
        raise RasterPublicationError(f"retained raster size differs from renderer metadata: {path}")


def _publish_retained_raster(
    source: Path,
    destination: Path,
    *,
    expected_size: int,
) -> None:
    destination_parent = _canonical_directory(destination.parent)
    if destination.parent != destination_parent:
        raise RasterPublicationError("retained raster destination is not canonical")
    try:
        os.link(source, destination, follow_symlinks=False)
    except FileExistsError:
        pass
    except OSError as error:
        raise RasterPublicationError(f"cannot publish retained raster: {destination}") from error
    _validate_raster_entry(
        destination,
        expected_size=expected_size,
    )
    directory_fd = os.open(destination_parent, os.O_RDONLY)
    try:
        os.fsync(directory_fd)
    finally:
        os.close(directory_fd)


def _success_record(
    *,
    provenance: PageProvenance,
    rendered: RenderedPage,
    response: OcrResponse,
    server_info: ServerInfo,
    config: PipelineConfig,
    raster_path: Path | None,
    raw_response_path: Path,
    extraction_started_at: datetime,
    extraction_completed_at: datetime,
    queue_duration_ms: float,
    inference_duration_ms: float,
    persist_duration_ms: float,
    total_duration_ms: float,
) -> PageExtractionRecord:
    return PageExtractionRecord.model_validate(
        {
            **provenance.model_dump(mode="python"),
            **rendered.raster.model_dump(mode="python"),
            "inference_endpoint": config.vllm.endpoint,
            "inference_model": server_info.model_repository,
            "inference_served_model_name": server_info.served_model_name,
            "inference_model_revision": server_info.model_revision,
            "inference_server_engine": "vllm",
            "inference_server_engine_version": server_info.version,
            "inference_server_base_image": server_info.container_base_image,
            "inference_server_build_manifest_sha256": (server_info.container_build_manifest_sha256),
            "inference_server_contract_sha256": server_info.contract_sha256,
            "inference_speculative_method": server_info.speculative_method,
            "inference_num_speculative_tokens": server_info.num_speculative_tokens,
            "inference_prompt": config.vllm.prompt,
            "inference_prompt_sha256": sha256_bytes(config.vllm.prompt.encode("utf-8")),
            "inference_temperature": config.vllm.sampling.temperature,
            "inference_top_p": config.vllm.sampling.top_p,
            "inference_top_k": config.vllm.sampling.top_k,
            "inference_repetition_penalty": config.vllm.sampling.repetition_penalty,
            "inference_max_tokens": config.vllm.sampling.max_tokens,
            "inference_seed": config.vllm.sampling.seed,
            "inference_repetition_detection_min_pattern_size": (
                config.vllm.repetition_detection.min_pattern_size
            ),
            "inference_repetition_detection_max_pattern_size": (
                config.vllm.repetition_detection.max_pattern_size
            ),
            "inference_repetition_detection_min_count": (
                config.vllm.repetition_detection.min_count
            ),
            "inference_request_id": response.request_id,
            "inference_server_request_id": response.server_request_id,
            "inference_finish_reason": response.finish_reason,
            "inference_prompt_tokens": response.prompt_tokens,
            "inference_completion_tokens": response.completion_tokens,
            "inference_attempt_count": response.attempts[-1].attempt_number,
            "inference_duration_ms": inference_duration_ms,
            "extraction_started_at": extraction_started_at,
            "extraction_completed_at": extraction_completed_at,
            "queue_duration_ms": queue_duration_ms,
            "persist_duration_ms": persist_duration_ms,
            "total_duration_ms": total_duration_ms,
            "raw_ocr_text": response.text,
            "raw_ocr_text_sha256": sha256_bytes(response.text.encode("utf-8")),
            "raster_path": None if raster_path is None else raster_path.as_posix(),
            "raw_response_sha256": response.raw_response_sha256,
            "raw_response_path": raw_response_path.as_posix(),
        },
        strict=True,
    )


async def _process_page(
    *,
    provenance: PageProvenance,
    materialized: MaterializedSource,
    inspected_document: PdfInspection,
    config: PipelineConfig,
    paths: RunPaths,
    ledger: ExtractionLedger,
    client: _OcrClient,
    server_info: ServerInfo,
    inference_semaphore: asyncio.Semaphore,
    executor: Executor,
) -> None:
    extraction_started_at = datetime.now(UTC)
    extraction_start = time.perf_counter()
    suffix = ".png" if config.raster.image_format == "png" else ".jpg"
    raster_directory = paths.raster_scratch / provenance.document_id
    raster_directory.mkdir(parents=True, exist_ok=True)
    raster_path = raster_directory / f"page-{provenance.page_index:06d}{suffix}"
    loop = asyncio.get_running_loop()
    try:
        prior_attempt_count, prior_request_id = await ledger.attempt_state(
            config.run.run_id, provenance.extraction_id
        )
        try:
            rendered = await loop.run_in_executor(
                executor,
                render_page,
                str(materialized.path),
                provenance.page_index,
                str(raster_path),
                config.raster,
            )
            if rendered.document != inspected_document:
                raise RenderSourceChangedError(
                    "renderer source identity changed after document inspection"
                )
        except (PdfRendererError, OSError) as error:
            failure = _page_failure(
                provenance=provenance,
                stage="render",
                error=error,
                attempt_count=prior_attempt_count,
                retryable=False,
                last_request_id=prior_request_id,
            )
            await ledger.record_failure(failure=failure)
            if config.run.fail_fast:
                raise PipelineError(f"page render failed: {provenance.page_id}") from error
            return

        retained_raster_path: Path | None = None
        inference_raster_path = Path(rendered.output_path)
        if config.output.retain_page_images:
            retained_raster_path = _raster_relative_path(provenance, rendered)
            try:
                await asyncio.to_thread(
                    _publish_retained_raster,
                    Path(rendered.output_path),
                    paths.run_root / retained_raster_path,
                    expected_size=rendered.raster.raster_size_bytes,
                )
            except (RasterPublicationError, OSError) as error:
                failure = _page_failure(
                    provenance=provenance,
                    stage="raster_publish",
                    error=error,
                    attempt_count=prior_attempt_count,
                    retryable=False,
                    last_request_id=prior_request_id,
                )
                await ledger.record_failure(failure=failure)
                if config.run.fail_fast:
                    raise PipelineError(
                        f"retained raster publication failed: {provenance.page_id}"
                    ) from error
                return
            inference_raster_path = paths.run_root / retained_raster_path

        queue_started = time.perf_counter()
        async with inference_semaphore:
            queue_duration_ms = (time.perf_counter() - queue_started) * 1000.0
            first_attempt = await ledger.next_attempt_number(
                config.run.run_id, provenance.extraction_id
            )
            inference_started = time.perf_counter()
            try:
                response = await client.recognize_page(
                    inference_raster_path,
                    mime_type=rendered.raster.raster_mime_type,
                    raster_sha256=rendered.raster.raster_sha256,
                    request_id=provenance.extraction_id,
                    first_attempt_number=first_attempt,
                )
            except VllmRasterError as error:
                failure = _page_failure(
                    provenance=provenance,
                    stage="raster_validate",
                    error=error,
                    attempt_count=first_attempt - 1,
                    retryable=False,
                    last_request_id=prior_request_id,
                )
                await ledger.record_failure(failure=failure)
                if config.run.fail_fast:
                    raise PipelineError(
                        f"raster validation failed: {provenance.page_id}"
                    ) from error
                return
            except VllmClientError as error:
                inference_duration_ms = (time.perf_counter() - inference_started) * 1000.0
                failed_attempts = tuple(
                    attempt.to_inference_attempt(provenance) for attempt in error.attempts
                )
                last = error.attempts[-1]
                failure = _page_failure(
                    provenance=provenance,
                    stage="inference",
                    error=error,
                    attempt_count=last.attempt_number,
                    retryable=last.retryable,
                    last_request_id=last.inference_request_id,
                )
                await ledger.record_failure(failure=failure, attempts=failed_attempts)
                if config.run.fail_fast:
                    raise PipelineError(
                        f"page inference failed: {provenance.page_id} "
                        f"after {inference_duration_ms:.1f} ms"
                    ) from error
                return
            inference_duration_ms = (time.perf_counter() - inference_started) * 1000.0

        persist_started = time.perf_counter()
        raw_relative_path = _raw_response_relative_path(provenance, response.raw_response_sha256)
        successful_attempts = tuple(
            attempt.to_inference_attempt(provenance) for attempt in response.attempts
        )
        try:
            await asyncio.to_thread(
                atomic_publish_bytes,
                paths.run_root / raw_relative_path,
                response.raw_response,
            )
            extraction_completed_at = datetime.now(UTC)
            persist_duration_ms = (time.perf_counter() - persist_started) * 1000.0
            record = _success_record(
                provenance=provenance,
                rendered=rendered,
                response=response,
                server_info=server_info,
                config=config,
                raster_path=retained_raster_path,
                raw_response_path=raw_relative_path,
                extraction_started_at=extraction_started_at,
                extraction_completed_at=extraction_completed_at,
                queue_duration_ms=queue_duration_ms,
                inference_duration_ms=inference_duration_ms,
                persist_duration_ms=persist_duration_ms,
                total_duration_ms=(time.perf_counter() - extraction_start) * 1000.0,
            )
        except (AtomicConflictError, OSError, ValueError) as error:
            last = response.attempts[-1]
            failure = _page_failure(
                provenance=provenance,
                stage="persist",
                error=error,
                attempt_count=last.attempt_number,
                retryable=False,
                last_request_id=last.inference_request_id,
            )
            await ledger.record_failure(failure=failure, attempts=successful_attempts)
            if config.run.fail_fast:
                raise PipelineError(f"page persistence failed: {provenance.page_id}") from error
            return
        await ledger.record_success(result=record, attempts=successful_attempts)
    finally:
        await asyncio.to_thread(raster_path.unlink, missing_ok=True)
        with suppress(OSError):
            raster_directory.rmdir()


async def _process_document(
    *,
    inventory_source: SourceObject,
    config: PipelineConfig,
    config_sha256: str,
    pipeline_fingerprint: str,
    paths: RunPaths,
    ledger: ExtractionLedger,
    client: _OcrClient,
    server_info: ServerInfo,
    inference_semaphore: asyncio.Semaphore,
    executor: Executor,
    s3_client: Any | None,
) -> None:
    if await ledger.is_document_complete(config.run.run_id, inventory_source.document_id):
        return

    materialized: MaterializedSource | None = None
    try:
        try:
            materialized = await asyncio.to_thread(
                partial(
                    materialize_source,
                    inventory_source,
                    paths.source_scratch,
                    max_pdf_bytes=config.raster.max_pdf_bytes,
                    s3_client=s3_client,
                )
            )
        except (SourceError, OSError) as error:
            failure = _document_failure(
                source=inventory_source,
                config=config,
                config_sha256=config_sha256,
                pipeline_fingerprint=pipeline_fingerprint,
                stage="materialize",
                error=error,
            )
            await ledger.record_document_failure(failure=failure)
            if config.run.fail_fast:
                raise PipelineError(
                    f"source materialization failed: {inventory_source.document_id}"
                ) from error
            return

        loop = asyncio.get_running_loop()
        try:
            inspection = await loop.run_in_executor(
                executor, inspect_pdf, str(materialized.path), config.raster
            )
        except (PdfRendererError, OSError) as error:
            failure = _document_failure(
                source=inventory_source,
                config=config,
                config_sha256=config_sha256,
                pipeline_fingerprint=pipeline_fingerprint,
                stage="pdf_open",
                error=error,
            )
            await ledger.record_document_failure(failure=failure)
            if config.run.fail_fast:
                raise PipelineError(
                    f"PDF inspection failed: {inventory_source.document_id}"
                ) from error
            return

        source_sha256 = materialized.source.source_sha256
        if source_sha256 is None:
            raise PipelineError(
                f"materialized source has no SHA-256: {inventory_source.document_id}"
            )
        await ledger.register_document(
            run_id=config.run.run_id,
            inventory_source=inventory_source,
            source_sha256=source_sha256,
            page_count=inspection.page_count,
        )
        pages = tuple(
            _page_provenance(
                source=materialized.source,
                config=config,
                config_sha256=config_sha256,
                pipeline_fingerprint=pipeline_fingerprint,
                page_count=inspection.page_count,
                page_index=page_index,
            )
            for page_index in range(inspection.page_count)
        )
        await ledger.register_pages(
            run_id=config.run.run_id,
            document_id=inventory_source.document_id,
            pages=tuple((page.extraction_id, page.page_id, page.page_index) for page in pages),
        )
        succeeded = await ledger.successful_extraction_ids(
            config.run.run_id, tuple(page.extraction_id for page in pages)
        )
        remaining = tuple(page for page in pages if page.extraction_id not in succeeded)
        if not remaining:
            return

        page_queue: asyncio.Queue[PageProvenance] = asyncio.Queue()
        for page in remaining:
            page_queue.put_nowait(page)

        async def page_worker() -> None:
            while True:
                try:
                    page = page_queue.get_nowait()
                except asyncio.QueueEmpty:
                    return
                try:
                    await _process_page(
                        provenance=page,
                        materialized=materialized,
                        inspected_document=inspection,
                        config=config,
                        paths=paths,
                        ledger=ledger,
                        client=client,
                        server_info=server_info,
                        inference_semaphore=inference_semaphore,
                        executor=executor,
                    )
                finally:
                    page_queue.task_done()

        worker_count = min(config.concurrency.max_inflight_pages_per_document, len(remaining))
        async with asyncio.TaskGroup() as group:
            for _ in range(worker_count):
                group.create_task(page_worker())
    finally:
        if materialized is not None:
            await asyncio.to_thread(materialized.path.unlink, missing_ok=True)


async def _execute_documents(
    *,
    inventory: FrozenSourceInventory,
    config: PipelineConfig,
    config_sha256: str,
    pipeline_fingerprint: str,
    paths: RunPaths,
    ledger: ExtractionLedger,
    client: _OcrClient,
    executor: Executor,
    s3_client: Any | None,
    progress: PipelineProgressCallback | None,
) -> ServerInfo:
    total_documents = len(inventory.objects)
    if progress is not None:
        progress(
            PipelineProgress(
                phase="checking_server",
                total_documents=total_documents,
                processed_documents=0,
            )
        )
    server_info = await client.check_readiness()
    inference_semaphore = asyncio.Semaphore(config.concurrency.max_inflight_pages_global)
    document_queue: asyncio.Queue[SourceObject] = asyncio.Queue()
    for source in inventory.objects:
        document_queue.put_nowait(source)

    processed_documents = 0

    async def document_worker() -> None:
        nonlocal processed_documents
        while True:
            try:
                source = document_queue.get_nowait()
            except asyncio.QueueEmpty:
                return
            try:
                await _process_document(
                    inventory_source=source,
                    config=config,
                    config_sha256=config_sha256,
                    pipeline_fingerprint=pipeline_fingerprint,
                    paths=paths,
                    ledger=ledger,
                    client=client,
                    server_info=server_info,
                    inference_semaphore=inference_semaphore,
                    executor=executor,
                    s3_client=s3_client,
                )
                processed_documents += 1
                if progress is not None:
                    progress(
                        PipelineProgress(
                            phase="extracting",
                            total_documents=total_documents,
                            processed_documents=processed_documents,
                        )
                    )
            finally:
                document_queue.task_done()

    worker_count = min(config.concurrency.max_active_documents, len(inventory.objects))
    async with asyncio.TaskGroup() as group:
        for _ in range(worker_count):
            group.create_task(document_worker())
    return server_info


async def _publish_and_complete(
    *,
    ledger: ExtractionLedger,
    config: PipelineConfig,
    paths: RunPaths,
) -> tuple[PublishedDataset, dict[str, int]]:
    published = await publish_complete_dataset(
        ledger=ledger,
        run_id=config.run.run_id,
        run_root=paths.run_root,
        batch_rows=config.output.write_batch_rows,
        compression=config.output.parquet_compression,
        retain_page_images=config.output.retain_page_images,
    )
    await ledger.mark_complete(
        run_id=config.run.run_id,
        manifest_sha256=published.manifest_sha256,
        manifest=published.manifest,
    )
    return published, await ledger.require_complete(config.run.run_id)


async def run_pipeline(
    *,
    project_root: Path,
    config: PipelineConfig,
    ocr_client: _OcrClient | None = None,
    executor: Executor | None = None,
    s3_client: Any | None = None,
    progress: PipelineProgressCallback | None = None,
) -> PipelineResult:
    """Run or resume extraction; a dataset manifest appears only on full success."""

    paths = run_paths(config)
    owned_s3_client = s3_client
    if owned_s3_client is None and isinstance(config.source, S3SourceConfig):
        owned_s3_client = await asyncio.to_thread(_make_s3_client, config)
    inventory = await prepare_inventory(config, s3_client=owned_s3_client)
    provenance, pipeline_fingerprint, config_sha256 = await _prepare_provenance(
        project_root=project_root,
        config=config,
        inventory=inventory,
        paths=paths,
    )

    async with ExtractionLedger(paths.ledger) as ledger:
        prior_status = await ledger.initialize_run(
            run_id=config.run.run_id,
            config_sha256=config_sha256,
            pipeline_fingerprint=pipeline_fingerprint,
            config=config.model_dump(mode="json"),
            inventory_sha256=inventory.sha256,
            inventory_document_count=len(inventory.objects),
            provenance=provenance,
            resume=config.run.resume,
        )
        if prior_status == "complete":
            published, summary = await _publish_and_complete(
                ledger=ledger, config=config, paths=paths
            )
            return PipelineResult(
                run_id=config.run.run_id,
                resumed_complete_run=True,
                summary=summary,
                dataset_manifest=published.manifest_path,
                server_info=None,
            )

        owned_executor = executor is None
        active_executor = executor or _new_process_pool(config.concurrency.renderer_processes)
        try:
            if ocr_client is None:
                async with VllmOcrClient(
                    config.vllm,
                    max_connections=config.concurrency.max_inflight_pages_global,
                ) as client:
                    server_info = await _execute_documents(
                        inventory=inventory,
                        config=config,
                        config_sha256=config_sha256,
                        pipeline_fingerprint=pipeline_fingerprint,
                        paths=paths,
                        ledger=ledger,
                        client=client,
                        executor=active_executor,
                        s3_client=owned_s3_client,
                        progress=progress,
                    )
            else:
                server_info = await _execute_documents(
                    inventory=inventory,
                    config=config,
                    config_sha256=config_sha256,
                    pipeline_fingerprint=pipeline_fingerprint,
                    paths=paths,
                    ledger=ledger,
                    client=ocr_client,
                    executor=active_executor,
                    s3_client=owned_s3_client,
                    progress=progress,
                )
            if progress is not None:
                progress(
                    PipelineProgress(
                        phase="publishing",
                        total_documents=len(inventory.objects),
                        processed_documents=len(inventory.objects),
                    )
                )
            published, summary = await _publish_and_complete(
                ledger=ledger, config=config, paths=paths
            )
        except BaseException:
            await ledger.mark_failed(config.run.run_id)
            raise
        finally:
            if owned_executor:
                await asyncio.to_thread(
                    cast(ProcessPoolExecutor, active_executor).shutdown,
                    wait=True,
                    cancel_futures=True,
                )
        return PipelineResult(
            run_id=config.run.run_id,
            resumed_complete_run=False,
            summary=summary,
            dataset_manifest=published.manifest_path,
            server_info=server_info,
        )


async def require_run_complete(config: PipelineConfig) -> dict[str, int]:
    """Read-only status probe used by the CLI."""

    ledger_path = Path(config.output.root) / "runs" / config.run.run_id / "state.sqlite3"
    async with ExtractionLedger(ledger_path, read_only=True) as ledger:
        return await ledger.require_complete(config.run.run_id)
