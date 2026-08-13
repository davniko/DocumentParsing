"""Reproducible concurrency benchmarking for a running vLLM OCR server.

The benchmark deliberately accepts already-rendered pages.  This isolates the
HTTP/model path from PDF rendering while retaining exact, content-addressed
raster provenance.  No server is started by this module: callers must supply a
client whose readiness and ``recognize_page`` contracts match
``VllmOcrClient``.
"""

from __future__ import annotations

import asyncio
import heapq
import importlib.metadata
import json
import math
import os
import platform
import stat
import sys
import time
from collections import deque
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
from typing import Annotated, Any, Literal, Protocol

from PIL import Image, UnidentifiedImageError
from pydantic import BaseModel, ConfigDict, StringConstraints, field_validator

from document_ocr.atomic import atomic_publish_bytes
from document_ocr.client import OcrResponse, ServerInfo
from document_ocr.config import (
    BenchmarkConcurrencyPoint,
    BenchmarkSweepConfig,
    PipelineConfig,
)
from document_ocr.hashing import (
    canonical_json_bytes,
    canonical_json_sha256,
    sha256_bytes,
    sha256_file,
)
from document_ocr.vllm_contract import runtime_contract_payload, runtime_contract_sha256

NonEmptyString = Annotated[str, StringConstraints(min_length=1)]
_SHA256_PATTERN = "0123456789abcdef"
_HARNESS_SOURCE_FILES = (
    "atomic.py",
    "benchmark.py",
    "client.py",
    "config.py",
    "hashing.py",
    "vllm_contract.py",
)
_HARNESS_DISTRIBUTIONS = (
    "document-ocr-pipeline",
    "httpcore",
    "httpx",
    "pillow",
    "pydantic",
)


class BenchmarkError(RuntimeError):
    """Base class for benchmark contract and execution failures."""


class RasterManifestError(BenchmarkError):
    """The benchmark raster manifest or one of its files is invalid."""


class BenchmarkExecutionError(BenchmarkError):
    """A benchmark request failed or returned an inconsistent result."""


class OcrBenchmarkClient(Protocol):
    """The subset of ``VllmOcrClient`` required by this harness."""

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


class RasterManifestEntry(BaseModel):
    """One immutable raster used by the concurrency sweep."""

    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)

    page_id: NonEmptyString
    document_id: NonEmptyString
    raster_path: NonEmptyString
    mime_type: Literal["image/png", "image/jpeg"]
    raster_sha256: str

    @field_validator("page_id", "document_id")
    @classmethod
    def identity_must_not_be_padded(cls, value: str) -> str:
        if value != value.strip():
            raise ValueError("identity fields must not have leading or trailing whitespace")
        if any(ord(character) < 0x20 for character in value):
            raise ValueError("identity fields must not contain control characters")
        return value

    @field_validator("raster_path")
    @classmethod
    def raster_path_must_be_absolute(cls, value: str) -> str:
        if value != value.strip():
            raise ValueError("raster_path must not have leading or trailing whitespace")
        if not Path(value).is_absolute():
            raise ValueError("raster_path must be absolute")
        return value

    @field_validator("raster_sha256")
    @classmethod
    def validate_raster_sha256(cls, value: str) -> str:
        if len(value) != 64 or any(character not in _SHA256_PATTERN for character in value):
            raise ValueError("raster_sha256 must be a lowercase 64-character SHA-256")
        return value


@dataclass(frozen=True, slots=True)
class RasterFileIdentity:
    """Observed filesystem and image identity for one validated raster."""

    canonical_path: Path
    device: int
    inode: int
    size_bytes: int
    mtime_ns: int
    width_pixels: int
    height_pixels: int


@dataclass(frozen=True, slots=True)
class LoadedRasterManifest:
    """A parsed JSONL manifest bound to its exact bytes and raster files."""

    source_path: Path
    source_file_sha256: str
    canonical_sha256: str
    entries: tuple[RasterManifestEntry, ...]
    identities: Mapping[str, RasterFileIdentity]


@dataclass(frozen=True, slots=True)
class PublishedBenchmarkReport:
    """Identity of a completely published benchmark report."""

    path: Path
    sha256: str
    created: bool
    report: Mapping[str, Any]


@dataclass(frozen=True, slots=True)
class _RequestSample:
    sequence_index: int
    manifest_index: int
    page_id: str
    document_id: str
    client_request_id: str
    inference_request_id: str
    inference_server_request_id: str | None
    request_started: float
    request_completed: float
    request_latency_ms: float
    identity_validation_before_ms: float
    identity_validation_after_ms: float
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int
    attempt_count: int
    text_sha256: str
    raw_response_sha256: str

    def report_row(self) -> dict[str, Any]:
        return {
            "sequence_index": self.sequence_index,
            "manifest_index": self.manifest_index,
            "page_id": self.page_id,
            "document_id": self.document_id,
            "client_request_id": self.client_request_id,
            "inference_request_id": self.inference_request_id,
            "inference_server_request_id": self.inference_server_request_id,
            "request_latency_ms": self.request_latency_ms,
            "identity_validation_before_ms": self.identity_validation_before_ms,
            "identity_validation_after_ms": self.identity_validation_after_ms,
            "prompt_tokens": self.prompt_tokens,
            "completion_tokens": self.completion_tokens,
            "total_tokens": self.total_tokens,
            "attempt_count": self.attempt_count,
            "text_sha256": self.text_sha256,
            "raw_response_sha256": self.raw_response_sha256,
        }


def _reject_duplicate_json_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON key {key!r}")
        result[key] = value
    return result


def _reject_nonfinite_json(value: str) -> None:
    raise ValueError(f"non-standard JSON number {value!r}")


def _harness_identity() -> dict[str, Any]:
    package_root = Path(__file__).resolve().parent
    source_files: dict[str, str] = {}
    for name in _HARNESS_SOURCE_FILES:
        path = package_root / name
        if not path.is_file() or path.is_symlink():
            raise BenchmarkExecutionError(f"benchmark harness source is unavailable: {name}")
        source_files[name] = sha256_file(path)

    distributions: dict[str, str] = {}
    for name in _HARNESS_DISTRIBUTIONS:
        try:
            distributions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError as error:
            raise BenchmarkExecutionError(
                f"benchmark harness distribution is not installed: {name}"
            ) from error
    return {
        "schema_version": 1,
        "python_implementation": platform.python_implementation(),
        "python_version": sys.version,
        "distributions": distributions,
        "source_files_sha256": source_files,
    }


def _file_signature(path: Path) -> tuple[int, int, int, int]:
    try:
        details = path.stat(follow_symlinks=False)
    except OSError as error:
        raise RasterManifestError(f"cannot stat raster file: {path}") from error
    if not stat.S_ISREG(details.st_mode):
        raise RasterManifestError(f"raster path is not a regular file: {path}")
    if details.st_size <= 0:
        raise RasterManifestError(f"raster file is empty: {path}")
    return (details.st_dev, details.st_ino, details.st_size, details.st_mtime_ns)


def _canonical_regular_file(path: Path, *, description: str) -> Path:
    absolute = Path(os.path.abspath(path))
    try:
        resolved = path.resolve(strict=True)
    except OSError as error:
        raise RasterManifestError(f"{description} does not exist: {path}") from error
    if resolved != absolute:
        raise RasterManifestError(f"{description} must not traverse symbolic links: {path}")
    try:
        details = resolved.stat(follow_symlinks=False)
    except OSError as error:
        raise RasterManifestError(f"cannot stat {description}: {path}") from error
    if not stat.S_ISREG(details.st_mode):
        raise RasterManifestError(f"{description} is not a regular file: {path}")
    return resolved


def _inspect_raster(entry: RasterManifestEntry) -> RasterFileIdentity:
    path = _canonical_regular_file(Path(entry.raster_path), description="raster path")
    signature_before = _file_signature(path)
    try:
        actual_sha256 = sha256_file(path)
        with Image.open(path) as image:
            actual_format = image.format
            width, height = image.size
            image.verify()
    except (
        Image.DecompressionBombError,
        OSError,
        SyntaxError,
        UnidentifiedImageError,
        ValueError,
    ) as error:
        raise RasterManifestError(f"raster is not a valid image: {path}") from error
    signature_after = _file_signature(path)
    if signature_after != signature_before:
        raise RasterManifestError(f"raster changed while it was being validated: {path}")
    if actual_sha256 != entry.raster_sha256:
        raise RasterManifestError(f"raster SHA-256 does not match its manifest: {path}")
    expected_format = {"image/png": "PNG", "image/jpeg": "JPEG"}[entry.mime_type]
    if actual_format != expected_format:
        raise RasterManifestError(f"raster MIME type does not match its image encoding: {path}")
    if width <= 0 or height <= 0:
        raise RasterManifestError(f"raster has invalid pixel dimensions: {path}")
    return RasterFileIdentity(
        canonical_path=path,
        device=signature_after[0],
        inode=signature_after[1],
        size_bytes=signature_after[2],
        mtime_ns=signature_after[3],
        width_pixels=width,
        height_pixels=height,
    )


def _validate_current_raster(entry: RasterManifestEntry, expected: RasterFileIdentity) -> None:
    """Revalidate bytes and filesystem identity immediately around every request."""

    current_path = _canonical_regular_file(Path(entry.raster_path), description="raster path")
    if current_path != expected.canonical_path:
        raise RasterManifestError(f"raster canonical path changed: {entry.raster_path}")
    signature_before = _file_signature(current_path)
    expected_signature = (
        expected.device,
        expected.inode,
        expected.size_bytes,
        expected.mtime_ns,
    )
    if signature_before != expected_signature:
        raise RasterManifestError(f"raster filesystem identity changed: {entry.raster_path}")
    current_sha256 = sha256_file(current_path)
    signature_after = _file_signature(current_path)
    if signature_after != signature_before:
        raise RasterManifestError(
            f"raster changed while its identity was being checked: {entry.raster_path}"
        )
    if current_sha256 != entry.raster_sha256:
        raise RasterManifestError(f"raster SHA-256 changed: {entry.raster_path}")


def load_raster_manifest(path: str | Path) -> LoadedRasterManifest:
    """Parse strict JSONL and validate every referenced raster before benchmarking."""

    source_path = _canonical_regular_file(Path(path), description="raster manifest")
    signature_before = _file_signature(source_path)
    source_file_sha256 = sha256_file(source_path)
    entries: list[RasterManifestEntry] = []
    try:
        with source_path.open("r", encoding="utf-8", newline="") as stream:
            for line_number, line in enumerate(stream, start=1):
                if not line.strip():
                    raise RasterManifestError(
                        f"raster manifest contains a blank line at line {line_number}"
                    )
                try:
                    raw = json.loads(
                        line,
                        object_pairs_hook=_reject_duplicate_json_keys,
                        parse_constant=_reject_nonfinite_json,
                    )
                    entry = RasterManifestEntry.model_validate(raw, strict=True)
                except (TypeError, ValueError) as error:
                    raise RasterManifestError(
                        f"invalid raster manifest entry at line {line_number}"
                    ) from error
                entries.append(entry)
    except UnicodeError as error:
        raise RasterManifestError("raster manifest is not valid UTF-8") from error
    if not entries:
        raise RasterManifestError("raster manifest must contain at least one page")

    source_file_sha256_after = sha256_file(source_path)
    signature_after = _file_signature(source_path)
    if source_file_sha256_after != source_file_sha256 or signature_after != signature_before:
        raise RasterManifestError("raster manifest changed while it was being parsed")

    page_ids: set[str] = set()
    raster_paths: set[Path] = set()
    identities: dict[str, RasterFileIdentity] = {}
    for entry in entries:
        if entry.page_id in page_ids:
            raise RasterManifestError(f"duplicate page_id in raster manifest: {entry.page_id}")
        identity = _inspect_raster(entry)
        if identity.canonical_path in raster_paths:
            raise RasterManifestError(
                f"duplicate canonical raster_path in raster manifest: {entry.raster_path}"
            )
        page_ids.add(entry.page_id)
        raster_paths.add(identity.canonical_path)
        identities[entry.page_id] = identity

    canonical_manifest = [entry.model_dump(mode="json") for entry in entries]
    return LoadedRasterManifest(
        source_path=source_path,
        source_file_sha256=source_file_sha256,
        canonical_sha256=canonical_json_sha256(canonical_manifest),
        entries=tuple(entries),
        identities=MappingProxyType(identities),
    )


def _validate_server_info(server: ServerInfo, config: PipelineConfig) -> dict[str, Any]:
    runtime_payload = runtime_contract_payload(
        model=config.vllm.model,
        served_model_name=config.vllm.served_model_name,
        model_revision=config.vllm.revision,
        max_model_len=config.vllm.max_model_len,
        max_num_seqs=config.vllm.max_num_seqs,
        gpu_memory_utilization=config.vllm.gpu_memory_utilization,
        generation_config="vllm",
        speculative_method=config.vllm.speculative_decoding.method,
        num_speculative_tokens=config.vllm.speculative_decoding.num_speculative_tokens,
        image_limit_per_prompt=1,
        container_base_image=config.vllm.container_base_image,
        container_build_manifest_sha256=config.vllm.container_build_manifest_sha256,
    )
    expected: dict[str, object] = {
        "version": config.vllm.engine_version.removeprefix("v"),
        "model_repository": config.vllm.model,
        "served_model_name": config.vllm.served_model_name,
        "model_revision": config.vllm.revision,
        "max_model_len": config.vllm.max_model_len,
        "max_num_seqs": config.vllm.max_num_seqs,
        "gpu_memory_utilization": config.vllm.gpu_memory_utilization,
        "generation_config": "vllm",
        "speculative_method": config.vllm.speculative_decoding.method,
        "num_speculative_tokens": (config.vllm.speculative_decoding.num_speculative_tokens),
        "image_limit_per_prompt": 1,
        "container_base_image": config.vllm.container_base_image,
        "container_build_manifest_sha256": config.vllm.container_build_manifest_sha256,
        "contract_sha256": runtime_contract_sha256(runtime_payload),
    }
    observed: dict[str, object] = {
        "version": server.version.removeprefix("v"),
        "model_repository": server.model_repository,
        "served_model_name": server.served_model_name,
        "model_revision": server.model_revision,
        "max_model_len": server.max_model_len,
        "max_num_seqs": server.max_num_seqs,
        "gpu_memory_utilization": server.gpu_memory_utilization,
        "generation_config": server.generation_config,
        "speculative_method": server.speculative_method,
        "num_speculative_tokens": server.num_speculative_tokens,
        "image_limit_per_prompt": server.image_limit_per_prompt,
        "container_base_image": server.container_base_image,
        "container_build_manifest_sha256": server.container_build_manifest_sha256,
        "contract_sha256": server.contract_sha256,
    }
    mismatches = sorted(key for key in expected if observed[key] != expected[key])
    if mismatches:
        raise BenchmarkExecutionError(
            "benchmark server contract mismatch for: " + ", ".join(mismatches)
        )
    if server.models.count(config.vllm.served_model_name) != 1:
        raise BenchmarkExecutionError(
            "configured served model must appear exactly once in benchmark server identity"
        )
    return {
        "version": server.version,
        "models": sorted(server.models),
        "model_repository": server.model_repository,
        "served_model_name": server.served_model_name,
        "model_revision": server.model_revision,
        "max_model_len": server.max_model_len,
        "max_num_seqs": server.max_num_seqs,
        "gpu_memory_utilization": server.gpu_memory_utilization,
        "generation_config": server.generation_config,
        "speculative_method": server.speculative_method,
        "num_speculative_tokens": server.num_speculative_tokens,
        "image_limit_per_prompt": server.image_limit_per_prompt,
        "container_base_image": server.container_base_image,
        "container_build_manifest_sha256": server.container_build_manifest_sha256,
        "contract_sha256": server.contract_sha256,
    }


def _validate_response(response: OcrResponse, request_id: str, config: PipelineConfig) -> None:
    if not isinstance(response.text, str):
        raise BenchmarkExecutionError("OCR response text must be a string")
    if response.finish_reason != "stop":
        raise BenchmarkExecutionError("OCR response finish reason must be 'stop'")
    if not isinstance(response.request_id, str) or not response.request_id:
        raise BenchmarkExecutionError("OCR response request identity must be non-empty")
    if not isinstance(response.response_id, str) or not response.response_id:
        raise BenchmarkExecutionError("OCR response identity must be non-empty")
    if response.response_model != config.vllm.served_model_name:
        raise BenchmarkExecutionError("OCR response model does not match the configured model")
    if response.total_tokens != response.prompt_tokens + response.completion_tokens:
        raise BenchmarkExecutionError("OCR response token usage is inconsistent")
    if any(
        isinstance(value, bool) or not isinstance(value, int) or value < 0
        for value in (
            response.prompt_tokens,
            response.completion_tokens,
            response.total_tokens,
        )
    ):
        raise BenchmarkExecutionError("OCR response token usage must be non-negative integers")
    if not isinstance(response.raw_response, bytes):
        raise BenchmarkExecutionError("OCR raw response must contain exact bytes")
    if sha256_bytes(response.raw_response) != response.raw_response_sha256:
        raise BenchmarkExecutionError("OCR raw response SHA-256 is inconsistent")
    if not response.attempts:
        raise BenchmarkExecutionError("OCR response must contain at least one request attempt")
    attempt_numbers = [attempt.attempt_number for attempt in response.attempts]
    if any(isinstance(number, bool) or not isinstance(number, int) for number in attempt_numbers):
        raise BenchmarkExecutionError("OCR response attempt numbers must be integers")
    expected_numbers = list(range(1, len(response.attempts) + 1))
    if attempt_numbers != expected_numbers:
        raise BenchmarkExecutionError("OCR response attempt numbers are not contiguous")
    for attempt in response.attempts[:-1]:
        if attempt.outcome == "success" or not attempt.retryable:
            raise BenchmarkExecutionError("non-final OCR attempts must be retryable failures")
    final_attempt = response.attempts[-1]
    if final_attempt.outcome != "success" or final_attempt.retryable:
        raise BenchmarkExecutionError("final OCR response attempt must be successful")
    if final_attempt.inference_request_id != response.request_id:
        raise BenchmarkExecutionError("final attempt request identity is inconsistent")
    expected_final_request_id = f"{request_id}-a{len(response.attempts)}"
    if response.request_id != expected_final_request_id:
        raise BenchmarkExecutionError("OCR response request identity is inconsistent")


def _percentile(values: Sequence[float], quantile: float) -> float:
    """Return the R-7 linearly interpolated sample percentile."""

    if not values:
        raise ValueError("percentile requires at least one value")
    ordered = sorted(values)
    position = (len(ordered) - 1) * quantile
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    fraction = position - lower
    return ordered[lower] + (ordered[upper] - ordered[lower]) * fraction


def _rounded(value: float, digits: int = 6) -> float:
    if not math.isfinite(value):
        raise BenchmarkExecutionError("benchmark produced a non-finite metric")
    return round(value, digits)


def _phase_report(
    samples: Sequence[_RequestSample], wall_time_seconds: float, *, include_samples: bool
) -> dict[str, Any]:
    if not samples:
        return {
            "pages": 0,
            "wall_time_seconds": _rounded(wall_time_seconds, 9),
            "prompt_tokens": 0,
            "completion_tokens": 0,
            "total_tokens": 0,
            "attempts": 0,
            "retries": 0,
        }
    latencies = [sample.request_latency_ms for sample in samples]
    identity_times = [
        sample.identity_validation_before_ms + sample.identity_validation_after_ms
        for sample in samples
    ]
    prompt_tokens = sum(sample.prompt_tokens for sample in samples)
    completion_tokens = sum(sample.completion_tokens for sample in samples)
    total_tokens = sum(sample.total_tokens for sample in samples)
    attempt_count = sum(sample.attempt_count for sample in samples)
    request_window_seconds = max(sample.request_completed for sample in samples) - min(
        sample.request_started for sample in samples
    )
    if wall_time_seconds <= 0 or request_window_seconds <= 0:
        raise BenchmarkExecutionError("benchmark timing window must be positive")
    report: dict[str, Any] = {
        "pages": len(samples),
        "wall_time_seconds": _rounded(wall_time_seconds, 9),
        "request_window_seconds": _rounded(request_window_seconds, 9),
        "pages_per_second": _rounded(len(samples) / wall_time_seconds),
        "request_window_pages_per_second": _rounded(len(samples) / request_window_seconds),
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
        "total_tokens": total_tokens,
        "completion_tokens_per_second": _rounded(completion_tokens / wall_time_seconds),
        "attempts": attempt_count,
        "retries": attempt_count - len(samples),
        "request_latency_ms": {
            "minimum": _rounded(min(latencies)),
            "mean": _rounded(sum(latencies) / len(latencies)),
            "p50": _rounded(_percentile(latencies, 0.50)),
            "p95": _rounded(_percentile(latencies, 0.95)),
            "p99": _rounded(_percentile(latencies, 0.99)),
            "maximum": _rounded(max(latencies)),
        },
        "identity_validation_ms": {
            "mean": _rounded(sum(identity_times) / len(identity_times)),
            "maximum": _rounded(max(identity_times)),
        },
    }
    if include_samples:
        report["requests"] = [sample.report_row() for sample in samples]
    return report


async def _execute_request(
    *,
    client: OcrBenchmarkClient,
    config: PipelineConfig,
    entry: RasterManifestEntry,
    identity: RasterFileIdentity,
    sequence_index: int,
    manifest_index: int,
    request_id: str,
    consistency_hashes: dict[str, str],
    consistency_lock: asyncio.Lock,
) -> _RequestSample:
    validation_before_started = time.perf_counter()
    try:
        await asyncio.to_thread(_validate_current_raster, entry, identity)
    except RasterManifestError as error:
        raise BenchmarkExecutionError(
            f"raster identity validation failed before request for page {entry.page_id}"
        ) from error
    validation_before_ms = (time.perf_counter() - validation_before_started) * 1000.0

    request_started = time.perf_counter()
    try:
        response = await client.recognize_page(
            identity.canonical_path,
            mime_type=entry.mime_type,
            raster_sha256=entry.raster_sha256,
            request_id=request_id,
            first_attempt_number=1,
        )
    except Exception as error:
        try:
            await asyncio.to_thread(_validate_current_raster, entry, identity)
        except RasterManifestError as raster_error:
            raise BenchmarkExecutionError(
                f"raster identity validation failed after request for page {entry.page_id}"
            ) from raster_error
        raise BenchmarkExecutionError(
            f"OCR benchmark request failed for page {entry.page_id} ({type(error).__name__})"
        ) from error
    request_completed = time.perf_counter()

    validation_after_started = time.perf_counter()
    try:
        await asyncio.to_thread(_validate_current_raster, entry, identity)
    except RasterManifestError as error:
        raise BenchmarkExecutionError(
            f"raster identity validation failed after request for page {entry.page_id}"
        ) from error
    validation_after_ms = (time.perf_counter() - validation_after_started) * 1000.0

    _validate_response(response, request_id, config)
    try:
        text_bytes = response.text.encode("utf-8")
    except UnicodeEncodeError as error:
        raise BenchmarkExecutionError("OCR response text is not valid UTF-8") from error
    text_sha256 = sha256_bytes(text_bytes)
    async with consistency_lock:
        baseline = consistency_hashes.get(entry.page_id)
        if baseline is None:
            consistency_hashes[entry.page_id] = text_sha256
        elif baseline != text_sha256:
            raise BenchmarkExecutionError(
                f"OCR text hash changed across benchmark requests for page {entry.page_id}"
            )

    return _RequestSample(
        sequence_index=sequence_index,
        manifest_index=manifest_index,
        page_id=entry.page_id,
        document_id=entry.document_id,
        client_request_id=request_id,
        inference_request_id=response.request_id,
        inference_server_request_id=response.server_request_id,
        request_started=request_started,
        request_completed=request_completed,
        request_latency_ms=_rounded((request_completed - request_started) * 1000.0),
        identity_validation_before_ms=_rounded(validation_before_ms),
        identity_validation_after_ms=_rounded(validation_after_ms),
        prompt_tokens=response.prompt_tokens,
        completion_tokens=response.completion_tokens,
        total_tokens=response.total_tokens,
        attempt_count=len(response.attempts),
        text_sha256=text_sha256,
        raw_response_sha256=response.raw_response_sha256,
    )


async def _run_phase(
    *,
    client: OcrBenchmarkClient,
    config: PipelineConfig,
    manifest: LoadedRasterManifest,
    point: BenchmarkConcurrencyPoint,
    count: int,
    workload_offset: int,
    request_prefix: str,
    consistency_hashes: dict[str, str],
    consistency_lock: asyncio.Lock,
) -> tuple[list[_RequestSample], float]:
    if count == 0:
        return [], 0.0

    sequences_by_document: dict[str, deque[int]] = {}
    for sequence_index in range(count):
        manifest_index = (workload_offset + sequence_index) % len(manifest.entries)
        document_id = manifest.entries[manifest_index].document_id
        sequences_by_document.setdefault(document_id, deque()).append(sequence_index)

    active_by_document = dict.fromkeys(sequences_by_document, 0)
    eligible_documents: set[str] = set()
    eligible_heap: list[tuple[int, str]] = []

    def make_document_eligible(document_id: str) -> None:
        queued_sequences = sequences_by_document[document_id]
        if (
            document_id in eligible_documents
            or not queued_sequences
            or active_by_document[document_id] >= point.max_inflight_pages_per_document
        ):
            return
        heapq.heappush(eligible_heap, (queued_sequences[0], document_id))
        eligible_documents.add(document_id)

    for document_id in sequences_by_document:
        make_document_eligible(document_id)

    samples: list[_RequestSample] = []
    active_tasks: dict[asyncio.Task[_RequestSample], tuple[str, int]] = {}

    phase_started = time.perf_counter()
    try:
        while eligible_heap or active_tasks:
            while eligible_heap and len(active_tasks) < point.max_inflight_pages_global:
                sequence_index, document_id = heapq.heappop(eligible_heap)
                eligible_documents.remove(document_id)
                queued_sequence = sequences_by_document[document_id].popleft()
                if queued_sequence != sequence_index:
                    raise BenchmarkExecutionError(
                        "benchmark scheduler sequence identity is inconsistent"
                    )

                active_by_document[document_id] += 1
                make_document_eligible(document_id)

                manifest_index = (workload_offset + sequence_index) % len(manifest.entries)
                entry = manifest.entries[manifest_index]
                request_id = f"{request_prefix}-q{sequence_index:08d}"
                task = asyncio.create_task(
                    _execute_request(
                        client=client,
                        config=config,
                        entry=entry,
                        identity=manifest.identities[entry.page_id],
                        sequence_index=sequence_index,
                        manifest_index=manifest_index,
                        request_id=request_id,
                        consistency_hashes=consistency_hashes,
                        consistency_lock=consistency_lock,
                    )
                )
                active_tasks[task] = (document_id, sequence_index)

            if not active_tasks:
                raise BenchmarkExecutionError(
                    "benchmark scheduler has queued work but no eligible request"
                )

            completed, _ = await asyncio.wait(active_tasks, return_when=asyncio.FIRST_COMPLETED)
            for task in sorted(completed, key=lambda candidate: active_tasks[candidate][1]):
                document_id, _sequence_index = active_tasks.pop(task)
                active_by_document[document_id] -= 1
                make_document_eligible(document_id)
                samples.append(task.result())
    except BaseException:
        for task in active_tasks:
            task.cancel()
        await asyncio.gather(*active_tasks, return_exceptions=True)
        raise
    wall_time_seconds = time.perf_counter() - phase_started
    samples.sort(key=lambda sample: sample.sequence_index)
    if len(samples) != count:
        raise BenchmarkExecutionError("benchmark phase completed without every requested page")
    return samples, wall_time_seconds


async def run_benchmark(
    *,
    config: PipelineConfig,
    raster_manifest_path: str | Path,
    report_path: str | Path,
    client: OcrBenchmarkClient,
) -> PublishedBenchmarkReport:
    """Run the configured sweep and atomically publish a complete JSON report.

    The function never constructs or launches a model server.  It validates all
    raster bytes first, checks the supplied client's server identity, aborts the
    entire sweep on any request/output inconsistency, and publishes only after
    every repetition succeeds.
    """

    sweep: BenchmarkSweepConfig | None = config.benchmark
    if sweep is None:
        raise BenchmarkError("pipeline config has no benchmark sweep")
    manifest = await asyncio.to_thread(load_raster_manifest, raster_manifest_path)
    harness_identity = await asyncio.to_thread(_harness_identity)
    harness_identity_sha256 = canonical_json_sha256(harness_identity)
    try:
        server = await client.check_readiness()
    except Exception as error:
        raise BenchmarkExecutionError(
            f"benchmark server readiness failed ({type(error).__name__})"
        ) from error
    observed_server = _validate_server_info(server, config)

    resolved_config = config.model_dump(mode="json")
    benchmark_config = sweep.model_dump(mode="json")
    resolved_config_sha256 = canonical_json_sha256(resolved_config)
    benchmark_config_sha256 = canonical_json_sha256(benchmark_config)
    configured_server = config.vllm.model_dump(mode="json")
    server_identity = {
        "configured": configured_server,
        "observed": observed_server,
    }
    benchmark_identity = canonical_json_sha256(
        {
            "schema": "document-ocr-vllm-benchmark-v1",
            "resolved_config_sha256": resolved_config_sha256,
            "benchmark_config_sha256": benchmark_config_sha256,
            "raster_manifest_sha256": manifest.canonical_sha256,
            "raster_manifest_file_sha256": manifest.source_file_sha256,
            "server_identity_sha256": canonical_json_sha256(server_identity),
            "harness_identity_sha256": harness_identity_sha256,
        }
    )

    consistency_hashes: dict[str, str] = {}
    consistency_lock = asyncio.Lock()
    repetitions: list[dict[str, Any]] = []
    for point_index, point in enumerate(sweep.points):
        for repetition_index in range(sweep.repetitions):
            common_prefix = (
                f"bench-{benchmark_identity[:16]}-p{point_index:03d}-r{repetition_index:03d}"
            )
            warmup_samples, warmup_wall = await _run_phase(
                client=client,
                config=config,
                manifest=manifest,
                point=point,
                count=sweep.warmup_pages,
                workload_offset=0,
                request_prefix=f"{common_prefix}-warmup",
                consistency_hashes=consistency_hashes,
                consistency_lock=consistency_lock,
            )
            measured_samples, measured_wall = await _run_phase(
                client=client,
                config=config,
                manifest=manifest,
                point=point,
                count=sweep.measured_pages,
                workload_offset=sweep.warmup_pages,
                request_prefix=f"{common_prefix}-measured",
                consistency_hashes=consistency_hashes,
                consistency_lock=consistency_lock,
            )
            repetitions.append(
                {
                    "point_index": point_index,
                    "repetition_index": repetition_index,
                    "max_inflight_pages_global": point.max_inflight_pages_global,
                    "max_inflight_pages_per_document": (point.max_inflight_pages_per_document),
                    "warmup": _phase_report(warmup_samples, warmup_wall, include_samples=False),
                    "measured": _phase_report(
                        measured_samples, measured_wall, include_samples=True
                    ),
                }
            )

    raster_entries = [entry.model_dump(mode="json") for entry in manifest.entries]
    validated_rasters = [
        {
            "page_id": entry.page_id,
            "size_bytes": manifest.identities[entry.page_id].size_bytes,
            "width_pixels": manifest.identities[entry.page_id].width_pixels,
            "height_pixels": manifest.identities[entry.page_id].height_pixels,
        }
        for entry in manifest.entries
    ]
    final_harness_identity = await asyncio.to_thread(_harness_identity)
    if final_harness_identity != harness_identity:
        raise BenchmarkExecutionError("benchmark harness changed while the sweep was running")
    report: dict[str, Any] = {
        "schema_version": 1,
        "report_type": "glm-ocr-vllm-concurrency-benchmark",
        "benchmark_identity_sha256": benchmark_identity,
        "resolved_config": resolved_config,
        "resolved_config_sha256": resolved_config_sha256,
        "benchmark_config": benchmark_config,
        "benchmark_config_sha256": benchmark_config_sha256,
        "harness_identity": harness_identity,
        "harness_identity_sha256": harness_identity_sha256,
        "raster_manifest": {
            "source_path": str(manifest.source_path),
            "source_file_sha256": manifest.source_file_sha256,
            "canonical_sha256": manifest.canonical_sha256,
            "entry_count": len(manifest.entries),
            "entries": raster_entries,
            "validated_rasters": validated_rasters,
        },
        "server_identity": server_identity,
        "server_identity_sha256": canonical_json_sha256(server_identity),
        "text_consistency": {
            "algorithm": "sha256-utf8",
            "page_count_observed": len(consistency_hashes),
            "text_sha256_by_page_id": dict(sorted(consistency_hashes.items())),
        },
        "repetitions": repetitions,
    }
    payload = canonical_json_bytes(report) + b"\n"
    target = Path(report_path)
    created = await asyncio.to_thread(atomic_publish_bytes, target, payload)
    return PublishedBenchmarkReport(
        path=target,
        sha256=sha256_bytes(payload),
        created=created,
        report=report,
    )
