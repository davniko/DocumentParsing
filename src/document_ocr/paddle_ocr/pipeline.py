"""Bounded, resumable structured OCR using shared source/render/publication machinery."""

from __future__ import annotations

import fcntl
import json
import multiprocessing
import os
import time
from collections import defaultdict, deque
from collections.abc import Callable, Iterator
from concurrent.futures import ProcessPoolExecutor
from contextlib import contextmanager
from pathlib import Path
from typing import Any

import pyarrow as pa
import pyarrow.parquet as pq

from document_ocr.atomic import atomic_publish_json, read_regular_file_bytes
from document_ocr.hashing import canonical_json_sha256, sha256_file
from document_ocr.models import SourceObject
from document_ocr.paddle_ocr.backend import (
    PaddleBackend,
    model_identity,
    normalize_result,
    runtime_identity,
)
from document_ocr.paddle_ocr.config import PaddleConfig
from document_ocr.renderer import RenderedPage, inspect_pdf, render_page
from document_ocr.sources import discover_sources


class IncompletePaddleRun(RuntimeError):
    """Failures remain; successful page commits are retained for exact resume."""


def run_root(config: PaddleConfig) -> Path:
    return Path(config.output.root) / config.run.run_id


def _read(path: Path) -> dict[str, Any]:
    value = json.loads(read_regular_file_bytes(path))
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path}")
    return value


@contextmanager
def _locked(root: Path) -> Iterator[None]:
    root.mkdir(parents=True, exist_ok=True)
    if root.resolve() != root.absolute():
        raise ValueError("run directory must not traverse symbolic links")
    with (root / ".lock").open("a+b") as stream:
        try:
            fcntl.flock(stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise ValueError(f"another process owns {root}") from exc
        try:
            yield
        finally:
            fcntl.flock(stream, fcntl.LOCK_UN)


def _pool(workers: int) -> ProcessPoolExecutor:
    # Neither PDFium nor Paddle state is inherited from the parent process.
    return ProcessPoolExecutor(max_workers=workers, mp_context=multiprocessing.get_context("spawn"))


def _checked_source(source: SourceObject) -> Path:
    path = Path(source.local_canonical_path or "")
    if path.resolve(strict=True) != path.absolute() or path.is_symlink():
        raise ValueError("source path is no longer canonical")
    stat = path.stat()
    if (stat.st_dev, stat.st_ino, stat.st_size, stat.st_mtime_ns) != (
        source.local_device,
        source.local_inode,
        source.source_size_bytes,
        source.local_mtime_ns,
    ):
        raise ValueError(f"source changed since hashing: {path}")
    return path


def _inspect(source: SourceObject, config: PaddleConfig) -> int:
    return inspect_pdf(_checked_source(source), config.raster).page_count


def _render(source: SourceObject, index: int, output: Path, config: PaddleConfig) -> RenderedPage:
    result = render_page(_checked_source(source), index, output, config.raster)
    _checked_source(source)
    return result


def prepare_inventory(config: PaddleConfig) -> dict[str, Any]:
    """Hash sources, retain aliases, and inspect every non-excluded unique PDF."""
    root = run_root(config)
    with _locked(root):
        return _prepare(config)


def _prepare(config: PaddleConfig) -> dict[str, Any]:
    root = run_root(config)
    atomic_publish_json(root / "config.json", config.model_dump(mode="json"))
    sources = tuple(
        source
        for entry in config.sources
        for source in discover_sources(entry, max_pdf_bytes=config.raster.max_pdf_bytes)
    )
    grouped: dict[str, list[SourceObject]] = defaultdict(list)
    for source in sources:
        if source.source_sha256 is None:
            raise ValueError("local source has no content hash")
        grouped[source.source_sha256].append(source)
    for actual, expected, name in (
        (len(sources), config.expected_source_documents, "source documents"),
        (len(grouped), config.expected_unique_documents, "unique documents"),
    ):
        if expected is not None and actual != expected:
            raise ValueError(f"expected {expected} {name}, found {actual}")
    exclusions = {item.source_sha256: item.reason for item in config.exclusions}
    if exclusions.keys() - grouped.keys():
        raise ValueError("configured exclusions must match actual source hashes")
    documents = []
    with _pool(config.concurrency.renderer_processes) as pool:
        # Limit outstanding inspection work as well as rendering work.
        pending: deque[Any] = deque()
        items = iter(sorted(grouped.items()))
        exhausted = False
        while pending or not exhausted:
            while not exhausted and len(pending) < config.concurrency.prefetch_pages:
                item = next(items, None)
                if item is None:
                    exhausted = True
                    break
                digest, aliases = item
                future = None if digest in exclusions else pool.submit(_inspect, aliases[0], config)
                pending.append((digest, aliases, future))
            if pending:
                digest, aliases, future = pending.popleft()
                documents.append(
                    {
                        "document_id": "pdf_" + digest,
                        "source_sha256": digest,
                        "sources": [source.model_dump(mode="json") for source in aliases],
                        "excluded_reason": exclusions.get(digest),
                        "page_count": None if future is None else future.result(),
                    }
                )
    pages = sum(doc["page_count"] or 0 for doc in documents)
    if config.run.expected_pages is not None and pages != config.run.expected_pages:
        raise ValueError(f"expected {config.run.expected_pages} pages, found {pages}")
    inventory = {
        "schema_version": 1,
        "source_documents": len(sources),
        "unique_documents": len(documents),
        "excluded_documents": len(exclusions),
        "pages": pages,
        "documents": documents,
    }
    atomic_publish_json(root / "inventory.json", inventory)
    return inventory


def _page_path(root: Path, document_id: str, index: int) -> Path:
    return root / "pages" / document_id / f"{index + 1:06}.json"


def _artifact(root: Path, path: Path) -> dict[str, Any]:
    return {
        "path": path.relative_to(root).as_posix(),
        "sha256": sha256_file(path),
        "size_bytes": path.stat().st_size,
    }


def _verify_artifact(root: Path, artifact: dict[str, Any]) -> Path:
    path: Path = root / artifact["path"]
    if not path.resolve(strict=True).is_relative_to(root.resolve()) or path.is_symlink():
        raise ValueError(f"invalid artifact path: {path}")
    if path.stat().st_size != artifact["size_bytes"] or sha256_file(path) != artifact["sha256"]:
        raise ValueError(f"artifact hash/size mismatch: {path}")
    return path


def _verified_page(
    root: Path, document: dict[str, Any], index: int, fingerprint: str
) -> dict[str, Any] | None:
    path = _page_path(root, document["document_id"], index)
    if not path.exists():
        return None
    record = _read(path)
    for name, expected in (
        ("document_id", document["document_id"]),
        ("source_sha256", document["source_sha256"]),
        ("page_index", index),
        ("pipeline_fingerprint", fingerprint),
    ):
        if record[name] != expected:
            raise ValueError(f"page identity mismatch: {path}: {name}")
    for artifact in record["artifacts"]:
        _verify_artifact(root, artifact)
    native = _read(root / record["result"]["native_path"])
    normalized = normalize_result(native, record["result"]["width"], record["result"]["height"])
    if any(record["result"][key] != value for key, value in normalized.items()):
        raise ValueError(f"structured result differs from native OCR: {path}")
    return record


def _provenance(config: PaddleConfig, project_root: Path) -> dict[str, Any]:
    paths = [*sorted((project_root / "src/document_ocr/paddle_ocr").glob("*.py"))]
    paths.extend(
        project_root / "src/document_ocr" / name
        for name in (
            "atomic.py",
            "sources.py",
            "renderer.py",
            "config.py",
            "models.py",
            "hashing.py",
        )
    )
    identity = {
        "config": config.model_dump(mode="json"),
        "runtime": runtime_identity(),
        "models": model_identity(config.paddle),
        "code": {p.relative_to(project_root).as_posix(): sha256_file(p) for p in paths},
        "inventory_sha256": sha256_file(run_root(config) / "inventory.json"),
    }
    return {**identity, "pipeline_fingerprint": canonical_json_sha256(identity)}


def _jobs(
    config: PaddleConfig, inventory: dict[str, Any], fingerprint: str
) -> Iterator[tuple[dict[str, Any], int]]:
    root = run_root(config)
    for doc in inventory["documents"]:
        for index in range(doc["page_count"] or 0):
            if _verified_page(root, doc, index, fingerprint) is None:
                yield doc, index


def run(
    config: PaddleConfig,
    project_root: Path,
    progress: Callable[[dict[str, Any]], None],
    *,
    backend_factory: Any = PaddleBackend,
) -> dict[str, Any]:
    root = run_root(config)
    with _locked(root):
        if not config.run.resume and (root / "provenance.json").exists():
            raise ValueError("run already exists and resume is disabled")
        inventory = _prepare(config)
        provenance = _provenance(config, project_root)
        atomic_publish_json(root / "provenance.json", provenance)
        if (root / "manifest.json").exists():
            return status(config)
        fingerprint = provenance["pipeline_fingerprint"]
        jobs = iter(_jobs(config, inventory, fingerprint))
        first = next(jobs, None)
        completed = 0
        failures = []
        started = time.perf_counter()
        if first is not None:
            backend = backend_factory(config.paddle)
            try:
                with _pool(config.concurrency.renderer_processes) as pool:
                    pending: deque[Any] = deque()
                    item: tuple[dict[str, Any], int] | None = first
                    while item is not None or pending:
                        while item is not None and len(pending) < config.concurrency.prefetch_pages:
                            doc, index = item
                            image = (
                                root
                                / "rasters"
                                / doc["document_id"]
                                / (
                                    f"{index + 1:06}."
                                    + ("png" if config.raster.image_format == "png" else "jpg")
                                )
                            )
                            image.parent.mkdir(parents=True, exist_ok=True)
                            source = SourceObject.model_validate_json(json.dumps(doc["sources"][0]))
                            pending.append(
                                (
                                    doc,
                                    index,
                                    image,
                                    pool.submit(_render, source, index, image, config),
                                )
                            )
                            item = next(jobs, None)
                        doc, index, image, future = pending.popleft()
                        try:
                            rendered = future.result()
                            page_started = time.perf_counter()
                            # A unique attempt directory preserves interrupted native results.
                            attempt_root = (
                                root
                                / "native"
                                / doc["document_id"]
                                / str(index + 1)
                                / str(time.time_ns())
                            )
                            result = backend.recognize(image, attempt_root)
                            elapsed = time.perf_counter() - page_started
                            native = Path(result["native_path"])
                            coordinate = Path(result["coordinate_image_path"])
                            artifacts = [_artifact(root, native)]
                            retain_coordinate = (
                                coordinate != image or config.output.retain_page_images
                            )
                            if retain_coordinate:
                                artifacts.append(_artifact(root, coordinate))
                            if config.output.retain_page_images and coordinate != image:
                                artifacts.append(_artifact(root, image))
                            result["native_path"] = native.relative_to(root).as_posix()
                            result["coordinate_image_path"] = (
                                coordinate.relative_to(root).as_posix()
                                if retain_coordinate
                                else None
                            )
                            record = {
                                "schema_version": 1,
                                "pipeline_fingerprint": fingerprint,
                                "document_id": doc["document_id"],
                                "source_sha256": doc["source_sha256"],
                                "page_index": index,
                                "page_number": index + 1,
                                "raster": rendered.raster.model_dump(mode="json"),
                                "recognition_seconds": elapsed,
                                "result": result,
                                "artifacts": artifacts,
                            }
                            atomic_publish_json(_page_path(root, doc["document_id"], index), record)
                            if not config.output.retain_page_images:
                                image.unlink()
                            completed += 1
                        except Exception as exc:
                            error = {
                                "document_id": doc["document_id"],
                                "page_index": index,
                                "error_type": type(exc).__name__,
                                "error": str(exc),
                            }
                            failures.append(error)
                            atomic_publish_json(root / "errors" / f"{time.time_ns()}.json", error)
                            if config.run.fail_fast:
                                raise
                        progress(
                            {
                                "processed_pages": completed + len(failures),
                                "successful_pages_this_invocation": completed,
                                "failed_pages_this_invocation": len(failures),
                                "total_pages": inventory["pages"],
                                "elapsed_seconds": round(time.perf_counter() - started, 3),
                            }
                        )
            finally:
                backend.close()
        if failures:
            raise IncompletePaddleRun(
                f"{len(failures)} pages failed; commits retained; rerun to retry"
            )
        _publish(config, inventory, fingerprint)
        return status(config)


def _publish(config: PaddleConfig, inventory: dict[str, Any], fingerprint: str) -> None:
    root = run_root(config)
    page_schema = pa.schema(
        [
            ("document_id", pa.string()),
            ("source_sha256", pa.string()),
            ("page_index", pa.int32()),
            ("width", pa.int32()),
            ("height", pa.int32()),
            ("coordinate_frame", pa.string()),
            ("coordinate_image_path", pa.string()),
            ("native_path", pa.string()),
            ("detections", pa.int32()),
            ("recognized_lines", pa.int32()),
        ]
    )
    line_schema = pa.schema(
        [
            ("document_id", pa.string()),
            ("page_index", pa.int32()),
            ("line_index", pa.int32()),
            ("detection_index", pa.int32()),
            ("text", pa.large_string()),
            ("recognition_score", pa.float64()),
            ("x1", pa.float64()),
            ("y1", pa.float64()),
            ("x2", pa.float64()),
            ("y2", pa.float64()),
            ("polygon_json", pa.string()),
        ]
    )
    schemas = {"pages": page_schema, "lines": line_schema}
    writers: dict[str, Any] = {}
    buffers: dict[str, list[dict[str, Any]]] = {"pages": [], "lines": []}
    counts = {"pages": 0, "lines": 0}
    compression = (
        None if config.output.parquet_compression == "none" else config.output.parquet_compression
    )

    def append(name: str, row: dict[str, Any]) -> None:
        buffers[name].append(row)
        counts[name] += 1
        if len(buffers[name]) >= config.output.write_batch_rows:
            writers[name].write_table(pa.Table.from_pylist(buffers[name], schema=schemas[name]))
            buffers[name].clear()

    commits = []
    try:
        for name, schema in schemas.items():
            writers[name] = pq.ParquetWriter(
                root / f".{name}.parquet.tmp", schema, compression=compression
            )
        for doc in inventory["documents"]:
            for index in range(doc["page_count"] or 0):
                record = _verified_page(root, doc, index, fingerprint)
                if record is None:
                    raise IncompletePaddleRun(f"missing page: {doc['document_id']}/{index}")
                commits.append(_artifact(root, _page_path(root, doc["document_id"], index)))
                result = record["result"]
                append(
                    "pages",
                    {
                        "document_id": doc["document_id"],
                        "source_sha256": doc["source_sha256"],
                        "page_index": index,
                        **{
                            k: result[k]
                            for k in (
                                "width",
                                "height",
                                "coordinate_frame",
                                "coordinate_image_path",
                                "native_path",
                            )
                        },
                        "detections": len(result["detections"]),
                        "recognized_lines": len(result["lines"]),
                    },
                )
                for line in result["lines"]:
                    append(
                        "lines",
                        {
                            "document_id": doc["document_id"],
                            "page_index": index,
                            "line_index": line["index"],
                            "detection_index": line["detection_index"],
                            "text": line["text"],
                            "recognition_score": line["recognition_score"],
                            **dict(zip(("x1", "y1", "x2", "y2"), line["bbox"], strict=True)),
                            "polygon_json": json.dumps(line["polygon"], separators=(",", ":")),
                        },
                    )
        for name in writers:
            if buffers[name]:
                writers[name].write_table(pa.Table.from_pylist(buffers[name], schema=schemas[name]))
    finally:
        for writer in writers.values():
            writer.close()
    tables = []
    for name in schemas:
        temporary = root / f".{name}.parquet.tmp"
        with temporary.open("rb") as stream:
            os.fsync(stream.fileno())
        destination = root / f"{name}.parquet"
        os.replace(temporary, destination)
        tables.append(_artifact(root, destination))
    atomic_publish_json(
        root / "manifest.json",
        {
            "schema_version": 1,
            "backend": "paddleocr",
            "pipeline_fingerprint": fingerprint,
            "source_documents": inventory["source_documents"],
            "unique_documents": inventory["unique_documents"],
            "excluded_documents": inventory["excluded_documents"],
            "counts": counts,
            "artifacts": [
                _artifact(root, root / name)
                for name in ("config.json", "inventory.json", "provenance.json")
            ]
            + tables
            + commits,
            "training_text_generated": False,
        },
    )


def status(config: PaddleConfig) -> dict[str, Any]:
    """Verify every committed page and native artifact, including on completed resume."""
    root = run_root(config)
    if _read(root / "config.json") != config.model_dump(mode="json"):
        raise ValueError("configuration differs from this run")
    inventory = _read(root / "inventory.json")
    if not (root / "provenance.json").exists():
        if (root / "manifest.json").exists() or any((root / "pages").rglob("*.json")):
            raise ValueError("OCR results exist without run provenance")
        return {
            "status": "not_started",
            "run_root": str(root),
            "source_documents": inventory["source_documents"],
            "unique_documents": inventory["unique_documents"],
            "excluded_documents": inventory["excluded_documents"],
            "total_pages": inventory["pages"],
            "successful_pages": 0,
            "remaining_pages": inventory["pages"],
        }
    provenance = _read(root / "provenance.json")
    fingerprint = provenance["pipeline_fingerprint"]
    if (
        canonical_json_sha256({k: v for k, v in provenance.items() if k != "pipeline_fingerprint"})
        != fingerprint
    ):
        raise ValueError("provenance fingerprint mismatch")
    if sha256_file(root / "inventory.json") != provenance["inventory_sha256"]:
        raise ValueError("inventory changed after processing")
    successful = 0
    for doc in inventory["documents"]:
        for index in range(doc["page_count"] or 0):
            successful += _verified_page(root, doc, index, fingerprint) is not None
    manifest_path = root / "manifest.json"
    if manifest_path.exists():
        manifest = _read(manifest_path)
        if (
            manifest["pipeline_fingerprint"] != fingerprint
            or manifest["counts"]["pages"] != inventory["pages"]
        ):
            raise ValueError("manifest identity/count mismatch")
        for artifact in manifest["artifacts"]:
            _verify_artifact(root, artifact)
    return {
        "status": "complete"
        if successful == inventory["pages"] and manifest_path.exists()
        else "incomplete",
        "run_root": str(root),
        "source_documents": inventory["source_documents"],
        "unique_documents": inventory["unique_documents"],
        "excluded_documents": inventory["excluded_documents"],
        "total_pages": inventory["pages"],
        "successful_pages": successful,
        "remaining_pages": inventory["pages"] - successful,
    }
