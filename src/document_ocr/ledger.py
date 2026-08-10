"""Durable, content-level extraction state for safe resume."""

from __future__ import annotations

import asyncio
import hashlib
import json
import math
import os
import stat
from collections.abc import AsyncIterator, Mapping, Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, cast

import aiosqlite

from document_ocr.config import PipelineConfig
from document_ocr.hashing import canonical_json_bytes, canonical_json_sha256, sha256_bytes
from document_ocr.models import (
    DatasetManifest,
    DocumentExtractionFailure,
    InferenceAttempt,
    PageExtractionFailure,
    PageExtractionRecord,
    PageProvenance,
    SourceObject,
)
from document_ocr.vllm_contract import runtime_contract_payload, runtime_contract_sha256


class LedgerConflictError(RuntimeError):
    """An immutable run, document, page, attempt, or result identity drifted."""


class IncompleteRunError(RuntimeError):
    """A final dataset was requested before every expected page succeeded."""


def _utc_now() -> str:
    return datetime.now(UTC).isoformat()


def _json_text(value: Mapping[str, Any]) -> str:
    return canonical_json_bytes(value).decode("utf-8")


def _absolute_path(path: Path) -> Path:
    return Path(os.path.abspath(path))


def _model_json(record: SourceObject) -> dict[str, Any]:
    return record.model_dump(mode="json")


def _source_from_record(record: SourceObject) -> SourceObject:
    values = {name: getattr(record, name) for name in SourceObject.model_fields}
    return SourceObject.model_validate(values, strict=True)


def _required_no_follow_flags(*, writable: bool = False) -> int:
    missing = [name for name in ("O_CLOEXEC", "O_NOFOLLOW") if not hasattr(os, name)]
    if missing:
        raise LedgerConflictError(
            "safe ledger artifact access requires operating-system flags: " + ", ".join(missing)
        )
    access = os.O_RDWR if writable else os.O_RDONLY
    return access | os.O_CLOEXEC | os.O_NOFOLLOW


def _hash_regular_artifact(path: Path, *, allowed_root: Path) -> tuple[int, str]:
    """Hash one artifact only after proving canonical, non-symlink containment."""

    try:
        root = allowed_root.resolve(strict=True)
        resolved = path.resolve(strict=True)
    except OSError as error:
        raise LedgerConflictError(f"published dataset artifact is missing: {path}") from error
    if (
        root != Path(os.path.abspath(allowed_root))
        or resolved != Path(os.path.abspath(path))
        or not resolved.is_relative_to(root)
    ):
        raise LedgerConflictError(f"published dataset artifact is not canonical: {path}")
    try:
        descriptor = os.open(resolved, _required_no_follow_flags())
    except OSError as error:
        raise LedgerConflictError(
            f"published dataset artifact cannot be opened safely: {path}"
        ) from error
    digest = hashlib.sha256()
    size_bytes = 0
    try:
        if not stat.S_ISREG(os.fstat(descriptor).st_mode):
            raise LedgerConflictError(f"published dataset artifact is not a regular file: {path}")
        with os.fdopen(descriptor, "rb", closefd=False) as stream:
            while chunk := stream.read(1024 * 1024):
                digest.update(chunk)
                size_bytes += len(chunk)
    finally:
        os.close(descriptor)
    return size_bytes, digest.hexdigest()


class ExtractionLedger:
    """Single-process async facade over a crash-safe SQLite WAL ledger.

    SQLite is the retry/resume authority. Parquet is a final, immutable projection
    produced only after :meth:`require_complete` proves page-level completeness.
    """

    def __init__(self, path: Path, *, read_only: bool = False) -> None:
        self.path = path
        self.read_only = read_only
        self._connection: aiosqlite.Connection | None = None
        self._write_lock = asyncio.Lock()

    async def __aenter__(self) -> ExtractionLedger:
        await self.open()
        return self

    async def __aexit__(self, *_: object) -> None:
        await self.close()

    @property
    def connection(self) -> aiosqlite.Connection:
        if self._connection is None:
            raise RuntimeError("ledger is not open")
        return self._connection

    async def open(self) -> None:
        if self._connection is not None:
            return
        if self.read_only:
            try:
                resolved = self.path.resolve(strict=True)
            except OSError as error:
                raise IncompleteRunError(f"run ledger does not exist: {self.path}") from error
            if resolved != _absolute_path(self.path) or not resolved.is_file():
                raise LedgerConflictError(
                    f"run ledger must be a canonical regular file: {self.path}"
                )
            connection = await aiosqlite.connect(
                f"{resolved.as_uri()}?mode=ro&immutable=1",
                uri=True,
                timeout=30.0,
            )
            connection.row_factory = aiosqlite.Row
            await connection.execute("PRAGMA query_only=ON")
            await connection.execute("PRAGMA foreign_keys=ON")
            await connection.execute("PRAGMA busy_timeout=30000")
            self._connection = connection
            return
        self.path.parent.mkdir(parents=True, exist_ok=True)
        absolute_parent = _absolute_path(self.path.parent)
        if self.path.parent.resolve(strict=True) != absolute_parent:
            raise LedgerConflictError(
                f"run ledger parent must not traverse symbolic links: {self.path.parent}"
            )
        absolute_path = _absolute_path(self.path)
        try:
            details = self.path.stat(follow_symlinks=False)
        except FileNotFoundError:
            flags = _required_no_follow_flags(writable=True) | os.O_CREAT | os.O_EXCL
            try:
                descriptor = os.open(absolute_path, flags, 0o600)
            except OSError as error:
                raise LedgerConflictError(
                    f"run ledger cannot be created safely: {self.path}"
                ) from error
            else:
                os.close(descriptor)
        else:
            if not stat.S_ISREG(details.st_mode) or self.path.resolve(strict=True) != absolute_path:
                raise LedgerConflictError(
                    f"run ledger must be a canonical regular file: {self.path}"
                )
        connection = await aiosqlite.connect(self.path, timeout=30.0)
        connection.row_factory = aiosqlite.Row
        await connection.execute("PRAGMA journal_mode=WAL")
        await connection.execute("PRAGMA synchronous=FULL")
        await connection.execute("PRAGMA foreign_keys=ON")
        await connection.execute("PRAGMA busy_timeout=30000")
        await connection.executescript(
            """
            CREATE TABLE IF NOT EXISTS runs (
                run_id TEXT PRIMARY KEY,
                config_sha256 TEXT NOT NULL,
                pipeline_fingerprint TEXT NOT NULL,
                config_json TEXT NOT NULL,
                inventory_sha256 TEXT NOT NULL,
                inventory_document_count INTEGER NOT NULL
                    CHECK (inventory_document_count > 0),
                provenance_json TEXT NOT NULL,
                status TEXT NOT NULL
                    CHECK (status IN ('running', 'failed', 'complete')),
                manifest_sha256 TEXT,
                manifest_json TEXT,
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL
            );

            CREATE TABLE IF NOT EXISTS documents (
                run_id TEXT NOT NULL,
                document_id TEXT NOT NULL,
                source_json TEXT NOT NULL,
                source_sha256 TEXT,
                page_count INTEGER CHECK (page_count > 0),
                status TEXT NOT NULL
                    CHECK (status IN ('pending', 'failed', 'incomplete', 'complete')),
                failure_json TEXT,
                updated_at TEXT NOT NULL,
                PRIMARY KEY (run_id, document_id),
                FOREIGN KEY (run_id) REFERENCES runs(run_id)
            );

            CREATE TABLE IF NOT EXISTS pages (
                run_id TEXT NOT NULL,
                extraction_id TEXT NOT NULL,
                document_id TEXT NOT NULL,
                page_id TEXT NOT NULL,
                page_index INTEGER NOT NULL CHECK (page_index >= 0),
                status TEXT NOT NULL CHECK (status IN ('pending', 'failed', 'success')),
                result_json TEXT,
                failure_json TEXT,
                ocr_text_sha256 TEXT,
                raw_response_sha256 TEXT,
                updated_at TEXT NOT NULL,
                PRIMARY KEY (run_id, extraction_id),
                UNIQUE (run_id, document_id, page_index),
                UNIQUE (run_id, page_id),
                FOREIGN KEY (run_id, document_id)
                    REFERENCES documents(run_id, document_id)
            );

            CREATE TABLE IF NOT EXISTS attempts (
                run_id TEXT NOT NULL,
                extraction_id TEXT NOT NULL,
                attempt_number INTEGER NOT NULL CHECK (attempt_number > 0),
                attempt_json TEXT NOT NULL,
                created_at TEXT NOT NULL,
                PRIMARY KEY (run_id, extraction_id, attempt_number),
                FOREIGN KEY (run_id, extraction_id)
                    REFERENCES pages(run_id, extraction_id)
            );
            """
        )
        await connection.commit()
        self._connection = connection

    async def close(self) -> None:
        if self._connection is None:
            return
        await self._connection.close()
        self._connection = None

    async def _fetchone(
        self, statement: str, parameters: tuple[object, ...]
    ) -> aiosqlite.Row | None:
        cursor = await self.connection.execute(statement, parameters)
        async with cursor:
            return await cursor.fetchone()

    async def _validate_page_record_contract(self, record: PageProvenance) -> aiosqlite.Row:
        row = await self._fetchone(
            """
            SELECT p.document_id, p.page_id, p.page_index,
                   d.source_json, d.source_sha256, d.page_count,
                   r.config_sha256, r.pipeline_fingerprint, r.config_json
            FROM pages p
            JOIN documents d
              ON d.run_id = p.run_id AND d.document_id = p.document_id
            JOIN runs r ON r.run_id = p.run_id
            WHERE p.run_id = ? AND p.extraction_id = ?
            """,
            (record.run_id, record.extraction_id),
        )
        if row is None:
            raise LedgerConflictError(f"unregistered extraction {record.extraction_id}")
        if (
            row["document_id"] != record.document_id
            or row["page_id"] != record.page_id
            or row["page_index"] != record.page_index
            or row["page_count"] != record.document_page_count
            or row["source_sha256"] != record.source_sha256
            or row["config_sha256"] != record.config_sha256
            or row["pipeline_fingerprint"] != record.pipeline_fingerprint
        ):
            raise LedgerConflictError(
                f"page provenance conflicts with registered extraction {record.extraction_id}"
            )
        try:
            inventory_source = cast(dict[str, Any], json.loads(row["source_json"]))
        except (json.JSONDecodeError, TypeError) as error:
            raise LedgerConflictError("registered document source JSON is corrupt") from error
        actual_source = _source_from_record(record).model_dump(mode="json")
        inventory_without_materialized_hash = {
            key: value for key, value in inventory_source.items() if key != "source_sha256"
        }
        actual_without_materialized_hash = {
            key: value for key, value in actual_source.items() if key != "source_sha256"
        }
        if inventory_without_materialized_hash != actual_without_materialized_hash:
            raise LedgerConflictError(
                f"source provenance conflicts with registered document {record.document_id}"
            )
        return row

    @staticmethod
    def _validate_success_execution_contract(
        result: PageExtractionRecord, run_row: aiosqlite.Row
    ) -> None:
        try:
            config = PipelineConfig.model_validate_json(run_row["config_json"], strict=True)
        except Exception as error:
            raise LedgerConflictError("registered run config JSON is corrupt") from error
        expected_values: dict[str, object] = {
            "inference_endpoint": config.vllm.endpoint,
            "inference_model": config.vllm.model,
            "inference_served_model_name": config.vllm.served_model_name,
            "inference_model_revision": config.vllm.revision,
            "inference_server_engine": "vllm",
            "inference_server_engine_version": config.vllm.engine_version,
            "inference_server_image": config.vllm.container_image,
            "inference_server_contract_sha256": runtime_contract_sha256(
                runtime_contract_payload(
                    model=config.vllm.model,
                    served_model_name=config.vllm.served_model_name,
                    model_revision=config.vllm.revision,
                    max_model_len=config.vllm.max_model_len,
                    max_num_seqs=config.vllm.max_num_seqs,
                    gpu_memory_utilization=config.vllm.gpu_memory_utilization,
                    generation_config="vllm",
                    speculative_method=config.vllm.speculative_decoding.method,
                    num_speculative_tokens=(
                        config.vllm.speculative_decoding.num_speculative_tokens
                    ),
                    image_limit_per_prompt=1,
                    container_image=config.vllm.container_image,
                )
            ),
            "inference_speculative_method": config.vllm.speculative_decoding.method,
            "inference_num_speculative_tokens": (
                config.vllm.speculative_decoding.num_speculative_tokens
            ),
            "inference_prompt": config.vllm.prompt,
            "inference_prompt_sha256": sha256_bytes(config.vllm.prompt.encode("utf-8")),
            "inference_temperature": config.vllm.sampling.temperature,
            "inference_top_p": config.vllm.sampling.top_p,
            "inference_top_k": config.vllm.sampling.top_k,
            "inference_repetition_penalty": config.vllm.sampling.repetition_penalty,
            "inference_max_tokens": config.vllm.sampling.max_tokens,
            "inference_seed": config.vllm.sampling.seed,
            "requested_dpi": config.raster.dpi,
            "raster_image_format": config.raster.image_format,
            "raster_mime_type": (
                "image/png" if config.raster.image_format == "png" else "image/jpeg"
            ),
            "draw_annotations": config.raster.draw_annotations,
        }
        for field_name, expected in expected_values.items():
            if getattr(result, field_name) != expected:
                raise LedgerConflictError(
                    f"result field {field_name} conflicts with initialized extraction config"
                )
        if result.inference_finish_reason != "stop":
            raise LedgerConflictError("successful result finish_reason must be 'stop'")
        if (
            max(result.raster_width_px, result.raster_height_px) > config.raster.max_side_pixels
            or result.raster_width_px * result.raster_height_px > config.raster.max_pixels
        ):
            raise LedgerConflictError("result raster dimensions exceed initialized bounds")
        expected_width = math.ceil(result.page_width_points * result.render_scale)
        expected_height = math.ceil(result.page_height_points * result.render_scale)
        if (result.raster_width_px, result.raster_height_px) != (
            expected_width,
            expected_height,
        ) or not math.isclose(
            result.effective_dpi,
            result.render_scale * 72.0,
            rel_tol=1e-12,
            abs_tol=1e-9,
        ):
            raise LedgerConflictError("result raster geometry is internally inconsistent")
        expected_raw_path = (
            f"raw-responses/{result.document_id}/{result.page_id}/{result.raw_response_sha256}.json"
        )
        if result.raw_response_path != expected_raw_path:
            raise LedgerConflictError(
                "raw_response_path is not bound to page identity and response hash"
            )

    async def initialize_run(
        self,
        *,
        run_id: str,
        config_sha256: str,
        pipeline_fingerprint: str,
        config: Mapping[str, Any],
        inventory_sha256: str,
        inventory_document_count: int,
        provenance: Mapping[str, Any],
        resume: bool,
    ) -> str:
        """Create a run or prove that a requested resume is byte-identical."""

        try:
            typed_config = PipelineConfig.model_validate(config, strict=True)
        except Exception as error:
            raise LedgerConflictError(
                "run configuration does not satisfy PipelineConfig"
            ) from error
        if typed_config.run.run_id != run_id:
            raise LedgerConflictError("run_id does not match the validated run configuration")
        config_payload = typed_config.model_dump(mode="json")
        actual_config_sha256 = canonical_json_sha256(config_payload)
        if config_sha256 != actual_config_sha256:
            raise LedgerConflictError(
                "config_sha256 does not match the canonical validated run configuration"
            )
        config_json = _json_text(config_payload)
        provenance_json = _json_text(provenance)
        async with self._write_lock:
            row = await self._fetchone(
                """
                SELECT config_sha256, pipeline_fingerprint, config_json,
                       inventory_sha256, inventory_document_count,
                       provenance_json, status
                FROM runs WHERE run_id = ?
                """,
                (run_id,),
            )
            if row is not None:
                if not resume:
                    raise LedgerConflictError(
                        f"run_id {run_id!r} already exists; choose a new run_id or enable resume"
                    )
                expected = (
                    config_sha256,
                    pipeline_fingerprint,
                    config_json,
                    inventory_sha256,
                    inventory_document_count,
                    provenance_json,
                )
                actual = tuple(
                    row[key]
                    for key in (
                        "config_sha256",
                        "pipeline_fingerprint",
                        "config_json",
                        "inventory_sha256",
                        "inventory_document_count",
                        "provenance_json",
                    )
                )
                if actual != expected:
                    raise LedgerConflictError(
                        f"run_id {run_id!r} cannot resume with changed executable state, "
                        "configuration, source inventory, or runtime provenance"
                    )
                status = cast(str, row["status"])
                if status != "complete":
                    await self.connection.execute(
                        "UPDATE runs SET status = 'running', updated_at = ? WHERE run_id = ?",
                        (_utc_now(), run_id),
                    )
                    await self.connection.commit()
                return status

            now = _utc_now()
            await self.connection.execute(
                """
                INSERT INTO runs (
                    run_id, config_sha256, pipeline_fingerprint, config_json,
                    inventory_sha256, inventory_document_count, provenance_json,
                    status, created_at, updated_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, 'running', ?, ?)
                """,
                (
                    run_id,
                    config_sha256,
                    pipeline_fingerprint,
                    config_json,
                    inventory_sha256,
                    inventory_document_count,
                    provenance_json,
                    now,
                    now,
                ),
            )
            await self.connection.commit()
            return "running"

    async def register_document(
        self,
        *,
        run_id: str,
        inventory_source: SourceObject,
        source_sha256: str,
        page_count: int,
    ) -> None:
        """Register the immutable source identity after PDF inspection succeeds."""

        document_id = inventory_source.document_id
        source_json = _json_text(_model_json(inventory_source))
        async with self._write_lock:
            row = await self._fetchone(
                """
                SELECT source_json, source_sha256, page_count, status
                FROM documents WHERE run_id = ? AND document_id = ?
                """,
                (run_id, document_id),
            )
            if row is not None:
                if row["source_json"] != source_json:
                    raise LedgerConflictError(f"document identity conflict for {document_id}")
                for key, value in (("source_sha256", source_sha256), ("page_count", page_count)):
                    if row[key] is not None and row[key] != value:
                        raise LedgerConflictError(
                            f"document {document_id} resumed with changed {key}"
                        )
                resumed_status = "complete" if row["status"] == "complete" else "pending"
                await self.connection.execute(
                    """
                    UPDATE documents SET source_sha256 = ?, page_count = ?,
                        status = ?, failure_json = NULL, updated_at = ?
                    WHERE run_id = ? AND document_id = ?
                    """,
                    (
                        source_sha256,
                        page_count,
                        resumed_status,
                        _utc_now(),
                        run_id,
                        document_id,
                    ),
                )
            else:
                await self.connection.execute(
                    """
                    INSERT INTO documents (
                        run_id, document_id, source_json, source_sha256,
                        page_count, status, updated_at
                    ) VALUES (?, ?, ?, ?, ?, 'pending', ?)
                    """,
                    (
                        run_id,
                        document_id,
                        source_json,
                        source_sha256,
                        page_count,
                        _utc_now(),
                    ),
                )
            await self.connection.commit()

    async def record_document_failure(
        self,
        *,
        failure: DocumentExtractionFailure,
    ) -> None:
        """Persist an explicit failure for a PDF that could not expose page rows."""

        run_id = failure.run_id
        document_id = failure.document_id
        source_json = _json_text(_model_json(_source_from_record(failure)))
        failure_json = _json_text(failure.model_dump(mode="json"))
        async with self._write_lock:
            run = await self._fetchone(
                "SELECT config_sha256, pipeline_fingerprint FROM runs WHERE run_id = ?",
                (run_id,),
            )
            if run is None:
                raise LedgerConflictError(f"unknown run_id {run_id!r}")
            if (
                run["config_sha256"] != failure.config_sha256
                or run["pipeline_fingerprint"] != failure.pipeline_fingerprint
            ):
                raise LedgerConflictError(
                    f"document failure provenance conflicts with run {run_id!r}"
                )
            row = await self._fetchone(
                """
                SELECT source_json FROM documents
                WHERE run_id = ? AND document_id = ?
                """,
                (run_id, document_id),
            )
            if row is not None and row["source_json"] != source_json:
                raise LedgerConflictError(f"document identity conflict for {document_id}")
            if row is not None:
                await self.connection.execute(
                    """
                    UPDATE documents SET status = 'failed', failure_json = ?, updated_at = ?
                    WHERE run_id = ? AND document_id = ?
                    """,
                    (failure_json, _utc_now(), run_id, document_id),
                )
            else:
                await self.connection.execute(
                    """
                    INSERT INTO documents (
                        run_id, document_id, source_json, status, failure_json, updated_at
                    ) VALUES (?, ?, ?, 'failed', ?, ?)
                    """,
                    (run_id, document_id, source_json, failure_json, _utc_now()),
                )
            await self.connection.commit()

    async def register_page(
        self,
        *,
        run_id: str,
        extraction_id: str,
        document_id: str,
        page_id: str,
        page_index: int,
    ) -> None:
        async with self._write_lock:
            row = await self._fetchone(
                """
                SELECT document_id, page_id, page_index
                FROM pages WHERE run_id = ? AND extraction_id = ?
                """,
                (run_id, extraction_id),
            )
            identity = (document_id, page_id, page_index)
            if row is not None:
                actual = tuple(row[key] for key in ("document_id", "page_id", "page_index"))
                if actual != identity:
                    raise LedgerConflictError(f"extraction identity conflict for {extraction_id}")
                return
            try:
                await self.connection.execute(
                    """
                    INSERT INTO pages (
                        run_id, extraction_id, document_id, page_id,
                        page_index, status, updated_at
                    ) VALUES (?, ?, ?, ?, ?, 'pending', ?)
                    """,
                    (run_id, extraction_id, document_id, page_id, page_index, _utc_now()),
                )
            except aiosqlite.IntegrityError as error:
                raise LedgerConflictError(
                    f"page identity conflicts with an existing page: {page_id}"
                ) from error
            await self.connection.commit()

    async def register_pages(
        self,
        *,
        run_id: str,
        document_id: str,
        pages: Sequence[tuple[str, str, int]],
    ) -> None:
        """Register a document's page identities in one durable transaction."""

        if not pages:
            raise ValueError("pages must not be empty")
        if len({page_index for _, _, page_index in pages}) != len(pages):
            raise LedgerConflictError(f"duplicate page indexes for document {document_id}")
        async with self._write_lock:
            document = await self._fetchone(
                "SELECT page_count FROM documents WHERE run_id = ? AND document_id = ?",
                (run_id, document_id),
            )
            if document is None or document["page_count"] is None:
                raise LedgerConflictError(f"unregistered document {document_id}")
            expected_indexes = set(range(int(document["page_count"])))
            supplied_indexes = {page_index for _, _, page_index in pages}
            if supplied_indexes != expected_indexes:
                raise LedgerConflictError(
                    f"bulk page inventory for {document_id} must be exactly zero-based and complete"
                )
            cursor = await self.connection.execute(
                """
                SELECT extraction_id, page_id, page_index
                FROM pages WHERE run_id = ? AND document_id = ?
                """,
                (run_id, document_id),
            )
            async with cursor:
                existing_rows = await cursor.fetchall()
            existing = {
                int(row["page_index"]): (row["extraction_id"], row["page_id"])
                for row in existing_rows
            }
            expected = {
                page_index: (extraction_id, page_id) for extraction_id, page_id, page_index in pages
            }
            for page_index, identity in existing.items():
                if expected.get(page_index) != identity:
                    raise LedgerConflictError(
                        f"page identity conflict for document {document_id} page {page_index}"
                    )
            new_rows = [
                (run_id, extraction_id, document_id, page_id, page_index, _utc_now())
                for extraction_id, page_id, page_index in pages
                if page_index not in existing
            ]
            try:
                await self.connection.executemany(
                    """
                    INSERT INTO pages (
                        run_id, extraction_id, document_id, page_id,
                        page_index, status, updated_at
                    ) VALUES (?, ?, ?, ?, ?, 'pending', ?)
                    """,
                    new_rows,
                )
            except aiosqlite.IntegrityError as error:
                raise LedgerConflictError(
                    f"page identity conflicts with an existing page in document {document_id}"
                ) from error
            await self.connection.commit()

    async def is_document_complete(self, run_id: str, document_id: str) -> bool:
        row = await self._fetchone(
            "SELECT status FROM documents WHERE run_id = ? AND document_id = ?",
            (run_id, document_id),
        )
        return row is not None and row["status"] == "complete"

    async def successful_extraction_ids(
        self, run_id: str, extraction_ids: Sequence[str]
    ) -> set[str]:
        if not extraction_ids:
            return set()
        placeholders = ",".join("?" for _ in extraction_ids)
        cursor = await self.connection.execute(
            f"""
            SELECT extraction_id FROM pages
            WHERE run_id = ? AND status = 'success'
              AND extraction_id IN ({placeholders})
            """,
            (run_id, *extraction_ids),
        )
        async with cursor:
            rows = await cursor.fetchall()
        return {cast(str, row["extraction_id"]) for row in rows}

    async def is_successful(self, run_id: str, extraction_id: str) -> bool:
        row = await self._fetchone(
            "SELECT status FROM pages WHERE run_id = ? AND extraction_id = ?",
            (run_id, extraction_id),
        )
        return row is not None and row["status"] == "success"

    async def next_attempt_number(self, run_id: str, extraction_id: str) -> int:
        row = await self._fetchone(
            """
            SELECT COALESCE(MAX(attempt_number), 0) AS maximum
            FROM attempts WHERE run_id = ? AND extraction_id = ?
            """,
            (run_id, extraction_id),
        )
        if row is None:
            raise RuntimeError("attempt query returned no aggregate row")
        return int(row["maximum"]) + 1

    async def attempt_state(self, run_id: str, extraction_id: str) -> tuple[int, str | None]:
        """Return the contiguous attempt count and latest client request identity."""

        row = await self._fetchone(
            """
            SELECT COUNT(*) AS count, COALESCE(MAX(attempt_number), 0) AS maximum,
                   (
                       SELECT json_extract(latest.attempt_json, '$.inference_request_id')
                       FROM attempts latest
                       WHERE latest.run_id = ? AND latest.extraction_id = ?
                       ORDER BY latest.attempt_number DESC LIMIT 1
                   ) AS last_request_id
            FROM attempts WHERE run_id = ? AND extraction_id = ?
            """,
            (run_id, extraction_id, run_id, extraction_id),
        )
        if row is None:
            raise RuntimeError("attempt-state query returned no aggregate row")
        count = int(row["count"])
        if count != int(row["maximum"]):
            raise LedgerConflictError(f"attempt history is noncontiguous for {extraction_id}")
        last_request_id = row["last_request_id"]
        if count == 0:
            if last_request_id is not None:
                raise LedgerConflictError(f"empty attempt history has an ID for {extraction_id}")
            return 0, None
        if not isinstance(last_request_id, str) or not last_request_id:
            raise LedgerConflictError(f"attempt history has no last request ID for {extraction_id}")
        return count, last_request_id

    async def _record_attempt_in_transaction(self, attempt: InferenceAttempt) -> None:
        """Insert one immutable attempt inside the caller's transaction."""

        run_id = attempt.run_id
        extraction_id = attempt.extraction_id
        attempt_number = attempt.attempt_number
        attempt_json = _json_text(attempt.model_dump(mode="json"))
        await self._validate_page_record_contract(attempt)
        existing = await self._fetchone(
            """
            SELECT attempt_json FROM attempts
            WHERE run_id = ? AND extraction_id = ? AND attempt_number = ?
            """,
            (run_id, extraction_id, attempt_number),
        )
        if existing is not None:
            if existing["attempt_json"] != attempt_json:
                raise LedgerConflictError(
                    f"attempt conflict for {extraction_id} attempt {attempt_number}"
                )
            return
        maximum = await self._fetchone(
            """
            SELECT COALESCE(MAX(attempt_number), 0) AS value
            FROM attempts WHERE run_id = ? AND extraction_id = ?
            """,
            (run_id, extraction_id),
        )
        if maximum is None or attempt_number != int(maximum["value"]) + 1:
            raise LedgerConflictError(f"attempt numbers must be contiguous for {extraction_id}")
        page = await self._fetchone(
            "SELECT status FROM pages WHERE run_id = ? AND extraction_id = ?",
            (run_id, extraction_id),
        )
        if page is None:
            raise LedgerConflictError(f"unregistered extraction {extraction_id}")
        if page["status"] == "success":
            raise LedgerConflictError(f"cannot append an attempt to completed {extraction_id}")
        try:
            await self.connection.execute(
                """
                INSERT INTO attempts (
                    run_id, extraction_id, attempt_number, attempt_json, created_at
                ) VALUES (?, ?, ?, ?, ?)
                """,
                (run_id, extraction_id, attempt_number, attempt_json, _utc_now()),
            )
        except aiosqlite.IntegrityError as error:
            row = await self._fetchone(
                """
                SELECT attempt_json FROM attempts
                WHERE run_id = ? AND extraction_id = ? AND attempt_number = ?
                """,
                (run_id, extraction_id, attempt_number),
            )
            if row is None or row["attempt_json"] != attempt_json:
                raise LedgerConflictError(
                    f"attempt conflict for {extraction_id} attempt {attempt_number}"
                ) from error

    async def record_success(
        self,
        *,
        result: PageExtractionRecord,
        attempts: Sequence[InferenceAttempt],
    ) -> None:
        """Atomically commit a successful request sequence and its page result."""

        if not attempts:
            raise ValueError("successful page persistence requires at least one attempt")
        if any(attempt.run_id != result.run_id for attempt in attempts) or any(
            attempt.extraction_id != result.extraction_id for attempt in attempts
        ):
            raise LedgerConflictError("successful attempts must belong to the page result")
        if any(attempt.attempt_outcome != "retryable_error" for attempt in attempts[:-1]):
            raise LedgerConflictError(
                "non-final attempts in a successful request sequence must be retryable errors"
            )
        final_attempt = attempts[-1]
        if final_attempt.attempt_outcome != "success":
            raise LedgerConflictError("final attempt in a successful request sequence must succeed")
        if (
            result.inference_attempt_count != final_attempt.attempt_number
            or result.inference_request_id != final_attempt.inference_request_id
            or result.inference_server_request_id != final_attempt.inference_server_request_id
        ):
            raise LedgerConflictError(
                "successful result identity must match the final inference attempt"
            )

        run_id = result.run_id
        extraction_id = result.extraction_id
        result_json = _json_text(result.model_dump(mode="json"))
        ocr_text_sha256 = result.raw_ocr_text_sha256
        raw_response_sha256 = result.raw_response_sha256
        async with self._write_lock:
            await self.connection.execute("BEGIN IMMEDIATE")
            try:
                run_row = await self._validate_page_record_contract(result)
                self._validate_success_execution_contract(result, run_row)
                row = await self._fetchone(
                    """
                    SELECT status, result_json, ocr_text_sha256, raw_response_sha256,
                           document_id, page_id, page_index, failure_json
                    FROM pages WHERE run_id = ? AND extraction_id = ?
                    """,
                    (run_id, extraction_id),
                )
                if row is None:
                    raise LedgerConflictError(f"unregistered extraction {extraction_id}")
                if row["status"] == "success":
                    if not (
                        row["result_json"] == result_json
                        and row["ocr_text_sha256"] == ocr_text_sha256
                        and row["raw_response_sha256"] == raw_response_sha256
                    ):
                        raise LedgerConflictError(f"conflicting successful OCR for {extraction_id}")
                    for attempt in attempts:
                        existing = await self._fetchone(
                            """
                            SELECT attempt_json FROM attempts
                            WHERE run_id = ? AND extraction_id = ? AND attempt_number = ?
                            """,
                            (run_id, extraction_id, attempt.attempt_number),
                        )
                        if existing is None or existing["attempt_json"] != _json_text(
                            attempt.model_dump(mode="json")
                        ):
                            raise LedgerConflictError(
                                f"completed extraction {extraction_id} has conflicting attempts"
                            )
                    await self.connection.commit()
                    return

                existing_successes = await self._fetchone(
                    """
                    SELECT COUNT(*) AS count FROM attempts
                    WHERE run_id = ? AND extraction_id = ?
                      AND json_extract(attempt_json, '$.attempt_outcome') = 'success'
                    """,
                    (run_id, extraction_id),
                )
                if existing_successes is None:
                    raise RuntimeError("successful-attempt query returned no aggregate row")
                if int(existing_successes["count"]) != 0:
                    try:
                        prior_failure = json.loads(row["failure_json"])
                    except (json.JSONDecodeError, TypeError) as error:
                        raise LedgerConflictError(
                            f"pending extraction {extraction_id} contains "
                            "an orphaned success attempt"
                        ) from error
                    if row["status"] != "failed" or prior_failure.get("failure_stage") != "persist":
                        raise LedgerConflictError(
                            f"pending extraction {extraction_id} contains "
                            "an orphaned success attempt"
                        )
                for attempt in attempts:
                    await self._record_attempt_in_transaction(attempt)
                attempt_summary = await self._fetchone(
                    """
                    SELECT COUNT(*) AS count, MAX(attempt_number) AS maximum,
                           SUM(CASE
                               WHEN json_extract(attempt_json, '$.attempt_outcome') = 'success'
                               THEN 1 ELSE 0 END
                           ) AS successes
                    FROM attempts WHERE run_id = ? AND extraction_id = ?
                    """,
                    (run_id, extraction_id),
                )
                if (
                    attempt_summary is None
                    or int(attempt_summary["count"]) != result.inference_attempt_count
                    or int(attempt_summary["maximum"]) != result.inference_attempt_count
                    or int(attempt_summary["successes"]) < 1
                ):
                    raise LedgerConflictError(
                        f"successful attempt history is incomplete for {extraction_id}"
                    )
                await self.connection.execute(
                    """
                    UPDATE pages SET status = 'success', result_json = ?, failure_json = NULL,
                        ocr_text_sha256 = ?, raw_response_sha256 = ?, updated_at = ?
                    WHERE run_id = ? AND extraction_id = ?
                    """,
                    (
                        result_json,
                        ocr_text_sha256,
                        raw_response_sha256,
                        _utc_now(),
                        run_id,
                        extraction_id,
                    ),
                )
                await self._refresh_document_status(run_id, extraction_id)
            except BaseException:
                await self.connection.rollback()
                raise
            await self.connection.commit()

    async def record_failure(
        self,
        *,
        failure: PageExtractionFailure,
        attempts: Sequence[InferenceAttempt] = (),
    ) -> None:
        """Atomically persist a failed request sequence and its page failure."""

        run_id = failure.run_id
        extraction_id = failure.extraction_id
        if any(attempt.run_id != run_id for attempt in attempts) or any(
            attempt.extraction_id != extraction_id for attempt in attempts
        ):
            raise LedgerConflictError("failure attempts must belong to the failed page")
        if failure.failure_stage in {"inference", "persist"} and not attempts:
            raise LedgerConflictError(
                "inference and persistence failures require their current attempt sequence"
            )
        if failure.failure_stage == "inference" and (
            attempts[-1].attempt_outcome not in {"retryable_error", "terminal_error"}
            or any(attempt.attempt_outcome != "retryable_error" for attempt in attempts[:-1])
        ):
            raise LedgerConflictError(
                "inference failure requires retryable attempts followed by a failed attempt"
            )
        if failure.failure_stage == "persist" and (
            attempts[-1].attempt_outcome != "success"
            or any(attempt.attempt_outcome != "retryable_error" for attempt in attempts[:-1])
        ):
            raise LedgerConflictError(
                "persistence failure requires retryable attempts followed by a success"
            )
        async with self._write_lock:
            await self.connection.execute("BEGIN IMMEDIATE")
            try:
                await self._validate_page_record_contract(failure)
                row = await self._fetchone(
                    """
                    SELECT status, document_id, page_id, page_index
                    FROM pages WHERE run_id = ? AND extraction_id = ?
                    """,
                    (run_id, extraction_id),
                )
                if row is None:
                    raise LedgerConflictError(f"unregistered extraction {extraction_id}")
                if row["status"] == "success":
                    raise LedgerConflictError(
                        f"cannot replace successful OCR {extraction_id} with failure"
                    )
                for attempt in attempts:
                    await self._record_attempt_in_transaction(attempt)
                attempt_state = await self._fetchone(
                    """
                    SELECT COUNT(*) AS count, COALESCE(MAX(attempt_number), 0) AS maximum,
                           (
                               SELECT json_extract(latest.attempt_json, '$.inference_request_id')
                               FROM attempts latest
                               WHERE latest.run_id = ? AND latest.extraction_id = ?
                               ORDER BY latest.attempt_number DESC LIMIT 1
                           ) AS last_request_id,
                           (
                               SELECT json_extract(latest.attempt_json, '$.attempt_outcome')
                               FROM attempts latest
                               WHERE latest.run_id = ? AND latest.extraction_id = ?
                               ORDER BY latest.attempt_number DESC LIMIT 1
                           ) AS last_outcome
                    FROM attempts WHERE run_id = ? AND extraction_id = ?
                    """,
                    (
                        run_id,
                        extraction_id,
                        run_id,
                        extraction_id,
                        run_id,
                        extraction_id,
                    ),
                )
                if (
                    attempt_state is None
                    or int(attempt_state["count"]) != failure.inference_attempt_count
                    or int(attempt_state["maximum"]) != failure.inference_attempt_count
                    or attempt_state["last_request_id"] != failure.last_inference_request_id
                ):
                    raise LedgerConflictError(
                        f"failure attempt history is inconsistent for {extraction_id}"
                    )
                if failure.failure_stage == "inference" and attempt_state["last_outcome"] not in {
                    "retryable_error",
                    "terminal_error",
                }:
                    raise LedgerConflictError(
                        f"inference failure has no failed final attempt for {extraction_id}"
                    )
                if (
                    failure.failure_stage == "persist"
                    and attempt_state["last_outcome"] != "success"
                ):
                    raise LedgerConflictError(
                        f"persistence failure has no successful final attempt for {extraction_id}"
                    )
                await self.connection.execute(
                    """
                    UPDATE pages SET status = 'failed', failure_json = ?, updated_at = ?
                    WHERE run_id = ? AND extraction_id = ?
                    """,
                    (
                        _json_text(failure.model_dump(mode="json")),
                        _utc_now(),
                        run_id,
                        extraction_id,
                    ),
                )
                await self._refresh_document_status(run_id, extraction_id)
            except BaseException:
                await self.connection.rollback()
                raise
            await self.connection.commit()

    async def _refresh_document_status(self, run_id: str, extraction_id: str) -> None:
        page = await self._fetchone(
            "SELECT document_id FROM pages WHERE run_id = ? AND extraction_id = ?",
            (run_id, extraction_id),
        )
        if page is None:
            raise RuntimeError(f"page disappeared while updating {extraction_id}")
        document_id = cast(str, page["document_id"])
        counts = await self._fetchone(
            """
            SELECT d.page_count AS expected,
                   COUNT(p.extraction_id) AS registered,
                   SUM(CASE WHEN p.status = 'success' THEN 1 ELSE 0 END) AS succeeded
            FROM documents d
            LEFT JOIN pages p
              ON p.run_id = d.run_id AND p.document_id = d.document_id
            WHERE d.run_id = ? AND d.document_id = ?
            GROUP BY d.page_count
            """,
            (run_id, document_id),
        )
        if counts is None:
            raise RuntimeError(f"document disappeared while updating {extraction_id}")
        complete = (
            counts["expected"] is not None
            and counts["registered"] == counts["expected"]
            and counts["succeeded"] == counts["expected"]
        )
        await self.connection.execute(
            """
            UPDATE documents SET status = ?, failure_json = NULL, updated_at = ?
            WHERE run_id = ? AND document_id = ?
            """,
            ("complete" if complete else "incomplete", _utc_now(), run_id, document_id),
        )

    async def validation_summary(self, run_id: str) -> dict[str, int]:
        pages = await self._fetchone(
            """
            SELECT COUNT(*) AS registered_pages,
                   SUM(CASE WHEN status = 'success' THEN 1 ELSE 0 END) AS successful_pages,
                   SUM(CASE WHEN status = 'failed' THEN 1 ELSE 0 END) AS failed_pages,
                   SUM(CASE WHEN status = 'pending' THEN 1 ELSE 0 END) AS pending_pages
            FROM pages WHERE run_id = ?
            """,
            (run_id,),
        )
        documents = await self._fetchone(
            """
            SELECT COUNT(*) AS registered_documents,
                   SUM(CASE WHEN page_count IS NOT NULL THEN page_count ELSE 0 END)
                       AS expected_pages,
                   SUM(CASE WHEN status = 'complete' THEN 1 ELSE 0 END)
                       AS complete_documents,
                   SUM(CASE WHEN status = 'failed' THEN 1 ELSE 0 END)
                       AS failed_documents
            FROM documents WHERE run_id = ?
            """,
            (run_id,),
        )
        run = await self._fetchone(
            "SELECT inventory_document_count FROM runs WHERE run_id = ?",
            (run_id,),
        )
        attempts = await self._fetchone(
            "SELECT COUNT(*) AS attempt_rows FROM attempts WHERE run_id = ?",
            (run_id,),
        )
        audited = await self._fetchone(
            """
            SELECT COUNT(*) AS audited_successful_pages
            FROM pages p
            WHERE p.run_id = ? AND p.status = 'success'
              AND json_extract(p.result_json, '$.inference_attempt_count') = (
                  SELECT COUNT(*) FROM attempts a
                  WHERE a.run_id = p.run_id AND a.extraction_id = p.extraction_id
              )
              AND 1 <= (
                  SELECT COUNT(*) FROM attempts a
                  WHERE a.run_id = p.run_id AND a.extraction_id = p.extraction_id
                    AND json_extract(a.attempt_json, '$.attempt_outcome') = 'success'
              )
              AND 'success' = (
                  SELECT json_extract(a.attempt_json, '$.attempt_outcome')
                  FROM attempts a
                  WHERE a.run_id = p.run_id AND a.extraction_id = p.extraction_id
                  ORDER BY a.attempt_number DESC LIMIT 1
              )
              AND json_extract(p.result_json, '$.inference_request_id') = (
                  SELECT json_extract(a.attempt_json, '$.inference_request_id')
                  FROM attempts a
                  WHERE a.run_id = p.run_id AND a.extraction_id = p.extraction_id
                  ORDER BY a.attempt_number DESC LIMIT 1
              )
            """,
            (run_id,),
        )
        if pages is None or documents is None or run is None or attempts is None or audited is None:
            raise LedgerConflictError(f"unknown run_id {run_id!r}")
        combined = {
            "inventory_documents": run["inventory_document_count"],
            **dict(documents),
            **dict(pages),
            **dict(attempts),
            **dict(audited),
        }
        return {key: int(value or 0) for key, value in combined.items()}

    async def run_contract(self, run_id: str) -> dict[str, Any]:
        row = await self._fetchone(
            """
            SELECT run_id, config_sha256, pipeline_fingerprint,
                   inventory_sha256, inventory_document_count
            FROM runs WHERE run_id = ?
            """,
            (run_id,),
        )
        if row is None:
            raise LedgerConflictError(f"unknown run_id {run_id!r}")
        return dict(row)

    async def require_complete(self, run_id: str) -> dict[str, int]:
        summary = await self.validation_summary(run_id)
        complete = (
            summary["inventory_documents"] > 0
            and summary["registered_documents"] == summary["inventory_documents"]
            and summary["failed_documents"] == 0
            and summary["registered_pages"] == summary["expected_pages"]
            and summary["successful_pages"] == summary["expected_pages"]
            and summary["audited_successful_pages"] == summary["successful_pages"]
            and summary["failed_pages"] == 0
            and summary["pending_pages"] == 0
            and summary["complete_documents"] == summary["inventory_documents"]
        )
        if not complete:
            raise IncompleteRunError(f"run {run_id!r} is incomplete: {summary}")
        return summary

    async def iter_success_records(self, run_id: str) -> AsyncIterator[dict[str, Any]]:
        cursor = await self.connection.execute(
            """
            SELECT result_json FROM pages
            WHERE run_id = ? AND status = 'success'
            ORDER BY document_id, page_index
            """,
            (run_id,),
        )
        async with cursor:
            async for row in cursor:
                yield cast(dict[str, Any], json.loads(row["result_json"]))

    async def iter_attempts(self, run_id: str) -> AsyncIterator[dict[str, Any]]:
        cursor = await self.connection.execute(
            """
            SELECT a.attempt_json
            FROM attempts a
            JOIN pages p
              ON p.run_id = a.run_id AND p.extraction_id = a.extraction_id
            WHERE a.run_id = ?
            ORDER BY p.document_id, p.page_index, a.attempt_number
            """,
            (run_id,),
        )
        async with cursor:
            async for row in cursor:
                yield cast(dict[str, Any], json.loads(row["attempt_json"]))

    async def iter_page_failures(self, run_id: str) -> AsyncIterator[dict[str, Any]]:
        cursor = await self.connection.execute(
            """
            SELECT failure_json FROM pages
            WHERE run_id = ? AND status = 'failed'
            ORDER BY document_id, page_index
            """,
            (run_id,),
        )
        async with cursor:
            async for row in cursor:
                yield cast(dict[str, Any], json.loads(row["failure_json"]))

    async def iter_document_failures(self, run_id: str) -> AsyncIterator[dict[str, Any]]:
        cursor = await self.connection.execute(
            """
            SELECT failure_json FROM documents
            WHERE run_id = ? AND status = 'failed'
            ORDER BY document_id
            """,
            (run_id,),
        )
        async with cursor:
            async for row in cursor:
                yield cast(dict[str, Any], json.loads(row["failure_json"]))

    async def mark_failed(self, run_id: str) -> None:
        async with self._write_lock:
            await self.connection.execute(
                "UPDATE runs SET status = 'failed', updated_at = ? WHERE run_id = ?",
                (_utc_now(), run_id),
            )
            await self.connection.commit()

    async def mark_complete(
        self,
        *,
        run_id: str,
        manifest_sha256: str,
        manifest: Mapping[str, Any],
    ) -> None:
        """Bind a complete ledger to the already-published immutable manifest."""

        summary = await self.require_complete(run_id)
        try:
            typed_manifest = DatasetManifest.model_validate_json(
                canonical_json_bytes(manifest), strict=True
            )
        except Exception as error:
            raise LedgerConflictError(
                "dataset manifest does not satisfy its typed contract"
            ) from error
        manifest_payload = typed_manifest.model_dump(mode="json")
        if manifest_payload["summary"] != summary:
            raise LedgerConflictError("manifest summary does not match the complete ledger")
        manifest_bytes = canonical_json_bytes(manifest_payload) + b"\n"
        manifest_json = manifest_bytes[:-1].decode("utf-8")
        actual_manifest_sha256 = sha256_bytes(manifest_bytes)
        if actual_manifest_sha256 != manifest_sha256:
            raise LedgerConflictError("manifest_sha256 does not match canonical manifest bytes")
        contract = await self.run_contract(run_id)
        expected_manifest_identity = {
            "run_id": contract["run_id"],
            "config_sha256": contract["config_sha256"],
            "pipeline_fingerprint": contract["pipeline_fingerprint"],
            "inventory_sha256": contract["inventory_sha256"],
        }
        for key, expected in expected_manifest_identity.items():
            if manifest_payload.get(key) != expected:
                raise LedgerConflictError(
                    f"manifest {key} does not match the initialized run contract"
                )
        run_root = self.path.parent
        on_disk_manifest = run_root / "dataset" / "manifest.json"
        manifest_size, on_disk_manifest_sha256 = await asyncio.to_thread(
            _hash_regular_artifact,
            on_disk_manifest,
            allowed_root=run_root,
        )
        if manifest_size != len(manifest_bytes) or on_disk_manifest_sha256 != manifest_sha256:
            raise LedgerConflictError(
                "published dataset manifest bytes do not match the completion request"
            )
        for artifact in typed_manifest.files:
            artifact_path = run_root / artifact.path
            artifact_size, artifact_sha256 = await asyncio.to_thread(
                _hash_regular_artifact,
                artifact_path,
                allowed_root=run_root,
            )
            if artifact_size != artifact.bytes or artifact_sha256 != artifact.sha256:
                raise LedgerConflictError(
                    f"published dataset artifact does not match its manifest: {artifact.path}"
                )
        async with self._write_lock:
            row = await self._fetchone(
                "SELECT status, manifest_sha256, manifest_json FROM runs WHERE run_id = ?",
                (run_id,),
            )
            if row is None:
                raise LedgerConflictError(f"unknown run_id {run_id!r}")
            if row["status"] == "complete":
                if (
                    row["manifest_sha256"] == manifest_sha256
                    and row["manifest_json"] == manifest_json
                ):
                    return
                raise LedgerConflictError(f"run {run_id!r} has a conflicting manifest")
            await self.connection.execute(
                """
                UPDATE runs SET status = 'complete', manifest_sha256 = ?,
                    manifest_json = ?, updated_at = ? WHERE run_id = ?
                """,
                (manifest_sha256, manifest_json, _utc_now(), run_id),
            )
            await self.connection.commit()
