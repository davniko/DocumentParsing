"""Resumable table recognition over page rasters frozen by validated labels."""

from __future__ import annotations

import asyncio
import json
import math
import os
import sqlite3
import time
from collections import Counter, defaultdict
from collections.abc import AsyncIterator, Callable, Sequence
from contextlib import asynccontextmanager
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from statistics import median
from typing import Annotated, Any, Literal, Protocol, cast

import pyarrow.parquet as pq
from pydantic import BaseModel, ConfigDict, Field, model_validator

from document_ocr.atomic import (
    atomic_publish_bytes,
    atomic_publish_json,
    read_regular_file_bytes,
)
from document_ocr.benchmark import (
    RasterManifestEntry,
    inspect_raster,
    validate_current_raster,
)
from document_ocr.client import OcrResponse, ServerInfo, VllmClientError, VllmOcrClient
from document_ocr.config import PipelineConfig
from document_ocr.exporter import arrow_schema
from document_ocr.hashing import (
    canonical_json_bytes,
    canonical_json_sha256,
    sha256_bytes,
    sha256_file,
    stable_id,
)
from document_ocr.label_schemas.bill_of_lading import BillOfLadingAnnotation
from document_ocr.label_schemas.bill_of_lading_v3 import (
    BillOfLadingDualCargoAnnotation,
)
from document_ocr.models import (
    DatasetManifest,
    InferenceAttempt,
    PageExtractionFailure,
    PageExtractionRecord,
    SourceObject,
)
from document_ocr.sources import load_source_inventory
from document_ocr.table_views.config import (
    RawExtractionRasterSourceConfig,
    RawExtractionRunConfig,
    TableConcurrencyConfig,
    TableViewConfig,
    ValidatedLabelRasterSourceConfig,
)
from document_ocr.table_views.models import (
    TableAttemptRecord,
    TableFailureRecord,
    TableInputPage,
    TablePageCommit,
    TablePageResult,
)


class TableViewError(RuntimeError):
    """The table-view run violated its immutable input or output contract."""


class IncompleteTableRunError(TableViewError):
    """At least one selected page lacks a valid successful commit."""


class _TableClient(Protocol):
    async def check_readiness(self) -> ServerInfo: ...

    async def recognize_page(
        self,
        raster_path: Path,
        *,
        mime_type: Literal["image/png", "image/jpeg"],
        raster_sha256: str,
        request_id: str,
        first_attempt_number: int = 1,
        max_attempts: int | None = None,
    ) -> OcrResponse: ...


class _BasePageLedgerProjection(BaseModel):
    """Exact v1/v2 ledger fields needed to bind an auxiliary inference view."""

    model_config = ConfigDict(
        extra="allow",
        frozen=True,
        strict=True,
        validate_default=True,
    )

    schema_version: Literal[1, 2]
    run_id: str
    extraction_id: str
    document_id: str
    document_page_count: Annotated[int, Field(gt=0)]
    page_index: Annotated[int, Field(ge=0)]
    page_number: Annotated[int, Field(gt=0)]
    page_id: str
    raw_ocr_text: str
    raw_ocr_text_sha256: str
    source_sha256: str
    raster_path: str
    raster_mime_type: Literal["image/png", "image/jpeg"]
    raster_size_bytes: Annotated[int, Field(gt=0)]
    raster_sha256: str

    @model_validator(mode="after")
    def identity_and_hashes_are_consistent(self) -> _BasePageLedgerProjection:
        if self.page_number != self.page_index + 1:
            raise ValueError("page_number must equal page_index + 1")
        if self.page_number > self.document_page_count:
            raise ValueError("page_number must not exceed document_page_count")
        if sha256_bytes(self.raw_ocr_text.encode("utf-8")) != self.raw_ocr_text_sha256:
            raise ValueError("raw_ocr_text_sha256 does not match raw_ocr_text")
        for field_name, value in (
            ("raw_ocr_text_sha256", self.raw_ocr_text_sha256),
            ("source_sha256", self.source_sha256),
            ("raster_sha256", self.raster_sha256),
        ):
            if len(value) != 64 or any(character not in "0123456789abcdef" for character in value):
                raise ValueError(f"{field_name} must be a lowercase SHA-256")
        raster_path = PurePosixPath(self.raster_path)
        if (
            raster_path.is_absolute()
            or any(part in {"", ".", ".."} for part in raster_path.parts)
            or "\\" in self.raster_path
        ):
            raise ValueError("raster_path must be a safe relative POSIX path")
        return self


@dataclass(frozen=True, slots=True)
class TableRunPaths:
    output_root: Path
    run_root: Path
    input_pages: Path
    input_pages_digest: Path
    config_snapshot: Path
    provenance: Path
    state_root: Path
    dataset_root: Path


@dataclass(frozen=True, slots=True)
class TableProgress:
    total_pages: int
    processed_pages: int
    successful_pages: int
    failed_pages_this_invocation: int
    elapsed_seconds: float
    throughput_pages_per_second: float
    eta_seconds: float | None

    def __post_init__(self) -> None:
        if self.total_pages <= 0:
            raise ValueError("table progress total_pages must be positive")
        if not 0 <= self.processed_pages <= self.total_pages:
            raise ValueError("table progress processed_pages is outside the page total")
        if not 0 <= self.successful_pages <= self.processed_pages:
            raise ValueError("table progress successful_pages is outside the processed total")
        if self.failed_pages_this_invocation < 0:
            raise ValueError("table progress invocation failures cannot be negative")
        if self.successful_pages + self.failed_pages_this_invocation != self.processed_pages:
            raise ValueError("processed table pages must equal successes plus failures")
        for name, value in (
            ("elapsed_seconds", self.elapsed_seconds),
            ("throughput_pages_per_second", self.throughput_pages_per_second),
        ):
            if not math.isfinite(value) or value < 0.0:
                raise ValueError(f"table progress {name} must be finite and non-negative")
        if self.eta_seconds is not None and (
            not math.isfinite(self.eta_seconds) or self.eta_seconds < 0.0
        ):
            raise ValueError("table progress eta_seconds must be finite and non-negative")

    @property
    def remaining_pages(self) -> int:
        return self.total_pages - self.processed_pages


TableProgressCallback = Callable[[TableProgress], None]


def _canonical_directory(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    absolute = Path(os.path.abspath(path))
    resolved = path.resolve(strict=True)
    if resolved != absolute:
        raise TableViewError(f"table-view directories must not traverse symlinks: {path}")
    return resolved


def table_run_paths(config: TableViewConfig) -> TableRunPaths:
    output_root = _canonical_directory(Path(config.output.root))
    run_root = _canonical_directory(output_root / "runs" / config.run.run_id)
    state_root = _canonical_directory(run_root / "state")
    dataset_root = _canonical_directory(run_root / "dataset")
    return TableRunPaths(
        output_root=output_root,
        run_root=run_root,
        input_pages=run_root / "input-pages.jsonl",
        input_pages_digest=run_root / "input-pages.jsonl.sha256",
        config_snapshot=run_root / "resolved-config.json",
        provenance=run_root / "run-provenance.json",
        state_root=state_root,
        dataset_root=dataset_root,
    )


def _strict_json(payload: bytes, *, context: str) -> Any:
    def reject_duplicates(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        value: dict[str, Any] = {}
        for key, item in pairs:
            if key in value:
                raise ValueError(f"duplicate key {key!r}")
            value[key] = item
        return value

    def reject_constant(value: str) -> None:
        raise ValueError(f"non-finite number {value!r}")

    try:
        return json.loads(
            payload.decode("utf-8"),
            object_pairs_hook=reject_duplicates,
            parse_constant=reject_constant,
        )
    except (UnicodeError, ValueError, TypeError) as error:
        raise TableViewError(f"invalid JSON in {context}") from error


def _validated_bill_of_lading_annotation(
    payload: bytes, *, document_id: str
) -> BillOfLadingAnnotation | BillOfLadingDualCargoAnnotation:
    """Parse only annotation contracts whose training target is unambiguous."""

    value = _strict_json(payload, context=f"validated annotation {document_id}")
    if not isinstance(value, dict):
        raise TableViewError(f"annotation must be a JSON object: {document_id}")
    schema_version = value.get("annotationSchemaVersion")
    annotation_type: (
        type[BillOfLadingAnnotation] | type[BillOfLadingDualCargoAnnotation]
    )
    if schema_version == "2.0.0":
        annotation_type = BillOfLadingAnnotation
    elif schema_version == "3.0.0-experimental":
        annotation_type = BillOfLadingDualCargoAnnotation
    else:
        raise TableViewError(
            f"unsupported annotation schema version for {document_id}: "
            f"{schema_version!r}"
        )
    try:
        # JSON arrays are the canonical representation of tuple fields. Parse the
        # duplicate-checked value through Pydantic's JSON path to retain strict
        # scalar validation without incorrectly rejecting those arrays.
        return annotation_type.model_validate_json(
            canonical_json_bytes(value), strict=True
        )
    except ValueError as error:
        raise TableViewError(
            f"annotation schema validation failed: {document_id}"
        ) from error


def _annotation_training_target(
    annotation: BillOfLadingAnnotation | BillOfLadingDualCargoAnnotation,
) -> dict[str, Any]:
    if isinstance(annotation, BillOfLadingAnnotation):
        return annotation.label.canonical_target()
    return annotation.relationExplicitLabel.canonical_target()


def _canonical_regular_file(path: Path, *, root: Path, context: str) -> Path:
    absolute = Path(os.path.abspath(path))
    try:
        resolved_root = root.resolve(strict=True)
        resolved = path.resolve(strict=True)
    except OSError as error:
        raise TableViewError(f"missing {context}: {path}") from error
    if (
        resolved_root != Path(os.path.abspath(root))
        or resolved != absolute
        or not resolved.is_relative_to(resolved_root)
        or not resolved.is_file()
        or resolved.is_symlink()
    ):
        raise TableViewError(f"{context} is not a canonical contained regular file: {path}")
    return resolved


def _relative_artifact(value: str, *, context: str) -> PurePosixPath:
    path = PurePosixPath(value)
    if path.is_absolute() or any(part in {"", ".", ".."} for part in path.parts):
        raise TableViewError(f"{context} must be a safe relative POSIX path")
    return path


def _read_jsonl(path: Path, *, expected_sha256: str, expected_rows: int) -> list[dict[str, Any]]:
    payload = read_regular_file_bytes(path)
    if sha256_bytes(payload) != expected_sha256:
        raise TableViewError(f"JSONL SHA-256 mismatch: {path}")
    rows: list[dict[str, Any]] = []
    for line_number, line in enumerate(payload.splitlines(), start=1):
        if not line:
            raise TableViewError(f"blank JSONL line at {path}:{line_number}")
        value = _strict_json(line, context=f"{path}:{line_number}")
        if not isinstance(value, dict):
            raise TableViewError(f"JSONL row is not an object at {path}:{line_number}")
        rows.append(cast(dict[str, Any], value))
    if len(rows) != expected_rows:
        raise TableViewError(
            f"JSONL row-count mismatch for {path}: expected {expected_rows}, found {len(rows)}"
        )
    return rows


def _load_successful_base_pages(
    *,
    run_root: Path,
    run_id: str,
) -> dict[str, _BasePageLedgerProjection]:
    """Load one completed OCR run in a single read-only ledger scan."""

    ledger_path = _canonical_regular_file(
        run_root / "state.sqlite3", root=run_root, context="source extraction ledger"
    )
    rows: dict[str, _BasePageLedgerProjection] = {}
    try:
        connection = sqlite3.connect(f"file:{ledger_path}?mode=ro", uri=True, timeout=30.0)
        cursor = connection.execute(
            """
            SELECT extraction_id, result_json
            FROM pages
            WHERE run_id = ? AND status = 'success'
            ORDER BY extraction_id
            """,
            (run_id,),
        )
        for extraction_id, result_json in cursor:
            if not isinstance(extraction_id, str) or not isinstance(result_json, str):
                raise TableViewError("source extraction ledger returned invalid column types")
            if extraction_id in rows:
                raise TableViewError(
                    f"source extraction ledger contains duplicate identity: {extraction_id}"
                )
            try:
                result = _BasePageLedgerProjection.model_validate_json(result_json, strict=True)
            except ValueError as error:
                raise TableViewError(
                    f"source page projection is invalid: {extraction_id}: {error}"
                ) from error
            if result.run_id != run_id or result.extraction_id != extraction_id:
                raise TableViewError("source ledger query returned a conflicting page identity")
            rows[extraction_id] = result
    except sqlite3.Error as error:
        raise TableViewError("source extraction ledger query failed") from error
    finally:
        if "connection" in locals():
            connection.close()
    if not rows:
        raise TableViewError(f"source extraction run has no successful pages: {run_id}")
    return rows


@dataclass(frozen=True, slots=True)
class _RawExtractionRunIdentity:
    root: Path
    ledger: Path
    provenance: Path
    provenance_sha256: str
    inventory: dict[str, SourceObject]
    pipeline_config: PipelineConfig
    status: Literal["failed", "complete"]
    manifest: DatasetManifest | None


def _verify_complete_raw_manifest(
    *,
    source: RawExtractionRunConfig,
    run_root: Path,
    ledger_manifest_sha256: str | None,
    ledger_manifest_json: str | None,
) -> DatasetManifest:
    manifest_path = _canonical_regular_file(
        run_root / "dataset" / "manifest.json",
        root=run_root,
        context="published raw OCR dataset manifest",
    )
    manifest_bytes = read_regular_file_bytes(manifest_path)
    actual_manifest_sha256 = sha256_bytes(manifest_bytes)
    if source.manifest_sha256 is not None and actual_manifest_sha256 != source.manifest_sha256:
        raise TableViewError(f"raw OCR manifest SHA-256 mismatch: {source.run_id}")
    try:
        manifest = DatasetManifest.model_validate_json(manifest_bytes, strict=True)
    except ValueError as error:
        raise TableViewError(f"raw OCR manifest is invalid: {source.run_id}") from error
    canonical_manifest = canonical_json_bytes(manifest.model_dump(mode="json"))
    if manifest_bytes != canonical_manifest + b"\n":
        raise TableViewError(
            f"raw OCR manifest is not canonical manifest-last output: {source.run_id}"
        )
    if (
        manifest.run_id != source.run_id
        or manifest.config_sha256 != source.config_sha256
        or manifest.pipeline_fingerprint != source.pipeline_fingerprint
        or manifest.inventory_sha256 != source.inventory_sha256
        or ledger_manifest_sha256 != actual_manifest_sha256
        or ledger_manifest_json != canonical_manifest.decode("utf-8")
        or manifest.summary.inventory_documents != source.expected_documents
        or manifest.summary.registered_documents != source.expected_documents
        or manifest.summary.complete_documents != source.expected_documents
        or manifest.summary.failed_documents != 0
        or manifest.summary.expected_pages != source.expected_pages
        or manifest.summary.registered_pages != source.expected_pages
        or manifest.summary.successful_pages != source.expected_pages
        or manifest.summary.failed_pages != 0
        or manifest.summary.pending_pages != 0
        or manifest.summary.audited_successful_pages != source.expected_pages
        or not manifest.page_images.retained
    ):
        raise TableViewError(
            f"complete raw OCR manifest conflicts with its run contract: {source.run_id}"
        )
    artifacts = {artifact.record_model: artifact for artifact in manifest.files}
    if set(artifacts) != {"PageExtractionRecord", "InferenceAttempt"}:
        raise TableViewError(f"raw OCR manifest artifact roles differ: {source.run_id}")
    _verified_published_parquet(
        run_root=run_root,
        artifact=artifacts["PageExtractionRecord"],
        record_model=PageExtractionRecord,
    )
    _verified_published_parquet(
        run_root=run_root,
        artifact=artifacts["InferenceAttempt"],
        record_model=InferenceAttempt,
    )
    return manifest


def _verified_published_parquet(
    *,
    run_root: Path,
    artifact: Any,
    record_model: type[PageExtractionRecord] | type[InferenceAttempt],
) -> Path:
    path = _canonical_regular_file(
        run_root / Path(artifact.path),
        root=run_root,
        context=f"published raw OCR {artifact.record_model} artifact",
    )
    if path.stat().st_size != artifact.bytes or sha256_file(path) != artifact.sha256:
        raise TableViewError(
            f"published raw OCR artifact differs from its manifest: {artifact.path}"
        )
    try:
        parquet = pq.ParquetFile(path)
    except Exception as error:
        raise TableViewError(f"published raw OCR artifact is not Parquet: {path}") from error
    expected_schema = arrow_schema(record_model)
    if parquet.metadata.num_rows != artifact.rows:
        raise TableViewError(f"published raw OCR Parquet row count differs from manifest: {path}")
    if not parquet.schema_arrow.equals(expected_schema, check_metadata=True):
        raise TableViewError(f"published raw OCR Parquet schema differs: {path}")
    if sha256_bytes(parquet.schema_arrow.serialize().to_pybytes()) != (
        artifact.arrow_schema_sha256
    ):
        raise TableViewError(f"published raw OCR Parquet schema hash differs from manifest: {path}")
    return path


def _load_raw_extraction_run_identity(
    source: RawExtractionRunConfig,
) -> _RawExtractionRunIdentity:
    run_root = Path(source.root)
    try:
        resolved_root = run_root.resolve(strict=True)
    except OSError as error:
        raise TableViewError(f"missing raw extraction run root: {run_root}") from error
    if resolved_root != Path(os.path.abspath(run_root)) or not resolved_root.is_dir():
        raise TableViewError(f"raw extraction run root must be a canonical directory: {run_root}")
    ledger_path = _canonical_regular_file(
        resolved_root / "state.sqlite3",
        root=resolved_root,
        context="raw extraction run ledger",
    )
    try:
        connection = sqlite3.connect(f"file:{ledger_path}?mode=ro", uri=True, timeout=30.0)
        rows = connection.execute(
            """
            SELECT config_sha256, pipeline_fingerprint, config_json,
                   inventory_sha256, inventory_document_count, provenance_json,
                   status, manifest_sha256, manifest_json
            FROM runs
            WHERE run_id = ?
            """,
            (source.run_id,),
        ).fetchall()
    except sqlite3.Error as error:
        raise TableViewError(
            f"raw extraction run contract query failed: {source.run_id}"
        ) from error
    finally:
        if "connection" in locals():
            connection.close()
    if len(rows) != 1:
        raise TableViewError(f"raw extraction ledger has no unique run row: {source.run_id}")
    (
        config_sha256,
        pipeline_fingerprint,
        config_json,
        inventory_sha256,
        inventory_document_count,
        provenance_json,
        status,
        manifest_sha256,
        manifest_json,
    ) = rows[0]
    if status not in {"failed", "complete"}:
        raise TableViewError(
            f"raw extraction run must be quiescent (failed or complete): {source.run_id}"
        )
    if (
        config_sha256 != source.config_sha256
        or pipeline_fingerprint != source.pipeline_fingerprint
        or inventory_sha256 != source.inventory_sha256
        or inventory_document_count != source.expected_documents
    ):
        raise TableViewError(
            f"raw extraction ledger identity differs from configuration: {source.run_id}"
        )
    if not isinstance(config_json, str) or not isinstance(provenance_json, str):
        raise TableViewError(f"raw extraction ledger contract is corrupt: {source.run_id}")
    try:
        pipeline_config = PipelineConfig.model_validate_json(config_json, strict=True)
    except ValueError as error:
        raise TableViewError(f"raw extraction ledger config is invalid: {source.run_id}") from error
    canonical_config = canonical_json_bytes(pipeline_config.model_dump(mode="json"))
    if (
        config_json != canonical_config.decode("utf-8")
        or sha256_bytes(canonical_config) != source.config_sha256
        or pipeline_config.run.run_id != source.run_id
        or not pipeline_config.output.retain_page_images
    ):
        raise TableViewError(
            f"raw extraction config does not retain exact page rasters: {source.run_id}"
        )
    expected_root = Path(pipeline_config.output.root) / "runs" / source.run_id
    try:
        resolved_expected_root = expected_root.resolve(strict=True)
    except OSError as error:
        raise TableViewError(f"raw extraction output root is missing: {source.run_id}") from error
    if resolved_expected_root != resolved_root:
        raise TableViewError(f"raw extraction root differs from its stored config: {source.run_id}")

    provenance_path = _canonical_regular_file(
        resolved_root / "run-provenance.json",
        root=resolved_root,
        context="raw extraction run provenance",
    )
    provenance_bytes = read_regular_file_bytes(provenance_path)
    actual_provenance_sha256 = sha256_bytes(provenance_bytes)
    if (
        source.provenance_sha256 is not None
        and actual_provenance_sha256 != source.provenance_sha256
    ):
        raise TableViewError(f"raw extraction provenance SHA-256 mismatch: {source.run_id}")
    provenance = _strict_json(provenance_bytes, context=str(provenance_path))
    canonical_provenance = canonical_json_bytes(provenance)
    if (
        provenance_bytes != canonical_provenance + b"\n"
        or provenance_json != canonical_provenance.decode("utf-8")
        or provenance.get("resolved_config_sha256") != source.config_sha256
        or provenance.get("pipeline_fingerprint") != source.pipeline_fingerprint
        or provenance.get("inventory_sha256") != source.inventory_sha256
        or provenance.get("inventory_document_count") != source.expected_documents
    ):
        raise TableViewError(
            f"raw extraction provenance conflicts with its run contract: {source.run_id}"
        )

    try:
        inventory = load_source_inventory(resolved_root / "inventory.jsonl")
    except Exception as error:
        raise TableViewError(f"raw extraction inventory is invalid: {source.run_id}") from error
    inventory_by_document = {row.document_id: row for row in inventory.objects}
    if (
        inventory.sha256 != source.inventory_sha256
        or len(inventory.objects) != source.expected_documents
        or len(inventory_by_document) != len(inventory.objects)
    ):
        raise TableViewError(
            f"raw extraction inventory conflicts with its run contract: {source.run_id}"
        )

    manifest_path = resolved_root / "dataset" / "manifest.json"
    manifest: DatasetManifest | None
    if status == "complete":
        manifest = _verify_complete_raw_manifest(
            source=source,
            run_root=resolved_root,
            ledger_manifest_sha256=cast(str | None, manifest_sha256),
            ledger_manifest_json=cast(str | None, manifest_json),
        )
    else:
        if source.manifest_sha256 is not None:
            raise TableViewError(
                f"failed raw extraction run must set manifest_sha256 to null: {source.run_id}"
            )
        if manifest_path.exists() or manifest_path.is_symlink():
            raise TableViewError(
                f"failed raw extraction run has an unbound dataset manifest: {source.run_id}"
            )
        if manifest_sha256 is not None or manifest_json is not None:
            raise TableViewError(
                f"failed raw extraction ledger has unexpected manifest metadata: {source.run_id}"
            )
        manifest = None

    return _RawExtractionRunIdentity(
        root=resolved_root,
        ledger=ledger_path,
        provenance=provenance_path,
        provenance_sha256=actual_provenance_sha256,
        inventory=inventory_by_document,
        pipeline_config=pipeline_config,
        status=cast(Literal["failed", "complete"], status),
        manifest=manifest,
    )


def _input_identity(config: TableViewConfig) -> str:
    contract = {
        "schema_version": 1,
        "view_type": config.view_type,
        "model": config.vllm.model,
        "model_revision": config.vllm.revision,
        "prompt": config.vllm.prompt,
        "sampling": config.vllm.sampling.model_dump(mode="json"),
        "speculative_decoding": config.vllm.speculative_decoding.model_dump(mode="json"),
        "repetition_detection": config.vllm.repetition_detection.model_dump(mode="json"),
    }
    return canonical_json_sha256(contract)


def _prepare_validated_label_input_pages(
    config: TableViewConfig,
    source: ValidatedLabelRasterSourceConfig,
) -> tuple[TableInputPage, ...]:
    extraction_roots = {
        row.run_id: Path(row.root).resolve(strict=True) for row in source.extraction_runs
    }
    base_pages_by_run = {
        run_id: _load_successful_base_pages(run_root=root, run_id=run_id)
        for run_id, root in extraction_roots.items()
    }
    contract_sha256 = _input_identity(config)
    pages: list[TableInputPage] = []
    document_ids: set[str] = set()
    page_ids: set[str] = set()
    extraction_ids: set[str] = set()
    raster_paths: set[Path] = set()

    for record_set in source.record_sets:
        root = Path(record_set.root).resolve(strict=True)
        records_path = _canonical_regular_file(
            root / Path(record_set.records_path), root=root, context="label records"
        )
        rows = _read_jsonl(
            records_path,
            expected_sha256=record_set.records_sha256,
            expected_rows=record_set.records,
        )
        for row in rows:
            document_id = row.get("documentId")
            if not isinstance(document_id, str) or document_id in document_ids:
                raise TableViewError("label records contain an invalid or duplicate documentId")
            document_ids.add(document_id)
            annotation_relative = row.get("validatedAnnotationPath")
            annotation_sha256 = row.get("validatedAnnotationSha256")
            joined_text_sha256 = row.get("joinedRawTextSha256")
            target = row.get("target")
            if not all(
                isinstance(value, str)
                for value in (annotation_relative, annotation_sha256, joined_text_sha256)
            ) or not isinstance(target, dict):
                raise TableViewError(f"label record has invalid provenance: {document_id}")
            relative = _relative_artifact(
                cast(str, annotation_relative), context="validatedAnnotationPath"
            )
            annotation_path = _canonical_regular_file(
                root / Path(relative), root=root, context="validated annotation"
            )
            annotation_bytes = read_regular_file_bytes(annotation_path)
            if sha256_bytes(annotation_bytes) != annotation_sha256:
                raise TableViewError(f"annotation SHA-256 mismatch: {document_id}")
            annotation = _validated_bill_of_lading_annotation(
                annotation_bytes, document_id=document_id
            )
            if (
                annotation.source.documentId != document_id
                or annotation.source.joinedRawTextSha256 != joined_text_sha256
                or _annotation_training_target(annotation) != target
                or annotation.reviewStatus != "validated"
            ):
                raise TableViewError(f"label record and annotation disagree: {document_id}")
            run_id = annotation.source.extractionRunId
            run_root = extraction_roots.get(run_id)
            if run_root is None:
                raise TableViewError(f"annotation references an unknown extraction run: {run_id}")

            for source_page in annotation.source.pages:
                base = base_pages_by_run[run_id].get(source_page.extractionId)
                if base is None:
                    raise TableViewError(
                        f"source page is not a successful ledger row: {source_page.extractionId}"
                    )
                if (
                    base.document_id != document_id
                    or base.document_page_count != annotation.source.documentPageCount
                    or base.page_index != source_page.pageIndex
                    or base.page_number != source_page.pageNumber
                    or base.page_id != source_page.pageId
                    or base.extraction_id != source_page.extractionId
                    or base.raw_ocr_text_sha256 != source_page.rawOcrTextSha256
                    or base.source_sha256 != annotation.source.sourceSha256
                    or base.raster_path != source_page.rasterPath
                    or base.raster_sha256 != source_page.rasterSha256
                ):
                    raise TableViewError(f"annotation and base page disagree: {source_page.pageId}")
                raster_relative = _relative_artifact(source_page.rasterPath, context="rasterPath")
                raster_path = _canonical_regular_file(
                    run_root / Path(raster_relative), root=run_root, context="retained raster"
                )
                entry = RasterManifestEntry.model_validate(
                    {
                        "page_id": source_page.pageId,
                        "document_id": document_id,
                        "raster_path": str(raster_path),
                        "mime_type": base.raster_mime_type,
                        "raster_sha256": source_page.rasterSha256,
                    },
                    strict=True,
                )
                identity = inspect_raster(entry)
                if identity.size_bytes != base.raster_size_bytes:
                    raise TableViewError(
                        f"source raster size differs from ledger metadata: {source_page.pageId}"
                    )
                table_view_id = stable_id(
                    "table",
                    "glm-ocr-table-page-v1",
                    source_page.pageId,
                    source_page.rasterSha256,
                    contract_sha256,
                )
                if source_page.pageId in page_ids or source_page.extractionId in extraction_ids:
                    raise TableViewError("duplicate page or extraction identity in selected inputs")
                if identity.canonical_path in raster_paths:
                    raise TableViewError("duplicate canonical raster in selected inputs")
                page_ids.add(source_page.pageId)
                extraction_ids.add(source_page.extractionId)
                raster_paths.add(identity.canonical_path)
                pages.append(
                    TableInputPage.model_validate(
                        {
                            "schemaVersion": 1,
                            "recordSetId": record_set.id,
                            "annotationPath": str(annotation_path),
                            "annotationSha256": annotation_sha256,
                            "documentId": document_id,
                            "documentPageCount": annotation.source.documentPageCount,
                            "pageIndex": source_page.pageIndex,
                            "pageNumber": source_page.pageNumber,
                            "pageId": source_page.pageId,
                            "baseExtractionId": source_page.extractionId,
                            "baseExtractionRunId": run_id,
                            "baseRawOcrTextSha256": source_page.rawOcrTextSha256,
                            "sourceSha256": annotation.source.sourceSha256,
                            "rasterPath": str(raster_path),
                            "rasterMimeType": base.raster_mime_type,
                            "rasterSizeBytes": identity.size_bytes,
                            "rasterSha256": source_page.rasterSha256,
                            "tableViewId": table_view_id,
                        },
                        strict=True,
                    )
                )

    if len(document_ids) != source.expected_documents:
        raise TableViewError("prepared document count does not match expected_documents")
    if len(pages) != source.expected_pages:
        raise TableViewError("prepared page count does not match expected_pages")
    by_document: dict[str, list[TableInputPage]] = defaultdict(list)
    for page in pages:
        by_document[page.documentId].append(page)
    for document_id, document_pages in by_document.items():
        indexes = [row.pageIndex for row in document_pages]
        if indexes != list(range(document_pages[0].documentPageCount)):
            raise TableViewError(f"selected pages are not complete and ordered: {document_id}")
    return tuple(pages)


def _record_source(record: PageExtractionRecord | PageExtractionFailure) -> dict[str, Any]:
    return {field_name: getattr(record, field_name) for field_name in SourceObject.model_fields}


def _failed_page_raster(
    *,
    identity: _RawExtractionRunIdentity,
    failure: PageExtractionFailure,
) -> tuple[Path, Literal["image/png", "image/jpeg"], str]:
    directory = identity.root / "page-images" / failure.document_id / failure.page_id
    try:
        resolved_directory = directory.resolve(strict=True)
    except OSError as error:
        raise TableViewError(
            f"failed raw OCR page has no retained raster: {failure.page_id}"
        ) from error
    if (
        resolved_directory != Path(os.path.abspath(directory))
        or not resolved_directory.is_dir()
        or resolved_directory.is_symlink()
    ):
        raise TableViewError(
            f"failed raw OCR page raster directory is not canonical: {failure.page_id}"
        )
    entries = list(resolved_directory.iterdir())
    if len(entries) != 1:
        raise TableViewError(
            "failed raw OCR page must have exactly one retained raster: "
            f"{failure.page_id}; found {len(entries)}"
        )
    suffix = ".png" if identity.pipeline_config.raster.image_format == "png" else ".jpg"
    mime_type: Literal["image/png", "image/jpeg"] = (
        "image/png" if suffix == ".png" else "image/jpeg"
    )
    raster = _canonical_regular_file(
        entries[0], root=identity.root, context="failed raw OCR retained raster"
    )
    digest = raster.stem
    if (
        raster.parent != resolved_directory
        or raster.suffix != suffix
        or len(digest) != 64
        or any(character not in "0123456789abcdef" for character in digest)
    ):
        raise TableViewError(
            f"failed raw OCR retained raster path is not content-addressed: {failure.page_id}"
        )
    return raster, mime_type, digest


def _prepare_raw_extraction_input_pages(
    config: TableViewConfig,
    source: RawExtractionRasterSourceConfig,
) -> tuple[TableInputPage, ...]:
    contract_sha256 = _input_identity(config)
    pages: list[TableInputPage] = []
    document_ids: set[str] = set()
    page_ids: set[str] = set()
    extraction_ids: set[str] = set()
    raster_paths: set[Path] = set()

    for extraction_run in source.extraction_runs:
        identity = _load_raw_extraction_run_identity(extraction_run)
        try:
            connection = sqlite3.connect(f"file:{identity.ledger}?mode=ro", uri=True, timeout=30.0)
            document_rows = connection.execute(
                """
                SELECT document_id, source_json, source_sha256, page_count, status,
                       failure_json
                FROM documents
                WHERE run_id = ?
                ORDER BY document_id
                """,
                (extraction_run.run_id,),
            ).fetchall()
            raw_page_rows = connection.execute(
                """
                SELECT extraction_id, document_id, page_id, page_index, status,
                       result_json, failure_json
                FROM pages
                WHERE run_id = ?
                ORDER BY document_id, page_index
                """,
                (extraction_run.run_id,),
            ).fetchall()
        except sqlite3.Error as error:
            raise TableViewError(
                f"raw extraction page inventory query failed: {extraction_run.run_id}"
            ) from error
        finally:
            if "connection" in locals():
                connection.close()
                del connection

        if len(document_rows) != extraction_run.expected_documents:
            raise TableViewError(
                f"raw extraction registered document count differs: {extraction_run.run_id}"
            )
        ledger_document_ids = {cast(str, row[0]) for row in document_rows}
        if ledger_document_ids != set(identity.inventory):
            raise TableViewError(
                f"raw extraction registered documents differ from inventory: "
                f"{extraction_run.run_id}"
            )
        duplicate_document_ids = document_ids.intersection(ledger_document_ids)
        if duplicate_document_ids:
            raise TableViewError(
                "duplicate documentId across raw extraction runs: "
                f"{sorted(duplicate_document_ids)!r}"
            )
        document_ids.update(ledger_document_ids)
        document_page_counts: dict[str, int] = {}
        document_source_sha256: dict[str, str] = {}
        for (
            document_id,
            source_json,
            source_sha256,
            page_count,
            document_status,
            document_failure_json,
        ) in document_rows:
            if (
                document_status not in {"complete", "incomplete"}
                or document_failure_json is not None
                or type(page_count) is not int
                or page_count <= 0
                or not isinstance(source_sha256, str)
            ):
                raise TableViewError(
                    f"raw extraction document is pending, failed, or uninspected: {document_id}"
                )
            try:
                registered_source = SourceObject.model_validate_json(source_json, strict=True)
            except ValueError as error:
                raise TableViewError(
                    f"raw extraction document source is invalid: {document_id}"
                ) from error
            inventory_source = identity.inventory[cast(str, document_id)]
            if registered_source != inventory_source or (
                inventory_source.source_sha256 is not None
                and source_sha256 != inventory_source.source_sha256
            ):
                raise TableViewError(
                    f"raw extraction document differs from its inventory: {document_id}"
                )
            document_page_counts[cast(str, document_id)] = page_count
            document_source_sha256[cast(str, document_id)] = source_sha256

        if (
            sum(document_page_counts.values()) != extraction_run.expected_pages
            or len(raw_page_rows) != extraction_run.expected_pages
        ):
            raise TableViewError(
                f"raw extraction registered page count differs: {extraction_run.run_id}"
            )
        page_indexes_by_document: dict[str, list[int]] = defaultdict(list)
        for raw_page in raw_page_rows:
            page_indexes_by_document[cast(str, raw_page[1])].append(cast(int, raw_page[3]))
        for document_id, page_count in document_page_counts.items():
            if page_indexes_by_document.get(document_id) != list(range(page_count)):
                raise TableViewError(
                    f"raw extraction pages are incomplete or out of order: {document_id}"
                )

        for (
            extraction_id,
            document_id,
            page_id,
            page_index,
            page_status,
            result_json,
            failure_json,
        ) in raw_page_rows:
            page_count = document_page_counts[cast(str, document_id)]
            source_sha256 = document_source_sha256[cast(str, document_id)]
            raw_text_sha256: str | None
            if page_status == "success" and result_json is not None and failure_json is None:
                try:
                    record = PageExtractionRecord.model_validate_json(result_json, strict=True)
                except ValueError as error:
                    raise TableViewError(
                        f"raw extraction successful page is invalid: {extraction_id}"
                    ) from error
                if record.raster_path is None:
                    raise TableViewError(
                        f"raw extraction successful page has no retained raster: {page_id}"
                    )
                raster_relative = _relative_artifact(record.raster_path, context="raster_path")
                raster_path = _canonical_regular_file(
                    identity.root / Path(raster_relative),
                    root=identity.root,
                    context="retained raw OCR raster",
                )
                raster_mime_type = record.raster_mime_type
                raster_size_bytes = record.raster_size_bytes
                raster_sha256 = record.raster_sha256
                raw_text_sha256 = record.raw_ocr_text_sha256
                provenance_record: PageExtractionRecord | PageExtractionFailure = record
            elif page_status == "failed" and failure_json is not None and result_json is None:
                try:
                    failure = PageExtractionFailure.model_validate_json(failure_json, strict=True)
                except ValueError as error:
                    raise TableViewError(
                        f"raw extraction failed page is invalid: {extraction_id}"
                    ) from error
                if failure.failure_stage not in {"inference", "persist"}:
                    raise TableViewError(
                        "raw extraction page failed before a trustworthy retained raster: "
                        f"{page_id}: {failure.failure_stage}"
                    )
                (
                    raster_path,
                    raster_mime_type,
                    raster_sha256,
                ) = _failed_page_raster(identity=identity, failure=failure)
                raster_size_bytes = raster_path.stat().st_size
                raw_text_sha256 = None
                provenance_record = failure
            else:
                raise TableViewError(
                    f"raw extraction page is pending or internally inconsistent: {extraction_id}"
                )

            record_source = _record_source(provenance_record)
            inventory_payload = identity.inventory[cast(str, document_id)].model_dump(mode="python")
            record_source.pop("source_sha256")
            inventory_payload.pop("source_sha256")
            if (
                provenance_record.run_id != extraction_run.run_id
                or provenance_record.extraction_id != extraction_id
                or provenance_record.document_id != document_id
                or provenance_record.page_id != page_id
                or provenance_record.page_index != page_index
                or provenance_record.page_number != page_index + 1
                or provenance_record.document_page_count != page_count
                or provenance_record.config_sha256 != extraction_run.config_sha256
                or provenance_record.pipeline_fingerprint != extraction_run.pipeline_fingerprint
                or provenance_record.source_sha256 != source_sha256
                or record_source != inventory_payload
            ):
                raise TableViewError(
                    f"raw extraction page conflicts with its run or inventory: {extraction_id}"
                )

            entry = RasterManifestEntry.model_validate(
                {
                    "page_id": page_id,
                    "document_id": document_id,
                    "raster_path": str(raster_path),
                    "mime_type": raster_mime_type,
                    "raster_sha256": raster_sha256,
                },
                strict=True,
            )
            raster_identity = inspect_raster(entry)
            if raster_identity.size_bytes != raster_size_bytes:
                raise TableViewError(f"retained raw OCR raster size differs: {page_id}")
            if page_id in page_ids or extraction_id in extraction_ids:
                raise TableViewError(
                    "duplicate page or extraction identity in raw extraction inputs"
                )
            if raster_identity.canonical_path in raster_paths:
                raise TableViewError("duplicate canonical raster in raw extraction inputs")
            page_ids.add(cast(str, page_id))
            extraction_ids.add(cast(str, extraction_id))
            raster_paths.add(raster_identity.canonical_path)
            table_view_id = stable_id(
                "table",
                "glm-ocr-table-page-v1",
                cast(str, page_id),
                raster_sha256,
                contract_sha256,
            )
            pages.append(
                TableInputPage.model_validate(
                    {
                        "schemaVersion": 1,
                        "recordSetId": extraction_run.id,
                        "annotationPath": str(identity.provenance),
                        "annotationSha256": identity.provenance_sha256,
                        "documentId": document_id,
                        "documentPageCount": page_count,
                        "pageIndex": page_index,
                        "pageNumber": page_index + 1,
                        "pageId": page_id,
                        "baseExtractionId": extraction_id,
                        "baseExtractionRunId": extraction_run.run_id,
                        "baseRawOcrTextSha256": raw_text_sha256,
                        "sourceSha256": source_sha256,
                        "rasterPath": str(raster_path),
                        "rasterMimeType": raster_mime_type,
                        "rasterSizeBytes": raster_identity.size_bytes,
                        "rasterSha256": raster_sha256,
                        "tableViewId": table_view_id,
                    },
                    strict=True,
                )
            )

    if len(document_ids) != source.expected_documents:
        raise TableViewError("prepared document count does not match expected_documents")
    if len(pages) != source.expected_pages:
        raise TableViewError("prepared page count does not match expected_pages")
    return tuple(pages)


def _prepare_input_pages(config: TableViewConfig) -> tuple[TableInputPage, ...]:
    source = config.source
    if isinstance(source, ValidatedLabelRasterSourceConfig):
        return _prepare_validated_label_input_pages(config, source)
    if isinstance(source, RawExtractionRasterSourceConfig):
        return _prepare_raw_extraction_input_pages(config, source)
    raise TableViewError(f"unsupported table source type: {type(source).__name__}")


def prepare_table_inputs(config: TableViewConfig) -> tuple[TableInputPage, ...]:
    """Freeze and validate the exact successful source-page subset."""

    paths = table_run_paths(config)
    pages = _prepare_input_pages(config)
    payload = b"".join(canonical_json_bytes(page.model_dump(mode="json")) + b"\n" for page in pages)
    digest = sha256_bytes(payload)
    atomic_publish_bytes(paths.input_pages, payload)
    atomic_publish_bytes(
        paths.input_pages_digest,
        f"{digest}  {paths.input_pages.name}\n".encode("ascii"),
    )
    resolved_config = config.model_dump(mode="json")
    config_sha256 = canonical_json_sha256(resolved_config)
    atomic_publish_json(paths.config_snapshot, resolved_config)
    provenance: dict[str, Any] = {
        "schema_version": 1,
        "config_sha256": config_sha256,
        "input_pages_sha256": digest,
        "documents": config.source.expected_documents,
        "pages": config.source.expected_pages,
        "view_contract_sha256": _input_identity(config),
    }
    if isinstance(config.source, ValidatedLabelRasterSourceConfig):
        provenance["record_sets"] = [
            {
                "id": row.id,
                "records_sha256": row.records_sha256,
                "records": row.records,
            }
            for row in config.source.record_sets
        ]
    else:
        provenance["raw_extraction_runs"] = [
            {
                "id": row.id,
                "run_id": row.run_id,
                "config_sha256": row.config_sha256,
                "pipeline_fingerprint": row.pipeline_fingerprint,
                "inventory_sha256": row.inventory_sha256,
                "provenance_sha256": row.provenance_sha256,
                "manifest_sha256": row.manifest_sha256,
                "documents": row.expected_documents,
                "pages": row.expected_pages,
            }
            for row in config.source.extraction_runs
        ]
    atomic_publish_json(paths.provenance, provenance)
    return pages


def load_frozen_prepared_pages(config: TableViewConfig) -> tuple[TableInputPage, ...]:
    """Load and validate the frozen page inventory without rereading source assets."""

    paths = table_run_paths(config)
    expected_sidecar = read_regular_file_bytes(paths.input_pages_digest).decode("ascii")
    expected_digest = expected_sidecar.split(maxsplit=1)[0]
    payload = read_regular_file_bytes(paths.input_pages)
    if sha256_bytes(payload) != expected_digest:
        raise TableViewError("prepared input-pages digest does not match")
    pages: list[TableInputPage] = []
    for line_number, line in enumerate(payload.splitlines(), start=1):
        try:
            pages.append(TableInputPage.model_validate_json(line, strict=True))
        except ValueError as error:
            raise TableViewError(f"invalid prepared page at line {line_number}") from error
    if len(pages) != config.source.expected_pages:
        raise TableViewError("frozen prepared page count differs from configuration")
    return tuple(pages)


def load_prepared_pages(config: TableViewConfig) -> tuple[TableInputPage, ...]:
    pages = load_frozen_prepared_pages(config)
    current = _prepare_input_pages(config)
    if tuple(page.model_dump(mode="json") for page in pages) != tuple(
        page.model_dump(mode="json") for page in current
    ):
        raise TableViewError("prepared pages no longer match the validated source artifacts")
    return tuple(pages)


def _relative_to_run(path: Path, run_root: Path) -> str:
    try:
        return path.relative_to(run_root).as_posix()
    except ValueError as error:
        raise TableViewError("artifact path escapes the table run") from error


def _attempt_record(page: TableInputPage, attempt: Any) -> TableAttemptRecord:
    return TableAttemptRecord.model_validate(
        {
            "schemaVersion": 1,
            "tableViewId": page.tableViewId,
            "documentId": page.documentId,
            "pageId": page.pageId,
            "attemptNumber": attempt.attempt_number,
            "startedAt": attempt.attempt_started_at,
            "completedAt": attempt.attempt_completed_at,
            "durationMs": attempt.attempt_duration_ms,
            "outcome": attempt.outcome,
            "retryable": attempt.retryable,
            "inferenceRequestId": attempt.inference_request_id,
            "inferenceServerRequestId": attempt.inference_server_request_id,
            "httpStatusCode": attempt.http_status_code,
            "retryAfterSeconds": attempt.retry_after_seconds,
            "errorType": attempt.error_type,
            "errorMessage": attempt.error_message,
        },
        strict=True,
    )


def _publish_attempts(
    *,
    paths: TableRunPaths,
    page: TableInputPage,
    attempts: Sequence[Any],
) -> tuple[tuple[str, ...], tuple[str, ...]]:
    if not attempts:
        raise TableViewError("inference returned no immutable attempt records")
    artifact_paths: list[str] = []
    digests: list[str] = []
    for attempt in attempts:
        record = _attempt_record(page, attempt)
        payload = canonical_json_bytes(record.model_dump(mode="json")) + b"\n"
        digest = sha256_bytes(payload)
        path = (
            paths.state_root
            / "attempts"
            / page.tableViewId
            / f"{record.attemptNumber:06d}-{digest[:16]}.json"
        )
        atomic_publish_bytes(path, payload)
        artifact_paths.append(_relative_to_run(path, paths.run_root))
        digests.append(digest)
    return tuple(artifact_paths), tuple(digests)


def _failure_state(
    paths: TableRunPaths, page: TableInputPage
) -> tuple[int, tuple[str, ...], tuple[str, ...], bool]:
    directory = paths.state_root / "failures" / page.tableViewId
    if not directory.exists():
        return 0, (), (), True
    failures: list[TableFailureRecord] = []
    for path in sorted(directory.glob("*.json")):
        try:
            record = TableFailureRecord.model_validate_json(
                read_regular_file_bytes(path), strict=True
            )
        except ValueError as error:
            raise TableViewError(f"invalid table failure artifact: {path}") from error
        if record.tableViewId != page.tableViewId:
            raise TableViewError("table failure artifact has the wrong tableViewId")
        failures.append(record)
    if not failures:
        return 0, (), (), True
    highest = max(row.lastAttemptNumber for row in failures)
    latest = [row for row in failures if row.lastAttemptNumber == highest]
    if len(latest) != 1:
        raise TableViewError("conflicting latest failure artifacts")
    row = latest[0]
    expected_numbers = list(range(1, row.lastAttemptNumber + 1))
    observed_numbers: list[int] = []
    for relative_value, digest in zip(row.attemptPaths, row.attemptSha256s, strict=True):
        relative = _relative_artifact(relative_value, context="failure attemptPath")
        attempt_path = _canonical_regular_file(
            paths.run_root / Path(relative), root=paths.run_root, context="table attempt"
        )
        payload = read_regular_file_bytes(attempt_path)
        if sha256_bytes(payload) != digest:
            raise TableViewError("failure attempt digest differs from failure metadata")
        try:
            attempt = TableAttemptRecord.model_validate_json(payload, strict=True)
        except ValueError as error:
            raise TableViewError(f"invalid table attempt artifact: {attempt_path}") from error
        if attempt.tableViewId != page.tableViewId or attempt.pageId != page.pageId:
            raise TableViewError("failure attempt identity differs from its input page")
        observed_numbers.append(attempt.attemptNumber)
    if observed_numbers != expected_numbers:
        raise TableViewError("failure attempt sequence is incomplete or out of order")
    return row.lastAttemptNumber, row.attemptPaths, row.attemptSha256s, row.retryable


def _commit_path(paths: TableRunPaths, page: TableInputPage) -> Path:
    return paths.state_root / "commits" / page.documentId / page.pageId / f"{page.tableViewId}.json"


def _load_commit(
    paths: TableRunPaths, page: TableInputPage
) -> tuple[TablePageCommit, TablePageResult] | None:
    commit_path = _commit_path(paths, page)
    if not commit_path.exists():
        return None
    try:
        commit = TablePageCommit.model_validate_json(
            read_regular_file_bytes(commit_path), strict=True
        )
    except ValueError as error:
        raise TableViewError(f"invalid table commit: {commit_path}") from error
    input_digest = canonical_json_sha256(page.model_dump(mode="json"))
    if commit.tableViewId != page.tableViewId or commit.inputPageSha256 != input_digest:
        raise TableViewError("table commit conflicts with the prepared input page")
    result_relative = _relative_artifact(commit.resultPath, context="resultPath")
    result_path = _canonical_regular_file(
        paths.run_root / Path(result_relative), root=paths.run_root, context="table result"
    )
    result_bytes = read_regular_file_bytes(result_path)
    if sha256_bytes(result_bytes) != commit.resultSha256:
        raise TableViewError("table result digest differs from its commit")
    try:
        result = TablePageResult.model_validate_json(result_bytes, strict=True)
    except ValueError as error:
        raise TableViewError(f"invalid committed table result: {result_path}") from error
    if (
        result.tableViewId != page.tableViewId
        or result.inputPageSha256 != input_digest
        or result.pageId != page.pageId
        or result.rasterSha256 != page.rasterSha256
        or sha256_bytes(result.rawTableText.encode("utf-8")) != result.rawTableTextSha256
    ):
        raise TableViewError("committed table result conflicts with its input")
    raw_relative = _relative_artifact(result.rawResponsePath, context="rawResponsePath")
    raw_path = _canonical_regular_file(
        paths.run_root / Path(raw_relative), root=paths.run_root, context="raw table response"
    )
    if sha256_file(raw_path) != result.rawResponseSha256:
        raise TableViewError("raw table response digest differs from result metadata")
    if len(result.attemptPaths) != len(result.attemptSha256s):
        raise TableViewError("table result attempt paths and hashes differ in length")
    attempt_numbers: list[int] = []
    for relative_value, digest in zip(result.attemptPaths, result.attemptSha256s, strict=True):
        relative = _relative_artifact(relative_value, context="attemptPath")
        attempt_path = _canonical_regular_file(
            paths.run_root / Path(relative), root=paths.run_root, context="table attempt"
        )
        if sha256_file(attempt_path) != digest:
            raise TableViewError("table attempt digest differs from result metadata")
        try:
            attempt = TableAttemptRecord.model_validate_json(
                read_regular_file_bytes(attempt_path), strict=True
            )
        except ValueError as error:
            raise TableViewError(f"invalid table attempt artifact: {attempt_path}") from error
        if (
            attempt.tableViewId != page.tableViewId
            or attempt.pageId != page.pageId
            or attempt.attemptNumber != len(attempt_numbers) + 1
        ):
            raise TableViewError("table result attempt sequence conflicts with its input")
        attempt_numbers.append(attempt.attemptNumber)
    if not attempt_numbers or attempt_numbers[-1] != result.attemptCount:
        raise TableViewError("table result attemptCount differs from its attempt sequence")
    return commit, result


async def _process_page(
    *,
    config: TableViewConfig,
    paths: TableRunPaths,
    page: TableInputPage,
    client: _TableClient,
    server: ServerInfo,
    global_limit: asyncio.Semaphore,
    retry_limit: asyncio.Semaphore,
) -> bool:
    if _load_commit(paths, page) is not None:
        return True
    prior_count, prior_paths, prior_hashes, retryable = _failure_state(paths, page)
    if not retryable or prior_count >= config.run.max_total_attempts_per_page:
        return False
    remaining = config.run.max_total_attempts_per_page - prior_count
    request_attempts = min(config.vllm.retry.max_attempts, remaining)
    entry = RasterManifestEntry.model_validate(
        {
            "page_id": page.pageId,
            "document_id": page.documentId,
            "raster_path": page.rasterPath,
            "mime_type": page.rasterMimeType,
            "raster_sha256": page.rasterSha256,
        },
        strict=True,
    )
    identity = await asyncio.to_thread(inspect_raster, entry)
    started = time.perf_counter()
    try:
        async with _inference_slot(
            prior_attempt_count=prior_count,
            global_limit=global_limit,
            retry_limit=retry_limit,
        ):
            response = await client.recognize_page(
                Path(page.rasterPath),
                mime_type=page.rasterMimeType,
                raster_sha256=page.rasterSha256,
                request_id=page.tableViewId,
                first_attempt_number=prior_count + 1,
                max_attempts=request_attempts,
            )
    except VllmClientError as error:
        new_paths, new_hashes = _publish_attempts(paths=paths, page=page, attempts=error.attempts)
        all_paths = (*prior_paths, *new_paths)
        all_hashes = (*prior_hashes, *new_hashes)
        last = error.attempts[-1]
        failure = TableFailureRecord.model_validate(
            {
                "schemaVersion": 1,
                "tableViewId": page.tableViewId,
                "documentId": page.documentId,
                "pageId": page.pageId,
                "lastAttemptNumber": last.attempt_number,
                "retryable": last.retryable,
                "errorType": type(error).__name__,
                "errorMessage": str(error),
                "attemptPaths": all_paths,
                "attemptSha256s": all_hashes,
            },
            strict=True,
        )
        payload = canonical_json_bytes(failure.model_dump(mode="json")) + b"\n"
        digest = sha256_bytes(payload)
        failure_path = (
            paths.state_root
            / "failures"
            / page.tableViewId
            / f"{last.attempt_number:06d}-{digest[:16]}.json"
        )
        atomic_publish_bytes(failure_path, payload)
        if config.run.fail_fast:
            raise TableViewError(f"table inference failed for {page.pageId}") from error
        return False
    duration_ms = (time.perf_counter() - started) * 1000.0
    await asyncio.to_thread(validate_current_raster, entry, identity)

    new_paths, new_hashes = _publish_attempts(paths=paths, page=page, attempts=response.attempts)
    all_paths = (*prior_paths, *new_paths)
    all_hashes = (*prior_hashes, *new_hashes)
    raw_path = (
        paths.state_root
        / "raw-responses"
        / page.documentId
        / page.pageId
        / f"{response.raw_response_sha256}.json"
    )
    atomic_publish_bytes(raw_path, response.raw_response)
    input_digest = canonical_json_sha256(page.model_dump(mode="json"))
    result = TablePageResult.model_validate(
        {
            "schemaVersion": 1,
            "tableViewId": page.tableViewId,
            "inputPageSha256": input_digest,
            "documentId": page.documentId,
            "documentPageCount": page.documentPageCount,
            "pageIndex": page.pageIndex,
            "pageNumber": page.pageNumber,
            "pageId": page.pageId,
            "baseExtractionId": page.baseExtractionId,
            "baseExtractionRunId": page.baseExtractionRunId,
            "rasterPath": page.rasterPath,
            "rasterSha256": page.rasterSha256,
            "prompt": config.vllm.prompt,
            "promptSha256": sha256_bytes(config.vllm.prompt.encode("utf-8")),
            "model": server.model_repository,
            "modelRevision": server.model_revision,
            "servedModelName": server.served_model_name,
            "vllmEngineVersion": server.version,
            "serverContractSha256": server.contract_sha256,
            "responseId": response.response_id,
            "inferenceRequestId": response.request_id,
            "inferenceServerRequestId": response.server_request_id,
            "finishReason": response.finish_reason,
            "promptTokens": response.prompt_tokens,
            "completionTokens": response.completion_tokens,
            "totalTokens": response.total_tokens,
            "attemptCount": response.attempts[-1].attempt_number,
            "inferenceDurationMs": duration_ms,
            "rawTableText": response.text,
            "rawTableTextSha256": sha256_bytes(response.text.encode("utf-8")),
            "rawResponsePath": _relative_to_run(raw_path, paths.run_root),
            "rawResponseSha256": response.raw_response_sha256,
            "attemptPaths": all_paths,
            "attemptSha256s": all_hashes,
        },
        strict=True,
    )
    result_payload = canonical_json_bytes(result.model_dump(mode="json")) + b"\n"
    result_sha256 = sha256_bytes(result_payload)
    result_path = (
        paths.state_root / "results" / page.documentId / page.pageId / f"{result_sha256}.json"
    )
    atomic_publish_bytes(result_path, result_payload)
    commit = TablePageCommit.model_validate(
        {
            "schemaVersion": 1,
            "tableViewId": page.tableViewId,
            "inputPageSha256": input_digest,
            "resultPath": _relative_to_run(result_path, paths.run_root),
            "resultSha256": result_sha256,
        },
        strict=True,
    )
    atomic_publish_bytes(
        _commit_path(paths, page),
        canonical_json_bytes(commit.model_dump(mode="json")) + b"\n",
    )
    _load_commit(paths, page)
    return True


@asynccontextmanager
async def _inference_slot(
    *,
    prior_attempt_count: int,
    global_limit: asyncio.Semaphore,
    retry_limit: asyncio.Semaphore,
) -> AsyncIterator[None]:
    """Bound the retry tail separately from first-pass throughput.

    Pages with committed failed attempts have already demonstrated pathological
    latency under the saturated first pass. Acquiring the recovery semaphore
    before the ordinary global slot prevents deterministic resume attempts from
    recreating that same overloaded batch.
    """

    if prior_attempt_count > 0:
        async with retry_limit, global_limit:
            yield
        return
    async with global_limit:
        yield


def _retry_recovery_concurrency(config: TableConcurrencyConfig) -> int:
    """Give retried pages at most one fair document share of the global batch."""

    fair_document_share = max(1, config.max_inflight_pages_global // config.max_active_documents)
    return min(config.max_inflight_pages_per_document, fair_document_share)


async def run_table_view(
    config: TableViewConfig,
    *,
    client: _TableClient | None = None,
    progress: TableProgressCallback | None = None,
) -> dict[str, int]:
    """Run or resume bounded table inference and publish the dataset if complete."""

    prepare_table_inputs(config)
    pages = load_prepared_pages(config)
    paths = table_run_paths(config)
    owns_client = client is None
    active_client: _TableClient = client or VllmOcrClient(
        config.vllm, max_connections=config.concurrency.max_inflight_pages_global
    )
    global_limit = asyncio.Semaphore(config.concurrency.max_inflight_pages_global)
    retry_limit = asyncio.Semaphore(_retry_recovery_concurrency(config.concurrency))
    document_limit = asyncio.Semaphore(config.concurrency.max_active_documents)
    total_documents = len({page.documentId for page in pages})
    committed_pages: list[TableInputPage] = []
    remaining_pages: list[TableInputPage] = []
    for page in pages:
        target = committed_pages if _load_commit(paths, page) is not None else remaining_pages
        target.append(page)
    grouped: dict[str, list[TableInputPage]] = defaultdict(list)
    for page in remaining_pages:
        grouped[page.documentId].append(page)
    successful = len(committed_pages)
    failed = 0
    processed = successful
    processed_this_invocation = 0
    started_at = time.perf_counter()
    progress_lock = asyncio.Lock()

    def progress_snapshot() -> TableProgress:
        elapsed_seconds = max(0.0, time.perf_counter() - started_at)
        throughput = (
            processed_this_invocation / elapsed_seconds if elapsed_seconds > 0.0 else 0.0
        )
        remaining = len(pages) - processed
        eta_seconds = (
            0.0
            if remaining == 0
            else remaining / throughput
            if throughput > 0.0
            else None
        )
        return TableProgress(
            total_pages=len(pages),
            processed_pages=processed,
            successful_pages=successful,
            failed_pages_this_invocation=failed,
            elapsed_seconds=elapsed_seconds,
            throughput_pages_per_second=throughput,
            eta_seconds=eta_seconds,
        )

    async def process_document(document_pages: Sequence[TableInputPage]) -> None:
        nonlocal failed, processed, processed_this_invocation, successful
        async with document_limit:
            page_limit = asyncio.Semaphore(config.concurrency.max_inflight_pages_per_document)

            async def bounded(page: TableInputPage) -> None:
                nonlocal failed, processed, processed_this_invocation, successful
                async with page_limit:
                    ok = await _process_page(
                        config=config,
                        paths=paths,
                        page=page,
                        client=active_client,
                        server=server,
                        global_limit=global_limit,
                        retry_limit=retry_limit,
                    )
                async with progress_lock:
                    processed += 1
                    processed_this_invocation += 1
                    if ok:
                        successful += 1
                    else:
                        failed += 1
                    if progress is not None:
                        progress(progress_snapshot())

            async with asyncio.TaskGroup() as group:
                for page in document_pages:
                    group.create_task(bounded(page))

    try:
        server = await active_client.check_readiness()
        if progress is not None:
            progress(progress_snapshot())
        async with asyncio.TaskGroup() as group:
            for document_pages in grouped.values():
                group.create_task(process_document(document_pages))
    finally:
        if owns_client:
            await cast(VllmOcrClient, active_client).close()
    summary = {
        "documents": total_documents,
        "pages": len(pages),
        "successful_pages": successful,
        "failed_pages": failed,
    }
    if successful == len(pages):
        publish_table_dataset(config)
    else:
        raise IncompleteTableRunError(f"table-view run is incomplete: {summary}")
    return summary


def _committed_results(
    config: TableViewConfig,
) -> tuple[tuple[TableInputPage, TablePageResult], ...]:
    pages = load_prepared_pages(config)
    paths = table_run_paths(config)
    rows: list[tuple[TableInputPage, TablePageResult]] = []
    for page in pages:
        committed = _load_commit(paths, page)
        if committed is None:
            continue
        rows.append((page, committed[1]))
    if len(rows) != len(pages):
        raise IncompleteTableRunError(
            f"table-view run has {len(rows)}/{len(pages)} committed pages"
        )
    return tuple(rows)


def _table_output_class(value: str) -> str:
    stripped = value.strip().lower()
    if not stripped:
        return "empty"
    has_open = "<table" in stripped
    has_close = "</table>" in stripped
    if has_open and has_close:
        return "complete_html_table_markup"
    if has_open or has_close:
        return "incomplete_html_table_markup"
    return "other_nonempty"


def _numeric_distribution(values: Sequence[int | float]) -> dict[str, float]:
    if not values:
        raise TableViewError("quality distribution requires at least one value")
    ordered = sorted(float(value) for value in values)
    p95_index = max(0, (95 * len(ordered) + 99) // 100 - 1)
    return {
        "min": ordered[0],
        "median": float(median(ordered)),
        "p95": ordered[p95_index],
        "max": ordered[-1],
        "mean": sum(ordered) / len(ordered),
    }


def _table_quality_artifacts(
    results: Sequence[TablePageResult],
    attempts: Sequence[dict[str, Any]],
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    classes = Counter(_table_output_class(row.rawTableText) for row in results)
    finish_reasons = Counter(row.finishReason for row in results)
    attempt_outcomes = Counter(cast(str, row["outcome"]) for row in attempts)
    documents_with_complete_markup = {
        row.documentId
        for row in results
        if _table_output_class(row.rawTableText) == "complete_html_table_markup"
    }
    flags: list[dict[str, Any]] = []
    for row in results:
        output_class = _table_output_class(row.rawTableText)
        reasons: list[str] = []
        if row.finishReason == "repetition":
            reasons.append("repetition_finish")
        if output_class == "incomplete_html_table_markup":
            reasons.append("incomplete_html_table_markup")
        elif output_class == "other_nonempty":
            reasons.append("non_html_nonempty_output")
        if reasons:
            flags.append(
                {
                    "documentId": row.documentId,
                    "pageId": row.pageId,
                    "pageNumber": row.pageNumber,
                    "tableViewId": row.tableViewId,
                    "finishReason": row.finishReason,
                    "completionTokens": row.completionTokens,
                    "rawTableTextSha256": row.rawTableTextSha256,
                    "outputClass": output_class,
                    "reviewReasons": reasons,
                }
            )
    summary = {
        "schema_version": 1,
        "pages": len(results),
        "documents": len({row.documentId for row in results}),
        "documents_with_complete_html_table_markup": len(documents_with_complete_markup),
        "page_output_classes": dict(sorted(classes.items())),
        "finish_reasons": dict(sorted(finish_reasons.items())),
        "pages_requiring_quality_review": len(flags),
        "pages_with_multiple_attempts": sum(row.attemptCount > 1 for row in results),
        "completion_tokens": _numeric_distribution([row.completionTokens for row in results]),
        "raw_table_characters": _numeric_distribution([len(row.rawTableText) for row in results]),
        "attempts": {
            "count": len(attempts),
            "outcomes": dict(sorted(attempt_outcomes.items())),
            "duration_ms": _numeric_distribution(
                [cast(float, row["durationMs"]) for row in attempts]
            ),
        },
    }
    return summary, flags


def publish_table_dataset(config: TableViewConfig) -> Path:
    """Publish deterministic JSONL projections and the manifest last."""

    rows = _committed_results(config)
    paths = table_run_paths(config)
    page_payload = b"".join(
        canonical_json_bytes(result.model_dump(mode="json")) + b"\n" for _, result in rows
    )
    page_sha256 = sha256_bytes(page_payload)
    page_path = paths.dataset_root / f"table-pages-{page_sha256[:16]}.jsonl"
    atomic_publish_bytes(page_path, page_payload)

    attempts: list[dict[str, Any]] = []
    for _, result in rows:
        for relative_value, digest in zip(result.attemptPaths, result.attemptSha256s, strict=True):
            relative = _relative_artifact(relative_value, context="attemptPath")
            attempt_path = _canonical_regular_file(
                paths.run_root / Path(relative), root=paths.run_root, context="table attempt"
            )
            payload = read_regular_file_bytes(attempt_path)
            if sha256_bytes(payload) != digest:
                raise TableViewError("table attempt changed before dataset publication")
            value = _strict_json(payload, context=str(attempt_path))
            if not isinstance(value, dict):
                raise TableViewError("table attempt artifact root is not an object")
            attempts.append(cast(dict[str, Any], value))
    attempt_payload = b"".join(canonical_json_bytes(row) + b"\n" for row in attempts)
    attempt_sha256 = sha256_bytes(attempt_payload)
    attempt_path = paths.dataset_root / f"attempts-{attempt_sha256[:16]}.jsonl"
    atomic_publish_bytes(attempt_path, attempt_payload)

    results = [result for _, result in rows]
    quality_summary, quality_flags = _table_quality_artifacts(results, attempts)
    quality_summary_path = paths.dataset_root / "quality-summary.json"
    atomic_publish_json(quality_summary_path, quality_summary)
    quality_summary_payload = read_regular_file_bytes(quality_summary_path)
    quality_flags_payload = b"".join(canonical_json_bytes(row) + b"\n" for row in quality_flags)
    quality_flags_path = paths.dataset_root / "quality-flags.jsonl"
    atomic_publish_bytes(quality_flags_path, quality_flags_payload)

    manifest = {
        "schema_version": 1,
        "run_id": config.run.run_id,
        "view_type": config.view_type,
        "config_sha256": canonical_json_sha256(config.model_dump(mode="json")),
        "input_pages_sha256": sha256_file(paths.input_pages),
        "documents": config.source.expected_documents,
        "pages": len(rows),
        "attempts": len(attempts),
        "orchestration": {
            "first_pass_max_inflight_pages_global": (config.concurrency.max_inflight_pages_global),
            "retry_recovery_max_inflight_pages_global": (
                _retry_recovery_concurrency(config.concurrency)
            ),
            "retry_recovery_policy": "one_fair_active_document_share",
        },
        "quality_summary": quality_summary,
        "files": [
            {
                "kind": "table_pages",
                "path": _relative_to_run(page_path, paths.run_root),
                "rows": len(rows),
                "bytes": len(page_payload),
                "sha256": page_sha256,
            },
            {
                "kind": "attempts",
                "path": _relative_to_run(attempt_path, paths.run_root),
                "rows": len(attempts),
                "bytes": len(attempt_payload),
                "sha256": attempt_sha256,
            },
            {
                "kind": "quality_summary",
                "path": _relative_to_run(quality_summary_path, paths.run_root),
                "rows": 1,
                "bytes": len(quality_summary_payload),
                "sha256": sha256_bytes(quality_summary_payload),
            },
            {
                "kind": "quality_flags",
                "path": _relative_to_run(quality_flags_path, paths.run_root),
                "rows": len(quality_flags),
                "bytes": len(quality_flags_payload),
                "sha256": sha256_bytes(quality_flags_payload),
            },
        ],
    }
    manifest_path = paths.dataset_root / "manifest.json"
    atomic_publish_json(manifest_path, manifest)
    return manifest_path


def table_status(config: TableViewConfig) -> dict[str, int]:
    pages = load_prepared_pages(config)
    paths = table_run_paths(config)
    successful = 0
    exhausted = 0
    retryable = 0
    for page in pages:
        if _load_commit(paths, page) is not None:
            successful += 1
            continue
        attempt_count, _, _, can_retry = _failure_state(paths, page)
        if attempt_count == 0:
            continue
        if can_retry and attempt_count < config.run.max_total_attempts_per_page:
            retryable += 1
        else:
            exhausted += 1
    return {
        "documents": config.source.expected_documents,
        "pages": len(pages),
        "successful_pages": successful,
        "failed_pages": exhausted,
        "retryable_pages": retryable,
        "pending_pages": len(pages) - successful - exhausted - retryable,
    }
