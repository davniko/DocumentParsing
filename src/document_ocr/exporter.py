"""Streaming, verified, manifest-last Parquet publication."""

from __future__ import annotations

import asyncio
import hashlib
import os
import stat
import types
import uuid
from collections.abc import AsyncIterator, Mapping
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Annotated, Any, Literal, Union, get_args, get_origin

import pyarrow as pa
import pyarrow.parquet as pq
from pydantic import AwareDatetime, BaseModel

from document_ocr.atomic import ArtifactReadError, atomic_publish_bytes, read_regular_file_bytes
from document_ocr.hashing import canonical_json_bytes, sha256_bytes, sha256_file
from document_ocr.ledger import ExtractionLedger
from document_ocr.models import (
    DatasetManifest,
    InferenceAttempt,
    PageExtractionRecord,
)


class DatasetPublicationError(RuntimeError):
    """A staged or previously published dataset failed integrity validation."""


@dataclass(frozen=True, slots=True)
class PublishedDataset:
    manifest: dict[str, Any]
    manifest_sha256: str
    manifest_path: Path


def _unwrap_annotation(annotation: object) -> tuple[object, bool]:
    nullable = False
    current = annotation
    while True:
        origin = get_origin(current)
        if origin is Annotated:
            current = get_args(current)[0]
            continue
        if origin in {Union, types.UnionType}:
            members = list(get_args(current))
            if type(None) in members:
                nullable = True
                members.remove(type(None))
            if len(members) != 1:
                raise TypeError(f"unsupported multi-type field annotation: {annotation!r}")
            current = members[0]
            continue
        return current, nullable


def _arrow_type(annotation: object, field_name: str) -> pa.DataType:
    value, _ = _unwrap_annotation(annotation)
    origin = get_origin(value)
    if origin is Literal:
        literal_values = get_args(value)
        value_types = {type(item) for item in literal_values}
        if len(value_types) != 1:
            raise TypeError(f"mixed literal types are unsupported for {field_name}")
        value = next(iter(value_types))

    if value is str:
        return pa.large_string() if field_name in {"raw_ocr_text", "error_message"} else pa.string()
    if value is bool:
        return pa.bool_()
    if value is int:
        return pa.int64()
    if value is float:
        return pa.float64()
    if value in {datetime, AwareDatetime}:
        return pa.timestamp("us", tz="UTC")
    raise TypeError(f"unsupported Arrow annotation for {field_name}: {annotation!r}")


def arrow_schema(model: type[BaseModel]) -> pa.Schema:
    fields: list[pa.Field] = []
    for name, model_field in model.model_fields.items():
        _, nullable = _unwrap_annotation(model_field.annotation)
        fields.append(pa.field(name, _arrow_type(model_field.annotation, name), nullable=nullable))
    metadata = {
        b"record_model": f"{model.__module__}.{model.__name__}".encode(),
        b"dataset_schema_version": b"2",
    }
    return pa.schema(fields, metadata=metadata)


def _validated_record(model: type[BaseModel], raw: Mapping[str, Any]) -> BaseModel:
    try:
        record = model.model_validate_json(canonical_json_bytes(raw), strict=True)
    except Exception as error:
        raise DatasetPublicationError(
            f"ledger row does not satisfy {model.__name__}: {error}"
        ) from error
    return record


def _validate_page_artifact(
    run_root: Path,
    *,
    relative_path: str,
    expected_sha256: str,
    expected_size: int | None,
    artifact_name: str,
) -> int:
    root = run_root.resolve(strict=True)
    target = run_root / relative_path
    try:
        resolved = target.resolve(strict=True)
    except OSError as error:
        raise DatasetPublicationError(
            f"{artifact_name} artifact is missing: {relative_path}"
        ) from error
    if not resolved.is_relative_to(root) or resolved != Path(os.path.abspath(target)):
        raise DatasetPublicationError(
            f"{artifact_name} artifact traverses outside the run: {relative_path}"
        )
    missing_flags = [name for name in ("O_CLOEXEC", "O_NOFOLLOW") if not hasattr(os, name)]
    if missing_flags:
        raise DatasetPublicationError(
            f"safe {artifact_name} validation requires operating-system flags: "
            + ", ".join(missing_flags)
        )
    flags = os.O_RDONLY | os.O_CLOEXEC | os.O_NOFOLLOW
    try:
        descriptor = os.open(resolved, flags)
    except OSError as error:
        raise DatasetPublicationError(
            f"{artifact_name} artifact cannot be opened safely: {relative_path}"
        ) from error
    digest = hashlib.sha256()
    size_bytes = 0
    try:
        if not stat.S_ISREG(os.fstat(descriptor).st_mode):
            raise DatasetPublicationError(
                f"{artifact_name} artifact is not a regular file: {relative_path}"
            )
        with os.fdopen(descriptor, "rb", closefd=False) as stream:
            while chunk := stream.read(1024 * 1024):
                digest.update(chunk)
                size_bytes += len(chunk)
    finally:
        os.close(descriptor)
    if digest.hexdigest() != expected_sha256:
        raise DatasetPublicationError(f"{artifact_name} artifact hash mismatch: {relative_path}")
    if expected_size is not None and size_bytes != expected_size:
        raise DatasetPublicationError(f"{artifact_name} artifact size mismatch: {relative_path}")
    return size_bytes


def _validate_raw_response(run_root: Path, record: PageExtractionRecord) -> int:
    return _validate_page_artifact(
        run_root,
        relative_path=record.raw_response_path,
        expected_sha256=record.raw_response_sha256,
        expected_size=None,
        artifact_name="raw response",
    )


def _validate_raster(run_root: Path, record: PageExtractionRecord) -> int:
    if record.raster_path is None:
        raise DatasetPublicationError("cannot validate a missing retained raster path")
    return _validate_page_artifact(
        run_root,
        relative_path=record.raster_path,
        expected_sha256=record.raster_sha256,
        expected_size=record.raster_size_bytes,
        artifact_name="retained raster",
    )


def _write_rows(
    writer: pq.ParquetWriter,
    schema: pa.Schema,
    rows: list[dict[str, Any]],
) -> None:
    writer.write_table(pa.Table.from_pylist(rows, schema=schema), row_group_size=len(rows))


async def _stage_parquet(
    *,
    records: AsyncIterator[dict[str, Any]],
    model: type[BaseModel],
    path: Path,
    run_root: Path,
    batch_rows: int,
    compression: str,
) -> dict[str, Any]:
    if batch_rows <= 0:
        raise ValueError("batch_rows must be positive")
    schema = arrow_schema(model)
    codec = None if compression == "none" else compression
    writer = pq.ParquetWriter(
        path,
        schema,
        compression=codec,
        use_dictionary=False,
        write_statistics=True,
        data_page_version="2.0",
    )
    row_count = 0
    raw_response_bytes = 0
    raw_response_paths: set[str] = set()
    raster_bytes = 0
    raster_paths: set[str] = set()
    batch: list[dict[str, Any]] = []
    try:
        async for raw in records:
            record = _validated_record(model, raw)
            if isinstance(record, PageExtractionRecord):
                if record.raw_response_path in raw_response_paths:
                    raise DatasetPublicationError(
                        f"duplicate raw response pointer: {record.raw_response_path}"
                    )
                raw_response_paths.add(record.raw_response_path)
                raw_response_bytes += await asyncio.to_thread(
                    _validate_raw_response, run_root, record
                )
                if record.raster_path is not None:
                    if record.raster_path in raster_paths:
                        raise DatasetPublicationError(
                            f"duplicate retained raster pointer: {record.raster_path}"
                        )
                    raster_paths.add(record.raster_path)
                    raster_bytes += await asyncio.to_thread(_validate_raster, run_root, record)
            batch.append(record.model_dump(mode="python"))
            if len(batch) == batch_rows:
                await asyncio.to_thread(_write_rows, writer, schema, batch)
                row_count += len(batch)
                batch = []
        if batch:
            await asyncio.to_thread(_write_rows, writer, schema, batch)
            row_count += len(batch)
    finally:
        await asyncio.to_thread(writer.close)

    metadata = await asyncio.to_thread(_validate_staged_parquet, path, schema, row_count, model)
    metadata["raw_response_artifact_count"] = len(raw_response_paths)
    metadata["raw_response_bytes"] = raw_response_bytes
    metadata["raster_artifact_count"] = len(raster_paths)
    metadata["raster_bytes"] = raster_bytes
    return metadata


def _validate_staged_parquet(
    path: Path,
    schema: pa.Schema,
    row_count: int,
    model: type[BaseModel],
) -> dict[str, Any]:
    with path.open("rb") as stream:
        os.fsync(stream.fileno())
    parquet = pq.ParquetFile(path)
    if parquet.metadata.num_rows != row_count:
        raise DatasetPublicationError(
            f"Parquet row count mismatch for {path}: wrote {row_count}, "
            f"read {parquet.metadata.num_rows}"
        )
    if not parquet.schema_arrow.equals(schema, check_metadata=True):
        raise DatasetPublicationError(f"Parquet schema mismatch after reopening {path}")
    return {
        "record_model": model.__name__,
        "rows": row_count,
        "bytes": path.stat().st_size,
        "sha256": sha256_file(path),
        "arrow_schema_sha256": sha256_bytes(schema.serialize().to_pybytes()),
    }


def _publish_content_addressed(staged: Path, dataset_root: Path, stem: str) -> tuple[Path, str]:
    digest = sha256_file(staged)
    destination = dataset_root / f"{stem}-{digest[:16]}.parquet"
    try:
        os.link(staged, destination, follow_symlinks=False)
    except FileExistsError:
        if (
            not destination.is_file()
            or destination.is_symlink()
            or sha256_file(destination) != digest
        ):
            raise DatasetPublicationError(
                f"published artifact conflicts at {destination}"
            ) from None
    staged.unlink()
    directory_fd = os.open(dataset_root, os.O_RDONLY)
    try:
        os.fsync(directory_fd)
    finally:
        os.close(directory_fd)
    return destination, digest


async def publish_complete_dataset(
    *,
    ledger: ExtractionLedger,
    run_id: str,
    run_root: Path,
    batch_rows: int,
    compression: str,
    retain_page_images: bool,
) -> PublishedDataset:
    """Publish verified page/attempt Parquet files and the manifest last."""

    summary = await ledger.require_complete(run_id)
    contract = await ledger.run_contract(run_id)
    dataset_root = run_root / "dataset"
    dataset_root.mkdir(parents=True, exist_ok=True)
    staging_root = dataset_root / f".staging-{uuid.uuid4().hex}"
    staging_root.mkdir(parents=False, exist_ok=False)

    specifications = (
        (
            "pages",
            PageExtractionRecord,
            ledger.iter_success_records(run_id),
        ),
        (
            "attempts",
            InferenceAttempt,
            ledger.iter_attempts(run_id),
        ),
    )
    files: list[dict[str, Any]] = []
    expected_rows = {
        "pages": summary["successful_pages"],
        "attempts": summary["attempt_rows"],
    }
    try:
        for stem, model, iterator in specifications:
            staged = staging_root / f"{stem}.parquet"
            metadata = await _stage_parquet(
                records=iterator,
                model=model,
                path=staged,
                run_root=run_root,
                batch_rows=batch_rows,
                compression=compression,
            )
            if metadata["rows"] != expected_rows[stem]:
                raise DatasetPublicationError(
                    f"{stem} export row count {metadata['rows']} does not match "
                    f"ledger count {expected_rows[stem]}"
                )
            published, digest = await asyncio.to_thread(
                _publish_content_addressed, staged, dataset_root, stem
            )
            if digest != metadata["sha256"]:
                raise DatasetPublicationError(f"staged hash changed while publishing {stem}")
            files.append(
                {
                    **metadata,
                    "path": published.relative_to(run_root).as_posix(),
                }
            )
        staging_root.rmdir()
    except BaseException:
        # Keep non-empty staging directories for forensic inspection.
        if staging_root.exists() and not any(staging_root.iterdir()):
            staging_root.rmdir()
        raise

    manifest_payload: dict[str, Any] = {
        "schema_version": 2,
        "dataset_kind": "glm-ocr-raw-page-extractions",
        "run_id": run_id,
        "config_sha256": contract["config_sha256"],
        "pipeline_fingerprint": contract["pipeline_fingerprint"],
        "inventory_sha256": contract["inventory_sha256"],
        "summary": summary,
        "raw_responses": {
            "artifacts": files[0]["raw_response_artifact_count"],
            "bytes": files[0]["raw_response_bytes"],
        },
        "page_images": {
            "retained": retain_page_images,
            "artifacts": files[0]["raster_artifact_count"],
            "bytes": files[0]["raster_bytes"],
        },
        "files": files,
    }
    try:
        typed_manifest = DatasetManifest.model_validate_json(
            canonical_json_bytes(manifest_payload), strict=True
        )
    except Exception as error:
        raise DatasetPublicationError("generated dataset manifest is invalid") from error
    manifest = typed_manifest.model_dump(mode="json")
    manifest_bytes = canonical_json_bytes(manifest) + b"\n"
    manifest_path = dataset_root / "manifest.json"
    atomic_publish_bytes(manifest_path, manifest_bytes)
    try:
        persisted = await asyncio.to_thread(read_regular_file_bytes, manifest_path)
    except ArtifactReadError as error:
        raise DatasetPublicationError(
            f"published manifest cannot be read safely at {manifest_path}"
        ) from error
    if persisted != manifest_bytes:
        raise DatasetPublicationError(f"published manifest differs at {manifest_path}")
    return PublishedDataset(
        manifest=manifest,
        manifest_sha256=sha256_bytes(persisted),
        manifest_path=manifest_path,
    )
