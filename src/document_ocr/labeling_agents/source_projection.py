"""Immutable projection from a published OCR quality view to labeling work items."""

from __future__ import annotations

import json
import os
from collections.abc import Iterator, Mapping
from pathlib import Path, PurePosixPath
from typing import Annotated, Any, Literal, cast

from pydantic import ConfigDict, Field, StringConstraints, model_validator

from document_ocr.atomic import atomic_publish_bytes, read_regular_file_bytes
from document_ocr.hashing import canonical_json_bytes, sha256_bytes, sha256_file
from document_ocr.label_schemas.common import LabelSchemaModel
from document_ocr.labeling_agents.work_items import AgentWorkItem
from document_ocr.models import PageExtractionRecord, SourceObject

NonEmptyString = Annotated[str, StringConstraints(strip_whitespace=True, min_length=1)]
Sha256 = Annotated[str, StringConstraints(pattern=r"^[0-9a-f]{64}$")]
_DATASET_KIND = "ocr-conditioned-labeling-work-items"
_QUALITY_DATASET_KIND = "glm-ocr-quality-filtered-page-extractions"


class LabelingSourceProjectionError(RuntimeError):
    """A quality-filter artifact cannot be projected without losing provenance."""


class _ProjectionModel(LabelSchemaModel):
    model_config = ConfigDict(extra="forbid", frozen=True, strict=True, allow_inf_nan=False)


class _QualityPageReference(_ProjectionModel):
    extraction_id: NonEmptyString
    page_id: NonEmptyString
    page_index: Annotated[int, Field(ge=0)]
    page_number: Annotated[int, Field(gt=0)]
    raw_ocr_text_sha256: Sha256
    raw_response_path: NonEmptyString
    raster_path: NonEmptyString
    raster_sha256: Sha256

    @model_validator(mode="after")
    def number_matches_index(self) -> _QualityPageReference:
        if self.page_number != self.page_index + 1:
            raise ValueError("page_number must equal page_index + 1")
        return self


class _QualityDocument(_ProjectionModel):
    schema_version: Literal[1]
    run_id: NonEmptyString
    document_id: NonEmptyString
    document_page_count: Annotated[int, Field(gt=0)]
    source: SourceObject
    pages: tuple[_QualityPageReference, ...] = Field(min_length=1)

    @model_validator(mode="after")
    def provenance_is_consistent(self) -> _QualityDocument:
        if self.source.document_id != self.document_id:
            raise ValueError("source document_id differs from document_id")
        if len(self.pages) != self.document_page_count:
            raise ValueError("page-reference count differs from document_page_count")
        if [page.page_index for page in self.pages] != list(range(self.document_page_count)):
            raise ValueError("page references must be complete and ordered")
        return self


def _strict_json(payload: bytes, *, context: str) -> dict[str, Any]:
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
        decoded = json.loads(
            payload.decode("utf-8"),
            object_pairs_hook=reject_duplicates,
            parse_constant=reject_constant,
        )
    except (TypeError, UnicodeError, ValueError) as error:
        raise LabelingSourceProjectionError(f"invalid JSON: {context}") from error
    if not isinstance(decoded, dict):
        raise LabelingSourceProjectionError(f"JSON root must be an object: {context}")
    return cast(dict[str, Any], decoded)


def _canonical_directory(path: Path, *, context: str, create: bool = False) -> Path:
    if create:
        path.mkdir(parents=True, exist_ok=True)
    absolute = Path(os.path.abspath(path))
    try:
        resolved = path.resolve(strict=True)
    except OSError as error:
        raise LabelingSourceProjectionError(f"missing {context}: {path}") from error
    if resolved != absolute or not resolved.is_dir() or resolved.is_symlink():
        raise LabelingSourceProjectionError(f"{context} must be a canonical directory: {path}")
    return resolved


def _contained_file(root: Path, relative_value: str, *, context: str) -> Path:
    relative = PurePosixPath(relative_value)
    if relative.is_absolute() or any(part in {"", ".", ".."} for part in relative.parts):
        raise LabelingSourceProjectionError(f"unsafe {context} path: {relative_value}")
    unresolved = root / Path(relative)
    absolute = Path(os.path.abspath(unresolved))
    try:
        resolved = unresolved.resolve(strict=True)
    except OSError as error:
        raise LabelingSourceProjectionError(f"missing {context}: {unresolved}") from error
    if (
        resolved != absolute
        or not resolved.is_relative_to(root)
        or not resolved.is_file()
        or resolved.is_symlink()
    ):
        raise LabelingSourceProjectionError(
            f"{context} must be a canonical contained regular file: {unresolved}"
        )
    return resolved


def _jsonl_rows(path: Path) -> Iterator[tuple[int, dict[str, Any]]]:
    with path.open("rb") as stream:
        for line_number, line in enumerate(stream, start=1):
            if not line.rstrip(b"\r\n"):
                raise LabelingSourceProjectionError(f"blank JSONL row: {path}:{line_number}")
            yield line_number, _strict_json(line, context=f"{path}:{line_number}")


def _manifest_files(
    quality_root: Path, manifest: Mapping[str, Any]
) -> dict[str, tuple[Path, Mapping[str, Any]]]:
    rows = manifest.get("files")
    if not isinstance(rows, list):
        raise LabelingSourceProjectionError("quality-filter manifest files are absent")
    files: dict[str, tuple[Path, Mapping[str, Any]]] = {}
    for entry in rows:
        if not isinstance(entry, dict):
            raise LabelingSourceProjectionError("quality-filter manifest file entry is invalid")
        path_value = entry.get("path")
        expected_bytes = entry.get("bytes")
        expected_rows = entry.get("rows")
        expected_sha256 = entry.get("sha256")
        if (
            not isinstance(path_value, str)
            or not isinstance(expected_bytes, int)
            or expected_bytes < 0
            or not isinstance(expected_rows, int)
            or expected_rows < 0
            or not isinstance(expected_sha256, str)
        ):
            raise LabelingSourceProjectionError("quality-filter manifest file entry is invalid")
        if path_value in files:
            raise LabelingSourceProjectionError("quality-filter manifest paths are not unique")
        path = _contained_file(quality_root, path_value, context="quality-filter artifact")
        if path.stat().st_size != expected_bytes or sha256_file(path) != expected_sha256:
            raise LabelingSourceProjectionError(
                f"quality-filter artifact identity changed: {path_value}"
            )
        files[path_value] = (path, entry)
    required = {"eligible-documents.jsonl", "eligible-pages.jsonl"}
    if not required.issubset(files):
        raise LabelingSourceProjectionError(
            "quality-filter manifest omits eligible document/page artifacts"
        )
    return files


def _source_run_root(manifest: Mapping[str, Any], *, expected_run_id: str) -> Path:
    source = manifest.get("source")
    if not isinstance(source, dict) or not isinstance(source.get("run_directory"), str):
        raise LabelingSourceProjectionError("quality-filter source run directory is absent")
    root = _canonical_directory(Path(source["run_directory"]), context="source extraction run")
    source_files = source.get("files")
    if not isinstance(source_files, dict):
        raise LabelingSourceProjectionError("quality-filter source file identities are absent")
    for name in ("state.sqlite3", "inventory.jsonl", "run-provenance.json"):
        identity = source_files.get(name)
        if (
            not isinstance(identity, dict)
            or not isinstance(identity.get("bytes"), int)
            or not isinstance(identity.get("sha256"), str)
        ):
            raise LabelingSourceProjectionError(f"source identity is absent: {name}")
        path = _contained_file(root, name, context="source run artifact")
        if path.stat().st_size != identity["bytes"] or sha256_file(path) != identity["sha256"]:
            raise LabelingSourceProjectionError(f"source run artifact identity changed: {name}")
    provenance_path = _contained_file(root, "run-provenance.json", context="provenance")
    provenance = _strict_json(
        read_regular_file_bytes(provenance_path), context=str(provenance_path)
    )
    resolved_config = provenance.get("resolved_config")
    run = resolved_config.get("run") if isinstance(resolved_config, dict) else None
    if not isinstance(run, dict) or run.get("run_id") != expected_run_id:
        raise LabelingSourceProjectionError("quality-filter and source run IDs differ")
    return root


def _work_item(
    document: _QualityDocument,
    pages: tuple[PageExtractionRecord, ...],
) -> AgentWorkItem:
    if document.source.source_type != "local":
        raise LabelingSourceProjectionError(
            f"labeling requires a local source PDF: {document.document_id}"
        )
    local_path = document.source.local_canonical_path
    source_sha256 = document.source.source_sha256
    if local_path is None or source_sha256 is None:
        raise LabelingSourceProjectionError(
            f"local source provenance is incomplete: {document.document_id}"
        )
    pdf = Path(local_path)
    if not pdf.is_file() or sha256_file(pdf) != source_sha256:
        raise LabelingSourceProjectionError(
            f"local source PDF identity changed: {document.document_id}"
        )

    joined: list[str] = []
    page_references: list[dict[str, Any]] = []
    for expected, page in zip(document.pages, pages, strict=True):
        if (
            page.run_id != document.run_id
            or page.document_id != document.document_id
            or page.document_page_count != document.document_page_count
            or page.extraction_id != expected.extraction_id
            or page.page_id != expected.page_id
            or page.page_index != expected.page_index
            or page.page_number != expected.page_number
            or page.raw_ocr_text_sha256 != expected.raw_ocr_text_sha256
            or page.raw_response_path != expected.raw_response_path
            or page.raster_path != expected.raster_path
            or page.raster_sha256 != expected.raster_sha256
            or page.inference_finish_reason != "stop"
        ):
            raise LabelingSourceProjectionError(
                f"eligible document/page projection differs: {document.document_id} "
                f"page {expected.page_number}"
            )
        if sha256_bytes(page.raw_ocr_text.encode("utf-8")) != page.raw_ocr_text_sha256:
            raise LabelingSourceProjectionError(
                f"raw OCR text identity changed: {page.extraction_id}"
            )
        if page.raster_path is None:
            raise LabelingSourceProjectionError(f"retained page raster is absent: {page.page_id}")
        joined.append(f"--- PAGE {page.page_number} ---\n{page.raw_ocr_text}")
        page_references.append(
            {
                "pageIndex": page.page_index,
                "pageNumber": page.page_number,
                "pageId": page.page_id,
                "extractionId": page.extraction_id,
                "rawOcrTextSha256": page.raw_ocr_text_sha256,
                "rawResponsePath": page.raw_response_path,
                "rawResponseSha256": page.raw_response_sha256,
                "rasterPath": page.raster_path,
                "rasterSha256": page.raster_sha256,
            }
        )
    joined_text = "\n\n".join(joined)
    return AgentWorkItem.model_validate_json(
        canonical_json_bytes(
            {
                "source": {
                    "documentId": document.document_id,
                    "extractionRunId": document.run_id,
                    "sourceUri": document.source.source_uri,
                    "localCanonicalPath": local_path,
                    "sourceSha256": source_sha256,
                    "documentPageCount": document.document_page_count,
                    "joinedRawTextSha256": sha256_bytes(joined_text.encode("utf-8")),
                    "pages": page_references,
                },
                "joinedRawText": joined_text,
            }
        ),
        strict=True,
    )


def project_quality_filter_work_items(
    *,
    quality_filter_root: Path,
    quality_filter_manifest_sha256: str,
    output_root: Path,
) -> tuple[Path, int, int, str]:
    """Publish exact complete-document work items and return manifest metadata."""

    quality_root = _canonical_directory(quality_filter_root, context="quality-filter root")
    manifest_path = _contained_file(quality_root, "manifest.json", context="quality manifest")
    manifest_payload = read_regular_file_bytes(manifest_path)
    if sha256_bytes(manifest_payload) != quality_filter_manifest_sha256:
        raise LabelingSourceProjectionError("quality-filter manifest SHA-256 mismatch")
    manifest = _strict_json(manifest_payload, context=str(manifest_path))
    if (
        manifest.get("schema_version") != 1
        or manifest.get("dataset_kind") != _QUALITY_DATASET_KIND
        or manifest.get("publication") != "manifest_last_immutable"
        or not isinstance(manifest.get("run_id"), str)
    ):
        raise LabelingSourceProjectionError("quality-filter manifest identity is invalid")
    run_id = cast(str, manifest["run_id"])
    source_run_root = _source_run_root(manifest, expected_run_id=run_id)
    files = _manifest_files(quality_root, manifest)
    document_path, document_entry = files["eligible-documents.jsonl"]
    page_path, page_entry = files["eligible-pages.jsonl"]

    output = _canonical_directory(output_root, context="labeling source output", create=True)
    if output == Path(output.anchor):
        raise LabelingSourceProjectionError("labeling source output must not be filesystem root")
    work_item_root = _canonical_directory(
        output / "work-items", context="labeling work-item output", create=True
    )
    page_iterator = _jsonl_rows(page_path)
    inventory_rows: list[dict[str, Any]] = []
    selection_rows: list[dict[str, Any]] = []
    total_pages = 0
    for document_line, raw_document in _jsonl_rows(document_path):
        try:
            document = _QualityDocument.model_validate_json(
                canonical_json_bytes(raw_document), strict=True
            )
        except ValueError as error:
            raise LabelingSourceProjectionError(
                f"eligible document failed validation: {document_path}:{document_line}: {error}"
            ) from error
        page_records: list[PageExtractionRecord] = []
        for _ in range(document.document_page_count):
            try:
                page_line, raw_page = next(page_iterator)
            except StopIteration as error:
                raise LabelingSourceProjectionError(
                    f"eligible pages ended inside document {document.document_id}"
                ) from error
            try:
                page_records.append(
                    PageExtractionRecord.model_validate_json(
                        canonical_json_bytes(raw_page), strict=True
                    )
                )
            except ValueError as error:
                raise LabelingSourceProjectionError(
                    f"eligible page failed validation: {page_path}:{page_line}: {error}"
                ) from error
        item = _work_item(document, tuple(page_records))
        payload = canonical_json_bytes(item.model_dump(mode="json")) + b"\n"
        item_sha256 = sha256_bytes(payload)
        relative_path = f"work-items/{document.document_id}.json"
        atomic_publish_bytes(work_item_root / f"{document.document_id}.json", payload)
        inventory_rows.append(
            {
                "documentId": document.document_id,
                "documentPageCount": document.document_page_count,
                "joinedRawTextSha256": item.source.joinedRawTextSha256,
                "path": relative_path,
                "sourceSha256": item.source.sourceSha256,
                "workItemSha256": item_sha256,
            }
        )
        selection_rows.append(
            {
                "documentId": document.document_id,
                "joinedRawTextSha256": item.source.joinedRawTextSha256,
                "qualityFilterManifestSha256": quality_filter_manifest_sha256,
                "qualityFilterRunId": run_id,
                "sourceSha256": item.source.sourceSha256,
                "workItemSha256": item_sha256,
            }
        )
        total_pages += document.document_page_count
    try:
        extra_page = next(page_iterator)
    except StopIteration:
        extra_page = None
    if extra_page is not None:
        raise LabelingSourceProjectionError(
            f"eligible page has no parent document: {page_path}:{extra_page[0]}"
        )

    expected_documents = document_entry.get("rows")
    expected_pages = page_entry.get("rows")
    if len(inventory_rows) != expected_documents or total_pages != expected_pages:
        raise LabelingSourceProjectionError(
            "projected work-item counts differ from quality-filter manifest"
        )
    inventory_payload = b"".join(canonical_json_bytes(row) + b"\n" for row in inventory_rows)
    selection_payload = b"".join(canonical_json_bytes(row) + b"\n" for row in selection_rows)
    output_inventory = output / "inventory.jsonl"
    output_selection = output / "selection-records.jsonl"
    atomic_publish_bytes(output_inventory, inventory_payload)
    atomic_publish_bytes(output_selection, selection_payload)
    output_manifest = {
        "schema_version": 1,
        "dataset_kind": _DATASET_KIND,
        "source": {
            "quality_filter_root": str(quality_root),
            "quality_filter_manifest_sha256": quality_filter_manifest_sha256,
            "run_id": run_id,
            "run_directory": str(source_run_root),
        },
        "documents": len(inventory_rows),
        "pages": total_pages,
        "files": [
            {
                "path": "inventory.jsonl",
                "rows": len(inventory_rows),
                "bytes": len(inventory_payload),
                "sha256": sha256_bytes(inventory_payload),
            },
            {
                "path": "selection-records.jsonl",
                "rows": len(selection_rows),
                "bytes": len(selection_payload),
                "sha256": sha256_bytes(selection_payload),
            },
        ],
        "work_items": {
            "root": "work-items",
            "files": len(inventory_rows),
            "inventory_sha256": sha256_bytes(inventory_payload),
        },
        "publication": "manifest_last_immutable",
    }
    output_manifest_payload = canonical_json_bytes(output_manifest) + b"\n"
    output_manifest_path = output / "manifest.json"
    atomic_publish_bytes(output_manifest_path, output_manifest_payload)
    return (
        output_manifest_path,
        len(inventory_rows),
        total_pages,
        sha256_bytes(output_manifest_payload),
    )
