"""Content-pinned local snapshots of classifier-selected S3 PDF corpora.

The source bucket used by this project is not versioned.  This module closes
that reproducibility gap by conditionally downloading the audited S3 objects,
hashing the complete PDF bytes, retaining exact classification evidence, and
publishing a manifest-last local snapshot.  A completed snapshot is verified
without contacting S3 and can be consumed by the existing local-source path.
"""

from __future__ import annotations

import hashlib
import json
import multiprocessing
import os
import stat
import tempfile
from collections import defaultdict
from collections.abc import Callable, Iterable, Mapping
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
from contextlib import suppress
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path, PurePosixPath
from typing import Any, Literal

import boto3
from botocore.config import Config as BotoClientConfig
from pydantic import (
    AwareDatetime,
    BaseModel,
    ConfigDict,
    Field,
    field_validator,
    model_validator,
)

from document_ocr.atomic import (
    AtomicConflictError,
    atomic_publish_bytes,
    atomic_publish_json,
    read_regular_file_bytes,
)
from document_ocr.config import S3LocalSnapshotConfig
from document_ocr.hashing import canonical_json_bytes, sha256_bytes

_PDF_HEADER_SCAN_BYTES = 1024
_CLASSIFICATION_MANIFEST_MAX_BYTES = 256 * 1024 * 1024
_CHECKSUM_RESPONSE_FIELDS = {
    "s3_checksum_crc32": "ChecksumCRC32",
    "s3_checksum_crc32c": "ChecksumCRC32C",
    "s3_checksum_sha1": "ChecksumSHA1",
    "s3_checksum_sha256": "ChecksumSHA256",
    "s3_checksum_crc64nvme": "ChecksumCRC64NVME",
    "s3_checksum_type": "ChecksumType",
}


class _FrozenRecord(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, strict=True, allow_inf_nan=False)


class CandidateSelectionRecord(_FrozenRecord):
    """The exact canonical row used by the audited selection digest."""

    source_bucket: str = Field(min_length=1)
    source_key: str = Field(min_length=1)
    source_size_bytes: int = Field(gt=0)
    source_etag: str = Field(min_length=1)
    source_last_modified: str = Field(min_length=1)
    document_type: str = Field(min_length=1, max_length=32, pattern=r"^[a-z][a-z0-9_-]*$")
    classification_doc_id: str = Field(min_length=1)
    classification_manifest_key: str = Field(min_length=1)
    document_page_count: int = Field(gt=0)

    @field_validator("source_etag")
    @classmethod
    def etag_is_quoted(cls, value: str) -> str:
        if len(value) < 3 or not value.startswith('"') or not value.endswith('"'):
            raise ValueError("source_etag must retain S3's quoted ETag form")
        return value

    @field_validator("source_last_modified")
    @classmethod
    def last_modified_is_canonical_utc(cls, value: str) -> str:
        try:
            parsed = datetime.fromisoformat(value)
        except ValueError as error:
            raise ValueError("source_last_modified must be an ISO-8601 datetime") from error
        if parsed.tzinfo is None or parsed.utcoffset() is None:
            raise ValueError("source_last_modified must be timezone-aware")
        if value != parsed.astimezone(UTC).isoformat():
            raise ValueError("source_last_modified must be a canonical UTC ISO-8601 datetime")
        return value


class DownloadStateRecord(CandidateSelectionRecord):
    """Durable per-object commit marker used for crash-safe download resume."""

    schema_version: Literal[1]
    classification_manifest_sha256: str
    local_relative_key: str = Field(min_length=1)
    snapshot_relative_path: str = Field(min_length=1)
    source_sha256: str
    s3_checksum_crc32: str | None = None
    s3_checksum_crc32c: str | None = None
    s3_checksum_sha1: str | None = None
    s3_checksum_sha256: str | None = None
    s3_checksum_crc64nvme: str | None = None
    s3_checksum_type: Literal["COMPOSITE", "FULL_OBJECT"] | None = None

    @field_validator("classification_manifest_sha256", "source_sha256")
    @classmethod
    def hashes_are_sha256(cls, value: str) -> str:
        if not _is_sha256(value):
            raise ValueError("hash must be a lowercase 64-character SHA-256")
        return value

    @field_validator(
        "s3_checksum_crc32",
        "s3_checksum_crc32c",
        "s3_checksum_sha1",
        "s3_checksum_sha256",
        "s3_checksum_crc64nvme",
    )
    @classmethod
    def checksums_are_nonempty(cls, value: str | None) -> str | None:
        if value == "":
            raise ValueError("S3 checksum values must not be empty")
        return value

    @model_validator(mode="after")
    def local_paths_are_safe_and_consistent(self) -> DownloadStateRecord:
        local = _validated_relative_path(self.local_relative_key, "local_relative_key")
        snapshot = _validated_relative_path(
            self.snapshot_relative_path,
            "snapshot_relative_path",
        )
        expected = PurePosixPath("files", self.document_type, *local.parts)
        if snapshot != expected:
            raise ValueError(
                "snapshot_relative_path must equal files/<document_type>/<local_relative_key>"
            )
        return self


class SnapshotFileRecord(DownloadStateRecord):
    """One immutable local PDF plus its remote and classification provenance."""

    actual_page_count: int = Field(gt=0)
    extraction_status: Literal["ready", "quarantined"]
    extraction_relative_path: str | None = None
    quarantine_reason: str | None = None

    @model_validator(mode="after")
    def extraction_path_matches_status(self) -> SnapshotFileRecord:
        local = _validated_relative_path(self.local_relative_key, "local_relative_key")
        if self.extraction_status == "ready":
            if self.quarantine_reason is not None:
                raise ValueError("ready records must not have a quarantine reason")
            if self.extraction_relative_path is None:
                raise ValueError("ready records require extraction_relative_path")
            extraction = _validated_relative_path(
                self.extraction_relative_path,
                "extraction_relative_path",
            )
            expected = PurePosixPath("extraction", self.document_type, *local.parts)
            if extraction != expected:
                raise ValueError(
                    "extraction_relative_path must equal "
                    "extraction/<document_type>/<local_relative_key>"
                )
            return self
        if self.extraction_relative_path is not None:
            raise ValueError("quarantined records must not have extraction_relative_path")
        if self.quarantine_reason is None or not self.quarantine_reason.strip():
            raise ValueError("quarantined records require a reason")
        return self


class StoredClassificationManifest(_FrozenRecord):
    key: str = Field(min_length=1)
    sha256: str
    snapshot_relative_path: str = Field(min_length=1)
    source_etag: str = Field(min_length=1)
    source_last_modified: AwareDatetime
    source_size_bytes: int = Field(gt=0)

    @field_validator("sha256")
    @classmethod
    def hash_is_sha256(cls, value: str) -> str:
        if not _is_sha256(value):
            raise ValueError("sha256 must be a lowercase 64-character SHA-256")
        return value


class SnapshotTypeSummary(_FrozenRecord):
    documents: int = Field(gt=0)
    source_bytes: int = Field(gt=0)
    classified_pages: int = Field(gt=0)
    actual_pages: int = Field(gt=0)
    extraction_documents: int = Field(ge=0)
    extraction_pages: int = Field(ge=0)
    quarantined_documents: int = Field(ge=0)

    @model_validator(mode="after")
    def extraction_and_quarantine_partition_documents(self) -> SnapshotTypeSummary:
        if self.documents != self.extraction_documents + self.quarantined_documents:
            raise ValueError("documents must equal extraction_documents plus quarantined_documents")
        if self.extraction_pages > self.actual_pages:
            raise ValueError("extraction_pages must not exceed actual_pages")
        return self


class SnapshotSelectionAudit(_FrozenRecord):
    raw_pdf_count: int = Field(gt=0)
    raw_pdf_bytes: int = Field(gt=0)
    raw_pdfs_without_filtered_manifest_evidence: int = Field(ge=0)
    filtered_manifest_documents_without_raw_pdf: int = Field(ge=0)


class SnapshotCommit(_FrozenRecord):
    """Manifest-last marker proving that the local snapshot is complete."""

    schema_version: Literal[1]
    bucket: str = Field(min_length=1)
    raw_prefix: str = Field(min_length=1)
    region: str = Field(min_length=1)
    document_types: list[str] = Field(min_length=1)
    selection_path: Literal["selection.jsonl"]
    selection_sha256: str
    manifest_path: Literal["manifest.jsonl"]
    manifest_sha256: str
    classification_manifests: list[StoredClassificationManifest] = Field(min_length=1)
    selection_audit: SnapshotSelectionAudit
    by_document_type: dict[str, SnapshotTypeSummary]
    document_count: int = Field(gt=0)
    source_bytes: int = Field(gt=0)
    classified_pages: int = Field(gt=0)
    actual_pages: int = Field(gt=0)
    extraction_document_count: int = Field(gt=0)
    extraction_pages: int = Field(gt=0)
    quarantined_document_count: int = Field(ge=0)

    @field_validator("selection_sha256", "manifest_sha256")
    @classmethod
    def hashes_are_sha256(cls, value: str) -> str:
        if not _is_sha256(value):
            raise ValueError("hash must be a lowercase 64-character SHA-256")
        return value

    @model_validator(mode="after")
    def totals_match_type_summaries(self) -> SnapshotCommit:
        if set(self.by_document_type) != set(self.document_types):
            raise ValueError("by_document_type must exactly cover document_types")
        if self.document_count != sum(item.documents for item in self.by_document_type.values()):
            raise ValueError("document_count does not match type summaries")
        if self.source_bytes != sum(item.source_bytes for item in self.by_document_type.values()):
            raise ValueError("source_bytes does not match type summaries")
        if self.classified_pages != sum(
            item.classified_pages for item in self.by_document_type.values()
        ):
            raise ValueError("classified_pages does not match type summaries")
        if self.actual_pages != sum(item.actual_pages for item in self.by_document_type.values()):
            raise ValueError("actual_pages does not match type summaries")
        if self.extraction_document_count != sum(
            item.extraction_documents for item in self.by_document_type.values()
        ):
            raise ValueError("extraction_document_count does not match type summaries")
        if self.extraction_pages != sum(
            item.extraction_pages for item in self.by_document_type.values()
        ):
            raise ValueError("extraction_pages does not match type summaries")
        if self.quarantined_document_count != sum(
            item.quarantined_documents for item in self.by_document_type.values()
        ):
            raise ValueError("quarantined_document_count does not match type summaries")
        if self.document_count != (
            self.extraction_document_count + self.quarantined_document_count
        ):
            raise ValueError(
                "document_count must equal extraction plus quarantined document counts"
            )
        if self.document_count > self.selection_audit.raw_pdf_count:
            raise ValueError("document_count exceeds the audited raw PDF count")
        if self.source_bytes > self.selection_audit.raw_pdf_bytes:
            raise ValueError("source_bytes exceeds the audited raw PDF bytes")
        if (
            self.selection_audit.raw_pdfs_without_filtered_manifest_evidence
            > self.selection_audit.raw_pdf_count
        ):
            raise ValueError("raw PDFs without evidence exceed the audited raw PDF count")
        return self


class SnapshotError(RuntimeError):
    """Base class for explicit local-snapshot failures."""


class SnapshotSelectionError(SnapshotError):
    """The S3 listing and classifier evidence do not form the pinned selection."""


class SnapshotRemoteChangedError(SnapshotError):
    """An unversioned S3 object changed relative to the captured selection."""


class SnapshotIntegrityError(SnapshotError):
    """Local snapshot bytes or metadata violate the immutable contract."""


@dataclass(frozen=True, slots=True)
class _ClassifierDocument:
    original_pdf_path: str
    final_label: str
    page_count: int


@dataclass(frozen=True, slots=True)
class _FetchedManifest:
    config_key: str
    local_name: str
    sha256: str
    payload: bytes
    source_etag: str
    source_last_modified: datetime
    source_size_bytes: int


@dataclass(frozen=True, slots=True)
class _SelectionPlan:
    candidates: tuple[CandidateSelectionRecord, ...]
    selection_payload: bytes
    selection_sha256: str
    raw_pdf_count: int
    raw_pdf_bytes: int
    raw_pdfs_without_filtered_manifest_evidence: int
    filtered_manifest_documents_without_raw_pdf: int


@dataclass(frozen=True, slots=True)
class SnapshotResult:
    root: Path
    commit: SnapshotCommit
    created: bool


def _is_sha256(value: str) -> bool:
    return len(value) == 64 and all(character in "0123456789abcdef" for character in value)


def _validated_relative_path(value: str, field: str) -> PurePosixPath:
    if "\\" in value or "\x00" in value:
        raise ValueError(f"{field} must be a relative POSIX path")
    path = PurePosixPath(value)
    if path.is_absolute() or not path.parts or any(part in {"", ".", ".."} for part in path.parts):
        raise ValueError(f"{field} must be a safe relative POSIX path")
    return path


def _utc_iso(value: Any, field: str, locator: str) -> str:
    if not isinstance(value, datetime) or value.tzinfo is None or value.utcoffset() is None:
        raise SnapshotSelectionError(f"{field} must be timezone-aware: {locator}")
    return value.astimezone(UTC).isoformat()


def _canonical_jsonl(records: Iterable[BaseModel]) -> bytes:
    return b"".join(
        canonical_json_bytes(record.model_dump(mode="json")) + b"\n" for record in records
    )


def _digest_sidecar(digest: str, name: str) -> bytes:
    return f"{digest}  {name}\n".encode("ascii")


def _candidate_from_state(record: DownloadStateRecord) -> CandidateSelectionRecord:
    return CandidateSelectionRecord.model_validate(
        {name: getattr(record, name) for name in CandidateSelectionRecord.model_fields},
        strict=True,
    )


def _candidate_from_file(record: SnapshotFileRecord) -> CandidateSelectionRecord:
    return CandidateSelectionRecord.model_validate(
        {name: getattr(record, name) for name in CandidateSelectionRecord.model_fields},
        strict=True,
    )


def _safe_snapshot_directory(path: Path, *, create: bool) -> Path:
    absolute = Path(os.path.abspath(path))
    if not absolute.is_absolute() or absolute == Path(absolute.anchor):
        raise SnapshotIntegrityError(f"snapshot directory is unsafe: {path}")
    required_flags = ("O_CLOEXEC", "O_NOFOLLOW", "O_DIRECTORY")
    missing = [name for name in required_flags if not hasattr(os, name)]
    if missing:
        raise SnapshotIntegrityError(
            "safe snapshot traversal requires operating-system flags: " + ", ".join(missing)
        )
    flags = os.O_RDONLY | os.O_CLOEXEC | os.O_NOFOLLOW | os.O_DIRECTORY
    directory_fd = os.open("/", flags)
    try:
        for component in absolute.parts[1:]:
            try:
                next_fd = os.open(component, flags, dir_fd=directory_fd)
            except FileNotFoundError:
                if not create:
                    raise SnapshotIntegrityError(
                        f"snapshot directory does not exist: {absolute}"
                    ) from None
                try:
                    with suppress(FileExistsError):
                        os.mkdir(component, dir_fd=directory_fd)
                    # Concurrent workers may create the same safe ancestor
                    # between the failed open and the mkdir above.
                    next_fd = os.open(component, flags, dir_fd=directory_fd)
                except OSError as error:
                    raise SnapshotIntegrityError(
                        f"snapshot directory cannot be created safely: {absolute}"
                    ) from error
            except OSError as error:
                raise SnapshotIntegrityError(
                    f"snapshot directory must not traverse symbolic links: {absolute}"
                ) from error
            os.close(directory_fd)
            directory_fd = next_fd
        if not stat.S_ISDIR(os.fstat(directory_fd).st_mode):
            raise SnapshotIntegrityError(f"snapshot path is not a directory: {absolute}")
    finally:
        os.close(directory_fd)
    return absolute


def _open_absolute_no_symlinks(path: Path) -> int:
    absolute = Path(os.path.abspath(path))
    if path != absolute or not absolute.is_absolute():
        raise SnapshotIntegrityError(f"snapshot file path is not canonical: {path}")
    required_flags = ("O_CLOEXEC", "O_NOFOLLOW", "O_DIRECTORY")
    missing = [name for name in required_flags if not hasattr(os, name)]
    if missing:
        raise SnapshotIntegrityError(
            "safe snapshot reads require operating-system flags: " + ", ".join(missing)
        )
    directory_flags = os.O_RDONLY | os.O_CLOEXEC | os.O_NOFOLLOW | os.O_DIRECTORY
    file_flags = os.O_RDONLY | os.O_CLOEXEC | os.O_NOFOLLOW
    directory_fd = os.open("/", directory_flags)
    try:
        for component in absolute.parts[1:-1]:
            next_fd = os.open(component, directory_flags, dir_fd=directory_fd)
            os.close(directory_fd)
            directory_fd = next_fd
        return os.open(absolute.name, file_flags, dir_fd=directory_fd)
    except OSError as error:
        raise SnapshotIntegrityError(
            f"snapshot file cannot be opened without symbolic-link traversal: {absolute}"
        ) from error
    finally:
        os.close(directory_fd)


def _hash_local_pdf(path: Path, *, max_pdf_bytes: int) -> tuple[str, int]:
    descriptor = _open_absolute_no_symlinks(path)
    digest = hashlib.sha256()
    header = bytearray()
    size_bytes = 0
    try:
        before = os.fstat(descriptor)
        if not stat.S_ISREG(before.st_mode):
            raise SnapshotIntegrityError(f"snapshot PDF is not a regular file: {path}")
        while True:
            chunk = os.read(descriptor, 1024 * 1024)
            if not chunk:
                break
            size_bytes += len(chunk)
            if size_bytes > max_pdf_bytes:
                raise SnapshotIntegrityError(
                    f"snapshot PDF exceeds max_pdf_bytes ({size_bytes} > {max_pdf_bytes}): {path}"
                )
            digest.update(chunk)
            if len(header) < _PDF_HEADER_SCAN_BYTES:
                remaining = _PDF_HEADER_SCAN_BYTES - len(header)
                header.extend(chunk[:remaining])
        after = os.fstat(descriptor)
    finally:
        os.close(descriptor)
    if size_bytes <= 0:
        raise SnapshotIntegrityError(f"snapshot PDF is empty: {path}")
    if b"%PDF-" not in header:
        raise SnapshotIntegrityError(
            f"snapshot file has no PDF header in its first {_PDF_HEADER_SCAN_BYTES} bytes: {path}"
        )
    before_identity = (before.st_dev, before.st_ino, before.st_size, before.st_mtime_ns)
    after_identity = (after.st_dev, after.st_ino, after.st_size, after.st_mtime_ns)
    if before_identity != after_identity or size_bytes != before.st_size:
        raise SnapshotIntegrityError(f"snapshot PDF changed while hashing: {path}")
    return digest.hexdigest(), size_bytes


def _required_response_value(
    response: Mapping[str, Any],
    field: str,
    expected_type: type[Any],
    locator: str,
) -> Any:
    value = response.get(field)
    if not isinstance(value, expected_type) or (expected_type is int and isinstance(value, bool)):
        raise SnapshotRemoteChangedError(
            f"S3 response field {field} has an invalid type: {locator}"
        )
    if isinstance(value, str) and not value:
        raise SnapshotRemoteChangedError(f"S3 response field {field} is empty: {locator}")
    return value


def _read_response_body(
    response: Mapping[str, Any],
    *,
    expected_size: int,
    maximum_size: int,
    chunk_size: int,
    locator: str,
) -> bytes:
    body = response.get("Body")
    if body is None or not callable(getattr(body, "read", None)):
        raise SnapshotRemoteChangedError(f"S3 response has no readable body: {locator}")
    payload = bytearray()
    try:
        while True:
            chunk = body.read(chunk_size)
            if not isinstance(chunk, bytes):
                raise SnapshotRemoteChangedError(f"S3 body returned non-bytes data: {locator}")
            if not chunk:
                break
            payload.extend(chunk)
            if len(payload) > maximum_size:
                raise SnapshotRemoteChangedError(
                    f"S3 object exceeds the configured metadata bound: {locator}"
                )
    finally:
        close = getattr(body, "close", None)
        if callable(close):
            close()
    if len(payload) != expected_size:
        raise SnapshotRemoteChangedError(
            f"S3 body length changed ({len(payload)} != {expected_size}): {locator}"
        )
    return bytes(payload)


def _fetch_classification_manifest(
    *,
    client: Any,
    config: S3LocalSnapshotConfig,
    manifest_key: str,
    expected_sha256: str,
    local_name: str,
) -> _FetchedManifest:
    locator = f"s3://{config.bucket}/{manifest_key}"
    try:
        head = client.head_object(Bucket=config.bucket, Key=manifest_key, ChecksumMode="ENABLED")
    except Exception as error:
        raise SnapshotRemoteChangedError(
            f"failed to inspect classification manifest: {locator}"
        ) from error
    if not isinstance(head, Mapping):
        raise SnapshotRemoteChangedError(f"S3 HEAD response is not a mapping: {locator}")
    size_bytes = _required_response_value(head, "ContentLength", int, locator)
    etag = _required_response_value(head, "ETag", str, locator)
    last_modified = _required_response_value(head, "LastModified", datetime, locator)
    if size_bytes <= 0 or size_bytes > _CLASSIFICATION_MANIFEST_MAX_BYTES:
        raise SnapshotRemoteChangedError(
            f"classification manifest size is outside the allowed bound: {locator}"
        )
    try:
        response = client.get_object(
            Bucket=config.bucket,
            Key=manifest_key,
            IfMatch=etag,
            ChecksumMode="ENABLED",
        )
    except Exception as error:
        raise SnapshotRemoteChangedError(
            f"failed conditional GET for classification manifest: {locator}"
        ) from error
    if not isinstance(response, Mapping):
        raise SnapshotRemoteChangedError(f"S3 GET response is not a mapping: {locator}")
    if _required_response_value(response, "ContentLength", int, locator) != size_bytes:
        raise SnapshotRemoteChangedError(f"classification manifest size changed: {locator}")
    if _required_response_value(response, "ETag", str, locator) != etag:
        raise SnapshotRemoteChangedError(f"classification manifest ETag changed: {locator}")
    get_last_modified = _required_response_value(response, "LastModified", datetime, locator)
    if _utc_iso(get_last_modified, "LastModified", locator) != _utc_iso(
        last_modified,
        "LastModified",
        locator,
    ):
        raise SnapshotRemoteChangedError(f"classification manifest LastModified changed: {locator}")
    payload = _read_response_body(
        response,
        expected_size=size_bytes,
        maximum_size=_CLASSIFICATION_MANIFEST_MAX_BYTES,
        chunk_size=config.download.chunk_size_bytes,
        locator=locator,
    )
    digest = sha256_bytes(payload)
    if digest != expected_sha256:
        raise SnapshotRemoteChangedError(
            f"classification manifest SHA-256 changed ({digest} != {expected_sha256}): {locator}"
        )
    return _FetchedManifest(
        config_key=manifest_key,
        local_name=local_name,
        sha256=digest,
        payload=payload,
        source_etag=etag,
        source_last_modified=last_modified.astimezone(UTC),
        source_size_bytes=size_bytes,
    )


def _parse_classification_manifest(
    manifest: _FetchedManifest,
) -> dict[str, _ClassifierDocument]:
    grouped: dict[str, dict[str, Any]] = {}
    if not manifest.payload or not manifest.payload.endswith(b"\n"):
        raise SnapshotSelectionError(
            f"classification manifest must be non-empty newline-delimited JSON: "
            f"{manifest.config_key}"
        )
    for line_number, line in enumerate(manifest.payload.splitlines(), start=1):
        try:
            row = json.loads(line)
        except (UnicodeDecodeError, json.JSONDecodeError) as error:
            raise SnapshotSelectionError(
                f"invalid JSON at {manifest.config_key}:{line_number}"
            ) from error
        if not isinstance(row, Mapping):
            raise SnapshotSelectionError(
                f"classification row is not an object at {manifest.config_key}:{line_number}"
            )
        doc_id = row.get("doc_id")
        original_pdf_path = row.get("original_pdf_path")
        final_label = row.get("final_classification_label")
        page_index = row.get("page_index")
        is_first_page = row.get("is_first_page")
        if not isinstance(doc_id, str) or not doc_id:
            raise SnapshotSelectionError(
                f"classification doc_id is invalid at {manifest.config_key}:{line_number}"
            )
        if not isinstance(original_pdf_path, str) or not original_pdf_path:
            raise SnapshotSelectionError(
                f"original_pdf_path is invalid at {manifest.config_key}:{line_number}"
            )
        if not isinstance(final_label, str) or not final_label:
            raise SnapshotSelectionError(
                f"final_classification_label is invalid at {manifest.config_key}:{line_number}"
            )
        if not isinstance(page_index, int) or isinstance(page_index, bool) or page_index < 0:
            raise SnapshotSelectionError(
                f"page_index is invalid at {manifest.config_key}:{line_number}"
            )
        if not isinstance(is_first_page, bool):
            raise SnapshotSelectionError(
                f"is_first_page is invalid at {manifest.config_key}:{line_number}"
            )
        aggregate = grouped.setdefault(
            doc_id,
            {
                "original_pdf_path": original_pdf_path,
                "final_label": final_label,
                "page_indices": [],
                "first_page_indices": [],
            },
        )
        if aggregate["original_pdf_path"] != original_pdf_path:
            raise SnapshotSelectionError(
                f"classification document has conflicting original_pdf_path: "
                f"{manifest.config_key}:{doc_id}"
            )
        if aggregate["final_label"] != final_label:
            raise SnapshotSelectionError(
                f"classification document has conflicting final label: "
                f"{manifest.config_key}:{doc_id}"
            )
        aggregate["page_indices"].append(page_index)
        if is_first_page:
            aggregate["first_page_indices"].append(page_index)

    documents: dict[str, _ClassifierDocument] = {}
    for doc_id, aggregate in grouped.items():
        page_indices = sorted(aggregate["page_indices"])
        if page_indices != list(range(len(page_indices))):
            raise SnapshotSelectionError(
                f"classification page indices are not contiguous and unique: "
                f"{manifest.config_key}:{doc_id}"
            )
        if aggregate["first_page_indices"] != [0]:
            raise SnapshotSelectionError(
                f"classification document must have exactly one first page at index 0: "
                f"{manifest.config_key}:{doc_id}"
            )
        documents[doc_id] = _ClassifierDocument(
            original_pdf_path=aggregate["original_pdf_path"],
            final_label=aggregate["final_label"],
            page_count=len(page_indices),
        )
    if not documents:
        raise SnapshotSelectionError(
            f"classification manifest contains no documents: {manifest.config_key}"
        )
    return documents


def _list_raw_pdfs(
    *,
    client: Any,
    config: S3LocalSnapshotConfig,
) -> tuple[dict[str, Mapping[str, Any]], int, int]:
    try:
        paginator = client.get_paginator("list_objects_v2")
        pages = paginator.paginate(Bucket=config.bucket, Prefix=config.raw_prefix)
    except Exception as error:
        raise SnapshotSelectionError(
            f"failed to list s3://{config.bucket}/{config.raw_prefix}"
        ) from error
    by_basename: dict[str, Mapping[str, Any]] = {}
    total_bytes = 0
    count = 0
    for page in pages:
        if not isinstance(page, Mapping):
            raise SnapshotSelectionError("S3 listing page is not a mapping")
        contents = page.get("Contents", [])
        if not isinstance(contents, list):
            raise SnapshotSelectionError("S3 listing Contents is not a list")
        for item in contents:
            if not isinstance(item, Mapping):
                raise SnapshotSelectionError("S3 listing item is not a mapping")
            key = item.get("Key")
            if not isinstance(key, str) or not key.startswith(config.raw_prefix):
                raise SnapshotSelectionError("S3 listing returned an invalid key")
            if PurePosixPath(key).suffix.lower() != ".pdf":
                continue
            basename = PurePosixPath(key).name
            if basename in by_basename:
                prior = by_basename[basename].get("Key")
                raise SnapshotSelectionError(
                    f"raw PDF basenames are not unique: {prior!r}, {key!r}"
                )
            size = item.get("Size")
            if not isinstance(size, int) or isinstance(size, bool) or size <= 0:
                raise SnapshotSelectionError(
                    f"raw PDF has invalid Size: s3://{config.bucket}/{key}"
                )
            if size > config.download.max_pdf_bytes:
                raise SnapshotSelectionError(
                    f"raw PDF exceeds max_pdf_bytes ({size} > "
                    f"{config.download.max_pdf_bytes}): s3://{config.bucket}/{key}"
                )
            etag = item.get("ETag")
            if not isinstance(etag, str) or not etag:
                raise SnapshotSelectionError(
                    f"raw PDF has invalid ETag: s3://{config.bucket}/{key}"
                )
            _utc_iso(item.get("LastModified"), "LastModified", f"s3://{config.bucket}/{key}")
            by_basename[basename] = item
            count += 1
            total_bytes += size
    if not by_basename:
        raise SnapshotSelectionError(
            f"no PDFs found under s3://{config.bucket}/{config.raw_prefix}"
        )
    return by_basename, count, total_bytes


def _build_selection_plan(
    *,
    config: S3LocalSnapshotConfig,
    client: Any,
    manifests: tuple[_FetchedManifest, ...],
) -> _SelectionPlan:
    raw_by_basename, raw_pdf_count, raw_pdf_bytes = _list_raw_pdfs(
        client=client,
        config=config,
    )
    selected_labels = set(config.classification_label_mapping)
    matched_raw_keys: set[str] = set()
    selected_raw_keys: set[str] = set()
    filtered_manifest_documents_without_raw_pdf = 0
    candidates: list[CandidateSelectionRecord] = []
    for manifest in manifests:
        documents = _parse_classification_manifest(manifest)
        for doc_id, document in documents.items():
            basename = PurePosixPath(document.original_pdf_path).name
            item = raw_by_basename.get(basename)
            if item is None:
                filtered_manifest_documents_without_raw_pdf += 1
                continue
            key = item["Key"]
            if not isinstance(key, str):
                raise SnapshotSelectionError("validated S3 key changed type")
            if key in matched_raw_keys:
                raise SnapshotSelectionError(
                    f"one raw PDF is referenced by multiple classification documents: {key}"
                )
            matched_raw_keys.add(key)
            if document.final_label not in selected_labels:
                continue
            document_type = config.classification_label_mapping[document.final_label]
            if key in selected_raw_keys:
                raise SnapshotSelectionError(f"duplicate selected raw PDF key: {key}")
            selected_raw_keys.add(key)
            relative = key.removeprefix(config.raw_prefix)
            _validated_relative_path(relative, "source_key relative to raw_prefix")
            candidates.append(
                CandidateSelectionRecord(
                    source_bucket=config.bucket,
                    source_key=key,
                    source_size_bytes=item["Size"],
                    source_etag=item["ETag"],
                    source_last_modified=_utc_iso(
                        item["LastModified"],
                        "LastModified",
                        f"s3://{config.bucket}/{key}",
                    ),
                    document_type=document_type,
                    classification_doc_id=doc_id,
                    classification_manifest_key=manifest.config_key,
                    document_page_count=document.page_count,
                )
            )
    ordered = tuple(sorted(candidates, key=lambda item: item.source_key))
    if not ordered:
        raise SnapshotSelectionError("classifier selection contains no requested documents")
    selection_payload = _canonical_jsonl(ordered)
    digest = sha256_bytes(selection_payload)
    if digest != config.expected_selection_sha256:
        raise SnapshotSelectionError(
            f"selection SHA-256 changed ({digest} != {config.expected_selection_sha256})"
        )
    selected_types = {item.document_type for item in ordered}
    expected_types = set(config.document_types)
    if selected_types != expected_types:
        raise SnapshotSelectionError(
            f"selection does not contain every configured document type: {sorted(selected_types)}"
        )
    return _SelectionPlan(
        candidates=ordered,
        selection_payload=selection_payload,
        selection_sha256=digest,
        raw_pdf_count=raw_pdf_count,
        raw_pdf_bytes=raw_pdf_bytes,
        raw_pdfs_without_filtered_manifest_evidence=(raw_pdf_count - len(matched_raw_keys)),
        filtered_manifest_documents_without_raw_pdf=(filtered_manifest_documents_without_raw_pdf),
    )


def _local_paths(
    root: Path,
    config: S3LocalSnapshotConfig,
    candidate: CandidateSelectionRecord,
) -> tuple[str, str, Path, Path, Path]:
    if not candidate.source_key.startswith(config.raw_prefix):
        raise SnapshotSelectionError(f"selected key is outside raw_prefix: {candidate.source_key}")
    local_relative_key = candidate.source_key.removeprefix(config.raw_prefix)
    relative = _validated_relative_path(local_relative_key, "local_relative_key")
    snapshot_relative = PurePosixPath("files", candidate.document_type, *relative.parts)
    target = root / Path(*snapshot_relative.parts)
    state_name = sha256_bytes(candidate.source_key.encode("utf-8")) + ".json"
    state_path = root / "download-state" / candidate.document_type / state_name
    work_path = root / "work" / candidate.document_type / (state_name + ".partial")
    return (
        local_relative_key,
        snapshot_relative.as_posix(),
        target,
        state_path,
        work_path,
    )


def _state_payload_matches_candidate(
    state: DownloadStateRecord,
    candidate: CandidateSelectionRecord,
    *,
    classification_manifest_sha256: str,
    local_relative_key: str,
    snapshot_relative_path: str,
) -> bool:
    return (
        _candidate_from_state(state) == candidate
        and state.classification_manifest_sha256 == classification_manifest_sha256
        and state.local_relative_key == local_relative_key
        and state.snapshot_relative_path == snapshot_relative_path
    )


def _load_download_state(path: Path) -> DownloadStateRecord:
    try:
        payload = read_regular_file_bytes(path)
        return DownloadStateRecord.model_validate_json(payload, strict=True)
    except Exception as error:
        raise SnapshotIntegrityError(f"invalid download state: {path}") from error


def _download_one(
    *,
    client: Any,
    config: S3LocalSnapshotConfig,
    root: Path,
    candidate: CandidateSelectionRecord,
    manifest_hashes: Mapping[str, str],
) -> tuple[DownloadStateRecord, bool]:
    try:
        classification_manifest_sha256 = manifest_hashes[candidate.classification_manifest_key]
    except KeyError as error:
        raise SnapshotSelectionError(
            f"selected object references an unknown classification manifest: "
            f"{candidate.classification_manifest_key}"
        ) from error
    (
        local_relative_key,
        snapshot_relative_path,
        target,
        state_path,
        work_path,
    ) = _local_paths(root, config, candidate)
    if state_path.exists() or state_path.is_symlink():
        state = _load_download_state(state_path)
        if not _state_payload_matches_candidate(
            state,
            candidate,
            classification_manifest_sha256=classification_manifest_sha256,
            local_relative_key=local_relative_key,
            snapshot_relative_path=snapshot_relative_path,
        ):
            raise SnapshotIntegrityError(
                f"download state conflicts with the selected source: {state_path}"
            )
        if not target.exists() or target.is_symlink():
            raise SnapshotIntegrityError(f"download state exists without its regular PDF: {target}")
        existing_digest, size_bytes = _hash_local_pdf(
            target,
            max_pdf_bytes=config.download.max_pdf_bytes,
        )
        if existing_digest != state.source_sha256 or size_bytes != state.source_size_bytes:
            raise SnapshotIntegrityError(f"resumable local PDF changed: {target}")
        return state, True

    target_parent = _safe_snapshot_directory(target.parent, create=True)
    _safe_snapshot_directory(state_path.parent, create=True)
    work_parent = _safe_snapshot_directory(work_path.parent, create=True)
    locator = f"s3://{candidate.source_bucket}/{candidate.source_key}"
    try:
        response = client.get_object(
            Bucket=candidate.source_bucket,
            Key=candidate.source_key,
            IfMatch=candidate.source_etag,
            ChecksumMode="ENABLED",
        )
    except Exception as error:
        raise SnapshotRemoteChangedError(
            f"failed conditional GET for selected PDF: {locator}"
        ) from error
    if not isinstance(response, Mapping):
        raise SnapshotRemoteChangedError(f"S3 GET response is not a mapping: {locator}")
    content_length = _required_response_value(response, "ContentLength", int, locator)
    etag = _required_response_value(response, "ETag", str, locator)
    last_modified = _required_response_value(response, "LastModified", datetime, locator)
    if content_length != candidate.source_size_bytes:
        raise SnapshotRemoteChangedError(f"selected PDF size changed: {locator}")
    if etag != candidate.source_etag:
        raise SnapshotRemoteChangedError(f"selected PDF ETag changed: {locator}")
    if _utc_iso(last_modified, "LastModified", locator) != candidate.source_last_modified:
        raise SnapshotRemoteChangedError(f"selected PDF LastModified changed: {locator}")
    body = response.get("Body")
    if body is None or not callable(getattr(body, "read", None)):
        raise SnapshotRemoteChangedError(f"S3 GET response has no readable Body: {locator}")

    checksums: dict[str, str | None] = {}
    for model_field, response_field in _CHECKSUM_RESPONSE_FIELDS.items():
        value = response.get(response_field)
        if value is not None and not isinstance(value, str):
            raise SnapshotRemoteChangedError(
                f"S3 response field {response_field} has an invalid type: {locator}"
            )
        checksums[model_field] = value

    download_digest = hashlib.sha256()
    header = bytearray()
    size_bytes = 0
    temporary_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w+b",
            dir=work_parent,
            prefix=f".{work_path.name}.",
            suffix=".tmp",
            delete=False,
        ) as output:
            temporary_path = Path(output.name)
            while True:
                chunk = body.read(config.download.chunk_size_bytes)
                if not isinstance(chunk, bytes):
                    raise SnapshotRemoteChangedError(
                        f"selected PDF body returned non-bytes data: {locator}"
                    )
                if not chunk:
                    break
                size_bytes += len(chunk)
                if size_bytes > config.download.max_pdf_bytes:
                    raise SnapshotRemoteChangedError(
                        f"selected PDF exceeds max_pdf_bytes while downloading: {locator}"
                    )
                download_digest.update(chunk)
                if len(header) < _PDF_HEADER_SCAN_BYTES:
                    remaining = _PDF_HEADER_SCAN_BYTES - len(header)
                    header.extend(chunk[:remaining])
                written = output.write(chunk)
                if written != len(chunk):
                    raise OSError(f"short local snapshot write: {target}")
            output.flush()
            os.fsync(output.fileno())
        if size_bytes != candidate.source_size_bytes or size_bytes != content_length:
            raise SnapshotRemoteChangedError(f"selected PDF body length changed: {locator}")
        if b"%PDF-" not in header:
            raise SnapshotRemoteChangedError(
                f"selected object has no PDF header in its first "
                f"{_PDF_HEADER_SCAN_BYTES} bytes: {locator}"
            )
        source_sha256 = download_digest.hexdigest()
        try:
            os.link(temporary_path, target_parent / target.name, follow_symlinks=False)
        except FileExistsError:
            existing_digest, existing_size = _hash_local_pdf(
                target,
                max_pdf_bytes=config.download.max_pdf_bytes,
            )
            if existing_digest != source_sha256 or existing_size != size_bytes:
                raise SnapshotIntegrityError(
                    f"existing local PDF conflicts with conditional S3 download: {target}"
                ) from None
        directory_fd = os.open(target_parent, os.O_RDONLY | os.O_DIRECTORY)
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
    finally:
        close = getattr(body, "close", None)
        if callable(close):
            close()
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)

    state = DownloadStateRecord.model_validate(
        {
            **candidate.model_dump(),
            "schema_version": 1,
            "classification_manifest_sha256": classification_manifest_sha256,
            "local_relative_key": local_relative_key,
            "snapshot_relative_path": snapshot_relative_path,
            "source_sha256": source_sha256,
            **checksums,
        },
        strict=True,
    )
    try:
        atomic_publish_json(state_path, state.model_dump(mode="json"))
    except AtomicConflictError as error:
        raise SnapshotIntegrityError(
            f"download state publication conflict: {state_path}"
        ) from error
    return state, False


def _download_selection(
    *,
    config: S3LocalSnapshotConfig,
    client: Any,
    root: Path,
    plan: _SelectionPlan,
    manifests: tuple[_FetchedManifest, ...],
    progress: Callable[[int, int, int, int], None] | None,
) -> tuple[DownloadStateRecord, ...]:
    manifest_hashes = {item.config_key: item.sha256 for item in manifests}
    completed = 0
    reused = 0
    completed_bytes = 0
    results: list[DownloadStateRecord] = []
    executor = ThreadPoolExecutor(
        max_workers=config.download.workers,
        thread_name_prefix="s3-snapshot",
    )
    try:
        futures = {
            executor.submit(
                _download_one,
                client=client,
                config=config,
                root=root,
                candidate=candidate,
                manifest_hashes=manifest_hashes,
            ): candidate
            for candidate in plan.candidates
        }
        for future in as_completed(futures):
            try:
                state, was_reused = future.result()
            except BaseException:
                for pending in futures:
                    pending.cancel()
                raise
            results.append(state)
            completed += 1
            reused += int(was_reused)
            completed_bytes += state.source_size_bytes
            if progress is not None:
                progress(completed, len(plan.candidates), completed_bytes, reused)
    finally:
        executor.shutdown(wait=True, cancel_futures=True)
    return tuple(sorted(results, key=lambda item: item.source_key))


def _pdf_page_count(path: str) -> int:
    import pypdfium2

    pdf_path = Path(path)
    descriptor = _open_absolute_no_symlinks(pdf_path)
    try:
        before = os.fstat(descriptor)
        if not stat.S_ISREG(before.st_mode):
            raise ValueError(f"PDF is not a regular file: {path}")
        with os.fdopen(descriptor, "rb", closefd=False) as stream:
            document = pypdfium2.PdfDocument(stream, autoclose=False)
            try:
                page_count = len(document)
            finally:
                document.close()
        after = os.fstat(descriptor)
    finally:
        os.close(descriptor)
    before_identity = (before.st_dev, before.st_ino, before.st_size, before.st_mtime_ns)
    after_identity = (after.st_dev, after.st_ino, after.st_size, after.st_mtime_ns)
    if before_identity != after_identity:
        raise ValueError(f"PDF changed during page-count inspection: {path}")
    if page_count <= 0:
        raise ValueError(f"PDF has no pages: {path}")
    return page_count


def _inspect_page_counts(
    *,
    root: Path,
    states: tuple[DownloadStateRecord, ...],
    config: S3LocalSnapshotConfig,
) -> tuple[SnapshotFileRecord, ...]:
    context = multiprocessing.get_context("spawn")
    paths = [str(root / Path(*PurePosixPath(item.snapshot_relative_path).parts)) for item in states]
    with ProcessPoolExecutor(
        max_workers=config.download.page_inspection_processes,
        mp_context=context,
    ) as executor:
        page_counts = tuple(executor.map(_pdf_page_count, paths, chunksize=16))
    records: list[SnapshotFileRecord] = []
    quarantine_by_key = {item.source_key: item for item in config.page_count_quarantine}
    mismatches: list[str] = []
    used_quarantine_keys: set[str] = set()
    for state, page_count in zip(states, page_counts, strict=True):
        quarantine = quarantine_by_key.get(state.source_key)
        if quarantine is not None:
            used_quarantine_keys.add(state.source_key)
            if (
                state.source_sha256 != quarantine.source_sha256
                or state.document_page_count != quarantine.classified_page_count
                or page_count != quarantine.actual_page_count
                or page_count == state.document_page_count
            ):
                mismatches.append(
                    f"declared quarantine no longer matches {state.snapshot_relative_path} "
                    f"(sha={state.source_sha256}, classified={state.document_page_count}, "
                    f"actual={page_count})"
                )
            extraction_status: Literal["ready", "quarantined"] = "quarantined"
            extraction_relative_path: str | None = None
            quarantine_reason: str | None = quarantine.reason
        else:
            if page_count != state.document_page_count:
                mismatches.append(
                    f"undeclared mismatch {state.snapshot_relative_path} "
                    f"({state.document_page_count} != {page_count})"
                )
            local = PurePosixPath(state.local_relative_key)
            extraction_status = "ready"
            extraction_relative_path = PurePosixPath(
                "extraction",
                state.document_type,
                *local.parts,
            ).as_posix()
            quarantine_reason = None
        records.append(
            SnapshotFileRecord(
                **state.model_dump(),
                actual_page_count=page_count,
                extraction_status=extraction_status,
                extraction_relative_path=extraction_relative_path,
                quarantine_reason=quarantine_reason,
            )
        )
    unused_quarantine_keys = sorted(set(quarantine_by_key) - used_quarantine_keys)
    if unused_quarantine_keys:
        mismatches.append(
            "configured quarantine keys are not in the selection: "
            + ", ".join(unused_quarantine_keys[:5])
        )
    if mismatches:
        raise SnapshotIntegrityError(
            f"page-count quarantine contract failed for {len(mismatches)} entries; "
            f"examples: {'; '.join(mismatches[:5])}"
        )
    return tuple(records)


def _publish_extraction_links(
    *,
    root: Path,
    records: tuple[SnapshotFileRecord, ...],
) -> None:
    for original_parent, extraction_parent, names in _extraction_groups(root, records):
        source = _safe_snapshot_directory(original_parent, create=False)
        destination = _safe_snapshot_directory(extraction_parent, create=True)
        source_fd = os.open(
            source,
            os.O_RDONLY | os.O_CLOEXEC | os.O_NOFOLLOW | os.O_DIRECTORY,
        )
        destination_fd = os.open(
            destination,
            os.O_RDONLY | os.O_CLOEXEC | os.O_NOFOLLOW | os.O_DIRECTORY,
        )
        try:
            for name in names:
                try:
                    os.link(
                        name,
                        name,
                        src_dir_fd=source_fd,
                        dst_dir_fd=destination_fd,
                        follow_symlinks=False,
                    )
                except FileExistsError:
                    _validate_hard_link(
                        name=name,
                        source_fd=source_fd,
                        destination_fd=destination_fd,
                        destination_parent=destination,
                    )
            os.fsync(destination_fd)
        finally:
            os.close(destination_fd)
            os.close(source_fd)

    _validate_extraction_links(root=root, records=records)


def _extraction_groups(
    root: Path,
    records: tuple[SnapshotFileRecord, ...],
) -> tuple[tuple[Path, Path, tuple[str, ...]], ...]:
    grouped: dict[tuple[Path, Path], list[str]] = defaultdict(list)
    for record in records:
        if record.extraction_status != "ready":
            continue
        if record.extraction_relative_path is None:
            raise SnapshotIntegrityError("ready record has no extraction path")
        original = root / Path(*PurePosixPath(record.snapshot_relative_path).parts)
        extraction = root / Path(*PurePosixPath(record.extraction_relative_path).parts)
        if original.name != extraction.name:
            raise SnapshotIntegrityError("extraction filename differs from its original")
        grouped[(original.parent, extraction.parent)].append(original.name)
    return tuple(
        (source, destination, tuple(sorted(names)))
        for (source, destination), names in sorted(
            grouped.items(),
            key=lambda item: (str(item[0][0]), str(item[0][1])),
        )
    )


def _validate_hard_link(
    *,
    name: str,
    source_fd: int,
    destination_fd: int,
    destination_parent: Path,
) -> None:
    try:
        original_stat = os.stat(name, dir_fd=source_fd, follow_symlinks=False)
        extraction_stat = os.stat(name, dir_fd=destination_fd, follow_symlinks=False)
    except OSError as error:
        raise SnapshotIntegrityError(
            f"cannot validate extraction hard link: {destination_parent / name}"
        ) from error
    original_identity = (original_stat.st_dev, original_stat.st_ino)
    extraction_identity = (extraction_stat.st_dev, extraction_stat.st_ino)
    if (
        not stat.S_ISREG(original_stat.st_mode)
        or not stat.S_ISREG(extraction_stat.st_mode)
        or original_identity != extraction_identity
    ):
        raise SnapshotIntegrityError(
            f"extraction file is not the immutable original hard link: {destination_parent / name}"
        )


def _validate_extraction_links(
    *,
    root: Path,
    records: tuple[SnapshotFileRecord, ...],
) -> None:

    expected = {
        root / Path(*PurePosixPath(record.extraction_relative_path).parts)
        for record in records
        if record.extraction_relative_path is not None
    }
    extraction_root = _safe_snapshot_directory(root / "extraction", create=False)
    actual = set(_safe_walk_files(extraction_root))
    if actual != expected:
        raise SnapshotIntegrityError(
            "extraction file tree does not exactly match the ready snapshot records"
        )
    for original_parent, extraction_parent, names in _extraction_groups(root, records):
        source = _safe_snapshot_directory(original_parent, create=False)
        destination = _safe_snapshot_directory(extraction_parent, create=False)
        source_fd = os.open(
            source,
            os.O_RDONLY | os.O_CLOEXEC | os.O_NOFOLLOW | os.O_DIRECTORY,
        )
        destination_fd = os.open(
            destination,
            os.O_RDONLY | os.O_CLOEXEC | os.O_NOFOLLOW | os.O_DIRECTORY,
        )
        try:
            for name in names:
                _validate_hard_link(
                    name=name,
                    source_fd=source_fd,
                    destination_fd=destination_fd,
                    destination_parent=destination,
                )
        finally:
            os.close(destination_fd)
            os.close(source_fd)


def _publish_digest_bound_artifact(path: Path, payload: bytes) -> str:
    digest = sha256_bytes(payload)
    try:
        atomic_publish_bytes(path, payload)
        atomic_publish_bytes(
            path.with_name(path.name + ".sha256"),
            _digest_sidecar(digest, path.name),
        )
    except AtomicConflictError as error:
        raise SnapshotIntegrityError(f"immutable snapshot artifact conflicts: {path}") from error
    return digest


def _stored_manifest_record(item: _FetchedManifest) -> StoredClassificationManifest:
    relative = PurePosixPath("classification", item.local_name)
    return StoredClassificationManifest(
        key=item.config_key,
        sha256=item.sha256,
        snapshot_relative_path=relative.as_posix(),
        source_etag=item.source_etag,
        source_last_modified=item.source_last_modified,
        source_size_bytes=item.source_size_bytes,
    )


def _build_commit(
    *,
    config: S3LocalSnapshotConfig,
    selection_sha256: str,
    manifest_sha256: str,
    manifests: tuple[_FetchedManifest, ...],
    records: tuple[SnapshotFileRecord, ...],
    selection_audit: SnapshotSelectionAudit,
) -> SnapshotCommit:
    by_type: dict[str, SnapshotTypeSummary] = {}
    for document_type in sorted(config.document_types):
        selected = [item for item in records if item.document_type == document_type]
        if not selected:
            raise SnapshotIntegrityError(
                f"completed snapshot has no records for document type: {document_type}"
            )
        by_type[document_type] = SnapshotTypeSummary(
            documents=len(selected),
            source_bytes=sum(item.source_size_bytes for item in selected),
            classified_pages=sum(item.document_page_count for item in selected),
            actual_pages=sum(item.actual_page_count for item in selected),
            extraction_documents=sum(item.extraction_status == "ready" for item in selected),
            extraction_pages=sum(
                item.actual_page_count for item in selected if item.extraction_status == "ready"
            ),
            quarantined_documents=sum(item.extraction_status == "quarantined" for item in selected),
        )
    return SnapshotCommit(
        schema_version=1,
        bucket=config.bucket,
        raw_prefix=config.raw_prefix,
        region=config.region,
        document_types=sorted(config.document_types),
        selection_path="selection.jsonl",
        selection_sha256=selection_sha256,
        manifest_path="manifest.jsonl",
        manifest_sha256=manifest_sha256,
        classification_manifests=[
            _stored_manifest_record(item)
            for item in sorted(manifests, key=lambda value: value.config_key)
        ],
        selection_audit=selection_audit,
        by_document_type=by_type,
        document_count=len(records),
        source_bytes=sum(item.source_size_bytes for item in records),
        classified_pages=sum(item.document_page_count for item in records),
        actual_pages=sum(item.actual_page_count for item in records),
        extraction_document_count=sum(item.extraction_status == "ready" for item in records),
        extraction_pages=sum(
            item.actual_page_count for item in records if item.extraction_status == "ready"
        ),
        quarantined_document_count=sum(item.extraction_status == "quarantined" for item in records),
    )


def _new_s3_client(config: S3LocalSnapshotConfig) -> Any:
    client_config = BotoClientConfig(
        max_pool_connections=config.download.workers,
        retries={
            "mode": "standard",
            "total_max_attempts": config.download.max_attempts,
        },
    )
    return boto3.client("s3", region_name=config.region, config=client_config)


def materialize_snapshot(
    config: S3LocalSnapshotConfig,
    *,
    s3_client: Any | None = None,
    progress: Callable[[int, int, int, int], None] | None = None,
) -> SnapshotResult:
    """Create or verify a complete classifier-selected local source snapshot.

    The completed ``snapshot.json`` marker is published only after every PDF,
    classifier manifest, digest sidecar, page-count check, and canonical JSONL
    manifest is durable.  If that marker already exists, the method performs a
    complete local verification and does not contact S3.
    """

    root = _safe_snapshot_directory(Path(config.destination_root), create=True)
    commit_path = root / "snapshot.json"
    if commit_path.exists() or commit_path.is_symlink():
        return verify_snapshot(config, progress=progress)

    client = _new_s3_client(config) if s3_client is None else s3_client
    manifests = tuple(
        _fetch_classification_manifest(
            client=client,
            config=config,
            manifest_key=item.key,
            expected_sha256=item.expected_sha256,
            local_name=item.local_name,
        )
        for item in config.classification_manifests
    )
    plan = _build_selection_plan(config=config, client=client, manifests=manifests)

    classification_directory = _safe_snapshot_directory(root / "classification", create=True)
    for item in manifests:
        _publish_digest_bound_artifact(classification_directory / item.local_name, item.payload)
    selection_digest = _publish_digest_bound_artifact(
        root / "selection.jsonl",
        plan.selection_payload,
    )
    if selection_digest != config.expected_selection_sha256:
        raise SnapshotIntegrityError("published selection digest differs from configuration")

    states = _download_selection(
        config=config,
        client=client,
        root=root,
        plan=plan,
        manifests=manifests,
        progress=progress,
    )
    records = _inspect_page_counts(
        root=root,
        states=states,
        config=config,
    )
    _validate_original_file_tree(root=root, records=records)
    _publish_extraction_links(root=root, records=records)
    manifest_payload = _canonical_jsonl(records)
    manifest_sha256 = _publish_digest_bound_artifact(root / "manifest.jsonl", manifest_payload)
    commit = _build_commit(
        config=config,
        selection_sha256=selection_digest,
        manifest_sha256=manifest_sha256,
        manifests=manifests,
        records=records,
        selection_audit=SnapshotSelectionAudit(
            raw_pdf_count=plan.raw_pdf_count,
            raw_pdf_bytes=plan.raw_pdf_bytes,
            raw_pdfs_without_filtered_manifest_evidence=(
                plan.raw_pdfs_without_filtered_manifest_evidence
            ),
            filtered_manifest_documents_without_raw_pdf=(
                plan.filtered_manifest_documents_without_raw_pdf
            ),
        ),
    )
    try:
        atomic_publish_json(commit_path, commit.model_dump(mode="json"))
    except AtomicConflictError as error:
        raise SnapshotIntegrityError(
            f"snapshot commit publication conflict: {commit_path}"
        ) from error
    return SnapshotResult(root=root, commit=commit, created=True)


def _read_digest_bound_artifact(path: Path, expected_digest: str) -> bytes:
    try:
        payload = read_regular_file_bytes(path)
        sidecar = read_regular_file_bytes(path.with_name(path.name + ".sha256"))
    except Exception as error:
        raise SnapshotIntegrityError(f"cannot read snapshot artifact and digest: {path}") from error
    digest = sha256_bytes(payload)
    if digest != expected_digest:
        raise SnapshotIntegrityError(
            f"snapshot artifact SHA-256 changed ({digest} != {expected_digest}): {path}"
        )
    if sidecar != _digest_sidecar(digest, path.name):
        raise SnapshotIntegrityError(f"snapshot digest sidecar changed: {path}.sha256")
    return payload


def _load_snapshot_records(payload: bytes, path: Path) -> tuple[SnapshotFileRecord, ...]:
    if not payload or not payload.endswith(b"\n"):
        raise SnapshotIntegrityError(f"snapshot manifest must be non-empty JSONL: {path}")
    records: list[SnapshotFileRecord] = []
    for line_number, line in enumerate(payload.splitlines(), start=1):
        try:
            records.append(SnapshotFileRecord.model_validate_json(line, strict=True))
        except Exception as error:
            raise SnapshotIntegrityError(
                f"invalid snapshot manifest row {line_number}: {path}"
            ) from error
    ordered = tuple(sorted(records, key=lambda item: item.source_key))
    if len({item.source_key for item in ordered}) != len(ordered):
        raise SnapshotIntegrityError("snapshot manifest contains duplicate source_key values")
    if len({item.snapshot_relative_path for item in ordered}) != len(ordered):
        raise SnapshotIntegrityError(
            "snapshot manifest contains duplicate snapshot_relative_path values"
        )
    if payload != _canonical_jsonl(ordered):
        raise SnapshotIntegrityError(
            "snapshot manifest is not canonical and deterministically sorted"
        )
    return ordered


def _safe_walk_files(directory: Path) -> tuple[Path, ...]:
    root = _safe_snapshot_directory(directory, create=False)
    files: list[Path] = []

    def walk(path: Path) -> None:
        try:
            with os.scandir(path) as iterator:
                entries = sorted(iterator, key=lambda item: item.name)
        except OSError as error:
            raise SnapshotIntegrityError(f"cannot enumerate snapshot directory: {path}") from error
        for entry in entries:
            entry_path = path / entry.name
            try:
                entry_stat = entry.stat(follow_symlinks=False)
            except OSError as error:
                raise SnapshotIntegrityError(f"cannot stat snapshot entry: {entry_path}") from error
            if stat.S_ISLNK(entry_stat.st_mode):
                raise SnapshotIntegrityError(
                    f"symbolic links are forbidden in snapshot files: {entry_path}"
                )
            if stat.S_ISDIR(entry_stat.st_mode):
                walk(entry_path)
            elif stat.S_ISREG(entry_stat.st_mode):
                files.append(entry_path)
            else:
                raise SnapshotIntegrityError(
                    f"non-regular entries are forbidden in snapshot files: {entry_path}"
                )

    walk(root)
    return tuple(files)


def _verify_local_records(
    *,
    config: S3LocalSnapshotConfig,
    root: Path,
    records: tuple[SnapshotFileRecord, ...],
    progress: Callable[[int, int, int, int], None] | None,
) -> None:
    _validate_original_file_tree(root=root, records=records)

    completed = 0
    completed_bytes = 0
    with ThreadPoolExecutor(
        max_workers=config.download.workers,
        thread_name_prefix="snapshot-verify",
    ) as executor:
        futures = {
            executor.submit(
                _hash_local_pdf,
                root / Path(*PurePosixPath(item.snapshot_relative_path).parts),
                max_pdf_bytes=config.download.max_pdf_bytes,
            ): item
            for item in records
        }
        for future in as_completed(futures):
            item = futures[future]
            digest, size_bytes = future.result()
            if digest != item.source_sha256 or size_bytes != item.source_size_bytes:
                raise SnapshotIntegrityError(
                    f"local PDF differs from snapshot manifest: {item.snapshot_relative_path}"
                )
            completed += 1
            completed_bytes += size_bytes
            if progress is not None:
                progress(completed, len(records), completed_bytes, 0)

    states = tuple(
        DownloadStateRecord.model_validate(
            {name: getattr(record, name) for name in DownloadStateRecord.model_fields},
            strict=True,
        )
        for record in records
    )
    inspected = _inspect_page_counts(
        root=root,
        states=states,
        config=config,
    )
    if inspected != records:
        raise SnapshotIntegrityError(
            "local PDF page counts or quarantine status differ from snapshot manifest"
        )
    _validate_extraction_links(root=root, records=records)


def _validate_original_file_tree(
    *,
    root: Path,
    records: tuple[SnapshotFileRecord, ...],
) -> None:
    expected_paths = {
        root / Path(*PurePosixPath(item.snapshot_relative_path).parts) for item in records
    }
    files_root = root / "files"
    actual_paths = set(_safe_walk_files(files_root))
    if actual_paths != expected_paths:
        missing = sorted(str(path) for path in expected_paths - actual_paths)
        unexpected = sorted(str(path) for path in actual_paths - expected_paths)
        raise SnapshotIntegrityError(
            f"snapshot file tree differs from manifest; missing={missing[:3]!r}, "
            f"unexpected={unexpected[:3]!r}"
        )


def _validate_commit_against_config(
    commit: SnapshotCommit,
    config: S3LocalSnapshotConfig,
) -> None:
    if commit.bucket != config.bucket:
        raise SnapshotIntegrityError("snapshot bucket differs from configuration")
    if commit.raw_prefix != config.raw_prefix:
        raise SnapshotIntegrityError("snapshot raw_prefix differs from configuration")
    if commit.region != config.region:
        raise SnapshotIntegrityError("snapshot region differs from configuration")
    if commit.document_types != sorted(config.document_types):
        raise SnapshotIntegrityError("snapshot document_types differ from configuration")
    if commit.selection_sha256 != config.expected_selection_sha256:
        raise SnapshotIntegrityError("snapshot selection SHA-256 differs from configuration")
    expected_manifests = {
        item.key: item.expected_sha256 for item in config.classification_manifests
    }
    committed_manifests = {item.key: item.sha256 for item in commit.classification_manifests}
    if committed_manifests != expected_manifests:
        raise SnapshotIntegrityError("snapshot classification manifests differ from configuration")


def verify_snapshot(
    config: S3LocalSnapshotConfig,
    *,
    progress: Callable[[int, int, int, int], None] | None = None,
) -> SnapshotResult:
    """Hash and inspect every local artifact without contacting S3 or a model."""

    root = _safe_snapshot_directory(Path(config.destination_root), create=False)
    commit_path = root / "snapshot.json"
    try:
        commit_payload = read_regular_file_bytes(commit_path)
        commit = SnapshotCommit.model_validate_json(commit_payload, strict=True)
    except Exception as error:
        raise SnapshotIntegrityError(
            f"invalid or missing snapshot commit: {commit_path}"
        ) from error
    expected_commit_payload = (
        json.dumps(
            commit.model_dump(mode="json"),
            allow_nan=False,
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        ).encode("utf-8")
        + b"\n"
    )
    if commit_payload != expected_commit_payload:
        raise SnapshotIntegrityError(f"snapshot commit is not canonical: {commit_path}")
    _validate_commit_against_config(commit, config)

    selection_payload = _read_digest_bound_artifact(
        root / commit.selection_path,
        commit.selection_sha256,
    )
    manifest_payload = _read_digest_bound_artifact(
        root / commit.manifest_path,
        commit.manifest_sha256,
    )
    records = _load_snapshot_records(manifest_payload, root / commit.manifest_path)
    candidate_payload = _canonical_jsonl(_candidate_from_file(item) for item in records)
    if candidate_payload != selection_payload:
        raise SnapshotIntegrityError(
            "snapshot selection rows do not equal manifest provenance rows"
        )
    if len(records) != commit.document_count:
        raise SnapshotIntegrityError("snapshot document count differs from commit")

    for manifest in commit.classification_manifests:
        relative = _validated_relative_path(
            manifest.snapshot_relative_path,
            "classification snapshot_relative_path",
        )
        _read_digest_bound_artifact(root / Path(*relative.parts), manifest.sha256)

    manifest_hashes = {item.key: item.sha256 for item in commit.classification_manifests}
    for record in records:
        if manifest_hashes.get(record.classification_manifest_key) != (
            record.classification_manifest_sha256
        ):
            raise SnapshotIntegrityError(
                f"PDF classification hash differs from stored manifest: {record.source_key}"
            )
    recomputed_commit = _build_commit(
        config=config,
        selection_sha256=commit.selection_sha256,
        manifest_sha256=commit.manifest_sha256,
        manifests=tuple(
            _FetchedManifest(
                config_key=item.key,
                local_name=PurePosixPath(item.snapshot_relative_path).name,
                sha256=item.sha256,
                payload=b"unused-after-hash-validation",
                source_etag=item.source_etag,
                source_last_modified=item.source_last_modified,
                source_size_bytes=item.source_size_bytes,
            )
            for item in commit.classification_manifests
        ),
        records=records,
        selection_audit=commit.selection_audit,
    )
    if recomputed_commit != commit:
        raise SnapshotIntegrityError("snapshot totals differ from its manifest rows")

    _verify_local_records(
        config=config,
        root=root,
        records=records,
        progress=progress,
    )
    return SnapshotResult(root=root, commit=commit, created=False)
