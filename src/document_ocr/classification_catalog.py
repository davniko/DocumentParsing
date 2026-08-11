"""Deterministic document-level catalog for all classifier and raw-source evidence.

The source bucket is unversioned, so every small classifier artifact is copied
and content-pinned while the raw PDF population is frozen as an exact S3
metadata inventory.  The catalog joins the initial and filtered manifests at
document granularity, retains every explicit filter report, represents raw-only
PDFs without inventing labels, and binds any already-materialized local
snapshot aliases to their content SHA-256 values.
"""

from __future__ import annotations

import csv
import io
import json
import os
import re
from collections import Counter, defaultdict
from collections.abc import Callable, Iterable, Mapping
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path, PurePosixPath
from typing import Any, Literal, cast

import boto3
from botocore.config import Config as BotoClientConfig
from pydantic import BaseModel, ConfigDict, Field, JsonValue, field_validator, model_validator

from document_ocr.atomic import AtomicConflictError, atomic_publish_json, read_regular_file_bytes
from document_ocr.config import (
    CatalogArtifactConfig,
    CatalogLineageConfig,
    ClassificationCatalogConfig,
    load_snapshot_config,
)
from document_ocr.hashing import canonical_json_bytes, sha256_bytes
from document_ocr.snapshot import (
    SnapshotFileRecord,
    _load_snapshot_records,
    _publish_digest_bound_artifact,
    _read_digest_bound_artifact,
    _safe_snapshot_directory,
    _safe_walk_files,
    verify_snapshot,
)

_DOCUMENT_FILENAME = re.compile(
    r"^(?P<document_date>\d{4}-\d{2}-\d{2})_"
    r"(?P<document_uuid>[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[1-5][0-9a-fA-F]{3}-"
    r"[89aAbB][0-9a-fA-F]{3}-[0-9a-fA-F]{12})\.pdf$"
)
_ARTIFACT_KINDS = (
    "initial_manifest",
    "final_manifest",
    "dropped_report",
    "relabeled_report",
    "review_report",
)

CatalogStatus = Literal[
    "retained_final",
    "excluded_reported",
    "missing_final_without_drop_report",
    "never_classified",
]
CatalogArtifactKind = Literal[
    "initial_manifest",
    "final_manifest",
    "dropped_report",
    "relabeled_report",
    "review_report",
]
CatalogProgress = Callable[[str, int, int], None]


def _is_sha256(value: str) -> bool:
    return len(value) == 64 and all(character in "0123456789abcdef" for character in value)


def _safe_relative_path(value: str, *, description: str) -> PurePosixPath:
    path = PurePosixPath(value)
    if path.is_absolute() or any(part in {"", ".", ".."} for part in path.parts):
        raise ValueError(f"{description} must be a safe relative path")
    return path


class _FrozenRecord(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, strict=True, allow_inf_nan=False)


class ManifestDocumentEvidence(_FrozenRecord):
    """One document's stable evidence and exact row locations in an initial manifest."""

    manifest_key: str = Field(min_length=1)
    manifest_sha256: str
    doc_id: str = Field(min_length=1)
    original_pdf_path: str = Field(min_length=1)
    declared_source_key: str = Field(min_length=1)
    classification_label: str = Field(min_length=1)
    document_page_count: int = Field(gt=0)
    source_lines_by_page: list[int] = Field(min_length=1)
    classification_llm: dict[str, JsonValue]
    dummy: dict[str, JsonValue]
    triage: dict[str, JsonValue]

    @field_validator("manifest_sha256")
    @classmethod
    def hash_is_sha256(cls, value: str) -> str:
        if not _is_sha256(value):
            raise ValueError("manifest_sha256 must be a lowercase SHA-256")
        return value

    @model_validator(mode="after")
    def page_locations_and_path_are_consistent(self) -> ManifestDocumentEvidence:
        if len(self.source_lines_by_page) != self.document_page_count:
            raise ValueError("initial source line count must equal document_page_count")
        if len(set(self.source_lines_by_page)) != len(self.source_lines_by_page):
            raise ValueError("initial source lines must be unique")
        _safe_relative_path(self.declared_source_key, description="declared_source_key")
        return self


class FinalDocumentEvidence(_FrozenRecord):
    """One retained document's resolved label and exact final-manifest rows."""

    manifest_key: str = Field(min_length=1)
    manifest_sha256: str
    final_classification_label: str = Field(min_length=1)
    document_page_count: int = Field(gt=0)
    source_lines_by_page: list[int] = Field(min_length=1)

    @field_validator("manifest_sha256")
    @classmethod
    def hash_is_sha256(cls, value: str) -> str:
        if not _is_sha256(value):
            raise ValueError("manifest_sha256 must be a lowercase SHA-256")
        return value

    @model_validator(mode="after")
    def page_locations_are_complete(self) -> FinalDocumentEvidence:
        if len(self.source_lines_by_page) != self.document_page_count:
            raise ValueError("final source line count must equal document_page_count")
        if len(set(self.source_lines_by_page)) != len(self.source_lines_by_page):
            raise ValueError("final source lines must be unique")
        return self


class ReportEvidence(_FrozenRecord):
    """One exact CSV filter-report row and its immutable source locator."""

    report_key: str = Field(min_length=1)
    report_sha256: str
    source_line: int = Field(ge=2)
    values: dict[str, str]

    @field_validator("report_sha256")
    @classmethod
    def hash_is_sha256(cls, value: str) -> str:
        if not _is_sha256(value):
            raise ValueError("report_sha256 must be a lowercase SHA-256")
        return value


class CatalogS3Alias(_FrozenRecord):
    """One current unversioned S3 object spelling represented by a document row."""

    source_bucket: str = Field(min_length=1)
    lineage: str = Field(min_length=1)
    role: Literal["primary", "mirror"]
    source_key: str = Field(min_length=1)
    source_size_bytes: int = Field(gt=0)
    source_etag: str = Field(min_length=3)
    source_last_modified: str = Field(min_length=1)
    source_storage_class: str = Field(min_length=1)
    source_version_id: None = None
    checksum_algorithms: list[str]
    checksum_type: str | None

    @field_validator("source_etag")
    @classmethod
    def etag_is_quoted(cls, value: str) -> str:
        if not value.startswith('"') or not value.endswith('"'):
            raise ValueError("source_etag must retain S3's quoted form")
        return value

    @field_validator("source_last_modified")
    @classmethod
    def last_modified_is_canonical_utc(cls, value: str) -> str:
        try:
            parsed = datetime.fromisoformat(value)
        except ValueError as error:
            raise ValueError("source_last_modified must be ISO-8601") from error
        if parsed.tzinfo is None or parsed.utcoffset() is None:
            raise ValueError("source_last_modified must be timezone-aware")
        if value != parsed.astimezone(UTC).isoformat():
            raise ValueError("source_last_modified must be canonical UTC ISO-8601")
        return value

    @model_validator(mode="after")
    def key_and_checksums_are_canonical(self) -> CatalogS3Alias:
        _safe_relative_path(self.source_key, description="source_key")
        if self.checksum_algorithms != sorted(set(self.checksum_algorithms)):
            raise ValueError("checksum_algorithms must be unique and sorted")
        return self


class CatalogLocalAlias(_FrozenRecord):
    """One content-addressed local snapshot original represented by the row."""

    snapshot_config_path: str = Field(min_length=1)
    snapshot_root: str = Field(min_length=1)
    snapshot_config_sha256: str
    snapshot_commit_sha256: str
    snapshot_manifest_sha256: str
    snapshot_selection_sha256: str
    snapshot_record: SnapshotFileRecord

    @field_validator(
        "snapshot_config_sha256",
        "snapshot_commit_sha256",
        "snapshot_manifest_sha256",
        "snapshot_selection_sha256",
    )
    @classmethod
    def hashes_are_sha256(cls, value: str) -> str:
        if not _is_sha256(value):
            raise ValueError("snapshot hash must be a lowercase SHA-256")
        return value

    @model_validator(mode="after")
    def paths_are_absolute(self) -> CatalogLocalAlias:
        if not Path(self.snapshot_config_path).is_absolute() or not Path(
            self.snapshot_root
        ).is_absolute():
            raise ValueError("local snapshot paths must be absolute")
        return self


class CatalogDocumentRecord(_FrozenRecord):
    """One logical document and every known classification/source provenance alias."""

    schema_version: Literal[1]
    catalog_document_id: str
    lineage: str = Field(min_length=1)
    status: CatalogStatus
    document_filename: str = Field(min_length=1)
    document_uuid: str | None
    initial: ManifestDocumentEvidence | None
    final: FinalDocumentEvidence | None
    dropped_report: ReportEvidence | None
    relabeled_report: ReportEvidence | None
    review_report: ReportEvidence | None
    declared_source_present: bool
    s3_source_aliases: list[CatalogS3Alias]
    local_snapshot_aliases: list[CatalogLocalAlias]

    @field_validator("catalog_document_id")
    @classmethod
    def id_is_sha256(cls, value: str) -> str:
        if not _is_sha256(value):
            raise ValueError("catalog_document_id must be a lowercase SHA-256")
        return value

    @model_validator(mode="after")
    def evidence_and_aliases_are_consistent(self) -> CatalogDocumentRecord:
        if PurePosixPath(self.document_filename).name != self.document_filename:
            raise ValueError("document_filename must be a basename")
        expected_uuid = _filename_uuid(self.document_filename)
        if self.document_uuid != expected_uuid:
            raise ValueError("document_uuid must equal the UUID encoded by document_filename")
        if self.initial is None:
            if self.status != "never_classified":
                raise ValueError("only never_classified records may omit initial evidence")
            if any(
                item is not None
                for item in (
                    self.final,
                    self.dropped_report,
                    self.relabeled_report,
                    self.review_report,
                )
            ):
                raise ValueError("never-classified records cannot contain classifier evidence")
            if not any(alias.role == "primary" for alias in self.s3_source_aliases):
                raise ValueError("never-classified records require a primary S3 source")
        else:
            if PurePosixPath(self.initial.declared_source_key).name != self.document_filename:
                raise ValueError("initial source basename differs from document_filename")
            if self.status == "retained_final":
                if self.final is None or self.dropped_report is not None:
                    raise ValueError("retained_final requires final evidence and no drop report")
            elif self.status == "excluded_reported":
                if self.final is not None or self.dropped_report is None:
                    raise ValueError("excluded_reported requires a drop report and no final row")
            elif self.status == "missing_final_without_drop_report":
                if self.final is not None or self.dropped_report is not None:
                    raise ValueError("missing-final status cannot have final/drop evidence")
            else:
                raise ValueError("classified records cannot have never_classified status")
            if self.final is not None and (
                self.final.document_page_count != self.initial.document_page_count
            ):
                raise ValueError("initial and final page counts differ")
        ordered_s3 = sorted(
            self.s3_source_aliases,
            key=lambda item: (item.role, item.source_key),
        )
        if self.s3_source_aliases != ordered_s3:
            raise ValueError("s3_source_aliases must be canonically sorted")
        s3_keys = [(item.role, item.source_key) for item in self.s3_source_aliases]
        if len(s3_keys) != len(set(s3_keys)):
            raise ValueError("s3_source_aliases must be unique")
        if any(item.lineage != self.lineage for item in self.s3_source_aliases):
            raise ValueError("S3 alias lineage differs from document lineage")
        declared_present = self.initial is not None and any(
            item.role == "primary" and item.source_key == self.initial.declared_source_key
            for item in self.s3_source_aliases
        )
        if self.declared_source_present != declared_present:
            raise ValueError("declared_source_present differs from exact primary S3 evidence")
        ordered_local = sorted(
            self.local_snapshot_aliases,
            key=lambda item: (
                item.snapshot_record.source_key,
                item.snapshot_manifest_sha256,
                item.snapshot_config_sha256,
            ),
        )
        if self.local_snapshot_aliases != ordered_local:
            raise ValueError("local_snapshot_aliases must be canonically sorted")
        local_keys = [
            (
                item.snapshot_config_sha256,
                item.snapshot_manifest_sha256,
                item.snapshot_record.source_key,
            )
            for item in self.local_snapshot_aliases
        ]
        if len(local_keys) != len(set(local_keys)):
            raise ValueError("local_snapshot_aliases must be unique")
        for alias in self.local_snapshot_aliases:
            if PurePosixPath(alias.snapshot_record.source_key).name != self.document_filename:
                raise ValueError("local snapshot alias filename differs from document_filename")
        return self


class CatalogSourceArtifact(_FrozenRecord):
    lineage: str = Field(min_length=1)
    kind: CatalogArtifactKind
    source_bucket: str = Field(min_length=1)
    source_key: str = Field(min_length=1)
    source_size_bytes: int = Field(gt=0)
    source_etag: str = Field(min_length=3)
    source_last_modified: str = Field(min_length=1)
    source_version_id: str | None
    sha256: str
    catalog_relative_path: str = Field(min_length=1)

    @field_validator("sha256")
    @classmethod
    def hash_is_sha256(cls, value: str) -> str:
        if not _is_sha256(value):
            raise ValueError("artifact sha256 must be a lowercase SHA-256")
        return value

    @model_validator(mode="after")
    def paths_and_metadata_are_valid(self) -> CatalogSourceArtifact:
        _safe_relative_path(self.source_key, description="artifact source_key")
        _safe_relative_path(
            self.catalog_relative_path,
            description="artifact catalog_relative_path",
        )
        if not self.source_etag.startswith('"') or not self.source_etag.endswith('"'):
            raise ValueError("artifact ETag must retain S3's quoted form")
        _canonical_utc(self.source_last_modified)
        return self


class CatalogSnapshotSummary(_FrozenRecord):
    snapshot_config_path: str = Field(min_length=1)
    snapshot_root: str = Field(min_length=1)
    snapshot_config_sha256: str
    snapshot_commit_sha256: str
    snapshot_manifest_sha256: str
    snapshot_selection_sha256: str
    document_count: int = Field(gt=0)
    source_bytes: int = Field(gt=0)

    @field_validator(
        "snapshot_config_sha256",
        "snapshot_commit_sha256",
        "snapshot_manifest_sha256",
        "snapshot_selection_sha256",
    )
    @classmethod
    def hashes_are_sha256(cls, value: str) -> str:
        if not _is_sha256(value):
            raise ValueError("snapshot summary hash must be a lowercase SHA-256")
        return value


class LineageStatistics(_FrozenRecord):
    catalog_documents: int = Field(ge=0)
    initial_documents: int = Field(ge=0)
    initial_page_rows: int = Field(ge=0)
    final_documents: int = Field(ge=0)
    final_page_rows: int = Field(ge=0)
    dummy_documents: int = Field(ge=0)
    reported_drop_documents: int = Field(ge=0)
    reported_relabel_documents: int = Field(ge=0)
    review_documents: int = Field(ge=0)
    declared_source_present_documents: int = Field(ge=0)
    declared_source_missing_documents: int = Field(ge=0)
    documents_with_any_s3_source: int = Field(ge=0)
    documents_with_local_snapshot: int = Field(ge=0)
    s3_source_aliases: int = Field(ge=0)
    local_snapshot_aliases: int = Field(ge=0)
    status_counts: dict[str, int]
    initial_label_counts: dict[str, int]
    final_label_counts: dict[str, int]


class CatalogStatistics(_FrozenRecord):
    schema_version: Literal[1]
    catalog_documents: int = Field(gt=0)
    initial_documents: int = Field(ge=0)
    initial_page_rows: int = Field(ge=0)
    final_documents: int = Field(ge=0)
    final_page_rows: int = Field(ge=0)
    dummy_documents: int = Field(ge=0)
    reported_drop_documents: int = Field(ge=0)
    reported_relabel_documents: int = Field(ge=0)
    review_documents: int = Field(ge=0)
    declared_source_present_documents: int = Field(ge=0)
    declared_source_missing_documents: int = Field(ge=0)
    documents_with_any_s3_source: int = Field(ge=0)
    documents_without_any_s3_source: int = Field(ge=0)
    raw_only_documents: int = Field(ge=0)
    s3_source_aliases: int = Field(ge=0)
    s3_primary_objects: int = Field(ge=0)
    s3_primary_bytes: int = Field(ge=0)
    s3_mirror_objects: int = Field(ge=0)
    s3_mirror_bytes: int = Field(ge=0)
    local_snapshot_aliases: int = Field(ge=0)
    documents_with_local_snapshot: int = Field(ge=0)
    local_unique_content_sha256: int = Field(ge=0)
    status_counts: dict[str, int]
    initial_label_counts: dict[str, int]
    initial_llm_prediction_counts: dict[str, int]
    final_label_counts: dict[str, int]
    drop_reason_counts: dict[str, int]
    review_reason_counts: dict[str, int]
    by_lineage: dict[str, LineageStatistics]


class ClassificationCatalogCommit(_FrozenRecord):
    """Manifest-last proof that every catalog artifact has been published."""

    schema_version: Literal[1]
    source_bucket: str = Field(min_length=1)
    source_region: str = Field(min_length=1)
    source_bucket_versioning_status: Literal["Enabled", "Suspended"] | None
    configuration_sha256: str
    catalog_path: Literal["catalog.jsonl"]
    catalog_sha256: str
    raw_inventory_path: Literal["raw-inventory.jsonl"]
    raw_inventory_sha256: str
    statistics_path: Literal["statistics.json"]
    statistics_sha256: str
    source_artifacts: list[CatalogSourceArtifact] = Field(min_length=1)
    local_snapshots: list[CatalogSnapshotSummary] = Field(min_length=1)
    document_count: int = Field(gt=0)
    s3_source_alias_count: int = Field(ge=0)
    local_snapshot_alias_count: int = Field(ge=0)

    @field_validator(
        "configuration_sha256",
        "catalog_sha256",
        "raw_inventory_sha256",
        "statistics_sha256",
    )
    @classmethod
    def hashes_are_sha256(cls, value: str) -> str:
        if not _is_sha256(value):
            raise ValueError("catalog commit hash must be a lowercase SHA-256")
        return value


class CatalogPlanSummary(_FrozenRecord):
    catalog_sha256: str
    raw_inventory_sha256: str
    statistics_sha256: str
    statistics: CatalogStatistics

    @field_validator("catalog_sha256", "raw_inventory_sha256", "statistics_sha256")
    @classmethod
    def hashes_are_sha256(cls, value: str) -> str:
        if not _is_sha256(value):
            raise ValueError("plan hash must be a lowercase SHA-256")
        return value


@dataclass(frozen=True, slots=True)
class CatalogResult:
    root: Path
    commit: ClassificationCatalogCommit
    statistics: CatalogStatistics
    created: bool


@dataclass(frozen=True, slots=True)
class _FetchedArtifact:
    metadata: CatalogSourceArtifact
    payload: bytes


@dataclass(frozen=True, slots=True)
class _ManifestRow:
    source_line: int
    values: dict[str, JsonValue]


@dataclass(frozen=True, slots=True)
class _ManifestDocument:
    doc_id: str
    rows_by_page: tuple[_ManifestRow, ...]

    @property
    def first(self) -> dict[str, JsonValue]:
        return self.rows_by_page[0].values

    @property
    def source_lines_by_page(self) -> list[int]:
        return [item.source_line for item in self.rows_by_page]


@dataclass(frozen=True, slots=True)
class _ReportRow:
    source_line: int
    values: dict[str, str]


@dataclass(frozen=True, slots=True)
class _VerifiedCatalogSnapshot:
    summary: CatalogSnapshotSummary
    aliases: tuple[CatalogLocalAlias, ...]


@dataclass(frozen=True, slots=True)
class _CatalogPlan:
    records: tuple[CatalogDocumentRecord, ...]
    raw_inventory: tuple[CatalogS3Alias, ...]
    source_artifacts: tuple[_FetchedArtifact, ...]
    local_snapshots: tuple[_VerifiedCatalogSnapshot, ...]
    statistics: CatalogStatistics
    catalog_payload: bytes
    raw_inventory_payload: bytes
    statistics_payload: bytes
    bucket_versioning_status: Literal["Enabled", "Suspended"] | None

    @property
    def catalog_sha256(self) -> str:
        return sha256_bytes(self.catalog_payload)

    @property
    def raw_inventory_sha256(self) -> str:
        return sha256_bytes(self.raw_inventory_payload)

    @property
    def statistics_sha256(self) -> str:
        return sha256_bytes(self.statistics_payload)


class ClassificationCatalogError(RuntimeError):
    """Base class for explicit catalog failures."""


class CatalogSourceError(ClassificationCatalogError):
    """A remote artifact or raw inventory violates its pinned contract."""


class CatalogJoinError(ClassificationCatalogError):
    """Classifier, report, raw, or local identities cannot be joined exactly."""


class CatalogIntegrityError(ClassificationCatalogError):
    """A locally published catalog artifact violates its immutable contract."""


def _canonical_utc(value: datetime | str) -> str:
    parsed = datetime.fromisoformat(value) if isinstance(value, str) else value
    if parsed.tzinfo is None or parsed.utcoffset() is None:
        raise ValueError("timestamp must be timezone-aware")
    return parsed.astimezone(UTC).isoformat()


def _filename_uuid(filename: str) -> str | None:
    match = _DOCUMENT_FILENAME.fullmatch(filename)
    return None if match is None else match.group("document_uuid").lower()


def _document_id(*, lineage: str, identity_kind: str, identity: str) -> str:
    return sha256_bytes(
        canonical_json_bytes(
            {
                "identity": identity,
                "identity_kind": identity_kind,
                "lineage": lineage,
                "schema_version": 1,
            }
        )
    )


def _canonical_jsonl(records: Iterable[BaseModel]) -> bytes:
    return b"".join(
        canonical_json_bytes(record.model_dump(mode="json")) + b"\n" for record in records
    )


def _artifact_specs(
    config: ClassificationCatalogConfig,
) -> tuple[tuple[str, CatalogArtifactKind, CatalogArtifactConfig], ...]:
    specs: list[tuple[str, CatalogArtifactKind, CatalogArtifactConfig]] = []
    for lineage in config.lineages:
        for raw_kind in _ARTIFACT_KINDS:
            kind = cast(CatalogArtifactKind, raw_kind)
            artifact = cast(CatalogArtifactConfig | None, getattr(lineage, kind))
            if artifact is not None:
                specs.append((lineage.name, kind, artifact))
    return tuple(sorted(specs, key=lambda item: (item[0], item[1], item[2].key)))


def _new_s3_client(config: ClassificationCatalogConfig) -> Any:
    return boto3.client(
        "s3",
        region_name=config.region,
        config=BotoClientConfig(
            max_pool_connections=config.network_workers,
            retries={"mode": "standard", "total_max_attempts": config.max_attempts},
        ),
    )


def _read_bounded_body(
    body: Any,
    *,
    expected_size: int,
    max_bytes: int,
    chunk_size: int,
) -> bytes:
    chunks: list[bytes] = []
    total = 0
    try:
        while True:
            chunk = body.read(chunk_size)
            if not chunk:
                break
            if not isinstance(chunk, bytes):
                raise CatalogSourceError("S3 response body returned non-bytes content")
            total += len(chunk)
            if total > max_bytes or total > expected_size:
                raise CatalogSourceError("S3 artifact exceeded its declared or configured size")
            chunks.append(chunk)
    finally:
        body.close()
    if total != expected_size:
        raise CatalogSourceError(
            f"S3 artifact body size changed ({total} != {expected_size})"
        )
    return b"".join(chunks)


def _fetch_artifact(
    *,
    client: Any,
    config: ClassificationCatalogConfig,
    lineage: str,
    kind: CatalogArtifactKind,
    artifact: CatalogArtifactConfig,
) -> _FetchedArtifact:
    try:
        head = client.head_object(Bucket=config.bucket, Key=artifact.key)
    except Exception as error:
        raise CatalogSourceError(
            f"cannot inspect catalog source artifact: {artifact.key}"
        ) from error
    size = head.get("ContentLength")
    etag = head.get("ETag")
    last_modified = head.get("LastModified")
    if not isinstance(size, int) or isinstance(size, bool) or size <= 0:
        raise CatalogSourceError(f"catalog source artifact has invalid size: {artifact.key}")
    if size > config.max_source_artifact_bytes:
        raise CatalogSourceError(
            f"catalog source artifact exceeds configured maximum: {artifact.key}"
        )
    if not isinstance(etag, str) or not etag.startswith('"') or not etag.endswith('"'):
        raise CatalogSourceError(f"catalog source artifact has invalid ETag: {artifact.key}")
    if not isinstance(last_modified, datetime):
        raise CatalogSourceError(
            f"catalog source artifact has invalid LastModified: {artifact.key}"
        )
    request: dict[str, Any] = {
        "Bucket": config.bucket,
        "Key": artifact.key,
        "IfMatch": etag,
    }
    version_id = head.get("VersionId")
    if version_id is not None:
        if not isinstance(version_id, str) or not version_id:
            raise CatalogSourceError(f"catalog artifact has invalid VersionId: {artifact.key}")
        request["VersionId"] = version_id
    try:
        response = client.get_object(**request)
    except Exception as error:
        raise CatalogSourceError(
            f"cannot conditionally read catalog source artifact: {artifact.key}"
        ) from error
    if response.get("ETag") != etag or response.get("ContentLength") != size:
        raise CatalogSourceError(f"catalog source artifact metadata changed: {artifact.key}")
    payload = _read_bounded_body(
        response["Body"],
        expected_size=size,
        max_bytes=config.max_source_artifact_bytes,
        chunk_size=config.download_chunk_size_bytes,
    )
    digest = sha256_bytes(payload)
    if digest != artifact.expected_sha256:
        raise CatalogSourceError(
            f"catalog source artifact SHA-256 changed ({digest} != "
            f"{artifact.expected_sha256}): {artifact.key}"
        )
    metadata = CatalogSourceArtifact(
        lineage=lineage,
        kind=kind,
        source_bucket=config.bucket,
        source_key=artifact.key,
        source_size_bytes=size,
        source_etag=etag,
        source_last_modified=_canonical_utc(last_modified),
        source_version_id=version_id,
        sha256=digest,
        catalog_relative_path=PurePosixPath("sources", lineage, artifact.local_name).as_posix(),
    )
    return _FetchedArtifact(metadata=metadata, payload=payload)


def _fetch_artifacts(
    config: ClassificationCatalogConfig,
    client: Any,
    *,
    progress: CatalogProgress | None,
) -> tuple[_FetchedArtifact, ...]:
    specs = _artifact_specs(config)
    fetched: list[_FetchedArtifact] = []
    with ThreadPoolExecutor(
        max_workers=min(config.network_workers, len(specs)),
        thread_name_prefix="catalog-artifact",
    ) as executor:
        futures = {
            executor.submit(
                _fetch_artifact,
                client=client,
                config=config,
                lineage=lineage,
                kind=kind,
                artifact=artifact,
            ): (lineage, kind)
            for lineage, kind, artifact in specs
        }
        for future in as_completed(futures):
            fetched.append(future.result())
            if progress is not None:
                progress("fetch_source_artifact", len(fetched), len(specs))
    return tuple(
        sorted(fetched, key=lambda item: (item.metadata.lineage, item.metadata.kind))
    )


def _meaningful_pdf_key(key: str) -> bool:
    path = PurePosixPath(key)
    return (
        key.lower().endswith(".pdf")
        and "__MACOSX" not in path.parts
        and not path.name.startswith("._")
    )


def _list_raw_source(
    *,
    client: Any,
    config: ClassificationCatalogConfig,
    lineage: str,
    role: Literal["primary", "mirror"],
    prefix: str,
) -> tuple[CatalogS3Alias, ...]:
    records: list[CatalogS3Alias] = []
    try:
        pages = client.get_paginator("list_objects_v2").paginate(
            Bucket=config.bucket,
            Prefix=prefix,
        )
        for page in pages:
            for item in page.get("Contents", []):
                key = item.get("Key")
                if not isinstance(key, str) or not _meaningful_pdf_key(key):
                    continue
                size = item.get("Size")
                etag = item.get("ETag")
                modified = item.get("LastModified")
                if not isinstance(size, int) or isinstance(size, bool) or size <= 0:
                    raise CatalogSourceError(f"raw PDF has invalid size: {key}")
                if not isinstance(etag, str):
                    raise CatalogSourceError(f"raw PDF has invalid ETag: {key}")
                if not isinstance(modified, datetime):
                    raise CatalogSourceError(f"raw PDF has invalid LastModified: {key}")
                algorithms = item.get("ChecksumAlgorithm", [])
                if not isinstance(algorithms, list) or any(
                    not isinstance(value, str) or not value for value in algorithms
                ):
                    raise CatalogSourceError(f"raw PDF has invalid checksum algorithms: {key}")
                checksum_type = item.get("ChecksumType")
                if checksum_type is not None and (
                    not isinstance(checksum_type, str) or not checksum_type
                ):
                    raise CatalogSourceError(f"raw PDF has invalid checksum type: {key}")
                storage_class = item.get("StorageClass", "STANDARD")
                if not isinstance(storage_class, str) or not storage_class:
                    raise CatalogSourceError(f"raw PDF has invalid storage class: {key}")
                records.append(
                    CatalogS3Alias(
                        source_bucket=config.bucket,
                        lineage=lineage,
                        role=role,
                        source_key=key,
                        source_size_bytes=size,
                        source_etag=etag,
                        source_last_modified=_canonical_utc(modified),
                        source_storage_class=storage_class,
                        source_version_id=None,
                        checksum_algorithms=sorted(set(algorithms)),
                        checksum_type=checksum_type,
                    )
                )
    except ClassificationCatalogError:
        raise
    except Exception as error:
        raise CatalogSourceError(f"cannot enumerate raw S3 prefix: {prefix}") from error
    return tuple(sorted(records, key=lambda item: item.source_key))


def _list_raw_inventory(
    config: ClassificationCatalogConfig,
    client: Any,
    *,
    progress: CatalogProgress | None,
) -> tuple[CatalogS3Alias, ...]:
    records: list[CatalogS3Alias] = []
    sources = sorted(config.raw_sources, key=lambda item: (item.lineage, item.role, item.prefix))
    with ThreadPoolExecutor(
        max_workers=min(config.network_workers, len(sources)),
        thread_name_prefix="catalog-inventory",
    ) as executor:
        futures = {
            executor.submit(
                _list_raw_source,
                client=client,
                config=config,
                lineage=source.lineage,
                role=source.role,
                prefix=source.prefix,
            ): source
            for source in sources
        }
        for future in as_completed(futures):
            records.extend(future.result())
            if progress is not None:
                progress("list_raw_source", len(records), -1)
    ordered = tuple(sorted(records, key=lambda item: (item.lineage, item.role, item.source_key)))
    keys = [item.source_key for item in ordered]
    if len(keys) != len(set(keys)):
        raise CatalogSourceError("configured raw prefixes yielded duplicate S3 object keys")
    return ordered


def _json_object_no_duplicates(pairs: list[tuple[str, JsonValue]]) -> dict[str, JsonValue]:
    value: dict[str, JsonValue] = {}
    for key, item in pairs:
        if key in value:
            raise ValueError(f"duplicate JSON object key: {key}")
        value[key] = item
    return value


def _parse_json_object(line: bytes, *, location: str) -> dict[str, JsonValue]:
    try:
        decoded = line.decode("utf-8")
        parsed = json.loads(
            decoded,
            object_pairs_hook=_json_object_no_duplicates,
            parse_constant=lambda value: (_ for _ in ()).throw(
                ValueError(f"non-finite JSON number: {value}")
            ),
        )
    except Exception as error:
        raise CatalogJoinError(f"invalid JSON object at {location}") from error
    if not isinstance(parsed, dict):
        raise CatalogJoinError(f"JSONL row must be an object at {location}")
    return parsed


def _required_value(
    row: Mapping[str, JsonValue],
    key: str,
    expected_type: type[Any],
    *,
    location: str,
) -> Any:
    if key not in row:
        raise CatalogJoinError(f"manifest row lacks {key!r} at {location}")
    value = row[key]
    if expected_type is int:
        valid = isinstance(value, int) and not isinstance(value, bool)
    else:
        valid = isinstance(value, expected_type)
    if not valid:
        raise CatalogJoinError(f"manifest field {key!r} has invalid type at {location}")
    if isinstance(value, str) and not value:
        raise CatalogJoinError(f"manifest field {key!r} is empty at {location}")
    return value


def _parse_manifest(
    artifact: _FetchedArtifact,
    *,
    final: bool,
) -> dict[str, _ManifestDocument]:
    payload = artifact.payload
    if not payload or not payload.endswith(b"\n"):
        raise CatalogJoinError(
            "manifest must be non-empty newline-terminated JSONL: "
            f"{artifact.metadata.source_key}"
        )
    grouped: dict[str, list[_ManifestRow]] = defaultdict(list)
    for line_number, line in enumerate(payload.splitlines(), start=1):
        location = f"{artifact.metadata.source_key}:{line_number}"
        row = _parse_json_object(line, location=location)
        doc_id = _required_value(row, "doc_id", str, location=location)
        _required_value(row, "original_pdf_path", str, location=location)
        _required_value(row, "image_path", str, location=location)
        _required_value(row, "classification_label", str, location=location)
        page_index = _required_value(row, "page_index", int, location=location)
        if page_index < 0:
            raise CatalogJoinError(f"negative page_index at {location}")
        _required_value(row, "is_first_page", bool, location=location)
        _required_value(row, "classification_llm", dict, location=location)
        dummy = _required_value(row, "dummy", dict, location=location)
        _required_value(dummy, "is_dummy", bool, location=location)
        _required_value(row, "triage", dict, location=location)
        if final:
            _required_value(row, "final_classification_label", str, location=location)
        elif "final_classification_label" in row:
            raise CatalogJoinError(f"initial manifest unexpectedly has final label at {location}")
        grouped[doc_id].append(_ManifestRow(source_line=line_number, values=row))

    documents: dict[str, _ManifestDocument] = {}
    for doc_id, raw_rows in grouped.items():
        rows = sorted(
            raw_rows,
            key=lambda item: cast(int, item.values["page_index"]),
        )
        page_indexes = [cast(int, item.values["page_index"]) for item in rows]
        if page_indexes != list(range(len(rows))):
            raise CatalogJoinError(f"manifest pages are not contiguous for {doc_id!r}")
        first_flags = [bool(item.values["is_first_page"]) for item in rows]
        if first_flags != [True, *([False] * (len(rows) - 1))]:
            raise CatalogJoinError(f"is_first_page flags are inconsistent for {doc_id!r}")
        stable_keys: tuple[str, ...] = (
            "doc_id",
            "original_pdf_path",
            "classification_label",
            "classification_llm",
            "dummy",
            "triage",
        )
        if final:
            stable_keys = (*stable_keys, "final_classification_label")
        first = rows[0].values
        for manifest_row in rows[1:]:
            if any(manifest_row.values[key] != first[key] for key in stable_keys):
                raise CatalogJoinError(
                    f"document-level manifest evidence varies by page: {doc_id!r}"
                )
        documents[doc_id] = _ManifestDocument(doc_id=doc_id, rows_by_page=tuple(rows))
    return documents


def _parse_report(
    artifact: _FetchedArtifact,
    *,
    expected_header: tuple[str, ...],
) -> dict[str, _ReportRow]:
    try:
        text = artifact.payload.decode("utf-8-sig")
    except UnicodeDecodeError as error:
        raise CatalogJoinError(
            f"filter report is not UTF-8: {artifact.metadata.source_key}"
        ) from error
    if not text or not text.endswith("\n"):
        raise CatalogJoinError(
            "filter report must be non-empty and newline-terminated: "
            f"{artifact.metadata.source_key}"
        )
    reader = csv.DictReader(io.StringIO(text, newline=""))
    if tuple(reader.fieldnames or ()) != expected_header:
        raise CatalogJoinError(
            f"filter report header differs from expected {expected_header!r}: "
            f"{artifact.metadata.source_key}"
        )
    records: dict[str, _ReportRow] = {}
    for line_number, row in enumerate(reader, start=2):
        if None in row or any(value is None for value in row.values()):
            raise CatalogJoinError(
                f"malformed CSV row at {artifact.metadata.source_key}:{line_number}"
            )
        values = {str(key): str(value) for key, value in row.items()}
        if any(not value for value in values.values()):
            raise CatalogJoinError(
                f"empty CSV value at {artifact.metadata.source_key}:{line_number}"
            )
        doc_id = values["doc_id"]
        if doc_id in records:
            raise CatalogJoinError(f"duplicate report doc_id {doc_id!r}")
        records[doc_id] = _ReportRow(source_line=line_number, values=values)
    return records


def _declared_source_key(original_pdf_path: str) -> str:
    prefix = "/workspace/"
    if not original_pdf_path.startswith(prefix):
        raise CatalogJoinError(
            f"original_pdf_path is not under the documented /workspace root: {original_pdf_path}"
        )
    value = original_pdf_path.removeprefix(prefix)
    _safe_relative_path(value, description="manifest original_pdf_path")
    return value


def _evidence(
    artifact: _FetchedArtifact,
    document: _ManifestDocument,
) -> ManifestDocumentEvidence:
    first = document.first
    return ManifestDocumentEvidence(
        manifest_key=artifact.metadata.source_key,
        manifest_sha256=artifact.metadata.sha256,
        doc_id=document.doc_id,
        original_pdf_path=str(first["original_pdf_path"]),
        declared_source_key=_declared_source_key(str(first["original_pdf_path"])),
        classification_label=str(first["classification_label"]),
        document_page_count=len(document.rows_by_page),
        source_lines_by_page=document.source_lines_by_page,
        classification_llm=cast(dict[str, JsonValue], first["classification_llm"]),
        dummy=cast(dict[str, JsonValue], first["dummy"]),
        triage=cast(dict[str, JsonValue], first["triage"]),
    )


def _final_evidence(
    artifact: _FetchedArtifact,
    document: _ManifestDocument,
) -> FinalDocumentEvidence:
    return FinalDocumentEvidence(
        manifest_key=artifact.metadata.source_key,
        manifest_sha256=artifact.metadata.sha256,
        final_classification_label=str(document.first["final_classification_label"]),
        document_page_count=len(document.rows_by_page),
        source_lines_by_page=document.source_lines_by_page,
    )


def _report_evidence(artifact: _FetchedArtifact, row: _ReportRow) -> ReportEvidence:
    return ReportEvidence(
        report_key=artifact.metadata.source_key,
        report_sha256=artifact.metadata.sha256,
        source_line=row.source_line,
        values=row.values,
    )


def _artifact_map(
    artifacts: tuple[_FetchedArtifact, ...],
) -> dict[tuple[str, CatalogArtifactKind], _FetchedArtifact]:
    result = {(item.metadata.lineage, item.metadata.kind): item for item in artifacts}
    if len(result) != len(artifacts):
        raise CatalogJoinError("source artifacts do not have unique lineage/kind identities")
    return result


def _strip_final(row: Mapping[str, JsonValue]) -> dict[str, JsonValue]:
    return {key: value for key, value in row.items() if key != "final_classification_label"}


def _validate_lineage_chain(
    lineage: CatalogLineageConfig,
    artifacts: Mapping[tuple[str, CatalogArtifactKind], _FetchedArtifact],
) -> tuple[
    dict[str, _ManifestDocument],
    dict[str, _ManifestDocument],
    dict[str, _ReportRow],
    dict[str, _ReportRow],
    dict[str, _ReportRow],
]:
    initial_artifact = artifacts[(lineage.name, "initial_manifest")]
    final_artifact = artifacts[(lineage.name, "final_manifest")]
    dropped_artifact = artifacts[(lineage.name, "dropped_report")]
    relabeled_artifact = artifacts[(lineage.name, "relabeled_report")]
    initial = _parse_manifest(initial_artifact, final=False)
    final = _parse_manifest(final_artifact, final=True)
    dropped = _parse_report(dropped_artifact, expected_header=("doc_id", "reason"))
    relabeled = _parse_report(
        relabeled_artifact,
        expected_header=("doc_id", "from", "to", "confidence", "rule_id"),
    )
    review: dict[str, _ReportRow] = {}
    if lineage.review_report is not None:
        review = _parse_report(
            artifacts[(lineage.name, "review_report")],
            expected_header=("doc_id", "reason"),
        )

    if not set(final).issubset(initial):
        examples = sorted(set(final) - set(initial))[:5]
        raise CatalogJoinError(f"final manifest has documents absent from initial: {examples!r}")
    for doc_id, final_document in final.items():
        initial_document = initial[doc_id]
        initial_rows = {
            cast(int, item.values["page_index"]): item.values
            for item in initial_document.rows_by_page
        }
        final_rows = {
            cast(int, item.values["page_index"]): _strip_final(item.values)
            for item in final_document.rows_by_page
        }
        if initial_rows != final_rows:
            raise CatalogJoinError(f"final manifest altered initial page evidence: {doc_id!r}")
    if not set(dropped).issubset(initial) or set(dropped) & set(final):
        raise CatalogJoinError("dropped report does not describe only initial, non-final documents")
    if not set(relabeled).issubset(final):
        raise CatalogJoinError("relabeled report contains a document absent from final manifest")
    if not set(review).issubset(initial):
        raise CatalogJoinError("review report contains a document absent from initial manifest")

    expected_relabeled = {
        doc_id
        for doc_id, item in final.items()
        if item.first["classification_label"] != item.first["final_classification_label"]
    }
    if set(relabeled) != expected_relabeled:
        raise CatalogJoinError("relabeled report does not exactly cover changed final labels")
    for doc_id, row in relabeled.items():
        first = final[doc_id].first
        classification_llm = cast(dict[str, JsonValue], first["classification_llm"])
        confidence = classification_llm.get("confidence")
        if (
            not isinstance(confidence, (int, float))
            or isinstance(confidence, bool)
            or row.values["from"] != first["classification_label"]
            or row.values["to"] != first["final_classification_label"]
            or row.values["confidence"] != f"{float(confidence):.3f}"
        ):
            raise CatalogJoinError(f"relabeled report values differ from manifest: {doc_id!r}")
    return initial, final, dropped, relabeled, review


def _canonical_path(path: Path, *, description: str) -> Path:
    absolute = Path(os.path.abspath(path))
    try:
        resolved = path.resolve(strict=True)
    except OSError as error:
        raise CatalogIntegrityError(f"{description} does not exist: {path}") from error
    if path != absolute or resolved != absolute:
        raise CatalogIntegrityError(
            f"{description} must be absolute and must not traverse symbolic links: {path}"
        )
    return absolute


def _verified_local_snapshots(
    config: ClassificationCatalogConfig,
    *,
    progress: CatalogProgress | None,
) -> tuple[_VerifiedCatalogSnapshot, ...]:
    verified: list[_VerifiedCatalogSnapshot] = []
    destination = Path(os.path.abspath(config.destination_root))
    for index, source in enumerate(config.local_snapshots, start=1):
        config_path = _canonical_path(
            Path(source.snapshot_config),
            description="snapshot configuration",
        )
        config_payload = read_regular_file_bytes(config_path)
        snapshot_config = load_snapshot_config(config_path)
        if read_regular_file_bytes(config_path) != config_payload:
            raise CatalogIntegrityError(
                f"snapshot configuration changed while loading: {config_path}"
            )
        result = verify_snapshot(snapshot_config)
        if (
            destination == result.root
            or destination in result.root.parents
            or result.root in destination.parents
        ):
            raise CatalogIntegrityError("catalog destination must not overlap a local snapshot")
        commit_payload = read_regular_file_bytes(result.root / "snapshot.json")
        manifest_payload = read_regular_file_bytes(result.root / result.commit.manifest_path)
        if sha256_bytes(manifest_payload) != result.commit.manifest_sha256:
            raise CatalogIntegrityError(f"verified snapshot manifest changed: {result.root}")
        records = _load_snapshot_records(
            manifest_payload,
            result.root / result.commit.manifest_path,
        )
        summary = CatalogSnapshotSummary(
            snapshot_config_path=str(config_path),
            snapshot_root=str(result.root),
            snapshot_config_sha256=sha256_bytes(config_payload),
            snapshot_commit_sha256=sha256_bytes(commit_payload),
            snapshot_manifest_sha256=result.commit.manifest_sha256,
            snapshot_selection_sha256=result.commit.selection_sha256,
            document_count=len(records),
            source_bytes=sum(item.source_size_bytes for item in records),
        )
        aliases = tuple(
            CatalogLocalAlias(
                snapshot_config_path=str(config_path),
                snapshot_root=str(result.root),
                snapshot_config_sha256=summary.snapshot_config_sha256,
                snapshot_commit_sha256=summary.snapshot_commit_sha256,
                snapshot_manifest_sha256=summary.snapshot_manifest_sha256,
                snapshot_selection_sha256=summary.snapshot_selection_sha256,
                snapshot_record=record,
            )
            for record in records
        )
        verified.append(_VerifiedCatalogSnapshot(summary=summary, aliases=aliases))
        if progress is not None:
            progress("verify_local_snapshot", index, len(config.local_snapshots))
    identities = [
        (item.summary.snapshot_config_sha256, item.summary.snapshot_manifest_sha256)
        for item in verified
    ]
    if len(identities) != len(set(identities)):
        raise CatalogJoinError("local snapshots must have unique verified identities")
    return tuple(
        sorted(
            verified,
            key=lambda item: (
                item.summary.snapshot_config_sha256,
                item.summary.snapshot_manifest_sha256,
            ),
        )
    )


@dataclass(slots=True)
class _MutableDocument:
    lineage: str
    filename: str
    initial: ManifestDocumentEvidence | None
    final: FinalDocumentEvidence | None
    dropped_report: ReportEvidence | None
    relabeled_report: ReportEvidence | None
    review_report: ReportEvidence | None
    s3_aliases: list[CatalogS3Alias]
    local_aliases: list[CatalogLocalAlias]
    raw_identity: str | None = None


def _build_documents(
    config: ClassificationCatalogConfig,
    artifacts: tuple[_FetchedArtifact, ...],
    raw_inventory: tuple[CatalogS3Alias, ...],
    snapshots: tuple[_VerifiedCatalogSnapshot, ...],
) -> tuple[CatalogDocumentRecord, ...]:
    artifact_lookup = _artifact_map(artifacts)
    documents: dict[tuple[str, str], _MutableDocument] = {}
    declared_keys: dict[str, tuple[str, str]] = {}
    filenames: dict[tuple[str, str], tuple[str, str]] = {}

    for lineage in sorted(config.lineages, key=lambda item: item.name):
        initial, final, dropped, relabeled, review = _validate_lineage_chain(
            lineage,
            artifact_lookup,
        )
        initial_artifact = artifact_lookup[(lineage.name, "initial_manifest")]
        final_artifact = artifact_lookup[(lineage.name, "final_manifest")]
        dropped_artifact = artifact_lookup[(lineage.name, "dropped_report")]
        relabeled_artifact = artifact_lookup[(lineage.name, "relabeled_report")]
        review_artifact = artifact_lookup.get((lineage.name, "review_report"))
        for doc_id, item in sorted(initial.items()):
            initial_evidence = _evidence(initial_artifact, item)
            filename = PurePosixPath(initial_evidence.declared_source_key).name
            identity = (lineage.name, doc_id)
            filename_identity = (lineage.name, filename)
            if initial_evidence.declared_source_key in declared_keys:
                raise CatalogJoinError(
                    f"declared source key occurs in multiple manifest documents: "
                    f"{initial_evidence.declared_source_key}"
                )
            if filename_identity in filenames:
                raise CatalogJoinError(
                    f"manifest filename is not unique within lineage {lineage.name}: {filename}"
                )
            declared_keys[initial_evidence.declared_source_key] = identity
            filenames[filename_identity] = identity
            final_item = final.get(doc_id)
            dropped_item = dropped.get(doc_id)
            relabeled_item = relabeled.get(doc_id)
            review_item = review.get(doc_id)
            documents[identity] = _MutableDocument(
                lineage=lineage.name,
                filename=filename,
                initial=initial_evidence,
                final=(
                    None if final_item is None else _final_evidence(final_artifact, final_item)
                ),
                dropped_report=(
                    None
                    if dropped_item is None
                    else _report_evidence(dropped_artifact, dropped_item)
                ),
                relabeled_report=(
                    None
                    if relabeled_item is None
                    else _report_evidence(relabeled_artifact, relabeled_item)
                ),
                review_report=(
                    None
                    if review_item is None or review_artifact is None
                    else _report_evidence(review_artifact, review_item)
                ),
                s3_aliases=[],
                local_aliases=[],
            )

    raw_by_key = {item.source_key: item for item in raw_inventory}
    if len(raw_by_key) != len(raw_inventory):
        raise CatalogJoinError("raw inventory contains duplicate S3 keys")
    raw_targets: dict[str, tuple[str, str]] = {}
    for alias in raw_inventory:
        target = declared_keys.get(alias.source_key)
        if alias.role == "mirror":
            filename_target = filenames.get((alias.lineage, PurePosixPath(alias.source_key).name))
            if filename_target is None:
                raise CatalogJoinError(
                    "mirror raw PDF does not match exactly one classified filename: "
                    f"{alias.source_key}"
                )
            target = filename_target
        if target is None:
            if alias.role != "primary":
                raise CatalogJoinError(f"unmatched mirror alias: {alias.source_key}")
            raw_identity = (alias.lineage, f"raw:{alias.source_key}")
            filename_identity = (alias.lineage, PurePosixPath(alias.source_key).name)
            if filename_identity in filenames:
                raise CatalogJoinError(
                    f"unclassified raw key collides with a classified filename: {alias.source_key}"
                )
            if raw_identity in documents:
                raise CatalogJoinError(f"duplicate raw-only document identity: {alias.source_key}")
            filenames[filename_identity] = raw_identity
            documents[raw_identity] = _MutableDocument(
                lineage=alias.lineage,
                filename=PurePosixPath(alias.source_key).name,
                initial=None,
                final=None,
                dropped_report=None,
                relabeled_report=None,
                review_report=None,
                s3_aliases=[],
                local_aliases=[],
                raw_identity=alias.source_key,
            )
            target = raw_identity
        if documents[target].lineage != alias.lineage:
            raise CatalogJoinError(f"raw alias lineage differs from target: {alias.source_key}")
        documents[target].s3_aliases.append(alias)
        raw_targets[alias.source_key] = target

    local_alias_count = 0
    for snapshot in snapshots:
        for local_alias in snapshot.aliases:
            source = local_alias.snapshot_record
            inventory = raw_by_key.get(source.source_key)
            if inventory is None:
                raise CatalogJoinError(
                    "local snapshot source is absent from frozen raw inventory: "
                    f"{source.source_key}"
                )
            if (
                inventory.source_size_bytes != source.source_size_bytes
                or inventory.source_etag != source.source_etag
                or inventory.source_last_modified != source.source_last_modified
            ):
                raise CatalogJoinError(
                    f"local snapshot S3 identity differs from raw inventory: {source.source_key}"
                )
            target = raw_targets.get(source.source_key)
            if target is None:
                raise CatalogJoinError(
                    f"local snapshot source has no catalog document: {source.source_key}"
                )
            documents[target].local_aliases.append(local_alias)
            local_alias_count += 1
    if local_alias_count != sum(len(item.aliases) for item in snapshots):
        raise CatalogJoinError("not every local snapshot record was attached exactly once")

    records: list[CatalogDocumentRecord] = []
    for document in documents.values():
        if document.initial is None:
            status: CatalogStatus = "never_classified"
            id_kind = "raw_source_key"
            id_value = document.raw_identity
            if id_value is None:
                raise CatalogJoinError("raw-only document lacks its source-key identity")
        elif document.final is not None:
            status = "retained_final"
            id_kind = "classifier_doc_id"
            id_value = document.initial.doc_id
        elif document.dropped_report is not None:
            status = "excluded_reported"
            id_kind = "classifier_doc_id"
            id_value = document.initial.doc_id
        else:
            status = "missing_final_without_drop_report"
            id_kind = "classifier_doc_id"
            id_value = document.initial.doc_id
        ordered_s3 = sorted(document.s3_aliases, key=lambda item: (item.role, item.source_key))
        ordered_local = sorted(
            document.local_aliases,
            key=lambda item: (
                item.snapshot_record.source_key,
                item.snapshot_manifest_sha256,
                item.snapshot_config_sha256,
            ),
        )
        records.append(
            CatalogDocumentRecord(
                schema_version=1,
                catalog_document_id=_document_id(
                    lineage=document.lineage,
                    identity_kind=id_kind,
                    identity=id_value,
                ),
                lineage=document.lineage,
                status=status,
                document_filename=document.filename,
                document_uuid=_filename_uuid(document.filename),
                initial=document.initial,
                final=document.final,
                dropped_report=document.dropped_report,
                relabeled_report=document.relabeled_report,
                review_report=document.review_report,
                declared_source_present=(
                    document.initial is not None
                    and any(
                        item.role == "primary"
                        and item.source_key == document.initial.declared_source_key
                        for item in ordered_s3
                    )
                ),
                s3_source_aliases=ordered_s3,
                local_snapshot_aliases=ordered_local,
            )
        )
    ordered = tuple(
        sorted(
            records,
            key=lambda item: (item.lineage, item.document_filename, item.catalog_document_id),
        )
    )
    ids = [item.catalog_document_id for item in ordered]
    if len(ids) != len(set(ids)):
        raise CatalogJoinError("catalog_document_id collision")
    if sum(len(item.s3_source_aliases) for item in ordered) != len(raw_inventory):
        raise CatalogJoinError("not every raw inventory object was attached exactly once")
    return ordered


def _counter(values: Iterable[str]) -> dict[str, int]:
    return dict(sorted(Counter(values).items()))


def _is_dummy(record: CatalogDocumentRecord) -> bool:
    if record.initial is None:
        return False
    value = record.initial.dummy.get("is_dummy")
    if not isinstance(value, bool):
        raise CatalogJoinError("catalog initial dummy evidence lacks boolean is_dummy")
    return value


def _llm_prediction(record: CatalogDocumentRecord) -> str | None:
    if record.initial is None:
        return None
    value = record.initial.classification_llm.get("predicted")
    if not isinstance(value, str) or not value:
        raise CatalogJoinError("catalog initial classifier evidence lacks predicted label")
    return value


def _lineage_statistics(records: list[CatalogDocumentRecord]) -> LineageStatistics:
    classified = [item for item in records if item.initial is not None]
    retained = [item for item in records if item.final is not None]
    initial_evidence = [cast(ManifestDocumentEvidence, item.initial) for item in classified]
    final_evidence = [cast(FinalDocumentEvidence, item.final) for item in retained]
    return LineageStatistics(
        catalog_documents=len(records),
        initial_documents=len(classified),
        initial_page_rows=sum(item.document_page_count for item in initial_evidence),
        final_documents=len(retained),
        final_page_rows=sum(item.document_page_count for item in final_evidence),
        dummy_documents=sum(_is_dummy(item) for item in classified),
        reported_drop_documents=sum(item.dropped_report is not None for item in records),
        reported_relabel_documents=sum(item.relabeled_report is not None for item in records),
        review_documents=sum(item.review_report is not None for item in records),
        declared_source_present_documents=sum(item.declared_source_present for item in records),
        declared_source_missing_documents=sum(
            item.initial is not None and not item.declared_source_present for item in records
        ),
        documents_with_any_s3_source=sum(bool(item.s3_source_aliases) for item in records),
        documents_with_local_snapshot=sum(bool(item.local_snapshot_aliases) for item in records),
        s3_source_aliases=sum(len(item.s3_source_aliases) for item in records),
        local_snapshot_aliases=sum(len(item.local_snapshot_aliases) for item in records),
        status_counts=_counter(item.status for item in records),
        initial_label_counts=_counter(item.classification_label for item in initial_evidence),
        final_label_counts=_counter(
            item.final_classification_label for item in final_evidence
        ),
    )


def _statistics(
    records: tuple[CatalogDocumentRecord, ...],
    raw_inventory: tuple[CatalogS3Alias, ...],
) -> CatalogStatistics:
    classified = [item for item in records if item.initial is not None]
    retained = [item for item in records if item.final is not None]
    initial_evidence = [cast(ManifestDocumentEvidence, item.initial) for item in classified]
    final_evidence = [cast(FinalDocumentEvidence, item.final) for item in retained]
    primary = [item for item in raw_inventory if item.role == "primary"]
    mirrors = [item for item in raw_inventory if item.role == "mirror"]
    local_aliases = [alias for item in records for alias in item.local_snapshot_aliases]
    lineage_groups: dict[str, list[CatalogDocumentRecord]] = defaultdict(list)
    for item in records:
        lineage_groups[item.lineage].append(item)
    return CatalogStatistics(
        schema_version=1,
        catalog_documents=len(records),
        initial_documents=len(classified),
        initial_page_rows=sum(item.document_page_count for item in initial_evidence),
        final_documents=len(retained),
        final_page_rows=sum(item.document_page_count for item in final_evidence),
        dummy_documents=sum(_is_dummy(item) for item in classified),
        reported_drop_documents=sum(item.dropped_report is not None for item in records),
        reported_relabel_documents=sum(item.relabeled_report is not None for item in records),
        review_documents=sum(item.review_report is not None for item in records),
        declared_source_present_documents=sum(item.declared_source_present for item in records),
        declared_source_missing_documents=sum(
            item.initial is not None and not item.declared_source_present for item in records
        ),
        documents_with_any_s3_source=sum(bool(item.s3_source_aliases) for item in records),
        documents_without_any_s3_source=sum(not item.s3_source_aliases for item in records),
        raw_only_documents=sum(item.status == "never_classified" for item in records),
        s3_source_aliases=len(raw_inventory),
        s3_primary_objects=len(primary),
        s3_primary_bytes=sum(item.source_size_bytes for item in primary),
        s3_mirror_objects=len(mirrors),
        s3_mirror_bytes=sum(item.source_size_bytes for item in mirrors),
        local_snapshot_aliases=len(local_aliases),
        documents_with_local_snapshot=sum(bool(item.local_snapshot_aliases) for item in records),
        local_unique_content_sha256=len(
            {item.snapshot_record.source_sha256 for item in local_aliases}
        ),
        status_counts=_counter(item.status for item in records),
        initial_label_counts=_counter(item.classification_label for item in initial_evidence),
        initial_llm_prediction_counts=_counter(
            value for item in classified if (value := _llm_prediction(item)) is not None
        ),
        final_label_counts=_counter(
            item.final_classification_label for item in final_evidence
        ),
        drop_reason_counts=_counter(
            item.dropped_report.values["reason"]
            for item in records
            if item.dropped_report is not None
        ),
        review_reason_counts=_counter(
            item.review_report.values["reason"]
            for item in records
            if item.review_report is not None
        ),
        by_lineage={
            lineage: _lineage_statistics(items)
            for lineage, items in sorted(lineage_groups.items())
        },
    )


def _assemble_plan(
    *,
    config: ClassificationCatalogConfig,
    artifacts: tuple[_FetchedArtifact, ...],
    raw_inventory: tuple[CatalogS3Alias, ...],
    snapshots: tuple[_VerifiedCatalogSnapshot, ...],
    bucket_versioning_status: Literal["Enabled", "Suspended"] | None,
) -> _CatalogPlan:
    ordered_artifacts = tuple(
        sorted(artifacts, key=lambda item: (item.metadata.lineage, item.metadata.kind))
    )
    ordered_inventory = tuple(
        sorted(raw_inventory, key=lambda item: (item.lineage, item.role, item.source_key))
    )
    ordered_snapshots = tuple(
        sorted(
            snapshots,
            key=lambda item: (
                item.summary.snapshot_config_sha256,
                item.summary.snapshot_manifest_sha256,
            ),
        )
    )
    records = _build_documents(
        config,
        ordered_artifacts,
        ordered_inventory,
        ordered_snapshots,
    )
    statistics = _statistics(records, ordered_inventory)
    catalog_payload = _canonical_jsonl(records)
    inventory_payload = _canonical_jsonl(ordered_inventory)
    statistics_payload = canonical_json_bytes(statistics.model_dump(mode="json")) + b"\n"
    return _CatalogPlan(
        records=records,
        raw_inventory=ordered_inventory,
        source_artifacts=ordered_artifacts,
        local_snapshots=ordered_snapshots,
        statistics=statistics,
        catalog_payload=catalog_payload,
        raw_inventory_payload=inventory_payload,
        statistics_payload=statistics_payload,
        bucket_versioning_status=bucket_versioning_status,
    )


def _remote_plan(
    config: ClassificationCatalogConfig,
    *,
    client: Any,
    progress: CatalogProgress | None,
) -> _CatalogPlan:
    try:
        versioning_response = client.get_bucket_versioning(Bucket=config.bucket)
    except Exception as error:
        raise CatalogSourceError("cannot inspect source bucket versioning") from error
    versioning = versioning_response.get("Status")
    if versioning not in {None, "Enabled", "Suspended"}:
        raise CatalogSourceError(
            f"source bucket returned invalid versioning status: {versioning!r}"
        )
    artifacts = _fetch_artifacts(config, client, progress=progress)
    raw_inventory = _list_raw_inventory(config, client, progress=progress)
    snapshots = _verified_local_snapshots(config, progress=progress)
    return _assemble_plan(
        config=config,
        artifacts=artifacts,
        raw_inventory=raw_inventory,
        snapshots=snapshots,
        bucket_versioning_status=versioning,
    )


def _configuration_sha256(config: ClassificationCatalogConfig) -> str:
    return sha256_bytes(canonical_json_bytes(config.model_dump(mode="json")))


def _build_commit(
    config: ClassificationCatalogConfig,
    plan: _CatalogPlan,
) -> ClassificationCatalogCommit:
    return ClassificationCatalogCommit(
        schema_version=1,
        source_bucket=config.bucket,
        source_region=config.region,
        source_bucket_versioning_status=plan.bucket_versioning_status,
        configuration_sha256=_configuration_sha256(config),
        catalog_path="catalog.jsonl",
        catalog_sha256=plan.catalog_sha256,
        raw_inventory_path="raw-inventory.jsonl",
        raw_inventory_sha256=plan.raw_inventory_sha256,
        statistics_path="statistics.json",
        statistics_sha256=plan.statistics_sha256,
        source_artifacts=[item.metadata for item in plan.source_artifacts],
        local_snapshots=[item.summary for item in plan.local_snapshots],
        document_count=len(plan.records),
        s3_source_alias_count=len(plan.raw_inventory),
        local_snapshot_alias_count=sum(
            len(item.local_snapshot_aliases) for item in plan.records
        ),
    )


def plan_catalog(
    config: ClassificationCatalogConfig,
    *,
    s3_client: Any | None = None,
    progress: CatalogProgress | None = None,
) -> CatalogPlanSummary:
    """Build and validate the complete plan without publishing local catalog files."""

    client = _new_s3_client(config) if s3_client is None else s3_client
    plan = _remote_plan(config, client=client, progress=progress)
    return CatalogPlanSummary(
        catalog_sha256=plan.catalog_sha256,
        raw_inventory_sha256=plan.raw_inventory_sha256,
        statistics_sha256=plan.statistics_sha256,
        statistics=plan.statistics,
    )


def _publish_artifact(path: Path, payload: bytes) -> str:
    try:
        return _publish_digest_bound_artifact(path, payload)
    except Exception as error:
        raise CatalogIntegrityError(f"cannot publish immutable catalog artifact: {path}") from error


def _expected_catalog_files(
    root: Path,
    source_artifacts: Iterable[CatalogSourceArtifact],
) -> set[Path]:
    artifact_paths = {
        root / Path(*PurePosixPath(item.catalog_relative_path).parts)
        for item in source_artifacts
    }
    digest_paths = {path.with_name(path.name + ".sha256") for path in artifact_paths}
    core = {
        root / "catalog.jsonl",
        root / "catalog.jsonl.sha256",
        root / "raw-inventory.jsonl",
        root / "raw-inventory.jsonl.sha256",
        root / "statistics.json",
        root / "statistics.json.sha256",
        root / "catalog.json",
    }
    return artifact_paths | digest_paths | core


def _verify_file_tree(
    root: Path,
    source_artifacts: Iterable[CatalogSourceArtifact],
) -> None:
    expected = _expected_catalog_files(root, source_artifacts)
    actual = set(_safe_walk_files(root))
    if actual != expected:
        missing = sorted(str(path) for path in expected - actual)
        unexpected = sorted(str(path) for path in actual - expected)
        raise CatalogIntegrityError(
            f"catalog file tree differs from commit; missing={missing[:3]!r}, "
            f"unexpected={unexpected[:3]!r}"
        )


def materialize_catalog(
    config: ClassificationCatalogConfig,
    *,
    s3_client: Any | None = None,
    progress: CatalogProgress | None = None,
) -> CatalogResult:
    """Publish or fully verify the content-addressed classification catalog."""

    root = _safe_snapshot_directory(Path(config.destination_root), create=True)
    commit_path = root / "catalog.json"
    if commit_path.exists() or commit_path.is_symlink():
        return verify_catalog(config, progress=progress)
    client = _new_s3_client(config) if s3_client is None else s3_client
    plan = _remote_plan(config, client=client, progress=progress)
    if plan.catalog_sha256 != config.expected_catalog_sha256:
        raise CatalogJoinError(
            f"catalog SHA-256 changed ({plan.catalog_sha256} != "
            f"{config.expected_catalog_sha256})"
        )
    if plan.raw_inventory_sha256 != config.expected_raw_inventory_sha256:
        raise CatalogSourceError(
            f"raw inventory SHA-256 changed ({plan.raw_inventory_sha256} != "
            f"{config.expected_raw_inventory_sha256})"
        )
    for item in plan.source_artifacts:
        target = root / Path(*PurePosixPath(item.metadata.catalog_relative_path).parts)
        if _publish_artifact(target, item.payload) != item.metadata.sha256:
            raise CatalogIntegrityError(f"published source artifact digest changed: {target}")
    if _publish_artifact(root / "raw-inventory.jsonl", plan.raw_inventory_payload) != (
        plan.raw_inventory_sha256
    ):
        raise CatalogIntegrityError("published raw inventory digest changed")
    if _publish_artifact(root / "catalog.jsonl", plan.catalog_payload) != plan.catalog_sha256:
        raise CatalogIntegrityError("published catalog digest changed")
    if _publish_artifact(root / "statistics.json", plan.statistics_payload) != (
        plan.statistics_sha256
    ):
        raise CatalogIntegrityError("published catalog statistics digest changed")
    commit = _build_commit(config, plan)
    try:
        atomic_publish_json(commit_path, commit.model_dump(mode="json"))
    except AtomicConflictError as error:
        raise CatalogIntegrityError(
            f"catalog commit publication conflict: {commit_path}"
        ) from error
    _verify_file_tree(root, commit.source_artifacts)
    return CatalogResult(root=root, commit=commit, statistics=plan.statistics, created=True)


def _load_catalog_records(payload: bytes, path: Path) -> tuple[CatalogDocumentRecord, ...]:
    if not payload or not payload.endswith(b"\n"):
        raise CatalogIntegrityError(f"catalog must be non-empty newline-terminated JSONL: {path}")
    records: list[CatalogDocumentRecord] = []
    for line_number, line in enumerate(payload.splitlines(), start=1):
        try:
            records.append(CatalogDocumentRecord.model_validate_json(line, strict=True))
        except Exception as error:
            raise CatalogIntegrityError(f"invalid catalog row {line_number}: {path}") from error
    ordered = tuple(
        sorted(
            records,
            key=lambda item: (item.lineage, item.document_filename, item.catalog_document_id),
        )
    )
    if _canonical_jsonl(ordered) != payload:
        raise CatalogIntegrityError("catalog rows are not canonical and deterministically sorted")
    return ordered


def _load_raw_inventory(payload: bytes, path: Path) -> tuple[CatalogS3Alias, ...]:
    if not payload or not payload.endswith(b"\n"):
        raise CatalogIntegrityError(
            f"raw inventory must be non-empty newline-terminated JSONL: {path}"
        )
    records: list[CatalogS3Alias] = []
    for line_number, line in enumerate(payload.splitlines(), start=1):
        try:
            records.append(CatalogS3Alias.model_validate_json(line, strict=True))
        except Exception as error:
            raise CatalogIntegrityError(
                f"invalid raw inventory row {line_number}: {path}"
            ) from error
    ordered = tuple(sorted(records, key=lambda item: (item.lineage, item.role, item.source_key)))
    if _canonical_jsonl(ordered) != payload:
        raise CatalogIntegrityError("raw inventory is not canonical and deterministically sorted")
    return ordered


def _local_artifacts_from_commit(
    root: Path,
    commit: ClassificationCatalogCommit,
) -> tuple[_FetchedArtifact, ...]:
    fetched: list[_FetchedArtifact] = []
    for metadata in commit.source_artifacts:
        path = root / Path(*PurePosixPath(metadata.catalog_relative_path).parts)
        try:
            payload = _read_digest_bound_artifact(path, metadata.sha256)
        except Exception as error:
            raise CatalogIntegrityError(f"cannot verify pinned catalog source: {path}") from error
        if len(payload) != metadata.source_size_bytes:
            raise CatalogIntegrityError(f"pinned catalog source size changed: {path}")
        fetched.append(_FetchedArtifact(metadata=metadata, payload=payload))
    return tuple(sorted(fetched, key=lambda item: (item.metadata.lineage, item.metadata.kind)))


def verify_catalog(
    config: ClassificationCatalogConfig,
    *,
    progress: CatalogProgress | None = None,
) -> CatalogResult:
    """Offline verification of source copies, raw inventory, snapshots, joins, and commit."""

    root = _safe_snapshot_directory(Path(config.destination_root), create=False)
    commit_path = root / "catalog.json"
    try:
        commit_payload = read_regular_file_bytes(commit_path)
        commit = ClassificationCatalogCommit.model_validate_json(commit_payload, strict=True)
    except Exception as error:
        raise CatalogIntegrityError(f"invalid or missing catalog commit: {commit_path}") from error
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
        raise CatalogIntegrityError("catalog commit is not canonical")
    if commit.source_bucket != config.bucket or commit.source_region != config.region:
        raise CatalogIntegrityError("catalog source bucket/region differs from configuration")
    if commit.configuration_sha256 != _configuration_sha256(config):
        raise CatalogIntegrityError("catalog configuration differs from committed configuration")
    if commit.catalog_sha256 != config.expected_catalog_sha256:
        raise CatalogIntegrityError("catalog digest differs from configuration")
    if commit.raw_inventory_sha256 != config.expected_raw_inventory_sha256:
        raise CatalogIntegrityError("raw inventory digest differs from configuration")
    expected_specs = {
        (lineage, kind, artifact.key, artifact.expected_sha256)
        for lineage, kind, artifact in _artifact_specs(config)
    }
    committed_specs = {
        (item.lineage, item.kind, item.source_key, item.sha256)
        for item in commit.source_artifacts
    }
    if committed_specs != expected_specs:
        raise CatalogIntegrityError("committed source artifacts differ from configuration")

    artifacts = _local_artifacts_from_commit(root, commit)
    try:
        inventory_payload = _read_digest_bound_artifact(
            root / commit.raw_inventory_path,
            commit.raw_inventory_sha256,
        )
        catalog_payload = _read_digest_bound_artifact(
            root / commit.catalog_path,
            commit.catalog_sha256,
        )
        statistics_payload = _read_digest_bound_artifact(
            root / commit.statistics_path,
            commit.statistics_sha256,
        )
    except Exception as error:
        raise CatalogIntegrityError("cannot verify a digest-bound catalog output") from error
    raw_inventory = _load_raw_inventory(inventory_payload, root / commit.raw_inventory_path)
    stored_records = _load_catalog_records(catalog_payload, root / commit.catalog_path)
    try:
        stored_statistics = CatalogStatistics.model_validate_json(
            statistics_payload,
            strict=True,
        )
    except Exception as error:
        raise CatalogIntegrityError("invalid catalog statistics") from error
    if statistics_payload != (
        canonical_json_bytes(stored_statistics.model_dump(mode="json")) + b"\n"
    ):
        raise CatalogIntegrityError("catalog statistics are not canonical")
    snapshots = _verified_local_snapshots(config, progress=progress)
    plan = _assemble_plan(
        config=config,
        artifacts=artifacts,
        raw_inventory=raw_inventory,
        snapshots=snapshots,
        bucket_versioning_status=commit.source_bucket_versioning_status,
    )
    if plan.catalog_payload != catalog_payload or plan.records != stored_records:
        raise CatalogIntegrityError("catalog rows differ from pinned sources and local snapshots")
    if plan.statistics_payload != statistics_payload or plan.statistics != stored_statistics:
        raise CatalogIntegrityError("catalog statistics differ from recomputed values")
    recomputed_commit = _build_commit(config, plan)
    if recomputed_commit != commit:
        raise CatalogIntegrityError("catalog commit differs from recomputed provenance and totals")
    _verify_file_tree(root, commit.source_artifacts)
    return CatalogResult(root=root, commit=commit, statistics=stored_statistics, created=False)
