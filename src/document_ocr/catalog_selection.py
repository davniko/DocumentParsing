"""Immutable catalog-derived local surfaces for full OCR extraction runs.

Unlike the bounded quality-stratified pilot selector, this module selects the
entire locally available, non-dummy surface for one or more final classifier
labels.  Previously selected pilot content and byte-identical catalog aliases
are removed by SHA-256.  Every classifier alias remains in the manifest, so a
classification conflict is visible even though OCR is performed only once.
"""

from __future__ import annotations

import json
import os
import re
import stat
from collections import Counter, defaultdict
from collections.abc import Callable, Iterable
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from datetime import date
from pathlib import Path, PurePosixPath
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from document_ocr.atomic import AtomicConflictError, atomic_publish_json, read_regular_file_bytes
from document_ocr.classification_catalog import (
    CatalogDocumentRecord,
    CatalogLocalAlias,
    CatalogResult,
    verify_catalog,
)
from document_ocr.config import (
    CatalogExtractionSelectionConfig,
    load_catalog_config,
)
from document_ocr.hashing import canonical_json_bytes, canonical_json_sha256, sha256_bytes
from document_ocr.pilot import PilotCommit, PilotDocumentRecord
from document_ocr.snapshot import (
    _hash_local_pdf,
    _publish_digest_bound_artifact,
    _read_digest_bound_artifact,
    _safe_snapshot_directory,
    _safe_walk_files,
)

_DOCUMENT_FILENAME = re.compile(
    r"^(?P<document_date>\d{4}-\d{2}-\d{2})_"
    r"(?P<document_uuid>[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[1-5][0-9a-fA-F]{3}-"
    r"[89aAbB][0-9a-fA-F]{3}-[0-9a-fA-F]{12})\.pdf$"
)

SelectionProgress = Callable[[str, int, int], None]


class _FrozenRecord(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, strict=True, allow_inf_nan=False)


def _is_sha256(value: str) -> bool:
    return len(value) == 64 and all(character in "0123456789abcdef" for character in value)


def _canonical_record_payload(record: BaseModel) -> bytes:
    return (
        json.dumps(
            record.model_dump(mode="json"),
            allow_nan=False,
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        ).encode("utf-8")
        + b"\n"
    )


def _canonical_jsonl(records: Iterable[BaseModel]) -> bytes:
    return b"".join(
        canonical_json_bytes(record.model_dump(mode="json")) + b"\n" for record in records
    )


class CatalogSelectionDocumentRecord(_FrozenRecord):
    """One unique PDF payload and all classifier rows that resolve to it."""

    schema_version: Literal[1]
    selection_rank: int = Field(gt=0)
    catalog_sha256: str
    final_labels: list[str] = Field(min_length=1)
    primary_filename: str = Field(min_length=1)
    source_sha256: str
    source_size_bytes: int = Field(gt=0)
    document_page_count: int = Field(gt=0)
    selection_relative_path: str = Field(min_length=1)
    selected_local_alias: CatalogLocalAlias
    catalog_records: list[CatalogDocumentRecord] = Field(min_length=1)

    @field_validator("catalog_sha256", "source_sha256")
    @classmethod
    def hashes_are_sha256(cls, value: str) -> str:
        if not _is_sha256(value):
            raise ValueError("selection hashes must be lowercase SHA-256 values")
        return value

    @model_validator(mode="after")
    def provenance_is_consistent(self) -> CatalogSelectionDocumentRecord:
        if self.final_labels != sorted(set(self.final_labels)):
            raise ValueError("final_labels must be unique and sorted")
        ordered_records = sorted(
            self.catalog_records,
            key=lambda row: (
                row.final.final_classification_label if row.final is not None else "",
                row.document_filename,
                row.catalog_document_id,
            ),
        )
        if self.catalog_records != ordered_records:
            raise ValueError("catalog_records must be canonically sorted")
        observed_labels: set[str] = set()
        selected_alias_found = False
        for record in self.catalog_records:
            if record.status != "retained_final" or record.final is None or record.initial is None:
                raise ValueError("selection records must contain retained final evidence")
            if record.initial.dummy.get("is_dummy") is not False:
                raise ValueError("selection records must carry explicit non-dummy evidence")
            observed_labels.add(record.final.final_classification_label)
            ready = [
                alias
                for alias in record.local_snapshot_aliases
                if alias.snapshot_record.extraction_status == "ready"
            ]
            if not ready:
                raise ValueError("selection record has no extraction-ready local alias")
            for alias in ready:
                source = alias.snapshot_record
                if (
                    source.source_sha256 != self.source_sha256
                    or source.source_size_bytes != self.source_size_bytes
                    or source.actual_page_count != self.document_page_count
                ):
                    raise ValueError("catalog aliases disagree with selected content identity")
                selected_alias_found |= alias == self.selected_local_alias
            if record.final.document_page_count != self.document_page_count:
                raise ValueError("classifier and local PDF page counts disagree")
        if sorted(observed_labels) != self.final_labels:
            raise ValueError("final_labels differ from the classifier alias set")
        if not selected_alias_found:
            raise ValueError("selected_local_alias is absent from catalog_records")

        match = _DOCUMENT_FILENAME.fullmatch(self.primary_filename)
        if match is None:
            raise ValueError("primary_filename must be YYYY-MM-DD_<UUID>.pdf")
        try:
            date.fromisoformat(match.group("document_date"))
        except ValueError as error:
            raise ValueError("primary_filename contains an invalid date") from error
        label_directory = (
            self.final_labels[0]
            if len(self.final_labels) == 1
            else "_".join(self.final_labels) + "_conflict"
        )
        expected_path = PurePosixPath(
            "files",
            label_directory,
            match.group("document_date")[:4],
            self.primary_filename,
        ).as_posix()
        if self.selection_relative_path != expected_path:
            raise ValueError("selection_relative_path does not match its identity")
        return self


class CatalogSelectionStatistics(_FrozenRecord):
    schema_version: Literal[2]
    catalog_status_counts: dict[str, int]
    catalog_retained_target_documents: int = Field(ge=0)
    non_dummy_target_documents: int = Field(ge=0)
    locally_ready_target_documents: int = Field(ge=0)
    locally_unavailable_target_documents: int = Field(ge=0)
    content_groups_before_exclusions: int = Field(ge=0)
    duplicate_catalog_rows_collapsed: int = Field(ge=0)
    excluded_pilot_manifest_documents: int = Field(ge=0)
    excluded_pilot_content_groups: int = Field(ge=0)
    excluded_page_limit_content_groups: int = Field(ge=0)
    excluded_page_limit_pages: int = Field(ge=0)
    excluded_page_limit_bytes: int = Field(ge=0)
    selected_documents: int = Field(gt=0)
    selected_pages: int = Field(gt=0)
    selected_bytes: int = Field(gt=0)
    classification_conflict_documents: int = Field(ge=0)
    by_classification_set: dict[str, int]
    by_page_count: dict[str, int]
    by_primary_lineage: dict[str, int]
    excluded_by_page_count: dict[str, int]

    @model_validator(mode="after")
    def breakdowns_are_consistent(self) -> CatalogSelectionStatistics:
        if self.non_dummy_target_documents != (
            self.locally_ready_target_documents + self.locally_unavailable_target_documents
        ):
            raise ValueError("local availability does not partition non-dummy target rows")
        if self.duplicate_catalog_rows_collapsed != (
            self.locally_ready_target_documents - self.content_groups_before_exclusions
        ):
            raise ValueError("duplicate row count does not match content grouping")
        if self.selected_documents != (
            self.content_groups_before_exclusions
            - self.excluded_pilot_content_groups
            - self.excluded_page_limit_content_groups
        ):
            raise ValueError("pilot and page-limit exclusions do not partition content groups")
        if sum(self.excluded_by_page_count.values()) != self.excluded_page_limit_content_groups:
            raise ValueError("page-limit exclusion breakdown does not match its group count")
        for breakdown in (
            self.by_classification_set,
            self.by_page_count,
            self.by_primary_lineage,
        ):
            if sum(breakdown.values()) != self.selected_documents:
                raise ValueError("selection breakdown does not sum to selected_documents")
        if self.classification_conflict_documents != sum(
            count for key, count in self.by_classification_set.items() if "+" in key
        ):
            raise ValueError("classification conflict count differs from breakdown")
        return self


class CatalogSelectionCommit(_FrozenRecord):
    schema_version: Literal[2]
    configuration_sha256: str
    catalog_config_path: str = Field(min_length=1)
    catalog_config_sha256: str
    catalog_sha256: str
    final_labels: list[str] = Field(min_length=1)
    dummy_is_dummy: Literal[False]
    deduplicate_by_content_sha256: Literal[True]
    max_pages_per_document: int = Field(gt=0)
    manifest_path: Literal["manifest.jsonl"]
    manifest_sha256: str
    statistics: CatalogSelectionStatistics

    @field_validator(
        "configuration_sha256",
        "catalog_config_sha256",
        "catalog_sha256",
        "manifest_sha256",
    )
    @classmethod
    def hashes_are_sha256(cls, value: str) -> str:
        if not _is_sha256(value):
            raise ValueError("commit hashes must be lowercase SHA-256 values")
        return value


class CatalogSelectionPlanSummary(_FrozenRecord):
    manifest_sha256: str
    statistics: CatalogSelectionStatistics

    @field_validator("manifest_sha256")
    @classmethod
    def hash_is_sha256(cls, value: str) -> str:
        if not _is_sha256(value):
            raise ValueError("manifest_sha256 must be a lowercase SHA-256")
        return value


@dataclass(frozen=True, slots=True)
class CatalogSelectionResult:
    root: Path
    commit: CatalogSelectionCommit
    created: bool


@dataclass(frozen=True, slots=True)
class _VerifiedCatalog:
    config_path: Path
    config_sha256: str
    result: CatalogResult
    records: tuple[CatalogDocumentRecord, ...]


@dataclass(frozen=True, slots=True)
class _ContentGroup:
    source_sha256: str
    source_size_bytes: int
    document_page_count: int
    records: tuple[CatalogDocumentRecord, ...]
    selected_alias: CatalogLocalAlias
    primary_record: CatalogDocumentRecord


@dataclass(frozen=True, slots=True)
class _SelectionPlan:
    catalog: _VerifiedCatalog
    records: tuple[CatalogSelectionDocumentRecord, ...]
    manifest_payload: bytes
    manifest_sha256: str
    statistics: CatalogSelectionStatistics


class CatalogSelectionError(RuntimeError):
    """Base class for explicit full-surface selection failures."""


class CatalogSelectionIntegrityError(CatalogSelectionError):
    """A source or published artifact differs from its immutable contract."""


def _canonical_path(path: Path, *, description: str) -> Path:
    absolute = Path(os.path.abspath(path))
    try:
        resolved = path.resolve(strict=True)
    except OSError as error:
        raise CatalogSelectionIntegrityError(f"{description} does not exist: {path}") from error
    if path != absolute or resolved != absolute:
        raise CatalogSelectionIntegrityError(
            f"{description} must be absolute and must not traverse symbolic links: {path}"
        )
    return absolute


def _load_catalog_records(payload: bytes, path: Path) -> tuple[CatalogDocumentRecord, ...]:
    if not payload or not payload.endswith(b"\n"):
        raise CatalogSelectionIntegrityError(f"catalog is not newline-terminated: {path}")
    records: list[CatalogDocumentRecord] = []
    for line_number, line in enumerate(payload.splitlines(), start=1):
        try:
            records.append(CatalogDocumentRecord.model_validate_json(line, strict=True))
        except Exception as error:
            raise CatalogSelectionIntegrityError(
                f"invalid catalog row at {path}:{line_number}"
            ) from error
    return tuple(records)


def _verified_catalog(
    config: CatalogExtractionSelectionConfig,
    *,
    progress: SelectionProgress | None,
) -> _VerifiedCatalog:
    config_path = _canonical_path(Path(config.catalog_config), description="catalog configuration")
    config_payload = read_regular_file_bytes(config_path)
    catalog_config = load_catalog_config(config_path)
    if read_regular_file_bytes(config_path) != config_payload:
        raise CatalogSelectionIntegrityError("catalog configuration changed while loading")
    if catalog_config.expected_catalog_sha256 != config.expected_catalog_sha256:
        raise CatalogSelectionError("selection and catalog configurations bind different hashes")
    result = verify_catalog(catalog_config, progress=progress)
    if result.commit.catalog_sha256 != config.expected_catalog_sha256:
        raise CatalogSelectionIntegrityError("verified catalog differs from expected hash")
    destination = Path(os.path.abspath(config.destination_root))
    protected_roots = [
        result.root,
        *(Path(row.snapshot_root) for row in result.commit.local_snapshots),
    ]
    for protected in protected_roots:
        if (
            destination == protected
            or destination in protected.parents
            or protected in destination.parents
        ):
            raise CatalogSelectionError("selection destination overlaps an input artifact root")
    catalog_path = result.root / result.commit.catalog_path
    payload = read_regular_file_bytes(catalog_path)
    if sha256_bytes(payload) != result.commit.catalog_sha256:
        raise CatalogSelectionIntegrityError("verified catalog changed before selection")
    records = _load_catalog_records(payload, catalog_path)
    if len(records) != result.commit.document_count:
        raise CatalogSelectionIntegrityError("catalog row count differs from its commit")
    return _VerifiedCatalog(
        config_path=config_path,
        config_sha256=sha256_bytes(config_payload),
        result=result,
        records=records,
    )


def _load_excluded_pilot(
    root: Path,
    *,
    expected_manifest_sha256: str,
    expected_catalog_sha256: str,
) -> tuple[PilotDocumentRecord, ...]:
    safe_root = _safe_snapshot_directory(root, create=False)
    commit_path = safe_root / "pilot.json"
    try:
        commit_payload = read_regular_file_bytes(commit_path)
        commit = PilotCommit.model_validate_json(commit_payload, strict=True)
    except Exception as error:
        raise CatalogSelectionIntegrityError(f"invalid excluded pilot: {root}") from error
    if commit_payload != _canonical_record_payload(commit):
        raise CatalogSelectionIntegrityError(f"excluded pilot commit is not canonical: {root}")
    if commit.manifest_sha256 != expected_manifest_sha256:
        raise CatalogSelectionIntegrityError(f"excluded pilot manifest hash differs: {root}")
    if commit.catalog_sha256 != expected_catalog_sha256:
        raise CatalogSelectionError(f"excluded pilot binds another catalog: {root}")
    manifest_path = safe_root / commit.manifest_path
    manifest = _read_digest_bound_artifact(manifest_path, commit.manifest_sha256)
    rows: list[PilotDocumentRecord] = []
    for line_number, line in enumerate(manifest.splitlines(), start=1):
        try:
            rows.append(PilotDocumentRecord.model_validate_json(line, strict=True))
        except Exception as error:
            raise CatalogSelectionIntegrityError(
                f"invalid excluded pilot row at {manifest_path}:{line_number}"
            ) from error
    if len(rows) != commit.statistics.selected_documents:
        raise CatalogSelectionIntegrityError("excluded pilot row count differs from commit")
    return tuple(rows)


def _excluded_content(
    config: CatalogExtractionSelectionConfig,
) -> tuple[frozenset[str], frozenset[str], int]:
    source_sha256s: set[str] = set()
    filenames: set[str] = set()
    manifest_documents = 0
    for excluded in config.excluded_pilots:
        rows = _load_excluded_pilot(
            Path(excluded.root),
            expected_manifest_sha256=excluded.expected_manifest_sha256,
            expected_catalog_sha256=config.expected_catalog_sha256,
        )
        current_hashes = {row.source_sha256 for row in rows}
        current_filenames = {row.document_filename for row in rows}
        if len(current_hashes) != len(rows) or len(current_filenames) != len(rows):
            raise CatalogSelectionIntegrityError("excluded pilot is not content-unique")
        if source_sha256s & current_hashes or filenames & current_filenames:
            raise CatalogSelectionError("excluded pilot manifests overlap")
        source_sha256s.update(current_hashes)
        filenames.update(current_filenames)
        manifest_documents += len(rows)
    return frozenset(source_sha256s), frozenset(filenames), manifest_documents


def _ready_aliases(record: CatalogDocumentRecord) -> tuple[CatalogLocalAlias, ...]:
    return tuple(
        alias
        for alias in record.local_snapshot_aliases
        if alias.snapshot_record.extraction_status == "ready"
    )


def _content_group(records: list[CatalogDocumentRecord]) -> _ContentGroup:
    ordered_records = tuple(
        sorted(
            records,
            key=lambda row: (
                row.final.final_classification_label if row.final is not None else "",
                row.document_filename,
                row.catalog_document_id,
            ),
        )
    )
    candidates: list[tuple[CatalogDocumentRecord, CatalogLocalAlias]] = []
    identities: set[tuple[str, int, int]] = set()
    for record in ordered_records:
        if record.final is None:
            raise CatalogSelectionError("content group lacks final classifier evidence")
        for alias in _ready_aliases(record):
            source = alias.snapshot_record
            identities.add(
                (source.source_sha256, source.source_size_bytes, source.actual_page_count)
            )
            if source.actual_page_count != record.final.document_page_count:
                raise CatalogSelectionError(
                    f"catalog/local page count conflict: {record.document_filename}"
                )
            candidates.append((record, alias))
    if len(identities) != 1:
        raise CatalogSelectionError("byte-identical catalog group has conflicting local metadata")
    source_sha256, size_bytes, page_count = next(iter(identities))
    candidates.sort(
        key=lambda item: (
            item[0].final.final_classification_label if item[0].final is not None else "",
            item[0].document_filename,
            item[1].snapshot_record.source_key,
            item[1].snapshot_root,
        )
    )
    primary_record, selected_alias = candidates[0]
    return _ContentGroup(
        source_sha256=source_sha256,
        source_size_bytes=size_bytes,
        document_page_count=page_count,
        records=ordered_records,
        selected_alias=selected_alias,
        primary_record=primary_record,
    )


def _build_plan(
    config: CatalogExtractionSelectionConfig,
    *,
    progress: SelectionProgress | None,
) -> _SelectionPlan:
    catalog = _verified_catalog(config, progress=progress)
    excluded_hashes, excluded_filenames, excluded_manifest_documents = _excluded_content(config)
    target_labels = set(config.final_labels)
    status_counts = Counter(record.status for record in catalog.records)
    retained_target = tuple(
        record
        for record in catalog.records
        if record.status == "retained_final"
        and record.final is not None
        and record.final.final_classification_label in target_labels
    )
    non_dummy = tuple(
        record
        for record in retained_target
        if record.initial is not None
        and record.initial.dummy.get("is_dummy") is config.dummy_is_dummy
    )
    local = tuple(record for record in non_dummy if _ready_aliases(record))
    by_hash: dict[str, list[CatalogDocumentRecord]] = defaultdict(list)
    for record in local:
        identities = {
            alias.snapshot_record.source_sha256 for alias in _ready_aliases(record)
        }
        if len(identities) != 1:
            raise CatalogSelectionError(
                f"local aliases disagree on content: {record.document_filename}"
            )
        by_hash[next(iter(identities))].append(record)
    groups = tuple(_content_group(rows) for _, rows in sorted(by_hash.items()))
    post_pilot_groups = tuple(
        group
        for group in groups
        if group.source_sha256 not in excluded_hashes
        and not any(record.document_filename in excluded_filenames for record in group.records)
    )
    excluded_group_count = len(groups) - len(post_pilot_groups)
    if excluded_group_count != len(excluded_hashes):
        raise CatalogSelectionError(
            "excluded pilot content is not a one-to-one subset of the eligible surface"
        )
    page_limit_exclusions = tuple(
        group
        for group in post_pilot_groups
        if group.document_page_count > config.max_pages_per_document
    )
    selected_groups = tuple(
        group
        for group in post_pilot_groups
        if group.document_page_count <= config.max_pages_per_document
    )

    ordered_groups = sorted(
        selected_groups,
        key=lambda group: (
            tuple(
                sorted(
                    {
                        record.final.final_classification_label
                        for record in group.records
                        if record.final is not None
                    }
                )
            ),
            group.primary_record.document_filename,
            group.source_sha256,
        ),
    )
    records: list[CatalogSelectionDocumentRecord] = []
    for rank, group in enumerate(ordered_groups, start=1):
        labels = sorted(
            {
                record.final.final_classification_label
                for record in group.records
                if record.final is not None
            }
        )
        filename = group.primary_record.document_filename
        match = _DOCUMENT_FILENAME.fullmatch(filename)
        if match is None:
            raise CatalogSelectionError(f"source filename is not a stable document ID: {filename}")
        label_directory = labels[0] if len(labels) == 1 else "_".join(labels) + "_conflict"
        records.append(
            CatalogSelectionDocumentRecord(
                schema_version=1,
                selection_rank=rank,
                catalog_sha256=config.expected_catalog_sha256,
                final_labels=labels,
                primary_filename=filename,
                source_sha256=group.source_sha256,
                source_size_bytes=group.source_size_bytes,
                document_page_count=group.document_page_count,
                selection_relative_path=PurePosixPath(
                    "files", label_directory, match.group("document_date")[:4], filename
                ).as_posix(),
                selected_local_alias=group.selected_alias,
                catalog_records=list(group.records),
            )
        )
    selected = tuple(records)
    by_classification_set = Counter("+".join(row.final_labels) for row in selected)
    statistics = CatalogSelectionStatistics(
        schema_version=2,
        catalog_status_counts=dict(sorted(status_counts.items())),
        catalog_retained_target_documents=len(retained_target),
        non_dummy_target_documents=len(non_dummy),
        locally_ready_target_documents=len(local),
        locally_unavailable_target_documents=len(non_dummy) - len(local),
        content_groups_before_exclusions=len(groups),
        duplicate_catalog_rows_collapsed=len(local) - len(groups),
        excluded_pilot_manifest_documents=excluded_manifest_documents,
        excluded_pilot_content_groups=excluded_group_count,
        excluded_page_limit_content_groups=len(page_limit_exclusions),
        excluded_page_limit_pages=sum(
            group.document_page_count for group in page_limit_exclusions
        ),
        excluded_page_limit_bytes=sum(group.source_size_bytes for group in page_limit_exclusions),
        selected_documents=len(selected),
        selected_pages=sum(row.document_page_count for row in selected),
        selected_bytes=sum(row.source_size_bytes for row in selected),
        classification_conflict_documents=sum(
            count for key, count in by_classification_set.items() if "+" in key
        ),
        by_classification_set=dict(sorted(by_classification_set.items())),
        by_page_count=dict(
            sorted(
                Counter(str(row.document_page_count) for row in selected).items(),
                key=lambda item: int(item[0]),
            )
        ),
        by_primary_lineage=dict(
            sorted(Counter(row.catalog_records[0].lineage for row in selected).items())
        ),
        excluded_by_page_count=dict(
            sorted(
                Counter(
                    str(group.document_page_count) for group in page_limit_exclusions
                ).items(),
                key=lambda item: int(item[0]),
            )
        ),
    )
    if statistics.selected_documents != config.expected_documents:
        raise CatalogSelectionError("selected document count differs from expected_documents")
    if statistics.selected_pages != config.expected_pages:
        raise CatalogSelectionError("selected page count differs from expected_pages")
    payload = _canonical_jsonl(selected)
    return _SelectionPlan(
        catalog=catalog,
        records=selected,
        manifest_payload=payload,
        manifest_sha256=sha256_bytes(payload),
        statistics=statistics,
    )


def _source_path(record: CatalogSelectionDocumentRecord) -> Path:
    alias = record.selected_local_alias
    return Path(alias.snapshot_root) / Path(
        *PurePosixPath(alias.snapshot_record.snapshot_relative_path).parts
    )


def _link_groups(
    root: Path,
    plan: _SelectionPlan,
) -> tuple[tuple[Path, Path, tuple[tuple[str, str], ...]], ...]:
    grouped: dict[tuple[Path, Path], list[tuple[str, str]]] = defaultdict(list)
    for record in plan.records:
        source = _source_path(record)
        destination = root / Path(*PurePosixPath(record.selection_relative_path).parts)
        grouped[(source.parent, destination.parent)].append((source.name, destination.name))
    return tuple(
        (source, destination, tuple(sorted(names)))
        for (source, destination), names in sorted(
            grouped.items(), key=lambda item: (str(item[0][0]), str(item[0][1]))
        )
    )


def _validate_hard_link(
    *,
    source_name: str,
    destination_name: str,
    source_fd: int,
    destination_fd: int,
    destination_parent: Path,
) -> None:
    try:
        source_stat = os.stat(source_name, dir_fd=source_fd, follow_symlinks=False)
        destination_stat = os.stat(
            destination_name, dir_fd=destination_fd, follow_symlinks=False
        )
    except OSError as error:
        raise CatalogSelectionIntegrityError(
            f"cannot validate selection hard link: {destination_parent / destination_name}"
        ) from error
    if (
        not stat.S_ISREG(source_stat.st_mode)
        or not stat.S_ISREG(destination_stat.st_mode)
        or (source_stat.st_dev, source_stat.st_ino)
        != (destination_stat.st_dev, destination_stat.st_ino)
    ):
        raise CatalogSelectionIntegrityError(
            f"selection PDF is not its source hard link: {destination_parent / destination_name}"
        )


def _publish_links(root: Path, plan: _SelectionPlan) -> None:
    for source_parent, destination_parent, names in _link_groups(root, plan):
        source = _safe_snapshot_directory(source_parent, create=False)
        destination = _safe_snapshot_directory(destination_parent, create=True)
        source_fd = os.open(
            source, os.O_RDONLY | os.O_CLOEXEC | os.O_NOFOLLOW | os.O_DIRECTORY
        )
        destination_fd = os.open(
            destination, os.O_RDONLY | os.O_CLOEXEC | os.O_NOFOLLOW | os.O_DIRECTORY
        )
        try:
            for source_name, destination_name in names:
                try:
                    os.link(
                        source_name,
                        destination_name,
                        src_dir_fd=source_fd,
                        dst_dir_fd=destination_fd,
                        follow_symlinks=False,
                    )
                except FileExistsError:
                    pass
                except OSError as error:
                    raise CatalogSelectionIntegrityError(
                        f"failed to hard-link selection PDF: {destination / destination_name}"
                    ) from error
                _validate_hard_link(
                    source_name=source_name,
                    destination_name=destination_name,
                    source_fd=source_fd,
                    destination_fd=destination_fd,
                    destination_parent=destination,
                )
            os.fsync(destination_fd)
        finally:
            os.close(destination_fd)
            os.close(source_fd)


def _verify_pdfs(
    config: CatalogExtractionSelectionConfig,
    root: Path,
    plan: _SelectionPlan,
    *,
    progress: SelectionProgress | None,
) -> None:
    expected = {
        root / Path(*PurePosixPath(record.selection_relative_path).parts)
        for record in plan.records
    }
    actual = set(_safe_walk_files(root / "files"))
    if actual != expected:
        raise CatalogSelectionIntegrityError("selection PDF tree differs from manifest")
    completed = 0
    with ThreadPoolExecutor(
        max_workers=config.verification_workers,
        thread_name_prefix="catalog-selection-verify",
    ) as executor:
        futures = {
            executor.submit(
                _hash_local_pdf,
                root / Path(*PurePosixPath(record.selection_relative_path).parts),
                max_pdf_bytes=config.max_pdf_bytes,
            ): record
            for record in plan.records
        }
        for future in as_completed(futures):
            record = futures[future]
            digest, size_bytes = future.result()
            if digest != record.source_sha256 or size_bytes != record.source_size_bytes:
                raise CatalogSelectionIntegrityError(
                    f"selection PDF differs from manifest: {record.selection_relative_path}"
                )
            completed += 1
            if progress is not None and (completed % 50 == 0 or completed == len(plan.records)):
                progress("verify_selection_pdf", completed, len(plan.records))
    for source_parent, destination_parent, names in _link_groups(root, plan):
        source_fd = os.open(
            _safe_snapshot_directory(source_parent, create=False),
            os.O_RDONLY | os.O_CLOEXEC | os.O_NOFOLLOW | os.O_DIRECTORY,
        )
        destination = _safe_snapshot_directory(destination_parent, create=False)
        destination_fd = os.open(
            destination, os.O_RDONLY | os.O_CLOEXEC | os.O_NOFOLLOW | os.O_DIRECTORY
        )
        try:
            for source_name, destination_name in names:
                _validate_hard_link(
                    source_name=source_name,
                    destination_name=destination_name,
                    source_fd=source_fd,
                    destination_fd=destination_fd,
                    destination_parent=destination,
                )
        finally:
            os.close(destination_fd)
            os.close(source_fd)


def _configuration_sha256(config: CatalogExtractionSelectionConfig) -> str:
    return canonical_json_sha256(config.model_dump(mode="json"))


def _build_commit(
    config: CatalogExtractionSelectionConfig,
    plan: _SelectionPlan,
) -> CatalogSelectionCommit:
    return CatalogSelectionCommit(
        schema_version=2,
        configuration_sha256=_configuration_sha256(config),
        catalog_config_path=str(plan.catalog.config_path),
        catalog_config_sha256=plan.catalog.config_sha256,
        catalog_sha256=plan.catalog.result.commit.catalog_sha256,
        final_labels=config.final_labels,
        dummy_is_dummy=config.dummy_is_dummy,
        deduplicate_by_content_sha256=config.deduplicate_by_content_sha256,
        max_pages_per_document=config.max_pages_per_document,
        manifest_path="manifest.jsonl",
        manifest_sha256=plan.manifest_sha256,
        statistics=plan.statistics,
    )


def _expected_tree(root: Path, plan: _SelectionPlan) -> set[Path]:
    return {
        *(root / Path(*PurePosixPath(row.selection_relative_path).parts) for row in plan.records),
        root / "manifest.jsonl",
        root / "manifest.jsonl.sha256",
        root / "selection.json",
    }


def plan_catalog_selection(
    config: CatalogExtractionSelectionConfig,
    *,
    progress: SelectionProgress | None = None,
) -> CatalogSelectionPlanSummary:
    plan = _build_plan(config, progress=progress)
    return CatalogSelectionPlanSummary(
        manifest_sha256=plan.manifest_sha256,
        statistics=plan.statistics,
    )


def materialize_catalog_selection(
    config: CatalogExtractionSelectionConfig,
    *,
    progress: SelectionProgress | None = None,
) -> CatalogSelectionResult:
    root = _safe_snapshot_directory(Path(config.destination_root), create=True)
    commit_path = root / "selection.json"
    if commit_path.exists() or commit_path.is_symlink():
        return verify_catalog_selection(config, progress=progress)
    plan = _build_plan(config, progress=progress)
    if plan.manifest_sha256 != config.expected_manifest_sha256:
        raise CatalogSelectionError(
            f"selection manifest SHA-256 changed ({plan.manifest_sha256} != "
            f"{config.expected_manifest_sha256})"
        )
    _publish_links(root, plan)
    _verify_pdfs(config, root, plan, progress=progress)
    published_digest = _publish_digest_bound_artifact(
        root / "manifest.jsonl", plan.manifest_payload
    )
    if published_digest != plan.manifest_sha256:
        raise CatalogSelectionIntegrityError("published selection manifest digest changed")
    commit = _build_commit(config, plan)
    try:
        atomic_publish_json(commit_path, commit.model_dump(mode="json"))
    except AtomicConflictError as error:
        raise CatalogSelectionIntegrityError("selection commit publication conflict") from error
    if set(_safe_walk_files(root)) != _expected_tree(root, plan):
        raise CatalogSelectionIntegrityError("selection tree differs after commit publication")
    return CatalogSelectionResult(root=root, commit=commit, created=True)


def verify_catalog_selection(
    config: CatalogExtractionSelectionConfig,
    *,
    progress: SelectionProgress | None = None,
) -> CatalogSelectionResult:
    root = _safe_snapshot_directory(Path(config.destination_root), create=False)
    commit_path = root / "selection.json"
    try:
        commit_payload = read_regular_file_bytes(commit_path)
        commit = CatalogSelectionCommit.model_validate_json(commit_payload, strict=True)
    except Exception as error:
        raise CatalogSelectionIntegrityError("invalid or missing selection commit") from error
    if commit_payload != _canonical_record_payload(commit):
        raise CatalogSelectionIntegrityError("selection commit is not canonical")
    if commit.configuration_sha256 != _configuration_sha256(config):
        raise CatalogSelectionIntegrityError("selection config differs from committed config")
    plan = _build_plan(config, progress=progress)
    if commit != _build_commit(config, plan):
        raise CatalogSelectionIntegrityError("selection commit differs from recomputed plan")
    manifest = _read_digest_bound_artifact(
        root / commit.manifest_path, commit.manifest_sha256
    )
    if manifest != plan.manifest_payload:
        raise CatalogSelectionIntegrityError("selection manifest differs from recomputed plan")
    _verify_pdfs(config, root, plan, progress=progress)
    if set(_safe_walk_files(root)) != _expected_tree(root, plan):
        raise CatalogSelectionIntegrityError("selection tree differs from commit")
    return CatalogSelectionResult(root=root, commit=commit, created=False)
