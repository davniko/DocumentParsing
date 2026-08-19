"""Deterministic, quality-filtered pilot corpora from the classification catalog.

The pilot is a separate immutable filesystem view, not another PDF copy.  Each
selected file is a hard link to one verified snapshot original, while the
manifest retains the complete catalog row and the exact chosen local alias.
The commit marker is published last so an apparently complete pilot is always
fully auditable and reproducible.
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
from typing import Literal, cast

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from document_ocr.atomic import AtomicConflictError, atomic_publish_json, read_regular_file_bytes
from document_ocr.classification_catalog import (
    CatalogDocumentRecord,
    CatalogLocalAlias,
    CatalogResult,
    verify_catalog,
)
from document_ocr.config import (
    CatalogPilotConfig,
    PilotQualityConfig,
    PilotStratumConfig,
    load_catalog_config,
)
from document_ocr.hashing import (
    canonical_json_bytes,
    canonical_json_sha256,
    identity_sha256,
    sha256_bytes,
)
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

PilotProgress = Callable[[str, int, int], None]


class _FrozenRecord(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, strict=True, allow_inf_nan=False)


def _is_sha256(value: str) -> bool:
    return len(value) == 64 and all(character in "0123456789abcdef" for character in value)


class PilotDocumentRecord(_FrozenRecord):
    """One selected PDF with complete catalog and local-snapshot provenance."""

    schema_version: Literal[1]
    selection_rank: int = Field(gt=0)
    stratum_rank: int = Field(gt=0)
    selection_score_sha256: str
    catalog_sha256: str
    final_label: str = Field(min_length=1)
    lineage: str = Field(min_length=1)
    triage_category: Literal["photo", "rendered", "scanned"]
    document_filename: str = Field(min_length=1)
    source_sha256: str
    source_size_bytes: int = Field(gt=0)
    document_page_count: int = Field(gt=0)
    pilot_relative_path: str = Field(min_length=1)
    selected_local_alias: CatalogLocalAlias
    catalog_record: CatalogDocumentRecord

    @field_validator("selection_score_sha256", "catalog_sha256", "source_sha256")
    @classmethod
    def hashes_are_sha256(cls, value: str) -> str:
        if not _is_sha256(value):
            raise ValueError("pilot hashes must be lowercase SHA-256 values")
        return value

    @model_validator(mode="after")
    def provenance_and_selection_are_bound(self) -> PilotDocumentRecord:
        match = _DOCUMENT_FILENAME.fullmatch(self.document_filename)
        if match is None:
            raise ValueError("pilot filename must be YYYY-MM-DD_<UUID>.pdf")
        try:
            date.fromisoformat(match.group("document_date"))
        except ValueError as error:
            raise ValueError("pilot filename contains an invalid calendar date") from error
        expected_path = PurePosixPath(
            "files",
            self.final_label,
            match.group("document_date")[:4],
            self.document_filename,
        ).as_posix()
        if self.pilot_relative_path != expected_path:
            raise ValueError("pilot_relative_path must equal files/<label>/<year>/<filename>")

        catalog = self.catalog_record
        if (
            catalog.status != "retained_final"
            or catalog.final is None
            or catalog.initial is None
            or catalog.final.final_classification_label != self.final_label
            or catalog.lineage != self.lineage
            or catalog.document_filename != self.document_filename
        ):
            raise ValueError("pilot identity differs from its retained final catalog row")
        details = catalog.initial.triage.get("details")
        if not isinstance(details, dict) or details.get("category") != self.triage_category:
            raise ValueError("pilot triage_category differs from catalog evidence")
        if self.selected_local_alias not in catalog.local_snapshot_aliases:
            raise ValueError("selected_local_alias is absent from catalog provenance")
        source = self.selected_local_alias.snapshot_record
        if source.extraction_status != "ready":
            raise ValueError("pilot source alias must be extraction-ready")
        if (
            source.source_sha256 != self.source_sha256
            or source.source_size_bytes != self.source_size_bytes
            or source.actual_page_count != self.document_page_count
            or catalog.final.document_page_count != self.document_page_count
        ):
            raise ValueError("pilot content identity differs from catalog/snapshot evidence")
        return self


class PilotStratumSummary(_FrozenRecord):
    lineage: str = Field(min_length=1)
    triage_category: Literal["photo", "rendered", "scanned"]
    quota_documents: int = Field(gt=0)
    eligible_documents: int = Field(ge=0)
    eligible_unique_content_sha256: int = Field(ge=0)
    selected_documents: int = Field(gt=0)
    selected_pages: int = Field(gt=0)
    selected_bytes: int = Field(gt=0)

    @model_validator(mode="after")
    def quota_is_filled(self) -> PilotStratumSummary:
        if self.selected_documents != self.quota_documents:
            raise ValueError("pilot stratum selected_documents must fill its quota")
        return self


class PilotStatistics(_FrozenRecord):
    schema_version: Literal[1]
    catalog_final_label_local_documents: int = Field(ge=0)
    quality_eligible_documents: int = Field(ge=0)
    bounded_eligible_documents: int = Field(ge=0)
    bounded_unique_content_sha256: int = Field(ge=0)
    selected_documents: int = Field(gt=0)
    selected_pages: int = Field(gt=0)
    selected_bytes: int = Field(gt=0)
    selected_unique_content_sha256: int = Field(gt=0)
    by_lineage: dict[str, int]
    by_triage_category: dict[str, int]
    by_page_count: dict[str, int]
    strata: list[PilotStratumSummary] = Field(min_length=1)

    @model_validator(mode="after")
    def selected_breakdowns_match(self) -> PilotStatistics:
        if self.selected_unique_content_sha256 != self.selected_documents:
            raise ValueError("pilot selection must contain no duplicate PDF content")
        if sum(self.by_lineage.values()) != self.selected_documents:
            raise ValueError("pilot lineage counts do not sum to selected_documents")
        if sum(self.by_triage_category.values()) != self.selected_documents:
            raise ValueError("pilot category counts do not sum to selected_documents")
        if sum(self.by_page_count.values()) != self.selected_documents:
            raise ValueError("pilot page-count buckets do not sum to selected_documents")
        if sum(item.selected_documents for item in self.strata) != self.selected_documents:
            raise ValueError("pilot stratum counts do not sum to selected_documents")
        return self


class PilotCommit(_FrozenRecord):
    """Manifest-last proof for one immutable quality-filtered pilot."""

    schema_version: Literal[1, 2]
    final_label: str = Field(min_length=1)
    configuration_sha256: str
    catalog_config_path: str = Field(min_length=1)
    catalog_config_sha256: str
    catalog_sha256: str
    selection_namespace: str = Field(min_length=1)
    quality: PilotQualityConfig
    deduplicate_by_content_sha256: Literal[True]
    max_pages_per_document: int = Field(gt=0)
    strata: list[PilotStratumConfig] = Field(min_length=1)
    manifest_path: Literal["manifest.jsonl"]
    manifest_sha256: str
    statistics: PilotStatistics

    @field_validator(
        "configuration_sha256",
        "catalog_config_sha256",
        "catalog_sha256",
        "manifest_sha256",
    )
    @classmethod
    def hashes_are_sha256(cls, value: str) -> str:
        if not _is_sha256(value):
            raise ValueError("pilot commit hashes must be lowercase SHA-256 values")
        return value


class PilotPlanSummary(_FrozenRecord):
    manifest_sha256: str
    statistics: PilotStatistics

    @field_validator("manifest_sha256")
    @classmethod
    def hash_is_sha256(cls, value: str) -> str:
        if not _is_sha256(value):
            raise ValueError("pilot manifest hash must be a lowercase SHA-256")
        return value


@dataclass(frozen=True, slots=True)
class PilotResult:
    root: Path
    commit: PilotCommit
    created: bool


@dataclass(frozen=True, slots=True)
class _VerifiedCatalog:
    config_path: Path
    config_sha256: str
    result: CatalogResult
    records: tuple[CatalogDocumentRecord, ...]


@dataclass(frozen=True, slots=True)
class _Candidate:
    record: CatalogDocumentRecord
    alias: CatalogLocalAlias
    category: Literal["photo", "rendered", "scanned"]
    score: str


@dataclass(frozen=True, slots=True)
class _PilotPlan:
    catalog: _VerifiedCatalog
    records: tuple[PilotDocumentRecord, ...]
    manifest_payload: bytes
    manifest_sha256: str
    statistics: PilotStatistics


@dataclass(frozen=True, slots=True)
class _ExcludedPilotDocuments:
    document_filenames: frozenset[str]
    source_sha256s: frozenset[str]


class PilotError(RuntimeError):
    """Base class for explicit pilot selection/publication failures."""


class PilotSelectionError(PilotError):
    """The verified catalog cannot satisfy the configured pilot policy."""


class PilotIntegrityError(PilotError):
    """Pilot files or provenance differ from the immutable contract."""


def _canonical_jsonl(records: Iterable[BaseModel]) -> bytes:
    return b"".join(
        canonical_json_bytes(record.model_dump(mode="json")) + b"\n" for record in records
    )


def _canonical_path(path: Path, *, description: str) -> Path:
    absolute = Path(os.path.abspath(path))
    try:
        resolved = path.resolve(strict=True)
    except OSError as error:
        raise PilotIntegrityError(f"{description} does not exist: {path}") from error
    if path != absolute or resolved != absolute:
        raise PilotIntegrityError(
            f"{description} must be absolute and must not traverse symbolic links: {path}"
        )
    return absolute


def _load_catalog_records(payload: bytes, path: Path) -> tuple[CatalogDocumentRecord, ...]:
    if not payload or not payload.endswith(b"\n"):
        raise PilotIntegrityError(f"verified catalog is not newline-terminated JSONL: {path}")
    records: list[CatalogDocumentRecord] = []
    for line_number, line in enumerate(payload.splitlines(), start=1):
        try:
            records.append(CatalogDocumentRecord.model_validate_json(line, strict=True))
        except Exception as error:
            raise PilotIntegrityError(
                f"verified catalog row is invalid at {path}:{line_number}"
            ) from error
    return tuple(records)


def _load_pilot_records(payload: bytes, path: Path) -> tuple[PilotDocumentRecord, ...]:
    if not payload or not payload.endswith(b"\n"):
        raise PilotIntegrityError(
            f"excluded pilot manifest is not newline-terminated JSONL: {path}"
        )
    records: list[PilotDocumentRecord] = []
    for line_number, line in enumerate(payload.splitlines(), start=1):
        try:
            records.append(PilotDocumentRecord.model_validate_json(line, strict=True))
        except Exception as error:
            raise PilotIntegrityError(
                f"excluded pilot manifest row is invalid at {path}:{line_number}"
            ) from error
    return tuple(records)


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


def _load_canonical_pilot_commit(root: Path) -> tuple[Path, PilotCommit]:
    commit_path = root / "pilot.json"
    try:
        payload = read_regular_file_bytes(commit_path)
        commit = PilotCommit.model_validate_json(payload, strict=True)
    except Exception as error:
        raise PilotIntegrityError(f"invalid or missing pilot commit: {commit_path}") from error
    if payload != _canonical_record_payload(commit):
        raise PilotIntegrityError(f"pilot commit is not canonical: {commit_path}")
    return commit_path, commit


def _excluded_pilot_documents(config: CatalogPilotConfig) -> _ExcludedPilotDocuments:
    filenames: set[str] = set()
    source_sha256s: set[str] = set()
    destination = Path(os.path.abspath(config.destination_root))
    for excluded in config.excluded_pilots:
        root = _safe_snapshot_directory(Path(excluded.root), create=False)
        if destination == root or destination in root.parents or root in destination.parents:
            raise PilotSelectionError("pilot destination overlaps an excluded pilot root")
        _, commit = _load_canonical_pilot_commit(root)
        if commit.manifest_sha256 != excluded.expected_manifest_sha256:
            raise PilotIntegrityError(
                f"excluded pilot manifest hash differs from configuration: {root}"
            )
        if commit.catalog_sha256 != config.expected_catalog_sha256:
            raise PilotSelectionError("excluded pilot and new pilot bind different catalogs")
        if commit.final_label != config.final_label:
            raise PilotSelectionError("excluded pilot and new pilot use different final labels")
        manifest_path = root / commit.manifest_path
        manifest_payload = _read_digest_bound_artifact(
            manifest_path,
            commit.manifest_sha256,
        )
        records = _load_pilot_records(manifest_payload, manifest_path)
        if len(records) != commit.statistics.selected_documents:
            raise PilotIntegrityError("excluded pilot row count differs from its commit")
        current_filenames = {record.document_filename for record in records}
        current_source_sha256s = {record.source_sha256 for record in records}
        if len(current_filenames) != len(records) or len(current_source_sha256s) != len(records):
            raise PilotIntegrityError("excluded pilot does not contain unique documents")
        if filenames & current_filenames or source_sha256s & current_source_sha256s:
            raise PilotSelectionError("excluded pilots overlap each other")
        for record in records:
            if record.catalog_sha256 != config.expected_catalog_sha256:
                raise PilotIntegrityError("excluded pilot row binds a different catalog")
            if record.final_label != config.final_label:
                raise PilotIntegrityError("excluded pilot row uses a different final label")
        filenames.update(current_filenames)
        source_sha256s.update(current_source_sha256s)
    return _ExcludedPilotDocuments(
        document_filenames=frozenset(filenames),
        source_sha256s=frozenset(source_sha256s),
    )


def _verified_catalog(
    config: CatalogPilotConfig,
    *,
    progress: PilotProgress | None,
) -> _VerifiedCatalog:
    config_path = _canonical_path(Path(config.catalog_config), description="catalog configuration")
    config_payload = read_regular_file_bytes(config_path)
    catalog_config = load_catalog_config(config_path)
    if read_regular_file_bytes(config_path) != config_payload:
        raise PilotIntegrityError(f"catalog configuration changed while loading: {config_path}")
    if catalog_config.expected_catalog_sha256 != config.expected_catalog_sha256:
        raise PilotSelectionError("pilot and catalog configurations bind different catalog hashes")
    result = verify_catalog(catalog_config, progress=progress)
    if result.commit.catalog_sha256 != config.expected_catalog_sha256:
        raise PilotIntegrityError("verified catalog differs from pilot expected_catalog_sha256")
    destination = Path(os.path.abspath(config.destination_root))
    if (
        destination == result.root
        or destination in result.root.parents
        or result.root in destination.parents
    ):
        raise PilotSelectionError("pilot destination must not overlap the catalog root")
    for snapshot in result.commit.local_snapshots:
        snapshot_root = Path(snapshot.snapshot_root)
        if (
            destination == snapshot_root
            or destination in snapshot_root.parents
            or snapshot_root in destination.parents
        ):
            raise PilotSelectionError("pilot destination must not overlap a source snapshot")
    catalog_path = result.root / result.commit.catalog_path
    payload = read_regular_file_bytes(catalog_path)
    if sha256_bytes(payload) != result.commit.catalog_sha256:
        raise PilotIntegrityError("verified catalog bytes changed before pilot selection")
    records = _load_catalog_records(payload, catalog_path)
    if len(records) != result.commit.document_count:
        raise PilotIntegrityError("verified catalog row count differs from its commit")
    return _VerifiedCatalog(
        config_path=config_path,
        config_sha256=sha256_bytes(config_payload),
        result=result,
        records=records,
    )


def _ready_aliases(record: CatalogDocumentRecord) -> tuple[CatalogLocalAlias, ...]:
    aliases = tuple(
        alias
        for alias in record.local_snapshot_aliases
        if alias.snapshot_record.extraction_status == "ready"
    )
    return aliases


def _quality_matches(record: CatalogDocumentRecord, quality: PilotQualityConfig) -> bool:
    if record.initial is None:
        return False
    details = record.initial.triage.get("details")
    return (
        record.initial.dummy.get("is_dummy") is quality.dummy_is_dummy
        and record.initial.triage.get("requires_augmentation")
        is quality.triage_requires_augmentation
        and isinstance(details, dict)
        and details.get("readability") == quality.readability
        and details.get("augmentation_need") == quality.augmentation_need
    )


def _candidate(
    config: CatalogPilotConfig,
    record: CatalogDocumentRecord,
) -> _Candidate:
    aliases = _ready_aliases(record)
    if not aliases:
        raise PilotSelectionError("candidate has no extraction-ready local snapshot alias")
    identities = {
        (
            alias.snapshot_record.source_sha256,
            alias.snapshot_record.source_size_bytes,
            alias.snapshot_record.actual_page_count,
        )
        for alias in aliases
    }
    if len(identities) != 1:
        raise PilotSelectionError(
            f"candidate local aliases disagree on content identity: {record.document_filename}"
        )
    if record.final is None or record.initial is None:
        raise PilotSelectionError("candidate lacks required classifier evidence")
    source = aliases[0].snapshot_record
    if source.actual_page_count != record.final.document_page_count:
        raise PilotSelectionError(
            f"catalog and local page counts differ: {record.document_filename}"
        )
    details = record.initial.triage.get("details")
    category = details.get("category") if isinstance(details, dict) else None
    if category not in {"photo", "rendered", "scanned"}:
        raise PilotSelectionError(
            f"candidate has unsupported triage category {category!r}: {record.document_filename}"
        )
    score = identity_sha256(
        "catalog-pilot-selection-v1",
        config.selection_namespace,
        config.expected_catalog_sha256,
        record.catalog_document_id,
    )
    return _Candidate(
        record=record,
        alias=aliases[0],
        category=cast(Literal["photo", "rendered", "scanned"], category),
        score=score,
    )


def _record_is_local_final_label(record: CatalogDocumentRecord, final_label: str) -> bool:
    return (
        record.status == "retained_final"
        and record.final is not None
        and record.final.final_classification_label == final_label
        and bool(_ready_aliases(record))
    )


def _statistics(
    *,
    config: CatalogPilotConfig,
    local_surface_count: int,
    quality_count: int,
    bounded: tuple[_Candidate, ...],
    selected: tuple[PilotDocumentRecord, ...],
    candidates_by_stratum: dict[tuple[str, str], tuple[_Candidate, ...]],
) -> PilotStatistics:
    stratum_summaries: list[PilotStratumSummary] = []
    for stratum in config.strata:
        key = (stratum.lineage, stratum.triage_category)
        candidates = candidates_by_stratum.get(key, ())
        chosen = [item for item in selected if (item.lineage, item.triage_category) == key]
        stratum_summaries.append(
            PilotStratumSummary(
                lineage=stratum.lineage,
                triage_category=stratum.triage_category,
                quota_documents=stratum.documents,
                eligible_documents=len(candidates),
                eligible_unique_content_sha256=len(
                    {item.alias.snapshot_record.source_sha256 for item in candidates}
                ),
                selected_documents=len(chosen),
                selected_pages=sum(item.document_page_count for item in chosen),
                selected_bytes=sum(item.source_size_bytes for item in chosen),
            )
        )
    return PilotStatistics(
        schema_version=1,
        catalog_final_label_local_documents=local_surface_count,
        quality_eligible_documents=quality_count,
        bounded_eligible_documents=len(bounded),
        bounded_unique_content_sha256=len(
            {item.alias.snapshot_record.source_sha256 for item in bounded}
        ),
        selected_documents=len(selected),
        selected_pages=sum(item.document_page_count for item in selected),
        selected_bytes=sum(item.source_size_bytes for item in selected),
        selected_unique_content_sha256=len({item.source_sha256 for item in selected}),
        by_lineage=dict(sorted(Counter(item.lineage for item in selected).items())),
        by_triage_category=dict(sorted(Counter(item.triage_category for item in selected).items())),
        by_page_count=dict(
            sorted(
                Counter(str(item.document_page_count) for item in selected).items(),
                key=lambda item: int(item[0]),
            )
        ),
        strata=stratum_summaries,
    )


def _build_plan(
    config: CatalogPilotConfig,
    *,
    progress: PilotProgress | None,
) -> _PilotPlan:
    catalog = _verified_catalog(config, progress=progress)
    excluded = _excluded_pilot_documents(config)
    local_surface = tuple(
        record
        for record in catalog.records
        if _record_is_local_final_label(record, config.final_label)
    )
    quality_eligible = tuple(
        record for record in local_surface if _quality_matches(record, config.quality)
    )
    bounded = tuple(
        _candidate(config, record)
        for record in quality_eligible
        if record.final is not None
        and record.final.document_page_count <= config.max_pages_per_document
        and record.document_filename not in excluded.document_filenames
        and all(
            alias.snapshot_record.source_sha256 not in excluded.source_sha256s
            for alias in _ready_aliases(record)
        )
    )

    candidates_by_stratum: dict[tuple[str, str], tuple[_Candidate, ...]] = {}
    grouped: dict[tuple[str, str], list[_Candidate]] = defaultdict(list)
    for candidate in bounded:
        grouped[(candidate.record.lineage, candidate.category)].append(candidate)
    for key, candidates in grouped.items():
        candidates_by_stratum[key] = tuple(
            sorted(candidates, key=lambda item: (item.score, item.record.catalog_document_id))
        )

    used_content: set[str] = set()
    used_filenames: set[str] = set()
    selected_candidates: list[tuple[_Candidate, int]] = []
    for stratum in config.strata:
        key = (stratum.lineage, stratum.triage_category)
        stratum_selected: list[_Candidate] = []
        for candidate in candidates_by_stratum.get(key, ()):
            source_sha256 = candidate.alias.snapshot_record.source_sha256
            filename = candidate.record.document_filename
            if source_sha256 in used_content or filename in used_filenames:
                continue
            used_content.add(source_sha256)
            used_filenames.add(filename)
            stratum_selected.append(candidate)
            if len(stratum_selected) == stratum.documents:
                break
        if len(stratum_selected) != stratum.documents:
            raise PilotSelectionError(
                f"stratum {key!r} can provide only {len(stratum_selected)} unique documents "
                f"for quota {stratum.documents} after global content deduplication"
            )
        selected_candidates.extend(
            (candidate, rank) for rank, candidate in enumerate(stratum_selected, start=1)
        )

    records: list[PilotDocumentRecord] = []
    for selection_rank, (candidate, stratum_rank) in enumerate(
        selected_candidates,
        start=1,
    ):
        source = candidate.alias.snapshot_record
        match = _DOCUMENT_FILENAME.fullmatch(candidate.record.document_filename)
        if match is None:
            raise PilotSelectionError(
                "selected filename is not a stable document ID: "
                f"{candidate.record.document_filename}"
            )
        records.append(
            PilotDocumentRecord(
                schema_version=1,
                selection_rank=selection_rank,
                stratum_rank=stratum_rank,
                selection_score_sha256=candidate.score,
                catalog_sha256=config.expected_catalog_sha256,
                final_label=config.final_label,
                lineage=candidate.record.lineage,
                triage_category=candidate.category,
                document_filename=candidate.record.document_filename,
                source_sha256=source.source_sha256,
                source_size_bytes=source.source_size_bytes,
                document_page_count=source.actual_page_count,
                pilot_relative_path=PurePosixPath(
                    "files",
                    config.final_label,
                    match.group("document_date")[:4],
                    candidate.record.document_filename,
                ).as_posix(),
                selected_local_alias=candidate.alias,
                catalog_record=candidate.record,
            )
        )
    selected = tuple(records)
    if len(selected) != config.document_count:
        raise PilotSelectionError("selected pilot size differs from document_count")
    payload = _canonical_jsonl(selected)
    statistics = _statistics(
        config=config,
        local_surface_count=len(local_surface),
        quality_count=len(quality_eligible),
        bounded=bounded,
        selected=selected,
        candidates_by_stratum=candidates_by_stratum,
    )
    return _PilotPlan(
        catalog=catalog,
        records=selected,
        manifest_payload=payload,
        manifest_sha256=sha256_bytes(payload),
        statistics=statistics,
    )


def _source_path(record: PilotDocumentRecord) -> Path:
    alias = record.selected_local_alias
    return Path(alias.snapshot_root) / Path(
        *PurePosixPath(alias.snapshot_record.snapshot_relative_path).parts
    )


def _link_groups(
    root: Path,
    plan: _PilotPlan,
) -> tuple[tuple[Path, Path, tuple[str, ...]], ...]:
    grouped: dict[tuple[Path, Path], list[str]] = defaultdict(list)
    for record in plan.records:
        source = _source_path(record)
        destination = root / Path(*PurePosixPath(record.pilot_relative_path).parts)
        if source.name != destination.name:
            raise PilotIntegrityError("pilot source and destination filenames differ")
        grouped[(source.parent, destination.parent)].append(source.name)
    return tuple(
        (source, destination, tuple(sorted(names)))
        for (source, destination), names in sorted(
            grouped.items(), key=lambda item: (str(item[0][0]), str(item[0][1]))
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
        source_stat = os.stat(name, dir_fd=source_fd, follow_symlinks=False)
        destination_stat = os.stat(name, dir_fd=destination_fd, follow_symlinks=False)
    except OSError as error:
        raise PilotIntegrityError(
            f"cannot validate pilot hard link: {destination_parent / name}"
        ) from error
    if (
        not stat.S_ISREG(source_stat.st_mode)
        or not stat.S_ISREG(destination_stat.st_mode)
        or (source_stat.st_dev, source_stat.st_ino)
        != (destination_stat.st_dev, destination_stat.st_ino)
    ):
        raise PilotIntegrityError(
            f"pilot PDF is not its immutable source hard link: {destination_parent / name}"
        )


def _publish_links(root: Path, plan: _PilotPlan) -> None:
    for source_parent, destination_parent, names in _link_groups(root, plan):
        source = _safe_snapshot_directory(source_parent, create=False)
        destination = _safe_snapshot_directory(destination_parent, create=True)
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
                    pass
                except OSError as error:
                    raise PilotIntegrityError(
                        f"failed to hard-link pilot PDF: {destination / name}"
                    ) from error
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


def _verify_pdfs(
    config: CatalogPilotConfig,
    root: Path,
    plan: _PilotPlan,
    *,
    progress: PilotProgress | None,
) -> None:
    expected = {
        root / Path(*PurePosixPath(record.pilot_relative_path).parts) for record in plan.records
    }
    actual = set(_safe_walk_files(root / "files"))
    if actual != expected:
        missing = sorted(str(path) for path in expected - actual)
        unexpected = sorted(str(path) for path in actual - expected)
        raise PilotIntegrityError(
            f"pilot PDF tree differs from manifest; missing={missing[:3]!r}, "
            f"unexpected={unexpected[:3]!r}"
        )

    completed = 0
    with ThreadPoolExecutor(
        max_workers=config.verification_workers,
        thread_name_prefix="pilot-verify",
    ) as executor:
        futures = {
            executor.submit(
                _hash_local_pdf,
                root / Path(*PurePosixPath(record.pilot_relative_path).parts),
                max_pdf_bytes=config.max_pdf_bytes,
            ): record
            for record in plan.records
        }
        for future in as_completed(futures):
            record = futures[future]
            digest, size_bytes = future.result()
            if digest != record.source_sha256 or size_bytes != record.source_size_bytes:
                raise PilotIntegrityError(
                    f"pilot PDF differs from manifest: {record.pilot_relative_path}"
                )
            completed += 1
            if progress is not None and (completed % 50 == 0 or completed == len(plan.records)):
                progress("verify_pilot_pdf", completed, len(plan.records))

    for source_parent, destination_parent, names in _link_groups(root, plan):
        source = _safe_snapshot_directory(source_parent, create=False)
        destination = _safe_snapshot_directory(destination_parent, create=False)
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


def _configuration_sha256(config: CatalogPilotConfig) -> str:
    payload = config.model_dump(mode="json")
    if config.schema_version == 1 and payload.pop("excluded_pilots") != []:
        raise PilotIntegrityError("pilot schema version 1 unexpectedly contains exclusions")
    return canonical_json_sha256(payload)


def _build_commit(config: CatalogPilotConfig, plan: _PilotPlan) -> PilotCommit:
    return PilotCommit(
        schema_version=config.schema_version,
        final_label=config.final_label,
        configuration_sha256=_configuration_sha256(config),
        catalog_config_path=str(plan.catalog.config_path),
        catalog_config_sha256=plan.catalog.config_sha256,
        catalog_sha256=plan.catalog.result.commit.catalog_sha256,
        selection_namespace=config.selection_namespace,
        quality=config.quality,
        deduplicate_by_content_sha256=config.deduplicate_by_content_sha256,
        max_pages_per_document=config.max_pages_per_document,
        strata=config.strata,
        manifest_path="manifest.jsonl",
        manifest_sha256=plan.manifest_sha256,
        statistics=plan.statistics,
    )


def _expected_tree(root: Path, plan: _PilotPlan) -> set[Path]:
    pdfs = {
        root / Path(*PurePosixPath(record.pilot_relative_path).parts) for record in plan.records
    }
    return pdfs | {
        root / "manifest.jsonl",
        root / "manifest.jsonl.sha256",
        root / "pilot.json",
    }


def plan_pilot(
    config: CatalogPilotConfig,
    *,
    progress: PilotProgress | None = None,
) -> PilotPlanSummary:
    """Verify catalog inputs and report the deterministic pilot manifest digest."""

    plan = _build_plan(config, progress=progress)
    return PilotPlanSummary(
        manifest_sha256=plan.manifest_sha256,
        statistics=plan.statistics,
    )


def materialize_pilot(
    config: CatalogPilotConfig,
    *,
    progress: PilotProgress | None = None,
) -> PilotResult:
    """Hard-link or fully verify a manifest-last catalog pilot corpus."""

    root = _safe_snapshot_directory(Path(config.destination_root), create=True)
    commit_path = root / "pilot.json"
    if commit_path.exists() or commit_path.is_symlink():
        return verify_pilot(config, progress=progress)
    plan = _build_plan(config, progress=progress)
    if plan.manifest_sha256 != config.expected_manifest_sha256:
        raise PilotSelectionError(
            f"pilot manifest SHA-256 changed ({plan.manifest_sha256} != "
            f"{config.expected_manifest_sha256})"
        )
    _publish_links(root, plan)
    _verify_pdfs(config, root, plan, progress=progress)
    published_digest = _publish_digest_bound_artifact(
        root / "manifest.jsonl",
        plan.manifest_payload,
    )
    if published_digest != plan.manifest_sha256:
        raise PilotIntegrityError("published pilot manifest digest changed")
    commit = _build_commit(config, plan)
    try:
        atomic_publish_json(commit_path, commit.model_dump(mode="json"))
    except AtomicConflictError as error:
        raise PilotIntegrityError(f"pilot commit publication conflict: {commit_path}") from error
    if set(_safe_walk_files(root)) != _expected_tree(root, plan):
        raise PilotIntegrityError("pilot file tree differs immediately after commit publication")
    return PilotResult(root=root, commit=commit, created=True)


def verify_pilot(
    config: CatalogPilotConfig,
    *,
    progress: PilotProgress | None = None,
) -> PilotResult:
    """Offline-verify catalog selection, every hard link, manifest, and commit."""

    root = _safe_snapshot_directory(Path(config.destination_root), create=False)
    _, commit = _load_canonical_pilot_commit(root)
    if commit.configuration_sha256 != _configuration_sha256(config):
        raise PilotIntegrityError("pilot configuration differs from committed configuration")
    if commit.manifest_sha256 != config.expected_manifest_sha256:
        raise PilotIntegrityError("pilot manifest hash differs from configuration")

    plan = _build_plan(config, progress=progress)
    if commit != _build_commit(config, plan):
        raise PilotIntegrityError("pilot commit differs from recomputed catalog selection")
    manifest_payload = _read_digest_bound_artifact(
        root / commit.manifest_path,
        commit.manifest_sha256,
    )
    if manifest_payload != plan.manifest_payload:
        raise PilotIntegrityError("pilot manifest differs from recomputed catalog selection")
    _verify_pdfs(config, root, plan, progress=progress)
    expected_tree = _expected_tree(root, plan)
    actual_tree = set(_safe_walk_files(root))
    if actual_tree != expected_tree:
        missing = sorted(str(path) for path in expected_tree - actual_tree)
        unexpected = sorted(str(path) for path in actual_tree - expected_tree)
        raise PilotIntegrityError(
            f"pilot file tree differs from commit; missing={missing[:3]!r}, "
            f"unexpected={unexpected[:3]!r}"
        )
    return PilotResult(root=root, commit=commit, created=False)
