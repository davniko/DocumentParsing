"""Manifest-last local corpora assembled from immutable S3 snapshots.

The OCR pipeline consumes one local root.  This module composes several
already-verified snapshot roots without copying PDF payloads: exact filenames
are the document key, identical duplicates become provenance aliases, and each
published corpus PDF is a hard link to one immutable snapshot original.
"""

from __future__ import annotations

import json
import os
import re
import stat
from collections import defaultdict
from collections.abc import Callable, Iterable
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from datetime import date
from pathlib import Path, PurePosixPath
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from document_ocr.atomic import (
    AtomicConflictError,
    atomic_publish_json,
    read_regular_file_bytes,
)
from document_ocr.config import LocalCorpusConfig, load_snapshot_config
from document_ocr.hashing import canonical_json_bytes, sha256_bytes
from document_ocr.snapshot import (
    SnapshotFileRecord,
    SnapshotResult,
    _hash_local_pdf,
    _load_snapshot_records,
    _publish_digest_bound_artifact,
    _read_digest_bound_artifact,
    _safe_snapshot_directory,
    _safe_walk_files,
    verify_snapshot,
)

_DOCUMENT_FILENAME = re.compile(
    r"^(?P<document_date>\d{4}-\d{2}-\d{2})_"
    r"(?P<document_uuid>[0-9a-f]{8}-[0-9a-f]{4}-[1-5][0-9a-f]{3}-"
    r"[89ab][0-9a-f]{3}-[0-9a-f]{12})\.pdf$"
)


class _FrozenRecord(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, strict=True, allow_inf_nan=False)


class CorpusSourceAlias(_FrozenRecord):
    """One exact snapshot source represented by a combined corpus document."""

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
        if len(value) != 64 or any(character not in "0123456789abcdef" for character in value):
            raise ValueError("hash must be a lowercase 64-character SHA-256")
        return value

    @model_validator(mode="after")
    def source_is_ready(self) -> CorpusSourceAlias:
        if self.snapshot_record.extraction_status != "ready":
            raise ValueError("corpus source aliases must reference extraction-ready records")
        return self


class CorpusDocumentRecord(_FrozenRecord):
    """One exact filename in the combined corpus and all of its source aliases."""

    schema_version: Literal[1]
    document_type: str = Field(min_length=1, max_length=32, pattern=r"^[a-z][a-z0-9_-]*$")
    document_key: str = Field(min_length=1)
    document_date: str
    document_uuid: str
    filename: str = Field(min_length=1)
    source_sha256: str
    source_size_bytes: int = Field(gt=0)
    document_page_count: int = Field(gt=0)
    corpus_relative_path: str = Field(min_length=1)
    source_aliases: list[CorpusSourceAlias] = Field(min_length=1)

    @field_validator("source_sha256")
    @classmethod
    def source_hash_is_sha256(cls, value: str) -> str:
        if len(value) != 64 or any(character not in "0123456789abcdef" for character in value):
            raise ValueError("source_sha256 must be a lowercase 64-character SHA-256")
        return value

    @model_validator(mode="after")
    def identity_paths_and_aliases_are_consistent(self) -> CorpusDocumentRecord:
        match = _DOCUMENT_FILENAME.fullmatch(self.filename)
        if match is None:
            raise ValueError("filename must be YYYY-MM-DD_<UUID>.pdf")
        try:
            date.fromisoformat(match.group("document_date"))
        except ValueError as error:
            raise ValueError("filename contains an invalid calendar date") from error
        if self.document_key != self.filename.removesuffix(".pdf"):
            raise ValueError("document_key must equal the filename stem")
        if self.document_date != match.group("document_date"):
            raise ValueError("document_date must equal the filename date")
        if self.document_uuid != match.group("document_uuid"):
            raise ValueError("document_uuid must equal the filename UUID")
        expected_path = PurePosixPath(
            "files",
            self.document_type,
            self.document_date[:4],
            self.filename,
        )
        if self.corpus_relative_path != expected_path.as_posix():
            raise ValueError("corpus_relative_path must equal files/<type>/<year>/<filename>")
        ordered_aliases = sorted(
            self.source_aliases,
            key=lambda item: (
                item.snapshot_record.source_key,
                item.snapshot_manifest_sha256,
                item.snapshot_config_sha256,
            ),
        )
        if self.source_aliases != ordered_aliases:
            raise ValueError("source_aliases must be canonically sorted")
        alias_keys = [
            (
                item.snapshot_config_sha256,
                item.snapshot_manifest_sha256,
                item.snapshot_record.source_key,
            )
            for item in self.source_aliases
        ]
        if len(alias_keys) != len(set(alias_keys)):
            raise ValueError("source_aliases must be unique")
        for alias in self.source_aliases:
            record = alias.snapshot_record
            if record.document_type != self.document_type:
                raise ValueError("source alias document type differs from corpus document type")
            if PurePosixPath(record.source_key).name != self.filename:
                raise ValueError("source alias filename differs from corpus filename")
            if (
                record.source_sha256 != self.source_sha256
                or record.source_size_bytes != self.source_size_bytes
                or record.actual_page_count != self.document_page_count
            ):
                raise ValueError("source alias content identity differs from corpus document")
        return self


class CorpusSourceSummary(_FrozenRecord):
    snapshot_root_name: str = Field(min_length=1)
    snapshot_config_sha256: str
    snapshot_commit_sha256: str
    snapshot_manifest_sha256: str
    snapshot_selection_sha256: str
    selected_documents: int = Field(gt=0)
    selected_pages: int = Field(gt=0)
    selected_bytes: int = Field(gt=0)
    quarantined_documents: int = Field(ge=0)

    @field_validator(
        "snapshot_config_sha256",
        "snapshot_commit_sha256",
        "snapshot_manifest_sha256",
        "snapshot_selection_sha256",
    )
    @classmethod
    def hashes_are_sha256(cls, value: str) -> str:
        if len(value) != 64 or any(character not in "0123456789abcdef" for character in value):
            raise ValueError("hash must be a lowercase 64-character SHA-256")
        return value


class CorpusCommit(_FrozenRecord):
    """Manifest-last proof that a combined local corpus is complete."""

    schema_version: Literal[1]
    document_type: str = Field(min_length=1)
    manifest_path: Literal["manifest.jsonl"]
    manifest_sha256: str
    source_snapshots: list[CorpusSourceSummary] = Field(min_length=2)
    document_count: int = Field(gt=0)
    document_page_count: int = Field(gt=0)
    logical_source_bytes: int = Field(gt=0)
    source_alias_count: int = Field(gt=0)
    deduplicated_alias_count: int = Field(ge=0)

    @field_validator("manifest_sha256")
    @classmethod
    def manifest_hash_is_sha256(cls, value: str) -> str:
        if len(value) != 64 or any(character not in "0123456789abcdef" for character in value):
            raise ValueError("manifest_sha256 must be a lowercase 64-character SHA-256")
        return value

    @model_validator(mode="after")
    def aliases_partition_into_documents_and_duplicates(self) -> CorpusCommit:
        if self.source_alias_count != self.document_count + self.deduplicated_alias_count:
            raise ValueError("source_alias_count must equal documents plus deduplicated aliases")
        if self.source_alias_count != sum(
            item.selected_documents for item in self.source_snapshots
        ):
            raise ValueError("source_alias_count differs from source snapshot summaries")
        return self


@dataclass(frozen=True, slots=True)
class CorpusResult:
    root: Path
    commit: CorpusCommit
    created: bool


@dataclass(frozen=True, slots=True)
class _VerifiedSnapshot:
    root: Path
    config_sha256: str
    commit_sha256: str
    result: SnapshotResult
    selected_records: tuple[SnapshotFileRecord, ...]
    summary: CorpusSourceSummary

    @property
    def identity(self) -> tuple[str, str]:
        return self.config_sha256, self.result.commit.manifest_sha256


@dataclass(frozen=True, slots=True)
class _CorpusPlan:
    records: tuple[CorpusDocumentRecord, ...]
    manifest_payload: bytes
    manifest_sha256: str
    sources: tuple[_VerifiedSnapshot, ...]


class CorpusError(RuntimeError):
    """Base class for explicit local-corpus failures."""


class CorpusSelectionError(CorpusError):
    """Source snapshots cannot form the configured exact-filename corpus."""


class CorpusIntegrityError(CorpusError):
    """Published corpus bytes or metadata violate the immutable contract."""


CorpusProgress = Callable[[str, int, int], None]


def _canonical_jsonl(records: Iterable[BaseModel]) -> bytes:
    return b"".join(
        canonical_json_bytes(record.model_dump(mode="json")) + b"\n" for record in records
    )


def _canonical_path(path: Path, *, description: str) -> Path:
    absolute = Path(os.path.abspath(path))
    try:
        resolved = path.resolve(strict=True)
    except OSError as error:
        raise CorpusIntegrityError(f"{description} does not exist: {path}") from error
    if path != absolute or resolved != absolute:
        raise CorpusIntegrityError(
            f"{description} must be absolute and must not traverse symbolic links: {path}"
        )
    return absolute


def _verified_sources(
    config: LocalCorpusConfig,
    *,
    progress: CorpusProgress | None,
) -> tuple[_VerifiedSnapshot, ...]:
    verified: list[_VerifiedSnapshot] = []
    total = len(config.sources)
    for index, source in enumerate(config.sources, start=1):
        if progress is not None:
            progress("verify_source_snapshot", index - 1, total)
        config_path = _canonical_path(
            Path(source.snapshot_config),
            description="snapshot configuration",
        )
        config_payload = read_regular_file_bytes(config_path)
        snapshot_config = load_snapshot_config(config_path)
        if read_regular_file_bytes(config_path) != config_payload:
            raise CorpusIntegrityError(
                f"snapshot configuration changed while loading: {config_path}"
            )
        result = verify_snapshot(snapshot_config)
        destination = Path(os.path.abspath(config.destination_root))
        if (
            destination == result.root
            or destination in result.root.parents
            or result.root in destination.parents
        ):
            raise CorpusSelectionError(
                "combined corpus destination must not overlap a source snapshot root"
            )
        commit_payload = read_regular_file_bytes(result.root / "snapshot.json")
        expected_commit_payload = (
            json.dumps(
                result.commit.model_dump(mode="json"),
                allow_nan=False,
                ensure_ascii=False,
                indent=2,
                sort_keys=True,
            ).encode("utf-8")
            + b"\n"
        )
        if commit_payload != expected_commit_payload:
            raise CorpusIntegrityError(
                f"verified snapshot commit changed after verification: {result.root}"
            )
        manifest_payload = read_regular_file_bytes(result.root / result.commit.manifest_path)
        if sha256_bytes(manifest_payload) != result.commit.manifest_sha256:
            raise CorpusIntegrityError(
                f"verified snapshot manifest changed after verification: {result.root}"
            )
        records = _load_snapshot_records(
            manifest_payload,
            result.root / result.commit.manifest_path,
        )
        selected = tuple(
            record
            for record in records
            if record.document_type == config.document_type and record.extraction_status == "ready"
        )
        if not selected:
            raise CorpusSelectionError(
                f"snapshot has no ready {config.document_type!r} documents: {result.root}"
            )
        summary = CorpusSourceSummary(
            snapshot_root_name=result.root.name,
            snapshot_config_sha256=sha256_bytes(config_payload),
            snapshot_commit_sha256=sha256_bytes(commit_payload),
            snapshot_manifest_sha256=result.commit.manifest_sha256,
            snapshot_selection_sha256=result.commit.selection_sha256,
            selected_documents=len(selected),
            selected_pages=sum(item.actual_page_count for item in selected),
            selected_bytes=sum(item.source_size_bytes for item in selected),
            quarantined_documents=sum(
                item.document_type == config.document_type
                and item.extraction_status == "quarantined"
                for item in records
            ),
        )
        verified.append(
            _VerifiedSnapshot(
                root=result.root,
                config_sha256=summary.snapshot_config_sha256,
                commit_sha256=summary.snapshot_commit_sha256,
                result=result,
                selected_records=selected,
                summary=summary,
            )
        )
        if progress is not None:
            progress("verify_source_snapshot", index, total)
    identities = [item.identity for item in verified]
    if len(identities) != len(set(identities)):
        raise CorpusSelectionError("source snapshots must have unique verified identities")
    return tuple(sorted(verified, key=lambda item: item.identity))


def _source_alias(source: _VerifiedSnapshot, record: SnapshotFileRecord) -> CorpusSourceAlias:
    return CorpusSourceAlias(
        snapshot_config_sha256=source.config_sha256,
        snapshot_commit_sha256=source.commit_sha256,
        snapshot_manifest_sha256=source.result.commit.manifest_sha256,
        snapshot_selection_sha256=source.result.commit.selection_sha256,
        snapshot_record=record,
    )


def _build_plan(
    config: LocalCorpusConfig,
    *,
    progress: CorpusProgress | None,
) -> _CorpusPlan:
    sources = _verified_sources(config, progress=progress)
    grouped: dict[str, list[CorpusSourceAlias]] = defaultdict(list)
    for source in sources:
        for record in source.selected_records:
            filename = PurePosixPath(record.source_key).name
            grouped[filename].append(_source_alias(source, record))

    records: list[CorpusDocumentRecord] = []
    conflicts: list[str] = []
    for filename, raw_aliases in sorted(grouped.items()):
        match = _DOCUMENT_FILENAME.fullmatch(filename)
        if match is None:
            raise CorpusSelectionError(
                f"BLC filename does not match YYYY-MM-DD_<UUID>.pdf: {filename}"
            )
        aliases = sorted(
            raw_aliases,
            key=lambda item: (
                item.snapshot_record.source_key,
                item.snapshot_manifest_sha256,
                item.snapshot_config_sha256,
            ),
        )
        first = aliases[0].snapshot_record
        has_conflict = False
        for alias in aliases[1:]:
            candidate = alias.snapshot_record
            if (
                candidate.source_sha256 != first.source_sha256
                or candidate.source_size_bytes != first.source_size_bytes
                or candidate.actual_page_count != first.actual_page_count
            ):
                has_conflict = True
        if has_conflict:
            conflicts.append(filename)
            continue
        records.append(
            CorpusDocumentRecord(
                schema_version=1,
                document_type=config.document_type,
                document_key=filename.removesuffix(".pdf"),
                document_date=match.group("document_date"),
                document_uuid=match.group("document_uuid"),
                filename=filename,
                source_sha256=first.source_sha256,
                source_size_bytes=first.source_size_bytes,
                document_page_count=first.actual_page_count,
                corpus_relative_path=PurePosixPath(
                    "files",
                    config.document_type,
                    match.group("document_date")[:4],
                    filename,
                ).as_posix(),
                source_aliases=aliases,
            )
        )
    if conflicts:
        raise CorpusSelectionError(
            f"same filename has conflicting content in {len(conflicts)} source groups; "
            f"examples: {', '.join(conflicts[:5])}"
        )
    ordered = tuple(records)
    payload = _canonical_jsonl(ordered)
    return _CorpusPlan(
        records=ordered,
        manifest_payload=payload,
        manifest_sha256=sha256_bytes(payload),
        sources=sources,
    )


def _source_for_alias(
    sources: tuple[_VerifiedSnapshot, ...],
    alias: CorpusSourceAlias,
) -> _VerifiedSnapshot:
    matches = [
        source
        for source in sources
        if source.config_sha256 == alias.snapshot_config_sha256
        and source.result.commit.manifest_sha256 == alias.snapshot_manifest_sha256
    ]
    if len(matches) != 1:
        raise CorpusIntegrityError("corpus alias does not identify exactly one source snapshot")
    return matches[0]


def _link_groups(
    root: Path,
    plan: _CorpusPlan,
) -> tuple[tuple[Path, Path, tuple[str, ...]], ...]:
    grouped: dict[tuple[Path, Path], list[str]] = defaultdict(list)
    for record in plan.records:
        primary = record.source_aliases[0]
        source = _source_for_alias(plan.sources, primary)
        source_path = source.root / Path(
            *PurePosixPath(primary.snapshot_record.snapshot_relative_path).parts
        )
        destination = root / Path(*PurePosixPath(record.corpus_relative_path).parts)
        if source_path.name != destination.name:
            raise CorpusIntegrityError("source and corpus filenames differ")
        grouped[(source_path.parent, destination.parent)].append(source_path.name)
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
        source_stat = os.stat(name, dir_fd=source_fd, follow_symlinks=False)
        destination_stat = os.stat(name, dir_fd=destination_fd, follow_symlinks=False)
    except OSError as error:
        raise CorpusIntegrityError(
            f"cannot validate corpus hard link: {destination_parent / name}"
        ) from error
    if (
        not stat.S_ISREG(source_stat.st_mode)
        or not stat.S_ISREG(destination_stat.st_mode)
        or (source_stat.st_dev, source_stat.st_ino)
        != (destination_stat.st_dev, destination_stat.st_ino)
    ):
        raise CorpusIntegrityError(
            f"corpus PDF is not its immutable source hard link: {destination_parent / name}"
        )


def _publish_links(root: Path, plan: _CorpusPlan) -> None:
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
                    raise CorpusIntegrityError(
                        f"failed to hard-link corpus PDF: {destination / name}"
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


def _expected_file_paths(root: Path, plan: _CorpusPlan) -> set[Path]:
    return {
        root / Path(*PurePosixPath(record.corpus_relative_path).parts) for record in plan.records
    }


def _verify_files(
    config: LocalCorpusConfig,
    root: Path,
    plan: _CorpusPlan,
    *,
    progress: CorpusProgress | None,
) -> None:
    expected = _expected_file_paths(root, plan)
    actual = set(_safe_walk_files(root / "files"))
    if actual != expected:
        missing = sorted(str(path) for path in expected - actual)
        unexpected = sorted(str(path) for path in actual - expected)
        raise CorpusIntegrityError(
            f"corpus file tree differs from manifest; missing={missing[:3]!r}, "
            f"unexpected={unexpected[:3]!r}"
        )

    completed = 0
    with ThreadPoolExecutor(
        max_workers=config.verification_workers,
        thread_name_prefix="corpus-verify",
    ) as executor:
        futures = {
            executor.submit(
                _hash_local_pdf,
                root / Path(*PurePosixPath(record.corpus_relative_path).parts),
                max_pdf_bytes=config.max_pdf_bytes,
            ): record
            for record in plan.records
        }
        for future in as_completed(futures):
            record = futures[future]
            digest, size_bytes = future.result()
            if digest != record.source_sha256 or size_bytes != record.source_size_bytes:
                raise CorpusIntegrityError(
                    f"corpus PDF differs from manifest: {record.corpus_relative_path}"
                )
            completed += 1
            if progress is not None and (completed % 100 == 0 or completed == len(plan.records)):
                progress("verify_corpus_pdf", completed, len(plan.records))

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


def _build_commit(config: LocalCorpusConfig, plan: _CorpusPlan) -> CorpusCommit:
    alias_count = sum(len(record.source_aliases) for record in plan.records)
    return CorpusCommit(
        schema_version=1,
        document_type=config.document_type,
        manifest_path="manifest.jsonl",
        manifest_sha256=plan.manifest_sha256,
        source_snapshots=[source.summary for source in plan.sources],
        document_count=len(plan.records),
        document_page_count=sum(record.document_page_count for record in plan.records),
        logical_source_bytes=sum(record.source_size_bytes for record in plan.records),
        source_alias_count=alias_count,
        deduplicated_alias_count=alias_count - len(plan.records),
    )


def plan_corpus(
    config: LocalCorpusConfig,
    *,
    progress: CorpusProgress | None = None,
) -> tuple[str, int]:
    """Verify source snapshots and return the deterministic manifest digest and row count."""

    plan = _build_plan(config, progress=progress)
    return plan.manifest_sha256, len(plan.records)


def materialize_corpus(
    config: LocalCorpusConfig,
    *,
    progress: CorpusProgress | None = None,
) -> CorpusResult:
    """Create or fully verify a hard-linked, manifest-last combined corpus."""

    root = _safe_snapshot_directory(Path(config.destination_root), create=True)
    commit_path = root / "corpus.json"
    if commit_path.exists() or commit_path.is_symlink():
        return verify_corpus(config, progress=progress)

    plan = _build_plan(config, progress=progress)
    if plan.manifest_sha256 != config.expected_manifest_sha256:
        raise CorpusSelectionError(
            f"corpus manifest SHA-256 changed ({plan.manifest_sha256} != "
            f"{config.expected_manifest_sha256})"
        )
    _publish_links(root, plan)
    _verify_files(config, root, plan, progress=progress)
    published_digest = _publish_digest_bound_artifact(
        root / "manifest.jsonl",
        plan.manifest_payload,
    )
    if published_digest != plan.manifest_sha256:
        raise CorpusIntegrityError("published corpus manifest digest changed")
    commit = _build_commit(config, plan)
    try:
        atomic_publish_json(commit_path, commit.model_dump(mode="json"))
    except AtomicConflictError as error:
        raise CorpusIntegrityError(f"corpus commit publication conflict: {commit_path}") from error
    return CorpusResult(root=root, commit=commit, created=True)


def verify_corpus(
    config: LocalCorpusConfig,
    *,
    progress: CorpusProgress | None = None,
) -> CorpusResult:
    """Re-verify source snapshots, manifest bytes, links, and every corpus PDF."""

    root = _safe_snapshot_directory(Path(config.destination_root), create=False)
    commit_path = root / "corpus.json"
    try:
        commit_payload = read_regular_file_bytes(commit_path)
        commit = CorpusCommit.model_validate_json(commit_payload, strict=True)
    except Exception as error:
        raise CorpusIntegrityError(f"invalid or missing corpus commit: {commit_path}") from error
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
        raise CorpusIntegrityError(f"corpus commit is not canonical: {commit_path}")
    if commit.document_type != config.document_type:
        raise CorpusIntegrityError("corpus document type differs from configuration")
    if commit.manifest_sha256 != config.expected_manifest_sha256:
        raise CorpusIntegrityError("corpus manifest digest differs from configuration")

    plan = _build_plan(config, progress=progress)
    if plan.manifest_sha256 != commit.manifest_sha256:
        raise CorpusIntegrityError("current source snapshots differ from corpus commit")
    manifest_payload = _read_digest_bound_artifact(
        root / commit.manifest_path,
        commit.manifest_sha256,
    )
    if manifest_payload != plan.manifest_payload:
        raise CorpusIntegrityError("corpus manifest rows differ from verified source plan")
    recomputed_commit = _build_commit(config, plan)
    if recomputed_commit != commit:
        raise CorpusIntegrityError("corpus totals differ from manifest and source snapshots")
    _verify_files(config, root, plan, progress=progress)
    return CorpusResult(root=root, commit=commit, created=False)
