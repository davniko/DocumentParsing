"""Deterministic, immutable publication of pre-training JSONL splits."""

from __future__ import annotations

import json
import re
from collections import Counter
from collections.abc import Sequence
from dataclasses import dataclass
from decimal import ROUND_HALF_UP, Decimal
from pathlib import Path
from typing import Any, cast

from document_ocr.atomic import atomic_publish_bytes, atomic_publish_json, read_regular_file_bytes
from document_ocr.hashing import canonical_json_bytes, identity_sha256, sha256_bytes
from document_ocr.training.config import (
    DatasetPartitionConfig,
    RuntimeDatasetPartitionConfig,
    ValidationFractionConfig,
    ValidationRecordsConfig,
    resolve_config_path,
)

_SHA256_PATTERN = re.compile(r"^[0-9a-f]{64}$")
_RANK_NAMESPACE = "document_ocr.training_split.seeded_sha256_rank.v1"


@dataclass(frozen=True, slots=True)
class _SourceRecord:
    encoded_line: bytes
    document_id: str
    input_sha256: str
    target_leaf_paths: frozenset[str]


@dataclass(frozen=True, slots=True)
class PartitionCandidate:
    """The immutable identity and target coverage needed to assign one record."""

    document_id: str
    input_sha256: str
    target_leaf_paths: frozenset[str]


@dataclass(frozen=True, slots=True)
class DatasetPartitionSelection:
    """One deterministic in-memory train/validation membership selection."""

    algorithm: str
    rank_namespace: str
    seed: int
    requested_validation_size: dict[str, Any]
    resolved_validation_records: int
    coverage_policy: str
    coverage_skipped_document_ids: tuple[str, ...]
    train_document_ids: tuple[str, ...]
    validation_document_ids: tuple[str, ...]

    def to_dict(self) -> dict[str, Any]:
        def split_descriptor(document_ids: tuple[str, ...]) -> dict[str, Any]:
            return {
                "records": len(document_ids),
                "document_ids": list(document_ids),
                "document_ids_sha256": sha256_bytes(
                    canonical_json_bytes(list(document_ids))
                ),
            }

        return {
            "algorithm": self.algorithm,
            "rank_namespace": self.rank_namespace,
            "seed": self.seed,
            "requested_validation_size": self.requested_validation_size,
            "resolved_validation_records": self.resolved_validation_records,
            "coverage_policy": self.coverage_policy,
            "coverage_skipped_document_ids": list(
                self.coverage_skipped_document_ids
            ),
            "outputs": {
                "train": split_descriptor(self.train_document_ids),
                "validation": split_descriptor(self.validation_document_ids),
            },
        }


def _require_string(record: dict[str, Any], field: str, context: str) -> str:
    value = record.get(field)
    if not isinstance(value, str) or not value.strip() or "\x00" in value:
        raise ValueError(f"{context}: field {field!r} must be a non-empty NUL-free string")
    return value


def target_leaf_paths(value: Any, prefix: str = "") -> set[str]:
    """Return coverage paths, treating valid empty collections as terminal values."""

    if isinstance(value, dict):
        if not value:
            return {prefix or "<root>"}
        paths: set[str] = set()
        for key, child in value.items():
            if not isinstance(key, str) or not key:
                raise ValueError("target object keys must be non-empty strings")
            child_prefix = f"{prefix}.{key}" if prefix else key
            paths.update(target_leaf_paths(child, child_prefix))
        return paths
    if isinstance(value, list):
        if not value:
            return {prefix or "<root>"}
        paths = set()
        for child in value:
            paths.update(target_leaf_paths(child, f"{prefix}[]"))
        return paths
    if value is None:
        raise ValueError(f"target contains null at {prefix}")
    return {prefix}


def resolve_validation_records(
    config: RuntimeDatasetPartitionConfig,
    total_records: int,
) -> int:
    """Resolve a record or fractional validation request without binary rounding drift."""

    requested = config.validation_size
    if isinstance(requested, ValidationRecordsConfig):
        resolved = requested.value
    elif isinstance(requested, ValidationFractionConfig):
        resolved = int(
            (Decimal(str(requested.value)) * Decimal(total_records)).quantize(
                Decimal("1"),
                rounding=ROUND_HALF_UP,
            )
        )
    else:  # pragma: no cover - the strict discriminated config union is exhaustive
        raise TypeError(f"unsupported validation-size config: {type(requested).__name__}")
    if resolved <= 0 or resolved >= total_records:
        raise ValueError(
            "resolved validation size must be between one and source.records - 1: "
            f"resolved={resolved}, source.records={total_records}"
        )
    return resolved


def _select_partition(
    candidates: Sequence[PartitionCandidate],
    *,
    algorithm: str,
    seed: int,
    requested_validation_size: dict[str, Any],
    validation_records: int,
    coverage_policy: str,
) -> DatasetPartitionSelection:
    if algorithm != "seeded_sha256_rank_v1":
        raise ValueError(f"unsupported partition algorithm: {algorithm!r}")
    if coverage_policy != "retain_each_target_leaf_in_train":
        raise ValueError(f"unsupported partition coverage policy: {coverage_policy!r}")
    if validation_records <= 0 or validation_records >= len(candidates):
        raise ValueError("validation_records must be between one and source records - 1")

    seen_document_ids: set[str] = set()
    seen_input_sha256: set[str] = set()
    for candidate in candidates:
        if candidate.document_id in seen_document_ids:
            raise ValueError(f"duplicate document ID {candidate.document_id!r}")
        if candidate.input_sha256 in seen_input_sha256:
            raise ValueError(
                f"duplicate input SHA-256 {candidate.input_sha256!r}; deduplicate upstream"
            )
        if not candidate.target_leaf_paths:
            raise ValueError(
                f"partition candidate {candidate.document_id!r} has no target leaf paths"
            )
        seen_document_ids.add(candidate.document_id)
        seen_input_sha256.add(candidate.input_sha256)

    total_leaf_counts = Counter(
        path for candidate in candidates for path in candidate.target_leaf_paths
    )
    validation_leaf_counts: Counter[str] = Counter()
    validation_ids: set[str] = set()
    coverage_skipped_ids: list[str] = []
    ranked = sorted(
        candidates,
        key=lambda candidate: (
            identity_sha256(
                _RANK_NAMESPACE,
                seed,
                candidate.document_id,
                candidate.input_sha256,
            ),
            candidate.document_id,
        ),
    )
    for candidate in ranked:
        keeps_training_coverage = all(
            validation_leaf_counts[path] < total_leaf_counts[path] - 1
            for path in candidate.target_leaf_paths
        )
        if not keeps_training_coverage:
            coverage_skipped_ids.append(candidate.document_id)
            continue
        validation_ids.add(candidate.document_id)
        validation_leaf_counts.update(candidate.target_leaf_paths)
        if len(validation_ids) == validation_records:
            break
    if len(validation_ids) != validation_records:
        raise ValueError(
            "coverage policy cannot produce the requested validation size: "
            f"requested {validation_records}, selected {len(validation_ids)}"
        )

    train_document_ids = tuple(
        candidate.document_id
        for candidate in candidates
        if candidate.document_id not in validation_ids
    )
    validation_document_ids = tuple(
        candidate.document_id
        for candidate in candidates
        if candidate.document_id in validation_ids
    )
    return DatasetPartitionSelection(
        algorithm=algorithm,
        rank_namespace=_RANK_NAMESPACE,
        seed=seed,
        requested_validation_size=requested_validation_size,
        resolved_validation_records=validation_records,
        coverage_policy=coverage_policy,
        coverage_skipped_document_ids=tuple(coverage_skipped_ids),
        train_document_ids=train_document_ids,
        validation_document_ids=validation_document_ids,
    )


def select_runtime_partition(
    candidates: Sequence[PartitionCandidate],
    config: RuntimeDatasetPartitionConfig,
) -> DatasetPartitionSelection:
    """Select a configured runtime partition while preserving source order per split."""

    return _select_partition(
        candidates,
        algorithm=config.algorithm,
        seed=config.seed,
        requested_validation_size=config.validation_size.model_dump(mode="json"),
        validation_records=resolve_validation_records(config, len(candidates)),
        coverage_policy=config.coverage_policy,
    )


def _reject_json_constant(constant: str) -> Any:
    raise ValueError(f"non-standard JSON constant {constant!r}")


def _source_path(project_root: Path, config: DatasetPartitionConfig) -> Path:
    unresolved = resolve_config_path(project_root, config.source.path)
    if unresolved.is_symlink():
        raise ValueError(f"partition source must not be a symbolic link: {unresolved}")
    try:
        path = unresolved.resolve(strict=True)
    except FileNotFoundError as error:
        raise ValueError(f"partition source does not exist: {unresolved}") from error
    if not path.is_file():
        raise ValueError(f"partition source is not a regular file: {path}")
    return path


def _output_directory(project_root: Path, configured: str) -> Path:
    unresolved = resolve_config_path(project_root, configured)
    if unresolved == Path(unresolved.anchor):
        raise ValueError("partition output directory must not be a filesystem root")
    current = Path(unresolved.anchor) if unresolved.is_absolute() else project_root
    parts = unresolved.parts[1:] if unresolved.is_absolute() else Path(configured).parts
    for part in parts:
        current /= part
        if current.is_symlink():
            raise ValueError(f"partition output path must not traverse a symbolic link: {current}")
    path = unresolved.resolve()
    path.mkdir(parents=True, exist_ok=True)
    if not path.is_dir() or path.is_symlink():
        raise ValueError(f"partition output must be a real directory: {path}")
    return path


def _parse_records(
    payload: bytes,
    config: DatasetPartitionConfig,
    source_path: Path,
) -> list[_SourceRecord]:
    if payload.startswith(b"\xef\xbb\xbf"):
        raise ValueError("partition source must not contain a UTF-8 byte-order mark")
    if b"\x00" in payload:
        raise ValueError("partition source must not contain a NUL byte")
    if payload and not payload.endswith(b"\n"):
        raise ValueError("partition source JSONL must end with a newline")

    records: list[_SourceRecord] = []
    seen_document_ids: set[str] = set()
    seen_input_hashes: set[str] = set()
    fields = config.fields
    for line_number, encoded_line in enumerate(payload.splitlines(keepends=True), start=1):
        context = f"{source_path}:{line_number}"
        if not encoded_line.strip():
            raise ValueError(f"{context}: blank JSONL lines are forbidden")
        try:
            decoded = encoded_line.decode("utf-8")
        except UnicodeDecodeError as error:
            raise ValueError(f"{context}: line is not valid UTF-8") from error
        try:
            value = json.loads(decoded, parse_constant=_reject_json_constant)
        except (json.JSONDecodeError, ValueError) as error:
            raise ValueError(f"{context}: invalid strict JSON: {error}") from error
        if not isinstance(value, dict):
            raise ValueError(f"{context}: each JSONL record must be an object")
        record = cast(dict[str, Any], value)
        document_id = _require_string(record, fields.document_id, context)
        input_sha256 = _require_string(record, fields.input_sha256, context)
        if not _SHA256_PATTERN.fullmatch(input_sha256):
            raise ValueError(f"{context}: input SHA-256 must be lowercase hexadecimal")
        if document_id in seen_document_ids:
            raise ValueError(f"{context}: duplicate document ID {document_id!r}")
        if input_sha256 in seen_input_hashes:
            raise ValueError(
                f"{context}: duplicate input SHA-256 {input_sha256!r}; deduplicate upstream"
            )
        seen_document_ids.add(document_id)
        seen_input_hashes.add(input_sha256)
        target = record.get(fields.target)
        if not isinstance(target, dict):
            raise ValueError(f"{context}: target field {fields.target!r} must be an object")
        records.append(
            _SourceRecord(
                encoded_line=encoded_line,
                document_id=document_id,
                input_sha256=input_sha256,
                target_leaf_paths=frozenset(target_leaf_paths(target)),
            )
        )
    if len(records) != config.source.records:
        raise ValueError(
            "partition source record-count mismatch: "
            f"expected {config.source.records}, found {len(records)}"
        )
    return records


def partition_dataset(
    *,
    project_root: Path,
    config: DatasetPartitionConfig,
) -> dict[str, Any]:
    """Publish train/validation files deterministically, with the manifest last."""

    source_path = _source_path(project_root, config)
    source_payload = read_regular_file_bytes(source_path)
    source_sha256 = sha256_bytes(source_payload)
    if source_sha256 != config.source.sha256:
        raise ValueError(
            "partition source SHA-256 mismatch: "
            f"expected {config.source.sha256}, found {source_sha256}"
        )
    records = _parse_records(source_payload, config, source_path)

    selection = _select_partition(
        [
            PartitionCandidate(
                document_id=record.document_id,
                input_sha256=record.input_sha256,
                target_leaf_paths=record.target_leaf_paths,
            )
            for record in records
        ],
        algorithm=config.algorithm,
        seed=config.seed,
        requested_validation_size={
            "kind": "records",
            "value": config.validation_records,
        },
        validation_records=config.validation_records,
        coverage_policy=config.coverage_policy,
    )
    validation_ids = set(selection.validation_document_ids)

    train = [record for record in records if record.document_id not in validation_ids]
    validation = [record for record in records if record.document_id in validation_ids]
    train_payload = b"".join(record.encoded_line for record in train)
    validation_payload = b"".join(record.encoded_line for record in validation)
    output_dir = _output_directory(project_root, config.output_dir)
    train_path = output_dir / "train.jsonl"
    validation_path = output_dir / "validation.jsonl"
    atomic_publish_bytes(train_path, train_payload)
    atomic_publish_bytes(validation_path, validation_payload)

    def descriptor(
        path: Path,
        payload: bytes,
        split_records: list[_SourceRecord],
    ) -> dict[str, Any]:
        return {
            "path": path.name,
            "sha256": sha256_bytes(payload),
            "bytes": len(payload),
            "records": len(split_records),
            "document_ids": [record.document_id for record in split_records],
            "document_ids_sha256": sha256_bytes(
                canonical_json_bytes(
                    [record.document_id for record in split_records]
                )
            ),
        }

    manifest: dict[str, Any] = {
        "schema_version": 1,
        "publication_status": "complete",
        "algorithm": config.algorithm,
        "rank_namespace": _RANK_NAMESPACE,
        "seed": config.seed,
        "requested_validation_size": selection.requested_validation_size,
        "resolved_validation_records": selection.resolved_validation_records,
        "coverage_policy": config.coverage_policy,
        "coverage_skipped_document_ids": list(
            selection.coverage_skipped_document_ids
        ),
        "source": {
            "path": config.source.path,
            "sha256": source_sha256,
            "bytes": len(source_payload),
            "records": len(records),
        },
        "fields": config.fields.model_dump(mode="json"),
        "outputs": {
            "train": descriptor(train_path, train_payload, train),
            "validation": descriptor(validation_path, validation_payload, validation),
        },
    }
    atomic_publish_json(output_dir / "manifest.json", manifest)
    return manifest
