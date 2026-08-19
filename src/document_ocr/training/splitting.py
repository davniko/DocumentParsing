"""Deterministic, immutable publication of pre-training JSONL splits."""

from __future__ import annotations

import json
import re
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast

from document_ocr.atomic import atomic_publish_bytes, atomic_publish_json, read_regular_file_bytes
from document_ocr.hashing import identity_sha256, sha256_bytes
from document_ocr.training.config import DatasetPartitionConfig, resolve_config_path

_SHA256_PATTERN = re.compile(r"^[0-9a-f]{64}$")
_RANK_NAMESPACE = "document_ocr.training_split.seeded_sha256_rank.v1"


@dataclass(frozen=True, slots=True)
class _SourceRecord:
    encoded_line: bytes
    document_id: str
    input_sha256: str
    target_leaf_paths: frozenset[str]
    rank: str


def _require_string(record: dict[str, Any], field: str, context: str) -> str:
    value = record.get(field)
    if not isinstance(value, str) or not value.strip() or "\x00" in value:
        raise ValueError(f"{context}: field {field!r} must be a non-empty NUL-free string")
    return value


def _target_leaf_paths(value: Any, prefix: str = "") -> set[str]:
    if isinstance(value, dict):
        if not value:
            raise ValueError(f"target contains an empty object at {prefix or '<root>'}")
        paths: set[str] = set()
        for key, child in value.items():
            if not isinstance(key, str) or not key:
                raise ValueError("target object keys must be non-empty strings")
            child_prefix = f"{prefix}.{key}" if prefix else key
            paths.update(_target_leaf_paths(child, child_prefix))
        return paths
    if isinstance(value, list):
        if not value:
            raise ValueError(f"target contains an empty list at {prefix}")
        paths = set()
        for child in value:
            paths.update(_target_leaf_paths(child, f"{prefix}[]"))
        return paths
    if value is None:
        raise ValueError(f"target contains null at {prefix}")
    return {prefix}


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
                target_leaf_paths=frozenset(_target_leaf_paths(target)),
                rank=identity_sha256(
                    _RANK_NAMESPACE,
                    config.seed,
                    document_id,
                    input_sha256,
                ),
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

    total_leaf_counts = Counter(
        path for record in records for path in record.target_leaf_paths
    )
    validation_leaf_counts: Counter[str] = Counter()
    validation_ids: set[str] = set()
    coverage_skipped_ids: list[str] = []
    for record in sorted(records, key=lambda item: (item.rank, item.document_id)):
        keeps_training_coverage = all(
            validation_leaf_counts[path] < total_leaf_counts[path] - 1
            for path in record.target_leaf_paths
        )
        if not keeps_training_coverage:
            coverage_skipped_ids.append(record.document_id)
            continue
        validation_ids.add(record.document_id)
        validation_leaf_counts.update(record.target_leaf_paths)
        if len(validation_ids) == config.validation_records:
            break
    if len(validation_ids) != config.validation_records:
        raise ValueError(
            "coverage policy cannot produce the requested validation size: "
            f"requested {config.validation_records}, selected {len(validation_ids)}"
        )

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
        }

    manifest: dict[str, Any] = {
        "schema_version": 1,
        "publication_status": "complete",
        "algorithm": config.algorithm,
        "rank_namespace": _RANK_NAMESPACE,
        "seed": config.seed,
        "coverage_policy": config.coverage_policy,
        "coverage_skipped_document_ids": coverage_skipped_ids,
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
