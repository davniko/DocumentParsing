"""Verify, normalize, round-trip, and publish the current synthesis source corpus."""

from __future__ import annotations

import hashlib
import json
import resource
import time
from pathlib import Path
from typing import Any, cast

from document_ocr.atomic import atomic_publish_bytes, atomic_publish_json
from document_ocr.hashing import canonical_json_bytes, sha256_bytes, sha256_file
from document_ocr.synthesis.config import SynthesisFoundationConfig
from document_ocr.synthesis.normalization import (
    denormalize_target,
    normalize_target,
)
from document_ocr.training.tasks import RelationExplicitTaskConstraints, get_training_task


def _resolve_file(project_root: Path, value: str, label: str) -> Path:
    unresolved = Path(value)
    path = unresolved if unresolved.is_absolute() else project_root / unresolved
    if path.is_symlink():
        raise ValueError(f"{label} must not be a symbolic link: {path}")
    resolved = path.resolve(strict=True)
    if not resolved.is_file():
        raise ValueError(f"{label} is not a regular file: {resolved}")
    return resolved


def _task(project_root: Path, config: SynthesisFoundationConfig) -> Any:
    task = get_training_task(config.task)
    if config.task_constraints is None:
        return task
    path = _resolve_file(project_root, config.task_constraints.path, "task-constraints")
    actual = sha256_file(path)
    if actual != config.task_constraints.sha256:
        raise ValueError("task-constraints SHA-256 mismatch")
    constraints = RelationExplicitTaskConstraints.model_validate_json(
        path.read_bytes(), strict=True
    )
    canonical = canonical_json_bytes(constraints.model_dump(mode="json")) + b"\n"
    if path.read_bytes() != canonical:
        raise ValueError("task-constraints artifact must use canonical JSON plus one newline")
    return task.bind_constraints(constraints)


def _jsonl(rows: list[dict[str, Any]]) -> bytes:
    return b"".join(
        json.dumps(
            row, ensure_ascii=False, allow_nan=False, sort_keys=True, separators=(",", ":")
        ).encode("utf-8")
        + b"\n"
        for row in rows
    )


def _provenance_path(path: Path, project_root: Path) -> str:
    try:
        return str(path.relative_to(project_root.resolve()))
    except ValueError:
        return str(path)


def _sdv_metadata() -> dict[str, Any]:
    return {
        "METADATA_SPEC_VERSION": "V1",
        "tables": {
            "documents": {
                "primary_key": "document_id",
                "columns": {
                    "document_id": {"sdtype": "id"},
                    "source_row_index": {"sdtype": "numerical", "computer_representation": "Int64"},
                    "target_sha256": {"sdtype": "categorical"},
                    "root_node_id": {"sdtype": "categorical"},
                    "node_count": {"sdtype": "numerical", "computer_representation": "Int64"},
                },
            },
            "nodes": {
                "primary_key": "node_id",
                "columns": {
                    "document_id": {"sdtype": "id"},
                    "node_id": {"sdtype": "id"},
                    "parent_node_id": {"sdtype": "categorical"},
                    "edge_kind": {"sdtype": "categorical"},
                    "edge_name": {"sdtype": "categorical"},
                    "edge_index": {"sdtype": "numerical", "computer_representation": "Int64"},
                    "child_order": {"sdtype": "numerical", "computer_representation": "Int64"},
                    "node_kind": {"sdtype": "categorical"},
                    "scalar_json": {"sdtype": "categorical"},
                },
            },
        },
        "relationships": [
            {
                "parent_table_name": "documents",
                "parent_primary_key": "document_id",
                "child_table_name": "nodes",
                "child_foreign_key": "document_id",
            }
        ],
    }


def _validate_sdv(
    metadata_value: dict[str, Any],
    documents: list[dict[str, Any]],
    nodes: list[dict[str, Any]],
) -> str:
    try:
        import pandas as pd  # type: ignore[import-untyped]
        from sdv.metadata import Metadata  # type: ignore[import-not-found]
    except ImportError as error:
        raise RuntimeError("SDV validation requires the isolated synthesis environment") from error
    metadata = Metadata.load_from_dict(metadata_value)
    metadata.validate()
    metadata.validate_data(
        data={"documents": pd.DataFrame(documents), "nodes": pd.DataFrame(nodes)}
    )
    return str(importlib_version("sdv"))


def importlib_version(package: str) -> str:
    from importlib.metadata import version

    return version(package)


def prepare_synthesis_foundation(
    *, project_root: Path, config_path: Path, config: SynthesisFoundationConfig
) -> dict[str, Any]:
    task = _task(project_root, config)
    source = _resolve_file(project_root, config.source.file.path, "synthesis source")
    digest = hashlib.sha256()
    documents: list[dict[str, Any]] = []
    nodes: list[dict[str, Any]] = []
    seen: set[str] = set()
    fields = config.source.fields
    assert fields.input_sha256 is not None
    started = time.perf_counter()
    with source.open("rb") as stream:
        for row_index, encoded in enumerate(stream):
            digest.update(encoded)
            try:
                source_row = json.loads(encoded)
            except (UnicodeDecodeError, json.JSONDecodeError) as error:
                raise ValueError(f"{source}:{row_index + 1}: invalid UTF-8 JSON") from error
            if not isinstance(source_row, dict):
                raise ValueError(f"{source}:{row_index + 1}: row must be an object")
            document_id = source_row.get(fields.document_id)
            input_text = source_row.get(fields.input_text)
            input_sha256 = source_row.get(fields.input_sha256)
            target_value = source_row.get(fields.target)
            if not isinstance(document_id, str) or not document_id.strip():
                raise ValueError(f"{source}:{row_index + 1}: invalid document ID")
            if document_id in seen:
                raise ValueError(f"duplicate document ID: {document_id}")
            seen.add(document_id)
            if not isinstance(input_text, str) or not input_text:
                raise ValueError(f"{source}:{row_index + 1}: input text must be non-empty")
            expected_input_sha256 = hashlib.sha256(input_text.encode("utf-8")).hexdigest()
            if input_sha256 != expected_input_sha256:
                raise ValueError(
                    f"{source}:{row_index + 1}: input SHA-256 mismatch for {document_id}"
                )
            if not isinstance(target_value, dict):
                raise ValueError(f"{source}:{row_index + 1}: target must be an object")
            canonical = task.canonicalize(cast(dict[str, Any], target_value))
            normalized = normalize_target(document_id, canonical)
            rebuilt = denormalize_target(normalized)
            if rebuilt != canonical or task.canonicalize(rebuilt) != canonical:
                raise RuntimeError(f"lossless round-trip failed for {document_id}")
            documents.append(
                {
                    "document_id": document_id,
                    "source_row_index": row_index,
                    "target_sha256": sha256_bytes(canonical_json_bytes(canonical)),
                    "root_node_id": normalized[0].node_id,
                    "node_count": len(normalized),
                }
            )
            nodes.extend(node.to_dict() for node in normalized)
    actual_digest = digest.hexdigest()
    expected = config.source.file
    if actual_digest != expected.sha256:
        raise ValueError(
            f"source SHA-256 mismatch: expected {expected.sha256}, found {actual_digest}"
        )
    if len(documents) != expected.records:
        raise ValueError(
            f"source record count mismatch: expected {expected.records}, found {len(documents)}"
        )
    metadata = _sdv_metadata()
    sdv_version = (
        _validate_sdv(metadata, documents, nodes) if config.sdv.validate_with_sdv else None
    )
    output_root = Path(config.run.output_dir)
    if not output_root.is_absolute():
        output_root = project_root / output_root
    output_root = output_root.resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    run_dir = output_root / config.run.run_id
    document_payload = _jsonl(documents)
    node_payload = _jsonl(nodes)
    atomic_publish_bytes(run_dir / "tables" / "documents.jsonl", document_payload)
    atomic_publish_bytes(run_dir / "tables" / "nodes.jsonl", node_payload)
    atomic_publish_json(run_dir / "sdv-metadata.json", metadata)
    manifest = {
        "schemaVersion": 1,
        "runId": config.run.run_id,
        "task": config.task,
        "source": {
            "path": _provenance_path(source, project_root),
            "sha256": actual_digest,
            "records": len(documents),
        },
        "roundTripValidatedRecords": len(documents),
        "nodes": len(nodes),
        "sdvValidation": {
            "enabled": config.sdv.validate_with_sdv,
            "version": sdv_version,
            "metadataSha256": sha256_bytes(canonical_json_bytes(metadata)),
        },
        "files": {
            "documents.jsonl": sha256_bytes(document_payload),
            "nodes.jsonl": sha256_bytes(node_payload),
        },
        "config": {
            "path": _provenance_path(config_path.resolve(), project_root),
            "sha256": sha256_file(config_path),
        },
    }
    atomic_publish_json(run_dir / "manifest.json", manifest)
    return {
        **manifest,
        "elapsedSeconds": time.perf_counter() - started,
        "peakRssMiB": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0,
    }
