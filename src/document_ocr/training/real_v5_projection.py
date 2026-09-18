"""Project the frozen real relation-v3 split into the latest relation-v5 task.

The prior training run selected the comparison validation cohort at runtime.  This
publisher pins that exact cohort, migrates every target through the audited v5
boundary, and records any source-grounded correction needed before migration.
The train and validation JSONLs are published before a final completion manifest.
"""

from __future__ import annotations

import json
from collections.abc import Mapping
from copy import deepcopy
from pathlib import Path
from typing import Any, cast

from document_ocr.atomic import atomic_publish_bytes, atomic_publish_json, read_regular_file_bytes
from document_ocr.hashing import canonical_json_bytes, sha256_bytes, sha256_file
from document_ocr.synthesis.template_compiler.latest_target import latest_target_from_source
from document_ocr.training.config import (
    RealV5EquipmentCorrectionConfig,
    RealV5ProjectionConfig,
    resolve_config_path,
)


def _reject_json_constant(constant: str) -> Any:
    raise ValueError(f"non-standard JSON constant {constant!r}")


def _reject_duplicate_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    value: dict[str, Any] = {}
    for key, item in pairs:
        if key in value:
            raise ValueError(f"duplicate JSON key {key!r}")
        value[key] = item
    return value


def _pinned_file(project_root: Path, configured_path: str, expected_sha256: str) -> Path:
    unresolved = resolve_config_path(project_root, configured_path)
    if unresolved.is_symlink():
        raise ValueError(f"pinned input must not be a symbolic link: {unresolved}")
    try:
        path = unresolved.resolve(strict=True)
    except FileNotFoundError as error:
        raise ValueError(f"pinned input does not exist: {unresolved}") from error
    if not path.is_file():
        raise ValueError(f"pinned input is not a regular file: {path}")
    actual_sha256 = sha256_file(path)
    if actual_sha256 != expected_sha256:
        raise ValueError(
            f"pinned input SHA-256 mismatch for {path}: expected {expected_sha256}, "
            f"found {actual_sha256}"
        )
    return path


def _output_directory(project_root: Path, configured: str) -> Path:
    unresolved = resolve_config_path(project_root, configured)
    if unresolved == Path(unresolved.anchor):
        raise ValueError("projection output directory must not be a filesystem root")
    current = Path(unresolved.anchor) if unresolved.is_absolute() else project_root
    parts = unresolved.parts[1:] if unresolved.is_absolute() else Path(configured).parts
    for part in parts:
        current /= part
        if current.is_symlink():
            raise ValueError(f"projection output path must not traverse a symlink: {current}")
    path = unresolved.resolve()
    path.mkdir(parents=True, exist_ok=True)
    if path.is_symlink() or not path.is_dir():
        raise ValueError(f"projection output must be a real directory: {path}")
    return path


def _strict_json_object(payload: bytes, context: str) -> dict[str, Any]:
    try:
        decoded = payload.decode("utf-8")
    except UnicodeDecodeError as error:
        raise ValueError(f"{context} is not valid UTF-8") from error
    try:
        value = json.loads(
            decoded,
            object_pairs_hook=_reject_duplicate_keys,
            parse_constant=_reject_json_constant,
        )
    except (json.JSONDecodeError, ValueError) as error:
        raise ValueError(f"{context} is not strict JSON: {error}") from error
    if not isinstance(value, dict):
        raise ValueError(f"{context} root must be an object")
    return cast(dict[str, Any], value)


def _required_text(record: Mapping[str, Any], field: str, context: str) -> str:
    value = record.get(field)
    if not isinstance(value, str) or not value.strip() or "\x00" in value:
        raise ValueError(f"{context}: {field} must be non-empty NUL-free text")
    return value


def _validation_ids(report: Mapping[str, Any], config: RealV5ProjectionConfig) -> list[str]:
    inspection = report.get("inspection")
    partition = inspection.get("partition") if isinstance(inspection, dict) else None
    outputs = partition.get("outputs") if isinstance(partition, dict) else None
    validation = outputs.get("validation") if isinstance(outputs, dict) else None
    values = validation.get("document_ids") if isinstance(validation, dict) else None
    declared_sha256 = (
        validation.get("document_ids_sha256") if isinstance(validation, dict) else None
    )
    if not isinstance(values, list) or not all(isinstance(value, str) for value in values):
        raise ValueError("baseline dataset report has no valid validation document ID list")
    document_ids = cast(list[str], values)
    if len(document_ids) != len(set(document_ids)):
        raise ValueError("baseline validation document IDs are not unique")
    actual_sha256 = sha256_bytes(canonical_json_bytes(document_ids))
    if declared_sha256 != actual_sha256:
        raise ValueError("baseline validation document ID hash is internally inconsistent")
    if len(document_ids) != config.validation.records:
        raise ValueError(
            "baseline validation record count differs from projection contract: "
            f"expected {config.validation.records}, found {len(document_ids)}"
        )
    if actual_sha256 != config.validation.document_ids_sha256:
        raise ValueError(
            "baseline validation document IDs differ from projection contract: "
            f"expected {config.validation.document_ids_sha256}, found {actual_sha256}"
        )
    source_files = inspection.get("source_files") if isinstance(inspection, dict) else None
    if not isinstance(source_files, list) or not any(
        isinstance(item, dict)
        and item.get("sha256") == config.source.sha256
        and item.get("records") == config.source.records
        for item in source_files
    ):
        raise ValueError("baseline dataset report does not identify the configured source")
    return document_ids


def _apply_equipment_correction(
    *,
    target: dict[str, Any],
    raw_text: str,
    correction: RealV5EquipmentCorrectionConfig,
) -> dict[str, Any]:
    if raw_text.count(correction.evidence_text) != 1:
        raise ValueError(
            f"{correction.document_id}: correction evidence must occur exactly once in OCR"
        )
    if correction.type_description.casefold() not in correction.evidence_text.casefold():
        raise ValueError(
            f"{correction.document_id}: type description is not literal correction evidence"
        )
    corrected = deepcopy(target)
    patch = corrected.get("documentPatch")
    containers = patch.get("containers") if isinstance(patch, dict) else None
    if not isinstance(containers, list) or correction.container_index >= len(containers):
        raise ValueError(f"{correction.document_id}: correction container does not exist")
    container = containers[correction.container_index]
    if not isinstance(container, dict):
        raise ValueError(f"{correction.document_id}: correction container is not an object")
    if container.get("temperatureSetpoint") is None:
        raise ValueError(f"{correction.document_id}: correction container has no temperature")
    conflicting = {
        key: container.get(key)
        for key in ("typeDescription", "typeCategory", "sizeCategory")
        if container.get(key) is not None
    }
    if conflicting:
        raise ValueError(
            f"{correction.document_id}: correction would overwrite equipment: {conflicting}"
        )
    container["typeDescription"] = correction.type_description
    return corrected


def _descriptor(payload: bytes, document_ids: list[str]) -> dict[str, Any]:
    return {
        "sha256": sha256_bytes(payload),
        "bytes": len(payload),
        "records": len(document_ids),
        "documentIdsSha256": sha256_bytes(canonical_json_bytes(document_ids)),
    }


def project_real_v5_split(
    *, project_root: Path, config: RealV5ProjectionConfig, config_sha256: str
) -> dict[str, Any]:
    """Publish the exact baseline split with canonical relation-v5 targets."""

    source_path = _pinned_file(project_root, config.source.path, config.source.sha256)
    report_path = _pinned_file(
        project_root,
        config.baseline_dataset_report.path,
        config.baseline_dataset_report.sha256,
    )
    report = _strict_json_object(read_regular_file_bytes(report_path), str(report_path))
    validation_document_ids = _validation_ids(report, config)
    validation_ids = set(validation_document_ids)
    corrections = {item.document_id: item for item in config.corrections}
    used_corrections: set[str] = set()
    seen_ids: set[str] = set()
    train_lines: list[bytes] = []
    validation_lines: list[bytes] = []
    train_ids: list[str] = []
    output_validation_ids: list[str] = []

    source_payload = read_regular_file_bytes(source_path)
    if source_payload and not source_payload.endswith(b"\n"):
        raise ValueError("projection source JSONL must end with a newline")
    for line_number, encoded_line in enumerate(source_payload.splitlines(), start=1):
        context = f"{source_path}:{line_number}"
        if not encoded_line.strip():
            raise ValueError(f"{context}: blank JSONL lines are forbidden")
        record = _strict_json_object(encoded_line, context)
        document_id = _required_text(record, "documentId", context)
        if document_id in seen_ids:
            raise ValueError(f"{context}: duplicate document ID {document_id}")
        seen_ids.add(document_id)
        raw_text = _required_text(record, "joinedRawText", context)
        input_sha256 = _required_text(record, "joinedRawTextSha256", context)
        actual_input_sha256 = sha256_bytes(raw_text.encode("utf-8"))
        if input_sha256 != actual_input_sha256:
            raise ValueError(f"{context}: joinedRawTextSha256 mismatch")
        source_target = record.get("target")
        if not isinstance(source_target, dict):
            raise ValueError(f"{context}: target must be an object")
        target = cast(dict[str, Any], source_target)
        correction = corrections.get(document_id)
        if correction is not None:
            if correction.input_sha256 != input_sha256:
                raise ValueError(f"{context}: correction input SHA-256 mismatch")
            target = _apply_equipment_correction(
                target=target,
                raw_text=raw_text,
                correction=correction,
            )
            used_corrections.add(document_id)
        migrated = latest_target_from_source(target)
        output_record = {
            "documentId": document_id,
            "joinedRawText": raw_text,
            "joinedRawTextSha256": input_sha256,
            "target": migrated,
        }
        output_line = canonical_json_bytes(output_record) + b"\n"
        if document_id in validation_ids:
            validation_lines.append(output_line)
            output_validation_ids.append(document_id)
        else:
            train_lines.append(output_line)
            train_ids.append(document_id)

    if len(seen_ids) != config.source.records:
        raise ValueError(
            "projection source count mismatch: "
            f"expected {config.source.records}, found {len(seen_ids)}"
        )
    if used_corrections != set(corrections):
        missing = sorted(set(corrections) - used_corrections)
        raise ValueError(f"configured corrections did not match source documents: {missing}")
    if set(output_validation_ids) != validation_ids:
        missing = sorted(validation_ids - set(output_validation_ids))
        raise ValueError(f"baseline validation documents are absent from source: {missing}")
    if output_validation_ids != validation_document_ids:
        raise ValueError("projected validation order differs from the baseline validation order")
    if len(output_validation_ids) != config.validation.records:
        raise AssertionError("validated projection emitted the wrong validation count")

    train_payload = b"".join(train_lines)
    validation_payload = b"".join(validation_lines)
    correction_rows = [
        {
            **item.model_dump(mode="json"),
            "action": "insert_source_grounded_container_type_description_before_v5_migration",
        }
        for item in config.corrections
    ]
    corrections_payload = b"".join(
        canonical_json_bytes(item) + b"\n" for item in correction_rows
    )
    output_dir = _output_directory(project_root, config.output_dir)
    train_path = output_dir / "train.jsonl"
    validation_path = output_dir / "validation.jsonl"
    corrections_path = output_dir / "corrections.jsonl"
    atomic_publish_bytes(train_path, train_payload)
    atomic_publish_bytes(validation_path, validation_payload)
    atomic_publish_bytes(corrections_path, corrections_payload)

    manifest: dict[str, Any] = {
        "schemaVersion": 1,
        "publicationStatus": "complete",
        "projection": config.projection,
        "targetTask": "bill_of_lading_relation_explicit_v5",
        "targetSchemaVersion": "5.0.0-experimental",
        "configSha256": config_sha256,
        "source": {
            "path": config.source.path,
            "sha256": config.source.sha256,
            "records": config.source.records,
        },
        "baselineDatasetReport": {
            "path": config.baseline_dataset_report.path,
            "sha256": config.baseline_dataset_report.sha256,
        },
        "validationContract": {
            "records": config.validation.records,
            "documentIdsSha256": config.validation.document_ids_sha256,
        },
        "corrections": {
            "path": corrections_path.name,
            **_descriptor(corrections_payload, [item.document_id for item in config.corrections]),
        },
        "outputs": {
            "train": {"path": train_path.name, **_descriptor(train_payload, train_ids)},
            "validation": {
                "path": validation_path.name,
                **_descriptor(validation_payload, output_validation_ids),
            },
        },
    }
    atomic_publish_json(output_dir / "manifest.json", manifest)
    return manifest
