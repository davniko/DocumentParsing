"""Immutable merge of disjoint, consolidated labeling datasets."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from collections.abc import Sequence
from pathlib import Path
from typing import Annotated, Any, Literal, NoReturn

from pydantic import ConfigDict, Field, StringConstraints, field_validator, model_validator

from document_ocr.atomic import atomic_publish_bytes, atomic_publish_json, read_regular_file_bytes
from document_ocr.config import load_strict_yaml_mapping
from document_ocr.hashing import canonical_json_bytes, sha256_bytes
from document_ocr.label_schemas.bill_of_lading import BillOfLadingExclusion
from document_ocr.label_schemas.bill_of_lading_v3 import BillOfLadingDualCargoAnnotation
from document_ocr.label_schemas.common import LabelSchemaModel
from document_ocr.labeling_agents.models import NeedsReviewRecord

_SAFE_ID_PATTERN = r"^[A-Za-z0-9][A-Za-z0-9._-]*$"
_OUTCOME_STATUSES = ("validated", "excluded", "needs_review")
_STATUS_DIRECTORY = {
    "validated": "validated",
    "excluded": "exclusions",
    "needs_review": "needs-review",
}


class DatasetMergeError(RuntimeError):
    """A source dataset or merged publication failed validation."""


class _ConfigModel(LabelSchemaModel):
    model_config = ConfigDict(extra="forbid", frozen=True, strict=True, allow_inf_nan=False)


class SourceDatasetConfig(_ConfigModel):
    dataset_id: Annotated[str, StringConstraints(pattern=_SAFE_ID_PATTERN)]
    root: str

    @field_validator("root")
    @classmethod
    def root_is_absolute(cls, value: str) -> str:
        if not Path(value).is_absolute():
            raise ValueError("source dataset root must be absolute")
        return value


class DatasetMergeConfig(_ConfigModel):
    schema_version: Literal[1]
    dataset_id: Annotated[str, StringConstraints(pattern=_SAFE_ID_PATTERN)]
    output_root: str
    source_datasets: list[SourceDatasetConfig] = Field(min_length=2)

    @field_validator("output_root")
    @classmethod
    def output_root_is_absolute(cls, value: str) -> str:
        if not Path(value).is_absolute():
            raise ValueError("merge output root must be absolute")
        return value

    @model_validator(mode="after")
    def identifiers_are_unique(self) -> DatasetMergeConfig:
        dataset_ids = tuple(row.dataset_id for row in self.source_datasets)
        roots = tuple(row.root for row in self.source_datasets)
        if len(dataset_ids) != len(set(dataset_ids)) or len(roots) != len(set(roots)):
            raise ValueError("source dataset IDs and roots must be unique")
        if self.dataset_id in set(dataset_ids):
            raise ValueError("merged dataset ID must differ from every source dataset ID")
        return self


def load_dataset_merge_config(path: Path) -> DatasetMergeConfig:
    payload = load_strict_yaml_mapping(path)
    try:
        return DatasetMergeConfig.model_validate(payload, strict=True)
    except ValueError as error:
        raise DatasetMergeError(f"invalid dataset merge config: {path}: {error}") from error


def _contained_file(root: Path, relative: str, *, context: str) -> Path:
    unresolved = root / relative
    try:
        path = unresolved.resolve(strict=True)
    except OSError as error:
        raise DatasetMergeError(f"{context} is absent: {unresolved}") from error
    if path != root and root not in path.parents:
        raise DatasetMergeError(f"{context} escapes its source dataset: {relative}")
    if not path.is_file():
        raise DatasetMergeError(f"{context} is not a regular file: {path}")
    return path


def _json(payload: bytes, *, context: str) -> dict[str, Any]:
    try:
        value = json.loads(payload)
    except (UnicodeError, json.JSONDecodeError) as error:
        raise DatasetMergeError(f"{context} is not valid JSON") from error
    if not isinstance(value, dict):
        raise DatasetMergeError(f"{context} must be a JSON object")
    return value


def _jsonl(payload: bytes, *, context: str) -> tuple[dict[str, Any], ...]:
    rows: list[dict[str, Any]] = []
    for line_number, line in enumerate(payload.splitlines(), start=1):
        if not line:
            raise DatasetMergeError(f"{context} contains a blank row at line {line_number}")
        rows.append(_json(line, context=f"{context} line {line_number}"))
    return tuple(rows)


def _required_int(value: Any, *, context: str, minimum: int = 0) -> int:
    if not isinstance(value, int) or isinstance(value, bool) or value < minimum:
        raise DatasetMergeError(f"{context} must be an integer >= {minimum}")
    return value


def _manifest_file(
    root: Path,
    manifest: dict[str, Any],
    *,
    kind: str,
) -> tuple[bytes, tuple[dict[str, Any], ...]]:
    entries = manifest.get("files")
    if not isinstance(entries, list):
        raise DatasetMergeError(f"source manifest files must be a list: {root}")
    matches = [entry for entry in entries if isinstance(entry, dict) and entry.get("kind") == kind]
    if len(matches) != 1:
        raise DatasetMergeError(f"source manifest must contain one {kind} file: {root}")
    entry = matches[0]
    relative = entry.get("path")
    if not isinstance(relative, str):
        raise DatasetMergeError(f"source manifest {kind} path must be a string: {root}")
    path = _contained_file(root, relative, context=f"source {kind}")
    payload = read_regular_file_bytes(path)
    if len(payload) != _required_int(entry.get("bytes"), context=f"{kind} bytes"):
        raise DatasetMergeError(f"source {kind} byte count differs: {path}")
    if sha256_bytes(payload) != entry.get("sha256"):
        raise DatasetMergeError(f"source {kind} SHA-256 differs: {path}")
    rows = _jsonl(payload, context=f"source {kind}")
    if len(rows) != _required_int(entry.get("rows"), context=f"{kind} rows"):
        raise DatasetMergeError(f"source {kind} row count differs: {path}")
    return payload, rows


def _document_ids(
    rows: tuple[dict[str, Any], ...], *, context: str
) -> tuple[str, ...]:
    values = tuple(row.get("documentId") for row in rows)
    if any(
        not isinstance(value, str) or not value.startswith("doc_") or len(value) != 68
        for value in values
    ):
        raise DatasetMergeError(f"{context} document IDs are absent or malformed")
    document_ids = tuple(str(value) for value in values)
    if len(document_ids) != len(set(document_ids)):
        raise DatasetMergeError(f"{context} document IDs must be unique")
    return document_ids


def _validate_final_artifact(
    *,
    payload: bytes,
    status: str,
    document_id: str,
    path: Path,
) -> None:
    if status == "validated":
        try:
            annotation = BillOfLadingDualCargoAnnotation.model_validate_json(payload, strict=True)
        except ValueError as error:
            raise DatasetMergeError(
                f"validated annotation failed schema validation: {path}"
            ) from error
        artifact_document_id = annotation.source.documentId
    elif status == "excluded":
        try:
            exclusion = BillOfLadingExclusion.model_validate_json(payload, strict=True)
        except ValueError as error:
            raise DatasetMergeError(f"exclusion failed schema validation: {path}") from error
        artifact_document_id = exclusion.source.documentId
    else:
        try:
            annotation = BillOfLadingDualCargoAnnotation.model_validate_json(payload, strict=True)
            artifact_document_id = annotation.source.documentId
        except ValueError:
            try:
                record = NeedsReviewRecord.model_validate_json(payload, strict=True)
            except ValueError as error:
                raise DatasetMergeError(
                    f"needs-review artifact failed supported schema validation: {path}"
                ) from error
            artifact_document_id = record.documentId
    if artifact_document_id != document_id:
        raise DatasetMergeError(f"source artifact document ID differs: {path}")


def _source_dataset(
    configured: SourceDatasetConfig,
) -> tuple[
    Path,
    dict[str, Any],
    str,
    bytes,
    tuple[dict[str, Any], ...],
    bytes,
    tuple[dict[str, Any], ...],
    bytes,
    tuple[dict[str, Any], ...],
]:
    try:
        root = Path(configured.root).resolve(strict=True)
    except OSError as error:
        raise DatasetMergeError(f"source dataset root is absent: {configured.root}") from error
    manifest_path = _contained_file(root, "manifest.json", context="source manifest")
    manifest_payload = read_regular_file_bytes(manifest_path)
    manifest = _json(manifest_payload, context=f"source manifest {root}")
    if manifest.get("datasetId") != configured.dataset_id:
        raise DatasetMergeError(f"source manifest dataset ID differs: {root}")
    if manifest.get("schemaVersion") not in {1, 2}:
        raise DatasetMergeError(f"source manifest schema version is unsupported: {root}")
    if manifest.get("task") != "bill_of_lading_relation_single_source_v4":
        raise DatasetMergeError(f"source dataset task differs: {root}")

    selection_payload, selection = _manifest_file(root, manifest, kind="selection")
    training_payload, training = _manifest_file(root, manifest, kind="training_records")
    lineage_payload, lineage = _manifest_file(root, manifest, kind="lineage")
    selection_ids = _document_ids(selection, context="selection")
    training_ids = _document_ids(training, context="training records")
    lineage_ids = _document_ids(lineage, context="lineage")
    selected_documents = _required_int(
        manifest.get("selectedDocuments"), context="selectedDocuments", minimum=1
    )
    if len(selection_ids) != selected_documents or set(lineage_ids) != set(selection_ids):
        raise DatasetMergeError(f"source selection and lineage coverage differ: {root}")
    selected_pages = sum(
        _required_int(
            row.get("documentPageCount"),
            context="selection documentPageCount",
            minimum=1,
        )
        for row in selection
    )
    if selected_pages != _required_int(
        manifest.get("selectedPages"), context="selectedPages", minimum=1
    ):
        raise DatasetMergeError(f"source selected-page count differs: {root}")

    outcome_value = manifest.get("outcomes")
    if not isinstance(outcome_value, dict) or set(outcome_value) != set(_OUTCOME_STATUSES):
        raise DatasetMergeError(f"source outcome counts are malformed: {root}")
    manifest_counts = {
        status: _required_int(outcome_value.get(status), context=f"outcomes.{status}")
        for status in _OUTCOME_STATUSES
    }
    lineage_counts: Counter[str] = Counter()
    artifacts: dict[str, tuple[str, str]] = {}
    for row in lineage:
        document_id = str(row["documentId"])
        status = row.get("consolidatedStatus")
        if status not in _OUTCOME_STATUSES:
            raise DatasetMergeError(f"source lineage status is invalid: {document_id}")
        relative = row.get("consolidatedArtifactPath")
        artifact_sha256 = row.get("consolidatedArtifactSha256")
        expected_relative = f"{_STATUS_DIRECTORY[str(status)]}/{document_id}.json"
        if relative != expected_relative or not isinstance(artifact_sha256, str):
            raise DatasetMergeError(f"source lineage artifact reference is invalid: {document_id}")
        artifact_path = _contained_file(root, relative, context="source final artifact")
        artifact_payload = read_regular_file_bytes(artifact_path)
        if sha256_bytes(artifact_payload) != artifact_sha256:
            raise DatasetMergeError(f"source final-artifact SHA-256 differs: {document_id}")
        _validate_final_artifact(
            payload=artifact_payload,
            status=str(status),
            document_id=document_id,
            path=artifact_path,
        )
        artifacts[document_id] = (relative, artifact_sha256)
        lineage_counts[str(status)] += 1
    if dict(lineage_counts) != {key: value for key, value in manifest_counts.items() if value}:
        raise DatasetMergeError(f"source manifest and lineage outcome counts differ: {root}")
    if sum(manifest_counts.values()) != selected_documents:
        raise DatasetMergeError(f"source outcome counts do not cover the selection: {root}")
    if set(training_ids) != {
        document_id
        for document_id, row in zip(lineage_ids, lineage, strict=True)
        if row["consolidatedStatus"] == "validated"
    }:
        raise DatasetMergeError(f"source training rows differ from validated lineage: {root}")
    if len(training) != _required_int(
        manifest.get("trainingRecords"), context="trainingRecords"
    ):
        raise DatasetMergeError(f"source training-record count differs: {root}")
    for row in training:
        document_id = str(row["documentId"])
        expected_path, expected_sha256 = artifacts[document_id]
        if (
            row.get("validatedAnnotationPath") != expected_path
            or row.get("validatedAnnotationSha256") != expected_sha256
        ):
            raise DatasetMergeError(f"source training annotation reference differs: {document_id}")
    expected_ready = manifest_counts["needs_review"] == 0
    if manifest.get("trainingReady") is not expected_ready:
        raise DatasetMergeError(f"source training-ready flag differs from outcomes: {root}")

    return (
        root,
        manifest,
        sha256_bytes(manifest_payload),
        selection_payload,
        selection,
        training_payload,
        training,
        lineage_payload,
        lineage,
    )


def _merged_lineage_row(
    row: dict[str, Any],
    *,
    source_dataset_id: str,
    source_manifest_sha256: str,
) -> dict[str, Any]:
    """Add immediate-source provenance while preserving earlier merge ancestry."""

    legacy_id = row.get("sourceDatasetId")
    legacy_sha256 = row.get("sourceDatasetManifestSha256")
    existing_chain = row.get("sourceDatasetLineage")
    if existing_chain is None and legacy_id is None and legacy_sha256 is None:
        return {
            **row,
            "sourceDatasetId": source_dataset_id,
            "sourceDatasetManifestSha256": source_manifest_sha256,
        }
    if existing_chain is not None and (legacy_id is not None or legacy_sha256 is not None):
        raise DatasetMergeError("source lineage mixes legacy and chained dataset provenance")
    if existing_chain is None:
        if not isinstance(legacy_id, str) or not isinstance(legacy_sha256, str):
            raise DatasetMergeError("source lineage has incomplete dataset provenance")
        chain: list[dict[str, str]] = [
            {"datasetId": legacy_id, "manifestSha256": legacy_sha256}
        ]
    else:
        if not isinstance(existing_chain, list) or not existing_chain:
            raise DatasetMergeError("source dataset lineage must be a non-empty list")
        chain = []
        for value in existing_chain:
            if (
                not isinstance(value, dict)
                or not isinstance(value.get("datasetId"), str)
                or not isinstance(value.get("manifestSha256"), str)
            ):
                raise DatasetMergeError("source dataset lineage entry is malformed")
            chain.append(
                {
                    "datasetId": str(value["datasetId"]),
                    "manifestSha256": str(value["manifestSha256"]),
                }
            )
    if chain[-1]["datasetId"] == source_dataset_id:
        raise DatasetMergeError("source dataset lineage already ends at the immediate source")
    base = {
        key: value
        for key, value in row.items()
        if key
        not in {
            "sourceDatasetId",
            "sourceDatasetManifestSha256",
            "sourceDatasetLineage",
        }
    }
    return {
        **base,
        "sourceDatasetLineage": [
            *chain,
            {
                "datasetId": source_dataset_id,
                "manifestSha256": source_manifest_sha256,
            },
        ],
    }


def merge_label_datasets(config: DatasetMergeConfig) -> Path:
    """Publish one immutable dataset from disjoint consolidated source datasets."""

    sources = tuple(
        (configured, _source_dataset(configured)) for configured in config.source_datasets
    )
    all_document_ids: set[str] = set()
    selection_payloads: list[bytes] = []
    training_payloads: list[bytes] = []
    lineage_rows: list[dict[str, Any]] = []
    counts = {status: 0 for status in _OUTCOME_STATUSES}
    selected_documents = 0
    selected_pages = 0
    training_records = 0
    source_manifest_rows: list[dict[str, Any]] = []

    output_root = Path(config.output_root).resolve(strict=True)
    dataset_root = output_root / "consolidated" / config.dataset_id

    for configured, source in sources:
        (
            source_root,
            manifest,
            manifest_sha256,
            selection_payload,
            selection,
            training_payload,
            training,
            _lineage_payload,
            lineage,
        ) = source
        source_ids = set(_document_ids(selection, context="selection"))
        overlap = sorted(all_document_ids & source_ids)
        if overlap:
            raise DatasetMergeError(
                "source datasets contain duplicate document IDs: " + ", ".join(overlap)
            )
        all_document_ids.update(source_ids)
        selection_payloads.append(selection_payload)
        training_payloads.append(training_payload)
        selected_documents += int(manifest["selectedDocuments"])
        selected_pages += int(manifest["selectedPages"])
        training_records += len(training)
        for status in _OUTCOME_STATUSES:
            counts[status] += int(manifest["outcomes"][status])
        source_manifest_rows.append(
            {
                "datasetId": configured.dataset_id,
                "root": str(source_root),
                "manifestPath": "manifest.json",
                "manifestSha256": manifest_sha256,
                "selectedDocuments": manifest["selectedDocuments"],
                "trainingRecords": manifest["trainingRecords"],
                "trainingReady": manifest["trainingReady"],
            }
        )
        for row in lineage:
            relative = str(row["consolidatedArtifactPath"])
            source_artifact = _contained_file(
                source_root, relative, context="source final artifact"
            )
            artifact_payload = read_regular_file_bytes(source_artifact)
            atomic_publish_bytes(dataset_root / relative, artifact_payload)
            lineage_rows.append(
                _merged_lineage_row(
                    row,
                    source_dataset_id=configured.dataset_id,
                    source_manifest_sha256=manifest_sha256,
                )
            )

    selection_payload = b"".join(selection_payloads)
    training_payload = b"".join(training_payloads)
    lineage_payload = b"".join(canonical_json_bytes(row) + b"\n" for row in lineage_rows)
    selection_path = dataset_root / "selection.jsonl"
    training_path = dataset_root / "training" / "records.jsonl"
    lineage_path = dataset_root / "lineage.jsonl"
    atomic_publish_bytes(selection_path, selection_payload)
    atomic_publish_bytes(training_path, training_payload)
    atomic_publish_bytes(lineage_path, lineage_payload)

    if training_records != counts["validated"]:
        raise DatasetMergeError("merged validated and training-record counts differ")
    manifest = {
        "schemaVersion": 2,
        "datasetId": config.dataset_id,
        "task": "bill_of_lading_relation_single_source_v4",
        "selectedDocuments": selected_documents,
        "selectedPages": selected_pages,
        "outcomes": counts,
        "trainingRecords": training_records,
        "trainingReady": counts["needs_review"] == 0,
        "sourceDatasets": source_manifest_rows,
        "files": [
            {
                "kind": "selection",
                "path": selection_path.relative_to(dataset_root).as_posix(),
                "rows": selected_documents,
                "bytes": len(selection_payload),
                "sha256": sha256_bytes(selection_payload),
            },
            {
                "kind": "training_records",
                "path": training_path.relative_to(dataset_root).as_posix(),
                "rows": training_records,
                "bytes": len(training_payload),
                "sha256": sha256_bytes(training_payload),
            },
            {
                "kind": "lineage",
                "path": lineage_path.relative_to(dataset_root).as_posix(),
                "rows": selected_documents,
                "bytes": len(lineage_payload),
                "sha256": sha256_bytes(lineage_payload),
            },
        ],
    }
    manifest_path = dataset_root / "manifest.json"
    atomic_publish_json(manifest_path, manifest)
    return manifest_path


class _ArgumentParser(argparse.ArgumentParser):
    def error(self, message: str) -> NoReturn:
        raise DatasetMergeError(message)


def main(argv: Sequence[str] | None = None) -> int:
    parser = _ArgumentParser(description="Merge disjoint consolidated labeling datasets.")
    parser.add_argument("--config", required=True, type=Path)
    arguments = parser.parse_args(argv)
    config = load_dataset_merge_config(arguments.config)
    manifest = merge_label_datasets(config)
    print(
        canonical_json_bytes(
            {"command": "merge-datasets", "manifest_path": str(manifest), "status": "complete"}
        ).decode("utf-8")
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
