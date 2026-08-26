"""Immutable consolidation of fragmented labeling runs into one audited dataset."""

from __future__ import annotations

import argparse
import json
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
from document_ocr.labeling_agents.models import DocumentRunOutcome, NeedsReviewRecord
from document_ocr.labeling_agents.work_items import (
    AgentWorkItem,
    validate_annotation_evidence,
    validate_exclusion_evidence,
)

_DOCUMENT_ID_PATTERN = r"^doc_[0-9a-f]{64}$"
_SHA256_PATTERN = r"^[0-9a-f]{64}$"
_SAFE_ID_PATTERN = r"^[A-Za-z0-9][A-Za-z0-9._-]*$"


class ConsolidationError(RuntimeError):
    """A source lineage or consolidated artifact failed validation."""


class _ConfigModel(LabelSchemaModel):
    model_config = ConfigDict(extra="forbid", frozen=True, strict=True, allow_inf_nan=False)


class SourceRunConfig(_ConfigModel):
    run_id: Annotated[str, StringConstraints(pattern=_SAFE_ID_PATTERN)]
    root: str

    @field_validator("root")
    @classmethod
    def root_is_absolute(cls, value: str) -> str:
        if not Path(value).is_absolute():
            raise ValueError("source run root must be absolute")
        return value


class QualityHoldConfig(_ConfigModel):
    document_id: Annotated[str, StringConstraints(pattern=_DOCUMENT_ID_PATTERN)]
    code: Literal[
        "ambiguous_numeric_date_order",
        "audited_label_defect",
        "evidence_revalidation_required",
    ]
    message: Annotated[str, StringConstraints(strip_whitespace=True, min_length=1)]
    raw_values: list[Annotated[str, StringConstraints(min_length=1)]] = Field(
        min_length=1
    )


class ConsolidationConfig(_ConfigModel):
    schema_version: Literal[1]
    dataset_id: Annotated[str, StringConstraints(pattern=_SAFE_ID_PATTERN)]
    output_root: str
    selection_run_root: str
    selection_sha256: Annotated[str, StringConstraints(pattern=_SHA256_PATTERN)]
    expected_documents: Annotated[int, Field(gt=0)]
    source_runs: list[SourceRunConfig] = Field(min_length=1)
    quality_holds: list[QualityHoldConfig] = Field(default_factory=list)

    @field_validator("output_root", "selection_run_root")
    @classmethod
    def roots_are_absolute(cls, value: str) -> str:
        if not Path(value).is_absolute():
            raise ValueError("consolidation roots must be absolute")
        return value

    @model_validator(mode="after")
    def identifiers_are_unique(self) -> ConsolidationConfig:
        run_ids = tuple(row.run_id for row in self.source_runs)
        roots = tuple(row.root for row in self.source_runs)
        hold_ids = tuple(row.document_id for row in self.quality_holds)
        if len(run_ids) != len(set(run_ids)) or len(roots) != len(set(roots)):
            raise ValueError("source run IDs and roots must be unique")
        if len(hold_ids) != len(set(hold_ids)):
            raise ValueError("quality-hold document IDs must be unique")
        return self


def load_consolidation_config(path: Path) -> ConsolidationConfig:
    payload = load_strict_yaml_mapping(path)
    try:
        return ConsolidationConfig.model_validate(payload, strict=True)
    except ValueError as error:
        raise ConsolidationError(f"invalid consolidation config: {path}: {error}") from error


def _contained_file(root: Path, relative: str, *, context: str) -> Path:
    unresolved = root / relative
    try:
        path = unresolved.resolve(strict=True)
    except OSError as error:
        raise ConsolidationError(f"{context} is absent: {unresolved}") from error
    if path != root and root not in path.parents:
        raise ConsolidationError(f"{context} escapes its source run: {relative}")
    if not path.is_file():
        raise ConsolidationError(f"{context} is not a regular file: {path}")
    return path


def _json(payload: bytes, *, context: str) -> dict[str, Any]:
    try:
        value = json.loads(payload)
    except (UnicodeError, json.JSONDecodeError) as error:
        raise ConsolidationError(f"{context} is not valid JSON") from error
    if not isinstance(value, dict):
        raise ConsolidationError(f"{context} must be a JSON object")
    return value


def _selection_rows(payload: bytes, *, expected: int) -> tuple[dict[str, Any], ...]:
    rows: list[dict[str, Any]] = []
    for line_number, line in enumerate(payload.splitlines(), start=1):
        if not line:
            raise ConsolidationError(f"selection contains a blank row at line {line_number}")
        rows.append(_json(line, context=f"selection line {line_number}"))
    if len(rows) != expected:
        raise ConsolidationError(
            f"selection count differs from expected_documents: {len(rows)} != {expected}"
        )
    ids = tuple(row.get("documentId") for row in rows)
    if any(not isinstance(value, str) for value in ids) or len(ids) != len(set(ids)):
        raise ConsolidationError("selection document IDs must be present and unique")
    return tuple(rows)


def _source_run_root(configured: SourceRunConfig) -> Path:
    try:
        root = Path(configured.root).resolve(strict=True)
    except OSError as error:
        raise ConsolidationError(f"source run root is absent: {configured.root}") from error
    manifest = _json(
        read_regular_file_bytes(_contained_file(root, "manifest.json", context="source manifest")),
        context=f"source manifest {root}",
    )
    resolved = _json(
        read_regular_file_bytes(
            _contained_file(root, "resolved-config.json", context="source resolved config")
        ),
        context=f"source resolved config {root}",
    )
    if manifest.get("run_id") != configured.run_id:
        raise ConsolidationError(f"source manifest run ID differs: {root}")
    run_value = resolved.get("run")
    if not isinstance(run_value, dict) or run_value.get("run_id") != configured.run_id:
        raise ConsolidationError(f"source resolved-config run ID differs: {root}")
    return root


def _load_outcome(root: Path, document_id: str) -> tuple[DocumentRunOutcome, str] | None:
    unresolved = root / "state" / "outcomes" / f"{document_id}.json"
    if not unresolved.exists():
        return None
    path = _contained_file(
        root,
        f"state/outcomes/{document_id}.json",
        context="source outcome",
    )
    payload = read_regular_file_bytes(path)
    try:
        outcome = DocumentRunOutcome.model_validate_json(payload, strict=True)
    except ValueError as error:
        raise ConsolidationError(f"source outcome failed validation: {path}") from error
    if outcome.documentId != document_id:
        raise ConsolidationError(f"source outcome document ID differs: {path}")
    return outcome, sha256_bytes(payload)


def _work_item(selection_root: Path, row: dict[str, Any]) -> AgentWorkItem:
    document_id = row["documentId"]
    path = _contained_file(
        selection_root,
        f"work-items/{document_id}.json",
        context="selection work item",
    )
    payload = read_regular_file_bytes(path)
    if sha256_bytes(payload) != row.get("workItemSha256"):
        raise ConsolidationError(f"selection work-item hash differs: {document_id}")
    try:
        item = AgentWorkItem.model_validate_json(payload, strict=True)
    except ValueError as error:
        raise ConsolidationError(f"selection work item failed validation: {document_id}") from error
    if item.source.documentId != document_id:
        raise ConsolidationError(f"selection work-item document ID differs: {document_id}")
    return item


def consolidate_labeling_runs(config: ConsolidationConfig) -> Path:
    selection_root = Path(config.selection_run_root).resolve(strict=True)
    selection_path = _contained_file(
        selection_root, "selection.jsonl", context="source selection"
    )
    selection_payload = read_regular_file_bytes(selection_path)
    if sha256_bytes(selection_payload) != config.selection_sha256:
        raise ConsolidationError("selection SHA-256 differs from the consolidation config")
    selection = _selection_rows(selection_payload, expected=config.expected_documents)
    selected_ids = {str(row["documentId"]) for row in selection}
    holds = {row.document_id: row for row in config.quality_holds}
    if not set(holds).issubset(selected_ids):
        raise ConsolidationError("quality hold references a document outside the selection")

    source_roots = tuple(
        (configured, _source_run_root(configured)) for configured in config.source_runs
    )
    output_root = Path(config.output_root).resolve(strict=True)
    dataset_root = output_root / "consolidated" / config.dataset_id

    training_rows: list[dict[str, Any]] = []
    lineage_rows: list[dict[str, Any]] = []
    counts = {"validated": 0, "excluded": 0, "needs_review": 0}
    selected_pages = 0

    for selection_row in selection:
        document_id = str(selection_row["documentId"])
        item = _work_item(selection_root, selection_row)
        selected_pages += item.source.documentPageCount
        history: list[dict[str, Any]] = []
        chosen: tuple[SourceRunConfig, Path, DocumentRunOutcome, str] | None = None
        for source_config, source_root in source_roots:
            loaded = _load_outcome(source_root, document_id)
            if loaded is None:
                continue
            outcome, outcome_sha256 = loaded
            history.append(
                {
                    "runId": source_config.run_id,
                    "status": outcome.status,
                    "outcomePath": f"state/outcomes/{document_id}.json",
                    "outcomeSha256": outcome_sha256,
                    "artifactPath": outcome.finalArtifactPath,
                    "artifactSha256": outcome.finalArtifactSha256,
                }
            )
            chosen = (source_config, source_root, outcome, outcome_sha256)
        if chosen is None:
            raise ConsolidationError(
                f"no source outcome exists for selected document: {document_id}"
            )

        source_config, source_root, outcome, outcome_sha256 = chosen
        source_artifact = _contained_file(
            source_root, outcome.finalArtifactPath, context="source final artifact"
        )
        artifact_payload = read_regular_file_bytes(source_artifact)
        if sha256_bytes(artifact_payload) != outcome.finalArtifactSha256:
            raise ConsolidationError(f"source final-artifact hash differs: {document_id}")

        hold = holds.get(document_id)
        consolidated_status = "needs_review" if hold is not None else outcome.status
        if consolidated_status == "validated":
            try:
                annotation = BillOfLadingDualCargoAnnotation.model_validate_json(
                    artifact_payload, strict=True
                )
            except ValueError as error:
                raise ConsolidationError(
                    f"validated annotation failed schema validation: {document_id}"
                ) from error
            validate_annotation_evidence(item, annotation)
            destination = dataset_root / "validated" / f"{document_id}.json"
            atomic_publish_bytes(destination, artifact_payload)
            normal_target = annotation.normalLabel.canonical_target()
            relation_target = annotation.relationExplicitLabel.canonical_target()
            training_rows.append(
                {
                    "documentId": document_id,
                    "joinedRawText": item.joinedRawText,
                    "joinedRawTextSha256": item.source.joinedRawTextSha256,
                    "target": relation_target,
                    "normalTarget": normal_target,
                    "validatedAnnotationPath": destination.relative_to(dataset_root).as_posix(),
                    "validatedAnnotationSha256": outcome.finalArtifactSha256,
                }
            )
        elif consolidated_status == "excluded":
            try:
                exclusion = BillOfLadingExclusion.model_validate_json(
                    artifact_payload, strict=True
                )
            except ValueError as error:
                raise ConsolidationError(
                    f"exclusion failed schema validation: {document_id}"
                ) from error
            validate_exclusion_evidence(item, exclusion)
            destination = dataset_root / "exclusions" / f"{document_id}.json"
            atomic_publish_bytes(destination, artifact_payload)
        elif hold is not None:
            try:
                annotation = BillOfLadingDualCargoAnnotation.model_validate_json(
                    artifact_payload, strict=True
                )
            except ValueError as error:
                raise ConsolidationError(
                    f"quality-held annotation failed schema validation: {document_id}"
                ) from error
            validate_annotation_evidence(item, annotation)
            if any(raw_value not in item.joinedRawText for raw_value in hold.raw_values):
                raise ConsolidationError(f"quality-hold raw value is not in OCR: {document_id}")
            destination = dataset_root / "needs-review" / f"{document_id}.json"
            atomic_publish_bytes(destination, artifact_payload)
        else:
            try:
                NeedsReviewRecord.model_validate_json(artifact_payload, strict=True)
            except ValueError as error:
                raise ConsolidationError(
                    f"needs-review artifact failed schema validation: {document_id}"
                ) from error
            destination = dataset_root / "needs-review" / f"{document_id}.json"
            atomic_publish_bytes(destination, artifact_payload)

        counts[consolidated_status] += 1
        lineage_rows.append(
            {
                "documentId": document_id,
                "consolidatedStatus": consolidated_status,
                "consolidatedArtifactPath": destination.relative_to(dataset_root).as_posix(),
                "consolidatedArtifactSha256": outcome.finalArtifactSha256,
                "selectedSourceRunId": source_config.run_id,
                "selectedSourceOutcomePath": f"state/outcomes/{document_id}.json",
                "selectedSourceOutcomeSha256": outcome_sha256,
                "selectedSourceArtifactPath": outcome.finalArtifactPath,
                "qualityHold": hold.model_dump(mode="json") if hold is not None else None,
                "consideredSourceOutcomes": history,
            }
        )

    training_payload = b"".join(canonical_json_bytes(row) + b"\n" for row in training_rows)
    lineage_payload = b"".join(canonical_json_bytes(row) + b"\n" for row in lineage_rows)
    training_path = dataset_root / "training" / "records.jsonl"
    lineage_path = dataset_root / "lineage.jsonl"
    copied_selection = dataset_root / "selection.jsonl"
    atomic_publish_bytes(copied_selection, selection_payload)
    atomic_publish_bytes(training_path, training_payload)
    atomic_publish_bytes(lineage_path, lineage_payload)

    manifest = {
        "schemaVersion": 1,
        "datasetId": config.dataset_id,
        "task": "bill_of_lading_relation_single_source_v4",
        "selectedDocuments": config.expected_documents,
        "selectedPages": selected_pages,
        "outcomes": counts,
        "trainingRecords": len(training_rows),
        "trainingReady": counts["needs_review"] == 0,
        "selectionSource": {
            "runRoot": str(selection_root),
            "path": "selection.jsonl",
            "sha256": config.selection_sha256,
        },
        "sourcePrecedence": [row.model_dump(mode="json") for row in config.source_runs],
        "qualityHolds": [row.model_dump(mode="json") for row in config.quality_holds],
        "files": [
            {
                "kind": "selection",
                "path": copied_selection.relative_to(dataset_root).as_posix(),
                "rows": config.expected_documents,
                "bytes": len(selection_payload),
                "sha256": sha256_bytes(selection_payload),
            },
            {
                "kind": "training_records",
                "path": training_path.relative_to(dataset_root).as_posix(),
                "rows": len(training_rows),
                "bytes": len(training_payload),
                "sha256": sha256_bytes(training_payload),
            },
            {
                "kind": "lineage",
                "path": lineage_path.relative_to(dataset_root).as_posix(),
                "rows": len(lineage_rows),
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
        raise ConsolidationError(message)


def main(argv: Sequence[str] | None = None) -> int:
    parser = _ArgumentParser(description="Consolidate immutable labeling-run outcomes.")
    parser.add_argument("--config", required=True, type=Path)
    arguments = parser.parse_args(argv)
    config = load_consolidation_config(arguments.config)
    manifest = consolidate_labeling_runs(config)
    print(
        canonical_json_bytes(
            {"command": "consolidate", "manifest_path": str(manifest), "status": "complete"}
        ).decode("utf-8")
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
