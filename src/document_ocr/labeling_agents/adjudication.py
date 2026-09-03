"""Immutable, evidence-checked overseer adjudication of labeling candidates."""

from __future__ import annotations

import argparse
import copy
import re
from collections.abc import Sequence
from datetime import date
from pathlib import Path
from typing import Annotated, Literal, NoReturn

from pydantic import (
    ConfigDict,
    Field,
    JsonValue,
    StringConstraints,
    field_validator,
    model_validator,
)

from document_ocr.atomic import atomic_publish_bytes, atomic_publish_json, read_regular_file_bytes
from document_ocr.config import load_strict_yaml_mapping
from document_ocr.hashing import canonical_json_bytes, sha256_bytes
from document_ocr.label_schemas.bill_of_lading_agent_v4 import (
    AgentBillOfLadingRelationExplicitLabel,
)
from document_ocr.label_schemas.bill_of_lading_v3 import BillOfLadingDualCargoAnnotation
from document_ocr.label_schemas.common import LabelSchemaModel, LabelWarning
from document_ocr.labeling_agents.deterministic_annotation import build_compact_annotation
from document_ocr.labeling_agents.models import CompactAnnotationDraft, DocumentRunOutcome
from document_ocr.labeling_agents.work_items import AgentWorkItem, validate_annotation_evidence

_DOCUMENT_ID_PATTERN = r"^doc_[0-9a-f]{64}$"
_SHA256_PATTERN = r"^[0-9a-f]{64}$"
_SAFE_ID_PATTERN = r"^[A-Za-z0-9][A-Za-z0-9._-]*$"
_NUMERIC_DATE = re.compile(
    r"^(?P<first>[0-9]{1,2})[-/.](?P<second>[0-9]{1,2})"
    r"[-/.](?P<year>[0-9]{2,4})$"
)
_JSON_POINTER = re.compile(r"^/documentPatch(?:/(?:[^~/]|~[01])+)+$")


class AdjudicationError(RuntimeError):
    """An adjudication input, correction, or publication failed validation."""


class _ConfigModel(LabelSchemaModel):
    model_config = ConfigDict(extra="forbid", frozen=True, strict=True, allow_inf_nan=False)


class DateCorrectionConfig(_ConfigModel):
    target_path: Literal[
        "documentPatch.issueDate",
        "documentPatch.shippedOnBoardDate",
    ]
    raw_value: Annotated[str, StringConstraints(strip_whitespace=True, min_length=1)]
    normalized_value: date


class SupportingArtifactConfig(_ConfigModel):
    path: str
    sha256: Annotated[str, StringConstraints(pattern=_SHA256_PATTERN)]

    @field_validator("path")
    @classmethod
    def path_is_absolute(cls, value: str) -> str:
        if not Path(value).is_absolute():
            raise ValueError("supporting artifact path must be absolute")
        return value


class CandidateRebuildConfig(_ConfigModel):
    """Re-derive projections/evidence after an audited policy or document-type correction."""

    document_type: Literal["bill_of_lading", "sea_waybill"] | None = None
    decision_notes: list[
        Annotated[str, StringConstraints(strip_whitespace=True, min_length=1)]
    ] = Field(min_length=1, max_length=3)


class TargetCorrectionConfig(_ConfigModel):
    """One exact, source-guarded replacement inside the relation-explicit target."""

    target_path: Annotated[str, StringConstraints(strip_whitespace=True, min_length=1)]
    expected_value: JsonValue
    corrected_value: JsonValue

    @field_validator("target_path")
    @classmethod
    def target_path_is_a_document_patch_pointer(cls, value: str) -> str:
        if _JSON_POINTER.fullmatch(value) is None:
            raise ValueError(
                "target correction must be an RFC 6901 pointer below /documentPatch"
            )
        return value


class WarningRemovalConfig(_ConfigModel):
    """Exact YAML-friendly source guard for removing one resolved warning."""

    code: Literal[
        "aggregate_not_allocated",
        "ambiguous_ocr_candidates",
        "image_only_value_omitted",
        "invalid_identifier_omitted",
        "schema_cannot_represent",
        "unsupported_or_unclear_code",
        "other",
    ]
    message: Annotated[str, StringConstraints(strip_whitespace=True, min_length=1)]
    pageNumbers: list[Annotated[int, Field(gt=0)]] = Field(default_factory=list)
    targetPath: str | None = None

    @model_validator(mode="after")
    def pages_are_canonical(self) -> WarningRemovalConfig:
        if self.pageNumbers != sorted(set(self.pageNumbers)):
            raise ValueError("warning-removal pageNumbers must be unique and sorted")
        return self

    def as_warning(self) -> LabelWarning:
        return LabelWarning.model_validate(
            {
                "code": self.code,
                "message": self.message,
                "pageNumbers": tuple(self.pageNumbers),
                "targetPath": self.targetPath,
            },
            strict=True,
        )


class AdjudicationItemConfig(_ConfigModel):
    document_id: Annotated[str, StringConstraints(pattern=_DOCUMENT_ID_PATTERN)]
    work_item_path: str
    work_item_sha256: Annotated[str, StringConstraints(pattern=_SHA256_PATTERN)]
    candidate_path: str
    candidate_sha256: Annotated[str, StringConstraints(pattern=_SHA256_PATTERN)]
    candidate_kind: Literal["annotation", "compact_draft"] = "annotation"
    pdf_grouping_used: bool | None = None
    rebuild: CandidateRebuildConfig | None = None
    target_corrections: list[TargetCorrectionConfig] = Field(default_factory=list)
    warning_removals: list[WarningRemovalConfig] = Field(default_factory=list)
    warning_additions: list[WarningRemovalConfig] = Field(default_factory=list)
    date_corrections: list[DateCorrectionConfig] = Field(default_factory=list)
    adjudication_notes: list[
        Annotated[str, StringConstraints(strip_whitespace=True, min_length=1)]
    ] = Field(min_length=1)
    supporting_artifacts: list[SupportingArtifactConfig] = Field(default_factory=list)

    @field_validator("work_item_path", "candidate_path")
    @classmethod
    def paths_are_absolute(cls, value: str) -> str:
        if not Path(value).is_absolute():
            raise ValueError("adjudication input paths must be absolute")
        return value

    @model_validator(mode="after")
    def correction_paths_are_unique(self) -> AdjudicationItemConfig:
        date_paths = tuple(row.target_path for row in self.date_corrections)
        if len(date_paths) != len(set(date_paths)):
            raise ValueError("date correction target paths must be unique per document")
        target_paths = tuple(row.target_path for row in self.target_corrections)
        if len(target_paths) != len(set(target_paths)):
            raise ValueError("target correction paths must be unique per document")
        if self.target_corrections and self.rebuild is None:
            raise ValueError("target corrections require an evidence rebuild")
        if (self.warning_removals or self.warning_additions) and self.rebuild is None:
            raise ValueError("warning updates require an evidence rebuild")
        warning_removal_values = tuple(
            canonical_json_bytes(row.as_warning().model_dump(mode="json"))
            for row in self.warning_removals
        )
        if len(warning_removal_values) != len(set(warning_removal_values)):
            raise ValueError("warning removals must be unique per document")
        warning_addition_values = tuple(
            canonical_json_bytes(row.as_warning().model_dump(mode="json"))
            for row in self.warning_additions
        )
        if len(warning_addition_values) != len(set(warning_addition_values)):
            raise ValueError("warning additions must be unique per document")
        if self.candidate_kind == "compact_draft" and self.pdf_grouping_used is None:
            raise ValueError(
                "compact-draft candidates require an explicit pdf_grouping_used value"
            )
        if self.candidate_kind == "annotation" and self.pdf_grouping_used is not None:
            raise ValueError(
                "pdf_grouping_used is derived from full annotations and must be omitted"
            )
        return self


class AdjudicationConfig(_ConfigModel):
    schema_version: Literal[1]
    run_id: Annotated[str, StringConstraints(pattern=_SAFE_ID_PATTERN)]
    output_root: str
    items: list[AdjudicationItemConfig] = Field(min_length=1)

    @field_validator("output_root")
    @classmethod
    def output_root_is_absolute(cls, value: str) -> str:
        if not Path(value).is_absolute():
            raise ValueError("adjudication output_root must be absolute")
        return value

    @model_validator(mode="after")
    def document_ids_are_unique(self) -> AdjudicationConfig:
        values = tuple(row.document_id for row in self.items)
        if len(values) != len(set(values)):
            raise ValueError("adjudication document IDs must be unique")
        return self


def load_adjudication_config(path: Path) -> AdjudicationConfig:
    try:
        return AdjudicationConfig.model_validate(load_strict_yaml_mapping(path), strict=True)
    except ValueError as error:
        raise AdjudicationError(f"invalid adjudication config: {path}: {error}") from error


def _read_pinned(path_value: str, expected_sha256: str, *, context: str) -> bytes:
    path = Path(path_value).resolve(strict=True)
    payload = read_regular_file_bytes(path)
    actual = sha256_bytes(payload)
    if actual != expected_sha256:
        raise AdjudicationError(
            f"{context} SHA-256 differs: expected {expected_sha256}, found {actual}: {path}"
        )
    return payload


def _day_first_date(raw_value: str) -> date:
    found = _NUMERIC_DATE.fullmatch(raw_value.strip())
    if found is None:
        raise AdjudicationError(
            f"date correction raw value is not an ambiguous numeric date: {raw_value}"
        )
    day = int(found.group("first"))
    month = int(found.group("second"))
    if day > 12 or month > 12:
        raise AdjudicationError(
            f"date correction raw value is not ambiguous under day/month order: {raw_value}"
        )
    year = int(found.group("year"))
    if year < 100:
        year += 2000 if year <= 68 else 1900
    try:
        return date(year, month, day)
    except ValueError as error:
        raise AdjudicationError(f"day-first date is invalid: {raw_value}") from error


def _correct_date(
    annotation: BillOfLadingDualCargoAnnotation,
    correction: DateCorrectionConfig,
) -> BillOfLadingDualCargoAnnotation:
    expected = _day_first_date(correction.raw_value)
    if correction.normalized_value != expected:
        raise AdjudicationError(
            f"configured normalized date differs from day-first policy: "
            f"{correction.normalized_value.isoformat()} != {expected.isoformat()}"
        )
    evidence = tuple(
        row for row in annotation.evidence if row.targetPath == correction.target_path
    )
    if not evidence or not any(
        raw.rawValue == correction.raw_value
        for row in evidence
        for raw in row.rawOcrEvidence
    ):
        raise AdjudicationError(
            f"date correction lacks exact existing evidence at {correction.target_path}: "
            f"{correction.raw_value}"
        )
    field_name = correction.target_path.rsplit(".", 1)[1]
    normal_patch = annotation.normalLabel.documentPatch.model_copy(
        update={field_name: correction.normalized_value}
    )
    relation_patch = annotation.relationExplicitLabel.documentPatch.model_copy(
        update={field_name: correction.normalized_value}
    )
    return annotation.model_copy(
        update={
            "normalLabel": annotation.normalLabel.model_copy(
                update={"documentPatch": normal_patch}
            ),
            "relationExplicitLabel": annotation.relationExplicitLabel.model_copy(
                update={"documentPatch": relation_patch}
            ),
            "warnings": tuple(
                warning
                for warning in annotation.warnings
                if not (
                    warning.code == "ambiguous_ocr_candidates"
                    and warning.targetPath == correction.target_path
                )
            ),
        }
    )


def _pointer_tokens(pointer: str) -> tuple[str, ...]:
    return tuple(
        token.replace("~1", "/").replace("~0", "~")
        for token in pointer[1:].split("/")
    )


def _replace_pointer_value(
    value: JsonValue,
    correction: TargetCorrectionConfig,
) -> JsonValue:
    """Replace one existing JSON value after proving the pinned source value."""

    result = copy.deepcopy(value)
    tokens = _pointer_tokens(correction.target_path)
    current: JsonValue = result
    for token in tokens[:-1]:
        if isinstance(current, dict):
            if token not in current:
                raise AdjudicationError(
                    f"target correction path does not exist: {correction.target_path}"
                )
            current = current[token]
        elif isinstance(current, list):
            if not token.isdigit() or (len(token) > 1 and token.startswith("0")):
                raise AdjudicationError(
                    f"target correction list index is invalid: {correction.target_path}"
                )
            index = int(token)
            if index >= len(current):
                raise AdjudicationError(
                    f"target correction list index is out of range: {correction.target_path}"
                )
            current = current[index]
        else:
            raise AdjudicationError(
                f"target correction traverses a scalar: {correction.target_path}"
            )
    final = tokens[-1]
    if isinstance(current, dict):
        if final not in current:
            raise AdjudicationError(
                f"target correction path does not exist: {correction.target_path}"
            )
        existing = current[final]
        if canonical_json_bytes(existing) != canonical_json_bytes(correction.expected_value):
            raise AdjudicationError(
                f"target correction expected value differs: {correction.target_path}"
            )
        current[final] = copy.deepcopy(correction.corrected_value)
    elif isinstance(current, list):
        if not final.isdigit() or (len(final) > 1 and final.startswith("0")):
            raise AdjudicationError(
                f"target correction list index is invalid: {correction.target_path}"
            )
        index = int(final)
        if index >= len(current):
            raise AdjudicationError(
                f"target correction list index is out of range: {correction.target_path}"
            )
        if canonical_json_bytes(current[index]) != canonical_json_bytes(
            correction.expected_value
        ):
            raise AdjudicationError(
                f"target correction expected value differs: {correction.target_path}"
            )
        current[index] = copy.deepcopy(correction.corrected_value)
    else:
        raise AdjudicationError(
            f"target correction parent is a scalar: {correction.target_path}"
        )
    return result


def _apply_relation_corrections(
    relation: AgentBillOfLadingRelationExplicitLabel,
    corrections: Sequence[TargetCorrectionConfig],
) -> AgentBillOfLadingRelationExplicitLabel:
    payload: JsonValue = relation.model_dump(mode="json")
    for correction in corrections:
        payload = _replace_pointer_value(payload, correction)
    try:
        return AgentBillOfLadingRelationExplicitLabel.model_validate_json(
            canonical_json_bytes(payload), strict=True
        )
    except ValueError as error:
        raise AdjudicationError(
            "target corrections produced an invalid relation-explicit label"
        ) from error


def _correct_relation_label(
    annotation: BillOfLadingDualCargoAnnotation,
    corrections: Sequence[TargetCorrectionConfig],
) -> AgentBillOfLadingRelationExplicitLabel:
    return _apply_relation_corrections(
        AgentBillOfLadingRelationExplicitLabel.from_canonical(
            annotation.relationExplicitLabel
        ),
        corrections,
    )


def _warnings_after_updates(
    warnings: Sequence[LabelWarning],
    removals: Sequence[WarningRemovalConfig],
    additions: Sequence[WarningRemovalConfig],
) -> tuple[LabelWarning, ...]:
    retained = list(warnings)
    for configured_removal in removals:
        removal = configured_removal.as_warning()
        payload = canonical_json_bytes(removal.model_dump(mode="json"))
        try:
            index = next(
                index
                for index, warning in enumerate(retained)
                if canonical_json_bytes(warning.model_dump(mode="json")) == payload
            )
        except StopIteration as error:
            raise AdjudicationError(
                "configured warning removal differs from the pinned candidate"
            ) from error
        retained.pop(index)
    existing = {
        canonical_json_bytes(warning.model_dump(mode="json")) for warning in retained
    }
    for configured_addition in additions:
        addition = configured_addition.as_warning()
        payload = canonical_json_bytes(addition.model_dump(mode="json"))
        if payload in existing:
            raise AdjudicationError(
                "configured warning addition already exists in the pinned candidate"
            )
        retained.append(addition)
        existing.add(payload)
    return tuple(retained)


def _update_warnings(
    annotation: BillOfLadingDualCargoAnnotation,
    removals: Sequence[WarningRemovalConfig],
    additions: Sequence[WarningRemovalConfig],
) -> BillOfLadingDualCargoAnnotation:
    return annotation.model_copy(
        update={
            "warnings": _warnings_after_updates(
                annotation.warnings, removals, additions
            )
        }
    )


def _adjudicate_item(
    item: AdjudicationItemConfig,
) -> tuple[BillOfLadingDualCargoAnnotation, AgentWorkItem]:
    work_item_payload = _read_pinned(
        item.work_item_path,
        item.work_item_sha256,
        context="work item",
    )
    candidate_payload = _read_pinned(
        item.candidate_path,
        item.candidate_sha256,
        context="candidate",
    )
    try:
        work_item = AgentWorkItem.model_validate_json(work_item_payload, strict=True)
        if item.candidate_kind == "compact_draft":
            if item.pdf_grouping_used is None:
                raise AdjudicationError(
                    "compact-draft adjudication is missing pdf_grouping_used"
                )
            compact_draft = CompactAnnotationDraft.model_validate_json(
                candidate_payload, strict=True
            )
            corrected_relation = _apply_relation_corrections(
                compact_draft.relationExplicitLabel,
                item.target_corrections,
            )
            warnings = _warnings_after_updates(
                compact_draft.warnings,
                item.warning_removals,
                item.warning_additions,
            )
            if item.rebuild is not None:
                compact_draft = CompactAnnotationDraft.model_validate(
                    {
                        "decision": "annotation",
                        "documentType": (
                            item.rebuild.document_type or compact_draft.documentType
                        ),
                        "relationExplicitLabel": corrected_relation,
                        "warnings": warnings,
                        "decisionNotes": tuple(item.rebuild.decision_notes),
                    },
                    strict=True,
                )
            else:
                compact_draft = compact_draft.model_copy(
                    update={
                        "relationExplicitLabel": corrected_relation,
                        "warnings": warnings,
                    }
                )
            annotation = build_compact_annotation(
                work_item,
                compact_draft,
                pdf_grouping_used=item.pdf_grouping_used,
            )
        else:
            annotation = BillOfLadingDualCargoAnnotation.model_validate_json(
                candidate_payload, strict=True
            )
    except ValueError as error:
        raise AdjudicationError(
            f"adjudication input failed strict schema validation: {item.document_id}"
        ) from error
    if work_item.source.documentId != item.document_id:
        raise AdjudicationError("work-item document ID differs from adjudication item")
    if annotation.source != work_item.source:
        raise AdjudicationError("candidate source provenance differs from the pinned work item")
    if annotation.reviewStatus not in {"candidate", "validated"}:
        raise AdjudicationError("only candidate or validated annotations may be adjudicated")
    for support in item.supporting_artifacts:
        _read_pinned(support.path, support.sha256, context="supporting artifact")
    if item.candidate_kind == "annotation":
        annotation = _update_warnings(
            annotation, item.warning_removals, item.warning_additions
        )
        corrected_relation = _correct_relation_label(annotation, item.target_corrections)
        if item.rebuild is not None:
            draft = CompactAnnotationDraft.model_validate(
                {
                    "decision": "annotation",
                    "documentType": item.rebuild.document_type or annotation.documentType,
                    "relationExplicitLabel": corrected_relation,
                    "warnings": annotation.warnings,
                    "decisionNotes": tuple(item.rebuild.decision_notes),
                },
                strict=True,
            )
            annotation = build_compact_annotation(
                work_item,
                draft,
                pdf_grouping_used=any(
                    row.pdfUse == "grouping_only" for row in annotation.relationEvidence
                ),
            )
    for correction in item.date_corrections:
        annotation = _correct_date(annotation, correction)
    annotation = annotation.model_copy(
        update={
            "reviewStatus": "validated",
            "reviewNotes": (*annotation.reviewNotes, *item.adjudication_notes),
        }
    )
    payload = canonical_json_bytes(annotation.model_dump(mode="json"))
    try:
        final = BillOfLadingDualCargoAnnotation.model_validate_json(payload, strict=True)
    except ValueError as error:
        raise AdjudicationError("adjudicated annotation failed strict schema validation") from error
    validate_annotation_evidence(work_item, final)
    return final, work_item


def publish_adjudication_run(config: AdjudicationConfig) -> Path:
    output_root = Path(config.output_root).resolve(strict=True)
    run_root = output_root / "runs" / config.run_id
    adjudicated = tuple((item, *_adjudicate_item(item)) for item in config.items)
    resolved = {
        "schemaVersion": 1,
        "run": {"run_id": config.run_id},
        "adjudication": config.model_dump(mode="json"),
    }
    atomic_publish_json(run_root / "resolved-config.json", resolved)
    manifest_items: list[dict[str, object]] = []
    for item, annotation, work_item in adjudicated:
        artifact_payload = canonical_json_bytes(annotation.model_dump(mode="json"))
        artifact_sha256 = sha256_bytes(artifact_payload)
        relative_artifact = f"validated/{artifact_sha256}.json"
        atomic_publish_bytes(run_root / relative_artifact, artifact_payload)
        outcome = DocumentRunOutcome.model_validate(
            {
                "schemaVersion": 1,
                "documentId": item.document_id,
                "status": "validated",
                "attempts": 1,
                "reviews": 1,
                "documentEscalations": 0,
                "finalArtifactPath": relative_artifact,
                "finalArtifactSha256": artifact_sha256,
                "callReceiptPaths": (),
                "callReceiptSha256s": (),
            },
            strict=True,
        )
        atomic_publish_bytes(
            run_root / "state" / "outcomes" / f"{item.document_id}.json",
            canonical_json_bytes(outcome.model_dump(mode="json")),
        )
        manifest_items.append(
            {
                "documentId": item.document_id,
                "workItemPath": item.work_item_path,
                "workItemSha256": item.work_item_sha256,
                "sourceCandidatePath": item.candidate_path,
                "sourceCandidateSha256": item.candidate_sha256,
                "sourceCandidateKind": item.candidate_kind,
                "sourcePdfGroupingUsed": item.pdf_grouping_used,
                "dateCorrections": [row.model_dump(mode="json") for row in item.date_corrections],
                "targetCorrections": [
                    row.model_dump(mode="json") for row in item.target_corrections
                ],
                "warningRemovals": [
                    row.model_dump(mode="json") for row in item.warning_removals
                ],
                "warningAdditions": [
                    row.model_dump(mode="json") for row in item.warning_additions
                ],
                "rebuild": item.rebuild.model_dump(mode="json") if item.rebuild else None,
                "supportingArtifacts": [
                    row.model_dump(mode="json") for row in item.supporting_artifacts
                ],
                "finalArtifactPath": relative_artifact,
                "finalArtifactSha256": artifact_sha256,
                "joinedRawTextSha256": work_item.source.joinedRawTextSha256,
            }
        )
    manifest = {
        "schemaVersion": 1,
        "run_id": config.run_id,
        "kind": "overseer_adjudication",
        "validatedDocuments": len(manifest_items),
        "items": manifest_items,
    }
    manifest_path = run_root / "manifest.json"
    atomic_publish_json(manifest_path, manifest)
    return manifest_path


class _ArgumentParser(argparse.ArgumentParser):
    def error(self, message: str) -> NoReturn:
        raise AdjudicationError(message)


def main(argv: Sequence[str] | None = None) -> int:
    parser = _ArgumentParser(description="Publish an immutable overseer-adjudication run.")
    parser.add_argument("--config", required=True, type=Path)
    arguments = parser.parse_args(argv)
    config = load_adjudication_config(arguments.config)
    manifest = publish_adjudication_run(config)
    print(
        canonical_json_bytes(
            {"command": "adjudicate", "manifest_path": str(manifest), "status": "complete"}
        ).decode("utf-8")
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
