"""Independent semantic-review contract for OCR-conditioned KIE candidates."""

from __future__ import annotations

from typing import Annotated, Literal

from pydantic import Field, model_validator

from document_ocr.label_schemas.common import (
    CorrectionTargetPath,
    LabelSchemaModel,
    RawOcrValueEvidence,
    Sha256,
    TrimmedString,
)

ReasoningEffort = Literal["low", "medium", "high", "xhigh", "max"]
ReviewResult = Literal["pass", "fail"]


class SemanticReviewChecks(LabelSchemaModel):
    """Exhaustive reviewer gates; every gate must pass for promotion."""

    rawOcrTruthBoundary: ReviewResult
    evidenceIntegrity: ReviewResult
    semanticCompleteness: ReviewResult
    semanticCorrectness: ReviewResult
    contaminationAndRedundancy: ReviewResult
    documentUnitAndRelationships: ReviewResult


class SemanticReviewFinding(LabelSchemaModel):
    """One exact, OCR-grounded semantic problem or advisory observation."""

    severity: Literal["blocking", "advisory"]
    category: Literal[
        "truth_boundary",
        "evidence",
        "missing_field",
        "incorrect_field",
        "contamination",
        "redundancy",
        "document_unit",
        "relationship",
        "ambiguity",
        "other",
    ]
    message: TrimmedString
    targetPaths: tuple[CorrectionTargetPath, ...] = ()
    rawOcrEvidence: tuple[RawOcrValueEvidence, ...] = Field(min_length=1)
    imageUse: Literal[
        "not_used",
        "grouping_only",
        "candidate_disambiguation",
        "discrepancy_check",
    ] = "not_used"

    @model_validator(mode="after")
    def evidence_is_canonical(self) -> SemanticReviewFinding:
        if len(set(self.targetPaths)) != len(self.targetPaths):
            raise ValueError("targetPaths must be unique and source ordered")
        evidence_keys = tuple(
            (row.pageNumber, row.rawValue, row.ocrExcerpt) for row in self.rawOcrEvidence
        )
        if len(set(evidence_keys)) != len(evidence_keys):
            raise ValueError("rawOcrEvidence entries must be unique")
        if tuple(row.pageNumber for row in self.rawOcrEvidence) != tuple(
            sorted(row.pageNumber for row in self.rawOcrEvidence)
        ):
            raise ValueError("rawOcrEvidence must be ordered by pageNumber")
        return self


class BillOfLadingIndependentReview(LabelSchemaModel):
    """One fresh reviewer's immutable verdict on one exact candidate attempt."""

    reviewSchemaVersion: Literal["1.0.0"] = "1.0.0"
    taskType: Literal["bill_of_lading_kie_semantic_review"] = "bill_of_lading_kie_semantic_review"
    documentId: Annotated[str, Field(pattern=r"^doc_[0-9a-f]{64}$")]
    candidateAttempt: Annotated[int, Field(gt=0)]
    reviewNumber: Annotated[int, Field(gt=0)]
    workItemSha256: Sha256
    candidateSha256: Sha256
    reviewerModel: Literal["gpt-5.6-luna"]
    reviewerReasoningEffort: ReasoningEffort
    result: ReviewResult
    checks: SemanticReviewChecks
    findings: tuple[SemanticReviewFinding, ...] = ()
    summary: TrimmedString

    @model_validator(mode="after")
    def result_matches_checks_and_findings(self) -> BillOfLadingIndependentReview:
        check_values = tuple(self.checks.model_dump(mode="python").values())
        blocking_count = sum(row.severity == "blocking" for row in self.findings)
        if self.result == "pass":
            if any(value != "pass" for value in check_values):
                raise ValueError("a passing review requires every semantic check to pass")
            if blocking_count:
                raise ValueError("a passing review cannot contain blocking findings")
        else:
            if all(value == "pass" for value in check_values):
                raise ValueError("a failing review requires at least one failed check")
            if not blocking_count:
                raise ValueError("a failing review requires a blocking finding")
        return self
