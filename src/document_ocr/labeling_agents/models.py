"""Typed model decisions and immutable PydanticAI run receipts."""

from __future__ import annotations

from decimal import Decimal
from typing import Annotated, Literal

from pydantic import AwareDatetime, Field, JsonValue, model_validator

from document_ocr.label_schemas.bill_of_lading import BillOfLadingLabel
from document_ocr.label_schemas.bill_of_lading_agent_v4 import (
    AgentBillOfLadingRelationExplicitLabel,
)
from document_ocr.label_schemas.bill_of_lading_v3 import (
    BillOfLadingRelationExplicitLabel,
    CargoRelationEvidence,
    validate_dual_cargo_consistency,
)
from document_ocr.label_schemas.common import (
    CorrectionTargetPath,
    FieldEvidence,
    LabelSchemaModel,
    LabelWarning,
    NonEmptyString,
    RawOcrAnchor,
    RawOcrValueEvidence,
    Sha256,
    TargetPath,
    TrimmedString,
)
from document_ocr.label_schemas.semantic_review import (
    SemanticReviewChecks,
    SemanticReviewFinding,
)

PdfConstructionMethod = Literal[
    "pypdf_strict",
    "pypdf_strict_empty_password",
    "pypdf_recovery",
    "pypdf_recovery_empty_password",
]


class AnnotationDraft(LabelSchemaModel):
    decision: Literal["annotation"]
    documentType: Literal["bill_of_lading", "sea_waybill"]
    normalLabel: BillOfLadingLabel
    relationExplicitLabel: BillOfLadingRelationExplicitLabel
    evidence: tuple[FieldEvidence, ...] = Field(min_length=1)
    relationEvidence: tuple[CargoRelationEvidence, ...] = ()
    warnings: tuple[LabelWarning, ...] = ()
    decisionNotes: tuple[TrimmedString, ...] = Field(min_length=1, max_length=12)

    @model_validator(mode="after")
    def views_are_consistent_and_categories_remain_unmapped(self) -> AnnotationDraft:
        validate_dual_cargo_consistency(self.normalLabel, self.relationExplicitLabel)
        patch = self.relationExplicitLabel.documentPatch
        if any(row.typeCategory is not None for row in patch.containers or ()):
            raise ValueError(
                "agent drafts must retain printed container types; category mapping is downstream"
            )
        if any(row.typeCategory is not None for row in patch.cargoPackages or ()):
            raise ValueError(
                "agent drafts must retain printed package types; category mapping is downstream"
            )
        expected = tuple((row.groupId, row.coverage) for row in patch.cargoAllocationGroups or ())
        actual = tuple((row.groupId, row.coverage) for row in self.relationEvidence)
        if actual != expected:
            raise ValueError(
                "relationEvidence must exactly cover source-ordered cargo allocation groups"
            )
        return self


class CompactAnnotationDraft(LabelSchemaModel):
    """Single model-generated source of truth for the optimized flow."""

    decision: Literal["annotation"]
    documentType: Literal["bill_of_lading", "sea_waybill"]
    relationExplicitLabel: AgentBillOfLadingRelationExplicitLabel
    warnings: tuple[LabelWarning, ...] = ()
    decisionNotes: tuple[TrimmedString, ...] = Field(min_length=1, max_length=3)


class CompactCorrectionEnvelope(LabelSchemaModel):
    """Exact-path correction values retained from one constrained model response."""

    corrections: dict[CorrectionTargetPath, JsonValue] = Field(min_length=1)


class CompactExclusionDraft(LabelSchemaModel):
    """Provider-facing exclusion with deterministic OCR-context hydration."""

    decision: Literal["exclusion"]
    reason: Literal[
        "multiple_transport_documents",
        "not_bill_of_lading_or_sea_waybill",
        "insufficient_ocr",
        "non_latin_text",
    ]
    rawOcrEvidence: tuple[RawOcrAnchor, ...] = Field(min_length=1)
    reviewNotes: tuple[TrimmedString, ...] = Field(min_length=1, max_length=3)


class ExclusionDraft(LabelSchemaModel):
    decision: Literal["exclusion"]
    reason: Literal[
        "multiple_transport_documents",
        "not_bill_of_lading_or_sea_waybill",
        "insufficient_ocr",
        "non_latin_text",
    ]
    rawOcrEvidence: tuple[RawOcrValueEvidence, ...] = Field(min_length=1)
    reviewNotes: tuple[TrimmedString, ...] = Field(min_length=1, max_length=12)


class DocumentAssistanceRequest(LabelSchemaModel):
    decision: Literal["document_required"]
    pageNumbers: tuple[Annotated[int, Field(gt=0)], ...] = Field(min_length=1)
    ambiguity: TrimmedString
    affectedTargetPaths: tuple[TargetPath, ...] = ()
    rawOcrEvidence: tuple[RawOcrValueEvidence, ...] = Field(min_length=1)

    @model_validator(mode="after")
    def pages_are_unique_and_ordered(self) -> DocumentAssistanceRequest:
        if tuple(sorted(set(self.pageNumbers))) != self.pageNumbers:
            raise ValueError("pageNumbers must be unique and sorted")
        return self


class CompactDocumentAssistanceRequest(LabelSchemaModel):
    """Provider-facing PDF request; exact excerpts are supplied locally."""

    decision: Literal["document_required"]
    pageNumbers: tuple[Annotated[int, Field(gt=0)], ...] = Field(min_length=1)
    ambiguity: TrimmedString
    affectedTargetPaths: tuple[TargetPath, ...] = ()
    rawOcrEvidence: tuple[RawOcrAnchor, ...] = Field(min_length=1)


ExtractionDecision = Annotated[
    AnnotationDraft | ExclusionDraft | DocumentAssistanceRequest,
    Field(discriminator="decision"),
]

CompactExtractionDecision = Annotated[
    CompactAnnotationDraft | CompactExclusionDraft | CompactDocumentAssistanceRequest,
    Field(discriminator="decision"),
]


class ReviewDraft(LabelSchemaModel):
    decision: Literal["review"]
    result: Literal["pass", "fail"]
    checks: SemanticReviewChecks
    findings: tuple[SemanticReviewFinding, ...] = ()
    summary: TrimmedString

    @model_validator(mode="after")
    def verdict_is_consistent(self) -> ReviewDraft:
        checks = tuple(self.checks.model_dump(mode="python").values())
        blocking = sum(row.severity == "blocking" for row in self.findings)
        if self.result == "pass":
            if any(value != "pass" for value in checks) or blocking:
                raise ValueError("passing review requires all checks and no blocking finding")
        elif all(value == "pass" for value in checks) or not blocking:
            raise ValueError("failing review requires a failed check and blocking finding")
        return self


class CompactSemanticReviewFinding(LabelSchemaModel):
    """Provider-facing finding whose OCR excerpts are hydrated locally."""

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
    rawOcrEvidence: tuple[RawOcrAnchor, ...] = Field(min_length=1)
    imageUse: Literal[
        "not_used",
        "grouping_only",
        "candidate_disambiguation",
        "discrepancy_check",
    ] = "not_used"


class CompactReviewWireDraft(LabelSchemaModel):
    """Provider-facing grounded findings; verdict/checks are deterministic."""

    decision: Literal["review"]
    findings: tuple[CompactSemanticReviewFinding, ...] = ()
    summary: TrimmedString


class CompactReviewDraft(LabelSchemaModel):
    """Hydrated findings consumed by the durable review state machine."""

    decision: Literal["review"]
    findings: tuple[SemanticReviewFinding, ...] = ()
    summary: TrimmedString


class ReviewDocumentAssistanceRequest(LabelSchemaModel):
    decision: Literal["document_required"]
    pageNumbers: tuple[Annotated[int, Field(gt=0)], ...] = Field(min_length=1)
    ambiguity: TrimmedString
    affectedTargetPaths: tuple[TargetPath, ...] = ()
    rawOcrEvidence: tuple[RawOcrValueEvidence, ...] = Field(min_length=1)

    @model_validator(mode="after")
    def pages_are_unique_and_ordered(self) -> ReviewDocumentAssistanceRequest:
        if tuple(sorted(set(self.pageNumbers))) != self.pageNumbers:
            raise ValueError("pageNumbers must be unique and sorted")
        return self


class CompactReviewDocumentAssistanceRequest(LabelSchemaModel):
    """Provider-facing review PDF request with local excerpt hydration."""

    decision: Literal["document_required"]
    pageNumbers: tuple[Annotated[int, Field(gt=0)], ...] = Field(min_length=1)
    ambiguity: TrimmedString
    affectedTargetPaths: tuple[TargetPath, ...] = ()
    rawOcrEvidence: tuple[RawOcrAnchor, ...] = Field(min_length=1)


ReviewDecision = Annotated[
    ReviewDraft | ReviewDocumentAssistanceRequest,
    Field(discriminator="decision"),
]

CompactReviewDecision = Annotated[
    CompactReviewWireDraft | CompactReviewDocumentAssistanceRequest,
    Field(discriminator="decision"),
]


class LayoutObservation(LabelSchemaModel):
    pageNumber: Annotated[int, Field(gt=0)]
    observationType: Literal[
        "heading_scope",
        "row_association",
        "column_association",
        "page_continuation",
        "document_boundary",
    ]
    rawOcrAnchors: tuple[RawOcrValueEvidence, ...] = Field(min_length=1)
    guidance: TrimmedString


class DocumentLayoutGuidance(LabelSchemaModel):
    decision: Literal["layout_guidance"]
    observations: tuple[LayoutObservation, ...] = Field(min_length=1)
    decisionNotes: tuple[TrimmedString, ...] = Field(min_length=1, max_length=12)


class CompactLayoutObservation(LabelSchemaModel):
    """Provider-facing layout observation with minimal exact OCR anchors."""

    pageNumber: Annotated[int, Field(gt=0)]
    observationType: Literal[
        "heading_scope",
        "row_association",
        "column_association",
        "page_continuation",
        "document_boundary",
    ]
    rawOcrAnchors: tuple[RawOcrAnchor, ...] = Field(min_length=1)
    guidance: TrimmedString


class CompactDocumentLayoutGuidance(LabelSchemaModel):
    """Provider-facing layout guidance hydrated before orchestration."""

    decision: Literal["layout_guidance"]
    observations: tuple[CompactLayoutObservation, ...] = Field(min_length=1)
    decisionNotes: tuple[TrimmedString, ...] = Field(min_length=1, max_length=3)


class ModelResponseReceipt(LabelSchemaModel):
    providerResponseId: NonEmptyString | None = None
    providerName: NonEmptyString | None = None
    modelName: NonEmptyString | None = None
    finishReason: NonEmptyString | None = None
    inputTokens: Annotated[int, Field(ge=0)]
    cacheReadTokens: Annotated[int, Field(ge=0)]
    cacheWriteTokens: Annotated[int, Field(ge=0)]
    outputTokens: Annotated[int, Field(ge=0)]
    reasoningTokens: Annotated[int, Field(ge=0)]
    costUsd: Annotated[Decimal, Field(ge=0, decimal_places=12)] | None = None

    @model_validator(mode="after")
    def cache_buckets_fit_input_total(self) -> ModelResponseReceipt:
        if self.cacheReadTokens + self.cacheWriteTokens > self.inputTokens:
            raise ValueError("cached input token buckets exceed inputTokens")
        if self.reasoningTokens > self.outputTokens:
            raise ValueError("reasoningTokens must be a subset of outputTokens")
        return self


class PdfAttachmentReceipt(LabelSchemaModel):
    sourcePdfSha256: Sha256
    sourcePageNumbers: tuple[Annotated[int, Field(gt=0)], ...] = Field(min_length=1)
    attachmentPdfSha256: Sha256
    attachmentBytes: Annotated[int, Field(gt=0)]
    mediaType: Literal["application/pdf"]
    constructionMethod: PdfConstructionMethod

    @model_validator(mode="after")
    def page_numbers_are_unique_and_ordered(self) -> PdfAttachmentReceipt:
        if tuple(sorted(set(self.sourcePageNumbers))) != self.sourcePageNumbers:
            raise ValueError("sourcePageNumbers must be unique and sorted")
        return self


class AgentCallTranscriptEvent(LabelSchemaModel):
    """One locally retained, model-visible response or repair instruction."""

    sequence: Annotated[int, Field(ge=0)]
    eventType: Literal["model_response", "validation_feedback"]
    payload: JsonValue


class AgentCallTranscript(LabelSchemaModel):
    """Reviewable provider output history for one model call.

    Initial prompts and source documents are already frozen and hash-addressed by
    the call receipt.  The transcript therefore retains every provider response
    and every validation/repair message without duplicating raw OCR or PDF bytes.
    """

    transcriptSchemaVersion: Literal[1]
    events: tuple[AgentCallTranscriptEvent, ...] = ()

    @model_validator(mode="after")
    def events_are_contiguous(self) -> AgentCallTranscript:
        if tuple(row.sequence for row in self.events) != tuple(range(len(self.events))):
            raise ValueError("transcript event sequence must be contiguous and ordered")
        return self


class AgentCallReceipt(LabelSchemaModel):
    receiptSchemaVersion: Literal[2, 3]
    callId: NonEmptyString
    documentId: Annotated[str, Field(pattern=r"^doc_[0-9a-f]{64}$")]
    stage: Literal["extract", "review", "document_layout"]
    candidateAttempt: Annotated[int, Field(gt=0)]
    providerId: NonEmptyString
    providerKind: Literal["openai_responses", "ollama_openai_chat"]
    model: NonEmptyString
    reasoningEffort: NonEmptyString | None = None
    workerId: NonEmptyString
    startedAt: AwareDatetime
    completedAt: AwareDatetime
    durationMs: Annotated[float, Field(ge=0, allow_inf_nan=False)]
    workItemSha256: Sha256
    promptSha256: Sha256
    status: Literal["success", "error"] = "success"
    outputSha256: Sha256 | None = None
    errorType: NonEmptyString | None = None
    errorMessage: NonEmptyString | None = None
    requests: Annotated[int, Field(ge=0)]
    responses: tuple[ModelResponseReceipt, ...] = ()
    inputTokens: Annotated[int, Field(ge=0)]
    cacheReadTokens: Annotated[int, Field(ge=0)]
    cacheWriteTokens: Annotated[int, Field(ge=0)]
    outputTokens: Annotated[int, Field(ge=0)]
    costUsd: Annotated[Decimal, Field(ge=0, decimal_places=12)] | None = None
    costStatus: Literal["priced", "not_applicable_local", "unavailable"]
    pdfAttachment: PdfAttachmentReceipt | None = None
    transcript: AgentCallTranscript | None = None

    @model_validator(mode="after")
    def totals_match_responses(self) -> AgentCallReceipt:
        if self.completedAt < self.startedAt:
            raise ValueError("completedAt must not precede startedAt")
        if self.requests != len(self.responses):
            raise ValueError("requests must equal the response receipt count")
        if self.status == "success":
            if (
                self.outputSha256 is None
                or self.errorType is not None
                or self.errorMessage is not None
            ):
                raise ValueError("successful calls require outputSha256 and no error fields")
            if not self.responses:
                raise ValueError("successful calls require at least one provider response")
        elif self.outputSha256 is not None or self.errorType is None or self.errorMessage is None:
            raise ValueError("failed calls require error fields and no outputSha256")
        totals = (
            sum(row.inputTokens for row in self.responses),
            sum(row.cacheReadTokens for row in self.responses),
            sum(row.cacheWriteTokens for row in self.responses),
            sum(row.outputTokens for row in self.responses),
        )
        if totals != (
            self.inputTokens,
            self.cacheReadTokens,
            self.cacheWriteTokens,
            self.outputTokens,
        ):
            raise ValueError("call token totals differ from response receipts")
        priced = [row.costUsd for row in self.responses]
        if self.costStatus == "priced":
            if not priced:
                raise ValueError("priced calls require at least one provider response")
            if any(value is None for value in priced):
                raise ValueError("priced calls require a cost for every response")
            if self.costUsd != sum((value for value in priced if value is not None), Decimal(0)):
                raise ValueError("call cost differs from response costs")
        elif self.costUsd is not None:
            raise ValueError("unpriced calls must not carry costUsd")
        if self.providerKind == "openai_responses":
            if self.costStatus not in {"priced", "unavailable"}:
                raise ValueError("OpenAI calls must be priced or explicitly unavailable")
        elif self.costStatus != "not_applicable_local":
            raise ValueError("local calls must use not_applicable_local cost status")
        if self.receiptSchemaVersion == 2:
            if self.transcript is not None:
                raise ValueError("receipt schema v2 cannot carry a call transcript")
        elif self.transcript is None:
            raise ValueError("receipt schema v3 requires a call transcript")
        if self.stage == "document_layout":
            if self.pdfAttachment is None:
                raise ValueError("document-layout calls require a PDF attachment receipt")
        elif self.pdfAttachment is not None:
            raise ValueError("only document-layout calls may carry a PDF attachment receipt")
        return self


class DocumentRunOutcome(LabelSchemaModel):
    schemaVersion: Literal[1]
    documentId: Annotated[str, Field(pattern=r"^doc_[0-9a-f]{64}$")]
    status: Literal["validated", "excluded", "needs_review"]
    attempts: Annotated[int, Field(gt=0)]
    reviews: Annotated[int, Field(ge=0)]
    documentEscalations: Annotated[int, Field(ge=0)]
    finalArtifactPath: NonEmptyString
    finalArtifactSha256: Sha256
    callReceiptPaths: tuple[NonEmptyString, ...] = ()
    callReceiptSha256s: tuple[Sha256, ...] = ()

    @model_validator(mode="after")
    def receipt_paths_match_hashes(self) -> DocumentRunOutcome:
        if len(self.callReceiptPaths) != len(self.callReceiptSha256s):
            raise ValueError("call receipt paths and hashes must have the same length")
        return self


class IndependentReviewArtifact(LabelSchemaModel):
    schemaVersion: Literal[1]
    documentId: Annotated[str, Field(pattern=r"^doc_[0-9a-f]{64}$")]
    candidateAttempt: Annotated[int, Field(gt=0)]
    reviewNumber: Annotated[int, Field(gt=0)]
    workItemSha256: Sha256
    candidateSha256: Sha256
    providerId: NonEmptyString
    reviewerModel: NonEmptyString
    reviewerReasoningEffort: NonEmptyString | None = None
    result: Literal["pass", "fail"]
    checks: SemanticReviewChecks
    findings: tuple[SemanticReviewFinding, ...] = ()
    summary: TrimmedString

    @model_validator(mode="after")
    def verdict_is_consistent(self) -> IndependentReviewArtifact:
        draft = ReviewDraft.model_validate(
            {
                "decision": "review",
                "result": self.result,
                "checks": self.checks,
                "findings": self.findings,
                "summary": self.summary,
            },
            strict=True,
        )
        if draft.result != self.result:
            raise ValueError("review artifact verdict is inconsistent")
        return self


class NeedsReviewRecord(LabelSchemaModel):
    schemaVersion: Literal[1]
    documentId: Annotated[str, Field(pattern=r"^doc_[0-9a-f]{64}$")]
    workItemSha256: Sha256
    attempts: Annotated[int, Field(gt=0)]
    reviews: Annotated[int, Field(ge=0)]
    documentEscalations: Annotated[int, Field(ge=0)]
    reason: Literal[
        "candidate_attempts_exhausted",
        "correction_non_convergence",
        "document_escalation_unavailable",
        "provider_or_validation_failure",
        "training_truth_ambiguity",
    ]
    findings: tuple[TrimmedString, ...] = Field(min_length=1)
