"""Immutable records for the auxiliary table-recognition view."""

from __future__ import annotations

from typing import Annotated, Literal

from pydantic import AwareDatetime, Field, StringConstraints, model_validator

from document_ocr.label_schemas.common import LabelSchemaModel

NonEmptyString = Annotated[str, StringConstraints(strip_whitespace=True, min_length=1)]
Sha256 = Annotated[str, StringConstraints(pattern=r"^[0-9a-f]{64}$")]


class TableInputPage(LabelSchemaModel):
    schemaVersion: Literal[1]
    recordSetId: NonEmptyString
    annotationPath: NonEmptyString
    annotationSha256: Sha256
    documentId: Annotated[str, StringConstraints(pattern=r"^doc_[0-9a-f]{64}$")]
    documentPageCount: Annotated[int, Field(gt=0)]
    pageIndex: Annotated[int, Field(ge=0)]
    pageNumber: Annotated[int, Field(gt=0)]
    pageId: NonEmptyString
    baseExtractionId: NonEmptyString
    baseExtractionRunId: NonEmptyString
    baseRawOcrTextSha256: Sha256 | None
    sourceSha256: Sha256
    rasterPath: NonEmptyString
    rasterMimeType: Literal["image/png", "image/jpeg"]
    rasterSizeBytes: Annotated[int, Field(gt=0)]
    rasterSha256: Sha256
    tableViewId: NonEmptyString

    @model_validator(mode="after")
    def page_identity_is_consistent(self) -> TableInputPage:
        if self.pageNumber != self.pageIndex + 1:
            raise ValueError("pageNumber must equal pageIndex + 1")
        if self.pageNumber > self.documentPageCount:
            raise ValueError("pageNumber must not exceed documentPageCount")
        return self


class TableAttemptRecord(LabelSchemaModel):
    schemaVersion: Literal[1]
    tableViewId: NonEmptyString
    documentId: NonEmptyString
    pageId: NonEmptyString
    attemptNumber: Annotated[int, Field(gt=0)]
    startedAt: AwareDatetime
    completedAt: AwareDatetime
    durationMs: Annotated[float, Field(ge=0, allow_inf_nan=False)]
    outcome: Literal["success", "http_error", "transport_error", "invalid_response"]
    retryable: bool
    inferenceRequestId: NonEmptyString
    inferenceServerRequestId: NonEmptyString | None = None
    httpStatusCode: Annotated[int, Field(ge=100, le=599)] | None = None
    retryAfterSeconds: Annotated[float, Field(ge=0, allow_inf_nan=False)] | None = None
    errorType: NonEmptyString | None = None
    errorMessage: str | None = None

    @model_validator(mode="after")
    def timestamps_and_outcome_are_consistent(self) -> TableAttemptRecord:
        if self.completedAt < self.startedAt:
            raise ValueError("completedAt must not precede startedAt")
        if self.outcome == "success":
            if self.retryable or self.errorType is not None or self.errorMessage is not None:
                raise ValueError("successful attempts must not carry retry/error metadata")
        elif self.errorType is None or self.errorMessage is None:
            raise ValueError("failed attempts must carry errorType and errorMessage")
        return self


class TablePageResult(LabelSchemaModel):
    schemaVersion: Literal[1]
    tableViewId: NonEmptyString
    inputPageSha256: Sha256
    documentId: NonEmptyString
    documentPageCount: Annotated[int, Field(gt=0)]
    pageIndex: Annotated[int, Field(ge=0)]
    pageNumber: Annotated[int, Field(gt=0)]
    pageId: NonEmptyString
    baseExtractionId: NonEmptyString
    baseExtractionRunId: NonEmptyString
    rasterPath: NonEmptyString
    rasterSha256: Sha256
    prompt: Literal["Table Recognition:"]
    promptSha256: Sha256
    model: NonEmptyString
    modelRevision: NonEmptyString
    servedModelName: NonEmptyString
    vllmEngineVersion: NonEmptyString
    serverContractSha256: Sha256
    responseId: NonEmptyString
    inferenceRequestId: NonEmptyString
    inferenceServerRequestId: NonEmptyString | None = None
    finishReason: Literal["stop", "repetition"]
    promptTokens: Annotated[int, Field(ge=0)]
    completionTokens: Annotated[int, Field(ge=0)]
    totalTokens: Annotated[int, Field(ge=0)]
    attemptCount: Annotated[int, Field(gt=0)]
    inferenceDurationMs: Annotated[float, Field(ge=0, allow_inf_nan=False)]
    rawTableText: str
    rawTableTextSha256: Sha256
    rawResponsePath: NonEmptyString
    rawResponseSha256: Sha256
    attemptPaths: tuple[NonEmptyString, ...] = Field(min_length=1)
    attemptSha256s: tuple[Sha256, ...] = Field(min_length=1)

    @model_validator(mode="after")
    def page_tokens_and_attempts_are_consistent(self) -> TablePageResult:
        if self.pageNumber != self.pageIndex + 1:
            raise ValueError("pageNumber must equal pageIndex + 1")
        if self.pageNumber > self.documentPageCount:
            raise ValueError("pageNumber must not exceed documentPageCount")
        if self.totalTokens != self.promptTokens + self.completionTokens:
            raise ValueError("totalTokens must equal promptTokens + completionTokens")
        if len(self.attemptPaths) != len(self.attemptSha256s):
            raise ValueError("attempt paths and hashes must have the same length")
        if self.attemptCount != len(self.attemptPaths):
            raise ValueError("attemptCount must equal the number of immutable attempt records")
        return self


class TablePageCommit(LabelSchemaModel):
    schemaVersion: Literal[1]
    tableViewId: NonEmptyString
    inputPageSha256: Sha256
    resultPath: NonEmptyString
    resultSha256: Sha256


class TableFailureRecord(LabelSchemaModel):
    schemaVersion: Literal[1]
    tableViewId: NonEmptyString
    documentId: NonEmptyString
    pageId: NonEmptyString
    lastAttemptNumber: Annotated[int, Field(gt=0)]
    retryable: bool
    errorType: NonEmptyString
    errorMessage: NonEmptyString
    attemptPaths: tuple[NonEmptyString, ...] = Field(min_length=1)
    attemptSha256s: tuple[Sha256, ...] = Field(min_length=1)

    @model_validator(mode="after")
    def attempts_are_consistent(self) -> TableFailureRecord:
        if len(self.attemptPaths) != len(self.attemptSha256s):
            raise ValueError("attempt paths and hashes must have the same length")
        if self.lastAttemptNumber != len(self.attemptPaths):
            raise ValueError("lastAttemptNumber must equal the number of attempt records")
        return self


class JoinedTablePage(LabelSchemaModel):
    """One page-aligned table view embedded in a cloned training record."""

    pageIndex: Annotated[int, Field(ge=0)]
    pageNumber: Annotated[int, Field(gt=0)]
    pageId: NonEmptyString
    tableViewId: NonEmptyString
    rasterSha256: Sha256
    finishReason: Literal["stop", "repetition"]
    completionTokens: Annotated[int, Field(ge=0)]
    rawTableText: str
    rawTableTextSha256: Sha256
    resultSha256: Sha256

    @model_validator(mode="after")
    def page_number_matches_index(self) -> JoinedTablePage:
        if self.pageNumber != self.pageIndex + 1:
            raise ValueError("pageNumber must equal pageIndex + 1")
        return self


class JoinedTableDocumentView(LabelSchemaModel):
    """Complete auxiliary table view for one multi-page document."""

    schemaVersion: Literal[1]
    viewType: Literal["glm_ocr_table_recognition"]
    runId: NonEmptyString
    documentPageCount: Annotated[int, Field(gt=0)]
    joinedTableText: str
    joinedTableTextSha256: Sha256
    pages: tuple[JoinedTablePage, ...] = Field(min_length=1)

    @model_validator(mode="after")
    def pages_are_complete_and_ordered(self) -> JoinedTableDocumentView:
        indexes = [page.pageIndex for page in self.pages]
        if indexes != list(range(self.documentPageCount)):
            raise ValueError("table-view pages must be complete and page ordered")
        if len({page.pageId for page in self.pages}) != len(self.pages):
            raise ValueError("table-view pageId values must be unique")
        return self
