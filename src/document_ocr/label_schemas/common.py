"""Shared provenance and evidence models for OCR-conditioned KIE labels."""

from __future__ import annotations

import re
from typing import Annotated, Literal

from pydantic import (
    AfterValidator,
    BaseModel,
    ConfigDict,
    Field,
    StringConstraints,
    model_validator,
)


def _non_blank(value: str) -> str:
    if not value or not value.strip():
        raise ValueError("value must contain a non-whitespace character")
    return value


def _trimmed(value: str) -> str:
    if value != value.strip():
        raise ValueError("value must not contain leading or trailing whitespace")
    return value


NonEmptyString = Annotated[str, AfterValidator(_non_blank)]
TrimmedString = Annotated[str, AfterValidator(_non_blank), AfterValidator(_trimmed)]
Sha256 = Annotated[str, StringConstraints(pattern=r"^[0-9a-f]{64}$")]
TargetPath = Annotated[
    str,
    StringConstraints(
        pattern=(
            r"^documentPatch(?:\.[A-Za-z_][A-Za-z0-9_]*)+"
            r"(?:\[[0-9]+\](?:\.[A-Za-z_][A-Za-z0-9_]*)*)*$"
        )
    ),
]

_DOCUMENT_ID = re.compile(r"^doc_[0-9a-f]{64}$")


class LabelSchemaModel(BaseModel):
    """Strict immutable base for durable label artifacts."""

    model_config = ConfigDict(
        extra="forbid",
        frozen=True,
        strict=True,
        validate_default=True,
    )


class ExtractionPageReference(LabelSchemaModel):
    """Exact extraction and retained-artifact identity for one source page."""

    pageIndex: Annotated[int, Field(ge=0)]
    pageNumber: Annotated[int, Field(gt=0)]
    pageId: NonEmptyString
    extractionId: NonEmptyString
    rawOcrTextSha256: Sha256
    rawResponsePath: NonEmptyString
    rawResponseSha256: Sha256
    rasterPath: NonEmptyString
    rasterSha256: Sha256

    @model_validator(mode="after")
    def page_number_matches_index(self) -> ExtractionPageReference:
        if self.pageNumber != self.pageIndex + 1:
            raise ValueError("pageNumber must equal pageIndex + 1")
        return self


class ExtractionSourceReference(LabelSchemaModel):
    """One complete multi-page OCR input and its immutable provenance."""

    documentId: NonEmptyString
    extractionRunId: NonEmptyString
    sourceUri: NonEmptyString
    localCanonicalPath: NonEmptyString
    sourceSha256: Sha256
    documentPageCount: Annotated[int, Field(gt=0)]
    joinedRawTextSha256: Sha256
    pages: tuple[ExtractionPageReference, ...] = Field(min_length=1)

    @model_validator(mode="after")
    def pages_are_complete_and_ordered(self) -> ExtractionSourceReference:
        if not _DOCUMENT_ID.fullmatch(self.documentId):
            raise ValueError("documentId must be doc_ followed by a lowercase SHA-256")
        expected_indexes = list(range(self.documentPageCount))
        actual_indexes = [page.pageIndex for page in self.pages]
        if actual_indexes != expected_indexes:
            raise ValueError("pages must be complete and ordered by contiguous pageIndex")
        if len({page.pageId for page in self.pages}) != len(self.pages):
            raise ValueError("pageId values must be unique within a document")
        if len({page.extractionId for page in self.pages}) != len(self.pages):
            raise ValueError("extractionId values must be unique within a document")
        return self


class RawOcrValueEvidence(LabelSchemaModel):
    """One exact pre-mapping value and its surrounding raw-OCR context."""

    pageNumber: Annotated[int, Field(gt=0)]
    rawValue: NonEmptyString
    ocrExcerpt: NonEmptyString

    @model_validator(mode="after")
    def raw_value_occurs_verbatim_in_excerpt(self) -> RawOcrValueEvidence:
        if self.rawValue not in self.ocrExcerpt:
            raise ValueError("rawValue must occur verbatim within ocrExcerpt")
        return self


class FieldEvidence(LabelSchemaModel):
    """Raw-OCR evidence for exactly one emitted target leaf."""

    targetPath: TargetPath
    evidenceKind: Literal[
        "verbatim",
        "normalized",
        "contextual_code",
        "cross_page_resolution",
    ]
    rawOcrEvidence: tuple[RawOcrValueEvidence, ...] = Field(min_length=1)
    imageUse: Literal[
        "not_used",
        "grouping_only",
        "candidate_disambiguation",
        "discrepancy_check",
    ] = "not_used"
    normalizationRule: TrimmedString | None = None
    note: NonEmptyString | None = None

    @model_validator(mode="after")
    def evidence_is_canonical(self) -> FieldEvidence:
        raw_page_numbers = tuple(item.pageNumber for item in self.rawOcrEvidence)
        if raw_page_numbers != tuple(sorted(raw_page_numbers)):
            raise ValueError("rawOcrEvidence must be ordered by pageNumber")
        raw_items = tuple(
            (item.pageNumber, item.rawValue, item.ocrExcerpt) for item in self.rawOcrEvidence
        )
        if len(raw_items) != len(set(raw_items)):
            raise ValueError("rawOcrEvidence entries must be unique")
        if self.evidenceKind == "verbatim" and self.normalizationRule is not None:
            raise ValueError("verbatim evidence must not declare a normalizationRule")
        if self.evidenceKind != "verbatim" and self.normalizationRule is None:
            raise ValueError("non-verbatim evidence requires a normalizationRule")
        return self


class LabelWarning(LabelSchemaModel):
    """A surfaced ambiguity or omission; warnings never alter the training target."""

    code: Literal[
        "aggregate_not_allocated",
        "ambiguous_ocr_candidates",
        "image_only_value_omitted",
        "invalid_identifier_omitted",
        "schema_cannot_represent",
        "unsupported_or_unclear_code",
        "other",
    ]
    message: NonEmptyString
    pageNumbers: tuple[Annotated[int, Field(gt=0)], ...] = ()
    targetPath: TargetPath | None = None

    @model_validator(mode="after")
    def warning_pages_are_canonical(self) -> LabelWarning:
        if tuple(sorted(set(self.pageNumbers))) != self.pageNumbers:
            raise ValueError("warning pageNumbers must be unique and sorted")
        return self
