"""Small review contracts and section views of the single V7 extraction schema."""

from __future__ import annotations

from copy import deepcopy
from typing import Any, Literal

from pydantic import Field, create_model, model_validator

from document_ocr.label_schemas.bill_of_lading_v7 import BillOfLadingDocumentPatchV7
from document_ocr.label_schemas.common import LabelSchemaModel

Section = Literal["parties", "route_transport", "metadata_freight", "equipment", "cargo"]
SECTION_FIELDS: dict[Section, tuple[str, ...]] = {
    "parties": ("parties",),
    "route_transport": ("route", "transport"),
    "metadata_freight": (
        "billOfLadingNumber",
        "originalBillOfLadingNumber",
        "masterBillOfLadingNumber",
        "issueDate",
        "shippedOnBoardDate",
        "negotiability",
        "placeOfIssue",
        "freight",
        "forwardingAndExportReferences",
    ),
    "equipment": ("containerInformation",),
    "cargo": ("goodsItemDetails",),
}


class SectionProjectionV7(BillOfLadingDocumentPatchV7):
    """Internal cross-section validation view, never a publishable extraction.

    A cargo/party repair must not depend on metadata being in its validation
    scope. Complete extraction and metadata-section outputs still require the
    actual negotiability value.
    """

    negotiability: Literal["negotiable", "non_negotiable"] | None = None


# Responsibility checklists, not a second field schema. Shared by review and correction.
SECTION_PRIORITIES: dict[Section, str] = {
    "parties": (
        "Check each role's identity, complete postal block and all owned contacts, including "
        "continuations where the role definition permits them. Consignee details are "
        "restricted to its own printed block, not the goods area. Separate postal zones "
        "from company names and "
        "tax data from addresses. Look for competing addresses rather than concatenating "
        "them. Distinguish owned contact homepages from links to specific legal/help content."
        " When role captions conflict, inspect the printed caption and its connected block; "
        "OCR can misread headings. Follow continuation markers before assigning ownership."
        " Detached headings and repeated identities require layout review: assign roles "
        "from labeled spatial blocks, not proximity in the OCR reading order. PDF may "
        "clarify ownership of an OCR occurrence, but cannot restore a missing role's "
        "occurrence by copying a repeated identity retained only under another role."
    ),
    "route_transport": (
        "Check the role of every location, vessel and voyage against its heading/context, "
        "not OCR reading order or a plausible itinerary. Extract locality names and "
        "normalize explicitly supported countries as their field definitions require. "
        "Check country evidence for that location, not a nearby party. Distinguish "
        "receipt/loading, discharge/delivery and freight-payment places. Request layout "
        "to establish roles when headings and values are detached, including when "
        "checking an apparently correct candidate."
        " Trace matching continuation/footnote markers across the page; their linked content "
        "retains its semantic role even when printed inside a different column."
    ),
    "metadata_freight": (
        "Distinguish the carrier's B/L identifier from booking, customs, uploaded-file and "
        "electronic-platform references. Check date roles and selected freight terms, not "
        "empty form alternatives. Always include negotiability: negotiable for the actual "
        "consignee's order instruction, non_negotiable for a readable named consignee "
        "without it, null when the consignee instruction is unavailable in OCR. "
        "Check negotiability against the actual consignee "
        "instruction in OCR, including when the candidate omits it or the party name "
        "has already lost its TO ORDER preamble. A populated Consigned to order of "
        "field is an instruction, not an unselected conditional caption; follow the "
        "schema's precedence over copy/document titles. Review explicit references "
        "and their owner/type; identifiers with dedicated fields are not export "
        "references. Formatting variants are not different facts. When correcting "
        "ownership, retain a supported fact in its proper field as part of the same correction."
    ),
    "equipment": (
        "Inventory distinct containers and every owned seal, including joined or repeated "
        "rows. Check identifier boundaries, type/size and temperature ownership. Format "
        "validation cannot decide an ambiguous identifier split; do not invent characters "
        "or discard an OCR-supported identifier just to obtain valid output. Equipment "
        "whose ID is absent from OCR cannot form an ID-required target; its omission "
        "is valid, not an unresolved missing field. Use goods only as "
        "read-only association context."
        " Use the complete shipment-owned specification before applying equipment defaults; "
        "tariff examples and cargo/package dimensions are not container specifications. "
        "For an opaque or apparently corrupted equipment code, compare owned repeated "
        "declarations and request PDF layout/source clarification when needed."
        " A continuation marker can refer back to transport or another block; column position "
        "alone does not turn its text into an equipment identifier."
    ),
    "cargo": (
        "Review each group's complete product wording, including specifications and printed "
        "per-package capacity, plus codes, markings, DG, origin and handling. "
        "Grouping, shipment counts/masses/volumes and container allocations belong exclusively "
        "to the cargo-accounting reviewer, which resolves their shared row ownership. "
        "Inspect existing lists before "
        "calling a handling instruction missing. Layout establishes associations; "
        "extract product wording even when printed in a package-label block. Field meaning "
        "distinguishes product identity, markings, quantities, references and destinations."
    ),
}


def _section_model(section: Section, names: tuple[str, ...]) -> type[LabelSchemaModel]:
    fields: dict[str, Any] = {
        name: (field.annotation, deepcopy(field))
        for name in names
        for field in (BillOfLadingDocumentPatchV7.model_fields[name],)
    }
    return create_model(
        f"{section.title().replace('_', '')}Section",
        __base__=LabelSchemaModel,
        __doc__=f"Complete {section.replace('_', ' ')} extraction section; absent facts stay null.",
        **fields,
    )


SECTION_MODELS = {
    section: _section_model(section, names) for section, names in SECTION_FIELDS.items()
}


class ReviewFinding(LabelSchemaModel):
    """One actionable semantic defect or ambiguity, not an inventory of correct scalar values."""

    field: str = Field(
        min_length=1,
        description=(
            "Affected field/entity, e.g. shipper.addressLine or goods item 2. For a "
            "missing item identify its OCR product/party name; JSON-pointer syntax is "
            "unnecessary."
        ),
    )
    issue: Literal[
        "missing",
        "unsupported",
        "wrong_owner",
        "wrong_value",
        "boundary",
        "normalization",
        "grouping",
        "allocation",
        "ambiguous",
    ] = Field(
        description=(
            "Class of the actual problem; use ambiguous only when the source or target "
            "policy cannot decide it."
        )
    )
    explanation: str = Field(
        min_length=1,
        description="Short concrete explanation of the mismatch or competing interpretations.",
    )
    suggestedCorrection: str | None = Field(
        default=None,
        description=(
            "Specific source-supported remedy, or null if genuinely unresolved. Do not "
            "request changes merely to stylistic preference. An ambiguous source can still "
            "have a safe remedy: omit an assumed value when the field requires absence."
        ),
    )
    ocrExcerpt: str | None = Field(
        default=None,
        description=(
            "Optional short verbatim OCR excerpt supporting a disputed correction. No "
            "line coordinates or evidence for already-correct values."
        ),
    )
    reassignTo: Section | None = Field(
        default=None,
        description=(
            "Destination section when a supported fact belongs outside your assigned "
            "section; use wrong_owner and name its destination field/entity in "
            "suggestedCorrection. A relocation preserves the fact, unlike removal of "
            "unsupported information. Null for a correction wholly within this section."
        ),
    )

    @model_validator(mode="after")
    def relocation_has_destination(self) -> ReviewFinding:
        if self.reassignTo is not None and (
            self.issue != "wrong_owner" or not self.suggestedCorrection
        ):
            raise ValueError("relocation requires wrong_owner and a destination remedy")
        return self


class SectionReview(LabelSchemaModel):
    """Support, ownership and completeness review of exactly one extraction section."""

    status: Literal["pass", "corrections_needed", "unresolved"] = Field(
        description=(
            "pass means the section is correct and complete under its contract, "
            "including legitimate absence. corrections_needed requires actionable "
            "defects; unresolved requires competing plausible values for an applicable "
            "target fact. Correctly omitted unowned or out-of-schema text does not block."
        )
    )
    explanation: str = Field(
        min_length=1,
        description=(
            "Concise source-based justification of the verdict, including a pass. "
            "For re-review explain whether the changes resolved the defects or "
            "introduced others. State the decisive evidence/policy, not private reasoning."
        ),
    )
    findings: list[ReviewFinding] = Field(
        description=(
            "All detected problems, or an empty list for a clean section. Do not "
            "manufacture findings or repeat correct values."
        )
    )

    @model_validator(mode="after")
    def coherent_status(self) -> SectionReview:
        if (self.status == "pass") != (not self.findings):
            raise ValueError("pass requires no findings; other statuses require findings")
        if self.status == "corrections_needed" and not any(
            f.suggestedCorrection for f in self.findings
        ):
            raise ValueError("corrections_needed requires an actionable finding")
        if self.status == "unresolved" and not any(f.issue == "ambiguous" for f in self.findings):
            raise ValueError("unresolved requires an ambiguous finding")
        return self


class LayoutRequest(LabelSchemaModel):
    """Request PDF pages for layout/ownership, never to recover absent OCR values."""

    pages: list[int] = Field(
        min_length=1,
        max_length=3,
        description="One-based PDF page numbers, at most three distinct pages.",
    )
    question: str = Field(
        min_length=1,
        description="Specific ownership/layout question that OCR text alone cannot resolve.",
    )

    @model_validator(mode="after")
    def valid_pages(self) -> LayoutRequest:
        if any(p < 1 for p in self.pages) or len(set(self.pages)) != len(self.pages):
            raise ValueError("PDF pages must be positive and unique")
        return self


class CorrectionHold(LabelSchemaModel):
    """An unresolved source/policy conflict preventing a safe section correction."""

    reason: str = Field(
        min_length=1,
        description=(
            "Exact conflict preventing correction; unsupported reviewer suggestions can"
            " instead be rejected by returning the unchanged section."
        ),
    )


class CorrectionDecision(LabelSchemaModel):
    """Auditor's explicit decision on one review finding before emitting corrected values."""

    findingId: str = Field(min_length=1, description="Exact supplied finding ID.")
    changedFields: list[str] = Field(
        description=(
            "Select the permitted value paths this finding authorizes changing, including "
            "removals and direct consequences. Choices are object leaves or whole lists. "
            "A selected path can remain unchanged after adjudication; every actual change "
            "must be selected. Findings may share a path when they affect the same value "
            "or list. Reject requires an empty list."
        )
    )
    disposition: Literal["accept", "reject", "revise", "unresolved"] = Field(
        description=(
            "accept: defect and remedy are correct; reject: original is correct or "
            "suggestion unsupported; revise: defect is real but remedy needs changing; "
            "unresolved: competing source interpretations remain; apply any field-defined "
            "safe omission and retain the ambiguity in this explanation. Preserve other facts."
        )
    )
    explanation: str = Field(
        min_length=1,
        description=(
            "Short justification using the applicable field rule and decisive source "
            "fact. Explain the actual remedy when revising; no internal thought transcript."
        ),
    )


ReasoningEffort = Literal["low", "medium", "high", "xhigh"]


class DirectLabelingConfig(LabelSchemaModel):
    """Explicit settings for a bounded direct-labeling pilot; no automatic bulk selection."""

    model: str = Field(
        min_length=1, description="OpenAI Responses model identifier used for this pilot."
    )
    reasoning_effort: ReasoningEffort = Field(
        description="Reasoning effort for extraction, review and correction."
    )
    audit_reasoning_effort: ReasoningEffort | None = Field(
        default=None,
        description=(
            "Optional effort override for correction/adjudication and final re-review; "
            "initial extraction/review retain reasoning_effort. Null uses the same effort."
        ),
    )
    max_output_tokens: int = Field(
        gt=0, description="Per-request cap including reasoning and output tokens."
    )
    timeout_seconds: float = Field(
        gt=0,
        description="Network timeout; failed calls are recorded without invisible retries.",
    )
    concurrency: int = Field(
        gt=0, le=16, description="Maximum concurrent section calls in this flow."
    )
    package_registry: str = Field(
        min_length=1,
        description="Project-relative package category registry containing tokens and meanings.",
    )
    package_registry_sha256: str = Field(
        pattern=r"^[a-f0-9]{64}$",
        description="Expected registry file hash, checked before any model request.",
    )
