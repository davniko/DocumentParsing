"""PydanticAI provider adapters, shared concurrency, usage, and explicit pricing."""

from __future__ import annotations

import json
import os
import re
import time
import uuid
from collections.abc import Callable, Mapping
from copy import deepcopy
from dataclasses import dataclass
from datetime import UTC, datetime
from decimal import ROUND_HALF_UP, Decimal
from pathlib import Path
from typing import Any, Literal, cast
from urllib.parse import urlsplit

import httpx
from dotenv import dotenv_values
from openai import AsyncOpenAI
from pydantic import BaseModel, ConfigDict, TypeAdapter, ValidationError
from pydantic_ai import (
    Agent,
    BinaryContent,
    ModelRetry,
    NativeOutput,
    RunContext,
    StructuredDict,
    capture_run_messages,
)
from pydantic_ai.concurrency import ConcurrencyLimiter
from pydantic_ai.exceptions import (
    AgentRunError,
    ConcurrencyLimitExceeded,
    ContentFilterError,
    ModelAPIError,
    RunCancelled,
    UnexpectedModelBehavior,
    UsageLimitExceeded,
)
from pydantic_ai.messages import (
    ModelMessagesTypeAdapter,
    ModelRequest,
    ModelResponse,
    RetryPromptPart,
)
from pydantic_ai.models import Model
from pydantic_ai.models.ollama import OllamaModel
from pydantic_ai.models.openai import OpenAIResponsesModel, OpenAIResponsesModelSettings
from pydantic_ai.providers.ollama import OllamaProvider
from pydantic_ai.providers.openai import OpenAIProvider
from pydantic_ai.settings import ModelSettings
from pydantic_ai.usage import RequestUsage, UsageLimits

from document_ocr.hashing import canonical_json_bytes, sha256_bytes
from document_ocr.labeling_agents.config import (
    AgentLabelingConfig,
    OllamaProviderConfig,
    OpenAIPricingConfig,
    OpenAIResponsesProviderConfig,
    ProviderConfig,
)
from document_ocr.labeling_agents.corrections import (
    correction_envelope_json_schema,
    merge_correction_values,
    normalize_correction_paths,
)
from document_ocr.labeling_agents.deterministic_annotation import build_compact_annotation
from document_ocr.labeling_agents.models import (
    AgentCallReceipt,
    AgentCallTranscript,
    AnnotationDraft,
    CompactAnnotationDraft,
    CompactCorrectionEnvelope,
    CompactDocumentAssistanceRequest,
    CompactDocumentLayoutGuidance,
    CompactExclusionDraft,
    CompactExtractionDecision,
    CompactReviewDecision,
    CompactReviewDocumentAssistanceRequest,
    CompactReviewDraft,
    CompactReviewWireDraft,
    DocumentAssistanceRequest,
    DocumentLayoutGuidance,
    ExclusionDraft,
    ExtractionDecision,
    ModelResponseReceipt,
    PdfAttachmentReceipt,
    ReviewDecision,
    ReviewDocumentAssistanceRequest,
    ReviewDraft,
)
from document_ocr.labeling_agents.review_policy import validate_review_policy
from document_ocr.labeling_agents.work_items import (
    AgentWorkItem,
    PdfAssistancePayload,
    WorkItemError,
    materialize_raw_ocr_evidence,
    page_texts,
)

_USD_QUANTUM = Decimal("0.000000000001")
_OLLAMA_VERSION = re.compile(r"^(?P<major>[0-9]+)\.(?P<minor>[0-9]+)(?:\.(?P<patch>[0-9]+))?")

_EXTRACTION_DECISION_ADAPTER: TypeAdapter[ExtractionDecision] = TypeAdapter(ExtractionDecision)
_REVIEW_DECISION_ADAPTER: TypeAdapter[ReviewDecision] = TypeAdapter(ReviewDecision)
_COMPACT_EXTRACTION_DECISION_ADAPTER: TypeAdapter[CompactExtractionDecision] = TypeAdapter(
    CompactExtractionDecision
)
_COMPACT_REVIEW_DECISION_ADAPTER: TypeAdapter[CompactReviewDecision] = TypeAdapter(
    CompactReviewDecision
)


@dataclass(frozen=True, slots=True)
class _ExtractionValidationContext:
    work_item: AgentWorkItem
    pdf_grouping_used: bool
    correction_base: CompactAnnotationDraft | None
    correction_paths: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class _ReviewValidationContext:
    work_item: AgentWorkItem
    candidate: Mapping[str, Any]


@dataclass(frozen=True, slots=True)
class _LayoutValidationContext:
    work_item: AgentWorkItem


_PROVIDER_FIELD_DESCRIPTIONS = {
    "decision": "Select exactly one provider-enforced decision shape for this document and stage.",
    "documentType": (
        "Semantic maritime document class: bill_of_lading for negotiable/original-surrender "
        "B/L-equivalent documents, or sea_waybill for explicit non-negotiable/express-release "
        "maritime equivalents. A completed zero-original count supports sea_waybill because no "
        "original surrender is required. Exclude non-maritime transport documents."
    ),
    "relationExplicitLabel": (
        "The sole model-authored factual label; normal targets and evidence are derived locally."
    ),
    "schemaVersion": "Emit the fixed schema version required by this contract.",
    "documentPatch": (
        "Sparse OCR-grounded document facts; omit unsupported or absent facts with null."
    ),
    "billOfLadingNumber": "Value under an explicit Bill of Lading or sea-waybill number heading.",
    "originalBillOfLadingNumber": (
        "Explicitly identified original Bill of Lading number, not a generic copy count."
    ),
    "masterBillOfLadingNumber": "Explicitly headed master Bill of Lading number.",
    "issueDate": (
        "Document issue date only, normalized to YYYY-MM-DD; never use departure, sailing, or "
        "ETD; named-month and ordinal dates are unambiguous; whenever both leading numeric "
        "components are 12 or below, interpret the printed token as day/month/year even if another "
        "date in the document uses a different order; never use geographic locality to choose it."
    ),
    "shippedOnBoardDate": (
        "Explicit shipped/on-board date normalized to YYYY-MM-DD; never use departure, sailing, "
        "or ETD; whenever both leading numeric components are 12 or below, interpret the printed "
        "token as day/month/year even if another date in the document uses a different order; "
        "never use geographic locality to choose it."
    ),
    "negotiability": (
        "Use negotiable for an order consignee or other explicit negotiable basis. Use "
        "non_negotiable for a completed named, non-order consignee, explicit non-negotiable/"
        "sea-waybill/express-release wording, or a completed zero-original count. Generic "
        "original-surrender/title boilerplate does not override a named non-order consignee."
    ),
    "placeOfIssue": "Printed place of issue, keeping locality and country text as written.",
    "route": (
        "Only locations under their exact receipt/loading/transshipment/discharge/delivery/final-"
        "destination route roles; keep printed locality/country text rather than ISO or UN/LOCODE "
        "codes and never infer a route role from sequence."
    ),
    "placeOfReceipt": "Location explicitly headed as place of receipt, not port of loading.",
    "portOfLoading": "Sea port explicitly headed as port of loading, not place of receipt.",
    "transshipmentPort": (
        "Port explicitly headed as transshipment/transhipment port; never infer from itinerary."
    ),
    "portOfDischarge": "Sea port explicitly headed as port of discharge, not place of delivery.",
    "placeOfDelivery": "Location explicitly headed as place of delivery, not port of discharge.",
    "finalDestination": (
        "Location explicitly headed as final destination; never infer it from delivery sequence."
    ),
    "transport": "Explicit vessel, voyage, IMO, and flag facts, with vessel and voyage split.",
    "freight": (
        "Explicit supported freight payment facts only; do not map operating terms or Incoterms."
    ),
    "paymentPlace": (
        "Printed named freight-payment locality only; generic Origin or Destination is not a "
        "location value."
    ),
    "parties": (
        "Role-headed parties and their contacts. A blank role heading stays null; never infer a "
        "role from nearby prose, layout sequence, signing language, or an unscoped VIA line."
    ),
    "shipper": "Party explicitly scoped by the Shipper/Exporter role heading.",
    "consignee": "Party explicitly scoped by the Consignee role heading.",
    "notifyParties": (
        "Every source-ordered party explicitly scoped by a Notify Party role; sameAs is allowed "
        "only for explicit SAME AS SHIPPER/CONSIGNEE wording and a present referenced party."
    ),
    "carrier": (
        "Carrier explicitly named by a Carrier role or an X AS AGENT FOR Y THE CARRIER clause; in "
        "that clause Y is the carrier."
    ),
    "forwardingAgent": (
        "Party explicitly scoped by a Forwarding Agent role heading; a blank heading or unscoped "
        "VIA party does not populate this role."
    ),
    "deliveryAgent": (
        "Party explicitly scoped by Delivery Agent, Destination Agent, Discharge Port Agent, "
        "Application for Delivery Must Be Made To, For Delivery Please Contact, or Shipping "
        "Agency at Port of Discharge wording; never infer from a notify party."
    ),
    "consolidator": "Party explicitly scoped by a Consolidator role heading.",
    "containers": (
        "Source-ordered valid printed ISO container identifiers and their directly associated "
        "facts."
    ),
    "forwardingAndExportReferences": (
        "Value-only explicit forwarding/export or forwarding-agent reference, AES, ITN, CAED, "
        "shipping-bill, ED-number, invoice, or P/I/proforma-invoice values. A generic REF line is "
        "eligible only when the nearest governing party section is Forwarding Agent, even across "
        "a visual blank line. Exclude labels and explicit empty placeholders such as NO REF, "
        "N/A, NONE, and NIL; blank "
        "adjacent headings such as 'Export references Svc Contract'; booking/shipper references; "
        "and ERN, ACID, tax/VAT/CNPJ, customs, exporter/importer identity, registration, GPC, "
        "portal, blockchain, or document-hash identifiers."
    ),
    "cargoGroups": (
        "Source-ordered goods group facts, one gN for each source-supported cargo grouping."
    ),
    "cargoPackages": (
        "Source-ordered package levels, one pN per distinct printed package level. A container-row "
        "count and shipment total for the same type are one level; row counts belong in "
        "allocations."
    ),
    "cargoAllocationGroups": (
        "At most one record per groupId, covering only explicit container-to-cargo/package "
        "relationships; never create one when no valid printed container participates, and "
        "never infer by order or arithmetic. cargoPackages.groupId already connects a package "
        "level to its cargo group."
    ),
    "groupId": "Deterministic source-order identifier g1, g2, and so on.",
    "packageId": "Deterministic global source-order package identifier p1, p2, and so on.",
    "packageIds": "Source-ordered package identifiers explicitly covered by this allocation group.",
    "coverage": "Choose the narrow allocation shape exactly supported by the printed relationship.",
    "allocations": (
        "Source-ordered explicit container allocations required by the selected coverage shape."
    ),
    "containerNumber": (
        "Valid printed ISO 6346 identifier normalized only by removing printed separators."
    ),
    "packageQuantity": "Integer package quantity explicitly allocated to this container.",
    "quantity": (
        "Printed integer quantity for this package level. A dot or comma followed by exactly "
        "three digits before a count noun is a thousands separator, including livestock counts "
        "such as 4.713 BULLS = 4713; it is not a fractional package count."
    ),
    "typeDescription": (
        "Printed package or container type wording; do not invent a registry category or code. "
        "For competing variants on one container, use the equipment-row-associated value rather "
        "than concatenating it with a shipment-level phrase."
    ),
    "verifiedGrossMass": (
        "Mass explicitly and locally labeled VGM or verified gross mass; an ordinary container-row "
        "KGS/LBS value is not VGM."
    ),
    "sealNumbers": "Unique source-ordered seals explicitly associated with this container.",
    "temperatureSetpoint": "Explicit reefer temperature setpoint with printed unit.",
    "description": (
        "Joined product/goods wording only; retain integral product model/specification text such "
        "as a pump model's SIZE 2X2-8, but exclude shipment counts, weights, volumes, headings, "
        "packing construction, and boilerplate."
    ),
    "additionalInformation": (
        "Unique cargo-useful qualifiers only; never a duplicate, metadata dump, or product "
        "description."
    ),
    "grossWeight": (
        "Aggregate cargo mass explicitly labeled gross, with printed unit; never derive it by "
        "summing row or container measures."
    ),
    "netWeight": (
        "Aggregate cargo mass explicitly labeled net, not per-package mass or an unqualified total."
    ),
    "volume": (
        "Explicit cargo volume with printed unit; never derive it by summing row or container "
        "measures."
    ),
    "marksAndNumbers": (
        "Unique cargo marks only; exclude represented containers, metadata, counts, and headings."
    ),
    "hsCodes": (
        "Every unique 6-18 digit code under explicit HS, HSN, HTS, or tariff context, preserving "
        "all digits. Omit values outside 6-18 digits; never truncate, pad, or repair them."
    ),
    "handlingInstructions": (
        "Explicit operational cargo handling instructions, separate from product description and "
        "legal/responsibility boilerplate such as Shippers Load, Stow and Count."
    ),
    "dangerousGoods": "Only explicitly printed dangerous-goods facts with their own context.",
    "origin": "Printed goods origin only when explicitly scoped to cargo, never a party country.",
    "identifier": "Printed goods-origin identifier only when explicitly scoped to cargo origin.",
    "value": "Exact printed numeric measure after deterministic separator normalization.",
    "unit": (
        "Semantic unit supported by the explicitly printed source unit. Preserve a directly "
        "printed "
        "cargo MT/M-T/metric-tonne value as metric_tonne instead of converting it to kilograms; "
        "container VGM remains limited to directly printed kilograms or pounds."
    ),
    "name": "Printed entity or locality name without headings or separately modeled fields.",
    "address": (
        "One single-line joined postal address without party name, locality/country duplicates, "
        "contacts, or IDs."
    ),
    "city": "Printed city/locality as written, without inferred standardization.",
    "country": "Printed country text as written; never infer an ISO code.",
    "contactDetails": (
        "Explicit contact details belonging to this role-headed party. For a notify party with "
        "sameAs, these may be the only accompanying fields and override the referenced party's "
        "contacts for that notify role."
    ),
    "contactName": "Printed contact name copied with exact OCR spelling and without its heading.",
    "phoneNumbers": (
        "Unique printed telephone values containing digits; exclude standalone FAX values, "
        "locality, and label text. A jointly labeled TEL & FAX value is also a phone target."
    ),
    "emailAddresses": "Unique printed email addresses belonging to this party.",
    "websiteUrls": "Unique printed website URLs belonging to this party.",
    "vesselName": "Printed vessel name only, with any voyage suffix removed into voyageNumber.",
    "vesselImoNumber": (
        "Explicit seven-digit vessel IMO number with a valid IMO checksum. Omit and warn on a "
        "printed seven-digit Lloyds/MO value that fails the checksum; never repair it from PDF."
    ),
    "voyageNumber": "Printed voyage identifier, split from vessel name when combined.",
    "vesselFlagCountry": "Printed vessel flag country text as written.",
    "paymentArrangement": (
        "Only explicit freight prepaid, collect, third-party, or payable-elsewhere status. An "
        "explicit 'Freight and Charges payable at destination: Yes' means collect; Destination "
        "does not become paymentPlace. Do not map FIO/liner/handling terms, AS ARRANGED, "
        "Incoterms, charges, or amounts."
    ),
    "sameAs": (
        "Notify-party reference only: use shipper/consignee solely for explicit SAME AS wording "
        "when that referenced party is present. Identity and location come from the referenced "
        "party; only explicit notify-specific contactDetails may accompany sameAs as an override."
    ),
    "unNumber": "Exactly four printed digits under explicit UN/dangerous-goods context.",
    "hazardCategory": "Explicit primary dangerous-goods hazard class only.",
    "subsidiaryHazardCategory": "Explicit subsidiary dangerous-goods hazard class only.",
    "flashPoint": "Explicit dangerous-goods flash point and its printed temperature unit.",
    "temperature": "Explicit flash-point temperature with its printed unit.",
    "packingGroupCategory": "Explicit dangerous-goods packing group I, II, or III.",
    "warnings": "Only grounded omissions or representational limits; never duplicate label facts.",
    "code": "Select the warning category that exactly describes the grounded omission or limit.",
    "targetPath": (
        "A real emitted target field affected by the warning, or null when the schema cannot "
        "represent the cited fact."
    ),
    "decisionNotes": (
        "One to three short audit conclusions, not hidden reasoning or duplicated target data."
    ),
    "reason": "Select the exact supported exclusion reason established by the raw OCR document.",
    "reviewNotes": "One to three short grounded conclusions supporting exclusion.",
    "pageNumbers": (
        "Unique sorted source pages required only for consequential layout. For an unheaded "
        "continuation row, include both its row page and the nearest preceding governing "
        "table/header page."
    ),
    "ambiguity": (
        "Concise layout ambiguity that cannot responsibly be resolved from flattened OCR alone."
    ),
    "affectedTargetPaths": (
        "Relation-explicit target paths affected by the requested layout decision."
    ),
    "rawOcrEvidence": (
        "Minimal source-ordered anchors: exact page number and verbatim rawValue only."
    ),
    "pageNumber": "One-based source page number containing the verbatim raw value.",
    "rawValue": "A verbatim contiguous value copied from the cited raw OCR page.",
    "findings": (
        "All and only real candidate defects; return an empty list when the candidate is correct."
    ),
    "severity": (
        "Use blocking for any representable missing, incorrect, contaminated, or unsupported "
        "target fact; advisory only for genuine non-target uncertainty."
    ),
    "category": "The single defect category that determines the failed deterministic review check.",
    "message": (
        "Concise actionable defect statement naming what must change without proposing "
        "OCR-external data."
    ),
    "targetPaths": (
        "Relation-explicit candidate paths affected by this finding; empty only for "
        "document-unit issues."
    ),
    "imageUse": "How requested PDF layout evidence was used; it can never supply a target value.",
    "summary": "Short audit conclusion; do not include hidden reasoning or restate correct fields.",
    "observationType": "The exact layout relationship resolved from the requested PDF page.",
    "rawOcrAnchors": "Exact raw OCR anchors involved in this layout observation.",
    "guidance": "Layout-only conclusion; never add or correct factual text from the PDF.",
}


def _provider_schema(model_type: type[BaseModel]) -> dict[str, Any]:
    schema = deepcopy(model_type.model_json_schema(mode="validation"))

    def describe(node: Any) -> None:
        if isinstance(node, dict):
            properties = node.get("properties")
            if isinstance(properties, dict):
                for property_name, property_schema in properties.items():
                    if isinstance(property_schema, dict):
                        description = _PROVIDER_FIELD_DESCRIPTIONS.get(property_name)
                        if description is not None:
                            property_schema.setdefault("description", description)
            for child in node.values():
                describe(child)
        elif isinstance(node, list):
            for child in node:
                describe(child)

    describe(schema)
    return schema


def _wire_type(model_type: type[BaseModel], *, name: str) -> type[dict[str, Any]]:
    """Expose a strict provider schema without Python-mode model coercion.

    JSON arrays and ISO date strings are the only JSON representations of
    tuples and dates.  Pydantic's strict *Python* validation rejects those
    decoded values, while strict JSON validation correctly accepts them.  A
    StructuredDict keeps native constrained decoding active and lets the
    output validator cross the boundary through ``validate_json``.
    """

    return StructuredDict(_provider_schema(model_type), name=name)


_ANNOTATION_WIRE = _wire_type(AnnotationDraft, name="AnnotationDraft")
_COMPACT_ANNOTATION_WIRE = _wire_type(CompactAnnotationDraft, name="CompactAnnotationDraft")
_EXCLUSION_WIRE = _wire_type(ExclusionDraft, name="ExclusionDraft")
_COMPACT_EXCLUSION_WIRE = _wire_type(CompactExclusionDraft, name="CompactExclusionDraft")
_DOCUMENT_REQUEST_WIRE = _wire_type(DocumentAssistanceRequest, name="DocumentAssistanceRequest")
_COMPACT_DOCUMENT_REQUEST_WIRE = _wire_type(
    CompactDocumentAssistanceRequest, name="CompactDocumentAssistanceRequest"
)
_REVIEW_WIRE = _wire_type(ReviewDraft, name="ReviewDraft")
_COMPACT_REVIEW_WIRE = _wire_type(CompactReviewWireDraft, name="CompactReviewWireDraft")
_REVIEW_DOCUMENT_REQUEST_WIRE = _wire_type(
    ReviewDocumentAssistanceRequest, name="ReviewDocumentAssistanceRequest"
)
_COMPACT_REVIEW_DOCUMENT_REQUEST_WIRE = _wire_type(
    CompactReviewDocumentAssistanceRequest,
    name="CompactReviewDocumentAssistanceRequest",
)
_COMPACT_LAYOUT_GUIDANCE_WIRE = _wire_type(
    CompactDocumentLayoutGuidance, name="CompactDocumentLayoutGuidance"
)


def _compact_correction_wire(paths: tuple[str, ...]) -> type[dict[str, Any]]:
    """Return a provider wire containing exactly the normalized authorized paths."""

    schema = correction_envelope_json_schema(paths)
    properties = schema["properties"]["corrections"]["properties"]
    for path, value_schema in properties.items():
        fields = re.findall(r"(?:^|\.)([A-Za-z_][A-Za-z0-9_]*)", path)
        field_name = fields[-1] if fields else path
        if description := _PROVIDER_FIELD_DESCRIPTIONS.get(field_name):
            value_schema.setdefault("description", description)
    return StructuredDict(
        schema,
        name="BillOfLadingExactPathCorrection",
        description=(
            "Return exactly one replacement value for every reviewer-authorized normalized "
            "correction path and no other fields."
        ),
    )


def _validation_feedback(error: ValidationError) -> str:
    details = error.errors(
        include_url=False,
        include_context=False,
        include_input=False,
    )
    return (
        "The JSON matched the provider-enforced schema but failed semantic validation. "
        "Correct only these errors and return the complete object again:\n"
        + json.dumps(details, ensure_ascii=False, separators=(",", ":"))
    )


def _parse_extraction_wire(value: dict[str, Any]) -> ExtractionDecision:
    return _EXTRACTION_DECISION_ADAPTER.validate_json(canonical_json_bytes(value), strict=True)


def _validate_extraction_wire(value: dict[str, Any]) -> dict[str, Any]:
    try:
        _parse_extraction_wire(value)
    except ValidationError as error:
        raise ModelRetry(_validation_feedback(error)) from error
    return value


_SOURCE_ORDERED_UNIQUE_TEXT_FIELDS = {
    "additionalInformation",
    "emailAddresses",
    "forwardingAndExportReferences",
    "handlingInstructions",
    "hsCodes",
    "marksAndNumbers",
    "phoneNumbers",
    "sealNumbers",
    "websiteUrls",
}


def _join_physical_ocr_lines(value: str) -> str:
    return re.sub(r"[ \t]*[\r\n]+[ \t]*", " ", value).strip()


def _normalize_patch_value(
    node: Any, *, field_name: str | None = None, collapse_null_object: bool = True
) -> Any:
    if isinstance(node, str):
        joined = _join_physical_ocr_lines(node)
        if field_name == "containerNumber":
            return re.sub(r"[^A-Za-z0-9]", "", joined).upper()
        return joined
    if isinstance(node, dict):
        mapped_children = {
            key: _normalize_patch_value(child, field_name=key) for key, child in node.items()
        }
        if (
            collapse_null_object
            and mapped_children
            and all(child is None for child in mapped_children.values())
        ):
            return None
        return mapped_children
    if isinstance(node, list):
        sequence_children = [_normalize_patch_value(child, field_name=field_name) for child in node]
        if field_name in _SOURCE_ORDERED_UNIQUE_TEXT_FIELDS and all(
            isinstance(child, str) for child in sequence_children
        ):
            return list(dict.fromkeys(sequence_children))
        return sequence_children
    return node


def _normalize_document_patch_wire(patch: dict[str, Any]) -> dict[str, Any]:
    """Normalize representation artifacts inside one model-authored patch.

    The model-visible transcript remains byte-for-byte retained. The accepted
    value joins physical OCR wrapping in semantic scalar fields and removes
    exact repeated scalar facts only where the durable schema requires
    source-order uniqueness. No spelling, identifier, number, or semantic
    value changes.
    """

    normalized = _normalize_patch_value(patch, collapse_null_object=False)
    if not isinstance(normalized, dict):
        raise AssertionError("document patch normalization changed the root object type")
    return normalized


def _normalize_compact_extraction_wire(value: dict[str, Any]) -> dict[str, Any]:
    normalized = deepcopy(value)
    relation = normalized.get("relationExplicitLabel")
    if not isinstance(relation, dict):
        return normalized
    patch = relation.get("documentPatch")
    if isinstance(patch, dict):
        relation["documentPatch"] = _normalize_document_patch_wire(patch)
    return normalized


def _normalize_compact_correction_wire(value: dict[str, Any]) -> dict[str, Any]:
    normalized = deepcopy(value)
    corrections = normalized.get("corrections")
    if isinstance(corrections, dict):
        normalized_values: dict[str, Any] = {}
        for path, replacement in corrections.items():
            fields = re.findall(r"(?:^|\.)([A-Za-z_][A-Za-z0-9_]*)", path)
            field_name = fields[-1] if fields and path != "documentType" else None
            normalized_values[path] = _normalize_patch_value(
                replacement,
                field_name=field_name,
            )
        normalized["corrections"] = normalized_values
    return normalized


def _parse_compact_extraction_wire(value: dict[str, Any]) -> CompactExtractionDecision:
    normalized = _normalize_compact_extraction_wire(value)
    return _COMPACT_EXTRACTION_DECISION_ADAPTER.validate_json(
        canonical_json_bytes(normalized), strict=True
    )


def _parse_compact_correction_wire(value: dict[str, Any]) -> CompactCorrectionEnvelope:
    normalized = _normalize_compact_correction_wire(value)
    return CompactCorrectionEnvelope.model_validate_json(
        canonical_json_bytes(normalized), strict=True
    )


def _deterministic_retry_feedback(error: ValueError | WorkItemError) -> str:
    return (
        "The JSON matched the provider-enforced schema but failed deterministic local "
        f"validation: {type(error).__name__}: {error}. Correct that exact issue and return "
        "the complete object again without adding facts outside raw OCR."
    )


def _correction_retry_feedback(error: ValueError | WorkItemError) -> str:
    return (
        "The exact-path JSON correction matched the provider-enforced schema but failed "
        f"deterministic local validation: {type(error).__name__}: {error}. Return only "
        "the authorized exact correction values again; do not copy the complete candidate "
        "or change unrelated facts."
    )


def _validate_compact_extraction_wire(
    ctx: RunContext[_ExtractionValidationContext], value: dict[str, Any]
) -> dict[str, Any]:
    normalized = _normalize_compact_extraction_wire(value)
    try:
        parsed = _parse_compact_extraction_wire(normalized)
    except ValidationError as error:
        raise ModelRetry(_validation_feedback(error)) from error
    try:
        if isinstance(parsed, CompactAnnotationDraft):
            build_compact_annotation(
                ctx.deps.work_item,
                parsed,
                pdf_grouping_used=ctx.deps.pdf_grouping_used,
            )
        elif isinstance(parsed, CompactExclusionDraft):
            _hydrate_exclusion(ctx.deps.work_item, parsed)
        else:
            _hydrate_document_request(ctx.deps.work_item, parsed, review=False)
    except (ValueError, WorkItemError) as error:
        raise ModelRetry(_deterministic_retry_feedback(error)) from error
    return normalized


def _validate_compact_correction_wire(
    ctx: RunContext[_ExtractionValidationContext], value: dict[str, Any]
) -> dict[str, Any]:
    normalized = _normalize_compact_correction_wire(value)
    try:
        parsed = _parse_compact_correction_wire(normalized)
    except ValidationError as error:
        raise ModelRetry(_validation_feedback(error)) from error
    if ctx.deps.correction_base is None or not ctx.deps.correction_paths:
        raise ModelRetry("An exact-path correction requires a base candidate and correction paths.")
    try:
        merged = merge_correction_values(
            ctx.deps.correction_base,
            parsed.corrections,
            ctx.deps.correction_paths,
        )
        build_compact_annotation(
            ctx.deps.work_item,
            merged,
            pdf_grouping_used=ctx.deps.pdf_grouping_used,
        )
    except (ValueError, WorkItemError) as error:
        raise ModelRetry(_correction_retry_feedback(error)) from error
    return normalized


def _parse_review_wire(value: dict[str, Any]) -> ReviewDecision:
    return _REVIEW_DECISION_ADAPTER.validate_json(canonical_json_bytes(value), strict=True)


def _validate_review_wire(value: dict[str, Any]) -> dict[str, Any]:
    try:
        _parse_review_wire(value)
    except ValidationError as error:
        raise ModelRetry(_validation_feedback(error)) from error
    return value


def _parse_compact_review_wire(value: dict[str, Any]) -> CompactReviewDecision:
    return _COMPACT_REVIEW_DECISION_ADAPTER.validate_json(canonical_json_bytes(value), strict=True)


def _validate_compact_review_wire(
    ctx: RunContext[_ReviewValidationContext], value: dict[str, Any]
) -> dict[str, Any]:
    try:
        parsed = _parse_compact_review_wire(value)
    except ValidationError as error:
        raise ModelRetry(_validation_feedback(error)) from error
    try:
        if isinstance(parsed, CompactReviewWireDraft):
            hydrated = _hydrate_review(ctx.deps.work_item, parsed)
            validate_review_policy(
                hydrated.findings,
                candidate=ctx.deps.candidate,
                raw_ocr_text=ctx.deps.work_item.joinedRawText,
            )
        else:
            _hydrate_document_request(ctx.deps.work_item, parsed, review=True)
    except (ValueError, WorkItemError) as error:
        raise ModelRetry(_deterministic_retry_feedback(error)) from error
    return value


def _parse_layout_wire(value: dict[str, Any]) -> CompactDocumentLayoutGuidance:
    return CompactDocumentLayoutGuidance.model_validate_json(
        canonical_json_bytes(value), strict=True
    )


def _validate_layout_wire(
    ctx: RunContext[_LayoutValidationContext], value: dict[str, Any]
) -> dict[str, Any]:
    try:
        parsed = _parse_layout_wire(value)
    except ValidationError as error:
        raise ModelRetry(_validation_feedback(error)) from error
    try:
        _hydrate_layout_guidance(ctx.deps.work_item, parsed)
    except (ValueError, WorkItemError) as error:
        raise ModelRetry(_deterministic_retry_feedback(error)) from error
    return value


def _hydrate_document_request(
    work_item: AgentWorkItem,
    request: CompactDocumentAssistanceRequest | CompactReviewDocumentAssistanceRequest,
    *,
    review: bool,
) -> DocumentAssistanceRequest | ReviewDocumentAssistanceRequest:
    request_type = ReviewDocumentAssistanceRequest if review else DocumentAssistanceRequest
    return request_type.model_validate(
        {
            "decision": request.decision,
            "pageNumbers": tuple(sorted(set(request.pageNumbers))),
            "ambiguity": request.ambiguity,
            "affectedTargetPaths": tuple(dict.fromkeys(request.affectedTargetPaths)),
            "rawOcrEvidence": materialize_raw_ocr_evidence(
                page_texts(work_item.joinedRawText), request.rawOcrEvidence
            ),
        },
        strict=True,
    )


def _hydrate_exclusion(work_item: AgentWorkItem, draft: CompactExclusionDraft) -> ExclusionDraft:
    return ExclusionDraft.model_validate(
        {
            "decision": draft.decision,
            "reason": draft.reason,
            "rawOcrEvidence": materialize_raw_ocr_evidence(
                page_texts(work_item.joinedRawText), draft.rawOcrEvidence
            ),
            "reviewNotes": draft.reviewNotes,
        },
        strict=True,
    )


def _hydrate_review(work_item: AgentWorkItem, draft: CompactReviewWireDraft) -> CompactReviewDraft:
    pages = page_texts(work_item.joinedRawText)
    return CompactReviewDraft.model_validate(
        {
            "decision": draft.decision,
            "findings": tuple(
                {
                    "severity": finding.severity,
                    "category": finding.category,
                    "message": finding.message,
                    "targetPaths": tuple(dict.fromkeys(finding.targetPaths)),
                    "rawOcrEvidence": materialize_raw_ocr_evidence(pages, finding.rawOcrEvidence),
                    "imageUse": finding.imageUse,
                }
                for finding in draft.findings
            ),
            "summary": draft.summary,
        },
        strict=True,
    )


def _hydrate_layout_guidance(
    work_item: AgentWorkItem, guidance: CompactDocumentLayoutGuidance
) -> DocumentLayoutGuidance:
    pages = page_texts(work_item.joinedRawText)
    return DocumentLayoutGuidance.model_validate(
        {
            "decision": guidance.decision,
            "observations": tuple(
                {
                    "pageNumber": observation.pageNumber,
                    "observationType": observation.observationType,
                    "rawOcrAnchors": materialize_raw_ocr_evidence(pages, observation.rawOcrAnchors),
                    "guidance": observation.guidance,
                }
                for observation in guidance.observations
            ),
            "decisionNotes": guidance.decisionNotes,
        },
        strict=True,
    )


def _call_transcript(messages: list[Any]) -> AgentCallTranscript:
    events: list[dict[str, Any]] = []
    for message in messages:
        if isinstance(message, ModelResponse):
            serialized = ModelMessagesTypeAdapter.dump_python([message], mode="json")[0]
            events.append(
                {
                    "sequence": len(events),
                    "eventType": "model_response",
                    "payload": serialized,
                }
            )
        elif isinstance(message, ModelRequest):
            for part in message.parts:
                if isinstance(part, RetryPromptPart):
                    events.append(
                        {
                            "sequence": len(events),
                            "eventType": "validation_feedback",
                            "payload": part.model_response(),
                        }
                    )
    return AgentCallTranscript.model_validate(
        {"transcriptSchemaVersion": 1, "events": tuple(events)}, strict=True
    )


def _postprocess_failure(
    result: AgentCallResult[Any], error: ValueError | WorkItemError
) -> AgentCallFailure:
    """Turn a local post-response failure into a fully receipted retry."""

    value = result.receipt.model_dump(mode="python")
    value.update(
        {
            "status": "error",
            "outputSha256": None,
            "errorType": type(error).__name__,
            "errorMessage": str(error).strip() or type(error).__name__,
        }
    )
    receipt = AgentCallReceipt.model_validate(value, strict=True)
    return AgentCallFailure(
        f"{type(error).__name__}: {error}",
        receipt=receipt,
        fatal_run=False,
        terminal_document=False,
    )


class AgentProviderError(RuntimeError):
    """A configured model provider cannot satisfy the labeling contract."""


class AgentCallFailure(AgentProviderError):
    """A receipted provider failure with an explicit run/document retry scope."""

    def __init__(
        self,
        message: str,
        *,
        receipt: AgentCallReceipt,
        fatal_run: bool,
        terminal_document: bool,
    ) -> None:
        super().__init__(message)
        self.receipt = receipt
        self.fatal_run = fatal_run
        self.terminal_document = terminal_document


class _ProviderProbe(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)
    ok: Literal[True]


def _parse_ollama_version(value: str) -> tuple[int, int, int]:
    """Parse Ollama's numeric release prefix without accepting malformed values."""

    match = _OLLAMA_VERSION.match(value)
    if match is None:
        raise AgentProviderError(f"Ollama returned an invalid version string: {value!r}")
    patch = match.group("patch")
    return (int(match.group("major")), int(match.group("minor")), int(patch or 0))


@dataclass(frozen=True, slots=True)
class AgentCallResult[OutputT: BaseModel]:
    output: OutputT
    receipt: AgentCallReceipt


@dataclass(frozen=True, slots=True)
class _ProviderRuntime:
    config: ProviderConfig
    model: Model
    settings: ModelSettings


def _load_openai_key(environment_file: Path) -> str:
    from_environment = os.environ.get("OPENAI_API_KEY")
    from_file = dotenv_values(environment_file).get("OPENAI_API_KEY")
    key = from_environment if from_environment is not None else from_file
    if not isinstance(key, str):
        raise AgentProviderError(
            "OPENAI_API_KEY is absent from the process environment and repository .env"
        )
    if not key.strip():
        raise AgentProviderError("OPENAI_API_KEY must not be empty")
    return key


def _runtime(
    provider_config: ProviderConfig,
    *,
    environment_file: Path,
) -> _ProviderRuntime:
    if isinstance(provider_config, OpenAIResponsesProviderConfig):
        client = AsyncOpenAI(
            api_key=_load_openai_key(environment_file),
            max_retries=provider_config.transport_max_retries,
            timeout=provider_config.request_timeout_seconds,
        )
        provider = OpenAIProvider(openai_client=client)
        response_settings = OpenAIResponsesModelSettings(
            max_tokens=provider_config.max_output_tokens,
            timeout=provider_config.request_timeout_seconds,
            openai_reasoning_effort=provider_config.reasoning_effort,
            openai_reasoning_mode="standard",
            openai_reasoning_context="current_turn",
            openai_store=False,
            openai_text_verbosity="low",
        )
        response_model = OpenAIResponsesModel(provider_config.model, provider=provider)
        return _ProviderRuntime(
            provider_config, response_model, cast(ModelSettings, response_settings)
        )

    client = AsyncOpenAI(
        base_url=provider_config.base_url,
        api_key=provider_config.api_key,
        max_retries=provider_config.transport_max_retries,
        timeout=provider_config.request_timeout_seconds,
    )
    ollama_provider = OllamaProvider(openai_client=client)
    chat_settings = ModelSettings(
        max_tokens=provider_config.max_output_tokens,
        timeout=provider_config.request_timeout_seconds,
        temperature=0.0,
    )
    chat_model = OllamaModel(provider_config.model, provider=ollama_provider)
    return _ProviderRuntime(provider_config, chat_model, chat_settings)


def _price_request(usage: RequestUsage, pricing: OpenAIPricingConfig) -> Decimal:
    uncached = usage.input_tokens - usage.cache_read_tokens - usage.cache_write_tokens
    if uncached < 0:
        raise AgentProviderError("provider usage reports cache tokens above total input tokens")
    input_multiplier = Decimal(1)
    output_multiplier = Decimal(1)
    if usage.input_tokens > pricing.long_context_input_threshold_tokens:
        input_multiplier = Decimal(str(pricing.long_context_input_multiplier))
        output_multiplier = Decimal(str(pricing.long_context_output_multiplier))
    input_rate = Decimal(str(pricing.input_usd_per_million)) * input_multiplier
    cached_rate = Decimal(str(pricing.cached_input_usd_per_million)) * input_multiplier
    write_rate = input_rate * Decimal(str(pricing.cache_write_multiplier))
    output_rate = Decimal(str(pricing.output_usd_per_million)) * output_multiplier
    cost = (
        Decimal(uncached) * input_rate
        + Decimal(usage.cache_read_tokens) * cached_rate
        + Decimal(usage.cache_write_tokens) * write_rate
        + Decimal(usage.output_tokens) * output_rate
    ) / Decimal(1_000_000)
    return cost.quantize(_USD_QUANTUM, rounding=ROUND_HALF_UP)


def _response_receipt(
    response: ModelResponse,
    provider_config: ProviderConfig,
) -> ModelResponseReceipt:
    usage = response.usage
    reasoning = usage.details.get("reasoning_tokens", 0)
    if not isinstance(reasoning, int) or reasoning < 0:
        raise AgentProviderError("provider returned invalid reasoning token usage")
    cost = (
        _price_request(usage, provider_config.pricing)
        if isinstance(provider_config, OpenAIResponsesProviderConfig)
        else None
    )
    return ModelResponseReceipt.model_validate(
        {
            "providerResponseId": response.provider_response_id,
            "providerName": response.provider_name,
            "modelName": response.model_name,
            "finishReason": response.finish_reason,
            "inputTokens": usage.input_tokens,
            "cacheReadTokens": usage.cache_read_tokens,
            "cacheWriteTokens": usage.cache_write_tokens,
            "outputTokens": usage.output_tokens,
            "reasoningTokens": reasoning,
            "costUsd": cost,
        },
        strict=True,
    )


def _call_receipt(
    *,
    runtime: _ProviderRuntime,
    call_id: str,
    worker_id: str,
    document_id: str,
    stage: str,
    candidate_attempt: int,
    started_at: datetime,
    completed_at: datetime,
    duration_ms: float,
    work_item_sha256: str,
    prompt_identity: bytes,
    responses: tuple[ModelResponseReceipt, ...],
    transcript: AgentCallTranscript,
    output_sha256: str | None,
    error: AgentRunError | None,
    pdf_attachment: PdfAttachmentReceipt | None,
) -> AgentCallReceipt:
    if isinstance(runtime.config, OpenAIResponsesProviderConfig):
        cost_status = "priced" if responses else "unavailable"
    else:
        cost_status = "not_applicable_local"
    cost = (
        sum((row.costUsd for row in responses if row.costUsd is not None), Decimal(0))
        if cost_status == "priced"
        else None
    )
    return AgentCallReceipt.model_validate(
        {
            "receiptSchemaVersion": 3,
            "callId": call_id,
            "documentId": document_id,
            "stage": stage,
            "candidateAttempt": candidate_attempt,
            "providerId": runtime.config.id,
            "providerKind": runtime.config.kind,
            "model": runtime.config.model,
            "reasoningEffort": (
                runtime.config.reasoning_effort
                if isinstance(runtime.config, OpenAIResponsesProviderConfig)
                else runtime.config.reasoning_mode
            ),
            "workerId": worker_id,
            "startedAt": started_at,
            "completedAt": completed_at,
            "durationMs": duration_ms,
            "workItemSha256": work_item_sha256,
            "promptSha256": sha256_bytes(prompt_identity),
            "status": "error" if error is not None else "success",
            "outputSha256": output_sha256,
            "errorType": type(error).__name__ if error is not None else None,
            "errorMessage": str(error) if error is not None else None,
            "requests": len(responses),
            "responses": responses,
            "inputTokens": sum(row.inputTokens for row in responses),
            "cacheReadTokens": sum(row.cacheReadTokens for row in responses),
            "cacheWriteTokens": sum(row.cacheWriteTokens for row in responses),
            "outputTokens": sum(row.outputTokens for row in responses),
            "costUsd": cost,
            "costStatus": cost_status,
            "pdfAttachment": pdf_attachment,
            "transcript": transcript,
        },
        strict=True,
    )


class PydanticAgentGateway:
    """Three stateless structured-output agents sharing one request limiter."""

    def __init__(
        self,
        config: AgentLabelingConfig,
        *,
        project_root: Path,
        extractor_prompt: str,
        reviewer_prompt: str,
        document_prompt: str,
    ) -> None:
        self.config = config
        environment_file = project_root / config.environment_file
        if environment_file.is_symlink():
            raise AgentProviderError(".env must not be a symbolic link")
        runtimes = {
            row.id: _runtime(row, environment_file=environment_file) for row in config.providers
        }
        self._runtimes = runtimes
        limiter = ConcurrencyLimiter(
            config.workflow.max_concurrent_model_requests,
            name=f"{config.run.run_id}-all-model-calls",
        )
        self._limiter = limiter
        self._extractor_prompt = extractor_prompt
        self._reviewer_prompt = reviewer_prompt
        self._document_prompt = document_prompt
        extraction_wires: tuple[type[dict[str, Any]], ...]
        extraction_parser: Callable[[dict[str, Any]], BaseModel]
        extraction_validator: Callable[..., dict[str, Any]]
        review_wires: tuple[type[dict[str, Any]], ...]
        review_parser: Callable[[dict[str, Any]], BaseModel]
        review_validator: Callable[..., dict[str, Any]]
        if config.schema_version == 4:
            extraction_wires = (
                _COMPACT_ANNOTATION_WIRE,
                _COMPACT_EXCLUSION_WIRE,
                _COMPACT_DOCUMENT_REQUEST_WIRE,
            )
            extraction_parser = _parse_compact_extraction_wire
            extraction_validator = _validate_compact_extraction_wire
            review_wires = (
                _COMPACT_REVIEW_WIRE,
                _COMPACT_REVIEW_DOCUMENT_REQUEST_WIRE,
            )
            review_parser = _parse_compact_review_wire
            review_validator = _validate_compact_review_wire
        else:
            extraction_wires = (_ANNOTATION_WIRE, _EXCLUSION_WIRE, _DOCUMENT_REQUEST_WIRE)
            extraction_parser = _parse_extraction_wire
            extraction_validator = _validate_extraction_wire
            review_wires = (_REVIEW_WIRE, _REVIEW_DOCUMENT_REQUEST_WIRE)
            review_parser = _parse_review_wire
            review_validator = _validate_review_wire
        self._extraction_parser = extraction_parser
        self._review_parser = review_parser
        self._extractor = Agent[Any, dict[str, Any]](
            runtimes[config.assignments.labeler].model,
            output_type=NativeOutput(
                extraction_wires,
                name="bill_of_lading_extraction_decision",
                description=(
                    "Extract one complete OCR-grounded Bill of Lading or sea-waybill label, "
                    "exclude an ineligible document, or request page-scoped layout assistance."
                ),
                strict=True,
            ),
            system_prompt=extractor_prompt,
            deps_type=(_ExtractionValidationContext if config.schema_version == 4 else object),
            model_settings=runtimes[config.assignments.labeler].settings,
            retries=config.workflow.structured_output_retries,
            max_concurrency=limiter,
            name="bill-of-lading-labeler",
        )
        self._extractor.output_validator(extraction_validator)
        self._correctors: dict[tuple[str, ...], Agent[Any, dict[str, Any]]] = {}
        self._reviewer = Agent[Any, dict[str, Any]](
            runtimes[config.assignments.reviewer].model,
            output_type=NativeOutput(
                review_wires,
                name="bill_of_lading_independent_review",
                description=(
                    "Independently identify all real grounded defects in one candidate; return "
                    "no findings when it is complete and correct."
                ),
                strict=True,
            ),
            system_prompt=reviewer_prompt,
            deps_type=_ReviewValidationContext if config.schema_version == 4 else object,
            model_settings=runtimes[config.assignments.reviewer].settings,
            retries=config.workflow.structured_output_retries,
            max_concurrency=limiter,
            name="bill-of-lading-reviewer",
        )
        self._reviewer.output_validator(review_validator)
        self._document_agent = Agent[Any, dict[str, Any]](
            runtimes[config.assignments.document_layout].model,
            output_type=NativeOutput(
                _COMPACT_LAYOUT_GUIDANCE_WIRE,
                name="bill_of_lading_layout_guidance",
                description=(
                    "Resolve only requested page layout, heading-scope, row, continuation, or "
                    "document-boundary ambiguity without adding PDF-only facts."
                ),
                strict=True,
            ),
            system_prompt=document_prompt,
            deps_type=_LayoutValidationContext,
            model_settings=runtimes[config.assignments.document_layout].settings,
            retries=config.workflow.structured_output_retries,
            max_concurrency=limiter,
            name="bill-of-lading-pdf-layout-helper",
        )
        self._document_agent.output_validator(_validate_layout_wire)

    def _corrector_for_paths(self, correction_paths: tuple[str, ...]) -> Agent[Any, dict[str, Any]]:
        if self.config.schema_version != 4:
            raise AgentProviderError("exact-path correction requires schema version 4")
        normalized_paths = tuple(sorted(normalize_correction_paths(correction_paths)))
        if not normalized_paths:
            raise AgentProviderError("exact-path correction requires at least one path")
        if (corrector := self._correctors.get(normalized_paths)) is not None:
            return corrector

        runtime = self._runtimes[self.config.assignments.labeler]
        corrector = Agent[Any, dict[str, Any]](
            runtime.model,
            output_type=NativeOutput(
                _compact_correction_wire(normalized_paths),
                name="bill_of_lading_exact_path_correction",
                description=(
                    "Return exactly one replacement value for each reviewer-authorized "
                    "normalized correction path and no unrelated fields."
                ),
                strict=True,
            ),
            system_prompt=self._extractor_prompt,
            deps_type=_ExtractionValidationContext,
            model_settings=runtime.settings,
            retries=self.config.workflow.structured_output_retries,
            max_concurrency=self._limiter,
            name="bill-of-lading-corrector",
        )
        corrector.output_validator(_validate_compact_correction_wire)
        self._correctors[normalized_paths] = corrector
        return corrector

    def _limits(self) -> UsageLimits:
        workflow = self.config.workflow
        return UsageLimits(
            request_limit=workflow.max_requests_per_agent_run,
            input_tokens_limit=workflow.input_tokens_limit_per_agent_run,
            output_tokens_limit=workflow.output_tokens_limit_per_agent_run,
        )

    async def _call[CallOutputT: BaseModel](
        self,
        *,
        agent: Agent[Any, dict[str, Any]],
        output_parser: Callable[[dict[str, Any]], CallOutputT],
        user_prompt: str | list[str | BinaryContent],
        prompt_identity: bytes,
        provider_id: str,
        document_id: str,
        stage: str,
        candidate_attempt: int,
        work_item_sha256: str,
        pdf_attachment: PdfAttachmentReceipt | None = None,
        deps: Any = None,
    ) -> AgentCallResult[CallOutputT]:
        runtime = self._runtimes[provider_id]
        worker_id = f"worker_{uuid.uuid4().hex}"
        call_id = f"call_{uuid.uuid4().hex}"
        started_at = datetime.now(UTC)
        started = time.perf_counter()
        with capture_run_messages() as captured_messages:
            try:
                result = await agent.run(
                    user_prompt,
                    deps=deps,
                    usage_limits=self._limits(),
                )
            except AgentRunError as error:
                completed_at = datetime.now(UTC)
                duration_ms = (time.perf_counter() - started) * 1000.0
                responses = tuple(
                    _response_receipt(message, runtime.config)
                    for message in captured_messages
                    if isinstance(message, ModelResponse)
                )
                transcript = _call_transcript(captured_messages)
                receipt = _call_receipt(
                    runtime=runtime,
                    call_id=call_id,
                    worker_id=worker_id,
                    document_id=document_id,
                    stage=stage,
                    candidate_attempt=candidate_attempt,
                    started_at=started_at,
                    completed_at=completed_at,
                    duration_ms=duration_ms,
                    work_item_sha256=work_item_sha256,
                    prompt_identity=prompt_identity,
                    responses=responses,
                    transcript=transcript,
                    output_sha256=None,
                    error=error,
                    pdf_attachment=pdf_attachment,
                )
                fatal_run = isinstance(
                    error,
                    (ModelAPIError, RunCancelled, ConcurrencyLimitExceeded),
                )
                terminal_document = isinstance(
                    error,
                    (UsageLimitExceeded, ContentFilterError),
                )
                if (
                    not fatal_run
                    and not terminal_document
                    and not isinstance(error, UnexpectedModelBehavior)
                ):
                    fatal_run = True
                raise AgentCallFailure(
                    f"{type(error).__name__}: {error}",
                    receipt=receipt,
                    fatal_run=fatal_run,
                    terminal_document=terminal_document,
                ) from error
        completed_at = datetime.now(UTC)
        duration_ms = (time.perf_counter() - started) * 1000.0
        output = output_parser(result.output)
        responses = tuple(
            _response_receipt(message, runtime.config)
            for message in result.new_messages()
            if isinstance(message, ModelResponse)
        )
        if not responses or result.usage.requests != len(responses):
            raise AgentProviderError("PydanticAI request count differs from response receipts")
        output_sha256 = sha256_bytes(canonical_json_bytes(output.model_dump(mode="json")))
        transcript = _call_transcript(captured_messages)
        receipt = _call_receipt(
            runtime=runtime,
            call_id=call_id,
            worker_id=worker_id,
            document_id=document_id,
            stage=stage,
            candidate_attempt=candidate_attempt,
            started_at=started_at,
            completed_at=completed_at,
            duration_ms=duration_ms,
            work_item_sha256=work_item_sha256,
            prompt_identity=prompt_identity,
            responses=responses,
            transcript=transcript,
            output_sha256=output_sha256,
            error=None,
            pdf_attachment=pdf_attachment,
        )
        return AgentCallResult(output, receipt)

    async def extract(
        self,
        work_item: AgentWorkItem,
        *,
        work_item_sha256: str,
        candidate_attempt: int,
        retry_feedback: str | None,
        layout_guidance: DocumentLayoutGuidance | None,
        correction_base: CompactAnnotationDraft | None = None,
        correction_paths: tuple[str, ...] = (),
    ) -> AgentCallResult[BaseModel]:
        if (correction_base is None) != (not correction_paths):
            raise AgentProviderError(
                "correction_base and correction_paths must either both be supplied or both absent"
            )
        if correction_base is not None:
            correction_paths = normalize_correction_paths(correction_paths)
        payload = {
            "task": (
                "correct_only_authorized_bill_of_lading_target_scopes"
                if correction_base is not None
                else "extract_one_bill_of_lading_label_from_raw_ocr"
            ),
            "candidateAttempt": candidate_attempt,
            "retryFeedback": retry_feedback,
            "layoutGuidance": (
                layout_guidance.model_dump(mode="json") if layout_guidance else None
            ),
            "correctionBase": (
                correction_base.model_dump(mode="json") if correction_base else None
            ),
            "allowedCorrectionPaths": list(correction_paths),
            "rawOcrDocument": work_item.joinedRawText,
        }
        user_prompt = json.dumps(payload, ensure_ascii=False, separators=(",", ":"))
        prompt_identity = self._extractor_prompt.encode() + b"\0" + user_prompt.encode()
        correction_mode = correction_base is not None
        agent = self._corrector_for_paths(correction_paths) if correction_mode else self._extractor
        result = await self._call(
            agent=agent,
            output_parser=(
                _parse_compact_correction_wire if correction_mode else self._extraction_parser
            ),
            user_prompt=user_prompt,
            prompt_identity=prompt_identity,
            provider_id=self.config.assignments.labeler,
            document_id=work_item.source.documentId,
            stage="extract",
            candidate_attempt=candidate_attempt,
            work_item_sha256=work_item_sha256,
            deps=(
                _ExtractionValidationContext(
                    work_item=work_item,
                    pdf_grouping_used=layout_guidance is not None,
                    correction_base=correction_base,
                    correction_paths=correction_paths,
                )
                if self.config.schema_version == 4
                else None
            ),
        )
        output: BaseModel = result.output
        try:
            if isinstance(output, CompactCorrectionEnvelope):
                if correction_base is None:
                    raise AgentProviderError("corrector returned without a base candidate")
            elif isinstance(output, CompactExclusionDraft):
                output = _hydrate_exclusion(work_item, output)
            elif isinstance(output, CompactDocumentAssistanceRequest):
                output = _hydrate_document_request(work_item, output, review=False)
        except (ValueError, WorkItemError) as error:
            raise _postprocess_failure(result, error) from error
        return AgentCallResult(output, result.receipt)

    async def review(
        self,
        work_item: AgentWorkItem,
        candidate: BaseModel,
        *,
        work_item_sha256: str,
        candidate_attempt: int,
        layout_guidance: DocumentLayoutGuidance | None,
    ) -> AgentCallResult[BaseModel]:
        candidate_payload = candidate.model_dump(mode="json")
        if self.config.schema_version == 4 and "relationExplicitLabel" in candidate_payload:
            candidate_payload = {
                "documentType": candidate_payload["documentType"],
                "relationExplicitLabel": candidate_payload["relationExplicitLabel"],
                "warnings": candidate_payload["warnings"],
            }
        payload = {
            "task": "independently_review_one_bill_of_lading_candidate",
            "candidateAttempt": candidate_attempt,
            "candidate": candidate_payload,
            "layoutGuidance": (
                layout_guidance.model_dump(mode="json") if layout_guidance else None
            ),
            "rawOcrDocument": work_item.joinedRawText,
        }
        user_prompt = json.dumps(payload, ensure_ascii=False, separators=(",", ":"))
        prompt_identity = self._reviewer_prompt.encode() + b"\0" + user_prompt.encode()
        result = await self._call(
            agent=self._reviewer,
            output_parser=self._review_parser,
            user_prompt=user_prompt,
            prompt_identity=prompt_identity,
            provider_id=self.config.assignments.reviewer,
            document_id=work_item.source.documentId,
            stage="review",
            candidate_attempt=candidate_attempt,
            work_item_sha256=work_item_sha256,
            deps=(
                _ReviewValidationContext(work_item=work_item, candidate=candidate_payload)
                if self.config.schema_version == 4
                else None
            ),
        )
        output: BaseModel = result.output
        try:
            if isinstance(output, CompactReviewWireDraft):
                output = _hydrate_review(work_item, output)
                validate_review_policy(
                    output.findings,
                    candidate=candidate_payload,
                    raw_ocr_text=work_item.joinedRawText,
                )
            elif isinstance(output, CompactReviewDocumentAssistanceRequest):
                output = _hydrate_document_request(work_item, output, review=True)
        except (ValueError, WorkItemError) as error:
            raise _postprocess_failure(result, error) from error
        return AgentCallResult(output, result.receipt)

    async def document_layout(
        self,
        work_item: AgentWorkItem,
        *,
        work_item_sha256: str,
        candidate_attempt: int,
        request: BaseModel,
        pdf: PdfAssistancePayload,
    ) -> AgentCallResult[DocumentLayoutGuidance]:
        request_json = json.dumps(
            request.model_dump(mode="json"), ensure_ascii=False, separators=(",", ":")
        )
        user_content: list[str | BinaryContent] = [
            json.dumps(
                {
                    "task": "resolve_layout_only_without_adding_pdf_only_facts",
                    "documentRequest": json.loads(request_json),
                    "pdfPageMap": [
                        {
                            "attachmentPageNumber": index,
                            "sourcePageNumber": page_number,
                        }
                        for index, page_number in enumerate(pdf.source_page_numbers, start=1)
                    ],
                    "rawOcrDocument": work_item.joinedRawText,
                },
                ensure_ascii=False,
                separators=(",", ":"),
            )
        ]
        user_content.append(
            "Requested source pages as one PDF attachment; use only the pdfPageMap above "
            "when citing source page numbers:"
        )
        user_content.append(BinaryContent(data=pdf.data, media_type=pdf.media_type))
        identity_parts = [
            self._document_prompt.encode(),
            b"\0",
            request_json.encode(),
            b"\0",
            pdf.source_pdf_sha256.encode(),
            b"\0",
            pdf.attachment_pdf_sha256.encode(),
            b"\0",
            ",".join(str(value) for value in pdf.source_page_numbers).encode(),
        ]
        attachment_receipt = PdfAttachmentReceipt.model_validate(
            {
                "sourcePdfSha256": pdf.source_pdf_sha256,
                "sourcePageNumbers": pdf.source_page_numbers,
                "attachmentPdfSha256": pdf.attachment_pdf_sha256,
                "attachmentBytes": len(pdf.data),
                "mediaType": pdf.media_type,
                "constructionMethod": pdf.construction_method,
            },
            strict=True,
        )
        result = await self._call(
            agent=self._document_agent,
            output_parser=_parse_layout_wire,
            user_prompt=user_content,
            prompt_identity=b"".join(identity_parts),
            provider_id=self.config.assignments.document_layout,
            document_id=work_item.source.documentId,
            stage="document_layout",
            candidate_attempt=candidate_attempt,
            work_item_sha256=work_item_sha256,
            pdf_attachment=attachment_receipt,
            deps=_LayoutValidationContext(work_item=work_item),
        )
        try:
            output = _hydrate_layout_guidance(work_item, result.output)
        except (ValueError, WorkItemError) as error:
            raise _postprocess_failure(result, error) from error
        return AgentCallResult(output, result.receipt)

    async def preflight_ollama(self) -> dict[str, dict[str, Any]]:
        """Verify exact local tags, Ollama version, and native structured output."""

        reports: dict[str, dict[str, Any]] = {}
        for provider in self.config.providers:
            if not isinstance(provider, OllamaProviderConfig):
                continue
            parsed = urlsplit(provider.base_url)
            root = f"{parsed.scheme}://{parsed.netloc}"
            timeout = httpx.Timeout(provider.request_timeout_seconds)
            async with httpx.AsyncClient(timeout=timeout) as client:
                version_response = await client.get(f"{root}/api/version")
                version_response.raise_for_status()
                version_payload = version_response.json()
                version = version_payload.get("version")
                if not isinstance(version, str):
                    raise AgentProviderError("Ollama /api/version returned no version")
                parsed_version = _parse_ollama_version(version)
                if parsed_version < (0, 5, 0):
                    raise AgentProviderError("Ollama 0.5 or newer is required")
                show_response = await client.post(
                    f"{root}/api/show", json={"model": provider.model}
                )
                show_response.raise_for_status()
                show_payload = show_response.json()
                capabilities = show_payload.get("capabilities", [])
                if not isinstance(capabilities, list) or any(
                    not isinstance(value, str) for value in capabilities
                ):
                    raise AgentProviderError("Ollama /api/show returned invalid capabilities")
                runtime = self._runtimes[provider.id]
                probe = Agent(
                    runtime.model,
                    output_type=NativeOutput(
                        _ProviderProbe,
                        name="native_json_schema_probe",
                        strict=True,
                    ),
                    model_settings=runtime.settings,
                    retries=0,
                    name="ollama-native-json-schema-probe",
                )
                probe_result = await probe.run(
                    "Return ok=true using the required native JSON schema.",
                    usage_limits=UsageLimits(request_limit=1, output_tokens_limit=32),
                )
                if probe_result.output.ok is not True:
                    raise AgentProviderError("Ollama native JSON-schema probe failed")
                reports[provider.id] = {
                    "version": version,
                    "model": provider.model,
                    "capabilities": sorted(capabilities),
                    "native_json_schema_required": provider.native_json_schema,
                    "native_json_schema_probe": "passed",
                    "probe_input_tokens": probe_result.usage.input_tokens,
                    "probe_output_tokens": probe_result.usage.output_tokens,
                }
        return reports
