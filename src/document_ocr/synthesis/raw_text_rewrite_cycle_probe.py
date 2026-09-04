"""Atomic, provider-constrained rewriting of synthetic Bill of Lading OCR text.

The model submits one terminal line-range patch.  The output function validates
and applies the complete patch atomically, so successful edits are not sent back
through another model turn.  A separate compact semantic reviewer may request a
bounded correction.  Every provider message and exact before/after artifact is
retained, while no training record is published by this probe.
"""

from __future__ import annotations

import asyncio
import copy
import json
import os
import re
import resource
import time
import unicodedata
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from datetime import UTC, date, datetime
from decimal import ROUND_DOWN, ROUND_HALF_UP, Decimal
from difflib import SequenceMatcher
from importlib.metadata import version
from itertools import pairwise
from pathlib import Path
from types import MappingProxyType
from typing import Annotated, Any, Literal, cast
from urllib.parse import urlsplit, urlunsplit

from openai import AsyncOpenAI
from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    JsonValue,
    StringConstraints,
    create_model,
    model_validator,
)
from pydantic_ai import (
    Agent,
    ModelRetry,
    NativeOutput,
    RunContext,
    ToolOutput,
    capture_run_messages,
)
from pydantic_ai.concurrency import ConcurrencyLimiter
from pydantic_ai.messages import CachePoint, ModelResponse, UserContent
from pydantic_ai.models.openai import OpenAIResponsesModel, OpenAIResponsesModelSettings
from pydantic_ai.providers.openai import OpenAIProvider
from pydantic_ai.usage import UsageLimits

from document_ocr.atomic import json_artifact_bytes, read_regular_file_bytes
from document_ocr.hashing import canonical_json_bytes, sha256_bytes, sha256_file
from document_ocr.label_schemas import bill_of_lading_v3, bill_of_lading_v5
from document_ocr.labeling_agents.deterministic_annotation import (
    explicit_hs_codes_from_text,
)
from document_ocr.synthesis.config import SynthesisRawTextRewriteCycleProbeConfig
from document_ocr.synthesis.container_semantics import review_source_equipment_surface
from document_ocr.synthesis.country_registry import CountryRegistry, load_iso_country_registry
from document_ocr.synthesis.generators import DeterministicStream, largest_remainder_allocation
from document_ocr.synthesis.linguistic_completion_pipeline import DocumentLinguisticPlan
from document_ocr.synthesis.linguistic_probe_runtime import (
    LinguisticUsageReceipt,
    load_openai_key,
    model_messages,
    openai_responses_settings,
    usage_receipt,
)
from document_ocr.synthesis.package_registry import LoadedPackageRegistry, load_package_registry
from document_ocr.synthesis.raw_text_rewrite_probe import (
    ChangedLeaf,
    _load_jsonl,
    _resolve_pinned_file,
    _validate_linguistic_run,
    changed_leaves,
    unified_text_diff,
)
from document_ocr.synthesis.route_registry import (
    UnlocodeLocation,
    load_pinned_unlocode_locations,
)
from document_ocr.synthesis.run_safety import StagedArtifactRun
from document_ocr.synthesis.task_adapter import (
    BILL_OF_LADING_TASK_ADAPTER,
    BILL_OF_LADING_V5_TASK_ADAPTER,
)
from document_ocr.synthesis.thermal_goods import render_synthetic_hs_code
from document_ocr.synthesis.transport_capacity import (
    EquipmentCapacity,
    EquipmentFamily,
    TransportCapacityLimits,
    capacity_limits,
    equipment_capacity,
    reproject_measures_for_semantic_equipment,
)
from document_ocr.training.config import resolve_config_path

NonEmptyText = Annotated[str, StringConstraints(strip_whitespace=True, min_length=1)]
EvidenceText = Annotated[str, StringConstraints(min_length=1, max_length=600)]
Sha256 = Annotated[str, StringConstraints(pattern=r"^[0-9a-f]{64}$")]
_STRICT = ConfigDict(extra="forbid", frozen=True, strict=True, allow_inf_nan=False)
_IMPLEMENTATION_PATH = Path(__file__).resolve(strict=True)
_PAGE_MARKER = re.compile(r"^--- PAGE [1-9][0-9]* ---[ \t]*$")
_NEWLINE = re.compile(r"\r\n|\r|\n")
_PLACEHOLDER = re.compile(
    r"(?<![A-Z0-9])(?:UNAVAILABLE|NOT[ \t]+AVAILABLE|UNKNOWN|TBD|TBA|N[./]A[.]?)(?![A-Z0-9])",
    re.IGNORECASE,
)
_INLINE_SLOT_LABEL = re.compile(
    r"(?<![A-Z0-9])(?P<label>PHONE|FAX|MOBILE|TEL(?:EPHONE)?|PH|E-?MAIL|WEBSITE|WEB|T)"
    r"(?P<spacing>[ \t]*):",
    re.IGNORECASE,
)
_ELECTRONIC_CONTACT_SURFACE = re.compile(
    r"(?:https?://|www\.)[^\s]+|\S*@\S+",
    re.IGNORECASE,
)
_PRESERVABLE_STATUS_LINE = re.compile(
    r"^[ \t]*(?:EXPRESS[ \t]+RELEASE|NON-NEGOTIABLE(?:[ \t]+SEA[ \t]+WAYBILL)?|"
    r"DELIVERY[ \t]+STATUS[ \t]+FREE[ \t]+OUT|FREE[ \t]+OUT|FCL/FCL|"
    r"FREIGHT[ \t]+(?:PREPAID|COLLECT))[ \t]*$",
    re.IGNORECASE,
)
_PRESERVABLE_LEGAL_BOILERPLATE_LINE = re.compile(
    r"^[ \t]*(?:"
    r"THE[ \t]+SURRENDER[ \t]+OF\b.*\bBILL[ \t]+OF[ \t]+LADING\b.*|"
    r"IN[ \t]+WITNESS[ \t]+WHEREOF\b.*\bBILLS?[ \t]+OF[ \t]+LADING\b.*|"
    r"ATTENTION[ \t]+OF[ \t]+SHIPPER\b.*\bTERMS?[ \t]+AND[ \t]+CONDITIONS?\b.*"
    r")[ \t]*$",
    re.IGNORECASE,
)
_STRUCTURAL_LINE_PREFIX = re.compile(
    r"^(?P<prefix>[ \t]*(?:\([0-9]+\)|[0-9]+(?:\.[0-9]+)+\.)\s+)"
)
_POSITIVE_REEFER_OPERATION = re.compile(
    r"\b(?:TEMPERATURE[ \t]+(?:IS[ \t]+)?(?:TO[ \t]+BE[ \t]+)?SET|"
    r"PLUGGING[ \t]+FOR[ \t]+(?:THE[ \t]+)?ACCOUNT)\b",
    re.IGNORECASE,
)
_NON_OPERATING_REEFER = re.compile(
    r"\b(?:NON[- ]OPERATING|SWITCHED[ \t]+OFF|COOLING[ \t]+NOT[ \t]+OPERATING)\b",
    re.IGNORECASE,
)
_COMPOUND_PARTY_RELATIONS: tuple[
    tuple[Literal["on_behalf_of", "trading_as"], re.Pattern[str]], ...
] = (
    ("on_behalf_of", re.compile(r"\b(?:ON\s+BEHALF\s+OF|O\s*/?\s*B\s*/?\s*O)\b", re.I)),
    (
        "trading_as",
        re.compile(r"\b(?:TRADING\s+AS|DOING\s+BUSINESS\s+AS|T\s*/\s*A|D\s*/\s*B\s*/\s*A)\b", re.I),
    ),
)
_COMPOUND_PARTY_ROLES = (
    "shipper",
    "consignee",
    "carrier",
    "forwardingAgent",
    "deliveryAgent",
    "consolidator",
)
_CONTAINER_TYPE_DESCRIPTION_PATH = re.compile(
    r"^documentPatch\.containers\[([0-9]+)\]\.typeDescription$"
)
_CONTAINER_SIZE_CATEGORY_PATH = re.compile(r"^documentPatch\.containers\[([0-9]+)\]\.sizeCategory$")
_CONTAINER_TYPE_CATEGORY_PATH = re.compile(r"^documentPatch\.containers\[([0-9]+)\]\.typeCategory$")
_NUMERIC_DATE = re.compile(
    r"(?<![0-9])(?P<a>[0-9]{1,4})(?P<s>[./-])(?P<b>[0-9]{1,2})(?P=s)(?P<c>[0-9]{1,4})(?![0-9])"
)
_DAY_NAMED_MONTH_DATE = re.compile(
    r"(?<![A-Z0-9])(?P<d>[0-9]{1,2})(?P<s>[- /])(?P<m>[A-Z]{3,9})"
    r"(?P=s)(?P<y>[0-9]{2,4})(?![A-Z0-9])",
    re.IGNORECASE,
)
_NAMED_MONTH_DAY_DELIMITED_DATE = re.compile(
    r"(?<![A-Z0-9])(?P<m>[A-Z]{3,9})(?P<s>[-/])(?P<d>[0-9]{1,2})"
    r"(?P=s)(?P<y>[0-9]{2,4})(?![A-Z0-9])",
    re.IGNORECASE,
)
_NAMED_MONTH_DAY_DATE = re.compile(
    r"(?<![A-Z0-9])(?P<m>[A-Z]{3,9})[ ]+(?P<d>[0-9]{1,2})"
    r"(?P<comma>,?)[ ]+(?P<y>[0-9]{2,4})(?![A-Z0-9])",
    re.IGNORECASE,
)
_NUMERIC_SURFACE = re.compile(r"(?<![0-9])[0-9](?:[0-9. /-]*[0-9])?(?![0-9])")
_ISSUE_DATE_CONTEXT = re.compile(
    r"(?:PLACE\s+(?:(?:AND|&)\s+DATE\s+OF\s+ISSUE|OF\s+ISSUE\s+DATE)|DATE\s+OF\s+ISSUE)",
    re.I,
)
_SHIPPED_DATE_CONTEXT = re.compile(
    r"(?:(?:SHIP|SHIPPED)\s+ON\s+BOARD(?:\s+DATE)?|ON\s+BOARD\s+DATE)", re.I
)
_ANONYMOUS_EQUIPMENT = re.compile(
    r"(?:/|X\s+)?(?:20|40|45)[\u2019'\"]?(?:\s*FT)?\s*"
    r"(?:HC|HQ|HIGH\s+CUBE|DRY|GP|DC|REEFER|RF)?"
    r"\s*CONTAINERS?\s+SAID\s+TO\s+CONTAIN",
    re.I,
)
_CARRIER_RECEIPT_COUNT = re.compile(
    r"(?P<prefix>TOTAL\s+NUMBER\s+OF\s+(?:CONTAINERS?\s+OR\s+PACKAGES?|"
    r"PACKAGES?|CONTAINERS?)\s+)(?P<count>[0-9][0-9,]*)"
    r"(?P<suffix>\s+RECEIVED\s+BY\s+CARRIER\s*:?)",
    re.IGNORECASE,
)
_CARRIER_RECEIPT_HEADING = re.compile(r"\bCARRIER'?S[ \t]+RECEIPT\b", re.IGNORECASE)
_EQUIPMENT_BREAKDOWN_LINE = re.compile(
    r"^[ \t]*[0-9][0-9,]*[ \t]+X[ \t]+(?:20|40|45)'"
    r"(?:[ \t]*\+[ \t]*[0-9][0-9,]*[ \t]+X[ \t]+(?:20|40|45)')*[ \t]*$",
    re.IGNORECASE,
)
_RAW_AGENT_FOR_CARRIER_LINE = re.compile(
    r"^[ \t]*AS[ \t]+AGENTS?[ \t]+FOR[ \t]+THE[ \t]+CARRIER"
    r"(?:[ \t]+AND[ \t]+SERVICE[ \t]+PROVIDER)?[ \t:]+"
    r"(?P<principal>[^\r\n]{2,160})[ \t]*$",
    re.IGNORECASE,
)
_RAW_ISSUED_AGENT_BY_LINE = re.compile(
    r"^[ \t]*(?:\([0-9]+\)[ \t]+)?ISSUED[ \t]+AS[ \t]+AGENTS?[ \t]+FOR[ \t]+"
    r"(?P<principal>[^\r\n]{2,160}?)[ \t]+AS[ \t]+CARRIER[ \t]+BY:[ \t]*$",
    re.IGNORECASE,
)
_RAW_INLINE_AGENT_FOR_CARRIER_LINE = re.compile(
    r"^[ \t]*(?P<identity>[^\r\n]{2,160}?),?[ \t]+AS[ \t]+AGENTS?[ \t]+FOR[ \t]+"
    r"THE[ \t]+CARRIER(?:[ \t]+(?P<principal>[^\r\n]{2,160}))?[ \t]*$",
    re.IGNORECASE,
)
_RAW_SIGNED_ON_BEHALF_LINE = re.compile(
    r"^[ \t]*SIGNED[ \t]+ON[ \t]+BEHALF[ \t]+OF[ \t]+"
    r"(?:THE[ \t]+CARRIER[ \t]+)?(?P<principal>[^\r\n]{2,160})[ \t]*$",
    re.IGNORECASE,
)
_RAW_BY_AGENT_LINE = re.compile(
    r"^[ \t]*BY[ \t]+(?P<identity>[^\r\n]{2,160}?)[ \t]+AS[ \t]+AGENT"
    r"(?:[ \t]+[^\r\n]{1,160})?[ \t]*$",
    re.IGNORECASE,
)
_RAW_GROSS_WEIGHT = re.compile(
    r"(?<![A-Z0-9])(?:G\.?[ \t]*W\.?|GROSS[ \t]+(?:WEIGHT|WT\.?))"
    r"[ \t]*[:#-]?[ \t]*(?P<value>[0-9]+(?:[.,][0-9]+)*)",
    re.IGNORECASE,
)
_RAW_VOLUME = re.compile(
    r"(?<![A-Z0-9])(?:CBM|CUBIC[ \t]+MET(?:ER|RE)S?|VOLUME)"
    r"[ \t]*[:#-]?[ \t]*(?P<value>[0-9]+(?:[.,][0-9]+)*)",
    re.IGNORECASE,
)
_RAW_INLINE_CONTAINER_MEASURES = re.compile(
    r"\((?P<gross>[0-9]+(?:[.,][0-9]+)*)[ \t]*KG[ \t]*/[ \t]*"
    r"(?P<volume>[0-9]+(?:[.,][0-9]+)*)[ \t]*M3[ \t]*/[ \t]*"
    r"(?P<packages>[0-9][0-9,]*)[ \t]*PK\)",
    re.IGNORECASE,
)
_RAW_MEASUREMENT_NUMBER = re.compile(r"(?<![0-9.,])(?P<value>[0-9]+(?:[.,][0-9]+)*)(?![0-9.,])")
_CONTAINER_EQUIPMENT_REVIEW_PATH = re.compile(
    r"documentPatch\.containers\[([0-9]+)\]\."
    r"(?:typeDescription|printedEquipmentSurface|sizeCategory|typeCategory)"
)
_ANCHORABLE_IDENTIFIER_PATH = re.compile(
    r"^documentPatch\.(?:"
    r"billOfLadingNumber|"
    r"forwardingAndExportReferences\[[0-9]+\]|"
    r"transport\.voyageNumber|"
    r"containers\[[0-9]+\]\.(?:containerNumber|sealNumbers\[[0-9]+\])|"
    r"cargoAllocationGroups\[[0-9]+\]\.allocations\[[0-9]+\]\.containerNumber|"
    r"cargoGroups\[[0-9]+\]\.marksAndNumbers\[[0-9]+\]|"
    r"cargoGroups\[[0-9]+\]\.dangerousGoods\[[0-9]+\]\.unNumber"
    r")$"
)
_ANCHORABLE_CONTACT_PATH = re.compile(
    r"^documentPatch\.parties\.(?:shipper|consignee|carrier|deliveryAgent|"
    r"forwardingAgent|notifyParties\[[0-9]+\])\.contactDetails\."
    r"(?:contactName|phoneNumbers\[[0-9]+\]|emailAddresses\[[0-9]+\]|"
    r"websiteUrls\[[0-9]+\])$"
)
_SPECIAL_USE_TLDS = frozenset({"example", "invalid", "localhost", "test"})
_SPECIAL_USE_DOMAINS = frozenset(
    {
        "example.com",
        "example.net",
        "example.org",
        "localhost",
    }
)
_COUNTRY_CODE = re.compile(r"^[A-Z]{2}$")
_DOMAIN_TOKEN = re.compile(r"[^a-z0-9]+")
_MONTHS = {
    "JAN": 1,
    "JANUARY": 1,
    "FEB": 2,
    "FEBRUARY": 2,
    "MAR": 3,
    "MARCH": 3,
    "APR": 4,
    "APRIL": 4,
    "MAY": 5,
    "JUN": 6,
    "JUNE": 6,
    "JUL": 7,
    "JULY": 7,
    "AUG": 8,
    "AUGUST": 8,
    "SEP": 9,
    "SEPT": 9,
    "SEPTEMBER": 9,
    "OCT": 10,
    "OCTOBER": 10,
    "NOV": 11,
    "NOVEMBER": 11,
    "DEC": 12,
    "DECEMBER": 12,
}


class RawTextRewriteCycleError(RuntimeError):
    """The atomic rewrite probe cannot establish its artifact contract."""


class LineRangeReplacement(BaseModel):
    """One inclusive line-ID range replaced in the current OCR text."""

    model_config = _STRICT

    startLineId: Annotated[
        str,
        StringConstraints(pattern=r"^L[0-9]{5}$"),
        Field(description="Exact ID preceding the first line to replace."),
    ]
    endLineId: Annotated[
        str,
        StringConstraints(pattern=r"^L[0-9]{5}$"),
        Field(description="Exact ID preceding the inclusive final line to replace."),
    ]
    newText: Annotated[
        str,
        StringConstraints(max_length=50_000),
        Field(
            description=(
                "Complete replacement text without a trailing newline. Emit exactly one output "
                "line per selected source line and preserve which selected lines are blank. "
                "Never use unavailable, unknown, or N/A placeholders; a target-absent assertion "
                "must be replaced in its occupied slot with coherent non-extractable flavor."
            )
        ),
    ]


class AppliedLineRangeReplacement(BaseModel):
    model_config = _STRICT

    startLine: Annotated[int, Field(ge=1)]
    endLine: Annotated[int, Field(ge=1)]
    startLineId: str
    endLineId: str
    oldText: str
    newText: str
    oldTextSha256: Sha256
    sourceLinesReplaced: Annotated[int, Field(ge=1)]
    outputLinesInserted: Annotated[int, Field(ge=0)]


class AppliedDeterministicPrefill(BaseModel):
    """One exact line-bound value or derived surface completed without a provider call."""

    model_config = _STRICT

    lineId: Annotated[str, StringConstraints(pattern=r"^L[0-9]{5}$")]
    targetPaths: Annotated[tuple[NonEmptyText, ...], Field(min_length=1)]
    sourceSurface: NonEmptyText
    targetSurface: NonEmptyText
    beforeLine: str
    afterLine: str


class CompoundPartyFlavorRealization(BaseModel):
    """One target-primary identity plus fictional source-topology flavor."""

    model_config = _STRICT

    targetPath: Annotated[
        str,
        StringConstraints(
            pattern=(
                r"^documentPatch\.parties\."
                r"(?:shipper|consignee|carrier|forwardingAgent|deliveryAgent|consolidator)\.name$"
            )
        ),
        Field(description="Exact party-name path from compoundPartyFlavorRequirements."),
    ]
    targetPrimaryName: NonEmptyText
    renderedName: Annotated[
        str,
        StringConstraints(strip_whitespace=True, min_length=5, max_length=360),
        Field(
            description=(
                "Complete raw-text realization beginning with the exact target primary name, "
                "then preserving the source relationship with a distinct fictional identity. "
                "This auxiliary flavor never changes the target label."
            )
        ),
    ]


class AppliedCompoundPartyFlavorRealization(BaseModel):
    model_config = _STRICT

    targetPath: NonEmptyText
    relationships: tuple[Literal["on_behalf_of", "trading_as"], ...]
    targetPrimaryName: NonEmptyText
    renderedName: NonEmptyText


class CompoundPartyFlavorRequirement(BaseModel):
    model_config = _STRICT

    targetPath: NonEmptyText
    relationships: tuple[Literal["on_behalf_of", "trading_as"], ...] = Field(min_length=1)
    sourceLabelName: NonEmptyText
    targetPrimaryName: NonEmptyText


class LabelChangeDirective(BaseModel):
    """One authoritative model-facing semantic change between reviewed labels."""

    model_config = _STRICT

    path: NonEmptyText
    action: Literal[
        "replace",
        "deactivate_extractable_fact_preserve_slot",
        "add",
        "preserve_equivalent_equipment_surface",
        "replace_equipment_surface",
        "add_equipment_surface",
    ]
    sourceValue: JsonValue
    targetValue: JsonValue


class CompactLabelChangeDirective(BaseModel):
    """One deduplicated raw-text change shared by one or more schema paths."""

    model_config = _STRICT

    paths: Annotated[tuple[NonEmptyText, ...], Field(min_length=1)]
    action: Literal[
        "replace",
        "deactivate_extractable_fact_preserve_slot",
        "add",
        "preserve_equivalent_equipment_surface",
        "replace_equipment_surface",
        "add_equipment_surface",
    ]
    sourceValue: JsonValue
    targetValue: JsonValue


class SurfaceRenderingRequirement(BaseModel):
    """One exact document surface derived from a canonical target value."""

    model_config = _STRICT

    kind: Literal[
        "date",
        "hs_code",
        "hs_code_block",
        "carrier_header_identity",
        "carrier_receipt_count",
        "carrier_receipt_equipment_breakdown",
    ]
    targetPath: NonEmptyText
    sourceSurface: NonEmptyText
    targetSurface: NonEmptyText
    sourceOccurrences: Annotated[int, Field(ge=1)]
    contextEvidence: Annotated[str, StringConstraints(min_length=1, max_length=600)]


class SourceSemanticRoleHint(BaseModel):
    """A deterministic role for flattened OCR text that resembles another field."""

    model_config = _STRICT

    role: Literal["anonymous_equipment_count"]
    sourceLineId: Annotated[str, StringConstraints(pattern=r"^L[0-9]{5}$")]
    sourceSurface: NonEmptyText
    requiredOutputSurface: NonEmptyText
    sourceEvidence: Annotated[str, StringConstraints(min_length=1, max_length=600)]
    forbiddenTargetPathPrefixes: tuple[NonEmptyText, ...]


class InlineSlotTopologyRequirement(BaseModel):
    """One labeled inline source slot whose occupied state is part of the template."""

    model_config = _STRICT

    lineId: Annotated[str, StringConstraints(pattern=r"^L[0-9]{5}$")]
    ordinal: Annotated[int, Field(ge=0)]
    labelSurface: NonEmptyText
    sourcePopulated: bool


class SourceStatusPreservationRequirement(BaseModel):
    """One unchanged operational/document status that must remain byte-identical."""

    model_config = _STRICT

    sourceSurface: NonEmptyText
    sourceOccurrences: Annotated[int, Field(ge=1)]


class TargetLiteralRequirement(BaseModel):
    """One changed free-text target value that must occur in the rewritten OCR."""

    model_config = _STRICT

    targetPath: NonEmptyText
    targetValue: NonEmptyText


class TargetValueOccurrenceRequirement(BaseModel):
    """Exact occurrence count for a changed scalar whose duplication changes semantics."""

    model_config = _STRICT

    targetPaths: Annotated[tuple[NonEmptyText, ...], Field(min_length=1)]
    targetValue: NonEmptyText
    requiredOccurrences: Annotated[int, Field(ge=1)]


class AnchoredScalarReplacementRequirement(BaseModel):
    """Exact changed label scalar that must remain on every source line where it was printed."""

    model_config = _STRICT

    targetPaths: Annotated[tuple[NonEmptyText, ...], Field(min_length=1)]
    sourceLineIds: Annotated[tuple[NonEmptyText, ...], Field(min_length=1)]
    sourceSurface: NonEmptyText
    targetSurface: NonEmptyText
    surfaceKind: Literal["scalar", "measurement"] = "scalar"


class RawAuxiliaryIdentityRequirement(BaseModel):
    """A raw-only agency identity that must be anonymized without changing its relationship."""

    model_config = _STRICT

    requirementId: NonEmptyText
    relationship: Literal["agent_for_carrier"]
    sourceIdentity: NonEmptyText
    sourceIdentityLineCount: Annotated[int, Field(ge=1, le=3)]
    gapLineCount: Annotated[int, Field(ge=0, le=2)]
    consistencyGroupId: NonEmptyText
    targetPrincipalName: NonEmptyText
    sourceEvidence: EvidenceText


class OperationalFlavorRequirement(BaseModel):
    """One exact container-row fact derived from the synthetic shipment semantics."""

    model_config = _STRICT

    requirementId: NonEmptyText
    kind: Literal["gross_weight_kg", "volume_m3", "package_quantity"]
    sourceGrammar: Literal["labeled_measurement", "inline_container_breakdown"]
    sourceLineId: Annotated[str, StringConstraints(pattern=r"^L[0-9]{5}$")]
    sourceMeasurementStartColumn: Annotated[int, Field(ge=0)]
    sourceValueSurface: NonEmptyText
    targetValueSurface: NonEmptyText
    consistencyGroupId: NonEmptyText
    targetContainerNumber: NonEmptyText
    targetEquipmentFamily: EquipmentFamily
    maximumValue: NonEmptyText | None
    sameLineFollowingContainerNumber: NonEmptyText | None
    empiricalProfileDocumentId: NonEmptyText | None
    samplingMethod: Literal[
        "empirical_capacity_utilization_resample_v1",
        "target_measure_allocation_v1",
        "target_package_allocation_v1",
    ]
    sourceEvidence: EvidenceText


class CargoFlavorRewriteRequirement(BaseModel):
    """Every occupied OCR line belonging to one changed cargo-description block."""

    model_config = _STRICT

    requirementId: NonEmptyText
    targetPath: NonEmptyText
    targetDescription: NonEmptyText
    sourceLineIds: Annotated[tuple[NonEmptyText, ...], Field(min_length=1)]
    sourceSurfaces: Annotated[tuple[NonEmptyText, ...], Field(min_length=1)]

    @model_validator(mode="after")
    def line_ids_match_surfaces(self) -> CargoFlavorRewriteRequirement:
        if len(self.sourceLineIds) != len(self.sourceSurfaces):
            raise ValueError("cargo-flavor source line IDs and surfaces must have equal length")
        line_numbers = tuple(int(line_id[1:]) for line_id in self.sourceLineIds)
        if line_numbers != tuple(sorted(set(line_numbers))):
            raise ValueError("cargo-flavor source line IDs must be unique and ascending")
        return self


class AppliedRawAuxiliaryIdentityRealization(BaseModel):
    model_config = _STRICT

    requirementId: NonEmptyText
    relationship: Literal["agent_for_carrier"]
    sourceIdentity: NonEmptyText
    sourceIdentityLineCount: Annotated[int, Field(ge=1, le=3)]
    gapLineCount: Annotated[int, Field(ge=0, le=2)]
    consistencyGroupId: NonEmptyText
    targetPrincipalName: NonEmptyText
    renderedIdentity: NonEmptyText


class TargetIntegrityChange(BaseModel):
    """One audited repair applied before the raw-text model sees a target."""

    model_config = _STRICT

    path: NonEmptyText
    reason: Literal[
        "special_use_domain",
        "country_incoherent_cctld",
        "package_surface_category_mismatch",
    ]
    before: NonEmptyText
    after: NonEmptyText


class RecoveredExplicitHsTargetFact(BaseModel):
    """One task-extractable HS value restored from pinned semantic-plan provenance."""

    model_config = _STRICT

    targetPath: NonEmptyText
    sourceEvidenceCodes: Annotated[tuple[NonEmptyText, ...], Field(min_length=1)]
    semanticPlanPath: NonEmptyText
    hs6: Annotated[str, StringConstraints(pattern=r"^[0-9]{6}$")]
    renderedCode: Annotated[str, StringConstraints(pattern=r"^[0-9]{6,18}$")]
    outputDigits: Annotated[int, Field(ge=6, le=18)]
    method: Literal["explicit_raw_hs_slot_from_pinned_semantic_plan_v1"]


class CustomsProgramEntry(BaseModel):
    """One authoritative jurisdiction-bound document-program surface."""

    model_config = _STRICT

    program_id: Annotated[str, StringConstraints(pattern=r"^[a-z][a-z0-9_]+$")]
    official_name: NonEmptyText
    authority: NonEmptyText
    official_source_url: Annotated[str, StringConstraints(pattern=r"^https://[^\s]+$")]
    jurisdiction_country_code: Annotated[str, StringConstraints(pattern=r"^[A-Z]{2}$")]
    trade_direction: Literal["import", "export"]
    source_surfaces: Annotated[tuple[NonEmptyText, ...], Field(min_length=1)]
    generic_replacement_surface: NonEmptyText

    @model_validator(mode="after")
    def surfaces_are_unique_and_longest_first(self) -> CustomsProgramEntry:
        normalized = tuple(_semantic_normalize(value) for value in self.source_surfaces)
        if any(not value for value in normalized) or len(normalized) != len(set(normalized)):
            raise ValueError("customs-program source surfaces must be nonempty and unique")
        if tuple(sorted(self.source_surfaces, key=len, reverse=True)) != self.source_surfaces:
            raise ValueError("customs-program source surfaces must be longest-first")
        return self


class CustomsProgramRegistry(BaseModel):
    model_config = _STRICT

    schema_version: Literal[1]
    entries: Annotated[tuple[CustomsProgramEntry, ...], Field(min_length=1)]

    @model_validator(mode="after")
    def entries_are_unique(self) -> CustomsProgramRegistry:
        identities = tuple(row.program_id for row in self.entries)
        if len(identities) != len(set(identities)):
            raise ValueError("customs-program registry IDs must be unique")
        surfaces = [
            (row.trade_direction, _semantic_normalize(surface))
            for row in self.entries
            for surface in row.source_surfaces
        ]
        if len(surfaces) != len(set(surfaces)):
            raise ValueError("customs-program surfaces overlap within a trade direction")
        return self


class JurisdictionalSurfaceRequirement(BaseModel):
    """A stale named customs program that must be generalized for a new route."""

    model_config = _STRICT

    programId: NonEmptyText
    tradeDirection: Literal["import", "export"]
    programJurisdictionCountryCode: Annotated[str, StringConstraints(pattern=r"^[A-Z]{2}$")]
    targetRouteCountryCode: Annotated[str, StringConstraints(pattern=r"^[A-Z]{2}$")]
    sourceSurface: NonEmptyText
    targetSurface: NonEmptyText
    sourceOccurrences: Annotated[int, Field(ge=1)]
    authority: NonEmptyText
    officialSourceUrl: Annotated[str, StringConstraints(pattern=r"^https://[^\s]+$")]


@dataclass(frozen=True, slots=True)
class TargetIntegrityResources:
    countries: CountryRegistry
    packages: LoadedPackageRegistry
    route_countries_by_name: Mapping[str, frozenset[str]]
    customs_programs: tuple[CustomsProgramEntry, ...]
    operational_profiles: tuple[EmpiricalOperationalProfile, ...] = ()


@dataclass(frozen=True, slots=True)
class ParsedMeasurementSurface:
    value: Decimal
    decimal_separator: str | None
    grouping_separator: str | None
    decimal_places: int


@dataclass(frozen=True, slots=True)
class EmpiricalOperationalProfile:
    document_id: str
    container_number: str
    equipment_family: EquipmentFamily
    gross_weight_kg: Decimal | None
    gross_utilization: Decimal | None
    volume_m3: Decimal | None
    volume_utilization: Decimal | None


class AtomicRewriteCommit(BaseModel):
    model_config = _STRICT

    beforeTextSha256: Sha256
    afterTextSha256: Sha256
    beforeTargetLabelSha256: Sha256
    afterTargetLabelSha256: Sha256
    compoundPartyFlavorRealizations: tuple[AppliedCompoundPartyFlavorRealization, ...]
    rawAuxiliaryIdentityRealizations: tuple[AppliedRawAuxiliaryIdentityRealization, ...]
    appliedReplacements: tuple[AppliedLineRangeReplacement, ...]


class ResidualCandidate(BaseModel):
    model_config = _STRICT

    targetPath: NonEmptyText
    sourceValue: JsonValue
    currentOccurrences: Annotated[int, Field(ge=1)]
    currentEvidence: EvidenceText


class DeterministicRewriteAudit(BaseModel):
    model_config = _STRICT

    currentTextSha256: Sha256
    textChanged: bool
    pageMarkersPreserved: bool
    newlineConventionPreserved: bool
    lineCountPreserved: bool
    blankLineTopologyPreserved: bool
    inlineSlotTopologyPreserved: bool
    structuralLinePrefixesPreserved: bool
    isolatedFormattingChangesAbsent: bool
    sourceStatusSurfacesPreserved: bool
    blankLineDepthBounded: bool
    noPlaceholderIntroduced: bool
    requiredSurfacesRendered: bool
    jurisdictionalSurfacesRendered: bool
    requiredTargetLiteralsRendered: bool
    targetValueOccurrencesExact: bool
    anchoredScalarReplacementsRendered: bool
    rawAuxiliaryIdentitiesReplaced: bool
    operationalFlavorRendered: bool
    deactivatedReeferOperationCoherent: bool
    cargoFlavorRewritten: bool
    residualCandidates: tuple[ResidualCandidate, ...]

    @property
    def core_passed(self) -> bool:
        return all(
            (
                self.textChanged,
                self.pageMarkersPreserved,
                self.newlineConventionPreserved,
                self.lineCountPreserved,
                self.blankLineTopologyPreserved,
                self.inlineSlotTopologyPreserved,
                self.structuralLinePrefixesPreserved,
                self.isolatedFormattingChangesAbsent,
                self.sourceStatusSurfacesPreserved,
                self.blankLineDepthBounded,
                self.noPlaceholderIntroduced,
                self.requiredSurfacesRendered,
                self.jurisdictionalSurfacesRendered,
                self.requiredTargetLiteralsRendered,
                self.targetValueOccurrencesExact,
                self.anchoredScalarReplacementsRendered,
                self.rawAuxiliaryIdentitiesReplaced,
                self.operationalFlavorRendered,
                self.deactivatedReeferOperationCoherent,
                self.cargoFlavorRewritten,
            )
        )


class SemanticReviewFinding(BaseModel):
    model_config = _STRICT

    category: Literal[
        "missing_or_wrong_target_fact",
        "stale_source_fact",
        "incomplete_auxiliary_anonymization",
        "incoherent_synthetic_flavor",
        "derived_fact_mismatch",
        "legal_relationship_damage",
        "format_or_layout_damage",
        "unnecessary_change",
    ]
    affectedPaths: Annotated[tuple[NonEmptyText, ...], Field(min_length=1, max_length=8)]
    currentEvidence: Annotated[tuple[EvidenceText, ...], Field(min_length=1, max_length=4)]
    correction: Annotated[
        str, StringConstraints(strip_whitespace=True, min_length=1, max_length=800)
    ]


class SemanticReviewReceipt(BaseModel):
    """Compact independent judgment over a complete rewritten OCR sample."""

    model_config = _STRICT

    verdict: Literal["pass", "revise"]
    targetFacts: Literal["pass", "fail"]
    staleSourceFacts: Literal["pass", "fail"]
    auxiliaryFlavor: Literal["pass", "fail"]
    cargoAndOperationalRealism: Literal["pass", "fail"]
    legalTopology: Literal["pass", "fail"]
    formattingAndMinimality: Literal["pass", "fail"]
    findings: Annotated[tuple[SemanticReviewFinding, ...], Field(max_length=24)]

    @model_validator(mode="after")
    def verdict_matches_findings(self) -> SemanticReviewReceipt:
        checks = (
            self.targetFacts,
            self.staleSourceFacts,
            self.auxiliaryFlavor,
            self.cargoAndOperationalRealism,
            self.legalTopology,
            self.formattingAndMinimality,
        )
        if self.verdict == "pass" and (self.findings or any(row != "pass" for row in checks)):
            raise ValueError("pass requires every checklist item to pass and no findings")
        if self.verdict == "revise" and (not self.findings or all(row == "pass" for row in checks)):
            raise ValueError("revise requires a finding and at least one failed checklist item")
        return self


class RewriteStageRecord(BaseModel):
    model_config = _STRICT

    schemaVersion: Literal[15]
    sequence: Annotated[int, Field(ge=1)]
    stage: Literal["draft_editor", "correction_editor", "semantic_reviewer"]
    correctionCycle: Annotated[int, Field(ge=0, le=3)]
    providerModel: NonEmptyText
    reasoningEffort: NonEmptyText
    startedAt: datetime
    completedAt: datetime
    inputPayloadSha256: Sha256
    currentTextSha256Before: Sha256
    currentTextSha256After: Sha256
    editorCommit: AtomicRewriteCommit | None
    semanticReview: SemanticReviewReceipt | None
    deterministicAudit: DeterministicRewriteAudit | None
    usage: LinguisticUsageReceipt
    modelMessages: JsonValue
    errorType: NonEmptyText | None
    errorMessage: NonEmptyText | None


class RewriteCycleResult(BaseModel):
    model_config = _STRICT

    schemaVersion: Literal[15]
    caseNumber: Annotated[int, Field(ge=1, le=10)]
    documentId: NonEmptyText
    scenarioId: NonEmptyText
    status: Literal["quality_validated", "needs_review", "call_failed", "cost_limited"]
    reason: NonEmptyText
    sourceTextSha256: Sha256
    outputTextSha256: Sha256
    sourceLabelSha256: Sha256
    upstreamTargetLabelSha256: Sha256
    targetLabelSha256: Sha256
    targetIntegritySha256: Sha256
    targetCapacityReprojected: bool
    changedLeavesSha256: Sha256
    labelChangeContractSha256: Sha256
    editorOutputSchemaSha256: Sha256
    reviewerOutputSchemaSha256: Sha256
    correctionCycles: Annotated[int, Field(ge=0, le=3)]
    editorPasses: Annotated[int, Field(ge=0)]
    reviewPasses: Annotated[int, Field(ge=0)]
    compoundPartyFlavorRealizations: Annotated[int, Field(ge=0)]
    rawAuxiliaryIdentityRealizations: Annotated[int, Field(ge=0)]
    deterministicPrefills: Annotated[int, Field(ge=0)]
    replacements: Annotated[int, Field(ge=0)]
    sourceLinesReplaced: Annotated[int, Field(ge=0)]
    outputLinesInserted: Annotated[int, Field(ge=0)]
    durationMs: Annotated[float, Field(ge=0)]
    totalUsage: LinguisticUsageReceipt


class RewriteCycleBundle(BaseModel):
    model_config = _STRICT

    schemaVersion: Literal[15]
    result: RewriteCycleResult
    feature: dict[str, JsonValue]
    sourceLabel: dict[str, JsonValue]
    upstreamTargetLabel: dict[str, JsonValue]
    targetLabel: dict[str, JsonValue]
    targetIntegrity: dict[str, JsonValue]
    changedLeaves: tuple[ChangedLeaf, ...]
    labelChangeContract: tuple[LabelChangeDirective, ...]
    compactLabelChangeContract: tuple[CompactLabelChangeDirective, ...]
    surfaceRenderingRequirements: tuple[SurfaceRenderingRequirement, ...]
    targetLiteralRequirements: tuple[TargetLiteralRequirement, ...]
    targetValueOccurrenceRequirements: tuple[TargetValueOccurrenceRequirement, ...]
    anchoredScalarReplacementRequirements: tuple[AnchoredScalarReplacementRequirement, ...]
    sourceSemanticRoleHints: tuple[SourceSemanticRoleHint, ...]
    inlineSlotTopologyRequirements: tuple[InlineSlotTopologyRequirement, ...]
    sourceStatusPreservationRequirements: tuple[SourceStatusPreservationRequirement, ...]
    compoundPartyFlavorRequirements: tuple[CompoundPartyFlavorRequirement, ...]
    rawAuxiliaryIdentityRequirements: tuple[RawAuxiliaryIdentityRequirement, ...]
    operationalFlavorRequirements: tuple[OperationalFlavorRequirement, ...]
    cargoFlavorRewriteRequirements: tuple[CargoFlavorRewriteRequirement, ...]
    jurisdictionalSurfaceRequirements: tuple[JurisdictionalSurfaceRequirement, ...]
    sourceText: str
    outputText: str
    deterministicPrefills: tuple[AppliedDeterministicPrefill, ...]
    appliedReplacements: tuple[AppliedLineRangeReplacement, ...]
    deterministicAudits: tuple[DeterministicRewriteAudit, ...]
    semanticReviews: tuple[SemanticReviewReceipt, ...]
    stageRecords: tuple[RewriteStageRecord, ...]
    diff: str


@dataclass(slots=True)
class RewriteWorkspace:
    original_text: str
    current_text: str
    scenario_id: str = "test-scenario"
    source_label: dict[str, JsonValue] = field(default_factory=dict)
    upstream_target_label: dict[str, JsonValue] = field(default_factory=dict)
    current_target_label: dict[str, JsonValue] = field(default_factory=dict)
    surface_requirements: tuple[SurfaceRenderingRequirement, ...] = ()
    target_literal_requirements: tuple[TargetLiteralRequirement, ...] = ()
    target_value_occurrence_requirements: tuple[TargetValueOccurrenceRequirement, ...] = ()
    anchored_scalar_replacement_requirements: tuple[AnchoredScalarReplacementRequirement, ...] = ()
    source_role_hints: tuple[SourceSemanticRoleHint, ...] = ()
    inline_slot_requirements: tuple[InlineSlotTopologyRequirement, ...] = ()
    source_status_requirements: tuple[SourceStatusPreservationRequirement, ...] = ()
    jurisdictional_requirements: tuple[JurisdictionalSurfaceRequirement, ...] = ()
    raw_auxiliary_identity_requirements: tuple[RawAuxiliaryIdentityRequirement, ...] = ()
    operational_flavor_requirements: tuple[OperationalFlavorRequirement, ...] = ()
    cargo_flavor_rewrite_requirements: tuple[CargoFlavorRewriteRequirement, ...] = ()
    deterministic_prefills: list[AppliedDeterministicPrefill] = field(default_factory=list)
    commits: list[AtomicRewriteCommit] = field(default_factory=list)

    @property
    def current_sha256(self) -> str:
        return sha256_bytes(self.current_text.encode("utf-8"))

    @property
    def current_target_sha256(self) -> str:
        return sha256_bytes(canonical_json_bytes(self.current_target_label))


@dataclass(slots=True)
class RewriteState:
    case_number: int
    document_id: str
    scenario_id: str
    feature: dict[str, JsonValue]
    source_label: dict[str, JsonValue]
    target_integrity: dict[str, JsonValue]
    workspace: RewriteWorkspace
    started_at: datetime
    stages: list[RewriteStageRecord] = field(default_factory=list)
    audits: list[DeterministicRewriteAudit] = field(default_factory=list)
    reviews: list[SemanticReviewReceipt] = field(default_factory=list)

    @property
    def target_label(self) -> dict[str, JsonValue]:
        return self.workspace.current_target_label

    @property
    def changed_leaves(self) -> tuple[ChangedLeaf, ...]:
        return rewrite_changed_leaves(self.source_label, self.target_label)


def _page_markers(value: str) -> tuple[str, ...]:
    return tuple(line for line in value.splitlines() if _PAGE_MARKER.fullmatch(line))


def _newline_convention(value: str) -> frozenset[str]:
    return frozenset(match.group(0) for match in _NEWLINE.finditer(value))


def _all_occurrences(text: str, needle: str) -> tuple[int, ...]:
    offsets: list[int] = []
    cursor = 0
    while True:
        offset = text.find(needle, cursor)
        if offset < 0:
            return tuple(offsets)
        offsets.append(offset)
        cursor = offset + len(needle)


def _evidence_occurs(evidence: str, text: str) -> bool:
    if re.search(
        rf"(?<![A-Za-z0-9]){re.escape(evidence)}(?![A-Za-z0-9])",
        text,
    ) is not None:
        return True
    normalized_evidence = _semantic_normalize(evidence)
    if not normalized_evidence:
        return False
    normalized_text = _semantic_normalize(text)
    return (
        re.search(
            rf"(?<![A-Z0-9]){re.escape(normalized_evidence)}(?![A-Z0-9])",
            normalized_text,
        )
        is not None
    )


def _max_consecutive_blank_lines(value: str) -> int:
    maximum = current = 0
    for line in value.splitlines():
        if line.strip():
            current = 0
        else:
            current += 1
            maximum = max(maximum, current)
    return maximum


def _blank_line_topology(value: str) -> tuple[bool, ...]:
    """Return the source template's occupied-versus-blank line structure."""

    return tuple(not line.strip() for line in value.splitlines())


def _inline_slots(line: str) -> tuple[tuple[str, bool], ...]:
    matches = tuple(_INLINE_SLOT_LABEL.finditer(line))
    slots: list[tuple[str, bool]] = []
    for index, match in enumerate(matches):
        end = matches[index + 1].start() if index + 1 < len(matches) else len(line)
        value = line[match.end() : end]
        slots.append((match.group(0), re.search(r"[A-Z0-9]", value, re.I) is not None))
    return tuple(slots)


def inline_slot_topology_requirements(raw_text: str) -> tuple[InlineSlotTopologyRequirement, ...]:
    """Describe inline contact slots without exposing a guessed semantic value."""

    requirements: list[InlineSlotTopologyRequirement] = []
    for line_number, line in enumerate(raw_text.splitlines(), start=1):
        for ordinal, (label, populated) in enumerate(_inline_slots(line)):
            requirements.append(
                InlineSlotTopologyRequirement(
                    lineId=_line_id(line_number, line),
                    ordinal=ordinal,
                    labelSurface=label,
                    sourcePopulated=populated,
                )
            )
    return tuple(requirements)


def _inline_slot_topology_preserved(source: str, output: str) -> bool:
    source_lines = source.splitlines()
    output_lines = output.splitlines()
    return len(source_lines) == len(output_lines) and all(
        _inline_slots(source_line) == _inline_slots(output_line)
        for source_line, output_line in zip(source_lines, output_lines, strict=True)
    )


def _structural_line_prefixes_preserved(source: str, output: str) -> bool:
    """Protect numbered clause/field prefixes while allowing their values to change."""

    source_lines = source.splitlines()
    output_lines = output.splitlines()
    if len(source_lines) != len(output_lines):
        return False
    for source_line, output_line in zip(source_lines, output_lines, strict=True):
        source_match = _STRUCTURAL_LINE_PREFIX.match(source_line)
        if source_match is None:
            continue
        output_match = _STRUCTURAL_LINE_PREFIX.match(output_line)
        if output_match is None or output_match.group("prefix") != source_match.group("prefix"):
            return False
    return True


def _isolated_formatting_changes(
    workspace: RewriteWorkspace, output: str
) -> tuple[tuple[int, str, str], ...]:
    """Find punctuation/whitespace edits isolated from any lexical value change.

    Synthetic rendering may replace words, identifiers, and numbers, but it must not
    opportunistically clean up punctuation elsewhere on the same mostly-preserved line.
    Fully regenerated party and cargo values are exempt. A formatting-only diff is
    legitimate only when no unchanged
    alphanumeric token separates it from a lexical value edit.
    """

    source_lines = workspace.original_text.splitlines()
    output_lines = output.splitlines()
    if len(source_lines) != len(output_lines):
        return ((0, "line-count-mismatch", "line-count-mismatch"),)
    isolated: list[tuple[int, str, str]] = []
    for line_number, (source_line, output_line) in enumerate(
        zip(source_lines, output_lines, strict=True), start=1
    ):
        if source_line == output_line:
            continue
        source_comparison = source_line
        output_comparison = output_line
        controlled_value_line = False
        for leaf in rewrite_changed_leaves(
            workspace.source_label, workspace.current_target_label
        ):
            if not isinstance(leaf.sourceValue, str) or not isinstance(leaf.targetValue, str):
                continue
            if (
                len(_semantic_normalize(leaf.sourceValue)) >= 4
                and leaf.sourceValue in source_comparison
            ) or (
                len(_semantic_normalize(leaf.targetValue)) >= 4
                and leaf.targetValue in output_comparison
            ):
                controlled_value_line = True
                break
            if leaf.sourceValue in source_comparison:
                source_comparison = source_comparison.replace(leaf.sourceValue, "VALUE")
            if leaf.targetValue in output_comparison:
                output_comparison = output_comparison.replace(leaf.targetValue, "VALUE")
        if controlled_value_line:
            continue
        matcher = SequenceMatcher(None, source_comparison, output_comparison, autojunk=False)
        opcodes = matcher.get_opcodes()
        equal_alphanumeric = sum(
            sum(character.isalnum() for character in source_comparison[source_start:source_end])
            for tag, source_start, source_end, _, _ in opcodes
            if tag == "equal"
        )
        source_alphanumeric = sum(character.isalnum() for character in source_comparison)
        output_alphanumeric = sum(character.isalnum() for character in output_comparison)
        # A line that remains at least 90% the same is an existing prose template with sparse
        # value slots. The integer ratio keeps this check deterministic across Python versions.
        # Completely regenerated party/cargo values may legitimately use different punctuation.
        if (
            equal_alphanumeric * 10 < source_alphanumeric * 9
            or equal_alphanumeric * 10 < output_alphanumeric * 9
        ):
            continue
        changed = [row for row in opcodes if row[0] != "equal"]
        lexical = [
            row
            for row in changed
            if re.search(
                r"[A-Za-z0-9]",
                source_comparison[row[1] : row[2]] + output_comparison[row[3] : row[4]],
            )
        ]
        for _, source_start, source_end, output_start, output_end in changed:
            before = source_comparison[source_start:source_end]
            after = output_comparison[output_start:output_end]
            if re.search(r"[A-Za-z0-9]", before + after):
                continue
            associated = any(
                not re.search(
                    r"[A-Za-z0-9]",
                    source_comparison[
                        min(source_end, row[2]) : max(source_start, row[1])
                    ]
                    + output_comparison[
                        min(output_end, row[4]) : max(output_start, row[3])
                    ],
                )
                for row in lexical
            )
            if not associated:
                isolated.append((line_number, before, after))
    return tuple(isolated)


def _label_has_temperature_setpoint(label: Mapping[str, Any]) -> bool:
    patch = label.get("documentPatch")
    containers = patch.get("containers") if isinstance(patch, Mapping) else None
    return isinstance(containers, list) and any(
        isinstance(container, Mapping) and container.get("temperatureSetpoint") is not None
        for container in containers
    )


def _deactivated_reefer_operation_coherent(workspace: RewriteWorkspace, output: str) -> bool:
    """Reject positive source operation claims after every target setpoint is removed."""

    if _NON_OPERATING_REEFER.search(output) is not None and re.search(
        r"\bPLUGGING[ \t]+FOR[ \t]+(?:THE[ \t]+)?ACCOUNT\b", output, re.IGNORECASE
    ) is not None:
        return False
    source_has_setpoint = _label_has_temperature_setpoint(workspace.source_label)
    target_has_setpoint = _label_has_temperature_setpoint(workspace.current_target_label)
    if not source_has_setpoint or target_has_setpoint:
        return True
    source_lines = workspace.original_text.splitlines()
    output_lines = output.splitlines()
    if len(source_lines) != len(output_lines):
        return False
    return all(
        _POSITIVE_REEFER_OPERATION.search(output_line) is None
        for source_line, output_line in zip(source_lines, output_lines, strict=True)
        if _POSITIVE_REEFER_OPERATION.search(source_line) is not None
    )


def source_status_preservation_requirements(
    raw_text: str, leaves: Sequence[ChangedLeaf]
) -> tuple[SourceStatusPreservationRequirement, ...]:
    """Protect unchanged generic status lines while allowing target-controlled changes."""

    changed_source_surfaces = tuple(
        _semantic_normalize(leaf.sourceValue)
        for leaf in leaves
        if leaf.requiresTextEdit
        and leaf.sourcePresent
        and isinstance(leaf.sourceValue, str)
        and leaf.sourceValue != leaf.targetValue
    )
    legal_document_semantics_changed = any(
        leaf.requiresTextEdit
        and leaf.sourceValue != leaf.targetValue
        and leaf.path.rsplit(".", 1)[-1] in {"negotiability", "numberOfOriginals"}
        for leaf in leaves
    )
    counts: dict[str, int] = {}
    for line in raw_text.splitlines():
        status_line = _PRESERVABLE_STATUS_LINE.fullmatch(line) is not None
        legal_line = (
            not legal_document_semantics_changed
            and _PRESERVABLE_LEGAL_BOILERPLATE_LINE.fullmatch(line) is not None
        )
        if not status_line and not legal_line:
            continue
        normalized = _semantic_normalize(line)
        if any(surface and surface in normalized for surface in changed_source_surfaces):
            continue
        counts[line] = counts.get(line, 0) + 1
    return tuple(
        SourceStatusPreservationRequirement(
            sourceSurface=surface,
            sourceOccurrences=occurrences,
        )
        for surface, occurrences in sorted(counts.items())
    )


def _source_status_surfaces_preserved(
    value: str, requirements: Sequence[SourceStatusPreservationRequirement]
) -> bool:
    lines = value.splitlines()
    return all(lines.count(row.sourceSurface) == row.sourceOccurrences for row in requirements)


def _placeholder_count(value: str) -> int:
    return sum(1 for _ in _PLACEHOLDER.finditer(value))


def _line_body(value: str) -> str:
    if value.endswith("\r\n"):
        return value[:-2]
    if value.endswith(("\r", "\n")):
        return value[:-1]
    return value


def _line_ending(value: str) -> str:
    if value.endswith("\r\n"):
        return "\r\n"
    if value.endswith("\r"):
        return "\r"
    if value.endswith("\n"):
        return "\n"
    return ""


def _line_id(index: int, value: str) -> str:
    del value
    return f"L{index:05d}"


def _line_number(line_id: str) -> int:
    return int(line_id[1:6])


def _default_newline(value: str) -> str:
    conventions = _newline_convention(value)
    if len(conventions) > 1:
        raise ValueError("mixed newline conventions are not supported by line-range editing")
    return next(iter(conventions), "\n")


def indexed_ocr_lines(value: str) -> str:
    """Return compact line addresses without changing the underlying OCR."""

    return "\n".join(
        f"{_line_id(index, line)}|{_line_body(line)}"
        for index, line in enumerate(value.splitlines(keepends=True), start=1)
    )


def editable_indexed_ocr_lines(value: str) -> str:
    """Expose only occupied, non-marker lines while retaining absolute line addresses.

    Blank lines and page markers are immutable template structure. Omitting them from the editor's
    model-visible workspace prevents accidental selection without hiding adjacency: gaps in the
    absolute IDs show exactly where protected lines occur.
    """

    return "\n".join(
        f"{_line_id(index, line)}|{body}"
        for index, line in enumerate(value.splitlines(keepends=True), start=1)
        if (body := _line_body(line)).strip() and _PAGE_MARKER.fullmatch(body) is None
    )


def _semantic_normalize(value: str) -> str:
    ascii_value = unicodedata.normalize("NFKD", value).encode("ascii", "ignore").decode()
    return " ".join(re.findall(r"[A-Z0-9]+", ascii_value.upper()))


def _party_scalar_occurrence_count(text: str, surface: str) -> int:
    """Count printed party scalars without mistaking email/URL tokens for identities.

    Party names commonly reappear in email domains (for example, ``MLH SHIPPING`` and
    ``ops@mlh-shipping.com``).  Domain tokens are contact metadata, not an additional printed
    party occurrence.  Remove only those electronic-contact spans, then count the complete scalar
    with tolerant OCR punctuation boundaries across the document. Counting after whitespace
    normalization also recognizes a name or address that the source OCR wraps over several lines.
    """

    normalized_surface = _semantic_normalize(surface)
    if not normalized_surface:
        return 0
    pattern = re.compile(
        rf"(?<![A-Z0-9]){re.escape(normalized_surface)}(?![A-Z0-9])"
    )
    normalized_text = _semantic_normalize(_ELECTRONIC_CONTACT_SURFACE.sub(" ", text))
    return sum(1 for _ in pattern.finditer(normalized_text))


_CARGO_MATCH_STOPWORDS = frozenset(
    {
        "AND",
        "AS",
        "BY",
        "CARGO",
        "DESCRIPTION",
        "FOR",
        "FROM",
        "GOODS",
        "IN",
        "ITEM",
        "OF",
        "ON",
        "OR",
        "PRODUCT",
        "PRODUCTS",
        "THE",
        "TO",
        "WITH",
    }
)
_CARGO_QUANTITY_CONTINUATION = re.compile(
    r"^[ \t]*[0-9][0-9., /-]*[ \t]+(?:EA|PCS?|PIECES?|PKGS?|PACKAGES?|"
    r"BAGS?|BALES?|BOXES?|CARTONS?|CASES?|CRATES?|DRUMS?|PALLETS?|ROLLS?|SETS?|"
    r"UNITS?)[ \t]*$",
    re.IGNORECASE,
)


def _cargo_match_tokens(value: str) -> tuple[str, ...]:
    return tuple(
        token
        for token in _semantic_normalize(value).split()
        if token not in _CARGO_MATCH_STOPWORDS
        and len(token) >= 3
        and any("A" <= ch <= "Z" for ch in token)
    )


def _line_matches_cargo_description(line: str, source_description: str) -> bool:
    """Match a source-description fragment without relying on a document-template regex."""

    line_normalized = _semantic_normalize(line)
    source_normalized = _semantic_normalize(source_description)
    if not line_normalized or not source_normalized:
        return False
    if len(source_normalized) >= 4 and (
        source_normalized in line_normalized or line_normalized in source_normalized
    ):
        return True
    line_tokens = _cargo_match_tokens(line)
    source_tokens = _cargo_match_tokens(source_description)
    if not line_tokens or not source_tokens:
        return False
    shared = set(line_tokens) & set(source_tokens)
    if len(shared) >= 2:
        return True
    source_bigrams = set(pairwise(source_tokens))
    line_bigrams = set(pairwise(line_tokens))
    if source_bigrams & line_bigrams:
        return True
    longest = SequenceMatcher(
        None, line_normalized, source_normalized, autojunk=False
    ).find_longest_match()
    return longest.size >= 10 and longest.size / len(line_normalized) >= 0.60


def _cargo_cluster_score(
    lines: Sequence[str], source_description: str, start: int, end: int
) -> float:
    candidate = " ".join(lines[start : end + 1])
    source_normalized = _semantic_normalize(source_description)
    candidate_normalized = _semantic_normalize(candidate)
    source_tokens = set(_cargo_match_tokens(source_description))
    candidate_tokens = set(_cargo_match_tokens(candidate))
    shared = source_tokens & candidate_tokens
    precision = len(shared) / len(candidate_tokens) if candidate_tokens else 0.0
    recall = len(shared) / len(source_tokens) if source_tokens else 0.0
    token_f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    character_ratio = SequenceMatcher(
        None, candidate_normalized, source_normalized, autojunk=False
    ).ratio()
    containment = float(
        source_normalized in candidate_normalized or candidate_normalized in source_normalized
    )
    return (0.55 * token_f1) + (0.35 * character_ratio) + (0.10 * containment)


def _cargo_candidate_clusters(
    lines: Sequence[str], source_description: str
) -> tuple[tuple[int, int, float], ...]:
    hits = [
        index
        for index, line in enumerate(lines)
        if line.strip()
        and _PAGE_MARKER.fullmatch(line) is None
        and _line_matches_cargo_description(line, source_description)
    ]
    clusters: list[list[int]] = []
    for hit in hits:
        if (
            not clusters
            or hit - clusters[-1][-1] > 3
            or any(not lines[index].strip() for index in range(clusters[-1][-1] + 1, hit))
            or any(
                _PAGE_MARKER.fullmatch(lines[index]) is not None
                for index in range(clusters[-1][-1] + 1, hit)
            )
        ):
            clusters.append([hit])
        else:
            clusters[-1].append(hit)
    candidates: list[tuple[int, int, float]] = []
    for cluster in clusters:
        start = cluster[0]
        end = cluster[-1]
        while start > 0 and _CARGO_QUANTITY_CONTINUATION.fullmatch(lines[start - 1]):
            start -= 1
        while end + 1 < len(lines) and _CARGO_QUANTITY_CONTINUATION.fullmatch(lines[end + 1]):
            end += 1
        candidates.append((start, end, _cargo_cluster_score(lines, source_description, start, end)))
    return tuple(candidates)


def _cargo_group_description_pairs(
    source_label: Mapping[str, Any], target_label: Mapping[str, Any]
) -> tuple[tuple[int, str, str], ...]:
    source_patch = source_label.get("documentPatch")
    target_patch = target_label.get("documentPatch")
    if not isinstance(source_patch, Mapping) or not isinstance(target_patch, Mapping):
        return ()
    source_groups = source_patch.get("cargoGroups")
    target_groups = target_patch.get("cargoGroups")
    if not isinstance(source_groups, Sequence) or isinstance(source_groups, (str, bytes)):
        return ()
    if not isinstance(target_groups, Sequence) or isinstance(target_groups, (str, bytes)):
        return ()
    target_by_group_id = {
        row.get("groupId"): row
        for row in target_groups
        if isinstance(row, Mapping) and isinstance(row.get("groupId"), str)
    }
    pairs: list[tuple[int, str, str]] = []
    for index, source_group in enumerate(source_groups):
        if not isinstance(source_group, Mapping):
            continue
        group_id = source_group.get("groupId")
        target_group = target_by_group_id.get(group_id)
        if target_group is None and index < len(target_groups):
            candidate = target_groups[index]
            target_group = candidate if isinstance(candidate, Mapping) else None
        if target_group is None:
            continue
        source_description = source_group.get("description")
        target_description = target_group.get("description")
        if not isinstance(source_description, str) or not isinstance(target_description, str):
            continue
        if _semantic_normalize(source_description) == _semantic_normalize(target_description):
            continue
        pairs.append((index, source_description, target_description))
    return tuple(pairs)


def cargo_flavor_rewrite_requirements(
    raw_text: str,
    source_label: Mapping[str, Any],
    target_label: Mapping[str, Any],
) -> tuple[CargoFlavorRewriteRequirement, ...]:
    """Locate complete source cargo-description spans that must be rewritten.

    Reviewed labels commonly concatenate a visual cargo table while the OCR keeps product rows and
    quantity continuations on separate lines. Strong lexical anchors locate the rows; only occupied
    intervening slots are included. This prevents an editor from changing the headline while
    leaving a trailing source product row or quantity behind.
    """

    lines = raw_text.splitlines()
    pairs = _cargo_group_description_pairs(source_label, target_label)
    candidates_by_group = {
        group_index: _cargo_candidate_clusters(lines, source_description)
        for group_index, source_description, _ in pairs
    }
    assigned: dict[int, tuple[int, int]] = {}
    claimed: set[tuple[int, int]] = set()

    repeated_descriptions: dict[str, list[int]] = {}
    for group_index, source_description, _ in pairs:
        repeated_descriptions.setdefault(_semantic_normalize(source_description), []).append(
            group_index
        )
    for group_indices in repeated_descriptions.values():
        if len(group_indices) < 2:
            continue
        available = sorted(
            {
                (start, end)
                for group_index in group_indices
                for start, end, _ in candidates_by_group[group_index]
            }
        )
        for group_index, span in zip(sorted(group_indices), available, strict=False):
            assigned[group_index] = span
            claimed.add(span)

    scored_options = sorted(
        (
            (score, group_index, start, end)
            for group_index, candidates in candidates_by_group.items()
            if group_index not in assigned
            for start, end, score in candidates
        ),
        key=lambda row: (-row[0], row[1], row[2], row[3]),
    )
    for _, group_index, start, end in scored_options:
        span = (start, end)
        if group_index in assigned or span in claimed:
            continue
        assigned[group_index] = span
        claimed.add(span)

    target_description_by_group = {
        group_index: target_description for group_index, _, target_description in pairs
    }
    requirements: list[CargoFlavorRewriteRequirement] = []
    for group_index, (start, end) in sorted(assigned.items()):
        source_indices = tuple(
            index
            for index in range(start, end + 1)
            if lines[index].strip() and _PAGE_MARKER.fullmatch(lines[index]) is None
        )
        if not source_indices:
            continue
        requirements.append(
            CargoFlavorRewriteRequirement(
                requirementId=f"cargo-group-{group_index + 1}-span-1",
                targetPath=f"documentPatch.cargoGroups[{group_index}].description",
                targetDescription=target_description_by_group[group_index],
                sourceLineIds=tuple(_line_id(index + 1, lines[index]) for index in source_indices),
                sourceSurfaces=tuple(lines[index] for index in source_indices),
            )
        )
    return tuple(requirements)


def _cargo_flavor_rewrite_requirements_rendered(
    value: str, requirements: Sequence[CargoFlavorRewriteRequirement]
) -> bool:
    lines = value.splitlines()
    normalized_value = _semantic_normalize(value)
    for requirement in requirements:
        if _semantic_normalize(requirement.targetDescription) not in normalized_value:
            return False
        for line_id, source_surface in zip(
            requirement.sourceLineIds, requirement.sourceSurfaces, strict=True
        ):
            line_number = _line_number(line_id)
            if line_number > len(lines):
                return False
            if _semantic_normalize(lines[line_number - 1]) == _semantic_normalize(source_surface):
                return False
    return True


def rewrite_changed_leaves(
    source: Mapping[str, JsonValue], target: Mapping[str, JsonValue]
) -> tuple[ChangedLeaf, ...]:
    """Return label changes that require a different semantic OCR surface.

    Raw OCR retains the template's capitalization and punctuation. A target that differs only in
    those presentation details (for example ``EGYPT`` versus ``Egypt``) is already rendered and
    must not be rewritten or later reported as a stale source fact.
    """

    return tuple(
        leaf
        for leaf in changed_leaves(source, target)
        if not (
            isinstance(leaf.sourceValue, str)
            and isinstance(leaf.targetValue, str)
            and _semantic_normalize(leaf.sourceValue) == _semantic_normalize(leaf.targetValue)
        )
    )


def _load_customs_program_registry(path: Path, *, expected_entries: int) -> CustomsProgramRegistry:
    registry = CustomsProgramRegistry.model_validate_json(
        read_regular_file_bytes(path), strict=True
    )
    if len(registry.entries) != expected_entries:
        raise ValueError(
            "customs-program registry count differs from its pin: "
            f"expected={expected_entries}, actual={len(registry.entries)}"
        )
    return registry


def _route_country_index(
    locations: Sequence[UnlocodeLocation],
) -> Mapping[str, frozenset[str]]:
    countries: dict[str, set[str]] = {}
    for location in locations:
        for name in (location.name, location.name_without_diacritics):
            normalized = _semantic_normalize(name)
            if normalized:
                countries.setdefault(normalized, set()).add(location.country_code)
    return MappingProxyType({name: frozenset(values) for name, values in countries.items()})


def _location_country_code(value: Any, resources: TargetIntegrityResources) -> str | None:
    if not isinstance(value, Mapping):
        return None
    explicit: str | None = None
    country = value.get("country")
    if isinstance(country, str):
        explicit = resources.countries.resolve(country)
        if explicit is None:
            raise ValueError(
                f"target route country is absent from the pinned ISO registry: {country!r}"
            )
    inferred: str | None = None
    name = value.get("name")
    if isinstance(name, str):
        candidates = resources.route_countries_by_name.get(_semantic_normalize(name), frozenset())
        if len(candidates) == 1:
            inferred = next(iter(candidates))
    if explicit is not None and inferred is not None and explicit != inferred:
        raise ValueError(
            "target route location country conflicts with pinned UN/LOCODE: "
            f"name={name!r}, explicit={explicit}, unlocode={inferred}"
        )
    return explicit or inferred


_ROUTE_FIELDS_BY_DIRECTION: Mapping[Literal["import", "export"], tuple[str, ...]] = {
    "import": ("placeOfDelivery", "finalDestination", "portOfDischarge"),
    "export": ("portOfLoading", "placeOfReceipt"),
}


def _target_route_country_code(
    target_label: Mapping[str, Any],
    *,
    direction: Literal["import", "export"],
    resources: TargetIntegrityResources,
) -> str | None:
    patch = target_label.get("documentPatch")
    route = patch.get("route") if isinstance(patch, Mapping) else None
    if not isinstance(route, Mapping):
        return None
    for field_name in _ROUTE_FIELDS_BY_DIRECTION[direction]:
        country = _location_country_code(route.get(field_name), resources)
        if country is not None:
            return country
    return None


def _target_route_jurisdictions(
    target_label: Mapping[str, Any], resources: TargetIntegrityResources
) -> dict[str, JsonValue]:
    """Expose authoritative route jurisdictions for raw-only legal/customs flavor."""

    output: dict[str, JsonValue] = {}
    directions: tuple[Literal["export", "import"], ...] = ("export", "import")
    for direction in directions:
        code = _target_route_country_code(
            target_label,
            direction=direction,
            resources=resources,
        )
        output[direction] = (
            None
            if code is None
            else {
                "isoAlpha2": code,
                "name": resources.countries.printable_name(code),
            }
        )
    return output


def _literal_phrase_pattern(surface: str) -> re.Pattern[str]:
    tokens = re.findall(r"[A-Z0-9]+", _semantic_normalize(surface))
    if not tokens:
        raise ValueError(f"jurisdictional source surface has no tokens: {surface!r}")
    body = r"[\s./_-]+".join(re.escape(token) for token in tokens)
    return re.compile(rf"(?<![A-Z0-9]){body}(?![A-Z0-9])", re.IGNORECASE)


def _anchored_scalar_pattern(surface: str) -> re.Pattern[str]:
    """Match an identifier/reference surface without discarding meaningful punctuation."""

    stripped = surface.strip()
    if not stripped:
        raise ValueError("anchored scalar source surface is empty")
    parts = re.split(r"(\s+)", stripped)
    body = "".join(r"\s+" if part.isspace() else re.escape(part) for part in parts)
    prefix = r"(?<![A-Z0-9])" if stripped[0].isalnum() else ""
    suffix = r"(?![A-Z0-9])" if stripped[-1].isalnum() else ""
    return re.compile(f"{prefix}{body}{suffix}", re.IGNORECASE)


def _anchored_requirement_pattern(
    requirement: AnchoredScalarReplacementRequirement,
    surface: str,
) -> re.Pattern[str]:
    if requirement.surfaceKind == "scalar":
        return _anchored_scalar_pattern(surface)
    stripped = surface.strip()
    if not stripped:
        raise ValueError("anchored measurement surface is empty")
    return re.compile(
        rf"(?<![0-9.,]){re.escape(stripped)}(?![0-9.,])",
        re.IGNORECASE,
    )


def jurisdictional_surface_requirements(
    raw_text: str,
    target_label: Mapping[str, Any],
    resources: TargetIntegrityResources,
) -> tuple[JurisdictionalSurfaceRequirement, ...]:
    """Require named customs programs to agree with the synthetic route jurisdiction."""

    output: list[JurisdictionalSurfaceRequirement] = []
    for program in resources.customs_programs:
        matched: list[tuple[str, int]] = []
        occupied: list[tuple[int, int]] = []
        for surface in program.source_surfaces:
            matches = tuple(
                match
                for match in _literal_phrase_pattern(surface).finditer(raw_text)
                if not any(match.start() < end and match.end() > start for start, end in occupied)
            )
            count = len(matches)
            if count:
                matched.append((surface, count))
                occupied.extend((match.start(), match.end()) for match in matches)
        if not matched:
            continue
        target_country = _target_route_country_code(
            target_label,
            direction=program.trade_direction,
            resources=resources,
        )
        if target_country is None:
            raise ValueError(
                "cannot safely rewrite a jurisdiction-bound customs program without a resolved "
                f"target {program.trade_direction} route country: {program.program_id}"
            )
        if target_country == program.jurisdiction_country_code:
            continue
        output.extend(
            JurisdictionalSurfaceRequirement(
                programId=program.program_id,
                tradeDirection=program.trade_direction,
                programJurisdictionCountryCode=program.jurisdiction_country_code,
                targetRouteCountryCode=target_country,
                sourceSurface=surface,
                targetSurface=program.generic_replacement_surface,
                sourceOccurrences=occurrences,
                authority=program.authority,
                officialSourceUrl=program.official_source_url,
            )
            for surface, occurrences in matched
        )
    return tuple(output)


def _jurisdictional_surfaces_rendered(
    value: str, requirements: Sequence[JurisdictionalSurfaceRequirement]
) -> bool:
    for requirement in requirements:
        if _literal_phrase_pattern(requirement.sourceSurface).search(value) is not None:
            return False
        if _literal_phrase_pattern(requirement.targetSurface).search(value) is None:
            return False
    return True


_NON_LITERAL_TARGET_FIELD = re.compile(
    r"(?:^|\.)(?:"
    r"typeCategory|sizeCategory|hazardCategory|subsidiaryHazardCategories|"
    r"packingGroupCategory|paymentArrangement|negotiability|unit|sameAsConsignee"
    r")(?:\[[0-9]+\])?$"
)


def target_literal_requirements(
    leaves: Sequence[ChangedLeaf],
) -> tuple[TargetLiteralRequirement, ...]:
    """Require changed human-readable strings while excluding schema/category encodings."""

    rows: list[TargetLiteralRequirement] = []
    seen: set[tuple[str, str]] = set()
    for leaf in leaves:
        value = leaf.targetValue
        if (
            not leaf.requiresTextEdit
            or not leaf.targetPresent
            or not isinstance(value, str)
            or len(value.strip()) < 4
            or leaf.path.endswith("Date")
            or ".hsCodes[" in leaf.path
            or _NON_LITERAL_TARGET_FIELD.search(leaf.path) is not None
        ):
            continue
        normalized = _semantic_normalize(value)
        if not normalized or (leaf.path, normalized) in seen:
            continue
        rows.append(TargetLiteralRequirement(targetPath=leaf.path, targetValue=value))
        seen.add((leaf.path, normalized))
    return tuple(rows)


def _missing_target_literals(
    value: str, requirements: Sequence[TargetLiteralRequirement]
) -> tuple[TargetLiteralRequirement, ...]:
    normalized = _semantic_normalize(value)
    return tuple(
        row for row in requirements if _semantic_normalize(row.targetValue) not in normalized
    )


def target_value_occurrence_requirements(
    raw_text: str,
    leaves: Sequence[ChangedLeaf],
    raw_auxiliary_identities: Sequence[RawAuxiliaryIdentityRequirement] = (),
    surface_requirements: Sequence[SurfaceRenderingRequirement] = (),
) -> tuple[TargetValueOccurrenceRequirement, ...]:
    """Preserve role-bound party scalar cardinality without duplicating shared values.

    A source document can deliberately print one contact value for several semantic roles. For
    example, consignee and notify party can share one phone number and print it once in each party
    block. Synthetic labels normally give those roles different replacement numbers. Counting all
    raw occurrences independently for every changed leaf would require both replacements in both
    slots. Instead, divide the printed occurrences of a shared source value across the label leaves
    that own it, then aggregate those per-leaf multiplicities by replacement value. Party names and
    addresses use the source party-name repetition count, so surplus address lines must receive
    distinct auxiliary detail rather than duplicate the labeled address.
    """

    contact_leaves: list[ChangedLeaf] = []
    party_scalar_leaves: list[ChangedLeaf] = []
    party_path = re.compile(
        r"^(?P<prefix>documentPatch\.parties\.(?:notifyParties\[[0-9]+\]|[A-Za-z]+))\."
        r"(?P<field>name|address)$"
    )
    for leaf in leaves:
        if (
            not leaf.requiresTextEdit
            or not leaf.sourcePresent
            or not leaf.targetPresent
            or not isinstance(leaf.sourceValue, str)
            or not isinstance(leaf.targetValue, str)
            or len(leaf.targetValue.strip()) < 4
            or leaf.sourceValue == leaf.targetValue
        ):
            continue
        if ".contactDetails." in leaf.path:
            contact_leaves.append(leaf)
        elif party_path.fullmatch(leaf.path) is not None:
            party_scalar_leaves.append(leaf)

    by_source: dict[str, list[ChangedLeaf]] = {}
    for leaf in contact_leaves:
        source_value = cast(str, leaf.sourceValue)
        by_source.setdefault(source_value, []).append(leaf)

    contributions_by_path: dict[str, int] = {}
    for source_value, source_leaves in sorted(by_source.items()):
        printed_occurrences = raw_text.count(source_value)
        if printed_occurrences == 0:
            continue
        semantic_owners = len(source_leaves)
        if printed_occurrences % semantic_owners:
            paths = ", ".join(sorted(leaf.path for leaf in source_leaves))
            raise ValueError(
                "printed contact occurrences cannot be assigned exactly across semantic owners: "
                f"{source_value!r} occurs {printed_occurrences} time(s) for {semantic_owners} "
                f"owner(s): {paths}"
            )
        per_owner = printed_occurrences // semantic_owners
        for leaf in source_leaves:
            contributions_by_path[leaf.path] = per_owner

    party_owners_by_source: dict[tuple[str, str], list[ChangedLeaf]] = {}
    for leaf in party_scalar_leaves:
        match = party_path.fullmatch(leaf.path)
        if match is not None:
            party_owners_by_source.setdefault(
                (match.group("field"), cast(str, leaf.sourceValue)), []
            ).append(leaf)
    auxiliary_name_occurrences: dict[str, int] = {}
    for requirement in raw_auxiliary_identities:
        key = _semantic_normalize(requirement.sourceIdentity)
        auxiliary_name_occurrences[key] = auxiliary_name_occurrences.get(key, 0) + 1
    for (field_name, source_value), owners in sorted(party_owners_by_source.items()):
        normalized_source = _semantic_normalize(source_value)
        printed_occurrences = _party_scalar_occurrence_count(raw_text, source_value)
        source_paths = {leaf.path for leaf in owners}
        if field_name == "name":
            printed_occurrences -= auxiliary_name_occurrences.get(normalized_source, 0)
            printed_occurrences += sum(
                requirement.sourceOccurrences
                for requirement in surface_requirements
                if requirement.kind == "carrier_header_identity"
                and requirement.targetPath in source_paths
                and _semantic_normalize(requirement.sourceSurface) != normalized_source
            )
        if printed_occurrences < 0:
            raise ValueError(
                "raw auxiliary identities exceed printed party-name occurrences: "
                f"{source_value!r}"
            )
        if printed_occurrences == 0:
            continue
        if printed_occurrences % len(owners):
            raise ValueError(
                "printed party scalar occurrences cannot be assigned exactly across semantic "
                f"owners: {source_value!r} occurs {printed_occurrences} time(s) for "
                f"{len(owners)} owner(s): {', '.join(sorted(source_paths))}"
            )
        copies = printed_occurrences // len(owners)
        for leaf in owners:
            contributions_by_path[leaf.path] = copies

    grouped: dict[str, dict[str, Any]] = {}
    eligible = (*contact_leaves, *party_scalar_leaves)
    changed_source_values = frozenset(cast(str, leaf.sourceValue) for leaf in eligible)
    for leaf in eligible:
        contribution = contributions_by_path.get(leaf.path)
        if contribution is None:
            continue
        target_value = cast(str, leaf.targetValue)
        normalized = _semantic_normalize(target_value)
        row = grouped.setdefault(
            normalized,
            {
                "targetValue": target_value,
                "paths": [],
                "requiredOccurrences": (
                    raw_text.count(target_value) if target_value not in changed_source_values else 0
                ),
            },
        )
        if row["targetValue"] != target_value:
            raise ValueError("contact values normalize identically but have different surfaces")
        row["paths"].append(leaf.path)
        row["requiredOccurrences"] += contribution

    output: list[TargetValueOccurrenceRequirement] = []
    for normalized in sorted(grouped):
        row = grouped[normalized]
        target_value = cast(str, row["targetValue"])
        required = cast(int, row["requiredOccurrences"])
        if required < 1:
            continue
        output.append(
            TargetValueOccurrenceRequirement(
                targetPaths=tuple(sorted(cast(list[str], row["paths"]))),
                targetValue=target_value,
                requiredOccurrences=required,
            )
        )
    return tuple(output)


def _target_value_occurrences_exact(
    value: str, requirements: Sequence[TargetValueOccurrenceRequirement]
) -> bool:
    return all(
        _target_value_occurrence_count(value, row) == row.requiredOccurrences
        for row in requirements
    )


def _target_value_occurrence_count(
    value: str, requirement: TargetValueOccurrenceRequirement
) -> int:
    if any(".contactDetails." in path for path in requirement.targetPaths):
        return value.count(requirement.targetValue)
    if any(
        path.startswith("documentPatch.parties.")
        and path.rsplit(".", 1)[-1] in {"name", "address"}
        for path in requirement.targetPaths
    ):
        return _party_scalar_occurrence_count(value, requirement.targetValue)
    normalized_value = _semantic_normalize(value)
    normalized_target = _semantic_normalize(requirement.targetValue)
    if not normalized_target:
        return 0
    # Flattened OCR often concatenates a heading and its value (``Notify PartyACME LTD``). Exact
    # token boundaries would reject a value that is visibly present in the correct source style.
    # Cardinality is still exact because the complete normalized target scalar is counted.
    return normalized_value.count(normalized_target)


def anchored_scalar_replacement_requirements(
    raw_text: str, leaves: Sequence[ChangedLeaf]
) -> tuple[AnchoredScalarReplacementRequirement, ...]:
    """Anchor unambiguous exact scalar replacements to their original printed lines.

    This contract is intentionally limited to schema fields whose values are identifiers,
    references, or exact contact scalars and therefore cannot be legitimately paraphrased. Party
    and route prose remains under role-aware occurrence and semantic-review contracts. Dates and
    HS codes have dedicated source-format renderers.
    """

    semantic_source_targets: dict[str, set[str]] = {}
    for leaf in leaves:
        if not isinstance(leaf.sourceValue, str) or not isinstance(leaf.targetValue, str):
            continue
        normalized_source = _semantic_normalize(leaf.sourceValue)
        normalized_target = _semantic_normalize(leaf.targetValue)
        if normalized_source and normalized_target:
            semantic_source_targets.setdefault(normalized_source, set()).add(normalized_target)

    grouped: dict[str, list[ChangedLeaf]] = {}
    for leaf in leaves:
        if (
            not leaf.requiresTextEdit
            or not leaf.sourcePresent
            or not leaf.targetPresent
            or not isinstance(leaf.sourceValue, str)
            or not isinstance(leaf.targetValue, str)
            or leaf.sourceValue == leaf.targetValue
            or len(leaf.sourceValue.strip()) < 4
            or "\n" in leaf.sourceValue
            or "\n" in leaf.targetValue
            or leaf.path.endswith("Date")
            or ".hsCodes[" in leaf.path
            or _NON_LITERAL_TARGET_FIELD.search(leaf.path) is not None
            or (
                _ANCHORABLE_IDENTIFIER_PATH.fullmatch(leaf.path) is None
                and _ANCHORABLE_CONTACT_PATH.fullmatch(leaf.path) is None
            )
        ):
            continue
        # The same OCR surface can represent different fields. For example, a reviewed mark
        # ``*EGYPT`` can also appear in a notify-party country line. Replacing every literal
        # occurrence would move the new mark into an address. Only anchor a surface globally when
        # all semantically equivalent changed source values resolve to the same target surface.
        normalized_source = _semantic_normalize(leaf.sourceValue)
        if len(semantic_source_targets.get(normalized_source, set())) > 1:
            continue
        grouped.setdefault(leaf.sourceValue, []).append(leaf)

    source_lines = raw_text.splitlines()
    requirements: list[AnchoredScalarReplacementRequirement] = []
    for source_surface, source_leaves in sorted(grouped.items()):
        target_surfaces = {cast(str, leaf.targetValue) for leaf in source_leaves}
        if len(target_surfaces) != 1:
            continue
        pattern = _anchored_scalar_pattern(source_surface)
        source_line_ids = tuple(
            _line_id(index, line)
            for index, line in enumerate(source_lines, start=1)
            if pattern.search(line) is not None
        )
        if not source_line_ids:
            continue
        requirements.append(
            AnchoredScalarReplacementRequirement(
                targetPaths=tuple(sorted(leaf.path for leaf in source_leaves)),
                sourceLineIds=source_line_ids,
                sourceSurface=source_surface,
                targetSurface=target_surfaces.pop(),
            )
        )
    return tuple(requirements)


def _anchored_scalar_replacements_rendered(
    value: str, requirements: Sequence[AnchoredScalarReplacementRequirement]
) -> bool:
    lines = value.splitlines()
    for requirement in requirements:
        target_pattern = _anchored_requirement_pattern(requirement, requirement.targetSurface)
        source_pattern = _anchored_requirement_pattern(requirement, requirement.sourceSurface)
        for line_id in requirement.sourceLineIds:
            line_number = _line_number(line_id)
            if line_number > len(lines):
                return False
            line = lines[line_number - 1]
            if target_pattern.search(line) is None or source_pattern.search(line) is not None:
                return False
    return True


def merge_anchored_scalar_replacement_requirements(
    *groups: Sequence[AnchoredScalarReplacementRequirement],
) -> tuple[AnchoredScalarReplacementRequirement, ...]:
    """Merge duplicate exact substitutions and reject contradictory line authority."""

    merged: dict[tuple[str, str, str], dict[str, set[str]]] = {}
    line_targets: dict[tuple[str, str], str] = {}
    for requirement in (row for group in groups for row in group):
        for line_id in requirement.sourceLineIds:
            authority_key = (line_id, requirement.sourceSurface)
            existing_target = line_targets.setdefault(authority_key, requirement.targetSurface)
            if existing_target != requirement.targetSurface:
                raise ValueError(
                    "conflicting deterministic scalar targets for "
                    f"{line_id} and {requirement.sourceSurface!r}: "
                    f"{existing_target!r} != {requirement.targetSurface!r}"
                )
        merged_row = merged.setdefault(
            (requirement.sourceSurface, requirement.targetSurface, requirement.surfaceKind),
            {"paths": set(), "lines": set()},
        )
        merged_row["paths"].update(requirement.targetPaths)
        merged_row["lines"].update(requirement.sourceLineIds)
    return tuple(
        AnchoredScalarReplacementRequirement(
            targetPaths=tuple(sorted(values["paths"])),
            sourceLineIds=tuple(sorted(values["lines"], key=_line_number)),
            sourceSurface=source_surface,
            targetSurface=target_surface,
            surfaceKind=cast(Literal["scalar", "measurement"], surface_kind),
        )
        for (source_surface, target_surface, surface_kind), values in merged.items()
    )


def apply_deterministic_prefills(
    workspace: RewriteWorkspace,
) -> tuple[AppliedDeterministicPrefill, ...]:
    """Apply exact scalar and unambiguous derived-surface substitutions before rewriting.

    Anchored scalars plus target-rendered dates, HS codes, receipt counts, and receipt equipment
    breakdowns have no linguistic degrees of freedom. Doing them locally is cheaper and more
    reliable than asking a model to copy exact values. A derived source surface is prefilled only
    when every requirement for that surface agrees on one target; ambiguous projections stay in
    the model-visible contract and remain fail-closed. The provider sees all resulting lines and
    must preserve them.
    """

    lines = workspace.current_text.splitlines(keepends=True)
    applied: list[AppliedDeterministicPrefill] = []
    for requirement in workspace.anchored_scalar_replacement_requirements:
        source_pattern = _anchored_requirement_pattern(requirement, requirement.sourceSurface)
        target_pattern = _anchored_requirement_pattern(requirement, requirement.targetSurface)
        for line_id in requirement.sourceLineIds:
            line_index = _line_number(line_id) - 1
            if not 0 <= line_index < len(lines):
                raise ValueError(f"deterministic prefill line is outside the OCR: {line_id}")
            before = _line_body(lines[line_index])
            ending = _line_ending(lines[line_index])
            if source_pattern.search(before) is None:
                if target_pattern.search(before) is not None:
                    continue
                raise ValueError(
                    "deterministic prefill cannot locate its source scalar on "
                    f"{line_id}: {requirement.sourceSurface!r}"
                )
            literal_replacement = requirement.targetSurface.replace("\\", "\\\\")
            after = source_pattern.sub(literal_replacement, before)
            if source_pattern.search(after) is not None or target_pattern.search(after) is None:
                raise ValueError(f"deterministic prefill did not replace its scalar on {line_id}")
            lines[line_index] = after + ending
            applied.append(
                AppliedDeterministicPrefill(
                    lineId=line_id,
                    targetPaths=requirement.targetPaths,
                    sourceSurface=requirement.sourceSurface,
                    targetSurface=requirement.targetSurface,
                    beforeLine=before,
                    afterLine=after,
                )
            )

    deterministic_surface_kinds = {
        "date",
        "hs_code",
        "hs_code_block",
        "carrier_receipt_count",
        "carrier_receipt_equipment_breakdown",
    }
    surface_groups: dict[str, list[SurfaceRenderingRequirement]] = {}
    for surface_requirement in workspace.surface_requirements:
        if surface_requirement.kind in deterministic_surface_kinds:
            surface_groups.setdefault(surface_requirement.sourceSurface, []).append(
                surface_requirement
            )
    for source_surface, requirements in surface_groups.items():
        target_surfaces = {row.targetSurface for row in requirements}
        if len(target_surfaces) != 1:
            continue
        target_surface = next(iter(target_surfaces))
        if source_surface == target_surface:
            continue
        leading_boundary = r"(?<![A-Za-z0-9])" if source_surface[0].isalnum() else ""
        trailing_boundary = r"(?![A-Za-z0-9])" if source_surface[-1].isalnum() else ""
        source_pattern = re.compile(
            leading_boundary + re.escape(source_surface) + trailing_boundary
        )
        literal_replacement = target_surface.replace("\\", "\\\\")
        target_paths = tuple(sorted({row.targetPath for row in requirements}))
        matched = False
        for line_index, line in enumerate(lines):
            before = _line_body(line)
            if source_pattern.search(before) is None:
                continue
            matched = True
            ending = _line_ending(line)
            after = source_pattern.sub(literal_replacement, before)
            if after == before:
                raise ValueError(
                    "deterministic surface prefill did not replace "
                    f"{source_surface!r} on line {line_index + 1}"
                )
            lines[line_index] = after + ending
            applied.append(
                AppliedDeterministicPrefill(
                    lineId=_line_id(line_index + 1, before),
                    targetPaths=target_paths,
                    sourceSurface=source_surface,
                    targetSurface=target_surface,
                    beforeLine=before,
                    afterLine=after,
                )
            )
        if not matched:
            current = "".join(lines)
            if target_surface not in current:
                raise ValueError(
                    "deterministic surface prefill cannot locate source or target surface: "
                    f"{source_surface!r} -> {target_surface!r}"
                )
    updated = "".join(lines)
    if len(updated.splitlines()) != len(workspace.original_text.splitlines()):
        raise ValueError("deterministic prefill changes the source line count")
    if _blank_line_topology(updated) != _blank_line_topology(workspace.original_text):
        raise ValueError("deterministic prefill changes the source blank-line topology")
    if not _anchored_scalar_replacements_rendered(
        updated, workspace.anchored_scalar_replacement_requirements
    ):
        raise ValueError("deterministic prefill did not satisfy every anchored scalar")
    workspace.current_text = updated
    workspace.deterministic_prefills.extend(applied)
    return tuple(applied)


def _introduced_source_line_duplicates(source: str, target: str) -> tuple[str, ...]:
    """Return long source lines whose exact normalized count was increased by editing."""

    def counts(value: str) -> dict[str, int]:
        output: dict[str, int] = {}
        for line in value.splitlines():
            normalized = " ".join(line.split())
            if len(normalized) < 24 or _PAGE_MARKER.fullmatch(normalized):
                continue
            output[normalized] = output.get(normalized, 0) + 1
        return output

    before = counts(source)
    after = counts(target)
    return tuple(sorted(line for line, count in before.items() if after.get(line, 0) > count))


def compact_label_change_contract(
    directives: Sequence[LabelChangeDirective],
) -> tuple[CompactLabelChangeDirective, ...]:
    """Deduplicate schema aliases that require the same printed substitution."""

    grouped: dict[tuple[str, bytes, bytes], list[str]] = {}
    values: dict[tuple[str, bytes, bytes], tuple[JsonValue, JsonValue]] = {}
    for directive in directives:
        source = canonical_json_bytes(directive.sourceValue)
        target = canonical_json_bytes(directive.targetValue)
        key = (directive.action, source, target)
        grouped.setdefault(key, []).append(directive.path)
        values[key] = (directive.sourceValue, directive.targetValue)
    return tuple(
        CompactLabelChangeDirective(
            paths=tuple(paths),
            action=cast(Any, key[0]),
            sourceValue=values[key][0],
            targetValue=values[key][1],
        )
        for key, paths in grouped.items()
    )


def _date_value(value: Any) -> date | None:
    if not isinstance(value, str):
        return None
    try:
        return date.fromisoformat(value)
    except ValueError:
        return None


def _year(value: str) -> int:
    parsed = int(value)
    if len(value) == 2:
        return 2000 + parsed if parsed <= 79 else 1900 + parsed
    return parsed


def _numeric_date_order(
    match: re.Match[str], expected: date
) -> Literal["ymd", "dmy", "mdy"] | None:
    a, b, c = match.group("a"), match.group("b"), match.group("c")
    candidates: list[tuple[Literal["ymd", "dmy", "mdy"], tuple[int, int, int]]] = []
    if len(a) == 4:
        candidates.append(("ymd", (int(a), int(b), int(c))))
    else:
        candidates.extend(
            (
                ("dmy", (_year(c), int(b), int(a))),
                ("mdy", (_year(c), int(a), int(b))),
            )
        )
    matches: list[Literal["ymd", "dmy", "mdy"]] = []
    for order, parts in candidates:
        try:
            if date(*parts) == expected:
                matches.append(order)
        except ValueError:
            continue
    if "dmy" in matches:
        # The frozen project policy is day-first when a numeric surface is ambiguous.
        return "dmy"
    return matches[0] if matches else None


def _render_numeric_date(match: re.Match[str], target: date, order: str) -> str:
    a, b, c = match.group("a"), match.group("b"), match.group("c")
    separator = match.group("s")

    def padded(value: int, width: int) -> str:
        return str(value).zfill(width) if width > 1 else str(value)

    year = str(target.year) if max(len(a), len(c)) == 4 else str(target.year % 100).zfill(2)
    if order == "ymd":
        values = (year, padded(target.month, len(b)), padded(target.day, len(c)))
    elif order == "dmy":
        values = (padded(target.day, len(a)), padded(target.month, len(b)), year)
    else:
        values = (padded(target.month, len(a)), padded(target.day, len(b)), year)
    return separator.join(values)


def _named_month_number(value: str) -> int | None:
    return _MONTHS.get(value.upper())


def _month_surface(source: str, month: int) -> str:
    full = date(2000, month, 1).strftime("%B")
    rendered = full[: len(source)] if len(source) <= 4 else full
    if source.isupper():
        return rendered.upper()
    if source.islower():
        return rendered.lower()
    return rendered.title()


def _rendered_date_candidates(
    raw_text: str,
    *,
    source: date,
    target: date,
) -> tuple[tuple[int, int, str, str], ...]:
    output: list[tuple[int, int, str, str]] = []
    for match in _NUMERIC_DATE.finditer(raw_text):
        if (order := _numeric_date_order(match, source)) is not None:
            output.append(
                (
                    match.start(),
                    match.end(),
                    match.group(0),
                    _render_numeric_date(match, target, order),
                )
            )
    for match in _DAY_NAMED_MONTH_DATE.finditer(raw_text):
        month = _named_month_number(match.group("m"))
        if month is None:
            continue
        try:
            observed = date(_year(match.group("y")), month, int(match.group("d")))
        except ValueError:
            continue
        if observed == source:
            year = (
                str(target.year) if len(match.group("y")) == 4 else str(target.year % 100).zfill(2)
            )
            output.append(
                (
                    match.start(),
                    match.end(),
                    match.group(0),
                    match.group("s").join(
                        (
                            str(target.day).zfill(len(match.group("d"))),
                            _month_surface(match.group("m"), target.month),
                            year,
                        )
                    ),
                )
            )
    for match in _NAMED_MONTH_DAY_DELIMITED_DATE.finditer(raw_text):
        month = _named_month_number(match.group("m"))
        if month is None:
            continue
        try:
            observed = date(_year(match.group("y")), month, int(match.group("d")))
        except ValueError:
            continue
        if observed == source:
            year = (
                str(target.year) if len(match.group("y")) == 4 else str(target.year % 100).zfill(2)
            )
            separator = match.group("s")
            output.append(
                (
                    match.start(),
                    match.end(),
                    match.group(0),
                    separator.join(
                        (
                            _month_surface(match.group("m"), target.month),
                            str(target.day).zfill(len(match.group("d"))),
                            year,
                        )
                    ),
                )
            )
    for match in _NAMED_MONTH_DAY_DATE.finditer(raw_text):
        month = _named_month_number(match.group("m"))
        if month is None:
            continue
        try:
            observed = date(_year(match.group("y")), month, int(match.group("d")))
        except ValueError:
            continue
        if observed == source:
            year = (
                str(target.year) if len(match.group("y")) == 4 else str(target.year % 100).zfill(2)
            )
            output.append(
                (
                    match.start(),
                    match.end(),
                    match.group(0),
                    f"{_month_surface(match.group('m'), target.month)} "
                    f"{str(target.day).zfill(len(match.group('d')))}{match.group('comma')} {year}",
                )
            )
    return tuple(output)


def _date_context_path(raw_text: str, start: int) -> str | None:
    prefix = raw_text[max(0, start - 180) : start]
    issue = tuple(_ISSUE_DATE_CONTEXT.finditer(prefix))
    shipped = tuple(_SHIPPED_DATE_CONTEXT.finditer(prefix))
    if not issue and not shipped:
        return None
    if issue and (not shipped or issue[-1].start() > shipped[-1].start()):
        return "documentPatch.issueDate"
    return "documentPatch.shippedOnBoardDate"


def _format_numeric_surface(source: str, target_digits: str) -> str:
    groups = tuple(re.finditer(r"[0-9]+", source))
    if not groups:
        raise ValueError("numeric source surface has no digits")
    cursor = 0
    chunks: list[str] = []
    for index, group in enumerate(groups):
        if cursor >= len(target_digits):
            break
        if index:
            previous = groups[index - 1]
            chunks.append(source[previous.end() : group.start()])
        width = len(group.group(0))
        chunks.append(target_digits[cursor : cursor + width])
        cursor += width
    if cursor < len(target_digits):
        chunks.append(target_digits[cursor:])
    return "".join(chunks)


def _carrier_receipt_target_count(
    observed: int,
    source_label: Mapping[str, Any],
    target_label: Mapping[str, Any],
) -> int:
    """Project an explicit carrier-receipt count through matching source label quantities."""

    source_patch = cast(Mapping[str, Any], source_label.get("documentPatch") or {})
    target_patch = cast(Mapping[str, Any], target_label.get("documentPatch") or {})
    candidates: list[int] = []
    source_containers = source_patch.get("containers") or []
    target_containers = target_patch.get("containers") or []
    if len(source_containers) == observed:
        candidates.append(len(target_containers))
    source_packages = source_patch.get("cargoPackages") or []
    target_packages = target_patch.get("cargoPackages") or []
    for source_package, target_package in zip(source_packages, target_packages, strict=False):
        if not isinstance(source_package, Mapping) or not isinstance(target_package, Mapping):
            continue
        source_quantity = source_package.get("quantity")
        target_quantity = target_package.get("quantity")
        if source_quantity == observed and isinstance(target_quantity, int):
            candidates.append(target_quantity)
    source_quantities: list[int] = []
    target_quantities: list[int] = []
    for row in source_packages:
        quantity = row.get("quantity") if isinstance(row, Mapping) else None
        if isinstance(quantity, int) and not isinstance(quantity, bool):
            source_quantities.append(quantity)
    for row in target_packages:
        quantity = row.get("quantity") if isinstance(row, Mapping) else None
        if isinstance(quantity, int) and not isinstance(quantity, bool):
            target_quantities.append(quantity)
    if source_quantities and sum(source_quantities) == observed and target_quantities:
        candidates.append(sum(target_quantities))
    resolved = set(candidates)
    if not resolved:
        raise ValueError(
            "carrier-receipt count is not grounded in source container/package quantities: "
            f"{observed}"
        )
    if len(resolved) != 1:
        raise ValueError(
            "carrier-receipt count has conflicting target projections: "
            f"source={observed}, targets={sorted(resolved)}"
        )
    return next(iter(resolved))


def _target_equipment_counts(target_label: Mapping[str, Any]) -> dict[str, int]:
    patch = cast(Mapping[str, Any], target_label.get("documentPatch") or {})
    counts: dict[str, int] = {}
    for container in patch.get("containers") or []:
        if not isinstance(container, Mapping):
            raise ValueError("target container is not an object")
        size = container.get("sizeCategory")
        if not isinstance(size, str):
            raise ValueError("target container has no semantic size category")
        if size.startswith("TWENTY_FOOT"):
            token = "20"
        elif size.startswith("FORTY_FIVE_FOOT"):
            token = "45"
        elif size.startswith("FORTY_FOOT"):
            token = "40"
        else:
            raise ValueError(f"unsupported target equipment size category: {size}")
        counts[token] = counts.get(token, 0) + 1
    return counts


def _carrier_receipt_equipment_breakdown_requirements(
    raw_text: str, target_label: Mapping[str, Any]
) -> tuple[SurfaceRenderingRequirement, ...]:
    lines = raw_text.splitlines()
    counts = _target_equipment_counts(target_label)
    if not counts:
        return ()
    output: list[SurfaceRenderingRequirement] = []
    for index, line in enumerate(lines):
        if _CARRIER_RECEIPT_HEADING.search(line) is None:
            continue
        for candidate_index in range(index + 1, min(len(lines), index + 5)):
            source_surface = lines[candidate_index]
            if not source_surface.strip():
                continue
            if _EQUIPMENT_BREAKDOWN_LINE.fullmatch(source_surface) is None:
                break
            source_order = tuple(dict.fromkeys(re.findall(r"(?:20|40|45)(?=')", source_surface)))
            target_order = source_order + tuple(
                token
                for token in ("40", "20", "45")
                if token in counts and token not in source_order
            )
            target_surface = " + ".join(
                f"{counts[token]} X {token}'" for token in target_order if counts.get(token, 0) > 0
            )
            output.append(
                SurfaceRenderingRequirement(
                    kind="carrier_receipt_equipment_breakdown",
                    targetPath="auxiliary.carrierReceiptEquipmentBreakdown",
                    sourceSurface=source_surface,
                    targetSurface=target_surface,
                    sourceOccurrences=lines.count(source_surface),
                    contextEvidence="\n".join(
                        lines[max(0, index - 1) : min(len(lines), candidate_index + 2)]
                    ),
                )
            )
            break
    return tuple(output)


def surface_rendering_requirements(
    raw_text: str,
    source_label: Mapping[str, Any],
    target_label: Mapping[str, Any],
) -> tuple[SurfaceRenderingRequirement, ...]:
    """Derive exact date/HS surfaces from source formatting and canonical targets."""

    source_patch = cast(Mapping[str, Any], source_label.get("documentPatch") or {})
    target_patch = cast(Mapping[str, Any], target_label.get("documentPatch") or {})
    requirements: list[SurfaceRenderingRequirement] = []
    source_carrier = _party_name(source_label, "carrier")
    target_carrier = _party_name(target_label, "carrier")
    if source_carrier is not None and target_carrier is not None:
        source_tokens = set(re.findall(r"[A-Z0-9]+", _semantic_normalize(source_carrier)))
        seen_carrier_header_surfaces: set[str] = set()
        for source_surface in raw_text.splitlines():
            normalized_surface = _semantic_normalize(source_surface)
            surface_tokens = set(re.findall(r"[A-Z0-9]+", normalized_surface))
            overlap = source_tokens & surface_tokens
            if (
                normalized_surface == _semantic_normalize(source_carrier)
                or len(surface_tokens) < 3
                or len(overlap) / len(surface_tokens) < 0.8
                or len(overlap) / len(source_tokens) < 0.6
            ):
                continue
            if normalized_surface in seen_carrier_header_surfaces:
                continue
            seen_carrier_header_surfaces.add(normalized_surface)
            requirements.append(
                SurfaceRenderingRequirement(
                    kind="carrier_header_identity",
                    targetPath="documentPatch.parties.carrier.name",
                    sourceSurface=source_surface,
                    targetSurface=target_carrier,
                    sourceOccurrences=raw_text.splitlines().count(source_surface),
                    contextEvidence=source_surface,
                )
            )
    for date_field in ("issueDate", "shippedOnBoardDate"):
        source = _date_value(source_patch.get(date_field))
        target = _date_value(target_patch.get(date_field))
        if source is None or target is None or source == target:
            continue
        path = f"documentPatch.{date_field}"
        candidates = _rendered_date_candidates(raw_text, source=source, target=target)
        contextual = tuple(
            row for row in candidates if _date_context_path(raw_text, row[0]) == path
        )
        selected = contextual[:1] or (candidates if len(candidates) == 1 else ())
        if not selected:
            raise ValueError(f"cannot locate an unambiguous source date surface for {path}")
        for start, end, source_surface, target_surface in selected:
            requirements.append(
                SurfaceRenderingRequirement(
                    kind="date",
                    targetPath=path,
                    sourceSurface=source_surface,
                    targetSurface=target_surface,
                    sourceOccurrences=1,
                    contextEvidence=raw_text[max(0, start - 90) : min(len(raw_text), end + 40)],
                )
            )

    source_groups = source_patch.get("cargoGroups") or []
    target_groups = target_patch.get("cargoGroups") or []
    hs_pairs: dict[str, list[tuple[str, str]]] = {}
    for group_index, (source_group, target_group) in enumerate(
        zip(source_groups, target_groups, strict=False)
    ):
        if not isinstance(source_group, Mapping) or not isinstance(target_group, Mapping):
            continue
        source_codes = source_group.get("hsCodes") or []
        target_codes = target_group.get("hsCodes") or []
        for code_index, (source_code, target_code) in enumerate(
            zip(source_codes, target_codes, strict=False)
        ):
            if not isinstance(source_code, str) or not isinstance(target_code, str):
                continue
            source_digits = "".join(re.findall(r"[0-9]", source_code))
            target_digits = "".join(re.findall(r"[0-9]", target_code))
            if source_digits == target_digits:
                continue
            path = f"documentPatch.cargoGroups[{group_index}].hsCodes[{code_index}]"
            hs_pairs.setdefault(source_digits, []).append((path, target_digits))
    for source_digits, pairs in hs_pairs.items():
        matches = tuple(
            match
            for match in _NUMERIC_SURFACE.finditer(raw_text)
            if "".join(re.findall(r"[0-9]", match.group(0))) == source_digits
        )
        if len(matches) < len(pairs):
            raise ValueError(
                f"cannot locate every source HS surface for paths {[path for path, _ in pairs]}"
            )
        for index, match in enumerate(matches):
            path, target_digits = pairs[index % len(pairs)]
            requirements.append(
                SurfaceRenderingRequirement(
                    kind="hs_code",
                    targetPath=path,
                    sourceSurface=match.group(0),
                    targetSurface=_format_numeric_surface(match.group(0), target_digits),
                    sourceOccurrences=1,
                    contextEvidence=raw_text[
                        max(0, match.start() - 90) : min(len(raw_text), match.end() + 40)
                    ],
                )
            )
    for match in _CARRIER_RECEIPT_COUNT.finditer(raw_text):
        source_surface = match.group(0)
        observed = int(match.group("count").replace(",", ""))
        target_count = _carrier_receipt_target_count(observed, source_label, target_label)
        target_digits = f"{target_count:,}" if "," in match.group("count") else str(target_count)
        target_surface = match.group("prefix") + target_digits + match.group("suffix")
        requirements.append(
            SurfaceRenderingRequirement(
                kind="carrier_receipt_count",
                targetPath="auxiliary.carrierReceiptCount",
                sourceSurface=source_surface,
                targetSurface=target_surface,
                sourceOccurrences=1,
                contextEvidence=raw_text[
                    max(0, match.start() - 60) : min(len(raw_text), match.end() + 60)
                ],
            )
        )
    requirements.extend(_carrier_receipt_equipment_breakdown_requirements(raw_text, target_label))
    return tuple(requirements)


def source_semantic_role_hints(raw_text: str) -> tuple[SourceSemanticRoleHint, ...]:
    """Recognize explicit flattened grammar without guessing from isolated numbers."""

    lines = raw_text.splitlines()
    output: list[SourceSemanticRoleHint] = []
    for index, line in enumerate(lines):
        if re.fullmatch(r"[ \t]*[1-9][0-9]*[ \t]*", line) is None:
            continue
        end = min(len(lines), index + 8)
        block = "\n".join(lines[index:end])
        if _ANONYMOUS_EQUIPMENT.search(block) is None:
            continue
        output.append(
            SourceSemanticRoleHint(
                role="anonymous_equipment_count",
                sourceLineId=_line_id(index + 1, line),
                sourceSurface=line,
                requiredOutputSurface=line,
                sourceEvidence=block,
                forbiddenTargetPathPrefixes=("documentPatch.cargoGroups[].marksAndNumbers",),
            )
        )
    return tuple(output)


def _source_semantic_role_surfaces_preserved(
    value: str, requirements: Sequence[SourceSemanticRoleHint]
) -> bool:
    lines = value.splitlines()
    return all(
        _line_number(row.sourceLineId) <= len(lines)
        and lines[_line_number(row.sourceLineId) - 1] == row.requiredOutputSurface
        for row in requirements
    )


def _label_has_marks(label: Mapping[str, Any]) -> bool:
    patch = cast(Mapping[str, Any], label.get("documentPatch") or {})
    return any(
        isinstance(group, Mapping) and bool(group.get("marksAndNumbers"))
        for group in patch.get("cargoGroups") or []
    )


def _finding_misclassifies_anonymous_equipment(
    finding: SemanticReviewFinding,
    workspace: RewriteWorkspace,
) -> bool:
    if not any(row.role == "anonymous_equipment_count" for row in workspace.source_role_hints):
        return False
    if _label_has_marks(workspace.source_label) or _label_has_marks(workspace.current_target_label):
        return False
    return any("marksAndNumbers" in path for path in finding.affectedPaths)


def _finding_conflicts_with_equipment_authority(
    finding: SemanticReviewFinding,
    workspace: RewriteWorkspace,
) -> bool:
    """Reject equipment findings whose cited container proves the target semantics."""

    affected = {
        int(match.group(1))
        for path in finding.affectedPaths
        if (match := _CONTAINER_EQUIPMENT_REVIEW_PATH.search(path)) is not None
    }
    if not affected:
        return False
    patch = workspace.current_target_label.get("documentPatch")
    containers = patch.get("containers") if isinstance(patch, Mapping) else None
    if not isinstance(containers, list):
        return False
    for evidence in finding.currentEvidence:
        normalized_evidence = _semantic_normalize(evidence)
        for index, container in enumerate(containers):
            if not isinstance(container, Mapping):
                continue
            number = container.get("containerNumber")
            if (
                not isinstance(number, str)
                or _semantic_normalize(number) not in normalized_evidence
            ):
                continue
            if index not in affected:
                return True
            size = container.get("sizeCategory")
            category = container.get("typeCategory")
            if not isinstance(size, str) or not isinstance(category, str):
                continue
            reviewed = review_source_equipment_surface(
                evidence,
                temperature_present=container.get("temperatureSetpoint") is not None,
            )
            if (
                finding.category in {"missing_or_wrong_target_fact", "stale_source_fact"}
                and reviewed.resolution == "reviewed_source_grammar"
                and reviewed.size_category == size
                and reviewed.type_category == category
            ):
                return True
    return False


def _finding_conflicts_with_surface_authority(
    finding: SemanticReviewFinding,
    workspace: RewriteWorkspace,
) -> bool:
    """Reject a review that asks to change an already-proven derived exact surface."""

    for requirement in workspace.surface_requirements:
        if requirement.kind not in {
            "carrier_receipt_count",
            "carrier_receipt_equipment_breakdown",
        }:
            continue
        if requirement.targetSurface not in workspace.current_text:
            continue
        if any(
            requirement.targetSurface in evidence or evidence in requirement.targetSurface
            for evidence in finding.currentEvidence
        ):
            return True
    return False


def _finding_conflicts_with_anchored_scalar_authority(
    finding: SemanticReviewFinding,
    workspace: RewriteWorkspace,
) -> bool:
    """Reject reclassification of a proven line-bound target scalar as auxiliary flavor."""

    if finding.category not in {"stale_source_fact", "incomplete_auxiliary_anonymization"}:
        return False

    current_lines = workspace.current_text.splitlines()
    affected_paths = set(finding.affectedPaths)
    for requirement in workspace.anchored_scalar_replacement_requirements:
        if affected_paths & set(requirement.targetPaths):
            continue
        target_pattern = _anchored_requirement_pattern(requirement, requirement.targetSurface)
        for line_id in requirement.sourceLineIds:
            line_number = _line_number(line_id)
            if line_number > len(current_lines):
                continue
            current_line = current_lines[line_number - 1]
            if target_pattern.search(current_line) is None:
                continue
            if any(
                _evidence_is_isolated_anchored_target(evidence, requirement.targetSurface)
                for evidence in finding.currentEvidence
            ):
                return True
    return False


def _evidence_is_isolated_anchored_target(evidence: str, target_surface: str) -> bool:
    """Distinguish reclassifying a target scalar from flagging a neighboring stale value."""

    normalized_evidence = _semantic_normalize(evidence)
    normalized_target = _semantic_normalize(target_surface)
    if normalized_evidence == normalized_target:
        return True
    if normalized_target not in normalized_evidence:
        return False
    remainder = normalized_evidence.replace(normalized_target, " ", 1).split()
    return len(remainder) <= 3 and all(token.isalpha() for token in remainder)


def _finding_grounds_format_damage_only_in_unchanged_text(
    finding: SemanticReviewFinding,
    workspace: RewriteWorkspace,
) -> bool:
    """Formatting evidence must quote changed output, not bytes identical to the source."""

    return finding.category in {"format_or_layout_damage", "unnecessary_change"} and all(
        evidence in workspace.original_text for evidence in finding.currentEvidence
    )


def _required_surfaces_rendered(
    value: str, requirements: Sequence[SurfaceRenderingRequirement]
) -> bool:
    target_surfaces = {row.targetSurface for row in requirements}
    for row in requirements:
        if row.targetSurface not in value:
            return False
        if (
            row.sourceSurface != row.targetSurface
            and row.sourceSurface not in target_surfaces
            and row.sourceSurface in value
        ):
            return False
    return True


def _compound_relationships(
    value: str,
) -> tuple[Literal["on_behalf_of", "trading_as"], ...]:
    return tuple(name for name, pattern in _COMPOUND_PARTY_RELATIONS if pattern.search(value))


def _party_name(label: Mapping[str, Any], role: str) -> str | None:
    patch = label.get("documentPatch")
    if not isinstance(patch, Mapping):
        return None
    parties = patch.get("parties")
    if not isinstance(parties, Mapping):
        return None
    party = parties.get(role)
    if not isinstance(party, Mapping):
        return None
    value = party.get("name")
    return value if isinstance(value, str) else None


def compound_party_flavor_requirements(
    source_label: Mapping[str, Any], target_label: Mapping[str, Any]
) -> tuple[CompoundPartyFlavorRequirement, ...]:
    """Find source name relationships to preserve as fictional raw-text flavor.

    The synthetic target remains authoritative and unchanged. A compound relationship in the
    reviewed source name describes template topology: the raw realization keeps that relationship
    around the exact synthetic primary identity using a distinct fictional secondary identity.
    Similar wording occurring solely in raw signature or agency text is handled as ordinary
    auxiliary flavor by the editor.
    """

    output: list[CompoundPartyFlavorRequirement] = []
    for role in _COMPOUND_PARTY_ROLES:
        source_name = _party_name(source_label, role)
        target_name = _party_name(target_label, role)
        if source_name is None or target_name is None:
            continue
        relationships = _compound_relationships(source_name)
        if not relationships or set(relationships) <= set(_compound_relationships(target_name)):
            continue
        output.append(
            CompoundPartyFlavorRequirement(
                targetPath=f"documentPatch.parties.{role}.name",
                relationships=relationships,
                sourceLabelName=source_name,
                targetPrimaryName=target_name,
            )
        )
    return tuple(output)


def _active_compound_party_flavor_requirements(
    workspace: RewriteWorkspace,
) -> tuple[CompoundPartyFlavorRequirement, ...]:
    """Return every compound relationship that the current atomic edit must attest."""

    return compound_party_flavor_requirements(
        workspace.source_label, workspace.current_target_label
    )


def _validate_compound_party_flavor_realizations(
    workspace: RewriteWorkspace,
    realizations: Sequence[CompoundPartyFlavorRealization],
) -> tuple[AppliedCompoundPartyFlavorRealization, ...]:
    requirements = _active_compound_party_flavor_requirements(workspace)
    required_by_path = {row.targetPath: row for row in requirements}
    submitted_by_path = {row.targetPath: row for row in realizations}
    if len(submitted_by_path) != len(realizations):
        raise ValueError("compound-party flavor realizations contain duplicate paths")
    if set(submitted_by_path) != set(required_by_path):
        missing = sorted(set(required_by_path) - set(submitted_by_path))
        extra = sorted(set(submitted_by_path) - set(required_by_path))
        raise ValueError(
            "compound-party flavor realizations differ from requirements; "
            f"missing={missing}, extra={extra}"
        )
    if not requirements:
        return ()
    applied: list[AppliedCompoundPartyFlavorRealization] = []
    for path in sorted(required_by_path):
        requirement = required_by_path[path]
        realization = submitted_by_path[path]
        if realization.targetPrimaryName != requirement.targetPrimaryName:
            raise ValueError(f"compound-party realization uses a stale target name: {path}")
        normalized_primary = _semantic_normalize(requirement.targetPrimaryName)
        normalized_rendered = _semantic_normalize(realization.renderedName)
        if not normalized_rendered.startswith(normalized_primary + " "):
            raise ValueError(
                f"compound raw realization must begin with the target primary identity: {path}"
            )
        observed_relationships = _compound_relationships(realization.renderedName)
        if set(observed_relationships) != set(requirement.relationships):
            raise ValueError(
                f"compound raw realization does not preserve the relation topology: {path}"
            )
        for relationship, pattern in _COMPOUND_PARTY_RELATIONS:
            if relationship not in requirement.relationships:
                continue
            match = pattern.search(realization.renderedName)
            assert match is not None
            auxiliary = _semantic_normalize(realization.renderedName[match.end() :])
            if len(auxiliary.split()) < 2 or auxiliary == normalized_primary:
                raise ValueError(
                    "compound raw realization needs a distinct realistic auxiliary identity: "
                    f"{path}"
                )
        applied.append(
            AppliedCompoundPartyFlavorRealization(
                targetPath=path,
                relationships=requirement.relationships,
                targetPrimaryName=requirement.targetPrimaryName,
                renderedName=realization.renderedName,
            )
        )
    return tuple(applied)


def _raw_agent_blocks(value: str) -> tuple[tuple[int, str, int, str, str], ...]:
    """Return explicit raw-only agency identities on either side of their legal relation."""

    lines = value.splitlines()
    output: list[tuple[int, str, int, str, str]] = []
    for relation_index, relation_line in enumerate(lines):
        match = _RAW_AGENT_FOR_CARRIER_LINE.fullmatch(relation_line)
        if match is None:
            continue
        cursor = relation_index - 1
        gap = 0
        while cursor >= 0 and not lines[cursor].strip() and gap < 2:
            gap += 1
            cursor -= 1
        end = cursor + 1
        while (
            cursor >= 0
            and end - cursor <= 3
            and lines[cursor].strip()
            and _PAGE_MARKER.fullmatch(lines[cursor]) is None
        ):
            cursor -= 1
        start = cursor + 1
        identity_lines = lines[start:end]
        if not identity_lines:
            continue
        output.append(
            (
                start + 1,
                "\n".join(identity_lines),
                gap,
                match.group("principal").strip(),
                "\n".join(lines[start : relation_index + 1]),
            )
        )
    for relation_index, relation_line in enumerate(lines):
        match = _RAW_ISSUED_AGENT_BY_LINE.fullmatch(relation_line)
        if match is None:
            continue
        cursor = relation_index + 1
        gap = 0
        while cursor < len(lines) and not lines[cursor].strip() and gap < 2:
            gap += 1
            cursor += 1
        start = cursor
        while (
            cursor < len(lines)
            and cursor - start < 3
            and lines[cursor].strip()
            and _PAGE_MARKER.fullmatch(lines[cursor]) is None
        ):
            cursor += 1
        identity_lines = lines[start:cursor]
        if not identity_lines:
            continue
        output.append(
            (
                start + 1,
                "\n".join(identity_lines),
                gap,
                match.group("principal").strip(),
                "\n".join(lines[relation_index:cursor]),
            )
        )
    for relation_index, relation_line in enumerate(lines):
        match = _RAW_INLINE_AGENT_FOR_CARRIER_LINE.fullmatch(relation_line)
        if match is None:
            continue
        principal = match.group("principal")
        gap = 0
        evidence_end = relation_index + 1
        if principal is None:
            cursor = relation_index + 1
            while cursor < len(lines) and not lines[cursor].strip() and gap < 2:
                gap += 1
                cursor += 1
            if cursor >= len(lines) or not lines[cursor].strip():
                continue
            principal = lines[cursor].strip()
            evidence_end = cursor + 1
        identity = match.group("identity").strip().rstrip(",").strip()
        if not identity:
            continue
        output.append(
            (
                relation_index + 1,
                identity,
                gap,
                principal.strip(),
                "\n".join(lines[relation_index:evidence_end]),
            )
        )
    for relation_index, relation_line in enumerate(lines):
        relation = _RAW_SIGNED_ON_BEHALF_LINE.fullmatch(relation_line)
        if relation is None:
            continue
        cursor = relation_index + 1
        gap = 0
        while cursor < len(lines) and not lines[cursor].strip() and gap < 2:
            gap += 1
            cursor += 1
        if cursor >= len(lines):
            continue
        agent = _RAW_BY_AGENT_LINE.fullmatch(lines[cursor])
        if agent is None:
            continue
        output.append(
            (
                cursor + 1,
                agent.group("identity").strip(),
                gap,
                relation.group("principal").strip(),
                "\n".join(lines[relation_index : cursor + 1]),
            )
        )
    return tuple(sorted(output, key=lambda row: row[0]))


def _auxiliary_identity_token_key(value: str) -> str:
    return " ".join(
        sorted(
            token for token in re.findall(r"[A-Z0-9]+", _semantic_normalize(value)) if token != "BY"
        )
    )


def _auxiliary_identity_consistency_group(value: str) -> str:
    return (
        "carrier-agent-" + sha256_bytes(_auxiliary_identity_token_key(value).encode("utf-8"))[:16]
    )


def raw_auxiliary_identity_requirements(
    raw_text: str,
    source_label: Mapping[str, Any],
    target_label: Mapping[str, Any],
) -> tuple[RawAuxiliaryIdentityRequirement, ...]:
    """Locate syntactically explicit raw-only agents adjacent to carrier relationships."""

    source_carrier = _party_name(source_label, "carrier")
    target_carrier = _party_name(target_label, "carrier")
    if target_carrier is None:
        return ()
    requirements: list[RawAuxiliaryIdentityRequirement] = []
    for (
        line_number,
        source_identity,
        gap,
        source_principal,
        evidence,
    ) in _raw_agent_blocks(raw_text):
        normalized = _semantic_normalize(source_identity)
        normalized_source_principal = _semantic_normalize(source_principal)
        normalized_source_carrier = _semantic_normalize(source_carrier or "")
        if normalized_source_carrier and not (
            normalized_source_carrier in normalized_source_principal
            or normalized_source_principal in normalized_source_carrier
        ):
            continue
        if (
            normalized == normalized_source_carrier
            or normalized == _semantic_normalize(target_carrier)
            or len(normalized.split()) < 2
        ):
            continue
        requirements.append(
            RawAuxiliaryIdentityRequirement(
                requirementId=f"carrier-agent-L{line_number:05d}",
                relationship="agent_for_carrier",
                sourceIdentity=source_identity,
                sourceIdentityLineCount=len(source_identity.splitlines()),
                gapLineCount=gap,
                consistencyGroupId=_auxiliary_identity_consistency_group(source_identity),
                targetPrincipalName=target_carrier,
                sourceEvidence=evidence,
            )
        )
    return tuple(requirements)


def _cargo_measure_present(label: Mapping[str, Any], field_name: str) -> bool:
    patch = label.get("documentPatch")
    if not isinstance(patch, Mapping):
        return False
    return any(
        isinstance(group, Mapping) and group.get(field_name) is not None
        for group in cast(Sequence[Any], patch.get("cargoGroups") or ())
    )


def _measurement_pattern(kind: Literal["gross_weight_kg", "volume_m3"]) -> re.Pattern[str]:
    return _RAW_GROSS_WEIGHT if kind == "gross_weight_kg" else _RAW_VOLUME


def _parse_measurement_surface(
    surface: str,
    *,
    maximum: Decimal | None,
) -> ParsedMeasurementSurface | None:
    """Parse a carrier numeric surface without assuming a national locale.

    A three-digit suffix is ambiguous between decimal and grouping notation.  In that case the
    physically valid interpretation with the larger magnitude is selected: ``13.519`` gross kg
    means 13,519 when both 13.519 and 13,519 fit, while ``70.1530`` CBM has an unambiguous
    four-place decimal suffix.  Capacity bounds reject impossible alternatives.
    """

    value = surface.strip()
    if not re.fullmatch(r"[0-9]+(?:[.,][0-9]+)*", value):
        return None
    candidates: list[ParsedMeasurementSurface] = []
    separators = {character for character in value if character in ".,"}
    if len(separators) == 2:
        decimal_separator = "." if value.rfind(".") > value.rfind(",") else ","
        grouping_separator = "," if decimal_separator == "." else "."
        integer, fraction = value.rsplit(decimal_separator, 1)
        normalized = integer.replace(grouping_separator, "") + "." + fraction
        candidates.append(
            ParsedMeasurementSurface(
                value=Decimal(normalized),
                decimal_separator=decimal_separator,
                grouping_separator=grouping_separator,
                decimal_places=len(fraction),
            )
        )
    elif len(separators) == 1:
        separator = next(iter(separators))
        pieces = value.split(separator)
        if len(pieces) == 2:
            integer, fraction = pieces
            candidates.append(
                ParsedMeasurementSurface(
                    value=Decimal(integer + "." + fraction),
                    decimal_separator=separator,
                    grouping_separator=None,
                    decimal_places=len(fraction),
                )
            )
            if len(fraction) == 3:
                candidates.append(
                    ParsedMeasurementSurface(
                        value=Decimal(integer + fraction),
                        decimal_separator=None,
                        grouping_separator=separator,
                        decimal_places=0,
                    )
                )
        elif all(len(piece) == 3 for piece in pieces[1:]):
            candidates.append(
                ParsedMeasurementSurface(
                    value=Decimal("".join(pieces)),
                    decimal_separator=None,
                    grouping_separator=separator,
                    decimal_places=0,
                )
            )
    else:
        candidates.append(
            ParsedMeasurementSurface(
                value=Decimal(value),
                decimal_separator=None,
                grouping_separator=None,
                decimal_places=0,
            )
        )
    valid = [
        row
        for row in candidates
        if row.value.is_finite() and row.value > 0 and (maximum is None or row.value <= maximum)
    ]
    return max(valid, key=lambda row: row.value) if valid else None


def _render_measurement_surface(
    value: Decimal,
    *,
    style: ParsedMeasurementSurface,
    maximum: Decimal | None,
) -> str:
    quantum = Decimal(1).scaleb(-style.decimal_places)
    rounded = value.quantize(quantum, rounding=ROUND_HALF_UP)
    if maximum is not None and rounded > maximum:
        rounded = maximum.quantize(quantum, rounding=ROUND_DOWN)
    if rounded <= 0:
        raise ValueError("resampled operational measurement rounds to zero")
    plain = f"{rounded:.{style.decimal_places}f}"
    integer, dot, fraction = plain.partition(".")
    if style.grouping_separator is not None:
        integer = f"{int(integer):,}".replace(",", style.grouping_separator)
    if style.decimal_separator is None:
        return integer
    return integer + style.decimal_separator + fraction if dot else integer


def _line_has_measurement_role(
    lines: Sequence[str], line_index: int, kind: Literal["gross_weight_kg", "volume_m3"]
) -> bool:
    current = _semantic_normalize(lines[line_index])
    preceding = _semantic_normalize(" ".join(lines[max(0, line_index - 2) : line_index]))
    numeric_only = re.fullmatch(r"[0-9]+(?:[.,][0-9]+)*", lines[line_index].strip()) is not None
    if kind == "gross_weight_kg":
        terms: tuple[str, ...] = ("GROSS WEIGHT", "GROSS WT", "G W")
        return any(surface in current for surface in terms) or (
            numeric_only and any(surface in preceding for surface in terms)
        )
    terms = ("MEASUREMENT", "VOLUME", "CBM", "CUBIC METER", "CUBIC METRE")
    return any(surface in current for surface in terms) or (
        numeric_only and any(surface in preceding for surface in terms)
    )


def anchored_measurement_replacement_requirements(
    raw_text: str,
    source_label: Mapping[str, Any],
    target_label: Mapping[str, Any],
) -> tuple[AnchoredScalarReplacementRequirement, ...]:
    """Render changed labeled measures in the exact grouping and precision of their source line."""

    source_patch = cast(Mapping[str, Any], source_label.get("documentPatch") or {})
    target_patch = cast(Mapping[str, Any], target_label.get("documentPatch") or {})
    source_groups = cast(Sequence[Any], source_patch.get("cargoGroups") or ())
    target_groups = cast(Sequence[Any], target_patch.get("cargoGroups") or ())
    kinds: tuple[Literal["gross_weight_kg", "volume_m3"], ...] = (
        "gross_weight_kg",
        "volume_m3",
    )
    field_names = {"gross_weight_kg": "grossWeight", "volume_m3": "volume"}
    pairs: list[tuple[int, Literal["gross_weight_kg", "volume_m3"], Decimal, Decimal]] = []
    source_value_counts: dict[tuple[str, Decimal], int] = {}
    for group_index, (source_group, target_group) in enumerate(
        zip(source_groups, target_groups, strict=False)
    ):
        if not isinstance(source_group, Mapping) or not isinstance(target_group, Mapping):
            continue
        for kind in kinds:
            source_value = _target_measure_value(source_group, kind)
            target_value = _target_measure_value(target_group, kind)
            if source_value is None or target_value is None or source_value == target_value:
                continue
            pairs.append((group_index, kind, source_value, target_value))
            key = (kind, source_value)
            source_value_counts[key] = source_value_counts.get(key, 0) + 1

    lines = raw_text.splitlines()
    claimed_lines: set[int] = set()
    requirements: list[AnchoredScalarReplacementRequirement] = []
    for group_index, kind, source_value, target_value in pairs:
        candidates: list[tuple[int, str, str]] = []
        for line_index, line in enumerate(lines):
            if not _line_has_measurement_role(lines, line_index, kind):
                continue
            for match in _RAW_MEASUREMENT_NUMBER.finditer(line):
                source_surface = match.group("value")
                style = _parse_measurement_surface(source_surface, maximum=source_value)
                if style is None or style.value != source_value:
                    continue
                target_surface = _render_measurement_surface(
                    target_value, style=style, maximum=None
                )
                candidates.append((line_index, source_surface, target_surface))
        if not candidates:
            continue
        available = [row for row in candidates if row[0] not in claimed_lines]
        if not available:
            continue
        selected = available if source_value_counts[(kind, source_value)] == 1 else available[:1]
        grouped: dict[tuple[str, str], list[int]] = {}
        for line_index, source_surface, target_surface in selected:
            claimed_lines.add(line_index)
            grouped.setdefault((source_surface, target_surface), []).append(line_index)
        for (source_surface, target_surface), line_indices in grouped.items():
            requirements.append(
                AnchoredScalarReplacementRequirement(
                    targetPaths=(
                        f"documentPatch.cargoGroups[{group_index}].{field_names[kind]}.value",
                    ),
                    sourceLineIds=tuple(
                        _line_id(line_index + 1, lines[line_index]) for line_index in line_indices
                    ),
                    sourceSurface=source_surface,
                    targetSurface=target_surface,
                    surfaceKind="measurement",
                )
            )
    return tuple(requirements)


def _source_equipment_capacity(
    container: Mapping[str, Any],
    *,
    local_evidence: str,
    limits: TransportCapacityLimits,
) -> EquipmentCapacity:
    printed = next(
        (
            cast(str, container[key])
            for key in ("typeDescription", "typeCode")
            if isinstance(container.get(key), str)
        ),
        None,
    )
    reviewed = review_source_equipment_surface(
        printed,
        temperature_present=bool(
            re.search(r"[-+]?[0-9]+(?:[.,][0-9]+)?\s*°?\s*C\b", local_evidence, re.I)
        ),
    )
    if reviewed.size_category is not None and reviewed.type_category is not None:
        return equipment_capacity(
            {
                "sizeCategory": reviewed.size_category,
                "typeCategory": reviewed.type_category,
            },
            limits,
        )
    return equipment_capacity(container, limits)


def _container_measurement_occurrences(
    raw_text: str,
    containers: Sequence[Mapping[str, Any]],
) -> dict[
    int,
    tuple[
        tuple[
            Literal["gross_weight_kg", "volume_m3", "package_quantity"],
            int,
            str,
            str,
            Literal["labeled_measurement", "inline_container_breakdown"],
        ],
        ...,
    ],
]:
    """Find only measurement rows in the local printed block headed by a container number."""

    lines = raw_text.splitlines()
    numbers = [
        cast(str, row["containerNumber"])
        for row in containers
        if isinstance(row.get("containerNumber"), str)
    ]
    output: dict[
        int,
        list[
            tuple[
                Literal["gross_weight_kg", "volume_m3", "package_quantity"],
                int,
                str,
                str,
                Literal["labeled_measurement", "inline_container_breakdown"],
            ]
        ],
    ] = {}
    for container_index, number in enumerate(numbers):
        header = re.compile(rf"^[ \t]*{re.escape(number)}(?:\b|(?=[ \t/,:#-]))", re.I)
        for line_index, line in enumerate(lines):
            if header.search(line) is None:
                continue
            evidence_lines = [line]
            found: set[str] = set()
            # Carrier container-detail blocks put seal/tare between the container header and the
            # local measurements. Six following occupied lines covers the reviewed grammar while
            # preventing a last container from absorbing document-level totals farther below.
            for offset in range(0, 7):
                cursor = line_index + offset
                if cursor >= len(lines):
                    break
                current = lines[cursor]
                if offset and (
                    _PAGE_MARKER.fullmatch(current)
                    or not current.strip()
                    or any(
                        re.match(rf"^[ \t]*{re.escape(other)}(?:\b|(?=[ \t/,:#-]))", current, re.I)
                        for other in numbers
                    )
                ):
                    break
                if offset:
                    evidence_lines.append(current)
                for kind in ("gross_weight_kg", "volume_m3"):
                    if kind in found:
                        continue
                    match = _measurement_pattern(cast(Any, kind)).search(current)
                    if match is not None:
                        output.setdefault(container_index, []).append(
                            (
                                cast(Any, kind),
                                cursor + 1,
                                match.group("value"),
                                "\n".join(evidence_lines)[-600:],
                                "labeled_measurement",
                            )
                        )
                        found.add(kind)
                if len(found) == 2:
                    break
    # Some carrier riders encode one container's measures at the start of the line that introduces
    # the next container. Preserve that topology by assigning the parenthesized tuple to the most
    # recently printed container before advancing the container cursor.
    last_container_index: int | None = None
    for line_index, line in enumerate(lines):
        inline = _RAW_INLINE_CONTAINER_MEASURES.search(line)
        if inline is not None and last_container_index is not None:
            evidence = "\n".join(lines[max(0, line_index - 1) : line_index + 1])[-600:]
            output.setdefault(last_container_index, []).extend(
                (
                    (
                        "gross_weight_kg",
                        line_index + 1,
                        inline.group("gross"),
                        evidence,
                        "inline_container_breakdown",
                    ),
                    (
                        "volume_m3",
                        line_index + 1,
                        inline.group("volume"),
                        evidence,
                        "inline_container_breakdown",
                    ),
                    (
                        "package_quantity",
                        line_index + 1,
                        inline.group("packages"),
                        evidence,
                        "inline_container_breakdown",
                    ),
                )
            )
        for container_index, number in enumerate(numbers):
            if re.search(rf"(?<![A-Z0-9]){re.escape(number)}(?![A-Z0-9])", line, re.I):
                last_container_index = container_index
                break
    return {key: tuple(value) for key, value in output.items()}


def build_empirical_operational_profiles(
    source_rows: Sequence[Mapping[str, Any]],
    *,
    target_field: str,
    limits: TransportCapacityLimits,
) -> tuple[EmpiricalOperationalProfile, ...]:
    """Build paired real-document utilization support for raw-only measurement synthesis."""

    profiles: list[EmpiricalOperationalProfile] = []
    seen: set[tuple[str, str, Decimal | None, Decimal | None]] = set()
    for row in source_rows:
        document_id = row.get("documentId")
        raw_text = row.get("joinedRawText")
        target = row.get(target_field)
        if (
            not isinstance(document_id, str)
            or not isinstance(raw_text, str)
            or not isinstance(target, Mapping)
        ):
            continue
        patch = target.get("documentPatch")
        if not isinstance(patch, Mapping):
            continue
        containers = tuple(
            cast(Mapping[str, Any], value)
            for value in cast(Sequence[Any], patch.get("containers") or ())
            if isinstance(value, Mapping)
        )
        occurrences = _container_measurement_occurrences(raw_text, containers)
        for container_index, rows in occurrences.items():
            if container_index >= len(containers):
                continue
            container = containers[container_index]
            evidence = "\n".join(value[3] for value in rows)
            capacity = _source_equipment_capacity(
                container,
                local_evidence=evidence,
                limits=limits,
            )
            values: dict[str, Decimal] = {}
            for kind, _, surface, _, _ in rows:
                if kind == "package_quantity":
                    continue
                maximum = capacity.payload_kg if kind == "gross_weight_kg" else capacity.volume_m3
                parsed = _parse_measurement_surface(surface, maximum=maximum)
                if parsed is not None:
                    values[kind] = parsed.value
            gross = values.get("gross_weight_kg")
            volume = values.get("volume_m3")
            if gross is None and volume is None:
                continue
            container_number = container.get("containerNumber")
            if not isinstance(container_number, str):
                continue
            identity = (document_id, container_number, gross, volume)
            if identity in seen:
                continue
            seen.add(identity)
            profiles.append(
                EmpiricalOperationalProfile(
                    document_id=document_id,
                    container_number=container_number,
                    equipment_family=capacity.family,
                    gross_weight_kg=gross,
                    gross_utilization=(gross / capacity.payload_kg if gross is not None else None),
                    volume_m3=volume,
                    volume_utilization=(
                        volume / capacity.volume_m3
                        if volume is not None and capacity.volume_m3 is not None
                        else None
                    ),
                )
            )
    return tuple(sorted(profiles, key=lambda row: (row.document_id, row.container_number)))


_MASS_TO_KG = {
    "kilogram": Decimal(1),
    "metric_tonne": Decimal(1000),
    "pound": Decimal("0.45359237"),
}


def _target_measure_value(
    group: Mapping[str, Any], kind: Literal["gross_weight_kg", "volume_m3"]
) -> Decimal | None:
    field_name = "grossWeight" if kind == "gross_weight_kg" else "volume"
    measure = group.get(field_name)
    if measure is None:
        return None
    if not isinstance(measure, Mapping) or not isinstance(measure.get("unit"), str):
        raise ValueError(f"target {field_name} is not a canonical measure")
    try:
        value = Decimal(str(measure["value"]))
    except (KeyError, ArithmeticError) as error:
        raise ValueError(f"target {field_name} has no finite numeric value") from error
    if kind == "gross_weight_kg":
        factor = _MASS_TO_KG.get(cast(str, measure["unit"]))
        if factor is None:
            raise ValueError(f"unsupported target gross-weight unit: {measure['unit']}")
        value *= factor
    elif measure["unit"] != "cubic_metre":
        raise ValueError(f"unsupported target volume unit: {measure['unit']}")
    if not value.is_finite() or value <= 0:
        raise ValueError(f"target {field_name} must be finite and positive")
    return value


def _decimal_places(value: Decimal) -> int:
    return max(0, -cast(int, value.as_tuple().exponent))


def _target_measure_allocations(
    target_label: Mapping[str, Any],
    *,
    kind: Literal["gross_weight_kg", "volume_m3"],
    printed_precision: int,
    limits: TransportCapacityLimits,
) -> dict[str, Decimal]:
    """Allocate each labeled cargo-group measure across its explicit container relations."""

    patch = cast(Mapping[str, Any], target_label.get("documentPatch") or {})
    containers = {
        cast(str, row["containerNumber"]): cast(Mapping[str, Any], row)
        for row in cast(Sequence[Any], patch.get("containers") or ())
        if isinstance(row, Mapping) and isinstance(row.get("containerNumber"), str)
    }
    allocation_groups = {
        cast(str, row["groupId"]): cast(Mapping[str, Any], row)
        for row in cast(Sequence[Any], patch.get("cargoAllocationGroups") or ())
        if isinstance(row, Mapping) and isinstance(row.get("groupId"), str)
    }
    output: dict[str, Decimal] = {}
    for group in cast(Sequence[Any], patch.get("cargoGroups") or ()):
        if not isinstance(group, Mapping) or not isinstance(group.get("groupId"), str):
            continue
        total = _target_measure_value(group, kind)
        if total is None:
            continue
        allocation_group = allocation_groups.get(cast(str, group["groupId"]))
        weights: dict[str, int] = {}
        if allocation_group is not None:
            for allocation in cast(Sequence[Any], allocation_group.get("allocations") or ()):
                if not isinstance(allocation, Mapping):
                    continue
                number = allocation.get("containerNumber")
                quantity = allocation.get("packageQuantity")
                if isinstance(number, str) and isinstance(quantity, int) and quantity >= 0:
                    weights[number] = weights.get(number, 0) + quantity
        if not weights and len(containers) == 1:
            weights = {next(iter(containers)): 1}
        unknown = sorted(set(weights) - set(containers))
        if unknown or not weights or sum(weights.values()) <= 0:
            raise ValueError(
                f"target {kind} has no complete positive container allocation for "
                f"group {group['groupId']}: unknown={unknown}"
            )
        precision = max(printed_precision, _decimal_places(total))
        scale = 10**precision
        total_units = int((total * scale).to_integral_exact())
        numbers = tuple(weights)
        units = largest_remainder_allocation(total_units, tuple(weights[row] for row in numbers))
        for number, allocated_units in zip(numbers, units, strict=True):
            output[number] = output.get(number, Decimal(0)) + Decimal(allocated_units) / scale
    for number, value in output.items():
        capacity = equipment_capacity(containers[number], limits)
        maximum = capacity.payload_kg if kind == "gross_weight_kg" else capacity.volume_m3
        if maximum is not None and value > maximum:
            raise ValueError(
                f"allocated target {kind} exceeds {number} capacity: {value} > {maximum}"
            )
    return output


def _target_package_allocations(target_label: Mapping[str, Any]) -> dict[str, int]:
    """Return the explicit task-facing package count allocated to each container."""

    patch = cast(Mapping[str, Any], target_label.get("documentPatch") or {})
    containers = {
        cast(str, row["containerNumber"])
        for row in cast(Sequence[Any], patch.get("containers") or ())
        if isinstance(row, Mapping) and isinstance(row.get("containerNumber"), str)
    }
    output: dict[str, int] = {}
    for group in cast(Sequence[Any], patch.get("cargoAllocationGroups") or ()):
        if not isinstance(group, Mapping):
            continue
        for allocation in cast(Sequence[Any], group.get("allocations") or ()):
            if not isinstance(allocation, Mapping):
                continue
            number = allocation.get("containerNumber")
            quantity = allocation.get("packageQuantity")
            if not isinstance(number, str) or not isinstance(quantity, int) or quantity < 0:
                raise ValueError("target cargo allocation has an invalid package quantity")
            if number not in containers:
                raise ValueError(f"target cargo allocation names unknown container {number}")
            output[number] = output.get(number, 0) + quantity
    return output


def _render_integer_surface(value: int, *, source_surface: str) -> str:
    if value < 0:
        raise ValueError("cannot render a negative package quantity")
    return f"{value:,}" if "," in source_surface else str(value)


def operational_flavor_requirements(
    raw_text: str,
    source_label: Mapping[str, Any],
    target_label: Mapping[str, Any],
    *,
    source_document_id: str,
    scenario_id: str,
    profiles: Sequence[EmpiricalOperationalProfile],
    limits: TransportCapacityLimits,
) -> tuple[OperationalFlavorRequirement, ...]:
    """Create exact capacity-safe edits for container-local operational measurements.

    When the target label contains an aggregate measure, its explicit cargo-to-container
    allocations determine a lossless per-container breakdown. Otherwise, raw-only measurements
    are resampled from paired real-document utilization profiles.
    """

    source_patch = cast(Mapping[str, Any], source_label.get("documentPatch") or {})
    target_patch = cast(Mapping[str, Any], target_label.get("documentPatch") or {})
    source_containers = tuple(
        cast(Mapping[str, Any], value)
        for value in cast(Sequence[Any], source_patch.get("containers") or ())
        if isinstance(value, Mapping)
    )
    target_containers = tuple(
        cast(Mapping[str, Any], value)
        for value in cast(Sequence[Any], target_patch.get("containers") or ())
        if isinstance(value, Mapping)
    )
    if len(source_containers) != len(target_containers):
        raise ValueError("cannot pair raw operational flavor across different container counts")
    occurrences = _container_measurement_occurrences(raw_text, source_containers)
    if not occurrences:
        return ()
    requirements: list[OperationalFlavorRequirement] = []
    kinds: tuple[Literal["gross_weight_kg", "volume_m3"], ...] = (
        "gross_weight_kg",
        "volume_m3",
    )
    source_fields = {"gross_weight_kg": "grossWeight", "volume_m3": "volume"}
    target_has_measure = {
        measure_kind: _cargo_measure_present(target_label, source_fields[measure_kind])
        for measure_kind in kinds
    }
    for measure_kind in kinds:
        if (
            _cargo_measure_present(source_label, source_fields[measure_kind])
            and not target_has_measure[measure_kind]
        ):
            raise ValueError(
                f"source exposes labeled {measure_kind} but synthetic target removes it "
                "from a fixed measurement topology"
            )
    printed_precision: dict[Literal["gross_weight_kg", "volume_m3"], int] = {
        "gross_weight_kg": 0,
        "volume_m3": 0,
    }
    source_capacities: dict[int, EquipmentCapacity] = {}
    for container_index, observed_rows in occurrences.items():
        source_capacity = _source_equipment_capacity(
            source_containers[container_index],
            local_evidence="\n".join(row[3] for row in observed_rows),
            limits=limits,
        )
        source_capacities[container_index] = source_capacity
        for operational_kind, _, surface, _, _ in observed_rows:
            if operational_kind == "package_quantity":
                continue
            maximum = (
                source_capacity.payload_kg
                if operational_kind == "gross_weight_kg"
                else source_capacity.volume_m3
            )
            parsed = _parse_measurement_surface(surface, maximum=maximum)
            if parsed is None:
                raise ValueError(f"cannot parse source {operational_kind} surface {surface!r}")
            printed_precision[operational_kind] = max(
                printed_precision[operational_kind], parsed.decimal_places
            )
    allocated_values = {
        measure_kind: (
            _target_measure_allocations(
                target_label,
                kind=measure_kind,
                printed_precision=printed_precision[measure_kind],
                limits=limits,
            )
            if target_has_measure[measure_kind]
            else {}
        )
        for measure_kind in kinds
    }
    package_allocations = _target_package_allocations(target_label)
    raw_lines = raw_text.splitlines()
    source_numbers = [
        cast(str, row["containerNumber"])
        for row in source_containers
        if isinstance(row.get("containerNumber"), str)
    ]
    for container_index, observed_rows in occurrences.items():
        target_container = target_containers[container_index]
        target_number = target_container.get("containerNumber")
        if not isinstance(target_number, str):
            raise ValueError("synthetic container has no containerNumber")
        capacity = equipment_capacity(target_container, limits)
        source_capacity = source_capacities[container_index]
        empirical_kinds = {
            row[0]
            for row in observed_rows
            if row[0] != "package_quantity" and not target_has_measure[row[0]]
        }
        candidates = [
            row
            for row in profiles
            if row.document_id != source_document_id
            and all(
                (row.gross_utilization is not None)
                if kind == "gross_weight_kg"
                else (row.volume_utilization is not None or row.volume_m3 is not None)
                for kind in empirical_kinds
            )
        ]
        same_family = [row for row in candidates if row.equipment_family == capacity.family]
        if same_family:
            candidates = same_family
        if empirical_kinds and not candidates:
            raise ValueError(f"no empirical operational profile supports container {target_number}")
        digest = int.from_bytes(
            bytes.fromhex(
                sha256_bytes(f"{scenario_id}\0{target_number}\0operational-profile-v1".encode())
            ),
            "big",
        )
        selected_profile: EmpiricalOperationalProfile | None = None
        selected_empirical_surfaces: dict[int, str] = {}
        for offset in range(max(1, len(candidates))):
            if not empirical_kinds:
                break
            candidate = candidates[(digest + offset) % len(candidates)]
            rendered: dict[int, str] = {}
            for row_index, (operational_kind, _, source_surface, _, _) in enumerate(observed_rows):
                if operational_kind == "package_quantity":
                    continue
                if operational_kind not in empirical_kinds:
                    continue
                maximum = (
                    capacity.payload_kg
                    if operational_kind == "gross_weight_kg"
                    else capacity.volume_m3
                )
                source_maximum = (
                    source_capacity.payload_kg
                    if operational_kind == "gross_weight_kg"
                    else source_capacity.volume_m3
                )
                source_style = _parse_measurement_surface(source_surface, maximum=source_maximum)
                utilization = (
                    candidate.gross_utilization
                    if operational_kind == "gross_weight_kg"
                    else candidate.volume_utilization
                )
                absolute = (
                    candidate.gross_weight_kg
                    if operational_kind == "gross_weight_kg"
                    else candidate.volume_m3
                )
                target_value = (
                    utilization * maximum
                    if utilization is not None and maximum is not None
                    else absolute
                )
                if source_style is None or target_value is None:
                    rendered = {}
                    break
                rendered[row_index] = _render_measurement_surface(
                    target_value,
                    style=source_style,
                    maximum=maximum,
                )
            if rendered and all(
                target_surface != observed_rows[row_index][2]
                for row_index, target_surface in rendered.items()
            ):
                selected_profile = candidate
                selected_empirical_surfaces = rendered
                break
        if empirical_kinds and selected_profile is None:
            raise ValueError(
                f"empirical support cannot replace every stale measurement for {target_number}"
            )
        for row_index, (
            operational_kind,
            line_number,
            source_surface,
            evidence,
            grammar,
        ) in enumerate(observed_rows):
            if operational_kind == "package_quantity":
                try:
                    target_quantity = package_allocations[target_number]
                except KeyError as error:
                    raise ValueError(
                        f"target package allocations do not cover printed container {target_number}"
                    ) from error
                target_surface = _render_integer_surface(
                    target_quantity, source_surface=source_surface
                )
                maximum = None
                profile_document_id = None
                method: Literal[
                    "empirical_capacity_utilization_resample_v1",
                    "target_measure_allocation_v1",
                    "target_package_allocation_v1",
                ] = "target_package_allocation_v1"
            else:
                maximum = (
                    capacity.payload_kg
                    if operational_kind == "gross_weight_kg"
                    else capacity.volume_m3
                )
                source_maximum = (
                    source_capacity.payload_kg
                    if operational_kind == "gross_weight_kg"
                    else source_capacity.volume_m3
                )
                source_style = _parse_measurement_surface(source_surface, maximum=source_maximum)
                if source_style is None:
                    raise ValueError(f"cannot preserve source {operational_kind} numeric grammar")
                if target_has_measure[operational_kind]:
                    try:
                        target_value = allocated_values[operational_kind][target_number]
                    except KeyError as error:
                        raise ValueError(
                            f"target {operational_kind} has no allocation for printed container "
                            f"{target_number}"
                        ) from error
                    target_surface = _render_measurement_surface(
                        target_value,
                        style=source_style,
                        maximum=maximum,
                    )
                    profile_document_id = None
                    method = "target_measure_allocation_v1"
                else:
                    target_surface = selected_empirical_surfaces[row_index]
                    profile_document_id = cast(
                        EmpiricalOperationalProfile, selected_profile
                    ).document_id
                    method = "empirical_capacity_utilization_resample_v1"
            source_line = raw_lines[line_number - 1]
            if grammar == "inline_container_breakdown":
                inline = _RAW_INLINE_CONTAINER_MEASURES.search(source_line)
                if inline is None:
                    raise ValueError("inline operational measurement lost its source grammar")
                measurement_start = inline.start()
                following_number = next(
                    (
                        target_containers[index].get("containerNumber")
                        for index, source_number in enumerate(source_numbers)
                        if re.search(
                            rf"(?<![A-Z0-9]){re.escape(source_number)}(?![A-Z0-9])",
                            source_line[inline.end() :],
                            re.I,
                        )
                    ),
                    None,
                )
                if following_number is not None and not isinstance(following_number, str):
                    raise ValueError("following synthetic container has no containerNumber")
            else:
                measurement_match = _measurement_pattern(cast(Any, operational_kind)).search(
                    source_line
                )
                if measurement_match is None:
                    raise ValueError("labeled operational measurement lost its source grammar")
                measurement_start = measurement_match.start()
                following_number = None
            requirements.append(
                OperationalFlavorRequirement(
                    requirementId=f"operational-L{line_number:05d}-{operational_kind}",
                    kind=operational_kind,
                    sourceGrammar=grammar,
                    sourceLineId=f"L{line_number:05d}",
                    sourceMeasurementStartColumn=measurement_start,
                    sourceValueSurface=source_surface,
                    targetValueSurface=target_surface,
                    consistencyGroupId=f"{target_number}-{operational_kind}",
                    targetContainerNumber=target_number,
                    targetEquipmentFamily=capacity.family,
                    maximumValue=str(maximum) if maximum is not None else None,
                    sameLineFollowingContainerNumber=following_number,
                    empiricalProfileDocumentId=profile_document_id,
                    samplingMethod=method,
                    sourceEvidence=evidence,
                )
            )
    return tuple(requirements)


def _operational_flavor_requirements_rendered(
    value: str, requirements: Sequence[OperationalFlavorRequirement]
) -> bool:
    lines = value.splitlines()
    for requirement in requirements:
        line_number = _line_number(requirement.sourceLineId)
        if not 1 <= line_number <= len(lines):
            return False
        line = lines[line_number - 1]
        if requirement.sourceGrammar == "inline_container_breakdown":
            match = _RAW_INLINE_CONTAINER_MEASURES.search(line)
            group = {
                "gross_weight_kg": "gross",
                "volume_m3": "volume",
                "package_quantity": "packages",
            }[requirement.kind]
        else:
            match = _measurement_pattern(cast(Any, requirement.kind)).search(line)
            group = "value"
        if (
            match is None
            or match.start() != requirement.sourceMeasurementStartColumn
            or match.group(group) != requirement.targetValueSurface
        ):
            return False
        following = requirement.sameLineFollowingContainerNumber
        if following is not None:
            following_match = re.search(
                rf"(?<![A-Z0-9]){re.escape(following)}(?![A-Z0-9])", line, re.I
            )
            if following_match is None or following_match.start() < match.end():
                return False
    return True


def _derive_raw_auxiliary_identity_realizations(
    workspace: RewriteWorkspace,
    updated: str,
) -> tuple[AppliedRawAuxiliaryIdentityRealization, ...]:
    """Derive raw-agent audit receipts from the rewritten lines, never duplicate model output."""

    blocks_by_line: dict[int, list[tuple[str, int, str]]] = {}
    for line_number, identity, gap, principal, _ in _raw_agent_blocks(updated):
        blocks_by_line.setdefault(line_number, []).append((identity, gap, principal))
    applied: list[AppliedRawAuxiliaryIdentityRealization] = []
    rendered_by_group: dict[str, str] = {}
    for requirement in workspace.raw_auxiliary_identity_requirements:
        requirement_id = requirement.requirementId
        source_line = int(requirement_id.rsplit("-L", 1)[1])
        candidates = blocks_by_line.get(source_line, [])
        if len(candidates) != 1:
            raise ValueError(
                "raw auxiliary identity relationship is missing or ambiguous at its source line: "
                f"{requirement_id}"
            )
        rendered, observed_gap, principal = candidates[0]
        normalized_rendered = _semantic_normalize(rendered)
        rendered_lines = rendered.splitlines()
        if len(rendered_lines) != requirement.sourceIdentityLineCount or any(
            not line.strip() for line in rendered_lines
        ):
            raise ValueError(
                f"raw auxiliary identity must preserve its populated line count: {requirement_id}"
            )
        if observed_gap != requirement.gapLineCount:
            raise ValueError(
                f"raw auxiliary identity must preserve its gap line count: {requirement_id}"
            )
        if not _principal_surface_realizes_target(principal, requirement.targetPrincipalName):
            raise ValueError(
                "raw auxiliary identity must retain its exact target carrier principal: "
                f"{requirement_id}"
            )
        if len(normalized_rendered.split()) < 2:
            raise ValueError(
                f"raw auxiliary identity must contain at least two lexical tokens: {requirement_id}"
            )
        if normalized_rendered in {
            _semantic_normalize(requirement.sourceIdentity),
            _semantic_normalize(requirement.targetPrincipalName),
        }:
            raise ValueError(
                "raw auxiliary identity must be distinct from source and principal: "
                f"{requirement_id}"
            )
        rendered_key = _auxiliary_identity_token_key(rendered)
        previous = rendered_by_group.setdefault(requirement.consistencyGroupId, rendered_key)
        if previous != rendered_key:
            raise ValueError(
                "repeated raw auxiliary identity has inconsistent fictional realizations: "
                f"{requirement.consistencyGroupId}"
            )
        applied.append(
            AppliedRawAuxiliaryIdentityRealization(
                requirementId=requirement_id,
                relationship=requirement.relationship,
                sourceIdentity=requirement.sourceIdentity,
                sourceIdentityLineCount=requirement.sourceIdentityLineCount,
                gapLineCount=requirement.gapLineCount,
                consistencyGroupId=requirement.consistencyGroupId,
                targetPrincipalName=requirement.targetPrincipalName,
                renderedIdentity=rendered,
            )
        )
    return tuple(applied)


def _raw_auxiliary_identity_realization_rendered(
    value: str, realization: AppliedRawAuxiliaryIdentityRealization
) -> bool:
    """Require the fictional agent and target principal in the original agency grammar."""

    return any(
        _semantic_normalize(identity) == _semantic_normalize(realization.renderedIdentity)
        and gap == realization.gapLineCount
        and _principal_surface_realizes_target(principal, realization.targetPrincipalName)
        for _, identity, gap, principal, _ in _raw_agent_blocks(value)
    )


def _principal_surface_realizes_target(surface: str, target_name: str) -> bool:
    normalized_surface = _semantic_normalize(surface)
    normalized_target = _semantic_normalize(target_name)
    if normalized_surface == normalized_target:
        return True
    return any(
        normalized_surface.startswith(f"{normalized_target} {marker}")
        for marker in ("TRADING AS", "DOING BUSINESS AS", "T A", "D B A")
    )


def _raw_auxiliary_identity_realizations_rendered(
    value: str, realizations: Sequence[AppliedRawAuxiliaryIdentityRealization]
) -> bool:
    required: dict[tuple[str, int, str], int] = {}
    for realization in realizations:
        key = (
            _semantic_normalize(realization.renderedIdentity),
            realization.gapLineCount,
            _semantic_normalize(realization.targetPrincipalName),
        )
        required[key] = required.get(key, 0) + 1
    blocks = _raw_agent_blocks(value)
    return all(
        sum(
            _semantic_normalize(identity) == rendered
            and observed_gap == gap
            and _principal_surface_realizes_target(principal, target)
            for _, identity, observed_gap, principal, _ in blocks
        )
        == count
        for (rendered, gap, target), count in required.items()
    )


def apply_line_range_replacements(
    workspace: RewriteWorkspace,
    edits: Sequence[LineRangeReplacement],
    compound_party_flavor_realizations: Sequence[CompoundPartyFlavorRealization] = (),
) -> AtomicRewriteCommit:
    """Validate one complete line-range patch and commit it atomically."""

    if not edits:
        raise ValueError("at least one line-range replacement is required")
    applied_realizations = _validate_compound_party_flavor_realizations(
        workspace, compound_party_flavor_realizations
    )
    snapshot = workspace.current_text
    lines = snapshot.splitlines(keepends=True)
    if not lines:
        raise ValueError("current OCR has no addressable lines")
    default_newline = _default_newline(snapshot)
    ordered = sorted(
        edits,
        key=lambda row: (_line_number(row.startLineId), _line_number(row.endLineId)),
    )
    previous_end = 0
    applied: list[AppliedLineRangeReplacement] = []
    chunks: list[str] = []
    cursor = 0
    for index, edit in enumerate(ordered):
        start_line = _line_number(edit.startLineId)
        end_line = _line_number(edit.endLineId)
        if end_line < start_line:
            raise ValueError(f"edit[{index}] end line precedes its start line")
        if start_line <= previous_end:
            raise ValueError(f"edit[{index}] overlaps or duplicates a preceding line range")
        if end_line > len(lines):
            raise ValueError(
                f"edit[{index}] ends at line {end_line}, but current OCR has {len(lines)} lines"
            )
        start_index = start_line - 1
        end_index = end_line
        old_lines = lines[start_index:end_index]
        if any(_PAGE_MARKER.fullmatch(_line_body(line)) for line in old_lines):
            raise ValueError(f"edit[{index}] includes a protected page marker")
        new_text = edit.newText
        if new_text.endswith("\r\n"):
            new_text = new_text[:-2]
        elif new_text.endswith(("\r", "\n")):
            new_text = new_text[:-1]
        normalized_new = new_text.replace("\r\n", "\n").replace("\r", "\n")
        if any(_PAGE_MARKER.fullmatch(line) for line in normalized_new.split("\n")):
            raise ValueError(f"edit[{index}] introduces a protected page marker")
        old_bodies = tuple(_line_body(line) for line in old_lines)
        new_lines = tuple(normalized_new.split("\n")) if normalized_new else ()
        if len(new_lines) != len(old_bodies):
            raise ValueError(
                f"edit[{index}] changes source line count: selected={len(old_bodies)}, "
                f"output={len(new_lines)}; preserve one output line per source line"
            )
        old_blank = tuple(not line.strip() for line in old_bodies)
        new_blank = tuple(not line.strip() for line in new_lines)
        if new_blank != old_blank:
            raise ValueError(
                f"edit[{index}] changes occupied/blank line topology; preserve every source "
                "slot and synthesize coherent content in populated target-absent slots"
            )
        ending = _line_ending(old_lines[-1])
        replacement = normalized_new.replace("\n", default_newline)
        if replacement and ending:
            replacement += ending
        old_text = "".join(old_lines)
        if _placeholder_count(normalized_new) > _placeholder_count(old_text):
            raise ValueError(f"edit[{index}] introduces an unavailable/unknown placeholder")
        if replacement == old_text:
            previous_end = end_line
            continue
        chunks.append("".join(lines[cursor:start_index]))
        chunks.append(replacement)
        applied.append(
            AppliedLineRangeReplacement(
                startLine=start_line,
                endLine=end_line,
                startLineId=edit.startLineId,
                endLineId=edit.endLineId,
                oldText=old_text,
                newText=replacement,
                oldTextSha256=sha256_bytes(old_text.encode("utf-8")),
                sourceLinesReplaced=end_line - start_line + 1,
                outputLinesInserted=(len(normalized_new.split("\n")) if normalized_new else 0),
            )
        )
        cursor = end_index
        previous_end = end_line
    chunks.append("".join(lines[cursor:]))
    updated = "".join(chunks)
    if updated == snapshot:
        raise ValueError("the atomic line-range patch did not change the OCR")
    if _page_markers(updated) != _page_markers(workspace.original_text):
        raise ValueError("the line-range patch changes page markers or page order")
    if not _newline_convention(updated) <= _newline_convention(workspace.original_text):
        raise ValueError("the line-range patch introduces a different newline convention")
    if len(updated.splitlines()) != len(workspace.original_text.splitlines()):
        raise ValueError("the line-range patch changes the source line count")
    if _blank_line_topology(updated) != _blank_line_topology(workspace.original_text):
        raise ValueError("the line-range patch changes the source blank-line topology")
    if not _inline_slot_topology_preserved(workspace.original_text, updated):
        raise ValueError(
            "the line-range patch changes an inline labeled slot or its populated/empty state"
        )
    if not _structural_line_prefixes_preserved(workspace.original_text, updated):
        raise ValueError(
            "the line-range patch changes or removes a numbered structural line prefix"
        )
    isolated_formatting = _isolated_formatting_changes(workspace, updated)
    if isolated_formatting:
        raise ValueError(
            "the line-range patch changes punctuation or whitespace outside an actual value "
            f"change: {isolated_formatting[:6]}"
        )
    if not _source_semantic_role_surfaces_preserved(updated, workspace.source_role_hints):
        missing = [
            row.model_dump(mode="json")
            for row in workspace.source_role_hints
            if not _source_semantic_role_surfaces_preserved(updated, (row,))
        ]
        raise ValueError(
            f"the line-range patch changes a deterministic flattened-OCR role surface: {missing}"
        )
    if not _source_status_surfaces_preserved(updated, workspace.source_status_requirements):
        missing = [
            row.model_dump(mode="json")
            for row in workspace.source_status_requirements
            if updated.splitlines().count(row.sourceSurface) != row.sourceOccurrences
        ]
        raise ValueError(
            f"the line-range patch changes an unchanged source status surface: {missing}"
        )
    if _max_consecutive_blank_lines(updated) > (
        _max_consecutive_blank_lines(workspace.original_text) + 1
    ):
        lines = updated.splitlines()
        blank_index = next(
            (
                index
                for index in range(1, len(lines))
                if not lines[index].strip() and not lines[index - 1].strip()
            ),
            0,
        )
        context = "\n".join(lines[max(0, blank_index - 2) : blank_index + 3])
        raise ValueError(
            "the line-range patch introduces more than one additional blank line; "
            f"include the surrounding lines in one replacement; first gap context={context!r}"
        )
    if _placeholder_count(updated) > _placeholder_count(snapshot):
        raise ValueError("the line-range patch introduces an unavailable/unknown placeholder")
    duplicated_lines = _introduced_source_line_duplicates(workspace.original_text, updated)
    if duplicated_lines:
        raise ValueError(
            f"the line-range patch duplicates unchanged source lines: {list(duplicated_lines[:6])}"
        )
    if not _required_surfaces_rendered(updated, workspace.surface_requirements):
        missing = [
            row.model_dump(mode="json")
            for row in workspace.surface_requirements
            if row.targetSurface not in updated
            or (
                row.sourceSurface != row.targetSurface
                and row.sourceSurface
                not in {requirement.targetSurface for requirement in workspace.surface_requirements}
                and row.sourceSurface in updated
            )
        ]
        raise ValueError(
            f"the line-range patch violates deterministic date/HS rendering requirements: {missing}"
        )
    if not _jurisdictional_surfaces_rendered(updated, workspace.jurisdictional_requirements):
        missing = [
            row.model_dump(mode="json")
            for row in workspace.jurisdictional_requirements
            if _literal_phrase_pattern(row.sourceSurface).search(updated) is not None
            or _literal_phrase_pattern(row.targetSurface).search(updated) is None
        ]
        raise ValueError(
            "the line-range patch retains a stale named customs program or omits its coherent "
            f"generic role: {missing}"
        )
    missing_literals = _missing_target_literals(updated, workspace.target_literal_requirements)
    if missing_literals:
        raise ValueError(
            "the line-range patch omits changed target text: "
            f"{[row.model_dump(mode='json') for row in missing_literals]}"
        )
    if not _target_value_occurrences_exact(updated, workspace.target_value_occurrence_requirements):
        incorrect = [
            row.model_dump(mode="json")
            for row in workspace.target_value_occurrence_requirements
            if _target_value_occurrence_count(updated, row) != row.requiredOccurrences
        ]
        raise ValueError(
            f"the line-range patch duplicates or omits role-bound party values: {incorrect}"
        )
    if not _anchored_scalar_replacements_rendered(
        updated, workspace.anchored_scalar_replacement_requirements
    ):
        incorrect = [
            row.model_dump(mode="json")
            for row in workspace.anchored_scalar_replacement_requirements
            if not _anchored_scalar_replacements_rendered(updated, (row,))
        ]
        raise ValueError(
            f"the line-range patch moves or omits an exact changed label scalar: {incorrect}"
        )
    normalized_updated = _semantic_normalize(updated)
    applied_raw_identities = _derive_raw_auxiliary_identity_realizations(workspace, updated)
    for compound_realization in applied_realizations:
        if _semantic_normalize(compound_realization.renderedName) not in normalized_updated:
            raise ValueError(
                "compound-party flavor realization is not rendered in rewritten OCR: "
                f"{compound_realization.targetPath}"
            )
    for raw_realization in applied_raw_identities:
        if _semantic_normalize(raw_realization.sourceIdentity) in normalized_updated:
            raise ValueError(
                "raw auxiliary source identity remains after rewrite: "
                f"{raw_realization.requirementId}"
            )
        if _semantic_normalize(raw_realization.renderedIdentity) not in normalized_updated:
            raise ValueError(
                "raw auxiliary identity realization is not rendered in rewritten OCR: "
                f"{raw_realization.requirementId}"
            )
        if not _raw_auxiliary_identity_realization_rendered(updated, raw_realization):
            raise ValueError(
                "raw auxiliary identity does not preserve the agent-for-carrier relationship: "
                f"{raw_realization.requirementId}"
            )
    if not _raw_auxiliary_identity_realizations_rendered(updated, applied_raw_identities):
        raise ValueError(
            "raw auxiliary identity realization counts do not match the source agency topology"
        )
    if not _operational_flavor_requirements_rendered(
        updated, workspace.operational_flavor_requirements
    ):
        incorrect = [
            row.model_dump(mode="json")
            for row in workspace.operational_flavor_requirements
            if not _operational_flavor_requirements_rendered(updated, (row,))
        ]
        raise ValueError(
            "the line-range patch retains or misformats a raw-only operational measurement: "
            f"{incorrect}"
        )
    if not _deactivated_reefer_operation_coherent(workspace, updated):
        raise ValueError(
            "the line-range patch retains a positive temperature or plugging assertion after "
            "the target deactivates every reefer setpoint; rewrite each occupied source slot "
            "as coherent non-operating/ventilation flavor"
        )
    if not _cargo_flavor_rewrite_requirements_rendered(
        updated, workspace.cargo_flavor_rewrite_requirements
    ):
        incorrect = [
            row.model_dump(mode="json")
            for row in workspace.cargo_flavor_rewrite_requirements
            if not _cargo_flavor_rewrite_requirements_rendered(updated, (row,))
        ]
        raise ValueError(
            "the line-range patch leaves a source cargo-description line unchanged or omits the "
            f"target description: {incorrect}"
        )
    commit = AtomicRewriteCommit(
        beforeTextSha256=workspace.current_sha256,
        afterTextSha256=sha256_bytes(updated.encode("utf-8")),
        beforeTargetLabelSha256=workspace.current_target_sha256,
        afterTargetLabelSha256=workspace.current_target_sha256,
        compoundPartyFlavorRealizations=applied_realizations,
        rawAuxiliaryIdentityRealizations=applied_raw_identities,
        appliedReplacements=tuple(applied),
    )
    workspace.current_text = updated
    workspace.commits.append(commit)
    return commit


EditBatch = Annotated[list[LineRangeReplacement], Field(min_length=1, max_length=256)]


class AtomicRewriteArguments(BaseModel):
    """Audit mirror of the strict terminal output-function arguments."""

    model_config = _STRICT

    edits: EditBatch
    compoundPartyFlavorRealizations: Annotated[
        list[CompoundPartyFlavorRealization],
        Field(
            max_length=6,
            description=(
                "Exactly the auxiliary raw-text realizations listed in "
                "compoundPartyFlavorRequirements; empty when none are required."
            ),
        ),
    ]


def commit_atomic_rewrite(
    ctx: RunContext[RewriteWorkspace],
    edits: EditBatch,
    compoundPartyFlavorRealizations: Annotated[
        list[CompoundPartyFlavorRealization],
        Field(
            max_length=6,
            description=(
                "Exactly the auxiliary raw-text realizations listed in "
                "compoundPartyFlavorRequirements; empty when none are required."
            ),
        ),
    ],
) -> AtomicRewriteCommit:
    """Atomically apply the complete line-range patch and finish this editor pass.

    Args:
        ctx: Isolated mutable OCR workspace for this document.
        edits: Complete non-overlapping line-range patch against currentRawOcrLines.
    """

    try:
        return apply_line_range_replacements(
            ctx.deps,
            edits,
            compound_party_flavor_realizations=compoundPartyFlavorRealizations,
        )
    except ValueError as error:
        raise ModelRetry(
            "Atomic rewrite rejected and fully rolled back; zero edits were applied. "
            "Resubmit the complete patch against the unchanged current OCR, including every "
            f"previously attempted edit plus the correction. Cause: {error}"
        ) from error


def _terminal_editor_output(
    workspace: RewriteWorkspace,
    *,
    max_retries: int,
) -> ToolOutput[AtomicRewriteCommit]:
    """Build a strict output tool whose line IDs are limited to editable source lines.

    Blank lines and page markers are deliberately omitted from the editor payload. Encoding the
    same finite set as JSON Schema enums makes a model-selected omitted line impossible under
    provider strict decoding, instead of paying for a post-generation retry that can only fail.
    The complete patch still passes through the transactional semantic validators.
    """

    allowed_line_ids = tuple(
        _line_id(index, line)
        for index, line in enumerate(workspace.current_text.splitlines(keepends=True), start=1)
        if (body := _line_body(line)).strip() and _PAGE_MARKER.fullmatch(body) is None
    )
    if not allowed_line_ids:
        raise ValueError("current OCR has no editable populated lines")
    allowed_line_id_type = cast(Any, Literal).__getitem__(allowed_line_ids)
    model_suffix = sha256_bytes("\n".join(allowed_line_ids).encode("utf-8"))[:12]
    allowed_line_replacement = create_model(
        f"AllowedLineRangeReplacement_{model_suffix}",
        __config__=_STRICT,
        startLineId=(
            allowed_line_id_type,
            Field(description="First populated editable line ID in this replacement."),
        ),
        endLineId=(
            allowed_line_id_type,
            Field(description="Last populated editable line ID in this replacement."),
        ),
        newText=(
            NonEmptyText,
            Field(description="Replacement text with exactly one output line per selected line."),
        ),
    )
    dynamic_edit_batch = cast(Any, Annotated).__class_getitem__(
        (
            list.__class_getitem__(allowed_line_replacement),
            Field(min_length=1, max_length=256),
        )
    )
    compound_batch = Annotated[
        list[CompoundPartyFlavorRealization],
        Field(
            max_length=6,
            description=(
                "Exactly the auxiliary raw-text realizations listed in "
                "compoundPartyFlavorRequirements; empty when none are required."
            ),
        ),
    ]

    def commit_for_allowed_lines(
        ctx: RunContext[RewriteWorkspace],
        edits: list[BaseModel],
        compoundPartyFlavorRealizations: list[CompoundPartyFlavorRealization],
    ) -> AtomicRewriteCommit:
        parsed_edits = [
            LineRangeReplacement.model_validate(row.model_dump(mode="python")) for row in edits
        ]
        return commit_atomic_rewrite(
            ctx,
            parsed_edits,
            compoundPartyFlavorRealizations,
        )

    commit_for_allowed_lines.__name__ = "commit_atomic_rewrite"
    commit_for_allowed_lines.__doc__ = commit_atomic_rewrite.__doc__
    commit_for_allowed_lines.__annotations__ = {
        "ctx": RunContext[RewriteWorkspace],
        "edits": dynamic_edit_batch,
        "compoundPartyFlavorRealizations": compound_batch,
        "return": AtomicRewriteCommit,
    }
    return ToolOutput(
        commit_for_allowed_lines,
        name="commit_atomic_rewrite",
        description=(
            "Submit the complete line-addressed patch once. It is validated and applied "
            "atomically; a successful call ends the editor pass."
        ),
        max_retries=max_retries,
        strict=True,
        sequential=True,
    )


def _residual_candidates(
    workspace: RewriteWorkspace, leaves: Sequence[ChangedLeaf]
) -> tuple[ResidualCandidate, ...]:
    candidates: list[ResidualCandidate] = []
    for leaf in leaves:
        if not leaf.requiresTextEdit or leaf.sourceValue == leaf.targetValue:
            continue
        if not isinstance(leaf.sourceValue, (str, int, float)) or isinstance(
            leaf.sourceValue, bool
        ):
            continue
        if _source_equipment_surface_matches_target(workspace, leaf):
            continue
        surface = str(leaf.sourceValue)
        if len(surface.strip()) < 4:
            continue
        offsets = _all_occurrences(workspace.current_text, surface)
        if not offsets or not _all_occurrences(workspace.original_text, surface):
            continue
        offset = offsets[0]
        start = max(0, offset - 120)
        end = min(len(workspace.current_text), offset + len(surface) + 120)
        candidates.append(
            ResidualCandidate(
                targetPath=leaf.path,
                sourceValue=leaf.sourceValue,
                currentOccurrences=len(offsets),
                currentEvidence=workspace.current_text[start:end],
            )
        )
    return tuple(candidates)


def _source_equipment_surface_matches_target(
    workspace: RewriteWorkspace, leaf: ChangedLeaf
) -> bool:
    """Recognize a v3 printed equipment surface that already realizes v5 semantics."""

    match = _CONTAINER_TYPE_DESCRIPTION_PATH.fullmatch(leaf.path)
    if match is None or not isinstance(leaf.sourceValue, str):
        return False
    patch = workspace.current_target_label.get("documentPatch")
    if not isinstance(patch, dict):
        return False
    containers = patch.get("containers")
    index = int(match.group(1))
    if not isinstance(containers, list) or index >= len(containers):
        return False
    target = containers[index]
    if not isinstance(target, dict):
        return False
    size = target.get("sizeCategory")
    type_category = target.get("typeCategory")
    if not isinstance(size, str) or not isinstance(type_category, str):
        return False
    reviewed = review_source_equipment_surface(
        leaf.sourceValue,
        temperature_present=target.get("temperatureSetpoint") is not None,
    )
    return (
        reviewed.resolution == "reviewed_source_grammar"
        and reviewed.size_category == size
        and reviewed.type_category == type_category
    )


def _target_equipment_semantics(
    workspace: RewriteWorkspace, index: int
) -> dict[str, JsonValue] | None:
    patch = workspace.current_target_label.get("documentPatch")
    containers = patch.get("containers") if isinstance(patch, dict) else None
    if not isinstance(containers, list) or index >= len(containers):
        return None
    target = containers[index]
    if not isinstance(target, dict):
        return None
    size = target.get("sizeCategory")
    type_category = target.get("typeCategory")
    if not isinstance(size, str) or not isinstance(type_category, str):
        return None
    return {
        "sizeCategory": size,
        "typeCategory": type_category,
    }


def label_change_contract(workspace: RewriteWorkspace) -> tuple[LabelChangeDirective, ...]:
    """Project raw schema deltas into concise, printable semantic directives.

    Source labels use a reviewed free-text container ``typeDescription`` while target labels use
    semantic size/type categories.  Treating those three leaves independently caused models to
    print enum names or delete already-correct surfaces.  This projection makes that cross-schema
    relationship explicit while leaving every other reviewed-label change exact and auditable.
    """

    leaves = changed_leaves(workspace.source_label, workspace.current_target_label)
    suppressed_paths: set[str] = set()
    projected_by_path: dict[str, LabelChangeDirective] = {}
    for leaf in leaves:
        match = _CONTAINER_TYPE_DESCRIPTION_PATH.fullmatch(leaf.path)
        if match is None or not isinstance(leaf.sourceValue, str):
            continue
        index = int(match.group(1))
        target_semantics = _target_equipment_semantics(workspace, index)
        if target_semantics is None:
            continue
        projected_path = f"documentPatch.containers[{index}].printedEquipmentSurface"
        projected_by_path[leaf.path] = LabelChangeDirective(
            path=projected_path,
            action=(
                "preserve_equivalent_equipment_surface"
                if _source_equipment_surface_matches_target(workspace, leaf)
                else "replace_equipment_surface"
            ),
            sourceValue=leaf.sourceValue,
            targetValue=target_semantics,
        )
        suppressed_paths.update(
            {
                leaf.path,
                f"documentPatch.containers[{index}].sizeCategory",
                f"documentPatch.containers[{index}].typeCategory",
            }
        )

    projected_indices = {
        int(match.group(1))
        for leaf in leaves
        if (match := _CONTAINER_TYPE_DESCRIPTION_PATH.fullmatch(leaf.path)) is not None
        and leaf.path in suppressed_paths
    }
    target_category_triggers: dict[int, str] = {}
    for leaf in leaves:
        match = _CONTAINER_SIZE_CATEGORY_PATH.fullmatch(leaf.path)
        if match is None:
            match = _CONTAINER_TYPE_CATEGORY_PATH.fullmatch(leaf.path)
        if match is None:
            continue
        index = int(match.group(1))
        if index in projected_indices or index in target_category_triggers:
            continue
        target_semantics = _target_equipment_semantics(workspace, index)
        if target_semantics is None:
            continue
        projected_path = f"documentPatch.containers[{index}].printedEquipmentSurface"
        projected_by_path[leaf.path] = LabelChangeDirective(
            path=projected_path,
            action="add_equipment_surface",
            sourceValue=None,
            targetValue=target_semantics,
        )
        target_category_triggers[index] = leaf.path
        suppressed_paths.update(
            {
                f"documentPatch.containers[{index}].sizeCategory",
                f"documentPatch.containers[{index}].typeCategory",
            }
        )

    directives: list[LabelChangeDirective] = []
    emitted_projections: set[str] = set()
    for leaf in leaves:
        projection = projected_by_path.get(leaf.path)
        if projection is not None and projection.path not in emitted_projections:
            directives.append(projection)
            emitted_projections.add(projection.path)
            continue
        if leaf.path in suppressed_paths or not leaf.requiresTextEdit:
            continue
        if leaf.sourcePresent and leaf.targetPresent:
            action: Literal["replace", "deactivate_extractable_fact_preserve_slot", "add"] = (
                "replace"
            )
        elif leaf.sourcePresent:
            action = "deactivate_extractable_fact_preserve_slot"
        else:
            action = "add"
        directives.append(
            LabelChangeDirective(
                path=leaf.path,
                action=action,
                sourceValue=leaf.sourceValue,
                targetValue=leaf.targetValue,
            )
        )
    return tuple(directives)


def deterministic_rewrite_audit(
    workspace: RewriteWorkspace, leaves: Sequence[ChangedLeaf]
) -> DeterministicRewriteAudit:
    normalized_current = _semantic_normalize(workspace.current_text)
    applied_raw_identities = {
        row.requirementId: row
        for commit in workspace.commits
        for row in commit.rawAuxiliaryIdentityRealizations
    }
    return DeterministicRewriteAudit(
        currentTextSha256=workspace.current_sha256,
        textChanged=workspace.current_text != workspace.original_text,
        pageMarkersPreserved=_page_markers(workspace.current_text)
        == _page_markers(workspace.original_text),
        newlineConventionPreserved=_newline_convention(workspace.current_text)
        <= _newline_convention(workspace.original_text),
        lineCountPreserved=(
            len(workspace.current_text.splitlines()) == len(workspace.original_text.splitlines())
        ),
        blankLineTopologyPreserved=(
            _blank_line_topology(workspace.current_text)
            == _blank_line_topology(workspace.original_text)
        ),
        inlineSlotTopologyPreserved=_inline_slot_topology_preserved(
            workspace.original_text, workspace.current_text
        ),
        structuralLinePrefixesPreserved=_structural_line_prefixes_preserved(
            workspace.original_text, workspace.current_text
        ),
        isolatedFormattingChangesAbsent=not _isolated_formatting_changes(
            workspace, workspace.current_text
        ),
        sourceStatusSurfacesPreserved=_source_status_surfaces_preserved(
            workspace.current_text, workspace.source_status_requirements
        ),
        blankLineDepthBounded=_max_consecutive_blank_lines(workspace.current_text)
        <= _max_consecutive_blank_lines(workspace.original_text) + 1,
        noPlaceholderIntroduced=_placeholder_count(workspace.current_text)
        <= _placeholder_count(workspace.original_text),
        requiredSurfacesRendered=_required_surfaces_rendered(
            workspace.current_text, workspace.surface_requirements
        ),
        jurisdictionalSurfacesRendered=_jurisdictional_surfaces_rendered(
            workspace.current_text, workspace.jurisdictional_requirements
        ),
        requiredTargetLiteralsRendered=not _missing_target_literals(
            workspace.current_text, workspace.target_literal_requirements
        ),
        targetValueOccurrencesExact=_target_value_occurrences_exact(
            workspace.current_text, workspace.target_value_occurrence_requirements
        ),
        anchoredScalarReplacementsRendered=_anchored_scalar_replacements_rendered(
            workspace.current_text,
            workspace.anchored_scalar_replacement_requirements,
        ),
        rawAuxiliaryIdentitiesReplaced=all(
            _semantic_normalize(requirement.sourceIdentity) not in normalized_current
            and requirement.requirementId in applied_raw_identities
            for requirement in workspace.raw_auxiliary_identity_requirements
        )
        and _raw_auxiliary_identity_realizations_rendered(
            workspace.current_text, tuple(applied_raw_identities.values())
        ),
        operationalFlavorRendered=_operational_flavor_requirements_rendered(
            workspace.current_text, workspace.operational_flavor_requirements
        ),
        deactivatedReeferOperationCoherent=_deactivated_reefer_operation_coherent(
            workspace, workspace.current_text
        ),
        cargoFlavorRewritten=_cargo_flavor_rewrite_requirements_rendered(
            workspace.current_text, workspace.cargo_flavor_rewrite_requirements
        ),
        residualCandidates=_residual_candidates(workspace, leaves),
    )


def _review_audit_payload(audit: DeterministicRewriteAudit) -> dict[str, JsonValue]:
    """Keep deterministic signals without echoing OCR evidence already in the prompt."""

    return {
        "textChanged": audit.textChanged,
        "pageMarkersPreserved": audit.pageMarkersPreserved,
        "newlineConventionPreserved": audit.newlineConventionPreserved,
        "lineCountPreserved": audit.lineCountPreserved,
        "blankLineTopologyPreserved": audit.blankLineTopologyPreserved,
        "inlineSlotTopologyPreserved": audit.inlineSlotTopologyPreserved,
        "structuralLinePrefixesPreserved": audit.structuralLinePrefixesPreserved,
        "sourceStatusSurfacesPreserved": audit.sourceStatusSurfacesPreserved,
        "blankLineDepthBounded": audit.blankLineDepthBounded,
        "noPlaceholderIntroduced": audit.noPlaceholderIntroduced,
        "requiredSurfacesRendered": audit.requiredSurfacesRendered,
        "jurisdictionalSurfacesRendered": audit.jurisdictionalSurfacesRendered,
        "requiredTargetLiteralsRendered": audit.requiredTargetLiteralsRendered,
        "targetValueOccurrencesExact": audit.targetValueOccurrencesExact,
        "anchoredScalarReplacementsRendered": audit.anchoredScalarReplacementsRendered,
        "rawAuxiliaryIdentitiesReplaced": audit.rawAuxiliaryIdentitiesReplaced,
        "operationalFlavorRendered": audit.operationalFlavorRendered,
        "deactivatedReeferOperationCoherent": audit.deactivatedReeferOperationCoherent,
        "cargoFlavorRewritten": audit.cargoFlavorRewritten,
        "residualCandidates": cast(
            JsonValue,
            [
                {
                    "targetPath": row.targetPath,
                    "sourceValue": row.sourceValue,
                    "currentOccurrences": row.currentOccurrences,
                }
                for row in audit.residualCandidates
            ],
        ),
    }


def _combine_usage(rows: Sequence[LinguisticUsageReceipt]) -> LinguisticUsageReceipt:
    return LinguisticUsageReceipt(
        requests=sum(row.requests for row in rows),
        providerResponseIds=tuple(value for row in rows for value in row.providerResponseIds),
        finishReasons=tuple(value for row in rows for value in row.finishReasons),
        inputTokens=sum(row.inputTokens for row in rows),
        cacheReadTokens=sum(row.cacheReadTokens for row in rows),
        cacheWriteTokens=sum(row.cacheWriteTokens for row in rows),
        outputTokens=sum(row.outputTokens for row in rows),
        reasoningTokens=sum(row.reasoningTokens for row in rows),
        visibleOutputTokens=sum(row.visibleOutputTokens for row in rows),
        estimatedCostUsd=sum((row.estimatedCostUsd for row in rows), Decimal(0)),
    )


def _usage_by_stage(
    bundles: Sequence[RewriteCycleBundle],
) -> dict[str, LinguisticUsageReceipt]:
    grouped: dict[str, list[LinguisticUsageReceipt]] = {
        "draft_editor": [],
        "correction_editor": [],
        "semantic_reviewer": [],
    }
    for bundle in bundles:
        for record in bundle.stageRecords:
            grouped[record.stage].append(record.usage)
    return {
        stage: _combine_usage(rows) if rows else _empty_usage() for stage, rows in grouped.items()
    }


def _settings(
    provider: Any,
    *,
    stage: Literal["editor", "reviewer"],
    prompt_sha256: str,
) -> OpenAIResponsesModelSettings:
    settings = openai_responses_settings(provider)
    settings["parallel_tool_calls"] = False
    # Retries need the validation feedback, not a replay of earlier hidden reasoning. Keeping
    # reasoning scoped to the current turn avoids paying input tokens for stale thought traces.
    settings["openai_reasoning_context"] = "current_turn"
    # Only the stable system prompt, strict schema, and marker preceding CachePoint are cached.
    # The document-specific suffix is deliberately outside the explicit breakpoint.
    settings["openai_prompt_cache_key"] = (
        f"dococr:rewrite15:{provider.model}:{stage[0]}:{prompt_sha256[:12]}"
    )
    settings["openai_prompt_cache_options"] = {"mode": "explicit", "ttl": "30m"}
    # Provider responses are retained for audit, so let PydanticAI continue validation-repair
    # requests from the stored response instead of replaying the same prompt and tool call. This
    # applies only inside one Agent.run; independent editor/reviewer passes remain isolated.
    if provider.store_responses:
        settings["openai_previous_response_id"] = "auto"
    return settings


def _prompt_content(
    *,
    stage: Literal["editor", "reviewer"],
    immutable: Mapping[str, JsonValue],
    active: Mapping[str, JsonValue],
) -> tuple[UserContent, ...]:
    """Place every case-specific byte after the explicit provider cache boundary."""

    dynamic = (
        "IMMUTABLE CASE CONTEXT\n"
        + json.dumps(immutable, ensure_ascii=False, separators=(",", ":"))
        + "\nACTIVE PASS\n"
        + json.dumps(active, ensure_ascii=False, separators=(",", ":"))
    )
    return (
        f"Stable synthetic B/L atomic {stage} contract version 15.",
        CachePoint(),
        dynamic,
    )


def _prompt_payload_sha256(content: Sequence[UserContent]) -> str:
    serializable: list[JsonValue] = []
    for value in content:
        if isinstance(value, str):
            serializable.append(value)
        elif isinstance(value, CachePoint):
            serializable.append({"kind": value.kind, "ttl": value.ttl})
        else:
            raise TypeError(f"unsupported prompt payload content: {type(value).__name__}")
    return sha256_bytes(canonical_json_bytes(serializable))


class RewriteRuntime:
    def __init__(
        self,
        *,
        config: SynthesisRawTextRewriteCycleProbeConfig,
        editor_prompt: str,
        reviewer_prompt: str,
        editor_model: OpenAIResponsesModel,
        reviewer_model: OpenAIResponsesModel,
        target_integrity_resources: TargetIntegrityResources,
        limiter: ConcurrencyLimiter,
        staged: StagedArtifactRun,
    ) -> None:
        self.config = config
        self.editor_prompt = editor_prompt
        self.reviewer_prompt = reviewer_prompt
        self.editor_model = editor_model
        self.reviewer_model = reviewer_model
        self.target_integrity_resources = target_integrity_resources
        self.limiter = limiter
        self.staged = staged

    @staticmethod
    def _case_prefix(state: RewriteState) -> str:
        return f"cases/{state.case_number:02d}-{state.document_id}"

    def _persist_stage(self, state: RewriteState, record: RewriteStageRecord) -> None:
        if record.sequence != len(state.stages) + 1:
            raise RuntimeError("rewrite stage sequence is not contiguous")
        relative = f"{self._case_prefix(state)}/stages/{record.sequence:02d}-{record.stage}.json"
        self.staged.publish_json(relative, record.model_dump(mode="json"))
        state.stages.append(record)

    def cost(self, state: RewriteState) -> Decimal:
        return sum((row.usage.estimatedCostUsd for row in state.stages), Decimal(0))

    def cost_limited(self, state: RewriteState) -> bool:
        return self.cost(state) >= Decimal(
            str(self.config.workflow.max_total_estimated_cost_usd_per_case)
        )

    async def run_editor(
        self,
        state: RewriteState,
        *,
        phase: Literal["draft", "correction"],
        feedback: SemanticReviewReceipt | None,
    ) -> AtomicRewriteCommit | None:
        provider = self.config.providers.editor
        change_contract = label_change_contract(state.workspace)
        compact_contract = compact_label_change_contract(change_contract)
        immutable: dict[str, JsonValue] = {
            "documentId": state.document_id,
            "syntheticTargetLabel": cast(JsonValue, state.target_label),
            "targetRouteJurisdictions": cast(
                JsonValue,
                _target_route_jurisdictions(state.target_label, self.target_integrity_resources),
            ),
            "operationalCapacityLimits": cast(
                JsonValue,
                self.config.target_integrity.transport_capacity.model_dump(mode="json"),
            ),
            "labelChangeContract": cast(
                JsonValue, [row.model_dump(mode="json") for row in compact_contract]
            ),
            "surfaceRenderingRequirements": cast(
                JsonValue,
                [row.model_dump(mode="json") for row in state.workspace.surface_requirements],
            ),
            "targetValueOccurrenceRequirements": cast(
                JsonValue,
                [
                    row.model_dump(mode="json")
                    for row in state.workspace.target_value_occurrence_requirements
                ],
            ),
            "anchoredScalarReplacementRequirements": cast(
                JsonValue,
                [
                    row.model_dump(mode="json")
                    for row in state.workspace.anchored_scalar_replacement_requirements
                ],
            ),
            "deterministicPrefills": cast(
                JsonValue,
                [
                    {
                        "lineId": row.lineId,
                        "targetPaths": list(row.targetPaths),
                        "targetSurface": row.targetSurface,
                    }
                    for row in state.workspace.deterministic_prefills
                ],
            ),
            "sourceSemanticRoleHints": cast(
                JsonValue,
                [row.model_dump(mode="json") for row in state.workspace.source_role_hints],
            ),
            "inlineSlotTopologyRequirements": cast(
                JsonValue,
                [row.model_dump(mode="json") for row in state.workspace.inline_slot_requirements],
            ),
            "sourceStatusPreservationRequirements": cast(
                JsonValue,
                [row.model_dump(mode="json") for row in state.workspace.source_status_requirements],
            ),
            "compoundPartyFlavorRequirements": cast(
                JsonValue,
                [
                    row.model_dump(mode="json")
                    for row in _active_compound_party_flavor_requirements(state.workspace)
                ],
            ),
            "rawAuxiliaryIdentityRequirements": cast(
                JsonValue,
                [
                    row.model_dump(mode="json")
                    for row in state.workspace.raw_auxiliary_identity_requirements
                ],
            ),
            "operationalFlavorRequirements": cast(
                JsonValue,
                [
                    row.model_dump(mode="json")
                    for row in state.workspace.operational_flavor_requirements
                ],
            ),
            "cargoFlavorRewriteRequirements": cast(
                JsonValue,
                [
                    row.model_dump(mode="json", exclude={"sourceSurfaces"})
                    for row in state.workspace.cargo_flavor_rewrite_requirements
                ],
            ),
            "jurisdictionalSurfaceRequirements": cast(
                JsonValue,
                [
                    row.model_dump(mode="json")
                    for row in state.workspace.jurisdictional_requirements
                ],
            ),
            # The correction pass edits the complete current OCR and already receives the
            # authoritative label delta plus grounded reviewer findings. Re-sending the nearly
            # identical original OCR would only increase input cost and reasoning noise.
            "editableSourceRawOcrLines": (
                editable_indexed_ocr_lines(state.workspace.current_text)
                if phase == "draft"
                else None
            ),
        }
        active: dict[str, JsonValue] = {
            "task": "Rewrite current OCR to faithfully render the synthetic target label.",
            "phase": phase,
            "editableCurrentRawOcrLines": (
                editable_indexed_ocr_lines(state.workspace.current_text)
                if phase == "correction"
                else None
            ),
            "reviewFeedback": cast(
                JsonValue, feedback.model_dump(mode="json") if feedback is not None else None
            ),
        }
        payload = _prompt_content(stage="editor", immutable=immutable, active=active)
        settings = _settings(
            provider,
            stage="editor",
            prompt_sha256=self.config.prompts.editor.sha256,
        )
        agent = Agent[RewriteWorkspace, AtomicRewriteCommit](
            self.editor_model,
            deps_type=RewriteWorkspace,
            output_type=_terminal_editor_output(
                state.workspace,
                max_retries=self.config.workflow.editor_output_retries,
            ),
            system_prompt=self.editor_prompt,
            model_settings=settings,
            retries={"output": self.config.workflow.editor_output_retries},
            max_concurrency=self.limiter,
            name="synthetic-bl-atomic-raw-text-editor",
        )
        before_sha = state.workspace.current_sha256
        started_at = datetime.now(UTC)
        commit: AtomicRewriteCommit | None = None
        error_type: str | None = None
        error_message: str | None = None
        messages: JsonValue = []
        with capture_run_messages() as captured:
            try:
                result = await agent.run(
                    payload,
                    deps=state.workspace,
                    usage_limits=UsageLimits(
                        request_limit=self.config.workflow.max_editor_requests_per_pass,
                        output_tokens_limit=(
                            provider.max_output_tokens
                            * self.config.workflow.max_editor_requests_per_pass
                        ),
                    ),
                )
                commit = result.output
            except Exception as error:
                error_type = type(error).__name__
                error_message = str(error)
            finally:
                messages = model_messages(captured)
        responses = tuple(row for row in captured if isinstance(row, ModelResponse))
        record = RewriteStageRecord(
            schemaVersion=15,
            sequence=len(state.stages) + 1,
            stage="draft_editor" if phase == "draft" else "correction_editor",
            correctionCycle=sum(row.stage == "correction_editor" for row in state.stages),
            providerModel=provider.model,
            reasoningEffort=provider.reasoning_effort,
            startedAt=started_at,
            completedAt=datetime.now(UTC),
            inputPayloadSha256=_prompt_payload_sha256(payload),
            currentTextSha256Before=before_sha,
            currentTextSha256After=state.workspace.current_sha256,
            editorCommit=commit,
            semanticReview=None,
            deterministicAudit=None,
            usage=usage_receipt(responses, provider.pricing),
            modelMessages=messages,
            errorType=error_type,
            errorMessage=error_message,
        )
        self._persist_stage(state, record)
        return commit

    async def run_reviewer(
        self, state: RewriteState, audit: DeterministicRewriteAudit
    ) -> SemanticReviewReceipt | None:
        provider = self.config.providers.reviewer
        change_contract = label_change_contract(state.workspace)
        compact_contract = compact_label_change_contract(change_contract)
        immutable: dict[str, JsonValue] = {
            "documentId": state.document_id,
            "reviewedSourceLabel": cast(JsonValue, state.source_label),
            "syntheticTargetLabel": cast(JsonValue, state.target_label),
            "targetRouteJurisdictions": cast(
                JsonValue,
                _target_route_jurisdictions(state.target_label, self.target_integrity_resources),
            ),
            "operationalCapacityLimits": cast(
                JsonValue,
                self.config.target_integrity.transport_capacity.model_dump(mode="json"),
            ),
            "labelChangeContract": cast(
                JsonValue, [row.model_dump(mode="json") for row in compact_contract]
            ),
            "surfaceRenderingRequirements": cast(
                JsonValue,
                [row.model_dump(mode="json") for row in state.workspace.surface_requirements],
            ),
            "targetValueOccurrenceRequirements": cast(
                JsonValue,
                [
                    row.model_dump(mode="json")
                    for row in state.workspace.target_value_occurrence_requirements
                ],
            ),
            "anchoredScalarReplacementRequirements": cast(
                JsonValue,
                [
                    row.model_dump(mode="json")
                    for row in state.workspace.anchored_scalar_replacement_requirements
                ],
            ),
            "sourceSemanticRoleHints": cast(
                JsonValue,
                [row.model_dump(mode="json") for row in state.workspace.source_role_hints],
            ),
            "inlineSlotTopologyRequirements": cast(
                JsonValue,
                [row.model_dump(mode="json") for row in state.workspace.inline_slot_requirements],
            ),
            "sourceStatusPreservationRequirements": cast(
                JsonValue,
                [row.model_dump(mode="json") for row in state.workspace.source_status_requirements],
            ),
            "compoundPartyFlavorRequirements": cast(
                JsonValue,
                [
                    row.model_dump(mode="json")
                    for row in compound_party_flavor_requirements(
                        state.source_label, state.target_label
                    )
                ],
            ),
            "rawAuxiliaryIdentityRequirements": cast(
                JsonValue,
                [
                    row.model_dump(mode="json")
                    for row in state.workspace.raw_auxiliary_identity_requirements
                ],
            ),
            "operationalFlavorRequirements": cast(
                JsonValue,
                [
                    row.model_dump(mode="json")
                    for row in state.workspace.operational_flavor_requirements
                ],
            ),
            "cargoFlavorRewriteRequirements": cast(
                JsonValue,
                [
                    row.model_dump(mode="json", exclude={"sourceSurfaces"})
                    for row in state.workspace.cargo_flavor_rewrite_requirements
                ],
            ),
            "jurisdictionalSurfaceRequirements": cast(
                JsonValue,
                [
                    row.model_dump(mode="json")
                    for row in state.workspace.jurisdictional_requirements
                ],
            ),
            "sourceRawOcr": state.workspace.original_text,
        }
        active: dict[str, JsonValue] = {
            "task": "Independently accept or request exact corrections to the rewritten OCR.",
            "rewrittenRawOcr": state.workspace.current_text,
            "deterministicAudit": cast(JsonValue, _review_audit_payload(audit)),
        }
        payload = _prompt_content(stage="reviewer", immutable=immutable, active=active)
        settings = _settings(
            provider,
            stage="reviewer",
            prompt_sha256=self.config.prompts.reviewer.sha256,
        )
        agent = Agent[None, SemanticReviewReceipt](
            self.reviewer_model,
            output_type=NativeOutput(
                SemanticReviewReceipt,
                name="synthetic_bl_atomic_rewrite_review",
                description="Compact evidence-grounded semantic acceptance or correction list.",
                strict=True,
            ),
            system_prompt=self.reviewer_prompt,
            model_settings=settings,
            retries={"output": self.config.workflow.reviewer_output_retries},
            max_concurrency=self.limiter,
            name="synthetic-bl-atomic-raw-text-reviewer",
        )

        @agent.output_validator
        def validate_output(output: SemanticReviewReceipt) -> SemanticReviewReceipt:
            try:
                for finding in output.findings:
                    for evidence in finding.currentEvidence:
                        if not _evidence_occurs(evidence, state.workspace.current_text):
                            raise ValueError("current review evidence is not grounded")
                    if _finding_misclassifies_anonymous_equipment(finding, state.workspace):
                        raise ValueError(
                            "reviewer classified a deterministic anonymous equipment count as marks"
                        )
                    if _finding_conflicts_with_equipment_authority(finding, state.workspace):
                        raise ValueError(
                            "reviewer equipment finding conflicts with the reviewed surface grammar"
                        )
                    if _finding_conflicts_with_surface_authority(finding, state.workspace):
                        raise ValueError(
                            "reviewer finding conflicts with a deterministic derived surface"
                        )
                    if _finding_conflicts_with_anchored_scalar_authority(finding, state.workspace):
                        raise ValueError(
                            "reviewer reclassified a line-bound target scalar as auxiliary flavor"
                        )
                    if _finding_grounds_format_damage_only_in_unchanged_text(
                        finding, state.workspace
                    ):
                        raise ValueError(
                            "reviewer formatting finding quotes only source-identical text"
                        )
                if output.verdict == "pass" and not audit.core_passed:
                    raise ValueError("review cannot pass a failing deterministic core audit")
                return output
            except ValueError as error:
                raise ModelRetry(f"Semantic review rejected: {error}") from error

        before_sha = state.workspace.current_sha256
        started_at = datetime.now(UTC)
        review: SemanticReviewReceipt | None = None
        error_type: str | None = None
        error_message: str | None = None
        messages: JsonValue = []
        with capture_run_messages() as captured:
            try:
                result = await agent.run(
                    payload,
                    usage_limits=UsageLimits(
                        request_limit=self.config.workflow.max_reviewer_requests_per_cycle,
                        output_tokens_limit=(
                            provider.max_output_tokens
                            * self.config.workflow.max_reviewer_requests_per_cycle
                        ),
                    ),
                )
                review = result.output
            except Exception as error:
                error_type = type(error).__name__
                error_message = str(error)
            finally:
                messages = model_messages(captured)
        responses = tuple(row for row in captured if isinstance(row, ModelResponse))
        record = RewriteStageRecord(
            schemaVersion=15,
            sequence=len(state.stages) + 1,
            stage="semantic_reviewer",
            correctionCycle=sum(row.stage == "correction_editor" for row in state.stages),
            providerModel=provider.model,
            reasoningEffort=provider.reasoning_effort,
            startedAt=started_at,
            completedAt=datetime.now(UTC),
            inputPayloadSha256=_prompt_payload_sha256(payload),
            currentTextSha256Before=before_sha,
            currentTextSha256After=state.workspace.current_sha256,
            editorCommit=None,
            semanticReview=review,
            deterministicAudit=audit,
            usage=usage_receipt(responses, provider.pricing),
            modelMessages=messages,
            errorType=error_type,
            errorMessage=error_message,
        )
        self._persist_stage(state, record)
        return review


async def _run_state(
    state: RewriteState, runtime: RewriteRuntime
) -> tuple[Literal["quality_validated", "needs_review", "call_failed", "cost_limited"], str]:
    feedback: SemanticReviewReceipt | None = None
    for cycle in range(runtime.config.workflow.max_correction_cycles + 1):
        phase: Literal["draft", "correction"] = "draft" if cycle == 0 else "correction"
        commit = await runtime.run_editor(state, phase=phase, feedback=feedback)
        if commit is None:
            return "call_failed", f"{phase} editor call failed"
        audit = deterministic_rewrite_audit(state.workspace, state.changed_leaves)
        state.audits.append(audit)
        if not audit.core_passed:
            return "needs_review", "atomic output failed a deterministic fidelity check"
        if runtime.cost_limited(state):
            return "cost_limited", "configured per-case estimated-cost limit reached"
        review = await runtime.run_reviewer(state, audit)
        if review is None:
            return "call_failed", "semantic reviewer call failed"
        state.reviews.append(review)
        if review.verdict == "pass":
            return "quality_validated", "atomic rewrite passed deterministic and semantic review"
        if runtime.cost_limited(state):
            return "cost_limited", "configured per-case estimated-cost limit reached"
        feedback = review
    return "needs_review", "semantic findings remain after the bounded correction cycles"


def _empty_usage() -> LinguisticUsageReceipt:
    return LinguisticUsageReceipt(
        requests=0,
        providerResponseIds=(),
        finishReasons=(),
        inputTokens=0,
        cacheReadTokens=0,
        cacheWriteTokens=0,
        outputTokens=0,
        reasoningTokens=0,
        visibleOutputTokens=0,
        estimatedCostUsd=Decimal(0),
    )


def _target_before_semantic_equipment(target: Mapping[str, Any]) -> dict[str, Any]:
    """Recover the capacity envelope used before relation-v5 equipment assignment."""

    source = copy.deepcopy(dict(target))
    patch = source.get("documentPatch")
    if not isinstance(patch, dict):
        raise ValueError("synthetic target has no documentPatch")
    for container in patch.get("containers") or []:
        if not isinstance(container, dict):
            raise ValueError("synthetic target container is not an object")
        container.pop("sizeCategory", None)
        container.pop("typeCategory", None)
    return source


def _topology_signature(label: Mapping[str, Any]) -> dict[str, JsonValue]:
    patch = cast(Mapping[str, Any], label.get("documentPatch") or {})
    parties = cast(Mapping[str, Any], patch.get("parties") or {})
    allocation_groups = patch.get("cargoAllocationGroups") or []
    return {
        "partyRoles": cast(
            list[JsonValue],
            sorted(key for key, value in parties.items() if value not in (None, [], {})),
        ),
        "notifyPartyCount": len(parties.get("notifyParties") or []),
        "containerCount": len(patch.get("containers") or []),
        "cargoGroupCount": len(patch.get("cargoGroups") or []),
        "cargoPackageCount": len(patch.get("cargoPackages") or []),
        "allocationGroupCount": len(allocation_groups),
        "allocationCount": sum(
            len(row.get("allocations") or [])
            for row in allocation_groups
            if isinstance(row, Mapping)
        ),
    }


def _party_slug(party: Mapping[str, Any]) -> str:
    name = party.get("name")
    if not isinstance(name, str):
        raise ValueError("cannot synthesize a domain for a party without a name")
    slug = _DOMAIN_TOKEN.sub(
        "", unicodedata.normalize("NFKD", name).encode("ascii", "ignore").decode().lower()
    )
    if len(slug) < 3:
        raise ValueError("party name is too short to derive a synthetic domain")
    return slug[:48]


def _domain_reason(hostname: str, country: str | None, countries: CountryRegistry) -> str | None:
    lowered = hostname.rstrip(".").casefold()
    tld = lowered.rsplit(".", 1)[-1]
    if lowered in _SPECIAL_USE_DOMAINS or tld in _SPECIAL_USE_TLDS:
        return "special_use_domain"
    if country is None or len(tld) != 2 or not tld.isalpha():
        return None
    country_code = countries.resolve(country)
    if country_code is None:
        raise ValueError(
            f"target party country is absent from the pinned ISO registry: {country!r}"
        )
    return "country_incoherent_cctld" if tld != country_code.casefold() else None


def _repair_party_domains(
    target: dict[str, Any], resources: TargetIntegrityResources
) -> tuple[TargetIntegrityChange, ...]:
    patch = cast(dict[str, Any], target["documentPatch"])
    parties = cast(dict[str, Any], patch.get("parties") or {})
    changes: list[TargetIntegrityChange] = []
    for role, value in parties.items():
        rows = value if role == "notifyParties" and isinstance(value, list) else [value]
        for occurrence, party in enumerate(rows):
            if not isinstance(party, dict):
                continue
            contacts = party.get("contactDetails")
            if not isinstance(contacts, dict):
                continue
            country = party.get("country") if isinstance(party.get("country"), str) else None
            domain = f"{_party_slug(party)}.com"
            prefix = f"documentPatch.parties.{role}"
            if role == "notifyParties":
                prefix += f"[{occurrence}]"
            for contact_field in ("emailAddresses", "websiteUrls"):
                values = contacts.get(contact_field)
                if not isinstance(values, list):
                    continue
                for index, before in enumerate(values):
                    if not isinstance(before, str):
                        continue
                    if contact_field == "emailAddresses":
                        if "@" not in before:
                            continue
                        local, old_domain = before.rsplit("@", 1)
                        reason = _domain_reason(old_domain, country, resources.countries)
                        after = f"{local}@{domain}"
                    else:
                        parsed = urlsplit(before if "://" in before else f"https://{before}")
                        reason = _domain_reason(parsed.hostname or "", country, resources.countries)
                        after = urlunsplit(
                            (
                                parsed.scheme or "https",
                                domain,
                                parsed.path,
                                parsed.query,
                                parsed.fragment,
                            )
                        )
                    if reason is None:
                        continue
                    values[index] = after
                    changes.append(
                        TargetIntegrityChange(
                            path=f"{prefix}.contactDetails.{contact_field}[{index}]",
                            reason=cast(Any, reason),
                            before=before,
                            after=after,
                        )
                    )
    return tuple(changes)


def _pluralize_package_surface(value: str) -> str:
    lowered = value.casefold()
    if lowered.endswith(("s", "x", "z", "ch", "sh")):
        return value + "es"
    if lowered.endswith("y") and len(value) > 1 and lowered[-2] not in "aeiou":
        return value[:-1] + "ies"
    return value + "s"


def _preserve_case(source: str, target: str) -> str:
    if source.isupper():
        return target.upper()
    if source.islower():
        return target.lower()
    if source.istitle():
        return target.title()
    return target


def _repair_package_information(
    target: dict[str, Any], resources: TargetIntegrityResources
) -> tuple[TargetIntegrityChange, ...]:
    patch = cast(dict[str, Any], target["documentPatch"])
    packages_by_group: dict[str, list[Mapping[str, Any]]] = {}
    for package in patch.get("cargoPackages") or []:
        if isinstance(package, Mapping) and isinstance(package.get("groupId"), str):
            packages_by_group.setdefault(cast(str, package["groupId"]), []).append(package)
    base_surfaces = {row.displayName.split(",", 1)[0] for row in resources.packages.payload.entries}
    known_surfaces = sorted(
        {surface for base in base_surfaces for surface in (base, _pluralize_package_surface(base))},
        key=len,
        reverse=True,
    )
    noun_pattern = re.compile(
        r"(?P<noun>" + "|".join(re.escape(value) for value in known_surfaces) + r")\b",
        re.IGNORECASE,
    )
    changes: list[TargetIntegrityChange] = []
    for group_index, group in enumerate(patch.get("cargoGroups") or []):
        if not isinstance(group, dict) or not isinstance(group.get("groupId"), str):
            continue
        packages = packages_by_group.get(cast(str, group["groupId"]), [])
        if len(packages) != 1:
            continue
        package = packages[0]
        quantity = package.get("quantity")
        category = package.get("typeCategory")
        if not isinstance(quantity, int) or not isinstance(category, str):
            continue
        desired_base = resources.packages.entry(category).displayName.split(",", 1)[0]
        desired = _pluralize_package_surface(desired_base) if quantity != 1 else desired_base
        quantity_pattern = re.compile(
            rf"(?<![0-9])(?:{re.escape(str(quantity))}|{re.escape(f'{quantity:,}')})\s+",
            re.IGNORECASE,
        )
        values = group.get("additionalInformation")
        if not isinstance(values, list):
            continue
        for value_index, before in enumerate(values):
            if not isinstance(before, str):
                continue
            quantity_match = quantity_pattern.search(before)
            if quantity_match is None:
                continue
            noun_match = noun_pattern.match(before, quantity_match.end())
            if noun_match is None:
                continue
            observed = noun_match.group("noun")
            if observed.casefold() in {desired_base.casefold(), desired.casefold()}:
                continue
            replacement = _preserve_case(observed, desired)
            after = before[: noun_match.start()] + replacement + before[noun_match.end() :]
            values[value_index] = after
            changes.append(
                TargetIntegrityChange(
                    path=(
                        f"documentPatch.cargoGroups[{group_index}].additionalInformation"
                        f"[{value_index}]"
                    ),
                    reason="package_surface_category_mismatch",
                    before=before,
                    after=after,
                )
            )
    return tuple(changes)


def _label_hs_codes(label: Mapping[str, Any]) -> tuple[str, ...]:
    patch = label.get("documentPatch")
    if not isinstance(patch, Mapping):
        return ()
    output: list[str] = []
    for group in cast(Sequence[Any], patch.get("cargoGroups") or ()):
        if not isinstance(group, Mapping):
            continue
        for value in cast(Sequence[Any], group.get("hsCodes") or ()):
            if isinstance(value, str):
                output.append("".join(re.findall(r"[0-9]", value)))
    return tuple(output)


def _explicit_hs_surfaces(raw_text: str, code: str) -> tuple[tuple[str, int, int], ...]:
    matches = tuple(
        (match.group(0), match.start(), match.end())
        for match in _NUMERIC_SURFACE.finditer(raw_text)
        if "".join(re.findall(r"[0-9]", match.group(0))) == code
    )
    if not matches:
        raise ValueError(f"explicit HS evidence parser returned an unlocatable code: {code}")
    return matches


def recover_explicit_hs_target_facts(
    *,
    raw_text: str,
    source_label: Mapping[str, Any],
    target: Mapping[str, Any],
    plan: DocumentLinguisticPlan,
) -> tuple[
    dict[str, Any],
    tuple[RecoveredExplicitHsTargetFact, ...],
    tuple[SurfaceRenderingRequirement, ...],
]:
    """Restore a source-visible HS slot omitted by the reviewed source label.

    The semantic completion plan is the sole source of replacement identities.  The model never
    invents a task-extractable code.  A single printed aggregate source code may expand to one
    code per target cargo identity; multiple printed codes must have exactly the same cardinality
    as the target identities.  Any partial or ambiguous situation fails before a provider call.
    """

    assigned = copy.deepcopy(dict(target))
    explicit_codes = explicit_hs_codes_from_text(raw_text)
    if not explicit_codes:
        return assigned, (), ()
    source_codes = _label_hs_codes(source_label)
    target_codes = _label_hs_codes(assigned)
    missing_from_source_label = tuple(code for code in explicit_codes if code not in source_codes)
    if not missing_from_source_label:
        return assigned, (), ()
    if source_codes or target_codes:
        raise ValueError(
            "explicit raw HS evidence is only partially represented by source/target labels; "
            f"explicit={explicit_codes}, source={source_codes}, target={target_codes}"
        )
    if plan.baseDocumentId != plan.cargoSeed.sourceDocumentId:
        raise ValueError("linguistic plan document identity differs from its cargo seed")
    if plan.scenarioId != plan.cargoSeed.scenarioId:
        raise ValueError("linguistic plan scenario identity differs from its cargo seed")

    patch = assigned.get("documentPatch")
    if not isinstance(patch, dict):
        raise ValueError("synthetic target has no documentPatch")
    target_groups = patch.get("cargoGroups")
    if not isinstance(target_groups, list):
        raise ValueError("synthetic target has no cargo groups")
    plan_groups = {group.groupId: group for group in plan.cargoSeed.cargoGroups}
    target_group_ids = tuple(
        group.get("groupId") if isinstance(group, Mapping) else None for group in target_groups
    )
    if any(not isinstance(group_id, str) for group_id in target_group_ids):
        raise ValueError("synthetic target cargo group lacks a string groupId")
    if set(cast(tuple[str, ...], target_group_ids)) != set(plan_groups):
        raise ValueError(
            "linguistic plan and synthetic target cargo-group identities differ: "
            f"target={target_group_ids}, plan={tuple(plan_groups)}"
        )

    planned: list[tuple[int, int, str, str]] = []
    for group_index, group in enumerate(target_groups):
        group_id = cast(str, group["groupId"])
        identities = plan_groups[group_id].goodsIdentities
        if not identities or any(identity.hsCode is None for identity in identities):
            raise ValueError(
                "explicit raw HS slot cannot be populated because its pinned semantic plan lacks "
                f"an HS identity for cargo group {group_id!r}"
            )
        for identity_index, identity in enumerate(identities):
            hs_code = cast(str, identity.hsCode)
            planned.append((group_index, identity_index, group_id, hs_code[:6]))
    if len(explicit_codes) not in {1, len(planned)}:
        raise ValueError(
            "explicit raw HS slot cardinality cannot be projected without guessing: "
            f"printed={len(explicit_codes)}, target_identities={len(planned)}"
        )

    source_widths = (
        (len(explicit_codes[0]),) * len(planned)
        if len(explicit_codes) == 1
        else tuple(len(code) for code in explicit_codes)
    )
    stream = DeterministicStream(
        seed=0,
        namespace="raw-text-explicit-hs-target-recovery-v1",
        identity=f"{plan.baseDocumentId}:{plan.scenarioId}",
    )
    rendered: list[str] = []
    recoveries: list[RecoveredExplicitHsTargetFact] = []
    codes_by_group: dict[int, list[str]] = {}
    target_paths: list[str] = []
    for (group_index, identity_index, group_id, hs6), width in zip(
        planned, source_widths, strict=True
    ):
        code = render_synthetic_hs_code(
            hs6=hs6,
            output_digits=width,
            stream=stream.derive(f"{group_id}:{identity_index}"),
        )
        path = f"documentPatch.cargoGroups[{group_index}].hsCodes[{identity_index}]"
        rendered.append(code)
        target_paths.append(path)
        codes_by_group.setdefault(group_index, []).append(code)
        recoveries.append(
            RecoveredExplicitHsTargetFact(
                targetPath=path,
                sourceEvidenceCodes=explicit_codes,
                semanticPlanPath=(
                    f"cargoSeed.cargoGroups[{group_id}].goodsIdentities[{identity_index}].hsCode"
                ),
                hs6=hs6,
                renderedCode=code,
                outputDigits=width,
                method="explicit_raw_hs_slot_from_pinned_semantic_plan_v1",
            )
        )
    for group_index, codes in codes_by_group.items():
        cast(dict[str, Any], target_groups[group_index])["hsCodes"] = codes

    requirements: list[SurfaceRenderingRequirement] = []
    if len(explicit_codes) == 1:
        source_code = explicit_codes[0]
        surfaces = _explicit_hs_surfaces(raw_text, source_code)
        for source_surface in dict.fromkeys(surface for surface, _, _ in surfaces):
            target_surface = ", ".join(
                _format_numeric_surface(source_surface, code) for code in rendered
            )
            first = next(row for row in surfaces if row[0] == source_surface)
            requirements.append(
                SurfaceRenderingRequirement(
                    kind="hs_code_block",
                    targetPath=";".join(target_paths),
                    sourceSurface=source_surface,
                    targetSurface=target_surface,
                    sourceOccurrences=sum(row[0] == source_surface for row in surfaces),
                    contextEvidence=raw_text[
                        max(0, first[1] - 120) : min(len(raw_text), first[2] + 80)
                    ],
                )
            )
    else:
        for source_code, target_code, target_path in zip(
            explicit_codes, rendered, target_paths, strict=True
        ):
            surfaces = _explicit_hs_surfaces(raw_text, source_code)
            for source_surface in dict.fromkeys(surface for surface, _, _ in surfaces):
                first = next(row for row in surfaces if row[0] == source_surface)
                requirements.append(
                    SurfaceRenderingRequirement(
                        kind="hs_code",
                        targetPath=target_path,
                        sourceSurface=source_surface,
                        targetSurface=_format_numeric_surface(source_surface, target_code),
                        sourceOccurrences=sum(row[0] == source_surface for row in surfaces),
                        contextEvidence=raw_text[
                            max(0, first[1] - 120) : min(len(raw_text), first[2] + 80)
                        ],
                    )
                )
    return assigned, tuple(recoveries), tuple(requirements)


def prepare_target_integrity(
    source_label: Mapping[str, Any],
    target: Mapping[str, Any],
    config: SynthesisRawTextRewriteCycleProbeConfig,
    resources: TargetIntegrityResources,
) -> tuple[dict[str, JsonValue], dict[str, JsonValue]]:
    """Validate and, if necessary, reproject a synthetic target before any API call."""

    assigned = copy.deepcopy(dict(target))
    source_topology = _topology_signature(source_label)
    target_topology = _topology_signature(assigned)
    if source_topology != target_topology:
        raise ValueError(
            "source/target structural topology differs; select a compatible template before "
            f"rewriting: source={source_topology}, target={target_topology}"
        )
    semantic_changes = [
        *_repair_party_domains(assigned, resources),
        *_repair_package_information(assigned, resources),
    ]
    source = _target_before_semantic_equipment(assigned)
    receipt = reproject_measures_for_semantic_equipment(
        source_target=source,
        assigned_target=assigned,
        limits=capacity_limits(config.target_integrity.transport_capacity),
    )
    canonical = BILL_OF_LADING_V5_TASK_ADAPTER.validate_target(
        document_id="synthetic-target-integrity",
        target=assigned,
    )
    payload = cast(dict[str, JsonValue], receipt.to_dict())
    payload["capacity_changed"] = cast(bool, payload["changed"])
    payload["changed"] = bool(payload["changed"] or semantic_changes)
    payload["semantic_changes"] = cast(
        JsonValue, [row.model_dump(mode="json") for row in semantic_changes]
    )
    payload["source_topology"] = cast(JsonValue, source_topology)
    payload["target_topology"] = cast(JsonValue, target_topology)
    payload["topology_matched"] = True
    return canonical, payload


async def _run_case(
    *,
    case_number: int,
    source_row: Mapping[str, Any],
    target_row: Mapping[str, Any],
    linguistic_plan: DocumentLinguisticPlan,
    feature: Mapping[str, JsonValue],
    runtime: RewriteRuntime,
) -> RewriteCycleBundle:
    document_id = cast(str, source_row["documentId"])
    source_text = cast(str, source_row["joinedRawText"])
    source_label = cast(
        dict[str, JsonValue],
        BILL_OF_LADING_TASK_ADAPTER.validate_target(
            document_id=document_id,
            target=cast(Mapping[str, Any], source_row[runtime.config.inputs.source_target_field]),
        ),
    )
    scenario_id = cast(str, target_row["scenarioId"])
    upstream_target_label = cast(
        dict[str, JsonValue],
        BILL_OF_LADING_V5_TASK_ADAPTER.validate_target(
            document_id=scenario_id,
            target=cast(Mapping[str, Any], target_row["target"]),
        ),
    )
    recovered_target_label, recovered_hs_facts, recovered_hs_surfaces = (
        recover_explicit_hs_target_facts(
            raw_text=source_text,
            source_label=source_label,
            target=upstream_target_label,
            plan=linguistic_plan,
        )
    )
    effective_target_label, target_integrity = prepare_target_integrity(
        source_label,
        recovered_target_label,
        runtime.config,
        runtime.target_integrity_resources,
    )
    target_integrity["recovered_task_facts"] = cast(
        JsonValue, [row.model_dump(mode="json") for row in recovered_hs_facts]
    )
    target_integrity["recovered_task_fact_count"] = len(recovered_hs_facts)
    rendering_requirements = (
        *surface_rendering_requirements(source_text, source_label, effective_target_label),
        *recovered_hs_surfaces,
    )
    jurisdiction_requirements = jurisdictional_surface_requirements(
        source_text, effective_target_label, runtime.target_integrity_resources
    )
    role_hints = source_semantic_role_hints(source_text)
    leaves = rewrite_changed_leaves(source_label, effective_target_label)
    inline_slot_requirements = inline_slot_topology_requirements(source_text)
    status_requirements = source_status_preservation_requirements(source_text, leaves)
    literal_requirements = target_literal_requirements(leaves)
    auxiliary_identity_requirements = raw_auxiliary_identity_requirements(
        source_text, source_label, effective_target_label
    )
    operational_requirements = operational_flavor_requirements(
        source_text,
        source_label,
        effective_target_label,
        source_document_id=document_id,
        scenario_id=scenario_id,
        profiles=runtime.target_integrity_resources.operational_profiles,
        limits=capacity_limits(runtime.config.target_integrity.transport_capacity),
    )
    cargo_rewrite_requirements = cargo_flavor_rewrite_requirements(
        source_text, source_label, effective_target_label
    )
    occurrence_requirements = target_value_occurrence_requirements(
        source_text,
        leaves,
        auxiliary_identity_requirements,
        rendering_requirements,
    )
    anchored_replacement_requirements = merge_anchored_scalar_replacement_requirements(
        anchored_scalar_replacement_requirements(source_text, leaves),
        anchored_measurement_replacement_requirements(
            source_text, source_label, effective_target_label
        ),
    )
    if not any(row.requiresTextEdit for row in leaves):
        raise ValueError(f"rewrite case has no printable target changes: {document_id}")
    workspace = RewriteWorkspace(
        scenario_id=scenario_id,
        source_label=source_label,
        upstream_target_label=upstream_target_label,
        current_target_label=copy.deepcopy(effective_target_label),
        surface_requirements=rendering_requirements,
        target_literal_requirements=literal_requirements,
        target_value_occurrence_requirements=occurrence_requirements,
        anchored_scalar_replacement_requirements=anchored_replacement_requirements,
        source_role_hints=role_hints,
        inline_slot_requirements=inline_slot_requirements,
        source_status_requirements=status_requirements,
        jurisdictional_requirements=jurisdiction_requirements,
        raw_auxiliary_identity_requirements=auxiliary_identity_requirements,
        operational_flavor_requirements=operational_requirements,
        cargo_flavor_rewrite_requirements=cargo_rewrite_requirements,
        original_text=source_text,
        current_text=source_text,
    )
    apply_deterministic_prefills(workspace)
    state = RewriteState(
        case_number=case_number,
        document_id=document_id,
        scenario_id=scenario_id,
        feature=dict(feature),
        source_label=source_label,
        target_integrity=target_integrity,
        workspace=workspace,
        started_at=datetime.now(UTC),
    )
    status, reason = await _run_state(state, runtime)
    usage = _combine_usage([row.usage for row in state.stages]) if state.stages else _empty_usage()
    commits = [row.editorCommit for row in state.stages if row.editorCommit is not None]
    replacements = tuple(
        replacement for commit in commits for replacement in commit.appliedReplacements
    )
    effective_leaves = state.changed_leaves
    effective_change_contract = label_change_contract(state.workspace)
    compact_change_contract = compact_label_change_contract(effective_change_contract)
    flavor_requirements = compound_party_flavor_requirements(source_label, state.target_label)
    result = RewriteCycleResult(
        schemaVersion=15,
        caseNumber=case_number,
        documentId=document_id,
        scenarioId=scenario_id,
        status=status,
        reason=reason,
        sourceTextSha256=sha256_bytes(source_text.encode("utf-8")),
        outputTextSha256=state.workspace.current_sha256,
        sourceLabelSha256=sha256_bytes(canonical_json_bytes(source_label)),
        upstreamTargetLabelSha256=sha256_bytes(canonical_json_bytes(upstream_target_label)),
        targetLabelSha256=state.workspace.current_target_sha256,
        targetIntegritySha256=sha256_bytes(canonical_json_bytes(state.target_integrity)),
        targetCapacityReprojected=bool(state.target_integrity["capacity_changed"]),
        changedLeavesSha256=sha256_bytes(
            canonical_json_bytes([row.model_dump(mode="json") for row in effective_leaves])
        ),
        labelChangeContractSha256=sha256_bytes(
            canonical_json_bytes([row.model_dump(mode="json") for row in effective_change_contract])
        ),
        editorOutputSchemaSha256=sha256_bytes(
            canonical_json_bytes(AtomicRewriteArguments.model_json_schema(mode="validation"))
        ),
        reviewerOutputSchemaSha256=sha256_bytes(
            canonical_json_bytes(SemanticReviewReceipt.model_json_schema(mode="validation"))
        ),
        correctionCycles=sum(row.stage == "correction_editor" for row in state.stages),
        editorPasses=sum(row.stage != "semantic_reviewer" for row in state.stages),
        reviewPasses=sum(row.stage == "semantic_reviewer" for row in state.stages),
        compoundPartyFlavorRealizations=sum(
            len(commit.compoundPartyFlavorRealizations) for commit in commits
        ),
        rawAuxiliaryIdentityRealizations=sum(
            len(commit.rawAuxiliaryIdentityRealizations) for commit in commits
        ),
        deterministicPrefills=len(state.workspace.deterministic_prefills),
        replacements=len(replacements),
        sourceLinesReplaced=sum(row.sourceLinesReplaced for row in replacements),
        outputLinesInserted=sum(row.outputLinesInserted for row in replacements),
        durationMs=(datetime.now(UTC) - state.started_at).total_seconds() * 1000.0,
        totalUsage=usage,
    )
    bundle = RewriteCycleBundle(
        schemaVersion=15,
        result=result,
        feature=state.feature,
        sourceLabel=source_label,
        upstreamTargetLabel=upstream_target_label,
        targetLabel=state.target_label,
        targetIntegrity=state.target_integrity,
        changedLeaves=effective_leaves,
        labelChangeContract=effective_change_contract,
        compactLabelChangeContract=compact_change_contract,
        surfaceRenderingRequirements=state.workspace.surface_requirements,
        targetLiteralRequirements=state.workspace.target_literal_requirements,
        targetValueOccurrenceRequirements=(state.workspace.target_value_occurrence_requirements),
        anchoredScalarReplacementRequirements=(
            state.workspace.anchored_scalar_replacement_requirements
        ),
        sourceSemanticRoleHints=state.workspace.source_role_hints,
        inlineSlotTopologyRequirements=state.workspace.inline_slot_requirements,
        sourceStatusPreservationRequirements=state.workspace.source_status_requirements,
        compoundPartyFlavorRequirements=flavor_requirements,
        rawAuxiliaryIdentityRequirements=(state.workspace.raw_auxiliary_identity_requirements),
        operationalFlavorRequirements=(state.workspace.operational_flavor_requirements),
        cargoFlavorRewriteRequirements=(state.workspace.cargo_flavor_rewrite_requirements),
        jurisdictionalSurfaceRequirements=state.workspace.jurisdictional_requirements,
        sourceText=source_text,
        outputText=state.workspace.current_text,
        deterministicPrefills=tuple(state.workspace.deterministic_prefills),
        appliedReplacements=replacements,
        deterministicAudits=tuple(state.audits),
        semanticReviews=tuple(state.reviews),
        stageRecords=tuple(state.stages),
        diff=unified_text_diff(source_text, state.workspace.current_text),
    )
    runtime.staged.publish_json(
        f"cases/{case_number:02d}-{document_id}/bundle.json", bundle.model_dump(mode="json")
    )
    return bundle


def _load_bundle(path: Path) -> RewriteCycleBundle | None:
    if not path.exists():
        return None
    if path.is_symlink() or not path.is_file():
        raise ValueError(f"rewrite bundle resume path is not a regular file: {path}")
    return RewriteCycleBundle.model_validate_json(read_regular_file_bytes(path), strict=True)


async def _run_cases(
    *,
    selected: Sequence[
        tuple[
            Mapping[str, Any],
            Mapping[str, Any],
            DocumentLinguisticPlan,
            Mapping[str, JsonValue],
        ]
    ],
    source_rows: Sequence[Mapping[str, Any]],
    editor_prompt: str,
    reviewer_prompt: str,
    staged: StagedArtifactRun,
    config: SynthesisRawTextRewriteCycleProbeConfig,
    project_root: Path,
) -> tuple[RewriteCycleBundle, ...]:
    iso_path = _resolve_pinned_file(
        project_root,
        config.target_integrity.iso3166_snapshot.path,
        config.target_integrity.iso3166_snapshot.sha256,
        label="target-integrity ISO-3166 snapshot",
    )
    package_path = _resolve_pinned_file(
        project_root,
        config.target_integrity.package_registry.path,
        config.target_integrity.package_registry.sha256,
        label="target-integrity package registry",
    )
    route_locations_path = _resolve_pinned_file(
        project_root,
        config.target_integrity.route_locations.path,
        config.target_integrity.route_locations.sha256,
        label="target-integrity UN/LOCODE locations",
    )
    customs_program_path = _resolve_pinned_file(
        project_root,
        config.target_integrity.customs_program_registry.path,
        config.target_integrity.customs_program_registry.sha256,
        label="target-integrity customs-program registry",
    )
    route_locations = load_pinned_unlocode_locations(
        route_locations_path,
        expected_sha256=config.target_integrity.route_locations.sha256,
        expected_records=config.target_integrity.route_location_records,
    )
    customs_programs = _load_customs_program_registry(
        customs_program_path,
        expected_entries=config.target_integrity.customs_program_registry_entries,
    )
    transport_limits = capacity_limits(config.target_integrity.transport_capacity)
    operational_profiles = build_empirical_operational_profiles(
        source_rows,
        target_field=config.inputs.source_target_field,
        limits=transport_limits,
    )
    if not operational_profiles:
        raise ValueError("pinned source corpus produced no empirical operational profiles")
    staged.publish_json(
        "generation/operational-profile-support.json",
        {
            "schemaVersion": 1,
            "method": "empirical_capacity_utilization_resample_v1",
            "profiles": len(operational_profiles),
            "documents": len({row.document_id for row in operational_profiles}),
            "grossWeightProfiles": sum(
                row.gross_utilization is not None for row in operational_profiles
            ),
            "volumeProfiles": sum(
                row.volume_utilization is not None or row.volume_m3 is not None
                for row in operational_profiles
            ),
            "equipmentFamilies": {
                family: sum(row.equipment_family == family for row in operational_profiles)
                for family in sorted({row.equipment_family for row in operational_profiles})
            },
        },
    )
    target_integrity_resources = TargetIntegrityResources(
        countries=load_iso_country_registry(
            iso_path=iso_path,
            iso_sha256=config.target_integrity.iso3166_snapshot.sha256,
        ),
        packages=load_package_registry(
            package_path,
            expected_sha256=config.target_integrity.package_registry.sha256,
            expected_entries=config.target_integrity.package_registry_entries,
        ),
        route_countries_by_name=_route_country_index(route_locations),
        customs_programs=customs_programs.entries,
        operational_profiles=operational_profiles,
    )
    key = load_openai_key(project_root, config.environment_file)
    client = AsyncOpenAI(
        api_key=key,
        max_retries=max(
            config.providers.editor.transport_max_retries,
            config.providers.reviewer.transport_max_retries,
        ),
        timeout=max(
            config.providers.editor.request_timeout_seconds,
            config.providers.reviewer.request_timeout_seconds,
        ),
    )
    provider = OpenAIProvider(openai_client=client)
    limiter = ConcurrencyLimiter(
        config.workflow.max_concurrent_requests,
        name="synthetic-raw-text-atomic-rewrite-provider-requests",
    )
    runtime = RewriteRuntime(
        config=config,
        editor_prompt=editor_prompt,
        reviewer_prompt=reviewer_prompt,
        editor_model=OpenAIResponsesModel(config.providers.editor.model, provider=provider),
        reviewer_model=OpenAIResponsesModel(config.providers.reviewer.model, provider=provider),
        target_integrity_resources=target_integrity_resources,
        limiter=limiter,
        staged=staged,
    )
    case_limiter = asyncio.Semaphore(config.workflow.max_concurrent_cases)
    progress_lock = asyncio.Lock()
    results: list[RewriteCycleBundle | None] = [None] * len(selected)
    processed = 0
    started = time.perf_counter()

    async def run_one(index: int) -> None:
        nonlocal processed
        source, target, linguistic_plan, feature = selected[index]
        document_id = cast(str, source["documentId"])
        existing = _load_bundle(
            staged.stage_root / f"cases/{index + 1:02d}-{document_id}/bundle.json"
        )
        if existing is not None:
            results[index] = existing
        else:
            async with case_limiter:
                results[index] = await _run_case(
                    case_number=index + 1,
                    source_row=source,
                    target_row=target,
                    linguistic_plan=linguistic_plan,
                    feature=feature,
                    runtime=runtime,
                )
        async with progress_lock:
            processed += 1
            elapsed = time.perf_counter() - started
            rate = processed / elapsed if elapsed else 0.0
            print(
                json.dumps(
                    {
                        "command": "run-raw-text-rewrite-cycle-probe",
                        "phase": "atomic_edit_review_correct",
                        "processed_cases": processed,
                        "remaining_cases": len(selected) - processed,
                        "quality_validated_cases": sum(
                            row is not None and row.result.status == "quality_validated"
                            for row in results
                        ),
                        "elapsed_seconds": round(elapsed, 3),
                        "throughput_cases_per_hour": round(rate * 3600.0, 3),
                        "eta_seconds": round((len(selected) - processed) / rate, 3)
                        if rate
                        else None,
                        "status": "progress",
                    },
                    allow_nan=False,
                    sort_keys=True,
                ),
                flush=True,
            )

    try:
        await asyncio.gather(*(run_one(index) for index in range(len(selected))))
    finally:
        await client.close()
    if any(row is None for row in results):
        raise RuntimeError("atomic rewrite worker pool returned an incomplete result set")
    return tuple(cast(RewriteCycleBundle, row) for row in results)


def _publish_case_files(staged: StagedArtifactRun, bundle: RewriteCycleBundle) -> None:
    prefix = f"cases/{bundle.result.caseNumber:02d}-{bundle.result.documentId}"
    payloads = {
        "source-label.json": json_artifact_bytes(bundle.sourceLabel),
        "upstream-target-label.json": json_artifact_bytes(bundle.upstreamTargetLabel),
        "target-label.json": json_artifact_bytes(bundle.targetLabel),
        "target-integrity.json": json_artifact_bytes(bundle.targetIntegrity),
        "changed-leaves.json": json_artifact_bytes(
            [row.model_dump(mode="json") for row in bundle.changedLeaves]
        ),
        "label-change-contract.json": json_artifact_bytes(
            [row.model_dump(mode="json") for row in bundle.labelChangeContract]
        ),
        "model-change-contract.json": json_artifact_bytes(
            [row.model_dump(mode="json") for row in bundle.compactLabelChangeContract]
        ),
        "surface-rendering-requirements.json": json_artifact_bytes(
            [row.model_dump(mode="json") for row in bundle.surfaceRenderingRequirements]
        ),
        "target-literal-requirements.json": json_artifact_bytes(
            [row.model_dump(mode="json") for row in bundle.targetLiteralRequirements]
        ),
        "target-value-occurrence-requirements.json": json_artifact_bytes(
            [row.model_dump(mode="json") for row in bundle.targetValueOccurrenceRequirements]
        ),
        "anchored-scalar-replacement-requirements.json": json_artifact_bytes(
            [row.model_dump(mode="json") for row in bundle.anchoredScalarReplacementRequirements]
        ),
        "deterministic-prefills.json": json_artifact_bytes(
            [row.model_dump(mode="json") for row in bundle.deterministicPrefills]
        ),
        "source-semantic-role-hints.json": json_artifact_bytes(
            [row.model_dump(mode="json") for row in bundle.sourceSemanticRoleHints]
        ),
        "inline-slot-topology-requirements.json": json_artifact_bytes(
            [row.model_dump(mode="json") for row in bundle.inlineSlotTopologyRequirements]
        ),
        "source-status-preservation-requirements.json": json_artifact_bytes(
            [row.model_dump(mode="json") for row in bundle.sourceStatusPreservationRequirements]
        ),
        "compound-party-flavor-requirements.json": json_artifact_bytes(
            [row.model_dump(mode="json") for row in bundle.compoundPartyFlavorRequirements]
        ),
        "raw-auxiliary-identity-requirements.json": json_artifact_bytes(
            [row.model_dump(mode="json") for row in bundle.rawAuxiliaryIdentityRequirements]
        ),
        "operational-flavor-requirements.json": json_artifact_bytes(
            [row.model_dump(mode="json") for row in bundle.operationalFlavorRequirements]
        ),
        "cargo-flavor-rewrite-requirements.json": json_artifact_bytes(
            [row.model_dump(mode="json") for row in bundle.cargoFlavorRewriteRequirements]
        ),
        "jurisdictional-surface-requirements.json": json_artifact_bytes(
            [row.model_dump(mode="json") for row in bundle.jurisdictionalSurfaceRequirements]
        ),
        "raw-before.txt": bundle.sourceText.encode("utf-8"),
        "raw-after.txt": bundle.outputText.encode("utf-8"),
        "diff.patch": bundle.diff.encode("utf-8"),
        "replacements.json": json_artifact_bytes(
            [row.model_dump(mode="json") for row in bundle.appliedReplacements]
        ),
        "deterministic-audits.json": json_artifact_bytes(
            [row.model_dump(mode="json") for row in bundle.deterministicAudits]
        ),
        "semantic-reviews.json": json_artifact_bytes(
            [row.model_dump(mode="json") for row in bundle.semanticReviews]
        ),
        "stages.json": json_artifact_bytes(
            [row.model_dump(mode="json") for row in bundle.stageRecords]
        ),
        "result.json": json_artifact_bytes(bundle.result.model_dump(mode="json")),
    }
    for name, payload in payloads.items():
        staged.publish_bytes(f"{prefix}/{name}", payload)


def _fence(value: str, language: str) -> str:
    fence = "~~~~"
    while fence in value:
        fence += "~"
    return f"{fence}{language}\n{value}\n{fence}"


def _report(summary: Mapping[str, Any], bundles: Sequence[RewriteCycleBundle]) -> str:
    lines = [
        "# Atomic synthetic B/L raw-text rewrite probe",
        "",
        "## Outcome",
        "",
        f"- Quality validated: **{summary['qualityValidatedCases']}/{summary['cases']}**",
        f"- Needs review / call failures: **{summary['needsReviewCases']} / "
        f"{summary['callFailedCases']}**",
        f"- Requests: **{summary['requests']}**",
        f"- Estimated list-price cost: **${summary['estimatedCostUsd']}**",
        f"- Estimated list-price cost per 1,000 at this pass rate: "
        f"**${summary['estimatedCostUsdPerThousand']}**",
        f"- Input / cache-read / cache-write tokens: **{summary['inputTokens']} / "
        f"{summary['cacheReadTokens']} / {summary['cacheWriteTokens']}**",
        f"- Visible / reasoning output tokens: **{summary['visibleOutputTokens']} / "
        f"{summary['reasoningTokens']}**",
        f"- Wall time / throughput: **{summary['wallSeconds']:.3f}s / "
        f"{summary['throughputCasesPerHour']:.3f} cases/hour**",
        "",
        "Cost is a rate-card estimate from provider-reported token receipts, not a confirmed "
        "billing-account debit. The OpenAI billing dashboard remains authoritative and may post "
        "usage asynchronously.",
        "",
        "The editor uses one strict terminal output function per pass. Its compact line-range ",
        "patch is applied atomically in-process and is never replayed to the model. A separate ",
        "provider-constrained reviewer can request a bounded correction. Training records remain ",
        "unpublished.",
        "",
        "## Usage by stage",
        "",
        "| Stage | Requests | Input | Cache read | Cache write | Reasoning | Visible | "
        "Estimated list cost |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for stage, usage in summary["usageByStage"].items():
        lines.append(
            f"| `{stage}` | {usage['requests']} | {usage['inputTokens']} | "
            f"{usage['cacheReadTokens']} | {usage['cacheWriteTokens']} | "
            f"{usage['reasoningTokens']} | {usage['visibleOutputTokens']} | "
            f"${usage['estimatedCostUsd']} |"
        )
    lines.extend(
        [
            "",
            "## Case index",
            "",
            "| Case | Document | Status | Editor/review/correction | Requests | Input | "
            "Reasoning | "
            "Visible | Cost |",
            "|---:|---|---|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for bundle in bundles:
        row = bundle.result
        lines.append(
            f"| {row.caseNumber} | `{row.documentId}` | {row.status} | "
            f"{row.editorPasses}/{row.reviewPasses}/{row.correctionCycles} | "
            f"{row.totalUsage.requests} | {row.totalUsage.inputTokens} | "
            f"{row.totalUsage.reasoningTokens} | {row.totalUsage.visibleOutputTokens} | "
            f"${row.totalUsage.estimatedCostUsd} |"
        )
    for bundle in bundles:
        row = bundle.result
        lines.extend(
            [
                "",
                f"## Case {row.caseNumber}: `{row.documentId}`",
                "",
                f"- Status: **{row.status}** — {row.reason}",
                f"- Line ranges / source lines / output lines: **{row.replacements} / "
                f"{row.sourceLinesReplaced} / {row.outputLinesInserted}**",
                f"- Deterministic scalar prefills: **{row.deterministicPrefills}**",
                f"- Requests / estimated cost: **{row.totalUsage.requests} / "
                f"${row.totalUsage.estimatedCostUsd}**",
                f"- Retained provider call trace: "
                f"`cases/{row.caseNumber:02d}-{row.documentId}/stages.json`",
                "",
                "### Stage usage",
                "",
                "| Sequence | Stage | Requests | Input | Reasoning | Visible | Cost |",
                "|---:|---|---:|---:|---:|---:|---:|",
                *[
                    f"| {stage.sequence} | `{stage.stage}` | {stage.usage.requests} | "
                    f"{stage.usage.inputTokens} | {stage.usage.reasoningTokens} | "
                    f"{stage.usage.visibleOutputTokens} | ${stage.usage.estimatedCostUsd} |"
                    for stage in bundle.stageRecords
                ],
                "",
                "### Reviews",
                "",
                _fence(
                    json.dumps(
                        [review.model_dump(mode="json") for review in bundle.semanticReviews],
                        ensure_ascii=False,
                        indent=2,
                    ),
                    "json",
                ),
                "",
                "### Authoritative label-change contract",
                "",
                _fence(
                    json.dumps(
                        [
                            directive.model_dump(mode="json")
                            for directive in bundle.labelChangeContract
                        ],
                        ensure_ascii=False,
                        indent=2,
                    ),
                    "json",
                ),
                "",
                "### Compact model-facing change contract",
                "",
                _fence(
                    json.dumps(
                        [
                            directive.model_dump(mode="json")
                            for directive in bundle.compactLabelChangeContract
                        ],
                        ensure_ascii=False,
                        indent=2,
                    ),
                    "json",
                ),
                "",
                "### Exact surface, semantic-role, compound-flavor, and jurisdiction requirements",
                "",
                _fence(
                    json.dumps(
                        {
                            "surfaceRenderingRequirements": [
                                requirement.model_dump(mode="json")
                                for requirement in bundle.surfaceRenderingRequirements
                            ],
                            "targetLiteralRequirements": [
                                requirement.model_dump(mode="json")
                                for requirement in bundle.targetLiteralRequirements
                            ],
                            "targetValueOccurrenceRequirements": [
                                requirement.model_dump(mode="json")
                                for requirement in bundle.targetValueOccurrenceRequirements
                            ],
                            "anchoredScalarReplacementRequirements": [
                                requirement.model_dump(mode="json")
                                for requirement in bundle.anchoredScalarReplacementRequirements
                            ],
                            "sourceSemanticRoleHints": [
                                hint.model_dump(mode="json")
                                for hint in bundle.sourceSemanticRoleHints
                            ],
                            "inlineSlotTopologyRequirements": [
                                requirement.model_dump(mode="json")
                                for requirement in bundle.inlineSlotTopologyRequirements
                            ],
                            "sourceStatusPreservationRequirements": [
                                requirement.model_dump(mode="json")
                                for requirement in bundle.sourceStatusPreservationRequirements
                            ],
                            "compoundPartyFlavorRequirements": [
                                requirement.model_dump(mode="json")
                                for requirement in bundle.compoundPartyFlavorRequirements
                            ],
                            "rawAuxiliaryIdentityRequirements": [
                                requirement.model_dump(mode="json")
                                for requirement in bundle.rawAuxiliaryIdentityRequirements
                            ],
                            "operationalFlavorRequirements": [
                                requirement.model_dump(mode="json")
                                for requirement in bundle.operationalFlavorRequirements
                            ],
                            "cargoFlavorRewriteRequirements": [
                                requirement.model_dump(mode="json")
                                for requirement in bundle.cargoFlavorRewriteRequirements
                            ],
                            "jurisdictionalSurfaceRequirements": [
                                requirement.model_dump(mode="json")
                                for requirement in bundle.jurisdictionalSurfaceRequirements
                            ],
                        },
                        ensure_ascii=False,
                        indent=2,
                    ),
                    "json",
                ),
                "",
                "### Target-integrity receipt",
                "",
                _fence(
                    json.dumps(bundle.targetIntegrity, ensure_ascii=False, indent=2),
                    "json",
                ),
                "",
                "### Source label",
                "",
                _fence(json.dumps(bundle.sourceLabel, ensure_ascii=False, indent=2), "json"),
                "",
                "### Synthetic target label",
                "",
                "The upstream target is shown first. Target-integrity checks may reproject "
                "physically incompatible measures, but auxiliary compound identities remain "
                "raw-text flavor and never alter the effective training target.",
                "",
                "#### Upstream target",
                "",
                _fence(
                    json.dumps(bundle.upstreamTargetLabel, ensure_ascii=False, indent=2),
                    "json",
                ),
                "",
                "#### Effective training target",
                "",
                _fence(json.dumps(bundle.targetLabel, ensure_ascii=False, indent=2), "json"),
                "",
                "### Source raw OCR",
                "",
                _fence(bundle.sourceText, "text"),
                "",
                "### Final raw OCR",
                "",
                _fence(bundle.outputText, "text"),
                "",
                "### Exact diff",
                "",
                _fence(bundle.diff or "(no diff)", "diff"),
            ]
        )
    return "\n".join(lines) + "\n"


def _plot_bytes(bundles: Sequence[RewriteCycleBundle]) -> dict[str, bytes]:
    import io

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import pandas as pd  # type: ignore[import-untyped]
    import seaborn as sns  # type: ignore[import-untyped]

    rows = [
        {
            "case": row.result.caseNumber,
            "status": row.result.status,
            "cost_usd": float(row.result.totalUsage.estimatedCostUsd),
            "requests": row.result.totalUsage.requests,
            "visible_output": row.result.totalUsage.visibleOutputTokens,
            "reasoning": row.result.totalUsage.reasoningTokens,
            "input": row.result.totalUsage.inputTokens,
            "seconds": row.result.durationMs / 1000.0,
        }
        for row in bundles
    ]
    frame = pd.DataFrame(rows)
    sns.set_theme(style="whitegrid", context="notebook")
    plots: dict[str, bytes] = {}

    def render(name: str, figure: Any) -> None:
        stream = io.BytesIO()
        figure.savefig(stream, format="png", dpi=170, bbox_inches="tight")
        plots[name] = stream.getvalue()
        plt.close(figure)

    figure, axes = plt.subplots(1, 2, figsize=(14, 5.5))
    sns.barplot(data=frame, x="case", y="cost_usd", hue="status", dodge=False, ax=axes[0])
    sns.barplot(data=frame, x="case", y="seconds", hue="status", dodge=False, ax=axes[1])
    axes[0].set(title="Estimated cost by case", xlabel="case", ylabel="USD")
    axes[1].set(title="Wall time by case", xlabel="case", ylabel="seconds")
    render("01_cost_and_duration.png", figure)

    token_frame = frame.melt(
        id_vars=("case",),
        value_vars=("input", "visible_output", "reasoning"),
        var_name="token_kind",
        value_name="tokens",
    )
    figure, axis = plt.subplots(figsize=(12, 5.5))
    sns.barplot(data=token_frame, x="case", y="tokens", hue="token_kind", ax=axis)
    axis.set(title="Provider token usage", xlabel="case", ylabel="tokens")
    render("02_tokens.png", figure)

    stage_rows = [
        {
            "case": bundle.result.caseNumber,
            "stage": record.stage,
            "reasoning": record.usage.reasoningTokens,
            "visible_output": record.usage.visibleOutputTokens,
            "cost_usd": float(record.usage.estimatedCostUsd),
        }
        for bundle in bundles
        for record in bundle.stageRecords
    ]
    stage_frame = pd.DataFrame(stage_rows)
    figure, axes = plt.subplots(1, 2, figsize=(14, 5.5))
    stage_tokens = stage_frame.melt(
        id_vars=("stage",),
        value_vars=("reasoning", "visible_output"),
        var_name="token_kind",
        value_name="tokens",
    )
    sns.barplot(
        data=stage_tokens, x="stage", y="tokens", hue="token_kind", estimator="sum", ax=axes[0]
    )
    sns.barplot(data=stage_frame, x="stage", y="cost_usd", estimator="sum", ax=axes[1])
    axes[0].set(title="Output tokens by stage", xlabel="stage", ylabel="tokens")
    axes[1].set(title="Estimated list cost by stage", xlabel="stage", ylabel="USD")
    axes[0].tick_params(axis="x", rotation=15)
    axes[1].tick_params(axis="x", rotation=15)
    render("03_stage_usage.png", figure)
    return plots


def _artifact_inventory(root: Path) -> list[str]:
    rows: list[str] = []
    for directory, directory_names, filenames in os.walk(root, followlinks=False):
        directory_path = Path(directory)
        if any((directory_path / name).is_symlink() for name in directory_names):
            raise ValueError("rewrite artifact tree contains a symlink directory")
        for name in filenames:
            path = directory_path / name
            if path.is_symlink():
                raise ValueError("rewrite artifact tree contains a symlink file")
            relative = path.relative_to(root).as_posix()
            if relative not in {"_TRANSACTION.json", "_COMMIT.json"}:
                rows.append(relative)
    return sorted(rows)


def run_raw_text_rewrite_cycle_probe(
    *,
    project_root: Path,
    config_path: Path,
    config: SynthesisRawTextRewriteCycleProbeConfig,
) -> dict[str, JsonValue]:
    linguistic_root = _validate_linguistic_run(project_root, config)
    results_path = _resolve_pinned_file(
        project_root,
        config.inputs.linguistic_results.path,
        config.inputs.linguistic_results.sha256,
        label="linguistic results",
    )
    targets_path = _resolve_pinned_file(
        project_root,
        config.inputs.synthetic_targets.path,
        config.inputs.synthetic_targets.sha256,
        label="linguistic targets",
    )
    summary_path = _resolve_pinned_file(
        project_root,
        config.inputs.linguistic_summary.path,
        config.inputs.linguistic_summary.sha256,
        label="linguistic summary",
    )
    plan_path = linguistic_root / "planning/document-plans.jsonl"
    if plan_path.is_symlink() or not plan_path.is_file():
        raise ValueError(f"linguistic completion run has no regular document plan: {plan_path}")
    plan_path = plan_path.resolve(strict=True)
    for path in (results_path, targets_path, summary_path, plan_path):
        if not path.is_relative_to(linguistic_root):
            raise ValueError("linguistic artifact is outside the pinned committed run")
    source_path = _resolve_pinned_file(
        project_root,
        config.inputs.source_corpus.path,
        config.inputs.source_corpus.sha256,
        label="source corpus",
    )
    feature_path = _resolve_pinned_file(
        project_root,
        config.inputs.document_features.path,
        config.inputs.document_features.sha256,
        label="document features",
    )
    editor_prompt_path = _resolve_pinned_file(
        project_root,
        config.prompts.editor.path,
        config.prompts.editor.sha256,
        label="rewrite editor prompt",
    )
    reviewer_prompt_path = _resolve_pinned_file(
        project_root,
        config.prompts.reviewer.path,
        config.prompts.reviewer.sha256,
        label="rewrite reviewer prompt",
    )
    editor_prompt_bytes = read_regular_file_bytes(editor_prompt_path)
    reviewer_prompt_bytes = read_regular_file_bytes(reviewer_prompt_path)
    editor_prompt = editor_prompt_bytes.decode("utf-8")
    reviewer_prompt = reviewer_prompt_bytes.decode("utf-8")

    results = _load_jsonl(
        results_path,
        records=config.inputs.linguistic_results.records,
        key="baseDocumentId",
        label="linguistic results",
    )
    targets = _load_jsonl(
        targets_path,
        records=config.inputs.synthetic_targets.records,
        key="baseDocumentId",
        label="linguistic targets",
    )
    raw_plans = _load_jsonl(
        plan_path,
        records=config.inputs.linguistic_results.records,
        key="baseDocumentId",
        label="linguistic document plans",
    )
    plans = tuple(
        DocumentLinguisticPlan.model_validate_json(canonical_json_bytes(row), strict=True)
        for row in raw_plans
    )
    sources = _load_jsonl(
        source_path,
        records=config.inputs.source_corpus.records,
        key="documentId",
        label="source corpus",
    )
    features = _load_jsonl(
        feature_path,
        records=config.inputs.document_features.records,
        key="document_id",
        label="document features",
    )
    result_by_id = {row["baseDocumentId"]: row for row in results}
    target_by_id = {row["baseDocumentId"]: row for row in targets}
    plan_by_id = {row.baseDocumentId: row for row in plans}
    source_by_id = {row["documentId"]: row for row in sources}
    feature_by_id = {row["document_id"]: row for row in features}
    if tuple(result_by_id) != tuple(target_by_id) or tuple(result_by_id) != tuple(plan_by_id):
        raise ValueError("linguistic result, target, and document-plan order differs")
    for document_id, result in result_by_id.items():
        target = target_by_id[document_id]
        plan = plan_by_id[document_id]
        if result.get("status") != "success" or result.get("target") != target.get("target"):
            raise ValueError(f"linguistic result/target mismatch for {document_id}")
        if target.get("scenarioId") != plan.scenarioId:
            raise ValueError(f"linguistic plan/target scenario mismatch for {document_id}")
        target_sha = sha256_bytes(canonical_json_bytes(target["target"]))
        if target.get("targetSha256") != target_sha or result.get("targetSha256") != target_sha:
            raise ValueError(f"linguistic target SHA-256 mismatch for {document_id}")

    selected: list[
        tuple[
            Mapping[str, Any],
            Mapping[str, Any],
            DocumentLinguisticPlan,
            Mapping[str, JsonValue],
        ]
    ] = []
    for case in config.cases:
        try:
            source = source_by_id[case.document_id]
            target = target_by_id[case.document_id]
            plan = plan_by_id[case.document_id]
            feature = feature_by_id[case.document_id]
        except KeyError as error:
            raise ValueError(
                f"rewrite case is absent from pinned inputs: {case.document_id}"
            ) from error
        raw_text = source.get("joinedRawText")
        if not isinstance(raw_text, str) or source.get("joinedRawTextSha256") != sha256_bytes(
            raw_text.encode("utf-8")
        ):
            raise ValueError(f"source raw text fails its SHA-256: {case.document_id}")
        selected.append((source, target, plan, cast(Mapping[str, JsonValue], feature)))

    editor_schema = AtomicRewriteArguments.model_json_schema(mode="validation")
    reviewer_schema = SemanticReviewReceipt.model_json_schema(mode="validation")
    transaction = {
        "schemaVersion": 15,
        "runId": config.run.run_id,
        "configSha256": sha256_file(config_path),
        "linguisticRunTransactionSha256": (
            config.inputs.linguistic_completion_run.transaction_sha256
        ),
        "linguisticResultsSha256": config.inputs.linguistic_results.sha256,
        "linguisticDocumentPlansSha256": sha256_file(plan_path),
        "syntheticTargetsSha256": config.inputs.synthetic_targets.sha256,
        "sourceCorpusSha256": config.inputs.source_corpus.sha256,
        "documentFeaturesSha256": config.inputs.document_features.sha256,
        "iso3166SnapshotSha256": config.target_integrity.iso3166_snapshot.sha256,
        "routeLocationsSha256": config.target_integrity.route_locations.sha256,
        "customsProgramRegistrySha256": (config.target_integrity.customs_program_registry.sha256),
        "packageRegistrySha256": config.target_integrity.package_registry.sha256,
        "editorPromptSha256": config.prompts.editor.sha256,
        "reviewerPromptSha256": config.prompts.reviewer.sha256,
        "editorSchemaSha256": sha256_bytes(canonical_json_bytes(editor_schema)),
        "reviewerSchemaSha256": sha256_bytes(canonical_json_bytes(reviewer_schema)),
        "caseDocumentIds": [row.document_id for row in config.cases],
        "implementationSha256": sha256_file(_IMPLEMENTATION_PATH),
        "sourceSchemaImplementationSha256": sha256_file(
            Path(bill_of_lading_v3.__file__).resolve(strict=True)
        ),
        "targetSchemaImplementationSha256": sha256_file(
            Path(bill_of_lading_v5.__file__).resolve(strict=True)
        ),
        "runtime": {
            "editorModel": config.providers.editor.model,
            "editorReasoningEffort": config.providers.editor.reasoning_effort,
            "reviewerModel": config.providers.reviewer.model,
            "reviewerReasoningEffort": config.providers.reviewer.reasoning_effort,
            "pydanticAiVersion": version("pydantic-ai-slim"),
            "openaiVersion": version("openai"),
        },
    }
    staged = StagedArtifactRun(
        output_parent=resolve_config_path(project_root, config.run.output_dir),
        run_name=config.run.run_id,
        transaction_sha256=sha256_bytes(canonical_json_bytes(transaction)),
    )
    if staged.completed:
        return cast(
            dict[str, JsonValue],
            json.loads(read_regular_file_bytes(staged.final_root / "generation/summary.json")),
        )
    staged.recover_interrupted_temporary_files()
    staged.publish_bytes("config.yaml", read_regular_file_bytes(config_path))
    staged.publish_bytes("prompts/editor.md", editor_prompt_bytes)
    staged.publish_bytes("prompts/reviewer.md", reviewer_prompt_bytes)
    staged.publish_bytes(
        "schema/line-range-replacement.schema.json", json_artifact_bytes(editor_schema)
    )
    staged.publish_bytes("schema/semantic-review.schema.json", json_artifact_bytes(reviewer_schema))
    started = time.perf_counter()
    bundles = asyncio.run(
        _run_cases(
            selected=selected,
            source_rows=tuple(sources),
            editor_prompt=editor_prompt,
            reviewer_prompt=reviewer_prompt,
            staged=staged,
            config=config,
            project_root=project_root,
        )
    )
    wall_seconds = time.perf_counter() - started
    for bundle in bundles:
        _publish_case_files(staged, bundle)
    total_usage = _combine_usage([row.result.totalUsage for row in bundles])
    usage_by_stage = _usage_by_stage(bundles)
    estimated_cost_per_thousand = total_usage.estimatedCostUsd / Decimal(len(bundles)) * 1000
    summary: dict[str, JsonValue] = {
        "schemaVersion": 15,
        "runId": config.run.run_id,
        "status": "complete",
        "cases": len(bundles),
        "qualityValidatedCases": sum(row.result.status == "quality_validated" for row in bundles),
        "needsReviewCases": sum(row.result.status == "needs_review" for row in bundles),
        "callFailedCases": sum(row.result.status == "call_failed" for row in bundles),
        "costLimitedCases": sum(row.result.status == "cost_limited" for row in bundles),
        "editorPasses": sum(row.result.editorPasses for row in bundles),
        "reviewPasses": sum(row.result.reviewPasses for row in bundles),
        "correctionCycles": sum(row.result.correctionCycles for row in bundles),
        "requests": total_usage.requests,
        "inputTokens": total_usage.inputTokens,
        "cacheReadTokens": total_usage.cacheReadTokens,
        "cacheWriteTokens": total_usage.cacheWriteTokens,
        "outputTokens": total_usage.outputTokens,
        "reasoningTokens": total_usage.reasoningTokens,
        "visibleOutputTokens": total_usage.visibleOutputTokens,
        "estimatedCostUsd": str(total_usage.estimatedCostUsd),
        "estimatedCostUsdPerThousand": str(estimated_cost_per_thousand),
        "usageByStage": cast(
            JsonValue,
            {stage: usage.model_dump(mode="json") for stage, usage in usage_by_stage.items()},
        ),
        "wallSeconds": wall_seconds,
        "throughputCasesPerHour": len(bundles) / wall_seconds * 3600.0,
        "peakResidentMemoryMiB": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0,
        "trainingRecordsPublished": False,
        "terminalOutputFunction": True,
        "lineAddressedAtomicPatches": True,
    }
    staged.publish_bytes(
        "generation/results.jsonl",
        b"".join(
            canonical_json_bytes(row.result.model_dump(mode="json")) + b"\n" for row in bundles
        ),
    )
    staged.publish_json("generation/summary.json", summary)
    for name, payload in _plot_bytes(bundles).items():
        staged.publish_bytes(f"plots/{name}", payload)
    staged.publish_bytes("REPORT.md", _report(summary, bundles).encode("utf-8"))
    commit = staged.commit(
        expected_artifacts=_artifact_inventory(staged.stage_root),
        metadata={"cases": len(bundles), "schema_version": 15, "training_records": 0},
    )
    summary["commitContentSha256"] = commit.receipt.content_sha256
    return summary
