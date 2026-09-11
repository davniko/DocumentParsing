"""Full-document inventory and regression guards for synthetic OCR rendering.

The raw-text editor is allowed to see only bounded source lines.  This module owns the
complementary safety contract: every source semantic value that changed, every source-only
auxiliary identifier/contact, and every shipment-dependent operational surface must be inventoried
before a rewritten OCR document can be considered training-ready.

The detectors here grant no write authority.  They only locate candidate source evidence and
provide deterministic postconditions.  A caller may render a candidate locally or delegate its
host-owned lines, but an unhandled candidate remains a blocking finding.
"""

from __future__ import annotations

import re
from collections import defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from decimal import Decimal
from typing import Annotated, Any, Literal, cast

from pydantic import BaseModel, ConfigDict, Field, JsonValue, StringConstraints

from document_ocr.hashing import canonical_json_bytes, sha256_bytes
from document_ocr.synthesis.container_semantics import review_source_equipment_surface
from document_ocr.synthesis.generators import (
    DeterministicStream,
    generate_from_surface_pattern,
    surface_pattern,
)
from document_ocr.synthesis.raw_text_hybrid_probe import (
    HybridWorkItem,
    _context_bound_surface_lines,
    _party_block_line_numbers,
    _party_heading_roles,
)
from document_ocr.synthesis.raw_text_rewrite_cycle_probe import (
    _PARTY_HEADING_LINE,
    _PRESERVABLE_LEGAL_BOILERPLATE_LINE,
    AnchoredScalarReplacementRequirement,
    AppliedDeterministicPrefill,
    ParsedMeasurementSurface,
    SurfaceRenderingRequirement,
    _numeric_span_overlaps_date,
    _parse_measurement_surface,
    _raw_agent_blocks,
    _render_measurement_surface,
)
from document_ocr.synthesis.raw_text_rewrite_probe import ChangedLeaf, changed_leaves

NonEmptyText = Annotated[str, StringConstraints(strip_whitespace=True, min_length=1)]
LineId = Annotated[str, StringConstraints(pattern=r"^L[0-9]{5}$")]
Sha256 = Annotated[str, StringConstraints(pattern=r"^[0-9a-f]{64}$")]
_STRICT = ConfigDict(extra="forbid", frozen=True, strict=True, allow_inf_nan=False)

_PAGE_MARKER = re.compile(r"^--- PAGE [1-9][0-9]* ---[ \t]*$")
_SPACE_RUN = re.compile(r"\s+")
_NUMBER = re.compile(r"(?<![0-9.,])[-+\N{MINUS SIGN}]?[0-9]+(?:[.,][0-9]+)*(?![0-9])")
_ATTACHED_NUMERIC_UNIT = re.compile(
    r"^(?:KGS?|KGM|KILOGRAMS?|CBM|M3|CUM|CUFT|MT|TONS?|"
    r"PKGS?|PACKAGES?|PLTS?|PALLETS?|CTNS?|CARTONS?|PCS?|PIECES?|"
    r"BAGS?|BALES?|BOX(?:ES)?|CRATES?|CASES?|DRUMS?|ROLLS?|BUNDLES?|"
    r"SETS?|LOTS?|UNITS?|SACKS?|JERRICANS?|TINS?|CANS?|BARRELS?|[CF])\b",
    re.IGNORECASE,
)
_IDENTIFIER_CANONICAL = re.compile(r"[^A-Za-z0-9]+")

# These are semantic field headings, not mappings from observed values.  Values remain opaque and
# are regenerated from their own surface grammar.  A line may contain several fields, so headings
# and values are parsed as bounded spans rather than greedily consuming the remainder of the line.
# The pattern deliberately requires explicit field syntax; generic legal prose never grants write
# authority merely because it mentions customs, invoices, taxes, or references.
_AUXILIARY_LABEL_PATTERN = (
    r"(?:"
    r"CONTACT\s+PERSON(?:\s+NAME)?|"
    r"ACID(?:\s*-\s*ADVANCE\s+CARGO\s+INFORMATION\s+DECLARATION)?"
    r"(?:\s+(?:NO|NUMBER|N[\N{DEGREE SIGN}\N{MASCULINE ORDINAL INDICATOR}]))?|"
    r"ACI(?:\s+(?:NO|NUMBER))?|"
    r"(?:EG(?:YPTIAN)?\s+)?(?:IMPORTER|CONSIGNEE|EXPORTER)?\s*"
    r"(?:TAX|VAT)(?:ATION)?(?:\s+(?:ID|NO|NUM|NUMBER))?|"
    r"(?:FOREIGN\s+)?(?:IMPORTER|CONSIGNEE|EXPORTER)"
    r"(?:\s+(?:REGISTRATION|IDENTIFICATION))?\s+(?:ID|NO|NUMBER)|"
    r"CUSTOMS(?:\s+(?:REFERENCE|REF|NO|NUMBER))?|"
    r"CERS|"
    r"CONSOLIDATION(?:\s+(?:NO|NUMBER))?|"
    r"CARGO\s+X\s+ID|"
    r"EORI(?:\s+(?:NO|NUMBER))?|"
    r"(?:CARRIER\s+)?BOOKING(?:\s+(?:REFERENCE|REF|NO|NUMBER))?|"
    r"CARRIER(?:'S|\N{RIGHT SINGLE QUOTATION MARK}S)?\s+"
    r"(?:REFERENCES?|REF)(?:\s+(?:NOS?|NUMBERS?))?|"
    r"(?:COMPANY|CARRIER)?\s*(?:REGISTRATION|REGISTERED)"
    r"(?:\s+(?:ID|NO|NUM|NUMBER))?|"
    r"R\s*\.?\s*C\s*\.?\s*S\s*\.?(?:\s+(?:ID|NO|NUM|NUMBER))?|"
    r"THERMOGRAPHS?(?:\s+(?:ID|NO|NUM|NUMBER))?|"
    r"SERVICE\s+CONTRACT(?:\s+(?:NO|NUMBER))?|"
    r"(?:CSO\s*/\s*)?AGREEMENT\s+(?:NO|NUMBER)|"
    r"INVOICE(?:\s+(?:NO|NUMBER))?|"
    r"S\s*/?\s*BILL(?:\s+(?:NO|NUMBER))?|"
    r"(?:OUTWARD\s+FORWARDERS?\s+)?"
    r"(?:(?:SUB|SHP|SHIPPER'?S|FORWARDERS?|DOCUMENT|DOC|EXTERNAL)\s+)?"
    r"(?:REFERENCES?|REF)(?:\s+(?:NOS?|NUMBERS?))?|"
    r"(?:PURCHASE\s+ORDER|P\s*\.?\s*O\.?)"
    r"(?![.\s]*BOX\b)(?:\s+(?:NO|NUMBER))?|"
    r"(?:PI|P\.I\.)(?:\s+(?:NO|NUMBER))?|"
    r"(?:AS\s+PER\s+)?ORDER(?:\s+(?:REF|REFERENCE|NO|NUMBER))?|"
    r"BATCH(?:\s+(?:NO|NUMBER))?|"
    r"LOT(?:\s+(?:REF|REFERENCE|NO|NUMBER))?|"
    r"PRODUCT\s+(?:CODE|ID|NO|NUMBER)|"
    r"RMS(?:\s+(?:NO|NUMBER))?|"
    r"CUSTOMER\s+CODE|"
    r"F\s*\.?\s*M\s*\.?\s*C\s*\.?(?:\s*-\s*OTI)?"
    r"(?:\s+(?:NO|NUMBER))?|"
    r"SHIPPER(?:'S)?\s+(?:ID|NO|NUMBER)|"
    r"SCAC(?:\s+CODE)?|"
    r"(?:SWIFT(?:\s*/\s*BIC)?|BIC)(?:\s+CODE)?|"
    r"ABN|"
    r"(?:DIRECT\s+)?(?:PHONE|TELEPHONE|TEL|MOBILE|PH|FAX)"
    r"(?:\s+(?:NO|NUMBER))?|"
    r"(?:EMAIL|E-MAIL)"
    r")"
)
_FIELD_SEPARATOR_PATTERN = r"(?:[ \t]*(?::|\#|=|\-)[ \t]*|[ \t]*\.[ \t]*|[ \t]+)"
_AUXILIARY_LABEL = re.compile(
    rf"(?ix)(?<![A-Z0-9])(?P<label>{_AUXILIARY_LABEL_PATTERN})"
    rf"(?P<separator>{_FIELD_SEPARATOR_PATTERN})"
)
_NON_AUXILIARY_FIELD_BOUNDARY = re.compile(
    rf"(?ix)(?<![A-Z0-9])(?P<label>DATE(?:\s+OF\s+ISSUE)?)"
    rf"(?P<separator>{_FIELD_SEPARATOR_PATTERN})"
)
_AUXILIARY_STANDALONE_HEADING = re.compile(rf"(?ix)^\s*(?:{_AUXILIARY_LABEL_PATTERN})[.:#=\-\s]*$")
_EMAIL_VALUE = re.compile(
    r"(?i)(?<![A-Z0-9._%+\-])[A-Z0-9._%+\-]+@[A-Z0-9.\-]+\.[A-Z]{2,}"
    r"(?![A-Z0-9._%+\-])"
)
_ELECTRONIC_TOKEN = re.compile(r"(?:https?://|www\.)[^\s]+|\S*@\S+", re.IGNORECASE)
_PHONE_VALUE = re.compile(
    r"(?<![A-Za-z0-9])\+?[ \t]*(?:\([0-9]{1,4}\)|[0-9]{1,4})"
    r"(?:[ \t()./\-]*[0-9]){5,}(?![A-Za-z0-9])"
)
_IDENTIFIER_VALUE = re.compile(
    r"(?<![A-Za-z0-9])(?:"
    r"[A-Za-z0-9]{2,}(?:[._/#\-][ \t]*[A-Za-z0-9]+)+|"
    r"[A-Za-z0-9][A-Za-z0-9._+@/#\-]{4,}"
    r")(?![A-Za-z0-9])"
)
_CONTACT_LABEL = re.compile(r"(?i)^(?:DIRECT\s+)?(?:PHONE|TELEPHONE|TEL|MOBILE|PH|FAX)")
_EMAIL_LABEL = re.compile(r"(?i)^(?:EMAIL|E-MAIL)$")
_CONTACT_PERSON_LABEL = re.compile(r"(?i)^CONTACT\s+PERSON(?:\s+NAME)?$")
_PRODUCT_IDENTIFIER_LABEL = re.compile(r"(?i)^PRODUCT\s+(?:CODE|ID|NO|NUMBER)$")
_ALPHABETIC_IDENTIFIER_LABEL = re.compile(
    r"(?i)^(?:SCAC(?:\s+CODE)?|(?:SWIFT(?:\s*/\s*BIC)?|BIC)(?:\s+CODE)?)$"
)
_ALPHABETIC_IDENTIFIER_VALUE = re.compile(r"(?i)(?<![A-Z0-9])[A-Z][A-Z0-9]{3,10}(?![A-Z0-9])")
_STANDALONE_OPAQUE_IDENTIFIER = re.compile(
    r"^[ \t*#()\[\]{}'\"]*"
    r"(?P<value>(?=[A-Za-z0-9._/#\-]{10,64}$)"
    r"(?=[A-Za-z0-9._/#\-]*[A-Za-z])"
    r"(?=(?:[^0-9]*[0-9]){3})"
    r"[A-Za-z0-9][A-Za-z0-9._/#\-]*)"
    r"[ \t*#()\[\]{}'\"]*$"
)
_STANDALONE_MEASUREMENT = re.compile(
    r"^[+\N{MINUS SIGN}-]?[0-9]+(?:[.,][0-9]+)*(?:"
    r"KGS?|KGM|KILOGRAMS?|CBM|M3|M\N{SUPERSCRIPT THREE}|CUM|CUFT|MT|TONS?"
    r")$",
    re.IGNORECASE,
)
_STANDALONE_DATE = re.compile(
    r"^(?:"
    r"[0-9]{1,4}[./-][0-9]{1,2}[./-][0-9]{1,4}|"
    r"[0-9]{1,2}[./-][A-Za-z]{3,9}[./-][0-9]{2,4}|"
    r"[A-Za-z]{3,9}[./-][0-9]{1,2}[./-][0-9]{2,4}"
    r")$"
)
_EMBEDDED_LONG_IDENTIFIER = re.compile(r"(?<![A-Z0-9])(?P<value>[0-9]{8,})(?![A-Z0-9])")
_BARE_TAX_OR_VAT_LABEL = re.compile(
    r"(?i)^(?:EG(?:YPTIAN)?\s+)?(?:IMPORTER|CONSIGNEE|EXPORTER)?\s*(?:TAX|VAT)$"
)
_AMBIGUOUS_BARE_LABEL = re.compile(r"(?i)^(?:BOOKING|INVOICE|CUSTOMS|REFERENCE|REFERENCES|REF)$")
_MULTI_VALUE_IDENTIFIER_LABEL = re.compile(
    r"(?i)^(?:(?:REFERENCES?|REF)[ \t]+(?:NOS?|NUMBERS?)|"
    r"CARRIER(?:'S|\N{RIGHT SINGLE QUOTATION MARK}S)?[ \t]+(?:REFERENCES?|REF)"
    r"(?:[ \t]+(?:NOS?|NUMBERS?))?|THERMOGRAPHS?)$"
)
_SPACED_REGISTRATION_LABEL = re.compile(
    r"(?i)^(?:(?:COMPANY|CARRIER)?\s*(?:REGISTRATION|REGISTERED)"
    r"(?:\s+(?:ID|NO|NUM|NUMBER))?|"
    r"R\s*\.?\s*C\s*\.?\s*S\s*\.?(?:\s+(?:ID|NO|NUM|NUMBER))?)$"
)
_SPACED_REGISTRATION_VALUE = re.compile(
    r"(?i)^[A-Z0-9][A-Z0-9./#\-]*(?:[ \t]+[A-Z0-9][A-Z0-9./#\-]*)+$"
)
_TRAILING_REGISTRATION_FIELD = re.compile(
    r"(?ix)^\s*(?P<value>[A-Z0-9][A-Z0-9./#\-]*"
    r"(?:[ \t]+[A-Z0-9][A-Z0-9./#\-]*)+?)[ \t]+"
    r"(?P<label>R\s*\.?\s*C\s*\.?\s*S\s*\.?|REG(?:ISTRATION)?\.?\s+NO\.?)\b"
)
_AGENCY_RELATION = re.compile(
    r"\b(?:AS\s+AGENT\s+FOR|ON\s+BEHALF\s+OF|TRADING\s+AS|T/?A)\b",
    re.IGNORECASE,
)
_REEFER_ASSERTION = re.compile(
    r"\b(?:REEFER\s+CARGO|REFRIGERATED(?:\s+CONTAINER)?|CARRYING\s+TEMPERATURE|"
    r"TEMPERATURE\s+(?:OF|SET|SETTING)|VENTIL+ATION\s+REQUIRED|"
    r"PLUGGING\s+FOR\s+THE\s+ACCOUNT)\b",
    re.IGNORECASE,
)
_SHIPMENT_AGGREGATE_FIELD = re.compile(
    r"(?ix)^\s*(?:\([0-9]+\)\s*|[0-9]+\.\s*)?(?:"
    r"TOTAL(?:S|\s+ITEMS?|\s+NUMBER|\s+NO\.|\s+GROSS|\s+NET|\s+WEIGHT|\s*:)|"
    r"CARGO\s+GROSS\s+WEIGHT|GROSS\s+WEIGHT|NET\s+WEIGHT|MEASUREMENT|"
    r"WEIGHT\s+IN\s+KGS\s+TOTAL|CARRIER['\N{RIGHT SINGLE QUOTATION MARK}]?S\s+RECEIPT|"
    r"[0-9]+\s+CONTAINERS?\s+SAID\s+TO\s+CONTAIN"
    r")(?=\s|:|$)"
)
_ATTACHED_PAGE_COUNT = re.compile(
    r"(?ix)^\s*TOTAL\s+(?:NUMBER|NO\.?)\s+OF\s+ATTACHED\s+"
    r"(?:[0-9][0-9,]*\s+PAGES?|PAGES?\s*:?\s*[0-9][0-9,]*)\s*$"
)
_POPULATED_CARRIER_RECEIPT_CONTAINER_COUNT = re.compile(
    r"(?ix)\bCARRIER['\N{RIGHT SINGLE QUOTATION MARK}]?S\s+RECEIPT\b"
    r"[^\r\n]{0,96}?\b(?P<count>[0-9][0-9,]*)\s+CONTAINER\(S\)(?=\s|$)"
)
_ACKNOWLEDGED_CARRIER_RECEIPT_CONTAINER_COUNT = re.compile(
    r"(?ix)^\s*TOTAL\s+(?:NUMBER|NO\.)\s+OF\s+CONTAINERS?/PACKAGES?\s+"
    r"RECEIVED\s*&\s*ACKNOWLEDGED\s+BY\s+(?:THE\s+)?CARRIER\s*"
    r"(?:FOR\s+THE\s+PURPOSE\s+OF\s+CALCULATION\s+OF\s+PACKAGE\s+LIMITATION\s*"
    r"\(\s*IF\s+APPLICABLE\s*\)\s*)?"
    r"(?:\(\s*SEE\s+CLAUSE\s+[0-9]+(?:\.[0-9]+)*\s*\))?\s*:\s*"
    r"(?P<count>[0-9][0-9,]*)\s+CONTAINER\(S\)/PACKAGE\(S\)\s*$"
)
_TRAILING_CARRIER_RECEIPT_CONTAINER_COUNT = re.compile(
    r"(?ix)^\s*TOTAL\s+(?:NUMBER|NO\.)\s+OF\s+CONTAINERS?\s+RECEIVED\s+BY\s+"
    r"(?:THE\s+)?CARRIER\s*:?\s*(?P<count>[0-9][0-9,]*)\s*$"
)
_TOTAL_CONTAINER_COUNT = re.compile(
    r"(?ix)^\s*(?:WEIGHT\s+IN\s+KGS\s+)?TOTAL\s*:?[ \t]*"
    r"(?P<count>[0-9][0-9,]*)[ \t]+CONTAINER\(S\)\s*$"
)
_SPLIT_CARRIER_RECEIPT_COUNT = re.compile(
    r"(?ix)^\s*TOTAL\s+(?:NUMBER|NO\.)\s+OF\s+CONTAINERS?\s+OR\s+PACKAGES?\s+"
    r"(?P<count>[0-9][0-9,]*)\s*$"
)
_CARRIER_RECEIPT_CONTINUATION = re.compile(r"(?ix)^\s*RECEIVED\s+BY\s+(?:THE\s+)?CARRIER\s*:?\s*$")
_LEADING_FIELD_ORDINAL = re.compile(r"^\s*(?:\([0-9]+\)|[0-9]+\.)\s*")
_CLAUSE_REFERENCE = re.compile(
    r"(?i)\b(?:SEE\s+)?CLAUSE\s+[0-9]+(?:\.[0-9]+)*"
    r"(?:\s*(?:AND|,|/)\s*[0-9]+(?:\.[0-9]+)*)*"
)
_CAPTION_FIELD_NUMBER = re.compile(
    r"(?i)^\s*(?:GROSS\s+WEIGHT|NET\s+WEIGHT|MEASUREMENT)\s*\([0-9]+\)\s*$"
)
_BRANDED_WEBSITE = re.compile(
    r"(?i)\b(?:web[ -]?site|website)\b[^\r\n]{0,80}"
    r"(?P<url>(?:https?://|www\.)[A-Z0-9][A-Z0-9._~:/?#\[\]@!$&'()*+,;=%-]*)"
)
_CARRIER_ALIAS = re.compile(
    r"(?i)(?P<alias>\([A-Z0-9][A-Z0-9 .&/-]{0,14}\))(?=\s*,?\s*AS\s+CARRIER\b)"
)
_SIGNED_CARRIER_IDENTITY = re.compile(
    r"(?i)^\s*SIGNED\s+(?!BY\b|FOR\b|ON\s+BEHALF\b)"
    r"(?P<identity>[A-Z0-9][A-Z0-9 .,&'()/+-]{3,180})\s*$"
)
_SIGNED_BY_HEADING = re.compile(r"(?i)^\s*SIGNED\s+BY\s*:?\s*$")
_SIGNING_RELATION_LINE = re.compile(r"(?i)^\s*(?:AS\s+AGENTS?\b|ON\s+BEHALF\b|BY\s*:?)")
_SOURCE_ONLY_FREE_TIME = re.compile(
    r"(?i)^\s*[0-9]{1,3}(?:\s*\([^\r\n)]{1,20}\))?\s+DAYS?\s+"
    r"(?:FREE\s*TIME|FREETIME)\b[^\r\n]{0,100}$"
)
_ANONYMOUS_EQUIPMENT_ASSERTION = re.compile(
    r"(?:"
    r"(?:/|X\s+)?(?:20|40|45)[\u2019'\"]?(?:\s*FT)?\s*"
    r"(?:HC|HQ|HIGH\s+CUBE|DRY|GP|DC|REEFER|RF)?"
    r"\s*CONTAINERS?\s+SAID\s+TO\s+CONTAIN"
    r"|"
    r"(?:TOTAL\s*:\s*)?[0-9]+\s*X\s*(?:20|40|45)[\u2019'\"]?\s*"
    r"(?:HC|HQ|HIGH\s+CUBE|DRY|GP|DC|REEFER|RF)?\s*"
    r"(?:LCL\s+)?(?:CNTR\(S\)|CONTAINERS?)"
    r")",
    re.IGNORECASE,
)
_WEIGHT_AGGREGATE_VALUE = re.compile(
    r"(?i)(?P<value>[0-9]+(?:[.,][0-9]+)*)[ \t]*"
    r"(?P<unit>KGS?|KILOGRAMS?|METRIC[ \t]+TON(?:NE)?S?|MTS?|LBS?|POUNDS?)\b"
)
_VOLUME_AGGREGATE_VALUE = re.compile(
    r"(?i)(?P<value>[0-9]+(?:[.,][0-9]+)*)[ \t]*"
    r"(?P<unit>M3|M\^3|CBM|CUBIC[ \t]+MET(?:ER|RE)S?)\b"
)
_BARE_VOLUME_AGGREGATE = re.compile(
    r"(?ix)^\s*(?:MEAS(?:UREMENT)?\s*[:=-]?\s*)?"
    r"[0-9]+(?:[.,][0-9]+)*[ \t]*"
    r"(?:M3|M\^3|CBM|CUBIC[ \t]+MET(?:ER|RE)S?)[ \t.;,:-]*$"
)
_GROSS_WEIGHT_HEADING = re.compile(
    r"(?i)\b(?:GROSS\s+(?:CARGO\s+)?WEIGHT|CARGO\s+GROSS\s+WEIGHT)\b"
)
_NET_WEIGHT_HEADING = re.compile(r"(?i)\bNET\s+(?:CARGO\s+)?WEIGHT\b")
_PARENTHESIZED_FIELD_NUMBER = re.compile(r"\([0-9]+\)")
_LEGACY_CONTAINER_TYPE_PATH = re.compile(
    r"^documentPatch\.containers\[([0-9]+)\]\.typeDescription$"
)

_MASS_TO_KG = {
    "kilogram": Decimal(1),
    "metric_tonne": Decimal(1000),
    "pound": Decimal("0.45359237"),
}


class InventoryCandidate(BaseModel):
    """One host-located mutable source surface and its intended disposition."""

    model_config = _STRICT

    candidateId: Annotated[str, StringConstraints(pattern=r"^I[0-9]{4}$")]
    category: Literal[
        "changed_source_occurrence",
        "changed_source_auxiliary_copy",
        "source_only_auxiliary",
        "source_only_contact_identity",
        "shipment_dependent_reefer",
        "shipment_dependent_aggregate",
        "carrier_dependent_branding",
        "source_only_signing_identity",
        "source_only_operational_scalar",
        "template_profile_residual",
        "regression_oracle_gap",
    ]
    sourceSurface: NonEmptyText
    lineIds: Annotated[tuple[LineId, ...], Field(min_length=1)]
    targetPaths: tuple[NonEmptyText, ...]
    targetSemantics: JsonValue
    disposition: Literal["deterministic_shape_replacement", "model_residual"]
    rationale: NonEmptyText


class DeterministicInventoryEdit(BaseModel):
    model_config = _STRICT

    sourceSurface: NonEmptyText
    targetSurface: NonEmptyText
    lineIds: Annotated[tuple[LineId, ...], Field(min_length=1)]
    method: Literal["hmac_shape_preserving_auxiliary_v1"]


class RegressionAssertion(BaseModel):
    model_config = _STRICT

    assertionId: NonEmptyText
    category: NonEmptyText
    sourceSurfacesMustDisappear: Annotated[tuple[NonEmptyText, ...], Field(min_length=1)]
    instruction: NonEmptyText

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> RegressionAssertion:
        payload = dict(value)
        surfaces = payload.get("sourceSurfacesMustDisappear")
        if isinstance(surfaces, list):
            payload["sourceSurfacesMustDisappear"] = tuple(surfaces)
        return cls.model_validate(payload, strict=True)


class RegressionCase(BaseModel):
    model_config = _STRICT

    documentId: Annotated[str, StringConstraints(pattern=r"^doc_[0-9a-f]{64}$")]
    assertions: Annotated[tuple[RegressionAssertion, ...], Field(min_length=1)]


class RegressionOracle(BaseModel):
    model_config = _STRICT

    schemaVersion: Literal[1]
    name: NonEmptyText
    cases: Annotated[tuple[RegressionCase, ...], Field(min_length=1)]


class TemplateMutationLine(BaseModel):
    """One independently audited mutable line in a reusable source-template profile."""

    model_config = _STRICT

    lineId: LineId
    sourceLineSha256: Sha256
    findingKinds: Annotated[tuple[NonEmptyText, ...], Field(min_length=1)]


class TemplateMutationCase(BaseModel):
    """Mutation evidence tied to the exact bytes and topology of one source template."""

    model_config = _STRICT

    documentId: Annotated[str, StringConstraints(pattern=r"^doc_[0-9a-f]{64}$")]
    sourceTextSha256: Sha256
    lines: Annotated[tuple[TemplateMutationLine, ...], Field(min_length=1)]


class TemplateMutationProfileSource(BaseModel):
    """Pinned local evidence used to build a mutation profile."""

    model_config = _STRICT

    path: NonEmptyText
    sha256: Sha256
    kind: Literal[
        "certification_commit",
        "manual_audit",
    ]


class TemplateMutationProfile(BaseModel):
    """Reviewed source-template line ownership reusable across synthetic descendants."""

    model_config = _STRICT

    schemaVersion: Literal[1]
    name: NonEmptyText
    sources: Annotated[tuple[TemplateMutationProfileSource, ...], Field(min_length=1)]
    cases: Annotated[tuple[TemplateMutationCase, ...], Field(min_length=1)]


class FullDocumentFinding(BaseModel):
    model_config = _STRICT

    findingId: NonEmptyText
    category: Literal[
        "retained_changed_source_value",
        "retained_source_auxiliary",
        "reefer_state_contradiction",
        "regression_oracle_failure",
        "inventory_candidate_unhandled",
        "semantic_slot_mismatch",
        "lexical_line_degenerated",
        "legal_relation_topology_mismatch",
    ]
    lineIds: tuple[LineId, ...]
    sourceSurface: NonEmptyText
    targetPaths: tuple[NonEmptyText, ...]
    explanation: NonEmptyText


class FullDocumentAudit(BaseModel):
    model_config = _STRICT

    documentId: Annotated[str, StringConstraints(pattern=r"^doc_[0-9a-f]{64}$")]
    sourceTextSha256: Sha256
    outputTextSha256: Sha256
    inventorySha256: Sha256
    candidates: Annotated[int, Field(ge=0)]
    deterministicCandidates: Annotated[int, Field(ge=0)]
    residualCandidates: Annotated[int, Field(ge=0)]
    findings: tuple[FullDocumentFinding, ...]
    passed: bool


@dataclass(frozen=True, slots=True)
class LocatedAuxiliaryValue:
    category: str
    value: str
    line_numbers: tuple[int, ...]


def line_id(line_number: int) -> str:
    if line_number <= 0:
        raise ValueError("line number must be positive")
    return f"L{line_number:05d}"


def line_number(value: str) -> int:
    if re.fullmatch(r"L[0-9]{5}", value) is None:
        raise ValueError(f"invalid line ID: {value!r}")
    return int(value[1:])


def _semantic_surface(value: str) -> str:
    return " ".join(value.casefold().split())


def _source_value_group_key(value: str | int | float) -> tuple[str, str]:
    """Group only case/whitespace-equivalent source values into one semantic decision.

    Reviewed labels can preserve the casing printed by different occurrences of the same fact
    (for example ``TAIWAN`` as a party country and ``Taiwan`` as cargo origin). Treating those as
    independent source surfaces can assign two incompatible targets to the same physical line.
    Numbers remain separate from text so a string identifier such as ``"20"`` is never merged
    with a measured numeric value merely because their display happens to match.
    """

    if isinstance(value, str):
        return "text", _semantic_surface(value)
    return "number", str(value)


def _party_surface_key(value: str) -> str:
    """Normalize only punctuation and spacing for complete party-scalar ownership checks."""

    return " ".join(_IDENTIFIER_CANONICAL.sub(" ", value).casefold().split())


def _contains_surface(text: str, surface: str) -> bool:
    pieces = [piece for piece in _SPACE_RUN.split(surface.strip()) if piece]
    if not pieces:
        return False
    stripped = surface.strip()
    prefix = r"(?<![A-Za-z0-9])" if stripped[0].isalnum() else ""
    suffix = r"(?![A-Za-z0-9])" if stripped[-1].isalnum() else ""
    return (
        re.search(
            prefix + r"\s+".join(re.escape(piece) for piece in pieces) + suffix,
            text,
            re.IGNORECASE,
        )
        is not None
    )


def _contains_rendered_surface(text: str, surface: str) -> bool:
    """Match an exact renderer-owned surface without imposing word boundaries.

    OCR frequently concatenates a field label and its value (for example ``ON2024-03-16``).
    A path-owned rendering requirement already proves semantic ownership, so an alphanumeric
    boundary would reject a correct substitution solely because the source omitted whitespace.
    """

    pieces = [piece for piece in _SPACE_RUN.split(surface.strip()) if piece]
    return (
        bool(pieces)
        and re.search(r"\s+".join(re.escape(piece) for piece in pieces), text, re.IGNORECASE)
        is not None
    )


def _contains_ordered_party_address(text: str, address: str) -> bool:
    """Accept a wrapped address whose own party city/country is interleaved by the template.

    Flattened OCR often places the city between two address fragments.  Exact role-block
    ownership is established separately; here we require every alphanumeric address atom, in
    order, so an interleaved same-party locality is allowed without weakening completeness.
    """

    target_atoms = re.findall(r"[^\W_]+", address.casefold(), re.UNICODE)
    observed_atoms = re.findall(r"[^\W_]+", text.casefold(), re.UNICODE)
    if not target_atoms:
        return False
    cursor = iter(observed_atoms)
    return all(any(observed == target for observed in cursor) for target in target_atoms)


def _is_freight_option_header(raw_line: str) -> bool:
    """Identify a static form caption that prints both freight-payment options."""

    normalized = _semantic_surface(raw_line)
    # A shipment can select either arrangement, while a caption/column header prints both.  The
    # slash is not stable after OCR (``Prepaid/Collect`` and ``Prepaid Collect`` are equivalent),
    # so the simultaneous presence of both mutually exclusive terms is the semantic proof.  A
    # separate selected value such as ``FREIGHT PREPAID`` contains only one and remains mutable.
    return "prepaid" in normalized and "collect" in normalized


def _source_only_signing_identity_blocks(text: str) -> tuple[tuple[str, set[int]], ...]:
    """Locate identities occupying lines beneath an explicit ``SIGNED BY`` heading."""

    lines = text.splitlines()
    output: list[tuple[str, set[int]]] = []
    for heading_index, raw_line in enumerate(lines):
        if _SIGNED_BY_HEADING.fullmatch(raw_line) is None:
            continue
        cursor = heading_index + 1
        while cursor < len(lines) and not lines[cursor].strip():
            cursor += 1
        identity_lines: list[tuple[int, str]] = []
        while cursor < len(lines) and len(identity_lines) < 3:
            candidate = lines[cursor]
            if (
                not candidate.strip()
                or _PAGE_MARKER.fullmatch(candidate) is not None
                or _PARTY_HEADING_LINE.fullmatch(candidate) is not None
                or _SIGNING_RELATION_LINE.match(candidate) is not None
                or _AGENCY_RELATION.search(candidate) is not None
            ):
                break
            identity_lines.append((cursor + 1, candidate.strip()))
            cursor += 1
        if identity_lines:
            output.append(
                (
                    "\n".join(value for _, value in identity_lines),
                    {number for number, _ in identity_lines},
                )
            )
    return tuple(output)


def _surface_lines(text: str, surface: str) -> set[int]:
    lines: set[int] = set()
    for number, line in enumerate(text.splitlines(), start=1):
        if _contains_surface(line, surface):
            lines.add(number)
    if lines:
        return lines
    canonical = _IDENTIFIER_CANONICAL.sub("", surface).casefold()
    if len(canonical) >= 7 and any(character.isdigit() for character in canonical):
        for number, line in enumerate(text.splitlines(), start=1):
            if canonical in _IDENTIFIER_CANONICAL.sub("", line).casefold():
                lines.add(number)
    return lines


def _party_surface_lines(text: str, surface: str) -> set[int]:
    """Locate a complete party scalar despite immaterial OCR punctuation differences.

    Reviewed labels commonly retain commas that OCR drops (``Transport, S.A.`` versus
    ``TRANSPORT S.A.``).  General fuzzy matching would grant unsafe write authority, so this is
    restricted to a complete, token-bound alphanumeric party scalar on one physical line.
    Electronic-contact spans are removed first to avoid treating a company name repeated in an
    email domain or URL as another legal identity.
    """

    normalized_surface = _party_surface_key(surface)
    if len(normalized_surface) < 4:
        return set()
    pattern = re.compile(
        rf"(?<![a-z0-9]){re.escape(normalized_surface)}(?![a-z0-9])",
        re.IGNORECASE,
    )
    output: set[int] = set()
    for number, raw_line in enumerate(text.splitlines(), start=1):
        scrubbed = _ELECTRONIC_TOKEN.sub(" ", raw_line)
        normalized_line = _party_surface_key(scrubbed)
        if pattern.search(normalized_line) is not None:
            output.add(number)
    return output


def _numeric_values(surface: str) -> frozenset[Decimal]:
    value = surface.strip().replace("\N{MINUS SIGN}", "-")
    if re.fullmatch(r"[-+]?[0-9]+(?:[.,][0-9]+)*", value) is None:
        return frozenset()
    sign = Decimal(-1) if value.startswith("-") else Decimal(1)
    unsigned = value.lstrip("+-")
    output: set[Decimal] = set()
    separators = {character for character in unsigned if character in ".,"}
    if not separators:
        output.add(sign * Decimal(unsigned))
    elif len(separators) == 1:
        separator = next(iter(separators))
        pieces = unsigned.split(separator)
        output.add(sign * Decimal(pieces[0] + "." + "".join(pieces[1:])))
        if all(len(piece) == 3 for piece in pieces[1:]):
            output.add(sign * Decimal("".join(pieces)))
    else:
        decimal_separator = "." if unsigned.rfind(".") > unsigned.rfind(",") else ","
        grouping_separator = "," if decimal_separator == "." else "."
        integer, fraction = unsigned.rsplit(decimal_separator, 1)
        output.add(sign * Decimal(integer.replace(grouping_separator, "") + "." + fraction))
    return frozenset(output)


def _numeric_lines(text: str, value: int | float) -> set[int]:
    expected = Decimal(str(value))
    output: set[int] = set()
    for number, line in enumerate(text.splitlines(), start=1):
        if _PAGE_MARKER.fullmatch(line) is not None:
            continue
        for match in _NUMBER.finditer(line):
            if _numeric_span_overlaps_date(line, match.start(), match.end()):
                continue
            if match.start() > 0 and line[match.start() - 1].isalnum():
                continue
            if (
                match.end() < len(line)
                and line[match.end()].isalpha()
                and _ATTACHED_NUMERIC_UNIT.match(line[match.end() :]) is None
            ):
                continue
            if expected in _numeric_values(match.group(0)):
                output.add(number)
                break
    return output


def _is_shipment_aggregate_value_line(raw_line: str) -> bool:
    """Recognize a populated shipment summary, never legal prose or a numbered caption.

    The OCR corpus contains long carriage clauses mentioning totals, packages, liability, and
    gross weight.  Keyword search is therefore unsafe.  A mutable summary must start with one of
    the observed field grammars and contain a numeric value after removing a leading form-field
    ordinal and any ``see clause`` cross-reference.  Captions such as ``MEASUREMENT (20)`` are
    explicitly not values.
    """

    if _ATTACHED_PAGE_COUNT.fullmatch(raw_line) is not None:
        return False
    if _SHIPMENT_AGGREGATE_FIELD.search(raw_line) is None:
        return False
    value_region = _LEADING_FIELD_ORDINAL.sub("", raw_line, count=1)
    if _CAPTION_FIELD_NUMBER.fullmatch(value_region) is not None:
        return False
    value_region = _CLAUSE_REFERENCE.sub("", value_region)
    return _NUMBER.search(value_region) is not None


def _carrier_receipt_count_matches_target(
    raw_line: str,
    target_label: Mapping[str, Any],
    *,
    following_line: str | None = None,
) -> bool:
    """Prove that a populated carrier-receipt count already matches target equipment.

    The count is a derived presentation of the explicit target container rows.  It may remain
    byte-identical only when the observed grammar is unambiguous and the target container list is
    present and complete.  A mismatch, missing target list, or unfamiliar grammar fails closed.
    """

    match = _POPULATED_CARRIER_RECEIPT_CONTAINER_COUNT.search(raw_line)
    if match is None:
        match = _ACKNOWLEDGED_CARRIER_RECEIPT_CONTAINER_COUNT.fullmatch(raw_line)
    if match is None:
        match = _TRAILING_CARRIER_RECEIPT_CONTAINER_COUNT.fullmatch(raw_line)
    if match is None:
        match = _TOTAL_CONTAINER_COUNT.fullmatch(raw_line)
    if match is None:
        split = _SPLIT_CARRIER_RECEIPT_COUNT.fullmatch(raw_line)
        if (
            split is not None
            and following_line is not None
            and _CARRIER_RECEIPT_CONTINUATION.fullmatch(following_line) is not None
        ):
            match = split
    if match is None:
        return False
    patch = target_label.get("documentPatch")
    containers = patch.get("containers") if isinstance(patch, Mapping) else None
    if (
        not isinstance(containers, Sequence)
        or isinstance(containers, (str, bytes))
        or not containers
    ):
        return False
    return int(match.group("count").replace(",", "")) == len(containers)


def _target_weight_total_kg(target_label: Mapping[str, Any], *, field_name: str) -> Decimal | None:
    """Return a complete cargo-weight total, or ``None`` when it cannot be proven.

    A printed shipment total is only safe to preserve when every target cargo group carries the
    corresponding measure.  Summing the subset that happens to be labelled would turn missing
    supervision into a false assertion of completeness.
    """

    patch = target_label.get("documentPatch")
    groups = patch.get("cargoGroups") if isinstance(patch, Mapping) else None
    if not isinstance(groups, Sequence) or isinstance(groups, (str, bytes)) or not groups:
        return None
    total = Decimal(0)
    for group in groups:
        measure = group.get(field_name) if isinstance(group, Mapping) else None
        if not isinstance(measure, Mapping):
            return None
        unit = measure.get("unit")
        value = measure.get("value")
        if (
            not isinstance(unit, str)
            or isinstance(value, bool)
            or not isinstance(value, (int, float, str, Decimal))
        ):
            return None
        factor = _MASS_TO_KG.get(unit)
        if factor is None:
            return None
        parsed = Decimal(str(value))
        if not parsed.is_finite() or parsed <= 0:
            return None
        total += parsed * factor
    return total


def _printed_weight_factor_to_kg(unit: str) -> Decimal | None:
    normalized = re.sub(r"\s+", " ", unit.strip().upper())
    if normalized in {"KG", "KGS", "KILOGRAM", "KILOGRAMS"}:
        return Decimal(1)
    if normalized in {"MT", "MTS", "METRIC TON", "METRIC TONS", "METRIC TONNE", "METRIC TONNES"}:
        return Decimal(1000)
    if normalized in {"LB", "LBS", "POUND", "POUNDS"}:
        return Decimal("0.45359237")
    return None


def _weight_aggregate_matches_target(raw_line: str, target_label: Mapping[str, Any]) -> bool:
    """Prove that a printed gross/net aggregate already represents the target.

    The comparison renders the canonical target total through the source number's grouping and
    precision envelope.  Thus ``9097 KG`` correctly represents a target of ``9096.8 kg`` while
    ``57,072.40 KGS`` must agree to two decimal places.  Lines containing another non-caption
    number are deliberately left mutable because they may combine a stale package/container
    count with an otherwise correct weight.
    """

    gross = _GROSS_WEIGHT_HEADING.search(raw_line) is not None
    net = _NET_WEIGHT_HEADING.search(raw_line) is not None
    if gross == net:
        return False
    measurements = tuple(_WEIGHT_AGGREGATE_VALUE.finditer(raw_line))
    if len(measurements) != 1:
        return False
    match = measurements[0]
    without_caption = _PARENTHESIZED_FIELD_NUMBER.sub("", raw_line)
    # Re-evaluate after caption removal instead of relying on shifted offsets.  The actual
    # measurement must be the only shipment value on the line.
    stripped_measurements = tuple(_WEIGHT_AGGREGATE_VALUE.finditer(without_caption))
    if len(stripped_measurements) != 1:
        return False
    stripped_match = stripped_measurements[0]
    other_region = (
        without_caption[: stripped_match.start()] + without_caption[stripped_match.end() :]
    )
    if _NUMBER.search(other_region) is not None:
        return False
    factor = _printed_weight_factor_to_kg(match.group("unit"))
    total_kg = _target_weight_total_kg(
        target_label, field_name="grossWeight" if gross else "netWeight"
    )
    if factor is None or total_kg is None:
        return False
    style: ParsedMeasurementSurface | None = _parse_measurement_surface(
        match.group("value"), maximum=None
    )
    if style is None:
        return False
    target_in_printed_unit = total_kg / factor
    return _render_measurement_surface(
        target_in_printed_unit,
        style=style,
        maximum=None,
    ) == match.group("value")


def _target_volume_total_m3(target_label: Mapping[str, Any]) -> Decimal | None:
    """Return a complete single-group cargo volume in cubic metres.

    An unowned bare ``M3``/``CBM`` row has no group identifier. It can only be preserved without
    opening a model-owned residual when the target has exactly one cargo group and that group
    carries an explicit volume. Multi-group rows remain mutable because a shipment total and a
    per-group value cannot be distinguished from the number alone.
    """

    patch = target_label.get("documentPatch")
    groups = patch.get("cargoGroups") if isinstance(patch, Mapping) else None
    if (
        not isinstance(groups, Sequence)
        or isinstance(groups, (str, bytes))
        or len(groups) != 1
        or not isinstance(groups[0], Mapping)
    ):
        return None
    measure = groups[0].get("volume")
    if not isinstance(measure, Mapping) or measure.get("unit") != "cubic_metre":
        return None
    value = measure.get("value")
    if isinstance(value, bool) or not isinstance(value, (int, float, str, Decimal)):
        return None
    parsed = Decimal(str(value))
    return parsed if parsed.is_finite() and parsed > 0 else None


def _volume_aggregate_matches_target(raw_line: str, target_label: Mapping[str, Any]) -> bool:
    """Prove one explicit volume surface agrees at its printed precision."""

    measurements = tuple(_VOLUME_AGGREGATE_VALUE.finditer(raw_line))
    if len(measurements) != 1:
        return False
    match = measurements[0]
    other_region = raw_line[: match.start()] + raw_line[match.end() :]
    if _NUMBER.search(other_region) is not None:
        return False
    total_m3 = _target_volume_total_m3(target_label)
    if total_m3 is None:
        return False
    surface = match.group("value")
    styles: list[ParsedMeasurementSurface] = []
    parsed = _parse_measurement_surface(surface, maximum=None)
    if parsed is not None:
        styles.append(parsed)
    separators = {character for character in surface if character in ".,"}
    if len(separators) == 1:
        separator = next(iter(separators))
        pieces = surface.split(separator)
        if len(pieces) == 2:
            integer, fraction = pieces
            styles.append(
                ParsedMeasurementSurface(
                    value=Decimal(integer + "." + fraction),
                    decimal_separator=separator,
                    grouping_separator=None,
                    decimal_places=len(fraction),
                )
            )
    return any(
        _render_measurement_surface(total_m3, style=style, maximum=None) == surface
        for style in styles
    )


def _flatten_scalars(value: Any) -> tuple[JsonValue, ...]:
    output: list[JsonValue] = []
    if isinstance(value, Mapping):
        for child in value.values():
            output.extend(_flatten_scalars(child))
    elif isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        for child in value:
            output.extend(_flatten_scalars(child))
    elif value is None or isinstance(value, (str, int, float, bool)):
        output.append(cast(JsonValue, value))
    return tuple(output)


def locate_auxiliary_values(text: str) -> tuple[LocatedAuxiliaryValue, ...]:
    """Locate opaque source-only values from explicit field syntax.

    The parser intentionally does not infer meaning from a value.  It requires a semantic heading
    and extracts only high-entropy identifier/contact tokens, so generic carrier boilerplate is
    never made writable merely because it mentions customs, invoices, or references.
    """

    def value_surfaces(label: str, separator: str, segment: str) -> tuple[str, ...]:
        value = segment.strip(" \t:;,#*.")
        if not value:
            return ()
        # Bare TAX/VAT headings followed only by whitespace are legitimate in surfaces such as
        # ``EGYPT VAT 712098623`` but not in the measurement label ``TAX CBM: 25.630``.  Requiring
        # a digit in the first token distinguishes an identifier from another alphabetic field
        # qualifier without maintaining a brittle list of forbidden measurement words.
        if (
            (
                _BARE_TAX_OR_VAT_LABEL.fullmatch(label.strip()) is not None
                or _AMBIGUOUS_BARE_LABEL.fullmatch(label.strip()) is not None
            )
            and not separator.strip()
            and not any(character.isdigit() for character in value.split(maxsplit=1)[0])
        ):
            return ()
        if _EMAIL_LABEL.fullmatch(label.strip()) is not None:
            return tuple(match.group(0) for match in _EMAIL_VALUE.finditer(value))
        if _CONTACT_LABEL.match(label.strip()) is not None:
            return tuple(match.group(0).strip() for match in _PHONE_VALUE.finditer(value))
        if _CONTACT_PERSON_LABEL.fullmatch(label.strip()) is not None:
            letters = tuple(character for character in value if character.isalpha())
            if (
                bool(separator.strip())
                and len(value) <= 160
                and len(letters) >= 3
                and not any(character.isdigit() for character in value)
                and "@" not in value
            ):
                return (value,)
            return ()
        if _ALPHABETIC_IDENTIFIER_LABEL.fullmatch(label.strip()) is not None:
            return tuple(match.group(0) for match in _ALPHABETIC_IDENTIFIER_VALUE.finditer(value))
        if _SPACED_REGISTRATION_LABEL.fullmatch(label.strip()) is not None:
            canonical = re.sub(r"[^A-Z0-9]", "", value.upper())
            if (
                _SPACED_REGISTRATION_VALUE.fullmatch(value) is not None
                and len(canonical) >= 6
                and sum(character.isdigit() for character in canonical) >= 5
            ):
                return (value,)
        candidates = tuple(match.group(0).strip() for match in _IDENTIFIER_VALUE.finditer(value))
        filtered = tuple(
            candidate
            for candidate in candidates
            if any(character.isdigit() or character == "@" for character in candidate)
        )
        # A singular identifier heading owns one following value. OCR frequently keeps unrelated
        # cargo columns on that same physical line (for example ``PO-123 96 CARTONS /40HQ/``).
        # Treating every later identifier-shaped token as part of the purchase order corrupted
        # package and equipment surfaces. Explicit plural reference headings are the one syntax
        # in this grammar that deliberately licenses multiple values.
        if _MULTI_VALUE_IDENTIFIER_LABEL.fullmatch(label.strip()) is not None:
            return filtered
        if _PRODUCT_IDENTIFIER_LABEL.fullmatch(label.strip()) is not None:
            # A product-code row can include an intervening presentation such as
            # ``100G*100BAGS/CARTON``. The printed identifier is the final identifier-shaped
            # token in that field, not a package token inside the formulation.
            return filtered[-1:]
        return filtered[:1]

    lines = text.splitlines()
    located: dict[tuple[str, str], set[int]] = defaultdict(set)
    for index, raw_line in enumerate(lines, start=1):
        # E-mail addresses are intrinsically private contact values even when OCR drops their
        # heading or places the address on the following line. Unlike generic identifiers, their
        # syntax is self-describing and therefore grants narrowly bounded write authority.
        for match in _EMAIL_VALUE.finditer(raw_line):
            located[("email", match.group(0))].add(index)
        trailing_registration = _TRAILING_REGISTRATION_FIELD.match(raw_line)
        if trailing_registration is not None:
            value = trailing_registration.group("value").strip()
            canonical = re.sub(r"[^A-Z0-9]", "", value.upper())
            if (
                _SPACED_REGISTRATION_VALUE.fullmatch(value) is not None
                and len(canonical) >= 6
                and sum(character.isdigit() for character in canonical) >= 5
            ):
                located[(_semantic_surface(trailing_registration.group("label")), value)].add(index)
        trailing_heading_label: str | None = None
        if len(raw_line) <= 500:
            auxiliary_markers = list(_AUXILIARY_LABEL.finditer(raw_line))
            boundaries = sorted(
                [
                    (match.start(), match.end())
                    for match in (
                        *auxiliary_markers,
                        *_NON_AUXILIARY_FIELD_BOUNDARY.finditer(raw_line),
                    )
                ],
                key=lambda row: (row[0], -(row[1] - row[0])),
            )
            for match in auxiliary_markers:
                end = next(
                    (
                        boundary_start
                        for boundary_start, _boundary_end in boundaries
                        if boundary_start >= match.end()
                    ),
                    len(raw_line),
                )
                label = _semantic_surface(match.group("label"))
                segment = raw_line[match.end() : end]
                observed_values = value_surfaces(
                    match.group("label"), match.group("separator"), segment
                )
                for value in observed_values:
                    located[(label, value)].add(index)
                # OCR can append an auxiliary heading to another populated row and place its
                # value on the next physical line, e.g. ``(...)/ACID NUMBER:\n123...``.  The
                # explicit trailing heading still owns that next-line value; requiring the whole
                # current line to be a heading missed exactly this common form.
                if (
                    not observed_values
                    and not segment.strip(" \t:;,#*.")
                    and not raw_line[match.end() :].strip()
                ):
                    trailing_heading_label = label
        heading_match = _AUXILIARY_STANDALONE_HEADING.fullmatch(raw_line)
        if heading_match is not None or trailing_heading_label is not None:
            following = index
            while following < len(lines) and not lines[following].strip():
                following += 1
            if following < len(lines):
                value_line = lines[following]
                if len(value_line) <= 200 and not (
                    _AUXILIARY_STANDALONE_HEADING.fullmatch(value_line) is not None
                    or _NON_AUXILIARY_FIELD_BOUNDARY.match(value_line) is not None
                ):
                    label = trailing_heading_label or _semantic_surface(
                        raw_line.strip(" \t:;,#*.-")
                    )
                    for value in value_surfaces(label, ":", value_line):
                        located[(label, value)].add(following + 1)
    return tuple(
        LocatedAuxiliaryValue(category=category, value=value, line_numbers=tuple(sorted(numbers)))
        for (category, value), numbers in sorted(located.items())
    )


def locate_standalone_opaque_identifiers(text: str) -> tuple[LocatedAuxiliaryValue, ...]:
    """Locate uncaptioned identifier rows without interpreting their value vocabulary.

    Some B/L templates print a company-registration number, cargo tracking value, or the
    continuation of a split identifier on a line of its own.  The structural contract is narrow:
    the complete non-whitespace row must be one 10--64-character mixed alphanumeric token with at
    least three digits.  Numeric measurements and date-shaped values are explicitly excluded.
    Whether a located value is source-only is decided later against the target label; this helper
    only reports the observed surface and line.
    """

    located: dict[str, set[int]] = defaultdict(set)
    for line_number_value, raw_line in enumerate(text.splitlines(), start=1):
        match = _STANDALONE_OPAQUE_IDENTIFIER.fullmatch(raw_line)
        if match is None:
            continue
        value = match.group("value")
        if (
            _STANDALONE_MEASUREMENT.fullmatch(value) is not None
            or _STANDALONE_DATE.fullmatch(value) is not None
            or _EMAIL_VALUE.fullmatch(value) is not None
        ):
            continue
        located[value].add(line_number_value)
    return tuple(
        LocatedAuxiliaryValue(
            category="unlabelled opaque identifier",
            value=value,
            line_numbers=tuple(sorted(line_numbers)),
        )
        for value, line_numbers in sorted(located.items())
    )


def auxiliary_label_surfaces(line: str) -> tuple[str, ...]:
    """Return exact semantic field-label surfaces that a line editor must preserve."""

    return tuple(dict.fromkeys(match.group(0) for match in _AUXILIARY_LABEL.finditer(line)))


def locate_embedded_long_identifiers(text: str) -> tuple[LocatedAuxiliaryValue, ...]:
    """Locate unlabeled long numeric identifiers while excluding dates, phones, and measures."""

    located: dict[str, set[int]] = defaultdict(set)
    for line_number_value, raw_line in enumerate(text.splitlines(), start=1):
        phone_spans = tuple(
            phone.span()
            for marker in _AUXILIARY_LABEL.finditer(raw_line)
            if _CONTACT_LABEL.match(marker.group("label")) is not None
            for phone in _PHONE_VALUE.finditer(raw_line, marker.end())
        )
        excluded_spans = (
            *phone_spans,
            *(match.span() for match in _STANDALONE_DATE.finditer(raw_line)),
            *(match.span() for match in _EMAIL_VALUE.finditer(raw_line)),
        )
        for match in _EMBEDDED_LONG_IDENTIFIER.finditer(raw_line):
            if any(start < match.end() and match.start() < end for start, end in excluded_spans):
                continue
            trailing = raw_line[match.end() :].lstrip()
            if _ATTACHED_NUMERIC_UNIT.match(trailing) is not None:
                continue
            located[match.group("value")].add(line_number_value)
    return tuple(
        LocatedAuxiliaryValue(
            category="unlabelled embedded numeric identifier",
            value=value,
            line_numbers=tuple(sorted(line_numbers)),
        )
        for value, line_numbers in sorted(located.items())
    )


def _label_has_temperature(label: Mapping[str, Any]) -> bool:
    patch = label.get("documentPatch")
    containers = patch.get("containers") if isinstance(patch, Mapping) else None
    return (
        isinstance(containers, Sequence)
        and not isinstance(containers, (str, bytes))
        and any(
            isinstance(container, Mapping) and container.get("temperatureSetpoint") is not None
            for container in containers
        )
    )


def _target_equipment_semantics(target_label: Mapping[str, Any]) -> JsonValue:
    patch = cast(Mapping[str, Any], target_label.get("documentPatch") or {})
    output: list[dict[str, JsonValue]] = []
    for container in patch.get("containers") or ():
        if not isinstance(container, Mapping):
            continue
        size = container.get("sizeCategory")
        kind = container.get("typeCategory")
        if isinstance(size, str) and isinstance(kind, str):
            output.append(
                {
                    "sizeCategory": size,
                    "typeCategory": kind,
                    "temperatureControlled": container.get("temperatureSetpoint") is not None,
                }
            )
    return cast(JsonValue, output)


def _party_name(label: Mapping[str, Any], role: str) -> str | None:
    patch = label.get("documentPatch")
    parties = patch.get("parties") if isinstance(patch, Mapping) else None
    party = parties.get(role) if isinstance(parties, Mapping) else None
    value = party.get("name") if isinstance(party, Mapping) else None
    return value if isinstance(value, str) and value.strip() else None


def _target_string_surfaces(label: Mapping[str, Any]) -> tuple[str, ...]:
    return tuple(
        str(value) for value in _flatten_scalars(label) if isinstance(value, str) and value.strip()
    )


def _surface_required_by_target(surface: str, target_surfaces: Sequence[str]) -> bool:
    """Return whether a source token sequence is intentionally present in a target value.

    This is deliberately boundary-aware: ``EGYPT`` is not licensed by ``Egyptian``, while
    ``ALEXANDRIA`` is licensed by ``El Iskandariya (Alexandria)``.
    """

    return any(_contains_surface(target_surface, surface) for target_surface in target_surfaces)


_REFERENCE_CONTEXT = re.compile(
    r"\b(?:EXPORT(?:[ \t]+REFERENCES?)?|INVOICE|SHIPPING[ \t]+BILL|"
    r"SB[ \t]*(?:NO\.?|NUMBER)|CUSTOMS|REFERENCE|REF(?:ERENCE)?[ \t]*NO|"
    r"F/?AGENT[^\r\n]{0,30}\bREF|SHIPPER'?S[ \t]+REF)\b",
    re.IGNORECASE,
)


def _target_licensed_occurrence_lines(
    text: str,
    source_surface: str,
    target_surfaces: Sequence[str],
) -> set[int]:
    """Return source occurrences that are proven constituents of a rendered target value.

    A changed cargo token can legitimately survive inside an unrelated target field, such as a
    crop year also appearing inside a new forwarding-reference date.  Licensing is occurrence
    based: the complete target must be rendered on that line (or in the explicit two-line
    forwarding-reference grammar).  Other copies remain mutable and cannot hide behind a global
    substring coincidence.
    """

    bodies = text.splitlines()
    occurrence_lines = _surface_lines(text, source_surface)
    licensed: set[int] = set()
    containing_targets = tuple(
        target for target in target_surfaces if _contains_surface(target, source_surface)
    )
    for number in occurrence_lines:
        for target in containing_targets:
            if _contains_surface(bodies[number - 1], target):
                licensed.add(number)
                break
            atoms = re.findall(r"[A-Z0-9]+", target.upper())
            if len(atoms) < 2:
                continue
            source_line = bodies[number - 1]
            continuation = (
                re.match(
                    r"^[ \t]*(?:DATED?|DT)\b",
                    source_line,
                    re.IGNORECASE,
                )
                is not None
            )
            starts = (
                (max(1, number - 1), number)
                if _REFERENCE_CONTEXT.search(source_line) is not None or continuation
                else (number,)
            )
            for start in starts:
                end = min(
                    len(bodies),
                    start + 1
                    if _REFERENCE_CONTEXT.search(source_line) is not None or continuation
                    else start,
                )
                window = "\n".join(bodies[start - 1 : end])
                observed = re.findall(r"[A-Z0-9]+", window.upper())
                cursor = iter(observed)
                if _REFERENCE_CONTEXT.search(window) is not None and all(
                    any(value == atom for value in cursor) for atom in atoms
                ):
                    licensed.add(number)
                    break
            if number in licensed:
                break
    return licensed


def _oracle_surface_applies(surface: str, *, target_label: Mapping[str, Any]) -> bool:
    """Keep historical false-pass fixtures semantic under a regenerated target.

    The oracle records source surfaces, not immutable desired outputs.  A phrase that correctly
    describes the new topology must not become forbidden merely because an earlier synthetic
    target removed that topology.  Private identifiers and changed scalar values remain covered
    by their normal inventory/audit paths.
    """

    target_surfaces = _target_string_surfaces(target_label)
    if _surface_required_by_target(surface, target_surfaces):
        return False
    if _REEFER_ASSERTION.search(surface) is not None and _label_has_temperature(target_label):
        return False
    patch = target_label.get("documentPatch")
    containers = patch.get("containers") if isinstance(patch, Mapping) else None
    return not (_ANONYMOUS_EQUIPMENT_ASSERTION.search(surface) is not None and not containers)


def _candidate_key(
    category: str, source_surface: str, line_numbers: Sequence[int], paths: Sequence[str]
) -> tuple[str, str, tuple[int, ...], tuple[str, ...]]:
    return category, source_surface, tuple(sorted(line_numbers)), tuple(sorted(paths))


def _party_role_path(path: str) -> str | None:
    """Return the role-level schema path for one party leaf."""

    match = re.match(
        r"^(documentPatch\.parties\.(?:notifyParties\[[0-9]+\]|[A-Za-z]+))(?:\.|$)",
        path,
    )
    return match.group(1) if match is not None else None


def _work_item_lines_by_path(
    work_items: Sequence[HybridWorkItem],
) -> dict[str, set[int]]:
    """Index host-proven line ownership by exact semantic path."""

    output: dict[str, set[int]] = defaultdict(set)
    for item in work_items:
        numbers = {line_number(value) for value in item.evidenceLineIds}
        for path in item.targetPaths:
            output[path].update(numbers)
    return output


def _is_projected_legacy_equipment_leaf(
    leaf: ChangedLeaf,
    target_label: Mapping[str, Any],
) -> bool:
    """Exclude a v3 display-only type string once v5 owns its semantic categories.

    ``typeDescription`` intentionally disappears from the v5 task label.  Its disappearance is
    not an instruction to erase or fictionalize the printed equipment surface: the paired
    ``sizeCategory`` and ``typeCategory`` fields own that surface through the cross-schema
    equipment projection.  Treating the legacy leaf as an ordinary removal previously changed
    valid reefer lines into dry-container text.
    """

    match = _LEGACY_CONTAINER_TYPE_PATH.fullmatch(leaf.path)
    if match is None or not isinstance(leaf.sourceValue, str):
        return False
    patch = target_label.get("documentPatch")
    containers = patch.get("containers") if isinstance(patch, Mapping) else None
    index = int(match.group(1))
    if (
        not isinstance(containers, Sequence)
        or isinstance(containers, (str, bytes))
        or index >= len(containers)
        or not isinstance(containers[index], Mapping)
    ):
        return False
    target = cast(Mapping[str, Any], containers[index])
    return isinstance(target.get("sizeCategory"), str) and isinstance(
        target.get("typeCategory"), str
    )


def _raw_agent_identity_surfaces_by_line(
    text: str,
    *,
    source_carrier: str | None,
    target_carrier: str | None,
) -> dict[int, tuple[str, ...]]:
    """Return source-only agent identities by occupied line, excluding carrier principals."""

    output: dict[int, list[str]] = defaultdict(list)
    accepted_principals = tuple(
        normalized
        for value in (source_carrier, target_carrier)
        if value is not None and (normalized := _semantic_surface(value))
    )
    for number, identity, _gap, principal, _evidence in _raw_agent_blocks(text):
        normalized_principal = _semantic_surface(principal)
        generic_principal = normalized_principal in {"carrier", "the carrier"}
        if (
            accepted_principals
            and not generic_principal
            and not any(
                expected in normalized_principal or normalized_principal in expected
                for expected in accepted_principals
            )
        ):
            continue
        for offset, _fragment in enumerate(identity.splitlines()):
            output[number + offset].append(identity)
    return {line: tuple(identities) for line, identities in output.items()}


def _raw_agent_identity_line_numbers(
    text: str,
    *,
    source_carrier: str | None,
    target_carrier: str | None,
) -> set[int]:
    """Return lines occupied by a source-only agent identity, excluding its carrier principal."""

    return set(
        _raw_agent_identity_surfaces_by_line(
            text,
            source_carrier=source_carrier,
            target_carrier=target_carrier,
        )
    )


def build_mutable_inventory(
    *,
    source_text: str,
    current_text: str,
    source_label: Mapping[str, Any],
    target_label: Mapping[str, Any],
    work_items: Sequence[HybridWorkItem],
    oracle_case: RegressionCase | None,
    template_profile_case: TemplateMutationCase | None = None,
    surface_requirements: Sequence[SurfaceRenderingRequirement] = (),
    anchored_replacements: Sequence[AnchoredScalarReplacementRequirement] = (),
) -> tuple[InventoryCandidate, ...]:
    """Compile all automatically detectable mutable evidence before a model request."""

    target_scalars = {
        _semantic_surface(str(value))
        for value in _flatten_scalars(target_label)
        if isinstance(value, (str, int, float)) and not isinstance(value, bool)
    }
    source_scalars = {
        _semantic_surface(str(value))
        for value in _flatten_scalars(source_label)
        if isinstance(value, (str, int, float)) and not isinstance(value, bool)
    }
    target_string_surfaces = _target_string_surfaces(target_label)
    current_text_lines = current_text.splitlines()
    raw_candidates: dict[tuple[str, str, tuple[int, ...], tuple[str, ...]], dict[str, Any]] = {}

    def add(
        *,
        category: str,
        source_surface: str,
        lines: set[int],
        paths: Sequence[str],
        target: JsonValue,
        disposition: str,
        rationale: str,
    ) -> None:
        if not lines or not source_surface.strip():
            return
        key = _candidate_key(category, source_surface, tuple(lines), paths)
        raw_candidates[key] = {
            "category": category,
            "sourceSurface": source_surface,
            "lineIds": tuple(line_id(value) for value in sorted(lines)),
            "targetPaths": tuple(paths),
            "targetSemantics": target,
            "disposition": disposition,
            "rationale": rationale,
        }

    leaves = changed_leaves(source_label, target_label)
    work_item_lines = _work_item_lines_by_path(work_items)
    work_items_by_line: dict[int, list[HybridWorkItem]] = defaultdict(list)
    for item in work_items:
        for evidence_line_id in item.evidenceLineIds:
            work_items_by_line[line_number(evidence_line_id)].append(item)
    raw_agent_identities_by_line = _raw_agent_identity_surfaces_by_line(
        current_text,
        source_carrier=_party_name(source_label, "carrier"),
        target_carrier=_party_name(target_label, "carrier"),
    )
    raw_agent_lines = set(raw_agent_identities_by_line)
    leaves_by_surface: dict[tuple[str, str], list[ChangedLeaf]] = defaultdict(list)
    for leaf in leaves:
        if not leaf.requiresTextEdit or leaf.sourceValue == leaf.targetValue:
            continue
        if _is_projected_legacy_equipment_leaf(leaf, target_label):
            continue
        if not isinstance(leaf.sourceValue, (str, int, float)) or isinstance(
            leaf.sourceValue, bool
        ):
            continue
        source_surface = str(leaf.sourceValue)
        if len(source_surface.strip()) < 2:
            continue
        leaves_by_surface[_source_value_group_key(leaf.sourceValue)].append(leaf)

    # A raw-only entity can contain several strings that also occur in labeled fields.  If one
    # constituent has multiple path-specific targets and this physical line belongs to none of
    # those paths, the line is proven to be auxiliary context.  Do not then bind a second,
    # single-target constituent on that same line to an arbitrary labeled role.  For example,
    # ``Collection Business Unit Maersk Taiwan Ltd - Taipei`` is neither the notify party nor a
    # route location merely because ``TAIPEI`` has one task target elsewhere in the document.
    # This is derived entirely from changed-label ambiguity and compiler-owned lines; it does not
    # introduce organization or locality aliases.
    conflicted_unowned_lines: set[int] = set()
    for related in leaves_by_surface.values():
        if not related or not all(isinstance(row.sourceValue, str) for row in related):
            continue
        unique_targets = {canonical_json_bytes(row.targetValue) for row in related}
        if len(unique_targets) <= 1:
            continue
        source_surface = min(
            (cast(str, row.sourceValue) for row in related),
            key=lambda value: (len(value), value.casefold(), value),
        )
        path_owned_lines = {
            number for row in related for number in work_item_lines.get(row.path, ())
        }
        conflicted_unowned_lines.update(
            _surface_lines(current_text, source_surface) - path_owned_lines
        )

    for _surface_key, related in sorted(leaves_by_surface.items()):
        # Every string in a group differs only by casing/whitespace. ``_surface_lines`` is
        # case-insensitive and whitespace-flexible, so one stable representative locates every
        # occurrence while the complete set of path targets remains available below.
        source_surface = min(
            (str(row.sourceValue) for row in related),
            key=lambda value: (len(value), value.casefold(), value),
        )
        if all(isinstance(row.sourceValue, (int, float)) for row in related):
            path_owned_lines = {
                line_number(line)
                for item in work_items
                if set(item.targetPaths) & {row.path for row in related}
                for line in item.evidenceLineIds
            }
            # Numeric equality alone does not establish semantic identity: a source temperature
            # of ``1`` may coexist with page 1, one original B/L, or one container.  Only a
            # path-owned occurrence can represent the changed leaf.  Shipment-wide numeric
            # derivatives outside those lines are inventoried separately by their explicit
            # aggregate/measurement grammar below.
            lines = (
                _numeric_lines(current_text, cast(int | float, related[0].sourceValue))
                & path_owned_lines
            )
        elif all(".contactDetails." in row.path for row in related):
            # Punctuation is part of a contact value's identity and role evidence. Falling back
            # to punctuation-stripped matching would bind ``+202-...`` from a changed consignee
            # field to the distinct source-only emergency number ``202-...``.
            lines = {
                number
                for number, line in enumerate(current_text.splitlines(), start=1)
                if _contains_surface(line, source_surface)
            }
        else:
            lines = _surface_lines(current_text, source_surface)
        party_name_or_address = all(
            row.path.startswith("documentPatch.parties.")
            and row.path.rsplit(".", 1)[-1] in {"name", "address"}
            for row in related
        )
        if party_name_or_address:
            lines.update(_party_surface_lines(current_text, source_surface))
            current_text_lines = current_text.splitlines()
            lines.difference_update(
                number
                for number in tuple(lines)
                if _PARTY_HEADING_LINE.fullmatch(current_text_lines[number - 1]) is not None
                or _PRESERVABLE_LEGAL_BOILERPLATE_LINE.fullmatch(current_text_lines[number - 1])
                is not None
            )
            # A party name can be a constituent of another changed scalar, most notably a
            # ``C/O <company>`` phrase inside a different party's address. The larger scalar's
            # path-owned work item already owns and validates that line; treating the embedded
            # name as another occurrence of its own party creates a false cardinality obligation.
            lines.difference_update(
                number
                for number in tuple(lines)
                if any(
                    isinstance(item.sourceValue, str)
                    and _semantic_surface(item.sourceValue) != _semantic_surface(source_surface)
                    and _contains_surface(item.sourceValue, source_surface)
                    for item in work_items_by_line.get(number, ())
                )
            )
            party_paths = {row.path for row in related}
            source_key = _party_surface_key(source_surface)
            # A party name may also be printed as the exact value of a different labeled field.
            # For example, marks and numbers can be an exporter name that differs from the
            # shipper only by punctuation.  Party matching is intentionally punctuation-tolerant,
            # but an exact foreign scalar is stronger evidence than that tolerant match.  Give
            # the exact scalar its own path and do not create an impossible dual-target line.
            foreign_exact_lines = {
                number
                for number in tuple(lines)
                if not re.search(
                    re.escape(source_surface),
                    current_text_lines[number - 1],
                    flags=re.IGNORECASE,
                )
                and any(
                    row.path not in party_paths
                    and isinstance(row.sourceValue, str)
                    and row.sourceValue != source_surface
                    and _party_surface_key(row.sourceValue) == source_key
                    and re.search(
                        re.escape(row.sourceValue),
                        current_text_lines[number - 1],
                        flags=re.IGNORECASE,
                    )
                    is not None
                    for row in leaves
                )
            }
            lines.difference_update(foreign_exact_lines)
            # A shorter party scalar can be embedded in a different complete labeled identity.
            # ``TRANSGLORY`` inside forwarding agent ``TRANSGLORY, S.A.`` remains owned by the
            # forwarding-agent path even when OCR punctuation differs from the reviewed label.
            # Complete tolerant party-scalar matching proves the foreign owner; a loose token or
            # organization-name dictionary would not.
            embedded_foreign_owner_lines: set[int] = set()
            for row in leaves:
                if (
                    row.path in party_paths
                    or not isinstance(row.sourceValue, str)
                    or row.sourceValue == source_surface
                ):
                    continue
                foreign_key = _party_surface_key(row.sourceValue)
                if (
                    foreign_key != source_key
                    and re.search(
                        rf"(?<![a-z0-9]){re.escape(source_key)}(?![a-z0-9])",
                        foreign_key,
                    )
                    is not None
                ):
                    embedded_foreign_owner_lines.update(
                        _party_surface_lines(current_text, row.sourceValue)
                    )
            lines.difference_update(embedded_foreign_owner_lines)
            # A punctuation-equivalent party scalar can intentionally be printed as a different
            # task field (most commonly marks and numbers).  That path-owned occurrence must
            # render its own target; assigning the same physical line to the party as well creates
            # an impossible dual-target contract.  Only an explicit work item with the same
            # normalized source surface may claim precedence, so unrelated values sharing a line
            # do not hide genuine repeated party identities.
            lines.difference_update(
                number
                for number in tuple(lines)
                if any(
                    not party_paths.intersection(item.targetPaths)
                    and isinstance(item.sourceValue, str)
                    and _party_surface_key(item.sourceValue) == source_key
                    for item in work_items_by_line.get(number, ())
                )
            )
        if all(row.path == "documentPatch.freight.paymentArrangement" for row in related):
            # The same words occur in static option captions, payer-amendment boilerplate, and
            # individual charge rows. Only compiler-proven evidence represents the task-level
            # freight arrangement and is therefore authorized to change.
            lines.intersection_update(
                line_number(line)
                for item in work_items
                if "documentPatch.freight.paymentArrangement" in item.targetPaths
                for line in item.evidenceLineIds
            )
        # A locality, carrier token, or other label scalar can also occur *inside* a distinct
        # raw-only signing-agent identity. That substring belongs to the identity anonymization
        # contract, not to the coincidentally equal freight, route, or party field. Exclude only
        # the lines whose parsed identity contains the scalar; other content on the same physical
        # line (for example a vessel before the agent name) remains owned by its normal work item.
        lines.difference_update(
            number
            for number in tuple(lines)
            if any(
                _contains_surface(identity, source_surface)
                for identity in raw_agent_identities_by_line.get(number, ())
            )
        )
        # A larger changed scalar with a different schema path owns its complete printed line.
        # This commonly occurs when an origin country is embedded in a marks string (``Made in
        # Taiwan``) or a party locality is embedded in a complete source-only identity.  Giving
        # the embedded scalar a second independent target creates contradictory instructions.
        # The larger exact source value and compiler-owned line provide the ownership proof.
        related_paths = {row.path for row in related}
        lines.difference_update(
            number
            for number in tuple(lines)
            if any(
                isinstance(item.sourceValue, str)
                and _semantic_surface(item.sourceValue) != _semantic_surface(source_surface)
                and related_paths.isdisjoint(item.targetPaths)
                and _contains_surface(item.sourceValue, source_surface)
                for item in work_items_by_line.get(number, ())
            )
        )
        # An exact shared scalar is handled by path-owned work items and role validation.  When
        # the source is only a constituent of a larger target, license only the occurrences that
        # actually render that complete target; separate stale copies remain mutable.
        if _semantic_surface(source_surface) in target_scalars:
            continue
        if _surface_required_by_target(source_surface, target_string_surfaces):
            lines.difference_update(
                _target_licensed_occurrence_lines(
                    current_text,
                    source_surface,
                    target_string_surfaces,
                )
            )
        if not lines:
            continue
        related_by_path = {row.path: row for row in related}
        line_groups: dict[tuple[str, ...], set[int]] = defaultdict(set)
        for number in lines:
            owners = tuple(
                sorted(path for path in related_by_path if number in work_item_lines.get(path, ()))
            )
            line_groups[owners].add(number)
        for owners, owned_lines in sorted(line_groups.items()):
            if owners:
                target_map = [
                    {"path": path, "target": related_by_path[path].targetValue} for path in owners
                ]
                add(
                    category="changed_source_occurrence",
                    source_surface=source_surface,
                    lines=owned_lines,
                    paths=owners,
                    target=cast(JsonValue, target_map),
                    disposition="model_residual",
                    rationale=(
                        "This occurrence belongs to the listed host-proven semantic paths and "
                        "must render only their path-specific targets."
                    ),
                )
                continue

            owned_party_paths = tuple(
                sorted(path for path in related_by_path if _party_role_path(path) is not None)
            )
            carrier_principal_paths = tuple(
                path for path in owned_party_paths if path == "documentPatch.parties.carrier.name"
            )
            carrier_principal_lines = owned_lines - raw_agent_lines
            if carrier_principal_paths and carrier_principal_lines:
                carrier_target = related_by_path[carrier_principal_paths[0]].targetValue
                add(
                    category="changed_source_occurrence",
                    source_surface=source_surface,
                    lines=carrier_principal_lines,
                    paths=carrier_principal_paths,
                    target=cast(
                        JsonValue,
                        [
                            {
                                "path": carrier_principal_paths[0],
                                "target": carrier_target,
                            }
                        ],
                    ),
                    disposition="model_residual",
                    rationale=(
                        "An unblocked carrier-principal copy must carry the exact synthetic "
                        "carrier identity; explicit source-only agent lines are excluded."
                    ),
                )
                owned_lines = owned_lines - carrier_principal_lines
            if not owned_lines:
                continue

            # An explicitly captioned foreign party role is source-only flavor, not another copy
            # of the labeled source party. For example, a notify-party name repeated as
            # ``EXPORTER: ...`` must become a distinct fictional exporter rather than forcing an
            # extra notify-party target occurrence. Role captions are structural grammar; no
            # company-name alias or observed-value mapping is involved.
            source_roles = {
                "notify"
                if ".notifyParties[" in path
                else path.split("documentPatch.parties.", 1)[1].split(".", 1)[0]
                for path in owned_party_paths
            }
            explicit_foreign_role_lines = (
                {
                    number
                    for number in owned_lines
                    if (roles := _party_heading_roles(current_text_lines[number - 1]))
                    and roles.isdisjoint(source_roles)
                }
                if owned_party_paths
                else set()
            )
            if explicit_foreign_role_lines:
                add(
                    category="changed_source_auxiliary_copy",
                    source_surface=source_surface,
                    lines=explicit_foreign_role_lines,
                    paths=tuple(sorted(related_by_path)),
                    target={
                        "policy": "synthesize_distinct_context_compatible_auxiliary_value",
                        "mustDifferFromSource": True,
                        "mustNotDuplicateAnyLabeledTarget": True,
                    },
                    disposition="model_residual",
                    rationale=(
                        "The source scalar is printed under an explicit different party-role "
                        "caption and must be fictionalized as that auxiliary role."
                    ),
                )
                owned_lines = owned_lines - explicit_foreign_role_lines
            if not owned_lines:
                continue

            unique_targets = {
                canonical_json_bytes(row.targetValue) for row in related_by_path.values()
            }
            if len(unique_targets) == 1:
                auxiliary_lines = owned_lines & conflicted_unowned_lines
                if auxiliary_lines:
                    add(
                        category="changed_source_auxiliary_copy",
                        source_surface=source_surface,
                        lines=auxiliary_lines,
                        paths=tuple(sorted(related_by_path)),
                        target={
                            "policy": "synthesize_distinct_context_compatible_auxiliary_value",
                            "mustDifferFromSource": True,
                            "mustNotDuplicateAnyLabeledTarget": True,
                        },
                        disposition="model_residual",
                        rationale=(
                            "This unowned scalar shares a physical line with a changed source "
                            "value that has multiple path-specific targets. The line is therefore "
                            "source-only auxiliary context, not another labeled-role occurrence."
                        ),
                    )
                    owned_lines = owned_lines - auxiliary_lines
                if not owned_lines:
                    continue
                target_map = [
                    {
                        "path": sorted(related_by_path)[0],
                        "target": next(iter(related_by_path.values())).targetValue,
                    }
                ]
                add(
                    category="changed_source_occurrence",
                    source_surface=source_surface,
                    lines=owned_lines,
                    paths=tuple(sorted(related_by_path)),
                    target=cast(JsonValue, target_map),
                    disposition="model_residual",
                    rationale=(
                        "An unowned repeated scalar copy has one unambiguous semantic target and "
                        "must render that same target. Explicit legal-agent identities were "
                        "removed before this decision."
                    ),
                )
                continue

            add(
                category="changed_source_auxiliary_copy",
                source_surface=source_surface,
                lines=owned_lines,
                paths=tuple(sorted(related_by_path)),
                target={
                    "policy": "synthesize_distinct_context_compatible_auxiliary_value",
                    "mustDifferFromSource": True,
                    "mustNotDuplicateAnyLabeledTarget": True,
                },
                disposition="model_residual",
                rationale=(
                    "This source value is printed outside every path-owned scalar slot. It must "
                    "be fictionalized as auxiliary flavor, not duplicated as a labeled target."
                ),
            )

    current_lines = current_text.splitlines()
    explicit_auxiliary = locate_auxiliary_values(source_text)
    explicit_auxiliary_occurrences = {
        (value.value, line_number_value)
        for value in explicit_auxiliary
        for line_number_value in value.line_numbers
    }
    standalone_auxiliary = tuple(
        auxiliary
        for auxiliary in locate_standalone_opaque_identifiers(source_text)
        if not _surface_required_by_target(auxiliary.value, target_string_surfaces)
    )
    embedded_auxiliary = tuple(
        auxiliary
        for auxiliary in locate_embedded_long_identifiers(source_text)
        if _semantic_surface(auxiliary.value) not in source_scalars
        and _semantic_surface(auxiliary.value) not in target_scalars
        and not any(
            (auxiliary.value, line_number_value) in explicit_auxiliary_occurrences
            for line_number_value in auxiliary.line_numbers
        )
    )
    for auxiliary in (*explicit_auxiliary, *standalone_auxiliary, *embedded_auxiliary):
        # Deterministic shape replacement requires the exact punctuation-bearing source bytes.
        # A task-owned phone may have already been replaced while a punctuation-equivalent
        # source-only phone remains elsewhere (``+202-...`` versus ``202-...``).  The canonical
        # fallback used by semantic inventory discovery must not authorize an exact local write
        # with bytes that are absent from that line; the separately parsed observed auxiliary
        # surface owns the remaining value.
        contact_person = _CONTACT_PERSON_LABEL.fullmatch(auxiliary.category) is not None
        contact_value = (
            contact_person
            or _CONTACT_LABEL.match(auxiliary.category) is not None
            or _EMAIL_LABEL.fullmatch(auxiliary.category) is not None
        )
        candidate_lines = (
            set(auxiliary.line_numbers)
            if contact_value
            else _surface_lines(current_text, auxiliary.value)
        )
        auxiliary_lines = {
            number
            for number in candidate_lines
            if 1 <= number <= len(current_lines)
            and _contains_surface(current_lines[number - 1], auxiliary.value)
            # An explicit auxiliary-style heading can also be the printed grammar for a
            # task-facing field (for example ``BATCH: PR7XB93`` or
            # ``INVOICE NO. 1123200938 DATED ...``).  The semantic work item owns that
            # occurrence and its target value.  Independently anonymizing the embedded token
            # would create two contradictory targets for one physical span.
            and not any(
                any(
                    isinstance(source_scalar, str)
                    and _contains_surface(source_scalar, auxiliary.value)
                    for source_scalar in _flatten_scalars(item.sourceValue)
                )
                for item in work_items_by_line.get(number, ())
            )
        }
        if not auxiliary_lines:
            continue
        add(
            category=("source_only_contact_identity" if contact_value else "source_only_auxiliary"),
            source_surface=auxiliary.value,
            lines=auxiliary_lines,
            paths=(),
            target=(
                {
                    "policy": "synthesize_realistic_fictional_person_identity",
                    "preserveRepeatedIdentity": True,
                    "preservePrintedDelimiters": True,
                }
                if contact_person
                else (
                    {
                        "policy": "synthesize_realistic_fictional_contact_value",
                        "semanticField": auxiliary.category,
                        "preservePrintedDelimiters": True,
                        "preserveRepeatedIdentity": True,
                    }
                    if contact_value
                    else {
                        "policy": "synthesize_shape_compatible_fictional_value",
                        "semanticField": auxiliary.category,
                        "preserveRepeatedIdentity": True,
                    }
                )
            ),
            disposition=("model_residual" if contact_value else "deterministic_shape_replacement"),
            rationale=(
                "A named source-only contact must be replaced with a realistic fictional "
                "identity while preserving its printed role and delimiters."
                if contact_value
                else (
                    "An explicit source-only identifier or contact value must be anonymized "
                    "even though it is outside the task label."
                )
            ),
        )

    if _label_has_temperature(source_label) and not _label_has_temperature(target_label):
        for number, raw_line in enumerate(current_text.splitlines(), start=1):
            if _REEFER_ASSERTION.search(raw_line) is not None:
                add(
                    category="shipment_dependent_reefer",
                    source_surface=raw_line.strip(),
                    lines={number},
                    paths=("documentPatch.containers[].temperatureSetpoint",),
                    target={
                        "thermalOperation": "ambient_or_not_asserted",
                        "targetEquipmentSemantics": _target_equipment_semantics(target_label),
                    },
                    disposition="model_residual",
                    rationale=(
                        "The source asserts active refrigerated operation while the target has no "
                        "temperature-controlled container."
                    ),
                )

    cargo_changed = any(
        row.path.startswith(
            (
                "documentPatch.cargoGroups",
                "documentPatch.cargoPackages",
                "documentPatch.cargoAllocationGroups",
                "documentPatch.containers",
            )
        )
        and row.requiresTextEdit
        for row in leaves
    )
    if cargo_changed:
        current_lines = current_text.splitlines()
        known_lines = {line_number(value) for item in work_items for value in item.evidenceLineIds}
        # Deterministic/path-owned renderers run before this residual inventory.  Their output
        # lines are already governed by stricter exact-surface postconditions and must never be
        # re-opened as generic shipment aggregates merely because the rendered line starts with
        # ``TOTAL`` or ``CARRIER'S RECEIPT``.
        known_lines.update(
            line_number(value)
            for requirement in anchored_replacements
            for value in requirement.sourceLineIds
        )
        for requirement in surface_requirements:
            for number in _context_bound_surface_lines(source_text, requirement):
                if 1 <= number <= len(current_lines) and _contains_surface(
                    current_lines[number - 1], requirement.targetSurface
                ):
                    known_lines.add(number)
        target_patch = target_label.get("documentPatch")
        target_containers = (
            target_patch.get("containers") if isinstance(target_patch, Mapping) else None
        )
        target_has_explicit_containers = bool(
            isinstance(target_containers, Sequence)
            and not isinstance(target_containers, (str, bytes))
            and target_containers
        )
        for number, raw_line in enumerate(current_text.splitlines(), start=1):
            aggregate_line = _is_shipment_aggregate_value_line(raw_line)
            bare_volume_line = _BARE_VOLUME_AGGREGATE.fullmatch(raw_line) is not None
            if number not in known_lines and (aggregate_line or bare_volume_line):
                following_line = current_lines[number] if number < len(current_lines) else None
                if _carrier_receipt_count_matches_target(
                    raw_line,
                    target_label,
                    following_line=following_line,
                ):
                    continue
                # A source document can print an anonymous equipment summary even when the task
                # label has no container identities.  In that case the line is valid source-only
                # topology, not a cargo-package total.  Rewriting ``2 x 40 HC containers`` as
                # ``9 skids`` corrupts the document's semantics; preserve it verbatim and let the
                # separately labelled package-total line carry the synthetic package change.
                if (
                    not target_has_explicit_containers
                    and _ANONYMOUS_EQUIPMENT_ASSERTION.search(raw_line) is not None
                ):
                    continue
                if _weight_aggregate_matches_target(raw_line, target_label):
                    continue
                if _volume_aggregate_matches_target(raw_line, target_label):
                    continue
                add(
                    category="shipment_dependent_aggregate",
                    source_surface=raw_line.strip(),
                    lines={number},
                    paths=("documentPatch.cargoGroups", "documentPatch.cargoPackages"),
                    target={
                        "policy": "reconcile_from_target_cargo_and_container_graph",
                        "targetCargo": cast(
                            JsonValue,
                            cast(Mapping[str, Any], target_label.get("documentPatch", {})).get(
                                "cargoGroups"
                            ),
                        ),
                        "targetPackages": cast(
                            JsonValue,
                            cast(Mapping[str, Any], target_label.get("documentPatch", {})).get(
                                "cargoPackages"
                            ),
                        ),
                    },
                    disposition="model_residual",
                    rationale=(
                        "A printed aggregate or measurement outside the label-owned lines is "
                        "shipment-dependent and cannot retain the source shipment value."
                    ),
                )

    source_carrier = _party_name(source_label, "carrier")
    target_carrier = _party_name(target_label, "carrier")
    if source_carrier != target_carrier and target_carrier is not None:
        for match in _BRANDED_WEBSITE.finditer(current_text):
            value = match.group("url").rstrip(".,;)")
            # Deterministic anchored replacement can already have rendered the target carrier's
            # website before inventory compilation. A target-owned surface is not stale branding.
            if _semantic_surface(value) in target_scalars or _surface_required_by_target(
                value, target_string_surfaces
            ):
                continue
            add(
                category="carrier_dependent_branding",
                source_surface=value,
                lines=_surface_lines(current_text, value),
                paths=("documentPatch.parties.carrier.name",),
                target={
                    "policy": "synthesize_carrier_coherent_fictional_website",
                    "targetCarrierName": target_carrier,
                },
                disposition="model_residual",
                rationale=(
                    "A carrier-branded website outside the task label must not identify the "
                    "source carrier after its identity changes."
                ),
            )
        for number, raw_line in enumerate(current_text.splitlines(), start=1):
            alias = _CARRIER_ALIAS.search(raw_line)
            if alias is None:
                continue
            add(
                category="carrier_dependent_branding",
                source_surface=alias.group("alias"),
                lines={number},
                paths=("documentPatch.parties.carrier.name",),
                target={
                    "policy": "synthesize_short_alias_from_target_carrier",
                    "targetCarrierName": target_carrier,
                },
                disposition="model_residual",
                rationale=(
                    "A printed short carrier alias must agree with the synthetic carrier name."
                ),
            )
        text_lines = current_text.splitlines()
        for index, raw_line in enumerate(text_lines):
            signed = _SIGNED_CARRIER_IDENTITY.fullmatch(raw_line)
            if signed is None:
                continue
            following = "\n".join(text_lines[index + 1 : index + 9])
            if _AGENCY_RELATION.search(following) is None:
                continue
            identity = signed.group("identity").strip()
            add(
                category="source_only_signing_identity",
                source_surface=identity,
                lines={index + 1},
                paths=("documentPatch.parties.carrier.name",),
                target={
                    "policy": "synthesize_distinct_carrier_signing_agent_or_affiliate",
                    "targetCarrierName": target_carrier,
                    "mustDifferFromTargetCarrier": True,
                    "preserveSignedPrefix": True,
                },
                disposition="model_residual",
                rationale=(
                    "A source-only signing agent or affiliate linked to the carrier must be "
                    "fictionalized as a distinct identity while preserving the agency relation."
                ),
            )

    for identity, identity_lines in _source_only_signing_identity_blocks(current_text):
        # If the complete multiline identity is exactly the labeled carrier, the normal changed
        # party contract owns it.  Otherwise it is raw-only signing flavor and must be regenerated
        # even when the synthetic target intentionally has no carrier field.
        if (
            source_carrier is not None
            and target_carrier is not None
            and _semantic_surface(identity) == _semantic_surface(source_carrier)
        ):
            continue
        add(
            category="source_only_signing_identity",
            source_surface=identity,
            lines=identity_lines,
            paths=("documentPatch.parties.carrier.name",) if target_carrier else (),
            target={
                "policy": "synthesize_distinct_carrier_signing_agent_or_affiliate",
                "targetCarrierName": target_carrier,
                "mustDifferFromNamedTargetParties": True,
                "preserveSignedByHeading": True,
                "preserveOccupiedLineCount": True,
            },
            disposition="model_residual",
            rationale=(
                "A multiline source-only signing identity must be fictionalized across its "
                "existing occupied lines while preserving the SIGNED BY topology."
            ),
        )

    for number, raw_line in enumerate(current_text.splitlines(), start=1):
        if _SOURCE_ONLY_FREE_TIME.fullmatch(raw_line) is None:
            continue
        add(
            category="source_only_operational_scalar",
            source_surface=raw_line.strip(),
            lines={number},
            paths=(),
            target={
                "policy": "synthesize_plausible_distinct_value_preserve_printed_grammar",
                "semanticField": "container_free_time_days",
            },
            disposition="model_residual",
            rationale=(
                "A shipment-specific free-time scalar outside the task label must be "
                "regenerated rather than copied from the source shipment."
            ),
        )

    if oracle_case is not None:
        for assertion in oracle_case.assertions:
            for surface in assertion.sourceSurfacesMustDisappear:
                if not _oracle_surface_applies(surface, target_label=target_label):
                    continue
                lines = _surface_lines(current_text, surface)
                if lines:
                    add(
                        category="regression_oracle_gap",
                        source_surface=surface,
                        lines=lines,
                        paths=(),
                        target={"instruction": assertion.instruction},
                        disposition="model_residual",
                        rationale=(
                            "A manually adjudicated false-pass regression surface remains in the "
                            "candidate output."
                        ),
                    )

    if template_profile_case is not None:
        if template_profile_case.sourceTextSha256 != sha256_bytes(source_text.encode("utf-8")):
            raise ValueError("template mutation profile source text SHA-256 differs")
        source_lines = source_text.splitlines()
        current_lines = current_text.splitlines()
        if len(source_lines) != len(current_lines):
            raise ValueError("template mutation profile requires stable source/current topology")
        for profile_line in template_profile_case.lines:
            number = line_number(profile_line.lineId)
            if not 1 <= number <= len(source_lines):
                raise ValueError(
                    "template mutation profile line is outside the source OCR: "
                    f"{profile_line.lineId}"
                )
            source_line = source_lines[number - 1]
            if sha256_bytes(source_line.encode("utf-8")) != profile_line.sourceLineSha256:
                raise ValueError(
                    f"template mutation profile line SHA-256 differs: {profile_line.lineId}"
                )
            if not source_line.strip() or _PAGE_MARKER.fullmatch(source_line) is not None:
                raise ValueError(
                    "template mutation profile cannot own a blank/page-marker line: "
                    f"{profile_line.lineId}"
                )
            add(
                category="template_profile_residual",
                source_surface=source_line.strip(),
                lines={number},
                paths=(),
                target={
                    "policy": "reevaluate_exact_line_against_complete_synthetic_target",
                    "findingKinds": list(profile_line.findingKinds),
                    "sourceLineId": profile_line.lineId,
                    "allowUnchangedOnlyWhenAlreadyTargetCoherent": True,
                },
                disposition="model_residual",
                rationale=(
                    "Independent whole-document review previously found this exact source-"
                    "template line capable of retaining stale, private, repeated, derived, or "
                    "contradictory shipment semantics. Reconcile it against the complete current "
                    "target; preserve it only when it is already target-coherent."
                ),
            )

    # Present the inventory in document order.  Category-first ordering makes a later auxiliary
    # copy appear before the labeled occurrence that explains it, increasing provider context
    # switching and making audit receipts harder to read.  The remaining fields provide a stable
    # tie-break without making candidate identifiers dependent on dictionary insertion order.
    ordered_payloads = sorted(
        raw_candidates.values(),
        key=lambda payload: (
            min(line_number(value) for value in payload["lineIds"]),
            payload["category"],
            payload["sourceSurface"],
            tuple(payload["targetPaths"]),
        ),
    )
    return tuple(
        InventoryCandidate(candidateId=f"I{ordinal:04d}", **payload)
        for ordinal, payload in enumerate(ordered_payloads, start=1)
    )


def apply_deterministic_auxiliary_edits(
    *,
    text: str,
    document_id: str,
    scenario_id: str,
    candidates: Sequence[InventoryCandidate],
) -> tuple[str, tuple[DeterministicInventoryEdit, ...]]:
    """Apply only opaque, shape-preserving auxiliary substitutions locally."""

    lines = text.splitlines(keepends=True)
    authorized_line_ids: dict[str, set[str]] = defaultdict(set)
    for candidate in candidates:
        if candidate.disposition != "deterministic_shape_replacement":
            continue
        authorized_line_ids[candidate.sourceSurface].update(candidate.lineIds)
    sources = tuple(sorted(authorized_line_ids, key=lambda value: (len(value), value)))
    replacements: dict[str, str] = {}
    for source in sources:
        pattern = surface_pattern(source)
        if not any(token in pattern for token in ("A", "a", "9")):
            raise ValueError(f"auxiliary source surface has no replaceable characters: {source!r}")
        stream = DeterministicStream(
            seed=0,
            namespace="raw-text-template-auxiliary-v1",
            identity=f"{document_id}:{scenario_id}:{_semantic_surface(source)}",
        )
        generated = generate_from_surface_pattern(
            pattern=pattern,
            stream=stream,
            excluded=frozenset(replacements.values()),
            additional_excluded=source,
        )
        # A longer formal identifier can embed a shorter identifier belonging to the same source
        # role (for example, an importer tax number can prefix an ACID value). Generate shorter
        # values first, then preserve every observed substring relationship in the longer value.
        occupied: dict[int, str] = {}
        for child_source, child_target in sorted(
            replacements.items(), key=lambda row: (-len(row[0]), row[0])
        ):
            offset = source.find(child_source)
            while offset >= 0:
                for relative, character in enumerate(child_target):
                    position = offset + relative
                    previous = occupied.get(position)
                    if previous is not None and previous != character:
                        raise ValueError(
                            "overlapping auxiliary identifier relationships are inconsistent"
                        )
                    occupied[position] = character
                offset = source.find(child_source, offset + 1)
        if occupied:
            rendered = list(generated)
            for position, character in occupied.items():
                rendered[position] = character
            generated = "".join(rendered)
        if generated == source or generated in replacements.values():
            raise RuntimeError("linked auxiliary generation produced a forbidden collision")
        replacements[source] = generated

    edits: list[DeterministicInventoryEdit] = []
    original_lines = tuple(line.rstrip("\r\n") for line in lines)
    touched_by_source: dict[str, tuple[str, ...]] = {}
    for source in replacements:
        authorized = tuple(sorted(authorized_line_ids[source], key=line_number))
        for authorized_line_id in authorized:
            index = line_number(authorized_line_id) - 1
            if not 0 <= index < len(original_lines):
                raise ValueError(
                    f"deterministic auxiliary line is outside the OCR: {authorized_line_id}"
                )
            if source not in original_lines[index]:
                raise ValueError(
                    "deterministic auxiliary source is absent from its authorized line: "
                    f"{source!r} on {authorized_line_id}"
                )
        touched_by_source[source] = authorized
    for source, target in sorted(replacements.items(), key=lambda row: (-len(row[0]), row[0])):
        for authorized_line_id in touched_by_source[source]:
            index = line_number(authorized_line_id) - 1
            raw_line = lines[index]
            body = raw_line.rstrip("\r\n")
            ending = raw_line[len(body) :]
            if source not in body:
                # A longer authorized identifier may already have replaced an embedded copy of
                # this shorter source on the same line. The generated longer value preserves the
                # source relationship by construction, so there is no second write to perform.
                continue
            rewritten = body.replace(source, target)
            if len(rewritten) != len(body):
                raise ValueError("shape-preserving auxiliary edit changed line length")
            lines[index] = rewritten + ending
        touched = touched_by_source[source]
        if not touched:
            raise ValueError(
                f"deterministic auxiliary source was not present before apply: {source!r}"
            )
        edits.append(
            DeterministicInventoryEdit(
                sourceSurface=source,
                targetSurface=target,
                lineIds=tuple(touched),
                method="hmac_shape_preserving_auxiliary_v1",
            )
        )
    output = "".join(lines)
    if len(output.splitlines()) != len(text.splitlines()):
        raise ValueError("deterministic auxiliary edits changed line count")
    return output, tuple(edits)


def _finding_lines(text: str, surface: str) -> tuple[str, ...]:
    return tuple(line_id(value) for value in sorted(_surface_lines(text, surface)))


def audit_full_document(
    *,
    document_id: str,
    source_text: str,
    output_text: str,
    source_label: Mapping[str, Any],
    target_label: Mapping[str, Any],
    inventory: Sequence[InventoryCandidate],
    deterministic_edits: Sequence[DeterministicInventoryEdit],
    oracle_case: RegressionCase | None,
    surface_requirements: Sequence[SurfaceRenderingRequirement] = (),
    anchored_replacements: Sequence[AnchoredScalarReplacementRequirement] = (),
    deterministic_prefills: Sequence[AppliedDeterministicPrefill] = (),
    work_items: Sequence[HybridWorkItem] = (),
) -> FullDocumentAudit:
    """Evaluate source-to-output completeness independently of the model contract."""

    findings: list[FullDocumentFinding] = []
    deterministic_sources = {row.sourceSurface for row in deterministic_edits}
    target_scalars = {
        _semantic_surface(str(value))
        for value in _flatten_scalars(target_label)
        if isinstance(value, (str, int, float)) and not isinstance(value, bool)
    }
    target_string_surfaces = _target_string_surfaces(target_label)
    source_lines = source_text.splitlines()
    output_lines = output_text.splitlines()
    if len(source_lines) != len(output_lines):
        raise ValueError("full-document audit requires stable OCR line topology")
    ordinal = 0

    def add(
        category: str,
        surface: str,
        *,
        paths: Sequence[str] = (),
        line_numbers: Sequence[int] | None = None,
        explanation: str,
    ) -> None:
        nonlocal ordinal
        ordinal += 1
        findings.append(
            FullDocumentFinding(
                findingId=f"F{ordinal:04d}",
                category=cast(Any, category),
                lineIds=(
                    tuple(line_id(value) for value in line_numbers)
                    if line_numbers is not None
                    else _finding_lines(output_text, surface)
                ),
                sourceSurface=surface,
                targetPaths=tuple(paths),
                explanation=explanation,
            )
        )

    grouped: dict[tuple[str, str], list[ChangedLeaf]] = defaultdict(list)
    for leaf in changed_leaves(source_label, target_label):
        if (
            leaf.requiresTextEdit
            and leaf.sourceValue != leaf.targetValue
            and not _is_projected_legacy_equipment_leaf(leaf, target_label)
            and isinstance(leaf.sourceValue, (str, int, float))
            and not isinstance(leaf.sourceValue, bool)
        ):
            grouped[_source_value_group_key(leaf.sourceValue)].append(leaf)
    for _surface_key, leaves in sorted(grouped.items()):
        surface = min(
            (str(row.sourceValue) for row in leaves),
            key=lambda value: (len(value), value.casefold(), value),
        )
        owned_lines = {
            line_number(line)
            for item in work_items
            if set(item.targetPaths) & {leaf.path for leaf in leaves}
            for line in item.evidenceLineIds
        }
        occurrence_lines = (
            _numeric_lines(output_text, cast(int | float, leaves[0].sourceValue)) & owned_lines
            if all(isinstance(row.sourceValue, (int, float)) for row in leaves)
            else _surface_lines(output_text, surface)
        )
        if all(
            leaf.path.startswith("documentPatch.parties.")
            and leaf.path.rsplit(".", 1)[-1] in {"name", "address"}
            for leaf in leaves
        ):
            heading_lines = {
                number
                for number in occurrence_lines
                if _PARTY_HEADING_LINE.fullmatch(source_lines[number - 1]) is not None
                or _PRESERVABLE_LEGAL_BOILERPLATE_LINE.fullmatch(source_lines[number - 1])
                is not None
            }
            occurrence_lines.difference_update(heading_lines)
        if all(leaf.path == "documentPatch.freight.paymentArrangement" for leaf in leaves):
            # Unowned occurrences describe headers or charge-level terms, not this label path.
            # Requiring their mutation would make a synthetic B/L less faithful to its template.
            occurrence_lines.intersection_update(owned_lines)
        rendered_changed_target_lines = {
            number
            for number in occurrence_lines & owned_lines
            if any(
                isinstance(leaf.targetValue, (str, int, float))
                and not isinstance(leaf.targetValue, bool)
                and _contains_surface(output_lines[number - 1], str(leaf.targetValue))
                for leaf in leaves
            )
        }
        stale_lines = (occurrence_lines & owned_lines) - rendered_changed_target_lines
        unowned_lines = occurrence_lines - owned_lines
        changed_paths = {leaf.path for leaf in leaves}
        shared_target_owned_lines = {
            line_number(line)
            for item in work_items
            if not (set(item.targetPaths) & changed_paths)
            and isinstance(item.targetValue, (str, int, float))
            and not isinstance(item.targetValue, bool)
            and _semantic_surface(str(item.targetValue)) == _semantic_surface(surface)
            for line in item.evidenceLineIds
            if 1 <= line_number(line) <= len(output_lines)
            and _contains_surface(output_lines[line_number(line) - 1], str(item.targetValue))
        }
        shared_target_owned_lines.update(
            line_number(prefill.lineId)
            for prefill in deterministic_prefills
            if not (set(prefill.targetPaths) & changed_paths)
            and _semantic_surface(prefill.targetSurface) == _semantic_surface(surface)
            and 1 <= line_number(prefill.lineId) <= len(output_lines)
            and _contains_surface(
                output_lines[line_number(prefill.lineId) - 1], prefill.targetSurface
            )
        )
        # The same printed surface can be stale for one path and required by another.  Only the
        # exact target path's proven evidence lines are licensed; a global target-value match
        # would let a stale occurrence in the changed role hide behind an unrelated field.
        stale_lines.difference_update(shared_target_owned_lines)
        if _semantic_surface(surface) in target_scalars:
            # An exact shared scalar may legitimately remain outside the changed field's proven
            # evidence lines, but it never licenses the changed path itself.
            pass
        elif _surface_required_by_target(surface, target_string_surfaces):
            licensed_lines = _target_licensed_occurrence_lines(
                output_text,
                surface,
                target_string_surfaces,
            )
            stale_lines.update(number for number in unowned_lines if number not in licensed_lines)
        else:
            stale_lines.update(unowned_lines)
        if stale_lines:
            add(
                "retained_changed_source_value",
                surface,
                paths=tuple(row.path for row in leaves),
                line_numbers=tuple(sorted(stale_lines)),
                explanation=(
                    "A source semantic value that is absent from the complete target still "
                    "occurs in the rewritten OCR."
                ),
            )

    # A target party scalar must occur in the same role-owned block as its source.  Global
    # occurrence checks can be fooled by repeated legal suffixes (for example ``S.A.C.`` in an
    # unrelated signing affiliate), while the labeled shipper itself is missing punctuation.
    checked_party_paths: set[str] = set()
    for leaf in changed_leaves(source_label, target_label):
        if (
            not leaf.requiresTextEdit
            or leaf.sourceValue == leaf.targetValue
            or not isinstance(leaf.targetValue, str)
            or not leaf.path.startswith("documentPatch.parties.")
            or leaf.path in checked_party_paths
        ):
            continue
        checked_party_paths.add(leaf.path)
        path_anchors = tuple(
            requirement
            for requirement in anchored_replacements
            if leaf.path in requirement.targetPaths
            and requirement.targetSurface == leaf.targetValue
        )
        if path_anchors and all(
            all(
                1 <= line_number(line) <= len(output_lines)
                and _contains_surface(
                    output_lines[line_number(line) - 1], requirement.targetSurface
                )
                for line in requirement.sourceLineIds
            )
            for requirement in path_anchors
        ):
            # Some reviewed party contacts are printed in a detached operational section rather
            # than inside the visual party block.  The path-bound anchor is stronger evidence
            # than inventing a duplicate copy inside that block, and its deterministic rewrite is
            # already checked line by line by the core contract.
            continue
        owned_lines = {
            line_number(line)
            for item in work_items
            if leaf.path in item.targetPaths
            for line in item.evidenceLineIds
        }
        # A long address/name may be reflowed across neighboring lines inside the same visual
        # role block. Exact work-item evidence identifies the role but is not the complete
        # rendering envelope, so audit the union rather than incorrectly requiring one scalar on
        # its original single line.
        owned_lines.update(
            _party_block_line_numbers(
                source_text,
                path=leaf.path,
                source_label=cast(Mapping[str, JsonValue], source_label),
            )
        )
        rendered_block = "\n".join(output_lines[number - 1] for number in sorted(owned_lines))
        contains_target = (
            _contains_rendered_surface(rendered_block, leaf.targetValue)
            if leaf.path.endswith(".name")
            else _contains_ordered_party_address(rendered_block, leaf.targetValue)
            if leaf.path.endswith(".address")
            else _contains_surface(rendered_block, leaf.targetValue)
        )
        if owned_lines and contains_target:
            continue
        add(
            "semantic_slot_mismatch",
            leaf.targetValue,
            paths=(leaf.path,),
            line_numbers=tuple(sorted(owned_lines)),
            explanation=(
                "The exact target party scalar is absent from its source-role block; an "
                "occurrence in another party or signing block cannot ground this field."
            ),
        )

    for auxiliary in locate_auxiliary_values(source_text):
        if auxiliary.value in deterministic_sources:
            if _contains_surface(output_text, auxiliary.value):
                add(
                    "retained_source_auxiliary",
                    auxiliary.value,
                    explanation="A deterministically scheduled auxiliary value survived.",
                )
            continue
        if _contains_surface(output_text, auxiliary.value):
            add(
                "retained_source_auxiliary",
                auxiliary.value,
                explanation=(
                    "An explicit source-only identifier/contact remained unchanged in output."
                ),
            )

    for number, (source_line, output_line) in enumerate(
        zip(source_lines, output_lines, strict=True), start=1
    ):
        if not any(character.isalnum() for character in source_line):
            continue
        if any(character.isalnum() for character in output_line):
            continue
        ordinal += 1
        findings.append(
            FullDocumentFinding(
                findingId=f"F{ordinal:04d}",
                category="lexical_line_degenerated",
                lineIds=(line_id(number),),
                sourceSurface=source_line.strip(),
                targetPaths=(),
                explanation=(
                    "A lexical source line degenerated into punctuation-only content instead "
                    "of a grammatical template-preserving realization."
                ),
            )
        )

    for number, (source_line, output_line) in enumerate(
        zip(source_lines, output_lines, strict=True), start=1
    ):
        for relation in _AGENCY_RELATION.finditer(source_line):
            marker = relation.group(0)
            if _contains_surface(output_line, marker):
                continue
            ordinal += 1
            findings.append(
                FullDocumentFinding(
                    findingId=f"F{ordinal:04d}",
                    category="legal_relation_topology_mismatch",
                    lineIds=(line_id(number),),
                    sourceSurface=marker,
                    targetPaths=(
                        "documentPatch.parties.carrier",
                        "documentPatch.parties.shipper",
                    ),
                    explanation=(
                        "A legal agency/trading relationship marker moved or disappeared; "
                        "only its linked identities may change."
                    ),
                )
            )

    for requirement in surface_requirements:
        if requirement.kind not in {
            "date",
            "date_global",
            "carrier_header_identity",
            "carrier_principal_identity",
        } or (requirement.sourceSurface == requirement.targetSurface):
            continue
        owned_lines = _context_bound_surface_lines(source_text, requirement)
        if not owned_lines:
            if requirement.kind in {"date", "date_global"}:
                # When issue and shipped-on-board dates have the same source value, the contract
                # intentionally contains the observed print formats under both semantic paths.
                # A particular format can belong only to the other path (for example, the
                # hyphenated copy appears solely after ``Shipped on Board``). The global atomic
                # requirement still requires the target surface and removes the stale source;
                # absence of a path-owned occurrence is therefore not itself a mismatch.
                continue
            add(
                "semantic_slot_mismatch",
                requirement.sourceSurface,
                paths=(requirement.targetPath,),
                explanation=("The source rendered surface has no unambiguous path-owned location."),
            )
            continue
        mismatched = tuple(
            number
            for number in sorted(owned_lines)
            if not _contains_rendered_surface(output_lines[number - 1], requirement.targetSurface)
            or (
                requirement.sourceSurface not in requirement.targetSurface
                and _contains_rendered_surface(output_lines[number - 1], requirement.sourceSurface)
            )
        )
        if mismatched:
            ordinal += 1
            findings.append(
                FullDocumentFinding(
                    findingId=f"F{ordinal:04d}",
                    category="semantic_slot_mismatch",
                    lineIds=tuple(line_id(number) for number in mismatched),
                    sourceSurface=requirement.sourceSurface,
                    targetPaths=(requirement.targetPath,),
                    explanation=(
                        "A path-owned source line does not contain its exact target-rendered "
                        "surface. Presence elsewhere in the document is insufficient."
                    ),
                )
            )

    for item in work_items:
        if item.action not in {"replace_equipment_surface", "add_equipment_surface"}:
            continue
        deterministic_surfaces = tuple(
            requirement
            for requirement in surface_requirements
            if requirement.kind
            in {
                "aggregate_equipment_breakdown",
                "carrier_receipt_equipment_breakdown",
            }
            and requirement.targetPath in item.targetPaths
        )
        if deterministic_surfaces and all(
            requirement.targetSurface in output_text for requirement in deterministic_surfaces
        ):
            # A mixed anonymous summary can represent several indexed containers on one line.
            # Each path is independently tied to the same host-derived exact surface, so an
            # empty per-container evidence set is expected rather than a missing realization.
            continue
        target = item.targetValue
        if not isinstance(target, Mapping):
            continue
        expected_size = target.get("sizeCategory")
        expected_type = target.get("typeCategory")
        if not isinstance(expected_size, str) or not isinstance(expected_type, str):
            continue
        anchored_equipment_lines = tuple(
            sorted(
                {
                    line_number(line_id)
                    for requirement in anchored_replacements
                    if any(path in requirement.targetPaths for path in item.targetPaths)
                    for line_id in requirement.sourceLineIds
                    if 1 <= line_number(line_id) <= len(output_lines)
                }
            )
        )
        if anchored_equipment_lines and all(
            (
                (
                    reviewed := review_source_equipment_surface(
                        output_lines[number - 1],
                        temperature_present=expected_type.startswith("REFRIGERATED"),
                    )
                ).resolution
                == "reviewed_source_grammar"
                and reviewed.size_category == expected_size
                and reviewed.type_category == expected_type
            )
            for number in anchored_equipment_lines
        ):
            # Compiler-owned equipment spans intentionally have no agent evidence lines. Verify
            # their final printed semantics directly here rather than interpreting that empty
            # residual set as a missing realization. The general anchored audit separately
            # proves that every contracted source occurrence disappeared.
            continue
        evidence_lines = tuple(
            sorted(
                number
                for number in (line_number(value) for value in item.evidenceLineIds)
                if 1 <= number <= len(output_lines)
            )
        )
        matched = False
        for number in evidence_lines:
            reviewed = review_source_equipment_surface(
                output_lines[number - 1],
                temperature_present=expected_type.startswith("REFRIGERATED"),
            )
            if (
                reviewed.resolution == "reviewed_source_grammar"
                and reviewed.size_category == expected_size
                and reviewed.type_category == expected_type
            ):
                matched = True
                break
        if matched:
            continue
        add(
            "semantic_slot_mismatch",
            f"{expected_size}/{expected_type}",
            paths=item.targetPaths,
            line_numbers=evidence_lines,
            explanation=(
                "The path-owned equipment line does not resolve to the target size and type "
                "semantics. A container identifier alone cannot ground equipment categories."
            ),
        )

    if _label_has_temperature(source_label) and not _label_has_temperature(target_label):
        for number, (source_line, output_line) in enumerate(
            zip(source_text.splitlines(), output_text.splitlines(), strict=True), start=1
        ):
            if (
                _REEFER_ASSERTION.search(source_line) is not None
                and _REEFER_ASSERTION.search(output_line) is not None
            ):
                ordinal += 1
                findings.append(
                    FullDocumentFinding(
                        findingId=f"F{ordinal:04d}",
                        category="reefer_state_contradiction",
                        lineIds=(line_id(number),),
                        sourceSurface=output_line.strip(),
                        targetPaths=("documentPatch.containers[].temperatureSetpoint",),
                        explanation=(
                            "Output retains a positive refrigerated-operation assertion although "
                            "the target shipment is not temperature-controlled."
                        ),
                    )
                )

    if oracle_case is not None:
        for assertion in oracle_case.assertions:
            for surface in assertion.sourceSurfacesMustDisappear:
                if not _oracle_surface_applies(surface, target_label=target_label):
                    continue
                if _contains_surface(output_text, surface):
                    add(
                        "regression_oracle_failure",
                        surface,
                        explanation=f"{assertion.assertionId}: {assertion.instruction}",
                    )

    for candidate in inventory:
        if candidate.disposition == "deterministic_shape_replacement":
            handled = candidate.sourceSurface in deterministic_sources
        else:
            candidate_output = "\n".join(
                output_lines[number - 1]
                for number in (line_number(value) for value in candidate.lineIds)
                if 1 <= number <= len(output_lines)
            )
            handled = not _contains_surface(candidate_output, candidate.sourceSurface)
            if candidate.category == "template_profile_residual":
                # A profile grants reviewed write/attention authority; it is not an assertion
                # that this line must change for every descendant. A later target may agree with
                # the source. This stage never publishes; independent whole-document
                # certification owns semantic acceptance.
                handled = True
            if not handled and candidate.category == "changed_source_occurrence":
                target_rows = (
                    candidate.targetSemantics if isinstance(candidate.targetSemantics, list) else []
                )
                targets: list[str | int | float] = []
                for row in target_rows:
                    target = row.get("target") if isinstance(row, Mapping) else None
                    if (
                        isinstance(target, (str, int, float))
                        and not isinstance(target, bool)
                        and target not in targets
                    ):
                        targets.append(target)
                # A source numeral can legitimately remain on a shared line under a different
                # semantic role (for example source package count ``20`` becomes target equipment
                # size ``20'``).  The path-specific delta is handled when every non-null target
                # semantic is rendered in the candidate-owned lines; other validators continue
                # to police the remaining source surface under its own role.
                handled = bool(targets) and all(
                    bool(_numeric_lines(candidate_output, target))
                    if isinstance(target, (int, float)) and not isinstance(target, bool)
                    else _contains_surface(candidate_output, str(target))
                    for target in targets
                )
        if not handled and not any(
            finding.sourceSurface == candidate.sourceSurface for finding in findings
        ):
            add(
                "inventory_candidate_unhandled",
                candidate.sourceSurface,
                paths=candidate.targetPaths,
                line_numbers=tuple(
                    number
                    for number in (line_number(value) for value in candidate.lineIds)
                    if 1 <= number <= len(output_lines)
                    and _contains_surface(output_lines[number - 1], candidate.sourceSurface)
                ),
                explanation="A mutable inventory candidate has no verified disposition.",
            )

    inventory_sha = sha256_bytes(
        canonical_json_bytes([row.model_dump(mode="json") for row in inventory])
    )
    return FullDocumentAudit(
        documentId=document_id,
        sourceTextSha256=sha256_bytes(source_text.encode("utf-8")),
        outputTextSha256=sha256_bytes(output_text.encode("utf-8")),
        inventorySha256=inventory_sha,
        candidates=len(inventory),
        deterministicCandidates=sum(
            row.disposition == "deterministic_shape_replacement" for row in inventory
        ),
        residualCandidates=sum(row.disposition == "model_residual" for row in inventory),
        findings=tuple(findings),
        passed=not findings,
    )


def parse_regression_oracle(value: Mapping[str, Any]) -> RegressionOracle:
    payload = dict(value)
    raw_cases = payload.get("cases")
    if not isinstance(raw_cases, list):
        raise ValueError("regression oracle cases must be a list")
    cases: list[RegressionCase] = []
    for raw_case in raw_cases:
        if not isinstance(raw_case, Mapping):
            raise ValueError("regression oracle case must be an object")
        case_payload = dict(raw_case)
        raw_assertions = case_payload.get("assertions")
        if not isinstance(raw_assertions, list):
            raise ValueError("regression oracle assertions must be a list")
        case_payload["assertions"] = tuple(
            RegressionAssertion.from_mapping(cast(Mapping[str, Any], row)) for row in raw_assertions
        )
        cases.append(RegressionCase.model_validate(case_payload, strict=True))
    payload["cases"] = tuple(cases)
    return RegressionOracle.model_validate(payload, strict=True)


def parse_template_mutation_profile(value: Mapping[str, Any]) -> TemplateMutationProfile:
    """Parse a strict JSON profile and reject ambiguous duplicate ownership."""

    profile = TemplateMutationProfile.model_validate_json(canonical_json_bytes(value), strict=True)
    document_ids = tuple(row.documentId for row in profile.cases)
    if len(document_ids) != len(set(document_ids)):
        raise ValueError("template mutation profile repeats a document ID")
    for case in profile.cases:
        line_ids = tuple(row.lineId for row in case.lines)
        if len(line_ids) != len(set(line_ids)):
            raise ValueError(f"template mutation profile repeats a line ID: {case.documentId}")
        if line_ids != tuple(sorted(line_ids, key=line_number)):
            raise ValueError(
                f"template mutation profile lines are not in document order: {case.documentId}"
            )
    return profile
