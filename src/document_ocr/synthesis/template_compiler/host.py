from __future__ import annotations

import bisect
import json
import re
import unicodedata
from collections import Counter, defaultdict
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass, replace
from datetime import date, datetime
from decimal import Decimal, InvalidOperation
from functools import lru_cache
from itertools import combinations, pairwise
from pathlib import Path
from typing import Any, Literal, cast

from document_ocr.date_evidence import date_immediately_under_declared_value
from document_ocr.hashing import canonical_json_bytes, sha256_bytes, sha256_file
from document_ocr.synthesis.generators import surface_pattern
from document_ocr.synthesis.raw_text_template import (
    TemplateSlot,
    build_template_slot,
    compile_raw_text_template,
    render_compiled_template,
    sentinel_bindings,
)

from .coherence import (
    InclusiveRangeSurface,
    inclusive_range_surfaces,
    validate_coherence_contracts,
    whole_inclusive_range_surface,
)
from .generation_contract import (
    party_owned_surfaces,
    validate_party_evidence,
    validate_seal_realization,
)
from .models import (
    AgentBindingProposal,
    AgentOccurrence,
    AnchorOverride,
    BindingRealization,
    CapabilityContract,
    CarrierAssessment,
    CarrierBinding,
    CertifiedSemanticTemplate,
    CoherenceConstraint,
    CriticAgentOutput,
    CriticFinding,
    CriticOccurrenceRemoval,
    LiteralCertification,
    RiskCandidate,
    SemanticBinding,
    SemanticOnlyTargetFact,
    SemanticOnlyTargetFactProposal,
    SlotRealization,
    SourceBindingRelationship,
    TargetValueSnapshot,
    TemplateCertification,
)
from .semantic_plan import build_auxiliary_semantic_plan

_PAGE_HEADER = re.compile(r"(?m)^--- PAGE ([1-9][0-9]*) ---$")
_EMAIL = re.compile(r"(?i)(?<![\w.+-])[\w.+-]+@[a-z0-9-]+(?:\.[a-z0-9-]+)+")
_URL = re.compile(
    r"(?i)(?:https?://|www\.)[^\s<>]+|(?<![@\w])(?:[a-z0-9-]+\.)+(?:com|net|org|biz|io|co|cn|de|dk|fr|it|jp|kr|nl|no|se|sg|uk)(?:/[^\s<>]*)?"
)
_EQUIPMENT = re.compile(r"(?<![A-Z0-9])[A-Z]{3}[UJZ][ -]?[0-9]{6,7}(?![A-Z0-9])")
_DATE = re.compile(
    r"(?i)(?<!\w)(?:[12][0-9]{3}[-/.][01]?[0-9][-/.][0-3]?[0-9]|"
    r"[0-3]?[0-9][-/\.][01]?[0-9][-/\.](?:[12][0-9]{3}|[0-9]{2})|"
    r"[0-3]?[0-9][ -](?:jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)[a-z]*"
    r"[ -](?:[12][0-9]{3}|[0-9]{2}))"
    r"(?!\w)"
)
_FMC_CARRIER_IDENTIFIER_PREFIX = re.compile(r"(?i)(?:^|\b)FMC\s*(?:NO\.?|NUMBER|#)?\s*$")
_MEASUREMENT = re.compile(
    r"(?i)(?<!\w)[-+]?[0-9][0-9,.]*\s*(?:kg|kgs|kilograms?|mt|tons?|tonnes?|lb|lbs|"
    r"cbm|m3|m\^3|cubic\s+met(?:er|re)s?|°?c|deg\.?\s*c)(?!\w)"
)
_PHONE = re.compile(r"(?<!\w)(?:\+?[0-9][0-9 ()/.-]{6,}[0-9])(?!\w)")
_SELECTED_OPERATIONAL_TEXT = re.compile(r"(?i)\bSHIPPED\s+ON\s+BOARD\b")
_SELECTED_TYPE_VALUE = re.compile(
    r"(?i)\bTYPE\s*:\s*(?P<value>[A-Z][A-Z0-9]*(?:[ \t]+[A-Z0-9][A-Z0-9./_-]*)*)[ \t]*$"
)
_CARE_OF_SEPARATOR = re.compile(r"(?i)(?<![A-Za-z])C\s*/\s*O(?![A-Za-z])")
_ORIGINAL_BILL_COUNT_CAPTION = re.compile(
    r"(?i)^\s*(?:NUMBER|NO\.?)\s+OF\s+ORIGINAL\s+"
    r"(?:BILLS?\s+OF\s+LADING|FBL'?S?)\s*$"
)
_STANDALONE_ORIGINAL = re.compile(r"^\s*ORIGINAL\s*$", re.IGNORECASE)
_FORM_TITLE_WITHOUT_SELECTED_NEGOTIABILITY = re.compile(
    r"(?i)^\s*(?:(?:ORIGINAL|COPY|DUPLICATE)\s+)?"
    r"(?:SEA\s+WAYBILL|BILL\s+OF\s+LADING|"
    r"MULTIMODAL\s+TRANSPORT\s+BILL\s+OF\s+LADING)"
    r"(?:\s*\((?:CONTINUED|COPY|ORIGINAL)\))?\s*$"
)
_DOCUMENT_FORM_NOUN = re.compile(
    r"(?i)^\s*(?:SEA\s+WAYBILL|BILL\s+OF\s+LADING|"
    r"MULTIMODAL\s+TRANSPORT\s+BILL\s+OF\s+LADING)\s*$"
)
_EXPLICIT_NEGOTIABILITY_SELECTION = re.compile(
    r"(?i)\b(?:NON[ -]?NEGOTIABLE|NEGOTIABLE|TO\s+ORDER|STRAIGHT\s+BILL|"
    r"NUMBER\s+OF\s+ORIGINAL|NO\.?\s+OF\s+ORIGINAL)\b"
)
_TOKEN = re.compile(r"(?<![A-Za-z0-9])[A-Za-z0-9][A-Za-z0-9./_-]{5,}(?![A-Za-z0-9])")
_LONG_NUMBER = re.compile(r"(?<![0-9])[0-9]{5,}(?![0-9])")
_PATH_TOKEN = re.compile(r"([^.\[\]]+)|\[([0-9]+)\]")
_TARGET_PATH = re.compile(
    r"^[A-Za-z_][A-Za-z0-9_]*(?:\[[0-9]+\])?"
    r"(?:\.[A-Za-z_][A-Za-z0-9_]*(?:\[[0-9]+\])?)*$"
)
_CONTAINER_NUMBER_PATH = re.compile(r"^documentPatch\.containers\[([0-9]+)\]\.containerNumber$")
_CONTAINER_TYPE_PATH = re.compile(r"^documentPatch\.containers\[([0-9]+)\]\.typeDescription$")
_CONTAINER_OBJECT_PATH = re.compile(r"^documentPatch\.containers\[([0-9]+)\]$")
_PARTY_ADDRESS_PATH = re.compile(
    r"^documentPatch\.parties\.(?:[A-Za-z_][A-Za-z0-9_]*|notifyParties\[[0-9]+\])\.address$"
)
_CARRIER_NAME_PATH = "documentPatch.parties.carrier.name"
_NEGOTIABILITY_PATH = "documentPatch.negotiability"
_FREIGHT_PAYMENT_PATH = "documentPatch.freight.paymentArrangement"
_PORT_OF_LOADING_PATH = "documentPatch.route.portOfLoading.name"
_SHIPPED_ON_BOARD_DATE_PATH = "documentPatch.shippedOnBoardDate"
_EXTRACTION_DATE_FIELDS = ("issueDate", "shippedOnBoardDate")
_EXTRACTION_DATE_PATHS = frozenset(
    f"documentPatch.{field}" for field in _EXTRACTION_DATE_FIELDS
)
_VESSEL_NAME_PATH = "documentPatch.transport.vesselName"
_VOYAGE_NUMBER_PATH = "documentPatch.transport.voyageNumber"
_LOADING_TERMINAL_CAPTION = "loadingpierterminal"
_SOURCE_ONLY_FIELD_CAPTIONS: dict[str, tuple[str, str, str, str]] = {
    "foreignexportercountry": (
        "agent:customs:foreign_exporter_country",
        "customs",
        "customs:foreign_exporter",
        "location",
    ),
    "freightpayableat": (
        "agent:commercial:freight_payable_location",
        "commercial",
        "commercial:freight",
        "commercial_text",
    ),
    "datecargoreceived": (
        "agent:transport:cargo_received_date",
        "transport",
        "transport:cargo_receipt",
        "date",
    ),
}
_SHIPPER_NAME_ADDRESS_CAPTION = "shippernameaddressphone"
_PLACE_OF_RECEIPT_PATH = "documentPatch.route.placeOfReceipt.name"
_SHIPPER_ADDRESS_PATH = "documentPatch.parties.shipper.address"
_SHIPPER_CITY_PATH = "documentPatch.parties.shipper.city"
_SHIPPER_NAME_PATH = "documentPatch.parties.shipper.name"
_BILL_OF_LADING_PATH = "documentPatch.billOfLadingNumber"
_PARTY_LOCATION_PATH = re.compile(
    r"^documentPatch\.parties\.(?:[A-Za-z_][A-Za-z0-9_]*|notifyParties\[[0-9]+\])\."
    r"(?:city|country)$"
)
_PARTY_BLOCK_RECOVERABLE_PATH = re.compile(
    r"^documentPatch\.parties\.(?:[A-Za-z_][A-Za-z0-9_]*|notifyParties\[[0-9]+\])\."
    r"(?:address|city|country)$"
)
_PARTY_CITY_COUNTRY_VARIANT = re.compile(r"^\s*(?P<value>[A-Z][A-Z .'-]{1,50},[ \t]*[A-Z]{2})\s*$")
_LABELED_AUXILIARY_ORGANIZATION = re.compile(
    r"(?i)^(?:EGYPTIAN\s+)?IMPORTER\s+NAME\s*:\s*(?P<value>\S(?:.*\S)?)\s*$"
)
_LABELED_COUNTRY_VALUE = re.compile(
    r"(?i)^\s*(?:[A-Z][A-Z ]{1,60}\s+)?COUNTRY\s*:\s*(?P<value>[A-Z][A-Z .'-]*[A-Z])\s*$"
)
_LABELED_COUNTRY_CODE_VALUE = re.compile(
    r"(?i)^\s*(?:[A-Z][A-Z ]{1,60}\s+)?COUNTRY\s+CODE\s*:\s*(?P<value>[A-Z]{2})\s*$"
)
_EXPORTED_FROM_COUNTRY_CONTEXT = re.compile(
    r"(?i)\b(?:THESE\s+)?COMMODITIES\s+WERE\s+EXPORTED\s+FROM\s+THE\s+$"
)
_SELECTED_MEASUREMENT_UNIT = re.compile(r"(?i)^(?:KGM|KGS|KG)$")
_PROFORMA_INVOICE_DATE = re.compile(
    r"(?i)\bP\s*/\s*I\s+NO\..*?\bDD\.\s*(?P<value>[0-3]?[0-9]/[01]?[0-9]/[0-9]{3,4})(?![0-9])"
)
_LADEN_ON_BOARD_DATE_CAPTION = re.compile(r"(?i)^\s*DATE\s+LADEN\s+ON\s+BOARD(?:\s+[O*])?\s*$")
_FORWARDING_REFERENCE_PATH = re.compile(r"^documentPatch\.forwardingAndExportReferences\[[0-9]+\]$")
_PORT_CHARGE_SELECTION = re.compile(
    r"(?i)^\s*(?P<place>ORIGIN|DESTINATION)\s+PORT\s+CHARGE\s+"
    r"(?P<payment>PREPAID|COLLECT)\s*$"
)
_COMBINED_BOOKING_CAPTION = re.compile(r"(?i)^\s*BOOKING\s+NO\.\s+SEA\s+WAYBILL\s+NO\.")
_BILL_OF_LADING_NUMBER_LINE = re.compile(
    r"(?i)^\s*B\s*/\s*L\s+NO\.?\s*:?\s*(?P<value>[A-Z0-9][A-Z0-9./_-]{5,})\s*$"
)
_ACID_VALUE_LINE = re.compile(r"(?i)^\s*ACID\s*:\s*(?P<value>[A-Z0-9][A-Z0-9./_-]{11,})\s*$")
_LABELED_AUXILIARY_IDENTIFIER = re.compile(
    r"(?i)^\s*[A-Z][A-Z0-9 _./-]{1,40}[:#]\s*"
    r"(?P<value>[A-Z0-9][A-Z0-9./_-]{5,})\s*$"
)
_NUMBER_CAPTION_IDENTIFIER = re.compile(
    r"(?i)^\s*(?:NO\.|NUMBER\s*[-:#])\s*(?P<value>[0-9][A-Z0-9./_-]{5,})\s*$"
)
_COUNTRY_CLAUSE_HEADING = re.compile(r"^\s*(?P<country>[A-Z][A-Z ]*[A-Z])\s+CLAUSE\s*$")
_LABELED_ROUTE_TARGETS: dict[str, str] = {
    "portofdischarge": "documentPatch.route.portOfDischarge.name",
    "portofloading": "documentPatch.route.portOfLoading.name",
    "placeofdelivery": "documentPatch.route.placeOfDelivery.name",
    "placeofreceipt": "documentPatch.route.placeOfReceipt.name",
}
_LABELED_NEXT_LINE_TARGETS: dict[str, str] = {
    **_LABELED_ROUTE_TARGETS,
    "freightpayableat": "documentPatch.freight.paymentPlace.name",
    "freightpayable": "documentPatch.freight.paymentPlace.name",
}
_FORM_FIELD_CAPTIONS = frozenset(
    {
        *_SOURCE_ONLY_FIELD_CAPTIONS,
        *_LABELED_NEXT_LINE_TARGETS,
        _LOADING_TERMINAL_CAPTION,
        "originalstobereleasedat",
        "vessel",
        "oceanvessel",
        "voyageno",
        "voyagenumber",
        "precarriageby",
        "placeofissue",
        "dateofissue",
        "shipper",
        "consignee",
        "notifyparty",
    }
)
_INLINE_LABELED_ROUTE_VALUE = re.compile(
    r"(?i)^\s*(?P<caption>PORT\s+OF\s+(?:LOADING|DISCHARGE)|PLACE\s+OF\s+DELIVERY)"
    r"\s*:\s*(?P<value>\S(?:.*\S)?)\s*$"
)
_INLINE_LABELED_VESSEL_VOYAGE = re.compile(
    r"(?i)^\s*VESSEL\s+NAME\s*:\s*(?P<vessel>\S(?:.*?\S)?)\s+"
    r"VOYAGE(?:\s+(?:NO\.?|NUMBER))?\s*:\s*(?P<voyage>\S(?:.*\S)?)\s*$"
)
_FREE_DETENTION_TERM = re.compile(
    r"(?i)^[0-9]+\s+DAYS?\s+FREE\s+DETENTION(?:\s+AT\s+DESTINATION)?$"
)
_IMPORT_FREE_OUT_INSTRUCTION = re.compile(
    r"(?i)^[A-Z][A-Z .'-]{1,40}\s+IMPORT\s+IS\s+FREE\s+OUT\s*[,;]"
    r"\s*ALL\s+EXPENSES\b.+\bRECEIVERS?'?\s+ACCOUNT\.?$"
)
_RELATIONAL_COLLECTION_PATH = re.compile(
    r"^documentPatch\.(cargoGroups|cargoPackages|cargoAllocationGroups|containers)"
    r"\[([0-9]+)\](?:\.|$)"
)
_ALLOCATION_PATH = re.compile(
    r"^documentPatch\.cargoAllocationGroups\[([0-9]+)\]\.allocations\[([0-9]+)\]"
    r"(?:\.|$)"
)
_PHONE_CONTEXT = re.compile(r"(?i)\b(?:phone|telephone|tel|mobile|fax|phn|fx)\b")
_COUNTRY_CODE_CAPTION = re.compile(
    r"(?is)\b(?:country(?:\s+of\s+origin)?|nationality)\s+code\s*[:=-]?\s*$"
)
_COUNTRY_CODE_LABELED_VALUE = re.compile(
    r"(?i)\b(?:country(?:\s+of\s+origin)?|nationality)\s+code\s*[:=-]?\s*"
    r"(?P<value>[A-Z]{2})(?![A-Z0-9])"
)
_DANGEROUS_GOODS_CLASS_SURFACE = re.compile(
    r"(?i)^\s*CL(?:ASS)?\s*(?P<value>[0-9](?:\.[0-9]+)?)\s*$"
)
_BARE_DANGEROUS_GOODS_CLASS_SURFACE = re.compile(r"^\s*(?P<value>[0-9](?:\.[0-9]+)?)\s*$")
_CARRIER_REGISTRY_MARKER = re.compile(r"(?i)(?<![A-Z])R\s*\.?\s*C\s*\.?\s*S\s*\.?(?![A-Z])")
_GROUPED_NUMERIC_REGISTRY_IDENTIFIER = re.compile(
    r"^\s*(?P<value>[0-9]{2,}(?:[ .-]+[0-9]{2,})+)\s*$"
)
_EQUIPMENT_RECEIPT_PREFIX = re.compile(r"(?i)^\s*[0-9]+\s*[x\u00d7]\s*[0-9]")
_BARE_EQUIPMENT_MULTIPLIER = re.compile(r"(?i)^\s*[0-9]+\s*[x\u00d7]\s*$")
_TRAILING_FOOTNOTE_MARKER = re.compile(
    r"(?s)^(?P<body>.*[A-Za-z0-9])(?P<suffix>[ \t]+[*\u2020\u2021\u2666]+)$"
)
_SEMANTIC_ONLY_REJECTION_RECEIPT = (
    "Host rejected the compiler's semantic-only declaration for pinned structured evidence."
)


@dataclass(frozen=True)
class LineSpan:
    number: int
    line_id: str
    char_start: int
    char_end: int
    byte_start: int
    byte_end: int
    text: str


@dataclass(frozen=True)
class SpanDraft:
    draft_id: str
    logical_key: str
    render_mode: str
    value_kind: str
    group_kind: str
    group_key: str
    target_paths: tuple[str, ...]
    derivation: str | None
    dependency_paths: tuple[str, ...]
    dependency_bindings: tuple[str, ...]
    char_start: int
    char_end: int
    source_text: str
    evidence_origin: str
    render_policy: str
    rationale: str


class DraftSourceAlignmentError(ValueError):
    """A draft's character coordinates do not reproduce its claimed pinned OCR bytes."""


def validate_draft_source_alignment(*, raw: str, drafts: Sequence[SpanDraft]) -> None:
    """Require every draft surface to be the exact character slice it claims to own.

    Drafts pass through several deterministic normalizers after provider output is resolved.
    Any normalizer that changes a boundary must update ``source_text`` in the same transaction;
    otherwise downstream occurrence indexing can no longer identify the selected source bytes.
    Keep this as an explicit state invariant instead of letting a later payload renderer discover
    the corruption outside the compiler/critic rejection boundary.
    """

    for draft in drafts:
        valid_bounds = 0 <= draft.char_start < draft.char_end <= len(raw)
        pinned_surface = raw[draft.char_start : draft.char_end] if valid_bounds else None
        if pinned_surface != draft.source_text:
            raise DraftSourceAlignmentError(
                f"draft {draft.draft_id} ({draft.logical_key}) source span "
                f"[{draft.char_start},{draft.char_end}) does not match pinned OCR; "
                f"claimed={draft.source_text!r}; pinned={pinned_surface!r}"
            )


@dataclass(frozen=True)
class RequiredTargetCoBinding:
    relationship: str
    target_paths: tuple[str, ...]


@dataclass(frozen=True)
class TargetBindingSurfaceAnalysis:
    """Host-proven relationship between one target binding and its structured scalar."""

    literal_prefix: str
    literal_suffix: str
    omitted_target_prefix_tokens: tuple[str, ...]
    omitted_target_suffix_tokens: tuple[str, ...]


@dataclass(frozen=True)
class RelationalAnchorLocalityReport:
    """Deterministic receipt for source-label relationship based anchor normalization."""

    orientation: Literal["forward", "backward"] | None
    forward_evidence: int
    backward_evidence: int
    pivot_count: int
    cargo_block_assignments: tuple[str, ...]
    relocated_anchor_ids: tuple[str, ...]
    expanded_anchor_ids: tuple[str, ...]
    recovered_topology_anchor_ids: tuple[str, ...]
    suppressed_anchor_ids: tuple[str, ...]
    deduplicated_anchor_ids: tuple[str, ...]
    preserved_anchor_ids: tuple[str, ...]


def verify_file(path: Path, expected_sha256: str) -> None:
    actual = sha256_file(path)
    if actual != expected_sha256:
        raise ValueError(f"SHA-256 mismatch for {path}: {actual} != {expected_sha256}")


def load_jsonl(path: Path, expected: int | None = None) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("rb") as stream:
        for line_number, line in enumerate(stream, start=1):
            try:
                row = json.loads(line)
            except json.JSONDecodeError as error:
                raise ValueError(f"invalid JSON at {path}:{line_number}") from error
            if not isinstance(row, dict):
                raise ValueError(f"JSONL row is not an object at {path}:{line_number}")
            rows.append(row)
    if expected is not None and len(rows) != expected:
        raise ValueError(f"{path} contains {len(rows)} rows, expected {expected}")
    return rows


def page_body_spans(value: str) -> dict[int, tuple[int, int]]:
    matches = list(_PAGE_HEADER.finditer(value))
    numbers = [int(match.group(1)) for match in matches]
    if numbers != list(range(1, len(matches) + 1)):
        raise ValueError("joined OCR page headers must be contiguous from page one")
    output: dict[int, tuple[int, int]] = {}
    for index, match in enumerate(matches):
        section_end = matches[index + 1].start() if index + 1 < len(matches) else len(value)
        start = match.end()
        while start < section_end and value[start] in "\r\n":
            start += 1
        end = section_end
        while end > start and value[end - 1] in "\r\n":
            end -= 1
        output[int(match.group(1))] = (start, end)
    return output


@lru_cache(maxsize=64)
def line_spans(value: str) -> tuple[LineSpan, ...]:
    output: list[LineSpan] = []
    char_cursor = 0
    byte_cursor = 0
    for number, raw_line in enumerate(value.splitlines(keepends=True), start=1):
        line = raw_line.rstrip("\r\n")
        encoded = line.encode("utf-8")
        output.append(
            LineSpan(
                number=number,
                line_id=f"L{number:05d}",
                char_start=char_cursor,
                char_end=char_cursor + len(line),
                byte_start=byte_cursor,
                byte_end=byte_cursor + len(encoded),
                text=line,
            )
        )
        char_cursor += len(raw_line)
        byte_cursor += len(raw_line.encode("utf-8"))
    if (
        (not output or char_cursor != len(value))
        and value
        and (not output or char_cursor < len(value))
    ):
        number = len(output) + 1
        line = value[char_cursor:]
        output.append(
            LineSpan(
                number=number,
                line_id=f"L{number:05d}",
                char_start=char_cursor,
                char_end=len(value),
                byte_start=byte_cursor,
                byte_end=len(value.encode("utf-8")),
                text=line,
            )
        )
    return tuple(output)


def line_number_for_char(lines: Sequence[LineSpan], offset: int) -> int:
    starts = [line.char_start for line in lines]
    index = bisect.bisect_right(starts, offset) - 1
    if index < 0 or index >= len(lines):
        raise ValueError(f"character offset is outside source lines: {offset}")
    return lines[index].number


def line_range_for_chars(lines: Sequence[LineSpan], start: int, end: int) -> tuple[str, str]:
    if end <= start:
        raise ValueError("empty character span")
    return (
        f"L{line_number_for_char(lines, start):05d}",
        f"L{line_number_for_char(lines, end - 1):05d}",
    )


def _render_policy(anchor_rows: Sequence[Mapping[str, Any]]) -> str:
    families = {str(row["surface_family"]) for row in anchor_rows}
    paths = {str(row["relation_target_path"]) for row in anchor_rows}
    physical_surfaces = {
        (
            int(row["page_number"]),
            int(row["page_start"]),
            int(row["page_end"]),
            str(row["raw_value"]),
        )
        for row in anchor_rows
    }
    if len(physical_surfaces) == 1 and (
        "iso6346_compact" in families
        or any(
            path.endswith(("billOfLadingNumber", "voyageNumber", "containerNumber"))
            or ".sealNumbers[" in path
            for path in paths
        )
    ):
        return "opaque_identifier"
    if any(
        family.startswith(("iso_ymd", "numeric_dmy", "month_name", "date_")) for family in families
    ):
        return "date_surface"
    if families and families <= {"integer", "decimal_measure"}:
        return "numeric_surface"
    if any(path.endswith(("typeCategory", "sizeCategory", "typeDescription")) for path in paths):
        return "categorical_surface"
    return "natural_text"


def _indexed_group(path: str, collection: str, label: str) -> str | None:
    match = re.search(rf"\.{re.escape(collection)}\[([0-9]+)\]", path)
    return f"{label}:{match.group(1)}" if match else None


def _canonical_target_group(paths: Sequence[str]) -> tuple[str, str]:
    groups: list[tuple[str, str]] = []
    for path in paths:
        if ".parties." in path:
            notify = _indexed_group(path, "notifyParties", "party:notify")
            if notify:
                groups.append(("party", notify))
            else:
                role = path.split(".parties.", 1)[1].split(".", 1)[0]
                groups.append(("carrier" if role == "carrier" else "party", f"party:{role}:0"))
            continue
        dangerous_goods = re.search(
            r"\.cargoGroups\[([0-9]+)\]\.dangerousGoods\[([0-9]+)\]",
            path,
        )
        if dangerous_goods:
            cargo_group, dangerous_goods_row = dangerous_goods.groups()
            groups.append(
                (
                    "dangerous_goods",
                    f"dangerous_goods:{cargo_group}:{dangerous_goods_row}",
                )
            )
            continue
        allocation = re.search(
            r"\.cargoAllocationGroups\[([0-9]+)\](?:\.allocations\[([0-9]+)\])?",
            path,
        )
        if allocation:
            group, row = allocation.groups()
            groups.append(
                (
                    "cargo",
                    f"allocation:{group}:{row}" if row is not None else f"allocation:{group}",
                )
            )
            continue
        for collection, kind, label in (
            ("containers", "equipment", "container"),
            ("cargoPackages", "package", "package"),
            ("cargoGroups", "cargo", "cargo"),
        ):
            group = _indexed_group(path, collection, label)
            if group:
                groups.append((kind, group))
                break
        else:
            if ".route." in path:
                groups.append(("route", "route:" + path.rsplit(".", 2)[-2]))
            elif ".placeOfIssue." in path:
                groups.append(("document", "document"))
            elif ".transport." in path:
                groups.append(("transport", "transport:" + path.rsplit(".", 1)[-1]))
            elif ".freight." in path:
                groups.append(("commercial", "commercial:freight"))
            else:
                groups.append(("document", "document"))
    unique = tuple(dict.fromkeys(groups))
    for preferred_kind in ("carrier", "equipment", "package"):
        preferred = [row for row in unique if row[0] == preferred_kind]
        other = [row for row in unique if row[0] not in {preferred_kind, "cargo"}]
        if len(preferred) == 1 and not other:
            return preferred[0]
    if len(unique) == 1:
        return unique[0]
    kinds = {kind for kind, _group in unique}
    if len(kinds) == 1:
        return next(iter(kinds)), "compound:" + "|".join(group for _kind, group in unique)
    return "other", "compound:" + "|".join(group for _kind, group in unique)


def _value_kind(paths: Sequence[str], policy: str) -> str:
    joined = " ".join(paths).lower()
    if "emailaddresses" in joined:
        return "email"
    if "phonenumbers" in joined:
        return "phone"
    if "websiteurls" in joined or "urlordomains" in joined:
        return "url_or_domain"
    if ".parties." in joined and ".address" in joined:
        return "address"
    if ".parties." in joined and "contactname" in joined:
        return "contact_name"
    if ".parties." in joined and joined.endswith(".name"):
        return "organization"
    if "country" in joined:
        return "location"
    if "date" in joined or policy == "date_surface":
        return "date"
    if "containernumber" in joined or "sealnumbers" in joined:
        return "equipment"
    if "billofladingnumber" in joined or "voyagenumber" in joined:
        return "identifier"
    # Units are textual enum leaves even when their parent is a numeric measurement.
    # This check must precede the parent-name checks below (``grossWeight``, ``volume``,
    # and ``temperature``), otherwise a unit such as KGS is misclassified as a number.
    if any(path.lower().endswith(".unit") for path in paths):
        return "other_text"
    if ".containers[" in joined and any(
        field in joined for field in ("typecategory", "sizecategory", "typedescription")
    ):
        return "equipment"
    if "temperature" in joined:
        return "temperature"
    if "dangerousgoods" in joined:
        return "dangerous_goods"
    if "hscodes" in joined:
        return "identifier"
    if ".transport.vesselname" in joined:
        # A vessel name is lexical text, not a fixed-width equipment identifier.
        return "other_text"
    if ".freight.paymentplace." in joined:
        return "location"
    if ".freight.paymentarrangement" in joined:
        return "commercial_text"
    if any(word in joined for word in ("weight", "volume")):
        return "decimal_measurement"
    if any(word in joined for word in ("quantity", "count")):
        return "integer"
    if ".cargopackages[" in joined and "typecategory" in joined:
        return "package"
    if "route" in joined or "placeofissue" in joined:
        return "location"
    if any(field in joined for field in (".city", ".postalcode")):
        return "location"
    if "package" in joined:
        return "package"
    if "cargo" in joined or "marksandnumbers" in joined:
        return "cargo_text"
    if policy == "opaque_identifier":
        return "identifier"
    return "other_text"


def _is_renderable_anchor(row: Mapping[str, Any]) -> bool:
    path = str(row["relation_target_path"])
    if not path.endswith(".negotiability"):
        return True
    raw = re.sub(r"[^a-z0-9]+", " ", str(row["raw_value"]).lower()).strip()
    target = str(row.get("target_value", "")).lower()
    if target == "non_negotiable":
        return any(
            phrase in raw
            for phrase in ("non negotiable", "straight bill", "sea waybill", "seawaybill")
        )
    if target == "negotiable":
        return "to order" in raw or ("negotiable" in raw and "non negotiable" not in raw)
    return False


def normalize_relational_anchor_locality(
    *,
    raw: str,
    document_id: str,
    anchors: Sequence[Mapping[str, Any]],
    source_target: Mapping[str, Any],
) -> tuple[tuple[dict[str, Any], ...], RelationalAnchorLocalityReport]:
    """Relocate accepted anchors only when the structured row relationship proves locality.

    The preparation inventory locates equal values independently, so repeated cargo and package
    values can all point at the first textual occurrence.  This pass uses only source-label
    ``groupId``/allocation/container relationships and exact OCR spans.  It changes an anchor only
    when a document-level row orientation is evidenced by at least three unique facts with at
    least 75% directional agreement.  A contradicted anchor without a provable replacement is
    disabled rather than left as false accepted evidence.
    """

    document_patch = source_target.get("documentPatch")
    if not isinstance(document_patch, Mapping):
        raise ValueError("source target lacks documentPatch for relational anchor locality")

    collections: dict[str, tuple[Mapping[str, Any], ...]] = {}
    for name in (
        "cargoGroups",
        "cargoPackages",
        "cargoAllocationGroups",
        "containers",
    ):
        value = document_patch.get(name, ())
        if value is None:
            value = ()
        if not isinstance(value, (tuple, list)) or any(
            not isinstance(row, Mapping) for row in value
        ):
            raise ValueError(f"source target {name} is not a row sequence")
        collections[name] = tuple(value)

    cluster_by_collection_row: dict[tuple[str, int], str] = {}
    for collection_name in ("cargoGroups", "cargoPackages", "cargoAllocationGroups"):
        for index, row in enumerate(collections[collection_name]):
            group_id = row.get("groupId")
            if isinstance(group_id, str) and group_id:
                cluster_by_collection_row[(collection_name, index)] = group_id

    container_groups_by_number: dict[str, set[str]] = defaultdict(set)
    for allocation_group in collections["cargoAllocationGroups"]:
        group_id = allocation_group.get("groupId")
        allocations = allocation_group.get("allocations", ())
        if allocations is None:
            allocations = ()
        if not isinstance(allocations, (tuple, list)) or any(
            not isinstance(row, Mapping) for row in allocations
        ):
            raise ValueError("cargo allocation group allocations are not a row sequence")
        if not isinstance(group_id, str) or not group_id:
            continue
        for allocation in allocations:
            container_number = allocation.get("containerNumber")
            if isinstance(container_number, str) and container_number:
                container_groups_by_number[container_number].add(group_id)

    container_indexes_by_number: dict[str, set[int]] = defaultdict(set)
    for index, row in enumerate(collections["containers"]):
        container_number = row.get("containerNumber")
        if not isinstance(container_number, str) or not container_number:
            continue
        container_indexes_by_number[container_number].add(index)
        groups = container_groups_by_number.get(container_number, set())
        if len(groups) == 1:
            cluster_by_collection_row[("containers", index)] = next(iter(groups))

    pages = page_body_spans(raw)

    def anchor_id(row: Mapping[str, Any]) -> str:
        value = row.get("anchor_id")
        if not isinstance(value, str) or not value:
            raise ValueError("relational anchor lacks a non-empty anchor ID")
        return value

    def absolute_span(row: Mapping[str, Any]) -> tuple[int, int] | None:
        patchable = row.get("patchable")
        if not isinstance(patchable, bool):
            raise ValueError(f"anchor {anchor_id(row)} has a non-boolean patchable flag")
        if not patchable:
            return None
        page_number = row.get("page_number")
        page_start = row.get("page_start")
        page_end = row.get("page_end")
        if (
            not isinstance(page_number, int)
            or isinstance(page_number, bool)
            or not isinstance(page_start, int)
            or isinstance(page_start, bool)
            or not isinstance(page_end, int)
            or isinstance(page_end, bool)
            or page_number not in pages
        ):
            raise ValueError(f"patchable anchor {anchor_id(row)} has invalid page coordinates")
        body_start, body_end = pages[page_number]
        start = body_start + page_start
        end = body_start + page_end
        surface = row.get("raw_value")
        if (
            not isinstance(surface, str)
            or not surface
            or start < body_start
            or end > body_end
            or raw[start:end] != surface
        ):
            raise ValueError(f"patchable anchor {anchor_id(row)} no longer matches pinned OCR")
        return start, end

    def parsed_collection(path: str) -> tuple[str, int] | None:
        match = _RELATIONAL_COLLECTION_PATH.match(path)
        if match is None:
            return None
        return match.group(1), int(match.group(2))

    def cluster_for_path(path: str) -> str | None:
        parsed = parsed_collection(path)
        return cluster_by_collection_row.get(parsed) if parsed is not None else None

    def container_index_for_path(path: str) -> int | None:
        parsed = parsed_collection(path)
        if parsed is not None and parsed[0] == "containers":
            return parsed[1]
        match = _ALLOCATION_PATH.match(path)
        if match is None:
            return None
        group_index = int(match.group(1))
        allocation_index = int(match.group(2))
        allocation_groups = collections["cargoAllocationGroups"]
        if group_index >= len(allocation_groups):
            raise ValueError(f"allocation target path is outside source target: {path}")
        allocations = allocation_groups[group_index].get("allocations", ())
        if not isinstance(allocations, (tuple, list)) or allocation_index >= len(allocations):
            raise ValueError(f"allocation target path is outside source target: {path}")
        allocation = allocations[allocation_index]
        if not isinstance(allocation, Mapping):
            raise ValueError(f"allocation target path does not resolve to an object: {path}")
        container_number = allocation.get("containerNumber")
        if not isinstance(container_number, str):
            return None
        indexes = container_indexes_by_number.get(container_number, set())
        return next(iter(indexes)) if len(indexes) == 1 else None

    exact_match_cache: dict[str, tuple[tuple[int, int], ...]] = {}

    def exact_matches(surface: str) -> tuple[tuple[int, int], ...]:
        if not surface:
            raise ValueError("relational anchor has an empty raw value")
        cached = exact_match_cache.get(surface)
        if cached is not None:
            return cached
        matches = tuple(
            (match.start(), match.end())
            for match in re.finditer(re.escape(surface), raw)
            if is_token_bounded_surface_span(raw, match.start(), match.end())
        )
        exact_match_cache[surface] = matches
        return matches

    # Preparation anchors are value-oriented. A short scalar can therefore be pinned inside a
    # longer, unrelated alphanumeric token (for example the country code ``US`` inside a document
    # number). Such an occurrence is not safe evidence. Preserve deliberate adjacent structured
    # segmentation such as ``3815.840`` + ``KGS``; otherwise relocate only when one token-bounded
    # occurrence shares a blank-line-delimited block with another safe anchor from the same
    # structured entity. Ambiguous and context-free cases are disabled for semantic compilation.
    source_lines = line_spans(raw)
    block_by_line: dict[int, int] = {}
    block_index = -1
    inside_block = False
    for line in source_lines:
        if line.text.strip():
            if not inside_block:
                block_index += 1
                inside_block = True
            block_by_line[line.number] = block_index
        else:
            inside_block = False

    def line_block(position: int) -> int | None:
        line_number = line_number_for_char(source_lines, position)
        return block_by_line.get(line_number)

    raw_anchor_spans = tuple(
        (row, span) for row in anchors if (span := absolute_span(row)) is not None
    )

    def has_complete_adjacent_segmentation(span: tuple[int, int]) -> bool:
        start, end = span
        left_inside = start > 0 and raw[start - 1].isalnum() and raw[start].isalnum()
        right_inside = end < len(raw) and raw[end - 1].isalnum() and raw[end].isalnum()
        left_owned = any(
            other_span[1] == start for _other, other_span in raw_anchor_spans if other_span != span
        )
        right_owned = any(
            other_span[0] == end for _other, other_span in raw_anchor_spans if other_span != span
        )
        return (not left_inside or left_owned) and (not right_inside or right_owned)

    malformed_ids = {
        anchor_id(row)
        for row, span in raw_anchor_spans
        if not has_complete_adjacent_segmentation(span)
    }
    safe_contexts: dict[tuple[str, str], set[int]] = defaultdict(set)
    for row, span in raw_anchor_spans:
        if anchor_id(row) in malformed_ids:
            continue
        path = row.get("relation_target_path")
        block = line_block(span[0])
        if isinstance(path, str) and block is not None:
            safe_contexts[_canonical_target_group((path,))].add(block)

    boundary_candidates: dict[str, tuple[int, int] | None] = {}
    for row, _span in raw_anchor_spans:
        current_id = anchor_id(row)
        if current_id not in malformed_ids:
            continue
        path = row.get("relation_target_path")
        surface = row.get("raw_value")
        if not isinstance(path, str) or not isinstance(surface, str):
            boundary_candidates[current_id] = None
            continue
        context_blocks = safe_contexts.get(_canonical_target_group((path,)), set())
        candidates = tuple(
            candidate
            for candidate in exact_matches(surface)
            if line_block(candidate[0]) in context_blocks
        )
        boundary_candidates[current_id] = candidates[0] if len(candidates) == 1 else None

    claims_by_span: dict[tuple[int, int], set[str]] = defaultdict(set)
    for row in anchors:
        candidate = boundary_candidates.get(anchor_id(row))
        path = row.get("relation_target_path")
        if candidate is not None and isinstance(path, str):
            claims_by_span[candidate].add(path)

    boundary_relocated: list[str] = []
    boundary_suppressed: list[str] = []
    boundary_normalized: list[Mapping[str, Any]] = []
    for source_row in anchors:
        current_id = anchor_id(source_row)
        if current_id not in malformed_ids:
            boundary_normalized.append(source_row)
            continue
        row = dict(source_row)
        candidate = boundary_candidates[current_id]
        if candidate is None or len(claims_by_span[candidate]) != 1:
            row["patchable"] = False
            row["page_start"] = None
            row["page_end"] = None
            boundary_suppressed.append(current_id)
            boundary_normalized.append(row)
            continue
        coordinates = next(
            (
                (page_number, candidate[0] - body_start, candidate[1] - body_start)
                for page_number, (body_start, body_end) in pages.items()
                if body_start <= candidate[0] and candidate[1] <= body_end
            ),
            None,
        )
        if coordinates is None:
            raise ValueError("token-local anchor candidate is outside a page body")
        row.update(
            {
                "page_number": coordinates[0],
                "page_start": coordinates[1],
                "page_end": coordinates[2],
                "location_status": "host_token_locality_unique",
                "normalization_rule": (
                    "relocated malformed sub-token anchor to unique same-entity block value"
                ),
            }
        )
        boundary_relocated.append(current_id)
        boundary_normalized.append(row)
    anchors = tuple(boundary_normalized)

    pivot_rows: set[tuple[int, int, str, int]] = set()
    for row in anchors:
        if row.get("document_id") != document_id:
            raise ValueError("relational locality received an anchor for another document")
        path = row.get("relation_target_path")
        if not isinstance(path, str):
            raise ValueError(f"anchor {anchor_id(row)} lacks a target path")
        container_index = container_index_for_path(path)
        span = absolute_span(row)
        if container_index is None or span is None or not path.endswith(".containerNumber"):
            continue
        cluster = cluster_by_collection_row.get(("containers", container_index))
        if cluster is not None:
            pivot_rows.add((*span, cluster, container_index))
    pivots = tuple(sorted(pivot_rows))
    pivot_starts = tuple(row[0] for row in pivots)
    pivot_container_indexes = {row[3] for row in pivots}

    def previous_pivot(span: tuple[int, int]) -> tuple[int, int, str, int] | None:
        index = bisect.bisect_right(pivot_starts, span[0]) - 1
        return pivots[index] if index >= 0 else None

    def next_pivot(span: tuple[int, int]) -> tuple[int, int, str, int] | None:
        index = bisect.bisect_left(pivot_starts, span[1])
        return pivots[index] if index < len(pivots) else None

    forward_evidence = 0
    backward_evidence = 0
    for row in anchors:
        path = str(row.get("relation_target_path", ""))
        parsed = parsed_collection(path)
        cluster = cluster_for_path(path)
        span = absolute_span(row)
        surface = row.get("raw_value")
        if (
            parsed is None
            or parsed[0] not in {"cargoGroups", "cargoPackages", "cargoAllocationGroups"}
            or path.endswith(".containerNumber")
            or cluster is None
            or span is None
            or not isinstance(surface, str)
            or len(exact_matches(surface)) != 1
        ):
            continue
        preceding = previous_pivot(span)
        following = next_pivot(span)
        forward_match = preceding is not None and preceding[2] == cluster
        backward_match = following is not None and following[2] == cluster
        if forward_match and not backward_match:
            forward_evidence += 1
        elif backward_match and not forward_match:
            backward_evidence += 1

    directional_evidence = forward_evidence + backward_evidence
    orientation: Literal["forward", "backward"] | None = None
    if directional_evidence >= 3:
        if forward_evidence * 4 >= directional_evidence * 3:
            orientation = "forward"
        elif backward_evidence * 4 >= directional_evidence * 3:
            orientation = "backward"

    pivot_clusters = {row[2] for row in pivots}

    def group_owner(span: tuple[int, int]) -> str | None:
        if orientation == "forward":
            pivot = previous_pivot(span)
            return pivot[2] if pivot is not None else None
        if orientation == "backward":
            pivot = next_pivot(span)
            return pivot[2] if pivot is not None else None
        if len(pivot_clusters) == 1:
            return next(iter(pivot_clusters))
        preceding = previous_pivot(span)
        following = next_pivot(span)
        if preceding is not None and following is not None and preceding[2] == following[2]:
            return preceding[2]
        return None

    target_paths_by_local_surface: dict[tuple[str, str, str], set[str]] = defaultdict(set)
    for row in anchors:
        path = row.get("relation_target_path")
        surface = row.get("raw_value")
        if not isinstance(path, str) or not isinstance(surface, str):
            continue
        parsed = parsed_collection(path)
        cluster = cluster_for_path(path)
        if parsed is None or cluster is None:
            continue
        container_index = container_index_for_path(path)
        container_scoped = parsed[0] == "containers" or _ALLOCATION_PATH.match(path) is not None
        if container_scoped and container_index is not None:
            local_surface_kind = "container"
            local_surface_key = str(container_index)
        else:
            local_surface_kind = "group"
            local_surface_key = cluster
        target_paths_by_local_surface[(local_surface_kind, local_surface_key, surface)].add(path)
    ambiguous_local_surfaces = {
        key
        for key, paths in target_paths_by_local_surface.items()
        if len(paths) > 1 and len(target_fact_components(source_target, tuple(paths))) > 1
    }

    def nearest_container_indexes(span: tuple[int, int]) -> set[int]:
        distances: dict[int, int] = {}
        for start, end, _cluster, container_index in pivots:
            distance = max(start - span[1], span[0] - end, 0)
            prior = distances.get(container_index)
            if prior is None or distance < prior:
                distances[container_index] = distance
        if not distances:
            return set()
        minimum = min(distances.values())
        return {index for index, distance in distances.items() if distance == minimum}

    def page_coordinates(span: tuple[int, int]) -> tuple[int, int, int] | None:
        matches = tuple(
            (page_number, span[0] - body_start, span[1] - body_start)
            for page_number, (body_start, body_end) in pages.items()
            if body_start <= span[0] and span[1] <= body_end
        )
        return matches[0] if len(matches) == 1 else None

    relocated: list[str] = list(boundary_relocated)
    expanded: list[str] = []
    suppressed: list[str] = list(boundary_suppressed)
    deduplicated: list[str] = []
    preserved: list[str] = []
    normalized: list[dict[str, Any]] = []
    emitted_relational_spans: set[tuple[str, int, int]] = set()
    for source_row in anchors:
        row = dict(source_row)
        path = str(row.get("relation_target_path", ""))
        parsed = parsed_collection(path)
        original_span = absolute_span(row)
        surface = row.get("raw_value")
        if parsed is None or not isinstance(surface, str) or original_span is None:
            normalized.append(row)
            continue

        selected_spans: tuple[tuple[int, int], ...] = (original_span,)
        locality_selector: tuple[str, str, str] | None = None
        container_index = container_index_for_path(path)
        container_scoped = parsed[0] == "containers" or _ALLOCATION_PATH.match(path) is not None
        if container_scoped and not path.endswith(".containerNumber"):
            if container_index is None:
                normalized.append(row)
                continue
            if container_index not in pivot_container_indexes:
                normalized.append(row)
                continue
            locality_selector = ("container", str(container_index), surface)
            selected_spans = tuple(
                span
                for span in exact_matches(surface)
                if nearest_container_indexes(span) == {container_index}
                and page_coordinates(span) is not None
            )
        elif parsed[0] in {
            "cargoGroups",
            "cargoPackages",
            "cargoAllocationGroups",
        } and not path.endswith(".containerNumber"):
            cluster = cluster_for_path(path)
            if cluster is None:
                normalized.append(row)
                continue
            locality_selector = ("group", cluster, surface)
            if locality_selector in ambiguous_local_surfaces:
                selected_spans = ()
            else:
                matching_spans = tuple(
                    span
                    for span in exact_matches(surface)
                    if group_owner(span) == cluster and page_coordinates(span) is not None
                )
                if parsed[0] == "cargoGroups":
                    selected_spans = matching_spans
                elif original_span in matching_spans:
                    selected_spans = (original_span,)
                elif len(matching_spans) == 1:
                    selected_spans = matching_spans
                else:
                    selected_spans = ()
        else:
            normalized.append(row)
            continue

        current_anchor_id = anchor_id(row)
        if not selected_spans or locality_selector in ambiguous_local_surfaces:
            row["patchable"] = False
            row["page_start"] = None
            row["page_end"] = None
            suppressed.append(current_anchor_id)
            normalized.append(row)
            continue

        unique_spans = tuple(
            span
            for span in dict.fromkeys(selected_spans)
            if (path, span[0], span[1]) not in emitted_relational_spans
        )
        if not unique_spans:
            deduplicated.append(current_anchor_id)
            continue
        primary_span = original_span if original_span in unique_spans else unique_spans[0]
        ordered_spans = (
            primary_span,
            *(span for span in unique_spans if span != primary_span),
        )
        for index, selected_span in enumerate(ordered_spans):
            coordinates = page_coordinates(selected_span)
            if coordinates is None:
                raise ValueError("relational anchor candidate is outside a page body")
            emitted_relational_spans.add((path, selected_span[0], selected_span[1]))
            emitted_row = row if index == 0 else dict(row)
            emitted_row["page_number"], emitted_row["page_start"], emitted_row["page_end"] = (
                coordinates
            )
            if index == 0:
                if selected_span == original_span:
                    preserved.append(current_anchor_id)
                else:
                    relocated.append(current_anchor_id)
            else:
                expanded_id = (
                    "anchor_relational_"
                    + sha256_bytes(
                        f"{document_id}\0{path}\0{selected_span[0]}\0{selected_span[1]}".encode()
                    )[:24]
                )
                emitted_row["anchor_id"] = expanded_id
                expanded.append(expanded_id)
            normalized.append(emitted_row)

    cargo_rows_by_path: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for row in anchors:
        path = row.get("relation_target_path")
        if not isinstance(path, str):
            continue
        parsed = parsed_collection(path)
        if parsed is not None and parsed[0] == "cargoGroups":
            cargo_rows_by_path[path].append(row)

    # Blank-line-delimited cargo blocks provide stronger evidence than the nearest container
    # pivot when a multi-column continuation crosses a page boundary.  In that layout the next
    # container number can be printed before the previous container's continued cargo row.  Build
    # a block signature from at least three distinctive target scalars, require 75% target-row
    # coverage, and use source order only when three or more allocation/container pivots prove the
    # cargo-group order monotonically.  Ambiguous assignments remain untouched for model review.
    source_lines = line_spans(raw)
    source_tokens = _surface_token_spans(raw)
    cargo_blocks: list[tuple[int, int, int, int]] = []
    current_block: list[LineSpan] = []
    for line in source_lines:
        if line.text.strip():
            current_block.append(line)
            continue
        if current_block:
            cargo_blocks.append(
                (
                    current_block[0].char_start,
                    current_block[-1].char_end,
                    current_block[0].number,
                    current_block[-1].number,
                )
            )
            current_block = []
    if current_block:
        cargo_blocks.append(
            (
                current_block[0].char_start,
                current_block[-1].char_end,
                current_block[0].number,
                current_block[-1].number,
            )
        )

    cargo_indexes_by_cluster: dict[str, set[int]] = defaultdict(set)
    for cargo_index, cargo_group in enumerate(collections["cargoGroups"]):
        group_id = cargo_group.get("groupId")
        if isinstance(group_id, str) and group_id:
            cargo_indexes_by_cluster[group_id].add(cargo_index)
    pivot_cargo_order: list[int] = []
    for _start, _end, cluster, _container_index in pivots:
        indexes = cargo_indexes_by_cluster.get(cluster, set())
        if len(indexes) != 1:
            continue
        cargo_index = next(iter(indexes))
        if not pivot_cargo_order or pivot_cargo_order[-1] != cargo_index:
            pivot_cargo_order.append(cargo_index)
    cargo_order_proven = (
        len(pivot_cargo_order) >= 3
        and len(set(pivot_cargo_order)) == len(pivot_cargo_order)
        and all(left < right for left, right in pairwise(pivot_cargo_order))
    )

    block_candidate_cache: dict[str, tuple[tuple[int, int], ...]] = {}

    def block_candidate_spans(value: str) -> tuple[tuple[int, int], ...]:
        cached = block_candidate_cache.get(value)
        if cached is not None:
            return cached
        spans = {
            (start, start + len(value))
            for start in _exact_offsets(raw, value)
            if is_token_bounded_surface_span(raw, start, start + len(value))
        }
        spans.update(_token_equivalent_offsets(source_tokens, value))
        spans.update(_labeled_identifier_projection_offsets(source_tokens, value))
        # Prefer the complete matching surface over a nested identifier-only projection.
        maximal = tuple(
            sorted(
                span
                for span in spans
                if not any(
                    other_start <= span[0]
                    and span[1] <= other_end
                    and (other_start, other_end) != span
                    for other_start, other_end in spans
                )
            )
        )
        block_candidate_cache[value] = maximal
        return maximal

    def cargo_string_leaves(value: Any, path: str) -> tuple[tuple[str, str], ...]:
        output: list[tuple[str, str]] = []

        def visit(item: Any, item_path: str) -> None:
            if isinstance(item, Mapping):
                for key, child in item.items():
                    if key != "groupId":
                        visit(child, f"{item_path}.{key}")
            elif isinstance(item, (tuple, list)):
                for index, child in enumerate(item):
                    visit(child, f"{item_path}[{index}]")
            elif isinstance(item, str) and item:
                output.append((item_path, item))

        visit(value, path)
        return tuple(output)

    eligible_blocks_by_cargo: dict[int, tuple[int, ...]] = {}
    if cargo_order_proven:
        for cargo_index, cargo_group in enumerate(collections["cargoGroups"]):
            leaves = cargo_string_leaves(
                cargo_group,
                f"documentPatch.cargoGroups[{cargo_index}]",
            )
            distinctive_surfaces = {
                _normalized_surface(value)
                for _path, value in leaves
                if len(_normalized_surface(value)) >= 5
            }
            matched_surfaces_by_block: dict[int, set[str]] = defaultdict(set)
            for _path, value in leaves:
                normalized_surface = _normalized_surface(value)
                if normalized_surface not in distinctive_surfaces:
                    continue
                for start, end in block_candidate_spans(value):
                    for block_index, (block_start, block_end, _line_start, _line_end) in enumerate(
                        cargo_blocks
                    ):
                        if block_start <= start and end <= block_end:
                            matched_surfaces_by_block[block_index].add(normalized_surface)
                            break
            best_match_count = max(
                (len(values) for values in matched_surfaces_by_block.values()),
                default=0,
            )
            if best_match_count < 3 or best_match_count * 4 < len(distinctive_surfaces) * 3:
                continue
            eligible_blocks_by_cargo[cargo_index] = tuple(
                sorted(
                    block_index
                    for block_index, values in matched_surfaces_by_block.items()
                    if len(values) == best_match_count
                )
            )

    assigned_blocks: dict[int, int] = {}
    singleton_claims: dict[int, list[int]] = defaultdict(list)
    for cargo_index, block_candidates in eligible_blocks_by_cargo.items():
        if len(block_candidates) == 1:
            singleton_claims[block_candidates[0]].append(cargo_index)
    for block_index, cargo_indexes in singleton_claims.items():
        if len(cargo_indexes) == 1:
            assigned_blocks[cargo_indexes[0]] = block_index
    if any(
        left_block >= right_block
        for (_left_index, left_block), (_right_index, right_block) in pairwise(
            sorted(assigned_blocks.items())
        )
    ):
        assigned_blocks = {}

    def viable_blocks(cargo_index: int) -> tuple[int, ...]:
        lower_bound = max(
            (block for index, block in assigned_blocks.items() if index < cargo_index),
            default=-1,
        )
        upper_bound = min(
            (block for index, block in assigned_blocks.items() if index > cargo_index),
            default=len(cargo_blocks),
        )
        occupied_blocks = set(assigned_blocks.values())
        return tuple(
            block
            for block in eligible_blocks_by_cargo.get(cargo_index, ())
            if lower_bound < block < upper_bound and block not in occupied_blocks
        )

    changed = True
    while changed:
        changed = False
        for cargo_index in sorted(eligible_blocks_by_cargo):
            if cargo_index in assigned_blocks:
                continue
            viable = viable_blocks(cargo_index)
            if len(viable) == 1:
                assigned_blocks[cargo_index] = viable[0]
                changed = True
        if changed:
            continue
        equal_candidate_groups: dict[tuple[int, ...], list[int]] = defaultdict(list)
        for cargo_index in sorted(eligible_blocks_by_cargo):
            if cargo_index not in assigned_blocks:
                equal_candidate_groups[viable_blocks(cargo_index)].append(cargo_index)
        for viable, cargo_indexes in equal_candidate_groups.items():
            if len(viable) <= 1 or len(viable) != len(cargo_indexes):
                continue
            for cargo_index, block_index in zip(cargo_indexes, viable, strict=True):
                assigned_blocks[cargo_index] = block_index
            changed = True

    cargo_block_assignments = tuple(
        f"cargoGroups[{cargo_index}]:L{cargo_blocks[block_index][2]:05d}-"
        f"L{cargo_blocks[block_index][3]:05d}"
        for cargo_index, block_index in sorted(assigned_blocks.items())
    )
    block_recovered: list[str] = []
    if assigned_blocks:
        rows_by_cargo_path: dict[tuple[int, str], list[Mapping[str, Any]]] = defaultdict(list)
        for path, rows in cargo_rows_by_path.items():
            parsed = parsed_collection(path)
            if parsed is not None:
                rows_by_cargo_path[(parsed[1], path)].extend(rows)

        candidate_spans_by_cargo_path: dict[tuple[int, str], tuple[tuple[int, int], ...]] = {}

        def repeated_description_component_spans(
            *,
            target_value: str,
            complete_spans: Sequence[tuple[int, int]],
            block_start: int,
            block_end: int,
        ) -> tuple[tuple[int, int], ...]:
            """Find a separately printed prefix/suffix already present in a full description."""

            target_tokens = tuple(
                token for token, _start, _end in _surface_token_spans(target_value)
            )
            if len(target_tokens) < 3 or not complete_spans:
                return ()
            block_tokens = tuple(
                (token, start, end)
                for token, start, end in source_tokens
                if block_start <= start and end <= block_end
            )
            output: set[tuple[int, int]] = set()
            for token_start in range(len(block_tokens)):
                maximum_width = min(
                    len(target_tokens) - 1,
                    len(block_tokens) - token_start,
                )
                for width in range(2, maximum_width + 1):
                    candidate_tokens = tuple(
                        token
                        for token, _start, _end in block_tokens[token_start : token_start + width]
                    )
                    if len("".join(candidate_tokens)) < 8 or not (
                        target_tokens[:width] == candidate_tokens
                        or target_tokens[-width:] == candidate_tokens
                    ):
                        continue
                    start = block_tokens[token_start][1]
                    end = block_tokens[token_start + width - 1][2]
                    if any(
                        complete_start <= start and end <= complete_end
                        for complete_start, complete_end in complete_spans
                    ):
                        continue
                    output.add((start, end))
            return tuple(sorted(output))

        for (cargo_index, path), rows in rows_by_cargo_path.items():
            assigned_block_index = assigned_blocks.get(cargo_index)
            if assigned_block_index is None:
                continue
            block_start, block_end, _line_start, _line_end = cargo_blocks[assigned_block_index]
            target_value = _resolve_target_path(source_target, path)
            if not isinstance(target_value, str) or not target_value:
                continue
            surfaces = tuple(
                dict.fromkeys(
                    (
                        target_value,
                        *(
                            str(row["raw_value"])
                            for row in rows
                            if isinstance(row.get("raw_value"), str)
                            and _normalized_surface(str(row["raw_value"]))
                            == _normalized_surface(target_value)
                        ),
                    )
                )
            )
            cargo_candidate_spans = {
                span
                for surface in surfaces
                for span in block_candidate_spans(surface)
                if block_start <= span[0] and span[1] <= block_end
            }
            if path.endswith(".description"):
                cargo_candidate_spans.update(
                    repeated_description_component_spans(
                        target_value=target_value,
                        complete_spans=tuple(cargo_candidate_spans),
                        block_start=block_start,
                        block_end=block_end,
                    )
                )
            maximal_candidates = tuple(
                sorted(
                    span
                    for span in cargo_candidate_spans
                    if not any(
                        other_start <= span[0]
                        and span[1] <= other_end
                        and (other_start, other_end) != span
                        for other_start, other_end in cargo_candidate_spans
                    )
                )
            )
            if maximal_candidates:
                candidate_spans_by_cargo_path[(cargo_index, path)] = maximal_candidates

        paths_by_candidate_span: dict[tuple[int, int, int], set[str]] = defaultdict(set)
        for (cargo_index, path), candidates in candidate_spans_by_cargo_path.items():
            for start, end in candidates:
                paths_by_candidate_span[(cargo_index, start, end)].add(path)
        candidate_spans_by_cargo_path = {
            key: tuple(
                (start, end)
                for start, end in candidates
                if len(paths_by_candidate_span[(key[0], start, end)]) == 1
            )
            for key, candidates in candidate_spans_by_cargo_path.items()
        }

        current_spans_by_cargo_path: dict[tuple[int, str], set[tuple[int, int]]] = defaultdict(set)
        for row in normalized:
            path_value = row.get("relation_target_path")
            parsed = parsed_collection(path_value) if isinstance(path_value, str) else None
            span = absolute_span(row)
            if (
                isinstance(path_value, str)
                and parsed is not None
                and parsed[0] == "cargoGroups"
                and span is not None
            ):
                current_spans_by_cargo_path[(parsed[1], path_value)].add(span)
        affected_cargo_indexes = {
            cargo_index
            for (cargo_index, path), candidates in candidate_spans_by_cargo_path.items()
            if candidates
            and set(candidates) != current_spans_by_cargo_path.get((cargo_index, path), set())
        }

        for row in normalized:
            path_value = row.get("relation_target_path")
            parsed = parsed_collection(path_value) if isinstance(path_value, str) else None
            if (
                parsed is None
                or parsed[0] != "cargoGroups"
                or parsed[1] not in affected_cargo_indexes
            ):
                continue
            assert isinstance(path_value, str)
            candidates = candidate_spans_by_cargo_path.get((parsed[1], path_value), ())
            span = absolute_span(row)
            if span is None or not candidates or span in candidates:
                continue
            row["patchable"] = False
            row["page_start"] = None
            row["page_end"] = None
            current_anchor_id = anchor_id(row)
            if current_anchor_id not in suppressed:
                suppressed.append(current_anchor_id)

        existing_spans = {
            (str(row.get("relation_target_path")), span[0], span[1])
            for row in normalized
            if (span := absolute_span(row)) is not None
        }
        for (cargo_index, path), candidates in sorted(candidate_spans_by_cargo_path.items()):
            if cargo_index not in affected_cargo_indexes:
                continue
            source_rows = rows_by_cargo_path[(cargo_index, path)]
            for start, end in candidates:
                if (path, start, end) in existing_spans:
                    continue
                coordinates = page_coordinates((start, end))
                if coordinates is None:
                    raise ValueError("relational cargo-block candidate is outside a page body")
                recovered_id = (
                    "anchor_relational_block_"
                    + sha256_bytes(f"{document_id}\0{path}\0{start}\0{end}".encode())[:24]
                )
                recovered_row = dict(source_rows[0])
                recovered_row.update(
                    {
                        "anchor_id": recovered_id,
                        "patchable": True,
                        "page_number": coordinates[0],
                        "page_start": coordinates[1],
                        "page_end": coordinates[2],
                        "raw_value": raw[start:end],
                        "location_status": "host_relational_cargo_block_unique",
                        "normalization_rule": (
                            "recovered exact target projection from ordered cargo-block signature"
                        ),
                    }
                )
                normalized.append(recovered_row)
                existing_spans.add((path, start, end))
                block_recovered.append(recovered_id)

    provisional_drafts = normalize_target_cobindings(
        drafts=anchor_drafts(raw=raw, document_id=document_id, anchors=normalized),
        source_target=source_target,
    )

    def cluster_candidate_filter(expected_cluster: str) -> Callable[[str, int, int], bool]:
        return lambda _path, start, end: group_owner((start, end)) == expected_cluster

    topology_candidates: list[SpanDraft] = []
    for path, source_rows in sorted(cargo_rows_by_path.items()):
        parsed = parsed_collection(path)
        if parsed is not None and parsed[1] in assigned_blocks:
            # The stronger block-signature pass already enumerated every unambiguous occurrence
            # for this cargo entity. Re-running the document-wide bracket search is redundant and
            # was the dominant CPU cost of the added deterministic preprocessing.
            continue
        cluster = cluster_for_path(path)
        if cluster is None:
            continue
        topology_drafts = _repair_uniquely_bracketed_target_paths(
            raw=raw,
            missing_paths=(path,),
            occupied=provisional_drafts,
            source_target=source_target,
            source_hints={
                path: tuple(
                    dict.fromkeys(
                        str(row["raw_value"])
                        for row in source_rows
                        if isinstance(row.get("raw_value"), str) and row["raw_value"]
                    )
                )
            },
            candidate_span_filter=cluster_candidate_filter(cluster),
        )
        topology_candidates.extend(
            candidate
            for candidate in topology_drafts
            if group_owner((candidate.char_start, candidate.char_end)) == cluster
        )
    safe_topology_candidates = tuple(
        candidate
        for index, candidate in enumerate(topology_candidates)
        if not any(
            index != other_index
            and candidate.target_paths != other.target_paths
            and candidate.char_start < other.char_end
            and other.char_start < candidate.char_end
            for other_index, other in enumerate(topology_candidates)
        )
    )
    recovered_topology: list[str] = list(block_recovered)
    for topology_candidate in safe_topology_candidates:
        path = topology_candidate.target_paths[0]
        source_row = dict(cargo_rows_by_path[path][0])
        coordinates = page_coordinates((topology_candidate.char_start, topology_candidate.char_end))
        if coordinates is None:
            raise ValueError("relational topology candidate is outside a page body")
        recovered_id = (
            "anchor_relational_topology_"
            + sha256_bytes(
                f"{document_id}\0{path}\0{topology_candidate.char_start}\0"
                f"{topology_candidate.char_end}".encode()
            )[:24]
        )
        source_row.update(
            {
                "anchor_id": recovered_id,
                "patchable": True,
                "page_number": coordinates[0],
                "page_start": coordinates[1],
                "page_end": coordinates[2],
                "raw_value": topology_candidate.source_text,
                "location_status": "host_relational_topology_unique",
                "normalization_rule": (
                    "recovered exact target projection from unique same-group topology"
                ),
            }
        )
        normalized.append(source_row)
        recovered_topology.append(recovered_id)

    report = RelationalAnchorLocalityReport(
        orientation=orientation,
        forward_evidence=forward_evidence,
        backward_evidence=backward_evidence,
        pivot_count=len(pivots),
        cargo_block_assignments=cargo_block_assignments,
        relocated_anchor_ids=tuple(relocated),
        expanded_anchor_ids=tuple(expanded),
        recovered_topology_anchor_ids=tuple(recovered_topology),
        suppressed_anchor_ids=tuple(suppressed),
        deduplicated_anchor_ids=tuple(deduplicated),
        preserved_anchor_ids=tuple(preserved),
    )
    return tuple(normalized), report


def anchor_drafts(
    *, raw: str, document_id: str, anchors: Sequence[Mapping[str, Any]]
) -> tuple[SpanDraft, ...]:
    pages = page_body_spans(raw)
    located: list[tuple[int, int, Mapping[str, Any]]] = []
    for row in anchors:
        if (
            row.get("document_id") != document_id
            or not row.get("patchable")
            or not _is_renderable_anchor(row)
        ):
            continue
        page = int(row["page_number"])
        page_start, page_end = pages[page]
        local_start = row.get("page_start")
        local_end = row.get("page_end")
        if not isinstance(local_start, int) or not isinstance(local_end, int):
            raise ValueError("patchable anchor lacks integer character offsets")
        absolute_start = page_start + local_start
        absolute_end = page_start + local_end
        if absolute_end > page_end or raw[absolute_start:absolute_end] != row["raw_value"]:
            raise ValueError(f"accepted anchor no longer matches pinned OCR: {row['anchor_id']}")
        located.append((absolute_start, absolute_end, row))
    located.sort(key=lambda item: (item[0], item[1]))

    groups: list[list[tuple[int, int, Mapping[str, Any]]]] = []
    for item in located:
        if not groups or item[0] >= max(existing[1] for existing in groups[-1]):
            groups.append([item])
        else:
            groups[-1].append(item)

    drafts: list[SpanDraft] = []
    for ordinal, group in enumerate(groups, start=1):
        start = min(item[0] for item in group)
        end = max(item[1] for item in group)
        rows = [item[2] for item in group]
        row_anchor_ids = tuple(str(row["anchor_id"]) for row in rows)
        host_inferred = any(
            anchor_id.startswith(
                (
                    "anchor_relational_",
                    "anchor_relational_block_",
                    "anchor_relational_topology_",
                )
            )
            for anchor_id in row_anchor_ids
        )
        paths = tuple(sorted({str(row["relation_target_path"]) for row in rows}))
        policy = _render_policy(rows)
        carrier = any(path.startswith("documentPatch.parties.carrier") for path in paths)
        group_kind, group_key = _canonical_target_group(paths)
        drafts.append(
            SpanDraft(
                draft_id=f"anchor_binding_{ordinal:04d}",
                logical_key="anchor:" + "|".join(paths),
                render_mode="carrier_static" if carrier else "target_binding",
                value_kind=_value_kind(paths, policy),
                group_kind="carrier" if carrier else group_kind,
                group_key=group_key,
                target_paths=paths,
                derivation=None,
                dependency_paths=(),
                dependency_bindings=(),
                char_start=start,
                char_end=end,
                source_text=raw[start:end],
                evidence_origin=(
                    "host_verified_agent_proposal" if host_inferred else "accepted_label_evidence"
                ),
                render_policy=policy,
                rationale=(
                    f"Host-inferred relational evidence group {ordinal}."
                    if host_inferred
                    else f"Pinned accepted-label evidence group {ordinal}."
                ),
            )
        )
    return tuple(drafts)


def _overlaps(start: int, end: int, spans: Sequence[SpanDraft]) -> bool:
    return any(start < span.char_end and span.char_start < end for span in spans)


def risk_candidates(raw: str, owned: Sequence[SpanDraft]) -> tuple[RiskCandidate, ...]:
    candidates: list[tuple[int, int, str, str, str]] = []
    patterns: tuple[tuple[str, re.Pattern[str], int | str], ...] = (
        ("selected_text", _SELECTED_OPERATIONAL_TEXT, 0),
        ("selected_text", _SELECTED_TYPE_VALUE, "value"),
        ("email", _EMAIL, 0),
        ("url_or_domain", _URL, 0),
        ("equipment_identifier", _EQUIPMENT, 0),
        ("date", _DATE, 0),
        ("measurement", _MEASUREMENT, 0),
        ("phone", _PHONE, 0),
        ("alphanumeric_identifier", _TOKEN, 0),
        ("long_numeric_identifier", _LONG_NUMBER, 0),
    )
    source_lines = line_spans(raw)
    for line in source_lines:
        if not line.text.strip() or _PAGE_HEADER.fullmatch(line.text):
            continue
        accepted_on_line: list[tuple[int, int]] = []
        for kind, pattern, capture in patterns:
            for match in pattern.finditer(line.text):
                text = match.group(capture).rstrip(".,;:)]}")
                if not text:
                    continue
                local_start = match.start(capture)
                local_end = local_start + len(text)
                if kind == "phone" and sum(character.isdigit() for character in text) < 7:
                    continue
                if kind == "phone" and not (
                    _PHONE_CONTEXT.search(line.text) or text.lstrip().startswith("+")
                ):
                    continue
                if kind == "alphanumeric_identifier" and not (
                    any(character.isalpha() for character in text)
                    and any(character.isdigit() for character in text)
                ):
                    continue
                start = line.char_start + local_start
                end = line.char_start + local_end
                if _overlaps(start, end, owned):
                    continue
                if any(
                    start < prior_end and prior_start < end
                    for prior_start, prior_end in accepted_on_line
                ):
                    continue
                accepted_on_line.append((start, end))
                candidates.append((start, end, kind, line.line_id, raw[start:end]))

    # A selected value under this caption is operational document data even when it contains no
    # generic identifier/date token (for example ``E / Express B/L``). Keep it in the mandatory
    # ownership inventory so neither compiler nor critic can silently leave the value literal.
    existing_spans = {(start, end) for start, end, _kind, _line_id, _text in candidates}
    for caption_index, caption in enumerate(source_lines[:-1]):
        if _ORIGINAL_BILL_COUNT_CAPTION.fullmatch(caption.text) is None:
            continue
        value_line = next(
            (
                candidate
                for candidate in source_lines[caption_index + 1 : caption_index + 4]
                if candidate.text.strip()
            ),
            None,
        )
        if value_line is None:
            continue
        leading = len(value_line.text) - len(value_line.text.lstrip())
        trailing = len(value_line.text.rstrip())
        start = value_line.char_start + leading
        end = value_line.char_start + trailing
        if (
            start >= end
            or end - start > 80
            or not any(character.isalnum() for character in raw[start:end])
            or _overlaps(start, end, owned)
            or (start, end) in existing_spans
        ):
            continue
        candidates.append((start, end, "selected_text", value_line.line_id, raw[start:end]))
        existing_spans.add((start, end))
    candidates.sort(key=lambda row: (row[0], row[1], row[2]))
    return tuple(
        RiskCandidate.model_validate(
            {
                "risk_id": f"risk_{index:04d}",
                "kind": kind,
                "line_id": line_id,
                "byte_start": len(raw[:start].encode("utf-8")),
                "byte_end": len(raw[:end].encode("utf-8")),
                "source_text": text,
            }
        )
        for index, (start, end, kind, line_id, text) in enumerate(candidates, start=1)
    )


def _structured_package_quantity_paths(source_target: Mapping[str, Any]) -> dict[str, int]:
    """Return exact package/allocation quantity paths that can ground a printed interval."""

    patch = source_target.get("documentPatch")
    if not isinstance(patch, Mapping):
        return {}
    quantities: dict[str, int] = {}

    def visit(value: Any, path: str) -> None:
        if isinstance(value, Mapping):
            for key, child in value.items():
                child_path = f"{path}.{key}"
                if (
                    key in {"quantity", "packageQuantity"}
                    and isinstance(child, int)
                    and not isinstance(child, bool)
                    and child > 0
                    and path.startswith(
                        (
                            "documentPatch.cargoPackages[",
                            "documentPatch.cargoAllocationGroups[",
                        )
                    )
                ):
                    quantities[child_path] = child
                visit(child, child_path)
        elif isinstance(value, list):
            for index, child in enumerate(value):
                visit(child, f"{path}[{index}]")

    visit(patch, "documentPatch")
    return quantities


def _structured_package_quantities(source_target: Mapping[str, Any]) -> frozenset[int]:
    """Return exact package/allocation quantities that can ground a printed interval."""

    return frozenset(_structured_package_quantity_paths(source_target).values())


def all_risk_candidates(
    raw: str,
    owned: Sequence[SpanDraft],
    source_target: Mapping[str, Any],
) -> tuple[RiskCandidate, ...]:
    """Build the mandatory lexical and source/target-relational risk inventory.

    A compact integer interval is shipment-dependent when its inclusive cardinality equals a
    structured cargo-package or allocation quantity.  This relationship is independent of the
    surrounding language and remains detectable when OCR splits either side of the separator
    across lines.  It therefore catches package-number assertions without maintaining a list of
    words such as ``PACKAGE``, ``BAG``, or ``PALLET``.
    """

    from . import cargo_identity_derivations, shipment_totals

    lexical = risk_candidates(raw, owned)
    total_risks: list[RiskCandidate] = []
    for start, end in cargo_identity_derivations.unowned_alias_spans(raw, owned):
        lines = line_spans(raw)
        total_risks.append(
            RiskCandidate(
                risk_id=f"risk_{len(lexical) + len(total_risks) + 1:04d}",
                kind="cargo_identity_alias",
                line_id=lines[line_number_for_char(lines, start) - 1].line_id,
                byte_start=len(raw[:start].encode()),
                byte_end=len(raw[:end].encode()),
                source_text=raw[start:end],
            )
        )
    for total_match in shipment_totals.occurrences(raw):
        start, end = total_match.span("count")
        if _overlaps(start, end, owned):
            continue
        byte_start, byte_end = len(raw[:start].encode()), len(raw[:end].encode())
        if any(r.byte_start < byte_end and r.byte_end > byte_start for r in lexical):
            continue
        lines = line_spans(raw)
        total_line = lines[line_number_for_char(lines, start) - 1]
        total_risks.append(
            RiskCandidate(
                risk_id=f"risk_{len(lexical) + len(total_risks) + 1:04d}",
                kind="shipment_package_total",
                line_id=total_line.line_id,
                byte_start=byte_start,
                byte_end=byte_end,
                source_text=raw[start:end],
            )
        )
    lexical = (*lexical, *total_risks)
    quantities = _structured_package_quantities(source_target)
    if not quantities:
        return lexical
    occupied = [(row.byte_start, row.byte_end) for row in lexical]
    relational: list[tuple[int, int, str, str]] = []
    lines = line_spans(raw)
    for match in inclusive_range_surfaces(raw):
        if match.cardinality not in quantities:
            continue
        char_start, char_end = match.char_start, match.char_end
        if _overlaps(char_start, char_end, owned):
            continue
        byte_start = len(raw[:char_start].encode("utf-8"))
        byte_end = len(raw[:char_end].encode("utf-8"))
        if any(byte_start < end and start < byte_end for start, end in occupied):
            continue
        line = next(
            (row for row in lines if row.char_start <= char_start < row.char_end),
            None,
        )
        if line is None:
            raise ValueError("relational numeric range starts outside source lines")
        occupied.append((byte_start, byte_end))
        relational.append((byte_start, byte_end, line.line_id, raw[char_start:char_end]))

    rows = [
        (row.byte_start, row.byte_end, row.kind, row.line_id, row.source_text) for row in lexical
    ]
    rows.extend(
        (start, end, "relational_numeric_range", line_id, text)
        for start, end, line_id, text in relational
    )
    rows.sort(key=lambda row: (row[0], row[1], row[2]))
    return tuple(
        RiskCandidate.model_validate(
            {
                "risk_id": f"risk_{index:04d}",
                "kind": kind,
                "line_id": line_id,
                "byte_start": start,
                "byte_end": end,
                "source_text": text,
            }
        )
        for index, (start, end, kind, line_id, text) in enumerate(rows, start=1)
    )


def normalize_inclusive_quantity_ranges(
    *, raw: str, drafts: Sequence[SpanDraft], source_target: Mapping[str, Any]
) -> tuple[SpanDraft, ...]:
    """Compile a structurally linked printed integer interval as one deterministic surface.

    OCR may split ``1 - 18`` over multiple lines and the extraction model may consequently emit
    one source-only start binding plus one quantity-backed end binding.  Treating those numbers as
    independent slots is unsafe: the generated start can then disagree with the new package
    quantity.  This normalization acts only when an overlapping binding supplies exactly one
    structured package/allocation quantity path whose value equals the interval cardinality.  It
    therefore uses an existing semantic link and never guesses from coincident document numbers.
    """

    quantity_paths = _structured_package_quantity_paths(source_target)
    if not quantity_paths:
        return tuple(drafts)
    replacements: list[SpanDraft] = []
    removed_ids: set[str] = set()
    for match in inclusive_range_surfaces(raw):
        cardinality = match.cardinality
        start, end = match.char_start, match.char_end
        overlaps = tuple(
            draft
            for draft in drafts
            if draft.draft_id not in removed_ids
            and start < draft.char_end
            and draft.char_start < end
        )
        if not overlaps:
            continue
        surface_start = min(start, *(draft.char_start for draft in overlaps))
        surface_end = max(end, *(draft.char_end for draft in overlaps))
        surface_text = raw[surface_start:surface_end]
        surface_match = _inclusive_range_surface_match(surface_text)
        if surface_match is None:
            continue
        candidate_paths = {
            path
            for draft in overlaps
            for path in (*draft.target_paths, *draft.dependency_paths)
            if quantity_paths.get(path) == cardinality
        }
        if len(candidate_paths) != 1:
            continue
        path = next(iter(candidate_paths))
        endpoint_spans = (
            (match.start_start, match.start_end),
            (match.end_start, match.end_end),
        )
        if any(
            any(
                raw[offset].isalnum()
                and not any(draft.char_start <= offset < draft.char_end for draft in overlaps)
                for offset in range(endpoint_start, endpoint_end)
            )
            for endpoint_start, endpoint_end in endpoint_spans
        ):
            continue
        external_owner = any(
            draft.draft_id not in {row.draft_id for row in overlaps} and path in draft.target_paths
            for draft in drafts
        )
        canonical_targets = () if external_owner else (path,)
        already_canonical = tuple(
            draft
            for draft in overlaps
            if draft.char_start == surface_start
            and draft.char_end == surface_end
            and draft.render_mode == "deterministic_derived"
            and draft.value_kind == "integer"
            and draft.derivation == "inclusive_range_cardinality"
            and draft.target_paths == canonical_targets
            and draft.dependency_paths == (path,)
            and not draft.dependency_bindings
            and draft.render_policy == "derived_surface"
        )
        if len(already_canonical) == 1 and len(overlaps) == 1:
            continue
        semantic_owner = next(
            (draft for draft in overlaps if path in {*draft.target_paths, *draft.dependency_paths}),
            overlaps[0],
        )
        removed_ids.update(draft.draft_id for draft in overlaps)
        replacements.append(
            SpanDraft(
                draft_id=(
                    "host_inclusive_quantity_range_"
                    + sha256_bytes(f"{path}\0{surface_start}\0{surface_end}".encode())[:16]
                ),
                logical_key=f"agent:inclusive_range:{path}",
                render_mode="deterministic_derived",
                value_kind="integer",
                group_kind=semantic_owner.group_kind,
                group_key=semantic_owner.group_key,
                target_paths=canonical_targets,
                derivation="inclusive_range_cardinality",
                dependency_paths=(path,),
                dependency_bindings=(),
                char_start=surface_start,
                char_end=surface_end,
                source_text=surface_text,
                evidence_origin="derived_operational_fact",
                render_policy="derived_surface",
                rationale=(
                    "Host joined the complete printed inclusive interval to its uniquely linked "
                    "structured package/allocation quantity; its endpoint is derived from the "
                    "quantity while the observed separator and direction are preserved."
                ),
            )
        )
    if not replacements:
        return tuple(drafts)
    return merge_drafts(
        tuple(draft for draft in drafts if draft.draft_id not in removed_ids),
        replacements,
    )


def _inclusive_range_surface_match(value: str) -> InclusiveRangeSurface | None:
    return whole_inclusive_range_surface(value)


def numbered_source(raw: str) -> str:
    return "\n".join(f"{line.line_id} | {line.text}" for line in line_spans(raw))


def anchor_summary(
    raw: str,
    drafts: Sequence[SpanDraft],
    source_target: Mapping[str, Any],
) -> list[dict[str, Any]]:
    lines = line_spans(raw)

    def occurrence_summary(occurrence: SpanDraft) -> dict[str, Any]:
        line_start, line_end = line_range_for_chars(
            lines, occurrence.char_start, occurrence.char_end
        )
        return {
            "anchorBindingId": occurrence.draft_id,
            "lineStart": line_start,
            "lineEnd": line_end,
            "sourceText": occurrence.source_text,
            "evidenceOrigin": occurrence.evidence_origin,
        }

    grouped: dict[str, list[SpanDraft]] = defaultdict(list)
    for draft in drafts:
        grouped[draft.logical_key].append(draft)
    output: list[dict[str, Any]] = []
    for logical_drafts in sorted(
        grouped.values(), key=lambda rows: min(row.char_start for row in rows)
    ):
        draft = logical_drafts[0]
        output.append(
            {
                "logicalKey": draft.logical_key,
                "renderMode": draft.render_mode,
                "valueKind": draft.value_kind,
                "groupKind": draft.group_kind,
                "groupKey": draft.group_key,
                "targetPaths": draft.target_paths,
                "targetRelationship": target_path_relationship(source_target, draft.target_paths),
                "independentTargetFactComponents": target_fact_components(
                    source_target, draft.target_paths
                ),
                "renderPolicy": draft.render_policy,
                "occurrences": tuple(
                    occurrence_summary(occurrence) for occurrence in logical_drafts
                ),
            }
        )
    return output


def _agent_logical_key(value: str) -> str:
    return value if value.startswith(("anchor:", "agent:")) else "agent:" + value


def _exact_offsets(value: str, needle: str) -> tuple[int, ...]:
    offsets: list[int] = []
    cursor = 0
    while True:
        found = value.find(needle, cursor)
        if found < 0:
            return tuple(offsets)
        offsets.append(found)
        # Exact occurrences can overlap.  For example, ``C.C.`` occurs twice in
        # ``C.C.C.`` and the second occurrence is the only non-overlapping
        # source span when the first ``C`` is already a temperature-unit slot.
        cursor = found + 1


def _whitespace_equivalent_offsets(value: str, needle: str) -> tuple[tuple[int, int], ...]:
    """Locate exact-character spans after deleting whitespace on both sides.

    This deliberately does not normalize case, punctuation, or any non-whitespace character.
    The returned spans always point back to the original source bytes.
    """

    compact_needle = "".join(character for character in needle if not character.isspace())
    if not compact_needle:
        return ()
    compact_value: list[str] = []
    source_indexes: list[int] = []
    for index, character in enumerate(value):
        if character.isspace():
            continue
        compact_value.append(character)
        source_indexes.append(index)
    compact_source = "".join(compact_value)
    output: list[tuple[int, int]] = []
    for offset in _exact_offsets(compact_source, compact_needle):
        start = source_indexes[offset]
        end = source_indexes[offset + len(compact_needle) - 1] + 1
        if value[start:end] != needle:
            output.append((start, end))
    return tuple(output)


def is_token_bounded_surface_span(raw: str, start: int, end: int) -> bool:
    """Return whether a candidate is not embedded in a larger alphanumeric token."""

    if not 0 <= start < end <= len(raw):
        raise ValueError("candidate surface span is outside source bounds")
    starts_inside_token = start > 0 and raw[start - 1].isalnum() and raw[start].isalnum()
    ends_inside_token = end < len(raw) and raw[end - 1].isalnum() and raw[end].isalnum()
    return not (starts_inside_token or ends_inside_token)


def _occurrence_resolution_error(
    *,
    raw: str,
    occurrence: AgentOccurrence,
    lines: Sequence[LineSpan],
    range_start: int,
    range_end: int,
    range_offsets: Sequence[int],
    global_offsets: Sequence[int],
) -> ValueError:
    candidate_ranges = tuple(
        f"{start_line}-{end_line}"
        for offset in global_offsets
        for start_line, end_line in (
            line_range_for_chars(
                lines,
                offset,
                offset + len(occurrence.source_text),
            ),
        )
    )
    return ValueError(
        "exact quote resolution failed; "
        f"declaredRange={occurrence.line_start}-{occurrence.line_end}; "
        f"occurrenceIndex={occurrence.occurrence_index}; "
        f"proposedSourceText={json.dumps(occurrence.source_text, ensure_ascii=False)}; "
        f"declaredRangeText={json.dumps(raw[range_start:range_end], ensure_ascii=False)}; "
        f"exactMatchesInDeclaredRange={len(range_offsets)}; "
        f"exactMatchesInDocument={len(global_offsets)}; "
        f"documentMatchRanges={json.dumps(candidate_ranges, ensure_ascii=False)}. "
        "Copy source_text byte-for-byte from the declared inclusive line range. A multi-line "
        "quote must include every intervening character and line prefix; otherwise emit "
        "separate disjoint occurrences."
    )


def _resolve_occurrence(
    *,
    raw: str,
    occurrence: AgentOccurrence,
    line_rows: Sequence[LineSpan] | None = None,
) -> tuple[int, int]:
    """Resolve an exact quote, using line scope first and global uniqueness second.

    A single exact match inside the declared line range is already unambiguous, so a redundant
    non-zero occurrence index cannot redirect it. If the model copied the exact globally unique
    source value but misstated its line, global uniqueness still proves one safe span. Ambiguous
    quotes always fail closed.
    """

    resolved_lines = tuple(line_rows) if line_rows is not None else line_spans(raw)
    lines = {line.number: line for line in resolved_lines}
    start_number = int(occurrence.line_start[1:])
    end_number = int(occurrence.line_end[1:])
    if start_number not in lines or end_number not in lines:
        raise ValueError("exact occurrence line range is outside source")
    range_start = lines[start_number].char_start
    range_end = lines[end_number].char_end
    region = raw[range_start:range_end]
    offsets = _exact_offsets(region, occurrence.source_text)
    if len(offsets) == 1:
        start = range_start + offsets[0]
        return start, start + len(occurrence.source_text)
    if occurrence.occurrence_index < len(offsets):
        start = range_start + offsets[occurrence.occurrence_index]
        return start, start + len(occurrence.source_text)
    whitespace_offsets = _whitespace_equivalent_offsets(region, occurrence.source_text)
    if len(whitespace_offsets) == 1 and occurrence.occurrence_index == 0:
        start, end = whitespace_offsets[0]
        return range_start + start, range_start + end
    # A model can preserve every character position yet substitute a Unicode case/diacritic
    # variant while copying (for example Turkish dotted capital I versus ASCII capital I).  The
    # declared range itself is a deterministic selector only when it covers the complete proposed
    # surface and the strings differ solely by case/combining marks.  In that narrow case, retain
    # the source bytes instead of paying for another full structured-output retry.
    proposed_skeleton = "".join(
        character
        for character in unicodedata.normalize("NFKD", occurrence.source_text).casefold()
        if unicodedata.category(character) != "Mn"
    )
    region_skeleton = "".join(
        character
        for character in unicodedata.normalize("NFKD", region).casefold()
        if unicodedata.category(character) != "Mn"
    )
    if (
        occurrence.occurrence_index == 0
        and len(region) == len(occurrence.source_text)
        and region != occurrence.source_text
        and region_skeleton == proposed_skeleton
    ):
        return range_start, range_end
    global_offsets = _exact_offsets(raw, occurrence.source_text)
    if len(global_offsets) != 1:
        raise _occurrence_resolution_error(
            raw=raw,
            occurrence=occurrence,
            lines=resolved_lines,
            range_start=range_start,
            range_end=range_end,
            range_offsets=offsets,
            global_offsets=global_offsets,
        )
    start = global_offsets[0]
    return start, start + len(occurrence.source_text)


def _resolve_unique_local_phone_typo(
    *,
    raw: str,
    proposal: AgentBindingProposal,
    occurrence: AgentOccurrence,
    lines: Sequence[LineSpan],
) -> tuple[int, int] | None:
    """Recover one copied phone-character error only when its declared line proves the span."""

    if (
        proposal.render_mode != "deterministic_auxiliary"
        or proposal.value_kind != "phone"
        or proposal.target_paths
        or occurrence.occurrence_index != 0
        or occurrence.line_start != occurrence.line_end
    ):
        return None
    line_number = int(occurrence.line_start[1:])
    line = next((row for row in lines if row.number == line_number), None)
    if line is None:
        return None
    proposed_digits = "".join(
        character for character in occurrence.source_text if character.isdigit()
    )
    if len(proposed_digits) < 7:
        return None
    candidates: list[tuple[int, int]] = []
    region = raw[line.char_start : line.char_end]
    for match in _PHONE.finditer(region):
        candidate_digits = "".join(character for character in match.group(0) if character.isdigit())
        if candidate_digits == proposed_digits or not _edit_distance_at_most_one(
            candidate_digits, proposed_digits
        ):
            continue
        candidates.append((line.char_start + match.start(), line.char_start + match.end()))
    return candidates[0] if len(candidates) == 1 else None


def _resolve_unique_labeled_auxiliary_organization_typo(
    *,
    raw: str,
    proposal: AgentBindingProposal,
    occurrence: AgentOccurrence,
    lines: Sequence[LineSpan],
) -> tuple[int, int] | None:
    """Recover one copied organization-character error from an exact labeled line."""

    if (
        proposal.render_mode != "deterministic_auxiliary"
        or proposal.value_kind != "organization"
        or proposal.target_paths
        or proposal.dependency_paths
        or proposal.dependency_bindings
        or occurrence.occurrence_index != 0
        or occurrence.line_start != occurrence.line_end
    ):
        return None
    line_number = int(occurrence.line_start[1:])
    line = next((row for row in lines if row.number == line_number), None)
    if line is None:
        return None
    match = _LABELED_AUXILIARY_ORGANIZATION.fullmatch(line.text)
    if match is None:
        return None
    candidate = match.group("value")
    candidate_normalized = _normalized_surface(candidate)
    proposed_normalized = _normalized_surface(occurrence.source_text.strip())
    if (
        len(candidate_normalized) < 5
        or candidate_normalized == proposed_normalized
        or not _edit_distance_at_most_one(candidate_normalized, proposed_normalized)
    ):
        return None
    return (
        line.char_start + match.start("value"),
        line.char_start + match.end("value"),
    )


def _resolve_unique_adjacent_exact_target_occurrence(
    *,
    raw: str,
    proposal: AgentBindingProposal,
    occurrence: AgentOccurrence,
    lines: Sequence[LineSpan],
) -> tuple[int, int] | None:
    """Repair a one-line selector shift only when local source bytes prove one exact span."""

    if (
        not proposal.target_paths
        or occurrence.occurrence_index != 0
        or occurrence.line_start != occurrence.line_end
    ):
        return None
    by_number = {line.number: line for line in lines}
    line_number = int(occurrence.line_start[1:])
    neighboring = tuple(
        by_number[number] for number in (line_number - 1, line_number + 1) if number in by_number
    )
    matches = tuple(
        (line.char_start + offset, line.char_start + offset + len(occurrence.source_text))
        for line in neighboring
        for offset in _exact_offsets(raw[line.char_start : line.char_end], occurrence.source_text)
    )
    return matches[0] if len(matches) == 1 else None


def _resolve_unique_local_numeric_format(
    *,
    raw: str,
    proposal: AgentBindingProposal,
    occurrence: AgentOccurrence,
    lines: Sequence[LineSpan],
) -> tuple[int, int] | None:
    """Retain the one numerically equivalent source token in an exact declared line."""

    if (
        not proposal.target_paths
        or proposal.value_kind not in {"integer", "decimal_measurement", "temperature"}
        or occurrence.occurrence_index != 0
        or occurrence.line_start != occurrence.line_end
    ):
        return None
    proposed_values = _decimal_candidates(
        occurrence.source_text,
        integer=proposal.value_kind == "integer",
    )
    if len(proposed_values) != 1:
        return None
    line_number = int(occurrence.line_start[1:])
    line = next((row for row in lines if row.number == line_number), None)
    if line is None:
        return None
    relative = _matching_numeric_span(
        raw[line.char_start : line.char_end],
        next(iter(proposed_values)),
        integer=proposal.value_kind == "integer",
    )
    if relative is None:
        return None
    return line.char_start + relative[0], line.char_start + relative[1]


def _resolve_unique_wrapped_numeric_suffix(
    *,
    proposal: AgentBindingProposal,
    occurrence: AgentOccurrence,
    lines: Sequence[LineSpan],
) -> tuple[int, int] | None:
    """Recover a numeric suffix split after a slash at an OCR line boundary."""

    if (
        len(proposal.target_paths) != 1
        or proposal.value_kind not in {"integer", "decimal_measurement", "temperature"}
        or occurrence.occurrence_index != 0
        or occurrence.line_start != occurrence.line_end
        or re.fullmatch(r"\s*[0-9][0-9 ]*\s*", occurrence.source_text) is None
    ):
        return None
    integer = proposal.value_kind == "integer"
    proposed_values = _decimal_candidates(occurrence.source_text, integer=integer)
    if len(proposed_values) != 1:
        return None
    line_number = int(occurrence.line_start[1:])
    by_number = {line.number: line for line in lines}
    line = by_number.get(line_number)
    previous = by_number.get(line_number - 1)
    if line is None or previous is None:
        return None
    proposed_digits = "".join(
        character for character in occurrence.source_text if character.isdigit()
    )
    candidates: list[tuple[int, int]] = []
    for match in re.finditer(r"(?<![0-9])[0-9]+(?![0-9])", line.text):
        if line.text[: match.start()].strip():
            continue
        suffix = match.group(0)
        if len(suffix) >= len(proposed_digits) or not proposed_digits.endswith(suffix):
            continue
        prefix = proposed_digits[: -len(suffix)]
        if re.search(r"/\s*" + re.escape(prefix) + r"\s*$", previous.text) is None:
            continue
        candidates.append((line.char_start + match.start(), line.char_start + match.end()))
    return candidates[0] if len(candidates) == 1 else None


def _is_redundant_unresolved_equipment_projection(
    *,
    proposal: AgentBindingProposal,
    occurrence: AgentOccurrence,
    resolved: Sequence[tuple[AgentBindingProposal, AgentOccurrence, int, int, str | None]],
) -> bool:
    """Prove that an invalid source-only token is already inside a target surface."""

    if (
        proposal.render_mode != "deterministic_auxiliary"
        or proposal.value_kind != "equipment"
        or proposal.target_paths
    ):
        return False
    projected = _normalized_surface(occurrence.source_text)
    if not projected:
        return False
    occurrence_start = int(occurrence.line_start[1:])
    occurrence_end = int(occurrence.line_end[1:])
    return any(
        other_proposal is not proposal
        and other_proposal.value_kind == "equipment"
        and other_proposal.group_key == proposal.group_key
        and other_proposal.render_mode in {"target_binding", "deterministic_derived"}
        and bool(other_proposal.target_paths)
        and int(other_occurrence.line_start[1:]) <= occurrence_end
        and occurrence_start <= int(other_occurrence.line_end[1:])
        and projected in _normalized_surface(other_occurrence.source_text)
        for other_proposal, other_occurrence, _char_start, _char_end, _note in resolved
    )


def resolve_agent_proposals(
    *, raw: str, proposals: Sequence[AgentBindingProposal]
) -> tuple[SpanDraft, ...]:
    lines = line_spans(raw)
    by_number = {line.number: line for line in lines}
    resolved: list[tuple[AgentBindingProposal, AgentOccurrence, int, int, str | None]] = []
    resolution_errors: list[str] = []
    unresolved_proposal_errors: list[tuple[AgentBindingProposal, AgentOccurrence, ValueError]] = []
    for proposal in proposals:
        proposal_resolved: list[
            tuple[AgentBindingProposal, AgentOccurrence, int, int, str | None]
        ] = []
        proposal_errors: list[tuple[AgentOccurrence, ValueError]] = []
        for occurrence in proposal.occurrences:
            start_number = int(occurrence.line_start[1:])
            end_number = int(occurrence.line_end[1:])
            if start_number not in by_number or end_number not in by_number:
                resolution_errors.append(
                    f"{proposal.logical_key} {occurrence.line_start}-{occurrence.line_end}: "
                    "declared line range is outside source"
                )
                continue
            normalization_note: str | None = None
            try:
                char_start, char_end = _resolve_occurrence(
                    raw=raw, occurrence=occurrence, line_rows=lines
                )
            except ValueError as resolve_error:
                recovered = _resolve_unique_adjacent_exact_target_occurrence(
                    raw=raw,
                    proposal=proposal,
                    occurrence=occurrence,
                    lines=lines,
                )
                if recovered is not None:
                    normalization_note = (
                        " Host relocated a one-line selector shift to the unique exact target "
                        "surface in the immediately adjacent source lines."
                    )
                if recovered is None:
                    recovered = _resolve_unique_local_numeric_format(
                        raw=raw,
                        proposal=proposal,
                        occurrence=occurrence,
                        lines=lines,
                    )
                    if recovered is not None:
                        normalization_note = (
                            " Host retained the unique numerically equivalent source token in "
                            "the exact declared line."
                        )
                if recovered is None:
                    recovered = _resolve_unique_local_phone_typo(
                        raw=raw,
                        proposal=proposal,
                        occurrence=occurrence,
                        lines=lines,
                    )
                    if recovered is not None:
                        normalization_note = (
                            " Host localized a one-character phone transcription error to the "
                            "unique typed value in its declared source line."
                        )
                if recovered is None:
                    recovered = _resolve_unique_wrapped_numeric_suffix(
                        proposal=proposal,
                        occurrence=occurrence,
                        lines=lines,
                    )
                    if recovered is not None:
                        normalization_note = (
                            " Host retained the unique numeric suffix after proving the omitted "
                            "prefix is split immediately before the OCR line boundary."
                        )
                if recovered is None:
                    recovered = _resolve_unique_labeled_auxiliary_organization_typo(
                        raw=raw,
                        proposal=proposal,
                        occurrence=occurrence,
                        lines=lines,
                    )
                    if recovered is not None:
                        normalization_note = (
                            " Host localized a one-character labeled organization "
                            "transcription error to the unique value in its declared source line."
                        )
                if recovered is None:
                    proposal_errors.append((occurrence, resolve_error))
                    continue
                char_start, char_end = recovered
            resolved_source_text = raw[char_start:char_end]
            if (
                normalization_note is None
                and resolved_source_text != occurrence.source_text
                and "".join(
                    character for character in resolved_source_text if not character.isspace()
                )
                == "".join(
                    character for character in occurrence.source_text if not character.isspace()
                )
            ):
                normalization_note = (
                    " Host retained the unique exact source span after proving "
                    "whitespace-only equivalence."
                )
            proposal_resolved.append(
                (proposal, occurrence, char_start, char_end, normalization_note)
            )
        dropped_unresolved_occurrences = 0
        for occurrence, resolution_error in proposal_errors:
            global_offsets = set(_exact_offsets(raw, occurrence.source_text))
            resolved_offsets = {
                char_start
                for (
                    _proposal,
                    resolved_occurrence,
                    char_start,
                    _char_end,
                    _note,
                ) in proposal_resolved
                if resolved_occurrence.source_text == occurrence.source_text
            }
            if global_offsets and resolved_offsets == global_offsets:
                # The invalid declaration cannot identify another physical slot: the same
                # proposal already selected every exact source occurrence of this value.  Drop
                # only that provably redundant hallucinated duplicate.
                dropped_unresolved_occurrences += 1
                continue
            if (
                proposal.render_mode in {"target_binding", "carrier_static"}
                and bool(proposal.target_paths)
                and any(
                    resolved_occurrence.source_text == occurrence.source_text
                    for (
                        _proposal,
                        resolved_occurrence,
                        _char_start,
                        _char_end,
                        _note,
                    ) in proposal_resolved
                )
            ):
                # The malformed handle contributes no source bytes or new semantic value.  Keep
                # the exact resolved occurrences so the independent whole-document critic can
                # still audit whether another physical repeat is missing, instead of paying for a
                # complete compiler retry merely to delete this unusable selector.
                dropped_unresolved_occurrences += 1
                continue
            unresolved_proposal_errors.append((proposal, occurrence, resolution_error))
        if dropped_unresolved_occurrences:
            audit_note = (
                " Host discarded "
                f"{dropped_unresolved_occurrences} unresolved duplicate occurrence declaration(s) "
                "that contributed no source bytes; the independent completeness critic remains "
                "responsible for detecting omitted physical repeats."
            )
            proposal_resolved = [
                (
                    resolved_proposal,
                    resolved_occurrence,
                    char_start,
                    char_end,
                    (normalization_note or "") + audit_note,
                )
                for (
                    resolved_proposal,
                    resolved_occurrence,
                    char_start,
                    char_end,
                    normalization_note,
                ) in proposal_resolved
            ]
        resolved.extend(proposal_resolved)
    for proposal, occurrence, resolution_error in unresolved_proposal_errors:
        if _is_redundant_unresolved_equipment_projection(
            proposal=proposal,
            occurrence=occurrence,
            resolved=resolved,
        ):
            continue
        resolution_errors.append(
            f"{proposal.logical_key} {occurrence.line_start}-{occurrence.line_end}: "
            f"{resolution_error}"
        )
    if resolution_errors:
        raise ValueError(
            "agent source texts cannot be resolved unambiguously; correct every listed "
            "occurrence in one complete replacement:\n- " + "\n- ".join(resolution_errors)
        )

    drafts: list[SpanDraft] = []
    for proposal, occurrence, char_start, char_end, normalization_note in resolved:
        resolved_source_text = raw[char_start:char_end]
        if _PAGE_HEADER.search(resolved_source_text):
            raise ValueError("agent binding cannot own page marker text")
        target_paths = tuple(sorted(proposal.target_paths))
        policy = _policy_for_agent_binding(proposal)
        if proposal.target_paths and proposal.render_mode in {
            "target_binding",
            "carrier_static",
        }:
            group_kind, group_key = _canonical_target_group(target_paths)
            value_kind = _value_kind(target_paths, policy)
            policy = _render_policy_for_value_kind(value_kind, proposal.render_mode)
            logical_key = "anchor:" + "|".join(target_paths)
        else:
            group_kind, group_key = proposal.group_kind, proposal.group_key
            value_kind = proposal.value_kind
            logical_key = _agent_logical_key(proposal.logical_key)
        drafts.append(
            SpanDraft(
                draft_id=(
                    "agent_binding_"
                    + sha256_bytes(
                        (
                            proposal.logical_key
                            + "\0"
                            + occurrence.line_start
                            + "\0"
                            + occurrence.line_end
                            + "\0"
                            + resolved_source_text
                            + "\0"
                            + str(occurrence.occurrence_index)
                        ).encode("utf-8")
                    )[:16]
                ),
                logical_key=logical_key,
                render_mode=proposal.render_mode,
                value_kind=value_kind,
                group_kind=group_kind,
                group_key=group_key,
                target_paths=target_paths,
                derivation=proposal.derivation,
                dependency_paths=proposal.dependency_paths,
                dependency_bindings=tuple(
                    _agent_logical_key(value) for value in proposal.dependency_bindings
                ),
                char_start=char_start,
                char_end=char_end,
                source_text=resolved_source_text,
                evidence_origin=(
                    "derived_operational_fact"
                    if proposal.render_mode == "deterministic_derived"
                    else "host_verified_agent_proposal"
                ),
                render_policy=policy,
                rationale=(
                    proposal.rationale
                    + (
                        normalization_note
                        if normalization_note is not None
                        else (
                            " Host retained exact declared source bytes after proving Unicode "
                            "case and diacritic equivalence."
                            if resolved_source_text != occurrence.source_text
                            else ""
                        )
                    )
                ),
            )
        )
    return tuple(drafts)


def resolve_agent_proposal_inventory(
    *, raw: str, proposals: Sequence[AgentBindingProposal]
) -> tuple[tuple[SpanDraft, ...], tuple[str, ...]]:
    """Resolve every valid occurrence while explicitly inventorying rejected edit handles.

    This is diagnostic state for a local repair, never an accepted compiler state. Resolving each
    occurrence independently preserves the valid ownership context of a partially malformed
    binding without suppressing or accepting any invalid occurrence.
    """

    resolved: list[SpanDraft] = []
    rejected: list[str] = []
    for proposal in proposals:
        for occurrence in proposal.occurrences:
            singleton = proposal.model_copy(update={"occurrences": (occurrence,)})
            try:
                resolved.extend(resolve_agent_proposals(raw=raw, proposals=(singleton,)))
            except ValueError:
                rejected.append(
                    f"{proposal.logical_key} {occurrence.line_start}-{occurrence.line_end}"
                )
    return tuple(resolved), tuple(rejected)


def _resolve_target_path(source_target: Mapping[str, Any], path: str) -> Any:
    if not _TARGET_PATH.fullmatch(path):
        raise ValueError(f"invalid target path syntax: {path}")
    matches = list(_PATH_TOKEN.finditer(path))
    current: Any = source_target
    for match in matches:
        key, index = match.groups()
        if key is not None:
            if not isinstance(current, Mapping) or key not in current:
                raise ValueError(f"target path does not exist in source label: {path}")
            current = current[key]
        else:
            offset = int(index)
            if not isinstance(current, list) or offset >= len(current):
                raise ValueError(f"target path index does not exist in source label: {path}")
            current = current[offset]
    return current


def target_path_relationship(source_target: Mapping[str, Any], paths: Sequence[str]) -> str:
    """Classify how one physical source surface relates to its target paths.

    Multiple paths with byte-identical canonical values form an explicit equality
    constraint for descendant pairing. Unequal values require one composite
    renderer (normally an agent residual) or disjoint replacement spans; they
    cannot be rendered as one direct target binding.
    """

    if not paths:
        return "none"
    values = tuple(_resolve_target_path(source_target, path) for path in paths)
    if len(values) == 1:
        return "single_target"
    encoded = tuple(canonical_json_bytes(value) for value in values)
    if len(set(encoded)) == 1:
        return "shared_value_equality"
    return "composite_target_surface"


_SINGLE_PRINTED_HS_LINE = re.compile(
    r"(?im)^\s*H\.?S\.?\s+CODES?\s*:\s*(?P<code>\d[\d. -]{4,14}\d)\s*$"
)


def single_printed_shared_hs_paths(raw: str, source_target: Mapping[str, Any]) -> tuple[str, ...]:
    """Return HS leaves proven to share one explicit source field, or none.

    The evidence rule does not infer scope from equal values alone. Other cargo
    layouts remain under ordinary compilation and independent review.
    """
    patch = source_target.get("documentPatch")
    if not isinstance(patch, Mapping):
        return ()
    groups = patch.get("cargoGroups")
    if not isinstance(groups, list) or len(groups) < 2:
        return ()
    codes: list[str] = []
    for group in groups:
        if not isinstance(group, Mapping):
            return ()
        values = group.get("hsCodes")
        if not isinstance(values, list) or len(values) != 1 or not isinstance(values[0], str):
            return ()
        codes.append(re.sub(r"\D", "", values[0]))
    if len(set(codes)) != 1 or not 6 <= len(codes[0]) <= 10:
        return ()
    printed = tuple(_SINGLE_PRINTED_HS_LINE.finditer(raw))
    if len(printed) != 1 or re.sub(r"\D", "", printed[0].group("code")) != codes[0]:
        return ()
    # Another occurrence may be a cargo-row code, in which case the caption is
    # not sufficient evidence that every group is represented by one surface.
    if raw.count(printed[0].group("code")) != 1:
        return ()
    return tuple(f"documentPatch.cargoGroups[{index}].hsCodes[0]" for index in range(len(groups)))


def validate_compiler_extraction_dates(
    *,
    raw: str,
    source_target: Mapping[str, Any],
    drafts: Sequence[SpanDraft],
    semantic_only_target_facts: Sequence[SemanticOnlyTargetFact],
) -> None:
    """Require every labelled document date to have role-correct OCR ownership."""
    patch = source_target.get("documentPatch")
    if not isinstance(patch, Mapping):
        raise ValueError("source target lacks documentPatch")
    semantic = {fact.target_path for fact in semantic_only_target_facts}
    for field in _EXTRACTION_DATE_FIELDS:
        path = f"documentPatch.{field}"
        if patch.get(field) is None:
            continue
        if path in semantic:
            raise ValueError(f"document extraction date is semantic-only: {path}")
        owners = [draft for draft in drafts if path in draft.target_paths]
        if not owners:
            raise ValueError(f"document extraction date lacks an OCR binding: {path}")
        if any(date_immediately_under_declared_value(raw, draft.char_start) for draft in owners):
            raise ValueError(f"document extraction date is bound under Declared Value: {path}")


def validate_compiled_extraction_dates(
    *, raw: str, source_target: Mapping[str, Any], template: CertifiedSemanticTemplate
) -> None:
    """Reject old catalogs that would carry an ungrounded date into synthesis."""
    patch = source_target.get("documentPatch")
    if not isinstance(patch, Mapping):
        raise ValueError("source target lacks documentPatch")
    semantic = {fact.target_path for fact in template.semantic_only_target_facts}
    encoded = raw.encode("utf-8")
    for field in _EXTRACTION_DATE_FIELDS:
        path = f"documentPatch.{field}"
        if patch.get(field) is None:
            continue
        if path in semantic:
            raise ValueError(f"compiled document extraction date is semantic-only: {path}")
        owners = [binding for binding in template.bindings if path in binding.target_paths]
        if not owners:
            raise ValueError(f"compiled document extraction date lacks an OCR binding: {path}")
        for binding in owners:
            for slot in binding.occurrences:
                char_start = len(encoded[: slot.byte_start].decode("utf-8"))
                if date_immediately_under_declared_value(raw, char_start):
                    raise ValueError(
                        f"compiled document extraction date is bound under Declared Value: {path}"
                    )


def validate_single_printed_hs_scope(
    *,
    raw: str,
    source_target: Mapping[str, Any],
    drafts: Sequence[SpanDraft],
    semantic_only_target_facts: Sequence[SemanticOnlyTargetFact],
) -> None:
    """Reject compiler drafts that lose an unambiguous shared HS dependency."""

    paths = single_printed_shared_hs_paths(raw, source_target)
    if not paths:
        return
    printed = next(_SINGLE_PRINTED_HS_LINE.finditer(raw))
    semantic = {fact.target_path for fact in semantic_only_target_facts}
    owners = {
        path: {draft.logical_key for draft in drafts if path in draft.target_paths}
        for path in paths
    }
    owner_keys = {next(iter(keys)) for keys in owners.values() if len(keys) == 1}
    valid = (
        not (semantic & set(paths))
        and all(len(keys) == 1 for keys in owners.values())
        and len(owner_keys) == 1
        and any(
            draft.logical_key in owner_keys
            and draft.render_mode == "target_binding"
            and set(paths) <= set(draft.target_paths)
            and draft.char_start <= printed.start("code")
            and draft.char_end >= printed.end("code")
            for draft in drafts
        )
    )
    if not valid:
        raise ValueError(
            "one printed HS-code field and equal cargo-group labels require one "
            "shared-value target binding owning every group HS path; independent or "
            "semantic-only HS leaves would create ungrounded synthetic labels"
        )


def validate_compiled_single_printed_hs_scope(
    *, raw: str, source_target: Mapping[str, Any], template: CertifiedSemanticTemplate
) -> None:
    """Fail synthesis preflight for old catalogs with independently mutable HS leaves."""

    paths = single_printed_shared_hs_paths(raw, source_target)
    if not paths:
        return
    semantic = {fact.target_path for fact in template.semantic_only_target_facts}
    owners = [
        binding for binding in template.bindings if set(paths).intersection(binding.target_paths)
    ]
    if (
        semantic.intersection(paths)
        or len(owners) != 1
        or not set(paths) <= set(owners[0].target_paths)
        or owners[0].target_relationship != "shared_value_equality"
        or owners[0].realization.requires_agent
    ):
        raise ValueError(
            "compiled template loses the one-printed-field shared HS scope; "
            "recompile or use a corrected catalog before synthesis"
        )


_GLOBAL_SET_TEMP = re.compile(r"\bSET TEMP:\s*(?P<number>[+-]?\d+(?:[.,]\d+)?)", re.I)
_GLOBAL_CARRYING_TEMP = re.compile(
    r"\bCARRYING TEMPERATURE:\s*(?P<number>[+-]?\d+(?:[.,]\d+)?)", re.I
)
_SIGNED_TEMPERATURE_WORD = re.compile(
    r"\b(?P<word>PLUS|MINUS)\s+(?P<number>\d+(?:[.,]\d+)?)\s+DEG['\u2019]?\s*C\b", re.I
)
_REPEATED_CARGO_CARRYING = re.compile(
    r"\bCARRYING\s+TEMPERATURE\s+OF\s+"
    r"(?P<word>PLUS|MINUS)\s+(?P<number>\d+(?:[.,]\d+)?)\s+"
    r"DEG['\u2019]?\s*C\b",
    re.I,
)


def repeated_cargo_temperature_contract(
    raw: str, source_target: Mapping[str, Any]
) -> dict[str, tuple[str, ...]]:
    """Find a printed global setting over one repeated commodity and its reefers.

    Exact repeated goods, package loads, one-to-one allocation, reefer equipment,
    and one unqualified carrying instruction are required. The function only
    recognizes review-worthy evidence; it never adds missing source labels.
    """
    from document_ocr.synthesis.container_semantics import review_source_equipment_surface
    from document_ocr.synthesis.semantic_completion_pipeline import _group_allocations

    patch = source_target.get("documentPatch")
    if not isinstance(patch, Mapping):
        return {}
    groups, packages, containers = (
        patch.get("cargoGroups"),
        patch.get("cargoPackages"),
        patch.get("containers"),
    )
    if (
        not isinstance(groups, list)
        or not isinstance(packages, list)
        or not isinstance(containers, list)
        or len(groups) < 2
        or len(groups) != len(packages)
        or len(groups) != len(containers)
    ):
        return {}
    for field in ("description", "hsCodes", "handlingInstructions", "netWeight", "volume"):
        if len({canonical_json_bytes(group.get(field)) for group in groups}) != 1:
            return {}
    if any(
        len(group.get("hsCodes") or ()) != 1 or len(group.get("handlingInstructions") or ()) != 1
        for group in groups
    ):
        return {}
    if (
        len({package.get("quantity") for package in packages}) != 1
        or len({package.get("typeCategory") for package in packages}) != 1
    ):
        return {}
    if any(
        review_source_equipment_surface(
            container.get("typeDescription"),
            temperature_present="temperatureSetpoint" in container,
        ).type_category
        != "REFRIGERATED"
        for container in containers
    ):
        return {}
    allocation = _group_allocations(patch)
    owned = [allocation.get(group["groupId"], ()) for group in groups]
    if any(len(numbers) != 1 for numbers in owned) or {numbers[0] for numbers in owned} != {
        container["containerNumber"] for container in containers
    }:
        return {}
    printed = tuple(_REPEATED_CARGO_CARRYING.finditer(raw))
    if len(printed) != 1:
        return {}
    observed = Decimal(printed[0]["number"].replace(",", "."))
    if printed[0]["word"].upper() == "MINUS":
        observed = -observed
    settings = [container.get("temperatureSetpoint") for container in containers]
    if any(
        isinstance(setting, Mapping)
        and (setting.get("unit") != "celsius" or Decimal(str(setting["value"])) != observed)
        for setting in settings
    ):
        raise ValueError("repeated reefer carrying sentence contradicts a source setpoint label")
    count = len(containers)
    return {
        "value": tuple(
            f"documentPatch.containers[{index}].temperatureSetpoint.value" for index in range(count)
        ),
        "unit": tuple(
            f"documentPatch.containers[{index}].temperatureSetpoint.unit" for index in range(count)
        ),
        "description": tuple(
            f"documentPatch.cargoGroups[{index}].description" for index in range(count)
        ),
        "hs": tuple(f"documentPatch.cargoGroups[{index}].hsCodes[0]" for index in range(count)),
        "handling": tuple(
            f"documentPatch.cargoGroups[{index}].handlingInstructions[0]" for index in range(count)
        ),
    }


def validate_repeated_cargo_temperature_scope(
    *, raw: str, source_target: Mapping[str, Any], drafts: Sequence[SpanDraft]
) -> None:
    contract = repeated_cargo_temperature_contract(raw, source_target)
    if not contract:
        return
    containers = source_target["documentPatch"]["containers"]
    if any("temperatureSetpoint" not in container for container in containers):
        raise ValueError(
            "repeated reefer commodity has one shared printed carrying temperature "
            "but incomplete container setpoint labels"
        )
    for name, paths in contract.items():
        owners = [draft for draft in drafts if set(paths).intersection(draft.target_paths)]
        if (
            not owners
            or len({draft.logical_key for draft in owners}) != 1
            or any(draft.render_mode != "target_binding" for draft in owners)
            or any(set(draft.target_paths) != set(paths) for draft in owners)
        ):
            raise ValueError(
                "repeated reefer commodity requires one shared printed " + name + " binding"
            )


def validate_compiled_repeated_cargo_temperature_scope(
    *, raw: str, source_target: Mapping[str, Any], template: CertifiedSemanticTemplate
) -> None:
    contract = repeated_cargo_temperature_contract(raw, source_target)
    if not contract:
        return
    containers = source_target["documentPatch"]["containers"]
    if any("temperatureSetpoint" not in container for container in containers):
        raise ValueError(
            "repeated reefer commodity has one shared printed carrying temperature "
            "but incomplete container setpoint labels"
        )
    for name, paths in contract.items():
        owners = [
            binding
            for binding in template.bindings
            if set(paths).intersection(binding.target_paths)
        ]
        if (
            len(owners) != 1
            or set(owners[0].target_paths) != set(paths)
            or owners[0].target_relationship != "shared_value_equality"
            or not owners[0].realization.deterministic
        ):
            raise ValueError(
                "compiled repeated reefer commodity lost its shared " + name + " binding"
            )


def global_shared_temperature_paths(raw: str, source_target: Mapping[str, Any]) -> tuple[str, ...]:
    """Identify a one-cargo, two-global-mention setpoint shared by every reefer.

    Equal source labels alone are insufficient. Both document-wide printed
    temperature instructions must independently agree with the source labels.
    """
    patch = source_target.get("documentPatch")
    if not isinstance(patch, Mapping):
        return ()
    groups, containers = patch.get("cargoGroups"), patch.get("containers")
    if not isinstance(groups, list) or len(groups) != 1:
        return ()
    if not isinstance(containers, list) or len(containers) < 2:
        return ()
    settings = [
        c.get("temperatureSetpoint") if isinstance(c, Mapping) else None for c in containers
    ]
    if any(not isinstance(s, Mapping) or s.get("unit") != "celsius" for s in settings):
        return ()
    source_values: list[Decimal] = []
    for setting in settings:
        if not isinstance(setting, Mapping):
            return ()
        source_values.append(Decimal(str(setting["value"])))
    if any(not value.is_finite() or value != source_values[0] for value in source_values):
        return ()
    printed = (tuple(_GLOBAL_SET_TEMP.finditer(raw)), tuple(_GLOBAL_CARRYING_TEMP.finditer(raw)))
    if any(len(group) != 1 for group in printed):
        return ()
    carrying = printed[1][0]
    if not re.match(r"\s+DEGREES\s+CELSIUS\b", raw[carrying.end() :], re.I):
        return ()
    if any(Decimal(group[0]["number"].replace(",", ".")) != source_values[0] for group in printed):
        return ()
    return tuple(
        f"documentPatch.containers[{index}].temperatureSetpoint.value"
        for index in range(len(containers))
    )


def validate_global_shared_temperature_scope(
    *,
    raw: str,
    source_target: Mapping[str, Any],
    drafts: Sequence[SpanDraft],
    semantic_only_target_facts: Sequence[SemanticOnlyTargetFact],
) -> None:
    """A global printed setpoint cannot certify independently mutable labels."""
    paths = global_shared_temperature_paths(raw, source_target)
    if not paths:
        return
    mentions = (next(_GLOBAL_SET_TEMP.finditer(raw)), next(_GLOBAL_CARRYING_TEMP.finditer(raw)))
    semantic = {fact.target_path for fact in semantic_only_target_facts}
    owners = [draft for draft in drafts if set(paths).intersection(draft.target_paths)]
    keys = {draft.logical_key for draft in owners}
    if (
        semantic.intersection(paths)
        or len(keys) != 1
        or not owners
        or any(draft.render_mode != "target_binding" for draft in owners)
        or not all(set(paths) <= set(draft.target_paths) for draft in owners)
        or not all(
            any(
                draft.char_start <= match.start("number") and draft.char_end >= match.end("number")
                for draft in owners
            )
            for match in mentions
        )
    ):
        raise ValueError(
            "two global carrying-temperature surfaces require one shared-value target "
            "binding for every container setpoint; independent temperatures would be ungrounded"
        )


def validate_compiled_global_shared_temperature_scope(
    *, raw: str, source_target: Mapping[str, Any], template: CertifiedSemanticTemplate
) -> None:
    paths = global_shared_temperature_paths(raw, source_target)
    if not paths:
        return
    mentions = (next(_GLOBAL_SET_TEMP.finditer(raw)), next(_GLOBAL_CARRYING_TEMP.finditer(raw)))
    owners = [
        binding for binding in template.bindings if set(paths).intersection(binding.target_paths)
    ]
    semantic = {fact.target_path for fact in template.semantic_only_target_facts}
    if (
        semantic.intersection(paths)
        or len(owners) != 1
        or not set(paths) <= set(owners[0].target_paths)
        or owners[0].target_relationship != "shared_value_equality"
        or not owners[0].realization.deterministic
        or owners[0].realization.requires_agent
        or not all(
            any(
                slot.byte_start <= len(raw[: match.start("number")].encode("utf-8"))
                and slot.byte_end >= len(raw[: match.end("number")].encode("utf-8"))
                for slot in owners[0].occurrences
            )
            for match in mentions
        )
    ):
        raise ValueError(
            "compiled template loses the global shared-temperature scope; "
            "recompile or use a corrected catalog before synthesis"
        )


def signed_temperature_word_contract(
    raw: str, source_target: Mapping[str, Any]
) -> tuple[str, int, int] | None:
    """Locate one source sign word whose numeric owner is unambiguous."""
    patch = source_target.get("documentPatch")
    if not isinstance(patch, Mapping) or not isinstance(patch.get("containers"), list):
        return None
    owned = [
        (index, container["temperatureSetpoint"])
        for index, container in enumerate(patch["containers"])
        if isinstance(container, Mapping)
        and isinstance(container.get("temperatureSetpoint"), Mapping)
    ]
    matches = tuple(_SIGNED_TEMPERATURE_WORD.finditer(raw))
    if len(owned) != 1 or len(matches) != 1:
        return None
    index, setting = owned[0]
    if setting.get("unit") != "celsius":
        return None
    match = matches[0]
    value = Decimal(str(setting["value"]))
    printed = Decimal(match["number"].replace(",", "."))
    if match["word"].upper() == "MINUS":
        printed = -printed
    if not value.is_finite() or printed != value:
        raise ValueError("signed temperature word contradicts its source setpoint label")
    return (
        f"documentPatch.containers[{index}].temperatureSetpoint.value",
        match.start("word"),
        match.end("number"),
    )


def validate_signed_temperature_word_scope(
    *, raw: str, source_target: Mapping[str, Any], drafts: Sequence[SpanDraft]
) -> None:
    contract = signed_temperature_word_contract(raw, source_target)
    if contract is None:
        return
    path, start, end = contract
    owners = [draft for draft in drafts if path in draft.target_paths]
    if (
        len(owners) != 1
        or owners[0].render_mode != "target_binding"
        or owners[0].value_kind != "temperature"
        or owners[0].char_start != start
        or owners[0].char_end != end
    ):
        raise ValueError(
            "signed carrying-temperature word and magnitude must be one mutable "
            "temperature binding; a fixed PLUS/MINUS word can contradict generated values"
        )


def validate_compiled_signed_temperature_word_scope(
    *, raw: str, source_target: Mapping[str, Any], template: CertifiedSemanticTemplate
) -> None:
    contract = signed_temperature_word_contract(raw, source_target)
    if contract is None:
        return
    path, start, end = contract
    owners = [binding for binding in template.bindings if path in binding.target_paths]
    start_byte = len(raw[:start].encode("utf-8"))
    end_byte = len(raw[:end].encode("utf-8"))
    if (
        len(owners) != 1
        or owners[0].realization.adapter != "signed_temperature_word"
        or owners[0].realization.requires_agent
        or len(owners[0].occurrences) != 1
        or owners[0].occurrences[0].byte_start != start_byte
        or owners[0].occurrences[0].byte_end != end_byte
    ):
        raise ValueError(
            "compiled template leaves a temperature sign word fixed beside a mutable "
            "setpoint; recompile or use a corrected catalog before synthesis"
        )


def required_target_cobindings(
    source_target: Mapping[str, Any],
) -> tuple[RequiredTargetCoBinding, ...]:
    """Derive structured-label identities that one semantic binding must own together."""

    patch = source_target.get("documentPatch")
    if not isinstance(patch, Mapping):
        raise ValueError("source target lacks documentPatch")

    def collection(name: str) -> list[Any]:
        value = patch.get(name, [])
        if not isinstance(value, list):
            raise ValueError(f"documentPatch.{name} must be a list")
        return value

    containers = collection("containers")
    allocation_groups = collection("cargoAllocationGroups")
    packages = collection("cargoPackages")

    container_indexes: dict[str, int] = {}
    for index, container in enumerate(containers):
        if not isinstance(container, Mapping):
            raise ValueError(f"documentPatch.containers[{index}] must be an object")
        number = container.get("containerNumber")
        if number is None:
            continue
        if not isinstance(number, str) or not number:
            raise ValueError(f"documentPatch.containers[{index}].containerNumber is invalid")
        if number in container_indexes:
            raise ValueError(f"duplicate structured container number: {number}")
        container_indexes[number] = index

    package_indexes: dict[str, int] = {}
    for index, package in enumerate(packages):
        if not isinstance(package, Mapping):
            raise ValueError(f"documentPatch.cargoPackages[{index}] must be an object")
        package_id = package.get("packageId")
        if not isinstance(package_id, str) or not package_id:
            raise ValueError(f"documentPatch.cargoPackages[{index}].packageId is invalid")
        if package_id in package_indexes:
            raise ValueError(f"duplicate structured package ID: {package_id}")
        package_indexes[package_id] = index

    container_paths: dict[int, list[str]] = {
        index: [f"documentPatch.containers[{index}].containerNumber"]
        for index in container_indexes.values()
    }
    requirements: list[RequiredTargetCoBinding] = []
    for group_index, group in enumerate(allocation_groups):
        if not isinstance(group, Mapping):
            raise ValueError(
                f"documentPatch.cargoAllocationGroups[{group_index}] must be an object"
            )
        allocations = group.get("allocations", [])
        if not isinstance(allocations, list):
            raise ValueError(
                f"documentPatch.cargoAllocationGroups[{group_index}].allocations must be a list"
            )
        for allocation_index, allocation in enumerate(allocations):
            if not isinstance(allocation, Mapping):
                raise ValueError(
                    "documentPatch.cargoAllocationGroups"
                    f"[{group_index}].allocations[{allocation_index}] must be an object"
                )
            number = allocation.get("containerNumber")
            if number is not None:
                if not isinstance(number, str) or not number:
                    raise ValueError(f"allocation container number is invalid: {number!r}")
                if number in container_indexes:
                    container_paths[container_indexes[number]].append(
                        "documentPatch.cargoAllocationGroups"
                        f"[{group_index}].allocations[{allocation_index}].containerNumber"
                    )

        coverage = group.get("coverage")
        if coverage not in {
            "one_to_one_package_allocations",
            "single_package_level",
        }:
            continue
        package_ids = group.get("packageIds")
        if not isinstance(package_ids, list):
            continue
        if coverage == "one_to_one_package_allocations":
            allocation_package_ids = tuple(
                allocation.get("packageId") if isinstance(allocation, Mapping) else None
                for allocation in allocations
            )
            if (
                not package_ids
                or len(package_ids) != len(allocations)
                or tuple(package_ids) != allocation_package_ids
            ):
                continue
            pairs = tuple(enumerate(zip(package_ids, allocations, strict=True)))
        else:
            # Multiple allocations at one package level are summands of the package total.  Only a
            # single allocation is scalar-identical to that total and can be co-bound directly.
            if len(package_ids) != 1 or len(allocations) != 1:
                continue
            allocation = allocations[0]
            if not isinstance(allocation, Mapping) or allocation.get("packageId") is not None:
                continue
            pairs = ((0, (package_ids[0], allocation)),)
        for allocation_index, (package_id, allocation) in pairs:
            if not isinstance(package_id, str) or not isinstance(allocation, Mapping):
                continue
            if package_id not in package_indexes:
                continue
            package_index = package_indexes[package_id]
            package = packages[package_index]
            assert isinstance(package, Mapping)
            if package.get("groupId") != group.get("groupId"):
                continue
            if canonical_json_bytes(allocation.get("packageQuantity")) != canonical_json_bytes(
                package.get("quantity")
            ):
                continue
            requirements.append(
                RequiredTargetCoBinding(
                    relationship=(
                        "one_to_one_package_quantity"
                        if coverage == "one_to_one_package_allocations"
                        else "single_package_single_allocation_quantity"
                    ),
                    target_paths=(
                        "documentPatch.cargoAllocationGroups"
                        f"[{group_index}].allocations[{allocation_index}].packageQuantity",
                        f"documentPatch.cargoPackages[{package_index}].quantity",
                    ),
                )
            )

    requirements.extend(
        RequiredTargetCoBinding(
            relationship="allocation_container_identity",
            target_paths=tuple(paths),
        )
        for _index, paths in sorted(container_paths.items())
        if len(paths) > 1
    )
    return tuple(sorted(requirements, key=lambda row: row.target_paths))


def target_fact_components(
    source_target: Mapping[str, Any], paths: Sequence[str]
) -> tuple[tuple[str, ...], ...]:
    """Partition target paths into independently mutable facts.

    Host-derived co-binding paths are one fact. An explicitly co-bound field of
    identical complete concrete parties is also one fact, matching the existing
    generator identity policy. Equal isolated fields never establish identity.
    """

    component_by_path: dict[str, int] = {}
    for component_index, requirement in enumerate(required_target_cobindings(source_target)):
        for path in requirement.target_paths:
            prior = component_by_path.setdefault(path, component_index)
            if prior != component_index:
                raise ValueError(f"target path belongs to multiple co-binding components: {path}")
    if len(paths) < 2:
        return tuple((path,) for path in paths)
    party_components: dict[str, str] = {}
    parties = source_target.get("documentPatch", {}).get("parties", {})
    identity_paths: dict[bytes, list[tuple[str, str]]] = defaultdict(list)
    for path in paths:
        match = re.fullmatch(r"(documentPatch\.parties\.([A-Za-z]+)(?:\[\d+\])?)\.(.+)", path)
        if match is None or match[2] == "carrier":
            continue
        party = _resolve_target_path(source_target, match[1])
        if (
            not isinstance(party, Mapping)
            or any(
                not isinstance(party.get(field), str) or not party[field].strip()
                for field in ("name", "address")
            )
            or "sameAs" in party
        ):
            continue
        signature = canonical_json_bytes(party)
        # Origin and destination roles remain independent. A repeated name or
        # address alone cannot decide which commercial endpoint owns a party.
        primary_matches = sum(
            isinstance(parties.get(role), Mapping)
            and canonical_json_bytes(parties[role]) == signature
            for role in ("shipper", "consignee")
        )
        if primary_matches > 1:
            continue
        identity_paths[canonical_json_bytes((party, match[3]))].append((path, match[1]))
    for members in identity_paths.values():
        if len({owner for _, owner in members}) < 2:
            continue
        canonical = min(path for path, _ in members)
        party_components.update((path, canonical) for path, _ in members)
    grouped: dict[tuple[str, int | str], list[str]] = {}
    for path in paths:
        key: tuple[str, int | str]
        if path in component_by_path:
            key = ("required", component_by_path[path])
        elif path in party_components:
            key = ("concrete_party_identity", party_components[path])
        else:
            key = ("single", path)
        grouped.setdefault(key, []).append(path)
    return tuple(tuple(group) for group in grouped.values())


def normalize_target_cobindings(
    *, drafts: Sequence[SpanDraft], source_target: Mapping[str, Any]
) -> tuple[SpanDraft, ...]:
    """Merge provably identical structured facts into host-owned logical bindings.

    Accepted-label anchors are value-oriented and can assign equal package/allocation facts to
    different duplicate occurrences. Structured IDs prove which paths are the same fact. This
    normalization joins their owner groups deterministically and adds an unowned equivalent path
    when another member of its component is visibly owned.
    """

    by_key: dict[str, list[SpanDraft]] = defaultdict(list)
    path_owners: dict[str, set[str]] = defaultdict(set)
    for draft in drafts:
        by_key[draft.logical_key].append(draft)
        for path in draft.target_paths:
            path_owners[path].add(draft.logical_key)
    parent = {logical_key: logical_key for logical_key in by_key}

    def find(logical_key: str) -> str:
        root = logical_key
        while parent[root] != root:
            root = parent[root]
        while parent[logical_key] != logical_key:
            next_key = parent[logical_key]
            parent[logical_key] = root
            logical_key = next_key
        return root

    def union(left: str, right: str) -> None:
        left_root = find(left)
        right_root = find(right)
        if left_root != right_root:
            parent[max(left_root, right_root)] = min(left_root, right_root)

    active_requirements: list[tuple[RequiredTargetCoBinding, tuple[str, ...]]] = []
    for requirement in required_target_cobindings(source_target):
        required_paths = set(requirement.target_paths)
        owners = tuple(
            sorted(
                {owner for path in requirement.target_paths for owner in path_owners.get(path, ())}
            )
        )
        if not owners:
            continue
        owner_paths = {
            owner: {path for draft in by_key[owner] for path in draft.target_paths}
            for owner in owners
        }
        if any(not paths <= required_paths for paths in owner_paths.values()):
            # A value-oriented accepted anchor can contain several independently mutable facts
            # merely because their source values are currently equal. Such an owner must not
            # bridge otherwise disjoint structured identity components. Leave it unchanged so
            # the compiler must resolve the ambiguity and the relationship validator can reject
            # an incomplete repair.
            continue
        for owner in owners[1:]:
            union(owners[0], owner)
        active_requirements.append((requirement, owners))

    keys_by_root: dict[str, set[str]] = defaultdict(set)
    for logical_key in by_key:
        keys_by_root[find(logical_key)].add(logical_key)
    required_paths_by_root: dict[str, set[str]] = defaultdict(set)
    relationships_by_root: dict[str, set[str]] = defaultdict(set)
    for requirement, owners in active_requirements:
        root = find(owners[0])
        required_paths_by_root[root].update(requirement.target_paths)
        relationships_by_root[root].add(requirement.relationship)

    normalized: list[SpanDraft] = []
    for root, logical_keys in keys_by_root.items():
        component = [draft for key in logical_keys for draft in by_key[key]]
        existing_paths = {path for draft in component for path in draft.target_paths}
        target_paths = tuple(sorted(existing_paths | required_paths_by_root[root]))
        if len(logical_keys) == 1 and existing_paths == set(target_paths):
            normalized.extend(component)
            continue
        render_modes = {draft.render_mode for draft in component}
        render_policies = {draft.render_policy for draft in component}
        if render_modes != {"target_binding"} or len(render_policies) != 1:
            normalized.extend(component)
            continue
        if target_path_relationship(source_target, target_paths) == "composite_target_surface":
            normalized.extend(component)
            continue
        render_policy = next(iter(render_policies))
        group_kind, group_key = _canonical_target_group(target_paths)
        logical_key = "anchor:" + "|".join(target_paths)
        relationship_text = ",".join(sorted(relationships_by_root[root])) or "target_identity"
        normalized.extend(
            replace(
                draft,
                logical_key=logical_key,
                value_kind=_value_kind(target_paths, render_policy),
                group_kind=group_kind,
                group_key=group_key,
                target_paths=target_paths,
                rationale=(
                    "Host-normalized structured target co-binding: " + relationship_text + "."
                ),
            )
            for draft in component
        )
    return merge_drafts(normalized)


_CONTAINER_COUNT_SURFACE = re.compile(
    r"(?i)^\s*(?:[0-9][0-9,]*|[a-z]+(?:[ -][a-z]+)*)\s+"
    r"(?:container(?:\(s\)|s)?|cont\.)\s*$"
)
_TEMPERATURE_TARGET_PATH = re.compile(
    r"^(?P<base>documentPatch\.containers\[[0-9]+\]\.temperatureSetpoint)\."
    r"(?P<field>unit|value)$"
)
_TEMPERATURE_SURFACE = re.compile(
    r"(?i)^\s*(?P<number>[+-]?[0-9]+(?:[.,][0-9]+)?)\s*"
    r"(?P<degree>°)?\s*(?P<unit>C|F|CELSIUS|FAHRENHEIT)\s*$"
)


def _temperature_target_pair(paths: Sequence[str]) -> tuple[str, str] | None:
    if len(paths) != 2:
        return None
    matched = tuple(_TEMPERATURE_TARGET_PATH.fullmatch(path) for path in paths)
    if any(match is None for match in matched):
        return None
    typed = tuple(match for match in matched if match is not None)
    if len({match.group("base") for match in typed}) != 1 or {
        match.group("field") for match in typed
    } != {"unit", "value"}:
        return None
    by_field = {match.group("field"): path for match, path in zip(typed, paths, strict=True)}
    return by_field["unit"], by_field["value"]


def _temperature_surface_matches_target(
    *, draft: SpanDraft, source_target: Mapping[str, Any]
) -> bool:
    pair = _temperature_target_pair(draft.target_paths)
    match = _TEMPERATURE_SURFACE.fullmatch(draft.source_text)
    if pair is None or match is None:
        return False
    unit_path, value_path = pair
    unit = _resolve_target_path(source_target, unit_path)
    value = _resolve_target_path(source_target, value_path)
    unit_map = {"c": "celsius", "celsius": "celsius", "f": "fahrenheit", "fahrenheit": "fahrenheit"}
    if not isinstance(unit, str) or unit_map[match.group("unit").casefold()] != unit.casefold():
        return False
    if isinstance(value, bool) or not isinstance(value, (int, float, Decimal)):
        return False
    try:
        return Decimal(match.group("number").replace(",", ".")) == Decimal(str(value))
    except InvalidOperation:
        return False


def _normalize_temperature_setpoints(
    *, drafts: Sequence[SpanDraft], source_target: Mapping[str, Any]
) -> tuple[SpanDraft, ...]:
    grouped: dict[str, list[SpanDraft]] = defaultdict(list)
    for draft in drafts:
        expanded = draft
        if (
            draft.render_mode == "target_binding"
            and draft.value_kind == "temperature"
            and len(draft.target_paths) == 1
        ):
            path = draft.target_paths[0]
            leaf_match = _TEMPERATURE_TARGET_PATH.fullmatch(path)
            base = (
                path
                if path.endswith(".temperatureSetpoint")
                else leaf_match.group("base")
                if (
                    leaf_match is not None
                    and leaf_match.group("field") == "value"
                    and _TEMPERATURE_SURFACE.fullmatch(draft.source_text) is not None
                )
                else None
            )
            if base is not None:
                value = _resolve_target_path(source_target, base)
                if isinstance(value, Mapping) and {"unit", "value"} <= set(value):
                    expanded = replace(
                        draft,
                        target_paths=(base + ".unit", base + ".value"),
                    )
        grouped[expanded.logical_key].append(expanded)
    output: list[SpanDraft] = []
    for rows in grouped.values():
        first = rows[0]
        can_derive = (
            first.render_mode in {"target_binding", "agent_residual"}
            and first.value_kind == "temperature"
            and _temperature_target_pair(first.target_paths) is not None
            and all(
                row.target_paths == first.target_paths
                and _temperature_surface_matches_target(draft=row, source_target=source_target)
                for row in rows
            )
        )
        if not can_derive:
            output.extend(rows)
            continue
        output.extend(
            replace(
                row,
                render_mode="deterministic_derived",
                derivation="temperature_setpoint",
                dependency_paths=first.target_paths,
                dependency_bindings=(),
                evidence_origin="derived_operational_fact",
                render_policy="derived_surface",
                rationale=(
                    row.rationale
                    + " Host proved the numeric value and unit form a deterministic temperature "
                    "setpoint surface."
                ),
            )
            for row in rows
        )
    return merge_drafts(output)


def _edit_distance_at_most_one(left: str, right: str) -> bool:
    if left == right:
        return True
    if abs(len(left) - len(right)) > 1:
        return False
    if len(left) == len(right):
        return sum(a != b for a, b in zip(left, right, strict=True)) <= 1
    shorter, longer = (left, right) if len(left) < len(right) else (right, left)
    short_index = 0
    long_index = 0
    differences = 0
    while short_index < len(shorter) and long_index < len(longer):
        if shorter[short_index] == longer[long_index]:
            short_index += 1
            long_index += 1
            continue
        differences += 1
        if differences > 1:
            return False
        long_index += 1
    return True


def _canonicalize_near_ocr_source_only_keys(
    drafts: Sequence[SpanDraft],
) -> tuple[SpanDraft, ...]:
    """Give near-OCR source-only locality variants one deterministic key namespace.

    The values remain separate typed variants; this only prevents spelling noise from inventing
    unrelated semantic owners. Complete-link clustering prevents a chain of one-edit variants
    from collapsing endpoints that differ by more than one edit. Exact equal values under distinct
    keys are intentionally left alone because equality can be a coincidence between semantic roles.
    """

    by_scope: dict[tuple[str, str, str], dict[str, list[SpanDraft]]] = defaultdict(
        lambda: defaultdict(list)
    )
    for draft in drafts:
        if (
            draft.render_mode == "deterministic_auxiliary"
            and draft.value_kind in {"address", "location"}
            and not draft.target_paths
            and not draft.dependency_paths
            and not draft.dependency_bindings
        ):
            by_scope[(draft.group_kind, draft.group_key, draft.value_kind)][
                draft.logical_key
            ].append(draft)

    canonical_key: dict[str, str] = {}
    for keys in by_scope.values():
        normalized_by_key = {
            key: {_normalized_surface(row.source_text) for row in rows}
            for key, rows in keys.items()
        }
        eligible = {
            key: values
            for key, values in normalized_by_key.items()
            if values and all(len(value) >= 5 for value in values)
        }
        clusters: list[list[str]] = []
        for key in sorted(eligible):
            placed = False
            for cluster in clusters:
                combined_values = set(eligible[key])
                combined_values.update(value for member in cluster for value in eligible[member])
                if len(combined_values) > 1 and all(
                    _edit_distance_at_most_one(left, right)
                    for left, right in combinations(sorted(combined_values), 2)
                ):
                    cluster.append(key)
                    placed = True
                    break
            if not placed:
                clusters.append([key])
        for cluster in clusters:
            combined_values = {value for key in cluster for value in eligible[key]}
            if len(cluster) < 2 or len(combined_values) < 2:
                continue
            base_key = min(cluster)
            canonical_key.update({key: base_key for key in cluster})

    return tuple(
        replace(
            draft,
            logical_key=canonical_key[draft.logical_key],
            rationale=(
                draft.rationale
                + " Host placed same-scope one-edit OCR locality variants in one typed key "
                "namespace without asserting equal rendered values."
            ),
        )
        if draft.logical_key in canonical_key
        else draft
        for draft in drafts
    )


def _split_exact_composite_target_surfaces(
    *, drafts: Sequence[SpanDraft], source_target: Mapping[str, Any]
) -> tuple[SpanDraft, ...]:
    """Split a composite when exact target-component tokens prove every boundary."""

    output: list[SpanDraft] = []
    for draft in drafts:
        if (
            draft.render_mode not in {"agent_residual", "target_binding"}
            or len(draft.target_paths) < 2
        ):
            output.append(draft)
            continue
        components = target_fact_components(source_target, draft.target_paths)
        if len(components) < 2:
            output.append(draft)
            continue
        source_tokens = _surface_token_spans(draft.source_text)
        if not source_tokens:
            output.append(draft)
            continue
        matches: list[tuple[tuple[str, ...], int, int]] = []
        occupied_token_indexes: set[int] = set()
        for component in components:
            surfaces = tuple(
                _scalar_surface(_resolve_target_path(source_target, path)) for path in component
            )
            normalized_surfaces = {
                _normalized_surface(surface) for surface in surfaces if surface is not None
            }
            if len(normalized_surfaces) != 1 or any(surface is None for surface in surfaces):
                matches = []
                break
            target = next(surface for surface in surfaces if surface is not None)
            target_tokens = tuple(
                token for token, _start, _end in _surface_token_spans(target or "")
            )
            if not target_tokens:
                matches = []
                break
            starts = tuple(
                index
                for index in range(len(source_tokens) - len(target_tokens) + 1)
                if tuple(
                    token
                    for token, _start, _end in source_tokens[index : index + len(target_tokens)]
                )
                == target_tokens
            )
            if len(starts) != 1:
                matches = []
                break
            start_index = starts[0]
            token_indexes = set(range(start_index, start_index + len(target_tokens)))
            if occupied_token_indexes & token_indexes:
                matches = []
                break
            occupied_token_indexes.update(token_indexes)
            relative_start = source_tokens[start_index][1]
            relative_end = source_tokens[start_index + len(target_tokens) - 1][2]
            matches.append((component, relative_start, relative_end))
        if not matches or occupied_token_indexes != set(range(len(source_tokens))):
            output.append(draft)
            continue
        for component, relative_start, relative_end in matches:
            char_start = draft.char_start + relative_start
            char_end = draft.char_start + relative_end
            value_kind = _value_kind(component, "natural_text")
            group_kind, group_key = _canonical_target_group(component)
            logical_key = "anchor:" + "|".join(component)
            output.append(
                replace(
                    draft,
                    draft_id=(
                        "normalized_composite_"
                        + sha256_bytes(f"{draft.draft_id}\0{logical_key}".encode())[:16]
                    ),
                    logical_key=logical_key,
                    render_mode="target_binding",
                    value_kind=value_kind,
                    group_kind=group_kind,
                    group_key=group_key,
                    target_paths=component,
                    char_start=char_start,
                    char_end=char_end,
                    source_text=draft.source_text[relative_start:relative_end],
                    evidence_origin="host_verified_agent_proposal",
                    render_policy=_render_policy_for_value_kind(value_kind, "target_binding"),
                    rationale=(
                        draft.rationale
                        + " Host split the composite into disjoint exact target-token surfaces."
                    ),
                )
            )
    return merge_drafts(output)


def _split_exact_isolated_residual_occurrences(
    *, drafts: Sequence[SpanDraft], source_target: Mapping[str, Any]
) -> tuple[SpanDraft, ...]:
    """Separate an exact scalar occurrence from a multi-occurrence residual contract.

    A residual can legitimately own an inseparable composite. It cannot use separate physical
    occurrences as segments of different independently mutable facts. When one occurrence is the
    complete normalized surface of exactly one structured fact component, and no other occurrence
    in the same logical binding has that exact value, the host can isolate it without semantic
    inference. Schema-proven co-bound paths are treated as one component and remain together.
    """

    grouped: dict[str, list[SpanDraft]] = defaultdict(list)
    for draft in drafts:
        grouped[draft.logical_key].append(draft)
    output: list[SpanDraft] = []
    for rows in grouped.values():
        first = rows[0]
        if first.render_mode != "agent_residual" or len(rows) < 2 or len(first.target_paths) < 2:
            output.extend(rows)
            continue
        components = target_fact_components(source_target, first.target_paths)
        normalized_components: dict[tuple[str, ...], str] = {}
        for component in components:
            surfaces = tuple(
                _scalar_surface(_resolve_target_path(source_target, path)) for path in component
            )
            normalized_surfaces = {
                _normalized_surface(surface) for surface in surfaces if surface is not None
            }
            if len(normalized_surfaces) == 1 and all(surface is not None for surface in surfaces):
                normalized_components[component] = next(iter(normalized_surfaces))
        exact_components_by_row = {
            row.draft_id: tuple(
                component
                for component, normalized_target in normalized_components.items()
                if normalized_target and _normalized_surface(row.source_text) == normalized_target
            )
            for row in rows
        }
        exact_row_count_by_component = Counter(
            component
            for matching_components in exact_components_by_row.values()
            for component in matching_components
        )
        isolated = {
            row.draft_id: matching_components[0]
            for row in rows
            if len(matching_components := exact_components_by_row[row.draft_id]) == 1
            and exact_row_count_by_component[matching_components[0]] == 1
        }
        remaining_rows = [row for row in rows if row.draft_id not in isolated]
        isolated_paths = {path for component in isolated.values() for path in component}
        remaining_paths = tuple(path for path in first.target_paths if path not in isolated_paths)
        if not isolated or bool(remaining_rows) != bool(remaining_paths):
            output.extend(rows)
            continue
        for row in rows:
            isolated_path = isolated.get(row.draft_id)
            if isolated_path is None:
                value_kind = _value_kind(remaining_paths, "natural_text")
                group_kind, group_key = _canonical_target_group(remaining_paths)
                output.append(
                    replace(
                        row,
                        logical_key=(
                            "agent:residual:"
                            + sha256_bytes("|".join(remaining_paths).encode())[:16]
                        ),
                        value_kind=value_kind,
                        group_kind=group_kind,
                        group_key=group_key,
                        target_paths=remaining_paths,
                        rationale=(
                            row.rationale
                            + " Host removed separately printed exact scalar components from "
                            "this bounded residual."
                        ),
                    )
                )
                continue
            value_kind = _value_kind(isolated_path, "natural_text")
            group_kind, group_key = _canonical_target_group(isolated_path)
            output.append(
                replace(
                    row,
                    draft_id=(
                        "normalized_isolated_residual_"
                        + sha256_bytes(f"{row.draft_id}\0{'|'.join(isolated_path)}".encode())[:16]
                    ),
                    logical_key="anchor:" + "|".join(isolated_path),
                    render_mode="target_binding",
                    value_kind=value_kind,
                    group_kind=group_kind,
                    group_key=group_key,
                    target_paths=isolated_path,
                    render_policy=_render_policy_for_value_kind(value_kind, "target_binding"),
                    rationale=(
                        row.rationale
                        + " Host isolated a complete exact scalar occurrence from a segmented "
                        "residual contract."
                    ),
                )
            )
    duplicate_residuals: dict[tuple[int, int, tuple[str, ...]], list[SpanDraft]] = defaultdict(list)
    for row in output:
        if row.render_mode == "agent_residual":
            key = (row.char_start, row.char_end, tuple(sorted(row.target_paths)))
            duplicate_residuals[key].append(row)
    duplicate_ids: set[str] = set()
    replacements: dict[str, SpanDraft] = {}
    for (_start, _end, paths), rows in duplicate_residuals.items():
        if len(rows) < 2:
            continue
        value_kind = _value_kind(paths, "natural_text")
        group_kind, group_key = _canonical_target_group(paths)
        canonical_key = "agent:residual:" + sha256_bytes("|".join(paths).encode())[:16]
        replacements[rows[0].draft_id] = replace(
            rows[0],
            logical_key=canonical_key,
            value_kind=value_kind,
            group_kind=group_kind,
            group_key=group_key,
            target_paths=paths,
            rationale=(
                rows[0].rationale
                + " Host consolidated duplicate exact residual ownership for the same target "
                "contract."
            ),
        )
        duplicate_ids.update(row.draft_id for row in rows[1:])
    output = [
        replacements.get(row.draft_id, row) for row in output if row.draft_id not in duplicate_ids
    ]
    # This normalizer also runs before overlap reconciliation. Do not invoke
    # ``merge_drafts`` here: overlaps such as a residual address ending in an
    # independently owned country are inputs for the later ownership passes,
    # not invalid final state. Exact duplicate residual contracts were
    # already consolidated above; the enclosing pipeline performs the final
    # overlap and logical-contract validation.
    return tuple(output)


def _split_exact_repeated_line_target_surfaces(
    *, drafts: Sequence[SpanDraft], source_target: Mapping[str, Any]
) -> tuple[SpanDraft, ...]:
    """Turn one repeated multiline quote into exact physical occurrences of one target fact."""

    output: list[SpanDraft] = []
    for draft in drafts:
        target_surfaces = _target_scalar_surfaces(draft, source_target)
        separators = tuple(re.finditer(r"\r\n|\n|\r", draft.source_text))
        boundaries = (
            (
                (0, separators[0].start()),
                *((left.end(), right.start()) for left, right in pairwise(separators)),
                (separators[-1].end(), len(draft.source_text)),
            )
            if separators
            else ()
        )
        surfaces = tuple(draft.source_text[start:end] for start, end in boundaries)
        can_split = (
            draft.render_mode == "target_binding"
            and len(target_surfaces) == 1
            and "\n" not in target_surfaces[0]
            and "\r" not in target_surfaces[0]
            and len(surfaces) >= 2
            and bool(surfaces[0])
            and all(surface == surfaces[0] for surface in surfaces[1:])
            and _matching_token_projection(surfaces[0], target_surfaces[0]) is not None
        )
        if not can_split:
            output.append(draft)
            continue
        for index, (relative_start, relative_end) in enumerate(boundaries):
            char_start = draft.char_start + relative_start
            char_end = draft.char_start + relative_end
            output.append(
                replace(
                    draft,
                    draft_id=(
                        "normalized_repeat_"
                        + sha256_bytes(f"{draft.draft_id}\0{char_start}\0{char_end}".encode())[:16]
                    ),
                    char_start=char_start,
                    char_end=char_end,
                    source_text=surfaces[index],
                    evidence_origin="host_verified_agent_proposal",
                    rationale=(
                        draft.rationale
                        + " Host split an exact repeated-line target projection into separate "
                        "physical occurrences."
                    ),
                )
            )
    return merge_drafts(output)


def _normalize_exact_single_target_residuals(
    *, drafts: Sequence[SpanDraft], source_target: Mapping[str, Any]
) -> tuple[SpanDraft, ...]:
    """Turn a residual that exactly prints one scalar target into a direct binding.

    A residual is justified only when semantic composition remains. If its complete token stream
    is exactly the token stream of one scalar target, replacement is already a deterministic
    projection. This includes punctuation variants such as ``NON-NEGOTIABLE`` for a structured
    ``NON_NEGOTIABLE`` enum, while excluding abbreviations, reordered dates, and framed text.
    """

    grouped: dict[str, list[SpanDraft]] = defaultdict(list)
    for draft in drafts:
        grouped[draft.logical_key].append(draft)
    convertible: dict[str, str] = {}
    for logical_key, rows in grouped.items():
        first = rows[0]
        if not (
            first.render_mode == "agent_residual"
            and len(first.target_paths) == 1
            and first.derivation is None
            and not first.dependency_paths
            and not first.dependency_bindings
        ):
            continue
        path = first.target_paths[0]
        target_surface = _scalar_surface(_resolve_target_path(source_target, path))
        if target_surface is not None and all(
            _matching_token_projection(row.source_text, target_surface) == ((), ()) for row in rows
        ):
            convertible[logical_key] = path

    output: list[SpanDraft] = []
    for draft in drafts:
        output_path = convertible.get(draft.logical_key)
        if output_path is None:
            output.append(draft)
            continue
        value_kind = _value_kind((output_path,), "natural_text")
        group_kind, group_key = _canonical_target_group((output_path,))
        output.append(
            replace(
                draft,
                logical_key="anchor:" + output_path,
                render_mode="target_binding",
                value_kind=value_kind,
                group_kind=group_kind,
                group_key=group_key,
                evidence_origin="host_verified_agent_proposal",
                render_policy=_render_policy_for_value_kind(value_kind, "target_binding"),
                rationale=(
                    draft.rationale
                    + " Host converted this exact one-scalar token projection from residual "
                    "editing to a direct deterministic target binding."
                ),
            )
        )
    return merge_drafts(output)


def _normalize_unique_source_only_identifier_projections(
    *, raw: str, drafts: Sequence[SpanDraft], source_target: Mapping[str, Any]
) -> tuple[SpanDraft, ...]:
    """Attach provably coupled source-only identifiers to their target scalar.

    A compiler can correctly spot a shipment identifier yet classify it as independently
    generated even though that exact token is part of a structured text scalar already owned by
    the template. Leaving those two values independent permits descendant synthesis to contradict
    the target. This normalization joins them only when every occurrence in one source-only
    identifier binding is a unique token projection of exactly one target-owned scalar.

    Same-scope projections are already semantically localized by the compiler. Cross-scope
    projections require an alphanumeric identifier surface and exhaustive ownership of every
    byte-exact, token-bounded source occurrence. That second proof excludes low-information
    coincidences such as a booking reference ``2`` also appearing in an unrelated address.
    """

    grouped: dict[str, list[SpanDraft]] = defaultdict(list)
    for draft in drafts:
        grouped[draft.logical_key].append(draft)

    target_owners: list[tuple[SpanDraft, tuple[SpanDraft, ...], str]] = []
    for rows in grouped.values():
        first = rows[0]
        if (
            first.render_mode != "target_binding"
            or not first.target_paths
            or first.render_policy != "natural_text"
        ):
            continue
        values = tuple(_resolve_target_path(source_target, path) for path in first.target_paths)
        if len({canonical_json_bytes(value) for value in values}) != 1:
            continue
        target_surface = _scalar_surface(values[0])
        if target_surface is not None:
            target_owners.append((first, tuple(rows), target_surface))

    replacements: dict[str, SpanDraft] = {}
    for rows in grouped.values():
        first = rows[0]
        if not (
            first.render_mode in {"deterministic_auxiliary", "agent_residual"}
            and first.value_kind == "identifier"
            and not first.target_paths
            and first.derivation is None
            and not first.dependency_paths
            and not first.dependency_bindings
        ):
            continue
        candidates = tuple(
            (owner, owner_rows, target_surface)
            for owner, owner_rows, target_surface in target_owners
            if all(
                _matching_token_projection(row.source_text, target_surface) is not None
                for row in rows
            )
        )
        candidate_keys = {owner.logical_key for owner, _owner_rows, _surface in candidates}
        if len(candidate_keys) != 1:
            continue
        owner, owner_rows, _target_surface = next(
            candidate for candidate in candidates if candidate[0].logical_key in candidate_keys
        )
        same_scope = (first.group_kind, first.group_key) == (
            owner.group_kind,
            owner.group_key,
        )
        source_surfaces = tuple(dict.fromkeys(row.source_text for row in rows))
        structurally_specific = all(
            any(character.isalpha() for character in surface)
            and any(character.isdigit() for character in surface)
            for surface in source_surfaces
        )
        owned_projection_rows = (*rows, *owner_rows)
        exhaustive = structurally_specific
        for source_surface in source_surfaces:
            for start in _exact_offsets(raw, source_surface):
                end = start + len(source_surface)
                if not is_token_bounded_surface_span(raw, start, end):
                    continue
                if not any(
                    candidate.char_start <= start and end <= candidate.char_end
                    for candidate in owned_projection_rows
                ):
                    exhaustive = False
                    break
            if not exhaustive:
                break
        if not (same_scope or (structurally_specific and exhaustive)):
            continue
        for row in rows:
            replacements[row.draft_id] = replace(
                row,
                logical_key=owner.logical_key,
                render_mode=owner.render_mode,
                value_kind=owner.value_kind,
                group_kind=owner.group_kind,
                group_key=owner.group_key,
                target_paths=owner.target_paths,
                derivation=owner.derivation,
                dependency_paths=owner.dependency_paths,
                dependency_bindings=owner.dependency_bindings,
                evidence_origin="host_verified_agent_proposal",
                render_policy=owner.render_policy,
                rationale=(
                    row.rationale
                    + " Host proved this source-only identifier is a unique token projection "
                    "of the target-owned scalar and joined their descendant value contract."
                ),
            )
    return merge_drafts(replacements.get(row.draft_id, row) for row in drafts)


def _normalize_package_count_residuals(
    *, drafts: Sequence[SpanDraft], source_target: Mapping[str, Any]
) -> tuple[SpanDraft, ...]:
    """Split a linked quantity-and-kind residual into direct mutable leaf owners.

    ``package_count`` means the cardinality of a collection.  A printed surface such as
    ``2 BOXES`` instead expresses a package quantity and category, so treating scalar leaf paths
    as a collection-count dependency is both semantically wrong and not executable by the
    descendant renderer.  The split below is admitted only when package/allocation topology and
    every source component are mechanically proved.
    """

    patch = source_target.get("documentPatch")
    packages = patch.get("cargoPackages") if isinstance(patch, Mapping) else None
    allocation_groups = patch.get("cargoAllocationGroups") if isinstance(patch, Mapping) else None
    if not isinstance(packages, list) or not isinstance(allocation_groups, list):
        return tuple(drafts)

    grouped: dict[str, list[SpanDraft]] = defaultdict(list)
    for draft in drafts:
        grouped[draft.logical_key].append(draft)
    discarded: set[str] = set()
    additions: list[SpanDraft] = []
    for logical_key, rows in grouped.items():
        first = rows[0]
        if not (
            first.render_mode == "agent_residual"
            and first.value_kind == "package"
            and first.derivation is None
            and not first.dependency_paths
            and not first.dependency_bindings
            and all(
                row.render_mode == first.render_mode
                and row.value_kind == first.value_kind
                and row.target_paths == first.target_paths
                and row.derivation is None
                and not row.dependency_paths
                and not row.dependency_bindings
                for row in rows
            )
        ):
            continue
        package_quantity_matches = tuple(
            match
            for path in first.target_paths
            if (match := _PACKAGE_QUANTITY_PATH.fullmatch(path)) is not None
        )
        package_type_matches = tuple(
            match
            for path in first.target_paths
            if (match := _PACKAGE_TYPE_PATH.fullmatch(path)) is not None
        )
        allocation_matches = tuple(
            match
            for path in first.target_paths
            if (match := _ALLOCATION_PACKAGE_QUANTITY_PATH.fullmatch(path)) is not None
        )
        if not (
            len(first.target_paths) == 3
            and len(package_quantity_matches) == 1
            and len(package_type_matches) == 1
            and len(allocation_matches) == 1
        ):
            continue
        package_index = int(package_quantity_matches[0].group(1))
        if int(package_type_matches[0].group(1)) != package_index or package_index >= len(packages):
            continue
        group_index = int(allocation_matches[0].group(1))
        allocation_index = int(allocation_matches[0].group(2))
        package = packages[package_index]
        allocation_group = (
            allocation_groups[group_index] if group_index < len(allocation_groups) else None
        )
        allocations = (
            allocation_group.get("allocations") if isinstance(allocation_group, Mapping) else None
        )
        allocation = (
            allocations[allocation_index]
            if isinstance(allocations, list) and allocation_index < len(allocations)
            else None
        )
        if not (
            isinstance(package, Mapping)
            and isinstance(allocation_group, Mapping)
            and isinstance(allocation, Mapping)
            and package.get("packageId") == allocation.get("packageId")
            and package.get("groupId") == allocation_group.get("groupId")
            and canonical_json_bytes(package.get("quantity"))
            == canonical_json_bytes(allocation.get("packageQuantity"))
        ):
            continue
        components = tuple(
            _package_component_spans(
                row.source_text,
                quantity=package.get("quantity"),
                category=package.get("typeCategory"),
            )
            for row in rows
        )
        if any(component is None for component in components):
            continue
        quantity_paths = tuple(
            sorted(
                (
                    package_quantity_matches[0].group(0),
                    allocation_matches[0].group(0),
                )
            )
        )
        type_path = package_type_matches[0].group(0)
        quantity_group = _canonical_target_group(quantity_paths)
        type_group = _canonical_target_group((type_path,))
        discarded.update(row.draft_id for row in rows)
        for row, component in zip(rows, components, strict=True):
            assert component is not None
            for role, (relative_start, relative_end), target_paths, value_kind, group in (
                ("quantity", component[0], quantity_paths, "integer", quantity_group),
                ("category", component[1], (type_path,), "package", type_group),
            ):
                start = row.char_start + relative_start
                end = row.char_start + relative_end
                additions.append(
                    replace(
                        row,
                        draft_id=(
                            "host_package_component_"
                            + sha256_bytes(f"{logical_key}\0{role}\0{start}\0{end}".encode())[:16]
                        ),
                        logical_key="anchor:" + "|".join(target_paths),
                        render_mode="target_binding",
                        value_kind=value_kind,
                        group_kind=group[0],
                        group_key=group[1],
                        target_paths=target_paths,
                        derivation=None,
                        dependency_paths=(),
                        dependency_bindings=(),
                        char_start=start,
                        char_end=end,
                        source_text=row.source_text[relative_start:relative_end],
                        evidence_origin="host_verified_agent_proposal",
                        render_policy=_render_policy_for_value_kind(value_kind, "target_binding"),
                        rationale=(
                            row.rationale
                            + " Host split this linked package surface into a direct mutable "
                            f"{role} leaf after proving package/allocation topology."
                        ),
                    )
                )

    return (*tuple(row for row in drafts if row.draft_id not in discarded), *additions)


def _normalize_direct_target_derivations(
    *, drafts: Sequence[SpanDraft], source_target: Mapping[str, Any]
) -> tuple[SpanDraft, ...]:
    """Prefer a proven direct scalar owner over an unnecessary derivation.

    A structured aggregate can be printed directly even when related component values also
    appear elsewhere.  If every occurrence is already a host-proven formatting realization of
    its sole target scalar, deriving it from components adds a weaker and potentially false
    contract.  Canonicalizing it back to the direct target is both cheaper and strictly more
    faithful to the pinned source label.
    """

    grouped: dict[str, list[SpanDraft]] = defaultdict(list)
    for draft in drafts:
        grouped[draft.logical_key].append(draft)
    convertible: dict[str, tuple[str, str, str, str]] = {}
    for logical_key, rows in grouped.items():
        first = rows[0]
        if (
            first.render_mode != "deterministic_derived"
            or len(first.target_paths) != 1
            or first.derivation == "inclusive_range_cardinality"
        ):
            continue
        path = first.target_paths[0]
        target_value = _resolve_target_path(source_target, path)
        candidate_policy = _render_policy_for_value_kind(first.value_kind, "target_binding")
        candidate = replace(
            first,
            render_mode="target_binding",
            derivation=None,
            dependency_paths=(),
            dependency_bindings=(),
            render_policy=candidate_policy,
        )
        adapter = _surface_adapter(candidate)
        if not all(
            _surface_match(
                source=row.source_text,
                target=target_value,
                adapter=adapter,
                value_kind=first.value_kind,
            )
            is not None
            for row in rows
        ):
            continue
        group_kind, group_key = _canonical_target_group((path,))
        convertible[logical_key] = (path, group_kind, group_key, candidate_policy)

    output: list[SpanDraft] = []
    for draft in drafts:
        contract = convertible.get(draft.logical_key)
        if contract is None:
            output.append(draft)
            continue
        path, group_kind, group_key, render_policy = contract
        output.append(
            replace(
                draft,
                logical_key="anchor:" + path,
                render_mode="target_binding",
                group_kind=group_kind,
                group_key=group_key,
                derivation=None,
                dependency_paths=(),
                dependency_bindings=(),
                evidence_origin="host_verified_agent_proposal",
                render_policy=render_policy,
                rationale=(
                    draft.rationale
                    + " Host replaced this unnecessary derivation with the directly printed "
                    "structured scalar; every occurrence is a proven target realization."
                ),
            )
        )
    return merge_drafts(output)


def _normalize_direct_target_equivalence_subset(
    *, drafts: Sequence[SpanDraft], source_target: Mapping[str, Any]
) -> tuple[SpanDraft, ...]:
    """Remove provably incompatible paths from a compiler-added direct equality binding.

    A compiler can attach one printed summary to several currently similar target fields even
    when one field contains additional semantics. For a direct binding, this is not composition:
    all target values must be equal. Restrict the proposal only when every occurrence realizes one
    and only one canonical-value subset. Accepted-label composite evidence remains untouched for
    explicit semantic review.
    """

    grouped: dict[str, list[SpanDraft]] = defaultdict(list)
    for draft in drafts:
        grouped[draft.logical_key].append(draft)
    contracts: dict[str, tuple[tuple[str, ...], str, str, str, str]] = {}
    for logical_key, rows in grouped.items():
        first = rows[0]
        if (
            first.render_mode != "target_binding"
            or first.evidence_origin == "accepted_label_evidence"
            or len(first.target_paths) < 2
            or any(row.target_paths != first.target_paths for row in rows)
        ):
            continue
        paths_by_value: dict[bytes, list[str]] = defaultdict(list)
        values_by_encoding: dict[bytes, Any] = {}
        for path in first.target_paths:
            value = _resolve_target_path(source_target, path)
            encoded = canonical_json_bytes(value)
            paths_by_value[encoded].append(path)
            values_by_encoding[encoded] = value
        if len(paths_by_value) < 2:
            continue
        adapter = _surface_adapter(first)
        matching = tuple(
            tuple(paths)
            for encoded, paths in paths_by_value.items()
            if all(
                _surface_match(
                    source=row.source_text,
                    target=values_by_encoding[encoded],
                    adapter=adapter,
                    value_kind=first.value_kind,
                )
                is not None
                for row in rows
            )
        )
        if len(matching) != 1:
            continue
        target_paths = tuple(sorted(matching[0]))
        group_kind, group_key = _canonical_target_group(target_paths)
        render_policy = _render_policy_for_value_kind(
            _value_kind(target_paths, first.render_policy), "target_binding"
        )
        contracts[logical_key] = (
            target_paths,
            group_kind,
            group_key,
            _value_kind(target_paths, render_policy),
            render_policy,
        )

    output: list[SpanDraft] = []
    for draft in drafts:
        contract = contracts.get(draft.logical_key)
        if contract is None:
            output.append(draft)
            continue
        target_paths, group_kind, group_key, value_kind, render_policy = contract
        output.append(
            replace(
                draft,
                logical_key="anchor:" + "|".join(target_paths),
                value_kind=value_kind,
                group_kind=group_kind,
                group_key=group_key,
                target_paths=target_paths,
                render_policy=render_policy,
                rationale=(
                    draft.rationale
                    + " Host removed incompatible direct target paths after proving every "
                    "occurrence realizes exactly one canonical-value subset."
                ),
            )
        )
    return merge_drafts(output)


def _normalize_ambiguous_structural_derived_targets(
    *, drafts: Sequence[SpanDraft], source_target: Mapping[str, Any]
) -> tuple[SpanDraft, ...]:
    """Keep a shared collection as an input, not two different derived scalar owners.

    Several totals can depend on the same collection, but the collection itself cannot be the
    represented output of both logical bindings. Remove only a structural target claimed by more
    than one derivation when every claimant already declares strict descendant dependencies. The
    dependency contract and every physical occurrence remain intact.
    """

    eligible_by_path: dict[str, set[str]] = defaultdict(set)
    claimers_by_path: dict[str, set[str]] = defaultdict(set)
    for draft in drafts:
        for path in draft.target_paths:
            claimers_by_path[path].add(draft.logical_key)
            try:
                value = _resolve_target_path(source_target, path)
            except ValueError:
                continue
            if (
                draft.render_mode == "deterministic_derived"
                and isinstance(value, (Mapping, list))
                and draft.dependency_paths
                and all(
                    dependency.startswith(path + ".") or dependency.startswith(path + "[")
                    for dependency in draft.dependency_paths
                )
            ):
                eligible_by_path[path].add(draft.logical_key)
    removable = {
        path
        for path, claimers in claimers_by_path.items()
        if len(claimers) > 1 and eligible_by_path.get(path) == claimers
    }
    if not removable:
        return tuple(drafts)
    return merge_drafts(
        replace(
            draft,
            target_paths=tuple(path for path in draft.target_paths if path not in removable),
            rationale=(
                draft.rationale
                + " Host retained the shared structural collection only as a derivation input; "
                "multiple different scalar totals cannot own the same collection target."
            ),
        )
        if set(draft.target_paths).intersection(removable)
        else draft
        for draft in drafts
    )


def _normalize_source_only_alphanumeric_identifier_kinds(
    drafts: Sequence[SpanDraft],
) -> tuple[SpanDraft, ...]:
    """Classify a complete high-entropy alphanumeric token as an identifier.

    The proof is grammatical, not vocabulary-specific: every occurrence must be one bounded token
    with both letters and digits and no count/noun separator.  This prevents a package-row
    reference such as ``EGLHOPHO24A1227`` from inheriting a ``package`` value kind merely because
    of its surrounding row.
    """

    grouped: dict[str, list[SpanDraft]] = defaultdict(list)
    for draft in drafts:
        grouped[draft.logical_key].append(draft)
    eligible: set[str] = set()
    for logical_key, rows in grouped.items():
        first = rows[0]
        if not (
            first.render_mode == "deterministic_auxiliary"
            and first.value_kind != "identifier"
            and not first.target_paths
            and not first.dependency_paths
            and not first.dependency_bindings
            and all(
                row.render_mode == first.render_mode
                and row.value_kind == first.value_kind
                and not row.target_paths
                and not row.dependency_paths
                and not row.dependency_bindings
                and re.fullmatch(r"[A-Za-z0-9]{8,}", row.source_text) is not None
                and any(character.isalpha() for character in row.source_text)
                and any(character.isdigit() for character in row.source_text)
                for row in rows
            )
        ):
            continue
        eligible.add(logical_key)
    return merge_drafts(
        replace(
            row,
            value_kind="identifier",
            render_policy="opaque_identifier",
            rationale=(
                row.rationale
                + " Host classified this complete bounded alphanumeric source token as an "
                "identifier rather than its surrounding row vocabulary."
            ),
        )
        if row.logical_key in eligible
        else row
        for row in drafts
    )


def normalize_source_seal_ownership(
    *, drafts: Sequence[SpanDraft], source_target: Mapping[str, Any]
) -> tuple[SpanDraft, ...]:
    """Promote exact, container-local source seal slots over detached auxiliaries.

    The model may classify a printed seal as source-only even though the source
    label contains that same seal. Promotion requires a unique auxiliary logical
    binding in the exact container scope; ambiguous or unprinted seals remain
    unresolved for certification review rather than being guessed.
    """
    patch = source_target.get("documentPatch")
    containers = patch.get("containers") if isinstance(patch, Mapping) else None
    if not isinstance(containers, list):
        return tuple(drafts)
    owned = {path for draft in drafts for path in draft.target_paths}
    by_key: dict[str, list[SpanDraft]] = defaultdict(list)
    for draft in drafts:
        by_key[draft.logical_key].append(draft)
    promotions: dict[str, str] = {}
    for container_index, container in enumerate(containers):
        if not isinstance(container, Mapping):
            continue
        seals = container.get("sealNumbers")
        if not isinstance(seals, list):
            continue
        for seal_index, seal in enumerate(seals):
            path = f"documentPatch.containers[{container_index}].sealNumbers[{seal_index}]"
            if path in owned or not isinstance(seal, str) or not seal:
                continue
            source_value = _normalized_surface(seal)
            candidates = {
                draft.logical_key
                for draft in drafts
                if draft.render_mode in {"deterministic_auxiliary", "agent_residual"}
                and draft.group_kind == "equipment"
                and draft.group_key == f"container:{container_index}"
                and not draft.target_paths
                and not draft.dependency_paths
                and not draft.dependency_bindings
                and _normalized_surface(draft.source_text) == source_value
            }
            if len(candidates) != 1:
                continue
            logical_key = next(iter(candidates))
            if any(
                row.group_key != f"container:{container_index}"
                or _normalized_surface(row.source_text) != source_value
                for row in by_key[logical_key]
            ):
                continue
            previous = promotions.setdefault(logical_key, path)
            if previous != path:
                raise ValueError(f"one seal binding matches multiple target leaves: {logical_key}")
    return merge_drafts(
        replace(
            draft,
            logical_key="anchor:" + promotions[draft.logical_key],
            render_mode="target_binding",
            value_kind="equipment",
            target_paths=(promotions[draft.logical_key],),
            evidence_origin="host_verified_agent_proposal",
            render_policy="opaque_identifier",
            rationale=(
                draft.rationale
                + " Host promoted the exact printed seal to its unique container-local "
                "source-label leaf."
            ),
        )
        if draft.logical_key in promotions
        else draft
        for draft in drafts
    )


def validate_source_seal_ownership(
    *, drafts: Sequence[SpanDraft], source_target: Mapping[str, Any]
) -> None:
    """Certification cannot infer a seal leaf from its parent container binding."""
    patch = source_target.get("documentPatch")
    containers = patch.get("containers") if isinstance(patch, Mapping) else None
    if not isinstance(containers, list):
        return
    owners: dict[str, set[str]] = defaultdict(set)
    for draft in drafts:
        for path in draft.target_paths:
            if re.fullmatch(r"documentPatch\.containers\[\d+\]\.sealNumbers\[\d+\]", path):
                owners[path].add(draft.logical_key)
    for container_index, container in enumerate(containers):
        if not isinstance(container, Mapping):
            continue
        seals = container.get("sealNumbers")
        if not isinstance(seals, list):
            continue
        for seal_index, seal in enumerate(seals):
            path = f"documentPatch.containers[{container_index}].sealNumbers[{seal_index}]"
            keys = owners.get(path, set())
            if len(keys) != 1:
                raise ValueError(f"source seal requires exactly one leaf binding: {path}")
            if any(
                draft.group_kind != "equipment" or draft.group_key != f"container:{container_index}"
                for draft in drafts
                if draft.logical_key in keys
            ):
                raise ValueError(f"source seal binding has wrong container ownership: {path}")
            if isinstance(seal, str) and any(
                draft.render_mode in {"deterministic_auxiliary", "agent_residual"}
                and not draft.target_paths
                and draft.group_kind == "equipment"
                and draft.group_key == f"container:{container_index}"
                and _normalized_surface(draft.source_text) == _normalized_surface(seal)
                for draft in drafts
            ):
                raise ValueError(f"source seal also has an independent auxiliary binding: {path}")


def normalize_deterministic_draft_semantics(
    *, raw: str, drafts: Sequence[SpanDraft], source_target: Mapping[str, Any]
) -> tuple[SpanDraft, ...]:
    """Canonicalize pure derivations and typed source-only variants before criticism."""

    containers_path = "documentPatch.containers"
    try:
        containers = _resolve_target_path(source_target, containers_path)
    except ValueError:
        containers = None
    has_container_collection = isinstance(containers, Sequence) and not isinstance(
        containers, (str, bytes)
    )
    source_kind_normalized = _normalize_source_only_alphanumeric_identifier_kinds(drafts)
    policy_normalized = tuple(
        replace(
            draft,
            render_policy=_render_policy_for_value_kind(draft.value_kind, draft.render_mode),
        )
        for draft in source_kind_normalized
    )
    seal_normalized = normalize_source_seal_ownership(
        drafts=policy_normalized,
        source_target=source_target,
    )
    package_count_normalized = _normalize_package_count_residuals(
        drafts=seal_normalized,
        source_target=source_target,
    )
    exact_residual_normalized = _normalize_exact_single_target_residuals(
        drafts=package_count_normalized,
        source_target=source_target,
    )
    isolated_residual_normalized = _split_exact_isolated_residual_occurrences(
        drafts=exact_residual_normalized,
        source_target=source_target,
    )
    composite_normalized = _split_exact_composite_target_surfaces(
        drafts=isolated_residual_normalized,
        source_target=source_target,
    )
    repeated_line_normalized = _split_exact_repeated_line_target_surfaces(
        drafts=composite_normalized,
        source_target=source_target,
    )
    temperature_normalized = _normalize_temperature_setpoints(
        drafts=repeated_line_normalized, source_target=source_target
    )
    direct_subset_normalized = _normalize_direct_target_equivalence_subset(
        drafts=temperature_normalized,
        source_target=source_target,
    )
    direct_target_normalized = _normalize_direct_target_derivations(
        drafts=direct_subset_normalized,
        source_target=source_target,
    )
    structural_target_normalized = _normalize_ambiguous_structural_derived_targets(
        drafts=direct_target_normalized,
        source_target=source_target,
    )
    dependency_normalized: list[SpanDraft] = []
    for draft in structural_target_normalized:
        duplicated_source_dependencies = (
            draft.render_mode == "deterministic_derived"
            and draft.derivation
            in {
                "sum_package_quantity",
                "sum_gross_weight",
                "sum_net_weight",
                "sum_tare_weight",
                "sum_volume",
                "package_count",
                "sum_monetary_amounts",
                "sum_decimal_values",
            }
            and not draft.target_paths
            and not draft.dependency_paths
            and bool(draft.dependency_bindings)
            and len(set(draft.dependency_bindings)) < len(draft.dependency_bindings)
        )
        if not duplicated_source_dependencies:
            dependency_normalized.append(draft)
            continue
        dependency_normalized.append(
            replace(
                draft,
                render_mode="deterministic_auxiliary",
                derivation=None,
                dependency_paths=(),
                dependency_bindings=(),
                evidence_origin="host_verified_agent_proposal",
                render_policy=_render_policy_for_value_kind(
                    draft.value_kind, "deterministic_auxiliary"
                ),
                rationale=(
                    draft.rationale
                    + " Host preserved this printed source-only total as an auxiliary because "
                    "repeating one logical dependency does not prove independent summands."
                ),
            )
        )
    regulatory_semantics_normalized: list[SpanDraft] = []
    for draft in dependency_normalized:
        is_hazard_category = len(draft.target_paths) == 1 and draft.target_paths[0].endswith(
            ".hazardCategory"
        )
        target_surface = (
            _scalar_surface(_resolve_target_path(source_target, draft.target_paths[0]))
            if is_hazard_category
            else None
        )
        if (
            draft.render_mode in {"target_binding", "deterministic_derived"}
            and is_hazard_category
            and _DANGEROUS_GOODS_CLASS_SURFACE.fullmatch(draft.source_text) is not None
            and target_surface is not None
            and _normalized_surface(draft.source_text) != _normalized_surface(target_surface)
        ):
            regulatory_semantics_normalized.append(
                replace(
                    draft,
                    render_mode="agent_residual",
                    derivation=None,
                    dependency_paths=(),
                    dependency_bindings=(),
                    render_policy=_render_policy_for_value_kind(draft.value_kind, "agent_residual"),
                    rationale=(
                        draft.rationale
                        + " Host bounded an unsupported dangerous-goods category-to-regulatory-"
                        "class conversion as agent residual; no authoritative classification "
                        "registry is available to prove a deterministic derivation."
                    ),
                )
            )
        else:
            regulatory_semantics_normalized.append(draft)

    def is_container_count_projection(draft: SpanDraft) -> bool:
        return (
            draft.render_mode == "deterministic_derived"
            and containers_path in {*draft.target_paths, *draft.dependency_paths}
            and (
                draft.derivation == "container_count"
                or _CONTAINER_COUNT_SURFACE.fullmatch(draft.source_text) is not None
            )
        )

    # A collection can have several deterministic textual projections (for example, a complete
    # equipment receipt and a separate ``20 cntrs`` count).  Only one binding owns the structured
    # collection; the other projections depend on it.  Determine that ownership before rewriting
    # count rows so the canonicalizer cannot manufacture a duplicate target owner.
    container_collection_owned_by_other_projection = any(
        containers_path in draft.target_paths and not is_container_count_projection(draft)
        for draft in regulatory_semantics_normalized
    )
    canonical_count_targets = (
        () if container_collection_owned_by_other_projection else (containers_path,)
    )
    count_normalized: list[SpanDraft] = []
    count_key_rewrites: dict[str, str] = {}
    for draft in regulatory_semantics_normalized:
        if has_container_collection and is_container_count_projection(draft):
            canonical_logical_key = "agent:container_count:documentPatch.containers"
            canonical = (
                draft.logical_key == canonical_logical_key
                and draft.value_kind == "integer"
                and draft.group_kind == "equipment"
                and draft.group_key == "equipment:all"
                and draft.derivation == "container_count"
                and draft.target_paths == canonical_count_targets
                and draft.dependency_paths == (containers_path,)
                and not draft.dependency_bindings
                and draft.render_policy == "derived_surface"
            )
            if canonical:
                count_normalized.append(draft)
                continue
            count_key_rewrites[draft.logical_key] = canonical_logical_key
            count_normalized.append(
                replace(
                    draft,
                    logical_key=canonical_logical_key,
                    render_mode="deterministic_derived",
                    value_kind="integer",
                    group_kind="equipment",
                    group_key="equipment:all",
                    derivation="container_count",
                    target_paths=canonical_count_targets,
                    dependency_paths=(containers_path,),
                    dependency_bindings=(),
                    render_policy="derived_surface",
                    rationale=(
                        draft.rationale
                        + " Host canonicalized the complete container-count noun surface to "
                        "container_count."
                    ),
                )
            )
        else:
            count_normalized.append(draft)

    if count_key_rewrites:
        count_normalized = [
            replace(
                draft,
                dependency_bindings=tuple(
                    dict.fromkeys(
                        count_key_rewrites.get(dependency, dependency)
                        for dependency in draft.dependency_bindings
                    )
                ),
                rationale=(
                    draft.rationale
                    + " Host updated this binding's dependency references after canonicalizing "
                    "the container-count owner."
                ),
            )
            if any(dependency in count_key_rewrites for dependency in draft.dependency_bindings)
            else draft
            for draft in count_normalized
        ]

    key_normalized = _canonicalize_near_ocr_source_only_keys(count_normalized)

    grouped: dict[str, list[SpanDraft]] = defaultdict(list)
    for draft in key_normalized:
        grouped[draft.logical_key].append(draft)
    output: list[SpanDraft] = []
    for logical_key, rows in grouped.items():
        first = rows[0]
        surfaces: dict[str, list[SpanDraft]] = defaultdict(list)
        for row in rows:
            surfaces[_normalized_surface(row.source_text)].append(row)
        normalized_values = tuple(value for value in surfaces if value)
        distinct_source_only_dates = False
        if (
            first.render_mode in {"deterministic_auxiliary", "agent_residual"}
            and first.value_kind == "date"
            and not first.target_paths
            and len(normalized_values) > 1
        ):
            parsed_dates = {
                normalized_value: frozenset(
                    candidate
                    for row in variant_rows
                    for candidate in _date_candidates(row.source_text)
                )
                for normalized_value, variant_rows in surfaces.items()
            }
            distinct_source_only_dates = all(
                parsed_dates[value] for value in normalized_values
            ) and all(
                parsed_dates[left].isdisjoint(parsed_dates[right])
                for left, right in combinations(normalized_values, 2)
            )
        if distinct_source_only_dates:
            for normalized_value, variant_rows in sorted(surfaces.items()):
                suffix = sha256_bytes(normalized_value.encode())[:12]
                output.extend(
                    replace(
                        row,
                        logical_key=f"{logical_key}:typed_date:{suffix}",
                        render_mode="deterministic_auxiliary",
                        group_key=f"{first.group_key}:date:{suffix}",
                        derivation=None,
                        dependency_paths=(),
                        dependency_bindings=(),
                        rationale=(
                            row.rationale
                            + " Host split source-only occurrences that parse to disjoint "
                            "calendar dates into independent typed deterministic bindings."
                        ),
                    )
                    for row in variant_rows
                )
            continue
        location_variant_group = (
            first.render_mode == "agent_residual"
            and first.value_kind == "location"
            and not first.target_paths
            and len(normalized_values) > 1
        )
        shortest = min(normalized_values, key=len) if normalized_values else ""
        exact_containing_location_variants = (
            location_variant_group
            and len(shortest) >= 3
            and all(shortest in value for value in normalized_values)
        )
        near_ocr_variants = (
            first.render_mode == "deterministic_auxiliary"
            and first.value_kind in {"address", "location"}
            and not first.target_paths
            and len(normalized_values) > 1
            and all(
                _edit_distance_at_most_one(normalized_values[0], value)
                for value in normalized_values[1:]
            )
        )
        if not (exact_containing_location_variants or near_ocr_variants):
            output.extend(rows)
            continue
        for normalized_value, variant_rows in sorted(surfaces.items()):
            variant_key = (
                logical_key
                if normalized_value == shortest
                else logical_key + ":typed_variant:" + sha256_bytes(normalized_value.encode())[:12]
            )
            output.extend(
                replace(
                    row,
                    logical_key=variant_key,
                    render_mode="deterministic_auxiliary",
                    derivation=None,
                    dependency_paths=(),
                    dependency_bindings=(),
                    rationale=(
                        row.rationale
                        + " Host split mechanically distinguishable source-only text variants "
                        "into same-scope typed deterministic bindings."
                    ),
                )
                for row in variant_rows
            )
    return _normalize_unique_source_only_identifier_projections(
        raw=raw,
        drafts=merge_drafts(output),
        source_target=source_target,
    )


def validate_target_binding_relationships(
    *, drafts: Sequence[SpanDraft], source_target: Mapping[str, Any], raw: str | None = None
) -> None:
    for draft in drafts:
        relationship = target_path_relationship(source_target, draft.target_paths)
        if draft.render_mode == "target_binding" and relationship == "composite_target_surface":
            from .temperature_prose import composite_instruction_contract

            source_values = {
                path: _resolve_target_path(source_target, path) for path in draft.target_paths
            }
            if (
                composite_instruction_contract(draft.target_paths, draft.source_text, source_values)
                is not None
            ):
                continue
            raise ValueError(
                "direct target binding combines unequal target values: "
                f"{draft.logical_key} ({', '.join(draft.target_paths)})"
            )
    owners: dict[str, set[str]] = defaultdict(set)
    for draft in drafts:
        for path in draft.target_paths:
            owners[path].add(draft.logical_key)
    duplicated = {
        path: sorted(logical_keys) for path, logical_keys in owners.items() if len(logical_keys) > 1
    }
    if duplicated:
        drafts_by_key: dict[str, list[SpanDraft]] = defaultdict(list)
        for draft in drafts:
            drafts_by_key[draft.logical_key].append(draft)
        details = "; ".join(
            f"{path} sourceValue="
            f"{json.dumps(_resolve_target_path(source_target, path), ensure_ascii=False)} -> "
            + ", ".join(
                f"{logical_key}[mode={drafts_by_key[logical_key][0].render_mode}; "
                "sourceTexts="
                + json.dumps(
                    tuple(row.source_text for row in drafts_by_key[logical_key]),
                    ensure_ascii=False,
                )
                + "]"
                for logical_key in logical_keys
            )
            for path, logical_keys in sorted(duplicated.items())
        )
        raise ValueError(
            "target path has multiple logical owners. A target path must have exactly one "
            "logical owner: consolidate its full, repeated, segmented, or token-projection "
            "occurrences into that binding; remove the path from a derived/source-only surface; "
            "or bind a wider exact target scalar without redundantly claiming its embedded "
            "structured facts. Conflicts: " + details
        )

    validate_repeated_binding_fact_topology(drafts=drafts, source_target=source_target, raw=raw)

    requirements = required_target_cobindings(source_target)
    co_binding_errors: list[str] = []
    for requirement in requirements:
        present = {
            path: next(iter(owners[path])) for path in requirement.target_paths if path in owners
        }
        if not present:
            continue
        missing = sorted(set(requirement.target_paths) - set(present))
        if missing:
            co_binding_errors.append(f"{requirement.relationship} missing {','.join(missing)}")
            continue
        logical_keys = set(present.values())
        if len(logical_keys) != 1:
            co_binding_errors.append(
                f"{requirement.relationship} split {','.join(requirement.target_paths)}"
            )
    if co_binding_errors:
        raise ValueError("required target co-binding violations: " + "; ".join(co_binding_errors))


def validate_repeated_binding_fact_topology(
    *, drafts: Sequence[SpanDraft], source_target: Mapping[str, Any], raw: str | None = None
) -> None:
    """Reject one repeated owner that conflates independently mutable target facts.

    This check is independent of global target ownership and required co-bindings. Keeping it as
    a separate host primitive lets the compiler report the defect even when another, unrelated
    overlap prevents construction of the complete candidate state.
    """

    grouped_drafts: dict[str, list[SpanDraft]] = defaultdict(list)
    for draft in drafts:
        grouped_drafts[draft.logical_key].append(draft)
    aggregation_errors: list[str] = []
    repeated_path_sets = (
        tuple(map(frozenset, repeated_cargo_temperature_contract(raw, source_target).values()))
        if raw is not None
        else ()
    )
    for logical_key, occurrences in grouped_drafts.items():
        first = occurrences[0]
        if len(occurrences) < 2 or len(first.target_paths) < 2:
            continue
        independent_components = target_fact_components(source_target, first.target_paths)
        if len(independent_components) <= 1 or first.render_mode == "deterministic_derived":
            continue
        if (
            raw is not None
            and first.render_mode == "target_binding"
            and set(first.target_paths) == set(global_shared_temperature_paths(raw, source_target))
        ):
            # The two document-wide printed instructions and one cargo group
            # prove shared scope; equal labels without that evidence remain rejected.
            continue
        if (
            first.render_mode == "target_binding"
            and frozenset(first.target_paths) in repeated_path_sets
        ):
            # Identical printed commodity, package load, one-to-one reefer
            # allocation, and one global carrying setting prove this equality.
            continue
        if first.render_mode == "agent_residual":
            normalized_occurrences = {
                _normalized_surface(occurrence.source_text) for occurrence in occurrences
            }
            if len(normalized_occurrences) == 1:
                continue
            aggregation_errors.append(
                f"{logical_key} agent residual owns {len(first.target_paths)} target paths from "
                f"{len(independent_components)} independent facts, but its "
                f"{len(occurrences)} occurrences are not the same complete composite"
            )
        else:
            aggregation_errors.append(
                f"{logical_key} owns {len(first.target_paths)} target paths from "
                f"{len(independent_components)} independent facts across "
                f"{len(occurrences)} physical occurrences"
            )
    if aggregation_errors:
        raise ValueError(
            "repeated binding aggregates independently mutable target facts: "
            + "; ".join(aggregation_errors)
        )


def validate_agent_proposal_paths(
    *, proposals: Sequence[AgentBindingProposal], source_target: Mapping[str, Any]
) -> None:
    for proposal in proposals:
        for path in (*proposal.target_paths, *proposal.dependency_paths):
            _resolve_target_path(source_target, path)


def materialize_semantic_only_target_facts(
    *,
    proposals: Sequence[SemanticOnlyTargetFactProposal],
    source_target: Mapping[str, Any],
    provenance: Literal[
        "compiler_audited_unprinted",
        "critic_audited_unprinted",
    ],
) -> tuple[SemanticOnlyTargetFact, ...]:
    paths = tuple(proposal.target_path for proposal in proposals)
    if len(set(paths)) != len(paths):
        raise ValueError("compiler semantic-only target paths must be unique")
    output: list[SemanticOnlyTargetFact] = []
    for proposal in proposals:
        if proposal.target_path in _EXTRACTION_DATE_PATHS:
            raise ValueError(
                "document extraction dates require role-correct OCR evidence, "
                "not a semantic-only target fact: " + proposal.target_path
            )
        source_value = _resolve_target_path(source_target, proposal.target_path)
        if isinstance(source_value, (Mapping, list)):
            raise ValueError(
                "semantic-only target fact must be a scalar leaf: " + proposal.target_path
            )
        output.append(
            SemanticOnlyTargetFact.model_validate(
                {
                    "target_path": proposal.target_path,
                    "source_value": source_value,
                    "provenance": provenance,
                    "rationale": proposal.rationale,
                }
            )
        )
    return tuple(output)


def _policy_for_agent_binding(proposal: AgentBindingProposal) -> str:
    if proposal.target_paths and all(
        re.fullmatch(r"documentPatch\.containers\[\d+\]\.typeDescription", path)
        for path in proposal.target_paths
    ):
        return "natural_text"
    return _render_policy_for_value_kind(proposal.value_kind, proposal.render_mode)


def _render_policy_for_value_kind(value_kind: str, render_mode: str) -> str:
    if render_mode == "deterministic_derived":
        return "derived_surface"
    if value_kind in {"identifier", "equipment"}:
        return "opaque_identifier"
    if value_kind == "date":
        return "date_surface"
    if value_kind in {"integer", "decimal_measurement", "temperature"}:
        return "numeric_surface"
    if value_kind == "package":
        return "categorical_surface"
    return "natural_text"


def _semantic_signature(draft: SpanDraft) -> tuple[Any, ...]:
    return (
        draft.render_mode,
        draft.value_kind,
        draft.group_kind,
        draft.group_key,
        draft.target_paths,
        draft.derivation,
        draft.dependency_paths,
        draft.dependency_bindings,
    )


_SPLITTABLE_NATURAL_VALUE_KINDS = frozenset(
    {
        "organization",
        "person",
        "address",
        "contact_name",
        "location",
        "cargo_text",
        "commercial_text",
        "legal_text",
        "operational_text",
        "other_text",
    }
)


def _target_scalar_surfaces(draft: SpanDraft, source_target: Mapping[str, Any]) -> tuple[str, ...]:
    surfaces: list[str] = []
    for path in draft.target_paths:
        surface = _scalar_surface(_resolve_target_path(source_target, path))
        if surface is None:
            return ()
        surfaces.append(surface)
    return tuple(dict.fromkeys(surfaces))


def _provably_unrelated_target_occurrence(
    draft: SpanDraft, source_target: Mapping[str, Any]
) -> bool:
    if draft.render_mode != "target_binding" or not draft.target_paths:
        return False
    surfaces = _target_scalar_surfaces(draft, source_target)
    source = _normalized_surface(draft.source_text)
    return bool(
        source
        and surfaces
        and all(
            source not in _normalized_surface(surface)
            and _normalized_surface(surface) not in source
            for surface in surfaces
        )
    )


def _reconciled_split_drafts(
    *,
    raw: str,
    outer: SpanDraft,
    inners: Sequence[SpanDraft],
    source_target: Mapping[str, Any],
) -> tuple[SpanDraft, ...]:
    """Partition a natural target surface only when every retained piece is host-provable."""

    if (
        outer.render_mode != "target_binding"
        or outer.render_policy != "natural_text"
        or outer.value_kind not in _SPLITTABLE_NATURAL_VALUE_KINDS
    ):
        return ()
    ordered_inners = tuple(
        sorted(
            (
                inner
                for inner in inners
                if inner.logical_key != outer.logical_key
                and outer.char_start <= inner.char_start
                and inner.char_end <= outer.char_end
                and (outer.char_start, outer.char_end) != (inner.char_start, inner.char_end)
            ),
            key=lambda row: (row.char_start, row.char_end),
        )
    )
    if not ordered_inners:
        return ()
    if any(left.char_end > right.char_start for left, right in pairwise(ordered_inners)):
        return ()
    if any(
        not (
            (inner.char_start == outer.char_start or not raw[inner.char_start - 1].isalnum())
            and (inner.char_end == outer.char_end or not raw[inner.char_end].isalnum())
        )
        for inner in ordered_inners
    ):
        return ()
    target_surfaces = _target_scalar_surfaces(outer, source_target)
    if len(target_surfaces) != 1:
        return ()
    pieces: list[SpanDraft] = []
    boundaries = (
        (outer.char_start, ordered_inners[0].char_start),
        *((left.char_end, right.char_start) for left, right in pairwise(ordered_inners)),
        (ordered_inners[-1].char_end, outer.char_end),
    )
    for char_start, char_end in boundaries:
        while char_start < char_end and not raw[char_start].isalnum():
            char_start += 1
        while char_end > char_start and not raw[char_end - 1].isalnum():
            char_end -= 1
        if char_start == char_end:
            continue
        source_text = raw[char_start:char_end]
        if _matching_token_projection(source_text, target_surfaces[0]) is None:
            return ()
        pieces.append(
            replace(
                outer,
                draft_id=(
                    "reconciled_binding_"
                    + sha256_bytes(f"{outer.draft_id}\0{char_start}\0{char_end}".encode())[:16]
                ),
                char_start=char_start,
                char_end=char_end,
                source_text=source_text,
                evidence_origin="host_verified_agent_proposal",
                rationale=(
                    outer.rationale
                    + " Host deterministically partitioned the natural target projection around "
                    + ", ".join(inner.logical_key for inner in ordered_inners)
                    + "."
                ),
            )
        )
    return tuple(pieces)


def _normalize_required_cobinding_dominance(
    *, drafts: Sequence[SpanDraft], source_target: Mapping[str, Any]
) -> tuple[SpanDraft, ...]:
    """Materialize an exact-span required co-binding before overlap validation.

    Accepted evidence and an agent proposal can independently own complementary paths from one
    host-proven structured fact while selecting the same physical value.  Waiting until the later
    global co-binding normalization is too late because exact-span overlap validation runs first.
    Canonicalize those complementary subsets here only when their union is exactly one required
    co-binding.  Coincidentally equal facts from different required components remain overlapping
    and therefore still fail closed.
    """

    required_sets = {
        frozenset(requirement.target_paths)
        for requirement in required_target_cobindings(source_target)
    }
    if not required_sets:
        return tuple(drafts)
    required_by_path: dict[str, frozenset[str]] = {}
    for required_paths in required_sets:
        for path in required_paths:
            prior = required_by_path.setdefault(path, required_paths)
            if prior != required_paths:
                raise ValueError(f"target path belongs to multiple required co-bindings: {path}")
    by_span: dict[tuple[int, int], list[SpanDraft]] = defaultdict(list)
    for draft in drafts:
        by_span[(draft.char_start, draft.char_end)].append(draft)
    replacements: dict[str, SpanDraft] = {}
    for same_span in by_span.values():
        target_rows = tuple(
            row for row in same_span if row.render_mode == "target_binding" and row.target_paths
        )
        participants_by_requirement: dict[frozenset[str], list[SpanDraft]] = defaultdict(list)
        for row in target_rows:
            row_paths = frozenset(row.target_paths)
            candidate_requirements = {
                required_by_path[path] for path in row_paths if path in required_by_path
            }
            if len(candidate_requirements) != 1:
                continue
            required_paths = next(iter(candidate_requirements))
            if row_paths <= required_paths:
                participants_by_requirement[required_paths].append(row)
        for required_paths, participant_rows in participants_by_requirement.items():
            participants = tuple(participant_rows)
            if (
                not participants
                or frozenset(path for row in participants for path in row.target_paths)
                != required_paths
            ):
                continue
            render_policies = {row.render_policy for row in participants}
            if len(render_policies) != 1:
                continue
            target_paths = tuple(sorted(required_paths))
            render_policy = next(iter(render_policies))
            group_kind, group_key = _canonical_target_group(target_paths)
            canonical = replace(
                participants[0],
                logical_key="anchor:" + "|".join(target_paths),
                value_kind=_value_kind(target_paths, render_policy),
                group_kind=group_kind,
                group_key=group_key,
                target_paths=target_paths,
                rationale=(
                    participants[0].rationale
                    + " Host joined complementary exact-span owners of one required "
                    "structured co-binding."
                ),
            )
            for participant in participants:
                previous_replacement = replacements.setdefault(participant.logical_key, canonical)
                if _semantic_signature(previous_replacement) != _semantic_signature(canonical):
                    return tuple(drafts)
    if not replacements:
        return tuple(drafts)
    normalized: list[SpanDraft] = []
    for draft in drafts:
        winner = replacements.get(draft.logical_key)
        if winner is None:
            normalized.append(draft)
            continue
        normalized.append(
            replace(
                draft,
                logical_key=winner.logical_key,
                render_mode=winner.render_mode,
                value_kind=winner.value_kind,
                group_kind=winner.group_kind,
                group_key=winner.group_key,
                target_paths=winner.target_paths,
                derivation=winner.derivation,
                dependency_paths=winner.dependency_paths,
                dependency_bindings=winner.dependency_bindings,
                render_policy=winner.render_policy,
                rationale=(
                    draft.rationale
                    + " Host promoted the exact required structured co-binding over its "
                    "narrower value-oriented anchor."
                ),
            )
        )
    return tuple(normalized)


def _trim_residual_edges_owned_by_exact_targets(
    *, raw: str, drafts: Sequence[SpanDraft]
) -> tuple[SpanDraft, ...]:
    """Remove a separately owned leading or trailing target from one residual surface.

    The inner target must be wholly contained at an alphanumeric-free edge and its target paths
    must already be declared by the residual. Middle-of-surface values and ambiguous containment
    remain unchanged. This preserves one contiguous residual instead of turning address fragments
    into artificial repeated agent occurrences.
    """

    output: list[SpanDraft] = []
    for outer in drafts:
        if outer.render_mode != "agent_residual" or not outer.target_paths:
            output.append(outer)
            continue
        inners = tuple(
            inner
            for inner in drafts
            if inner is not outer
            and inner.render_mode == "target_binding"
            and inner.target_paths
            and set(inner.target_paths) <= set(outer.target_paths)
            and outer.char_start <= inner.char_start
            and inner.char_end <= outer.char_end
            and (outer.char_start, outer.char_end) != (inner.char_start, inner.char_end)
        )
        if not inners:
            output.append(outer)
            continue
        new_start = outer.char_start
        new_end = outer.char_end
        removed_paths: set[str] = set()
        changed = True
        while changed:
            changed = False
            leading = [
                inner
                for inner in inners
                if inner.char_start >= new_start
                and inner.char_end <= new_end
                and not any(character.isalnum() for character in raw[new_start : inner.char_start])
            ]
            if leading:
                winner = max(leading, key=lambda row: row.char_end)
                new_start = winner.char_end
                removed_paths.update(winner.target_paths)
                changed = True
            trailing = [
                inner
                for inner in inners
                if inner.char_start >= new_start
                and inner.char_end <= new_end
                and not any(character.isalnum() for character in raw[inner.char_end : new_end])
            ]
            if trailing:
                winner = min(trailing, key=lambda row: row.char_start)
                new_end = winner.char_start
                removed_paths.update(winner.target_paths)
                changed = True
        while new_start < new_end and raw[new_start].isspace():
            new_start += 1
        while new_end > new_start and raw[new_end - 1].isspace():
            new_end -= 1
        remaining_paths = tuple(path for path in outer.target_paths if path not in removed_paths)
        if (
            not removed_paths
            or not remaining_paths
            or new_start >= new_end
            or not any(character.isalnum() for character in raw[new_start:new_end])
        ):
            output.append(outer)
            continue
        output.append(
            replace(
                outer,
                draft_id=(
                    "trimmed_residual_"
                    + sha256_bytes(f"{outer.draft_id}\0{new_start}\0{new_end}".encode())[:16]
                ),
                target_paths=remaining_paths,
                char_start=new_start,
                char_end=new_end,
                source_text=raw[new_start:new_end],
                evidence_origin="host_verified_agent_proposal",
                rationale=(
                    outer.rationale
                    + " Host trimmed a residual edge already owned by a disjoint exact target "
                    "binding and removed that target path from the residual contract."
                ),
            )
        )
    return tuple(output)


def _trim_targeted_residual_trailing_footnote_markers(
    *, raw: str, drafts: Sequence[SpanDraft]
) -> tuple[SpanDraft, ...]:
    """Exclude a separated trailing footnote marker from a target-backed residual.

    A marker separated from the last alphanumeric character by horizontal whitespace is layout
    syntax, not part of the modeled value.  This rule deliberately does not touch attached
    punctuation (for example ``S.A.E.``) or source-only residuals.
    """

    output: list[SpanDraft] = []
    for draft in drafts:
        match = (
            _TRAILING_FOOTNOTE_MARKER.fullmatch(draft.source_text)
            if draft.render_mode == "agent_residual" and draft.target_paths
            else None
        )
        if match is None:
            output.append(draft)
            continue
        body = match.group("body")
        new_end = draft.char_start + len(body)
        if raw[draft.char_start : new_end] != body:
            raise ValueError("target-backed residual footnote trim lost source alignment")
        output.append(
            replace(
                draft,
                draft_id=(
                    "trimmed_target_residual_footnote_"
                    + sha256_bytes(f"{draft.draft_id}\0{draft.char_start}\0{new_end}".encode())[:16]
                ),
                char_end=new_end,
                source_text=body,
                evidence_origin="host_verified_agent_proposal",
                rationale=(
                    draft.rationale
                    + " Host excluded a whitespace-separated trailing footnote marker from "
                    "the target-backed residual boundary."
                ),
            )
        )
    return tuple(output)


def _trim_targetless_residual_edges_owned_by_auxiliaries(
    *, raw: str, drafts: Sequence[SpanDraft]
) -> tuple[SpanDraft, ...]:
    """Keep a deterministic source-only edge separate from its containing residual clause."""

    output: list[SpanDraft] = []
    for outer in drafts:
        if (
            (
                outer.render_mode != "agent_residual"
                and not (
                    outer.render_mode == "deterministic_auxiliary"
                    and outer.value_kind == "operational_text"
                )
            )
            or outer.target_paths
            or outer.dependency_paths
            or outer.dependency_bindings
        ):
            output.append(outer)
            continue
        inners = tuple(
            inner
            for inner in drafts
            if inner is not outer
            and inner.render_mode == "deterministic_auxiliary"
            and not inner.target_paths
            and not inner.dependency_paths
            and not inner.dependency_bindings
            and outer.char_start <= inner.char_start
            and inner.char_end <= outer.char_end
            and (outer.char_start, outer.char_end) != (inner.char_start, inner.char_end)
        )
        new_start = outer.char_start
        new_end = outer.char_end
        trimmed_inner_ids: list[str] = []
        changed = True
        while changed:
            changed = False
            leading = tuple(
                inner
                for inner in inners
                if inner.char_start >= new_start
                and inner.char_end <= new_end
                and not any(character.isalnum() for character in raw[new_start : inner.char_start])
            )
            if len(leading) == 1:
                new_start = leading[0].char_end
                trimmed_inner_ids.append(leading[0].draft_id)
                changed = True
            trailing = tuple(
                inner
                for inner in inners
                if inner.char_start >= new_start
                and inner.char_end <= new_end
                and not any(character.isalnum() for character in raw[inner.char_end : new_end])
            )
            if len(trailing) == 1:
                new_end = trailing[0].char_start
                trimmed_inner_ids.append(trailing[0].draft_id)
                changed = True
        while new_start < new_end and raw[new_start].isspace():
            new_start += 1
        while new_end > new_start and raw[new_end - 1].isspace():
            new_end -= 1
        if (
            not trimmed_inner_ids
            or new_start >= new_end
            or not any(character.isalnum() for character in raw[new_start:new_end])
        ):
            output.append(outer)
            continue
        output.append(
            replace(
                outer,
                draft_id=(
                    "trimmed_source_residual_"
                    + sha256_bytes(f"{outer.draft_id}\0{new_start}\0{new_end}".encode())[:16]
                ),
                char_start=new_start,
                char_end=new_end,
                source_text=raw[new_start:new_end],
                evidence_origin="host_verified_agent_proposal",
                rationale=(
                    outer.rationale
                    + " Host trimmed deterministic source-only edge owner(s) "
                    + ", ".join(sorted(set(trimmed_inner_ids)))
                    + " from this residual clause."
                ),
            )
        )
    return tuple(output)


def _merge_adjacent_agent_residual_segments(
    *, raw: str, drafts: Sequence[SpanDraft]
) -> tuple[SpanDraft, ...]:
    """Rejoin one local residual split only around literal punctuation or line layout.

    A repair can legitimately express one composite address or cargo phrase as adjacent pieces
    after excluding a separately owned nested value.  Treating those pieces as repeated semantic
    appearances makes the topology validator reject a faithful repair.  The merge is allowed only
    across a same-line or immediately adjacent-line gap with no alphanumeric content and no other
    binding in the widened span.  Anything less local remains unchanged and fails closed later.
    """

    grouped: dict[str, list[SpanDraft]] = defaultdict(list)
    for draft in drafts:
        grouped[draft.logical_key].append(draft)
    replacements: dict[str, SpanDraft] = {}
    discarded_ids: set[str] = set()
    for rows in grouped.values():
        if (
            len(rows) < 2
            or rows[0].render_mode != "agent_residual"
            or len({_semantic_signature(row) for row in rows}) != 1
        ):
            continue
        ordered = sorted(rows, key=lambda row: (row.char_start, row.char_end))
        clusters: list[list[SpanDraft]] = [[ordered[0]]]
        for row in ordered[1:]:
            previous = clusters[-1][-1]
            gap = raw[previous.char_end : row.char_start]
            if (
                previous.char_end <= row.char_start
                and not any(character.isalnum() for character in gap)
                and gap.count("\n") <= 1
            ):
                clusters[-1].append(row)
            else:
                clusters.append([row])
        for cluster in clusters:
            if len(cluster) < 2:
                continue
            char_start = cluster[0].char_start
            char_end = cluster[-1].char_end
            cluster_ids = {row.draft_id for row in cluster}
            if any(
                other.draft_id not in cluster_ids
                and other.char_start < char_end
                and char_start < other.char_end
                for other in drafts
            ):
                continue
            merged = replace(
                cluster[0],
                draft_id=(
                    "reconciled_residual_"
                    + sha256_bytes(f"{cluster[0].logical_key}\0{char_start}\0{char_end}".encode())[
                        :16
                    ]
                ),
                char_start=char_start,
                char_end=char_end,
                source_text=raw[char_start:char_end],
                evidence_origin="host_verified_agent_proposal",
                rationale=(
                    cluster[0].rationale
                    + " Host rejoined adjacent residual segments separated only by literal "
                    "punctuation or line layout."
                ),
            )
            replacements[cluster[0].draft_id] = merged
            discarded_ids.update(row.draft_id for row in cluster[1:])
    return tuple(
        replacements.get(draft.draft_id, draft)
        for draft in drafts
        if draft.draft_id not in discarded_ids
    )


def _normalize_derived_dependency_containment(
    drafts: Sequence[SpanDraft],
) -> tuple[SpanDraft, ...]:
    """Let a complete deterministic surface own target text embedded inside it."""

    grouped: dict[str, list[SpanDraft]] = defaultdict(list)
    for draft in drafts:
        grouped[draft.logical_key].append(draft)
    discarded_keys: set[str] = set()
    for inner_key, inner_rows in grouped.items():
        inner = inner_rows[0]
        if inner.render_mode != "target_binding" or not inner.target_paths:
            continue
        candidates: list[str] = []
        for outer_key, outer_rows in grouped.items():
            outer = outer_rows[0]
            if (
                outer_key == inner_key
                or outer.render_mode != "deterministic_derived"
                or not set(inner.target_paths) <= set(outer.dependency_paths)
            ):
                continue
            if all(
                any(
                    outer_row.char_start <= inner_row.char_start
                    and inner_row.char_end <= outer_row.char_end
                    for outer_row in outer_rows
                )
                for inner_row in inner_rows
            ):
                candidates.append(outer_key)
        if len(candidates) == 1:
            discarded_keys.add(inner_key)
    return tuple(draft for draft in drafts if draft.logical_key not in discarded_keys)


def _expand_equipment_receipt_dependency_overlaps(
    *, raw: str, drafts: Sequence[SpanDraft]
) -> tuple[SpanDraft, ...]:
    """Complete a receipt surface across its overlapping structured type dependency.

    Risk extraction can expose the alphanumeric receipt prefix while an accepted target anchor
    owns the overlapping equipment-type suffix. Their union is mechanically one contiguous
    count-times-equipment surface when both bindings share row scope and the target path is an
    explicit receipt dependency. Expanding the derived owner lets the existing dependency-
    containment rule remove the now-redundant inner owner before overlap validation.
    """

    grouped: dict[str, list[SpanDraft]] = defaultdict(list)
    for draft in drafts:
        grouped[draft.logical_key].append(draft)
    replacements: dict[str, SpanDraft] = {}
    for outer_key, outer_rows in grouped.items():
        outer = outer_rows[0]
        if (
            outer.render_mode != "deterministic_derived"
            or outer.derivation != "equipment_receipt"
            or _EQUIPMENT_RECEIPT_PREFIX.match(outer.source_text) is None
        ):
            continue
        candidate_inner_groups = [
            inner_rows
            for inner_key, inner_rows in grouped.items()
            if inner_key != outer_key
            and inner_rows[0].render_mode == "target_binding"
            and inner_rows[0].group_kind == outer.group_kind
            and inner_rows[0].group_key == outer.group_key
            and bool(inner_rows[0].target_paths)
            and set(inner_rows[0].target_paths) <= set(outer.dependency_paths)
        ]
        if len(candidate_inner_groups) != 1:
            continue
        inner_rows = candidate_inner_groups[0]
        if len(inner_rows) != len(outer_rows):
            continue
        pairings: dict[str, SpanDraft] = {}
        pairing_is_unique = True
        for outer_row in outer_rows:
            matches = [
                inner_row
                for inner_row in inner_rows
                if outer_row.char_start < inner_row.char_end
                and inner_row.char_start < outer_row.char_end
            ]
            if len(matches) != 1 or matches[0].draft_id in {
                row.draft_id for row in pairings.values()
            }:
                pairing_is_unique = False
                break
            pairings[outer_row.draft_id] = matches[0]
        if not pairing_is_unique or len(pairings) != len(inner_rows):
            continue
        expanded: list[SpanDraft] = []
        for outer_row in outer_rows:
            inner_row = pairings[outer_row.draft_id]
            char_start = min(outer_row.char_start, inner_row.char_start)
            char_end = max(outer_row.char_end, inner_row.char_end)
            source_text = raw[char_start:char_end]
            if _EQUIPMENT_RECEIPT_PREFIX.match(source_text) is None:
                expanded = []
                break
            expanded.append(
                replace(
                    outer_row,
                    char_start=char_start,
                    char_end=char_end,
                    source_text=source_text,
                    evidence_origin="host_verified_agent_proposal",
                    rationale=(
                        outer_row.rationale
                        + " Host completed the contiguous equipment receipt across its "
                        "overlapping structured type dependency."
                    ),
                )
            )
        if expanded:
            replacements.update({row.draft_id: row for row in expanded})
    return tuple(replacements.get(draft.draft_id, draft) for draft in drafts)


def _has_fmc_carrier_identifier_prefix(
    *, raw: str, lines: Sequence[LineSpan], draft: SpanDraft
) -> bool:
    line = next(
        (row for row in lines if row.char_start <= draft.char_start < row.char_end),
        None,
    )
    if line is None or draft.char_end > line.char_end:
        return False
    return (
        _FMC_CARRIER_IDENTIFIER_PREFIX.search(raw[line.char_start : draft.char_start]) is not None
    )


def _normalize_fmc_carrier_duplicate_ownership(
    *, raw: str, drafts: Sequence[SpanDraft]
) -> tuple[SpanDraft, ...]:
    """Prefer the fixed-carrier owner for an exact duplicate FMC identifier group."""

    grouped: dict[str, list[SpanDraft]] = defaultdict(list)
    for draft in drafts:
        grouped[draft.logical_key].append(draft)
    lines = line_spans(raw)
    discarded_keys: set[str] = set()
    for carrier_key, carrier_rows in grouped.items():
        carrier = carrier_rows[0]
        if (
            carrier.render_mode != "carrier_static"
            or carrier.group_kind != "carrier"
            or not all(
                _has_fmc_carrier_identifier_prefix(raw=raw, lines=lines, draft=row)
                for row in carrier_rows
            )
        ):
            continue
        carrier_spans = {(row.char_start, row.char_end, row.source_text) for row in carrier_rows}
        for other_key, other_rows in grouped.items():
            other = other_rows[0]
            if (
                other_key != carrier_key
                and other.render_mode == "deterministic_auxiliary"
                and other.value_kind == carrier.value_kind
                and {(row.char_start, row.char_end, row.source_text) for row in other_rows}
                == carrier_spans
            ):
                discarded_keys.add(other_key)
    return tuple(draft for draft in drafts if draft.logical_key not in discarded_keys)


def _normalize_carrier_static_target_ownership(
    *, drafts: Sequence[SpanDraft], source_target: Mapping[str, Any]
) -> tuple[SpanDraft, ...]:
    """Keep carrier aliases static while assigning each target path one provenance owner.

    Carrier-bound aliases, domains, and signature entities never change during descendant
    rendering. They may therefore remain distinct typed static bindings, but only one logical
    owner may retain a structured carrier path. Prefer exact canonical target evidence and then
    accepted label evidence; the choice changes provenance only, never rendered text.
    """

    grouped: dict[str, list[SpanDraft]] = defaultdict(list)
    for draft in drafts:
        grouped[draft.logical_key].append(draft)
    owners: dict[str, set[str]] = defaultdict(set)
    for logical_key, rows in grouped.items():
        if rows[0].render_mode != "carrier_static":
            continue
        for path in rows[0].target_paths:
            owners[path].add(logical_key)
    winners: dict[str, str] = {}
    for path, logical_keys in owners.items():
        if len(logical_keys) < 2:
            continue
        target_surface = _scalar_surface(_resolve_target_path(source_target, path))
        normalized_target = _normalized_surface(target_surface or "")

        def priority(
            logical_key: str, target_value: str = normalized_target
        ) -> tuple[int, int, str]:
            rows = grouped[logical_key]
            exact = bool(target_value) and any(
                _normalized_surface(row.source_text) == target_value for row in rows
            )
            accepted = any(row.evidence_origin == "accepted_label_evidence" for row in rows)
            return (0 if exact else 1, 0 if accepted else 1, logical_key)

        winners[path] = min(logical_keys, key=priority)
    if not winners:
        return tuple(drafts)
    output: list[SpanDraft] = []
    for draft in drafts:
        if draft.render_mode != "carrier_static":
            output.append(draft)
            continue
        retained_paths = tuple(
            path
            for path in draft.target_paths
            if path not in winners or winners[path] == draft.logical_key
        )
        output.append(
            replace(
                draft,
                target_paths=retained_paths,
                rationale=(
                    draft.rationale
                    + " Host retained this carrier-bound static surface while assigning shared "
                    "structured provenance to one canonical owner."
                    if retained_paths != draft.target_paths
                    else draft.rationale
                ),
            )
        )
    return tuple(output)


_MEASUREMENT_COMPONENT_PATH = re.compile(
    r"^(?P<base>.+\.(?:temperatureSetpoint|temperature|grossWeight|netWeight|tareWeight|volume))\."
    r"(?P<field>unit|value)$"
)


_PACKAGE_KIND_PATH = re.compile(
    r"^documentPatch\.cargoPackages\[([0-9]+)\]\.(?:typeCategory|typeDescription)$"
)


def _measurement_component_spans(
    source: str,
    *,
    value: Any,
    unit: Any,
) -> tuple[tuple[int, int], tuple[int, int]] | None:
    value_span = _matching_numeric_span(source, value, integer=False)
    accepted_units = (
        _MEASUREMENT_UNIT_SURFACES.get(unit.casefold()) if isinstance(unit, str) else None
    )
    if value_span is None or not accepted_units:
        return None
    unit_candidates: list[tuple[int, int]] = []
    for start, end in ((0, value_span[0]), (value_span[1], len(source))):
        tokens = _surface_token_spans(source[start:end])
        if tokens and "".join(token for token, _left, _right in tokens) in accepted_units:
            unit_candidates.append((start + tokens[0][1], start + tokens[-1][2]))
    if len(unit_candidates) != 1:
        return None
    unit_span = unit_candidates[0]
    if value_span[0] < unit_span[1] and unit_span[0] < value_span[1]:
        return None
    if any(
        character.isalnum()
        for index, character in enumerate(source)
        if not (value_span[0] <= index < value_span[1] or unit_span[0] <= index < unit_span[1])
    ):
        return None
    return value_span, unit_span


def _split_inline_structured_component_bindings(
    *, drafts: Sequence[SpanDraft], source_target: Mapping[str, Any]
) -> tuple[SpanDraft, ...]:
    """Split an inline typed value/category contract into independently mutable leaves."""

    grouped: dict[str, list[SpanDraft]] = defaultdict(list)
    for draft in drafts:
        grouped[draft.logical_key].append(draft)
    discarded: set[str] = set()
    additions: list[SpanDraft] = []
    for logical_key, rows in grouped.items():
        first = rows[0]
        if not (
            first.render_mode in {"target_binding", "agent_residual"}
            and len(first.target_paths) == 2
            and all(row.target_paths == first.target_paths for row in rows)
        ):
            continue
        matches = tuple(_MEASUREMENT_COMPONENT_PATH.fullmatch(path) for path in first.target_paths)
        typed_matches = tuple(match for match in matches if match is not None)
        if (
            len(typed_matches) == 2
            and len({match.group("base") for match in typed_matches}) == 1
            and {match.group("field") for match in typed_matches} == {"unit", "value"}
        ):
            paths_by_field = {
                match.group("field"): path
                for path, match in zip(first.target_paths, typed_matches, strict=True)
            }
            fields = ("value", "unit")
            components = tuple(
                _measurement_component_spans(
                    row.source_text,
                    value=_resolve_target_path(source_target, paths_by_field["value"]),
                    unit=_resolve_target_path(source_target, paths_by_field["unit"]),
                )
                for row in rows
            )
        else:
            quantity = next(
                (
                    path
                    for path in first.target_paths
                    if _PACKAGE_QUANTITY_PATH.fullmatch(path) is not None
                ),
                None,
            )
            category = next(
                (path for path in first.target_paths if _PACKAGE_KIND_PATH.fullmatch(path)),
                None,
            )
            quantity_match = _PACKAGE_QUANTITY_PATH.fullmatch(quantity or "")
            category_match = _PACKAGE_KIND_PATH.fullmatch(category or "")
            if (
                quantity is None
                or category is None
                or quantity_match is None
                or category_match is None
                or quantity_match.group(1) != category_match.group(1)
            ):
                continue
            paths_by_field = {"quantity": quantity, "category": category}
            fields = ("quantity", "category")
            components = tuple(
                _package_component_spans(
                    row.source_text,
                    quantity=_resolve_target_path(source_target, quantity),
                    category=_resolve_target_path(source_target, category),
                )
                for row in rows
            )
        if any(component is None for component in components):
            continue
        discarded.update(row.draft_id for row in rows)
        for row, component in zip(rows, components, strict=True):
            assert component is not None
            for field, span in zip(fields, component, strict=True):
                path = paths_by_field[field]
                start = row.char_start + span[0]
                end = row.char_start + span[1]
                value_kind = _value_kind((path,), row.render_policy)
                group_kind, group_key = _canonical_target_group((path,))
                additions.append(
                    replace(
                        row,
                        draft_id=(
                            "host_measurement_component_"
                            + sha256_bytes(f"{logical_key}\0{field}\0{start}\0{end}".encode())[:16]
                        ),
                        logical_key="anchor:" + path,
                        render_mode="target_binding",
                        value_kind=value_kind,
                        group_kind=group_kind,
                        group_key=group_key,
                        target_paths=(path,),
                        derivation=None,
                        dependency_paths=(),
                        dependency_bindings=(),
                        char_start=start,
                        char_end=end,
                        source_text=row.source_text[span[0] : span[1]],
                        evidence_origin="host_verified_agent_proposal",
                        render_policy=_render_policy_for_value_kind(value_kind, "target_binding"),
                        rationale=(
                            row.rationale
                            + " Host split this inline measurement into independently mutable "
                            f"{field} and unit/value leaf ownership."
                        ),
                    )
                )
    return (*tuple(row for row in drafts if row.draft_id not in discarded), *additions)


def _partition_structured_component_bindings(
    *, drafts: Sequence[SpanDraft], source_target: Mapping[str, Any]
) -> tuple[SpanDraft, ...]:
    """Split compiler-bundled package and measurement components by source shape.

    Quantity/value and type/unit leaves are independently mutable even when the provider emits
    one repeated logical binding.  This repair is allowed only for the two closed structured
    families below and only when every occurrence has exactly one mechanically identifiable
    component role and both original paths remain owned.
    """

    grouped: dict[str, list[SpanDraft]] = defaultdict(list)
    for draft in drafts:
        grouped[draft.logical_key].append(draft)
    replacements: dict[str, SpanDraft] = {}
    for rows in grouped.values():
        first = rows[0]
        if not (
            first.render_mode == "target_binding"
            and len(first.target_paths) == 2
            and len(rows) >= 2
            and all(row.target_paths == first.target_paths for row in rows)
        ):
            continue

        assignments: dict[str, str] = {}
        measurement_matches = tuple(
            _MEASUREMENT_COMPONENT_PATH.fullmatch(path) for path in first.target_paths
        )
        if all(match is not None for match in measurement_matches):
            typed_measurements = tuple(match for match in measurement_matches if match is not None)
            bases = {match.group("base") for match in typed_measurements}
            fields = {match.group("field") for match in typed_measurements}
            if len(bases) == 1 and fields == {"unit", "value"}:
                paths_by_field = {
                    match.group("field"): path
                    for path, match in zip(first.target_paths, typed_measurements, strict=True)
                }
                unit_value = _resolve_target_path(source_target, paths_by_field["unit"])
                accepted_units = (
                    _MEASUREMENT_UNIT_SURFACES.get(unit_value.casefold())
                    if isinstance(unit_value, str)
                    else None
                )
                if accepted_units:
                    for row in rows:
                        normalized = _normalized_surface(row.source_text)
                        if normalized in accepted_units:
                            assignments[row.draft_id] = paths_by_field["unit"]
                        elif (
                            re.fullmatch(r"\s*[-+]?[0-9][0-9., ]*\s*", row.source_text) is not None
                        ):
                            assignments[row.draft_id] = paths_by_field["value"]

        if not assignments:
            quantity = next(
                (
                    path
                    for path in first.target_paths
                    if _PACKAGE_QUANTITY_PATH.fullmatch(path) is not None
                ),
                None,
            )
            kind = next(
                (path for path in first.target_paths if _PACKAGE_KIND_PATH.fullmatch(path)),
                None,
            )
            quantity_match = _PACKAGE_QUANTITY_PATH.fullmatch(quantity or "")
            kind_match = _PACKAGE_KIND_PATH.fullmatch(kind or "")
            if (
                quantity is not None
                and kind is not None
                and quantity_match is not None
                and kind_match is not None
                and quantity_match.group(1) == kind_match.group(1)
            ):
                for row in rows:
                    if re.fullmatch(r"\s*[0-9][0-9., ]*\s*", row.source_text) is not None:
                        assignments[row.draft_id] = quantity
                    elif any(character.isalpha() for character in row.source_text) and not any(
                        character.isdigit() for character in row.source_text
                    ):
                        assignments[row.draft_id] = kind

        if len(assignments) != len(rows) or set(assignments.values()) != set(first.target_paths):
            continue
        for row in rows:
            path = assignments[row.draft_id]
            value_kind = _value_kind((path,), row.render_policy)
            group_kind, group_key = _canonical_target_group((path,))
            replacements[row.draft_id] = replace(
                row,
                logical_key="anchor:" + path,
                value_kind=value_kind,
                group_kind=group_kind,
                group_key=group_key,
                target_paths=(path,),
                render_policy=_render_policy_for_value_kind(value_kind, "target_binding"),
                rationale=(
                    row.rationale + " Host partitioned this compiler-bundled structured value/type "
                    "component from its mechanically distinct source shape."
                ),
            )
    return tuple(replacements.get(row.draft_id, row) for row in drafts)


def _partition_compiler_multitarget_projections(
    *, drafts: Sequence[SpanDraft], source_target: Mapping[str, Any]
) -> tuple[SpanDraft, ...]:
    """Partition a compiler binding when each occurrence proves a different target subset.

    Exact target realizations take precedence over a looser token projection.  The partition is
    accepted only when every original target belongs to exactly one resulting logical contract;
    otherwise the candidate remains unchanged for explicit review.
    """

    grouped: dict[str, list[SpanDraft]] = defaultdict(list)
    for draft in drafts:
        grouped[draft.logical_key].append(draft)
    replacements: dict[str, SpanDraft] = {}
    for rows in grouped.values():
        first = rows[0]
        if not (
            first.render_mode in {"target_binding", "agent_residual"}
            and len(first.target_paths) > 1
            and len(rows) > 1
            and all(row.evidence_origin != "accepted_label_evidence" for row in rows)
            and all(row.target_paths == first.target_paths for row in rows)
            and len(target_fact_components(source_target, first.target_paths)) > 1
        ):
            continue
        surfaces = {
            path: _scalar_surface(_resolve_target_path(source_target, path))
            for path in first.target_paths
        }
        if any(surface is None for surface in surfaces.values()):
            continue
        assignments: dict[str, tuple[str, ...]] = {}
        for row in rows:
            projections: list[str] = []
            exact: list[str] = []
            for path, surface in surfaces.items():
                assert surface is not None
                match = _matching_token_projection(row.source_text, surface)
                if match is None:
                    continue
                projections.append(path)
                if match == ((), ()):
                    exact.append(path)
            selected = tuple(sorted(exact or projections))
            if not selected:
                break
            assignments[row.draft_id] = selected
        if len(assignments) != len(rows) or len(set(assignments.values())) < 2:
            continue
        groups_by_path: dict[str, set[tuple[str, ...]]] = defaultdict(set)
        for target_paths in assignments.values():
            for path in target_paths:
                groups_by_path[path].add(target_paths)
        if set(groups_by_path) != set(first.target_paths) or any(
            len(groups) != 1 for groups in groups_by_path.values()
        ):
            continue
        for row in rows:
            target_paths = assignments[row.draft_id]
            group_kind, group_key = _canonical_target_group(target_paths)
            render_policy = _render_policy_for_value_kind(
                _value_kind(target_paths, row.render_policy), "target_binding"
            )
            replacements[row.draft_id] = replace(
                row,
                logical_key="anchor:" + "|".join(target_paths),
                render_mode="target_binding",
                value_kind=_value_kind(target_paths, render_policy),
                group_kind=group_kind,
                group_key=group_key,
                target_paths=target_paths,
                render_policy=render_policy,
                rationale=(
                    row.rationale
                    + " Host partitioned this compiler binding by exact-versus-projected target "
                    "evidence; every original target belongs to exactly one resulting contract."
                ),
            )
    return tuple(replacements.get(row.draft_id, row) for row in drafts)


def _normalize_hazard_class_binding_contracts(
    *, raw: str, drafts: Sequence[SpanDraft]
) -> tuple[SpanDraft, ...]:
    """Canonicalize bare and captioned DG class occurrences as one bounded semantic owner."""

    grouped: dict[str, list[SpanDraft]] = defaultdict(list)
    for draft in drafts:
        grouped[draft.logical_key].append(draft)
    replacements: dict[str, SpanDraft] = {}
    captions: dict[tuple[int, int], SpanDraft] = {}
    for rows in grouped.values():
        first = rows[0]
        if not (
            len(first.target_paths) == 1
            and first.target_paths[0].endswith(".hazardCategory")
            and all(row.target_paths == first.target_paths for row in rows)
        ):
            continue
        matches = tuple(
            _DANGEROUS_GOODS_CLASS_SURFACE.fullmatch(row.source_text)
            or _BARE_DANGEROUS_GOODS_CLASS_SURFACE.fullmatch(row.source_text)
            for row in rows
        )
        if any(match is None for match in matches) or not any(
            _DANGEROUS_GOODS_CLASS_SURFACE.fullmatch(row.source_text) is not None for row in rows
        ):
            continue
        typed_matches = tuple(match for match in matches if match is not None)
        if len({_normalized_surface(match.group("value")) for match in typed_matches}) != 1:
            continue
        target_path = first.target_paths[0]
        group_kind, group_key = _canonical_target_group((target_path,))
        for row, match in zip(rows, typed_matches, strict=True):
            start = row.char_start + match.start("value")
            end = row.char_start + match.end("value")
            caption_tokens = tuple(
                (token_start, token_end)
                for token, token_start, token_end in _surface_token_spans(
                    row.source_text[: match.start("value")]
                )
                if token in {"cl", "class"}
            )
            if len(caption_tokens) == 1:
                caption_start = row.char_start + caption_tokens[0][0]
                caption_end = row.char_start + caption_tokens[0][1]
                captions[(caption_start, caption_end)] = SpanDraft(
                    draft_id=(
                        "host_hazard_class_caption_"
                        + sha256_bytes(f"{caption_start}\0{caption_end}".encode())[:16]
                    ),
                    logical_key="agent:static:dangerous_goods:class_caption",
                    render_mode="literal_static",
                    value_kind="other_text",
                    group_kind="dangerous_goods",
                    group_key="dangerous_goods:class_caption",
                    target_paths=(),
                    derivation=None,
                    dependency_paths=(),
                    dependency_bindings=(),
                    char_start=caption_start,
                    char_end=caption_end,
                    source_text=raw[caption_start:caption_end],
                    evidence_origin="host_verified_agent_proposal",
                    render_policy="natural_text",
                    rationale=(
                        "Host separated and explicitly owned this immutable dangerous-goods "
                        "class caption from its semantically rendered class value."
                    ),
                )
            replacements[row.draft_id] = replace(
                row,
                logical_key="anchor:" + target_path,
                render_mode="agent_residual",
                value_kind="dangerous_goods",
                group_kind=group_kind,
                group_key=group_key,
                target_paths=(target_path,),
                derivation=None,
                dependency_paths=(),
                dependency_bindings=(),
                char_start=start,
                char_end=end,
                source_text=raw[start:end],
                evidence_origin="host_verified_agent_proposal",
                render_policy="natural_text",
                rationale=(
                    row.rationale
                    + " Host separated the literal CLASS caption and retained the regulatory "
                    "category-to-class conversion as one bounded semantic contract."
                ),
            )
    return (*tuple(replacements.get(row.draft_id, row) for row in drafts), *captions.values())


def _promote_non_equivalent_auxiliary_groups(
    *, drafts: Sequence[SpanDraft]
) -> tuple[SpanDraft, ...]:
    """Route differently formatted copies of one source-only fact to bounded agent editing."""

    grouped: dict[str, list[SpanDraft]] = defaultdict(list)
    for draft in drafts:
        grouped[draft.logical_key].append(draft)
    promoted = {
        logical_key
        for logical_key, rows in grouped.items()
        if rows[0].render_mode == "deterministic_auxiliary"
        and len(rows) > 1
        and len({_normalized_surface(row.source_text) for row in rows}) > 1
    }
    return tuple(
        replace(
            row,
            render_mode="agent_residual",
            render_policy="natural_text",
            evidence_origin="host_verified_agent_proposal",
            rationale=(
                row.rationale
                + " Host routed non-equivalent physical variants of this one source-only fact "
                "to a bounded shared agent contract."
            ),
        )
        if row.logical_key in promoted
        else row
        for row in drafts
    )


def _adjacent_source_only_numeric_owner(
    *,
    raw: str,
    lines: Sequence[LineSpan],
    drafts: Sequence[SpanDraft],
    unit: SpanDraft,
) -> SpanDraft | None:
    """Return the sole source-only numeric owner immediately before one unit token."""

    unit_line = line_number_for_char(lines, unit.char_start)
    predecessors = tuple(
        candidate
        for candidate in drafts
        if candidate.draft_id != unit.draft_id
        and candidate.group_kind == unit.group_kind
        and candidate.group_key == unit.group_key
        and candidate.render_mode in {"deterministic_auxiliary", "agent_residual"}
        and candidate.value_kind in {"decimal_measurement", "integer"}
        and not candidate.target_paths
        and not candidate.dependency_paths
        and not candidate.dependency_bindings
        and candidate.char_end <= unit.char_start
        and line_number_for_char(lines, candidate.char_end - 1) == unit_line
        and not any(character.isalnum() for character in raw[candidate.char_end : unit.char_start])
    )
    if not predecessors:
        return None
    nearest_end = max(row.char_end for row in predecessors)
    nearest = tuple(row for row in predecessors if row.char_end == nearest_end)
    return nearest[0] if len(nearest) == 1 else None


def _normalize_measurement_unit_component_locality(
    *, raw: str, drafts: Sequence[SpanDraft], source_target: Mapping[str, Any]
) -> tuple[SpanDraft, ...]:
    """Relocate a structured unit to the unique token immediately following its value sibling."""

    by_base: dict[str, dict[str, list[SpanDraft]]] = defaultdict(lambda: {"unit": [], "value": []})
    for draft in drafts:
        if draft.render_mode != "target_binding" or len(draft.target_paths) != 1:
            continue
        match = _MEASUREMENT_COMPONENT_PATH.fullmatch(draft.target_paths[0])
        if match is not None:
            by_base[match.group("base")][match.group("field")].append(draft)
    lines = line_spans(raw)
    replacements: dict[str, SpanDraft] = {}
    discarded: set[str] = set()
    for base, components in by_base.items():
        units = components["unit"]
        values = components["value"]
        if not units or not values or len({row.logical_key for row in units}) != 1:
            continue
        try:
            target_unit = _resolve_target_path(source_target, base + ".unit")
        except ValueError:
            continue
        if not isinstance(target_unit, str):
            continue
        accepted_units = _MEASUREMENT_UNIT_SURFACES.get(target_unit.casefold())
        if not accepted_units:
            continue
        base_field = base.rsplit(".", maxsplit=1)[-1]
        caption_units: list[SpanDraft] = []
        for unit in units:
            if _normalized_surface(unit.source_text) not in accepted_units:
                continue
            line_number = line_number_for_char(lines, unit.char_start)
            line = lines[line_number - 1]
            if _trimmed_line_span(line) != (unit.char_start, unit.char_end):
                continue
            preceding = next(
                (
                    candidate
                    for candidate in reversed(lines[: line_number - 1])
                    if candidate.text.strip()
                ),
                None,
            )
            if preceding is None or _PAGE_HEADER.fullmatch(preceding.text.strip()):
                continue
            fields = tuple(
                field
                for caption_pattern, field in _MEASUREMENT_CAPTION_FIELDS
                if caption_pattern.search(preceding.text)
            )
            if fields == (base_field,):
                caption_units.append(unit)
        candidates: list[tuple[int, int]] = []
        for value in values:
            line_number = line_number_for_char(lines, value.char_start)
            if line_number != line_number_for_char(lines, value.char_end - 1):
                break
            line = lines[line_number - 1]
            local = tuple(
                (line.char_start + start, line.char_start + end)
                for token, start, end in _surface_token_spans(line.text)
                if token in accepted_units
                and line.char_start + start >= value.char_end
                and not any(
                    character.isalnum()
                    for character in raw[value.char_end : line.char_start + start]
                )
            )
            if len(local) != 1:
                break
            candidates.append(local[0])
        if len(candidates) != len(values) or len(set(candidates)) != len(candidates):
            continue
        unit_ids = {row.draft_id for row in units}
        if any(
            start < other.char_end and other.char_start < end and other.draft_id not in unit_ids
            for start, end in candidates
            for other in drafts
        ):
            continue
        owner = units[0]
        candidate_spans = set(candidates)
        caption_spans = {(unit.char_start, unit.char_end) for unit in caption_units}
        auxiliary_units: dict[str, SpanDraft] = {}
        all_units_classified = True
        for unit in units:
            span = (unit.char_start, unit.char_end)
            if span in candidate_spans or span in caption_spans:
                continue
            normalized_unit = _normalized_surface(unit.source_text)
            source_only_owner = _adjacent_source_only_numeric_owner(
                raw=raw,
                lines=lines,
                drafts=drafts,
                unit=unit,
            )
            if normalized_unit not in accepted_units:
                all_units_classified = False
                break
            if source_only_owner is None:
                embedded_alphanumeric_fragment = (
                    unit.char_start > 0 and raw[unit.char_start - 1].isalnum()
                ) or (unit.char_end < len(raw) and raw[unit.char_end].isalnum())
                if embedded_alphanumeric_fragment:
                    # The selected bytes are a substring of a different word (for example the
                    # C in CLASS), while a unique local unit beside the value is already proved.
                    # Dropping this misplaced occurrence cannot remove a physical unit surface.
                    continue
                all_units_classified = False
                break
            replacement_id = (
                "host_source_only_measurement_unit_"
                + sha256_bytes(f"{normalized_unit}\0{unit.char_start}\0{unit.char_end}".encode())[
                    :16
                ]
            )
            auxiliary_units[replacement_id] = replace(
                unit,
                draft_id=replacement_id,
                logical_key=f"agent:cargo:measurement_unit:{normalized_unit}",
                render_mode="deterministic_auxiliary",
                value_kind="other_text",
                group_kind="cargo",
                group_key=f"cargo:measurement_unit:{normalized_unit}",
                target_paths=(),
                derivation=None,
                dependency_paths=(),
                dependency_bindings=(),
                evidence_origin="host_verified_agent_proposal",
                render_policy="natural_text",
                rationale=(
                    unit.rationale
                    + " Host preserved this over-broad structured-unit occurrence as exact "
                    "source-only measurement vocabulary because it immediately follows one "
                    "independently mutable numeric cargo value."
                ),
            )
        if not all_units_classified:
            # At least one unmatched unit was not mechanically classifiable. Leave the complete
            # owner untouched for agent review instead of silently deleting an occurrence.
            continue
        discarded.update(unit_ids)
        replacements.update(auxiliary_units)
        for unit in caption_units:
            if (unit.char_start, unit.char_end) in candidate_spans:
                continue
            replacement_id = (
                "host_measurement_caption_unit_"
                + sha256_bytes(f"{base}\0{unit.char_start}\0{unit.char_end}".encode())[:16]
            )
            replacements[replacement_id] = replace(
                unit,
                draft_id=replacement_id,
                logical_key="anchor:" + base + ".unit",
                value_kind="other_text",
                group_kind=owner.group_kind,
                group_key=owner.group_key,
                target_paths=(base + ".unit",),
                derivation=None,
                dependency_paths=(),
                dependency_bindings=(),
                evidence_origin="host_verified_agent_proposal",
                render_policy="natural_text",
                rationale=(
                    unit.rationale
                    + " Host retained this standalone unit through the immediately preceding "
                    "unambiguous measurement-field caption."
                ),
            )
        for start, end in candidates:
            replacement_id = (
                "host_measurement_unit_locality_"
                + sha256_bytes(f"{base}\0{start}\0{end}".encode())[:16]
            )
            replacements[replacement_id] = replace(
                owner,
                draft_id=replacement_id,
                logical_key="anchor:" + base + ".unit",
                char_start=start,
                char_end=end,
                source_text=raw[start:end],
                evidence_origin="host_verified_agent_proposal",
                rationale=(
                    owner.rationale
                    + " Host relocated this structured unit to the unique unit token "
                    "immediately following its same-measurement value owner."
                ),
            )
    return (
        *tuple(row for row in drafts if row.draft_id not in discarded),
        *replacements.values(),
    )


def _split_measurement_value_unit_overlaps(
    *, raw: str, drafts: Sequence[SpanDraft]
) -> tuple[SpanDraft, ...]:
    """Split a numeric value from its separately owned lexical measurement unit.

    The same physical mistake occurs for source-only measurements and deterministic totals: the
    value owner selects the complete ``value unit`` surface while a second owner selects the unit.
    A split is mechanical only when the value is the sole numeric prefix, the nested unit is a
    known measurement unit in the same semantic scope, and no alphanumeric suffix remains.
    """

    known_units = set().union(*_MEASUREMENT_UNIT_SURFACES.values())
    replacements: dict[str, SpanDraft] = {}
    for outer in drafts:
        source_only_value = (
            outer.render_mode in {"deterministic_auxiliary", "agent_residual"}
            and not outer.target_paths
            and not outer.dependency_paths
            and not outer.dependency_bindings
        )
        derived_value = (
            outer.render_mode == "deterministic_derived"
            and not outer.target_paths
            and bool(outer.dependency_paths or outer.dependency_bindings)
            and outer.derivation
            in {
                "sum_gross_weight",
                "sum_net_weight",
                "sum_tare_weight",
                "sum_volume",
                "sum_decimal_values",
                "gross_minus_net_weight",
            }
        )
        if not (
            (source_only_value or derived_value)
            and outer.value_kind in {"decimal_measurement", "integer", "package"}
        ):
            continue
        nested = tuple(
            inner
            for inner in drafts
            if inner.logical_key != outer.logical_key
            and inner.group_kind == outer.group_kind
            and inner.group_key == outer.group_key
            and inner.render_mode in {"deterministic_auxiliary", "literal_static"}
            and not inner.target_paths
            and not inner.dependency_paths
            and not inner.dependency_bindings
            and outer.char_start <= inner.char_start
            and inner.char_end <= outer.char_end
            and (outer.char_start, outer.char_end) != (inner.char_start, inner.char_end)
            and _normalized_surface(inner.source_text) in known_units
        )
        if len(nested) != 1:
            continue
        unit = nested[0]
        before = raw[outer.char_start : unit.char_start]
        after = raw[unit.char_end : outer.char_end]
        match = re.fullmatch(
            r"\s*(?P<value>[-+]?(?:[0-9][0-9., ]*|[.,][0-9]+))\s*",
            before,
        )
        if match is None or any(character.isalnum() for character in after):
            continue
        start = outer.char_start + match.start("value")
        end = outer.char_start + match.end("value")
        while end > start and raw[end - 1] in " ,.":
            end -= 1
        if (
            start == outer.char_start
            and start > 0
            and raw[start - 1] in ".,+-"
            and (start == 1 or not raw[start - 2].isalnum())
        ):
            start -= 1
        if start >= end:
            continue
        replacements[outer.draft_id] = replace(
            outer,
            value_kind="decimal_measurement",
            char_start=start,
            char_end=end,
            source_text=raw[start:end],
            evidence_origin="host_verified_agent_proposal",
            render_policy=("derived_surface" if derived_value else "numeric_surface"),
            rationale=(
                outer.rationale
                + " Host split this numeric measurement value from its separately owned lexical "
                "unit."
            ),
        )
    return tuple(replacements.get(row.draft_id, row) for row in drafts)


def _remove_residual_occurrences_exactly_owned_by_direct_targets(
    *, drafts: Sequence[SpanDraft], source_target: Mapping[str, Any]
) -> tuple[SpanDraft, ...]:
    """Remove an accidental residual occurrence with one unambiguous direct target owner.

    A bounded residual can legitimately own several status or legal surfaces. It cannot also own
    the exact bytes of a disjoint date, location, or other structured scalar. We remove such a row
    only when one direct target binding occupies the identical span, independently realizes all of
    its target values, and the residual retains at least one non-colliding physical occurrence.
    """

    by_span: dict[tuple[int, int], list[SpanDraft]] = defaultdict(list)
    by_key: dict[str, list[SpanDraft]] = defaultdict(list)
    for draft in drafts:
        by_span[(draft.char_start, draft.char_end)].append(draft)
        by_key[draft.logical_key].append(draft)

    removable: dict[str, SpanDraft] = {}
    for logical_key, rows in by_key.items():
        first = rows[0]
        if first.render_mode != "agent_residual" or not first.target_paths:
            continue
        for row in rows:
            direct_owners = tuple(
                candidate
                for candidate in by_span[(row.char_start, row.char_end)]
                if candidate.logical_key != logical_key
                and candidate.render_mode == "target_binding"
                and candidate.target_paths
                and set(candidate.target_paths).isdisjoint(first.target_paths)
                and all(
                    _surface_match(
                        source=candidate.source_text,
                        target=_resolve_target_path(source_target, path),
                        adapter=_surface_adapter(candidate),
                        value_kind=candidate.value_kind,
                    )
                    is not None
                    for path in candidate.target_paths
                )
            )
            if len({owner.logical_key for owner in direct_owners}) == 1:
                removable[row.draft_id] = direct_owners[0]

    discarded: set[str] = set()
    receipts: dict[str, list[str]] = defaultdict(list)
    for _logical_key, rows in by_key.items():
        candidates = tuple(row for row in rows if row.draft_id in removable)
        retained = tuple(row for row in rows if row.draft_id not in removable)
        if not candidates or not retained:
            continue
        discarded.update(row.draft_id for row in candidates)
        receipt_owner = min(retained, key=lambda row: (row.char_start, row.char_end, row.draft_id))
        receipts[receipt_owner.draft_id].extend(
            f"{removable[row.draft_id].logical_key} [{row.char_start},{row.char_end})"
            for row in candidates
        )

    return tuple(
        replace(
            draft,
            rationale=(
                draft.rationale
                + " Host removed accidental residual occurrence(s) exactly and independently "
                "owned by " + ", ".join(sorted(receipts[draft.draft_id])) + "."
            ),
        )
        if draft.draft_id in receipts
        else draft
        for draft in drafts
        if draft.draft_id not in discarded
    )


def _narrow_overlapping_numeric_measurement_values(
    *, raw: str, drafts: Sequence[SpanDraft], source_target: Mapping[str, Any]
) -> tuple[SpanDraft, ...]:
    """Narrow a numeric value that accidentally includes its separately owned unit.

    A model can select ``+1,0 C`` for the numeric value and the nested ``C`` for the unit.  The
    structured value proves the unique numeric subspan, while the same structured measurement
    base proves that the nested unit is not unrelated text.  No correction is made without both
    facts, so ambiguous or non-measurement overlaps continue to fail closed.
    """

    units_by_base: dict[str, list[SpanDraft]] = defaultdict(list)
    for draft in drafts:
        if draft.render_mode != "target_binding" or len(draft.target_paths) != 1:
            continue
        match = _MEASUREMENT_COMPONENT_PATH.fullmatch(draft.target_paths[0])
        if match is not None and match.group("field") == "unit":
            units_by_base[match.group("base")].append(draft)

    replacements: dict[str, SpanDraft] = {}
    for draft in drafts:
        if draft.render_mode != "target_binding" or len(draft.target_paths) != 1:
            continue
        match = _MEASUREMENT_COMPONENT_PATH.fullmatch(draft.target_paths[0])
        if match is None or match.group("field") != "value":
            continue
        base = match.group("base")
        unit_path = base + ".unit"
        try:
            target_unit = _resolve_target_path(source_target, unit_path)
        except ValueError:
            continue
        nested_units = tuple(
            unit
            for unit in units_by_base.get(base, ())
            if draft.char_start <= unit.char_start
            and unit.char_end <= draft.char_end
            and (draft.char_start, draft.char_end) != (unit.char_start, unit.char_end)
        )
        if not nested_units:
            nested_units = tuple(
                unit
                for unit in drafts
                if unit.render_mode in {"deterministic_auxiliary", "literal_static"}
                and not unit.target_paths
                and not unit.dependency_paths
                and not unit.dependency_bindings
                and draft.char_start <= unit.char_start
                and unit.char_end <= draft.char_end
                and (draft.char_start, draft.char_end) != (unit.char_start, unit.char_end)
                and _surface_match(
                    source=unit.source_text,
                    target=target_unit,
                    adapter="measurement_unit",
                    value_kind="other_text",
                )
                is not None
            )
        if len(nested_units) != 1:
            continue
        target_value = _resolve_target_path(source_target, draft.target_paths[0])
        numeric_span = _matching_numeric_span(
            draft.source_text,
            target_value,
            integer=draft.value_kind == "integer",
        )
        if numeric_span is None:
            lexical_matches = tuple(
                (numeric.start(), numeric.start() + len(numeric.group(0).rstrip()))
                for numeric in re.finditer(
                    r"(?<![0-9])[-+]?[0-9][0-9., ]*(?![0-9])", draft.source_text
                )
                if numeric.group(0).rstrip()
            )
            numeric_span = lexical_matches[0] if len(lexical_matches) == 1 else None
        if numeric_span is None:
            continue
        char_start = draft.char_start + numeric_span[0]
        char_end = draft.char_start + numeric_span[1]
        unit = nested_units[0]
        if char_start < unit.char_end and unit.char_start < char_end:
            continue
        replacements[draft.draft_id] = replace(
            draft,
            char_start=char_start,
            char_end=char_end,
            source_text=raw[char_start:char_end],
            evidence_origin="host_verified_agent_proposal",
            rationale=(
                draft.rationale
                + " Host narrowed the numeric measurement component around its separately "
                "owned structured unit."
            ),
        )
        if unit.render_mode == "target_binding":
            continue
        group_kind, group_key = _canonical_target_group((unit_path,))
        value_kind = _value_kind((unit_path,), "natural_text")
        replacements[unit.draft_id] = replace(
            unit,
            logical_key="anchor:" + unit_path,
            render_mode="target_binding",
            value_kind=value_kind,
            group_kind=group_kind,
            group_key=group_key,
            target_paths=(unit_path,),
            evidence_origin="host_verified_agent_proposal",
            render_policy=_render_policy_for_value_kind(value_kind, "target_binding"),
            rationale=(
                unit.rationale
                + " Host retargeted this nested unit after proving it is the structured unit "
                "paired with the containing numeric measurement."
            ),
        )
    return tuple(replacements.get(draft.draft_id, draft) for draft in drafts)


def _narrow_direct_target_literal_frames(
    *, raw: str, drafts: Sequence[SpanDraft], source_target: Mapping[str, Any]
) -> tuple[SpanDraft, ...]:
    """Move a mechanically separable caption or unit frame outside a direct scalar slot."""

    output: list[SpanDraft] = []
    for draft in drafts:
        if draft.render_mode != "target_binding" or len(draft.target_paths) != 1:
            output.append(draft)
            continue
        try:
            target_value = _resolve_target_path(source_target, draft.target_paths[0])
        except ValueError:
            output.append(draft)
            continue
        match = _surface_match(
            source=draft.source_text,
            target=target_value,
            adapter=_surface_adapter(draft),
            value_kind=draft.value_kind,
        )
        if match is None or match == ("", ""):
            output.append(draft)
            continue
        start = draft.char_start + len(match[0])
        end = draft.char_end - len(match[1])
        if start >= end or not any(character.isalnum() for character in raw[start:end]):
            output.append(draft)
            continue
        output.append(
            replace(
                draft,
                char_start=start,
                char_end=end,
                source_text=raw[start:end],
                evidence_origin="host_verified_agent_proposal",
                rationale=(
                    draft.rationale
                    + " Host moved the mechanically separable literal frame outside the direct "
                    "target slot."
                ),
            )
        )
    return tuple(output)


_LEADING_FIELD_CAPTION = re.compile(
    r"^(?P<prefix>[A-Za-z][A-Za-z0-9 /().&'-]{1,31}:\s*)(?P<value>\S)"
)
_URI_SCHEME_PREFIX = re.compile(r"^[A-Za-z][A-Za-z0-9+.-]*://")


def _narrow_residual_leading_field_captions(
    *, raw: str, drafts: Sequence[SpanDraft]
) -> tuple[SpanDraft, ...]:
    """Keep a closed leading field label literal around a residual value."""

    output: list[SpanDraft] = []
    for draft in drafts:
        match = (
            _LEADING_FIELD_CAPTION.match(draft.source_text)
            if draft.render_mode == "agent_residual"
            and _URI_SCHEME_PREFIX.match(draft.source_text) is None
            else None
        )
        if match is None:
            output.append(draft)
            continue
        start = draft.char_start + match.start("value")
        output.append(
            replace(
                draft,
                char_start=start,
                source_text=raw[start : draft.char_end],
                evidence_origin="host_verified_agent_proposal",
                rationale=(
                    draft.rationale
                    + " Host moved the closed leading field caption outside the residual value "
                    "slot."
                ),
            )
        )
    return tuple(output)


def _trim_residual_trailing_separators(
    *, raw: str, drafts: Sequence[SpanDraft]
) -> tuple[SpanDraft, ...]:
    """Leave terminal field-separator punctuation outside a residual value slot."""

    output: list[SpanDraft] = []
    for draft in drafts:
        trimmed = (
            re.sub(r"[ \t]*[,;:|]+$", "", draft.source_text)
            if draft.render_mode == "agent_residual"
            else draft.source_text
        )
        if not trimmed or trimmed == draft.source_text:
            output.append(draft)
            continue
        end = draft.char_start + len(trimmed)
        output.append(
            replace(
                draft,
                char_end=end,
                source_text=raw[draft.char_start : end],
                evidence_origin="host_verified_agent_proposal",
                rationale=(
                    draft.rationale
                    + " Host left terminal field-separator punctuation outside the residual "
                    "value slot."
                ),
            )
        )
    return tuple(output)


def _remove_redundant_identifier_containment(
    drafts: Sequence[SpanDraft],
) -> tuple[SpanDraft, ...]:
    """Remove only a nested auxiliary identifier occurrence already owned by a wider identifier.

    The shorter logical identifier must retain another physical occurrence, and exactly one wider
    source-only identifier must cover the nested bytes.  This preserves both logical values and
    leaves ambiguous containment untouched for ordinary overlap rejection.
    """

    grouped: dict[str, list[SpanDraft]] = defaultdict(list)
    for draft in drafts:
        grouped[draft.logical_key].append(draft)
    discarded_ids: set[str] = set()
    absorption_receipts: dict[str, list[str]] = defaultdict(list)
    for logical_key, rows in grouped.items():
        first = rows[0]
        if (
            first.render_mode != "deterministic_auxiliary"
            or first.value_kind != "identifier"
            or first.target_paths
            or len(rows) < 2
        ):
            continue
        for row in rows:
            outers = tuple(
                other
                for other_key, other_rows in grouped.items()
                for other in other_rows
                if other_key != logical_key
                and other.render_mode == "deterministic_auxiliary"
                and other.value_kind == "identifier"
                and not other.target_paths
                and other.char_start <= row.char_start
                and row.char_end <= other.char_end
                and (other.char_start, other.char_end) != (row.char_start, row.char_end)
                and _normalized_surface(row.source_text) in _normalized_surface(other.source_text)
            )
            if len(outers) != 1:
                continue
            if not any(
                sibling.draft_id != row.draft_id
                and not (
                    outers[0].char_start <= sibling.char_start
                    and sibling.char_end <= outers[0].char_end
                )
                for sibling in rows
            ):
                continue
            discarded_ids.add(row.draft_id)
            absorption_receipts[outers[0].draft_id].append(
                f"{row.logical_key} [{row.char_start},{row.char_end})"
            )
    return tuple(
        replace(
            draft,
            rationale=(
                draft.rationale
                + " Host removed redundant nested auxiliary identifier occurrence(s) "
                + ", ".join(sorted(absorption_receipts[draft.draft_id]))
                + "; each shorter logical identifier retains a separate physical occurrence."
            ),
        )
        if draft.draft_id in absorption_receipts
        else draft
        for draft in drafts
        if draft.draft_id not in discarded_ids
    )


def _remove_unbounded_auxiliary_fragments(
    drafts: Sequence[SpanDraft],
) -> tuple[SpanDraft, ...]:
    """Reject sub-token auxiliary fragments in favor of their complete mutable owner.

    A substring such as the first ``1`` inside postal code ``01122`` is not an independently
    renderable source value. Neither is a single alphanumeric token nested inside a complete
    mutable phone/identifier surface, even when OCR whitespace makes it token-bounded. When
    exactly one non-auxiliary mutable binding owns the complete surface, discard only that
    malformed proposal occurrence. Every target-backed proposal remains outside this rule.
    """

    discarded_ids: set[str] = set()
    rejection_receipts: dict[str, list[str]] = defaultdict(list)
    for inner in drafts:
        if inner.render_mode != "deterministic_auxiliary" or inner.target_paths:
            continue
        containing = tuple(
            outer
            for outer in drafts
            if outer.logical_key != inner.logical_key
            and outer.render_mode in {"target_binding", "deterministic_derived", "agent_residual"}
            and outer.char_start <= inner.char_start
            and inner.char_end <= outer.char_end
            and (outer.char_start, outer.char_end) != (inner.char_start, inner.char_end)
        )
        if len(containing) != 1:
            continue
        outer = containing[0]
        relative_start = inner.char_start - outer.char_start
        relative_end = inner.char_end - outer.char_start
        left_is_token = relative_start > 0 and outer.source_text[relative_start - 1].isalnum()
        right_is_token = (
            relative_end < len(outer.source_text) and outer.source_text[relative_end].isalnum()
        )
        if left_is_token or right_is_token or len(_normalized_surface(inner.source_text)) <= 1:
            discarded_ids.add(inner.draft_id)
            rejection_receipts[outer.draft_id].append(
                f"{inner.logical_key} [{inner.char_start},{inner.char_end})"
            )
    return tuple(
        replace(
            draft,
            rationale=(
                draft.rationale
                + " Host rejected malformed sub-token auxiliary proposal occurrence(s) "
                + ", ".join(sorted(rejection_receipts[draft.draft_id]))
                + "; this complete mutable binding owns the containing token."
            ),
        )
        if draft.draft_id in rejection_receipts
        else draft
        for draft in drafts
        if draft.draft_id not in discarded_ids
    )


def validate_mutable_token_boundaries(*, raw: str, drafts: Sequence[SpanDraft]) -> None:
    """Reject a mutable slot carved from inside another owned alphanumeric token."""

    violations: list[str] = []
    for inner in drafts:
        if inner.render_mode not in {"target_binding", "agent_residual"}:
            continue
        outers = tuple(
            outer
            for outer in drafts
            if outer.logical_key != inner.logical_key
            and outer.render_mode in {"target_binding", "carrier_static", "agent_residual"}
            and outer.char_start <= inner.char_start
            and inner.char_end <= outer.char_end
            and (outer.char_start, outer.char_end) != (inner.char_start, inner.char_end)
        )
        if not outers:
            continue
        left_is_token = inner.char_start > 0 and raw[inner.char_start - 1].isalnum()
        right_is_token = inner.char_end < len(raw) and raw[inner.char_end].isalnum()
        if left_is_token or right_is_token:
            violations.append(
                f"{inner.logical_key} [{inner.char_start},{inner.char_end})={inner.source_text!r} "
                "is an alphanumeric substring of "
                + ", ".join(
                    f"{outer.logical_key} [{outer.char_start},{outer.char_end})" for outer in outers
                )
            )
    if violations:
        raise ValueError(
            "mutable bindings cannot select a substring inside another owned alphanumeric "
            "token: " + "; ".join(violations)
        )


def normalize_competing_auxiliary_outliers(
    drafts: Sequence[SpanDraft],
) -> tuple[SpanDraft, ...]:
    """Remove a non-modal auxiliary occurrence already owned by one complete binding.

    One deterministic auxiliary key renders one generated value into all of its slots.  When a
    unique repeated surface proves that value, a different surface cannot belong to that key. If
    exactly one other mutable binding already owns the complete outlier span, retaining both is
    neither necessary nor renderable. Ambiguous, uncovered, and tied surface clusters remain
    untouched and therefore continue to fail closed.
    """

    grouped: dict[str, list[SpanDraft]] = defaultdict(list)
    for draft in drafts:
        grouped[draft.logical_key].append(draft)
    discarded_ids: set[str] = set()
    ownership_receipts: dict[str, list[str]] = defaultdict(list)
    for logical_key, rows in grouped.items():
        first = rows[0]
        if first.render_mode != "deterministic_auxiliary" or first.target_paths or len(rows) < 3:
            continue
        surface_counts = Counter(_normalized_surface(row.source_text) for row in rows)
        ranked = surface_counts.most_common()
        if (
            not ranked
            or not ranked[0][0]
            or ranked[0][1] < 2
            or (len(ranked) > 1 and ranked[0][1] == ranked[1][1])
        ):
            continue
        dominant_surface = ranked[0][0]
        for row in rows:
            if _normalized_surface(row.source_text) == dominant_surface:
                continue
            owners = tuple(
                other
                for other in drafts
                if other.logical_key != logical_key
                and other.render_mode
                in {
                    "target_binding",
                    "deterministic_derived",
                    "deterministic_auxiliary",
                    "agent_residual",
                }
                and other.char_start <= row.char_start
                and row.char_end <= other.char_end
            )
            if len(owners) != 1:
                continue
            owner = owners[0]
            discarded_ids.add(row.draft_id)
            ownership_receipts[owner.draft_id].append(
                f"{logical_key} [{row.char_start},{row.char_end})"
            )
    return tuple(
        replace(
            draft,
            rationale=(
                draft.rationale
                + " Host discarded non-modal auxiliary duplicate occurrence(s) "
                + ", ".join(sorted(ownership_receipts[draft.draft_id]))
                + "; this complete mutable binding retains the source bytes."
            ),
        )
        if draft.draft_id in ownership_receipts
        else draft
        for draft in drafts
        if draft.draft_id not in discarded_ids
    )


def _remove_redundant_multitarget_occurrences(
    drafts: Sequence[SpanDraft],
) -> tuple[SpanDraft, ...]:
    """Discard an overlapping equality owner after every component has a singleton owner.

    Equal current values do not justify retaining two render contracts over the same bytes.  The
    removal is safe only when every path in the multi-target occurrence already has its own direct
    singleton owner and one of those singleton owners occupies the exact same span.  This preserves
    all structured ownership while removing only the redundant, non-renderable equality claim.
    """

    singleton_paths = {
        draft.target_paths[0]
        for draft in drafts
        if draft.render_mode == "target_binding" and len(draft.target_paths) == 1
    }
    discarded_ids: set[str] = set()
    receipts: dict[str, list[str]] = defaultdict(list)
    for draft in drafts:
        if (
            draft.render_mode != "target_binding"
            or len(draft.target_paths) < 2
            or not set(draft.target_paths) <= singleton_paths
        ):
            continue
        exact_singletons = tuple(
            other
            for other in drafts
            if other.draft_id != draft.draft_id
            and other.render_mode == "target_binding"
            and len(other.target_paths) == 1
            and other.target_paths[0] in draft.target_paths
            and (other.char_start, other.char_end) == (draft.char_start, draft.char_end)
        )
        if len(exact_singletons) != 1:
            continue
        owner = exact_singletons[0]
        discarded_ids.add(draft.draft_id)
        receipts[owner.draft_id].append(
            f"{draft.logical_key} [{draft.char_start},{draft.char_end})"
        )
    return tuple(
        replace(
            draft,
            rationale=(
                draft.rationale
                + " Host discarded redundant multi-target occurrence(s) "
                + ", ".join(sorted(receipts[draft.draft_id]))
                + "; every equality component retains an independent singleton owner."
            ),
        )
        if draft.draft_id in receipts
        else draft
        for draft in drafts
        if draft.draft_id not in discarded_ids
    )


def _remove_auxiliaries_contained_by_same_scope_vessel_targets(
    drafts: Sequence[SpanDraft],
) -> tuple[SpanDraft, ...]:
    """Discard a source-only vessel token wholly contained by its vessel-name binding.

    This handles a compiler proposing a repeated suffix (for example ``EXPRESS``) as an auxiliary
    even though every occurrence is already a whole-token subset of the repeated target surface
    (for example ``LONDON EXPRESS``).  Scope equality, strict containment, complete occurrence
    coverage, the exact vessel target, and a single owner are all required; other target kinds and
    ambiguous or partial groups fail closed.
    """

    grouped: dict[str, list[SpanDraft]] = defaultdict(list)
    for draft in drafts:
        grouped[draft.logical_key].append(draft)
    discarded_keys: set[str] = set()
    receipts: dict[str, list[str]] = defaultdict(list)
    for auxiliary_key, auxiliary_rows in grouped.items():
        auxiliary = auxiliary_rows[0]
        if not (
            auxiliary.render_mode == "deterministic_auxiliary"
            and not auxiliary.target_paths
            and not auxiliary.dependency_paths
            and not auxiliary.dependency_bindings
        ):
            continue
        target_owners_by_row: list[tuple[SpanDraft, ...]] = []
        for row in auxiliary_rows:
            owners = tuple(
                candidate
                for candidate in drafts
                if candidate.logical_key != auxiliary_key
                and candidate.render_mode == "target_binding"
                and candidate.target_paths == (_VESSEL_NAME_PATH,)
                and candidate.group_kind == row.group_kind
                and candidate.group_key == row.group_key
                and candidate.char_start <= row.char_start
                and row.char_end <= candidate.char_end
                and (candidate.char_start, candidate.char_end) != (row.char_start, row.char_end)
                and _matching_token_projection(row.source_text, candidate.source_text) is not None
            )
            if len(owners) != 1:
                target_owners_by_row = []
                break
            target_owners_by_row.append(owners)
        if not target_owners_by_row:
            continue
        owner_keys = {owner.logical_key for owners in target_owners_by_row for owner in owners}
        if len(owner_keys) != 1:
            continue
        discarded_keys.add(auxiliary_key)
        for row, owners in zip(auxiliary_rows, target_owners_by_row, strict=True):
            receipts[owners[0].draft_id].append(
                f"{auxiliary_key} [{row.char_start},{row.char_end})"
            )
    return tuple(
        replace(
            draft,
            rationale=(
                draft.rationale
                + " Host discarded same-scope auxiliary token projection(s) "
                + ", ".join(sorted(receipts[draft.draft_id]))
                + "; this complete target owner retains every source byte."
            ),
        )
        if draft.draft_id in receipts
        else draft
        for draft in drafts
        if draft.logical_key not in discarded_keys
    )


def _remove_projected_target_collisions(
    *, drafts: Sequence[SpanDraft], source_target: Mapping[str, Any]
) -> tuple[SpanDraft, ...]:
    """Let one exact target realization beat a colliding token projection.

    This is safe only when there is exactly one exact logical owner and every displaced projected
    owner retains another physical occurrence.  The projected target fact therefore remains
    represented while the shared byte span has one unambiguous owner.
    """

    by_span: dict[tuple[int, int], list[SpanDraft]] = defaultdict(list)
    by_key: dict[str, list[SpanDraft]] = defaultdict(list)
    for draft in drafts:
        by_span[(draft.char_start, draft.char_end)].append(draft)
        by_key[draft.logical_key].append(draft)
    discarded: set[str] = set()
    for rows in by_span.values():
        logical_keys = {row.logical_key for row in rows}
        if len(logical_keys) < 2 or any(
            row.render_mode != "target_binding" or not row.target_paths for row in rows
        ):
            continue
        exact_keys: set[str] = set()
        projected_keys: set[str] = set()
        for row in rows:
            surfaces = tuple(
                _scalar_surface(_resolve_target_path(source_target, path))
                for path in row.target_paths
            )
            if any(surface is None for surface in surfaces):
                break
            typed_surfaces = tuple(surface for surface in surfaces if surface is not None)
            if typed_surfaces and all(
                _normalized_surface(row.source_text) == _normalized_surface(surface)
                for surface in typed_surfaces
            ):
                exact_keys.add(row.logical_key)
            elif typed_surfaces and all(
                _matching_token_projection(row.source_text, surface) is not None
                for surface in typed_surfaces
            ):
                projected_keys.add(row.logical_key)
            else:
                break
        else:
            if (
                len(exact_keys) == 1
                and projected_keys == logical_keys - exact_keys
                and all(
                    len(by_key[key]) > sum(row.logical_key == key for row in rows)
                    for key in projected_keys
                )
            ):
                discarded.update(row.draft_id for row in rows if row.logical_key in projected_keys)
    return tuple(row for row in drafts if row.draft_id not in discarded)


def _remove_caption_disambiguated_auxiliary_collisions(
    *, raw: str, drafts: Sequence[SpanDraft]
) -> tuple[SpanDraft, ...]:
    """Resolve an exact source-only identifier collision from local structural scope.

    A globally unique identifier can be safely localized even when a model misstated its source
    line, but two proposals can then claim that same span under different entity groups.  Prefer a
    unique line-caption match first.  When captions such as ``TAX ID`` name the value but not the
    entity, use the blank-line-delimited block's already target-backed group as the discriminator.
    Both mechanisms remain fail-closed on ties or absent evidence.
    """

    lines = line_spans(raw)
    block_by_line: dict[int, int] = {}
    block_index = -1
    inside_block = False
    for line in lines:
        if line.text.strip():
            if not inside_block:
                block_index += 1
                inside_block = True
            block_by_line[line.number] = block_index
        else:
            inside_block = False
    by_span: dict[tuple[int, int], list[SpanDraft]] = defaultdict(list)
    for draft in drafts:
        by_span[(draft.char_start, draft.char_end)].append(draft)
    discarded: set[str] = set()
    for (start, end), rows in by_span.items():
        if len({row.logical_key for row in rows}) < 2 or any(
            row.render_mode != "deterministic_auxiliary"
            or row.value_kind != "identifier"
            or row.target_paths
            or row.dependency_paths
            or row.dependency_bindings
            for row in rows
        ):
            continue
        line_number = line_number_for_char(lines, start)
        line = lines[line_number - 1]
        if end > line.char_end:
            continue
        caption = raw[line.char_start : start]
        caption_tokens = frozenset(
            token.casefold() for token in re.findall(r"[A-Za-z]{3,}", caption)
        )
        if not caption_tokens:
            continue
        scores = {
            row.logical_key: len(
                caption_tokens.intersection(
                    token.casefold() for token in re.findall(r"[A-Za-z]{3,}", row.logical_key)
                )
            )
            for row in rows
        }
        best_score = max(scores.values())
        winners = {key for key, score in scores.items() if score == best_score and score > 0}
        if len(winners) != 1:
            collision_block = block_by_line.get(line_number)
            if collision_block is None:
                continue
            group_evidence = {
                row.group_key: sum(
                    candidate.logical_key not in {item.logical_key for item in rows}
                    and candidate.render_mode in {"target_binding", "carrier_static"}
                    and bool(candidate.target_paths)
                    and candidate.group_kind == row.group_kind
                    and candidate.group_key == row.group_key
                    and block_by_line.get(line_number_for_char(lines, candidate.char_start))
                    == collision_block
                    for candidate in drafts
                )
                for row in rows
            }
            best_group_score = max(group_evidence.values())
            winning_groups = {
                group_key
                for group_key, score in group_evidence.items()
                if score == best_group_score and score > 0
            }
            if len(winning_groups) != 1:
                continue
            winning_group = next(iter(winning_groups))
            winners = {row.logical_key for row in rows if row.group_key == winning_group}
            if len(winners) != 1:
                continue
        winner = next(iter(winners))
        discarded.update(row.draft_id for row in rows if row.logical_key != winner)
    return tuple(row for row in drafts if row.draft_id not in discarded)


def _split_source_only_bindings_around_target_owners(
    *, raw: str, drafts: Sequence[SpanDraft]
) -> tuple[SpanDraft, ...]:
    """Partition one source-only value segment around complete nested target ownership.

    The source-only owner may be deterministic, residual, or static; its mode does not alter the
    physical impossibility of overlapping a complete structured target. Only natural-language
    surfaces may be partitioned: opaque identifiers, dates, and numeric/measurement surfaces are
    atomic contracts whose partial remainder has no proven meaning. Deterministic natural text is
    split only when removal of all nested targets leaves exactly one alphanumeric segment;
    residual natural text can retain multiple explicitly agent-rendered segments.
    """

    output: list[SpanDraft] = []
    for outer in drafts:
        if (
            outer.render_mode
            not in {"deterministic_auxiliary", "agent_residual", "carrier_static", "literal_static"}
            or outer.evidence_origin == "accepted_label_evidence"
            or outer.value_kind
            not in {
                "address",
                "cargo_text",
                "dangerous_goods",
                "commercial_text",
                "legal_text",
                "operational_text",
                "other_text",
            }
            or outer.target_paths
            or outer.dependency_paths
            or outer.dependency_bindings
        ):
            output.append(outer)
            continue
        nested = tuple(
            sorted(
                (
                    inner
                    for inner in drafts
                    if inner.logical_key != outer.logical_key
                    and inner.render_mode in {"target_binding", "carrier_static"}
                    and inner.target_paths
                    and outer.char_start <= inner.char_start
                    and inner.char_end <= outer.char_end
                    and (outer.char_start, outer.char_end) != (inner.char_start, inner.char_end)
                ),
                key=lambda row: (row.char_start, row.char_end),
            )
        )
        if not nested or any(left.char_end > right.char_start for left, right in pairwise(nested)):
            output.append(outer)
            continue
        boundaries = (
            outer.char_start,
            *(value for row in nested for value in (row.char_start, row.char_end)),
            outer.char_end,
        )
        segments: list[tuple[int, int]] = []
        for index in range(0, len(boundaries) - 1, 2):
            start, end = boundaries[index], boundaries[index + 1]
            while start < end and raw[start].isspace():
                start += 1
            while end > start and raw[end - 1].isspace():
                end -= 1
            if start < end and any(character.isalnum() for character in raw[start:end]):
                segments.append((start, end))
        if outer.render_mode == "deterministic_auxiliary" and len(segments) != 1:
            output.append(outer)
            continue
        for start, end in segments:
            output.append(
                replace(
                    outer,
                    draft_id=(
                        "partitioned_source_residual_"
                        + sha256_bytes(f"{outer.draft_id}\0{start}\0{end}".encode())[:16]
                    ),
                    char_start=start,
                    char_end=end,
                    source_text=raw[start:end],
                    rationale=(
                        outer.rationale + " Host partitioned this source-only binding around "
                        "complete nested structured target ownership after proving exactly one "
                        "independently renderable remainder."
                    ),
                )
            )
    return tuple(output)


def _relocate_uniquely_nonoverlapping_numeric_tokens(
    *, raw: str, drafts: Sequence[SpanDraft]
) -> tuple[SpanDraft, ...]:
    """Relocate a numeric token only when overlap elimination proves one local source span.

    Models count raw substring matches poorly when a one-digit segmented value is also embedded
    inside an adjacent larger number. We retain fail-closed behavior unless exactly one
    token-bounded occurrence on the same source line is disjoint from every other binding.
    """

    current = list(drafts)
    lines = line_spans(raw)
    by_number = {line.number: line for line in lines}
    while True:
        ordered = sorted(current, key=lambda row: (row.char_start, row.char_end))
        conflicts: set[str] = set()
        for left_index, left in enumerate(ordered):
            for right in ordered[left_index + 1 :]:
                if right.char_start >= left.char_end:
                    break
                if left.logical_key != right.logical_key:
                    conflicts.update((left.draft_id, right.draft_id))
        if not conflicts:
            return tuple(current)
        relocations: list[tuple[str, int, int]] = []
        for draft in current:
            if (
                draft.draft_id not in conflicts
                or draft.evidence_origin == "accepted_label_evidence"
                or draft.render_mode not in {"target_binding", "deterministic_derived"}
                or draft.value_kind not in {"integer", "decimal_measurement", "temperature"}
                or not draft.source_text.isdigit()
            ):
                continue
            line_number = line_number_for_char(lines, draft.char_start)
            line = by_number[line_number]
            if draft.char_end > line.char_end:
                continue
            candidates: list[tuple[int, int]] = []
            for relative in _exact_offsets(line.text, draft.source_text):
                start = line.char_start + relative
                end = start + len(draft.source_text)
                if (start, end) == (draft.char_start, draft.char_end):
                    continue
                if not is_token_bounded_surface_span(raw, start, end):
                    continue
                if any(
                    other.draft_id != draft.draft_id
                    and start < other.char_end
                    and other.char_start < end
                    for other in current
                ):
                    continue
                candidates.append((start, end))
            if len(candidates) == 1:
                relocations.append((draft.draft_id, *candidates[0]))
        if len(relocations) != 1:
            return tuple(current)
        draft_id, start, end = relocations[0]
        current = [
            replace(
                draft,
                draft_id=(
                    "relocated_numeric_token_"
                    + sha256_bytes(f"{draft.draft_id}\0{start}\0{end}".encode())[:16]
                ),
                char_start=start,
                char_end=end,
                source_text=raw[start:end],
                rationale=(
                    draft.rationale + " Host relocated the ambiguous numeric substring to the only "
                    "token-bounded occurrence on the same line that does not overlap another "
                    "binding."
                ),
            )
            if draft.draft_id == draft_id
            else draft
            for draft in current
        ]


def _remove_redundant_contained_target_occurrences(
    *, drafts: Sequence[SpanDraft], source_target: Mapping[str, Any]
) -> tuple[SpanDraft, ...]:
    """Let one complete target surface own a nested repeat of another retained target.

    This applies only when both rows are direct single-target bindings, the containing source
    surface is an exact punctuation-insensitive realization of its own structured scalar, and the
    nested logical target retains a separate physical occurrence.  It therefore removes no target
    ownership and never guesses how to split a composite surface.
    """

    grouped: dict[str, list[SpanDraft]] = defaultdict(list)
    for draft in drafts:
        grouped[draft.logical_key].append(draft)
    discarded: set[str] = set()
    receipts: dict[str, list[str]] = defaultdict(list)
    for logical_key, rows in grouped.items():
        if len(rows) < 2:
            continue
        for inner in rows:
            if inner.render_mode != "target_binding" or len(inner.target_paths) != 1:
                continue
            inner_value = _resolve_target_path(source_target, inner.target_paths[0])
            inner_surface = (
                _package_category_surface(inner_value)
                if inner.target_paths[0].endswith(".typeCategory")
                else _scalar_surface(inner_value)
            )
            if inner_surface is None or _normalized_surface(
                inner.source_text
            ) != _normalized_surface(inner_surface):
                continue
            outers: list[SpanDraft] = []
            for outer in drafts:
                if (
                    outer.logical_key == logical_key
                    or outer.render_mode != "target_binding"
                    or len(outer.target_paths) != 1
                    or not (
                        outer.char_start <= inner.char_start
                        and inner.char_end <= outer.char_end
                        and (outer.char_start, outer.char_end) != (inner.char_start, inner.char_end)
                    )
                ):
                    continue
                outer_value = _resolve_target_path(source_target, outer.target_paths[0])
                outer_surface = (
                    _package_category_surface(outer_value)
                    if outer.target_paths[0].endswith(".typeCategory")
                    else _scalar_surface(outer_value)
                )
                if outer_surface is not None and _normalized_surface(
                    outer.source_text
                ) == _normalized_surface(outer_surface):
                    outers.append(outer)
            if len(outers) != 1:
                continue
            outer = outers[0]
            if not any(
                sibling.draft_id != inner.draft_id
                and not (
                    outer.char_start <= sibling.char_start and sibling.char_end <= outer.char_end
                )
                for sibling in rows
            ):
                continue
            discarded.add(inner.draft_id)
            receipts[outer.draft_id].append(
                f"{inner.logical_key} [{inner.char_start},{inner.char_end})"
            )
    return tuple(
        replace(
            draft,
            rationale=(
                draft.rationale
                + " Host discarded redundant nested target occurrence(s) "
                + ", ".join(sorted(receipts[draft.draft_id]))
                + "; each nested logical target retains a separate physical occurrence."
            ),
        )
        if draft.draft_id in receipts
        else draft
        for draft in drafts
        if draft.draft_id not in discarded
    )


def reconcile_draft_overlaps(
    *, raw: str, drafts: Sequence[SpanDraft], source_target: Mapping[str, Any]
) -> tuple[SpanDraft, ...]:
    """Canonicalize only overlaps whose ownership is mechanically provable.

    Exact duplicate semantics are handled by ``merge_drafts``. An exact target occurrence can be
    displaced only when it is provably unrelated to its target scalar and that logical target
    still has another occurrence. Strict containment can partition only natural target text at
    token boundaries, and every retained segment must independently project into one target value.
    All other overlaps remain hard errors.
    """

    normalized = _normalize_misplaced_package_count_bindings(
        raw=raw,
        drafts=drafts,
        source_target=source_target,
    )
    normalized = _split_inline_structured_component_bindings(
        drafts=normalized,
        source_target=source_target,
    )
    normalized = _partition_structured_component_bindings(
        drafts=normalized,
        source_target=source_target,
    )
    normalized = _partition_compiler_multitarget_projections(
        drafts=normalized,
        source_target=source_target,
    )
    normalized = _remove_projected_target_collisions(
        drafts=normalized,
        source_target=source_target,
    )
    normalized = _remove_caption_disambiguated_auxiliary_collisions(
        raw=raw,
        drafts=normalized,
    )
    normalized = _split_source_only_bindings_around_target_owners(
        raw=raw,
        drafts=normalized,
    )
    normalized = _relocate_uniquely_nonoverlapping_numeric_tokens(
        raw=raw,
        drafts=normalized,
    )
    normalized = _remove_redundant_contained_target_occurrences(
        drafts=normalized,
        source_target=source_target,
    )
    normalized = _normalize_hazard_class_binding_contracts(raw=raw, drafts=normalized)
    normalized = _normalize_measurement_unit_component_locality(
        raw=raw,
        drafts=normalized,
        source_target=source_target,
    )
    normalized = _split_measurement_value_unit_overlaps(
        raw=raw,
        drafts=normalized,
    )
    normalized = _remove_residual_occurrences_exactly_owned_by_direct_targets(
        drafts=normalized,
        source_target=source_target,
    )
    normalized = _promote_non_equivalent_auxiliary_groups(drafts=normalized)
    normalized = _narrow_direct_target_literal_frames(
        raw=raw,
        drafts=normalized,
        source_target=source_target,
    )
    normalized = _narrow_residual_leading_field_captions(raw=raw, drafts=normalized)
    normalized = _trim_residual_trailing_separators(raw=raw, drafts=normalized)
    normalized = _split_exact_isolated_residual_occurrences(
        drafts=normalized,
        source_target=source_target,
    )
    normalized = _narrow_overlapping_numeric_measurement_values(
        raw=raw,
        drafts=normalized,
        source_target=source_target,
    )
    normalized = normalize_competing_auxiliary_outliers(normalized)
    normalized = _remove_redundant_multitarget_occurrences(normalized)
    normalized = _remove_auxiliaries_contained_by_same_scope_vessel_targets(normalized)
    normalized = _remove_redundant_identifier_containment(normalized)
    normalized = _remove_unbounded_auxiliary_fragments(normalized)
    normalized = _normalize_carrier_static_target_ownership(
        drafts=normalized,
        source_target=source_target,
    )
    normalized = _normalize_fmc_carrier_duplicate_ownership(raw=raw, drafts=normalized)
    normalized = _normalize_required_cobinding_dominance(
        drafts=normalized,
        source_target=source_target,
    )
    normalized = _trim_residual_edges_owned_by_exact_targets(raw=raw, drafts=normalized)
    normalized = _trim_targeted_residual_trailing_footnote_markers(
        raw=raw,
        drafts=normalized,
    )
    normalized = _trim_targetless_residual_edges_owned_by_auxiliaries(
        raw=raw,
        drafts=normalized,
    )
    normalized = _expand_equipment_receipt_dependency_overlaps(raw=raw, drafts=normalized)
    normalized = _normalize_derived_dependency_containment(normalized)
    normalized = _merge_adjacent_agent_residual_segments(raw=raw, drafts=normalized)
    current = list(normalized)
    retained_anchor_ids = {
        draft.draft_id for draft in current if draft.evidence_origin == "accepted_label_evidence"
    }
    redundant_proposal_ids = {
        proposal.draft_id
        for proposal in current
        if proposal.evidence_origin != "accepted_label_evidence"
        and any(
            anchor.draft_id in retained_anchor_ids
            and _semantic_signature(anchor) == _semantic_signature(proposal)
            and proposal.char_start < anchor.char_end
            and anchor.char_start < proposal.char_end
            for anchor in current
        )
    }
    accepted_conflict_receipts: dict[str, list[str]] = defaultdict(list)
    for proposal in current:
        if (
            proposal.evidence_origin == "accepted_label_evidence"
            or proposal.render_mode != "target_binding"
            or not proposal.target_paths
            or proposal.dependency_paths
            or proposal.dependency_bindings
        ):
            continue
        exact_accepted_conflicts = tuple(
            anchor
            for anchor in current
            if anchor.draft_id in retained_anchor_ids
            and anchor.char_start == proposal.char_start
            and anchor.char_end == proposal.char_end
            and _semantic_signature(anchor) != _semantic_signature(proposal)
        )
        if not exact_accepted_conflicts:
            continue
        represented_elsewhere = _represented_target_inputs(
            tuple(row for row in current if row.draft_id != proposal.draft_id)
        )
        if not set(proposal.target_paths) <= represented_elsewhere:
            continue
        redundant_proposal_ids.add(proposal.draft_id)
        for anchor in exact_accepted_conflicts:
            accepted_conflict_receipts[anchor.draft_id].append(proposal.logical_key)
    if accepted_conflict_receipts:
        current = [
            replace(
                draft,
                rationale=(
                    draft.rationale
                    + " Host retained pinned occurrence ownership over equal-valued competing "
                    "target proposal(s) already represented at another source occurrence: "
                    + ", ".join(sorted(set(accepted_conflict_receipts[draft.draft_id])))
                    + "."
                ),
            )
            if draft.draft_id in accepted_conflict_receipts
            else draft
            for draft in current
        ]
    current = [draft for draft in current if draft.draft_id not in redundant_proposal_ids]
    by_span: dict[tuple[int, int], list[SpanDraft]] = defaultdict(list)
    for draft in current:
        by_span[(draft.char_start, draft.char_end)].append(draft)
    discarded: set[str] = set()
    for same_span in by_span.values():
        signatures = {_semantic_signature(draft) for draft in same_span}
        if len(signatures) <= 1:
            continue
        for draft in same_span:
            if not _provably_unrelated_target_occurrence(draft, source_target):
                continue
            if not any(
                other.logical_key == draft.logical_key
                and (other.char_start, other.char_end) != (draft.char_start, draft.char_end)
                and not _provably_unrelated_target_occurrence(other, source_target)
                for other in current
            ):
                continue
            if any(
                other is not draft
                and not _provably_unrelated_target_occurrence(other, source_target)
                for other in same_span
            ):
                discarded.add(draft.draft_id)
    current = [draft for draft in current if draft.draft_id not in discarded]

    while True:
        ordered = sorted(current, key=lambda row: (row.char_start, row.char_end))
        replacement: tuple[SpanDraft, tuple[SpanDraft, ...]] | None = None
        for outer in ordered:
            contained = tuple(
                inner
                for inner in ordered
                if inner is not outer
                and inner.logical_key != outer.logical_key
                and outer.char_start <= inner.char_start
                and inner.char_end <= outer.char_end
                and (outer.char_start, outer.char_end) != (inner.char_start, inner.char_end)
            )
            pieces = _reconciled_split_drafts(
                raw=raw,
                outer=outer,
                inners=contained,
                source_target=source_target,
            )
            if pieces:
                replacement = outer, pieces
                break
        if replacement is None:
            break
        outer, pieces = replacement
        current = [draft for draft in current if draft is not outer]
        current.extend(pieces)
    return merge_drafts(current)


def merge_drafts(*groups: Iterable[SpanDraft]) -> tuple[SpanDraft, ...]:
    ordered = sorted(
        (draft for group in groups for draft in group),
        key=lambda row: (row.char_start, row.char_end),
    )
    drafts: list[SpanDraft] = []
    for draft in ordered:
        if drafts and (draft.char_start, draft.char_end) == (
            drafts[-1].char_start,
            drafts[-1].char_end,
        ):
            prior = drafts[-1]
            if draft.source_text == prior.source_text and _semantic_signature(
                draft
            ) == _semantic_signature(prior):
                continue
        drafts.append(draft)
    overlaps: list[str] = []
    for left_index, left in enumerate(drafts):
        for right in drafts[left_index + 1 :]:
            if right.char_start >= left.char_end:
                break
            overlaps.append(
                f"{left.draft_id} {left.logical_key} [{left.char_start},{left.char_end}) and "
                f"{right.draft_id} {right.logical_key} [{right.char_start},{right.char_end})"
            )
    if overlaps:
        raise ValueError("template spans overlap:: " + "; ".join(overlaps))
    signatures: dict[str, tuple[Any, ...]] = {}
    for draft in drafts:
        signature = _semantic_signature(draft)
        prior_signature = signatures.setdefault(draft.logical_key, signature)
        if signature != prior_signature:
            raise ValueError(f"logical binding has inconsistent semantics: {draft.logical_key}")
    return tuple(drafts)


def binding_contract_signature(drafts: Sequence[SpanDraft]) -> tuple[tuple[Any, ...], ...]:
    """Return the rendering-relevant state, excluding edit provenance and explanations."""

    return tuple(
        sorted(
            (
                draft.logical_key,
                draft.render_mode,
                draft.value_kind,
                draft.group_kind,
                draft.group_key,
                draft.target_paths,
                draft.derivation,
                draft.dependency_paths,
                draft.dependency_bindings,
                draft.char_start,
                draft.char_end,
                draft.source_text,
                draft.render_policy,
            )
            for draft in drafts
        )
    )


_INDEXED_TARGET_ENTITY = re.compile(
    r"^(documentPatch\.(?:containers|cargoGroups|cargoPackages|cargoAllocationGroups)"
    r"\[[0-9]+\])"
)


def _indexed_target_entity(path: str) -> str | None:
    match = _INDEXED_TARGET_ENTITY.match(path)
    return match.group(1) if match is not None else None


def _same_page_span(
    pages: Mapping[int, tuple[int, int]], *, start: int, end: int
) -> tuple[int, int] | None:
    return next(
        (
            (page_start, page_end)
            for page_start, page_end in pages.values()
            if page_start <= start and end <= page_end
        ),
        None,
    )


def _bracketed_context_score(
    pages: Mapping[int, tuple[int, int]],
    start: int,
    end: int,
    contexts: Sequence[SpanDraft],
) -> tuple[int, int, int] | None:
    page = _same_page_span(pages, start=start, end=end)
    if page is None:
        return None
    page_start, page_end = page
    page_contexts = [
        draft for draft in contexts if page_start <= draft.char_start and draft.char_end <= page_end
    ]
    if not {path for draft in page_contexts for path in draft.target_paths}:
        return None
    preceding = [draft for draft in page_contexts if draft.char_end <= start]
    following = [draft for draft in page_contexts if end <= draft.char_start]
    if preceding and following:
        before = max(preceding, key=lambda row: row.char_end)
        after = min(following, key=lambda row: row.char_start)
        return (
            0,
            start - before.char_end + after.char_start - end,
            after.char_start - before.char_end,
        )
    if preceding:
        before = max(preceding, key=lambda row: row.char_end)
        return (
            1,
            start - before.char_end,
            start - min(row.char_start for row in page_contexts),
        )
    if following:
        after = min(following, key=lambda row: row.char_start)
        return (
            1,
            after.char_start - end,
            max(row.char_end for row in page_contexts) - end,
        )
    return None


def _topology_target_draft(*, raw: str, path: str, start: int, end: int) -> SpanDraft:
    group_kind, group_key = _canonical_target_group((path,))
    value_kind = _value_kind((path,), "natural_text")
    return SpanDraft(
        draft_id=("topology_binding_" + sha256_bytes(f"{path}\0{start}\0{end}".encode())[:16]),
        logical_key="anchor:" + path,
        render_mode="target_binding",
        value_kind=value_kind,
        group_kind=group_kind,
        group_key=group_key,
        target_paths=(path,),
        derivation=None,
        dependency_paths=(),
        dependency_bindings=(),
        char_start=start,
        char_end=end,
        source_text=raw[start:end],
        evidence_origin="host_verified_agent_proposal",
        render_policy=_render_policy_for_value_kind(value_kind, "target_binding"),
        rationale="Host recovered an exact target scalar from unique same-entity topology.",
    )


def _token_equivalent_offsets(
    source_tokens: Sequence[tuple[str, int, int]], surface: str
) -> tuple[tuple[int, int], ...]:
    """Locate contiguous token-identical surfaces across OCR whitespace and punctuation."""

    target_tokens = tuple(token for token, _start, _end in _surface_token_spans(surface))
    if not target_tokens:
        return ()
    width = len(target_tokens)
    return tuple(
        (source_tokens[index][1], source_tokens[index + width - 1][2])
        for index in range(len(source_tokens) - width + 1)
        if tuple(token for token, _start, _end in source_tokens[index : index + width])
        == target_tokens
    )


def _labeled_identifier_projection_offsets(
    source_tokens: Sequence[tuple[str, int, int]], surface: str
) -> tuple[tuple[int, int], ...]:
    """Locate a distinctive identifier printed without its alphabetic field caption."""

    target_tokens = tuple(token for token, _start, _end in _surface_token_spans(surface))
    first_identifier = next(
        (
            index
            for index, token in enumerate(target_tokens)
            if any(char.isdigit() for char in token)
        ),
        None,
    )
    if (
        first_identifier is None
        or first_identifier == 0
        or not all(token.isalpha() for token in target_tokens[:first_identifier])
    ):
        return ()
    projected = target_tokens[first_identifier:]
    if len("".join(projected)) < 5:
        return ()
    width = len(projected)
    return tuple(
        (source_tokens[index][1], source_tokens[index + width - 1][2])
        for index in range(len(source_tokens) - width + 1)
        if tuple(token for token, _start, _end in source_tokens[index : index + width]) == projected
    )


def _repair_uniquely_bracketed_target_paths(
    *,
    raw: str,
    missing_paths: Sequence[str],
    occupied: Sequence[SpanDraft],
    source_target: Mapping[str, Any],
    source_hints: Mapping[str, Sequence[str]] | None = None,
    candidate_span_filter: Callable[[str, int, int], bool] | None = None,
) -> tuple[SpanDraft, ...]:
    """Recover exact scalar slots only when same-entity row topology proves one span."""

    pages = page_body_spans(raw)
    source_tokens = _surface_token_spans(raw)
    contexts_by_path: dict[str, tuple[SpanDraft, ...]] = {}
    ranked_by_path: dict[str, tuple[tuple[tuple[int, int, int], int, int], ...]] = {}
    values_by_path: dict[str, str] = {}
    for path in sorted(missing_paths):
        entity = _indexed_target_entity(path)
        resolved_value = _resolve_target_path(source_target, path)
        value = _scalar_surface(resolved_value)
        if entity is None or value is None or not value:
            continue
        contexts = tuple(
            draft
            for draft in occupied
            if any(
                other_path != path and (other_path == entity or other_path.startswith(entity + "."))
                for other_path in draft.target_paths
            )
        )
        if not contexts:
            continue
        contexts_by_path[path] = contexts
        values_by_path[path] = value
        ranked: list[tuple[tuple[int, int, int], int, int]] = []
        target_surface = _normalized_surface(value)
        compatible_hints = tuple(
            hint
            for hint in (source_hints or {}).get(path, ())
            if _normalized_surface(hint) == target_surface
        )
        candidate_surfaces = tuple(dict.fromkeys((value, *compatible_hints)))
        seen_spans: set[tuple[int, int]] = set()
        for surface in candidate_surfaces:
            for start in _exact_offsets(raw, surface):
                end = start + len(surface)
                if (
                    (
                        candidate_span_filter is not None
                        and not candidate_span_filter(path, start, end)
                    )
                    or not is_token_bounded_surface_span(raw, start, end)
                    or (start, end) in seen_spans
                    or any(start < draft.char_end and draft.char_start < end for draft in occupied)
                ):
                    continue
                seen_spans.add((start, end))
                score = _bracketed_context_score(pages, start, end, contexts)
                if score is not None:
                    ranked.append((score, start, end))
        for start, end in _token_equivalent_offsets(source_tokens, value):
            if (
                (candidate_span_filter is not None and not candidate_span_filter(path, start, end))
                or (start, end) in seen_spans
                or any(start < draft.char_end and draft.char_start < end for draft in occupied)
            ):
                continue
            seen_spans.add((start, end))
            score = _bracketed_context_score(pages, start, end, contexts)
            if score is not None:
                ranked.append((score, start, end))
        for start, end in _labeled_identifier_projection_offsets(source_tokens, value):
            if (
                (candidate_span_filter is not None and not candidate_span_filter(path, start, end))
                or (start, end) in seen_spans
                or any(start < draft.char_end and draft.char_start < end for draft in occupied)
            ):
                continue
            seen_spans.add((start, end))
            score = _bracketed_context_score(pages, start, end, contexts)
            if score is not None:
                ranked.append((score, start, end))
        # A labeled-identifier fallback can be nested inside the complete token-equivalent value
        # found for the same path. Keeping both manufactures an artificial assignment ambiguity;
        # the complete surface is strictly more informative and owns the nested bytes already.
        ranked = [
            row
            for row in ranked
            if not any(
                other_start <= row[1]
                and row[2] <= other_end
                and (other_start, other_end) != (row[1], row[2])
                for _other_score, other_start, other_end in ranked
            )
        ]
        ranked.sort()
        if ranked:
            ranked_by_path[path] = tuple(ranked)

    proposals: list[tuple[str, int, int]] = []
    assigned_paths: set[str] = set()
    paths_by_surface: dict[tuple[str, str, str], list[str]] = defaultdict(list)
    for path, value in values_by_path.items():
        entity = _indexed_target_entity(path)
        assert entity is not None
        paths_by_surface[(value, entity.split("[", 1)[0], path[len(entity) :])].append(path)
    for grouped_paths in paths_by_surface.values():
        if len(grouped_paths) < 2 or any(path not in ranked_by_path for path in grouped_paths):
            continue
        candidate_sets = {
            path: {(start, end) for _score, start, end in ranked_by_path[path]}
            for path in grouped_paths
        }
        all_candidates = set().union(*candidate_sets.values())
        if len(all_candidates) != len(grouped_paths) or any(
            candidates != all_candidates for candidates in candidate_sets.values()
        ):
            continue
        context_positions = {
            path: (
                min(row.char_start for row in contexts_by_path[path]),
                max(row.char_end for row in contexts_by_path[path]),
            )
            for path in grouped_paths
        }
        if len(set(context_positions.values())) != len(grouped_paths):
            continue
        for path, (start, end) in zip(
            sorted(grouped_paths, key=context_positions.__getitem__),
            sorted(all_candidates),
            strict=True,
        ):
            proposals.append((path, start, end))
            assigned_paths.add(path)

    for path, path_ranking in ranked_by_path.items():
        if path in assigned_paths:
            continue
        tied_best = tuple(
            (start, end) for score, start, end in path_ranking if score == path_ranking[0][0]
        )
        if len(tied_best) > 1:
            ordered_ties = tuple(sorted(tied_best))
            repeated_surface = (
                len({_normalized_surface(raw[start:end]) for start, end in ordered_ties}) == 1
            )
            contiguous = all(
                not any(character.isalnum() for character in raw[left_end:right_start])
                for (_left_start, left_end), (right_start, _right_end) in pairwise(ordered_ties)
            )
            if repeated_surface and contiguous:
                proposals.extend((path, start, end) for start, end in ordered_ties)
                assigned_paths.add(path)
            continue
        _score, start, end = ranked[0]
        proposals.append((path, start, end))
    span_counts = Counter((start, end) for _path, start, end in proposals)
    return tuple(
        _topology_target_draft(raw=raw, path=path, start=start, end=end)
        for path, start, end in proposals
        if span_counts[(start, end)] == 1
    )


def _represented_target_inputs(drafts: Sequence[SpanDraft]) -> set[str]:
    """Return target data retained either directly or as a deterministic derivation input."""

    return {
        path
        for draft in drafts
        for path in (
            *draft.target_paths,
            *(draft.dependency_paths if draft.render_mode == "deterministic_derived" else ()),
        )
    }


def _derived_replacement_preserves_structural_target(
    *,
    path: str,
    removed: Sequence[SpanDraft],
    revised: Sequence[SpanDraft],
    source_target: Mapping[str, Any],
) -> bool:
    """Recognize a same-surface derivation that moves from an item node to its collection.

    Structural object/list paths are derivation inputs rather than scalar values. A corrected
    count derivation may therefore replace an item-scoped equipment receipt with the enclosing
    collection while every physical surface remains owned. Scalar paths deliberately require
    exact ownership and never use this relation.
    """

    source_value = _resolve_target_path(source_target, path)
    if not isinstance(source_value, (Mapping, list)):
        return False
    removed_rows = tuple(row for row in removed if path in row.target_paths)
    if not removed_rows:
        return False

    def is_ancestor(ancestor: str) -> bool:
        return path.startswith(ancestor + ".") or path.startswith(ancestor + "[")

    for removed_row in removed_rows:
        replacements = tuple(
            row
            for row in revised
            if row.char_start == removed_row.char_start
            and row.char_end == removed_row.char_end
            and row.render_mode == "deterministic_derived"
        )
        if not any(
            any(
                candidate == path or is_ancestor(candidate)
                for candidate in (*row.target_paths, *row.dependency_paths)
            )
            for row in replacements
        ):
            return False
    return True


def _retarget_unique_unmatched_proposal_occurrences(
    *,
    proposed: Sequence[SpanDraft],
    candidate_paths: Sequence[str],
    source_target: Mapping[str, Any],
) -> tuple[SpanDraft, ...]:
    """Reassign only a source projection that uniquely matches one displaced target fact.

    Compiler outputs occasionally attach one occurrence from a repeated binding to the adjacent
    equal-looking row. The host may correct that occurrence only when it does not project to its
    declared scalar and does project either to exactly one displaced target path globally or to
    exactly one within the same indexed entity. Equal-valued paths within one entity and every
    other ambiguity remain untouched.
    """

    target_surfaces = {
        path: surface
        for path in candidate_paths
        if (surface := _scalar_surface(_resolve_target_path(source_target, path))) is not None
    }
    if not target_surfaces:
        return tuple(proposed)
    output: list[SpanDraft] = []
    for draft in proposed:
        if draft.render_mode != "target_binding" or len(draft.target_paths) != 1:
            output.append(draft)
            continue
        current_path = draft.target_paths[0]
        current_surface = _scalar_surface(_resolve_target_path(source_target, current_path))
        if (
            current_surface is not None
            and _matching_token_projection(draft.source_text, current_surface) is not None
        ):
            output.append(draft)
            continue
        matches = tuple(
            path
            for path, surface in target_surfaces.items()
            if path != current_path
            and _matching_token_projection(draft.source_text, surface) is not None
        )
        current_entity = _indexed_target_entity(current_path)
        same_entity_matches = tuple(
            path for path in matches if _indexed_target_entity(path) == current_entity
        )
        selected_matches = same_entity_matches if len(same_entity_matches) == 1 else matches
        if len(selected_matches) != 1:
            output.append(draft)
            continue
        target_path = selected_matches[0]
        value_kind = _value_kind((target_path,), "natural_text")
        group_kind, group_key = _canonical_target_group((target_path,))
        output.append(
            replace(
                draft,
                draft_id=(
                    "retargeted_proposal_"
                    + sha256_bytes(f"{draft.draft_id}\0{target_path}".encode())[:16]
                ),
                logical_key="anchor:" + target_path,
                value_kind=value_kind,
                group_kind=group_kind,
                group_key=group_key,
                target_paths=(target_path,),
                render_policy=_render_policy_for_value_kind(value_kind, "target_binding"),
                rationale=(
                    draft.rationale
                    + " Host reassigned this unmatched occurrence to the unique displaced "
                    "target scalar that contains its complete token projection."
                ),
            )
        )
    return tuple(output)


def _expand_unique_containing_target_occurrences(
    *, raw: str, proposed: Sequence[SpanDraft], source_target: Mapping[str, Any]
) -> tuple[SpanDraft, ...]:
    """Expand a truncated direct proposal to its unique complete target token.

    This is deliberately narrower than fuzzy quote recovery. The proposal must already name one
    direct scalar target, its selected bytes must be strictly contained by an exact occurrence of
    that target scalar, the complete occurrence must have alphanumeric token boundaries, and that
    containing occurrence must be unique. Expansion is refused when it would cross another
    proposed owner. A numeric substring inside an unrelated identifier therefore remains invalid.
    """

    output: list[SpanDraft] = []
    for draft in proposed:
        if draft.render_mode != "target_binding" or len(draft.target_paths) != 1:
            output.append(draft)
            continue
        target_surface = _scalar_surface(_resolve_target_path(source_target, draft.target_paths[0]))
        if target_surface is None or len(target_surface) <= len(draft.source_text):
            output.append(draft)
            continue
        candidates: list[tuple[int, int]] = []
        for char_start in _exact_offsets(raw, target_surface):
            char_end = char_start + len(target_surface)
            if not (
                char_start <= draft.char_start
                and draft.char_end <= char_end
                and (char_start, char_end) != (draft.char_start, draft.char_end)
            ):
                continue
            if (char_start > 0 and raw[char_start - 1].isalnum()) or (
                char_end < len(raw) and raw[char_end].isalnum()
            ):
                continue
            if any(
                other.draft_id != draft.draft_id
                and char_start < other.char_end
                and other.char_start < char_end
                for other in proposed
            ):
                continue
            candidates.append((char_start, char_end))
        if len(candidates) != 1:
            output.append(draft)
            continue
        char_start, char_end = candidates[0]
        output.append(
            replace(
                draft,
                draft_id=(
                    "expanded_target_occurrence_"
                    + sha256_bytes(f"{draft.draft_id}\0{char_start}\0{char_end}".encode())[:16]
                ),
                char_start=char_start,
                char_end=char_end,
                source_text=raw[char_start:char_end],
                evidence_origin="host_verified_agent_proposal",
                rationale=(
                    draft.rationale
                    + " Host expanded the selected substring to the unique enclosing exact "
                    "target token with valid boundaries."
                ),
            )
        )
    return tuple(output)


def _relocate_competing_indexed_target_occurrences(
    *,
    raw: str,
    proposed: Sequence[SpanDraft],
    contexts: Sequence[SpanDraft],
    source_target: Mapping[str, Any],
) -> tuple[SpanDraft, ...]:
    """Resolve exact-span target collisions through one unique row-topology assignment."""

    proposed_ids = {draft.draft_id for draft in proposed}
    external_contexts = tuple(draft for draft in contexts if draft.draft_id not in proposed_ids)
    by_span: dict[tuple[int, int], list[SpanDraft]] = defaultdict(list)
    for draft in (*proposed, *external_contexts):
        by_span[(draft.char_start, draft.char_end)].append(draft)
    conflicting_ids: set[str] = set()
    conflicting_paths: list[str] = []
    source_hints: dict[str, tuple[str, ...]] = {}
    for same_span in by_span.values():
        proposed_rows = tuple(row for row in same_span if row.draft_id in proposed_ids)
        if len(same_span) < 2 or not proposed_rows:
            continue
        if any(
            draft.render_mode != "target_binding"
            or len(draft.target_paths) != 1
            or _indexed_target_entity(draft.target_paths[0]) is None
            for draft in same_span
        ):
            continue
        paths = tuple(draft.target_paths[0] for draft in same_span)
        if len(set(paths)) != len(paths):
            continue
        components = target_fact_components(source_target, paths)
        if len(components) != len(paths):
            continue
        conflicting_ids.update(draft.draft_id for draft in proposed_rows)
        conflicting_paths.extend(draft.target_paths[0] for draft in proposed_rows)
        for draft in proposed_rows:
            source_hints[draft.target_paths[0]] = (draft.source_text,)
    if not conflicting_ids:
        return tuple(proposed)

    # A collision with one retained row can reveal an offset assignment across a complete family
    # of equal-valued indexed facts. Repairing only the colliding member would select whichever
    # unoccupied duplicate remains, not the member's row. Include every proposed sibling with the
    # same collection, field suffix, and scalar value; the bracketed topology solver below still
    # applies the repair only when the complete one-to-one assignment is unique.
    family_signatures: set[tuple[str, str, str]] = set()
    for path in conflicting_paths:
        entity = _indexed_target_entity(path)
        surface = _scalar_surface(_resolve_target_path(source_target, path))
        if entity is not None and surface is not None:
            family_signatures.add(
                (entity.split("[", 1)[0], path[len(entity) :], _normalized_surface(surface))
            )
    family_rows: list[SpanDraft] = []
    for draft in proposed:
        if draft.render_mode != "target_binding" or len(draft.target_paths) != 1:
            continue
        path = draft.target_paths[0]
        entity = _indexed_target_entity(path)
        surface = _scalar_surface(_resolve_target_path(source_target, path))
        if entity is None or surface is None:
            continue
        signature = (
            entity.split("[", 1)[0],
            path[len(entity) :],
            _normalized_surface(surface),
        )
        if signature in family_signatures:
            family_rows.append(draft)
    family_paths = tuple(dict.fromkeys(row.target_paths[0] for row in family_rows))
    if len(target_fact_components(source_target, family_paths)) == len(family_paths):
        conflicting_ids.update(row.draft_id for row in family_rows)
        conflicting_paths.extend(family_paths)
        for row in family_rows:
            source_hints.setdefault(row.target_paths[0], (row.source_text,))

    retained = tuple(draft for draft in proposed if draft.draft_id not in conflicting_ids)
    occupied = tuple(
        draft for draft in (*contexts, *retained) if draft.draft_id not in conflicting_ids
    )
    repairs = _repair_uniquely_bracketed_target_paths(
        raw=raw,
        missing_paths=tuple(dict.fromkeys(conflicting_paths)),
        occupied=occupied,
        source_target=source_target,
        source_hints=source_hints,
    )
    repaired_paths = _represented_target_inputs(repairs)
    if repaired_paths != set(conflicting_paths):
        return tuple(proposed)
    repaired = tuple(
        replace(
            draft,
            rationale=(
                draft.rationale
                + " Host relocated colliding equal-valued target facts through a complete "
                "unique same-entity topology assignment."
            ),
        )
        for draft in repairs
    )
    return (*retained, *repaired)


_PACKAGE_QUANTITY_PATH = re.compile(r"^documentPatch\.cargoPackages\[([0-9]+)\]\.quantity$")
_PACKAGE_TYPE_PATH = re.compile(r"^documentPatch\.cargoPackages\[([0-9]+)\]\.typeCategory$")
_CARGO_DESCRIPTION_PATH = re.compile(r"^documentPatch\.cargoGroups\[([0-9]+)\]\.description$")
_ALLOCATION_PACKAGE_QUANTITY_PATH = re.compile(
    r"^documentPatch\.cargoAllocationGroups\[([0-9]+)\]\.allocations\[([0-9]+)\]"
    r"\.packageQuantity$"
)


def _normalize_misplaced_package_count_bindings(
    *, raw: str, drafts: Sequence[SpanDraft], source_target: Mapping[str, Any]
) -> tuple[SpanDraft, ...]:
    """Relocate one malformed quantity/type binding to proven direct leaf surfaces.

    The structured package pair must be unique in the source label. Candidate spans are derived
    from tokens, allowing a multiplication marker and parenthesized plural inflection. Quantity
    and category remain separate mutable facts; ambiguous equal package pairs remain untouched.
    """

    patch = source_target.get("documentPatch")
    packages = patch.get("cargoPackages") if isinstance(patch, Mapping) else None
    if not isinstance(packages, list):
        return tuple(drafts)
    pair_counts: Counter[tuple[bytes, bytes]] = Counter()
    for package in packages:
        if isinstance(package, Mapping):
            pair_counts[
                (
                    canonical_json_bytes(package.get("quantity")),
                    canonical_json_bytes(package.get("typeCategory")),
                )
            ] += 1

    grouped: dict[str, list[SpanDraft]] = defaultdict(list)
    for draft in drafts:
        grouped[draft.logical_key].append(draft)
    discarded: set[str] = set()
    additions: list[SpanDraft] = []
    lines = line_spans(raw)
    for rows in grouped.values():
        first = rows[0]
        quantity_paths = tuple(
            path for path in first.target_paths if _PACKAGE_QUANTITY_PATH.fullmatch(path)
        )
        type_paths = tuple(
            path for path in first.target_paths if _PACKAGE_TYPE_PATH.fullmatch(path)
        )
        if not (
            first.render_mode in {"target_binding", "agent_residual"}
            and len(first.target_paths) == 2
            and len(quantity_paths) == 1
            and len(type_paths) == 1
            and all(row.target_paths == first.target_paths for row in rows)
        ):
            continue
        quantity_match = _PACKAGE_QUANTITY_PATH.fullmatch(quantity_paths[0])
        type_match = _PACKAGE_TYPE_PATH.fullmatch(type_paths[0])
        assert quantity_match is not None and type_match is not None
        package_index = int(quantity_match.group(1))
        if package_index != int(type_match.group(1)) or package_index >= len(packages):
            continue
        package = packages[package_index]
        if not isinstance(package, Mapping):
            continue
        pair = (
            canonical_json_bytes(package.get("quantity")),
            canonical_json_bytes(package.get("typeCategory")),
        )
        if pair_counts[pair] != 1:
            continue
        quantity_surface = _scalar_surface(package.get("quantity"))
        category_surface = _package_category_surface(package.get("typeCategory"))
        if quantity_surface is None or category_surface is None:
            continue
        components: list[tuple[tuple[int, int], tuple[int, int]]] = []
        for line in lines:
            parsed_candidates = _package_component_span_candidates(
                line.text,
                quantity=package.get("quantity"),
                category=package.get("typeCategory"),
            )
            if not parsed_candidates:
                continue
            if len(parsed_candidates) != 1:
                components = []
                break
            quantity_span, category_span = parsed_candidates[0]
            components.append(
                (
                    (
                        line.char_start + quantity_span[0],
                        line.char_start + quantity_span[1],
                    ),
                    (
                        line.char_start + category_span[0],
                        line.char_start + category_span[1],
                    ),
                )
            )
        components = sorted(set(components))
        if not components:
            continue
        replaced_ids = {row.draft_id for row in rows}
        if any(
            start < other.char_end and other.char_start < end and other.draft_id not in replaced_ids
            for component in components
            for start, end in component
            for other in drafts
        ):
            continue
        discarded.update(replaced_ids)
        for role, target_paths, value_kind, component_index in (
            ("quantity", quantity_paths, "integer", 0),
            ("category", type_paths, "package", 1),
        ):
            group_kind, group_key = _canonical_target_group(target_paths)
            logical_key = "anchor:" + "|".join(target_paths)
            for component in components:
                start, end = component[component_index]
                additions.append(
                    replace(
                        first,
                        draft_id=(
                            "host_package_component_locality_"
                            + sha256_bytes(f"{logical_key}\0{role}\0{start}\0{end}".encode())[:16]
                        ),
                        logical_key=logical_key,
                        render_mode="target_binding",
                        value_kind=value_kind,
                        group_kind=group_kind,
                        group_key=group_key,
                        target_paths=target_paths,
                        derivation=None,
                        dependency_paths=(),
                        dependency_bindings=(),
                        char_start=start,
                        char_end=end,
                        source_text=raw[start:end],
                        evidence_origin="host_verified_agent_proposal",
                        render_policy=_render_policy_for_value_kind(value_kind, "target_binding"),
                        rationale=(
                            first.rationale
                            + " Host relocated the unique structured package pair and split its "
                            f"direct mutable {role} leaf."
                        ),
                    )
                )
    return (*tuple(row for row in drafts if row.draft_id not in discarded), *additions)


def normalize_package_quantity_row_locality(
    *, raw: str, drafts: Sequence[SpanDraft]
) -> tuple[SpanDraft, ...]:
    """Remove short package-quantity repeats that lack any package-field context.

    A short integer is not self-identifying. When one logical package quantity has a proven
    occurrence beside its package-category owner, other equal numerals remain occurrences only if
    they have the same row relationship or an explicit package-count caption. This prevents clause
    numbers and page-local legal numerals from entering the quantity binding.
    """

    lines = line_spans(raw)
    grouped: dict[str, list[SpanDraft]] = defaultdict(list)
    type_lines: dict[int, set[int]] = defaultdict(set)
    for row in drafts:
        grouped[row.logical_key].append(row)
        for path in row.target_paths:
            match = _PACKAGE_TYPE_PATH.fullmatch(path)
            if match is not None:
                type_lines[int(match.group(1))].add(line_number_for_char(lines, row.char_start))
    discarded: set[str] = set()
    for rows in grouped.values():
        first = rows[0]
        package_indexes = {
            int(match.group(1))
            for path in first.target_paths
            if (match := _PACKAGE_QUANTITY_PATH.fullmatch(path)) is not None
        }
        if (
            first.render_mode != "target_binding"
            or not package_indexes
            or any(not row.source_text.strip().isdigit() for row in rows)
        ):
            continue

        def has_package_context(
            row: SpanDraft, package_indexes: frozenset[int] = frozenset(package_indexes)
        ) -> bool:
            line_number = line_number_for_char(lines, row.char_start)
            if any(line_number in type_lines[index] for index in package_indexes):
                return True
            current = _normalized_surface(lines[line_number - 1].text)
            previous = _normalized_surface(lines[line_number - 2].text) if line_number >= 2 else ""
            context = current + previous
            package_markers = (
                "numberandkindofpackages",
                "numberofpackages",
                "noofpkgs",
                "packagesreceived",
            )
            if any(marker in context for marker in package_markers):
                return True

            # Column-oriented OCR often emits each table heading on its own line before the
            # first data row. Walk only through blank lines, recognized companion headings, and
            # one already-owned cargo value; any other text closes the header scope. This covers
            # arbitrary heading order without allowing a package heading to leak into later legal
            # clause numbers on the same page.
            companion_markers = (
                "containernossealnos",
                "containerno",
                "containernumber",
                "sealno",
                "sealnumber",
                "descriptionofgoods",
                "grossweight",
                "grosswt",
                "totalgrwt",
                "grwt",
                "netweight",
                "netwt",
                "tareweight",
                "tarewt",
                "measurement",
                "marksnos",
                "marksnumbers",
                "marksandnumbers",
                "volume",
                "volcbm",
            )
            for prior in reversed(lines[: line_number - 1]):
                normalized = _normalized_surface(prior.text)
                if not normalized:
                    continue
                if _PAGE_HEADER.fullmatch(prior.text.strip()) is not None:
                    return False
                if any(marker in normalized for marker in package_markers):
                    return True
                if any(marker in normalized for marker in companion_markers):
                    continue
                trimmed = _trimmed_line_span(prior)
                if trimmed is not None and any(
                    draft.group_kind in {"cargo", "package"}
                    and draft.char_start <= trimmed[0]
                    and trimmed[1] <= draft.char_end
                    for draft in drafts
                ):
                    continue
                return False
            return False

        contextual = tuple(row for row in rows if has_package_context(row))
        if not contextual or len(contextual) == len(rows):
            continue
        discarded.update(row.draft_id for row in rows if row not in contextual)
    return merge_drafts(row for row in drafts if row.draft_id not in discarded)


def normalize_package_quantity_repeats(
    *, raw: str, drafts: Sequence[SpanDraft], source_target: Mapping[str, Any]
) -> tuple[SpanDraft, ...]:
    """Append an exact quantity repeated immediately beside its package-type owner."""

    lines = line_spans(raw)
    grouped: dict[str, list[SpanDraft]] = defaultdict(list)
    type_rows_by_index: dict[int, list[SpanDraft]] = defaultdict(list)
    for row in drafts:
        grouped[row.logical_key].append(row)
        indexes = {
            int(match.group(1))
            for path in row.target_paths
            if (match := _PACKAGE_TYPE_PATH.fullmatch(path)) is not None
        }
        if len(indexes) == 1:
            type_rows_by_index[next(iter(indexes))].append(row)

    additions: list[SpanDraft] = []
    for rows in grouped.values():
        first = rows[0]
        package_indexes = {
            int(match.group(1))
            for path in first.target_paths
            if (match := _PACKAGE_QUANTITY_PATH.fullmatch(path)) is not None
        }
        if (
            len(package_indexes) != 1
            or any(
                row.render_mode != "target_binding"
                or row.target_paths != first.target_paths
                or row.source_text != first.source_text
                for row in rows
            )
            or target_path_relationship(source_target, first.target_paths)
            not in {"single_target", "shared_value_equality"}
        ):
            continue
        package_index = next(iter(package_indexes))
        for type_row in type_rows_by_index.get(package_index, ()):
            line_number = line_number_for_char(lines, type_row.char_start)
            candidate_lines = lines[max(0, line_number - 2) : min(len(lines), line_number + 1)]
            matches = tuple(
                span
                for line in candidate_lines
                for span in _line_token_sequence_spans(line, first.source_text)
            )
            adjacent = tuple(
                (start, end)
                for start, end in matches
                if (
                    (end <= type_row.char_start and not raw[end : type_row.char_start].strip())
                    or (type_row.char_end <= start and not raw[type_row.char_end : start].strip())
                )
            )
            if len(adjacent) != 1:
                continue
            start, end = adjacent[0]
            if any(start < row.char_end and row.char_start < end for row in drafts):
                continue
            additions.append(
                replace(
                    first,
                    draft_id=(
                        "host_package_quantity_repeat_"
                        + sha256_bytes(f"{first.logical_key}\0{start}\0{end}".encode())[:16]
                    ),
                    char_start=start,
                    char_end=end,
                    source_text=raw[start:end],
                    evidence_origin="host_verified_agent_proposal",
                    rationale=(
                        first.rationale
                        + " Host appended this exact quantity from its immediate same-row "
                        "package-category owner."
                    ),
                )
            )
    return merge_drafts(drafts, additions)


def normalize_cargo_description_linked_package_prefixes(
    *, raw: str, drafts: Sequence[SpanDraft], source_target: Mapping[str, Any]
) -> tuple[SpanDraft, ...]:
    """Assign an inline ``quantity + package type`` prefix to its cargo group.

    Repeated package values cannot identify their structured row by value.  A prefix immediately
    adjacent to a directly owned cargo description can: ``groupId`` joins that description to one
    package and its required allocation/package quantity co-binding.  This pass changes ownership
    only when the join is unique, both prefix tokens exactly realize the joined target values, and
    every competing owner covers precisely the same token with an equal-valued package target.
    """

    patch = source_target.get("documentPatch")
    if not isinstance(patch, Mapping):
        return tuple(drafts)
    cargo_groups = patch.get("cargoGroups")
    packages = patch.get("cargoPackages")
    if not isinstance(cargo_groups, list) or not isinstance(packages, list):
        return tuple(drafts)

    package_indexes_by_group: dict[str, list[int]] = defaultdict(list)
    for package_index, package in enumerate(packages):
        if not isinstance(package, Mapping):
            continue
        group_id = package.get("groupId")
        if isinstance(group_id, str) and group_id:
            package_indexes_by_group[group_id].append(package_index)

    quantity_paths_by_package: dict[int, tuple[str, ...]] = {}
    requirements = required_target_cobindings(source_target)
    for package_index in range(len(packages)):
        package_path = f"documentPatch.cargoPackages[{package_index}].quantity"
        matches = tuple(
            tuple(sorted(requirement.target_paths))
            for requirement in requirements
            if package_path in requirement.target_paths
        )
        if len(matches) == 1:
            quantity_paths_by_package[package_index] = matches[0]

    lines = line_spans(raw)
    planned: dict[tuple[int, int], tuple[tuple[str, ...], str, str]] = {}
    for description_owner in drafts:
        description_matches = tuple(
            _CARGO_DESCRIPTION_PATH.fullmatch(path)
            for path in description_owner.target_paths
            if _CARGO_DESCRIPTION_PATH.fullmatch(path) is not None
        )
        if description_owner.render_mode != "target_binding" or len(description_matches) != 1:
            continue
        description_match = description_matches[0]
        if description_match is None:
            continue
        cargo_index = int(description_match.group(1))
        if cargo_index >= len(cargo_groups) or not isinstance(cargo_groups[cargo_index], Mapping):
            continue
        group_id = cargo_groups[cargo_index].get("groupId")
        if not isinstance(group_id, str):
            continue
        package_indexes = package_indexes_by_group.get(group_id, ())
        if len(package_indexes) != 1:
            continue
        package_index = package_indexes[0]
        package = packages[package_index]
        if not isinstance(package, Mapping):
            continue
        quantity_paths = quantity_paths_by_package.get(package_index)
        type_path = f"documentPatch.cargoPackages[{package_index}].typeCategory"
        quantity_surface = (
            _scalar_surface(_resolve_target_path(source_target, quantity_paths[0]))
            if quantity_paths
            else None
        )
        category_surface = (
            _package_category_surface(_resolve_target_path(source_target, type_path))
            if "typeCategory" in package
            else None
        )
        if quantity_paths is None or quantity_surface is None or category_surface is None:
            continue

        line_number = line_number_for_char(lines, description_owner.char_start)
        line = lines[line_number - 1]
        category_candidates = tuple(
            (line.char_start + start, line.char_start + end)
            for token, start, end in _surface_token_spans(line.text)
            if token in _package_inflections(category_surface)
            and line.char_start + end <= description_owner.char_start
            and not raw[line.char_start + end : description_owner.char_start].strip()
        )
        pairs = tuple(
            (quantity_span, category_span)
            for category_span in category_candidates
            for quantity_span in _line_token_sequence_spans(line, quantity_surface)
            if quantity_span[1] <= category_span[0]
            and not raw[quantity_span[1] : category_span[0]].strip()
        )
        if len(pairs) != 1:
            continue
        quantity_span, category_span = pairs[0]
        candidate_rows = (
            (quantity_span, quantity_paths, "integer", "numeric_surface"),
            (category_span, (type_path,), "package", "categorical_surface"),
        )
        if any(
            span in planned and planned[span][0] != target_paths
            for span, target_paths, _value_kind_name, _render_policy_name in candidate_rows
        ):
            continue
        for span, target_paths, value_kind_name, render_policy_name in candidate_rows:
            planned[span] = (target_paths, value_kind_name, render_policy_name)

    if not planned:
        return tuple(drafts)

    discarded: set[str] = set()
    retained_spans: set[tuple[int, int]] = set()
    accepted_plans: dict[tuple[int, int], tuple[tuple[str, ...], str, str]] = {}
    for span, plan in sorted(planned.items()):
        target_paths, value_kind_name, _render_policy_name = plan
        start, end = span
        overlaps = tuple(row for row in drafts if start < row.char_end and row.char_start < end)
        exact_correct = tuple(
            row
            for row in overlaps
            if row.char_start == start
            and row.char_end == end
            and row.render_mode == "target_binding"
            and set(row.target_paths) == set(target_paths)
        )
        if exact_correct and len(exact_correct) == len(overlaps):
            retained_spans.add(span)
            accepted_plans[span] = plan
            continue

        def equal_valued_package_owner(
            row: SpanDraft,
            *,
            expected_start: int = start,
            expected_end: int = end,
            expected_value_kind: str = value_kind_name,
        ) -> bool:
            if (
                row.char_start != expected_start
                or row.char_end != expected_end
                or row.render_mode != "target_binding"
                or not row.target_paths
            ):
                return False
            if expected_value_kind == "integer":
                if not all(
                    path.endswith(".packageQuantity")
                    or _PACKAGE_QUANTITY_PATH.fullmatch(path) is not None
                    for path in row.target_paths
                ):
                    return False
                return all(
                    _normalized_surface(row.source_text)
                    == _normalized_surface(
                        _scalar_surface(_resolve_target_path(source_target, path)) or ""
                    )
                    for path in row.target_paths
                )
            if not all(_PACKAGE_TYPE_PATH.fullmatch(path) is not None for path in row.target_paths):
                return False
            return all(
                _normalized_surface(row.source_text)
                in _package_inflections(
                    _package_category_surface(_resolve_target_path(source_target, path)) or ""
                )
                for path in row.target_paths
            )

        if overlaps and not all(equal_valued_package_owner(row) for row in overlaps):
            continue
        discarded.update(row.draft_id for row in overlaps)
        accepted_plans[span] = plan

    additions: list[SpanDraft] = []
    for (start, end), (target_paths, value_kind_name, render_policy_name) in sorted(
        accepted_plans.items()
    ):
        if (start, end) in retained_spans:
            continue
        group_kind, group_key = _canonical_target_group(target_paths)
        additions.append(
            SpanDraft(
                draft_id=(
                    "host_cargo_linked_package_prefix_"
                    + sha256_bytes(f"{'|'.join(target_paths)}\0{start}\0{end}".encode())[:16]
                ),
                logical_key="anchor:" + "|".join(target_paths),
                render_mode="target_binding",
                value_kind=value_kind_name,
                group_kind=group_kind,
                group_key=group_key,
                target_paths=target_paths,
                derivation=None,
                dependency_paths=(),
                dependency_bindings=(),
                char_start=start,
                char_end=end,
                source_text=raw[start:end],
                evidence_origin="host_verified_agent_proposal",
                render_policy=render_policy_name,
                rationale=(
                    "Host assigned this inline quantity/package-category prefix through the "
                    "unique cargo-description groupId join."
                ),
            )
        )
    return merge_drafts(
        (row for row in drafts if row.draft_id not in discarded),
        additions,
    )


def normalize_container_linked_package_rows(
    *, raw: str, drafts: Sequence[SpanDraft], source_target: Mapping[str, Any]
) -> tuple[SpanDraft, ...]:
    """Own ``quantity + package type`` beside a uniquely linked container-type slot."""

    patch = source_target.get("documentPatch")
    if not isinstance(patch, Mapping):
        return tuple(drafts)
    containers = patch.get("containers")
    allocation_groups = patch.get("cargoAllocationGroups")
    packages = patch.get("cargoPackages")
    if not all(isinstance(value, list) for value in (containers, allocation_groups, packages)):
        return tuple(drafts)
    assert isinstance(containers, list)
    assert isinstance(allocation_groups, list)
    assert isinstance(packages, list)
    container_indexes = {
        container.get("containerNumber"): index
        for index, container in enumerate(containers)
        if isinstance(container, Mapping) and isinstance(container.get("containerNumber"), str)
    }
    package_indexes = {
        package.get("packageId"): index
        for index, package in enumerate(packages)
        if isinstance(package, Mapping) and isinstance(package.get("packageId"), str)
    }
    required_sets = {
        frozenset(requirement.target_paths)
        for requirement in required_target_cobindings(source_target)
    }
    links: dict[int, list[tuple[tuple[str, ...], str]]] = defaultdict(list)
    for group_index, group in enumerate(allocation_groups):
        if not isinstance(group, Mapping):
            continue
        allocations = group.get("allocations")
        if not isinstance(allocations, list):
            continue
        for allocation_index, allocation in enumerate(allocations):
            if not isinstance(allocation, Mapping):
                continue
            container_index = container_indexes.get(allocation.get("containerNumber"))
            package_index = package_indexes.get(allocation.get("packageId"))
            if container_index is None or package_index is None:
                continue
            package = packages[package_index]
            if not isinstance(package, Mapping) or package.get("groupId") != group.get("groupId"):
                continue
            quantity_paths = (
                "documentPatch.cargoAllocationGroups"
                f"[{group_index}].allocations[{allocation_index}].packageQuantity",
                f"documentPatch.cargoPackages[{package_index}].quantity",
            )
            type_path = f"documentPatch.cargoPackages[{package_index}].typeCategory"
            if frozenset(quantity_paths) not in required_sets or "typeCategory" not in package:
                continue
            links[container_index].append((quantity_paths, type_path))

    lines = line_spans(raw)
    additions: list[SpanDraft] = []
    seen: set[tuple[tuple[str, ...], int, int]] = set()
    for row in drafts:
        container_indexes_for_row = {
            int(match.group(1))
            for path in (*row.target_paths, *row.dependency_paths)
            if (match := _CONTAINER_TYPE_PATH.fullmatch(path)) is not None
        }
        if len(container_indexes_for_row) != 1:
            continue
        container_index = next(iter(container_indexes_for_row))
        if len(links.get(container_index, ())) != 1:
            continue
        linked_quantity_paths, type_path = links[container_index][0]
        line_number = line_number_for_char(lines, row.char_start)
        if line_number != line_number_for_char(lines, row.char_end - 1):
            continue
        line = lines[line_number - 1]
        quantity_surface = _scalar_surface(
            _resolve_target_path(source_target, linked_quantity_paths[0])
        )
        type_surface = _package_category_surface(_resolve_target_path(source_target, type_path))
        if quantity_surface is None or type_surface is None:
            continue
        quantity_spans = _line_token_sequence_spans(line, quantity_surface)
        type_spans = tuple(
            (line.char_start + start, line.char_start + end)
            for token, start, end in _surface_token_spans(line.text)
            if token in _package_inflections(type_surface)
        )
        pairs = tuple(
            (quantity_span, type_span)
            for quantity_span in quantity_spans
            for type_span in type_spans
            if (
                quantity_span[1] <= type_span[0]
                and not raw[quantity_span[1] : type_span[0]].strip()
            )
        )
        if len(pairs) != 1:
            continue
        for target_paths, span, value_kind, render_policy in (
            (linked_quantity_paths, pairs[0][0], "integer", "numeric_surface"),
            ((type_path,), pairs[0][1], "package", "categorical_surface"),
        ):
            start, end = span
            identity = (target_paths, start, end)
            if identity in seen:
                continue
            overlaps = tuple(
                other for other in drafts if start < other.char_end and other.char_start < end
            )
            if overlaps:
                if all(
                    other.render_mode == "target_binding"
                    and other.target_paths == target_paths
                    and other.char_start == start
                    and other.char_end == end
                    for other in overlaps
                ):
                    seen.add(identity)
                continue
            group_kind, group_key = _canonical_target_group(target_paths)
            additions.append(
                SpanDraft(
                    draft_id=(
                        "host_container_linked_package_"
                        + sha256_bytes(f"{'|'.join(target_paths)}\0{start}\0{end}".encode())[:16]
                    ),
                    logical_key="anchor:" + "|".join(target_paths),
                    render_mode="target_binding",
                    value_kind=value_kind,
                    group_kind=group_kind,
                    group_key=group_key,
                    target_paths=target_paths,
                    derivation=None,
                    dependency_paths=(),
                    dependency_bindings=(),
                    char_start=start,
                    char_end=end,
                    source_text=raw[start:end],
                    evidence_origin="host_verified_agent_proposal",
                    render_policy=render_policy,
                    rationale=(
                        "Host assigned this exact quantity/package-category pair from the "
                        "same row as a uniquely linked structured container type."
                    ),
                )
            )
            seen.add(identity)
    return merge_drafts(drafts, additions)


def normalize_package_type_row_locality(
    *,
    raw: str,
    proposed: Sequence[SpanDraft],
    contexts: Sequence[SpanDraft],
    candidate_paths: Sequence[str],
    source_target: Mapping[str, Any],
) -> tuple[SpanDraft, ...]:
    """Bind a package-category token to its uniquely co-printed package quantity row.

    Repeated ``CARTONS``-style tokens carry no identity by themselves. A package quantity target
    does: when one package's quantity binding and exactly one inflected category token share a
    source line, that row proves the category occurrence's package index. Rows that contain
    multiple quantities or category candidates remain untouched.
    """

    eligible_indexes = {
        int(match.group(1))
        for path in candidate_paths
        if (match := _PACKAGE_TYPE_PATH.fullmatch(path)) is not None
    } | {
        int(match.group(1))
        for draft in proposed
        for path in draft.target_paths
        if (match := _PACKAGE_TYPE_PATH.fullmatch(path)) is not None
    }
    if not eligible_indexes:
        return tuple(proposed)
    lines = line_spans(raw)
    line_by_id = {line.line_id: line for line in lines}
    assignments: dict[tuple[int, int], set[int]] = defaultdict(set)
    for context in contexts:
        package_indexes = {
            int(match.group(1))
            for path in context.target_paths
            if (match := _PACKAGE_QUANTITY_PATH.fullmatch(path)) is not None
            and int(match.group(1)) in eligible_indexes
        }
        if len(package_indexes) != 1:
            continue
        line_start, line_end = line_range_for_chars(lines, context.char_start, context.char_end)
        if line_start != line_end:
            continue
        package_index = next(iter(package_indexes))
        type_path = f"documentPatch.cargoPackages[{package_index}].typeCategory"
        category = _package_category_surface(_resolve_target_path(source_target, type_path))
        if category is None:
            continue
        accepted = _package_inflections(category)
        line = line_by_id[line_start]
        matches = tuple(
            (line.char_start + start, line.char_start + end)
            for token, start, end in _surface_token_spans(raw[line.char_start : line.char_end])
            if token in accepted
        )
        if len(matches) == 1:
            assignments[matches[0]].add(package_index)
    proven_assignments = {
        span: next(iter(indexes)) for span, indexes in assignments.items() if len(indexes) == 1
    }
    if not proven_assignments:
        return tuple(proposed)

    output: list[SpanDraft] = []
    occupied_assignments: set[tuple[int, int]] = set()
    for draft in proposed:
        type_paths = tuple(
            path for path in draft.target_paths if _PACKAGE_TYPE_PATH.fullmatch(path) is not None
        )
        assigned_package_index = proven_assignments.get((draft.char_start, draft.char_end))
        retargetable_source_only = (
            not draft.target_paths
            and not draft.dependency_paths
            and not draft.dependency_bindings
            and draft.render_mode in {"agent_residual", "deterministic_auxiliary", "literal_static"}
        )
        if assigned_package_index is None or not (
            (draft.render_mode == "target_binding" and len(type_paths) == 1)
            or retargetable_source_only
        ):
            output.append(draft)
            continue
        target_path = f"documentPatch.cargoPackages[{assigned_package_index}].typeCategory"
        group_kind, group_key = _canonical_target_group((target_path,))
        occupied_assignments.add((draft.char_start, draft.char_end))
        if (
            draft.render_mode == "target_binding"
            and draft.logical_key == "anchor:" + target_path
            and draft.value_kind == "package"
            and draft.group_kind == group_kind
            and draft.group_key == group_key
            and draft.target_paths == (target_path,)
            and draft.render_policy == "categorical_surface"
        ):
            output.append(draft)
            continue
        output.append(
            replace(
                draft,
                logical_key="anchor:" + target_path,
                value_kind="package",
                group_kind=group_kind,
                group_key=group_key,
                target_paths=(target_path,),
                render_policy="categorical_surface",
                rationale=(
                    draft.rationale
                    + " Host assigned this repeated package category to the unique package "
                    "quantity printed on the same source line."
                ),
            )
        )
    owners_by_path: dict[str, list[SpanDraft]] = defaultdict(list)
    for draft in output:
        for path in draft.target_paths:
            owners_by_path[path].append(draft)
    for (start, end), package_index in sorted(proven_assignments.items()):
        if (start, end) in occupied_assignments:
            continue
        target_path = f"documentPatch.cargoPackages[{package_index}].typeCategory"
        existing_owners = owners_by_path.get(target_path, [])
        if any(
            owner.render_mode == "target_binding"
            and owner.target_paths == (target_path,)
            and owner.char_start <= start
            and end <= owner.char_end
            for owner in existing_owners
        ):
            # OCR inflection can make the token scan find ``Pallet`` inside the already owned
            # complete value ``Pallet(s)``.  The wider target owner is authoritative; adding the
            # contained token would create overlapping ownership for the same fact.
            occupied_assignments.add((start, end))
            continue
        if existing_owners and not (
            len({owner.logical_key for owner in existing_owners}) == 1
            and all(
                owner.render_mode == "target_binding" and owner.target_paths == (target_path,)
                for owner in existing_owners
            )
        ):
            # A composite or residual owner already realizes this categorical fact.  Creating a
            # second direct owner would violate the one-owner invariant, so leave it unchanged.
            continue
        if existing_owners:
            owner = existing_owners[0]
            added = replace(
                owner,
                draft_id=(
                    "package_type_locality_"
                    + sha256_bytes(f"{target_path}\0{start}\0{end}".encode())[:16]
                ),
                char_start=start,
                char_end=end,
                source_text=raw[start:end],
                evidence_origin="host_verified_agent_proposal",
                rationale=(
                    owner.rationale
                    + " Host appended the unique package-category token co-printed with this "
                    "package's quantity as another occurrence of the same target owner."
                ),
            )
        else:
            group_kind, group_key = _canonical_target_group((target_path,))
            added = SpanDraft(
                draft_id=(
                    "package_type_locality_"
                    + sha256_bytes(f"{target_path}\0{start}\0{end}".encode())[:16]
                ),
                logical_key="anchor:" + target_path,
                render_mode="target_binding",
                value_kind="package",
                group_kind=group_kind,
                group_key=group_key,
                target_paths=(target_path,),
                derivation=None,
                dependency_paths=(),
                dependency_bindings=(),
                char_start=start,
                char_end=end,
                source_text=raw[start:end],
                evidence_origin="host_verified_agent_proposal",
                render_policy="categorical_surface",
                rationale=(
                    "Host bound the unique package-category token co-printed with this "
                    "package's quantity."
                ),
            )
        output.append(added)
        owners_by_path[target_path].append(added)
    return tuple(output)


def _drop_exact_source_only_duplicates(
    *, retained: Sequence[SpanDraft], proposed: Sequence[SpanDraft]
) -> tuple[SpanDraft, ...]:
    """Discard a targetless proposal that duplicates one exact target-owned byte span.

    One physical span cannot simultaneously be an independently generated source-only value and
    the printed realization of a structured target.  This correction is deliberately exact: the
    spans must be identical, the accepted owner must be target-backed, and the proposal may not
    declare any target or dependency.  Containment and unequal spans remain review work.
    """

    target_owned_spans = {
        (row.char_start, row.char_end)
        for row in retained
        if row.target_paths and row.render_mode in {"target_binding", "carrier_static"}
    }
    return tuple(
        row
        for row in proposed
        if not (
            (row.char_start, row.char_end) in target_owned_spans
            and not row.target_paths
            and not row.dependency_paths
            and not row.dependency_bindings
            and row.render_mode in {"deterministic_auxiliary", "agent_residual", "literal_static"}
        )
    )


def _resolve_exact_target_source_only_conflicts(
    *, drafts: Sequence[SpanDraft], source_target: Mapping[str, Any]
) -> tuple[SpanDraft, ...]:
    """Choose the uniquely target-compatible owner of an exact duplicate span.

    This handles model proposals that assign one byte span both to a direct target and to a
    source-only field. A target wins only when its printed surface is a normalized projection of
    the structured scalar. Otherwise it is removed only when that target path remains represented
    by another physical occurrence and the competing source-only contract is unique.
    """

    by_span: dict[tuple[int, int], list[SpanDraft]] = defaultdict(list)
    for row in drafts:
        by_span[(row.char_start, row.char_end)].append(row)
    discarded: set[str] = set()
    for rows in by_span.values():
        targets = tuple(
            row
            for row in rows
            if row.render_mode == "target_binding" and len(row.target_paths) == 1
        )
        source_only = tuple(
            row
            for row in rows
            if row.render_mode
            in {"deterministic_auxiliary", "agent_residual", "carrier_static", "literal_static"}
            and not row.target_paths
            and not row.dependency_paths
            and not row.dependency_bindings
        )
        if not targets or not source_only or len({row.logical_key for row in source_only}) != 1:
            continue
        compatible: list[SpanDraft] = []
        for row in targets:
            target_value = _resolve_target_path(source_target, row.target_paths[0])
            target_surface = (
                _package_category_surface(target_value)
                if row.target_paths[0].endswith(".typeCategory")
                else _scalar_surface(target_value)
            )
            if target_surface is not None and (
                _normalized_surface(row.source_text) in _normalized_surface(target_surface)
                or _normalized_surface(target_surface) in _normalized_surface(row.source_text)
            ):
                compatible.append(row)
        if len(compatible) == 1:
            discarded.update(row.draft_id for row in source_only)
            discarded.update(row.draft_id for row in targets if row is not compatible[0])
            continue
        if compatible:
            continue
        target_paths = {row.target_paths[0] for row in targets}
        if all(
            any(path in other.target_paths and other not in rows for other in drafts)
            for path in target_paths
        ):
            discarded.update(row.draft_id for row in targets)
    return tuple(row for row in drafts if row.draft_id not in discarded)


def _repair_overridden_freight_payment(
    *,
    raw: str,
    source_target: Mapping[str, Any],
    removed: Sequence[SpanDraft],
    occupied: Sequence[SpanDraft],
) -> tuple[SpanDraft, ...]:
    """Recover the selected token from an overridden ``FREIGHT <value>`` surface.

    This is deliberately narrower than general substring recovery: the structured value must be
    PREPAID or COLLECT, every removed owner must have one semantic contract, and the selected token
    must immediately follow FREIGHT without the opposite option on that same owned surface.
    """

    candidates = tuple(row for row in removed if _FREIGHT_PAYMENT_PATH in row.target_paths)
    if not candidates:
        return ()
    if (
        len({row.logical_key for row in candidates}) != 1
        or len({_semantic_signature(row) for row in candidates}) != 1
    ):
        return ()
    try:
        value = _resolve_target_path(source_target, _FREIGHT_PAYMENT_PATH)
    except ValueError:
        return ()
    surface = _scalar_surface(value)
    if surface is None:
        return ()
    selected = _normalized_surface(surface)
    if selected not in {"prepaid", "collect"}:
        return ()
    opposite = "collect" if selected == "prepaid" else "prepaid"
    repairs: list[SpanDraft] = []
    for row in candidates:
        tokens = _surface_token_spans(row.source_text)
        selected_tokens = tuple(
            (token_start, token_end)
            for index, (token, token_start, token_end) in enumerate(tokens)
            if token == selected
            and index > 0
            and tokens[index - 1][0] == "freight"
            and opposite not in {candidate for candidate, _start, _end in tokens}
        )
        if len(selected_tokens) != 1:
            return ()
        relative_start, relative_end = selected_tokens[0]
        start = row.char_start + relative_start
        end = row.char_start + relative_end
        if any(start < other.char_end and other.char_start < end for other in occupied):
            return ()
        repairs.append(
            replace(
                row,
                draft_id=(
                    "host_overridden_freight_payment_"
                    + sha256_bytes(f"{row.logical_key}\0{start}\0{end}".encode())[:16]
                ),
                char_start=start,
                char_end=end,
                source_text=raw[start:end],
                evidence_origin="host_verified_agent_proposal",
                rationale=(
                    row.rationale
                    + " Host retained only the selected payment token from the explicitly "
                    "overridden FREIGHT caption surface."
                ),
            )
        )
    return merge_drafts(repairs)


def _is_unselected_form_title_anchor(*, raw: str, anchor: SpanDraft) -> bool:
    """Return whether a negotiability anchor is only document-form caption grammar."""

    if anchor.target_paths != (_NEGOTIABILITY_PATH,):
        return False
    lines = line_spans(raw)
    line_number = line_number_for_char(lines, anchor.char_start)
    if line_number != line_number_for_char(lines, anchor.char_end - 1):
        return False
    line = lines[line_number - 1]
    if _FORM_TITLE_WITHOUT_SELECTED_NEGOTIABILITY.fullmatch(line.text) is not None:
        return True
    return bool(
        _heading_like(line)
        and _DOCUMENT_FORM_NOUN.fullmatch(anchor.source_text) is not None
        and _EXPLICIT_NEGOTIABILITY_SELECTION.search(line.text) is None
    )


def apply_anchor_overrides(
    *,
    raw: str,
    source_target: Mapping[str, Any],
    anchors: Sequence[SpanDraft],
    proposed: Sequence[SpanDraft],
    overrides: Sequence[AnchorOverride],
    semantic_only_target_paths: Sequence[str] = (),
) -> tuple[SpanDraft, ...]:
    by_id = {anchor.draft_id: anchor for anchor in anchors}
    if len(by_id) != len(anchors):
        raise ValueError("anchor binding IDs are not unique")
    override_ids = tuple(row.anchor_binding_id for row in overrides)
    if len(set(override_ids)) != len(override_ids):
        raise ValueError("compiler anchor overrides are not unique")
    unknown = sorted(set(override_ids) - set(by_id))
    if unknown:
        raise ValueError("compiler referenced unknown anchor overrides: " + ", ".join(unknown))
    proposed = _remove_unbounded_auxiliary_fragments(proposed)
    proposed = _trim_targetless_residual_edges_owned_by_auxiliaries(
        raw=raw,
        drafts=proposed,
    )
    proposal_ids = {row.draft_id for row in proposed}
    early_reconciled = _remove_redundant_multitarget_occurrences((*anchors, *proposed))
    early_reconciled = _remove_auxiliaries_contained_by_same_scope_vessel_targets(early_reconciled)
    proposed = tuple(row for row in early_reconciled if row.draft_id in proposal_ids)
    explicitly_overridden = set(override_ids)
    represented_by_proposals = _represented_target_inputs(proposed) | set(
        semantic_only_target_paths
    )
    implicitly_overridden: list[str] = []
    for anchor in anchors:
        if anchor.draft_id in explicitly_overridden:
            continue
        semantic_form_title = bool(
            set(anchor.target_paths).intersection(semantic_only_target_paths)
            and _is_unselected_form_title_anchor(raw=raw, anchor=anchor)
        )
        if semantic_form_title:
            implicitly_overridden.append(anchor.draft_id)
            continue
        if not set(anchor.target_paths) <= represented_by_proposals:
            continue
        has_conflicting_replacement = any(
            proposal.logical_key != anchor.logical_key
            and _semantic_signature(proposal) != _semantic_signature(anchor)
            and proposal.char_start < anchor.char_end
            and anchor.char_start < proposal.char_end
            for proposal in proposed
        )
        if has_conflicting_replacement:
            implicitly_overridden.append(anchor.draft_id)
    override_ids = (*override_ids, *implicitly_overridden)
    if implicitly_overridden:
        implicit_paths = {
            path for anchor_id in implicitly_overridden for path in by_id[anchor_id].target_paths
        }
        proposed = tuple(
            replace(
                proposal,
                rationale=(
                    proposal.rationale
                    + " Host inferred removal of a conflicting accepted anchor only after the "
                    "candidate transaction supplied complete replacement ownership for all of "
                    "that anchor's target paths."
                ),
            )
            if implicit_paths.intersection((*proposal.target_paths, *proposal.dependency_paths))
            else proposal
            for proposal in proposed
        )
    represented_after_override = _represented_target_inputs(
        tuple(anchor for anchor in anchors if anchor.draft_id not in override_ids)
    ) | _represented_target_inputs(proposed)
    required_cobindings = {
        frozenset(requirement.target_paths)
        for requirement in required_target_cobindings(source_target)
    }
    all_overridden_ids = frozenset(override_ids)
    restored_ids: set[str] = set()
    semantic_contradiction_ids: set[str] = set()
    for anchor in anchors:
        anchor_paths = frozenset(anchor.target_paths)
        contradicts_semantic_only = bool(anchor_paths.intersection(semantic_only_target_paths))
        lacks_replacement = not (
            anchor_paths.intersection(represented_after_override)
            or any(
                anchor_paths.intersection((*proposal.target_paths, *proposal.dependency_paths))
                or (proposal.char_start < anchor.char_end and anchor.char_start < proposal.char_end)
                for proposal in proposed
            )
        )
        if (
            anchor.draft_id not in all_overridden_ids
            or anchor.evidence_origin != "accepted_label_evidence"
            or anchor.render_mode != "target_binding"
            or anchor_paths not in required_cobindings
            or target_path_relationship(source_target, anchor.target_paths)
            != "shared_value_equality"
            or not (contradicts_semantic_only or lacks_replacement)
            or any(
                _surface_match(
                    source=anchor.source_text,
                    target=_resolve_target_path(source_target, path),
                    adapter=_surface_adapter(anchor),
                    value_kind=anchor.value_kind,
                )
                is None
                for path in anchor.target_paths
            )
        ):
            continue
        restored_ids.add(anchor.draft_id)
        if contradicts_semantic_only:
            semantic_contradiction_ids.add(anchor.draft_id)
    if restored_ids:
        anchors = tuple(
            replace(
                anchor,
                rationale=(
                    anchor.rationale
                    + (
                        " Host retained this accepted required co-binding and rejected the "
                        "compiler's semantic-only declaration because the pinned source span "
                        "still exactly realizes every equal structured target. "
                        + _SEMANTIC_ONLY_REJECTION_RECEIPT
                        if anchor.draft_id in semantic_contradiction_ids
                        else " Host retained this accepted required co-binding because the "
                        "explicit override supplied neither contradictory ownership nor a "
                        "replacement, and the source still exactly realizes every equal "
                        "structured target."
                    )
                ),
            )
            if anchor.draft_id in restored_ids
            else anchor
            for anchor in anchors
        )
        by_id = {anchor.draft_id: anchor for anchor in anchors}
        override_ids = tuple(
            anchor_id for anchor_id in override_ids if anchor_id not in restored_ids
        )
    removed = tuple(by_id[anchor_id] for anchor_id in override_ids)
    retained = tuple(anchor for anchor in anchors if anchor.draft_id not in override_ids)
    # Overrides address physical anchor occurrences, not an entire logical target.  A repeated
    # target remains completely owned when another accepted occurrence survives the transaction;
    # requiring the provider to re-propose that retained occurrence creates a duplicate edit with
    # no semantic effect.  Only paths that truly lose all accepted ownership require replacement.
    displaced_paths = sorted(
        {path for draft in removed for path in draft.target_paths}
        - _represented_target_inputs(retained)
        - set(semantic_only_target_paths)
    )
    proposed = _expand_unique_containing_target_occurrences(
        raw=raw,
        proposed=proposed,
        source_target=source_target,
    )
    proposed = _retarget_unique_unmatched_proposal_occurrences(
        proposed=proposed,
        candidate_paths=displaced_paths,
        source_target=source_target,
    )
    proposed = normalize_package_type_row_locality(
        raw=raw,
        proposed=proposed,
        contexts=(*retained, *proposed),
        candidate_paths=displaced_paths,
        source_target=source_target,
    )
    proposed = _relocate_competing_indexed_target_occurrences(
        raw=raw,
        proposed=proposed,
        contexts=(*retained, *proposed),
        source_target=source_target,
    )
    proposed_paths = _represented_target_inputs(proposed)
    missing_paths = sorted(set(displaced_paths) - proposed_paths)
    if _FREIGHT_PAYMENT_PATH in missing_paths:
        freight_repairs = _repair_overridden_freight_payment(
            raw=raw,
            source_target=source_target,
            removed=removed,
            occupied=(*retained, *proposed),
        )
        proposed = (*proposed, *freight_repairs)
        missing_paths = sorted(set(missing_paths) - _represented_target_inputs(freight_repairs))
    if missing_paths:
        source_hints = {
            path: tuple(
                dict.fromkeys(draft.source_text for draft in removed if path in draft.target_paths)
            )
            for path in missing_paths
        }
        repairs = _repair_uniquely_bracketed_target_paths(
            raw=raw,
            missing_paths=missing_paths,
            occupied=(*retained, *proposed),
            source_target=source_target,
            source_hints=source_hints,
        )
        proposed = (*proposed, *repairs)
        missing_paths = sorted(set(missing_paths) - _represented_target_inputs(repairs))
    if missing_paths:
        occupied = tuple((*retained, *proposed))
        lines = line_spans(raw)
        hints: list[str] = []
        for path in missing_paths:
            source_values = tuple(
                dict.fromkeys(draft.source_text for draft in removed if path in draft.target_paths)
            )
            candidates: list[str] = []
            for source_value in source_values:
                for start in _exact_offsets(raw, source_value):
                    end = start + len(source_value)
                    if not is_token_bounded_surface_span(raw, start, end) or any(
                        start < draft.char_end and draft.char_start < end for draft in occupied
                    ):
                        continue
                    line_start, line_end = line_range_for_chars(lines, start, end)
                    location = line_start if line_start == line_end else f"{line_start}-{line_end}"
                    candidates.append(f"{location}={source_value!r}")
            hints.append(
                f"{path} unowned exact candidates: "
                + (", ".join(dict.fromkeys(candidates)) if candidates else "none")
            )
        raise ValueError(
            "anchor override lacks replacement target ownership: "
            + ", ".join(missing_paths)
            + "; evidence hints: "
            + "; ".join(hints)
        )
    # A compiler-added repeat that explicitly joins an accepted anchor inherits that binding's
    # complete rendering contract.  Preserving only ``value_kind`` leaves one logical binding with
    # contradictory render/group/derivation semantics and forces a needless model repair.  Exact
    # target-path equality keeps this normalization narrow: proposals that redefine or repartition
    # an anchor must still use the explicit atomic override path above.
    retained_by_key: dict[str, SpanDraft] = {}
    for draft in retained:
        prior = retained_by_key.setdefault(draft.logical_key, draft)
        if (
            _semantic_signature(prior) != _semantic_signature(draft)
            or prior.render_policy != draft.render_policy
        ):
            raise ValueError(
                f"retained logical binding has inconsistent semantics: {draft.logical_key}"
            )

    harmonized_proposals: list[SpanDraft] = []
    for draft in proposed:
        retained_draft = retained_by_key.get(draft.logical_key)
        targetless_canonical_carrier_repeat = (
            retained_draft is not None
            and retained_draft.render_mode == "carrier_static"
            and retained_draft.target_paths
            and all(
                path.startswith("documentPatch.parties.carrier")
                for path in retained_draft.target_paths
            )
            and draft.render_mode == "carrier_static"
            and not draft.target_paths
            and _normalized_surface(draft.source_text)
            == _normalized_surface(retained_draft.source_text)
        )
        if retained_draft is None or (
            draft.target_paths != retained_draft.target_paths
            and not targetless_canonical_carrier_repeat
        ):
            harmonized_proposals.append(draft)
            continue
        harmonized_proposals.append(
            replace(
                draft,
                render_mode=retained_draft.render_mode,
                value_kind=retained_draft.value_kind,
                group_kind=retained_draft.group_kind,
                group_key=retained_draft.group_key,
                target_paths=retained_draft.target_paths,
                derivation=retained_draft.derivation,
                dependency_paths=retained_draft.dependency_paths,
                dependency_bindings=retained_draft.dependency_bindings,
                render_policy=retained_draft.render_policy,
                rationale=(
                    draft.rationale
                    + " Host inherited the complete accepted-anchor contract for this exact "
                    "repeated target occurrence."
                ),
            )
        )
    harmonized_proposals = list(
        _drop_exact_source_only_duplicates(
            retained=retained,
            proposed=harmonized_proposals,
        )
    )
    deduplicated = _resolve_exact_target_source_only_conflicts(
        drafts=(*retained, *harmonized_proposals),
        source_target=source_target,
    )
    deduplicated = _remove_redundant_multitarget_occurrences(deduplicated)
    deduplicated = _remove_auxiliaries_contained_by_same_scope_vessel_targets(deduplicated)
    caption_local = normalize_labeled_shipper_and_receipt_locality(
        raw=raw,
        drafts=deduplicated,
        source_target=source_target,
    )
    return reconcile_draft_overlaps(
        raw=raw,
        drafts=caption_local,
        source_target=source_target,
    )


def refuted_semantic_only_target_paths(drafts: Sequence[SpanDraft]) -> frozenset[str]:
    """Return semantic-only declarations disproved by retained pinned source evidence."""

    return frozenset(
        path
        for draft in drafts
        if draft.evidence_origin == "accepted_label_evidence"
        and _SEMANTIC_ONLY_REJECTION_RECEIPT in draft.rationale
        for path in draft.target_paths
    )


def inventory_binding_id(logical_key: str) -> str:
    """Return the opaque edit handle for one complete logical inventory binding."""

    return "inventory_binding_" + sha256_bytes(logical_key.encode("utf-8"))[:16]


def apply_critic_patch(
    *,
    raw: str,
    drafts: Sequence[SpanDraft],
    findings: Sequence[CriticFinding],
    remove_inventory_binding_ids: Sequence[str],
    additional_bindings: Sequence[AgentBindingProposal],
    occurrence_removals: Sequence[CriticOccurrenceRemoval] = (),
    semantic_only_target_paths: Sequence[str] = (),
    source_target: Mapping[str, Any],
) -> tuple[SpanDraft, ...]:
    """Apply one critic revision without permitting unrelated binding churn.

    Every newly claimed physical occurrence must touch a source line cited by a finding. An
    occurrence outside those lines is permitted only when it is the exact physical span of a
    binding removed by the same atomic transaction; this lets a valid split/rekey preserve prior
    ownership without licensing unrelated source edits. Existing edits are addressed either at
    logical-binding granularity or by exact occurrence. Partial occurrence removal preserves the
    existing semantic contract and is permitted only for cited source spans; it cannot empty the
    binding. An entirely uncited group may be removed only when a replacement covers all of its
    exact physical spans. Target ownership removed by the patch must still exist afterwards. The
    next independent critic pass remains responsible for semantic acceptance of the repaired
    inventory.
    """

    if not any(
        (
            remove_inventory_binding_ids,
            occurrence_removals,
            additional_bindings,
        )
    ):
        raise ValueError("critic revision did not supply a transactional patch")
    replacement_kinds = {
        "carrier_binding_error",
        "incorrect_semantic_owner",
        "incorrect_static_classification",
        "missing_derivation",
        "topology_or_grouping_error",
    }
    if any(finding.finding_kind in replacement_kinds for finding in findings) and not (
        remove_inventory_binding_ids or occurrence_removals
    ):
        raise ValueError("critic reported a visible binding defect without removing its owner")
    cited_line_numbers = {int(line_id[1:]) for finding in findings for line_id in finding.line_ids}
    if not cited_line_numbers:
        raise ValueError("critic patch has no cited source lines")
    lines = line_spans(raw)
    grouped: dict[str, list[SpanDraft]] = defaultdict(list)
    inventory_keys: dict[str, str] = {}
    for draft in drafts:
        binding_id = inventory_binding_id(draft.logical_key)
        prior_key = inventory_keys.setdefault(binding_id, draft.logical_key)
        if prior_key != draft.logical_key:
            raise ValueError("logical inventory binding ID collision")
        grouped[binding_id].append(draft)
    removal_ids = tuple(remove_inventory_binding_ids)
    unknown = sorted(set(removal_ids) - set(grouped))
    if unknown:
        raise ValueError(
            "critic referenced unknown inventory binding removals: " + ", ".join(unknown)
        )
    full_removal_keys = {inventory_keys[binding_id] for binding_id in removal_ids}
    removal_groups = tuple(grouped[binding_id] for binding_id in removal_ids)
    occurrence_removal_keys = tuple(row.logical_key for row in occurrence_removals)
    if len(set(occurrence_removal_keys)) != len(occurrence_removal_keys):
        raise ValueError("critic occurrence-removal logical keys must be unique")
    collision = sorted(full_removal_keys & set(occurrence_removal_keys))
    if collision:
        raise ValueError(
            "critic cannot fully and partially remove the same binding: " + ", ".join(collision)
        )
    grouped_by_key = {
        inventory_keys[binding_id]: logical_drafts for binding_id, logical_drafts in grouped.items()
    }
    partial_removal_spans: set[tuple[str, int, int]] = set()
    for removal in occurrence_removals:
        logical_drafts = grouped_by_key.get(removal.logical_key)
        if logical_drafts is None:
            raise ValueError(
                "critic occurrence removal references an unknown logical key: "
                + removal.logical_key
            )
        for occurrence in removal.occurrences:
            char_start, char_end = _resolve_occurrence(
                raw=raw,
                occurrence=occurrence,
                line_rows=lines,
            )
            identity = (removal.logical_key, char_start, char_end)
            if identity in partial_removal_spans:
                raise ValueError("critic occurrence-removal spans must be unique")
            if not any(
                draft.char_start == char_start and draft.char_end == char_end
                for draft in logical_drafts
            ):
                raise ValueError(
                    "critic occurrence removal does not belong to its logical binding: "
                    f"{removal.logical_key} {occurrence.line_start}-{occurrence.line_end}"
                )
            start_line, end_line = line_range_for_chars(lines, char_start, char_end)
            removal_occupied_lines = set(range(int(start_line[1:]), int(end_line[1:]) + 1))
            if not removal_occupied_lines & cited_line_numbers:
                raise ValueError(
                    "critic occurrence removal is outside its cited findings: "
                    f"{removal.logical_key} {start_line}-{end_line}"
                )
            partial_removal_spans.add(identity)
        if len(
            partial_removal_spans
            & {(removal.logical_key, draft.char_start, draft.char_end) for draft in logical_drafts}
        ) == len(logical_drafts):
            raise ValueError(
                "critic occurrence removal cannot empty a binding; use full removal: "
                + removal.logical_key
            )
    removed_physical_spans = {
        (draft.char_start, draft.char_end)
        for logical_drafts in removal_groups
        for draft in logical_drafts
    } | {(char_start, char_end) for _logical_key, char_start, char_end in partial_removal_spans}
    validate_agent_proposal_paths(proposals=additional_bindings, source_target=source_target)
    additions = resolve_agent_proposals(raw=raw, proposals=additional_bindings)
    scoped_additions: list[SpanDraft] = []
    for draft in additions:
        start_line, end_line = line_range_for_chars(lines, draft.char_start, draft.char_end)
        occupied_lines = set(range(int(start_line[1:]), int(end_line[1:]) + 1))
        if occupied_lines & cited_line_numbers:
            scoped_additions.append(draft)
            continue
        if (draft.char_start, draft.char_end) in removed_physical_spans:
            scoped_additions.append(
                replace(
                    draft,
                    rationale=(
                        draft.rationale
                        + " Host preserved an exact physical occurrence from a binding removed "
                        "by this atomic transaction."
                    ),
                )
            )
            continue
        raise ValueError(
            "critic logical addition is outside its cited findings and is not an exact "
            f"carried replacement span: {draft.logical_key} {start_line}-{end_line}"
        )
    additions = tuple(scoped_additions)
    additions_by_key: dict[str, list[SpanDraft]] = defaultdict(list)
    for draft in additions:
        additions_by_key[draft.logical_key].append(draft)
    replacement_spans = {
        (draft.char_start, draft.char_end)
        for logical_drafts in additions_by_key.values()
        for draft in logical_drafts
    }
    for binding_id, logical_drafts in zip(removal_ids, removal_groups, strict=True):
        removed_occupied_lines: set[int] = set()
        line_ranges: list[str] = []
        for draft in logical_drafts:
            start_line, end_line = line_range_for_chars(lines, draft.char_start, draft.char_end)
            removed_occupied_lines.update(range(int(start_line[1:]), int(end_line[1:]) + 1))
            line_ranges.append(f"{start_line}-{end_line}")
        if not removed_occupied_lines & cited_line_numbers and not all(
            (draft.char_start, draft.char_end) in replacement_spans for draft in logical_drafts
        ):
            raise ValueError(
                f"critic removal is outside its cited findings: {binding_id} "
                + ",".join(line_ranges)
            )
    partially_removed = tuple(
        draft
        for draft in drafts
        if (draft.logical_key, draft.char_start, draft.char_end) in partial_removal_spans
    )
    removed = (
        *(draft for logical_drafts in removal_groups for draft in logical_drafts),
        *partially_removed,
    )
    removal_set = set(removal_ids)
    retained = tuple(
        draft
        for draft in drafts
        if inventory_binding_id(draft.logical_key) not in removal_set
        and (draft.logical_key, draft.char_start, draft.char_end) not in partial_removal_spans
    )
    combined: list[SpanDraft] = []
    combined_identities: set[tuple[int, int, tuple[Any, ...]]] = set()
    for draft in (*retained, *additions):
        combined_identity = (draft.char_start, draft.char_end, _semantic_signature(draft))
        if combined_identity in combined_identities:
            continue
        combined_identities.add(combined_identity)
        combined.append(draft)
    revised = normalize_target_cobindings(
        drafts=reconcile_draft_overlaps(
            raw=raw,
            drafts=tuple(combined),
            source_target=source_target,
        ),
        source_target=source_target,
    )
    removed_target_paths = {path for draft in removed for path in draft.target_paths}
    invalid_semantic_only = sorted(set(semantic_only_target_paths) - removed_target_paths)
    if invalid_semantic_only:
        raise ValueError(
            "critic may classify only explicitly removed target ownership as semantic-only: "
            + ", ".join(invalid_semantic_only)
        )
    revised_target_paths = _represented_target_inputs(revised)
    missing = sorted(
        path
        for path in removed_target_paths - revised_target_paths - set(semantic_only_target_paths)
        if not _derived_replacement_preserves_structural_target(
            path=path,
            removed=removed,
            revised=revised,
            source_target=source_target,
        )
    )
    if missing:
        raise ValueError("critic patch drops target ownership: " + ", ".join(missing))
    validate_target_binding_relationships(drafts=revised, source_target=source_target, raw=raw)
    if binding_contract_signature(revised) == binding_contract_signature(drafts):
        raise ValueError(
            "critic transaction is a functional no-op after host canonicalization; "
            "do not repeat the finding or patch unless the rendering contract actually changes"
        )
    return revised


def uncovered_risks(
    raw: str, risks: Sequence[RiskCandidate], drafts: Sequence[SpanDraft]
) -> tuple[RiskCandidate, ...]:
    if not risks:
        return ()
    ordered_drafts = tuple(sorted(drafts, key=lambda row: (row.char_start, row.char_end)))
    draft_starts = tuple(draft.char_start for draft in ordered_drafts)

    def mutable_content_is_covered(char_start: int, char_end: int) -> bool:
        cursor = char_start
        first = max(0, bisect.bisect_right(draft_starts, char_start) - 1)
        for span in ordered_drafts[first:]:
            if span.char_end <= cursor or span.char_start >= char_end:
                if span.char_start >= char_end:
                    break
                continue
            gap_end = min(span.char_start, char_end)
            if any(character.isalnum() for character in raw[cursor:gap_end]):
                return False
            cursor = max(cursor, min(span.char_end, char_end))
            if cursor >= char_end:
                return True
        return not any(character.isalnum() for character in raw[cursor:char_end])

    output: list[RiskCandidate] = []
    encoded = raw.encode("utf-8")
    for risk in risks:
        try:
            char_start = len(encoded[: risk.byte_start].decode("utf-8", errors="strict"))
            char_end = len(encoded[: risk.byte_end].decode("utf-8", errors="strict"))
        except UnicodeDecodeError as error:
            raise ValueError("risk candidate byte span is not UTF-8 aligned") from error
        if not mutable_content_is_covered(char_start, char_end):
            output.append(risk)
    return tuple(output)


def masked_source(raw: str, drafts: Sequence[SpanDraft]) -> str:
    by_line: dict[int, list[tuple[int, int, str]]] = defaultdict(list)
    lines = line_spans(raw)
    for draft in drafts:
        for line in lines:
            start = max(draft.char_start, line.char_start)
            end = min(draft.char_end, line.char_end)
            if start < end:
                by_line[line.number].append(
                    (
                        start - line.char_start,
                        end - line.char_start,
                        f"⟦{draft.logical_key}:{draft.render_mode}⟧",
                    )
                )
    rendered: list[str] = []
    for line in lines:
        cursor = 0
        parts: list[str] = []
        for start, end, marker in sorted(by_line[line.number]):
            parts.append(line.text[cursor:start])
            parts.append(marker)
            cursor = end
        parts.append(line.text[cursor:])
        rendered.append(f"{line.line_id} | {''.join(parts)}")
    return "\n".join(rendered)


def annotated_source(raw: str, drafts: Sequence[SpanDraft]) -> str:
    """Render the complete source with explicit ownership boundaries and no hidden text."""

    by_line: dict[int, list[tuple[int, int, str]]] = defaultdict(list)
    lines = line_spans(raw)
    for draft in drafts:
        for line in lines:
            start = max(draft.char_start, line.char_start)
            end = min(draft.char_end, line.char_end)
            if start < end:
                by_line[line.number].append(
                    (
                        start - line.char_start,
                        end - line.char_start,
                        f"⟦{draft.logical_key}:{draft.render_mode}⟧",
                    )
                )
    rendered: list[str] = []
    for line in lines:
        cursor = 0
        parts: list[str] = []
        for start, end, marker in sorted(by_line[line.number]):
            parts.append(line.text[cursor:start])
            parts.append(marker)
            parts.append(line.text[start:end])
            parts.append("⟦/binding⟧")
            cursor = end
        parts.append(line.text[cursor:])
        rendered.append(f"{line.line_id} | {''.join(parts)}")
    return "\n".join(rendered)


def source_carrier(source_target: Mapping[str, Any]) -> str | None:
    try:
        value = source_target["documentPatch"]["parties"]["carrier"]["name"]
    except (KeyError, TypeError):
        return None
    if value is None:
        return None
    if not isinstance(value, str) or not value.strip():
        raise ValueError("source target carrier name is empty")
    return value.strip()


def normalize_pinned_carrier_assessment(
    *,
    assessment: CarrierAssessment,
    expected: str | None,
    raw: str,
    drafts: Sequence[SpanDraft],
) -> CarrierAssessment:
    """Derive pinned-carrier evidence from the validated target owner.

    When the source label already pins a carrier, the agent has no carrier-resolution decision to
    make. The exact carrier-static draft owning that target is stronger evidence than a second,
    fallible line citation in the model output. Missing target ownership still fails closed in
    ``validate_carrier_assessment``; only the redundant evidence pointer is replaced.
    """

    if expected is None:
        return assessment
    evidence_drafts = tuple(
        sorted(
            {
                (draft.char_start, draft.char_end): draft
                for draft in drafts
                if draft.render_mode == "carrier_static"
                and _CARRIER_NAME_PATH in draft.target_paths
            }.values(),
            key=lambda draft: (draft.char_start, draft.char_end),
        )
    )
    if not evidence_drafts:
        raise ValueError("pinned source carrier lacks carrier-static target evidence")
    lines = line_spans(raw)
    by_number = {line.number: line for line in lines}
    evidence: list[AgentOccurrence] = []
    for draft in evidence_drafts:
        line_start, line_end = line_range_for_chars(lines, draft.char_start, draft.char_end)
        range_start = by_number[int(line_start[1:])].char_start
        range_end = by_number[int(line_end[1:])].char_end
        offsets = _exact_offsets(raw[range_start:range_end], draft.source_text)
        relative_start = draft.char_start - range_start
        if relative_start not in offsets:
            raise ValueError("carrier-static target draft cannot be reconstructed as exact OCR")
        evidence.append(
            AgentOccurrence(
                line_start=line_start,
                line_end=line_end,
                source_text=draft.source_text,
                occurrence_index=offsets.index(relative_start),
            )
        )
    return CarrierAssessment(
        canonical_name=expected,
        aliases=(),
        evidence_occurrences=tuple(evidence),
        source="source_label_confirmed_by_ocr",
        rationale=(
            "The pinned source-label carrier is confirmed by exact carrier-static ownership of "
            "the structured carrier-name target."
        ),
    )


def validate_carrier_assessment(
    *,
    assessment: CarrierAssessment,
    expected: str | None,
    raw: str,
    anchor_drafts_value: Sequence[SpanDraft],
) -> None:
    if expected is not None and assessment.canonical_name != expected:
        raise ValueError(
            "agent carrier differs from source label: "
            f"{assessment.canonical_name!r} != {expected!r}"
        )
    expected_source = (
        "source_label_confirmed_by_ocr"
        if expected is not None
        else "ocr_resolved_missing_source_label"
    )
    if assessment.source != expected_source:
        raise ValueError(
            f"carrier assessment source is {assessment.source}, expected {expected_source}"
        )
    evidence_spans = tuple(
        _resolve_occurrence(raw=raw, occurrence=occurrence)
        for occurrence in assessment.evidence_occurrences
    )
    if expected is None:
        normalized_evidence = re.sub(
            r"[^a-z0-9]+",
            " ",
            " ".join(row.source_text for row in assessment.evidence_occurrences).lower(),
        ).strip()
        normalized_name = re.sub(r"[^a-z0-9]+", " ", assessment.canonical_name.lower()).strip()
        if normalized_name not in normalized_evidence:
            raise ValueError("canonical carrier name is not present in its exact OCR evidence")
        if assessment.canonical_name not in " ".join(
            row.source_text for row in assessment.evidence_occurrences
        ):
            raise ValueError(
                "OCR-resolved canonical carrier name is not copied exactly from evidence"
            )
    carrier_drafts = tuple(
        draft for draft in anchor_drafts_value if draft.render_mode == "carrier_static"
    )
    if not carrier_drafts:
        raise ValueError("source carrier lacks an exact carrier-static target binding")
    if not all(
        any(draft.char_start <= start and draft.char_end >= end for draft in carrier_drafts)
        for start, end in evidence_spans
    ):
        raise ValueError("carrier evidence is not fully owned by carrier-static bindings")
    if expected is not None and not any(
        any(path == "documentPatch.parties.carrier.name" for path in draft.target_paths)
        and any(
            draft.char_start <= start and draft.char_end >= end for start, end in evidence_spans
        )
        for draft in carrier_drafts
    ):
        raise ValueError("carrier evidence is not linked to the pinned carrier-name target")
    if len(set(assessment.aliases)) != len(assessment.aliases):
        raise ValueError("carrier aliases must be unique")
    for alias in assessment.aliases:
        if alias == assessment.canonical_name:
            raise ValueError("carrier alias duplicates the canonical name")
        starts: list[int] = []
        cursor = 0
        while True:
            start = raw.find(alias, cursor)
            if start < 0:
                break
            starts.append(start)
            cursor = start + len(alias)
        if not starts:
            raise ValueError(f"carrier alias is not copied exactly from OCR: {alias!r}")
        if not any(
            any(
                draft.char_start <= start and draft.char_end >= start + len(alias)
                for draft in carrier_drafts
            )
            for start in starts
        ):
            raise ValueError(f"carrier alias is not owned by a carrier-static binding: {alias!r}")


def capability_contract(
    feature: Mapping[str, Any], source_target: Mapping[str, Any]
) -> CapabilityContract:
    parties = source_target.get("documentPatch", {}).get("parties", {})
    roles = tuple(sorted(key for key, value in parties.items() if value not in (None, [], {})))
    return CapabilityContract.model_validate(
        {
            "document_type": feature["document_type"],
            "template_proxy_id": feature["template_proxy_id"],
            "page_count": feature["page_count"],
            "line_count": feature["ocr_lines"],
            "character_count": feature["ocr_characters"],
            "container_count": feature["container_count"],
            "seal_count": feature["seal_count"],
            "cargo_group_count": feature["goods_group_count"],
            "package_count": feature["package_fact_count"],
            "allocation_group_count": feature["allocation_group_count"],
            "allocation_row_count": feature["allocation_row_count"],
            "dangerous_goods_count": feature["dangerous_goods_count"],
            "temperature_count": feature["temperature_setting_count"],
            "additional_information_count": feature["additional_information_group_count"],
            "marks_count": feature["marks_group_count"],
            "party_roles": roles,
            "target_leaf_paths": tuple(
                sorted("documentPatch." + path for path in feature["target_leaf_paths"])
            ),
        }
    )


def _template_slots(raw: str, drafts: Sequence[SpanDraft]) -> tuple[TemplateSlot, ...]:
    slots: list[TemplateSlot] = []
    for index, draft in enumerate(drafts, start=1):
        byte_start = len(raw[: draft.char_start].encode("utf-8"))
        byte_end = len(raw[: draft.char_end].encode("utf-8"))
        slots.append(
            build_template_slot(
                slot_id=f"slot_{index:04d}",
                byte_start=byte_start,
                byte_end=byte_end,
                source_text=draft.source_text,
                target_paths=draft.target_paths,
                semantic_role=draft.group_key,
                evidence_origin=draft.evidence_origin,
                render_policy=draft.render_policy,
            )
        )
    return tuple(slots)


@lru_cache(maxsize=4096)
def _surface_token_spans(value: str) -> tuple[tuple[str, int, int], ...]:
    """Return Unicode alphanumeric tokens with offsets into the original surface."""

    spans: list[tuple[str, int, int]] = []
    start: int | None = None
    for index, character in enumerate(value):
        if character.isalnum():
            if start is None:
                start = index
        elif start is not None:
            spans.append((value[start:index].casefold(), start, index))
            start = None
    if start is not None:
        spans.append((value[start:].casefold(), start, len(value)))
    return tuple(spans)


def _normalized_surface(value: str) -> str:
    return "".join(token for token, _start, _end in _surface_token_spans(value))


def _scalar_surface(value: Any) -> str | None:
    if isinstance(value, bool) or value is None:
        return None
    if isinstance(value, str):
        return value
    if isinstance(value, (int, float, Decimal)):
        return str(value)
    return None


def _package_category_surface(value: Any) -> str | None:
    if not isinstance(value, str) or not value.startswith("PACKAGE_"):
        return None
    return value.removeprefix("PACKAGE_").replace("_", " ")


def _package_inflections(value: str) -> frozenset[str]:
    normalized = _normalized_surface(value)
    variants = {normalized}
    if normalized.endswith("y") and len(normalized) > 1:
        variants.add(normalized[:-1] + "ies")
    elif normalized.endswith(("s", "x", "z", "ch", "sh")):
        variants.add(normalized + "es")
    else:
        variants.add(normalized + "s")
    return frozenset(variants)


def _package_component_span_candidates(
    source: str,
    *,
    quantity: Any,
    category: Any,
) -> tuple[tuple[tuple[int, int], tuple[int, int]], ...]:
    """Return token-proven quantity/category pairs from a source surface."""

    category_surface = _package_category_surface(category)
    if (
        isinstance(quantity, bool)
        or not isinstance(quantity, (int, float, Decimal))
        or category_surface is None
    ):
        return ()
    target_quantity = Decimal(str(quantity))
    tokens = _surface_token_spans(source)
    accepted_categories = _package_inflections(category_surface)
    candidates: list[tuple[tuple[int, int], tuple[int, int]]] = []
    for numeric in re.finditer(r"(?<![0-9])[-+]?[0-9][0-9., ]*(?![0-9])", source):
        numeric_surface = numeric.group(0).rstrip()
        quantity_span = (numeric.start(), numeric.start() + len(numeric_surface))
        if target_quantity not in _decimal_candidates(numeric_surface, integer=True):
            continue
        after = tuple(
            (token, start, end) for token, start, end in tokens if start >= quantity_span[1]
        )
        if after and after[0][0] == "x":
            after = after[1:]
        matches = tuple(
            (after[0][1], after[end_index - 1][2])
            for end_index in range(1, min(3, len(after)) + 1)
            if "".join(token for token, _start, _end in after[:end_index]) in accepted_categories
        )
        if matches:
            category_span = max(matches, key=lambda span: span[1] - span[0])
            category_end = category_span[1]
            while (
                category_end < len(source)
                and source[category_end] == ")"
                and source[category_span[0] : category_end].count("(")
                > source[category_span[0] : category_end].count(")")
            ):
                category_end += 1
            candidates.append((quantity_span, (category_span[0], category_end)))
    return tuple(dict.fromkeys(candidates))


def _package_component_spans(
    source: str,
    *,
    quantity: Any,
    category: Any,
) -> tuple[tuple[int, int], tuple[int, int]] | None:
    """Prove disjoint leaf spans in one exhaustive package-count surface.

    A quantity leaf and a package-category leaf are independently mutable target facts.  This
    parser accepts only an exhaustive ``quantity [x] category`` surface, including document-native
    plural notation such as ``BOX(ES)``.  It deliberately returns the two leaf spans instead of
    inventing a collection-count derivation.
    """

    candidates = _package_component_span_candidates(
        source,
        quantity=quantity,
        category=category,
    )
    if len(candidates) != 1:
        return None
    quantity_span, category_span = candidates[0]
    allowed = (
        *range(quantity_span[0], quantity_span[1]),
        *range(category_span[0], category_span[1]),
    )
    allowed_positions = frozenset(allowed)
    outside = "".join(
        character
        for index, character in enumerate(source)
        if index not in allowed_positions and character.casefold() != "x"
    )
    if any(character.isalnum() for character in outside):
        return None
    return quantity_span, category_span


_MEASUREMENT_UNIT_SURFACES: dict[str, frozenset[str]] = {
    "kilogram": frozenset({"kg", "kgs", "kilogram", "kilograms", "kilo", "kilos"}),
    "metric_tonne": frozenset(
        {"mt", "mts", "metricton", "metrictons", "metrictonne", "metrictonnes", "tonne", "tonnes"}
    ),
    "pound": frozenset({"lb", "lbs", "pound", "pounds"}),
    "cubic_metre": frozenset(
        {"m3", "cbm", "cubicmeter", "cubicmeters", "cubicmetre", "cubicmetres"}
    ),
    "celsius": frozenset({"c", "degc", "degreec", "celsius"}),
}


def _date_candidates(value: str) -> frozenset[date]:
    cleaned = " ".join(value.replace(",", " ").split())
    formats = (
        "%Y-%m-%d",
        "%Y/%m/%d",
        "%Y.%m.%d",
        "%d-%m-%Y",
        "%d/%m/%Y",
        "%d.%m.%Y",
        "%d-%m-%y",
        "%d/%m/%y",
        "%d.%m.%y",
        "%d %b %Y",
        "%d %B %Y",
        "%d-%b-%Y",
        "%d-%B-%Y",
        "%b %d %Y",
        "%B %d %Y",
    )
    parsed: set[date] = set()
    for candidate_format in formats:
        try:
            parsed.add(datetime.strptime(cleaned, candidate_format).date())
        except ValueError:
            continue
    return frozenset(parsed)


def _decimal_candidates(value: str, *, integer: bool) -> frozenset[Decimal]:
    compact = value.strip().replace(" ", "")
    candidates: set[Decimal] = set()
    variants = {compact}
    if integer:
        variants.add(compact.replace(",", "").replace(".", ""))
    else:
        variants.add(compact.replace(",", ""))
        if compact.count(",") == 1 and "." not in compact:
            variants.add(compact.replace(",", "."))
        if compact.count(".") == 1 and "," in compact:
            variants.add(compact.replace(".", "").replace(",", "."))
    for variant in variants:
        try:
            parsed = Decimal(variant)
        except InvalidOperation:
            continue
        if parsed.is_finite() and (not integer or parsed == parsed.to_integral_value()):
            candidates.add(parsed)
    return frozenset(candidates)


def _matching_numeric_span(source: str, target: Any, *, integer: bool) -> tuple[int, int] | None:
    if isinstance(target, bool) or not isinstance(target, (int, float, Decimal)):
        return None
    target_decimal = Decimal(str(target))
    matches: list[tuple[int, int]] = []
    for match in re.finditer(r"(?<![0-9])[-+]?[0-9][0-9., ]*(?![0-9])", source):
        candidate = match.group(0).rstrip()
        end = match.start() + len(candidate)
        if target_decimal in _decimal_candidates(candidate, integer=integer):
            matches.append((match.start(), end))
    return matches[0] if len(matches) == 1 else None


def _matching_date_span(source: str, target: Any) -> tuple[int, int] | None:
    if not isinstance(target, str):
        return None
    target_dates = _date_candidates(target)
    if not target_dates:
        return None
    matches = [
        (match.start(), match.end())
        for match in _DATE.finditer(source)
        if target_dates & _date_candidates(match.group(0))
    ]
    if len(matches) == 1:
        return matches[0]
    if target_dates & _date_candidates(source):
        return (0, len(source))
    return None


def _matching_token_frame(source: str, target_surface: str) -> tuple[int, int] | None:
    source_tokens = _surface_token_spans(source)
    target_tokens = tuple(token for token, _start, _end in _surface_token_spans(target_surface))
    if not target_tokens or len(target_tokens) > len(source_tokens):
        return None
    matches: list[tuple[int, int]] = []
    for index in range(len(source_tokens) - len(target_tokens) + 1):
        window = source_tokens[index : index + len(target_tokens)]
        if tuple(token for token, _start, _end in window) == target_tokens:
            matches.append((window[0][1], window[-1][2]))
    return matches[0] if len(matches) == 1 else None


def _matching_token_projection(
    source: str, target_surface: str
) -> tuple[tuple[str, ...], tuple[str, ...]] | None:
    """Locate one exact source token sequence inside a target scalar.

    The returned omitted target prefix and suffix are explicit compatibility constraints for a
    descendant value. This converts abbreviated repeated surfaces such as ``13672297`` beside a
    full ``MATERIAL 13672297`` occurrence into a deterministic contract without assuming that an
    arbitrary future prefix or suffix can be discarded.
    """

    source_tokens = tuple(token for token, _start, _end in _surface_token_spans(source))
    target_tokens = tuple(token for token, _start, _end in _surface_token_spans(target_surface))
    if not source_tokens or len(source_tokens) > len(target_tokens):
        return None
    matches = tuple(
        index
        for index in range(len(target_tokens) - len(source_tokens) + 1)
        if target_tokens[index : index + len(source_tokens)] == source_tokens
    )
    if len(matches) != 1:
        return None
    start = matches[0]
    return target_tokens[:start], target_tokens[start + len(source_tokens) :]


def _matching_normalized_projection(source: str, target_surface: str) -> tuple[str, str] | None:
    """Locate one normalized source surface inside a normalized target scalar."""

    source_normalized = _normalized_surface(source)
    target_normalized = _normalized_surface(target_surface)
    if not source_normalized or len(source_normalized) > len(target_normalized):
        return None
    starts = tuple(
        match.start() for match in re.finditer(re.escape(source_normalized), target_normalized)
    )
    if len(starts) != 1:
        return None
    start = starts[0]
    return (
        target_normalized[:start],
        target_normalized[start + len(source_normalized) :],
    )


def _surface_match(
    *, source: str, target: Any, adapter: str, value_kind: str
) -> tuple[str, str] | None:
    """Prove a source slot is one formatted realization of a target scalar."""

    target_surface = (
        _package_category_surface(target)
        if adapter == "package_category"
        else _scalar_surface(target)
    )
    if target_surface is None:
        return None
    if adapter == "package_category":
        if _normalized_surface(source) in _package_inflections(target_surface):
            return "", ""
        return None
    if adapter == "measurement_unit":
        if not isinstance(target, str):
            return None
        accepted = _MEASUREMENT_UNIT_SURFACES.get(target)
        if accepted is not None and _normalized_surface(source) in accepted:
            return "", ""
        return None
    if adapter == "date":
        span = _matching_date_span(source, target)
        return (source[: span[0]], source[span[1] :]) if span is not None else None
    if adapter == "numeric":
        span = _matching_numeric_span(source, target, integer=value_kind == "integer")
        return (source[: span[0]], source[span[1] :]) if span is not None else None
    if adapter == "signed_temperature_word":
        match = re.fullmatch(r"(?i)(PLUS|MINUS)\s+(\d+(?:[.,]\d+)?)", source)
        if match is None:
            return None
        observed = Decimal(match[2].replace(",", "."))
        if match[1].upper() == "MINUS":
            observed = -observed
        return ("", "") if observed == Decimal(str(target)) else None
    if _normalized_surface(source) == _normalized_surface(target_surface):
        return "", ""
    span = _matching_token_frame(source, target_surface)
    return (source[: span[0]], source[span[1] :]) if span is not None else None


def _surface_adapter(draft: SpanDraft) -> str:
    if draft.value_kind == "temperature" and re.fullmatch(
        r"(?i)(?:PLUS|MINUS)\s+\d+(?:[.,]\d+)?", draft.source_text
    ):
        return "signed_temperature_word"
    if any(
        path.endswith(".typeDescription") and ".containers[" in path for path in draft.target_paths
    ):
        return "equipment_type"
    if any(
        path.endswith(".typeCategory") and ".cargoPackages[" in path for path in draft.target_paths
    ):
        return "package_category"
    if any(path.endswith(".unit") for path in draft.target_paths):
        return "measurement_unit"
    return {
        "opaque_identifier": "opaque_identifier",
        "date_surface": "date",
        "numeric_surface": "numeric",
        "categorical_surface": "categorical",
        "natural_text": "natural_text",
    }.get(draft.render_policy, "natural_text")


def target_binding_surface_analysis(
    *, draft: SpanDraft, source_target: Mapping[str, Any]
) -> TargetBindingSurfaceAnalysis | None:
    """Describe a direct target surface without assigning semantics to surrounding text.

    This exposes the same mechanically proved formatting and token-projection relationships used
    by the realization compiler.  It intentionally returns no result for unequal multi-path
    values or non-target bindings; those cases require their existing semantic contracts.
    """

    if draft.render_mode != "target_binding" or not draft.target_paths:
        return None
    target_values = tuple(_resolve_target_path(source_target, path) for path in draft.target_paths)
    if len({canonical_json_bytes(value) for value in target_values}) != 1:
        return None
    target_value = target_values[0]
    adapter = _surface_adapter(draft)
    surface_match = _surface_match(
        source=draft.source_text,
        target=target_value,
        adapter=adapter,
        value_kind=draft.value_kind,
    )
    target_surface = (
        _package_category_surface(target_value)
        if adapter == "package_category"
        else _scalar_surface(target_value)
    )
    token_projection = (
        _matching_token_projection(draft.source_text, target_surface)
        if surface_match is None and adapter == "natural_text" and target_surface is not None
        else None
    )
    return TargetBindingSurfaceAnalysis(
        literal_prefix=surface_match[0] if surface_match is not None else "",
        literal_suffix=surface_match[1] if surface_match is not None else "",
        omitted_target_prefix_tokens=token_projection[0] if token_projection is not None else (),
        omitted_target_suffix_tokens=token_projection[1] if token_projection is not None else (),
    )


def _slot_realization(
    slot: TemplateSlot,
    *,
    value_role: str,
    segment_index: int | None = None,
    literal_prefix: str = "",
    literal_suffix: str = "",
    required_target_prefix_tokens: tuple[str, ...] = (),
    required_target_suffix_tokens: tuple[str, ...] = (),
    required_target_prefix_normalized: str = "",
    required_target_suffix_normalized: str = "",
) -> SlotRealization:
    return SlotRealization.model_validate(
        {
            "slot_id": slot.slot_id,
            "value_role": value_role,
            "segment_index": segment_index,
            "source_token_count": len(_surface_token_spans(slot.source_text)),
            "literal_prefix": literal_prefix,
            "literal_suffix": literal_suffix,
            "required_target_prefix_tokens": required_target_prefix_tokens,
            "required_target_suffix_tokens": required_target_suffix_tokens,
            "required_target_prefix_normalized": required_target_prefix_normalized,
            "required_target_suffix_normalized": required_target_suffix_normalized,
        }
    )


def _target_snapshots(
    source_target: Mapping[str, Any], target_paths: Sequence[str]
) -> tuple[TargetValueSnapshot, ...]:
    return tuple(
        TargetValueSnapshot.model_validate(
            {"target_path": path, "source_value": _resolve_target_path(source_target, path)}
        )
        for path in target_paths
    )


def _agent_realization(
    *, slots: Sequence[TemplateSlot], snapshots: tuple[TargetValueSnapshot, ...], rationale: str
) -> BindingRealization:
    return BindingRealization.model_validate(
        {
            "mode": "agent_required",
            "adapter": "agent",
            "deterministic": False,
            "requires_agent": True,
            "target_values": snapshots,
            "slots": tuple(_slot_realization(slot, value_role="agent") for slot in slots),
            "rationale": rationale,
        }
    )


def binding_realization(
    *,
    draft: SpanDraft,
    slots: Sequence[TemplateSlot],
    source_target: Mapping[str, Any],
) -> BindingRealization:
    """Compile an explicit, fail-closed plan for every physical binding surface."""

    snapshots = _target_snapshots(source_target, draft.target_paths)
    if draft.render_mode in {"carrier_static", "literal_static"}:
        return BindingRealization.model_validate(
            {
                "mode": "static",
                "adapter": "static",
                "deterministic": True,
                "requires_agent": False,
                "target_values": snapshots,
                "slots": tuple(_slot_realization(slot, value_role="static") for slot in slots),
                "rationale": (
                    "Carrier-bound or literal source text remains unchanged across renders."
                ),
            }
        )
    if draft.render_mode == "deterministic_derived":
        return BindingRealization.model_validate(
            {
                "mode": "deterministic_derivation",
                "adapter": "deterministic_derivation",
                "deterministic": True,
                "requires_agent": False,
                "target_values": snapshots,
                "slots": tuple(_slot_realization(slot, value_role="derived") for slot in slots),
                "rationale": "The declared derivation and dependencies determine this surface.",
            }
        )
    if draft.render_mode == "agent_residual":
        return _agent_realization(
            slots=slots,
            snapshots=snapshots,
            rationale="The compiler explicitly classified this surface as requiring agent editing.",
        )
    if draft.render_mode == "deterministic_auxiliary":
        normalized = {_normalized_surface(slot.source_text) for slot in slots}
        if len(slots) > 1 and ("" in normalized or len(normalized) != 1):
            return _agent_realization(
                slots=slots,
                snapshots=snapshots,
                rationale=(
                    "Repeated source-only occurrences are not equivalent, so one deterministic "
                    "auxiliary value cannot safely realize every slot."
                ),
            )
        return BindingRealization.model_validate(
            {
                "mode": "generated_auxiliary",
                "adapter": "generated_auxiliary",
                "deterministic": True,
                "requires_agent": False,
                "target_values": snapshots,
                "slots": tuple(_slot_realization(slot, value_role="generated") for slot in slots),
                "rationale": (
                    "A typed auxiliary generator supplies one value; equivalent repeated slots "
                    "share it."
                ),
            }
        )
    if draft.render_mode != "target_binding":
        raise ValueError(f"unsupported render mode in realization compiler: {draft.render_mode}")
    if not snapshots:
        raise ValueError("target binding lacks target snapshots")
    if len(slots) == 1:
        from .temperature_prose import composite_instruction_contract

        if (
            composite_instruction_contract(
                draft.target_paths,
                slots[0].source_text,
                {row.target_path: row.source_value for row in snapshots},
            )
            is not None
        ):
            return BindingRealization.model_validate(
                {
                    "mode": "single_surface",
                    "adapter": "temperature_instruction",
                    "deterministic": True,
                    "requires_agent": False,
                    "target_values": snapshots,
                    "slots": (_slot_realization(slots[0], value_role="whole"),),
                    "rationale": (
                        "The printed handling instruction jointly realizes its exact "
                        "temperature value and unit through a verified deterministic edit."
                    ),
                }
            )
    encoded_values = {canonical_json_bytes(row.source_value) for row in snapshots}
    if len(encoded_values) != 1:
        return _agent_realization(
            slots=slots,
            snapshots=snapshots,
            rationale=(
                "One physical surface maps to unequal target values and needs agent composition."
            ),
        )
    target_value = snapshots[0].source_value
    adapter = _surface_adapter(draft)
    matches = tuple(
        _surface_match(
            source=slot.source_text,
            target=target_value,
            adapter=adapter,
            value_kind=draft.value_kind,
        )
        for slot in slots
    )
    if all(match is not None for match in matches):
        role = "whole" if len(slots) == 1 else "repeat"
        mode = "single_surface" if len(slots) == 1 else "repeated_surface"
        return BindingRealization.model_validate(
            {
                "mode": mode,
                "adapter": adapter,
                "deterministic": True,
                "requires_agent": False,
                "target_values": snapshots,
                "slots": tuple(
                    _slot_realization(
                        slot,
                        value_role=role,
                        literal_prefix=match[0],
                        literal_suffix=match[1],
                    )
                    for slot, match in zip(slots, matches, strict=True)
                    if match is not None
                ),
                "rationale": (
                    "Each physical slot is a host-proven formatted realization of the same "
                    "target scalar."
                ),
            }
        )
    target_surface = (
        _package_category_surface(target_value)
        if adapter == "package_category"
        else _scalar_surface(target_value)
    )
    if len(slots) > 1 and target_surface is not None:
        slot_norms = tuple(_normalized_surface(slot.source_text) for slot in slots)
        if all(slot_norms) and "".join(slot_norms) == _normalized_surface(target_surface):
            return BindingRealization.model_validate(
                {
                    "mode": "segmented_surface",
                    "adapter": adapter,
                    "deterministic": True,
                    "requires_agent": False,
                    "target_values": snapshots,
                    "slots": tuple(
                        _slot_realization(slot, value_role="segment", segment_index=index)
                        for index, slot in enumerate(slots)
                    ),
                    "rationale": (
                        "Ordered source segments concatenate exactly to the normalized target; "
                        "source token counts define the deterministic layout partition."
                    ),
                }
            )
    if adapter == "natural_text" and target_surface is not None:
        projections = tuple(
            _matching_token_projection(slot.source_text, target_surface) for slot in slots
        )
        if all(projection is not None for projection in projections) and any(
            projection != ((), ()) for projection in projections
        ):
            return BindingRealization.model_validate(
                {
                    "mode": "token_projected_surface",
                    "adapter": adapter,
                    "deterministic": True,
                    "requires_agent": False,
                    "target_values": snapshots,
                    "slots": tuple(
                        _slot_realization(
                            slot,
                            value_role="token_projection",
                            required_target_prefix_tokens=projection[0],
                            required_target_suffix_tokens=projection[1],
                        )
                        for slot, projection in zip(slots, projections, strict=True)
                        if projection is not None
                    ),
                    "rationale": (
                        "Every physical slot is a host-proven contiguous token projection of the "
                        "same target scalar; exact omitted tokens are descendant compatibility "
                        "constraints."
                    ),
                }
            )
    if adapter == "equipment_type" and target_surface is not None:
        normalized_projections = tuple(
            _matching_normalized_projection(slot.source_text, target_surface) for slot in slots
        )
        if all(projection is not None for projection in normalized_projections) and any(
            projection != ("", "") for projection in normalized_projections
        ):
            return BindingRealization.model_validate(
                {
                    "mode": "normalized_projected_surface",
                    "adapter": adapter,
                    "deterministic": True,
                    "requires_agent": False,
                    "target_values": snapshots,
                    "slots": tuple(
                        _slot_realization(
                            slot,
                            value_role="normalized_projection",
                            required_target_prefix_normalized=projection[0],
                            required_target_suffix_normalized=projection[1],
                        )
                        for slot, projection in zip(slots, normalized_projections, strict=True)
                        if projection is not None
                    ),
                    "rationale": (
                        "Every equipment slot is a unique normalized projection of the same "
                        "target description; omitted normalized fragments and the exact slot "
                        "format envelope define descendant compatibility and rendering."
                    ),
                }
            )
    return _agent_realization(
        slots=slots,
        snapshots=snapshots,
        rationale=(
            "The host cannot prove a deterministic scalar, repeated, or segmented mapping from "
            "the source surfaces to the target value."
        ),
    )


def _projected_equipment_type_paths(
    source_text: str, source_target: Mapping[str, Any]
) -> tuple[str, ...]:
    patch = source_target.get("documentPatch")
    if not isinstance(patch, Mapping):
        raise ValueError("source target lacks documentPatch")
    containers = patch.get("containers", [])
    if not isinstance(containers, list):
        raise ValueError("documentPatch.containers must be a list")
    target_surfaces: list[tuple[str, str]] = []
    for index, container in enumerate(containers):
        if not isinstance(container, Mapping):
            raise ValueError(f"documentPatch.containers[{index}] must be an object")
        description = container.get("typeDescription")
        if description is None:
            continue
        if not isinstance(description, str) or not description:
            raise ValueError(f"documentPatch.containers[{index}].typeDescription is invalid")
        target_surfaces.append(
            (
                f"documentPatch.containers[{index}].typeDescription",
                _normalized_surface(description),
            )
        )
    tokens = tuple(token for token, _start, _end in _surface_token_spans(source_text))
    candidates: set[str] = set()
    for start in range(len(tokens)):
        for width in range(1, min(3, len(tokens) - start) + 1):
            candidate = "".join(tokens[start : start + width])
            if not 4 <= len(candidate) <= 12:
                continue
            if not any(character.isalpha() for character in candidate):
                continue
            if not {character.upper() for character in candidate if character.isalpha()} - {"X"}:
                # OCR dimensions such as 20'X8'6" are common to unrelated equipment types.
                # The separator X alone is not semantic evidence for GENERAL PURPOSE vs OPEN TOP.
                continue
            if not any(character.isdigit() for character in candidate):
                continue
            candidates.add(candidate)
    return tuple(
        path
        for path, normalized_target in target_surfaces
        if any(candidate in normalized_target for candidate in candidates)
    )


def _is_explicit_missing_target_equipment_auxiliary(
    draft: SpanDraft, source_target: Mapping[str, Any]
) -> bool:
    if (
        draft.render_mode != "deterministic_auxiliary"
        or draft.group_kind != "equipment"
        or draft.value_kind != "equipment"
        or draft.target_paths
    ):
        return False
    match = re.fullmatch(r"container:([0-9]+)", draft.group_key)
    if match is None:
        return False
    patch = source_target.get("documentPatch")
    containers = patch.get("containers") if isinstance(patch, Mapping) else None
    index = int(match.group(1))
    if not isinstance(containers, list) or index >= len(containers):
        return False
    container = containers[index]
    return isinstance(container, Mapping) and "typeDescription" not in container


def _container_index_for_path(paths: Sequence[str], pattern: re.Pattern[str]) -> int | None:
    indexes = {
        int(match.group(1)) for path in paths if (match := pattern.fullmatch(path)) is not None
    }
    return next(iter(indexes)) if len(indexes) == 1 else None


def _short_equipment_token(draft: SpanDraft) -> bool:
    source = draft.source_text.strip()
    normalized = _normalized_surface(draft.source_text)
    return (
        not any(character.isspace() for character in source)
        and 4 <= len(normalized) <= 12
        and any(character.isalpha() for character in normalized)
        and any(character.isdigit() for character in normalized)
    )


def _nearest_container_number_index(
    *,
    raw: str,
    draft: SpanDraft,
    number_drafts: Mapping[int, Sequence[SpanDraft]],
    page_spans: Mapping[int, tuple[int, int]],
    lines: Sequence[LineSpan],
) -> int | None:
    """Return a uniquely local container row, never merely the nearest row on a page.

    OCR layouts commonly print a complete container-number list followed by cargo blocks.  Raw
    character distance across that layout is not row evidence: it incorrectly attaches the first
    cargo-block type to the last number in the preceding list.  A row association is mechanically
    defensible only when both spans share a source line, or when no alphanumeric content separates
    them (the common ``number\n40HQ`` and ``/40HQ/\n\nnumber`` layouts).
    """

    page_number = next(
        (
            number
            for number, (start, end) in page_spans.items()
            if start <= draft.char_start and draft.char_end <= end
        ),
        None,
    )
    if page_number is None:
        return None
    page_start, page_end = page_spans[page_number]
    draft_line_start, draft_line_end = line_range_for_chars(lines, draft.char_start, draft.char_end)
    draft_line_numbers = set(range(int(draft_line_start[1:]), int(draft_line_end[1:]) + 1))

    def distance(number: SpanDraft) -> int:
        if number.char_end <= draft.char_start:
            return draft.char_start - number.char_end
        if draft.char_end <= number.char_start:
            return number.char_start - draft.char_end
        return 0

    same_line: list[tuple[int, int]] = []
    separator_adjacent: list[tuple[int, int]] = []
    for index, numbers in number_drafts.items():
        for number in numbers:
            if not (page_start <= number.char_start and number.char_end <= page_end):
                continue
            number_line_start, number_line_end = line_range_for_chars(
                lines, number.char_start, number.char_end
            )
            number_line_numbers = set(
                range(int(number_line_start[1:]), int(number_line_end[1:]) + 1)
            )
            candidate = (distance(number), index)
            if draft_line_numbers & number_line_numbers:
                same_line.append(candidate)
                continue
            between_start = min(draft.char_end, number.char_end)
            between_end = max(draft.char_start, number.char_start)
            if between_start <= between_end and not any(
                character.isalnum() for character in raw[between_start:between_end]
            ):
                separator_adjacent.append(candidate)

    local_candidates = same_line or separator_adjacent
    distance_by_index: dict[int, int] = {}
    for candidate_distance, index in local_candidates:
        distance_by_index[index] = min(
            candidate_distance,
            distance_by_index.get(index, candidate_distance),
        )
    distances = sorted(
        (candidate_distance, index) for index, candidate_distance in distance_by_index.items()
    )
    if not distances or (len(distances) > 1 and distances[0][0] == distances[1][0]):
        return None
    return distances[0][1]


def _adjacent_equipment_receipt_owner(
    *,
    raw: str,
    char_start: int,
    char_end: int,
    source_text: str,
    drafts: Sequence[SpanDraft],
    source_target: Mapping[str, Any],
    page_spans: Mapping[int, tuple[int, int]],
) -> SpanDraft | None:
    """Resolve a compact type projection to one physically adjacent equipment owner.

    The relationship is accepted only when the compact token projects to an explicit equipment
    target and no alphanumeric content separates it from either a direct type owner or a derived
    receipt owner on the same page. This covers the common ``/.../40HQ/\n1X40'HQ CONTAINER``
    layout without inferring ownership from document order or equal type values elsewhere.
    """

    projected_paths = set(_projected_equipment_type_paths(source_text, source_target))
    if not projected_paths:
        return None
    token_page = next(
        (
            page_number
            for page_number, (page_start, page_end) in page_spans.items()
            if page_start <= char_start and char_end <= page_end
        ),
        None,
    )
    if token_page is None:
        return None

    grouped: dict[str, list[SpanDraft]] = defaultdict(list)
    for row in drafts:
        derived_receipt = (
            row.render_mode == "deterministic_derived"
            and row.derivation == "equipment_receipt"
            and row.group_kind == "equipment"
            and projected_paths.intersection(row.dependency_paths)
        )
        direct_type = (
            row.render_mode == "target_binding"
            and row.group_kind == "equipment"
            and projected_paths.intersection(row.target_paths)
            and any(_CONTAINER_TYPE_PATH.fullmatch(path) is not None for path in row.target_paths)
        )
        if derived_receipt or direct_type:
            grouped[row.logical_key].append(row)

    candidates: list[tuple[int, str, SpanDraft]] = []
    page_start, page_end = page_spans[token_page]
    for logical_key, rows in grouped.items():
        distances: list[int] = []
        for row in rows:
            if not (page_start <= row.char_start and row.char_end <= page_end):
                continue
            between_start = min(char_end, row.char_end)
            between_end = max(char_start, row.char_start)
            if between_start > between_end or any(
                character.isalnum() for character in raw[between_start:between_end]
            ):
                continue
            distances.append(between_end - between_start)
        if distances:
            candidates.append((min(distances), logical_key, rows[0]))
    candidates.sort(key=lambda row: (row[0], row[1]))
    if not candidates or (len(candidates) > 1 and candidates[0][0] == candidates[1][0]):
        return None
    return candidates[0][2]


def _host_discovered_compact_equipment(
    *,
    raw: str,
    drafts: Sequence[SpanDraft],
    source_target: Mapping[str, Any],
    page_spans: Mapping[int, tuple[int, int]],
    lines: Sequence[LineSpan],
) -> tuple[SpanDraft, ...]:
    """Materialize only mechanically localized, currently unowned equipment projections."""

    number_drafts: dict[int, list[SpanDraft]] = defaultdict(list)
    for draft in drafts:
        index = _container_index_for_path(draft.target_paths, _CONTAINER_NUMBER_PATH)
        if index is not None:
            number_drafts[index].append(draft)
    patch = source_target.get("documentPatch")
    containers = patch.get("containers") if isinstance(patch, Mapping) else None
    if containers is None:
        return ()
    if not isinstance(containers, list):
        raise ValueError("documentPatch.containers must be a list")

    direct_owners: dict[str, dict[str, SpanDraft]] = defaultdict(dict)
    for draft in drafts:
        if draft.render_mode != "target_binding":
            continue
        for path in draft.target_paths:
            if _CONTAINER_TYPE_PATH.fullmatch(path) is not None:
                direct_owners[path].setdefault(draft.logical_key, draft)

    output: list[SpanDraft] = []
    for token, char_start, char_end in _surface_token_spans(raw):
        if (
            not 4 <= len(token) <= 12
            or not any(character.isalpha() for character in token)
            or not any(character.isdigit() for character in token)
            or any(char_start < draft.char_end and draft.char_start < char_end for draft in drafts)
        ):
            continue
        source_text = raw[char_start:char_end]
        projected_paths = set(_projected_equipment_type_paths(source_text, source_target))
        if not projected_paths:
            continue
        receipt_owner = _adjacent_equipment_receipt_owner(
            raw=raw,
            char_start=char_start,
            char_end=char_end,
            source_text=source_text,
            drafts=drafts,
            source_target=source_target,
            page_spans=page_spans,
        )
        if receipt_owner is not None:
            output.append(
                replace(
                    receipt_owner,
                    draft_id=(
                        "host_equipment_receipt_projection_"
                        + sha256_bytes(f"{char_start}:{char_end}".encode())[:16]
                    ),
                    char_start=char_start,
                    char_end=char_end,
                    source_text=source_text,
                    evidence_origin="derived_operational_fact",
                    rationale=(
                        "Host attached this unowned compact equipment projection to the unique "
                        "physically adjacent direct or derived equipment owner."
                    ),
                )
            )
            continue

        probe = SpanDraft(
            draft_id="host_equipment_locality_probe",
            logical_key="host:equipment_locality_probe",
            render_mode="deterministic_auxiliary",
            value_kind="equipment",
            group_kind="equipment",
            group_key="host:equipment_locality_probe",
            target_paths=(),
            derivation=None,
            dependency_paths=(),
            dependency_bindings=(),
            char_start=char_start,
            char_end=char_end,
            source_text=source_text,
            evidence_origin="derived_operational_fact",
            render_policy="opaque_identifier",
            rationale="Host equipment locality probe.",
        )
        nearest_index = _nearest_container_number_index(
            raw=raw,
            draft=probe,
            number_drafts=number_drafts,
            page_spans=page_spans,
            lines=lines,
        )
        if nearest_index is None or nearest_index >= len(containers):
            continue
        container = containers[nearest_index]
        if not isinstance(container, Mapping):
            raise ValueError(f"documentPatch.containers[{nearest_index}] must be an object")
        target_path = f"documentPatch.containers[{nearest_index}].typeDescription"
        if "typeDescription" not in container:
            output.append(
                replace(
                    probe,
                    draft_id=(
                        "host_equipment_auxiliary_"
                        + sha256_bytes(f"{nearest_index}:{char_start}:{char_end}".encode())[:16]
                    ),
                    logical_key=f"agent:host_localized_container_{nearest_index}_type_token",
                    group_key=f"container:{nearest_index}",
                    rationale=(
                        "Host localized this unowned compact equipment token to the unique "
                        f"container:{nearest_index} row; that structured container has no "
                        "typeDescription field."
                    ),
                )
            )
            continue
        if target_path not in projected_paths:
            continue
        owners = direct_owners.get(target_path, {})
        if len(owners) > 1:
            raise ValueError(f"equipment type target has multiple direct owners: {target_path}")
        if owners:
            owner = next(iter(owners.values()))
            output.append(
                replace(
                    owner,
                    draft_id=(
                        "host_equipment_target_projection_"
                        + sha256_bytes(f"{nearest_index}:{char_start}:{char_end}".encode())[:16]
                    ),
                    char_start=char_start,
                    char_end=char_end,
                    source_text=source_text,
                    evidence_origin="derived_operational_fact",
                    rationale=(
                        "Host appended this unowned compact equipment projection to the unique "
                        f"existing owner of {target_path}."
                    ),
                )
            )
            continue
        group_kind, group_key = _canonical_target_group((target_path,))
        output.append(
            replace(
                probe,
                draft_id=(
                    "host_equipment_target_projection_"
                    + sha256_bytes(f"{nearest_index}:{char_start}:{char_end}".encode())[:16]
                ),
                logical_key="anchor:" + target_path,
                render_mode="target_binding",
                group_kind=group_kind,
                group_key=group_key,
                target_paths=(target_path,),
                rationale=(
                    "Host target-bound this unowned compact equipment projection to the unique "
                    f"local container:{nearest_index} typeDescription."
                ),
            )
        )
    return tuple(output)


def _localized_equipment_target_draft(*, draft: SpanDraft, container_index: int) -> SpanDraft:
    target_paths = (f"documentPatch.containers[{container_index}].typeDescription",)
    group_kind, group_key = _canonical_target_group(target_paths)
    logical_key = (
        "anchor:" + target_paths[0]
        if draft.render_mode in {"target_binding", "carrier_static"}
        else f"agent:host_localized_container_{container_index}_type_description"
    )
    return replace(
        draft,
        logical_key=logical_key,
        group_kind=group_kind,
        group_key=group_key,
        target_paths=target_paths,
        rationale=(
            draft.rationale
            + " Host-localized this compact equipment token to the source row containing the "
            f"container:{container_index} number."
        ),
    )


def normalize_compact_equipment_locality(
    *, raw: str, drafts: Sequence[SpanDraft], source_target: Mapping[str, Any]
) -> tuple[SpanDraft, ...]:
    """Canonicalize only host-proven compact equipment projections and row locality."""

    page_spans = page_body_spans(raw)
    lines = line_spans(raw)
    drafts = (
        *drafts,
        *_host_discovered_compact_equipment(
            raw=raw,
            drafts=drafts,
            source_target=source_target,
            page_spans=page_spans,
            lines=lines,
        ),
    )

    number_drafts: dict[int, list[SpanDraft]] = defaultdict(list)
    for draft in drafts:
        index = _container_index_for_path(draft.target_paths, _CONTAINER_NUMBER_PATH)
        if index is not None:
            number_drafts[index].append(draft)
    patch = source_target.get("documentPatch")
    containers = patch.get("containers") if isinstance(patch, Mapping) else None
    if containers is None:
        return tuple(drafts)
    if not isinstance(containers, list):
        raise ValueError("documentPatch.containers must be a list")
    output: list[SpanDraft] = []
    for draft in drafts:
        if (
            _short_equipment_token(draft)
            and draft.render_mode == "deterministic_auxiliary"
            and draft.value_kind == "equipment"
            and not draft.target_paths
        ):
            receipt_owner = _adjacent_equipment_receipt_owner(
                raw=raw,
                char_start=draft.char_start,
                char_end=draft.char_end,
                source_text=draft.source_text,
                drafts=drafts,
                source_target=source_target,
                page_spans=page_spans,
            )
            if receipt_owner is not None:
                if (
                    draft.group_kind != receipt_owner.group_kind
                    or draft.group_key != receipt_owner.group_key
                ):
                    raise ValueError(
                        f"{draft.logical_key} declares {draft.group_kind}:{draft.group_key} but "
                        "its compact equipment token is physically adjacent to "
                        f"{receipt_owner.logical_key} in {receipt_owner.group_kind}:"
                        f"{receipt_owner.group_key}"
                    )
                output.append(
                    replace(
                        draft,
                        logical_key=receipt_owner.logical_key,
                        render_mode=receipt_owner.render_mode,
                        value_kind=receipt_owner.value_kind,
                        group_kind=receipt_owner.group_kind,
                        group_key=receipt_owner.group_key,
                        target_paths=receipt_owner.target_paths,
                        derivation=receipt_owner.derivation,
                        dependency_paths=receipt_owner.dependency_paths,
                        dependency_bindings=receipt_owner.dependency_bindings,
                        evidence_origin="derived_operational_fact",
                        render_policy=receipt_owner.render_policy,
                        rationale=(
                            draft.rationale
                            + " Host canonicalized this compact projection as another occurrence "
                            f"of the unique adjacent receipt owner {receipt_owner.logical_key}."
                        ),
                    )
                )
                continue
            scoped_type_paths = {
                path
                for other in drafts
                if other.group_key == draft.group_key
                for path in other.target_paths
                if _CONTAINER_TYPE_PATH.fullmatch(path) is not None
            }
            scoped_target = next(iter(scoped_type_paths)) if len(scoped_type_paths) == 1 else None
            nearest_index = _nearest_container_number_index(
                raw=raw,
                draft=draft,
                number_drafts=number_drafts,
                page_spans=page_spans,
                lines=lines,
            )
            nearest_container = (
                containers[nearest_index]
                if nearest_index is not None and nearest_index < len(containers)
                else None
            )
            target_path = scoped_target or (
                f"documentPatch.containers[{nearest_index}].typeDescription"
                if nearest_index is not None
                else None
            )
            if (
                target_path is not None
                and (
                    scoped_target is not None
                    or (
                        isinstance(nearest_container, Mapping)
                        and "typeDescription" in nearest_container
                    )
                )
                and target_path in _projected_equipment_type_paths(draft.source_text, source_target)
            ):
                target_match = _CONTAINER_TYPE_PATH.fullmatch(target_path)
                if target_match is None:
                    raise AssertionError("localized equipment target path is invalid")
                resolved_index = int(target_match.group(1))
                group_kind, group_key = _canonical_target_group((target_path,))
                output.append(
                    replace(
                        draft,
                        logical_key="anchor:" + target_path,
                        render_mode="target_binding",
                        group_kind=group_kind,
                        group_key=group_key,
                        target_paths=(target_path,),
                        evidence_origin="host_verified_agent_proposal",
                        render_policy="opaque_identifier",
                        rationale=(
                            draft.rationale
                            + " Host-localized this source-only compact equipment token to the "
                            f"uniquely scoped container:{resolved_index} target."
                        ),
                    )
                )
                continue
        target_index = _container_index_for_path(draft.target_paths, _CONTAINER_TYPE_PATH)
        if target_index is None or not _short_equipment_token(draft):
            output.append(draft)
            continue
        nearest_index = _nearest_container_number_index(
            raw=raw,
            draft=draft,
            number_drafts=number_drafts,
            page_spans=page_spans,
            lines=lines,
        )
        if nearest_index is None or nearest_index == target_index:
            output.append(draft)
            continue
        nearest_container = containers[nearest_index] if nearest_index < len(containers) else None
        target_path = f"documentPatch.containers[{target_index}].typeDescription"
        siblings = tuple(
            row
            for row in drafts
            if row.draft_id != draft.draft_id
            and (
                target_path in row.target_paths
                or (
                    row.render_mode == "deterministic_derived"
                    and target_path in row.dependency_paths
                )
            )
        )
        target_has_stable_sibling = any(
            not _short_equipment_token(sibling)
            or _nearest_container_number_index(
                raw=raw,
                draft=sibling,
                number_drafts=number_drafts,
                page_spans=page_spans,
                lines=lines,
            )
            == target_index
            for sibling in siblings
        )
        if not isinstance(nearest_container, Mapping):
            output.append(draft)
            continue
        if "typeDescription" in nearest_container:
            output.append(
                _localized_equipment_target_draft(draft=draft, container_index=nearest_index)
            )
            continue
        if not target_has_stable_sibling:
            output.append(draft)
            continue
        output.append(
            SpanDraft(
                draft_id=(
                    "host_equipment_locality_"
                    + sha256_bytes(f"{nearest_index}:{draft.char_start}:{draft.char_end}".encode())[
                        :16
                    ]
                ),
                logical_key=f"agent:container_{nearest_index}_type_token",
                render_mode="deterministic_auxiliary",
                value_kind="equipment",
                group_kind="equipment",
                group_key=f"container:{nearest_index}",
                target_paths=(),
                derivation=None,
                dependency_paths=(),
                dependency_bindings=(),
                char_start=draft.char_start,
                char_end=draft.char_end,
                source_text=draft.source_text,
                evidence_origin="audited_source_auxiliary",
                render_policy="opaque_identifier",
                rationale=(
                    "Host-localized compact equipment token to the uniquely nearest container; "
                    "that source container lacks typeDescription while the originally targeted "
                    "container retains a separate type occurrence."
                ),
            )
        )
    return merge_drafts((), tuple(output))


def normalize_party_address_suffix_locality(
    *, raw: str, drafts: Sequence[SpanDraft], source_target: Mapping[str, Any]
) -> tuple[SpanDraft, ...]:
    """Complete a target address only from its uniquely local printed suffix.

    Party labels commonly exclude a postal suffix from the accepted address anchor because the
    suffix is co-printed with the separately owned city or country on the following line.  A suffix
    is added only when the existing address tokens are a strict prefix of the target address, the
    remaining target tokens occur exactly once on the immediately following line, and another
    target-backed field for the same party owns text on that line.  These conditions make the join
    role-local and prevent equal postal codes elsewhere in the document from being borrowed.
    """

    lines = line_spans(raw)
    grouped: dict[str, list[SpanDraft]] = defaultdict(list)
    for draft in drafts:
        grouped[draft.logical_key].append(draft)
    additions: list[SpanDraft] = []
    for rows in grouped.values():
        ordered = sorted(rows, key=lambda row: (row.char_start, row.char_end))
        first = ordered[0]
        if not (
            first.render_mode == "target_binding"
            and first.value_kind == "address"
            and first.group_kind == "party"
            and len(first.target_paths) == 1
            and _PARTY_ADDRESS_PATH.fullmatch(first.target_paths[0]) is not None
            and all(_semantic_signature(row) == _semantic_signature(first) for row in ordered)
        ):
            continue
        target_surface = _scalar_surface(_resolve_target_path(source_target, first.target_paths[0]))
        if target_surface is None:
            continue
        target_tokens = tuple(token for token, _start, _end in _surface_token_spans(target_surface))
        owned_tokens = tuple(
            token
            for row in ordered
            for token, _start, _end in _surface_token_spans(row.source_text)
        )
        if (
            not owned_tokens
            or len(owned_tokens) >= len(target_tokens)
            or target_tokens[: len(owned_tokens)] != owned_tokens
        ):
            continue
        missing_tokens = target_tokens[len(owned_tokens) :]
        last_line_number = line_number_for_char(lines, max(row.char_end for row in ordered) - 1)
        candidate_line = next(
            (line for line in lines if line.number == last_line_number + 1),
            None,
        )
        if candidate_line is None:
            continue
        same_party_context = any(
            row.logical_key != first.logical_key
            and row.render_mode == "target_binding"
            and row.group_kind == "party"
            and row.group_key == first.group_key
            and row.char_start < candidate_line.char_end
            and candidate_line.char_start < row.char_end
            for row in drafts
        )
        if not same_party_context:
            continue
        line_tokens = _surface_token_spans(candidate_line.text)
        candidates: list[tuple[int, int]] = []
        for index in range(len(line_tokens) - len(missing_tokens) + 1):
            window = line_tokens[index : index + len(missing_tokens)]
            if tuple(token for token, _start, _end in window) != missing_tokens:
                continue
            char_start = candidate_line.char_start + window[0][1]
            char_end = candidate_line.char_start + window[-1][2]
            if any(char_start < other.char_end and other.char_start < char_end for other in drafts):
                continue
            candidates.append((char_start, char_end))
        if len(candidates) != 1:
            continue
        char_start, char_end = candidates[0]
        additions.append(
            replace(
                first,
                draft_id=(
                    "host_party_address_suffix_"
                    + sha256_bytes(f"{first.logical_key}\0{char_start}\0{char_end}".encode())[:16]
                ),
                char_start=char_start,
                char_end=char_end,
                source_text=raw[char_start:char_end],
                evidence_origin="host_verified_agent_proposal",
                rationale=(
                    first.rationale
                    + " Host appended the unique remaining target-address suffix from the "
                    "immediately following line owned by the same party."
                ),
            )
        )
    return merge_drafts(drafts, additions)


def normalize_segmented_party_address_targets(
    *, drafts: Sequence[SpanDraft], source_target: Mapping[str, Any]
) -> tuple[SpanDraft, ...]:
    """Promote a complete set of party-scoped source-only address segments.

    Some labels model an address without a city even though OCR interleaves the city between two
    address lines.  Exact source-only address fragments can therefore be the complete deterministic
    realization of one target without ever forming a contiguous accepted anchor.  Promotion is
    allowed only when every address fragment in that party scope maps to one unique, disjoint
    target-token interval and the distinct intervals cover the target exactly in order.
    """

    patch = source_target.get("documentPatch")
    parties = patch.get("parties") if isinstance(patch, Mapping) else None
    if not isinstance(parties, Mapping):
        return tuple(drafts)

    address_targets: list[tuple[str, str, str]] = []
    for role, party in parties.items():
        if role == "notifyParties":
            if not isinstance(party, list):
                continue
            for index, notify_party in enumerate(party):
                if not isinstance(notify_party, Mapping) or not isinstance(
                    notify_party.get("address"), str
                ):
                    continue
                path = f"documentPatch.parties.notifyParties[{index}].address"
                _group_kind, group_key = _canonical_target_group((path,))
                address_targets.append((path, group_key, str(notify_party["address"])))
            continue
        if not isinstance(party, Mapping) or not isinstance(party.get("address"), str):
            continue
        path = f"documentPatch.parties.{role}.address"
        _group_kind, group_key = _canonical_target_group((path,))
        address_targets.append((path, group_key, str(party["address"])))

    represented_paths = _represented_target_inputs(drafts)
    promotions: dict[str, str] = {}
    for target_path, group_key, target_surface in address_targets:
        if target_path in represented_paths:
            continue
        candidates = tuple(
            row
            for row in drafts
            if row.render_mode in {"deterministic_auxiliary", "agent_residual"}
            and (
                row.value_kind == "address"
                or (
                    row.value_kind == "identifier"
                    and re.search(r"postal|postcode|zip", row.logical_key, re.IGNORECASE)
                    and re.fullmatch(r"[0-9]{4,10}(?:-[0-9]+)?", row.source_text.strip())
                )
            )
            and row.group_kind == "party"
            and row.group_key == group_key
            and not row.target_paths
            and not row.dependency_paths
            and not row.dependency_bindings
        )
        if not candidates:
            continue
        target_tokens = tuple(token for token, _start, _end in _surface_token_spans(target_surface))
        source_token_groups = {
            tuple(token for token, _start, _end in _surface_token_spans(row.source_text))
            for row in candidates
        }
        if not target_tokens or () in source_token_groups:
            continue
        intervals: list[tuple[int, int]] = []
        valid = True
        for source_tokens in source_token_groups:
            starts = tuple(
                index
                for index in range(len(target_tokens) - len(source_tokens) + 1)
                if target_tokens[index : index + len(source_tokens)] == source_tokens
            )
            if len(starts) != 1:
                valid = False
                break
            intervals.append((starts[0], starts[0] + len(source_tokens)))
        ordered = sorted(intervals)
        if not valid or not ordered or ordered[0][0] != 0 or ordered[-1][1] != len(target_tokens):
            continue
        if any(
            left_end != right_start
            for (_left_start, left_end), (right_start, _right_end) in pairwise(ordered)
        ):
            continue
        if len(set(ordered)) != len(ordered):
            continue
        for row in candidates:
            prior = promotions.setdefault(row.draft_id, target_path)
            if prior != target_path:
                raise ValueError(
                    f"party address fragment has competing exact target promotions: {row.draft_id}"
                )

    output: list[SpanDraft] = []
    for row in drafts:
        promoted_path = promotions.get(row.draft_id)
        if promoted_path is None:
            output.append(row)
            continue
        group_kind, group_key = _canonical_target_group((promoted_path,))
        output.append(
            replace(
                row,
                logical_key="anchor:" + promoted_path,
                render_mode="target_binding",
                value_kind="address",
                group_kind=group_kind,
                group_key=group_key,
                target_paths=(promoted_path,),
                evidence_origin="host_verified_agent_proposal",
                render_policy=(
                    "opaque_identifier" if row.value_kind == "identifier" else "natural_text"
                ),
                rationale=(
                    row.rationale
                    + " Host promoted the party-scoped address fragments because their unique "
                    "ordered token projections cover the structured address exactly."
                ),
            )
        )
    return merge_drafts(output)


def normalize_owned_party_address_components(
    *, drafts: Sequence[SpanDraft], source_target: Mapping[str, Any]
) -> tuple[SpanDraft, ...]:
    """Join separately owned postal/address fragments into their explicit label.

    Unlike adjacency completion, this handles a postcode separated from the street
    by an independently printed city. The source already assigns the fragment to
    this party. A unique uncovered target interval and a single address owner are
    required; target-backed cities and referenced auxiliary facts are never stolen.
    """
    grouped: dict[str, list[SpanDraft]] = defaultdict(list)
    for draft in drafts:
        grouped[draft.logical_key].append(draft)
    referenced = {key for draft in drafts for key in draft.dependency_bindings}
    promotions: dict[str, SpanDraft] = {}
    for owners in grouped.values():
        owner = owners[0]
        if not (
            owner.render_mode == "target_binding"
            and owner.value_kind == "address"
            and owner.group_kind == "party"
            and len(owner.target_paths) == 1
            and _PARTY_ADDRESS_PATH.fullmatch(owner.target_paths[0])
        ):
            continue
        value = _scalar_surface(_resolve_target_path(source_target, owner.target_paths[0]))
        if value is None:
            continue
        tokens = tuple(t[0] for t in _surface_token_spans(value))

        def intervals(
            surface: str, tokens: tuple[str, ...] = tokens
        ) -> tuple[tuple[int, int], ...]:
            fragment = tuple(t[0] for t in _surface_token_spans(surface))
            if not fragment:
                return ()
            return tuple(
                (i, i + len(fragment))
                for i in range(len(tokens) - len(fragment) + 1)
                if tokens[i : i + len(fragment)] == fragment
            )

        owned = [intervals(d.source_text) for d in owners]
        if any(len(spans) != 1 for spans in owned):
            continue
        covered = {i for spans in owned for start, end in spans for i in range(start, end)}
        candidates: list[tuple[list[SpanDraft], tuple[int, int]]] = []
        for key, rows in grouped.items():
            first = rows[0]
            if key in referenced or not all(
                d.group_kind == "party"
                and d.group_key == owner.group_key
                and d.render_mode in {"deterministic_auxiliary", "agent_residual"}
                and not d.target_paths
                and not d.dependency_paths
                and not d.dependency_bindings
                for d in rows
            ):
                continue
            if first.value_kind not in {"address", "location"} and not (
                first.value_kind == "identifier"
                and re.search(r"postal|postcode|zip", key, re.IGNORECASE)
            ):
                continue
            spans = [intervals(d.source_text) for d in rows]
            if any(len(value) != 1 for value in spans) or len({s[0] for s in spans}) != 1:
                continue
            start, end = spans[0][0]
            if not covered.intersection(range(start, end)):
                candidates.append((rows, (start, end)))
        for rows, (start, end) in candidates:
            if any(
                other is not rows and start < right and left < end
                for other, (left, right) in candidates
            ):
                continue
            for row in rows:
                previous = promotions.setdefault(row.draft_id, owner)
                if previous.target_paths != owner.target_paths:
                    raise ValueError("party component has competing explicit address owners")
    return merge_drafts(
        tuple(
            replace(
                row,
                logical_key=promotions[row.draft_id].logical_key,
                render_mode="target_binding",
                value_kind="address",
                target_paths=promotions[row.draft_id].target_paths,
                render_policy=promotions[row.draft_id].render_policy,
                evidence_origin="host_verified_agent_proposal",
                rationale=row.rationale
                + " This role-owned component uniquely completes the explicit address label; "
                "no city or country was inferred.",
            )
            if row.draft_id in promotions
            else row
            for row in drafts
        )
    )


def normalize_party_address_inline_segments(
    *, raw: str, drafts: Sequence[SpanDraft], source_target: Mapping[str, Any]
) -> tuple[SpanDraft, ...]:
    """Complete target-address tokens adjacent on the owner's boundary lines.

    Repeated forms sometimes split one structured address into several physical slots and omit a
    middle token run. The host expands only along an exact, uniquely positioned target-token
    sequence on the same source line. It therefore cannot borrow a matching location from another
    party or infer address text from vocabulary.
    """

    lines = line_spans(raw)
    grouped: dict[str, list[SpanDraft]] = defaultdict(list)
    for draft in drafts:
        grouped[draft.logical_key].append(draft)
    additions: list[SpanDraft] = []
    discarded_ids: set[str] = set()
    seen: set[tuple[str, int, int]] = set()
    for rows in grouped.values():
        first = rows[0]
        if not (
            first.render_mode == "target_binding"
            and first.value_kind == "address"
            and first.group_kind == "party"
            and len(first.target_paths) == 1
            and _PARTY_ADDRESS_PATH.fullmatch(first.target_paths[0]) is not None
            and all(_semantic_signature(row) == _semantic_signature(first) for row in rows)
        ):
            continue
        target_surface = _scalar_surface(_resolve_target_path(source_target, first.target_paths[0]))
        target_tokens = (
            tuple(token for token, _start, _end in _surface_token_spans(target_surface))
            if target_surface is not None
            else ()
        )
        if not target_tokens:
            continue
        for owner in rows:
            line_number = line_number_for_char(lines, owner.char_start)
            last_line = line_number_for_char(lines, owner.char_end - 1)
            source_offset = lines[line_number - 1].char_start
            source_text = raw[source_offset : lines[last_line - 1].char_end]
            owner_tokens = tuple(
                token for token, _start, _end in _surface_token_spans(owner.source_text)
            )
            if not owner_tokens:
                continue
            target_indexes = tuple(
                index
                for index in range(len(target_tokens) - len(owner_tokens) + 1)
                if target_tokens[index : index + len(owner_tokens)] == owner_tokens
            )
            source_tokens = _surface_token_spans(source_text)
            source_indexes = tuple(
                index
                for index in range(len(source_tokens) - len(owner_tokens) + 1)
                if tuple(
                    token
                    for token, _start, _end in source_tokens[index : index + len(owner_tokens)]
                )
                == owner_tokens
                and source_offset + source_tokens[index][1] >= owner.char_start
                and source_offset + source_tokens[index + len(owner_tokens) - 1][2]
                <= owner.char_end
            )
            if len(target_indexes) != 1 or len(source_indexes) != 1:
                continue
            target_start = target_indexes[0]
            source_start = source_indexes[0]
            target_end = target_start + len(owner_tokens)
            source_end = source_start + len(owner_tokens)
            while (
                target_start > 0
                and source_start > 0
                and target_tokens[target_start - 1] == source_tokens[source_start - 1][0]
            ):
                target_start -= 1
                source_start -= 1
            while (
                target_end < len(target_tokens)
                and source_end < len(source_tokens)
                and target_tokens[target_end] == source_tokens[source_end][0]
            ):
                target_end += 1
                source_end += 1
            candidate_ranges: list[tuple[int, int]] = []
            if source_start < source_indexes[0]:
                candidate_ranges.append(
                    (
                        source_offset + source_tokens[source_start][1],
                        source_offset + source_tokens[source_indexes[0] - 1][2],
                    )
                )
            owner_source_end = source_indexes[0] + len(owner_tokens)
            if source_end > owner_source_end:
                candidate_ranges.append(
                    (
                        source_offset + source_tokens[owner_source_end][1],
                        source_offset + source_tokens[source_end - 1][2],
                    )
                )
            for start, end in candidate_ranges:
                identity = (first.target_paths[0], start, end)
                if identity in seen:
                    continue
                overlaps = tuple(
                    row for row in drafts if start < row.char_end and row.char_start < end
                )
                replaceable = bool(overlaps) and all(
                    not row.target_paths
                    and not row.dependency_paths
                    and not row.dependency_bindings
                    and row.render_mode in {"deterministic_auxiliary", "agent_residual"}
                    and start <= row.char_start
                    and row.char_end <= end
                    for row in overlaps
                )
                if overlaps and not replaceable:
                    continue
                if replaceable:
                    discarded_ids.update(row.draft_id for row in overlaps)
                seen.add(identity)
                additions.append(
                    replace(
                        owner,
                        draft_id=(
                            "host_party_address_inline_"
                            + sha256_bytes(f"{first.target_paths[0]}\0{start}\0{end}".encode())[:16]
                        ),
                        char_start=start,
                        char_end=end,
                        source_text=raw[start:end],
                        evidence_origin="host_verified_agent_proposal",
                        rationale=(
                            owner.rationale
                            + " Host appended this uniquely adjacent same-line target-address "
                            "token segment."
                        ),
                    )
                )
    retained = tuple(row for row in drafts if row.draft_id not in discarded_ids)
    return merge_drafts(retained, additions)


def normalize_canonical_carrier_repeats(
    *, raw: str, drafts: Sequence[SpanDraft], source_target: Mapping[str, Any]
) -> tuple[SpanDraft, ...]:
    """Unify exact canonical-carrier repeats under the proven target owner.

    The structured carrier name and an existing carrier-static target binding prove the semantic
    owner.  The host expands only byte-identical, token-bounded repeats of that owner's source
    surface.  A separate source-only carrier-static binding may be merged only when it owns the
    exact same span and normalizes to the canonical target; aliases and overlapping wider carrier
    phrases remain untouched.
    """

    carrier = source_carrier(source_target)
    if carrier is None:
        return tuple(drafts)
    normalized_carrier = _normalized_surface(carrier)
    grouped: dict[str, list[SpanDraft]] = defaultdict(list)
    for draft in drafts:
        grouped[draft.logical_key].append(draft)
    owner_groups = tuple(
        rows
        for rows in grouped.values()
        if rows[0].render_mode == "carrier_static"
        and _CARRIER_NAME_PATH in rows[0].target_paths
        and all(_normalized_surface(row.source_text) == normalized_carrier for row in rows)
    )
    if len(owner_groups) != 1:
        return tuple(drafts)
    owner_rows = owner_groups[0]
    owner = owner_rows[0]
    candidate_spans = {
        (start, start + len(surface))
        for surface in {row.source_text for row in owner_rows}
        for start in _exact_offsets(raw, surface)
        if is_token_bounded_surface_span(raw, start, start + len(surface))
    }
    discarded_ids: set[str] = set()
    additions: list[SpanDraft] = []
    for start, end in sorted(candidate_spans):
        overlaps = tuple(row for row in drafts if start < row.char_end and row.char_start < end)
        if any(
            row.logical_key == owner.logical_key and row.char_start == start and row.char_end == end
            for row in overlaps
        ):
            continue
        mergeable = tuple(
            row
            for row in overlaps
            if row.char_start == start
            and row.char_end == end
            and row.render_mode == "carrier_static"
            and not row.target_paths
            and _normalized_surface(row.source_text) == normalized_carrier
        )
        if overlaps and len(mergeable) != len(overlaps):
            continue
        discarded_ids.update(row.draft_id for row in mergeable)
        additions.append(
            replace(
                owner,
                draft_id=(
                    "host_canonical_carrier_repeat_"
                    + sha256_bytes(f"{owner.logical_key}\0{start}\0{end}".encode())[:16]
                ),
                char_start=start,
                char_end=end,
                source_text=raw[start:end],
                evidence_origin="host_verified_agent_proposal",
                rationale=(
                    owner.rationale
                    + " Host attached this byte-identical canonical-carrier repeat to the "
                    "unique structured carrier-name owner."
                ),
            )
        )
    retained = tuple(row for row in drafts if row.draft_id not in discarded_ids)
    return merge_drafts(retained, additions)


def normalize_carrier_identity_block_details(
    *, raw: str, drafts: Sequence[SpanDraft], source_target: Mapping[str, Any]
) -> tuple[SpanDraft, ...]:
    """Complete exact carrier facts inside an already proven carrier-identity block.

    A block is eligible only when it contains the structured canonical carrier name plus at least
    two other structured carrier fields and contains no mutable non-carrier owner.  Within that
    closed scope the host can safely attach exact repeats of structured carrier scalars.  It also
    owns a grouped numeric identifier immediately preceding the standard company-registry marker;
    shipment registration fields elsewhere in the document are deliberately out of scope.
    """

    lines = line_spans(raw)
    block_by_line: dict[int, int] = {}
    block_index = -1
    inside_block = False
    for line in lines:
        if line.text.strip():
            if not inside_block:
                block_index += 1
                inside_block = True
            block_by_line[line.number] = block_index
        else:
            inside_block = False

    carrier_target_rows = tuple(
        row
        for row in drafts
        if row.render_mode == "carrier_static"
        and any(path.startswith("documentPatch.parties.carrier.") for path in row.target_paths)
    )
    candidate_blocks = {
        block_by_line[line_number_for_char(lines, row.char_start)]
        for row in carrier_target_rows
        if _CARRIER_NAME_PATH in row.target_paths
        and line_number_for_char(lines, row.char_start) in block_by_line
    }
    eligible_blocks: set[int] = set()
    for block in candidate_blocks:
        block_rows = tuple(
            row
            for row in drafts
            if block_by_line.get(line_number_for_char(lines, row.char_start)) == block
        )
        structured_paths = {
            path
            for row in block_rows
            if row.render_mode == "carrier_static"
            for path in row.target_paths
            if path.startswith("documentPatch.parties.carrier.")
        }
        if (
            _CARRIER_NAME_PATH in structured_paths
            and len(structured_paths) >= 3
            and all(row.render_mode == "carrier_static" for row in block_rows)
        ):
            eligible_blocks.add(block)
    if not eligible_blocks:
        return tuple(drafts)

    additions: list[SpanDraft] = []
    occupied: list[SpanDraft] = list(drafts)
    owner_groups: dict[str, list[SpanDraft]] = defaultdict(list)
    for row in carrier_target_rows:
        if len(row.target_paths) == 1:
            owner_groups[row.target_paths[0]].append(row)
    for path, owners in owner_groups.items():
        value = _resolve_target_path(source_target, path)
        surface = _scalar_surface(value)
        if not surface:
            continue
        owner = owners[0]
        for line in lines:
            if block_by_line.get(line.number) not in eligible_blocks:
                continue
            for start, end in _line_token_sequence_spans(line, surface):
                if any(start < row.char_end and row.char_start < end for row in occupied):
                    continue
                addition = replace(
                    owner,
                    draft_id=(
                        "host_carrier_block_repeat_"
                        + sha256_bytes(f"{path}\0{start}\0{end}".encode())[:16]
                    ),
                    char_start=start,
                    char_end=end,
                    source_text=raw[start:end],
                    evidence_origin="host_verified_agent_proposal",
                    rationale=(
                        owner.rationale
                        + " Host attached this exact structured carrier repeat inside the same "
                        "closed carrier-identity block."
                    ),
                )
                additions.append(addition)
                occupied.append(addition)

    for line in lines:
        if block_by_line.get(line.number) not in eligible_blocks:
            continue
        marker = _CARRIER_REGISTRY_MARKER.search(line.text)
        if marker is None:
            continue
        identifier = _GROUPED_NUMERIC_REGISTRY_IDENTIFIER.fullmatch(line.text[: marker.start()])
        if identifier is None:
            continue
        start = line.char_start + identifier.start("value")
        end = line.char_start + identifier.end("value")
        if any(start < row.char_end and row.char_start < end for row in occupied):
            continue
        addition = SpanDraft(
            draft_id=(
                "host_carrier_registry_identifier_" + sha256_bytes(f"{start}\0{end}".encode())[:16]
            ),
            logical_key=(
                "agent:carrier:registry_identifier:"
                + sha256_bytes(_normalized_surface(raw[start:end]).encode())[:16]
            ),
            render_mode="carrier_static",
            value_kind="identifier",
            group_kind="carrier",
            group_key="carrier:principal",
            target_paths=(),
            derivation=None,
            dependency_paths=(),
            dependency_bindings=(),
            char_start=start,
            char_end=end,
            source_text=raw[start:end],
            evidence_origin="host_verified_agent_proposal",
            render_policy="opaque_identifier",
            rationale=(
                "Host classified this grouped company-registry identifier as carrier-static "
                "inside a closed structured carrier-identity block."
            ),
        )
        additions.append(addition)
        occupied.append(addition)
    return merge_drafts(drafts, additions)


def normalize_canonical_carrier_initialism(
    *, raw: str, drafts: Sequence[SpanDraft], source_target: Mapping[str, Any]
) -> tuple[SpanDraft, ...]:
    """Own an unbound canonical-carrier initialism only in explicit carrier context.

    The initialism is derived from the structured carrier name rather than guessed from a model.
    Each added occurrence must be uppercase, token bounded, and locally identified by carrier
    relationship, tariff, terms, or signature language.  Occurrences already contained in a
    carrier name, domain, or affiliate binding remain with that wider owner.
    """

    carrier = source_carrier(source_target)
    if carrier is None:
        return tuple(drafts)
    words = tuple(re.findall(r"[A-Za-z0-9]+", carrier))
    initialism = "".join(word[0] for word in words).upper()
    if not 3 <= len(initialism) <= 8 or not initialism.isalpha():
        return tuple(drafts)
    if not any(
        row.render_mode == "carrier_static" and _CARRIER_NAME_PATH in row.target_paths
        for row in drafts
    ):
        return tuple(drafts)

    lines = line_spans(raw)
    additions: list[SpanDraft] = []
    for start in _exact_offsets(raw, initialism):
        end = start + len(initialism)
        if not is_token_bounded_surface_span(raw, start, end) or any(
            start < row.char_end and row.char_start < end for row in drafts
        ):
            continue
        line_number = line_number_for_char(lines, start)
        line = lines[line_number - 1]
        line_tokens = tuple(token for token, _start, _end in _surface_token_spans(line.text))
        local_suffix = raw[end : min(line.char_end, end + 3)]
        carrier_context = (
            bool(re.match(r"(?:['\u2019]s)\b", local_suffix, flags=re.IGNORECASE))
            or any(token.startswith("tariff") for token in line_tokens)
            or any(token in {"carrier", "signed", "signature", "terms"} for token in line_tokens)
        )
        if not carrier_context:
            continue
        additions.append(
            SpanDraft(
                draft_id=(
                    "host_carrier_initialism_"
                    + sha256_bytes(f"{initialism}\0{start}\0{end}".encode())[:16]
                ),
                logical_key=f"agent:carrier:canonical_initialism:{initialism.casefold()}",
                render_mode="carrier_static",
                value_kind="organization",
                group_kind="carrier",
                group_key="carrier:principal",
                target_paths=(),
                derivation=None,
                dependency_paths=(),
                dependency_bindings=(),
                char_start=start,
                char_end=end,
                source_text=raw[start:end],
                evidence_origin="host_verified_agent_proposal",
                render_policy="natural_text",
                rationale=(
                    "Host derived this uppercase initialism from the canonical carrier name and "
                    "classified the occurrence as carrier-static from its explicit local "
                    "carrier, tariff, terms, or signature context."
                ),
            )
        )
    return merge_drafts(drafts, additions)


def normalize_labeled_source_only_fields(
    *, raw: str, drafts: Sequence[SpanDraft]
) -> tuple[SpanDraft, ...]:
    """Own complete next-line values for exact source-only field captions."""

    lines = line_spans(raw)
    line_by_number = {line.number: line for line in lines}
    additions: list[SpanDraft] = []
    discarded_ids: set[str] = set()
    for caption in lines:
        contract = _SOURCE_ONLY_FIELD_CAPTIONS.get(_normalized_surface(caption.text))
        if contract is None:
            continue
        value_line = line_by_number.get(caption.number + 1)
        if value_line is None:
            continue
        left_trim = len(value_line.text) - len(value_line.text.lstrip())
        right_trim = len(value_line.text.rstrip())
        if right_trim <= left_trim:
            continue
        start = value_line.char_start + left_trim
        end = value_line.char_start + right_trim
        source_text = raw[start:end]
        if not any(character.isalpha() for character in source_text):
            continue
        logical_key, group_kind, group_key, value_kind = contract
        overlaps = tuple(row for row in drafts if start < row.char_end and row.char_start < end)
        if overlaps:
            already_owned = all(
                row.char_start == start
                and row.char_end == end
                and row.logical_key == logical_key
                and row.render_mode == "deterministic_auxiliary"
                and row.value_kind == value_kind
                and not row.target_paths
                for row in overlaps
            )
            if already_owned:
                continue
            literal_rows = tuple(
                row
                for row in overlaps
                if row.char_start == start
                and row.char_end == end
                and row.render_mode == "literal_static"
            )
            if len(literal_rows) != len(overlaps):
                continue
            discarded_ids.update(row.draft_id for row in literal_rows)
        additions.append(
            SpanDraft(
                draft_id=(
                    "host_labeled_source_location_"
                    + sha256_bytes(f"{logical_key}\0{start}\0{end}".encode())[:16]
                ),
                logical_key=logical_key,
                render_mode="deterministic_auxiliary",
                value_kind=value_kind,
                group_kind=group_kind,
                group_key=group_key,
                target_paths=(),
                derivation=None,
                dependency_paths=(),
                dependency_bindings=(),
                char_start=start,
                char_end=end,
                source_text=source_text,
                evidence_origin="host_verified_agent_proposal",
                render_policy="natural_text",
                rationale=(
                    "Host owned the complete next-line value under the exact source-only "
                    f"{caption.text.strip()} caption."
                ),
            )
        )
    retained = tuple(row for row in drafts if row.draft_id not in discarded_ids)
    return merge_drafts(retained, additions)


def normalize_labeled_movement_type_repeats(
    *, raw: str, drafts: Sequence[SpanDraft]
) -> tuple[SpanDraft, ...]:
    """Append an omitted movement-type repeat only inside a proven structured cargo row.

    A labeled ``TYPE OF MOVEMENT`` value establishes the source-only field.  To avoid treating text
    equality as semantics, automatic expansion additionally requires at least two already owned
    repeats between package and equipment owners on one line.  A new occurrence must have the same
    normalized surface, occupy that same package-before/equipment-after frame, and be separated by
    slash delimiters.  Repeated prose or captions therefore remain outside this rule.
    """

    lines = line_spans(raw)
    line_by_number = {line.number: line for line in lines}
    grouped: dict[str, list[SpanDraft]] = defaultdict(list)
    for draft in drafts:
        grouped[draft.logical_key].append(draft)

    def source_line(row: SpanDraft) -> LineSpan | None:
        start = line_number_for_char(lines, row.char_start)
        end = line_number_for_char(lines, row.char_end - 1)
        return line_by_number[start] if start == end else None

    def has_structured_row_frame(start: int, end: int, line: LineSpan) -> bool:
        package_before = any(
            row.group_kind == "package"
            and line.char_start <= row.char_start
            and row.char_end <= start
            for row in drafts
        )
        equipment_after = any(
            row.group_kind == "equipment"
            and end <= row.char_start
            and row.char_end <= line.char_end
            for row in drafts
        )
        before = raw[line.char_start : start].rstrip()
        after = raw[end : line.char_end].lstrip()
        return package_before and equipment_after and before.endswith("/") and after.startswith("/")

    additions: list[SpanDraft] = []
    for rows in grouped.values():
        ordered = sorted(rows, key=lambda row: (row.char_start, row.char_end))
        first = ordered[0]
        if not (
            first.render_mode == "deterministic_auxiliary"
            and first.value_kind == "operational_text"
            and first.group_kind == "transport"
            and not first.target_paths
        ):
            continue
        normalized_surfaces = {_normalized_surface(row.source_text) for row in ordered}
        if len(normalized_surfaces) != 1 or not next(iter(normalized_surfaces)):
            continue
        normalized_value = next(iter(normalized_surfaces))
        owned_label = any(
            (line := source_line(row)) is not None
            and (caption := line_by_number.get(line.number - 1)) is not None
            and _normalized_surface(caption.text).startswith("typeofmovement")
            for row in ordered
        )
        labeled_spans: set[tuple[int, int]] = set()
        for caption in lines:
            if not _normalized_surface(caption.text).startswith("typeofmovement"):
                continue
            value_line = line_by_number.get(caption.number + 1)
            if value_line is None:
                continue
            left_trim = len(value_line.text) - len(value_line.text.lstrip())
            right_trim = len(value_line.text.rstrip())
            if right_trim <= left_trim:
                continue
            start = value_line.char_start + left_trim
            end = value_line.char_start + right_trim
            if _normalized_surface(raw[start:end]) == normalized_value:
                labeled_spans.add((start, end))
        structured_rows = sum(
            1
            for row in ordered
            if (line := source_line(row)) is not None
            and has_structured_row_frame(row.char_start, row.char_end, line)
        )
        if not (owned_label or labeled_spans) or structured_rows < 2:
            continue

        candidate_spans: set[tuple[int, int]] = set(labeled_spans)
        for surface in dict.fromkeys(row.source_text for row in ordered):
            candidate_spans.update(
                (start, start + len(surface)) for start in _exact_offsets(raw, surface)
            )
        for start, end in sorted(candidate_spans):
            if any(start < row.char_end and row.char_start < end for row in drafts):
                continue
            line_number = line_number_for_char(lines, start)
            line = line_by_number[line_number]
            if line_number_for_char(lines, end - 1) != line_number:
                continue
            if (start, end) not in labeled_spans and not has_structured_row_frame(start, end, line):
                continue
            additions.append(
                replace(
                    first,
                    draft_id=(
                        "host_movement_type_repeat_"
                        + sha256_bytes(f"{first.logical_key}\0{start}\0{end}".encode())[:16]
                    ),
                    char_start=start,
                    char_end=end,
                    source_text=raw[start:end],
                    evidence_origin="host_verified_agent_proposal",
                    rationale=(
                        first.rationale
                        + (
                            " Host attached the complete value below the exact TYPE OF MOVEMENT "
                            "caption to the proven repeated movement binding."
                            if (start, end) in labeled_spans
                            else " Host appended this exact movement-type repeat from the proven "
                            "package-before/equipment-after cargo-row frame."
                        )
                    ),
                )
            )
    return merge_drafts(drafts, additions)


def normalize_explicit_loading_terminal_locality(
    *, raw: str, drafts: Sequence[SpanDraft]
) -> tuple[SpanDraft, ...]:
    """Own the complete value immediately below an explicit terminal-only caption.

    A mis-owned port target is detached only when another occurrence retains port-of-loading
    ownership.  An entirely unowned value line is materialized directly.  A sole ambiguous port
    occurrence or any partially owned line remains unchanged for semantic review.
    """

    # OCR commonly puts the next field heading immediately after an empty box.
    # A known caption is not the value of that box, regardless of capitalization.
    drafts = tuple(
        replace(
            row,
            logical_key="host:static_caption:" + row.logical_key,
            render_mode="literal_static",
            value_kind="other_text",
            group_kind="document",
            group_key="document:caption",
            rationale="Known form caption after an empty loading-terminal field, not geography.",
        )
        if row.group_key == "route:loading_terminal"
        and not row.target_paths
        and not row.dependency_paths
        and not row.dependency_bindings
        and _normalized_surface(row.source_text) in _FORM_FIELD_CAPTIONS
        else row
        for row in drafts
    )
    lines = line_spans(raw)
    line_by_number = {line.number: line for line in lines}
    grouped: dict[str, list[SpanDraft]] = defaultdict(list)
    for draft in drafts:
        grouped[draft.logical_key].append(draft)
    replacements: dict[str, SpanDraft] = {}
    for rows in grouped.values():
        first = rows[0]
        if not (
            first.render_mode == "target_binding"
            and first.target_paths == (_PORT_OF_LOADING_PATH,)
            and first.value_kind == "location"
        ):
            continue
        terminal_rows: list[SpanDraft] = []
        for row in rows:
            start_line = line_number_for_char(lines, row.char_start)
            end_line = line_number_for_char(lines, row.char_end - 1)
            caption = line_by_number.get(start_line - 1)
            if (
                start_line == end_line
                and caption is not None
                and _normalized_surface(caption.text) == _LOADING_TERMINAL_CAPTION
            ):
                terminal_rows.append(row)
        if not terminal_rows or len(terminal_rows) == len(rows):
            continue
        for row in terminal_rows:
            surface_key = sha256_bytes(_normalized_surface(row.source_text).encode("utf-8"))[:12]
            replacements[row.draft_id] = SpanDraft(
                draft_id=(
                    "host_loading_terminal_"
                    + sha256_bytes(f"{row.char_start}\0{row.char_end}".encode())[:16]
                ),
                logical_key=f"agent:route:loading_terminal:{surface_key}",
                render_mode="deterministic_auxiliary",
                value_kind="location",
                group_kind="route",
                group_key="route:loading_terminal",
                target_paths=(),
                derivation=None,
                dependency_paths=(),
                dependency_bindings=(),
                char_start=row.char_start,
                char_end=row.char_end,
                source_text=row.source_text,
                evidence_origin="host_verified_agent_proposal",
                render_policy="natural_text",
                rationale=(
                    "Host reclassified this location as a source-only loading terminal because its "
                    "immediately preceding line is the explicit LOADING PIER/TERMINAL caption and "
                    "another occurrence retains the structured port-of-loading target."
                ),
            )
    working = tuple(replacements.get(row.draft_id, row) for row in drafts)
    additions: list[SpanDraft] = []
    discarded_ids: set[str] = set()
    for caption in lines:
        if _normalized_surface(caption.text) != _LOADING_TERMINAL_CAPTION:
            continue
        value_line = line_by_number.get(caption.number + 1)
        if value_line is None:
            continue
        left_trim = len(value_line.text) - len(value_line.text.lstrip())
        right_trim = len(value_line.text.rstrip())
        if right_trim <= left_trim:
            continue
        start = value_line.char_start + left_trim
        end = value_line.char_start + right_trim
        source_text = raw[start:end]
        if _normalized_surface(source_text) in _FORM_FIELD_CAPTIONS:
            continue
        if not any(character.isalnum() for character in source_text):
            continue
        overlaps = tuple(row for row in working if start < row.char_end and row.char_start < end)
        if overlaps:
            already_owned = all(
                row.char_start == start
                and row.char_end == end
                and row.render_mode == "deterministic_auxiliary"
                and row.value_kind == "location"
                and row.group_kind == "route"
                and row.group_key == "route:loading_terminal"
                and not row.target_paths
                for row in overlaps
            )
            if already_owned:
                continue
            literal_rows = tuple(
                row
                for row in overlaps
                if row.char_start == start
                and row.char_end == end
                and row.render_mode == "literal_static"
            )
            if len(literal_rows) != len(overlaps):
                continue
            discarded_ids.update(row.draft_id for row in literal_rows)
        surface_key = sha256_bytes(_normalized_surface(source_text).encode("utf-8"))[:12]
        additions.append(
            SpanDraft(
                draft_id=("host_loading_terminal_" + sha256_bytes(f"{start}\0{end}".encode())[:16]),
                logical_key=f"agent:route:loading_terminal:{surface_key}",
                render_mode="deterministic_auxiliary",
                value_kind="location",
                group_kind="route",
                group_key="route:loading_terminal",
                target_paths=(),
                derivation=None,
                dependency_paths=(),
                dependency_bindings=(),
                char_start=start,
                char_end=end,
                source_text=source_text,
                evidence_origin="host_verified_agent_proposal",
                render_policy="natural_text",
                rationale=(
                    "Host classified the complete value line as a source-only loading terminal "
                    "because its immediately preceding line is the exact LOADING PIER/TERMINAL "
                    "caption."
                ),
            )
        )
    retained = tuple(row for row in working if row.draft_id not in discarded_ids)
    return merge_drafts(retained, additions)


def normalize_selected_operational_terms(
    *, raw: str, drafts: Sequence[SpanDraft]
) -> tuple[SpanDraft, ...]:
    """Own two exact selected-commercial line grammars without semantic extrapolation.

    A free-detention term has a deterministic numeric vocabulary realization.  A country-prefixed
    free-out instruction is intentionally agent residual: preserving the source country while
    other shipment geography changes would be incorrect, and no structured destination-country
    relationship is proven here.  Partial ownership or any non-literal overlap stays untouched.
    """

    working = tuple(drafts)
    discarded_ids: set[str] = set()
    additions: list[SpanDraft] = []
    for line in line_spans(raw):
        left_trim = len(line.text) - len(line.text.lstrip())
        right_trim = len(line.text.rstrip())
        if right_trim <= left_trim:
            continue
        start = line.char_start + left_trim
        end = line.char_start + right_trim
        source_text = raw[start:end]
        if _FREE_DETENTION_TERM.fullmatch(source_text) is not None:
            logical_key = "agent:commercial:free_detention"
            render_mode = "deterministic_auxiliary"
            rationale = (
                "Host classified the complete line as a selected numeric free-detention term "
                "using the exact supported line grammar."
            )
        elif _IMPORT_FREE_OUT_INSTRUCTION.fullmatch(source_text) is not None:
            logical_key = "agent:commercial:import_free_out_instruction"
            render_mode = "agent_residual"
            rationale = (
                "Host classified the complete country-prefixed free-out instruction as bounded "
                "linguistic residual because no structured destination-country relationship is "
                "proven by the source label."
            )
        else:
            continue
        overlaps = tuple(row for row in working if start < row.char_end and row.char_start < end)
        if overlaps:
            already_owned = all(
                row.char_start == start
                and row.char_end == end
                and row.render_mode == render_mode
                and row.logical_key == logical_key
                for row in overlaps
            )
            if already_owned:
                continue
            literal_rows = tuple(
                row
                for row in overlaps
                if row.char_start == start
                and row.char_end == end
                and row.render_mode == "literal_static"
            )
            if len(literal_rows) != len(overlaps):
                continue
            discarded_ids.update(row.draft_id for row in literal_rows)
        additions.append(
            SpanDraft(
                draft_id=(
                    "host_selected_operational_term_"
                    + sha256_bytes(f"{logical_key}\0{start}\0{end}".encode())[:16]
                ),
                logical_key=logical_key,
                render_mode=render_mode,
                value_kind="commercial_text",
                group_kind="transport",
                group_key="transport:commercial_terms",
                target_paths=(),
                derivation=None,
                dependency_paths=(),
                dependency_bindings=(),
                char_start=start,
                char_end=end,
                source_text=source_text,
                evidence_origin="host_verified_agent_proposal",
                render_policy="natural_text",
                rationale=rationale,
            )
        )
    retained = tuple(row for row in working if row.draft_id not in discarded_ids)
    return merge_drafts(retained, additions)


def normalize_selected_freight_payment_locality(
    *, raw: str, drafts: Sequence[SpanDraft], source_target: Mapping[str, Any]
) -> tuple[SpanDraft, ...]:
    """Move freight payment ownership from an option header to selected-value lines.

    ``PREPAID COLLECT`` in one heading is a mutually exclusive option vocabulary, not a selected
    value.  When the structured value is either option and the source also has an unambiguous
    ``FREIGHT <selected>`` line without the opposite option, the host makes the header occurrence
    literal-static and attaches the selected token to the existing target owner.  Without that
    replacement evidence the accepted anchor remains unchanged for semantic review.
    """

    try:
        payment_value = _resolve_target_path(source_target, _FREIGHT_PAYMENT_PATH)
    except ValueError:
        return tuple(drafts)
    payment_surface = _scalar_surface(payment_value)
    if payment_surface is None:
        return tuple(drafts)
    selected = _normalized_surface(payment_surface)
    if selected not in {"prepaid", "collect"}:
        return tuple(drafts)
    opposite = "collect" if selected == "prepaid" else "prepaid"

    narrowed: list[SpanDraft] = []
    lines = line_spans(raw)
    for row in drafts:
        if (
            row.render_mode != "target_binding"
            or row.target_paths != (_FREIGHT_PAYMENT_PATH,)
            or _normalized_surface(row.source_text) == selected
        ):
            narrowed.append(row)
            continue
        start_line = line_number_for_char(lines, row.char_start)
        end_line = line_number_for_char(lines, row.char_end - 1)
        tokens = _surface_token_spans(row.source_text)
        selected_tokens = tuple(
            (index, token_start, token_end)
            for index, (token, token_start, token_end) in enumerate(tokens)
            if token == selected
            and index > 0
            and tokens[index - 1][0] == "freight"
            and opposite not in {value for value, _start, _end in tokens}
        )
        if start_line != end_line or len(selected_tokens) != 1:
            narrowed.append(row)
            continue
        _index, relative_start, relative_end = selected_tokens[0]
        narrowed.append(
            replace(
                row,
                draft_id=(
                    "host_narrowed_freight_payment_"
                    + sha256_bytes(f"{row.draft_id}\0{relative_start}\0{relative_end}".encode())[
                        :16
                    ]
                ),
                char_start=row.char_start + relative_start,
                char_end=row.char_start + relative_end,
                source_text=row.source_text[relative_start:relative_end],
                evidence_origin="host_verified_agent_proposal",
                rationale=(
                    row.rationale
                    + " Host narrowed the direct payment target from its FREIGHT caption prefix "
                    "to the selected PREPAID/COLLECT value token."
                ),
            )
        )
    drafts = merge_drafts(narrowed)

    grouped: dict[str, list[SpanDraft]] = defaultdict(list)
    for draft in drafts:
        if _FREIGHT_PAYMENT_PATH in draft.target_paths:
            grouped[draft.logical_key].append(draft)
    if len(grouped) != 1:
        return tuple(drafts)
    owner_rows = next(iter(grouped.values()))
    owner = owner_rows[0]
    if owner.render_mode != "target_binding":
        return tuple(drafts)

    caption_ids: set[str] = set()
    for row in owner_rows:
        start_line = line_number_for_char(lines, row.char_start)
        end_line = line_number_for_char(lines, row.char_end - 1)
        if start_line != end_line or _normalized_surface(row.source_text) != selected:
            continue
        line = lines[start_line - 1]
        token_texts = tuple(token for token, _start, _end in _surface_token_spans(line.text))
        if (
            selected in token_texts
            and opposite in token_texts
            and any(token.startswith("freight") for token in token_texts)
        ):
            caption_ids.add(row.draft_id)
    if not caption_ids:
        return tuple(drafts)

    selected_spans: list[tuple[int, int]] = []
    for line in lines:
        tokens = _surface_token_spans(line.text)
        normalized_tokens = tuple(token for token, _start, _end in tokens)
        if opposite in normalized_tokens:
            continue
        for index, (token, token_start, token_end) in enumerate(tokens):
            if token != selected or index == 0 or tokens[index - 1][0] != "freight":
                continue
            selected_spans.append((line.char_start + token_start, line.char_start + token_end))
    if not selected_spans:
        return tuple(drafts)

    additions: list[SpanDraft] = []
    retained_selected = False
    for start, end in selected_spans:
        overlaps = tuple(row for row in drafts if start < row.char_end and row.char_start < end)
        if overlaps:
            if all(
                row.logical_key == owner.logical_key
                and row.char_start == start
                and row.char_end == end
                for row in overlaps
            ):
                retained_selected = True
            continue
        additions.append(
            replace(
                owner,
                draft_id=(
                    "host_selected_freight_payment_"
                    + sha256_bytes(f"{owner.logical_key}\0{start}\0{end}".encode())[:16]
                ),
                char_start=start,
                char_end=end,
                source_text=raw[start:end],
                evidence_origin="host_verified_agent_proposal",
                rationale=(
                    owner.rationale
                    + " Host moved ownership from the mutually exclusive PREPAID/COLLECT "
                    "header to this explicit FREIGHT selected-value occurrence."
                ),
            )
        )
    if not additions and not retained_selected:
        return tuple(drafts)
    retained = tuple(row for row in drafts if row.draft_id not in caption_ids)
    literal_headers = tuple(
        replace(
            row,
            draft_id=(
                "host_literal_freight_payment_option_"
                + sha256_bytes(f"{row.char_start}\0{row.char_end}".encode())[:16]
            ),
            logical_key="agent:literal:freight_payment_option_header",
            render_mode="literal_static",
            value_kind="commercial_text",
            group_kind="document",
            group_key="document:freight_payment_options",
            target_paths=(),
            derivation=None,
            dependency_paths=(),
            dependency_bindings=(),
            evidence_origin="host_verified_agent_proposal",
            render_policy="categorical_surface",
            rationale=(
                "Host classified this token as fixed option vocabulary because its line prints "
                "both mutually exclusive PREPAID and COLLECT choices and a separate FREIGHT "
                "line prints the selected structured value."
            ),
        )
        for row in owner_rows
        if row.draft_id in caption_ids
    )
    return merge_drafts(retained, literal_headers, additions)


def _target_rows_on_line(
    *, line: LineSpan, drafts: Sequence[SpanDraft], target_path: str
) -> tuple[SpanDraft, ...]:
    return tuple(
        row
        for row in drafts
        if row.render_mode == "target_binding"
        and target_path in row.target_paths
        and row.char_start < line.char_end
        and line.char_start < row.char_end
    )


def _at_location_on_frame(*, line: LineSpan, target_surface: str) -> tuple[int, int] | None:
    """Return the unique target-token window in an explicit ``AT <location> ON`` frame."""

    line_tokens = _surface_token_spans(line.text)
    target_tokens = tuple(token for token, _start, _end in _surface_token_spans(target_surface))
    if not target_tokens:
        return None
    matches: list[tuple[int, int]] = []
    for index in range(1, len(line_tokens) - len(target_tokens)):
        window = line_tokens[index : index + len(target_tokens)]
        if (
            tuple(token for token, _start, _end in window) == target_tokens
            and line_tokens[index - 1][0] == "at"
            and line_tokens[index + len(target_tokens)][0] == "on"
        ):
            matches.append(
                (
                    line.char_start + window[0][1],
                    line.char_start + window[-1][2],
                )
            )
    return matches[0] if len(matches) == 1 else None


def normalize_shipped_on_board_summary_locality(
    *, raw: str, drafts: Sequence[SpanDraft], source_target: Mapping[str, Any]
) -> tuple[SpanDraft, ...]:
    """Own selected on-board status and its explicitly framed loading-port repeat.

    ``SHIPPED ON BOARD`` is treated as selected status only when the same source line already owns
    both a vessel name and an on-board date in that order.  A loading-port repeat is added only from
    the unique ``AT <target port> ON`` token frame on that proven summary line.
    """

    working = tuple(drafts)
    discarded: set[str] = set()
    additions: list[SpanDraft] = []
    for line in line_spans(raw):
        status_matches = tuple(_SELECTED_OPERATIONAL_TEXT.finditer(line.text))
        if len(status_matches) != 1:
            continue
        vessel_rows = _target_rows_on_line(line=line, drafts=working, target_path=_VESSEL_NAME_PATH)
        date_rows = _target_rows_on_line(
            line=line, drafts=working, target_path=_SHIPPED_ON_BOARD_DATE_PATH
        )
        if len(vessel_rows) != 1 or len(date_rows) != 1:
            continue
        status_match = status_matches[0]
        status_start = line.char_start + status_match.start()
        status_end = line.char_start + status_match.end()
        vessel = vessel_rows[0]
        date_row = date_rows[0]
        if not (status_end <= vessel.char_start < vessel.char_end <= date_row.char_start):
            continue
        status_owners = tuple(
            row for row in working if status_start < row.char_end and row.char_start < status_end
        )
        status_is_selected = False
        if not status_owners:
            status_is_selected = True
        elif (
            len(status_owners) == 1
            and status_owners[0].char_start == status_start
            and status_owners[0].char_end == status_end
            and status_owners[0].render_mode == "literal_static"
        ):
            discarded.add(status_owners[0].draft_id)
            status_is_selected = True
        elif (
            len(status_owners) == 1
            and status_owners[0].char_start == status_start
            and status_owners[0].char_end == status_end
            and status_owners[0].render_mode == "deterministic_auxiliary"
            and status_owners[0].value_kind == "operational_text"
            and not status_owners[0].target_paths
        ):
            status_is_selected = True
        if not status_is_selected:
            continue
        if not status_owners or status_owners[0].render_mode == "literal_static":
            additions.append(
                SpanDraft(
                    draft_id=(
                        "host_shipped_on_board_status_"
                        + sha256_bytes(f"{status_start}\0{status_end}".encode())[:16]
                    ),
                    logical_key="agent:operational:shipped_on_board_status",
                    render_mode="deterministic_auxiliary",
                    value_kind="operational_text",
                    group_kind="transport",
                    group_key="transport:shipped_on_board_status",
                    target_paths=(),
                    derivation=None,
                    dependency_paths=(),
                    dependency_bindings=(),
                    char_start=status_start,
                    char_end=status_end,
                    source_text=raw[status_start:status_end],
                    evidence_origin="host_verified_agent_proposal",
                    render_policy="natural_text",
                    rationale=(
                        "Host classified this phrase as selected operational status because the "
                        "same populated summary line owns a vessel name followed by an on-board "
                        "date."
                    ),
                )
            )

        port_owners = tuple(row for row in working if _PORT_OF_LOADING_PATH in row.target_paths)
        port_logical_keys = {row.logical_key for row in port_owners}
        if not port_owners or len(port_logical_keys) != 1:
            continue
        port_value = _scalar_surface(_resolve_target_path(source_target, _PORT_OF_LOADING_PATH))
        if port_value is None:
            continue
        port_span = _at_location_on_frame(line=line, target_surface=port_value)
        if port_span is None or not (
            vessel.char_end <= port_span[0] < port_span[1] <= date_row.char_start
        ):
            continue
        overlaps = tuple(
            row for row in working if port_span[0] < row.char_end and row.char_start < port_span[1]
        )
        if overlaps:
            if not all(
                row.logical_key in port_logical_keys
                and row.char_start == port_span[0]
                and row.char_end == port_span[1]
                for row in overlaps
            ):
                continue
            continue
        owner = port_owners[0]
        additions.append(
            replace(
                owner,
                draft_id=(
                    "host_shipped_on_board_port_"
                    + sha256_bytes(f"{port_span[0]}\0{port_span[1]}".encode())[:16]
                ),
                char_start=port_span[0],
                char_end=port_span[1],
                source_text=raw[port_span[0] : port_span[1]],
                evidence_origin="host_verified_agent_proposal",
                rationale=(
                    owner.rationale
                    + " Host appended the unique AT <port-of-loading> ON occurrence from a "
                    "populated shipped-on-board summary."
                ),
            )
        )
    retained = tuple(row for row in working if row.draft_id not in discarded)
    return merge_drafts(retained, additions)


def normalize_shipped_on_board_vocabulary(
    *, raw: str, drafts: Sequence[SpanDraft]
) -> tuple[SpanDraft, ...]:
    """Preserve every otherwise-unowned shipped-on-board phrase as fixed form vocabulary.

    The phrase is invariant document wording whether it appears in a date caption, a legal clause,
    or a populated operational summary.  This normalization makes no claim about the surrounding
    line: it owns only the exact lexical phrase.  The stricter summary-locality normalizer may
    subsequently promote an exact literal owner to selected operational status when vessel and
    date evidence prove that interpretation.
    """

    additions: list[SpanDraft] = []
    for match in _SELECTED_OPERATIONAL_TEXT.finditer(raw):
        start, end = match.span()
        overlaps = tuple(row for row in drafts if start < row.char_end and row.char_start < end)
        if overlaps:
            continue
        source_text = raw[start:end]
        surface_key = sha256_bytes(source_text.encode("utf-8"))[:12]
        additions.append(
            SpanDraft(
                draft_id=(
                    "host_shipped_on_board_vocabulary_"
                    + sha256_bytes(f"{start}\0{end}".encode())[:16]
                ),
                logical_key=f"agent:literal:shipped_on_board:{surface_key}",
                render_mode="literal_static",
                value_kind="legal_text",
                group_kind="document",
                group_key="document:shipped_on_board_vocabulary",
                target_paths=(),
                derivation=None,
                dependency_paths=(),
                dependency_bindings=(),
                char_start=start,
                char_end=end,
                source_text=source_text,
                evidence_origin="host_verified_agent_proposal",
                render_policy="natural_text",
                rationale=(
                    "Host preserved only the exact shipped-on-board lexical phrase as fixed "
                    "document vocabulary without inferring semantics for its surrounding line."
                ),
            )
        )
    return merge_drafts(drafts, additions)


def normalize_literal_care_of_separators(
    *, raw: str, drafts: Sequence[SpanDraft]
) -> tuple[SpanDraft, ...]:
    """Preserve a C/O relation token bounded by two owned values in the same field group."""

    additions: list[SpanDraft] = []
    for line in line_spans(raw):
        for match in _CARE_OF_SEPARATOR.finditer(line.text):
            start = line.char_start + match.start()
            end = line.char_start + match.end()
            if any(start < row.char_end and row.char_start < end for row in drafts):
                continue
            left = tuple(
                row
                for row in drafts
                if line.char_start <= row.char_start < row.char_end <= start
                and not raw[row.char_end : start].strip()
            )
            right = tuple(
                row
                for row in drafts
                if end <= row.char_start < row.char_end <= line.char_end
                and not raw[end : row.char_start].strip()
            )
            if not left or not right:
                continue
            left_owner = max(left, key=lambda row: row.char_end)
            right_owner = min(right, key=lambda row: row.char_start)
            if (left_owner.group_kind, left_owner.group_key) != (
                right_owner.group_kind,
                right_owner.group_key,
            ):
                continue
            additions.append(
                SpanDraft(
                    draft_id=(
                        "host_literal_care_of_" + sha256_bytes(f"{start}\0{end}".encode())[:16]
                    ),
                    logical_key="agent:literal:care_of_relation",
                    render_mode="literal_static",
                    value_kind="legal_text",
                    group_kind=left_owner.group_kind,
                    group_key=left_owner.group_key,
                    target_paths=(),
                    derivation=None,
                    dependency_paths=(),
                    dependency_bindings=(),
                    char_start=start,
                    char_end=end,
                    source_text=raw[start:end],
                    evidence_origin="host_verified_agent_proposal",
                    render_policy="natural_text",
                    rationale=(
                        "Host preserved this exact C/O relation token because it is physically "
                        "bounded by two owned values in the same semantic field group."
                    ),
                )
            )
    return merge_drafts(drafts, additions)


def _trimmed_line_span(line: LineSpan) -> tuple[int, int] | None:
    left = len(line.text) - len(line.text.lstrip())
    right = len(line.text.rstrip())
    if right <= left:
        return None
    return line.char_start + left, line.char_start + right


def _line_token_sequence_spans(line: LineSpan, target_surface: str) -> tuple[tuple[int, int], ...]:
    target_tokens = tuple(token for token, _start, _end in _surface_token_spans(target_surface))
    source_tokens = _surface_token_spans(line.text)
    if not target_tokens or len(target_tokens) > len(source_tokens):
        return ()
    spans: list[tuple[int, int]] = []
    for index in range(len(source_tokens) - len(target_tokens) + 1):
        if (
            tuple(row[0] for row in source_tokens[index : index + len(target_tokens)])
            != target_tokens
        ):
            continue
        spans.append(
            (
                line.char_start + source_tokens[index][1],
                line.char_start + source_tokens[index + len(target_tokens) - 1][2],
            )
        )
    return tuple(spans)


def normalize_page_header_ownership(
    *, raw: str, drafts: Sequence[SpanDraft]
) -> tuple[SpanDraft, ...]:
    """Remove source-only mutable claims from synthetic ``--- PAGE n ---`` wrappers."""

    headers = tuple((match.start(), match.end()) for match in _PAGE_HEADER.finditer(raw))
    return tuple(
        row
        for row in drafts
        if not (
            row.render_mode
            in {
                "deterministic_auxiliary",
                "agent_residual",
                "literal_static",
            }
            and not row.target_paths
            and not row.dependency_paths
            and not row.dependency_bindings
            and any(start <= row.char_start and row.char_end <= end for start, end in headers)
        )
    )


def _unique_target_owner(*, drafts: Sequence[SpanDraft], target_path: str) -> SpanDraft | None:
    owners = tuple(
        row
        for row in drafts
        if row.render_mode == "target_binding" and row.target_paths == (target_path,)
    )
    if not owners or len({row.logical_key for row in owners}) != 1:
        return None
    return owners[0]


def _address_segments_in_lines(
    *, lines: Sequence[LineSpan], target_surface: str
) -> tuple[tuple[int, int], ...]:
    """Return a unique complete token partition of one address across contiguous lines."""

    target_tokens = tuple(token for token, _start, _end in _surface_token_spans(target_surface))
    if not target_tokens:
        return ()
    candidates: list[tuple[frozenset[int], tuple[int, int]]] = []
    for line in lines:
        source_tokens = _surface_token_spans(line.text)
        for source_index, (source_token, _start, _end) in enumerate(source_tokens):
            for target_index, target_token in enumerate(target_tokens):
                if source_token != target_token:
                    continue
                length = 0
                while (
                    source_index + length < len(source_tokens)
                    and target_index + length < len(target_tokens)
                    and source_tokens[source_index + length][0]
                    == target_tokens[target_index + length]
                ):
                    length += 1
                if not length:
                    continue
                can_extend_left = (
                    source_index > 0
                    and target_index > 0
                    and source_tokens[source_index - 1][0] == target_tokens[target_index - 1]
                )
                can_extend_right = (
                    source_index + length < len(source_tokens)
                    and target_index + length < len(target_tokens)
                    and source_tokens[source_index + length][0]
                    == target_tokens[target_index + length]
                )
                if can_extend_left or can_extend_right:
                    continue
                candidates.append(
                    (
                        frozenset(range(target_index, target_index + length)),
                        (
                            line.char_start + source_tokens[source_index][1],
                            line.char_start + source_tokens[source_index + length - 1][2],
                        ),
                    )
                )
    # Keep only maximal target-token runs. A complete address is accepted only when every target
    # token belongs to exactly one such run, making the physical partition unique.
    maximal = tuple(
        candidate
        for candidate in candidates
        if not any(
            candidate[0] < other[0]
            and candidate[1][0] >= other[1][0]
            and candidate[1][1] <= other[1][1]
            for other in candidates
        )
    )
    owners_by_token: dict[int, list[tuple[int, int]]] = defaultdict(list)
    for token_indexes, span in maximal:
        for token_index in token_indexes:
            owners_by_token[token_index].append(span)
    if set(owners_by_token) != set(range(len(target_tokens))) or any(
        len(set(spans)) != 1 for spans in owners_by_token.values()
    ):
        return ()
    selected = tuple(sorted({spans[0] for spans in owners_by_token.values()}))
    if any(left[1] > right[0] for left, right in pairwise(selected)):
        return ()
    return selected


def normalize_labeled_shipper_and_receipt_locality(
    *, raw: str, drafts: Sequence[SpanDraft], source_target: Mapping[str, Any]
) -> tuple[SpanDraft, ...]:
    """Reconstruct exact shipper-address and place-of-receipt roles under their captions.

    This handles repeated forms where equal city/state/postal tokens were anchored to the wrong
    role.  A rewrite is made only when the structured target has one existing direct owner and the
    caption-local OCR supplies a unique complete token partition (shipper address) or an exact
    whole value (place of receipt). Cross-boundary owners make the candidate ineligible.
    """

    lines = line_spans(raw)
    line_by_number = {line.number: line for line in lines}
    assignments: list[tuple[int, int, str, SpanDraft]] = []

    address_owner = _unique_target_owner(drafts=drafts, target_path=_SHIPPER_ADDRESS_PATH)
    try:
        address_target = _scalar_surface(_resolve_target_path(source_target, _SHIPPER_ADDRESS_PATH))
    except ValueError:
        address_target = None
    if address_owner is not None and address_target:
        for caption in lines:
            if not _normalized_surface(caption.text).startswith(_SHIPPER_NAME_ADDRESS_CAPTION):
                continue
            block: list[LineSpan] = []
            cursor = caption.number + 1
            while (line := line_by_number.get(cursor)) is not None and line.text.strip():
                if _PAGE_HEADER.fullmatch(line.text):
                    break
                block.append(line)
                cursor += 1
            for start, end in _address_segments_in_lines(
                lines=block,
                target_surface=address_target,
            ):
                assignments.append((start, end, _SHIPPER_ADDRESS_PATH, address_owner))

    city_owner = _unique_target_owner(drafts=drafts, target_path=_SHIPPER_CITY_PATH)
    try:
        city_target = _scalar_surface(_resolve_target_path(source_target, _SHIPPER_CITY_PATH))
    except ValueError:
        city_target = None
    if city_owner is not None and city_target:
        for caption in lines:
            if not _normalized_surface(caption.text).startswith(_SHIPPER_NAME_ADDRESS_CAPTION):
                continue
            cursor = caption.number + 1
            while (line := line_by_number.get(cursor)) is not None and line.text.strip():
                for start, end in _line_token_sequence_spans(line, city_target):
                    assignments.append((start, end, _SHIPPER_CITY_PATH, city_owner))
                cursor += 1

    name_owner = _unique_target_owner(drafts=drafts, target_path=_SHIPPER_NAME_PATH)
    try:
        name_target = _scalar_surface(_resolve_target_path(source_target, _SHIPPER_NAME_PATH))
    except ValueError:
        name_target = None
    if name_owner is not None and name_target:
        normalized_name = _normalized_surface(name_target)
        for caption in lines:
            if not _normalized_surface(caption.text).startswith(_SHIPPER_NAME_ADDRESS_CAPTION):
                continue
            value_line = line_by_number.get(caption.number + 1)
            span = _trimmed_line_span(value_line) if value_line is not None else None
            if span is not None and _normalized_surface(raw[span[0] : span[1]]) == normalized_name:
                assignments.append((span[0], span[1], _SHIPPER_NAME_PATH, name_owner))

    receipt_owner = _unique_target_owner(drafts=drafts, target_path=_PLACE_OF_RECEIPT_PATH)
    try:
        receipt_target = _scalar_surface(
            _resolve_target_path(source_target, _PLACE_OF_RECEIPT_PATH)
        )
    except ValueError:
        receipt_target = None
    if receipt_owner is not None and receipt_target:
        normalized_target = _normalized_surface(receipt_target)
        for caption in lines:
            caption_surface = _normalized_surface(caption.text)
            if not (
                caption_surface.startswith("placeofreceipt")
                and "combinedtransport" in caption_surface
            ):
                continue
            value_line = line_by_number.get(caption.number + 1)
            span = _trimmed_line_span(value_line) if value_line is not None else None
            if (
                span is not None
                and _normalized_surface(raw[span[0] : span[1]]) == normalized_target
            ):
                assignments.append((span[0], span[1], _PLACE_OF_RECEIPT_PATH, receipt_owner))

    if not assignments:
        return tuple(drafts)
    assignments = sorted(set(assignments), key=lambda row: (row[0], row[1], row[2]))
    accepted: list[tuple[int, int, str, SpanDraft]] = []
    for start, end, target_path, owner in assignments:
        if any(start < prior_end and prior_start < end for prior_start, prior_end, *_ in accepted):
            continue
        overlaps = tuple(row for row in drafts if start < row.char_end and row.char_start < end)
        if len(overlaps) == 1 and (
            overlaps[0].char_start,
            overlaps[0].char_end,
            overlaps[0].render_mode,
            overlaps[0].target_paths,
            overlaps[0].logical_key,
        ) == (start, end, "target_binding", (target_path,), owner.logical_key):
            continue
        accepted.append((start, end, target_path, owner))
    if not accepted:
        return tuple(drafts)

    def row_tokens_are_reassigned(row: SpanDraft) -> bool:
        token_spans = tuple(
            (row.char_start + token_start, row.char_start + token_end)
            for _token, token_start, token_end in _surface_token_spans(row.source_text)
        )
        return bool(token_spans) and all(
            any(start <= token_start and token_end <= end for start, end, *_ in accepted)
            for token_start, token_end in token_spans
        )

    unsafe_assignments = {
        (start, end, target_path)
        for start, end, target_path, _owner in accepted
        if any(
            start < row.char_end
            and row.char_start < end
            and not (start <= row.char_start and row.char_end <= end)
            and not row_tokens_are_reassigned(row)
            for row in drafts
        )
    }
    accepted = [row for row in accepted if (row[0], row[1], row[2]) not in unsafe_assignments]
    if not accepted:
        return tuple(drafts)

    discarded = {
        row.draft_id
        for row in drafts
        if row_tokens_are_reassigned(row)
        or any(
            start <= row.char_start and row.char_end <= end
            for start, end, _target_path, _owner in accepted
        )
    }
    additions = tuple(
        replace(
            owner,
            draft_id=(
                "host_caption_local_target_"
                + sha256_bytes(f"{target_path}\0{start}\0{end}".encode())[:16]
            ),
            char_start=start,
            char_end=end,
            source_text=raw[start:end],
            evidence_origin="host_verified_agent_proposal",
            rationale=(
                owner.rationale
                + " Host reassigned this exact value segment from the explicit local field "
                "caption and complete structured-target token coverage."
            ),
        )
        for start, end, target_path, owner in accepted
    )
    return reconcile_draft_overlaps(
        raw=raw,
        drafts=(*tuple(row for row in drafts if row.draft_id not in discarded), *additions),
        source_target=source_target,
    )


def normalize_party_location_repeats(
    *, raw: str, drafts: Sequence[SpanDraft], source_target: Mapping[str, Any]
) -> tuple[SpanDraft, ...]:
    """Append a party city/country repeat only beside another field of the same party."""

    lines = line_spans(raw)
    additions: list[SpanDraft] = []
    owners_by_path: dict[str, list[SpanDraft]] = defaultdict(list)
    party_contexts: dict[tuple[str, int], set[str]] = defaultdict(set)
    for row in drafts:
        if row.group_kind == "party" and (
            row.target_paths
            or (
                row.render_mode == "deterministic_auxiliary"
                and row.value_kind == "identifier"
                and not row.dependency_paths
                and not row.dependency_bindings
            )
        ):
            start_line = line_number_for_char(lines, row.char_start)
            end_line = line_number_for_char(lines, row.char_end - 1)
            for line_number in range(start_line, end_line + 1):
                party_contexts[(row.group_key, line_number)].add(row.logical_key)
        for path in row.target_paths:
            if _PARTY_LOCATION_PATH.fullmatch(path):
                owners_by_path[path].append(row)
    for path, owners in owners_by_path.items():
        if len({row.logical_key for row in owners}) != 1 or any(
            row.render_mode != "target_binding" or row.target_paths != (path,) for row in owners
        ):
            continue
        target_surface = _scalar_surface(_resolve_target_path(source_target, path))
        if not target_surface:
            continue
        owner = owners[0]
        for line in lines:
            same_party_context = bool(
                party_contexts.get((owner.group_key, line.number), set()) - {owner.logical_key}
            )
            if not same_party_context:
                continue
            for start, end in _line_token_sequence_spans(line, target_surface):
                if any(start < row.char_end and row.char_start < end for row in drafts):
                    continue
                additions.append(
                    replace(
                        owner,
                        draft_id=(
                            "host_party_location_repeat_"
                            + sha256_bytes(f"{path}\0{start}\0{end}".encode())[:16]
                        ),
                        char_start=start,
                        char_end=end,
                        source_text=raw[start:end],
                        evidence_origin="host_verified_agent_proposal",
                        rationale=(
                            owner.rationale
                            + " Host appended this city/country repeat from a line already "
                            "scoped by another target-backed field of the same party."
                        ),
                    )
                )
    return merge_drafts(drafts, additions)


def normalize_missing_party_locations_from_entity_blocks(
    *, raw: str, drafts: Sequence[SpanDraft], source_target: Mapping[str, Any]
) -> tuple[SpanDraft, ...]:
    """Recover unowned party address/city/country values from proven party blocks.

    The rule is structural: a blank-line-delimited block must already contain a name or address
    owner for the same party role, and that block must contain exactly one unowned token sequence
    for the structured location scalar. Competing role claims for one physical span are rejected.
    """

    lines = line_spans(raw)
    block_by_line: dict[int, int] = {}
    block_index = -1
    inside_block = False
    for line in lines:
        if line.text.strip():
            if not inside_block:
                block_index += 1
                inside_block = True
            block_by_line[line.number] = block_index
        else:
            inside_block = False

    context_blocks: dict[str, set[int]] = defaultdict(set)
    for row in drafts:
        if row.group_kind != "party" or not any(
            path.endswith((".name", ".address")) for path in row.target_paths
        ):
            continue
        start_line = line_number_for_char(lines, row.char_start)
        end_line = line_number_for_char(lines, row.char_end - 1)
        context_blocks[row.group_key].update(
            block
            for number in range(start_line, end_line + 1)
            if (block := block_by_line.get(number)) is not None
        )

    location_values: dict[str, str] = {}

    def visit(value: Any, path: str) -> None:
        if isinstance(value, Mapping):
            for key, child in value.items():
                visit(child, f"{path}.{key}")
        elif isinstance(value, list):
            for index, child in enumerate(value):
                visit(child, f"{path}[{index}]")
        elif isinstance(value, str) and _PARTY_BLOCK_RECOVERABLE_PATH.fullmatch(path):
            location_values[path] = value

    parties = source_target.get("documentPatch")
    if isinstance(parties, Mapping) and isinstance(parties.get("parties"), Mapping):
        visit(parties["parties"], "documentPatch.parties")

    owned_paths = {path for row in drafts for path in row.target_paths}
    proposed: list[tuple[str, int, int]] = []
    for path, target_surface in sorted(location_values.items()):
        if path in owned_paths:
            continue
        group_kind, group_key = _canonical_target_group((path,))
        if group_kind != "party" or not context_blocks.get(group_key):
            continue
        spans_by_block: dict[int, set[tuple[int, int]]] = defaultdict(set)
        for line in lines:
            block = block_by_line.get(line.number)
            if block not in context_blocks[group_key]:
                continue
            for start, end in _line_token_sequence_spans(line, target_surface):
                if any(start < row.char_end and row.char_start < end for row in drafts):
                    continue
                spans_by_block[block].add((start, end))
        if set(spans_by_block) != context_blocks[group_key] or any(
            len(spans) != 1 for spans in spans_by_block.values()
        ):
            continue
        proposed.extend((path, *next(iter(spans))) for spans in spans_by_block.values())

    paths_by_span: dict[tuple[int, int], set[str]] = defaultdict(set)
    for path, start, end in proposed:
        paths_by_span[(start, end)].add(path)
    additions: list[SpanDraft] = []
    for path, start, end in proposed:
        if len(paths_by_span[(start, end)]) != 1:
            continue
        group_kind, group_key = _canonical_target_group((path,))
        value_kind = _value_kind((path,), "natural_text")
        additions.append(
            SpanDraft(
                draft_id=(
                    "host_party_block_location_"
                    + sha256_bytes(f"{path}\0{start}\0{end}".encode())[:16]
                ),
                logical_key="anchor:" + path,
                render_mode="target_binding",
                value_kind=value_kind,
                group_kind=group_kind,
                group_key=group_key,
                target_paths=(path,),
                derivation=None,
                dependency_paths=(),
                dependency_bindings=(),
                char_start=start,
                char_end=end,
                source_text=raw[start:end],
                evidence_origin="host_verified_agent_proposal",
                render_policy=_render_policy_for_value_kind(value_kind, "target_binding"),
                rationale=(
                    "Host recovered this party field from a unique value in every "
                    "blank-line entity block already proven by the same party's name/address."
                ),
            )
        )
    return merge_drafts(drafts, additions)


def normalize_source_only_party_postal_city_suffixes(
    *, raw: str, drafts: Sequence[SpanDraft]
) -> tuple[SpanDraft, ...]:
    """Own an unmodeled city suffix beside a proven source-only party postal code."""

    lines = line_spans(raw)
    additions: list[SpanDraft] = []
    for postal in drafts:
        if not (
            postal.group_kind == "party"
            and postal.render_mode == "deterministic_auxiliary"
            and postal.value_kind == "identifier"
            and not postal.target_paths
            and not postal.dependency_paths
            and not postal.dependency_bindings
            and re.fullmatch(r"[0-9]{4,10}(?:-[0-9]+)?", postal.source_text.strip())
        ):
            continue
        line_number = line_number_for_char(lines, postal.char_start)
        if line_number != line_number_for_char(lines, postal.char_end - 1):
            continue
        line = lines[line_number - 1]
        if any(character.isalnum() for character in raw[line.char_start : postal.char_start]):
            continue
        suffix = raw[postal.char_end : line.char_end]
        match = re.fullmatch(r"\s+(?P<value>[A-Za-z][A-Za-z .'-]*[A-Za-z])\s*", suffix)
        if match is None:
            continue
        same_party_context = any(
            other.draft_id != postal.draft_id
            and other.group_kind == "party"
            and other.group_key == postal.group_key
            and abs(line_number_for_char(lines, other.char_start) - line_number) <= 2
            for other in drafts
        )
        if not same_party_context:
            continue
        start = postal.char_end + match.start("value")
        end = postal.char_end + match.end("value")
        if any(start < other.char_end and other.char_start < end for other in drafts):
            continue
        additions.append(
            SpanDraft(
                draft_id=(
                    "host_party_postal_city_"
                    + sha256_bytes(f"{postal.group_key}\0{start}\0{end}".encode())[:16]
                ),
                logical_key=(
                    "agent:party:postal_city:"
                    + sha256_bytes(
                        f"{postal.group_key}\0{_normalized_surface(raw[start:end])}".encode()
                    )[:16]
                ),
                render_mode="deterministic_auxiliary",
                value_kind="location",
                group_kind="party",
                group_key=postal.group_key,
                target_paths=(),
                derivation=None,
                dependency_paths=(),
                dependency_bindings=(),
                char_start=start,
                char_end=end,
                source_text=raw[start:end],
                evidence_origin="host_verified_agent_proposal",
                render_policy="natural_text",
                rationale=(
                    "Host owned this source-only city because it is the sole alphabetic suffix "
                    "beside a typed postal-code owner inside the same proven party block."
                ),
            )
        )
    return merge_drafts(drafts, additions)


def normalize_repeated_party_location_variants(
    *, raw: str, drafts: Sequence[SpanDraft]
) -> tuple[SpanDraft, ...]:
    """Own repeated ``CITY, CC`` lines bracketed by the same proven party scope."""

    lines = line_spans(raw)
    party_groups_by_line: dict[int, set[str]] = defaultdict(set)
    for row in drafts:
        if row.group_kind != "party":
            continue
        start_line = line_number_for_char(lines, row.char_start)
        end_line = line_number_for_char(lines, row.char_end - 1)
        for line_number in range(start_line, end_line + 1):
            party_groups_by_line[line_number].add(row.group_key)

    candidates: dict[tuple[str, str], list[tuple[int, int]]] = defaultdict(list)
    for line in lines:
        match = _PARTY_CITY_COUNTRY_VARIANT.fullmatch(line.text)
        if match is None:
            continue
        start = line.char_start + match.start("value")
        end = line.char_start + match.end("value")
        if any(start < row.char_end and row.char_start < end for row in drafts):
            continue
        bracketing_groups = party_groups_by_line.get(line.number - 1, set()).intersection(
            party_groups_by_line.get(line.number + 1, set())
        )
        if len(bracketing_groups) != 1:
            continue
        group_key = next(iter(bracketing_groups))
        candidates[(_normalized_surface(raw[start:end]), group_key)].append((start, end))

    additions: list[SpanDraft] = []
    for (surface, group_key), spans in candidates.items():
        if len(spans) < 2:
            continue
        logical_key = (
            "agent:party:location_variant:" + sha256_bytes(f"{group_key}\0{surface}".encode())[:16]
        )
        for start, end in spans:
            additions.append(
                SpanDraft(
                    draft_id=(
                        "host_party_location_variant_"
                        + sha256_bytes(f"{group_key}\0{start}\0{end}".encode())[:16]
                    ),
                    logical_key=logical_key,
                    render_mode="deterministic_auxiliary",
                    value_kind="location",
                    group_kind="party",
                    group_key=group_key,
                    target_paths=(),
                    derivation=None,
                    dependency_paths=(),
                    dependency_bindings=(),
                    char_start=start,
                    char_end=end,
                    source_text=raw[start:end],
                    evidence_origin="host_verified_agent_proposal",
                    render_policy="natural_text",
                    rationale=(
                        "Host owned this repeated CITY, country-code location line only after "
                        "the immediately preceding and following rows proved one party scope."
                    ),
                )
            )
    return merge_drafts(drafts, additions)


def normalize_exported_country_repeats(
    *, raw: str, drafts: Sequence[SpanDraft]
) -> tuple[SpanDraft, ...]:
    """Attach country repeats in explicit ``commodities were exported from`` statements."""

    lines = line_spans(raw)
    labeled_by_surface: dict[str, list[SpanDraft]] = defaultdict(list)
    for row in drafts:
        if not (
            row.render_mode == "deterministic_auxiliary"
            and row.value_kind == "location"
            and not row.target_paths
        ):
            continue
        line = lines[line_number_for_char(lines, row.char_start) - 1]
        match = _LABELED_COUNTRY_VALUE.fullmatch(line.text)
        if match is not None and _normalized_surface(match.group("value")) == _normalized_surface(
            row.source_text
        ):
            labeled_by_surface[_normalized_surface(row.source_text)].append(row)
    additions: list[SpanDraft] = []
    for normalized_country, owners in labeled_by_surface.items():
        if len({row.logical_key for row in owners}) != 1:
            continue
        owner = owners[0]
        for line in lines:
            for start, end in _line_token_sequence_spans(line, owner.source_text):
                prefix = raw[line.char_start : start]
                if not _EXPORTED_FROM_COUNTRY_CONTEXT.search(prefix):
                    continue
                if _normalized_surface(raw[start:end]) != normalized_country or any(
                    start < row.char_end and row.char_start < end for row in drafts
                ):
                    continue
                additions.append(
                    replace(
                        owner,
                        draft_id=(
                            "host_export_country_repeat_"
                            + sha256_bytes(f"{owner.logical_key}\0{start}\0{end}".encode())[:16]
                        ),
                        char_start=start,
                        char_end=end,
                        source_text=raw[start:end],
                        evidence_origin="host_verified_agent_proposal",
                        rationale=(
                            owner.rationale
                            + " Host appended this country repeat from the explicit commodities-"
                            "exported-from statement."
                        ),
                    )
                )
    return merge_drafts(drafts, additions)


def normalize_labeled_country_code_derivations(
    *, raw: str, drafts: Sequence[SpanDraft]
) -> tuple[SpanDraft, ...]:
    """Derive labeled two-letter country codes from one same-scope country value."""

    lines = line_spans(raw)
    grouped: dict[str, list[SpanDraft]] = defaultdict(list)
    for row in drafts:
        grouped[row.logical_key].append(row)
    country_groups: dict[tuple[str, str], list[str]] = defaultdict(list)
    country_paths: dict[tuple[str, str], list[str]] = defaultdict(list)
    code_keys: list[str] = []
    for logical_key, rows in grouped.items():
        first = rows[0]
        scope = (first.group_kind, first.group_key)
        direct_country_paths = tuple(
            path for path in first.target_paths if path.endswith(".country")
        )
        if (
            first.render_mode in {"target_binding", "carrier_static"}
            and len(first.target_paths) == 1
            and len(direct_country_paths) == 1
        ):
            country_paths[scope].append(direct_country_paths[0])
        if not (
            first.render_mode == "deterministic_auxiliary"
            and not first.target_paths
            and not first.dependency_paths
            and not first.dependency_bindings
        ):
            continue
        every_country = all(
            (
                (
                    match := _LABELED_COUNTRY_VALUE.fullmatch(
                        lines[line_number_for_char(lines, row.char_start) - 1].text
                    )
                )
                is not None
                and _normalized_surface(match.group("value"))
                == _normalized_surface(row.source_text)
            )
            for row in rows
        )
        every_code = all(
            (
                (
                    match := _LABELED_COUNTRY_CODE_VALUE.fullmatch(
                        lines[line_number_for_char(lines, row.char_start) - 1].text
                    )
                )
                is not None
                and match.group("value").casefold() == row.source_text.casefold()
            )
            for row in rows
        )
        if every_country and first.value_kind == "location":
            country_groups[scope].append(logical_key)
        if every_code:
            code_keys.append(logical_key)

    replacements: dict[str, SpanDraft] = {}
    for code_key in code_keys:
        rows = grouped[code_key]
        first = rows[0]
        scope = (first.group_kind, first.group_key)
        binding_dependencies = tuple(dict.fromkeys(country_groups.get(scope, [])))
        path_dependencies = tuple(dict.fromkeys(country_paths.get(scope, [])))
        if len(binding_dependencies) + len(path_dependencies) != 1:
            continue
        for row in rows:
            replacements[row.draft_id] = replace(
                row,
                render_mode="deterministic_derived",
                value_kind="identifier",
                derivation="country_code",
                dependency_paths=path_dependencies,
                dependency_bindings=binding_dependencies,
                render_policy="derived_surface",
                rationale=(
                    row.rationale
                    + " Host derived this explicitly labeled two-letter country code from the "
                    "single country dependency in the same semantic scope."
                ),
            )
    return merge_drafts(replacements.get(row.draft_id, row) for row in drafts)


def normalize_selected_measurement_units(
    *, raw: str, drafts: Sequence[SpanDraft]
) -> tuple[SpanDraft, ...]:
    """Own repeated standalone mass units in one deterministic unit-vocabulary binding."""

    lines = line_spans(raw)
    candidates: dict[str, list[tuple[int, int]]] = defaultdict(list)
    for line in lines:
        for token, relative_start, relative_end in _surface_token_spans(line.text):
            surface = line.text[relative_start:relative_end]
            if _SELECTED_MEASUREMENT_UNIT.fullmatch(surface) is None:
                continue
            prefix = line.text[:relative_start]
            if not (
                re.search(r"(?i)\bAS\s+PER\s*$", prefix)
                or re.search(r"(?i)\bPACKAGES?\s*$", prefix)
            ):
                continue
            start = line.char_start + relative_start
            end = line.char_start + relative_end
            candidates[token].append((start, end))
    additions: list[SpanDraft] = []
    replaced_ids: set[str] = set()
    for unit, spans in sorted(candidates.items()):
        if len(spans) < 2:
            continue
        logical_key = f"agent:cargo:measurement_unit:{unit}"
        existing_by_span: dict[tuple[int, int], SpanDraft | None] = {}
        compatible = True
        for start, end in spans:
            overlaps = tuple(row for row in drafts if start < row.char_end and row.char_start < end)
            if not overlaps:
                existing_by_span[(start, end)] = None
                continue
            if not (
                len(overlaps) == 1
                and overlaps[0].char_start == start
                and overlaps[0].char_end == end
                and overlaps[0].render_mode == "deterministic_auxiliary"
                and overlaps[0].derivation is None
                and not overlaps[0].target_paths
                and not overlaps[0].dependency_paths
                and not overlaps[0].dependency_bindings
            ):
                compatible = False
                break
            existing_by_span[(start, end)] = overlaps[0]
        if not compatible:
            continue
        for start, end in spans:
            existing = existing_by_span[(start, end)]
            if existing is not None:
                replaced_ids.add(existing.draft_id)
                canonical = (
                    existing.logical_key == logical_key
                    and existing.value_kind == "other_text"
                    and existing.group_kind == "cargo"
                    and existing.group_key == "cargo:measurement_unit"
                    and existing.render_policy == "natural_text"
                )
                additions.append(
                    existing
                    if canonical
                    else replace(
                        existing,
                        logical_key=logical_key,
                        value_kind="other_text",
                        group_kind="cargo",
                        group_key="cargo:measurement_unit",
                        render_policy="natural_text",
                        rationale=(
                            existing.rationale
                            + " Host joined identical selected mass-unit tokens into one "
                            "deterministic vocabulary binding across detail and total rows."
                        ),
                    )
                )
                continue
            additions.append(
                SpanDraft(
                    draft_id=(
                        "host_measurement_unit_"
                        + sha256_bytes(f"{logical_key}\0{start}\0{end}".encode())[:16]
                    ),
                    logical_key=logical_key,
                    render_mode="deterministic_auxiliary",
                    value_kind="other_text",
                    group_kind="cargo",
                    group_key="cargo:measurement_unit",
                    target_paths=(),
                    derivation=None,
                    dependency_paths=(),
                    dependency_bindings=(),
                    char_start=start,
                    char_end=end,
                    source_text=raw[start:end],
                    evidence_origin="host_verified_agent_proposal",
                    render_policy="natural_text",
                    rationale=(
                        "Host owned this repeated standalone measurement unit from explicit "
                        "AS PER or package-measurement context."
                    ),
                )
            )
    retained = tuple(row for row in drafts if row.draft_id not in replaced_ids)
    return merge_drafts(retained, additions)


def normalize_adjacent_source_only_measurement_units(
    *, raw: str, drafts: Sequence[SpanDraft]
) -> tuple[SpanDraft, ...]:
    """Promote a literal unit mechanically paired with a source-only numeric value.

    A structured total and its detail rows can share the same printed unit. The structured unit
    owner must remain local to the total, while units following independently synthesized detail
    values remain source-only deterministic vocabulary. This rule requires an exact known unit,
    an immediately preceding source-only numeric owner on the same line, and identical cargo
    scope; it does not infer a unit from nearby prose.
    """

    known_units = set().union(*_MEASUREMENT_UNIT_SURFACES.values())
    lines = line_spans(raw)
    replacements: dict[str, SpanDraft] = {}
    for unit in drafts:
        normalized_unit = _normalized_surface(unit.source_text)
        if not (
            unit.render_mode == "literal_static"
            and unit.group_kind == "cargo"
            and normalized_unit in known_units
            and not unit.target_paths
            and not unit.dependency_paths
            and not unit.dependency_bindings
        ):
            continue
        if (
            _adjacent_source_only_numeric_owner(
                raw=raw,
                lines=lines,
                drafts=drafts,
                unit=unit,
            )
            is None
        ):
            continue
        replacements[unit.draft_id] = replace(
            unit,
            logical_key=f"agent:cargo:measurement_unit:{normalized_unit}",
            render_mode="deterministic_auxiliary",
            value_kind="other_text",
            group_key=f"cargo:measurement_unit:{normalized_unit}",
            evidence_origin="host_verified_agent_proposal",
            render_policy="natural_text",
            rationale=(
                unit.rationale
                + " Host classified this exact unit as deterministic source-only vocabulary "
                "because it immediately follows one independently mutable numeric cargo value."
            ),
        )
    return merge_drafts(replacements.get(row.draft_id, row) for row in drafts)


def _split_labeled_route_equality_bindings(
    *, raw: str, drafts: Sequence[SpanDraft], source_target: Mapping[str, Any]
) -> tuple[SpanDraft, ...]:
    """Split equal-valued route facts only when explicit captions prove every owner.

    Value-oriented anchors can legitimately begin with one occurrence owning several currently
    equal route fields.  If a compiler then appends each separately printed value to that shared
    owner, descendant values would remain incorrectly coupled.  The host may separate the facts
    without semantic inference only when every physical occurrence in the shared owner is the
    exact next-line value of an explicit supported route caption, every target fact has at least
    one such occurrence, and the caption-derived occurrence sets are disjoint and exhaustive.
    Any missing, ambiguous, or extra occurrence leaves the binding untouched for agent review.
    """

    lines = line_spans(raw)
    line_by_number = {line.number: line for line in lines}
    captions_by_path: dict[str, tuple[str, ...]] = defaultdict(tuple)
    for caption_prefix, target_path in _LABELED_ROUTE_TARGETS.items():
        captions_by_path[target_path] = (*captions_by_path[target_path], caption_prefix)

    grouped: dict[str, list[SpanDraft]] = defaultdict(list)
    for draft in drafts:
        grouped[draft.logical_key].append(draft)

    replaced_keys: set[str] = set()
    replacements: list[SpanDraft] = []
    for logical_key, rows in grouped.items():
        first = rows[0]
        target_paths = first.target_paths
        if not (
            len(rows) >= 2
            and first.render_mode == "target_binding"
            and first.value_kind == "location"
            and first.group_kind == "route"
            and len(target_paths) >= 2
            and all(row.target_paths == target_paths for row in rows)
            and all(path in captions_by_path for path in target_paths)
            and not first.dependency_paths
            and not first.dependency_bindings
            and first.derivation is None
        ):
            continue
        components = target_fact_components(source_target, target_paths)
        if len(components) != len(target_paths) or any(
            len(component) != 1 for component in components
        ):
            continue
        if any(
            row.logical_key != logical_key and set(row.target_paths).intersection(target_paths)
            for row in drafts
        ):
            continue

        current_by_span = {(row.char_start, row.char_end): row for row in rows}
        if len(current_by_span) != len(rows):
            continue
        spans_by_path: dict[str, set[tuple[int, int]]] = {}
        for target_path in target_paths:
            try:
                target_surface = _scalar_surface(_resolve_target_path(source_target, target_path))
            except ValueError:
                break
            if target_surface is None:
                break
            normalized_target = _normalized_surface(target_surface)
            if not normalized_target:
                break
            matching_spans: set[tuple[int, int]] = set()
            for caption in lines:
                normalized_caption = re.sub(r"^[0-9]+", "", _normalized_surface(caption.text))
                if not any(
                    normalized_caption.startswith(prefix)
                    for prefix in captions_by_path[target_path]
                ):
                    continue
                value_line = line_by_number.get(caption.number + 1)
                span = _trimmed_line_span(value_line) if value_line is not None else None
                if (
                    span is not None
                    and _normalized_surface(raw[span[0] : span[1]]) == normalized_target
                ):
                    matching_spans.add(span)
            if not matching_spans:
                break
            spans_by_path[target_path] = matching_spans
        if len(spans_by_path) != len(target_paths):
            continue
        all_caption_spans = set().union(*spans_by_path.values())
        if any(
            left_path != right_path and left_spans.intersection(right_spans)
            for left_path, left_spans in spans_by_path.items()
            for right_path, right_spans in spans_by_path.items()
        ) or all_caption_spans != set(current_by_span):
            continue

        replaced_keys.add(logical_key)
        for target_path, spans in spans_by_path.items():
            group_kind, group_key = _canonical_target_group((target_path,))
            for start, end in sorted(spans):
                source_row = current_by_span[(start, end)]
                replacements.append(
                    replace(
                        source_row,
                        draft_id=(
                            "host_labeled_route_split_"
                            + sha256_bytes(f"{target_path}\0{start}\0{end}".encode())[:16]
                        ),
                        logical_key="anchor:" + target_path,
                        group_kind=group_kind,
                        group_key=group_key,
                        target_paths=(target_path,),
                        rationale=(
                            source_row.rationale
                            + " Host separated the shared equal-valued route owner because "
                            "explicit local captions provide a disjoint and exhaustive physical "
                            f"occurrence set for {target_path}."
                        ),
                    )
                )
    retained = tuple(row for row in drafts if row.logical_key not in replaced_keys)
    return merge_drafts(retained, replacements)


def normalize_labeled_route_target_repeats(
    *, raw: str, drafts: Sequence[SpanDraft], source_target: Mapping[str, Any]
) -> tuple[SpanDraft, ...]:
    """Append exact next-line location values under explicit field captions."""

    drafts = _split_labeled_route_equality_bindings(
        raw=raw,
        drafts=drafts,
        source_target=source_target,
    )
    lines = line_spans(raw)
    line_by_number = {line.number: line for line in lines}
    additions: list[SpanDraft] = []
    for caption_prefix, target_path in _LABELED_NEXT_LINE_TARGETS.items():
        owner = _unique_target_owner(drafts=drafts, target_path=target_path)
        if owner is None:
            continue
        try:
            target_surface = _scalar_surface(_resolve_target_path(source_target, target_path))
        except ValueError:
            continue
        if not target_surface:
            continue
        normalized_target = _normalized_surface(target_surface)
        for caption in lines:
            normalized_caption = re.sub(r"^[0-9]+", "", _normalized_surface(caption.text))
            if not normalized_caption.startswith(caption_prefix):
                continue
            value_line = line_by_number.get(caption.number + 1)
            span = _trimmed_line_span(value_line) if value_line is not None else None
            if (
                span is None
                or _normalized_surface(raw[span[0] : span[1]]) != normalized_target
                or any(span[0] < row.char_end and row.char_start < span[1] for row in drafts)
            ):
                continue
            additions.append(
                replace(
                    owner,
                    draft_id=(
                        "host_labeled_route_repeat_"
                        + sha256_bytes(f"{target_path}\0{span[0]}\0{span[1]}".encode())[:16]
                    ),
                    char_start=span[0],
                    char_end=span[1],
                    source_text=raw[span[0] : span[1]],
                    evidence_origin="host_verified_agent_proposal",
                    rationale=(
                        owner.rationale
                        + " Host appended this exact target value from the explicit local "
                        "field caption."
                    ),
                )
            )
    return merge_drafts(drafts, additions)


def normalize_labeled_vessel_voyage_repeats(
    *, raw: str, drafts: Sequence[SpanDraft], source_target: Mapping[str, Any]
) -> tuple[SpanDraft, ...]:
    """Split a next-line vessel/voyage value when the caption and targets prove both spans."""

    vessel_owner = _unique_target_owner(drafts=drafts, target_path=_VESSEL_NAME_PATH)
    voyage_owner = _unique_target_owner(drafts=drafts, target_path=_VOYAGE_NUMBER_PATH)
    if vessel_owner is None or voyage_owner is None:
        return tuple(drafts)
    try:
        vessel_surface = _scalar_surface(_resolve_target_path(source_target, _VESSEL_NAME_PATH))
        voyage_surface = _scalar_surface(_resolve_target_path(source_target, _VOYAGE_NUMBER_PATH))
    except ValueError:
        return tuple(drafts)
    if not vessel_surface or not voyage_surface:
        return tuple(drafts)

    lines = line_spans(raw)
    line_by_number = {line.number: line for line in lines}
    additions: list[SpanDraft] = []
    for caption in lines:
        normalized_caption = re.sub(r"^[0-9]+", "", _normalized_surface(caption.text))
        if "vessel" not in normalized_caption or "voy" not in normalized_caption:
            continue
        value_line = line_by_number.get(caption.number + 1)
        if value_line is None:
            continue
        vessel_spans = _line_token_sequence_spans(value_line, vessel_surface)
        voyage_spans = _line_token_sequence_spans(value_line, voyage_surface)
        if len(vessel_spans) != 1 or len(voyage_spans) != 1:
            continue
        vessel_span, voyage_span = vessel_spans[0], voyage_spans[0]
        if vessel_span[0] < voyage_span[1] and voyage_span[0] < vessel_span[1]:
            continue
        uncovered = raw[value_line.char_start : value_line.char_end]
        relative_owned = (
            (vessel_span[0] - value_line.char_start, vessel_span[1] - value_line.char_start),
            (voyage_span[0] - value_line.char_start, voyage_span[1] - value_line.char_start),
        )
        if any(
            character.isalnum() and not any(start <= index < end for start, end in relative_owned)
            for index, character in enumerate(uncovered)
        ):
            continue
        for target_path, owner, (start, end) in (
            (_VESSEL_NAME_PATH, vessel_owner, vessel_span),
            (_VOYAGE_NUMBER_PATH, voyage_owner, voyage_span),
        ):
            overlaps = tuple(row for row in drafts if start < row.char_end and row.char_start < end)
            if overlaps:
                continue
            additions.append(
                replace(
                    owner,
                    draft_id=(
                        "host_labeled_vessel_voyage_repeat_"
                        + sha256_bytes(f"{target_path}\0{start}\0{end}".encode())[:16]
                    ),
                    char_start=start,
                    char_end=end,
                    source_text=raw[start:end],
                    evidence_origin="host_verified_agent_proposal",
                    rationale=(
                        owner.rationale
                        + " Host appended this exact target token from a complete next-line "
                        "vessel/voyage field."
                    ),
                )
            )
    return merge_drafts(drafts, additions)


def normalize_inline_labeled_transport_repeats(
    *, raw: str, drafts: Sequence[SpanDraft], source_target: Mapping[str, Any]
) -> tuple[SpanDraft, ...]:
    """Append exact route and transport repeats from closed inline label grammars.

    The line must match a complete supported label frame and each captured value must equal its
    structured target.  This prevents an equal string in prose or another role from being adopted
    merely because it appears near a transport word.
    """

    candidates: list[tuple[str, tuple[int, int], str]] = []
    for line in line_spans(raw):
        route_match = _INLINE_LABELED_ROUTE_VALUE.fullmatch(line.text)
        if route_match is not None:
            caption = _normalized_surface(route_match.group("caption"))
            target_path = _LABELED_ROUTE_TARGETS.get(caption)
            if target_path is not None:
                try:
                    target_surface = _scalar_surface(
                        _resolve_target_path(source_target, target_path)
                    )
                except ValueError:
                    target_surface = None
                if target_surface and _normalized_surface(
                    route_match.group("value")
                ) == _normalized_surface(target_surface):
                    candidates.append(
                        (
                            target_path,
                            (
                                line.char_start + route_match.start("value"),
                                line.char_start + route_match.end("value"),
                            ),
                            "inline route",
                        )
                    )

        transport_match = _INLINE_LABELED_VESSEL_VOYAGE.fullmatch(line.text)
        if transport_match is None:
            continue
        captured = (
            (_VESSEL_NAME_PATH, "vessel", "inline vessel"),
            (_VOYAGE_NUMBER_PATH, "voyage", "inline voyage"),
        )
        resolved: list[tuple[str, tuple[int, int], str]] = []
        for target_path, group_name, receipt_label in captured:
            try:
                target_surface = _scalar_surface(_resolve_target_path(source_target, target_path))
            except ValueError:
                break
            if not target_surface or _normalized_surface(
                transport_match.group(group_name)
            ) != _normalized_surface(target_surface):
                break
            resolved.append(
                (
                    target_path,
                    (
                        line.char_start + transport_match.start(group_name),
                        line.char_start + transport_match.end(group_name),
                    ),
                    receipt_label,
                )
            )
        if len(resolved) == len(captured):
            candidates.extend(resolved)

    additions: list[SpanDraft] = []
    for target_path, (start, end), receipt_label in candidates:
        owner = _unique_target_owner(drafts=drafts, target_path=target_path)
        if owner is None:
            continue
        overlaps = tuple(row for row in drafts if start < row.char_end and row.char_start < end)
        if overlaps:
            if all(
                row.logical_key == owner.logical_key
                and row.char_start == start
                and row.char_end == end
                for row in overlaps
            ):
                continue
            continue
        additions.append(
            replace(
                owner,
                draft_id=(
                    "host_inline_transport_repeat_"
                    + sha256_bytes(f"{target_path}\0{start}\0{end}".encode())[:16]
                ),
                char_start=start,
                char_end=end,
                source_text=raw[start:end],
                evidence_origin="host_verified_agent_proposal",
                rationale=(
                    owner.rationale
                    + f" Host appended this exact target value from the complete {receipt_label} "
                    "label frame."
                ),
            )
        )
    return merge_drafts(drafts, additions)


def normalize_selected_commercial_dates_and_charges(
    *, raw: str, drafts: Sequence[SpanDraft], source_target: Mapping[str, Any]
) -> tuple[SpanDraft, ...]:
    """Own exact inline proforma dates and selected origin/destination charge terms."""

    additions: list[SpanDraft] = []
    discarded_ids: set[str] = set()
    lines = line_spans(raw)
    line_by_number = {line.number: line for line in lines}
    shipped_date_owners = tuple(
        row
        for row in drafts
        if row.render_mode == "target_binding"
        and row.target_paths == (_SHIPPED_ON_BOARD_DATE_PATH,)
    )
    shipped_date_owner = (
        shipped_date_owners[0]
        if shipped_date_owners and len({row.logical_key for row in shipped_date_owners}) == 1
        else None
    )
    try:
        shipped_date_target = _resolve_target_path(source_target, _SHIPPED_ON_BOARD_DATE_PATH)
    except ValueError:
        shipped_date_target = None
    shipped_date_candidates = (
        _date_candidates(shipped_date_target)
        if isinstance(shipped_date_target, str)
        else frozenset()
    )
    if shipped_date_owner is not None and shipped_date_candidates:
        for caption in lines:
            if _LADEN_ON_BOARD_DATE_CAPTION.fullmatch(caption.text) is None:
                continue
            value_line = line_by_number.get(caption.number + 1)
            span = _trimmed_line_span(value_line) if value_line is not None else None
            if span is None or not (
                shipped_date_candidates & _date_candidates(raw[span[0] : span[1]])
            ):
                continue
            start, end = span
            overlaps = tuple(row for row in drafts if start < row.char_end and row.char_start < end)
            if len(overlaps) == 1 and (
                overlaps[0].logical_key,
                overlaps[0].target_paths,
                overlaps[0].char_start,
                overlaps[0].char_end,
            ) == (
                shipped_date_owner.logical_key,
                (_SHIPPED_ON_BOARD_DATE_PATH,),
                start,
                end,
            ):
                continue
            replaceable = all(
                not row.target_paths
                and not row.dependency_paths
                and not row.dependency_bindings
                and row.render_mode
                in {
                    "deterministic_auxiliary",
                    "agent_residual",
                    "literal_static",
                }
                for row in overlaps
            )
            if overlaps and not replaceable:
                continue
            discarded_ids.update(row.draft_id for row in overlaps)
            additions.append(
                replace(
                    shipped_date_owner,
                    draft_id=(
                        "host_laden_on_board_date_" + sha256_bytes(f"{start}\0{end}".encode())[:16]
                    ),
                    char_start=start,
                    char_end=end,
                    source_text=raw[start:end],
                    evidence_origin="host_verified_agent_proposal",
                    rationale=(
                        shipped_date_owner.rationale
                        + " Host assigned this exact date beneath the explicit DATE LADEN ON "
                        "BOARD caption to the structured shipped-on-board date."
                    ),
                )
            )
    for caption in lines:
        if _normalized_surface(caption.text) != "date":
            continue
        value_line = line_by_number.get(caption.number + 1)
        span = _trimmed_line_span(value_line) if value_line is not None else None
        if span is None or _DATE.fullmatch(raw[span[0] : span[1]]) is None:
            continue
        start, end = span
        overlaps = tuple(row for row in drafts if start < row.char_end and row.char_start < end)
        if any(
            row.logical_key == "agent:document:generic_date"
            and row.render_mode == "deterministic_auxiliary"
            and row.char_start == start
            and row.char_end == end
            for row in overlaps
        ):
            continue
        replaceable_shipped_date = (
            bool(overlaps)
            and all(
                row.render_mode == "target_binding"
                and row.target_paths == (_SHIPPED_ON_BOARD_DATE_PATH,)
                and row.char_start == start
                and row.char_end == end
                for row in overlaps
            )
            and any(
                row.render_mode == "target_binding"
                and row.target_paths == (_SHIPPED_ON_BOARD_DATE_PATH,)
                and row not in overlaps
                for row in drafts
            )
        )
        if overlaps and not replaceable_shipped_date:
            continue
        if replaceable_shipped_date:
            discarded_ids.update(row.draft_id for row in overlaps)
        additions.append(
            SpanDraft(
                draft_id=(
                    "host_generic_document_date_" + sha256_bytes(f"{start}\0{end}".encode())[:16]
                ),
                logical_key="agent:document:generic_date",
                render_mode="deterministic_auxiliary",
                value_kind="date",
                group_kind="document",
                group_key="document:date",
                target_paths=(),
                derivation=None,
                dependency_paths=(),
                dependency_bindings=(),
                char_start=start,
                char_end=end,
                source_text=raw[start:end],
                evidence_origin="host_verified_agent_proposal",
                render_policy="date_surface",
                rationale=(
                    "Host separated this source-only date under the exact generic DATE caption "
                    "from a retained role-specific shipped-on-board date."
                ),
            )
        )
    for match in _PROFORMA_INVOICE_DATE.finditer(raw):
        start, end = match.span("value")
        if any(start < row.char_end and row.char_start < end for row in drafts):
            continue
        additions.append(
            SpanDraft(
                draft_id=(
                    "host_proforma_invoice_date_" + sha256_bytes(f"{start}\0{end}".encode())[:16]
                ),
                logical_key="agent:commercial:proforma_invoice_date",
                render_mode="deterministic_auxiliary",
                value_kind="date",
                group_kind="commercial",
                group_key="commercial:invoice",
                target_paths=(),
                derivation=None,
                dependency_paths=(),
                dependency_bindings=(),
                char_start=start,
                char_end=end,
                source_text=raw[start:end],
                evidence_origin="host_verified_agent_proposal",
                render_policy="date_surface",
                rationale=(
                    "Host owned the exact date selected by the inline P/I NO. ... DD. caption."
                ),
            )
        )

    for line in lines:
        port_match = _PORT_CHARGE_SELECTION.fullmatch(line.text)
        if port_match is None:
            continue
        role = port_match.group("place").casefold()
        for field, value_kind in (("place", "location"), ("payment", "commercial_text")):
            start = line.char_start + port_match.start(field)
            end = line.char_start + port_match.end(field)
            logical_key = f"agent:commercial:{role}_port_charge_{field}"
            overlaps = tuple(row for row in drafts if start < row.char_end and row.char_start < end)
            if any(
                row.logical_key == logical_key
                and row.render_mode == "deterministic_auxiliary"
                and row.char_start == start
                and row.char_end == end
                for row in overlaps
            ):
                continue
            replaceable_freight_payment = (
                field == "payment"
                and bool(overlaps)
                and all(
                    row.render_mode == "target_binding"
                    and row.target_paths == (_FREIGHT_PAYMENT_PATH,)
                    and row.char_start == start
                    and row.char_end == end
                    for row in overlaps
                )
                and any(
                    row.render_mode == "target_binding"
                    and row.target_paths == (_FREIGHT_PAYMENT_PATH,)
                    and row not in overlaps
                    for row in drafts
                )
            )
            if overlaps and not replaceable_freight_payment:
                continue
            if replaceable_freight_payment:
                discarded_ids.update(row.draft_id for row in overlaps)
            additions.append(
                SpanDraft(
                    draft_id=(
                        "host_port_charge_term_"
                        + sha256_bytes(f"{role}\0{field}\0{start}\0{end}".encode())[:16]
                    ),
                    logical_key=logical_key,
                    render_mode="deterministic_auxiliary",
                    value_kind=value_kind,
                    group_kind="commercial",
                    group_key="commercial:port_charges",
                    target_paths=(),
                    derivation=None,
                    dependency_paths=(),
                    dependency_bindings=(),
                    char_start=start,
                    char_end=end,
                    source_text=raw[start:end],
                    evidence_origin="host_verified_agent_proposal",
                    render_policy="natural_text",
                    rationale=(
                        "Host owned this selected place/payment token from the exact "
                        "ORIGIN/DESTINATION PORT CHARGE line grammar."
                    ),
                )
            )
    retained = tuple(row for row in drafts if row.draft_id not in discarded_ids)
    return merge_drafts(retained, additions)


def normalize_repeated_forwarding_references(
    *, raw: str, drafts: Sequence[SpanDraft], source_target: Mapping[str, Any]
) -> tuple[SpanDraft, ...]:
    """Append exact high-entropy repeats of one structured forwarding/export reference."""

    owners_by_path: dict[str, list[SpanDraft]] = defaultdict(list)
    for row in drafts:
        if row.render_mode != "target_binding" or len(row.target_paths) != 1:
            continue
        path = row.target_paths[0]
        if _FORWARDING_REFERENCE_PATH.fullmatch(path) is not None:
            owners_by_path[path].append(row)
    additions: list[SpanDraft] = []
    for path, owners in owners_by_path.items():
        if len({row.logical_key for row in owners}) != 1:
            continue
        target = _resolve_target_path(source_target, path)
        if not isinstance(target, str):
            continue
        normalized_target = _normalized_surface(target)
        if (
            len(normalized_target) < 8
            or not any(character.isalpha() for character in normalized_target)
            or not any(character.isdigit() for character in normalized_target)
        ):
            continue
        source_surfaces = {
            row.source_text
            for row in owners
            if _normalized_surface(row.source_text) == normalized_target
        }
        if len(source_surfaces) != 1:
            continue
        source_surface = next(iter(source_surfaces))
        owner = owners[0]
        for start in _exact_offsets(raw, source_surface):
            end = start + len(source_surface)
            if not is_token_bounded_surface_span(raw, start, end) or any(
                start < row.char_end and row.char_start < end for row in drafts
            ):
                continue
            additions.append(
                replace(
                    owner,
                    draft_id=(
                        "host_forwarding_reference_repeat_"
                        + sha256_bytes(f"{path}\0{start}\0{end}".encode())[:16]
                    ),
                    char_start=start,
                    char_end=end,
                    source_text=raw[start:end],
                    evidence_origin="host_verified_agent_proposal",
                    rationale=(
                        owner.rationale
                        + " Host appended this exact high-entropy forwarding/export reference "
                        "repeat."
                    ),
                )
            )
    return merge_drafts(drafts, additions)


def normalize_labeled_identifiers_and_operational_facts(
    *, raw: str, drafts: Sequence[SpanDraft], source_target: Mapping[str, Any]
) -> tuple[SpanDraft, ...]:
    """Own exact repeated document IDs and narrowly selected operational source facts."""

    lines = line_spans(raw)
    line_by_number = {line.number: line for line in lines}
    additions: list[SpanDraft] = []
    replaced_ids: set[str] = set()

    bol_owner = _unique_target_owner(drafts=drafts, target_path=_BILL_OF_LADING_PATH)
    try:
        bol_target = _scalar_surface(_resolve_target_path(source_target, _BILL_OF_LADING_PATH))
    except ValueError:
        bol_target = None
    if bol_owner is not None and bol_target:
        normalized_bol = _normalized_surface(bol_target)
        for caption in lines:
            normalized_caption = re.sub(r"^[0-9]+", "", _normalized_surface(caption.text))
            if not any(
                normalized_caption.startswith(prefix)
                for prefix in (
                    "billofladingnumber",
                    "blnumber",
                    "seawaybillnumber",
                    "waybillnumber",
                )
            ):
                continue
            value_line = line_by_number.get(caption.number + 1)
            span = _trimmed_line_span(value_line) if value_line is not None else None
            if (
                span is None
                or _normalized_surface(raw[span[0] : span[1]]) != normalized_bol
                or any(span[0] < row.char_end and row.char_start < span[1] for row in drafts)
            ):
                continue
            additions.append(
                replace(
                    bol_owner,
                    draft_id=(
                        "host_next_line_bol_repeat_"
                        + sha256_bytes(f"{span[0]}\0{span[1]}".encode())[:16]
                    ),
                    char_start=span[0],
                    char_end=span[1],
                    source_text=raw[span[0] : span[1]],
                    evidence_origin="host_verified_agent_proposal",
                    rationale=(
                        bol_owner.rationale
                        + " Host appended the exact next-line value under an explicit bill or "
                        "waybill number caption."
                    ),
                )
            )
        for line in lines:
            match = _BILL_OF_LADING_NUMBER_LINE.fullmatch(line.text)
            if match is None or _normalized_surface(match.group("value")) != normalized_bol:
                continue
            start = line.char_start + match.start("value")
            end = line.char_start + match.end("value")
            if any(start < row.char_end and row.char_start < end for row in drafts):
                continue
            additions.append(
                replace(
                    bol_owner,
                    draft_id=(
                        "host_labeled_bol_repeat_" + sha256_bytes(f"{start}\0{end}".encode())[:16]
                    ),
                    char_start=start,
                    char_end=end,
                    source_text=raw[start:end],
                    evidence_origin="host_verified_agent_proposal",
                    rationale=(
                        bol_owner.rationale
                        + " Host appended the exact value under the explicit B/L NO. label."
                    ),
                )
            )

        for caption in lines:
            if _COMBINED_BOOKING_CAPTION.fullmatch(caption.text) is None:
                continue
            value_line = line_by_number.get(caption.number + 1)
            if value_line is None:
                continue
            tokens = _surface_token_spans(value_line.text)
            bol_indexes = tuple(
                index
                for index, (token, _start, _end) in enumerate(tokens)
                if token == normalized_bol
            )
            if len(bol_indexes) != 1 or bol_indexes[0] == 0:
                continue
            booking_candidates = tuple(
                (token, start, end)
                for token, start, end in tokens[: bol_indexes[0]]
                if len(token) >= 6 and any(character.isdigit() for character in token)
            )
            if len(booking_candidates) != 1:
                continue
            _token, relative_start, relative_end = booking_candidates[0]
            start = value_line.char_start + relative_start
            end = value_line.char_start + relative_end
            if any(start < row.char_end and row.char_start < end for row in drafts):
                continue
            additions.append(
                SpanDraft(
                    draft_id=(
                        "host_combined_caption_booking_"
                        + sha256_bytes(f"{start}\0{end}".encode())[:16]
                    ),
                    logical_key="agent:commercial:booking_number",
                    render_mode="deterministic_auxiliary",
                    value_kind="identifier",
                    group_kind="commercial",
                    group_key="commercial:booking",
                    target_paths=(),
                    derivation=None,
                    dependency_paths=(),
                    dependency_bindings=(),
                    char_start=start,
                    char_end=end,
                    source_text=raw[start:end],
                    evidence_origin="host_verified_agent_proposal",
                    render_policy="opaque_identifier",
                    rationale=(
                        "Host owned the unique booking identifier printed before the structured "
                        "waybill value under the combined BOOKING NO./SEA WAYBILL NO. caption."
                    ),
                )
            )

    acid_surfaces: dict[str, str] = {}
    for line in lines:
        match = _ACID_VALUE_LINE.fullmatch(line.text)
        if match is None:
            continue
        surface = match.group("value")
        normalized = _normalized_surface(surface)
        prior = acid_surfaces.setdefault(normalized, surface)
        if prior != surface:
            acid_surfaces.pop(normalized, None)
    multiple_acid_values = len(acid_surfaces) > 1
    for normalized, surface in sorted(acid_surfaces.items()):
        spans = tuple(span for line in lines for span in _line_token_sequence_spans(line, surface))
        if len(spans) < 2:
            continue
        existing_by_span: dict[tuple[int, int], SpanDraft | None] = {}
        compatible = True
        for start, end in spans:
            overlaps = tuple(row for row in drafts if start < row.char_end and row.char_start < end)
            if not overlaps:
                existing_by_span[(start, end)] = None
                continue
            if not (
                len(overlaps) == 1
                and overlaps[0].char_start == start
                and overlaps[0].char_end == end
                and overlaps[0].render_mode == "deterministic_auxiliary"
                and overlaps[0].value_kind == "identifier"
                and overlaps[0].derivation is None
                and not overlaps[0].target_paths
                and not overlaps[0].dependency_paths
                and not overlaps[0].dependency_bindings
            ):
                compatible = False
                break
            existing_by_span[(start, end)] = overlaps[0]
        if not compatible:
            continue
        suffix = ":" + sha256_bytes(normalized.encode())[:12] if multiple_acid_values else ""
        logical_key = "agent:customs:acid_reference" + suffix
        group_key = "customs:acid" + suffix
        for start, end in spans:
            existing = existing_by_span[(start, end)]
            if existing is not None:
                replaced_ids.add(existing.draft_id)
                canonical = (
                    existing.logical_key == logical_key
                    and existing.group_kind == "customs"
                    and existing.group_key == group_key
                    and existing.render_policy == "opaque_identifier"
                )
                additions.append(
                    existing
                    if canonical
                    else replace(
                        existing,
                        logical_key=logical_key,
                        group_kind="customs",
                        group_key=group_key,
                        render_policy="opaque_identifier",
                        rationale=(
                            existing.rationale
                            + " Host joined this exact long identifier to its explicit ACID "
                            "occurrences under one customs-reference binding."
                        ),
                    )
                )
                continue
            additions.append(
                SpanDraft(
                    draft_id=(
                        "host_acid_reference_"
                        + sha256_bytes(f"{logical_key}\0{start}\0{end}".encode())[:16]
                    ),
                    logical_key=logical_key,
                    render_mode="deterministic_auxiliary",
                    value_kind="identifier",
                    group_kind="customs",
                    group_key=group_key,
                    target_paths=(),
                    derivation=None,
                    dependency_paths=(),
                    dependency_bindings=(),
                    char_start=start,
                    char_end=end,
                    source_text=raw[start:end],
                    evidence_origin="host_verified_agent_proposal",
                    render_policy="opaque_identifier",
                    rationale=(
                        "Host owned this exact long identifier as a repeated ACID customs "
                        "reference after an explicit labeled occurrence proved its type."
                    ),
                )
            )

    slac_spans: list[tuple[int, int]] = []
    for line in lines:
        tokens = _surface_token_spans(line.text)
        token_values = {token for token, _start, _end in tokens}
        if "container" not in token_values:
            continue
        slac_spans.extend(
            (line.char_start + start, line.char_start + end)
            for token, start, end in tokens
            if token == "slac"
            and not any(
                line.char_start + start < row.char_end and row.char_start < line.char_start + end
                for row in drafts
            )
        )
    if len(slac_spans) >= 2:
        additions.extend(
            SpanDraft(
                draft_id=("host_slac_operational_" + sha256_bytes(f"{start}\0{end}".encode())[:16]),
                logical_key="agent:cargo:slac_qualifier",
                render_mode="deterministic_auxiliary",
                value_kind="operational_text",
                group_kind="cargo",
                group_key="cargo:operational_qualifiers",
                target_paths=(),
                derivation=None,
                dependency_paths=(),
                dependency_bindings=(),
                char_start=start,
                char_end=end,
                source_text=raw[start:end],
                evidence_origin="host_verified_agent_proposal",
                render_policy="natural_text",
                rationale=(
                    "Host owned this repeated SLAC operational qualifier in an explicit "
                    "container-description row."
                ),
            )
            for start, end in slac_spans
        )

    country_surfaces = {
        _normalized_surface(surface)
        for row in drafts
        for path in row.target_paths
        if path.endswith(".country")
        and (surface := _scalar_surface(_resolve_target_path(source_target, path))) is not None
    }
    for line in lines:
        match = _COUNTRY_CLAUSE_HEADING.fullmatch(line.text)
        if match is None or _normalized_surface(match.group("country")) not in country_surfaces:
            continue
        span = _trimmed_line_span(line)
        if span is None or any(
            span[0] < row.char_end and row.char_start < span[1] for row in drafts
        ):
            continue
        additions.append(
            SpanDraft(
                draft_id=(
                    "host_country_clause_heading_"
                    + sha256_bytes(f"{span[0]}\0{span[1]}".encode())[:16]
                ),
                logical_key="agent:legal:country_clause_heading",
                render_mode="deterministic_auxiliary",
                value_kind="legal_text",
                group_kind="legal",
                group_key="legal:country_clause",
                target_paths=(),
                derivation=None,
                dependency_paths=(),
                dependency_bindings=(),
                char_start=span[0],
                char_end=span[1],
                source_text=raw[span[0] : span[1]],
                evidence_origin="host_verified_agent_proposal",
                render_policy="natural_text",
                rationale=(
                    "Host owned this selected country-clause heading after matching its country "
                    "to an exact structured party-country value."
                ),
            )
        )
    retained = tuple(row for row in drafts if row.draft_id not in replaced_ids)
    return merge_drafts(retained, additions)


def normalize_dangerous_goods_class_locality(
    *, raw: str, drafts: Sequence[SpanDraft], source_target: Mapping[str, Any]
) -> tuple[SpanDraft, ...]:
    """Relocate a hazard-category owner to its unique explicit ``UN ... CL`` value.

    Accepted character-level evidence can select the first digit of a decimal class and then
    repeat that digit into unrelated numbered text.  The correction is safe only when the indexed
    dangerous-goods record has a unique UN number in the structured target and every matching OCR
    line prints one equivalent class value.
    """

    patch = source_target.get("documentPatch")
    cargo_groups = patch.get("cargoGroups") if isinstance(patch, Mapping) else None
    if not isinstance(cargo_groups, list):
        return tuple(drafts)
    records: list[tuple[str, str]] = []
    for cargo_index, cargo_group in enumerate(cargo_groups):
        dangerous_goods = (
            cargo_group.get("dangerousGoods") if isinstance(cargo_group, Mapping) else None
        )
        if not isinstance(dangerous_goods, list):
            continue
        for dangerous_index, dangerous_good in enumerate(dangerous_goods):
            if not isinstance(dangerous_good, Mapping) or not {
                "hazardCategory",
                "unNumber",
            } <= set(dangerous_good):
                continue
            un_surface = _scalar_surface(dangerous_good["unNumber"])
            if un_surface is None:
                continue
            records.append(
                (
                    f"documentPatch.cargoGroups[{cargo_index}].dangerousGoods"
                    f"[{dangerous_index}].hazardCategory",
                    un_surface,
                )
            )
    un_counts = Counter(_normalized_surface(un_surface) for _path, un_surface in records)
    lines = line_spans(raw)
    replaced_ids: set[str] = set()
    additions: list[SpanDraft] = []
    for hazard_path, un_surface in records:
        if un_counts[_normalized_surface(un_surface)] != 1:
            continue
        owners = tuple(row for row in drafts if hazard_path in row.target_paths)
        if (
            not owners
            or len({row.logical_key for row in owners}) != 1
            or any(row.target_paths != (hazard_path,) for row in owners)
        ):
            continue
        pattern = re.compile(
            r"(?i)(?<![A-Z0-9])UN(?:\s*NO\.?)?[\s:#-]*(?:UN[\s:#-]*)?"
            + re.escape(un_surface)
            + r"(?![A-Z0-9]).*?\bCL(?:ASS)?[\s:#-]*(?P<value>[0-9](?:\.[0-9]+)?)"
        )
        candidates = list(
            (
                line.char_start + match.start("value"),
                line.char_start + match.end("value"),
            )
            for line in lines
            for match in pattern.finditer(line.text)
        )
        explicit_surfaces = {_normalized_surface(raw[start:end]) for start, end in candidates}
        if len(records) == 1 and len(explicit_surfaces) == 1:
            standalone_pattern = re.compile(
                r"(?i)^\s*CL(?:ASS)?\s*[:#-]?\s*(?P<value>[0-9](?:\.[0-9]+)?)\s*$"
            )
            for line in lines:
                match = standalone_pattern.fullmatch(line.text)
                if (
                    match is not None
                    and _normalized_surface(match.group("value")) in explicit_surfaces
                ):
                    candidates.append(
                        (
                            line.char_start + match.start("value"),
                            line.char_start + match.end("value"),
                        )
                    )
        candidates = list(dict.fromkeys(candidates))
        candidate_surfaces = {_normalized_surface(raw[start:end]) for start, end in candidates}
        if not candidates or len(candidate_surfaces) != 1:
            continue
        owner_ids = {row.draft_id for row in owners}
        if any(
            start < row.char_end and row.char_start < end and row.draft_id not in owner_ids
            for start, end in candidates
            for row in drafts
        ):
            continue
        first = owners[0]
        group_kind, group_key = _canonical_target_group((hazard_path,))
        replaced_ids.update(owner_ids)
        for start, end in candidates:
            existing = next(
                (row for row in owners if row.char_start == start and row.char_end == end),
                None,
            )
            canonical = (
                existing is not None
                and existing.render_mode == "agent_residual"
                and existing.value_kind == "dangerous_goods"
                and existing.group_kind == group_kind
                and existing.group_key == group_key
                and existing.target_paths == (hazard_path,)
                and existing.derivation is None
                and not existing.dependency_paths
                and not existing.dependency_bindings
                and existing.render_policy == "natural_text"
            )
            if canonical and existing is not None:
                additions.append(existing)
                continue
            additions.append(
                replace(
                    first,
                    draft_id=(
                        "host_dangerous_goods_class_"
                        + sha256_bytes(f"{hazard_path}\0{start}\0{end}".encode())[:16]
                    ),
                    render_mode="agent_residual",
                    value_kind="dangerous_goods",
                    group_kind=group_kind,
                    group_key=group_key,
                    target_paths=(hazard_path,),
                    derivation=None,
                    dependency_paths=(),
                    dependency_bindings=(),
                    char_start=start,
                    char_end=end,
                    source_text=raw[start:end],
                    evidence_origin="host_verified_agent_proposal",
                    render_policy="natural_text",
                    rationale=(
                        first.rationale
                        + " Host relocated the category owner to the complete class value on the "
                        f"unique explicit UN {un_surface} / CL row; category-to-class conversion "
                        "remains agent-bounded."
                    ),
                )
            )
    retained = tuple(row for row in drafts if row.draft_id not in replaced_ids)
    return merge_drafts(retained, additions)


def normalize_labeled_dangerous_goods_descriptions(
    *, raw: str, drafts: Sequence[SpanDraft], source_target: Mapping[str, Any]
) -> tuple[SpanDraft, ...]:
    """Own a unique free-text DG description following an indexed ``UN <number>`` value.

    The rule does not infer a description from arbitrary cargo prose. It activates only for a
    uniquely indexed structured UN number that already has target ownership, on a line beginning
    with that explicit UN label, and only when the complete remaining text is currently unowned.
    """

    patch = source_target.get("documentPatch")
    cargo_groups = patch.get("cargoGroups") if isinstance(patch, Mapping) else None
    if not isinstance(cargo_groups, list):
        return tuple(drafts)
    records: list[tuple[int, int, str, str]] = []
    for cargo_index, cargo_group in enumerate(cargo_groups):
        dangerous_goods = (
            cargo_group.get("dangerousGoods") if isinstance(cargo_group, Mapping) else None
        )
        if not isinstance(dangerous_goods, list):
            continue
        for dangerous_index, dangerous_good in enumerate(dangerous_goods):
            if not isinstance(dangerous_good, Mapping) or "unNumber" not in dangerous_good:
                continue
            un_surface = _scalar_surface(dangerous_good["unNumber"])
            if un_surface is None:
                continue
            records.append(
                (
                    cargo_index,
                    dangerous_index,
                    f"documentPatch.cargoGroups[{cargo_index}].dangerousGoods"
                    f"[{dangerous_index}].unNumber",
                    un_surface,
                )
            )
    un_counts = Counter(_normalized_surface(row[3]) for row in records)
    lines = line_spans(raw)
    additions: list[SpanDraft] = []
    for cargo_index, dangerous_index, un_path, un_surface in records:
        if un_counts[_normalized_surface(un_surface)] != 1:
            continue
        owners = tuple(
            row
            for row in drafts
            if un_path in row.target_paths
            and row.render_mode in {"target_binding", "agent_residual"}
        )
        if not owners:
            continue
        pattern = re.compile(
            r"(?i)^\s*UN[\s:#-]*"
            + re.escape(un_surface)
            + r"(?![A-Z0-9])\s+(?P<description>\S(?:.*\S)?)\s*$"
        )
        candidates: list[tuple[int, int]] = []
        for line in lines:
            if not any(line.char_start <= owner.char_start < line.char_end for owner in owners):
                continue
            match = pattern.fullmatch(line.text)
            if match is None:
                continue
            description = match.group("description")
            words = re.findall(r"[A-Za-z]+", description)
            if len(words) < 3 or words[0].casefold() in {
                "cl",
                "class",
                "pg",
                "packing",
                "flash",
            }:
                continue
            start = line.char_start + match.start("description")
            end = line.char_start + match.end("description")
            overlaps = tuple(row for row in drafts if start < row.char_end and row.char_start < end)
            if overlaps:
                if all(
                    row.char_start == start
                    and row.char_end == end
                    and row.render_mode == "deterministic_auxiliary"
                    and row.value_kind == "dangerous_goods"
                    for row in overlaps
                ):
                    continue
                candidates = []
                break
            candidates.append((start, end))
        if not candidates:
            continue
        surfaces = {_normalized_surface(raw[start:end]) for start, end in candidates}
        if len(surfaces) != 1:
            continue
        for start, end in candidates:
            additions.append(
                SpanDraft(
                    draft_id=(
                        "host_dangerous_goods_description_"
                        + sha256_bytes(f"{un_path}\0{start}\0{end}".encode())[:16]
                    ),
                    logical_key=(
                        f"agent:dangerous_goods:{cargo_index}:{dangerous_index}:description"
                    ),
                    render_mode="deterministic_auxiliary",
                    value_kind="dangerous_goods",
                    group_kind="dangerous_goods",
                    group_key=f"dangerous_goods:{cargo_index}:{dangerous_index}",
                    target_paths=(),
                    derivation=None,
                    dependency_paths=(),
                    dependency_bindings=(),
                    char_start=start,
                    char_end=end,
                    source_text=raw[start:end],
                    evidence_origin="host_verified_agent_proposal",
                    render_policy="natural_text",
                    rationale=(
                        "Host owned this free-text dangerous-goods description because it is the "
                        f"complete unowned suffix of the uniquely indexed UN {un_surface} line."
                    ),
                )
            )
    return merge_drafts(drafts, additions)


def normalize_bare_equipment_multipliers(
    *, raw: str, drafts: Sequence[SpanDraft], source_target: Mapping[str, Any]
) -> tuple[SpanDraft, ...]:
    """Complete a row-local ``N X <type>`` equipment receipt deterministically.

    A type-only receipt proposal and a separately owned bare multiplier describe one contiguous
    render surface.  The host joins them only when every occurrence of the logical type/receipt
    binding has one immediate same-line multiplier and all intervening ownership is a compatible
    multiplier for the same container.  Partial or ambiguous groups remain unchanged for review.
    """

    patch = source_target.get("documentPatch")
    containers = patch.get("containers") if isinstance(patch, Mapping) else None
    if not isinstance(containers, list):
        return tuple(drafts)

    def container_index(row: SpanDraft) -> int | None:
        direct = _container_index_for_path(row.target_paths, _CONTAINER_TYPE_PATH)
        if row.render_mode == "target_binding" and direct is not None:
            return direct
        if not (
            row.render_mode == "deterministic_derived" and row.derivation == "equipment_receipt"
        ):
            return None
        object_matches = tuple(
            match
            for path in row.target_paths
            if (match := _CONTAINER_OBJECT_PATH.fullmatch(path)) is not None
        )
        if len(object_matches) != 1:
            return None
        index = int(object_matches[0].group(1))
        container_path = f"documentPatch.containers[{index}]"
        type_path = f"documentPatch.containers[{index}].typeDescription"
        if any(path not in {container_path, type_path} for path in row.target_paths):
            return None
        return index if type_path in {*row.target_paths, *row.dependency_paths} else None

    lines = line_spans(raw)
    grouped: dict[str, list[SpanDraft]] = defaultdict(list)
    for row in drafts:
        grouped[row.logical_key].append(row)
    eligible_groups: dict[int, list[tuple[list[SpanDraft], list[int], set[str]]]] = defaultdict(
        list
    )
    for rows in grouped.values():
        indexes = {container_index(row) for row in rows}
        if len(indexes) != 1 or None in indexes:
            continue
        index = next(iter(indexes))
        assert index is not None
        if index >= len(containers) or not isinstance(containers[index], Mapping):
            continue
        type_path = f"documentPatch.containers[{index}].typeDescription"
        if "typeDescription" not in containers[index]:
            continue
        starts: list[int] = []
        compatible_prefix_ids: set[str] = set()
        valid = True
        for row in rows:
            line_number = line_number_for_char(lines, row.char_start)
            line = lines[line_number - 1]
            prefix = raw[line.char_start : row.char_start]
            match = re.search(r"(?i)(?P<value>[0-9]+\s*[x\u00d7])\s*$", prefix)
            if match is None:
                valid = False
                break
            start = line.char_start + match.start("value")
            full_surface = raw[start : row.char_end]
            if _EQUIPMENT_RECEIPT_PREFIX.match(full_surface) is None:
                valid = False
                break
            overlaps = tuple(
                other
                for other in drafts
                if other.draft_id != row.draft_id
                and start < other.char_end
                and other.char_start < row.char_start
            )
            compatible = tuple(
                other
                for other in overlaps
                if other.char_start == start
                and other.char_end <= row.char_start
                and other.render_mode == "deterministic_derived"
                and other.derivation == "equipment_receipt"
                and other.target_paths == (f"documentPatch.containers[{index}]",)
                and _BARE_EQUIPMENT_MULTIPLIER.fullmatch(other.source_text) is not None
            )
            if len(compatible) != len(overlaps):
                valid = False
                break
            starts.append(start)
            compatible_prefix_ids.update(other.draft_id for other in compatible)
        if valid:
            eligible_groups[index].append((rows, starts, compatible_prefix_ids))

    additions: list[SpanDraft] = []
    replaced_ids: set[str] = set()
    for index, candidates in eligible_groups.items():
        if len(candidates) != 1:
            continue
        rows, starts, prefix_ids = candidates[0]
        container_path = f"documentPatch.containers[{index}]"
        type_path = container_path + ".typeDescription"
        logical_key = f"agent:equipment:container_receipt:{index}"
        replaced_ids.update(row.draft_id for row in rows)
        replaced_ids.update(prefix_ids)
        for row, start in zip(rows, starts, strict=True):
            additions.append(
                replace(
                    row,
                    logical_key=logical_key,
                    render_mode="deterministic_derived",
                    value_kind="equipment",
                    group_kind="equipment",
                    group_key=f"container:{index}",
                    target_paths=(container_path,),
                    derivation="equipment_receipt",
                    dependency_paths=(type_path,),
                    dependency_bindings=(),
                    char_start=start,
                    source_text=raw[start : row.char_end],
                    evidence_origin="host_verified_agent_proposal",
                    render_policy="derived_surface",
                    rationale=(
                        row.rationale
                        + " Host completed the immediate row-local multiplier and equipment type "
                        "as one deterministic equipment-receipt surface."
                    ),
                )
            )
    retained = tuple(row for row in drafts if row.draft_id not in replaced_ids)
    return merge_drafts(retained, additions)


def normalize_complete_equipment_receipt_surfaces(
    *, raw: str, drafts: Sequence[SpanDraft], source_target: Mapping[str, Any]
) -> tuple[SpanDraft, ...]:
    """Expand a derived equipment receipt through its complete exact type surface.

    A model may stop at ``1X40HIGH`` even when the linked structured type and the same source line
    continue with ``CUBE``.  The extension is deterministic only when stripping the printed
    multiplier from exactly one token-bounded prefix yields the complete normalized target type,
    and no other binding owns the added bytes.
    """

    lines = line_spans(raw)
    grouped: dict[str, list[SpanDraft]] = defaultdict(list)
    for draft in drafts:
        grouped[draft.logical_key].append(draft)
    replacement_ends: dict[str, int] = {}
    for _logical_key, rows in grouped.items():
        if not all(
            row.render_mode == "deterministic_derived" and row.derivation == "equipment_receipt"
            for row in rows
        ):
            continue
        planned: dict[str, int] = {}
        valid = True
        for row in rows:
            type_paths = tuple(
                path
                for path in (*row.target_paths, *row.dependency_paths)
                if _CONTAINER_TYPE_PATH.fullmatch(path) is not None
            )
            if len(set(type_paths)) != 1:
                valid = False
                break
            type_surface = _scalar_surface(_resolve_target_path(source_target, type_paths[0]))
            if type_surface is None:
                valid = False
                break
            line_number = line_number_for_char(lines, row.char_start)
            line = lines[line_number - 1]
            if line_number != line_number_for_char(lines, row.char_end - 1):
                valid = False
                break
            candidate_ends: list[int] = []
            for _token, _start, relative_end in _surface_token_spans(
                raw[row.char_start : line.char_end]
            ):
                end = row.char_start + relative_end
                surface = raw[row.char_start : end]
                multiplier = re.match(r"(?i)^\s*[0-9]+\s*[x\u00d7]\s*", surface)
                if multiplier is None:
                    continue
                if _normalized_surface(surface[multiplier.end() :]) == _normalized_surface(
                    type_surface
                ):
                    candidate_ends.append(end)
            if len(candidate_ends) != 1:
                valid = False
                break
            end = candidate_ends[0]
            if end > row.char_end and any(
                row.char_end < other.char_end
                and other.char_start < end
                and other.draft_id != row.draft_id
                for other in drafts
            ):
                valid = False
                break
            planned[row.draft_id] = end
        if valid and any(planned[row.draft_id] != row.char_end for row in rows):
            replacement_ends.update(planned)

    return merge_drafts(
        replace(
            row,
            char_end=replacement_ends[row.draft_id],
            source_text=raw[row.char_start : replacement_ends[row.draft_id]],
            evidence_origin="derived_operational_fact",
            rationale=(
                row.rationale
                + " Host expanded the row-local equipment receipt through the complete exact "
                "structured type surface."
            ),
        )
        if row.draft_id in replacement_ends and replacement_ends[row.draft_id] != row.char_end
        else row
        for row in drafts
    )


def normalize_abbreviated_container_counts(
    *, raw: str, drafts: Sequence[SpanDraft], source_target: Mapping[str, Any]
) -> tuple[SpanDraft, ...]:
    """Own ``N CONT.`` immediately preceding a structured container type.

    The abbreviation is accepted only when its printed number equals the pinned structured
    container-list length and the same line continues with a direct type owner.  This distinguishes
    a complete container-count noun surface from a bare equipment multiplier and from arbitrary
    prose abbreviations.
    """

    patch = source_target.get("documentPatch")
    containers = patch.get("containers") if isinstance(patch, Mapping) else None
    if not isinstance(containers, list) or not containers:
        return tuple(drafts)
    expected_count = len(containers)
    lines = line_spans(raw)
    candidates: list[tuple[int, int]] = []
    for row in drafts:
        index = _container_index_for_path(row.target_paths, _CONTAINER_TYPE_PATH)
        if row.render_mode != "target_binding" or index is None:
            continue
        line_number = line_number_for_char(lines, row.char_start)
        line = lines[line_number - 1]
        prefix = raw[line.char_start : row.char_start]
        match = re.search(r"(?i)(?P<surface>(?P<count>[0-9][0-9,]*)\s+CONT\.)\s*$", prefix)
        if match is None or int(match.group("count").replace(",", "")) != expected_count:
            continue
        start = line.char_start + match.start("surface")
        end = line.char_start + match.end("surface")
        candidates.append((start, end))
    if not candidates:
        return tuple(drafts)

    logical_key = "agent:container_count:documentPatch.containers"
    additions: list[SpanDraft] = []
    discarded_ids: set[str] = set()
    for start, end in sorted(set(candidates)):
        overlaps = tuple(row for row in drafts if start < row.char_end and row.char_start < end)
        if overlaps:
            already_owned = all(
                row.char_start == start
                and row.char_end == end
                and row.render_mode == "deterministic_derived"
                and row.derivation == "container_count"
                and "documentPatch.containers" in {*row.target_paths, *row.dependency_paths}
                for row in overlaps
            )
            if already_owned:
                continue
            literal_rows = tuple(
                row
                for row in overlaps
                if row.char_start == start
                and row.char_end == end
                and row.render_mode == "literal_static"
            )
            if len(literal_rows) != len(overlaps):
                continue
            discarded_ids.update(row.draft_id for row in literal_rows)
        additions.append(
            SpanDraft(
                draft_id=(
                    "host_abbreviated_container_count_"
                    + sha256_bytes(f"{start}\0{end}".encode())[:16]
                ),
                logical_key=logical_key,
                render_mode="deterministic_derived",
                value_kind="integer",
                group_kind="equipment",
                group_key="equipment:all",
                target_paths=("documentPatch.containers",),
                derivation="container_count",
                dependency_paths=("documentPatch.containers",),
                dependency_bindings=(),
                char_start=start,
                char_end=end,
                source_text=raw[start:end],
                evidence_origin="derived_operational_fact",
                render_policy="derived_surface",
                rationale=(
                    "Host derived this complete abbreviated container-count surface from the "
                    "pinned container collection and its immediate structured type row."
                ),
            )
        )
    retained = tuple(row for row in drafts if row.draft_id not in discarded_ids)
    return merge_drafts(retained, additions)


def normalize_source_only_package_nouns(
    *, raw: str, drafts: Sequence[SpanDraft]
) -> tuple[SpanDraft, ...]:
    """Own a package noun immediately following a source-only numeric value."""

    lines = line_spans(raw)
    candidates: dict[str, list[tuple[int, int]]] = defaultdict(list)
    for row in drafts:
        if not (
            row.render_mode in {"deterministic_auxiliary", "deterministic_derived"}
            and row.value_kind == "integer"
            and not row.target_paths
            and not row.dependency_paths
            and _decimal_candidates(row.source_text, integer=True)
        ):
            continue
        line_number = line_number_for_char(lines, row.char_end - 1)
        line = lines[line_number - 1]
        suffix = raw[row.char_end : line.char_end]
        match = re.match(r"\s+(?P<noun>PACKAGES?)\b", suffix, re.IGNORECASE)
        if match is None:
            continue
        start = row.char_end + match.start("noun")
        end = row.char_end + match.end("noun")
        candidates[_normalized_surface(raw[start:end])].append((start, end))
    if not candidates:
        return tuple(drafts)

    additions: list[SpanDraft] = []
    discarded_ids: set[str] = set()
    for normalized_noun, spans in sorted(candidates.items()):
        accepted: list[tuple[int, int]] = []
        for start, end in sorted(set(spans)):
            overlaps = tuple(row for row in drafts if start < row.char_end and row.char_start < end)
            if overlaps:
                already_owned = all(
                    row.char_start == start
                    and row.char_end == end
                    and row.render_mode == "deterministic_auxiliary"
                    and row.value_kind == "package"
                    and not row.target_paths
                    for row in overlaps
                )
                if already_owned:
                    continue
                literal_rows = tuple(
                    row
                    for row in overlaps
                    if row.char_start == start
                    and row.char_end == end
                    and row.render_mode == "literal_static"
                )
                if len(literal_rows) != len(overlaps):
                    continue
                discarded_ids.update(row.draft_id for row in literal_rows)
            accepted.append((start, end))
        logical_key = f"agent:package:source_only_kind:{normalized_noun}"
        for start, end in accepted:
            additions.append(
                SpanDraft(
                    draft_id=(
                        "host_source_only_package_noun_"
                        + sha256_bytes(f"{logical_key}\0{start}\0{end}".encode())[:16]
                    ),
                    logical_key=logical_key,
                    render_mode="deterministic_auxiliary",
                    value_kind="package",
                    group_kind="package",
                    group_key="package:source_only_kind",
                    target_paths=(),
                    derivation=None,
                    dependency_paths=(),
                    dependency_bindings=(),
                    char_start=start,
                    char_end=end,
                    source_text=raw[start:end],
                    evidence_origin="host_verified_agent_proposal",
                    render_policy="natural_text",
                    rationale=(
                        "Host owned this package-kind noun because it immediately completes an "
                        "independently owned source-only numeric package surface."
                    ),
                )
            )
    retained = tuple(row for row in drafts if row.draft_id not in discarded_ids)
    return merge_drafts(retained, additions)


def normalize_foreign_exporter_registration_types(
    *, raw: str, drafts: Sequence[SpanDraft]
) -> tuple[SpanDraft, ...]:
    """Own the selected value in ``FOREIGN EXPORTER REGISTRATION`` / ``TYPE: ...`` rows."""

    lines = line_spans(raw)
    previous_nonempty: LineSpan | None = None
    candidates: dict[str, list[tuple[int, int]]] = defaultdict(list)
    for line in lines:
        match = re.fullmatch(
            r"\s*TYPE\s*:\s*(?P<value>[A-Z][A-Z0-9]*(?:[ \t]+[A-Z0-9][A-Z0-9./_-]*)*)\s*",
            line.text,
            re.IGNORECASE,
        )
        if (
            match is not None
            and previous_nonempty is not None
            and _normalized_surface(previous_nonempty.text) == "foreignexporterregistration"
        ):
            start = line.char_start + match.start("value")
            end = line.char_start + match.end("value")
            candidates[_normalized_surface(raw[start:end])].append((start, end))
        if line.text.strip():
            previous_nonempty = line
    if not candidates:
        return tuple(drafts)

    additions: list[SpanDraft] = []
    discarded_ids: set[str] = set()
    for normalized_value, spans in sorted(candidates.items()):
        logical_key = "agent:customs:foreign_exporter_registration_type"
        if len(candidates) > 1:
            logical_key += ":" + sha256_bytes(normalized_value.encode())[:12]
        for start, end in sorted(set(spans)):
            overlaps = tuple(row for row in drafts if start < row.char_end and row.char_start < end)
            if overlaps:
                already_owned = all(
                    row.char_start == start
                    and row.char_end == end
                    and row.logical_key == logical_key
                    and row.render_mode == "deterministic_auxiliary"
                    and not row.target_paths
                    for row in overlaps
                )
                if already_owned:
                    continue
                literal_rows = tuple(
                    row
                    for row in overlaps
                    if row.char_start == start
                    and row.char_end == end
                    and row.render_mode == "literal_static"
                )
                if len(literal_rows) != len(overlaps):
                    continue
                discarded_ids.update(row.draft_id for row in literal_rows)
            additions.append(
                SpanDraft(
                    draft_id=(
                        "host_foreign_exporter_registration_type_"
                        + sha256_bytes(f"{logical_key}\0{start}\0{end}".encode())[:16]
                    ),
                    logical_key=logical_key,
                    render_mode="deterministic_auxiliary",
                    value_kind="identifier",
                    group_kind="customs",
                    group_key="customs:foreign_exporter",
                    target_paths=(),
                    derivation=None,
                    dependency_paths=(),
                    dependency_bindings=(),
                    char_start=start,
                    char_end=end,
                    source_text=raw[start:end],
                    evidence_origin="host_verified_agent_proposal",
                    render_policy="opaque_identifier",
                    rationale=(
                        "Host owned this selected foreign-exporter registration-type value under "
                        "its exact registration and TYPE captions."
                    ),
                )
            )
    retained = tuple(row for row in drafts if row.draft_id not in discarded_ids)
    return merge_drafts(retained, additions)


def normalize_labeled_registration_type_values(
    *, raw: str, drafts: Sequence[SpanDraft]
) -> tuple[SpanDraft, ...]:
    """Own an alphabetic value printed by an explicit registration-type field.

    The lexical TYPE detector is deliberately high recall. This narrower normalization requires
    the same line to say REGISTRATION immediately before TYPE and accepts only an alphabetic
    category value, so identifiers appended after a category remain subject to semantic review.
    """

    additions: list[SpanDraft] = []
    for line in line_spans(raw):
        for match in _SELECTED_TYPE_VALUE.finditer(line.text):
            prefix = _normalized_surface(line.text[: match.start()])
            value = match.group("value")
            if not prefix.endswith("registration") or any(
                character.isdigit() for character in value
            ):
                continue
            start = line.char_start + match.start("value")
            end = line.char_start + match.end("value")
            if any(start < row.char_end and row.char_start < end for row in drafts):
                continue
            normalized_value = _normalized_surface(value)
            additions.append(
                SpanDraft(
                    draft_id=(
                        "host_registration_type_" + sha256_bytes(f"{start}\0{end}".encode())[:16]
                    ),
                    logical_key=(
                        "agent:customs:registration:type:"
                        + sha256_bytes(normalized_value.encode())[:12]
                    ),
                    render_mode="deterministic_auxiliary",
                    value_kind="identifier",
                    group_kind="customs",
                    group_key="customs:registration_type",
                    target_paths=(),
                    derivation=None,
                    dependency_paths=(),
                    dependency_bindings=(),
                    char_start=start,
                    char_end=end,
                    source_text=raw[start:end],
                    evidence_origin="host_verified_agent_proposal",
                    render_policy="categorical_surface",
                    rationale=(
                        "Host classified this alphabetic value as fixed registration-type "
                        "vocabulary because the same line has an explicit REGISTRATION TYPE "
                        "field and contains no appended identifier digits."
                    ),
                )
            )
    return merge_drafts(drafts, additions)


def _heading_like(line: LineSpan) -> bool:
    stripped = line.text.strip()
    letters = tuple(character for character in stripped if character.isalpha())
    return bool(
        stripped
        and len(stripped) <= 100
        and (
            re.match(r"^\([0-9]+\)\s*", stripped) is not None
            or (letters and sum(character.isupper() for character in letters) / len(letters) >= 0.8)
        )
    )


def normalize_form_title_repeats(*, raw: str, drafts: Sequence[SpanDraft]) -> tuple[SpanDraft, ...]:
    """Append exact form-title repeats only from heading-like source lines."""

    grouped: dict[str, list[SpanDraft]] = defaultdict(list)
    for row in drafts:
        grouped[row.logical_key].append(row)
    additions: list[SpanDraft] = []
    for logical_key, rows in grouped.items():
        first = rows[0]
        key_tokens = frozenset(re.findall(r"[a-z0-9]+", logical_key.casefold()))
        if not (
            first.render_mode == "literal_static"
            and not first.target_paths
            and not first.dependency_paths
            and not first.dependency_bindings
            and {"form", "title"} <= key_tokens
            and all(row.source_text == first.source_text for row in rows)
        ):
            continue
        existing = {(row.char_start, row.char_end) for row in rows}
        for line in line_spans(raw):
            if not _heading_like(line):
                continue
            for relative_start in _exact_offsets(line.text, first.source_text):
                start = line.char_start + relative_start
                end = start + len(first.source_text)
                if (start, end) in existing or not is_token_bounded_surface_span(raw, start, end):
                    continue
                if any(start < row.char_end and row.char_start < end for row in drafts):
                    continue
                additions.append(
                    replace(
                        first,
                        draft_id=(
                            "host_form_title_repeat_"
                            + sha256_bytes(f"{logical_key}\0{start}\0{end}".encode())[:16]
                        ),
                        char_start=start,
                        char_end=end,
                        source_text=raw[start:end],
                        evidence_origin="host_verified_agent_proposal",
                        rationale=(
                            first.rationale
                            + " Host appended this exact form-title repeat from a bounded "
                            "heading-like line."
                        ),
                    )
                )
    return merge_drafts(drafts, additions)


def normalize_agent_appointment_suffixes(
    *, raw: str, drafts: Sequence[SpanDraft], source_target: Mapping[str, Any]
) -> tuple[SpanDraft, ...]:
    """Move text after ``AS AGENT ONLY FOR <carrier>`` out of carrier identity."""

    try:
        carrier_value = _resolve_target_path(source_target, _CARRIER_NAME_PATH)
    except ValueError:
        return tuple(drafts)
    carrier_name = _scalar_surface(carrier_value)
    if carrier_name is None:
        return tuple(drafts)
    replacements: dict[str, SpanDraft] = {}
    for line in line_spans(raw):
        frame = re.search(r"(?i)\bAS\s+AGENT\s+ONLY\s+FOR\b", line.text)
        if frame is None:
            continue
        carrier_spans = tuple(
            span
            for span in _line_token_sequence_spans(line, carrier_name)
            if span[0] >= line.char_start + frame.end()
        )
        if len(carrier_spans) != 1:
            continue
        suffix_start = carrier_spans[0][1]
        while suffix_start < line.char_end and raw[suffix_start] in " \t,;:-":
            suffix_start += 1
        suffix_end = line.char_end
        while suffix_end > suffix_start and raw[suffix_end - 1].isspace():
            suffix_end -= 1
        suffix = raw[suffix_start:suffix_end]
        if not suffix or any(character.isdigit() for character in suffix):
            continue
        owners = tuple(
            row
            for row in drafts
            if row.char_start == suffix_start
            and row.char_end == suffix_end
            and row.render_mode == "carrier_static"
            and not row.target_paths
        )
        if len(owners) != 1:
            continue
        owner = owners[0]
        replacements[owner.draft_id] = replace(
            owner,
            logical_key="agent:issuingAgent:0:location",
            render_mode="deterministic_auxiliary",
            value_kind="location",
            group_kind="party",
            group_key="party:issuingAgent:0",
            render_policy="natural_text",
            rationale=(
                owner.rationale
                + " Host moved this suffix out of carrier identity because it follows the "
                "complete canonical carrier in an explicit agent-appointment frame."
            ),
        )
    return merge_drafts(replacements.get(row.draft_id, row) for row in drafts)


_MEASUREMENT_CAPTION_FIELDS: tuple[tuple[re.Pattern[str], str], ...] = (
    (re.compile(r"(?i)\bGROSS(?:\s+CARGO)?\s+WEIGHT\b"), "grossWeight"),
    (re.compile(r"(?i)\bNET(?:\s+CARGO)?\s+WEIGHT\b"), "netWeight"),
    (re.compile(r"(?i)\bTARE(?:\s+CARGO)?\s+WEIGHT\b"), "tareWeight"),
    (re.compile(r"(?i)\b(?:VOLUME|MEASUREMENT)\b"), "volume"),
)


def normalize_measurement_caption_units(
    *, raw: str, drafts: Sequence[SpanDraft], source_target: Mapping[str, Any]
) -> tuple[SpanDraft, ...]:
    """Join a nearby unit to a uniquely identified captioned cargo measurement.

    OCR forms commonly print either ``GROSS WEIGHT (KG)`` followed by the value or three
    populated rows ``GROSS WEIGHT`` / value / ``KG``.  Both layouts carry the same structural
    relationship; accepting only the first silently left the unit mutable in the second.
    """

    patch = source_target.get("documentPatch")
    cargo_groups = patch.get("cargoGroups") if isinstance(patch, Mapping) else None
    if not isinstance(cargo_groups, list):
        return tuple(drafts)
    lines = line_spans(raw)
    additions: list[SpanDraft] = []
    replacements: dict[str, SpanDraft] = {}
    discarded: set[str] = set()
    for line_index, line in enumerate(lines):
        fields = tuple(
            field for pattern, field in _MEASUREMENT_CAPTION_FIELDS if pattern.search(line.text)
        )
        if len(fields) != 1:
            continue
        caption_unit_tokens = tuple(
            (token, start, end)
            for token, start, end in _surface_token_spans(line.text)
            if any(token in values for values in _MEASUREMENT_UNIT_SURFACES.values())
        )
        populated = tuple(
            candidate for candidate in lines[line_index + 1 :] if candidate.text.strip()
        )
        if not populated:
            continue
        if len(caption_unit_tokens) == 1:
            numeric_line = populated[0]
            unit_line = line
            unit_token = caption_unit_tokens[0]
        elif not caption_unit_tokens and len(populated) >= 2:
            first_tokens = _surface_token_spans(populated[0].text)
            second_tokens = _surface_token_spans(populated[1].text)
            first_units = tuple(
                token
                for token in first_tokens
                if any(token[0] in values for values in _MEASUREMENT_UNIT_SURFACES.values())
            )
            second_units = tuple(
                token
                for token in second_tokens
                if any(token[0] in values for values in _MEASUREMENT_UNIT_SURFACES.values())
            )
            if len(first_tokens) == len(first_units) == 1:
                unit_line = populated[0]
                unit_token = first_units[0]
                numeric_line = populated[1]
            elif len(second_tokens) == len(second_units) == 1:
                numeric_line = populated[0]
                unit_line = populated[1]
                unit_token = second_units[0]
            else:
                continue
        else:
            continue
        numeric = numeric_line.text.strip()
        if re.fullmatch(r"[-+]?[0-9][0-9., ]*", numeric) is None:
            continue
        unit_candidates: list[tuple[str, str, bool]] = []
        for group_index, group in enumerate(cargo_groups):
            measurement = group.get(fields[0]) if isinstance(group, Mapping) else None
            if not isinstance(measurement, Mapping):
                continue
            value = measurement.get("value")
            unit = measurement.get("unit")
            if isinstance(unit, str) and unit_token[0] in _MEASUREMENT_UNIT_SURFACES.get(
                unit.casefold(), ()
            ):
                base = f"documentPatch.cargoGroups[{group_index}].{fields[0]}"
                value_matches = (
                    isinstance(value, (int, float, Decimal))
                    and not isinstance(value, bool)
                    and Decimal(str(value)) in _decimal_candidates(numeric, integer=False)
                )
                unit_candidates.append((base + ".value", base + ".unit", value_matches))
        exact_value_candidates = tuple(
            (value_path, unit_path)
            for value_path, unit_path, value_matches in unit_candidates
            if value_matches
        )
        candidates = (
            exact_value_candidates
            if len(exact_value_candidates) == 1
            else tuple((value_path, unit_path) for value_path, unit_path, _ in unit_candidates)
            if len(unit_candidates) == 1
            else ()
        )
        if len(candidates) != 1:
            continue
        _value_path, unit_path = candidates[0]
        owner = _unique_target_owner(drafts=drafts, target_path=unit_path)
        if owner is None:
            continue
        for row in drafts:
            if row.logical_key == owner.logical_key:
                replacements[row.draft_id] = replace(
                    row,
                    value_kind="other_text",
                    render_policy="natural_text",
                )
        start = unit_line.char_start + unit_token[1]
        end = unit_line.char_start + unit_token[2]
        overlaps = tuple(row for row in drafts if start < row.char_end and row.char_start < end)
        if overlaps:
            if all(
                row.logical_key == owner.logical_key
                and row.char_start == start
                and row.char_end == end
                for row in overlaps
            ):
                continue
            if not all(
                row.char_start == start
                and row.char_end == end
                and row.render_mode == "literal_static"
                and not row.target_paths
                and not row.dependency_paths
                and not row.dependency_bindings
                for row in overlaps
            ):
                continue
            discarded.update(row.draft_id for row in overlaps)
        additions.append(
            replace(
                owner,
                draft_id=(
                    "host_measurement_caption_unit_"
                    + sha256_bytes(f"{owner.logical_key}\0{start}\0{end}".encode())[:16]
                ),
                char_start=start,
                char_end=end,
                source_text=raw[start:end],
                value_kind="other_text",
                render_policy="natural_text",
                evidence_origin="host_verified_agent_proposal",
                rationale=(
                    owner.rationale
                    + " Host joined this nearby unit through the uniquely matching numeric "
                    "measurement value after its caption."
                ),
            )
        )
    return merge_drafts(
        tuple(
            replacements.get(row.draft_id, row) for row in drafts if row.draft_id not in discarded
        ),
        additions,
    )


_ORIGINAL_COUNT_WORDS = {
    "zero": 0,
    "one": 1,
    "two": 2,
    "three": 3,
    "four": 4,
    "five": 5,
    "six": 6,
    "seven": 7,
    "eight": 8,
    "nine": 9,
}


def _original_count_value(surface: str) -> int | None:
    numbers = {int(value) for value in re.findall(r"(?<![0-9])[0-9](?![0-9])", surface)}
    numbers.update(
        value
        for word, value in _ORIGINAL_COUNT_WORDS.items()
        if re.search(rf"(?i)\b{word}\b", surface) is not None
    )
    return next(iter(numbers)) if len(numbers) == 1 else None


def _is_original_count_caption(line: str) -> bool:
    if _ORIGINAL_BILL_COUNT_CAPTION.fullmatch(line) is not None:
        return True
    normalized = _normalized_surface(line)
    return ("numberoforiginal" in normalized or "nooforiginal" in normalized) and any(
        token in normalized for token in ("billsoflading", "bsl", "bls", "fbl")
    )


def normalize_original_bill_count_repeats(
    *, raw: str, drafts: Sequence[SpanDraft]
) -> tuple[SpanDraft, ...]:
    """Append repeated count values under proved original-bill captions and witness clauses."""

    lines = line_spans(raw)
    caption_values: list[tuple[int, int, int]] = []
    for index, line in enumerate(lines):
        if not _is_original_count_caption(line.text):
            continue
        value_line = next(
            (candidate for candidate in lines[index + 1 :] if candidate.text.strip()), None
        )
        if value_line is None:
            continue
        span = _trimmed_line_span(value_line)
        if span is None:
            continue
        value = _original_count_value(raw[span[0] : span[1]])
        if value is not None:
            caption_values.append((span[0], span[1], value))
    if not caption_values:
        return tuple(drafts)
    owner_keys = {
        row.logical_key
        for start, end, _value in caption_values
        for row in drafts
        if row.char_start == start
        and row.char_end == end
        and (
            "original" in row.logical_key.casefold()
            or "negotiability" in row.logical_key.casefold()
        )
    }
    if len(owner_keys) != 1:
        return tuple(drafts)
    logical_key = next(iter(owner_keys))
    owner_rows = tuple(row for row in drafts if row.logical_key == logical_key)
    owner = owner_rows[0]
    owner_counts = {
        count for row in owner_rows if (count := _original_count_value(row.source_text)) is not None
    }
    if len(owner_counts) != 1:
        return tuple(drafts)
    count = next(iter(owner_counts))
    candidates = {(start, end) for start, end, value in caption_values if value == count}
    word_pattern = "|".join(_ORIGINAL_COUNT_WORDS)
    for line in lines:
        if not (
            re.search(r"(?i)\boriginal\b", line.text)
            and re.search(r"(?i)\bbills?\s+of\s+lading\b", line.text)
        ):
            continue
        for match in re.finditer(
            rf"(?i)\b(?P<word>{word_pattern})\s*\(\s*(?P<number>[0-9])\s*\)",
            line.text,
        ):
            if (
                _ORIGINAL_COUNT_WORDS[match.group("word").casefold()]
                == int(match.group("number"))
                == count
            ):
                candidates.add((line.char_start + match.start(), line.char_start + match.end()))
    additions: list[SpanDraft] = []
    existing = {(row.char_start, row.char_end) for row in owner_rows}
    for start, end in sorted(candidates - existing):
        if any(start < row.char_end and row.char_start < end for row in drafts):
            continue
        additions.append(
            replace(
                owner,
                draft_id=(
                    "host_original_count_repeat_"
                    + sha256_bytes(f"{logical_key}\0{start}\0{end}".encode())[:16]
                ),
                char_start=start,
                char_end=end,
                source_text=raw[start:end],
                evidence_origin="host_verified_agent_proposal",
                rationale=(
                    owner.rationale
                    + " Host appended this equivalent original-bill count from a proved caption "
                    "or internally consistent word-and-digit witness clause."
                ),
            )
        )
    return merge_drafts(drafts, additions)


def normalize_repeated_original_status_marks(
    *, raw: str, drafts: Sequence[SpanDraft]
) -> tuple[SpanDraft, ...]:
    """Own repeated standalone ``ORIGINAL`` marks without absorbing form-title grammar."""

    title_proves_status = (
        re.search(
            r"(?im)^\s*ORIGINAL\s+BILL\s+OF\s+LADING(?:\s*\([^)]*\))?\s*$",
            raw,
        )
        is not None
    )
    existing_typed_status = any(
        row.source_text.casefold() == "original"
        and row.render_mode in {"deterministic_auxiliary", "literal_static"}
        and not row.target_paths
        and "original" in row.logical_key.casefold()
        for row in drafts
    )
    if not (title_proves_status or existing_typed_status):
        return tuple(drafts)
    spans = tuple(
        span
        for line in line_spans(raw)
        if _STANDALONE_ORIGINAL.fullmatch(line.text) is not None
        and (span := _trimmed_line_span(line)) is not None
        and raw[span[0] : span[1]] == "ORIGINAL"
    )
    if not spans:
        return tuple(drafts)
    existing_by_span: dict[tuple[int, int], SpanDraft | None] = {}
    for start, end in spans:
        overlaps = tuple(row for row in drafts if start < row.char_end and row.char_start < end)
        if not overlaps:
            existing_by_span[(start, end)] = None
            continue
        if not (
            len(overlaps) == 1
            and overlaps[0].char_start == start
            and overlaps[0].char_end == end
            and overlaps[0].render_mode in {"deterministic_auxiliary", "literal_static"}
            and not overlaps[0].target_paths
            and not overlaps[0].dependency_paths
            and not overlaps[0].dependency_bindings
        ):
            return tuple(drafts)
        existing_by_span[(start, end)] = overlaps[0]

    logical_key = "agent:document:original_mark"
    additions: list[SpanDraft] = []
    replaced_ids: set[str] = set()
    for start, end in spans:
        existing = existing_by_span[(start, end)]
        if existing is not None:
            replaced_ids.add(existing.draft_id)
            additions.append(
                replace(
                    existing,
                    logical_key=logical_key,
                    render_mode="deterministic_auxiliary",
                    value_kind="other_text",
                    group_kind="document",
                    group_key="document:original_status",
                    derivation=None,
                    render_policy="natural_text",
                )
            )
            continue
        additions.append(
            SpanDraft(
                draft_id=("host_original_status_" + sha256_bytes(f"{start}\0{end}".encode())[:16]),
                logical_key=logical_key,
                render_mode="deterministic_auxiliary",
                value_kind="other_text",
                group_kind="document",
                group_key="document:original_status",
                target_paths=(),
                derivation=None,
                dependency_paths=(),
                dependency_bindings=(),
                char_start=start,
                char_end=end,
                source_text=raw[start:end],
                evidence_origin="host_verified_agent_proposal",
                render_policy="natural_text",
                rationale=(
                    "Host owned this repeated standalone ORIGINAL status mark independently of "
                    "the literal ORIGINAL BILL OF LADING form title."
                ),
            )
        )
    retained = tuple(row for row in drafts if row.draft_id not in replaced_ids)
    return merge_drafts(retained, additions)


def normalize_projected_context(
    *, raw: str, drafts: Sequence[SpanDraft], source_target: Mapping[str, Any]
) -> tuple[SpanDraft, ...]:
    """Declare printed same-party geography and same-cargo MADE IN dependencies."""
    from types import SimpleNamespace

    from .projected_context import eligible_edges

    grouped: dict[str, list[SpanDraft]] = defaultdict(list)
    for draft in drafts:
        grouped[draft.logical_key].append(draft)
    candidates = []
    for key, rows in grouped.items():
        first = rows[0]
        if not any(
            path.startswith(("documentPatch.parties.", "documentPatch.cargoGroups."))
            or path.startswith("documentPatch.cargoGroups[")
            for path in first.target_paths
        ):
            continue
        slots = _template_slots(raw, rows)
        candidates.append(
            SimpleNamespace(
                logical_key=key,
                target_paths=first.target_paths,
                dependency_paths=first.dependency_paths,
                dependency_bindings=first.dependency_bindings,
                occurrences=slots,
                realization=binding_realization(
                    draft=first, slots=slots, source_target=source_target
                ),
            )
        )
    additions = {b.logical_key: eligible_edges(b, candidates) for b in candidates}
    output = []
    for row in drafts:
        edges = additions.get(row.logical_key, ())
        if edges:
            row = replace(
                row,
                dependency_paths=tuple(
                    dict.fromkeys((*row.dependency_paths, *(e.target_path for e in edges)))
                ),
                dependency_bindings=tuple(
                    dict.fromkeys((*row.dependency_bindings, *(e.owner_key for e in edges)))
                ),
            )
        output.append(row)
    return tuple(output)


def normalize_structured_row_locality(
    *, raw: str, drafts: Sequence[SpanDraft], source_target: Mapping[str, Any]
) -> tuple[SpanDraft, ...]:
    """Apply exact structured and caption-local joins to source ownership."""

    from .cargo_span_coalescing import normalize as normalize_complete_cargo_spans
    from .contact_values import normalize_attention_repeats
    from .equipment_projection import normalize as normalize_equipment_projection

    drafts = normalize_complete_cargo_spans(raw=raw, drafts=drafts, source_target=source_target)
    drafts = normalize_attention_repeats(raw=raw, drafts=drafts, source_target=source_target)
    drafts = normalize_equipment_projection(raw=raw, drafts=drafts, source_target=source_target)

    patch = source_target.get("documentPatch")
    packages = patch.get("cargoPackages") if isinstance(patch, Mapping) else None
    type_paths = (
        tuple(
            f"documentPatch.cargoPackages[{index}].typeCategory"
            for index, package in enumerate(packages)
            if isinstance(package, Mapping) and "typeCategory" in package
        )
        if isinstance(packages, list)
        else ()
    )
    page_normalized = normalize_page_header_ownership(raw=raw, drafts=drafts)
    quantity_normalized = normalize_package_quantity_row_locality(
        raw=raw,
        drafts=page_normalized,
    )
    package_normalized = normalize_package_type_row_locality(
        raw=raw,
        proposed=quantity_normalized,
        contexts=quantity_normalized,
        candidate_paths=type_paths,
        source_target=source_target,
    )
    package_quantity_normalized = normalize_package_quantity_repeats(
        raw=raw,
        drafts=package_normalized,
        source_target=source_target,
    )
    inclusive_range_normalized = normalize_inclusive_quantity_ranges(
        raw=raw,
        drafts=package_quantity_normalized,
        source_target=source_target,
    )
    cargo_linked_package_normalized = normalize_cargo_description_linked_package_prefixes(
        raw=raw,
        drafts=inclusive_range_normalized,
        source_target=source_target,
    )
    package_noun_normalized = normalize_source_only_package_nouns(
        raw=raw,
        drafts=cargo_linked_package_normalized,
    )
    equipment_normalized = normalize_compact_equipment_locality(
        raw=raw,
        drafts=package_noun_normalized,
        source_target=source_target,
    )
    from .equipment_projection import normalize_owned_auxiliary_receipts, normalize_receipt_suffixes

    equipment_normalized = normalize_receipt_suffixes(
        raw=raw, drafts=equipment_normalized, source_target=source_target
    )
    equipment_normalized = normalize_owned_auxiliary_receipts(
        raw=raw, drafts=equipment_normalized, source_target=source_target
    )
    container_package_normalized = normalize_container_linked_package_rows(
        raw=raw,
        drafts=equipment_normalized,
        source_target=source_target,
    )
    equipment_receipt_normalized = normalize_bare_equipment_multipliers(
        raw=raw,
        drafts=container_package_normalized,
        source_target=source_target,
    )
    complete_equipment_receipt_normalized = normalize_complete_equipment_receipt_surfaces(
        raw=raw,
        drafts=equipment_receipt_normalized,
        source_target=source_target,
    )
    container_count_normalized = normalize_abbreviated_container_counts(
        raw=raw,
        drafts=complete_equipment_receipt_normalized,
        source_target=source_target,
    )
    from . import shipment_totals

    shipment_total_normalized = shipment_totals.normalize(
        raw=raw, drafts=container_count_normalized, source_target=source_target
    )
    dangerous_goods_normalized = normalize_dangerous_goods_class_locality(
        raw=raw,
        drafts=shipment_total_normalized,
        source_target=source_target,
    )
    dangerous_goods_description_normalized = normalize_labeled_dangerous_goods_descriptions(
        raw=raw,
        drafts=dangerous_goods_normalized,
        source_target=source_target,
    )
    movement_normalized = normalize_labeled_movement_type_repeats(
        raw=raw,
        drafts=dangerous_goods_description_normalized,
    )
    caption_local_normalized = normalize_labeled_shipper_and_receipt_locality(
        raw=raw,
        drafts=movement_normalized,
        source_target=source_target,
    )
    segmented_address_normalized = normalize_segmented_party_address_targets(
        drafts=caption_local_normalized,
        source_target=source_target,
    )
    address_normalized = normalize_party_address_suffix_locality(
        raw=raw,
        drafts=segmented_address_normalized,
        source_target=source_target,
    )
    inline_address_normalized = normalize_party_address_inline_segments(
        raw=raw,
        drafts=address_normalized,
        source_target=source_target,
    )
    component_address_normalized = normalize_owned_party_address_components(
        drafts=inline_address_normalized,
        source_target=source_target,
    )
    missing_party_location_normalized = normalize_missing_party_locations_from_entity_blocks(
        raw=raw,
        drafts=component_address_normalized,
        source_target=source_target,
    )
    source_only_postal_city_normalized = normalize_source_only_party_postal_city_suffixes(
        raw=raw,
        drafts=missing_party_location_normalized,
    )
    party_location_normalized = normalize_party_location_repeats(
        raw=raw,
        drafts=source_only_postal_city_normalized,
        source_target=source_target,
    )
    party_location_variant_normalized = normalize_repeated_party_location_variants(
        raw=raw,
        drafts=party_location_normalized,
    )
    carrier_normalized = normalize_canonical_carrier_repeats(
        raw=raw,
        drafts=party_location_variant_normalized,
        source_target=source_target,
    )
    carrier_block_normalized = normalize_carrier_identity_block_details(
        raw=raw,
        drafts=carrier_normalized,
        source_target=source_target,
    )
    carrier_alias_normalized = normalize_canonical_carrier_initialism(
        raw=raw,
        drafts=carrier_block_normalized,
        source_target=source_target,
    )
    agent_appointment_normalized = normalize_agent_appointment_suffixes(
        raw=raw,
        drafts=carrier_alias_normalized,
        source_target=source_target,
    )
    source_field_normalized = normalize_labeled_source_only_fields(
        raw=raw,
        drafts=agent_appointment_normalized,
    )
    registration_type_normalized = normalize_foreign_exporter_registration_types(
        raw=raw,
        drafts=source_field_normalized,
    )
    labeled_registration_type_normalized = normalize_labeled_registration_type_values(
        raw=raw,
        drafts=registration_type_normalized,
    )
    country_code_normalized = normalize_labeled_country_code_derivations(
        raw=raw,
        drafts=labeled_registration_type_normalized,
    )
    export_country_normalized = normalize_exported_country_repeats(
        raw=raw,
        drafts=country_code_normalized,
    )
    measurement_normalized = normalize_selected_measurement_units(
        raw=raw,
        drafts=export_country_normalized,
    )
    measurement_caption_normalized = normalize_measurement_caption_units(
        raw=raw,
        drafts=measurement_normalized,
        source_target=source_target,
    )
    source_only_measurement_unit_normalized = normalize_adjacent_source_only_measurement_units(
        raw=raw,
        drafts=measurement_caption_normalized,
    )
    route_normalized = normalize_labeled_route_target_repeats(
        raw=raw,
        drafts=source_only_measurement_unit_normalized,
        source_target=source_target,
    )
    labeled_vessel_normalized = normalize_labeled_vessel_voyage_repeats(
        raw=raw,
        drafts=route_normalized,
        source_target=source_target,
    )
    inline_transport_normalized = normalize_inline_labeled_transport_repeats(
        raw=raw,
        drafts=labeled_vessel_normalized,
        source_target=source_target,
    )
    from .transport_derivations import normalize_explicit_names

    independent_transport_normalized = normalize_explicit_names(
        raw=raw, drafts=inline_transport_normalized, source_target=source_target
    )
    commercial_normalized = normalize_selected_commercial_dates_and_charges(
        raw=raw,
        drafts=independent_transport_normalized,
        source_target=source_target,
    )
    labeled_fact_normalized = normalize_labeled_identifiers_and_operational_facts(
        raw=raw,
        drafts=commercial_normalized,
        source_target=source_target,
    )
    forwarding_reference_normalized = normalize_repeated_forwarding_references(
        raw=raw,
        drafts=labeled_fact_normalized,
        source_target=source_target,
    )
    form_title_normalized = normalize_form_title_repeats(
        raw=raw,
        drafts=forwarding_reference_normalized,
    )
    original_count_normalized = normalize_original_bill_count_repeats(
        raw=raw,
        drafts=form_title_normalized,
    )
    original_status_normalized = normalize_repeated_original_status_marks(
        raw=raw,
        drafts=original_count_normalized,
    )
    terminal_normalized = normalize_explicit_loading_terminal_locality(
        raw=raw,
        drafts=original_status_normalized,
    )
    care_of_normalized = normalize_literal_care_of_separators(
        raw=raw,
        drafts=terminal_normalized,
    )
    onboard_vocabulary_normalized = normalize_shipped_on_board_vocabulary(
        raw=raw,
        drafts=care_of_normalized,
    )
    onboard_normalized = normalize_shipped_on_board_summary_locality(
        raw=raw,
        drafts=onboard_vocabulary_normalized,
        source_target=source_target,
    )
    operational_normalized = normalize_selected_operational_terms(
        raw=raw, drafts=onboard_normalized
    )
    freight_normalized = normalize_selected_freight_payment_locality(
        raw=raw,
        drafts=operational_normalized,
        source_target=source_target,
    )
    categorical_equipment = tuple(
        replace(draft, render_policy="natural_text")
        if draft.render_policy == "opaque_identifier"
        and draft.target_paths
        and all(
            re.fullmatch(r"documentPatch\.containers\[\d+\]\.typeDescription", path)
            for path in draft.target_paths
        )
        else draft
        for draft in freight_normalized
    )
    return normalize_projected_context(
        raw=raw, drafts=categorical_equipment, source_target=source_target
    )


def validate_compact_equipment_locality(
    *, raw: str, drafts: Sequence[SpanDraft], source_target: Mapping[str, Any]
) -> None:
    """Reject a short equipment surface assigned away from its unique local container row."""

    page_spans = page_body_spans(raw)
    lines = line_spans(raw)
    number_drafts: dict[int, list[SpanDraft]] = defaultdict(list)
    for draft in drafts:
        index = _container_index_for_path(draft.target_paths, _CONTAINER_NUMBER_PATH)
        if index is not None:
            number_drafts[index].append(draft)
    errors: list[str] = []
    for draft in drafts:
        if not _short_equipment_token(draft):
            continue
        target_index = _container_index_for_path(draft.target_paths, _CONTAINER_TYPE_PATH)
        auxiliary_match = re.fullmatch(r"container:([0-9]+)", draft.group_key)
        declared_index = (
            target_index
            if target_index is not None
            else int(auxiliary_match.group(1))
            if draft.render_mode == "deterministic_auxiliary"
            and draft.value_kind == "equipment"
            and auxiliary_match is not None
            else None
        )
        if declared_index is None:
            continue
        nearest_index = _nearest_container_number_index(
            raw=raw,
            draft=draft,
            number_drafts=number_drafts,
            page_spans=page_spans,
            lines=lines,
        )
        if nearest_index is not None and nearest_index != declared_index:
            errors.append(
                f"{draft.logical_key} compact equipment at chars "
                f"[{draft.char_start},{draft.char_end}) is scoped to container:{declared_index} "
                f"but uniquely nearest container:{nearest_index}"
            )
    if errors:
        raise ValueError("compact equipment locality violations: " + "; ".join(errors))


_HYPHENATED_NUMERIC_IDENTIFIER = re.compile(
    r"(?<![A-Za-z0-9])(?P<value>[0-9]{4,}(?:-[0-9]+)+)(?![A-Za-z0-9])"
)
_DECIMAL_MEASUREMENT_SUFFIX = re.compile(r"(?P<suffix>[,.][0-9]+)(?![A-Za-z0-9])")


def normalize_decimal_measurement_boundaries(
    *, raw: str, drafts: Sequence[SpanDraft]
) -> tuple[SpanDraft, ...]:
    """Expand a digit-only decimal measurement to its adjacent fractional suffix.

    The correction is deliberately limited to typed decimal measurements whose selected surface
    is only digits and whose immediately adjacent suffix is one unambiguous comma/dot fraction.
    It cannot cross another owner. This covers OCR totals such as ``43956,000`` without guessing
    about identifiers, addresses, or separated units.
    """

    output: list[SpanDraft] = []
    for draft in drafts:
        if draft.value_kind != "decimal_measurement" or not draft.source_text.isdigit():
            output.append(draft)
            continue
        suffix = _DECIMAL_MEASUREMENT_SUFFIX.match(raw, draft.char_end)
        if suffix is None:
            output.append(draft)
            continue
        end = suffix.end("suffix")
        if any(
            other.logical_key != draft.logical_key
            and draft.char_start < other.char_end
            and other.char_start < end
            for other in drafts
        ):
            output.append(draft)
            continue
        output.append(
            replace(
                draft,
                draft_id=(
                    "expanded_decimal_measurement_"
                    + sha256_bytes(f"{draft.draft_id}\0{draft.char_start}\0{end}".encode())[:16]
                ),
                char_end=end,
                source_text=raw[draft.char_start : end],
                rationale=(
                    draft.rationale
                    + " Host expanded the digit-only decimal measurement to its immediately "
                    "adjacent fractional suffix."
                ),
            )
        )
    return merge_drafts(output)


def normalize_auxiliary_identifier_boundaries(
    *, raw: str, drafts: Sequence[SpanDraft]
) -> tuple[SpanDraft, ...]:
    """Expand a repeated numeric auxiliary prefix to one proven complete hyphenated token.

    Source-only identifiers have no target scalar against which the host can validate their
    surface. Expansion is therefore intentionally narrow: every occurrence in the logical binding
    must be the same digit-only prefix, each must sit inside exactly one hyphenated numeric token,
    every complete token must be byte-identical, and the expanded spans must remain disjoint from
    every other owner. Anything else remains unchanged for semantic review.
    """

    grouped: dict[str, list[SpanDraft]] = defaultdict(list)
    for draft in drafts:
        grouped[draft.logical_key].append(draft)
    replacements: dict[str, SpanDraft] = {}
    for rows in grouped.values():
        first = rows[0]
        if not (
            first.render_mode == "deterministic_auxiliary"
            and first.value_kind == "identifier"
            and len(first.source_text) >= 4
            and first.source_text.isdigit()
            and all(
                row.render_mode == first.render_mode
                and row.value_kind == first.value_kind
                and row.source_text == first.source_text
                for row in rows
            )
        ):
            continue
        expansions: list[tuple[int, int, str]] = []
        for row in rows:
            candidates = tuple(
                (match.start("value"), match.end("value"), match.group("value"))
                for match in _HYPHENATED_NUMERIC_IDENTIFIER.finditer(raw)
                if match.start("value") <= row.char_start
                and row.char_end <= match.end("value")
                and (match.start("value"), match.end("value")) != (row.char_start, row.char_end)
            )
            if len(candidates) != 1:
                expansions = []
                break
            expansions.append(candidates[0])
        if not expansions or len({surface for _start, _end, surface in expansions}) != 1:
            continue
        if any(
            any(
                other.logical_key != first.logical_key
                and start < other.char_end
                and other.char_start < end
                for other in drafts
            )
            for start, end, _surface in expansions
        ):
            continue
        for row, (start, end, surface) in zip(rows, expansions, strict=True):
            replacements[row.draft_id] = replace(
                row,
                draft_id=(
                    "expanded_auxiliary_identifier_"
                    + sha256_bytes(f"{row.draft_id}\0{start}\0{end}".encode())[:16]
                ),
                char_start=start,
                char_end=end,
                source_text=surface,
                rationale=(
                    row.rationale
                    + " Host expanded the repeated numeric auxiliary prefix to its identical "
                    "complete hyphenated identifier token."
                ),
            )
    return merge_drafts(replacements.get(draft.draft_id, draft) for draft in drafts)


def normalize_labeled_auxiliary_identifier_boundaries(
    *, raw: str, drafts: Sequence[SpanDraft]
) -> tuple[SpanDraft, ...]:
    """Separate explicit field-number captions from their source-only value."""

    replacements: dict[str, SpanDraft] = {}
    for row in drafts:
        if not (
            row.render_mode == "deterministic_auxiliary"
            and row.value_kind == "identifier"
            and not row.target_paths
            and not row.dependency_paths
            and not row.dependency_bindings
        ):
            continue
        match = _LABELED_AUXILIARY_IDENTIFIER.fullmatch(row.source_text)
        if match is None:
            match = _NUMBER_CAPTION_IDENTIFIER.fullmatch(row.source_text)
        if match is None:
            continue
        start = row.char_start + match.start("value")
        end = row.char_start + match.end("value")
        if any(
            other.draft_id != row.draft_id and start < other.char_end and other.char_start < end
            for other in drafts
        ):
            continue
        replacements[row.draft_id] = replace(
            row,
            draft_id=(
                "host_labeled_auxiliary_identifier_"
                + sha256_bytes(f"{row.draft_id}\0{start}\0{end}".encode())[:16]
            ),
            char_start=start,
            char_end=end,
            source_text=raw[start:end],
            rationale=(
                row.rationale
                + " Host narrowed this source-only identifier to the complete value after its "
                "literal label separator."
            ),
        )
    return merge_drafts(replacements.get(row.draft_id, row) for row in drafts)


def normalize_country_code_locality(
    *, raw: str, drafts: Sequence[SpanDraft]
) -> tuple[SpanDraft, ...]:
    """Move a caption-fragment country code only to its unique labeled value on that line."""

    lines = line_spans(raw)
    output: list[SpanDraft] = []
    for draft in drafts:
        if draft.derivation != "country_code":
            output.append(draft)
            continue
        left_is_boundary = draft.char_start == 0 or not raw[draft.char_start - 1].isalnum()
        right_is_boundary = draft.char_end == len(raw) or not raw[draft.char_end].isalnum()
        caption_immediately_precedes = _COUNTRY_CODE_CAPTION.search(raw[: draft.char_start])
        if (left_is_boundary and right_is_boundary) or caption_immediately_precedes:
            output.append(draft)
            continue
        line = next(
            (
                row
                for row in lines
                if row.char_start <= draft.char_start and draft.char_end <= row.char_end
            ),
            None,
        )
        candidates: list[tuple[int, int]] = []
        if line is not None:
            line_text = raw[line.char_start : line.char_end]
            for match in _COUNTRY_CODE_LABELED_VALUE.finditer(line_text):
                start = line.char_start + match.start("value")
                end = line.char_start + match.end("value")
                if raw[start:end].casefold() != draft.source_text.casefold():
                    continue
                if any(
                    other is not draft and start < other.char_end and other.char_start < end
                    for other in drafts
                ):
                    continue
                candidates.append((start, end))
        if len(candidates) != 1:
            output.append(draft)
            continue
        start, end = candidates[0]
        output.append(
            replace(
                draft,
                char_start=start,
                char_end=end,
                source_text=raw[start:end],
                rationale=(
                    draft.rationale
                    + " Host moved a caption-fragment country code to the unique explicitly "
                    "labeled value on the same line."
                ),
            )
        )
    return merge_drafts(output)


def normalize_labeled_auxiliary_organization_boundaries(
    *, raw: str, drafts: Sequence[SpanDraft]
) -> tuple[SpanDraft, ...]:
    """Trim an exact importer-name caption from a source-only organization binding."""

    grouped: dict[str, list[SpanDraft]] = defaultdict(list)
    for row in drafts:
        grouped[row.logical_key].append(row)
    replacements: dict[str, SpanDraft] = {}
    for rows in grouped.values():
        first = rows[0]
        if not (
            first.render_mode in {"deterministic_auxiliary", "agent_residual"}
            and first.value_kind == "organization"
            and not first.target_paths
            and not first.dependency_paths
            and not first.dependency_bindings
        ):
            continue
        matches = tuple(_LABELED_AUXILIARY_ORGANIZATION.fullmatch(row.source_text) for row in rows)
        if any(match is None for match in matches):
            continue
        typed_matches = tuple(match for match in matches if match is not None)
        values = tuple(match.group("value") for match in typed_matches)
        if len({_normalized_surface(value) for value in values}) != 1:
            continue
        for row, match in zip(rows, typed_matches, strict=True):
            start = row.char_start + match.start("value")
            end = row.char_start + match.end("value")
            replacements[row.draft_id] = replace(
                row,
                draft_id=(
                    "host_labeled_auxiliary_value_"
                    + sha256_bytes(f"{row.draft_id}\0{start}\0{end}".encode())[:16]
                ),
                char_start=start,
                char_end=end,
                source_text=raw[start:end],
                evidence_origin="host_verified_agent_proposal",
                rationale=(
                    row.rationale
                    + " Host narrowed this source-only organization to the value after its "
                    "exact importer-name caption."
                ),
            )
    return merge_drafts(replacements.get(row.draft_id, row) for row in drafts)


def normalize_source_boundaries(*, raw: str, drafts: Sequence[SpanDraft]) -> tuple[SpanDraft, ...]:
    """Apply every host-proven source-boundary correction in canonical order."""

    return normalize_country_code_locality(
        raw=raw,
        drafts=normalize_labeled_auxiliary_organization_boundaries(
            raw=raw,
            drafts=normalize_auxiliary_identifier_boundaries(
                raw=raw,
                drafts=normalize_labeled_auxiliary_identifier_boundaries(
                    raw=raw,
                    drafts=normalize_decimal_measurement_boundaries(raw=raw, drafts=drafts),
                ),
            ),
        ),
    )


def validate_binding_realizations(
    *, raw: str, drafts: Sequence[SpanDraft], source_target: Mapping[str, Any]
) -> None:
    """Reject mutable bindings whose declared semantics cannot realize their own surfaces."""

    from .temperature_prose import require_temperature_classification

    validate_draft_source_alignment(raw=raw, drafts=drafts)
    require_temperature_classification(drafts)
    validate_compact_equipment_locality(raw=raw, drafts=drafts, source_target=source_target)
    quantity_paths = _structured_package_quantity_paths(source_target)
    split_range_errors: list[str] = []
    for match in inclusive_range_surfaces(raw):
        cardinality = match.cardinality
        start, end = match.char_start, match.char_end
        overlaps = tuple(
            draft for draft in drafts if start < draft.char_end and draft.char_start < end
        )
        linked_paths = {
            path
            for draft in overlaps
            for path in (*draft.target_paths, *draft.dependency_paths)
            if quantity_paths.get(path) == cardinality
        }
        endpoint_owned = all(
            all(
                not raw[offset].isalnum()
                or any(draft.char_start <= offset < draft.char_end for draft in overlaps)
                for offset in range(endpoint_start, endpoint_end)
            )
            for endpoint_start, endpoint_end in (
                (match.start_start, match.start_end),
                (match.end_start, match.end_end),
            )
        )
        full_owner = any(draft.char_start <= start and end <= draft.char_end for draft in overlaps)
        if len(linked_paths) == 1 and endpoint_owned and not full_owner:
            split_range_errors.append(
                f"printed interval {raw[start:end]!r} at chars [{start},{end}) is split across "
                "independent bindings; compile one inclusive_range_cardinality surface"
            )
    lines = line_spans(raw)
    slots = _template_slots(raw, drafts)
    grouped: dict[str, list[tuple[SpanDraft, TemplateSlot]]] = defaultdict(list)
    for draft, slot in zip(drafts, slots, strict=True):
        grouped[draft.logical_key].append((draft, slot))
    from types import SimpleNamespace

    from . import mixed_inventory

    aggregate_keys: frozenset[str] = frozenset()
    if (
        sum(
            rows[0][0].value_kind == "equipment"
            and not rows[0][0].target_paths
            and rows[0][0].dependency_paths == ("documentPatch.containers",)
            and rows[0][0].derivation != "same_as_binding"
            for rows in grouped.values()
        )
        >= 2
    ):
        aggregate_bindings = tuple(
            SimpleNamespace(
                logical_key=key,
                value_kind=rows[0][0].value_kind,
                target_paths=rows[0][0].target_paths,
                render_mode=rows[0][0].render_mode,
                derivation=rows[0][0].derivation,
                dependency_paths=rows[0][0].dependency_paths,
                dependency_bindings=rows[0][0].dependency_bindings,
                occurrences=tuple(slot for _, slot in rows),
            )
            for key, rows in grouped.items()
        )
        aggregate = mixed_inventory.compile_bindings(
            raw.encode(), source_target, aggregate_bindings
        )
        if aggregate is not None:
            aggregate_keys = frozenset(aggregate.binding_keys)
    errors: list[str] = []
    errors.extend(split_range_errors)
    for logical_key, rows in grouped.items():
        first = rows[0][0]
        from . import cargo_identity_derivations, dangerous_goods_realization, transport_derivations
        from .route_derivations import DERIVATIONS as route_derivations
        from .route_derivations import endpoint

        if first.derivation == "gross_minus_net_weight":
            from types import SimpleNamespace

            from .descendant import _render_gross_minus_net_weight

            candidate = SimpleNamespace(
                **{
                    name: getattr(first, name)
                    for name in (
                        "logical_key",
                        "target_paths",
                        "dependency_paths",
                        "dependency_bindings",
                        "render_mode",
                        "value_kind",
                        "group_kind",
                    )
                },
                occurrences=tuple(slot for _, slot in rows),
            )
            case = SimpleNamespace(
                source=raw.encode(),
                source_target=source_target,
                target=source_target,
                template=SimpleNamespace(
                    bindings=tuple(
                        SimpleNamespace(target_paths=d.target_paths, occurrences=(slot,))
                        for d, slot in zip(drafts, slots, strict=True)
                    )
                ),
            )
            try:
                _render_gross_minus_net_weight(cast(Any, candidate), cast(Any, case))
            except (ValueError, KeyError, IndexError) as error:
                errors.append(f"{logical_key}: {error}")

        if first.derivation in dangerous_goods_realization.DERIVATIONS:
            declarations = {
                f"documentPatch.cargoGroups[{gi}].dangerousGoods[{di}]"
                for gi, group in enumerate(source_target["documentPatch"].get("cargoGroups", ()))
                for di, _ in enumerate(group.get("dangerousGoods", ()))
            }
            for draft, _ in rows:
                try:
                    dangerous_goods_realization.explicit_surface(draft, declarations)
                except ValueError as error:
                    errors.append(f"{logical_key}: {error}")

        if first.derivation in transport_derivations.DERIVATIONS:
            try:
                transport_derivations.validate(first)
            except ValueError as error:
                errors.append(f"{logical_key}: {error}")

        if first.derivation in cargo_identity_derivations.DERIVATIONS:
            try:
                cargo_identity_derivations.owner(first)
            except ValueError as error:
                errors.append(f"{logical_key}: {error}")

        if first.derivation in route_derivations:
            try:
                endpoint(first)
            except (KeyError, ValueError) as error:
                errors.append(f"{logical_key}: {error}")
        if first.derivation == "inclusive_range_cardinality":
            expected = None
            valid_path_contract = (
                len(first.dependency_paths) == 1
                and first.dependency_paths[0] in quantity_paths
                and first.target_paths in {(), first.dependency_paths}
                and not first.dependency_bindings
                and first.render_mode == "deterministic_derived"
                and first.value_kind == "integer"
            )
            valid_binding_contract = (
                not first.dependency_paths
                and not first.target_paths
                and len(first.dependency_bindings) == 1
                and first.dependency_bindings[0] != logical_key
                and first.render_mode == "deterministic_derived"
                and first.value_kind == "integer"
            )
            if valid_path_contract:
                expected = quantity_paths[first.dependency_paths[0]]
            elif valid_binding_contract:
                from .descendant import _numeric_value

                owners = grouped.get(first.dependency_bindings[0], ())
                try:
                    values = {_numeric_value(owner.source_text) for owner, _ in owners}
                    if (
                        not owners
                        or any(
                            owner.value_kind not in {"integer", "package"}
                            or owner.group_kind != "cargo"
                            for owner, _ in owners
                        )
                        or len(values) != 1
                    ):
                        raise ValueError("range needs one explicit cargo quantity binding")
                    value = values.pop()
                    if value <= 0 or value != value.to_integral_value():
                        raise ValueError("range quantity binding must be a positive integer")
                    expected = int(value)
                except ValueError as error:
                    errors.append(f"{logical_key}: {error}")
            else:
                errors.append(
                    f"{logical_key} inclusive_range_cardinality has an invalid quantity-owner "
                    "contract"
                )
            if expected is not None:
                for draft, _slot in rows:
                    range_surface = _inclusive_range_surface_match(draft.source_text)
                    observed = range_surface.cardinality if range_surface is not None else None
                    if observed != expected:
                        errors.append(
                            f"{logical_key} inclusive interval {draft.source_text!r} has "
                            f"cardinality {observed}, expected {expected}"
                        )
        if first.derivation == "package_count":
            invalid_dependencies = tuple(
                path
                for path in first.dependency_paths
                if not isinstance(_resolve_target_path(source_target, path), (Mapping, list))
            )
            if invalid_dependencies:
                errors.append(
                    f"{logical_key} package_count uses scalar dependencies: "
                    + ", ".join(invalid_dependencies)
                )
        if first.derivation == "equipment_receipt" and logical_key not in aggregate_keys:
            from .descendant import _number_to_words
            from .equipment_receipts import owned_inventory, validate_source_receipt

            try:
                inventory = owned_inventory(
                    source_target, (*first.target_paths, *first.dependency_paths)
                )
                for draft, _slot in rows:
                    validate_source_receipt(
                        draft.source_text, inventory, number_words=_number_to_words
                    )
            except ValueError as error:
                errors.append(f"{logical_key}: {error}")
        if "original" in logical_key.casefold() and any(
            token in logical_key.casefold() for token in ("mark", "status")
        ):
            for draft, _slot in rows:
                if draft.source_text.casefold() != "original":
                    continue
                line = lines[line_number_for_char(lines, draft.char_start) - 1]
                if _STANDALONE_ORIGINAL.fullmatch(line.text) is None:
                    errors.append(
                        f"{logical_key} original-status owner selects caption grammar at chars "
                        f"[{draft.char_start},{draft.char_end}); select only a standalone mark"
                    )
        if first.derivation == "container_count":
            for draft, _slot in rows:
                if _BARE_EQUIPMENT_MULTIPLIER.fullmatch(draft.source_text) is not None:
                    errors.append(
                        f"{logical_key} container_count selects bare equipment multiplier "
                        f"{draft.source_text!r} at chars [{draft.char_start},{draft.char_end}); "
                        "use the row-local equipment_receipt derivation"
                    )
        if first.derivation == "country_code":
            for draft, _slot in rows:
                left_is_boundary = draft.char_start == 0 or not raw[draft.char_start - 1].isalnum()
                right_is_boundary = draft.char_end == len(raw) or not raw[draft.char_end].isalnum()
                caption_immediately_precedes = _COUNTRY_CODE_CAPTION.search(raw[: draft.char_start])
                if not (left_is_boundary and right_is_boundary) and not (
                    caption_immediately_precedes
                ):
                    errors.append(
                        f"{logical_key} country_code derivation selects an alphanumeric "
                        f"caption/word fragment at chars [{draft.char_start},{draft.char_end}); "
                        "select the printed country-code value occurrence"
                    )
        group_slots = tuple(slot for _draft, slot in rows)
        realization = binding_realization(
            draft=first,
            slots=group_slots,
            source_target=source_target,
        )
        if first.render_mode == "deterministic_auxiliary" and realization.requires_agent:
            errors.append(
                f"{logical_key} is declared deterministic_auxiliary but its "
                "non-equivalent occurrences require an agent"
            )
        if first.render_mode not in {"deterministic_auxiliary", "agent_residual"}:
            continue
        projected_paths = {
            path
            for draft, _slot in rows
            for path in _projected_equipment_type_paths(draft.source_text, source_target)
        }
        if (
            projected_paths
            and not projected_paths.intersection(first.target_paths)
            and not _is_explicit_missing_target_equipment_auxiliary(first, source_target)
        ):
            errors.append(
                f"{logical_key} owns an equipment-type token as source-only text; split the "
                "mixed surface and target-bind the equipment token to one of: "
                + ", ".join(sorted(projected_paths))
            )
    if errors:
        raise ValueError("binding realization contract violations: " + "; ".join(errors))


def _source_binding_relationships(
    grouped: Mapping[str, Sequence[tuple[SpanDraft, TemplateSlot]]],
) -> dict[str, tuple[SourceBindingRelationship, ...]]:
    """Record exact containment between separately owned mutable identifiers.

    These relationships are evidence, not a guess about an identifier standard. They let a
    descendant renderer preserve an observed dependency or route it to the explicitly declared
    residual agent instead of generating mutually inconsistent values.
    """

    unique_identifier_surfaces: dict[str, tuple[SpanDraft, str]] = {}
    for logical_key, rows in grouped.items():
        first = rows[0][0]
        values = {slot.source_text for _draft, slot in rows}
        if (
            first.value_kind == "identifier"
            and first.render_mode not in {"carrier_static", "literal_static"}
            and len(values) == 1
        ):
            unique_identifier_surfaces[logical_key] = (first, next(iter(values)))

    output: dict[str, tuple[SourceBindingRelationship, ...]] = {
        logical_key: () for logical_key in grouped
    }
    for owner_key, (owner, owner_surface) in unique_identifier_surfaces.items():
        relationship_rows: list[SourceBindingRelationship] = []
        if owner.render_mode in {"deterministic_auxiliary", "agent_residual"}:
            for dependency_key, (
                dependency,
                dependency_surface,
            ) in unique_identifier_surfaces.items():
                if (
                    dependency_key == owner_key
                    or dependency.render_mode == "deterministic_derived"
                    # The shorter value is the embedded identifier regardless of which
                    # binding owns this loop iteration.  Requiring only the dependency to
                    # be long admitted incidental one-character/digit overlaps whenever
                    # the owner happened to be the embedded side.
                    or min(len(dependency_surface), len(owner_surface)) < 6
                    or len(dependency_surface) == len(owner_surface)
                ):
                    continue
                if (
                    len(dependency_surface) < len(owner_surface)
                    and owner_surface.count(dependency_surface) == 1
                ):
                    containing_surface = owner_surface
                    embedded_surface = dependency_surface
                    relationship = "embeds_exact_source_identifier"
                elif (
                    len(owner_surface) < len(dependency_surface)
                    and dependency_surface.count(owner_surface) == 1
                ):
                    containing_surface = dependency_surface
                    embedded_surface = owner_surface
                    relationship = "is_embedded_in_exact_source_identifier"
                else:
                    continue
                start = containing_surface.index(embedded_surface)
                prefix = containing_surface[:start]
                suffix = containing_surface[start + len(embedded_surface) :]
                relationship_rows.append(
                    SourceBindingRelationship.model_validate(
                        {
                            "dependency_binding": dependency_key,
                            "relationship": relationship,
                            "source_prefix": prefix,
                            "source_suffix": suffix,
                            "source_prefix_pattern": surface_pattern(prefix) if prefix else None,
                            "source_suffix_pattern": surface_pattern(suffix) if suffix else None,
                        }
                    )
                )
        output[owner_key] = tuple(
            sorted(
                relationship_rows,
                key=lambda row: (row.dependency_binding, row.source_prefix),
            )
        )
    return output


def certify_template(
    *,
    raw: str,
    document_id: str,
    feature: Mapping[str, Any],
    source_target: Mapping[str, Any],
    assessment: CarrierAssessment,
    drafts: Sequence[SpanDraft],
    risks: Sequence[RiskCandidate],
    critic_outputs: Sequence[CriticAgentOutput],
    semantic_only_target_facts: Sequence[SemanticOnlyTargetFact] = (),
    coherence_constraints: Sequence[CoherenceConstraint] = (),
) -> CertifiedSemanticTemplate:
    if not critic_outputs or critic_outputs[-1].verdict != "pass":
        raise ValueError("template lacks a final independent critic pass")
    from .temperature_prose import require_singleton_printed_setpoint

    require_singleton_printed_setpoint(source_target)
    validate_compiler_extraction_dates(
        raw=raw,
        source_target=source_target,
        drafts=drafts,
        semantic_only_target_facts=semantic_only_target_facts,
    )
    validate_single_printed_hs_scope(
        raw=raw,
        source_target=source_target,
        drafts=drafts,
        semantic_only_target_facts=semantic_only_target_facts,
    )
    validate_global_shared_temperature_scope(
        raw=raw,
        source_target=source_target,
        drafts=drafts,
        semantic_only_target_facts=semantic_only_target_facts,
    )
    validate_repeated_cargo_temperature_scope(raw=raw, source_target=source_target, drafts=drafts)
    validate_signed_temperature_word_scope(raw=raw, source_target=source_target, drafts=drafts)
    validate_source_seal_ownership(drafts=drafts, source_target=source_target)
    validate_target_binding_relationships(drafts=drafts, source_target=source_target, raw=raw)
    validate_binding_realizations(raw=raw, drafts=drafts, source_target=source_target)
    validate_coherence_contracts(
        raw=raw,
        bindings=drafts,
        source_target=source_target,
        constraints=coherence_constraints,
    )
    expected_carrier = source_carrier(source_target)
    carrier_drafts = tuple(draft for draft in drafts if draft.render_mode == "carrier_static")
    validate_carrier_assessment(
        assessment=assessment,
        expected=expected_carrier,
        raw=raw,
        anchor_drafts_value=carrier_drafts,
    )
    uncovered = uncovered_risks(raw, risks, drafts)
    if uncovered:
        raise ValueError(
            "template leaves deterministic risk candidates unowned: "
            + ", ".join(row.risk_id for row in uncovered)
        )
    owned_target_paths = {path for draft in drafts for path in draft.target_paths}
    duplicate_semantic_paths = sorted(
        {
            fact.target_path
            for fact in semantic_only_target_facts
            if fact.target_path in owned_target_paths
        }
    )
    if duplicate_semantic_paths:
        raise ValueError(
            "semantic-only target facts also have source bindings: "
            + ", ".join(duplicate_semantic_paths)
        )
    slots = _template_slots(raw, drafts)
    byte_template = compile_raw_text_template(
        document_id=document_id, source=raw.encode("utf-8"), slots=slots
    )
    round_trip, round_trip_proof = render_compiled_template(
        source=raw.encode("utf-8"),
        template=byte_template,
        bindings={slot.slot_id: slot.source_text for slot in slots},
    )
    if round_trip != raw.encode("utf-8") or not round_trip_proof.source_round_trip:
        raise ValueError("compiled semantic template does not round-trip its source")
    _sentinel, sentinel_proof = render_compiled_template(
        source=raw.encode("utf-8"),
        template=byte_template,
        bindings=sentinel_bindings(byte_template),
        validate_format=False,
    )

    slots_by_draft = dict(zip(drafts, slots, strict=True))
    grouped: dict[str, list[tuple[SpanDraft, TemplateSlot]]] = defaultdict(list)
    for draft in drafts:
        grouped[draft.logical_key].append((draft, slots_by_draft[draft]))
    ordered_groups = sorted(
        grouped.items(), key=lambda row: min(item[0].char_start for item in row[1])
    )
    source_relationships = _source_binding_relationships(grouped)
    bindings: list[SemanticBinding] = []
    for index, (logical_key, rows) in enumerate(ordered_groups, start=1):
        first = rows[0][0]
        for draft, _slot in rows[1:]:
            comparable = (
                draft.render_mode,
                draft.value_kind,
                draft.group_kind,
                draft.group_key,
                draft.target_paths,
                draft.derivation,
                draft.dependency_paths,
                draft.dependency_bindings,
            )
            expected = (
                first.render_mode,
                first.value_kind,
                first.group_kind,
                first.group_key,
                first.target_paths,
                first.derivation,
                first.dependency_paths,
                first.dependency_bindings,
            )
            if comparable != expected:
                raise ValueError(f"logical binding has inconsistent occurrences: {logical_key}")
        bindings.append(
            SemanticBinding.model_validate(
                {
                    "binding_id": f"binding_{index:04d}",
                    "logical_key": logical_key,
                    "render_mode": first.render_mode,
                    "value_kind": first.value_kind,
                    "group_kind": first.group_kind,
                    "group_key": first.group_key,
                    "target_paths": first.target_paths,
                    "target_relationship": target_path_relationship(
                        source_target, first.target_paths
                    ),
                    "derivation": first.derivation,
                    "dependency_paths": first.dependency_paths,
                    "dependency_bindings": first.dependency_bindings,
                    "occurrences": tuple(slot for _draft, slot in rows),
                    "realization": binding_realization(
                        draft=first,
                        slots=tuple(slot for _draft, slot in rows),
                        source_target=source_target,
                    ),
                    "source_relationships": source_relationships[logical_key],
                    "rationale": first.rationale,
                }
            )
        )
    logical_keys = {binding.logical_key for binding in bindings}
    for binding in bindings:
        missing = sorted(set(binding.dependency_bindings) - logical_keys)
        if missing:
            raise ValueError(
                f"derived binding {binding.logical_key} has unknown dependencies: "
                + ", ".join(missing)
            )
    validate_seal_realization(
        target=source_target,
        bindings=bindings,
        slot_values={slot.slot_id: slot.source_text for slot in slots},
    )
    graph = {
        binding.logical_key: binding.dependency_bindings
        for binding in bindings
        if binding.dependency_bindings
    }
    visiting: set[str] = set()
    visited: set[str] = set()

    def visit(logical_key: str) -> None:
        if logical_key in visited:
            return
        if logical_key in visiting:
            raise ValueError(f"derived binding dependency cycle includes {logical_key}")
        visiting.add(logical_key)
        for dependency in graph.get(logical_key, ()):
            visit(dependency)
        visiting.remove(logical_key)
        visited.add(logical_key)

    for logical_key in graph:
        visit(logical_key)
    from . import (
        cargo_identity_derivations,
        labelled_context,
        route_derivations,
        transport_derivations,
    )

    if any(binding.derivation in route_derivations.DERIVATIONS for binding in bindings):
        route_derivations.validate_source(source_target, bindings)
    transport_derivations.validate_source(bindings, source_target)
    cargo_identity_derivations.validate_source(bindings, source_target)
    cargo_identity_derivations.require_tariff_owners(raw.encode("utf-8"), bindings)
    labelled_context.validate_source(bindings, source_target)
    from .measurement_prose import require_product_count_owners

    require_product_count_owners(raw.encode("utf-8"), bindings)
    auxiliary_semantic_plan = build_auxiliary_semantic_plan(
        raw=raw,
        bindings=bindings,
        source_target=source_target,
    )
    validate_party_evidence(
        target=source_target,
        binding_paths={path for binding in bindings for path in binding.target_paths},
        party_surfaces=party_owned_surfaces(
            bindings,
            {slot.slot_id: slot.source_text for slot in slots},
            auxiliary_semantic_plan.entities,
        ),
    )
    masked = masked_source(raw, drafts)
    return CertifiedSemanticTemplate.model_validate(
        {
            "schema_version": 6,
            "compiler": "carrier_bound_semantic_template_v6",
            "document_id": document_id,
            "source_sha256": sha256_bytes(raw.encode("utf-8")),
            "source_size_bytes": len(raw.encode("utf-8")),
            "carrier": CarrierBinding.model_validate(
                {
                    "canonical_name": assessment.canonical_name,
                    "family": (
                        feature["carrier_family"]
                        if feature["carrier_family"] not in (None, "<MISSING>")
                        else "OCR_RESOLVED::" + assessment.canonical_name
                    ),
                    "aliases": assessment.aliases,
                    "evidence_occurrences": assessment.evidence_occurrences,
                    "source": assessment.source,
                }
            ),
            "capability": capability_contract(feature, source_target),
            "semantic_only_target_facts": tuple(semantic_only_target_facts),
            "coherence_constraints": tuple(coherence_constraints),
            "auxiliary_semantic_plan": auxiliary_semantic_plan,
            "bindings": tuple(bindings),
            "byte_template": byte_template,
            "literal_certification": LiteralCertification.model_validate(
                {
                    "masked_literal_sha256": sha256_bytes(masked.encode("utf-8")),
                    "final_critic_pass": True,
                    "critic_passes": len(critic_outputs),
                    "remaining_unowned_risk_candidates": 0,
                }
            ),
            "certification": TemplateCertification.model_validate(
                {
                    "source_hash_valid": True,
                    "source_round_trip": True,
                    "disjoint_utf8_spans": True,
                    "exact_literal_regions": sentinel_proof.exact_literal_regions,
                    "page_markers_unchanged": sentinel_proof.page_markers_unchanged,
                    "line_endings_preserved": sentinel_proof.line_endings_preserved,
                    "sentinel_isolation": True,
                    "carrier_resolution_valid": True,
                    "carrier_matches_source_label": (
                        assessment.canonical_name == expected_carrier
                        if expected_carrier is not None
                        else None
                    ),
                    "all_risk_candidates_owned": True,
                    "final_critic_pass": True,
                    "all_bindings_realization_planned": True,
                    "all_unprinted_target_facts_classified": True,
                    "auxiliary_semantic_plan_valid": True,
                    "semantic_coherence_valid": True,
                }
            ),
        }
    )


def template_summary(template: CertifiedSemanticTemplate) -> dict[str, Any]:
    modes: dict[str, int] = defaultdict(int)
    realization_modes: dict[str, int] = defaultdict(int)
    values: dict[str, int] = defaultdict(int)
    coherence_kinds: dict[str, int] = defaultdict(int)
    for binding in template.bindings:
        modes[binding.render_mode] += 1
        realization_modes[binding.realization.mode] += 1
        values[binding.value_kind] += 1
    for constraint in template.coherence_constraints:
        coherence_kinds[constraint.kind] += 1
    return {
        "documentId": template.document_id,
        "sourceSha256": template.source_sha256,
        "carrier": template.carrier.canonical_name,
        "carrierFamily": template.carrier.family,
        "templateProxyId": template.capability.template_proxy_id,
        "routeCapabilities": {
            "transshipment": any(
                path == "documentPatch.route.transshipmentPort"
                or path.startswith("documentPatch.route.transshipmentPort.")
                for binding in template.bindings
                for path in (*binding.target_paths, *binding.dependency_paths)
            ),
        },
        "documentType": template.capability.document_type,
        "pages": template.capability.page_count,
        "lines": template.capability.line_count,
        "characters": template.capability.character_count,
        "bindings": len(template.bindings),
        "occurrences": len(template.byte_template.slots),
        "renderModes": dict(sorted(modes.items())),
        "realizationModes": dict(sorted(realization_modes.items())),
        "valueKinds": dict(sorted(values.items())),
        "agentResidualBindings": modes.get("agent_residual", 0),
        "agentAssistedBindings": realization_modes.get("agent_required", 0),
        "semanticOnlyTargetFacts": len(template.semantic_only_target_facts),
        "deterministicBindings": sum(realization_modes.values())
        - realization_modes.get("agent_required", 0),
        "coherenceContracts": dict(sorted(coherence_kinds.items())),
        "coherenceBindings": len(
            {
                logical_key
                for constraint in template.coherence_constraints
                for logical_key in constraint.member_logical_keys
            }
        ),
        "auxiliaryEntities": len(template.auxiliary_semantic_plan.entities),
        "auxiliaryCompositeNumbers": len(template.auxiliary_semantic_plan.composite_numbers),
        "auxiliaryDocumentSequences": len(template.auxiliary_semantic_plan.document_sequences),
        "auxiliaryBindingDispositions": len(template.auxiliary_semantic_plan.dispositions),
        "compiler": template.compiler,
        "certified": True,
    }
