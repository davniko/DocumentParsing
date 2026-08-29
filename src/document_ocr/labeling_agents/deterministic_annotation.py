"""Deterministic projection and OCR evidence for compact agent labels.

Luna selects semantic facts and relationships once.  This module performs the
mechanical work that previously consumed most model output: projecting the
normal cargo view, grounding every normal target leaf in raw OCR, and building
the relationship-evidence sidecar.  Every normalization is exact and
fail-closed; fuzzy edit-distance matching is deliberately absent.
"""

from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass
from datetime import date, datetime
from decimal import Decimal, InvalidOperation
from itertools import pairwise
from typing import Any, Literal

from pydantic import TypeAdapter, ValidationError

from document_ocr.label_schemas.bill_of_lading import BillOfLadingLabel
from document_ocr.label_schemas.bill_of_lading_v3 import (
    BillOfLadingDualCargoAnnotation,
    BillOfLadingRelationExplicitLabel,
    CargoRelationEvidence,
    project_relation_to_normal,
)
from document_ocr.label_schemas.common import FieldEvidence, RawOcrValueEvidence
from document_ocr.label_schemas.mpci_bill_of_lading import ContainerIdentifier
from document_ocr.labeling_agents.models import CompactAnnotationDraft
from document_ocr.labeling_agents.review_policy import (
    has_flash_point_context,
    named_non_order_consignee_spans,
    non_negotiable_copy_is_only_copy_status,
)
from document_ocr.labeling_agents.work_items import (
    AgentWorkItem,
    WorkItemError,
    page_texts,
    validate_annotation_evidence,
)


class DeterministicAnnotationError(WorkItemError):
    """A model fact cannot be projected or grounded without an assumption."""


_NUMBER = re.compile(
    r"(?<![A-Za-z0-9])(?<![0-9]['\u2019\u2032\"])[-+]?(?:"
    r"[0-9]{1,3}(?:\.[0-9]{3})+,[0-9]+|"
    r"[0-9]{1,3}(?:,[0-9]{3})+\.[0-9]+|"
    r"[0-9]{1,3}(?:[ ,][0-9]{3})+(?:\.[0-9]+)?|"
    r"[0-9]+(?:[.,][0-9]+)?"
    r")"
    r"(?=$|[^A-Za-z0-9]|[xX]|(?:KGS?|KGM|LBS?|M/?TS?|TONNES?|CBM|MTQ|M(?:3|³)|PCS?|"
    r"CS|PK|PKGS?|PACKAGES?|CARTONS?|PIECES?|"
    r"CTNS?|BAGS?|PLTS?|PALLETS?|BUNDLES?|DRUMS?|CRATES?|CASES?|BULLS?|CATTLE|HEADS?)\b)"
)
_COUNT_NUMBER = re.compile(
    r"(?ix)(?<![A-Za-z0-9])(?P<value>[0-9]+)"
    r"(?=\s*(?:PCS?|PIECES?|CS|PK|PKGS?|PACKAGES?|CARTONS?|CTNS?|BAGS?|"
    r"PLTS?|PALLETS?|BUNDLES?|DRUMS?|CRATES?|CASES?|BULLS?|CATTLE|HEADS?)\b)"
)
_GLUED_CELSIUS_TEMPERATURE = re.compile(
    r"(?ix)(?<![A-Za-z0-9])[-+]?[0-9]+(?:[.,][0-9]+)?"
    r"(?=\s*(?:\N{DEGREE SIGN}\s*)?C(?:\.?\s*C\.?(?:\s*C\.?)?)?\b)"
)
_MONTH_NAME = (
    r"jan(?:uary)?|feb(?:ruary)?|mar(?:ch)?|apr(?:il)?|may|jun(?:e)?|"
    r"jul(?:y)?|aug(?:ust)?|sep(?:tember)?|oct(?:ober)?|nov(?:ember)?|"
    r"dec(?:ember)?"
)
_DATE_TOKEN = re.compile(
    r"(?ix)\b(?:"
    r"[0-9]{4}[-/.][0-9]{1,2}[-/.][0-9]{1,2}|"
    rf"[0-9]{{4}}[-/.](?:{_MONTH_NAME})\.?[-/.][0-9]{{1,2}}|"
    r"[0-9]{1,2}[-/.][0-9]{1,2}[-/.][0-9]{2,4}|"
    rf"(?:[0-9]\s+[0-9]|[0-9]{{1,2}})(?:st|nd|rd|th)?\s*[-/. ]\s*"
    rf"(?:{_MONTH_NAME})\.?"
    r"\s*[-/., ]+\s*[0-9]{2,4}|"
    rf"(?:{_MONTH_NAME})\.?\s*(?:[-/.,]\s*|\s+)"
    r"[0-9]{1,2}(?:st|nd|rd|th)?\s*(?:,\s*|[-/.]\s*|\s+)"
    r"[0-9]{2,4}"
    r")\b"
)
_NUMERIC_DATE_TOKEN = re.compile(
    r"(?<![0-9])(?P<first>[0-9]{1,2})[-/.](?P<second>[0-9]{1,2})"
    r"[-/.](?P<year>[0-9]{2,4})(?![0-9])"
)
_FORBIDDEN_METADATA = re.compile(
    r"(?ix)\b(?:acid(?:\s*(?:code|number|no))?|tax(?:\s*(?:id|number|no))?|"
    r"vat(?:\s*(?:id|number|no))?|c\.?n\.?p\.?j|customs\s+(?:id|number|no)|"
    r"company\s+registration|exporter\s+registration|importer\s+registration|"
    r"gpc(?:\s*(?:id|number|no))?|blockchain|document\s+hash|portal\s+id)\b"
)
_NON_METADATA_FIELD_HEADING = re.compile(
    r"(?ix)\b(?:VIN(?:\s+NUMBERS?)?|COMM(?:ERCIAL)?\.?\s*(?:NO|NUMBER)|"
    r"ENGINE\s*(?:NO|NUMBER)|INV(?:OICE)?\.?\s*(?:NO|NUMBER|REF)|"
    r"H\.?S\.?N?\.?\s*(?:CODE|NO|NUMBER)?|HTS|TARIFF|NCM|GTIP|CAED|"
    r"MARKS?(?:\s+AND\s+NO(?:S|S\.)?)?|FORWARD(?:ING)?|"
    r"DOMESTIC\s+ROUTING\s*/?\s*EXPORT\s+INSTRUCTIONS?|"
    r"EXPORT\s+REF(?:ERENCE)?S?|"
    r"AES|ITN|SHIPPING\s+BILL|S/?BILL|S[./]?B(?:\s*(?:NO|NUMBER))?|"
    r"DUS|ED\s*(?:NO|NUMBER)|DU-E|"
    r"F\s*/\s*AGENT(?:\s+NAME)?\s*&\s*REF|PRN\b|P\.?\s*E\.?)\b"
)
_QUALIFYING_NON_INVOICE_REFERENCE = re.compile(
    r"(?ix)\b(?:forward(?:ing)?|export\s+(?:ref(?:erence)?s?|no|number)|"
    r"domestic\s+routing\s*/?\s*export\s+instructions?|"
    r"ref\s*\.\s*exp\s*\.?|aes|itn|caed(?:\s*(?:no|number))?|"
    r"shipping\s+bill|s/?bill|du-e|dus|f\s*/\s*agent(?:\s+name)?\s*&\s*ref|"
    r"s[./]?b\.?(?:\s*(?:no|number))?|imp\s*/+\s*exp\s*[#]?|"
    r"prn(?:\s*\(\s*proof\s+of\s+report\s+number\s*\))?|p\.?\s*e\.?|"
    r"exp\s*(?=[:#-]?\s*[0-9])|"
    r"ed\s*(?:no|number)\.?\s*(?=[:#-]|\s|$))"
)
_REFERENCE_LABEL_PREFIX = re.compile(
    r"(?ix)^\s*(?:ref\s*\.\s*exp\s*\.?|"
    r"export\s+(?:ref(?:erence)?s?|no|number)|aes|itn|caed(?:\s*(?:no|number))?|"
    r"shipping\s+bill|s/?bill|s[./]?b(?:\s*(?:no|number))?|dus|"
    r"ed\s*(?:no|number)|du-e|imp\s*/+\s*exp|exp|"
    r"f\s*/\s*agent(?:\s+name)?\s*&\s*ref|"
    r"prn(?:\s*\(\s*proof\s+of\s+report\s+number\s*\))?|p\.?\s*e\.?|"
    r"p\s*/?\s*i\s*(?:no|number|ref))"
    r"(?:\s*[:#.-]\s*|\s+)"
)
_REFERENCE_PLACEHOLDER = re.compile(
    r"(?ix)^\s*(?:NO\s+REF(?:ERENCE)?|N/?A|NONE|NIL|NOT\s+APPLICABLE)\.?\s*$"
)
_GENERIC_REFERENCE_LINE = re.compile(r"(?ix)^\s*REF\s*(?:\#|NO\.?|NUMBER)?\s*[:.-]")
_PARTY_SECTION_HEADING = re.compile(
    r"(?im)^\s*(?:(?P<forward>FORWARD(?:ING)?\s+AGENT)\b|"
    r"SHIPPER\b|CONSIGNEE\b|NOTIFY(?:\s+PART(?:Y|IES))?\b|"
    r"DELIVERY\s+AGENT\b|CONSOLIDATOR\b|CARRIER(?:'S)?\s+AGENT\b)"
)
_BOOKING_REFERENCE = re.compile(r"(?ix)\bbooking(?:\s*(?:no|number|ref))?\b")
_PACKING_CONSTRUCTION = re.compile(
    r"(?ix)\b(?:packed\s+(?:with|in|into)|polyethylene\s+(?:inner\s+)?bag|"
    r"inner\s+bag|kraft\s+bag|(?:bag|carton|crate)\s+with\s+[0-9]+\s+layers?)\b"
)
_PER_PACKAGE_NET_MASS = re.compile(
    r"(?ix)\b(?:1|one|per)\b.{0,36}\b(?:bag|drum|package|carton|case|crate|"
    r"pallet|bundle)s?\b.{0,72}\b(?:net\s*(?:wt|weight)|kgs?\s+net)\b"
)
_FACE_FIELD_DESTINATION_PAYABLE = re.compile(
    r"(?im)^[ \t]*FREIGHT(?:[ \t]+AND[ \t]+CHARGES)?[ \t]+PAYABLE[ \t]+AT[ \t]+"
    r"DESTINATION(?:[ \t]*:?[ \t]*(?:\r?\n[ \t]*)?YES)?[ \t]*$"
)
_FACE_FIELD_PREPAID = re.compile(
    r"(?im)^[ \t]*FREIGHT(?:[ \t]+AND[ \t]+CHARGES)?[ \t]+PAYABLE[ \t]+AT"
    r"[ \t]*(?:[:#-][ \t]*)?(?:\r?\n[ \t]*)?PREPAID[ \t]*$"
)
_EXPRESS_RELEASE_NO_ORIGINALS = re.compile(
    r"(?is)\bEXPRESS\s+RELEASE\b.{0,80}\bNO\s+ORIGINALS?\s+ISSUED\b"
)
_HS_HEADING = re.compile(
    r"(?ix)(?:\b(?:H\s*[.-]?\s*S\s*\.?\s*N?\s*\.?|HTS|TARIFF|NCM|GTIP)"
    r"(?:\s*(?:CODE|NO|NUMBER))?\b|"
    r"\bCUSTOMS\s+C(?:O)?DE\b|"
    r"\b(?:BAGS?|BOX(?:ES)?|CARTONS?|CASES?|CRATES?|DRUMS?|PACKAGES?|PALLETS?)"
    r"HS(?=\s*[:#]))"
)
_UN_NUMBER_CONTEXT = re.compile(
    r"(?ix)\bUN(?:DG)?(?:\s*(?:NO|NUMBER)\.?)?\s*[:#.-]?\s*[0-9]"
)
_MADE_IN_CARGO_ORIGIN = re.compile(
    r"(?m)\b(?:MADE IN|Made in)[ \t]+[A-Z][A-Za-z .'-]{1,40}[ \t]*$"
)
_INVOICE_REFERENCE_PREFIX = (
    r"\b(?:(?:INV(?:OICE)?\.?|NVOICE\.?|PROFORMA\s+INVOICES?)\s*"
    r"(?:(?:NO(?:S)?|NUMBER(?:S)?|REF)\.?\s*[:#-]?\s*|[:#-]\s*)?|"
    r"P\s*/?\s*I(?![A-Z])\s*(?:(?:NO|NUMBER|REF)\.?\s*[:#-]?\s*|[:#-]\s*))"
)
_EXPLICIT_INVOICE_REFERENCE = re.compile(
    rf"(?ix){_INVOICE_REFERENCE_PREFIX}"
    r"(?P<value>(?=[A-Z0-9&._/()-]*[0-9])\.?[A-Z0-9][A-Z0-9&._/-]*"
    r"(?:\([A-Z0-9&._/-]+\))?)"
)
_EXPLICIT_INVOICE_HEADING = re.compile(
    r"(?ix)^\s*(?:.*\bAS\s+PER\s+)?"
    r"(?:(?:INV(?:OICE)?\.?|NVOICE\.?|PROFORMA\s+INVOICES?)\s*"
    r"(?:NO(?:S)?|NUMBER(?:S)?|REF)\.?\s*[:#-]?|"
    r"P\s*/?\s*I\s*(?:NO|NUMBER|REF)\.?\s*[:#-]?)\s*$"
)
_ENGLISH_NUMBER_WORD = (
    r"zero|one|two|three|four|five|six|seven|eight|nine|ten|eleven|twelve|"
    r"thirteen|fourteen|fifteen|sixteen|seventeen|eighteen|nineteen|twenty|"
    r"thirty|forty|fifty|sixty|seventy|eighty|ninety|hundred|thousand|million"
)
_PACKAGE_TOTAL_IN_WORDS = re.compile(
    rf"(?ix)\b(?:TOTAL\s+)?NUMBER\s+OF\s+PACKAGES?\s*"
    rf"\(\s*IN\s+WORDS\s*\)\s*:?[ \t]*"
    rf"(?P<value>(?:\b(?:{_ENGLISH_NUMBER_WORD})\b(?:[ \t-]+|[ \t]+AND[ \t]+)){{0,31}}"
    rf"\b(?:{_ENGLISH_NUMBER_WORD})\b)"
)
_UNAMBIGUOUS_COMMA_DECIMAL = re.compile(
    r"(?<![0-9.,])[0-9]{1,3}(?:\.[0-9]{3})+,[0-9]+(?![0-9])"
)
_UNAMBIGUOUS_DOT_DECIMAL = re.compile(
    r"(?<![0-9.,])[0-9]{1,3}(?:,[0-9]{3})+\.[0-9]+(?![0-9])"
)
_COMMA_GROUPED_INTEGER_CONTEXT = re.compile(
    r"(?ix)(?<![0-9.,])[0-9]{1,3}(?:,[0-9]{3})+(?=\s*(?:x(?=[0-9])|(?:pcs?|pkgs?|"
    r"packages?|bags?|cartons?|pallets?|bundles?|drums?|crates?)\b))"
)
_DOT_GROUPED_INTEGER_CONTEXT = re.compile(
    r"(?ix)(?<![0-9.,])[0-9]{1,3}(?:\.[0-9]{3})+(?=\s*(?:x(?=[0-9])|(?:pcs?|pkgs?|"
    r"packages?|bags?|cartons?|pallets?|bundles?|drums?|crates?|bulls?|cattle|heads?)\b))"
)
_MASS_UNIT = r"(?:KGS?|KGM|LBS?|M\s*/?\s*TS?|TON(?:NE)?S?)"
_VOLUME_UNIT = r"(?:CBM|MTQ|M(?:3|³)|CU\.?\s*M\.?|CUBIC\s+MET(?:ER|RE)S?)"
_METRIC_TONNE_SUFFIX = re.compile(
    r"(?ix)^\s*(?:M\s*/?\s*TS?|METRIC\s+TONS?|TONNES?)\b"
)
_COMMA_GROUPED_MASS_CONTEXT = re.compile(
    rf"(?ix)(?<![0-9.,])[0-9]{{1,3}}(?:,[0-9]{{3}})+"
    rf"(?=\s*\(?\s*{_MASS_UNIT}\b)"
)
_DOT_GROUPED_MASS_CONTEXT = re.compile(
    rf"(?ix)(?<![0-9.,])[0-9]{{1,3}}(?:\.[0-9]{{3}})+"
    rf"(?=\s*\(?\s*{_MASS_UNIT}\b)"
)
_COMMA_DECIMAL_VOLUME_CONTEXT = re.compile(
    rf"(?ix)(?<![0-9])[0-9]+,[0-9]+"
    rf"(?=\s*\(?\s*{_VOLUME_UNIT}(?![A-Z]))"
)
_DOT_DECIMAL_VOLUME_CONTEXT = re.compile(
    rf"(?ix)(?<![0-9])[0-9]+\.[0-9]+"
    rf"(?=\s*\(?\s*{_VOLUME_UNIT}(?![A-Z]))"
)
_END_OF_BILL_OF_LADING = re.compile(
    r"(?im)^\s*END\s+OF\s+(?:THE\s+)?BILL\s+OF\s+LADING"
    r"(?:\s+[A-Z0-9._/-]+)?\s*$"
)
_HEADED_COMMA_DECIMAL_VOLUME = re.compile(
    rf"(?ix)(?:\bMEASURE(?:MENT)?\s*)?{_VOLUME_UNIT}\s*[:#-]?\s*"
    r"(?:\r?\n\s*)+(?<![0-9])[0-9]+,[0-9]+(?![0-9])"
)
_HEADED_DOT_DECIMAL_VOLUME = re.compile(
    rf"(?ix)(?:\bMEASURE(?:MENT)?\s*)?{_VOLUME_UNIT}\s*[:#-]?\s*"
    r"(?:\r?\n\s*)+(?<![0-9])[0-9]+\.[0-9]+(?![0-9])"
)
_HEADED_MULTILINE_GROSS_MASS = re.compile(
    r"(?im)^[ \t]*GROSS\s+WEIGHT[ \t]*\r?\n"
    r"(?:[ \t]*(?:CARGO)?[ \t]*\r?\n){0,3}"
    r"[ \t]*(?:KGS?|KGM)[ \t]*\r?\n"
    r"[ \t]*(?P<value>[0-9]+(?:[.,][0-9]+)?)[ \t]*$"
)
_CONTAINER_ROW = re.compile(
    r"(?i)(?<![A-Z0-9])(?P<prefix>[A-Z]{4})[ /-]?"
    r"(?P<serial>[0-9]{6})[ /-]?(?P<check>[0-9])(?![0-9])"
)
_ROW_VOLUME_COUNT = re.compile(r"(?i)\bVOLUMES?\s*[:#-]?\s*(?P<count>[0-9]+)\b")
_PALLET_RANGE = re.compile(
    r"(?im)^[ \t]*PALLETS?\s+(?:NO|NUMBER)\.?\s*:?\s*"
    r"(?P<first>[0-9]+)(?:\s*[-\u2010-\u2015]\s*(?P<last>[0-9]+))?\s*:?\s*$"
)
_PALLET_PACKAGE_TYPE = re.compile(r"(?ix)^\s*(?:PALLETS?(?:\(S\))?|PLTS?)\s*$")
_TOTAL_PACKAGES_LOADED_CONTAINER = re.compile(
    r"(?is)\bTOTAL\s+PACKAGES?\s*:\s*(?P<count>[0-9]+)\b.{0,240}?"
    r"\bLOADED\s+INTO\s+CONTAINER(?:\(S\)|S)?\s*:\s*"
    r"(?P<prefix>[A-Z]{4})[ /-]?(?P<serial>[0-9]{6})[ /-]?"
    r"(?P<check>[0-9])(?![0-9])"
)
_CONTAINER_IDENTIFIER_ADAPTER: TypeAdapter[str] = TypeAdapter(ContainerIdentifier)
_NumericStyle = Literal["comma_decimal", "dot_decimal"]
@dataclass(frozen=True, slots=True)
class _GroundedMatch:
    page_number: int
    start: int
    end: int
    raw_value: str
    excerpt: str
    evidence_kind: str
    normalization_rule: str | None
    score: int


def _fold_with_offsets(value: str) -> tuple[str, tuple[int, ...]]:
    folded: list[str] = []
    offsets: list[int] = []
    for offset, character in enumerate(value):
        normalized = unicodedata.normalize("NFKD", character).casefold()
        for item in normalized:
            if item.isalnum():
                folded.append(item)
                offsets.append(offset)
    return "".join(folded), tuple(offsets)


def _context_excerpt_bounds(source: str, start: int, end: int) -> tuple[int, int]:
    line_starts = [0]
    line_starts.extend(match.end() for match in re.finditer(r"\n", source))
    start_line = max(index for index, value in enumerate(line_starts) if value <= start)
    end_line = max(index for index, value in enumerate(line_starts) if value < max(end, 1))
    excerpt_start = line_starts[max(0, start_line - 2)]
    next_line = end_line + 2
    excerpt_end = line_starts[next_line] - 1 if next_line < len(line_starts) else len(source)
    return excerpt_start, excerpt_end


def _context_excerpt(source: str, start: int, end: int) -> str:
    excerpt_start, excerpt_end = _context_excerpt_bounds(source, start, end)
    excerpt = source[excerpt_start:excerpt_end]
    return excerpt if excerpt.strip() else source[start:end]


def _occurrence_excerpt(source: str, start: int, end: int) -> str:
    """Return local context that resolves to this exact repeated occurrence."""

    raw_value = source[start:end]
    occurrences: list[int] = []
    offset = 0
    while (found := source.find(raw_value, offset)) >= 0:
        occurrences.append(found)
        offset = found + max(1, len(raw_value))
    if len(occurrences) < 2:
        return _context_excerpt(source, start, end)
    occurrence_index = occurrences.index(start)
    excerpt_start, excerpt_end = _context_excerpt_bounds(source, start, end)
    if occurrence_index > 0:
        excerpt_start = max(
            excerpt_start,
            occurrences[occurrence_index - 1] + len(raw_value),
        )
    if occurrence_index + 1 < len(occurrences):
        excerpt_end = min(excerpt_end, occurrences[occurrence_index + 1])
    excerpt = source[excerpt_start:excerpt_end]
    if raw_value not in excerpt:
        raise AssertionError("locally generated OCR excerpt omitted its raw value")
    return excerpt


def _line_bounds(source: str, offset: int) -> tuple[int, int]:
    start = source.rfind("\n", 0, offset) + 1
    end = source.find("\n", offset)
    return start, len(source) if end < 0 else end


_CONTACT_FIELD_HEADING = re.compile(
    r"(?ix)(?:(?P<combined>\bTEL(?:EPHONE)?\s*(?:/|&|\bAND\b)\s*"
    r"(?:FAX|FACSIMILE))|"
    r"(?P<fax>\b(?:FAX|FACSIMILE))|"
    r"(?P<phone>\b(?:TEL(?:EPHONE)?|PHONE|MOBILE)))"
    r"(?:\s*(?:NO|NUMBER))?\s*(?:[:#-]|\s+(?=[+0-9(]))"
)


def _standalone_fax_governs(source: str, start: int, end: int) -> bool:
    line_start, line_end = _line_bounds(source, start)
    line = source[line_start:line_end]
    relative_start = start - line_start
    headings = tuple(_CONTACT_FIELD_HEADING.finditer(line))
    for index, heading in enumerate(headings):
        field_end = headings[index + 1].start() if index + 1 < len(headings) else len(line)
        if (
            heading.group("fax") is not None
            and heading.end() <= relative_start < field_end
            and end - line_start <= field_end
        ):
            return True
    return False


def _line_and_preceding_line(source: str, start: int, end: int) -> str:
    line_start, line_end = _line_bounds(source, start)
    if line_start == 0:
        return source[line_start:line_end]
    previous_end = line_start - 1
    previous_start = source.rfind("\n", 0, previous_end) + 1
    return source[previous_start:line_end]


def _reference_scope(source: str, start: int, end: int) -> str:
    line_start, line_end = _line_bounds(source, start)
    line = source[line_start:line_end]
    if _has_qualifying_reference(line):
        return line
    if _GENERIC_REFERENCE_LINE.search(line):
        preceding_roles = tuple(_PARTY_SECTION_HEADING.finditer(source, 0, start))
        if preceding_roles and preceding_roles[-1].group("forward") is not None:
            # GLM-OCR frequently inserts a visual spacer inside one forwarding-
            # agent box. A generic REF line is eligible only when the nearest
            # preceding party section is the forwarding agent, so a shipper or
            # booking reference cannot leak into the target.
            return f"FORWARDING AGENT\n{line}"
    adjacent = _line_and_preceding_line(source, start, end)
    if _has_qualifying_reference(adjacent):
        return adjacent
    # Values in pro-forma and invoice boxes are commonly printed after one
    # visual spacer. Permit only an explicit, value-bearing invoice heading;
    # generic words such as ``invoice description`` remain ineligible.
    previous_end = line_start - 1
    while previous_end >= 0:
        previous_start = source.rfind("\n", 0, previous_end) + 1
        previous = source[previous_start:previous_end].strip()
        if previous:
            if _EXPLICIT_INVOICE_HEADING.fullmatch(previous) is not None:
                return f"{previous}\n{line}"
            break
        previous_end = previous_start - 1
    # Forwarding-agent reference values are often printed several nonblank
    # address/contact lines below their role heading. Keep the lookup within
    # that exact contiguous block rather than widening to unrelated text.
    block = _contiguous_nonblank_block(source, start, end)
    return block if _has_qualifying_reference(block) else adjacent


def _forbidden_metadata_governs(source: str, start: int, end: int) -> bool:
    """Return whether a forbidden metadata heading governs this exact occurrence.

    GLM-OCR commonly flattens a complete cargo table row onto one physical line.
    A later ACID/tax field must therefore not contaminate valid invoice, VIN, or
    engine values earlier on that line. Conversely, a bare numeric ACID/tax value
    must remain rejected even though the value itself does not repeat its heading.
    """

    line_start, line_end = _line_bounds(source, start)
    line = source[line_start:line_end]
    relative_start = start - line_start
    relative_end = end - line_start
    if any(
        match.start() < relative_end and relative_start < match.end()
        for match in _PACKING_CONSTRUCTION.finditer(line)
    ):
        return False
    headings = [
        *((match.start(), "forbidden") for match in _FORBIDDEN_METADATA.finditer(line)),
        *((match.start(), "allowed") for match in _NON_METADATA_FIELD_HEADING.finditer(line)),
        # GLM-OCR may flatten a complete cargo paragraph onto the same physical
        # line as an earlier ACID/tax field.  An explicit packing-construction
        # phrase starts a new cargo fact and must not inherit that metadata
        # heading merely because the OCR omitted a line break.
        *((match.start(), "allowed") for match in _PACKING_CONSTRUCTION.finditer(line)),
    ]
    preceding = [heading for heading in headings if heading[0] <= relative_start]
    return bool(preceding and max(preceding, key=lambda row: row[0])[1] == "forbidden")


def _explicit_invoice_reference_governs(source: str, start: int, end: int) -> bool:
    line_start, line_end = _line_bounds(source, start)
    line = source[line_start:line_end]
    relative_start = start - line_start
    relative_end = end - line_start
    return any(
        found.start("value") <= relative_start
        and relative_end <= found.end("value")
        for found in _EXPLICIT_INVOICE_REFERENCE.finditer(line)
    )


def _has_qualifying_reference(value: str) -> bool:
    return bool(
        _QUALIFYING_NON_INVOICE_REFERENCE.search(value)
        or _EXPLICIT_INVOICE_REFERENCE.search(value)
        or any(
            _EXPLICIT_INVOICE_HEADING.fullmatch(line) is not None
            for line in value.splitlines()
        )
    )


def _contiguous_nonblank_block(source: str, start: int, end: int) -> str:
    line_start, line_end = _line_bounds(source, start)
    block_start = line_start
    while block_start > 0:
        previous_end = block_start - 1
        previous_start = source.rfind("\n", 0, previous_end) + 1
        if not source[previous_start:previous_end].strip():
            break
        block_start = previous_start
    block_end = line_end
    while block_end < len(source):
        next_start = block_end + 1
        next_end = source.find("\n", next_start)
        if next_end < 0:
            next_end = len(source)
        if not source[next_start:next_end].strip():
            break
        block_end = next_end
    return source[block_start:block_end]


def _path_hints(path: str) -> tuple[str, ...]:
    hints: list[str] = []
    mapping = (
        ("billOfLadingNumber", ("BILL OF LADING", "B/L", "WAYBILL")),
        ("originalBillOfLadingNumber", ("ORIGINAL", "B/L")),
        ("masterBillOfLadingNumber", ("MASTER", "MBL")),
        ("issueDate", ("DATE OF ISSUE", "ISSUE DATE", "PLACE AND DATE")),
        ("shippedOnBoardDate", ("SHIPPED ON BOARD", "ON BOARD DATE")),
        ("placeOfIssue", ("PLACE OF ISSUE", "PLACE AND DATE")),
        ("placeOfReceipt", ("PLACE OF RECEIPT",)),
        ("portOfLoading", ("PORT OF LOADING", "POL")),
        ("transshipmentPort", ("TRANSSHIP",)),
        ("portOfDischarge", ("PORT OF DISCHARGE", "POD")),
        ("placeOfDelivery", ("PLACE OF DELIVERY",)),
        ("finalDestination", ("FINAL DESTINATION",)),
        ("vesselName", ("VESSEL",)),
        ("vesselImoNumber", ("IMO",)),
        ("voyageNumber", ("VOYAGE", "VOY")),
        ("vesselFlagCountry", ("FLAG",)),
        ("paymentArrangement", ("FREIGHT",)),
        ("paymentPlace", ("FREIGHT", "PAYABLE", "PREPAID")),
        ("containerNumber", ("CONTAINER", "CNTR")),
        ("sealNumbers", ("SEAL",)),
        ("verifiedGrossMass", ("VGM", "VERIFIED GROSS")),
        ("temperatureSetpoint", ("SET", "TEMPERATURE", "REEFER")),
        ("grossWeight", ("GROSS",)),
        ("netWeight", ("NET",)),
        ("volume", ("VOLUME", "CBM")),
        ("hsCodes", ("HS", "H.S", "HTS", "TARIFF")),
        ("unNumber", ("UN", "DANGEROUS")),
        ("hazardClass", ("CLASS", "HAZARD")),
        ("packingGroup", ("PACKING GROUP", "PG")),
        ("origin", ("ORIGIN",)),
    )
    for marker, values in mapping:
        if marker in path:
            hints.extend(values)
    role_hints = (
        ("parties.shipper", "SHIPPER"),
        ("parties.consignee", "CONSIGNEE"),
        ("parties.notifyParties", "NOTIFY"),
        ("parties.carrier", "CARRIER"),
        ("parties.forwardingAgent", "FORWARDING"),
        ("parties.deliveryAgent", "DELIVERY AGENT"),
        ("parties.consolidator", "CONSOLIDATOR"),
    )
    for marker, value in role_hints:
        if marker in path:
            hints.append(value)
    return tuple(dict.fromkeys(hints))


def _score_match(path: str, source: str, start: int, end: int, base: int) -> int:
    context = source[max(0, start - 300) : min(len(source), end + 180)].upper()
    line_start = source.rfind("\n", 0, start) + 1
    line_end = source.find("\n", end)
    line = source[line_start : line_end if line_end >= 0 else len(source)].upper()
    score = base
    for hint in _path_hints(path):
        if hint in line:
            score += 24
        elif hint in context:
            score += 6
    if ".phoneNumbers[" in path and _standalone_fax_governs(source, start, end):
        score -= 200
    if "forwardingAndExportReferences" in path:
        reference_scope = _reference_scope(source, start, end)
        if _has_qualifying_reference(reference_scope):
            score += 30
        if _forbidden_metadata_governs(source, start, end):
            score -= 80
        if _BOOKING_REFERENCE.search(reference_scope) and not _has_qualifying_reference(
            reference_scope
        ):
            score -= 40
    return score


def _match(
    *,
    path: str,
    page_number: int,
    source: str,
    start: int,
    end: int,
    evidence_kind: str,
    normalization_rule: str | None,
    base_score: int,
) -> _GroundedMatch:
    return _GroundedMatch(
        page_number=page_number,
        start=start,
        end=end,
        raw_value=source[start:end],
        excerpt=_occurrence_excerpt(source, start, end),
        evidence_kind=evidence_kind,
        normalization_rule=normalization_rule,
        score=_score_match(path, source, start, end, base_score),
    )


def _literal_matches(
    pages: dict[int, str], path: str, target: str
) -> list[_GroundedMatch]:
    matches: list[_GroundedMatch] = []
    for page_number, source in pages.items():
        offset = 0
        while (found := source.find(target, offset)) >= 0:
            matches.append(
                _match(
                    path=path,
                    page_number=page_number,
                    source=source,
                    start=found,
                    end=found + len(target),
                    evidence_kind="verbatim",
                    normalization_rule=None,
                    base_score=120,
                )
            )
            offset = found + max(1, len(target))
    return matches


def _folded_matches(
    pages: dict[int, str], path: str, target: str
) -> list[_GroundedMatch]:
    folded_target, _ = _fold_with_offsets(target)
    if len(folded_target) < 3:
        return []
    kind = (
        "contextual_code"
        if any(
            marker in path
            for marker in (
                "hsCodes",
                "containerNumber",
                "vesselImoNumber",
                "unNumber",
                "typeCode",
            )
        )
        else "normalized"
    )
    rule = (
        "removed printed separators from an explicitly contextualized identifier"
        if kind == "contextual_code"
        else "matched the same OCR characters after case, spacing, and punctuation normalization"
    )
    matches: list[_GroundedMatch] = []
    for page_number, source in pages.items():
        folded_source, offsets = _fold_with_offsets(source)
        offset = 0
        while (found := folded_source.find(folded_target, offset)) >= 0:
            start = offsets[found]
            end = offsets[found + len(folded_target) - 1] + 1
            matches.append(
                _match(
                    path=path,
                    page_number=page_number,
                    source=source,
                    start=start,
                    end=end,
                    evidence_kind=kind,
                    normalization_rule=rule,
                    base_score=90,
                )
            )
            offset = found + max(1, len(folded_target))
    return matches


def _numeric_scope_sources(pages: dict[int, str], path: str) -> tuple[str, ...]:
    marker: re.Pattern[str] | None = None
    if path.endswith(".volume.value"):
        marker = re.compile(rf"(?ix)\bVOLUMES?\b|{_VOLUME_UNIT}")
    elif path.endswith(".verifiedGrossMass.value"):
        marker = re.compile(r"(?ix)\bVGM\b|\bVERIFIED\s+GROSS\b")
    elif path.endswith(".grossWeight.value"):
        marker = re.compile(r"(?ix)\bGROSS\b")
    elif path.endswith(".netWeight.value"):
        marker = re.compile(r"(?ix)\bNET\b")
    elif path.endswith(".temperatureSetpoint.value"):
        marker = re.compile(r"(?ix)\b(?:SET|TEMPERATURE|REEFER)\b")
    elif path.endswith((".quantity", ".packageQuantity")):
        marker = re.compile(
            r"(?ix)\b(?:PCS?|CS|PK|PKGS?|PACKAGES?|CARTONS?|CTNS?|BAGS?|"
            r"PIECES?|PLTS?|PALLETS?|BUNDLES?|DRUMS?|CRATES?|CASES?|"
            r"BULLS?|CATTLE|HEADS?)\b"
        )
    if marker is None:
        return tuple(pages.values())
    relevant = tuple(
        line
        for source in pages.values()
        for line in source.splitlines()
        if marker.search(line)
    )
    return relevant or tuple(pages.values())


def _numeric_style(pages: dict[int, str], path: str) -> _NumericStyle | None:
    sources = _numeric_scope_sources(pages, path)
    mass_path = path.endswith(
        (".grossWeight.value", ".netWeight.value", ".verifiedGrossMass.value")
    )
    if path.endswith(".volume.value"):
        headed_comma_decimal = any(
            _HEADED_COMMA_DECIMAL_VOLUME.search(source) for source in pages.values()
        )
        headed_dot_decimal = any(
            _HEADED_DOT_DECIMAL_VOLUME.search(source) for source in pages.values()
        )
        if headed_comma_decimal != headed_dot_decimal:
            return "comma_decimal" if headed_comma_decimal else "dot_decimal"
        comma_volume_decimal = any(
            _COMMA_DECIMAL_VOLUME_CONTEXT.search(source) for source in sources
        )
        dot_volume_decimal = any(
            _DOT_DECIMAL_VOLUME_CONTEXT.search(source) for source in sources
        )
        if comma_volume_decimal != dot_volume_decimal:
            return "comma_decimal" if comma_volume_decimal else "dot_decimal"
        # Flattened table rows often carry the unit only in the header, e.g.
        # ``... Gross weight Kg M3`` followed by ``... 20,79 0,414``.  A
        # zero-prefixed final cell is unambiguously decimal, so use it without
        # guessing how a three-digit nonzero separator should be interpreted.
        tabular_styles: set[_NumericStyle] = set()
        for source in pages.values():
            lines = source.splitlines()
            for index, line in enumerate(lines[:-1]):
                if re.search(_VOLUME_UNIT, line, flags=re.IGNORECASE | re.VERBOSE) is None:
                    continue
                for following in lines[index + 1 : index + 4]:
                    if not following.strip():
                        continue
                    numbers = tuple(_NUMBER.finditer(following))
                    if numbers:
                        final_number = numbers[-1].group().lstrip("+-")
                        if final_number.startswith("0,"):
                            tabular_styles.add("comma_decimal")
                        elif final_number.startswith("0."):
                            tabular_styles.add("dot_decimal")
                    break
        if len(tabular_styles) == 1:
            return next(iter(tabular_styles))

    comma_decimal = any(_UNAMBIGUOUS_COMMA_DECIMAL.search(source) for source in sources)
    dot_decimal = any(_UNAMBIGUOUS_DOT_DECIMAL.search(source) for source in sources)
    if comma_decimal == dot_decimal:
        if comma_decimal:
            return None
        comma_grouped_pattern = (
            _COMMA_GROUPED_MASS_CONTEXT if mass_path else _COMMA_GROUPED_INTEGER_CONTEXT
        )
        dot_grouped_pattern = (
            _DOT_GROUPED_MASS_CONTEXT if mass_path else _DOT_GROUPED_INTEGER_CONTEXT
        )
        # OCR flattens table headers and their values onto separate physical
        # lines. For mass fields, the unit-qualified number is stronger style
        # evidence than proximity to the GROSS/NET header, so inspect the whole
        # page with the mass-specific regex. Other targets retain their scope.
        grouped_sources = tuple(pages.values()) if mass_path else sources
        comma_grouped = any(
            comma_grouped_pattern.search(source) for source in grouped_sources
        )
        dot_grouped = any(dot_grouped_pattern.search(source) for source in grouped_sources)
        if comma_grouped == dot_grouped:
            return None
        return "dot_decimal" if comma_grouped else "comma_decimal"
    return "comma_decimal" if comma_decimal else "dot_decimal"


def _decimal(
    value: str,
    *,
    style: _NumericStyle | None,
    integer_target: bool,
) -> Decimal | None:
    normalized = value.replace(" ", "")
    sign = ""
    if normalized[:1] in {"+", "-"}:
        sign, normalized = normalized[0], normalized[1:]
    comma_count = normalized.count(",")
    dot_count = normalized.count(".")
    if comma_count and dot_count:
        if normalized.rfind(",") > normalized.rfind("."):
            normalized = normalized.replace(".", "").replace(",", ".")
        else:
            normalized = normalized.replace(",", "")
    elif comma_count:
        groups = normalized.split(",")
        if comma_count > 1 and all(len(group) == 3 for group in groups[1:]):
            normalized = "".join(groups)
        elif comma_count == 1:
            fractional_digits = len(groups[1])
            grouping_shape = len(groups[0]) <= 3 and fractional_digits == 3
            if integer_target and grouping_shape:
                normalized = "".join(groups)
            elif style == "comma_decimal" or not grouping_shape:
                normalized = ".".join(groups)
            elif style == "dot_decimal" and grouping_shape:
                normalized = "".join(groups)
            else:
                return None
        else:
            return None
    elif dot_count:
        groups = normalized.split(".")
        if dot_count > 1 and all(len(group) == 3 for group in groups[1:]):
            normalized = "".join(groups)
        elif dot_count == 1:
            fractional_digits = len(groups[1])
            grouping_shape = len(groups[0]) <= 3 and fractional_digits == 3
            if (integer_target and grouping_shape) or (style == "comma_decimal" and grouping_shape):
                normalized = "".join(groups)
            elif style in {None, "dot_decimal"} or not grouping_shape:
                normalized = ".".join(groups)
            else:
                return None
    try:
        return Decimal(sign + normalized)
    except InvalidOperation:
        return None


def _numeric_matches(
    pages: dict[int, str], path: str, target: int | float
) -> list[_GroundedMatch]:
    expected = Decimal(str(target))
    style = _numeric_style(pages, path)
    matches: list[_GroundedMatch] = []
    for page_number, source in pages.items():
        if path.endswith(".grossWeight.value"):
            for headed in _HEADED_MULTILINE_GROSS_MASS.finditer(source):
                raw_number = headed.group("value")
                try:
                    parsed = Decimal(raw_number.replace(",", "."))
                except InvalidOperation:
                    continue
                if parsed == expected:
                    matches.append(
                        _match(
                            path=path,
                            page_number=page_number,
                            source=source,
                            start=headed.start("value"),
                            end=headed.end("value"),
                            evidence_kind="normalized",
                            normalization_rule=(
                                "parsed the scalar in the explicit multiline GROSS WEIGHT "
                                "KGS field"
                            ),
                            base_score=150,
                        )
                    )
        found_numbers = list(_NUMBER.finditer(source))
        if path.endswith((".quantity", ".packageQuantity")):
            existing_spans = {(found.start(), found.end()) for found in found_numbers}
            found_numbers.extend(
                found
                for found in _COUNT_NUMBER.finditer(source)
                if (found.start(), found.end()) not in existing_spans
            )
        if path.endswith(
            (".flashPoint.temperature.value", ".temperatureSetpoint.value")
        ):
            existing_spans = {(found.start(), found.end()) for found in found_numbers}
            found_numbers.extend(
                found
                for found in _GLUED_CELSIUS_TEMPERATURE.finditer(source)
                if (found.start(), found.end()) not in existing_spans
            )
        for found in found_numbers:
            raw_number = found.groupdict().get("value") or found.group()
            occurrence_style = style
            if (
                path.endswith(
                    (".grossWeight.value", ".netWeight.value", ".verifiedGrossMass.value")
                )
                and _METRIC_TONNE_SUFFIX.match(source[found.end() :])
                and not ("." in raw_number and "," in raw_number)
            ):
                if "." in raw_number:
                    occurrence_style = "dot_decimal"
                elif "," in raw_number:
                    occurrence_style = "comma_decimal"
            if _decimal(
                raw_number,
                style=occurrence_style,
                integer_target=isinstance(target, int),
            ) != expected:
                continue
            matches.append(
                _match(
                    path=path,
                    page_number=page_number,
                    source=source,
                    start=found.start(),
                    end=found.end(),
                    evidence_kind="normalized",
                    normalization_rule="parsed the exact printed numeric value",
                    base_score=80,
                )
            )
    return matches


_SMALL_ENGLISH_NUMBERS = {
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
    "ten": 10,
    "eleven": 11,
    "twelve": 12,
    "thirteen": 13,
    "fourteen": 14,
    "fifteen": 15,
    "sixteen": 16,
    "seventeen": 17,
    "eighteen": 18,
    "nineteen": 19,
    "twenty": 20,
    "thirty": 30,
    "forty": 40,
    "fifty": 50,
    "sixty": 60,
    "seventy": 70,
    "eighty": 80,
    "ninety": 90,
}


def _english_cardinal(value: str) -> int | None:
    """Parse a conventional English cardinal without accepting arbitrary prose."""

    tokens = tuple(
        token
        for token in re.split(r"[\s-]+", value.casefold().strip())
        if token and token != "and"
    )
    if not tokens:
        return None
    total = 0
    section = 0
    previous_scale = 1_000_000_001
    for token in tokens:
        if token in _SMALL_ENGLISH_NUMBERS:
            number = _SMALL_ENGLISH_NUMBERS[token]
            if section % 100 >= 20 and number >= 10:
                return None
            section += number
        elif token == "hundred":
            if not 1 <= section <= 9:
                return None
            section *= 100
        elif token in {"thousand", "million"}:
            scale = 1_000 if token == "thousand" else 1_000_000
            if section == 0 or scale >= previous_scale:
                return None
            total += section * scale
            section = 0
            previous_scale = scale
        else:
            return None
    return total + section


def _word_quantity_matches(
    pages: dict[int, str], path: str, target: int
) -> list[_GroundedMatch]:
    if not path.endswith((".quantity", ".packageQuantity")):
        return []
    matches: list[_GroundedMatch] = []
    for page_number, source in pages.items():
        for found in _PACKAGE_TOTAL_IN_WORDS.finditer(source):
            if _english_cardinal(found.group("value")) != target:
                continue
            matches.append(
                _match(
                    path=path,
                    page_number=page_number,
                    source=source,
                    start=found.start("value"),
                    end=found.end("value"),
                    evidence_kind="normalized",
                    normalization_rule=(
                        "parsed the explicit English package-total words as an integer"
                    ),
                    base_score=90,
                )
            )
    return matches


_DATE_FORMATS = (
    "%Y-%m-%d",
    "%Y/%m/%d",
    "%Y.%m.%d",
    "%d-%m-%Y",
    "%d/%m/%Y",
    "%d.%m.%Y",
    "%m-%d-%Y",
    "%m/%d/%Y",
    "%m.%d.%Y",
    "%Y %b %d",
    "%Y %B %d",
    "%d-%b-%Y",
    "%d-%B-%Y",
    "%d %b %Y",
    "%d %B %Y",
    "%d %b, %Y",
    "%d %B, %Y",
    "%b %d %Y",
    "%B %d %Y",
    "%b %d, %Y",
    "%B %d, %Y",
    "%d-%b-%y",
    "%d %b %y",
    "%b %d %y",
    "%d/%b/%y",
    "%d/%b/%Y",
    "%b/%d/%y",
    "%b/%d/%Y",
    "%d/%m/%y",
    "%m/%d/%y",
    "%d.%m.%y",
    "%m.%d.%y",
)


def _parsed_dates(raw: str) -> set[date]:
    normalized = raw.strip()
    normalized = re.sub(r"(?i)(?<=\d)(?:st|nd|rd|th)\b", "", normalized)
    # OCR can insert a space between the two digits of a day (for example,
    # ``3 0 JAN 2024``).  Collapse only the day position immediately before a
    # named month so ordinary spaced numbers elsewhere remain untouched.
    normalized = re.sub(
        rf"(?ix)(?<![0-9])([0-3])\s+([0-9])"
        rf"(?=\s*(?:[-/. ]\s*)?(?:{_MONTH_NAME})\b)",
        r"\1\2",
        normalized,
    )
    if re.search(rf"(?ix)\b(?:{_MONTH_NAME})\b", normalized):
        normalized = re.sub(r"[/.,-]+", " ", normalized)
    normalized = re.sub(r"\s+", " ", normalized)
    normalized = re.sub(r"\s*,\s*", ", ", normalized)
    values: set[date] = set()
    for fmt in _DATE_FORMATS:
        try:
            values.add(datetime.strptime(normalized, fmt).date())
        except ValueError:
            continue
    return values


def _day_first_numeric_date(raw: str) -> date | None:
    found = _NUMERIC_DATE_TOKEN.fullmatch(raw.strip())
    if found is None:
        return None
    first = int(found.group("first"))
    second = int(found.group("second"))
    year = int(found.group("year"))
    if year < 100:
        year += 2000 if year <= 68 else 1900
    day, month = first, second
    try:
        return date(year, month, day)
    except ValueError:
        return None


def _date_matches(
    pages: dict[int, str], path: str, target: str
) -> list[_GroundedMatch]:
    expected = date.fromisoformat(target)
    matches: list[_GroundedMatch] = []
    for page_number, source in pages.items():
        for found in _DATE_TOKEN.finditer(source):
            parsed = _parsed_dates(found.group())
            if expected not in parsed:
                continue
            if (
                len(parsed) > 1
                and _day_first_numeric_date(found.group()) != expected
            ):
                continue
            matches.append(
                _match(
                    path=path,
                    page_number=page_number,
                    source=source,
                    start=found.start(),
                    end=found.end(),
                    evidence_kind="normalized",
                    normalization_rule="normalized the printed headed date to ISO 8601",
                    base_score=85,
                )
            )
    return matches


_SEMANTIC_PATTERNS: dict[str, tuple[re.Pattern[str], ...]] = {
    "kilogram": (
        re.compile(r"(?i)(?<![A-Z])(?:KGS?|KGM)\.?\b"),
        re.compile(r"(?i)\bKILOGRAMS?\b"),
        re.compile(r"(?i)\bKILOS?\b"),
    ),
    "pound": (re.compile(r"(?i)LBS?\.?\b"), re.compile(r"(?i)\bPOUNDS?\b")),
    "metric_tonne": (
        re.compile(
            r"(?i)(?<![A-Z])M(?:ETRIC)?\s*\.?/?\s*T(?:S|ON(?:NE)?S?)?\.?(?![A-Z])"
        ),
        re.compile(r"(?i)\bTONNES?\b"),
    ),
    "cubic_metre": (
        re.compile(r"(?i)CBM\b"),
        re.compile(r"(?i)MTQ\b"),
        re.compile(r"(?i)(?<![A-Z])M(?:3|³)(?![A-Z])"),
        re.compile(r"(?i)(?<![A-Z])CU\.?\s*M\.?(?![A-Z])"),
        re.compile(r"(?i)\bCUBIC\s+MET(?:ER|RE)S?\b"),
    ),
    "celsius": (re.compile(r"(?i)(?:°\s*)?C\b"), re.compile(r"(?i)\bCELSIUS\b")),
    "fahrenheit": (re.compile(r"(?i)(?:°\s*)?F\b"), re.compile(r"(?i)\bFAHRENHEIT\b")),
    "non_negotiable": (
        re.compile(r"(?i)\bNON[- ]?NEGOTIABLE(?:\s+WAYBILL)?\b"),
        re.compile(r"(?i)\bNOT\s+NEGOTIABLE\b"),
        re.compile(r"(?i)\bSEA\s+WAYBILL\b"),
        re.compile(r"(?i)\bEXPRESS\s+(?:BILL\s+OF\s+LADING|B/?L)\b"),
        re.compile(r"(?i)\bTELEX\s+RELEASE\b"),
        re.compile(
            r"(?is)\b(?:NO\.?|NUMBER)\s+OF\s+ORIGINAL(?:\s+B\(S\)/L)?\b"
            r".{0,32}\b0\s*/?\s*ORIGINAL\b"
        ),
        re.compile(
            r"(?is)\b(?:NO\.?|NUMBER)\s+OF\s+ORIGINAL(?:\s+BL'?S?|\s+B\(S\)/L)?\b"
            r".{0,48}\b0+\s*/\s*ZERO(?:E)?S\b"
        ),
        _EXPRESS_RELEASE_NO_ORIGINALS,
    ),
    "negotiable": (
        re.compile(r"(?i)(?<!NON[- ])\bTO\s+(?:THE\s+)?ORDER(?:\s+OF)?\b"),
        re.compile(r"(?i)\bDELIVER(?:ED|Y)\s+UNTO\s+ORDER\s+OR\s+ASSIGNS\b"),
        re.compile(r"(?i)\b(?:must|shall)\s+be\s+surrendered\b.{0,48}\bduly\s+endorsed\b"),
        re.compile(
            r"(?i)\bORIGINAL\s+BILLS?\s+OF\s+LADING\b.{0,120}"
            r"\bMUST\s+BE\s+SURRENDERED\b.{0,48}\b(?:DULY\s+)?ENDORSED\b"
        ),
        re.compile(
            r"(?i)\bPRESENTATION\s+OF\s+(?:THIS\s+DOCUMENT|ONE\s+ORIGINAL\s+OF\s+"
            r"THIS\s+BILL\s+OF\s+LADING)\b.{0,96}\bDULY\s+ENDORSED\b"
        ),
        re.compile(
            r"(?i)\bONE\s+ORIGINAL\s+OF\s+THIS\s+BILL\s+OF\s+LADING\b.{0,48}"
            r"\bDULY\s+ENDORSED\b.{0,96}\bPRESENTATION\b"
        ),
        re.compile(
            r"(?i)\bONE\s+ORIGINAL\s+BILL\s+OF\s+LADING\b.{0,96}"
            r"\bDULY\s+ENDORSED\b.{0,48}\bMUST\s+BE\s+SURRENDERED\b"
        ),
        re.compile(
            r"(?i)\bSURRENDER\s+ONE\s+ORIGINAL\s+BILL\s+OF\s+LADING\b.{0,64}"
            r"\bDULY\s+ENDORSED\b"
        ),
        re.compile(
            r"(?i)\bONE\s+OF\s+(?:THIS|THE)\s+BILL\s+OF\s+LADING\b.{0,64}"
            r"\bDULY\s+ENDORSED\b.{0,48}\bMUST\s+BE\s+SURRENDERED\b"
        ),
        re.compile(
            r"(?i)\bONE\b.{0,64}\bACCOMPLISHED\b.{0,64}"
            r"\bOTHERS?(?:\s*\(S\))?(?=\s|$).{0,32}\b(?:STAND|BE)\s+VOID\b"
        ),
    ),
    "prepaid": (re.compile(r"(?i)\bFREIGHT\s+PREPAID\b"), re.compile(r"(?i)\bPREPAID\b")),
    "collect": (
        re.compile(r"(?i)\bFREIGHT\s+COLLECT\b"),
        re.compile(r"(?i)\bCOLLECT\b"),
        re.compile(
            r"(?i)\bFREIGHT(?:\s+AND\s+CHARGES)?\s+PAYABLE\s+AT\s+"
            r"DESTINATION(?:\s*:?\s*YES\b|\b)"
        ),
    ),
    "third_party": (re.compile(r"(?i)\bTHIRD[- ]PARTY\b"),),
    "payable_elsewhere": (re.compile(r"(?i)\bPAYABLE\s+ELSEWHERE\b"),),
    "consignee": (re.compile(r"(?i)\bSAME\s+AS\s+CONSIGNEE\b"),),
    "shipper": (re.compile(r"(?i)\bSAME\s+AS\s+SHIPPER\b"),),
}


def _semantic_matches(
    pages: dict[int, str], path: str, target: str
) -> list[_GroundedMatch]:
    patterns = _SEMANTIC_PATTERNS.get(target, ())
    if path.endswith("hazardClass") or path.endswith("subsidiaryHazard"):
        patterns = (
            re.compile(rf"(?i)\b(?:IMO\s+)?(?:HAZARD\s+)?CLASS\s*[:.]?\s*{re.escape(target)}\b"),
        )
    elif path.endswith("packingGroup"):
        patterns = (
            re.compile(rf"(?i)\b(?:PACKING\s+GROUP|P\.?G\.?)\s*[:.]?\s*{re.escape(target)}\b"),
        )
    matches: list[_GroundedMatch] = []
    if path.endswith("negotiability") and target == "non_negotiable":
        for page_number, source in pages.items():
            for start, end in named_non_order_consignee_spans(source):
                matches.append(
                    _match(
                        path=path,
                        page_number=page_number,
                        source=source,
                        start=start,
                        end=end,
                        evidence_kind="normalized",
                        normalization_rule=(
                            "mapped the completed named non-order consignee to a straight, "
                            "non-negotiable bill"
                        ),
                        base_score=110,
                    )
                )
    for page_number, source in pages.items():
        for pattern in patterns:
            for found in pattern.finditer(source):
                if ".flashPoint." in path and not has_flash_point_context(
                    _contiguous_nonblank_block(source, found.start(), found.end())
                ):
                    continue
                matches.append(
                    _match(
                        path=path,
                        page_number=page_number,
                        source=source,
                        start=found.start(),
                        end=found.end(),
                        evidence_kind=(
                            "contextual_code"
                            if path.endswith(("sameAs", "hazardClass", "subsidiaryHazard"))
                            else "normalized"
                        ),
                        normalization_rule=(
                            "mapped explicit OCR wording to the semantic target value"
                        ),
                        base_score=95,
                    )
                )
    return matches


def _anchor_match_key(
    pages: dict[int, str], match: _GroundedMatch, anchor: str
) -> tuple[int, int, int]:
    anchors = _folded_matches(pages, "documentPatch.containers[0].containerNumber", anchor)
    same_page = tuple(row for row in anchors if row.page_number == match.page_number)
    if not same_page:
        return (0, 0, -len(pages[match.page_number]))
    source = pages[match.page_number]
    match_line = _line_bounds(source, match.start)
    same_line = any(_line_bounds(source, row.start) == match_line for row in same_page)
    distance = min(
        max(row.start - match.end, match.start - row.end, 0) for row in same_page
    )
    return (int(same_line), 1, -distance)


def _best_match(
    pages: dict[int, str], path: str, target: Any, *, anchor: str | None = None
) -> _GroundedMatch:
    matches: list[_GroundedMatch] = []
    if isinstance(target, bool):
        raise DeterministicAnnotationError(f"unsupported Boolean target leaf: {path}")
    if isinstance(target, (int, float)):
        numeric_matches = _numeric_matches(pages, path, target)
        matches.extend(numeric_matches)
        if isinstance(target, int) and not numeric_matches:
            matches.extend(_word_quantity_matches(pages, path, target))
    elif isinstance(target, str):
        if path.endswith(("issueDate", "shippedOnBoardDate")):
            matches.extend(_date_matches(pages, path, target))
        matches.extend(_semantic_matches(pages, path, target))
        matches.extend(_literal_matches(pages, path, target))
        matches.extend(_folded_matches(pages, path, target))
    else:
        raise DeterministicAnnotationError(
            f"unsupported target leaf type at {path}: {type(target).__name__}"
        )
    if not matches:
        if isinstance(target, str) and path.endswith(("issueDate", "shippedOnBoardDate")):
            raise DeterministicAnnotationError(
                "date target is not exactly groundable under the document-internal, "
                f"day-first-on-ambiguity policy: {path}={target!r}"
            )
        raise DeterministicAnnotationError(
            f"target leaf is not exactly groundable in raw OCR: {path}={target!r}"
        )
    valid_matches: list[_GroundedMatch] = []
    policy_errors: list[DeterministicAnnotationError] = []
    for match in matches:
        try:
            _validate_semantic_policy(pages, path, target, match)
        except DeterministicAnnotationError as error:
            policy_errors.append(error)
        else:
            valid_matches.append(match)
    if not valid_matches:
        # Do not let a high-scoring but prohibited occurrence hide a second,
        # clean occurrence of the same printed value. If every occurrence is
        # prohibited, retain the original fail-closed diagnostic.
        raise policy_errors[0]
    return max(
        valid_matches,
        key=lambda row: (
            *(_anchor_match_key(pages, row, anchor) if anchor is not None else ()),
            row.score,
            -row.page_number,
            -row.start,
        ),
    )


def _explicit_footnote_linked_gap(
    source: str, previous_end: int, current_start: int
) -> bool:
    """Return whether two same-page text fragments share an explicit footnote marker."""

    previous_line_end = source.find("\n", previous_end)
    if previous_line_end < 0:
        previous_line_end = len(source)
    current_line_start = source.rfind("\n", 0, current_start) + 1
    trailing = source[previous_end:previous_line_end].strip()
    leading = source[current_line_start:current_start].strip()
    trailing_marker = re.fullmatch(r"(?P<marker>[*\N{DAGGER}\N{DOUBLE DAGGER}]+)", trailing)
    leading_marker = re.fullmatch(r"(?P<marker>[*\N{DAGGER}\N{DOUBLE DAGGER}]+)", leading)
    return bool(
        trailing_marker is not None
        and leading_marker is not None
        and trailing_marker.group("marker") == leading_marker.group("marker")
    )


def _pdf_grouped_straight_consignee_match(
    pages: dict[int, str], patch: dict[str, Any], path: str, target: Any
) -> _GroundedMatch | None:
    """Ground straight negotiability in an OCR name grouped by the reviewed PDF.

    The PDF may resolve which visually adjacent OCR value belongs to the
    consignee field, but it may not supply or correct the value. The emitted
    consignee name must therefore still occur exactly in raw OCR.
    """

    if not (path.endswith(".negotiability") and target == "non_negotiable"):
        return None
    parties = patch.get("parties")
    consignee = parties.get("consignee") if isinstance(parties, dict) else None
    name = consignee.get("name") if isinstance(consignee, dict) else None
    if not isinstance(name, str) or re.search(
        r"(?ix)\b(?:to\s+(?:the\s+)?order(?:\s+of)?|order\s+of)\b", name
    ):
        return None
    candidates = [
        *_literal_matches(pages, "documentPatch.parties.consignee.name", name),
        *_folded_matches(pages, "documentPatch.parties.consignee.name", name),
    ]
    if not candidates:
        return None
    chosen = max(candidates, key=lambda row: (row.score, -row.page_number, -row.start))
    return _match(
        path=path,
        page_number=chosen.page_number,
        source=pages[chosen.page_number],
        start=chosen.start,
        end=chosen.end,
        evidence_kind="normalized",
        normalization_rule=(
            "mapped the exact OCR-printed, PDF-grouped named consignee to a straight, "
            "non-negotiable bill"
        ),
        base_score=110,
    )


def _composite_text_matches(
    pages: dict[int, str], path: str, target: str
) -> tuple[_GroundedMatch, ...]:
    """Ground source-ordered text fragments around separately modeled facts."""

    target_tokens = tuple(re.findall(r"[\w]+", target, flags=re.UNICODE))
    folded_tokens = tuple(_fold_with_offsets(token)[0] for token in target_tokens)
    if len(folded_tokens) < 2 or any(not token for token in folded_tokens):
        return ()
    global_characters: list[str] = []
    global_locations: list[tuple[int, int] | None] = []
    for page_number in sorted(pages):
        folded_source, offsets = _fold_with_offsets(pages[page_number])
        global_characters.extend(folded_source)
        global_locations.extend((page_number, offset) for offset in offsets)
        global_characters.append("\0")
        global_locations.append(None)
    global_source = "".join(global_characters)
    folded_target_length = sum(len(token) for token in folded_tokens)
    max_in_page_gap = max(600, folded_target_length * 6)
    candidates: list[
        tuple[int, int, int, int, tuple[tuple[int, int], ...]]
    ] = []
    first_offset = 0
    while (first := global_source.find(folded_tokens[0], first_offset)) >= 0:
        positions: list[tuple[int, int]] = [(first, first + len(folded_tokens[0]))]
        cursor = positions[-1][1]
        valid = True
        for token in folded_tokens[1:]:
            found = global_source.find(token, cursor)
            if found < 0:
                valid = False
                break
            positions.append((found, found + len(token)))
            cursor = positions[-1][1]
        first_location = global_locations[first]
        last_location = global_locations[positions[-1][1] - 1] if valid else None
        token_locations = (
            [global_locations[position[0]] for position in positions] if valid else []
        )
        in_page_gap = 0
        if valid:
            for previous, current in pairwise(positions):
                previous_location = global_locations[previous[1] - 1]
                current_location = global_locations[current[0]]
                if (
                    previous_location is not None
                    and current_location is not None
                    and previous_location[0] == current_location[0]
                ):
                    previous_raw_end = previous_location[1] + 1
                    current_raw_start = current_location[1]
                    if not (
                        path.endswith(".address")
                        and _explicit_footnote_linked_gap(
                            pages[previous_location[0]],
                            previous_raw_end,
                            current_raw_start,
                        )
                    ):
                        in_page_gap += current[0] - previous[1]
        if (
            valid
            and first_location is not None
            and last_location is not None
            and all(location is not None for location in token_locations)
            and last_location[0] - first_location[0] <= 3
            and in_page_gap <= max_in_page_gap
        ):
            first_page, first_raw_start = first_location
            first_source = pages[first_page]
            score = _score_match(
                path,
                first_source,
                first_raw_start,
                min(len(first_source), first_raw_start + len(target)),
                65,
            )
            page_span = last_location[0] - first_location[0]
            candidates.append(
                (score, -in_page_gap, -page_span, -first, tuple(positions))
            )
        first_offset = first + max(1, len(folded_tokens[0]))
    if not candidates:
        return ()
    *_, chosen_positions = max(candidates)
    spans: list[tuple[int, int, int]] = []
    for folded_start, folded_end in chosen_positions:
        start_location = global_locations[folded_start]
        end_location = global_locations[folded_end - 1]
        if start_location is None or end_location is None:
            raise AssertionError("folded target token crossed a page delimiter")
        page_number, start = start_location
        end_page, raw_end = end_location
        if page_number != end_page:
            raise AssertionError("folded target token crossed a page delimiter")
        end = raw_end + 1
        source = pages[page_number]
        if (
            spans
            and spans[-1][0] == page_number
            and not any(
                character.isalnum()
                for character in source[spans[-1][2] : start]
            )
        ):
            spans[-1] = (page_number, spans[-1][1], end)
        else:
            spans.append((page_number, start, end))
    return tuple(
        _GroundedMatch(
            page_number=page_number,
            start=start,
            end=end,
            raw_value=pages[page_number][start:end],
            excerpt=pages[page_number][start:end],
            evidence_kind="normalized",
            normalization_rule=(
                "joined source-ordered OCR fragments while excluding intervening separately "
                "modeled values"
            ),
            score=65,
        )
        for page_number, start, end in spans
    )


def _leaf_items(value: Any, prefix: str) -> list[tuple[str, Any]]:
    if isinstance(value, dict):
        rows: list[tuple[str, Any]] = []
        for key, child in value.items():
            rows.extend(_leaf_items(child, f"{prefix}.{key}"))
        return rows
    if isinstance(value, list):
        rows = []
        for index, child in enumerate(value):
            rows.extend(_leaf_items(child, f"{prefix}[{index}]"))
        return rows
    return [(prefix, value)]


def _explicit_invoice_references(pages: dict[int, str]) -> tuple[str, ...]:
    values: list[str] = []
    seen: set[str] = set()
    for page_number in sorted(pages):
        for match in _EXPLICIT_INVOICE_REFERENCE.finditer(pages[page_number]):
            value = match.group("value")
            trailing = pages[page_number][match.end() :]
            if value.upper().endswith("FREIGHT") and re.match(
                r"(?i)^\s+(?:COLLECT|PREPAID)\b", trailing
            ):
                # Flattened OCR can glue the next freight field to an invoice
                # value without a separator (``...-25FREIGHT COLLECT``).  The
                # headed invoice value ends before that independently modelled
                # field; retain the exact source substring and trim only the
                # unambiguous heading suffix.
                value = value[: -len("FREIGHT")]
            if not any(character.isdigit() for character in value):
                continue
            key, _ = _fold_with_offsets(value)
            if key and key not in seen:
                seen.add(key)
                values.append(value)
    return tuple(values)


def _hs_codes_from_value_line(value: str) -> tuple[str, ...]:
    candidate = value.strip(" \t:;,-|/")
    if not candidate or re.search(r"[A-Za-z]", candidate):
        return ()
    tokens = re.findall(r"(?<![0-9])[0-9]+(?:[.-][0-9]+)*(?![0-9])", candidate)
    if not tokens:
        return ()
    digit_tokens = tuple(re.sub(r"[^0-9]", "", token) for token in tokens)
    individually_valid = tuple(
        token for token in digit_tokens if 6 <= len(token) <= 18
    )
    if len(individually_valid) == len(digit_tokens):
        return individually_valid
    combined = "".join(digit_tokens)
    if (
        len(digit_tokens) > 1
        and all(1 <= len(token) <= 4 for token in digit_tokens)
        and 6 <= len(combined) <= 18
    ):
        return (combined,)
    return individually_valid


def _explicit_hs_codes(pages: dict[int, str]) -> tuple[str, ...]:
    values: list[str] = []
    seen: set[str] = set()
    for page_number in sorted(pages):
        lines = pages[page_number].splitlines()
        for index, line in enumerate(lines):
            heading = _HS_HEADING.search(line)
            if heading is None:
                continue
            value_lines = [line[heading.end() :]]
            next_index = index + 1
            while next_index < len(lines) and lines[next_index].strip():
                next_row_values = _hs_codes_from_value_line(lines[next_index])
                suffix_only = re.fullmatch(
                    r"\s*[0-9]{1,2}\s*,\s*", lines[next_index]
                )
                if not next_row_values and suffix_only is None:
                    break
                value_lines.append(lines[next_index])
                next_index += 1
            candidates: list[str] = []
            for value_index, value_line in enumerate(value_lines):
                row_codes = list(_hs_codes_from_value_line(value_line))
                if value_index + 1 < len(value_lines):
                    trailing = re.search(r"(?<![0-9])(?P<prefix>[0-9]{6,17})\s*$", value_line)
                    continuation = re.match(
                        r"^\s*(?P<suffix>[0-9]{1,2})\s*,\s*(?P<remainder>.*)$",
                        value_lines[value_index + 1],
                    )
                    if trailing is not None and continuation is not None:
                        prefix = trailing.group("prefix")
                        combined = prefix + continuation.group("suffix")
                        if prefix in row_codes and len(combined) <= 18:
                            row_codes[row_codes.index(prefix)] = combined
                            value_lines[value_index + 1] = continuation.group("remainder")
                candidates.extend(row_codes)
            for value in candidates:
                if value not in seen:
                    seen.add(value)
                    values.append(value)
    return tuple(values)


def _explicit_container_volume_rows(
    pages: dict[int, str],
) -> tuple[tuple[str, int], ...]:
    """Return source-ordered container rows carrying an explicit ``Volumes`` count.

    GLM-OCR flattens these rows into a container line followed by named measure lines.
    We inspect only the container line and the following two lines, stopping at the
    next container.  The count remains source-authored; no remainder or row-order
    allocation is inferred.
    """

    rows: list[tuple[str, int]] = []
    seen: set[str] = set()
    for page_number in sorted(pages):
        lines = pages[page_number].splitlines()
        for index, line in enumerate(lines):
            container = _CONTAINER_ROW.search(line)
            if container is None:
                continue
            identifier = (
                container.group("prefix")
                + container.group("serial")
                + container.group("check")
            ).upper()
            window = [line]
            for following in lines[index + 1 : index + 3]:
                if _CONTAINER_ROW.search(following):
                    break
                window.append(following)
            count = _ROW_VOLUME_COUNT.search("\n".join(window))
            if count is None or identifier in seen:
                continue
            seen.add(identifier)
            rows.append((identifier, int(count.group("count"))))
    return tuple(rows)


def _explicit_valid_container_identifiers(pages: dict[int, str]) -> tuple[str, ...]:
    values: list[str] = []
    seen: set[str] = set()
    for page_number in sorted(pages):
        for found in _CONTAINER_ROW.finditer(pages[page_number]):
            identifier = (
                found.group("prefix") + found.group("serial") + found.group("check")
            ).upper()
            try:
                _CONTAINER_IDENTIFIER_ADAPTER.validate_python(identifier, strict=True)
            except ValidationError:
                continue
            if identifier not in seen:
                seen.add(identifier)
                values.append(identifier)
    return tuple(values)


def _bill_of_lading_scope_pages(pages: dict[int, str]) -> dict[int, str]:
    """Exclude later correspondence after an explicit terminal B/L page marker."""

    scoped: dict[int, str] = {}
    for page_number in sorted(pages):
        scoped[page_number] = pages[page_number]
        if _END_OF_BILL_OF_LADING.search(pages[page_number]):
            break
    return scoped


def _explicit_total_package_container_links(
    pages: dict[int, str],
) -> tuple[tuple[str, int], ...]:
    """Return explicitly stated total-package-to-container links.

    This recognizes only the completed construction ``TOTAL PACKAGES: N ...
    LOADED INTO CONTAINER(S): ID`` within one OCR page. It does not infer a
    relationship from generic adjacency, list position, or arithmetic.
    """

    values: list[tuple[str, int]] = []
    seen: set[tuple[str, int]] = set()
    for page_number in sorted(pages):
        for found in _TOTAL_PACKAGES_LOADED_CONTAINER.finditer(pages[page_number]):
            identifier = (
                found.group("prefix") + found.group("serial") + found.group("check")
            ).upper()
            try:
                _CONTAINER_IDENTIFIER_ADAPTER.validate_python(identifier, strict=True)
            except ValidationError:
                continue
            value = (identifier, int(found.group("count")))
            if value not in seen:
                seen.add(value)
                values.append(value)
    return tuple(values)


def _explicit_destination_collect(pages: dict[int, str]) -> bool:
    return any(
        _FACE_FIELD_DESTINATION_PAYABLE.search(source) for source in pages.values()
    )


def _explicit_face_field_prepaid(pages: dict[int, str]) -> bool:
    return any(_FACE_FIELD_PREPAID.search(source) for source in pages.values())


def _explicit_fact_completeness_errors(
    pages: dict[int, str], relation_label: BillOfLadingRelationExplicitLabel
) -> tuple[str, ...]:
    completeness_pages = _bill_of_lading_scope_pages(pages)
    patch = relation_label.documentPatch
    present_references = {
        _fold_with_offsets(value)[0]
        for value in patch.forwardingAndExportReferences or ()
    }
    missing_invoices = tuple(
        value
        for value in _explicit_invoice_references(completeness_pages)
        if not any(
            present == _fold_with_offsets(value)[0]
            or (
                present.startswith(_fold_with_offsets(value)[0])
                and len(present) > len(_fold_with_offsets(value)[0])
            )
            for present in present_references
        )
    )
    present_hs_codes = {
        value
        for group in patch.cargoGroups or ()
        for value in group.hsCodes or ()
    }
    missing_hs_codes = tuple(
        value
        for value in _explicit_hs_codes(completeness_pages)
        if value not in present_hs_codes
    )
    present_containers = {row.containerNumber for row in patch.containers or ()}
    missing_containers = tuple(
        value
        for value in _explicit_valid_container_identifiers(completeness_pages)
        if value not in present_containers
    )
    errors: list[str] = []
    if _explicit_destination_collect(completeness_pages) and (
        patch.freight is None or patch.freight.paymentArrangement != "collect"
    ):
        errors.append(
            "explicit 'Freight and Charges payable at destination: Yes' requires "
            "freight.paymentArrangement=collect and does not populate paymentPlace"
        )
    if _explicit_face_field_prepaid(completeness_pages) and (
        patch.freight is None or patch.freight.paymentArrangement != "prepaid"
    ):
        errors.append(
            "explicit face field 'Freight payable at PREPAID' requires "
            "freight.paymentArrangement=prepaid"
        )
    if any(
        _EXPRESS_RELEASE_NO_ORIGINALS.search(source)
        for source in completeness_pages.values()
    ) and (
        patch.negotiability != "non_negotiable"
    ):
        errors.append(
            "explicit 'EXPRESS RELEASE - NO ORIGINALS ISSUED' requires "
            "negotiability=non_negotiable"
        )
    if missing_invoices:
        errors.append(
            "explicit invoice reference(s) missing from "
            f"forwardingAndExportReferences: {', '.join(missing_invoices)}"
        )
    if missing_hs_codes:
        errors.append(
            "explicit headed HS/HSN code(s) missing from cargoGroups: "
            f"{', '.join(missing_hs_codes)}"
        )
    if missing_containers:
        errors.append(
            "valid printed ISO 6346 container identifier(s) missing from containers: "
            f"{', '.join(missing_containers)}"
        )
    groups = patch.cargoGroups or ()
    packages = patch.cargoPackages or ()
    package_container_links = _explicit_total_package_container_links(completeness_pages)
    if len(groups) == 1 and len(package_container_links) == 1:
        container_number, package_quantity = package_container_links[0]
        matching_packages = tuple(
            row
            for row in packages
            if row.groupId == groups[0].groupId and row.quantity == package_quantity
        )
        if len(matching_packages) == 1:
            package_id = matching_packages[0].packageId
            allocation = next(
                (
                    row
                    for row in patch.cargoAllocationGroups or ()
                    if row.groupId == groups[0].groupId
                ),
                None,
            )
            linked = bool(
                allocation is not None
                and allocation.coverage == "single_package_level"
                and allocation.packageIds == (package_id,)
                and any(
                    row.containerNumber == container_number
                    and row.packageQuantity == package_quantity
                    for row in allocation.allocations
                )
            )
            if not linked:
                errors.append(
                    "explicit 'TOTAL PACKAGES: "
                    f"{package_quantity} ... LOADED INTO CONTAINER(S): {container_number}' "
                    "requires a single_package_level allocation for the matching package fact"
                )
    if (
        len(groups) == 1
        and any(
            _MADE_IN_CARGO_ORIGIN.search(source)
            for source in completeness_pages.values()
        )
        and groups[0].origin is None
    ):
        errors.append(
            "explicit 'MADE IN <country>' cargo wording requires "
            "cargoGroups[0].origin"
        )
    volume_rows = _explicit_container_volume_rows(completeness_pages)
    if len(groups) == 1 and volume_rows:
        group_id = groups[0].groupId
        total = sum(quantity for _, quantity in volume_rows)
        matching_packages = tuple(
            row
            for row in packages
            if row.groupId == group_id and row.quantity == total
        )
        emitted_containers = {row.containerNumber for row in patch.containers or ()}
        if (
            len(matching_packages) == 1
            and all(container in emitted_containers for container, _ in volume_rows)
        ):
            package_id = matching_packages[0].packageId
            expected_allocations = volume_rows
            allocation = next(
                (
                    row
                    for row in patch.cargoAllocationGroups or ()
                    if row.groupId == group_id
                ),
                None,
            )
            actual_allocations = (
                tuple(
                    (row.containerNumber, row.packageQuantity)
                    for row in allocation.allocations
                )
                if allocation is not None
                else ()
            )
            if (
                allocation is None
                or allocation.coverage != "single_package_level"
                or allocation.packageIds != (package_id,)
                or actual_allocations != expected_allocations
            ):
                rendered = ", ".join(
                    f"{container}={quantity}"
                    for container, quantity in expected_allocations
                )
                errors.append(
                    "explicit container Volumes rows exactly reconcile to package "
                    f"{package_id} ({total}) and require single_package_level allocations: "
                    f"{rendered}"
                )
    return tuple(errors)


def _raise_deterministic_issues(errors: list[str]) -> None:
    if not errors:
        return
    details = "; ".join(
        f"{index}. {message}" for index, message in enumerate(errors, start=1)
    )
    raise DeterministicAnnotationError(
        f"deterministic annotation validation found {len(errors)} issue(s): {details}"
    )


def _supports_composite_grounding(path: str, target: Any) -> bool:
    if not isinstance(target, str):
        return False
    return (
        path.endswith((".address", ".description"))
        or (".parties." in path and path.endswith(".name"))
        or any(
            marker in path
            for marker in (
                ".additionalInformation[",
                ".handlingInstructions[",
                ".marksAndNumbers[",
            )
        )
    )
def _validate_semantic_policy(
    pages: dict[int, str],
    path: str,
    target: Any,
    match: _GroundedMatch,
    *,
    allow_pdf_grouped_straight_consignee: bool = False,
) -> None:
    source = pages[match.page_number]
    raw_line_start, raw_line_end = _line_bounds(source, match.start)
    raw_line = source[raw_line_start:raw_line_end]
    if (
        path.endswith(".negotiability")
        and target == "non_negotiable"
        and not allow_pdf_grouped_straight_consignee
        and non_negotiable_copy_is_only_copy_status(
            "\n".join(pages[page_number] for page_number in sorted(pages))
        )
    ):
        raise DeterministicAnnotationError(
            "NON-NEGOTIABLE COPY is a copy-status stamp and cannot override the underlying "
            "positive-original bill terms"
        )
    if any(
        marker in path
        for marker in (
            ".additionalInformation",
            ".marksAndNumbers",
        )
    ) and (
        _FORBIDDEN_METADATA.search(match.raw_value)
        or _forbidden_metadata_governs(source, match.start, match.end)
    ):
        raise DeterministicAnnotationError(
            f"target path {path} contains excluded tax/regulatory/portal metadata"
        )
    if "forwardingAndExportReferences" in path:
        if isinstance(target, str) and _REFERENCE_PLACEHOLDER.fullmatch(target):
            raise DeterministicAnnotationError(
                f"forwarding/export reference is an explicit empty placeholder: {path}"
            )
        if isinstance(target, str) and _REFERENCE_LABEL_PREFIX.match(target):
            raise DeterministicAnnotationError(
                f"forwarding/export reference must contain the value only, not its label: {path}"
            )
        reference_scope = _reference_scope(source, match.start, match.end)
        if _forbidden_metadata_governs(source, match.start, match.end):
            raise DeterministicAnnotationError(
                f"target path {path} contains excluded tax/regulatory/portal metadata"
            )
        if not (
            _explicit_invoice_reference_governs(source, match.start, match.end)
            or _has_qualifying_reference(reference_scope)
        ):
            raise DeterministicAnnotationError(
                f"forwarding/export reference lacks a qualifying heading or inline label: {path}"
            )
        if _BOOKING_REFERENCE.search(reference_scope) and not _has_qualifying_reference(
            reference_scope
        ):
            raise DeterministicAnnotationError(
                f"generic booking identifier is not a forwarding/export reference: {path}"
            )
    if ".hsCodes[" in path and not _HS_HEADING.search(
        _contiguous_nonblank_block(source, match.start, match.end)
    ):
        raise DeterministicAnnotationError(f"HS code lacks explicit HS/tariff context: {path}")
    if path.endswith(".unNumber") and not _UN_NUMBER_CONTEXT.search(match.excerpt):
        raise DeterministicAnnotationError(f"UN number lacks explicit UN context: {path}")
    if ".flashPoint." in path and not has_flash_point_context(
        _contiguous_nonblank_block(source, match.start, match.end)
    ):
        raise DeterministicAnnotationError(
            f"flash point lacks explicit flash-point context: {path}"
        )
    if ".phoneNumbers[" in path and not re.search(r"[0-9]", match.raw_value):
        raise DeterministicAnnotationError(f"phone number contains no printed digit: {path}")
    if ".phoneNumbers[" in path and _standalone_fax_governs(
        source, match.start, match.end
    ):
        raise DeterministicAnnotationError(
            f"standalone FAX value cannot populate phoneNumbers: {path}"
        )
    if (
        ".parties." in path
        and path.endswith(".name")
        and isinstance(target, str)
        and not any(character.isalpha() for character in target)
    ):
        raise DeterministicAnnotationError(
            f"party name contains no alphabetic character and is an identifier-like value: {path}"
        )
    if path.endswith(".parties.carrier.name") and re.search(
        r"(?i)\b(?:as\s+agents?\s+for|the\s+carrier)\b", match.raw_value
    ):
        raise DeterministicAnnotationError(
            "carrier name contains signing-role prose instead of the carrier identity"
        )
    if path.endswith(".transport.vesselName") and re.search(
        r"[-/\\]\s*(?=[A-Za-z0-9]*[A-Za-z])(?=[A-Za-z0-9]*[0-9])[A-Za-z0-9]{5,}$",
        match.raw_value,
    ):
        raise DeterministicAnnotationError(
            "vessel name contains a joined voyage suffix that must be modeled separately"
        )
    if path.endswith(".description") and isinstance(target, str) and _PACKING_CONSTRUCTION.search(
        target
    ):
        raise DeterministicAnnotationError(
            "cargo description contains packing-construction text; keep product wording in "
            "description and move useful packing qualifiers to additionalInformation"
        )
    if path.endswith(".verifiedGrossMass.value") and not re.search(
        r"(?ix)\bVGM\b|\bVERIFIED\s+GROSS(?:\s+MASS)?\b",
        match.excerpt,
    ):
        raise DeterministicAnnotationError(
            "container verifiedGrossMass requires explicit local VGM or verified-gross context"
        )
    if path.endswith(".netWeight.value") and _PER_PACKAGE_NET_MASS.search(raw_line):
        raise DeterministicAnnotationError(
            "per-package net mass cannot populate the cargo-group aggregate netWeight"
        )


_ALLOCATION_QUANTITY_PATH = re.compile(
    r"^documentPatch\.goodsItems\[(?P<goods>[0-9]+)\]\.containerAllocations"
    r"\[(?P<allocation>[0-9]+)\]\.packageQuantity$"
)


def _allocation_quantity_anchor(patch: dict[str, Any], path: str) -> str | None:
    found = _ALLOCATION_QUANTITY_PATH.fullmatch(path)
    if found is None:
        return None
    goods = patch.get("goodsItems")
    if not isinstance(goods, list):
        return None
    goods_index = int(found.group("goods"))
    if goods_index >= len(goods) or not isinstance(goods[goods_index], dict):
        return None
    allocations = goods[goods_index].get("containerAllocations")
    if not isinstance(allocations, list):
        return None
    allocation_index = int(found.group("allocation"))
    if allocation_index >= len(allocations) or not isinstance(
        allocations[allocation_index], dict
    ):
        return None
    container_number = allocations[allocation_index].get("containerNumber")
    return container_number if isinstance(container_number, str) else None


def _partitioned_pallet_range_evidence(
    pages: dict[int, str],
    patch: dict[str, Any],
) -> tuple[dict[str, _GroundedMatch], frozenset[str]]:
    """Ground inclusive pallet-range counts in a fully reconciled single-container partition.

    A source such as ``Pallet No. 5 - 6`` states which two outer pallets contain the
    following inner-package/product row; it does not print a standalone package count of
    ``2``. Generic numeric matching can otherwise bind that target to an unrelated page
    number. This normalization is deliberately narrow: every range must form one contiguous
    partition starting at pallet 1, the package facts must match the partition in source order,
    and one emitted container row must print the reconciled pallet total.
    """

    ranges: list[tuple[int, re.Match[str], int]] = []
    for page_number in sorted(pages):
        source = pages[page_number]
        for found in _PALLET_RANGE.finditer(source):
            first = int(found.group("first"))
            last = int(found.group("last") or found.group("first"))
            if last < first:
                return {}, frozenset()
            ranges.append((page_number, found, last - first + 1))
    if len(ranges) < 2:
        return {}, frozenset()

    starts = tuple(int(found.group("first")) for _, found, _ in ranges)
    ends = tuple(
        int(found.group("last") or found.group("first")) for _, found, _ in ranges
    )
    if starts[0] != 1 or any(
        current != previous + 1
        for previous, current in zip(ends[:-1], starts[1:], strict=True)
    ):
        return {}, frozenset()

    goods = patch.get("goodsItems")
    containers = patch.get("containers")
    if not isinstance(goods, list) or not isinstance(containers, list) or len(containers) != 1:
        return {}, frozenset()
    container_number = containers[0].get("containerNumber")
    if not isinstance(container_number, str):
        return {}, frozenset()
    total = sum(count for _, _, count in ranges)
    printed_container_total = re.compile(
        rf"(?im)^[^\n]*{re.escape(container_number)}[^\n]*\b{total}\s+PALLETS?\b"
    )
    if not any(printed_container_total.search(source) for source in pages.values()):
        return {}, frozenset()

    pallet_facts: list[tuple[int, int, int]] = []
    governed_paths: set[str] = set()
    for goods_index, item in enumerate(goods):
        if not isinstance(item, dict):
            continue
        packages = item.get("packages")
        if not isinstance(packages, list):
            continue
        for package_index, package in enumerate(packages):
            if not isinstance(package, dict):
                continue
            package_type = package.get("type")
            quantity = package.get("quantity")
            if (
                isinstance(package_type, str)
                and _PALLET_PACKAGE_TYPE.fullmatch(package_type) is not None
                and isinstance(quantity, int)
            ):
                path = (
                    f"documentPatch.goodsItems[{goods_index}].packages[{package_index}].quantity"
                )
                pallet_facts.append((goods_index, package_index, quantity))
                governed_paths.add(path)
                allocations = item.get("containerAllocations")
                if isinstance(allocations, list):
                    for allocation_index, allocation in enumerate(allocations):
                        if (
                            isinstance(allocation, dict)
                            and allocation.get("containerNumber") == container_number
                            and allocation.get("packageQuantity") == quantity
                        ):
                            governed_paths.add(
                                "documentPatch.goodsItems"
                                f"[{goods_index}].containerAllocations"
                                f"[{allocation_index}].packageQuantity"
                            )

    expected_quantities = tuple(count for _, _, count in ranges)
    actual_quantities = tuple(quantity for _, _, quantity in pallet_facts)
    if len(pallet_facts) != len(goods) or actual_quantities != expected_quantities:
        return {}, frozenset(governed_paths)

    matches: dict[str, _GroundedMatch] = {}
    for (goods_index, package_index, quantity), (page_number, found, _) in zip(
        pallet_facts, ranges, strict=True
    ):
        source = pages[page_number]
        package_path = (
            f"documentPatch.goodsItems[{goods_index}].packages[{package_index}].quantity"
        )
        matches[package_path] = _match(
            path=package_path,
            page_number=page_number,
            source=source,
            start=found.start(),
            end=found.end(),
            evidence_kind="normalized",
            normalization_rule=(
                "computed the inclusive outer-pallet quantity from the printed pallet-number range"
            ),
            base_score=140,
        )
        allocations = goods[goods_index].get("containerAllocations")
        if not isinstance(allocations, list):
            continue
        for allocation_index, allocation in enumerate(allocations):
            if (
                isinstance(allocation, dict)
                and allocation.get("containerNumber") == container_number
                and allocation.get("packageQuantity") == quantity
            ):
                allocation_path = (
                    "documentPatch.goodsItems"
                    f"[{goods_index}].containerAllocations"
                    f"[{allocation_index}].packageQuantity"
                )
                matches[allocation_path] = _match(
                    path=allocation_path,
                    page_number=page_number,
                    source=source,
                    start=found.start(),
                    end=found.end(),
                    evidence_kind="normalized",
                    normalization_rule=(
                        "computed the inclusive outer-pallet allocation from the printed "
                        "pallet-number range"
                    ),
                    base_score=140,
                )
    return matches, frozenset(governed_paths)


def derive_field_evidence(
    work_item: AgentWorkItem,
    normal_label: BillOfLadingLabel,
    *,
    pdf_grouping_used: bool = False,
) -> tuple[FieldEvidence, ...]:
    """Ground every normal target leaf by exact deterministic normalization."""

    pages = page_texts(work_item.joinedRawText)
    patch = normal_label.canonical_target()["documentPatch"]
    pallet_range_matches, pallet_range_governed_paths = _partitioned_pallet_range_evidence(
        pages, patch
    )
    evidence: list[FieldEvidence] = []
    errors: list[str] = []
    for path, target in _leaf_items(patch, "documentPatch"):
        matches: tuple[_GroundedMatch, ...]
        grouping_only = False
        range_match = pallet_range_matches.get(path)
        if range_match is not None:
            matches = (range_match,)
        elif path in pallet_range_governed_paths:
            errors.append(
                "partitioned pallet-range quantity does not match the complete, source-ordered "
                f"inclusive range partition: {path}={target!r}"
            )
            continue
        else:
            try:
                matches = (
                    _best_match(
                        pages,
                        path,
                        target,
                        anchor=_allocation_quantity_anchor(patch, path),
                    ),
                )
            except DeterministicAnnotationError as error:
                grouped_match = (
                    _pdf_grouped_straight_consignee_match(
                        pages, patch, path, target
                    )
                    if pdf_grouping_used
                    else None
                )
                if grouped_match is not None:
                    matches = (grouped_match,)
                    grouping_only = True
                else:
                    matches = (
                        _composite_text_matches(pages, path, target)
                        if _supports_composite_grounding(path, target)
                        else ()
                    )
                if not matches:
                    errors.append(str(error))
                    continue
        try:
            for match in matches:
                _validate_semantic_policy(
                    pages,
                    path,
                    target,
                    match,
                    allow_pdf_grouped_straight_consignee=grouping_only,
                )
        except DeterministicAnnotationError as error:
            errors.append(str(error))
            continue
        unique_matches = tuple(
            {
                (match.page_number, match.raw_value, match.excerpt): match
                for match in matches
            }.values()
        )
        matches = unique_matches
        evidence_kind = matches[0].evidence_kind
        normalization_rule = matches[0].normalization_rule
        evidence.append(
            FieldEvidence.model_validate(
                {
                    "targetPath": path,
                    "evidenceKind": evidence_kind,
                    "rawOcrEvidence": tuple(
                        {
                            "pageNumber": match.page_number,
                            "rawValue": match.raw_value,
                            "ocrExcerpt": match.excerpt,
                        }
                        for match in matches
                    ),
                    "imageUse": "grouping_only" if grouping_only else "not_used",
                    "normalizationRule": normalization_rule,
                    "note": None,
                },
                strict=True,
            )
        )
    if errors:
        details = "; ".join(
            f"{index}. {message}" for index, message in enumerate(errors, start=1)
        )
        raise DeterministicAnnotationError(
            f"deterministic evidence validation found {len(errors)} issue(s): {details}"
        )
    return tuple(evidence)


def _raw_sort_key(
    raw: RawOcrValueEvidence, pages: dict[int, str]
) -> tuple[int, int, str]:
    source = pages[raw.pageNumber]
    excerpt_offset = source.find(raw.ocrExcerpt)
    raw_inside_excerpt = raw.ocrExcerpt.find(raw.rawValue)
    if excerpt_offset < 0 or raw_inside_excerpt < 0:
        raise DeterministicAnnotationError("locally generated relationship evidence is absent")
    return (raw.pageNumber, excerpt_offset + raw_inside_excerpt, raw.rawValue)


def derive_relation_evidence(
    work_item: AgentWorkItem,
    relation_label: BillOfLadingRelationExplicitLabel,
    field_evidence: tuple[FieldEvidence, ...],
    *,
    pdf_grouping_used: bool,
) -> tuple[CargoRelationEvidence, ...]:
    """Build relationship sidecars from already grounded allocation/package leaves."""

    allocation_groups = relation_label.documentPatch.cargoAllocationGroups or ()
    if not allocation_groups:
        return ()
    pages = page_texts(work_item.joinedRawText)
    evidence_by_path = {row.targetPath: row.rawOcrEvidence for row in field_evidence}
    packages = relation_label.documentPatch.cargoPackages or ()
    package_by_id = {row.packageId: row for row in packages}
    package_position_by_group: dict[tuple[str, str], int] = {}
    for group_id in {row.groupId for row in packages}:
        for index, package in enumerate(
            row for row in packages if row.groupId == group_id
        ):
            package_position_by_group[(group_id, package.packageId)] = index

    records: list[CargoRelationEvidence] = []
    for group in allocation_groups:
        goods_index = int(group.groupId[1:]) - 1
        anchors: list[RawOcrValueEvidence] = []
        for allocation_index, allocation in enumerate(group.allocations):
            base = (
                f"documentPatch.goodsItems[{goods_index}].containerAllocations"
                f"[{allocation_index}]"
            )
            anchors.extend(evidence_by_path[f"{base}.containerNumber"])
            if allocation.packageQuantity is not None:
                anchors.extend(evidence_by_path[f"{base}.packageQuantity"])
        for package_id in group.packageIds:
            package = package_by_id[package_id]
            package_index = package_position_by_group[(group.groupId, package_id)]
            base = f"documentPatch.goodsItems[{goods_index}].packages[{package_index}]"
            if package.quantity is not None:
                anchors.extend(evidence_by_path[f"{base}.quantity"])
            if package.typeDescription is not None:
                anchors.extend(evidence_by_path[f"{base}.type"])
        # A scalar used by both the package and allocation projections is one
        # printed relationship anchor, even if each leaf generated a different
        # local excerpt.  The absolute OCR position is the stable source of
        # identity and preserves source order without duplicating evidence.
        unique = {_raw_sort_key(row, pages): row for row in anchors}
        ordered = tuple(unique[key] for key in sorted(unique))
        if not ordered:
            raise DeterministicAnnotationError(
                f"cargo allocation group {group.groupId} has no grounded relationship anchors"
            )
        if group.coverage == "container_membership_only":
            basis = "container_membership"
        elif group.coverage in {
            "single_package_level",
            "all_package_levels_combined",
        }:
            basis = "quantity_reconciliation"
        else:
            basis = "row_alignment"
        records.append(
            CargoRelationEvidence.model_validate(
                {
                    "groupId": group.groupId,
                    "coverage": group.coverage,
                    "relationshipBasis": basis,
                    "rawOcrEvidence": ordered,
                    "pdfUse": "grouping_only" if pdf_grouping_used else "not_used",
                    "note": (
                        "Deterministically assembled from the exact OCR anchors of the emitted "
                        "package and container-allocation facts."
                    ),
                },
                strict=True,
            )
        )
    return tuple(records)


def build_compact_annotation(
    work_item: AgentWorkItem,
    draft: CompactAnnotationDraft,
    *,
    pdf_grouping_used: bool,
) -> BillOfLadingDualCargoAnnotation:
    relation_label = draft.relationExplicitLabel.to_canonical()
    normal_label = project_relation_to_normal(relation_label)
    pages = page_texts(work_item.joinedRawText)
    errors = list(_explicit_fact_completeness_errors(pages, relation_label))
    for warning in draft.warnings:
        if (
            warning.code == "ambiguous_ocr_candidates"
            and warning.targetPath
            in {"documentPatch.issueDate", "documentPatch.shippedOnBoardDate"}
        ):
            errors.append(
                f"{warning.targetPath} numeric order must use the day-first ambiguity policy, "
                "not an ambiguity warning"
            )
        if (
            warning.code == "ambiguous_ocr_candidates"
            and warning.targetPath is not None
            and re.fullmatch(
                r"documentPatch\.cargoPackages\[[0-9]+\]\.quantity",
                warning.targetPath,
            )
            and _DOT_GROUPED_INTEGER_CONTEXT.search(work_item.joinedRawText)
        ):
            errors.append(
                f"{warning.targetPath} uses a dot-grouped integer immediately before a "
                "supported count noun and must be normalized as an integer, not emitted as "
                "an ambiguity warning"
            )
    try:
        evidence = derive_field_evidence(
            work_item,
            normal_label,
            pdf_grouping_used=pdf_grouping_used,
        )
    except DeterministicAnnotationError as error:
        errors.append(str(error))
        evidence = ()
    _raise_deterministic_issues(errors)
    relation_evidence = derive_relation_evidence(
        work_item,
        relation_label,
        evidence,
        pdf_grouping_used=pdf_grouping_used,
    )
    annotation = BillOfLadingDualCargoAnnotation.model_validate(
        {
            "annotationSchemaVersion": "3.0.0-experimental",
            "taskType": "bill_of_lading_dual_cargo_kie",
            "documentType": draft.documentType,
            "source": work_item.source,
            "normalLabel": normal_label,
            "relationExplicitLabel": relation_label,
            "evidence": evidence,
            "relationEvidence": relation_evidence,
            "warnings": draft.warnings,
            "reviewStatus": "candidate",
            "reviewNotes": draft.decisionNotes,
        },
        strict=True,
    )
    validate_annotation_evidence(work_item, annotation)
    return annotation
