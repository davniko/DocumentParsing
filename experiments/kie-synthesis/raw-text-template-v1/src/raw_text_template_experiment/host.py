from __future__ import annotations

import bisect
import json
import re
import unicodedata
from collections import Counter, defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, replace
from datetime import date, datetime
from decimal import Decimal, InvalidOperation
from itertools import pairwise
from pathlib import Path
from typing import Any, Literal

from document_ocr.hashing import canonical_json_bytes, sha256_bytes, sha256_file
from document_ocr.synthesis.generators import surface_pattern
from document_ocr.synthesis.raw_text_template import (
    TemplateSlot,
    build_template_slot,
    compile_raw_text_template,
    render_compiled_template,
    sentinel_bindings,
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
_TOKEN = re.compile(r"(?<![A-Za-z0-9])[A-Za-z0-9][A-Za-z0-9./_-]{5,}(?![A-Za-z0-9])")
_LONG_NUMBER = re.compile(r"(?<![0-9])[0-9]{5,}(?![0-9])")
_PATH_TOKEN = re.compile(r"([^.\[\]]+)|\[([0-9]+)\]")
_TARGET_PATH = re.compile(
    r"^[A-Za-z_][A-Za-z0-9_]*(?:\[[0-9]+\])?"
    r"(?:\.[A-Za-z_][A-Za-z0-9_]*(?:\[[0-9]+\])?)*$"
)
_CONTAINER_NUMBER_PATH = re.compile(r"^documentPatch\.containers\[([0-9]+)\]\.containerNumber$")
_CONTAINER_TYPE_PATH = re.compile(r"^documentPatch\.containers\[([0-9]+)\]\.typeDescription$")
_PHONE_CONTEXT = re.compile(r"(?i)\b(?:phone|telephone|tel|mobile|fax|phn|fx)\b")
_COUNTRY_CODE_CAPTION = re.compile(
    r"(?is)\b(?:country(?:\s+of\s+origin)?|nationality)\s+code\s*[:=-]?\s*$"
)
_COUNTRY_CODE_LABELED_VALUE = re.compile(
    r"(?i)\b(?:country(?:\s+of\s+origin)?|nationality)\s+code\s*[:=-]?\s*"
    r"(?P<value>[A-Z]{2})(?![A-Z0-9])"
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


@dataclass(frozen=True)
class RequiredTargetCoBinding:
    relationship: str
    target_paths: tuple[str, ...]


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
    if ".containers[" in joined and any(
        field in joined for field in ("typecategory", "sizecategory", "typedescription")
    ):
        return "equipment"
    if "dangerousgoods" in joined:
        return "dangerous_goods"
    if "hscodes" in joined:
        return "identifier"
    if ".transport.vesselname" in joined:
        return "equipment"
    if ".freight.paymentplace." in joined:
        return "location"
    if ".freight.paymentarrangement" in joined:
        return "commercial_text"
    if "temperature" in joined:
        return "temperature"
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
                evidence_origin="accepted_label_evidence",
                render_policy=policy,
                rationale=f"Pinned accepted-label evidence group {ordinal}.",
            )
        )
    return tuple(drafts)


def _overlaps(start: int, end: int, spans: Sequence[SpanDraft]) -> bool:
    return any(start < span.char_end and span.char_start < end for span in spans)


def risk_candidates(raw: str, owned: Sequence[SpanDraft]) -> tuple[RiskCandidate, ...]:
    candidates: list[tuple[int, int, str, str, str]] = []
    patterns: tuple[tuple[str, re.Pattern[str]], ...] = (
        ("email", _EMAIL),
        ("url_or_domain", _URL),
        ("equipment_identifier", _EQUIPMENT),
        ("date", _DATE),
        ("measurement", _MEASUREMENT),
        ("phone", _PHONE),
        ("alphanumeric_identifier", _TOKEN),
        ("long_numeric_identifier", _LONG_NUMBER),
    )
    for line in line_spans(raw):
        if not line.text.strip() or _PAGE_HEADER.fullmatch(line.text):
            continue
        accepted_on_line: list[tuple[int, int]] = []
        for kind, pattern in patterns:
            for match in pattern.finditer(line.text):
                text = match.group(0).rstrip(".,;:)]}")
                if not text:
                    continue
                local_start = match.start()
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


def _is_redundant_unresolved_equipment_projection(
    *,
    proposal: AgentBindingProposal,
    occurrence: AgentOccurrence,
    resolved: Sequence[tuple[AgentBindingProposal, AgentOccurrence, int, int]],
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
        for other_proposal, other_occurrence, _char_start, _char_end in resolved
    )


def resolve_agent_proposals(
    *, raw: str, proposals: Sequence[AgentBindingProposal]
) -> tuple[SpanDraft, ...]:
    lines = line_spans(raw)
    by_number = {line.number: line for line in lines}
    resolved: list[tuple[AgentBindingProposal, AgentOccurrence, int, int]] = []
    resolution_errors: list[str] = []
    unresolved_proposal_errors: list[tuple[AgentBindingProposal, AgentOccurrence, ValueError]] = []
    for proposal in proposals:
        proposal_resolved: list[tuple[AgentBindingProposal, AgentOccurrence, int, int]] = []
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
            try:
                char_start, char_end = _resolve_occurrence(
                    raw=raw, occurrence=occurrence, line_rows=lines
                )
            except ValueError as error:
                recovered = _resolve_unique_local_phone_typo(
                    raw=raw,
                    proposal=proposal,
                    occurrence=occurrence,
                    lines=lines,
                )
                if recovered is None:
                    proposal_errors.append((occurrence, error))
                    continue
                char_start, char_end = recovered
            proposal_resolved.append((proposal, occurrence, char_start, char_end))
        resolved.extend(proposal_resolved)
        for occurrence, error in proposal_errors:
            global_offsets = set(_exact_offsets(raw, occurrence.source_text))
            resolved_offsets = {
                char_start
                for _proposal, resolved_occurrence, char_start, _char_end in proposal_resolved
                if resolved_occurrence.source_text == occurrence.source_text
            }
            if global_offsets and resolved_offsets == global_offsets:
                # The invalid declaration cannot identify another physical slot: the same
                # proposal already selected every exact source occurrence of this value.  Drop
                # only that provably redundant hallucinated duplicate.
                continue
            unresolved_proposal_errors.append((proposal, occurrence, error))
    for proposal, occurrence, error in unresolved_proposal_errors:
        if _is_redundant_unresolved_equipment_projection(
            proposal=proposal,
            occurrence=occurrence,
            resolved=resolved,
        ):
            continue
        resolution_errors.append(
            f"{proposal.logical_key} {occurrence.line_start}-{occurrence.line_end}: {error}"
        )
    if resolution_errors:
        raise ValueError(
            "agent source texts cannot be resolved unambiguously; correct every listed "
            "occurrence in one complete replacement:\n- " + "\n- ".join(resolution_errors)
        )

    drafts: list[SpanDraft] = []
    for proposal, occurrence, char_start, char_end in resolved:
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
                        " Host localized a one-character phone transcription error to the "
                        "unique typed value in its declared source line."
                        if resolved_source_text != occurrence.source_text
                        else ""
                    )
                ),
            )
        )
    return tuple(drafts)


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
        # A single allocation against a single referenced package is the same scalar fact for both
        # supported coverage shapes. Multi-allocation single-package groups instead encode a sum,
        # so cardinality, explicit group reference, and value equality are all required here.
        if not isinstance(package_ids, list) or len(package_ids) != 1 or len(allocations) != 1:
            continue
        package_id = package_ids[0]
        allocation = allocations[0]
        if not isinstance(package_id, str):
            continue
        if coverage == "one_to_one_package_allocations":
            if allocation.get("packageId") != package_id:
                continue
        elif allocation.get("packageId") is not None:
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
                    f"[{group_index}].allocations[0].packageQuantity",
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

    Host-derived co-binding paths are one fact. Every other target path is independent even when
    its current canonical value equals another path's value.
    """

    component_by_path: dict[str, int] = {}
    for component_index, requirement in enumerate(required_target_cobindings(source_target)):
        for path in requirement.target_paths:
            prior = component_by_path.setdefault(path, component_index)
            if prior != component_index:
                raise ValueError(f"target path belongs to multiple co-binding components: {path}")
    grouped: dict[tuple[str, int | str], list[str]] = {}
    for path in paths:
        key: tuple[str, int | str]
        if path in component_by_path:
            key = ("required", component_by_path[path])
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
    r"(?i)^\s*(?:[0-9][0-9,]*|[a-z]+(?:[ -][a-z]+)*)\s+container(?:\(s\)|s)?\s*$"
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
            and draft.target_paths[0].endswith(".temperatureSetpoint")
        ):
            base = draft.target_paths[0]
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
            first.render_mode == "target_binding"
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


def _split_exact_composite_target_surfaces(
    *, drafts: Sequence[SpanDraft], source_target: Mapping[str, Any]
) -> tuple[SpanDraft, ...]:
    """Split an inseparable proposal when exact target token sequences prove every boundary."""

    output: list[SpanDraft] = []
    for draft in drafts:
        if draft.render_mode != "agent_residual" or len(draft.target_paths) < 2:
            output.append(draft)
            continue
        source_tokens = _surface_token_spans(draft.source_text)
        if not source_tokens:
            output.append(draft)
            continue
        matches: list[tuple[str, int, int]] = []
        occupied_token_indexes: set[int] = set()
        for path in draft.target_paths:
            target = _scalar_surface(_resolve_target_path(source_target, path))
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
            matches.append((path, relative_start, relative_end))
        if not matches or occupied_token_indexes != set(range(len(source_tokens))):
            output.append(draft)
            continue
        for path, relative_start, relative_end in matches:
            char_start = draft.char_start + relative_start
            char_end = draft.char_start + relative_end
            value_kind = _value_kind((path,), "natural_text")
            group_kind, group_key = _canonical_target_group((path,))
            output.append(
                replace(
                    draft,
                    draft_id=(
                        "normalized_composite_"
                        + sha256_bytes(f"{draft.draft_id}\0{path}".encode())[:16]
                    ),
                    logical_key="anchor:" + path,
                    render_mode="target_binding",
                    value_kind=value_kind,
                    group_kind=group_kind,
                    group_key=group_key,
                    target_paths=(path,),
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


def normalize_deterministic_draft_semantics(
    *, drafts: Sequence[SpanDraft], source_target: Mapping[str, Any]
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
    policy_normalized = tuple(
        replace(
            draft,
            render_policy=_render_policy_for_value_kind(draft.value_kind, draft.render_mode),
        )
        for draft in drafts
    )
    composite_normalized = _split_exact_composite_target_surfaces(
        drafts=policy_normalized,
        source_target=source_target,
    )
    repeated_line_normalized = _split_exact_repeated_line_target_surfaces(
        drafts=composite_normalized,
        source_target=source_target,
    )
    temperature_normalized = _normalize_temperature_setpoints(
        drafts=repeated_line_normalized, source_target=source_target
    )
    count_normalized: list[SpanDraft] = []
    for draft in temperature_normalized:
        if (
            has_container_collection
            and draft.render_mode == "deterministic_derived"
            and containers_path in {*draft.target_paths, *draft.dependency_paths}
            and (
                draft.derivation == "container_count"
                or _CONTAINER_COUNT_SURFACE.fullmatch(draft.source_text) is not None
            )
        ):
            count_normalized.append(
                replace(
                    draft,
                    logical_key="agent:container_count:documentPatch.containers",
                    render_mode="deterministic_derived",
                    value_kind="integer",
                    group_kind="equipment",
                    group_key="equipment:all",
                    derivation="container_count",
                    target_paths=(containers_path,),
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

    grouped: dict[str, list[SpanDraft]] = defaultdict(list)
    for draft in count_normalized:
        grouped[draft.logical_key].append(draft)
    output: list[SpanDraft] = []
    for logical_key, rows in grouped.items():
        first = rows[0]
        surfaces: dict[str, list[SpanDraft]] = defaultdict(list)
        for row in rows:
            surfaces[_normalized_surface(row.source_text)].append(row)
        normalized_values = tuple(value for value in surfaces if value)
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
    return merge_drafts(output)


def validate_target_binding_relationships(
    *, drafts: Sequence[SpanDraft], source_target: Mapping[str, Any]
) -> None:
    for draft in drafts:
        relationship = target_path_relationship(source_target, draft.target_paths)
        if draft.render_mode == "target_binding" and relationship == "composite_target_surface":
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

    validate_repeated_binding_fact_topology(drafts=drafts, source_target=source_target)

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
    *, drafts: Sequence[SpanDraft], source_target: Mapping[str, Any]
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
    for logical_key, occurrences in grouped_drafts.items():
        first = occurrences[0]
        if (
            len(occurrences) < 2
            or len(first.target_paths) < 2
            or first.render_mode in {"agent_residual", "deterministic_derived"}
        ):
            continue
        independent_components = target_fact_components(source_target, first.target_paths)
        if len(independent_components) > 1:
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
                prior = replacements.setdefault(participant.logical_key, canonical)
                if _semantic_signature(prior) != _semantic_signature(canonical):
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

    current = list(
        _normalize_derived_dependency_containment(
            _normalize_required_cobinding_dominance(
                drafts=_normalize_fmc_carrier_duplicate_ownership(
                    raw=raw,
                    drafts=drafts,
                ),
                source_target=source_target,
            )
        )
    )
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


def merge_drafts(*groups: Sequence[SpanDraft]) -> tuple[SpanDraft, ...]:
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
        prior = signatures.setdefault(draft.logical_key, signature)
        if signature != prior:
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


def _repair_uniquely_bracketed_target_paths(
    *,
    raw: str,
    missing_paths: Sequence[str],
    occupied: Sequence[SpanDraft],
    source_target: Mapping[str, Any],
    source_hints: Mapping[str, Sequence[str]] | None = None,
) -> tuple[SpanDraft, ...]:
    """Recover exact scalar slots only when same-entity row topology proves one span."""

    pages = page_body_spans(raw)
    contexts_by_path: dict[str, tuple[SpanDraft, ...]] = {}
    ranked_by_path: dict[str, tuple[tuple[tuple[int, int, int], int, int], ...]] = {}
    values_by_path: dict[str, str] = {}
    for path in sorted(missing_paths):
        entity = _indexed_target_entity(path)
        value = _resolve_target_path(source_target, path)
        if entity is None or not isinstance(value, str) or not value:
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
        candidate_surfaces = tuple(dict.fromkeys((value, *((source_hints or {}).get(path, ())))))
        seen_spans: set[tuple[int, int]] = set()
        for surface in candidate_surfaces:
            for start in _exact_offsets(raw, surface):
                end = start + len(surface)
                if (start, end) in seen_spans or any(
                    start < draft.char_end and draft.char_start < end for draft in occupied
                ):
                    continue
                seen_spans.add((start, end))
                score = _bracketed_context_score(pages, start, end, contexts)
                if score is not None:
                    ranked.append((score, start, end))
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

    for path, ranked in ranked_by_path.items():
        if path in assigned_paths or (len(ranked) > 1 and ranked[0][0] == ranked[1][0]):
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
    removed = tuple(by_id[anchor_id] for anchor_id in override_ids)
    retained = tuple(anchor for anchor in anchors if anchor.draft_id not in override_ids)
    proposed_paths = _represented_target_inputs(proposed)
    missing_paths = sorted(
        {path for draft in removed for path in draft.target_paths}
        - proposed_paths
        - set(semantic_only_target_paths)
    )
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
                    if any(start < draft.char_end and draft.char_start < end for draft in occupied):
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
    # A compiler-added repeat joins the retained anchor's logical binding. Changing the semantic
    # value kind requires overriding that complete binding; the critic can later do so atomically.
    retained_semantics = {draft.logical_key: draft.value_kind for draft in retained}
    harmonized_proposals = tuple(
        replace(
            draft,
            value_kind=retained_semantics.get(draft.logical_key, draft.value_kind),
        )
        for draft in proposed
    )
    return reconcile_draft_overlaps(
        raw=raw,
        drafts=(*retained, *harmonized_proposals),
        source_target=source_target,
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

    Every proposed logical binding must touch a source line cited by a finding. Existing edits are
    addressed either at logical-binding granularity or by exact occurrence. Partial occurrence
    removal preserves the existing semantic contract and is permitted only for cited source spans;
    it cannot empty the binding. An entirely uncited group may be removed only when an in-scope
    replacement covers all of its exact physical spans. Target ownership removed by the patch must
    still exist afterwards. The next independent critic pass remains responsible for semantic
    acceptance of the repaired inventory.
    """

    if not remove_inventory_binding_ids and not occurrence_removals and not additional_bindings:
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
            occupied = set(range(int(start_line[1:]), int(end_line[1:]) + 1))
            if not occupied & cited_line_numbers:
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
    validate_agent_proposal_paths(proposals=additional_bindings, source_target=source_target)
    additions = resolve_agent_proposals(raw=raw, proposals=additional_bindings)
    additions_by_key: dict[str, list[SpanDraft]] = defaultdict(list)
    for draft in additions:
        additions_by_key[draft.logical_key].append(draft)
    in_scope_additions: set[str] = set()
    for logical_key, logical_drafts in additions_by_key.items():
        if any(
            set(
                range(
                    int(line_range_for_chars(lines, draft.char_start, draft.char_end)[0][1:]),
                    int(line_range_for_chars(lines, draft.char_start, draft.char_end)[1][1:]) + 1,
                )
            )
            & cited_line_numbers
            for draft in logical_drafts
        ):
            in_scope_additions.add(logical_key)
        else:
            first = logical_drafts[0]
            start_line, end_line = line_range_for_chars(lines, first.char_start, first.char_end)
            raise ValueError(
                "critic logical addition is outside its cited findings: "
                f"{logical_key} {start_line}-{end_line}"
            )
    replacement_spans = {
        (draft.char_start, draft.char_end)
        for logical_key, logical_drafts in additions_by_key.items()
        if logical_key in in_scope_additions
        for draft in logical_drafts
    }
    removal_groups = tuple(grouped[binding_id] for binding_id in removal_ids)
    for binding_id, logical_drafts in zip(removal_ids, removal_groups, strict=True):
        occupied: set[int] = set()
        line_ranges: list[str] = []
        for draft in logical_drafts:
            start_line, end_line = line_range_for_chars(lines, draft.char_start, draft.char_end)
            occupied.update(range(int(start_line[1:]), int(end_line[1:]) + 1))
            line_ranges.append(f"{start_line}-{end_line}")
        if not occupied & cited_line_numbers and not all(
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
    revised = normalize_target_cobindings(
        drafts=reconcile_draft_overlaps(
            raw=raw,
            drafts=(*retained, *additions),
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
    validate_target_binding_relationships(drafts=revised, source_target=source_target)
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
    if _normalized_surface(source) == _normalized_surface(target_surface):
        return "", ""
    span = _matching_token_frame(source, target_surface)
    return (source[: span[0]], source[span[1] :]) if span is not None else None


def _surface_adapter(draft: SpanDraft) -> str:
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
        projections = tuple(
            _matching_normalized_projection(slot.source_text, target_surface) for slot in slots
        )
        if all(projection is not None for projection in projections) and any(
            projection != ("", "") for projection in projections
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
                        for slot, projection in zip(slots, projections, strict=True)
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
    *, raw: str, draft: SpanDraft, number_drafts: Mapping[int, Sequence[SpanDraft]]
) -> int | None:
    """Return a uniquely local container row, never merely the nearest row on a page.

    OCR layouts commonly print a complete container-number list followed by cargo blocks.  Raw
    character distance across that layout is not row evidence: it incorrectly attaches the first
    cargo-block type to the last number in the preceding list.  A row association is mechanically
    defensible only when both spans share a source line, or when nothing but whitespace separates
    them (the common two-line ``number\n40HQ`` layout).
    """

    page_spans = page_body_spans(raw)
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
    lines = line_spans(raw)
    draft_line_start, draft_line_end = line_range_for_chars(lines, draft.char_start, draft.char_end)
    draft_line_numbers = set(range(int(draft_line_start[1:]), int(draft_line_end[1:]) + 1))

    def distance(number: SpanDraft) -> int:
        if number.char_end <= draft.char_start:
            return draft.char_start - number.char_end
        if draft.char_end <= number.char_start:
            return number.char_start - draft.char_end
        return 0

    same_line: list[tuple[int, int]] = []
    whitespace_adjacent: list[tuple[int, int]] = []
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
            if between_start <= between_end and not raw[between_start:between_end].strip():
                whitespace_adjacent.append(candidate)

    local_candidates = same_line or whitespace_adjacent
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
    """Safely detach a compact type token from the wrong neighboring container binding."""

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
            scoped_type_paths = {
                path
                for other in drafts
                if other.group_key == draft.group_key
                for path in other.target_paths
                if _CONTAINER_TYPE_PATH.fullmatch(path) is not None
            }
            scoped_target = next(iter(scoped_type_paths)) if len(scoped_type_paths) == 1 else None
            nearest_index = _nearest_container_number_index(
                raw=raw, draft=draft, number_drafts=number_drafts
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
            raw=raw, draft=draft, number_drafts=number_drafts
        )
        if nearest_index is None or nearest_index == target_index:
            output.append(draft)
            continue
        nearest_container = containers[nearest_index] if nearest_index < len(containers) else None
        siblings = tuple(
            row
            for row in drafts
            if row.logical_key == draft.logical_key and row.draft_id != draft.draft_id
        )
        target_has_local_sibling = any(
            _nearest_container_number_index(raw=raw, draft=sibling, number_drafts=number_drafts)
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
        if not target_has_local_sibling:
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


def validate_compact_equipment_locality(
    *, raw: str, drafts: Sequence[SpanDraft], source_target: Mapping[str, Any]
) -> None:
    """Reject a short equipment surface assigned away from its unique local container row."""

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
            raw=raw, draft=draft, number_drafts=number_drafts
        )
        if nearest_index is not None and nearest_index != declared_index:
            errors.append(
                f"{draft.logical_key} compact equipment at chars "
                f"[{draft.char_start},{draft.char_end}) is scoped to container:{declared_index} "
                f"but uniquely nearest container:{nearest_index}"
            )
    if errors:
        raise ValueError("compact equipment locality violations: " + "; ".join(errors))


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


def validate_binding_realizations(
    *, raw: str, drafts: Sequence[SpanDraft], source_target: Mapping[str, Any]
) -> None:
    """Reject mutable bindings whose declared semantics cannot realize their own surfaces."""

    validate_compact_equipment_locality(raw=raw, drafts=drafts, source_target=source_target)
    slots = _template_slots(raw, drafts)
    grouped: dict[str, list[tuple[SpanDraft, TemplateSlot]]] = defaultdict(list)
    for draft, slot in zip(drafts, slots, strict=True):
        grouped[draft.logical_key].append((draft, slot))
    errors: list[str] = []
    for logical_key, rows in grouped.items():
        first = rows[0][0]
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
        rows: list[SourceBindingRelationship] = []
        if owner.render_mode in {"deterministic_auxiliary", "agent_residual"}:
            for dependency_key, (
                dependency,
                dependency_surface,
            ) in unique_identifier_surfaces.items():
                if (
                    dependency_key == owner_key
                    or dependency.render_mode == "deterministic_derived"
                    or len(dependency_surface) < 6
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
                rows.append(
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
            sorted(rows, key=lambda row: (row.dependency_binding, row.source_prefix))
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
) -> CertifiedSemanticTemplate:
    if not critic_outputs or critic_outputs[-1].verdict != "pass":
        raise ValueError("template lacks a final independent critic pass")
    validate_target_binding_relationships(drafts=drafts, source_target=source_target)
    validate_binding_realizations(raw=raw, drafts=drafts, source_target=source_target)
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
    masked = masked_source(raw, drafts)
    return CertifiedSemanticTemplate.model_validate(
        {
            "schema_version": 4,
            "compiler": "carrier_bound_semantic_template_v4",
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
                }
            ),
        }
    )


def template_summary(template: CertifiedSemanticTemplate) -> dict[str, Any]:
    modes: dict[str, int] = defaultdict(int)
    realization_modes: dict[str, int] = defaultdict(int)
    values: dict[str, int] = defaultdict(int)
    for binding in template.bindings:
        modes[binding.render_mode] += 1
        realization_modes[binding.realization.mode] += 1
        values[binding.value_kind] += 1
    return {
        "documentId": template.document_id,
        "sourceSha256": template.source_sha256,
        "carrier": template.carrier.canonical_name,
        "carrierFamily": template.carrier.family,
        "templateProxyId": template.capability.template_proxy_id,
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
        "compiler": template.compiler,
        "certified": True,
    }
