"""Deterministic, byte-preserving rendering for audited OCR evidence spans."""

from __future__ import annotations

import re
from collections import defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import date, datetime
from decimal import Decimal, InvalidOperation
from itertools import pairwise
from typing import Any, cast

from document_ocr.hashing import canonical_json_bytes, sha256_bytes
from document_ocr.synthesis.generation_models import (
    DeterministicTextEdit,
    DeterministicTextPatchPlan,
    SemanticChange,
)

_PAGE_HEADER = re.compile(r"(?m)^--- PAGE ([1-9][0-9]*) ---$")
_NUMBER = re.compile(r"[+-]?[0-9](?:[0-9., '\u00a0]*[0-9])?")
_DATE_FORMATS = (
    "%Y-%m-%d",
    "%Y/%m/%d",
    "%Y.%m.%d",
    "%d/%m/%Y",
    "%m/%d/%Y",
    "%d-%m-%Y",
    "%m-%d-%Y",
    "%d.%m.%Y",
    "%m.%d.%Y",
    "%d/%m/%y",
    "%m/%d/%y",
    "%d-%m-%y",
    "%m-%d-%y",
    "%d %b %Y",
    "%d %B %Y",
    "%b %d %Y",
    "%B %d %Y",
    "%d %b, %Y",
    "%d %B, %Y",
    "%b %d, %Y",
    "%B %d, %Y",
)


@dataclass(frozen=True, slots=True)
class _PageSpan:
    number: int
    body_start: int
    body_end: int


def _page_spans(joined_raw_text: str) -> tuple[_PageSpan, ...]:
    matches = list(_PAGE_HEADER.finditer(joined_raw_text))
    numbers = [int(match.group(1)) for match in matches]
    if numbers != list(range(1, len(matches) + 1)):
        raise ValueError("joined OCR page headers must be contiguous and start at page one")
    spans: list[_PageSpan] = []
    for index, match in enumerate(matches):
        section_end = (
            matches[index + 1].start() if index + 1 < len(matches) else len(joined_raw_text)
        )
        body_start = match.end()
        while body_start < section_end and joined_raw_text[body_start] == "\n":
            body_start += 1
        body_end = section_end
        while body_end > body_start and joined_raw_text[body_end - 1] == "\n":
            body_end -= 1
        spans.append(_PageSpan(int(match.group(1)), body_start, body_end))
    return tuple(spans)


def _case_like(source: str, rendered: str) -> str:
    letters = [character for character in source if character.isalpha()]
    if letters and all(character.isupper() for character in letters):
        return rendered.upper()
    if letters and all(character.islower() for character in letters):
        return rendered.lower()
    return rendered


def _preserve_unpadded_day(source: str, rendered: str, *, old: date, new: date) -> str:
    if old.day >= 10:
        return rendered
    unpadded = re.search(rf"(?<![0-9]){old.day}(?![0-9])", source)
    padded = re.search(rf"(?<![0-9])0{old.day}(?![0-9])", source)
    if unpadded is None or padded is not None:
        return rendered
    return re.sub(
        rf"(?<![0-9])0{new.day}(?![0-9])",
        str(new.day),
        rendered,
        count=1,
    )


def render_date_surface(raw: str, old_iso: str, new_iso: str) -> str:
    old = date.fromisoformat(old_iso)
    new = date.fromisoformat(new_iso)
    compact = raw.strip()
    prefix_length = len(raw) - len(raw.lstrip())
    suffix_length = len(raw) - len(raw.rstrip())
    for format_string in _DATE_FORMATS:
        try:
            parsed = datetime.strptime(compact, format_string).date()
        except ValueError:
            continue
        if parsed == old:
            rendered = _case_like(compact, new.strftime(format_string))
            rendered = _preserve_unpadded_day(compact, rendered, old=old, new=new)
            suffix = raw[len(raw) - suffix_length :] if suffix_length else ""
            return raw[:prefix_length] + rendered + suffix
    raise ValueError(f"unsupported audited date surface: {raw!r}")


def _numeric_interpretations(token: str) -> list[tuple[Decimal, str | None, str | None, int]]:
    compact = token.replace(" ", "").replace("\u00a0", "").replace("'", "")
    separators = [character for character in compact if character in ".,"]
    candidates: list[tuple[str | None, str | None]] = [(None, None)]
    if separators:
        for decimal_separator in (".", ",", None):
            grouping_separator = None
            if decimal_separator is not None:
                other = "," if decimal_separator == "." else "."
                if other in compact:
                    grouping_separator = other
            elif len(set(separators)) == 1:
                grouping_separator = separators[0]
            candidates.append((decimal_separator, grouping_separator))
    output: list[tuple[Decimal, str | None, str | None, int]] = []
    for decimal_separator, grouping_separator in candidates:
        normalized = compact
        if grouping_separator is not None:
            normalized = normalized.replace(grouping_separator, "")
        decimal_places = 0
        if decimal_separator is not None:
            if normalized.count(decimal_separator) != 1:
                continue
            decimal_places = len(normalized.rsplit(decimal_separator, 1)[1])
            normalized = normalized.replace(decimal_separator, ".")
        elif "." in normalized or "," in normalized:
            continue
        try:
            value = Decimal(normalized)
        except InvalidOperation:
            continue
        candidate = (value, decimal_separator, grouping_separator, decimal_places)
        if candidate not in output:
            output.append(candidate)
    return output


def _group_digits(value: str, separator: str | None) -> str:
    if separator is None:
        return value
    sign = ""
    digits = value
    if digits.startswith(("+", "-")):
        sign, digits = digits[0], digits[1:]
    chunks = []
    while digits:
        chunks.append(digits[-3:])
        digits = digits[:-3]
    return sign + separator.join(reversed(chunks))


def render_number_surface(
    raw: str,
    old_value: int | float | Decimal,
    new_value: int | float | Decimal,
) -> str:
    old = Decimal(str(old_value))
    matches: list[tuple[re.Match[str], str | None, str | None, int]] = []
    for match in _NUMBER.finditer(raw):
        interpretations = _numeric_interpretations(match.group(0))
        for parsed, decimal_separator, grouping_separator, decimal_places in interpretations:
            if parsed == old:
                matches.append((match, decimal_separator, grouping_separator, decimal_places))
    unique = {
        (match.start(), match.end(), decimal, grouping, places)
        for match, decimal, grouping, places in matches
    }
    if len(unique) != 1:
        raise ValueError(f"audited numeric surface does not identify one source value: {raw!r}")
    start, end, decimal_separator, grouping_separator, decimal_places = next(iter(unique))
    quantum = Decimal(1).scaleb(-decimal_places)
    rendered_decimal = Decimal(str(new_value)).quantize(quantum)
    if rendered_decimal != Decimal(str(new_value)):
        raise ValueError("new numeric value cannot be represented in the audited source precision")
    plain = f"{rendered_decimal:.{decimal_places}f}"
    integer, _dot, fraction = plain.partition(".")
    integer = _group_digits(integer, grouping_separator)
    rendered = integer
    if decimal_places:
        rendered += cast(str, decimal_separator) + fraction
    return raw[:start] + rendered + raw[end:]


def _render_surface(change: SemanticChange, raw: str) -> str:
    if change.family == "document_date":
        return render_date_surface(raw, cast(str, change.old_value), cast(str, change.new_value))
    if change.family in {"package_quantity", "cargo_measure", "allocation_quantity"}:
        return render_number_surface(
            raw,
            cast(int | float, change.old_value),
            cast(int | float, change.new_value),
        )
    old = str(change.old_value)
    if raw != old:
        raise ValueError(
            f"identifier evidence surface differs from target value: {raw!r} != {old!r}"
        )
    return str(change.new_value)


def build_patch_plan(
    *,
    synthetic_document_id: str,
    source_raw_text_sha256: str,
    deterministic_target_sha256: str,
    changes: Sequence[SemanticChange],
    anchors: Sequence[Mapping[str, Any]],
) -> DeterministicTextPatchPlan:
    """Resolve all deterministic changes to immutable audited spans or fail explicitly."""

    anchors_by_path: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for row in anchors:
        anchors_by_path[cast(str, row["relation_target_path"])].append(row)
    planned: dict[tuple[int, int, int], dict[str, Any]] = {}
    unrendered: list[str] = []
    for change in changes:
        rows = anchors_by_path.get(change.target_path, [])
        if not rows or any(not bool(row.get("patchable")) for row in rows):
            unrendered.append(change.target_path)
            continue
        staged: dict[tuple[int, int, int], dict[str, Any]] = {}
        try:
            for row in rows:
                start = row.get("page_start")
                end = row.get("page_end")
                if not isinstance(start, int) or not isinstance(end, int):
                    raise ValueError("patchable anchor has no integer span")
                exact_old = cast(str, row["raw_value"])
                replacement = _render_surface(change, exact_old)
                key = (cast(int, row["page_number"]), start, end)
                existing = staged.get(key) or planned.get(key)
                if existing is not None:
                    if (
                        existing["exact_old_text"] != exact_old
                        or existing["replacement_text"] != replacement
                    ):
                        raise ValueError("semantic changes disagree on one OCR evidence span")
                    staged[key] = {
                        **existing,
                        "target_paths": set(existing["target_paths"]) | {change.target_path},
                    }
                else:
                    staged[key] = {
                        "page_number": key[0],
                        "page_start": start,
                        "page_end": end,
                        "exact_old_text": exact_old,
                        "replacement_text": replacement,
                        "target_paths": {change.target_path},
                        "family": change.family,
                        "coupling_group": change.coupling_group,
                    }
        except ValueError:
            unrendered.append(change.target_path)
            continue
        if not staged:
            unrendered.append(change.target_path)
            continue
        planned.update(staged)
    edits = []
    for key, row in sorted(planned.items()):
        row["target_paths"] = tuple(sorted(row["target_paths"]))
        row["edit_id"] = (
            "edit_"
            + sha256_bytes(
                canonical_json_bytes([synthetic_document_id, *key, row["replacement_text"]])
            )[:24]
        )
        edits.append(DeterministicTextEdit.model_validate(row, strict=True))
    return DeterministicTextPatchPlan.model_validate(
        {
            "schema_version": 1,
            "synthetic_document_id": synthetic_document_id,
            "source_raw_text_sha256": source_raw_text_sha256,
            "deterministic_target_sha256": deterministic_target_sha256,
            "edits": tuple(edits),
            "unrendered_change_paths": tuple(sorted(set(unrendered))),
            "planner": "deterministic_anchor_context_v1",
        },
        strict=True,
    )


def apply_patch_plan(joined_raw_text: str, plan: DeterministicTextPatchPlan) -> tuple[str, int]:
    """Apply disjoint page-local edits while preserving every untouched source byte."""

    spans = {span.number: span for span in _page_spans(joined_raw_text)}
    absolute_edits: list[tuple[int, int, DeterministicTextEdit]] = []
    for edit in plan.edits:
        page = spans.get(edit.page_number)
        if page is None:
            raise ValueError("patch edit references a missing page")
        start = page.body_start + edit.page_start
        end = page.body_start + edit.page_end
        if end > page.body_end or joined_raw_text[start:end] != edit.exact_old_text:
            raise ValueError("patch edit no longer matches the pinned OCR source")
        absolute_edits.append((start, end, edit))
    ordered = sorted(absolute_edits)
    for left, right in pairwise(ordered):
        if left[1] > right[0]:
            raise ValueError("absolute OCR edits overlap")
    output = joined_raw_text
    changed_source_characters = 0
    for start, end, edit in reversed(ordered):
        output = output[:start] + edit.replacement_text + output[end:]
        changed_source_characters += end - start
    return output, changed_source_characters
