"""Deterministic source-template integrity checks for synthetic B/L rendering."""

from __future__ import annotations

import re
from collections.abc import Mapping, Sequence
from typing import Any

_CARRIER_RECEIPT_HEADING = re.compile(
    r"\bCARRIER['\N{RIGHT SINGLE QUOTATION MARK}]?S[ \t]+RECEIPT\b",
    re.IGNORECASE,
)
_EXPLICIT_RECEIVED_COUNT = re.compile(
    r"^\s*Total\s+(?:No\.?|Number)\s+of\s+Containers\s+received\s+by\s+(?:the\s+)?Carrier\s*:\s*(?P<count>[0-9]+)\s*$",
    re.IGNORECASE,
)
_WEIGHT_TOTAL_CONTAINER_FIELD = re.compile(r"^\s*Weight\s+in\s+Kgs\s+Total\s*:", re.IGNORECASE)
_SAME_LINE_CONTAINER_COUNT = re.compile(
    r"(?:\(|\b)(?P<count>[0-9][0-9,]*|"
    r"(?:ZERO|ONE|TWO|THREE|FOUR|FIVE|SIX|SEVEN|EIGHT|NINE|TEN|ELEVEN|TWELVE|"
    r"THIRTEEN|FOURTEEN|FIFTEEN|SIXTEEN|SEVENTEEN|EIGHTEEN|NINETEEN|TWENTY|"
    r"THIRTY|FORTY|FIFTY|SIXTY|SEVENTY|EIGHTY|NINETY|HUNDRED|THOUSAND|AND)"
    r"(?:[ -]+(?:ZERO|ONE|TWO|THREE|FOUR|FIVE|SIX|SEVEN|EIGHT|NINE|TEN|ELEVEN|"
    r"TWELVE|THIRTEEN|FOURTEEN|FIFTEEN|SIXTEEN|SEVENTEEN|EIGHTEEN|NINETEEN|"
    r"TWENTY|THIRTY|FORTY|FIFTY|SIXTY|SEVENTY|EIGHTY|NINETY|HUNDRED|THOUSAND|"
    r"AND))*)\)?[ \t]+(?:CNTRS?|CONTAINER\(S\)|CONTAINERS?)\b",
    re.IGNORECASE,
)

_SMALL_CARDINALS = {
    word: value
    for value, word in enumerate(
        (
            "ZERO",
            "ONE",
            "TWO",
            "THREE",
            "FOUR",
            "FIVE",
            "SIX",
            "SEVEN",
            "EIGHT",
            "NINE",
            "TEN",
            "ELEVEN",
            "TWELVE",
            "THIRTEEN",
            "FOURTEEN",
            "FIFTEEN",
            "SIXTEEN",
            "SEVENTEEN",
            "EIGHTEEN",
            "NINETEEN",
        )
    )
}
_TENS_CARDINALS = {
    "TWENTY": 20,
    "THIRTY": 30,
    "FORTY": 40,
    "FIFTY": 50,
    "SIXTY": 60,
    "SEVENTY": 70,
    "EIGHTY": 80,
    "NINETY": 90,
}


def _english_cardinal(value: str) -> int | None:
    compact = value.replace(",", "").strip()
    if compact.isdigit():
        return int(compact)
    tokens = compact.upper().replace("-", " ").split()
    if not tokens:
        return None
    total = 0
    group = 0
    previous = "start"
    seen_hundred = False
    seen_thousand = False
    for token in tokens:
        if token == "AND":
            if previous in {"start", "and"}:
                return None
            previous = "and"
            continue
        if token in _SMALL_CARDINALS:
            value_part = _SMALL_CARDINALS[token]
            if previous in {"small", "hundred"} and not (previous == "hundred" and value_part < 20):
                return None
            if previous == "tens" and value_part >= 10:
                return None
            group += value_part
            previous = "small"
            continue
        if token in _TENS_CARDINALS:
            if previous in {"small", "tens"}:
                return None
            group += _TENS_CARDINALS[token]
            previous = "tens"
            continue
        if token == "HUNDRED":
            if seen_hundred or previous != "small" or not 1 <= group <= 9:
                return None
            group *= 100
            seen_hundred = True
            previous = "hundred"
            continue
        if token == "THOUSAND":
            if seen_thousand or group <= 0 or previous == "and":
                return None
            total += group * 1000
            group = 0
            seen_hundred = False
            seen_thousand = True
            previous = "thousand"
            continue
        return None
    if previous == "and":
        return None
    return total + group


def _container_count_in(value: str) -> int | None:
    for match in _SAME_LINE_CONTAINER_COUNT.finditer(value):
        parsed = _english_cardinal(match.group("count"))
        if parsed is not None:
            return parsed
    return None


def explicit_carrier_receipt_container_counts(raw_text: str) -> tuple[int, ...]:
    """Return counts printed in an explicit carrier-receipt field.

    The parser deliberately does not infer from cargo prose. It accepts a count on the carrier-
    receipt heading itself, or on the first non-empty physical line immediately following that
    heading. This is sufficient to prove an unusable template while avoiding generic mentions of
    containers elsewhere in the document.
    """

    lines = raw_text.splitlines()
    counts: list[int] = []
    for index, line in enumerate(lines):
        explicit = _EXPLICIT_RECEIVED_COUNT.fullmatch(line)
        if explicit:
            counts.append(int(explicit.group("count")))
            continue
        weight_total = _WEIGHT_TOTAL_CONTAINER_FIELD.match(line)
        if weight_total:
            count = _container_count_in(line[weight_total.end() :])
            if count is not None:
                counts.append(count)
            continue
        heading = _CARRIER_RECEIPT_HEADING.search(line)
        if heading is None:
            continue
        same_line = _container_count_in(line[heading.end() :])
        if same_line is not None:
            counts.append(same_line)
            continue
        for following in lines[index + 1 : index + 4]:
            if not following.strip():
                continue
            following_count = _container_count_in(following)
            if following_count is not None:
                counts.append(following_count)
            break
    return tuple(counts)


def source_template_integrity_issues(
    raw_text: str,
    target: Mapping[str, Any],
) -> tuple[str, ...]:
    """Report only proven source/label topology contradictions relevant to synthesis."""

    patch = target.get("documentPatch")
    if not isinstance(patch, Mapping):
        return ()
    containers = patch.get("containers")
    if not isinstance(containers, Sequence) or isinstance(containers, (str, bytes)):
        labeled_count = 0
    else:
        labeled_count = sum(
            isinstance(row, Mapping) and isinstance(row.get("containerNumber"), str)
            for row in containers
        )
    printed_counts = explicit_carrier_receipt_container_counts(raw_text)
    conflicting = sorted({count for count in printed_counts if count != labeled_count})
    if not conflicting:
        return ()
    return (
        "explicit_carrier_receipt_container_count_differs_from_labeled_containers:"
        + ",".join(str(value) for value in conflicting)
        + f"_vs_{labeled_count}",
    )
