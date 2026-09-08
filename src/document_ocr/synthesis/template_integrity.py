"""Deterministic source-template integrity checks for synthetic B/L rendering."""

from __future__ import annotations

import re
from collections.abc import Mapping, Sequence
from typing import Any, cast

_CARRIER_RECEIPT_HEADING = re.compile(
    r"\bCARRIER['\N{RIGHT SINGLE QUOTATION MARK}]?S[ \t]+RECEIPT\b",
    re.IGNORECASE,
)
_SAME_LINE_CONTAINER_COUNT = re.compile(
    r"\b(?P<count>[0-9][0-9,]*)[ \t]+(?:CNTRS?|CONTAINER\(S\)|CONTAINERS?)\b",
    re.IGNORECASE,
)
_STANDALONE_CONTAINER_COUNT = re.compile(
    r"^[ \t]*(?P<count>[0-9][0-9,]*)[ \t]+"
    r"(?:CNTRS?|CONTAINER\(S\)|CONTAINERS?)[ \t]*$",
    re.IGNORECASE,
)


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
        heading = _CARRIER_RECEIPT_HEADING.search(line)
        if heading is None:
            continue
        same_line = _SAME_LINE_CONTAINER_COUNT.search(line[heading.end() :])
        if same_line is not None:
            counts.append(int(same_line.group("count").replace(",", "")))
            continue
        for following in lines[index + 1 : index + 4]:
            if not following.strip():
                continue
            match = _STANDALONE_CONTAINER_COUNT.fullmatch(following)
            if match is not None:
                counts.append(int(match.group("count").replace(",", "")))
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
            for row in cast(Sequence[Any], containers)
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
