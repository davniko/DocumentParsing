"""OCR-local evidence boundaries for bill-of-lading document dates."""

from __future__ import annotations

import re

_DECLARED_VALUE_LINE = re.compile(
    r"(?im)^[ \t]*(?:merchant['\u2019]s[ \t]+)?declared[ \t]+value\b[^\n]*"
)
_INLINE_CAPTION = re.compile(
    r"(?i)^[ \t]*(?:merchant['\u2019]s[ \t]+)?declared[ \t]+value"
    r"(?:[ \t]*\([^)]*\))?[ \t]*:?[ \t]*$"
)


def date_immediately_under_declared_value(text: str, start: int) -> bool:
    """Identify a date governed by a declared-value caption in linearized OCR.

    This intentionally recognizes only an adjacent line or inline value. A
    different printed heading between the caption and date ends its scope.
    """
    if not 0 <= start <= len(text):
        raise ValueError("date occurrence lies outside OCR text")
    for heading in _DECLARED_VALUE_LINE.finditer(text):
        if heading.start() > start:
            break
        if heading.start() <= start < heading.end():
            if _INLINE_CAPTION.fullmatch(text[heading.start() : start]):
                return True
        elif heading.end() <= start:
            between = text[heading.end() : start]
            if between.count("\n") <= 3 and not between.strip():
                return True
    return False
