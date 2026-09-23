"""Closed OCR row grammar: two measurement values followed by their two units.

This is positional evidence, not guessing a dimension from numeric magnitude.
UNECE Recommendation 20 defines KGM, LBR, MTQ and FTQ as kg, lb, m3 and ft3:
https://service.unece.org/trade/uncefact/vocabulary/rec20/
"""

from __future__ import annotations

import re

_VALUE = rb"[0-9]+(?:[.,][0-9]+)*"
_UNIT = rb"(?:KGM|KGS?|LBR|LBS?|MTQ|CBM|M3|FTQ|CBF|CFT|FT3)"
_ROW = re.compile(
    rb"(?<!\S)(?P<first>"
    + _VALUE
    + rb")[ \t]+(?P<second>"
    + _VALUE
    + rb")[ \t]+(?P<first_unit>"
    + _UNIT
    + rb")[ \t]+(?P<second_unit>"
    + _UNIT
    + rb")[ \t]*$",
    re.I,
)


def ordered_column_unit(source: bytes, *, byte_end: int, surface: str) -> str | None:
    """Associate only an exact numeric span within the complete closed row."""
    encoded = surface.encode()
    start = byte_end - len(encoded)
    if start < 0 or source[start:byte_end] != encoded:
        raise ValueError("measurement column span differs from source")
    left = source.rfind(b"\n", 0, start) + 1
    right = source.find(b"\n", byte_end)
    if right < 0:
        right = len(source)
    row = source[left:right].rstrip(b"\r")
    match = _ROW.search(row)
    if match is not None:
        for name in ("first", "second"):
            if match.span(name) == (start - left, byte_end - left):
                return match[name + "_unit"].decode().upper()
    return None
