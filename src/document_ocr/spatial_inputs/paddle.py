"""Paddle-only input contract: measured regions, native order, one geometry view.

Text is never borrowed from another OCR engine. Synthetic geometry is represented
by the same full-precision rectangles, but must carry its separate provenance.
"""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass
from typing import Literal


@dataclass(frozen=True)
class PaddleRegion:
    index: int
    text: str
    box: tuple[float, float, float, float]


@dataclass(frozen=True)
class PaddlePage:
    index: int
    width: float
    height: float
    regions: tuple[PaddleRegion, ...]


def pages_from_record(record: dict) -> tuple[PaddlePage, ...]:
    """Load explicit measured/rendered frames without reordering or filling fields."""
    return tuple(
        PaddlePage(
            p["index"],
            p["width"],
            p["height"],
            tuple(PaddleRegion(r["index"], r["text"], tuple(r["box"])) for r in p["regions"]),
        )
        for p in record["pages"]
    )


def validate(pages: tuple[PaddlePage, ...]) -> None:
    if not pages:
        raise ValueError("a document requires page frames, including empty pages")
    for p, page in enumerate(pages):
        if type(page.index) is not int or page.index != p:
            raise ValueError("pages must retain contiguous native order")
        if not all(math.isfinite(v) and v > 0 for v in (page.width, page.height)):
            raise ValueError("invalid page dimensions")
        for i, region in enumerate(page.regions):
            if type(region.index) is not int or region.index != i:
                raise ValueError("regions must retain contiguous native order")
            # The pinned Paddle run contains measured regions with empty/space
            # recognition. Retain them, including their indices and geometry;
            # they supply no text evidence and cannot satisfy a target span.
            if not isinstance(region.text, str):
                raise ValueError("region text must be a string")
            if "\n" in region.text or "\r" in region.text:
                raise ValueError("one native recognized region must occupy one input line")
            if len(region.box) != 4 or not all(math.isfinite(v) for v in region.box):
                raise ValueError("invalid region rectangle")
            x1, y1, x2, y2 = region.box
            if not 0 <= x1 < x2 <= page.width or not 0 <= y1 < y2 <= page.height:
                raise ValueError("region outside page or empty rectangle")


def serialize(
    pages: tuple[PaddlePage, ...], *, geometry: Literal["centroid", "bbox", "none"] = "centroid"
) -> str:
    """Round only at serialization, in a normalized 0..1000 page frame.

    Native region order is authoritative; sorting flattened text into semantic blocks
    would hide the layout the model actually receives at inference.
    """
    validate(pages)
    if geometry not in {"centroid", "bbox", "none"}:
        raise ValueError("unknown geometry view")
    output = []
    for page in pages:
        lines = []
        for region in page.regions:
            x1, y1, x2, y2 = region.box
            if geometry == "centroid":
                values = ((x1 + x2) * 500 / page.width, (y1 + y2) * 500 / page.height)
            else:
                values = (
                    x1 * 1000 / page.width,
                    y1 * 1000 / page.height,
                    x2 * 1000 / page.width,
                    y2 * 1000 / page.height,
                )
            suffix = "" if geometry == "none" else " || " + ",".join(str(round(v)) for v in values)
            lines.append(region.text + suffix)
        output.append(f"--- PAGE {page.index + 1} ---\n" + "\n".join(lines))
    return "\n\n".join(output)


def fingerprint(pages: tuple[PaddlePage, ...]) -> str:
    validate(pages)
    payload = [
        (p.index, p.width, p.height, [(r.index, r.text, r.box) for r in p.regions]) for p in pages
    ]
    return hashlib.sha256(
        json.dumps(payload, ensure_ascii=False, separators=(",", ":")).encode()
    ).hexdigest()


@dataclass(frozen=True)
class RegionSpan:
    page: int
    region: int
    start: int
    end: int
    expected: str


def project(pages: tuple[PaddlePage, ...], spans: tuple[RegionSpan, ...], *, separator: str) -> str:
    """Project explicitly reviewed owners, including interleaved column fragments.

    No fuzzy matching, punctuation repair, automatic sorting, or implicit separator.
    Caller-owned policy decides whether joining means a space or no separator (IDs).
    """
    validate(pages)
    if not spans:
        raise ValueError("an evidence projection requires nonempty spans")
    seen: set[tuple[int, int, int]] = set()
    parts = []
    for span in spans:
        if not 0 <= span.page < len(pages):
            raise ValueError("evidence page absent")
        regions = pages[span.page].regions
        if not 0 <= span.region < len(regions):
            raise ValueError("evidence region absent")
        text = regions[span.region].text
        if (
            not 0 <= span.start < span.end <= len(text)
            or text[span.start : span.end] != span.expected
        ):
            raise ValueError("evidence span differs from pinned OCR")
        positions = {(span.page, span.region, n) for n in range(span.start, span.end)}
        if seen & positions:
            raise ValueError("overlapping evidence in one projection")
        seen.update(positions)
        parts.append(span.expected)
    return separator.join(parts)
