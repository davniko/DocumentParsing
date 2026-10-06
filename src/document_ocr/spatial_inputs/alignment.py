"""Conservative whole-region alignment of two OCR readings of the same page.

No gold labels, language model, approximate string replacement, global reading-order
assumption, or fabricated word geometry is involved. Unresolved lines are explicit.
"""

from __future__ import annotations

import math
import re
import unicodedata
from collections import Counter, defaultdict
from dataclasses import dataclass

from pydantic import BaseModel, ConfigDict, Field


class AlignmentPolicy(BaseModel):
    """Geometry tolerances in line heights, not document-specific pixel distances."""

    model_config = ConfigDict(extra="forbid", frozen=True)
    grid_size: int = Field(ge=1)
    horizontal_gap_heights: float = Field(gt=0)
    vertical_gap_heights: float = Field(gt=0)
    vertical_overlap_tolerance: float = Field(ge=0, le=1)
    context_lines: int = Field(ge=1)
    context_radius_heights: float = Field(gt=0)
    context_margin_heights: float = Field(gt=0)
    left_alignment_weight: float = Field(ge=0)
    max_group_regions: int = Field(ge=2)
    max_search_states: int = Field(ge=1)


@dataclass(frozen=True)
class Region:
    index: int
    text: str
    box: tuple[float, float, float, float]
    score: float

    @property
    def height(self) -> float:
        return self.box[3] - self.box[1]


@dataclass(frozen=True)
class Line:
    number: int
    page: int
    block: int
    text: str


@dataclass(frozen=True)
class Match:
    method: str
    regions: tuple[int, ...] = ()
    anchors: tuple[int, ...] = ()


PAGE_MARKER = re.compile(r"--- PAGE ([1-9][0-9]*) ---")
SUFFIX = re.compile(r" \|\|(?: [0-9]+,[0-9]+)?\Z")


def folded(text: str) -> str:
    return "".join(unicodedata.normalize("NFKC", text).casefold().split())


def text_key(text: str) -> str:
    return "".join(c for c in folded(text) if c.isalnum())


def parse_lines(text: str, page_count: int) -> list[Line]:
    """Preserve original line numbers; blank lines delimit local context blocks."""
    page = -1
    block = 0
    markers = []
    result = []
    for number, line in enumerate(text.splitlines(), 1):
        if marker := PAGE_MARKER.fullmatch(line.strip()):
            page = int(marker[1]) - 1
            markers.append(page)
            block += 1
        elif not line.strip():
            block += 1
        else:
            if page < 0:
                raise ValueError("OCR content precedes its page marker")
            if SUFFIX.search(line):
                raise ValueError("OCR already contains a positional suffix")
            result.append(Line(number, page, block, line))
    if markers != list(range(page_count)):
        raise ValueError(f"OCR/PDF page mismatch: {markers}, expected {page_count} pages")
    return result


def _neighbors(a: Region, b: Region, policy: AlignmentPolicy) -> bool:
    h = max(a.height, b.height)
    ax1, ay1, ax2, ay2 = a.box
    bx1, by1, bx2, by2 = b.box
    same_row = min(ay2, by2) > max(ay1, by1)
    same_column = min(ax2, bx2) > max(ax1, bx1)
    right = same_row and bx1 >= ax2 - h and bx1 - ax2 <= policy.horizontal_gap_heights * h
    below = (
        same_column
        and by1 >= ay2 - policy.vertical_overlap_tolerance * h
        and by1 - ay2 <= policy.vertical_gap_heights * h
    )
    return right or below


def _group_paths(
    key: str, regions: list[Region], keys: list[str], policy: AlignmentPolicy
) -> tuple[list[tuple[int, ...]], bool]:
    """Find at most two alternatives; hitting a search cap explicitly prevents acceptance."""
    paths: list[tuple[int, ...]] = []
    states = 0
    limited = False

    def walk(prefix: str, indices: tuple[int, ...]) -> None:
        nonlocal states, limited
        if len(paths) >= 2 or limited:
            return
        states += 1
        if states > policy.max_search_states:
            limited = True
            return
        if prefix == key:
            paths.append(indices)
            return
        if len(indices) >= policy.max_group_regions:
            limited = True
            return
        for j, candidate in enumerate(regions):
            if j in indices or not keys[j] or not key.startswith(prefix + keys[j]):
                continue
            if _neighbors(regions[indices[-1]], candidate, policy):
                walk(prefix + keys[j], (*indices, j))

    for i, k in enumerate(keys):
        if k and len(k) < len(key) and key.startswith(k):
            walk(k, (i,))
    return paths, limited


def _reject_collisions(matches: list[Match]) -> list[Match]:
    claims: dict[int, list[int]] = defaultdict(list)
    for n, match in enumerate(matches):
        for i in match.regions:
            claims[i].append(n)
    conflicts = {n for owners in claims.values() if len(owners) > 1 for n in owners}
    return [
        Match("conflicting_region_claims") if n in conflicts else m for n, m in enumerate(matches)
    ]


def _context_match(
    n: int,
    candidates: list[int],
    lines: list[Line],
    matches: list[Match],
    regions: list[Region],
    policy: AlignmentPolicy,
) -> Match:
    votes: list[tuple[int, int]] = []
    for direction in (-1, 1):
        for gap in range(1, policy.context_lines + 1):
            k = n + direction * gap
            if not 0 <= k < len(lines) or lines[k].block != lines[n].block:
                break
            # Context never bootstraps from another inferred/contextual match.
            if matches[k].method != "exact_unique":
                continue
            a = regions[matches[k].regions[0]]
            ranked = []
            for i in candidates:
                b = regions[i]
                h = max(a.height, b.height)
                dx = max(0, max(a.box[0], b.box[0]) - min(a.box[2], b.box[2]))
                dy = max(0, max(a.box[1], b.box[1]) - min(a.box[3], b.box[3]))
                distance = (
                    math.hypot(dx, dy) + policy.left_alignment_weight * abs(a.box[0] - b.box[0])
                ) / h
                ranked.append((distance, i))
            ranked.sort()
            if (
                ranked[0][0] <= policy.context_radius_heights
                and ranked[1][0] - ranked[0][0] >= policy.context_margin_heights
            ):
                votes.append((ranked[0][1], k))
            break
    if votes and len({i for i, _ in votes}) == 1:
        return Match("repeated_context", (votes[0][0],), tuple(k for _, k in votes))
    return Match("ambiguous_repeated_text")


def align_page(
    lines: list[Line], regions: list[Region], width: int, height: int, policy: AlignmentPolicy
) -> list[Match]:
    """Match complete lines/regions; never assign a containing box to a substring."""
    if width <= 0 or height <= 0:
        raise ValueError("invalid page dimensions")
    if len({line.page for line in lines}) > 1:
        raise ValueError("alignment must be page-local")
    for i, r in enumerate(regions):
        x1, y1, x2, y2 = r.box
        if r.index != i or not (0 <= x1 < x2 <= width and 0 <= y1 < y2 <= height):
            raise ValueError("invalid region order or geometry")
        if not math.isfinite(r.score) or not 0 <= r.score <= 1:
            raise ValueError("invalid recognition confidence")
    exact: dict[str, list[int]] = defaultdict(list)
    loose: dict[str, list[int]] = defaultdict(list)
    keys = [text_key(r.text) for r in regions]
    for i, r in enumerate(regions):
        exact[folded(r.text)].append(i)
        if keys[i]:
            loose[keys[i]].append(i)
    source_exact = Counter(folded(line.text) for line in lines)
    source_loose = Counter(text_key(line.text) for line in lines)
    matches = []
    repeated = {}
    for n, line in enumerate(lines):
        key = text_key(line.text)
        strict = folded(line.text)
        ids = exact.get(strict, [])
        if not key:
            match = Match("no_alphanumeric_text")
        elif ids:
            if len(ids) == 1 and source_exact[strict] == 1:
                match = Match("exact_unique", (ids[0],))
            else:
                match = Match("ambiguous_repeated_text")
                repeated[n] = ids
        elif ids := loose.get(key, []):
            if len(ids) == 1 and source_loose[key] == 1:
                match = Match("punctuation_unique", (ids[0],))
            else:
                match = Match("ambiguous_repeated_text")
                repeated[n] = ids
        else:
            paths, limited = _group_paths(key, regions, keys, policy)
            if limited:
                match = Match("group_search_limit")
            elif len(paths) == 1 and source_loose[key] == 1:
                match = Match("spatial_group", paths[0])
            elif paths:
                match = Match("ambiguous_group")
            elif any(key in k for k in keys):
                match = Match("only_partial_region")
            else:
                match = Match("no_complete_text_match")
        matches.append(match)
    matches = _reject_collisions(matches)
    anchored = list(matches)
    for n, candidates in repeated.items():
        if len(candidates) > 1:
            matches[n] = _context_match(n, candidates, lines, anchored, regions, policy)
    matches = _reject_collisions(matches)
    # A contextual coordinate loses authorization if its anchor lost a collision.
    return [
        Match("context_anchor_conflict") if any(not matches[k].regions for k in m.anchors) else m
        for m in matches
    ]


def enclosing_box(regions: list[Region], match: Match) -> tuple[float, float, float, float]:
    if not match.regions:
        raise ValueError("unresolved line has no box")
    boxes = [regions[i].box for i in match.regions]
    return (
        min(b[0] for b in boxes),
        min(b[1] for b in boxes),
        max(b[2] for b in boxes),
        max(b[3] for b in boxes),
    )


def centroid(
    box: tuple[float, float, float, float], width: int, height: int, grid: int
) -> tuple[int, int]:
    return round(grid * (box[0] + box[2]) / (2 * width)), round(
        grid * (box[1] + box[3]) / (2 * height)
    )


def enrich(text: str, coordinates: dict[int, tuple[int, int] | None]) -> str:
    """Append a suffix without altering even the source's whitespace or line endings."""
    result = []
    for n, line in enumerate(text.splitlines(keepends=True), 1):
        if n in coordinates:
            body = line.rstrip("\r\n")
            ending = line[len(body) :]
            xy = coordinates[n]
            suffix = " ||" if xy is None else f" || {xy[0]},{xy[1]}"
            result.append(body + suffix + ending)
        else:
            result.append(line)
    return "".join(result)


def verify_preservation(original: str, augmented: str, expected_lines: set[int]) -> None:
    source = original.splitlines(keepends=True)
    result = augmented.splitlines(keepends=True)
    if len(source) != len(result):
        raise ValueError("line count changed")
    for n, (before, after) in enumerate(zip(source, result, strict=True), 1):
        if n not in expected_lines:
            if before != after:
                raise ValueError("structural line changed")
            continue
        body = before.rstrip("\r\n")
        ending = before[len(body) :]
        suffix = after[len(body) : len(after) - len(ending) if ending else len(after)]
        if not after.startswith(body) or not after.endswith(ending) or not SUFFIX.fullmatch(suffix):
            raise ValueError(f"unexpected text edit on line {n}")
