"""Compile reviewed character owners into disjoint physical Paddle blocks."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

from document_ocr.spatial_inputs.paddle import PaddlePage, RegionSpan, project
from document_ocr.synthesis.paddle_reflow import PaddleBlock


@dataclass(frozen=True)
class PaddleReplacement:
    owner: str
    spans: tuple[RegionSpan, ...]
    text: str


def validate_existing_repetition(
    owners: tuple[PaddleReplacement, ...],
    expected: tuple[RegionSpan, ...],
    *,
    semantic_text: str | None = None,
) -> None:
    """Validate logical ownership before its physical frame partitioning.

    A split owner is still one sampled value. Its frame suffixes are presentation
    bookkeeping, not new semantic owners, and must not hide it from this guard.
    """
    wanted = {(s.page, s.region, s.start, s.end) for s in expected}
    got = {(s.page, s.region, s.start, s.end) for owner in owners for s in owner.spans}
    if got != wanted:
        raise ValueError("existing owner does not cover its reviewed native repetition")
    if semantic_text is not None and any(owner.text != semantic_text for owner in owners):
        raise ValueError("existing native owner disagrees with its semantic projection")


def partition_region_frames(
    replacement: PaddleReplacement,
    measure: Callable[[str], float],
    *,
    allow_empty_continuations: bool = False,
) -> tuple[PaddleReplacement, ...]:
    """Keep reviewed line frames separate without losing any generated words.

    The source's relative glyph widths determine only proposed word boundaries.
    Each resulting region still goes through ordinary measured fit and collision
    validation. This is explicit presentation, never a failed-envelope fallback.
    """
    spans = replacement.spans
    refs = [(s.page, s.region) for s in spans]
    if len(spans) < 2 or len(set(refs)) != len(refs) or refs != sorted(refs):
        raise ValueError("separate frames require distinct ordered native regions")
    if len({s.page for s in spans}) != 1:
        raise ValueError("separate frames cannot cross a page")
    words = replacement.text.split()
    if not words:
        raise ValueError("separate frames require nonempty generated text")
    if len(words) < len(spans) and not allow_empty_continuations:
        raise ValueError("insufficient words for the reviewed nonempty frames")
    weights = [measure(s.expected) for s in spans]
    if any(w <= 0 for w in weights):
        raise ValueError("separate frames need positive source glyph measures")
    if len(words) < len(spans):
        # An explicitly reviewed short realization may leave trailing source
        # frames unused. Keep every word and omit only its owned continuation.
        return tuple(
            PaddleReplacement(
                f"{replacement.owner}:frame:{i}",
                (span,),
                words[i] if i < len(words) else "",
            )
            for i, span in enumerate(spans)
        )
    cumulative = [measure(" ".join(words[:n])) for n in range(1, len(words) + 1)]
    output, start, used = [], 0, 0.0
    for i, span in enumerate(spans):
        used += weights[i]
        if i == len(spans) - 1:
            end = len(words)
        else:
            target = cumulative[-1] * used / sum(weights)
            last = len(words) - (len(spans) - i - 1)
            end = min(range(start + 1, last + 1), key=lambda n: abs(cumulative[n - 1] - target))
        output.append(
            PaddleReplacement(f"{replacement.owner}:frame:{i}", (span,), " ".join(words[start:end]))
        )
        start = end
    return tuple(output)


def project_replacements(
    pages: tuple[PaddlePage, ...],
    spans: tuple[RegionSpan, ...],
    replacements: tuple[PaddleReplacement, ...],
) -> str:
    """Read a reviewed logical field after edits, independently of active owners.

    This projects the same literal passage for single- and multi-row shipment
    recipes. It must not recover text from pre-anonymization or pre-render labels.
    Partial overlap with an edited owner cannot establish a field and is rejected.
    """
    if not spans:
        raise ValueError("literal target needs at least one reviewed fragment")
    project(pages, spans, separator="")
    selected = {(s.page, s.region, i) for s in spans for i in range(s.start, s.end)}
    if len(selected) != sum(s.end - s.start for s in spans):
        raise ValueError("literal target fragments overlap")
    edits, occupied = {}, set()
    for replacement in replacements:
        project(pages, replacement.spans, separator="")
        owned = {(s.page, s.region, i) for s in replacement.spans for i in range(s.start, s.end)}
        if occupied & owned:
            raise ValueError("overlapping replacement ownership")
        occupied.update(owned)
        if not owned & selected:
            continue
        if not owned <= selected:
            raise ValueError("literal target intersects only part of a private owner")
        for j, s in enumerate(replacement.spans):
            edits.setdefault((s.page, s.region), []).append(
                (s.start, s.end, replacement.text if j == 0 else "")
            )
    pieces = []
    for s in spans:
        value = pages[s.page].regions[s.region].text[s.start : s.end]
        for a, b, replacement in sorted(edits.get((s.page, s.region), []), reverse=True):
            if a < s.end and s.start < b:
                if not s.start <= a < b <= s.end:
                    raise ValueError("literal fragment splits a private owner")
                value = value[: a - s.start] + replacement + value[b - s.start :]
        pieces.append(value)
    return " ".join(" ".join(pieces).split())


def compile_blocks(
    pages: tuple[PaddlePage, ...],
    replacements: tuple[PaddleReplacement, ...],
    *,
    physical_groups: tuple[tuple[int, tuple[int, ...]], ...] = (),
) -> tuple[PaddleBlock, ...]:
    """Combine subfields sharing a physical region without overwriting neighbours.

    Multi-region owners are explicitly ordered. No textual matching or semantic
    allocation occurs here. Empty output is an explicit omission from the given
    replacement, never a fallback for a failed lookup.
    """
    if len({r.owner for r in replacements}) != len(replacements):
        raise ValueError("duplicate replacement owner")
    occupied, parents, edits = set(), {}, {}

    def find(ref):
        parents.setdefault(ref, ref)
        while parents[ref] != ref:
            parents[ref] = parents[parents[ref]]
            ref = parents[ref]
        return ref

    for replacement in replacements:
        project(pages, replacement.spans, separator="")  # checks exact proof
        if len({s.page for s in replacement.spans}) != 1:
            raise ValueError("a physical owner cannot cross a page boundary")
        refs = list(dict.fromkeys((s.page, s.region) for s in replacement.spans))
        if refs != sorted(refs):
            raise ValueError("owner fragments must retain native within-column order")
        root = find(refs[0])
        for ref in refs[1:]:
            parents[find(ref)] = root
        for n, span in enumerate(replacement.spans):
            positions = {(span.page, span.region, i) for i in range(span.start, span.end)}
            if occupied & positions:
                collision = min(occupied & positions)
                raise ValueError(
                    f"overlapping character ownership: {replacement.owner} at {collision}"
                )
            occupied.update(positions)
            edits.setdefault((span.page, span.region), []).append(
                (span.start, span.end, replacement.text if n == 0 else "")
            )
    for page, indices in physical_groups:
        if len(indices) < 2 or tuple(sorted(set(indices))) != indices:
            raise ValueError("physical group requires distinct ordered native regions")
        if not 0 <= page < len(pages) or any(
            not 0 <= i < len(pages[page].regions) for i in indices
        ):
            raise ValueError("physical group references an absent region")
        root = find((page, indices[0]))
        for i in indices[1:]:
            parents[find((page, i))] = root
    groups = {}
    for ref in parents:
        groups.setdefault(find(ref), []).append(ref)
    output = []
    for refs in sorted(groups.values()):
        refs.sort()
        fragments, expected = [], []
        for p, i in refs:
            source = pages[p].regions[i].text
            expected.append(source)
            for start, end, value in sorted(edits.get((p, i), []), reverse=True):
                source = source[:start] + value + source[end:]
            if source.strip():
                fragments.append(source)
        text = "\n".join(fragments)
        output.append(
            PaddleBlock(
                owner="regions:" + ":".join(f"{p}.{i}" for p, i in refs),
                page=refs[0][0],
                regions=tuple(i for _, i in refs),
                expected_texts=tuple(expected),
                text=text,
                action="replace" if text.strip() else "omit",
            )
        )
    return tuple(output)
