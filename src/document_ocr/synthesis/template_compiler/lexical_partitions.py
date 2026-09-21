"""Generate owned cargo fragments; never split a new item name by word weights.

The compiler's source intervals define identities, not widths in a new sentence.
Fragment values live in the already-hashed auxiliary context, outside training
labels. The complete target is assembled before it crosses the frozen boundary.
"""

from __future__ import annotations

import re
from collections.abc import Mapping
from dataclasses import dataclass
from functools import lru_cache
from itertools import pairwise
from typing import Any

from .models import CertifiedSemanticTemplate, SemanticBinding
from .realization_contract import token_projection_intervals


@dataclass(frozen=True)
class CargoPartition:
    binding: SemanticBinding
    regions: tuple[tuple[int, int], ...]
    intervals: tuple[tuple[int, int], ...]
    source_parts: tuple[str, ...]
    mutable: frozenset[int]
    owners: tuple[int, ...] = ()
    separators: tuple[str, ...] = ()

    def key(self, index: int) -> str:
        if self.owners:
            index = self.owners[index]
        return f"lexical-part:{self.binding.logical_key}:{index}"

    def pieces(self, auxiliary: Mapping[str, str]) -> tuple[str, ...]:
        result = tuple(
            auxiliary[self.key(i)] if i in self.mutable else part
            for i, part in enumerate(self.source_parts)
        )
        if any(
            not part.strip() or part != part.strip() or "\n" in part or "\r" in part
            for part in result
        ):
            raise ValueError("cargo fragments must be nonempty single-line values")
        return result

    def assemble(self, auxiliary: Mapping[str, str]) -> str:
        pieces = self.pieces(auxiliary)
        if self.separators:
            return self.separators[0] + "".join(
                piece + separator
                for piece, separator in zip(pieces, self.separators[1:], strict=True)
            )
        return " ".join(pieces)


def partition(binding: SemanticBinding) -> CargoPartition | None:
    from .descendant import _token_spans

    if not any(
        re.fullmatch(r"documentPatch\.cargoGroups\[\d+\]\.description", p)
        for p in binding.target_paths
    ):
        return None
    if binding.realization.mode not in {
        "segmented_surface",
        "token_projected_surface",
        "agent_required",
    }:
        return None
    originals = tuple(v.source_value for v in binding.realization.target_values)
    if (
        not originals
        or not isinstance(originals[0], str)
        or any(v != originals[0] for v in originals)
    ):
        if binding.realization.mode == "agent_required":
            return None
        raise ValueError("cargo partition lacks a unique source scalar")
    original = originals[0]
    tokens = _token_spans(original)
    if binding.realization.mode == "agent_required":
        return _repeated_fragment_partition(binding, original)
    intervals = token_projection_intervals(binding)
    if binding.realization.mode == "segmented_surface":
        rows = [_token_spans(slot.source_text) for slot in binding.occurrences]
        if "".join(token[0] for row in rows for token in row) != "".join(t[0] for t in tokens):
            raise ValueError("cargo segments do not prove an exact source-token partition")
        # OCR line wraps can split a word within a slot. Prove the entire
        # character sequence and require slot boundaries to remain token edges.
        token_edges = {0: 0}
        offset = 0
        for i, token in enumerate(tokens):
            offset += len(token[0])
            token_edges[offset] = i + 1
        bounds = [0]
        offset = 0
        for row in rows:
            offset += sum(len(token[0]) for token in row)
            if offset not in token_edges:
                raise ValueError("cargo source slot cuts across a semantic token boundary")
            bounds.append(token_edges[offset])
        intervals = tuple(pairwise(bounds))
    if not intervals or any(start >= end for start, end in intervals):
        raise ValueError("cargo partition has an empty source interval")
    boundaries = sorted({0, len(tokens), *(p for interval in intervals for p in interval)})
    regions = tuple(pairwise(boundaries))
    mutable = frozenset(
        i
        for i, (a, b) in enumerate(regions)
        if any(start <= a and b <= end for start, end in intervals)
    )
    if len(mutable) < 2:
        return None
    character_edges = {0: 0, len(tokens): len(original)}
    for edge in boundaries[1:-1]:
        offset = tokens[edge][1]
        before = original[:offset].rstrip()
        for slot, (start, _) in zip(binding.occurrences, intervals, strict=True):
            if start != edge:
                continue
            if slot.source_text.lstrip()[:1].isalnum():
                continue
            slot_tokens = _token_spans(slot.source_text)
            prefix = slot.source_text[: slot_tokens[0][1]].strip()
            if prefix and before.endswith(prefix):
                offset = min(offset, len(before) - len(prefix))
        character_edges[edge] = offset
    parts = tuple(original[character_edges[a] : character_edges[b]].strip() for a, b in regions)
    return CargoPartition(binding, regions, intervals, parts, mutable)


def _repeated_fragment_partition(binding: SemanticBinding, original: str) -> CargoPartition | None:
    """Prove a complete lexical cover, including repeated and reordered item parts.

    Ambiguous identical source phrases share one generated fragment. Partial
    overlaps, unowned words, and non-token-aligned fragments are not admitted.
    No meaning or missing ownership is inferred from document-specific names.
    """
    from .descendant import _token_spans

    tokens = _token_spans(original)
    words = tuple(t[0] for t in tokens)
    candidates = []
    for slot in binding.occurrences:
        phrase = tuple(t[0] for t in _token_spans(slot.source_text))
        matches = tuple(
            (i, i + len(phrase))
            for i in range(len(words) - len(phrase) + 1)
            if phrase and words[i : i + len(phrase)] == phrase
        )
        if not matches:
            return None
        candidates.append(matches)
    intervals = sorted({interval for matches in candidates for interval in matches})
    if (
        len(intervals) < 2
        or intervals[0][0] != 0
        or intervals[-1][1] != len(words)
        or any(left[1] > right[0] for left, right in pairwise(intervals))
    ):
        return None
    # Each region is independently printed, so label-only separators are retained
    # in assembly and removed from the corresponding physical fragment below.
    spans = [(tokens[a][1], tokens[b - 1][2]) for a, b in intervals]
    parts = tuple(original[a:b] for a, b in spans)
    separators = (
        original[: spans[0][0]],
        *(original[left[1] : right[0]] for left, right in pairwise(spans)),
        original[spans[-1][1] :],
    )
    # Non-punctuation gaps must be explicit label/value separators. Their exact
    # printed ownership is additionally proved against source bytes when used.
    if any(
        any(c.isalnum() for c in separator)
        and not re.fullmatch(r"[\s;,]*[A-Za-z][A-Za-z0-9 ./_-]*:\s*", separator)
        for separator in separators
    ):
        return None
    owners_by_phrase: dict[tuple[str, ...], int] = {}
    owners = tuple(owners_by_phrase.setdefault(words[a:b], i) for i, (a, b) in enumerate(intervals))
    # Tied phrases must have identical punctuation as well as identical tokens.
    if any(parts[i] != parts[owner] for i, owner in enumerate(owners)):
        return None
    return CargoPartition(
        binding,
        tuple(intervals),
        tuple(matches[0] for matches in candidates),
        parts,
        frozenset(range(len(parts))),
        owners,
        separators,
    )


def known_keys(template: CertifiedSemanticTemplate) -> set[str]:
    return {
        plan.key(i)
        for binding in template.bindings
        if (plan := partition(binding)) is not None
        for i in plan.mutable
    }


@lru_cache(maxsize=4096)
def _boundary_candidates(semantic: str, surface: str) -> tuple[str, str]:
    """Cache only immutable source grammar, not per-sample generated values."""
    from .descendant import _token_spans

    a, b = _token_spans(semantic), _token_spans(surface)
    if not a or not b:
        raise ValueError("cargo punctuation frame lacks lexical content")
    prefix, suffix = semantic[: a[0][1]].strip(), semantic[a[-1][2] :].strip()
    surface_prefix, surface_suffix = surface[: b[0][1]].strip(), surface[b[-1][2] :].strip()
    return (
        prefix[: len(prefix) - len(surface_prefix)] if prefix.endswith(surface_prefix) else "",
        suffix[len(surface_suffix) :] if suffix.startswith(surface_suffix) else "",
    )


def boundary_punctuation(plan: CargoPartition, index: int, source: bytes | None) -> tuple[str, str]:
    """Remove only punctuation proven present outside this physical source slot."""
    if plan.separators:
        # Label separators are separately owned by the exact assembly contract,
        # never requested from the model or injected into physical item slots.
        start, end = plan.intervals[index]
        region = plan.regions.index((start, end))
        label = tuple(re.findall(r"[A-Za-z0-9]+", plan.separators[region].casefold()))
        if label:
            if source is None:
                raise ValueError("labelled cargo fragments require pinned source bytes")
            slot = plan.binding.occurrences[index]
            label_before = source[
                source.rfind(b"\n", 0, slot.byte_start) + 1 : slot.byte_start
            ].decode()
            printed = tuple(re.findall(r"[A-Za-z0-9]+", label_before.casefold()))
            if printed[-len(label) :] != label:
                raise ValueError("cargo fragment label is not printed before its source slot")
        return "", ""
    start, end = plan.intervals[index]
    semantic = " ".join(
        p
        for (a, b), p in zip(plan.regions, plan.source_parts, strict=True)
        if start <= a and b <= end
    )
    slot = plan.binding.occurrences[index]
    prefix, suffix = _boundary_candidates(semantic, slot.source_text)
    if not prefix and not suffix:
        return "", ""
    if source is None:
        raise ValueError("cargo punctuation ownership requires pinned source bytes")
    before = source[: slot.byte_start].rstrip()
    after = source[slot.byte_end :].lstrip()
    return (
        prefix if prefix and before.endswith(prefix.encode()) else "",
        suffix if suffix and after.startswith(suffix.encode()) else "",
    )


def requests(plan: CargoPartition, source: bytes, *, key: str) -> tuple[dict[str, Any], ...]:
    result: list[dict[str, Any]] = []
    for i in sorted(plan.mutable):
        if plan.owners and plan.owners[i] != i:
            continue
        a, b = plan.regions[i]
        contexts = []
        minimum_words = 1
        frames: set[tuple[str, str]] = set()
        for slot_index, (slot, (start, end)) in enumerate(
            zip(plan.binding.occurrences, plan.intervals, strict=True)
        ):
            if not (start <= a and b <= end):
                continue
            before = source[source.rfind(b"\n", 0, slot.byte_start) + 1 : slot.byte_start].decode()
            newline = source.find(b"\n", slot.byte_end)
            after = source[slot.byte_end : len(source) if newline == -1 else newline].decode()
            context = {"before": before, "sourceSlot": slot.source_text, "after": after}
            if context not in contexts:
                contexts.append(context)
            if (a, b) == (start, end):
                minimum_words = max(minimum_words, len(slot.format_envelope.newline_sequence) + 1)
            prefix, suffix = boundary_punctuation(plan, slot_index, source)
            prefix = prefix if a == start else ""
            suffix = suffix if b == end else ""
            if prefix or suffix:
                frames.add((prefix, suffix))
        result.append(
            {
                "key": f"{key}_part_{i}",
                "paths": [],
                "auxiliaryKey": plan.key(i),
                "source": plan.source_parts[i],
                "constraints": [{"minimumWords": minimum_words}],
                "cargoFragment": {
                    "targetPaths": list(plan.binding.target_paths),
                    "index": i,
                    "sourceParts": list(plan.source_parts),
                    "contexts": contexts,
                },
                "instructions": (
                    "Generate only this cargo fragment, keeping its source role: a complete "
                    "item name when the source is an item, or its continuation/model details "
                    "otherwise. Do not move names or model details between fragments. Related "
                    "fragments describe one coherent shipment. Quantities outside this slot "
                    "are host-owned; do not duplicate them."
                ),
            }
        )
        if frames:
            if len(frames) != 1:
                raise ValueError("repeated cargo fragments have conflicting punctuation frames")
            prefix, suffix = next(iter(frames))
            result[-1]["requiredBoundaryPunctuation"] = dict(prefix=prefix, suffix=suffix)
            result[-1]["instructions"] += (
                " Include requiredBoundaryPunctuation exactly once at the value edges. "
                "It belongs to the full label; the host already owns its printed static bytes."
            )
    return tuple(result)


def assembled_targets(
    template: CertifiedSemanticTemplate, auxiliary: Mapping[str, str]
) -> dict[str, str]:
    values: dict[str, str] = {}
    for binding in template.bindings:
        plan = partition(binding)
        if plan is None:
            continue
        missing = {plan.key(i) for i in plan.mutable} - auxiliary.keys()
        if missing:
            raise ValueError("missing generated cargo fragments: " + ", ".join(sorted(missing)))
        text = plan.assemble(auxiliary)
        for path in binding.target_paths:
            if values.setdefault(path, text) != text:
                raise ValueError("cargo fragment owners disagree on a target")
    return values


def render_parts(
    plan: CargoPartition, value: str, auxiliary: Mapping[str, str], *, source: bytes | None = None
) -> dict[str, str]:
    from .descendant import _layout_like_source

    try:
        pieces = plan.pieces(auxiliary)
    except KeyError as error:
        raise ValueError(
            "generated cargo fragment context is required; weighted splitting is forbidden"
        ) from error
    if plan.assemble(auxiliary) != value:
        raise ValueError("cargo fragment context differs from the frozen target")
    result = {}
    for index, (slot, (start, end)) in enumerate(
        zip(plan.binding.occurrences, plan.intervals, strict=True)
    ):
        text = " ".join(
            pieces[i] for i, (a, b) in enumerate(plan.regions) if start <= a and b <= end
        )
        prefix, suffix = boundary_punctuation(plan, index, source)
        if (prefix and not text.startswith(prefix)) or (suffix and not text.endswith(suffix)):
            raise ValueError(
                "generated cargo fragment lost its required static boundary punctuation"
            )
        text = text[len(prefix) : len(text) - len(suffix) if suffix else len(text)].strip()
        result[slot.slot_id] = _layout_like_source(slot.source_text, text)
    return result
