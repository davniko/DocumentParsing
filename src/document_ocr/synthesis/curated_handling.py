"""Literal handling evidence, including explicitly reviewed separated fields.

This is a validation gate, not a semantic classifier or target repair fallback.
Source targets and ownership decide which clauses are handling. Rendering must
preserve that decision and cannot paraphrase the extracted text.
"""

from __future__ import annotations

import re
from collections.abc import Mapping, Sequence
from typing import Any

from document_ocr.synthesis.curated import flat
from document_ocr.synthesis.curated_templates import SamplingBlueprint

_PATH = re.compile(r"documentPatch\.goodsItemDetails\[\d+\]\.handlingInstructions\[\d+\]")


def _normalized(text: str) -> str:
    return " ".join(text.split()).casefold()


def validate_rendered_handling(
    blueprint: SamplingBlueprint,
    rendered: str,
    target: dict[str, Any],
    edits: Sequence[Mapping[str, Any]],
) -> None:
    """Require literal final-text evidence; never drop or rewrite a failed label."""
    fields = {path: value for path, value in flat(target).items() if _PATH.fullmatch(path)}
    joins = blueprint.ownership_data.get("handling_fragments", {})
    if not isinstance(joins, dict) or not set(joins) <= fields.keys():
        raise ValueError("handling fragments must name existing handling fields")
    if not fields:
        return
    normalized = _normalized(rendered)
    source, final = blueprint.source.encode(), rendered.encode()
    for path, value in fields.items():
        if path not in joins:
            if _normalized(value) not in normalized:
                raise ValueError(f"handling target lacks literal rendered evidence: {path}")
            continue
        selectors = joins[path]
        if not isinstance(selectors, list) or len(selectors) < 2:
            raise ValueError("separated handling needs at least two reviewed fragments")
        fragments, previous_end = [], 0
        for selector in selectors:
            if set(selector) != {"text", "occurrence"} or type(selector["occurrence"]) is not int:
                raise ValueError("invalid handling fragment selector")
            text, occurrence = selector["text"], selector["occurrence"]
            matches = list(re.finditer(re.escape(text.encode()), source))
            if not text or not 1 <= occurrence <= len(matches):
                raise ValueError("handling fragment occurrence is absent")
            match = matches[occurrence - 1]
            start, end = match.span()
            if start < previous_end:
                raise ValueError("handling fragments overlap or are out of source order")
            previous_end = end
            chunks, cursor, shift = [], start, 0
            for edit in edits:
                left, right = edit["byteStart"], edit["byteEnd"]
                replacement = edit["after"].encode()
                if right <= start:
                    shift += len(replacement) - (right - left)
                elif left < end:
                    if not cursor <= left < right <= end:
                        raise ValueError("render edit crosses reviewed handling fragment")
                    if source[left:right].decode() != edit["before"]:
                        raise ValueError("handling fragment edit differs from source")
                    chunks.extend((source[cursor:left], replacement))
                    cursor = right
            chunks.append(source[cursor:end])
            fragment = b"".join(chunks)
            if final[start + shift : start + shift + len(fragment)] != fragment:
                raise ValueError("handling fragment differs from final rendered bytes")
            fragments.append(fragment.decode())
        if _normalized(" ".join(fragments)) != _normalized(value):
            raise ValueError(f"handling target differs from reviewed fragment projection: {path}")
