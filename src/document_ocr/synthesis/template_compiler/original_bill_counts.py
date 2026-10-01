"""Source-pinned private original-bill counts, distinct from negotiability."""

from __future__ import annotations

import re

from .models import SemanticBinding

_WORDS = {
    "ZERO": 0,
    "ONE": 1,
    "TWO": 2,
    "THREE": 3,
    "FOUR": 4,
    "FIVE": 5,
    "SIX": 6,
    "SEVEN": 7,
    "EIGHT": 8,
    "NINE": 9,
}
_COUNT = re.compile(r"(?i)(ZERO|ONE|TWO|THREE|FOUR|FIVE|SIX|SEVEN|EIGHT|NINE)\s*\(\s*([0-9])\s*\)")
_CAPTION = re.compile(r"(?i)^\s*(?:NUMBER|NO\.?)(?:\s+OF)?\s+ORIGINAL\s+B\s*\(?S\)?\s*/\s*L\s*$")


def parse_count(text: str) -> int:
    match = _COUNT.fullmatch(text)
    if match is None or _WORDS[match[1].upper()] != int(match[2]):
        raise ValueError("original-bill count lacks a consistent word-and-digit surface")
    return int(match[2])


def _source_caption_proves(source: bytes, *, start: int, end: int, surface: str) -> None:
    if source[start:end] != surface.encode("utf-8"):
        raise ValueError("original-bill count slot differs from source bytes")
    line_start = source.rfind(b"\n", 0, start) + 1
    line_end = source.find(b"\n", end)
    if line_end < 0:
        line_end = len(source)
    if source[line_start:line_end].strip() != surface.encode("utf-8"):
        raise ValueError("original-bill count is not a standalone selected value")
    previous = source[:line_start].decode("utf-8").splitlines()
    previous_nonempty = next((line for line in reversed(previous) if line.strip()), None)
    if previous_nonempty is None or _CAPTION.fullmatch(previous_nonempty) is None:
        raise ValueError("original-bill count lacks its explicit preceding caption")


def validate_binding(binding: SemanticBinding, *, source: bytes | None = None) -> int:
    """Certify a fixed private count that can never become negotiability text."""

    if (
        binding.value_kind != "original_bill_count"
        or binding.logical_key != "aux:document:original_bill_count"
        or binding.render_mode != "deterministic_auxiliary"
        or binding.group_kind != "legal"
        or binding.group_key != "legal:original_bill_count"
        or binding.target_paths
        or binding.target_relationship != "none"
        or binding.derivation is not None
        or binding.dependency_paths
        or binding.dependency_bindings
        or binding.source_relationships
        or binding.realization.mode != "generated_auxiliary"
        or not binding.occurrences
    ):
        raise ValueError("original-bill count lacks independent private-fact ownership")
    values = set()
    for slot in binding.occurrences:
        if slot.target_paths or slot.semantic_role != binding.group_key:
            raise ValueError("original-bill count slot has a target or incorrect semantic role")
        values.add(parse_count(slot.source_text))
        if source is not None:
            _source_caption_proves(
                source,
                start=slot.byte_start,
                end=slot.byte_end,
                surface=slot.source_text,
            )
    if len(values) != 1:
        raise ValueError("repeated original-bill count surfaces disagree")
    return next(iter(values))
