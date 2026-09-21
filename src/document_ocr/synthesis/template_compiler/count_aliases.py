"""Explicit English-word/digit count aliases and immutable source constraints.

An unowned alias is source context, not permission to mutate outside compiled
slots. Its dependent package quantities are fixed before scenario generation.
Fully owned aliases remain variable and are rendered from one integer.
"""

from __future__ import annotations

import re
from collections.abc import Mapping
from typing import Any

from .models import CertifiedSemanticTemplate

_WORDS = [
    "zero",
    "one",
    "two",
    "three",
    "four",
    "five",
    "six",
    "seven",
    "eight",
    "nine",
    "ten",
    "eleven",
    "twelve",
    "thirteen",
    "fourteen",
    "fifteen",
    "sixteen",
    "seventeen",
    "eighteen",
    "nineteen",
    "twenty",
    "thirty",
    "forty",
    "fifty",
    "sixty",
    "seventy",
    "eighty",
    "ninety",
    "hundred",
    "thousand",
    "million",
    "and",
]
_WORD = r"(?:" + "|".join(_WORDS) + r")"
_PAIR = re.compile(
    rf"\b(?P<words>{_WORD}(?:[ -]+{_WORD})*)[ \t]*\([ \t]*(?P<digits>\d[\d,]*)[ \t]*\)",
    re.IGNORECASE,
)
_QUANTITY = re.compile(r"documentPatch\.cargoPackages\[\d+\]\.quantity")


def aliases(text: str) -> tuple[tuple[re.Match[str], int, int], ...]:
    from .descendant import _number_word_value

    result = []
    for match in _PAIR.finditer(text):
        number = _number_word_value(match["words"])
        if number is not None:
            result.append((match, number, int(match["digits"].replace(",", ""))))
    return tuple(result)


def validate(text: str) -> None:
    for match, words, digits in aliases(text):
        if words != digits:
            raise ValueError(f"contradictory number-word/digit count: {match[0]}")


def fixed_quantities(
    source: bytes, template: CertifiedSemanticTemplate, target: Mapping[str, Any]
) -> dict[str, int]:
    from .descendant import _resolve_path

    text = source.decode("utf-8")
    validate(text)
    bindings = {b.logical_key: b for b in template.bindings}
    result: dict[str, int] = {}
    for match, count, _ in aliases(text):
        owners = []
        for part in ("words", "digits"):
            start, end = (len(text[:position].encode()) for position in match.span(part))
            overlaps = [
                b
                for b in template.bindings
                for slot in b.occurrences
                if slot.byte_start < end and slot.byte_end > start
            ]
            covered = [
                b
                for b in overlaps
                if any(s.byte_start <= start and s.byte_end >= end for s in b.occurrences)
            ]
            if overlaps and not covered:
                raise ValueError("count alias is only partially owned by compiled slots")
            owners.append(covered)
        if all(owners) or not any(owners):
            continue
        for binding in owners[0] or owners[1]:
            paths = set(binding.target_paths)
            if binding.derivation == "number_to_words":
                paths.update(binding.dependency_paths)
                for key in binding.dependency_bindings:
                    paths.update(bindings[key].target_paths)
            quantities = {path for path in paths if _QUANTITY.fullmatch(path)}
            if not quantities:
                continue  # Container/original counts have separate fixed-topology contracts.
            values = {path: _resolve_path(target, path) for path in quantities}
            if any(type(v) is not int for v in values.values()) or sum(values.values()) != count:
                raise ValueError("immutable count alias disagrees with its quantity ownership")
            result.update(values)
    return result


def render_pair(text: str, old: int, new: int) -> str | None:
    from .descendant import _case_like, _number_to_words

    pairs = aliases(text)
    if not pairs:
        return None
    replacements = []
    for match, words, digits in pairs:
        if words != old or digits != old:
            raise ValueError("count alias disagrees with its declared derivation")
        new_words = _number_to_words(new)
        new_words = (
            new_words.title() if match["words"].istitle() else _case_like(match["words"], new_words)
        )
        replacements.extend(
            [
                (*match.span("words"), new_words),
                (*match.span("digits"), f"{new:,}" if "," in match["digits"] else str(new)),
            ]
        )
    for start, end, value in sorted(replacements, reverse=True):
        text = text[:start] + value + text[end:]
    return text
