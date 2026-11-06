"""Unique literal-frame parsing for retained synthetic receipts.

Unlike character-diff alignment, this cannot leave an inserted prefix outside
its field. Every byte between owners must remain identical. More than one
possible field partition is a hold, never a chosen first match.
"""

from functools import cache
import re


def unique_owner_spans(before, after, owners, *, expected_literals=None, expected_patterns=None, max_states=25000):
    if not owners or any(a > b for a, b in owners):
        raise ValueError("literal-frame parsing needs ordered owners")
    gaps, end = [], 0
    for a, b in owners:
        if a < end:
            raise ValueError("historical owners overlap")
        gaps.append(before[end:a])
        end = b
    gaps.append(before[end:])
    if not after.startswith(gaps[0]) or not after.endswith(gaps[-1]):
        raise ValueError("current text changed the external literal frame")
    visited = 0
    expected_literals = expected_literals or [None] * len(owners)
    expected_patterns = expected_patterns or [None] * len(owners)
    if len(expected_literals) != len(owners) or len(expected_patterns) != len(owners):
        raise ValueError("literal authorities must cover the exact owner sequence")
    patterns = [re.compile(p) if p is not None else None for p in expected_patterns]
    if any(not gap and owners[i][0] != owners[i][1] and owners[i+1][0] != owners[i+1][1]
           and not any((expected_literals[i], expected_literals[i + 1], patterns[i], patterns[i + 1]))
           for i, gap in enumerate(gaps[1:-1])):
        raise ValueError("unseparated owners require explicit semantic composition")

    def allowed(index, start, end):
        # An intentional source deletion has no retained rendered value. Its
        # two literal neighbours must remain adjacent; do not fill the hole
        # with an unrelated insertion merely to obtain a successful parse.
        if owners[index][0] == owners[index][1] and start != end:
            return False
        expected = expected_literals[index]
        pattern = patterns[index]
        if expected is None and pattern is None:
            return True
        try:
            value = after[start:end].decode()
        except UnicodeDecodeError:
            return False
        return (expected is None or " ".join(value.upper().split()) == expected) and (
            pattern is None or pattern.fullmatch(value) is not None
        )

    @cache
    def suffix(index, start):
        nonlocal visited
        visited += 1
        if visited > max_states:
            raise ValueError("literal-frame search exceeded its explicit review budget")
        if index == len(owners) - 1:
            finish = len(after) - len(gaps[-1])
            minimum = 0 if owners[index][0] == owners[index][1] else 1
            return (((start, finish),),) if finish - start >= minimum and allowed(index, start, finish) else ()
        gap = gaps[index + 1]
        result = []
        minimum = 0 if owners[index][0] == owners[index][1] else 1
        position = after.find(gap, start + minimum)
        while position >= 0:
            expected = expected_literals[index]
            if expected is not None:
                # Normalized length cannot decrease as this span grows.
                # Once it exceeds an exact literal's length, all later gap
                # occurrences are impossible. This preserves exhaustive
                # ambiguity detection without scanning the whole document
                # for every short, semantically constrained field.
                try:
                    normalized = " ".join(after[start:position].decode().upper().split())
                except UnicodeDecodeError:
                    normalized = None
                if normalized is not None and len(normalized) > len(expected):
                    break
            if allowed(index, start, position):
                for continuation in suffix(index + 1, position + len(gap)):
                    result.append(((start, position), *continuation))
                    if len(result) == 2:
                        return tuple(result)
            position = after.find(gap, position + 1)
        return tuple(result)

    solutions = suffix(0, len(gaps[0]))
    if not solutions:
        raise ValueError("current text changed protected inter-owner literals")
    if len(solutions) != 1:
        differing = [i for i, (a, b) in enumerate(zip(solutions[0], solutions[1], strict=True)) if a != b]
        raise ValueError(f"current literal frame permits ambiguous owner partitions at indices {differing}")
    return solutions[0]
