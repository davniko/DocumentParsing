"""Numeric date interpretation after a caller has established field ownership.

Geography and hidden historical labels never establish a printed date convention.
Callers may supply an explicit printed order or independently owned witnesses from
the same convention; absent either, genuinely ambiguous numeric dates are day-first.
"""

from __future__ import annotations

import re
from collections.abc import Iterable
from datetime import date
from typing import Literal

DateOrder = Literal["DMY", "MDY", "YMD"]
_NUMERIC = re.compile(r"(?P<a>\d{1,4})(?P<s>[-/. ])(?P<b>\d{1,2})(?P=s)(?P<c>\d{2,4})")
_ORDER_MARKERS = {
    "DMY": re.compile(
        r"\b(?:DD\s*[/.-]\s*MM\s*[/.-]\s*(?:YYYY|YY)|DAY\s*[/ .-]\s*MONTH\s*[/ .-]\s*YEAR)\b", re.I
    ),
    "MDY": re.compile(
        r"\b(?:MM\s*[/.-]\s*DD\s*[/.-]\s*(?:YYYY|YY)|MONTH\s*[/ .-]\s*DAY\s*[/ .-]\s*YEAR)\b", re.I
    ),
    "YMD": re.compile(
        r"\b(?:YYYY\s*[/.-]\s*MM\s*[/.-]\s*DD|YEAR\s*[/ .-]\s*MONTH\s*[/ .-]\s*DAY)\b", re.I
    ),
}


def adjacent_printed_date_order(source: str, start: int, end: int) -> DateOrder | None:
    """Read an explicit order attached to a known date occurrence.

    Same physical line and immediately adjacent nonempty lines are eligible only
    when they print an actual date-format marker. This establishes formatting,
    not the semantic owner of the date. It never borrows distant page metadata.
    """
    line_start = source.rfind("\n", 0, start) + 1
    line_end = source.find("\n", end)
    if line_end < 0:
        line_end = len(source)
    before = [line for line in source[:line_start].splitlines() if line.strip()]
    after = [line for line in source[line_end:].splitlines() if line.strip()]
    context = "\n".join(
        [before[-1] if before else "", source[line_start:line_end], after[0] if after else ""]
    )
    orders = {order for order, pattern in _ORDER_MARKERS.items() if pattern.search(context)}
    if len(orders) > 1:
        raise ValueError("conflicting adjacent printed date formats")
    return next(iter(orders)) if orders else None


def numeric_date_candidates(surface: str) -> dict[DateOrder, date]:
    """Return calendar-valid interpretations, rejecting unsupported token shapes."""
    match = _NUMERIC.fullmatch(surface.strip())
    if match is None:
        raise ValueError(f"unsupported numeric date surface: {surface!r}")
    first, second, last = (int(match[name]) for name in ("a", "b", "c"))
    if len(match["a"]) == 4:
        try:
            return {"YMD": date(first, second, last)}
        except ValueError as exc:
            raise ValueError(f"impossible numeric date: {surface!r}") from exc
    if len(match["a"]) > 2 or len(match["c"]) not in (2, 4):
        raise ValueError(f"unsupported numeric date surface: {surface!r}")
    year = last + (2000 if last <= 68 else 1900) if len(match["c"]) == 2 else last
    candidates: dict[DateOrder, date] = {}
    for order, month, day in (("DMY", second, first), ("MDY", first, second)):
        try:
            candidates[order] = date(year, month, day)
        except ValueError:
            continue
    if not candidates:
        raise ValueError(f"impossible numeric date: {surface!r}")
    return candidates


def resolve_numeric_date(
    surface: str,
    *,
    explicit_order: DateOrder | None = None,
    same_convention_witnesses: Iterable[str] = (),
) -> date:
    """Use explicit/source-convention evidence first, otherwise day-first.

    Witness ownership and convention membership are required caller evidence, not
    guessed from proximity, country, carrier or language. Contradictory evidence is
    an error. A year-first token is already self-describing and is not a witness for
    ordering the day/month positions of a year-last token.
    """
    candidates = numeric_date_candidates(surface)
    if "YMD" in candidates:
        if explicit_order not in (None, "YMD"):
            raise ValueError("explicit date order contradicts four-digit year-first source")
        return candidates["YMD"]
    orders: set[DateOrder] = set()
    for witness in same_convention_witnesses:
        possibilities = numeric_date_candidates(witness)
        if len(possibilities) == 1 and "YMD" not in possibilities:
            orders.update(possibilities)
    if explicit_order is not None:
        orders.add(explicit_order)
    if len(orders) > 1:
        raise ValueError("conflicting explicit/same-convention date order evidence")
    if orders:
        order = orders.pop()
        if order not in candidates:
            raise ValueError("date contradicts its established source convention")
        return candidates[order]
    return candidates.get("DMY", next(iter(candidates.values())))


def render_numeric_date(surface: str, source: date, target: date) -> str:
    """Render a certified numeric date, using visible YMD or the DMY default.

    Explicit source-format recipes are a separate renderer path. This function has
    no witness input, so a historical MDY source does not license ambiguous MDY
    descendants. Source separators and per-position zero padding are retained.
    """
    candidates = numeric_date_candidates(surface)
    if source not in candidates.values():
        raise ValueError("numeric date surface does not express its certified source value")
    match = _NUMERIC.fullmatch(surface.strip())
    assert match is not None
    if "YMD" in candidates:
        values = (target.year, target.month, target.day)
    else:
        values = (
            target.day,
            target.month,
            target.year % 100 if len(match["c"]) == 2 else target.year,
        )
    parts = [
        f"{value:0{len(match[key])}d}" for value, key in zip(values, ("a", "b", "c"), strict=True)
    ]
    rendered = match["s"].join(parts)
    return (
        surface[: len(surface) - len(surface.lstrip())]
        + rendered
        + surface[len(surface.rstrip()) :]
    )
