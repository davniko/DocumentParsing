"""Reject generated goods descriptions that invent unlabelled shipment measurements.

The linguistic cargo writer may mention a total mass or volume. Such a total
belongs to the same cargo group's structured fact, not only its description.
Per-piece/package capacities are different facts and are deliberately excluded.
"""

from __future__ import annotations

import re
from collections.abc import Mapping
from typing import Any

_MASS_TOTAL = re.compile(
    r"\b(?P<kind>gross(?:\s+cargo)?\s+(?:weight|mass)|net\s+(?:weight|mass))"
    r"\s*(?::|=)?\s*(?:of\s+)?\d[\d,.]*\s*"
    r"(?:kgs?|kilograms?|mt|metric\s+tonn?es?|tonn?es?|lbs?|pounds?)\b",
    re.IGNORECASE,
)
_VOLUME_TOTAL = re.compile(
    r"\b(?:total\s+)?(?P<kind>volume|measurement)"
    r"\s*(?::|=)?\s*(?:of\s+)?\d[\d,.]*\s*"
    r"(?:cbm|m3|m³|cubic\s+met(?:er|re)s?|cft|ft3|ft³|cubic\s+feet)\b",
    re.IGNORECASE,
)
_PER_UNIT_PREFIX = re.compile(r"\b(?:unit|each|per\s+\w+)\s*$", re.IGNORECASE)
_PER_UNIT_SUFFIX = re.compile(
    r"^\s*(?:[,;]\s*)?(?:per\b|each\b|/\s*(?:\d+\s*)?"
    r"(?:cartons?|boxes?|packages?|units?|pieces?|pcs?|ctns?|pkgs?|drums?|pallets?)\b)",
    re.IGNORECASE,
)


def _is_per_unit(description: str, start: int, end: int) -> bool:
    return bool(
        _PER_UNIT_PREFIX.search(description[max(0, start - 30) : start])
        or _PER_UNIT_SUFFIX.match(description[end : end + 32])
    )


def validate(target: Mapping[str, Any]) -> None:
    """Fail closed when a generated cargo clause has no same-group typed fact."""
    patch = target.get("documentPatch")
    if not isinstance(patch, Mapping):
        raise ValueError("generated target lacks documentPatch")
    groups = patch.get("cargoGroups", [])
    if not isinstance(groups, list):
        raise ValueError("generated cargoGroups is not a list")
    for index, group in enumerate(groups):
        if not isinstance(group, Mapping):
            raise ValueError(f"generated cargo group {index} is not an object")
        description = group.get("description")
        if description is None:
            continue
        if not isinstance(description, str):
            raise ValueError(f"generated cargo group {index} description is not text")
        for pattern in (_MASS_TOTAL, _VOLUME_TOTAL):
            for clause in pattern.finditer(description):
                if _is_per_unit(description, clause.start(), clause.end()):
                    continue
                kind = clause.group("kind").casefold()
                field = (
                    "grossWeight"
                    if kind.startswith("gross")
                    else "netWeight"
                    if kind.startswith("net")
                    else "volume"
                )
                if group.get(field) is None:
                    raise ValueError(
                        f"generated cargo group {index} has an explicit {field} "
                        "total in its description without the same-group typed fact"
                    )
