"""Validate source-proved package arithmetic that is not in the extraction target.

The guard only applies when the original OCR and typed source package facts
establish a complete equation. It does not infer a hierarchy from a lone
package word, a product code, or an unrelated number in legal text.
"""

from __future__ import annotations

import re
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

from .package_equations import _LEVEL_UNITS

_UNIT_CATEGORIES = {
    **_LEVEL_UNITS,
    "INTERMEDIATE BULK CONTAINER": "PACKAGE_INTERMEDIATE_BULK_CONTAINER",
    "INTERMEDIATE BULK CONTAINERS": "PACKAGE_INTERMEDIATE_BULK_CONTAINER",
}
_UNIT = "|".join(re.escape(unit) for unit in sorted(_UNIT_CATEGORIES, key=lambda item: -len(item)))
_NUMBER = r"[1-9][0-9]*(?:,[0-9]{3})*"
_TOTAL_LINE = re.compile(rf"^\s*(?P<count>{_NUMBER})\s+(?P<unit>{_UNIT})\s*$", re.I)
_PRODUCT_ROW = re.compile(rf"^\s*.+\(\s*(?P<count>{_NUMBER})\s+(?P<unit>{_UNIT})\s*\)\s*$", re.I)
_MULTIPLICATION = re.compile(
    rf"^\s*(?P<outer>{_NUMBER})\s+(?P<outer_unit>{_UNIT})\s*[X\u00d7]\s*"
    rf"(?P<per>{_NUMBER})\s+(?P<inner_unit>{_UNIT})\s*=\s*"
    rf"(?P<total>{_NUMBER})\s+(?P<total_unit>{_UNIT})\s*$",
    re.I,
)
_OUTER_TOTAL = re.compile(rf"\bTOTAL\s+(?P<count>{_NUMBER})\s+PALLETS?\b", re.I)
_PALLET_MARK_PREFIX = re.compile(r"^\s*PALLET\s+NO\s*:", re.I)
_PALLET_MARK_RANGE = re.compile(
    r"^\s*PALLET\s+NO\s*:\s*(?P<first>[0-9]+)\s*-\s*(?P<last>[0-9]+)(?=\s|$)",
    re.I,
)
_SOURCE_COUNTED_OUTER_PALLETS = re.compile(
    rf"\bLOADED\s+ONTO\s+{_NUMBER}\s+PALLETS?\s+LOADED\s+INTO\b", re.I
)
_RENDERED_COUNTLESS_OUTER_PALLETS = re.compile(r"\bLOADED\s+ONTO\s+PALLETS?\s+INTO\b", re.I)
_PRINTED_BOX_COUNT = re.compile(rf"\b{_NUMBER}\s+BOXES\b", re.I)
_PRINTED_OUTER_PALLET_COUNT = re.compile(rf"\bON\s+{_NUMBER}\s+PALLETS\b", re.I)
_CARTON_NOUN = re.compile(r"\bCARTONS?\b", re.I)
_ADDED_CARTON_OVERPACK = re.compile(r"\bPACKED\s+IN\s+(?:[A-Z]+\s+){0,2}CARTONS\b", re.I)
_BOX_OF_LINE = re.compile(
    rf"^[ \t]*(?P<count>{_NUMBER})[ \t]+BOX\(ES\)[ \t]+of\b", re.I | re.M
)
_TOTAL_ITEMS_LINE = re.compile(
    rf"^[ \t]*Total[ \t]+Items[ \t]*:[ \t]*(?P<count>{_NUMBER})[ \t]*$", re.I | re.M
)
_COUNTED_CARTONS = re.compile(rf"\b{_NUMBER}\s+CARTONS?\b", re.I)


def _count(value: str) -> int:
    return int(value.replace(",", ""))


def _category(value: str) -> str:
    return _UNIT_CATEGORIES[value.upper()]


@dataclass(frozen=True, slots=True)
class _CountBlock:
    count: int
    category: str
    components: tuple[int, ...]


def _count_blocks(text: str) -> tuple[_CountBlock, ...]:
    result: list[_CountBlock] = []
    total: re.Match[str] | None = None
    components: list[int] = []
    for line in text.splitlines():
        found = _TOTAL_LINE.fullmatch(line)
        if found is not None:
            if total is not None:
                result.append(
                    _CountBlock(_count(total["count"]), _category(total["unit"]), tuple(components))
                )
            total = found
            components = []
            continue
        if (
            total is not None
            and (component := _PRODUCT_ROW.fullmatch(line)) is not None
            and _category(component["unit"]) == _category(total["unit"])
        ):
            components.append(_count(component["count"]))
    if total is not None:
        result.append(
            _CountBlock(_count(total["count"]), _category(total["unit"]), tuple(components))
        )
    return tuple(result)


@dataclass(frozen=True, slots=True)
class _MultiplicationRow:
    outer: int
    outer_category: str
    per: int
    inner_category: str
    total: int


def _multiplication_rows(text: str) -> tuple[_MultiplicationRow, ...]:
    result = []
    for line in text.splitlines():
        match = _MULTIPLICATION.fullmatch(line)
        if match is None or _category(match["inner_unit"]) != _category(match["total_unit"]):
            continue
        result.append(
            _MultiplicationRow(
                outer=_count(match["outer"]),
                outer_category=_category(match["outer_unit"]),
                per=_count(match["per"]),
                inner_category=_category(match["inner_unit"]),
                total=_count(match["total"]),
            )
        )
    return tuple(result)


def _packages(target: Mapping[str, Any]) -> tuple[Mapping[str, Any], ...]:
    return tuple(target.get("documentPatch", {}).get("cargoPackages", ()))


def _pallet_mark_lines(text: str) -> tuple[str, ...]:
    return tuple(line for line in text.splitlines() if _PALLET_MARK_PREFIX.match(line))


def _valid_pallet_mark_range(line: str) -> bool:
    match = _PALLET_MARK_RANGE.match(line)
    if match is None:
        return False
    first, last = int(match["first"]), int(match["last"])
    return 1 <= first <= last


def validate(
    *,
    source: bytes,
    rendered: bytes,
    source_target: Mapping[str, Any],
    target: Mapping[str, Any],
) -> None:
    """Reject stale product counts and retained equations after package sampling.

    A complete source count block or equation must match typed source packages
    before imposing any constraint on a descendant. Unmatched layout/phrasing
    is deliberately left to the broader template review rather than guessed.
    """
    original = source.decode("utf-8")
    revised = rendered.decode("utf-8")
    revised_upper = revised.upper()
    if (
        "ONTO" in revised_upper
        and "PALLET" in revised_upper
        and _RENDERED_COUNTLESS_OUTER_PALLETS.search(revised)
        and _SOURCE_COUNTED_OUTER_PALLETS.search(original)
    ):
        raise ValueError("source-proved outer pallet clause lost its count and loading verb")
    source_packages, target_packages = _packages(source_target), _packages(target)
    if len(source_packages) == len(target_packages) == 1:
        source_package, target_package = source_packages[0], target_packages[0]
        source_quantity = source_package.get("quantity")
        if (
            source_package.get("typeCategory") == "PACKAGE_BOX"
            and isinstance(source_quantity, int)
            and not isinstance(source_quantity, bool)
            and "TOTAL ITEMS" in original.upper()
        ):
            source_boxes = [_count(m["count"]) for m in _BOX_OF_LINE.finditer(original)]
            source_items = [_count(m["count"]) for m in _TOTAL_ITEMS_LINE.finditer(original)]
            if (
                source_boxes
                and source_items
                and set(source_boxes) == set(source_items) == {source_quantity}
            ):
                target_quantity = target_package.get("quantity")
                rendered_items = [_count(m["count"]) for m in _TOTAL_ITEMS_LINE.finditer(revised)]
                if (
                    not isinstance(target_quantity, int)
                    or isinstance(target_quantity, bool)
                    or len(rendered_items) != len(source_items)
                    or any(count != target_quantity for count in rendered_items)
                ):
                    raise ValueError("source-proved Total Items disagree with sampled packages")
                if (
                    target_package.get("typeCategory") == "PACKAGE_BOX"
                    and not _CARTON_NOUN.search(original)
                    and "CARTON" in revised_upper
                    and _COUNTED_CARTONS.search(revised)
                ):
                    raise ValueError("source-proved box cargo gained unsupported counted cartons")
    if (
        "PACKED" in revised_upper
        and "CARTON" in revised_upper
        and _ADDED_CARTON_OVERPACK.search(revised)
        and len(source_packages) == len(target_packages) == 1
        and source_packages[0].get("typeCategory")
        == target_packages[0].get("typeCategory")
        == "PACKAGE_BOX"
        and _PRINTED_BOX_COUNT.search(original)
        and _PRINTED_OUTER_PALLET_COUNT.search(original)
        and not _CARTON_NOUN.search(original)
    ):
        raise ValueError("generated carton overpack is unsupported by box/pallet source")
    source_mark_lines = _pallet_mark_lines(original)
    if (
        source_mark_lines
        and all(_valid_pallet_mark_range(line) for line in source_mark_lines)
        and any(not _valid_pallet_mark_range(line) for line in _pallet_mark_lines(revised))
    ):
        raise ValueError("source-proved pallet mark range is malformed")
    source_blocks = _count_blocks(original)
    if source_blocks and len(source_blocks) == len(source_packages):
        changed_blocks = _count_blocks(revised)
        for index, (block, package) in enumerate(zip(source_blocks, source_packages, strict=True)):
            if (
                len(block.components) < 2
                or sum(block.components) != block.count
                or block.count != package.get("quantity")
                or block.category != package.get("typeCategory")
            ):
                continue
            if len(changed_blocks) != len(source_blocks) or len(target_packages) != len(
                source_packages
            ):
                raise ValueError("source-proved product package blocks changed topology")
            observed, expected = changed_blocks[index], target_packages[index]
            if (
                observed.category != block.category
                or expected.get("typeCategory") != block.category
                or observed.count != expected.get("quantity")
                or len(observed.components) != len(block.components)
                or sum(observed.components) != observed.count
            ):
                raise ValueError("source-proved product package subtotals disagree with target")

    source_equations = _multiplication_rows(original)
    if len(source_equations) < 2 or len(source_packages) != 1 or len(target_packages) != 1:
        return
    source_package, target_package = source_packages[0], target_packages[0]
    inner_categories = {row.inner_category for row in source_equations}
    outer_categories = {row.outer_category for row in source_equations}
    if (
        len(inner_categories) != 1
        or len(outer_categories) != 1
        or next(iter(inner_categories)) != source_package.get("typeCategory")
        or any(row.outer * row.per != row.total for row in source_equations)
        or sum(row.total for row in source_equations) != source_package.get("quantity")
    ):
        return
    revised_equations = _multiplication_rows(revised)
    if revised_equations and (
        len(revised_equations) != len(source_equations)
        or target_package.get("typeCategory") != source_package.get("typeCategory")
        or any(
            row.outer * row.per != row.total
            or row.outer_category != next(iter(outer_categories))
            or row.inner_category != next(iter(inner_categories))
            for row in revised_equations
        )
        or sum(row.total for row in revised_equations) != target_package.get("quantity")
    ):
        raise ValueError("retained source-proved package equation disagrees with target")
    outer_total = sum(row.outer for row in source_equations)
    source_outer = {_count(match["count"]) for match in _OUTER_TOTAL.finditer(original)}
    if source_outer == {outer_total}:
        rendered_outer = {_count(match["count"]) for match in _OUTER_TOTAL.finditer(revised)}
        if rendered_outer:
            target_quantity = target_package.get("quantity")
            if not isinstance(target_quantity, int) or len(rendered_outer) != 1:
                raise ValueError("printed outer package count has no unique typed quantity")
            printed_outer = next(iter(rendered_outer))
            if target_package.get("typeCategory") == next(iter(outer_categories)):
                if printed_outer != target_quantity:
                    raise ValueError("printed outer package count disagrees with pallet target")
            elif printed_outer > target_quantity:
                raise ValueError("printed outer package count exceeds sampled inner packages")
