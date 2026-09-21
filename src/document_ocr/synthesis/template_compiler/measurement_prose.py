"""Proven mass summaries in target text, including component rows summing a total."""

from __future__ import annotations

import re
from collections.abc import Mapping
from decimal import ROUND_FLOOR, Decimal
from itertools import product
from typing import Any

_MASS = re.compile(
    r"(?<![A-Za-z0-9.])(?P<number>[0-9]+(?:[.,][0-9]+)*)\s*"
    r"(?P<unit>KGS?|KGM|KILOGRAMS?|MT|MTS|METRIC[ \t]+TONS?|TONNES?)\b",
    re.I,
)


def formal_paths(source: Mapping[str, Any]) -> frozenset[str]:
    """Keep arithmetic-only summaries host-owned; product prose stays linguistic."""
    generated = generate(source, source)
    result = set()
    for path, text in generated.items():
        remainder = _MASS.sub("", text)
        if re.fullmatch(
            r"[\s:.,;()\-]*(?:(?:WITH|TOTAL|NET|GROSS|WEIGHT|MASS)[\s:.,;()\-]*)*",
            remainder,
            re.I,
        ):
            result.add(path)
    return frozenset(result)


def generate(source: Mapping[str, Any], target: Mapping[str, Any]) -> dict[str, str]:
    from document_ocr.synthesis.rendering import render_number_surface

    from . import descendant as r

    updates: dict[str, str] = {}
    for index, group in enumerate(source["documentPatch"].get("cargoGroups", [])):
        rows = []
        for suffix, text in r._flatten_leaves(group).items():
            if not isinstance(text, str) or not re.fullmatch(
                r"description|additionalInformation\[\d+\]|handlingInstructions\[\d+\]", suffix
            ):
                continue
            for match in _MASS.finditer(text):
                factor = Decimal(1 if match["unit"].upper().startswith("K") else 1000)
                options = tuple(v for v in r._numeric_interpretations(match["number"]) if v > 0)
                if not options:
                    raise ValueError("cargo mass prose has no numeric interpretation")
                rows.append(
                    (f"documentPatch.cargoGroups[{index}].{suffix}", match, factor, options)
                )
        if not rows:
            continue
        original_totals: dict[Decimal, set[Decimal]] = {}
        for field in ("netWeight", "grossWeight"):
            if field not in group:
                continue
            old, new = group[field], target["documentPatch"]["cargoGroups"][index][field]
            factor = {
                "kilogram": Decimal(1),
                "metric_tonne": Decimal(1000),
                "pound": Decimal("0.45359237"),
            }[old["unit"]]
            if new["unit"] != old["unit"]:
                raise ValueError("cargo mass summary unit changed")
            original_totals.setdefault(Decimal(str(old["value"])) * factor, set()).add(
                Decimal(str(new["value"])) * factor
            )
        assignments: list[tuple[tuple[Decimal, ...], Decimal, Decimal]] = []
        # Source evidence must select exactly one interpretation and one target total.
        for values in product(*(row[3] for row in rows)):
            total = sum((v * row[2] for v, row in zip(values, rows, strict=True)), Decimal(0))
            if total in original_totals:
                assignments.extend((values, total, new) for new in original_totals[total])
        if len(assignments) != 1:
            # Separate total repetitions are not component rows. Each must itself
            # resolve uniquely to a declared mass; otherwise compilation is incomplete.
            assignments = []
            for row in rows:
                choices = [
                    (v, v * row[2], new)
                    for v in row[3]
                    for new in original_totals.get(v * row[2], ())
                ]
                if len(choices) != 1:
                    raise ValueError(
                        f"cargo mass prose lacks a unique total/component contract: {row[0]}"
                    )
                value, total, new = choices[0]
                assignments.append(((value,), total, new))
            groups = [
                ([row], assignment) for row, assignment in zip(rows, assignments, strict=True)
            ]
        else:
            groups = [(rows, assignments[0])]
        edits: dict[str, list[tuple[int, int, str]]] = {}
        for members, (values, total, new_total) in groups:
            # Use the coarsest printed quantum shared by all rows. The exact
            # target total must be representable; no rounding away physical mass.
            quanta = [
                Decimal(1).scaleb(int(value.as_tuple().exponent)) * row[2]
                for row, value in zip(members, values, strict=True)
            ]
            quantum = max(quanta)
            units = new_total / quantum
            if units != units.to_integral_value():
                raise ValueError("cargo mass total exceeds its prose precision")
            desired = [
                value * row[2] / total * units for row, value in zip(members, values, strict=True)
            ]
            allocated = [v.to_integral_value(rounding=ROUND_FLOOR) for v in desired]
            remainder = int(units - sum(allocated))
            for i in sorted(range(len(desired)), key=lambda i: (-(desired[i] - allocated[i]), i))[
                :remainder
            ]:
                allocated[i] += 1
            for row, old, count in zip(members, values, allocated, strict=True):
                if count <= 0:
                    raise ValueError("cargo mass component would become nonpositive")
                path, match, factor, _ = row
                rendered = render_number_surface(match["number"], old, count * quantum / factor)
                edits.setdefault(path, []).append((*match.span("number"), rendered))
        for path, changes in edits.items():
            text = r._resolve_path(source, path)
            for start, end, replacement in sorted(changes, reverse=True):
                text = text[:start] + replacement + text[end:]
            updates[path] = text
    return updates


def validate(source: Mapping[str, Any], target: Mapping[str, Any]) -> None:
    from . import descendant as r

    for path, expected in generate(source, target).items():
        actual = r._resolve_path(target, path)

        def signature(text: str) -> list[tuple[str, str]]:
            return sorted((m["number"], m["unit"].upper()) for m in _MASS.finditer(text))

        if signature(actual) != signature(expected):
            raise ValueError(f"generated cargo prose contradicts its structured mass: {path}")
