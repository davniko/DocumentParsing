"""Proven mass summaries in target text, including component rows summing a total."""

from __future__ import annotations

import re
from collections.abc import Mapping, Sequence
from decimal import ROUND_FLOOR, Decimal
from itertools import product
from typing import Any

from .models import SemanticBinding

EXPLICIT_MASS_PRODUCT = re.compile(
    r"\(\s*(?P<unitmass>\d[\d.,]*)\s*(?P<unit>KGS?|KGM)\s*[X\u00d7*]\s*"
    r"(?P<count>\d+)\s*\)\s*(?P<total>\d[\d.,]*)\s*(?P<totalunit>KGS?|KGM)\b",
    re.I,
)


def require_product_count_owners(source: bytes, bindings: Sequence[SemanticBinding]) -> None:
    """Every explicit mass-times-count total needs a dynamic count owner.

    Equal source numbers alone are not a dependency contract. Repeated counts
    must be bound to a package quantity or explicitly depend on its owner;
    otherwise generation could change the total and leave a stale literal.
    """
    text = source.decode("utf-8")
    for match in EXPLICIT_MASS_PRODUCT.finditer(text):
        start = len(text[: match.start("count")].encode("utf-8"))
        end = len(text[: match.end("count")].encode("utf-8"))
        if not any(
            any(slot.byte_start <= start and end <= slot.byte_end for slot in b.occurrences)
            and any(
                re.fullmatch(r"documentPatch\.cargoPackages\[\d+\]\.quantity", path)
                for path in (*b.target_paths, *b.dependency_paths)
            )
            for b in bindings
        ):
            raise ValueError(
                "explicit mass-product count lacks package ownership: " + match.group()
            )


_MASS = re.compile(
    r"(?<![A-Za-z0-9.])(?P<number>[0-9]+(?:(?:[.,]|[ \u00a0](?=[0-9]{3}\b))[0-9]+)*)\s*"
    r"(?P<unit>KGS?|KGM|KILOGRAMS?|MT|MTS|METRIC[ \t]+TONS?|TONNES?)\b",
    re.I,
)
_PACK = r"(?:BAGS?|CARTONS?|BOX(?:ES)?|DRUMS?|SACKS?|BALES?|PAILS?)"
_MATERIAL_NAME = r"(?:NEW|STEEL|PE|PP|PAPER|PLASTIC|POLYPROPYLENE|JUMBO|FIBER|FIBRE)"
_MATERIAL = rf"(?:{_MATERIAL_NAME}\s+)*"
_UNIT_TAIL = re.compile(
    r"\.?\s*(?:(?:(?:NET|GROSS)(?:\s+WEIGHT)?\s+)?EACH\b|[X\u00d7*]\s*(?P<count>[0-9]+)\b|"
    r"(?:(?:NET|GROSS)(?:\s+WEIGHT)?\s+)?(?:PER\s+|/\s*)?" + _MATERIAL + _PACK + r"\b)",
    re.I,
)
_UNIT_PREFIX = re.compile(_PACK + rf"\s*(?:\(\s*{_MATERIAL_NAME}\s*\)\s*)?(?:OF\s*|\(\s*)?$", re.I)


def _paired_unit_scope(text: str, match: re.Match[str]) -> str | None:
    """An explicit NET AND GROSS PER BOX clause owns both measurement tokens."""
    qualifier = re.match(r"\.?\s*(NET|GROSS)\b", text[match.end() :], re.I)
    if qualifier is None:
        return None
    suffix = text[match.end() + qualifier.end() :]
    if re.match(r"\s+PER\s+" + _MATERIAL + _PACK + r"\b", suffix, re.I):
        return qualifier[1].lower() + "Weight"
    connector = re.match(r"\s+AND\s+", suffix, re.I)
    if connector is None:
        return None
    next_start = match.end() + qualifier.end() + connector.end()
    other = _MASS.match(text, next_start)
    if other is None:
        return None
    scope = re.match(
        r"\.?\s*(NET|GROSS)\s+PER\s+" + _MATERIAL + _PACK + r"\b",
        text[other.end() :],
        re.I,
    )
    if scope is None or scope[1].upper() == qualifier[1].upper():
        return None
    return qualifier[1].lower() + "Weight"


def _unit_contract(
    source: Mapping[str, Any], index: int, text: str, match: re.Match[str]
) -> tuple[int, tuple[tuple[str, Decimal], ...], tuple[int, int] | None] | None:
    """Prove an explicit EACH or multiplication against one complete package total."""
    from . import descendant as r

    tail = _UNIT_TAIL.match(text, match.end())
    prefix = _UNIT_PREFIX.search(text[: match.start()])
    paired_field = _paired_unit_scope(text, match)
    qualifier = re.match(r"\.?\s*(NET|GROSS)\b", text[match.end() :], re.I)
    explicit_field = paired_field or (qualifier[1].lower() + "Weight" if qualifier else None)
    # A standalone packing-weight field and an explicit WT suffix are also
    # observations, but become unit contracts only with the same exact
    # quantity-times-unit-equals-total proof required below. A product's bare
    # weight suffix is not sufficient, nor is a missing shipment total.
    packing_weight = (
        tail is None
        and prefix is None
        and (
            (not text[: match.start()].strip() and not text[match.end() :].strip())
            or re.fullmatch(r"\s*WT\.?\s*", text[match.end() :], re.I) is not None
        )
    )
    if tail is None and prefix is None and not packing_weight and paired_field is None:
        return None
    group = source["documentPatch"]["cargoGroups"][index]
    packages = [
        (i, p)
        for i, p in enumerate(source["documentPatch"].get("cargoPackages", []))
        if p["groupId"] == group["groupId"]
    ]
    if len(packages) != 1 or "quantity" not in packages[0][1]:
        if packing_weight and tail is None and prefix is None:
            return None  # It may instead be a complete shipment/component mass.
        raise ValueError("per-unit mass needs an unambiguous printed package owner")
    package_index, package = packages[0]
    quantity = Decimal(str(package["quantity"]))
    if packing_weight and tail is None and prefix is None and quantity <= 1:
        return None  # A single package cannot distinguish unit mass from total mass.
    explicit = Decimal(tail["count"]) if tail is not None and tail["count"] is not None else None
    factor = Decimal(1 if match["unit"].upper().startswith("K") else 1000)
    candidates = []
    for unit in r._numeric_interpretations(match["number"]):
        if unit <= 0:
            continue
        for inner, outer in (
            ((Decimal(1), False),)
            if explicit is None
            else (
                ((Decimal(1), True), (explicit, False))
                if explicit == quantity
                else ((explicit, False),)
            )
        ):
            fields = []
            for field in ("netWeight", "grossWeight"):
                if explicit_field is not None and field != explicit_field:
                    continue
                if field not in group:
                    # An explicit unit NET observation is already a physical
                    # fact even when the extraction label omits its total. Keep
                    # it private; an unqualified mass still needs a total proof.
                    if field == "netWeight" and explicit_field == field:
                        private_total = unit * factor * inner * quantity
                        gross = group.get("grossWeight")
                        if gross is not None:
                            gross_kg = (
                                Decimal(str(gross["value"]))
                                * {
                                    "kilogram": Decimal(1),
                                    "metric_tonne": Decimal(1000),
                                    "pound": Decimal("0.45359237"),
                                }[gross["unit"]]
                            )
                            if private_total > gross_kg:
                                continue
                        fields.append((field, unit * factor * inner))
                    continue
                old = group[field]
                mass_factor = {
                    "kilogram": Decimal(1),
                    "metric_tonne": Decimal(1000),
                    "pound": Decimal("0.45359237"),
                }[old["unit"]]
                if unit * factor * inner * quantity == Decimal(str(old["value"])) * mass_factor:
                    fields.append((field, unit * factor * inner / mass_factor))
            if fields:
                if outer and tail is None:
                    raise ValueError("explicit outer package count has no printed count evidence")
                candidates.append(
                    (package_index, tuple(fields), tail.span("count") if outer and tail else None)
                )
    if not candidates and packing_weight and tail is None and prefix is None:
        return None  # Do not reinterpret an ordinary total as a packing weight.
    if len(candidates) != 1:
        raise ValueError("per-unit mass lacks a unique source quantity/total proof")
    return candidates[0]


def per_package_totals(source: Mapping[str, Any], target: Mapping[str, Any]) -> dict[str, Decimal]:
    """Retain proved unit packing weights; calculate totals from generated quantities."""
    from . import descendant as r

    updates: dict[str, Decimal] = {}
    for index, group in enumerate(source["documentPatch"].get("cargoGroups", [])):
        for suffix, text in r._flatten_leaves(group).items():
            if not isinstance(text, str) or not re.fullmatch(
                r"description|additionalInformation\[\d+\]|handlingInstructions\[\d+\]", suffix
            ):
                continue
            for match in _MASS.finditer(text):
                contract = _unit_contract(source, index, text, match)
                if contract is None:
                    continue
                package_index, fields, _ = contract
                count = Decimal(
                    str(target["documentPatch"]["cargoPackages"][package_index]["quantity"])
                )
                for field, multiplier in fields:
                    if field not in group:
                        continue  # Physical-only totals must never become labels.
                    path = f"documentPatch.cargoGroups[{index}].{field}.value"
                    value = count * multiplier
                    if updates.setdefault(path, value) != value:
                        raise ValueError("per-unit mass contracts disagree")
    return updates


def private_net_weights(source: Mapping[str, Any], target: Mapping[str, Any]) -> dict[str, Decimal]:
    """Explicit unit net masses, owned by one package row, in kilograms.

    This private view is used by goods compatibility and capacity validation.
    It never adds inferred totals to the extraction target.
    """
    from . import descendant as r

    result: dict[str, Decimal] = {}
    for index, group in enumerate(source["documentPatch"].get("cargoGroups", ())):
        if "netWeight" in group:
            continue
        for suffix, text in r._flatten_leaves(group).items():
            if not isinstance(text, str) or not re.fullmatch(
                r"description|additionalInformation\[\d+\]|handlingInstructions\[\d+\]", suffix
            ):
                continue
            for match in _MASS.finditer(text):
                contract = _unit_contract(source, index, text, match)
                if contract is None:
                    continue
                package_index, fields, _ = contract
                observed = source["documentPatch"]["cargoPackages"][package_index]
                sampled = target["documentPatch"]["cargoPackages"][package_index]
                if any(
                    sampled.get(k) != observed.get(k) for k in ("typeCategory", "typeDescription")
                ):
                    raise ValueError("private unit mass package kind changed")
                for field, multiplier in fields:
                    if field != "netWeight":
                        continue
                    value = Decimal(str(sampled["quantity"])) * multiplier
                    if value <= 0 or result.setdefault(group["groupId"], value) != value:
                        raise ValueError("private per-unit net mass contracts disagree")
    return result


def physical_group(group: Mapping[str, Any], private_net_kg: Decimal | None) -> dict[str, Any]:
    """Validate the observed gross mass against the privately proved net mass."""
    result = dict(group)
    if private_net_kg is None:
        return result
    if "netWeight" in group:
        raise ValueError("private net mass cannot overwrite a labelled measurement")
    gross = group.get("grossWeight")
    if (
        gross is not None
        and Decimal(str(gross["value"]))
        * {
            "kilogram": Decimal(1),
            "metric_tonne": Decimal(1000),
            "pound": Decimal("0.45359237"),
        }[gross["unit"]]
        < private_net_kg
    ):
        raise ValueError("private per-unit net mass exceeds generated gross mass")
    result["netWeight"] = {"unit": "kilogram", "value": float(private_net_kg)}
    return result


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
    unit_totals = per_package_totals(source, target)
    for path, value in unit_totals.items():
        if Decimal(str(r._resolve_path(target, path))) != value:
            raise ValueError("generated group total contradicts its proved unit packing weight")
    for index, group in enumerate(source["documentPatch"].get("cargoGroups", [])):
        rows = []
        unit_edits: dict[str, list[tuple[int, int, str]]] = {}
        for suffix, text in r._flatten_leaves(group).items():
            if not isinstance(text, str) or not re.fullmatch(
                r"description|additionalInformation\[\d+\]|handlingInstructions\[\d+\]", suffix
            ):
                continue
            for match in _MASS.finditer(text):
                contract = _unit_contract(source, index, text, match)
                if contract is not None:
                    package_index, _, count_span = contract
                    path = f"documentPatch.cargoGroups[{index}].{suffix}"
                    unit_edits.setdefault(path, [])
                    if count_span is not None:
                        count = target["documentPatch"]["cargoPackages"][package_index]["quantity"]
                        unit_edits[path].append((*count_span, str(count)))
                    continue
                factor = Decimal(1 if match["unit"].upper().startswith("K") else 1000)
                options = tuple(v for v in r._numeric_interpretations(match["number"]) if v > 0)
                if not options:
                    raise ValueError("cargo mass prose has no numeric interpretation")
                rows.append(
                    (f"documentPatch.cargoGroups[{index}].{suffix}", match, factor, options)
                )
        for path, changes in unit_edits.items():
            text = r._resolve_path(source, path)
            for start, end, replacement in sorted(changes, reverse=True):
                text = text[:start] + replacement + text[end:]
            updates[path] = text
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
        edits: dict[str, list[tuple[int, int, str]]] = unit_edits
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

    private = private_net_weights(source, target)
    for group in target["documentPatch"].get("cargoGroups", ()):
        physical_group(group, private.get(group["groupId"]))

    for path, expected in generate(source, target).items():
        actual = r._resolve_path(target, path)

        def signature(text: str) -> list[tuple[str, str]]:
            return sorted((m["number"], m["unit"].upper()) for m in _MASS.finditer(text))

        if signature(actual) != signature(expected):
            raise ValueError(f"generated cargo prose contradicts its structured mass: {path}")

        def products(text: str) -> list[tuple[str, str, str]]:
            return sorted(
                (match["number"], match["unit"].upper(), tail["count"])
                for match in _MASS.finditer(text)
                if (tail := _UNIT_TAIL.match(text, match.end())) is not None
                and tail["count"] is not None
            )

        if products(actual) != products(expected):
            raise ValueError(f"generated cargo prose contradicts its package-mass product: {path}")
