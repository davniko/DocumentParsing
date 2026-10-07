"""Source-proven numeric typography and dependent unit-converted surfaces.

Conversions use declared physical units, never ratios fitted to source numbers.
A converted printout is an alias of a canonical target measure, not another
independently sampled shipment total.
"""

from __future__ import annotations

import re
from decimal import ROUND_HALF_UP, Decimal
from typing import TYPE_CHECKING

from document_ocr.synthesis.curated import _number_style, flat, locate
from document_ocr.synthesis.rendering import _numeric_interpretations

if TYPE_CHECKING:
    from document_ocr.synthesis.curated_templates import SamplingBlueprint

# International pound and foot definitions are exact; volume scales cubically.
_UNITS = {
    "kilogram": ("mass", Decimal(1), r"KILOGRAMS?|KGS?|KGM"),
    "metric_tonne": ("mass", Decimal(1000), r"METRIC\s+TONNES?|TONNES?|MTS?"),
    "pound": ("mass", Decimal("0.45359237"), r"POUNDS?|LBS?"),
    "cubic_metre": ("volume", Decimal(1), r"CBM|M3|M³|CU\.?\s*M\.?"),
    "cubic_foot": ("volume", Decimal("0.028316846592"), r"CBF|FT3|FT³|CU\.?\s*FT\.?"),
}


def _presentations(surface: str) -> tuple[str, dict[Decimal, Decimal]]:
    matches = list(re.finditer(r"[+-]?\d(?:[\d., ]*\d)?", surface))
    if len(matches) != 1:
        raise ValueError(f"numeric owner is not a single measured scalar: {surface!r}")
    token = matches[0][0]
    values = {}
    for value, _, _, places in _numeric_interpretations(token):
        try:
            if _number_style(token, str(value), str(value)) == token:
                quantum = Decimal(1).scaleb(-places)
                values[value] = min(values.get(value, quantum), quantum)
        except ValueError:
            continue
    return token, values


def measured_number(surface: str, baseline: Decimal | None = None) -> tuple[str, Decimal]:
    """Select one numeric surface only when its typography exactly round-trips."""
    token, values = _presentations(surface)
    if baseline is not None:
        if baseline not in values:
            raise ValueError(f"numeric owner does not print its baseline {baseline}: {surface!r}")
        return token, baseline
    if len(values) != 1:
        raise ValueError(f"numeric owner lacks an unambiguous baseline: {surface!r}")
    return token, next(iter(values))


def printed_quantum(surface: str, baseline: Decimal) -> Decimal:
    """Recover precision from the source presentation, including grouped numbers."""
    _, values = _presentations(surface)
    if baseline not in values:
        raise ValueError(f"numeric owner does not print its baseline {baseline}: {surface!r}")
    return values[baseline]


def format_measure(surface: str, baseline: Decimal, value: Decimal) -> str:
    token, _ = measured_number(surface, baseline)
    return surface.replace(token, _number_style(token, str(baseline), str(value)), 1)


def _printed_unit(raw: str, occurrence: dict, unit: str) -> bool:
    start, end = occurrence["byte_start"], occurrence["byte_end"]
    encoded = raw.encode()
    if encoded[start:end].decode() != occurrence["source_text"]:
        raise ValueError("converted measure occurrence differs from its source bytes")
    text = occurrence["source_text"]
    pattern = r"(?<![A-Za-z])(?:" + _UNITS[unit][2] + r")(?![A-Za-z0-9])"
    if re.search(pattern, text, re.I):
        return True
    following = encoded[end:].decode().split("\n", 1)[0]
    preceding = encoded[:start].decode().rsplit("\n", 1)[-1]
    return bool(
        re.match(r"\s*(?:" + pattern + r")", following, re.I)
        or re.search(r"(?:" + pattern + r")\s*[:=]?\s*$", preceding, re.I)
    )


def converted_measure_surfaces(
    blueprint: SamplingBlueprint, target: dict, key: str, recipe: dict
) -> list[str]:
    """Validate and render a declared converted alias of a printed target measure.

    A source may round both unit representations separately. Compatibility is
    therefore intersection of their printed precision intervals, not equality
    of rounded central values. Unchanged facts retain their observed surfaces;
    changed facts use the exact unit factor and the alias's original precision.
    """
    if set(recipe) != {"measure_path", "render_unit"}:
        raise ValueError("converted measure recipe requires exactly measure_path and render_unit")
    path, unit = recipe["measure_path"], recipe["render_unit"]
    match = re.fullmatch(
        r"documentPatch\.goodsItemDetails\[\d+\]\.(grossWeight|netWeight|volume)", path
    )
    if not match or unit not in _UNITS:
        raise ValueError("converted measure requires a known measure path and explicit unit")
    previous, current = flat(blueprint.target), flat(target)
    old_unit, new_unit = previous[path + ".unit"], current[path + ".unit"]
    if old_unit not in _UNITS or new_unit not in _UNITS:
        raise ValueError("converted measure has an unsupported canonical unit")
    dimension = "volume" if match[1] == "volume" else "mass"
    if any(_UNITS[u][0] != dimension for u in (old_unit, new_unit, unit)):
        raise ValueError("converted measure unit dimension differs from its target measure")
    if key not in blueprint.historical_bindings or not any(
        r.key == key and r.curated_key is None for r in blueprint.regions
    ):
        raise ValueError("converted measure must have its own active historical surface owner")
    # The canonical measurement must itself have a source-grounded numeric owner.
    expressions = [b.expression for b in blueprint.contract.targets if b.path == path + ".value"]
    if len(expressions) != 1 or not re.fullmatch(r"\{[a-z][a-z0-9_]*\}", expressions[0]):
        raise ValueError("converted measure requires one direct canonical numeric owner")
    canonical_key = expressions[0][1:-1]
    variables = [v for v in blueprint.contract.variables if v.key == canonical_key]
    if len(variables) != 1 or variables[0].kind != dimension:
        raise ValueError("converted measure canonical owner is not a typed measurement")
    baseline = Decimal(str(previous[path + ".value"]))
    generated = Decimal(str(current[path + ".value"]))
    if not baseline.is_finite() or not generated.is_finite() or min(baseline, generated) < 0:
        raise ValueError("converted measure requires finite nonnegative measurement values")
    for occurrence in variables[0].occurrences:
        start, end = locate(blueprint.source, occurrence)
        if not _printed_unit(
            blueprint.source,
            dict(byte_start=start, byte_end=end, source_text=occurrence.text),
            old_unit,
        ):
            raise ValueError("canonical measurement unit is not printed beside its numeric owner")
    canonical_quantum = min(printed_quantum(o.text, baseline) for o in variables[0].occurrences)
    old_factor, new_factor, render_factor = (_UNITS[u][1] for u in (old_unit, new_unit, unit))
    lower = (baseline - canonical_quantum / 2) * old_factor
    upper = (baseline + canonical_quantum / 2) * old_factor
    rendered = []
    for occurrence in blueprint.historical_bindings[key]["occurrences"]:
        if not _printed_unit(blueprint.source, occurrence, unit):
            raise ValueError(f"declared converted unit is not printed beside its number: {key}")
        source = occurrence["source_text"]
        # Punctuation can mean a decimal or grouping separator. The fixed unit
        # conversion and printed precision must establish exactly one reading.
        matches = list(re.finditer(r"[+-]?\d(?:[\d., ]*\d)?", source))
        if len(matches) != 1:
            raise ValueError("converted measure must print exactly one numeric scalar")
        compatible = {}
        for observed, *_ in _numeric_interpretations(matches[0][0]):
            try:
                quantum = printed_quantum(source, observed)
            except ValueError:
                continue
            alias_lower = (observed - quantum / 2) * render_factor
            alias_upper = (observed + quantum / 2) * render_factor
            if max(lower, alias_lower) < min(upper, alias_upper):
                compatible[observed] = quantum
        if len(compatible) != 1:
            raise ValueError("converted source disagrees with the canonical printed interval")
        observed, quantum = next(iter(compatible.items()))
        if generated == baseline and old_unit == new_unit:
            rendered.append(source)
        else:
            value = (generated * new_factor / render_factor).quantize(
                quantum, rounding=ROUND_HALF_UP
            )
            rendered.append(format_measure(source, observed, value))
    return rendered
