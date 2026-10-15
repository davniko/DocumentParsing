"""Explicit source ventilation meaning shared by sampling and rendering.

Carrying airflow is not cargo volume, humidity, or an unqualified vent opening.
Only printed volumetric rates and explicit closed/zero declarations supply a
canonical cubic-metre/hour value. Rendering retains the source's own units.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from decimal import Decimal
from functools import lru_cache

# The international foot is exactly 0.3048 m; one minute is exactly 1/60 hour.
# NIST SP811 Appendix B.9 publishes the corresponding ft³/min → m³/s factor.
CFM_TO_CBM_PER_HOUR = Decimal("0.3048") ** 3 * 60
_CAPTION = re.compile(r"\b(?:VENT(?:IL\w*|S)?|FRESH\s+AIR(?:\s+VENTS?)?)\b", re.I)
_RATE = re.compile(
    r"(?P<value>[+-]?\d+(?:\.\d+)?)\s*"
    r"(?P<unit>(?:CBM|M3|M³)\s*(?:/\s*H(?:R)?|PER\s+HOUR)|CFM)\b",
    re.I,
)
_CLOSED = re.compile(r"\b(?:CLOSED|CLOSE|SHUT|OFF|CERRAD[AO])\b", re.I)
_OPEN = re.compile(r"\b(?:OPEN|ABIERT[AO])\b", re.I)
_STATE = re.compile(
    _CAPTION.pattern + r"\s*[:=.]*\s*(?P<negation>(?:NOT|NEVER|NON|NO)\s+)?"
    r"(?P<state>CLOSED|CLOSE|SHUT|OFF|CERRAD[AO]|OPEN|ABIERT[AO])\b",
    re.I,
)


@dataclass(frozen=True, slots=True)
class VentilationMeaning:
    """One source-owned rate; qualitative closed text has no invented unit."""

    rate_cbm_per_hour: Decimal
    numeric_spans: tuple[tuple[int, int, str], ...]
    closed: bool


class UnrepresentableVentilationRate(ValueError):
    """A sampled rate cannot fit the source's explicit CFM numeric grid."""


def is_ventilation_instruction(text: str) -> bool:
    return _CAPTION.search(text) is not None


def temperature_in_unit(value: Decimal, source_unit: str, target_unit: str) -> Decimal:
    """Convert only explicitly established Celsius/Fahrenheit carrying units."""
    if source_unit not in {"celsius", "fahrenheit"} or target_unit not in {"celsius", "fahrenheit"}:
        raise ValueError("unsupported carrying temperature unit")
    if source_unit == target_unit:
        return value
    return (value - 32) * 5 / 9 if target_unit == "celsius" else value * 9 / 5 + 32


@lru_cache(maxsize=8192)
def ventilation_meaning(text: str, *, owned: bool = False) -> VentilationMeaning | None:
    """Parse explicit airflow only; ownership permits caption-free rate fragments."""
    if not owned and not is_ventilation_instruction(text):
        return None
    rates, spans = set(), []
    for match in _RATE.finditer(text):
        value = Decimal(match["value"])
        if value < 0:
            raise ValueError("printed ventilation rate cannot be negative")
        unit = "cfm" if match["unit"].upper() == "CFM" else "cbm_per_hour"
        rates.add(value * CFM_TO_CBM_PER_HOUR if unit == "cfm" else value)
        spans.append((*match.span("value"), unit))
    state = _STATE.search(text)
    closed = bool(state and _CLOSED.fullmatch(state["state"])) or bool(
        owned and _CLOSED.fullmatch(text.strip())
    )
    if state:
        clause = re.split(r"[;\n]", text[state.start() :], maxsplit=1)[0]
        if state["negation"] or (_OPEN.search(clause) and _CLOSED.search(clause)):
            raise ValueError("ventilation open/closed declaration is ambiguous")
    if closed:
        rates.add(Decimal(0))
    if len(rates) > 1:
        raise ValueError("observed shipment has conflicting ventilation rates")
    return VentilationMeaning(rates.pop(), tuple(spans), closed) if rates else None


def render_ventilation(text: str, rate_cbm_per_hour: Decimal) -> str:
    """Change only owned airflow numbers, keeping explicit units and other text."""
    if not rate_cbm_per_hour.is_finite() or rate_cbm_per_hour < 0:
        raise ValueError("sampled ventilation rate must be finite and nonnegative")
    meaning = ventilation_meaning(text, owned=True)
    if meaning is None:
        raise ValueError("ventilation owner lacks an explicit airflow unit or closed declaration")
    if meaning.closed and rate_cbm_per_hour != 0:
        raise ValueError("a closed ventilation owner cannot print a positive airflow rate")
    if not meaning.numeric_spans:
        if rate_cbm_per_hour != 0:
            raise ValueError("a closed ventilation owner cannot print a positive airflow rate")
        return text
    result = text
    for start, end, unit in reversed(meaning.numeric_spans):
        value = rate_cbm_per_hour / CFM_TO_CBM_PER_HOUR if unit == "cfm" else rate_cbm_per_hour
        # Preserve the existing CBM rendering byte-for-byte. CFM conversion is
        # source-grid exact, never an implausible recurring decimal expansion.
        if unit == "cfm":
            if rate_cbm_per_hour == meaning.rate_cbm_per_hour:
                continue
            source_value = Decimal(text[start:end])
            quantum = Decimal(1).scaleb(source_value.as_tuple().exponent)
            presented = value.quantize(quantum)
            if presented * CFM_TO_CBM_PER_HOUR != rate_cbm_per_hour:
                raise UnrepresentableVentilationRate(
                    "sampled airflow cannot fit the source CFM precision without changing its rate"
                )
            rendered = format(presented, "f")
        else:
            rendered = str(value)
        result = result[:start] + rendered + result[end:]
    return result
