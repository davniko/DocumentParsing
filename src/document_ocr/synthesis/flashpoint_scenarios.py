"""Conditional synthetic flashpoints for maritime dangerous goods.

The value is treated as a synthetic formulation property, not as a registry
fact attached to a UN number.  Class-3 packing-group bounds come from IMDG
2.3.2.6.  Subsidiary class-3 hazards use the broader flammable-liquid range
because the primary hazard can determine the packing group.
"""

from __future__ import annotations

import re
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Literal

from document_ocr.synthesis.generators import DeterministicStream


@dataclass(frozen=True, slots=True)
class SyntheticFlashpoint:
    value: float
    unit: Literal["celsius"]
    basis: Literal[
        "primary_class3_packing_group_bound_v1",
        "class3_broad_formulation_range_v1",
        "non_class3_explicit_liquid_above_class3_threshold_v1",
        "desensitized_flammable_solid_formulation_range_v1",
    ]
    eligibility: Literal[
        "CLASS3_FLAMMABLE_LIQUID",
        "NON_CLASS3_EXPLICIT_LIQUID",
        "DESENSITIZED_FLAMMABLE_SOLID",
    ]


FlashpointEligibility = Literal[
    "CLASS3_FLAMMABLE_LIQUID",
    "NON_CLASS3_EXPLICIT_LIQUID",
    "DESENSITIZED_FLAMMABLE_SOLID",
]


def _contains_class3(value: object) -> bool:
    return value == "FLAMMABLE_LIQUIDS"


def classify_flashpoint_eligibility(
    *,
    proper_shipping_name: str,
    hazard_category: str | None,
    subsidiary_hazard_categories: tuple[str, ...],
) -> FlashpointEligibility | None:
    """Resolve only explicit hazard or physical-form evidence.

    This is not a substance-to-property lookup.  It identifies formulations
    for which sampling a printed flashpoint is coherent while leaving the
    numeric property independent of the UN identity.
    """

    if _contains_class3(hazard_category) or any(
        _contains_class3(value) for value in subsidiary_hazard_categories
    ):
        return "CLASS3_FLAMMABLE_LIQUID"
    normalized = " ".join(proper_shipping_name.upper().split())
    if re.search(r"\b(?:LIQUID|SOLUTION)\b", normalized):
        return "NON_CLASS3_EXPLICIT_LIQUID"
    if hazard_category == "FLAMMABLE_SOLIDS" and re.search(
        r"\b(?:WETTED|DAMPENED|DAMPED|ALCOHOL)\b",
        normalized,
    ):
        return "DESENSITIZED_FLAMMABLE_SOLID"
    return None


def sample_flashpoint(
    *,
    proper_shipping_name: str,
    hazard_category: str | None,
    subsidiary_hazard_categories: tuple[str, ...],
    packing_group_category: str | None,
    stream: DeterministicStream,
    presence_permyriad: Mapping[FlashpointEligibility, int],
    class3_minimum_celsius: float,
    class3_maximum_celsius: float,
    non_class3_liquid_minimum_celsius: float,
    non_class3_liquid_maximum_celsius: float,
    desensitized_solid_minimum_celsius: float,
    desensitized_solid_maximum_celsius: float,
    step_celsius: float,
) -> SyntheticFlashpoint | None:
    """Generate a noisy, formulation-level value when semantics allow it."""

    is_primary = _contains_class3(hazard_category)
    eligibility = classify_flashpoint_eligibility(
        proper_shipping_name=proper_shipping_name,
        hazard_category=hazard_category,
        subsidiary_hazard_categories=subsidiary_hazard_categories,
    )
    if eligibility is None:
        return None
    expected = {
        "CLASS3_FLAMMABLE_LIQUID",
        "NON_CLASS3_EXPLICIT_LIQUID",
        "DESENSITIZED_FLAMMABLE_SOLID",
    }
    if set(presence_permyriad) != expected or any(
        not 0 <= value <= 10_000 for value in presence_permyriad.values()
    ):
        raise ValueError("flashpoint presence must cover every eligibility class")
    if stream.derive("presence").randbelow(10_000) >= presence_permyriad[eligibility]:
        return None
    if step_celsius <= 0:
        raise ValueError("flashpoint step must be positive")

    basis: Literal[
        "primary_class3_packing_group_bound_v1",
        "class3_broad_formulation_range_v1",
        "non_class3_explicit_liquid_above_class3_threshold_v1",
        "desensitized_flammable_solid_formulation_range_v1",
    ]
    if eligibility == "NON_CLASS3_EXPLICIT_LIQUID":
        lower, upper = (
            non_class3_liquid_minimum_celsius,
            non_class3_liquid_maximum_celsius,
        )
        basis = "non_class3_explicit_liquid_above_class3_threshold_v1"
    elif eligibility == "DESENSITIZED_FLAMMABLE_SOLID":
        lower, upper = (
            desensitized_solid_minimum_celsius,
            desensitized_solid_maximum_celsius,
        )
        basis = "desensitized_flammable_solid_formulation_range_v1"
    else:
        lower, upper = class3_minimum_celsius, class3_maximum_celsius
        basis = "class3_broad_formulation_range_v1"
    if (
        eligibility == "CLASS3_FLAMMABLE_LIQUID"
        and is_primary
        and packing_group_category == "MEDIUM_DANGER"
    ):
        upper = min(upper, 23.0 - step_celsius)
        basis = "primary_class3_packing_group_bound_v1"
    elif (
        eligibility == "CLASS3_FLAMMABLE_LIQUID"
        and is_primary
        and packing_group_category == "LOW_DANGER"
    ):
        lower = max(lower, 23.0)
        upper = min(upper, 60.0)
        basis = "primary_class3_packing_group_bound_v1"
    elif eligibility == "CLASS3_FLAMMABLE_LIQUID":
        upper = min(upper, 60.0)
        basis = "class3_broad_formulation_range_v1"
    if lower > upper:
        raise ValueError("flashpoint packing-group bounds have no configured support")
    steps = round((upper - lower) / step_celsius)
    if abs(lower + steps * step_celsius - upper) > 1e-9:
        raise ValueError("flashpoint range must be exactly divisible by its step")
    value = lower + stream.derive("value").randbelow(steps + 1) * step_celsius
    return SyntheticFlashpoint(
        value=round(value, 6),
        unit="celsius",
        basis=basis,
        eligibility=eligibility,
    )
