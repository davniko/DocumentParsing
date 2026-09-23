"""Goods-first thermal profiles for coherent reefer synthesis.

Thermal requirements are derived from a sampled HS semantic identity before
equipment is selected.  The deterministic layer intentionally covers only
unambiguous registry wording:

* ``frozen`` goods; and
* animal/fish goods explicitly described as ``chilled``.

Fresh produce, pharmaceuticals, chocolate, and similar context-dependent goods
remain outside this classifier.  They can be added through a pinned commodity
profile table or the later agent layer without changing this interface.
"""

from __future__ import annotations

import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Literal

from document_ocr.synthesis.generators import DeterministicStream
from document_ocr.synthesis.hs_registry import HsGlobalSubheading, UkGlobalTariffRegistry

ThermalProfile = Literal["FROZEN", "CHILLED"]


@dataclass(frozen=True, slots=True)
class ThermalGoodsIdentity:
    hs6: str
    chapter_description: str
    heading_description: str
    description: str
    profile: ThermalProfile
    observed_setpoints_celsius: tuple[float, ...] = ()
    fit_document_ids: tuple[str, ...] = ()


@dataclass(frozen=True, slots=True)
class AmbientGoodsIdentity:
    hs6: str
    chapter_description: str
    heading_description: str
    description: str


@dataclass(frozen=True, slots=True)
class ThermalGoodsSupport:
    frozen: tuple[ThermalGoodsIdentity, ...]
    chilled: tuple[ThermalGoodsIdentity, ...]
    ambient: tuple[AmbientGoodsIdentity, ...]
    ambient_chapters: tuple[str, ...]
    ambient_headings: tuple[str, ...] = ()
    observation_review: tuple[tuple[str, str], ...] = ()

    def thermal_candidates(self, profile: ThermalProfile) -> tuple[ThermalGoodsIdentity, ...]:
        return self.frozen if profile == "FROZEN" else self.chilled


@dataclass(frozen=True, slots=True)
class TemperatureSetpoint:
    profile: ThermalProfile
    value: float
    unit: Literal["celsius"] = "celsius"


def classify_thermal_hs(row: HsGlobalSubheading) -> ThermalProfile | None:
    """Resolve the nearest unambiguous thermal wording in the HS hierarchy.

    A broad parent may contain both ``fresh/chilled`` and ``frozen`` while the
    leaf selects only one branch.  Concatenating the hierarchy would therefore
    misclassify entries such as fresh lamb as frozen.
    """

    for value in (row.description, row.heading_description, row.chapter_description):
        text = value.casefold()
        frozen = re.search(r"\bfrozen\b", text) is not None
        chilled = re.search(r"\bchilled\b", text) is not None
        fresh = re.search(r"\bfresh\b", text) is not None
        if frozen and not chilled and not fresh:
            return "FROZEN"
        if chilled and not frozen:
            return "CHILLED"
        if frozen or chilled:
            return None
    return None


def build_thermal_goods_support(
    *,
    registry: UkGlobalTariffRegistry,
    ambient_chapters: Sequence[str],
    ambient_headings: Sequence[str] = (),
) -> ThermalGoodsSupport:
    chapters = tuple(sorted(set(ambient_chapters)))
    headings = tuple(sorted(set(ambient_headings)))
    if any(re.fullmatch(r"[0-9]{4}", value) is None for value in headings):
        raise ValueError("ambient HS headings must be exact four-digit identities")
    if not chapters or any(re.fullmatch(r"[0-9]{2}", value) is None for value in chapters):
        raise ValueError("ambient HS chapters must be non-empty exact two-digit identities")
    frozen: list[ThermalGoodsIdentity] = []
    chilled: list[ThermalGoodsIdentity] = []
    ambient: list[AmbientGoodsIdentity] = []
    for code in registry.global_codes:
        row = registry.require_global(code, on_date=registry.receipt.snapshot_date)
        profile = classify_thermal_hs(row)
        if profile is not None:
            identity = ThermalGoodsIdentity(
                hs6=code,
                chapter_description=row.chapter_description,
                heading_description=row.heading_description,
                description=row.description,
                profile=profile,
            )
            (frozen if profile == "FROZEN" else chilled).append(identity)
        elif row.chapter_code in chapters or code[:4] in headings:
            ambient.append(
                AmbientGoodsIdentity(
                    hs6=code,
                    chapter_description=row.chapter_description,
                    heading_description=row.heading_description,
                    description=row.description,
                )
            )
    if not frozen or not chilled or not ambient:
        raise ValueError("thermal goods support is missing a required candidate pool")
    return ThermalGoodsSupport(
        frozen=tuple(frozen),
        chilled=tuple(chilled),
        ambient=tuple(ambient),
        ambient_chapters=chapters,
        ambient_headings=headings,
    )


def sample_thermal_profile(
    *,
    weights_permyriad: Mapping[ThermalProfile, int],
    stream: DeterministicStream,
) -> ThermalProfile:
    expected = {"FROZEN", "CHILLED"}
    if set(weights_permyriad) != expected or sum(weights_permyriad.values()) != 10_000:
        raise ValueError("thermal profile weights must name FROZEN and CHILLED and sum to 10000")
    if any(value < 0 for value in weights_permyriad.values()):
        raise ValueError("thermal profile weights cannot be negative")
    draw = stream.derive("thermal-profile").randbelow(10_000)
    return "FROZEN" if draw < weights_permyriad["FROZEN"] else "CHILLED"


def sample_thermal_goods(
    *,
    support: ThermalGoodsSupport,
    profile: ThermalProfile,
    stream: DeterministicStream,
) -> ThermalGoodsIdentity:
    candidates = support.thermal_candidates(profile)
    return candidates[stream.derive("thermal-hs6").randbelow(len(candidates))]


def sample_ambient_goods(
    *,
    support: ThermalGoodsSupport,
    stream: DeterministicStream,
) -> AmbientGoodsIdentity:
    return support.ambient[stream.derive("ambient-hs6").randbelow(len(support.ambient))]


def render_synthetic_hs_code(
    *,
    hs6: str,
    output_digits: int,
    stream: DeterministicStream,
) -> str:
    if re.fullmatch(r"[0-9]{6}", hs6) is None or not 6 <= output_digits <= 18:
        raise ValueError("synthetic HS output requires a six-digit parent and 6-18 digits")
    if output_digits == 6:
        return hs6
    suffix = "".join(
        str(stream.derive(f"suffix-{index}").randbelow(10)) for index in range(output_digits - 6)
    )
    return hs6 + suffix


def sample_temperature_setpoint(
    *,
    profile: ThermalProfile,
    stream: DeterministicStream,
    frozen_minimum_celsius: float,
    frozen_maximum_celsius: float,
    chilled_minimum_celsius: float,
    chilled_maximum_celsius: float,
    step_celsius: float,
) -> TemperatureSetpoint:
    bounds = (
        (frozen_minimum_celsius, frozen_maximum_celsius)
        if profile == "FROZEN"
        else (chilled_minimum_celsius, chilled_maximum_celsius)
    )
    minimum, maximum = bounds
    if not minimum <= maximum or step_celsius <= 0:
        raise ValueError("temperature bounds or step are invalid")
    steps = round((maximum - minimum) / step_celsius)
    if abs(minimum + steps * step_celsius - maximum) > 1e-9:
        raise ValueError("temperature range must be exactly divisible by its step")
    value = minimum + stream.derive("setpoint").randbelow(steps + 1) * step_celsius
    return TemperatureSetpoint(profile=profile, value=round(value, 6))
