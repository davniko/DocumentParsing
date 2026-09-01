"""Lightweight synthetic IMO and vessel-flag generation.

These fields train value recognition, not vessel-identity lookup.  IMO values
therefore use only the public checksum contract and collision guards.  Flags
are sampled from ISO countries that occur in the pinned maritime-port
whitelist; no real-world vessel-to-flag assertion is made.
"""

from __future__ import annotations

import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass

from document_ocr.synthesis.country_registry import CountryRegistry
from document_ocr.synthesis.generators import DeterministicStream
from document_ocr.synthesis.world_port_registry import WorldPortRecord


@dataclass(frozen=True, slots=True)
class SyntheticImoNumber:
    value: str
    attempts: int
    method: str = "random_six_digits_plus_imo_check_digit_v1"


@dataclass(frozen=True, slots=True)
class SyntheticVesselFlag:
    country_code: str
    printed_country: str
    method: str = "uniform_pinned_world_port_country_v1"


def imo_check_digit(six_digits: str) -> int:
    if len(six_digits) != 6 or not six_digits.isdigit():
        raise ValueError("IMO body must contain exactly six digits")
    return (
        sum(int(digit) * weight for digit, weight in zip(six_digits, range(7, 1, -1), strict=True))
        % 10
    )


def sample_imo_number(
    *,
    stream: DeterministicStream,
    excluded_values: Sequence[str],
    used_values: set[str],
    maximum_attempts: int,
    leading_digit_weights: Mapping[str, int],
) -> SyntheticImoNumber:
    if maximum_attempts <= 0:
        raise ValueError("IMO generation requires positive maximum attempts")
    if not leading_digit_weights or any(
        key not in {str(value) for value in range(1, 10)} or weight <= 0
        for key, weight in leading_digit_weights.items()
    ):
        raise ValueError("IMO leading-digit weights must be positive digits 1-9")
    invalid_excluded = tuple(
        value
        for value in excluded_values
        if re.fullmatch(r"[1-9][0-9]{6}", value) is None
        or imo_check_digit(value[:6]) != int(value[-1])
    )
    if invalid_excluded:
        raise ValueError("excluded IMO values must be checksum-valid seven-digit identities")
    excluded = frozenset(excluded_values)
    ordered = tuple(sorted(leading_digit_weights.items()))
    total = sum(weight for _, weight in ordered)
    for attempt in range(1, maximum_attempts + 1):
        attempt_stream = stream.derive(f"attempt-{attempt}")
        draw = attempt_stream.derive("leading").randbelow(total)
        cumulative = 0
        leading = ""
        for digit, weight in ordered:
            cumulative += weight
            if draw < cumulative:
                leading = digit
                break
        if not leading:
            raise RuntimeError("IMO leading-digit weights are inconsistent")
        body = leading + "".join(
            str(attempt_stream.derive(f"digit-{index}").randbelow(10)) for index in range(1, 6)
        )
        candidate = body + str(imo_check_digit(body))
        if candidate not in excluded and candidate not in used_values:
            return SyntheticImoNumber(value=candidate, attempts=attempt)
    raise ValueError("IMO generation exhausted its collision budget")


def maritime_flag_country_codes(world_ports: Sequence[WorldPortRecord]) -> tuple[str, ...]:
    codes = tuple(sorted({row.country_code for row in world_ports}))
    if not codes:
        raise ValueError("world-port whitelist contains no flag-country support")
    return codes


def sample_vessel_flag(
    *,
    country_codes: Sequence[str],
    countries: CountryRegistry,
    stream: DeterministicStream,
) -> SyntheticVesselFlag:
    codes = tuple(sorted(set(country_codes)))
    if not codes:
        raise ValueError("vessel-flag sampling requires maritime country support")
    for code in codes:
        countries.entry(code)
    selected = codes[stream.derive("vessel-flag").randbelow(len(codes))]
    return SyntheticVesselFlag(
        country_code=selected,
        printed_country=countries.printable_name(selected),
    )
