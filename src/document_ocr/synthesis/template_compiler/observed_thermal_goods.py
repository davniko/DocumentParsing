"""Explicit train-fit cold-chain scenarios for otherwise unclassified HS identities.

This is an empirical synthesis support policy, not an assertion that every good
under an HS code requires refrigeration. A newly admitted identity may only use
the exact setpoints observed with it, and still needs joint package/measurement
support. Registry-defined frozen/chilled identities keep their existing policy.
"""

from __future__ import annotations

import re
from collections import defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import replace
from typing import Any

from document_ocr.synthesis.hs_registry import UkGlobalTariffRegistry
from document_ocr.synthesis.semantic_completion_pipeline import _group_allocations
from document_ocr.synthesis.thermal_goods import (
    ThermalGoodsIdentity,
    ThermalGoodsSupport,
    ThermalProfile,
)


def extend_support(
    support: ThermalGoodsSupport,
    *,
    registry: UkGlobalTariffRegistry,
    fit_targets: Mapping[str, Mapping[str, Any]],
    bounds: Mapping[ThermalProfile, tuple[float, float]],
) -> ThermalGoodsSupport:
    known = {i.hs6 for i in (*support.frozen, *support.chilled)}
    registered = frozenset(registry.global_codes)
    observations: dict[str, dict[ThermalProfile, set[float]]] = defaultdict(
        lambda: defaultdict(set)
    )
    documents: dict[str, set[str]] = defaultdict(set)
    review: set[tuple[str, str]] = set()
    for sid, target in fit_targets.items():
        patch = target["documentPatch"]
        containers = {c["containerNumber"]: c for c in patch.get("containers", [])}
        groups = patch.get("cargoGroups", [])
        allocations = _group_allocations(patch)
        if len(groups) == 1:
            allocations = {groups[0]["groupId"]: tuple(containers)}
        for group in groups:
            if group.get("dangerousGoods"):
                continue
            numbers = allocations.get(group["groupId"], ())
            if not numbers:
                continue
            settings = [containers[n].get("temperatureSetpoint") for n in numbers]
            if any(s is None or s.get("unit") != "celsius" for s in settings):
                continue  # No complete observed Celsius scenario for this group.
            temperatures = {float(s["value"]) for s in settings if s is not None}
            profiles = [
                profile
                for profile, (low, high) in bounds.items()
                if all(low <= value <= high for value in temperatures)
            ]
            for raw in group.get("hsCodes", []):
                code = re.sub(r"\D", "", raw)[:6]
                if code in known:
                    continue
                if code not in registered:
                    review.add((code, "fit identity absent from pinned HS6 registry"))
                    continue
                if len(profiles) != 1:
                    review.add((code, "fit setpoints do not prove one configured profile"))
                    continue
                observations[code][profiles[0]].update(temperatures)
                documents[code].add(sid)
    additions: dict[ThermalProfile, list[ThermalGoodsIdentity]] = {"FROZEN": [], "CHILLED": []}
    for code, observed_profiles in sorted(observations.items()):
        if len(observed_profiles) != 1 or any(c == code for c, _ in review):
            review.add((code, "conflicting or unsupported fit thermal observations"))
            continue
        profile, values = next(iter(observed_profiles.items()))
        row = registry.require_global(code, on_date=registry.receipt.snapshot_date)
        additions[profile].append(
            ThermalGoodsIdentity(
                hs6=code,
                chapter_description=row.chapter_description,
                heading_description=row.heading_description,
                description=row.description,
                profile=profile,
                observed_setpoints_celsius=tuple(sorted(values)),
                fit_document_ids=tuple(sorted(documents[code])),
            )
        )
    added_codes = {i.hs6 for values in additions.values() for i in values}
    return replace(
        support,
        frozen=(*support.frozen, *additions["FROZEN"]),
        chilled=(*support.chilled, *additions["CHILLED"]),
        # Do not simultaneously call a fit-only cold-chain identity ambient.
        ambient=tuple(i for i in support.ambient if i.hs6 not in added_codes),
        observation_review=tuple(sorted(review)),
    )


def shared_setpoints(identities: Sequence[Mapping[str, Any]]) -> tuple[float, ...] | None:
    """All cargo sharing an active box must allow the SAME observed setpoint."""
    observations = [
        set(row["observedSetpointsCelsius"])
        for row in identities
        if row.get("observedSetpointsCelsius")
    ]
    if not observations:
        return None  # These identities use the registry-defined profile policy.
    allowed = set.intersection(*observations)
    if not allowed:
        raise ValueError("joint goods lack a common fit-observed cold-chain setpoint")
    return tuple(sorted(allowed))


def validate_setpoints(
    target: Mapping[str, Any], identities: Mapping[str, list[dict[str, Any]]]
) -> None:
    if not any(row.get("observedSetpointsCelsius") for rows in identities.values() for row in rows):
        return
    patch = target["documentPatch"]
    containers = {c["containerNumber"]: c for c in patch.get("containers", [])}
    groups = patch.get("cargoGroups", [])
    allocations = _group_allocations(patch)
    if len(groups) == 1:
        allocations = {groups[0]["groupId"]: tuple(containers)}
    for gid, values in identities.items():
        allowed = shared_setpoints(values)
        if allowed is None:
            continue
        numbers = allocations.get(gid, ())
        if not numbers:
            raise ValueError("fit-observed cold-chain goods lack owned active equipment")
        for number in numbers:
            setting = containers[number].get("temperatureSetpoint")
            if setting is None:
                raise ValueError("fit-observed cold-chain goods lost their setpoint")
            value, unit = setting["value"], setting["unit"]
            celsius = (
                value
                if unit == "celsius"
                else (value - 32) / 1.8
                if unit == "fahrenheit"
                else value - 273.15
                if unit == "kelvin"
                else None
            )
            if celsius is None or not any(abs(celsius - v) < 1e-8 for v in allowed):
                raise ValueError("cold-chain setpoint is outside the exact fit observations")
