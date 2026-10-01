"""Recognize shipment-specific vehicle evidence before commodity sampling.

This is a fail-closed classifier, not a vehicle scenario generator. Carrier
boilerplate that merely discusses vehicles is insufficient by itself; a VIN
caption, a vehicle package, or a vehicle-bearing cargo fact is required.
"""

from __future__ import annotations

import re
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

_IDENTITY = re.compile(r"(?<![A-Z0-9])[A-Z0-9]{17}(?![A-Z0-9])", re.I)
_CAPTION = re.compile(r"\b(?:VIN|FIN|CHASSIS)\b", re.I)
_VEHICLE = re.compile(r"\bVEHICLES?\b|\bVEHICLE\s*\(S\)", re.I)
_WHOLE = re.compile(
    r"\b(?:NEW|USED|SECONDHAND|PERSONAL|UNPACKED)\b.{0,90}?"
    r"\b(?:CAR(?:S)?|TRUCK(?:S)?|TRAILER(?:S)?|VEHICLE(?:S)?|MOTORCYCLE|MOTOR\s+BIKE)\b",
    re.I,
)
_PARTS = re.compile(r"\b(?:SPARE\s*PART(?:S|ES)|PARTS\s+(?:FOR|OF)|AUTO\s+PARTS)\b", re.I)
_VEHICLE_PARTS = re.compile(
    r"\b(?:CAR|TRUCK|VEHICLE)\s+(?:(?:AUTO|SPARE)\s+)?PARTS\b", re.I
)
_DISPLACEMENT = re.compile(r"\b(?:19|20)\d{2}\b.{0,40}?\b\d{3,5}\s*C[Cc]\b")
_VEHICLE_HS = re.compile(r"\bHS(?:\s+CODE)?\s*[:.#-]?\s*87(?:03|04|05|11|16)\b", re.I)
_UNPACKED_VEHICLE_CLAUSE = re.compile(
    r"\bCAR\s*\(S\)\s*,\s*VEHICLE\s*\(S\)\s*,\s*UNPACKED\b", re.I
)
_NEW_USED_UNITS = re.compile(r"\b\d+\s+(?:NEW|USED)\s+UNITS?\b", re.I)


@dataclass(frozen=True, slots=True)
class VehicleSourceEvidence:
    reasons: tuple[str, ...]
    identities: tuple[str, ...]


def vehicle_source_evidence(
    source: str, target: Mapping[str, Any]
) -> VehicleSourceEvidence | None:
    """Find concrete vehicle facts that generic goods draws cannot preserve.

    A 17-character mark alone is not enough: industrial cargo also prints long
    opaque references. Conversely, European VINs may contain letters in the
    last six positions, so we do not impose a North-American serial grammar.
    """
    patch = target.get("documentPatch", {})
    groups = patch.get("cargoGroups", ())
    packages = patch.get("cargoPackages", ())
    reasons: set[str] = set()
    identities: set[str] = set()

    lines = source.splitlines()
    for index, line in enumerate(lines):
        caption = _CAPTION.search(line)
        if caption is None:
            continue
        # Some forms print a CHASSIS NUMBER heading directly above the value.
        window = " ".join(lines[index : index + 3])
        for match in _IDENTITY.finditer(window):
            if 0 <= match.start() - caption.end() <= 120:
                reasons.add("printed_vehicle_identity_caption")
                identities.add(match.group().upper())

    if any(p.get("typeCategory") == "PACKAGE_VEHICLE" for p in packages):
        reasons.add("vehicle_package_category")
    if (
        _UNPACKED_VEHICLE_CLAUSE.search(source)
        and _NEW_USED_UNITS.search(source)
        and any(p.get("typeCategory") == "PACKAGE_UNIT" for p in packages)
    ):
        reasons.add("printed_new_or_used_units_under_vehicle_clause")

    descriptions = [g.get("description", "") for g in groups]
    for description in descriptions:
        if not isinstance(description, str):
            continue
        for match in _WHOLE.finditer(description):
            phrase = match.group()
            if not _PARTS.search(phrase) and not _VEHICLE_PARTS.search(
                description[match.start() : match.end() + 12]
            ):
                reasons.add("whole_vehicle_cargo_description")
                break

    source_identities = {m.group().upper() for m in _IDENTITY.finditer(source)}
    marks = {
        mark.upper()
        for group in groups
        for mark in group.get("marksAndNumbers", ())
        if isinstance(mark, str) and _IDENTITY.fullmatch(mark)
    }
    owned_marks = source_identities & marks
    if owned_marks and (
        _VEHICLE.search(source)
        or _VEHICLE_HS.search(source)
        or any(
            isinstance(d, str) and _DISPLACEMENT.search(d)
            for d in descriptions
        )
    ):
        reasons.add("vehicle_context_for_printed_seventeen_character_mark")
        identities.update(owned_marks)

    if not reasons:
        return None
    return VehicleSourceEvidence(tuple(sorted(reasons)), tuple(sorted(identities)))
