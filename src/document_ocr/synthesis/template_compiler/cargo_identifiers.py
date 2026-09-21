"""Source-proven vehicle serials embedded in otherwise linguistic cargo text.

The observed vehicle identity prefix stays fixed; only its six-digit production
serial varies. This preserves the source's make/model context. It is not a claim
of an authentic manufacturer-issued VIN or jurisdiction-specific check digit.
"""

from __future__ import annotations

import re
from collections import Counter
from collections.abc import Mapping, Sequence
from typing import Any

from document_ocr.synthesis.generators import (
    DeterministicStream,
    generate_from_surface_pattern,
    surface_pattern,
)

from .models import CertifiedSemanticTemplate
from .realization_contract import token_projection_intervals

_REFERENCE = re.compile(
    r"(?P<label>(?:ORD(?:ER)?|P\.?O|PART|C/P)\.?\s*(?:NO\.?|NUMBER)\s*:\s*)"
    r"(?P<value>[A-Z0-9][A-Z0-9./-]*)",
    re.IGNORECASE,
)


def labelled_references(target: Mapping[str, Any]) -> dict[str, str]:
    return {
        f"documentPatch.cargoGroups[{i}].marksAndNumbers[{j}]": text
        for i, group in enumerate(target["documentPatch"].get("cargoGroups", []))
        for j, text in enumerate(group.get("marksAndNumbers", []))
        if _REFERENCE.fullmatch(text)
    }


def reference_patterns(
    target: Mapping[str, Any], template: CertifiedSemanticTemplate
) -> dict[str, str]:
    """Intersect every printed projection before deciding which code characters vary."""
    references = labelled_references(target)
    mutable_by_value: dict[str, set[int]] = {}
    for path, text in references.items():
        match = _REFERENCE.fullmatch(text)
        assert match is not None
        owners = [b for b in template.bindings if path in b.target_paths]
        if not owners:
            raise ValueError("labelled reference has no compiled target owner")
        mutable = set(range(len(match["value"])))
        for binding in owners:
            intervals = token_projection_intervals(binding)
            if intervals is None:
                continue
            tokens = tuple(re.finditer(r"[A-Za-z0-9]+", text))
            positions = {
                i - match.start("value")
                for start, end in intervals
                for token in tokens[start:end]
                for i in range(token.start(), token.end())
            }
            mutable.intersection_update(positions)
        mutable_by_value.setdefault(match["value"], set(range(len(match["value"]))))
        mutable_by_value[match["value"]].intersection_update(mutable)
    result = {}
    for path, text in references.items():
        match = _REFERENCE.fullmatch(text)
        assert match is not None
        result[path] = "".join(
            surface_pattern(char) if i in mutable_by_value[match["value"]] else "\\" + char
            for i, char in enumerate(match["value"])
        )
    return result


def fixed_references(target: Mapping[str, Any], template: CertifiedSemanticTemplate) -> set[str]:
    return {
        path
        for path, pattern in reference_patterns(target, template).items()
        if not re.sub(r"\\.", "", pattern)
    }


def generate_references(
    target: Mapping[str, Any], stream: DeterministicStream, template: CertifiedSemanticTemplate
) -> dict[str, str]:
    result = {}
    patterns = reference_patterns(target, template)
    for path, text in labelled_references(target).items():
        match = _REFERENCE.fullmatch(text)
        assert match is not None
        value = match["value"]
        if not re.sub(r"\\.", "", patterns[path]):
            result[path] = text  # Explicit zero-mutable-character contract, before generation.
            continue
        replacement = generate_from_surface_pattern(
            pattern=patterns[path],
            stream=stream.derive("cargo-reference:" + value),
            additional_excluded=value,
        )
        result[path] = match["label"] + replacement
    return result


_VEHICLE = re.compile(
    r"\b(?:VIN|FIN|CHASSIS(?:\s+(?:NO|NUMBER))?)\s*[.:#-]*\s*"
    r"([A-HJ-NPR-Z0-9]{11}[0-9]{6})(?![A-Z0-9])",
    re.IGNORECASE,
)


def vehicle_ids(text: str) -> tuple[str, ...]:
    return tuple(m.group(1).upper() for m in _VEHICLE.finditer(text))


def conditioned_requests(
    fields: Sequence[Mapping[str, Any]], *, sample_id: str, seed: int
) -> tuple[Mapping[str, Any], ...]:
    stream = DeterministicStream(seed, "cargo-vehicle-serials-v1", sample_id)
    result = []
    for field in fields:
        source_ids = (
            vehicle_ids(str(field["source"]))
            if (
                "cargoFragment" in field
                or any(
                    re.fullmatch(r"documentPatch\.cargoGroups\[\d+\]\.description", p)
                    for p in field["paths"]
                )
            )
            else ()
        )
        if not source_ids:
            result.append(field)
            continue
        replacements = []
        for original in source_ids:
            # A nonzero modular offset guarantees novelty without retry loops.
            serial = (int(original[-6:]) + 1 + stream.derive(original).randbelow(999999)) % 1000000
            replacements.append(original[:11] + f"{serial:06d}")
        result.append(
            {
                **field,
                "requiredVehicleIdentifiers": replacements,
                "vehicleIdentityRule": (
                    "Use these host-generated FIN/VIN identifiers exactly, in source order. "
                    "Keep the source make/model, fuel, engine capacity and vehicle class; "
                    "vary commercial trim, condition or accessories, not manufacturer identity."
                ),
            }
        )
    return tuple(result)


def validate(field: Mapping[str, Any], value: str) -> None:
    if "requiredVehicleIdentifiers" in field and Counter(vehicle_ids(value)) != Counter(
        field["requiredVehicleIdentifiers"]
    ):
        raise ValueError("cargo vehicle identifiers differ from the host-generated serial contract")
