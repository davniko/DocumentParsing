"""Source-certified consignment PED references outside the extraction target.

The supported physical statement is the three-reference SEARA goods block. It
is intentionally not interpreted as a package or container assignment.
"""

from __future__ import annotations

import re
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

from document_ocr.synthesis.generators import DeterministicStream

from .models import SemanticBinding

_SURFACE = re.compile(
    r"PED\.\n(?P<base>[1-9][0-9]{5})\."
    r"(?P<second>[0-9]{2}), (?P<base2>[1-9][0-9]{5})\."
    r"(?P<third>[0-9]{2}), (?P<base3>[1-9][0-9]{5})\."
    r"(?P<first>[0-9]{2})"
)
_CONTEXT_BEFORE = re.compile(
    r"FREIGHT PREPAID AT ABROAD BY SEARA ALIMENTOS LTDA - BRAND: SEARA - $"
)
_CONTEXT_AFTER = " - NCM:"
_CARGO_OWNER = "cargo:0"


@dataclass(frozen=True, slots=True)
class PedReferenceList:
    base: int
    first_suffix: int


def parse_source_surface(text: str) -> PedReferenceList:
    match = _SURFACE.fullmatch(text)
    if match is None:
        raise ValueError("PED references lack the certified three-ID source surface")
    base = int(match["base"])
    first = int(match["first"])
    if (
        match["base2"] != match["base"]
        or match["base3"] != match["base"]
        or int(match["second"]) != first + 1
        or int(match["third"]) != first + 2
        or first > 97
    ):
        raise ValueError("PED references lack one shared base and consecutive suffixes")
    return PedReferenceList(base=base, first_suffix=first)


def validate_binding(
    binding: SemanticBinding,
    *,
    source: bytes | None = None,
    target: Mapping[str, Any] | None = None,
) -> PedReferenceList:
    """Prove a unique source-owned PED list with no training-label claim."""

    if (
        binding.value_kind != "ped_identifier_list"
        or binding.render_mode != "deterministic_auxiliary"
        or binding.group_kind != "cargo"
        or binding.group_key != _CARGO_OWNER
        or binding.target_paths
        or binding.target_relationship != "none"
        or binding.derivation is not None
        or binding.dependency_paths
        or binding.dependency_bindings
        or binding.source_relationships
        or len(binding.occurrences) != 1
        or binding.realization.mode != "generated_auxiliary"
        or binding.realization.adapter != "generated_auxiliary"
        or binding.realization.target_values
        or len(binding.realization.slots) != 1
    ):
        raise ValueError("PED list lacks a source-only consignment contract")
    slot = binding.occurrences[0]
    if (
        slot.target_paths
        or slot.semantic_role != _CARGO_OWNER
        or slot.evidence_origin != "audited_source_auxiliary"
        or slot.render_policy != "natural_text"
        or binding.realization.slots[0].slot_id != slot.slot_id
    ):
        raise ValueError("PED slot lacks audited goods ownership")
    parsed = parse_source_surface(slot.source_text)
    if source is not None:
        if source[slot.byte_start : slot.byte_end] != slot.source_text.encode():
            raise ValueError("PED slot differs from pinned source bytes")
        text = source.decode("utf-8")
        if text.count("PED.") != 1 or text.count(slot.source_text) != 1:
            raise ValueError("PED list is not the unique printed consignment statement")
        before = source[: slot.byte_start].decode("utf-8")
        after = source[slot.byte_end :].decode("utf-8")
        if _CONTEXT_BEFORE.search(before) is None or not after.startswith(_CONTEXT_AFTER):
            raise ValueError("PED references lack the certified source neighborhood")
    if target is not None:
        patch = target.get("documentPatch")
        groups = patch.get("cargoGroups") if isinstance(patch, Mapping) else None
        if not isinstance(groups, list) or len(groups) != 1 or not isinstance(groups[0], Mapping):
            raise ValueError("PED list has no unique goods owner")
        if any(
            isinstance(value, str) and "PED." in value
            for key in ("description", "additionalInformation", "marksAndNumbers")
            for value in ([groups[0].get(key)] if key == "description" else groups[0].get(key, []))
        ):
            raise ValueError("source-only PED reference remains in extraction target")
    return parsed


def render_binding(
    binding: SemanticBinding,
    *,
    stream: DeterministicStream,
    source: bytes,
    target: Mapping[str, Any],
) -> str:
    """Generate a fresh shared six-digit base and three related two-digit IDs."""

    previous = validate_binding(binding, source=source, target=target)
    draw = stream.derive(binding.logical_key)
    for counter in range(10_000):
        base = 100_000 + draw.randbelow(900_000, counter=counter * 2)
        first = draw.randbelow(98, counter=counter * 2 + 1)
        if base != previous.base or first != previous.first_suffix:
            return f"PED.\n{base}.{first + 1:02d}, {base}.{first + 2:02d}, {base}.{first:02d}"
    raise ValueError("PED reference generator cannot avoid the source list")
