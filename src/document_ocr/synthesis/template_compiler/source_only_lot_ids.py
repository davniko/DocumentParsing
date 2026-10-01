"""Source-certified goods-owned LOT lists that are not extraction target fields.

The supported physical form is intentionally narrow: one `LOT NO.` line with
three consecutive five-digit identifiers. A different line shape or owner must
be certified explicitly rather than silently copied or randomized as prose.
"""

from __future__ import annotations

import re
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

from document_ocr.synthesis.generators import DeterministicStream

from .models import SemanticBinding

_SOURCE_LINE = re.compile(r"LOT NO\. ([0-9]{5}), ([0-9]{5}), ([0-9]{5})")
_FIVE_DIGIT_TOKEN = re.compile(r"(?<![0-9])[0-9]{5}(?![0-9])")
_CARGO_OWNER = re.compile(r"cargo:([0-9]+)")
_CERTIFIED_GOODS_PRELUDE = (
    "3 X 20' DC CONTAINERS",
    "60 PACKAGES",
    "60,000KG FERRO MOLYBDENUM",
    "NET WEIGHT: 60,000KG",
    "GROSS WEIGHT: 60,960KG",
    "60 PALLETS ( 60 BAGS )",
    "HS CODE: 7202.7000",
)
_CERTIFIED_GOODS_HEADING = "Description of Packages and Goods"
_CERTIFIED_PARTICULARS_HEADING = "PARTICULARS FURNISHED BY SHIPPER"
_CERTIFIED_PACKAGE_COLUMN = "No.of Containers or Other Pkgs"
_CERTIFIED_GOODS_FOLLOWING = "TERMS OF DELIVERY ACCORDING TO INCOTERMS 2020"


@dataclass(frozen=True, slots=True)
class LotIdentifierList:
    source_values: tuple[int, int, int]
    cargo_group_index: int


def parse_source_line(source_text: str) -> tuple[int, int, int]:
    """Certify the complete source form and its three-ID sequence relation."""

    match = _SOURCE_LINE.fullmatch(source_text)
    if match is None:
        raise ValueError("LOT identifier list lacks the certified three-ID source surface")
    values = (int(match[1]), int(match[2]), int(match[3]))
    if values != tuple(range(values[0], values[0] + 3)):
        raise ValueError("LOT identifier list is not three consecutive IDs")
    return values


def validate_source_context(source: bytes, *, source_text: str) -> None:
    """Prove the LOT line is in one goods block, not a container row."""

    text = source.decode("utf-8")
    lines = text.splitlines()
    lot_lines = tuple(index for index, line in enumerate(lines) if "LOT NO." in line)
    if len(lot_lines) != 1 or lines[lot_lines[0]] != source_text:
        raise ValueError("LOT identifier list is not the unique goods-block line")
    index = lot_lines[0]
    earlier = lines[:index]
    section_proven = _CERTIFIED_GOODS_HEADING in earlier or (
        _CERTIFIED_PARTICULARS_HEADING in earlier
        and _CERTIFIED_PACKAGE_COLUMN in earlier
        and "Marks and Numbers" in earlier
    )
    if (
        index < len(_CERTIFIED_GOODS_PRELUDE)
        or tuple(lines[index - len(_CERTIFIED_GOODS_PRELUDE) : index]) != _CERTIFIED_GOODS_PRELUDE
        or index + 1 >= len(lines)
        or lines[index + 1] != _CERTIFIED_GOODS_FOLLOWING
        or not section_proven
    ):
        raise ValueError("LOT identifier list lacks its source-certified goods neighborhood")


def validate_binding(
    binding: SemanticBinding,
    *,
    source: bytes | None = None,
    target: Mapping[str, Any] | None = None,
) -> LotIdentifierList:
    """Require an independently owned physical goods line with no target claim."""

    owner = _CARGO_OWNER.fullmatch(binding.group_key)
    if (
        binding.value_kind != "lot_identifier_list"
        or binding.render_mode != "deterministic_auxiliary"
        or binding.group_kind != "cargo"
        or owner is None
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
        raise ValueError("LOT identifier list binding lacks a source-only goods contract")
    slot = binding.occurrences[0]
    if (
        slot.target_paths
        or slot.semantic_role != binding.group_key
        or slot.evidence_origin != "audited_source_auxiliary"
        or slot.render_policy != "natural_text"
        or binding.realization.slots[0].slot_id != slot.slot_id
    ):
        raise ValueError("LOT identifier list slot lacks audited cargo ownership")
    values = parse_source_line(slot.source_text)
    if source is not None:
        if source[slot.byte_start : slot.byte_end] != slot.source_text.encode():
            raise ValueError("LOT identifier list slot differs from pinned source bytes")
        validate_source_context(source, source_text=slot.source_text)
    index = int(owner[1])
    if target is not None:
        patch = target.get("documentPatch")
        groups = patch.get("cargoGroups") if isinstance(patch, Mapping) else None
        if not isinstance(groups, list) or len(groups) != 1 or index != 0:
            raise ValueError("LOT identifier list cargo owner is absent from target")
        group = groups[index]
        if not isinstance(group, Mapping):
            raise ValueError("LOT identifier list cargo owner is not a goods group")
        for field in ("additionalInformation", "marksAndNumbers"):
            entries = group.get(field, [])
            if not isinstance(entries, list):
                raise ValueError(f"LOT identifier list target {field} is not a list")
            if any(isinstance(value, str) and _SOURCE_LINE.fullmatch(value) for value in entries):
                raise ValueError("source-only LOT list remains duplicated in extraction target")
    return LotIdentifierList(source_values=values, cargo_group_index=index)


def render_binding(
    binding: SemanticBinding,
    *,
    stream: DeterministicStream,
    source: bytes,
    target: Mapping[str, Any],
) -> str:
    """Make one stable new consecutive triplet without reusing printed source IDs."""

    validate_binding(binding, source=source, target=target)
    reserved = set(_FIVE_DIGIT_TOKEN.findall(source.decode("utf-8")))
    values_stream = stream.derive(binding.logical_key)
    for counter in range(10_000):
        base = 10_000 + values_stream.randbelow(89_998, counter=counter)
        values = tuple(str(base + offset) for offset in range(3))
        if not reserved.intersection(values):
            return f"LOT NO. {values[0]}, {values[1]}, {values[2]}"
    raise ValueError("LOT identifier list cannot avoid source identifier collisions")
