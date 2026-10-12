"""Reviewed equipment aliases applied only to complete, source-owned expressions.

Sampling chooses wording independently of cargo donors. Rendering retains the
source's explicit counts and movement framing, but never converts a size-only
fragment into a complete type declaration or changes unrelated containers.
"""

from __future__ import annotations

import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from functools import lru_cache
from typing import TYPE_CHECKING, Any, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from document_ocr.label_schemas.bill_of_lading_v5 import (
    ContainerSizeCategory,
    ContainerTypeCategory,
)
from document_ocr.synthesis.container_semantics import (
    _COUNTABLE_EQUIPMENT_START,
    _OWNED_COUNT_PREFIX,
    _TERMINAL_MOVEMENT,
    _normalize,
    review_source_equipment_surface,
)
from document_ocr.synthesis.curated_templates import current_path
from document_ocr.synthesis.generators import DeterministicStream

if TYPE_CHECKING:
    from document_ocr.synthesis.curated_templates import SamplingBlueprint

_EQUIPMENT_PATH = re.compile(
    r"^documentPatch\.containerInformation\[(\d+)\]\."
    r"(typeDescription|sizeCategory|typeCategory)$"
)
_SINGLE_MOVEMENT = re.compile(r"[ \t]+(?:FCL|LCL|CY)[ \t]*$", re.I)
_UNFRAMED_COUNT = re.compile(
    rf"^[1-9][0-9]*(?:\s*[Xx]\s*|\s+)(?=\(?{_COUNTABLE_EQUIPMENT_START})",
    re.I,
)
_PARTIAL = re.compile(
    r"(?:20|40|45)(?:\s*(?:FT|FOOT|FEET))?|"
    r"HC|HQ|HICU|HI CUBE|HIGH CUBE|HIGHCUBE|8 6|9 6"
)


def validate_equipment_alias(alias: str, size: str, kind: str) -> None:
    """Require a complete matching pair; never silently substitute an alias."""
    observed = review_source_equipment_surface(alias, temperature_present=False)
    if (observed.size_category, observed.type_category) != (size, kind):
        raise ValueError(
            f"equipment alias {alias!r} conflicts with {(size, kind)!r}: "
            f"{(observed.size_category, observed.type_category)!r}; {observed.review_rule}"
        )


class EquipmentAliasSet(BaseModel):
    """Unweighted reviewed wordings for one full canonical equipment pair."""

    model_config = ConfigDict(extra="forbid", frozen=True)
    sizeCategory: ContainerSizeCategory
    typeCategory: ContainerTypeCategory
    aliases: tuple[str, ...] = Field(min_length=1)

    @model_validator(mode="after")
    def reviewed_aliases(self) -> EquipmentAliasSet:
        if len(set(self.aliases)) != len(self.aliases):
            raise ValueError("equipment registry contains duplicate aliases")
        for alias in self.aliases:
            if not alias or alias != alias.strip() or "\n" in alias or "\r" in alias:
                raise ValueError("equipment alias must be one nonempty trimmed line")
            frame = _frame(alias)
            if frame.prefix or frame.suffix:
                raise ValueError(
                    "equipment registry aliases cannot own counts or documentary framing"
                )
            validate_equipment_alias(alias, self.sizeCategory, self.typeCategory)
        return self


class EquipmentAliasRegistry(BaseModel):
    """Pinned registry of reviewed aliases; absence is an error, not a fallback."""

    model_config = ConfigDict(extra="forbid", frozen=True)
    version: Literal[1]
    entries: tuple[EquipmentAliasSet, ...] = Field(min_length=1)

    @model_validator(mode="after")
    def unique_pairs(self) -> EquipmentAliasRegistry:
        pairs = [(entry.sizeCategory, entry.typeCategory) for entry in self.entries]
        if len(set(pairs)) != len(pairs):
            raise ValueError("equipment registry contains duplicate canonical pairs")
        return self

    @classmethod
    def from_data(cls, data: Mapping[str, Any]) -> EquipmentAliasRegistry:
        return cls.model_validate(data)

    def for_pair(self, size: str, kind: str) -> tuple[str, ...]:
        for entry in self.entries:
            if (entry.sizeCategory, entry.typeCategory) == (size, kind):
                return entry.aliases
        raise ValueError(f"equipment alias registry has no entry for {(size, kind)!r}")

    def select(self, size: str, kind: str, stream: DeterministicStream) -> str:
        aliases = self.for_pair(size, kind)
        return aliases[stream.randbelow(len(aliases))]


@dataclass(frozen=True, slots=True)
class _EquipmentFrame:
    core: str
    partial: bool
    prefix: str
    suffix: str


@lru_cache(maxsize=8192)
def _frame(original: str) -> _EquipmentFrame:
    leading = original[: len(original) - len(original.lstrip())]
    trailing = original[len(original.rstrip()) :]
    core = original.strip()
    opening = closing = ""
    if core.startswith("(") and core.endswith(")") and core.count("(") == core.count(")") == 1:
        opening, closing = "(", ")"
        core = core[1:-1]
    movement = _TERMINAL_MOVEMENT.search(core) or _SINGLE_MOVEMENT.search(core)
    movement_surface = movement[0] if movement else ""
    if movement:
        core = core[: movement.start()]
    counted = _OWNED_COUNT_PREFIX.match(core)
    count_surface = counted[0] if counted else ""
    if counted:
        core = core[counted.end() :]
    elif _UNFRAMED_COUNT.match(core):
        raise ValueError("equipment count has unsupported source framing")
    return _EquipmentFrame(
        core,
        _PARTIAL.fullmatch(_normalize(core)) is not None,
        leading + opening + count_surface,
        movement_surface + closing + trailing,
    )


def apply_equipment_aliases(
    blueprint: SamplingBlueprint,
    target: Mapping[str, Any],
    surfaces: Mapping[str, str | Sequence[str]],
    alias: str,
) -> tuple[dict[str, str | list[str]], list[dict[str, Any]]]:
    """Replace complete bound equipment surfaces and return an ownership receipt.

    Input mappings and targets remain unchanged. Partial/count-only owners keep
    their ordinary-renderer output. A requested alias with no eligible owner
    fails explicitly; shared complete owners cannot describe mixed pairs.
    """
    result = {
        key: values if isinstance(values, str) else list(values) for key, values in surfaces.items()
    }
    old = blueprint.target["documentPatch"].get("containerInformation", [])
    new = target["documentPatch"].get("containerInformation", [])
    active = {region.key for region in blueprint.regions if region.curated_key is None}
    receipts = []
    for key, binding in blueprint.historical_bindings.items():
        if key not in active:
            continue
        paths = [current_path(path) for path in binding.get("target_paths", ())]
        fields: dict[int, set[str]] = {}
        for path in paths:
            if match := _EQUIPMENT_PATH.fullmatch(path):
                fields.setdefault(int(match[1]), set()).add(match[2])
        indices = sorted(
            index
            for index, owned in fields.items()
            if "typeDescription" in owned or {"sizeCategory", "typeCategory"} <= owned
        )
        if not indices:
            continue
        originals = [o["source_text"] for o in binding.get("occurrences", ())]
        frames = [_frame(original) for original in originals]
        reviews = [
            review_source_equipment_surface(frame.core, temperature_present=False)
            for frame in frames
        ]
        complete = [
            i
            for i, (frame, review) in enumerate(zip(frames, reviews, strict=True))
            if not frame.partial
            and review.size_category is not None
            and review.type_category is not None
        ]
        if not complete:
            # A complete typeDescription owner containing a bare length gets
            # the agreed standard-box default, unlike a size-only owner.
            complete = [
                i
                for i, review in enumerate(reviews)
                if review.size_category is not None and review.type_category is not None
            ]
        if not complete:
            continue
        if any(not frame.partial and i not in complete for i, frame in enumerate(frames)):
            raise ValueError(f"unreviewed mixed equipment occurrences in {key}")
        source_pairs = {(reviews[i].size_category, reviews[i].type_category) for i in complete}
        if len(source_pairs) != 1:
            raise ValueError(f"incompatible complete equipment occurrences in {key}")
        if any(index >= len(old) or index >= len(new) for index in indices):
            raise ValueError(f"equipment owner index out of range in {key}")
        if any(
            (old[index].get("sizeCategory"), old[index].get("typeCategory")) not in source_pairs
            for index in indices
        ):
            raise ValueError(f"complete printed equipment conflicts with its source label in {key}")
        rendered = result.get(key)
        rendered = [rendered] if isinstance(rendered, str) else rendered
        if rendered is None or len(rendered) != len(originals):
            raise ValueError(
                f"ordinary renderer lacks occurrence-aligned equipment surfaces for {key}"
            )
        pairs = {
            (new[index].get("sizeCategory"), new[index].get("typeCategory")) for index in indices
        }
        if len(pairs) != 1:
            raise ValueError(f"one equipment type owner refers to unequal sampled types: {key}")
        size, kind = next(iter(pairs))
        validate_equipment_alias(alias, size, kind)
        variants = list(rendered)
        for i in complete:
            variants[i] = frames[i].prefix + alias + frames[i].suffix
        result[key] = variants
        receipts.append(
            {
                "owner": key,
                "kind": "equipment",
                "values": variants,
                "paths": paths,
                "referencedContainerIndices": indices,
                "changedOccurrences": complete,
                "preservedPartialOccurrences": [
                    i for i in range(len(originals)) if i not in complete
                ],
            }
        )
    if not receipts:
        raise ValueError("requested equipment alias has no complete owned template surface")
    return result, receipts
