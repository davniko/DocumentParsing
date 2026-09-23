"""Explicit allocation-owned outer counts with printed per-package inner counts.

The inner value in ``17 PALLETS WITH 68 PIECES EACH`` is not a shipment-level
piece count. Keep that packaging detail in its observed prose and derive only
the outer count from the reviewed allocation owner. Repeated shared prose imposes
an explicit equality constraint on its owners before scenario sampling.
"""

from __future__ import annotations

import re
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

from .numeric_auxiliary import resolve

_FORM = re.compile(
    r"(?P<outer>[1-9][0-9]*)\s+PALLETS?\s+WITH\s+(?P<inner>[1-9][0-9]*)\s+PIECES?\s+EACH", re.I
)
_FIELD = re.compile(r"documentPatch\.cargoGroups\[(\d+)\]\.additionalInformation\[(\d+)\]")
_OWNER = re.compile(
    r"documentPatch\.cargoAllocationGroups\[(\d+)\]\.allocations\[(\d+)\]\.packageQuantity"
)


@dataclass(frozen=True)
class NestedPackageProse:
    target_path: str
    source_text: str
    owner_paths: tuple[str, ...]
    outer_span: tuple[int, int]


def compile_contracts(template: Any, source: Mapping[str, Any]) -> tuple[NestedPackageProse, ...]:
    if not any(
        _FORM.fullmatch(text)
        for group in source.get("documentPatch", {}).get("cargoGroups", ())
        for text in group.get("additionalInformation", ())
    ):
        return ()
    result = []
    for binding in template.bindings:
        fields = [p for p in binding.target_paths if _FIELD.fullmatch(p)]
        dependencies = getattr(binding, "dependency_paths", ())
        owners = tuple(p for p in dependencies if _OWNER.fullmatch(p))
        if not fields or not owners:
            continue
        forms = [_FORM.fullmatch(resolve(source, p)) for p in fields]
        if not any(forms):
            continue  # Other explicit cargo-text contracts belong to their own handlers.
        if len(fields) != 1 or len(owners) != len(dependencies) or len(set(owners)) != len(owners):
            raise ValueError(
                "nested packing prose requires one field and distinct explicit allocation owners"
            )
        path = fields[0]
        text = resolve(source, path)
        form = _FORM.fullmatch(text)
        if form is None:
            raise ValueError(
                "allocation-owned nested packing prose has an unsupported complete form"
            )
        if len(binding.occurrences) != len(owners) or any(
            s.source_text != text for s in binding.occurrences
        ):
            raise ValueError("nested packing prose occurrences do not cover exact reviewed owners")
        field_match = _FIELD.fullmatch(path)
        assert field_match is not None
        group = source["documentPatch"]["cargoGroups"][int(field_match[1])]["groupId"]
        for slot, owner in zip(binding.occurrences, owners, strict=True):
            match = _OWNER.fullmatch(owner)
            assert match is not None
            allocation_group = source["documentPatch"]["cargoAllocationGroups"][int(match[1])]
            allocation = allocation_group["allocations"][int(match[2])]
            if allocation_group["groupId"] != group or allocation["packageQuantity"] != int(
                form["outer"]
            ):
                raise ValueError(
                    "nested packing prose contradicts its source allocation quantity or cargo group"
                )
            # Exact previously certified identity spans prove the source row.
            preceding = [
                (s.byte_end, resolve(source, p))
                for b in template.bindings
                for p in b.target_paths
                if p.endswith(".containerNumber")
                for s in b.occurrences
                if s.byte_end <= slot.byte_start
            ]
            if not preceding:
                raise ValueError(
                    "nested packing prose has no preceding observed container identity"
                )
            closest = max(x[0] for x in preceding)
            identities = {identity for end, identity in preceding if end == closest}
            if identities != {allocation["containerNumber"]}:
                raise ValueError(
                    "nested packing prose allocation contradicts its printed container row"
                )
        result.append(NestedPackageProse(path, text, owners, form.span("outer")))
    if len({r.target_path for r in result}) != len(result):
        raise ValueError("nested packing prose field has competing ownership contracts")
    return tuple(result)


def allocation_equalities(template: Any, source: Mapping[str, Any]) -> dict[int, tuple[int, ...]]:
    """Return reviewed equal-count allocation sets for the pre-generation sampler."""
    result: dict[int, tuple[int, ...]] = {}
    for contract in compile_contracts(template, source):
        if len(contract.owner_paths) < 2:
            continue
        matches: list[re.Match[str]] = []
        for path in contract.owner_paths:
            match = _OWNER.fullmatch(path)
            assert match is not None
            matches.append(match)
        groups = {int(m[1]) for m in matches}
        if len(groups) != 1:
            raise ValueError("shared nested prose spans multiple allocation groups")
        group = groups.pop()
        indices = tuple(sorted(int(m[2]) for m in matches))
        if group in result and result[group] != indices:
            raise ValueError("overlapping nested packing equalities require a joint integer solver")
        result[group] = indices
    return result


def generate(template: Any, source: Mapping[str, Any], target: Mapping[str, Any]) -> dict[str, str]:
    updates = {}
    for contract in compile_contracts(template, source):
        counts = {resolve(target, p) for p in contract.owner_paths}
        if len(counts) != 1:
            raise ValueError("shared nested packing prose requires equal generated owner counts")
        count = counts.pop()
        if type(count) is not int or count <= 0:
            raise ValueError("nested packing outer count must be a positive integer")
        left, right = contract.outer_span
        updates[contract.target_path] = (
            contract.source_text[:left] + str(count) + contract.source_text[right:]
        )
    return updates


def validate(template: Any, source: Mapping[str, Any], target: Mapping[str, Any]) -> None:
    for path, text in generate(template, source, target).items():
        if resolve(target, path) != text:
            raise ValueError(
                "rendered nested packing prose disagrees with its exact allocation owner"
            )
