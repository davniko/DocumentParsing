"""Source-proven package interval arithmetic, independent of linguistic generation."""

from __future__ import annotations

import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

from document_ocr.synthesis.rendering import render_number_surface

from .coherence import _resolve_path, inclusive_range_surfaces
from .models import CertifiedSemanticTemplate, SemanticBinding


def render_composite_range(
    binding: SemanticBinding,
    source_target: Mapping[str, Any],
    target: Mapping[str, Any],
    source: bytes,
) -> dict[str, str] | None:
    """Prove caption + interval jointly expresses marks, package type and count.

    Compiled value slots can exclude a static caption. The source bytes, not a
    guess about a logical-key name, must prove that caption belongs to the value.
    """
    from .descendant import _string_semantics_match

    mark_paths = [
        p
        for p in binding.target_paths
        if re.fullmatch(r"documentPatch\.cargoGroups\[\d+\]\.marksAndNumbers\[\d+\]", p)
    ]
    quantity_paths = [
        p
        for p in binding.target_paths
        if re.fullmatch(
            r"documentPatch\.(?:cargoPackages\[\d+\]\.quantity|"
            r"cargoAllocationGroups\[\d+\]\.allocations\[\d+\]\.packageQuantity)",
            p,
        )
    ]
    categories = [
        p
        for p in binding.target_paths
        if re.fullmatch(r"documentPatch\.cargoPackages\[\d+\]\.typeCategory", p)
    ]
    if (
        len(mark_paths) != 1
        or not quantity_paths
        or set(binding.target_paths) != set(mark_paths + quantity_paths + categories)
    ):
        return None
    old, new = _text(source_target, mark_paths[0]), _text(target, mark_paths[0])
    old_parts, new_parts = inclusive_range_surfaces(old), inclusive_range_surfaces(new)
    if len(old_parts) != 1 or len(new_parts) != 1 or not formal_range_text(old):
        return None
    a, b = old_parts[0], new_parts[0]
    prefix, suffix = old[: a.char_start], old[a.char_end :]
    if new[: b.char_start] != prefix or new[b.char_end :] != suffix:
        raise ValueError("composite range changed its source-proven caption")
    if any(
        _quantity(source_target, p) != a.cardinality or _quantity(target, p) != b.cardinality
        for p in quantity_paths
    ):
        raise ValueError("composite range and package cardinality disagree")
    if any(
        _resolve_path(source_target, p) != _resolve_path(target, p)
        or not _string_semantics_match(_text(source_target, p), old)
        for p in categories
    ):
        raise ValueError("composite range category is not proven by its caption")
    old_range, new_range = old[a.char_start : a.char_end], new[b.char_start : b.char_end]
    replacements = {}
    for slot in binding.occurrences:
        if slot.source_text == old:
            replacements[slot.slot_id] = new
        elif (
            slot.source_text == old_range
            and source[: slot.byte_start].rstrip().endswith(prefix.rstrip().encode())
            and (
                not suffix or source[slot.byte_end :].lstrip().startswith(suffix.lstrip().encode())
            )
        ):
            replacements[slot.slot_id] = new_range
        else:
            return None
    return replacements


@dataclass(frozen=True)
class RangePlan:
    target_values: Mapping[str, str]
    auxiliary_surfaces: Mapping[str, Mapping[str, str]]


def _text(target: Mapping[str, Any], path: str) -> str:
    value = _resolve_path(target, path)
    if not isinstance(value, str):
        raise ValueError("range text path is not a string")
    return value


def _quantity(target: Mapping[str, Any], path: str) -> int:
    value = _resolve_path(target, path)
    if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
        raise ValueError("range dependency is not a positive integer")
    return value


def _shift_interval_numbers(text: str, delta: int) -> str:
    result = text
    for part in reversed(inclusive_range_surfaces(text)):
        for start, end, number in (
            (part.end_start, part.end_end, part.end),
            (part.start_start, part.start_end, part.start),
        ):
            surface = text[start:end]
            value = render_number_surface(surface, number, number + delta)
            if surface.isdigit():
                value = value.zfill(len(surface))
            result = result[:start] + value + result[end:]
    return result


def formal_range_text(text: str) -> bool:
    """Only arithmetic labels/captions, not product names or party-dependent prose."""
    residue = text
    for part in reversed(inclusive_range_surfaces(text)):
        residue = residue[: part.char_start] + " " + residue[part.char_end :]
    words = re.findall(r"[A-Za-z]+", residue.upper())
    return bool(inclusive_range_surfaces(text)) and all(
        word
        in {
            "PALLET",
            "PALLETS",
            "PLT",
            "CASE",
            "CASES",
            "NO",
            "NOS",
            "C",
            "P",
            "W",
            "L",
            "N",
            "NUMBER",
            "NUMBERS",
            "PKG",
            "PKGS",
            "BUNDLES",
            "BUNDLE",
        }
        for word in words
    )


def condition_lexical_ranges(
    template: CertifiedSemanticTemplate,
    source_target: Mapping[str, Any],
    target: Mapping[str, Any],
    fields: Sequence[Mapping[str, Any]],
) -> tuple[Mapping[str, Any], ...]:
    """Separate a new shipping-mark identity from its proven interval math.

    Complete identity + package-number-caption + interval fields need no
    linguistic inference: a synthetic mark identifies the shipment, while
    the existing range solver supplies every endpoint.
    Arbitrary prose, unproven intervals and existing projection frames are not
    reinterpreted as this grammar.
    """
    ranges = plan_ranges(template, source_target, target).target_values
    caption = re.compile(r"\s+(?:(?:C|CASE|PALLET|PACKAGE)(?:/|\s+)NO\.?\s*:?\s*)$", re.I)
    result = []
    for field in fields:
        paths = field["paths"]
        if (
            not paths
            or "hostAssembly" in field
            or not all(
                re.fullmatch(r"documentPatch\.cargoGroups\[\d+\]\.marksAndNumbers\[\d+\]", p)
                and p in ranges
                for p in paths
            )
        ):
            result.append(field)
            continue
        old = field["source"]
        intervals = inclusive_range_surfaces(old)
        match = caption.search(old[: intervals[0].char_start]) if intervals else None
        if (
            match is None
            or not old[: match.start()].strip()
            or not formal_range_text(old[match.start() :])
        ):
            result.append(field)
            continue
        values = {ranges[p] for p in paths}
        if len(values) != 1:
            raise ValueError("one lexical range field has inconsistent interval owners")
        new = values.pop()
        identity = old[: match.start()]
        if not new.startswith(identity):
            raise ValueError("range solver changed its unowned linguistic identity")
        result.append(
            {
                **field,
                "hostRangeIdentity": True,
                "hostAssembly": dict(
                    prefix="",
                    suffix=new[len(identity) :],
                    mutableSource=identity,
                    minimumWords=1,
                ),
            }
        )
    return tuple(result)


def _apportion(total: int, weights: Sequence[int]) -> list[int]:
    if total < len(weights):
        raise ValueError("range cardinalities cannot preserve positive source support")
    if total > sum(weights):
        parts = [divmod((total - sum(weights)) * weight, sum(weights)) for weight in weights]
        result = [weight + quotient for weight, (quotient, _) in zip(weights, parts, strict=True)]
        for index in sorted(range(len(weights)), key=lambda i: (-parts[i][1], i))[
            : total - sum(result)
        ]:
            result[index] += 1
        return result
    values = [1] * len(weights)
    remaining = total - len(weights)
    while remaining:
        eligible = [i for i, weight in enumerate(weights) if values[i] < weight]
        capacity = sum(weights[i] - values[i] for i in eligible)
        fractions = {i: divmod(remaining * (weights[i] - values[i]), capacity) for i in eligible}
        added = sum(q for q, _ in fractions.values())
        for i, (q, _) in fractions.items():
            values[i] += q
        remaining -= added
        for i in sorted(eligible, key=lambda i: (-fractions[i][1], i)):
            if not remaining:
                break
            if values[i] < weights[i]:
                values[i] += 1
                remaining -= 1
    return values


def plan_ranges(
    template: CertifiedSemanticTemplate,
    source_target: Mapping[str, Any],
    target: Mapping[str, Any],
) -> RangePlan:
    constraints = [
        c
        for c in template.coherence_constraints
        if c.kind
        in {
            "inclusive_range_cardinality",
            "aggregate_inclusive_range_cardinality",
        }
    ]
    if not constraints:
        return RangePlan({}, {})
    keys = {k for c in constraints for k in c.member_logical_keys}
    bindings = {b.logical_key: b for b in template.bindings if b.logical_key in keys}
    texts: dict[str, tuple[str, ...]] = {}
    paths: dict[str, tuple[str, ...]] = {}
    weights: dict[tuple[str, int, int], int] = {}
    for key, binding in bindings.items():
        selected = tuple(
            p
            for p in binding.target_paths
            if isinstance(_resolve_path(source_target, p), str)
            and inclusive_range_surfaces(_text(source_target, p))
        )
        paths[key] = selected
        texts[key] = (
            tuple(dict.fromkeys(_text(source_target, p) for p in selected))
            if selected
            else tuple(dict.fromkeys(s.source_text for s in binding.occurrences))
        )
        if binding.target_paths and not selected:
            raise ValueError("range binding has no complete target interval evidence")
        for index, text in enumerate(texts[key]):
            parts = inclusive_range_surfaces(text)
            if not parts:
                raise ValueError("range source-only binding has incomplete interval evidence")
            for number, part in enumerate(parts):
                weights[key, index, number] = part.cardinality
    relations = []
    for constraint in constraints:
        members = frozenset(n for n in weights if n[0] in constraint.member_logical_keys)
        if constraint.kind == "inclusive_range_cardinality" and len(members) != 1:
            raise ValueError("single interval contract has ambiguous multiple interval evidence")
        old = sum(_quantity(source_target, p) for p in constraint.dependency_paths)
        new = sum(_quantity(target, p) for p in constraint.dependency_paths)
        if sum(weights[n] for n in members) != old:
            raise ValueError("range dependency does not reproduce source arithmetic")
        relations.append((members, new))
    for i, (left, _) in enumerate(relations):
        for right, _ in relations[i + 1 :]:
            if left & right and not (left <= right or right <= left):
                raise ValueError("crossing package range contracts need source relationship review")
    assigned: dict[tuple[str, int, int], int] = {}
    for members, total in sorted(relations, key=lambda row: (len(row[0]), sorted(row[0]))):
        free = sorted(members - assigned.keys())
        remaining = total - sum(assigned[n] for n in members & assigned.keys())
        if not free:
            if remaining:
                raise ValueError("package range dependency totals disagree")
            continue
        for node, value in zip(
            free, _apportion(remaining, [weights[n] for n in free]), strict=True
        ):
            assigned[node] = value
    rewritten: dict[str, dict[str, str]] = {}
    for key, originals in texts.items():
        rewritten[key] = {}
        for index, original in enumerate(originals):
            text = original
            for number, part in reversed(tuple(enumerate(inclusive_range_surfaces(original)))):
                end = part.start + (1 if part.end >= part.start else -1) * (
                    assigned[key, index, number] - 1
                )
                old_end = original[part.end_start : part.end_end]
                replacement = render_number_surface(old_end, part.end, end)
                if old_end.isdigit():
                    replacement = replacement.zfill(len(old_end))
                text = part.replace_end(text, replacement)
            rewritten[key][original] = text
    target_values = {
        path: rewritten[key][_text(source_target, path)]
        for key, selected in paths.items()
        for path in selected
    }
    # Different source range marks must not collapse into an identical array
    # entry when quantities round to the same integer. Shift identities, never
    # cardinalities or accepted labels. Stable source formatting is checked by
    # the ordinary renderer after this generation step.
    reserved: dict[str, set[str]] = {}
    for path, range_text in target_values.items():
        parent, _, _ = path.rpartition("[")
        if not parent:
            continue
        if parent not in reserved:
            siblings = _resolve_path(target, parent)
            if not isinstance(siblings, list):
                raise ValueError("range identity parent is not a list")
            reserved[parent] = {
                v
                for i, v in enumerate(siblings)
                if isinstance(v, str) and f"{parent}[{i}]" not in target_values
            }
        delta = 0
        candidate = range_text
        while candidate in reserved[parent]:
            delta += 1
            candidate = _shift_interval_numbers(range_text, delta)
        target_values[path] = candidate
        reserved[parent].add(candidate)
    auxiliaries = {
        key: {slot.slot_id: rewritten[key][slot.source_text] for slot in bindings[key].occurrences}
        for key in keys
        if not bindings[key].target_paths
    }
    return RangePlan(target_values, auxiliaries)
