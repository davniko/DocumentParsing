"""Evidence-preserving realization corrections for compiled source spans.

These corrections do not move slots, alter source bytes, or relax identifier
contracts. They correct lexical fields that were classified as identifiers.
The original catalog stays immutable; consumers record both contract hashes.
"""

from __future__ import annotations

import re
from collections.abc import Mapping
from typing import Any

from document_ocr.hashing import canonical_json_bytes
from document_ocr.synthesis.raw_text_template import format_envelope

from .models import CertifiedSemanticTemplate, SemanticBinding


def character_partition_frames(
    binding: SemanticBinding,
) -> tuple[tuple[str, str, str], ...] | None:
    """Prove an uninterrupted scalar split around punctuation-only continuation marks.

    Email addresses and URLs are one token even when printed across pages. Their
    source characters, unlike free prose, must partition exactly without loss.
    """
    if binding.realization.mode != "segmented_surface" or binding.value_kind not in {
        "email",
        "url",
    }:
        return None
    originals = tuple(row.source_value for row in binding.realization.target_values)
    if (
        not originals
        or not isinstance(originals[0], str)
        or any(value != originals[0] for value in originals)
    ):
        raise ValueError("segmented scalar has no unique source value")
    original = originals[0]
    cursor = 0
    frames = []
    for slot in binding.occurrences:
        text = slot.source_text
        matches = []
        for start in range(len(text)):
            if any(c.isalnum() for c in text[:start]):
                break
            length = 0
            while (
                start + length < len(text)
                and cursor + length < len(original)
                and text[start + length].casefold() == original[cursor + length].casefold()
            ):
                length += 1
            if length and not any(c.isalnum() for c in text[start + length :]):
                matches.append((length, start))
        if not matches:
            raise ValueError("segmented scalar lacks an exact source-character partition")
        width = max(length for length, _ in matches)
        starts = [start for length, start in matches if length == width]
        if len(starts) != 1:
            raise ValueError("segmented scalar source-character partition is ambiguous")
        start = starts[0]
        frames.append((text[:start], text[start : start + width], text[start + width :]))
        cursor += width
    if cursor != len(original):
        raise ValueError("segmented scalar partition does not cover its complete source value")
    return tuple(frames)


def token_projection_intervals(binding: SemanticBinding) -> tuple[tuple[int, int], ...] | None:
    """Prove that projected occurrences collectively print the complete target.

    The old compiler stored other segments as required *values*, even when those
    values had their own mutable slots. Such a complete partition can vary safely;
    truly unprinted prefixes/suffixes remain constrained instead.
    """
    if binding.realization.mode != "token_projected_surface":
        return None
    snapshots = tuple(row.source_value for row in binding.realization.target_values)
    if (
        not snapshots
        or not isinstance(snapshots[0], str)
        or any(v != snapshots[0] for v in snapshots)
    ):
        return None
    tokens = tuple(re.findall(r"[A-Za-z0-9]+", snapshots[0].casefold()))
    intervals = []
    for slot in binding.realization.slots:
        prefix = tuple(v.casefold() for v in slot.required_target_prefix_tokens)
        suffix = tuple(v.casefold() for v in slot.required_target_suffix_tokens)
        start, end = len(prefix), len(tokens) - len(suffix)
        if tokens[:start] != prefix or (suffix and tokens[end:] != suffix) or start >= end:
            raise ValueError(f"invalid source token-projection proof: {binding.logical_key}")
        intervals.append((start, end))
    return tuple(intervals)


def fixed_projection_ranges(
    binding: SemanticBinding,
) -> tuple[tuple[int, int, tuple[str, ...]], ...]:
    intervals = token_projection_intervals(binding)
    if intervals is None:
        return ()
    tokens = tuple(
        re.findall(
            r"[A-Za-z0-9]+", str(binding.realization.target_values[0].source_value).casefold()
        )
    )
    covered = {i for start, end in intervals for i in range(start, end)}
    gaps = []
    index = 0
    while index < len(tokens):
        if index in covered:
            index += 1
            continue
        end = index + 1
        while end < len(tokens) and end not in covered:
            end += 1
        gaps.append((index, end, tokens[index:end]))
        index = end
    return tuple(gaps)


def complete_token_intervals(binding: SemanticBinding) -> tuple[tuple[int, int], ...] | None:
    intervals = token_projection_intervals(binding)
    return intervals if intervals is not None and not fixed_projection_ranges(binding) else None


def token_projection_pattern(binding: SemanticBinding) -> str | None:
    """Constrain generation to the proved fixed literals and nonempty mutable regions.

    The renderer remains authoritative for unique literal matches and exact slot
    ownership. This native grammar prevents an interior literal from moving to
    an edge, or a mutable region from losing the tokens needed by its slots.
    """
    intervals = token_projection_intervals(binding)
    fixed = fixed_projection_ranges(binding)
    if intervals is None or not fixed:
        return None
    size = len(re.findall(r"[A-Za-z0-9]+", str(binding.realization.target_values[0].source_value)))
    boundaries = {i for interval in intervals for i in interval}
    pieces = []
    cursor = 0
    for start, end, tokens in (*fixed, (size, size, ())):
        if start > cursor:
            minimum = len({cursor, start, *[i for i in boundaries if cursor < i < start]}) - 1
            pieces.append(r"[A-Za-z0-9]+(?:[^A-Za-z0-9]+[A-Za-z0-9]+){" + str(minimum - 1) + ",}")
        pieces.extend(
            "".join(f"[{c.lower()}{c.upper()}]" if c.isalpha() else re.escape(c) for c in token)
            for token in tokens
        )
        cursor = end
    return r"^[^A-Za-z0-9]*" + r"[^A-Za-z0-9]+".join(pieces) + r"[^A-Za-z0-9]*$"


def shared_projection_ranges(
    binding: SemanticBinding,
    template: CertifiedSemanticTemplate | None,
    target: Mapping[str, Any],
) -> tuple[tuple[int, int, tuple[str, ...]], ...]:
    """Share fixed coordinate anchors only for equal source AND descendant scalars.

    A field can print the full shared address while another prints only its street.
    Its repeated office fragment must use the street owner's exact suffix anchor,
    not an independently weighted cut. No party equality is inferred or imposed.
    """
    from .descendant import _binding_target_value

    own = fixed_projection_ranges(binding)
    if template is None or not binding.realization.target_values:
        return own
    original = binding.realization.target_values[0].source_value
    if not isinstance(original, str):
        return own
    current = _binding_target_value(target, binding.target_paths[0])
    covered = {i for start, end, _ in own for i in range(start, end)}
    for other in template.bindings:
        if other is binding or other.realization.mode != "token_projected_surface":
            continue
        if (
            not other.realization.target_values
            or any(v.source_value != original for v in other.realization.target_values)
            or any(_binding_target_value(target, p) != current for p in other.target_paths)
        ):
            continue
        covered.update(
            i for start, end, _ in fixed_projection_ranges(other) for i in range(start, end)
        )
    tokens = tuple(re.findall(r"[A-Za-z0-9]+", original.casefold()))
    ranges = []
    start = 0
    while start < len(tokens):
        if start not in covered:
            start += 1
            continue
        end = start + 1
        while end in covered:
            end += 1
        ranges.append((start, end, tokens[start:end]))
        start = end
    from .projected_context import sampled_ranges

    return sampled_ranges(binding, template, target, ranges)


def effective_realization_template(
    template: CertifiedSemanticTemplate,
) -> CertifiedSemanticTemplate:
    equipment_keys = {
        binding.logical_key
        for binding in template.bindings
        if binding.target_paths
        and all(
            re.fullmatch(r"documentPatch\.containers\[\d+\]\.typeDescription", path)
            for path in binding.target_paths
        )
        and any(slot.render_policy == "opaque_identifier" for slot in binding.occurrences)
    }
    lexical_keys = {
        binding.logical_key
        for binding in template.bindings
        if (
            (
                binding.target_paths
                and all(
                    path == "documentPatch.transport.vesselName" for path in binding.target_paths
                )
            )
            or binding.derivation == "sampled_auxiliary_vessel"
        )
        and any(slot.render_policy == "opaque_identifier" for slot in binding.occurrences)
    }
    if not lexical_keys and not equipment_keys:
        return template
    payload = template.model_dump(mode="json")
    updated_slots = {}
    for binding in payload["bindings"]:
        if binding["logical_key"] not in lexical_keys | equipment_keys:
            continue
        # Independent feeders keep their typed physical-vessel contract. Names
        # are lexical surfaces, not identifiers with a fixed character count.
        if (
            binding["logical_key"] in lexical_keys
            and binding["derivation"] != "sampled_auxiliary_vessel"
        ):
            binding["value_kind"] = "other_text"
        if binding["realization"]["adapter"] == "opaque_identifier":
            binding["realization"]["adapter"] = "natural_text"
        for slot in binding["occurrences"]:
            slot["render_policy"] = "natural_text"
            slot["format_envelope"] = format_envelope(
                slot["source_text"], render_policy="natural_text"
            ).model_dump(mode="json")
            updated_slots[slot["slot_id"]] = slot
    payload["byte_template"]["slots"] = [
        updated_slots.get(slot["slot_id"], slot) for slot in payload["byte_template"]["slots"]
    ]
    return CertifiedSemanticTemplate.model_validate_json(canonical_json_bytes(payload), strict=True)


def projected_auxiliary_values(
    template: CertifiedSemanticTemplate, target: Mapping[str, Any]
) -> dict[str, str]:
    """Derive separately owned context from the target's explicit literal contract.

    Only whole source-only occurrences are eligible. Target-backed changing facts
    cannot be frozen here. This runs BEFORE target acceptance and before any
    independent auxiliary generator, never as a repair of an accepted scenario.
    """
    from . import descendant as r

    owners: dict[tuple[str, ...], list[str]] = {}
    for binding in template.bindings:
        if binding.target_paths or binding.derivation is not None:
            continue
        surfaces = {
            tuple(token[0] for token in r._token_spans(s.source_text)) for s in binding.occurrences
        }
        if len(surfaces) == 1:
            owners.setdefault(surfaces.pop(), []).append(binding.logical_key)
    values: dict[str, str] = {}
    for binding in template.bindings:
        for start, end, fragment in fixed_projection_ranges(binding):
            projections = [(key, 0, len(fragment)) for key in owners.get(fragment, ())]
            # A fixed frame may contain a printed label outside an owned slot:
            # e.g. literal C/O: followed by an independently bound company name.
            # Only explicitly linked facets of this same target party can be
            # projected from a proper subrange of that proven immutable frame.
            for entity in template.auxiliary_semantic_plan.entities:
                for member in entity.members:
                    path = f"{entity.target_party_path}.{member.field}"
                    if path not in binding.target_paths:
                        continue
                    for surface, keys in owners.items():
                        if not surface or member.logical_key not in keys or surface == fragment:
                            continue
                        offsets = [
                            i
                            for i in range(len(fragment) - len(surface) + 1)
                            if fragment[i : i + len(surface)] == surface
                        ]
                        if len(offsets) == 1:
                            projections.append((member.logical_key, offsets[0], len(surface)))
            if not projections:
                continue
            raw = r._resolve_path(target, binding.target_paths[0])
            if not isinstance(raw, str):
                raise ValueError("projected auxiliary context requires a string target")
            tokens = r._token_spans(raw)
            normalized = tuple(row[0] for row in tokens)
            source_value = binding.realization.target_values[0].source_value
            if not isinstance(source_value, str):
                raise ValueError("projected auxiliary context requires a string source")
            source_size = len(r._token_spans(source_value))
            matches = [
                i
                for i in range(len(tokens) - len(fragment) + 1)
                if normalized[i : i + len(fragment)] == fragment
                and (start != 0 or i == 0)
                and (end != source_size or i + len(fragment) == len(tokens))
            ]
            if len(matches) != 1:
                raise ValueError("projected auxiliary context is not unique in completed target")
            index = matches[0]
            for key, offset, width in projections:
                text = raw[tokens[index + offset][1] : tokens[index + offset + width - 1][2]]
                previous = values.setdefault(key, text)
                if (
                    tuple(x[0] for x in r._token_spans(previous))
                    != fragment[offset : offset + width]
                ):
                    raise ValueError("projected auxiliary contexts disagree")
    return values
