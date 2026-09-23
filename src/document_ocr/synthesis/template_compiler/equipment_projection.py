"""Recover complete equipment surfaces split by coincident package counts.

Only an exact source-labelled type in its own container row qualifies. A quantity
occurrence may be displaced only if its separate noun-qualified occurrence still
prints the same target fact. No container type or quantity is inferred.
"""

from __future__ import annotations

import re
from collections import defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import replace
from itertools import pairwise
from typing import TYPE_CHECKING, Any

from document_ocr.synthesis.container_semantics import review_source_equipment_surface

if TYPE_CHECKING:
    from .host import SpanDraft

_PATH = re.compile(r"documentPatch\.containers\[(\d+)\]\.typeDescription")
_QUANTITY = re.compile(
    r"documentPatch\.(?:cargoPackages\[\d+\]\.quantity|"
    r"cargoAllocationGroups\[\d+\]\.allocations\[\d+\]\.packageQuantity)"
)
_NOUN = re.compile(
    r"\s*(?:PALLETS?|CARTONS?|PACKAGES?|CASES?|BAGS?|ROLLS?|BOXES|DRUMS?|"
    r"PIECES?|UNITS?|CRATES?)\b",
    re.I,
)


def normalize_receipt_suffixes(
    *, raw: str, drafts: Sequence[SpanDraft], source_target: Mapping[str, Any]
) -> tuple[SpanDraft, ...]:
    """Join a source-only type suffix to its explicitly owned receipt.

    A split ``1x20'`` / ``GP`` must constrain the same private physical row.
    Neither proximity alone nor a group key supplies ownership: the first slot
    already needs an executable receipt dependency, and the complete spelling
    must pass the closed equipment/count grammar against that exact inventory.
    """
    from .descendant import _number_to_words
    from .equipment_receipts import owned_inventory, validate_source_receipt
    from .host import validate_draft_source_alignment

    validate_draft_source_alignment(raw=raw, drafts=drafts)
    referenced = {key for d in drafts for key in d.dependency_bindings}
    ordered = sorted(drafts, key=lambda d: (d.char_start, d.char_end, d.draft_id))
    replacements: dict[str, SpanDraft] = {}
    absorbed: set[str] = set()
    proposed: dict[str, list[tuple[SpanDraft, SpanDraft]]] = defaultdict(list)
    for left, right in pairwise(ordered):
        if not (
            left.derivation == "equipment_receipt"
            and left.render_mode == "deterministic_derived"
            and left.group_kind == right.group_kind == "equipment"
            and left.group_key == right.group_key
            and right.value_kind == "equipment"
            and right.render_mode == "deterministic_auxiliary"
            and not right.target_paths
            and not right.dependency_paths
            and not right.dependency_bindings
            and not right.derivation
            and right.logical_key not in referenced
            and left.char_end <= right.char_start
            and re.fullmatch(r"[ \t'\u2019`]*", raw[left.char_end : right.char_start])
        ):
            continue
        rows = owned_inventory(source_target, left.dependency_paths)
        try:
            validate_source_receipt(
                raw[left.char_start : right.char_end], rows, number_words=_number_to_words
            )
        except ValueError:
            continue  # Unproved observations remain unchanged and fail the admission guard.
        proposed[right.logical_key].append((left, right))
    counts: dict[str, int] = defaultdict(int)
    for d in drafts:
        counts[d.logical_key] += 1
    for key, pairs in proposed.items():
        if len(pairs) != counts[key]:
            continue  # Repeated modifiers cannot be repaired only at some occurrences.
        for left, right in pairs:
            replacements[left.draft_id] = replace(
                left,
                char_end=right.char_end,
                source_text=raw[left.char_start : right.char_end],
                rationale=left.rationale + " Host joined the adjacent source-only type modifier "
                "to its already declared physical receipt owner; full count/type proof passed.",
            )
            absorbed.add(right.draft_id)
    return tuple(replacements.get(d.draft_id, d) for d in drafts if d.draft_id not in absorbed)


def normalize_owned_auxiliary_receipts(
    *, raw: str, drafts: Sequence[SpanDraft], source_target: Mapping[str, Any]
) -> tuple[SpanDraft, ...]:
    """Give a compiler-owned, source-only equipment observation an executable owner.

    The compiler already uses container:i for printed types absent from labels.
    Promote that explicit scope, not proximity or a guessed container identity.
    Incomplete tokens and mixed inventories remain for contract review.
    """
    from .descendant import _number_to_words
    from .equipment_receipts import validate_source_receipt
    from .host import validate_compact_equipment_locality

    grouped: dict[str, list[SpanDraft]] = defaultdict(list)
    for draft in drafts:
        grouped[draft.logical_key].append(draft)
    eligible: dict[str, int] = {}
    containers = source_target.get("documentPatch", {}).get("containers", [])
    for key, occurrences in grouped.items():
        first = occurrences[0]
        match = re.fullmatch(r"container:([0-9]+)", first.group_key)
        if match is None or not all(
            d.group_key == first.group_key
            and d.render_mode == "deterministic_auxiliary"
            and d.value_kind == "equipment"
            and d.group_kind == "equipment"
            and not d.target_paths
            and not d.dependency_paths
            and not d.dependency_bindings
            and d.derivation is None
            for d in occurrences
        ):
            continue
        index = int(match[1])
        if index >= len(containers):
            raise ValueError("source-only equipment declares an absent container owner")
        if "typeDescription" in containers[index]:
            continue  # A partial lexical fragment needs a complete surface contract.
        try:
            for draft in occurrences:
                validate_source_receipt(
                    draft.source_text, (containers[index],), number_words=_number_to_words
                )
        except ValueError:
            continue  # Unresolved grammar is not eligible for automatic promotion.
        eligible[key] = index
    if not eligible:
        return tuple(drafts)
    validate_compact_equipment_locality(raw=raw, drafts=drafts, source_target=source_target)
    return tuple(
        replace(
            d,
            render_mode="deterministic_derived",
            derivation="equipment_receipt",
            dependency_paths=(f"documentPatch.containers[{eligible[d.logical_key]}]",),
            render_policy="natural_text",
            rationale=d.rationale + " Host made the explicit source-only container owner "
            "executable as a receipt; extraction label visibility is unchanged.",
        )
        if d.logical_key in eligible
        else d
        for d in drafts
    )


def validate_source_surfaces(binding: Any) -> None:
    """Partial lexical labels may still own a complete physical observation.

    A size/type abbreviation can omit FCL wording or height, but a bare DRY
    suffix cannot own a container-length prefix assigned to another field.
    Reuse the closed equipment grammar; arbitrary substring matches are unsafe.
    """
    from .equipment_receipts import _matches, _semantic

    values = {v.source_value for v in binding.realization.target_values}
    if len(values) != 1:
        raise ValueError("projected equipment lacks one source type observation")
    original = next(iter(values))
    if not isinstance(original, str):
        raise ValueError("projected equipment source type is not text")
    for slot in binding.occurrences:
        try:
            shape = _semantic(slot.source_text)
        except ValueError as error:
            raise ValueError(
                "partial equipment surface requires complete physical ownership before synthesis"
            ) from error
        if (shape.size is None and shape.length is None) or not _matches(
            {"typeDescription": original}, shape
        ):
            raise ValueError(
                "partial equipment surface requires complete physical ownership before synthesis"
            )


def normalize(
    *, raw: str, drafts: Sequence[SpanDraft], source_target: Mapping[str, Any]
) -> tuple[SpanDraft, ...]:
    from .host import _surface_token_spans, merge_drafts

    removed: set[str] = set()
    replacements: dict[str, SpanDraft] = {}
    for row in drafts:
        if row.render_mode != "target_binding" or len(row.target_paths) != 1:
            continue
        match = _PATH.fullmatch(row.target_paths[0])
        if match is None:
            continue
        container = source_target["documentPatch"]["containers"][int(match[1])]
        value = container["typeDescription"]
        reviewed = review_source_equipment_surface(
            value, temperature_present="temperatureSetpoint" in container
        )
        if not reviewed.size_category or not reviewed.type_category:
            continue
        full = tuple(t[0] for t in _surface_token_spans(value))
        observed = tuple(t[0] for t in _surface_token_spans(row.source_text))
        if full == observed:
            continue
        left = raw.rfind("\n", 0, row.char_start) + 1
        right = raw.find("\n", row.char_end)
        if right < 0:
            right = len(raw)
        line = raw[left:right]
        if container["containerNumber"] not in line[: row.char_start - left]:
            continue
        tokens = _surface_token_spans(line)
        spans = [
            (left + tokens[i][1], left + tokens[i + len(full) - 1][2])
            for i in range(len(tokens) - len(full) + 1)
            if tuple(t[0] for t in tokens[i : i + len(full)]) == full
            and left + tokens[i][1] <= row.char_start
            and row.char_end <= left + tokens[i + len(full) - 1][2]
        ]
        if len(spans) != 1:
            continue
        start, end = spans[0]
        overlaps = [
            d
            for d in drafts
            if d.draft_id != row.draft_id and start < d.char_end and d.char_start < end
        ]
        proven = True
        for other in overlaps:
            if (
                not other.target_paths
                or not all(_QUANTITY.fullmatch(p) for p in other.target_paths)
                or not start <= other.char_start < other.char_end <= end
                or not any(
                    d.draft_id != other.draft_id
                    and d.logical_key == other.logical_key
                    and d.target_paths == other.target_paths
                    and not (start < d.char_end and d.char_start < end)
                    and d.source_text == other.source_text
                    and _NOUN.match(raw[d.char_end :])
                    for d in drafts
                )
            ):
                proven = False
                break
        if not proven:
            continue
        removed.update(d.draft_id for d in overlaps)
        replacements[row.draft_id] = replace(
            row,
            char_start=start,
            char_end=end,
            source_text=raw[start:end],
            render_policy="natural_text",
            rationale=row.rationale + " Host recovered the exact complete equipment label in its "
            "own container row; every displaced quantity retains independent "
            "noun-qualified evidence.",
        )
    return merge_drafts(
        tuple(replacements.get(d.draft_id, d) for d in drafts if d.draft_id not in removed)
    )
