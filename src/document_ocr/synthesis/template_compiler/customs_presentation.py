"""Registry-proven, source-bound customs captions, outside the training target.

This is an explicit extension of the literal render contract, not a search/replace
over generated OCR. Existing value slots and the original certified catalog are
immutable. Ambiguous overlaps require review before a provider can be called.
"""

from __future__ import annotations

import re
from collections.abc import Mapping
from dataclasses import dataclass, replace
from typing import TYPE_CHECKING

from document_ocr.hashing import canonical_json_bytes, sha256_bytes
from document_ocr.synthesis.raw_text_hybrid_probe import _render_route_neutral_customs_heading
from document_ocr.synthesis.raw_text_rewrite_cycle_probe import (
    CustomsProgramRegistry,
    _customs_selector_match_is_safe,
    _literal_phrase_pattern,
)
from document_ocr.synthesis.raw_text_template import (
    CompiledRawTextTemplate,
    TemplateRenderProof,
    build_template_slot,
    compile_raw_text_template,
    render_compiled_template,
)

if TYPE_CHECKING:
    from .host import SpanDraft


def normalize_registered_caption_ownership(
    *, raw: str, drafts: tuple[SpanDraft, ...], registry: CustomsProgramRegistry
) -> tuple[SpanDraft, ...]:
    """Separate registered literal captions from unlabelled identifier values.

    This is a compilation repair, never a text rewrite. Target-backed fields,
    dependent values, and mixed prose are deliberately not reclassified.
    """
    from .host import merge_drafts

    surfaces = [s for entry in registry.entries for s in entry.source_surfaces] + [
        rewrite.source_surface
        for entry in registry.entries
        for rewrite in entry.party_surface_rewrites
    ]
    captions: list[tuple[int, int, bool]] = []
    for surface in sorted(surfaces, key=len, reverse=True):
        for match in _literal_phrase_pattern(surface).finditer(raw):
            if not _customs_selector_match_is_safe(raw, match, surface):
                continue
            if any(match.start() < end and start < match.end() for start, end, _ in captions):
                continue
            captions.append(
                (
                    *match.span(),
                    bool(re.search(r"\b(?:SHIPMENT|BOUND|DESTINED|TRANSIT)\b", surface, re.I)),
                )
            )
    output = []
    for row in drafts:
        # A redundant country occurrence inside a registered customs assertion
        # is presentation context, not the sole source of a route-country label.
        # Keep its independently printed route occurrence as the target owner.
        redundant_route_caption = bool(
            row.target_paths
            and row.group_kind == "route"
            and all(
                re.fullmatch(r"documentPatch\.route\.[^.]+\.country", p) for p in row.target_paths
            )
            and any(
                start <= row.char_start and row.char_end <= end and legal
                for start, end, legal in captions
            )
            and any(
                other.logical_key == row.logical_key
                and other.target_paths == row.target_paths
                and not any(
                    start < other.char_end and other.char_start < end for start, end, _ in captions
                )
                for other in drafts
            )
        )
        if (
            (row.target_paths and not redundant_route_caption)
            or row.dependency_paths
            or row.dependency_bindings
            or row.derivation is not None
            or row.render_mode in {"carrier_static", "literal_static"}
        ):
            output.append(row)
            continue
        for start, end, legal in captions:
            complete_caption = (
                row.char_start == start
                and row.char_end >= end
                and not raw[end : row.char_end].strip(" \t\r\n.:#")
            )
            if ((start <= row.char_start and row.char_end <= end) or complete_caption) and (
                not legal or complete_caption or redundant_route_caption
            ):
                row = replace(
                    row,
                    logical_key="caption:" + row.logical_key,
                    render_mode="literal_static",
                    value_kind="other_text",
                    group_kind="document",
                    group_key="document:customs_caption",
                    target_paths=(),
                    render_policy="natural_text",
                    rationale=(
                        "Exact literal part of a pinned registered customs heading; "
                        "not a shipment value."
                    ),
                )
                break
            if (
                not legal
                and start <= row.char_start < end < row.char_end
                and row.value_kind == "identifier"
            ):
                tail = raw[end : row.char_end]
                identifier = re.fullmatch(r"[\s.:#-]*(?P<value>[0-9][A-Z0-9./_-]{5,})", tail, re.I)
                if identifier is not None:
                    value_start = end + identifier.start("value")
                    row = replace(
                        row,
                        char_start=value_start,
                        source_text=raw[value_start : row.char_end],
                        rationale=(
                            row.rationale + " Host separated the pinned registered customs "
                            "caption from its identifier value."
                        ),
                    )
                    break
        output.append(row)
    return merge_drafts(output)


@dataclass(frozen=True)
class CustomsPresentation:
    template: CompiledRawTextTemplate
    replacements: Mapping[str, str]
    evidence: tuple[dict[str, object], ...]
    retired_static_slots: Mapping[str, str]

    @property
    def sha256(self) -> str:
        return sha256_bytes(canonical_json_bytes(self.evidence))

    def render(
        self, source: bytes, bindings: Mapping[str, str]
    ) -> tuple[bytes, TemplateRenderProof]:
        if set(bindings) & set(self.replacements):
            raise ValueError("customs caption and value slot identities overlap")
        values = dict(bindings)
        for slot_id, original in self.retired_static_slots.items():
            if values.pop(slot_id, None) != original:
                raise ValueError("customs extension can only retire unchanged static slots")
        return render_compiled_template(
            source=source, template=self.template, bindings={**values, **self.replacements}
        )


def _multiline_caption(observed: str, rendered: str) -> str:
    """Keep physical line endings and edges; distribute whole replacement words."""
    endings = re.findall(r"\r\n|\r|\n", observed)
    if not endings:
        return rendered
    lines = re.split(r"\r\n|\r|\n", observed)
    words = rendered.split()
    if len(words) < len(lines):
        raise ValueError("neutral customs caption has fewer words than its printed lines")
    output = []
    cursor = 0
    for index, line in enumerate(lines):
        remaining_lines = len(lines) - index - 1
        count = (
            len(words) - cursor
            if not remaining_lines
            else min(max(1, len(line.split())), len(words) - cursor - remaining_lines)
        )
        leading = re.match(r"^[ \t]*", line)
        trailing = re.search(r"[ \t]*$", line)
        assert leading is not None and trailing is not None
        output.append(leading.group() + " ".join(words[cursor : cursor + count]) + trailing.group())
        if index < len(endings):
            output.append(endings[index])
        cursor += count
    if cursor != len(words):
        raise AssertionError("customs caption words were not consumed exactly")
    return "".join(output)


def compile_neutral_customs(
    *,
    source: bytes,
    template: CompiledRawTextTemplate,
    registry: CustomsProgramRegistry,
    static_slot_ids: frozenset[str] = frozenset(),
) -> CustomsPresentation:
    """Neutralize registered program captions for all destinations, including Egypt.

    Generic customs references are an explicit configured presentation policy;
    this never invents membership in a different country's regulatory program.
    Carrier company-registry captions are deliberately not changed.
    """
    if sha256_bytes(source) != template.source_sha256:
        raise ValueError("customs presentation source differs from compiled source")
    raw = source.decode("utf-8")
    registered = [
        (surface, entry.generic_replacement_surface, entry.program_id)
        for entry in registry.entries
        for surface in entry.source_surfaces
    ] + [
        (rewrite.source_surface, rewrite.generic_replacement_surface, entry.program_id)
        for entry in registry.entries
        for rewrite in entry.party_surface_rewrites
    ]
    occupied: list[tuple[int, int]] = []
    slots = list(template.slots)
    next_id = max((int(s.slot_id.split("_")[1]) for s in slots), default=0) + 1
    replacements: dict[str, str] = {}
    evidence: list[dict[str, object]] = []
    retired: dict[str, str] = {}
    for surface, generic, program_id in sorted(
        registered, key=lambda row: len(row[0]), reverse=True
    ):
        for match in _literal_phrase_pattern(surface).finditer(raw):
            if not _customs_selector_match_is_safe(raw, match, surface):
                continue
            if any(match.start() < end and match.end() > start for start, end in occupied):
                continue
            start, end = len(raw[: match.start()].encode()), len(raw[: match.end()].encode())
            overlaps = [s for s in template.slots if s.byte_start < end and s.byte_end > start]
            if any(s.slot_id not in static_slot_ids for s in overlaps):
                raise ValueError(
                    f"customs caption overlaps existing value ownership: {surface!r}: "
                    f"{[s.slot_id for s in overlaps]}"
                )
            retired.update((s.slot_id, s.source_text) for s in overlaps)
            rendered = _render_route_neutral_customs_heading(
                match, registered_source=surface, generic_target=generic
            )
            if "".join(c for c in match.group() if c.isalpha()).isupper():
                rendered = rendered.upper()
            rendered = _multiline_caption(match.group(), rendered)
            slot = build_template_slot(
                slot_id=f"slot_{next_id:04d}",
                byte_start=start,
                byte_end=end,
                source_text=match.group(),
                target_paths=(),
                semantic_role=f"customs_caption:{program_id}",
                evidence_origin="audited_source_auxiliary",
                render_policy="natural_text",
            )
            slots.append(slot)
            replacements[slot.slot_id] = rendered
            evidence.append(
                {
                    "slotId": slot.slot_id,
                    "programId": program_id,
                    "byteStart": start,
                    "byteEnd": end,
                    "source": match.group(),
                    "replacement": rendered,
                    "basis": "registered_jurisdiction_neutral_caption",
                }
            )
            next_id += 1
            occupied.append(match.span())
    expanded = compile_raw_text_template(
        document_id=template.document_id,
        source=source,
        slots=[s for s in slots if s.slot_id not in retired],
    )
    result = CustomsPresentation(expanded, replacements, tuple(evidence), retired)
    # Check real formatting and page/line guarantees, not only span arithmetic.
    result.render(source, {s.slot_id: s.source_text for s in template.slots})
    return result
