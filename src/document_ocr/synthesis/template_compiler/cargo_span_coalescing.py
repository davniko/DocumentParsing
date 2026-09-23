"""Coalesce over-segmented complete cargo values without crossing other facts.

Line breaks alone need not create independent product-generation requests. This
normalizer only joins a consecutive run when its entire source span proves the
complete existing scalar. Partial items separated by quantities stay separate.
"""

from __future__ import annotations

import re
from collections.abc import Mapping, Sequence
from dataclasses import replace
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from .host import SpanDraft


def normalize(
    *, raw: str, drafts: Sequence[SpanDraft], source_target: Mapping[str, Any]
) -> tuple[SpanDraft, ...]:
    from .host import _resolve_target_path, _surface_match, validate_draft_source_alignment

    validate_draft_source_alignment(raw=raw, drafts=drafts)
    ordered = sorted(drafts, key=lambda d: (d.char_start, d.char_end, d.draft_id))
    result: list[SpanDraft] = []
    index = 0
    while index < len(ordered):
        first = ordered[index]
        end = index + 1
        if (
            first.render_mode == "target_binding"
            and first.value_kind == "cargo_text"
            and first.render_policy == "natural_text"
            and first.target_paths
            and all(
                re.fullmatch(r"documentPatch\.cargoGroups\[\d+\]\.description", p)
                for p in first.target_paths
            )
            and not first.derivation
            and not first.dependency_paths
            and not first.dependency_bindings
        ):
            while end < len(ordered):
                previous, current = ordered[end - 1], ordered[end]
                gap = raw[previous.char_end : current.char_start]
                if (
                    current.char_start <= previous.char_end
                    or not gap.isspace()
                    or any(
                        getattr(current, field) != getattr(first, field)
                        for field in (
                            "logical_key",
                            "target_paths",
                            "group_key",
                            "group_kind",
                            "render_mode",
                            "value_kind",
                            "render_policy",
                            "derivation",
                            "dependency_paths",
                            "dependency_bindings",
                        )
                    )
                ):
                    break
                end += 1
            if end > index + 1:
                text = raw[first.char_start : ordered[end - 1].char_end]
                values = [_resolve_target_path(source_target, path) for path in first.target_paths]
                if all(
                    isinstance(value, str)
                    and _surface_match(
                        source=text, target=value, adapter="natural_text", value_kind="cargo_text"
                    )
                    == ("", "")
                    for value in values
                ):
                    result.append(
                        replace(
                            first,
                            char_end=ordered[end - 1].char_end,
                            source_text=text,
                            rationale=(
                                first.rationale + " Host joined consecutive cargo spans whose "
                                "whitespace-only union expresses the complete target scalar; "
                                "no numeric, reference or other owned surface was crossed."
                            ),
                        )
                    )
                    index = end
                    continue
        result.append(first)
        index += 1
    return tuple(result)
