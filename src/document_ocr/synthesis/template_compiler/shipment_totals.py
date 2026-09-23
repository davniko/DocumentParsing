"""Explicit shipment package-total captions must not remain immutable context."""

from __future__ import annotations

import re
from collections.abc import Mapping, Sequence
from typing import Any

from document_ocr.hashing import sha256_bytes

_TOTAL = re.compile(
    r"(?im)^[ \t]*TOTAL[ \t]+(?:ITEMS|(?:NO\.?[ \t]+OF[ \t]+)?PACKAGES)"
    r"[ \t]*:[ \t]*(?P<count>[0-9]+(?:,[0-9]{3})*)(?![0-9.,])"
)


def occurrences(raw: str) -> tuple[re.Match[str], ...]:
    return tuple(_TOTAL.finditer(raw))


def normalize(
    *, raw: str, drafts: Sequence[Any], source_target: Mapping[str, Any]
) -> tuple[Any, ...]:
    """Own an unbound total only when a single observed package row proves it.

    More complex packing hierarchies require explicit compilation review. This
    does not pick one of several levels or overwrite an existing semantic owner.
    """
    from .host import SpanDraft, merge_drafts

    packages = source_target.get("documentPatch", {}).get("cargoPackages", ())
    if len(packages) != 1 or type(packages[0].get("quantity")) is not int:
        return tuple(drafts)
    expected = packages[0]["quantity"]
    additions = []
    for match in occurrences(raw):
        start, end = match.span("count")
        if int(match["count"].replace(",", "")) != expected:
            continue
        if any(d.char_start < end and d.char_end > start for d in drafts):
            continue
        path = "documentPatch.cargoPackages[0].quantity"
        additions.append(
            SpanDraft(
                draft_id="host_shipment_total_" + sha256_bytes(f"{start}:{end}".encode())[:16],
                logical_key="host:shipment_package_total",
                render_mode="deterministic_derived",
                value_kind="integer",
                group_kind="cargo",
                group_key="cargo:shipment_package_total",
                target_paths=(),
                derivation="sum_package_quantity",
                dependency_paths=(path,),
                dependency_bindings=(),
                char_start=start,
                char_end=end,
                source_text=raw[start:end],
                evidence_origin="derived_operational_fact",
                render_policy="derived_surface",
                rationale="Explicit shipment total equals the sole printed package quantity.",
            )
        )
    return merge_drafts(drafts, additions) if additions else tuple(drafts)


def require_owned(source: bytes, template: Any) -> None:
    """Reject uncovered or wrongly owned totals before any paid generation."""
    raw = source.decode()
    for match in occurrences(raw):
        start, end = (len(raw[:p].encode()) for p in match.span("count"))
        owners = [
            b
            for b in template.bindings
            if any(s.byte_start <= start and s.byte_end >= end for s in b.occurrences)
        ]
        if len(owners) != 1:
            raise ValueError("explicit shipment package total lacks one complete compiled owner")
        binding = owners[0]
        direct = any(
            re.fullmatch(r"documentPatch\.cargoPackages\[\d+\]\.quantity", p)
            for p in binding.target_paths
        )
        derived = binding.derivation in {"sum_package_quantity", "sum_values", "package_count"}
        if binding.derivation == "same_as_binding":
            derived = bool(binding.dependency_paths) and all(
                re.fullmatch(
                    r"documentPatch\.(?:cargoPackages\[\d+\]\.quantity|"
                    r"cargoAllocationGroups\[\d+\]\.allocations\[\d+\]\.packageQuantity)",
                    p,
                )
                for p in binding.dependency_paths
            )
        numeric = (
            not binding.target_paths
            and binding.derivation is None
            and binding.value_kind in {"integer", "package"}
            and binding.realization.mode != "static"
        )
        if not (direct or derived or numeric):
            raise ValueError("explicit shipment package total has an unrelated compiled owner")
