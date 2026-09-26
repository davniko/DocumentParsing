"""Own printed package counts before varying structured package quantities.

The compiler may prove a unique source quantity from a printed package caption.
Different packing levels and repeated equal rows need explicit review instead
of an inferred equality. The same screen runs at synthesis preflight so an
unowned source count cannot silently survive in a generated document.
"""

from __future__ import annotations

import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

from document_ocr.hashing import sha256_bytes

_AFTER = re.compile(
    rb"^\s*(?:[(/-]\s*)?(?P<noun>PACKAGES?|PACKGS?|PKGS?|CARTONS?|CTNS?|"
    rb"PALLETS?|BAGS?|DRUMS?|PCS|PIECES|SKIDS?|BOXES|CRATES?|BALES?|ROLLS?|"
    rb"BUNDLES?)\b",
    re.I,
)
_BEFORE = re.compile(
    rb"\b(?P<noun>PACKAGES?|PACKGS?|PKGS?|CARTONS?|CTNS?|PALLETS?|BAGS?|"
    rb"DRUMS?|PCS|PIECES|SKIDS?|BOXES|CRATES?|BALES?|ROLLS?|BUNDLES?|QTY)"
    rb"\s*[:#=-]?\s*$",
    re.I,
)
_HEADING = re.compile(
    rb"\b(?:NO\.?\s*(?:OF\s*)?|NUMBER\s+OF\s+|TOTAL\s*(?:NO\.?\s*OF\s*)?)"
    rb"(?:PACKAGES?|PACKGS?|PKGS?|PACKS?|CARTONS?|CTNS?)\b",
    re.I,
)
_GENERIC = frozenset({"PACK", "PACKS", "PACKG", "PACKGS", "PKG", "PKGS", "PACKAGE", "PACKAGES"})


@dataclass(frozen=True)
class PrintedCount:
    byte_start: int
    byte_end: int
    quantity: int
    noun: str
    kind: str


def occurrences(source: bytes, target: Mapping[str, Any]) -> tuple[PrintedCount, ...]:
    packages = target.get("documentPatch", {}).get("cargoPackages", ())
    quantities = {
        p["quantity"] for p in packages if type(p.get("quantity")) is int and p["quantity"] > 0
    }
    lines = source.splitlines(keepends=True)
    result: list[PrintedCount] = []
    offset = 0
    for index, line in enumerate(lines):
        for quantity in quantities:
            number = re.compile(
                rb"(?<![\d.,])(?:"
                + f"{quantity:,}".encode()
                + rb"|"
                + str(quantity).encode()
                + rb")(?![\d.,])"
            )
            for match in number.finditer(line):
                after = _AFTER.match(line[match.end() : match.end() + 35])
                before = _BEFORE.search(line[max(0, match.start() - 45) : match.start()])
                headed = (
                    not line[: match.start()].strip()
                    and not line[match.end() :].strip()
                    and any(_HEADING.search(row) for row in lines[max(0, index - 2) : index])
                )
                if after is None and before is None and not headed:
                    continue
                noun = (
                    after["noun"].decode().upper()
                    if after is not None
                    else before["noun"].decode().upper()
                    if before is not None
                    else "PACKAGE"
                )
                result.append(
                    PrintedCount(
                        byte_start=offset + match.start(),
                        byte_end=offset + match.end(),
                        quantity=quantity,
                        noun=noun,
                        kind="headed" if headed else "direct",
                    )
                )
        offset += len(line)
    return tuple(sorted(result, key=lambda row: row.byte_start))


def _owner(ranges: Sequence[tuple[int, int]], byte_start: int, byte_end: int) -> bool:
    return any(start < byte_end and end > byte_start for start, end in ranges)


def normalize(
    *, raw: str, drafts: Sequence[Any], source_target: Mapping[str, Any]
) -> tuple[Any, ...]:
    """Bind only unique, same-level or generic package captions automatically."""
    from .host import SpanDraft, merge_drafts
    from .package_equations import _CATEGORIES

    packages = source_target.get("documentPatch", {}).get("cargoPackages", ())
    if len(packages) != 1 or type(packages[0].get("quantity")) is not int:
        return tuple(drafts)
    quantity = packages[0]["quantity"]
    category = packages[0].get("typeCategory")
    source = raw.encode("utf-8")
    ranges = tuple(
        (len(raw[: draft.char_start].encode("utf-8")), len(raw[: draft.char_end].encode("utf-8")))
        for draft in drafts
    )
    additions = []
    for occurrence in occurrences(source, source_target):
        if occurrence.quantity != quantity or _owner(
            ranges, occurrence.byte_start, occurrence.byte_end
        ):
            continue
        noun = occurrence.noun
        if (
            occurrence.kind != "headed"
            and noun not in _GENERIC
            and (_CATEGORIES.get(noun) != category)
        ):
            continue
        start = len(source[: occurrence.byte_start].decode("utf-8"))
        end = len(source[: occurrence.byte_end].decode("utf-8"))
        path = "documentPatch.cargoPackages[0].quantity"
        additions.append(
            SpanDraft(
                draft_id="host_package_count_"
                + sha256_bytes(f"{occurrence.byte_start}:{occurrence.byte_end}".encode())[:16],
                logical_key="host:package_count:" + str(occurrence.byte_start),
                render_mode="deterministic_derived",
                value_kind="integer",
                group_kind="cargo",
                group_key="cargo:package_count",
                target_paths=(),
                derivation="sum_package_quantity",
                dependency_paths=(path,),
                dependency_bindings=(),
                char_start=start,
                char_end=end,
                source_text=raw[start:end],
                evidence_origin="derived_operational_fact",
                render_policy="derived_surface",
                rationale=(
                    "Unique printed package count equals the sole structured package quantity."
                ),
            )
        )
    return merge_drafts(drafts, additions) if additions else tuple(drafts)


def require_owned(source: bytes, template: Any, target: Mapping[str, Any]) -> None:
    """Fail closed if a source quantity survives outside all compiled slots."""
    for occurrence in occurrences(source, target):
        owners = [
            binding.logical_key
            for binding in template.bindings
            if any(
                slot.byte_start < occurrence.byte_end and slot.byte_end > occurrence.byte_start
                for slot in binding.occurrences
            )
        ]
        if not owners:
            raise ValueError(
                "printed package count lacks a compiled owner: "
                f"{occurrence.quantity} {occurrence.noun} at byte {occurrence.byte_start}"
            )
