"""Certify the source-only pallet level in a printed one-pallet-per-bag declaration."""

from __future__ import annotations

import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

from .models import SemanticBinding

_QUANTITY_PATH = "documentPatch.cargoPackages[0].quantity"
_CATEGORY_PATH = "documentPatch.cargoPackages[0].typeCategory"
_OUTER_COUNT_KEY = "derived:outer_pallet_count_equals_bag_quantity"
_OUTER_CATEGORY_KEY = "static:outer_pallet_category"
_COUNT_KEY = "anchor:" + _QUANTITY_PATH
_CATEGORY_KEY = "anchor:" + _CATEGORY_PATH
_LINE = re.compile(
    r"(?m)^(?P<outer>[1-9][0-9]*) (?P<outer_unit>PALLETS?) "
    r"\( (?P<inner>[1-9][0-9]*) (?P<inner_unit>BAGS?) \)$"
)


@dataclass(frozen=True, slots=True)
class OneToOnePalletBagConstraint:
    package_index: int
    source_quantity: int
    inner_category: str


def _one_slot_at(binding: SemanticBinding, source: bytes, start: int, end: int) -> None:
    matches = [
        slot
        for slot in binding.occurrences
        if slot.byte_start == start and slot.byte_end == end
    ]
    if len(matches) != 1 or matches[0].source_text.encode("utf-8") != source[start:end]:
        raise ValueError(f"pallet/bag source span lacks exact binding: {binding.logical_key}")


def _byte_span(text: str, match: re.Match[str], name: str) -> tuple[int, int]:
    return (
        len(text[: match.start(name)].encode("utf-8")),
        len(text[: match.end(name)].encode("utf-8")),
    )


def certify(
    *, source: bytes, bindings: Sequence[SemanticBinding], source_target: Mapping[str, Any]
) -> OneToOnePalletBagConstraint | None:
    """Return the exact private packing constraint, or fail on an incomplete declaration."""
    indexed = {binding.logical_key: binding for binding in bindings}
    if _OUTER_COUNT_KEY not in indexed:
        if _OUTER_CATEGORY_KEY in indexed:
            raise ValueError("outer pallet noun exists without its typed count owner")
        return None
    if len(indexed) != len(bindings):
        raise ValueError("duplicate logical keys in pallet/bag template")
    try:
        count = indexed[_OUTER_COUNT_KEY]
        pallet = indexed[_OUTER_CATEGORY_KEY]
        inner_count = indexed[_COUNT_KEY]
        inner_category = indexed[_CATEGORY_KEY]
    except KeyError as error:
        raise ValueError("one-to-one pallet/bag contract lacks a required binding") from error
    if (
        count.render_mode != "deterministic_derived"
        or count.derivation != "same_as_binding"
        or count.dependency_bindings != (_COUNT_KEY,)
        or count.target_paths
        or count.dependency_paths
        or count.group_key != "package:0:outer_pallets"
        or pallet.render_mode != "literal_static"
        or pallet.target_paths
        or pallet.dependency_paths
        or pallet.dependency_bindings
        or pallet.group_key != count.group_key
        or inner_count.render_mode != "target_binding"
        or inner_count.target_paths != (_QUANTITY_PATH,)
        or inner_category.render_mode != "target_binding"
        or inner_category.target_paths != (_CATEGORY_PATH,)
    ):
        raise ValueError("one-to-one pallet/bag roles or dependencies are incomplete")
    patch = source_target.get("documentPatch")
    if not isinstance(patch, Mapping):
        raise ValueError("pallet/bag source target lacks documentPatch")
    packages = patch.get("cargoPackages")
    groups = patch.get("cargoGroups")
    if (
        not isinstance(packages, list)
        or len(packages) != 1
        or not isinstance(groups, list)
        or len(groups) != 1
        or groups[0].get("additionalInformation")
        or packages[0].get("typeCategory") != "PACKAGE_BAG"
    ):
        raise ValueError("pallet/bag source requires one BAG fact and no retained AAI")
    text = source.decode("utf-8")
    matches = list(_LINE.finditer(text))
    if len(matches) != 1:
        raise ValueError("source needs one complete printed pallet/bag declaration")
    match = matches[0]
    quantity = packages[0].get("quantity")
    if (
        not isinstance(quantity, int)
        or isinstance(quantity, bool)
        or quantity <= 0
        or int(match["outer"]) != quantity
        or int(match["inner"]) != quantity
    ):
        raise ValueError("printed pallet/bag quantities do not equal the source BAG fact")
    for binding, name in (
        (count, "outer"),
        (pallet, "outer_unit"),
        (inner_count, "inner"),
        (inner_category, "inner_unit"),
    ):
        _one_slot_at(binding, source, *_byte_span(text, match, name))
    return OneToOnePalletBagConstraint(0, quantity, "PACKAGE_BAG")


def validate_descendant(
    *,
    source: bytes,
    rendered: bytes,
    bindings: Sequence[SemanticBinding],
    source_target: Mapping[str, Any],
    target: Mapping[str, Any],
) -> None:
    constraint = certify(source=source, bindings=bindings, source_target=source_target)
    if constraint is None:
        return
    packages = target.get("documentPatch", {}).get("cargoPackages", ())
    if len(packages) != 1 or packages[0].get("typeCategory") != constraint.inner_category:
        raise ValueError("sampled inner package is not certified for the printed pallet level")
    quantity = packages[0].get("quantity")
    if not isinstance(quantity, int) or isinstance(quantity, bool) or quantity <= 0:
        raise ValueError("sampled pallet/bag quantity must be a positive integer")
    rendered_matches = list(_LINE.finditer(rendered.decode("utf-8")))
    if (
        len(rendered_matches) != 1
        or int(rendered_matches[0]["outer"]) != quantity
        or int(rendered_matches[0]["inner"]) != quantity
    ):
        raise ValueError("rendered one-to-one pallet/bag declaration contradicts target")
