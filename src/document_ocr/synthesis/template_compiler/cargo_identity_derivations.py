"""Source-only commodity names owned by a specific printed tariff identity.

These names repeat goods already represented by a cargo group's HS code. They
are not independent linguistic requests and must not survive a changed goods
scenario as fixed source text. Compilation supplies the explicit owner; this
module never infers semantic ownership from proximity or equal words.
"""

from __future__ import annotations

import re
from collections.abc import Mapping, Sequence
from typing import Any

DERIVATIONS = frozenset({"sampled_cargo_identity"})
_OWNER = re.compile(r"documentPatch\.cargoGroups\[(\d+)\]\.hsCodes\[(\d+)\]")
_FOR_GOODS = re.compile(r"[ \t]+FOR[ \t]+(?P<name>[A-Za-z][A-Za-z0-9 /'-]*[A-Za-z0-9])", re.I)
_TARIFF_CAPTION = re.compile(
    r"\b(?:H\.?S\.?\s*(?:CODE|NUMBER|NO\.?)?|COMMODITY\s+CODE)\s*[:#]?\s*$", re.I
)


def require_tariff_owners(source: bytes, bindings: Sequence[Any]) -> None:
    """An explicitly captioned tariff number cannot use a random reference generator."""
    for binding in bindings:
        if any(_OWNER.fullmatch(p) for p in (*binding.target_paths, *binding.dependency_paths)):
            continue
        for slot in binding.occurrences:
            if (
                re.fullmatch(r"[0-9 .-]+", slot.source_text)
                and 6 <= len(re.sub(r"\D", "", slot.source_text)) <= 12
                and _TARIFF_CAPTION.search(source[: slot.byte_start].decode("utf-8")[-80:])
            ):
                raise ValueError(
                    "explicit tariff code lacks a goods/HS owner: " + binding.logical_key
                )


def unowned_alias_spans(raw: str, bindings: Sequence[Any]) -> tuple[tuple[int, int], ...]:
    """Screen the explicit 'HS-code FOR goods' grammar, without guessing its owner.

    Both compiler drafts and certified bindings are accepted. Matching a printed
    HS owner plus FOR identifies a changing commodity assertion, not boilerplate.
    Any existing mutable owner is left to normal semantic-contract validation.
    """
    encoded = raw.encode()
    ascii_only = len(encoded) == len(raw)
    spans: list[tuple[int, int, Any]] = []
    for binding in bindings:
        if hasattr(binding, "occurrences"):
            slots = tuple(
                (
                    s.byte_start if ascii_only else len(encoded[: s.byte_start].decode()),
                    s.byte_end if ascii_only else len(encoded[: s.byte_end].decode()),
                )
                for s in binding.occurrences
            )
        else:
            slots = ((binding.char_start, binding.char_end),)
        spans.extend((start, end, binding) for start, end in slots)
    output = set()
    for _, end, binding in spans:
        if not any(_OWNER.fullmatch(p) for p in binding.target_paths):
            continue
        match = _FOR_GOODS.match(raw, end)
        if match is None:
            continue
        start, stop = match.span("name")
        if not any(
            left <= start
            and stop <= right
            and owner_binding.render_mode not in {"literal_static", "carrier_static"}
            for left, right, owner_binding in spans
        ):
            output.add((start, stop))
    return tuple(sorted(output))


def owner(binding: Any) -> tuple[int, int]:
    paths = binding.dependency_paths
    match = _OWNER.fullmatch(paths[0]) if len(paths) == 1 else None
    if (
        binding.derivation not in DERIVATIONS
        or binding.target_paths
        or binding.dependency_bindings
        or binding.render_mode != "deterministic_derived"
        or binding.group_kind != "cargo"
        or binding.value_kind != "cargo_text"
        or match is None
    ):
        raise ValueError("repeated commodity name requires one explicit printed HS owner")
    return int(match[1]), int(match[2])


def validate_source(bindings: Sequence[Any], target: Mapping[str, Any]) -> None:
    for binding in bindings:
        if binding.derivation not in DERIVATIONS:
            continue
        gi, hi = owner(binding)
        try:
            code = target["documentPatch"]["cargoGroups"][gi]["hsCodes"][hi]
        except (KeyError, IndexError) as error:
            raise ValueError("repeated commodity name has an absent printed HS owner") from error
        if not isinstance(code, str) or not re.fullmatch(r"[0-9 .-]+", code):
            raise ValueError("repeated commodity name has an invalid printed HS owner")
        if len(re.sub(r"\D", "", code)) < 6:
            raise ValueError("repeated commodity name requires at least a six-digit HS owner")


def description(path: str, scenario: Any) -> str:
    match = _OWNER.fullmatch(path)
    if match is None:
        raise ValueError("commodity request has an invalid HS owner")
    gi, hi = int(match[1]), int(match[2])
    group = scenario.target["documentPatch"]["cargoGroups"][gi]
    hs6 = re.sub(r"\D", "", group["hsCodes"][hi])[:6]
    names = {
        identity.get("properShippingName") or identity["requiredDescription"]
        for identity in scenario.identities[group["groupId"]]
        if identity.get("hs6") == hs6
    }
    if len(names) != 1:
        raise ValueError("repeated commodity name lacks a unique generated goods identity")
    name = names.pop()
    if not isinstance(name, str) or not name.strip():
        raise ValueError("repeated commodity name lacks a unique generated goods identity")
    return name
