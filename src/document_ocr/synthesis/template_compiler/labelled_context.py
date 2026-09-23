"""Reviewed cargo labels derived from their explicit party or route owner.

Dependencies are compilation contracts, never inferred from coincident text.
They produce target facts before rendering; the renderer cannot silently repair
a conflicting accepted target. Destination facilities are synthetic names, not
assertions that a source warehouse is an alias of its port.
"""

from __future__ import annotations

import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any, Literal

_MARK = re.compile(r"documentPatch\.cargoGroups\[\d+\]\.marksAndNumbers\[\d+\]")
_HANDLING = re.compile(r"documentPatch\.cargoGroups\[\d+\]\.handlingInstructions\[\d+\]")
_PARTY = re.compile(
    r"(?P<party>documentPatch\.parties\.(?:[A-Za-z]+|notifyParties\[\d+\]))"
    r"\.(?P<field>name|city|country)"
)
_DESTINATION = re.compile(
    r"documentPatch\.route\.(?:portOfDischarge|placeOfDelivery|finalDestination)\.name"
)
_WAREHOUSE = re.compile(
    r"(?P<prefix>CARGO\s+IN\s+TRANSIT\s+TO\s+)"
    r"(?P<name>[A-Z0-9][A-Z0-9 .&/'-]*?)"
    r"(?P<suffix>\s+BONDED\s+WAREHOUSE)",
    re.I,
)


@dataclass(frozen=True, slots=True)
class _Contract:
    target_path: str
    owner_paths: tuple[str, ...]
    kind: Literal["party_name", "party_locality", "synthetic_destination_warehouse"]
    source: str
    spans: tuple[tuple[int, int], ...] = ()


def _text(target: Mapping[str, Any], path: str) -> str:
    from .descendant import _resolve_path

    value = _resolve_path(target, path)
    if not isinstance(value, str) or not value.strip() or "\n" in value or "\r" in value:
        raise ValueError("labelled context owner must be nonempty single-line text: " + path)
    return value


def _tokens(value: str) -> tuple[str, ...]:
    from .descendant import _token_spans

    return tuple(t[0] for t in _token_spans(value))


def _contracts(bindings: Sequence[Any], target: Mapping[str, Any]) -> tuple[_Contract, ...]:
    by_key = None
    result = []
    seen = set()
    for binding in bindings:
        if (
            not binding.dependency_paths
            or len(binding.target_paths) != 1
            or not (
                _MARK.fullmatch(binding.target_paths[0])
                or _HANDLING.fullmatch(binding.target_paths[0])
            )
            or binding.render_mode != "target_binding"
            or binding.derivation is not None
        ):
            continue
        path = binding.target_paths[0]
        owners = tuple(binding.dependency_paths)
        parties = [_PARTY.fullmatch(p) for p in owners]
        party_mark = bool(_MARK.fullmatch(path) and all(parties))
        facility = bool(
            _HANDLING.fullmatch(path) and len(owners) == 1 and _DESTINATION.fullmatch(owners[0])
        )
        if not party_mark and not facility:
            continue
        if by_key is None:
            by_key = {b.logical_key: b for b in bindings}
        if path in seen:
            raise ValueError("labelled context has duplicate target ownership: " + path)
        seen.add(path)
        if binding.realization.mode not in {"single_surface", "repeated_surface"}:
            raise ValueError("labelled context requires complete owned target surfaces")
        if len(set(owners)) != len(owners):
            raise ValueError("labelled context repeats a dependency path")
        dependencies = []
        for key in binding.dependency_bindings:
            if key not in by_key or key == binding.logical_key:
                raise ValueError("labelled context names an absent or cyclic owner binding")
            dependencies.append(by_key[key])
        if (
            not dependencies
            or len(set(binding.dependency_bindings)) != len(dependencies)
            or any(sum(p in b.target_paths for b in dependencies) != 1 for p in owners)
            or any(not set(b.target_paths).intersection(owners) for b in dependencies)
        ):
            raise ValueError("labelled context paths and declared owner bindings disagree")
        original = _text(target, path)
        if any(_tokens(slot.source_text) != _tokens(original) for slot in binding.occurrences):
            raise ValueError("labelled context source occurrence does not express its whole label")
        if facility:
            # Explicit visible route owner, not inferred warehouse locality.
            _text(target, owners[0])
            if _WAREHOUSE.fullmatch(original) is None:
                raise ValueError(
                    "destination facility requires the reviewed transit/warehouse frame"
                )
            result.append(_Contract(path, owners, "synthetic_destination_warehouse", original))
            continue
        party_names = {match["party"] for match in parties if match is not None}
        fields = tuple(match["field"] for match in parties if match is not None)
        if len(party_names) != 1:
            raise ValueError("one party mark cannot combine different party owners")
        if fields == ("name",):
            if _tokens(original) != _tokens(_text(target, owners[0])):
                raise ValueError("party name mark does not match its declared source owner")
            result.append(_Contract(path, owners, "party_name", original))
        elif set(fields) == {"city", "country"} and len(fields) == 2:
            from .descendant import _token_spans

            source_spans = _token_spans(original)
            tokens = tuple(t[0] for t in source_spans)
            spans = []
            covered: set[int] = set()
            for owner in owners:
                expected = _tokens(_text(target, owner))
                matches = [
                    i
                    for i in range(len(tokens) - len(expected) + 1)
                    if tokens[i : i + len(expected)] == expected
                ]
                if len(matches) != 1 or not expected:
                    raise ValueError("party locality mark lacks unique visible owner components")
                start = matches[0]
                indices = set(range(start, start + len(expected)))
                if covered.intersection(indices):
                    raise ValueError("party locality components overlap")
                covered.update(indices)
                spans.append((source_spans[start][1], source_spans[start + len(expected) - 1][2]))
            if covered != set(range(len(tokens))):
                raise ValueError("party locality mark contains unowned semantic content")
            result.append(_Contract(path, owners, "party_locality", original, tuple(spans)))
        else:
            raise ValueError("party mark requires one name or one city/country pair")
    return tuple(result)


def validate_source(bindings: Sequence[Any], target: Mapping[str, Any]) -> None:
    """Reject malformed explicit contracts before compiler certification."""
    _contracts(bindings, target)


def owned_paths(source: Any) -> frozenset[str]:
    """These labels are assembled after independent facts, never requested from an LLM."""
    return frozenset(c.target_path for c in _contracts(source.template.bindings, source.target))


def generated_values(source: Any, target: Mapping[str, Any]) -> dict[str, str]:
    """Derive new labels from accepted owners; never restore their original values."""
    from .descendant import _case_like

    result = {}
    for contract in _contracts(source.template.bindings, source.target):
        values = tuple(_text(target, p) for p in contract.owner_paths)
        if contract.kind == "party_name":
            value = _case_like(contract.source, values[0])
            # The mark's final punctuation can differ from its party-name label.
            if contract.source.endswith(".") and not value.endswith("."):
                value += "."
        elif contract.kind == "party_locality":
            value = contract.source
            for (start, end), replacement in sorted(
                zip(contract.spans, values, strict=True), reverse=True
            ):
                value = value[:start] + _case_like(value[start:end], replacement) + value[end:]
        else:
            frame = _WAREHOUSE.fullmatch(contract.source)
            assert frame is not None  # Source validation above proves the complete frame.
            value = frame["prefix"] + _case_like(frame["name"], values[0]) + frame["suffix"]
        result[contract.target_path] = value
    return result


def validate_final(source: Any, target: Mapping[str, Any]) -> None:
    """Reject stale or independently generated dependent labels without repairing them."""
    for path, expected in generated_values(source, target).items():
        if _text(target, path) != expected:
            raise ValueError(
                "accepted cargo label conflicts with its declared context owner: " + path
            )
