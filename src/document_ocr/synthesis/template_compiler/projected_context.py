"""Explicit same-entity geography components shared by lexical projections.

A country printed inside a company name can also own the country label. The
company's remaining spans are not permission to freeze that country. These
contracts bind both surfaces to one sampled fact, without geocoding or new labels.
"""

from __future__ import annotations

import re
from collections.abc import Mapping, Sequence
from copy import deepcopy
from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class ContextEdge:
    start: int
    end: int
    tokens: tuple[str, ...]
    owner_key: str
    target_path: str


def eligible_edges(binding: Any, bindings: Sequence[Any]) -> tuple[ContextEdge, ...]:
    from .descendant import _token_spans
    from .realization_contract import fixed_projection_ranges

    if len(binding.target_paths) != 1:
        return ()
    path = binding.target_paths[0]
    party = re.fullmatch(
        r"(documentPatch\.parties\.(?:[A-Za-z]+|notifyParties\[\d+\]))\.(?:name|address)", path
    )
    mark = re.fullmatch(r"(documentPatch\.cargoGroups\[\d+\])\.marksAndNumbers\[\d+\]", path)
    if party is None and mark is None:
        return ()
    owner_paths = (
        {party[1] + ".city", party[1] + ".country"}
        if party is not None
        else {mark[1] + ".origin.name"}
        if mark is not None
        else set()
    )
    result = []
    for start, end, tokens in fixed_projection_ranges(binding):
        if mark is not None:
            snapshots = binding.realization.target_values
            if len(snapshots) != 1 or not isinstance(snapshots[0].source_value, str):
                continue
            prefix = tuple(t[0] for t in _token_spans(snapshots[0].source_value))[:start]
            if prefix[-2:] != ("made", "in"):
                continue  # A destination/brand mark is not evidence of cargo origin.
        owners = []
        for owner in bindings:
            if owner is binding or len(owner.target_paths) != 1:
                continue
            owner_path = owner.target_paths[0]
            if owner_path not in owner_paths:
                continue
            snapshots = owner.realization.target_values
            if len(snapshots) != 1 or not isinstance(snapshots[0].source_value, str):
                continue
            if tuple(t[0] for t in _token_spans(snapshots[0].source_value)) != tokens:
                continue
            if not all(
                tuple(t[0] for t in _token_spans(s.source_text)) == tokens
                for s in owner.occurrences
            ):
                continue
            owners.append(owner)
        if len(owners) == 1:
            owner = owners[0]
            result.append(ContextEdge(start, end, tokens, owner.logical_key, owner.target_paths[0]))
    return tuple(result)


def declared_edges(binding: Any, bindings: Sequence[Any]) -> tuple[ContextEdge, ...]:
    from .realization_contract import fixed_projection_ranges

    if binding.realization.mode != "token_projected_surface":
        return ()
    if not any(
        p.startswith(("documentPatch.parties.", "documentPatch.cargoGroups["))
        for p in binding.target_paths
    ):
        return ()
    if not binding.dependency_paths or not binding.dependency_bindings:
        return ()
    if not fixed_projection_ranges(binding):
        return ()
    return tuple(
        e
        for e in eligible_edges(binding, bindings)
        if e.target_path in binding.dependency_paths and e.owner_key in binding.dependency_bindings
    )


def sampled_ranges(
    binding: Any,
    template: Any,
    target: Mapping[str, Any],
    ranges: Sequence[tuple[int, int, tuple[str, ...]]],
) -> tuple[tuple[int, int, tuple[str, ...]], ...]:
    from .descendant import _resolve_path, _token_spans

    edges = {(e.start, e.end): e for e in declared_edges(binding, template.bindings)}
    result = []
    for start, end, tokens in ranges:
        edge = edges.get((start, end))
        if edge is not None:
            value = _resolve_path(target, edge.target_path)
            if not isinstance(value, str) or not value.strip():
                raise ValueError("projected context owner has no sampled text")
            tokens = tuple(t[0] for t in _token_spans(value))
        result.append((start, end, tokens))
    return tuple(result)


def condition_requests(
    source: Any, fields: Sequence[Mapping[str, Any]], target: Mapping[str, Any]
) -> tuple[dict[str, Any], ...]:
    """Replace declared context obligations before linguistic generation, not after."""
    from .descendant import _resolve_path, _token_spans

    output = []
    for field in fields:
        revised = deepcopy(dict(field))
        substitutions: dict[str, str] = {}
        for binding in source.template.bindings:
            if not set(binding.target_paths).intersection(field["paths"]):
                continue
            for edge in declared_edges(binding, source.template.bindings):
                old = _resolve_path(source.target, edge.target_path)
                new = _resolve_path(target, edge.target_path)
                if not isinstance(old, str) or not isinstance(new, str):
                    raise ValueError("projected context values must be strings")
                if old in substitutions and substitutions[old] != new:
                    raise ValueError("one lexical frame has conflicting context owners")
                substitutions[old] = new
        if substitutions:
            replacements = {
                tuple(t[0] for t in _token_spans(k)): tuple(t[0] for t in _token_spans(v))
                for k, v in substitutions.items()
            }
            for constraint in revised["constraints"]:
                constraint["fixedLiteralTokens"] = [
                    list(replacements.get(tuple(tokens), tuple(tokens)))
                    for tokens in constraint.get("fixedLiteralTokens", [])
                ]
            if "hostAssembly" in revised:
                for side in ("prefix", "suffix"):
                    text = revised["hostAssembly"][side]
                    tokens = _token_spans(text)
                    edits = []
                    for old, new in substitutions.items():
                        expected = tuple(t[0] for t in _token_spans(old))
                        normalized = tuple(t[0] for t in tokens)
                        matches = [
                            i
                            for i in range(len(tokens) - len(expected) + 1)
                            if normalized[i : i + len(expected)] == expected
                        ]
                        if len(matches) > 1:
                            raise ValueError("projected context is ambiguous in its lexical frame")
                        for i in matches:
                            edits.append((tokens[i][1], tokens[i + len(expected) - 1][2], new))
                    for start, end, value in sorted(edits, reverse=True):
                        text = text[:start] + value + text[end:]
                    revised["hostAssembly"][side] = text
            revised["sampledContext"] = substitutions
        output.append(revised)
    return tuple(output)
