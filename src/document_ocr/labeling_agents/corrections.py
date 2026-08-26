"""Deterministic, reviewer-target-scoped correction merging."""

from __future__ import annotations

import re
from collections.abc import Mapping
from copy import deepcopy
from functools import lru_cache
from typing import Any

from pydantic import ValidationError

from document_ocr.hashing import canonical_json_bytes
from document_ocr.label_schemas.bill_of_lading_agent_v4 import (
    AgentRelationExplicitDocumentPatch,
)
from document_ocr.labeling_agents.models import CompactAnnotationDraft

_PATH_TOKEN = re.compile(r"\.([A-Za-z_][A-Za-z0-9_]*)|\[([0-9]+)\]")
_MISSING = object()


class TargetScopedCorrectionError(ValueError):
    """A reviewer correction cannot be merged without changing uncited facts."""


def _path_tokens(path: str) -> tuple[str | int, ...]:
    if path == "documentType":
        return (path,)
    if path == "warnings" or path.startswith(("warnings.", "warnings[")):
        raise TargetScopedCorrectionError(
            f"correction path targets warnings sidecar instead of a label fact: {path}"
        )
    if path == "decisionNotes" or path.startswith(("decisionNotes.", "decisionNotes[")):
        raise TargetScopedCorrectionError(
            f"correction path targets decisionNotes sidecar instead of a label fact: {path}"
        )

    prefix = "documentPatch"
    if not path.startswith(prefix):
        raise TargetScopedCorrectionError(
            f"correction path must start with documentPatch or equal documentType: {path}"
        )
    tokens: list[str | int] = [prefix]
    position = len(prefix)
    for match in _PATH_TOKEN.finditer(path, position):
        if match.start() != position:
            raise TargetScopedCorrectionError(f"correction path is malformed: {path}")
        field, index = match.groups()
        tokens.append(field if field is not None else int(index))
        position = match.end()
    if position != len(path) or len(tokens) < 2:
        raise TargetScopedCorrectionError(f"correction path is malformed: {path}")
    if len(tokens) >= 2 and tokens[1] in {"warnings", "decisionNotes"}:
        raise TargetScopedCorrectionError(
            f"correction path targets {tokens[1]} sidecar instead of a label fact: {path}"
        )
    return tuple(tokens)


def _format_path(tokens: tuple[str | int, ...]) -> str:
    value = str(tokens[0])
    for token in tokens[1:]:
        value += f"[{token}]" if isinstance(token, int) else f".{token}"
    return value


def _stable_scope(tokens: tuple[str | int, ...]) -> tuple[str | int, ...]:
    # A list element cannot be deleted or reordered by replacing the element at
    # its old numeric index. The indivisible operation is the parent collection.
    # Scalar fields inside an indexed object remain narrow corrections.
    if isinstance(tokens[-1], int):
        return tokens[:-1]
    # Allocation rows are a discriminated union. Changing coverage necessarily
    # changes which sibling keys are legal, so the indivisible correction unit
    # is the complete source-ordered allocation group.
    if len(tokens) >= 3 and tokens[1] == "cargoAllocationGroups" and isinstance(tokens[2], int):
        return tokens[:3]
    # GoodsOrigin is an optional, content-bearing object: Pydantic rejects an
    # object whose name and identifier are both null. Removing its final cited
    # value must therefore replace the complete origin atomically with null,
    # rather than leave an invalid empty parent behind.
    if (
        len(tokens) == 5
        and tokens[1] == "cargoGroups"
        and isinstance(tokens[2], int)
        and tokens[3] == "origin"
        and tokens[4] in {"name", "identifier"}
    ):
        return tokens[:4]
    return tokens


@lru_cache(maxsize=1)
def _document_patch_schema() -> dict[str, Any]:
    return AgentRelationExplicitDocumentPatch.model_json_schema()


def _resolve_schema_reference(root: dict[str, Any], reference: str) -> dict[str, Any]:
    if not reference.startswith("#/"):
        return {}
    node: Any = root
    for component in reference[2:].split("/"):
        component = component.replace("~1", "/").replace("~0", "~")
        if not isinstance(node, dict) or component not in node:
            return {}
        node = node[component]
    return node if isinstance(node, dict) else {}


def _schema_variants(
    schema: dict[str, Any], root: dict[str, Any], *, seen: frozenset[str] = frozenset()
) -> tuple[dict[str, Any], ...]:
    reference = schema.get("$ref")
    if isinstance(reference, str):
        if reference in seen:
            return ()
        resolved = _resolve_schema_reference(root, reference)
        return _schema_variants(resolved, root, seen=seen | {reference})
    for keyword in ("anyOf", "oneOf"):
        branches = schema.get(keyword)
        if isinstance(branches, list):
            return tuple(
                variant
                for branch in branches
                if isinstance(branch, dict)
                for variant in _schema_variants(branch, root, seen=seen)
                if variant.get("type") != "null"
            )
    return (schema,)


def _schema_contains_path(tokens: tuple[str | int, ...]) -> bool:
    if tokens == ("documentType",):
        return True
    if not tokens or tokens[0] != "documentPatch" or len(tokens) < 2:
        return False

    root = _document_patch_schema()
    variants: tuple[dict[str, Any], ...] = (root,)
    for token in tokens[1:]:
        next_variants: list[dict[str, Any]] = []
        for variant in variants:
            for expanded in _schema_variants(variant, root):
                if isinstance(token, str):
                    properties = expanded.get("properties")
                    child = properties.get(token) if isinstance(properties, dict) else None
                else:
                    child = expanded.get("items") if expanded.get("type") == "array" else None
                if isinstance(child, dict):
                    next_variants.extend(_schema_variants(child, root))
        variants = tuple(next_variants)
        if not variants:
            return False
    return True


def _value_schema_for_tokens(tokens: tuple[str | int, ...]) -> dict[str, Any]:
    if tokens == ("documentType",):
        return {
            "type": "string",
            "enum": ["bill_of_lading", "sea_waybill"],
            "description": "Corrected maritime document class.",
        }

    root = _document_patch_schema()
    candidates: tuple[dict[str, Any], ...] = (root,)
    for token in tokens[1:]:
        children: list[dict[str, Any]] = []
        for candidate in candidates:
            for expanded in _schema_variants(candidate, root):
                if isinstance(token, str):
                    properties = expanded.get("properties")
                    child = properties.get(token) if isinstance(properties, dict) else None
                else:
                    child = expanded.get("items") if expanded.get("type") == "array" else None
                if isinstance(child, dict):
                    children.append(deepcopy(child))
        if not children:
            raise TargetScopedCorrectionError(
                f"correction path does not have a value schema: {_format_path(tokens)}"
            )
        candidates = tuple(children)

    unique: list[dict[str, Any]] = []
    encoded: set[bytes] = set()
    for candidate in candidates:
        key = canonical_json_bytes(candidate)
        if key not in encoded:
            encoded.add(key)
            unique.append(candidate)
    return unique[0] if len(unique) == 1 else {"anyOf": unique}


def correction_envelope_json_schema(paths: tuple[str, ...]) -> dict[str, Any]:
    """Build an exact-path correction schema for one normalized reviewer scope set.

    OpenAI strict mode requires every declared property. Declaring only the
    normalized paths therefore makes omission impossible without forcing null
    placeholders for unrelated fields. A required nullable value remains
    distinguishable from an omitted path.
    """

    normalized_paths = normalize_correction_paths(paths)
    if not normalized_paths:
        raise TargetScopedCorrectionError("correction schema requires at least one path")
    root = _document_patch_schema()
    schema: dict[str, Any] = {
        "type": "object",
        "properties": {
            "corrections": {
                "type": "object",
                "properties": {
                    path: _value_schema_for_tokens(_path_tokens(path)) for path in normalized_paths
                },
                "required": list(normalized_paths),
                "additionalProperties": False,
                "description": (
                    "Exactly one replacement value for every reviewer-authorized normalized "
                    "correction path."
                ),
            }
        },
        "required": ["corrections"],
        "additionalProperties": False,
        "title": "BillOfLadingExactPathCorrection",
    }
    if definitions := root.get("$defs"):
        schema["$defs"] = deepcopy(definitions)
    return schema


def _contract_issue_message(errors: list[str]) -> str:
    details = "; ".join(f"{index}. {message}" for index, message in enumerate(errors, start=1))
    return f"correction contract found {len(errors)} issue(s): {details}"


def _raise_contract_issues(errors: list[str]) -> None:
    if errors:
        raise TargetScopedCorrectionError(_contract_issue_message(errors))


@lru_cache(maxsize=512)
def _validated_stable_scope(path: str) -> tuple[str | int, ...]:
    tokens = _path_tokens(path)
    if not _schema_contains_path(tokens):
        raise TargetScopedCorrectionError(
            f"correction path does not exist in the extraction label schema: {path}"
        )
    return _stable_scope(tokens)


def normalize_correction_paths(paths: tuple[str, ...]) -> tuple[str, ...]:
    """Return unique minimal paths that form the authorized correction surface."""

    scopes: list[tuple[str | int, ...]] = []
    errors: list[str] = []
    for path in paths:
        try:
            scopes.append(_validated_stable_scope(path))
        except TargetScopedCorrectionError as error:
            errors.append(str(error))
    _raise_contract_issues(errors)

    unique: list[tuple[str | int, ...]] = []
    for scope in scopes:
        if scope not in unique:
            unique.append(scope)
    minimal = [
        scope
        for scope in unique
        if not any(len(parent) < len(scope) and scope[: len(parent)] == parent for parent in unique)
    ]
    return tuple(_format_path(scope) for scope in minimal)


def unchanged_correction_paths(
    base: CompactAnnotationDraft,
    revision: CompactAnnotationDraft,
    target_paths: tuple[str, ...],
) -> tuple[str, ...]:
    """Return reviewer-authorized scopes whose proposed values did not change.

    A null ancestor means the leaf is semantically absent, not that comparison
    is impossible. This matters when a correction creates a contactDetails or
    other optional object in order to add one exact leaf.
    """

    normalized_paths = normalize_correction_paths(target_paths)
    if not normalized_paths:
        raise TargetScopedCorrectionError("correction comparison requires at least one path")
    base_value = base.model_dump(mode="json")
    revision_value = revision.model_dump(mode="json")
    unchanged: list[str] = []
    errors: list[str] = []
    for path in normalized_paths:
        tokens = _path_tokens(path)
        try:
            before = _read_comparable_path(base_value, _source_tokens(tokens), path)
            _, after = _read_path(revision_value, _source_tokens(tokens), path)
        except TargetScopedCorrectionError as error:
            errors.append(str(error))
            continue
        if canonical_json_bytes(before) == canonical_json_bytes(after):
            unchanged.append(path)
    _raise_contract_issues(errors)
    return tuple(unchanged)


def _read_comparable_path(node: Any, tokens: tuple[str | int, ...], path: str) -> Any:
    """Read a correction target while treating a null ancestor as an absent leaf."""

    current = node
    for offset, token in enumerate(tokens):
        if current is None and offset < len(tokens):
            return None
        if isinstance(token, str) and isinstance(current, dict):
            current = current.get(token, _MISSING)
        elif isinstance(token, int) and isinstance(current, list) and token < len(current):
            current = current[token]
        else:
            current = _MISSING
        if current is _MISSING:
            raise TargetScopedCorrectionError(f"candidate does not contain correction path: {path}")
    return current


def _paths_overlap(left: tuple[str | int, ...], right: tuple[str | int, ...]) -> bool:
    shared = min(len(left), len(right))
    return left[:shared] == right[:shared]


def _regenerate_warning_sidecar(
    base_value: dict[str, Any], correction_scopes: tuple[tuple[str | int, ...], ...]
) -> list[dict[str, Any]]:
    """Retain only still-applicable, schema-valid warnings from the prior candidate.

    A correction response is authoritative only for its normalized label scopes.
    Model-authored replacement warnings are therefore ignored. Warnings attached
    to a changed scope are removed because their premise may no longer be true;
    unscoped and untouched warnings remain immutable.
    """

    regenerated: list[dict[str, Any]] = []
    seen: set[bytes] = set()
    for warning in base_value["warnings"]:
        target_path = warning.get("targetPath")
        if target_path is not None:
            try:
                warning_tokens = _path_tokens(target_path)
            except TargetScopedCorrectionError:
                continue
            if not _schema_contains_path(warning_tokens) or any(
                _paths_overlap(warning_tokens, scope) for scope in correction_scopes
            ):
                continue
        encoded = canonical_json_bytes(warning)
        if encoded not in seen:
            seen.add(encoded)
            regenerated.append(warning)
    return regenerated


def _source_tokens(tokens: tuple[str | int, ...]) -> tuple[str | int, ...]:
    if tokens == ("documentType",):
        return tokens
    return ("relationExplicitLabel", *tokens)


def _destination_tokens(tokens: tuple[str | int, ...]) -> tuple[str | int, ...]:
    if tokens == ("documentType",):
        return tokens
    return ("relationExplicitLabel", *tokens)


def _read_path(
    node: Any, tokens: tuple[str | int, ...], path: str
) -> tuple[tuple[str | int, ...], Any]:
    current = node
    for offset, token in enumerate(tokens):
        if isinstance(token, str) and isinstance(current, dict):
            current = current.get(token, _MISSING)
        elif isinstance(token, int) and isinstance(current, list) and token < len(current):
            current = current[token]
        else:
            current = _MISSING
        if current is _MISSING:
            raise TargetScopedCorrectionError(f"revision does not contain correction path: {path}")
        if current is None and offset < len(tokens) - 1:
            display_tokens = tokens[1:] if tokens[0] == "relationExplicitLabel" else tokens
            display_offset = offset if tokens[0] == "relationExplicitLabel" else offset + 1
            raise TargetScopedCorrectionError(
                f"revision contains a null ancestor for narrow correction path: {path}; "
                f"target {_format_path(display_tokens[:display_offset])} explicitly to remove it"
            )
    return tokens, current


def _write_path(
    node: Any,
    tokens: tuple[str | int, ...],
    replacement: Any,
    *,
    path: str,
) -> None:
    current = node
    for offset, token in enumerate(tokens[:-1]):
        next_token = tokens[offset + 1]
        if isinstance(token, str) and isinstance(current, dict):
            child = current.get(token)
            if child is None and isinstance(next_token, str):
                child = {}
                current[token] = child
            elif child is None:
                raise TargetScopedCorrectionError(
                    f"missing collection parent must be targeted as a whole: {path}"
                )
            current = child
        elif isinstance(token, int) and isinstance(current, list) and token < len(current):
            current = current[token]
        else:
            raise TargetScopedCorrectionError(
                f"base candidate does not contain correction parent: {path}"
            )

    final = tokens[-1]
    valid_mapping_target = isinstance(final, str) and isinstance(current, dict)
    valid_sequence_target = (
        isinstance(final, int) and isinstance(current, list) and final < len(current)
    )
    if valid_mapping_target or valid_sequence_target:
        current[final] = deepcopy(replacement)
    else:
        raise TargetScopedCorrectionError(
            f"base candidate does not contain correction target: {path}"
        )


def merge_target_scoped_correction(
    base: CompactAnnotationDraft,
    revision: CompactAnnotationDraft,
    target_paths: tuple[str, ...],
) -> CompactAnnotationDraft:
    """Merge only reviewer-authorized target paths into a prior valid candidate.

    The provider's full revision remains in its immutable transcript. This
    returned value is the candidate consumed downstream, preventing a repair
    from silently deleting or changing unrelated, already-reviewed facts.
    """

    if not target_paths:
        raise TargetScopedCorrectionError("target-scoped correction requires at least one path")
    normalized_paths = normalize_correction_paths(target_paths)
    base_value = base.model_dump(mode="json")
    revision_value = revision.model_dump(mode="json")
    replacements: list[tuple[str, tuple[str | int, ...], Any]] = []
    errors: list[str] = []
    for path in normalized_paths:
        tokens = _path_tokens(path)
        try:
            resolved_source_tokens, replacement = _read_path(
                revision_value,
                _source_tokens(tokens),
                path,
            )
            replacements.append(
                (
                    path,
                    resolved_source_tokens,
                    replacement,
                )
            )
        except TargetScopedCorrectionError as error:
            errors.append(str(error))
    _raise_contract_issues(errors)

    # model_dump() already returns a detached mutable tree; no second full-copy
    # is needed. Individual replacement values remain defensively deep-copied.
    merged_value = base_value
    errors = []
    for path, resolved_tokens, replacement in replacements:
        try:
            _write_path(merged_value, resolved_tokens, replacement, path=path)
        except TargetScopedCorrectionError as error:
            errors.append(str(error))
    _raise_contract_issues(errors)

    return _finalize_correction(base_value, merged_value, normalized_paths)


def merge_correction_values(
    base: CompactAnnotationDraft,
    correction_values: Mapping[str, Any],
    target_paths: tuple[str, ...],
) -> CompactAnnotationDraft:
    """Merge one exact replacement for every normalized authorized path.

    Values are attached directly to their exact destination tokens. In
    particular, a null value can clear only the explicitly authorized path and
    can never be reinterpreted as permission to remove one of its ancestors.
    """

    if not target_paths:
        raise TargetScopedCorrectionError("exact-path correction requires at least one path")
    normalized_paths = normalize_correction_paths(target_paths)
    expected = set(normalized_paths)
    actual = set(correction_values)
    errors = [
        *(
            f"correction response is missing authorized path: {path}"
            for path in normalized_paths
            if path not in actual
        ),
        *(
            f"correction response contains unauthorized path: {path}"
            for path in correction_values
            if path not in expected
        ),
    ]
    _raise_contract_issues(errors)

    base_value = base.model_dump(mode="json")
    merged_value = base_value
    errors = []
    for path in normalized_paths:
        try:
            _write_path(
                merged_value,
                _destination_tokens(_path_tokens(path)),
                correction_values[path],
                path=path,
            )
        except TargetScopedCorrectionError as error:
            errors.append(str(error))
    _raise_contract_issues(errors)
    return _finalize_correction(base_value, merged_value, normalized_paths)


def _finalize_correction(
    base_value: dict[str, Any],
    merged_value: dict[str, Any],
    normalized_paths: tuple[str, ...],
) -> CompactAnnotationDraft:
    correction_scopes = tuple(_path_tokens(path) for path in normalized_paths)
    merged_value["warnings"] = _regenerate_warning_sidecar(base_value, correction_scopes)
    merged_value["decisionNotes"] = [
        "Applied reviewer-authorized target-scoped correction to: " + ", ".join(normalized_paths)
    ]
    try:
        return CompactAnnotationDraft.model_validate_json(
            canonical_json_bytes(merged_value), strict=True
        )
    except ValidationError as error:
        validation_errors = [
            f"merged candidate {'.'.join(str(part) for part in row['loc'])}: {row['msg']}"
            for row in error.errors(include_url=False)
        ]
        raise TargetScopedCorrectionError(_contract_issue_message(validation_errors)) from error
