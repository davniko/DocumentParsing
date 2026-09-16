from __future__ import annotations

import re
from collections.abc import Mapping
from typing import Any, cast

_LINE_ID = re.compile(r"^L(?P<number>[0-9]{5})$")
BINDING_COLUMNS = (
    "bindingId",
    "sourceBindingIds",
    "logicalKey",
    "renderMode",
    "valueKind",
    "groupKind",
    "groupKey",
    "targetPathIds",
    "targetRelationship",
    "independentTargetFactComponentPathIds",
    "derivation",
    "dependencyPathIds",
    "dependencyBindings",
    "occurrenceIds",
)
OCCURRENCE_COLUMNS = (
    "occurrenceId",
    "sourceBindingId",
    "lineStartNumber",
    "lineEndNumber",
    "sourceText",
    "occurrenceIndex",
    "exactMatchCount",
    "exactMatchCandidates",
)


def _line_number(value: Any) -> int:
    if not isinstance(value, str) or (match := _LINE_ID.fullmatch(value)) is None:
        raise ValueError(f"invalid line ID in critic inventory: {value!r}")
    return int(match.group("number"))


def compact_critic_payload(payload: Mapping[str, Any]) -> dict[str, Any]:
    """Losslessly encode the critic's large repeated inventories with short references."""

    required = {
        "allowedTargetPaths",
        "allowedRemovalLogicalKeys",
        "bindingInventory",
        "maskedTemplate",
    }
    missing = sorted(required - payload.keys())
    if missing:
        raise ValueError(f"critic payload lacks required keys: {missing}")
    target_paths = payload["allowedTargetPaths"]
    bindings = payload["bindingInventory"]
    removals = payload["allowedRemovalLogicalKeys"]
    masked_template = payload["maskedTemplate"]
    annotated_source = payload.get("annotatedSource")
    if (
        not isinstance(target_paths, (list, tuple))
        or not isinstance(bindings, (list, tuple))
        or not isinstance(removals, (list, tuple))
        or not isinstance(masked_template, str)
        or (annotated_source is not None and not isinstance(annotated_source, str))
    ):
        raise ValueError("critic payload has invalid compactable field types")
    if len(set(target_paths)) != len(target_paths):
        raise ValueError("critic target paths are not unique")
    path_ids = {value: f"path_{index:04d}" for index, value in enumerate(target_paths)}
    logical_keys = [row.get("logicalKey") for row in bindings]
    if any(not isinstance(value, str) for value in logical_keys):
        raise ValueError("critic binding inventory has a non-text logical key")
    if len(set(logical_keys)) != len(logical_keys):
        raise ValueError("critic binding logical keys are not unique")
    binding_ids = {value: f"binding_{index:04d}" for index, value in enumerate(logical_keys)}
    if any(value not in binding_ids for value in removals):
        raise ValueError("critic removal key is absent from the binding inventory")

    source_binding_ids: list[str] = []
    source_binding_index: dict[str, int] = {}

    def source_index(value: Any) -> int:
        if not isinstance(value, str):
            raise ValueError("critic source binding ID is not text")
        if value not in source_binding_index:
            source_binding_index[value] = len(source_binding_ids)
            source_binding_ids.append(value)
        return source_binding_index[value]

    occurrence_rows: list[list[Any]] = []
    binding_rows: list[list[Any]] = []
    compact_template = masked_template
    compact_annotated_source = annotated_source
    expected_binding_keys = {
        "sourceBindingIds",
        "logicalKey",
        "renderMode",
        "valueKind",
        "groupKind",
        "groupKey",
        "targetPaths",
        "targetRelationship",
        "independentTargetFactComponents",
        "derivation",
        "dependencyPaths",
        "dependencyBindings",
        "occurrences",
    }
    expected_occurrence_keys = {
        "sourceBindingId",
        "lineStart",
        "lineEnd",
        "sourceText",
        "occurrenceIndex",
        "exactMatchCount",
        "exactMatchCandidates",
    }
    for binding_index, binding in enumerate(bindings):
        if not isinstance(binding, dict) or set(binding) != expected_binding_keys:
            raise ValueError("critic binding inventory shape differs from the pinned contract")
        logical_key = cast(str, binding["logicalKey"])
        binding_id = binding_ids[logical_key]
        render_mode = binding["renderMode"]
        marker = f"⟦{logical_key}:{render_mode}⟧"
        if marker not in compact_template:
            raise ValueError(f"binding marker is absent from masked template: {logical_key}")
        compact_template = compact_template.replace(marker, f"⟦{binding_id}⟧")
        if compact_annotated_source is not None:
            if marker not in compact_annotated_source:
                raise ValueError(f"binding marker is absent from annotated source: {logical_key}")
            compact_annotated_source = compact_annotated_source.replace(marker, f"⟦{binding_id}⟧")
        occurrence_ids: list[str] = []
        for occurrence in binding["occurrences"]:
            if not isinstance(occurrence, dict) or set(occurrence) != expected_occurrence_keys:
                raise ValueError("critic occurrence shape differs from the pinned contract")
            # This hot loop's percent formatter benchmarks materially faster than an f-string.
            occurrence_id = "occ_%s_%05d" % (  # noqa: UP031
                occurrence["lineStart"],
                len(occurrence_rows),
            )
            occurrence_ids.append(occurrence_id)
            occurrence_rows.append(
                [
                    occurrence_id,
                    source_index(occurrence["sourceBindingId"]),
                    _line_number(occurrence["lineStart"]),
                    _line_number(occurrence["lineEnd"]),
                    occurrence["sourceText"],
                    occurrence["occurrenceIndex"],
                    occurrence["exactMatchCount"],
                    occurrence["exactMatchCandidates"],
                ]
            )

        def path_id(value: Any) -> str:
            if value not in path_ids:
                raise ValueError(f"binding path is absent from allowed target paths: {value!r}")
            return path_ids[value]

        binding_rows.append(
            [
                binding_id,
                [source_index(value) for value in binding["sourceBindingIds"]],
                logical_key,
                render_mode,
                binding["valueKind"],
                binding["groupKind"],
                binding["groupKey"],
                [path_id(value) for value in binding["targetPaths"]],
                binding["targetRelationship"],
                [
                    [path_id(value) for value in component]
                    for component in binding["independentTargetFactComponents"]
                ],
                binding["derivation"],
                [path_id(value) for value in binding["dependencyPaths"]],
                binding["dependencyBindings"],
                occurrence_ids,
            ]
        )
        if binding_id != f"binding_{binding_index:04d}":
            raise AssertionError("non-contiguous compact binding IDs")

    output = {
        key: value
        for key, value in payload.items()
        if key
        not in {
            "allowedTargetPaths",
            "allowedRemovalLogicalKeys",
            "bindingInventory",
            "maskedTemplate",
            "annotatedSource",
        }
    }
    output.update(
        {
            "compactContract": {
                "schemaVersion": 3,
                "lineIdFormat": "L%05d",
                "occurrenceIdFormat": "occ_L%05d_%05d",
                "bindingColumns": BINDING_COLUMNS,
                "occurrenceColumns": OCCURRENCE_COLUMNS,
                "sourceBindingIdEncoding": "zero-based index into sourceBindingIdTable",
            },
            "targetPathTable": [[path_ids[value], value] for value in target_paths],
            "sourceBindingIdTable": source_binding_ids,
            "occurrenceRows": occurrence_rows,
            "bindingRows": binding_rows,
            # Retain exact logical keys because the response schema enumerates this field.
            "allowedRemovalLogicalKeys": list(removals),
            "maskedTemplate": compact_template,
        }
    )
    if compact_annotated_source is not None:
        output["annotatedSource"] = compact_annotated_source
    return output


def compact_critic_semantic_counts(payload: Mapping[str, Any]) -> tuple[int, int, int]:
    """Return stable table cardinalities for runtime receipts and tests."""

    return (
        len(cast(list[Any], payload["targetPathTable"])),
        len(cast(list[Any], payload["bindingRows"])),
        len(cast(list[Any], payload["occurrenceRows"])),
    )
