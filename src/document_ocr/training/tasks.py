"""Document-task-specific target validation behind a small shared contract."""

from __future__ import annotations

import json
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

from pydantic import BaseModel

from document_ocr.label_schemas.bill_of_lading import BillOfLadingLabel

Canonicalizer = Callable[[dict[str, Any]], dict[str, Any]]


@dataclass(frozen=True, slots=True)
class TrainingTask:
    """A named schema boundary for one decoder-target family."""

    name: str
    canonicalize: Canonicalizer
    target_model: type[BaseModel]

    def prompt_schema_json(self) -> str:
        """Return a compact, sparse JSON Schema derived from the target model."""

        schema = _sparse_prompt_schema(self.target_model.model_json_schema(mode="serialization"))
        return json.dumps(
            schema,
            allow_nan=False,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        )


_PROMPT_SCHEMA_NOISE = frozenset({"default", "description", "title"})
_PROMPT_SCHEMA_NAMED_MAPS = frozenset(
    {"$defs", "definitions", "dependentSchemas", "patternProperties", "properties"}
)


def _sparse_prompt_schema(value: Any) -> Any:
    """Remove presentation noise and null branches from a sparse target schema."""

    if isinstance(value, list):
        return [_sparse_prompt_schema(item) for item in value]
    if not isinstance(value, dict):
        return value

    compact: dict[str, Any] = {}
    for key, item in value.items():
        if key in _PROMPT_SCHEMA_NOISE:
            continue
        if key in _PROMPT_SCHEMA_NAMED_MAPS and isinstance(item, dict):
            # Keys inside schema maps are model field/definition names, not JSON Schema
            # annotations. A real field named `description` must therefore survive.
            compact[key] = {
                name: _sparse_prompt_schema(child_schema)
                for name, child_schema in item.items()
            }
        else:
            compact[key] = _sparse_prompt_schema(item)
    alternatives = compact.get("anyOf")
    if not isinstance(alternatives, list):
        return compact

    non_null = [item for item in alternatives if item != {"type": "null"}]
    if len(non_null) == len(alternatives):
        return compact
    if not non_null:
        raise ValueError("target prompt schema contains a null-only field")
    if len(non_null) > 1:
        compact["anyOf"] = non_null
        return compact

    wrapper = {key: item for key, item in compact.items() if key != "anyOf"}
    overlap = set(wrapper) & set(non_null[0])
    if overlap:
        raise ValueError(
            "nullable target schema cannot be compacted without overwriting keys: "
            + ", ".join(sorted(overlap))
        )
    return {**wrapper, **non_null[0]}


def canonical_json(value: dict[str, Any]) -> str:
    """Serialize one validated target into the exact decoder representation."""

    return json.dumps(
        value,
        allow_nan=False,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )


def _canonicalize_bill_of_lading(value: dict[str, Any]) -> dict[str, Any]:
    # Strict JSON validation is intentional: JSON date strings and arrays are valid wire values,
    # whereas Pydantic's strict Python-object path correctly refuses to coerce them.
    encoded = canonical_json(value)
    label = BillOfLadingLabel.model_validate_json(encoded, strict=True)
    canonical = label.canonical_target()
    if canonical != value:
        raise ValueError("target differs from the task schema's canonical sparse representation")
    return canonical


_TASKS = {
    "bill_of_lading_semantic_v2": TrainingTask(
        name="bill_of_lading_semantic_v2",
        canonicalize=_canonicalize_bill_of_lading,
        target_model=BillOfLadingLabel,
    )
}


def get_training_task(name: str) -> TrainingTask:
    """Resolve an explicitly registered document task."""

    try:
        return _TASKS[name]
    except KeyError as error:
        raise ValueError(f"unsupported training task: {name!r}") from error
