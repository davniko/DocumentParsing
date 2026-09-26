"""Document-task-specific target validation behind a small shared contract."""

from __future__ import annotations

import json
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Annotated, Any, Literal, cast

from pydantic import BaseModel, ConfigDict, StringConstraints, model_validator

from document_ocr.hashing import canonical_json_bytes, sha256_bytes
from document_ocr.label_schemas.bill_of_lading import BillOfLadingLabel
from document_ocr.label_schemas.bill_of_lading_v3 import (
    BillOfLadingRelationExplicitLabel,
    CategoryToken,
)
from document_ocr.label_schemas.bill_of_lading_v4 import BillOfLadingRelationExplicitV4Label
from document_ocr.label_schemas.bill_of_lading_v5 import BillOfLadingRelationExplicitV5Label
from document_ocr.label_schemas.bill_of_lading_v6 import BillOfLadingMPCIAlignedV6Label
from document_ocr.training.config import TrainingConfig, resolve_config_path

Canonicalizer = Callable[[dict[str, Any]], dict[str, Any]]
Sha256 = Annotated[str, StringConstraints(pattern=r"^[0-9a-f]{64}$")]


class RelationExplicitTaskConstraints(BaseModel):
    """Frozen category vocabulary required by the relation-explicit task."""

    model_config = ConfigDict(extra="forbid", frozen=True, strict=True, allow_inf_nan=False)

    schemaVersion: Literal[1]
    task: Literal[
        "bill_of_lading_relation_explicit_v3",
        "bill_of_lading_relation_explicit_v4",
        "bill_of_lading_relation_explicit_v5",
        "bill_of_lading_mpci_aligned_v6",
    ]
    basePromptSchemaSha256: Sha256
    targetSchemaSha256: Sha256
    packageRegistrySha256: Sha256
    containerRegistrySha256: Sha256
    packageCategoryTokens: tuple[CategoryToken, ...]
    containerCategoryTokens: tuple[CategoryToken, ...]

    @model_validator(mode="after")
    def vocabularies_are_unique_and_sorted(self) -> RelationExplicitTaskConstraints:
        for name in ("packageCategoryTokens", "containerCategoryTokens"):
            values = getattr(self, name)
            if values != tuple(sorted(set(values))):
                raise ValueError(f"{name} must be unique and sorted")
        return self


@dataclass(frozen=True, slots=True)
class TrainingTask:
    """A named schema boundary for one decoder-target family."""

    name: str
    canonicalizer: Canonicalizer
    target_model: type[BaseModel]
    constraints: RelationExplicitTaskConstraints | None = None

    def _base_prompt_schema(self) -> dict[str, Any]:
        return cast(
            dict[str, Any],
            _sparse_prompt_schema(self.target_model.model_json_schema(mode="serialization")),
        )

    def base_prompt_schema_sha256(self) -> str:
        return sha256_bytes(canonical_json_bytes(self._base_prompt_schema()))

    def prompt_schema_json(self) -> str:
        """Return a compact, sparse JSON Schema derived from the target model."""

        schema = self._base_prompt_schema()
        if self.constraints is not None:
            definitions = schema.get("$defs")
            if not isinstance(definitions, dict):
                raise ValueError("relation-explicit prompt schema has no $defs map")
            container_definition = (
                "ContainerInformationV6"
                if self.name == "bill_of_lading_mpci_aligned_v6"
                else "RelationExplicitContainerV5"
                if self.name == "bill_of_lading_relation_explicit_v5"
                else "RelationExplicitContainer"
            )
            constrained_fields = (
                (
                    container_definition,
                    self.constraints.containerCategoryTokens,
                ),
                (
                    "NumberAndTypeOfPackagesV6"
                    if self.name == "bill_of_lading_mpci_aligned_v6"
                    else "CargoPackageFact",
                    self.constraints.packageCategoryTokens,
                ),
            )
            for definition_name, tokens in constrained_fields:
                definition = definitions.get(definition_name)
                properties = definition.get("properties") if isinstance(definition, dict) else None
                if not isinstance(properties, dict) or "typeCategory" not in properties:
                    raise ValueError(
                        f"relation-explicit prompt schema lacks {definition_name}.typeCategory"
                    )
                if tokens:
                    properties["typeCategory"] = {
                        "enum": list(tokens),
                        "type": "string",
                    }
                else:
                    del properties["typeCategory"]
        ordered = _sorted_json_value(schema)
        if self.name == "bill_of_lading_relation_explicit_v5":
            properties = ordered["$defs"]["RelationExplicitDocumentPatchV5"]["properties"]
            properties["cargoAllocationGroups"] = properties.pop("cargoAllocationGroups")
        if self.name == "bill_of_lading_mpci_aligned_v6":
            properties = ordered["$defs"]["GoodsItemDetailsV6"]["properties"]
            if "splitGoodsPlacement" in properties:
                properties["splitGoodsPlacement"] = properties.pop("splitGoodsPlacement")
        return json.dumps(
            ordered,
            allow_nan=False,
            ensure_ascii=False,
            sort_keys=False,
            separators=(",", ":"),
        )

    def canonicalize(self, value: dict[str, Any]) -> dict[str, Any]:
        canonical = self.canonicalizer(value)
        if self.constraints is None:
            return canonical
        patch = canonical["documentPatch"]
        package_tokens = set(self.constraints.packageCategoryTokens)
        container_tokens = set(self.constraints.containerCategoryTokens)
        container_field = (
            "containerInformation"
            if self.name == "bill_of_lading_mpci_aligned_v6"
            else "containers"
        )
        for container in patch.get(container_field, []):
            token = container.get("typeCategory")
            if token is not None and token not in container_tokens:
                raise ValueError(
                    f"container typeCategory is outside the frozen vocabulary: {token}"
                )
        packages = (
            (
                package
                for goods in patch.get("goodsItemDetails", [])
                for package in goods.get("numberAndTypeOfPackages", [])
            )
            if self.name == "bill_of_lading_mpci_aligned_v6"
            else patch.get("cargoPackages", [])
        )
        for package in packages:
            token = package.get("typeCategory")
            if token is not None and token not in package_tokens:
                raise ValueError(f"package typeCategory is outside the frozen vocabulary: {token}")
        return canonical

    def bind_constraints(self, constraints: RelationExplicitTaskConstraints) -> TrainingTask:
        if self.name != constraints.task:
            raise ValueError("task-constraints artifact names a different training task")
        if self.base_prompt_schema_sha256() != constraints.basePromptSchemaSha256:
            raise ValueError("task-constraints base prompt schema SHA-256 differs from code")
        target_schema_sha256 = sha256_bytes(
            canonical_json_bytes(self.target_model.model_json_schema(mode="serialization"))
        )
        if target_schema_sha256 != constraints.targetSchemaSha256:
            raise ValueError("task-constraints target schema SHA-256 differs from code")
        return TrainingTask(self.name, self.canonicalizer, self.target_model, constraints)


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
                name: _sparse_prompt_schema(child_schema) for name, child_schema in item.items()
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


def _sorted_json_value(value: Any) -> Any:
    """Preserve canonical ordering while allowing an explicit decoder field order."""

    if isinstance(value, dict):
        return {key: _sorted_json_value(value[key]) for key in sorted(value)}
    if isinstance(value, list):
        return [_sorted_json_value(item) for item in value]
    return value


def canonical_json(value: dict[str, Any]) -> str:
    """Serialize one validated target into the exact decoder representation."""

    ordered = _sorted_json_value(value)
    if value.get("schemaVersion") == "5.0.0-experimental":
        patch = ordered.get("documentPatch")
        if isinstance(patch, dict) and "cargoAllocationGroups" in patch:
            patch["cargoAllocationGroups"] = patch.pop("cargoAllocationGroups")
    if value.get("schemaVersion") == "6.0.0-experimental":
        patch = ordered.get("documentPatch")
        if isinstance(patch, dict):
            for goods in patch.get("goodsItemDetails", []):
                if isinstance(goods, dict) and "splitGoodsPlacement" in goods:
                    goods["splitGoodsPlacement"] = goods.pop("splitGoodsPlacement")
    return json.dumps(
        ordered,
        allow_nan=False,
        ensure_ascii=False,
        sort_keys=False,
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


def _canonicalize_bill_of_lading_relation_explicit_v3(
    value: dict[str, Any],
) -> dict[str, Any]:
    encoded = canonical_json(value)
    label = BillOfLadingRelationExplicitLabel.model_validate_json(encoded, strict=True)
    canonical = label.canonical_target()
    if canonical != value:
        raise ValueError("target differs from the task schema's canonical sparse representation")
    return canonical


def _canonicalize_bill_of_lading_relation_explicit_v4(
    value: dict[str, Any],
) -> dict[str, Any]:
    encoded = canonical_json(value)
    label = BillOfLadingRelationExplicitV4Label.model_validate_json(encoded, strict=True)
    canonical = label.canonical_target()
    if canonical != value:
        raise ValueError("target differs from the task schema's canonical sparse representation")
    return canonical


def _canonicalize_bill_of_lading_relation_explicit_v5(
    value: dict[str, Any],
) -> dict[str, Any]:
    encoded = canonical_json(value)
    label = BillOfLadingRelationExplicitV5Label.model_validate_json(encoded, strict=True)
    canonical = label.canonical_target()
    if canonical != value:
        raise ValueError("target differs from the task schema's canonical sparse representation")
    return canonical


def _canonicalize_bill_of_lading_mpci_aligned_v6(value: dict[str, Any]) -> dict[str, Any]:
    encoded = canonical_json(value)
    label = BillOfLadingMPCIAlignedV6Label.model_validate_json(encoded, strict=True)
    canonical = label.canonical_target()
    if canonical != value:
        raise ValueError("target differs from the task schema's canonical sparse representation")
    return canonical


_TASKS = {
    "bill_of_lading_semantic_v2": TrainingTask(
        name="bill_of_lading_semantic_v2",
        canonicalizer=_canonicalize_bill_of_lading,
        target_model=BillOfLadingLabel,
    ),
    "bill_of_lading_relation_explicit_v3": TrainingTask(
        name="bill_of_lading_relation_explicit_v3",
        canonicalizer=_canonicalize_bill_of_lading_relation_explicit_v3,
        target_model=BillOfLadingRelationExplicitLabel,
    ),
    "bill_of_lading_relation_explicit_v4": TrainingTask(
        name="bill_of_lading_relation_explicit_v4",
        canonicalizer=_canonicalize_bill_of_lading_relation_explicit_v4,
        target_model=BillOfLadingRelationExplicitV4Label,
    ),
    "bill_of_lading_relation_explicit_v5": TrainingTask(
        name="bill_of_lading_relation_explicit_v5",
        canonicalizer=_canonicalize_bill_of_lading_relation_explicit_v5,
        target_model=BillOfLadingRelationExplicitV5Label,
    ),
    "bill_of_lading_mpci_aligned_v6": TrainingTask(
        name="bill_of_lading_mpci_aligned_v6",
        canonicalizer=_canonicalize_bill_of_lading_mpci_aligned_v6,
        target_model=BillOfLadingMPCIAlignedV6Label,
    ),
}


def get_training_task(name: str) -> TrainingTask:
    """Resolve an explicitly registered document task."""

    try:
        return _TASKS[name]
    except KeyError as error:
        raise ValueError(f"unsupported training task: {name!r}") from error


def load_training_task(project_root: Path, config: TrainingConfig) -> TrainingTask:
    """Load a registered task and bind its hash-pinned constraints, when required."""

    task = get_training_task(config.task)
    configured = config.task_constraints
    if configured is None:
        return task
    unresolved = resolve_config_path(project_root, configured.path)
    if unresolved.is_symlink():
        raise ValueError(f"task-constraints path must not be a symbolic link: {unresolved}")
    try:
        path = unresolved.resolve(strict=True)
    except FileNotFoundError as error:
        raise ValueError(f"task-constraints file does not exist: {unresolved}") from error
    if not path.is_file():
        raise ValueError(f"task-constraints path is not a regular file: {path}")
    payload = path.read_bytes()
    if sha256_bytes(payload) != configured.sha256:
        raise ValueError(f"task-constraints SHA-256 mismatch: {path}")
    try:
        constraints = RelationExplicitTaskConstraints.model_validate_json(payload, strict=True)
    except ValueError as error:
        raise ValueError(f"task-constraints artifact failed validation: {path}") from error
    canonical_payload = canonical_json_bytes(constraints.model_dump(mode="json")) + b"\n"
    if payload != canonical_payload:
        raise ValueError("task-constraints artifact must use canonical JSON plus one newline")
    return task.bind_constraints(constraints)
