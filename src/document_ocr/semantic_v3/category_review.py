"""Publish exact, reviewed source-key categorical assignments for semantic v3."""

from __future__ import annotations

import os
from datetime import date
from pathlib import Path
from typing import Annotated, Literal, cast

from pydantic import Field, StringConstraints, field_validator, model_validator

from document_ocr.atomic import atomic_publish_json, read_regular_file_bytes
from document_ocr.config import load_strict_yaml_mapping
from document_ocr.hashing import sha256_bytes
from document_ocr.label_schemas.bill_of_lading_v3 import CategoryToken
from document_ocr.label_schemas.common import LabelSchemaModel
from document_ocr.semantic_v3.config import (
    FrozenArtifactInput,
    SemanticV3TransformConfig,
)
from document_ocr.semantic_v3.transform import (
    CategoryAssignments,
    CategoryRegistry,
    _canonical_file,
    _load_frozen_model,
    _source_inventory_sha256,
    _strict_json,
)

Sha256 = Annotated[str, StringConstraints(pattern=r"^[0-9a-f]{64}$")]
NonEmptyString = Annotated[str, StringConstraints(min_length=1)]
PositiveInteger = Annotated[int, Field(gt=0)]


class CategoryReviewError(RuntimeError):
    """A categorical review artifact cannot be proven complete and consistent."""


class FrozenJsonlInput(LabelSchemaModel):
    path: NonEmptyString
    sha256: Sha256
    rows: PositiveInteger

    @field_validator("path")
    @classmethod
    def path_is_absolute_jsonl(cls, value: str) -> str:
        path = Path(value)
        if not path.is_absolute() or path.suffix != ".jsonl":
            raise ValueError("review inventory path must be an absolute .jsonl file")
        return value


class PackageResolutionGroup(LabelSchemaModel):
    category_token: CategoryToken
    source_type_texts: list[NonEmptyString] = Field(min_length=1)

    @model_validator(mode="after")
    def source_texts_are_unique(self) -> PackageResolutionGroup:
        if len(self.source_type_texts) != len(set(self.source_type_texts)):
            raise ValueError("package resolution group contains duplicate source text")
        return self


class CategoryReviewConfig(LabelSchemaModel):
    schema_version: Literal[1]
    source_inventory_sha256: Sha256
    reviewed_by: NonEmptyString
    reviewed_on: date
    package_registry: FrozenArtifactInput
    container_registry: FrozenArtifactInput
    package_inventory: FrozenJsonlInput
    container_inventory: FrozenJsonlInput
    expected_package_occurrences: PositiveInteger
    expected_container_occurrences: PositiveInteger
    package_resolution_groups: list[PackageResolutionGroup] = Field(min_length=1)
    unresolved_package_type_texts: list[NonEmptyString]
    container_policy: Literal["preserve_printed_type_no_semantic_registry"]
    output_path: NonEmptyString

    @field_validator("output_path")
    @classmethod
    def output_is_absolute_json(cls, value: str) -> str:
        path = Path(value)
        if not path.is_absolute() or path.suffix != ".json":
            raise ValueError("category assignment output must be an absolute .json file")
        normalized = Path(os.path.abspath(value))
        if normalized == Path(normalized.anchor):
            raise ValueError("category assignment output must not be the filesystem root")
        return value

    @model_validator(mode="after")
    def decisions_are_unique(self) -> CategoryReviewConfig:
        resolved = [
            value
            for group in self.package_resolution_groups
            for value in group.source_type_texts
        ]
        unresolved = list(self.unresolved_package_type_texts)
        if len(resolved) != len(set(resolved)):
            raise ValueError("package source text is resolved more than once")
        if len(unresolved) != len(set(unresolved)):
            raise ValueError("unresolved package source text is duplicated")
        if set(resolved).intersection(unresolved):
            raise ValueError("package source text is both resolved and unresolved")
        return self


class _PackageInventoryRow(LabelSchemaModel):
    affectedDocuments: list[str] = Field(min_length=1)
    assignmentStatus: Literal["unassigned"]
    candidateCategoryToken: None
    occurrences: PositiveInteger
    sourceTypeCode: str | None
    sourceTypeText: str | None


class _ContainerInventoryRow(LabelSchemaModel):
    affectedDocuments: list[str] = Field(min_length=1)
    assignmentStatus: Literal["unassigned"]
    candidateCategoryToken: None
    occurrences: PositiveInteger
    sourceTypeCode: str | None
    sourceTypeDescription: str | None


def load_category_review_config(path: str | Path) -> CategoryReviewConfig:
    return CategoryReviewConfig.model_validate(load_strict_yaml_mapping(path), strict=True)


def _load_inventory[RowT: LabelSchemaModel](
    configured: FrozenJsonlInput,
    model: type[RowT],
) -> tuple[RowT, ...]:
    path = _canonical_file(Path(configured.path), context="categorical review inventory")
    payload = read_regular_file_bytes(path)
    if sha256_bytes(payload) != configured.sha256:
        raise CategoryReviewError(f"categorical review inventory SHA-256 mismatch: {path}")
    rows: list[RowT] = []
    for row_number, line in enumerate(payload.splitlines(), start=1):
        if not line:
            raise CategoryReviewError(f"blank categorical review row: {path}:{row_number}")
        value = _strict_json(line, context=f"{path}:{row_number}")
        try:
            rows.append(model.model_validate(value, strict=True))
        except ValueError as error:
            raise CategoryReviewError(
                f"categorical review row failed validation: {path}:{row_number}"
            ) from error
    if len(rows) != configured.rows:
        raise CategoryReviewError(
            f"categorical review row count differs: expected {configured.rows}, found {len(rows)}"
        )
    return tuple(rows)


def publish_category_assignments(
    transform_config: SemanticV3TransformConfig,
    review_config_path: str | Path,
) -> Path:
    """Publish an immutable, exact-coverage assignment artifact."""

    review_path = _canonical_file(Path(review_config_path), context="category review config")
    review_payload = read_regular_file_bytes(review_path)
    review = load_category_review_config(review_path)
    source_inventory_sha256 = _source_inventory_sha256(transform_config)
    if review.source_inventory_sha256 != source_inventory_sha256:
        raise CategoryReviewError("category review targets a different source inventory")

    package_registry, package_registry_sha256 = _load_frozen_model(
        review.package_registry, CategoryRegistry
    )
    container_registry, container_registry_sha256 = _load_frozen_model(
        review.container_registry, CategoryRegistry
    )
    if package_registry.registryKind != "package":
        raise CategoryReviewError("package registry has the wrong registryKind")
    if container_registry.registryKind != "container":
        raise CategoryReviewError("container registry has the wrong registryKind")
    if container_registry.entries:
        raise CategoryReviewError(
            "preserve-only container policy requires an empty fail-closed semantic registry"
        )

    package_rows = _load_inventory(review.package_inventory, _PackageInventoryRow)
    container_rows = _load_inventory(review.container_inventory, _ContainerInventoryRow)
    if sum(row.occurrences for row in package_rows) != review.expected_package_occurrences:
        raise CategoryReviewError("package occurrence count differs from reviewed expectation")
    if sum(row.occurrences for row in container_rows) != review.expected_container_occurrences:
        raise CategoryReviewError("container occurrence count differs from reviewed expectation")
    if any(row.sourceTypeText is None or row.sourceTypeCode is not None for row in package_rows):
        raise CategoryReviewError(
            "this reviewed package mapping requires text-only source keys"
        )

    registry_tokens = {row.categoryToken for row in package_registry.entries}
    token_by_text = {
        text: group.category_token
        for group in review.package_resolution_groups
        for text in group.source_type_texts
    }
    if not set(token_by_text.values()).issubset(registry_tokens):
        raise CategoryReviewError("package decision references a token absent from the registry")
    source_texts = {cast(str, row.sourceTypeText) for row in package_rows}
    decided_texts = set(token_by_text).union(review.unresolved_package_type_texts)
    if source_texts != decided_texts:
        missing = sorted(source_texts - decided_texts)
        extra = sorted(decided_texts - source_texts)
        raise CategoryReviewError(
            "package decisions do not exactly cover source keys: "
            f"missing={missing!r}, extra={extra!r}"
        )

    reviewed_on = review.reviewed_on
    package_assignments = [
        {
            "sourceTypeText": row.sourceTypeText,
            "sourceTypeCode": row.sourceTypeCode,
            "occurrences": row.occurrences,
            "categoryToken": token_by_text.get(cast(str, row.sourceTypeText)),
            "reviewBasis": (
                "manual_semantic_review"
                if row.sourceTypeText in token_by_text
                else "insufficient_source_specificity"
            ),
            "reviewedBy": review.reviewed_by,
            "reviewedOn": reviewed_on,
        }
        for row in package_rows
    ]
    container_assignments = [
        {
            "sourceTypeDescription": row.sourceTypeDescription,
            "sourceTypeCode": row.sourceTypeCode,
            "occurrences": row.occurrences,
            "categoryToken": None,
            "reviewBasis": "insufficient_source_specificity",
            "reviewedBy": review.reviewed_by,
            "reviewedOn": reviewed_on,
        }
        for row in container_rows
    ]
    artifact = CategoryAssignments.model_validate(
        {
            "schemaVersion": 2,
            "reviewConfigSha256": sha256_bytes(review_payload),
            "sourceInventorySha256": source_inventory_sha256,
            "packageRegistrySha256": package_registry_sha256,
            "containerRegistrySha256": container_registry_sha256,
            "packageAssignments": tuple(package_assignments),
            "containerAssignments": tuple(container_assignments),
        },
        strict=True,
    )
    output_path = Path(review.output_path)
    atomic_publish_json(output_path, artifact.model_dump(mode="json"))
    return output_path
