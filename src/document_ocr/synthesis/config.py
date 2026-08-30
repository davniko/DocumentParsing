"""Strict configuration for lossless synthesis-foundation publication."""

from __future__ import annotations

from collections.abc import Hashable
from datetime import date
from pathlib import Path, PurePath
from typing import Annotated, Any, Literal

import yaml
from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    StringConstraints,
    field_validator,
    model_validator,
)
from yaml.constructor import ConstructorError
from yaml.nodes import MappingNode

from document_ocr.training.config import (
    DatasetFieldsConfig,
    DatasetFileConfig,
    TaskConstraintsConfig,
)

NonEmptyString = Annotated[str, StringConstraints(strip_whitespace=True, min_length=1)]


class _StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True, allow_inf_nan=False)


class _UniqueKeySafeLoader(yaml.SafeLoader):
    def construct_mapping(self, node: MappingNode, deep: bool = False) -> dict[Any, Any]:
        self.flatten_mapping(node)
        mapping: dict[Any, Any] = {}
        for key_node, value_node in node.value:
            key = self.construct_object(key_node, deep=deep)
            if not isinstance(key, Hashable):
                raise ConstructorError(
                    "while constructing a mapping",
                    node.start_mark,
                    "found an unhashable key",
                    key_node.start_mark,
                )
            if key in mapping:
                raise ConstructorError(
                    "while constructing a mapping",
                    node.start_mark,
                    f"found duplicate key {key!r}",
                    key_node.start_mark,
                )
            mapping[key] = self.construct_object(value_node, deep=deep)
        return mapping


def _safe_path(value: str) -> str:
    path = PurePath(value)
    if not value.strip() or "\x00" in value:
        raise ValueError("path must be a non-empty filesystem path")
    if not path.is_absolute() and any(part in {"", ".", ".."} for part in path.parts):
        raise ValueError("relative paths must not contain empty, '.' or '..' components")
    if path.is_absolute() and path == PurePath(path.anchor):
        raise ValueError("path must not be a filesystem root")
    return value


class SynthesisRunConfig(_StrictModel):
    run_id: Annotated[str, StringConstraints(pattern=r"^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$")]
    output_dir: NonEmptyString

    @field_validator("output_dir")
    @classmethod
    def safe_output_dir(cls, value: str) -> str:
        return _safe_path(value)


class SynthesisSourceConfig(_StrictModel):
    format: Literal["jsonl"]
    file: DatasetFileConfig
    fields: DatasetFieldsConfig


class SdvProfileConfig(_StrictModel):
    metadata_spec: Literal["V1"]
    validate_with_sdv: bool


class SynthesisFoundationConfig(_StrictModel):
    schema_version: Literal[1]
    task: NonEmptyString
    run: SynthesisRunConfig
    source: SynthesisSourceConfig
    task_constraints: TaskConstraintsConfig | None
    sdv: SdvProfileConfig

    @model_validator(mode="after")
    def cross_section_contract(self) -> SynthesisFoundationConfig:
        relation_task = self.task == "bill_of_lading_relation_explicit_v3"
        if relation_task != (self.task_constraints is not None):
            raise ValueError("relation-explicit synthesis requires task_constraints")
        if self.source.fields.input_sha256 is None:
            raise ValueError("synthesis source requires an input_sha256 field")
        return self


class PinnedDirectoryConfig(_StrictModel):
    path: NonEmptyString
    manifest_sha256: Annotated[str, StringConstraints(pattern=r"^[0-9a-f]{64}$")]

    @field_validator("path")
    @classmethod
    def safe_path(cls, value: str) -> str:
        return _safe_path(value)


class SynthesisSidecarsConfig(_StrictModel):
    lineage: DatasetFileConfig
    package_metadata: DatasetFileConfig
    category_metadata: DatasetFileConfig
    current_annotations: PinnedDirectoryConfig
    document_features: DatasetFileConfig
    template_groups: DatasetFileConfig


class AnchorPreparationConfig(_StrictModel):
    source: Literal["validated_annotation_evidence"]
    require_all_source_fact_evidence: bool
    patchable_locations: tuple[Literal["exact_unique", "excerpt_scoped_unique"], ...] = Field(
        min_length=1
    )

    @field_validator("patchable_locations", mode="before")
    @classmethod
    def freeze_locations(cls, value: Any) -> Any:
        return tuple(value) if isinstance(value, list) else value


class GeneratorContractConfig(_StrictModel):
    random_stream: Literal["hmac_sha256_counter_v1"]
    container_identifiers: Literal["preserve_observed_prefix_iso6346_v1"]
    seals: Literal["simple_scalar_surface_pattern_v1"]
    dates: Literal["joint_bounded_day_shift_v1"]
    quantities: Literal["coverage_aware_largest_remainder_v1"]
    masses: Literal["decimal_scale_preserve_unit_v1"]


SynthesisCohort = Literal[
    "all_documents",
    "multi_page",
    "multi_container",
    "multi_cargo",
    "multiple_package_levels",
    "container_allocations",
    "refrigerated",
    "dangerous_goods",
    "delivery_agent",
    "forwarding_agent",
    "hs_codes",
]


class SupportPreparationConfig(_StrictModel):
    cohorts: tuple[SynthesisCohort, ...] = Field(min_length=1)
    minimum_template_documents: Annotated[int, Field(gt=0)]

    @field_validator("cohorts", mode="before")
    @classmethod
    def freeze_cohorts(cls, value: Any) -> Any:
        return tuple(value) if isinstance(value, list) else value

    @model_validator(mode="after")
    def unique_cohorts(self) -> SupportPreparationConfig:
        if len(self.cohorts) != len(set(self.cohorts)):
            raise ValueError("support cohorts must be unique")
        return self


class SynthesisPreparationConfig(_StrictModel):
    schema_version: Literal[1]
    task: Literal["bill_of_lading_relation_explicit_v3"]
    run: SynthesisRunConfig
    source: SynthesisSourceConfig
    task_constraints: TaskConstraintsConfig
    sidecars: SynthesisSidecarsConfig
    sdv: SdvProfileConfig
    anchors: AnchorPreparationConfig
    generators: GeneratorContractConfig
    support: SupportPreparationConfig

    @model_validator(mode="after")
    def pinned_inputs(self) -> SynthesisPreparationConfig:
        if self.source.fields.input_sha256 is None:
            raise ValueError("synthesis preparation requires an input SHA-256 field")
        return self


class PinnedFileConfig(_StrictModel):
    path: NonEmptyString
    sha256: Annotated[str, StringConstraints(pattern=r"^[0-9a-f]{64}$")]

    @field_validator("path")
    @classmethod
    def safe_path(cls, value: str) -> str:
        return _safe_path(value)


MutationFamily = Literal[
    "container_identifier",
    "seal_identifier",
    "document_dates",
    "package_quantity",
    "cargo_mass",
]
DETERMINISTIC_MUTATION_FAMILIES = frozenset(
    {
        "container_identifier",
        "seal_identifier",
        "document_dates",
        "package_quantity",
        "cargo_mass",
    }
)


class GenerationInputsConfig(_StrictModel):
    preparation: PinnedDirectoryConfig
    anchors: DatasetFileConfig
    document_features: DatasetFileConfig
    template_groups: DatasetFileConfig
    partition_report: PinnedFileConfig


class SmokeSelectionConfig(_StrictModel):
    split: NonEmptyString
    requested_documents: Annotated[int, Field(gt=0)]
    seed: int
    minimum_template_documents: Annotated[int, Field(gt=0)]
    require_template_wholly_in_split: bool
    maximum_per_template: Annotated[int, Field(gt=0)]
    maximum_per_carrier: Annotated[int, Field(gt=0)]
    minimum_carriers: Annotated[int, Field(gt=0)]
    family_exact: dict[MutationFamily, Annotated[int, Field(ge=0)]]
    strata_exact: dict[NonEmptyString, dict[NonEmptyString, Annotated[int, Field(ge=0)]]]
    context_minimums: dict[NonEmptyString, Annotated[int, Field(ge=0)]]

    @model_validator(mode="after")
    def exact_counts(self) -> SmokeSelectionConfig:
        if set(self.family_exact) != DETERMINISTIC_MUTATION_FAMILIES:
            raise ValueError("family_exact must name every deterministic mutation family")
        if sum(self.family_exact.values()) != self.requested_documents:
            raise ValueError("family_exact counts must sum to requested_documents")
        for name, values in self.strata_exact.items():
            if not values or sum(values.values()) != self.requested_documents:
                raise ValueError(f"strata_exact.{name} must partition requested_documents")
        if self.minimum_carriers > self.requested_documents:
            raise ValueError("minimum_carriers exceeds requested_documents")
        return self


class DeterministicGenerationConfig(_StrictModel):
    random_stream: Literal["hmac_sha256_counter_v1"]
    seed: int
    variant_index: Annotated[int, Field(ge=0)]
    date_minimum: date
    date_maximum: date
    preserve_source_missingness: Literal[True]
    preserve_source_cardinality: Literal[True]
    publish_training_records: Literal[False]

    @model_validator(mode="after")
    def ordered_dates(self) -> DeterministicGenerationConfig:
        if self.date_minimum >= self.date_maximum:
            raise ValueError("generation date window must contain at least two dates")
        return self


class SynthesisDeterministicSmokeConfig(_StrictModel):
    schema_version: Literal[1]
    task: Literal["bill_of_lading_relation_explicit_v3"]
    run: SynthesisRunConfig
    source: SynthesisSourceConfig
    task_constraints: TaskConstraintsConfig
    inputs: GenerationInputsConfig
    selection: SmokeSelectionConfig
    generation: DeterministicGenerationConfig

    @model_validator(mode="after")
    def pinned_source(self) -> SynthesisDeterministicSmokeConfig:
        if self.source.fields.input_sha256 is None:
            raise ValueError("deterministic synthesis requires source input SHA-256 values")
        return self


def load_synthesis_foundation_config(path: Path) -> SynthesisFoundationConfig:
    try:
        value = yaml.load(path.read_bytes(), Loader=_UniqueKeySafeLoader)
    except UnicodeDecodeError as error:
        raise ValueError("synthesis configuration is not valid UTF-8") from error
    if not isinstance(value, dict):
        raise ValueError("synthesis configuration root must be a mapping")
    return SynthesisFoundationConfig.model_validate(value, strict=True)


def load_synthesis_preparation_config(path: Path) -> SynthesisPreparationConfig:
    try:
        value = yaml.load(path.read_bytes(), Loader=_UniqueKeySafeLoader)
    except UnicodeDecodeError as error:
        raise ValueError("synthesis configuration is not valid UTF-8") from error
    if not isinstance(value, dict):
        raise ValueError("synthesis configuration root must be a mapping")
    return SynthesisPreparationConfig.model_validate(value, strict=True)


def load_synthesis_deterministic_smoke_config(
    path: Path,
) -> SynthesisDeterministicSmokeConfig:
    try:
        value = yaml.load(path.read_bytes(), Loader=_UniqueKeySafeLoader)
    except UnicodeDecodeError as error:
        raise ValueError("synthesis configuration is not valid UTF-8") from error
    if not isinstance(value, dict):
        raise ValueError("synthesis configuration root must be a mapping")
    return SynthesisDeterministicSmokeConfig.model_validate(value, strict=True)
