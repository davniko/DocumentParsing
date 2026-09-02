"""Strict configuration for lossless synthesis-foundation publication."""

from __future__ import annotations

from collections.abc import Hashable
from datetime import date
from pathlib import Path, PurePath
from typing import Annotated, Any, Literal
from urllib.parse import urlsplit

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

from document_ocr.label_schemas.bill_of_lading_v3 import HazardCategory
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


class CommittedDirectoryConfig(PinnedDirectoryConfig):
    """Pinned immutable run directory with its transaction receipt."""

    commit_sha256: Annotated[str, StringConstraints(pattern=r"^[0-9a-f]{64}$")]
    transaction_sha256: Annotated[str, StringConstraints(pattern=r"^[0-9a-f]{64}$")]


class CommittedArtifactDirectoryConfig(_StrictModel):
    """Pinned staged run whose commit receipt is its complete artifact inventory."""

    path: NonEmptyString
    commit_sha256: Annotated[str, StringConstraints(pattern=r"^[0-9a-f]{64}$")]
    transaction_sha256: Annotated[str, StringConstraints(pattern=r"^[0-9a-f]{64}$")]

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


StatisticalCandidate = Literal[
    "empirical",
    "gaussian_copula_default",
    "gaussian_copula_gaussian_kde",
    "gaussian_copula_domain_mixed",
    "gaussian_copula_physical_factors",
    "ctgan",
    "tvae",
]


class StructuredGenerationInputsConfig(_StrictModel):
    preparation: PinnedDirectoryConfig
    document_features: DatasetFileConfig
    template_groups: DatasetFileConfig
    partition_report: PinnedFileConfig
    iso3166_snapshot: PinnedFileConfig
    category_metadata: DatasetFileConfig
    package_hierarchy_metadata: DatasetFileConfig
    package_registry: PinnedFileConfig
    package_registry_entries: Annotated[int, Field(gt=0)]


class RuntimeFingerprintConfig(_StrictModel):
    image_reference: NonEmptyString
    image_digest: Annotated[str, StringConstraints(pattern=r"^sha256:[0-9a-f]{64}$")]
    expected_versions: dict[NonEmptyString, NonEmptyString]

    @model_validator(mode="after")
    def required_versions_are_pinned(self) -> RuntimeFingerprintConfig:
        required = {"numpy", "pandas", "scipy", "sdmetrics", "sdv"}
        if set(self.expected_versions) != required:
            raise ValueError(
                "expected_versions must pin exactly numpy, pandas, scipy, sdmetrics, and sdv"
            )
        return self


class StructuredSelectionConfig(_StrictModel):
    split: NonEmptyString
    requested_documents: Annotated[int, Field(gt=0)]
    seed: int
    minimum_template_documents: Annotated[int, Field(gt=0)]
    require_template_wholly_in_split: Literal[True]
    require_route_synthesis_support: Literal[True]
    maximum_per_template: Annotated[int, Field(gt=0)]
    maximum_per_carrier: Annotated[int, Field(gt=0)]
    minimum_carriers: Annotated[int, Field(gt=0)]
    strata_exact: dict[NonEmptyString, dict[NonEmptyString, Annotated[int, Field(ge=0)]]]
    context_minimums: dict[NonEmptyString, Annotated[int, Field(ge=0)]]

    @model_validator(mode="after")
    def requested_counts_are_coherent(self) -> StructuredSelectionConfig:
        for name, values in self.strata_exact.items():
            if not values or sum(values.values()) != self.requested_documents:
                raise ValueError(f"strata_exact.{name} must partition requested_documents")
        if self.minimum_carriers > self.requested_documents:
            raise ValueError("minimum_carriers exceeds requested_documents")
        if self.maximum_per_template > self.requested_documents:
            raise ValueError("maximum_per_template exceeds requested_documents")
        if self.maximum_per_carrier > self.requested_documents:
            raise ValueError("maximum_per_carrier exceeds requested_documents")
        return self


class ProfileRouteThresholdConfig(_StrictModel):
    minimum_rows: Annotated[int, Field(ge=2)]
    minimum_templates: Annotated[int, Field(ge=2)]


class ProfileRoutingConfig(_StrictModel):
    exact_identity_role: ProfileRouteThresholdConfig
    semantic_family_role: ProfileRouteThresholdConfig
    role_profile: ProfileRouteThresholdConfig


class ContextualPlausibilityConfig(_StrictModel):
    """Minimum isolated-train evidence for package-aware proposal checks."""

    exact_identity_role: ProfileRouteThresholdConfig
    semantic_family_role: ProfileRouteThresholdConfig


class ProfileSourceQualityConfig(_StrictModel):
    maximum_gross_to_net_ratio: Annotated[float, Field(gt=1)]
    robust_minimum_rows: Annotated[int, Field(ge=4)]
    robust_z_threshold: Annotated[float, Field(gt=1)]
    maximum_median_factor: Annotated[float, Field(gt=1)]


class StatisticalBenchmarkConfig(_StrictModel):
    view: Literal["cargo_group_numeric"]
    candidates: tuple[StatisticalCandidate, ...] = Field(min_length=2)
    selectable_candidates: tuple[StatisticalCandidate, ...] = Field(min_length=1)
    grouped_folds: Annotated[int, Field(ge=2)]
    seeds: tuple[int, ...] = Field(min_length=2)
    fold_seed: int
    neural_epochs: Annotated[int, Field(gt=0)]
    minimum_rows_for_neural: Annotated[int, Field(gt=0)]
    minimum_templates_for_neural: Annotated[int, Field(gt=0)]
    simplest_within_quality_margin: Annotated[float, Field(ge=0, le=1)]
    stability_penalty: Annotated[float, Field(ge=0)]
    maximum_mean_quality_deficit_vs_empirical: Annotated[float, Field(ge=0, le=1)]
    maximum_property_quality_deficit_vs_empirical: Annotated[float, Field(ge=0, le=1)]
    maximum_detail_quality_deficit_vs_empirical: Annotated[float, Field(ge=0, le=1)] = 0.10
    paired_confidence_level: Annotated[float, Field(gt=0.5, lt=1)]
    maximum_paired_quality_lcb_deficit: Annotated[float, Field(ge=0, le=1)]
    maximum_raw_proposals_per_accept: Annotated[int, Field(gt=0)]
    proposal_batch_rows: Annotated[int, Field(gt=0)]
    proposal_assignment_oversample_factor: Annotated[int, Field(ge=2)]
    minimum_final_acceptance_yield: Annotated[float, Field(gt=0, le=1)]
    minimum_final_novelty_fraction: Annotated[float, Field(ge=0, le=1)]
    routing: ProfileRoutingConfig
    contextual_plausibility: ContextualPlausibilityConfig
    source_quality: ProfileSourceQualityConfig

    @field_validator("candidates", "selectable_candidates", "seeds", mode="before")
    @classmethod
    def freeze_sequences(cls, value: Any) -> Any:
        return tuple(value) if isinstance(value, list) else value

    @model_validator(mode="after")
    def benchmark_is_identifiable(self) -> StatisticalBenchmarkConfig:
        if len(self.candidates) != len(set(self.candidates)):
            raise ValueError("statistical candidates must be unique")
        if len(self.selectable_candidates) != len(set(self.selectable_candidates)):
            raise ValueError("selectable statistical candidates must be unique")
        if not set(self.selectable_candidates) <= set(self.candidates):
            raise ValueError("selectable candidates must be benchmark candidates")
        if len(self.seeds) != len(set(self.seeds)):
            raise ValueError("statistical benchmark seeds must be unique")
        if "empirical" not in self.candidates:
            raise ValueError("statistical candidates must include the empirical baseline")
        if "empirical" in self.selectable_candidates:
            raise ValueError("empirical is a benchmark baseline, not a production proposal model")
        routing_thresholds = (
            self.routing.exact_identity_role,
            self.routing.semantic_family_role,
            self.routing.role_profile,
        )
        if any(
            threshold.minimum_templates < self.grouped_folds for threshold in routing_thresholds
        ):
            raise ValueError(
                "every routing tier must require at least grouped_folds distinct templates"
            )
        return self


class TransportCapacityConfig(_StrictModel):
    """Published upper-bound policy for source and generated container loads."""

    policy: Literal["source_type_aware_maersk_upper_bounds_v1"]
    published_reference_margin_fraction: Annotated[float, Field(ge=0, le=0.10)]
    twenty_standard_payload_kg: Annotated[float, Field(gt=0)]
    twenty_standard_volume_m3: Annotated[float, Field(gt=0)]
    forty_standard_payload_kg: Annotated[float, Field(gt=0)]
    forty_standard_volume_m3: Annotated[float, Field(gt=0)]
    forty_high_cube_payload_kg: Annotated[float, Field(gt=0)]
    forty_high_cube_volume_m3: Annotated[float, Field(gt=0)]
    forty_five_high_cube_payload_kg: Annotated[float, Field(gt=0)]
    forty_five_high_cube_volume_m3: Annotated[float, Field(gt=0)]
    out_of_gauge_payload_kg: Annotated[float, Field(gt=0)]
    unclassified_payload_kg: Annotated[float, Field(gt=0)]
    unclassified_volume_m3: Annotated[float, Field(gt=0)]


class StructuredScenarioGenerationConfig(_StrictModel):
    random_stream: Literal["hmac_sha256_counter_v1"]
    seed: int
    date_minimum: date
    date_maximum: date
    preserve_source_missingness: Literal[True]
    preserve_source_cardinality: Literal[True]
    preserve_relation_topology: Literal[True]
    regenerate_all_supported_identifiers: Literal[True]
    date_method: Literal["train_empirical_anchor_plus_bounded_jitter_v1"]
    maximum_date_jitter_days: Annotated[int, Field(gt=0)]
    quantity_method: Literal["profile_routed_group_driver_then_reconcile_v1"]
    quantity_compatibility_profile: Literal["package_category_role_hierarchical_pool_v1"]
    cargo_measure_method: Literal["profile_routed_per_driver_measure_then_exact_unit_inverse_v1"]
    package_category_policy: Literal["preserve_validated_source_category_v1"]
    container_type_policy: Literal["preserve_source_nonidentifying_category_v1"]
    transport_capacity: TransportCapacityConfig
    publish_training_records: Literal[False]

    @model_validator(mode="after")
    def date_window_is_ordered(self) -> StructuredScenarioGenerationConfig:
        if self.date_minimum >= self.date_maximum:
            raise ValueError("structured generation date window must contain at least two dates")
        return self


class SynthesisStructuredBaselineConfig(_StrictModel):
    schema_version: Literal[1]
    task: Literal["bill_of_lading_relation_explicit_v3"]
    run: SynthesisRunConfig
    source: SynthesisSourceConfig
    task_constraints: TaskConstraintsConfig
    inputs: StructuredGenerationInputsConfig
    runtime: RuntimeFingerprintConfig
    selection: StructuredSelectionConfig
    modeling: StatisticalBenchmarkConfig
    generation: StructuredScenarioGenerationConfig

    @model_validator(mode="after")
    def structured_baseline_is_non_publishable(self) -> SynthesisStructuredBaselineConfig:
        if self.source.fields.input_sha256 is None:
            raise ValueError("structured synthesis requires source input SHA-256 values")
        if self.selection.requested_documents < 2:
            raise ValueError("structured baseline requires at least two selected documents")
        return self


class RouteScenarioInputsConfig(_StrictModel):
    """Pinned inputs for route-first scenario fitting and sampling."""

    upstream_selection: DatasetFileConfig
    template_groups: DatasetFileConfig
    partition_report: PinnedFileConfig
    iso3166_snapshot: PinnedFileConfig
    route_locations: DatasetFileConfig
    route_registry_manifest: PinnedFileConfig
    world_ports: DatasetFileConfig
    world_port_registry_manifest: PinnedFileConfig
    locality_registry: PinnedDirectoryConfig
    trade_flows: DatasetFileConfig
    trade_flow_registry_manifest: PinnedFileConfig


class RouteScenarioSelectionConfig(_StrictModel):
    split: NonEmptyString
    requested_documents: Annotated[int, Field(gt=0)]
    require_template_wholly_in_split: Literal[True]
    maximum_per_template: Annotated[int, Field(gt=0)]


class RouteScenarioObservedOriginComponentConfig(_StrictModel):
    """Observed-exporter component of the commercial-origin mixture."""

    mixture_permyriad: Annotated[int, Field(gt=0, lt=10_000)]
    weighting: Literal["train_isolated_shipper_country_document_count_v1"]


class RouteScenarioMaritimeOriginComponentConfig(_StrictModel):
    """Pinned maritime-registry component of the commercial-origin mixture."""

    mixture_permyriad: Annotated[int, Field(gt=0, lt=10_000)]
    weighting: Literal["uniform_route_feasible_iso_country_v1"]


class RouteScenarioOriginPriorConfig(_StrictModel):
    """Explicit component-first commercial-origin mixture."""

    method: Literal["observed_exporter_plus_maritime_registry_mixture_v1"]
    observed_exporter: RouteScenarioObservedOriginComponentConfig
    maritime_registry: RouteScenarioMaritimeOriginComponentConfig

    @model_validator(mode="after")
    def mixture_weights_sum_to_one(self) -> RouteScenarioOriginPriorConfig:
        total = self.observed_exporter.mixture_permyriad + self.maritime_registry.mixture_permyriad
        if total != 10_000:
            raise ValueError("commercial-origin mixture weights must sum to 10000")
        return self


class RouteScenarioGenerationConfig(_StrictModel):
    random_stream: Literal["hmac_sha256_counter_v1"]
    seed: int
    commercial_origin_prior: RouteScenarioOriginPriorConfig
    commercial_destination_prior: Literal[
        "wits_latest_bilateral_numeric_iso_identity_conditioned_on_origin_v2"
    ]
    physical_endpoint_relation_method: Literal[
        "train_empirical_loading_by_origin_discharge_by_destination_else_commercial_destination_v1"
    ]
    party_locality_relation_method: Literal["train_empirical_role_relation_and_locality_mode_v1"]
    freight_method: Literal["train_empirical_arrangement_and_payment_side_v1"]
    port_method: Literal["observed_empirical_plus_nga_wpi_whitelist_mixture_v1"]
    locality_method: Literal["geonames_cities15000_population_weighted_v1"]
    registry_exploration_permyriad: Annotated[int, Field(ge=0, le=10_000)]
    preserve_source_leaf_presence: Literal[True]
    preserve_source_party_cardinality: Literal[True]
    direct_routes_only: Literal[True]
    publish_training_records: Literal[False]


class SynthesisRouteScenarioPilotConfig(_StrictModel):
    """Non-publishable route/party-locality scenario pilot contract."""

    schema_version: Literal[3]
    task: Literal["bill_of_lading_relation_explicit_v3"]
    run: SynthesisRunConfig
    source: SynthesisSourceConfig
    task_constraints: TaskConstraintsConfig
    inputs: RouteScenarioInputsConfig
    selection: RouteScenarioSelectionConfig
    generation: RouteScenarioGenerationConfig

    @model_validator(mode="after")
    def route_pilot_is_pinned_and_non_publishable(self) -> SynthesisRouteScenarioPilotConfig:
        if self.source.fields.input_sha256 is None:
            raise ValueError("route scenario pilot requires source input SHA-256 values")
        if self.inputs.upstream_selection.records != self.selection.requested_documents:
            raise ValueError("upstream selection count differs from route scenario selection")
        return self


class ControlledPilotInputsConfig(_StrictModel):
    """Pinned semantic and statistical dependencies for the controlled pilot."""

    route_scenario_run: CommittedDirectoryConfig
    route_scenarios: DatasetFileConfig
    route_targets: DatasetFileConfig
    preparation: PinnedDirectoryConfig
    template_groups: DatasetFileConfig
    partition_report: PinnedFileConfig
    category_metadata: DatasetFileConfig
    package_hierarchy_metadata: DatasetFileConfig
    package_registry: PinnedFileConfig
    package_registry_entries: Annotated[int, Field(gt=0)]
    iso3166_snapshot: PinnedFileConfig
    equipment_registry_manifest: PinnedFileConfig
    hs_registry_manifest: PinnedFileConfig
    hs_metadata: PinnedFileConfig
    hs_commodities_report: PinnedFileConfig
    party_identity_benchmark_summary: PinnedFileConfig
    vessel_name_registry: DatasetFileConfig | None = None
    vessel_name_registry_receipt: PinnedFileConfig | None = None


class ControlledHsGenerationConfig(_StrictModel):
    schema_version: Literal[1]
    observed_chapter_mixture_permyriad: Annotated[int, Field(ge=0, le=10_000)]
    registry_wide_mixture_permyriad: Annotated[int, Field(ge=0, le=10_000)]
    gb_tariff_extension_permyriad: Annotated[int, Field(ge=0, le=10_000)]
    observed_chapter_weighting: Literal["fit_document_count_v1"]
    observed_chapter_hs6_weighting: Literal["uniform_registry_hs6_within_selected_chapter_v1"]
    registry_hs6_weighting: Literal["uniform_registry_hs6_v1"]
    gb_tariff_leaf_weighting: Literal["uniform_registry_leaves_v1"]
    output_length_method: Literal["preserve_source_length_exact_registry_else_random_suffix_v1"]
    minimum_output_digits: Literal[6]
    maximum_output_digits: Annotated[int, Field(ge=6, le=18)]
    maximum_extension_attempts: Annotated[int, Field(gt=0)]

    @model_validator(mode="after")
    def chapter_mixture_is_complete(self) -> ControlledHsGenerationConfig:
        if self.observed_chapter_mixture_permyriad + self.registry_wide_mixture_permyriad != 10_000:
            raise ValueError("controlled HS mixture weights must sum exactly to 10000")
        if self.maximum_output_digits < self.minimum_output_digits:
            raise ValueError("controlled HS output digit bounds are reversed")
        return self


class ControlledTransportGenerationConfig(_StrictModel):
    vessel_name_method: Literal[
        "deferred_by_explicit_scope_v1",
        "public_cargo_vessel_registry_uniform_v1",
    ]
    voyage_number_method: Literal[
        "observed_character_class_shape_v1",
        "preserve_for_upstream_structured_identifier_v1",
    ]
    minimum_normalized_edit_distance: Annotated[float, Field(ge=0, lt=1)]
    maximum_realization_attempts: Annotated[int, Field(gt=0)]
    imo_policy: Literal["absent_without_authoritative_assigned_number_registry_v1"]


class ControlledPilotGenerationConfig(_StrictModel):
    random_stream: Literal["hmac_sha256_counter_v1"]
    seed: int
    package_method: Literal["fit_role_conditioned_registry_category_v2"]
    package_registry_exploration_permyriad: Annotated[int, Field(ge=0, le=10_000)]
    package_registry_exploration_scope: Literal[
        "same_authoritative_display_family_within_task_vocabulary_v1"
    ]
    equipment_method: Literal["fit_exact_bic_identity_plus_registry_exploration_v1"]
    equipment_registry_exploration_permyriad: Annotated[int, Field(ge=0, le=10_000)]
    hs: ControlledHsGenerationConfig
    cargo_origin_method: Literal["fit_country_relation_reproject_name_only_origin_v2"]
    cargo_origin_name_only_policy: Literal["replace_without_resolving_or_copying_source_name_v2"]
    dangerous_goods_method: Literal["deferred_goods_first_coherent_semantic_realization_v1"]
    handling_instructions_method: Literal["pending_linguistic_realization_v1"]
    transport: ControlledTransportGenerationConfig
    preserve_source_field_presence: Literal[True]
    preserve_source_cardinality: Literal[True]
    preserve_relation_topology: Literal[True]
    publish_training_records: Literal[False]


class SynthesisControlledPilotConfig(_StrictModel):
    """Non-publishable controlled semantic generation over one route pilot."""

    schema_version: Literal[1]
    task: Literal["bill_of_lading_relation_explicit_v3"]
    run: SynthesisRunConfig
    source: SynthesisSourceConfig
    task_constraints: TaskConstraintsConfig
    inputs: ControlledPilotInputsConfig
    selection: RouteScenarioSelectionConfig
    generation: ControlledPilotGenerationConfig

    @model_validator(mode="after")
    def vessel_registry_matches_generation_method(self) -> SynthesisControlledPilotConfig:
        uses_registry = (
            self.generation.transport.vessel_name_method
            == "public_cargo_vessel_registry_uniform_v1"
        )
        has_registry = self.inputs.vessel_name_registry is not None
        has_receipt = self.inputs.vessel_name_registry_receipt is not None
        if uses_registry != has_registry or uses_registry != has_receipt:
            raise ValueError(
                "public vessel-name sampling requires both pinned registry and receipt; "
                "deferred sampling requires neither"
            )
        return self

    @model_validator(mode="after")
    def controlled_pilot_is_pinned_and_non_publishable(
        self,
    ) -> SynthesisControlledPilotConfig:
        if self.source.fields.input_sha256 is None:
            raise ValueError("controlled synthesis requires source input SHA-256 values")
        if self.inputs.route_scenarios.records != self.selection.requested_documents:
            raise ValueError("route scenario count differs from controlled selection")
        if self.inputs.route_targets.records != self.selection.requested_documents:
            raise ValueError("route target count differs from controlled selection")
        return self


class SemanticPlanInputsConfig(_StrictModel):
    """Hash-pinned stages that must resolve to one identical source selection."""

    structured_run: CommittedDirectoryConfig
    structured_selection: DatasetFileConfig
    structured_targets: DatasetFileConfig
    structured_plans: DatasetFileConfig
    controlled_run: CommittedDirectoryConfig
    controlled_scenarios: DatasetFileConfig
    controlled_targets: DatasetFileConfig


class SemanticPlanGenerationConfig(_StrictModel):
    scenario_namespace: NonEmptyString
    seed: Annotated[int, Field(ge=0, lt=2**64)]
    variant_index_method: Literal["selected_position_zero_based_v1"]
    structured_stage_contract: Literal["structured_numeric_identifier_temporal_v1"]
    controlled_stage_contract: Literal["route_and_controlled_semantics_v1"]
    require_disjoint_change_ownership: Literal[True]
    dangerous_goods_policy: Literal["defer_goods_first_coherent_realization_v1"]
    publish_training_records: Literal[False]


class SynthesisSemanticPlanConfig(_StrictModel):
    """Compose the complete non-linguistic same-template semantic target."""

    schema_version: Literal[1]
    task: Literal["bill_of_lading_relation_explicit_v3"]
    run: SynthesisRunConfig
    source: SynthesisSourceConfig
    inputs: SemanticPlanInputsConfig
    generation: SemanticPlanGenerationConfig

    @model_validator(mode="after")
    def stage_counts_and_source_contract_match(self) -> SynthesisSemanticPlanConfig:
        if self.source.fields.input_sha256 is None:
            raise ValueError("semantic composition requires source input SHA-256 values")
        counts = {
            self.inputs.structured_selection.records,
            self.inputs.structured_targets.records,
            self.inputs.structured_plans.records,
            self.inputs.controlled_scenarios.records,
            self.inputs.controlled_targets.records,
        }
        if len(counts) != 1:
            raise ValueError("semantic plan input record counts differ")
        return self


class DangerousGoodsRegistryInputsConfig(_StrictModel):
    source_root: NonEmptyString
    source_manifest: PinnedFileConfig

    @field_validator("source_root")
    @classmethod
    def safe_source_root(cls, value: str) -> str:
        return _safe_path(value)


class SynthesisDangerousGoodsRegistryConfig(_StrictModel):
    """Pinned source-to-compiled-registry contract."""

    schema_version: Literal[1]
    run: SynthesisRunConfig
    inputs: DangerousGoodsRegistryInputsConfig


class DangerousGoodsRegistryArtifactsConfig(_StrictModel):
    run: CommittedArtifactDirectoryConfig
    receipt: PinnedFileConfig
    hmt_records: DatasetFileConfig
    ecics_links: DatasetFileConfig


class SynthesisDangerousGoodsAnalysisConfig(_StrictModel):
    """Plot-heavy registry and real-corpus coverage analysis."""

    schema_version: Literal[1]
    run: SynthesisRunConfig
    registry: DangerousGoodsRegistryArtifactsConfig
    corpus: DatasetFileConfig


class DangerousGoodsPlanInputsConfig(_StrictModel):
    semantic_plan_run: CommittedDirectoryConfig
    semantic_plans: DatasetFileConfig
    source_task_constraints: PinnedFileConfig
    registry: DangerousGoodsRegistryArtifactsConfig


class DangerousGoodsPlanGenerationConfig(_StrictModel):
    random_stream: Literal["hmac_sha256_counter_v1"]
    seed: Annotated[int, Field(ge=0, lt=2**64)]
    branch_method: Literal["source_hs_presence_exact_identity_chemical_else_general_v2"]
    category_weights_permyriad: dict[HazardCategory, Annotated[int, Field(gt=0)]]
    unsupported_category_policy: Literal["renormalize_over_branch_support_v1"]
    maximum_subsidiary_hazards: Literal[1]
    maximum_collision_attempts: Annotated[int, Field(gt=0)]
    sampling_validation_draws_per_branch: Annotated[int, Field(ge=1000)]
    maximum_category_deviation_permyriad: Annotated[int, Field(gt=0, le=1000)]
    flashpoint_policy: Literal["omit_without_formulation_property_source_v1"]
    preserve_dangerous_goods_cardinality: Literal[True]
    publish_training_records: Literal[False]

    @model_validator(mode="after")
    def category_weights_are_complete(self) -> DangerousGoodsPlanGenerationConfig:
        if sum(self.category_weights_permyriad.values()) != 10_000:
            raise ValueError("DG category weights must sum exactly to 10000")
        return self


class SynthesisDangerousGoodsPlanConfig(_StrictModel):
    """Final non-linguistic DG stage over one committed semantic plan."""

    schema_version: Literal[1]
    source_task: Literal["bill_of_lading_relation_explicit_v3"]
    target_task: Literal["bill_of_lading_relation_explicit_v4"]
    run: SynthesisRunConfig
    inputs: DangerousGoodsPlanInputsConfig
    generation: DangerousGoodsPlanGenerationConfig

    @model_validator(mode="after")
    def counts_match(self) -> SynthesisDangerousGoodsPlanConfig:
        if self.inputs.semantic_plans.records <= 0:
            raise ValueError("DG plan requires at least one semantic plan")
        return self


class SemanticCompletionInputsConfig(_StrictModel):
    dangerous_goods_run: CommittedArtifactDirectoryConfig
    dangerous_goods_plans: DatasetFileConfig
    equipment_registry_manifest: PinnedFileConfig
    mpci_container_registry: PinnedFileConfig
    hs_registry_manifest: PinnedFileConfig
    hs_metadata: PinnedFileConfig
    hs_commodities_report: PinnedFileConfig
    iso3166_snapshot: PinnedFileConfig
    world_ports: DatasetFileConfig
    world_port_registry_receipt: PinnedFileConfig


class SemanticCompletionThermalConfig(_StrictModel):
    document_prevalence_permyriad: Annotated[int, Field(ge=0, le=10_000)]
    prevalence_method: Literal["exact_hmac_ranked_quota_over_eligible_documents_v1"]
    profile_weights_permyriad: dict[
        Literal["FROZEN", "CHILLED"],
        Annotated[int, Field(ge=0, le=10_000)],
    ]
    ambient_hs_chapters: tuple[Annotated[str, StringConstraints(pattern=r"^[0-9]{2}$")], ...]
    frozen_minimum_celsius: float
    frozen_maximum_celsius: float
    chilled_minimum_celsius: float
    chilled_maximum_celsius: float
    step_celsius: Annotated[float, Field(gt=0)]

    @field_validator("ambient_hs_chapters", mode="before")
    @classmethod
    def freeze_ambient_hs_chapters(cls, value: Any) -> Any:
        return tuple(value) if isinstance(value, list) else value

    @model_validator(mode="after")
    def thermal_policy_is_complete(self) -> SemanticCompletionThermalConfig:
        if (
            set(self.profile_weights_permyriad) != {"FROZEN", "CHILLED"}
            or sum(self.profile_weights_permyriad.values()) != 10_000
        ):
            raise ValueError("thermal profile weights must name both profiles and sum to 10000")
        if not self.ambient_hs_chapters or self.ambient_hs_chapters != tuple(
            sorted(set(self.ambient_hs_chapters))
        ):
            raise ValueError("ambient HS chapters must be non-empty, unique, and sorted")
        for minimum, maximum, label in (
            (self.frozen_minimum_celsius, self.frozen_maximum_celsius, "frozen"),
            (self.chilled_minimum_celsius, self.chilled_maximum_celsius, "chilled"),
        ):
            if minimum > maximum:
                raise ValueError(f"{label} temperature bounds are reversed")
            steps = round((maximum - minimum) / self.step_celsius)
            if abs(minimum + steps * self.step_celsius - maximum) > 1e-9:
                raise ValueError(f"{label} temperature range is not divisible by its step")
        return self


class SemanticCompletionEquipmentConfig(_StrictModel):
    distribution_method: Literal[
        "source_type_marginal_then_size_conditional_with_configurable_overrides_v1"
    ]
    configured_active_joint_weights: dict[NonEmptyString, Annotated[int, Field(ge=0)]] = Field(
        default_factory=dict
    )
    configured_inactive_joint_weights: dict[NonEmptyString, Annotated[int, Field(ge=0)]] = Field(
        default_factory=dict
    )
    generated_label_method: Literal["separate_readable_size_and_type_categories_v1"]
    application_projection: Literal["bic_size_plus_mpci_type_group_v1"]


class SemanticCompletionTransportConfig(_StrictModel):
    imo_presence_permyriad: Annotated[int, Field(ge=0, le=10_000)]
    flag_presence_permyriad: Annotated[int, Field(ge=0, le=10_000)]
    presence_method: Literal["configured_exact_hmac_ranked_quota_v1"]
    imo_method: Literal["random_six_digits_plus_imo_check_digit_v1"]
    imo_leading_digit_weights: dict[
        Annotated[str, StringConstraints(pattern=r"^[1-9]$")],
        Annotated[int, Field(gt=0)],
    ]
    maximum_collision_attempts: Annotated[int, Field(gt=0)]
    flag_method: Literal["uniform_pinned_world_port_country_v1"]
    source_correlation_policy: Literal["independent_no_supported_source_correlation_v1"]


class SemanticCompletionFlashpointConfig(_StrictModel):
    method: Literal["hazard_and_physical_form_conditioned_property_v1"]
    presence_permyriad: dict[
        Literal[
            "CLASS3_FLAMMABLE_LIQUID",
            "NON_CLASS3_EXPLICIT_LIQUID",
            "DESENSITIZED_FLAMMABLE_SOLID",
        ],
        Annotated[int, Field(ge=0, le=10_000)],
    ]
    class3_minimum_celsius: float
    class3_maximum_celsius: Annotated[float, Field(le=60)]
    non_class3_liquid_minimum_celsius: Annotated[float, Field(gt=60)]
    non_class3_liquid_maximum_celsius: float
    desensitized_solid_minimum_celsius: float
    desensitized_solid_maximum_celsius: Annotated[float, Field(le=60)]
    step_celsius: Annotated[float, Field(gt=0)]

    @model_validator(mode="after")
    def flashpoint_grid_is_exact(self) -> SemanticCompletionFlashpointConfig:
        expected = {
            "CLASS3_FLAMMABLE_LIQUID",
            "NON_CLASS3_EXPLICIT_LIQUID",
            "DESENSITIZED_FLAMMABLE_SOLID",
        }
        if set(self.presence_permyriad) != expected:
            raise ValueError("flashpoint presence weights must name every eligibility class")
        for minimum, maximum, label in (
            (self.class3_minimum_celsius, self.class3_maximum_celsius, "class 3"),
            (
                self.non_class3_liquid_minimum_celsius,
                self.non_class3_liquid_maximum_celsius,
                "non-class-3 liquid",
            ),
            (
                self.desensitized_solid_minimum_celsius,
                self.desensitized_solid_maximum_celsius,
                "desensitized solid",
            ),
        ):
            if minimum > maximum:
                raise ValueError(f"{label} flashpoint bounds are reversed")
            steps = round((maximum - minimum) / self.step_celsius)
            if abs(minimum + steps * self.step_celsius - maximum) > 1e-9:
                raise ValueError(f"{label} flashpoint range is not divisible by its step")
        return self


class SemanticCompletionGenerationConfig(_StrictModel):
    random_stream: Literal["hmac_sha256_counter_v1"]
    seed: Annotated[int, Field(ge=0, lt=2**64)]
    equipment: SemanticCompletionEquipmentConfig
    thermal: SemanticCompletionThermalConfig
    transport: SemanticCompletionTransportConfig
    flashpoint: SemanticCompletionFlashpointConfig
    preserve_container_cardinality: Literal[True]
    preserve_cargo_cardinality: Literal[True]
    preserve_relation_topology: Literal[True]
    publish_training_records: Literal[False]


class SynthesisSemanticCompletionConfig(_StrictModel):
    """Final deterministic/registry semantic stage before linguistic rendering."""

    schema_version: Literal[1]
    source_task: Literal["bill_of_lading_relation_explicit_v4"]
    target_task: Literal["bill_of_lading_relation_explicit_v5"]
    run: SynthesisRunConfig
    source: SynthesisSourceConfig
    inputs: SemanticCompletionInputsConfig
    generation: SemanticCompletionGenerationConfig

    @model_validator(mode="after")
    def completion_inputs_match(self) -> SynthesisSemanticCompletionConfig:
        if self.source.fields.input_sha256 is None:
            raise ValueError("semantic completion requires source input SHA-256 values")
        return self


class PartyStructureBenchmarkInputsConfig(_StrictModel):
    preparation_root: NonEmptyString
    preparation_manifest: PinnedFileConfig
    template_groups: DatasetFileConfig
    partition_report: PinnedFileConfig
    iso3166_snapshot: PinnedFileConfig


class PartyStructureBenchmarkSelectionConfig(_StrictModel):
    split: NonEmptyString
    require_template_wholly_in_split: Literal[True]
    unresolved_source_country_policy: Literal["exclude_document_and_audit_v1"]


class PartyStructureBenchmarkModelingConfig(_StrictModel):
    scope: Literal["full_gpu"]
    view: Literal["party_structure"]
    candidates: list[Literal["empirical", "gaussian_copula", "ctgan", "tvae"]] = Field(
        min_length=4, max_length=4
    )
    fold_count: Annotated[int, Field(ge=2)]
    fold_seed: Annotated[int, Field(ge=0, lt=2**32)]
    seeds: list[Annotated[int, Field(ge=0, lt=2**32)]] = Field(min_length=1)
    neural_epochs: Annotated[int, Field(gt=0)]
    neural_batch_size: Annotated[int, Field(ge=10)]
    proposal_multiplier: Annotated[int, Field(gt=0)]
    proposal_batch_rows: Annotated[int, Field(gt=0)] | None = None
    production_selection: Literal[False]

    @model_validator(mode="after")
    def comparison_is_complete_and_identifiable(self) -> PartyStructureBenchmarkModelingConfig:
        expected = ("empirical", "gaussian_copula", "ctgan", "tvae")
        if tuple(self.candidates) != expected:
            raise ValueError(f"party benchmark requires candidates in order {expected}")
        if len(self.seeds) != len(set(self.seeds)):
            raise ValueError("party benchmark seeds must be unique")
        if self.neural_batch_size % 10:
            raise ValueError("neural_batch_size must be divisible by CTGAN pac=10")
        return self


class SynthesisPartyStructureBenchmarkConfig(_StrictModel):
    schema_version: Literal[1]
    task: Literal["bill_of_lading_relation_explicit_v3"]
    run: SynthesisRunConfig
    inputs: PartyStructureBenchmarkInputsConfig
    selection: PartyStructureBenchmarkSelectionConfig
    modeling: PartyStructureBenchmarkModelingConfig


PartyIdentityRole = Literal[
    "shipper",
    "consignee",
    "notifyParties",
    "carrier",
    "forwardingAgent",
    "deliveryAgent",
]


class PartyIdentityProbeInputsConfig(_StrictModel):
    semantic_completion_run: CommittedArtifactDirectoryConfig
    completion_plans: DatasetFileConfig
    source_corpus: DatasetFileConfig
    source_target_field: Literal["target"]


class PartyIdentityProbeCaseConfig(_StrictModel):
    document_id: Annotated[str, StringConstraints(pattern=r"^doc_[0-9a-f]{64}$")]
    party_role: PartyIdentityRole
    occurrence: Annotated[int, Field(ge=0)]


class PartyIdentityProbePromptConfig(_StrictModel):
    path: NonEmptyString
    sha256: Annotated[str, StringConstraints(pattern=r"^[0-9a-f]{64}$")]

    @field_validator("path")
    @classmethod
    def safe_path(cls, value: str) -> str:
        return _safe_path(value)


class PartyIdentityProbePricingConfig(_StrictModel):
    currency: Literal["USD"]
    effective_date: date
    source_url: NonEmptyString
    input_usd_per_million: Annotated[float, Field(gt=0)]
    cached_input_usd_per_million: Annotated[float, Field(gt=0)]
    cache_write_multiplier: Annotated[float, Field(ge=1)]
    output_usd_per_million: Annotated[float, Field(gt=0)]

    @field_validator("source_url")
    @classmethod
    def source_is_https(cls, value: str) -> str:
        parsed = urlsplit(value)
        if parsed.scheme != "https" or not parsed.netloc:
            raise ValueError("pricing source_url must be an absolute HTTPS URL")
        return value


class PartyIdentityProbeProviderConfig(_StrictModel):
    kind: Literal["openai_responses"]
    model: Literal["gpt-5.6-luna"]
    api_key_env: Literal["OPENAI_API_KEY"]
    reasoning_effort: Literal["none", "low", "medium", "high", "xhigh", "max"]
    request_timeout_seconds: Annotated[float, Field(gt=0)]
    transport_max_retries: Annotated[int, Field(ge=0, le=5)]
    max_output_tokens: Annotated[int, Field(gt=0)]
    store_responses: Literal[True]
    pricing: PartyIdentityProbePricingConfig


class PartyIdentityProbeWorkflowConfig(_StrictModel):
    max_concurrent_requests: Annotated[int, Field(ge=1, le=5)]
    requests_per_case: Literal[1]
    structured_output_retries: Literal[0]
    provider_native_strict_json_schema: Literal[True]
    preserve_source_field_presence: Literal[True]
    provide_source_name_style_reference: Literal[True]
    persist_model_visible_messages: Literal[True]


class SynthesisPartyIdentityProbeConfig(_StrictModel):
    """Five or fewer first-pass, independently attributable party generations."""

    schema_version: Literal[1]
    task: Literal["bill_of_lading_relation_explicit_v5_party_identity_probe"]
    environment_file: NonEmptyString
    run: SynthesisRunConfig
    inputs: PartyIdentityProbeInputsConfig
    cases: tuple[PartyIdentityProbeCaseConfig, ...] = Field(min_length=3, max_length=5)
    prompt: PartyIdentityProbePromptConfig
    provider: PartyIdentityProbeProviderConfig
    workflow: PartyIdentityProbeWorkflowConfig

    @field_validator("environment_file")
    @classmethod
    def safe_environment_file(cls, value: str) -> str:
        return _safe_path(value)

    @field_validator("cases", mode="before")
    @classmethod
    def freeze_cases(cls, value: Any) -> Any:
        return tuple(value) if isinstance(value, list) else value

    @model_validator(mode="after")
    def cases_are_unique_and_fit_concurrency(self) -> SynthesisPartyIdentityProbeConfig:
        identities = tuple(
            (row.document_id, row.party_role, row.occurrence) for row in self.cases
        )
        if len(identities) != len(set(identities)):
            raise ValueError("party identity probe cases must be unique")
        if self.workflow.max_concurrent_requests > len(self.cases):
            raise ValueError("party identity concurrency cannot exceed the case count")
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


def load_synthesis_structured_baseline_config(
    path: Path,
) -> SynthesisStructuredBaselineConfig:
    try:
        value = yaml.load(path.read_bytes(), Loader=_UniqueKeySafeLoader)
    except UnicodeDecodeError as error:
        raise ValueError("synthesis configuration is not valid UTF-8") from error
    if not isinstance(value, dict):
        raise ValueError("synthesis configuration root must be a mapping")
    return SynthesisStructuredBaselineConfig.model_validate(value, strict=True)


def load_synthesis_route_scenario_pilot_config(
    path: Path,
) -> SynthesisRouteScenarioPilotConfig:
    try:
        value = yaml.load(path.read_bytes(), Loader=_UniqueKeySafeLoader)
    except UnicodeDecodeError as error:
        raise ValueError("synthesis configuration is not valid UTF-8") from error
    if not isinstance(value, dict):
        raise ValueError("synthesis configuration root must be a mapping")
    return SynthesisRouteScenarioPilotConfig.model_validate(value, strict=True)


def load_synthesis_party_structure_benchmark_config(
    path: Path,
) -> SynthesisPartyStructureBenchmarkConfig:
    try:
        value = yaml.load(path.read_bytes(), Loader=_UniqueKeySafeLoader)
    except UnicodeDecodeError as error:
        raise ValueError("synthesis configuration is not valid UTF-8") from error
    if not isinstance(value, dict):
        raise ValueError("synthesis configuration root must be a mapping")
    return SynthesisPartyStructureBenchmarkConfig.model_validate(value, strict=True)


def load_synthesis_controlled_pilot_config(path: Path) -> SynthesisControlledPilotConfig:
    try:
        value = yaml.load(path.read_bytes(), Loader=_UniqueKeySafeLoader)
    except UnicodeDecodeError as error:
        raise ValueError("synthesis configuration is not valid UTF-8") from error
    if not isinstance(value, dict):
        raise ValueError("synthesis configuration root must be a mapping")
    return SynthesisControlledPilotConfig.model_validate(value, strict=True)


def load_synthesis_semantic_plan_config(path: Path) -> SynthesisSemanticPlanConfig:
    try:
        value = yaml.load(path.read_bytes(), Loader=_UniqueKeySafeLoader)
    except UnicodeDecodeError as error:
        raise ValueError("synthesis configuration is not valid UTF-8") from error
    if not isinstance(value, dict):
        raise ValueError("synthesis configuration root must be a mapping")
    return SynthesisSemanticPlanConfig.model_validate(value, strict=True)


def load_synthesis_dangerous_goods_registry_config(
    path: Path,
) -> SynthesisDangerousGoodsRegistryConfig:
    try:
        value = yaml.load(path.read_bytes(), Loader=_UniqueKeySafeLoader)
    except UnicodeDecodeError as error:
        raise ValueError("synthesis configuration is not valid UTF-8") from error
    if not isinstance(value, dict):
        raise ValueError("synthesis configuration root must be a mapping")
    return SynthesisDangerousGoodsRegistryConfig.model_validate(value, strict=True)


def load_synthesis_dangerous_goods_analysis_config(
    path: Path,
) -> SynthesisDangerousGoodsAnalysisConfig:
    try:
        value = yaml.load(path.read_bytes(), Loader=_UniqueKeySafeLoader)
    except UnicodeDecodeError as error:
        raise ValueError("synthesis configuration is not valid UTF-8") from error
    if not isinstance(value, dict):
        raise ValueError("synthesis configuration root must be a mapping")
    return SynthesisDangerousGoodsAnalysisConfig.model_validate(value, strict=True)


def load_synthesis_dangerous_goods_plan_config(
    path: Path,
) -> SynthesisDangerousGoodsPlanConfig:
    try:
        value = yaml.load(path.read_bytes(), Loader=_UniqueKeySafeLoader)
    except UnicodeDecodeError as error:
        raise ValueError("synthesis configuration is not valid UTF-8") from error
    if not isinstance(value, dict):
        raise ValueError("synthesis configuration root must be a mapping")
    return SynthesisDangerousGoodsPlanConfig.model_validate(value, strict=True)


def load_synthesis_semantic_completion_config(
    path: Path,
) -> SynthesisSemanticCompletionConfig:
    try:
        value = yaml.load(path.read_bytes(), Loader=_UniqueKeySafeLoader)
    except UnicodeDecodeError as error:
        raise ValueError("synthesis configuration is not valid UTF-8") from error
    if not isinstance(value, dict):
        raise ValueError("synthesis configuration root must be a mapping")
    return SynthesisSemanticCompletionConfig.model_validate(value, strict=True)


def load_synthesis_party_identity_probe_config(
    path: Path,
) -> SynthesisPartyIdentityProbeConfig:
    try:
        value = yaml.load(path.read_bytes(), Loader=_UniqueKeySafeLoader)
    except UnicodeDecodeError as error:
        raise ValueError("synthesis configuration is not valid UTF-8") from error
    if not isinstance(value, dict):
        raise ValueError("synthesis configuration root must be a mapping")
    return SynthesisPartyIdentityProbeConfig.model_validate(value, strict=True)
