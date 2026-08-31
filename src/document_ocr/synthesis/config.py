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


class CommittedDirectoryConfig(PinnedDirectoryConfig):
    """Pinned immutable run directory with its transaction receipt."""

    commit_sha256: Annotated[str, StringConstraints(pattern=r"^[0-9a-f]{64}$")]
    transaction_sha256: Annotated[str, StringConstraints(pattern=r"^[0-9a-f]{64}$")]


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
    seed: int
    require_template_wholly_in_split: Literal[True]
    maximum_per_template: Literal[1]


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

    schema_version: Literal[2]
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


class ControlledHsGenerationConfig(_StrictModel):
    schema_version: Literal[1]
    observed_chapter_mixture_permyriad: Annotated[int, Field(ge=0, le=10_000)]
    registry_wide_mixture_permyriad: Annotated[int, Field(ge=0, le=10_000)]
    gb_tariff_extension_permyriad: Annotated[int, Field(ge=0, le=10_000)]
    observed_chapter_weighting: Literal["fit_document_count_v1"]
    observed_chapter_hs6_weighting: Literal["uniform_registry_hs6_within_selected_chapter_v1"]
    registry_hs6_weighting: Literal["uniform_registry_hs6_v1"]
    gb_tariff_leaf_weighting: Literal["uniform_registry_leaves_v1"]

    @model_validator(mode="after")
    def chapter_mixture_is_complete(self) -> ControlledHsGenerationConfig:
        if self.observed_chapter_mixture_permyriad + self.registry_wide_mixture_permyriad != 10_000:
            raise ValueError("controlled HS mixture weights must sum exactly to 10000")
        return self


class ControlledTransportGenerationConfig(_StrictModel):
    vessel_name_method: Literal["deferred_by_explicit_scope_v1"]
    voyage_number_method: Literal["observed_character_class_shape_v1"]
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
    cargo_origin_name_only_policy: Literal[
        "replace_without_resolving_or_copying_source_name_v2"
    ]
    dangerous_goods_method: Literal["disabled_pending_licensed_maritime_authoritative_registry_v1"]
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
