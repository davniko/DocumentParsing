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


class PossiblyEmptyDatasetFileConfig(DatasetFileConfig):
    """Pinned JSONL artifact whose valid result may contain no rows."""

    records: Annotated[int, Field(ge=0)]


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
    expected_isolated_fit_documents: Annotated[int, Field(gt=0)]
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
    sampling_namespace: NonEmptyString
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
    printed_topology_policy: Literal[
        "sample_optional_fields_v1",
        "preserve_selected_template_v1",
    ]
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
    fit_partition_report: PinnedFileConfig
    package_registry: PinnedFileConfig
    package_registry_entries: Annotated[int, Field(gt=0)]
    package_compatibility_resolutions: PossiblyEmptyDatasetFileConfig
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
    profile_compatibility_policy: Literal[
        "renormalize_configured_weights_over_fit_supported_profiles_v1"
    ]
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
    sampling_namespace: NonEmptyString
    seed: Annotated[int, Field(ge=0, lt=2**64)]
    package_goods_method: Literal["fit_hs_heading_or_thermal_joint_with_constrained_dg_catalog_v1"]
    package_signature_distribution: Literal["source_group_occurrence_weighted_v1"]
    registry_identity_distribution: Literal["uniform_within_selected_hs_heading_v1"]
    equipment: SemanticCompletionEquipmentConfig
    thermal: SemanticCompletionThermalConfig
    transport: SemanticCompletionTransportConfig
    flashpoint: SemanticCompletionFlashpointConfig
    preserve_container_cardinality: Literal[True]
    preserve_cargo_cardinality: Literal[True]
    preserve_relation_topology: Literal[True]
    printed_topology_policy: Literal[
        "sample_optional_fields_v1",
        "preserve_selected_template_v1",
    ]
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
    "consolidator",
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


class LinguisticProbePromptConfig(_StrictModel):
    path: NonEmptyString
    sha256: Annotated[str, StringConstraints(pattern=r"^[0-9a-f]{64}$")]

    @field_validator("path")
    @classmethod
    def safe_path(cls, value: str) -> str:
        return _safe_path(value)


class LinguisticProbePricingConfig(_StrictModel):
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


class LinguisticGenerationSettingsConfig(_StrictModel):
    """Optional provider-native sampling controls.

    Every field is optional by design.  When this section is absent, or when a
    field is omitted, the caller must not send that setting to PydanticAI so
    that the model/provider default remains authoritative.
    """

    temperature: Annotated[float, Field(ge=0, le=2)] | None = None
    top_p: Annotated[float, Field(ge=0, le=1)] | None = None
    text_verbosity: Literal["low", "medium", "high"] | None = None


class LinguisticProbeProviderConfig(_StrictModel):
    kind: Literal["openai_responses"]
    model: Literal["gpt-5.6-luna", "gpt-5.6-terra"]
    api_key_env: Literal["OPENAI_API_KEY"]
    reasoning_effort: Literal["none", "low", "medium", "high", "xhigh", "max"]
    request_timeout_seconds: Annotated[float, Field(gt=0)]
    transport_max_retries: Annotated[int, Field(ge=0, le=5)]
    max_output_tokens: Annotated[int, Field(gt=0)]
    store_responses: Literal[True]
    service_tier: Literal["auto", "default", "flex", "priority"] | None = None
    generation_settings: LinguisticGenerationSettingsConfig | None = None
    pricing: LinguisticProbePricingConfig


class OpenRouterMaxPriceConfig(_StrictModel):
    """Hard OpenRouter endpoint-price ceiling, expressed in USD per million tokens."""

    prompt: Annotated[float, Field(gt=0)]
    completion: Annotated[float, Field(gt=0)]


class OpenRouterRewriteProviderConfig(_StrictModel):
    """OpenRouter configuration supported by the atomic raw-text rewrite flow."""

    kind: Literal["openrouter"]
    model: Annotated[
        str,
        StringConstraints(
            strip_whitespace=True,
            pattern=r"^[A-Za-z0-9][A-Za-z0-9._~-]*/[A-Za-z0-9][A-Za-z0-9._~:/-]*$",
        ),
    ]
    api_key_env: Literal["OPENROUTER_API_KEY"]
    reasoning_effort: Literal["none", "minimal", "low", "medium", "high", "xhigh"]
    request_timeout_seconds: Annotated[float, Field(gt=0)]
    transport_max_retries: Annotated[int, Field(ge=0, le=5)]
    max_output_tokens: Annotated[int, Field(gt=0)]
    require_parameters: Literal[True]
    data_collection: Literal["deny"]
    allow_fallbacks: bool = True
    provider_only: tuple[NonEmptyString, ...] | None = None
    provider_order: tuple[NonEmptyString, ...] | None = None
    provider_sort: Literal["price", "throughput", "latency"] | None = None
    max_price: OpenRouterMaxPriceConfig | None = None
    generation_settings: LinguisticGenerationSettingsConfig | None = None
    pricing: LinguisticProbePricingConfig

    @field_validator("provider_only", "provider_order", mode="before")
    @classmethod
    def freeze_provider_lists(cls, value: Any) -> Any:
        return tuple(value) if isinstance(value, list) else value

    @model_validator(mode="after")
    def settings_are_openrouter_compatible(self) -> OpenRouterRewriteProviderConfig:
        if (
            self.generation_settings is not None
            and self.generation_settings.text_verbosity is not None
        ):
            raise ValueError("OpenRouter rewrite models do not support text_verbosity")
        if self.provider_only is not None:
            if not self.provider_only:
                raise ValueError("provider_only cannot be empty")
            if len(self.provider_only) != len(set(self.provider_only)):
                raise ValueError("provider_only entries must be unique")
        if self.provider_order is not None:
            if not self.provider_order:
                raise ValueError("provider_order cannot be empty")
            if len(self.provider_order) != len(set(self.provider_order)):
                raise ValueError("provider_order entries must be unique")
            if self.provider_sort is not None:
                raise ValueError("provider_order and provider_sort are mutually exclusive")
            if self.provider_only is not None and not set(self.provider_order).issubset(
                self.provider_only
            ):
                raise ValueError("provider_order entries must be allowed by provider_only")
        if self.max_price is not None:
            if self.pricing.input_usd_per_million > self.max_price.prompt:
                raise ValueError("pinned input price exceeds the OpenRouter prompt-price ceiling")
            if self.pricing.output_usd_per_million > self.max_price.completion:
                raise ValueError(
                    "pinned output price exceeds the OpenRouter completion-price ceiling"
                )
        return self


RawTextRewriteProviderConfig = Annotated[
    LinguisticProbeProviderConfig | OpenRouterRewriteProviderConfig,
    Field(discriminator="kind"),
]


class PackageCompatibilityCatalogInputsConfig(_StrictModel):
    dangerous_goods_run: CommittedArtifactDirectoryConfig
    dangerous_goods_plans: DatasetFileConfig
    source_task_constraints: PinnedFileConfig
    fit_partition_report: PinnedFileConfig
    package_registry: PinnedFileConfig
    package_registry_entries: Annotated[int, Field(gt=0)]


class PackageCompatibilityCatalogWorkflowConfig(_StrictModel):
    max_concurrent_requests: Annotated[int, Field(ge=1, le=8)]
    requests_per_context: Literal[1]
    structured_output_retries: Literal[0]
    provider_native_strict_json_schema: Literal[True]
    persist_model_visible_messages: Literal[True]


class PackageCompatibilityFitTemperatureConfig(_StrictModel):
    frozen_minimum_celsius: float
    frozen_maximum_celsius: float
    chilled_minimum_celsius: float
    chilled_maximum_celsius: float

    @model_validator(mode="after")
    def ranges_are_ordered_and_disjoint(self) -> PackageCompatibilityFitTemperatureConfig:
        if self.frozen_minimum_celsius > self.frozen_maximum_celsius:
            raise ValueError("frozen fit temperature range is reversed")
        if self.chilled_minimum_celsius > self.chilled_maximum_celsius:
            raise ValueError("chilled fit temperature range is reversed")
        if self.frozen_maximum_celsius >= self.chilled_minimum_celsius:
            raise ValueError("frozen and chilled fit temperature ranges must be disjoint")
        return self


class SynthesisPackageCompatibilityCatalogConfig(_StrictModel):
    """Compile reusable constrained decisions only for empirically unsupported DG contexts."""

    schema_version: Literal[1]
    task: Literal["bill_of_lading_package_goods_compatibility_catalog_v1"]
    environment_file: NonEmptyString
    run: SynthesisRunConfig
    source: SynthesisSourceConfig
    inputs: PackageCompatibilityCatalogInputsConfig
    fit_temperature: PackageCompatibilityFitTemperatureConfig
    prompt: LinguisticProbePromptConfig
    provider: LinguisticProbeProviderConfig
    workflow: PackageCompatibilityCatalogWorkflowConfig

    @field_validator("environment_file")
    @classmethod
    def safe_environment_file(cls, value: str) -> str:
        return _safe_path(value)

    @model_validator(mode="after")
    def source_and_plan_counts_are_valid(self) -> SynthesisPackageCompatibilityCatalogConfig:
        if self.source.fields.input_sha256 is None:
            raise ValueError("package compatibility requires source input SHA-256 values")
        if self.inputs.dangerous_goods_plans.records <= 0:
            raise ValueError("package compatibility requires at least one DG plan")
        return self


class PartyIdentityProbeWorkflowConfig(_StrictModel):
    max_concurrent_requests: Annotated[int, Field(ge=1, le=5)]
    requests_per_case: Literal[1]
    structured_output_retries: Literal[0]
    provider_native_strict_json_schema: Literal[True]
    preserve_source_field_presence: Literal[True]
    provide_source_name_style_reference: Literal[True]
    persist_model_visible_messages: Literal[True]


class SynthesisPartyIdentityProbeConfig(_StrictModel):
    """Up to fifty first-pass, independently attributable party generations."""

    schema_version: Literal[1]
    task: Literal["bill_of_lading_relation_explicit_v5_party_identity_probe"]
    environment_file: NonEmptyString
    run: SynthesisRunConfig
    inputs: PartyIdentityProbeInputsConfig
    cases: tuple[PartyIdentityProbeCaseConfig, ...] = Field(min_length=3, max_length=50)
    prompt: LinguisticProbePromptConfig
    provider: LinguisticProbeProviderConfig
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
        identities = tuple((row.document_id, row.party_role, row.occurrence) for row in self.cases)
        if len(identities) != len(set(identities)):
            raise ValueError("party identity probe cases must be unique")
        if self.workflow.max_concurrent_requests > len(self.cases):
            raise ValueError("party identity concurrency cannot exceed the case count")
        return self


class CargoLanguageProbeInputsConfig(_StrictModel):
    semantic_completion_run: CommittedArtifactDirectoryConfig
    completion_plans: DatasetFileConfig


class CargoLanguageProbeCaseConfig(_StrictModel):
    document_id: Annotated[str, StringConstraints(pattern=r"^doc_[0-9a-f]{64}$")]


class CargoLanguageProbeWorkflowConfig(_StrictModel):
    max_concurrent_requests: Annotated[int, Field(ge=1, le=5)]
    requests_per_case: Literal[1]
    structured_output_retries: Literal[0]
    provider_native_strict_json_schema: Literal[True]
    preserve_source_field_presence: Literal[True]
    preserve_generic_marks_literals: Literal[True]
    exclude_categorical_printed_surfaces: Literal[True]
    persist_model_visible_messages: Literal[True]


class SynthesisCargoLanguageProbeConfig(_StrictModel):
    """Between one and fifty independent cargo-language generations."""

    schema_version: Literal[1]
    task: Literal["bill_of_lading_relation_explicit_v5_cargo_language_probe"]
    environment_file: NonEmptyString
    run: SynthesisRunConfig
    inputs: CargoLanguageProbeInputsConfig
    cases: tuple[CargoLanguageProbeCaseConfig, ...] = Field(min_length=1, max_length=50)
    prompt: LinguisticProbePromptConfig
    provider: LinguisticProbeProviderConfig
    workflow: CargoLanguageProbeWorkflowConfig

    @field_validator("environment_file")
    @classmethod
    def safe_environment_file(cls, value: str) -> str:
        return _safe_path(value)

    @field_validator("cases", mode="before")
    @classmethod
    def freeze_cases(cls, value: Any) -> Any:
        return tuple(value) if isinstance(value, list) else value

    @model_validator(mode="after")
    def cases_are_unique_and_fit_concurrency(self) -> SynthesisCargoLanguageProbeConfig:
        document_ids = tuple(row.document_id for row in self.cases)
        if len(document_ids) != len(set(document_ids)):
            raise ValueError("cargo-language probe cases must be unique")
        if self.workflow.max_concurrent_requests > len(self.cases):
            raise ValueError("cargo-language concurrency cannot exceed the case count")
        return self


class LinguisticCompletionInputsConfig(_StrictModel):
    semantic_completion_run: CommittedArtifactDirectoryConfig
    completion_plans: DatasetFileConfig
    source_corpus: DatasetFileConfig
    source_target_field: Literal["target"]
    source_target_schema: Literal["bill_of_lading_relation_explicit_v3"]
    resume_run: CommittedArtifactDirectoryConfig | None = None


class LinguisticCompletionPromptsConfig(_StrictModel):
    party_identity: LinguisticProbePromptConfig
    cargo_language: LinguisticProbePromptConfig


class LinguisticCompletionStageLimitsConfig(_StrictModel):
    party_max_output_tokens: Annotated[int, Field(gt=0)]
    cargo_max_output_tokens: Annotated[int, Field(gt=0)]


class LinguisticCompletionWorkflowConfig(_StrictModel):
    max_concurrent_requests: Annotated[int, Field(ge=1, le=64)]
    max_concurrent_documents: Annotated[int, Field(ge=1, le=64)]
    requests_per_attempt: Literal[1]
    semantic_validation_attempts: Annotated[int, Field(ge=1, le=3)]
    structured_output_retries: Literal[0]
    provider_native_strict_json_schema: Literal[True]
    static_schema_first_attempt: Literal[True]
    topology_constrained_schema_on_retry: Literal[True]
    preserve_source_field_presence: Literal[True]
    preserve_same_as_without_generation: Literal[True]
    reuse_explicit_duplicate_notify_identity: Literal[True]
    preserve_generic_marks_literals: Literal[True]
    exclude_categorical_printed_surfaces: Literal[True]
    persist_model_visible_messages: Literal[True]
    process_all_completion_plans: Literal[True]
    publish_training_records: Literal[False]


class SynthesisLinguisticCompletionConfig(_StrictModel):
    """Complete party and cargo language over one committed semantic-plan set."""

    schema_version: Literal[1]
    task: Literal["bill_of_lading_relation_explicit_v5_linguistic_completion"]
    environment_file: NonEmptyString
    run: SynthesisRunConfig
    inputs: LinguisticCompletionInputsConfig
    prompts: LinguisticCompletionPromptsConfig
    provider: LinguisticProbeProviderConfig
    stage_limits: LinguisticCompletionStageLimitsConfig
    workflow: LinguisticCompletionWorkflowConfig

    @field_validator("environment_file")
    @classmethod
    def safe_environment_file(cls, value: str) -> str:
        return _safe_path(value)

    @model_validator(mode="after")
    def completion_contract_is_bounded(self) -> SynthesisLinguisticCompletionConfig:
        if self.inputs.completion_plans.records <= 0:
            raise ValueError("linguistic completion requires at least one semantic plan")
        if self.stage_limits.party_max_output_tokens > self.provider.max_output_tokens:
            raise ValueError("party output limit exceeds the provider output limit")
        if self.stage_limits.cargo_max_output_tokens > self.provider.max_output_tokens:
            raise ValueError("cargo output limit exceeds the provider output limit")
        return self


class RawTextRewriteInputsConfig(_StrictModel):
    linguistic_completion_run: CommittedArtifactDirectoryConfig
    linguistic_results: DatasetFileConfig
    synthetic_targets: DatasetFileConfig
    linguistic_summary: PinnedFileConfig
    source_corpus: DatasetFileConfig
    source_target_field: Literal["target"]
    source_target_schema: Literal["bill_of_lading_relation_explicit_v3"]
    synthetic_target_schema: Literal["bill_of_lading_relation_explicit_v5"]
    document_features: DatasetFileConfig


class RawTextRewriteCaseConfig(_StrictModel):
    document_id: Annotated[str, StringConstraints(pattern=r"^doc_[0-9a-f]{64}$")]


class RawTextRewriteWorkflowConfig(_StrictModel):
    max_concurrent_requests: Annotated[int, Field(ge=1, le=16)]
    max_concurrent_cases: Annotated[int, Field(ge=1, le=10)]
    max_model_requests_per_case: Annotated[int, Field(ge=2, le=32)]
    tool_retries: Annotated[int, Field(ge=1, le=8)]
    provider_native_strict_json_schema: Literal[True]
    require_edit_tool: Literal[True]
    require_diff_inspection: Literal[True]
    preserve_page_markers: Literal[True]
    preserve_newline_sequence: Literal[True]
    preserve_untouched_bytes: Literal[True]
    anonymize_auxiliary_personal_data: Literal[True]
    persist_model_visible_messages: Literal[True]
    publish_training_records: Literal[False]
    include_full_text_in_report: Literal[True]


class SynthesisRawTextRewriteProbeConfig(_StrictModel):
    """Ten independent tool-mediated synthetic raw-OCR rewrite probes."""

    schema_version: Literal[1]
    task: Literal["bill_of_lading_synthetic_raw_text_rewrite_probe_v1"]
    environment_file: NonEmptyString
    run: SynthesisRunConfig
    inputs: RawTextRewriteInputsConfig
    cases: tuple[RawTextRewriteCaseConfig, ...] = Field(min_length=10, max_length=10)
    prompt: LinguisticProbePromptConfig
    provider: LinguisticProbeProviderConfig
    workflow: RawTextRewriteWorkflowConfig

    @field_validator("environment_file")
    @classmethod
    def safe_environment_file(cls, value: str) -> str:
        return _safe_path(value)

    @field_validator("cases", mode="before")
    @classmethod
    def freeze_cases(cls, value: Any) -> Any:
        return tuple(value) if isinstance(value, list) else value

    @model_validator(mode="after")
    def cases_are_unique_and_fit_concurrency(self) -> SynthesisRawTextRewriteProbeConfig:
        document_ids = tuple(row.document_id for row in self.cases)
        if len(document_ids) != len(set(document_ids)):
            raise ValueError("raw-text rewrite probe cases must be unique")
        if self.workflow.max_concurrent_cases > len(self.cases):
            raise ValueError("raw-text rewrite concurrency exceeds the case count")
        if self.workflow.max_concurrent_requests < self.workflow.max_concurrent_cases:
            raise ValueError("request concurrency must cover concurrent rewrite cases")
        if self.provider.max_output_tokens < 2048:
            raise ValueError("raw-text rewrite output limit is too small for audit receipts")
        return self


class RawTextRewriteCyclePromptsConfig(_StrictModel):
    editor: LinguisticProbePromptConfig
    reviewer: LinguisticProbePromptConfig


class RawTextRewriteCycleProvidersConfig(_StrictModel):
    editor: RawTextRewriteProviderConfig
    reviewer: RawTextRewriteProviderConfig

    @model_validator(mode="after")
    def one_credential_contract(self) -> RawTextRewriteCycleProvidersConfig:
        if self.editor.kind != self.reviewer.kind:
            raise ValueError("rewrite editor and reviewer must use the same provider kind")
        if self.editor.api_key_env != self.reviewer.api_key_env:
            raise ValueError("rewrite editor and reviewer must use the same credential source")
        return self


class RawTextRewriteTargetIntegrityConfig(_StrictModel):
    capacity_reprojection_method: Literal["preserve_sampled_capacity_utilization_v1"]
    transport_capacity: TransportCapacityConfig
    iso3166_snapshot: PinnedFileConfig
    route_locations: PinnedFileConfig
    route_location_records: Annotated[int, Field(gt=0)]
    customs_program_registry: PinnedFileConfig
    customs_program_registry_entries: Annotated[int, Field(gt=0)]
    package_registry: PinnedFileConfig
    package_registry_entries: Annotated[int, Field(gt=0)]


class RawTextRewriteCycleWorkflowConfig(_StrictModel):
    max_concurrent_requests: Annotated[int, Field(ge=1, le=16)]
    max_concurrent_cases: Annotated[int, Field(ge=1, le=10)]
    max_editor_requests_per_pass: Annotated[int, Field(ge=1, le=3)]
    max_reviewer_requests_per_cycle: Annotated[int, Field(ge=1, le=4)]
    editor_output_retries: Annotated[int, Field(ge=0, le=2)]
    reviewer_output_retries: Annotated[int, Field(ge=0, le=3)]
    max_correction_cycles: Annotated[int, Field(ge=1, le=3)]
    max_total_estimated_cost_usd_per_case: Annotated[float, Field(gt=0)]
    provider_strict_output_function_schema: Literal[True]
    require_terminal_atomic_edit_output: Literal[True]
    require_line_addressed_atomic_patches: Literal[True]
    require_independent_semantic_review: Literal[True]
    require_review_evidence_substrings: Literal[True]
    preserve_page_markers_and_order: Literal[True]
    preserve_source_newline_convention: Literal[True]
    preserve_source_line_count: Literal[True]
    preserve_source_blank_line_topology: Literal[True]
    preserve_inline_labeled_slot_population: Literal[True]
    preserve_unchanged_source_status_surfaces: Literal[True]
    enforce_target_value_occurrence_counts: Literal[True]
    synthesize_raw_auxiliary_agent_identities: Literal[True]
    enforce_carrier_receipt_equipment_breakdown: Literal[True]
    preserve_untouched_bytes: Literal[True]
    anonymize_auxiliary_sensitive_data: Literal[True]
    synthesize_auxiliary_flavor: Literal[True]
    preserve_auxiliary_slot_topology: Literal[True]
    preserve_target_label_and_synthesize_compound_party_flavor: Literal[True]
    provide_authoritative_label_change_contract: Literal[True]
    remove_target_absent_extractable_assertions_naturally: Literal[True]
    preserve_source_only_raw_slots_as_synthetic_flavor: Literal[True]
    preserve_semantically_equivalent_status_surfaces: Literal[True]
    forbid_new_unanchored_extractable_facts: Literal[True]
    reject_placeholder_substitutions: Literal[True]
    adapt_jurisdiction_bound_auxiliary_flavor: Literal[True]
    require_source_target_topology_match: Literal[True]
    enforce_deterministic_surface_requirements: Literal[True]
    use_compact_model_change_contract: Literal[True]
    persist_every_stage_and_model_message: Literal[True]
    use_provider_explicit_prompt_cache: bool
    publish_training_records: Literal[False]
    include_full_text_in_report: Literal[True]


class SynthesisRawTextRewriteCycleProbeConfig(_StrictModel):
    """Atomic edit-review-correct probes over one to ten pinned documents."""

    schema_version: Literal[12, 13]
    task: Literal[
        "bill_of_lading_synthetic_raw_text_atomic_rewrite_probe_v12",
        "bill_of_lading_synthetic_raw_text_atomic_rewrite_probe_v13",
    ]
    environment_file: NonEmptyString
    run: SynthesisRunConfig
    inputs: RawTextRewriteInputsConfig
    cases: tuple[RawTextRewriteCaseConfig, ...] = Field(min_length=1, max_length=10)
    prompts: RawTextRewriteCyclePromptsConfig
    providers: RawTextRewriteCycleProvidersConfig
    target_integrity: RawTextRewriteTargetIntegrityConfig
    workflow: RawTextRewriteCycleWorkflowConfig

    @field_validator("environment_file")
    @classmethod
    def safe_environment_file(cls, value: str) -> str:
        return _safe_path(value)

    @field_validator("cases", mode="before")
    @classmethod
    def freeze_cases(cls, value: Any) -> Any:
        return tuple(value) if isinstance(value, list) else value

    @model_validator(mode="after")
    def cases_and_limits_are_consistent(self) -> SynthesisRawTextRewriteCycleProbeConfig:
        expected_task = (
            "bill_of_lading_synthetic_raw_text_atomic_rewrite_probe_v12"
            if self.schema_version == 12
            else "bill_of_lading_synthetic_raw_text_atomic_rewrite_probe_v13"
        )
        if self.task != expected_task:
            raise ValueError("raw-text rewrite task version differs from schema_version")
        uses_openrouter = self.providers.editor.kind == "openrouter"
        if uses_openrouter and self.schema_version != 13:
            raise ValueError("OpenRouter raw-text rewrite runs require schema_version 13")
        if uses_openrouter and self.workflow.use_provider_explicit_prompt_cache:
            raise ValueError(
                "OpenRouter GLM rewrite runs cannot enable unsupported explicit cache points"
            )
        if not uses_openrouter and not self.workflow.use_provider_explicit_prompt_cache:
            raise ValueError("OpenAI rewrite runs require their configured explicit prompt cache")
        document_ids = tuple(row.document_id for row in self.cases)
        if len(document_ids) != len(set(document_ids)):
            raise ValueError("raw-text rewrite cycle cases must be unique")
        if self.workflow.max_concurrent_cases > len(self.cases):
            raise ValueError("raw-text rewrite cycle concurrency exceeds the case count")
        if self.workflow.max_concurrent_requests < self.workflow.max_concurrent_cases:
            raise ValueError("request concurrency must cover concurrent rewrite cycle cases")
        if self.providers.editor.max_output_tokens < 4096:
            raise ValueError("rewrite editor output limit is too small for atomic patch output")
        if self.providers.reviewer.max_output_tokens < 1024:
            raise ValueError("rewrite reviewer output limit is too small for semantic findings")
        if self.workflow.max_editor_requests_per_pass <= self.workflow.editor_output_retries:
            raise ValueError("editor request limit must exceed its output retry count")
        if self.workflow.max_reviewer_requests_per_cycle <= self.workflow.reviewer_output_retries:
            raise ValueError("reviewer request limit must exceed its output retry count")
        return self


class RawTextHybridProbeInputsConfig(_StrictModel):
    """Pinned corpus plus the audited atomic-run contracts used by the hybrid probe."""

    baseline_atomic_run: CommittedArtifactDirectoryConfig
    source_corpus: DatasetFileConfig
    synthetic_targets: DatasetFileConfig


class RawTextHybridProbeWorkflowConfig(_StrictModel):
    """Fail-closed bounds for compiler coverage and the deliberately tiny API probe."""

    audit_documents: Annotated[int, Field(ge=1, le=50)]
    max_model_documents: Literal[1, 2]
    max_model_requests_per_stage: Literal[1]
    context_lines_per_residual: Annotated[int, Field(ge=0, le=3)]
    merge_residual_gap_lines: Annotated[int, Field(ge=0, le=3)]
    provider_native_json_schema: Literal[True]
    require_host_owned_spans: Literal[True]
    require_transactional_validation: Literal[True]
    require_compact_independent_review: Literal[True]
    persist_every_prompt_response_and_receipt: Literal[True]
    publish_training_records: Literal[False]


class SynthesisRawTextHybridProbeConfig(_StrictModel):
    """Compiler-first raw-OCR rewrite experiment with at most two model documents."""

    schema_version: Literal[1]
    task: Literal["bill_of_lading_synthetic_raw_text_hybrid_compiler_probe_v1"]
    environment_file: NonEmptyString
    run: SynthesisRunConfig
    inputs: RawTextHybridProbeInputsConfig
    cases: tuple[RawTextRewriteCaseConfig, ...] = Field(min_length=1, max_length=2)
    prompts: RawTextRewriteCyclePromptsConfig
    providers: RawTextRewriteCycleProvidersConfig
    workflow: RawTextHybridProbeWorkflowConfig

    @field_validator("environment_file")
    @classmethod
    def safe_environment_file(cls, value: str) -> str:
        return _safe_path(value)

    @field_validator("cases", mode="before")
    @classmethod
    def freeze_cases(cls, value: Any) -> Any:
        return tuple(value) if isinstance(value, list) else value

    @model_validator(mode="after")
    def cases_and_provider_are_bounded(self) -> SynthesisRawTextHybridProbeConfig:
        document_ids = tuple(row.document_id for row in self.cases)
        if len(document_ids) != len(set(document_ids)):
            raise ValueError("hybrid raw-text probe cases must be unique")
        if len(self.cases) > self.workflow.max_model_documents:
            raise ValueError("hybrid case count exceeds max_model_documents")
        if self.workflow.audit_documents > self.inputs.synthetic_targets.records:
            raise ValueError("audit_documents exceeds the pinned synthetic target count")
        if self.providers.editor.kind == "openrouter":
            if self.providers.editor.max_output_tokens > 4096:
                raise ValueError("hybrid OpenRouter editor output must remain compact")
            if self.providers.reviewer.max_output_tokens > 2048:
                raise ValueError("hybrid OpenRouter reviewer output must remain compact")
        return self


class RawTextHybridBatchWorkflowConfig(_StrictModel):
    """Bounded compiler contract over one complete, explicitly pinned target cohort."""

    audit_documents: Annotated[int, Field(ge=1, le=100_000)]
    max_concurrent_cases: Annotated[int, Field(ge=1, le=16)]
    max_concurrent_requests: Annotated[int, Field(ge=1, le=16)]
    max_model_requests_per_stage: Literal[1]
    context_lines_per_residual: Annotated[int, Field(ge=0, le=3)]
    merge_residual_gap_lines: Annotated[int, Field(ge=0, le=3)]
    provider_native_json_schema: Literal[True]
    require_host_owned_spans: Literal[True]
    require_transactional_validation: Literal[True]
    require_compact_independent_review: Literal[True]
    require_manual_full_text_audit_before_training: Literal[True]
    persist_every_prompt_response_and_receipt: Literal[True]
    publish_training_records: Literal[False]


class SynthesisRawTextHybridBatchConfig(_StrictModel):
    """Compiler-first model comparison; never publishes training records."""

    schema_version: Literal[2]
    task: Literal["bill_of_lading_synthetic_raw_text_hybrid_batch_v2"]
    environment_file: NonEmptyString
    run: SynthesisRunConfig
    inputs: RawTextRewriteInputsConfig
    selection: Literal["all_pinned_targets_in_order"]
    prompts: RawTextRewriteCyclePromptsConfig
    providers: RawTextRewriteCycleProvidersConfig
    target_integrity: RawTextRewriteTargetIntegrityConfig
    workflow: RawTextHybridBatchWorkflowConfig

    @field_validator("environment_file")
    @classmethod
    def safe_environment_file(cls, value: str) -> str:
        return _safe_path(value)

    @model_validator(mode="after")
    def batch_contract_is_bounded(self) -> SynthesisRawTextHybridBatchConfig:
        if self.inputs.synthetic_targets.records != self.workflow.audit_documents:
            raise ValueError("hybrid batch audit count differs from pinned synthetic targets")
        if self.workflow.max_concurrent_cases > self.inputs.synthetic_targets.records:
            raise ValueError("hybrid batch concurrency exceeds the case count")
        if self.workflow.max_concurrent_requests < self.workflow.max_concurrent_cases:
            raise ValueError("request concurrency must cover concurrent hybrid cases")
        if self.providers.editor.max_output_tokens > 8192:
            raise ValueError("hybrid batch editor output exceeds the measured compact bound")
        if self.providers.reviewer.max_output_tokens > 2048:
            raise ValueError("hybrid batch reviewer output must remain compact")
        return self


class RawTextInventoryReferenceRunsConfig(_StrictModel):
    """Immutable negative fixtures produced by the earlier restricted-span flow."""

    glm: CommittedArtifactDirectoryConfig
    luna: CommittedArtifactDirectoryConfig


class RawTextInventoryProbeWorkflowConfig(_StrictModel):
    """Bounds for the full-document inventory and one-request rendering experiment."""

    regression_documents: Literal[12]
    max_live_documents: Literal[1, 2]
    output_mode: Literal["native", "tool"]
    max_model_requests_per_document: Annotated[int, Field(ge=1, le=3)]
    deterministic_auxiliary_rendering: Literal[True]
    require_every_output_slot: Literal[True]
    require_full_document_regression_oracle: Literal[True]
    require_no_retained_changed_source_values: Literal[True]
    require_no_retained_source_auxiliary_values: Literal[True]
    persist_every_prompt_response_and_receipt: Literal[True]
    publish_training_records: Literal[False]


class SynthesisRawTextInventoryProbeConfig(_StrictModel):
    """Two-case maximum experiment for inventory-complete bounded OCR rendering."""

    schema_version: Literal[1]
    task: Literal["bill_of_lading_synthetic_raw_text_inventory_probe_v1"]
    environment_file: NonEmptyString
    run: SynthesisRunConfig
    base_batch_config: PinnedFileConfig
    regression_oracle: PinnedFileConfig
    reference_runs: RawTextInventoryReferenceRunsConfig
    cases: tuple[RawTextRewriteCaseConfig, ...] = Field(min_length=1, max_length=2)
    prompt: PinnedFileConfig
    provider: RawTextRewriteProviderConfig
    workflow: RawTextInventoryProbeWorkflowConfig

    @field_validator("environment_file")
    @classmethod
    def safe_environment_file(cls, value: str) -> str:
        return _safe_path(value)

    @field_validator("cases", mode="before")
    @classmethod
    def freeze_cases(cls, value: Any) -> Any:
        return tuple(value) if isinstance(value, list) else value

    @model_validator(mode="after")
    def probe_is_bounded_and_consistent(self) -> SynthesisRawTextInventoryProbeConfig:
        document_ids = tuple(row.document_id for row in self.cases)
        if len(document_ids) != len(set(document_ids)):
            raise ValueError("inventory probe cases must be unique")
        if len(self.cases) > self.workflow.max_live_documents:
            raise ValueError("inventory probe case count exceeds max_live_documents")
        if self.provider.max_output_tokens > 8192:
            raise ValueError("inventory probe output allowance exceeds its bounded contract")
        return self


class RawTextInventoryBatchWorkflowConfig(_StrictModel):
    """Bounds for an inventory-complete evaluation run."""

    regression_documents: Literal[12]
    documents: Annotated[int, Field(ge=1, le=100_000)]
    max_concurrent_documents: Annotated[int, Field(ge=1, le=16)]
    template_profile_context_lines: Annotated[int, Field(ge=0, le=4)] = 0
    max_initial_slots_per_request: Annotated[int, Field(ge=16, le=256)] | None = None
    output_mode: Literal["native"]
    max_successful_model_responses_per_document: Annotated[int, Field(ge=1, le=4)]
    max_provider_route_rounds: Annotated[int, Field(ge=1, le=4)]
    retry_initial_delay_seconds: Annotated[float, Field(ge=0, le=60)]
    retry_delay_multiplier: Annotated[float, Field(ge=1, le=4)]
    retry_max_delay_seconds: Annotated[float, Field(ge=0, le=300)]
    retry_jitter_seconds: Annotated[float, Field(ge=0, le=10)]
    deterministic_auxiliary_rendering: Literal[True]
    require_every_output_slot: Literal[True]
    require_full_document_regression_oracle: Literal[True]
    require_no_retained_changed_source_values: Literal[True]
    require_no_retained_source_auxiliary_values: Literal[True]
    persist_every_prompt_response_and_receipt: Literal[True]
    enable_case_checkpoints: Literal[True]
    publish_training_records: Literal[False]

    @model_validator(mode="after")
    def retry_schedule_is_consistent(self) -> RawTextInventoryBatchWorkflowConfig:
        if self.max_concurrent_documents > self.documents:
            raise ValueError("inventory batch concurrency exceeds the document count")
        if self.retry_max_delay_seconds < self.retry_initial_delay_seconds:
            raise ValueError("inventory batch retry maximum is below its initial delay")
        return self


class SynthesisRawTextInventoryBatchConfig(_StrictModel):
    """Inventory-complete bounded-response evaluation over all pinned targets."""

    schema_version: Literal[2]
    task: Literal["bill_of_lading_synthetic_raw_text_inventory_batch_v2"]
    environment_file: NonEmptyString
    run: SynthesisRunConfig
    base_batch_config: PinnedFileConfig
    regression_oracle: PinnedFileConfig
    template_mutation_profile: PinnedFileConfig | None = None
    reference_runs: RawTextInventoryReferenceRunsConfig
    selection: Literal[
        "all_pinned_targets_in_order",
        "explicit_pinned_document_ids",
    ]
    cases: tuple[RawTextRewriteCaseConfig, ...] = ()
    prompt: PinnedFileConfig
    provider: RawTextRewriteProviderConfig
    workflow: RawTextInventoryBatchWorkflowConfig

    @field_validator("environment_file")
    @classmethod
    def safe_environment_file(cls, value: str) -> str:
        return _safe_path(value)

    @field_validator("cases", mode="before")
    @classmethod
    def freeze_cases(cls, value: Any) -> Any:
        return tuple(value) if isinstance(value, list) else value

    @model_validator(mode="after")
    def batch_is_bounded_and_consistent(self) -> SynthesisRawTextInventoryBatchConfig:
        if self.selection == "all_pinned_targets_in_order" and self.cases:
            raise ValueError("all-target inventory selection cannot also define explicit cases")
        if self.selection == "explicit_pinned_document_ids":
            if not self.cases:
                raise ValueError("explicit inventory selection requires pinned document cases")
            document_ids = tuple(row.document_id for row in self.cases)
            if len(document_ids) != len(set(document_ids)):
                raise ValueError("explicit inventory selection document IDs must be unique")
            if len(document_ids) != self.workflow.documents:
                raise ValueError(
                    "explicit inventory selection count differs from workflow.documents"
                )
        if self.provider.max_output_tokens > 8192:
            raise ValueError("inventory batch output allowance exceeds its bounded contract")
        if self.provider.kind != "openrouter":
            raise ValueError("inventory batch currently requires receipted OpenRouter routing")
        if self.provider.transport_max_retries != 0:
            raise ValueError(
                "inventory batch transport retries must be application-visible (set provider "
                "transport_max_retries to zero)"
            )
        if not self.provider.allow_fallbacks or not self.provider.provider_order:
            raise ValueError("inventory batch requires an explicit observable fallback order")
        if len(self.provider.provider_order) < 2:
            raise ValueError("inventory batch requires at least two provider routes")
        if self.template_mutation_profile is None:
            if self.workflow.template_profile_context_lines != 0:
                raise ValueError(
                    "inventory batch cannot configure template context without a mutation profile"
                )
        elif self.workflow.template_profile_context_lines == 0:
            raise ValueError(
                "inventory batch mutation profiles require read-only neighboring context"
            )
        return self


class RawTextCertificationWorkflowConfig(_StrictModel):
    """Bounded read-only semantic audits over one immutable rendered candidate."""

    documents: Annotated[int, Field(ge=1, le=50)]
    max_concurrent_documents: Annotated[int, Field(ge=1, le=16)]
    semantic_audit_passes: Literal[1, 2]
    confirmation_reasoning_effort: Literal["high"] | None = None
    evaluation_only: bool = False
    require_complete_dimension_coverage: bool = False
    require_unanimous_clean: bool = False
    max_provider_route_rounds: Annotated[int, Field(ge=1, le=4)]
    retry_initial_delay_seconds: Annotated[float, Field(ge=0, le=60)]
    retry_delay_multiplier: Annotated[float, Field(ge=1, le=4)]
    retry_max_delay_seconds: Annotated[float, Field(ge=0, le=300)]
    retry_jitter_seconds: Annotated[float, Field(ge=0, le=10)]
    provider_native_json_schema: Literal[True]
    require_exact_finding_evidence: Literal[True]
    require_input_candidate_immutable: Literal[True]
    require_exact_line_and_page_topology: Literal[True]
    require_prior_host_contract: Literal[True]
    persist_every_prompt_response_and_receipt: Literal[True]
    publish_training_records: Literal[False]

    @model_validator(mode="after")
    def bounds_are_consistent(self) -> RawTextCertificationWorkflowConfig:
        if self.max_concurrent_documents > self.documents:
            raise ValueError("certification concurrency exceeds the document count")
        if self.retry_max_delay_seconds < self.retry_initial_delay_seconds:
            raise ValueError("certification retry maximum is below its initial delay")
        if self.semantic_audit_passes == 1 and self.confirmation_reasoning_effort is not None:
            raise ValueError("single-pass certification cannot configure a confirmation effort")
        if self.semantic_audit_passes == 2 and self.confirmation_reasoning_effort != "high":
            raise ValueError("two-pass certification requires an explicit high confirmation")
        return self


class RawTextCertificationInvariantInputsConfig(_StrictModel):
    """Pinned, read-only facts used by the contract-v3 deterministic audit."""

    iso3166_snapshot: PinnedFileConfig
    geonames_registry: CommittedArtifactDirectoryConfig
    geonames_registry_receipt_sha256: Annotated[str, StringConstraints(pattern=r"^[0-9a-f]{64}$")]
    phonenumberslite_version: NonEmptyString
    package_registry: PinnedFileConfig
    package_registry_entries: Annotated[int, Field(gt=0)]
    transport_capacity: TransportCapacityConfig


class SynthesisRawTextCertificationConfig(_StrictModel):
    """Fail-closed read-only semantic audit over a committed raw-text candidate run."""

    schema_version: Literal[2, 3]
    task: Literal[
        "bill_of_lading_synthetic_raw_text_certification_v2",
        "bill_of_lading_synthetic_raw_text_certification_v3",
    ]
    # Committed v2 certification artifacts predate the explicit audit-contract marker.  Treat
    # that immutable on-disk shape as contract 1; every newly hardened configuration writes 2.
    audit_contract_version: Literal[1, 2, 3] = 1
    environment_file: NonEmptyString
    run: SynthesisRunConfig
    input_run: CommittedArtifactDirectoryConfig
    benchmark_plan: CommittedArtifactDirectoryConfig | None = None
    invariant_inputs: RawTextCertificationInvariantInputsConfig | None = None
    input_case_contract_filename: Literal["contract.json", "source-contract.json"] = "contract.json"
    case_ids: tuple[Annotated[str, StringConstraints(pattern=r"^doc_[0-9a-f]{64}$")], ...] = Field(
        min_length=1, max_length=50
    )
    prompt: PinnedFileConfig
    provider: RawTextRewriteProviderConfig
    workflow: RawTextCertificationWorkflowConfig

    @field_validator("environment_file")
    @classmethod
    def safe_environment_file(cls, value: str) -> str:
        return _safe_path(value)

    @field_validator("case_ids", mode="before")
    @classmethod
    def freeze_case_ids(cls, value: Any) -> Any:
        return tuple(value) if isinstance(value, list) else value

    @model_validator(mode="after")
    def certification_contract_is_consistent(self) -> SynthesisRawTextCertificationConfig:
        if len(self.case_ids) != len(set(self.case_ids)):
            raise ValueError("certification case IDs must be unique")
        if len(self.case_ids) != self.workflow.documents:
            raise ValueError("certification case count differs from workflow.documents")
        if self.provider.transport_max_retries != 0:
            raise ValueError(
                "certification transport retries must be application-visible (set provider "
                "transport_max_retries to zero)"
            )
        if self.audit_contract_version in {1, 2}:
            if (
                self.schema_version != 2
                or self.task != "bill_of_lading_synthetic_raw_text_certification_v2"
                or self.invariant_inputs is not None
            ):
                raise ValueError(
                    "certification contracts v1/v2 require their original schema and no "
                    "contract-v3 invariant inputs"
                )
        elif (
            self.schema_version != 3
            or self.task != "bill_of_lading_synthetic_raw_text_certification_v3"
            or self.invariant_inputs is None
        ):
            raise ValueError(
                "certification contract v3 requires schema/task v3 and pinned invariant inputs"
            )
        if self.audit_contract_version == 1:
            if self.provider.max_output_tokens > 8192:
                raise ValueError("legacy certification output allowance exceeds its contract")
            if self.benchmark_plan is not None:
                raise ValueError("legacy certification cannot claim a benchmark plan")
            if self.provider.kind == "openrouter" and (
                not self.provider.allow_fallbacks
                or not self.provider.provider_order
                or len(self.provider.provider_order) < 2
            ):
                raise ValueError(
                    "legacy OpenRouter certification requires at least two fallback routes"
                )
            if (
                self.workflow.semantic_audit_passes != 1
                or self.workflow.evaluation_only
                or self.workflow.require_complete_dimension_coverage
                or self.workflow.require_unanimous_clean
            ):
                raise ValueError(
                    "legacy certification requires its original single-pass finding contract"
                )
        else:
            if self.provider.kind != "openrouter":
                raise ValueError(
                    "certification contracts v2/v3 currently require their proven OpenRouter "
                    "model identity contract"
                )
            if self.provider.model != "z-ai/glm-5.3-flash":
                raise ValueError(
                    "certification contracts v2/v3 require the empirically evaluated GLM model"
                )
            routed_provider_ids = self.provider.provider_order or ()
            has_routed_fallback = self.provider.allow_fallbacks and len(routed_provider_ids) >= 2
            isolated_provider_ids = self.provider.provider_only or ()
            has_isolated_evaluation_route = (
                self.workflow.evaluation_only
                and not self.provider.allow_fallbacks
                and self.provider.provider_order is None
                and self.provider.provider_sort is None
                and len(isolated_provider_ids) == 1
            )
            if not (has_routed_fallback or has_isolated_evaluation_route):
                raise ValueError(
                    "certification contracts v2/v3 require either production fallback routes or "
                    "one isolated evaluation route"
                )
            active_provider_ids = (
                routed_provider_ids if has_routed_fallback else isolated_provider_ids
            )
            if not set(active_provider_ids).issubset(
                {
                    "deepinfra/fp4",
                    "coreweave/fp8",
                    "fireworks",
                    "nextbit/fp8",
                }
            ):
                raise ValueError(
                    "certification contracts v2/v3 contain an unevaluated provider route"
                )
            if self.provider.max_output_tokens != 12288:
                raise ValueError(
                    "certification contracts v2/v3 require their measured bounded output allowance"
                )
            if not self.workflow.require_complete_dimension_coverage:
                raise ValueError(
                    "certification contracts v2/v3 require complete dimension coverage"
                )
            if self.workflow.evaluation_only:
                if (
                    self.workflow.semantic_audit_passes != 1
                    or self.workflow.require_unanimous_clean
                ):
                    raise ValueError(
                        "evaluation-only certification v2/v3 requires one non-unanimous pass"
                    )
            elif self.benchmark_plan is not None:
                raise ValueError("production certification cannot claim an evaluation plan")
            elif not (
                self.workflow.semantic_audit_passes == 2
                and self.workflow.require_unanimous_clean
                and self.provider.reasoning_effort == "low"
            ):
                raise ValueError(
                    "production certification v2/v3 requires a low/high unanimous two-pass cascade"
                )
            if self.workflow.evaluation_only and (
                self.provider.reasoning_effort not in {"low", "medium", "high"}
            ):
                raise ValueError(
                    "single-pass certification contract v2/v3 requires low, medium, or high effort"
                )
        return self


class RawTextCertifiedCorrectionWorkflowConfig(_StrictModel):
    """Bounds for one exact-evidence correction pass over certified candidates."""

    documents: Annotated[int, Field(ge=1, le=50)]
    # Default 1 only decodes immutable historical configs.  Every new correction must opt in to
    # the complete contract-v2 audit authority explicitly.
    required_audit_contract_version: Literal[1, 2, 3] = 1
    max_concurrent_documents: Annotated[int, Field(ge=1, le=16)]
    max_provider_route_rounds: Annotated[int, Field(ge=1, le=4)]
    max_successful_model_responses_per_document: Annotated[int, Field(ge=1, le=3)]
    retry_initial_delay_seconds: Annotated[float, Field(ge=0, le=60)]
    retry_delay_multiplier: Annotated[float, Field(ge=1, le=4)]
    retry_max_delay_seconds: Annotated[float, Field(ge=0, le=300)]
    retry_jitter_seconds: Annotated[float, Field(ge=0, le=10)]
    provider_native_json_schema: Literal[True]
    require_exact_finding_evidence: Literal[True]
    require_only_cited_lines_mutable: Literal[True]
    require_exact_line_and_page_topology: Literal[True]
    persist_every_prompt_response_and_receipt: Literal[True]
    publish_training_records: Literal[False]

    @model_validator(mode="after")
    def bounds_are_consistent(self) -> RawTextCertifiedCorrectionWorkflowConfig:
        if self.max_concurrent_documents > self.documents:
            raise ValueError("correction concurrency exceeds the document count")
        if self.retry_max_delay_seconds < self.retry_initial_delay_seconds:
            raise ValueError("correction retry maximum is below its initial delay")
        return self


class SynthesisRawTextCertifiedCorrectionConfig(_StrictModel):
    """Exact-line correction driven by a committed, read-only certification run."""

    schema_version: Literal[1]
    task: Literal["bill_of_lading_synthetic_raw_text_certified_correction_v1"]
    environment_file: NonEmptyString
    run: SynthesisRunConfig
    certification_run: CommittedArtifactDirectoryConfig
    case_ids: tuple[Annotated[str, StringConstraints(pattern=r"^doc_[0-9a-f]{64}$")], ...] = Field(
        min_length=1, max_length=50
    )
    prompt: PinnedFileConfig
    provider: RawTextRewriteProviderConfig
    workflow: RawTextCertifiedCorrectionWorkflowConfig

    @field_validator("environment_file")
    @classmethod
    def safe_environment_file(cls, value: str) -> str:
        return _safe_path(value)

    @field_validator("case_ids", mode="before")
    @classmethod
    def freeze_case_ids(cls, value: Any) -> Any:
        return tuple(value) if isinstance(value, list) else value

    @model_validator(mode="after")
    def correction_contract_is_consistent(self) -> SynthesisRawTextCertifiedCorrectionConfig:
        if len(self.case_ids) != len(set(self.case_ids)):
            raise ValueError("correction case IDs must be unique")
        if len(self.case_ids) != self.workflow.documents:
            raise ValueError("correction case count differs from workflow.documents")
        if self.provider.transport_max_retries != 0:
            raise ValueError(
                "correction transport retries must be application-visible (set provider "
                "transport_max_retries to zero)"
            )
        if self.provider.kind == "openrouter":
            if not self.provider.allow_fallbacks or not self.provider.provider_order:
                raise ValueError("OpenRouter correction requires an explicit fallback order")
            if len(self.provider.provider_order) < 2:
                raise ValueError("OpenRouter correction requires at least two provider routes")
        if self.provider.max_output_tokens > 8192:
            raise ValueError("correction output allowance exceeds its bounded contract")
        return self


class RawTextCertifiedPublicationSourceConfig(_StrictModel):
    """One immutable certification run and the exact certified subset selected from it."""

    run: CommittedArtifactDirectoryConfig
    certified_documents: Annotated[int, Field(ge=1, le=100_000)]
    certified_document_ids_sha256: Annotated[str, StringConstraints(pattern=r"^[0-9a-f]{64}$")]


class RawTextCertifiedPublicationWorkflowConfig(_StrictModel):
    documents: Annotated[int, Field(ge=1, le=100_000)]
    # Default 1 only decodes immutable historical configs; active publication requires an
    # explicitly configured contract-v2 source policy.
    required_audit_contract_version: Literal[1, 2, 3] = 1
    require_every_case_certified: Literal[True]
    require_unique_source_documents: Literal[True]
    require_unique_scenarios: Literal[True]
    replay_current_host_audit: Literal[True]
    require_v5_canonical_target: Literal[True]
    publish_training_records: Literal[True]


class SynthesisRawTextCertifiedPublicationConfig(_StrictModel):
    """Publish only an explicitly pinned union of independently certified OCR pairs."""

    schema_version: Literal[1]
    task: Literal["bill_of_lading_synthetic_raw_text_certified_publication_v1"]
    run: SynthesisRunConfig
    certification_sources: tuple[RawTextCertifiedPublicationSourceConfig, ...] = Field(min_length=1)
    workflow: RawTextCertifiedPublicationWorkflowConfig

    @field_validator("certification_sources", mode="before")
    @classmethod
    def freeze_sources(cls, value: Any) -> Any:
        return tuple(value) if isinstance(value, list) else value

    @model_validator(mode="after")
    def publication_scope_is_exact(self) -> SynthesisRawTextCertifiedPublicationConfig:
        paths = tuple(row.run.path for row in self.certification_sources)
        if len(paths) != len(set(paths)):
            raise ValueError("certified publication sources must be unique")
        selected = sum(row.certified_documents for row in self.certification_sources)
        if selected != self.workflow.documents:
            raise ValueError("certified source counts differ from the publication document count")
        return self


class RawTextPipelineCertificationConfig(_StrictModel):
    """Template and bounds for dynamically sharded independent certification runs."""

    environment_file: NonEmptyString
    # See SynthesisRawTextCertificationConfig.audit_contract_version.  The default is needed to
    # replay immutable pipeline ancestors, while generated current children serialize it.
    audit_contract_version: Literal[1, 2, 3] = 1
    invariant_inputs: RawTextCertificationInvariantInputsConfig | None = None
    shard_size: Annotated[int, Field(ge=1, le=50)]
    max_attempts_per_candidate: Annotated[int, Field(ge=1, le=4)]
    prompt: PinnedFileConfig
    provider: RawTextRewriteProviderConfig
    workflow: RawTextCertificationWorkflowConfig

    @field_validator("environment_file")
    @classmethod
    def safe_environment_file(cls, value: str) -> str:
        return _safe_path(value)

    @model_validator(mode="after")
    def template_matches_shard_capacity(self) -> RawTextPipelineCertificationConfig:
        if self.workflow.documents != self.shard_size:
            raise ValueError("pipeline certification template documents must equal shard_size")
        if self.audit_contract_version in {1, 2} and self.invariant_inputs is not None:
            raise ValueError("pipeline certification v1/v2 cannot configure v3 invariant inputs")
        if self.audit_contract_version == 3 and self.invariant_inputs is None:
            raise ValueError("pipeline certification v3 requires pinned invariant inputs")
        if self.audit_contract_version == 1:
            if (
                self.workflow.semantic_audit_passes != 1
                or self.workflow.evaluation_only
                or self.workflow.require_complete_dimension_coverage
                or self.workflow.require_unanimous_clean
            ):
                raise ValueError(
                    "legacy pipeline certification requires its original finding contract"
                )
        else:
            if self.provider.kind != "openrouter":
                raise ValueError("pipeline certification v2/v3 requires OpenRouter")
            if self.provider.model != "z-ai/glm-5.3-flash":
                raise ValueError("pipeline certification v2/v3 requires the evaluated GLM model")
            if (
                not self.provider.allow_fallbacks
                or self.provider.provider_order is None
                or len(self.provider.provider_order) < 2
            ):
                raise ValueError(
                    "pipeline certification v2/v3 requires at least two fallback routes"
                )
            if not set(self.provider.provider_order).issubset(
                {
                    "deepinfra/fp4",
                    "coreweave/fp8",
                    "fireworks",
                    "nextbit/fp8",
                }
            ):
                raise ValueError("pipeline certification v2/v3 has an unevaluated provider route")
            if self.provider.max_output_tokens != 12288:
                raise ValueError(
                    "pipeline certification contract v2/v3 requires its measured bounded "
                    "output allowance"
                )
            if not (
                self.workflow.require_complete_dimension_coverage
                and self.workflow.require_unanimous_clean
                and not self.workflow.evaluation_only
            ):
                raise ValueError(
                    "pipeline certification contract v2/v3 requires complete coverage and "
                    "unanimous clean"
                )
            if not (
                self.workflow.semantic_audit_passes == 2 and self.provider.reasoning_effort == "low"
            ):
                raise ValueError(
                    "pipeline certification v2/v3 requires a native low/high two-pass cascade"
                )
        return self


class RawTextPipelineCorrectionConfig(_StrictModel):
    """Template and bounds for exact-evidence correction/recertification cycles."""

    environment_file: NonEmptyString
    shard_size: Annotated[int, Field(ge=1, le=50)]
    max_rounds_per_document: Annotated[int, Field(ge=1, le=6)]
    max_attempts_per_round: Annotated[int, Field(ge=1, le=4)] = 1
    prompt: PinnedFileConfig
    provider: RawTextRewriteProviderConfig
    workflow: RawTextCertifiedCorrectionWorkflowConfig

    @field_validator("environment_file")
    @classmethod
    def safe_environment_file(cls, value: str) -> str:
        return _safe_path(value)

    @model_validator(mode="after")
    def template_matches_shard_capacity(self) -> RawTextPipelineCorrectionConfig:
        if self.workflow.documents != self.shard_size:
            raise ValueError("pipeline correction template documents must equal shard_size")
        return self


class RawTextPipelineWorkflowConfig(_StrictModel):
    """Fail-closed bounds for one complete rendered/certified publication cohort."""

    documents: Annotated[int, Field(ge=1, le=100_000)]
    max_inventory_rounds: Annotated[int, Field(ge=1, le=4)]
    require_every_inventory_case_training_ready: Literal[True]
    require_every_case_independently_certified: Literal[True]
    require_complete_cohort_publication: Literal[True]


class SynthesisRawTextPipelineConfig(_StrictModel):
    """One configurable inventory -> certify -> correct -> publish production workflow."""

    # Schema v1 is retained solely to decode and replay the immutable contract-v1/v2
    # pipeline lineage.  Contract-v3 production runs have a distinct top-level schema so
    # selecting the old semantic authority cannot be mistaken for the current launch path.
    schema_version: Literal[1, 2]
    task: Literal[
        "bill_of_lading_synthetic_raw_text_pipeline_v1",
        "bill_of_lading_synthetic_raw_text_pipeline_v2",
    ]
    run: SynthesisRunConfig
    inventory_config: PinnedFileConfig
    inventory_resume_run: CommittedArtifactDirectoryConfig | None = None
    pipeline_resume_run: CommittedArtifactDirectoryConfig | None = None
    certification: RawTextPipelineCertificationConfig
    correction: RawTextPipelineCorrectionConfig
    publication: RawTextCertifiedPublicationWorkflowConfig
    workflow: RawTextPipelineWorkflowConfig

    @model_validator(mode="after")
    def pipeline_contract_is_consistent(self) -> SynthesisRawTextPipelineConfig:
        if self.certification.audit_contract_version == 3:
            if (
                self.schema_version != 2
                or self.task != "bill_of_lading_synthetic_raw_text_pipeline_v2"
            ):
                raise ValueError(
                    "contract-v3 pipeline certification requires top-level schema/task v2"
                )
        elif (
            self.schema_version != 1 or self.task != "bill_of_lading_synthetic_raw_text_pipeline_v1"
        ):
            raise ValueError(
                "legacy pipeline certification requires its original top-level schema/task v1"
            )
        if self.publication.documents != self.workflow.documents:
            raise ValueError("pipeline publication count differs from workflow.documents")
        if self.inventory_resume_run is not None and self.pipeline_resume_run is not None:
            raise ValueError("inventory_resume_run and pipeline_resume_run are mutually exclusive")
        if self.certification.audit_contract_version in {2, 3} and (
            self.correction.workflow.required_audit_contract_version
            != self.certification.audit_contract_version
            or self.publication.required_audit_contract_version
            != self.certification.audit_contract_version
        ):
            raise ValueError(
                "pipeline certification requires matching correction/publication audit contracts"
            )
        if self.certification.audit_contract_version == 3 and (
            self.correction.max_rounds_per_document != 1
            or self.correction.max_attempts_per_round != 1
            or self.correction.workflow.max_provider_route_rounds != 1
            or self.correction.workflow.max_successful_model_responses_per_document != 1
        ):
            raise ValueError(
                "contract-v3 deterministic correction requires exactly one round and one "
                "attempt; provider retry/response bounds are inactive and must be one"
            )
        # Generated child run IDs append bounded stage/round/shard suffixes.
        if len(self.run.run_id) > 96:
            raise ValueError("pipeline run_id is too long for deterministic child run IDs")
        return self


class LinguisticProbeAnalysisInputConfig(_StrictModel):
    run: CommittedArtifactDirectoryConfig
    results: DatasetFileConfig
    summary: PinnedFileConfig


class SynthesisLinguisticProbeAnalysisConfig(_StrictModel):
    """Pinned comparison and EDA over party and cargo linguistic probes."""

    schema_version: Literal[1]
    task: Literal["bill_of_lading_linguistic_probe_eda_v1"]
    run: SynthesisRunConfig
    party: LinguisticProbeAnalysisInputConfig
    cargo: LinguisticProbeAnalysisInputConfig
    cargo_contract_probe: LinguisticProbeAnalysisInputConfig

    @model_validator(mode="after")
    def expected_probe_sizes_are_pinned(self) -> SynthesisLinguisticProbeAnalysisConfig:
        if self.party.results.records != 50 or self.cargo.results.records != 50:
            raise ValueError("linguistic EDA requires exactly fifty party and fifty cargo cases")
        if self.cargo_contract_probe.results.records != 1:
            raise ValueError("linguistic EDA requires exactly one targeted contract probe")
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


def load_synthesis_cargo_language_probe_config(
    path: Path,
) -> SynthesisCargoLanguageProbeConfig:
    try:
        value = yaml.load(path.read_bytes(), Loader=_UniqueKeySafeLoader)
    except UnicodeDecodeError as error:
        raise ValueError("synthesis configuration is not valid UTF-8") from error
    if not isinstance(value, dict):
        raise ValueError("synthesis configuration root must be a mapping")
    return SynthesisCargoLanguageProbeConfig.model_validate(value, strict=True)


def load_synthesis_linguistic_completion_config(
    path: Path,
) -> SynthesisLinguisticCompletionConfig:
    try:
        value = yaml.load(path.read_bytes(), Loader=_UniqueKeySafeLoader)
    except UnicodeDecodeError as error:
        raise ValueError("synthesis configuration is not valid UTF-8") from error
    if not isinstance(value, dict):
        raise ValueError("synthesis configuration root must be a mapping")
    return SynthesisLinguisticCompletionConfig.model_validate(value, strict=True)


def load_synthesis_raw_text_rewrite_probe_config(
    path: Path,
) -> SynthesisRawTextRewriteProbeConfig:
    try:
        value = yaml.load(path.read_bytes(), Loader=_UniqueKeySafeLoader)
    except UnicodeDecodeError as error:
        raise ValueError("synthesis configuration is not valid UTF-8") from error
    if not isinstance(value, dict):
        raise ValueError("synthesis configuration root must be a mapping")
    return SynthesisRawTextRewriteProbeConfig.model_validate(value, strict=True)


def load_synthesis_raw_text_rewrite_cycle_probe_config(
    path: Path,
) -> SynthesisRawTextRewriteCycleProbeConfig:
    try:
        value = yaml.load(path.read_bytes(), Loader=_UniqueKeySafeLoader)
    except UnicodeDecodeError as error:
        raise ValueError("synthesis configuration is not valid UTF-8") from error
    if not isinstance(value, dict):
        raise ValueError("synthesis configuration root must be a mapping")
    return SynthesisRawTextRewriteCycleProbeConfig.model_validate(value, strict=True)


def load_synthesis_raw_text_hybrid_probe_config(
    path: Path,
) -> SynthesisRawTextHybridProbeConfig:
    try:
        value = yaml.load(path.read_bytes(), Loader=_UniqueKeySafeLoader)
    except UnicodeDecodeError as error:
        raise ValueError("synthesis configuration is not valid UTF-8") from error
    if not isinstance(value, dict):
        raise ValueError("synthesis configuration root must be a mapping")
    return SynthesisRawTextHybridProbeConfig.model_validate(value, strict=True)


def load_synthesis_raw_text_hybrid_batch_config(
    path: Path,
) -> SynthesisRawTextHybridBatchConfig:
    try:
        value = yaml.load(path.read_bytes(), Loader=_UniqueKeySafeLoader)
    except UnicodeDecodeError as error:
        raise ValueError("synthesis configuration is not valid UTF-8") from error
    if not isinstance(value, dict):
        raise ValueError("synthesis configuration root must be a mapping")
    return SynthesisRawTextHybridBatchConfig.model_validate(value, strict=True)


def load_synthesis_raw_text_inventory_probe_config(
    path: Path,
) -> SynthesisRawTextInventoryProbeConfig:
    try:
        value = yaml.load(path.read_bytes(), Loader=_UniqueKeySafeLoader)
    except UnicodeDecodeError as error:
        raise ValueError("synthesis configuration is not valid UTF-8") from error
    if not isinstance(value, dict):
        raise ValueError("synthesis configuration root must be a mapping")
    return SynthesisRawTextInventoryProbeConfig.model_validate(value, strict=True)


def load_synthesis_raw_text_inventory_batch_config(
    path: Path,
) -> SynthesisRawTextInventoryBatchConfig:
    try:
        value = yaml.load(path.read_bytes(), Loader=_UniqueKeySafeLoader)
    except UnicodeDecodeError as error:
        raise ValueError("synthesis configuration is not valid UTF-8") from error
    if not isinstance(value, dict):
        raise ValueError("synthesis configuration root must be a mapping")
    return SynthesisRawTextInventoryBatchConfig.model_validate(value, strict=True)


def load_synthesis_raw_text_certification_config(
    path: Path,
) -> SynthesisRawTextCertificationConfig:
    try:
        value = yaml.load(path.read_bytes(), Loader=_UniqueKeySafeLoader)
    except UnicodeDecodeError as error:
        raise ValueError("synthesis configuration is not valid UTF-8") from error
    if not isinstance(value, dict):
        raise ValueError("synthesis configuration root must be a mapping")
    return SynthesisRawTextCertificationConfig.model_validate(value, strict=True)


def load_synthesis_raw_text_certified_correction_config(
    path: Path,
) -> SynthesisRawTextCertifiedCorrectionConfig:
    try:
        value = yaml.load(path.read_bytes(), Loader=_UniqueKeySafeLoader)
    except UnicodeDecodeError as error:
        raise ValueError("synthesis configuration is not valid UTF-8") from error
    if not isinstance(value, dict):
        raise ValueError("synthesis configuration root must be a mapping")
    return SynthesisRawTextCertifiedCorrectionConfig.model_validate(value, strict=True)


def load_synthesis_raw_text_certified_publication_config(
    path: Path,
) -> SynthesisRawTextCertifiedPublicationConfig:
    try:
        value = yaml.load(path.read_bytes(), Loader=_UniqueKeySafeLoader)
    except UnicodeDecodeError as error:
        raise ValueError("synthesis configuration is not valid UTF-8") from error
    if not isinstance(value, dict):
        raise ValueError("synthesis configuration root must be a mapping")
    return SynthesisRawTextCertifiedPublicationConfig.model_validate(value, strict=True)


def load_synthesis_raw_text_pipeline_config(path: Path) -> SynthesisRawTextPipelineConfig:
    try:
        value = yaml.load(path.read_bytes(), Loader=_UniqueKeySafeLoader)
    except UnicodeDecodeError as error:
        raise ValueError("synthesis configuration is not valid UTF-8") from error
    if not isinstance(value, dict):
        raise ValueError("synthesis configuration root must be a mapping")
    return SynthesisRawTextPipelineConfig.model_validate(value, strict=True)


def load_synthesis_package_compatibility_catalog_config(
    path: Path,
) -> SynthesisPackageCompatibilityCatalogConfig:
    try:
        value = yaml.load(path.read_bytes(), Loader=_UniqueKeySafeLoader)
    except UnicodeDecodeError as error:
        raise ValueError("synthesis configuration is not valid UTF-8") from error
    if not isinstance(value, dict):
        raise ValueError("synthesis configuration root must be a mapping")
    return SynthesisPackageCompatibilityCatalogConfig.model_validate(value, strict=True)


def load_synthesis_linguistic_probe_analysis_config(
    path: Path,
) -> SynthesisLinguisticProbeAnalysisConfig:
    try:
        value = yaml.load(path.read_bytes(), Loader=_UniqueKeySafeLoader)
    except UnicodeDecodeError as error:
        raise ValueError("synthesis configuration is not valid UTF-8") from error
    if not isinstance(value, dict):
        raise ValueError("synthesis configuration root must be a mapping")
    return SynthesisLinguisticProbeAnalysisConfig.model_validate(value, strict=True)
