"""Audited profile-routed cargo proposals for structured B/L synthesis."""

from __future__ import annotations

import math
from collections import Counter, defaultdict
from collections.abc import Mapping, Sequence
from copy import deepcopy
from dataclasses import dataclass
from decimal import ROUND_HALF_UP, Decimal
from statistics import fmean, stdev
from typing import Any, cast

from document_ocr.hashing import canonical_json_bytes, sha256_bytes
from document_ocr.synthesis.config import SynthesisStructuredBaselineConfig
from document_ocr.synthesis.generators import (
    largest_remainder_allocation,
    reconcile_allocation_group,
    validate_mass_order,
)
from document_ocr.synthesis.modeling_views import (
    CanonicalMassProjection,
    CargoGroupNumericObservation,
)
from document_ocr.synthesis.package_registry import normalize_printed_surface
from document_ocr.synthesis.profile_sdv import (
    MeasureProfile,
    NumericSourceRow,
    ProfileBenchmarkSettings,
    ProfileNumericProposal,
    ProfileNumericRow,
    ProfileRouteRequest,
    ProfileRoutingSettings,
    ProfileRunResult,
    QualityAuditSettings,
    RouteDecision,
    RouteSupportThreshold,
    SourceQualityAudit,
    audit_source_rows,
    benchmark_and_sample_profile,
    route_profile_requests,
)
from document_ocr.synthesis.sdv_evaluation import MetricDetail, MetricReport
from document_ocr.synthesis.structured_models import CargoGroupNumericProposal
from document_ocr.synthesis.structured_semantics import derive_scaled_group_quantities
from document_ocr.synthesis.transport_capacity import (
    TransportCapacityLimits,
    capacity_limits,
    group_capacity_budgets,
)


class StructuredProfileError(RuntimeError):
    """A routed statistical proposal cannot meet the synthesis contract."""


class ProfileCandidateAssignmentError(StructuredProfileError):
    """A source-equivalence class cannot receive a distinct contextual sample."""

    def __init__(
        self,
        message: str,
        *,
        failed_equivalence_id: str,
        class_option_counts: Mapping[str, int],
    ) -> None:
        self.failed_equivalence_id = failed_equivalence_id
        self.class_option_counts = dict(class_option_counts)
        super().__init__(message)


class ProfileQualityRejection(StructuredProfileError):
    """A completed benchmark failed one or more predeclared acceptance gates."""

    def __init__(self, receipt: Mapping[str, Any]) -> None:
        self.receipt = deepcopy(dict(receipt))
        super().__init__(
            "profile acceptance gates failed: "
            f"failures={self.receipt['failures']}, "
            f"selected={self.receipt['selected_candidate']}, "
            f"paired_seed_runs={self.receipt['paired_runs']}, "
            f"independent_folds={self.receipt['independent_folds']}, "
            f"fold_receipts={self.receipt['folds']}, "
            f"gate_receipts={self.receipt['gates']}, "
            f"candidate_means={self.receipt['candidate_quality_means']}, "
            f"property_receipts={self.receipt['properties']}, "
            f"detail_receipts={self.receipt['details']}, "
            f"raw_proposals={self.receipt['raw_proposals']}, "
            f"rejected_proposals={self.receipt['rejected_proposals']}"
        )


@dataclass(frozen=True, slots=True)
class GroupProfileContext:
    """One cargo-group source row and its explicit profile-routing request."""

    observation: CargoGroupNumericObservation
    source_row: NumericSourceRow
    request: ProfileRouteRequest

    @property
    def request_id(self) -> str:
        return self.request.request_id

    @property
    def document_id(self) -> str:
        return self.source_row.document_id

    @property
    def group_id(self) -> str:
        return self.observation.cargo_group_id


@dataclass(frozen=True, slots=True)
class PreparedProfileSource:
    """One train-only source audit and routing decision for every modeled group."""

    contexts: tuple[GroupProfileContext, ...]
    audit: SourceQualityAudit
    decisions_by_request: Mapping[str, RouteDecision]
    clean_rows_by_id: Mapping[str, ProfileNumericRow]


@dataclass(frozen=True, slots=True)
class ContextualSupportEnvelope:
    """Package-aware support region derived only from isolated training rows.

    The statistical cohort may need to pool at role level to obtain enough rows
    for grouped evaluation.  This envelope prevents a pooled proposal from
    crossing into a package identity's unsupported numeric regime.
    """

    tier: str
    row_ids: tuple[str, ...]
    template_ids: tuple[str, ...]
    columns: tuple[str, ...]
    bounds: Mapping[str, tuple[float, float]]
    normalized_vectors: tuple[tuple[float, ...], ...]
    log_minima: tuple[float, ...]
    log_spans: tuple[float, ...]
    maximum_nearest_neighbor_distance: float
    support_sha256: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "tier": self.tier,
            "rows": len(self.row_ids),
            "templates": len(self.template_ids),
            "row_ids_sha256": sha256_bytes(canonical_json_bytes(self.row_ids)),
            "template_ids_sha256": sha256_bytes(canonical_json_bytes(self.template_ids)),
            "columns": list(self.columns),
            "bounds": {
                name: {"minimum": limits[0], "maximum": limits[1]}
                for name, limits in self.bounds.items()
            },
            "maximum_nearest_neighbor_distance": (self.maximum_nearest_neighbor_distance),
            "support_sha256": self.support_sha256,
        }


@dataclass(frozen=True, slots=True)
class AssignedGroupProposal:
    """One candidate assigned to one source group after deterministic preflight."""

    request_id: str
    cohort_id: str
    sampled_row_index: int
    source_numeric_equivalence_id: str
    route_tier: str
    contextual_support_tier: str
    contextual_support_rows: int
    contextual_support_templates: int
    contextual_support_sha256: str
    contextual_distance: float
    contextual_maximum_distance: float
    proposal: CargoGroupNumericProposal
    sampled_values: Mapping[str, Any]


@dataclass(frozen=True, slots=True)
class ProfileCohortResult:
    """Benchmark result plus exact request-to-proposal assignment for one route."""

    cohort_id: str
    request_ids: tuple[str, ...]
    sample_seed: int
    run: ProfileRunResult
    quality_acceptance: Mapping[str, Any]
    assignments: tuple[AssignedGroupProposal, ...]


ProfileCohortKey = tuple[str, str, tuple[str, ...]]


@dataclass(frozen=True, slots=True)
class ProfileCohortRejection:
    """A receipted route cohort that is unavailable for structured generation."""

    cohort_key: ProfileCohortKey
    cohort_id: str
    request_ids: tuple[str, ...]
    sample_seed: int
    run: ProfileRunResult
    quality_acceptance: Mapping[str, Any]


class ProfileCohortBatchRejected(StructuredProfileError):
    """One or more selected cohorts failed statistical acceptance."""

    def __init__(
        self,
        rejections: Sequence[ProfileCohortRejection],
        accepted: Sequence[ProfileCohortResult],
    ) -> None:
        if not rejections:
            raise ValueError("profile cohort rejection batch must not be empty")
        self.rejections = tuple(rejections)
        self.accepted = tuple(accepted)
        super().__init__(
            "selected profile cohorts failed statistical acceptance: "
            + ", ".join(
                f"{row.cohort_id}({row.quality_acceptance['failures']})" for row in self.rejections
            )
        )


@dataclass(frozen=True, slots=True)
class ProfileContextRejection:
    """A selected source whose sampled cohort cannot realize a plausible value."""

    cohort_id: str
    request_id: str
    document_id: str
    exact_identity: str
    support: Mapping[str, Any]
    sampled_rows: int
    unique_sampled_rows: int
    rejection_counts: Mapping[str, int]

    def to_dict(self) -> dict[str, Any]:
        return {
            "cohort_id": self.cohort_id,
            "request_id": self.request_id,
            "document_id": self.document_id,
            "exact_identity": self.exact_identity,
            "support": deepcopy(dict(self.support)),
            "sampled_rows": self.sampled_rows,
            "unique_sampled_rows": self.unique_sampled_rows,
            "rejection_counts": dict(sorted(self.rejection_counts.items())),
        }


class ProfileContextBatchRejected(StructuredProfileError):
    """Selected documents failed package-aware proposal realization."""

    def __init__(
        self,
        rejections: Sequence[ProfileContextRejection],
        accepted: Sequence[ProfileCohortResult] = (),
    ) -> None:
        if not rejections:
            raise ValueError("profile context rejection batch must not be empty")
        self.rejections = tuple(rejections)
        self.accepted = tuple(accepted)
        super().__init__(
            "selected profile contexts failed package-aware plausibility: "
            + ", ".join(
                f"{row.request_id}({dict(row.rejection_counts)})" for row in self.rejections
            )
        )


@dataclass(frozen=True, slots=True)
class _PairedQualityRun:
    """One empirical/model comparison sharing a validation fold and sample seed."""

    fold_index: int
    seed: int
    empirical: MetricReport
    selected: MetricReport


@dataclass(frozen=True, slots=True)
class _QualityDetailScore:
    """A contract-identified SDMetrics detail score."""

    identity: str
    property_name: str
    metric: str
    columns: tuple[str, ...]
    score: float
    real_correlation: float | None


_MASS_FACTORS = {
    "kilogram": Decimal("1"),
    "metric_tonne": Decimal("1000"),
    "pound": Decimal("0.45359237"),
}
_VOLUME_FACTORS = {"cubic_metre": Decimal("1")}

# These values describe missing package semantics, not coherent package classes.
# Pooling printed surfaces such as BALES, ITEMS, and CARBOUY under either label
# would make the family-level plausibility envelope actively misleading.
_NON_SEMANTIC_PACKAGE_FAMILIES = frozenset({"UNREGISTERED_PRINTED_PACKAGE", "UNTYPED_PACKAGE"})


def _identity_and_family(observation: CargoGroupNumericObservation) -> tuple[str, str]:
    features = observation.features
    if features.driver_type_category is not None:
        identity = features.driver_type_category
        tokens = identity.removeprefix("PACKAGE_").split("_")
        if not tokens or not tokens[0]:
            raise ValueError("package category cannot define a semantic family")
        return identity, f"PACKAGE_{tokens[0]}"
    if features.driver_printed_surface is not None:
        return (
            "SURFACE:" + normalize_printed_surface(features.driver_printed_surface),
            "UNREGISTERED_PRINTED_PACKAGE",
        )
    return "UNTYPED_PACKAGE", "UNTYPED_PACKAGE"


def _measure_profile(observation: CargoGroupNumericObservation) -> MeasureProfile:
    features = observation.features
    return MeasureProfile(
        gross=features.gross_weight_kg is not None,
        net=features.net_weight_kg is not None,
        volume=features.volume_value is not None,
    )


def build_group_profile_contexts(
    observations: Sequence[CargoGroupNumericObservation],
    *,
    template_by_document: Mapping[str, str],
    partition: str,
) -> tuple[GroupProfileContext, ...]:
    """Convert unique cargo groups into canonical train-only profile rows."""

    output: list[GroupProfileContext] = []
    seen: set[str] = set()
    for observation in observations:
        if observation.driver_package_id is None:
            continue
        document_id = observation.projection.source_document_id
        request_id = observation.projection.view_row_key
        if request_id in seen:
            raise ValueError(f"duplicate cargo profile request: {request_id}")
        seen.add(request_id)
        identity, family = _identity_and_family(observation)
        features = observation.features
        if features.driver_package_role is None or features.driver_quantity is None:
            raise ValueError("cargo-group driver metadata is incomplete")
        source_row = NumericSourceRow(
            row_id=request_id,
            document_id=document_id,
            template_id=template_by_document[document_id],
            partition=partition,
            exact_identity=identity,
            semantic_family=family,
            package_role=features.driver_package_role,
            quantity=features.driver_quantity,
            gross_value=features.gross_weight_kg,
            gross_unit="KG" if features.gross_weight_kg is not None else None,
            net_value=features.net_weight_kg,
            net_unit="KG" if features.net_weight_kg is not None else None,
            volume_value=features.volume_value,
            volume_unit=features.volume_unit,
        )
        request = ProfileRouteRequest(
            request_id=request_id,
            exact_identity=identity,
            semantic_family=family,
            package_role=features.driver_package_role,
            profile=_measure_profile(observation),
        )
        output.append(GroupProfileContext(observation, source_row, request))
    return tuple(sorted(output, key=lambda row: row.request_id))


def _quality_settings(config: SynthesisStructuredBaselineConfig) -> QualityAuditSettings:
    values = config.modeling.source_quality
    return QualityAuditSettings(
        allowed_partition=config.selection.split,
        maximum_gross_to_net_ratio=values.maximum_gross_to_net_ratio,
        robust_minimum_rows=values.robust_minimum_rows,
        robust_z_threshold=values.robust_z_threshold,
        maximum_median_factor=values.maximum_median_factor,
    )


def _routing_settings(config: SynthesisStructuredBaselineConfig) -> ProfileRoutingSettings:
    values = config.modeling.routing

    def threshold(name: str) -> RouteSupportThreshold:
        value = getattr(values, name)
        return RouteSupportThreshold(value.minimum_rows, value.minimum_templates)

    return ProfileRoutingSettings(
        exact_identity_role=threshold("exact_identity_role"),
        semantic_family_role=threshold("semantic_family_role"),
        role_profile=threshold("role_profile"),
    )


def _benchmark_settings(config: SynthesisStructuredBaselineConfig) -> ProfileBenchmarkSettings:
    return ProfileBenchmarkSettings(
        fold_count=config.modeling.grouped_folds,
        seeds=config.modeling.seeds,
        fold_seed=config.modeling.fold_seed,
        quality_margin=config.modeling.simplest_within_quality_margin,
        stability_penalty=config.modeling.stability_penalty,
        proposal_multiplier=config.modeling.maximum_raw_proposals_per_accept,
        proposal_batch_rows=config.modeling.proposal_batch_rows,
        neural_minimum_rows=config.modeling.minimum_rows_for_neural,
        neural_minimum_templates=config.modeling.minimum_templates_for_neural,
        neural_epochs=config.modeling.neural_epochs,
        candidate_identifiers=config.modeling.candidates,
        production_selectable_candidates=config.modeling.selectable_candidates,
        maximum_mean_quality_deficit_vs_empirical=(
            config.modeling.maximum_mean_quality_deficit_vs_empirical
        ),
        maximum_property_quality_deficit_vs_empirical=(
            config.modeling.maximum_property_quality_deficit_vs_empirical
        ),
        maximum_detail_quality_deficit_vs_empirical=(
            config.modeling.maximum_detail_quality_deficit_vs_empirical
        ),
        paired_confidence_level=config.modeling.paired_confidence_level,
        maximum_paired_quality_lcb_deficit=(config.modeling.maximum_paired_quality_lcb_deficit),
        minimum_final_acceptance_yield=config.modeling.minimum_final_acceptance_yield,
        minimum_final_novelty_fraction=config.modeling.minimum_final_novelty_fraction,
    )


def prepare_profile_source(
    contexts: Sequence[GroupProfileContext],
    *,
    config: SynthesisStructuredBaselineConfig,
) -> PreparedProfileSource:
    """Audit the fit source once, then resolve every explicit routing request."""

    if not contexts:
        raise ValueError("profile source requires at least one quantified cargo group")
    rows = tuple(row.source_row for row in contexts)
    audit = audit_source_rows(rows, settings=_quality_settings(config))
    routing = route_profile_requests(
        audit.accepted_rows,
        tuple(row.request for row in contexts),
        settings=_routing_settings(config),
    )
    decisions = {row.request.request_id: row for row in routing.decisions}
    if len(decisions) != len(contexts):
        raise ValueError("profile routing did not return one decision per request")
    return PreparedProfileSource(
        contexts=tuple(contexts),
        audit=audit,
        decisions_by_request=decisions,
        clean_rows_by_id={row.row_id: row for row in audit.accepted_rows},
    )


def _packages_for_group(target: Mapping[str, Any], group_id: str) -> tuple[Mapping[str, Any], ...]:
    return tuple(
        package
        for package in cast(
            Sequence[Mapping[str, Any]],
            target["documentPatch"].get("cargoPackages") or [],
        )
        if package["groupId"] == group_id
    )


def _contextual_values(row: ProfileNumericRow) -> dict[str, float]:
    return {name: float(value) for name, value in row.model_values().items()}


def _normalized_log_vectors(
    values: Sequence[Mapping[str, float]], columns: Sequence[str]
) -> tuple[tuple[tuple[float, ...], ...], tuple[float, ...], tuple[float, ...]]:
    log_rows = tuple(tuple(math.log(row[column]) for column in columns) for row in values)
    minima = tuple(min(row[index] for row in log_rows) for index in range(len(columns)))
    maxima = tuple(max(row[index] for row in log_rows) for index in range(len(columns)))
    spans = tuple(maximum - minimum for minimum, maximum in zip(minima, maxima, strict=True))
    normalized = tuple(
        tuple(
            0.0 if spans[index] == 0 else (value - minima[index]) / spans[index]
            for index, value in enumerate(row)
        )
        for row in log_rows
    )
    return normalized, minima, spans


def _euclidean(left: Sequence[float], right: Sequence[float]) -> float:
    return math.sqrt(sum((a - b) ** 2 for a, b in zip(left, right, strict=True)))


def contextual_support_envelope(
    context: GroupProfileContext,
    *,
    prepared: PreparedProfileSource,
    config: SynthesisStructuredBaselineConfig,
) -> ContextualSupportEnvelope | None:
    """Build the narrowest support-eligible package-aware numeric envelope.

    Rows are already source-audited and template-isolated.  Exact package
    identity is preferred; semantic-family support is used only when the exact
    identity misses its configured evidence minimum.  Unconditioned role-wide
    support is deliberately not a plausibility fallback.
    """

    request = context.request
    candidates = tuple(
        row
        for row in prepared.clean_rows_by_id.values()
        if row.package_role == request.package_role and row.profile == request.profile
    )
    tiers = [
        (
            "exact_identity_role",
            config.modeling.contextual_plausibility.exact_identity_role,
            tuple(row for row in candidates if row.exact_identity == request.exact_identity),
        ),
    ]
    if request.semantic_family not in _NON_SEMANTIC_PACKAGE_FAMILIES:
        tiers.append(
            (
                "semantic_family_role",
                config.modeling.contextual_plausibility.semantic_family_role,
                tuple(row for row in candidates if row.semantic_family == request.semantic_family),
            )
        )
    selected_tier: str | None = None
    selected_rows: tuple[ProfileNumericRow, ...] = ()
    for tier, threshold, rows in tiers:
        templates = {row.template_id for row in rows}
        if len(rows) >= threshold.minimum_rows and len(templates) >= threshold.minimum_templates:
            selected_tier = tier
            selected_rows = tuple(sorted(rows, key=lambda row: row.row_id))
            break
    if selected_tier is None:
        return None

    values = tuple(_contextual_values(row) for row in selected_rows)
    columns = tuple(values[0])
    if any(tuple(row) != columns for row in values):
        raise StructuredProfileError("contextual support rows have inconsistent profiles")
    bounds = {
        column: (
            min(row[column] for row in values),
            max(row[column] for row in values),
        )
        for column in columns
    }
    normalized, minima, spans = _normalized_log_vectors(values, columns)
    nearest_distances = tuple(
        min(
            _euclidean(vector, other)
            for other_index, other in enumerate(normalized)
            if other_index != index
        )
        for index, vector in enumerate(normalized)
    )
    support_payload = [
        {
            "row_id": row.row_id,
            "document_id": row.document_id,
            "template_id": row.template_id,
            "values": values[index],
        }
        for index, row in enumerate(selected_rows)
    ]
    return ContextualSupportEnvelope(
        tier=selected_tier,
        row_ids=tuple(row.row_id for row in selected_rows),
        template_ids=tuple(sorted({row.template_id for row in selected_rows})),
        columns=columns,
        bounds=bounds,
        normalized_vectors=normalized,
        log_minima=minima,
        log_spans=spans,
        maximum_nearest_neighbor_distance=max(nearest_distances),
        support_sha256=sha256_bytes(canonical_json_bytes(support_payload)),
    )


def _contextual_candidate_assessment(
    candidate: ProfileNumericProposal,
    envelope: ContextualSupportEnvelope,
) -> tuple[str | None, float]:
    candidate_values = {
        "quantity": candidate.quantity,
        "gross_per_driver_package_kg": candidate.gross_per_driver_package_kg,
        "net_per_driver_package_kg": candidate.net_per_driver_package_kg,
        "volume_per_driver_package_m3": candidate.volume_per_driver_package_m3,
    }
    values = {name: float(value) for name, value in candidate_values.items() if value is not None}
    if tuple(values) != envelope.columns:
        raise StructuredProfileError("sampled proposal differs from contextual hard profile")
    for column in envelope.columns:
        minimum, maximum = envelope.bounds[column]
        value = values[column]
        tolerance = 1e-12 * max(1.0, abs(minimum), abs(maximum))
        if value < minimum - tolerance or value > maximum + tolerance:
            return f"outside_{envelope.tier}_{column}_range", math.inf
    normalized = tuple(
        0.0
        if envelope.log_spans[index] == 0
        else (math.log(values[column]) - envelope.log_minima[index]) / envelope.log_spans[index]
        for index, column in enumerate(envelope.columns)
    )
    distance = min(_euclidean(normalized, support) for support in envelope.normalized_vectors)
    tolerance = 1e-12 * max(1.0, envelope.maximum_nearest_neighbor_distance)
    if distance > envelope.maximum_nearest_neighbor_distance + tolerance:
        return f"outside_{envelope.tier}_multivariate_support", distance
    return None, distance


def _support_row_proposal(row: ProfileNumericRow) -> ProfileNumericProposal:
    return ProfileNumericProposal(
        quantity=row.quantity,
        gross_per_driver_package_kg=row.gross_per_driver_package_kg,
        net_per_driver_package_kg=row.net_per_driver_package_kg,
        volume_per_driver_package_m3=row.volume_per_driver_package_m3,
        gross_total_kg=(
            row.gross_per_driver_package_kg * row.quantity
            if row.gross_per_driver_package_kg is not None
            else None
        ),
        net_total_kg=(
            row.net_per_driver_package_kg * row.quantity
            if row.net_per_driver_package_kg is not None
            else None
        ),
        volume_total_m3=(
            row.volume_per_driver_package_m3 * row.quantity
            if row.volume_per_driver_package_m3 is not None
            else None
        ),
    )


def profile_support_reasons(
    context: GroupProfileContext,
    *,
    prepared: PreparedProfileSource,
    source_target: Mapping[str, Any],
    source_targets: Mapping[str, Mapping[str, Any]],
    config: SynthesisStructuredBaselineConfig,
) -> tuple[str, ...]:
    """Return explicit reasons a source group cannot produce a changed proposal."""

    reasons: list[str] = []
    if context.request_id not in prepared.clean_rows_by_id:
        reasons.append("source_cargo_group_quarantined")
    decision = prepared.decisions_by_request[context.request_id]
    if not decision.routed:
        reasons.append("cargo_profile_has_no_supported_route")
        return tuple(reasons)
    envelope = contextual_support_envelope(context, prepared=prepared, config=config)
    if envelope is None:
        reasons.append("cargo_profile_has_no_package_aware_support")
        return tuple(reasons)
    driver_id = context.observation.driver_package_id
    if driver_id is None:
        reasons.append("cargo_group_has_no_quantified_driver")
        return tuple(reasons)
    packages = _packages_for_group(source_target, context.group_id)
    source_quantity = context.observation.features.driver_quantity
    alternatives = {
        prepared.clean_rows_by_id[row_id].quantity
        for row_id in envelope.row_ids
        if prepared.clean_rows_by_id[row_id].quantity != source_quantity
    }
    if not any(
        derive_scaled_group_quantities(
            packages=packages,
            driver_package_id=driver_id,
            generated_driver_quantity=value,
        )
        is not None
        for value in alternatives
    ):
        reasons.append("cargo_profile_has_no_coherent_changed_quantity")
        return tuple(reasons)
    capacity_policy = capacity_limits(config.generation.transport_capacity)
    if not any(
        _candidate_to_group_proposal(
            candidate=_support_row_proposal(prepared.clean_rows_by_id[row_id]),
            context=context,
            prepared=prepared,
            source_target=source_target,
            source_targets=source_targets,
            capacity_policy=capacity_policy,
        )
        is not None
        for row_id in envelope.row_ids
    ):
        reasons.append("cargo_profile_has_no_contextually_realizable_changed_proposal")
    return tuple(reasons)


def _source_numeric_value(value: Decimal, *, source: int | float) -> int | float:
    exponent = Decimal(str(source)).as_tuple().exponent
    if not isinstance(exponent, int):
        raise ValueError("source numeric value must have a finite decimal exponent")
    places = max(0, -exponent)
    rounded = value.quantize(Decimal(1).scaleb(-places), rounding=ROUND_HALF_UP)
    if not rounded.is_finite() or rounded <= 0:
        raise ValueError("canonical proposal cannot be represented as a positive source value")
    return int(rounded) if type(source) is int else float(rounded)


def _mass_to_source(
    canonical_kg: float | None,
    projection: CanonicalMassProjection,
) -> int | float | None:
    if canonical_kg is None:
        if projection.source_value is not None:
            raise ValueError("mass proposal changes source missingness")
        return None
    if projection.source_value is None or projection.source_unit is None:
        raise ValueError("mass proposal changes source missingness")
    source = Decimal(str(canonical_kg)) / _MASS_FACTORS[projection.source_unit]
    return _source_numeric_value(source, source=projection.source_value)


def _volume_to_source(
    canonical_m3: float | None,
    source_measure: Mapping[str, Any] | None,
) -> int | float | None:
    if canonical_m3 is None:
        if source_measure is not None:
            raise ValueError("volume proposal changes source missingness")
        return None
    if source_measure is None:
        raise ValueError("volume proposal changes source missingness")
    unit = cast(str, source_measure["unit"])
    try:
        factor = _VOLUME_FACTORS[unit]
    except KeyError as error:
        raise ValueError(f"unsupported target volume unit: {unit!r}") from error
    return _source_numeric_value(
        Decimal(str(canonical_m3)) / factor,
        source=cast(int | float, source_measure["value"]),
    )


def _cohort_per_container_bounds(
    prepared: PreparedProfileSource,
    decision: RouteDecision,
    *,
    source_targets: Mapping[str, Mapping[str, Any]],
    target_is_containerized: bool,
) -> Mapping[str, tuple[float, float]]:
    contexts = {row.request_id: row for row in prepared.contexts}
    values: dict[str, list[float]] = defaultdict(list)
    for row_id in decision.selected_row_ids:
        row = prepared.clean_rows_by_id[row_id]
        source_target = source_targets[row.document_id]
        patch = cast(Mapping[str, Any], source_target["documentPatch"])
        document_container_count = len(
            cast(Sequence[Mapping[str, Any]], patch.get("containers") or ())
        )
        if bool(document_container_count) != target_is_containerized:
            continue
        allocated = contexts[row_id].observation.features.group_allocation_container_count
        divisor = allocated or document_container_count or 1
        for name, attribute in (
            ("gross", "gross_per_driver_package_kg"),
            ("net", "net_per_driver_package_kg"),
            ("volume", "volume_per_driver_package_m3"),
        ):
            per_driver = getattr(row, attribute)
            if per_driver is not None:
                values[name].append(float(per_driver) * row.quantity / divisor)
    return {name: (min(rows), max(rows)) for name, rows in values.items()}


def _allocation_preflight(
    *,
    target: Mapping[str, Any],
    group_id: str,
    quantities: Mapping[str, int],
    source_driver_quantity: int,
    generated_driver_quantity: int,
) -> None:
    packages = deepcopy(
        cast(list[dict[str, Any]], target["documentPatch"].get("cargoPackages") or [])
    )
    for package in packages:
        package_id = cast(str, package["packageId"])
        if package_id in quantities:
            package["quantity"] = quantities[package_id]
    for allocation_group in cast(
        Sequence[Mapping[str, Any]],
        target["documentPatch"].get("cargoAllocationGroups") or [],
    ):
        if allocation_group["groupId"] != group_id:
            continue
        reconciled = reconcile_allocation_group(
            packages=packages,
            allocation_group=allocation_group,
        )
        if allocation_group["coverage"] == "unlinked_package_quantities":
            old = [
                cast(int, row["packageQuantity"])
                for row in cast(Sequence[Mapping[str, Any]], allocation_group["allocations"])
            ]
            total = int(
                (
                    Decimal(sum(old))
                    * Decimal(generated_driver_quantity)
                    / Decimal(source_driver_quantity)
                ).quantize(Decimal("1"), rounding=ROUND_HALF_UP)
            )
            generated = largest_remainder_allocation(total, old)
            if any(value <= 0 for value in generated):
                raise ValueError("unlinked allocation preflight would erase a membership")
        elif any(
            row.get("packageQuantity") is not None and row["packageQuantity"] <= 0
            for row in cast(Sequence[Mapping[str, Any]], reconciled["allocations"])
        ):
            raise ValueError("allocation preflight would erase a membership")


def _candidate_to_group_proposal(
    *,
    candidate: ProfileNumericProposal,
    context: GroupProfileContext,
    prepared: PreparedProfileSource,
    source_target: Mapping[str, Any],
    source_targets: Mapping[str, Mapping[str, Any]],
    capacity_policy: TransportCapacityLimits,
) -> CargoGroupNumericProposal | None:
    observation = context.observation
    driver_id = observation.driver_package_id
    source_quantity = observation.features.driver_quantity
    if driver_id is None or source_quantity is None or candidate.quantity == source_quantity:
        return None
    packages = _packages_for_group(source_target, context.group_id)
    quantities = derive_scaled_group_quantities(
        packages=packages,
        driver_package_id=driver_id,
        generated_driver_quantity=candidate.quantity,
    )
    if quantities is None:
        return None
    decision = prepared.decisions_by_request[context.request_id]
    patch = cast(Mapping[str, Any], source_target["documentPatch"])
    document_container_count = len(cast(Sequence[Mapping[str, Any]], patch.get("containers") or ()))
    bounds = _cohort_per_container_bounds(
        prepared,
        decision,
        source_targets=source_targets,
        target_is_containerized=bool(document_container_count),
    )
    source_container_count = context.observation.features.group_allocation_container_count
    divisor = source_container_count or document_container_count or 1
    capacity_budget = group_capacity_budgets(source_target, capacity_policy)[context.group_id]
    for name, total in (
        ("gross", candidate.gross_total_kg),
        ("net", candidate.net_total_kg),
        ("volume", candidate.volume_total_m3),
    ):
        if total is None:
            if name in bounds:
                return None
            continue
        budget = capacity_budget[name]
        if budget is not None and Decimal(str(total)) > budget:
            return None
        per_container = total / divisor
        if name not in bounds or not bounds[name][0] <= per_container <= bounds[name][1]:
            return None
    projections = {row.feature: row for row in observation.mass_projections}
    source_group = next(
        group
        for group in cast(Sequence[Mapping[str, Any]], patch.get("cargoGroups") or [])
        if group["groupId"] == context.group_id
    )
    try:
        gross = _mass_to_source(
            candidate.gross_total_kg,
            projections["gross_weight_kg"],
        )
        net = _mass_to_source(
            candidate.net_total_kg,
            projections["net_weight_kg"],
        )
        volume = _volume_to_source(
            candidate.volume_total_m3,
            cast(Mapping[str, Any] | None, source_group.get("volume")),
        )
        for name, value in (
            ("grossWeight", gross),
            ("netWeight", net),
            ("volume", volume),
        ):
            source_measure = cast(Mapping[str, Any] | None, source_group.get(name))
            if source_measure is not None and value == source_measure["value"]:
                return None
        validate_mass_order(
            gross_weight=(
                {"value": gross, "unit": source_group["grossWeight"]["unit"]}
                if gross is not None
                else None
            ),
            net_weight=(
                {"value": net, "unit": source_group["netWeight"]["unit"]}
                if net is not None
                else None
            ),
        )
        _allocation_preflight(
            target=source_target,
            group_id=context.group_id,
            quantities=quantities,
            source_driver_quantity=source_quantity,
            generated_driver_quantity=candidate.quantity,
        )
    except ValueError:
        return None
    return CargoGroupNumericProposal(
        group_id=context.group_id,
        driver_package_id=driver_id,
        quantity_by_package_id=quantities,
        gross_weight_value=gross,
        net_weight_value=net,
        volume_value=volume,
    )


def _quality_acceptance(
    result: ProfileRunResult,
    *,
    config: SynthesisStructuredBaselineConfig,
) -> Mapping[str, Any]:
    selected = result.selection.selected_candidate
    if selected not in config.modeling.selectable_candidates:
        raise StructuredProfileError("profile model selected an unconfigured candidate")
    paired: dict[tuple[int, int], dict[str, Any]] = defaultdict(dict)
    for row in result.benchmark_runs:
        if row.candidate in {"empirical", selected}:
            key = (row.fold_index, row.seed)
            if row.candidate in paired[key]:
                raise StructuredProfileError(
                    "profile benchmark repeats a paired empirical/model run: "
                    f"fold={row.fold_index}, seed={row.seed}, candidate={row.candidate}"
                )
            paired[key][row.candidate] = row
    expected_keys = {
        (fold_index, seed)
        for fold_index in range(config.modeling.grouped_folds)
        for seed in config.modeling.seeds
    }
    if set(paired) != expected_keys or any(
        set(rows) != {"empirical", selected} for rows in paired.values()
    ):
        incomplete = {
            key: sorted({"empirical", selected} - set(rows))
            for key, rows in sorted(paired.items())
            if set(rows) != {"empirical", selected}
        }
        raise StructuredProfileError(
            "profile benchmark lacks the configured paired empirical/model runs: "
            f"missing_keys={sorted(expected_keys - set(paired))}, "
            f"extra_keys={sorted(set(paired) - expected_keys)}, "
            f"incomplete={incomplete}"
        )
    paired_runs = tuple(
        _PairedQualityRun(
            fold_index=fold_index,
            seed=seed,
            empirical=rows["empirical"].evaluation.quality,
            selected=rows[selected].evaluation.quality,
        )
        for (fold_index, seed), rows in sorted(paired.items())
    )
    fold_receipts = _paired_fold_quality_receipts(paired_runs)
    fold_differences = [row["difference"] for row in fold_receipts]
    candidate_quality_means = {
        candidate: fmean(
            row.evaluation.quality.score
            for row in result.benchmark_runs
            if row.candidate == candidate
        )
        for candidate in sorted({row.candidate for row in result.benchmark_runs})
    }
    if len(fold_differences) < 2:
        raise StructuredProfileError("profile quality confidence bound needs two independent folds")
    mean_difference = fmean(fold_differences)
    lower_bound = _one_sided_mean_lower_bound(
        fold_differences,
        confidence_level=config.modeling.paired_confidence_level,
    )
    property_receipts = _paired_property_receipts(
        tuple(
            (
                run.empirical.properties,
                run.selected.properties,
            )
            for run in paired_runs
        ),
        fold_seed_keys=tuple((run.fold_index, run.seed) for run in paired_runs),
    )
    property_receipts = tuple(
        {
            **row,
            "maximum_allowed_deficit": (
                config.modeling.maximum_property_quality_deficit_vs_empirical
            ),
            "passed": row["difference"]
            >= -config.modeling.maximum_property_quality_deficit_vs_empirical,
        }
        for row in property_receipts
    )
    detail_receipts = _paired_detail_receipts(paired_runs)
    detail_receipts = tuple(
        {
            **row,
            "maximum_allowed_deficit": (
                config.modeling.maximum_detail_quality_deficit_vs_empirical
            ),
            "passed": row["difference"]
            >= -config.modeling.maximum_detail_quality_deficit_vs_empirical,
        }
        for row in detail_receipts
    )
    novelty = result.final_novelty.novel_rows / result.final_novelty.rows
    acceptance_yield = result.final_proposal.acceptance_yield
    gate_receipts: dict[str, dict[str, Any]] = {
        "mean_quality_difference": {
            "observed": mean_difference,
            "minimum": -config.modeling.maximum_mean_quality_deficit_vs_empirical,
            "passed": mean_difference >= -config.modeling.maximum_mean_quality_deficit_vs_empirical,
        },
        "paired_quality_lower_bound": {
            "observed": lower_bound,
            "minimum": -config.modeling.maximum_paired_quality_lcb_deficit,
            "confidence_level": config.modeling.paired_confidence_level,
            "independent_unit": "validation_fold",
            "passed": lower_bound >= -config.modeling.maximum_paired_quality_lcb_deficit,
        },
        "final_novelty_fraction": {
            "observed": novelty,
            "minimum": config.modeling.minimum_final_novelty_fraction,
            "passed": novelty >= config.modeling.minimum_final_novelty_fraction,
        },
        "final_acceptance_yield": {
            "observed": acceptance_yield,
            "minimum": config.modeling.minimum_final_acceptance_yield,
            "passed": acceptance_yield >= config.modeling.minimum_final_acceptance_yield,
        },
    }
    failures: list[dict[str, Any]] = [
        {"gate": name, **receipt}
        for name, receipt in gate_receipts.items()
        if not receipt["passed"]
    ]
    failures.extend(
        {"gate": "property_quality_deficit", **row}
        for row in property_receipts
        if not row["passed"]
    )
    failures.extend(
        {"gate": "detail_quality_deficit", **row} for row in detail_receipts if not row["passed"]
    )
    receipt = {
        "selected_candidate": selected,
        "paired_runs": len(paired_runs),
        "independent_folds": len(fold_receipts),
        "independent_unit": "validation_fold",
        "folds": list(fold_receipts),
        "mean_quality_difference_vs_empirical": mean_difference,
        "paired_difference_lower_bound": lower_bound,
        "paired_confidence_level": config.modeling.paired_confidence_level,
        "gates": gate_receipts,
        "properties": list(property_receipts),
        "details": list(detail_receipts),
        "candidate_quality_means": candidate_quality_means,
        "final_novelty_fraction": novelty,
        "final_acceptance_yield": acceptance_yield,
        "raw_proposals": result.final_proposal.raw_proposals,
        "rejected_proposals": result.final_proposal.rejected_proposals,
        "failures": failures,
        "accepted": not failures,
    }
    if failures:
        raise ProfileQualityRejection(receipt)
    return receipt


def _one_sided_mean_lower_bound(values: Sequence[float], *, confidence_level: float) -> float:
    """Return the Student-t lower bound at the stated one-sided confidence."""

    if len(values) < 2:
        raise ValueError("a Student-t lower bound requires at least two values")
    if not 0.5 < confidence_level < 1:
        raise ValueError("one-sided confidence must be between 0.5 and 1")
    if any(not math.isfinite(value) for value in values):
        raise ValueError("one-sided confidence input must be finite")
    scipy_stats = __import__("scipy.stats", fromlist=["t"])
    critical = float(scipy_stats.t.ppf(confidence_level, df=len(values) - 1))
    return fmean(values) - critical * stdev(values) / math.sqrt(len(values))


def _paired_fold_quality_receipts(
    paired_runs: Sequence[_PairedQualityRun],
) -> tuple[dict[str, Any], ...]:
    """Average seed replicates inside each fold before statistical inference."""

    if not paired_runs:
        raise ValueError("paired fold quality comparison requires at least one run")
    by_fold: dict[int, list[_PairedQualityRun]] = defaultdict(list)
    for run in paired_runs:
        by_fold[run.fold_index].append(run)
    receipts = []
    for fold_index, runs in sorted(by_fold.items()):
        ordered = sorted(runs, key=lambda row: row.seed)
        seeds = [row.seed for row in ordered]
        if len(seeds) != len(set(seeds)):
            raise StructuredProfileError(
                f"profile quality comparison repeats a seed in fold {fold_index}"
            )
        empirical_mean = fmean(row.empirical.score for row in ordered)
        selected_mean = fmean(row.selected.score for row in ordered)
        receipts.append(
            {
                "fold_index": fold_index,
                "seeds": seeds,
                "seed_replicates": len(ordered),
                "empirical_mean": empirical_mean,
                "selected_mean": selected_mean,
                "difference": selected_mean - empirical_mean,
            }
        )
    return tuple(receipts)


def _paired_property_receipts(
    pairs: Sequence[tuple[Mapping[str, float], Mapping[str, float]]],
    *,
    fold_seed_keys: Sequence[tuple[int, int]] | None = None,
) -> tuple[dict[str, Any], ...]:
    """Compare each property with seeds averaged inside applicable folds."""

    if not pairs:
        raise ValueError("paired property comparison requires at least one pair")
    keys = (
        tuple((index, 0) for index in range(len(pairs)))
        if fold_seed_keys is None
        else tuple(fold_seed_keys)
    )
    if len(keys) != len(pairs) or len(keys) != len(set(keys)):
        raise ValueError("paired property fold/seed keys must be unique and aligned")
    for empirical, selected in pairs:
        if set(empirical) != set(selected):
            raise StructuredProfileError("profile quality properties differ within a paired run")
        for name, score in (*empirical.items(), *selected.items()):
            if not name or not math.isfinite(score) or not 0 <= score <= 1:
                raise StructuredProfileError(
                    f"profile quality property {name!r} is not finite in [0, 1]"
                )
    by_fold: dict[int, list[tuple[int, Mapping[str, float], Mapping[str, float]]]] = defaultdict(
        list
    )
    for (fold_index, seed), (empirical, selected) in zip(keys, pairs, strict=True):
        by_fold[fold_index].append((seed, empirical, selected))
    for fold_index, rows in by_fold.items():
        property_sets = {frozenset(empirical) for _seed, empirical, _selected in rows}
        if len(property_sets) != 1:
            raise StructuredProfileError(
                "profile quality property applicability differs across seeds within "
                f"fold {fold_index}"
            )
    names = sorted({name for empirical, _selected in pairs for name in empirical})
    receipts: list[dict[str, Any]] = []
    for name in names:
        folds = []
        for fold_index, rows in sorted(by_fold.items()):
            applicable = [row for row in rows if name in row[1]]
            if not applicable:
                continue
            empirical_fold_mean = fmean(row[1][name] for row in applicable)
            selected_fold_mean = fmean(row[2][name] for row in applicable)
            folds.append(
                {
                    "fold_index": fold_index,
                    "seed_replicates": len(applicable),
                    "empirical_mean": empirical_fold_mean,
                    "selected_mean": selected_fold_mean,
                    "difference": selected_fold_mean - empirical_fold_mean,
                }
            )
        empirical_mean = fmean(row["empirical_mean"] for row in folds)
        selected_mean = fmean(row["selected_mean"] for row in folds)
        difference = selected_mean - empirical_mean
        receipts.append(
            {
                "property": name,
                "applicable_paired_runs": sum(row["seed_replicates"] for row in folds),
                "total_paired_runs": len(pairs),
                "applicable_folds": len(folds),
                "total_folds": len(by_fold),
                "empirical_mean": empirical_mean,
                "selected_mean": selected_mean,
                "difference": difference,
                "folds": folds,
            }
        )
    return tuple(receipts)


def _quality_detail_scores(details: Sequence[MetricDetail]) -> dict[str, _QualityDetailScore]:
    scores: dict[str, _QualityDetailScore] = {}
    for detail in details:
        values = detail.values
        columns: tuple[str, ...]
        if detail.property_name == "Column Shapes":
            expected = {"Column", "Metric", "Score"}
            if set(values) != expected or values.get("Metric") != "KSComplement":
                raise StructuredProfileError(
                    "Column Shapes detail differs from the KSComplement contract"
                )
            column = values.get("Column")
            if not isinstance(column, str) or not column:
                raise StructuredProfileError("Column Shapes detail has an invalid column")
            metric = "KSComplement"
            columns = (column,)
            real_correlation = None
        elif detail.property_name == "Column Pair Trends":
            expected = {
                "Column 1",
                "Column 2",
                "Metric",
                "Real Correlation",
                "Score",
                "Status",
            }
            if set(values) != expected or values.get("Metric") != "CorrelationSimilarity":
                raise StructuredProfileError(
                    "Column Pair Trends detail differs from the correlation contract"
                )
            left = values.get("Column 1")
            right = values.get("Column 2")
            status = values.get("Status")
            if (
                not isinstance(left, str)
                or not left
                or not isinstance(right, str)
                or not right
                or left == right
                or status not in {"scored", "synthetic_constant_scored_zero"}
            ):
                raise StructuredProfileError("Column Pair Trends detail is invalid")
            correlation = values.get("Real Correlation")
            if (
                isinstance(correlation, bool)
                or not isinstance(correlation, (int, float))
                or not math.isfinite(float(correlation))
                or abs(float(correlation)) <= 0.5
            ):
                raise StructuredProfileError(
                    "Column Pair Trends detail has an invalid real correlation"
                )
            metric = "CorrelationSimilarity"
            columns = (left, right)
            real_correlation = float(correlation)
        else:
            raise StructuredProfileError(
                f"unsupported profile quality detail property: {detail.property_name!r}"
            )
        score = values.get("Score")
        if (
            isinstance(score, bool)
            or not isinstance(score, (int, float))
            or not math.isfinite(float(score))
            or not 0 <= float(score) <= 1
        ):
            raise StructuredProfileError("profile quality detail score is not finite in [0, 1]")
        identity = "|".join((detail.property_name, metric, *columns))
        if identity in scores:
            raise StructuredProfileError(f"profile quality detail is repeated: {identity}")
        scores[identity] = _QualityDetailScore(
            identity=identity,
            property_name=detail.property_name,
            metric=metric,
            columns=columns,
            score=float(score),
            real_correlation=real_correlation,
        )
    return scores


def _paired_detail_receipts(
    paired_runs: Sequence[_PairedQualityRun],
) -> tuple[dict[str, Any], ...]:
    """Produce fold-weighted receipts for every applicable column/detail metric."""

    if not paired_runs:
        raise ValueError("paired detail comparison requires at least one run")
    scored_runs: list[
        tuple[_PairedQualityRun, dict[str, _QualityDetailScore], dict[str, _QualityDetailScore]]
    ] = []
    by_fold_identities: dict[int, set[frozenset[str]]] = defaultdict(set)
    for run in paired_runs:
        empirical = _quality_detail_scores(run.empirical.details)
        selected = _quality_detail_scores(run.selected.details)
        if set(empirical) != set(selected):
            raise StructuredProfileError(
                "profile quality details differ within a paired run: "
                f"fold={run.fold_index}, seed={run.seed}, "
                f"empirical_only={sorted(set(empirical) - set(selected))}, "
                f"selected_only={sorted(set(selected) - set(empirical))}"
            )
        for identity in empirical:
            empirical_detail = empirical[identity]
            selected_detail = selected[identity]
            if (
                empirical_detail.property_name != selected_detail.property_name
                or empirical_detail.metric != selected_detail.metric
                or empirical_detail.columns != selected_detail.columns
                or empirical_detail.real_correlation != selected_detail.real_correlation
            ):
                raise StructuredProfileError(
                    f"profile quality detail metadata differs within paired run: {identity}"
                )
        by_fold_identities[run.fold_index].add(frozenset(empirical))
        scored_runs.append((run, empirical, selected))
    inconsistent_folds = sorted(
        fold_index
        for fold_index, identity_sets in by_fold_identities.items()
        if len(identity_sets) != 1
    )
    if inconsistent_folds:
        raise StructuredProfileError(
            "profile quality detail applicability differs across seeds within folds: "
            f"{inconsistent_folds}"
        )
    identities = sorted(
        {identity for _run, empirical, _selected in scored_runs for identity in empirical}
    )
    receipts = []
    for identity in identities:
        exemplar = next(
            empirical[identity]
            for _run, empirical, _selected in scored_runs
            if identity in empirical
        )
        folds = []
        for fold_index in sorted(by_fold_identities):
            applicable = [
                (run, empirical[identity], selected[identity])
                for run, empirical, selected in scored_runs
                if run.fold_index == fold_index and identity in empirical
            ]
            if not applicable:
                continue
            empirical_fold_mean = fmean(row[1].score for row in applicable)
            selected_fold_mean = fmean(row[2].score for row in applicable)
            folds.append(
                {
                    "fold_index": fold_index,
                    "seed_replicates": len(applicable),
                    "empirical_mean": empirical_fold_mean,
                    "selected_mean": selected_fold_mean,
                    "difference": selected_fold_mean - empirical_fold_mean,
                }
            )
        empirical_mean = fmean(row["empirical_mean"] for row in folds)
        selected_mean = fmean(row["selected_mean"] for row in folds)
        receipts.append(
            {
                "detail": identity,
                "property": exemplar.property_name,
                "metric": exemplar.metric,
                "columns": list(exemplar.columns),
                "applicable_paired_runs": sum(row["seed_replicates"] for row in folds),
                "total_paired_runs": len(paired_runs),
                "applicable_folds": len(folds),
                "total_folds": len(by_fold_identities),
                "empirical_mean": empirical_mean,
                "selected_mean": selected_mean,
                "difference": selected_mean - empirical_mean,
                "folds": folds,
            }
        )
    if not receipts:
        raise StructuredProfileError("profile quality report contains no detail receipts")
    return tuple(receipts)


def _cohort_key(decision: RouteDecision) -> ProfileCohortKey:
    if decision.selected_tier is None:
        raise ValueError("unrouted request cannot define a cohort")
    return decision.request.profile.key, decision.selected_tier, decision.selected_row_ids


def profile_cohort_key(
    prepared: PreparedProfileSource, context: GroupProfileContext
) -> ProfileCohortKey:
    """Return the stable statistical cohort required by one source cargo group."""

    decision = prepared.decisions_by_request.get(context.request_id)
    if decision is None:
        raise ValueError(f"profile request has no routing decision: {context.request_id}")
    return _cohort_key(decision)


def _match_proposals(
    *,
    contexts: Sequence[GroupProfileContext],
    candidates: Sequence[ProfileNumericProposal],
    prepared: PreparedProfileSource,
    source_targets: Mapping[str, Mapping[str, Any]],
    cohort_id: str,
    capacity_policy: TransportCapacityLimits,
    config: SynthesisStructuredBaselineConfig,
) -> tuple[AssignedGroupProposal, ...]:
    unique: list[tuple[int, ProfileNumericProposal]] = []
    seen: set[bytes] = set()
    for index, candidate in enumerate(candidates):
        key = canonical_json_bytes(candidate.to_dict())
        if key not in seen:
            seen.add(key)
            unique.append((index, candidate))
    options: dict[
        str,
        list[tuple[int, CargoGroupNumericProposal, ProfileNumericProposal, float]],
    ] = {}
    envelopes: dict[str, ContextualSupportEnvelope] = {}
    contextual_rejections: list[ProfileContextRejection] = []
    for context in contexts:
        envelope = contextual_support_envelope(context, prepared=prepared, config=config)
        if envelope is None:
            contextual_rejections.append(
                ProfileContextRejection(
                    cohort_id=cohort_id,
                    request_id=context.request_id,
                    document_id=context.document_id,
                    exact_identity=context.request.exact_identity,
                    support={"tier": None, "rows": 0, "templates": 0},
                    sampled_rows=len(candidates),
                    unique_sampled_rows=len(unique),
                    rejection_counts={"no_package_aware_support": len(unique)},
                )
            )
            continue
        envelopes[context.request_id] = envelope
        rows: list[tuple[int, CargoGroupNumericProposal, ProfileNumericProposal, float]] = []
        rejection_counts: Counter[str] = Counter()
        for index, candidate in unique:
            rejection, distance = _contextual_candidate_assessment(candidate, envelope)
            if rejection is not None:
                rejection_counts[rejection] += 1
                continue
            proposal = _candidate_to_group_proposal(
                candidate=candidate,
                context=context,
                prepared=prepared,
                source_target=source_targets[context.document_id],
                source_targets=source_targets,
                capacity_policy=capacity_policy,
            )
            if proposal is not None:
                rows.append((index, proposal, candidate, distance))
            else:
                rejection_counts["projection_capacity_or_no_change"] += 1
        if not rows:
            contextual_rejections.append(
                ProfileContextRejection(
                    cohort_id=cohort_id,
                    request_id=context.request_id,
                    document_id=context.document_id,
                    exact_identity=context.request.exact_identity,
                    support=envelope.to_dict(),
                    sampled_rows=len(candidates),
                    unique_sampled_rows=len(unique),
                    rejection_counts=dict(rejection_counts),
                )
            )
            continue
        options[context.request_id] = rows
    if contextual_rejections:
        raise ProfileContextBatchRejected(contextual_rejections)

    context_by_id = {row.request_id: row for row in contexts}
    equivalence_ids = {
        request_id: _source_numeric_equivalence_id(context)
        for request_id, context in context_by_id.items()
    }
    try:
        selected_indices = _assign_profile_candidate_indices(
            options={
                request_id: tuple(index for index, _proposal, _candidate, _distance in rows)
                for request_id, rows in options.items()
            },
            equivalence_ids=equivalence_ids,
            cohort_id=cohort_id,
            sampled_rows=len(candidates),
            unique_sampled_rows=len(unique),
        )
    except ProfileCandidateAssignmentError as error:
        failed_contexts = tuple(
            context
            for context in contexts
            if equivalence_ids[context.request_id] == error.failed_equivalence_id
        )
        raise ProfileContextBatchRejected(
            tuple(
                ProfileContextRejection(
                    cohort_id=cohort_id,
                    request_id=context.request_id,
                    document_id=context.document_id,
                    exact_identity=context.request.exact_identity,
                    support=envelopes[context.request_id].to_dict(),
                    sampled_rows=len(candidates),
                    unique_sampled_rows=len(unique),
                    rejection_counts={
                        "unique_assignment_infeasible": 1,
                        "candidate_options": error.class_option_counts[error.failed_equivalence_id],
                    },
                )
                for context in failed_contexts
            )
        ) from error
    option_lookup = {
        request_id: {
            index: (proposal, candidate, distance) for index, proposal, candidate, distance in rows
        }
        for request_id, rows in options.items()
    }
    selected = {
        request_id: (
            index,
            option_lookup[request_id][index][0],
            option_lookup[request_id][index][1],
            option_lookup[request_id][index][2],
        )
        for request_id, index in selected_indices.items()
    }
    return tuple(
        AssignedGroupProposal(
            request_id=request_id,
            cohort_id=cohort_id,
            sampled_row_index=selected[request_id][0],
            source_numeric_equivalence_id=equivalence_ids[request_id],
            route_tier=cast(str, prepared.decisions_by_request[request_id].selected_tier),
            contextual_support_tier=envelopes[request_id].tier,
            contextual_support_rows=len(envelopes[request_id].row_ids),
            contextual_support_templates=len(envelopes[request_id].template_ids),
            contextual_support_sha256=envelopes[request_id].support_sha256,
            contextual_distance=selected[request_id][3],
            contextual_maximum_distance=(envelopes[request_id].maximum_nearest_neighbor_distance),
            proposal=selected[request_id][1],
            sampled_values=selected[request_id][2].to_dict(),
        )
        for request_id in sorted(context_by_id)
    )


def _source_numeric_equivalence_id(context: GroupProfileContext) -> str:
    """Identify source-equal numeric rows whose equality must survive synthesis."""

    return sha256_bytes(
        canonical_json_bytes(
            {
                "document_id": context.document_id,
                "exact_identity": context.source_row.exact_identity,
                "semantic_family": context.source_row.semantic_family,
                "package_role": context.source_row.package_role,
                "source_values": {
                    "quantity": context.source_row.quantity,
                    "gross_value": context.source_row.gross_value,
                    "gross_unit": context.source_row.gross_unit,
                    "net_value": context.source_row.net_value,
                    "net_unit": context.source_row.net_unit,
                    "volume_value": context.source_row.volume_value,
                    "volume_unit": context.source_row.volume_unit,
                },
            }
        )
    )


def _assign_profile_candidate_indices(
    *,
    options: Mapping[str, Sequence[int]],
    equivalence_ids: Mapping[str, str],
    cohort_id: str,
    sampled_rows: int,
    unique_sampled_rows: int,
) -> dict[str, int]:
    """Match one unique candidate per source-equivalence class.

    Repeated numeric rows within one document are one source relation, not
    independent observations. Reusing their assigned candidate preserves that
    equality while unrelated source rows still receive distinct candidates.
    """

    if not options or set(options) != set(equivalence_ids):
        raise ValueError("profile candidate options and equivalence IDs must align")
    if not cohort_id or sampled_rows < 1 or unique_sampled_rows < 1:
        raise ValueError("profile assignment diagnostics must be positive and named")
    if any(not values or len(values) != len(set(values)) for values in options.values()):
        raise ValueError("profile candidate options must be non-empty and unique")
    members: dict[str, list[str]] = defaultdict(list)
    for request_id, equivalence_id in sorted(equivalence_ids.items()):
        if not equivalence_id:
            raise ValueError("source numeric equivalence IDs must be non-empty")
        members[equivalence_id].append(request_id)
    class_options: dict[str, tuple[int, ...]] = {}
    for equivalence_id, request_ids in sorted(members.items()):
        common = set(options[request_ids[0]])
        for request_id in request_ids[1:]:
            common.intersection_update(options[request_id])
        if not common:
            option_counts = {key: len(options[key]) for key in request_ids}
            raise StructuredProfileError(
                "source-equal profile requests have no shared realizable sample: "
                f"cohort={cohort_id}, equivalence_id={equivalence_id}, "
                f"request_ids={request_ids}, option_counts={option_counts}"
            )
        class_options[equivalence_id] = tuple(sorted(common))

    assigned_class: dict[int, str] = {}
    selected_by_class: dict[str, int] = {}

    def assign(equivalence_id: str, visited: set[int]) -> bool:
        for index in class_options[equivalence_id]:
            if index in visited:
                continue
            visited.add(index)
            previous = assigned_class.get(index)
            if previous is None or assign(previous, visited):
                assigned_class[index] = equivalence_id
                selected_by_class[equivalence_id] = index
                return True
        return False

    order = sorted(class_options, key=lambda value: (len(class_options[value]), value))
    for equivalence_id in order:
        if not assign(equivalence_id, set()):
            reachable = {index for values in class_options.values() for index in values}
            class_option_counts = {key: len(value) for key, value in sorted(class_options.items())}
            raise ProfileCandidateAssignmentError(
                "profile samples cannot be uniquely assigned across independent "
                "source numeric equivalence classes: "
                f"cohort={cohort_id}, failed_equivalence_id={equivalence_id}, "
                f"requests={len(options)}, equivalence_classes={len(class_options)}, "
                f"sampled_rows={sampled_rows}, unique_sampled_rows={unique_sampled_rows}, "
                f"contextually_reachable_rows={len(reachable)}, "
                f"matched_classes_before_failure={len(selected_by_class)}, "
                f"class_option_counts={class_option_counts}",
                failed_equivalence_id=equivalence_id,
                class_option_counts=class_option_counts,
            )
    return {
        request_id: selected_by_class[equivalence_id]
        for request_id, equivalence_id in equivalence_ids.items()
    }


def generate_profile_proposals(
    *,
    prepared: PreparedProfileSource,
    selected_contexts: Sequence[GroupProfileContext],
    source_targets: Mapping[str, Mapping[str, Any]],
    config: SynthesisStructuredBaselineConfig,
) -> tuple[ProfileCohortResult, ...]:
    """Benchmark each unique routed cohort once and assign bounded novel proposals."""

    context_by_request = {row.request_id: row for row in selected_contexts}
    if len(context_by_request) != len(selected_contexts):
        raise ValueError("selected profile contexts must be unique")
    cohorts: dict[ProfileCohortKey, list[GroupProfileContext]] = defaultdict(list)
    for context in selected_contexts:
        reasons = profile_support_reasons(
            context,
            prepared=prepared,
            source_target=source_targets[context.document_id],
            source_targets=source_targets,
            config=config,
        )
        if reasons:
            raise StructuredProfileError(
                f"selected profile request lost support: {context.request_id}: {reasons}"
            )
        decision = prepared.decisions_by_request[context.request_id]
        cohorts[_cohort_key(decision)].append(context)

    rows = tuple(row.source_row for row in prepared.contexts)
    output: list[ProfileCohortResult] = []
    rejections: list[ProfileCohortRejection] = []
    context_rejections: list[ProfileContextRejection] = []
    transport_limits = capacity_limits(config.generation.transport_capacity)
    expected_candidates = tuple(config.modeling.candidates)
    for key, cohort_contexts in sorted(cohorts.items(), key=lambda row: row[0]):
        profile_key, route_tier, selected_row_ids = key
        cohort_id = (
            "cohort_"
            + sha256_bytes(canonical_json_bytes([profile_key, route_tier, selected_row_ids]))[:24]
        )
        ordered_contexts = tuple(sorted(cohort_contexts, key=lambda row: row.request_id))
        seed = int(
            sha256_bytes(canonical_json_bytes([config.generation.seed, cohort_id]))[:8],
            16,
        )
        requested = len(ordered_contexts) * config.modeling.proposal_assignment_oversample_factor
        result = benchmark_and_sample_profile(
            rows=rows,
            request=ordered_contexts[0].request,
            requested_rows=requested,
            sample_seed=seed,
            quality_settings=_quality_settings(config),
            routing_settings=_routing_settings(config),
            benchmark_settings=_benchmark_settings(config),
        )
        observed_candidates = tuple(row.candidate for row in result.candidate_eligibility)
        if observed_candidates != expected_candidates:
            raise StructuredProfileError(
                "profile candidate inventory differs from configured candidates"
            )
        if result.routing.decisions[0].selected_row_ids != selected_row_ids:
            raise StructuredProfileError("profile cohort changed between routing and fit")
        try:
            quality = _quality_acceptance(result, config=config)
        except ProfileQualityRejection as error:
            rejections.append(
                ProfileCohortRejection(
                    cohort_key=key,
                    cohort_id=cohort_id,
                    request_ids=tuple(row.request_id for row in ordered_contexts),
                    sample_seed=seed,
                    run=result,
                    quality_acceptance=error.receipt,
                )
            )
            continue
        try:
            assignments = _match_proposals(
                contexts=ordered_contexts,
                candidates=result.proposals,
                prepared=prepared,
                source_targets=source_targets,
                cohort_id=cohort_id,
                capacity_policy=transport_limits,
                config=config,
            )
        except ProfileContextBatchRejected as error:
            context_rejections.extend(error.rejections)
            continue
        quality = {
            **quality,
            "downstream_projection": {
                "requested_assignments": len(ordered_contexts),
                "realized_assignments": len(assignments),
                "unique_sample_rows": len({row.sampled_row_index for row in assignments}),
                "source_numeric_equivalence_classes": len(
                    {row.source_numeric_equivalence_id for row in assignments}
                ),
                "reused_rows_preserving_source_numeric_equality": (
                    len(assignments)
                    - len({row.source_numeric_equivalence_id for row in assignments})
                ),
                "constraints": [
                    "source_equal_numeric_rows_within_a_document_remain_equal",
                    "all_present_package_levels_share_one_positive_driver_scale",
                    "all_allocations_reconcile_without_erasing_membership",
                    "gross_weight_is_not_below_net_weight_after_unit_inverse",
                    "same_containerization_class_measure_is_within_clean_route_support",
                    "package_identity_or_semantic_family_values_remain_within_"
                    "isolated_train_support_and_multivariate_neighborhood",
                    "source_share_and_explicit_allocation_capacity_budgets_hold",
                    "source_missingness_units_cardinality_and_topology_are_preserved",
                ],
                "accepted": True,
            },
        }
        output.append(
            ProfileCohortResult(
                cohort_id=cohort_id,
                request_ids=tuple(row.request_id for row in ordered_contexts),
                sample_seed=seed,
                run=result,
                quality_acceptance=quality,
                assignments=assignments,
            )
        )
    if rejections:
        raise ProfileCohortBatchRejected(rejections, output)
    if context_rejections:
        raise ProfileContextBatchRejected(context_rejections, output)
    if {request for row in output for request in row.request_ids} != set(context_by_request):
        raise StructuredProfileError("profile cohort results do not cover selected requests")
    return tuple(output)
