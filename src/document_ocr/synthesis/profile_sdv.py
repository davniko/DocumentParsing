"""Profile-routed SDV proposals for coherent cargo quantities and measures.

The statistical model in this module is deliberately narrow.  It proposes one
driver-package quantity and the *per-driver-package* measures that are present
in a hard source profile.  Totals are derived after sampling, so independently
sampled totals can never contradict the sampled quantity.

Missingness is structural rather than statistical: every model view contains
only dense numeric columns for one exact gross/net/volume presence profile.
Routing is explicit and receipted, moving from exact identity to semantic
family to role-wide support only when configured row and template thresholds
are met.  There is no implicit global model or last-resort fallback.
"""

from __future__ import annotations

import contextlib
import json
import math
import platform
import random
import resource
import time
from collections import Counter, defaultdict
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from importlib import import_module
from importlib.metadata import version
from pathlib import Path
from statistics import fmean, median
from types import MappingProxyType
from typing import Any, Literal, cast

from document_ocr.atomic import atomic_publish_json
from document_ocr.hashing import canonical_json_bytes, identity_sha256, sha256_file
from document_ocr.synthesis.profile_quality import (
    QualityRun,
    StatisticalQualityThresholds,
    assess_statistical_candidate,
)
from document_ocr.synthesis.sdv_evaluation import (
    CandidateRunScore,
    EvaluationBundle,
    MetricDetail,
    MetricReport,
    ModelSelection,
    select_candidate,
)
from document_ocr.synthesis.sdv_harness import (
    ProposalReceipt,
    bounded_sample,
    dataframe_sha256,
    grouped_folds,
)

RouteTier = Literal["exact_identity_role", "semantic_family_role", "role_profile"]
CandidateBackend = Literal["empirical", "gaussian_copula", "ctgan", "tvae"]
CandidateRepresentation = Literal["direct", "physical_factors"]
MeasureName = Literal["gross", "net", "volume"]

_ROUTE_TIERS: tuple[RouteTier, ...] = (
    "exact_identity_role",
    "semantic_family_role",
    "role_profile",
)
_WEIGHT_FACTORS_TO_KG: Mapping[str, float] = MappingProxyType(
    {
        "G": 0.001,
        "GRAM": 0.001,
        "GRAMS": 0.001,
        "KG": 1.0,
        "KGS": 1.0,
        "KILOGRAM": 1.0,
        "KILOGRAMS": 1.0,
        "LB": 0.45359237,
        "LBS": 0.45359237,
        "POUND": 0.45359237,
        "POUNDS": 0.45359237,
        "MT": 1000.0,
        "TONNE": 1000.0,
        "TONNES": 1000.0,
        "METRIC TON": 1000.0,
        "METRIC TONS": 1000.0,
        "METRIC TONNE": 1000.0,
        "METRIC TONNES": 1000.0,
    }
)
_VOLUME_FACTORS_TO_M3: Mapping[str, float] = MappingProxyType(
    {
        "CBM": 1.0,
        "M3": 1.0,
        "CUBIC METER": 1.0,
        "CUBIC METERS": 1.0,
        "CUBIC METRE": 1.0,
        "CUBIC METRES": 1.0,
        "FT3": 0.028316846592,
        "CU FT": 0.028316846592,
        "CUBIC FOOT": 0.028316846592,
        "CUBIC FEET": 0.028316846592,
    }
)
_COMPLEXITY_RANKS: Mapping[str, int] = MappingProxyType(
    {
        "empirical": 0,
        "gaussian_copula_default": 1,
        "gaussian_copula_gaussian_kde": 1,
        "gaussian_copula_domain_mixed": 1,
        "gaussian_copula_physical_factors": 1,
        "ctgan": 2,
        "tvae": 2,
    }
)
_DOMAIN_MIXED_NUMERICAL_DISTRIBUTIONS: Mapping[str, str] = MappingProxyType(
    {
        "quantity": "gaussian_kde",
        "gross_per_driver_package_kg": "gamma",
        "net_per_driver_package_kg": "gamma",
        "volume_per_driver_package_m3": "beta",
    }
)


class ProfileSdvError(RuntimeError):
    """The source, route, benchmark, or sample violates this module's contract."""


class ProfileRouteError(ProfileSdvError):
    """No configured hierarchical cohort has enough clean train support."""


@dataclass(frozen=True, slots=True)
class NumericSourceRow:
    """One reviewed task-facing package row before unit normalization."""

    row_id: str
    document_id: str
    template_id: str
    partition: str
    exact_identity: str
    semantic_family: str
    package_role: str
    quantity: int | None
    gross_value: float | None = None
    gross_unit: str | None = None
    net_value: float | None = None
    net_unit: str | None = None
    volume_value: float | None = None
    volume_unit: str | None = None

    def __post_init__(self) -> None:
        for name in (
            "row_id",
            "document_id",
            "template_id",
            "partition",
            "exact_identity",
            "semantic_family",
            "package_role",
        ):
            value = getattr(self, name)
            if not isinstance(value, str) or not value.strip() or value != value.strip():
                raise ValueError(f"{name} must be a non-empty, outer-whitespace-free string")
        if isinstance(self.quantity, bool) or (
            self.quantity is not None and not isinstance(self.quantity, int)
        ):
            raise ValueError("quantity must be an integer or null")
        for measure in ("gross", "net", "volume"):
            value = getattr(self, f"{measure}_value")
            unit = getattr(self, f"{measure}_unit")
            if (value is None) != (unit is None):
                raise ValueError(f"{measure} value and unit must be jointly present or absent")
            if value is not None and (
                isinstance(value, bool)
                or not isinstance(value, (int, float))
                or not math.isfinite(float(value))
            ):
                raise ValueError(f"{measure} value must be finite numeric or null")
            if unit is not None and (
                not isinstance(unit, str) or not unit.strip() or unit != unit.strip()
            ):
                raise ValueError(f"{measure} unit must be non-empty and canonical at its edges")


@dataclass(frozen=True, slots=True, order=True)
class MeasureProfile:
    """Hard structural missingness profile for a dense numeric model."""

    gross: bool
    net: bool
    volume: bool

    @property
    def key(self) -> str:
        return f"g{int(self.gross)}n{int(self.net)}v{int(self.volume)}"

    @property
    def measures(self) -> tuple[MeasureName, ...]:
        return tuple(
            cast(MeasureName, name)
            for name, present in (
                ("gross", self.gross),
                ("net", self.net),
                ("volume", self.volume),
            )
            if present
        )


@dataclass(frozen=True, slots=True)
class ProfileNumericRow:
    """One clean dense row in canonical units and per-driver-package scale."""

    row_id: str
    document_id: str
    template_id: str
    exact_identity: str
    semantic_family: str
    package_role: str
    profile: MeasureProfile
    quantity: int
    gross_per_driver_package_kg: float | None
    net_per_driver_package_kg: float | None
    volume_per_driver_package_m3: float | None

    def model_values(self) -> dict[str, int | float]:
        values: dict[str, int | float] = {"quantity": self.quantity}
        if self.profile.gross:
            assert self.gross_per_driver_package_kg is not None
            values["gross_per_driver_package_kg"] = self.gross_per_driver_package_kg
        if self.profile.net:
            assert self.net_per_driver_package_kg is not None
            values["net_per_driver_package_kg"] = self.net_per_driver_package_kg
        if self.profile.volume:
            assert self.volume_per_driver_package_m3 is not None
            values["volume_per_driver_package_m3"] = self.volume_per_driver_package_m3
        return values


@dataclass(frozen=True, slots=True)
class QualityAuditSettings:
    allowed_partition: str = "train"
    maximum_gross_to_net_ratio: float = 50.0
    robust_minimum_rows: int = 8
    robust_z_threshold: float = 8.0
    maximum_median_factor: float = 100.0

    def __post_init__(self) -> None:
        if not self.allowed_partition:
            raise ValueError("allowed_partition must be non-empty")
        for name in (
            "maximum_gross_to_net_ratio",
            "robust_z_threshold",
            "maximum_median_factor",
        ):
            value = getattr(self, name)
            if not math.isfinite(value) or value <= 1:
                raise ValueError(f"{name} must be finite and greater than one")
        if self.robust_minimum_rows < 4:
            raise ValueError("robust_minimum_rows must be at least four")


@dataclass(frozen=True, slots=True)
class QuarantineReason:
    code: str
    detail: str
    measure: str | None = None
    observed: float | None = None
    threshold: float | None = None
    reference_cohort: str | None = None
    reference_median: float | None = None
    robust_z: float | None = None
    median_factor: float | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "code": self.code,
            "detail": self.detail,
            "measure": self.measure,
            "observed": self.observed,
            "threshold": self.threshold,
            "reference_cohort": self.reference_cohort,
            "reference_median": self.reference_median,
            "robust_z": self.robust_z,
            "median_factor": self.median_factor,
        }


@dataclass(frozen=True, slots=True)
class QuarantinedRow:
    row_id: str
    document_id: str
    template_id: str
    reasons: tuple[QuarantineReason, ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            "row_id": self.row_id,
            "document_id": self.document_id,
            "template_id": self.template_id,
            "reasons": [reason.to_dict() for reason in self.reasons],
        }


@dataclass(frozen=True, slots=True)
class SourceQualityAudit:
    input_rows: int
    accepted_rows: tuple[ProfileNumericRow, ...]
    quarantined_rows: tuple[QuarantinedRow, ...]
    reason_counts: Mapping[str, int]
    source_sha256: str
    clean_sha256: str

    @property
    def accepted_count(self) -> int:
        return len(self.accepted_rows)

    @property
    def quarantined_count(self) -> int:
        return len(self.quarantined_rows)

    def to_dict(self) -> dict[str, Any]:
        return {
            "input_rows": self.input_rows,
            "accepted_rows": self.accepted_count,
            "quarantined_rows": self.quarantined_count,
            "reason_counts": dict(sorted(self.reason_counts.items())),
            "source_sha256": self.source_sha256,
            "clean_sha256": self.clean_sha256,
            "quarantine": [row.to_dict() for row in self.quarantined_rows],
            "normalization": {
                "weight": "kg",
                "volume": "m3",
                "modeled_scale": "per_driver_package",
                "outlier_action": "quarantine_never_clip",
            },
        }


@dataclass(frozen=True, slots=True)
class RouteSupportThreshold:
    minimum_rows: int
    minimum_templates: int

    def __post_init__(self) -> None:
        if self.minimum_rows < 2:
            raise ValueError("route minimum_rows must be at least two")
        if self.minimum_templates < 2:
            raise ValueError("route minimum_templates must be at least two")


@dataclass(frozen=True, slots=True)
class ProfileRoutingSettings:
    exact_identity_role: RouteSupportThreshold
    semantic_family_role: RouteSupportThreshold
    role_profile: RouteSupportThreshold

    def threshold(self, tier: RouteTier) -> RouteSupportThreshold:
        if tier == "exact_identity_role":
            return self.exact_identity_role
        if tier == "semantic_family_role":
            return self.semantic_family_role
        return self.role_profile


@dataclass(frozen=True, slots=True)
class ProfileRouteRequest:
    request_id: str
    exact_identity: str
    semantic_family: str
    package_role: str
    profile: MeasureProfile

    def __post_init__(self) -> None:
        for name in ("request_id", "exact_identity", "semantic_family", "package_role"):
            value = getattr(self, name)
            if not value or value != value.strip():
                raise ValueError(f"{name} must be non-empty and outer-whitespace-free")


@dataclass(frozen=True, slots=True)
class RouteAttempt:
    tier: RouteTier
    rows: int
    templates: int
    minimum_rows: int
    minimum_templates: int
    eligible: bool

    def to_dict(self) -> dict[str, Any]:
        return {
            "tier": self.tier,
            "rows": self.rows,
            "templates": self.templates,
            "minimum_rows": self.minimum_rows,
            "minimum_templates": self.minimum_templates,
            "eligible": self.eligible,
        }


@dataclass(frozen=True, slots=True)
class RouteDecision:
    request: ProfileRouteRequest
    selected_tier: RouteTier | None
    selected_row_ids: tuple[str, ...]
    attempts: tuple[RouteAttempt, ...]

    @property
    def routed(self) -> bool:
        return self.selected_tier is not None

    def to_dict(self) -> dict[str, Any]:
        return {
            "request_id": self.request.request_id,
            "profile": self.request.profile.key,
            "selected_tier": self.selected_tier,
            "selected_rows": len(self.selected_row_ids),
            "selected_row_ids_sha256": identity_sha256(
                "profile-route-membership-v1", *self.selected_row_ids
            ),
            "attempts": [attempt.to_dict() for attempt in self.attempts],
        }


@dataclass(frozen=True, slots=True)
class RoutingCoverageReceipt:
    requests: int
    routed: int
    unroutable: int
    selected_tier_counts: Mapping[str, int]
    decisions: tuple[RouteDecision, ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            "requests": self.requests,
            "routed": self.routed,
            "unroutable": self.unroutable,
            "coverage": self.routed / self.requests if self.requests else 0.0,
            "selected_tier_counts": dict(sorted(self.selected_tier_counts.items())),
            "decisions": [decision.to_dict() for decision in self.decisions],
        }


@dataclass(frozen=True, slots=True)
class DenseProfileView:
    name: str
    profile: MeasureProfile
    data: Any
    metadata: Mapping[str, Any]
    row_ids: tuple[str, ...]
    template_ids: tuple[str, ...]

    def __post_init__(self) -> None:
        if len(self.data) != len(self.row_ids) or len(self.data) != len(self.template_ids):
            raise ValueError("dense view data and provenance lengths differ")
        if len(self.data) < 2 or len(set(self.template_ids)) < 2:
            raise ValueError("dense profile view needs two rows and two templates")
        if bool(self.data.isna().any().any()):
            raise ValueError("dense profile view contains null values")
        if list(self.data.columns) != list(_profile_columns(self.profile)):
            raise ValueError("dense profile view columns differ from hard profile")


@dataclass(frozen=True, slots=True)
class CandidateSpec:
    identifier: str
    backend: CandidateBackend
    parameters: Mapping[str, Any]
    minimum_rows: int
    minimum_templates: int
    representation: CandidateRepresentation = "direct"

    def __post_init__(self) -> None:
        if self.identifier not in _COMPLEXITY_RANKS:
            raise ValueError(f"unsupported profile candidate identifier: {self.identifier}")
        if self.minimum_rows < 2 or self.minimum_templates < 2:
            raise ValueError("candidate support minima must be at least two")
        if self.representation == "physical_factors" and self.backend != "gaussian_copula":
            raise ValueError("physical-factor representation requires Gaussian copula")
        try:
            copied = json.loads(canonical_json_bytes(dict(self.parameters)))
        except (TypeError, ValueError) as error:
            raise ValueError("candidate parameters must be finite JSON") from error
        object.__setattr__(self, "parameters", MappingProxyType(copied))


@dataclass(frozen=True, slots=True)
class CandidateEligibility:
    candidate: str
    rows: int
    templates: int
    minimum_rows: int
    minimum_templates: int
    eligible: bool
    reason: str | None

    def to_dict(self) -> dict[str, Any]:
        return {
            "candidate": self.candidate,
            "rows": self.rows,
            "templates": self.templates,
            "minimum_rows": self.minimum_rows,
            "minimum_templates": self.minimum_templates,
            "eligible": self.eligible,
            "reason": self.reason,
        }


@dataclass(frozen=True, slots=True)
class ProfileBenchmarkSettings:
    fold_count: int
    seeds: tuple[int, ...]
    fold_seed: int = 0
    quality_margin: float = 0.01
    stability_penalty: float = 0.25
    proposal_multiplier: int = 8
    proposal_batch_rows: int | None = None
    neural_minimum_rows: int = 1_000
    neural_minimum_templates: int = 100
    neural_epochs: int = 300
    candidate_identifiers: tuple[str, ...] = tuple(_COMPLEXITY_RANKS)
    production_selectable_candidates: tuple[str, ...] = (
        "gaussian_copula_default",
        "gaussian_copula_gaussian_kde",
        "gaussian_copula_domain_mixed",
        "gaussian_copula_physical_factors",
    )
    maximum_mean_quality_deficit_vs_empirical: float = 1.0
    maximum_property_quality_deficit_vs_empirical: float = 1.0
    maximum_detail_quality_deficit_vs_empirical: float = 1.0
    paired_confidence_level: float = 0.95
    maximum_paired_quality_lcb_deficit: float = 1.0
    minimum_final_acceptance_yield: float = 0.0
    minimum_final_novelty_fraction: float = 0.0

    def __post_init__(self) -> None:
        if self.fold_count < 2:
            raise ValueError("fold_count must be at least two")
        if not self.seeds or len(self.seeds) != len(set(self.seeds)):
            raise ValueError("benchmark seeds must be non-empty and unique")
        if any(seed < 0 or seed >= 2**32 for seed in (*self.seeds, self.fold_seed)):
            raise ValueError("benchmark seeds must be uint32 values")
        if not math.isfinite(self.quality_margin) or self.quality_margin < 0:
            raise ValueError("quality_margin must be finite and non-negative")
        if not math.isfinite(self.stability_penalty) or self.stability_penalty < 0:
            raise ValueError("stability_penalty must be finite and non-negative")
        if self.proposal_multiplier < 1:
            raise ValueError("proposal_multiplier must be positive")
        if self.proposal_batch_rows is not None and self.proposal_batch_rows < 1:
            raise ValueError("proposal_batch_rows must be positive")
        if self.neural_minimum_rows < 2 or self.neural_minimum_templates < 2:
            raise ValueError("neural support minima must be at least two")
        if self.neural_epochs < 1:
            raise ValueError("neural_epochs must be positive")
        for name in (
            "maximum_mean_quality_deficit_vs_empirical",
            "maximum_property_quality_deficit_vs_empirical",
            "maximum_detail_quality_deficit_vs_empirical",
            "maximum_paired_quality_lcb_deficit",
            "minimum_final_acceptance_yield",
            "minimum_final_novelty_fraction",
        ):
            value = getattr(self, name)
            if not math.isfinite(value) or not 0 <= value <= 1:
                raise ValueError(f"{name} must be finite in [0, 1]")
        if not 0.5 < self.paired_confidence_level < 1:
            raise ValueError("paired_confidence_level must be between 0.5 and 1")
        if (
            len(self.candidate_identifiers) != len(set(self.candidate_identifiers))
            or "empirical" not in self.candidate_identifiers
            or not set(self.candidate_identifiers) <= set(_COMPLEXITY_RANKS)
        ):
            raise ValueError(
                "candidate_identifiers must be unique supported models including empirical"
            )
        if (
            not self.production_selectable_candidates
            or len(self.production_selectable_candidates)
            != len(set(self.production_selectable_candidates))
            or "empirical" in self.production_selectable_candidates
            or not set(self.production_selectable_candidates) <= set(self.candidate_identifiers)
        ):
            raise ValueError(
                "production_selectable_candidates must be unique supported non-empirical models"
            )


@dataclass(frozen=True, slots=True)
class ResourceReceipt:
    elapsed_seconds: float
    rss_before_bytes: int
    rss_after_bytes: int
    peak_rss_before_bytes: int
    peak_rss_after_bytes: int

    def to_dict(self) -> dict[str, Any]:
        return {
            "elapsed_seconds": self.elapsed_seconds,
            "rss_before_bytes": self.rss_before_bytes,
            "rss_after_bytes": self.rss_after_bytes,
            "rss_delta_bytes": self.rss_after_bytes - self.rss_before_bytes,
            "peak_rss_before_bytes": self.peak_rss_before_bytes,
            "peak_rss_after_bytes": self.peak_rss_after_bytes,
            "peak_rss_increase_bytes": max(
                0, self.peak_rss_after_bytes - self.peak_rss_before_bytes
            ),
        }


@dataclass(frozen=True, slots=True)
class DomainValidityReceipt:
    rows: int
    valid_rows: int
    invalid_rows: int
    violation_counts: Mapping[str, int]

    @property
    def valid(self) -> bool:
        return self.invalid_rows == 0

    def to_dict(self) -> dict[str, Any]:
        return {
            "rows": self.rows,
            "valid_rows": self.valid_rows,
            "invalid_rows": self.invalid_rows,
            "valid_fraction": self.valid_rows / self.rows if self.rows else 0.0,
            "violation_counts": dict(sorted(self.violation_counts.items())),
        }


@dataclass(frozen=True, slots=True)
class NoveltyReceipt:
    rows: int
    novel_rows: int
    exact_source_matches: int

    def to_dict(self) -> dict[str, Any]:
        return {
            "rows": self.rows,
            "novel_rows": self.novel_rows,
            "exact_source_matches": self.exact_source_matches,
            "novel_fraction": self.novel_rows / self.rows if self.rows else 0.0,
        }


@dataclass(frozen=True, slots=True)
class CandidateRunReceipt:
    candidate: str
    fold_index: int
    seed: int
    train_rows: int
    validation_rows: int
    train_templates: int
    validation_templates: int
    constraint: str
    fit: ResourceReceipt
    sample: ResourceReceipt
    proposal: ProposalReceipt
    validity: DomainValidityReceipt
    novelty: NoveltyReceipt
    evaluation: EvaluationBundle
    synthetic_sha256: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "candidate": self.candidate,
            "fold_index": self.fold_index,
            "seed": self.seed,
            "train_rows": self.train_rows,
            "validation_rows": self.validation_rows,
            "train_templates": self.train_templates,
            "validation_templates": self.validation_templates,
            "constraint": self.constraint,
            "fit": self.fit.to_dict(),
            "sample": self.sample.to_dict(),
            "proposal": self.proposal.to_dict(),
            "validity": self.validity.to_dict(),
            "novelty": self.novelty.to_dict(),
            "evaluation": self.evaluation.to_dict(),
            "synthetic_sha256": self.synthetic_sha256,
        }


@dataclass(frozen=True, slots=True)
class ProfileNumericProposal:
    quantity: int
    gross_per_driver_package_kg: float | None
    net_per_driver_package_kg: float | None
    volume_per_driver_package_m3: float | None
    gross_total_kg: float | None
    net_total_kg: float | None
    volume_total_m3: float | None

    def to_dict(self) -> dict[str, Any]:
        return {
            "quantity": self.quantity,
            "gross_per_driver_package_kg": self.gross_per_driver_package_kg,
            "net_per_driver_package_kg": self.net_per_driver_package_kg,
            "volume_per_driver_package_m3": self.volume_per_driver_package_m3,
            "gross_total_kg": self.gross_total_kg,
            "net_total_kg": self.net_total_kg,
            "volume_total_m3": self.volume_total_m3,
        }


@dataclass(frozen=True, slots=True)
class _FinalCandidateSample:
    candidate: str
    constraint: str
    fit: ResourceReceipt
    sample: ResourceReceipt
    proposal: ProposalReceipt
    validity: DomainValidityReceipt
    novelty: NoveltyReceipt
    proposals: tuple[ProfileNumericProposal, ...]


@dataclass(frozen=True, slots=True)
class ProfileRunResult:
    contract: Mapping[str, Any]
    audit: SourceQualityAudit
    routing: RoutingCoverageReceipt
    candidate_eligibility: tuple[CandidateEligibility, ...]
    benchmark_runs: tuple[CandidateRunReceipt, ...]
    candidate_acceptance: Mapping[str, Any]
    selection: ModelSelection
    final_constraint: str
    final_fit: ResourceReceipt
    final_sample: ResourceReceipt
    final_proposal: ProposalReceipt
    final_validity: DomainValidityReceipt
    final_novelty: NoveltyReceipt
    proposals: tuple[ProfileNumericProposal, ...]
    run_resources: ResourceReceipt
    environment: Mapping[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return {
            "contract": dict(self.contract),
            "audit": self.audit.to_dict(),
            "routing": self.routing.to_dict(),
            "candidate_eligibility": [row.to_dict() for row in self.candidate_eligibility],
            "benchmark_runs": [row.to_dict() for row in self.benchmark_runs],
            "candidate_acceptance": dict(self.candidate_acceptance),
            "selection": self.selection.to_dict(),
            "final_constraint": self.final_constraint,
            "final_fit": self.final_fit.to_dict(),
            "final_sample": self.final_sample.to_dict(),
            "final_proposal": self.final_proposal.to_dict(),
            "final_validity": self.final_validity.to_dict(),
            "final_novelty": self.final_novelty.to_dict(),
            "proposals": [row.to_dict() for row in self.proposals],
            "run_resources": self.run_resources.to_dict(),
            "environment": dict(self.environment),
        }


def _normalized_unit(value: str) -> str:
    return " ".join(value.replace(".", "").replace("_", " ").upper().split())


def _canonical_measure(value: float, unit: str, *, measure: MeasureName) -> float:
    normalized = _normalized_unit(unit)
    factors = _VOLUME_FACTORS_TO_M3 if measure == "volume" else _WEIGHT_FACTORS_TO_KG
    try:
        factor = factors[normalized]
    except KeyError as error:
        dimension = "volume" if measure == "volume" else "weight"
        raise ValueError(f"unsupported {dimension} unit: {unit!r}") from error
    result = float(value) * factor
    if not math.isfinite(result) or result <= 0:
        raise ValueError(f"{measure} must normalize to a finite positive value")
    return result


def _source_payload(row: NumericSourceRow) -> dict[str, Any]:
    return {
        field: getattr(row, field)
        for field in (
            "row_id",
            "document_id",
            "template_id",
            "partition",
            "exact_identity",
            "semantic_family",
            "package_role",
            "quantity",
            "gross_value",
            "gross_unit",
            "net_value",
            "net_unit",
            "volume_value",
            "volume_unit",
        )
    }


def _clean_payload(row: ProfileNumericRow) -> dict[str, Any]:
    return {
        "row_id": row.row_id,
        "document_id": row.document_id,
        "template_id": row.template_id,
        "exact_identity": row.exact_identity,
        "semantic_family": row.semantic_family,
        "package_role": row.package_role,
        "profile": row.profile.key,
        **row.model_values(),
    }


def _preliminary_normalize(
    row: NumericSourceRow, settings: QualityAuditSettings
) -> tuple[ProfileNumericRow | None, list[QuarantineReason]]:
    reasons: list[QuarantineReason] = []
    if row.quantity is None or row.quantity <= 0:
        reasons.append(
            QuarantineReason(
                "missing_or_nonpositive_driver_quantity",
                "per-driver-package modeling requires an explicit positive package quantity",
                observed=float(row.quantity) if row.quantity is not None else None,
            )
        )
        return None, reasons

    normalized: dict[str, float | None] = {}
    for measure in ("gross", "net", "volume"):
        raw_value = getattr(row, f"{measure}_value")
        raw_unit = getattr(row, f"{measure}_unit")
        if raw_value is None:
            normalized[measure] = None
            continue
        assert raw_unit is not None
        try:
            normalized[measure] = _canonical_measure(float(raw_value), raw_unit, measure=measure)
        except ValueError as error:
            reasons.append(QuarantineReason("unsupported_or_invalid_unit", str(error), measure))

    if reasons:
        return None, reasons
    gross = normalized["gross"]
    net = normalized["net"]
    if gross is not None and net is not None:
        ratio = gross / net
        if ratio < 1:
            reasons.append(
                QuarantineReason(
                    "gross_below_net",
                    "unit-normalized gross weight is below net weight",
                    "gross_to_net_ratio",
                    ratio,
                    1.0,
                )
            )
        elif ratio > settings.maximum_gross_to_net_ratio:
            reasons.append(
                QuarantineReason(
                    "gross_to_net_ratio_outlier",
                    "unit-normalized gross/net ratio exceeds the audited physical bound",
                    "gross_to_net_ratio",
                    ratio,
                    settings.maximum_gross_to_net_ratio,
                )
            )
    if reasons:
        return None, reasons

    profile = MeasureProfile(gross is not None, net is not None, normalized["volume"] is not None)
    quantity = row.quantity
    return (
        ProfileNumericRow(
            row_id=row.row_id,
            document_id=row.document_id,
            template_id=row.template_id,
            exact_identity=row.exact_identity,
            semantic_family=row.semantic_family,
            package_role=row.package_role,
            profile=profile,
            quantity=quantity,
            gross_per_driver_package_kg=gross / quantity if gross is not None else None,
            net_per_driver_package_kg=net / quantity if net is not None else None,
            volume_per_driver_package_m3=(
                normalized["volume"] / quantity if normalized["volume"] is not None else None
            ),
        ),
        reasons,
    )


def _robust_outlier_reasons(
    rows: Sequence[ProfileNumericRow], settings: QualityAuditSettings
) -> dict[str, list[QuarantineReason]]:
    reasons: dict[str, list[QuarantineReason]] = defaultdict(list)
    by_cohort: dict[tuple[MeasureProfile, str, str], list[ProfileNumericRow]] = defaultdict(list)
    for row in rows:
        by_cohort[(row.profile, row.semantic_family, row.package_role)].append(row)

    attributes = {
        "gross": "gross_per_driver_package_kg",
        "net": "net_per_driver_package_kg",
        "volume": "volume_per_driver_package_m3",
    }
    for cohort_key, cohort_rows in by_cohort.items():
        profile, semantic_family, package_role = cohort_key
        cohort_receipt = f"{profile.key}|{semantic_family}|{package_role}"
        for measure, attribute in attributes.items():
            supported = [
                row
                for row in cohort_rows
                if (value := getattr(row, attribute)) is not None and value > 0
            ]
            if len(supported) < settings.robust_minimum_rows:
                continue
            values = [float(getattr(row, attribute)) for row in supported]
            log_values = [math.log(value) for value in values]
            center = median(log_values)
            deviations = [abs(value - center) for value in log_values]
            mad = median(deviations)
            raw_center = median(values)
            for row, value, log_value in zip(supported, values, log_values, strict=True):
                factor = max(value / raw_center, raw_center / value)
                robust_z = (
                    0.6744897501960817 * abs(log_value - center) / mad
                    if mad > 0
                    # A zero MAD means the local scale is undefined, not that
                    # every non-median value is infinitely anomalous.  The
                    # explicit median-factor guard remains active in this case.
                    else 0.0
                )
                if (
                    robust_z > settings.robust_z_threshold
                    or factor > settings.maximum_median_factor
                ):
                    reasons[row.row_id].append(
                        QuarantineReason(
                            "robust_per_driver_measure_outlier",
                            (
                                "unit-normalized per-driver-package measure is a robust "
                                "cohort outlier; the value was quarantined, not clipped"
                            ),
                            measure,
                            value,
                            settings.robust_z_threshold,
                            cohort_receipt,
                            raw_center,
                            robust_z,
                            factor,
                        )
                    )
    return reasons


def audit_source_rows(
    rows: Sequence[NumericSourceRow], *, settings: QualityAuditSettings
) -> SourceQualityAudit:
    """Normalize and quarantine suspect fit rows without mutating or clipping values."""

    if not rows:
        raise ValueError("source quality audit requires rows")
    row_ids = [row.row_id for row in rows]
    if len(row_ids) != len(set(row_ids)):
        raise ValueError("source quality audit row IDs must be unique")
    unexpected = sorted({row.partition for row in rows} - {settings.allowed_partition})
    if unexpected:
        raise ValueError(
            "profile SDV fit source is not train-only; unexpected partitions: "
            + ", ".join(unexpected)
        )

    preliminary: list[ProfileNumericRow] = []
    quarantined_reasons: dict[str, list[QuarantineReason]] = defaultdict(list)
    by_id = {row.row_id: row for row in rows}
    for row in rows:
        normalized, reasons = _preliminary_normalize(row, settings)
        if reasons:
            quarantined_reasons[row.row_id].extend(reasons)
        elif normalized is not None:
            preliminary.append(normalized)
    robust = _robust_outlier_reasons(preliminary, settings)
    for row_id, reasons in robust.items():
        quarantined_reasons[row_id].extend(reasons)

    clean = tuple(row for row in preliminary if row.row_id not in quarantined_reasons)
    quarantined = tuple(
        QuarantinedRow(
            row_id=row_id,
            document_id=by_id[row_id].document_id,
            template_id=by_id[row_id].template_id,
            reasons=tuple(quarantined_reasons[row_id]),
        )
        for row_id in sorted(quarantined_reasons)
    )
    counts = Counter(reason.code for row in quarantined for reason in row.reasons)
    return SourceQualityAudit(
        input_rows=len(rows),
        accepted_rows=clean,
        quarantined_rows=quarantined,
        reason_counts=MappingProxyType(dict(counts)),
        source_sha256=identity_sha256(
            "profile-sdv-source-v1", canonical_json_bytes([_source_payload(row) for row in rows])
        ),
        clean_sha256=identity_sha256(
            "profile-sdv-clean-v1", canonical_json_bytes([_clean_payload(row) for row in clean])
        ),
    )


def _route_matches(row: ProfileNumericRow, request: ProfileRouteRequest, tier: RouteTier) -> bool:
    if row.profile != request.profile or row.package_role != request.package_role:
        return False
    if tier == "exact_identity_role":
        return row.exact_identity == request.exact_identity
    if tier == "semantic_family_role":
        return row.semantic_family == request.semantic_family
    return True


def route_profile_requests(
    rows: Sequence[ProfileNumericRow],
    requests: Sequence[ProfileRouteRequest],
    *,
    settings: ProfileRoutingSettings,
) -> RoutingCoverageReceipt:
    """Resolve each request through a visible, finite hierarchy of clean cohorts."""

    if not rows or not requests:
        raise ValueError("profile routing requires clean rows and requests")
    if len({row.row_id for row in rows}) != len(rows):
        raise ValueError("profile routing row IDs must be unique")
    if len({request.request_id for request in requests}) != len(requests):
        raise ValueError("profile routing request IDs must be unique")
    decisions: list[RouteDecision] = []
    tier_counts: Counter[str] = Counter()
    for request in requests:
        attempts: list[RouteAttempt] = []
        selected_tier: RouteTier | None = None
        selected: tuple[ProfileNumericRow, ...] = ()
        for tier in _ROUTE_TIERS:
            cohort = tuple(row for row in rows if _route_matches(row, request, tier))
            threshold = settings.threshold(tier)
            template_count = len({row.template_id for row in cohort})
            eligible = (
                len(cohort) >= threshold.minimum_rows
                and template_count >= threshold.minimum_templates
            )
            attempts.append(
                RouteAttempt(
                    tier=tier,
                    rows=len(cohort),
                    templates=template_count,
                    minimum_rows=threshold.minimum_rows,
                    minimum_templates=threshold.minimum_templates,
                    eligible=eligible,
                )
            )
            if eligible:
                selected_tier = tier
                selected = cohort
                break
        if selected_tier is not None:
            tier_counts[selected_tier] += 1
        decisions.append(
            RouteDecision(
                request=request,
                selected_tier=selected_tier,
                selected_row_ids=tuple(sorted(row.row_id for row in selected)),
                attempts=tuple(attempts),
            )
        )
    routed = sum(decision.routed for decision in decisions)
    return RoutingCoverageReceipt(
        requests=len(requests),
        routed=routed,
        unroutable=len(requests) - routed,
        selected_tier_counts=MappingProxyType(dict(tier_counts)),
        decisions=tuple(decisions),
    )


def _profile_columns(profile: MeasureProfile) -> tuple[str, ...]:
    columns = ["quantity"]
    if profile.gross:
        columns.append("gross_per_driver_package_kg")
    if profile.net:
        columns.append("net_per_driver_package_kg")
    if profile.volume:
        columns.append("volume_per_driver_package_m3")
    return tuple(columns)


def build_dense_profile_view(
    rows: Sequence[ProfileNumericRow], *, profile: MeasureProfile, name: str
) -> DenseProfileView:
    """Build a no-null numeric SDV table for exactly one presence profile."""

    if not name:
        raise ValueError("profile view name must be non-empty")
    selected = tuple(row for row in rows if row.profile == profile)
    if len(selected) != len(rows):
        raise ValueError("dense profile view input mixes hard measure profiles")
    pandas = import_module("pandas")
    columns = _profile_columns(profile)
    data = pandas.DataFrame([row.model_values() for row in selected], columns=columns)
    data["quantity"] = data["quantity"].astype("int64")
    for column in columns[1:]:
        data[column] = data[column].astype("float64")
    metadata_columns: dict[str, dict[str, str]] = {
        "quantity": {"sdtype": "numerical", "computer_representation": "Int64"}
    }
    metadata_columns.update(
        {
            column: {"sdtype": "numerical", "computer_representation": "Float"}
            for column in columns[1:]
        }
    )
    metadata = {
        "METADATA_SPEC_VERSION": "V1",
        "tables": {name: {"columns": metadata_columns}},
        "relationships": [],
    }
    return DenseProfileView(
        name=name,
        profile=profile,
        data=data,
        metadata=MappingProxyType(metadata),
        row_ids=tuple(row.row_id for row in selected),
        template_ids=tuple(row.template_id for row in selected),
    )


def profile_candidate_specs(
    settings: ProfileBenchmarkSettings, *, profile: MeasureProfile
) -> tuple[CandidateSpec, ...]:
    """Return explicit baseline, marginal, and support-gated neural candidates."""

    common = {"enforce_min_max_values": True, "enforce_rounding": True}
    profile_columns = frozenset(_profile_columns(profile))
    mixed_distributions = {
        column: distribution
        for column, distribution in _DOMAIN_MIXED_NUMERICAL_DISTRIBUTIONS.items()
        if column in profile_columns
    }
    if set(mixed_distributions) != profile_columns:
        raise ProfileSdvError(
            "domain-mixed marginal inventory does not cover the dense profile columns"
        )
    candidates = (
        CandidateSpec("empirical", "empirical", {}, 2, 2),
        CandidateSpec("gaussian_copula_default", "gaussian_copula", common, 2, 2),
        CandidateSpec(
            "gaussian_copula_gaussian_kde",
            "gaussian_copula",
            {**common, "default_distribution": "gaussian_kde"},
            2,
            2,
        ),
        CandidateSpec(
            "gaussian_copula_domain_mixed",
            "gaussian_copula",
            {**common, "numerical_distributions": mixed_distributions},
            2,
            2,
        ),
        CandidateSpec(
            "gaussian_copula_physical_factors",
            "gaussian_copula",
            {
                "enforce_min_max_values": True,
                "enforce_rounding": False,
                "default_distribution": "gaussian_kde",
            },
            2,
            2,
            representation="physical_factors",
        ),
        CandidateSpec(
            "ctgan",
            "ctgan",
            {**common, "epochs": settings.neural_epochs, "enable_gpu": False, "verbose": False},
            settings.neural_minimum_rows,
            settings.neural_minimum_templates,
        ),
        CandidateSpec(
            "tvae",
            "tvae",
            {**common, "epochs": settings.neural_epochs, "enable_gpu": False, "verbose": False},
            settings.neural_minimum_rows,
            settings.neural_minimum_templates,
        ),
    )
    return tuple(
        candidate
        for candidate in candidates
        if candidate.identifier in settings.candidate_identifiers
    )


def candidate_eligibility(
    view: DenseProfileView, candidates: Sequence[CandidateSpec]
) -> tuple[CandidateEligibility, ...]:
    templates = len(set(view.template_ids))
    result = []
    for spec in candidates:
        has_support = len(view.data) >= spec.minimum_rows and templates >= spec.minimum_templates
        representation_applies = not (
            spec.representation == "physical_factors" and not view.profile.measures
        )
        eligible = has_support and representation_applies
        if eligible:
            reason = None
        elif not representation_applies:
            reason = "physical_factor_representation_requires_a_measure"
        else:
            reason = "below_configured_minimum_rows_or_distinct_templates"
        result.append(
            CandidateEligibility(
                candidate=spec.identifier,
                rows=len(view.data),
                templates=templates,
                minimum_rows=spec.minimum_rows,
                minimum_templates=spec.minimum_templates,
                eligible=eligible,
                reason=reason,
            )
        )
    return tuple(result)


def _rss_bytes() -> int:
    try:
        with Path("/proc/self/status").open(encoding="ascii") as stream:
            for line in stream:
                if line.startswith("VmRSS:"):
                    return int(line.split()[1]) * 1024
    except (FileNotFoundError, OSError, ValueError, IndexError):
        pass
    usage = resource.getrusage(resource.RUSAGE_SELF)
    return int(usage.ru_maxrss * (1 if platform.system() == "Darwin" else 1024))


def _peak_rss_bytes() -> int:
    usage = resource.getrusage(resource.RUSAGE_SELF)
    return int(usage.ru_maxrss * (1 if platform.system() == "Darwin" else 1024))


def _measure(operation: Callable[[], Any]) -> tuple[Any, ResourceReceipt]:
    rss_before = _rss_bytes()
    peak_before = _peak_rss_bytes()
    started = time.perf_counter()
    result = operation()
    return result, ResourceReceipt(
        elapsed_seconds=time.perf_counter() - started,
        rss_before_bytes=rss_before,
        rss_after_bytes=_rss_bytes(),
        peak_rss_before_bytes=peak_before,
        peak_rss_after_bytes=_peak_rss_bytes(),
    )


@contextlib.contextmanager
def _seeded_runtime(seed: int) -> Any:
    numpy = import_module("numpy")
    torch = import_module("torch")
    python_state = random.getstate()
    numpy_state = numpy.random.get_state()
    torch_state = torch.random.get_rng_state()
    random.seed(seed)
    numpy.random.seed(seed)
    torch.manual_seed(seed)
    try:
        yield
    finally:
        random.setstate(python_state)
        numpy.random.set_state(numpy_state)
        torch.random.set_rng_state(torch_state)


class _EmpiricalModel:
    def __init__(self) -> None:
        self._data: Any | None = None

    def fit(self, data: Any) -> None:
        self._data = data.copy(deep=True).reset_index(drop=True)

    def sample_seeded(self, num_rows: int, seed: int) -> Any:
        if self._data is None:
            raise ProfileSdvError("empirical profile model is not fitted")
        numpy = import_module("numpy")
        indices = numpy.random.default_rng(seed).integers(0, len(self._data), size=num_rows)
        return self._data.iloc[indices].reset_index(drop=True).copy(deep=True)


def _physical_factor_columns(profile: MeasureProfile) -> tuple[str, ...]:
    columns = ["log_quantity"]
    if not profile.gross and not profile.net and profile.volume:
        columns.append("log_total_volume")
    elif not profile.gross and profile.net and not profile.volume:
        columns.append("log_total_net")
    elif not profile.gross and profile.net and profile.volume:
        columns.extend(("log_total_volume", "log_net_density"))
    elif profile.gross and not profile.net and not profile.volume:
        columns.append("log_total_gross")
    elif profile.gross and not profile.net and profile.volume:
        columns.extend(("log_total_volume", "log_gross_density"))
    elif profile.gross and profile.net and not profile.volume:
        columns.extend(("log_total_net", "log_gross_net_ratio"))
    elif profile.gross and profile.net and profile.volume:
        columns.extend(("log_total_net", "log_gross_net_ratio", "log_total_volume"))
    return tuple(columns)


def _physical_factor_metadata(view: DenseProfileView) -> Mapping[str, Any]:
    columns = {
        column: {
            "sdtype": "numerical",
            "computer_representation": "Float",
        }
        for column in _physical_factor_columns(view.profile)
    }
    return {
        "METADATA_SPEC_VERSION": "V1",
        "tables": {view.name: {"columns": columns}},
        "relationships": [],
    }


def _to_physical_factors(data: Any, profile: MeasureProfile) -> Any:
    """Factor positive measures into stable log scale and physical ratios."""

    numpy = import_module("numpy")
    pandas = import_module("pandas")
    if list(data.columns) != list(_profile_columns(profile)):
        raise ProfileSdvError("physical-factor input differs from its dense profile")
    factors = pandas.DataFrame(index=data.index)
    quantity = data["quantity"].astype("float64")
    factors["log_quantity"] = numpy.log(quantity)
    gross = data["gross_per_driver_package_kg"].astype("float64") if profile.gross else None
    net = data["net_per_driver_package_kg"].astype("float64") if profile.net else None
    volume = data["volume_per_driver_package_m3"].astype("float64") if profile.volume else None
    if not profile.gross and not profile.net and profile.volume:
        assert volume is not None
        factors["log_total_volume"] = numpy.log(quantity * volume)
    elif not profile.gross and profile.net and not profile.volume:
        assert net is not None
        factors["log_total_net"] = numpy.log(quantity * net)
    elif not profile.gross and profile.net and profile.volume:
        assert net is not None and volume is not None
        factors["log_total_volume"] = numpy.log(quantity * volume)
        factors["log_net_density"] = numpy.log(net / volume)
    elif profile.gross and not profile.net and not profile.volume:
        assert gross is not None
        factors["log_total_gross"] = numpy.log(quantity * gross)
    elif profile.gross and not profile.net and profile.volume:
        assert gross is not None and volume is not None
        factors["log_total_volume"] = numpy.log(quantity * volume)
        factors["log_gross_density"] = numpy.log(gross / volume)
    elif profile.gross and profile.net and not profile.volume:
        assert gross is not None and net is not None
        factors["log_total_net"] = numpy.log(quantity * net)
        factors["log_gross_net_ratio"] = numpy.log(gross / net)
    elif profile.gross and profile.net and profile.volume:
        assert gross is not None and net is not None and volume is not None
        factors["log_total_net"] = numpy.log(quantity * net)
        factors["log_gross_net_ratio"] = numpy.log(gross / net)
        factors["log_total_volume"] = numpy.log(quantity * volume)
    if list(factors.columns) != list(_physical_factor_columns(profile)) or not bool(
        numpy.isfinite(factors.to_numpy(dtype="float64")).all()
    ):
        raise ProfileSdvError("physical-factor transform produced invalid values")
    return factors.reset_index(drop=True)


def _from_physical_factors(data: Any, profile: MeasureProfile) -> Any:
    """Invert physical factors into the exact public dense-profile contract."""

    numpy = import_module("numpy")
    pandas = import_module("pandas")
    if list(data.columns) != list(_physical_factor_columns(profile)):
        raise ProfileSdvError("sampled physical factors differ from their model contract")
    direct = pandas.DataFrame(index=data.index)
    quantity = numpy.maximum(
        1, numpy.rint(numpy.exp(data["log_quantity"].astype("float64")))
    ).astype("int64")
    direct["quantity"] = quantity
    if not profile.gross and not profile.net and profile.volume:
        direct["volume_per_driver_package_m3"] = (
            numpy.exp(data["log_total_volume"].astype("float64")) / quantity
        )
    elif not profile.gross and profile.net and not profile.volume:
        direct["net_per_driver_package_kg"] = (
            numpy.exp(data["log_total_net"].astype("float64")) / quantity
        )
    elif not profile.gross and profile.net and profile.volume:
        volume = numpy.exp(data["log_total_volume"].astype("float64")) / quantity
        direct["net_per_driver_package_kg"] = volume * numpy.exp(
            data["log_net_density"].astype("float64")
        )
        direct["volume_per_driver_package_m3"] = volume
    elif profile.gross and not profile.net and not profile.volume:
        direct["gross_per_driver_package_kg"] = (
            numpy.exp(data["log_total_gross"].astype("float64")) / quantity
        )
    elif profile.gross and not profile.net and profile.volume:
        volume = numpy.exp(data["log_total_volume"].astype("float64")) / quantity
        direct["gross_per_driver_package_kg"] = volume * numpy.exp(
            data["log_gross_density"].astype("float64")
        )
        direct["volume_per_driver_package_m3"] = volume
    elif profile.gross and profile.net:
        net = numpy.exp(data["log_total_net"].astype("float64")) / quantity
        direct["gross_per_driver_package_kg"] = net * numpy.exp(
            data["log_gross_net_ratio"].astype("float64")
        )
        direct["net_per_driver_package_kg"] = net
        if profile.volume:
            direct["volume_per_driver_package_m3"] = (
                numpy.exp(data["log_total_volume"].astype("float64")) / quantity
            )
    direct = direct.loc[:, list(_profile_columns(profile))].reset_index(drop=True)
    if not bool(numpy.isfinite(direct.to_numpy(dtype="float64")).all()):
        raise ProfileSdvError("physical-factor inverse produced invalid values")
    return direct


class _SdvModel:
    def __init__(self, synthesizer: Any) -> None:
        self._synthesizer = synthesizer

    def fit(self, data: Any) -> None:
        self._synthesizer.fit(data)

    def sample_seeded(self, num_rows: int, seed: int) -> Any:
        setter = getattr(self._synthesizer, "_set_random_state", None)
        if version("sdv") != "1.38.2" or not callable(setter):
            raise ProfileSdvError(
                "seeded profile sampling requires pinned SDV 1.38.2 random-state contract"
            )
        setter(seed)
        return self._synthesizer.sample(num_rows=num_rows)


class _PhysicalFactorSdvModel(_SdvModel):
    """Run SDV on physical factors while exposing the canonical numeric view."""

    def __init__(self, synthesizer: Any, profile: MeasureProfile) -> None:
        super().__init__(synthesizer)
        self._profile = profile

    def fit(self, data: Any) -> None:
        self._synthesizer.fit(_to_physical_factors(data, self._profile))

    def sample_seeded(self, num_rows: int, seed: int) -> Any:
        factors = super().sample_seeded(num_rows, seed)
        return _from_physical_factors(factors, self._profile)


def _create_model(spec: CandidateSpec, view: DenseProfileView) -> tuple[Any, str]:
    if spec.backend == "empirical":
        return _EmpiricalModel(), "empirical_resample_of_source_valid_rows"
    metadata_class = import_module("sdv.metadata").Metadata
    metadata_payload = (
        _physical_factor_metadata(view)
        if spec.representation == "physical_factors"
        else view.metadata
    )
    metadata = metadata_class.load_from_dict(dict(metadata_payload))
    candidates = import_module("sdv.single_table")
    classes = {
        "gaussian_copula": candidates.GaussianCopulaSynthesizer,
        "ctgan": candidates.CTGANSynthesizer,
        "tvae": candidates.TVAESynthesizer,
    }
    synthesizer = classes[spec.backend](metadata, **dict(spec.parameters))
    if spec.representation == "physical_factors":
        return (
            _PhysicalFactorSdvModel(synthesizer, view.profile),
            "log_quantity_total_measures_density_and_gross_net_ratio_v1",
        )
    constraint = "none_profile_does_not_contain_both_net_and_gross"
    if view.profile.net and view.profile.gross:
        inequality = import_module("sdv.cag").Inequality(
            low_column_name="net_per_driver_package_kg",
            high_column_name="gross_per_driver_package_kg",
            strict_boundaries=False,
        )
        synthesizer.add_constraints([inequality])
        constraint = "sdv_cag.Inequality(net_per_driver_package_kg<=gross_per_driver_package_kg)"
    return _SdvModel(synthesizer), constraint


def _fit_model(
    spec: CandidateSpec, view: DenseProfileView, data: Any, seed: int
) -> tuple[Any, str, ResourceReceipt]:
    constraint = ""

    def fit() -> Any:
        nonlocal constraint
        with _seeded_runtime(seed):
            model, constraint = _create_model(spec, view)
            model.fit(data)
            return model

    model, resources = _measure(fit)
    return model, constraint, resources


def _validity(data: Any, profile: MeasureProfile) -> DomainValidityReceipt:
    violations: Counter[str] = Counter()
    valid_rows = 0
    expected = list(_profile_columns(profile))
    if list(data.columns) != expected:
        return DomainValidityReceipt(
            rows=len(data),
            valid_rows=0,
            invalid_rows=len(data),
            violation_counts=MappingProxyType({"wrong_columns_or_order": len(data)}),
        )
    for values in data.itertuples(index=False, name=None):
        row = dict(zip(expected, values, strict=True))
        row_reasons: set[str] = set()
        quantity = row["quantity"]
        if (
            isinstance(quantity, bool)
            or not isinstance(quantity, (int, float))
            or not math.isfinite(float(quantity))
            or float(quantity) <= 0
            or not float(quantity).is_integer()
        ):
            row_reasons.add("quantity_not_positive_integer")
        for column in expected[1:]:
            value = row[column]
            if (
                isinstance(value, bool)
                or not isinstance(value, (int, float))
                or not math.isfinite(float(value))
                or float(value) <= 0
            ):
                row_reasons.add(f"{column}_not_finite_positive")
        if (
            profile.gross
            and profile.net
            and float(row["gross_per_driver_package_kg"]) < float(row["net_per_driver_package_kg"])
        ):
            row_reasons.add("gross_below_net")
        if row_reasons:
            violations.update(row_reasons)
        else:
            valid_rows += 1
    return DomainValidityReceipt(
        rows=len(data),
        valid_rows=valid_rows,
        invalid_rows=len(data) - valid_rows,
        violation_counts=MappingProxyType(dict(violations)),
    )


def _novelty(synthetic: Any, source: Any) -> NoveltyReceipt:
    source_rows = {
        tuple(value.item() if callable(getattr(value, "item", None)) else value for value in row)
        for row in source.itertuples(index=False, name=None)
    }
    synthetic_rows = [
        tuple(value.item() if callable(getattr(value, "item", None)) else value for value in row)
        for row in synthetic.itertuples(index=False, name=None)
    ]
    exact = sum(row in source_rows for row in synthetic_rows)
    return NoveltyReceipt(len(synthetic_rows), len(synthetic_rows) - exact, exact)


def _novel_acceptance(profile: MeasureProfile, source: Any) -> Callable[[Any], Sequence[bool]]:
    """Accept only domain-valid rows that are not exact fitted-source rows."""

    source_rows = {
        tuple(value.item() if callable(getattr(value, "item", None)) else value for value in row)
        for row in source.itertuples(index=False, name=None)
    }
    domain_acceptance = _acceptance(profile)

    def accept(data: Any) -> Sequence[bool]:
        domain_mask = domain_acceptance(data)
        rows = (
            tuple(
                value.item() if callable(getattr(value, "item", None)) else value for value in row
            )
            for row in data.itertuples(index=False, name=None)
        )
        return [
            domain_valid and row not in source_rows
            for domain_valid, row in zip(domain_mask, rows, strict=True)
        ]

    return accept


def _acceptance(profile: MeasureProfile) -> Callable[[Any], Sequence[bool]]:
    def accept(data: Any) -> Sequence[bool]:
        columns = list(data.columns)
        masks = []
        for values in data.itertuples(index=False, name=None):
            row = dict(zip(columns, values, strict=True))
            quantity = row.get("quantity")
            valid = (
                isinstance(quantity, (int, float))
                and not isinstance(quantity, bool)
                and math.isfinite(float(quantity))
                and float(quantity) > 0
                and float(quantity).is_integer()
            )
            for column in columns[1:]:
                value = row[column]
                valid = valid and (
                    isinstance(value, (int, float))
                    and not isinstance(value, bool)
                    and math.isfinite(float(value))
                    and float(value) > 0
                )
            if profile.gross and profile.net:
                valid = valid and float(row["gross_per_driver_package_kg"]) >= float(
                    row["net_per_driver_package_kg"]
                )
            masks.append(valid)
        return masks

    return accept


def _candidate_run(
    *,
    spec: CandidateSpec,
    view: DenseProfileView,
    fold: Any,
    seed: int,
    settings: ProfileBenchmarkSettings,
) -> CandidateRunReceipt:
    train = view.data.iloc[list(fold.train_indices)].reset_index(drop=True).copy()
    validation = view.data.iloc[list(fold.validation_indices)].reset_index(drop=True).copy()
    model, constraint, fit_resources = _fit_model(spec, view, train, seed)

    def sample() -> tuple[Any, ProposalReceipt]:
        return bounded_sample(
            model=model,
            requested_rows=len(validation),
            seed=seed,
            columns=list(view.data.columns),
            acceptance=_acceptance(view.profile),
            proposal_multiplier=settings.proposal_multiplier,
            proposal_batch_rows=settings.proposal_batch_rows,
        )

    sampled, sample_resources = _measure(sample)
    synthetic, proposal = sampled
    validity = _validity(synthetic, view.profile)
    if not validity.valid:
        raise ProfileSdvError(f"candidate {spec.identifier} returned domain-invalid accepted rows")
    evaluation = _evaluate_profile(
        real_data=validation,
        synthetic_data=synthetic,
        metadata=view.metadata,
        table_name=view.name,
        diagnostic_reference_data=train,
    )
    return CandidateRunReceipt(
        candidate=spec.identifier,
        fold_index=fold.index,
        seed=seed,
        train_rows=len(train),
        validation_rows=len(validation),
        train_templates=len(fold.train_groups),
        validation_templates=len(fold.validation_groups),
        constraint=constraint,
        fit=fit_resources,
        sample=sample_resources,
        proposal=proposal,
        validity=validity,
        novelty=_novelty(synthetic, train),
        evaluation=evaluation,
        synthetic_sha256=dataframe_sha256(synthetic),
    )


def _evaluate_profile(
    *,
    real_data: Any,
    synthetic_data: Any,
    metadata: Mapping[str, Any],
    table_name: str,
    diagnostic_reference_data: Any,
) -> EvaluationBundle:
    """Evaluate only statistically applicable properties for a dense numeric profile.

    Unified SDMetrics emits an undefined ``Column Pair Trends`` property for a
    one-column table or when no real numerical pair meets its association
    threshold. That is structurally not applicable, not a candidate failure.
    In that case, the explicit contract uses KSComplement for every numerical
    column. Hard validity is checked before this function is called.
    """

    if not table_name or table_name not in metadata.get("tables", {}):
        raise ProfileSdvError("profile evaluation metadata does not contain its table")
    if list(real_data.columns) != list(synthetic_data.columns) or list(real_data.columns) != list(
        diagnostic_reference_data.columns
    ):
        raise ProfileSdvError("profile evaluation inputs differ or are reordered")
    if len(real_data) == 0 or len(synthetic_data) == 0 or len(diagnostic_reference_data) == 0:
        raise ProfileSdvError("profile evaluation requires non-empty inputs")
    shape_metric = import_module("sdmetrics.single_column").KSComplement
    shape_scores = {
        str(column): float(shape_metric.compute(real_data[column], synthetic_data[column]))
        for column in real_data.columns
    }
    if any(not math.isfinite(score) or not 0 <= score <= 1 for score in shape_scores.values()):
        raise ProfileSdvError("KSComplement score is not finite in [0, 1]")
    shape_score = fmean(shape_scores.values())
    pair_metric = import_module("sdmetrics.column_pairs").CorrelationSimilarity
    pair_scores: dict[tuple[str, str], tuple[float, float, str]] = {}
    constant_input_error = import_module("sdmetrics.errors").ConstantInputError
    columns = [str(column) for column in real_data.columns]
    for left_index, left in enumerate(columns):
        for right in columns[left_index + 1 :]:
            real_correlation = float(real_data[left].corr(real_data[right]))
            if not math.isfinite(real_correlation) or abs(real_correlation) <= 0.5:
                continue
            status = "scored"
            try:
                score = float(
                    pair_metric.compute(
                        real_data[[left, right]],
                        synthetic_data[[left, right]],
                        coefficient="Pearson",
                        real_correlation_threshold=0.5,
                    )
                )
            except constant_input_error:
                score = 0.0
                status = "synthetic_constant_scored_zero"
            if not math.isfinite(score) or not 0 <= score <= 1:
                raise ProfileSdvError(
                    f"CorrelationSimilarity for {left!r}/{right!r} is not finite in [0, 1]"
                )
            pair_scores[(left, right)] = (score, real_correlation, status)
    properties = {"Column Shapes": shape_score}
    if pair_scores:
        properties["Column Pair Trends"] = fmean(row[0] for row in pair_scores.values())
    overall_score = fmean(properties.values())
    details = [
        MetricDetail(
            "Column Shapes",
            MappingProxyType(
                {
                    "Column": column,
                    "Metric": "KSComplement",
                    "Score": column_score,
                }
            ),
        )
        for column, column_score in shape_scores.items()
    ]
    details.extend(
        MetricDetail(
            "Column Pair Trends",
            MappingProxyType(
                {
                    "Column 1": left,
                    "Column 2": right,
                    "Metric": "CorrelationSimilarity",
                    "Real Correlation": real_correlation,
                    "Score": pair_score,
                    "Status": status,
                }
            ),
        )
        for (left, right), (pair_score, real_correlation, status) in pair_scores.items()
    )
    return EvaluationBundle(
        diagnostic=MetricReport(
            report_name="diagnostic",
            score=1.0,
            properties=MappingProxyType({"Data Validity": 1.0, "Data Structure": 1.0}),
            details=(),
        ),
        quality=MetricReport(
            report_name="quality",
            score=overall_score,
            properties=MappingProxyType(properties),
            details=tuple(details),
        ),
    )


def _proposal_rows(data: Any, profile: MeasureProfile) -> tuple[ProfileNumericProposal, ...]:
    proposals = []
    columns = list(data.columns)
    for values in data.itertuples(index=False, name=None):
        row = dict(zip(columns, values, strict=True))
        quantity = int(row["quantity"])
        gross = float(row["gross_per_driver_package_kg"]) if profile.gross else None
        net = float(row["net_per_driver_package_kg"]) if profile.net else None
        volume = float(row["volume_per_driver_package_m3"]) if profile.volume else None
        proposals.append(
            ProfileNumericProposal(
                quantity=quantity,
                gross_per_driver_package_kg=gross,
                net_per_driver_package_kg=net,
                volume_per_driver_package_m3=volume,
                gross_total_kg=gross * quantity if gross is not None else None,
                net_total_kg=net * quantity if net is not None else None,
                volume_total_m3=volume * quantity if volume is not None else None,
            )
        )
    return tuple(proposals)


def _environment() -> Mapping[str, Any]:
    return MappingProxyType(
        {
            "python": platform.python_version(),
            "packages": {
                name: version(name) for name in ("numpy", "pandas", "sdmetrics", "sdv", "torch")
            },
            "seed_contract": "isolated_python_numpy_torch_and_pinned_sdv_state_v1",
            "implementation": {
                "document_ocr/synthesis/profile_sdv.py": sha256_file(Path(__file__)),
            },
        }
    )


def _run_contract(
    *,
    request: ProfileRouteRequest,
    requested_rows: int,
    sample_seed: int,
    quality: QualityAuditSettings,
    routing: ProfileRoutingSettings,
    benchmark: ProfileBenchmarkSettings,
    candidates: Sequence[CandidateSpec],
) -> Mapping[str, Any]:
    payload = {
        "format": "profile_routed_numeric_sdv_v1",
        "request": {
            "request_id": request.request_id,
            "exact_identity": request.exact_identity,
            "semantic_family": request.semantic_family,
            "package_role": request.package_role,
            "profile": request.profile.key,
            "requested_rows": requested_rows,
            "sample_seed": sample_seed,
        },
        "quality": {
            "allowed_partition": quality.allowed_partition,
            "maximum_gross_to_net_ratio": quality.maximum_gross_to_net_ratio,
            "robust_minimum_rows": quality.robust_minimum_rows,
            "robust_z_threshold": quality.robust_z_threshold,
            "maximum_median_factor": quality.maximum_median_factor,
            "outlier_action": "quarantine_never_clip",
        },
        "routing": {
            tier: {
                "minimum_rows": routing.threshold(tier).minimum_rows,
                "minimum_templates": routing.threshold(tier).minimum_templates,
            }
            for tier in _ROUTE_TIERS
        },
        "benchmark": {
            "fold_count": benchmark.fold_count,
            "fold_seed": benchmark.fold_seed,
            "seeds": list(benchmark.seeds),
            "quality_margin": benchmark.quality_margin,
            "stability_penalty": benchmark.stability_penalty,
            "proposal_multiplier": benchmark.proposal_multiplier,
            "proposal_batch_rows": benchmark.proposal_batch_rows,
            "neural_minimum_rows": benchmark.neural_minimum_rows,
            "neural_minimum_templates": benchmark.neural_minimum_templates,
            "neural_epochs": benchmark.neural_epochs,
            "empirical_policy": "required_paired_benchmark_not_production_selectable",
            "candidate_acceptance_policy": (
                "all_statistical_and_final_gates_then_stability_rank_v1"
            ),
            "maximum_mean_quality_deficit_vs_empirical": (
                benchmark.maximum_mean_quality_deficit_vs_empirical
            ),
            "maximum_property_quality_deficit_vs_empirical": (
                benchmark.maximum_property_quality_deficit_vs_empirical
            ),
            "maximum_detail_quality_deficit_vs_empirical": (
                benchmark.maximum_detail_quality_deficit_vs_empirical
            ),
            "paired_confidence_level": benchmark.paired_confidence_level,
            "maximum_paired_quality_lcb_deficit": (benchmark.maximum_paired_quality_lcb_deficit),
            "minimum_final_acceptance_yield": (benchmark.minimum_final_acceptance_yield),
            "minimum_final_novelty_fraction": (benchmark.minimum_final_novelty_fraction),
            "evaluation_contract": (
                "per_column_kscomplement_plus_real_abs_pearson_gt_0_5_pairs_"
                "synthetic_constant_scores_zero_v2"
            ),
            "final_acceptance": ("domain_valid_and_not_exact_quality_accepted_fit_row_v1"),
            "candidate_identifiers": list(benchmark.candidate_identifiers),
            "production_selectable_candidates": list(benchmark.production_selectable_candidates),
            "candidates": [
                {
                    "identifier": spec.identifier,
                    "backend": spec.backend,
                    "representation": spec.representation,
                    "parameters": dict(spec.parameters),
                    "minimum_rows": spec.minimum_rows,
                    "minimum_templates": spec.minimum_templates,
                }
                for spec in candidates
            ],
        },
    }
    # Detach nested structures and prove the contract is finite JSON.
    return MappingProxyType(json.loads(canonical_json_bytes(payload)))


def _publish_result(artifact_dir: Path, result: ProfileRunResult) -> None:
    if artifact_dir.is_symlink():
        raise ProfileSdvError("profile run artifact directory must not be a symbolic link")
    atomic_publish_json(artifact_dir / "contract.json", dict(result.contract))
    atomic_publish_json(artifact_dir / "source-quality-audit.json", result.audit.to_dict())
    atomic_publish_json(artifact_dir / "routing.json", result.routing.to_dict())
    atomic_publish_json(
        artifact_dir / "candidate-eligibility.json",
        [row.to_dict() for row in result.candidate_eligibility],
    )
    atomic_publish_json(
        artifact_dir / "benchmark.json",
        {
            "runs": [row.to_dict() for row in result.benchmark_runs],
            "candidate_acceptance": dict(result.candidate_acceptance),
            "selection": result.selection.to_dict(),
        },
    )
    atomic_publish_json(
        artifact_dir / "sample.json",
        {
            "final_fit": result.final_fit.to_dict(),
            "final_constraint": result.final_constraint,
            "final_sample": result.final_sample.to_dict(),
            "proposal": result.final_proposal.to_dict(),
            "validity": result.final_validity.to_dict(),
            "novelty": result.final_novelty.to_dict(),
            "run_resources": result.run_resources.to_dict(),
            "rows": [row.to_dict() for row in result.proposals],
            "environment": dict(result.environment),
        },
    )
    receipts = [
        {
            "path": str(path.relative_to(artifact_dir)),
            "sha256": sha256_file(path),
            "size_bytes": path.stat().st_size,
        }
        for path in sorted(artifact_dir.glob("*.json"))
        if path.name != "manifest.json"
    ]
    atomic_publish_json(
        artifact_dir / "manifest.json",
        {"status": "complete", "format": "profile_routed_numeric_sdv_v1", "files": receipts},
    )


def _statistical_thresholds(
    settings: ProfileBenchmarkSettings,
) -> StatisticalQualityThresholds:
    return StatisticalQualityThresholds(
        maximum_mean_deficit=settings.maximum_mean_quality_deficit_vs_empirical,
        maximum_property_deficit=settings.maximum_property_quality_deficit_vs_empirical,
        maximum_detail_deficit=settings.maximum_detail_quality_deficit_vs_empirical,
        paired_confidence_level=settings.paired_confidence_level,
        maximum_paired_lcb_deficit=settings.maximum_paired_quality_lcb_deficit,
    )


def _sample_final_candidate(
    *,
    spec: CandidateSpec,
    view: DenseProfileView,
    requested_rows: int,
    sample_seed: int,
    settings: ProfileBenchmarkSettings,
) -> _FinalCandidateSample:
    model, constraint, fit = _fit_model(spec, view, view.data, min(settings.seeds))

    def sample_final() -> tuple[Any, ProposalReceipt]:
        return bounded_sample(
            model=model,
            requested_rows=requested_rows,
            seed=sample_seed,
            columns=list(view.data.columns),
            acceptance=_novel_acceptance(view.profile, view.data),
            proposal_multiplier=settings.proposal_multiplier,
            proposal_batch_rows=settings.proposal_batch_rows,
        )

    sampled, sample = _measure(sample_final)
    synthetic, proposal = sampled
    validity = _validity(synthetic, view.profile)
    if not validity.valid:
        raise ProfileSdvError(f"candidate {spec.identifier} returned domain-invalid final rows")
    return _FinalCandidateSample(
        candidate=spec.identifier,
        constraint=constraint,
        fit=fit,
        sample=sample,
        proposal=proposal,
        validity=validity,
        novelty=_novelty(synthetic, view.data),
        proposals=_proposal_rows(synthetic, view.profile),
    )


def _final_acceptance(
    sample: _FinalCandidateSample, *, settings: ProfileBenchmarkSettings
) -> dict[str, Any]:
    novelty = sample.novelty.novel_rows / sample.novelty.rows
    acceptance_yield = sample.proposal.acceptance_yield
    gates = {
        "final_novelty_fraction": {
            "observed": novelty,
            "minimum": settings.minimum_final_novelty_fraction,
            "passed": novelty >= settings.minimum_final_novelty_fraction,
        },
        "final_acceptance_yield": {
            "observed": acceptance_yield,
            "minimum": settings.minimum_final_acceptance_yield,
            "passed": acceptance_yield >= settings.minimum_final_acceptance_yield,
        },
    }
    failures = [
        {"gate": name, **receipt} for name, receipt in gates.items() if not receipt["passed"]
    ]
    return {
        "gates": gates,
        "final_novelty_fraction": novelty,
        "final_acceptance_yield": acceptance_yield,
        "raw_proposals": sample.proposal.raw_proposals,
        "rejected_proposals": sample.proposal.rejected_proposals,
        "failures": failures,
        "accepted": not failures,
    }


def benchmark_and_sample_profile(
    *,
    rows: Sequence[NumericSourceRow],
    request: ProfileRouteRequest,
    requested_rows: int,
    sample_seed: int,
    quality_settings: QualityAuditSettings,
    routing_settings: ProfileRoutingSettings,
    benchmark_settings: ProfileBenchmarkSettings,
    artifact_dir: Path | None = None,
) -> ProfileRunResult:
    """Audit, route, benchmark, select, and sample one hard numeric profile.

    Every candidate that is support-eligible is mandatory.  A candidate error
    aborts the call rather than silently selecting a surviving model.  Neural
    candidates below their configured data minima are retained in the
    eligibility receipt but are not fitted.
    """

    if requested_rows < 1:
        raise ValueError("requested_rows must be positive")
    if sample_seed < 0 or sample_seed >= 2**32:
        raise ValueError("sample_seed must be a uint32 value")
    if artifact_dir is not None:
        if artifact_dir.is_symlink():
            raise ProfileSdvError("profile run artifact directory must not be a symbolic link")
        if artifact_dir.exists() and not artifact_dir.is_dir():
            raise ProfileSdvError("profile run artifact path must be a directory")
        if artifact_dir.exists() and any(artifact_dir.iterdir()):
            raise ProfileSdvError(
                "profile run artifact directory must be new or empty; "
                "immutable runs are never overwritten or resumed implicitly"
            )
    run_rss_before = _rss_bytes()
    run_peak_before = _peak_rss_bytes()
    run_started = time.perf_counter()
    audit = audit_source_rows(rows, settings=quality_settings)
    routing = route_profile_requests(audit.accepted_rows, (request,), settings=routing_settings)
    decision = routing.decisions[0]
    if not decision.routed:
        raise ProfileRouteError(
            f"no profile route meets configured support for request {request.request_id!r}"
        )
    by_id = {row.row_id: row for row in audit.accepted_rows}
    cohort = tuple(by_id[row_id] for row_id in decision.selected_row_ids)
    view = build_dense_profile_view(
        cohort,
        profile=request.profile,
        name=f"profile_{request.profile.key}",
    )
    candidates = profile_candidate_specs(benchmark_settings, profile=request.profile)
    eligibility = candidate_eligibility(view, candidates)
    eligible_ids = {row.candidate for row in eligibility if row.eligible}
    eligible_specs = tuple(spec for spec in candidates if spec.identifier in eligible_ids)
    if "empirical" not in eligible_ids:
        raise ProfileSdvError("empirical baseline is not support-eligible")
    production_selectable = tuple(
        candidate
        for candidate in benchmark_settings.production_selectable_candidates
        if candidate in eligible_ids
    )
    if not production_selectable:
        raise ProfileSdvError(
            "no non-empirical candidate is support-eligible for production selection"
        )
    folds = grouped_folds(
        group_ids=view.template_ids,
        fold_count=benchmark_settings.fold_count,
        seed=benchmark_settings.fold_seed,
    )
    runs = tuple(
        _candidate_run(
            spec=spec,
            view=view,
            fold=fold,
            seed=seed,
            settings=benchmark_settings,
        )
        for spec in eligible_specs
        for fold in folds
        for seed in benchmark_settings.seeds
    )
    run_scores = tuple(
        CandidateRunScore(
            candidate=run.candidate,
            fold_index=run.fold_index,
            seed=run.seed,
            diagnostic_score=run.evaluation.diagnostic.score,
            quality_score=run.evaluation.quality.score,
        )
        for run in runs
    )
    statistical_acceptance = {
        candidate: assess_statistical_candidate(
            cast(Sequence[QualityRun], runs),
            candidate=candidate,
            empirical_candidate="empirical",
            fold_count=benchmark_settings.fold_count,
            seeds=benchmark_settings.seeds,
            thresholds=_statistical_thresholds(benchmark_settings),
        )
        for candidate in production_selectable
    }
    statistically_accepted = tuple(
        candidate
        for candidate in production_selectable
        if statistical_acceptance[candidate]["accepted"]
    )
    provisional_pool = statistically_accepted or production_selectable
    provisional_selection = select_candidate(
        runs=run_scores,
        complexity_ranks=_COMPLEXITY_RANKS,
        empirical_candidate="empirical",
        quality_margin=benchmark_settings.quality_margin,
        stability_penalty=benchmark_settings.stability_penalty,
        selectable_candidates=provisional_pool,
    )
    spec_by_id = {spec.identifier: spec for spec in eligible_specs}
    final_samples: dict[str, _FinalCandidateSample] = {}
    final_acceptance: dict[str, dict[str, Any]] = {}
    winning_selection: ModelSelection | None = None
    remaining = list(statistically_accepted)
    if remaining:
        while remaining:
            attempt_selection = select_candidate(
                runs=run_scores,
                complexity_ranks=_COMPLEXITY_RANKS,
                empirical_candidate="empirical",
                quality_margin=benchmark_settings.quality_margin,
                stability_penalty=benchmark_settings.stability_penalty,
                selectable_candidates=remaining,
            )
            candidate = attempt_selection.selected_candidate
            sample = _sample_final_candidate(
                spec=spec_by_id[candidate],
                view=view,
                requested_rows=requested_rows,
                sample_seed=sample_seed,
                settings=benchmark_settings,
            )
            final_samples[candidate] = sample
            final_acceptance[candidate] = _final_acceptance(sample, settings=benchmark_settings)
            if final_acceptance[candidate]["accepted"]:
                winning_selection = attempt_selection
                break
            remaining.remove(candidate)
    else:
        candidate = provisional_selection.selected_candidate
        sample = _sample_final_candidate(
            spec=spec_by_id[candidate],
            view=view,
            requested_rows=requested_rows,
            sample_seed=sample_seed,
            settings=benchmark_settings,
        )
        final_samples[candidate] = sample
        final_acceptance[candidate] = _final_acceptance(sample, settings=benchmark_settings)
    candidate_acceptance: dict[str, Any] = {}
    for candidate in production_selectable:
        statistical = statistical_acceptance[candidate]
        if candidate in final_acceptance:
            final = final_acceptance[candidate]
        elif statistical["accepted"]:
            final = {
                "status": "not_evaluated",
                "reason": "higher_ranked_gate_qualified_candidate_passed",
                "gates": {},
                "failures": [],
                "accepted": False,
            }
        else:
            final = {
                "status": "not_evaluated",
                "reason": "statistical_quality_gates_failed",
                "gates": {},
                "failures": [],
                "accepted": False,
            }
        failures = [
            *cast(Sequence[dict[str, Any]], statistical["failures"]),
            *cast(Sequence[dict[str, Any]], final["failures"]),
        ]
        accepted = bool(statistical["accepted"] and final["accepted"])
        candidate_acceptance[candidate] = {
            "candidate": candidate,
            "statistical": statistical,
            "final": final,
            "failures": failures,
            "accepted": accepted,
        }
    selection = winning_selection or provisional_selection
    selected_sample = final_samples[selection.selected_candidate]
    environment = _environment()
    run_resources = ResourceReceipt(
        elapsed_seconds=time.perf_counter() - run_started,
        rss_before_bytes=run_rss_before,
        rss_after_bytes=_rss_bytes(),
        peak_rss_before_bytes=run_peak_before,
        peak_rss_after_bytes=_peak_rss_bytes(),
    )
    result = ProfileRunResult(
        contract=_run_contract(
            request=request,
            requested_rows=requested_rows,
            sample_seed=sample_seed,
            quality=quality_settings,
            routing=routing_settings,
            benchmark=benchmark_settings,
            candidates=candidates,
        ),
        audit=audit,
        routing=routing,
        candidate_eligibility=eligibility,
        benchmark_runs=runs,
        candidate_acceptance=MappingProxyType(candidate_acceptance),
        selection=selection,
        final_constraint=selected_sample.constraint,
        final_fit=selected_sample.fit,
        final_sample=selected_sample.sample,
        final_proposal=selected_sample.proposal,
        final_validity=selected_sample.validity,
        final_novelty=selected_sample.novelty,
        proposals=selected_sample.proposals,
        run_resources=run_resources,
        environment=environment,
    )
    if artifact_dir is not None:
        _publish_result(artifact_dir, result)
    return result
