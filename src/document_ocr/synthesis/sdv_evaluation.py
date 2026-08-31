"""Strict SDMetrics evaluation and deterministic SDV candidate selection.

The statistical synthesizer is a proposal mechanism.  This module deliberately
keeps SDMetrics' fidelity scores separate from hard validity: a candidate is
eligible for selection only when every diagnostic detail is finite, error-free,
and perfect.  Quality-report errors and undefined scores are never averaged
away.
"""

from __future__ import annotations

import contextlib
import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from importlib import import_module
from statistics import fmean, pstdev
from typing import Any


class SdvEvaluationError(RuntimeError):
    """An SDMetrics report is incomplete, erroneous, or non-finite."""


@dataclass(frozen=True, slots=True)
class MetricDetail:
    """One JSON-safe detail row from an SDMetrics report."""

    property_name: str
    values: Mapping[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return {"property": self.property_name, "values": dict(self.values)}


@dataclass(frozen=True, slots=True)
class MetricReport:
    """Complete, validated contents of one SDMetrics report."""

    report_name: str
    score: float
    properties: Mapping[str, float]
    details: tuple[MetricDetail, ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            "report": self.report_name,
            "score": self.score,
            "properties": dict(sorted(self.properties.items())),
            "details": [detail.to_dict() for detail in self.details],
        }


@dataclass(frozen=True, slots=True)
class EvaluationBundle:
    """Hard diagnostic and statistical-fidelity reports for one sample."""

    diagnostic: MetricReport
    quality: MetricReport

    def to_dict(self) -> dict[str, Any]:
        return {
            "diagnostic": self.diagnostic.to_dict(),
            "quality": self.quality.to_dict(),
        }


@dataclass(frozen=True, slots=True)
class CandidateRunScore:
    """The selection-relevant result of one candidate/fold/seed run."""

    candidate: str
    fold_index: int
    seed: int
    diagnostic_score: float
    quality_score: float


@dataclass(frozen=True, slots=True)
class CandidateScorecard:
    candidate: str
    complexity_rank: int
    run_count: int
    mean_quality: float
    quality_stddev: float
    stability_adjusted_quality: float
    mean_improvement_over_empirical: float
    minimum_diagnostic: float

    def to_dict(self) -> dict[str, Any]:
        return {
            "candidate": self.candidate,
            "complexity_rank": self.complexity_rank,
            "run_count": self.run_count,
            "mean_quality": self.mean_quality,
            "quality_stddev": self.quality_stddev,
            "stability_adjusted_quality": self.stability_adjusted_quality,
            "mean_improvement_over_empirical": self.mean_improvement_over_empirical,
            "minimum_diagnostic": self.minimum_diagnostic,
        }


@dataclass(frozen=True, slots=True)
class ModelSelection:
    selected_candidate: str
    selectable_candidates: tuple[str, ...]
    best_stability_adjusted_quality: float
    quality_margin: float
    stability_penalty: float
    selection_rule: str
    scorecards: tuple[CandidateScorecard, ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            "selected_candidate": self.selected_candidate,
            "selectable_candidates": list(self.selectable_candidates),
            "best_stability_adjusted_quality": self.best_stability_adjusted_quality,
            "quality_margin": self.quality_margin,
            "stability_penalty": self.stability_penalty,
            "selection_rule": self.selection_rule,
            "scorecards": [score.to_dict() for score in self.scorecards],
        }


def _json_scalar(value: Any, *, location: str, nan_means_not_applicable: bool = False) -> Any:
    """Convert pandas/numpy scalars while rejecting undefined metric values."""

    if value is None:
        return None
    item = getattr(value, "item", None)
    if callable(item):
        with contextlib.suppress(TypeError, ValueError):
            value = item()
    if isinstance(value, float):
        if math.isnan(value) and nan_means_not_applicable:
            return None
        if not math.isfinite(value):
            raise SdvEvaluationError(f"{location} is not finite: {value!r}")
        return value
    if isinstance(value, (str, int, bool)):
        return value
    # SDMetrics detail cells are expected to be scalar. Stringifying unknown
    # objects would hide an API/schema change, so fail closed.
    raise SdvEvaluationError(
        f"{location} has unsupported metric scalar type: {type(value).__name__}"
    )


def _rows(frame: Any, *, location: str) -> list[dict[str, Any]]:
    if not hasattr(frame, "to_dict"):
        raise SdvEvaluationError(f"{location} is not a tabular SDMetrics result")
    raw_rows = frame.to_dict(orient="records")
    if not isinstance(raw_rows, list):
        raise SdvEvaluationError(f"{location} did not produce record rows")
    rows: list[dict[str, Any]] = []
    for row_index, raw in enumerate(raw_rows):
        if not isinstance(raw, dict):
            raise SdvEvaluationError(f"{location}[{row_index}] is not an object")
        converted: dict[str, Any] = {}
        threshold_value = raw.get("Meets Threshold?")
        threshold_item = getattr(threshold_value, "item", None)
        if callable(threshold_item):
            with contextlib.suppress(TypeError, ValueError):
                threshold_value = threshold_item()
        below_applicability_threshold = threshold_value is False
        for key, value in raw.items():
            if not isinstance(key, str):
                raise SdvEvaluationError(f"{location}[{row_index}] has a non-string key")
            converted[key] = _json_scalar(
                value,
                location=f"{location}[{row_index}].{key}",
                # Unified SDMetrics uses NaN in auxiliary correlation or
                # association columns when that statistic does not apply to
                # the metric. Actual metric/property scores remain strict.
                nan_means_not_applicable=(key != "Score" or below_applicability_threshold),
            )
        rows.append(converted)
    return rows


def _validated_report(report: Any, *, report_name: str) -> MetricReport:
    score = _json_scalar(report.get_score(), location=f"{report_name}.score")
    if not isinstance(score, float):
        raise SdvEvaluationError(f"{report_name}.score is not a float")

    property_rows = _rows(report.get_properties(), location=f"{report_name}.properties")
    if not property_rows:
        raise SdvEvaluationError(f"{report_name} has no properties")
    properties: dict[str, float] = {}
    details: list[MetricDetail] = []
    for row in property_rows:
        property_name = row.get("Property")
        property_score = row.get("Score")
        if not isinstance(property_name, str) or not property_name:
            raise SdvEvaluationError(f"{report_name} has an invalid property name")
        if not isinstance(property_score, float):
            raise SdvEvaluationError(f"{report_name}.{property_name} has a non-float score")
        if property_name in properties:
            raise SdvEvaluationError(f"{report_name} repeats property {property_name!r}")
        properties[property_name] = property_score

        detail_rows = _rows(
            report.get_details(property_name),
            location=f"{report_name}.details.{property_name}",
        )
        # Some quality properties may contain no applicable metric pairs. Their
        # property score is then NaN and has already failed above. An empty but
        # finite detail table is retained rather than invented.
        for detail_index, detail in enumerate(detail_rows):
            error = detail.get("Error")
            if error not in (None, ""):
                raise SdvEvaluationError(
                    f"{report_name}.{property_name}.details[{detail_index}] "
                    f"reported an error: {error}"
                )
            details.append(MetricDetail(property_name, detail))
    return MetricReport(report_name, score, properties, tuple(details))


def evaluate_single_table(
    *,
    real_data: Any,
    synthetic_data: Any,
    metadata: Mapping[str, Any],
    table_name: str,
    diagnostic_reference_data: Any | None = None,
) -> EvaluationBundle:
    """Run complete unified SDMetrics reports for one modeling view.

    The unified 0.30 API expects table dictionaries even for a single table.
    Importing lazily keeps the base OCR environment independent of SDMetrics.
    """

    if not table_name:
        raise ValueError("table_name must be non-empty")
    diagnostic_real = real_data if diagnostic_reference_data is None else diagnostic_reference_data
    if list(real_data.columns) != list(synthetic_data.columns) or list(
        diagnostic_real.columns
    ) != list(synthetic_data.columns):
        raise SdvEvaluationError("real and synthetic columns differ or are reordered")
    if len(real_data) == 0 or len(diagnostic_real) == 0 or len(synthetic_data) == 0:
        raise SdvEvaluationError("SDMetrics evaluation requires non-empty datasets")

    reports = import_module("sdmetrics.reports")
    diagnostic = reports.DiagnosticReport()
    quality = reports.QualityReport()
    real_tables = {table_name: real_data}
    diagnostic_tables = {table_name: diagnostic_real}
    synthetic_tables = {table_name: synthetic_data}
    metadata_dict = dict(metadata)
    diagnostic.generate(diagnostic_tables, synthetic_tables, metadata_dict, verbose=False)
    quality.generate(real_tables, synthetic_tables, metadata_dict, verbose=False)
    diagnostic_result = _validated_report(diagnostic, report_name="diagnostic")
    quality_result = _validated_report(quality, report_name="quality")
    if diagnostic_result.score != 1.0 or any(
        score != 1.0 for score in diagnostic_result.properties.values()
    ):
        raise SdvEvaluationError(
            "SDMetrics diagnostic and every diagnostic property must equal 1.0"
        )
    return EvaluationBundle(diagnostic_result, quality_result)


def select_candidate(
    *,
    runs: Sequence[CandidateRunScore],
    complexity_ranks: Mapping[str, int],
    empirical_candidate: str = "empirical",
    quality_margin: float = 0.01,
    stability_penalty: float = 0.25,
    selectable_candidates: Sequence[str] | None = None,
) -> ModelSelection:
    """Select the simplest statistically competitive, perfectly valid model.

    Candidate comparisons are paired by ``(fold_index, seed)``. This prevents
    a model evaluated on an easier subset from receiving an unearned advantage.
    """

    if not runs:
        raise SdvEvaluationError("candidate selection requires run scores")
    if not math.isfinite(quality_margin) or quality_margin < 0:
        raise ValueError("quality_margin must be finite and non-negative")
    if not math.isfinite(stability_penalty) or stability_penalty < 0:
        raise ValueError("stability_penalty must be finite and non-negative")

    by_candidate: dict[str, list[CandidateRunScore]] = {}
    for row in runs:
        if row.candidate not in complexity_ranks:
            raise SdvEvaluationError(f"missing complexity rank: {row.candidate}")
        if not all(math.isfinite(value) for value in (row.diagnostic_score, row.quality_score)):
            raise SdvEvaluationError(f"non-finite candidate score: {row.candidate}")
        if row.diagnostic_score != 1.0:
            raise SdvEvaluationError(f"candidate {row.candidate} has non-perfect diagnostics")
        by_candidate.setdefault(row.candidate, []).append(row)
    if empirical_candidate not in by_candidate:
        raise SdvEvaluationError("empirical baseline is absent from candidate runs")
    selectable = (
        frozenset(by_candidate)
        if selectable_candidates is None
        else frozenset(selectable_candidates)
    )
    if not selectable:
        raise SdvEvaluationError("at least one benchmark candidate must be selectable")
    if len(selectable) != (
        len(by_candidate) if selectable_candidates is None else len(selectable_candidates)
    ):
        raise SdvEvaluationError("selectable candidates must be unique")
    unknown_selectable = sorted(selectable - set(by_candidate))
    if unknown_selectable:
        raise SdvEvaluationError(
            "selectable candidates were not benchmarked: " + ", ".join(unknown_selectable)
        )

    expected_pairs = {(row.fold_index, row.seed) for row in by_candidate[empirical_candidate]}
    if len(expected_pairs) != len(by_candidate[empirical_candidate]):
        raise SdvEvaluationError("empirical baseline repeats a fold/seed pair")
    baseline = {
        (row.fold_index, row.seed): row.quality_score for row in by_candidate[empirical_candidate]
    }

    scorecards: list[CandidateScorecard] = []
    for candidate, candidate_runs in sorted(by_candidate.items()):
        pairs = {(row.fold_index, row.seed) for row in candidate_runs}
        if len(pairs) != len(candidate_runs) or pairs != expected_pairs:
            raise SdvEvaluationError(
                f"candidate {candidate} does not have the same unique fold/seed pairs as baseline"
            )
        ordered = sorted(candidate_runs, key=lambda row: (row.fold_index, row.seed))
        qualities = [row.quality_score for row in ordered]
        mean_quality = fmean(qualities)
        quality_stddev = pstdev(qualities)
        improvement = fmean(
            row.quality_score - baseline[(row.fold_index, row.seed)] for row in ordered
        )
        scorecards.append(
            CandidateScorecard(
                candidate=candidate,
                complexity_rank=complexity_ranks[candidate],
                run_count=len(ordered),
                mean_quality=mean_quality,
                quality_stddev=quality_stddev,
                stability_adjusted_quality=mean_quality - stability_penalty * quality_stddev,
                mean_improvement_over_empirical=improvement,
                minimum_diagnostic=min(row.diagnostic_score for row in ordered),
            )
        )

    selectable_scorecards = [score for score in scorecards if score.candidate in selectable]
    best = max(score.stability_adjusted_quality for score in selectable_scorecards)
    competitive = [
        score
        for score in selectable_scorecards
        if score.stability_adjusted_quality >= best - quality_margin
    ]
    selected = min(
        competitive,
        key=lambda score: (
            score.complexity_rank,
            -score.stability_adjusted_quality,
            score.candidate,
        ),
    )
    return ModelSelection(
        selected_candidate=selected.candidate,
        selectable_candidates=tuple(sorted(selectable)),
        best_stability_adjusted_quality=best,
        quality_margin=quality_margin,
        stability_penalty=stability_penalty,
        selection_rule=(
            "perfect_diagnostics_then_max_mean_minus_stability_penalty_times_stddev;"
            "select_lowest_complexity_within_quality_margin_then_name"
        ),
        scorecards=tuple(sorted(scorecards, key=lambda score: score.candidate)),
    )
