"""Paired, fold-level statistical gates for profile-routed SDV candidates."""

from __future__ import annotations

import math
from collections import defaultdict
from collections.abc import Sequence
from dataclasses import dataclass
from statistics import fmean, stdev
from typing import Any, Protocol

from document_ocr.synthesis.sdv_evaluation import EvaluationBundle, MetricDetail, MetricReport


class ProfileQualityGateError(RuntimeError):
    """A paired candidate benchmark is incomplete or violates its metric contract."""


class QualityRun(Protocol):
    candidate: str
    fold_index: int
    seed: int
    evaluation: EvaluationBundle


@dataclass(frozen=True, slots=True)
class StatisticalQualityThresholds:
    maximum_mean_deficit: float
    maximum_property_deficit: float
    maximum_detail_deficit: float
    paired_confidence_level: float
    maximum_paired_lcb_deficit: float

    def __post_init__(self) -> None:
        for name in (
            "maximum_mean_deficit",
            "maximum_property_deficit",
            "maximum_detail_deficit",
            "maximum_paired_lcb_deficit",
        ):
            value = getattr(self, name)
            if not math.isfinite(value) or not 0 <= value <= 1:
                raise ValueError(f"{name} must be finite in [0, 1]")
        if not 0.5 < self.paired_confidence_level < 1:
            raise ValueError("paired_confidence_level must be between 0.5 and 1")


@dataclass(frozen=True, slots=True)
class _PairedRun:
    fold_index: int
    seed: int
    empirical: MetricReport
    candidate: MetricReport


@dataclass(frozen=True, slots=True)
class _DetailScore:
    identity: str
    property_name: str
    metric: str
    columns: tuple[str, ...]
    score: float
    real_correlation: float | None


def one_sided_mean_lower_bound(values: Sequence[float], *, confidence_level: float) -> float:
    """Return a Student-t lower bound over independent fold means."""

    if len(values) < 2:
        raise ValueError("a Student-t lower bound requires at least two values")
    if not 0.5 < confidence_level < 1:
        raise ValueError("one-sided confidence must be between 0.5 and 1")
    if any(not math.isfinite(value) for value in values):
        raise ValueError("one-sided confidence input must be finite")
    scipy_stats = __import__("scipy.stats", fromlist=["t"])
    critical = float(scipy_stats.t.ppf(confidence_level, df=len(values) - 1))
    return fmean(values) - critical * stdev(values) / math.sqrt(len(values))


def _paired_runs(
    runs: Sequence[QualityRun],
    *,
    candidate: str,
    empirical_candidate: str,
    fold_count: int,
    seeds: Sequence[int],
) -> tuple[_PairedRun, ...]:
    paired: dict[tuple[int, int], dict[str, QualityRun]] = defaultdict(dict)
    for run in runs:
        if run.candidate not in {empirical_candidate, candidate}:
            continue
        key = (run.fold_index, run.seed)
        if run.candidate in paired[key]:
            raise ProfileQualityGateError(
                f"candidate benchmark repeats {run.candidate}: fold={key[0]}, seed={key[1]}"
            )
        paired[key][run.candidate] = run
    expected = {(fold, seed) for fold in range(fold_count) for seed in seeds}
    if set(paired) != expected or any(
        set(values) != {empirical_candidate, candidate} for values in paired.values()
    ):
        raise ProfileQualityGateError(
            f"candidate benchmark is incomplete: candidate={candidate}, "
            f"missing_keys={sorted(expected - set(paired))}, "
            f"extra_keys={sorted(set(paired) - expected)}"
        )
    return tuple(
        _PairedRun(
            fold,
            seed,
            values[empirical_candidate].evaluation.quality,
            values[candidate].evaluation.quality,
        )
        for (fold, seed), values in sorted(paired.items())
    )


def _fold_receipts(runs: Sequence[_PairedRun]) -> tuple[dict[str, Any], ...]:
    by_fold: dict[int, list[_PairedRun]] = defaultdict(list)
    for run in runs:
        by_fold[run.fold_index].append(run)
    receipts = []
    for fold, values in sorted(by_fold.items()):
        ordered = sorted(values, key=lambda row: row.seed)
        if len({row.seed for row in ordered}) != len(ordered):
            raise ProfileQualityGateError(f"candidate benchmark repeats a seed in fold {fold}")
        empirical = fmean(row.empirical.score for row in ordered)
        candidate = fmean(row.candidate.score for row in ordered)
        receipts.append(
            {
                "fold_index": fold,
                "seeds": [row.seed for row in ordered],
                "seed_replicates": len(ordered),
                "empirical_mean": empirical,
                "selected_mean": candidate,
                "difference": candidate - empirical,
            }
        )
    return tuple(receipts)


def _property_receipts(runs: Sequence[_PairedRun]) -> tuple[dict[str, Any], ...]:
    by_fold: dict[int, list[_PairedRun]] = defaultdict(list)
    for run in runs:
        if set(run.empirical.properties) != set(run.candidate.properties):
            raise ProfileQualityGateError("paired candidate quality properties differ")
        by_fold[run.fold_index].append(run)
    for fold, values in by_fold.items():
        if len({frozenset(row.empirical.properties) for row in values}) != 1:
            raise ProfileQualityGateError(
                f"quality property applicability differs across seeds in fold {fold}"
            )
    names = sorted({name for run in runs for name in run.empirical.properties})
    receipts = []
    for name in names:
        folds = []
        for fold, values in sorted(by_fold.items()):
            applicable = [row for row in values if name in row.empirical.properties]
            if not applicable:
                continue
            empirical = fmean(row.empirical.properties[name] for row in applicable)
            candidate = fmean(row.candidate.properties[name] for row in applicable)
            folds.append(
                {
                    "fold_index": fold,
                    "seed_replicates": len(applicable),
                    "empirical_mean": empirical,
                    "selected_mean": candidate,
                    "difference": candidate - empirical,
                }
            )
        receipts.append(
            {
                "property": name,
                "applicable_paired_runs": sum(row["seed_replicates"] for row in folds),
                "total_paired_runs": len(runs),
                "applicable_folds": len(folds),
                "total_folds": len(by_fold),
                "empirical_mean": fmean(row["empirical_mean"] for row in folds),
                "selected_mean": fmean(row["selected_mean"] for row in folds),
                "difference": fmean(row["difference"] for row in folds),
                "folds": folds,
            }
        )
    return tuple(receipts)


def _detail_scores(details: Sequence[MetricDetail]) -> dict[str, _DetailScore]:
    result: dict[str, _DetailScore] = {}
    for detail in details:
        values = detail.values
        columns: tuple[str, ...]
        if detail.property_name == "Column Shapes":
            if (
                set(values) != {"Column", "Metric", "Score"}
                or values.get("Metric") != "KSComplement"
            ):
                raise ProfileQualityGateError("Column Shapes detail contract changed")
            column = values.get("Column")
            if not isinstance(column, str) or not column:
                raise ProfileQualityGateError("Column Shapes detail has an invalid column")
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
                raise ProfileQualityGateError("Column Pair Trends detail contract changed")
            left, right = values.get("Column 1"), values.get("Column 2")
            correlation = values.get("Real Correlation")
            if (
                not isinstance(left, str)
                or not left
                or not isinstance(right, str)
                or not right
                or left == right
                or values.get("Status") not in {"scored", "synthetic_constant_scored_zero"}
                or isinstance(correlation, bool)
                or not isinstance(correlation, (int, float))
                or not math.isfinite(float(correlation))
                or abs(float(correlation)) <= 0.5
            ):
                raise ProfileQualityGateError("Column Pair Trends detail is invalid")
            metric = "CorrelationSimilarity"
            columns = (left, right)
            real_correlation = float(correlation)
        else:
            raise ProfileQualityGateError(
                f"unsupported profile quality property: {detail.property_name}"
            )
        score = values.get("Score")
        if (
            isinstance(score, bool)
            or not isinstance(score, (int, float))
            or not math.isfinite(float(score))
            or not 0 <= float(score) <= 1
        ):
            raise ProfileQualityGateError("quality detail score is not finite in [0, 1]")
        identity = "|".join((detail.property_name, metric, *columns))
        if identity in result:
            raise ProfileQualityGateError(f"quality detail is repeated: {identity}")
        result[identity] = _DetailScore(
            identity,
            detail.property_name,
            metric,
            columns,
            float(score),
            real_correlation,
        )
    return result


def _detail_receipts(runs: Sequence[_PairedRun]) -> tuple[dict[str, Any], ...]:
    scored = []
    by_fold_identities: dict[int, set[frozenset[str]]] = defaultdict(set)
    for run in runs:
        empirical = _detail_scores(run.empirical.details)
        candidate = _detail_scores(run.candidate.details)
        if set(empirical) != set(candidate):
            raise ProfileQualityGateError(
                f"paired quality details differ: fold={run.fold_index}, seed={run.seed}"
            )
        for identity in empirical:
            left, right = empirical[identity], candidate[identity]
            if (
                left.property_name,
                left.metric,
                left.columns,
                left.real_correlation,
            ) != (
                right.property_name,
                right.metric,
                right.columns,
                right.real_correlation,
            ):
                raise ProfileQualityGateError(f"paired detail metadata differs: {identity}")
        by_fold_identities[run.fold_index].add(frozenset(empirical))
        scored.append((run, empirical, candidate))
    inconsistent = [fold for fold, values in by_fold_identities.items() if len(values) != 1]
    if inconsistent:
        raise ProfileQualityGateError(
            f"detail applicability differs across seeds in folds: {sorted(inconsistent)}"
        )
    identities = sorted({identity for _run, left, _right in scored for identity in left})
    receipts = []
    for identity in identities:
        exemplar = next(left[identity] for _run, left, _right in scored if identity in left)
        folds = []
        for fold in sorted(by_fold_identities):
            applicable = [
                (run, left[identity], right[identity])
                for run, left, right in scored
                if run.fold_index == fold and identity in left
            ]
            if not applicable:
                continue
            empirical_score = fmean(row[1].score for row in applicable)
            candidate_score = fmean(row[2].score for row in applicable)
            folds.append(
                {
                    "fold_index": fold,
                    "seed_replicates": len(applicable),
                    "empirical_mean": empirical_score,
                    "selected_mean": candidate_score,
                    "difference": candidate_score - empirical_score,
                }
            )
        receipts.append(
            {
                "detail": identity,
                "property": exemplar.property_name,
                "metric": exemplar.metric,
                "columns": list(exemplar.columns),
                "applicable_paired_runs": sum(row["seed_replicates"] for row in folds),
                "total_paired_runs": len(runs),
                "applicable_folds": len(folds),
                "total_folds": len(by_fold_identities),
                "empirical_mean": fmean(row["empirical_mean"] for row in folds),
                "selected_mean": fmean(row["selected_mean"] for row in folds),
                "difference": fmean(row["difference"] for row in folds),
                "folds": folds,
            }
        )
    if not receipts:
        raise ProfileQualityGateError("quality report contains no detail receipts")
    return tuple(receipts)


def assess_statistical_candidate(
    runs: Sequence[QualityRun],
    *,
    candidate: str,
    empirical_candidate: str,
    fold_count: int,
    seeds: Sequence[int],
    thresholds: StatisticalQualityThresholds,
) -> dict[str, Any]:
    """Assess one production candidate against paired empirical fold baselines."""

    paired = _paired_runs(
        runs,
        candidate=candidate,
        empirical_candidate=empirical_candidate,
        fold_count=fold_count,
        seeds=seeds,
    )
    folds = _fold_receipts(paired)
    differences = [row["difference"] for row in folds]
    mean_difference = fmean(differences)
    lower_bound = one_sided_mean_lower_bound(
        differences, confidence_level=thresholds.paired_confidence_level
    )
    properties = tuple(
        {
            **row,
            "maximum_allowed_deficit": thresholds.maximum_property_deficit,
            "passed": row["difference"] >= -thresholds.maximum_property_deficit,
        }
        for row in _property_receipts(paired)
    )
    details = tuple(
        {
            **row,
            "maximum_allowed_deficit": thresholds.maximum_detail_deficit,
            "passed": row["difference"] >= -thresholds.maximum_detail_deficit,
        }
        for row in _detail_receipts(paired)
    )
    gates: dict[str, dict[str, Any]] = {
        "mean_quality_difference": {
            "observed": mean_difference,
            "minimum": -thresholds.maximum_mean_deficit,
            "passed": mean_difference >= -thresholds.maximum_mean_deficit,
        },
        "paired_quality_lower_bound": {
            "observed": lower_bound,
            "minimum": -thresholds.maximum_paired_lcb_deficit,
            "confidence_level": thresholds.paired_confidence_level,
            "independent_unit": "validation_fold",
            "passed": lower_bound >= -thresholds.maximum_paired_lcb_deficit,
        },
    }
    failures = [
        {"gate": name, **receipt} for name, receipt in gates.items() if not receipt["passed"]
    ]
    failures.extend(
        {"gate": "property_quality_deficit", **row} for row in properties if not row["passed"]
    )
    failures.extend(
        {"gate": "detail_quality_deficit", **row} for row in details if not row["passed"]
    )
    return {
        "candidate": candidate,
        "paired_runs": len(paired),
        "independent_folds": len(folds),
        "independent_unit": "validation_fold",
        "folds": list(folds),
        "mean_quality_difference_vs_empirical": mean_difference,
        "paired_difference_lower_bound": lower_bound,
        "paired_confidence_level": thresholds.paired_confidence_level,
        "gates": gates,
        "properties": list(properties),
        "details": list(details),
        "failures": failures,
        "accepted": not failures,
    }
