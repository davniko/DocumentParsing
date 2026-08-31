"""Comparative SDV benchmark for the PII-free party structure view.

This experiment deliberately stops at paired model comparison.  It does not
select, refit, or publish a production synthesizer.  Party names, addresses,
cities, countries, and contact values never enter the view or its receipts.
"""

from __future__ import annotations

import hashlib
import math
import statistics
import sys
import time
from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from importlib.metadata import version
from pathlib import Path
from types import MappingProxyType
from typing import Any, Literal

from document_ocr.atomic import atomic_publish_json
from document_ocr.hashing import canonical_json_bytes, sha256_file
from document_ocr.synthesis.scenario_views import (
    PARTY_STRUCTURE_COLUMNS,
    party_structure_driver_violations,
)
from document_ocr.synthesis.sdv_evaluation import (
    EvaluationBundle,
    SdvEvaluationError,
    evaluate_single_table,
)
from document_ocr.synthesis.sdv_harness import (
    AcceleratorSelection,
    BenchmarkView,
    CandidateName,
    CandidateSpec,
    FittedCandidateModel,
    GroupedFold,
    PhaseResources,
    ProposalReceipt,
    dataframe_sha256,
    fit_candidate_model,
    grouped_folds,
    preflight_sdv_accelerator,
)

BenchmarkScope = Literal["cpu_baselines", "full_gpu"]
_EXPECTED_CANDIDATES: Mapping[BenchmarkScope, tuple[CandidateName, ...]] = MappingProxyType(
    {
        "cpu_baselines": ("empirical", "gaussian_copula"),
        "full_gpu": ("empirical", "gaussian_copula", "ctgan", "tvae"),
    }
)
_GAUSSIAN_PARAMETERS = frozenset(
    {
        "enforce_min_max_values",
        "enforce_rounding",
        "locales",
        "numerical_distributions",
        "default_distribution",
    }
)
_CTGAN_PARAMETERS = frozenset(
    {
        "enforce_min_max_values",
        "enforce_rounding",
        "locales",
        "embedding_dim",
        "generator_dim",
        "discriminator_dim",
        "generator_lr",
        "generator_decay",
        "discriminator_lr",
        "discriminator_decay",
        "batch_size",
        "discriminator_steps",
        "log_frequency",
        "verbose",
        "epochs",
        "pac",
        "enable_gpu",
    }
)
_TVAE_PARAMETERS = frozenset(
    {
        "enforce_min_max_values",
        "enforce_rounding",
        "embedding_dim",
        "compress_dims",
        "decompress_dims",
        "l2scale",
        "batch_size",
        "verbose",
        "epochs",
        "loss_factor",
        "enable_gpu",
    }
)
_PARAMETER_KEYS: Mapping[CandidateName, frozenset[str]] = MappingProxyType(
    {
        "empirical": frozenset(),
        "gaussian_copula": _GAUSSIAN_PARAMETERS,
        "ctgan": _CTGAN_PARAMETERS,
        "tvae": _TVAE_PARAMETERS,
    }
)


class ScenarioBenchmarkError(RuntimeError):
    """The party scenario benchmark violated its immutable experiment contract."""


def party_structure_candidate_specs(
    *,
    enable_gpu: bool,
    neural_epochs: int,
    neural_batch_size: int,
) -> tuple[CandidateSpec, ...]:
    """Return all four candidates with every pinned SDV parameter explicit."""

    if type(enable_gpu) is not bool:
        raise ValueError("enable_gpu must be boolean")
    if type(neural_epochs) is not int or neural_epochs < 1:
        raise ValueError("neural_epochs must be a positive integer")
    if type(neural_batch_size) is not int or neural_batch_size < 10 or neural_batch_size % 10:
        raise ValueError("neural_batch_size must be positive and divisible by CTGAN pac=10")
    bounded = {"enforce_min_max_values": True, "enforce_rounding": True}
    return (
        CandidateSpec("empirical", {}),
        CandidateSpec(
            "gaussian_copula",
            {
                **bounded,
                "locales": ["en_US"],
                "numerical_distributions": {},
                "default_distribution": "beta",
            },
        ),
        CandidateSpec(
            "ctgan",
            {
                **bounded,
                "locales": ["en_US"],
                "embedding_dim": 128,
                "generator_dim": [256, 256],
                "discriminator_dim": [256, 256],
                "generator_lr": 0.0002,
                "generator_decay": 0.000001,
                "discriminator_lr": 0.0002,
                "discriminator_decay": 0.000001,
                "batch_size": neural_batch_size,
                "discriminator_steps": 1,
                "log_frequency": True,
                "verbose": False,
                "epochs": neural_epochs,
                "pac": 10,
                "enable_gpu": enable_gpu,
            },
        ),
        CandidateSpec(
            "tvae",
            {
                **bounded,
                "embedding_dim": 128,
                "compress_dims": [128, 128],
                "decompress_dims": [128, 128],
                "l2scale": 0.00001,
                "batch_size": neural_batch_size,
                "verbose": False,
                "epochs": neural_epochs,
                "loss_factor": 2,
                "enable_gpu": enable_gpu,
            },
        ),
    )


@dataclass(frozen=True, slots=True)
class PartyStructureBenchmarkSettings:
    scope: BenchmarkScope
    candidates: tuple[CandidateSpec, ...]
    fold_count: int
    seeds: tuple[int, ...]
    fold_seed: int
    proposal_multiplier: int = 8
    proposal_batch_rows: int | None = None

    def __post_init__(self) -> None:
        expected = _EXPECTED_CANDIDATES.get(self.scope)
        if expected is None:
            raise ValueError(f"unsupported party benchmark scope: {self.scope!r}")
        names = tuple(candidate.name for candidate in self.candidates)
        if names != expected:
            raise ValueError(f"{self.scope} requires candidates in order {expected}, found {names}")
        for candidate in self.candidates:
            actual_keys = frozenset(candidate.parameters)
            if actual_keys != _PARAMETER_KEYS[candidate.name]:
                raise ValueError(
                    f"candidate {candidate.name} parameters are not fully explicit; "
                    f"expected={sorted(_PARAMETER_KEYS[candidate.name])}, "
                    f"found={sorted(actual_keys)}"
                )
        if self.scope == "full_gpu" and any(
            candidate.parameters.get("enable_gpu") is not True
            for candidate in self.candidates
            if candidate.name in {"ctgan", "tvae"}
        ):
            raise ValueError("full_gpu requires enable_gpu=true for CTGAN and TVAE")
        integer_values = {
            "fold_count": self.fold_count,
            "fold_seed": self.fold_seed,
            "proposal_multiplier": self.proposal_multiplier,
        }
        if any(type(value) is not int for value in integer_values.values()):
            raise ValueError("party benchmark integer settings must be strict integers")
        if self.fold_count < 2 or self.proposal_multiplier < 1:
            raise ValueError("fold_count must be >=2 and proposal_multiplier must be positive")
        if not 0 <= self.fold_seed < 2**32:
            raise ValueError("fold_seed must be uint32")
        if (
            not self.seeds
            or len(self.seeds) != len(set(self.seeds))
            or any(type(seed) is not int or not 0 <= seed < 2**32 for seed in self.seeds)
        ):
            raise ValueError("seeds must be unique strict uint32 values")
        if self.proposal_batch_rows is not None and (
            type(self.proposal_batch_rows) is not int or self.proposal_batch_rows < 1
        ):
            raise ValueError("proposal_batch_rows must be a positive integer when supplied")


@dataclass(frozen=True, slots=True)
class DomainValidityReceipt:
    raw_proposals: int
    valid_proposals: int
    invalid_proposals: int
    output_rows: int
    violation_counts: Mapping[str, int]

    @property
    def raw_valid_fraction(self) -> float:
        return self.valid_proposals / self.raw_proposals

    def to_dict(self) -> dict[str, Any]:
        return {
            "raw_proposals": self.raw_proposals,
            "valid_proposals": self.valid_proposals,
            "invalid_proposals": self.invalid_proposals,
            "raw_valid_fraction": self.raw_valid_fraction,
            "output_rows": self.output_rows,
            "output_valid_fraction": 1.0,
            "violation_counts": dict(sorted(self.violation_counts.items())),
        }


@dataclass(frozen=True, slots=True)
class NoveltyReceipt:
    rows: int
    novel_rows: int
    exact_training_matches: int
    unique_rows: int

    def to_dict(self) -> dict[str, Any]:
        return {
            "rows": self.rows,
            "novel_rows": self.novel_rows,
            "exact_training_matches": self.exact_training_matches,
            "novel_fraction": self.novel_rows / self.rows,
            "unique_rows": self.unique_rows,
            "unique_fraction": self.unique_rows / self.rows,
        }


@dataclass(frozen=True, slots=True)
class PartyCandidateRunReceipt:
    candidate: CandidateName
    parameters: Mapping[str, Any]
    fold: GroupedFold
    seed: int
    train_row_membership_sha256: str
    validation_row_membership_sha256: str
    train_data_sha256: str
    validation_data_sha256: str
    synthetic_data_sha256: str
    fit: PhaseResources
    sample: PhaseResources
    evaluation_elapsed_seconds: float
    proposal: ProposalReceipt
    validity: DomainValidityReceipt
    novelty: NoveltyReceipt
    evaluation: EvaluationBundle

    def to_dict(self) -> dict[str, Any]:
        return {
            "candidate": self.candidate,
            "parameters": dict(self.parameters),
            "fold": self.fold.to_dict(),
            "seed": self.seed,
            "train_row_membership_sha256": self.train_row_membership_sha256,
            "validation_row_membership_sha256": self.validation_row_membership_sha256,
            "train_data_sha256": self.train_data_sha256,
            "validation_data_sha256": self.validation_data_sha256,
            "synthetic_data_sha256": self.synthetic_data_sha256,
            "fit": self.fit.to_dict(),
            "sample": self.sample.to_dict(),
            "evaluation_elapsed_seconds": self.evaluation_elapsed_seconds,
            "total_elapsed_seconds": (
                self.fit.elapsed_seconds
                + self.sample.elapsed_seconds
                + self.evaluation_elapsed_seconds
            ),
            "proposal": self.proposal.to_dict(),
            "validity": self.validity.to_dict(),
            "novelty": self.novelty.to_dict(),
            "evaluation": self.evaluation.to_dict(),
        }


@dataclass(frozen=True, slots=True)
class PartyCandidateFailureReceipt:
    candidate: CandidateName
    parameters: Mapping[str, Any]
    fold: GroupedFold
    seed: int
    train_row_membership_sha256: str
    validation_row_membership_sha256: str
    train_data_sha256: str
    validation_data_sha256: str
    synthetic_data_sha256: str
    fit: PhaseResources
    sample: PhaseResources
    evaluation_elapsed_seconds: float
    proposal: ProposalReceipt
    validity: DomainValidityReceipt
    novelty: NoveltyReceipt
    failure_stage: Literal["evaluation"]
    error_type: str
    message: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "candidate": self.candidate,
            "parameters": dict(self.parameters),
            "fold": self.fold.to_dict(),
            "seed": self.seed,
            "train_row_membership_sha256": self.train_row_membership_sha256,
            "validation_row_membership_sha256": self.validation_row_membership_sha256,
            "train_data_sha256": self.train_data_sha256,
            "validation_data_sha256": self.validation_data_sha256,
            "synthetic_data_sha256": self.synthetic_data_sha256,
            "fit": self.fit.to_dict(),
            "sample": self.sample.to_dict(),
            "evaluation_elapsed_seconds": self.evaluation_elapsed_seconds,
            "total_elapsed_seconds": (
                self.fit.elapsed_seconds
                + self.sample.elapsed_seconds
                + self.evaluation_elapsed_seconds
            ),
            "proposal": self.proposal.to_dict(),
            "validity": self.validity.to_dict(),
            "novelty": self.novelty.to_dict(),
            "failure_stage": self.failure_stage,
            "error_type": self.error_type,
            "message": self.message,
            "evaluation": None,
        }


class _CandidateEvaluationFailure(RuntimeError):
    def __init__(
        self,
        receipt: PartyCandidateFailureReceipt,
        error: SdvEvaluationError,
    ) -> None:
        super().__init__(str(error))
        self.receipt = receipt
        self.error = error


@dataclass(frozen=True, slots=True)
class CandidateSummary:
    candidate: CandidateName
    run_count: int
    mean_quality: float
    quality_stddev: float
    minimum_diagnostic: float
    raw_valid_fraction: float
    novel_fraction: float
    unique_fraction: float
    total_fit_seconds: float
    total_sample_seconds: float
    total_evaluation_seconds: float
    devices: tuple[str, ...]
    maximum_cuda_peak_allocated_bytes: int | None
    maximum_cuda_peak_reserved_bytes: int | None

    def to_dict(self) -> dict[str, Any]:
        return {
            "candidate": self.candidate,
            "run_count": self.run_count,
            "mean_quality": self.mean_quality,
            "quality_stddev": self.quality_stddev,
            "minimum_diagnostic": self.minimum_diagnostic,
            "raw_valid_fraction": self.raw_valid_fraction,
            "novel_fraction": self.novel_fraction,
            "unique_fraction": self.unique_fraction,
            "total_fit_seconds": self.total_fit_seconds,
            "total_sample_seconds": self.total_sample_seconds,
            "total_evaluation_seconds": self.total_evaluation_seconds,
            "devices": list(self.devices),
            "maximum_cuda_peak_allocated_bytes": self.maximum_cuda_peak_allocated_bytes,
            "maximum_cuda_peak_reserved_bytes": self.maximum_cuda_peak_reserved_bytes,
        }


@dataclass(frozen=True, slots=True)
class PartyStructureBenchmarkResult:
    scope: BenchmarkScope
    row_count: int
    template_group_count: int
    folds: tuple[GroupedFold, ...]
    runs: tuple[PartyCandidateRunReceipt, ...]
    summaries: tuple[CandidateSummary, ...]
    requested_candidates: tuple[CandidateName, ...]
    complete_candidates: tuple[CandidateName, ...]
    failed_candidates: tuple[CandidateName, ...]
    candidate_failures: tuple[PartyCandidateFailureReceipt, ...]
    production_selection_performed: Literal[False] = False

    def __post_init__(self) -> None:
        requested = set(self.requested_candidates)
        complete = set(self.complete_candidates)
        failed = set(self.failed_candidates)
        if (
            len(requested) != len(self.requested_candidates)
            or complete & failed
            or complete | failed != requested
            or tuple(summary.candidate for summary in self.summaries) != self.complete_candidates
            or tuple(failure.candidate for failure in self.candidate_failures)
            != self.failed_candidates
            or any(run.candidate not in requested for run in self.runs)
        ):
            raise ValueError("benchmark candidate completion accounting is inconsistent")

    @property
    def status(self) -> Literal["complete", "complete_with_candidate_failures"]:
        return "complete_with_candidate_failures" if self.failed_candidates else "complete"

    def to_dict(self) -> dict[str, Any]:
        return {
            "status": self.status,
            "scope": self.scope,
            "row_count": self.row_count,
            "template_group_count": self.template_group_count,
            "folds": [fold.to_dict() for fold in self.folds],
            "runs": [run.to_dict() for run in self.runs],
            "summaries": [summary.to_dict() for summary in self.summaries],
            "requested_candidates": list(self.requested_candidates),
            "complete_candidates": list(self.complete_candidates),
            "failed_candidates": list(self.failed_candidates),
            "candidate_failures": [failure.to_dict() for failure in self.candidate_failures],
            "production_selection_performed": self.production_selection_performed,
            "production_model_artifact": None,
        }


def _strict_scalar(value: Any) -> str | int | bool:
    item = getattr(value, "item", None)
    if callable(item):
        value = item()
    if isinstance(value, bool):
        return value
    if isinstance(value, (str, int)):
        return value
    raise ScenarioBenchmarkError(
        f"party structure contains a non-string/non-integer scalar: {type(value).__name__}"
    )


def _row_key(row: Sequence[Any]) -> bytes:
    return canonical_json_bytes([_strict_scalar(value) for value in row])


def _membership_sha256(values: Sequence[str]) -> str:
    return hashlib.sha256(canonical_json_bytes(list(values))).hexdigest()


def _validate_party_row(row: Mapping[str, Any]) -> tuple[str, ...]:
    return party_structure_driver_violations(row)


@dataclass(slots=True)
class _ValidityAccumulator:
    raw: int = 0
    valid: int = 0
    violations: Counter[str] | None = None

    def __post_init__(self) -> None:
        self.violations = Counter()

    def accept(self, data: Any) -> Sequence[bool]:
        mask: list[bool] = []
        for values in data.itertuples(index=False, name=None):
            row = dict(zip(PARTY_STRUCTURE_COLUMNS, values, strict=True))
            row_violations = _validate_party_row(row)
            self.raw += 1
            if row_violations:
                assert self.violations is not None
                self.violations.update(row_violations)
                mask.append(False)
            else:
                self.valid += 1
                mask.append(True)
        return mask

    def receipt(self, *, proposal: ProposalReceipt) -> DomainValidityReceipt:
        if self.raw != proposal.raw_proposals or self.valid != (
            proposal.accepted_proposals_before_truncation
        ):
            raise ScenarioBenchmarkError("validity accounting differs from proposal receipt")
        return DomainValidityReceipt(
            raw_proposals=self.raw,
            valid_proposals=self.valid,
            invalid_proposals=self.raw - self.valid,
            output_rows=proposal.output_rows,
            violation_counts=MappingProxyType(dict(self.violations or {})),
        )


def _novelty(synthetic: Any, train: Any) -> NoveltyReceipt:
    training = {_row_key(row) for row in train.itertuples(index=False, name=None)}
    synthetic_rows = [_row_key(row) for row in synthetic.itertuples(index=False, name=None)]
    source_matches = sum(row in training for row in synthetic_rows)
    return NoveltyReceipt(
        rows=len(synthetic_rows),
        novel_rows=len(synthetic_rows) - source_matches,
        exact_training_matches=source_matches,
        unique_rows=len(set(synthetic_rows)),
    )


def _candidate_summary(
    candidate: CandidateName,
    runs: Sequence[PartyCandidateRunReceipt],
) -> CandidateSummary:
    selected = [run for run in runs if run.candidate == candidate]
    if not selected:
        raise ScenarioBenchmarkError(f"candidate {candidate} has no completed runs")
    qualities = [run.evaluation.quality.score for run in selected]
    valid_raw = sum(run.validity.valid_proposals for run in selected)
    raw = sum(run.validity.raw_proposals for run in selected)
    novel = sum(run.novelty.novel_rows for run in selected)
    unique = sum(run.novelty.unique_rows for run in selected)
    output = sum(run.novelty.rows for run in selected)
    cuda_allocated = [
        value
        for run in selected
        for value in (
            run.fit.cuda_peak_allocated_bytes,
            run.sample.cuda_peak_allocated_bytes,
        )
        if value is not None
    ]
    cuda_reserved = [
        value
        for run in selected
        for value in (
            run.fit.cuda_peak_reserved_bytes,
            run.sample.cuda_peak_reserved_bytes,
        )
        if value is not None
    ]
    return CandidateSummary(
        candidate=candidate,
        run_count=len(selected),
        mean_quality=statistics.fmean(qualities),
        quality_stddev=statistics.pstdev(qualities),
        minimum_diagnostic=min(run.evaluation.diagnostic.score for run in selected),
        raw_valid_fraction=valid_raw / raw,
        novel_fraction=novel / output,
        unique_fraction=unique / output,
        total_fit_seconds=sum(run.fit.elapsed_seconds for run in selected),
        total_sample_seconds=sum(run.sample.elapsed_seconds for run in selected),
        total_evaluation_seconds=sum(run.evaluation_elapsed_seconds for run in selected),
        devices=tuple(sorted({run.fit.accelerator.device for run in selected})),
        maximum_cuda_peak_allocated_bytes=(max(cuda_allocated) if cuda_allocated else None),
        maximum_cuda_peak_reserved_bytes=(max(cuda_reserved) if cuda_reserved else None),
    )


def _prepare_artifact_directory(path: Path) -> None:
    if path.is_symlink():
        raise ScenarioBenchmarkError("artifact directory cannot be a symbolic link")
    if path.exists():
        if not path.is_dir():
            raise ScenarioBenchmarkError("artifact path exists and is not a directory")
        if any(path.iterdir()):
            raise ScenarioBenchmarkError("artifact directory must be new or empty")
    else:
        path.mkdir(parents=True)


def _publish_manifest(
    artifact_dir: Path,
    *,
    result: PartyStructureBenchmarkResult,
) -> None:
    files: list[dict[str, Any]] = []
    for path in sorted(artifact_dir.rglob("*")):
        if path.is_symlink():
            raise ScenarioBenchmarkError(f"artifact tree contains a symbolic link: {path}")
        if path.is_file() and path.name != "manifest.json":
            files.append(
                {
                    "path": str(path.relative_to(artifact_dir)),
                    "sha256": sha256_file(path),
                    "size_bytes": path.stat().st_size,
                }
            )
    atomic_publish_json(
        artifact_dir / "manifest.json",
        {
            "status": result.status,
            "experiment": "party_structure_comparative_benchmark_v1",
            "requested_candidates": list(result.requested_candidates),
            "complete_candidates": list(result.complete_candidates),
            "failed_candidates": list(result.failed_candidates),
            "production_selection_performed": False,
            "files": files,
        },
    )


def _assert_party_view(view: BenchmarkView, settings: PartyStructureBenchmarkSettings) -> None:
    if view.name != "party_structure" or tuple(view.data.columns) != PARTY_STRUCTURE_COLUMNS:
        raise ValueError("party benchmark accepts only the exact PII-free party_structure view")
    if settings.fold_count > len(set(view.group_ids)):
        raise ValueError("fold_count exceeds party_structure template groups")
    for values in view.data.itertuples(index=False, name=None):
        row = dict(zip(PARTY_STRUCTURE_COLUMNS, values, strict=True))
        violations = _validate_party_row(row)
        if violations:
            raise ValueError(f"source party_structure row violates its contract: {violations}")


def _run_candidate(
    *,
    view: BenchmarkView,
    candidate: CandidateSpec,
    fold: GroupedFold,
    seed: int,
    settings: PartyStructureBenchmarkSettings,
) -> PartyCandidateRunReceipt:
    train = view.data.iloc[list(fold.train_indices)].reset_index(drop=True).copy()
    validation = view.data.iloc[list(fold.validation_indices)].reset_index(drop=True).copy()
    fitted: FittedCandidateModel = fit_candidate_model(
        view=view,
        spec=candidate,
        train_data=train,
        seed=seed,
    )
    if fitted.candidate != candidate.name or dict(fitted.parameters) != dict(candidate.parameters):
        raise ScenarioBenchmarkError(
            f"fitted candidate identity differs from requested candidate {candidate.name}"
        )
    validity = _ValidityAccumulator()
    synthetic, proposal, sample_resources = fitted.sample(
        requested_rows=len(validation),
        seed=seed,
        columns=PARTY_STRUCTURE_COLUMNS,
        acceptance=validity.accept,
        proposal_multiplier=settings.proposal_multiplier,
        proposal_batch_rows=settings.proposal_batch_rows,
    )
    validity_receipt = validity.receipt(proposal=proposal)
    train_membership_sha256 = _membership_sha256(
        tuple(view.row_ids[index] for index in fold.train_indices)
    )
    validation_membership_sha256 = _membership_sha256(
        tuple(view.row_ids[index] for index in fold.validation_indices)
    )
    train_data_sha256 = dataframe_sha256(train)
    validation_data_sha256 = dataframe_sha256(validation)
    synthetic_data_sha256 = dataframe_sha256(synthetic)
    novelty = _novelty(synthetic, train)
    evaluation_started = time.perf_counter()
    try:
        evaluation = evaluate_single_table(
            real_data=validation,
            synthetic_data=synthetic,
            metadata=view.metadata,
            table_name=view.name,
            diagnostic_reference_data=train,
        )
    except SdvEvaluationError as error:
        evaluation_elapsed = time.perf_counter() - evaluation_started
        failure = PartyCandidateFailureReceipt(
            candidate=candidate.name,
            parameters=candidate.parameters,
            fold=fold,
            seed=seed,
            train_row_membership_sha256=train_membership_sha256,
            validation_row_membership_sha256=validation_membership_sha256,
            train_data_sha256=train_data_sha256,
            validation_data_sha256=validation_data_sha256,
            synthetic_data_sha256=synthetic_data_sha256,
            fit=fitted.fit,
            sample=sample_resources,
            evaluation_elapsed_seconds=evaluation_elapsed,
            proposal=proposal,
            validity=validity_receipt,
            novelty=novelty,
            failure_stage="evaluation",
            error_type=type(error).__name__,
            message=str(error),
        )
        raise _CandidateEvaluationFailure(failure, error) from error
    evaluation_elapsed = time.perf_counter() - evaluation_started
    if not math.isfinite(evaluation_elapsed) or evaluation_elapsed < 0:
        raise ScenarioBenchmarkError("evaluation runtime is invalid")
    return PartyCandidateRunReceipt(
        candidate=candidate.name,
        parameters=candidate.parameters,
        fold=fold,
        seed=seed,
        train_row_membership_sha256=train_membership_sha256,
        validation_row_membership_sha256=validation_membership_sha256,
        train_data_sha256=train_data_sha256,
        validation_data_sha256=validation_data_sha256,
        synthetic_data_sha256=synthetic_data_sha256,
        fit=fitted.fit,
        sample=sample_resources,
        evaluation_elapsed_seconds=evaluation_elapsed,
        proposal=proposal,
        validity=validity_receipt,
        novelty=novelty,
        evaluation=evaluation,
    )


def run_party_structure_benchmark(
    *,
    view: BenchmarkView,
    settings: PartyStructureBenchmarkSettings,
    artifact_dir: Path,
) -> PartyStructureBenchmarkResult:
    """Run a no-selection comparison on identical template-grouped folds."""

    _assert_party_view(view, settings)
    _prepare_artifact_directory(artifact_dir)
    completed: list[PartyCandidateRunReceipt] = []
    try:
        folds = grouped_folds(
            group_ids=view.group_ids,
            fold_count=settings.fold_count,
            seed=settings.fold_seed,
        )
        if any(set(fold.train_groups) & set(fold.validation_groups) for fold in folds):
            raise RuntimeError("grouped fold construction leaked a template")
        implementation = {
            "document_ocr/synthesis/scenario_benchmark.py": sha256_file(Path(__file__)),
            "document_ocr/synthesis/scenario_views.py": sha256_file(
                Path(__file__).with_name("scenario_views.py")
            ),
            "document_ocr/synthesis/sdv_harness.py": sha256_file(
                Path(__file__).with_name("sdv_harness.py")
            ),
            "document_ocr/synthesis/sdv_evaluation.py": sha256_file(
                Path(__file__).with_name("sdv_evaluation.py")
            ),
        }
        contract = {
            "experiment": "party_structure_comparative_benchmark_v1",
            "scope": settings.scope,
            "view": view.name,
            "rows": len(view.data),
            "template_groups": len(set(view.group_ids)),
            "allowed_partition": view.allowed_partition,
            "columns": list(PARTY_STRUCTURE_COLUMNS),
            "view_data_sha256": dataframe_sha256(view.data),
            "row_membership_sha256": hashlib.sha256(
                canonical_json_bytes(
                    [
                        {"row": row_id, "group": group_id, "partition": partition}
                        for row_id, group_id, partition in zip(
                            view.row_ids,
                            view.group_ids,
                            view.partition_labels,
                            strict=True,
                        )
                    ]
                )
            ).hexdigest(),
            "candidates": [
                {"name": candidate.name, "parameters": dict(candidate.parameters)}
                for candidate in settings.candidates
            ],
            "settings": {
                "fold_count": settings.fold_count,
                "fold_seed": settings.fold_seed,
                "seeds": list(settings.seeds),
                "proposal_multiplier": settings.proposal_multiplier,
                "proposal_batch_rows": settings.proposal_batch_rows,
            },
            "folds": [fold.to_dict() for fold in folds],
            "implementation": implementation,
            "environment": {
                "python": sys.version,
                "sdv": version("sdv"),
                "sdmetrics": version("sdmetrics"),
            },
            "contains_raw_party_pii": False,
            "candidate_failure_policy": {
                "non_empirical_evaluation": "receipt_failure_and_continue_next_candidate",
                "empirical_or_infrastructure": "abort_benchmark",
                "failed_candidate_partial_summary": False,
            },
            "production_selection_performed": False,
        }
        atomic_publish_json(artifact_dir / "benchmark-contract.json", contract)
        if settings.scope == "full_gpu":
            accelerator: AcceleratorSelection = preflight_sdv_accelerator(enable_gpu=True)
            if not accelerator.enable_gpu:
                raise ScenarioBenchmarkError("full_gpu preflight did not select a GPU")
            atomic_publish_json(artifact_dir / "gpu-preflight.json", accelerator.to_dict())
        candidate_failures: list[PartyCandidateFailureReceipt] = []
        complete_candidates: list[CandidateName] = []
        for candidate in settings.candidates:
            candidate_failed = False
            for fold in folds:
                for seed in settings.seeds:
                    try:
                        receipt = _run_candidate(
                            view=view,
                            candidate=candidate,
                            fold=fold,
                            seed=seed,
                            settings=settings,
                        )
                    except _CandidateEvaluationFailure as error:
                        if candidate.name == "empirical":
                            raise error.error from error
                        candidate_failures.append(error.receipt)
                        atomic_publish_json(
                            artifact_dir
                            / "candidate-failures"
                            / candidate.name
                            / f"fold-{fold.index:02d}-seed-{seed}.json",
                            error.receipt.to_dict(),
                        )
                        candidate_failed = True
                        break
                    completed.append(receipt)
                    atomic_publish_json(
                        artifact_dir
                        / "run-receipts"
                        / candidate.name
                        / f"fold-{fold.index:02d}-seed-{seed}.json",
                        receipt.to_dict(),
                    )
                if candidate_failed:
                    break
            if not candidate_failed:
                complete_candidates.append(candidate.name)
        failed_candidates = tuple(failure.candidate for failure in candidate_failures)
        summaries = tuple(
            _candidate_summary(candidate, completed) for candidate in complete_candidates
        )
        result = PartyStructureBenchmarkResult(
            scope=settings.scope,
            row_count=len(view.data),
            template_group_count=len(set(view.group_ids)),
            folds=folds,
            runs=tuple(completed),
            summaries=summaries,
            requested_candidates=tuple(candidate.name for candidate in settings.candidates),
            complete_candidates=tuple(complete_candidates),
            failed_candidates=failed_candidates,
            candidate_failures=tuple(candidate_failures),
        )
        atomic_publish_json(artifact_dir / "result.json", result.to_dict())
        _publish_manifest(artifact_dir, result=result)
        return result
    except Exception as error:
        atomic_publish_json(
            artifact_dir / "failure.json",
            {
                "status": "failed",
                "experiment": "party_structure_comparative_benchmark_v1",
                "completed_runs": len(completed),
                "error_type": type(error).__name__,
                "message": str(error),
            },
        )
        raise ScenarioBenchmarkError(
            f"party structure benchmark failed after {len(completed)} runs: "
            f"{type(error).__name__}: {error}"
        ) from error
