"""Train-only, grouped SDV benchmarking for compact synthesis modeling views.

This module intentionally benchmarks independent single-table views rather than
pretending that Community SDV can faithfully model the full 17-table B/L graph.
Statistical models propose values; task-owned deterministic code remains
responsible for relations, registries, identifiers, arithmetic, and chronology.
"""

from __future__ import annotations

import contextlib
import datetime as dt
import decimal
import hashlib
import json
import math
import os
import platform
import random
import resource
import stat
import tempfile
import time
from collections import Counter
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from importlib import import_module
from importlib.metadata import version
from pathlib import Path
from types import MappingProxyType
from typing import Any, Literal

from document_ocr.atomic import (
    AtomicConflictError,
    atomic_publish_json,
    json_artifact_bytes,
    read_regular_file_bytes,
)
from document_ocr.hashing import canonical_json_bytes, identity_sha256, sha256_file
from document_ocr.synthesis.sdv_evaluation import (
    CandidateRunScore,
    EvaluationBundle,
    ModelSelection,
    evaluate_single_table,
    select_candidate,
)

CandidateName = Literal["empirical", "gaussian_copula", "ctgan", "tvae"]
SUPPORTED_CANDIDATES = frozenset({"empirical", "gaussian_copula", "ctgan", "tvae"})
DEFAULT_COMPLEXITY_RANKS: Mapping[str, int] = MappingProxyType(
    {"empirical": 0, "gaussian_copula": 1, "ctgan": 2, "tvae": 2}
)
SEED_CONTRACT = "isolated_python_numpy_torch_cuda_fit_and_pinned_sdv_sample_state_v2"


class SdvBenchmarkError(RuntimeError):
    """A benchmark input or candidate execution violates the audited contract."""


class ProposalYieldError(SdvBenchmarkError):
    """Bounded proposal sampling could not yield the requested accepted rows."""

    def __init__(self, message: str, *, raw_proposals: int, accepted_proposals: int) -> None:
        super().__init__(message)
        self.raw_proposals = raw_proposals
        self.accepted_proposals = accepted_proposals


@dataclass(frozen=True, slots=True)
class BenchmarkView:
    """One train-only, single-table statistical modeling view."""

    name: str
    data: Any
    metadata: Mapping[str, Any]
    row_ids: tuple[str, ...]
    group_ids: tuple[str, ...]
    partition_labels: tuple[str, ...]
    allowed_partition: str = "train"

    def __post_init__(self) -> None:
        if not self.name or not self.allowed_partition:
            raise ValueError("view name and allowed partition must be non-empty")
        row_count = len(self.data)
        if row_count < 2:
            raise ValueError("a benchmark view requires at least two rows")
        for label, values in (
            ("row_ids", self.row_ids),
            ("group_ids", self.group_ids),
            ("partition_labels", self.partition_labels),
        ):
            if len(values) != row_count:
                raise ValueError(f"{label} length differs from view row count")
            if any(not value for value in values):
                raise ValueError(f"{label} contains an empty value")
        if len(set(self.row_ids)) != row_count:
            raise ValueError("benchmark row IDs must be unique")
        unexpected = sorted(set(self.partition_labels) - {self.allowed_partition})
        if unexpected:
            raise ValueError(
                "benchmark fit data is not train-only; unexpected partitions: "
                + ", ".join(unexpected)
            )
        if len(set(self.group_ids)) < 2:
            raise ValueError("grouped evaluation requires at least two source/template groups")
        columns = tuple(str(column) for column in self.data.columns)
        if len(columns) != len(set(columns)) or any(not column for column in columns):
            raise ValueError("benchmark columns must be non-empty and unique")

        # Make metadata independent of caller mutation and prove it is strict
        # JSON before SDV sees it.
        try:
            canonical_json_bytes(dict(self.metadata))
            metadata_copy = json.loads(
                json.dumps(dict(self.metadata), allow_nan=False, ensure_ascii=False)
            )
        except (TypeError, ValueError) as error:
            raise ValueError("view metadata must be finite JSON") from error
        tables = metadata_copy.get("tables")
        if not isinstance(tables, dict) or set(tables) != {self.name}:
            raise ValueError("view metadata must describe exactly the named table")
        table = tables[self.name]
        metadata_columns = table.get("columns") if isinstance(table, dict) else None
        if not isinstance(metadata_columns, dict) or tuple(metadata_columns) != columns:
            raise ValueError("metadata columns differ from or reorder view columns")
        if metadata_copy.get("relationships") not in (None, []):
            raise ValueError("single-table benchmark views cannot contain relationships")
        object.__setattr__(self, "metadata", MappingProxyType(metadata_copy))


@dataclass(frozen=True, slots=True)
class CandidateSpec:
    name: CandidateName
    parameters: Mapping[str, Any]

    def __post_init__(self) -> None:
        if self.name not in SUPPORTED_CANDIDATES:
            raise ValueError(f"unsupported SDV benchmark candidate: {self.name}")
        try:
            parameters = json.loads(canonical_json_bytes(dict(self.parameters)))
        except (TypeError, ValueError) as error:
            raise ValueError(f"candidate {self.name} parameters must be finite JSON") from error
        if self.name == "empirical" and parameters:
            raise ValueError("empirical candidate accepts no parameters")
        if "cuda" in parameters:
            raise ValueError(
                f"candidate {self.name} must use enable_gpu; SDV's cuda parameter is deprecated"
            )
        if self.name in {"ctgan", "tvae"} and type(parameters.get("enable_gpu")) is not bool:
            raise ValueError(f"candidate {self.name} must explicitly set boolean enable_gpu")
        if self.name not in {"ctgan", "tvae"} and "enable_gpu" in parameters:
            raise ValueError(f"candidate {self.name} does not support enable_gpu")
        object.__setattr__(self, "parameters", MappingProxyType(parameters))


@dataclass(frozen=True, slots=True)
class BenchmarkSettings:
    fold_count: int
    seeds: tuple[int, ...]
    fold_seed: int = 0
    quality_margin: float = 0.01
    stability_penalty: float = 0.25
    proposal_multiplier: int = 1
    proposal_batch_rows: int | None = None
    persist_fold_models: bool = False
    fit_selected_model: bool = True
    selectable_candidates: tuple[CandidateName, ...] | None = None

    def __post_init__(self) -> None:
        if self.fold_count < 2:
            raise ValueError("fold_count must be at least two")
        if not self.seeds or len(self.seeds) != len(set(self.seeds)):
            raise ValueError("benchmark seeds must be non-empty and unique")
        if any(seed < 0 or seed >= 2**32 for seed in self.seeds):
            raise ValueError("benchmark seeds must be uint32 values")
        if self.fold_seed < 0 or self.fold_seed >= 2**32:
            raise ValueError("fold_seed must be a uint32 value")
        if not math.isfinite(self.quality_margin) or self.quality_margin < 0:
            raise ValueError("quality_margin must be finite and non-negative")
        if not math.isfinite(self.stability_penalty) or self.stability_penalty < 0:
            raise ValueError("stability_penalty must be finite and non-negative")
        if self.proposal_multiplier < 1:
            raise ValueError("proposal_multiplier must be positive")
        if self.proposal_batch_rows is not None and self.proposal_batch_rows < 1:
            raise ValueError("proposal_batch_rows must be positive when supplied")
        if self.selectable_candidates is not None and (
            not self.selectable_candidates
            or len(self.selectable_candidates) != len(set(self.selectable_candidates))
        ):
            raise ValueError("selectable_candidates must be non-empty and unique when supplied")


@dataclass(frozen=True, slots=True)
class GroupedFold:
    index: int
    train_indices: tuple[int, ...]
    validation_indices: tuple[int, ...]
    train_groups: tuple[str, ...]
    validation_groups: tuple[str, ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            "index": self.index,
            "train_rows": len(self.train_indices),
            "validation_rows": len(self.validation_indices),
            "train_groups": list(self.train_groups),
            "validation_groups": list(self.validation_groups),
        }


@dataclass(frozen=True, slots=True)
class AcceleratorSelection:
    """Audited Torch device selected before an SDV phase begins."""

    enable_gpu: bool
    torch_version: str
    torch_cuda_version: str | None
    cuda_available: bool
    cuda_device_count: int
    device: str
    device_name: str | None
    device_capability: tuple[int, int] | None
    device_total_memory_bytes: int | None

    def __post_init__(self) -> None:
        if type(self.enable_gpu) is not bool or type(self.cuda_available) is not bool:
            raise ValueError("accelerator boolean fields must be booleans")
        if (
            not isinstance(self.torch_version, str)
            or not self.torch_version
            or type(self.cuda_device_count) is not int
            or self.cuda_device_count < 0
        ):
            raise ValueError("accelerator Torch version/count is invalid")
        if self.torch_cuda_version is not None and (
            not isinstance(self.torch_cuda_version, str) or not self.torch_cuda_version
        ):
            raise ValueError("accelerator Torch CUDA version is invalid")
        if self.cuda_available != (self.cuda_device_count > 0):
            raise ValueError("CUDA availability and device count are inconsistent")
        if self.enable_gpu:
            try:
                device_index = int(self.device.removeprefix("cuda:"))
            except (AttributeError, ValueError):
                device_index = -1
            if (
                not self.cuda_available
                or self.torch_cuda_version is None
                or self.device != f"cuda:{device_index}"
                or not 0 <= device_index < self.cuda_device_count
                or not isinstance(self.device_name, str)
                or not self.device_name
                or self.device_capability is None
                or len(self.device_capability) != 2
                or any(type(value) is not int or value < 0 for value in self.device_capability)
                or type(self.device_total_memory_bytes) is not int
                or self.device_total_memory_bytes <= 0
            ):
                raise ValueError("GPU selection lacks a complete CUDA device receipt")
        elif (
            self.device != "cpu"
            or self.device_name is not None
            or self.device_capability is not None
            or self.device_total_memory_bytes is not None
        ):
            raise ValueError("CPU selection contains CUDA device-specific values")

    def to_dict(self) -> dict[str, Any]:
        return {
            "enable_gpu": self.enable_gpu,
            "torch_version": self.torch_version,
            "torch_cuda_version": self.torch_cuda_version,
            "cuda_available": self.cuda_available,
            "cuda_device_count": self.cuda_device_count,
            "device": self.device,
            "device_name": self.device_name,
            "device_capability": (
                list(self.device_capability) if self.device_capability is not None else None
            ),
            "device_total_memory_bytes": self.device_total_memory_bytes,
        }


@dataclass(frozen=True, slots=True)
class PhaseResources:
    elapsed_seconds: float
    rss_before_bytes: int
    rss_after_bytes: int
    peak_rss_before_bytes: int
    peak_rss_after_bytes: int
    accelerator: AcceleratorSelection
    cuda_allocated_before_bytes: int | None
    cuda_allocated_after_bytes: int | None
    cuda_reserved_before_bytes: int | None
    cuda_reserved_after_bytes: int | None
    cuda_peak_allocated_bytes: int | None
    cuda_peak_reserved_bytes: int | None

    def __post_init__(self) -> None:
        if (
            type(self.elapsed_seconds) not in (int, float)
            or not math.isfinite(float(self.elapsed_seconds))
            or self.elapsed_seconds < 0
        ):
            raise ValueError("phase elapsed time must be finite and non-negative")
        rss_values = (
            self.rss_before_bytes,
            self.rss_after_bytes,
            self.peak_rss_before_bytes,
            self.peak_rss_after_bytes,
        )
        if any(type(value) is not int or value < 0 for value in rss_values):
            raise ValueError("phase RSS values must be non-negative integers")
        cuda_values = (
            self.cuda_allocated_before_bytes,
            self.cuda_allocated_after_bytes,
            self.cuda_reserved_before_bytes,
            self.cuda_reserved_after_bytes,
            self.cuda_peak_allocated_bytes,
            self.cuda_peak_reserved_bytes,
        )
        if self.accelerator.enable_gpu:
            if any(type(value) is not int or value < 0 for value in cuda_values):
                raise ValueError("GPU phase must contain non-negative CUDA memory values")
            assert self.cuda_peak_allocated_bytes is not None
            assert self.cuda_peak_reserved_bytes is not None
            if self.cuda_peak_allocated_bytes < max(
                self.cuda_allocated_before_bytes or 0,
                self.cuda_allocated_after_bytes or 0,
            ) or self.cuda_peak_reserved_bytes < max(
                self.cuda_reserved_before_bytes or 0,
                self.cuda_reserved_after_bytes or 0,
            ):
                raise ValueError("CUDA peak memory is lower than an observed phase snapshot")
        elif any(value is not None for value in cuda_values):
            raise ValueError("CPU phase cannot contain CUDA memory values")

    def to_dict(self) -> dict[str, Any]:
        return {
            "elapsed_seconds": self.elapsed_seconds,
            "rss_before_bytes": self.rss_before_bytes,
            "rss_after_bytes": self.rss_after_bytes,
            "peak_rss_before_bytes": self.peak_rss_before_bytes,
            "peak_rss_after_bytes": self.peak_rss_after_bytes,
            "peak_rss_increase_bytes": max(
                0, self.peak_rss_after_bytes - self.peak_rss_before_bytes
            ),
            "accelerator": self.accelerator.to_dict(),
            "cuda_allocated_before_bytes": self.cuda_allocated_before_bytes,
            "cuda_allocated_after_bytes": self.cuda_allocated_after_bytes,
            "cuda_reserved_before_bytes": self.cuda_reserved_before_bytes,
            "cuda_reserved_after_bytes": self.cuda_reserved_after_bytes,
            "cuda_peak_allocated_bytes": self.cuda_peak_allocated_bytes,
            "cuda_peak_reserved_bytes": self.cuda_peak_reserved_bytes,
        }


@dataclass(frozen=True, slots=True)
class ProposalReceipt:
    requested_rows: int
    raw_proposals: int
    accepted_proposals_before_truncation: int
    rejected_proposals: int
    output_rows: int
    acceptance_yield: float
    bounded_maximum_raw_proposals: int

    def to_dict(self) -> dict[str, Any]:
        return {
            "requested_rows": self.requested_rows,
            "raw_proposals": self.raw_proposals,
            "accepted_proposals_before_truncation": self.accepted_proposals_before_truncation,
            "rejected_proposals": self.rejected_proposals,
            "output_rows": self.output_rows,
            "acceptance_yield": self.acceptance_yield,
            "bounded_maximum_raw_proposals": self.bounded_maximum_raw_proposals,
        }


@dataclass(frozen=True, slots=True)
class ModelArtifactReceipt:
    path: str
    sha256: str
    size_bytes: int
    serialization: str
    created: bool

    def __post_init__(self) -> None:
        path = Path(self.path)
        if (
            not self.path
            or path.is_absolute()
            or any(part in {"", ".", ".."} for part in path.parts)
        ):
            raise ValueError("model artifact receipt path must be safe and relative")
        if len(self.sha256) != 64 or any(
            character not in "0123456789abcdef" for character in self.sha256
        ):
            raise ValueError("model artifact receipt SHA-256 is invalid")
        if self.size_bytes < 0:
            raise ValueError("model artifact receipt size must be non-negative")

    def to_dict(self) -> dict[str, Any]:
        return {
            "path": self.path,
            "sha256": self.sha256,
            "size_bytes": self.size_bytes,
            "serialization": self.serialization,
            "created": self.created,
        }


@dataclass(frozen=True, slots=True)
class CandidateRunReceipt:
    candidate: str
    parameters: Mapping[str, Any]
    fold: GroupedFold
    seed: int
    train_row_ids_sha256: str
    validation_row_ids_sha256: str
    train_data_sha256: str
    validation_data_sha256: str
    synthetic_data_sha256: str
    fit: PhaseResources
    sample: PhaseResources
    proposal: ProposalReceipt
    evaluation: EvaluationBundle
    model_artifact: ModelArtifactReceipt | None
    sdv_version: str
    sdmetrics_version: str
    seed_contract: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "candidate": self.candidate,
            "parameters": dict(self.parameters),
            "fold": self.fold.to_dict(),
            "seed": self.seed,
            "train_row_ids_sha256": self.train_row_ids_sha256,
            "validation_row_ids_sha256": self.validation_row_ids_sha256,
            "train_data_sha256": self.train_data_sha256,
            "validation_data_sha256": self.validation_data_sha256,
            "synthetic_data_sha256": self.synthetic_data_sha256,
            "fit": self.fit.to_dict(),
            "sample": self.sample.to_dict(),
            "proposal": self.proposal.to_dict(),
            "evaluation": self.evaluation.to_dict(),
            "model_artifact": (
                self.model_artifact.to_dict() if self.model_artifact is not None else None
            ),
            "sdv_version": self.sdv_version,
            "sdmetrics_version": self.sdmetrics_version,
            "seed_contract": self.seed_contract,
        }


@dataclass(frozen=True, slots=True)
class SelectedModelReceipt:
    candidate: str
    seed: int
    full_train_data_sha256: str
    fit: PhaseResources
    artifact: ModelArtifactReceipt

    def to_dict(self) -> dict[str, Any]:
        return {
            "candidate": self.candidate,
            "seed": self.seed,
            "full_train_data_sha256": self.full_train_data_sha256,
            "fit": self.fit.to_dict(),
            "artifact": self.artifact.to_dict(),
        }


@dataclass(frozen=True, slots=True)
class ViewBenchmarkResult:
    view_name: str
    row_count: int
    group_count: int
    allowed_partition: str
    folds: tuple[GroupedFold, ...]
    runs: tuple[CandidateRunReceipt, ...]
    selection: ModelSelection
    selected_model: SelectedModelReceipt | None

    def to_dict(self) -> dict[str, Any]:
        return {
            "view_name": self.view_name,
            "row_count": self.row_count,
            "group_count": self.group_count,
            "allowed_partition": self.allowed_partition,
            "folds": [fold.to_dict() for fold in self.folds],
            "runs": [run.to_dict() for run in self.runs],
            "selection": self.selection.to_dict(),
            "selected_model": (
                self.selected_model.to_dict() if self.selected_model is not None else None
            ),
        }


@dataclass(frozen=True, slots=True)
class SelectedModelSampleReceipt:
    request_id: str
    candidate: str
    model_sha256: str
    seed: int
    acceptance_contract: str
    synthetic_data_sha256: str
    proposal: ProposalReceipt

    def to_dict(self) -> dict[str, Any]:
        return {
            "request_id": self.request_id,
            "candidate": self.candidate,
            "model_sha256": self.model_sha256,
            "seed": self.seed,
            "acceptance_contract": self.acceptance_contract,
            "synthetic_data_sha256": self.synthetic_data_sha256,
            "proposal": self.proposal.to_dict(),
        }


@dataclass(slots=True)
class LoadedSelectedModel:
    """Verified reusable model handle for many independent proposal requests."""

    candidate: str
    model_sha256: str
    table_name: str
    columns: tuple[str, ...]
    _model: Any

    def sample(
        self,
        *,
        request_id: str,
        num_rows: int,
        seed: int,
        acceptance: Callable[[Any], Sequence[bool]] | None = None,
        acceptance_contract: str = "accept_all_v1",
        proposal_multiplier: int = 1,
        proposal_batch_rows: int | None = None,
    ) -> tuple[Any, SelectedModelSampleReceipt]:
        if not request_id:
            raise ValueError("sample request_id must be non-empty")
        if seed < 0 or seed >= 2**32:
            raise ValueError("sample seed must be a uint32 value")
        if not acceptance_contract or (
            acceptance is not None and acceptance_contract == "accept_all_v1"
        ):
            raise ValueError(
                "a custom acceptance callable requires a non-default acceptance_contract"
            )
        data, proposal = bounded_sample(
            model=self._model,
            requested_rows=num_rows,
            seed=seed,
            columns=self.columns,
            acceptance=acceptance,
            proposal_multiplier=proposal_multiplier,
            proposal_batch_rows=proposal_batch_rows,
        )
        receipt = SelectedModelSampleReceipt(
            request_id=request_id,
            candidate=self.candidate,
            model_sha256=self.model_sha256,
            seed=seed,
            acceptance_contract=acceptance_contract,
            synthetic_data_sha256=dataframe_sha256(data),
            proposal=proposal,
        )
        return data, receipt


@dataclass(slots=True)
class FittedCandidateModel:
    """One audited fitted candidate exposed for focused benchmark extensions.

    The harness retains ownership of model construction, deterministic fit
    seeding, accelerator verification, bounded sampling, and resource
    measurement.  Task-specific benchmarks can therefore add domain validity
    or novelty metrics without duplicating those safety-critical paths.
    """

    candidate: CandidateName
    parameters: Mapping[str, Any]
    fit: PhaseResources
    _model: Any

    def sample(
        self,
        *,
        requested_rows: int,
        seed: int,
        columns: Sequence[str],
        acceptance: Callable[[Any], Sequence[bool]] | None,
        proposal_multiplier: int,
        proposal_batch_rows: int | None,
    ) -> tuple[Any, ProposalReceipt, PhaseResources]:
        """Run the existing bounded sampler with audited resource accounting."""

        def operation() -> tuple[Any, ProposalReceipt]:
            return bounded_sample(
                model=self._model,
                requested_rows=requested_rows,
                seed=seed,
                columns=columns,
                acceptance=acceptance,
                proposal_multiplier=proposal_multiplier,
                proposal_batch_rows=proposal_batch_rows,
            )

        sampled, resources = _measure(operation, accelerator=self._model.accelerator)
        data, proposal = sampled
        return data, proposal, resources


def default_candidate_specs(
    *, neural_epochs: int = 300, enable_gpu: bool = False
) -> tuple[CandidateSpec, ...]:
    """Return the complete Community-SDV comparison set.

    The official 300-epoch neural default is retained. Smaller values are for
    explicit smoke tests only and remain visible in every receipt. ``enable_gpu``
    is strict: requesting it requires a working CUDA runtime and the fitted SDV
    neural model must prove that it selected that CUDA device.
    """

    if neural_epochs < 1:
        raise ValueError("neural_epochs must be positive")
    if type(enable_gpu) is not bool:
        raise ValueError("enable_gpu must be boolean")
    shared = {"enforce_min_max_values": True, "enforce_rounding": True}
    return (
        CandidateSpec("empirical", {}),
        CandidateSpec("gaussian_copula", shared),
        CandidateSpec(
            "ctgan",
            {**shared, "epochs": neural_epochs, "enable_gpu": enable_gpu, "verbose": False},
        ),
        CandidateSpec(
            "tvae",
            {**shared, "epochs": neural_epochs, "enable_gpu": enable_gpu, "verbose": False},
        ),
    )


def grouped_folds(
    *, group_ids: Sequence[str], fold_count: int, seed: int
) -> tuple[GroupedFold, ...]:
    """Build deterministic, row-balanced folds without crossing a group."""

    if fold_count < 2:
        raise ValueError("fold_count must be at least two")
    counts = Counter(group_ids)
    if len(counts) < fold_count:
        raise ValueError("fold_count exceeds unique source/template groups")
    if any(not group for group in counts):
        raise ValueError("group IDs must be non-empty")

    ordered_groups = sorted(
        counts,
        key=lambda group: (
            -counts[group],
            identity_sha256("sdv-group-fold-rank-v1", seed, group),
            group,
        ),
    )
    assignments: list[list[str]] = [[] for _ in range(fold_count)]
    loads = [0] * fold_count
    for group in ordered_groups:
        fold_index = min(
            range(fold_count),
            key=lambda index: (loads[index], len(assignments[index]), index),
        )
        assignments[fold_index].append(group)
        loads[fold_index] += counts[group]

    all_groups = set(counts)
    folds: list[GroupedFold] = []
    validation_coverage: list[int] = []
    for fold_index, validation_group_rows in enumerate(assignments):
        validation_groups = frozenset(validation_group_rows)
        train_groups = all_groups - validation_groups
        train_indices = tuple(
            index for index, group in enumerate(group_ids) if group in train_groups
        )
        validation_indices = tuple(
            index for index, group in enumerate(group_ids) if group in validation_groups
        )
        if not train_indices or not validation_indices:
            raise ValueError("grouped fold has an empty train or validation side")
        validation_coverage.extend(validation_indices)
        folds.append(
            GroupedFold(
                index=fold_index,
                train_indices=train_indices,
                validation_indices=validation_indices,
                train_groups=tuple(sorted(train_groups)),
                validation_groups=tuple(sorted(validation_groups)),
            )
        )
    if sorted(validation_coverage) != list(range(len(group_ids))):
        raise RuntimeError("group-fold construction did not cover each row exactly once")
    return tuple(folds)


def _json_value(value: Any) -> Any:
    if value is None:
        return None
    if value is import_module("pandas").NA:
        return None
    item = getattr(value, "item", None)
    if callable(item):
        with contextlib.suppress(TypeError, ValueError):
            value = item()
    if isinstance(value, (dt.datetime, dt.date, dt.time)):
        return value.isoformat()
    if isinstance(value, decimal.Decimal):
        return str(value)
    if isinstance(value, float):
        if math.isnan(value):
            return None
        if not math.isfinite(value):
            raise ValueError("dataframe contains an infinite value")
        return value
    if isinstance(value, (str, int, bool)):
        return value
    is_na = import_module("pandas").isna(value)
    if isinstance(is_na, bool) and is_na:
        return None
    raise ValueError(f"unsupported dataframe scalar for provenance: {type(value).__name__}")


def dataframe_sha256(data: Any) -> str:
    """Hash ordered columns, dtypes, index-independent rows, and nulls."""

    columns = [str(column) for column in data.columns]
    records = [
        [_json_value(value) for value in row] for row in data.itertuples(index=False, name=None)
    ]
    payload = {
        "columns": columns,
        "dtypes": [str(dtype) for dtype in data.dtypes],
        "records": records,
    }
    return hashlib.sha256(canonical_json_bytes(payload)).hexdigest()


def _string_sequence_sha256(values: Sequence[str]) -> str:
    return hashlib.sha256(canonical_json_bytes(list(values))).hexdigest()


def _implementation_receipt() -> dict[str, str]:
    evaluation_path = Path(import_module("document_ocr.synthesis.sdv_evaluation").__file__ or "")
    paths = {
        "document_ocr/synthesis/sdv_harness.py": Path(__file__),
        "document_ocr/synthesis/sdv_evaluation.py": evaluation_path,
    }
    if any(not path.is_file() for path in paths.values()):
        raise SdvBenchmarkError("SDV benchmark implementation sources are unavailable")
    return {name: sha256_file(path) for name, path in sorted(paths.items())}


def _torch_environment_receipt() -> dict[str, Any]:
    torch = import_module("torch")
    cuda_available = bool(torch.cuda.is_available())
    device_count = int(torch.cuda.device_count())
    if cuda_available != (device_count > 0):
        raise SdvBenchmarkError("Torch reports inconsistent CUDA availability/device count")
    devices = []
    if cuda_available:
        for index in range(device_count):
            properties = torch.cuda.get_device_properties(index)
            devices.append(
                {
                    "index": index,
                    "name": str(properties.name),
                    "capability": list(torch.cuda.get_device_capability(index)),
                    "total_memory_bytes": int(properties.total_memory),
                }
            )
    return {
        "torch_version": str(torch.__version__),
        "torch_cuda_version": (str(torch.version.cuda) if torch.version.cuda is not None else None),
        "cuda_available": cuda_available,
        "cuda_device_count": device_count,
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "current_cuda_device": int(torch.cuda.current_device()) if cuda_available else None,
        "devices": devices,
    }


def _environment_receipt() -> dict[str, Any]:
    return {
        "python": platform.python_version(),
        "packages": {
            package: version(package)
            for package in ("numpy", "pandas", "sdmetrics", "sdv", "torch")
        },
        "torch_cuda": _torch_environment_receipt(),
    }


def preflight_sdv_accelerator(*, enable_gpu: bool) -> AcceleratorSelection:
    """Resolve the SDV Torch device, failing before fit when CUDA was requested.

    CTGAN 0.12.1 silently selects CPU when ``enable_gpu=True`` but CUDA is not
    available. The benchmark contract does not permit that fallback: a GPU arm
    is either proven to run on the selected CUDA device or it fails.
    """

    if type(enable_gpu) is not bool:
        raise ValueError("enable_gpu must be boolean")
    runtime = _torch_environment_receipt()
    if enable_gpu and (
        runtime["torch_cuda_version"] is None
        or not runtime["cuda_available"]
        or runtime["cuda_device_count"] < 1
    ):
        raise SdvBenchmarkError(
            "enable_gpu=True requires a CUDA-built Torch runtime with a visible CUDA device"
        )
    if not enable_gpu:
        return AcceleratorSelection(
            enable_gpu=False,
            torch_version=runtime["torch_version"],
            torch_cuda_version=runtime["torch_cuda_version"],
            cuda_available=runtime["cuda_available"],
            cuda_device_count=runtime["cuda_device_count"],
            device="cpu",
            device_name=None,
            device_capability=None,
            device_total_memory_bytes=None,
        )

    device_index = runtime["current_cuda_device"]
    if type(device_index) is not int or not 0 <= device_index < runtime["cuda_device_count"]:
        raise SdvBenchmarkError("Torch returned an invalid current CUDA device")
    device = runtime["devices"][device_index]
    return AcceleratorSelection(
        enable_gpu=True,
        torch_version=runtime["torch_version"],
        torch_cuda_version=runtime["torch_cuda_version"],
        cuda_available=True,
        cuda_device_count=runtime["cuda_device_count"],
        device=f"cuda:{device_index}",
        device_name=device["name"],
        device_capability=tuple(device["capability"]),
        device_total_memory_bytes=device["total_memory_bytes"],
    )


def _rss_bytes() -> int:
    try:
        with Path("/proc/self/status").open(encoding="ascii") as stream:
            for line in stream:
                if line.startswith("VmRSS:"):
                    return int(line.split()[1]) * 1024
    except (FileNotFoundError, OSError, ValueError, IndexError):
        pass
    usage = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return int(usage * (1 if os.uname().sysname == "Darwin" else 1024))


def _peak_rss_bytes() -> int:
    usage = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return int(usage * (1 if os.uname().sysname == "Darwin" else 1024))


def _measure(
    operation: Callable[[], Any], *, accelerator: AcceleratorSelection
) -> tuple[Any, PhaseResources]:
    torch = import_module("torch") if accelerator.enable_gpu else None
    cuda_before: tuple[int, int] | None = None
    if accelerator.enable_gpu:
        assert torch is not None
        device = torch.device(accelerator.device)
        torch.cuda.synchronize(device)
        torch.cuda.reset_peak_memory_stats(device)
        cuda_before = (
            int(torch.cuda.memory_allocated(device)),
            int(torch.cuda.memory_reserved(device)),
        )
    rss_before = _rss_bytes()
    peak_before = _peak_rss_bytes()
    started = time.perf_counter()
    result = operation()
    cuda_after: tuple[int, int, int, int] | None = None
    if accelerator.enable_gpu:
        assert torch is not None
        torch.cuda.synchronize(device)
        cuda_after = (
            int(torch.cuda.memory_allocated(device)),
            int(torch.cuda.memory_reserved(device)),
            int(torch.cuda.max_memory_allocated(device)),
            int(torch.cuda.max_memory_reserved(device)),
        )
    elapsed = time.perf_counter() - started
    receipt = PhaseResources(
        elapsed_seconds=elapsed,
        rss_before_bytes=rss_before,
        rss_after_bytes=_rss_bytes(),
        peak_rss_before_bytes=peak_before,
        peak_rss_after_bytes=_peak_rss_bytes(),
        accelerator=accelerator,
        cuda_allocated_before_bytes=(cuda_before[0] if cuda_before is not None else None),
        cuda_allocated_after_bytes=(cuda_after[0] if cuda_after is not None else None),
        cuda_reserved_before_bytes=(cuda_before[1] if cuda_before is not None else None),
        cuda_reserved_after_bytes=(cuda_after[1] if cuda_after is not None else None),
        cuda_peak_allocated_bytes=(cuda_after[2] if cuda_after is not None else None),
        cuda_peak_reserved_bytes=(cuda_after[3] if cuda_after is not None else None),
    )
    return result, receipt


@contextlib.contextmanager
def _seeded_fit_runtime(seed: int, *, accelerator: AcceleratorSelection) -> Any:
    """Isolate Python, NumPy, and Torch state used while an SDV model fits."""

    numpy = import_module("numpy")
    try:
        torch = import_module("torch")
    except ModuleNotFoundError as error:
        if error.name != "torch" or accelerator.enable_gpu:
            raise
        torch = None
    python_state = random.getstate()
    numpy_state = numpy.random.get_state()
    torch_state = torch.random.get_rng_state() if torch is not None else None
    cuda_states = (
        torch.cuda.get_rng_state_all()
        if torch is not None and accelerator.enable_gpu
        else None
    )
    random.seed(seed)
    numpy.random.seed(seed)
    if torch is not None:
        torch.manual_seed(seed)
    if accelerator.enable_gpu:
        assert torch is not None
        torch.cuda.manual_seed_all(seed)
    try:
        yield
    finally:
        random.setstate(python_state)
        numpy.random.set_state(numpy_state)
        if torch is not None and torch_state is not None:
            torch.random.set_rng_state(torch_state)
        if cuda_states is not None:
            assert torch is not None
            torch.cuda.set_rng_state_all(cuda_states)


class _EmpiricalSynthesizer:
    def __init__(self, *, accelerator: AcceleratorSelection) -> None:
        self._data: Any | None = None
        self.accelerator = accelerator

    def fit(self, data: Any) -> None:
        self._data = data.copy(deep=True).reset_index(drop=True)

    def sample_seeded(self, num_rows: int, seed: int) -> Any:
        if self._data is None:
            raise RuntimeError("empirical synthesizer has not been fitted")
        numpy = import_module("numpy")
        indices = numpy.random.default_rng(seed).integers(0, len(self._data), size=num_rows)
        return self._data.iloc[indices].reset_index(drop=True).copy(deep=True)

    def save(self, path: Path) -> None:
        if self._data is None:
            raise RuntimeError("empirical synthesizer has not been fitted")
        rows = [
            {
                str(column): _json_value(value)
                for column, value in zip(self._data.columns, row, strict=True)
            }
            for row in self._data.itertuples(index=False, name=None)
        ]
        path.write_bytes(
            json_artifact_bytes(
                {
                    "format": "empirical_bootstrap_model_v1",
                    "columns": [str(column) for column in self._data.columns],
                    "dtypes": [str(dtype) for dtype in self._data.dtypes],
                    "rows": rows,
                }
            )
        )


class _SdvSynthesizer:
    def __init__(
        self,
        synthesizer: Any,
        *,
        candidate: CandidateName,
        accelerator: AcceleratorSelection,
        fitted: bool = False,
    ) -> None:
        self._synthesizer = synthesizer
        self._candidate = candidate
        self.accelerator = accelerator
        if fitted:
            self._place_loaded_neural_model()

    def fit(self, data: Any) -> None:
        self._synthesizer.fit(data)
        self._verify_neural_device()

    def _neural_model(self) -> Any:
        model = getattr(self._synthesizer, "_model", None)
        if model is None:
            raise SdvBenchmarkError(
                f"fitted {self._candidate} synthesizer has no auditable underlying model"
            )
        return model

    def _actual_device(self) -> str:
        device_value = getattr(self._neural_model(), "_device", None)
        if device_value is None:
            raise SdvBenchmarkError(
                f"fitted {self._candidate} model exposes no selected Torch device"
            )
        device = str(device_value)
        if device == "cpu":
            return device
        if device == "cuda":
            torch = import_module("torch")
            return f"cuda:{int(torch.cuda.current_device())}"
        if device.startswith("cuda:"):
            try:
                index = int(device.removeprefix("cuda:"))
            except ValueError as error:
                raise SdvBenchmarkError(
                    f"fitted {self._candidate} model exposes invalid Torch device {device!r}"
                ) from error
            if index < 0:
                raise SdvBenchmarkError(
                    f"fitted {self._candidate} model exposes invalid Torch device {device!r}"
                )
            return f"cuda:{index}"
        raise SdvBenchmarkError(
            f"fitted {self._candidate} model exposes unsupported Torch device {device!r}"
        )

    def _verify_neural_device(self) -> None:
        if self._candidate not in {"ctgan", "tvae"}:
            return
        actual = self._actual_device()
        if actual != self.accelerator.device:
            raise SdvBenchmarkError(
                f"{self._candidate} selected {actual}, expected audited device "
                f"{self.accelerator.device}"
            )

    def _place_loaded_neural_model(self) -> None:
        if self._candidate not in {"ctgan", "tvae"}:
            return
        torch = import_module("torch")
        model = self._neural_model()
        setter = getattr(model, "set_device", None)
        if not callable(setter):
            raise SdvBenchmarkError(
                f"loaded {self._candidate} model cannot be placed on its audited device"
            )
        setter(torch.device(self.accelerator.device))
        self._verify_neural_device()

    def sample_seeded(self, num_rows: int, seed: int) -> Any:
        # SDV 1.38.2 exposes deterministic reset_sampling publicly, but not a
        # public varying seed. The pinned private method delegates directly to
        # the underlying model and is guarded so a future API change fails.
        setter = getattr(self._synthesizer, "_set_random_state", None)
        if version("sdv") != "1.38.2" or not callable(setter):
            raise SdvBenchmarkError(
                "seeded SDV sampling requires the pinned 1.38.2 _set_random_state contract"
            )
        setter(seed)
        return self._synthesizer.sample(num_rows=num_rows)

    def save(self, path: Path) -> None:
        self._synthesizer.save(filepath=str(path))


def _candidate_enable_gpu(spec: CandidateSpec) -> bool:
    if spec.name not in {"ctgan", "tvae"}:
        return False
    value = spec.parameters["enable_gpu"]
    if type(value) is not bool:
        raise SdvBenchmarkError(f"candidate {spec.name} enable_gpu is not boolean")
    return value


def _create_model(
    spec: CandidateSpec,
    metadata: Mapping[str, Any],
    *,
    accelerator: AcceleratorSelection,
) -> Any:
    requested_gpu = _candidate_enable_gpu(spec)
    if accelerator.enable_gpu != requested_gpu:
        raise SdvBenchmarkError(
            f"candidate {spec.name} accelerator preflight differs from enable_gpu"
        )
    if spec.name == "empirical":
        return _EmpiricalSynthesizer(accelerator=accelerator)
    metadata_class = import_module("sdv.metadata").Metadata
    sdv_metadata = metadata_class.load_from_dict(dict(metadata))
    candidates = import_module("sdv.single_table")
    classes = {
        "gaussian_copula": candidates.GaussianCopulaSynthesizer,
        "ctgan": candidates.CTGANSynthesizer,
        "tvae": candidates.TVAESynthesizer,
    }
    model_class = classes[spec.name]
    try:
        synthesizer = model_class(sdv_metadata, **dict(spec.parameters))
    except TypeError as error:
        raise SdvBenchmarkError(
            f"invalid {spec.name} parameters for pinned SDV: {error}"
        ) from error
    return _SdvSynthesizer(
        synthesizer,
        candidate=spec.name,
        accelerator=accelerator,
    )


def _validate_metadata(view: BenchmarkView) -> None:
    metadata_class = import_module("sdv.metadata").Metadata
    metadata = metadata_class.load_from_dict(dict(view.metadata))
    metadata.validate()
    metadata.validate_data({view.name: view.data})


def _round_seed(seed: int, round_index: int) -> int:
    digest = identity_sha256("sdv-proposal-round-seed-v1", seed, round_index)
    return int(digest[:8], 16)


def bounded_sample(
    *,
    model: Any,
    requested_rows: int,
    seed: int,
    columns: Sequence[str],
    acceptance: Callable[[Any], Sequence[bool]] | None,
    proposal_multiplier: int,
    proposal_batch_rows: int | None,
) -> tuple[Any, ProposalReceipt]:
    """Sample with a visible, finite rejection budget and exact accounting."""

    if requested_rows < 1 or proposal_multiplier < 1:
        raise ValueError("requested rows and proposal multiplier must be positive")
    pandas = import_module("pandas")
    maximum = requested_rows * proposal_multiplier
    batch_rows = proposal_batch_rows or requested_rows
    accepted_frames: list[Any] = []
    raw = 0
    accepted = 0
    round_index = 0
    while raw < maximum and accepted < requested_rows:
        count = min(batch_rows, maximum - raw)
        proposed = model.sample_seeded(count, _round_seed(seed, round_index))
        round_index += 1
        if len(proposed) != count or list(proposed.columns) != list(columns):
            raise SdvBenchmarkError("candidate returned the wrong row count or column order")
        raw += count
        if acceptance is None:
            mask = [True] * count
        else:
            supplied = list(acceptance(proposed))
            numpy = import_module("numpy")
            boolean_types = (bool, numpy.bool_)
            if len(supplied) != count or any(
                not isinstance(value, boolean_types) for value in supplied
            ):
                raise SdvBenchmarkError(
                    "proposal acceptance must return one strict boolean per row"
                )
            mask = [bool(value) for value in supplied]
        frame = proposed.loc[mask]
        accepted += len(frame)
        if len(frame):
            accepted_frames.append(frame)
    if accepted < requested_rows:
        raise ProposalYieldError(
            f"accepted {accepted}/{requested_rows} rows after {raw} bounded proposals",
            raw_proposals=raw,
            accepted_proposals=accepted,
        )
    output = pandas.concat(accepted_frames, ignore_index=True).iloc[:requested_rows].copy()
    receipt = ProposalReceipt(
        requested_rows=requested_rows,
        raw_proposals=raw,
        accepted_proposals_before_truncation=accepted,
        rejected_proposals=raw - accepted,
        output_rows=len(output),
        acceptance_yield=accepted / raw,
        bounded_maximum_raw_proposals=maximum,
    )
    return output, receipt


def _sha256_fd(descriptor: int) -> str:
    digest = hashlib.sha256()
    os.lseek(descriptor, 0, os.SEEK_SET)
    for chunk in iter(lambda: os.read(descriptor, 1024 * 1024), b""):
        digest.update(chunk)
    return digest.hexdigest()


def _publish_model(
    *,
    model: Any,
    target: Path,
    serialization: str,
    receipt_path: str,
) -> ModelArtifactReceipt:
    """Publish a potentially large model immutably without materializing it."""

    target.parent.mkdir(parents=True, exist_ok=True)
    temporary: Path | None = None
    try:
        descriptor, temporary_name = tempfile.mkstemp(
            dir=target.parent,
            prefix=f".{target.stem}.",
            suffix=target.suffix or ".tmp",
        )
        os.close(descriptor)
        temporary = Path(temporary_name)
        temporary.unlink()
        model.save(temporary)
        file_descriptor = os.open(temporary, os.O_RDONLY | os.O_CLOEXEC | os.O_NOFOLLOW)
        try:
            file_stat = os.fstat(file_descriptor)
            if not stat.S_ISREG(file_stat.st_mode):
                raise SdvBenchmarkError("serialized model is not a regular file")
            os.fsync(file_descriptor)
            digest = _sha256_fd(file_descriptor)
            size = file_stat.st_size
        finally:
            os.close(file_descriptor)
        try:
            os.link(temporary, target, follow_symlinks=False)
            created = True
        except FileExistsError:
            try:
                existing = os.open(target, os.O_RDONLY | os.O_CLOEXEC | os.O_NOFOLLOW)
            except OSError as error:
                raise AtomicConflictError(
                    f"immutable model artifact is not a readable regular file: {target}"
                ) from error
            try:
                existing_stat = os.fstat(existing)
                if not stat.S_ISREG(existing_stat.st_mode):
                    raise AtomicConflictError(
                        f"immutable model artifact is not a regular file: {target}"
                    )
                existing_digest = _sha256_fd(existing)
                existing_size = existing_stat.st_size
            finally:
                os.close(existing)
            if existing_digest != digest or existing_size != size:
                raise AtomicConflictError(
                    f"immutable model artifact conflicts with existing path: {target}"
                ) from None
            created = False
        directory_fd = os.open(target.parent, os.O_RDONLY)
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
        return ModelArtifactReceipt(receipt_path, digest, size, serialization, created)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def _read_json_object(path: Path, *, label: str) -> dict[str, Any]:
    try:
        value = json.loads(read_regular_file_bytes(path))
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise SdvBenchmarkError(f"{label} is not valid UTF-8 JSON: {path}") from error
    if not isinstance(value, dict):
        raise SdvBenchmarkError(f"{label} root is not an object: {path}")
    return value


def _exact_keys(value: Mapping[str, Any], expected: set[str], *, label: str) -> None:
    if set(value) != expected:
        raise SdvBenchmarkError(
            f"{label} keys differ: expected {sorted(expected)}, found {sorted(value)}"
        )


def _finite_float(value: Any, *, label: str) -> float:
    if type(value) not in (int, float) or not math.isfinite(float(value)):
        raise SdvBenchmarkError(f"{label} must be a finite number")
    return float(value)


def _nonnegative_int(value: Any, *, label: str) -> int:
    if type(value) is not int or value < 0:
        raise SdvBenchmarkError(f"{label} must be a non-negative integer")
    return value


def _nullable_nonnegative_int(value: Any, *, label: str) -> int | None:
    if value is None:
        return None
    return _nonnegative_int(value, label=label)


def _relative_artifact_path(root: Path, value: Any, *, label: str) -> Path:
    if not isinstance(value, str) or not value:
        raise SdvBenchmarkError(f"{label} path must be a non-empty string")
    relative = Path(value)
    if relative.is_absolute() or any(part in {"", ".", ".."} for part in relative.parts):
        raise SdvBenchmarkError(f"{label} path must be safe and relative: {value!r}")
    return root / relative


def _verify_file_receipt(
    *, root: Path, path_value: Any, digest_value: Any, size_value: Any, label: str
) -> Path:
    path = _relative_artifact_path(root, path_value, label=label)
    if not isinstance(digest_value, str) or len(digest_value) != 64:
        raise SdvBenchmarkError(f"{label} has an invalid SHA-256")
    expected_size = _nonnegative_int(size_value, label=f"{label}.size_bytes")
    try:
        descriptor = os.open(path, os.O_RDONLY | os.O_CLOEXEC | os.O_NOFOLLOW)
    except OSError as error:
        raise SdvBenchmarkError(f"{label} is not a readable regular file: {path}") from error
    try:
        file_stat = os.fstat(descriptor)
        if not stat.S_ISREG(file_stat.st_mode):
            raise SdvBenchmarkError(f"{label} is not a regular file: {path}")
        digest = _sha256_fd(descriptor)
    finally:
        os.close(descriptor)
    if file_stat.st_size != expected_size or digest != digest_value:
        raise SdvBenchmarkError(f"{label} fails its immutable size/SHA-256 receipt")
    return path


def _accelerator_from_dict(value: Any, *, label: str) -> AcceleratorSelection:
    if not isinstance(value, dict):
        raise SdvBenchmarkError(f"{label} is not an object")
    _exact_keys(
        value,
        {
            "enable_gpu",
            "torch_version",
            "torch_cuda_version",
            "cuda_available",
            "cuda_device_count",
            "device",
            "device_name",
            "device_capability",
            "device_total_memory_bytes",
        },
        label=label,
    )
    if type(value["enable_gpu"]) is not bool or type(value["cuda_available"]) is not bool:
        raise SdvBenchmarkError(f"{label} has a non-boolean flag")
    for key in ("torch_version", "device"):
        if not isinstance(value[key], str) or not value[key]:
            raise SdvBenchmarkError(f"{label}.{key} is invalid")
    for key in ("torch_cuda_version", "device_name"):
        if value[key] is not None and (not isinstance(value[key], str) or not value[key]):
            raise SdvBenchmarkError(f"{label}.{key} is invalid")
    capability_value = value["device_capability"]
    if capability_value is None:
        capability = None
    elif (
        not isinstance(capability_value, list)
        or len(capability_value) != 2
        or any(type(part) is not int or part < 0 for part in capability_value)
    ):
        raise SdvBenchmarkError(f"{label}.device_capability is invalid")
    else:
        capability = (capability_value[0], capability_value[1])
    try:
        return AcceleratorSelection(
            enable_gpu=value["enable_gpu"],
            torch_version=value["torch_version"],
            torch_cuda_version=value["torch_cuda_version"],
            cuda_available=value["cuda_available"],
            cuda_device_count=_nonnegative_int(
                value["cuda_device_count"], label=f"{label}.cuda_device_count"
            ),
            device=value["device"],
            device_name=value["device_name"],
            device_capability=capability,
            device_total_memory_bytes=_nullable_nonnegative_int(
                value["device_total_memory_bytes"],
                label=f"{label}.device_total_memory_bytes",
            ),
        )
    except ValueError as error:
        raise SdvBenchmarkError(f"{label} is inconsistent: {error}") from error


def _phase_resources_from_dict(value: Any, *, label: str) -> PhaseResources:
    if not isinstance(value, dict):
        raise SdvBenchmarkError(f"{label} is not an object")
    _exact_keys(
        value,
        {
            "elapsed_seconds",
            "rss_before_bytes",
            "rss_after_bytes",
            "peak_rss_before_bytes",
            "peak_rss_after_bytes",
            "peak_rss_increase_bytes",
            "accelerator",
            "cuda_allocated_before_bytes",
            "cuda_allocated_after_bytes",
            "cuda_reserved_before_bytes",
            "cuda_reserved_after_bytes",
            "cuda_peak_allocated_bytes",
            "cuda_peak_reserved_bytes",
        },
        label=label,
    )
    try:
        receipt = PhaseResources(
            elapsed_seconds=_finite_float(value["elapsed_seconds"], label=f"{label}.elapsed"),
            rss_before_bytes=_nonnegative_int(
                value["rss_before_bytes"], label=f"{label}.rss_before"
            ),
            rss_after_bytes=_nonnegative_int(value["rss_after_bytes"], label=f"{label}.rss_after"),
            peak_rss_before_bytes=_nonnegative_int(
                value["peak_rss_before_bytes"], label=f"{label}.peak_before"
            ),
            peak_rss_after_bytes=_nonnegative_int(
                value["peak_rss_after_bytes"], label=f"{label}.peak_after"
            ),
            accelerator=_accelerator_from_dict(value["accelerator"], label=f"{label}.accelerator"),
            cuda_allocated_before_bytes=_nullable_nonnegative_int(
                value["cuda_allocated_before_bytes"],
                label=f"{label}.cuda_allocated_before_bytes",
            ),
            cuda_allocated_after_bytes=_nullable_nonnegative_int(
                value["cuda_allocated_after_bytes"],
                label=f"{label}.cuda_allocated_after_bytes",
            ),
            cuda_reserved_before_bytes=_nullable_nonnegative_int(
                value["cuda_reserved_before_bytes"],
                label=f"{label}.cuda_reserved_before_bytes",
            ),
            cuda_reserved_after_bytes=_nullable_nonnegative_int(
                value["cuda_reserved_after_bytes"],
                label=f"{label}.cuda_reserved_after_bytes",
            ),
            cuda_peak_allocated_bytes=_nullable_nonnegative_int(
                value["cuda_peak_allocated_bytes"],
                label=f"{label}.cuda_peak_allocated_bytes",
            ),
            cuda_peak_reserved_bytes=_nullable_nonnegative_int(
                value["cuda_peak_reserved_bytes"],
                label=f"{label}.cuda_peak_reserved_bytes",
            ),
        )
    except ValueError as error:
        raise SdvBenchmarkError(f"{label} is inconsistent: {error}") from error
    if value["peak_rss_increase_bytes"] != receipt.to_dict()["peak_rss_increase_bytes"]:
        raise SdvBenchmarkError(f"{label}.peak_rss_increase_bytes is inconsistent")
    return receipt


def _metric_report_from_dict(value: Any, *, label: str) -> Any:
    from document_ocr.synthesis.sdv_evaluation import MetricDetail, MetricReport

    if not isinstance(value, dict):
        raise SdvBenchmarkError(f"{label} is not an object")
    _exact_keys(value, {"report", "score", "properties", "details"}, label=label)
    report_name = value["report"]
    if not isinstance(report_name, str) or not report_name:
        raise SdvBenchmarkError(f"{label}.report is invalid")
    score = _finite_float(value["score"], label=f"{label}.score")
    raw_properties = value["properties"]
    if not isinstance(raw_properties, dict) or not raw_properties:
        raise SdvBenchmarkError(f"{label}.properties is invalid")
    properties = {
        str(name): _finite_float(property_score, label=f"{label}.properties.{name}")
        for name, property_score in raw_properties.items()
        if isinstance(name, str) and name
    }
    if len(properties) != len(raw_properties):
        raise SdvBenchmarkError(f"{label}.properties has an invalid name")
    raw_details = value["details"]
    if not isinstance(raw_details, list):
        raise SdvBenchmarkError(f"{label}.details is not a list")
    details = []
    for index, raw_detail in enumerate(raw_details):
        if not isinstance(raw_detail, dict):
            raise SdvBenchmarkError(f"{label}.details[{index}] is not an object")
        _exact_keys(raw_detail, {"property", "values"}, label=f"{label}.details[{index}]")
        property_name = raw_detail["property"]
        detail_values = raw_detail["values"]
        if property_name not in properties or not isinstance(detail_values, dict):
            raise SdvBenchmarkError(f"{label}.details[{index}] is inconsistent")
        error = detail_values.get("Error")
        if error not in (None, ""):
            raise SdvBenchmarkError(f"{label}.details[{index}] contains an error")
        detail_score = detail_values.get("Score")
        if detail_score is None:
            if detail_values.get("Meets Threshold?") is not False:
                raise SdvBenchmarkError(f"{label}.details[{index}] has an unexplained null score")
        else:
            _finite_float(detail_score, label=f"{label}.details[{index}].Score")
        details.append(MetricDetail(property_name, detail_values))
    return MetricReport(report_name, score, properties, tuple(details))


def _evaluation_from_dict(value: Any) -> EvaluationBundle:
    if not isinstance(value, dict):
        raise SdvBenchmarkError("candidate evaluation is not an object")
    _exact_keys(value, {"diagnostic", "quality"}, label="candidate evaluation")
    diagnostic = _metric_report_from_dict(value["diagnostic"], label="diagnostic")
    quality = _metric_report_from_dict(value["quality"], label="quality")
    if diagnostic.report_name != "diagnostic" or quality.report_name != "quality":
        raise SdvBenchmarkError("candidate report names are inconsistent")
    if diagnostic.score != 1.0 or any(score != 1.0 for score in diagnostic.properties.values()):
        raise SdvBenchmarkError("resumed candidate diagnostics are not perfect")
    return EvaluationBundle(diagnostic, quality)


def _proposal_from_dict(value: Any, *, requested_rows: int) -> ProposalReceipt:
    if not isinstance(value, dict):
        raise SdvBenchmarkError("candidate proposal receipt is not an object")
    expected_keys = {
        "requested_rows",
        "raw_proposals",
        "accepted_proposals_before_truncation",
        "rejected_proposals",
        "output_rows",
        "acceptance_yield",
        "bounded_maximum_raw_proposals",
    }
    _exact_keys(value, expected_keys, label="candidate proposal receipt")
    receipt = ProposalReceipt(
        requested_rows=_nonnegative_int(value["requested_rows"], label="requested_rows"),
        raw_proposals=_nonnegative_int(value["raw_proposals"], label="raw_proposals"),
        accepted_proposals_before_truncation=_nonnegative_int(
            value["accepted_proposals_before_truncation"], label="accepted_proposals"
        ),
        rejected_proposals=_nonnegative_int(
            value["rejected_proposals"], label="rejected_proposals"
        ),
        output_rows=_nonnegative_int(value["output_rows"], label="output_rows"),
        acceptance_yield=_finite_float(value["acceptance_yield"], label="acceptance_yield"),
        bounded_maximum_raw_proposals=_nonnegative_int(
            value["bounded_maximum_raw_proposals"], label="bounded maximum"
        ),
    )
    if (
        receipt.requested_rows != requested_rows
        or receipt.output_rows != requested_rows
        or receipt.rejected_proposals
        != receipt.raw_proposals - receipt.accepted_proposals_before_truncation
        or receipt.accepted_proposals_before_truncation < requested_rows
        or receipt.raw_proposals > receipt.bounded_maximum_raw_proposals
        or receipt.acceptance_yield
        != receipt.accepted_proposals_before_truncation / receipt.raw_proposals
    ):
        raise SdvBenchmarkError("candidate proposal receipt is arithmetically inconsistent")
    return receipt


def _model_artifact_from_dict(
    value: Any, *, artifact_dir: Path, label: str
) -> ModelArtifactReceipt:
    if not isinstance(value, dict):
        raise SdvBenchmarkError(f"{label} is not an object")
    _exact_keys(
        value,
        {"path", "sha256", "size_bytes", "serialization", "created"},
        label=label,
    )
    serialization = value["serialization"]
    if serialization not in {
        "empirical_bootstrap_model_v1",
        "sdv_pickle_executable_version_pinned_v1",
    }:
        raise SdvBenchmarkError(f"{label} has an unsupported serialization")
    if type(value["created"]) is not bool:
        raise SdvBenchmarkError(f"{label}.created is not boolean")
    _verify_file_receipt(
        root=artifact_dir,
        path_value=value["path"],
        digest_value=value["sha256"],
        size_value=value["size_bytes"],
        label=label,
    )
    return ModelArtifactReceipt(
        path=value["path"],
        sha256=value["sha256"],
        size_bytes=value["size_bytes"],
        serialization=serialization,
        created=value["created"],
    )


def _load_candidate_receipt(
    *,
    path: Path,
    artifact_dir: Path,
    view: BenchmarkView,
    spec: CandidateSpec,
    fold: GroupedFold,
    seed: int,
    expect_model_artifact: bool,
) -> CandidateRunReceipt:
    value = _read_json_object(path, label="candidate run receipt")
    expected_keys = {
        "candidate",
        "parameters",
        "fold",
        "seed",
        "train_row_ids_sha256",
        "validation_row_ids_sha256",
        "train_data_sha256",
        "validation_data_sha256",
        "synthetic_data_sha256",
        "fit",
        "sample",
        "proposal",
        "evaluation",
        "model_artifact",
        "sdv_version",
        "sdmetrics_version",
        "seed_contract",
    }
    _exact_keys(value, expected_keys, label="candidate run receipt")
    if (
        value["candidate"] != spec.name
        or value["parameters"] != dict(spec.parameters)
        or value["fold"] != fold.to_dict()
        or value["seed"] != seed
    ):
        raise SdvBenchmarkError("candidate receipt identity differs from requested arm")
    train_data = view.data.iloc[list(fold.train_indices)].reset_index(drop=True)
    validation_data = view.data.iloc[list(fold.validation_indices)].reset_index(drop=True)
    expected_hashes = {
        "train_row_ids_sha256": _string_sequence_sha256(
            tuple(view.row_ids[index] for index in fold.train_indices)
        ),
        "validation_row_ids_sha256": _string_sequence_sha256(
            tuple(view.row_ids[index] for index in fold.validation_indices)
        ),
        "train_data_sha256": dataframe_sha256(train_data),
        "validation_data_sha256": dataframe_sha256(validation_data),
    }
    for key, expected in expected_hashes.items():
        if value[key] != expected:
            raise SdvBenchmarkError(f"resumed candidate receipt has stale {key}")
    synthetic_digest = value["synthetic_data_sha256"]
    if not isinstance(synthetic_digest, str) or len(synthetic_digest) != 64:
        raise SdvBenchmarkError("candidate synthetic data SHA-256 is invalid")
    if value["sdv_version"] != version("sdv") or value["sdmetrics_version"] != version("sdmetrics"):
        raise SdvBenchmarkError("candidate receipt package versions differ")
    seed_contract = SEED_CONTRACT
    if value["seed_contract"] != seed_contract:
        raise SdvBenchmarkError("candidate receipt seed contract differs")
    model_value = value["model_artifact"]
    if expect_model_artifact != (model_value is not None):
        raise SdvBenchmarkError("candidate receipt fold-model policy differs")
    model_artifact = (
        _model_artifact_from_dict(
            model_value, artifact_dir=artifact_dir, label="candidate fold model"
        )
        if model_value is not None
        else None
    )
    fit_resources = _phase_resources_from_dict(value["fit"], label="candidate fit resources")
    sample_resources = _phase_resources_from_dict(
        value["sample"], label="candidate sample resources"
    )
    requested_gpu = _candidate_enable_gpu(spec)
    if (
        fit_resources.accelerator.enable_gpu != requested_gpu
        or sample_resources.accelerator != fit_resources.accelerator
    ):
        raise SdvBenchmarkError("candidate phase accelerator receipts differ from its parameters")
    return CandidateRunReceipt(
        candidate=spec.name,
        parameters=spec.parameters,
        fold=fold,
        seed=seed,
        train_row_ids_sha256=value["train_row_ids_sha256"],
        validation_row_ids_sha256=value["validation_row_ids_sha256"],
        train_data_sha256=value["train_data_sha256"],
        validation_data_sha256=value["validation_data_sha256"],
        synthetic_data_sha256=synthetic_digest,
        fit=fit_resources,
        sample=sample_resources,
        proposal=_proposal_from_dict(value["proposal"], requested_rows=len(validation_data)),
        evaluation=_evaluation_from_dict(value["evaluation"]),
        model_artifact=model_artifact,
        sdv_version=value["sdv_version"],
        sdmetrics_version=value["sdmetrics_version"],
        seed_contract=seed_contract,
    )


def _verify_complete_manifest(artifact_dir: Path, *, view_name: str) -> bool:
    manifest_path = artifact_dir / "manifest.json"
    if not manifest_path.exists():
        return False
    manifest = _read_json_object(manifest_path, label="benchmark manifest")
    _exact_keys(
        manifest,
        {"status", "view", "sdv_version", "sdmetrics_version", "files"},
        label="benchmark manifest",
    )
    if (
        manifest["status"] != "complete"
        or manifest["view"] != view_name
        or manifest["sdv_version"] != version("sdv")
        or manifest["sdmetrics_version"] != version("sdmetrics")
    ):
        raise SdvBenchmarkError("complete benchmark manifest identity differs")
    files = manifest["files"]
    if not isinstance(files, list):
        raise SdvBenchmarkError("benchmark manifest file inventory is not a list")
    recorded: set[str] = set()
    for index, row in enumerate(files):
        if not isinstance(row, dict):
            raise SdvBenchmarkError(f"benchmark manifest files[{index}] is not an object")
        _exact_keys(row, {"path", "sha256", "size_bytes"}, label="manifest file receipt")
        if row["path"] in recorded:
            raise SdvBenchmarkError("benchmark manifest repeats a file path")
        _verify_file_receipt(
            root=artifact_dir,
            path_value=row["path"],
            digest_value=row["sha256"],
            size_value=row["size_bytes"],
            label=f"benchmark manifest file {row['path']}",
        )
        recorded.add(row["path"])
    actual: set[str] = set()
    for path in artifact_dir.rglob("*"):
        if path.is_symlink():
            raise SdvBenchmarkError(f"benchmark artifact tree contains a symbolic link: {path}")
        if path.is_file() and path != manifest_path:
            actual.add(str(path.relative_to(artifact_dir)))
    if actual != recorded:
        raise SdvBenchmarkError("benchmark manifest inventory differs from artifact tree")
    if "failure.json" in actual:
        raise SdvBenchmarkError("complete benchmark tree contains a failure receipt")
    return True


def _selection_from_receipts(
    receipts: Sequence[CandidateRunReceipt], settings: BenchmarkSettings
) -> ModelSelection:
    return select_candidate(
        runs=[
            CandidateRunScore(
                candidate=receipt.candidate,
                fold_index=receipt.fold.index,
                seed=receipt.seed,
                diagnostic_score=receipt.evaluation.diagnostic.score,
                quality_score=receipt.evaluation.quality.score,
            )
            for receipt in receipts
        ],
        complexity_ranks=DEFAULT_COMPLEXITY_RANKS,
        quality_margin=settings.quality_margin,
        stability_penalty=settings.stability_penalty,
        selectable_candidates=settings.selectable_candidates,
    )


def _receipt_path(artifact_dir: Path, *, candidate: str, fold_index: int, seed: int) -> Path:
    return artifact_dir / "run-receipts" / candidate / f"fold-{fold_index:02d}-seed-{seed}.json"


def _load_selected_model_receipt(
    *,
    value: Any,
    artifact_dir: Path,
    view: BenchmarkView,
    selection: ModelSelection,
    settings: BenchmarkSettings,
    selected_spec: CandidateSpec,
) -> SelectedModelReceipt | None:
    if not settings.fit_selected_model:
        if value is not None:
            raise SdvBenchmarkError("result unexpectedly contains a selected fitted model")
        return None
    if not isinstance(value, dict):
        raise SdvBenchmarkError("result is missing its selected fitted model")
    _exact_keys(
        value,
        {"candidate", "seed", "full_train_data_sha256", "fit", "artifact"},
        label="selected model receipt",
    )
    if (
        value["candidate"] != selection.selected_candidate
        or value["seed"] != min(settings.seeds)
        or value["full_train_data_sha256"] != dataframe_sha256(view.data)
    ):
        raise SdvBenchmarkError("selected model receipt identity differs")
    artifact = _model_artifact_from_dict(
        value["artifact"], artifact_dir=artifact_dir, label="selected model artifact"
    )
    expected_serialization = (
        "empirical_bootstrap_model_v1"
        if selection.selected_candidate == "empirical"
        else "sdv_pickle_executable_version_pinned_v1"
    )
    if artifact.serialization != expected_serialization:
        raise SdvBenchmarkError("selected model serialization differs from its candidate")
    fit_resources = _phase_resources_from_dict(value["fit"], label="selected model fit resources")
    if fit_resources.accelerator.enable_gpu != _candidate_enable_gpu(selected_spec):
        raise SdvBenchmarkError(
            "selected model accelerator receipt differs from its candidate parameters"
        )
    return SelectedModelReceipt(
        candidate=selection.selected_candidate,
        seed=min(settings.seeds),
        full_train_data_sha256=value["full_train_data_sha256"],
        fit=fit_resources,
        artifact=artifact,
    )


def _load_complete_result(
    *,
    artifact_dir: Path,
    view: BenchmarkView,
    candidates: Sequence[CandidateSpec],
    settings: BenchmarkSettings,
    folds: tuple[GroupedFold, ...],
) -> ViewBenchmarkResult | None:
    if not _verify_complete_manifest(artifact_dir, view_name=view.name):
        return None
    receipts = []
    for spec in candidates:
        for fold in folds:
            for seed in settings.seeds:
                receipts.append(
                    _load_candidate_receipt(
                        path=_receipt_path(
                            artifact_dir,
                            candidate=spec.name,
                            fold_index=fold.index,
                            seed=seed,
                        ),
                        artifact_dir=artifact_dir,
                        view=view,
                        spec=spec,
                        fold=fold,
                        seed=seed,
                        expect_model_artifact=settings.persist_fold_models,
                    )
                )
    selection = _selection_from_receipts(receipts, settings)
    selected_spec = next(
        candidate for candidate in candidates if candidate.name == selection.selected_candidate
    )
    raw_result = _read_json_object(artifact_dir / "result.json", label="benchmark result")
    selected_model = _load_selected_model_receipt(
        value=raw_result.get("selected_model"),
        artifact_dir=artifact_dir,
        view=view,
        selection=selection,
        settings=settings,
        selected_spec=selected_spec,
    )
    result = ViewBenchmarkResult(
        view_name=view.name,
        row_count=len(view.data),
        group_count=len(set(view.group_ids)),
        allowed_partition=view.allowed_partition,
        folds=folds,
        runs=tuple(receipts),
        selection=selection,
        selected_model=selected_model,
    )
    if raw_result != result.to_dict():
        raise SdvBenchmarkError("benchmark result differs from reconstructed immutable receipts")
    return result


def _fit_model(
    *, spec: CandidateSpec, metadata: Mapping[str, Any], train_data: Any, seed: int
) -> tuple[Any, PhaseResources]:
    accelerator = preflight_sdv_accelerator(enable_gpu=_candidate_enable_gpu(spec))

    def fit() -> Any:
        with _seeded_fit_runtime(seed, accelerator=accelerator):
            model = _create_model(spec, metadata, accelerator=accelerator)
            model.fit(train_data)
            return model

    return _measure(fit, accelerator=accelerator)


def fit_candidate_model(
    *,
    view: BenchmarkView,
    spec: CandidateSpec,
    train_data: Any,
    seed: int,
) -> FittedCandidateModel:
    """Fit one candidate through the harness's audited seed/device path.

    This is the supported extension seam for task-specific comparative
    benchmarks.  It deliberately does not select or persist a model.
    """

    if seed < 0 or seed >= 2**32:
        raise ValueError("candidate fit seed must be a uint32 value")
    if len(train_data) < 2:
        raise ValueError("candidate fit requires at least two training rows")
    if list(train_data.columns) != list(view.data.columns):
        raise ValueError("candidate fit columns differ from or reorder the benchmark view")
    _validate_metadata(view)
    model, resources = _fit_model(
        spec=spec,
        metadata=view.metadata,
        train_data=train_data,
        seed=seed,
    )
    return FittedCandidateModel(
        candidate=spec.name,
        parameters=spec.parameters,
        fit=resources,
        _model=model,
    )


def _candidate_run(
    *,
    view: BenchmarkView,
    spec: CandidateSpec,
    fold: GroupedFold,
    seed: int,
    settings: BenchmarkSettings,
    acceptance: Callable[[Any], Sequence[bool]] | None,
    model_target: Path | None,
    model_receipt_path: str | None,
) -> CandidateRunReceipt:
    train_data = view.data.iloc[list(fold.train_indices)].reset_index(drop=True).copy()
    validation_data = view.data.iloc[list(fold.validation_indices)].reset_index(drop=True).copy()
    model, fit_resources = _fit_model(
        spec=spec,
        metadata=view.metadata,
        train_data=train_data,
        seed=seed,
    )

    model_artifact = None
    if model_target is not None:
        if model_receipt_path is None:
            raise SdvBenchmarkError("model target requires a relative receipt path")
        serialization = (
            "empirical_bootstrap_model_v1"
            if spec.name == "empirical"
            else "sdv_pickle_executable_version_pinned_v1"
        )
        model_artifact = _publish_model(
            model=model,
            target=model_target,
            serialization=serialization,
            receipt_path=model_receipt_path,
        )

    def sample() -> tuple[Any, ProposalReceipt]:
        return bounded_sample(
            model=model,
            requested_rows=len(validation_data),
            seed=seed,
            columns=[str(column) for column in validation_data.columns],
            acceptance=acceptance,
            proposal_multiplier=settings.proposal_multiplier,
            proposal_batch_rows=settings.proposal_batch_rows,
        )

    sampled, sample_resources = _measure(sample, accelerator=model.accelerator)
    synthetic_data, proposal = sampled
    evaluation = evaluate_single_table(
        real_data=validation_data,
        synthetic_data=synthetic_data,
        metadata=view.metadata,
        table_name=view.name,
        diagnostic_reference_data=train_data,
    )
    return CandidateRunReceipt(
        candidate=spec.name,
        parameters=spec.parameters,
        fold=fold,
        seed=seed,
        train_row_ids_sha256=_string_sequence_sha256(
            tuple(view.row_ids[index] for index in fold.train_indices)
        ),
        validation_row_ids_sha256=_string_sequence_sha256(
            tuple(view.row_ids[index] for index in fold.validation_indices)
        ),
        train_data_sha256=dataframe_sha256(train_data),
        validation_data_sha256=dataframe_sha256(validation_data),
        synthetic_data_sha256=dataframe_sha256(synthetic_data),
        fit=fit_resources,
        sample=sample_resources,
        proposal=proposal,
        evaluation=evaluation,
        model_artifact=model_artifact,
        sdv_version=version("sdv"),
        sdmetrics_version=version("sdmetrics"),
        seed_contract=SEED_CONTRACT,
    )


def run_view_benchmark(
    *,
    view: BenchmarkView,
    candidates: Sequence[CandidateSpec],
    settings: BenchmarkSettings,
    artifact_dir: Path | None,
    acceptance: Callable[[Any], Sequence[bool]] | None = None,
    acceptance_contract: str = "accept_all_v1",
) -> ViewBenchmarkResult:
    """Benchmark and select one model for a train-only modeling view.

    Any candidate failure aborts the run and is immutably receipted when an
    artifact directory is supplied. There is deliberately no automatic
    candidate removal or fallback.
    """

    if not candidates or len({candidate.name for candidate in candidates}) != len(candidates):
        raise ValueError("benchmark candidates must be non-empty and unique")
    if "empirical" not in {candidate.name for candidate in candidates}:
        raise ValueError("benchmark candidates must include the empirical baseline")
    if settings.fold_count > len(set(view.group_ids)):
        raise ValueError("fold_count exceeds view source/template groups")
    benchmarked_names = {candidate.name for candidate in candidates}
    if settings.selectable_candidates is not None:
        unknown_selectable = sorted(set(settings.selectable_candidates) - benchmarked_names)
        if unknown_selectable:
            raise ValueError(
                "selectable candidates were not benchmarked: " + ", ".join(unknown_selectable)
            )
    if settings.fit_selected_model and artifact_dir is None:
        raise ValueError("fit_selected_model requires an artifact_dir")
    if not acceptance_contract or (
        acceptance is not None and acceptance_contract == "accept_all_v1"
    ):
        raise ValueError("a custom acceptance callable requires a non-default acceptance_contract")

    _validate_metadata(view)
    folds = grouped_folds(
        group_ids=view.group_ids,
        fold_count=settings.fold_count,
        seed=settings.fold_seed,
    )
    if artifact_dir is not None:
        atomic_publish_json(artifact_dir / "metadata.json", dict(view.metadata))
        atomic_publish_json(
            artifact_dir / "benchmark-contract.json",
            {
                "view": view.name,
                "rows": len(view.data),
                "groups": len(set(view.group_ids)),
                "allowed_partition": view.allowed_partition,
                "columns": [str(column) for column in view.data.columns],
                "data_sha256": dataframe_sha256(view.data),
                "row_membership_sha256": hashlib.sha256(
                    canonical_json_bytes(
                        [
                            {
                                "row_id": row_id,
                                "group_id": group_id,
                                "partition": partition,
                            }
                            for row_id, group_id, partition in zip(
                                view.row_ids,
                                view.group_ids,
                                view.partition_labels,
                                strict=True,
                            )
                        ]
                    )
                ).hexdigest(),
                "implementation": _implementation_receipt(),
                "environment": _environment_receipt(),
                "candidates": [
                    {"name": candidate.name, "parameters": dict(candidate.parameters)}
                    for candidate in candidates
                ],
                "settings": {
                    "fold_count": settings.fold_count,
                    "fold_seed": settings.fold_seed,
                    "seeds": list(settings.seeds),
                    "quality_margin": settings.quality_margin,
                    "stability_penalty": settings.stability_penalty,
                    "proposal_multiplier": settings.proposal_multiplier,
                    "proposal_batch_rows": settings.proposal_batch_rows,
                    "persist_fold_models": settings.persist_fold_models,
                    "fit_selected_model": settings.fit_selected_model,
                    "acceptance_contract": acceptance_contract,
                    "selectable_candidates": sorted(
                        settings.selectable_candidates or benchmarked_names
                    ),
                },
            },
        )
        if (artifact_dir / "failure.json").exists():
            raise SdvBenchmarkError(
                "immutable benchmark run is already failed; use a new artifact directory"
            )
        complete = _load_complete_result(
            artifact_dir=artifact_dir,
            view=view,
            candidates=candidates,
            settings=settings,
            folds=folds,
        )
        if complete is not None:
            return complete

    receipts: list[CandidateRunReceipt] = []
    try:
        for spec in candidates:
            for fold in folds:
                for seed in settings.seeds:
                    receipt_path = (
                        _receipt_path(
                            artifact_dir,
                            candidate=spec.name,
                            fold_index=fold.index,
                            seed=seed,
                        )
                        if artifact_dir is not None
                        else None
                    )
                    if receipt_path is not None and receipt_path.exists():
                        assert artifact_dir is not None
                        receipts.append(
                            _load_candidate_receipt(
                                path=receipt_path,
                                artifact_dir=artifact_dir,
                                view=view,
                                spec=spec,
                                fold=fold,
                                seed=seed,
                                expect_model_artifact=settings.persist_fold_models,
                            )
                        )
                        continue
                    model_target = None
                    model_receipt_path = None
                    if artifact_dir is not None and settings.persist_fold_models:
                        extension = "json" if spec.name == "empirical" else "pkl"
                        model_target = (
                            artifact_dir
                            / "fold-models"
                            / spec.name
                            / f"fold-{fold.index:02d}-seed-{seed}.{extension}"
                        )
                        model_receipt_path = str(model_target.relative_to(artifact_dir))
                    receipt = _candidate_run(
                        view=view,
                        spec=spec,
                        fold=fold,
                        seed=seed,
                        settings=settings,
                        acceptance=acceptance,
                        model_target=model_target,
                        model_receipt_path=model_receipt_path,
                    )
                    receipts.append(receipt)
                    if receipt_path is not None:
                        atomic_publish_json(
                            receipt_path,
                            receipt.to_dict(),
                        )
    except Exception as error:
        if artifact_dir is not None:
            atomic_publish_json(
                artifact_dir / "failure.json",
                {
                    "status": "failed",
                    "error_type": type(error).__name__,
                    "message": str(error),
                    "completed_runs": len(receipts),
                    "phase": "cross_validation",
                },
            )
        raise SdvBenchmarkError(
            f"SDV benchmark failed after {len(receipts)} completed candidate runs: "
            f"{type(error).__name__}: {error}"
        ) from error

    try:
        selection = _selection_from_receipts(receipts, settings)
        selected_model_receipt = None
        if settings.fit_selected_model:
            if artifact_dir is None:
                raise ValueError("fit_selected_model requires an artifact_dir")
            selected_spec = next(
                candidate
                for candidate in candidates
                if candidate.name == selection.selected_candidate
            )
            selected_seed = min(settings.seeds)
            model, fit_resources = _fit_model(
                spec=selected_spec,
                metadata=view.metadata,
                train_data=view.data,
                seed=selected_seed,
            )
            extension = "json" if selected_spec.name == "empirical" else "pkl"
            serialization = (
                "empirical_bootstrap_model_v1"
                if selected_spec.name == "empirical"
                else "sdv_pickle_executable_version_pinned_v1"
            )
            artifact = _publish_model(
                model=model,
                target=artifact_dir / "selected-model" / f"model.{extension}",
                serialization=serialization,
                receipt_path=f"selected-model/model.{extension}",
            )
            selected_model_receipt = SelectedModelReceipt(
                candidate=selected_spec.name,
                seed=selected_seed,
                full_train_data_sha256=dataframe_sha256(view.data),
                fit=fit_resources,
                artifact=artifact,
            )
    except Exception as error:
        if artifact_dir is not None:
            atomic_publish_json(
                artifact_dir / "failure.json",
                {
                    "status": "failed",
                    "error_type": type(error).__name__,
                    "message": str(error),
                    "completed_runs": len(receipts),
                    "phase": "selection_or_final_fit",
                },
            )
        raise SdvBenchmarkError(
            "SDV benchmark selection/final fit failed after "
            f"{len(receipts)} completed candidate runs: {type(error).__name__}: {error}"
        ) from error

    result = ViewBenchmarkResult(
        view_name=view.name,
        row_count=len(view.data),
        group_count=len(set(view.group_ids)),
        allowed_partition=view.allowed_partition,
        folds=folds,
        runs=tuple(receipts),
        selection=selection,
        selected_model=selected_model_receipt,
    )
    if artifact_dir is not None:
        atomic_publish_json(artifact_dir / "result.json", result.to_dict())
        file_receipts = []
        for path in sorted(artifact_dir.rglob("*")):
            if path.is_file() and path.name != "manifest.json":
                file_receipts.append(
                    {
                        "path": str(path.relative_to(artifact_dir)),
                        "sha256": sha256_file(path),
                        "size_bytes": path.stat().st_size,
                    }
                )
        atomic_publish_json(
            artifact_dir / "manifest.json",
            {
                "status": "complete",
                "view": view.name,
                "sdv_version": version("sdv"),
                "sdmetrics_version": version("sdmetrics"),
                "files": file_receipts,
            },
        )
    return result


def _load_empirical_model(
    path: Path,
    *,
    expected_columns: Sequence[str],
    accelerator: AcceleratorSelection,
) -> _EmpiricalSynthesizer:
    value = _read_json_object(path, label="empirical selected model")
    _exact_keys(
        value,
        {"format", "columns", "dtypes", "rows"},
        label="empirical selected model",
    )
    if value["format"] != "empirical_bootstrap_model_v1":
        raise SdvBenchmarkError("empirical selected model format differs")
    if value["columns"] != list(expected_columns):
        raise SdvBenchmarkError("empirical selected model columns differ from metadata")
    dtypes = value["dtypes"]
    rows = value["rows"]
    if (
        not isinstance(dtypes, list)
        or len(dtypes) != len(expected_columns)
        or not all(isinstance(dtype, str) and dtype for dtype in dtypes)
        or not isinstance(rows, list)
        or not rows
        or any(not isinstance(row, dict) or set(row) != set(expected_columns) for row in rows)
    ):
        raise SdvBenchmarkError("empirical selected model rows/dtypes are invalid")
    pandas = import_module("pandas")
    data = pandas.DataFrame(rows, columns=list(expected_columns))
    try:
        data = data.astype(dict(zip(expected_columns, dtypes, strict=True)))
    except (TypeError, ValueError) as error:
        raise SdvBenchmarkError("empirical selected model dtypes cannot be restored") from error
    model = _EmpiricalSynthesizer(accelerator=accelerator)
    model.fit(data)
    return model


def load_selected_model(artifact_dir: Path) -> LoadedSelectedModel:
    """Verify a complete benchmark once and return a reusable sampling handle."""

    manifest = _read_json_object(artifact_dir / "manifest.json", label="benchmark manifest")
    view_name = manifest.get("view")
    if not isinstance(view_name, str) or not view_name:
        raise SdvBenchmarkError("benchmark manifest has no valid view")
    _verify_complete_manifest(artifact_dir, view_name=view_name)
    result = _read_json_object(artifact_dir / "result.json", label="benchmark result")
    selection = result.get("selection")
    selected = result.get("selected_model")
    if not isinstance(selection, dict) or not isinstance(selected, dict):
        raise SdvBenchmarkError("benchmark result contains no selected fitted model")
    candidate = selection.get("selected_candidate")
    if candidate not in SUPPORTED_CANDIDATES or selected.get("candidate") != candidate:
        raise SdvBenchmarkError("selected model candidate is inconsistent")
    artifact = _model_artifact_from_dict(
        selected.get("artifact"), artifact_dir=artifact_dir, label="selected model artifact"
    )
    metadata = _read_json_object(artifact_dir / "metadata.json", label="benchmark metadata")
    contract = _read_json_object(
        artifact_dir / "benchmark-contract.json", label="benchmark contract"
    )
    candidate_values = contract.get("candidates")
    if not isinstance(candidate_values, list):
        raise SdvBenchmarkError("benchmark contract has no candidate specifications")
    matching_candidates = [
        value
        for value in candidate_values
        if isinstance(value, dict) and value.get("name") == candidate
    ]
    if len(matching_candidates) != 1 or set(matching_candidates[0]) != {
        "name",
        "parameters",
    }:
        raise SdvBenchmarkError("benchmark contract has no unique selected candidate")
    try:
        selected_spec = CandidateSpec(candidate, matching_candidates[0]["parameters"])
    except (TypeError, ValueError) as error:
        raise SdvBenchmarkError("selected candidate specification is invalid") from error
    accelerator = preflight_sdv_accelerator(enable_gpu=_candidate_enable_gpu(selected_spec))
    tables = metadata.get("tables")
    if not isinstance(tables, dict) or set(tables) != {view_name}:
        raise SdvBenchmarkError("benchmark metadata does not describe its selected view")
    table = tables[view_name]
    columns_value = table.get("columns") if isinstance(table, dict) else None
    if not isinstance(columns_value, dict) or not columns_value:
        raise SdvBenchmarkError("benchmark metadata has no view columns")
    contract_columns = contract.get("columns")
    if (
        not isinstance(contract_columns, list)
        or not contract_columns
        or not all(isinstance(column, str) and column for column in contract_columns)
        or set(contract_columns) != set(columns_value)
    ):
        raise SdvBenchmarkError("benchmark contract has invalid or stale view columns")
    columns = tuple(contract_columns)
    model_path = _relative_artifact_path(
        artifact_dir, artifact.path, label="selected model artifact"
    )
    if artifact.serialization == "empirical_bootstrap_model_v1":
        if candidate != "empirical":
            raise SdvBenchmarkError("empirical serialization belongs to a non-empirical candidate")
        model: Any = _load_empirical_model(
            model_path,
            expected_columns=columns,
            accelerator=accelerator,
        )
    elif artifact.serialization == "sdv_pickle_executable_version_pinned_v1":
        if candidate == "empirical":
            raise SdvBenchmarkError("SDV pickle belongs to the empirical candidate")
        if version("sdv") != manifest["sdv_version"]:
            raise SdvBenchmarkError("installed SDV differs from the model manifest")
        synthesizer = import_module("sdv.utils").load_synthesizer(filepath=str(model_path))
        model = _SdvSynthesizer(
            synthesizer,
            candidate=selected_spec.name,
            accelerator=accelerator,
            fitted=True,
        )
    else:
        raise SdvBenchmarkError("unsupported selected model serialization")
    return LoadedSelectedModel(
        candidate=candidate,
        model_sha256=artifact.sha256,
        table_name=view_name,
        columns=columns,
        _model=model,
    )


def sample_selected_model(
    *,
    artifact_dir: Path,
    request_id: str,
    num_rows: int,
    seed: int,
    acceptance: Callable[[Any], Sequence[bool]] | None = None,
    acceptance_contract: str = "accept_all_v1",
    proposal_multiplier: int = 1,
    proposal_batch_rows: int | None = None,
) -> tuple[Any, SelectedModelSampleReceipt]:
    """Convenience wrapper for one request; reuse ``load_selected_model`` for many."""

    handle = load_selected_model(artifact_dir)
    return handle.sample(
        request_id=request_id,
        num_rows=num_rows,
        seed=seed,
        acceptance=acceptance,
        acceptance_contract=acceptance_contract,
        proposal_multiplier=proposal_multiplier,
        proposal_batch_rows=proposal_batch_rows,
    )
