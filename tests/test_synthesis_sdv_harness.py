from __future__ import annotations

import importlib.util
import json
import math
from pathlib import Path
from types import SimpleNamespace

import pandas as pd
import pytest

from document_ocr.synthesis import sdv_harness
from document_ocr.synthesis.sdv_evaluation import (
    CandidateRunScore,
    EvaluationBundle,
    MetricReport,
    SdvEvaluationError,
    _validated_report,
    select_candidate,
)
from document_ocr.synthesis.sdv_harness import (
    AcceleratorSelection,
    BenchmarkSettings,
    BenchmarkView,
    CandidateRunReceipt,
    CandidateSpec,
    PhaseResources,
    ProposalReceipt,
    ProposalYieldError,
    SdvBenchmarkError,
    bounded_sample,
    dataframe_sha256,
    default_candidate_specs,
    grouped_folds,
    load_selected_model,
    preflight_sdv_accelerator,
    run_view_benchmark,
)


def _data(rows: int = 24) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "cargo_class": ["A" if index % 3 else "B" for index in range(rows)],
            "quantity": [index % 7 + 1 for index in range(rows)],
            "gross_weight": [(index % 7 + 1) * 10.5 for index in range(rows)],
        }
    )


def _metadata() -> dict[str, object]:
    return {
        "METADATA_SPEC_VERSION": "V1",
        "tables": {
            "cargo_view": {
                "columns": {
                    "cargo_class": {"sdtype": "categorical"},
                    "quantity": {
                        "sdtype": "numerical",
                        "computer_representation": "Int64",
                    },
                    "gross_weight": {
                        "sdtype": "numerical",
                        "computer_representation": "Float",
                    },
                }
            }
        },
        "relationships": [],
    }


def _view(*, rows: int = 24, partitions: tuple[str, ...] | None = None) -> BenchmarkView:
    data = _data(rows)
    return BenchmarkView(
        name="cargo_view",
        data=data,
        metadata=_metadata(),
        row_ids=tuple(f"row-{index}" for index in range(len(data))),
        group_ids=tuple(f"template-{index // 3}" for index in range(len(data))),
        partition_labels=partitions or ("train",) * len(data),
    )


def _cpu_accelerator() -> AcceleratorSelection:
    return AcceleratorSelection(
        enable_gpu=False,
        torch_version="2.0-test",
        torch_cuda_version=None,
        cuda_available=False,
        cuda_device_count=0,
        device="cpu",
        device_name=None,
        device_capability=None,
        device_total_memory_bytes=None,
    )


def _resources() -> PhaseResources:
    return PhaseResources(
        elapsed_seconds=0.01,
        rss_before_bytes=100,
        rss_after_bytes=110,
        peak_rss_before_bytes=120,
        peak_rss_after_bytes=130,
        accelerator=_cpu_accelerator(),
        cuda_allocated_before_bytes=None,
        cuda_allocated_after_bytes=None,
        cuda_reserved_before_bytes=None,
        cuda_reserved_after_bytes=None,
        cuda_peak_allocated_bytes=None,
        cuda_peak_reserved_bytes=None,
    )


def _fake_version(package: str) -> str:
    return {
        "numpy": "2.0-test",
        "pandas": "3.0-test",
        "sdmetrics": "0.30.0",
        "sdv": "1.38.2",
        "torch": "2.0-test",
    }[package]


def _fake_torch_environment_receipt() -> dict[str, object]:
    return {
        "torch_version": "2.0-test",
        "torch_cuda_version": None,
        "cuda_available": False,
        "cuda_device_count": 0,
        "cuda_visible_devices": None,
        "current_cuda_device": None,
        "devices": [],
    }


def _evaluation(quality: float) -> EvaluationBundle:
    return EvaluationBundle(
        diagnostic=MetricReport(
            "diagnostic", 1.0, {"Data Validity": 1.0, "Data Structure": 1.0}, ()
        ),
        quality=MetricReport(
            "quality", quality, {"Column Shapes": quality, "Column Pair Trends": quality}, ()
        ),
    )


def _fake_receipt(**kwargs: object) -> CandidateRunReceipt:
    spec = kwargs["spec"]
    fold = kwargs["fold"]
    seed = kwargs["seed"]
    assert isinstance(spec, CandidateSpec)
    assert isinstance(fold, sdv_harness.GroupedFold)
    assert isinstance(seed, int)
    validation_rows = len(fold.validation_indices)
    qualities = {
        "empirical": 0.70,
        "gaussian_copula": 0.82,
        "ctgan": 0.84,
        "tvae": 0.80,
    }
    model_artifact = None
    model_target = kwargs.get("model_target")
    model_receipt_path = kwargs.get("model_receipt_path")
    if model_target is not None:
        assert isinstance(model_target, Path)
        assert isinstance(model_receipt_path, str)

        class _FixtureModel:
            @staticmethod
            def save(path: Path) -> None:
                path.write_bytes(b"immutable fixture fold model")

        model_artifact = sdv_harness._publish_model(
            model=_FixtureModel(),
            target=model_target,
            serialization="empirical_bootstrap_model_v1",
            receipt_path=model_receipt_path,
        )
    return CandidateRunReceipt(
        candidate=spec.name,
        parameters=spec.parameters,
        fold=fold,
        seed=seed,
        train_row_ids_sha256=sdv_harness._string_sequence_sha256(
            tuple(f"row-{index}" for index in fold.train_indices)
        ),
        validation_row_ids_sha256=sdv_harness._string_sequence_sha256(
            tuple(f"row-{index}" for index in fold.validation_indices)
        ),
        train_data_sha256=dataframe_sha256(
            _data().iloc[list(fold.train_indices)].reset_index(drop=True)
        ),
        validation_data_sha256=dataframe_sha256(
            _data().iloc[list(fold.validation_indices)].reset_index(drop=True)
        ),
        synthetic_data_sha256="e" * 64,
        fit=_resources(),
        sample=_resources(),
        proposal=ProposalReceipt(
            validation_rows,
            validation_rows,
            validation_rows,
            0,
            validation_rows,
            1.0,
            validation_rows,
        ),
        evaluation=_evaluation(qualities[spec.name]),
        model_artifact=model_artifact,
        sdv_version="1.38.2",
        sdmetrics_version="0.30.0",
        seed_contract=sdv_harness.SEED_CONTRACT,
    )


def test_view_rejects_non_train_rows_before_any_fit() -> None:
    partitions = ("train",) * 23 + ("validation",)
    with pytest.raises(ValueError, match="not train-only"):
        _view(partitions=partitions)


def test_default_neural_candidates_make_gpu_request_explicit() -> None:
    cpu = default_candidate_specs(neural_epochs=7)
    gpu = default_candidate_specs(neural_epochs=7, enable_gpu=True)
    assert [candidate.parameters.get("enable_gpu") for candidate in cpu[2:]] == [False, False]
    assert [candidate.parameters.get("enable_gpu") for candidate in gpu[2:]] == [True, True]
    with pytest.raises(ValueError, match="explicitly set boolean enable_gpu"):
        CandidateSpec("ctgan", {"epochs": 1})
    with pytest.raises(ValueError, match="must use enable_gpu"):
        CandidateSpec("tvae", {"epochs": 1, "enable_gpu": True, "cuda": True})
    with pytest.raises(ValueError, match="does not support enable_gpu"):
        CandidateSpec("gaussian_copula", {"enable_gpu": True})


def test_gpu_preflight_fails_closed_when_cuda_is_not_available(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fake_torch = SimpleNamespace(
        __version__="2.13.0+cu130",
        version=SimpleNamespace(cuda="13.0"),
        cuda=SimpleNamespace(
            is_available=lambda: False,
            device_count=lambda: 0,
        ),
    )
    real_import = sdv_harness.import_module
    monkeypatch.setattr(
        sdv_harness,
        "import_module",
        lambda name: fake_torch if name == "torch" else real_import(name),
    )
    with pytest.raises(SdvBenchmarkError, match="requires a CUDA-built Torch runtime"):
        preflight_sdv_accelerator(enable_gpu=True)
    cpu = preflight_sdv_accelerator(enable_gpu=False)
    assert cpu.device == "cpu"
    assert cpu.torch_cuda_version == "13.0"
    assert not cpu.cuda_available


def test_fitted_neural_model_must_prove_its_underlying_device() -> None:
    accelerator = AcceleratorSelection(
        enable_gpu=True,
        torch_version="2.13.0+cu130",
        torch_cuda_version="13.0",
        cuda_available=True,
        cuda_device_count=1,
        device="cuda:0",
        device_name="fixture GPU",
        device_capability=(9, 0),
        device_total_memory_bytes=24 * 1024**3,
    )

    class _SilentCpuFallback:
        def fit(self, data: pd.DataFrame) -> None:
            del data
            self._model = SimpleNamespace(_device="cpu")

    model = sdv_harness._SdvSynthesizer(
        _SilentCpuFallback(),
        candidate="ctgan",
        accelerator=accelerator,
    )
    with pytest.raises(SdvBenchmarkError, match="selected cpu, expected audited device cuda:0"):
        model.fit(_data(2))


def test_gpu_phase_receipt_records_peak_allocated_and_reserved_vram(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    accelerator = AcceleratorSelection(
        enable_gpu=True,
        torch_version="2.13.0+cu130",
        torch_cuda_version="13.0",
        cuda_available=True,
        cuda_device_count=1,
        device="cuda:0",
        device_name="fixture GPU",
        device_capability=(9, 0),
        device_total_memory_bytes=24 * 1024**3,
    )
    allocated = iter((100, 140))
    reserved = iter((200, 260))
    fake_torch = SimpleNamespace(
        device=lambda value: value,
        cuda=SimpleNamespace(
            synchronize=lambda device: None,
            reset_peak_memory_stats=lambda device: None,
            memory_allocated=lambda device: next(allocated),
            memory_reserved=lambda device: next(reserved),
            max_memory_allocated=lambda device: 180,
            max_memory_reserved=lambda device: 300,
        ),
    )
    real_import = sdv_harness.import_module
    monkeypatch.setattr(
        sdv_harness,
        "import_module",
        lambda name: fake_torch if name == "torch" else real_import(name),
    )
    result, resources = sdv_harness._measure(lambda: "done", accelerator=accelerator)
    assert result == "done"
    assert resources.cuda_allocated_before_bytes == 100
    assert resources.cuda_allocated_after_bytes == 140
    assert resources.cuda_reserved_before_bytes == 200
    assert resources.cuda_reserved_after_bytes == 260
    assert resources.cuda_peak_allocated_bytes == 180
    assert resources.cuda_peak_reserved_bytes == 300
    assert resources.to_dict()["accelerator"]["device"] == "cuda:0"
    assert (
        sdv_harness._phase_resources_from_dict(resources.to_dict(), label="fixture GPU resources")
        == resources
    )


def test_grouped_folds_are_balanced_deterministic_and_group_disjoint() -> None:
    groups = ("large",) * 9 + ("medium",) * 5 + tuple(f"small-{index}" for index in range(8))
    first = grouped_folds(group_ids=groups, fold_count=4, seed=991)
    second = grouped_folds(group_ids=groups, fold_count=4, seed=991)
    assert first == second
    # The indivisible nine-row source group sets the optimum; all remaining
    # folds are balanced to four/four/five rows.
    assert sorted(len(fold.validation_indices) for fold in first) == [4, 4, 5, 9]
    assert sorted(index for fold in first for index in fold.validation_indices) == list(
        range(len(groups))
    )
    for fold in first:
        assert set(fold.train_groups).isdisjoint(fold.validation_groups)


def test_dataframe_hash_covers_order_types_values_and_nulls() -> None:
    data = _data(8)
    original = dataframe_sha256(data)
    assert dataframe_sha256(data.copy()) == original
    assert dataframe_sha256(data.iloc[::-1].reset_index(drop=True)) != original
    modified = data.copy()
    modified.loc[0, "gross_weight"] = float("nan")
    assert dataframe_sha256(modified) != original
    cast = data.copy()
    cast["quantity"] = cast["quantity"].astype("float64")
    assert dataframe_sha256(cast) != original


class _CountingModel:
    def __init__(self) -> None:
        self.offset = 0

    def sample_seeded(self, num_rows: int, seed: int) -> pd.DataFrame:
        del seed
        values = list(range(self.offset, self.offset + num_rows))
        self.offset += num_rows
        return pd.DataFrame({"value": values})


def test_bounded_sampling_reports_raw_rejection_and_never_fills_silently() -> None:
    sampled, receipt = bounded_sample(
        model=_CountingModel(),
        requested_rows=4,
        seed=9,
        columns=("value",),
        acceptance=lambda frame: (frame["value"] % 2 == 0).tolist(),
        proposal_multiplier=3,
        proposal_batch_rows=4,
    )
    assert sampled["value"].tolist() == [0, 2, 4, 6]
    assert receipt.raw_proposals == 8
    assert receipt.accepted_proposals_before_truncation == 4
    assert receipt.rejected_proposals == 4
    assert receipt.acceptance_yield == 0.5

    with pytest.raises(ProposalYieldError, match="accepted 0/3") as caught:
        bounded_sample(
            model=_CountingModel(),
            requested_rows=3,
            seed=9,
            columns=("value",),
            acceptance=lambda frame: [False] * len(frame),
            proposal_multiplier=2,
            proposal_batch_rows=2,
        )
    assert caught.value.raw_proposals == 6
    assert caught.value.accepted_proposals == 0


def test_candidate_selection_is_paired_baseline_relative_and_simplicity_aware() -> None:
    runs = []
    for fold in range(2):
        for seed in (7, 11):
            runs.extend(
                (
                    CandidateRunScore("empirical", fold, seed, 1.0, 0.80),
                    CandidateRunScore("gaussian_copula", fold, seed, 1.0, 0.905),
                    CandidateRunScore("ctgan", fold, seed, 1.0, 0.91),
                    CandidateRunScore("tvae", fold, seed, 1.0, 0.89),
                )
            )
    selection = select_candidate(
        runs=runs,
        complexity_ranks={"empirical": 0, "gaussian_copula": 1, "ctgan": 2, "tvae": 2},
        quality_margin=0.01,
        stability_penalty=0.25,
    )
    assert selection.selected_candidate == "gaussian_copula"
    gaussian = next(score for score in selection.scorecards if score.candidate == "gaussian_copula")
    assert gaussian.mean_improvement_over_empirical == pytest.approx(0.105)

    production_only = select_candidate(
        runs=(
            CandidateRunScore("empirical", 0, 7, 1.0, 0.99),
            CandidateRunScore("gaussian_copula", 0, 7, 1.0, 0.70),
        ),
        complexity_ranks={"empirical": 0, "gaussian_copula": 1},
        selectable_candidates=("gaussian_copula",),
    )
    assert production_only.selected_candidate == "gaussian_copula"
    assert production_only.selectable_candidates == ("gaussian_copula",)

    missing_pair = runs[:-1]
    with pytest.raises(SdvEvaluationError, match="same unique fold/seed pairs"):
        select_candidate(
            runs=missing_pair,
            complexity_ranks={
                "empirical": 0,
                "gaussian_copula": 1,
                "ctgan": 2,
                "tvae": 2,
            },
        )
    invalid = list(runs)
    invalid[1] = CandidateRunScore("gaussian_copula", 0, 7, 1.0, math.nan)
    with pytest.raises(SdvEvaluationError, match="non-finite"):
        select_candidate(
            runs=invalid,
            complexity_ranks={
                "empirical": 0,
                "gaussian_copula": 1,
                "ctgan": 2,
                "tvae": 2,
            },
        )


class _FakeMetricReport:
    def __init__(self, *, detail_score: float, meets_threshold: bool, error: str | None) -> None:
        self.detail_score = detail_score
        self.meets_threshold = meets_threshold
        self.error = error

    def get_score(self) -> float:
        return 0.8

    def get_properties(self) -> pd.DataFrame:
        return pd.DataFrame([{"Property": "Column Pair Trends", "Score": 0.8}])

    def get_details(self, property_name: str) -> pd.DataFrame:
        assert property_name == "Column Pair Trends"
        return pd.DataFrame(
            [
                {
                    "Metric": "CorrelationSimilarity",
                    "Score": self.detail_score,
                    "Meets Threshold?": self.meets_threshold,
                    "Error": self.error,
                }
            ]
        )


def test_metric_validation_only_nulls_explicitly_non_applicable_sdv_details() -> None:
    report = _validated_report(
        _FakeMetricReport(detail_score=math.nan, meets_threshold=False, error=None),
        report_name="quality",
    )
    assert report.details[0].values["Score"] is None
    with pytest.raises(SdvEvaluationError, match="not finite"):
        _validated_report(
            _FakeMetricReport(detail_score=math.nan, meets_threshold=True, error=None),
            report_name="quality",
        )
    with pytest.raises(SdvEvaluationError, match="reported an error"):
        _validated_report(
            _FakeMetricReport(detail_score=0.8, meets_threshold=True, error="metric failed"),
            report_name="quality",
        )


def test_benchmark_orchestration_evaluates_every_fold_seed_and_candidate(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    view = _view()
    settings = BenchmarkSettings(
        fold_count=3,
        seeds=(7, 11),
        fit_selected_model=False,
    )
    candidates = (
        CandidateSpec("empirical", {}),
        CandidateSpec("gaussian_copula", {}),
        CandidateSpec("ctgan", {"epochs": 1, "enable_gpu": False}),
        CandidateSpec("tvae", {"epochs": 1, "enable_gpu": False}),
    )
    monkeypatch.setattr(sdv_harness, "_validate_metadata", lambda _view: None)

    monkeypatch.setattr(sdv_harness, "_candidate_run", _fake_receipt)
    result = run_view_benchmark(
        view=view,
        candidates=candidates,
        settings=settings,
        artifact_dir=None,
    )
    assert len(result.runs) == 3 * 2 * 4
    assert result.selection.selected_candidate == "ctgan"


def test_candidate_failure_is_receipted_and_does_not_fall_back(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    view = _view()
    settings = BenchmarkSettings(fold_count=2, seeds=(7,), fit_selected_model=False)
    candidates = (CandidateSpec("empirical", {}), CandidateSpec("gaussian_copula", {}))
    monkeypatch.setattr(sdv_harness, "_validate_metadata", lambda _view: None)
    monkeypatch.setattr(sdv_harness, "version", _fake_version)
    monkeypatch.setattr(
        sdv_harness, "_torch_environment_receipt", _fake_torch_environment_receipt
    )

    def fail(**kwargs: object) -> CandidateRunReceipt:
        spec = kwargs["spec"]
        assert isinstance(spec, CandidateSpec)
        raise RuntimeError(f"deliberate {spec.name} failure")

    monkeypatch.setattr(sdv_harness, "_candidate_run", fail)
    with pytest.raises(SdvBenchmarkError, match="deliberate empirical failure"):
        run_view_benchmark(
            view=view,
            candidates=candidates,
            settings=settings,
            artifact_dir=tmp_path,
        )
    failure = json.loads((tmp_path / "failure.json").read_text())
    assert failure["status"] == "failed"
    assert failure["completed_runs"] == 0
    assert not (tmp_path / "result.json").exists()


def test_interrupted_run_resumes_existing_arms_and_complete_run_fast_paths(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    view = _view()
    settings = BenchmarkSettings(fold_count=2, seeds=(7,), fit_selected_model=False)
    candidates = (CandidateSpec("empirical", {}), CandidateSpec("gaussian_copula", {}))
    monkeypatch.setattr(sdv_harness, "_validate_metadata", lambda _view: None)
    monkeypatch.setattr(sdv_harness, "version", _fake_version)
    monkeypatch.setattr(
        sdv_harness, "_torch_environment_receipt", _fake_torch_environment_receipt
    )
    first_calls = 0

    def interrupt_after_one(**kwargs: object) -> CandidateRunReceipt:
        nonlocal first_calls
        first_calls += 1
        if first_calls == 2:
            raise KeyboardInterrupt
        return _fake_receipt(**kwargs)

    monkeypatch.setattr(sdv_harness, "_candidate_run", interrupt_after_one)
    with pytest.raises(KeyboardInterrupt):
        run_view_benchmark(
            view=view,
            candidates=candidates,
            settings=settings,
            artifact_dir=tmp_path,
        )
    assert first_calls == 2
    assert len(list((tmp_path / "run-receipts").rglob("*.json"))) == 1
    assert not (tmp_path / "failure.json").exists()

    resumed_calls = 0

    def resume_remaining(**kwargs: object) -> CandidateRunReceipt:
        nonlocal resumed_calls
        resumed_calls += 1
        return _fake_receipt(**kwargs)

    monkeypatch.setattr(sdv_harness, "_candidate_run", resume_remaining)
    resumed = run_view_benchmark(
        view=view,
        candidates=candidates,
        settings=settings,
        artifact_dir=tmp_path,
    )
    assert resumed_calls == 3
    assert len(resumed.runs) == 4
    assert (tmp_path / "manifest.json").is_file()

    def must_not_run(**kwargs: object) -> CandidateRunReceipt:
        raise AssertionError(f"complete fast path reran arm: {kwargs}")

    monkeypatch.setattr(sdv_harness, "_candidate_run", must_not_run)
    repeated = run_view_benchmark(
        view=view,
        candidates=candidates,
        settings=settings,
        artifact_dir=tmp_path,
    )
    assert repeated.to_dict() == resumed.to_dict()


def test_verified_selected_model_handle_supports_repeated_independent_requests(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(sdv_harness, "_validate_metadata", lambda _view: None)
    monkeypatch.setattr(sdv_harness, "_candidate_run", _fake_receipt)
    monkeypatch.setattr(sdv_harness, "version", _fake_version)
    monkeypatch.setattr(
        sdv_harness, "_torch_environment_receipt", _fake_torch_environment_receipt
    )
    run_view_benchmark(
        view=_view(),
        candidates=(CandidateSpec("empirical", {}),),
        settings=BenchmarkSettings(fold_count=2, seeds=(7,), fit_selected_model=True),
        artifact_dir=tmp_path,
    )
    handle = load_selected_model(tmp_path)
    first, first_receipt = handle.sample(request_id="package-a", num_rows=8, seed=101)
    repeated, repeated_receipt = handle.sample(request_id="package-a-repeat", num_rows=8, seed=101)
    _second, second_receipt = handle.sample(request_id="package-b", num_rows=8, seed=102)
    pd.testing.assert_frame_equal(first, repeated)
    assert first_receipt.synthetic_data_sha256 == repeated_receipt.synthetic_data_sha256
    assert second_receipt.synthetic_data_sha256 != first_receipt.synthetic_data_sha256
    assert second_receipt.model_sha256 == first_receipt.model_sha256
    assert second_receipt.proposal.acceptance_yield == 1.0


def test_completed_benchmark_remains_valid_after_staging_directory_rename(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    staging = tmp_path / "benchmark.staging"
    final = tmp_path / "benchmark-final"
    view = _view()
    candidates = (CandidateSpec("empirical", {}),)
    settings = BenchmarkSettings(
        fold_count=2,
        seeds=(7,),
        persist_fold_models=True,
        fit_selected_model=True,
    )
    monkeypatch.setattr(sdv_harness, "_validate_metadata", lambda _view: None)
    monkeypatch.setattr(sdv_harness, "_candidate_run", _fake_receipt)
    monkeypatch.setattr(sdv_harness, "version", _fake_version)
    monkeypatch.setattr(
        sdv_harness, "_torch_environment_receipt", _fake_torch_environment_receipt
    )
    staged_result = run_view_benchmark(
        view=view,
        candidates=candidates,
        settings=settings,
        artifact_dir=staging,
    )
    assert staged_result.selected_model is not None
    assert not Path(staged_result.selected_model.artifact.path).is_absolute()
    assert all(
        run.model_artifact is not None and not Path(run.model_artifact.path).is_absolute()
        for run in staged_result.runs
    )

    staging.rename(final)
    assert not staging.exists()

    def must_not_run(**kwargs: object) -> CandidateRunReceipt:
        raise AssertionError(f"relocated complete run reran arm: {kwargs}")

    monkeypatch.setattr(sdv_harness, "_candidate_run", must_not_run)
    relocated_result = run_view_benchmark(
        view=view,
        candidates=candidates,
        settings=settings,
        artifact_dir=final,
    )
    assert relocated_result.to_dict() == staged_result.to_dict()
    handle = load_selected_model(final)
    sample, receipt = handle.sample(request_id="after-rename", num_rows=6, seed=501)
    assert len(sample) == 6
    assert receipt.proposal.acceptance_yield == 1.0


@pytest.mark.skipif(importlib.util.find_spec("sdv") is None, reason="isolated synthesis runtime")
def test_real_gaussian_and_empirical_smoke(tmp_path: Path) -> None:
    result = run_view_benchmark(
        view=_view(rows=120),
        candidates=(CandidateSpec("empirical", {}), CandidateSpec("gaussian_copula", {})),
        settings=BenchmarkSettings(
            fold_count=2,
            seeds=(19,),
            fit_selected_model=True,
        ),
        artifact_dir=tmp_path,
    )
    assert len(result.runs) == 4
    assert all(run.evaluation.diagnostic.score == 1.0 for run in result.runs)
    assert result.selected_model is not None
    assert (tmp_path / result.selected_model.artifact.path).is_file()
    assert (tmp_path / "manifest.json").is_file()
