from __future__ import annotations

import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pandas as pd
import pytest

from document_ocr.synthesis import scenario_benchmark, sdv_harness
from document_ocr.synthesis.scenario_benchmark import (
    PartyStructureBenchmarkSettings,
    ScenarioBenchmarkError,
    party_structure_candidate_specs,
    run_party_structure_benchmark,
)
from document_ocr.synthesis.scenario_views import PARTY_STRUCTURE_COLUMNS
from document_ocr.synthesis.sdv_evaluation import (
    EvaluationBundle,
    MetricReport,
    SdvEvaluationError,
)
from document_ocr.synthesis.sdv_harness import (
    AcceleratorSelection,
    BenchmarkView,
    CandidateSpec,
    PhaseResources,
    ProposalReceipt,
    fit_candidate_model,
)


def _party_data(rows: int = 96) -> pd.DataFrame:
    values: list[dict[str, Any]] = []
    modes = (
        ("shipper", "origin", "endpoint_port", "city_and_country"),
        ("consignee", "destination", "other_same_country", "city_and_country"),
        ("carrier", "third_country", "third_country", "city_and_country"),
        ("notifyParties", "same_as", "missing", "neither"),
    )
    for index in range(rows):
        role, relation, city_mode, geography = modes[index % len(modes)]
        address = index % 3 != 0
        phone_count = index % 3
        email_count = (index // 2) % 2
        website_count = (index // 3) % 2
        address_characters = 12 + index % 9 if address else 0
        address_words = 3 + index % 3 if address else 0
        values.append(
            {
                "role": role,
                "relation_to_route": relation,
                "city_mode": city_mode,
                "geography_presence": geography,
                "name_present": index % 7 != 0,
                "contact_name_present": index % 5 == 0,
                "contact_count_profile": (
                    f"phone={phone_count}|email={email_count}|website={website_count}"
                ),
                "address_character_count": address_characters,
                "address_word_count": address_words,
            }
        )
    return pd.DataFrame.from_records(values, columns=PARTY_STRUCTURE_COLUMNS)


def _metadata() -> dict[str, Any]:
    categoricals = {
        "role",
        "relation_to_route",
        "city_mode",
        "geography_presence",
        "contact_count_profile",
    }
    booleans = {"name_present", "contact_name_present"}
    columns = {}
    for column in PARTY_STRUCTURE_COLUMNS:
        if column in categoricals:
            columns[column] = {"sdtype": "categorical"}
        elif column in booleans:
            columns[column] = {"sdtype": "boolean"}
        else:
            columns[column] = {
                "sdtype": "numerical",
                "computer_representation": "Int64",
            }
    return {
        "METADATA_SPEC_VERSION": "V1",
        "tables": {"party_structure": {"columns": columns}},
        "relationships": [],
    }


def _view(rows: int = 96) -> BenchmarkView:
    data = _party_data(rows)
    return BenchmarkView(
        name="party_structure",
        data=data,
        metadata=_metadata(),
        row_ids=tuple(f"party-{index}" for index in range(rows)),
        group_ids=tuple(f"template-{index // 4}" for index in range(rows)),
        partition_labels=("train",) * rows,
    )


def _cpu() -> AcceleratorSelection:
    return AcceleratorSelection(
        enable_gpu=False,
        torch_version="test-cpu",
        torch_cuda_version=None,
        cuda_available=False,
        cuda_device_count=0,
        device="cpu",
        device_name=None,
        device_capability=None,
        device_total_memory_bytes=None,
    )


def _gpu() -> AcceleratorSelection:
    return AcceleratorSelection(
        enable_gpu=True,
        torch_version="test-cuda",
        torch_cuda_version="13.0",
        cuda_available=True,
        cuda_device_count=1,
        device="cuda:0",
        device_name="test GPU",
        device_capability=(9, 0),
        device_total_memory_bytes=24 * 1024**3,
    )


def _resources() -> PhaseResources:
    return PhaseResources(
        elapsed_seconds=0.01,
        rss_before_bytes=100,
        rss_after_bytes=110,
        peak_rss_before_bytes=120,
        peak_rss_after_bytes=130,
        accelerator=_cpu(),
        cuda_allocated_before_bytes=None,
        cuda_allocated_after_bytes=None,
        cuda_reserved_before_bytes=None,
        cuda_reserved_after_bytes=None,
        cuda_peak_allocated_bytes=None,
        cuda_peak_reserved_bytes=None,
    )


def _evaluation() -> EvaluationBundle:
    return EvaluationBundle(
        diagnostic=MetricReport(
            "diagnostic", 1.0, {"Data Validity": 1.0, "Data Structure": 1.0}, ()
        ),
        quality=MetricReport("quality", 0.8, {"Column Shapes": 0.8, "Column Pair Trends": 0.8}, ()),
    )


class _FakeFitted:
    def __init__(self, candidate: str, parameters: Any, train: Any) -> None:
        self.candidate = candidate
        self.parameters = parameters
        self.train = train
        self.fit = _resources()

    def sample(self, **kwargs: Any) -> tuple[Any, ProposalReceipt, PhaseResources]:
        requested = kwargs["requested_rows"]
        repeats = requested // len(self.train) + 1
        synthetic = pd.concat([self.train] * repeats, ignore_index=True).iloc[:requested].copy()
        if self.candidate == "gaussian_copula":
            mask = synthetic["address_character_count"] > 0
            synthetic.loc[mask, "address_character_count"] += 1
        accepted = list(kwargs["acceptance"](synthetic))
        assert all(accepted)
        proposal = ProposalReceipt(
            requested_rows=requested,
            raw_proposals=requested,
            accepted_proposals_before_truncation=requested,
            rejected_proposals=0,
            output_rows=requested,
            acceptance_yield=1.0,
            bounded_maximum_raw_proposals=requested * kwargs["proposal_multiplier"],
        )
        return synthetic, proposal, _resources()


def _cpu_settings() -> PartyStructureBenchmarkSettings:
    candidates = party_structure_candidate_specs(
        enable_gpu=False,
        neural_epochs=3,
        neural_batch_size=20,
    )
    return PartyStructureBenchmarkSettings(
        scope="cpu_baselines",
        candidates=candidates[:2],
        fold_count=2,
        seeds=(7,),
        fold_seed=11,
    )


def _gpu_settings() -> PartyStructureBenchmarkSettings:
    return PartyStructureBenchmarkSettings(
        scope="full_gpu",
        candidates=party_structure_candidate_specs(
            enable_gpu=True,
            neural_epochs=3,
            neural_batch_size=20,
        ),
        fold_count=2,
        seeds=(7,),
        fold_seed=11,
    )


def test_candidate_surface_is_explicit_and_full_scope_requires_gpu() -> None:
    gpu = party_structure_candidate_specs(
        enable_gpu=True,
        neural_epochs=300,
        neural_batch_size=500,
    )
    settings = PartyStructureBenchmarkSettings(
        scope="full_gpu",
        candidates=gpu,
        fold_count=3,
        seeds=(17, 23),
        fold_seed=5,
    )

    assert settings.candidates[2].parameters["enable_gpu"] is True
    assert settings.candidates[3].parameters["enable_gpu"] is True
    assert settings.candidates[2].parameters["epochs"] == 300
    with pytest.raises(ValueError, match="full_gpu requires enable_gpu=true"):
        PartyStructureBenchmarkSettings(
            scope="full_gpu",
            candidates=party_structure_candidate_specs(
                enable_gpu=False,
                neural_epochs=3,
                neural_batch_size=20,
            ),
            fold_count=2,
            seeds=(1,),
            fold_seed=2,
        )


def test_public_fit_seam_retains_harness_fit_and_sampling_paths(monkeypatch: Any) -> None:
    view = _view(16)
    train = view.data.iloc[:8].copy()
    model = SimpleNamespace(accelerator=_cpu())
    monkeypatch.setattr(sdv_harness, "_validate_metadata", lambda _view: None)
    monkeypatch.setattr(
        sdv_harness,
        "_fit_model",
        lambda **_kwargs: (model, _resources()),
    )
    handle = fit_candidate_model(
        view=view,
        spec=CandidateSpec("empirical", {}),
        train_data=train,
        seed=3,
    )

    assert handle.candidate == "empirical"
    assert handle.fit == _resources()
    assert handle._model is model


def test_cpu_comparison_publishes_pii_free_grouped_immutable_receipts(
    tmp_path: Path, monkeypatch: Any
) -> None:
    monkeypatch.setattr(
        scenario_benchmark,
        "fit_candidate_model",
        lambda **kwargs: _FakeFitted(
            kwargs["spec"].name,
            kwargs["spec"].parameters,
            kwargs["train_data"],
        ),
    )
    monkeypatch.setattr(
        scenario_benchmark, "evaluate_single_table", lambda **_kwargs: _evaluation()
    )
    monkeypatch.setattr(scenario_benchmark, "version", lambda package: f"{package}-test")
    artifact_dir = tmp_path / "party-benchmark"

    result = run_party_structure_benchmark(
        view=_view(),
        settings=_cpu_settings(),
        artifact_dir=artifact_dir,
    )

    assert not result.production_selection_performed
    assert result.status == "complete"
    assert result.requested_candidates == ("empirical", "gaussian_copula")
    assert result.complete_candidates == result.requested_candidates
    assert result.failed_candidates == ()
    assert result.candidate_failures == ()
    assert len(result.runs) == 4
    assert result.runs[0].fold == result.runs[2].fold
    paired = {
        (run.fold.index, run.seed): (
            run.train_row_membership_sha256,
            run.validation_row_membership_sha256,
            run.train_data_sha256,
            run.validation_data_sha256,
        )
        for run in result.runs
        if run.candidate == "empirical"
    }
    assert all(
        paired[(run.fold.index, run.seed)]
        == (
            run.train_row_membership_sha256,
            run.validation_row_membership_sha256,
            run.train_data_sha256,
            run.validation_data_sha256,
        )
        for run in result.runs
        if run.candidate == "gaussian_copula"
    )
    summaries = {summary.candidate: summary for summary in result.summaries}
    assert summaries["empirical"].novel_fraction == 0
    assert summaries["gaussian_copula"].novel_fraction > 0
    assert all(run.validity.to_dict()["output_valid_fraction"] == 1.0 for run in result.runs)
    assert all(run.fit.to_dict()["accelerator"]["device"] == "cpu" for run in result.runs)
    assert (artifact_dir / "manifest.json").is_file()
    assert (
        json.loads((artifact_dir / "result.json").read_text())["production_model_artifact"] is None
    )
    artifact_text = "\n".join(path.read_text() for path in artifact_dir.rglob("*.json")).upper()
    for private_value in (
        "PRIVATE ALICE",
        "EXPORT ROAD",
        "PRIVATE@EXAMPLE.TEST",
        "SWITZERLAND",
    ):
        assert private_value not in artifact_text
    with pytest.raises(ScenarioBenchmarkError, match="new or empty"):
        run_party_structure_benchmark(
            view=_view(),
            settings=_cpu_settings(),
            artifact_dir=artifact_dir,
        )


def test_wrong_view_or_raw_column_is_rejected_before_fit(tmp_path: Path) -> None:
    view = _view()
    view.data["raw_name"] = "PRIVATE ALICE"

    with pytest.raises(ValueError, match="exact PII-free"):
        run_party_structure_benchmark(
            view=view,
            settings=_cpu_settings(),
            artifact_dir=tmp_path / "must-not-exist",
        )


@pytest.mark.parametrize(
    ("updates", "violation"),
    (
        (
            {"relation_to_route": "missing", "geography_presence": "country_only"},
            "country_requires_relation",
        ),
        (
            {"city_mode": "missing", "geography_presence": "city_and_country"},
            "city_country_require_mode",
        ),
    ),
)
def test_source_domain_invariants_are_rejected_before_fit(
    tmp_path: Path,
    updates: dict[str, Any],
    violation: str,
) -> None:
    view = _view()
    for column, value in updates.items():
        view.data.loc[0, column] = value

    with pytest.raises(ValueError, match=violation):
        run_party_structure_benchmark(
            view=view,
            settings=_cpu_settings(),
            artifact_dir=tmp_path / "must-not-exist",
        )


def test_fitted_candidate_identity_mismatch_fails_closed(tmp_path: Path, monkeypatch: Any) -> None:
    monkeypatch.setattr(scenario_benchmark, "version", lambda package: f"{package}-test")
    monkeypatch.setattr(
        scenario_benchmark,
        "fit_candidate_model",
        lambda **kwargs: _FakeFitted(
            "gaussian_copula",
            kwargs["spec"].parameters,
            kwargs["train_data"],
        ),
    )
    artifact_dir = tmp_path / "identity-failure"

    with pytest.raises(ScenarioBenchmarkError, match="fitted candidate identity"):
        run_party_structure_benchmark(
            view=_view(),
            settings=_cpu_settings(),
            artifact_dir=artifact_dir,
        )

    assert json.loads((artifact_dir / "failure.json").read_text())["completed_runs"] == 0


def test_missing_benchmark_environment_publishes_failure_receipt(
    tmp_path: Path, monkeypatch: Any
) -> None:
    def missing_version(package: str) -> str:
        raise RuntimeError(f"missing distribution: {package}")

    monkeypatch.setattr(scenario_benchmark, "version", missing_version)
    artifact_dir = tmp_path / "environment-failure"

    with pytest.raises(ScenarioBenchmarkError, match="missing distribution: sdv"):
        run_party_structure_benchmark(
            view=_view(),
            settings=_cpu_settings(),
            artifact_dir=artifact_dir,
        )

    failure = json.loads((artifact_dir / "failure.json").read_text())
    assert failure["completed_runs"] == 0
    assert failure["error_type"] == "RuntimeError"
    assert not (artifact_dir / "benchmark-contract.json").exists()


def test_full_gpu_preflight_fails_before_any_candidate_fit(
    tmp_path: Path, monkeypatch: Any
) -> None:
    settings = PartyStructureBenchmarkSettings(
        scope="full_gpu",
        candidates=party_structure_candidate_specs(
            enable_gpu=True,
            neural_epochs=3,
            neural_batch_size=20,
        ),
        fold_count=2,
        seeds=(7,),
        fold_seed=11,
    )
    monkeypatch.setattr(scenario_benchmark, "version", lambda package: f"{package}-test")
    monkeypatch.setattr(
        scenario_benchmark,
        "preflight_sdv_accelerator",
        lambda **_kwargs: (_ for _ in ()).throw(RuntimeError("no visible CUDA device")),
    )
    fit_called = False

    def forbidden_fit(**_kwargs: Any) -> Any:
        nonlocal fit_called
        fit_called = True
        raise AssertionError("fit must not be reached")

    monkeypatch.setattr(scenario_benchmark, "fit_candidate_model", forbidden_fit)
    artifact_dir = tmp_path / "gpu-failure"

    with pytest.raises(ScenarioBenchmarkError, match="no visible CUDA device"):
        run_party_structure_benchmark(
            view=_view(),
            settings=settings,
            artifact_dir=artifact_dir,
        )

    assert not fit_called
    failure = json.loads((artifact_dir / "failure.json").read_text())
    assert failure["completed_runs"] == 0
    assert not (artifact_dir / "manifest.json").exists()


def test_full_gpu_preflight_cannot_silently_return_cpu(tmp_path: Path, monkeypatch: Any) -> None:
    settings = PartyStructureBenchmarkSettings(
        scope="full_gpu",
        candidates=party_structure_candidate_specs(
            enable_gpu=True,
            neural_epochs=3,
            neural_batch_size=20,
        ),
        fold_count=2,
        seeds=(7,),
        fold_seed=11,
    )
    monkeypatch.setattr(scenario_benchmark, "version", lambda package: f"{package}-test")
    monkeypatch.setattr(
        scenario_benchmark,
        "preflight_sdv_accelerator",
        lambda **_kwargs: _cpu(),
    )
    fit_called = False

    def forbidden_fit(**_kwargs: Any) -> Any:
        nonlocal fit_called
        fit_called = True
        raise AssertionError("fit must not be reached")

    monkeypatch.setattr(scenario_benchmark, "fit_candidate_model", forbidden_fit)
    artifact_dir = tmp_path / "gpu-cpu-failure"

    with pytest.raises(ScenarioBenchmarkError, match="did not select a GPU"):
        run_party_structure_benchmark(
            view=_view(),
            settings=settings,
            artifact_dir=artifact_dir,
        )

    assert not fit_called
    assert json.loads((artifact_dir / "failure.json").read_text())["completed_runs"] == 0
    assert not (artifact_dir / "gpu-preflight.json").exists()


def test_non_empirical_evaluation_failure_is_receipted_and_later_candidates_continue(
    tmp_path: Path, monkeypatch: Any
) -> None:
    fitted_candidates: list[str] = []

    def fake_fit(**kwargs: Any) -> _FakeFitted:
        fitted_candidates.append(kwargs["spec"].name)
        return _FakeFitted(
            kwargs["spec"].name,
            kwargs["spec"].parameters,
            kwargs["train_data"],
        )

    evaluation_calls = 0

    def evaluate(**_kwargs: Any) -> EvaluationBundle:
        nonlocal evaluation_calls
        evaluation_calls += 1
        if evaluation_calls == 3:
            raise SdvEvaluationError("quality pair score is non-finite")
        return _evaluation()

    monkeypatch.setattr(scenario_benchmark, "fit_candidate_model", fake_fit)
    monkeypatch.setattr(scenario_benchmark, "evaluate_single_table", evaluate)
    monkeypatch.setattr(scenario_benchmark, "preflight_sdv_accelerator", lambda **_kwargs: _gpu())
    monkeypatch.setattr(scenario_benchmark, "version", lambda package: f"{package}-test")
    artifact_dir = tmp_path / "recoverable-model-failure"

    result = run_party_structure_benchmark(
        view=_view(),
        settings=_gpu_settings(),
        artifact_dir=artifact_dir,
    )

    assert result.status == "complete_with_candidate_failures"
    assert result.requested_candidates == (
        "empirical",
        "gaussian_copula",
        "ctgan",
        "tvae",
    )
    assert result.complete_candidates == ("empirical", "ctgan", "tvae")
    assert result.failed_candidates == ("gaussian_copula",)
    assert [summary.candidate for summary in result.summaries] == [
        "empirical",
        "ctgan",
        "tvae",
    ]
    assert [run.candidate for run in result.runs] == [
        "empirical",
        "empirical",
        "ctgan",
        "ctgan",
        "tvae",
        "tvae",
    ]
    assert fitted_candidates == [
        "empirical",
        "empirical",
        "gaussian_copula",
        "ctgan",
        "ctgan",
        "tvae",
        "tvae",
    ]
    assert len(result.candidate_failures) == 1
    failure = result.candidate_failures[0]
    assert failure.candidate == "gaussian_copula"
    assert failure.fold.index == 0
    assert failure.seed == 7
    assert failure.failure_stage == "evaluation"
    assert failure.error_type == "SdvEvaluationError"
    assert failure.message == "quality pair score is non-finite"
    assert failure.validity.to_dict()["output_valid_fraction"] == 1.0
    assert failure.proposal.output_rows == failure.novelty.rows
    failure_path = artifact_dir / "candidate-failures/gaussian_copula/fold-00-seed-7.json"
    published_failure = json.loads(failure_path.read_text())
    assert published_failure["evaluation"] is None
    assert "quality" not in published_failure
    result_json = json.loads((artifact_dir / "result.json").read_text())
    assert result_json["status"] == "complete_with_candidate_failures"
    assert result_json["failed_candidates"] == ["gaussian_copula"]
    assert [row["candidate"] for row in result_json["summaries"]] == [
        "empirical",
        "ctgan",
        "tvae",
    ]
    manifest = json.loads((artifact_dir / "manifest.json").read_text())
    assert manifest["status"] == "complete_with_candidate_failures"
    assert manifest["complete_candidates"] == ["empirical", "ctgan", "tvae"]
    assert manifest["failed_candidates"] == ["gaussian_copula"]
    assert any(
        row["path"] == "candidate-failures/gaussian_copula/fold-00-seed-7.json"
        and row["sha256"] == scenario_benchmark.sha256_file(failure_path)
        for row in manifest["files"]
    )
    assert not (artifact_dir / "failure.json").exists()


def test_empirical_evaluation_failure_aborts_the_whole_benchmark(
    tmp_path: Path, monkeypatch: Any
) -> None:
    fitted_candidates: list[str] = []

    def fake_fit(**kwargs: Any) -> _FakeFitted:
        fitted_candidates.append(kwargs["spec"].name)
        return _FakeFitted(
            kwargs["spec"].name,
            kwargs["spec"].parameters,
            kwargs["train_data"],
        )

    monkeypatch.setattr(scenario_benchmark, "fit_candidate_model", fake_fit)
    monkeypatch.setattr(
        scenario_benchmark,
        "evaluate_single_table",
        lambda **_kwargs: (_ for _ in ()).throw(SdvEvaluationError("empirical score failed")),
    )
    monkeypatch.setattr(scenario_benchmark, "preflight_sdv_accelerator", lambda **_kwargs: _gpu())
    monkeypatch.setattr(scenario_benchmark, "version", lambda package: f"{package}-test")
    artifact_dir = tmp_path / "empirical-failure"

    with pytest.raises(ScenarioBenchmarkError, match="SdvEvaluationError: empirical score failed"):
        run_party_structure_benchmark(
            view=_view(),
            settings=_gpu_settings(),
            artifact_dir=artifact_dir,
        )

    assert fitted_candidates == ["empirical"]
    failure = json.loads((artifact_dir / "failure.json").read_text())
    assert failure["error_type"] == "SdvEvaluationError"
    assert not (artifact_dir / "candidate-failures").exists()
    assert not (artifact_dir / "result.json").exists()
    assert not (artifact_dir / "manifest.json").exists()


@pytest.mark.skipif(
    importlib.util.find_spec("sdv") is None
    or importlib.util.find_spec("sdmetrics") is None
    or importlib.util.find_spec("torch") is None,
    reason="real CPU benchmark requires the dedicated synthesis environment",
)
def test_real_cpu_empirical_and_gaussian_smoke(tmp_path: Path) -> None:
    result = run_party_structure_benchmark(
        view=_view(96),
        settings=_cpu_settings(),
        artifact_dir=tmp_path / "real-cpu",
    )

    assert len(result.runs) == 4
    assert {run.candidate for run in result.runs} == {"empirical", "gaussian_copula"}
    assert all(run.evaluation.diagnostic.score == 1.0 for run in result.runs)
    assert all(run.fit.accelerator.device == "cpu" for run in result.runs)
