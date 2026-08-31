from __future__ import annotations

import importlib.util
import json
import math
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pandas as pd
import pytest

import document_ocr.synthesis.profile_sdv as profile_sdv_module
from document_ocr.synthesis.profile_sdv import (
    MeasureProfile,
    NumericSourceRow,
    ProfileBenchmarkSettings,
    ProfileNumericRow,
    ProfileRouteError,
    ProfileRouteRequest,
    ProfileRoutingSettings,
    ProfileSdvError,
    QualityAuditSettings,
    RouteSupportThreshold,
    _evaluate_profile,
    _from_physical_factors,
    _novel_acceptance,
    _proposal_rows,
    _to_physical_factors,
    _validity,
    audit_source_rows,
    benchmark_and_sample_profile,
    build_dense_profile_view,
    candidate_eligibility,
    profile_candidate_specs,
    route_profile_requests,
)


def _source_row(
    index: int,
    *,
    identity: str = "PACKAGE_CARTON",
    family: str = "BOX_LIKE",
    role: str = "direct_goods",
    quantity: int | None = 10,
    gross_value: float | None = None,
    gross_unit: str | None = None,
    net_value: float | None = None,
    net_unit: str | None = None,
    volume_value: float | None = None,
    volume_unit: str | None = None,
    partition: str = "train",
) -> NumericSourceRow:
    gross = gross_value if gross_value is not None else 100.0 + index * 3.0
    net = net_value if net_value is not None else 90.0 + index * 2.0
    return NumericSourceRow(
        row_id=f"row-{index:03d}",
        document_id=f"doc-{index:03d}",
        template_id=f"template-{index:03d}",
        partition=partition,
        exact_identity=identity,
        semantic_family=family,
        package_role=role,
        quantity=quantity,
        gross_value=gross,
        gross_unit=gross_unit or "KG",
        net_value=net,
        net_unit=net_unit or "KG",
        volume_value=volume_value,
        volume_unit=volume_unit,
    )


def _routing_settings(*, rows: int = 4, templates: int = 3) -> ProfileRoutingSettings:
    threshold = RouteSupportThreshold(rows, templates)
    return ProfileRoutingSettings(threshold, threshold, threshold)


def test_novel_acceptance_rejects_exact_fit_rows_after_domain_validation() -> None:
    source = pd.DataFrame({"quantity": [1, 2]})
    candidates = pd.DataFrame({"quantity": [1, 3, 0]})

    accepted = _novel_acceptance(MeasureProfile(False, False, False), source)(candidates)

    assert accepted == [False, True, False]


@pytest.mark.skipif(importlib.util.find_spec("sdmetrics") is None, reason="SDMetrics unavailable")
def test_profile_quality_omits_pair_trends_when_real_pairs_are_not_applicable() -> None:
    real = pd.DataFrame({"quantity": [1, 1, 2, 2], "measure": [10.0] * 4})
    synthetic = pd.DataFrame({"quantity": [1, 2, 2, 3], "measure": [9.0, 10.0, 10.0, 11.0]})

    result = _evaluate_profile(
        real_data=real,
        synthetic_data=synthetic,
        metadata={"tables": {"independent_profile": {}}},
        table_name="independent_profile",
        diagnostic_reference_data=real,
    )

    assert result.quality.properties.keys() == {"Column Shapes"}
    assert {detail.values["Column"] for detail in result.quality.details} == {
        "quantity",
        "measure",
    }


@pytest.mark.skipif(importlib.util.find_spec("sdmetrics") is None, reason="SDMetrics unavailable")
def test_profile_quality_scores_only_real_correlated_pairs() -> None:
    real = pd.DataFrame({"quantity": [1, 2, 3, 4], "measure": [2.0, 4.0, 6.0, 8.0]})
    synthetic = pd.DataFrame({"quantity": [1, 2, 3, 4], "measure": [2.0, 3.0, 7.0, 9.0]})

    result = _evaluate_profile(
        real_data=real,
        synthetic_data=synthetic,
        metadata={"tables": {"correlated_profile": {}}},
        table_name="correlated_profile",
        diagnostic_reference_data=real,
    )

    assert result.quality.properties.keys() == {
        "Column Shapes",
        "Column Pair Trends",
    }
    assert (
        sum(detail.property_name == "Column Pair Trends" for detail in result.quality.details) == 1
    )


@pytest.mark.skipif(importlib.util.find_spec("sdmetrics") is None, reason="SDMetrics unavailable")
def test_profile_quality_scores_synthetic_constant_pair_as_zero() -> None:
    real = pd.DataFrame({"quantity": [1, 2, 3, 4], "measure": [2.0, 4.0, 6.0, 8.0]})
    synthetic = pd.DataFrame({"quantity": [2, 2, 2, 2], "measure": [2.0, 3.0, 7.0, 9.0]})

    result = _evaluate_profile(
        real_data=real,
        synthetic_data=synthetic,
        metadata={"tables": {"constant_profile": {}}},
        table_name="constant_profile",
        diagnostic_reference_data=real,
    )

    pair_detail = next(
        detail for detail in result.quality.details if detail.property_name == "Column Pair Trends"
    )
    assert result.quality.properties["Column Pair Trends"] == 0.0
    assert pair_detail.values["Status"] == "synthetic_constant_scored_zero"


def test_source_audit_normalizes_units_and_derives_per_driver_values() -> None:
    source = NumericSourceRow(
        row_id="row-1",
        document_id="doc-1",
        template_id="template-1",
        partition="train",
        exact_identity="PACKAGE_DRUM",
        semantic_family="CYLINDRICAL",
        package_role="direct_goods",
        quantity=4,
        gross_value=0.12,
        gross_unit="metric_tonne",
        net_value=220.462262,
        net_unit="LB",
        volume_value=2.0,
        volume_unit="cubic_metre",
    )
    # A second template is not required for source-quality auditing.
    audit = audit_source_rows((source,), settings=QualityAuditSettings())
    assert audit.quarantined_count == 0
    assert audit.accepted_count == 1
    row = audit.accepted_rows[0]
    assert row.profile == MeasureProfile(True, True, True)
    assert row.gross_per_driver_package_kg == pytest.approx(30.0)
    assert row.net_per_driver_package_kg == pytest.approx(25.0)
    assert row.volume_per_driver_package_m3 == pytest.approx(0.5)


def test_source_audit_quarantines_thousand_fold_anomalies_without_clipping() -> None:
    normal = tuple(
        _source_row(
            index,
            gross_value=100.0 + index,
            net_value=90.0 + index,
        )
        for index in range(10)
    )
    ratio_outlier = _source_row(
        100,
        gross_value=1070.0,
        net_value=1.0,
    )
    jointly_scaled = _source_row(
        101,
        gross_value=105.0 * 1004.5,
        net_value=95.0 * 1004.5,
    )
    audit = audit_source_rows(
        (*normal, ratio_outlier, jointly_scaled),
        settings=QualityAuditSettings(
            maximum_gross_to_net_ratio=50.0,
            robust_minimum_rows=8,
            robust_z_threshold=8.0,
            maximum_median_factor=100.0,
        ),
    )
    by_id = {row.row_id: row for row in audit.quarantined_rows}
    assert set(by_id) == {"row-100", "row-101"}
    assert {reason.code for reason in by_id["row-100"].reasons} == {"gross_to_net_ratio_outlier"}
    assert {reason.code for reason in by_id["row-101"].reasons} == {
        "robust_per_driver_measure_outlier"
    }
    assert len(audit.accepted_rows) == len(normal)
    # The receipt exposes observed source-scale values; it never substitutes a cap.
    observations = [reason.observed for reason in by_id["row-101"].reasons]
    assert max(value for value in observations if value is not None) > 9_000


def test_source_audit_rejects_validation_leakage_before_fitting() -> None:
    with pytest.raises(ValueError, match="not train-only"):
        audit_source_rows(
            (_source_row(0, partition="validation"),),
            settings=QualityAuditSettings(),
        )


def test_dense_view_has_one_profile_no_nulls_and_exact_numeric_columns() -> None:
    audit = audit_source_rows(
        tuple(_source_row(index) for index in range(6)),
        settings=QualityAuditSettings(robust_minimum_rows=8),
    )
    view = build_dense_profile_view(
        audit.accepted_rows,
        profile=MeasureProfile(True, True, False),
        name="profile_g1n1v0",
    )
    assert list(view.data.columns) == [
        "quantity",
        "gross_per_driver_package_kg",
        "net_per_driver_package_kg",
    ]
    assert not view.data.isna().any().any()
    assert str(view.data.dtypes["quantity"]) == "int64"
    assert str(view.data.dtypes["gross_per_driver_package_kg"]) == "float64"

    mixed = audit_source_rows(
        (_source_row(20, volume_value=2.0, volume_unit="CBM"),),
        settings=QualityAuditSettings(),
    ).accepted_rows[0]
    with pytest.raises(ValueError, match="mixes hard measure profiles"):
        build_dense_profile_view(
            (*audit.accepted_rows, mixed),
            profile=MeasureProfile(True, True, False),
            name="mixed",
        )


def test_router_receipts_each_broadening_step_and_never_uses_global_fallback() -> None:
    rows = tuple(
        audit_source_rows(
            tuple(
                _source_row(
                    index,
                    identity="PACKAGE_CARTON" if index < 2 else "PACKAGE_CASE",
                    family="BOX_LIKE",
                )
                for index in range(6)
            ),
            settings=QualityAuditSettings(robust_minimum_rows=8),
        ).accepted_rows
    )
    request = ProfileRouteRequest(
        "request-1",
        "PACKAGE_CARTON",
        "BOX_LIKE",
        "direct_goods",
        MeasureProfile(True, True, False),
    )
    routing = route_profile_requests(rows, (request,), settings=_routing_settings())
    decision = routing.decisions[0]
    assert decision.selected_tier == "semantic_family_role"
    assert [attempt.tier for attempt in decision.attempts] == [
        "exact_identity_role",
        "semantic_family_role",
    ]
    assert decision.attempts[0].eligible is False
    assert decision.attempts[1].eligible is True
    assert routing.to_dict()["coverage"] == 1.0

    absent = ProfileRouteRequest(
        "request-2",
        "PACKAGE_BAG",
        "FLEXIBLE",
        "outer_transport",
        MeasureProfile(True, True, False),
    )
    failed = route_profile_requests(rows, (absent,), settings=_routing_settings())
    assert failed.unroutable == 1
    assert failed.decisions[0].selected_tier is None
    assert len(failed.decisions[0].attempts) == 3


def test_candidate_eligibility_keeps_neural_models_visible_but_ineligible() -> None:
    audit = audit_source_rows(
        tuple(_source_row(index) for index in range(12)),
        settings=QualityAuditSettings(robust_minimum_rows=20),
    )
    view = build_dense_profile_view(
        audit.accepted_rows,
        profile=MeasureProfile(True, True, False),
        name="profile",
    )
    settings = ProfileBenchmarkSettings(
        fold_count=3,
        seeds=(7,),
        neural_minimum_rows=100,
        neural_minimum_templates=50,
    )
    receipt = candidate_eligibility(
        view,
        profile_candidate_specs(
            settings,
            profile=MeasureProfile(True, True, False),
        ),
    )
    by_name = {row.candidate: row for row in receipt}
    assert by_name["empirical"].eligible
    assert by_name["gaussian_copula_default"].eligible
    assert by_name["gaussian_copula_gaussian_kde"].eligible
    assert by_name["gaussian_copula_domain_mixed"].eligible
    assert by_name["gaussian_copula_physical_factors"].eligible
    assert not by_name["ctgan"].eligible
    assert not by_name["tvae"].eligible
    assert by_name["ctgan"].reason == "below_configured_minimum_rows_or_distinct_templates"


def test_candidate_inventory_and_production_selection_are_explicitly_configurable() -> None:
    settings = ProfileBenchmarkSettings(
        fold_count=3,
        seeds=(7,),
        candidate_identifiers=(
            "empirical",
            "gaussian_copula_gaussian_kde",
        ),
        production_selectable_candidates=("gaussian_copula_gaussian_kde",),
    )

    assert [
        row.identifier
        for row in profile_candidate_specs(
            settings,
            profile=MeasureProfile(True, True, False),
        )
    ] == [
        "empirical",
        "gaussian_copula_gaussian_kde",
    ]

    with pytest.raises(ValueError, match="including empirical"):
        ProfileBenchmarkSettings(
            fold_count=3,
            seeds=(7,),
            candidate_identifiers=("gaussian_copula_default",),
        )


def test_physical_factor_candidate_is_receipted_but_ineligible_without_measures() -> None:
    settings = ProfileBenchmarkSettings(fold_count=3, seeds=(7,))
    profile = MeasureProfile(False, False, False)
    candidates = profile_candidate_specs(settings, profile=profile)
    rows = tuple(
        ProfileNumericRow(
            row_id=f"row-{index}",
            document_id=f"doc-{index}",
            template_id=f"template-{index}",
            exact_identity="PACKAGE_CARTON",
            semantic_family="BOX_LIKE",
            package_role="direct_goods",
            profile=profile,
            quantity=index + 1,
            gross_per_driver_package_kg=None,
            net_per_driver_package_kg=None,
            volume_per_driver_package_m3=None,
        )
        for index in range(5)
    )
    view = build_dense_profile_view(
        rows,
        name="quantity-only",
        profile=profile,
    )
    eligibility = {row.candidate: row for row in candidate_eligibility(view, candidates)}

    physical = eligibility["gaussian_copula_physical_factors"]
    assert not physical.eligible
    assert physical.reason == "physical_factor_representation_requires_a_measure"


@pytest.mark.parametrize(
    ("profile", "expected_distributions"),
    [
        (
            MeasureProfile(False, False, False),
            {"quantity": "gaussian_kde"},
        ),
        (
            MeasureProfile(True, True, False),
            {
                "quantity": "gaussian_kde",
                "gross_per_driver_package_kg": "gamma",
                "net_per_driver_package_kg": "gamma",
            },
        ),
        (
            MeasureProfile(False, False, True),
            {
                "quantity": "gaussian_kde",
                "volume_per_driver_package_m3": "beta",
            },
        ),
        (
            MeasureProfile(True, True, True),
            {
                "quantity": "gaussian_kde",
                "gross_per_driver_package_kg": "gamma",
                "net_per_driver_package_kg": "gamma",
                "volume_per_driver_package_m3": "beta",
            },
        ),
    ],
)
def test_domain_mixed_candidate_filters_marginals_to_dense_profile_columns(
    profile: MeasureProfile,
    expected_distributions: dict[str, str],
) -> None:
    settings = ProfileBenchmarkSettings(fold_count=3, seeds=(7,))

    candidates = profile_candidate_specs(settings, profile=profile)
    candidate = next(row for row in candidates if row.identifier == "gaussian_copula_domain_mixed")

    assert candidate.backend == "gaussian_copula"
    assert candidate.parameters == {
        "enforce_min_max_values": True,
        "enforce_rounding": True,
        "numerical_distributions": expected_distributions,
    }


def test_physical_factor_representation_is_reversible_and_preserves_mass_order() -> None:
    source = pd.DataFrame(
        {
            "quantity": [4, 11],
            "gross_per_driver_package_kg": [100.0, 250.0],
            "net_per_driver_package_kg": [90.0, 250.0],
            "volume_per_driver_package_m3": [0.4, 1.25],
        }
    )
    profile = MeasureProfile(True, True, True)

    factors = _to_physical_factors(source, profile)
    restored = _from_physical_factors(factors, profile)

    assert list(restored.columns) == list(source.columns)
    assert restored["quantity"].tolist() == source["quantity"].tolist()
    assert restored["gross_per_driver_package_kg"].tolist() == pytest.approx(
        source["gross_per_driver_package_kg"].tolist()
    )
    assert restored["net_per_driver_package_kg"].tolist() == pytest.approx(
        source["net_per_driver_package_kg"].tolist(), rel=1e-8
    )
    assert restored["volume_per_driver_package_m3"].tolist() == pytest.approx(
        source["volume_per_driver_package_m3"].tolist()
    )
    assert all(restored["gross_per_driver_package_kg"] >= restored["net_per_driver_package_kg"])


@pytest.mark.parametrize(
    "profile",
    [
        MeasureProfile(False, False, False),
        MeasureProfile(False, False, True),
        MeasureProfile(False, True, False),
        MeasureProfile(False, True, True),
        MeasureProfile(True, False, False),
        MeasureProfile(True, False, True),
        MeasureProfile(True, True, False),
        MeasureProfile(True, True, True),
    ],
)
def test_physical_factor_round_trip_covers_every_measure_profile(
    profile: MeasureProfile,
) -> None:
    data: dict[str, list[float] | list[int]] = {"quantity": [3, 9]}
    if profile.gross:
        data["gross_per_driver_package_kg"] = [120.0, 70.0]
    if profile.net:
        data["net_per_driver_package_kg"] = [100.0, 65.0]
    if profile.volume:
        data["volume_per_driver_package_m3"] = [0.8, 0.3]
    source = pd.DataFrame(data)

    restored = _from_physical_factors(_to_physical_factors(source, profile), profile)

    assert list(restored.columns) == list(source.columns)
    assert restored["quantity"].tolist() == source["quantity"].tolist()
    for column in source.columns[1:]:
        assert restored[column].tolist() == pytest.approx(source[column].tolist())
    if profile.gross and profile.net:
        assert all(restored["gross_per_driver_package_kg"] >= restored["net_per_driver_package_kg"])


def test_statistical_gate_filters_higher_scoring_failure_before_model_selection(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    rows = tuple(_source_row(index) for index in range(15))
    request = ProfileRouteRequest(
        "gate-aware-request",
        "PACKAGE_CARTON",
        "BOX_LIKE",
        "direct_goods",
        MeasureProfile(True, True, False),
    )

    def candidate_run(*, spec: Any, fold: Any, seed: int, **_: object) -> object:
        candidate = spec.identifier
        score = {"empirical": 0.85, "gaussian_copula_default": 0.90}.get(candidate, 0.80)
        return SimpleNamespace(
            candidate=candidate,
            fold_index=fold.index,
            seed=seed,
            evaluation=SimpleNamespace(
                diagnostic=SimpleNamespace(score=1.0),
                quality=SimpleNamespace(score=score),
            ),
        )

    def assess(_runs: object, *, candidate: str, **_: object) -> dict[str, object]:
        accepted = candidate == "gaussian_copula_physical_factors"
        return {
            "candidate": candidate,
            "failures": [] if accepted else [{"gate": "paired_quality_lower_bound"}],
            "accepted": accepted,
        }

    def final_sample(*, spec: Any, **_: object) -> object:
        candidate = spec.identifier
        return SimpleNamespace(
            candidate=candidate,
            constraint="physical",
            fit=SimpleNamespace(to_dict=lambda: {}),
            sample=SimpleNamespace(to_dict=lambda: {}),
            proposal=SimpleNamespace(
                acceptance_yield=1.0,
                raw_proposals=4,
                rejected_proposals=0,
            ),
            validity=SimpleNamespace(valid=True, to_dict=lambda: {}),
            novelty=SimpleNamespace(novel_rows=4, rows=4, to_dict=lambda: {}),
            proposals=(),
        )

    monkeypatch.setattr(profile_sdv_module, "_candidate_run", candidate_run)
    monkeypatch.setattr(profile_sdv_module, "assess_statistical_candidate", assess)
    monkeypatch.setattr(profile_sdv_module, "_sample_final_candidate", final_sample)
    monkeypatch.setattr(profile_sdv_module, "_environment", lambda: {})
    monkeypatch.setattr(profile_sdv_module, "_rss_bytes", lambda: 0)
    monkeypatch.setattr(profile_sdv_module, "_peak_rss_bytes", lambda: 0)

    result = benchmark_and_sample_profile(
        rows=rows,
        request=request,
        requested_rows=4,
        sample_seed=13,
        quality_settings=QualityAuditSettings(robust_minimum_rows=30),
        routing_settings=_routing_settings(rows=10, templates=5),
        benchmark_settings=ProfileBenchmarkSettings(
            fold_count=3,
            seeds=(7,),
            candidate_identifiers=(
                "empirical",
                "gaussian_copula_default",
                "gaussian_copula_physical_factors",
            ),
            production_selectable_candidates=(
                "gaussian_copula_default",
                "gaussian_copula_physical_factors",
            ),
        ),
    )

    assert result.selection.selected_candidate == "gaussian_copula_physical_factors"
    assert result.candidate_acceptance["gaussian_copula_default"]["accepted"] is False
    assert result.candidate_acceptance["gaussian_copula_physical_factors"]["accepted"] is True


def test_domain_validity_and_projection_derive_totals_from_one_driver_quantity() -> None:
    data = pd.DataFrame(
        {
            "quantity": [5, 3],
            "gross_per_driver_package_kg": [12.0, 8.0],
            "net_per_driver_package_kg": [10.0, 7.0],
        }
    )
    profile = MeasureProfile(True, True, False)
    validity = _validity(data, profile)
    assert validity.valid
    proposals = _proposal_rows(data, profile)
    assert proposals[0].gross_total_kg == 60.0
    assert proposals[0].net_total_kg == 50.0
    assert proposals[0].volume_total_m3 is None

    invalid = data.copy()
    invalid.loc[0, "net_per_driver_package_kg"] = 13.0
    receipt = _validity(invalid, profile)
    assert not receipt.valid
    assert receipt.violation_counts == {"gross_below_net": 1}


@pytest.mark.skipif(importlib.util.find_spec("sdv") is None, reason="SDV runtime unavailable")
def test_real_profile_benchmark_is_grouped_constrained_receipted_and_atomic(
    tmp_path: Path,
) -> None:
    rows = tuple(
        _source_row(
            index,
            quantity=index % 9 + 2,
            gross_value=(index % 9 + 2) * (12.0 + index * 0.15),
            net_value=(index % 9 + 2) * (10.0 + index * 0.12),
        )
        for index in range(30)
    )
    request = ProfileRouteRequest(
        "smoke-request",
        "PACKAGE_CARTON",
        "BOX_LIKE",
        "direct_goods",
        MeasureProfile(True, True, False),
    )
    artifact_dir = tmp_path / "profile-run"
    result = benchmark_and_sample_profile(
        rows=rows,
        request=request,
        requested_rows=8,
        sample_seed=991,
        quality_settings=QualityAuditSettings(robust_minimum_rows=40),
        routing_settings=_routing_settings(rows=12, templates=8),
        benchmark_settings=ProfileBenchmarkSettings(
            fold_count=3,
            seeds=(101,),
            neural_minimum_rows=1_000,
            neural_minimum_templates=100,
            proposal_multiplier=8,
        ),
        artifact_dir=artifact_dir,
    )
    assert len(result.proposals) == 8
    assert result.final_validity.valid
    assert result.run_resources.elapsed_seconds > 0
    assert result.run_resources.peak_rss_after_bytes >= result.run_resources.peak_rss_before_bytes
    assert result.selection.selected_candidate != "empirical"
    assert "empirical" not in result.selection.selectable_candidates
    assert (
        result.contract["benchmark"]["empirical_policy"]
        == "required_paired_benchmark_not_production_selectable"
    )
    assert result.final_constraint in {
        "sdv_cag.Inequality(net_per_driver_package_kg<=gross_per_driver_package_kg)",
        "log_quantity_total_measures_density_and_gross_net_ratio_v1",
    }
    assert len(result.benchmark_runs) == 15  # 5 eligible models x 3 grouped folds.
    mixed_runs = tuple(
        run for run in result.benchmark_runs if run.candidate == "gaussian_copula_domain_mixed"
    )
    assert len(mixed_runs) == 3
    assert all(run.synthetic_sha256 for run in mixed_runs)
    mixed_contract = next(
        candidate
        for candidate in result.contract["benchmark"]["candidates"]
        if candidate["identifier"] == "gaussian_copula_domain_mixed"
    )
    assert mixed_contract["parameters"] == {
        "enforce_min_max_values": True,
        "enforce_rounding": True,
        "numerical_distributions": {
            "gross_per_driver_package_kg": "gamma",
            "net_per_driver_package_kg": "gamma",
            "quantity": "gaussian_kde",
        },
    }
    physical_contract = next(
        candidate
        for candidate in result.contract["benchmark"]["candidates"]
        if candidate["identifier"] == "gaussian_copula_physical_factors"
    )
    assert physical_contract["representation"] == "physical_factors"
    assert all(
        set(run.to_dict()) >= {"fit", "sample", "validity", "novelty", "constraint"}
        for run in result.benchmark_runs
    )
    assert all(
        run.constraint.startswith("sdv_cag.Inequality")
        for run in result.benchmark_runs
        if run.candidate not in {"empirical", "gaussian_copula_physical_factors"}
    )
    assert all(
        run.constraint == "log_quantity_total_measures_density_and_gross_net_ratio_v1"
        for run in result.benchmark_runs
        if run.candidate == "gaussian_copula_physical_factors"
    )
    assert all(
        proposal.gross_total_kg is not None
        and proposal.net_total_kg is not None
        and proposal.gross_total_kg >= proposal.net_total_kg
        and math.isclose(
            proposal.gross_total_kg,
            proposal.quantity * proposal.gross_per_driver_package_kg,
        )
        for proposal in result.proposals
    )
    manifest = json.loads((artifact_dir / "manifest.json").read_text())
    assert manifest["status"] == "complete"
    assert {row["path"] for row in manifest["files"]} == {
        "benchmark.json",
        "candidate-eligibility.json",
        "contract.json",
        "routing.json",
        "sample.json",
        "source-quality-audit.json",
    }

    # Immutable publication rejects a semantically different result at the same path.
    changed_rows = list(rows)
    changed_rows[0] = _source_row(0, gross_value=999.0, net_value=900.0)
    with pytest.raises(ProfileSdvError, match="must be new or empty"):
        benchmark_and_sample_profile(
            rows=changed_rows,
            request=request,
            requested_rows=8,
            sample_seed=991,
            quality_settings=QualityAuditSettings(robust_minimum_rows=40),
            routing_settings=_routing_settings(rows=12, templates=8),
            benchmark_settings=ProfileBenchmarkSettings(
                fold_count=3,
                seeds=(101,),
                neural_minimum_rows=1_000,
                neural_minimum_templates=100,
            ),
            artifact_dir=artifact_dir,
        )


@pytest.mark.skipif(importlib.util.find_spec("sdv") is None, reason="SDV runtime unavailable")
def test_real_quantity_only_profile_uses_explicit_single_column_quality_contract() -> None:
    rows = tuple(
        NumericSourceRow(
            row_id=f"quantity-row-{index}",
            document_id=f"quantity-doc-{index}",
            template_id=f"quantity-template-{index}",
            partition="train",
            exact_identity="PACKAGE_PACKAGE",
            semantic_family="GENERIC",
            package_role="generic_aggregate",
            # Deliberately leave integer gaps so a rounded Gaussian model can
            # prove novelty instead of exhausting a fully enumerated support.
            quantity=(index % 7) * 10 + 1,
        )
        for index in range(18)
    )
    result = benchmark_and_sample_profile(
        rows=rows,
        request=ProfileRouteRequest(
            "quantity-only",
            "PACKAGE_PACKAGE",
            "GENERIC",
            "generic_aggregate",
            MeasureProfile(False, False, False),
        ),
        requested_rows=6,
        sample_seed=808,
        quality_settings=QualityAuditSettings(robust_minimum_rows=20),
        routing_settings=_routing_settings(rows=12, templates=8),
        benchmark_settings=ProfileBenchmarkSettings(
            fold_count=3,
            seeds=(9,),
            neural_minimum_rows=1_000,
            neural_minimum_templates=100,
        ),
    )
    assert result.final_validity.valid
    assert all(
        run.evaluation.quality.properties.keys() == {"Column Shapes"}
        for run in result.benchmark_runs
    )
    assert all(proposal.gross_total_kg is None for proposal in result.proposals)


def test_one_call_raises_for_unroutable_profile_before_any_sdv_import() -> None:
    rows = tuple(_source_row(index) for index in range(4))
    request = ProfileRouteRequest(
        "unroutable",
        "PACKAGE_DRUM",
        "CYLINDRICAL",
        "outer_transport",
        MeasureProfile(True, True, False),
    )
    with pytest.raises(ProfileRouteError, match="no profile route"):
        benchmark_and_sample_profile(
            rows=rows,
            request=request,
            requested_rows=2,
            sample_seed=1,
            quality_settings=QualityAuditSettings(robust_minimum_rows=8),
            routing_settings=_routing_settings(),
            benchmark_settings=ProfileBenchmarkSettings(fold_count=2, seeds=(1,)),
        )


def test_artifact_target_must_be_a_new_or_empty_directory(tmp_path: Path) -> None:
    artifact_file = tmp_path / "not-a-directory"
    artifact_file.write_text("occupied", encoding="utf-8")
    with pytest.raises(ProfileSdvError, match="must be a directory"):
        benchmark_and_sample_profile(
            rows=tuple(_source_row(index) for index in range(4)),
            request=ProfileRouteRequest(
                "artifact-contract",
                "PACKAGE_CARTON",
                "BOX_LIKE",
                "direct_goods",
                MeasureProfile(True, True, False),
            ),
            requested_rows=2,
            sample_seed=1,
            quality_settings=QualityAuditSettings(robust_minimum_rows=8),
            routing_settings=_routing_settings(),
            benchmark_settings=ProfileBenchmarkSettings(fold_count=2, seeds=(1,)),
            artifact_dir=artifact_file,
        )
