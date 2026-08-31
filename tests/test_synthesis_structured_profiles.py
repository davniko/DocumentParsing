from __future__ import annotations

import importlib.util
from pathlib import Path
from statistics import fmean
from types import SimpleNamespace

import pytest

from document_ocr.synthesis.config import (
    SynthesisStructuredBaselineConfig,
    load_synthesis_structured_baseline_config,
)
from document_ocr.synthesis.profile_sdv import (
    MeasureProfile,
    ProfileNumericProposal,
    ProfileNumericRow,
)
from document_ocr.synthesis.sdv_evaluation import (
    EvaluationBundle,
    MetricDetail,
    MetricReport,
)
from document_ocr.synthesis.structured_profiles import (
    ProfileCandidateAssignmentError,
    StructuredProfileError,
    _assign_profile_candidate_indices,
    _cohort_per_container_bounds,
    _contextual_candidate_assessment,
    _one_sided_mean_lower_bound,
    _paired_property_receipts,
    _quality_acceptance,
    contextual_support_envelope,
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
CONFIG = PROJECT_ROOT / "configs/synthesis/mpci_bl_combined1157_structured_baseline50.yaml"


def _profile_row(
    row_id: str,
    *,
    identity: str,
    family: str,
    quantity: int,
    volume: float,
) -> ProfileNumericRow:
    return ProfileNumericRow(
        row_id=row_id,
        document_id=f"doc-{row_id}",
        template_id=f"template-{row_id}",
        exact_identity=identity,
        semantic_family=family,
        package_role="direct_goods",
        profile=MeasureProfile(gross=False, net=False, volume=True),
        quantity=quantity,
        gross_per_driver_package_kg=None,
        net_per_driver_package_kg=None,
        volume_per_driver_package_m3=volume / quantity,
    )


def _context(identity: str, family: str) -> SimpleNamespace:
    return SimpleNamespace(
        request=SimpleNamespace(
            exact_identity=identity,
            semantic_family=family,
            package_role="direct_goods",
            profile=MeasureProfile(gross=False, net=False, volume=True),
        )
    )


def _volume_proposal(quantity: int, volume: float) -> ProfileNumericProposal:
    return ProfileNumericProposal(
        quantity=quantity,
        gross_per_driver_package_kg=None,
        net_per_driver_package_kg=None,
        volume_per_driver_package_m3=volume / quantity,
        gross_total_kg=None,
        net_total_kg=None,
        volume_total_m3=volume,
    )


def test_contextual_support_rejects_role_pooled_vehicle_quantity() -> None:
    config = load_synthesis_structured_baseline_config(CONFIG)
    rows = tuple(
        _profile_row(
            str(index),
            identity="PACKAGE_VEHICLE",
            family="PACKAGE_VEHICLE",
            quantity=1,
            volume=volume,
        )
        for index, volume in enumerate((28.0, 40.0, 62.0), start=1)
    )
    prepared = SimpleNamespace(clean_rows_by_id={row.row_id: row for row in rows})

    envelope = contextual_support_envelope(
        _context("PACKAGE_VEHICLE", "PACKAGE_VEHICLE"),
        prepared=prepared,
        config=config,
    )

    assert envelope is not None
    assert envelope.tier == "exact_identity_role"
    assert envelope.bounds["quantity"] == (1.0, 1.0)
    reason, _distance = _contextual_candidate_assessment(_volume_proposal(2528, 46.0), envelope)
    assert reason == "outside_exact_identity_role_quantity_range"
    assert _contextual_candidate_assessment(_volume_proposal(1, 46.0), envelope)[0] is None


def test_contextual_support_uses_semantic_family_only_when_exact_is_sparse() -> None:
    config = load_synthesis_structured_baseline_config(CONFIG)
    rows = (
        _profile_row(
            "plastic",
            identity="PACKAGE_DRUM_PLASTIC",
            family="PACKAGE_DRUM",
            quantity=24,
            volume=25.0,
        ),
        *tuple(
            _profile_row(
                f"drum-{index}",
                identity="PACKAGE_DRUM",
                family="PACKAGE_DRUM",
                quantity=quantity,
                volume=volume,
            )
            for index, (quantity, volume) in enumerate(
                ((20, 20.0), (30, 28.0), (40, 35.0), (50, 45.0)), start=1
            )
        ),
    )
    prepared = SimpleNamespace(clean_rows_by_id={row.row_id: row for row in rows})

    envelope = contextual_support_envelope(
        _context("PACKAGE_DRUM_PLASTIC", "PACKAGE_DRUM"),
        prepared=prepared,
        config=config,
    )

    assert envelope is not None
    assert envelope.tier == "semantic_family_role"
    assert len(envelope.row_ids) == 5


def test_contextual_support_does_not_pool_unregistered_printed_surfaces() -> None:
    config = load_synthesis_structured_baseline_config(CONFIG)
    rows = tuple(
        _profile_row(
            surface.lower(),
            identity=f"SURFACE:{surface}",
            family="UNREGISTERED_PRINTED_PACKAGE",
            quantity=quantity,
            volume=volume,
        )
        for surface, quantity, volume in (
            ("CARBOUY", 108, 12.0),
            ("BALES", 905, 48.0),
            ("ITEMS", 34, 18.0),
            ("COILS", 12, 30.0),
            ("SKIDS", 8, 22.0),
        )
    )
    prepared = SimpleNamespace(clean_rows_by_id={row.row_id: row for row in rows})

    envelope = contextual_support_envelope(
        _context("SURFACE:CARBOUY", "UNREGISTERED_PRINTED_PACKAGE"),
        prepared=prepared,
        config=config,
    )

    assert envelope is None


def test_source_equal_profile_rows_share_one_candidate_without_losing_cardinality() -> None:
    request_ids = tuple(f"request-{index}" for index in range(18))

    assigned = _assign_profile_candidate_indices(
        options={request_id: tuple(range(11)) for request_id in request_ids},
        equivalence_ids={request_id: "same-source-row" for request_id in request_ids},
        cohort_id="cohort-repeated-container-lines",
        sampled_rows=144,
        unique_sampled_rows=142,
    )

    assert set(assigned) == set(request_ids)
    assert len(set(assigned.values())) == 1


def test_independent_profile_rows_still_require_distinct_candidates() -> None:
    with pytest.raises(
        ProfileCandidateAssignmentError,
        match="independent source numeric equivalence classes",
    ) as raised:
        _assign_profile_candidate_indices(
            options={"request-a": (7,), "request-b": (7,)},
            equivalence_ids={"request-a": "class-a", "request-b": "class-b"},
            cohort_id="cohort-independent",
            sampled_rows=2,
            unique_sampled_rows=1,
        )

    assert raised.value.failed_equivalence_id == "class-b"
    assert raised.value.class_option_counts == {"class-a": 1, "class-b": 1}


def test_source_equal_profile_rows_require_a_candidate_valid_for_every_member() -> None:
    assigned = _assign_profile_candidate_indices(
        options={"request-a": (1, 2), "request-b": (2, 3)},
        equivalence_ids={"request-a": "same", "request-b": "same"},
        cohort_id="cohort-intersection",
        sampled_rows=3,
        unique_sampled_rows=3,
    )

    assert assigned == {"request-a": 2, "request-b": 2}


def test_empirical_measure_bounds_do_not_mix_containerization_classes() -> None:
    prepared = SimpleNamespace(
        contexts=(
            SimpleNamespace(
                request_id="containerized",
                observation=SimpleNamespace(
                    features=SimpleNamespace(group_allocation_container_count=0)
                ),
            ),
            SimpleNamespace(
                request_id="breakbulk",
                observation=SimpleNamespace(
                    features=SimpleNamespace(group_allocation_container_count=0)
                ),
            ),
        ),
        clean_rows_by_id={
            "containerized": SimpleNamespace(
                document_id="doc-containerized",
                quantity=10,
                gross_per_driver_package_kg=10.0,
                net_per_driver_package_kg=None,
                volume_per_driver_package_m3=None,
            ),
            "breakbulk": SimpleNamespace(
                document_id="doc-breakbulk",
                quantity=10,
                gross_per_driver_package_kg=100.0,
                net_per_driver_package_kg=None,
                volume_per_driver_package_m3=None,
            ),
        },
    )
    decision = SimpleNamespace(selected_row_ids=("containerized", "breakbulk"))
    targets = {
        "doc-containerized": {"documentPatch": {"containers": [{}, {}]}},
        "doc-breakbulk": {"documentPatch": {"containers": []}},
    }

    assert _cohort_per_container_bounds(
        prepared,
        decision,
        source_targets=targets,
        target_is_containerized=True,
    ) == {"gross": (50.0, 50.0)}
    assert _cohort_per_container_bounds(
        prepared,
        decision,
        source_targets=targets,
        target_is_containerized=False,
    ) == {"gross": (1000.0, 1000.0)}


def _quality_report(
    score: float,
    *,
    quantity_score: float = 0.8,
    gross_score: float = 0.8,
) -> MetricReport:
    property_score = fmean((quantity_score, gross_score))
    return MetricReport(
        report_name="quality",
        score=score,
        properties={"Column Shapes": property_score},
        details=(
            MetricDetail(
                "Column Shapes",
                {"Column": "quantity", "Metric": "KSComplement", "Score": quantity_score},
            ),
            MetricDetail(
                "Column Shapes",
                {
                    "Column": "gross_per_driver_package_kg",
                    "Metric": "KSComplement",
                    "Score": gross_score,
                },
            ),
        ),
    )


def _evaluation(report: MetricReport) -> EvaluationBundle:
    return EvaluationBundle(
        diagnostic=MetricReport(
            report_name="diagnostic",
            score=1.0,
            properties={"Data Validity": 1.0},
            details=(),
        ),
        quality=report,
    )


def _acceptance_result(
    *,
    config: SynthesisStructuredBaselineConfig,
    empirical_scores: tuple[float, ...] | None = None,
    selected_scores: tuple[float, ...] | None = None,
    empirical_detail_scores: tuple[float, float] = (0.8, 0.8),
    selected_detail_scores: tuple[float, float] = (0.8, 0.8),
    novelty: float = 1.0,
    acceptance_yield: float = 1.0,
) -> SimpleNamespace:
    folds = config.modeling.grouped_folds
    empirical_by_fold = empirical_scores or (0.8,) * folds
    selected_by_fold = selected_scores or (0.8,) * folds
    assert len(empirical_by_fold) == len(selected_by_fold) == folds
    benchmark_runs = []
    for fold_index, (empirical_score, selected_score) in enumerate(
        zip(empirical_by_fold, selected_by_fold, strict=True)
    ):
        for seed in config.modeling.seeds:
            for candidate, report in (
                (
                    "empirical",
                    _quality_report(
                        empirical_score,
                        quantity_score=empirical_detail_scores[0],
                        gross_score=empirical_detail_scores[1],
                    ),
                ),
                (
                    "gaussian_copula_default",
                    _quality_report(
                        selected_score,
                        quantity_score=selected_detail_scores[0],
                        gross_score=selected_detail_scores[1],
                    ),
                ),
            ):
                benchmark_runs.append(
                    SimpleNamespace(
                        candidate=candidate,
                        fold_index=fold_index,
                        seed=seed,
                        evaluation=_evaluation(report),
                    )
                )
    rows = 100
    return SimpleNamespace(
        selection=SimpleNamespace(selected_candidate="gaussian_copula_default"),
        benchmark_runs=tuple(benchmark_runs),
        final_novelty=SimpleNamespace(novel_rows=int(novelty * rows), rows=rows),
        final_proposal=SimpleNamespace(
            acceptance_yield=acceptance_yield,
            raw_proposals=rows,
            rejected_proposals=rows - int(acceptance_yield * rows),
        ),
    )


@pytest.mark.skipif(importlib.util.find_spec("scipy") is None, reason="SciPy runtime unavailable")
def test_one_sided_mean_lower_bound_uses_stated_confidence_not_two_sided_quantile() -> None:
    differences = (
        0.18181818181818188,
        -0.12121212121212133,
        -0.09090909090909094,
        -0.030303030303030165,
        -0.030303030303030276,
        -0.06060606060606055,
    )

    lower_bound = _one_sided_mean_lower_bound(differences, confidence_level=0.95)

    assert lower_bound == pytest.approx(-0.1136227499)


def test_paired_properties_allow_fold_specific_joint_applicability() -> None:
    receipts = _paired_property_receipts(
        (
            (
                {"Column Shapes": 0.8, "Column Pair Trends": 0.7},
                {"Column Shapes": 0.9, "Column Pair Trends": 0.75},
            ),
            ({"Column Shapes": 0.7}, {"Column Shapes": 0.72}),
        ),
    )

    by_name = {row["property"]: row for row in receipts}
    assert by_name["Column Shapes"]["applicable_paired_runs"] == 2
    assert by_name["Column Pair Trends"]["applicable_paired_runs"] == 1
    assert all(row["total_paired_runs"] == 2 for row in receipts)


def test_paired_properties_reject_candidate_disagreement_within_fold() -> None:
    with pytest.raises(StructuredProfileError, match="within a paired run"):
        _paired_property_receipts(
            (({"Column Shapes": 0.8}, {"Column Shapes": 0.8, "Column Pair Trends": 0.7}),),
        )


@pytest.mark.skipif(importlib.util.find_spec("scipy") is None, reason="SciPy runtime unavailable")
def test_quality_lcb_uses_folds_as_independent_units_after_averaging_seeds() -> None:
    config = load_synthesis_structured_baseline_config(CONFIG)
    fold_differences = (-0.08, -0.02, 0.01, 0.03, 0.06)
    result = _acceptance_result(
        config=config,
        empirical_scores=(0.8,) * config.modeling.grouped_folds,
        selected_scores=tuple(0.8 + difference for difference in fold_differences),
    )

    receipt = _quality_acceptance(result, config=config)

    assert receipt["paired_runs"] == 25
    assert receipt["independent_folds"] == 5
    assert receipt["independent_unit"] == "validation_fold"
    assert [row["seed_replicates"] for row in receipt["folds"]] == [5] * 5
    assert receipt["paired_difference_lower_bound"] == pytest.approx(
        _one_sided_mean_lower_bound(fold_differences, confidence_level=0.95)
    )


@pytest.mark.skipif(importlib.util.find_spec("scipy") is None, reason="SciPy runtime unavailable")
def test_detail_deficit_boundary_is_accepted_and_receipted() -> None:
    config = load_synthesis_structured_baseline_config(CONFIG)
    result = _acceptance_result(
        config=config,
        empirical_detail_scores=(0.9, 0.7),
        selected_detail_scores=(0.8, 0.8),
    )

    receipt = _quality_acceptance(result, config=config)

    detail = next(row for row in receipt["details"] if row["columns"] == ["quantity"])
    assert detail["difference"] == pytest.approx(-0.10)
    assert detail["maximum_allowed_deficit"] == 0.10
    assert detail["passed"] is True


@pytest.mark.skipif(importlib.util.find_spec("scipy") is None, reason="SciPy runtime unavailable")
def test_detail_deficit_beyond_boundary_fails_with_detailed_receipt() -> None:
    config = load_synthesis_structured_baseline_config(CONFIG)
    result = _acceptance_result(
        config=config,
        empirical_detail_scores=(0.9, 0.7),
        selected_detail_scores=(0.799, 0.801),
    )

    with pytest.raises(StructuredProfileError) as raised:
        _quality_acceptance(result, config=config)

    message = str(raised.value)
    assert "detail_quality_deficit" in message
    assert "Column Shapes|KSComplement|quantity" in message
    assert "maximum_allowed_deficit': 0.1" in message
    assert "detail_receipts=" in message


@pytest.mark.skipif(importlib.util.find_spec("scipy") is None, reason="SciPy runtime unavailable")
def test_acceptance_failure_consolidates_every_failed_gate_and_receipt() -> None:
    config = load_synthesis_structured_baseline_config(CONFIG)
    result = _acceptance_result(
        config=config,
        selected_scores=(0.6,) * config.modeling.grouped_folds,
        selected_detail_scores=(0.6, 0.6),
        novelty=0.5,
        acceptance_yield=0.1,
    )

    with pytest.raises(StructuredProfileError) as raised:
        _quality_acceptance(result, config=config)

    message = str(raised.value)
    for gate in (
        "mean_quality_difference",
        "paired_quality_lower_bound",
        "property_quality_deficit",
        "detail_quality_deficit",
        "final_novelty_fraction",
        "final_acceptance_yield",
    ):
        assert gate in message
    assert "fold_receipts=" in message
    assert "gate_receipts=" in message
    assert "property_receipts=" in message
    assert "detail_receipts=" in message


def test_detail_deficit_configuration_defaults_to_and_pins_ten_percent() -> None:
    loaded = load_synthesis_structured_baseline_config(CONFIG)
    assert loaded.modeling.maximum_detail_quality_deficit_vs_empirical == 0.10

    value = loaded.model_dump(mode="python")
    modeling = value["modeling"]
    assert isinstance(modeling, dict)
    modeling.pop("maximum_detail_quality_deficit_vs_empirical")
    defaulted = SynthesisStructuredBaselineConfig.model_validate(value, strict=True)

    assert defaulted.modeling.maximum_detail_quality_deficit_vs_empirical == 0.10
