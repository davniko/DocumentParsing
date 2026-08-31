import csv
import io
from copy import deepcopy
from typing import cast

import pytest

from document_ocr.synthesis.run_safety import StagedArtifactRun
from document_ocr.synthesis.structured_reporting import (
    StructuredReportingError,
    publish_structured_baseline_report,
)


def _resource(seconds: float) -> dict[str, float | int]:
    return {
        "elapsed_seconds": seconds,
        "rss_before_bytes": 100,
        "rss_after_bytes": 120,
        "rss_delta_bytes": 20,
        "peak_rss_before_bytes": 200,
        "peak_rss_after_bytes": 240,
        "peak_rss_increase_bytes": 40,
    }


def _benchmark(candidate: str, fold: int, score: float) -> dict[str, object]:
    return {
        "candidate": candidate,
        "fold_index": fold,
        "seed": 7,
        "train_rows": 10,
        "validation_rows": 5,
        "train_templates": 8,
        "validation_templates": 4,
        "constraint": "positive",
        "proposal": {
            "requested_rows": 5,
            "raw_proposals": 8,
            "accepted_proposals_before_truncation": 6,
            "rejected_proposals": 2,
            "output_rows": 5,
            "acceptance_yield": 0.75,
            "bounded_maximum_raw_proposals": 40,
        },
        "validity": {
            "rows": 5,
            "valid_rows": 5,
            "invalid_rows": 0,
            "valid_fraction": 1.0,
            "violation_counts": {},
        },
        "novelty": {
            "rows": 5,
            "novel_rows": 5,
            "exact_source_matches": 0,
            "novel_fraction": 1.0,
        },
        "evaluation": {
            "diagnostic": {
                "report": "diagnostic",
                "score": 1.0,
                "properties": {"Data Validity": 1.0},
                "details": [],
            },
            "quality": {
                "report": "quality",
                "score": score,
                "properties": {"Column Shapes": score},
                "details": [],
            },
        },
        "synthetic_sha256": "a" * 64,
    }


def _profile_result() -> dict[str, object]:
    benchmark_rows = [
        _benchmark(candidate, fold, score)
        for fold in (0, 1)
        for candidate, score in (("empirical", 0.80), ("gaussian_copula_default", 0.82))
    ]
    runtime_rows = [
        {
            "candidate": row["candidate"],
            "fold_index": row["fold_index"],
            "seed": row["seed"],
            "fit": _resource(0.2),
            "sample": _resource(0.1),
        }
        for row in benchmark_rows
    ]
    return {
        "cohort_id": "cohort_1",
        "request_ids": ["request_1"],
        "sample_seed": 11,
        "quality_acceptance": {
            "selected_candidate": "gaussian_copula_default",
            "paired_runs": 2,
            "mean_quality_difference_vs_empirical": 0.02,
            "paired_difference_lower_bound": 0.01,
            "properties": [],
            "final_novelty_fraction": 1.0,
            "final_acceptance_yield": 0.75,
            "accepted": True,
        },
        "run": {
            "contract": {"request": {"profile": "g1n1v0"}},
            "audit": {
                "accepted_rows": 15,
                "quarantined_rows": 1,
            },
            "routing": {
                "decisions": [
                    {
                        "selected_tier": "semantic_family_role",
                        "selected_rows": 15,
                    }
                ]
            },
            "candidate_eligibility": [
                {
                    "candidate": "empirical",
                    "rows": 15,
                    "templates": 12,
                    "minimum_rows": 2,
                    "minimum_templates": 2,
                    "eligible": True,
                    "reason": None,
                },
                {
                    "candidate": "gaussian_copula_default",
                    "rows": 15,
                    "templates": 12,
                    "minimum_rows": 2,
                    "minimum_templates": 2,
                    "eligible": True,
                    "reason": None,
                },
            ],
            "benchmark_runs": benchmark_rows,
            "selection": {"selected_candidate": "gaussian_copula_default"},
            "final_proposal": {"raw_proposals": 8, "rejected_proposals": 2},
            "final_novelty": {"novel_rows": 5},
        },
        "runtime": {
            "final_fit": _resource(0.5),
            "final_sample": _resource(0.2),
            "run_resources": _resource(2.0),
            "benchmark_runs": runtime_rows,
        },
    }


def _proposal() -> dict[str, object]:
    return {
        "proposal_id": "proposal_1",
        "synthetic_position": 0,
        "base_document_id": "document_1",
        "request_id": "request_1",
        "cohort_id": "cohort_1",
        "group_id": "group_1",
        "package_id": "package_1",
        "package_count_in_group": 2,
        "package_identity": "PACKAGE_CARTON",
        "package_role": "inner",
        "route_tier": "semantic_family_role",
        "contextual_support_tier": "exact_identity_role",
        "contextual_support_rows": 12,
        "contextual_support_templates": 8,
        "contextual_support_sha256": "a" * 64,
        "contextual_distance": 0.2,
        "contextual_maximum_distance": 0.5,
        "selected_candidate": "gaussian_copula_default",
        "source_quantity": 10,
        "generated_quantity": 12,
        "source_group_quantities": {"package_1": 10, "package_2": 5},
        "generated_group_quantities": {"package_1": 12, "package_2": 6},
        "source_gross_weight_value": 1000.0,
        "generated_gross_weight_value": 1200.0,
        "gross_weight_unit": "kilogram",
        "source_net_weight_value": 900.0,
        "generated_net_weight_value": 1080.0,
        "net_weight_unit": "kilogram",
        "source_volume_value": None,
        "generated_volume_value": None,
        "volume_unit": None,
        "sampled_features": {"quantity": 12},
        "raw_proposals": 8,
        "rejected_proposals": 2,
        "acceptance_yield": 0.75,
        "exact_train_row_copy": False,
    }


def _inputs() -> dict[str, object]:
    return {
        "run_id": "report-test",
        "selected_rows": [
            {
                "position": 0,
                "document_id": "document_1",
                "template_id": "template_1",
                "carrier_family": "carrier_1",
                "strata": {
                    "document_type": "BILL_OF_LADING",
                    "source_corpus": "corpus_1",
                    "page_bucket": "1",
                    "container_bucket": "1",
                },
                "contexts": ["dangerous_goods"],
                "cargo_profile_routes": [
                    {
                        "request_id": "request_1",
                        "route_tier": "semantic_family_role",
                    }
                ],
            }
        ],
        "proposal_rows": [_proposal()],
        "targets": [{"syntheticDocumentId": "synthetic_1"}],
        "change_rows": [
            [
                {
                    "target_path": "documentPatch.issueDate",
                    "family": "document_date",
                    "method": "joint_shift",
                    "coupling_group": "dates",
                    "old_value": "2024-01-02",
                    "new_value": "2024-02-03",
                },
                {
                    "target_path": "documentPatch.cargoPackages[0].quantity",
                    "family": "cargo_quantity",
                    "method": "profile_sdv",
                    "coupling_group": "group_1",
                    "old_value": 10,
                    "new_value": 12,
                },
            ]
        ],
        "distribution": {
            "selected_documents": 1,
            "contexts": {"dangerous_goods": 1},
            "pending_metadata_only_package_facts": 1,
            "transport_capacity": {
                "validated_documents": 1,
                "containerized_documents": 1,
                "non_containerized_documents": 0,
                "gross_payload_utilization": {"count": 1, "maximum": 0.5},
                "volume_utilization": {"count": 0},
            },
        },
        "capacity_rows": [
            {
                "synthetic_document_id": "synthetic_1",
                "base_document_id": "document_1",
                "policy": "source_type_aware_maersk_upper_bounds_v1",
                "container_count": 1,
                "equipment_families": ["twenty_standard"],
                "payload_capacity_kg": 28300.0,
                "volume_capacity_m3": 33.2,
                "gross_weight_kg": 14150.0,
                "net_weight_kg": None,
                "volume_m3": None,
                "gross_payload_utilization": 0.5,
                "net_payload_utilization": None,
                "volume_utilization": None,
                "violations": [],
                "valid": True,
            }
        ],
        "profile_results": [_profile_result()],
        "modeling_audit": {
            "train_document_count": 50,
            "cargo_group_numeric_rows": 80,
            "cargo_package_numeric_rows": 100,
            "package_hierarchy_rows": 120,
            "metadata_only_package_rows": 5,
        },
    }


def test_publish_structured_report_emits_complete_profile_tables_and_plots(tmp_path) -> None:
    stage = StagedArtifactRun(
        output_parent=tmp_path,
        run_name="report-test",
        transaction_sha256="a" * 64,
    )
    publish_structured_baseline_report(stage=stage, **_inputs())

    package_rows = list(
        csv.DictReader(io.StringIO((stage.stage_root / "data/package-quantities.csv").read_text()))
    )
    assert len(package_rows) == 2
    assert {row["package_id"] for row in package_rows} == {"package_1", "package_2"}
    assert "driver_package_identity" in package_rows[0]
    assert "driver_package_role" in package_rows[0]
    assert "package_identity" not in package_rows[0]
    assert "package_role" not in package_rows[0]
    score_rows = list(
        csv.DictReader(
            io.StringIO((stage.stage_root / "data/sdv-benchmark-scorecard.csv").read_text())
        )
    )
    assert len(score_rows) == 4
    assert {row["fit_seconds"] for row in score_rows} == {"0.2"}
    report = (stage.stage_root / "REPORT.md").read_text()
    assert "Package categories" in report
    assert "not generated or statistically modeled" in report
    assert "pending later realization" in report
    assert len(list((stage.stage_root / "plots").glob("*.png"))) == 13
    assert (stage.stage_root / "data/transport-capacity.csv").is_file()


def test_publish_structured_report_rejects_changed_package_topology(tmp_path) -> None:
    inputs = _inputs()
    proposal = deepcopy(cast(list[dict[str, object]], inputs["proposal_rows"])[0])
    proposal["generated_group_quantities"] = {"package_1": 12}
    inputs["proposal_rows"] = [proposal]
    stage = StagedArtifactRun(
        output_parent=tmp_path,
        run_name="report-test",
        transaction_sha256="b" * 64,
    )

    with pytest.raises(StructuredReportingError, match="package topology"):
        publish_structured_baseline_report(stage=stage, **inputs)
