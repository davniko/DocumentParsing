from __future__ import annotations

import json
from pathlib import Path
from typing import cast

import pytest
from pydantic import ValidationError

from document_ocr.synthesis.template_compiler.review_resolution import (
    TemplateReviewResolutionConfig,
    _bar_chart,
    _evidence_lines,
    load_review_resolution_config,
)


def _decision(document_id: str = "doc_review") -> dict[str, object]:
    return {
        "document_id": document_id,
        "resolution": "exclude",
        "issue_class": "incomplete_container_topology",
        "expected_review_reasons": (
            "source template integrity requires review: "
            "explicit_carrier_receipt_container_count_differs_from_labeled_containers:1_vs_0",
        ),
        "source_line_numbers": (1, 2),
        "rationale": "The declared container has no representable identity.",
        "expected_printed_container_counts": (1,),
        "expected_labeled_container_count": 0,
        "expected_observed_container_identity_count": 0,
    }


def _config() -> dict[str, object]:
    return {
        "schema_version": 1,
        "task": "bill_of_lading_template_manual_review_resolution_v1",
        "run_name": "manual-resolution-v1",
        "output_dir": "artifacts/manual-resolution",
        "source_run": {
            "path": "artifacts/source-run",
            "commit_sha256": "a" * 64,
            "transaction_sha256": "b" * 64,
        },
        "review_date": "2026-09-17",
        "review_method": "full_queue_source_target_and_checkpoint_review",
        "expected_source_documents": 3,
        "expected_source_certified": 1,
        "expected_source_rejected": 1,
        "expected_source_review_required": 1,
        "expected_usable_templates": 1,
        "expected_excluded_documents": 2,
        "require_zero_unresolved_reviews": True,
        "decisions": (_decision(),),
    }


def test_resolution_contract_requires_exact_closed_partition() -> None:
    config = TemplateReviewResolutionConfig.model_validate_json(json.dumps(_config()))

    assert config.expected_usable_templates == 1
    assert config.decisions[0].resolution == "exclude"


def test_resolution_contract_rejects_duplicate_manual_decisions() -> None:
    payload = _config()
    decision = _decision()
    payload["expected_source_documents"] = 4
    payload["expected_source_review_required"] = 2
    payload["expected_excluded_documents"] = 3
    payload["decisions"] = (decision, decision)

    with pytest.raises(ValidationError, match="uniquely cover"):
        TemplateReviewResolutionConfig.model_validate_json(json.dumps(payload))


def test_resolution_contract_rejects_missing_issue_specific_evidence() -> None:
    payload = _config()
    decisions = cast(tuple[dict[str, object], ...], payload["decisions"])
    decisions[0].pop("expected_labeled_container_count")

    with pytest.raises(ValidationError, match="container evidence fields"):
        TemplateReviewResolutionConfig.model_validate_json(json.dumps(payload))


def test_evidence_lines_are_exact_and_hashed() -> None:
    rows = _evidence_lines("alpha\nbeta\ngamma\n", (1, 3))

    assert [row["text"] for row in rows] == ["alpha", "gamma"]
    assert all(len(cast(str, row["sha256"])) == 64 for row in rows)


def test_bar_chart_is_a_png() -> None:
    payload = _bar_chart(title="Review outcomes", labels=("usable", "excluded"), values=(9, 1))

    assert payload.startswith(b"\x89PNG\r\n\x1a\n")


def test_production_manual_resolution_config_covers_all_review_cases() -> None:
    root = Path(__file__).resolve().parents[1]
    config = load_review_resolution_config(
        root / "configs/synthesis/production/"
        "mpci_bl_template_compilation_remaining710_a_manual_resolution_v1.yaml"
    )

    assert len(config.decisions) == 22
    assert sum(row.issue_class == "incomplete_container_topology" for row in config.decisions) == 21
    assert sum(row.issue_class == "compound_package_dependency" for row in config.decisions) == 1


@pytest.mark.parametrize(
    ("filename", "source_documents", "certified", "rejected", "review_required"),
    (
        (
            "mpci_bl_template_compilation_b_ordinals181_520_recovery211_manual_resolution_v1.yaml",
            211,
            193,
            13,
            5,
        ),
        (
            "mpci_bl_template_compilation_b_ordinals181_520_missing129_manual_resolution_v1.yaml",
            129,
            113,
            9,
            7,
        ),
        (
            "mpci_bl_template_compilation_b_ordinals521_710_manual_resolution_v1.yaml",
            190,
            166,
            11,
            13,
        ),
    ),
)
def test_half_b_manual_resolutions_close_every_review_case(
    filename: str,
    source_documents: int,
    certified: int,
    rejected: int,
    review_required: int,
) -> None:
    root = Path(__file__).resolve().parents[1]
    config = load_review_resolution_config(root / "configs/synthesis/production" / filename)

    assert config.expected_source_documents == source_documents
    assert config.expected_source_certified == certified
    assert config.expected_source_rejected == rejected
    assert config.expected_source_review_required == review_required
    assert config.expected_usable_templates == certified
    assert config.expected_excluded_documents == rejected + review_required
    assert len(config.decisions) == review_required
    assert all(row.resolution == "exclude" for row in config.decisions)
    assert all(row.issue_class == "incomplete_container_topology" for row in config.decisions)
    assert config.require_zero_unresolved_reviews is True
