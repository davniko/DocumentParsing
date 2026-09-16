from __future__ import annotations

import json
from pathlib import Path

import pytest

from document_ocr.synthesis.template_compiler.coherence import CoherenceReviewRequired
from document_ocr.synthesis.template_compiler.pipeline import load_config
from document_ocr.synthesis.template_compiler.readiness import (
    _full_corpus_config,
    _resolve_checkpoint_roots,
    _structural_coverage,
    _validate_expected_review_checkpoint,
    structural_variant_tags,
)

_PROJECT_ROOT = Path(__file__).resolve().parents[1]
_CONFIG = _PROJECT_ROOT / "configs/synthesis/mpci_bl_template_compilation200_v1_luna.yaml"


def _feature(**updates: object) -> dict[str, object]:
    row: dict[str, object] = {
        "page_count": 1,
        "goods_group_count": 1,
        "package_fact_count": 1,
        "container_count": 1,
        "allocation_row_count": 1,
        "dangerous_goods_count": 0,
        "temperature_setting_count": 0,
        "party_with_contact_details_count": 0,
        "document_type": "bill_of_lading",
        "carrier_family": "TEST",
        "template_proxy_id": "template_test",
    }
    row.update(updates)
    return row


def test_structural_variant_tags_cover_known_provider_wasting_shapes() -> None:
    raw = (
        "--- PAGE 1 ---\n"
        "Quantity/Number of Packages\n"
        "Description of Goods\n"
        "3495\n"
        "UN NO : UN1993\n"
        "CLASS : 3\n"
        "Gross Cargo Weight\n"
        "KGS\n"
        "PACKAGE 1-7\n"
    )

    assert set(
        structural_variant_tags(
            raw=raw,
            feature=_feature(
                page_count=2,
                goods_group_count=2,
                package_fact_count=2,
                container_count=2,
                allocation_row_count=2,
                dangerous_goods_count=1,
                temperature_setting_count=1,
                party_with_contact_details_count=1,
            ),
            source_target={
                "documentPatch": {
                    "cargoPackages": [{"quantity": 7}],
                    "cargoAllocationGroups": [
                        {
                            "coverage": "single_package_level",
                            "allocations": [{}, {}],
                        }
                    ],
                }
            },
        )
    ) == {
        "columnar_package_quantity",
        "allocation_coverage:single_package_level",
        "allocation_row_magnitude:2-9",
        "dangerous_goods",
        "dangerous_goods_labeled_un_variant",
        "dangerous_goods_repeated_un_prefix",
        "dangerous_goods_un_surface",
        "multi_allocation",
        "multi_cargo",
        "multi_container",
        "multi_package",
        "multi_page",
        "numeric_range_surface",
        "party_contacts",
        "standalone_dangerous_goods_class",
        "standalone_measurement_unit",
        "structured_quantity_range_surface",
        "temperature_control",
    }


def test_columnar_package_audit_does_not_cross_a_blank_block_boundary() -> None:
    raw = "--- PAGE 1 ---\nNUMBER OF PACKAGES\n\n3495\n"

    assert "columnar_package_quantity" not in structural_variant_tags(
        raw=raw, feature=_feature(), source_target={"documentPatch": {}}
    )


def test_structural_coverage_reports_missing_tags_and_exact_covering_prefix() -> None:
    sources = (
        {
            "documentId": "doc_a",
            "joinedRawText": "--- PAGE 1 ---\nPACKAGE 1-7\n",
            "target": {"documentPatch": {"cargoPackages": [{"quantity": 7}]}},
        },
        {
            "documentId": "doc_b",
            "joinedRawText": "--- PAGE 1 ---\nCLASS : 3\n",
            "target": {"documentPatch": {}},
        },
    )
    features = (
        {"document_id": "doc_a", **_feature(template_proxy_id="template_a")},
        {
            "document_id": "doc_b",
            **_feature(
                dangerous_goods_count=1,
                carrier_family="SECOND",
                template_proxy_id="template_b",
            ),
        },
    )

    partial = _structural_coverage(
        source_rows=sources,
        feature_rows=features,
        eligible_document_ids=("doc_a", "doc_b"),
        selected_document_ids=("doc_a",),
    )
    assert partial["selectedCoversCorpusStructuralTags"] is False
    assert partial["missingStructuralTags"] == (
        "dangerous_goods",
        "standalone_dangerous_goods_class",
    )

    complete = _structural_coverage(
        source_rows=sources,
        feature_rows=features,
        eligible_document_ids=("doc_a", "doc_b"),
        selected_document_ids=("doc_a", "doc_b"),
    )
    assert complete["selectedCoversCorpusStructuralTags"] is True
    assert complete["minimumSelectedPrefixForFullStructuralCoverage"] == 2
    assert complete["hardCanaryPrefixDocumentIds"] == ("doc_a", "doc_b")


def test_full_corpus_audit_config_is_provider_locked_and_ignores_probe_exclusions() -> None:
    config = load_config(_CONFIG)

    audit = _full_corpus_config(config=config, eligible_document_ids=("doc_a", "doc_b"))

    assert audit.workflow.documents == 2
    assert audit.workflow.provider_launch_authorized is False
    assert audit.pinned_document_ids == ("doc_a", "doc_b")
    assert audit.excluded_document_ids == ()
    assert audit.resume_from is None


def test_checkpoint_root_without_current_case_checkpoints_is_rejected(tmp_path: Path) -> None:
    root = tmp_path / "empty-run"
    (root / "cases").mkdir(parents=True)

    with pytest.raises(ValueError, match="contains no case checkpoints"):
        _resolve_checkpoint_roots(project_root=tmp_path, checkpoint_roots=(root,))


def test_deliberate_coherence_review_is_a_valid_fail_closed_checkpoint(tmp_path: Path) -> None:
    result_path = tmp_path / "result.json"
    diagnostic = "coherence_constraint_0123456789abcdef: unresolved source relationship"
    result_path.write_text(
        json.dumps(
            {
                "document_id": "doc_review",
                "status": "review_required",
                "rejection_reasons": [f"semantic coherence requires review: {diagnostic}"],
                "template_sha256": None,
                "compiler_stages": [],
                "critic_stages": [],
                "risk_candidates": [],
                "elapsed_seconds": 0.0,
            }
        ),
        encoding="utf-8",
    )

    _validate_expected_review_checkpoint(
        result_path=result_path,
        document_id="doc_review",
        error=CoherenceReviewRequired(diagnostic),
    )


def test_unrecorded_coherence_review_is_rejected(tmp_path: Path) -> None:
    result_path = tmp_path / "result.json"
    result_path.write_text(
        json.dumps(
            {
                "document_id": "doc_review",
                "status": "review_required",
                "rejection_reasons": ["different review reason"],
                "template_sha256": None,
                "compiler_stages": [],
                "critic_stages": [],
                "risk_candidates": [],
                "elapsed_seconds": 0.0,
            }
        ),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="not recorded"):
        _validate_expected_review_checkpoint(
            result_path=result_path,
            document_id="doc_review",
            error=CoherenceReviewRequired("expected diagnostic"),
        )
