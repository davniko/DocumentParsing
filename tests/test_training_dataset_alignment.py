from __future__ import annotations

import json
from pathlib import Path

import pytest

from document_ocr.training.dataset_alignment import (
    DatasetAlignmentError,
    PolicyCorrection,
    _align_relation_target,
    _apply_policy_corrections,
    _canonical_pair,
    _normal_type_policy,
    load_dataset_alignment_config,
)


def _legacy_targets() -> tuple[dict[str, object], dict[str, object]]:
    normal: dict[str, object] = {
        "schemaVersion": "2.0.0",
        "documentPatch": {
            "containers": [
                {
                    "containerNumber": "TCLU6905627",
                    "typeCode": "45G1",
                }
            ],
            "goodsItems": [
                {
                    "description": "WIDGETS IN CARTONS",
                    "packages": [{"quantity": 2, "type": "CARTONS"}],
                }
            ],
        },
    }
    relation: dict[str, object] = {
        "schemaVersion": "3.0.0-experimental",
        "documentPatch": {
            "containers": [
                {
                    "containerNumber": "TCLU6905627",
                    "typeCategory": "FORTY_FOOT_HIGH_CUBE",
                }
            ],
            "cargoGroups": [
                {
                    "groupId": "g1",
                    "description": "WIDGETS IN CARTONS",
                }
            ],
            "cargoPackages": [
                {
                    "packageId": "p1",
                    "groupId": "g1",
                    "quantity": 2,
                    "typeCategory": "PACKAGE_CARTON",
                }
            ],
        },
    }
    return normal, relation


def test_legacy_types_are_aligned_to_printed_single_source_contract() -> None:
    normal, relation = _legacy_targets()

    container_codes, package_codes = _normal_type_policy(normal)
    aligned_relation, package_categories, container_categories = _align_relation_target(
        relation, normal
    )
    canonical_normal, canonical_relation = _canonical_pair(normal, aligned_relation)

    normal_patch = canonical_normal["documentPatch"]
    relation_patch = canonical_relation["documentPatch"]
    assert (container_codes, package_codes) == (1, 0)
    assert (package_categories, container_categories) == (1, 1)
    assert normal_patch["containers"][0] == {
        "containerNumber": "TCLU6905627",
        "typeDescription": "45G1",
    }
    assert relation_patch["containers"][0] == normal_patch["containers"][0]
    assert normal_patch["goodsItems"][0]["packages"][0] == {
        "quantity": 2,
        "type": "CARTONS",
    }
    assert relation_patch["cargoPackages"][0]["typeDescription"] == "CARTONS"
    assert "typeCategory" not in relation_patch["cargoPackages"][0]


def test_reviewed_correction_requires_exact_before_value_and_raw_evidence() -> None:
    normal, _ = _legacy_targets()
    correction = PolicyCorrection.model_validate(
        {
            "correctionId": "widgets-description",
            "documentId": "doc_" + "a" * 64,
            "kind": "cargo_description",
            "goodsIndex": 0,
            "packageIndex": None,
            "before": "WIDGETS IN CARTONS",
            "after": "WIDGETS",
            "rawOcrEvidence": (
                {
                    "pageNumber": 1,
                    "rawValue": "WIDGETS IN CARTONS",
                },
            ),
            "normalizationRule": "Remove the separately modeled package wording.",
        },
        strict=True,
    )

    _apply_policy_corrections(
        normal,
        corrections=(correction,),
        joined_raw_text="--- PAGE 1 ---\nWIDGETS IN CARTONS\n",
    )
    assert normal["documentPatch"]["goodsItems"][0]["description"] == "WIDGETS"

    with pytest.raises(DatasetAlignmentError, match="before-value differs"):
        _apply_policy_corrections(
            normal,
            corrections=(correction,),
            joined_raw_text="--- PAGE 1 ---\nWIDGETS IN CARTONS\n",
        )


def test_repository_alignment_config_is_strict_and_loadable() -> None:
    config = load_dataset_alignment_config(
        Path("configs/transforms/mpci_bl_combined1109_current_policy_alignment.yaml")
    )

    assert config.legacy.expected_aligned_records == 483
    assert config.current.expected_training_records == 626
    assert config.output.combined_dataset_id == "mpci-bl-combined1109-current-policy-v1"


def test_policy_correction_json_remains_strict_json() -> None:
    path = Path("configs/transforms/mpci_bl_combined487_current_policy_corrections.json")
    payload = json.loads(path.read_bytes())

    assert payload["schemaVersion"] == 1
    assert len(payload["corrections"]) == 20
