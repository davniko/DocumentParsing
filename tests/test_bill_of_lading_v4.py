from __future__ import annotations

import json
from pathlib import Path

import pytest
from pydantic import ValidationError

from document_ocr.label_schemas.bill_of_lading_v4 import (
    BillOfLadingRelationExplicitV4Label,
    RelationExplicitDangerousGoodsV4,
    migrate_relation_v3_target_to_v4,
)
from document_ocr.label_schemas.mpci_bill_of_lading import (
    DangerousGoodsShipmentFlashpoint,
)
from document_ocr.label_schemas.mpci_projection import (
    MpciProjectionError,
    project_relation_v4_dangerous_goods_to_mpci,
)
from document_ocr.synthesis.bill_of_lading_domain import V4_ADAPTER
from document_ocr.synthesis.task_adapter import BILL_OF_LADING_V4_TASK_ADAPTER


def _v3_target() -> dict[str, object]:
    return {
        "schemaVersion": "3.0.0-experimental",
        "documentPatch": {
            "cargoGroups": [
                {
                    "groupId": "g1",
                    "dangerousGoods": [
                        {
                            "unNumber": "1993",
                            "hazardCategory": "FLAMMABLE_LIQUIDS",
                            "subsidiaryHazardCategory": "CORROSIVE_SUBSTANCES",
                            "flashPoint": {
                                "temperature": {"value": 16.0, "unit": "celsius"},
                                "packingGroupCategory": "MEDIUM_DANGER",
                            },
                        }
                    ],
                }
            ]
        },
    }


def test_v3_migration_moves_packing_group_and_wraps_subsidiary() -> None:
    migrated = migrate_relation_v3_target_to_v4(_v3_target())
    dangerous = migrated["documentPatch"]["cargoGroups"][0]["dangerousGoods"][0]
    assert migrated["schemaVersion"] == "4.0.0-experimental"
    assert dangerous == {
        "unNumber": "1993",
        "hazardCategory": "FLAMMABLE_LIQUIDS",
        "subsidiaryHazardCategories": ["CORROSIVE_SUBSTANCES"],
        "packingGroupCategory": "MEDIUM_DANGER",
        "flashPoint": {"temperature": {"value": 16.0, "unit": "celsius"}},
    }
    BillOfLadingRelationExplicitV4Label.model_validate_json(json.dumps(migrated), strict=True)


def test_v4_allows_independent_packing_group_and_flashpoint() -> None:
    packing_only = RelationExplicitDangerousGoodsV4.model_validate(
        {"unNumber": "1993", "packingGroupCategory": "LOW_DANGER"}, strict=True
    )
    flashpoint_only = RelationExplicitDangerousGoodsV4.model_validate(
        {"flashPoint": {"temperature": {"value": 42.0, "unit": "celsius"}}},
        strict=True,
    )
    assert packing_only.flashPoint is None
    assert flashpoint_only.packingGroupCategory is None
    with pytest.raises(ValidationError, match="must contain supported evidence"):
        RelationExplicitDangerousGoodsV4.model_validate({}, strict=True)


def test_v4_subsidiaries_are_unique_and_source_ordered() -> None:
    value = RelationExplicitDangerousGoodsV4.model_validate(
        {
            "hazardCategory": "GASES",
            "subsidiaryHazardCategories": (
                "TOXIC_AND_INFECTIOUS_SUBSTANCES",
                "CORROSIVE_SUBSTANCES",
            ),
        },
        strict=True,
    )
    assert value.subsidiaryHazardCategories == (
        "TOXIC_AND_INFECTIOUS_SUBSTANCES",
        "CORROSIVE_SUBSTANCES",
    )
    with pytest.raises(ValidationError, match="must be unique"):
        RelationExplicitDangerousGoodsV4.model_validate(
            {
                "subsidiaryHazardCategories": (
                    "CORROSIVE_SUBSTANCES",
                    "CORROSIVE_SUBSTANCES",
                )
            },
            strict=True,
        )


def test_mpci_sparse_flashpoint_row_accepts_packing_only_but_rejects_partial_pair() -> None:
    row = DangerousGoodsShipmentFlashpoint.model_validate(
        {"packagingDangerLevelCode": "2"}, strict=True
    )
    assert row.shipmentFlashpointDegree is None
    with pytest.raises(ValidationError, match="degree and unit"):
        DangerousGoodsShipmentFlashpoint.model_validate(
            {"shipmentFlashpointDegree": 12.0}, strict=True
        )
    with pytest.raises(ValidationError, match="degree and unit"):
        DangerousGoodsShipmentFlashpoint.model_validate({"measurementUnitCode": "CEL"}, strict=True)
    with pytest.raises(ValidationError, match="supported evidence"):
        DangerousGoodsShipmentFlashpoint.model_validate({}, strict=True)


def test_v4_mpci_projection_handles_packing_only_and_rejects_multiple_subsidiaries() -> None:
    value = RelationExplicitDangerousGoodsV4.model_validate(
        {
            "unNumber": "1993",
            "hazardCategory": "FLAMMABLE_LIQUIDS",
            "subsidiaryHazardCategories": ("CORROSIVE_SUBSTANCES",),
            "packingGroupCategory": "MEDIUM_DANGER",
        },
        strict=True,
    )
    assert project_relation_v4_dangerous_goods_to_mpci(value) == {
        "hazardCode": [
            {
                "hazardIdentificationCode": "3",
                "additionalHazardClassificationIdentifier": "8",
            }
        ],
        "undgInformation": {"identifier": "1993"},
        "dangerousGoodsShipmentFlashpoint": [{"packagingDangerLevelCode": "2"}],
    }
    ambiguous = RelationExplicitDangerousGoodsV4.model_validate(
        {
            "hazardCategory": "GASES",
            "subsidiaryHazardCategories": (
                "TOXIC_AND_INFECTIOUS_SUBSTANCES",
                "CORROSIVE_SUBSTANCES",
            ),
        },
        strict=True,
    )
    with pytest.raises(MpciProjectionError, match="multiple subsidiary"):
        project_relation_v4_dangerous_goods_to_mpci(ambiguous)


def test_v4_mpci_projection_distinguishes_not_assigned_from_absent_packing_group() -> None:
    explicit = RelationExplicitDangerousGoodsV4.model_validate(
        {"unNumber": "1993", "packingGroupCategory": "NOT_ASSIGNED"},
        strict=True,
    )
    absent = RelationExplicitDangerousGoodsV4.model_validate({"unNumber": "1993"}, strict=True)
    assert project_relation_v4_dangerous_goods_to_mpci(explicit) == {
        "undgInformation": {"identifier": "1993"},
        "dangerousGoodsShipmentFlashpoint": [{"packagingDangerLevelCode": "4"}],
    }
    assert project_relation_v4_dangerous_goods_to_mpci(absent) == {
        "undgInformation": {"identifier": "1993"}
    }


def test_v4_relational_inverse_preserves_ordered_subsidiaries() -> None:
    target = migrate_relation_v3_target_to_v4(_v3_target())
    tables = V4_ADAPTER.project(document_id="doc_1", source_row_index=0, target=target)
    assert tables.rows["dangerous_goods"][0]["packing_group_category"] == "MEDIUM_DANGER"
    assert tables.rows["dangerous_goods_subsidiary_hazards"] == [
        {
            "subsidiary_hazard_id": "doc_1:dangerous-goods-subsidiary:g1:0:0",
            "document_id": "doc_1",
            "dangerous_goods_id": "doc_1:dangerous-goods:g1:0",
            "subsidiary_hazard_order": 0,
            "hazard_category": "CORROSIVE_SUBSTANCES",
        }
    ]
    assert V4_ADAPTER.reconstruct(document_id="doc_1", tables=tables.rows) == target
    assert (
        BILL_OF_LADING_V4_TASK_ADAPTER.validate_target(document_id="doc_1", target=target) == target
    )


def test_real_v3_corpus_migration_preserves_all_represented_dg_facts() -> None:
    corpus_path = Path(
        "artifacts/kie-training/datasets/"
        "mpci-bl-combined1157-task-facing-package-categories-v2/records.jsonl"
    )
    if not corpus_path.is_file():
        pytest.skip("the immutable 1,157-record corpus is not present")

    documents = 0
    source_dg_rows = 0
    migrated_dg_rows = 0
    source_packing_groups = 0
    migrated_packing_groups = 0
    source_subsidiaries = 0
    migrated_subsidiaries = 0
    with corpus_path.open("rb") as stream:
        for raw in stream:
            row = json.loads(raw)
            source = row["target"]
            migrated = migrate_relation_v3_target_to_v4(source)
            document_id = row["documentId"]
            BILL_OF_LADING_V4_TASK_ADAPTER.validate_target(
                document_id=document_id,
                target=migrated,
            )
            source_groups = source["documentPatch"].get("cargoGroups", [])
            migrated_groups = migrated["documentPatch"].get("cargoGroups", [])
            for group in source_groups:
                for dangerous in group.get("dangerousGoods", []):
                    source_dg_rows += 1
                    source_subsidiaries += int(
                        dangerous.get("subsidiaryHazardCategory") is not None
                    )
                    source_packing_groups += int(
                        (dangerous.get("flashPoint") or {}).get("packingGroupCategory") is not None
                    )
            for group in migrated_groups:
                for dangerous in group.get("dangerousGoods", []):
                    migrated_dg_rows += 1
                    migrated_subsidiaries += len(dangerous.get("subsidiaryHazardCategories", []))
                    migrated_packing_groups += int(
                        dangerous.get("packingGroupCategory") is not None
                    )
            documents += 1

    assert documents == 1157
    assert source_dg_rows == migrated_dg_rows == 38
    assert source_packing_groups == migrated_packing_groups == 6
    assert source_subsidiaries == migrated_subsidiaries == 1
