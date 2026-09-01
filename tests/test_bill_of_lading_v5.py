from __future__ import annotations

import json

import pytest
from pydantic import ValidationError

from document_ocr.label_schemas.bill_of_lading_v5 import (
    BillOfLadingRelationExplicitV5Label,
    RelationExplicitContainerV5,
    migrate_relation_v4_target_to_v5,
    semantic_container_code,
)
from document_ocr.synthesis.bill_of_lading_domain import V5_ADAPTER
from document_ocr.synthesis.task_adapter import BILL_OF_LADING_V5_TASK_ADAPTER


def test_semantic_container_pair_projects_to_exact_application_code() -> None:
    assert semantic_container_code("FORTY_FOOT_HIGH_CUBE", "REFRIGERATED") == "45RE"
    assert semantic_container_code("TWENTY_FOOT_STANDARD_HEIGHT", "GENERAL_PURPOSE") == "22GP"


def test_temperature_requires_complete_temperature_capable_equipment_pair() -> None:
    with pytest.raises(ValidationError, match="present together"):
        RelationExplicitContainerV5.model_validate(
            {"containerNumber": "CAIU7896610", "typeCategory": "REFRIGERATED"},
            strict=True,
        )
    with pytest.raises(ValidationError, match="temperature-capable"):
        RelationExplicitContainerV5.model_validate(
            {
                "containerNumber": "CAIU7896610",
                "sizeCategory": "FORTY_FOOT_HIGH_CUBE",
                "typeCategory": "GENERAL_PURPOSE",
                "temperatureSetpoint": {"value": -18.0, "unit": "celsius"},
            },
            strict=True,
        )
    with pytest.raises(ValidationError, match="printed fallback"):
        RelationExplicitContainerV5.model_validate(
            {
                "containerNumber": "CAIU7896610",
                "temperatureSetpoint": {"value": -18.0, "unit": "celsius"},
            },
            strict=True,
        )
    with pytest.raises(ValidationError, match="printed fallback"):
        RelationExplicitContainerV5.model_validate(
            {
                "containerNumber": "CAIU7896610",
                "typeDescription": "40RH",
                "sizeCategory": "FORTY_FOOT_HIGH_CUBE",
                "typeCategory": "REFRIGERATED",
            },
            strict=True,
        )


def test_v4_migration_and_enrichment_preserve_relational_inverse() -> None:
    source = {
        "schemaVersion": "4.0.0-experimental",
        "documentPatch": {
            "containers": [
                {
                    "containerNumber": "CAIU7896610",
                    "typeDescription": "40RH",
                    "temperatureSetpoint": {"value": -18.0, "unit": "celsius"},
                }
            ]
        },
    }
    target = migrate_relation_v4_target_to_v5(source)
    container = target["documentPatch"]["containers"][0]
    container.pop("typeDescription")
    container["sizeCategory"] = "FORTY_FOOT_HIGH_CUBE"
    container["typeCategory"] = "REFRIGERATED"
    canonical = BillOfLadingRelationExplicitV5Label.model_validate_json(
        json.dumps(target), strict=True
    ).canonical_target()
    tables = V5_ADAPTER.project(document_id="doc_1", source_row_index=0, target=canonical)
    assert tables.rows["containers"][0]["size_category"] == "FORTY_FOOT_HIGH_CUBE"
    assert tables.rows["containers"][0]["type_category"] == "REFRIGERATED"
    assert V5_ADAPTER.reconstruct(document_id="doc_1", tables=tables.rows) == canonical
    assert (
        BILL_OF_LADING_V5_TASK_ADAPTER.validate_target(document_id="doc_1", target=canonical)
        == canonical
    )
