"""Decoder order is part of supervision, not the semantic hash contract."""

import json

from document_ocr.hashing import canonical_json_bytes
from document_ocr.label_schemas.bill_of_lading_v5 import BillOfLadingRelationExplicitV5Label
from document_ocr.training.tasks import canonical_json, get_training_task


def test_v5_decoder_places_relations_after_facts_without_changing_values():
    target = {
        "schemaVersion": "5.0.0-experimental",
        "documentPatch": {
            "cargoAllocationGroups": [
                {
                    "groupId": "g1",
                    "coverage": "container_membership_only",
                    "allocations": [{"containerNumber": "CAIU7896610"}],
                }
            ],
            "cargoGroups": [{"groupId": "g1", "description": "TEST GOODS"}],
            "containers": [{"containerNumber": "CAIU7896610"}],
            "transport": {"vesselName": "EXAMPLE"},
        },
    }
    label = BillOfLadingRelationExplicitV5Label.model_validate_json(json.dumps(target), strict=True)
    assert list(label.canonical_target()["documentPatch"])[-1] == "cargoAllocationGroups"
    encoded = canonical_json(target)
    decoded = json.loads(encoded)
    assert decoded == target
    assert list(decoded["documentPatch"])[-1] == "cargoAllocationGroups"
    assert next(iter(target["documentPatch"])) == "cargoAllocationGroups"
    assert canonical_json_bytes(decoded) == canonical_json_bytes(target)


def test_v5_prompt_orders_relation_property_last():
    task = get_training_task("bill_of_lading_relation_explicit_v5")
    schema = json.loads(task.prompt_schema_json())
    properties = schema["$defs"]["RelationExplicitDocumentPatchV5"]["properties"]
    assert list(properties)[-1] == "cargoAllocationGroups"
    assert "cargoPackages" in properties and "cargoGroups" in properties
    assert "containers" in properties


def test_other_targets_retain_exact_canonical_representation():
    target = {"z": 1, "a": {"cargoAllocationGroups": [], "transport": {"z": 2, "a": 1}}}
    assert canonical_json(target) == json.dumps(target, sort_keys=True, separators=(",", ":"))


def test_absent_relations_are_not_added_to_sparse_v5_target():
    target = {
        "schemaVersion": "5.0.0-experimental",
        "documentPatch": {"transport": {"vesselName": "EXAMPLE"}},
    }
    label = BillOfLadingRelationExplicitV5Label.model_validate_json(json.dumps(target), strict=True)
    assert label.canonical_target() == target
    assert json.loads(canonical_json(label.canonical_target())) == target
