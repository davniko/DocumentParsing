from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest

from document_ocr.training.config import PromptConfig
from document_ocr.training.metrics import structured_metrics
from document_ocr.training.prompting import load_prompt
from document_ocr.training.tasks import canonical_json, get_training_task


def target() -> dict:
    return {
        "schemaVersion": "7.0.0",
        "documentPatch": {
            "negotiability": "non_negotiable",
            "parties": {
                "shipper": {
                    "name": "EXPORTER LTD",
                    "addressLine": "UNIT 4, NEWBERG OR 97132, USA",
                    "country": "USA",
                }
            },
            "containerInformation": [{"equipmentIdentifier": "TTNU8111930"}],
            "goodsItemDetails": [
                {
                    "description": "PRODUCT A AND PRODUCT B",
                    "hsCodes": ["123456", "123457"],
                    "numberAndTypeOfPackages": [
                        {"packageQuantity": 24, "typeCategory": "PACKAGE_BAG"}
                    ],
                    "splitGoodsPlacement": [
                        {"equipmentIdentifier": "TTNU8111930", "packageQuantity": 22},
                        {"equipmentIdentifier": "TTNU8111930", "packageQuantity": 2},
                    ],
                }
            ],
        },
    }


@pytest.mark.parametrize(
    "task_name", ["bill_of_lading_extraction_v7", "bill_of_lading_extraction_v7_reduced"]
)
def test_full_postal_contract_and_relation_metrics_round_trip(task_name: str) -> None:
    task = get_training_task(task_name)
    source = target()
    assert task.canonicalize(source) == source
    wire = canonical_json(source)
    assert (
        list(json.loads(wire)["documentPatch"]["goodsItemDetails"][0])[-1] == "splitGoodsPlacement"
    )
    scores, _ = structured_metrics([wire], [wire], task)
    for metric in (
        "field_value_f1",
        "field_value_accuracy",
        "cargo_relation_f1",
        "category_value_f1",
    ):
        assert scores[metric] == 1.0
    changed = copy.deepcopy(source)
    changed["documentPatch"]["goodsItemDetails"][0]["splitGoodsPlacement"][1]["packageQuantity"] = 3
    scores, _ = structured_metrics([canonical_json(changed)], [wire], task)
    assert scores["cargo_relation_f1"] < 1
    assert scores["category_value_f1"] == 1


@pytest.mark.parametrize("field", ["address", "city"])
def test_legacy_party_fields_are_not_accepted(field: str) -> None:
    source = target()
    source["documentPatch"]["parties"]["shipper"][field] = "NEWBERG"
    with pytest.raises(ValueError, match="Extra inputs"):
        get_training_task("bill_of_lading_extraction_v7").canonicalize(source)


@pytest.mark.parametrize(
    "equipment",
    [
        {"typeCategory": "GENERAL_PURPOSE"},
        {"typeCategory": "REFRIGERATED", "temperatureSetpoint": {"value": 1, "unit": "celsius"}},
        {"sizeCategory": "FORTY_FOOT_HIGH_CUBE"},
        {"sizeCategory": "TWENTY_FOOT_STANDARD_HEIGHT", "typeCategory": "PRESSURIZED_TANK"},
    ],
)
def test_equipment_extracts_known_categories_without_inventing_other_dimensions(equipment):
    from jsonschema import Draft202012Validator

    source = target()
    source["documentPatch"]["containerInformation"][0].update(equipment)
    task = get_training_task("bill_of_lading_extraction_v7_reduced")
    assert task.canonicalize(source) == source
    Draft202012Validator(json.loads(task.prompt_schema_json())).validate(source)
    invalid = copy.deepcopy(source)
    invalid["documentPatch"]["containerInformation"][0]["typeDescription"] = "CONTAINER"
    with pytest.raises(ValueError, match="fallback"):
        task.canonicalize(invalid)
    if equipment.get("typeCategory") in {"GENERAL_PURPOSE", "PRESSURIZED_TANK"}:
        invalid = copy.deepcopy(source)
        invalid["documentPatch"]["containerInformation"][0]["temperatureSetpoint"] = {
            "value": 1,
            "unit": "celsius",
        }
        with pytest.raises(ValueError, match="temperature-capable"):
            task.canonicalize(invalid)


def test_no_cargo_overflow_or_dangling_placement() -> None:
    task = get_training_task("bill_of_lading_extraction_v7")
    source = target()
    source["documentPatch"]["goodsItemDetails"][0]["additionalInformation"] = ["PRODUCT C"]
    with pytest.raises(ValueError, match="Extra inputs"):
        task.canonicalize(source)
    del source["documentPatch"]["goodsItemDetails"][0]["additionalInformation"]
    del source["documentPatch"]["containerInformation"]
    with pytest.raises(ValueError, match="absent container"):
        task.canonicalize(source)


@pytest.mark.parametrize(
    "field", ["vesselFlagCountry", "marksAndNumbers", "forwardingAndExportReferences"]
)
@pytest.mark.parametrize("explicit_null", [False, True])
def test_full_annotation_fields_survive_canonicalization(field: str, explicit_null: bool) -> None:
    source = target()
    patch = source["documentPatch"]
    if field == "vesselFlagCountry":
        owner = patch["transport"] = {"vesselName": "EXAMPLE"}
        value = "EGYPT"
    elif field == "marksAndNumbers":
        owner, value = patch["goodsItemDetails"][0], ["N/M"]
    else:
        owner, value = patch, ["INV: 123"]
    owner[field] = None if explicit_null else value
    task = get_training_task("bill_of_lading_extraction_v7")
    if explicit_null:
        with pytest.raises(ValueError, match="canonical sparse"):
            task.canonicalize(source)
        from document_ocr.label_schemas.bill_of_lading_v7 import BillOfLadingExtractionV7Label

        result = BillOfLadingExtractionV7Label.model_validate_json(
            json.dumps(source)
        ).canonical_target()
        assert task.canonicalize(result) == result
    else:
        result = task.canonicalize(source)
    result_owner = result["documentPatch"]
    if field == "vesselFlagCountry":
        result_owner = result_owner["transport"]
    elif field == "marksAndNumbers":
        result_owner = result_owner["goodsItemDetails"][0]
    if explicit_null:
        assert field not in result_owner
    else:
        assert result_owner[field] == value


def test_training_prompt_matches_target_and_keeps_ocr_literal() -> None:
    root = Path(__file__).resolve().parents[1]
    task = get_training_task("bill_of_lading_extraction_v7")
    prompt = load_prompt(
        root,
        PromptConfig(
            path="prompts/training/mpci_bl_extraction_v7.txt",
            placeholder="{{document_text}}",
            schema_placeholder="{{output_schema}}",
        ),
        task,
    )
    schema = json.loads(task.prompt_schema_json())
    props = schema["$defs"]["ExtractionPartyV7"]["properties"]
    assert "addressLine" in props and "country" in props
    assert "city" not in props and "address" not in props
    assert "additionalInformation" not in schema["$defs"]["GoodsItemDetailsV7"]["properties"]
    assert "marksAndNumbers" in schema["$defs"]["GoodsItemDetailsV7"]["properties"]
    transport = schema["$defs"]["TransportV7"]["properties"]
    assert "vesselFlagCountry" in transport and "vesselImoNumber" in transport
    assert (
        "forwardingAndExportReferences"
        in schema["$defs"]["BillOfLadingDocumentPatchV7"]["properties"]
    )
    raw = "UNIT 4\nNEWBERG OR 97132\nUSA"
    assert raw in prompt.render(raw)
    assert "{{output_schema}}" not in prompt.text


@pytest.mark.parametrize(
    "definition,path,field,value",
    [
        ("ExtractionPartiesV7", ["parties"], "carrier", {"name": "CARRIER LTD"}),
        ("TransportV7", ["transport"], "vesselFlagCountry", "EGYPT"),
        ("GoodsItemDetailsV7", ["goodsItemDetails", 0], "marksAndNumbers", ["N/M"]),
        ("BillOfLadingDocumentPatchV7", [], "forwardingAndExportReferences", ["INV: 123"]),
    ],
)
def test_reduced_contract_rejects_dropped_fields_without_changing_annotation_scope(
    definition: str, path: list, field: str, value: object
) -> None:
    from jsonschema import Draft202012Validator, ValidationError

    task = get_training_task("bill_of_lading_extraction_v7_reduced")
    full_task = get_training_task("bill_of_lading_extraction_v7")
    source = target()
    assert task.canonicalize(source) == source
    prompt_schema = json.loads(task.prompt_schema_json())
    assert field not in prompt_schema["$defs"][definition]["properties"]
    assert field in json.loads(full_task.prompt_schema_json())["$defs"][definition]["properties"]
    owner = source["documentPatch"]
    for key in path:
        if isinstance(key, str) and key not in owner:
            owner[key] = {}
        owner = owner[key]
    owner[field] = value
    assert full_task.canonicalize(source) == source
    with pytest.raises(ValueError, match="outside reduced V7"):
        task.canonicalize(source)
    for schema in (task.target_schema(), prompt_schema):
        with pytest.raises(ValidationError):
            Draft202012Validator(schema).validate(source)
    assert field in full_task.target_schema()["$defs"][definition]["properties"]
