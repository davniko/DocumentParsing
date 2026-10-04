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


def test_full_postal_contract_and_relation_metrics_round_trip() -> None:
    task = get_training_task("bill_of_lading_extraction_v7")
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
    raw = "UNIT 4\nNEWBERG OR 97132\nUSA"
    assert raw in prompt.render(raw)
    assert "{{output_schema}}" not in prompt.text
