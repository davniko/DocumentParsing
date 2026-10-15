"""The direct decoder contract changes its envelope, never its extraction facts."""

from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest
from jsonschema import Draft202012Validator, ValidationError

from document_ocr.hashing import canonical_json_bytes, sha256_bytes, sha256_file
from document_ocr.training.config import DatasetFileConfig
from document_ocr.training.metrics import assess_prediction, structured_metrics
from document_ocr.training.relation_constraints import _source_tokens
from document_ocr.training.tasks import (
    RelationExplicitTaskConstraints,
    canonical_json,
    get_training_task,
)

DIRECT_TASK = "bill_of_lading_extraction_v7_reduced_direct"
HISTORICAL_TASK = "bill_of_lading_extraction_v7_reduced"


def facts() -> dict:
    return {
        "billOfLadingNumber": "BILL-123",
        "negotiability": None,
        "parties": {
            "shipper": {"name": "EXPORTER LTD", "addressLine": "UNIT 4, USA"},
            "notifyParties": [{"name": "NOTIFY LTD", "sameAs": None}],
        },
        "containerInformation": [
            {
                "equipmentIdentifier": "TTNU8111930",
                "sizeCategory": "FORTY_FOOT_HIGH_CUBE",
                "typeCategory": "GENERAL_PURPOSE",
            },
            {"equipmentIdentifier": "FCIU3651201", "typeCategory": "REFRIGERATED"},
        ],
        "goodsItemDetails": [
            {
                "description": "COMPLETE PRODUCT WORDING",
                "hsCodes": ["123456", "123457"],
                "numberAndTypeOfPackages": [
                    {"packageQuantity": 24, "typeCategory": "PACKAGE_BAG"}
                ],
                "splitGoodsPlacement": [
                    {"equipmentIdentifier": "TTNU8111930", "packageQuantity": 22},
                    {"equipmentIdentifier": "FCIU3651201", "packageQuantity": 2},
                ],
            }
        ],
    }


def wrapped(value: dict) -> dict:
    return {"schemaVersion": "7.0.0", "documentPatch": value}


def test_direct_target_is_exact_document_facts_and_rejects_envelopes() -> None:
    task = get_training_task(DIRECT_TASK)
    direct = facts()
    assert task.document_field is None
    assert task.document_facts(direct) is direct
    assert task.canonicalize(direct) == direct
    assert task.document_model() is task.target_model
    assert task.canonicalize(direct)["negotiability"] is None
    assert task.canonicalize(direct)["parties"]["notifyParties"][0]["sameAs"] is None
    with pytest.raises(ValueError):
        task.canonicalize(wrapped(direct))
    for field, value in (("schemaVersion", "7.0.0"), ("documentPatch", direct)):
        invalid = {**direct, field: value}
        with pytest.raises(ValueError, match="Extra inputs"):
            task.canonicalize(invalid)
    with pytest.raises(ValueError):
        get_training_task(HISTORICAL_TASK).canonicalize(direct)
    assert get_training_task(HISTORICAL_TASK).canonicalize(wrapped(direct)) == wrapped(direct)


def test_direct_export_and_prompt_schemas_have_no_envelope() -> None:
    task = get_training_task(DIRECT_TASK)
    for schema in (task.target_schema(), json.loads(task.prompt_schema_json())):
        assert "schemaVersion" not in schema["properties"]
        assert "documentPatch" not in schema["properties"]
        assert schema["required"] == ["negotiability"]
        assert "forwardingAndExportReferences" not in schema["properties"]
        assert "carrier" not in schema["$defs"]["ExtractionPartiesV7"]["properties"]
        assert "vesselFlagCountry" not in schema["$defs"]["TransportV7"]["properties"]
        assert "marksAndNumbers" not in schema["$defs"]["GoodsItemDetailsV7"]["properties"]
        Draft202012Validator(schema).validate(facts())
        with pytest.raises(ValidationError):
            Draft202012Validator(schema).validate(wrapped(facts()))
    prompt = json.loads(task.prompt_schema_json())
    assert {"type": "null"} in prompt["properties"]["negotiability"]["anyOf"]
    assert {"type": "null"} in prompt["$defs"]["NotifyPartyV7"]["properties"]["sameAs"]["anyOf"]


@pytest.mark.parametrize(
    "owner_path,field,value",
    [
        ([], "forwardingAndExportReferences", ["INV: 123"]),
        (["parties"], "carrier", {"name": "CARRIER LTD"}),
        (["transport"], "vesselFlagCountry", "EGYPT"),
        (["goodsItemDetails", 0], "marksAndNumbers", ["N/M"]),
    ],
)
def test_direct_reduced_target_rejects_excluded_fields(owner_path, field, value) -> None:
    direct = facts()
    owner = direct
    for key in owner_path:
        if isinstance(key, str) and key not in owner:
            owner[key] = {"vesselName": "VESSEL"}
        owner = owner[key]
    owner[field] = value
    with pytest.raises(ValueError, match="outside reduced V7"):
        get_training_task(DIRECT_TASK).canonicalize(direct)


@pytest.mark.parametrize("mutation", ["none", "quantity", "omission", "extra", "permutation"])
@pytest.mark.parametrize("case_sensitive", [True, False])
def test_direct_metrics_match_historical_on_identical_facts(mutation, case_sensitive) -> None:
    reference = facts()
    prediction = copy.deepcopy(reference)
    if mutation == "quantity":
        prediction["goodsItemDetails"][0]["splitGoodsPlacement"][0]["packageQuantity"] = 21
    elif mutation == "omission":
        prediction["containerInformation"].pop()
        prediction["goodsItemDetails"][0]["splitGoodsPlacement"].pop()
    elif mutation == "extra":
        prediction["parties"]["shipper"]["contactName"] = "UNSUPPORTED CONTACT"
    elif mutation == "permutation":
        prediction["containerInformation"].reverse()
        prediction["goodsItemDetails"][0]["hsCodes"].reverse()
        prediction["goodsItemDetails"][0]["splitGoodsPlacement"].reverse()
    direct_scores, direct_assessments = structured_metrics(
        [canonical_json(prediction)],
        [canonical_json(reference)],
        get_training_task(DIRECT_TASK),
        case_sensitive=case_sensitive,
    )
    old_scores, old_assessments = structured_metrics(
        [canonical_json(wrapped(prediction))],
        [canonical_json(wrapped(reference))],
        get_training_task(HISTORICAL_TASK),
        case_sensitive=case_sensitive,
    )
    assert direct_scores == old_scores
    direct, old = direct_assessments[0], old_assessments[0]
    for field in (
        "predicted_field_values", "reference_field_values", "predicted_cargo_relation_facts",
        "reference_cargo_relation_facts", "predicted_category_values", "reference_category_values",
    ):
        assert getattr(direct, field) == getattr(old, field)
    if mutation in {"none", "permutation"}:
        assert direct_scores["field_value_f1"] == 1
        assert direct_scores["field_value_accuracy"] == 1
        assert direct_scores["cargo_relation_f1"] == 1
        assert direct_scores["category_value_f1"] == 1


def test_direct_metrics_reject_old_envelope_even_when_its_facts_match() -> None:
    reference = facts()
    assessment = assess_prediction(
        canonical_json(wrapped(reference)), canonical_json(reference),
        get_training_task(DIRECT_TASK),
    )
    assert assessment.json_valid
    assert not assessment.schema_valid
    assert not assessment.canonical_exact_match
    assert not (assessment.predicted_field_values & assessment.reference_field_values)


def test_direct_serialization_preserves_placement_last() -> None:
    wire = json.loads(canonical_json(facts()))
    assert list(wire["goodsItemDetails"][0])[-1] == "splitGoodsPlacement"
    assert "schemaVersion" not in wire and "documentPatch" not in wire


def test_direct_task_constraints_preserve_shape_and_freeze_vocabularies() -> None:
    task = get_training_task(DIRECT_TASK)
    constraints = RelationExplicitTaskConstraints.model_validate(
        {
            "schemaVersion": 1,
            "task": DIRECT_TASK,
            "basePromptSchemaSha256": task.base_prompt_schema_sha256(),
            "targetSchemaSha256": sha256_bytes(canonical_json_bytes(task.target_schema())),
            "packageRegistrySha256": "a" * 64,
            "containerRegistrySha256": "b" * 64,
            "packageCategoryTokens": ("PACKAGE_BAG",),
            "containerCategoryTokens": ("GENERAL_PURPOSE", "REFRIGERATED"),
        },
        strict=True,
    )
    bound = task.bind_constraints(constraints)
    assert bound.document_field is None
    assert bound.canonicalize(facts()) == facts()
    Draft202012Validator(json.loads(bound.prompt_schema_json())).validate(facts())
    invalid = facts()
    invalid["containerInformation"][0]["typeCategory"] = "OPEN_TOP"
    with pytest.raises(ValueError, match="frozen vocabulary"):
        bound.canonicalize(invalid)


def test_direct_noncanonical_nulls_and_dangling_relations_remain_rejected() -> None:
    task = get_training_task(DIRECT_TASK)
    invalid = facts()
    invalid["issueDate"] = None
    with pytest.raises(ValueError, match="canonical sparse"):
        task.canonicalize(invalid)
    invalid = facts()
    invalid["containerInformation"].pop()
    with pytest.raises(ValueError, match="absent container"):
        task.canonicalize(invalid)


def test_direct_vocabulary_builder_reads_only_the_declared_root(tmp_path: Path) -> None:
    source = tmp_path / "direct.jsonl"
    source.write_bytes(canonical_json_bytes({"target": facts()}) + b"\n")
    configured = DatasetFileConfig(path=str(source), sha256=sha256_file(source), records=1)
    packages, equipment = _source_tokens(
        project_root=tmp_path, source=configured, target_field="target", task_name=DIRECT_TASK
    )
    assert packages == {"PACKAGE_BAG"}
    assert equipment == {"GENERAL_PURPOSE", "REFRIGERATED"}
    source.write_bytes(canonical_json_bytes({"target": wrapped(facts())}) + b"\n")
    configured = DatasetFileConfig(path=str(source), sha256=sha256_file(source), records=1)
    with pytest.raises(ValueError, match="not canonical"):
        _source_tokens(
            project_root=tmp_path, source=configured, target_field="target", task_name=DIRECT_TASK
        )
