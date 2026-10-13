"""Private identity generation and repeated document-reference ownership."""

from copy import deepcopy

import pytest

from document_ocr.synthesis.curated import SourceContract, digest
from document_ocr.synthesis.curated_campaign import host_variable_values, public_identity_updates
from document_ocr.synthesis.curated_templates import (
    compile_sampling_blueprint,
    render_sampling_blueprint,
)
from document_ocr.synthesis.curated_wording import (
    WordingField,
    WordingRequest,
    identity_key,
    unpack_wording,
    wording_field_requirement,
    wording_output_type,
)
from document_ocr.synthesis.generators import DeterministicStream, surface_pattern


@pytest.mark.parametrize(
    ("old", "new"),
    [
        ("SOURCE MARINE LTD (ALEXANDRIA OFFICE)", "SOURCE MARINE LTD (SHARJAH OFFICE)"),
        ("Source Marine Ltd", "SOURCE MARINE LTD (NEW DEPOT)"),
        ("SOURCE MARINE LTD (OLD BRANCH)", "Source Marine Ltd"),
        ("SOURCE MARINE LTD\n(OLD BRANCH)", "SOURCE MARINE LTD\n(NEW BRANCH)"),
        ("PERSON NAME (CONTACT)", "PERSON NAME (MANAGER)"),
    ],
)
def test_qualifier_only_identity_changes_fail_native_unpack(old, new):
    request = WordingRequest(
        "sample", "Company", (WordingField("n", "deliveryAgent name", old, "New name"),)
    )
    output = wording_output_type([request]).model_validate({"s0": {"n": new}})
    with pytest.raises(ValueError, match="copied source identity"):
        unpack_wording(output, [request])


@pytest.mark.parametrize(
    ("old", "new"),
    [
        ("SOURCE MARINE LTD (OLD BRANCH)", "NEW COAST LOGISTICS LLC (NEW BRANCH)"),
        ("FREIGHT SOLUTIONS", "GULF FREIGHT SOLUTIONS LLC"),
        ("(SOURCE COMPANY)", "(NEW COMPANY)"),
        ("SOURCE TRADING (MIDDLE EAST) LTD", "NEW TRADING (MIDDLE EAST) LTD"),
    ],
)
def test_new_identity_with_common_words_or_qualifiers_is_valid(old, new):
    request = WordingRequest(
        "sample", "Company", (WordingField("n", "deliveryAgent name", old, "New name"),)
    )
    output = wording_output_type([request]).model_validate({"s0": {"n": new}})
    assert unpack_wording(output, [request])["sample"]["n"] == new


def test_identity_requirement_is_carried_by_native_field_description():
    field = WordingField(
        "n", "deliveryAgent name", "SOURCE LTD", "New organization", "organization"
    )
    assert "office or location qualifier" in wording_field_requirement(field)
    request = WordingRequest("sample", "Company", (field,))
    schema = wording_output_type([request]).model_json_schema()
    assert "office or location qualifier" in str(schema)
    assert identity_key("(SOURCE)") == "source"


def test_document_ids_share_alias_values_and_preserve_formats():
    target = {
        "documentPatch": {
            "billOfLadingNumber": "ML250031",
            "masterBillOfLadingNumber": "ML250031",
            "originalBillOfLadingNumber": "OR-001/23",
            "transport": {"voyageNumber": "005E"},
        }
    }
    before = deepcopy(target)
    stream = DeterministicStream(13102026, "test", "sample")
    result = public_identity_updates(target, None, stream)
    patch = result["documentPatch"]
    assert target == before
    assert patch["billOfLadingNumber"] == patch["masterBillOfLadingNumber"] != "ML250031"
    for path in ("billOfLadingNumber", "masterBillOfLadingNumber", "originalBillOfLadingNumber"):
        assert surface_pattern(patch[path]) == surface_pattern(target["documentPatch"][path])
    assert patch["transport"]["voyageNumber"] != "005E"
    assert result == public_identity_updates(target, None, stream)


def test_master_reference_reuses_one_host_value_across_pages_and_label():
    text = "--- PAGE 1 ---\nBOOKING ML250031\nSTEEL BARS\n--- PAGE 2 ---\nMASTERBILLNO: ML250031\n"
    target = {
        "schemaVersion": "7.0.0",
        "documentPatch": {
            "negotiability": "non_negotiable",
            "masterBillOfLadingNumber": "ML250031",
            "goodsItemDetails": [{"description": "STEEL BARS"}],
        },
    }
    row = {"documentId": "source", "joinedRawText": text, "target": target}
    contract = SourceContract.model_validate(
        {
            "variables": [
                {
                    "key": "master_bill",
                    "kind": "identifier",
                    "value": "ML250031",
                    "meaning": "Master and booking alias",
                    "required_literals": [],
                    "occurrences": [
                        {"text": "ML250031", "occurrence": 1, "presentation": "text"},
                        {"text": "ML250031", "occurrence": 2, "presentation": "text"},
                    ],
                },
                {
                    "key": "goods",
                    "kind": "product",
                    "value": "STEEL BARS",
                    "meaning": "Goods",
                    "required_literals": [],
                    "occurrences": [
                        {"text": "STEEL BARS", "occurrence": 1, "presentation": "text"},
                    ],
                },
            ],
            "targets": [
                {"path": "documentPatch.masterBillOfLadingNumber", "expression": "{master_bill}"},
                {"path": "documentPatch.goodsItemDetails[0].description", "expression": "{goods}"},
            ],
            "fixed_context": "Other source context is fixed.",
        }
    )
    blueprint = compile_sampling_blueprint(
        row,
        {"document_id": "source", "bindings": []},
        contract,
        description_blocks={
            "source_sha256": digest(text.encode()),
            "fields": [
                {
                    "path": "documentPatch.goodsItemDetails[0].description",
                    "spans": [{"text": "STEEL BARS", "occurrence": 1}],
                }
            ],
        },
    )
    stream = DeterministicStream(13102026, "test", "sample")
    sampled = public_identity_updates(target, None, stream)
    values = host_variable_values(blueprint, sampled, stream)
    assert values["master_bill"] == sampled["documentPatch"]["masterBillOfLadingNumber"]
    rendered, gold, proof = render_sampling_blueprint(
        blueprint, sampled, {**values, "goods": "STEEL BARS"}
    )
    assert "ML250031" not in rendered
    assert rendered.count(values["master_bill"]) == 2
    assert gold["documentPatch"]["masterBillOfLadingNumber"] == values["master_bill"]
    assert len(proof["edits"]) == 2
