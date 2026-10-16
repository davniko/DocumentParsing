"""Approved classifier-free product guidance and archive/provenance boundaries."""

import json
from dataclasses import replace
from pathlib import Path

import pytest
from test_curated_campaign import postal_campaign as postal_campaign

from document_ocr.synthesis.curated_campaign import wording_request
from document_ocr.synthesis.curated_wording import (
    WORDING_PROMPT,
    goods_wording_requirement,
    wording_api_keys,
    wording_output_type,
)

ARCHIVE = Path(__file__).resolve().parents[1] / "docs/prompt-archives/goods-wording-20261015"


def test_original_and_first_polish_are_frozen_reference_archives():
    original = json.loads((ARCHIVE / "original.json").read_text())
    first = json.loads((ARCHIVE / "first-polish.json").read_text())
    assert "approximately the same total word count" not in original["systemPrompt"]
    assert "approximately the same total word count" in first["systemPrompt"]
    assert "{source_word_count}" in first["productFieldRequirementTemplate"]
    assert first["systemPrompt"] == WORDING_PROMPT
    assert original["sourceFiles"] == first["sourceFiles"]


@pytest.mark.parametrize("count", [1, 9, 55, 132, 1000])
def test_product_requirement_preserves_soft_count_details_and_context_boundary(count):
    first = json.loads((ARCHIVE / "first-polish.json").read_text())
    requirement = goods_wording_requirement(count)
    assert requirement.startswith(
        first["productFieldRequirementTemplate"].format(source_word_count=count)
    )
    assert "analogous attached specification" in requirement
    assert "Per-unit capacity or fill is part of that pattern" in requirement
    assert "omit that count from model-owned wording" in requirement
    assert "host prints declared shipment counts separately" in requirement
    assert "HS/tariff declarations, container/equipment or routing" in requirement
    assert "retaining analogous attached specifications and per-unit packing" in requirement
    assert "approximately" in requirement
    assert "classifier" not in requirement


@pytest.mark.parametrize("count", [0, -1, 1.5, True, "10"])
def test_invalid_source_count_cannot_silently_default(count):
    with pytest.raises(ValueError, match="positive integer"):
        goods_wording_requirement(count)


def test_request_counts_product_examples_once_not_party_or_repeated_occurrences(postal_campaign):
    bp = postal_campaign.blueprint
    product = next(v for v in bp.contract.variables if v.kind == "product")
    repeated = product.model_copy(update={"occurrences": [*product.occurrences] * 2})
    tail = product.model_copy(
        update={
            "key": "product_tail",
            "value": "GRADE A",
            "occurrences": [product.occurrences[0].model_copy(update={"text": "GRADE A"})],
        }
    )
    variables = [repeated if v.key == product.key else v for v in bp.contract.variables]
    contract = bp.contract.model_copy(
        update={
            "variables": [*variables, tail],
            "targets": [
                binding.model_copy(update={"expression": "{" + product.key + "} {product_tail}"})
                if binding.path.endswith(".description")
                else binding
                for binding in bp.contract.targets
            ],
        }
    )
    request = wording_request(replace(bp, contract=contract), postal_campaign.scenario, "s")
    count = len(product.occurrences[0].text.split()) + 2
    products = [field for field in request.fields if field.role == "goods description"]
    assert len(products) == 2
    assert "SOURCE DESCRIPTION ASSEMBLY" in request.context
    assert (
        "presentation example; fixed text is replaced by host, not requested wording"
        in request.context
    )
    assert all(field.requirement == goods_wording_requirement(count) for field in products)
    assert all(
        "source description contains" not in field.requirement
        for field in request.fields
        if field.role != "goods description"
    )
    properties = wording_output_type([request]).model_json_schema()["$defs"]["Shipment0"][
        "properties"
    ]
    aliases = wording_api_keys(request)
    for field in products:
        assert f"approximately {count} words" in properties[aliases[field.key]]["description"]
        assert "maxLength" not in properties[aliases[field.key]]


@pytest.mark.parametrize("profile", ["source_whole_units", "reviewed_whole_units"])
def test_host_owned_products_do_not_receive_lexical_guidance(postal_campaign, monkeypatch, profile):
    old = postal_campaign.scenario
    provenance = {**old.provenance}
    provenance["physicalProfile"] = profile
    scenario = old.model_copy(update={"provenance": provenance})

    def forbidden(_count):
        raise AssertionError("host-owned product reached lexical policy")

    monkeypatch.setattr(
        "document_ocr.synthesis.curated_campaign.goods_wording_requirement", forbidden
    )
    request = wording_request(postal_campaign.blueprint, scenario, "s", goods_only=True)
    assert not request.fields and not request.description_groups
