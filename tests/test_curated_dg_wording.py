"""DG commercial prose is model-owned; repeated regulatory identities are not."""

from copy import deepcopy
from dataclasses import replace

import pytest
from test_curated_campaign import postal_campaign as postal_campaign

from document_ocr.synthesis.curated_auxiliary import _render_recipe, _validate_recipe
from document_ocr.synthesis.curated_campaign import host_product_wording, wording_request
from document_ocr.synthesis.curated_dg_wording import dg_wording_context, proper_shipping_name
from document_ocr.synthesis.curated_ownership import ownership_surfaces
from document_ocr.synthesis.curated_wording import (
    WordingBatch,
    WordingField,
    WordingRequest,
    has_shipment_accounting,
    validate_wording,
)
from document_ocr.synthesis.generators import DeterministicStream


def dg_scenario(old, family="dg_chemical"):
    cargo = {
        **deepcopy(old.cargo),
        "goodsItemDetails": [{"dangerousGoods": [{"unNumber": "1175"}]}],
    }
    provenance = {
        **old.provenance,
        "capabilities": {"family": family},
        "physicalProfile": "source_whole_units",
        "dgPrintedFacts": {
            "properShippingName": "Ethylbenzene",
            "unNumber": "1175",
            "class": "3",
            "packingGroup": "II",
            "subsidiaryHazards": [],
        },
    }
    return old.model_copy(update={"cargo": cargo, "provenance": provenance})


@pytest.mark.parametrize("family", ["dg_chemical", "dg_vehicle"])
def test_dg_product_reaches_llm_even_for_whole_unit_load(postal_campaign, family):
    scenario = dg_scenario(postal_campaign.scenario, family)
    assert host_product_wording(postal_campaign.blueprint, scenario) == {}
    request = wording_request(postal_campaign.blueprint, scenario, "s", goods_only=True)
    assert [field.key for field in request.fields] == ["product"]
    assert request.description_groups == (("product",),)
    assert request.context.count("SAMPLED DG PRODUCT FACTS") == 1
    assert "properShippingName: Ethylbenzene" in request.context
    assert "only where the template prints them" in request.context
    # Only UN is printed. Registry context must not add class/PG to cargo.
    assert scenario.cargo["goodsItemDetails"][0]["dangerousGoods"] == [{"unNumber": "1175"}]


def test_regulatory_and_commercial_aliases_are_separate(postal_campaign):
    scenario = dg_scenario(postal_campaign.scenario)
    bp = replace(
        postal_campaign.blueprint,
        ownership_data={
            "surfaces": {
                "proper_shipping_name": {"dg_fact": "properShippingName"},
                "commercial_repeat": {"lexical_key": "product"},
            }
        },
    )
    values = {"product": "ETHYLBENZENE, TECHNICAL GRADE, PURITY 99.5% MIN"}
    surfaces = ownership_surfaces(bp, bp.target, scenario, lexical_values=values)
    assert surfaces["proper_shipping_name"] == "Ethylbenzene"
    assert surfaces["commercial_repeat"] == values["product"]
    recipe = {"dg_fact": "properShippingName", "prefix": "PSN: ", "suffix": "."}
    _validate_recipe(recipe)
    assert (
        _render_recipe(
            recipe, scenario, bp.target, DeterministicStream(1, "test", "s"), "PSN: OLD.", set()
        )
        == "PSN: Ethylbenzene."
    )


@pytest.mark.parametrize(
    "facts",
    [
        None,
        {},
        {"properShippingName": ""},
        {"properShippingName": "Ethylbenzene", "unNumber": "9999"},
    ],
)
def test_missing_or_mismatched_dg_facts_fail_not_fallback(postal_campaign, facts):
    scenario = dg_scenario(postal_campaign.scenario)
    scenario = scenario.model_copy(
        update={"provenance": {**scenario.provenance, "dgPrintedFacts": facts}}
    )
    for read in (proper_shipping_name, dg_wording_context):
        with pytest.raises(ValueError, match="matching its UN facts"):
            read(scenario)


@pytest.mark.parametrize(
    "recipe",
    [
        {"dg_fact": "class"},
        {"dg_fact": "properShippingName", "target": "x"},
        {"dg_fact": "properShippingName", "unreviewed": True},
    ],
)
def test_invalid_dg_recipes_rejected(recipe):
    with pytest.raises(ValueError):
        _validate_recipe(recipe)


@pytest.mark.parametrize(
    "text",
    [
        "CHEMICAL 25KG/BAG TOTAL 26MT",
        "PRODUCT TOTAL:26 MT",
        "SOLVENT TOTAL 23.75 TONNES",
        "PRODUCT TOTAL 1,500 KGS",
    ],
)
def test_explicit_mass_total_cannot_hide_inside_product_packing(text):
    assert has_shipment_accounting(text)


@pytest.mark.parametrize(
    "text",
    [
        "CHEMICAL 25KG/BAG",
        "160 FIBRE DRUMS X 85 KGS EACH",
        "TANK TOTAL CAPACITY 26MT",
        "MODEL TOTAL26MT",
    ],
)
def test_product_capacities_and_embedded_fills_remain_allowed(text):
    assert not has_shipment_accounting(text)


def test_dg_context_uses_actual_physical_plan_package_counts(postal_campaign):
    scenario = dg_scenario(postal_campaign.scenario)
    context = dg_wording_context(
        scenario,
        cargo={
            "goodsItemDetails": [
                {
                    "numberAndTypeOfPackages": [
                        {"packageQuantity": 78, "typeCategory": "PACKAGE_DRUM"}
                    ],
                    "netWeight": {"value": 99999, "unit": "kilogram"},
                }
            ]
        },
    )
    assert "packageQuantity: 78" in context and "99999" not in context


def test_stale_description_assembly_cannot_leak_marks_identifiers_to_model(postal_campaign):
    bp = postal_campaign.blueprint
    bindings = [
        binding.model_copy(update={"expression": binding.expression + "; CHASSIS NUMBER OLD123"})
        if binding.path.endswith(".description")
        else binding
        for binding in bp.contract.targets
    ]
    invalid = replace(bp, contract=bp.contract.model_copy(update={"targets": bindings}))
    with pytest.raises(ValueError, match="approved source block"):
        wording_request(invalid, dg_scenario(postal_campaign.scenario), "bad", goods_only=True)


@pytest.mark.parametrize(
    "caption", ["PACKING GROUP III", "IMO CLASS:3", "UN NUMBER:1175", "FLASH POINT12C"]
)
def test_generated_prose_does_not_invent_separately_owned_dg_declarations(caption):
    request = WordingRequest(
        "s", "DG context", (WordingField("g", "goods description", "OLD PRODUCT", "new"),)
    )
    output = WordingBatch.model_validate(
        {
            "shipments": [
                {"sample_id": "s", "values": [{"key": "g", "text": "ETHYLBENZENE, " + caption}]}
            ]
        }
    )
    with pytest.raises(ValueError, match="host-owned DG declaration"):
        validate_wording(output, [request])
