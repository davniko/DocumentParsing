"""Detached commodity references are linguistic surfaces, never extra labels."""

from __future__ import annotations

from copy import deepcopy
from dataclasses import replace
from types import SimpleNamespace

import pytest
from test_curated_campaign import postal_campaign as postal_campaign
from test_curated_templates import build_owned_blueprint, source_fixture

from document_ocr.synthesis.curated import TargetBinding
from document_ocr.synthesis.curated_campaign import validate_lexical_owners, wording_request
from document_ocr.synthesis.curated_ownership import ownership_surfaces
from document_ocr.synthesis.curated_product_references import (
    product_reference_baselines,
    product_reference_fields,
    product_reference_surface,
    product_references,
)
from document_ocr.synthesis.curated_templates import render_sampling_blueprint
from document_ocr.synthesis.curated_wording import (
    native_wording_prompt,
    unpack_wording,
    wording_api_keys,
    wording_output_type,
)

KEY = "reviewed:commodity_mark"


def reference_fixture():
    row, contract, history = source_fixture()
    row["joinedRawText"] += "MARKS: STEEL BOLTS\n"
    declaration = {
        "add": [
            {
                "key": KEY,
                "kind": "other_text",
                "paths": [],
                "occurrences": [{"text": "STEEL BOLTS", "occurrence": 2}],
            }
        ],
        "surfaces": {KEY: {"product_reference": {"goods_index": 0}}},
    }
    return build_owned_blueprint(row, history, contract, declaration)


def test_same_joint_request_has_short_linked_field_without_description_target(postal_campaign):
    fixture = postal_campaign
    source = reference_fixture()
    source = replace(
        source,
        ownership_data={**source.ownership_data, "identifier_privacy": {}},
    )
    request = wording_request(source, fixture.scenario, "short-reference")
    reference = next(field for field in request.fields if field.key == KEY)
    assert reference.role == "goods reference"
    assert "SAME current product" in reference.requirement
    assert "goods_description_component_1" in reference.requirement
    assert "{product}" not in reference.requirement
    assert "No packaging, package material, quantities" in reference.requirement
    assert "no extraction target" in reference.requirement
    assert KEY not in {v.key for v in source.contract.variables}
    assert all("{" + KEY + "}" not in binding.expression for binding in source.contract.targets)
    assert request.description_groups == (("product",),)
    prompt = native_wording_prompt([request])
    assert "{goods_description_component_1}" in prompt
    assert "goods_reference_component_1" in prompt
    aliases = wording_api_keys(request)
    values = {
        field.key: "CERAMIC TILES"
        if field.role in {"goods description", "goods reference"}
        else "NEW TRADING LTD"
        if field.role.endswith(" name")
        else "1 MAIN STREET UTRECHT NETHERLANDS"
        for field in request.fields
    }
    output = wording_output_type([request]).model_validate(
        {"s0": {aliases[key]: value for key, value in values.items()}}
    )
    assert unpack_wording(output, [request])["short-reference"][KEY] == "CERAMIC TILES"


def test_reference_is_short_independent_surface_not_full_product_copy():
    blueprint = reference_fixture()
    target = deepcopy(blueprint.target)
    target["documentPatch"]["goodsItemDetails"][0]["description"] = (
        "CERAMIC TILES, GLAZED, WHITE; PACKED IN CARTONS"
    )
    values = {
        "product": "CERAMIC TILES, GLAZED, WHITE; PACKED IN CARTONS",
        KEY: "CERAMIC TILES",
    }
    scenario = SimpleNamespace(
        party_localities={"documentPatch.parties.shipper": SimpleNamespace(country_code="IN")}
    )
    surfaces = ownership_surfaces(blueprint, target, scenario, lexical_values=values)
    text, actual_target, proof = render_sampling_blueprint(
        blueprint, target, {"product": values["product"]}, surface_values=surfaces
    )
    assert "MARKS: CERAMIC TILES\n" in text
    assert "MARKS: CERAMIC TILES, GLAZED" not in text
    assert actual_target == target
    assert proof["unchangedBytesPreserved"]
    assert all(KEY not in path for path in proof["coveredTargetPaths"])
    assert product_reference_baselines(blueprint) == {KEY: "STEEL BOLTS"}
    assert (
        ownership_surfaces(
            blueprint,
            blueprint.target,
            scenario,
            lexical_values=product_reference_baselines(blueprint),
        )[KEY]
        == "STEEL BOLTS"
    )


@pytest.mark.parametrize("value", [None, "", "   ", "CERAMIC TILES\nPACKING:25KG", "TILES\rBOXES"])
def test_missing_or_block_reference_cannot_render(value):
    blueprint = reference_fixture()
    with pytest.raises(ValueError, match="product reference"):
        product_reference_surface(blueprint, KEY, None if value is None else {KEY: value})


@pytest.mark.parametrize(
    "link",
    [
        None,
        {},
        {"goods_index": -1},
        {"goods_index": True},
        {"goods_index": "0"},
        {"goods_index": 1},
        {"goods_index": 0, "fallback": "source"},
    ],
)
def test_missing_or_ambiguous_link_fails_before_calls(link):
    blueprint = reference_fixture()
    ownership = deepcopy(blueprint.ownership_data)
    ownership["surfaces"][KEY] = {"product_reference": link}
    with pytest.raises(ValueError, match="goods_index"):
        product_references(replace(blueprint, ownership_data=ownership))


def test_public_or_partial_owner_and_duplicate_description_link_fail_closed():
    blueprint = reference_fixture()
    owner = next(r for r in blueprint.regions if r.key == KEY)
    public = replace(owner, target_paths=("documentPatch.goodsItemDetails[0].description",))
    with pytest.raises(ValueError, match="source-only"):
        product_references(
            replace(
                blueprint, regions=tuple(public if r.key == KEY else r for r in blueprint.regions)
            )
        )
    with pytest.raises(ValueError, match="source-only"):
        product_references(
            replace(blueprint, regions=tuple(r for r in blueprint.regions if r.key != KEY))
        )
    duplicate = TargetBinding(
        path="documentPatch.goodsItemDetails[0].description", expression="{product}"
    )
    contract = blueprint.contract.model_copy(
        update={"targets": [*blueprint.contract.targets, duplicate]}
    )
    with pytest.raises(ValueError, match="unique current goods description"):
        product_references(replace(blueprint, contract=contract))


def test_no_reference_metadata_is_exact_noop_and_host_description_is_explicit():
    blueprint = reference_fixture()
    ordinary = replace(blueprint, ownership_data={})
    assert product_references(ordinary) == ()
    assert product_reference_fields(ordinary, description_fields=(), host_products={}) == ()
    assert product_reference_baselines(ordinary) == {}
    fields = product_reference_fields(
        blueprint, description_fields=(), host_products={"product": "COFFEE BEANS"}
    )
    assert "'COFFEE BEANS'" in fields[0].requirement
    assert "{product}" not in fields[0].requirement
    validate_lexical_owners(blueprint)


def test_reference_cannot_link_to_an_unrequested_non_host_product():
    with pytest.raises(ValueError, match="requested main product field"):
        product_reference_fields(reference_fixture(), description_fields=(), host_products={})
