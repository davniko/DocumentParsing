"""Regression controls for source-owned field semantics and native wording aliases."""

from dataclasses import replace

import pytest
from pydantic import ValidationError
from test_curated_campaign import postal_campaign as postal_campaign

from document_ocr.synthesis.curated_campaign import (
    identity_review_context,
    validate_lexical_owners,
    wording_request,
)
from document_ocr.synthesis.curated_wording import (
    WordingField,
    WordingRequest,
    has_field_scope_caption,
    native_wording_prompt,
    rendered_review_prompt,
    unpack_wording,
    wording_api_keys,
    wording_output_type,
)


def test_native_semantic_aliases_map_back_to_unchanged_internal_keys():
    request = WordingRequest(
        "sample",
        "POSTAL ASSEMBLY shipper: {v002}. DESCRIPTION ASSEMBLY: {v010} {v011}",
        (
            WordingField("v001", "documentPatch.parties.shipper name", "OLD", "New identity"),
            WordingField("v002", "documentPatch.parties.shipper postal", "OLD ROAD", "New postal"),
            WordingField("v010", "goods description", "OLD PRODUCT", "New product"),
            WordingField("v011", "goods description", "OLD SPEC", "New spec"),
        ),
    )
    aliases = wording_api_keys(request)
    assert aliases == {
        "v001": "shipper_name_component_1",
        "v002": "shipper_postal_component_1",
        "v010": "goods_description_component_1",
        "v011": "goods_description_component_2",
    }
    schema = wording_output_type([request])
    properties = schema.model_json_schema()["$defs"]["Shipment0"]["properties"]
    assert set(properties) == set(aliases.values())
    values = ["NEW BUSINESS LTD", "12 NEW STREET", "COTTON YARN", "WHITE, GRADE A"]
    response = schema.model_validate({"s0": dict(zip(aliases.values(), values, strict=True))})
    result = unpack_wording(response, [request])["sample"]
    assert result == dict(zip(aliases, values, strict=True))
    prompt = native_wording_prompt([request])
    assert "{shipper_postal_component_1}" in prompt
    assert "{goods_description_component_1} {goods_description_component_2}" in prompt
    assert "{v002}" not in prompt
    with pytest.raises(ValidationError):
        schema.model_validate({"s0": {**response.s0.model_dump(by_alias=True), "other": "X"}})


def test_missing_native_field_cannot_pass():
    request = WordingRequest("s", "", (WordingField("v1", "shipper name", "OLD", "New"),))
    with pytest.raises(ValidationError):
        wording_output_type([request]).model_validate({"s0": {}})


def test_semantic_key_leak_is_rejected_but_model_identifier_is_allowed():
    request = WordingRequest("s", "", (WordingField("v1", "goods description", "OLD", "New"),))
    schema = wording_output_type([request])
    with pytest.raises(ValueError, match="requested field key"):
        unpack_wording(
            schema.model_validate({"s0": {"v1": "PRODUCT\nGOODS_DESCRIPTION_COMPONENT_1"}}),
            [request],
        )
    assert unpack_wording(
        schema.model_validate({"s0": {"v1": "MODEL GOODS_DESCRIPTION_COMPONENT_1 MOTOR"}}),
        [request],
    )


@pytest.mark.parametrize(
    "text",
    [
        "NEW LTD ATN: JOHN SMITH",
        "NEW LTD\nATTN JOHN SMITH",
        "NEW LTD ATTENTION: JOHN SMITH",
        "NEW LTD CONTACT: JOHN SMITH",
        "NEW LTD AGENTE: ROSA MARQUEZ",
        "NEW LTD MANAGING DIRECTOR JOHN SMITH",
        "NEW LTD (JOHN SMITH, MANAGING DIRECTOR)",
        "NEW LTD GERENTE GENERAL: JUAN PEREZ",
        "NEW LTD RIF J-40528716-3",
        "NEW LTD VAT: 12345678",
        "NEW LTD ON BEHALF OF ANOTHER LTD",
        "NEW LTD O/B ANOTHER LTD",
        "NEW LTD C/O ANOTHER LTD",
        "NEW LTD CARE OF ANOTHER LTD",
        "NEW LTD A/C ANOTHER LTD",
        "NEW LTD AS AGENT OF ANOTHER LTD",
    ],
)
def test_explicit_name_caption_holds_generation(text):
    field = WordingField("v1", "shipper name", "OLD", "New")
    request = WordingRequest("s", "", (field,))
    assert has_field_scope_caption(field, text)
    with pytest.raises(ValueError, match="caption"):
        unpack_wording(
            wording_output_type([request]).model_validate({"s0": {"v1": text}}), [request]
        )


@pytest.mark.parametrize(
    "text",
    [
        "ATLAS METALLURGIQUE DU RIF S.A.R.L.",
        "CONTACT INDUSTRIES LTD",
        "ADVANCED CONTACT TECHNOLOGIES LTD",
        "RIF TRADING LTD",
        "MILLBROOK TRADING PARK",
        "NEW COMPANY LTD\nJOHN SMITH",
        "AHMED ALSAYED GABR ALNASHAR",
    ],
)
def test_genuine_identity_words_and_uncaptioned_people_are_not_deleted(text):
    assert not has_field_scope_caption(WordingField("v1", "shipper name", "OLD", "New"), text)


@pytest.mark.parametrize("text", ["CO. KILDARE", "CO. TIPPERARY", "CANAL SHIPPING ST."])
def test_postal_site_and_county_controls_remain_valid(text):
    assert not has_field_scope_caption(WordingField("v1", "shipper postal", "OLD", "New"), text)


def test_product_caption_guard_preserves_product_contact_terms():
    field = WordingField("g", "goods description", "OLD", "New")
    assert has_field_scope_caption(field, "COTTON YARN. DELIVERY AGENT CONTACT: JOHN SMITH")
    assert not has_field_scope_caption(field, "ELECTRICAL CONTACT SWITCH, CONTACT RATING 20 A")


@pytest.mark.parametrize("kind", ["individual", "organization", "organization_with_proprietor"])
def test_source_certified_kind_reaches_native_description(postal_campaign, kind):
    bp = postal_campaign.blueprint
    data = {**bp.ownership_data, "identity_kinds": {"shipper_name": kind}}
    req = wording_request(replace(bp, ownership_data=data), postal_campaign.scenario, "s")
    field = next(f for f in req.fields if f.key == "shipper_name")
    assert field.identity_kind == kind
    alias = wording_api_keys(req)[field.key]
    description = wording_output_type([req]).model_json_schema()["$defs"]["Shipment0"][
        "properties"
    ][alias]["description"]
    assert {
        "individual": "Individual party",
        "organization": "Organization:",
        "organization_with_proprietor": "uncaptioned proprietor",
    }[kind] in description
    assert "Match the example's" not in description
    assert "An uncaptioned proprietor stays" not in description


@pytest.mark.parametrize(
    "declaration",
    [
        {"unknown": "individual"},
        {"product": "individual"},
        {"shipper_name": "guess"},
        {"shipper_name": []},
        ["shipper_name", "individual"],
    ],
)
def test_invalid_identity_certificate_fails_closed(postal_campaign, declaration):
    bp = replace(postal_campaign.blueprint, ownership_data={"identity_kinds": declaration})
    with pytest.raises(ValueError, match="identity kinds"):
        wording_request(bp, postal_campaign.scenario, "s")


def test_no_certificate_does_not_guess_identity_kind(postal_campaign):
    req = wording_request(postal_campaign.blueprint, postal_campaign.scenario, "s")
    assert all(field.identity_kind is None for field in req.fields)


def test_render_review_receives_certified_kind_and_actual_component(postal_campaign):
    bp = replace(
        postal_campaign.blueprint,
        ownership_data={"identity_kinds": {"shipper_name": "organization"}},
    )
    context = identity_review_context(bp, {"shipper_name": "JOHN SMITH\nNEW EXPORT LTD"})
    assert "documentPatch.parties.shipper.name (organization)" in context
    assert "only its legal or trading organization name" in context
    assert "JOHN SMITH\nNEW EXPORT LTD" in context
    assert "organization-only" in rendered_review_prompt("uppercase")
    with pytest.raises(ValueError, match="owned generated components"):
        identity_review_context(bp, {})
    assert identity_review_context(postal_campaign.blueprint, {}) == ""


@pytest.mark.parametrize("expression", ["OLD EXPORT BV", "{shipper_street}"])
def test_primary_name_requires_mutable_name_owner(postal_campaign, expression):
    bp = postal_campaign.blueprint
    targets = [
        binding.model_copy(update={"expression": expression})
        if binding.path == "documentPatch.parties.shipper.name"
        else binding
        for binding in bp.contract.targets
    ]
    bp = replace(bp, contract=bp.contract.model_copy(update={"targets": targets}))
    with pytest.raises(ValueError, match=r"shipper\.name"):
        validate_lexical_owners(bp)


def test_identity_coverage_ignores_route_names_and_contact_names(postal_campaign):
    from copy import deepcopy

    bp = postal_campaign.blueprint
    target = deepcopy(bp.target)
    target["documentPatch"]["route"] = {"portOfLoading": {"name": "ROTTERDAM"}}
    target["documentPatch"]["parties"]["shipper"]["contactName"] = "JOHN SMITH"
    validate_lexical_owners(replace(bp, target=target))


def test_native_output_requires_nonempty_unique_requests():
    request = WordingRequest("s", "", (WordingField("v1", "shipper name", "OLD", "New"),))
    for requests in ([], [request, request]):
        with pytest.raises(ValueError, match="unique shipments"):
            wording_output_type(requests)


def test_host_product_has_context_but_no_absent_output_instruction(postal_campaign):
    scenario = postal_campaign.scenario.model_copy(
        update={"provenance": {"physicalProfile": "source_whole_units"}}
    )
    req = wording_request(postal_campaign.blueprint, scenario, "s")
    assert not any(field.role == "goods description" for field in req.fields)
    assert req.description_groups == ()
    assert "Goods: UNROASTED COFFEE" in req.context
    assert "DESCRIPTION ASSEMBLY" not in req.context
    assert "SAMPLED COMMODITY BRIEF" not in req.context
    assert "{product}" not in native_wording_prompt([req])
    assert "POSTAL ASSEMBLY" in req.context


def test_reviewer_checks_semantic_ownership_not_only_string_equality():
    prompt = rendered_review_prompt("uppercase")
    assert "Check field meaning even when a generated value matches its label exactly" in prompt
    assert "Genuine named sites" in prompt
    assert "uncaptioned proprietors" in prompt
