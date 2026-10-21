from dataclasses import replace
from types import SimpleNamespace

from test_curated_campaign import (
    _country_component_postal_fixture,
)
from test_curated_campaign import (
    postal_campaign as postal_campaign,
)

from document_ocr.synthesis.curated_campaign import wording_request
from document_ocr.synthesis.curated_wording import native_wording_prompt, wording_output_type


def fields(request):
    return {field.key: field for field in request.fields}


def test_explicit_complete_country_owner_requests_country_name_only(postal_campaign):
    fixture = postal_campaign
    original = wording_request(fixture.blueprint, fixture.scenario, "sample")
    blueprint, _ = _country_component_postal_fixture(fixture, "NETHERLANDS")
    request = wording_request(blueprint, fixture.scenario, "sample")
    country = fields(request)["shipper_country"]
    assert country.requirement.startswith("Country name only, not a complete postal address.")
    assert "Output only NETHERLANDS" in country.requirement
    assert "Do not include a city/locality" in country.requirement
    assert "Certified country owner: NETHERLANDS" in country.requirement
    assert "approximate line span" not in country.requirement
    assert "Do not output 'NETHERLANDS'" not in country.requirement
    separate_country_hint = (
        " A separately owned component in this address assembly supplies the country. "
        "Do not output 'NETHERLANDS' or 'NL' as a country declaration in this component; "
        "do not append a country line or suffix. Return only this component, not a complete "
        "independent address. The required locality spelling is unaffected."
    )
    for key, field in fields(original).items():
        expected = (
            replace(field, requirement=field.requirement + separate_country_hint)
            if key in {"shipper_street", "shipper_locality"}
            else field
        )
        assert fields(request)[key] == expected
    assert country.requirement in native_wording_prompt([request])
    schema = wording_output_type([request]).model_json_schema()
    assert country.requirement in str(schema)


def test_postal_component_containing_country_is_not_a_country_only_owner(postal_campaign):
    fixture = postal_campaign
    role = "documentPatch.parties.consignee"
    blueprint = replace(
        fixture.blueprint,
        regions=(
            SimpleNamespace(
                kind="postal",
                curated_key="consignee_address",
                target_paths=(role + ".addressLine", role + ".country"),
            ),
        ),
    )
    requirement = fields(wording_request(blueprint, fixture.scenario, "sample"))[
        "consignee_address"
    ].requirement
    assert requirement.startswith("One postal component")
    assert "Country name only" not in requirement


def test_country_assembled_from_multiple_components_is_not_reclassified(postal_campaign):
    fixture = postal_campaign
    blueprint, _ = _country_component_postal_fixture(fixture, "NETHERLANDS")
    contract = blueprint.contract.model_copy(deep=True)
    next(
        binding for binding in contract.targets if binding.path.endswith(".country")
    ).expression = "{shipper_country} {shipper_locality}"
    request = wording_request(replace(blueprint, contract=contract), fixture.scenario, "sample")
    assert fields(request)["shipper_country"].requirement.startswith("One postal component")
    assert "Country name only" not in fields(request)["shipper_country"].requirement
