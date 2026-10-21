"""Typed care-of identities are not postal geography or mutable postal text."""

import json
from copy import deepcopy
from dataclasses import replace
from pathlib import Path

import pytest
from test_curated_campaign import postal_campaign as postal_campaign

from document_ocr.synthesis.curated import SourceContract, interpolate
from document_ocr.synthesis.curated_campaign import (
    postal_assembly_context,
    postal_wording_errors,
)
from document_ocr.synthesis.curated_party_geography import (
    postal_field_geography_requirement,
    postal_geography_expression,
    validate_postal_component_identities,
)
from document_ocr.synthesis.curated_scenarios import ShipmentScenario
from document_ocr.synthesis.curated_templates import SamplingBlueprint
from document_ocr.synthesis.curated_wording import validate_postal_geography


@pytest.fixture
def contract():
    return SourceContract.model_validate(
        {
            "variables": [
                {
                    "key": key,
                    "kind": kind,
                    "value": value,
                    "meaning": kind,
                    "required_literals": [],
                    "occurrences": [{"text": value, "occurrence": 1, "presentation": "text"}],
                }
                for key, kind, value in (
                    ("related", "name", "OLD FREIGHT LTD"),
                    ("postal", "postal", "OLD ROAD ROCHA"),
                    ("other_name", "name", "OTHER COMPANY LTD"),
                )
            ],
            "targets": [
                {
                    "path": "documentPatch.parties.shipper.addressLine",
                    "expression": "C/O {related} {postal}",
                },
                {"path": "documentPatch.parties.shipper.name", "expression": "{other_name}"},
            ],
            "fixed_context": "Explicitly typed related identity and postal component.",
        }
    )


def check(contract, expression, values, *, country_labelled=False):
    validate_postal_component_identities(
        contract, expression, values, locality="ROCHA", country="URUGUAY"
    )
    validate_postal_geography(
        interpolate(postal_geography_expression(contract, expression), values),
        locality="ROCHA",
        country="URUGUAY",
        country_code="UY",
        country_labelled=country_labelled,
    )


def test_country_inside_certified_careof_identity_does_not_declare_country(contract):
    values = {
        "related": "Transfleet Uruguay Ltda.",
        "postal": "Ruta 9 km 142, Establecimiento El Ceibo\nROCHA",
    }
    original = deepcopy(values)
    expression = "C/O {related} {postal}"
    with pytest.raises(ValueError, match="country was not requested"):
        validate_postal_geography(
            interpolate(expression, values),
            locality="ROCHA",
            country="URUGUAY",
            country_code="UY",
            country_labelled=False,
        )
    check(contract, expression, values)
    assert values == original
    assert interpolate(expression, values).startswith("C/O Transfleet Uruguay Ltda.")


@pytest.mark.parametrize(
    "postal",
    [
        "C/O TRANSFLEET URUGUAY LTDA.\nRUTA 9 KM 137\nROCHA",
        "TRANSFLEET URUGUAY\nLTDA.\nRUTA 9 KM 137\nROCHA",
        "RUTA 9 KM 137\nCARE OF Transfleet Uruguay Ltda.\nROCHA",
        "C/O TRANSFLEET URUGUAY LTDA\nRUTA 9 KM 137\nROCHA",
    ],
)
def test_certified_identity_cannot_be_copied_into_postal_slot(contract, postal):
    with pytest.raises(ValueError, match=r"postal.*repeats separately owned identity.*related"):
        check(
            contract,
            "C/O {related} {postal}",
            {
                "related": "Transfleet Uruguay Ltda.",
                "postal": postal,
            },
        )


@pytest.mark.parametrize("country", ["URUGUAY", "UY"])
def test_real_country_is_not_masked_with_related_identity(contract, country):
    with pytest.raises(ValueError, match="country was not requested"):
        check(
            contract,
            "C/O {related} {postal}",
            {
                "related": "Transfleet Uruguay Ltda.",
                "postal": f"RUTA 9 KM 142\nROCHA\n{country}",
            },
        )


def test_host_country_duplicate_is_still_rejected_and_values_unchanged(contract):
    values = {"related": "Transfleet Uruguay Ltda.", "postal": "RUTA 9 KM 142\nROCHA\nURUGUAY"}
    original = deepcopy(values)
    with pytest.raises(ValueError, match="country must occur once"):
        check(contract, "C/O {related} {postal} URUGUAY", values, country_labelled=True)
    assert values == original
    check(contract, "C/O {related} {postal}", values, country_labelled=True)


def test_protected_company_name_does_not_satisfy_locality(contract):
    with pytest.raises(ValueError, match="locality must occur once"):
        check(
            contract,
            "C/O {related} {postal}",
            {
                "related": "ROCHA FREIGHT LTD",
                "postal": "RUTA 9 KM 142",
            },
        )


def test_only_explicit_name_slots_are_excluded_not_arbitrary_names_or_literals(contract):
    assert postal_geography_expression(contract, "C/O {related} {postal} URUGUAY") == (
        "C/O   {postal} URUGUAY"
    )
    assert postal_geography_expression(contract, "C/O TRANSFLEET URUGUAY LTDA {postal}") == (
        "C/O TRANSFLEET URUGUAY LTDA {postal}"
    )
    assert postal_geography_expression(contract, "{postal}") == "{postal}"
    with pytest.raises(KeyError):
        postal_geography_expression(contract, "{unowned}")


def test_careof_or_named_building_without_separately_owned_duplicate_is_not_classified(contract):
    values = {"postal": "C/O OTHER COMPANY LTD\nOTHER COMPANY LTD BUILDING\nRUTA 9\nROCHA"}
    check(contract, "{postal}", values)


def test_country_named_identity_does_not_reject_genuine_geography(contract):
    values = {"related": "URUGUAY", "postal": "RUTA 9\nROCHA\nURUGUAY"}
    check(contract, "C/O {related} {postal}", values, country_labelled=True)


def test_different_company_and_identity_substring_are_not_guessed_duplicate(contract):
    values = {
        "related": "Transfleet Uruguay Ltda.",
        "postal": "TRANSFLEET ROAD\nRUTA 9\nROCHA",
    }
    check(contract, "C/O {related} {postal}", values)


def test_missing_identity_value_is_not_silently_ignored(contract):
    with pytest.raises(KeyError):
        check(contract, "C/O {related} {postal}", {"postal": "RUTA 9\nROCHA"})


def actual_case(prefix, sample):
    root = Path(__file__).resolve().parents[1]
    case = next(
        (root / "artifacts/synthesis-templates/mpci-bl-v7-reviewed/cases").glob(f"doc_{prefix}*")
    )
    receipt = json.loads(
        (
            root / "docs/analysis/r28-fifty-fresh-flat-probe-20261019/"
            f"campaign/run/wording/{sample}.json"
        ).read_bytes()
    )
    contract = SourceContract.model_validate(
        json.loads((case / "contract.json").read_bytes())["contract"]
    )
    blueprint = SamplingBlueprint(
        receipt["sourceDocumentId"],
        (case / "source.txt").read_text(),
        json.loads((case / "target.json").read_bytes()),
        contract,
        (),
        {},
        {},
        (),
    )
    return blueprint, ShipmentScenario.model_validate(receipt["scenario"]), receipt


def test_actual_h09_initial_careof_passes_and_copy_and_country_are_diagnosed():
    blueprint, scenario, receipt = actual_case("f766e12f", "syn_full_v7_3129dd68406e9ab386517765")
    initial = {**receipt["values"], **receipt["corrections"][0]["before"]}
    assert postal_wording_errors(blueprint, scenario, receipt["targetBeforeWording"], initial) == {}
    role = "documentPatch.parties.shipper"
    copied = postal_wording_errors(
        blueprint, scenario, receipt["targetBeforeWording"], receipt["values"]
    )
    assert copied == {
        role: [
            "generated postal component v014 repeats separately owned "
            "identity component v015_related"
        ]
    }
    real_country = {**initial, "v014": initial["v014"] + "\nURUGUAY"}
    assert postal_wording_errors(
        blueprint, scenario, receipt["targetBeforeWording"], real_country
    ) == {role: ["generated country was not requested for this party"]}
    context = postal_assembly_context(
        blueprint, scenario, receipt["targetBeforeWording"], {"v014"}, fixed_values=initial
    )
    assert "SEPARATELY OWNED IDENTITY {v015_related}: Transfleet Uruguay Ltda." in context
    assert "never copy into postal components" in context
    assert "COUNTRY: omitted from this party's source; keep that omission." in context
    assert "country is already supplied by a host-owned literal" not in context


def test_actual_h17_host_owned_suffix_duplicate_requires_scoped_correction():
    blueprint, scenario, receipt = actual_case("6018901a", "syn_full_v7_584859c172879408e10893c9")
    role = "documentPatch.parties.deliveryAgent"
    assert postal_wording_errors(
        blueprint, scenario, receipt["targetBeforeWording"], receipt["values"]
    ) == {role: ["generated postal country must occur once as its supplied name or code"]}
    context = postal_assembly_context(
        blueprint,
        scenario,
        receipt["targetBeforeWording"],
        {"delivery_postal"},
        fixed_values=receipt["values"],
    )
    assert "country is already supplied by a host-owned literal" in context
    assert "keep generated regions free of that country name/code" in context
    clean = dict(receipt["values"])
    clean["delivery_postal"] = clean["delivery_postal"].removesuffix("\nTANZANIA")
    assert postal_wording_errors(blueprint, scenario, receipt["targetBeforeWording"], clean) == {}


def test_fixed_contract_component_uses_same_certified_value_as_assembly(postal_campaign):
    fixture = postal_campaign
    blueprint = deepcopy(fixture.blueprint)
    contract = blueprint.contract.model_copy(deep=True)
    fixed = contract.variables[0].model_copy(
        update={"key": "fixed_country", "kind": "postal", "value": "NETHERLANDS"}
    )
    contract.variables.append(fixed)
    next(
        b for b in contract.targets if b.path == "documentPatch.parties.shipper.addressLine"
    ).expression = "{shipper_street} {shipper_locality} {fixed_country}"
    blueprint = replace(blueprint, contract=contract)
    values = {
        **fixture.receipt["values"],
        "shipper_street": "14 CANAL ROAD",
        "shipper_locality": "ROTTERDAM",
    }
    assert "fixed_country" not in values
    assert postal_wording_errors(blueprint, fixture.scenario, fixture.target, values) == {}


def test_actual_h17_field_excludes_host_country_without_automatic_text_change():
    blueprint, scenario, receipt = actual_case("6018901a", "syn_full_v7_584859c172879408e10893c9")
    role = "documentPatch.parties.deliveryAgent"
    from document_ocr.synthesis.curated_ownership import lexical_expression

    binding = next(b for b in blueprint.contract.targets if b.path == role + ".addressLine")
    expression = lexical_expression(blueprint, binding, scenario)
    location = scenario.party_localities[role]
    original = deepcopy(receipt["values"])
    hint = postal_field_geography_requirement(
        blueprint.contract,
        expression,
        "delivery_postal",
        locality=location.name,
        country=location.country,
        country_code=location.country_code,
        country_labelled=True,
    )
    assert "unchanged host-owned address literal already supplies" in hint
    assert "Do not output 'TANZANIA' or 'TZ'" in hint
    assert receipt["values"] == original


def test_country_only_owner_gets_no_exclusion_but_other_component_does(contract):
    common = dict(
        locality="ROCHA",
        country="URUGUAY",
        country_code="UY",
        country_labelled=True,
        country_owner_keys={"postal"},
    )
    assert postal_field_geography_requirement(contract, "{postal}", "postal", **common) == ""
    contract = contract.model_copy(deep=True)
    contract.variables.append(contract.variables[1].model_copy(update={"key": "street"}))
    hint = postal_field_geography_requirement(contract, "{street} {postal}", "street", **common)
    assert "separately owned component" in hint
    assert "Do not output 'URUGUAY' or 'UY'" in hint


@pytest.mark.parametrize("locality", ["ROCHA", "URUGUAY", "URUGUAY CITY"])
def test_omitted_country_hint_preserves_overlapping_required_locality(contract, locality):
    hint = postal_field_geography_requirement(
        contract,
        "C/O {related} {postal}",
        "postal",
        locality=locality,
        country="URUGUAY",
        country_code="UY",
        country_labelled=False,
    )
    assert "omits a postal-country declaration" in hint
    assert "Keep the required locality once" in hint
    assert "contains or equals the country name" in hint


def test_identity_country_word_does_not_create_host_country_exclusion(contract):
    assert (
        postal_field_geography_requirement(
            contract,
            "C/O {related} {postal}",
            "postal",
            locality="ROCHA",
            country="URUGUAY",
            country_code="UY",
            country_labelled=True,
        )
        == ""
    )


def test_field_country_requirement_recognizes_name_code_frame_and_no_unrelated_field(contract):
    hint = postal_field_geography_requirement(
        contract,
        "{postal} URUGUAY(UY)",
        "postal",
        locality="ROCHA",
        country="URUGUAY",
        country_code="UY",
        country_labelled=True,
    )
    assert "unchanged host-owned address literal" in hint
    assert (
        postal_field_geography_requirement(
            contract,
            "{related}",
            "postal",
            locality="ROCHA",
            country="URUGUAY",
            country_code="UY",
            country_labelled=True,
        )
        == ""
    )
    with pytest.raises(ValueError, match="certified postal field"):
        postal_field_geography_requirement(
            contract,
            "{related}",
            "related",
            locality="ROCHA",
            country="URUGUAY",
            country_code="UY",
            country_labelled=True,
        )
