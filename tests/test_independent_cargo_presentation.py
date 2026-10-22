"""A reviewed vehicle presentation is a registry domain, never a cargo donor."""

from copy import deepcopy

import pytest

from document_ocr.synthesis.independent_cargo_presentation import (
    CargoPresentation,
    validate_cargo_presentation,
)
from document_ocr.synthesis.independent_packages import (
    PackageChoiceRequest,
    package_choice_output_type,
    package_choice_prompt,
    validate_package_choice,
)

DOMAIN = {
    "hsHeadings": ["8701", "8702", "8703", "8704", "8705", "8711", "8716"],
    "excludedHs6": ["871690"],
    "packageTypes": ["PACKAGE_UNPACKED_OR_UNPACKAGED"],
}


def test_whole_vehicle_domain_excludes_parts_and_unrelated_goods():
    scope = CargoPresentation.model_validate(DOMAIN)
    assert scope.permits("870321") and scope.permits("871639")
    assert not scope.permits("871690") and not scope.permits("853921")


@pytest.mark.parametrize(
    "field,value",
    [
        ("hsHeadings", []),
        ("hsHeadings", ["870", "８７０１"]),  # noqa: RUF001 - reject non-ASCII digits
        ("hsHeadings", ["8701", "8701"]),
        ("excludedHs6", ["840690"]),
        ("packageTypes", []),
        ("packageTypes", ["BOGUS"]),
    ],
)
def test_malformed_review_domains_fail_closed(field, value):
    data = deepcopy(DOMAIN)
    data[field] = value
    with pytest.raises(ValueError):
        CargoPresentation.model_validate(data)


def test_frozen_request_must_match_review_and_selected_domain():
    request = {
        "cargoPresentation": DOMAIN,
        "goods": [{"hs6": "870321"}],
        "selectedCategory": "PACKAGE_UNPACKED_OR_UNPACKAGED",
    }
    validate_cargo_presentation(DOMAIN, request)
    with pytest.raises(ValueError, match="source authority"):
        validate_cargo_presentation(None, request)
    request["selectedCategory"] = "PACKAGE_BOX"
    with pytest.raises(ValueError, match="package violates"):
        validate_cargo_presentation(DOMAIN, request)
    request.pop("selectedCategory")
    request["goods"][0]["hs6"] = "853921"
    with pytest.raises(ValueError, match="goods violate"):
        validate_cargo_presentation(DOMAIN, request)


def package_request():
    return {
        "goods": [{"hs6": "870321", "description": "Motor cars"}],
        "regime": "ambient",
        "equipment": "Ro-ro shipment",
        "requiredCategory": None,
    }


def test_default_request_serialization_and_prompt_remain_unchanged():
    request = PackageChoiceRequest.model_validate(package_request())
    assert "allowedPackageTypes" not in request.model_dump(mode="json")
    assert "Reviewed source presentation" not in package_choice_prompt(request, hs_codes={"870321"})


def test_native_choices_are_scoped_without_inventing_an_alternative():
    request = {**package_request(), "allowedPackageTypes": DOMAIN["packageTypes"]}
    prompt = package_choice_prompt(request, hs_codes={"870321"})
    assert "Reviewed source presentation" in prompt
    output = {
        "alternatives": [
            {"packageType": "PACKAGE_UNPACKED_OR_UNPACKAGED", "declaredUnitContext": "vehicle"}
        ],
        "goodsForm": "Motor cars",
        "candidateCompatible": None,
    }
    assert not validate_package_choice(request, output).unknown
    output["alternatives"] = []
    assert validate_package_choice(request, output).unknown
    output["alternatives"] = [{"packageType": "PACKAGE_CARTON", "declaredUnitContext": None}]
    with pytest.raises(ValueError):
        validate_package_choice(request, output)
    schema = package_choice_output_type(request).model_json_schema()
    assert schema["properties"]["alternatives"]["maxItems"] == 1
    assert "PACKAGE_CARTON" not in str(schema)
    assert "PACKAGE_UNPACKED_OR_UNPACKAGED" in str(schema)


def test_candidate_outside_presentation_scope_is_rejected_before_provider_call():
    with pytest.raises(ValueError, match="candidate is outside"):
        PackageChoiceRequest.model_validate(
            {
                **package_request(),
                "requiredCategory": "PACKAGE_BALE",
                "allowedPackageTypes": DOMAIN["packageTypes"],
            }
        )
