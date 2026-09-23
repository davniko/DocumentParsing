from copy import deepcopy

import pytest

from document_ocr.synthesis.template_compiler import cargo_scenarios
from document_ocr.synthesis.template_compiler import package_observations as observations


def target(description=None, category=None):
    package = {"groupId": "g1", "packageId": "p1", "quantity": 12}
    if description is not None:
        package["typeDescription"] = description
    if category is not None:
        package["typeCategory"] = category
    return {"documentPatch": {"cargoPackages": [package]}}


def test_printed_description_is_fit_support_not_a_guessed_schema_category():
    source = target("BALES")
    fitted = observations.fit_target(source)
    assert fitted["documentPatch"]["cargoPackages"][0]["typeCategory"] == "printed-package:bales"
    assert "typeCategory" not in source["documentPatch"]["cargoPackages"][0]
    fitted["documentPatch"]["cargoPackages"][0]["quantity"] = 18
    projected = observations.observable_target(source, fitted)
    assert projected["documentPatch"]["cargoPackages"][0] == {
        "groupId": "g1",
        "packageId": "p1",
        "quantity": 18,
        "typeDescription": "BALES",
    }


def test_domains_distinguish_missing_untyped_and_typed_observations():
    packages = [
        target()["documentPatch"]["cargoPackages"][0],
        target("BALES")["documentPatch"]["cargoPackages"][0],
        target(category="PACKAGE_BOX")["documentPatch"]["cargoPackages"][0],
    ]
    domain = observations.category_domains(
        packages, frozenset({"printed-package:bales", "PACKAGE_BOX", "PACKAGE_CARTON"})
    )
    assert domain == (
        None,
        frozenset({"printed-package:bales"}),
        frozenset({"PACKAGE_BOX", "PACKAGE_CARTON"}),
    )
    assert observations.validate_signature(
        ("PACKAGE_BOX", "printed-package:bales", "PACKAGE_CARTON"), domain
    )
    assert not observations.validate_signature(
        ("PACKAGE_BOX", "PACKAGE_BOX", "PACKAGE_CARTON"), domain
    )
    assert not observations.validate_signature(
        ("PACKAGE_BOX", "printed-package:bales", "printed-package:bales"), domain
    )


@pytest.mark.parametrize(
    "noun",
    [
        "Bale",
        "Bundle",
        "Set",
        "Pack",
        "Crate",
        "Drum",
        "Roll",
        "Bag",
        "Carton",
        "Pallet",
        "Piece",
        "Unit",
    ],
)
def test_grammatical_number_shares_fit_support_without_changing_printed_labels(noun):
    assert observations.printed_signature(noun) == observations.printed_signature(noun + "s")
    assert observations.printed_signature(noun) == observations.printed_signature(noun + "(s)")
    original = target(noun + "(s)")
    assert observations.observable_target(original, observations.fit_target(original)) == original


def test_abbreviations_and_compound_packages_are_not_guessed():
    assert observations.printed_signature("PL") != observations.printed_signature("PALLET")
    assert observations.printed_signature("BALE OF SETS") != observations.printed_signature("BALES")


@pytest.mark.parametrize("description", ["CRT", "Crate", "Crate(s)", "CRATES"])
def test_reviewed_package_codes_share_support_without_adding_training_labels(description):
    original = target(description)
    fitted = observations.fit_target(original)
    assert fitted["documentPatch"]["cargoPackages"][0]["typeCategory"] == "PACKAGE_CRATE"
    assert observations.observable_target(original, fitted) == original
    domains = observations.category_domains(
        original["documentPatch"]["cargoPackages"], frozenset({"PACKAGE_CRATE", "PACKAGE_BOX"})
    )
    assert domains == (frozenset({"PACKAGE_CRATE"}),)


@pytest.mark.parametrize(
    "mutation", ["wrong_hidden_category", "changed_description", "missing_signature", "new_row"]
)
def test_inconsistent_package_observations_are_errors(mutation):
    source = target("BALES")
    sampled = observations.fit_target(source)
    package = sampled["documentPatch"]["cargoPackages"][0]
    if mutation == "wrong_hidden_category":
        package["typeCategory"] = "PACKAGE_CARTON"
    elif mutation == "changed_description":
        package["typeDescription"] = "CARTONS"
    elif mutation == "missing_signature":
        del package["typeCategory"]
    else:
        sampled["documentPatch"]["cargoPackages"].append(deepcopy(package))
    with pytest.raises(ValueError):
        observations.observable_target(source, sampled)


def test_linguistic_generation_cannot_expose_hidden_package_labels():
    expected = target("BALES")
    scenario = cargo_scenarios.CargoScenario(expected, {}, (), {})
    invented = observations.fit_target(expected)
    with pytest.raises(ValueError, match="package label visibility"):
        cargo_scenarios.validate_structured_facts(scenario, invented)
    changed = target("CARTONS")
    with pytest.raises(ValueError, match="sampled cargo fact changed"):
        cargo_scenarios.validate_structured_facts(scenario, changed)
