from copy import deepcopy
from decimal import Decimal
from types import SimpleNamespace as NS

import pytest
from pydantic import ValidationError

from document_ocr.synthesis.template_compiler import descendant as r
from document_ocr.synthesis.template_compiler import geographic_context, package_prose
from document_ocr.synthesis.template_compiler import numeric_auxiliary as numeric
from document_ocr.synthesis.template_compiler import package_equations as equations
from document_ocr.synthesis.template_compiler.complete_pipeline import _generation_schema
from document_ocr.synthesis.template_compiler.request_batches import lexical_payload


def scenario(texts, quantity):
    return {
        "documentPatch": {
            "cargoGroups": [{"groupId": "g1", "additionalInformation": texts}],
            "cargoPackages": [
                {"groupId": "g1", "quantity": quantity, "typeCategory": "PACKAGE_CARTON"}
            ],
        }
    }


@pytest.mark.parametrize("new", [30, 471, 713, 1800])
def test_mixed_packaging_retains_topology_and_solves_inner_sum(new):
    source = scenario(["24PKGS=18PLTS/707CTNS+6CTNS=713CTNS"], 713)
    target = deepcopy(source)
    target["documentPatch"]["cargoPackages"][0]["quantity"] = new
    expected = f"24PKGS=18PLTS/{new - 6}CTNS+6CTNS={new}CTNS"
    result = equations.generate(source, target)
    assert list(result.values()) == [expected]
    target["documentPatch"]["cargoGroups"][0]["additionalInformation"] = [expected]
    package_prose.validate(source, target)
    target["documentPatch"]["cargoGroups"][0]["additionalInformation"] = [
        f"16 PACKAGES={new} CARTONS"
    ]
    with pytest.raises(ValueError):
        package_prose.validate(source, target)


def test_partition_sum_is_exact_and_round_trips_source():
    source = scenario(["16PLTS = 128 CTNS", "5PLTS = 39 CTNS", "21 PLTS = 167 CTNS"], 167)
    assert set(equations.generate(source, source).values()) == set(
        source["documentPatch"]["cargoGroups"][0]["additionalInformation"]
    )
    target = deepcopy(source)
    target["documentPatch"]["cargoPackages"][0]["quantity"] = 143
    assert set(equations.generate(source, target).values()) == {
        "16PLTS = 110 CTNS",
        "5PLTS = 33 CTNS",
        "21 PLTS = 143 CTNS",
    }


def test_whole_document_outer_total_is_not_assumed_to_be_an_inner_total():
    source = scenario(["15 PLTS = 107 CTNS"], 8)
    source["documentPatch"]["cargoPackages"] = [
        {"groupId": f"g{i}", "quantity": q, "typeCategory": "PACKAGE_PALLET"}
        for i, q in enumerate([8, 6, 1], 1)
    ]
    target = deepcopy(source)
    target["documentPatch"]["cargoPackages"][0]["quantity"] = 6
    assert list(equations.generate(source, target).values()) == ["13 PLTS = 107 CTNS"]


@pytest.mark.parametrize("text", ["24PKGS=18PLTS/700CTNS+6CTNS=713CTNS", "10PLTS=800CTNS"])
def test_invalid_or_unbound_equations_fail_explicitly(text):
    source = scenario([text], 713)
    with pytest.raises(ValueError):
        equations.generate(source, source)


def test_order_identity_is_not_a_package_count_but_unlabelled_number_is():
    assert not package_prose._printed_package_claims("ORD.NO:202606148527 PALLET")
    assert not package_prose._printed_package_claims("PO NO:12345 PALLET")
    assert package_prose._printed_package_claims("12345 PALLET") == {("PACKAGE_PALLET", 12345)}
    assert package_prose._printed_package_claims("16PLTS") == {("PACKAGE_PALLET", 16)}


def test_numeric_summary_is_derived_from_equation_not_independently_scaled():
    source = scenario(["24PKGS=18PLTS/707CTNS+6CTNS=713CTNS"], 713)
    target = scenario(["24PKGS=18PLTS/465CTNS+6CTNS=471CTNS"], 471)
    binding = NS(
        logical_key="summary",
        target_paths=(),
        value_kind="package",
        occurrences=[NS(source_text="24 PACKAGES", slot_id="a")],
    )
    contract = numeric.NumericContract(
        mode="source_scaled",
        role="cargo_quantity",
        source_value="24",
        target_paths=[],
        multiplier="1",
        divisor=1,
        reason="independent scalar",
    )
    prepared = numeric.prepare(
        [binding], {"summary": contract}, source_target=source, target=target, scale=Decimal("0.7")
    )
    assert prepared["summary"].value == "24"
    assert prepared["summary"].contract.mode == "target_equation_count"
    assert numeric.render_prepared([binding], prepared, source_target=source, target=target) == {
        "summary": {"a": "24 PACKAGES"}
    }
    broken = {"summary": prepared["summary"].model_copy(update={"value": "17"})}
    with pytest.raises(ValueError, match="differs from its dependency"):
        numeric.render_prepared([binding], broken, source_target=source, target=target)


def test_immutable_registered_country_suffix_pins_only_its_owner():
    raw = b"Country: TAIWAN, PROVINCE OF CHINA\n"
    slot = NS(source_text="TAIWAN", byte_start=9, byte_end=15)
    binding = NS(occurrences=[slot])
    template = NS(bindings=[binding])
    countries = {"taiwan": "TW", "taiwanprovinceofchina": "TW", "china": "CN"}
    assert geographic_context.pinned_country(binding, template, raw, countries) == "TW"
    other = NS(occurrences=[NS(source_text="PROVINCE OF CHINA", byte_start=17, byte_end=34)])
    template.bindings.append(other)
    assert geographic_context.pinned_country(binding, template, raw, countries) is None
    assert (
        geographic_context._country_frame("Country: ", "Taiwan", "", tuple(countries.items()))
        is None
    )


def test_country_review_requires_explicit_component_not_locality_suffix():
    countries = {"china": "CN", "egypt": "EG", "jersey": "JE", "unitedstates": "US"}
    assert geographic_context._terminal_country("RAEA A3 BLOCK NO., 27 EGYPT.", countries) == "EG"
    assert geographic_context._terminal_country("NEW JERSEY", countries) is None
    assert geographic_context._terminal_country("125 ROAD, NEW JERSEY", countries) is None
    assert geographic_context._terminal_country("JERSEY", countries) == "JE"
    country = NS(logical_key="c", occurrences=[NS(source_text="CHINA")])
    address = NS(logical_key="a", occurrences=[NS(source_text="27 EGYPT")])
    entity = NS(
        entity_id="e",
        members=[NS(field="country", logical_key="c"), NS(field="address", logical_key="a")],
    )
    template = NS(bindings=[country, address], auxiliary_semantic_plan=NS(entities=[entity]))
    assert len(geographic_context.source_entity_conflicts(template, countries)) == 1
    template.auxiliary_semantic_plan.entities = [
        NS(entity_id="c", members=[entity.members[0]]),
        NS(entity_id="a", members=[entity.members[1]]),
    ]
    assert not geographic_context.source_entity_conflicts(template, countries)


def test_partial_country_preserves_source_fragment_under_proven_fixed_frame():
    binding = NS(
        logical_key="c",
        value_kind="country",
        target_paths=(),
        occurrences=[NS(slot_id="s", source_text="UNITED")],
    )
    entity = NS()
    member = NS(field="country")
    result = r._render_entity_auxiliary(
        binding,
        entity_member=(entity, member),
        country_code_style=None,
        geography_constraint=r.EntityGeographyConstraint(
            fixed_country_code="US", fixed_country_bindings=frozenset({"c"})
        ),
        stream=None,
        values=None,
        country_codes={"unitedstates": "US"},
    )
    assert result.replacements == {"s": "UNITED"}


def test_mutable_prompt_source_does_not_include_host_owned_frame():
    payload = {
        "structuredScenario": {"name": "FIXED OLD"},
        "requestedFields": [
            {
                "key": "field",
                "paths": ["name"],
                "source": "FIXED OLD",
                "constraints": [],
                "hostAssembly": {
                    "prefix": "FIXED ",
                    "suffix": "",
                    "mutableSource": "OLD",
                    "minimumWords": 1,
                },
            }
        ],
    }
    assert lexical_payload(payload, {"field": "name"})["requestedFields"][0]["source"] == "OLD"
    assert payload["requestedFields"][0]["source"] == "FIXED OLD"


def test_fragment_punctuation_is_native_validated():
    model = _generation_schema(
        [{"key": "part", "requiredBoundaryPunctuation": {"prefix": "(", "suffix": ")"}}], []
    )
    assert model.model_validate({"part": "(NEW PRODUCT)"}).part == "(NEW PRODUCT)"
    with pytest.raises(ValidationError):
        model.model_validate({"part": "NEW PRODUCT"})


def test_typographic_identity_does_not_equate_changed_measurement_signs():
    path = "documentPatch.instruction"
    binding = NS(
        target_paths=[path],
        realization=NS(target_values=[NS(target_path=path, source_value="SHIPPER\u2019S -18 C")]),
        occurrences=[NS(slot_id="s", source_text="SHIPPER\u2019S -18 C")],
    )
    assert (
        r._unchanged_target_output(binding, {"documentPatch": {"instruction": "SHIPPER'S -18 C"}})
        is not None
    )
    assert (
        r._unchanged_target_output(binding, {"documentPatch": {"instruction": "SHIPPER'S 18 C"}})
        is None
    )


def test_annotated_cargo_cover_preserves_item_lines_and_complete_names():
    paths = [
        "documentPatch.cargoGroups[0].description",
        "documentPatch.cargoGroups[0].additionalInformation[0]",
        "documentPatch.cargoGroups[0].additionalInformation[1]",
    ]
    source = scenario(["OLD A (12 IBC)", "OLD B (12 IBC)"], 24)
    source["documentPatch"]["cargoGroups"][0]["description"] = "OLD A OLD B"
    target = scenario(["NEW LONG PRODUCT A (12 IBC)", "NEW B (12 IBC)"], 24)
    target["documentPatch"]["cargoGroups"][0]["description"] = "NEW LONG PRODUCT A NEW B"
    b = NS(
        value_kind="cargo_text",
        target_paths=paths,
        occurrences=[NS(slot_id="s", source_text="OLD A (12 IBC)\nOLD B (12 IBC)")],
    )
    assert r._annotated_cargo_output(b, source, target).replacements == {
        "s": "NEW LONG PRODUCT A (12 IBC)\nNEW B (12 IBC)"
    }
    target["documentPatch"]["cargoGroups"][0]["description"] = "NEW B NEW LONG PRODUCT A"
    assert r._annotated_cargo_output(b, source, target) is None
