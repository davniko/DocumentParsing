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


@pytest.mark.parametrize("connector", ["STC", "said to contain", "containing", "="])
def test_overpack_sentence_binds_inner_quantity_without_relabeling_either_level(connector):
    text = f"9 Pallet(s) {connector} 220 CARTONS"
    source = scenario([text], 220)
    target = scenario([text], 202)
    result = equations.generate(source, target)
    expected = f"9 Pallet(s) {connector} 202 CARTONS"
    assert list(result.values()) == [expected]
    target["documentPatch"]["cargoGroups"][0]["additionalInformation"] = [expected]
    package_prose.validate(source, target)
    for invalid in ["8 Pallet(s) " + connector + " 202 CARTONS", text]:
        target["documentPatch"]["cargoGroups"][0]["additionalInformation"] = [invalid]
        with pytest.raises(ValueError):
            package_prose.validate(source, target)
    target["documentPatch"]["cargoPackages"][0]["typeCategory"] = "PACKAGE_PACKAGE"
    with pytest.raises(ValueError, match="printed package-level categories"):
        equations.generate(source, target)


def test_overpack_sentence_requires_exact_structured_source_proof():
    source = scenario(["9 Pallet(s) STC 220 CARTONS"], 200)
    with pytest.raises(ValueError, match="lacks a structured total"):
        equations.generate(source, source)


def test_pre_sampling_categories_use_the_same_proven_nested_packing_owner():
    source = scenario(["9 Pallet(s) containing 220 CARTONS"], 220)
    before = deepcopy(source)
    assert equations.required_package_categories(source) == {0: "PACKAGE_CARTON"}
    assert source == before
    # The outer pallet level is not a carton category and is not invented as
    # another extraction package row.
    assert len(source["documentPatch"]["cargoPackages"]) == 1


def test_pre_sampling_categories_reject_false_source_equations():
    source = scenario(["9 Pallet(s) STC 220 CARTONS"], 200)
    with pytest.raises(ValueError, match="lacks a structured total"):
        equations.required_package_categories(source)


@pytest.mark.parametrize("new", [202, 424, 900])
def test_slash_overpack_and_pallet_marks_keep_distinct_level_ownership(new):
    source = scenario(["15PLTS/540CTNS"], 540)
    source["documentPatch"]["cargoGroups"][0]["marksAndNumbers"] = ["P/NO.:P1-P15"]
    target = deepcopy(source)
    target["documentPatch"]["cargoPackages"][0]["quantity"] = new
    before = deepcopy(source)
    assert list(equations.overpack_surfaces(source, target).values()) == [
        f"15PLTS/{new}CTNS",
        "P/NO.:P1-P15",
    ]
    assert source == before
    target["documentPatch"]["cargoPackages"][0]["typeCategory"] = "PACKAGE_PACKAGE"
    with pytest.raises(ValueError, match="printed package-level categories"):
        equations.overpack_surfaces(source, target)


def test_slash_overpack_can_bind_structured_outer_level_and_rejects_unproven_marks():
    source = scenario(["15PLTS/540CTNS"], 15)
    source["documentPatch"]["cargoPackages"][0]["typeCategory"] = "PACKAGE_PALLET"
    source["documentPatch"]["cargoGroups"][0]["marksAndNumbers"] = ["P/NO.:P1-P15"]
    target = deepcopy(source)
    target["documentPatch"]["cargoPackages"][0]["quantity"] = 12
    assert list(equations.overpack_surfaces(source, target).values()) == [
        "12PLTS/540CTNS",
        "P/NO.:P1-P12",
    ]
    source["documentPatch"]["cargoGroups"][0]["marksAndNumbers"] = ["P/NO.:P1-P14"]
    with pytest.raises(ValueError, match="unique declared overpack total"):
        equations.overpack_surfaces(source, target)


@pytest.mark.parametrize(
    ("text", "categories", "old", "new", "expected"),
    [
        ("1,280 BAGS (32 PALLETS)", ["BAG"], [1280], [849], "849 BAGS (32 PALLETS)"),
        (
            "64PLTS=896CTNS=17024PCES",
            ["CARTON", "PIECE"],
            [896, 17024],
            [594, 11287],
            "64PLTS=594CTNS=11287PCES",
        ),
        (
            "20 PKGS (12 IBC TANKS + 8 PLTS)",
            ["INTERMEDIATE_BULK_CONTAINER"],
            [12],
            [10],
            "18 PKGS (10 IBC TANKS + 8 PLTS)",
        ),
        (
            "25 Pallet(s): 25 PIECES SLAC: 2296 CASES",
            ["PIECE", "CASE"],
            [25, 2296],
            [22, 2010],
            "22 Pallet(s): 22 PIECES SLAC: 2010 CASES",
        ),
        ("158 PACKAGES ON 21 PALLETS", ["PALLET"], [21], [18], "158 PACKAGES ON 18 PALLETS"),
    ],
)
def test_packing_levels_bind_each_unit_without_collapsing_or_erasing_context(
    text, categories, old, new, expected
):
    source = scenario([text], 1)
    source["documentPatch"]["cargoPackages"] = [
        dict(groupId="g1", quantity=q, typeCategory="PACKAGE_" + c)
        for c, q in zip(categories, old, strict=True)
    ]
    target = deepcopy(source)
    for p, q in zip(target["documentPatch"]["cargoPackages"], new, strict=True):
        p["quantity"] = q
    assert list(equations.overpack_surfaces(source, source).values()) == [text]
    assert list(equations.overpack_surfaces(source, target).values()) == [expected]
    target["documentPatch"]["cargoPackages"][0]["typeCategory"] = "PACKAGE_PACKAGE"
    with pytest.raises(ValueError, match="printed package-level categories"):
        equations.overpack_surfaces(source, target)


def test_numbered_pallet_lots_prove_a_partition_of_the_outer_level():
    source = scenario(["56PLTS=1000CTNS=20000PCES"], 1000)
    source["documentPatch"]["cargoPackages"].append(
        dict(groupId="g1", quantity=20000, typeCategory="PACKAGE_PIECE")
    )
    marks = ["P/NO.:B82J1/10-10/10", "P/NO.:B82J1/46-46/46"]
    source["documentPatch"]["cargoGroups"][0]["marksAndNumbers"] = marks
    target = deepcopy(source)
    target["documentPatch"]["cargoPackages"][0]["quantity"] = 900
    target["documentPatch"]["cargoPackages"][1]["quantity"] = 18000
    assert list(equations.overpack_surfaces(source, target).values()) == [
        "56PLTS=900CTNS=18000PCES",
        *marks,
    ]
    source["documentPatch"]["cargoGroups"][0]["marksAndNumbers"] = marks[:1]
    with pytest.raises(ValueError, match="unique declared overpack total"):
        equations.overpack_surfaces(source, target)


def test_outer_number_words_and_existing_equation_partitions_use_the_same_solver():
    source = scenario(["ONE PACKAGE OF 48 PIECES"], 48)
    source["documentPatch"]["cargoPackages"][0]["typeCategory"] = "PACKAGE_PIECE"
    target = deepcopy(source)
    target["documentPatch"]["cargoPackages"][0]["quantity"] = 37
    assert list(equations.overpack_surfaces(source, target).values()) == [
        "ONE PACKAGE OF 37 PIECES"
    ]
    source = scenario(["16PLTS = 128 CTNS", "5PLTS = 39 CTNS", "21 PLTS = 167 CTNS"], 167)
    source["documentPatch"]["cargoGroups"][0]["marksAndNumbers"] = ["PALLET NO:1-16"]
    target = deepcopy(source)
    target["documentPatch"]["cargoPackages"][0]["quantity"] = 143
    assert list(equations.overpack_surfaces(source, target).values()) == [
        "16PLTS = 110 CTNS",
        "5PLTS = 33 CTNS",
        "21 PLTS = 143 CTNS",
        "PALLET NO:1-16",
    ]
    source = scenario(["18 PLTS= 132 CTNS"], 132)
    source["documentPatch"]["cargoGroups"][0]["marksAndNumbers"] = [
        "PALLET NO:1-16",
        "PALLET NO:1-2",
    ]
    target = deepcopy(source)
    target["documentPatch"]["cargoPackages"][0]["quantity"] = 120
    assert list(equations.overpack_surfaces(source, target).values()) == [
        "18 PLTS= 120 CTNS",
        "PALLET NO:1-16",
        "PALLET NO:1-2",
    ]


def test_numbered_pallet_lot_partition_updates_all_count_aliases():
    source = scenario(["15 PLTS/540 CTNS"], 15)
    source["documentPatch"]["cargoPackages"][0]["typeCategory"] = "PACKAGE_PALLET"
    source["documentPatch"]["cargoGroups"][0]["marksAndNumbers"] = [
        "P/NO.:ABC/5-5/5",
        "P/NO.:ABC/10~10/10",
    ]
    target = deepcopy(source)
    target["documentPatch"]["cargoPackages"][0]["quantity"] = 12
    assert list(equations.overpack_surfaces(source, target).values()) == [
        "12 PLTS/540 CTNS",
        "P/NO.:ABC/4-4/4",
        "P/NO.:ABC/8~8/8",
    ]


def test_overpack_counts_cannot_invert_containment_or_parse_arbitrary_product_text():
    source = scenario(["15PLTS/540CTNS"], 540)
    target = scenario(["15PLTS/540CTNS"], 10)
    with pytest.raises(ValueError, match="invert their source containment"):
        equations.overpack_surfaces(source, target)
    source = scenario(["15PLTS OF PRODUCT X / 540CTNS"], 540)
    assert not equations.overpack_surfaces(source, source)


def test_overpack_component_sum_requires_exact_source_arithmetic():
    source = scenario(["20 PKGS (12 IBC TANKS + 9 PLTS)"], 12)
    source["documentPatch"]["cargoPackages"][0]["typeCategory"] = (
        "PACKAGE_INTERMEDIATE_BULK_CONTAINER"
    )
    with pytest.raises(ValueError, match="fails source arithmetic"):
        equations.overpack_surfaces(source, source)


def test_printed_equation_cannot_silently_change_package_level_category():
    source = scenario(["21PKGS=16PLTS/606CTNS+5CTNS=611CTNS"], 611)
    target = deepcopy(source)
    target["documentPatch"]["cargoPackages"][0]["typeCategory"] = "PACKAGE_BOX"
    with pytest.raises(ValueError, match="printed package-level categories"):
        equations.generate(source, target)


def test_carton_lot_ranges_are_apportioned_not_each_replaced_with_the_total():
    source = scenario([], 611)
    source["documentPatch"]["cargoGroups"][0]["marksAndNumbers"] = [
        "C/NO.: 1-540",
        "PART: ABC",
        "C/NO.: 1-71",
    ]
    target = deepcopy(source)
    target["documentPatch"]["cargoPackages"][0]["quantity"] = 372
    result = equations.mark_ranges(source, target)
    assert list(result.values()) == ["C/NO.: 1-329", "C/NO.: 1-43"]
    assert list(equations.mark_ranges(source, source).values()) == ["C/NO.: 1-540", "C/NO.: 1-71"]
    source["documentPatch"]["cargoPackages"][0]["quantity"] = 600
    assert not equations.mark_ranges(source, target)


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


def test_bare_container_pallet_component_requires_review_before_numeric_scaling(monkeypatch):
    binding = NS(logical_key="container-pallets", occurrences=[NS(source_text="32", byte_end=2)])
    monkeypatch.setattr(numeric, "numeric_bindings", lambda _: [binding])
    source = scenario(["64PLTS=896CTNS=17024PCES"], 896)
    with pytest.raises(ValueError, match="component dependency"):
        equations.require_numeric_level_coverage(b"32 PALLETS 16000 KGS", NS(bindings=()), source)
    # The following column is a mass, not proof of a pallet-level dependency.
    equations.require_numeric_level_coverage(b"32 KGS", NS(bindings=()), source)
    binding.target_paths = ()
    binding.occurrences = [NS(source_text="32 PALLETS", byte_end=10)]
    with pytest.raises(ValueError, match="component lacks"):
        equations.require_numeric_level_coverage(b"32 PALLETS", NS(bindings=()), source)
    binding.occurrences = [NS(source_text="64 PALLETS", byte_end=10)]
    equations.require_numeric_level_coverage(b"64 PALLETS", NS(bindings=()), source)


@pytest.mark.parametrize("count,allocation", [(700, None), (2100, None), (700, 700), (700, 600)])
def test_package_subtotal_requires_its_own_typed_quantity(count, allocation):
    source = scenario([], 2100)
    source["documentPatch"]["cargoPackages"][0]["typeCategory"] = "PACKAGE_BAG"
    source["documentPatch"]["cargoAllocations"] = [{"packageQuantity": allocation}]
    binding = NS(
        logical_key="cargo-row",
        target_paths=(
            "documentPatch.cargoPackages[0].typeCategory",
            "documentPatch.cargoGroups[0].additionalInformation[0]",
        ),
        dependency_paths=(
            () if allocation is None else ("documentPatch.cargoAllocations[0].packageQuantity",)
        ),
        occurrences=[NS(source_text=f"{count} BAG(S) OF CLAY ON 10 PALLETS")],
    )
    template = NS(bindings=[binding])
    if count != 2100 and allocation != count:
        with pytest.raises(ValueError, match="subtotal lacks a typed allocation"):
            equations.require_numeric_level_coverage(b"", template, source)
    else:
        equations.require_numeric_level_coverage(b"", template, source)


@pytest.mark.parametrize(
    "before,after,count",
    [
        ("24PKGS=18PLTS/707CTNS+6CTNS=713CTNS", "24PKGS=18PLTS/465CTNS+6CTNS=471CTNS", 24),
        ("16PLTS/713CTNS", "16PLTS/471CTNS", 16),
        ("16PLTS=713CTNS=14260PCES", "16PLTS=471CTNS=14260PCES", 16),
    ],
)
def test_numeric_summary_is_derived_from_equation_not_independently_scaled(before, after, count):
    source = scenario([before], 713)
    target = scenario([after], 471)
    printed = "24 PACKAGES" if count == 24 else "16 PALLET"
    binding = NS(
        logical_key="summary",
        target_paths=(),
        value_kind="package",
        occurrences=[NS(source_text=printed, slot_id="a")],
    )
    contract = numeric.NumericContract(
        mode="source_scaled",
        role="cargo_quantity",
        source_value=str(count),
        target_paths=[],
        multiplier="1",
        divisor=1,
        reason="independent scalar",
    )
    prepared = numeric.prepare(
        [binding], {"summary": contract}, source_target=source, target=target, scale=Decimal("0.7")
    )
    assert prepared["summary"].value == str(count)
    assert prepared["summary"].contract.mode == "target_equation_count"
    assert numeric.render_prepared([binding], prepared, source_target=source, target=target) == {
        "summary": {"a": printed}
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


def test_fixed_country_locode_keeps_surface_but_canonicalizes_country_receipt():
    binding = NS(
        logical_key="country_code",
        value_kind="location",
        group_kind="customs",
        target_paths=(),
        occurrences=[NS(slot_id="s", source_text="IDJKT")],
    )
    result = r._render_entity_auxiliary(
        binding,
        entity_member=(NS(), NS(field="country_code")),
        country_code_style="unlocode",
        geography_constraint=r.EntityGeographyConstraint(
            fixed_country_code="ID", fixed_country_bindings=frozenset({"country_code"})
        ),
        stream=None,
        values=None,
        country_codes={"indonesia": "ID"},
    )
    assert result.replacements == {"s": "IDJKT"}
    assert result.canonical_value == "ID"


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


def test_repeated_equal_overpack_components_require_equal_new_quantities():
    source = {
        "documentPatch": {
            "cargoGroups": [{"groupId": "g1", "additionalInformation": ["40 BAGS ON 2 PALLETS"]}],
            "cargoPackages": [
                {"groupId": "g1", "quantity": 40, "typeCategory": "PACKAGE_BAG"},
                {"groupId": "g1", "quantity": 40, "typeCategory": "PACKAGE_BAG"},
            ],
        }
    }
    target = deepcopy(source)
    for row in target["documentPatch"]["cargoPackages"]:
        row["quantity"] = 30
    path = "documentPatch.cargoGroups[0].additionalInformation[0]"
    assert equations.overpack_surfaces(source, target)[path] == "30 BAGS ON 2 PALLETS"
    target["documentPatch"]["cargoPackages"][1]["quantity"] = 29
    with pytest.raises(ValueError, match="candidate package owners disagree"):
        equations.overpack_surfaces(source, target)


def test_derived_package_noun_preserves_meaning_and_rejects_mixed_owners():
    source = scenario([], 40)
    source["documentPatch"]["cargoPackages"] *= 2
    source["documentPatch"]["cargoPackages"][0]["typeCategory"] = "PACKAGE_BAG"
    target = deepcopy(source)
    for row in target["documentPatch"]["cargoPackages"]:
        row["typeCategory"] = "PACKAGE_CARTON"
    binding = NS(
        derivation="same_as_binding",
        value_kind="package",
        logical_key="summary_noun",
        dependency_paths=tuple(f"documentPatch.cargoPackages[{i}].typeCategory" for i in range(2)),
        occurrences=(NS(slot_id="slot", source_text="BAGS"),),
    )
    case = NS(source_target=source, target=target)
    output = r._render_one_derivation(
        binding=binding, case=case, outputs={}, bindings={}, country_codes={}
    )
    assert output.canonical_value == "PACKAGE_CARTON"
    assert output.replacements == {"slot": "CTNS"}
    target["documentPatch"]["cargoPackages"][1] = {"quantity": 40, "typeCategory": "PACKAGE_BAG"}
    with pytest.raises(ValueError, match="equal owned package categories"):
        r._render_one_derivation(
            binding=binding, case=case, outputs={}, bindings={}, country_codes={}
        )
    case.target = source
    binding.occurrences = (NS(slot_id="slot", source_text="DRUMS"),)
    with pytest.raises(ValueError, match="contradicts its source category"):
        r._render_one_derivation(
            binding=binding, case=case, outputs={}, bindings={}, country_codes={}
        )


def test_group_pallet_marks_do_not_inherit_document_wide_overpack_total():
    source = {
        "documentPatch": {
            "cargoGroups": [
                {
                    "groupId": "g1",
                    "additionalInformation": ["15 PLTS = 107 CTNS"],
                    "marksAndNumbers": ["PALLET NO:1-8"],
                },
                {"groupId": "g2"},
                {"groupId": "g3"},
            ],
            "cargoPackages": [
                {"groupId": f"g{i}", "quantity": q, "typeCategory": "PACKAGE_PALLET"}
                for i, q in enumerate((8, 6, 1), 1)
            ],
        }
    }
    target = deepcopy(source)
    for row, quantity in zip(target["documentPatch"]["cargoPackages"], (6, 4, 1), strict=True):
        row["quantity"] = quantity
    result = equations.overpack_surfaces(source, target)
    assert result["documentPatch.cargoGroups[0].additionalInformation[0]"] == "11 PLTS = 107 CTNS"
    assert result["documentPatch.cargoGroups[0].marksAndNumbers[0]"] == "PALLET NO:1-6"
