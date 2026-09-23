from copy import deepcopy
from decimal import Decimal
from types import SimpleNamespace as NS

import pytest

from document_ocr.synthesis.template_compiler import measurement_prose, package_prose


@pytest.mark.parametrize(
    "text,quantity,total",
    [
        ("25 KG EACH", 1000, 25000),
        ("25 KGS NET WEIGHT EACH", 1000, 25000),
        ("64 DRUMS PER 50 KGS. EACH ON 8 PALLETS", 64, 3200),
        ("DRIED PP BAG 25KG", 1000, 25000),
        ("PP BAGS OF 25 KG", 1000, 25000),
        ("1000 PAPER BAGS (25 KG) ON 20 PALLETS", 1000, 25000),
        ("50 KGS BALE PACKING", 478, 23900),
        ("IN 225KG NEW STEEL DRUMS PALLETIZED", 224, 50400),
        ("PACKING : 480 x 25 kg Fiber Drums", 480, 12000),
        ("PACKING : 480 x 25 kg Fibre Drums", 480, 12000),
        ("CHAN. DRUM (STEEL) 215KG", 16, 3440),
        ("PACKING IN 25 KG NET WEIGHT BAGS", 1000, 25000),
        ("25KG PP BAGS", 1000, 25000),
        ("1000KG X 20 IN 1 CONTAINER ONLY", 20, 20000),
        ("3KG \u00d7 4 BOX", 1675, 20100),
        ("20 KG", 1250, 25000),
        ("CHICK PEAS 44/46C CHICKPEAS 25KGS WT", 4000, 100000),
    ],
)
def test_unit_mass_contracts_preserve_unit_weight_and_solve_group_total(text, quantity, total):
    source = {
        "documentPatch": {
            "cargoGroups": [
                {
                    "groupId": "g1",
                    "additionalInformation": [text],
                    "netWeight": {"value": total, "unit": "kilogram"},
                }
            ],
            "cargoPackages": [{"groupId": "g1", "quantity": quantity}],
        }
    }
    target = deepcopy(source)
    target["documentPatch"]["cargoPackages"][0]["quantity"] = quantity - 1
    values = measurement_prose.per_package_totals(source, target)
    expected = Decimal(total) / quantity * (quantity - 1)
    assert values == {"documentPatch.cargoGroups[0].netWeight.value": expected}
    with pytest.raises(ValueError, match="contradicts its proved unit"):
        measurement_prose.generate(source, target)
    target["documentPatch"]["cargoGroups"][0]["netWeight"]["value"] = float(expected)
    updates = measurement_prose.generate(source, target)
    target["documentPatch"]["cargoGroups"][0]["additionalInformation"][0] = updates[
        "documentPatch.cargoGroups[0].additionalInformation[0]"
    ]
    measurement_prose.validate(source, target)
    if "X 20" in text:
        assert "X 19" in target["documentPatch"]["cargoGroups"][0]["additionalInformation"][0]
        target["documentPatch"]["cargoGroups"][0]["additionalInformation"][0] = text
        with pytest.raises(ValueError, match="package-mass product"):
            measurement_prose.validate(source, target)


def test_unit_mass_never_invents_a_missing_total_or_guesses_an_ambiguous_owner():
    source = {
        "documentPatch": {
            "cargoGroups": [{"groupId": "g1", "description": "25 KG EACH"}],
            "cargoPackages": [{"groupId": "g1", "quantity": 20}],
        }
    }
    with pytest.raises(ValueError, match="unique source quantity/total proof"):
        measurement_prose.generate(source, source)


def private_unit_source(text="PACKED IN BAGS WITH 25 KGS NET WEIGHT EACH"):
    return {
        "documentPatch": {
            "cargoGroups": [
                {
                    "groupId": "g1",
                    "additionalInformation": [text],
                    "grossWeight": {"value": 27000, "unit": "kilogram"},
                }
            ],
            "cargoPackages": [{"groupId": "g1", "quantity": 1000, "typeCategory": "PACKAGE_BAG"}],
        }
    }


def test_explicit_unit_net_is_private_when_total_is_not_printed_as_a_label():
    source = private_unit_source()
    target = deepcopy(source)
    target["documentPatch"]["cargoPackages"][0]["quantity"] = 800
    target["documentPatch"]["cargoGroups"][0]["grossWeight"]["value"] = 21600
    original = deepcopy(target)
    assert measurement_prose.per_package_totals(source, target) == {}
    assert measurement_prose.private_net_weights(source, target) == {"g1": Decimal(20000)}
    measurement_prose.validate(source, target)
    assert target == original
    assert "netWeight" not in target["documentPatch"]["cargoGroups"][0]
    physical = measurement_prose.physical_group(
        target["documentPatch"]["cargoGroups"][0], Decimal(20000)
    )
    assert physical["netWeight"] == {"unit": "kilogram", "value": 20000.0}


@pytest.mark.parametrize("gross", [19999, 0])
def test_private_net_is_not_ignored_when_generated_gross_is_too_small(gross):
    source = private_unit_source()
    target = deepcopy(source)
    target["documentPatch"]["cargoPackages"][0]["quantity"] = 800
    target["documentPatch"]["cargoGroups"][0]["grossWeight"]["value"] = gross
    with pytest.raises(ValueError, match="net mass exceeds"):
        measurement_prose.validate(source, target)


@pytest.mark.parametrize(
    "mutation", ["wrong_kind", "wrong_unit", "conflicting_net", "multiple_packages"]
)
def test_private_unit_mass_rejects_ambiguous_or_contradictory_observations(mutation):
    source = private_unit_source()
    target = deepcopy(source)
    if mutation == "wrong_kind":
        target["documentPatch"]["cargoPackages"][0]["typeCategory"] = "PACKAGE_DRUM"
    elif mutation == "wrong_unit":
        source["documentPatch"]["cargoGroups"][0]["grossWeight"]["unit"] = "pound"
    elif mutation == "conflicting_net":
        source["documentPatch"]["cargoGroups"][0]["additionalInformation"].append("24 KG NET EACH")
    else:
        source["documentPatch"]["cargoPackages"].append(
            dict(source["documentPatch"]["cargoPackages"][0])
        )
    with pytest.raises(ValueError):
        measurement_prose.private_net_weights(source, target)


def test_private_unit_mass_does_not_reinterpret_explicit_net_as_gross():
    source = private_unit_source()
    source["documentPatch"]["cargoGroups"][0]["grossWeight"]["value"] = 25000
    assert measurement_prose.per_package_totals(source, source) == {}
    assert measurement_prose.private_net_weights(source, source) == {"g1": Decimal(25000)}


def test_private_unit_mass_does_not_add_a_total_without_any_weight_label():
    source = private_unit_source()
    del source["documentPatch"]["cargoGroups"][0]["grossWeight"]
    assert measurement_prose.private_net_weights(source, source) == {"g1": Decimal(25000)}
    assert measurement_prose.generate(source, source) == {
        "documentPatch.cargoGroups[0].additionalInformation[0]": (
            "PACKED IN BAGS WITH 25 KGS NET WEIGHT EACH"
        )
    }


def test_paired_net_and_gross_weights_share_the_explicit_per_box_scope():
    text = "CARTON BOXES TYPE 22XU, WITH 19.50 KG NET AND 21.92 KG GROSS PER BOX."
    source = {
        "documentPatch": {
            "cargoGroups": [
                {
                    "groupId": "g1",
                    "additionalInformation": [text],
                    "netWeight": {"value": 23400, "unit": "kilogram"},
                    "grossWeight": {"value": 26304, "unit": "kilogram"},
                }
            ],
            "cargoPackages": [{"groupId": "g1", "quantity": 1200}],
        }
    }
    target = deepcopy(source)
    target["documentPatch"]["cargoPackages"][0]["quantity"] = 1000
    assert measurement_prose.per_package_totals(source, target) == {
        "documentPatch.cargoGroups[0].netWeight.value": Decimal("19500"),
        "documentPatch.cargoGroups[0].grossWeight.value": Decimal("21920"),
    }
    for field, value in (("netWeight", 19500), ("grossWeight", 21920)):
        target["documentPatch"]["cargoGroups"][0][field]["value"] = value
    assert measurement_prose.generate(source, target) == {
        "documentPatch.cargoGroups[0].additionalInformation[0]": text
    }


def test_paired_unit_weights_cannot_swap_net_and_gross_owners():
    source = {
        "documentPatch": {
            "cargoGroups": [
                {
                    "groupId": "g1",
                    "description": "20 KG NET AND 25 KG GROSS PER BOX",
                    "netWeight": {"value": 250, "unit": "kilogram"},
                    "grossWeight": {"value": 200, "unit": "kilogram"},
                }
            ],
            "cargoPackages": [{"groupId": "g1", "quantity": 10}],
        }
    }
    with pytest.raises(ValueError, match="unique source quantity/total proof"):
        measurement_prose.generate(source, source)


def test_standalone_total_is_not_reinterpreted_as_per_package_mass():
    source = {
        "documentPatch": {
            "cargoGroups": [
                {
                    "groupId": "g1",
                    "additionalInformation": ["200 KG"],
                    "netWeight": {"value": 200, "unit": "kilogram"},
                }
            ],
            "cargoPackages": [{"groupId": "g1", "quantity": 10}],
        }
    }
    target = deepcopy(source)
    target["documentPatch"]["cargoGroups"][0]["netWeight"]["value"] = 300
    assert measurement_prose.per_package_totals(source, target) == {}
    assert measurement_prose.generate(source, target) == {
        "documentPatch.cargoGroups[0].additionalInformation[0]": "300 KG"
    }


def test_product_weight_suffix_without_packing_evidence_is_not_a_unit_contract():
    source = {
        "documentPatch": {
            "cargoGroups": [
                {
                    "groupId": "g1",
                    "description": "CENTEX 4060 B 10 KG FLOUR",
                    "netWeight": {"value": 1000, "unit": "kilogram"},
                }
            ],
            "cargoPackages": [{"groupId": "g1", "quantity": 100}],
        }
    }
    assert measurement_prose.per_package_totals(source, source) == {}
    with pytest.raises(ValueError, match="unique total/component contract"):
        measurement_prose.generate(source, source)


def test_space_grouped_mass_components_keep_grouping_and_conserve_total():
    source = {
        "documentPatch": {
            "cargoGroups": [
                {
                    "groupId": "g1",
                    "additionalInformation": ["BATCH A = 4 000 KG - BATCH B = 16 000 KG"],
                    "netWeight": {"value": 20000, "unit": "kilogram"},
                }
            ]
        }
    }
    target = deepcopy(source)
    target["documentPatch"]["cargoGroups"][0]["netWeight"]["value"] = 10000
    assert (
        measurement_prose.generate(source, target)[
            "documentPatch.cargoGroups[0].additionalInformation[0]"
        ]
        == "BATCH A = 2 000 KG - BATCH B = 8 000 KG"
    )


def test_source_proven_counts_update_all_levels_and_reject_stale_text():
    source = {
        "documentPatch": {
            "cargoGroups": [{"groupId": "g1", "additionalInformation": ["160 ROLLS (80 PACKS)"]}],
            "cargoPackages": [
                {"groupId": "g1", "quantity": 160, "typeCategory": "PACKAGE_ROLL"},
                {"groupId": "g1", "quantity": 80, "typeDescription": "PACKS"},
            ],
        }
    }
    target = deepcopy(source)
    target["documentPatch"]["cargoPackages"][0]["quantity"] = 147
    target["documentPatch"]["cargoPackages"][1]["quantity"] = 74
    path = "documentPatch.cargoGroups[0].additionalInformation[0]"
    updates, formal = package_prose.generate(NS(coherence_constraints=[]), source, target)
    assert updates == {path: "147 ROLLS (74 PACKS)"}
    assert formal == {path}
    with pytest.raises(ValueError, match="contradicts"):
        package_prose.validate(source, target)
    target["documentPatch"]["cargoGroups"][0]["additionalInformation"] = [updates[path]]
    package_prose.validate(source, target)


def test_lexical_repair_exposes_each_nested_quantity_without_changing_target():
    from document_ocr.synthesis.template_compiler.complete_targets import (
        lexical_repair_requirements,
    )

    path = "documentPatch.cargoGroups[0].additionalInformation[0]"
    text = "160 ROLLS (80 PACKS)"
    source = {
        "documentPatch": {
            "cargoGroups": [{"groupId": "g1", "additionalInformation": [text]}],
            "cargoPackages": [
                {"groupId": "g1", "quantity": 160, "typeCategory": "PACKAGE_ROLL"},
                {"groupId": "g1", "quantity": 80, "typeDescription": "PACKS"},
            ],
        }
    }
    target = deepcopy(source)
    target["documentPatch"]["cargoPackages"][0]["quantity"] = 100
    target["documentPatch"]["cargoPackages"][1]["quantity"] = 50
    before = deepcopy(target)
    requirements = lexical_repair_requirements(
        NS(target=source, template=NS(coherence_constraints=(), bindings=())),
        target,
        [
            {"key": "field", "paths": [path], "source": text, "constraints": []},
        ],
    )
    facts = [
        fact for row in requirements[0]["obligations"] for fact in row.get("quantityFacts", [])
    ]
    assert facts == [
        {"path": "documentPatch.cargoPackages[0].quantity", "value": 100},
        {"path": "documentPatch.cargoPackages[1].quantity", "value": 50},
    ]
    assert target == before


def test_ambiguous_equal_source_counts_cannot_be_given_distinct_new_values():
    source = {
        "documentPatch": {
            "cargoGroups": [{"groupId": "g1", "description": "100 CARTONS OF NAPKINS"}],
            "cargoPackages": [{"groupId": "g1", "quantity": 100, "typeCategory": "PACKAGE_CARTON"}]
            * 2,
        }
    }
    target = deepcopy(source)
    target["documentPatch"]["cargoPackages"] = [
        {**source["documentPatch"]["cargoPackages"][0], "quantity": value} for value in (60, 70)
    ]
    with pytest.raises(ValueError, match="ambiguous quantity ownership"):
        package_prose.generate(NS(coherence_constraints=[]), source, target)


def test_range_repair_does_not_demand_new_package_annotations():
    from document_ocr.synthesis.template_compiler.complete_targets import (
        lexical_repair_requirements,
    )
    from document_ocr.synthesis.template_compiler.models import AggregateRangeConstraint

    path = "documentPatch.cargoGroups[0].marksAndNumbers[0]"
    dependency = "documentPatch.cargoPackages[0].quantity"
    constraint = AggregateRangeConstraint(
        constraint_id="coherence_constraint_" + "a" * 16,
        kind="aggregate_inclusive_range_cardinality",
        candidate_fingerprint="a" * 64,
        member_logical_keys=("mark",),
        dependency_paths=(dependency,),
        rationale="source proof",
    )
    target = {
        "documentPatch": {
            "cargoGroups": [{"groupId": "g1", "marksAndNumbers": ["BRAND 1-458 500-957"]}],
            "cargoPackages": [{"groupId": "g1", "quantity": 916, "typeCategory": "PACKAGE_CARTON"}],
        }
    }
    source = NS(
        target=target,
        template=NS(
            coherence_constraints=[constraint],
            bindings=[
                NS(
                    logical_key="mark",
                    target_paths=[path],
                    realization=NS(mode="single_surface"),
                    derivation=None,
                    value_kind="other_text",
                )
            ],
        ),
    )
    rows = lexical_repair_requirements(
        source, target, [dict(key="f", paths=[path], source="BRAND 1-458 500-957", constraints=[])]
    )
    obligations = rows[0]["obligations"]
    rule = next(row["requirement"] for row in obligations if "coherenceContract" in row)
    assert "Do not append package-count annotations" in rule
    assert any("sourceRole" in row for row in obligations)


def test_first_group_summary_tracks_source_proven_document_wide_package_sum():
    path = "documentPatch.cargoGroups[0].additionalInformation[0]"
    source = {
        "documentPatch": {
            "cargoGroups": [{"groupId": "g1", "additionalInformation": ["15 PLTS = 107 CTNS"]}],
            "cargoPackages": [
                {"groupId": group, "quantity": count, "typeCategory": "PACKAGE_PALLET"}
                for group, count in (("g1", 8), ("g2", 6), ("g3", 1))
            ],
        }
    }
    target = deepcopy(source)
    for package, quantity in zip(target["documentPatch"]["cargoPackages"], (7, 5, 1), strict=True):
        package["quantity"] = quantity
    updates, _ = package_prose.generate(NS(coherence_constraints=[]), source, target)
    assert updates[path] == "13 PLTS = 107 CTNS"
    with pytest.raises(ValueError, match="complete package total 13"):
        package_prose.validate(source, target)
    target["documentPatch"]["cargoGroups"][0]["additionalInformation"][0] = updates[path]
    package_prose.validate(source, target)


@pytest.mark.parametrize("claim", ["ELEVEN CARTONS", "11 CARTONS", "TWENTY-ONE CARTONS"])
def test_new_unbacked_package_counts_are_rejected_even_when_written_as_words(claim):
    source = {
        "documentPatch": {
            "cargoGroups": [{"groupId": "g1", "description": "SUBSEA OILWELL SUPPLIES"}],
            "cargoPackages": [{"groupId": "g1", "quantity": 12, "typeCategory": "PACKAGE_PIECE"}],
        }
    }
    target = deepcopy(source)
    target["documentPatch"]["cargoGroups"][0]["description"] = claim + " OF NEW EQUIPMENT"
    with pytest.raises(ValueError, match="new package claim"):
        package_prose.validate(source, target)
    target["documentPatch"]["cargoGroups"][0]["description"] = "TWELVE CARTONS OF NEW EQUIPMENT"
    target["documentPatch"]["cargoPackages"][0]["typeCategory"] = "PACKAGE_CARTON"
    package_prose.validate(source, target)
    target["documentPatch"]["cargoPackages"][0]["typeCategory"] = "PACKAGE_BOX"
    package_prose.validate(source, target)
    target["documentPatch"]["cargoPackages"][0]["typeCategory"] = "PACKAGE_PACKAGE"
    package_prose.validate(source, target)
    target["documentPatch"]["cargoPackages"][0]["quantity"] = 13
    with pytest.raises(ValueError, match="new package claim"):
        package_prose.validate(source, target)


def test_component_masses_remain_an_exact_total_without_changing_product_dimensions():
    group = {
        "groupId": "g1",
        "additionalInformation": [
            "15.2 MM / 267.986 MT \u2013 88 Coils",
            "15.7 MM / 145.424 MT \u2013 48 Coils",
        ],
        "netWeight": {"value": 413410, "unit": "kilogram"},
    }
    source = {"documentPatch": {"cargoGroups": [group]}}
    target = deepcopy(source)
    target["documentPatch"]["cargoGroups"][0]["netWeight"]["value"] = 249700
    updates = measurement_prose.generate(source, target)
    assert len(updates) == 2
    import re
    from decimal import Decimal

    assert sum(Decimal(re.search(r"/ (\S+) MT", s)[1]) for s in updates.values()) == Decimal(
        "249.700"
    )
    assert next(iter(updates.values())).startswith("15.2 MM /")
    with pytest.raises(ValueError, match="contradicts"):
        measurement_prose.validate(source, target)
    target["documentPatch"]["cargoGroups"][0]["additionalInformation"] = list(updates.values())
    measurement_prose.validate(source, target)


def test_uncontracted_mass_cannot_be_copied_silently():
    source = {"documentPatch": {"cargoGroups": [{"groupId": "g1", "description": "12 KG STEEL"}]}}
    with pytest.raises(ValueError, match="lacks a unique"):
        measurement_prose.generate(source, source)


@pytest.mark.parametrize(
    "surface,unit,total,expected",
    [
        ("BOXES OF 25KG EACH", "kilogram", 24000, 15525),
        ("BOXES OF 25 KG EACH", "metric_tonne", 24, 15.525),
        ("CRATES OF 25 LBS EACH", "pound", 24000, 15525),
    ],
)
def test_per_package_mass_is_derived_after_integer_quantity_rounding(
    surface, unit, total, expected
):
    from decimal import Decimal

    template = NS(
        bindings=(
            NS(
                target_paths=("documentPatch.cargoPackages[0].typeCategory",),
                occurrences=(NS(source_text=surface),),
            ),
        )
    )
    source = {
        "documentPatch": {
            "cargoPackages": [{"groupId": "g1", "quantity": 960}],
            "cargoGroups": [{"groupId": "g1", "netWeight": {"value": total, "unit": unit}}],
        }
    }
    target = deepcopy(source)
    target["documentPatch"]["cargoPackages"][0]["quantity"] = 621
    path = "documentPatch.cargoGroups[0].netWeight.value"
    assert package_prose.package_mass_values(template, source, target) == {
        path: Decimal(str(expected))
    }
    with pytest.raises(ValueError, match="contradicts generated total"):
        package_prose.validate_package_masses(template, source, target)
    target["documentPatch"]["cargoGroups"][0]["netWeight"]["value"] = expected
    package_prose.validate_package_masses(template, source, target)
    source["documentPatch"]["cargoGroups"][0]["netWeight"]["value"] += 1
    with pytest.raises(ValueError, match="exact source group-total proof"):
        package_prose.package_mass_values(template, source, target)


def test_per_package_mass_rejects_ambiguous_ownership_and_ignores_unrelated_text():
    binding = NS(
        target_paths=(
            "documentPatch.cargoPackages[0].typeCategory",
            "documentPatch.cargoPackages[1].typeCategory",
        ),
        occurrences=(NS(source_text="BOXES OF 25KG EACH"),),
    )
    template = NS(bindings=(binding,))
    with pytest.raises(ValueError, match="ambiguous package ownership"):
        package_prose.package_mass_values(template, {}, {})
    binding.target_paths = ("documentPatch.parties.shipper.name",)
    assert package_prose.package_mass_values(template, {}, {}) == {}
