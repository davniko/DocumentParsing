from copy import deepcopy
from types import SimpleNamespace as NS

import pytest

from document_ocr.synthesis.template_compiler import measurement_prose, package_prose


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
