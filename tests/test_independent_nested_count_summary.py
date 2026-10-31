"""Inner/outer count captions remain private and consume a contained ledger."""

import pytest
from test_independent_numeric import (
    FakeBlueprint,
    certificate,
    draw,
    quoted,
    target,
    write_inventory,
)

from document_ocr.synthesis.independent_preparation import packing_presentation_context


def context(tmp_path, presentation="INNERS PACKAGE: {inner}\nOUTERS PACKAGE: {outer}"):
    text = "INNERS PACKAGE: 118\nOUTERS PACKAGE: 106"
    bp = FakeBlueprint("nested-counts", text, target())
    region = quoted(
        text, text, "nested_count_summary", presentation=presentation, factGroup="packing"
    )
    authority = write_inventory(tmp_path / "numeric.json", bp, certificate(bp, regions=[region]))
    return authority, authority.apply_blueprint(bp)


def three_level_draw(category, quantities):
    value = draw(category, quantities)
    value["provenance"]["numericEnvelope"] = {
        "authority": "native_goods_conditioned_composed_profile_v1",
        "composition": {
            "relation": "CONTAINS",
            "countedPerGroup": 1,
            "relatedPerGroup": 4,
            "supportPerGroup": 0,
            "loosePerGroup": 0,
            "supportTareKgPerGroup": "0",
            "contents": {
                "packageSingular": "BOTTLE",
                "packagePlural": "BOTTLES",
                "unitsPerRelated": 6,
            },
        },
    }
    return value


def test_counts_follow_current_related_units_without_new_public_rows(tmp_path):
    authority, bp = context(tmp_path)
    plan = {"privatePackingRelation": "CONTAINS", "relatedPackageType": "PACKAGE_BOX"}
    value = three_level_draw("PACKAGE_CARTON", (10,))
    receipt = authority.build_ledger(bp, value, plan, 17, 1)
    fact = receipt["facts"]["packing"]
    assert fact["innerQuantity"] >= fact["outerQuantity"] == receipt["relatedQuantity"]
    assert fact["sourceCountRatioUsed"] is fact["publicPackageRowsAdded"] is False
    assert list(receipt["surfaces"].values()) == [
        [f"INNERS PACKAGE: {fact['innerQuantity']}\nOUTERS PACKAGE: {fact['outerQuantity']}"]
    ]
    assert authority.allowed_private_relations(bp.document_id) == ("CONTAINS", "SUPPORTS")
    assert bp.target == target()
    assert fact["innerQuantity"] == 240 and fact["outerQuantity"] == 40


def test_missing_innermost_authority_is_not_replaced_with_random_counts(tmp_path):
    authority, bp = context(tmp_path)
    with pytest.raises(ValueError, match="innermost package authority"):
        authority.build_ledger(
            bp,
            draw("PACKAGE_CARTON", (10,)),
            {"privatePackingRelation": "CONTAINS", "relatedPackageType": "PACKAGE_BOX"},
            17,
            1,
        )


@pytest.mark.parametrize("relation", ["NONE", "ON_SUPPORT"])
def test_reverse_or_missing_nesting_is_rejected(tmp_path, relation):
    authority, bp = context(tmp_path)
    with pytest.raises(ValueError, match="presentation"):
        authority.build_ledger(
            bp, draw(), {"privatePackingRelation": relation, "relatedPackageType": None}, 1, 1
        )


@pytest.mark.parametrize("presentation", ["{inner}", "{inner}{inner}", "{inner}{outer}{unowned}"])
def test_every_caption_count_requires_exact_ownership(tmp_path, presentation):
    with pytest.raises(ValueError, match="inner/outer"):
        context(tmp_path, presentation)


def test_per_unit_packing_prints_owned_group_mass_without_rounding_inner_fill(tmp_path):
    text = "PACKING:100MLX100BOTTLE/CARTON"
    bp = FakeBlueprint("unit-packing", text, target())
    region = quoted(text, text, "unit_inner_packing")
    authority = write_inventory(tmp_path / "numeric.json", bp, certificate(bp, regions=[region]))
    owned = authority.apply_blueprint(bp)
    value = draw("PACKAGE_CARTON", (10,))
    plan = {"privatePackingRelation": "CONTAINS", "relatedPackageType": "PACKAGE_BAG"}
    result = authority.build_ledger(owned, value, plan, 17, 1)
    fact = result["facts"]["unitInnerPacking"]
    assert fact["countedUnitNetKg"] == "180"
    assert fact["innerPerCounted"] * 10 == result["relatedQuantity"]
    assert list(result["surfaces"].values()) == [
        [f"PACKING: {fact['innerPerCounted']} BAGS / CARTON; 180 KG NET / CARTON"]
    ]
    assert "100ML" not in str(result["surfaces"])


def test_partial_overpacks_do_not_claim_the_whole_shipment(tmp_path):
    text = "FIRST OVERPACK\nSECOND OVERPACK"
    bp = FakeBlueprint("partial-packing", text, target())
    regions = [
        quoted(
            text,
            quote,
            "subset_component_packing",
            factGroup="partial",
            componentCount=3,
            componentIndex=i,
        )
        for i, quote in enumerate(("FIRST OVERPACK", "SECOND OVERPACK"))
    ]
    authority = write_inventory(tmp_path / "numeric.json", bp, certificate(bp, regions=regions))
    owned = authority.apply_blueprint(bp)
    result = authority.build_ledger(
        owned,
        three_level_draw("PACKAGE_BOX", (13,)),
        {"privatePackingRelation": "CONTAINS", "relatedPackageType": "PACKAGE_BAG"},
        17,
        1,
    )
    fact = result["facts"]["partial"]
    assert sum(fact["quantities"]) == 13
    assert 0 < sum(fact["quantities"][:2]) < 13
    assert fact["unprintedRemainder"] == fact["quantities"][2] > 0
    assert len(result["surfaces"]) == 2
    for part in fact["innerContents"].values():
        assert part["innermostQuantity"] == 6 * part["intermediateQuantity"]
    assert all("EACH CONTAINING 6 BOTTLES" in texts[0] for texts in result["surfaces"].values())


def test_package_choice_knows_the_required_counting_level_before_selecting_a_receptacle():
    certificate = {"regions": [{"role": "unit_inner_packing"}]}
    text = packing_presentation_context(certificate, ("CONTAINS", "SUPPORTS"))
    assert "DECLARED COUNTED UNIT" in text and "inner receptacles" in text
    assert "THREE" not in text
    certificate["regions"][0]["role"] = "subset_component_packing"
    assert "THREE levels" in packing_presentation_context(certificate, ("CONTAINS",))
    assert packing_presentation_context(certificate, ("NONE",)) == ""
    assert packing_presentation_context(certificate, ("MIXED",)) == ""


def test_nested_alias_constructor_changes_only_presentations(tmp_path):
    text = "PACKING:100MLX100BOTTLE/CARTON"
    bp = FakeBlueprint("alias-unit-packing", text, target())
    region = quoted(text, text, "unit_inner_packing")
    authority = write_inventory(tmp_path / "numeric.json", bp, certificate(bp, regions=[region]))
    owned = authority.apply_blueprint(bp)
    value = draw("PACKAGE_CARTON", (10,))
    plan = {"privatePackingRelation": "CONTAINS", "relatedPackageType": "PACKAGE_BAG"}
    ordinary = authority.build_ledger(owned, value, plan, 17, 1)
    aliases = {"PACKAGE_CARTON": "CTN", "PACKAGE_BAG": "BAG(S)"}
    varied = authority.build_ledger(
        owned, value, plan, 17, 1, noun_renderer=lambda category, count: aliases[category]
    )
    assert {k: v for k, v in varied.items() if k != "surfaces"} == {
        k: v for k, v in ordinary.items() if k != "surfaces"
    }
    assert "BAG(S) / CTN" in str(varied["surfaces"])
    assert "BAGS / CARTON" in str(ordinary["surfaces"])
