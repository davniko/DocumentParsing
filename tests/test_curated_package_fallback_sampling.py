"""A printed free-text package unit remains source-owned, never a fake category."""

from copy import deepcopy
from types import SimpleNamespace

import pytest
from test_curated_dg_source_bundles import dg_support
from test_curated_scenarios import catalog, source_row
from test_curated_templates import compile_sampling_blueprint, source_fixture

from document_ocr.synthesis.curated import SourceContract
from document_ocr.synthesis.curated_campaign import wording_request
from document_ocr.synthesis.curated_scenarios import SourceCapabilities, _observation
from document_ocr.synthesis.curated_templates import render_sampling_blueprint


def fallback_row(noun="SETS"):
    row = source_row()
    goods = row["target"]["documentPatch"]["goodsItemDetails"][0]
    package = goods["numberAndTypeOfPackages"][0]
    del package["typeCategory"]
    package["typeOfPackages"] = noun
    goods["dangerousGoods"] = [{"unNumber": "1824", "hazardCategory": "CORROSIVE_SUBSTANCES"}]
    return row


@pytest.mark.parametrize("noun", ["SETS", "PARCEL"])
def test_fallback_package_is_observed_and_preserved_without_category_insertion(noun):
    row = fallback_row(noun)
    before = deepcopy(row)
    support = dg_support(row)
    cap = SourceCapabilities(family="dg_chemical", physical_profile="source_bundle")
    assert _observation(row).package_category is None
    for variant in range(1, 7):
        result = support.sample(row, seed=17, variant=variant, capabilities=cap)
        goods = result.cargo["goodsItemDetails"][0]
        package = goods["numberAndTypeOfPackages"][0]
        assert package.keys() == {"packageQuantity", "typeOfPackages"}
        assert package["typeOfPackages"] == noun
        assert package["packageQuantity"] > 0
        assert (
            sum(p["packageQuantity"] for p in goods["splitGoodsPlacement"])
            == package["packageQuantity"]
        )
        assert result.provenance["printedPackageFallback"] == {
            "typeOfPackages": noun,
            "basis": "exact_source_package_unit",
        }
    assert row == before


@pytest.mark.parametrize(
    "packages",
    [
        [{"packageQuantity": 1}],
        [{"packageQuantity": 1, "typeCategory": "PACKAGE_BOX", "typeOfPackages": "SETS"}],
        [{"packageQuantity": 1, "typeOfPackages": ""}],
        [{"packageQuantity": 1, "typeOfPackages": "  "}],
        [{"packageQuantity": 1, "typeOfPackages": None}],
        [{"typeOfPackages": "SETS"}],
    ],
)
def test_ambiguous_or_missing_package_unit_is_not_observed(packages):
    row = fallback_row()
    row["target"]["documentPatch"]["goodsItemDetails"][0]["numberAndTypeOfPackages"] = packages
    assert _observation(row) is None


def test_free_text_unit_requires_explicit_source_profile_and_cannot_match_named_domain():
    row = fallback_row()
    support = dg_support(row)
    with pytest.raises(ValueError, match="requires a source-backed physical profile"):
        support.sample(
            row, seed=17, variant=1, capabilities=SourceCapabilities(family="dg_chemical")
        )
    with pytest.raises(ValueError, match="no complete train-only cargo donors"):
        support.sample(
            row,
            seed=17,
            variant=1,
            capabilities=SourceCapabilities(
                family="dg_chemical",
                physical_profile="source_bundle",
                allowed_package_categories=("PACKAGE_PACKAGE",),
            ),
        )


def test_free_text_donor_does_not_enter_transferable_named_category_pool():
    typed = source_row()
    fallback = fallback_row()
    fallback["documentId"] = "fallback"
    del fallback["target"]["documentPatch"]["goodsItemDetails"][0]["dangerousGoods"]
    support = catalog([typed, fallback])
    donors = support._donors(_observation(typed), SourceCapabilities(family="ambient"))
    assert [row.document_id for row in donors] == [typed["documentId"]]


def test_source_whole_unit_profile_preserves_a_printed_vehicle_package_unit():
    row = fallback_row("VEHICLES")
    goods = row["target"]["documentPatch"]["goodsItemDetails"][0]
    del goods["dangerousGoods"]
    goods["description"], goods["hsCodes"] = "TRACTORS", ["870121"]
    support = catalog([row], phrases={"870121": "Road tractors", "870122": "Other tractors"})
    scenario = support.sample(
        row,
        seed=17,
        variant=1,
        capabilities=SourceCapabilities(
            family="vehicle",
            physical_profile="source_whole_units",
            allowed_hs_headings=("8701",),
        ),
    )
    package = scenario.cargo["goodsItemDetails"][0]["numberAndTypeOfPackages"][0]
    assert package.keys() == {"packageQuantity", "typeOfPackages"}
    assert package["typeOfPackages"] == "VEHICLES"
    assert scenario.provenance["printedPackageFallback"]["basis"] == "exact_source_package_unit"


def test_owned_fallback_noun_survives_count_rendering_without_alias_or_category_changes():
    row, contract, historical = source_fixture()
    raw = row["joinedRawText"]
    offset = len(raw[: raw.index("BOXES")].encode())
    row["joinedRawText"] = raw.replace("BOXES", "SETS")
    package = row["target"]["documentPatch"]["goodsItemDetails"][0]["numberAndTypeOfPackages"][0]
    del package["typeCategory"]
    package["typeOfPackages"] = "SETS"
    for binding in historical["bindings"]:
        for occurrence in binding["occurrences"]:
            if occurrence["byte_start"] > offset:
                occurrence["byte_start"] -= 1
                occurrence["byte_end"] -= 1
            if binding["logical_key"] == "package":
                occurrence["source_text"] = "SETS"
                occurrence["byte_end"] -= 1
                binding["target_paths"] = [
                    "documentPatch.goodsItemDetails[0].numberAndTypeOfPackages[0].typeOfPackages"
                ]
    blueprint = compile_sampling_blueprint(row, historical, contract)
    target = deepcopy(row["target"])
    target["documentPatch"]["goodsItemDetails"][0]["numberAndTypeOfPackages"][0][
        "packageQuantity"
    ] = 36
    rendered, actual, proof = render_sampling_blueprint(blueprint, target, {})
    assert "36 SETS" in rendered
    assert actual == target
    assert len(proof["edits"]) == 1
    assert proof["edits"][0]["before"] == "24"
    assert proof["edits"][0]["after"] == "36"


def test_generation_context_receives_literal_fallback_unit_without_fake_category():
    row = fallback_row()
    support = dg_support(row)
    scenario = support.sample(
        row,
        seed=17,
        variant=1,
        capabilities=SourceCapabilities(family="dg_chemical", physical_profile="source_bundle"),
    )
    contract = SourceContract(
        variables=[
            {
                "key": "product",
                "kind": "product",
                "value": "COTTON YARN",
                "meaning": "main goods passage",
                "required_literals": [],
                "occurrences": [{"text": "COTTON YARN", "occurrence": 1, "presentation": "text"}],
            }
        ],
        targets=[
            {"path": "documentPatch.goodsItemDetails[0].description", "expression": "{product}"}
        ],
        fixed_context="Reviewed source context.",
    )
    blueprint = SimpleNamespace(contract=contract, ownership_data={}, target=row["target"])
    request = wording_request(blueprint, scenario, "inspection", goods_only=True)
    assert "packageTypeDescriptions:\n  - SETS" in request.context
    assert "packageCategories: []" in request.context
    assert "PACKAGE_PACKAGE" not in request.context
