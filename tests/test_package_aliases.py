from copy import deepcopy
from random import Random

import pytest

from document_ocr.synthesis.curated import SourceContract, flat
from document_ocr.synthesis.curated_packaging import (
    compile_package_count_context,
    project_package_aliases_with_context,
)
from document_ocr.synthesis.curated_physical import PhysicalRenderPlan
from document_ocr.synthesis.curated_templates import (
    compile_sampling_blueprint,
    render_sampling_blueprint,
)
from document_ocr.synthesis.package_aliases import (
    PACKAGE_ALIASES,
    select_package_alias,
    validate_package_alias_registry,
)
from document_ocr.synthesis.package_registry import package_category_surface_present

PREFIX = "documentPatch.goodsItemDetails[0].numberAndTypeOfPackages[0]"


def _aliases(bp, target, surfaces, *, mode, rng):
    """Build authoritative physical count values for the controlled fixture."""
    leaves, values = flat(target), {}
    for variable in bp.contract.variables:
        if variable.kind != "count":
            continue
        counts = {
            leaves[binding.path]
            for binding in bp.contract.targets
            if binding.expression == "{" + variable.key + "}"
            and binding.path.endswith(".packageQuantity")
        }
        assert len(counts) == 1
        values[variable.key] = str(counts.pop())
    plan = PhysicalRenderPlan(target, values, surfaces, {})
    return project_package_aliases_with_context(
        bp, plan, compile_package_count_context(bp, plan), mode=mode, rng=rng
    )


def blueprint(source="CARTONS"):
    raw = "2 " + source + "\n"
    row = {
        "documentId": "package-alias",
        "joinedRawText": raw,
        "target": {
            "schemaVersion": "7.0.0",
            "documentPatch": {
                "negotiability": "non_negotiable",
                "goodsItemDetails": [
                    {
                        "numberAndTypeOfPackages": [
                            {"packageQuantity": 2, "typeCategory": "PACKAGE_CARTON"}
                        ]
                    }
                ],
            },
        },
    }
    contract = SourceContract(
        variables=[
            dict(
                key="quantity",
                kind="count",
                value="2",
                meaning="declared package count",
                required_literals=[],
                occurrences=[dict(text="2", occurrence=1, presentation="number")],
            )
        ],
        targets=[dict(path=PREFIX + ".packageQuantity", expression="{quantity}")],
        fixed_context="Alias-only test: no commodity or physical changes.",
    )
    history = {
        "document_id": row["documentId"],
        "bindings": [
            dict(
                logical_key="package",
                target_paths=[PREFIX + ".typeCategory"],
                value_kind="package",
                occurrences=[
                    dict(byte_start=2, byte_end=2 + len(source.encode()), source_text=source)
                ],
            )
        ],
    }
    return compile_sampling_blueprint(row, history, contract)


@pytest.mark.parametrize("category", sorted(PACKAGE_ALIASES))
@pytest.mark.parametrize("quantity", [1, 2])
def test_all_categories_retain_subtype_and_number_through_owned_renderer(category, quantity):
    bp = blueprint()
    target = deepcopy(bp.target)
    package = target["documentPatch"]["goodsItemDetails"][0]["numberAndTypeOfPackages"][0]
    package.update(typeCategory=category, packageQuantity=quantity)
    surfaces = _aliases(bp, target, {}, mode="source", rng=Random(7))
    expected = PACKAGE_ALIASES[category].variants(quantity)[0]
    assert surfaces == {"package": [expected]}
    rendered, labels, proof = render_sampling_blueprint(
        bp, target, {"quantity": str(quantity)}, surface_values=surfaces
    )
    assert rendered == f"{quantity} {expected}\n"
    assert labels == target
    assert package_category_surface_present(expected, category)
    assert all(
        set(edit["targetPaths"]) <= {PREFIX + ".packageQuantity", PREFIX + ".typeCategory"}
        for edit in proof["edits"]
    )


@pytest.mark.parametrize(
    "category",
    [category for category, entry in PACKAGE_ALIASES.items() if entry.alternate_singular],
)
@pytest.mark.parametrize("quantity", [1, 2])
def test_every_reviewed_alternate_can_be_drawn_and_rendered(category, quantity):
    bp = blueprint()
    target = deepcopy(bp.target)
    target["documentPatch"]["goodsItemDetails"][0]["numberAndTypeOfPackages"][0].update(
        typeCategory=category, packageQuantity=quantity
    )
    observed = set()
    for seed in range(12):
        surfaces = _aliases(bp, target, {}, mode="registry", rng=Random(seed))
        rendered, labels, _ = render_sampling_blueprint(
            bp, target, {"quantity": str(quantity)}, surface_values=surfaces
        )
        assert labels == target
        assert rendered == f"{quantity} {surfaces['package'][0]}\n"
        observed.add(surfaces["package"][0])
    assert observed == set(PACKAGE_ALIASES[category].variants(quantity))


@pytest.mark.parametrize(
    "source,category,quantity,expected",
    [
        ("CTNS", "PACKAGE_CARTON", 1, "CTN"),
        ("PLT", "PACKAGE_PALLET", 2, "PLTS"),
        ("CTNS", "PACKAGE_DRUM_PLASTIC", 2, "PLASTIC DRUMS"),
        ("PACKAGE(S)", "PACKAGE_BOX", 2, "BOX(ES)"),
        (
            "BOX(ES)",
            "PACKAGE_BOX_WOODEN_NATURAL_WOOD_ORDINARY",
            1,
            "ORDINARY WOODEN BOX(ES) (NATURAL WOOD)",
        ),
        ("PACKAGE(S)", "PACKAGE_BULK_LIQUID", 1, "BULK LIQUID"),
    ],
)
def test_source_style_never_drops_material_or_pluralizes_trailing_qualifiers(
    source, category, quantity, expected
):
    assert (
        select_package_alias(category, quantity, source, mode="source", rng=Random(2)) == expected
    )


@pytest.mark.parametrize("quantity", [None, True, 0, -1, 1.5])
def test_invalid_counts_fail(quantity):
    with pytest.raises(ValueError, match="positive integer"):
        select_package_alias("PACKAGE_BOX", quantity, "BOXES", mode="source", rng=Random(1))


def test_invalid_category_mode_and_source_fail():
    for category, source, mode in [
        ("PACKAGE_UNKNOWN", "BOX", "source"),
        ("PACKAGE_BOX", "", "source"),
        ("PACKAGE_BOX", "BOX", "guess"),
    ]:
        with pytest.raises(ValueError):
            select_package_alias(category, 1, source, mode=mode, rng=Random(1))


def test_registry_drift_and_missing_material_fail():
    categories = set(PACKAGE_ALIASES)
    codes = {category: entry.application_code for category, entry in PACKAGE_ALIASES.items()}
    assert len(categories) == 44
    validate_package_alias_registry(categories, codes)
    validate_package_alias_registry(categories, dict(codes, PACKAGE_OUTSIDE_TASK="ZZ"))
    with pytest.raises(ValueError, match="categories differ"):
        validate_package_alias_registry(categories - {"PACKAGE_BOX"}, codes)
    with pytest.raises(ValueError, match="codes differ"):
        validate_package_alias_registry(categories, dict(codes, PACKAGE_BOX="ZZ"))
    for category, wrong in [
        ("PACKAGE_DRUM_PLASTIC", "DRUMS"),
        ("PACKAGE_BAG_JUMBO", "BULK BAGS"),
        ("PACKAGE_BAG_PAPER_MULTI_WALL", "PAPER BAGS"),
    ]:
        assert not package_category_surface_present(wrong, category)


def test_rng_replay_and_input_surface_immutability():
    bp = blueprint()
    target = deepcopy(bp.target)
    target["documentPatch"]["goodsItemDetails"][0]["numberAndTypeOfPackages"][0]["typeCategory"] = (
        "PACKAGE_DRUM_PLASTIC"
    )
    before = {"package": ["OLD"], "unrelated": ["FROZEN"]}
    original = deepcopy(before)
    first = _aliases(bp, target, before, mode="registry", rng=Random(6))
    second = _aliases(bp, target, before, mode="registry", rng=Random(6))
    assert first == second
    assert before == original
    assert first["unrelated"] == ["FROZEN"]
    with pytest.raises(ValueError, match="occurrence count differs"):
        _aliases(bp, target, {"package": []}, mode="source", rng=Random(0))


def test_composite_count_owner_is_not_replaced_by_a_noun():
    from dataclasses import replace

    bp = blueprint()
    region = next(region for region in bp.regions if region.curated_key is None)
    composite = replace(
        region, target_paths=(PREFIX + ".typeCategory", PREFIX + ".packageQuantity")
    )
    bp = replace(bp, regions=(composite,))
    assert _aliases(bp, bp.target, {"package": ["2 CARTONS"]}, mode="registry", rng=Random(1)) == {
        "package": ["2 CARTONS"]
    }


@pytest.mark.parametrize("gap,expected", [(" ", "CARTON"), ("\t", "CARTON"), ("\n", "CARTONS")])
def test_allocation_count_controls_only_its_adjacent_package_noun(gap, expected):
    from dataclasses import replace

    from document_ocr.synthesis.curated_templates import RenderRegion

    bp = blueprint()
    target = deepcopy(bp.target)
    allocation_path = "documentPatch.goodsItemDetails[0].splitGoodsPlacement[0].packageQuantity"
    target["documentPatch"]["goodsItemDetails"][0]["splitGoodsPlacement"] = [{"packageQuantity": 1}]
    noun = next(region for region in bp.regions if region.curated_key is None)
    noun = replace(noun, start=1 + len(gap), end=8 + len(gap))
    count = RenderRegion("allocation", 0, 0, 1, "1", (allocation_path,), "integer")
    bp = replace(bp, source="1" + gap + "CARTONS\n", regions=(count, noun))
    assert _aliases(bp, target, {}, mode="source", rng=Random(0)) == {"package": [expected]}


def test_conflicting_adjacent_counts_fail_instead_of_guessing():
    from dataclasses import replace

    from document_ocr.synthesis.curated_templates import RenderRegion

    bp = blueprint()
    target = deepcopy(bp.target)
    allocation_path = "documentPatch.goodsItemDetails[0].splitGoodsPlacement[0].packageQuantity"
    target["documentPatch"]["goodsItemDetails"][0]["splitGoodsPlacement"] = [{"packageQuantity": 1}]
    noun = next(region for region in bp.regions if region.curated_key is None)
    first = RenderRegion("allocation", 0, 0, 1, "1", (allocation_path,), "integer")
    second = RenderRegion("total", 0, 10, 11, "2", (PREFIX + ".packageQuantity",), "integer")
    bp = replace(bp, source="1 CARTONS 2\n", regions=(first, noun, second))
    with pytest.raises(ValueError, match="conflicting adjacent count"):
        _aliases(bp, target, {}, mode="source", rng=Random(0))
