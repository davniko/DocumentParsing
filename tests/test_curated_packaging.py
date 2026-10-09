from copy import deepcopy
from types import SimpleNamespace

import pytest

from document_ocr.synthesis.curated_packaging import (
    contained_packing_context,
    contained_quantity_surfaces,
    validate_contained_quantity_sampling,
)

PATH = "documentPatch.goodsItemDetails[0].numberAndTypeOfPackages[0].packageQuantity"


def target(quantity):
    return {
        "documentPatch": {
            "goodsItemDetails": [
                {
                    "numberAndTypeOfPackages": [
                        {"packageQuantity": quantity, "typeCategory": "PACKAGE_PALLET"}
                    ]
                }
            ]
        }
    }


def blueprint(quantity=5, words=("54", "FIFTY FOUR")):
    source = "\n".join(s + " BOXES" for s in words)
    offset = 0
    occurrences = []
    for word in words:
        occurrences.append({"source_text": word, "byte_end": offset + len(word)})
        offset += len(word) + len(" BOXES\n")
    return SimpleNamespace(
        source=source,
        target=target(quantity),
        historical_bindings={"contents": {"target_paths": [], "occurrences": occurrences}},
    )


def test_both_levels_scale_without_relabeling_contents():
    bp = blueprint()
    proposal = target(10)
    original = deepcopy(proposal)
    assert contained_quantity_surfaces(
        bp,
        proposal,
        "contents",
        {"quantity_path": PATH, "source_quantity": 54, "source_unit": "BOXES"},
    ) == ["108", "ONE HUNDRED EIGHT"]
    assert proposal == original


@pytest.mark.parametrize(
    "source,inner,new,expected", [(20, 600, 11, "330"), (6, 240, 8, "320"), (4, 146, 6, "219")]
)
def test_distinct_reviewed_ratios(source, inner, new, expected):
    assert contained_quantity_surfaces(
        blueprint(source, (str(inner),)),
        target(new),
        "contents",
        {"quantity_path": PATH, "source_quantity": inner, "source_unit": "BOXES"},
    ) == [expected]


def test_fractional_content_is_not_rounded():
    with pytest.raises(ValueError, match="fractional"):
        contained_quantity_surfaces(
            blueprint(),
            target(6),
            "contents",
            {"quantity_path": PATH, "source_quantity": 54, "source_unit": "BOXES"},
        )


@pytest.mark.parametrize(
    "source,expected",
    [
        ("FIFTY FOUR", "ONE HUNDRED EIGHT"),
        ("Fifty Four", "One Hundred Eight"),
        ("Fifty four", "One hundred eight"),
        ("fifty four", "one hundred eight"),
    ],
)
def test_word_casing_is_preserved(source, expected):
    assert contained_quantity_surfaces(
        blueprint(words=(source,)),
        target(10),
        "contents",
        {"quantity_path": PATH, "source_quantity": 54, "source_unit": "BOXES"},
    ) == [expected]


@pytest.mark.parametrize("bad", [0, -1, True, 1.5, None])
def test_invalid_quantities_fail(bad):
    with pytest.raises(ValueError, match="positive integer"):
        contained_quantity_surfaces(
            blueprint(),
            target(bad),
            "contents",
            {"quantity_path": PATH, "source_quantity": 54, "source_unit": "BOXES"},
        )


def test_source_mismatch_and_public_ownership_fail():
    recipe = {"quantity_path": PATH, "source_quantity": 55, "source_unit": "BOXES"}
    with pytest.raises(ValueError, match="presentation"):
        contained_quantity_surfaces(blueprint(), target(5), "contents", recipe)
    bp = blueprint()
    bp.historical_bindings["contents"]["target_paths"] = [PATH]
    with pytest.raises(ValueError, match="cannot own public"):
        contained_quantity_surfaces(bp, target(5), "contents", dict(recipe, source_quantity=54))


def test_unrelated_path_or_recipe_key_fails():
    for recipe in (
        {"quantity_path": "mass", "source_quantity": 54, "source_unit": "BOXES"},
        {"quantity_path": PATH, "source_quantity": 54, "source_unit": "BOXES", "round": True},
    ):
        with pytest.raises(ValueError):
            contained_quantity_surfaces(blueprint(), target(5), "contents", recipe)


def test_config_rejects_fractional_draws_before_any_generation():
    bp = blueprint()
    bp.ownership_data = {
        "surfaces": {
            "contents": {
                "contained_quantity": {
                    "quantity_path": PATH,
                    "source_quantity": 54,
                    "source_unit": "BOXES",
                }
            }
        }
    }
    validate_contained_quantity_sampling(bp, multiple=5, fixed=None)
    validate_contained_quantity_sampling(bp, multiple=1, fixed=10)
    for multiple, fixed in [(1, None), (5, 6)]:
        with pytest.raises(ValueError, match="divisible by 5"):
            validate_contained_quantity_sampling(bp, multiple=multiple, fixed=fixed)


def test_private_context_matches_rendering_and_checks_printed_unit():
    bp = blueprint()
    recipe = {"quantity_path": PATH, "source_quantity": 54, "source_unit": "BOXES"}
    bp.ownership_data = {"surfaces": {"contents": {"contained_quantity": recipe}}}
    assert contained_packing_context(bp, target(10)) == [
        {
            "declared_package_path": PATH,
            "declared_quantity": 10,
            "contained_quantity": 108,
            "printed_contained_unit": "BOXES",
        }
    ]
    recipe["source_unit"] = "BAGS"
    with pytest.raises(ValueError, match="does not follow"):
        contained_packing_context(bp, target(10))


def test_no_private_packing_needs_no_extra_context():
    assert contained_packing_context(SimpleNamespace(ownership_data={}), {}) == []
