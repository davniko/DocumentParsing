import copy

import pytest

from document_ocr.synthesis.paddle_render import remove_optional_target


@pytest.mark.parametrize(
    "goods",
    [[], [{}], [{"description": "GOODS"}], [{"handlingInstructions": None}]],
)
def test_optional_exclusion_accepts_absent_values_without_changing_other_facts(goods):
    target = {"goodsItemDetails": copy.deepcopy(goods), "billOfLadingNumber": "ABC"}
    expected = copy.deepcopy(target)
    if expected["goodsItemDetails"]:
        expected["goodsItemDetails"][0].pop("handlingInstructions", None)
    remove_optional_target(target, "/goodsItemDetails/0/handlingInstructions")
    assert target == expected


def test_present_exclusion_is_idempotent_and_preserves_siblings():
    target = {"goodsItemDetails": [{"description": "GOODS", "handlingInstructions": ["X"]}]}
    remove_optional_target(target, "/goodsItemDetails/0/handlingInstructions")
    remove_optional_target(target, "/goodsItemDetails/0/handlingInstructions")
    assert target == {"goodsItemDetails": [{"description": "GOODS"}]}


@pytest.mark.parametrize("target", [{}, {"parties": None}, {"parties": {}}])
def test_absent_optional_parent_is_already_excluded(target):
    before = copy.deepcopy(target)
    remove_optional_target(target, "/parties/consignee/contactDetails")
    assert target == before


@pytest.mark.parametrize("path", ["", "/", "goods", "/goods//handling", None])
def test_malformed_paths_are_not_treated_as_absence(path):
    with pytest.raises(ValueError, match="absolute nonempty"):
        remove_optional_target({}, path)


@pytest.mark.parametrize("part", ["-1", "name", "١"])
def test_invalid_list_selectors_fail(part):
    with pytest.raises(ValueError, match="nonnegative list index"):
        remove_optional_target({"goods": [{}]}, "/goods/" + part)


def test_malformed_nonoptional_parent_fails():
    with pytest.raises(ValueError, match="non-container"):
        remove_optional_target({"goods": "broken"}, "/goods/0/handling")


def test_explicit_list_element_exclusion_retains_order():
    target = {"goods": [{"name": "A"}, {"name": "B"}]}
    remove_optional_target(target, "/goods/0")
    assert target == {"goods": [{"name": "B"}]}
