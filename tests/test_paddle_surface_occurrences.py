import pytest

from document_ocr.synthesis.paddle_render import extra_value


def render(value, index):
    return extra_value(
        {"owner": "repeated-type", "surfaceKey": "type", "surfaceIndex": index},
        {"surfaceValues": {"type": value}}, {}, "sample", None, None, None,
        project_root=None,
    )


def test_scalar_is_exactly_one_occurrence_not_characters():
    assert render("40HC", 0) == ("40HC", None)
    with pytest.raises(ValueError):
        render("40HC", 1)


def test_list_preserves_distinct_alias_occurrences():
    assert render(["40HC", "40HQ"], 1) == ("40HQ", None)


@pytest.mark.parametrize("value,index", [("40HC", -1), ("40HC", True), ([], 0), ([""], 0), ([4], 0), ({"0": "40HC"}, 0)])
def test_rejects_missing_invalid_or_unowned_occurrence(value, index):
    with pytest.raises(ValueError):
        render(value, index)
