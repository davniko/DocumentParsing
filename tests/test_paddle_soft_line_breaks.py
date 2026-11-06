import pytest

from document_ocr.synthesis.paddle_render import render_soft_line_breaks
from document_ocr.synthesis.paddle_templates import validate_override_selectors


@pytest.mark.parametrize("text", [
    "SHOP 9, AVENIDA DA PRAIA GRANDE,\nSE,\nMAC",
    "PLOT 8, AL MIRBAH ROAD\r\nTARIF KALBA\r\nA.E",
    "UNIT 1204, GATE AVENUE,\rARE",
    "A  B \n C\n\nD",
])
def test_only_reviewed_line_boundaries_become_soft(text):
    assert render_soft_line_breaks(text, {}) == text
    result = render_soft_line_breaks(text, {"softLineBreaks": True})
    assert not any(c in result for c in "\r\n")
    assert "".join(result.split()) == "".join(text.split())
    assert render_soft_line_breaks(result, {"softLineBreaks": True}) == result


def test_inline_spaces_punctuation_and_case_are_preserved():
    assert render_soft_line_breaks("Unit  12-B, St. John's\nAE", {"softLineBreaks": True}) == "Unit  12-B, St. John's AE"


@pytest.mark.parametrize("value", [False, 1, "true", None])
def test_invalid_enablement_fails_in_render_and_catalog(value):
    with pytest.raises(ValueError, match="explicit reviewed"):
        render_soft_line_breaks("A\nB", {"softLineBreaks": value})
    with pytest.raises(ValueError, match="explicit reviewed"):
        validate_override_selectors({"overrides": [{"key": "a", "softLineBreaks": value}]})
