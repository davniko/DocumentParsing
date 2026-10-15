import re

import pytest
from pydantic import TypeAdapter, ValidationError

from document_ocr.label_schemas.mpci_bill_of_lading import HsCode
from document_ocr.labeling_agents.direct_grounding import source_fidelity_findings
from document_ocr.synthesis.curated_hs import hs_digits, render_source_hs, sample_source_hs_codes
from document_ocr.synthesis.generators import DeterministicStream


@pytest.mark.parametrize("width", range(6, 21))
def test_full_width_and_leading_zero_identity(width):
    source = "020622" + "0" * (width - 6)
    codes = sample_source_hs_codes([source], ["010121"], DeterministicStream(4, "hs", "doc"))
    assert len(codes[0]) == width and codes[0].startswith("010121")
    assert TypeAdapter(HsCode).validate_python(codes[0]) == codes[0]
    assert codes == sample_source_hs_codes(
        [source], ["010121"], DeterministicStream(4, "hs", "doc")
    )


@pytest.mark.parametrize(
    "surface",
    [
        "8528.59.00.00.11",
        "85.28.59.00.00.11",
        "8528.59-0011",
        "8528 5900",
        "8528,59",
        "852859",
        "5205:22",
        "5205:22.00-11",
    ],
)
def test_source_separators_remain_in_rendering_and_supported_targets(surface):
    old = hs_digits(surface)
    code = sample_source_hs_codes([old], ["610910"], DeterministicStream(5, "hs", "doc"))[0]
    rendered = render_source_hs(surface, code)
    assert re.sub("[0-9]", "9", rendered) == re.sub("[0-9]", "9", surface)
    assert hs_digits(rendered) == code
    assert code.startswith("610910") and code.isascii() and code.isdigit()
    assert TypeAdapter(HsCode).validate_python(rendered) == rendered


def test_multiple_codes_and_shared_parent_extensions_remain_distinct():
    sources = ["020622", "02062200", "0206220000", "0206221000", "14049010"]
    codes = sample_source_hs_codes(
        sources, ["020714", "140110"], DeterministicStream(9, "hs", "doc")
    )
    assert [len(c) for c in codes] == [len(c) for c in sources]
    assert len(set(codes)) == len(sources)
    assert [c[:6] for c in codes] == ["020714"] * 4 + ["140110"]


@pytest.mark.parametrize(
    "surface",
    [
        "HS CODE: 123456",
        "12345",
        "123456789012345678901",
        "\uff11\uff12\uff13\uff14\uff15\uff16",
        "123456 / 234567",
        "ABC123456",
        ":520522",
        "520522:",
        "HS CODE:5205:22",
        "5205\uff1a22",
    ],
)
def test_captions_unrelated_roles_and_invalid_codes_rejected(surface):
    with pytest.raises(ValueError):
        hs_digits(surface)


def test_missing_identity_or_wrong_width_never_falls_back():
    with pytest.raises(ValueError, match="source code slots"):
        sample_source_hs_codes(
            ["12345678"], ["610910", "620111"], DeterministicStream(1, "hs", "doc")
        )
    with pytest.raises(ValueError, match="width"):
        render_source_hs("8528.59.00.00.11", "610910")
    with pytest.raises(ValidationError):
        TypeAdapter(HsCode).validate_python("1" * 21)


@pytest.mark.parametrize(
    "value", [":520522", "520522:", "HS CODE:5205:22", "5205\uff1a22"]
)
def test_hs_target_rejects_colon_captions_and_non_ascii_separators(value):
    with pytest.raises(ValidationError):
        TypeAdapter(HsCode).validate_python(value)


@pytest.mark.parametrize("code", ["5205:22", "520522"])
@pytest.mark.parametrize("printed", ["HS CODE:5205:22", "HS CODE: 520522"])
def test_colon_hs_grounding_accepts_complete_codes_without_rewriting(code, printed):
    candidate = {"documentPatch": {"goodsItemDetails": [{"hsCodes": [code]}]}}
    assert source_fidelity_findings(candidate, printed, "cargo") == []
    assert candidate["documentPatch"]["goodsItemDetails"][0]["hsCodes"] == [code]


@pytest.mark.parametrize("code,printed", [("520522", "520522:00"), ("520522", "11:520522")])
def test_colon_hs_grounding_rejects_prefix_or_suffix_of_a_longer_code(code, printed):
    candidate = {"documentPatch": {"goodsItemDetails": [{"hsCodes": [code]}]}}
    findings = source_fidelity_findings(candidate, "HS CODE: " + printed, "cargo")
    assert len(findings) == 1 and findings[0].issue == "unsupported"
