from types import SimpleNamespace as NS

import pytest

from document_ocr.synthesis.template_compiler import count_aliases as c


@pytest.mark.parametrize("text", ["TWO(1)", "THREE (2)", "FORTY-NINE(60)"])
def test_contradictions_are_rejected(text):
    with pytest.raises(ValueError, match="contradictory"):
        c.validate(text)


def test_only_complete_number_phrases_are_aliases():
    c.validate("AND (2) PACKAGES; ONE HUNDRED AND TWENTY NINE (129)")
    assert len(c.aliases("AND (2) PACKAGES")) == 0
    assert not c.aliases("Originals: 3/THREE\n\n(22) Place and date")
    assert c.render_pair("SEVEN (7) PACKAGES ONLY", 7, 4) == "FOUR (4) PACKAGES ONLY"
    assert c.render_pair("Seven (7)", 7, 2) == "Two (2)"
    with pytest.raises(ValueError, match="declared"):
        c.render_pair("SEVEN (8)", 7, 4)


def fixture(text, start, end, *, words=False):
    path = "documentPatch.cargoPackages[0].quantity"
    target = {"documentPatch": {"cargoPackages": [{"quantity": 7}]}}
    binding = NS(
        logical_key="quantity",
        occurrences=[NS(byte_start=start, byte_end=end)],
        target_paths=[] if words else [path],
        derivation="number_to_words" if words else None,
        dependency_paths=[path] if words else [],
        dependency_bindings=[],
    )
    return text.encode(), NS(bindings=[binding]), target


def test_unowned_word_and_unowned_digit_pin_only_explicit_quantity():
    assert c.fixed_quantities(*fixture("SEVEN (7)", 7, 8)) == {
        "documentPatch.cargoPackages[0].quantity": 7
    }
    assert c.fixed_quantities(*fixture("SEVEN (7)", 0, 5, words=True)) == {
        "documentPatch.cargoPackages[0].quantity": 7
    }


def test_fully_mutable_alias_remains_variable_and_offsets_are_bytes():
    assert not c.fixed_quantities(*fixture("SEVEN (7)", 0, 9))
    assert c.fixed_quantities(*fixture("é SEVEN (7)", 10, 11))
    with pytest.raises(ValueError, match="partially owned"):
        c.fixed_quantities(*fixture("SEVEN (7)", 1, 5))
