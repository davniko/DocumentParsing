from types import SimpleNamespace as NS

import pytest

from document_ocr.synthesis.template_compiler.descendant import _partition_target_surface


def slots():
    return (
        NS(source_text="TABRAK BUILDING 39 ST.HORYAL", render_policy="natural_text"),
        NS(source_text="21500", render_policy="opaque_identifier"),
    )


def test_long_address_does_not_allocate_country_words_to_postal_slot():
    value = "220 Lehigh Valley Commerce Drive, Whitehall Township, United States 92390"
    assert _partition_target_surface(value, slots()) == (
        "220 Lehigh Valley Commerce Drive, Whitehall Township, United States",
        "92390",
    )


@pytest.mark.parametrize("suffix", ["1234", "123456", "ABCDE", ""])
def test_numeric_component_must_have_the_certified_shape(suffix):
    with pytest.raises(ValueError, match="certified numeric component"):
        _partition_target_surface("10 New Road " + suffix, slots())


def test_ordinary_natural_text_segments_keep_word_partitioning():
    natural = tuple(NS(source_text=s.source_text, render_policy="natural_text") for s in slots())
    result = _partition_target_surface(
        "one two three four five six seven eight nine ten eleven", natural
    )
    assert " ".join(result) == "one two three four five six seven eight nine ten eleven"
    assert len(result[-1].split()) > 1
