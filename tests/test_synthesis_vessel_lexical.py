from __future__ import annotations

from document_ocr.synthesis.generators import DeterministicStream
from document_ocr.synthesis.vessel_lexical import (
    fit_character_ngram_vessel_renderer,
    lexical_realism_metrics,
    vessel_name_fit_exclusion_reason,
)

_FIT_NAMES = (
    "OCEAN SPIRIT",
    "NORTH STAR",
    "SILVER HORIZON",
    "PACIFIC VOYAGER",
    "MARINE VENTURE",
    "GOLDEN DAWN",
    "ATLANTIC BREEZE",
    "SOUTHERN LIGHT",
    "BLUEWATER NAVIGATOR",
    "CORAL MARINER",
    "ISLAND DISCOVERY",
    "MAJESTIC WAVE",
    "EASTERN VOYAGER",
    "WESTERN MARINER",
    "NEPTUNE HORIZON",
    "OCEAN NAVIGATOR",
    "PACIFIC BREEZE",
    "MARINE DISCOVERY",
    "SILVER VENTURE",
    "GOLDEN SPIRIT",
    "NORTHERN STAR",
    "SOUTHERN DAWN",
)


def test_character_ngram_renderer_is_deterministic_and_exact() -> None:
    renderer = fit_character_ngram_vessel_renderer(names=_FIT_NAMES, order=3)
    stream = DeterministicStream(seed=11, namespace="vessel-test", identity="row")

    first = renderer.render(
        word_count=2,
        character_count=15,
        digit_count=0,
        case_style="upper",
        stream=stream,
    )
    second = renderer.render(
        word_count=2,
        character_count=15,
        digit_count=0,
        case_style="upper",
        stream=stream,
    )

    assert first == second
    assert first[1] == "character_ngram"
    assert len(first[0]) == 15
    assert len(first[0].split()) == 2
    assert first[0].isupper()
    assert renderer.accepts(first[0])
    assert renderer.bits_per_character(_FIT_NAMES) > 0


def test_lexical_realism_is_maximal_for_identical_distributions() -> None:
    identical = lexical_realism_metrics(
        reference_names=_FIT_NAMES,
        generated_names=_FIT_NAMES,
    )
    distorted = lexical_realism_metrics(
        reference_names=_FIT_NAMES,
        generated_names=tuple("Z" * len(value.replace(" ", "")) for value in _FIT_NAMES),
    )

    assert identical["lexicalRealismScore"] == 1.0
    assert float(identical["lexicalRealismScore"]) > float(distorted["lexicalRealismScore"])


def test_fit_filter_excludes_voyage_contamination_and_field_delimiters() -> None:
    assert vessel_name_fit_exclusion_reason("MOLIVA/TES01S26") == "compound_or_field_delimiter"
    assert (
        vessel_name_fit_exclusion_reason("GULBENIZ A-0EL35W1TK")
        == "contains_numeric_or_voyage_suffix"
    )
    assert vessel_name_fit_exclusion_reason("PACIFIC HORIZON") is None
