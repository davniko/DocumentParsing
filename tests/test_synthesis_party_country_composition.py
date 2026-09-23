from types import SimpleNamespace as NS

import pytest

from document_ocr.synthesis.template_compiler import descendant as renderer


def binding(surface="OLD CO (EGYPT", *, role="consignee"):
    root = f"documentPatch.parties.{role}"
    return NS(
        target_paths=(root + ".country", root + ".name"),
        occurrences=(NS(slot_id="s1", source_text=surface),),
    )


def target(name, country):
    return {"documentPatch": {"parties": {"consignee": {"name": name, "country": country}}}}


@pytest.mark.parametrize("ending", ["", ")", "),", ")."])
@pytest.mark.parametrize("already_annotated", [False, True])
def test_frozen_party_name_and_country_use_source_parenthesis_ownership(ending, already_annotated):
    name = "New Logistics Ltd" + (" (France)" if already_annotated else "")
    output = renderer._party_country_annotation_output(
        binding("OLD CO (EGYPT" + ending),
        target("Old Co (Egypt)", "Egypt"),
        target(name, "France"),
    )
    assert output.replacements == {"s1": "NEW LOGISTICS LTD (FRANCE" + ending}
    assert output.canonical_value == ["France", name]
    assert renderer._string_semantics_match(name, output.replacements["s1"])


def test_multiline_country_annotation_preserves_lines_and_complete_target():
    output = renderer._party_country_annotation_output(
        binding("OLD\nCO (EGYPT"),
        target("Old Co (Egypt)", "Egypt"),
        target("New Company (Trading) Ltd", "Tanzania, United Republic of"),
    )
    surface = output.replacements["s1"]
    assert surface.count("\n") == 1
    assert renderer._string_semantics_match("New Company (Trading) Ltd", surface)
    assert renderer._string_semantics_match("Tanzania, United Republic of", surface)
    assert surface.endswith("OF")  # Closing parenthesis belongs to the literal outside this slot.


def test_source_country_can_be_an_annotation_outside_the_source_name_label():
    output = renderer._party_country_annotation_output(
        binding(), target("Old Co", "Egypt"), target("New Co", "France")
    )
    assert output.replacements == {"s1": "NEW CO (FRANCE"}


@pytest.mark.parametrize("surface", ["OLD CO (EGYPT) REF 99", "OLD CO / EGYPT", "EGYPT"])
def test_unowned_prose_or_unproven_composition_does_not_enter_deterministic_route(surface):
    assert (
        renderer._party_country_annotation_output(
            binding(surface), target("Old Co (Egypt)", "Egypt"), target("New Co", "France")
        )
        is None
    )


def test_country_and_name_from_different_parties_are_not_composed():
    b = binding()
    b.target_paths = ("documentPatch.parties.shipper.country", b.target_paths[1])
    assert (
        renderer._party_country_annotation_output(
            b, target("Old Co (Egypt)", "Egypt"), target("New Co", "France")
        )
        is None
    )


def test_distinct_country_inside_a_company_name_is_not_rewritten():
    output = renderer._party_country_annotation_output(
        binding(), target("Old Co (Egypt)", "Egypt"), target("New Co (Japan)", "France")
    )
    assert output.replacements == {"s1": "NEW CO (JAPAN) (FRANCE"}
