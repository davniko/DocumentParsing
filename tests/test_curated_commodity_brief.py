"""Commodity context preserves official taxonomy without making it a cargo fact."""

import pytest

from document_ocr.synthesis.curated_commodity_brief import commodity_brief
from document_ocr.synthesis.curated_scenarios import GoodsIdentity, HSClassificationContext
from document_ocr.synthesis.curated_wording import RENDERED_REVIEW_PROMPT, WORDING_PROMPT


def identity(phrase, role="commercial", hs6="720840"):
    return GoodsIdentity(
        hs6=hs6,
        phrase=phrase,
        phrase_role=role,
        classification=HSClassificationContext(
            authority="test-registry",
            chapter_code=hs6[:2],
            chapter_description="Chapter",
            heading_code=hs6[:4],
            heading_description="Heading",
            description_path=(phrase,),
            same_heading_alternatives=(),
            immediate_national_children=(),
        ),
    )


def test_official_exclusion_is_context_not_an_additional_physical_form():
    official = "Flat-rolled steel products, not in coils, with patterns in relief"
    result = commodity_brief(
        [identity(official, "classification_context")],
        physical_form="HOT-ROLLED STEEL IN COILED FORM, NOT FLAT SHEETS",
    )
    context, required = result.split("REQUIRED PRODUCT FACTS")
    assert official in context
    assert official not in required
    assert "Physical form: HOT-ROLLED STEEL IN COILED FORM, NOT FLAT SHEETS" in required
    assert "720840" not in result  # Code is printed by the host, not the wording model.


@pytest.mark.parametrize("phrase", ["FROZEN BONELESS BEEF", "ACETONE"])
def test_reviewed_thermal_and_dg_commercial_facts_remain_authoritative(phrase):
    value = identity(phrase)
    result = commodity_brief([value])
    assert "CLASSIFICATION REFERENCES" not in result
    assert result.endswith(
        "REQUIRED PRODUCT FACTS — authoritative for this shipment\nItem 1: " + phrase
    )
    assert "phrase_role" not in value.model_dump()


@pytest.mark.parametrize("form", [None, "", " \n\t", 7])
def test_classification_requires_explicit_nonempty_form(form):
    with pytest.raises(ValueError, match="physical form"):
        commodity_brief(
            [identity("Official nomenclature", "classification_context")], physical_form=form
        )


@pytest.mark.parametrize("phrase", [None, "", " \t", 4])
def test_invalid_phrase_cannot_be_silently_masked(phrase):
    corrupted = identity("Original").model_copy(update={"phrase": phrase})
    with pytest.raises(ValueError, match="requires nonempty wording"):
        commodity_brief([corrupted], physical_form="SOLID")


def test_empty_identity_sequence_fails():
    with pytest.raises(ValueError, match="at least one identity"):
        commodity_brief([])


def test_mixed_roles_preserve_every_identity_and_ordinal_once():
    result = commodity_brief(
        [identity("FIRST"), identity("SECOND", "classification_context"), identity("THIRD")],
        physical_form="POWDER",
    )
    assert result.count("Item 1: FIRST") == 1
    assert result.count("Item 2: SECOND") == 1
    assert result.count("Item 3: THIRD") == 1
    context, required = result.split("REQUIRED PRODUCT FACTS")
    assert "Item 2: SECOND" in context
    assert "Item 1: FIRST" in required and "Item 3: THIRD" in required
    assert required.index("FIRST") < required.index("THIRD")


def test_unknown_role_fails_even_when_model_validation_was_bypassed():
    invalid = identity("Original").model_copy(update={"phrase_role": "unreviewed"})
    with pytest.raises(ValueError, match="unknown commodity phrase role"):
        commodity_brief([invalid])


def test_prompt_has_no_conflicting_accounting_generation_instructions():
    assert "Classification\nreferences" in WORDING_PROMPT
    assert "Packing capacities and fill amounts remain host facts" in WORDING_PROMPT
    assert "Any item masses or package contents must agree" not in WORDING_PROMPT
    assert "newly generated quantities" not in WORDING_PROMPT
    assert "retaining\nevery supplied product attribute" not in WORDING_PROMPT
    assert "without line or\ncharacter quotas" in WORDING_PROMPT
    assert (
        "tariff alternatives and exclusions are not extra cargo requirements"
        in RENDERED_REVIEW_PROMPT
    )
