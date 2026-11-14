import copy

from document_ocr.labeling_agents.source_casing import (
    PrintedCaseIndex,
    recover_case,
    selected_fields,
)


def test_literal_case_retains_target_spacing_and_punctuation():
    value = "MOULDING PATTERNS, ACME LTD."
    result = recover_case(value, PrintedCaseIndex("Mouldi\nng Patterns, Acme\nLtd."))
    assert result["value"] == "Moulding Patterns, Acme Ltd."
    assert result["value"].upper() == value


def test_unicode_expansion_is_reversible_without_dropping_characters():
    assert recover_case("STRASSE 7", PrintedCaseIndex("Straße 7"))["value"] == "Straße 7"
    assert not PrintedCaseIndex("Straße").variants("S S")


def test_turkish_combining_dot_preserves_exact_printed_casing():
    assert (
        recover_case("MAKİNA TİCARET", PrintedCaseIndex("Maki̇na Ti̇caret"))["value"]
        == "Maki̇na Ti̇caret"
    )


def test_fragmented_combining_dot_words_remain_idempotent():
    value = "Deposito 14, Limani̇şletme Si̇tesi̇, Örnekdefter Sokak 8, Beyomu İlçesi̇, 34425, Turkey Istanbul"
    scope = PrintedCaseIndex("Deposito 14, Limani̇şletme Si̇tesi̇,\nÖrnekdefter Sokak 8, Beyomu İlçesi̇, 34425, Turkey\nPOSTAL CODE: Istanbul")
    for supplied in (value, value.upper()):
        assert recover_case(supplied, scope, owned=scope)['value'] == value


def test_combining_marks_do_not_hide_conflicting_word_case():
    value = "Maki̇na, 14"
    scope = PrintedCaseIndex("Maki̇na\nMAKİNA\nPOSTAL: 14")
    assert 'unresolved' in recover_case(value, scope, owned=scope)


def test_multiple_printed_cases_require_review_not_first_match():
    result = recover_case("ACME LTD", PrintedCaseIndex("Acme Ltd\nACME LTD"))
    assert "unresolved" in result


def test_owned_occurrence_beats_different_case_in_another_role():
    result = recover_case(
        "ACME LTD", PrintedCaseIndex("Acme Ltd\nACME LTD"), owned=PrintedCaseIndex("ACME LTD")
    )
    assert result["value"] == "ACME LTD"


def test_fragmented_owned_occurrence_beats_contiguous_other_role():
    result = recover_case(
        "ACME LTD",
        PrintedCaseIndex("Acme Ltd\nACME\nPHONE: 123\nLTD"),
        owned=PrintedCaseIndex("ACME\nPHONE: 123\nLTD"),
    )
    assert result["value"] == "ACME LTD"


def test_context_resolves_common_word_without_borrowing_boilerplate_case():
    result = recover_case(
        "PACKED IN DRUMS BATCH 42",
        PrintedCaseIndex("Packed in drums\nUN1234\nBatch 42\nIN WITNESS WHEREOF"),
    )
    assert result["value"] == "Packed in drums Batch 42"


def test_conflicting_contexts_are_not_silently_replaced_by_another_match():
    result = recover_case("ALPHA BETA GAMMA", PrintedCaseIndex("Alpha BETA\nbeta Gamma"))
    assert "unresolved" in result


def test_email_is_not_a_company_or_country_occurrence():
    result = recover_case(
        "LOGISTICS GATE", PrintedCaseIndex("LOGISTICS GATE\nlogisticsgate@example.com")
    )
    assert result["value"] == "LOGISTICS GATE"
    assert PrintedCaseIndex("msc.egypt@msc.com").variants("EGYPT") == ()


def test_absent_text_fails_without_guessing_a_spelling():
    assert "unresolved" in recover_case("ACME SUPPLY", PrintedCaseIndex("Acme Supplies"))


def test_exact_owned_field_does_not_index_unrelated_document_text():
    document = PrintedCaseIndex("ACME LTD\nOther text")
    assert (
        recover_case("ACME LTD", document, owned=PrintedCaseIndex("Acme Ltd"))["value"]
        == "Acme Ltd"
    )
    assert not document._prepared


def test_case_migration_preserves_existing_punctuation_even_between_fragments():
    result = recover_case("GROUP, ALEXANDRIA-EGYPT", PrintedCaseIndex("Group\nAlexandria Egypt"))
    assert result["value"] == "Group, Alexandria-Egypt"


def test_selected_fields_exclude_canonical_values_and_other_text():
    target = {
        "parties": {
            "shipper": {
                "name": "Acme",
                "addressLine": "Street 1",
                "country": "EGYPT",
                "contactDetails": {"contactName": "SAM"},
            },
            "notifyParties": [{"name": "Other"}],
        },
        "goodsItemDetails": [
            {
                "description": "Bolts",
                "handlingInstructions": ["KEEP DRY"],
                "origin": {"name": "EGYPT"},
            }
        ],
    }
    previous = copy.deepcopy(target)
    assert dict(selected_fields(target)) == {
        "/parties/shipper/name": "Acme",
        "/parties/shipper/addressLine": "Street 1",
        "/parties/notifyParties/0/name": "Other",
        "/goodsItemDetails/0/description": "Bolts",
    }
    assert target == previous
