import pytest

from document_ocr.synthesis.container_semantics import review_source_equipment_surface


@pytest.mark.parametrize(
    "code,size",
    [
        ("22G1", "TWENTY_FOOT_STANDARD_HEIGHT"),
        ("25G0", "TWENTY_FOOT_HIGH_CUBE"),
        ("42G1", "FORTY_FOOT_STANDARD_HEIGHT"),
        ("45G1", "FORTY_FOOT_HIGH_CUBE"),
        ("45 G1", "FORTY_FOOT_HIGH_CUBE"),
        ("55G1", "FORTY_FIVE_FOOT_HIGH_CUBE"),
    ],
)
def test_iso_dimensions_are_not_literal_feet(code, size):
    value = review_source_equipment_surface(code, temperature_present=False)
    assert (value.size_category, value.type_category) == (size, "GENERAL_PURPOSE")
    assert value.review_rule == "iso_6346_size_type"


@pytest.mark.parametrize("code", ["45G4", "45R7", "45K9", "95G1", "40G1"])
def test_unassigned_or_unrepresentable_iso_code_is_not_guessed_from_carrier_syntax(code):
    value = review_source_equipment_surface(code, temperature_present=False)
    assert value.resolution == "unresolved_source_surface"
    assert value.size_category is None and value.type_category is None


def test_iso_thermal_detail_and_carrier_shorthand_are_distinct():
    iso = review_source_equipment_surface("45R1", temperature_present=True)
    carrier = review_source_equipment_surface("40RH", temperature_present=True)
    assert iso.type_category == "REFRIGERATED_AND_HEATED"
    assert carrier.type_category == "REFRIGERATED"
    assert iso.size_category == carrier.size_category == "FORTY_FOOT_HIGH_CUBE"
    assert iso.thermal_operation == carrier.thermal_operation == "active"


def test_expected_temperature_type_cannot_make_dry_text_match():
    from document_ocr.synthesis.template_compiler.descendant import _equipment_semantics_match

    expected = {"sizeCategory": "FORTY_FOOT_HIGH_CUBE", "typeCategory": "REFRIGERATED"}
    assert not _equipment_semantics_match(expected, "40HC")
    assert not _equipment_semantics_match(expected, "45G1")
    assert _equipment_semantics_match(expected, "45R0")
    assert not _equipment_semantics_match(expected, "40 GP", observed_temperature=True)
    assert _equipment_semantics_match(expected, "40HC", observed_temperature=True)


@pytest.mark.parametrize(
    "surface,kind",
    [
        ("GP40", "GENERAL_PURPOSE"),
        ("20UT", "OPEN_TOP"),
        ("45RF", "REFRIGERATED"),
        ("RT40", "REFRIGERATED_AND_HEATED"),
    ],
)
def test_compact_explicit_type_group_is_decoded_without_substring_guesses(surface, kind):
    value = review_source_equipment_surface(surface, temperature_present=False)
    assert value.type_category == kind


def test_iso_candidates_keep_code_width_and_meaning():
    from document_ocr.synthesis.template_compiler.descendant import (
        _equipment_semantics_match,
        _equipment_surface_candidates,
    )

    expected = {"sizeCategory": "FORTY_FIVE_FOOT_HIGH_CUBE", "typeCategory": "REFRIGERATED"}
    candidate = _equipment_surface_candidates(expected, "45G1")[0]
    assert candidate == "55R0"
    assert _equipment_semantics_match(expected, candidate)


@pytest.mark.parametrize(
    "surface,size,kind",
    [
        ("DC 4H", "FORTY_FOOT_HIGH_CUBE", "GENERAL_PURPOSE"),
        ("DC4H", "FORTY_FOOT_HIGH_CUBE", "GENERAL_PURPOSE"),
        ("DC 4H CY/FO", "FORTY_FOOT_HIGH_CUBE", "GENERAL_PURPOSE"),
        ("40HO", "FORTY_FOOT_HIGH_CUBE", "OPEN_TOP"),
        ("40 HO", "FORTY_FOOT_HIGH_CUBE", "OPEN_TOP"),
        ("40OT OOG", "FORTY_FOOT_STANDARD_HEIGHT", "OPEN_TOP"),
        ("20 OT OOG", "TWENTY_FOOT_STANDARD_HEIGHT", "OPEN_TOP"),
    ],
)
def test_documented_carrier_surface_constrains_physics_without_label_enrichment(
    surface, size, kind
):
    from document_ocr.synthesis.container_semantics import partial_equipment_constraint

    container = {"containerNumber": "ABCU1234567", "typeDescription": surface}
    assert partial_equipment_constraint(container) == (None, kind, size)
    assert container == {"containerNumber": "ABCU1234567", "typeDescription": surface}
    resolved = review_source_equipment_surface(surface, temperature_present=False)
    assert resolved.review_rule == "documented_complete_carrier_equipment_surface"


@pytest.mark.parametrize("surface", ["20HO", "DC 5H", "40OH", "40MA", "40HO SPECIAL"])
def test_other_carrier_codes_are_not_inferred_from_the_new_spellings(surface):
    resolved = review_source_equipment_surface(surface, temperature_present=False)
    assert resolved.resolution == "unresolved_source_surface"


@pytest.mark.parametrize(
    "surface",
    ["20 FT ISO TANK CONTAINER(S)", "20 FT ISO TANKCONTAINER(S)", "20 FT ISO TANKCONTAINERS"],
)
def test_tank_noun_word_boundary_cannot_change_equipment_type(surface):
    resolved = review_source_equipment_surface(surface, temperature_present=False)
    assert resolved.size_category == "TWENTY_FOOT_STANDARD_HEIGHT"
    assert resolved.type_category == "PRESSURIZED_TANK"


def test_tank_word_boundary_normalization_is_not_a_substring_rewrite():
    resolved = review_source_equipment_surface("20 TANKCONTAINERIZATION", temperature_present=False)
    assert resolved.resolution == "unresolved_source_surface"


@pytest.mark.parametrize("length", ["20", "40", "45"])
def test_rfh_proves_refrigeration_but_not_height_or_unprinted_labels(length):
    from document_ocr.synthesis.container_semantics import partial_equipment_constraint

    container = {"containerNumber": "TRKU1100724", "typeDescription": length + "'RFH"}
    assert partial_equipment_constraint(container) == (length, "REFRIGERATED", None)
    assert container == {"containerNumber": "TRKU1100724", "typeDescription": length + "'RFH"}
    resolved = review_source_equipment_surface(length + "'RFH", temperature_present=False)
    assert resolved.type_category == "REFRIGERATED"
    assert resolved.size_category is None
    assert resolved.resolution == "unresolved_source_surface"


def test_equipment_memoization_retains_temperature_context_and_immutable_results():
    from dataclasses import FrozenInstanceError

    cold = review_source_equipment_surface("40'RFH", temperature_present=False)
    active = review_source_equipment_surface("40'RFH", temperature_present=True)
    assert cold.thermal_operation == "not_indicated"
    assert active.thermal_operation == "active"
    assert review_source_equipment_surface("40'RFH", temperature_present=True) is active
    with pytest.raises(FrozenInstanceError):
        active.type_category = "GENERAL_PURPOSE"


def test_complete_40rf96_code_is_high_cube_reefer_not_a_two_digit_seal():
    resolved = review_source_equipment_surface("40RF96", temperature_present=True)
    assert resolved.resolution == "reviewed_source_grammar"
    assert resolved.size_category == "FORTY_FOOT_HIGH_CUBE"
    assert resolved.type_category == "REFRIGERATED"
    assert resolved.thermal_operation == "active"
