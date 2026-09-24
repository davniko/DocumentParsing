from types import SimpleNamespace

import pytest

from document_ocr.synthesis.template_compiler.descendant import (
    _equipment_semantics_match,
    _render_equipment_receipt_binding,
)
from document_ocr.synthesis.template_compiler.host import SpanDraft, validate_binding_realizations


def equipment(size, kind="GENERAL_PURPOSE"):
    return {"sizeCategory": size, "typeCategory": kind}


def render(surface, old, new):
    binding = SimpleNamespace(
        target_paths=("documentPatch.containers",),
        dependency_paths=(),
        occurrences=(SimpleNamespace(slot_id="s", source_text=surface),),
    )
    return _render_equipment_receipt_binding(
        binding,
        source_target={"documentPatch": {"containers": old}},
        target={"documentPatch": {"containers": new}},
    ).replacements["s"]


@pytest.mark.parametrize(
    "surface",
    [
        "2 x 40HC CNTRS",
        "2 x 40' HIGH CUBE",
        "2 X HIGH CUBE 40' CONTAINER",
        "2 X 40' High Cube Standard Container",
        "2X40' CNTR(S)",
        "2X40HC",
        "TWO(2*40'HQ)",
        "TWO (40HCX2) CONTAINERS ONLY.",
        "2'40'HQ",
        "2 X 40H",
        "2 X 40'",
        "2X40HC CONTAINER(S)",
        "2X40' CNTR(S) S.T.C",
        "2 X DRY CONTAINERS 40'",
    ],
)
def test_every_receipt_grammar_projects_all_sampled_equipment(surface):
    result = render(
        surface,
        [{"typeDescription": "40GP" if surface == "2 X DRY CONTAINERS 40'" else "40HC"}] * 2,
        [
            equipment("FORTY_FIVE_FOOT_HIGH_CUBE"),
            equipment("TWENTY_FOOT_STANDARD_HEIGHT"),
        ],
    )
    assert "45" in result and "20" in result and "40" not in result
    assert (
        result.upper().count("1X")
        + result.upper().count("1 X")
        + result.upper().count("1*")
        + result.upper().count("X1")
        == 2
    )


def test_homogeneous_replacement_removes_stale_high_cube_words():
    result = render(
        "2 x 40' HIGH CUBE",
        [{"typeDescription": "40HC"}] * 2,
        [equipment("TWENTY_FOOT_STANDARD_HEIGHT")] * 2,
    )
    assert result == "2 x 20GP"


def test_explicit_standard_abbreviation_is_a_complete_receipt_type():
    from document_ocr.synthesis.container_semantics import review_source_equipment_surface

    observed = review_source_equipment_surface("20'STD", temperature_present=False)
    assert observed.size_category == "TWENTY_FOOT_STANDARD_HEIGHT"
    assert observed.type_category == "GENERAL_PURPOSE"
    old = [equipment("TWENTY_FOOT_STANDARD_HEIGHT")] * 17
    new = [equipment("FORTY_FOOT_HIGH_CUBE", "REFRIGERATED")] * 17
    assert render("17 X 20' STD FCL CONTAINERS STC", old, new) == (
        "17 X 40' HIGH CUBE REFRIGERATED FCL CONTAINERS STC"
    )
    with pytest.raises(ValueError, match="contradicts source count/type"):
        render("17 X 20' STD", [equipment("FORTY_FOOT_HIGH_CUBE")] * 17, new)


def test_parenthesized_loading_mode_retains_suffix_without_consuming_arbitrary_text():
    old = [equipment("FORTY_FOOT_HIGH_CUBE")] * 4
    new = [equipment("TWENTY_FOOT_STANDARD_HEIGHT")] * 4
    assert render("4X40'HC(FCL)", old, new) == "4X20'GP(FCL)"
    with pytest.raises(ValueError, match="source count/type identity"):
        render("3X40'HC(FCL)", old, new)
    for unknown in ("4X40'HC(UNKNOWN)", "4X40'HC(FCL OTHER)", "17 X 20' STD UNOWNED"):
        with pytest.raises(ValueError, match="unowned non-equipment"):
            render(unknown, old, new)


@pytest.mark.parametrize("description", ["1X 40 HIGH CUBE", "40' HIGH CUBE"])
def test_receipt_keeps_existing_complete_source_normalization_mutable(description):
    # The latest-label normalizer and the receipt grammar already classify these
    # complete observations. Partial-row reconciliation must not freeze them.
    assert (
        render(
            "1X40HC", [{"typeDescription": description}], [equipment("TWENTY_FOOT_STANDARD_HEIGHT")]
        )
        == "1X20GP"
    )


def test_count_wrapped_observed_row_has_same_semantics_as_its_receipt():
    old = [{"typeDescription": "20DRX1"}]
    new = [equipment("TWENTY_FOOT_STANDARD_HEIGHT")]
    assert "20DRX1" in render("SAY ONE (20DRX1) CONTAINER ONLY", old, new)
    with pytest.raises(ValueError, match="contradicts source count/type"):
        render("SAY ONE (40HCX1) CONTAINER ONLY", old, new)
    with pytest.raises(ValueError, match="source count/type identity"):
        render("SAY TWO (20DRX2) CONTAINERS ONLY", old, new)


def test_compact_reversed_count_is_proved_not_an_ignored_equipment_suffix():
    old = [{"typeDescription": "40HC"}]
    new = [equipment("TWENTY_FOOT_STANDARD_HEIGHT")]
    assert render("40HC1", old, new) == "20GP1"
    with pytest.raises(ValueError, match="source count/type identity"):
        render("40HC2", old, new)
    with pytest.raises(ValueError, match="contradicts source count/type"):
        render("40GP1", old, new)
    # An ISO type's numeric suffix is part of its code, not a container count.
    assert render("45G1", old, new) == "22G1"


def test_compact_reversed_mixed_inventory_remains_parseable():
    old = [{"typeDescription": "40HC"}] * 2
    new = [equipment("TWENTY_FOOT_STANDARD_HEIGHT"), equipment("FORTY_FIVE_FOOT_HIGH_CUBE")]
    value = render("40HC2", old, new)
    assert value == "20GP1 + 45HC1"
    assert render(value, new, new) == value


def test_matching_total_does_not_hide_contradictory_source_equipment():
    with pytest.raises(ValueError, match="contradicts source count/type"):
        render(
            "2X20GP",
            [{"typeDescription": "40HC"}] * 2,
            [equipment("TWENTY_FOOT_STANDARD_HEIGHT")] * 2,
        )
    with pytest.raises(ValueError, match="contradicts source count/type"):
        render(
            "1X40HC+1X20GP",
            [{"typeDescription": "40HC"}] * 2,
            [equipment("TWENTY_FOOT_STANDARD_HEIGHT")] * 2,
        )


@pytest.mark.parametrize("surface", ["1X40 FT. FULL CONTAINER", "1 X - 40'"])
def test_length_only_receipts_do_not_invent_a_dry_type(surface):
    old = [equipment("FORTY_FOOT_HIGH_CUBE", "REFRIGERATED")]
    assert "20" in render(surface, old, [equipment("TWENTY_FOOT_STANDARD_HEIGHT")])
    with pytest.raises(ValueError, match="contradicts source count/type"):
        render(surface, [equipment("TWENTY_FOOT_STANDARD_HEIGHT")], old)


def test_high_cube_only_receipt_checks_height_without_inventing_length_or_type():
    old = [equipment("FORTY_FOOT_HIGH_CUBE", "REFRIGERATED")]
    assert render("HI-CUBE", old, [equipment("TWENTY_FOOT_STANDARD_HEIGHT")]) == "STANDARD HEIGHT"
    with pytest.raises(ValueError, match="contradicts its source equipment"):
        render("HI-CUBE", [equipment("FORTY_FOOT_STANDARD_HEIGHT")], old)


def test_height_only_cells_never_expand_into_unowned_equipment_inventory_counts():
    old = [equipment("FORTY_FOOT_HIGH_CUBE")] * 2
    new = [
        equipment("TWENTY_FOOT_HIGH_CUBE"),
        equipment("FORTY_FIVE_FOOT_HIGH_CUBE", "REFRIGERATED"),
    ]
    assert render("HI-CUBE", old, new) == "HI-CUBE"
    assert (
        render("HI-CUBE", [{"typeDescription": "40H"}], [{"typeDescription": "40H"}]) == "HI-CUBE"
    )
    with pytest.raises(ValueError, match="mixed sampled heights"):
        render("HI-CUBE", old, [new[0], equipment("TWENTY_FOOT_STANDARD_HEIGHT")])
    with pytest.raises(ValueError, match="cardinality"):
        render("HI-CUBE", old, new[:1])
    with pytest.raises(ValueError, match="explicit sampled equipment heights"):
        render("HI-CUBE", old, [{}, {}])


def test_star_reversed_receipt_is_an_exact_counted_inventory_not_an_opaque_code():
    old = [equipment("FORTY_FOOT_HIGH_CUBE")] * 4
    new = [equipment("TWENTY_FOOT_STANDARD_HEIGHT")] * 4
    assert render("40HQ*4", old, new) == "20GP*4"
    with pytest.raises(ValueError, match="subset lacks"):
        render("40HQ*3", old, new)


def test_reviewed_ec_carrier_token_has_same_receipt_semantics_as_source_parser():
    old = [equipment("FORTY_FOOT_STANDARD_HEIGHT")]
    assert "20" in render("1 x 40EC", old, [equipment("TWENTY_FOOT_STANDARD_HEIGHT")])


@pytest.mark.parametrize("description", ["DC 4H", "DC 4H CY / FO"])
def test_quoted_h_height_marker_agrees_with_the_reviewed_carrier_equipment(description):
    assert "20" in render(
        "1 X 40'H DC CONTAINER",
        [{"typeDescription": description}],
        [equipment("TWENTY_FOOT_STANDARD_HEIGHT")],
    )
    with pytest.raises(ValueError, match="contradicts source count/type"):
        render(
            "1 X 40'H DC CONTAINER",
            [equipment("FORTY_FOOT_STANDARD_HEIGHT")],
            [equipment("TWENTY_FOOT_STANDARD_HEIGHT")],
        )


def test_dimensions_only_receipt_constrains_private_size_without_inventing_a_type():
    old = [{"typeDescription": "GENERAL PURPOSE CONT."}]
    surface = "1 CONT. 20'X8'6\""
    assert render(surface, old, old) == surface
    assert render(surface, old, [equipment("TWENTY_FOOT_STANDARD_HEIGHT")]) == surface
    with pytest.raises(ValueError, match="contradicts retained"):
        render(surface, old, [equipment("FORTY_FOOT_HIGH_CUBE")])
    # A complete row with an independently printed reefer type does not conflict
    # with a receipt that says only the dimensions.
    assert "40" in render(
        surface,
        [equipment("TWENTY_FOOT_STANDARD_HEIGHT", "REFRIGERATED")],
        [equipment("FORTY_FOOT_HIGH_CUBE", "REFRIGERATED")],
    )


@pytest.mark.parametrize("surface", ["1 CONT. 20'X7'6\"", "1 CONT. 45'X8'6\""])
def test_unsupported_dimensions_are_not_defaulted_to_standard_equipment(surface):
    with pytest.raises(ValueError, match="unreviewed type"):
        render(
            surface,
            [{"typeDescription": "GENERAL PURPOSE CONT."}],
            [equipment("TWENTY_FOOT_STANDARD_HEIGHT")],
        )


def test_iso_receipt_matches_decoded_size_not_the_first_two_digits():
    result = render(
        "2X45G1", [{"typeDescription": "40HC"}] * 2, [equipment("TWENTY_FOOT_STANDARD_HEIGHT")] * 2
    )
    assert "40" not in result and "45" not in result


@pytest.mark.parametrize("noun", ["TANK CONTAINER(S)", "TANKCONTAINER(S)", "TANKCONTAINERS"])
def test_iso_tank_receipts_match_spaced_and_fused_source_words(noun):
    surface = "2 X 20 FT ISO " + noun
    old = [{"typeDescription": "20 FT ISO TANK CONTAINER(S)"}] * 2
    assert "40" in render(surface, old, [equipment("FORTY_FOOT_STANDARD_HEIGHT")] * 2)
    with pytest.raises(ValueError, match="contradicts source count/type"):
        render(surface, [equipment("TWENTY_FOOT_STANDARD_HEIGHT")] * 2, old)


def test_iso_word_does_not_allow_arbitrary_non_equipment_prose():
    with pytest.raises(ValueError, match="unowned non-equipment wording"):
        render(
            "1 X 20 FT ISO TANK SHIPMENT",
            [equipment("TWENTY_FOOT_STANDARD_HEIGHT")],
            [equipment("TWENTY_FOOT_STANDARD_HEIGHT")],
        )


def test_rfh_receipt_preserves_the_unknown_source_height():
    source = [{"typeDescription": "40'RFH"}]
    assert render("1 X 40'RFH", source, source) == "1 X 40'RFH"
    for size in ("FORTY_FOOT_STANDARD_HEIGHT", "FORTY_FOOT_HIGH_CUBE"):
        assert render("1 X 40'RFH", source, [equipment(size, "REFRIGERATED")]) == "1 X 40'RFH"
    with pytest.raises(ValueError, match="contradicts retained"):
        render("1 X 40'RFH", source, [equipment("FORTY_FOOT_HIGH_CUBE")])


def test_overlapping_partial_receipt_terms_have_distinct_source_owners():
    result = render(
        "1X40'+1X40HC",
        [{"typeDescription": "40HC"}, {"typeDescription": "40GP"}],
        [equipment("TWENTY_FOOT_STANDARD_HEIGHT")] * 2,
    )
    assert "2X20" in result


@pytest.mark.parametrize(
    "surface", ["2 ctnrs", "2 cntrs", "2 CONTAINER(S)", "TWO CNTRS", "TWO (2) CONTAINER(S)"]
)
def test_count_only_receipt_does_not_constrain_equipment_types(surface):
    assert (
        render(
            surface,
            [{"typeDescription": "40HC"}] * 2,
            [
                equipment("FORTY_FIVE_FOOT_HIGH_CUBE"),
                equipment("TWENTY_FOOT_STANDARD_HEIGHT"),
            ],
        )
        == surface
    )


def test_separate_source_subsets_follow_their_actual_container_indices():
    old = [{"typeDescription": "20DC"}] * 4 + [{"typeDescription": "40HC"}]
    new = [equipment("FORTY_FIVE_FOOT_HIGH_CUBE")] * 3 + [
        equipment("TWENTY_FOOT_STANDARD_HEIGHT"),
        equipment("FORTY_FOOT_STANDARD_HEIGHT"),
    ]
    assert render("4 x 20DC", old, new) == "3 x 45HC + 1 x 20DC"
    assert render("1 x 40HC", old, new) == "1 x 40GP"
    with pytest.raises(ValueError, match="exact source count/type"):
        render("3 x 20DC", old, new)


def test_word_count_and_spelled_length_are_both_projected():
    assert (
        render(
            "ONE FORTY FT.HQ CONTAINER ONLY.",
            [{"typeDescription": "40HC"}],
            [equipment("TWENTY_FOOT_STANDARD_HEIGHT")],
        )
        == "1X20GP CONTAINER ONLY."
    )


def test_joint_summary_recounts_each_new_semantic_group():
    old = [{"typeDescription": "40HC"}] * 4 + [{"typeDescription": "20DC"}]
    new = [equipment("TWENTY_FOOT_STANDARD_HEIGHT")] * 2 + [
        equipment("FORTY_FIVE_FOOT_HIGH_CUBE")
    ] * 3
    assert render("4X40'HC+1X20'DC FCL CNTR(S)", old, new) == "2X20'GP + 3X45'HC FCL CNTR(S)"


def test_thermal_height_is_not_lost_by_using_a_reefer_abbreviation():
    result = render(
        "2 x 40' RH",
        [{"typeDescription": "40RH"}] * 2,
        [equipment("FORTY_FOOT_HIGH_CUBE", "REFRIGERATED")] * 2,
    )
    assert result == "2 x 40' RH"
    assert _equipment_semantics_match(equipment("FORTY_FOOT_HIGH_CUBE", "REFRIGERATED"), "40' RH")


def test_high_cube_receipt_is_height_only_and_stays_unchanged_for_reefer():
    old = [{"typeDescription": "40RF96"}]
    new = [equipment("FORTY_FOOT_HIGH_CUBE", "REFRIGERATED")]
    assert render("01X40'HC", old, new) == "01X40'HC"


def test_length_only_receipt_does_not_require_unprinted_type_or_height():
    old = [{"typeDescription": "40 FT"}, {"typeDescription": "40 FT"}]
    new = [equipment("FORTY_FOOT_HIGH_CUBE", "REFRIGERATED"), old[1]]
    assert render("2X40 FT", old, new) == "2X40 FT"


def test_unchanged_iso_and_reverse_count_source_spellings_are_preserved():
    rows = [equipment("FORTY_FOOT_HIGH_CUBE")] * 4
    assert render("40HQ*4", rows, rows) == "40HQ*4"
    assert render("45G1", rows, rows) == "45G1"


def test_split_count_multipliers_are_not_misclassified_as_mixed_inventory_types():
    from document_ocr.synthesis.template_compiler.anonymous_equipment import _observation

    assert _observation("2 x") == (2, None)
    assert _observation("03*") == (3, None)


def test_unproven_type_or_count_alias_fails_explicitly():
    old = [{"typeDescription": "40HC"}] * 2
    new = [equipment("FORTY_FIVE_FOOT_HIGH_CUBE")] * 2
    with pytest.raises(ValueError, match="word/digit"):
        render("THREE(2X40HC)", old, new)
    with pytest.raises(ValueError, match="complete target"):
        render("2X40HC", old, [new[0], {}])


def test_split_count_and_description_slots_do_not_duplicate_the_count():
    old = [{"typeDescription": "20ST"}]
    new = [equipment("FORTY_FIVE_FOOT_HIGH_CUBE")]
    assert render("1 x", old, new) == "1 x"
    assert render("20ST", old, new) == "45HC"


def test_description_only_slot_exposes_a_mixed_inventory_explicitly():
    old = [{"typeDescription": "40HC"}] * 2
    new = [equipment("FORTY_FIVE_FOOT_HIGH_CUBE"), equipment("TWENTY_FOOT_STANDARD_HEIGHT")]
    assert render("40' HIGH CUBE", old, new) == "MIXED (1X45HC + 1X20GP)"
    assert render("40'HC", old, new) == "MIXED (1X45'HC + 1X20'GP)"


def test_printed_count_cannot_exceed_the_complete_owned_inventory():
    with pytest.raises(ValueError, match="exact source count/type identity"):
        render("02 X 40HC", [{"typeDescription": "40HC"}], [equipment("FORTY_FIVE_FOOT_HIGH_CUBE")])


def test_receipt_projection_cannot_discard_non_equipment_facts():
    with pytest.raises(ValueError, match="unowned non-equipment"):
        render(
            "2 X 40HC INCLUDING EXPLOSIVES",
            [{"typeDescription": "40HC"}] * 2,
            [equipment("FORTY_FIVE_FOOT_HIGH_CUBE")] * 2,
        )


def test_unknown_exact_printed_equipment_can_remain_unknown_without_new_labels():
    old = [{"containerNumber": "OLD", "typeDescription": "20HO"}]
    new = [{"containerNumber": "NEW", "typeDescription": "20HO"}]
    assert render("1X20HO", old, new) == "1X20HO"
    new[0].update(sizeCategory="TWENTY_FOOT_STANDARD_HEIGHT", typeCategory="GENERAL_PURPOSE")
    with pytest.raises(ValueError, match="unowned non-equipment"):
        render("1X20HO", old, new)


def test_preserved_receipt_description_cannot_hide_a_changed_explicit_category():
    old = [{"typeDescription": "40HR"}]
    new = [{"typeDescription": "40HR", **equipment("FORTY_FOOT_HIGH_CUBE")}]
    with pytest.raises(ValueError, match="contradicts source count/type"):
        render("1X40HR", old, new)


def test_partial_length_subset_keeps_missing_type_labels_missing():
    old = [equipment("FORTY_FOOT_HIGH_CUBE"), {"typeDescription": "20'"}]
    new = [equipment("FORTY_FIVE_FOOT_HIGH_CUBE"), {"typeDescription": "20'"}]
    assert render("1 X 20'", old, new) == "1 X 20'"
    assert "45" in render("1 X 40HC", old, new)
    new[1]["typeDescription"] = "40'"
    with pytest.raises(ValueError, match="complete target size/type"):
        render("1 X 20'", old, new)


@pytest.mark.parametrize(
    "surface,old,physical",
    [
        (
            "4 X 40' CONTAINERS",
            [{"containerNumber": str(i)} for i in range(4)],
            [equipment("FORTY_FOOT_HIGH_CUBE")] * 4,
        ),
        (
            "1X40HR",
            [{"typeDescription": "40RQ"}],
            [equipment("FORTY_FOOT_HIGH_CUBE", "REFRIGERATED")],
        ),
        (
            "10X20FT",
            [{"typeDescription": "20ST"}] * 9 + [{}],
            [equipment("TWENTY_FOOT_STANDARD_HEIGHT")] * 10,
        ),
        ("1X40HC", [{"typeDescription": "CY/FO"}], [equipment("FORTY_FOOT_HIGH_CUBE")]),
    ],
)
def test_whole_inventory_receipt_supplies_missing_physical_evidence(surface, old, physical):
    from copy import deepcopy

    before = deepcopy(old)
    assert render(surface, old, old) == surface
    assert render(surface, old, physical) == surface
    # Publication keeps the original observable shape, not internal equipment.
    assert render(surface, old, before) == surface
    assert old == before


@pytest.mark.parametrize(
    "surface,old,bad",
    [
        (
            "1X40HR",
            [{"typeDescription": "40RQ"}],
            [equipment("FORTY_FOOT_STANDARD_HEIGHT", "REFRIGERATED")],
        ),
        ("1X40HR", [{}], [equipment("FORTY_FOOT_HIGH_CUBE")]),
        ("1X40'", [{}], [equipment("TWENTY_FOOT_STANDARD_HEIGHT")]),
        ("1X40HC", [{}], [equipment("FORTY_FOOT_HIGH_CUBE")] * 2),
    ],
)
def test_retained_receipt_constrains_every_private_equipment_candidate(surface, old, bad):
    with pytest.raises(ValueError, match="contradicts retained whole-inventory receipt"):
        render(surface, old, bad)


@pytest.mark.parametrize(
    "surface,old",
    [
        ("1X40HR", [{"typeDescription": "GENERAL PURPOSE CONT"}]),
        ("1X40HC", [{"typeDescription": "20'"}]),
        ("1X20GP", [{"typeDescription": "40RQ"}]),
    ],
)
def test_partial_observation_cannot_hide_a_real_receipt_contradiction(surface, old):
    with pytest.raises(ValueError, match="contradicts source count/type"):
        render(surface, old, old)


def test_unknown_rows_cannot_be_assigned_to_a_subset_or_mixed_inventory():
    with pytest.raises(ValueError, match="source count/type identity"):
        render("1X40HC", [{}, {}], [{}, {}])
    with pytest.raises(ValueError, match="contradicts source count/type"):
        render("1X40HC+1X20GP", [{}, {}], [{}, {}])


def test_part_container_qualifier_survives_inventory_projection():
    result = render(
        "SAY PART OF ONE (1X40'HC) CONTAINER ONLY.",
        [equipment("FORTY_FOOT_HIGH_CUBE")],
        [equipment("TWENTY_FOOT_STANDARD_HEIGHT")],
    )
    assert result.startswith("SAY PART OF ONE")
    assert "20" in result and "40" not in result


@pytest.mark.parametrize("surface", ["1 X 40 FT REEF", "01 X 40 REEFER", "1 x 40' REFRIGERATED"])
def test_prose_reefer_receipt_does_not_invent_a_standard_height(surface):
    result = render(
        surface,
        [equipment("FORTY_FOOT_HIGH_CUBE", "REFRIGERATED")],
        [equipment("FORTY_FIVE_FOOT_HIGH_CUBE", "REFRIGERATED")],
    )
    assert "45" in result and "40" not in result
    with pytest.raises(ValueError, match="contradicts source count/type"):
        render(
            surface, [equipment("FORTY_FOOT_HIGH_CUBE")], [equipment("TWENTY_FOOT_STANDARD_HEIGHT")]
        )


@pytest.mark.parametrize(
    "surface,description,valid",
    [
        ("2X20GP", "40HC", False),
        ("2X40HC", "40HC", True),
        ("3X40HC", "40HC", False),
        ("2 X 20'", "20'", True),
        ("2 CNTRS", None, True),
    ],
)
def test_compiler_checks_source_receipt_before_certification(surface, description, valid):
    draft = SpanDraft(
        draft_id="receipt",
        logical_key="agent:receipt",
        render_mode="deterministic_derived",
        value_kind="equipment",
        group_kind="container",
        group_key="container:all",
        target_paths=(),
        derivation="equipment_receipt",
        dependency_paths=("documentPatch.containers",),
        dependency_bindings=(),
        char_start=0,
        char_end=len(surface),
        source_text=surface,
        evidence_origin="host_verified_agent_proposal",
        render_policy="natural_text",
        rationale="Receipt source semantics must be proved at compile time.",
    )
    target = {
        "documentPatch": {
            "containers": [
                {"typeDescription": description} if description else {} for _ in range(2)
            ]
        }
    }
    if valid:
        validate_binding_realizations(raw=surface, drafts=(draft,), source_target=target)
    else:
        with pytest.raises(ValueError, match="binding realization contract violations"):
            validate_binding_realizations(raw=surface, drafts=(draft,), source_target=target)
