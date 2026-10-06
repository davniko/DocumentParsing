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
        ("L5G1", "FORTY_FIVE_FOOT_HIGH_CUBE"),
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


@pytest.mark.parametrize("source", ["22GO", "22 GO", "42GO", "45RO", "L5GO", "25HO"])
def test_iso_ocr_zero_aliases_round_trip_without_changing_type_detail(source):
    from document_ocr.synthesis.container_semantics import iso_equipment_surface

    canonical = source[:-1] + "0"
    original = review_source_equipment_surface(source, temperature_present=False)
    zero = review_source_equipment_surface(canonical, temperature_present=False)
    assert (original.size_category, original.type_category) == (
        zero.size_category,
        zero.type_category,
    )
    assert original.review_rule == "iso_6346_ocr_o_zero_alias"
    rendered = iso_equipment_surface(original.size_category, original.type_category, source)
    assert rendered == source
    assert iso_equipment_surface("FORTY_FOOT_HIGH_CUBE", "REFRIGERATED", source).endswith("RO")


@pytest.mark.parametrize("source", ["20HO", "4OGO", "22OO", "22XO"])
def test_ocr_alias_does_not_reinterpret_unknown_or_carrier_codes(source):
    from document_ocr.synthesis.container_semantics import iso_equipment_surface

    assert iso_equipment_surface("FORTY_FOOT_HIGH_CUBE", "REFRIGERATED", source) is None
    assert review_source_equipment_surface(source, temperature_present=False).size_category is None


@pytest.mark.parametrize("source", ["40RQ", "40 RQ", "40'RQ", "2X40RQ", "40RQ CONTAINER"])
@pytest.mark.parametrize("temperature", [False, True])
def test_documented_reefer_alias_has_height_but_does_not_invent_temperature(source, temperature):
    row = review_source_equipment_surface(source, temperature_present=temperature)
    assert (row.size_category, row.type_category) == ("FORTY_FOOT_HIGH_CUBE", "REFRIGERATED")
    assert row.thermal_operation == ("active" if temperature else "not_indicated")


@pytest.mark.parametrize(
    "source,size,kind",
    [
        ("20HO", "TWENTY_FOOT_HIGH_CUBE", "OPEN_TOP"),
    ],
)
def test_carrier_specific_equipment_needs_carrier_identity(source, size, kind):
    from document_ocr.labeling_agents.equipment_normalization import reconcile_equipment_categories

    label = {"documentPatch": {"containerInformation": [{"typeDescription": source}]}}
    for declaration in ("TARROS SPA AS CARRIER", "SIGNED ON BEHALF OF THE CARRIER TARROS S.P.A."):
        actual, _ = reconcile_equipment_categories(label, source_text=source + "\n" + declaration)
        assert actual["documentPatch"]["containerInformation"] == [
            {"sizeCategory": size, "typeCategory": kind}
        ]
    for declaration in (
        "VESSEL TARROS",
        "NLINE SHIPPING SRL AS CARRIER",
        "CARRIER: CMA CGM\nTARROS SPA AS CARRIER",
    ):
        assert (
            reconcile_equipment_categories(label, source_text=source + "\n" + declaration)[0]
            == label
        )


@pytest.mark.parametrize(
    "surface,length",
    [
        ("40RE", "40"),
        ("40RF", "40"),
        ("40 RF", "40"),
        ("RF40", "40"),
        ("20RF", "20"),
        ("45RF", "45"),
    ],
)
def test_compact_reefer_standard_policy_is_shared_by_labels_and_sampling(surface, length):
    from document_ocr.synthesis.container_semantics import partial_equipment_constraint
    from document_ocr.synthesis.template_compiler.descendant import _equipment_semantics_match

    reviewed = review_source_equipment_surface(surface, temperature_present=True)
    assert reviewed.type_category == "REFRIGERATED"
    size = {
        "20": "TWENTY_FOOT_STANDARD_HEIGHT",
        "40": "FORTY_FOOT_STANDARD_HEIGHT",
        "45": "FORTY_FIVE_FOOT_HIGH_CUBE",
    }[length]
    assert reviewed.size_category == size
    assert reviewed.resolution == "reviewed_source_grammar"
    assert reviewed.thermal_operation == "active"
    assert partial_equipment_constraint({"typeDescription": surface}) == (
        None,
        "REFRIGERATED",
        size,
    )
    assert _equipment_semantics_match(
        {"sizeCategory": size, "typeCategory": "REFRIGERATED"},
        surface,
    )


@pytest.mark.parametrize(
    "surface", ["40'X9'6\" REEFER CONTAINER", "40'H RF CONTAINER", '40" REEFER HIGH CUBIC']
)
def test_explicit_high_reefer_keeps_high_cube_semantics(surface):
    reviewed = review_source_equipment_surface(surface, temperature_present=True)
    assert (reviewed.size_category, reviewed.type_category) == (
        "FORTY_FOOT_HIGH_CUBE",
        "REFRIGERATED",
    )


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
    assert candidate == "L5R0"
    assert _equipment_semantics_match(expected, candidate)
    ocr_candidate = _equipment_surface_candidates(expected, "22GO")[0]
    assert ocr_candidate == "L5RO"
    assert _equipment_semantics_match(expected, ocr_candidate)


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
    [
        "20 FT ISO TANK CONTAINER(S)",
        "20 FT ISO TANKCONTAINER(S)",
        "20 FT ISO TANKCONTAINERS",
        "20TANK CONTAINER",
        "20TANKCONTAINER",
        "20TANK",
    ],
)
def test_generic_tank_uses_mpci_liquid_tank_bucket_without_thermal_inference(surface):
    resolved = review_source_equipment_surface(surface, temperature_present=False)
    assert resolved.size_category == "TWENTY_FOOT_STANDARD_HEIGHT"
    assert resolved.type_category == "PRESSURIZED_TANK"
    assert "generic_liquid_tank_target_category" in resolved.review_rule
    assert resolved.thermal_operation == "not_indicated"
    from document_ocr.synthesis.container_semantics import partial_equipment_constraint

    assert partial_equipment_constraint({"typeDescription": surface}) == (
        None,
        "PRESSURIZED_TANK",
        "TWENTY_FOOT_STANDARD_HEIGHT",
    )


def test_tank_word_boundary_normalization_is_not_a_substring_rewrite():
    resolved = review_source_equipment_surface("20 TANKCONTAINERIZATION", temperature_present=False)
    assert resolved.resolution == "unresolved_source_surface"


def test_compiler_and_generation_reject_unowned_printed_shipment_container_ids():
    from dataclasses import replace
    from types import SimpleNamespace

    from document_ocr.synthesis.template_compiler.host import (
        SpanDraft,
        validate_compiled_current_container_identifier_ownership,
        validate_current_container_identifier_ownership,
    )

    draft = SpanDraft(
        draft_id="id",
        logical_key="container_1_identifier",
        render_mode="deterministic_auxiliary",
        value_kind="identifier",
        group_kind="equipment",
        group_key="container:1",
        target_paths=(),
        derivation=None,
        dependency_paths=(),
        dependency_bindings=(),
        char_start=0,
        char_end=11,
        source_text="TRKU2030956",
        evidence_origin="agent",
        render_policy="opaque_identifier",
        rationale="printed row",
    )
    with pytest.raises(ValueError, match="lacks its label-backed owner"):
        validate_current_container_identifier_ownership((draft,))
    binding = SimpleNamespace(
        group_kind=draft.group_kind,
        group_key=draft.group_key,
        logical_key=draft.logical_key,
        target_paths=draft.target_paths,
        occurrences=(SimpleNamespace(source_text=draft.source_text),),
    )
    with pytest.raises(ValueError, match="lacks its label-backed owner"):
        validate_compiled_current_container_identifier_ownership(
            SimpleNamespace(bindings=(binding,))
        )
    owner = "documentPatch.containers[1].containerNumber"
    validate_current_container_identifier_ownership(
        (replace(draft, render_mode="target_binding", target_paths=(owner,)),)
    )
    validate_current_container_identifier_ownership(
        (replace(draft, logical_key="seal:container:1"),)
    )
    validate_current_container_identifier_ownership((replace(draft, group_key="container:other"),))


@pytest.mark.parametrize("length", ["20", "40", "45"])
def test_rfh_uses_default_size_without_mutating_printed_description(length):
    from document_ocr.synthesis.container_semantics import partial_equipment_constraint

    container = {"containerNumber": "TRKU1100724", "typeDescription": length + "'RFH"}
    size = {
        "20": "TWENTY_FOOT_STANDARD_HEIGHT",
        "40": "FORTY_FOOT_STANDARD_HEIGHT",
        "45": "FORTY_FIVE_FOOT_HIGH_CUBE",
    }[length]
    assert partial_equipment_constraint(container) == (None, "REFRIGERATED", size)
    assert container == {"containerNumber": "TRKU1100724", "typeDescription": length + "'RFH"}
    resolved = review_source_equipment_surface(length + "'RFH", temperature_present=False)
    assert resolved.type_category == "REFRIGERATED"
    assert resolved.size_category == size
    assert resolved.resolution == "reviewed_source_grammar"


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


def test_source_row_standard_dry_abbreviation_matches_explicit_standard_receipt():
    row = review_source_equipment_surface("20'SD", temperature_present=False)
    receipt = review_source_equipment_surface(
        "17 X 20' STD FCL CONTAINERS STC", temperature_present=False
    )
    assert (row.size_category, row.type_category) == (
        "TWENTY_FOOT_STANDARD_HEIGHT",
        "GENERAL_PURPOSE",
    )
    assert (row.size_category, row.type_category) == (
        receipt.size_category,
        receipt.type_category,
    )


@pytest.mark.parametrize("surface", ["GEN", "20GEN", "40EC", "40EQ", "20 BOX SPECIAL"])
@pytest.mark.parametrize("temperature", [False, True])
def test_unreviewed_alias_is_not_a_box_or_reefer_default(surface, temperature):
    observed = review_source_equipment_surface(surface, temperature_present=temperature)
    assert observed.resolution == "unresolved_source_surface"
    assert observed.size_category is None


@pytest.mark.parametrize("length", ["20", "40"])
@pytest.mark.parametrize(
    "suffix",
    ["", "'", "FT", "'CONT", "' CONTAINER", "' CONTAINER(S)", "' BOX", "'BOX", "'BO", "' FULL"],
)
def test_closed_generic_box_grammar_uses_consistent_standard_policy(length, suffix):
    from document_ocr.synthesis.container_semantics import partial_equipment_constraint

    surface = length + suffix
    observed = review_source_equipment_surface(surface, temperature_present=False)
    expected_size = {"20": "TWENTY_FOOT_STANDARD_HEIGHT", "40": "FORTY_FOOT_STANDARD_HEIGHT"}[
        length
    ]
    assert observed.resolution == "reviewed_source_grammar"
    assert (observed.size_category, observed.type_category) == (expected_size, "GENERAL_PURPOSE")
    assert partial_equipment_constraint({"typeDescription": surface}) == (
        None,
        "GENERAL_PURPOSE",
        expected_size,
    )
    thermal = review_source_equipment_surface(surface, temperature_present=True)
    assert (thermal.size_category, thermal.type_category) == (expected_size, "REFRIGERATED")


@pytest.mark.parametrize("surface", ["20BX", "40'BX", "20DY", "40 DY"])
def test_reviewed_dry_alias_default_is_not_limited_to_one_carrier(surface):
    observed = review_source_equipment_surface(surface, temperature_present=False)
    assert observed.type_category == "GENERAL_PURPOSE"
    assert observed.size_category in {"TWENTY_FOOT_STANDARD_HEIGHT", "FORTY_FOOT_STANDARD_HEIGHT"}
    assert "default_standard_height" in observed.review_rule


@pytest.mark.parametrize(
    "surface", ["40NOR", "40' NOR", "40 HC NOR", "40 HIGH CUBE REFRIGERATED NOR"]
)
def test_non_operating_reefer_cannot_silently_become_active(surface):
    from document_ocr.synthesis.container_semantics import partial_equipment_constraint

    observed = review_source_equipment_surface(surface, temperature_present=False)
    assert observed.resolution == "reviewed_source_grammar"
    assert observed.type_category == "REFRIGERATED"
    assert observed.thermal_operation == "non_operating"
    conflict = review_source_equipment_surface(surface, temperature_present=True)
    assert conflict.resolution == "unresolved_source_surface"
    assert conflict.review_rule == "non_operating_reefer_conflicts_with_setpoint"
    assert conflict.type_category is None
    with pytest.raises(ValueError, match="non-operating reefer conflicts"):
        partial_equipment_constraint(
            {"typeDescription": surface, "temperatureSetpoint": {"value": -18, "unit": "CEL"}}
        )


@pytest.mark.parametrize(
    "surface,kind",
    [
        ("HIGH CUBE CONTAINER", "GENERAL_PURPOSE"),
        ("STANDARD CONTAINER", "GENERAL_PURPOSE"),
        ("REFRIGERATED CONTAINER", "REFRIGERATED"),
    ],
)
def test_known_family_does_not_invent_missing_length(surface, kind):
    observed = review_source_equipment_surface(surface, temperature_present=False)
    assert observed.type_category == kind
    assert observed.size_category is None


def test_fallback_projection_uses_supported_source_and_explicit_carrier_not_vessel():
    from document_ocr.labeling_agents.equipment_normalization import reconcile_equipment_categories

    containers = [
        {"equipmentIdentifier": "ABCU1234567", "typeDescription": "40RA"},
        {"equipmentIdentifier": "DEFU1234567", "typeDescription": "40H (HI-CUBE)"},
        {"equipmentIdentifier": "GHIU1234567", "typeDescription": "40RO / 40HR"},
        {"equipmentIdentifier": "JKLU1234567", "typeDescription": "22GO"},
    ]
    target = {"schemaVersion": "7.0.0", "documentPatch": {"containerInformation": containers}}
    raw = (
        "CARRIER: CMA CGM Société Anonyme\n1X40RA\n"
        "DEFU1234567/40H/SEAL/1280 CARTONS (HI-CUBE)\n"
        "40RO\n1X40HR CONTAINER SAID TO CONTAIN\n22GO\n"
    )
    fixed, decisions = reconcile_equipment_categories(target, source_text=raw)
    values = fixed["documentPatch"]["containerInformation"]
    assert [x["action"] for x in decisions] == ["canonicalize"] * 4
    assert [x.get("typeCategory") for x in values] == [
        "REFRIGERATED",
        "GENERAL_PURPOSE",
        "REFRIGERATED",
        "GENERAL_PURPOSE",
    ]
    assert all(x["sizeCategory"] == "FORTY_FOOT_HIGH_CUBE" for x in values[:3])
    assert values[-1]["sizeCategory"] == "TWENTY_FOOT_STANDARD_HEIGHT"
    assert decisions[-1]["reason"] == "iso_6346_ocr_o_zero_alias"
    assert all("typeDescription" in x for x in containers)  # no input mutation
    assert reconcile_equipment_categories(fixed, source_text=raw)[0] == fixed
    vessel_only = raw.replace("CARRIER: CMA CGM Société Anonyme", "VESSEL: CMA CGM MEKONG")
    unchanged, _ = reconcile_equipment_categories(target, source_text=vessel_only)
    assert unchanged["documentPatch"]["containerInformation"][0] == {
        "equipmentIdentifier": "ABCU1234567",
        "typeCategory": "REFRIGERATED",
    }
    shorter = {
        "documentPatch": {
            "containerInformation": [
                {"equipmentIdentifier": "ABCU1234567", "typeDescription": "40H"}
            ]
        }
    }
    assert reconcile_equipment_categories(shorter, source_text="1X40HR")[0] == shorter
    conflicting = {
        "documentPatch": {
            "containerInformation": [
                {"equipmentIdentifier": "ABCU1234567", "typeDescription": "40GP / 40RH"}
            ]
        }
    }
    held, decisions = reconcile_equipment_categories(conflicting, source_text="40GP / 40RH")
    assert held == conflicting
    assert decisions[0]["reason"] == "conflicting_equipment_aliases"
