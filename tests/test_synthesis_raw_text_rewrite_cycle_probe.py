from __future__ import annotations

import json
from decimal import Decimal
from pathlib import Path

import pytest
from pydantic import TypeAdapter, ValidationError
from pydantic_ai import Agent
from pydantic_ai.models.test import TestModel

from document_ocr.synthesis.config import (
    load_synthesis_raw_text_rewrite_cycle_probe_config,
)
from document_ocr.synthesis.country_registry import load_iso_country_registry
from document_ocr.synthesis.linguistic_completion_pipeline import DocumentLinguisticPlan
from document_ocr.synthesis.package_registry import load_package_registry
from document_ocr.synthesis.raw_text_rewrite_cycle_probe import (
    AnchoredScalarReplacementRequirement,
    CompactLabelChangeDirective,
    CompoundPartyFlavorRealization,
    CustomsProgramEntry,
    EmpiricalOperationalProfile,
    LabelChangeDirective,
    LineRangeReplacement,
    RewriteWorkspace,
    SemanticReviewFinding,
    SemanticReviewReceipt,
    SurfaceRenderingRequirement,
    TargetIntegrityResources,
    _evidence_occurs,
    _finding_conflicts_with_anchored_scalar_authority,
    _finding_conflicts_with_equipment_authority,
    _finding_conflicts_with_surface_authority,
    _finding_grounds_format_damage_only_in_unchanged_text,
    _line_id,
    _prompt_content,
    _review_audit_payload,
    _settings,
    _target_route_jurisdictions,
    _terminal_editor_output,
    anchored_measurement_replacement_requirements,
    anchored_scalar_replacement_requirements,
    apply_deterministic_prefills,
    apply_line_range_replacements,
    build_empirical_operational_profiles,
    cargo_flavor_rewrite_requirements,
    compact_label_change_contract,
    compound_party_flavor_requirements,
    deterministic_rewrite_audit,
    editable_indexed_ocr_lines,
    indexed_ocr_lines,
    inline_slot_topology_requirements,
    jurisdictional_surface_requirements,
    label_change_contract,
    merge_anchored_scalar_replacement_requirements,
    operational_flavor_requirements,
    prepare_target_integrity,
    raw_auxiliary_identity_requirements,
    recover_explicit_hs_target_facts,
    rewrite_changed_leaves,
    source_semantic_role_hints,
    source_status_preservation_requirements,
    surface_rendering_requirements,
    target_literal_requirements,
    target_value_occurrence_requirements,
)
from document_ocr.synthesis.raw_text_rewrite_probe import ChangedLeaf
from document_ocr.synthesis.transport_capacity import capacity_limits


def _edit(
    source: str,
    start: int,
    end: int,
    new: str,
) -> LineRangeReplacement:
    lines = source.splitlines(keepends=True)
    return LineRangeReplacement(
        startLineId=_line_id(start, lines[start - 1]),
        endLineId=_line_id(end, lines[end - 1]),
        newText=new,
    )


def _workspace(text: str) -> RewriteWorkspace:
    return RewriteWorkspace(original_text=text, current_text=text)


def _target_integrity_resources() -> TargetIntegrityResources:
    return TargetIntegrityResources(
        countries=load_iso_country_registry(
            iso_path=Path("data/registries/countries/iso-codes-4.9.0-1/iso_3166-1.json"),
            iso_sha256="f7dc5542a692ad8e23b9b85a6a1800a63f7a05e5246065fac1ede04ed209ce00",
        ),
        packages=load_package_registry(
            Path(
                "artifacts/mpci-ai-schema/categorical-registry-followup/"
                "package-category-registry.json"
            ),
            expected_sha256=("2476deb46c3cabf2efe9a00bbb9ad1bf6b527e4adae0e4872d8a63807bbd374f"),
            expected_entries=405,
        ),
        route_countries_by_name={
            "IZUHARA": frozenset({"JP"}),
            "KOTZEBUE": frozenset({"US"}),
            "EL ISKANDARIYA ALEXANDRIA": frozenset({"EG"}),
        },
        customs_programs=(
            CustomsProgramEntry(
                program_id="egypt_advance_cargo_information",
                official_name="Egypt Advance Cargo Information",
                authority="NAFEZA",
                official_source_url="https://www.nafeza.gov.eg/en/pages/15",
                jurisdiction_country_code="EG",
                trade_direction="import",
                source_surfaces=("ACID NUMBER", "ACID NO", "ACID"),
                generic_replacement_surface="CUSTOMS REFERENCE",
            ),
        ),
    )


def _linguistic_plan_with_hs_codes(*codes: str | None) -> DocumentLinguisticPlan:
    document_id = "doc_" + "1" * 64
    return DocumentLinguisticPlan.model_validate_json(
        json.dumps(
            {
            "documentIndex": 0,
            "baseDocumentId": document_id,
            "scenarioId": "syn-test",
            "upstreamTargetSha256": "2" * 64,
            "partyUnits": [],
            "partyProjections": [],
            "cargoSeed": {
                "caseId": "cargo-1",
                "sourceDocumentId": document_id,
                "scenarioId": "syn-test",
                "routeContext": {
                    "portOfLoading": None,
                    "portOfDischarge": None,
                    "placeOfDelivery": None,
                },
                "cargoGroups": [
                    {
                        "groupId": f"g{index + 1}",
                        "goodsIdentities": [
                            {
                                "description": f"Synthetic goods {index + 1}",
                                "hsCode": code,
                                "chapterDescription": None,
                                "headingDescription": None,
                                "thermalProfile": None,
                            }
                        ],
                        "dangerousGoods": [],
                        "structuredFacts": {
                            "packages": [],
                            "equipment": [],
                            "cargoOrigin": None,
                            "grossWeight": None,
                            "netWeight": None,
                            "volume": None,
                        },
                        "fieldContract": {
                            "descriptionPresent": True,
                            "sourceDescriptionStyleReference": f"Source goods {index + 1}",
                            "additionalInformationSlots": [],
                            "marksAndNumbersSlots": [],
                            "handlingInstructionSlots": [],
                        },
                    }
                    for index, code in enumerate(codes)
                ],
                "excludedFinalPatchingFields": [
                    "packagePrintedSurfaces",
                    "containerPrintedSurfaces",
                    "hsCodePrintedSurfaces",
                ],
            },
            }
        ),
        strict=True,
    )


def test_line_index_is_one_based_and_does_not_mutate_text() -> None:
    source = "--- PAGE 1 ---\nB/L: OLD123\nCOPY: OLD123\n"

    assert indexed_ocr_lines(source) == (
        f"{_line_id(1, '--- PAGE 1 ---')}|--- PAGE 1 ---\n"
        f"{_line_id(2, 'B/L: OLD123')}|B/L: OLD123\n"
        f"{_line_id(3, 'COPY: OLD123')}|COPY: OLD123"
    )


def test_route_jurisdictions_resolve_from_pinned_port_registry() -> None:
    target = {
        "documentPatch": {
            "route": {
                "portOfLoading": {"name": "Surabaya", "country": "Indonesia"},
                "portOfDischarge": {"name": "El Iskandariya (Alexandria)"},
            }
        }
    }
    resources = _target_integrity_resources()

    assert _target_route_jurisdictions(target, resources) == {
        "export": {"isoAlpha2": "ID", "name": "Indonesia"},
        "import": {"isoAlpha2": "EG", "name": "Egypt"},
    }


def test_route_jurisdictions_allow_cross_border_place_of_receipt() -> None:
    target = {
        "documentPatch": {
            "route": {
                "placeOfReceipt": {"name": "Hong Kong", "country": "Hong Kong"},
                "portOfLoading": {"name": "Qingdao", "country": "China"},
                "portOfDischarge": {"name": "Si Racha", "country": "Thailand"},
            }
        }
    }

    assert _target_route_jurisdictions(target, _target_integrity_resources()) == {
        "export": {"isoAlpha2": "CN", "name": "China"},
        "import": {"isoAlpha2": "TH", "name": "Thailand"},
    }


def test_editor_workspace_omits_immutable_blank_lines_and_page_markers() -> None:
    source = "--- PAGE 1 ---\nB/L: OLD123\n\nCOPY: OLD123\n"

    assert editable_indexed_ocr_lines(source) == (
        f"{_line_id(2, 'B/L: OLD123')}|B/L: OLD123\n{_line_id(4, 'COPY: OLD123')}|COPY: OLD123"
    )


def test_missing_explicit_commodity_code_is_recovered_from_pinned_plan() -> None:
    raw_text = "--- PAGE 1 ---\nCOMMODITY CODES: 0204421000\n"
    source = {
        "documentPatch": {
            "cargoGroups": [
                {"groupId": "g1", "description": "SOURCE ONE"},
                {"groupId": "g2", "description": "SOURCE TWO"},
            ]
        }
    }
    target = {
        "documentPatch": {
            "cargoGroups": [
                {"groupId": "g1", "description": "TARGET ONE"},
                {"groupId": "g2", "description": "TARGET TWO"},
            ]
        }
    }

    effective, recoveries, requirements = recover_explicit_hs_target_facts(
        raw_text=raw_text,
        source_label=source,
        target=target,
        plan=_linguistic_plan_with_hs_codes("842290", "392410"),
    )

    generated = [
        effective["documentPatch"]["cargoGroups"][index]["hsCodes"][0]
        for index in range(2)
    ]
    assert [value[:6] for value in generated] == ["842290", "392410"]
    assert all(len(value) == 10 for value in generated)
    assert [row.renderedCode for row in recoveries] == generated
    assert len(requirements) == 1
    assert requirements[0].kind == "hs_code_block"
    assert requirements[0].targetSurface == ", ".join(generated)

    workspace = RewriteWorkspace(
        original_text=raw_text,
        current_text=raw_text,
        current_target_label=effective,
        surface_requirements=requirements,
    )
    apply_deterministic_prefills(workspace)
    assert workspace.current_text == (
        f"--- PAGE 1 ---\nCOMMODITY CODES: {generated[0]}, {generated[1]}\n"
    )


def test_missing_explicit_hs_code_fails_on_partial_label_coverage() -> None:
    with pytest.raises(ValueError, match="only partially represented"):
        recover_explicit_hs_target_facts(
            raw_text="HS CODE: 111111\nCOMMODITY CODE: 222222\n",
            source_label={
                "documentPatch": {
                    "cargoGroups": [{"groupId": "g1", "hsCodes": ["111111"]}]
                }
            },
            target={"documentPatch": {"cargoGroups": [{"groupId": "g1"}]}},
            plan=_linguistic_plan_with_hs_codes("842290"),
        )


def test_missing_explicit_hs_code_fails_on_ambiguous_cardinality() -> None:
    with pytest.raises(ValueError, match="cardinality cannot be projected"):
        recover_explicit_hs_target_facts(
            raw_text="HS CODES: 111111, 222222\n",
            source_label={
                "documentPatch": {
                    "cargoGroups": [
                        {"groupId": "g1"},
                        {"groupId": "g2"},
                        {"groupId": "g3"},
                    ]
                }
            },
            target={
                "documentPatch": {
                    "cargoGroups": [
                        {"groupId": "g1"},
                        {"groupId": "g2"},
                        {"groupId": "g3"},
                    ]
                }
            },
            plan=_linguistic_plan_with_hs_codes("842290", "392410", "847160"),
        )


def test_missing_explicit_hs_code_fails_when_plan_has_no_hs_identity() -> None:
    with pytest.raises(ValueError, match="lacks an HS identity"):
        recover_explicit_hs_target_facts(
            raw_text="COMMODITY CODE: 0204421000\n",
            source_label={"documentPatch": {"cargoGroups": [{"groupId": "g1"}]}},
            target={"documentPatch": {"cargoGroups": [{"groupId": "g1"}]}},
            plan=_linguistic_plan_with_hs_codes(None),
        )


def test_review_evidence_accepts_only_semantically_equivalent_grounded_text() -> None:
    text = "ON BEHALF OF SILVERCREST INDUSTRIAL\nTRADING LLC,"

    assert _evidence_occurs("ON BEHALF OF SILVERCREST INDUSTRIAL TRADING LLC,", text)
    assert not _evidence_occurs("ON BEHALF OF DIFFERENT INDUSTRIAL TRADING LLC,", text)


def test_review_evidence_tolerates_punctuation_spacing_but_not_partial_tokens() -> None:
    text = "QUY NHON,VIET NAM\nSOKHNA,EGYPT\nREFERENCE: ABC1234"

    assert _evidence_occurs("QUY NHON, VIET NAM\nSOKHNA, EGYPT", text)
    assert not _evidence_occurs("REFERENCE: ABC123", text)


def test_atomic_patch_applies_multiple_non_overlapping_ranges() -> None:
    source = "--- PAGE 1 ---\nB/L: OLD123\nGENERIC\nCOPY: OLD123\n"
    workspace = _workspace(source)

    commit = apply_line_range_replacements(
        workspace,
        (
            _edit(source, 2, 2, "B/L: NEW789"),
            _edit(source, 4, 4, "COPY: NEW789"),
        ),
    )

    assert workspace.current_text == source.replace("OLD123", "NEW789")
    assert commit.beforeTextSha256 != commit.afterTextSha256
    assert [row.startLine for row in commit.appliedReplacements] == [2, 4]
    assert sum(row.sourceLinesReplaced for row in commit.appliedReplacements) == 2
    assert len(workspace.commits) == 1


def test_atomic_patch_preserves_numbered_structural_line_prefixes() -> None:
    source = "--- PAGE 1 ---\n7.3. The United States law applies.\n"
    workspace = _workspace(source)

    with pytest.raises(ValueError, match="numbered structural line prefix"):
        apply_line_range_replacements(
            workspace,
            (_edit(source, 2, 2, "The Indian law applies."),),
        )

    assert workspace.current_text == source
    apply_line_range_replacements(
        workspace,
        (_edit(source, 2, 2, "7.3. The Indian law applies."),),
    )
    assert workspace.current_text == "--- PAGE 1 ---\n7.3. The Indian law applies.\n"


def test_atomic_patch_rejects_isolated_ocr_punctuation_cleanup() -> None:
    source = (
        "--- PAGE 1 ---\n"
        "After a Free Time of 10 days the rate applies per container Tariff remains unchanged.\n"
    )
    workspace = _workspace(source)

    with pytest.raises(ValueError, match="punctuation or whitespace outside an actual value"):
        apply_line_range_replacements(
            workspace,
            (
                _edit(
                    source,
                    2,
                    2,
                    "After a Free Time of 14 days the rate applies per container. "
                    "Tariff remains unchanged.",
                ),
            ),
        )

    assert workspace.current_text == source


def test_atomic_patch_allows_punctuation_inside_a_lexical_replacement() -> None:
    source = "--- PAGE 1 ---\nCarrier: OLD NAME\n"
    workspace = _workspace(source)

    apply_line_range_replacements(
        workspace,
        (_edit(source, 2, 2, "Carrier: NEW NAME, LTD."),),
    )

    assert workspace.current_text == "--- PAGE 1 ---\nCarrier: NEW NAME, LTD.\n"


def test_atomic_patch_rewrites_positive_operation_when_setpoint_is_deactivated() -> None:
    source = (
        "--- PAGE 1 ---\n"
        "Reefer temperature to be set at -21 C\n"
        "Plugging for the account of cargo\n"
    )
    source_label = {
        "documentPatch": {
            "containers": [
                {
                    "containerNumber": "MSCU1234567",
                    "temperatureSetpoint": {"value": -21.0, "unit": "celsius"},
                }
            ]
        }
    }
    target_label = {
        "documentPatch": {
            "containers": [
                {
                    "containerNumber": "MSCU7654321",
                    "typeCategory": "REFRIGERATED",
                }
            ]
        }
    }
    workspace = RewriteWorkspace(
        original_text=source,
        current_text=source,
        source_label=source_label,
        current_target_label=target_label,
    )

    with pytest.raises(ValueError, match="positive temperature or plugging assertion"):
        apply_line_range_replacements(
            workspace,
            (
                _edit(source, 2, 2, "Reefer non-operating; ventilation open"),
                _edit(source, 3, 3, "Plugging for the account of cargo"),
            ),
        )

    assert workspace.current_text == source
    apply_line_range_replacements(
        workspace,
        (
            _edit(source, 2, 2, "Reefer non-operating; ventilation open"),
            _edit(source, 3, 3, "Ventilation monitoring for the account of cargo"),
        ),
    )
    assert "Plugging" not in workspace.current_text


def test_atomic_patch_rejects_non_operating_raw_only_reefer_with_plugging() -> None:
    source = (
        "--- PAGE 1 ---\n"
        "Reefer carrying temperature to be set at -21.0 C\n"
        "Plugging for the account of cargo\n"
    )
    workspace = _workspace(source)

    with pytest.raises(ValueError, match="positive temperature or plugging assertion"):
        apply_line_range_replacements(
            workspace,
            (_edit(source, 2, 2, "Reefer unit in non-operating ventilation mode"),),
        )

    assert workspace.current_text == source


def test_out_of_bounds_line_address_rolls_back_complete_patch() -> None:
    source = "--- PAGE 1 ---\nB/L: OLD123\nCOPY: OLD123\n"
    workspace = _workspace(source)

    bad = _edit(source, 3, 3, "COPY: NEW789").model_copy(
        update={"startLineId": "L00004", "endLineId": "L00004"}
    )
    with pytest.raises(ValueError, match="ends at line 4"):
        apply_line_range_replacements(
            workspace,
            (
                _edit(source, 2, 2, "B/L: NEW789"),
                bad,
            ),
        )

    assert workspace.current_text == source
    assert workspace.commits == []


def test_trailing_line_terminator_and_noop_edits_do_not_force_model_retry() -> None:
    source = "--- PAGE 1 ---\nB/L: OLD123\nUNCHANGED\n"
    workspace = _workspace(source)

    commit = apply_line_range_replacements(
        workspace,
        (
            _edit(source, 2, 2, "B/L: NEW789\n"),
            _edit(source, 3, 3, "UNCHANGED"),
        ),
    )

    assert workspace.current_text == "--- PAGE 1 ---\nB/L: NEW789\nUNCHANGED\n"
    assert len(commit.appliedReplacements) == 1


def test_overlapping_line_ranges_are_rejected_atomically() -> None:
    source = "--- PAGE 1 ---\nSHIPPER: OLD\nADDRESS: OLD\n"
    workspace = _workspace(source)

    with pytest.raises(ValueError, match="overlaps or duplicates"):
        apply_line_range_replacements(
            workspace,
            (
                _edit(
                    source,
                    2,
                    3,
                    "SHIPPER: NEW\nADDRESS: NEW",
                ),
                _edit(source, 3, 3, "ADDRESS: OTHER"),
            ),
        )

    assert workspace.current_text == source


def test_page_marker_ranges_and_introduced_markers_are_rejected() -> None:
    source = "--- PAGE 1 ---\nB/L: OLD123\n"

    with pytest.raises(ValueError, match="includes a protected page marker"):
        apply_line_range_replacements(
            _workspace(source),
            (_edit(source, 1, 2, "B/L: NEW789"),),
        )
    with pytest.raises(ValueError, match="introduces a protected page marker"):
        apply_line_range_replacements(
            _workspace(source),
            (_edit(source, 2, 2, "--- PAGE 2 ---"),),
        )


@pytest.mark.parametrize("placeholder", ("UNAVAILABLE", "UNKNOWN", "N/A", "TBD", "TBA"))
def test_new_placeholder_substitutions_are_rejected(placeholder: str) -> None:
    workspace = _workspace("--- PAGE 1 ---\nEQUIPMENT: 1 X 40 HC\n")

    with pytest.raises(ValueError, match="introduces an unavailable/unknown placeholder"):
        apply_line_range_replacements(
            workspace,
            (_edit(workspace.original_text, 2, 2, f"EQUIPMENT: {placeholder}"),),
        )

    assert workspace.current_text.endswith("1 X 40 HC\n")


def test_locality_named_ban_na_is_not_mistaken_for_na_placeholder() -> None:
    workspace = _workspace("--- PAGE 1 ---\nDELIVERY: OLD PORT\n")

    apply_line_range_replacements(
        workspace,
        (_edit(workspace.original_text, 2, 2, "DELIVERY: Ban Na"),),
    )

    assert workspace.current_text.endswith("DELIVERY: Ban Na\n")


def test_customs_program_requirement_uses_target_port_country_and_token_boundaries() -> None:
    source = (
        "ACID NUMBER: 1002845232025040024\nACID#: 1004977722024020145\nACIDIC, ORGANIC, N.O.S.\n"
    )
    target = {
        "documentPatch": {
            "route": {
                "portOfDischarge": {"name": "Izuhara"},
                "placeOfDelivery": {"name": "Izuhara"},
            }
        }
    }

    requirements = jurisdictional_surface_requirements(
        source, target, _target_integrity_resources()
    )

    assert [(row.sourceSurface, row.sourceOccurrences) for row in requirements] == [
        ("ACID NUMBER", 1),
        ("ACID", 1),
    ]
    assert {row.targetRouteCountryCode for row in requirements} == {"JP"}


def test_customs_program_requirement_is_not_emitted_for_its_own_jurisdiction() -> None:
    requirements = jurisdictional_surface_requirements(
        "ACID: 1002845232025040024\n",
        {"documentPatch": {"route": {"portOfDischarge": {"name": "El Iskandariya (Alexandria)"}}}},
        _target_integrity_resources(),
    )

    assert requirements == ()


def test_atomic_patch_requires_stale_customs_program_replacement_without_matching_acidic() -> None:
    source = "--- PAGE 1 ---\nACID: 1002845232025040024\nGOODS: ACIDIC, ORGANIC\n"
    requirements = jurisdictional_surface_requirements(
        source,
        {"documentPatch": {"route": {"portOfDischarge": {"name": "Kotzebue"}}}},
        _target_integrity_resources(),
    )
    workspace = RewriteWorkspace(
        original_text=source,
        current_text=source,
        jurisdictional_requirements=requirements,
    )

    with pytest.raises(ValueError, match="stale named customs program"):
        apply_line_range_replacements(workspace, (_edit(source, 3, 3, "GOODS: NEW"),))

    commit = apply_line_range_replacements(
        workspace,
        (_edit(source, 2, 2, "CUSTOMS REFERENCE: US-847291563"),),
    )
    assert commit.afterTextSha256 == workspace.current_sha256
    assert "ACIDIC" in workspace.current_text


def test_auxiliary_legal_identity_is_synthesized_without_losing_topology() -> None:
    workspace = _workspace(
        "--- PAGE 1 ---\nSIGNED ON BEHALF OF OLD CARRIER\nBY OLD LOCAL AGENT AS AGENT\n"
    )

    apply_line_range_replacements(
        workspace,
        (
            _edit(
                workspace.original_text,
                2,
                3,
                "SIGNED ON BEHALF OF MERIDIAN OCEAN LINES\nBY HARBOR CREST AGENCIES AS AGENT",
            ),
        ),
    )

    assert "SIGNED ON BEHALF OF MERIDIAN OCEAN LINES" in workspace.current_text
    assert "BY HARBOR CREST AGENCIES AS AGENT" in workspace.current_text


def test_terminal_output_function_commits_in_one_model_request() -> None:
    source = "--- PAGE 1 ---\nB/L: OLD123\n"
    workspace = _workspace(source)
    model = TestModel(
        custom_output_args={
            "edits": [
                {
                    "startLineId": _line_id(2, "B/L: OLD123"),
                    "endLineId": _line_id(2, "B/L: OLD123"),
                    "newText": "B/L: NEW789",
                }
            ],
            "compoundPartyFlavorRealizations": [],
        }
    )
    agent = Agent(
        model,
        deps_type=RewriteWorkspace,
        output_type=_terminal_editor_output(workspace, max_retries=1),
    )

    result = agent.run_sync("rewrite", deps=workspace)

    assert result.usage.requests == 1
    assert workspace.current_text == "--- PAGE 1 ---\nB/L: NEW789\n"
    assert result.output.afterTextSha256 == workspace.current_sha256


def test_terminal_output_schema_only_allows_populated_editable_line_ids() -> None:
    source = "--- PAGE 1 ---\nB/L: OLD123\n\nCOPY: OLD123\n"
    workspace = _workspace(source)
    output = _terminal_editor_output(workspace, max_retries=1)
    edits_type = output.output.__annotations__["edits"]
    schema = TypeAdapter(edits_type).json_schema()
    line_schema = schema["$defs"][next(iter(schema["$defs"]))]["properties"]
    allowed = [_line_id(2, "B/L: OLD123"), _line_id(4, "COPY: OLD123")]

    assert line_schema["startLineId"]["enum"] == allowed
    assert line_schema["endLineId"]["enum"] == allowed
    assert _line_id(1, "--- PAGE 1 ---") not in line_schema["startLineId"]["enum"]
    assert _line_id(3, "") not in line_schema["startLineId"]["enum"]


def test_target_absent_extractable_row_must_keep_occupied_slot() -> None:
    source = "--- PAGE 1 ---\nCARGO: 25 CARTONS\nCONTAINER: ABCU1234567\nFREIGHT PREPAID\n"
    workspace = _workspace(source)

    with pytest.raises(ValueError, match="changes source line count"):
        apply_line_range_replacements(
            workspace,
            (_edit(source, 3, 3, ""),),
        )

    assert workspace.current_text == source


def test_target_absent_extractable_row_accepts_coherent_flavor_in_same_slot() -> None:
    source = "--- PAGE 1 ---\nCARGO: 25 CARTONS\nTEMPERATURE: 3.0 C\nFREIGHT PREPAID\n"
    workspace = _workspace(source)

    apply_line_range_replacements(
        workspace,
        (_edit(source, 3, 3, "REEFER USED IN NON-OPERATING MODE"),),
    )

    assert workspace.current_text == (
        "--- PAGE 1 ---\nCARGO: 25 CARTONS\nREEFER USED IN NON-OPERATING MODE\nFREIGHT PREPAID\n"
    )


def test_populated_line_cannot_be_replaced_with_a_blank_line() -> None:
    source = "--- PAGE 1 ---\nA\n\nOPTIONAL VALUE\n\nB\n"
    workspace = _workspace(source)

    with pytest.raises(ValueError, match="occupied/blank line topology"):
        apply_line_range_replacements(
            workspace,
            (_edit(source, 4, 4, " "),),
        )

    assert workspace.current_text == source


def test_atomic_patch_rejects_more_than_one_additional_blank_line() -> None:
    source = "--- PAGE 1 ---\nA\n\nB\n"
    workspace = _workspace(source)

    with pytest.raises(ValueError, match="changes source line count"):
        apply_line_range_replacements(
            workspace,
            (_edit(source, 4, 4, "\n\nB"),),
        )


def test_atomic_patch_rejects_collapsing_multiple_populated_lines() -> None:
    source = "--- PAGE 1 ---\nPARTY NAME\nADDRESS LINE\nCOUNTRY\n"

    with pytest.raises(ValueError, match="changes source line count"):
        apply_line_range_replacements(
            _workspace(source),
            (_edit(source, 2, 4, "NEW PARTY\nNEW COUNTRY"),),
        )


def test_inline_slot_requirements_distinguish_populated_and_empty_values() -> None:
    source = (
        "--- PAGE 1 ---\n"
        "Phone: 845-341-2100 Fax: 845-341-2121\n"
        "Phone: +20238204293 Fax:\n"
        "Phone: Fax:\n"
    )

    requirements = inline_slot_topology_requirements(source)

    assert [(row.lineId, row.labelSurface, row.sourcePopulated) for row in requirements] == [
        ("L00002", "Phone:", True),
        ("L00002", "Fax:", True),
        ("L00003", "Phone:", True),
        ("L00003", "Fax:", False),
        ("L00004", "Phone:", False),
        ("L00004", "Fax:", False),
    ]


def test_atomic_patch_rejects_filling_previously_empty_inline_slot() -> None:
    source = "--- PAGE 1 ---\nPhone: +20238204293 Fax:\n"

    with pytest.raises(ValueError, match="inline labeled slot"):
        apply_line_range_replacements(
            _workspace(source),
            (_edit(source, 2, 2, "Phone: +49 231 68492017 Fax: +49 231 68492018"),),
        )


def test_atomic_patch_rejects_emptying_previously_populated_inline_slot() -> None:
    source = "--- PAGE 1 ---\nPhone: 845-341-2100 Fax: 845-341-2121\n"

    with pytest.raises(ValueError, match="inline labeled slot"):
        apply_line_range_replacements(
            _workspace(source),
            (_edit(source, 2, 2, "Phone: +39 06 9487 3621 Fax:"),),
        )


def test_changed_contact_values_must_keep_exact_occurrence_counts() -> None:
    source = "--- PAGE 1 ---\nPhone: 111-111 Fax: 222-222\n"
    leaves = (
        ChangedLeaf(
            path="documentPatch.parties.shipper.contactDetails.phoneNumbers[0]",
            sourcePresent=True,
            targetPresent=True,
            sourceValue="111-111",
            targetValue="333-333",
            evidenceClass="printed_fact",
            requiresTextEdit=True,
        ),
        ChangedLeaf(
            path="documentPatch.parties.shipper.contactDetails.faxNumbers[0]",
            sourcePresent=True,
            targetPresent=True,
            sourceValue="222-222",
            targetValue="444-444",
            evidenceClass="printed_fact",
            requiresTextEdit=True,
        ),
    )
    requirements = target_value_occurrence_requirements(source, leaves)
    workspace = RewriteWorkspace(
        original_text=source,
        current_text=source,
        target_value_occurrence_requirements=requirements,
    )

    with pytest.raises(ValueError, match="duplicates or omits role-bound party values"):
        apply_line_range_replacements(
            workspace,
            (_edit(source, 2, 2, "Phone: 333-333 Fax: 333-333"),),
        )


def test_shared_source_contact_occurrences_are_partitioned_by_party_role() -> None:
    source = "--- PAGE 1 ---\nCONSIGNEE\nPhone: 111-111\nNOTIFY PARTY\nPhone: 111-111\n"
    leaves = (
        ChangedLeaf(
            path="documentPatch.parties.consignee.contactDetails.phoneNumbers[0]",
            sourcePresent=True,
            targetPresent=True,
            sourceValue="111-111",
            targetValue="222-222",
            evidenceClass="printed_fact",
            requiresTextEdit=True,
        ),
        ChangedLeaf(
            path="documentPatch.parties.notifyParties[0].contactDetails.phoneNumbers[0]",
            sourcePresent=True,
            targetPresent=True,
            sourceValue="111-111",
            targetValue="333-333",
            evidenceClass="printed_fact",
            requiresTextEdit=True,
        ),
    )

    requirements = target_value_occurrence_requirements(source, leaves)

    assert [(row.targetValue, row.requiredOccurrences) for row in requirements] == [
        ("222-222", 1),
        ("333-333", 1),
    ]


def test_ambiguous_shared_contact_cardinality_fails_closed_before_api_call() -> None:
    source = "--- PAGE 1 ---\nPhone: 111-111\n"
    leaves = (
        ChangedLeaf(
            path="documentPatch.parties.consignee.contactDetails.phoneNumbers[0]",
            sourcePresent=True,
            targetPresent=True,
            sourceValue="111-111",
            targetValue="222-222",
            evidenceClass="printed_fact",
            requiresTextEdit=True,
        ),
        ChangedLeaf(
            path="documentPatch.parties.notifyParties[0].contactDetails.phoneNumbers[0]",
            sourcePresent=True,
            targetPresent=True,
            sourceValue="111-111",
            targetValue="333-333",
            evidenceClass="printed_fact",
            requiresTextEdit=True,
        ),
    )

    with pytest.raises(ValueError, match="cannot be assigned exactly"):
        target_value_occurrence_requirements(source, leaves)


def test_exact_label_scalar_must_replace_its_original_printed_line() -> None:
    source = "--- PAGE 1 ---\nEXPORT REFERENCES\n\nINVOICE NO: OLDREF123\n"
    leaves = (
        ChangedLeaf(
            path="documentPatch.forwardingAndExportReferences[0]",
            sourcePresent=True,
            targetPresent=True,
            sourceValue="OLDREF123",
            targetValue="NEWREF789",
            evidenceClass="printed_fact",
            requiresTextEdit=True,
        ),
    )
    requirements = anchored_scalar_replacement_requirements(source, leaves)
    workspace = RewriteWorkspace(
        original_text=source,
        current_text=source,
        anchored_scalar_replacement_requirements=requirements,
    )

    assert requirements[0].sourceLineIds == ("L00004",)
    with pytest.raises(ValueError, match="moves or omits an exact changed label scalar"):
        apply_line_range_replacements(
            workspace,
            (
                _edit(source, 2, 2, "EXPORT REFERENCES: NEWREF789"),
                _edit(source, 4, 4, "INVOICE NO: AUXREF456"),
            ),
        )

    assert workspace.current_text == source
    apply_line_range_replacements(
        workspace,
        (_edit(source, 4, 4, "INVOICE NO: NEWREF789"),),
    )
    assert "INVOICE NO: NEWREF789" in workspace.current_text


def test_exact_contact_scalar_is_anchored_to_its_printed_line() -> None:
    source = "--- PAGE 1 ---\nNOTIFY PARTY\nTEL: +20 3000\nEMAIL: OLD@EXAMPLE.ORG\n"
    leaves = (
        ChangedLeaf(
            path="documentPatch.parties.notifyParties[0].contactDetails.emailAddresses[0]",
            sourcePresent=True,
            targetPresent=True,
            sourceValue="OLD@EXAMPLE.ORG",
            targetValue="cargo@newnotify.co.th",
            evidenceClass="printed_fact",
            requiresTextEdit=True,
        ),
    )

    requirements = anchored_scalar_replacement_requirements(source, leaves)

    assert len(requirements) == 1
    assert requirements[0].sourceLineIds == ("L00004",)
    assert requirements[0].targetSurface == "cargo@newnotify.co.th"


def test_reviewer_cannot_reclassify_anchored_contact_as_auxiliary_flavor() -> None:
    source = "--- PAGE 1 ---\nEMAIL: OLD@EXAMPLE.ORG\n"
    current = "--- PAGE 1 ---\nEMAIL: cargo@newnotify.co.th\n"
    leaves = (
        ChangedLeaf(
            path="documentPatch.parties.notifyParties[0].contactDetails.emailAddresses[0]",
            sourcePresent=True,
            targetPresent=True,
            sourceValue="OLD@EXAMPLE.ORG",
            targetValue="cargo@newnotify.co.th",
            evidenceClass="printed_fact",
            requiresTextEdit=True,
        ),
    )
    workspace = RewriteWorkspace(
        original_text=source,
        current_text=current,
        anchored_scalar_replacement_requirements=anchored_scalar_replacement_requirements(
            source, leaves
        ),
    )
    finding = SemanticReviewFinding(
        category="incomplete_auxiliary_anonymization",
        affectedPaths=("auxiliary.cargo_email",),
        currentEvidence=("EMAIL: cargo@newnotify.co.th",),
        correction="Replace this with a distinct fictional auxiliary email.",
    )

    assert _finding_conflicts_with_anchored_scalar_authority(finding, workspace)


def test_anchored_scalar_authority_does_not_hide_a_derived_fact_mismatch() -> None:
    source = "--- PAGE 1 ---\nGROSS WEIGHT: 1000.000 KG\n"
    current = "--- PAGE 1 ---\nGROSS WEIGHT: 900.000 KG\n"
    leaves = (
        ChangedLeaf(
            path="documentPatch.cargoGroups[0].grossWeight.value",
            sourcePresent=True,
            targetPresent=True,
            sourceValue=1000,
            targetValue=900,
            evidenceClass="printed_fact",
            requiresTextEdit=True,
        ),
    )
    workspace = RewriteWorkspace(
        original_text=source,
        current_text=current,
        anchored_scalar_replacement_requirements=anchored_scalar_replacement_requirements(
            source, leaves
        ),
    )
    finding = SemanticReviewFinding(
        category="derived_fact_mismatch",
        affectedPaths=("documentPatch.cargoGroups[0].grossWeight",),
        currentEvidence=("GROSS WEIGHT: 900.000 KG",),
        correction="Make the printed breakdown sum to the target aggregate.",
    )

    assert not _finding_conflicts_with_anchored_scalar_authority(finding, workspace)


def test_anchored_scalar_authority_does_not_hide_a_stale_neighboring_identifier() -> None:
    source = "--- PAGE 1 ---\nMSKU1675513 84096 9702638\n"
    current = "--- PAGE 1 ---\nMSKU2159954 84096 1340561 40' HIGH CUBE\n"
    leaves = (
        ChangedLeaf(
            path="documentPatch.containers[0].containerNumber",
            sourcePresent=True,
            targetPresent=True,
            sourceValue="MSKU1675513",
            targetValue="MSKU2159954",
            evidenceClass="printed_fact",
            requiresTextEdit=True,
        ),
    )
    workspace = RewriteWorkspace(
        original_text=source,
        current_text=current,
        anchored_scalar_replacement_requirements=anchored_scalar_replacement_requirements(
            source, leaves
        ),
    )
    finding = SemanticReviewFinding(
        category="stale_source_fact",
        affectedPaths=("auxiliary.container_file_id",),
        currentEvidence=("MSKU2159954 84096 1340561 40' HIGH CUBE",),
        correction="Replace the stale file identifier 84096 without changing the container.",
    )

    assert not _finding_conflicts_with_anchored_scalar_authority(finding, workspace)


def test_party_names_are_not_overconstrained_to_one_source_line() -> None:
    source = "--- PAGE 1 ---\nSHIPPER: SOURCE EXPORTS\n"
    leaves = (
        ChangedLeaf(
            path="documentPatch.parties.shipper.name",
            sourcePresent=True,
            targetPresent=True,
            sourceValue="SOURCE EXPORTS",
            targetValue="TARGET INTERNATIONAL EXPORTS",
            evidenceClass="printed_fact",
            requiresTextEdit=True,
        ),
    )

    assert anchored_scalar_replacement_requirements(source, leaves) == ()


def test_cross_role_surface_with_different_targets_is_not_globally_anchored() -> None:
    source = "Notify Party\n*EGYPT\n\nMARKS\n*EGYPT\n"
    leaves = (
        ChangedLeaf(
            path="documentPatch.cargoGroups[0].marksAndNumbers[0]",
            sourcePresent=True,
            targetPresent=True,
            sourceValue="*EGYPT",
            targetValue="ORION PAPER / R-537",
            evidenceClass="printed_fact",
            requiresTextEdit=True,
        ),
        ChangedLeaf(
            path="documentPatch.parties.notifyParties[0].country",
            sourcePresent=True,
            targetPresent=True,
            sourceValue="EGYPT",
            targetValue="Egypt",
            evidenceClass="printed_fact",
            requiresTextEdit=True,
        ),
    )

    assert anchored_scalar_replacement_requirements(source, leaves) == ()


def test_punctuated_mark_anchor_does_not_capture_plain_country_occurrences() -> None:
    source = "Notify Party\nCAIRO, EGYPT 11361\n\nMARKS\n*EGYPT\n"
    leaves = (
        ChangedLeaf(
            path="documentPatch.cargoGroups[0].marksAndNumbers[0]",
            sourcePresent=True,
            targetPresent=True,
            sourceValue="*EGYPT",
            targetValue="ORION PAPER / R-537",
            evidenceClass="printed_fact",
            requiresTextEdit=True,
        ),
    )

    requirements = anchored_scalar_replacement_requirements(source, leaves)

    assert len(requirements) == 1
    assert requirements[0].sourceLineIds == ("L00005",)


def test_repeated_party_address_must_appear_once_per_repeated_party_block() -> None:
    source = (
        "--- PAGE 1 ---\nCONSIGNEE: OLD COMPANY\nOLD STREET\nOLD CITY\n"
        "--- PAGE 2 ---\nCONSIGNEE: OLD COMPANY\nOLD STREET\nOLD CITY\n"
    )
    leaves = (
        ChangedLeaf(
            path="documentPatch.parties.consignee.name",
            sourcePresent=True,
            targetPresent=True,
            sourceValue="OLD COMPANY",
            targetValue="NEW COMPANY",
            evidenceClass="printed_fact",
            requiresTextEdit=True,
        ),
        ChangedLeaf(
            path="documentPatch.parties.consignee.address",
            sourcePresent=True,
            targetPresent=True,
            sourceValue="OLD STREET",
            targetValue="NEW STREET 27",
            evidenceClass="printed_fact",
            requiresTextEdit=True,
        ),
    )
    requirements = target_value_occurrence_requirements(source, leaves)
    workspace = RewriteWorkspace(
        original_text=source,
        current_text=source,
        target_value_occurrence_requirements=requirements,
    )

    assert {row.targetValue: row.requiredOccurrences for row in requirements} == {
        "NEW COMPANY": 2,
        "NEW STREET 27": 2,
    }
    with pytest.raises(ValueError, match="role-bound party values"):
        apply_line_range_replacements(
            workspace,
            (
                _edit(source, 2, 4, "CONSIGNEE: NEW COMPANY\nNEW STREET 27\nNEW STREET 27"),
                _edit(source, 6, 8, "CONSIGNEE: NEW COMPANY\nNEW STREET 27\nNEW CITY"),
            ),
        )


def test_party_name_occurrences_include_mixed_case_prefixed_copies() -> None:
    source = (
        "--- PAGE 1 ---\nMEDITERRANEAN SHIPPING COMPANY S.A.\n"
        "SIGNED FOR MSC Mediterranean Shipping Company S.A.\n"
    )
    leaves = (
        ChangedLeaf(
            path="documentPatch.parties.carrier.name",
            sourcePresent=True,
            targetPresent=True,
            sourceValue="MEDITERRANEAN SHIPPING COMPANY S.A.",
            targetValue="Aurelia Tideway Carriers Ltd.",
            evidenceClass="printed_fact",
            requiresTextEdit=True,
        ),
    )

    requirements = target_value_occurrence_requirements(source, leaves)

    assert requirements[0].requiredOccurrences == 2


def test_party_name_occurrences_do_not_count_email_domain_tokens() -> None:
    source = (
        "--- PAGE 1 ---\n"
        "Delivery Agent\n"
        "MLH SHIPPING\n"
        "MLH SHIPPING 11 BENNY STREET\n"
        "EMAIL: ***@mlh-shipping.com\n"
    )
    leaves = (
        ChangedLeaf(
            path="documentPatch.parties.deliveryAgent.name",
            sourcePresent=True,
            targetPresent=True,
            sourceValue="MLH SHIPPING",
            targetValue="Vietora Delivery Logistics Co., Ltd.",
            evidenceClass="printed_fact",
            requiresTextEdit=True,
        ),
        ChangedLeaf(
            path="documentPatch.parties.deliveryAgent.address",
            sourcePresent=True,
            targetPresent=True,
            sourceValue="11 BENNY STREET",
            targetValue="42 Nguyen Xuan On Street, Ward 3",
            evidenceClass="printed_fact",
            requiresTextEdit=True,
        ),
    )

    requirements = target_value_occurrence_requirements(source, leaves)

    assert {row.targetValue: row.requiredOccurrences for row in requirements} == {
        "Vietora Delivery Logistics Co., Ltd.": 2,
        "42 Nguyen Xuan On Street, Ward 3": 1,
    }


def test_party_address_occurrence_count_accepts_source_and_target_line_wrapping() -> None:
    source = (
        "--- PAGE 1 ---\nCONSIGNEE: OLD COMPANY\n"
        "12 LONG INDUSTRIAL\nROAD BUILDING 7\n"
    )
    leaves = (
        ChangedLeaf(
            path="documentPatch.parties.consignee.name",
            sourcePresent=True,
            targetPresent=True,
            sourceValue="OLD COMPANY",
            targetValue="NEW COMPANY",
            evidenceClass="printed_fact",
            requiresTextEdit=True,
        ),
        ChangedLeaf(
            path="documentPatch.parties.consignee.address",
            sourcePresent=True,
            targetPresent=True,
            sourceValue="12 LONG INDUSTRIAL ROAD BUILDING 7",
            targetValue="42 HARBOR AVENUE WAREHOUSE 3",
            evidenceClass="printed_fact",
            requiresTextEdit=True,
        ),
    )
    requirements = target_value_occurrence_requirements(source, leaves)
    workspace = RewriteWorkspace(
        original_text=source,
        current_text=source,
        target_value_occurrence_requirements=requirements,
    )

    apply_line_range_replacements(
        workspace,
        (
            _edit(
                source,
                2,
                4,
                "CONSIGNEE: NEW COMPANY\n42 HARBOR AVENUE\nWAREHOUSE 3",
            ),
        ),
    )

    assert workspace.current_text.endswith("42 HARBOR AVENUE\nWAREHOUSE 3\n")


def test_post_relation_signing_agent_is_auxiliary_not_a_second_party_copy() -> None:
    source = (
        "Notify Party\nSCHENKER FRANCE SAS\n\n"
        "(23) Issued as agents for The Great Ocean Line Pte. Ltd. as Carrier by:\n"
        "SCHENKER FRANCE SAS\n"
    )
    source_label = {
        "documentPatch": {
            "parties": {
                "carrier": {"name": "The Great Ocean Line Pte. Ltd."},
                "notifyParties": [{"name": "SCHENKER FRANCE SAS"}],
            }
        }
    }
    target_label = {
        "documentPatch": {
            "parties": {
                "carrier": {"name": "Meridian Crest Shipping Ltd."},
                "notifyParties": [{"name": "Siam Rivergate Commerce Co., Ltd."}],
            }
        }
    }
    leaves = rewrite_changed_leaves(source_label, target_label)
    auxiliary = raw_auxiliary_identity_requirements(source, source_label, target_label)

    assert len(auxiliary) == 1
    assert auxiliary[0].sourceIdentity == "SCHENKER FRANCE SAS"
    assert auxiliary[0].targetPrincipalName == "Meridian Crest Shipping Ltd."
    requirements = target_value_occurrence_requirements(source, leaves, auxiliary)
    notify_requirement = next(
        row for row in requirements if row.targetValue == "Siam Rivergate Commerce Co., Ltd."
    )
    assert notify_requirement.requiredOccurrences == 1

    workspace = RewriteWorkspace(
        original_text=source,
        current_text=source,
        source_label=source_label,
        upstream_target_label=target_label,
        current_target_label=target_label,
        target_value_occurrence_requirements=requirements,
        raw_auxiliary_identity_requirements=auxiliary,
    )
    apply_line_range_replacements(
        workspace,
        (
            _edit(source, 2, 2, "Siam Rivergate Commerce Co., Ltd."),
            _edit(
                source,
                4,
                5,
                "(23) Issued as agents for Meridian Crest Shipping Ltd. as Carrier by:\n"
                "Harborline Agency Services Ltd.",
            ),
        ),
    )
    assert "Siam Rivergate Commerce Co., Ltd." in workspace.current_text
    assert "Harborline Agency Services Ltd." in workspace.current_text


def test_split_signed_on_behalf_agent_is_an_explicit_auxiliary_identity() -> None:
    source = (
        "--- PAGE 1 ---\n"
        "SIGNED on behalf of the Carrier OLD OCEAN LINE S.A.\n"
        "by Old Shipping (Aust) Pty Ltd as Agent ABN 12003 760 638\n"
    )
    source_label = {
        "documentPatch": {"parties": {"carrier": {"name": "OLD OCEAN LINE S.A."}}}
    }
    target_label = {
        "documentPatch": {"parties": {"carrier": {"name": "NEW OCEAN LINE AG"}}}
    }
    requirements = raw_auxiliary_identity_requirements(source, source_label, target_label)

    assert len(requirements) == 1
    assert requirements[0].sourceIdentity == "Old Shipping (Aust) Pty Ltd"
    assert requirements[0].targetPrincipalName == "NEW OCEAN LINE AG"
    workspace = RewriteWorkspace(
        original_text=source,
        current_text=source,
        source_label=source_label,
        current_target_label=target_label,
        raw_auxiliary_identity_requirements=requirements,
    )
    apply_line_range_replacements(
        workspace,
        (
            _edit(
                source,
                2,
                3,
                "SIGNED on behalf of the Carrier NEW OCEAN LINE AG\n"
                "by Neue Hafenlogistik GmbH as Agent HRB 481729",
            ),
        ),
    )

    assert workspace.commits[0].rawAuxiliaryIdentityRealizations[0].renderedIdentity == (
        "Neue Hafenlogistik GmbH"
    )


def test_presentation_only_label_difference_does_not_require_ocr_edit() -> None:
    source = {"documentPatch": {"parties": {"consignee": {"country": "EGYPT"}}}}
    target = {"documentPatch": {"parties": {"consignee": {"country": "Egypt"}}}}

    assert rewrite_changed_leaves(source, target) == ()


def test_unchanged_status_requirements_protect_express_release() -> None:
    source = "--- PAGE 1 ---\nPLACE OF DELIVERY\nEXPRESS RELEASE\nNON-NEGOTIABLE\n"
    requirements = source_status_preservation_requirements(source, ())
    workspace = RewriteWorkspace(
        original_text=source,
        current_text=source,
        source_status_requirements=requirements,
    )

    assert [(row.sourceSurface, row.sourceOccurrences) for row in requirements] == [
        ("EXPRESS RELEASE", 1),
        ("NON-NEGOTIABLE", 1),
    ]
    with pytest.raises(ValueError, match="unchanged source status"):
        apply_line_range_replacements(
            workspace,
            (_edit(source, 3, 3, "NON-NEGOTIABLE"),),
        )


def test_target_controlled_status_is_not_protected() -> None:
    source = "--- PAGE 1 ---\nFREIGHT COLLECT\n"
    leaf = ChangedLeaf(
        path="documentPatch.freightPayment.arrangement",
        sourcePresent=True,
        targetPresent=True,
        sourceValue="COLLECT",
        targetValue="PREPAID",
        evidenceClass="printed_fact",
        requiresTextEdit=True,
    )

    assert source_status_preservation_requirements(source, (leaf,)) == ()


def test_unchanged_legal_boilerplate_and_bill_cardinality_are_byte_protected() -> None:
    surrender = (
        "The surrender of the original order bill of lading properly endorsed shall be "
        "required before delivery by the Carrier."
    )
    execution = (
        "IN WITNESS WHEREOF THE UNDERSIGNED HAS SIGNED THREE(3) BILLS OF LADING "
        "ALL OF THE SAME TENOR AND DATE ONE OF WHICH BEING ACCOMPLISHED THE OTHERS "
        "TO STAND VOID."
    )
    source = f"--- PAGE 1 ---\n{surrender}\n{execution}\n"

    requirements = source_status_preservation_requirements(source, ())

    assert [(row.sourceSurface, row.sourceOccurrences) for row in requirements] == [
        (execution, 1),
        (surrender, 1),
    ]
    workspace = RewriteWorkspace(
        original_text=source,
        current_text=source,
        source_status_requirements=requirements,
    )
    with pytest.raises(ValueError, match="unchanged source status"):
        apply_line_range_replacements(
            workspace,
            (_edit(source, 3, 3, "HAS SIGNED ONE BILL OF LADING."),),
        )


def test_changed_negotiability_releases_legal_boilerplate_for_rewrite() -> None:
    source = (
        "--- PAGE 1 ---\n"
        "The surrender of the original order bill of lading properly endorsed shall be "
        "required before delivery by the Carrier.\n"
    )
    leaf = ChangedLeaf(
        path="documentPatch.negotiability",
        sourcePresent=True,
        targetPresent=True,
        sourceValue="negotiable",
        targetValue="non_negotiable",
        evidenceClass="printed_fact",
        requiresTextEdit=True,
    )

    assert source_status_preservation_requirements(source, (leaf,)) == ()


def test_deterministic_audit_surfaces_old_label_values_but_passes_core() -> None:
    source = "--- PAGE 1 ---\nB/L: OLD123\nCOPY: OLD123\n"
    workspace = _workspace(source)
    apply_line_range_replacements(
        workspace,
        (_edit(source, 2, 2, "B/L: NEW789"),),
    )
    leaf = ChangedLeaf(
        path="documentPatch.billOfLadingNumber",
        sourcePresent=True,
        targetPresent=True,
        sourceValue="OLD123",
        targetValue="NEW789",
        evidenceClass="printed_fact",
        requiresTextEdit=True,
    )

    audit = deterministic_rewrite_audit(workspace, (leaf,))

    assert audit.core_passed is True
    assert len(audit.residualCandidates) == 1
    assert audit.residualCandidates[0].currentOccurrences == 1


def test_deterministic_audit_accepts_equivalent_v3_equipment_surface() -> None:
    source = "--- PAGE 1 ---\nCONT: NEW1234567 40HQ\n"
    target = {
        "schemaVersion": "5.0.0-experimental",
        "documentPatch": {
            "containers": [
                {
                    "containerNumber": "NEW1234567",
                    "sizeCategory": "FORTY_FOOT_HIGH_CUBE",
                    "typeCategory": "GENERAL_PURPOSE",
                }
            ]
        },
    }
    workspace = RewriteWorkspace(
        original_text=source,
        current_text=source,
        current_target_label=target,
    )
    leaf = ChangedLeaf(
        path="documentPatch.containers[0].typeDescription",
        sourcePresent=True,
        targetPresent=False,
        sourceValue="40HQ",
        targetValue=None,
        evidenceClass="printed_fact",
        requiresTextEdit=True,
    )

    audit = deterministic_rewrite_audit(workspace, (leaf,))

    assert audit.residualCandidates == ()


def test_deterministic_audit_retains_mismatched_v3_equipment_surface_signal() -> None:
    source = "--- PAGE 1 ---\nCONT: NEW1234567 40HQ\n"
    target = {
        "schemaVersion": "5.0.0-experimental",
        "documentPatch": {
            "containers": [
                {
                    "containerNumber": "NEW1234567",
                    "sizeCategory": "FORTY_FOOT_HIGH_CUBE",
                    "typeCategory": "REFRIGERATED",
                }
            ]
        },
    }
    workspace = RewriteWorkspace(
        original_text=source,
        current_text=source,
        current_target_label=target,
    )
    leaf = ChangedLeaf(
        path="documentPatch.containers[0].typeDescription",
        sourcePresent=True,
        targetPresent=False,
        sourceValue="40HQ",
        targetValue=None,
        evidenceClass="printed_fact",
        requiresTextEdit=True,
    )

    audit = deterministic_rewrite_audit(workspace, (leaf,))

    assert [row.sourceValue for row in audit.residualCandidates] == ["40HQ"]


def test_label_change_contract_projects_equivalent_equipment_surface_once() -> None:
    source = {
        "schemaVersion": "3.0.0",
        "documentPatch": {
            "containers": [
                {
                    "containerNumber": "NEW1234567",
                    "typeDescription": "40HQ",
                }
            ]
        },
    }
    target = {
        "schemaVersion": "5.0.0-experimental",
        "documentPatch": {
            "containers": [
                {
                    "containerNumber": "NEW1234567",
                    "sizeCategory": "FORTY_FOOT_HIGH_CUBE",
                    "typeCategory": "GENERAL_PURPOSE",
                }
            ]
        },
    }
    workspace = RewriteWorkspace(
        original_text="--- PAGE 1 ---\nCONT: NEW1234567 40HQ\n",
        current_text="--- PAGE 1 ---\nCONT: NEW1234567 40HQ\n",
        source_label=source,
        current_target_label=target,
    )

    contract = label_change_contract(workspace)

    assert [row.model_dump(mode="json") for row in contract] == [
        {
            "path": "documentPatch.containers[0].printedEquipmentSurface",
            "action": "preserve_equivalent_equipment_surface",
            "sourceValue": "40HQ",
            "targetValue": {
                "sizeCategory": "FORTY_FOOT_HIGH_CUBE",
                "typeCategory": "GENERAL_PURPOSE",
            },
        }
    ]


def test_label_change_contract_requires_natural_replacement_for_mismatched_equipment() -> None:
    source = {
        "schemaVersion": "3.0.0",
        "documentPatch": {"containers": [{"typeDescription": "40HQ"}]},
    }
    target = {
        "schemaVersion": "5.0.0-experimental",
        "documentPatch": {
            "containers": [
                {
                    "sizeCategory": "FORTY_FOOT_HIGH_CUBE",
                    "typeCategory": "REFRIGERATED",
                }
            ]
        },
    }
    workspace = RewriteWorkspace(
        original_text="--- PAGE 1 ---\nEQUIPMENT: 40HQ\n",
        current_text="--- PAGE 1 ---\nEQUIPMENT: 40HQ\n",
        source_label=source,
        current_target_label=target,
    )

    contract = label_change_contract(workspace)

    assert len(contract) == 1
    assert contract[0].action == "replace_equipment_surface"
    assert contract[0].targetValue == {
        "sizeCategory": "FORTY_FOOT_HIGH_CUBE",
        "typeCategory": "REFRIGERATED",
    }


def test_label_change_contract_projects_target_only_equipment_categories_once() -> None:
    source = {
        "schemaVersion": "3.0.0",
        "documentPatch": {"containers": [{"containerNumber": "NEW1234567"}]},
    }
    target = {
        "schemaVersion": "5.0.0-experimental",
        "documentPatch": {
            "containers": [
                {
                    "containerNumber": "NEW1234567",
                    "sizeCategory": "TWENTY_FOOT_STANDARD_HEIGHT",
                    "typeCategory": "GENERAL_PURPOSE",
                }
            ]
        },
    }
    workspace = RewriteWorkspace(
        original_text="--- PAGE 1 ---\nCONTAINER: NEW1234567\n",
        current_text="--- PAGE 1 ---\nCONTAINER: NEW1234567\n",
        source_label=source,
        current_target_label=target,
    )

    contract = label_change_contract(workspace)

    assert [row.model_dump(mode="json") for row in contract] == [
        {
            "path": "documentPatch.containers[0].printedEquipmentSurface",
            "action": "add_equipment_surface",
            "sourceValue": None,
            "targetValue": {
                "sizeCategory": "TWENTY_FOOT_STANDARD_HEIGHT",
                "typeCategory": "GENERAL_PURPOSE",
            },
        }
    ]


def test_label_change_contract_deactivates_target_absent_field_without_deleting_slot() -> None:
    source = {
        "schemaVersion": "3.0.0",
        "documentPatch": {
            "containers": [{"temperatureSetpoint": {"value": 3.0, "unit": "celsius"}}]
        },
    }
    target = {
        "schemaVersion": "5.0.0-experimental",
        "documentPatch": {"containers": [{}]},
    }
    workspace = RewriteWorkspace(
        original_text="--- PAGE 1 ---\nTEMPERATURE: 3.0 C\n",
        current_text="--- PAGE 1 ---\nTEMPERATURE: 3.0 C\n",
        source_label=source,
        current_target_label=target,
    )

    contract = label_change_contract(workspace)

    assert contract[0].action == "deactivate_extractable_fact_preserve_slot"
    assert contract[0].targetValue is None


def test_label_change_contract_keeps_reviewed_marks_target_controlled() -> None:
    source = {
        "schemaVersion": "3.0.0",
        "documentPatch": {
            "cargoGroups": [
                {"marksAndNumbers": ["MANUFACTURER: SOURCE LTD", "ATTENTION: OLD NAME"]}
            ]
        },
    }
    target = {
        "schemaVersion": "5.0.0-experimental",
        "documentPatch": {"cargoGroups": [{"marksAndNumbers": ["LOT: X7-412", "BATCH: Q9-775"]}]},
    }
    workspace = RewriteWorkspace(
        original_text="--- PAGE 1 ---\nMANUFACTURER: SOURCE LTD\nATTENTION: OLD NAME\n",
        current_text="--- PAGE 1 ---\nMANUFACTURER: SOURCE LTD\nATTENTION: OLD NAME\n",
        source_label=source,
        current_target_label=target,
    )

    contract = label_change_contract(workspace)

    assert [(row.path, row.action, row.sourceValue, row.targetValue) for row in contract] == [
        (
            "documentPatch.cargoGroups[0].marksAndNumbers[0]",
            "replace",
            "MANUFACTURER: SOURCE LTD",
            "LOT: X7-412",
        ),
        (
            "documentPatch.cargoGroups[0].marksAndNumbers[1]",
            "replace",
            "ATTENTION: OLD NAME",
            "BATCH: Q9-775",
        ),
    ]


def test_review_pass_requires_every_explicit_check() -> None:
    with pytest.raises(ValidationError, match="pass requires every checklist item"):
        SemanticReviewReceipt(
            verdict="pass",
            targetFacts="pass",
            staleSourceFacts="pass",
            auxiliaryFlavor="fail",
            cargoAndOperationalRealism="pass",
            legalTopology="pass",
            formattingAndMinimality="pass",
            findings=(),
        )


def test_review_revision_requires_failed_check_and_finding() -> None:
    finding = SemanticReviewFinding(
        category="incomplete_auxiliary_anonymization",
        affectedPaths=("auxiliary.signingAgent",),
        currentEvidence=("SIGNED ON BEHALF OF OLD CARRIER",),
        correction="Replace the source signing identity with a fictional identity.",
    )
    review = SemanticReviewReceipt(
        verdict="revise",
        targetFacts="pass",
        staleSourceFacts="pass",
        auxiliaryFlavor="fail",
        cargoAndOperationalRealism="pass",
        legalTopology="pass",
        formattingAndMinimality="pass",
        findings=(finding,),
    )

    assert review.verdict == "revise"
    verdict_schema = SemanticReviewReceipt.model_json_schema()["properties"]["verdict"]
    assert "blocked" not in verdict_schema["enum"]


def test_reviewer_contract_does_not_echo_hash_or_source_evidence() -> None:
    source = "--- PAGE 1 ---\nB/L: OLD123\n"
    workspace = _workspace(source)
    apply_line_range_replacements(
        workspace,
        (_edit(source, 2, 2, "B/L: NEW789"),),
    )
    audit = deterministic_rewrite_audit(workspace, ())

    payload = _review_audit_payload(audit)
    receipt_schema = SemanticReviewReceipt.model_json_schema()
    finding_schema = receipt_schema["$defs"]["SemanticReviewFinding"]

    assert "currentTextSha256" not in payload
    assert "currentTextSha256" not in receipt_schema["properties"]
    assert "sourceEvidence" not in finding_schema["properties"]


def test_prompts_require_populated_slots_and_preserve_distinct_identities() -> None:
    editor = Path("prompts/synthesis/mpci_bl_raw_text_atomic_editor_v5.md").read_text()
    reviewer = Path("prompts/synthesis/mpci_bl_raw_text_atomic_reviewer_v5.md").read_text()
    normalized_editor = " ".join(editor.split())
    normalized_reviewer = " ".join(reviewer.split())

    assert (
        "A missing auxiliary identity is ordinary generation work. Invent it." in normalized_editor
    )
    assert "Never use `N/A`, `UNAVAILABLE`, `UNKNOWN`, `TBD`" in normalized_editor
    assert "operationalCapacityLimits" in normalized_editor
    assert "source label and `labelChangeContract` are authoritative" in normalized_editor
    assert "converted to `N/A`, `UNAVAILABLE`, or vague filler" in normalized_reviewer
    assert "sourceSemanticRoleHint" in normalized_editor
    assert "may not become a mark" in normalized_reviewer
    assert "do not use an intuitive or generic container-volume estimate" in normalized_reviewer


@pytest.mark.parametrize(
    ("filename", "editor_model"),
    (
        ("mpci_bl_raw_text_atomic10_luna_high.yaml", "gpt-5.6-luna"),
        ("mpci_bl_raw_text_atomic10_terra_high.yaml", "gpt-5.6-terra"),
    ),
)
def test_comparison_configs_select_identical_ten_cases(filename: str, editor_model: str) -> None:
    config = load_synthesis_raw_text_rewrite_cycle_probe_config(
        Path("configs/synthesis") / filename
    )

    assert len(config.cases) == 10
    assert config.schema_version == 12
    assert config.providers.editor.model == editor_model
    assert config.providers.reviewer.model == "gpt-5.6-luna"
    assert config.providers.editor.reasoning_effort == "high"
    assert config.providers.reviewer.reasoning_effort == "low"
    assert config.workflow.max_editor_requests_per_pass == 3
    assert config.workflow.editor_output_retries == 2
    assert config.workflow.max_correction_cycles == 3
    assert config.workflow.require_terminal_atomic_edit_output is True
    assert config.workflow.require_line_addressed_atomic_patches is True
    assert config.workflow.synthesize_auxiliary_flavor is True
    assert config.workflow.preserve_auxiliary_slot_topology is True
    assert config.workflow.preserve_source_line_count is True
    assert config.workflow.preserve_source_blank_line_topology is True
    assert config.workflow.preserve_inline_labeled_slot_population is True
    assert config.workflow.preserve_unchanged_source_status_surfaces is True
    assert config.workflow.enforce_target_value_occurrence_counts is True
    assert config.workflow.synthesize_raw_auxiliary_agent_identities is True
    assert config.workflow.enforce_carrier_receipt_equipment_breakdown is True
    assert config.workflow.preserve_target_label_and_synthesize_compound_party_flavor is True
    assert config.workflow.provide_authoritative_label_change_contract is True
    assert config.workflow.remove_target_absent_extractable_assertions_naturally is True
    assert config.workflow.preserve_source_only_raw_slots_as_synthetic_flavor is True
    assert config.workflow.preserve_semantically_equivalent_status_surfaces is True
    assert config.workflow.forbid_new_unanchored_extractable_facts is True
    assert config.workflow.reject_placeholder_substitutions is True
    assert config.workflow.use_provider_explicit_prompt_cache is True
    assert config.workflow.publish_training_records is False


def test_comparison_configs_have_identical_cases_and_prompts() -> None:
    luna = load_synthesis_raw_text_rewrite_cycle_probe_config(
        Path("configs/synthesis/mpci_bl_raw_text_atomic10_luna_high.yaml")
    )
    terra = load_synthesis_raw_text_rewrite_cycle_probe_config(
        Path("configs/synthesis/mpci_bl_raw_text_atomic10_terra_high.yaml")
    )

    assert luna.cases == terra.cases
    assert luna.prompts == terra.prompts
    assert luna.target_integrity == terra.target_integrity


def test_rewrite_settings_use_stage_scoped_explicit_cache() -> None:
    config = load_synthesis_raw_text_rewrite_cycle_probe_config(
        Path("configs/synthesis/mpci_bl_raw_text_atomic10_luna_high.yaml")
    )

    settings = _settings(
        config.providers.editor,
        stage="editor",
        prompt_sha256=config.prompts.editor.sha256,
    )

    assert settings["openai_prompt_cache_options"] == {"mode": "explicit", "ttl": "30m"}
    assert settings["openai_reasoning_context"] == "current_turn"
    assert settings["openai_prompt_cache_key"] == (
        f"dococr:rewrite15:gpt-5.6-luna:e:{config.prompts.editor.sha256[:12]}"
    )
    assert len(settings["openai_prompt_cache_key"]) <= 64
    assert settings["openai_previous_response_id"] == "auto"


def test_rewrite_settings_do_not_reference_unstored_responses() -> None:
    config = load_synthesis_raw_text_rewrite_cycle_probe_config(
        Path("configs/synthesis/mpci_bl_raw_text_atomic10_luna_high.yaml")
    )

    settings = _settings(
        config.providers.editor.model_copy(update={"store_responses": False}),
        stage="editor",
        prompt_sha256=config.prompts.editor.sha256,
    )

    assert "openai_previous_response_id" not in settings


def test_explicit_cache_breakpoint_precedes_all_case_specific_content() -> None:
    content = _prompt_content(
        stage="editor",
        immutable={"documentId": "doc-private"},
        active={"sourceRawOcrLines": "PRIVATE OCR"},
    )

    assert content[0] == "Stable synthetic B/L atomic editor contract version 15."
    assert content[1].kind == "cache-point"
    assert "doc-private" not in str(content[:2])
    assert "PRIVATE OCR" in content[2]


def test_compact_change_contract_deduplicates_relational_aliases() -> None:
    compact = compact_label_change_contract(
        (
            LabelChangeDirective(
                path="documentPatch.containers[0].containerNumber",
                action="replace",
                sourceValue="OLDU1234567",
                targetValue="NEWU1234567",
            ),
            LabelChangeDirective(
                path="documentPatch.cargoAllocationGroups[0].allocations[0].containerNumber",
                action="replace",
                sourceValue="OLDU1234567",
                targetValue="NEWU1234567",
            ),
        )
    )

    assert compact == (
        CompactLabelChangeDirective(
            paths=(
                "documentPatch.containers[0].containerNumber",
                "documentPatch.cargoAllocationGroups[0].allocations[0].containerNumber",
            ),
            action="replace",
            sourceValue="OLDU1234567",
            targetValue="NEWU1234567",
        ),
    )


def test_date_and_hs_rendering_requirements_preserve_source_surface_style() -> None:
    raw = (
        "--- PAGE 1 ---\n"
        "SHIPPED ON BOARD DATE\n"
        "05/05/2024\n"
        "PLACE AND DATE OF ISSUE\n"
        "05/05/2024\n"
        "HS CODE: 2915.90.1800\n"
    )
    source = {
        "documentPatch": {
            "issueDate": "2024-05-05",
            "shippedOnBoardDate": "2024-05-05",
            "cargoGroups": [{"groupId": "g1", "hsCodes": ["2915901800"]}],
        }
    }
    target = {
        "documentPatch": {
            "issueDate": "2024-09-20",
            "shippedOnBoardDate": "2024-09-19",
            "cargoGroups": [{"groupId": "g1", "hsCodes": ["282619"]}],
        }
    }

    requirements = surface_rendering_requirements(raw, source, target)

    assert {(row.targetPath, row.targetSurface) for row in requirements} == {
        ("documentPatch.shippedOnBoardDate", "19/09/2024"),
        ("documentPatch.issueDate", "20/09/2024"),
        ("documentPatch.cargoGroups[0].hsCodes[0]", "2826.19"),
    }


def test_carrier_receipt_count_is_projected_from_the_corresponding_package_quantity() -> None:
    source = {
        "documentPatch": {"cargoPackages": [{"packageId": "p1", "groupId": "g1", "quantity": 5}]}
    }
    target = {
        "documentPatch": {"cargoPackages": [{"packageId": "p1", "groupId": "g1", "quantity": 7}]}
    }
    raw = "Total number of containers or packages 5 received by Carrier:\n"

    requirements = surface_rendering_requirements(raw, source, target)

    assert len(requirements) == 1
    assert requirements[0].kind == "carrier_receipt_count"
    assert requirements[0].targetSurface == (
        "Total number of containers or packages 7 received by Carrier:"
    )


def test_carrier_header_alias_is_bound_to_the_target_carrier_identity() -> None:
    raw = (
        "--- PAGE 1 ---\nMEDITERRANEAN SHIPPING COMPANY S.A.\n"
        "12-14, chemin Rieu\nBILL OF LADING\n"
        "SIGNED on behalf of the Carrier MSC Mediterranean Shipping Company S.A.\n"
    )
    source = {
        "documentPatch": {
            "parties": {"carrier": {"name": "MSC Mediterranean Shipping Company S.A."}}
        }
    }
    target = {"documentPatch": {"parties": {"carrier": {"name": "Helvetic Blueway Transport AG"}}}}
    leaves = rewrite_changed_leaves(source, target)
    surfaces = surface_rendering_requirements(raw, source, target)

    assert [(row.kind, row.sourceSurface, row.targetSurface) for row in surfaces] == [
        (
            "carrier_header_identity",
            "MEDITERRANEAN SHIPPING COMPANY S.A.",
            "Helvetic Blueway Transport AG",
        )
    ]
    occurrences = target_value_occurrence_requirements(raw, leaves, surface_requirements=surfaces)
    assert (
        next(
            row.requiredOccurrences
            for row in occurrences
            if row.targetValue == "Helvetic Blueway Transport AG"
        )
        == 2
    )


def test_carrier_receipt_equipment_breakdown_preserves_mixed_target_sizes() -> None:
    raw = "CARRIER'S RECEIPT\n12 X 40'\n"
    target = {
        "documentPatch": {
            "containers": [
                *[
                    {
                        "containerNumber": f"AAAA00000{index}",
                        "sizeCategory": "FORTY_FOOT",
                    }
                    for index in range(10)
                ],
                {
                    "containerNumber": "BBBB000001",
                    "sizeCategory": "TWENTY_FOOT",
                },
                {
                    "containerNumber": "BBBB000002",
                    "sizeCategory": "TWENTY_FOOT",
                },
            ]
        }
    }

    requirements = surface_rendering_requirements(raw, target, target)

    assert [(row.kind, row.targetSurface) for row in requirements] == [
        ("carrier_receipt_equipment_breakdown", "10 X 40' + 2 X 20'")
    ]


def test_reviewer_cannot_contradict_a_derived_carrier_receipt_surface() -> None:
    raw = "Total number of containers or packages 5 received by Carrier:\n"
    label = {
        "documentPatch": {"cargoPackages": [{"packageId": "p1", "groupId": "g1", "quantity": 5}]}
    }
    workspace = RewriteWorkspace(
        original_text=raw,
        current_text=raw,
        surface_requirements=surface_rendering_requirements(raw, label, label),
    )
    finding = SemanticReviewFinding(
        category="incomplete_auxiliary_anonymization",
        affectedPaths=("auxiliary.carrierReceipt",),
        currentEvidence=(raw.strip(),),
        correction="Change the count to a different value.",
    )

    assert _finding_conflicts_with_surface_authority(finding, workspace)


def test_reviewer_format_finding_must_quote_a_changed_surface() -> None:
    workspace = _workspace("Container Numbers, Seal Numbers and marks\n")
    finding = SemanticReviewFinding(
        category="format_or_layout_damage",
        affectedPaths=("rawOcr.heading",),
        currentEvidence=("Container Numbers, Seal Numbers and marks",),
        correction="Change the heading.",
    )

    assert _finding_grounds_format_damage_only_in_unchanged_text(finding, workspace)


def test_atomic_patch_rejects_wrong_date_or_hs_surface_before_commit() -> None:
    source = (
        "--- PAGE 1 ---\nSHIPPED ON BOARD DATE\n05/05/2024\n"
        "PLACE AND DATE OF ISSUE\n05/05/2024\nHS CODE: 2915.90.1800\n"
    )
    source_label = {
        "documentPatch": {
            "issueDate": "2024-05-05",
            "shippedOnBoardDate": "2024-05-05",
            "cargoGroups": [{"groupId": "g1", "hsCodes": ["2915901800"]}],
        }
    }
    target_label = {
        "documentPatch": {
            "issueDate": "2024-09-20",
            "shippedOnBoardDate": "2024-09-19",
            "cargoGroups": [{"groupId": "g1", "hsCodes": ["282619"]}],
        }
    }
    workspace = RewriteWorkspace(
        original_text=source,
        current_text=source,
        source_label=source_label,
        current_target_label=target_label,
        surface_requirements=surface_rendering_requirements(source, source_label, target_label),
    )

    with pytest.raises(ValueError, match="date/HS rendering requirements"):
        apply_line_range_replacements(
            workspace,
            (
                _edit(source, 3, 3, "09/19/2024"),
                _edit(source, 5, 5, "09/20/2024"),
                _edit(source, 6, 6, "HS CODE: 282.619"),
            ),
        )
    assert workspace.current_text == source


def test_flattened_equipment_count_is_not_a_marks_hint() -> None:
    raw = (
        "--- PAGE 1 ---\nContainer No. and Seal No.\nMarks & Nos.\n2\n\n"
        "Quantity and Kind of Packages\n\n/40' HC Containers Said to Contain\n"
    )

    hints = source_semantic_role_hints(raw)

    assert len(hints) == 1
    assert hints[0].role == "anonymous_equipment_count"
    assert hints[0].sourceLineId == "L00004"
    assert hints[0].sourceSurface == "2"
    assert hints[0].requiredOutputSurface == "2"
    assert hints[0].forbiddenTargetPathPrefixes == ("documentPatch.cargoGroups[].marksAndNumbers",)


def test_flattened_equipment_count_cannot_be_rewritten_as_invented_marks() -> None:
    raw = (
        "--- PAGE 1 ---\nContainer No. and Seal No.\nMarks & Nos.\n2\n\n"
        "Quantity and Kind of Packages\n\n/40' HC Containers Said to Contain\n"
    )
    workspace = RewriteWorkspace(
        original_text=raw,
        current_text=raw,
        source_role_hints=source_semantic_role_hints(raw),
    )

    with pytest.raises(ValueError, match="flattened-OCR role surface"):
        apply_line_range_replacements(
            workspace,
            (_edit(raw, 4, 4, "LOT RKM-27"),),
        )


def test_changed_free_text_is_required_but_schema_categories_are_not() -> None:
    leaves = (
        ChangedLeaf(
            path="documentPatch.cargoGroups[0].description",
            sourcePresent=True,
            targetPresent=True,
            sourceValue="OLD GOODS",
            targetValue="NEW STEEL GOODS",
            evidenceClass="printed_fact",
            requiresTextEdit=True,
        ),
        ChangedLeaf(
            path="documentPatch.cargoGroups[0].marksAndNumbers[0]",
            sourcePresent=True,
            targetPresent=True,
            sourceValue="N/M",
            targetValue="LOT: Q7M-4821",
            evidenceClass="printed_fact",
            requiresTextEdit=True,
        ),
        ChangedLeaf(
            path="documentPatch.cargoPackages[0].typeCategory",
            sourcePresent=True,
            targetPresent=True,
            sourceValue="PACKAGE_CARTON",
            targetValue="PACKAGE_BAG",
            evidenceClass="printed_fact",
            requiresTextEdit=True,
        ),
    )

    requirements = target_literal_requirements(leaves)

    assert [(row.targetPath, row.targetValue) for row in requirements] == [
        ("documentPatch.cargoGroups[0].description", "NEW STEEL GOODS"),
        ("documentPatch.cargoGroups[0].marksAndNumbers[0]", "LOT: Q7M-4821"),
    ]


def test_atomic_patch_rejects_missing_changed_target_literal() -> None:
    source = "--- PAGE 1 ---\nGOODS: OLD GOODS\nMARKS: N/M\n"
    leaves = (
        ChangedLeaf(
            path="documentPatch.cargoGroups[0].description",
            sourcePresent=True,
            targetPresent=True,
            sourceValue="OLD GOODS",
            targetValue="NEW STEEL GOODS",
            evidenceClass="printed_fact",
            requiresTextEdit=True,
        ),
        ChangedLeaf(
            path="documentPatch.cargoGroups[0].marksAndNumbers[0]",
            sourcePresent=True,
            targetPresent=True,
            sourceValue="N/M",
            targetValue="LOT: Q7M-4821",
            evidenceClass="printed_fact",
            requiresTextEdit=True,
        ),
    )
    workspace = RewriteWorkspace(
        original_text=source,
        current_text=source,
        target_literal_requirements=target_literal_requirements(leaves),
    )

    with pytest.raises(ValueError, match="omits changed target text"):
        apply_line_range_replacements(
            workspace,
            (
                _edit(source, 2, 2, "GOODS: NEW STEEL GOODS"),
                _edit(source, 3, 3, "MARKS: REMOVED"),
            ),
        )
    assert workspace.current_text == source


def test_atomic_patch_rejects_duplicated_unchanged_source_line() -> None:
    boilerplate = "THE SHIPMENTS ON MULTIPLE BILL OF LADING"
    source = f"--- PAGE 1 ---\n{boilerplate}\nB/L: OLD123\nGENERIC TEXT\n"

    with pytest.raises(ValueError, match="duplicates unchanged source lines"):
        apply_line_range_replacements(
            _workspace(source),
            (_edit(source, 3, 4, f"{boilerplate}\nB/L: NEW789"),),
        )


def test_equipment_review_cannot_reject_a_surface_that_resolves_to_target() -> None:
    workspace = RewriteWorkspace(
        original_text="--- PAGE 1 ---\nMSKU2159954 40' HIGH CUBE\n",
        current_text="--- PAGE 1 ---\nMSKU2159954 40' HIGH CUBE\n",
        current_target_label={
            "documentPatch": {
                "containers": [
                    {
                        "containerNumber": "MSKU2159954",
                        "sizeCategory": "FORTY_FOOT_HIGH_CUBE",
                        "typeCategory": "GENERAL_PURPOSE",
                    }
                ]
            }
        },
    )
    finding = SemanticReviewFinding(
        category="missing_or_wrong_target_fact",
        affectedPaths=("documentPatch.containers[0].printedEquipmentSurface",),
        currentEvidence=("MSKU2159954 40' HIGH CUBE",),
        correction="Add the literal words GENERAL PURPOSE.",
    )

    assert _finding_conflicts_with_equipment_authority(finding, workspace)


def test_target_integrity_repairs_reserved_domains_cctld_and_package_surface() -> None:
    config = load_synthesis_raw_text_rewrite_cycle_probe_config(
        Path("configs/synthesis/mpci_bl_raw_text_atomic10_luna_high.yaml")
    )
    target = {
        "schemaVersion": "5.0.0-experimental",
        "documentPatch": {
            "parties": {
                "carrier": {
                    "name": "Asterhaven Ocean Carriers Ltd.",
                    "contactDetails": {"websiteUrls": ["https://www.asterhaven.invalid"]},
                },
                "deliveryAgent": {
                    "name": "Mareb Corridor Delivery Services",
                    "country": "Eritrea",
                    "contactDetails": {"emailAddresses": ["dispatch@marebcorridor.et"]},
                },
            },
            "cargoGroups": [
                {
                    "groupId": "g1",
                    "description": "PRINTING PAPER IN ROLLS",
                    "additionalInformation": ["537 PACKAGES, GROSS WEIGHT 4,898.1 KGS"],
                }
            ],
            "cargoPackages": [
                {
                    "packageId": "p1",
                    "groupId": "g1",
                    "quantity": 537,
                    "typeCategory": "PACKAGE_PIECE",
                }
            ],
        },
    }

    effective, receipt = prepare_target_integrity(
        target, target, config, _target_integrity_resources()
    )

    assert effective["documentPatch"]["parties"]["carrier"]["contactDetails"]["websiteUrls"] == [
        "https://asterhavenoceancarriersltd.com"
    ]
    assert effective["documentPatch"]["parties"]["deliveryAgent"]["contactDetails"][
        "emailAddresses"
    ] == ["dispatch@marebcorridordeliveryservices.com"]
    assert effective["documentPatch"]["cargoGroups"][0]["additionalInformation"] == [
        "537 PIECES, GROSS WEIGHT 4,898.1 KGS"
    ]
    assert {row["reason"] for row in receipt["semantic_changes"]} == {
        "special_use_domain",
        "country_incoherent_cctld",
        "package_surface_category_mismatch",
    }


def test_rewrite_preflight_reprojects_an_overweight_semantic_target() -> None:
    config = load_synthesis_raw_text_rewrite_cycle_probe_config(
        Path("configs/synthesis/mpci_bl_raw_text_atomic10_luna_high.yaml")
    )
    target = {
        "schemaVersion": "5.0.0-experimental",
        "documentPatch": {
            "containers": [
                {
                    "containerNumber": "MSKU2159954",
                    "sizeCategory": "FORTY_FOOT_HIGH_CUBE",
                    "typeCategory": "GENERAL_PURPOSE",
                }
            ],
            "cargoGroups": [
                {
                    "groupId": "g1",
                    "grossWeight": {"value": 39434.1, "unit": "kilogram"},
                    "netWeight": {"value": 39418.2, "unit": "kilogram"},
                }
            ],
            "cargoPackages": [
                {
                    "packageId": "p1",
                    "groupId": "g1",
                    "quantity": 321,
                    "typeCategory": "PACKAGE_PACKAGE",
                }
            ],
        },
    }

    effective, receipt = prepare_target_integrity(
        target, target, config, _target_integrity_resources()
    )

    assert receipt["changed"] is True
    assert receipt["final_receipt"]["valid"] is True
    assert effective["documentPatch"]["cargoGroups"][0]["grossWeight"]["value"] < 39434.1


def test_operational_flavor_resamples_real_utilization_and_is_line_bound() -> None:
    config = load_synthesis_raw_text_rewrite_cycle_probe_config(
        Path("configs/synthesis/mpci_bl_raw_text_atomic10_luna_high.yaml")
    )
    limits = capacity_limits(config.target_integrity.transport_capacity)
    source = "--- PAGE 1 ---\nCAXU9173485 40' Dry Hi-Cube\nG.W 13519.000\nCBM 70.1530\n"
    source_label = {
        "documentPatch": {
            "containers": [
                {
                    "containerNumber": "CAXU9173485",
                    "typeDescription": "40' Dry Hi-Cube",
                }
            ],
            "cargoGroups": [{"groupId": "g1", "description": "SOURCE GOODS"}],
        }
    }
    target_label = {
        "documentPatch": {
            "containers": [
                {
                    "containerNumber": "MSKU2159954",
                    "sizeCategory": "TWENTY_FOOT_STANDARD_HEIGHT",
                    "typeCategory": "GENERAL_PURPOSE",
                }
            ],
            "cargoGroups": [{"groupId": "g1", "description": "TARGET GOODS"}],
        }
    }
    profile = EmpiricalOperationalProfile(
        document_id="profile-document",
        container_number="PROF0000000",
        equipment_family="twenty_standard",
        gross_weight_kg=Decimal("14857.5"),
        gross_utilization=Decimal("0.5"),
        volume_m3=Decimal("27.888"),
        volume_utilization=Decimal("0.8"),
    )

    requirements = operational_flavor_requirements(
        source,
        source_label,
        target_label,
        source_document_id="source-document",
        scenario_id="scenario-1",
        profiles=(profile,),
        limits=limits,
    )

    assert [(row.sourceLineId, row.targetValueSurface) for row in requirements] == [
        ("L00003", "14857.500"),
        ("L00004", "27.8880"),
    ]
    workspace = RewriteWorkspace(
        original_text=source,
        current_text=source,
        operational_flavor_requirements=requirements,
    )
    with pytest.raises(ValueError, match="raw-only operational measurement"):
        apply_line_range_replacements(
            workspace,
            (_edit(source, 2, 2, "MSKU2159954 20' DRY"),),
        )
    assert workspace.current_text == source
    apply_line_range_replacements(
        workspace,
        (
            _edit(source, 2, 2, "MSKU2159954 20' DRY"),
            _edit(source, 3, 3, "G.W 14857.500"),
            _edit(source, 4, 4, "CBM 27.8880"),
        ),
    )
    assert "G.W 14857.500\nCBM 27.8880" in workspace.current_text


def test_operational_flavor_projects_labeled_measure_truth_to_printed_rows() -> None:
    config = load_synthesis_raw_text_rewrite_cycle_probe_config(
        Path("configs/synthesis/mpci_bl_raw_text_atomic10_luna_high.yaml")
    )
    limits = capacity_limits(config.target_integrity.transport_capacity)
    source = "--- PAGE 1 ---\nCAXU9173485 40' Dry Hi-Cube\nG.W 13519.000\nCBM 70.1530\n"
    source_label = {
        "documentPatch": {
            "containers": [
                {
                    "containerNumber": "CAXU9173485",
                    "typeDescription": "40' Dry Hi-Cube",
                }
            ],
            "cargoGroups": [
                {
                    "groupId": "g1",
                    "description": "SOURCE GOODS",
                    "grossWeight": {"value": 13519, "unit": "kilogram"},
                    "volume": {"value": 70.153, "unit": "cubic_metre"},
                }
            ],
        }
    }
    target_label = {
        "documentPatch": {
            "containers": [
                {
                    "containerNumber": "MSKU2159954",
                    "sizeCategory": "FORTY_FOOT_HIGH_CUBE",
                    "typeCategory": "GENERAL_PURPOSE",
                }
            ],
            "cargoGroups": [
                {
                    "groupId": "g1",
                    "description": "TARGET GOODS",
                    "grossWeight": {"value": 14000, "unit": "kilogram"},
                    "volume": {"value": 71, "unit": "cubic_metre"},
                }
            ],
        }
    }
    profile = EmpiricalOperationalProfile(
        document_id="profile-document",
        container_number="PROF0000000",
        equipment_family="forty_high_cube",
        gross_weight_kg=Decimal("14000"),
        gross_utilization=Decimal("0.5"),
        volume_m3=Decimal("60"),
        volume_utilization=Decimal("0.75"),
    )

    requirements = operational_flavor_requirements(
        source,
        source_label,
        target_label,
        source_document_id="source-document",
        scenario_id="scenario-1",
        profiles=(profile,),
        limits=limits,
    )

    assert [(row.kind, row.targetValueSurface, row.samplingMethod) for row in requirements] == [
        ("gross_weight_kg", "14000.000", "target_measure_allocation_v1"),
        ("volume_m3", "71.0000", "target_measure_allocation_v1"),
    ]


def test_inline_container_breakdown_preserves_shifted_grammar_and_exact_totals() -> None:
    config = load_synthesis_raw_text_rewrite_cycle_probe_config(
        Path("configs/synthesis/mpci_bl_raw_text_atomic10_luna_high.yaml")
    )
    limits = capacity_limits(config.target_integrity.transport_capacity)
    source = (
        "--- PAGE 1 ---\n"
        "CAXU9173485/SOURCE1\n"
        "(100.000KG/10.000M3/1PK)/MSKU3245718/SOURCE2\n"
        "(200.000KG/20.000M3/2PK)/TGHU1042720/SOURCE3\n"
        "(300.000KG/30.000M3/3PK)/ACID NUMBER:\n"
    )
    source_label = {
        "documentPatch": {
            "containers": [
                {"containerNumber": "CAXU9173485", "typeDescription": "40 HC"},
                {"containerNumber": "MSKU3245718", "typeDescription": "40 HC"},
                {"containerNumber": "TGHU1042720", "typeDescription": "40 HC"},
            ],
            "cargoGroups": [
                {
                    "groupId": "g1",
                    "grossWeight": {"value": 600, "unit": "kilogram"},
                    "volume": {"value": 60, "unit": "cubic_metre"},
                }
            ],
        }
    }
    target_numbers = ("MEDU0412490", "MEDU8524671", "FCIU3300779")
    target_label = {
        "documentPatch": {
            "containers": [
                {
                    "containerNumber": number,
                    "sizeCategory": "FORTY_FOOT_HIGH_CUBE",
                    "typeCategory": "GENERAL_PURPOSE",
                }
                for number in target_numbers
            ],
            "cargoGroups": [
                {
                    "groupId": "g1",
                    "grossWeight": {"value": 1000, "unit": "kilogram"},
                    "volume": {"value": 50, "unit": "cubic_metre"},
                }
            ],
            "cargoPackages": [{"packageId": "p1", "groupId": "g1", "quantity": 10}],
            "cargoAllocationGroups": [
                {
                    "groupId": "g1",
                    "packageIds": ["p1"],
                    "coverage": "single_package_level",
                    "allocations": [
                        {"containerNumber": target_numbers[0], "packageQuantity": 2},
                        {"containerNumber": target_numbers[1], "packageQuantity": 3},
                        {"containerNumber": target_numbers[2], "packageQuantity": 5},
                    ],
                }
            ],
        }
    }
    requirements = operational_flavor_requirements(
        source,
        source_label,
        target_label,
        source_document_id="source-document",
        scenario_id="scenario-inline",
        profiles=(),
        limits=limits,
    )

    by_container = {
        row.targetContainerNumber: {
            candidate.kind: candidate.targetValueSurface
            for candidate in requirements
            if candidate.targetContainerNumber == row.targetContainerNumber
        }
        for row in requirements
    }
    assert by_container == {
        target_numbers[0]: {
            "gross_weight_kg": "200.000",
            "volume_m3": "10.000",
            "package_quantity": "2",
        },
        target_numbers[1]: {
            "gross_weight_kg": "300.000",
            "volume_m3": "15.000",
            "package_quantity": "3",
        },
        target_numbers[2]: {
            "gross_weight_kg": "500.000",
            "volume_m3": "25.000",
            "package_quantity": "5",
        },
    }
    assert {
        row.sameLineFollowingContainerNumber
        for row in requirements
        if row.targetContainerNumber == target_numbers[0]
    } == {target_numbers[1]}
    workspace = RewriteWorkspace(
        original_text=source,
        current_text=source,
        operational_flavor_requirements=requirements,
    )
    with pytest.raises(ValueError, match="raw-only operational measurement"):
        apply_line_range_replacements(
            workspace,
            (
                _edit(
                    source,
                    2,
                    5,
                    (
                        f"{target_numbers[0]}/TARGET1\n"
                        f"{target_numbers[1]}/TARGET2\n"
                        "(200.000KG/10.000M3/2PK)/ACID NUMBER:\n"
                        "(300.000KG/15.000M3/3PK)/ACID NUMBER:"
                    ),
                ),
            ),
        )
    expected = (
        f"{target_numbers[0]}/TARGET1\n"
        f"(200.000KG/10.000M3/2PK)/{target_numbers[1]}/TARGET2\n"
        f"(300.000KG/15.000M3/3PK)/{target_numbers[2]}/TARGET3\n"
        "(500.000KG/25.000M3/5PK)/ACID NUMBER:"
    )
    apply_line_range_replacements(workspace, (_edit(source, 2, 5, expected),))
    assert workspace.current_text.splitlines()[1:] == expected.splitlines()


def test_changed_cargo_span_includes_interleaved_quantities_and_trailing_product_row() -> None:
    source = (
        "--- PAGE 1 ---\n"
        "DESCRIPTION OF GOODS\n"
        "\n"
        "VCM 0.8*760*895 C.COVER PLATINUM SILVER\n"
        "4200 EA\n"
        "PCM 0.5*1544*936 BACK COVER DEEP BLUE\n"
        "3200 EA\n"
        "STS 430 0.4*388*C T/L FRAME 6747 EA\n"
        "\n"
        "HS CODE : 7208.54-9000\n"
    )
    source_label = {
        "documentPatch": {
            "cargoGroups": [
                {
                    "groupId": "g1",
                    "description": (
                        "VCM 0.8*760*895 C.COVER PLATINUM SILVER "
                        "PCM 0.5*1544*936 BACK COVER DEEP BLUE "
                        "STS 430 0.4*388*C T/L FRAME"
                    ),
                }
            ]
        }
    }
    target_description = "Ferro-vanadium ferro-alloy containing more than 4% carbon"
    target_label = {
        "documentPatch": {"cargoGroups": [{"groupId": "g1", "description": target_description}]}
    }
    requirements = cargo_flavor_rewrite_requirements(source, source_label, target_label)

    assert len(requirements) == 1
    assert requirements[0].sourceLineIds == (
        "L00004",
        "L00005",
        "L00006",
        "L00007",
        "L00008",
    )
    workspace = RewriteWorkspace(
        original_text=source,
        current_text=source,
        cargo_flavor_rewrite_requirements=requirements,
    )
    with pytest.raises(ValueError, match="source cargo-description line unchanged"):
        apply_line_range_replacements(
            workspace,
            (
                _edit(
                    source,
                    4,
                    7,
                    (
                        f"{target_description}\n"
                        "22 BAGS\n"
                        "FERRO-ALLOY LUMPS, INDUSTRIAL GRADE\n"
                        "VANADIUM CONTENT: 75% MINIMUM"
                    ),
                ),
            ),
        )
    assert workspace.current_text == source

    apply_line_range_replacements(
        workspace,
        (
            _edit(
                source,
                4,
                8,
                (
                    f"{target_description}\n"
                    "22 BAGS\n"
                    "FERRO-ALLOY LUMPS, INDUSTRIAL GRADE\n"
                    "VANADIUM CONTENT: 75% MINIMUM\n"
                    "PACKED IN LINED WOVEN BAGS"
                ),
            ),
        ),
    )
    assert deterministic_rewrite_audit(workspace, ()).cargoFlavorRewritten


def test_labeled_measurement_replacement_preserves_grouping_and_precision() -> None:
    source = "--- PAGE 1 ---\nGROSS WEIGHT\nKGS\n146,928.000\n\nMEASUREMENT\nCBM(M3)\n160.000\n"
    source_label = {
        "documentPatch": {
            "cargoGroups": [
                {
                    "groupId": "g1",
                    "grossWeight": {"value": 146928, "unit": "kilogram"},
                    "volume": {"value": 160, "unit": "cubic_metre"},
                }
            ]
        }
    }
    target_label = {
        "documentPatch": {
            "cargoGroups": [
                {
                    "groupId": "g1",
                    "grossWeight": {"value": 11956.1, "unit": "kilogram"},
                    "volume": {"value": 65.3, "unit": "cubic_metre"},
                }
            ]
        }
    }

    requirements = anchored_measurement_replacement_requirements(source, source_label, target_label)

    assert [(row.sourceLineIds, row.sourceSurface, row.targetSurface) for row in requirements] == [
        (("L00004",), "146,928.000", "11,956.100"),
        (("L00008",), "160.000", "65.300"),
    ]
    workspace = RewriteWorkspace(
        original_text=source,
        current_text=source,
        anchored_scalar_replacement_requirements=requirements,
    )
    with pytest.raises(ValueError, match="moves or omits an exact changed label scalar"):
        apply_line_range_replacements(
            workspace,
            (_edit(source, 4, 4, "11956.100"), _edit(source, 8, 8, "65.300")),
        )

    apply_line_range_replacements(
        workspace,
        (_edit(source, 4, 4, "11,956.100"), _edit(source, 8, 8, "65.300")),
    )


def test_deterministic_prefill_applies_every_repeated_anchored_value() -> None:
    source = (
        "--- PAGE 1 ---\nMEASUREMENT: 67.700CBM\nCONTAINER TOTAL: 67.700CBM\nREFERENCE: ABC-123\n"
    )
    requirements = (
        AnchoredScalarReplacementRequirement(
            targetPaths=("documentPatch.goods[0].volume.value",),
            sourceLineIds=(
                _line_id(2, "MEASUREMENT: 67.700CBM"),
                _line_id(3, "CONTAINER TOTAL: 67.700CBM"),
            ),
            sourceSurface="67.700",
            targetSurface="33.100",
            surfaceKind="measurement",
        ),
    )
    workspace = RewriteWorkspace(
        original_text=source,
        current_text=source,
        anchored_scalar_replacement_requirements=requirements,
    )

    applied = apply_deterministic_prefills(workspace)

    assert workspace.current_text == (
        "--- PAGE 1 ---\nMEASUREMENT: 33.100CBM\nCONTAINER TOTAL: 33.100CBM\nREFERENCE: ABC-123\n"
    )
    assert len(applied) == 2
    assert {row.lineId for row in applied} == set(requirements[0].sourceLineIds)
    assert all(row.targetPaths == requirements[0].targetPaths for row in applied)
    assert deterministic_rewrite_audit(workspace, ()).anchoredScalarReplacementsRendered


def test_deterministic_prefill_applies_unambiguous_derived_surfaces() -> None:
    source = (
        "--- PAGE 1 ---\nHS CODE: 8714.10.00\nON BOARD: 05/05/2025\n"
        "CARRIER RECEIPT\n12 X 40'\n"
    )
    requirements = (
        SurfaceRenderingRequirement(
            kind="hs_code",
            targetPath="documentPatch.cargoGroups[0].hsCodes[0]",
            sourceSurface="8714.10.00",
            targetSurface="7208.90.46",
            sourceOccurrences=1,
            contextEvidence="HS CODE: 8714.10.00",
        ),
        SurfaceRenderingRequirement(
            kind="date",
            targetPath="documentPatch.shippedOnBoardDate",
            sourceSurface="05/05/2025",
            targetSurface="19/09/2024",
            sourceOccurrences=1,
            contextEvidence="ON BOARD: 05/05/2025",
        ),
        SurfaceRenderingRequirement(
            kind="carrier_receipt_equipment_breakdown",
            targetPath="auxiliary.carrierReceiptEquipmentBreakdown",
            sourceSurface="12 X 40'",
            targetSurface="10 X 40' + 2 X 20'",
            sourceOccurrences=1,
            contextEvidence="CARRIER RECEIPT\n12 X 40'",
        ),
    )
    workspace = RewriteWorkspace(
        original_text=source,
        current_text=source,
        surface_requirements=requirements,
    )

    applied = apply_deterministic_prefills(workspace)

    assert workspace.current_text == (
        "--- PAGE 1 ---\nHS CODE: 7208.90.46\nON BOARD: 19/09/2024\n"
        "CARRIER RECEIPT\n10 X 40' + 2 X 20'\n"
    )
    assert {row.targetSurface for row in applied} == {
        "7208.90.46",
        "19/09/2024",
        "10 X 40' + 2 X 20'",
    }


def test_deterministic_prefill_leaves_conflicting_surface_projection_for_editor() -> None:
    source = "--- PAGE 1 ---\nREFERENCE DATE: 05/05/2025\n"
    first = SurfaceRenderingRequirement(
        kind="date",
        targetPath="documentPatch.issueDate",
        sourceSurface="05/05/2025",
        targetSurface="19/09/2024",
        sourceOccurrences=1,
        contextEvidence="REFERENCE DATE: 05/05/2025",
    )
    second = first.model_copy(
        update={
            "targetPath": "documentPatch.shippedOnBoardDate",
            "targetSurface": "20/09/2024",
        }
    )
    workspace = RewriteWorkspace(
        original_text=source,
        current_text=source,
        surface_requirements=(first, second),
    )

    applied = apply_deterministic_prefills(workspace)

    assert applied == ()
    assert workspace.current_text == source


def test_duplicate_anchored_scalar_authority_is_merged_before_prefill() -> None:
    shared = AnchoredScalarReplacementRequirement(
        targetPaths=("documentPatch.cargoGroups[0].volume.value",),
        sourceLineIds=("L00002", "L00003"),
        sourceSurface="25.000",
        targetSurface="33.100",
    )

    merged = merge_anchored_scalar_replacement_requirements((shared,), (shared,))

    assert merged == (shared,)


def test_conflicting_anchored_scalar_authority_is_rejected() -> None:
    first = AnchoredScalarReplacementRequirement(
        targetPaths=("documentPatch.cargoGroups[0].volume.value",),
        sourceLineIds=("L00002",),
        sourceSurface="25.000",
        targetSurface="33.100",
    )
    second = first.model_copy(update={"targetSurface": "34.200"})

    with pytest.raises(ValueError, match="conflicting deterministic scalar targets"):
        merge_anchored_scalar_replacement_requirements((first,), (second,))


def test_cargo_spans_assign_repeated_and_similar_descriptions_one_to_one() -> None:
    source = (
        "--- PAGE 1 ---\n"
        "Cheddar Cheese\n"
        "\n"
        "Cheddar Cheese\n"
        "\n"
        "FROZEN BONE IN LAMB LEG CHUMP OFF\n"
        "\n"
        "FROZEN BONE IN LAMB HINDSHANK\n"
    )
    source_label = {
        "documentPatch": {
            "cargoGroups": [
                {"groupId": "g1", "description": "Cheddar Cheese"},
                {"groupId": "g2", "description": "Cheddar Cheese"},
                {"groupId": "g3", "description": "FROZEN BONE IN LAMB LEG CHUMP OFF"},
                {"groupId": "g4", "description": "FROZEN BONE IN LAMB HINDSHANK"},
            ]
        }
    }
    target_label = {
        "documentPatch": {
            "cargoGroups": [
                {"groupId": "g1", "description": "Plastic film"},
                {"groupId": "g2", "description": "Natural polymers"},
                {"groupId": "g3", "description": "Packing machinery parts"},
                {"groupId": "g4", "description": "Mechanical seals"},
            ]
        }
    }

    requirements = cargo_flavor_rewrite_requirements(source, source_label, target_label)

    assert [row.sourceLineIds for row in requirements] == [
        ("L00002",),
        ("L00004",),
        ("L00006",),
        ("L00008",),
    ]


def test_empirical_operational_profiles_are_deduplicated_and_capacity_normalized() -> None:
    config = load_synthesis_raw_text_rewrite_cycle_probe_config(
        Path("configs/synthesis/mpci_bl_raw_text_atomic10_luna_high.yaml")
    )
    rows = (
        {
            "documentId": "doc-a",
            "joinedRawText": (
                "--- PAGE 1 ---\nCAXU9173485 40' Dry Hi-Cube\nG.W 13519.000\nCBM 70.1530\n"
            ),
            "target": {
                "documentPatch": {
                    "containers": [
                        {
                            "containerNumber": "CAXU9173485",
                            "typeDescription": "40' Dry Hi-Cube",
                        }
                    ]
                }
            },
        },
    )

    profiles = build_empirical_operational_profiles(
        rows,
        target_field="target",
        limits=capacity_limits(config.target_integrity.transport_capacity),
    )

    assert len(profiles) == 1
    assert profiles[0].equipment_family == "forty_high_cube"
    assert profiles[0].gross_weight_kg == Decimal("13519.000")
    assert profiles[0].volume_m3 == Decimal("70.1530")
    assert profiles[0].gross_utilization is not None
    assert profiles[0].volume_utilization is not None
    assert Decimal(0) < profiles[0].gross_utilization < Decimal(1)
    assert Decimal(0) < profiles[0].volume_utilization < Decimal(1)


def test_compound_party_flavor_requirement_comes_only_from_reviewed_source_label() -> None:
    source = {
        "documentPatch": {
            "parties": {
                "shipper": {"name": "SOURCE LTD ON BEHALF OF PRINCIPAL LLC"},
                "carrier": {"name": "PLAIN CARRIER"},
            }
        }
    }
    target = {
        "documentPatch": {
            "parties": {
                "shipper": {"name": "TARGET EXPORTS LTD"},
                "carrier": {"name": "TARGET CARRIER"},
            }
        }
    }

    requirements = compound_party_flavor_requirements(source, target)

    assert [row.targetPath for row in requirements] == ["documentPatch.parties.shipper.name"]
    assert requirements[0].relationships == ("on_behalf_of",)


def test_compound_party_flavor_preserves_target_and_updates_only_raw() -> None:
    source_label = {
        "schemaVersion": "3.0.0",
        "documentPatch": {
            "parties": {"shipper": {"name": "SOURCE LTD ON BEHALF OF PRINCIPAL LLC"}}
        },
    }
    target_label = {
        "schemaVersion": "5.0.0-experimental",
        "documentPatch": {"parties": {"shipper": {"name": "TARGET EXPORTS LTD"}}},
    }
    source = "--- PAGE 1 ---\nSHIPPER TARGET\n"
    workspace = RewriteWorkspace(
        original_text=source,
        current_text=source,
        scenario_id="synthetic-test",
        source_label=source_label,
        upstream_target_label=target_label,
        current_target_label=target_label,
    )

    commit = apply_line_range_replacements(
        workspace,
        (_edit(source, 2, 2, "TARGET EXPORTS LTD ON BEHALF OF HARBOR PRINCIPAL LLC"),),
        compound_party_flavor_realizations=(
            CompoundPartyFlavorRealization(
                targetPath="documentPatch.parties.shipper.name",
                targetPrimaryName="TARGET EXPORTS LTD",
                renderedName="TARGET EXPORTS LTD ON BEHALF OF HARBOR PRINCIPAL LLC",
            ),
        ),
    )

    assert (
        workspace.current_target_label["documentPatch"]["parties"]["shipper"]["name"]
        == "TARGET EXPORTS LTD"
    )
    assert commit.beforeTargetLabelSha256 == commit.afterTargetLabelSha256
    assert len(commit.compoundPartyFlavorRealizations) == 1

    corrected_source = workspace.current_text
    correction = apply_line_range_replacements(
        workspace,
        (_edit(corrected_source, 2, 2, "TARGET EXPORTS LTD ON BEHALF OF SEAWARD PRINCIPAL LLC"),),
        compound_party_flavor_realizations=(
            CompoundPartyFlavorRealization(
                targetPath="documentPatch.parties.shipper.name",
                targetPrimaryName="TARGET EXPORTS LTD",
                renderedName="TARGET EXPORTS LTD ON BEHALF OF SEAWARD PRINCIPAL LLC",
            ),
        ),
    )
    assert len(correction.compoundPartyFlavorRealizations) == 1
    assert "SEAWARD PRINCIPAL LLC" in workspace.current_text


def test_compound_party_flavor_is_required_and_rejects_missing_realization() -> None:
    source_label = {
        "schemaVersion": "3.0.0",
        "documentPatch": {
            "parties": {"carrier": {"name": "SOURCE CARRIER trading as SOURCE LINE"}}
        },
    }
    target_label = {
        "schemaVersion": "5.0.0-experimental",
        "documentPatch": {"parties": {"carrier": {"name": "TARGET CARRIER LTD"}}},
    }
    source = "--- PAGE 1 ---\nCARRIER: SOURCE CARRIER trading as SOURCE LINE\n"
    workspace = RewriteWorkspace(
        original_text=source,
        current_text=source,
        scenario_id="synthetic-test",
        source_label=source_label,
        upstream_target_label=target_label,
        current_target_label=target_label,
    )

    with pytest.raises(ValueError, match="differ from requirements"):
        apply_line_range_replacements(
            workspace,
            (_edit(source, 2, 2, "CARRIER: TARGET CARRIER LTD"),),
        )

    assert workspace.current_text == source
    assert workspace.current_target_label == target_label


def test_raw_only_carrier_agent_is_synthesized_in_its_existing_relationship() -> None:
    source_label = {"documentPatch": {"parties": {"carrier": {"name": "OCEANIC STAR LINE"}}}}
    target_label = {"documentPatch": {"parties": {"carrier": {"name": "HARBORCREST MARITIME"}}}}
    source = "--- PAGE 1 ---\nSZL MARINENGB LIMITED\n\nAs agent for the carrier OCEANIC STAR LINE\n"
    requirements = raw_auxiliary_identity_requirements(source, source_label, target_label)
    assert len(requirements) == 1
    assert requirements[0].sourceIdentity == "SZL MARINENGB LIMITED"
    workspace = RewriteWorkspace(
        original_text=source,
        current_text=source,
        current_target_label=target_label,
        raw_auxiliary_identity_requirements=requirements,
    )

    commit = apply_line_range_replacements(
        workspace,
        (
            _edit(
                source,
                2,
                4,
                "BLUEWATER AGENCY LIMITED\n\n"
                "As agent for the carrier HARBORCREST MARITIME trading as HARBOR LINE",
            ),
        ),
    )

    assert len(commit.rawAuxiliaryIdentityRealizations) == 1
    assert deterministic_rewrite_audit(workspace, ()).rawAuxiliaryIdentitiesReplaced


def test_repeated_inline_and_split_carrier_agents_share_one_fictional_identity() -> None:
    source_label = {
        "documentPatch": {"parties": {"carrier": {"name": "MEDITERRANEAN SHIPPING COMPANY S.A."}}}
    }
    target_label = {"documentPatch": {"parties": {"carrier": {"name": "ASTERHAVEN OCEAN LTD."}}}}
    source = (
        "--- PAGE 1 ---\n"
        "Mediterranean Shipping Company (Canada) Inc., as agents for the carrier "
        "MEDITERRANEAN SHIPPING COMPANY S.A.\n"
        "--- PAGE 2 ---\n"
        "Mediterranean Shipping Company (Canada) Inc., as agents for the carrier\n"
        "MEDITERRANEAN SHIPPING COMPANY S.A.\n"
    )
    requirements = raw_auxiliary_identity_requirements(source, source_label, target_label)

    assert len(requirements) == 2
    assert len({row.consistencyGroupId for row in requirements}) == 1
    assert {row.requirementId for row in requirements} == {
        "carrier-agent-L00002",
        "carrier-agent-L00004",
    }
    workspace = RewriteWorkspace(
        original_text=source,
        current_text=source,
        current_target_label=target_label,
        raw_auxiliary_identity_requirements=requirements,
    )
    rendered_agent = "Southern Cross Maritime Agency Ltd."
    apply_line_range_replacements(
        workspace,
        (
            _edit(
                source,
                2,
                2,
                f"{rendered_agent}, as agents for the carrier ASTERHAVEN OCEAN LTD.",
            ),
            _edit(
                source,
                4,
                5,
                f"{rendered_agent}, as agents for the carrier\nASTERHAVEN OCEAN LTD.",
            ),
        ),
    )

    assert deterministic_rewrite_audit(workspace, ()).rawAuxiliaryIdentitiesReplaced


def test_raw_only_carrier_agent_cannot_be_left_stale_or_unreported() -> None:
    source_label = {"documentPatch": {"parties": {"carrier": {"name": "OCEANIC STAR LINE"}}}}
    target_label = {"documentPatch": {"parties": {"carrier": {"name": "HARBORCREST MARITIME"}}}}
    source = "--- PAGE 1 ---\nSZL MARINENGB LIMITED\n\nAs agent for the carrier OCEANIC STAR LINE\n"
    workspace = RewriteWorkspace(
        original_text=source,
        current_text=source,
        current_target_label=target_label,
        raw_auxiliary_identity_requirements=raw_auxiliary_identity_requirements(
            source, source_label, target_label
        ),
    )

    with pytest.raises(ValueError, match="raw auxiliary identity must be distinct"):
        apply_line_range_replacements(
            workspace,
            (
                _edit(
                    source,
                    4,
                    4,
                    "As agent for the carrier HARBORCREST MARITIME",
                ),
            ),
        )


def test_raw_only_carrier_agent_correction_replaces_its_audit_realization() -> None:
    source_label = {"documentPatch": {"parties": {"carrier": {"name": "OCEANIC STAR LINE"}}}}
    target_label = {"documentPatch": {"parties": {"carrier": {"name": "HARBORCREST MARITIME"}}}}
    source = "--- PAGE 1 ---\nSZL MARINENGB LIMITED\n\nAs agent for the carrier OCEANIC STAR LINE\n"
    requirements = raw_auxiliary_identity_requirements(source, source_label, target_label)
    workspace = RewriteWorkspace(
        original_text=source,
        current_text=source,
        current_target_label=target_label,
        raw_auxiliary_identity_requirements=requirements,
    )
    apply_line_range_replacements(
        workspace,
        (
            _edit(
                source,
                2,
                4,
                "BLUEWATER AGENCY LIMITED\n\nAs agent for the carrier HARBORCREST MARITIME",
            ),
        ),
    )
    current = workspace.current_text

    corrected = apply_line_range_replacements(
        workspace,
        (_edit(current, 2, 2, "SEAWARD AGENCY LIMITED"),),
    )

    assert corrected.rawAuxiliaryIdentityRealizations[0].renderedIdentity == (
        "SEAWARD AGENCY LIMITED"
    )
    assert deterministic_rewrite_audit(workspace, ()).rawAuxiliaryIdentitiesReplaced


def test_two_line_raw_agent_identity_is_derived_from_the_patch() -> None:
    source_label = {"documentPatch": {"parties": {"carrier": {"name": "YANG MING"}}}}
    target_label = {"documentPatch": {"parties": {"carrier": {"name": "BLUE MERIDIAN"}}}}
    source = (
        "--- PAGE 1 ---\n"
        "Yang Ming Line\n"
        "By (Thailand) Co., LTD.\n"
        "As agent for the Carrier and Service Provider YANG MING\n"
    )
    requirements = raw_auxiliary_identity_requirements(source, source_label, target_label)
    workspace = RewriteWorkspace(
        original_text=source,
        current_text=source,
        current_target_label=target_label,
        raw_auxiliary_identity_requirements=requirements,
    )

    commit = apply_line_range_replacements(
        workspace,
        (
            _edit(
                source,
                2,
                4,
                "Pearl Delta Harbor\n"
                "By Logistics (China) Co., LTD.\n"
                "As agent for the Carrier and Service Provider BLUE MERIDIAN",
            ),
        ),
    )

    assert commit.rawAuxiliaryIdentityRealizations[0].renderedIdentity == (
        "Pearl Delta Harbor\nBy Logistics (China) Co., LTD."
    )
    assert deterministic_rewrite_audit(workspace, ()).rawAuxiliaryIdentitiesReplaced
