from decimal import Decimal
from pathlib import Path

import pytest
from openai import AsyncOpenAI
from pydantic_ai.exceptions import ModelAPIError, ModelHTTPError, UnexpectedModelBehavior

from document_ocr.synthesis.config import load_synthesis_raw_text_hybrid_probe_config
from document_ocr.synthesis.raw_text_hybrid_batch import _decode_checkpoint_models
from document_ocr.synthesis.raw_text_hybrid_probe import (
    HybridCaseResult,
    HybridModelStage,
    HybridWorkItem,
    ResidualSpan,
    _apply_compiler_requirements,
    _apply_native_residual_output,
    _apply_party_role_country_prefills,
    _baseline_reference_gap_lines,
    _cargo_requirement_line_numbers,
    _context_bound_surface_lines,
    _contextual_location_line_numbers,
    _empty_usage,
    _expanded_intervals,
    _line_set_for_directive,
    _literal_classification,
    _literal_line_numbers,
    _model_pair,
    _native_residual_output_type,
    _numeric_line_numbers,
    _numeric_surface_values,
    _party_block_line_numbers,
    _party_heading_roles,
    _provider_attempt_routes,
    _ResidualCommitContext,
    _retryable_route_error,
    _trim_context_blank_edges,
)
from document_ocr.synthesis.raw_text_rewrite_cycle_probe import (
    CargoFlavorRewriteRequirement,
    JurisdictionalSurfaceRequirement,
    RewriteWorkspace,
    SurfaceRenderingRequirement,
    apply_deterministic_prefills,
)
from document_ocr.synthesis.raw_text_rewrite_cycle_probe import _settings as rewrite_settings
from document_ocr.synthesis.raw_text_rewrite_probe import ChangedLeaf


def _leaf(source: str, target: str) -> ChangedLeaf:
    return ChangedLeaf(
        path="documentPatch.transport.vesselName",
        sourcePresent=True,
        targetPresent=True,
        sourceValue=source,
        targetValue=target,
        evidenceClass="printed_fact",
        requiresTextEdit=True,
    )


def test_hybrid_work_item_accepts_combined_rendered_party_evidence_locator() -> None:
    item = HybridWorkItem(
        workItemId="W0001",
        targetPaths=("documentPatch.parties.carrier.name",),
        action="replace",
        sourceValue="ARKAS LINE",
        targetValue="MAREVANTA OCEAN LINES, S.A.",
        state="agent_residual",
        evidenceLineIds=("L00002", "L00091"),
        spanIds=("S001",),
        locator="rendered_surface_and_party_role_block",
        rationale="The printed logo and labeled party block are jointly owned.",
    )

    assert item.locator == "rendered_surface_and_party_role_block"


def test_party_role_country_prefill_owns_the_party_block_not_detached_metadata() -> None:
    text = (
        "--- PAGE 1 ---\n"
        "Shipper\n"
        "INNOLUX CORPORATION\n"
        "NO.160 KESYUE RD., TAIWAN\n"
        "TAIWAN\n"
        "\n"
        "EXPORTER REGISTRATION COUNTRY: TAIWAN\n"
    )
    source_label = {
        "documentPatch": {
            "parties": {
                "shipper": {
                    "name": "INNOLUX CORPORATION",
                    "address": "NO.160 KESYUE RD.",
                    "country": "TAIWAN",
                }
            }
        }
    }
    target_label = {
        "documentPatch": {
            "parties": {
                "shipper": {
                    "name": "AURES MERIDIAN SARL",
                    "address": "Lotissement El Waha",
                    "country": "Algeria",
                }
            }
        }
    }
    workspace = RewriteWorkspace(
        original_text=text,
        current_text=text,
        source_label=source_label,
        current_target_label=target_label,
    )

    applied = _apply_party_role_country_prefills(workspace)

    expected = text.replace("NO.160 KESYUE RD., TAIWAN", "NO.160 KESYUE RD., ALGERIA")
    expected = expected.replace("TAIWAN\n\n", "ALGERIA\n\n", 1)
    assert workspace.current_text == expected
    assert [row.lineId for row in applied] == ["L00004", "L00005"]
    assert {row.targetPaths for row in applied} == {
        ("documentPatch.parties.shipper.country",)
    }
    assert {row.targetSurface for row in applied} == {"ALGERIA"}


def test_compiler_keeps_multi_selector_customs_clause_for_one_residual_rewrite() -> None:
    source = (
        "--- PAGE 1 ---\n"
        "ACID#: 1002845232025040024\n"
        "MERCHANT MUST PROVIDE ACID NUMBER AND ACID# BEFORE LOADING\n"
    )
    common = {
        "programId": "egypt_acid",
        "tradeDirection": "import",
        "programJurisdictionCountryCode": "EG",
        "targetRouteCountryCode": "US",
        "targetSurface": "IMPORT CUSTOMS REFERENCE",
        "authority": "Egyptian Customs Authority",
        "officialSourceUrl": "https://www.nafeza.gov.eg/",
    }
    requirements = (
        JurisdictionalSurfaceRequirement(
            requirementId="jurisdiction-acid-hash",
            sourceLineIds=("L00002", "L00003"),
            sourceSurface="ACID#",
            sourceOccurrences=2,
            **common,
        ),
        JurisdictionalSurfaceRequirement(
            requirementId="jurisdiction-acid-number",
            sourceLineIds=("L00003",),
            sourceSurface="ACID NUMBER",
            sourceOccurrences=1,
            **common,
        ),
    )
    workspace = RewriteWorkspace(
        original_text=source,
        current_text=source,
        jurisdictional_requirements=requirements,
    )

    edits = _apply_compiler_requirements(workspace)

    assert workspace.current_text == source
    assert edits == ()


def test_compiler_preserves_customs_heading_delimiter_and_value_spacing() -> None:
    source = "--- PAGE 1 ---\nACID : 1002845232025040024\n"
    requirement = JurisdictionalSurfaceRequirement(
        requirementId="jurisdiction-acid-colon",
        programId="egypt_acid",
        tradeDirection="import",
        programJurisdictionCountryCode="EG",
        targetRouteCountryCode="US",
        sourceLineIds=("L00002",),
        sourceSurface="ACID:",
        targetSurface="CUSTOMS REFERENCE",
        sourceOccurrences=1,
        authority="Egyptian Customs Authority",
        officialSourceUrl="https://www.nafeza.gov.eg/",
    )
    workspace = RewriteWorkspace(
        original_text=source,
        current_text=source,
        jurisdictional_requirements=(requirement,),
    )

    edits = _apply_compiler_requirements(workspace)

    assert workspace.current_text == "--- PAGE 1 ---\nCUSTOMS REFERENCE : 1002845232025040024\n"
    assert len(edits) == 1
    assert edits[0].targetSurface == "CUSTOMS REFERENCE :"


def test_hybrid_probe_config_is_strict_and_bounded() -> None:
    config = load_synthesis_raw_text_hybrid_probe_config(
        Path("configs/synthesis/mpci_bl_raw_text_hybrid_compiler50_glm53_flash_poc1.yaml")
    )

    assert config.workflow.audit_documents == 50
    assert config.workflow.max_model_documents == 1
    assert len(config.cases) == 1
    assert config.providers.editor.model == "z-ai/glm-5.3-flash"
    assert config.providers.editor.reasoning_effort == "minimal"
    assert config.providers.editor.allow_fallbacks is True
    assert config.workflow.provider_native_json_schema is True
    assert config.workflow.publish_training_records is False


def test_openrouter_provider_order_is_strict_and_preserved_in_settings() -> None:
    config = load_synthesis_raw_text_hybrid_probe_config(
        Path("configs/synthesis/mpci_bl_raw_text_hybrid_compiler50_glm53_flash_poc1.yaml")
    )

    assert config.providers.editor.provider_order == (
        "deepinfra/fp8",
        "wafer",
        "morph/fp8",
        "nextbit/fp8",
        "fireworks",
    )
    assert config.providers.editor.provider_only == config.providers.editor.provider_order
    settings = rewrite_settings(
        config.providers.editor,
        stage="editor",
        prompt_sha256=config.prompts.editor.sha256,
    )
    routing = settings["openrouter_provider"]
    assert routing["only"] == list(config.providers.editor.provider_order)
    assert routing["order"] == list(config.providers.editor.provider_order)
    assert "sort" not in routing


def test_openrouter_application_fallback_pins_each_route_and_records_retryable_errors() -> None:
    config = load_synthesis_raw_text_hybrid_probe_config(
        Path("configs/synthesis/mpci_bl_raw_text_hybrid_compiler50_glm53_flash_poc1.yaml")
    )
    provider = config.providers.editor

    routes = _provider_attempt_routes(provider)
    assert routes == provider.provider_order
    settings = rewrite_settings(
        provider,
        stage="editor",
        prompt_sha256=config.prompts.editor.sha256,
        route_provider=routes[1],
    )
    assert settings["openrouter_provider"] == {
        "require_parameters": True,
        "data_collection": "deny",
        "allow_fallbacks": False,
        "only": [routes[1]],
        "order": [routes[1]],
        "max_price": {"prompt": 0.15, "completion": 0.5},
    }
    assert _retryable_route_error(ModelHTTPError(429, provider.model, {"code": "overloaded"}))
    assert _retryable_route_error(ModelHTTPError(503, provider.model, None))
    assert _retryable_route_error(ModelAPIError(provider.model, "Connection error."))
    assert _retryable_route_error(
        UnexpectedModelBehavior("Model token limit exceeded before any response was generated.")
    )
    assert _retryable_route_error(
        ModelHTTPError(
            404,
            provider.model,
            {"metadata": {"failed_routing_step": "Filter by Tool Compatibility"}},
        )
    )
    assert not _retryable_route_error(ModelHTTPError(400, provider.model, None))
    assert not _retryable_route_error(ModelHTTPError(404, provider.model, None))


def test_checkpoint_models_round_trip_through_strict_json_boundary() -> None:
    usage = _empty_usage()
    result = HybridCaseResult(
        documentId="doc-checkpoint",
        status="needs_review",
        reason="checkpoint round trip",
        sourceLines=2,
        residualLines=1,
        residualSpans=1,
        workItems=1,
        deterministicWorkItems=0,
        residualWorkItems=1,
        blockedWorkItems=0,
        deterministicPrefills=0,
        compilerEdits=0,
        outputTextSha256="0" * 64,
        baselineOutputTextSha256=None,
        exactBaselineMatch=None,
        baselineReferenceGapLines=(1, 2),
        deterministicAudit=None,
        review=None,
        usage=usage,
    )
    stage = HybridModelStage(
        stage="residual_editor",
        providerModel="test-model",
        routeProvider=None,
        reasoningEffort="low",
        inputPayloadSha256="1" * 64,
        startedAtUnixSeconds=1.0,
        completedAtUnixSeconds=2.0,
        usage=usage,
        messages={},
        errorType=None,
        errorMessage=None,
    )

    restored_result, restored_stages = _decode_checkpoint_models(
        result.model_dump(mode="json"),
        [stage.model_dump(mode="json")],
    )

    assert restored_result == result
    assert restored_result.baselineReferenceGapLines == (1, 2)
    assert restored_result.usage.estimatedCostUsd == Decimal(0)
    assert restored_stages == (stage,)


def test_literal_discovery_distinguishes_conflicting_repeated_values() -> None:
    leaf = _leaf("BARCELONA", "SURABAYA")
    targets = {"barcelona": frozenset({"surabaya", "depok"})}

    assert (
        _literal_classification("BARCELONA\nBARCELONA\n", leaf, targets)
        == "exact_multiple_conflicting"
    )


def test_literal_locator_handles_multiline_whitespace_without_fuzzy_matching() -> None:
    lines, locator = _literal_line_numbers(
        "HEADER\nAE - AFZ BUILDING C1,\nOFFICE 908 H\nFOOTER\n",
        "AE - AFZ BUILDING C1, OFFICE 908 H",
    )

    assert locator == "flexible_whitespace_literal"
    assert lines == {2, 3}


def test_literal_locator_handles_ocr_line_break_inside_numeric_token() -> None:
    lines, locator = _literal_line_numbers(
        "HEADER\n80 DRUMS ON WOODEN PALLETS X 250.\n0000 KGS\nFOOTER\n",
        "ON WOODEN PALLETS X 250.0000 KGS",
    )

    assert locator == "flexible_whitespace_literal"
    assert lines == {2, 3}


def test_literal_locator_does_not_ignore_non_whitespace_differences() -> None:
    lines, locator = _literal_line_numbers(
        "HEADER\n80 DRUMS ON WOODEN PALLETS X 250-\n0000 KGS\nFOOTER\n",
        "ON WOODEN PALLETS X 250.0000 KGS",
    )

    assert locator == "unlocated"
    assert lines == set()


def test_numeric_locator_preserves_both_decimal_and_grouping_interpretations() -> None:
    text = "GROSS WEIGHT 20931.200KGS\nOTHER WEIGHT 3.741 KG\n"

    assert _numeric_surface_values("20931.200") == {Decimal("20931.200"), Decimal("20931200")}
    assert _numeric_line_numbers(text, 20931.2) == {1}
    assert _numeric_line_numbers(text, 3741) == {2}


def test_numeric_locator_preserves_signed_temperature_and_does_not_break_ranges() -> None:
    text = "SET TEMPERATURE -3 C\nPALLETS 20-21\n"

    assert _numeric_surface_values("-3") == {Decimal("-3")}
    assert _numeric_line_numbers(text, -3) == {1}
    assert _numeric_line_numbers(text, 21) == {2}


def test_numeric_locator_does_not_authorize_page_markers() -> None:
    assert _numeric_line_numbers("--- PAGE 3 ---\nQUANTITY 3\n", 3) == {2}


def test_party_block_locator_owns_split_address_and_separates_repeated_identity_roles() -> None:
    text = (
        "--- PAGE 1 ---\n"
        "CONSIGNEE\n"
        "AL SAKR INDUSTRIES\n"
        "3rd Industrial Area\n"
        "Borg El Arab\n"
        "Alexandria Egypt\n"
        "\n"
        "NOTIFY PARTY\n"
        "AL SAKR INDUSTRIES\n"
        "3rd Industrial Area\n"
        "Borg El Arab\n"
        "Alexandria Egypt\n"
    )
    source = {
        "documentPatch": {
            "parties": {
                "consignee": {
                    "name": "AL SAKR INDUSTRIES",
                    "address": "3rd Industrial Area, Borg El Arab",
                    "city": "Alexandria",
                    "country": "Egypt",
                },
                "notifyParties": [
                    {
                        "name": "AL SAKR INDUSTRIES",
                        "address": "3rd Industrial Area, Borg El Arab",
                        "city": "Alexandria",
                        "country": "Egypt",
                    }
                ],
            }
        }
    }

    assert _party_block_line_numbers(
        text,
        path="documentPatch.parties.consignee.address",
        source_label=source,
    ) == {3, 4, 5, 6}
    assert _party_block_line_numbers(
        text,
        path="documentPatch.parties.notifyParties[0].address",
        source_label=source,
    ) == {9, 10, 11, 12}


def test_party_block_locator_crosses_blank_after_heading_without_conflating_roles() -> None:
    text = (
        "TO OBTAIN DELIVERY CONTACT\n"
        "UNRELATED AGENT\n"
        "CAIRO, EGYPT\n"
        "\n"
        "CONSIGNEE\n"
        "\n"
        "SHARED INDUSTRIES SAE\n"
        "TENTH OF RAMADAN CITY, EGYPT\n"
        "\n"
        "NOTIFY\n"
        "\n"
        "SHARED INDUSTRIES SAE\n"
        "TENTH OF RAMADAN CITY, EGYPT\n"
    )
    source = {
        "documentPatch": {
            "parties": {
                "consignee": {
                    "name": "SHARED INDUSTRIES SAE",
                    "city": "TENTH OF RAMADAN CITY",
                    "country": "EGYPT",
                },
                "notifyParties": [
                    {
                        "name": "SHARED INDUSTRIES SAE",
                        "city": "TENTH OF RAMADAN CITY",
                        "country": "EGYPT",
                    }
                ],
            }
        }
    }

    assert _party_block_line_numbers(
        text,
        path="documentPatch.parties.consignee.name",
        source_label=source,
    ) == {7, 8}
    assert _party_block_line_numbers(
        text,
        path="documentPatch.parties.notifyParties[0].name",
        source_label=source,
    ) == {12, 13}


def test_party_block_locator_prefers_complete_identity_over_signature_copy() -> None:
    text = "CARRIER\nBLUE LINE LTD\n1 HARBOUR ROAD\nSINGAPORE\n\nSigned for BLUE LINE LTD\n"
    source = {
        "documentPatch": {
            "parties": {
                "carrier": {
                    "name": "BLUE LINE LTD",
                    "address": "1 HARBOUR ROAD",
                    "country": "SINGAPORE",
                }
            }
        }
    }

    assert _party_block_line_numbers(
        text,
        path="documentPatch.parties.carrier.name",
        source_label=source,
    ) == {2, 3, 4}


def test_party_block_locator_matches_ampersand_and_compact_address_punctuation() -> None:
    text = (
        "TURKON LINE\n"
        "TURKON CONTAINER TRANSPORTATION&SHIPPING INC.\n"
        "EXTRA STREET DETAIL\n"
        "NO. 33 ALTUNIZADE / ISTANBUL / TURKEY\n"
        "\n"
        "Signed for Turkon Container Transportation and Shipping Inc. "
        "No.33 ALTUNIZADE / ISTANBUL / TURKEY\n"
    )
    source = {
        "documentPatch": {
            "parties": {
                "carrier": {
                    "name": "Turkon Container Transportation and Shipping Inc.",
                    "address": "NO.33 ALTUNIZADE",
                    "city": "ISTANBUL",
                    "country": "TURKEY",
                }
            }
        }
    }

    assert _party_block_line_numbers(
        text,
        path="documentPatch.parties.carrier.address",
        source_label=source,
    ) == {1, 2, 3, 4, 6}


@pytest.mark.parametrize(
    ("line", "roles"),
    [
        ("FORWARDING AGENT AT&L CANADA INC", frozenset({"forwardingAgent"})),
        ("AGENT ADDRESS:", frozenset({"deliveryAgent"})),
        ("CARRIER: CMA CGM Société Anonyme", frozenset({"carrier"})),
        ("SIGNED on behalf of the Carrier MSC S.A.", frozenset()),
        ("CARRIER'S RECEIPT", frozenset()),
        ("PARTICULARS FURNISHED BY SHIPPER - CARRIER NOT RESPONSIBLE", frozenset()),
    ],
)
def test_party_heading_roles_are_structural(line: str, roles: frozenset[str]) -> None:
    assert _party_heading_roles(line) == roles


def test_location_locator_separates_same_surface_by_semantic_heading() -> None:
    text = (
        "PLACE OF RECEIPT\nNINGBO\n\n"
        "PORT OF LOADING\nNINGBO\n\n"
        "PLACE AND DATE OF ISSUE\nNINGBO 05-MAY-2024\n"
    )

    assert _contextual_location_line_numbers(
        text, "documentPatch.route.placeOfReceipt.name", "NINGBO"
    ) == {2}
    assert _contextual_location_line_numbers(
        text, "documentPatch.route.portOfLoading.name", "NINGBO"
    ) == {5}
    assert _contextual_location_line_numbers(text, "documentPatch.placeOfIssue.name", "NINGBO") == {
        8
    }


def test_location_locator_fails_closed_for_unheaded_repeated_surface() -> None:
    assert (
        _contextual_location_line_numbers(
            "NINGBO\nOTHER\nNINGBO\n",
            "documentPatch.route.portOfLoading.name",
            "NINGBO",
        )
        == set()
    )


def test_cargo_requirement_owns_every_exact_repeated_source_rendering() -> None:
    text = (
        "BOBA PEARL\n"
        "BOBA PEARL 1KG X 18 BAGS\n"
        "--- PAGE 2 ---\n"
        "BOBA PEARL\n"
        "BOBA PEARL 1KG X 18 BAGS INVOICE 42\n"
    )
    requirement = CargoFlavorRewriteRequirement(
        requirementId="cargo-group-1-span-1",
        targetPath="documentPatch.cargoGroups[0].description",
        sourceLineIds=("L00001", "L00002"),
        sourceSurfaces=("BOBA PEARL", "BOBA PEARL 1KG X 18 BAGS"),
        targetDescription="PAPER IN ROLLS",
    )

    assert _cargo_requirement_line_numbers(text, requirement) == {1, 2, 4, 5}


def test_freight_arrangement_locator_uses_authoritative_declaration_not_table_header() -> None:
    text = (
        "Rate Unit Currency Prepaid Collect\n"
        "FREIGHT: PREPAID\n"
        "Terms mention freight and charges collected later.\n"
    )

    assert _line_set_for_directive(
        text,
        "documentPatch.freight.paymentArrangement",
        "prepaid",
    ) == ({2}, "freight_arrangement_surface")


def test_freight_arrangement_locator_understands_payable_at_destination() -> None:
    assert _line_set_for_directive(
        "Freight payable at destination\n",
        "documentPatch.freight.paymentArrangement",
        "collect",
    ) == ({1}, "freight_arrangement_surface")


def test_numeric_locator_never_treats_page_marker_as_business_evidence() -> None:
    assert _line_set_for_directive(
        "--- PAGE 1 ---\n1 PALLET\n",
        "documentPatch.cargoPackages[0].quantity",
        1,
    ) == ({2}, "numeric_surface")


def test_carrier_locator_accepts_exact_name_wrapped_by_signed_by_form_labels() -> None:
    text = (
        "SIGNED ORIENT OVERSEAS CONTAINER LINE\n"
        "BY: (CHINA) CO., LTD\n"
    )

    assert _line_set_for_directive(
        text,
        "documentPatch.parties.carrier.name",
        "ORIENT OVERSEAS CONTAINER LINE (CHINA) CO., LTD",
    ) == ({1, 2}, "wrapped_carrier_signature_literal")
    assert _line_set_for_directive(
        "SIGNED OTHER LINE\nBY: (CHINA) CO., LTD\n",
        "documentPatch.parties.carrier.name",
        "ORIENT OVERSEAS CONTAINER LINE (CHINA) CO., LTD",
    ) == (set(), "unlocated")


def test_surface_requirement_context_disambiguates_repeated_date_lines() -> None:
    text = "Ship on Board Date\n05/05/2025\nPlace of Issue Date\n05/05/2025\n"
    requirement = SurfaceRenderingRequirement(
        kind="date",
        targetPath="documentPatch.issueDate",
        sourceSurface="05/05/2025",
        targetSurface="20/09/2024",
        sourceOccurrences=2,
        contextEvidence="Place of Issue Date\n05/05/2025\n",
    )

    assert _context_bound_surface_lines(text, requirement) == {4}


def test_conflicting_identical_dates_are_prefilled_by_semantic_heading() -> None:
    text = (
        "DATE LADEN ON BOARD\n"
        "11 MAY 2023\n\n"
        "PLACE OF BILL(S) ISSUE\n"
        "QINGDAO\n\n"
        "DATED\n"
        "11 MAY 2023\n"
    )
    workspace = RewriteWorkspace(
        original_text=text,
        current_text=text,
        surface_requirements=(
            SurfaceRenderingRequirement(
                kind="date",
                targetPath="documentPatch.issueDate",
                sourceSurface="11 MAY 2023",
                targetSurface="22 JUN 2025",
                sourceOccurrences=1,
                contextEvidence="DATED\n11 MAY 2023",
            ),
            SurfaceRenderingRequirement(
                kind="date",
                targetPath="documentPatch.shippedOnBoardDate",
                sourceSurface="11 MAY 2023",
                targetSurface="21 JUN 2025",
                sourceOccurrences=1,
                contextEvidence="DATE LADEN ON BOARD\n11 MAY 2023",
            ),
        ),
    )

    applied = apply_deterministic_prefills(workspace)

    assert workspace.current_text.splitlines()[1] == "21 JUN 2025"
    assert workspace.current_text.splitlines()[7] == "22 JUN 2025"
    assert {(row.lineId, row.targetPaths) for row in applied} == {
        ("L00002", ("documentPatch.shippedOnBoardDate",)),
        ("L00008", ("documentPatch.issueDate",)),
    }


def test_repeated_page_dates_remain_owned_by_their_semantic_heading() -> None:
    page = (
        "DATE LADEN ON BOARD\n"
        "11 MAY 2023\n\n"
        "DATED\n"
        "11 MAY 2023\n"
    )
    text = f"--- PAGE 1 ---\n{page}--- PAGE 2 ---\n{page}"
    workspace = RewriteWorkspace(
        original_text=text,
        current_text=text,
        surface_requirements=(
            SurfaceRenderingRequirement(
                kind="date",
                targetPath="documentPatch.issueDate",
                sourceSurface="11 MAY 2023",
                targetSurface="22 JUN 2025",
                sourceOccurrences=2,
                contextEvidence="DATED\n11 MAY 2023",
            ),
            SurfaceRenderingRequirement(
                kind="date",
                targetPath="documentPatch.shippedOnBoardDate",
                sourceSurface="11 MAY 2023",
                targetSurface="21 JUN 2025",
                sourceOccurrences=2,
                contextEvidence="DATE LADEN ON BOARD\n11 MAY 2023",
            ),
        ),
    )

    apply_deterministic_prefills(workspace)

    assert workspace.current_text.count("21 JUN 2025") == 2
    assert workspace.current_text.count("22 JUN 2025") == 2
    assert "11 MAY 2023" not in workspace.current_text


def test_residual_intervals_never_cross_page_markers() -> None:
    text = "--- PAGE 1 ---\nA\nB\n--- PAGE 2 ---\nC\nD\n"

    assert _expanded_intervals(text, {3, 5}, context=1, merge_gap=3) == ((2, 3), (5, 6))


def test_page_marker_split_discards_context_only_orphan_interval() -> None:
    text = "--- PAGE 1 ---\nCORE\n--- PAGE 2 ---\nCONTEXT ONLY\n"

    assert _expanded_intervals(text, {2}, context=2, merge_gap=3) == ((2, 2),)


def test_context_only_blank_span_edges_are_trimmed_but_internal_blanks_remain() -> None:
    bodies = ("", "VALUE", "", "SECOND", "")

    assert _trim_context_blank_edges(bodies, (1, 5), {2, 4}) == (2, 4)
    assert _trim_context_blank_edges(bodies, (1, 3), {1}) == (1, 2)


def test_reference_gap_detects_only_unchanged_source_lines() -> None:
    source = "UNCHANGED SECRET\nSOURCE VALUE\n"
    current = "UNCHANGED SECRET\nTARGET VALUE\n"
    reference = "ANONYMIZED SECRET\nALTERNATIVE TARGET\n"

    assert _baseline_reference_gap_lines(source, current, reference) == (1,)


def test_native_residual_output_is_applied_by_host_to_exact_span() -> None:
    workspace = RewriteWorkspace(original_text="OLD\nTAIL\n", current_text="OLD\nTAIL\n")
    context = _ResidualCommitContext(
        workspace=workspace,
        spans={
            "S001": ResidualSpan(
                spanId="S001",
                startLine=1,
                endLine=1,
                workItemIds=("W0001",),
                lines=("OLD",),
            )
        },
        authorized_line_ids=frozenset({"L00001"}),
    )
    output_type = _native_residual_output_type(context)
    output = output_type.model_validate(
        {
            "edits": [{"lineId": "L00001", "newLine": "NEW"}],
            "compoundPartyFlavorRealizations": [],
        },
        strict=True,
    )

    commit = _apply_native_residual_output(context, output)

    assert workspace.current_text == "NEW\nTAIL\n"
    assert len(commit.appliedReplacements) == 1


def test_native_residual_output_rejects_duplicate_line_edits() -> None:
    workspace = RewriteWorkspace(original_text="OLD\nTAIL\n", current_text="OLD\nTAIL\n")
    context = _ResidualCommitContext(
        workspace=workspace,
        spans={
            "S001": ResidualSpan(
                spanId="S001",
                startLine=1,
                endLine=2,
                workItemIds=("W0001",),
                lines=("OLD", "TAIL"),
            )
        },
        authorized_line_ids=frozenset({"L00001", "L00002"}),
    )
    output_type = _native_residual_output_type(context)
    output = output_type.model_validate(
        {
            "edits": [
                {"lineId": "L00001", "newLine": "NEW"},
                {"lineId": "L00001", "newLine": "OTHER"},
            ],
            "compoundPartyFlavorRealizations": [],
        },
        strict=True,
    )

    try:
        _apply_native_residual_output(context, output)
    except ValueError as error:
        assert str(error) == "native residual output repeats an authorized line ID"
    else:
        raise AssertionError("duplicate line edits were not rejected")


def test_native_residual_schema_rejects_empty_replacement_line() -> None:
    workspace = RewriteWorkspace(original_text="OLD\n", current_text="OLD\n")
    context = _ResidualCommitContext(
        workspace=workspace,
        spans={
            "S001": ResidualSpan(
                spanId="S001",
                startLine=1,
                endLine=1,
                workItemIds=("W0001",),
                lines=("OLD",),
            )
        },
        authorized_line_ids=frozenset({"L00001"}),
    )

    with pytest.raises(ValueError):
        _native_residual_output_type(context).model_validate(
            {
                "edits": [{"lineId": "L00001", "newLine": ""}],
                "compoundPartyFlavorRealizations": [],
            },
            strict=True,
        )


def test_native_residual_schema_rejects_context_only_line_edits() -> None:
    workspace = RewriteWorkspace(original_text="CORE\nCONTEXT\n", current_text="CORE\nCONTEXT\n")
    context = _ResidualCommitContext(
        workspace=workspace,
        spans={
            "S001": ResidualSpan(
                spanId="S001",
                startLine=1,
                endLine=2,
                workItemIds=("W0001",),
                lines=("CORE", "CONTEXT"),
            )
        },
        authorized_line_ids=frozenset({"L00001"}),
    )

    with pytest.raises(ValueError):
        _native_residual_output_type(context).model_validate(
            {
                "edits": [{"lineId": "L00002", "newLine": "CHANGED CONTEXT"}],
                "compoundPartyFlavorRealizations": [],
            },
            strict=True,
        )


def test_openrouter_pair_explicitly_enables_gateway_native_json_schema() -> None:
    config = load_synthesis_raw_text_hybrid_probe_config(
        Path("configs/synthesis/mpci_bl_raw_text_hybrid_compiler50_glm53_flash_poc1.yaml")
    )
    client = AsyncOpenAI(api_key="test", base_url="https://openrouter.ai/api/v1")

    editor, reviewer = _model_pair(
        client=client,
        editor_provider=config.providers.editor,
        reviewer_provider=config.providers.reviewer,
    )

    assert editor.profile["supports_json_schema_output"] is True
    assert reviewer.profile["supports_json_schema_output"] is True
