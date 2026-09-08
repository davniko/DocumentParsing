from __future__ import annotations

import json
from datetime import UTC, datetime
from decimal import Decimal
from pathlib import Path

import pytest
from pydantic import TypeAdapter, ValidationError
from pydantic_ai import Agent
from pydantic_ai.messages import ModelResponse, TextPart
from pydantic_ai.models.test import TestModel
from pydantic_ai.usage import RequestUsage

from document_ocr.synthesis.config import (
    load_synthesis_raw_text_rewrite_cycle_probe_config,
)
from document_ocr.synthesis.country_registry import load_iso_country_registry
from document_ocr.synthesis.linguistic_completion_pipeline import DocumentLinguisticPlan
from document_ocr.synthesis.linguistic_probe_runtime import LinguisticUsageReceipt, usage_receipt
from document_ocr.synthesis.package_registry import load_package_registry
from document_ocr.synthesis.raw_text_rewrite_cycle_probe import (
    AnchoredScalarReplacementRequirement,
    CargoFlavorRewriteRequirement,
    CompactLabelChangeDirective,
    CompoundPartyFlavorRealization,
    CustomsProgramEntry,
    EmpiricalOperationalProfile,
    LabelChangeDirective,
    LineRangeReplacement,
    OperationalFlavorRequirement,
    RewriteState,
    RewriteWorkspace,
    SemanticReviewFinding,
    SemanticReviewReceipt,
    SurfaceRenderingRequirement,
    TargetIntegrityResources,
    _aggregate_equipment_breakdown_requirements,
    _bounded_largest_remainder_allocation,
    _bounded_line_context,
    _cargo_flavor_rewrite_failures,
    _combine_usage,
    _container_measurement_occurrences,
    _dangerous_goods_context_surfaces,
    _dangerous_goods_proper_shipping_name_surfaces,
    _dangerous_goods_tuple_surfaces,
    _evidence_occurs,
    _finding_conflicts_with_anchored_scalar_authority,
    _finding_conflicts_with_equipment_authority,
    _finding_conflicts_with_surface_authority,
    _finding_grounds_format_damage_only_in_unchanged_text,
    _introduced_html_entities,
    _isolated_formatting_changes,
    _line_id,
    _membership_package_allocations,
    _missing_target_literals,
    _operational_flavor_requirements_rendered,
    _operational_measurement_match,
    _project_unrenderable_equipment_to_template,
    _prompt_content,
    _required_surfaces_rendered,
    _review_audit_payload,
    _settings,
    _single_container_unallocated_package_projection,
    _target_route_jurisdictions,
    _terminal_editor_output,
    aggregate_operational_replacement_requirements,
    anchored_measurement_replacement_requirements,
    anchored_package_quantity_replacement_requirements,
    anchored_scalar_replacement_requirements,
    apply_deterministic_prefills,
    apply_line_range_replacements,
    bind_cargo_package_surface_guards,
    build_empirical_operational_profiles,
    build_rewrite_contract_bundle,
    cargo_auxiliary_package_requirements,
    cargo_component_measurement_replacement_requirements,
    cargo_flavor_rewrite_requirements,
    cargo_package_quantity_replacement_requirements,
    cargo_package_type_replacement_requirements,
    compact_label_change_contract,
    compound_party_flavor_requirements,
    container_equipment_replacement_requirements,
    container_package_type_replacement_requirements,
    deterministic_rewrite_audit,
    editable_indexed_ocr_lines,
    exact_cargo_line_replacement_requirements,
    indexed_ocr_lines,
    inline_slot_topology_requirements,
    jurisdictional_surface_requirements,
    label_change_contract,
    merge_anchored_scalar_replacement_requirements,
    operational_flavor_requirements,
    party_country_metadata_replacement_requirements,
    prepare_target_integrity,
    raw_auxiliary_identity_requirements,
    recover_explicit_hs_target_facts,
    repair_overlapping_source_scalar_targets,
    rewrite_changed_leaves,
    source_semantic_role_hints,
    source_status_preservation_requirements,
    surface_rendering_requirements,
    target_literal_requirements,
    target_value_occurrence_requirements,
    unexpected_cargo_package_surfaces,
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


def test_repeated_cargo_descriptions_select_strong_rows_not_generic_heading() -> None:
    source_description = (
        "KNITTED FABRIC - RECYCLED KNITTED FABRIC - COMMODITY OF FABRIC"
    )
    raw_text = (
        "FABRIC\n\n"
        "BORU 701361-8 40' HW\n"
        "754 ROLLS - KNITTED FABRIC - RECYCLED KNITTED FABRIC - COMMODITY OF FABRIC\n"
        "HS CODE: 60062200\n\n"
        "BORU 701125-6 40' HW\n"
        "653 ROLLS - KNITTED FABRIC - RECYCLED KNITTED FABRIC - COMMODITY OF FABRIC\n"
        "HS CODE: 60062200\n"
    )
    source = {
        "documentPatch": {
            "cargoGroups": [
                {"groupId": "g1", "description": source_description, "hsCodes": ["60062200"]},
                {"groupId": "g2", "description": source_description, "hsCodes": ["60062200"]},
            ]
        }
    }
    target = {
        "documentPatch": {
            "cargoGroups": [
                {"groupId": "g1", "description": "SYNTHETIC PAPER LABELS", "hsCodes": ["48219046"]},
                {
                    "groupId": "g2",
                    "description": "SYNTHETIC BEARING HOUSINGS",
                    "hsCodes": ["84832073"],
                },
            ]
        }
    }

    requirements = cargo_flavor_rewrite_requirements(raw_text, source, target)

    assert [row.sourceLineIds for row in requirements] == [
        ("L00004",),
        ("L00008",),
    ]


def test_shared_hs_codes_are_projected_within_their_cargo_group_rows() -> None:
    source_description = (
        "KNITTED FABRIC - RECYCLED KNITTED FABRIC - COMMODITY OF FABRIC"
    )
    raw_text = (
        "BORU 701361-8 40' HW\n"
        "754 ROLLS - KNITTED FABRIC - RECYCLED KNITTED FABRIC - COMMODITY OF FABRIC\n"
        "HS CODE: 60062200\n\n"
        "BORU 701125-6 40' HW\n"
        "653 ROLLS - KNITTED FABRIC - RECYCLED KNITTED FABRIC - COMMODITY OF FABRIC\n"
        "HS CODE: 60062200\n"
    )
    source = {
        "documentPatch": {
            "cargoGroups": [
                {"groupId": "g1", "description": source_description, "hsCodes": ["60062200"]},
                {"groupId": "g2", "description": source_description, "hsCodes": ["60062200"]},
            ]
        }
    }
    target = {
        "documentPatch": {
            "cargoGroups": [
                {"groupId": "g1", "description": "SYNTHETIC PAPER LABELS", "hsCodes": ["48219046"]},
                {
                    "groupId": "g2",
                    "description": "SYNTHETIC BEARING HOUSINGS",
                    "hsCodes": ["84832073"],
                },
            ]
        }
    }

    requirements = tuple(
        row
        for row in surface_rendering_requirements(raw_text, source, target)
        if row.kind == "hs_code"
    )

    assert [(row.targetPath, row.targetSurface, row.sourceLineIds) for row in requirements] == [
        ("documentPatch.cargoGroups[0].hsCodes[0]", "48219046", ("L00003",)),
        ("documentPatch.cargoGroups[1].hsCodes[0]", "84832073", ("L00007",)),
    ]


def test_bounded_line_context_keeps_focus_and_omits_oversized_neighbor() -> None:
    lines = ("By", "Pte. Ltd. trading as Yang Ming", "X" * 700, "Sharon Liu")

    context = _bounded_line_context(
        lines,
        focus_index=1,
        before=1,
        after=2,
    )

    assert "Pte. Ltd. trading as Yang Ming" in context
    assert "By" in context
    assert "Sharon Liu" in context
    assert "X" * 700 not in context
    assert len(context) <= 600


def test_measure_allocation_redistributes_a_capacity_limited_weighted_share() -> None:
    assert _bounded_largest_remainder_allocation(
        1010,
        (132, 66),
        (348, 741),
    ) == (348, 662)


def test_single_container_aggregate_package_projection_requires_exact_source_total() -> None:
    source = {
        "documentPatch": {
            "cargoPackages": [
                {"groupId": "g1", "packageId": "p1", "quantity": 56},
                {"groupId": "g2", "packageId": "p2", "quantity": 55},
            ],
            "containers": [{"containerNumber": "MRSU3525663"}],
        }
    }
    target = {
        "documentPatch": {
            "cargoPackages": [
                {"groupId": "g1", "packageId": "p1", "quantity": 361},
                {"groupId": "g2", "packageId": "p2", "quantity": 602},
            ],
            "containers": [{"containerNumber": "MRSU0452645"}],
        }
    }
    occurrences = {0: (("package_quantity", 1, "111", "111 CARTONS", "labeled_measurement"),)}

    assert _single_container_unallocated_package_projection(
        source,
        target,
        (source["documentPatch"]["containers"][0],),
        (target["documentPatch"]["containers"][0],),
        occurrences,
    ) == {"MRSU0452645": 963}
    occurrences[0] = (("package_quantity", 1, "110", "110 CARTONS", "labeled_measurement"),)
    assert (
        _single_container_unallocated_package_projection(
            source,
            target,
            (source["documentPatch"]["containers"][0],),
            (target["documentPatch"]["containers"][0],),
            occurrences,
        )
        == {}
    )


def test_operational_output_allows_column_shift_before_owned_measurement() -> None:
    requirement = OperationalFlavorRequirement(
        requirementId="operational-L00001-package_quantity",
        kind="package_quantity",
        sourceGrammar="labeled_measurement",
        sourceLineId="L00001",
        sourceMeasurementStartColumn=19,
        sourceValueSurface="807",
        targetValueSurface="1637",
        sourceCanonicalValue="807",
        targetCanonicalValue="1637",
        consistencyGroupId="EITU6784428-package_quantity",
        targetContainerNumber="EITU6784428",
        targetEquipmentFamily="forty_high_cube",
        maximumValue=None,
        sameLineFollowingContainerNumber=None,
        empiricalProfileDocumentId=None,
        samplingMethod="target_package_allocation_v1",
        sourceEvidence="SOURCE/20'/SEAL/807 CARTONS",
    )

    assert _operational_flavor_requirements_rendered(
        "EITU6784428/40' HIGH CUBE GENERAL PURPOSE/SEAL/1637 PACKAGES\n",
        (requirement,),
    )

    assert _operational_flavor_requirements_rendered(
        "EITU6784428/40' HIGH CUBE GENERAL PURPOSE/SEAL/"
        "1637 INTERMEDIATE BULK CONTAINERS\n",
        (requirement,),
    )


def test_operational_package_parser_accepts_registry_vehicle_surface() -> None:
    match = _operational_measurement_match(
        "TOTAL: 85 VEHICLES",
        "package_quantity",
        expected_surface="85",
    )

    assert match is not None
    assert match.group("package").upper() == "VEHICLES"


def test_operational_package_prefill_owns_its_task_quantity_path() -> None:
    requirement = OperationalFlavorRequirement(
        requirementId="operational-L00001-package_quantity",
        kind="package_quantity",
        sourceGrammar="labeled_measurement",
        sourceLineId="L00001",
        sourceMeasurementStartColumn=23,
        sourceValueSurface="12",
        targetValueSurface="7",
        sourceCanonicalValue="12",
        targetCanonicalValue="7",
        consistencyGroupId="TEMU7054494-package_quantity",
        targetContainerNumber="TEMU7054494",
        targetEquipmentFamily="forty_high_cube",
        maximumValue=None,
        sameLineFollowingContainerNumber=None,
        empiricalProfileDocumentId=None,
        samplingMethod="target_package_allocation_v1",
        sourceEvidence="TEMU0157274 /218 295 / 12 PALLETS",
    )
    target = {
        "documentPatch": {
            "cargoPackages": [{"groupId": "g1", "packageId": "p1", "quantity": 7}],
            "cargoAllocationGroups": [
                {
                    "groupId": "g1",
                    "packageIds": ["p1"],
                    "allocations": [{"containerNumber": "TEMU7054494", "packageQuantity": 7}],
                }
            ],
            "containers": [{"containerNumber": "TEMU7054494"}],
        }
    }
    workspace = RewriteWorkspace(
        original_text="TEMU0157274 /218 295 / 12 PALLETS\n",
        current_text="TEMU0157274 /218 295 / 12 PALLETS\n",
        current_target_label=target,
        operational_flavor_requirements=(requirement,),
    )

    prefills = apply_deterministic_prefills(workspace)

    assert prefills[0].targetPaths == (
        "rawOperational.operational-L00001-package_quantity",
        "auxiliary.container[TEMU7054494].package_quantity",
        "documentPatch.cargoAllocationGroups[0].allocations[0].packageQuantity",
        "documentPatch.cargoPackages[0].quantity",
    )


def test_prepared_rewrite_contract_is_time_independent() -> None:
    source_label = {"documentPatch": {"transport": {"vesselName": "OLD VESSEL"}}}
    target_label = {"documentPatch": {"transport": {"vesselName": "NEW VESSEL"}}}
    workspace = RewriteWorkspace(
        original_text="VESSEL: OLD VESSEL\n",
        current_text="VESSEL: OLD VESSEL\n",
        source_label=source_label,
        upstream_target_label=target_label,
        current_target_label=target_label,
    )
    state = RewriteState(
        case_number=1,
        document_id="doc-contract",
        scenario_id="scenario-contract",
        feature={},
        source_label=source_label,
        target_integrity={"capacity_changed": False},
        workspace=workspace,
        started_at=datetime.min.replace(tzinfo=UTC),
    )

    first = build_rewrite_contract_bundle(state)
    state.started_at = datetime.now(UTC)
    second = build_rewrite_contract_bundle(state)

    assert first.result.durationMs == 0.0
    assert first.model_dump(mode="json") == second.model_dump(mode="json")


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
        route_port_countries_by_name={
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
                source_surfaces=(
                    "ACID-Advance Cargo information declaration",
                    "ACID NUMBER",
                    "ACID NO",
                    "ACID:",
                    "ACID#",
                ),
                generic_replacement_surface="CUSTOMS REFERENCE",
            ),
        ),
    )


def test_party_country_metadata_is_iso_resolved_and_source_format_preserving() -> None:
    raw_text = (
        "SHIPPER COUNTRY: GERMANY\n"
        "SHIPPER COUNTRY CODE: DE\n"
        "Foreign Exporter Country:HONG KONG\n"
        "FOREIGN EXPORTER COUNTRY CODE: HK\n"
        "CONSIGNEE COUNTRY: egypt\n"
    )
    target = {
        "documentPatch": {
            "parties": {
                "shipper": {"name": "TARGET EXPORTER", "country": "United States"},
                "consignee": {"name": "TARGET IMPORTER", "country": "Netherlands"},
            }
        }
    }

    requirements = party_country_metadata_replacement_requirements(
        raw_text, target, _target_integrity_resources()
    )
    workspace = RewriteWorkspace(
        original_text=raw_text,
        current_text=raw_text,
        anchored_scalar_replacement_requirements=requirements,
    )
    apply_deterministic_prefills(workspace)

    assert workspace.current_text == (
        "SHIPPER COUNTRY: UNITED STATES\n"
        "SHIPPER COUNTRY CODE: US\n"
        "Foreign Exporter Country:UNITED STATES\n"
        "FOREIGN EXPORTER COUNTRY CODE: US\n"
        "CONSIGNEE COUNTRY: netherlands\n"
    )
    assert {row.targetPaths for row in requirements} == {
        ("documentPatch.parties.shipper.country",),
        ("documentPatch.parties.consignee.country",),
    }


def test_party_country_metadata_uses_route_when_optional_party_country_is_absent() -> None:
    raw_text = "EXPORTER COUNTRY GERMANY\n"
    target = {
        "documentPatch": {
            "parties": {"shipper": {"name": "TARGET EXPORTER"}},
            "route": {"portOfLoading": {"name": "Kotzebue"}},
        }
    }

    requirements = party_country_metadata_replacement_requirements(
        raw_text, target, _target_integrity_resources()
    )
    workspace = RewriteWorkspace(
        original_text=raw_text,
        current_text=raw_text,
        anchored_scalar_replacement_requirements=requirements,
    )
    apply_deterministic_prefills(workspace)

    assert workspace.current_text == "EXPORTER COUNTRY UNITED STATES\n"
    assert requirements[0].targetPaths == ("documentPatch.route",)


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


def test_ambiguous_route_name_uses_only_unique_printed_party_country_intersection() -> None:
    baseline = _target_integrity_resources()
    resources = TargetIntegrityResources(
        countries=baseline.countries,
        packages=baseline.packages,
        route_countries_by_name={
            **baseline.route_countries_by_name,
            "ROTTERDAM": frozenset({"NL", "US"}),
        },
        route_port_countries_by_name={
            **baseline.route_port_countries_by_name,
            "ROTTERDAM": frozenset({"NL", "US"}),
        },
        customs_programs=baseline.customs_programs,
    )
    target = {
        "documentPatch": {
            "route": {"portOfDischarge": {"name": "Rotterdam"}},
            "parties": {
                "consignee": {"name": "TARGET BUYER", "country": "Netherlands"},
                "carrier": {"name": "THIRD PARTY CARRIER", "country": "Ukraine"},
            },
        }
    }

    assert _target_route_jurisdictions(target, resources)["import"] == {
        "isoAlpha2": "NL",
        "name": "Netherlands",
    }


def test_maritime_route_country_ignores_same_named_non_port_location() -> None:
    baseline = _target_integrity_resources()
    resources = TargetIntegrityResources(
        countries=baseline.countries,
        packages=baseline.packages,
        route_countries_by_name={
            **baseline.route_countries_by_name,
            "BASCO": frozenset({"PH", "US"}),
        },
        route_port_countries_by_name={
            **baseline.route_port_countries_by_name,
            "BASCO": frozenset({"PH"}),
        },
        customs_programs=baseline.customs_programs,
    )
    target = {
        "documentPatch": {
            "parties": {"shipper": {"name": "TARGET EXPORTER", "city": "Basco"}},
            "route": {"portOfLoading": {"name": "Basco"}},
        }
    }

    requirements = party_country_metadata_replacement_requirements(
        "EXPORTER REGISTRATION COUNTRY: CN\n", target, resources
    )
    workspace = RewriteWorkspace(
        original_text="EXPORTER REGISTRATION COUNTRY: CN\n",
        current_text="EXPORTER REGISTRATION COUNTRY: CN\n",
        anchored_scalar_replacement_requirements=requirements,
    )
    apply_deterministic_prefills(workspace)

    assert workspace.current_text == "EXPORTER REGISTRATION COUNTRY: PH\n"


def test_same_target_party_roles_own_an_extra_auxiliary_printed_occurrence() -> None:
    raw = "Consignee\nOLD FACTORY\nNotify Party\nOLD FACTORY\nIMPORTER: OLD FACTORY\n"
    leaves = rewrite_changed_leaves(
        {
            "documentPatch": {
                "parties": {
                    "consignee": {"name": "OLD FACTORY"},
                    "notifyParties": [{"name": "OLD FACTORY"}],
                }
            }
        },
        {
            "documentPatch": {
                "parties": {
                    "consignee": {"name": "NEW INDUSTRIES"},
                    "notifyParties": [{"name": "NEW INDUSTRIES"}],
                }
            }
        },
    )

    requirements = target_value_occurrence_requirements(raw, leaves)

    assert len(requirements) == 1
    assert requirements[0].targetValue == "NEW INDUSTRIES"
    assert requirements[0].requiredOccurrences == 3


def test_dangerous_goods_tuple_supports_explosive_compatibility_group() -> None:
    surfaces = _dangerous_goods_tuple_surfaces(
        "IMDG CLASS: 3 / UN NO.: 1993\n",
        source_un_number="1993",
        target_un_number="0402",
        target_exact_class="1.1D",
        target_packing_group=None,
    )

    assert surfaces == (("IMDG CLASS: 3 / UN NO.: 1993", "IMDG CLASS: 1.1D / UN NO.: 0402", False),)


def test_dangerous_goods_free_text_tail_is_replaced_with_target_shipping_name() -> None:
    source = "UN 3077 ENVIRONMENTALLY HAZARDOUS SUBSTANCE, SOLID, N.O.S.\n"

    surfaces = _dangerous_goods_proper_shipping_name_surfaces(
        source,
        source_un_number="3077",
        target_un_number="0402",
        target_proper_shipping_name="Ammonium perchlorate",
    )

    assert surfaces == (
        (
            "L00001",
            "UN 3077 ENVIRONMENTALLY HAZARDOUS SUBSTANCE, SOLID, N.O.S.",
            "UN 0402 AMMONIUM PERCHLORATE",
        ),
    )


def test_dangerous_goods_structured_tail_is_left_to_tuple_renderer() -> None:
    source = (
        "UN Number: 2078 - IMDG Class: 6.1 - PG: II\n"
        "Label/Subrisk: CLASS 9/- UN#: UN3077 Packaging Group: III "
        "Emergency Phone: 202-37609091\n"
        "CLASS:6.1 UNDG NO:2078**TAX\n"
    )

    assert not _dangerous_goods_proper_shipping_name_surfaces(
        source,
        source_un_number="2078",
        target_un_number="0213",
        target_proper_shipping_name="Trinitroanisole",
    )
    assert not _dangerous_goods_proper_shipping_name_surfaces(
        source,
        source_un_number="3077",
        target_un_number="0213",
        target_proper_shipping_name="Trinitroanisole",
    )
    assert not _dangerous_goods_proper_shipping_name_surfaces(
        source,
        source_un_number="2078",
        target_un_number="0213",
        target_proper_shipping_name="Trinitroanisole",
    )
    assert _dangerous_goods_tuple_surfaces(
        source,
        source_un_number="3077",
        target_un_number="0213",
        target_exact_class="1.1D",
        target_packing_group=None,
    ) == (
        (
            "CLASS 9/- UN#: UN3077 Packaging Group: III",
            "CLASS 1.1D/- UN#: UN0213",
            True,
        ),
    )


def test_dangerous_goods_context_rewrites_dense_and_labeled_semantics() -> None:
    source = (
        "9 3077 III\n"
        "Chemical Details:\n"
        "Substance Name(Proper Shipping Name): "
        "ENVIRONMENTALLY HAZARDOUS SUBSTANCE, SOLID, N.O.S.*\n"
        "DIMETHOMORPH 50% WP Class: 9\n"
        "Label/Subrisk: CLASS 9/- UN#: UN3077 Packaging Group: III\n"
    )

    surfaces = _dangerous_goods_context_surfaces(
        source,
        source_un_number="3077",
        target_un_number="0213",
        target_proper_shipping_name="TRINITROANISOLE",
        target_exact_class="1.1D",
        target_packing_group=None,
    )

    assert (
        "L00001",
        "9 3077 III",
        "1.1D 0213",
        "denseTuple",
    ) in surfaces
    assert (
        "L00003",
        "ENVIRONMENTALLY HAZARDOUS SUBSTANCE, SOLID, N.O.S.",
        "TRINITROANISOLE",
        "properShippingName",
    ) in surfaces
    assert (
        "L00004",
        "Class: 9",
        "Class: 1.1D",
        "exactHazardClass",
    ) in surfaces


def test_dg_component_weight_preserves_source_share_and_numeric_style() -> None:
    source = "(184 Fibreboard boxes-4G - 2008.000 kgs.) Proper Shipping Name: SOURCE\n"
    source_label = {
        "documentPatch": {
            "cargoGroups": [
                {
                    "groupId": "g1",
                    "description": "SOURCE",
                    "grossWeight": {"value": 2258.8, "unit": "kilogram"},
                }
            ]
        }
    }
    target_label = {
        "documentPatch": {
            "cargoGroups": [
                {
                    "groupId": "g1",
                    "description": "TARGET",
                    "grossWeight": {"value": 19464.1, "unit": "kilogram"},
                }
            ]
        }
    }
    surface = SurfaceRenderingRequirement(
        kind="dangerous_goods_tuple",
        targetPath="auxiliary.cargoGroups[0].dangerousGoods[0].properShippingName",
        sourceSurface="SOURCE",
        targetSurface="TARGET",
        sourceOccurrences=1,
        contextEvidence=source.strip(),
        sourceLineIds=("L00001",),
    )

    requirements = cargo_component_measurement_replacement_requirements(
        source,
        source_label,
        target_label,
        (),
        (surface,),
    )

    assert len(requirements) == 1
    assert requirements[0].sourceSurface == "2008.000"
    assert requirements[0].targetSurface == "17302.954"


@pytest.mark.parametrize(
    "source_line",
    ("BOBA PEARL 1KG X 18 BAGS", "132 CONTAINERS OF 24 KG NET EACH PRODUCT"),
)
def test_per_unit_packaging_weight_is_not_scaled_as_cargo_component(
    source_line: str,
) -> None:
    source_label = {
        "documentPatch": {
            "cargoGroups": [
                {
                    "groupId": "g1",
                    "description": "SOURCE GOODS",
                    "grossWeight": {"value": 12179.48, "unit": "kilogram"},
                }
            ]
        }
    }
    target_label = {
        "documentPatch": {
            "cargoGroups": [
                {
                    "groupId": "g1",
                    "description": "TARGET GOODS",
                    "grossWeight": {"value": 3040.41, "unit": "kilogram"},
                }
            ]
        }
    }
    cargo = (
        CargoFlavorRewriteRequirement(
            requirementId="cargo-group-1-span-1",
            targetPath="documentPatch.cargoGroups[0].description",
            targetDescription="TARGET GOODS",
            sourceLineIds=("L00001",),
            sourceSurfaces=(source_line,),
        ),
    )

    assert cargo_component_measurement_replacement_requirements(
        source_line + "\n",
        source_label,
        target_label,
        cargo,
        (),
    ) == ()


def test_changed_dg_package_material_drops_incompatible_un_packaging_code() -> None:
    source = "(184 Fibreboard boxes-4G - 2008.000 kgs.) Proper Shipping Name: SOURCE\n"
    source_label = {
        "documentPatch": {
            "cargoGroups": [{"groupId": "g1", "description": "SOURCE"}],
            "cargoPackages": [
                {
                    "groupId": "g1",
                    "packageId": "p1",
                    "quantity": 184,
                    "typeCategory": "PACKAGE_BOX_FIBREBOARD",
                }
            ],
        }
    }
    target_label = {
        "documentPatch": {
            "cargoGroups": [{"groupId": "g1", "description": "TARGET"}],
            "cargoPackages": [
                {
                    "groupId": "g1",
                    "packageId": "p1",
                    "quantity": 3160,
                    "typeCategory": "PACKAGE_CASE_WOODEN",
                }
            ],
        }
    }
    cargo = CargoFlavorRewriteRequirement(
        requirementId="cargo-group-1-span-1",
        targetPath="documentPatch.cargoGroups[0].description",
        targetDescription="TARGET",
        sourceLineIds=("L00001",),
        sourceSurfaces=(source.strip(),),
    )

    requirements = cargo_package_type_replacement_requirements(
        source,
        source_label,
        target_label,
        _target_integrity_resources().packages,
        (cargo,),
    )

    assert len(requirements) == 1
    assert requirements[0].sourceSurface == "Fibreboard boxes-4G"
    assert requirements[0].targetSurface == "wooden Cases"


def test_cargo_auxiliary_package_rows_are_owned_until_the_next_heading() -> None:
    source = (
        "3160 Wooden Case(s) of TARGET\n"
        "PRODUCT CODE:100G*100BAGS/CARTON: 0402620044\n"
        "300G*40BAGS/CARTON: 0402620045\n"
        "(3160 wooden Cases - 17302.954 kgs.) TARGET\n"
        "Total: 19,464.100 kgs. 44.100 cu. m.\n"
    )
    guarded = CargoFlavorRewriteRequirement(
        requirementId="cargo-group-1-span-1",
        targetPath="documentPatch.cargoGroups[0].description",
        targetDescription="TARGET",
        sourceLineIds=("L00001",),
        sourceSurfaces=("3160 Wooden Case(s) of SOURCE",),
        enforcePackageSurfaceGuard=True,
        allowedPackageSurfaces=("WOODEN CASE", "WOODEN CASES"),
    )

    requirements = cargo_auxiliary_package_requirements(source, (guarded,))

    assert len(requirements) == 2
    assert requirements[0].lineRole == "label_grounded"
    assert requirements[1].lineRole == "source_only_auxiliary_packaging"
    assert requirements[1].sourceLineIds == ("L00002", "L00003")
    assert unexpected_cargo_package_surfaces(source.splitlines()[1], guarded) == (
        "BAGS",
        "CARTON",
    )


def test_cargo_auxiliary_scan_stops_before_carrier_receipt_total() -> None:
    source = (
        "TARGET GOODS\n"
        "PRODUCT PACKED IN SOURCE BAGS\n"
        "Total number of containers or packages 1 received by Carrier:\n"
        "SIGNED BY CARRIER\n"
    )
    guarded = CargoFlavorRewriteRequirement(
        requirementId="cargo-group-1-span-1",
        targetPath="documentPatch.cargoGroups[0].description",
        targetDescription="TARGET GOODS",
        sourceLineIds=("L00001",),
        sourceSurfaces=("SOURCE GOODS",),
        enforcePackageSurfaceGuard=True,
        allowedPackageSurfaces=("CASE", "CASES"),
    )

    requirements = cargo_auxiliary_package_requirements(source, (guarded,))

    assert len(requirements) == 2
    assert requirements[1].sourceLineIds == ("L00002",)


def test_cargo_package_guard_ignores_non_package_lot_and_unit_prose() -> None:
    guarded = CargoFlavorRewriteRequirement(
        requirementId="cargo-group-1-span-1",
        targetPath="documentPatch.cargoGroups[0].description",
        targetDescription="TARGET GOODS",
        sourceLineIds=("L00001",),
        sourceSurfaces=("SOURCE GOODS",),
        enforcePackageSurfaceGuard=True,
        allowedPackageSurfaces=("CARTON", "CARTONS"),
    )

    assert unexpected_cargo_package_surfaces("TARGET GOODS EXPORT LOT", guarded) == ()
    assert unexpected_cargo_package_surfaces("Rate Unit Currency Prepaid Collect", guarded) == ()
    assert unexpected_cargo_package_surfaces("Collection Business Unit", guarded) == ()
    assert unexpected_cargo_package_surfaces(
        "CARGO IS STOWED IN A REFRIGERATED CONTAINER SET AT PLUS 1 DEG C", guarded
    ) == ()
    assert unexpected_cargo_package_surfaces("10 LOTS OF TARGET GOODS", guarded) == ("LOTS",)
    assert unexpected_cargo_package_surfaces("PACKED IN 4 UNITS", guarded) == ("UNITS",)
    assert unexpected_cargo_package_surfaces("8 SETS OF TARGET GOODS", guarded) == ("SETS",)


def test_cargo_auxiliary_scan_excludes_path_owned_container_row() -> None:
    source = (
        "SOURCE GOODS\n"
        "MSCU1234567 /SEAL 10 CASES /FCL/FCL /40HQ/\n"
        "PRODUCT CODE:100G*100BAGS/CARTON: 0402620044\n"
    )
    guarded = CargoFlavorRewriteRequirement(
        requirementId="cargo-group-1-span-1",
        targetPath="documentPatch.cargoGroups[0].description",
        targetDescription="TARGET GOODS",
        sourceLineIds=("L00001",),
        sourceSurfaces=("SOURCE GOODS",),
        enforcePackageSurfaceGuard=True,
        allowedPackageSurfaces=("PACKAGE", "PACKAGES"),
    )

    requirements = cargo_auxiliary_package_requirements(
        source,
        (guarded,),
        excluded_line_ids=frozenset(("L00002",)),
    )

    assert len(requirements) == 2
    assert requirements[1].sourceLineIds == ("L00003",)


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
        effective["documentPatch"]["cargoGroups"][index]["hsCodes"][0] for index in range(2)
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
                "documentPatch": {"cargoGroups": [{"groupId": "g1", "hsCodes": ["111111"]}]}
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


def test_atomic_patch_rejects_new_html_entities_in_plain_ocr() -> None:
    source = "--- PAGE 1 ---\nOLD COMPANY NAME\n"
    workspace = _workspace(source)

    assert _introduced_html_entities(source, "--- PAGE 1 ---\nHUA&#39;AN INDUSTRIES\n") == (
        "&#39;",
    )
    with pytest.raises(ValueError, match="HTML/XML entities"):
        apply_line_range_replacements(
            workspace,
            (_edit(source, 2, 2, "HUA&#39;AN INDUSTRIES"),),
        )

    assert workspace.current_text == source


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


def test_atomic_patch_rejects_unicode_lexical_line_degeneration() -> None:
    source = "--- PAGE 1 ---\nTEMPÉRATURE RÉGLÉE.\n"
    workspace = _workspace(source)

    with pytest.raises(ValueError, match="punctuation-only filler"):
        apply_line_range_replacements(
            workspace,
            (_edit(source, 2, 2, "."),),
        )

    assert workspace.current_text == source


def test_atomic_patch_rewrites_positive_operation_when_setpoint_is_deactivated() -> None:
    source = (
        "--- PAGE 1 ---\nReefer temperature to be set at -21 C\nPlugging for the account of cargo\n"
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
        ("ACID#", 1),
    ]
    assert {row.targetRouteCountryCode for row in requirements} == {"JP"}
    assert [row.sourceLineIds for row in requirements] == [("L00001",), ("L00002",)]
    assert len({row.requirementId for row in requirements}) == 2


def test_customs_program_punctuation_selector_requires_a_line_initial_field() -> None:
    source = "FATTY ACID: STEARIC ACID\nACID : 1002845232025040024\n** ACID:9079064167722408620\n"
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

    assert [(row.sourceSurface, row.sourceLineIds) for row in requirements] == [
        ("ACID:", ("L00002", "L00003")),
    ]


def test_customs_program_requirement_is_not_emitted_for_its_own_jurisdiction() -> None:
    requirements = jurisdictional_surface_requirements(
        "ACID#: 1002845232025040024\n",
        {"documentPatch": {"route": {"portOfDischarge": {"name": "El Iskandariya (Alexandria)"}}}},
        _target_integrity_resources(),
    )

    assert requirements == ()


def test_atomic_patch_requires_stale_customs_program_replacement_without_matching_acidic() -> None:
    source = "--- PAGE 1 ---\nACID#: 1002845232025040024\nGOODS: ACIDIC, ORGANIC\n"
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


def test_shared_target_contact_with_unlocated_owner_omits_unsafe_global_count() -> None:
    source = "DELIVERY AGENT\nTEL:+20 3 481 4950 / EXT\n112\nNOTIFY PARTY\nTEL:02 23546140\n"
    leaves = (
        ChangedLeaf(
            path="documentPatch.parties.deliveryAgent.contactDetails.phoneNumbers[0]",
            sourcePresent=True,
            targetPresent=True,
            sourceValue="+20 3 481 4950 / EXT 112",
            targetValue="+84 912 684 307",
            evidenceClass="printed_fact",
            requiresTextEdit=True,
        ),
        ChangedLeaf(
            path="documentPatch.parties.notifyParties[0].contactDetails.phoneNumbers[0]",
            sourcePresent=True,
            targetPresent=True,
            sourceValue="02 23546140",
            targetValue="+84 912 684 307",
            evidenceClass="printed_fact",
            requiresTextEdit=True,
        ),
    )
    anchored = anchored_scalar_replacement_requirements(source, leaves)

    requirements = target_value_occurrence_requirements(
        source,
        leaves,
        anchored_replacements=anchored,
    )

    # The delivery value spans two lines and has no exact anchored replacement. Counting only
    # the notify-party occurrence would make the valid duplicate target impossible. Role-block
    # postconditions still require the value independently in both parties.
    assert requirements == ()


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


def test_inventory_can_defer_ambiguous_repeated_party_scalar_cardinality() -> None:
    source = (
        "--- PAGE 1 ---\nCONSIGNEE\nOLD COMPANY\n"
        "--- PAGE 2 ---\nCONSIGNEE\nOLD COMPANY\nNOTIFY PARTY\nOLD COMPANY\n"
    )
    leaves = (
        ChangedLeaf(
            path="documentPatch.parties.consignee.name",
            sourcePresent=True,
            targetPresent=True,
            sourceValue="OLD COMPANY",
            targetValue="NEW CONSIGNEE LTD",
            evidenceClass="printed_fact",
            requiresTextEdit=True,
        ),
        ChangedLeaf(
            path="documentPatch.parties.notifyParties[0].name",
            sourcePresent=True,
            targetPresent=True,
            sourceValue="OLD COMPANY",
            targetValue="NEW NOTIFY LTD",
            evidenceClass="printed_fact",
            requiresTextEdit=True,
        ),
    )

    with pytest.raises(ValueError, match="cannot be assigned exactly"):
        target_value_occurrence_requirements(source, leaves)

    provisional = target_value_occurrence_requirements(
        source,
        leaves,
        defer_ambiguous_party_scalar_cardinality=True,
    )

    assert {row.targetValue: row.requiredOccurrences for row in provisional} == {
        "NEW CONSIGNEE LTD": 2,
        "NEW NOTIFY LTD": 1,
    }


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


def test_embedded_identifier_occurrence_is_owned_by_the_longer_printed_mark() -> None:
    source = "--- PAGE 1 ---\nMARK: ALX-BR22610\nEXPORT REF: BR22610\n"
    leaves = (
        ChangedLeaf(
            path="documentPatch.cargoGroups[0].marksAndNumbers[0]",
            sourcePresent=True,
            targetPresent=True,
            sourceValue="ALX-BR22610",
            targetValue="MVS-40731",
            evidenceClass="printed_fact",
            requiresTextEdit=True,
        ),
        ChangedLeaf(
            path="documentPatch.forwardingAndExportReferences[0]",
            sourcePresent=True,
            targetPresent=True,
            sourceValue="BR22610",
            targetValue="UE29015",
            evidenceClass="printed_fact",
            requiresTextEdit=True,
        ),
    )

    requirements = anchored_scalar_replacement_requirements(source, leaves)

    assert [(row.sourceSurface, row.sourceLineIds) for row in requirements] == [
        ("ALX-BR22610", ("L00002",)),
        ("BR22610", ("L00003",)),
    ]


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
    source = "--- PAGE 1 ---\nCONSIGNEE: OLD COMPANY\n12 LONG INDUSTRIAL\nROAD BUILDING 7\n"
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
    source_label = {"documentPatch": {"parties": {"carrier": {"name": "OLD OCEAN LINE S.A."}}}}
    target_label = {"documentPatch": {"parties": {"carrier": {"name": "NEW OCEAN LINE AG"}}}}
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


def test_decorated_party_heading_is_protected_from_party_value_rewrite() -> None:
    heading = "CONSIGNEE (3) (NOT NEGOTIABLE UNLESS CONSIGNED TO ORDER)"
    source = f"--- PAGE 1 ---\n{heading}\nTO ORDER\n"
    leaf = ChangedLeaf(
        path="documentPatch.parties.consignee.name",
        sourcePresent=True,
        targetPresent=True,
        sourceValue="TO ORDER",
        targetValue="Kestrel Meridian Trading Ltd.",
        evidenceClass="printed_fact",
        requiresTextEdit=True,
    )
    requirements = source_status_preservation_requirements(source, (leaf,))

    assert [(row.sourceSurface, row.sourceOccurrences) for row in requirements] == [(heading, 1)]
    workspace = RewriteWorkspace(
        original_text=source,
        current_text=source,
        source_status_requirements=requirements,
    )
    with pytest.raises(ValueError, match="unchanged source status"):
        apply_line_range_replacements(
            workspace,
            (_edit(source, 2, 2, "CONSIGNEE (3) (NOT NEGOTIABLE UNLESS CONSIGNED AS INDICATED)"),),
        )


def test_evergreen_legal_terms_are_not_parsed_as_auxiliary_carrier_identity() -> None:
    source = (
        "--- PAGE 1 ---\n"
        "NOT NEGOTIABLE UNLESS CONSIGNED TO ORDER\n"
        "ORIGINAL\n\n"
        "(TERMS OF BILL OF LADING ARE NOT NEGOTIABLE UNLESS CONSIGNED TO ORDER ORIGINAL)\n\n"
        "As agent for the Carrier and the Vessel Provider Evergreen Marine (Asia) Pte. Ltd.\n"
    )
    source_label = {
        "documentPatch": {
            "parties": {"carrier": {"name": "Evergreen Marine (Asia) Pte. Ltd."}}
        }
    }
    target_label = {
        "documentPatch": {
            "parties": {"carrier": {"name": "Marivanta Ocean Carriers GmbH"}}
        }
    }

    requirements = raw_auxiliary_identity_requirements(source, source_label, target_label)
    statuses = source_status_preservation_requirements(source, ())

    assert requirements == ()
    assert {row.sourceSurface for row in statuses} == {
        "NOT NEGOTIABLE UNLESS CONSIGNED TO ORDER",
        "(TERMS OF BILL OF LADING ARE NOT NEGOTIABLE UNLESS CONSIGNED TO ORDER ORIGINAL)",
    }


def test_legal_clause_with_changed_named_principal_is_not_byte_protected() -> None:
    line = (
        "(TERMS OF BILL OF LADING ARE CONTINUED ON THE SHEET. The legal provider "
        'Evergreen Marine (Asia) Pte. Ltd. does business as "Evergreen Line")'
    )
    leaf = ChangedLeaf(
        path="documentPatch.parties.carrier.name",
        sourcePresent=True,
        targetPresent=True,
        sourceValue="Evergreen Marine (Asia) Pte. Ltd.",
        targetValue="Marineridge Ocean Transport Ltd.",
        evidenceClass="printed_fact",
        requiresTextEdit=True,
    )

    assert source_status_preservation_requirements(line, (leaf,)) == ()


def test_legal_clause_to_order_grammar_remains_byte_protected() -> None:
    line = "(TERMS OF BILL OF LADING ARE NOT NEGOTIABLE UNLESS CONSIGNED TO ORDER ORIGINAL)"
    leaf = ChangedLeaf(
        path="documentPatch.parties.consignee.name",
        sourcePresent=True,
        targetPresent=True,
        sourceValue="TO ORDER",
        targetValue="Kestrel Meridian Trading Ltd.",
        evidenceClass="printed_fact",
        requiresTextEdit=True,
    )

    requirements = source_status_preservation_requirements(line, (leaf,))
    assert [(row.sourceSurface, row.sourceOccurrences) for row in requirements] == [(line, 1)]


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
        f"dococr:rewrite16:gpt-5.6-luna:e:{config.prompts.editor.sha256[:12]}"
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
        use_cache_point=True,
    )

    assert content[0] == "Stable synthetic B/L atomic editor contract version 16."
    assert content[1].kind == "cache-point"
    assert "doc-private" not in str(content[:2])
    assert "PRIVATE OCR" in content[2]


def test_openrouter_comparison_config_uses_same_cases_without_unsupported_cache() -> None:
    openrouter = load_synthesis_raw_text_rewrite_cycle_probe_config(
        Path("configs/synthesis/mpci_bl_raw_text_atomic10_glm_5_3_flash_openrouter.yaml")
    )
    luna = load_synthesis_raw_text_rewrite_cycle_probe_config(
        Path("configs/synthesis/mpci_bl_raw_text_atomic10_luna_high_v15.yaml")
    )

    assert openrouter.schema_version == 13
    assert openrouter.cases == luna.cases
    assert openrouter.prompts == luna.prompts
    assert openrouter.inputs == luna.inputs
    assert openrouter.target_integrity == luna.target_integrity
    assert openrouter.providers.editor.kind == "openrouter"
    assert openrouter.providers.editor.model == "z-ai/glm-5.3-flash"
    assert openrouter.providers.editor.api_key_env == "OPENROUTER_API_KEY"
    assert openrouter.providers.editor.require_parameters is True
    assert openrouter.providers.editor.data_collection == "deny"
    assert openrouter.providers.editor.provider_only == ("deepinfra",)
    assert openrouter.providers.editor.allow_fallbacks is False
    assert openrouter.providers.editor.max_price is not None
    assert openrouter.providers.editor.max_price.prompt == 0.075
    assert openrouter.providers.editor.max_price.completion == 0.25
    assert openrouter.workflow.use_provider_explicit_prompt_cache is False


def test_openrouter_settings_use_required_tool_parameters_without_openai_fields() -> None:
    config = load_synthesis_raw_text_rewrite_cycle_probe_config(
        Path("configs/synthesis/mpci_bl_raw_text_atomic10_glm_5_3_flash_openrouter.yaml")
    )

    settings = _settings(
        config.providers.editor,
        stage="editor",
        prompt_sha256=config.prompts.editor.sha256,
    )

    assert settings["openrouter_reasoning"] == {"effort": "high"}
    assert settings["openrouter_provider"] == {
        "require_parameters": True,
        "data_collection": "deny",
        "allow_fallbacks": False,
        "only": ["deepinfra"],
        "max_price": {"prompt": 0.075, "completion": 0.25},
    }
    assert "parallel_tool_calls" not in settings
    assert "openai_prompt_cache_key" not in settings
    assert "openai_previous_response_id" not in settings


def test_openrouter_prompt_payload_has_no_unsupported_cache_point() -> None:
    content = _prompt_content(
        stage="reviewer",
        immutable={"documentId": "doc-private"},
        active={"rewrittenRawOcr": "PRIVATE OCR"},
        use_cache_point=False,
    )

    assert content[0] == "Stable synthetic B/L atomic reviewer contract version 16."
    assert len(content) == 2
    assert "doc-private" not in content[0]
    assert "PRIVATE OCR" in content[1]


def test_openrouter_usage_receipt_records_gateway_cost_and_downstream_provider() -> None:
    config = load_synthesis_raw_text_rewrite_cycle_probe_config(
        Path("configs/synthesis/mpci_bl_raw_text_atomic10_glm_5_3_flash_openrouter.yaml")
    )
    response = ModelResponse(
        parts=[TextPart(content="ok")],
        usage=RequestUsage(
            input_tokens=100,
            output_tokens=20,
            details={"reasoning_tokens": 5},
        ),
        provider_response_id="generation-test",
        provider_details={"cost": 0.0000125, "downstream_provider": "Z.AI"},
        finish_reason="stop",
    )

    receipt = usage_receipt(
        [response],
        config.providers.editor.pricing,
        require_provider_cost=True,
    )

    assert receipt.providerReportedCostUsd == Decimal("0.000012500000")
    assert receipt.downstreamProviders == ("Z.AI",)
    assert receipt.reasoningTokens == 5
    assert receipt.visibleOutputTokens == 15


def test_usage_receipt_retains_provider_reasoning_accounting_anomaly() -> None:
    config = load_synthesis_raw_text_rewrite_cycle_probe_config(
        Path("configs/synthesis/mpci_bl_raw_text_atomic10_glm_5_3_flash_openrouter.yaml")
    )
    response = ModelResponse(
        parts=[TextPart(content="ok")],
        usage=RequestUsage(
            input_tokens=100,
            output_tokens=5,
            details={"reasoning_tokens": 7},
        ),
        provider_response_id="generation-anomalous-usage",
        provider_details={"cost": 0.0000125, "downstream_provider": "DeepInfra"},
        finish_reason="stop",
    )

    receipt = usage_receipt(
        [response],
        config.providers.editor.pricing,
        require_provider_cost=True,
    )

    assert receipt.outputTokens == 5
    assert receipt.reasoningTokens == 7
    assert receipt.visibleOutputTokens == 0
    assert receipt.providerTokenAccountingAnomaly is True


def test_combined_usage_never_publishes_a_partial_provider_cost_sum() -> None:
    common = {
        "requests": 1,
        "providerResponseIds": ("response",),
        "finishReasons": ("stop",),
        "inputTokens": 100,
        "cacheReadTokens": 0,
        "cacheWriteTokens": 0,
        "outputTokens": 20,
        "reasoningTokens": 5,
        "visibleOutputTokens": 15,
        "estimatedCostUsd": Decimal("0.000012"),
        "downstreamProviders": ("DeepInfra",),
    }
    receipted = LinguisticUsageReceipt(
        **common,
        providerReportedCostUsd=Decimal("0.000010"),
    )
    unreceipted = LinguisticUsageReceipt(
        **{**common, "providerResponseIds": ("response-2",)},
        providerReportedCostUsd=None,
    )

    combined = _combine_usage((receipted, unreceipted))

    assert combined.requests == 2
    assert combined.estimatedCostUsd == Decimal("0.000024")
    assert combined.providerReportedCostUsd is None


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
    assert {
        (row.targetPath, row.sourceOccurrences) for row in requirements if row.kind == "date"
    } == {
        ("documentPatch.shippedOnBoardDate", 1),
        ("documentPatch.issueDate", 1),
    }


def test_repeated_shared_dates_preserve_every_named_month_punctuation_style() -> None:
    raw = (
        "PLACE OF B(S)/L ISSUE/DATE\nNINGBO, CHINA JUN.20,2023\n"
        "SHIPPED ON BOARD, DATE\n20.JUN.2023\n"
    )
    source = {
        "documentPatch": {
            "issueDate": "2023-06-20",
            "shippedOnBoardDate": "2023-06-20",
        }
    }
    target = {
        "documentPatch": {
            "issueDate": "2024-05-02",
            "shippedOnBoardDate": "2024-05-02",
        }
    }

    requirements = surface_rendering_requirements(raw, source, target)

    assert {(row.sourceSurface, row.targetSurface) for row in requirements} == {
        ("JUN.20,2023", "MAY.02,2024"),
        ("20.JUN.2023", "02.MAY.2024"),
    }
    workspace = RewriteWorkspace(
        original_text=raw,
        current_text=raw,
        surface_requirements=requirements,
    )
    apply_deterministic_prefills(workspace)
    assert "MAY.02,2024" in workspace.current_text
    assert "02.MAY.2024" in workspace.current_text


def test_shared_identical_date_surface_uses_one_document_wide_occurrence_contract() -> None:
    raw = "SHIPPED ON BOARD DATE\n09/01/2024\nPLACE AND DATE OF ISSUE\n09/01/2024\n"
    source = {
        "documentPatch": {
            "issueDate": "2024-01-09",
            "shippedOnBoardDate": "2024-01-09",
        }
    }
    target = {
        "documentPatch": {
            "issueDate": "2023-05-08",
            "shippedOnBoardDate": "2023-05-08",
        }
    }

    requirements = surface_rendering_requirements(raw, source, target)

    assert [
        (row.kind, row.targetPath, row.sourceSurface, row.targetSurface, row.sourceOccurrences)
        for row in requirements
    ] == [
        (
            "date_global",
            "documentPatch.issueDate;documentPatch.shippedOnBoardDate",
            "09/01/2024",
            "08/05/2023",
            2,
        )
    ]
    workspace = RewriteWorkspace(
        original_text=raw,
        current_text=raw,
        surface_requirements=requirements,
    )
    apply_deterministic_prefills(workspace)
    assert workspace.current_text.count("08/05/2023") == 2
    assert "09/01/2024" not in workspace.current_text
    assert _required_surfaces_rendered(workspace.current_text, requirements)


def test_date_issued_heading_owns_issue_date_before_shipped_on_board_line() -> None:
    raw = "PLACE ISSUED: NANSHA ,China\nDATE ISSUED: May 7, 2025\nSHIPPED ON BOARD: May 7, 2025\n"
    source = {
        "documentPatch": {
            "issueDate": "2025-05-07",
            "shippedOnBoardDate": "2025-05-07",
        }
    }
    target = {
        "documentPatch": {
            "issueDate": "2025-06-28",
            "shippedOnBoardDate": "2025-06-29",
        }
    }

    requirements = surface_rendering_requirements(raw, source, target)

    assert {
        (row.targetPath, row.sourceOccurrences) for row in requirements if row.kind == "date"
    } == {
        ("documentPatch.issueDate", 1),
        ("documentPatch.shippedOnBoardDate", 1),
    }
    workspace = RewriteWorkspace(
        original_text=raw,
        current_text=raw,
        surface_requirements=requirements,
    )
    apply_deterministic_prefills(workspace)
    assert "DATE ISSUED: Jun 28, 2025" in workspace.current_text
    assert "SHIPPED ON BOARD: Jun 29, 2025" in workspace.current_text
    assert _required_surfaces_rendered(workspace.current_text, requirements)


def test_changed_date_does_not_overwrite_unchanged_field_with_same_source_value() -> None:
    raw = "SHIPPED ON BOARD DATE\n05/05/2024\nPLACE AND DATE OF ISSUE\n05/05/2024\n"
    source = {
        "documentPatch": {
            "issueDate": "2024-05-05",
            "shippedOnBoardDate": "2024-05-05",
        }
    }
    target = {
        "documentPatch": {
            "issueDate": "2024-09-20",
            "shippedOnBoardDate": "2024-05-05",
        }
    }

    requirements = surface_rendering_requirements(raw, source, target)

    assert [(row.kind, row.targetPath, row.sourceOccurrences) for row in requirements] == [
        ("date", "documentPatch.issueDate", 1)
    ]
    workspace = RewriteWorkspace(
        original_text=raw,
        current_text=raw,
        surface_requirements=requirements,
    )
    apply_deterministic_prefills(workspace)
    assert "SHIPPED ON BOARD DATE\n05/05/2024" in workspace.current_text
    assert "PLACE AND DATE OF ISSUE\n20/09/2024" in workspace.current_text
    assert _required_surfaces_rendered(workspace.current_text, requirements)


def test_date_rendering_supports_year_named_month_and_month_punctuation() -> None:
    source = {
        "documentPatch": {
            "issueDate": "2025-11-16",
            "shippedOnBoardDate": "2022-08-06",
        }
    }
    target = {
        "documentPatch": {
            "issueDate": "2024-12-16",
            "shippedOnBoardDate": "2024-04-08",
        }
    }
    raw = "Place and date of issue\nAntwerp / 2025-NOV-16\nOn Board Date\nAUG. 06, 2022\n"

    requirements = surface_rendering_requirements(raw, source, target)

    assert {(row.sourceSurface, row.targetSurface) for row in requirements} == {
        ("2025-NOV-16", "2024-DEC-16"),
        ("AUG. 06, 2022", "APR. 08, 2024"),
    }


def test_hs_rendering_splits_delimiter_list_and_expands_one_aggregate_slot() -> None:
    two_source = {
        "documentPatch": {
            "cargoGroups": [
                {"groupId": "g1", "hsCodes": ["52094200"]},
                {"groupId": "g2", "hsCodes": ["52114200"]},
            ]
        }
    }
    two_target = {
        "documentPatch": {
            "cargoGroups": [
                {"groupId": "g1", "hsCodes": ["48055080"]},
                {"groupId": "g2", "hsCodes": ["55121990"]},
            ]
        }
    }
    requirements = surface_rendering_requirements(
        "DENIM FABRIC HS CODE: 52094200 - 52114200\n", two_source, two_target
    )
    assert [(row.sourceSurface, row.targetSurface) for row in requirements] == [
        ("52094200", "48055080"),
        ("52114200", "55121990"),
    ]
    requirements = surface_rendering_requirements(
        "NCM: 52094200/52114200\n", two_source, two_target
    )
    assert [(row.sourceSurface, row.targetSurface) for row in requirements] == [
        ("52094200", "48055080"),
        ("52114200", "55121990"),
    ]

    aggregate_source = {
        "documentPatch": {
            "cargoGroups": [
                {"groupId": "g1", "hsCodes": ["2401100010"]},
                {"groupId": "g2", "hsCodes": ["2401100010"]},
            ]
        }
    }
    aggregate_target = {
        "documentPatch": {
            "cargoGroups": [
                {"groupId": "g1", "hsCodes": ["0902400010"]},
                {"groupId": "g2", "hsCodes": ["1209910010"]},
            ]
        }
    }
    requirements = surface_rendering_requirements(
        "HS CODE: 2401.100.010\n", aggregate_source, aggregate_target
    )
    assert len(requirements) == 1
    assert requirements[0].kind == "hs_code_block"
    assert requirements[0].sourceSurface == "2401.100.010"
    assert requirements[0].targetSurface == "0902.400.010, 1209.910.010"


def test_repeated_identical_hs_surfaces_keep_exact_row_ownership() -> None:
    source = {
        "documentPatch": {
            "cargoGroups": [
                {"groupId": "g1", "hsCodes": ["2401108590"]},
                {"groupId": "g2", "hsCodes": ["2401108590"]},
            ]
        }
    }
    target = {
        "documentPatch": {
            "cargoGroups": [
                {"groupId": "g1", "hsCodes": ["8418294258"]},
                {"groupId": "g2", "hsCodes": ["8524119097"]},
            ]
        }
    }
    raw = "HS CODE: 2401108590\nHS CODE: 2401108590\n"

    requirements = surface_rendering_requirements(raw, source, target)

    assert [(row.sourceLineIds, row.targetSurface) for row in requirements] == [
        (("L00001",), "8418294258"),
        (("L00002",), "8524119097"),
    ]
    workspace = RewriteWorkspace(
        original_text=raw,
        current_text=raw,
        surface_requirements=requirements,
    )

    apply_deterministic_prefills(workspace)

    assert workspace.current_text == ("HS CODE: 8418294258\nHS CODE: 8524119097\n")


def test_conflicting_hs_targets_on_one_line_are_rendered_by_exact_position() -> None:
    requirements = (
        SurfaceRenderingRequirement(
            kind="hs_code",
            targetPath="documentPatch.cargoGroups[0].hsCodes[0]",
            sourceSurface="60062200",
            targetSurface="48219046",
            sourceOccurrences=1,
            contextEvidence="HS CODE: 60062200 - 60062200",
            sourceLineIds=("L00001",),
        ),
        SurfaceRenderingRequirement(
            kind="hs_code",
            targetPath="documentPatch.cargoGroups[1].hsCodes[0]",
            sourceSurface="60062200",
            targetSurface="84832073",
            sourceOccurrences=1,
            contextEvidence="HS CODE: 60062200 - 60062200",
            sourceLineIds=("L00001",),
        ),
    )
    raw = "HS CODE: 60062200 - 60062200\n"
    workspace = RewriteWorkspace(
        original_text=raw,
        current_text=raw,
        surface_requirements=requirements,
    )

    apply_deterministic_prefills(workspace)

    assert workspace.current_text == "HS CODE: 48219046 - 84832073\n"
    assert [row.targetSurface for row in workspace.deterministic_prefills] == [
        "48219046",
        "84832073",
    ]


def test_short_hs_surface_is_not_bound_inside_a_longer_labeled_hs_code() -> None:
    source = {
        "documentPatch": {
            "cargoGroups": [
                {"groupId": "g1", "hsCodes": ["0810909000", "081090"]},
            ]
        }
    }
    target = {
        "documentPatch": {
            "cargoGroups": [
                {"groupId": "g1", "hsCodes": ["0302498113", "030289"]},
            ]
        }
    }
    raw = "HS CODE: 0810.90.9000\nHS CODE: 0810.90\n"

    requirements = surface_rendering_requirements(raw, source, target)

    assert [(row.sourceLineIds, row.sourceSurface, row.targetSurface) for row in requirements] == [
        (("L00001",), "0810.90.9000", "0302.49.8113"),
        (("L00002",), "0810.90", "0302.89"),
    ]
    workspace = RewriteWorkspace(
        original_text=raw,
        current_text=raw,
        surface_requirements=requirements,
    )
    apply_deterministic_prefills(workspace)
    assert workspace.current_text == "HS CODE: 0302.49.8113\nHS CODE: 0302.89\n"


def test_line_owned_date_prefill_handles_date_attached_to_heading_text() -> None:
    raw = "DATE OF ISSUE ON2024-03-16\n"
    source = {"documentPatch": {"issueDate": "2024-03-16"}}
    target = {"documentPatch": {"issueDate": "2024-05-13"}}

    requirements = surface_rendering_requirements(raw, source, target)

    assert requirements[0].sourceLineIds == ("L00001",)
    workspace = RewriteWorkspace(
        original_text=raw,
        current_text=raw,
        surface_requirements=requirements,
    )
    apply_deterministic_prefills(workspace)
    assert workspace.current_text == "DATE OF ISSUE ON2024-05-13\n"


def test_cargo_origin_prefill_changes_only_explicit_origin_grammar() -> None:
    raw = (
        "SHIPPER\nTAIWAN COMPONENTS LTD\n"
        "Exporter Registration Country: TAIWAN\n"
        "Made in Taiwan\nMade in Taiwan\n"
    )
    source = {"documentPatch": {"cargoGroups": [{"groupId": "g1", "origin": {"name": "Taiwan"}}]}}
    target = {
        "documentPatch": {
            "cargoGroups": [{"groupId": "g1", "origin": {"name": "Taiwan, Province of China"}}]
        }
    }

    requirements = surface_rendering_requirements(raw, source, target)
    origin = [row for row in requirements if row.kind == "cargo_origin"]

    assert len(origin) == 1
    assert origin[0].sourceLineIds == ("L00004", "L00005")
    workspace = RewriteWorkspace(
        original_text=raw,
        current_text=raw,
        surface_requirements=tuple(origin),
    )
    apply_deterministic_prefills(workspace)
    assert "TAIWAN COMPONENTS LTD" in workspace.current_text
    assert "Exporter Registration Country: TAIWAN" in workspace.current_text
    assert workspace.current_text.count("Made in Taiwan, Province of China") == 2


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
            "carrier_principal_identity",
            "MSC Mediterranean Shipping Company S.A.",
            "Helvetic Blueway Transport AG",
        ),
        (
            "carrier_header_identity",
            "MEDITERRANEAN SHIPPING COMPANY S.A.",
            "Helvetic Blueway Transport AG",
        ),
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


def test_legal_carrier_principals_are_prefilled_without_replacing_signing_agent() -> None:
    raw = (
        "CMA CGM\n"
        "CARRIER\n"
        "CMA CGM Société Anonyme\n"
        "SIGNED FOR THE CARRIER CMA CGM S.A.\n"
        "BY CMA CGM Deutschland GmbH Shipping Agency as agents for the carrier CMA CGM S. A.\n"
    )
    source = {"documentPatch": {"parties": {"carrier": {"name": "CMA CGM Société Anonyme"}}}}
    target = {"documentPatch": {"parties": {"carrier": {"name": "Helvetic Crest Navigation AG"}}}}
    requirements = surface_rendering_requirements(raw, source, target)
    principal = [row for row in requirements if row.kind == "carrier_principal_identity"]
    assert [(row.sourceLineIds, row.sourceSurface) for row in principal] == [
        (("L00004",), "CMA CGM S.A."),
        (("L00005",), "CMA CGM S. A."),
    ]
    workspace = RewriteWorkspace(
        original_text=raw,
        current_text=raw,
        current_target_label=target,
        surface_requirements=requirements,
    )

    apply_deterministic_prefills(workspace)

    lines = workspace.current_text.splitlines()
    assert lines[3] == "SIGNED FOR THE CARRIER Helvetic Crest Navigation AG"
    assert lines[4] == (
        "BY CMA CGM Deutschland GmbH Shipping Agency as agents for the carrier "
        "Helvetic Crest Navigation AG"
    )
    assert "CMA CGM Deutschland GmbH Shipping Agency" in lines[4]
    assert _required_surfaces_rendered(workspace.current_text, requirements)


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


def test_carrier_receipt_summary_is_not_partially_rendered_after_topology_projection() -> None:
    raw = "CARRIER'S RECEIPT\n2 X 40'\n"
    target = {
        "documentPatch": {
            "containers": [
                {
                    "containerNumber": "AAAA000001",
                    "sizeCategory": "FORTY_FOOT_HIGH_CUBE",
                    "typeCategory": "GENERAL_PURPOSE",
                },
                {"containerNumber": "BBBB000002"},
            ]
        }
    }

    requirements = surface_rendering_requirements(raw, target, target)

    assert not any(row.kind == "carrier_receipt_equipment_breakdown" for row in requirements)


def test_split_anonymous_equipment_summary_renders_every_target_semantic_pair() -> None:
    raw = "4X 40'CONTAINER SAID TO\nCONTAIN 1,886 CARTONS\n"
    source = {
        "documentPatch": {
            "containers": [
                {"containerNumber": f"AAAA00000{index}", "typeDescription": "40'CONTAINER"}
                for index in range(4)
            ]
        }
    }
    target = {
        "documentPatch": {
            "containers": [
                *[
                    {
                        "containerNumber": f"BBBB00000{index}",
                        "sizeCategory": "FORTY_FOOT_HIGH_CUBE",
                        "typeCategory": "GENERAL_PURPOSE",
                    }
                    for index in range(3)
                ],
                {
                    "containerNumber": "CCCC0000001",
                    "sizeCategory": "TWENTY_FOOT_STANDARD_HEIGHT",
                    "typeCategory": "GENERAL_PURPOSE",
                },
            ]
        }
    }

    requirements = surface_rendering_requirements(raw, source, target)
    aggregate = [row for row in requirements if row.kind == "aggregate_equipment_breakdown"]

    assert len(aggregate) == 4
    assert {row.targetPath for row in aggregate} == {
        f"documentPatch.containers[{index}].printedEquipmentSurface" for index in range(4)
    }
    assert {row.targetSurface for row in aggregate} == {
        "3X 40' HIGH CUBE GENERAL PURPOSE + "
        "1X 20' STANDARD HEIGHT GENERAL PURPOSE CONTAINERS SAID TO"
    }
    workspace = RewriteWorkspace(
        original_text=raw,
        current_text=raw,
        surface_requirements=tuple(aggregate),
    )
    prefills = apply_deterministic_prefills(workspace)
    assert workspace.current_text.splitlines()[0] == next(
        iter({row.targetSurface for row in aggregate})
    )
    assert {path for prefill in prefills for path in prefill.targetPaths} == {
        f"documentPatch.containers[{index}].printedEquipmentSurface" for index in range(4)
    }


def test_deterministic_prefills_do_not_cascade_across_same_line_targets() -> None:
    raw = "1 Package(s) of 01 UNIT NEW VEHICLE\n"
    requirements = (
        AnchoredScalarReplacementRequirement(
            targetPaths=("documentPatch.cargoPackages[0].typeCategory",),
            sourceLineIds=("L00001",),
            sourceSurface="Package",
            targetSurface="Unit",
            surfaceKind="package_noun",
        ),
        AnchoredScalarReplacementRequirement(
            targetPaths=("documentPatch.cargoPackages[0].typeCategory",),
            sourceLineIds=("L00001",),
            sourceSurface="UNIT",
            targetSurface="UNITS",
            surfaceKind="package_noun",
        ),
        AnchoredScalarReplacementRequirement(
            targetPaths=("documentPatch.cargoPackages[0].quantity",),
            sourceLineIds=("L00001",),
            sourceSurface="1",
            targetSurface="19",
            surfaceKind="measurement",
        ),
        AnchoredScalarReplacementRequirement(
            targetPaths=("documentPatch.cargoPackages[0].quantity",),
            sourceLineIds=("L00001",),
            sourceSurface="01",
            targetSurface="19",
            surfaceKind="measurement",
        ),
    )
    workspace = RewriteWorkspace(
        original_text=raw,
        current_text=raw,
        anchored_scalar_replacement_requirements=requirements,
    )

    apply_deterministic_prefills(workspace)

    assert workspace.current_text == "19 Unit(s) of 19 UNITS NEW VEHICLE\n"


def test_dangerous_goods_exact_class_and_un_preserve_printed_grammar() -> None:
    raw = "CLASS:6.1 UNDG NO:2078\nUN Number: 2078 - IMDG Class: 6.1 - PG: II\n"
    source = {
        "documentPatch": {
            "cargoGroups": [
                {
                    "groupId": "g1",
                    "dangerousGoods": [
                        {
                            "unNumber": "2078",
                            "hazardCategory": "TOXIC_AND_INFECTIOUS_SUBSTANCES",
                        }
                    ],
                }
            ]
        }
    }
    target = {
        "documentPatch": {
            "cargoGroups": [
                {
                    "groupId": "g1",
                    "dangerousGoods": [{"unNumber": "1013", "hazardCategory": "GASES"}],
                }
            ]
        }
    }
    payload = _linguistic_plan_with_hs_codes("281121").model_dump(mode="json")
    payload["cargoSeed"]["cargoGroups"][0]["dangerousGoods"] = [
        {
            "properShippingName": "CARBON DIOXIDE",
            "unNumber": "1013",
            "hazardCategory": "GASES",
            "exactHazardClass": "2.2",
            "subsidiaryHazardCategories": [],
            "packingGroupCategory": None,
            "flashpointCelsius": None,
        }
    ]
    plan = DocumentLinguisticPlan.model_validate_json(json.dumps(payload))

    requirements = surface_rendering_requirements(raw, source, target, plan)
    dangerous = [row for row in requirements if row.kind == "dangerous_goods_tuple"]

    assert {(row.sourceSurface, row.targetSurface) for row in dangerous} == {
        ("CLASS:6.1 UNDG NO:2078", "CLASS:2.2 UNDG NO:1013"),
        (
            "UN Number: 2078 - IMDG Class: 6.1 - PG: II",
            "UN Number: 1013 - IMDG Class: 2.2",
        ),
    }
    assert {row.targetPath for row in dangerous} == {
        "documentPatch.cargoGroups[0].dangerousGoods[0].unNumber",
        "documentPatch.cargoGroups[0].dangerousGoods[0].hazardCategory",
        "auxiliary.cargoGroups[0].dangerousGoods[0].packingGroup",
    }
    workspace = RewriteWorkspace(
        original_text=raw,
        current_text=raw,
        surface_requirements=tuple(dangerous),
    )
    apply_deterministic_prefills(workspace)
    assert workspace.current_text == ("CLASS:2.2 UNDG NO:1013\nUN Number: 1013 - IMDG Class: 2.2\n")


def test_sparse_dangerous_goods_target_rewrites_source_shipping_name_tail() -> None:
    raw = "UN 3077 ENVIRONMENTALLY HAZARDOUS SUBSTANCE, SOLID, N.O.S.\n"
    source = {
        "documentPatch": {
            "cargoGroups": [{"groupId": "g1", "dangerousGoods": [{"unNumber": "3077"}]}]
        }
    }
    target = {
        "documentPatch": {
            "cargoGroups": [{"groupId": "g1", "dangerousGoods": [{"unNumber": "0213"}]}]
        }
    }
    payload = _linguistic_plan_with_hs_codes("290930").model_dump(mode="json")
    payload["cargoSeed"]["cargoGroups"][0]["dangerousGoods"] = [
        {
            "properShippingName": "TRINITROANISOLE",
            "unNumber": "0213",
            "hazardCategory": "EXPLOSIVES",
            "exactHazardClass": "1.1D",
            "subsidiaryHazardCategories": [],
            "packingGroupCategory": None,
            "flashpointCelsius": None,
        }
    ]
    plan = DocumentLinguisticPlan.model_validate_json(json.dumps(payload))

    requirements = surface_rendering_requirements(raw, source, target, plan)

    assert len(requirements) == 1
    assert requirements[0].kind == "dangerous_goods_proper_shipping_name"
    workspace = RewriteWorkspace(
        original_text=raw,
        current_text=raw,
        surface_requirements=requirements,
    )
    apply_deterministic_prefills(workspace)
    assert workspace.current_text == "UN 0213 TRINITROANISOLE\n"


def test_dangerous_goods_prefill_defers_ocr_wrapped_occurrences_to_owned_cargo_span() -> None:
    raw = "TOLUENE DIISOCYANATE\nTOLUENE\nDIISOCYANATE)\n"
    requirements = (
        SurfaceRenderingRequirement(
            kind="dangerous_goods_proper_shipping_name",
            targetPath="auxiliary.cargoGroups[0].dangerousGoods[0].properShippingName",
            sourceSurface="TOLUENE DIISOCYANATE",
            targetSurface="TRINITROANISOLE",
            sourceOccurrences=2,
            contextEvidence="TOLUENE DIISOCYANATE",
            sourceLineIds=("L00001",),
        ),
        SurfaceRenderingRequirement(
            kind="dangerous_goods_proper_shipping_name",
            targetPath="documentPatch.cargoGroups[0].description",
            sourceSurface="TOLUENE DIISOCYANATE",
            targetSurface="TRINITROANISOLE",
            sourceOccurrences=2,
            contextEvidence="TOLUENE\nDIISOCYANATE)",
            sourceLineIds=("L00002", "L00003"),
        ),
    )
    workspace = RewriteWorkspace(
        original_text=raw,
        current_text=raw,
        surface_requirements=requirements,
    )

    prefills = apply_deterministic_prefills(workspace)

    assert workspace.current_text == "TRINITROANISOLE\nTOLUENE\nDIISOCYANATE)\n"
    assert len(prefills) == 1
    assert prefills[0].lineId == "L00001"


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


def test_ordered_forwarding_reference_accepts_printed_labels_between_target_atoms() -> None:
    leaf = ChangedLeaf(
        path="documentPatch.forwardingAndExportReferences[0]",
        sourcePresent=True,
        targetPresent=True,
        sourceValue="1123200938 27.11.2023",
        targetValue="785897 07.07.2023",
        evidenceClass="printed_fact",
        requiresTextEdit=True,
    )
    requirements = target_literal_requirements((leaf,))

    assert requirements[0].matchPolicy == "ordered_semantic_atoms"
    assert not _missing_target_literals("INVOICE NO. 785897 DATED: 07.07.2023\n", requirements)
    assert _missing_target_literals("B/L NO. 785897\nUNRELATED DATE: 07.07.2023\n", requirements)


def test_party_scalar_is_validated_by_role_block_not_global_literal() -> None:
    leaf = ChangedLeaf(
        path="documentPatch.parties.notifyParties[0].address",
        sourcePresent=True,
        targetPresent=True,
        sourceValue="OLD ROAD, OLD DISTRICT",
        targetValue="18 Al Mashtal Street, Industrial District",
        evidenceClass="printed_fact",
        requiresTextEdit=True,
    )

    assert target_literal_requirements((leaf,)) == ()


def test_date_surface_validation_is_bound_to_its_semantic_heading() -> None:
    requirement = SurfaceRenderingRequirement(
        kind="date",
        targetPath="documentPatch.issueDate",
        sourceSurface="MAY/15/2024",
        targetSurface="JUL/07/2023",
        sourceOccurrences=1,
        contextEvidence="PLACE AND DATE OF ISSUE MAY/15/2024",
    )
    output = "UNRELATED REFERENCE DATE MAY/15/2024\nPLACE AND DATE OF ISSUE JUL/07/2023\n"

    assert _required_surfaces_rendered(output, (requirement,))


def test_format_guard_treats_numeric_sign_as_part_of_changed_value() -> None:
    source = "Temperature: -18.0 C\n"
    target = "Temperature: 1.0 C\n"
    workspace = RewriteWorkspace(
        original_text=source,
        current_text=source,
        source_label={
            "documentPatch": {
                "containers": [{"temperatureSetpoint": {"unit": "celsius", "value": -18}}]
            }
        },
        current_target_label={
            "documentPatch": {
                "containers": [{"temperatureSetpoint": {"unit": "celsius", "value": 1}}]
            }
        },
    )

    assert _isolated_formatting_changes(workspace, target) == ()


def test_format_guard_accepts_hyphenation_inside_wrapped_changed_free_text() -> None:
    source = "STRETCH WRAPPED WITH EDGES PROTECTION\n"
    target = "STRETCH-WRAPPED WITH EDGE-PROTECTION\n"
    workspace = RewriteWorkspace(
        original_text=source,
        current_text=source,
        source_label={
            "documentPatch": {
                "cargoGroups": [
                    {
                        "additionalInformation": [
                            "STRETCH WRAPPED WITH EDGES PROTECTION FOAM AND CARDBOARD"
                        ]
                    }
                ]
            }
        },
        current_target_label={
            "documentPatch": {
                "cargoGroups": [
                    {
                        "additionalInformation": [
                            "STRETCH-WRAPPED WITH EDGE-PROTECTION FOAM AND CARDBOARD"
                        ]
                    }
                ]
            }
        },
    )

    assert _isolated_formatting_changes(workspace, target) == ()


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

    target = {"documentPatch": {"containers": [{}, {}]}}
    hints = source_semantic_role_hints(raw, target)

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
        source_role_hints=source_semantic_role_hints(
            raw, {"documentPatch": {"containers": [{}, {}]}}
        ),
    )

    with pytest.raises(ValueError, match="flattened-OCR role surface"):
        apply_line_range_replacements(
            workspace,
            (_edit(raw, 4, 4, "LOT RKM-27"),),
        )


def test_flattened_equipment_is_preserved_when_target_has_only_packages() -> None:
    raw = (
        "--- PAGE 1 ---\nContainer No. and Seal No.\nMarks & Nos.\n2\n\n"
        "Quantity and Kind of Packages\n\n/40' HC Containers Said to Contain\n"
    )
    target = {
        "documentPatch": {"cargoPackages": [{"groupId": "g1", "packageId": "p1", "quantity": 9}]}
    }
    workspace = RewriteWorkspace(
        original_text=raw,
        current_text=raw,
        source_role_hints=source_semantic_role_hints(raw, target),
    )

    applied = apply_deterministic_prefills(workspace)

    assert workspace.current_text.splitlines()[3] == "2"
    assert workspace.current_text.splitlines()[7] == "/40' HC Containers Said to Contain"
    assert [hint.role for hint in workspace.source_role_hints] == [
        "anonymous_equipment_count",
        "anonymous_equipment_surface",
    ]
    assert all(
        hint.requiredOutputSurface == hint.sourceSurface for hint in workspace.source_role_hints
    )
    assert applied == ()


def test_one_token_carrier_header_brand_is_projected_without_alias_table() -> None:
    raw = "--- PAGE 1 ---\nARKAS\n\nShipper\nSOURCE SHIPPER\n"
    source = {
        "documentPatch": {"parties": {"carrier": {"name": "Arkas Container Transport, S.A."}}}
    }
    target = {"documentPatch": {"parties": {"carrier": {"name": "Marevanta Ocean Lines, S.A."}}}}

    requirements = surface_rendering_requirements(raw, source, target)

    header = [row for row in requirements if row.kind == "carrier_header_identity"]
    assert len(header) == 1
    assert header[0].sourceSurface == "ARKAS"
    assert header[0].targetSurface == "MAREVANTA"


def test_wrapped_compound_carrier_is_not_misclassified_as_a_header_alias() -> None:
    raw = (
        "--- PAGE 1 ---\nYANG MING\n\nShipper\nSOURCE SHIPPER\n\nBy\n"
        "As agent for the Carrier and Service Provider Yang Ming (Singapore)\n"
        "Pte. Ltd. trading as Yang Ming\n"
    )
    source = {
        "documentPatch": {
            "parties": {"carrier": {"name": "Yang Ming (Singapore) Pte. Ltd. trading as Yang Ming"}}
        }
    }
    target = {
        "documentPatch": {"parties": {"carrier": {"name": "Asterwave Maritime Carriage Ltd."}}}
    }

    requirements = surface_rendering_requirements(raw, source, target)

    header = [row for row in requirements if row.kind == "carrier_header_identity"]
    assert [(row.sourceSurface, row.targetSurface) for row in header] == [
        ("YANG MING", "ASTERWAVE MARITIME")
    ]


def test_carrier_header_is_rendered_as_an_exact_line_without_global_substitution() -> None:
    raw = "--- PAGE 1 ---\nARKAS\n\nCARRIER\nARKAS CONTAINER TRANSPORT S.A.\n"
    requirement = SurfaceRenderingRequirement(
        kind="carrier_header_identity",
        targetPath="documentPatch.parties.carrier.name",
        sourceSurface="ARKAS",
        targetSurface="MAREVANTA",
        sourceOccurrences=1,
        contextEvidence="--- PAGE 1 ---\nARKAS\n\nCARRIER",
    )
    workspace = RewriteWorkspace(
        original_text=raw,
        current_text=raw,
        surface_requirements=(requirement,),
    )

    applied = apply_deterministic_prefills(workspace)

    assert workspace.current_text == raw.replace("\nARKAS\n", "\nMAREVANTA\n", 1)
    assert "ARKAS CONTAINER TRANSPORT S.A." in workspace.current_text
    assert [(row.lineId, row.targetSurface) for row in applied] == [("L00002", "MAREVANTA")]


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


def test_formatted_anchored_identifier_does_not_require_a_second_canonical_surface() -> None:
    path = "documentPatch.containers[0].containerNumber"
    leaf = ChangedLeaf(
        path=path,
        sourcePresent=True,
        targetPresent=True,
        sourceValue="HLBU9296520",
        targetValue="HLBU7872400",
        evidenceClass="printed_fact",
        requiresTextEdit=True,
    )
    anchored = AnchoredScalarReplacementRequirement(
        targetPaths=(path,),
        sourceLineIds=("L00002",),
        sourceSurface="HLBU 9296520",
        targetSurface="HLBU 7872400",
    )

    assert target_literal_requirements((leaf,), (anchored,)) == ()


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


def test_target_integrity_preserves_valid_contact_domain_when_party_name_is_absent() -> None:
    config = load_synthesis_raw_text_rewrite_cycle_probe_config(
        Path("configs/synthesis/mpci_bl_raw_text_atomic10_luna_high.yaml")
    )
    target = {
        "schemaVersion": "5.0.0-experimental",
        "documentPatch": {
            "parties": {
                "notifyParties": [
                    {
                        "country": "Egypt",
                        "contactDetails": {
                            "emailAddresses": ["operations@terminal-services.com"],
                            "websiteUrls": ["https://terminal-services.com/contact"],
                        },
                    }
                ]
            }
        },
    }

    effective, receipt = prepare_target_integrity(
        target, target, config, _target_integrity_resources()
    )

    assert effective == target
    assert receipt["semantic_changes"] == []


def test_unprinted_target_equipment_is_explicitly_projected_out_of_template() -> None:
    source = "CONTAINER: MCLU5086082\nSEAL: 12805\n"
    source_label = {
        "schemaVersion": "3.0.0-experimental",
        "documentPatch": {"containers": [{"containerNumber": "MCLU5086082"}]},
    }
    target = {
        "schemaVersion": "5.0.0-experimental",
        "documentPatch": {
            "containers": [
                {
                    "containerNumber": "MCLU4686729",
                    "sizeCategory": "TWENTY_FOOT_STANDARD_HEIGHT",
                    "typeCategory": "GENERAL_PURPOSE",
                }
            ]
        },
    }

    projected, changes = _project_unrenderable_equipment_to_template(
        source, source_label, target
    )

    assert "sizeCategory" not in projected["documentPatch"]["containers"][0]
    assert "typeCategory" not in projected["documentPatch"]["containers"][0]
    assert [row.reason for row in changes] == ["unprinted_equipment_topology"]
    assert target["documentPatch"]["containers"][0]["sizeCategory"] == (
        "TWENTY_FOOT_STANDARD_HEIGHT"
    )


def test_unprinted_temperature_bearing_equipment_fails_instead_of_downgrading() -> None:
    source = "CONTAINER: MCLU5086082\nSEAL: 12805\n"
    source_label = {
        "schemaVersion": "3.0.0-experimental",
        "documentPatch": {"containers": [{"containerNumber": "MCLU5086082"}]},
    }
    target = {
        "schemaVersion": "5.0.0-experimental",
        "documentPatch": {
            "containers": [
                {
                    "containerNumber": "MCLU4686729",
                    "sizeCategory": "FORTY_FOOT_HIGH_CUBE",
                    "typeCategory": "REFRIGERATED",
                    "temperatureSetpoint": {"value": -18.0, "unit": "celsius"},
                }
            ]
        },
    }

    with pytest.raises(ValueError, match="temperature-bearing target equipment"):
        _project_unrenderable_equipment_to_template(source, source_label, target)


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
    prefill_workspace = RewriteWorkspace(
        original_text=source,
        current_text=source,
        operational_flavor_requirements=requirements,
    )
    prefills = apply_deterministic_prefills(prefill_workspace)
    assert "G.W 14857.500\nCBM 27.8880" in prefill_workspace.current_text
    assert [(row.lineId, row.sourceSurface, row.targetSurface) for row in prefills] == [
        ("L00003", "13519.000", "14857.500"),
        ("L00004", "70.1530", "27.8880"),
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


def test_operational_flavor_projects_membership_only_printed_package_rows() -> None:
    config = load_synthesis_raw_text_rewrite_cycle_probe_config(
        Path("configs/synthesis/mpci_bl_raw_text_atomic10_luna_high.yaml")
    )
    source = "HASU4803711 40 DRY 9'6 36 PALLET\nCAAU7756961 40 DRY 9'6 52 PALLET\n"
    source_label = {
        "documentPatch": {
            "containers": [
                {"containerNumber": "HASU4803711", "typeDescription": "40 DRY 9'6"},
                {"containerNumber": "CAAU7756961", "typeDescription": "40 DRY 9'6"},
            ],
            "cargoPackages": [
                {"groupId": "g1", "packageId": "p1", "quantity": 1556},
                {"groupId": "g1", "packageId": "p2", "quantity": 31064},
            ],
            "cargoAllocationGroups": [
                {
                    "groupId": "g1",
                    "packageIds": [],
                    "coverage": "container_membership_only",
                    "allocations": [
                        {"containerNumber": "HASU4803711"},
                        {"containerNumber": "CAAU7756961"},
                    ],
                }
            ],
        }
    }
    target_label = {
        "documentPatch": {
            "containers": [
                {"containerNumber": "HASU7540590"},
                {"containerNumber": "CAAU9729374"},
            ],
            "cargoPackages": [
                {"groupId": "g1", "packageId": "p1", "quantity": 1562},
                {"groupId": "g1", "packageId": "p2", "quantity": 31184},
            ],
            "cargoAllocationGroups": [
                {
                    "groupId": "g1",
                    "packageIds": [],
                    "coverage": "container_membership_only",
                    "allocations": [
                        {"containerNumber": "HASU7540590"},
                        {"containerNumber": "CAAU9729374"},
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
        scenario_id="scenario-membership",
        profiles=(),
        limits=capacity_limits(config.target_integrity.transport_capacity),
    )

    assert [
        (row.targetContainerNumber, row.targetValueSurface, row.samplingMethod)
        for row in requirements
    ] == [
        ("HASU7540590", "639", "target_membership_package_projection_v1"),
        ("CAAU9729374", "923", "target_membership_package_projection_v1"),
    ]


def test_operational_discovery_does_not_treat_tare_net_vent_or_zero_as_cargo_measures() -> None:
    source = (
        "CARU5733550\n"
        "Tare Weight: 3,690 kgs.\n"
        "NET WEIGHT: 17472 KGS\n"
        "VENT.: 20.0 CBM/H\n"
        "0000 KGS\n"
        "Gross Cargo Weight: 1,996.000 kgs.\n"
        "Measurement: 15.582 cu. m.\n"
    )

    occurrences = _container_measurement_occurrences(
        source,
        ({"containerNumber": "CARU5733550", "typeDescription": "40' DRY VAN"},),
    )

    assert [(row[0], row[1], row[2]) for row in occurrences[0]] == [
        ("tare_weight_kg", 2, "3,690"),
        ("gross_weight_kg", 6, "1,996.000"),
        ("volume_m3", 7, "15.582"),
    ]
    assert _operational_measurement_match("Tare Weight: 3,690 kgs.", "gross_weight_kg") is None
    assert _operational_measurement_match("NET WEIGHT: 17472 KGS", "gross_weight_kg") is None
    assert _operational_measurement_match("VENT.: 20.0 CBM/H", "volume_m3") is None


def test_membership_only_without_container_local_package_rows_needs_no_projection() -> None:
    source_containers = ({"containerNumber": "MSMU7618640", "typeDescription": "40HC"},)
    target_containers = (
        {
            "containerNumber": "MSMU4058481",
            "sizeCategory": "TWENTY_FOOT_STANDARD_HEIGHT",
            "typeCategory": "GENERAL_PURPOSE",
        },
    )
    source_label = {
        "documentPatch": {
            "containers": list(source_containers),
            "cargoPackages": [
                {"groupId": "g1", "packageId": "p1", "quantity": 34}
            ],
            "cargoAllocationGroups": [
                {
                    "groupId": "g1",
                    "packageIds": [],
                    "coverage": "container_membership_only",
                    "allocations": [{"containerNumber": "MSMU7618640"}],
                }
            ],
        }
    }
    target_label = {
        "documentPatch": {
            "containers": list(target_containers),
            "cargoPackages": [
                {"groupId": "g1", "packageId": "p1", "quantity": 14}
            ],
            "cargoAllocationGroups": [
                {
                    "groupId": "g1",
                    "packageIds": [],
                    "coverage": "container_membership_only",
                    "allocations": [{"containerNumber": "MSMU4058481"}],
                }
            ],
        }
    }

    assert (
        _membership_package_allocations(
            source_label,
            target_label,
            source_containers,
            target_containers,
            {0: (("tare_weight_kg", 2, "3,840", "evidence", "labeled_measurement"),)},
        )
        == {}
    )


def test_changed_equipment_is_bound_to_its_own_container_row() -> None:
    source = (
        "CAXU9173485 40' Dry Hi-Cube\nDRYU9087067 40' Dry Hi-Cube\nECMU9427372 40' Dry Hi-Cube\n"
    )
    source_label = {
        "documentPatch": {
            "containers": [
                {
                    "containerNumber": number,
                    "typeDescription": "40' Dry Hi-Cube",
                }
                for number in ("CAXU9173485", "DRYU9087067", "ECMU9427372")
            ]
        }
    }
    target_label = {
        "documentPatch": {
            "containers": [
                {
                    "containerNumber": "CAXU7668171",
                    "sizeCategory": "FORTY_FOOT_HIGH_CUBE",
                    "typeCategory": "GENERAL_PURPOSE",
                },
                {
                    "containerNumber": "DRYU7101551",
                    "sizeCategory": "TWENTY_FOOT_STANDARD_HEIGHT",
                    "typeCategory": "GENERAL_PURPOSE",
                },
                {
                    "containerNumber": "ECMU7778845",
                    "sizeCategory": "FORTY_FOOT_HIGH_CUBE",
                    "typeCategory": "GENERAL_PURPOSE",
                },
            ]
        }
    }

    requirements = container_equipment_replacement_requirements(source, source_label, target_label)

    assert len(requirements) == 1
    assert requirements[0].sourceLineIds == ("L00002",)
    assert requirements[0].sourceSurface == "40' Dry Hi-Cube"
    assert requirements[0].targetSurface == "20' Standard Height General Purpose"
    workspace = RewriteWorkspace(
        original_text=source,
        current_text=source,
        anchored_scalar_replacement_requirements=requirements,
    )
    apply_deterministic_prefills(workspace)
    assert workspace.current_text.splitlines() == [
        "CAXU9173485 40' Dry Hi-Cube",
        "DRYU9087067 20' Standard Height General Purpose",
        "ECMU9427372 40' Dry Hi-Cube",
    ]


def test_changed_equipment_updates_a_container_local_cargo_row_alias() -> None:
    source = (
        "MSMU7790583\n"
        "40' HIGH CUBE\n"
        "20 Package(s) of 1x40 'HQ CONTAINING:\n"
        "20 x PACKAGE OF USED SPARE PARTS\n"
        "MSMU8832908\n"
        "40' HIGH CUBE\n"
        "20 Package(s) of 1x40 'HQ CONTAINING:\n"
        "20 x PACKAGE OF USED SPARE PARTS\n"
    )
    source_label = {
        "documentPatch": {
            "containers": [
                {"containerNumber": "MSMU7790583", "typeDescription": "40' HIGH CUBE"},
                {"containerNumber": "MSMU8832908", "typeDescription": "40' HIGH CUBE"},
            ],
            "cargoPackages": [
                {
                    "groupId": "g1",
                    "packageId": "p1",
                    "quantity": 40,
                    "typeCategory": "PACKAGE_PACKAGE",
                }
            ],
            "cargoAllocationGroups": [
                {
                    "groupId": "g1",
                    "packageIds": ["p1"],
                    "coverage": "single_package_level",
                    "allocations": [
                        {"containerNumber": "MSMU7790583", "packageQuantity": 20},
                        {"containerNumber": "MSMU8832908", "packageQuantity": 20},
                    ],
                }
            ],
        }
    }
    target_label = {
        "documentPatch": {
            "containers": [
                {
                    "containerNumber": "MSMU4885005",
                    "sizeCategory": "FORTY_FOOT_HIGH_CUBE",
                    "typeCategory": "GENERAL_PURPOSE",
                },
                {
                    "containerNumber": "MSMU9420443",
                    "sizeCategory": "TWENTY_FOOT_STANDARD_HEIGHT",
                    "typeCategory": "GENERAL_PURPOSE",
                },
            ],
            "cargoPackages": [
                {
                    "groupId": "g1",
                    "packageId": "p1",
                    "quantity": 135,
                    "typeCategory": "PACKAGE_PIECE",
                }
            ],
            "cargoAllocationGroups": [
                {
                    "groupId": "g1",
                    "packageIds": ["p1"],
                    "coverage": "single_package_level",
                    "allocations": [
                        {"containerNumber": "MSMU4885005", "packageQuantity": 68},
                        {"containerNumber": "MSMU9420443", "packageQuantity": 67},
                    ],
                }
            ],
        }
    }

    requirements = container_equipment_replacement_requirements(
        source, source_label, target_label
    )

    second = [
        row
        for row in requirements
        if row.targetPaths == ("documentPatch.containers[1].printedEquipmentSurface",)
    ]
    assert {(row.sourceLineIds, row.sourceSurface) for row in second} == {
        (("L00006",), "40' HIGH CUBE"),
        (("L00007",), "1x40 'HQ"),
    }
    workspace = RewriteWorkspace(
        original_text=source,
        current_text=source,
        anchored_scalar_replacement_requirements=requirements,
    )
    apply_deterministic_prefills(workspace)
    assert workspace.current_text.splitlines()[5] == "20' STANDARD HEIGHT GENERAL PURPOSE"
    assert workspace.current_text.splitlines()[6] == (
        "20 Package(s) of 1x20' STANDARD HEIGHT GENERAL PURPOSE CONTAINING:"
    )


def test_single_container_equipment_can_bind_to_a_separate_type_row() -> None:
    source = "Cntr/Chassis Nr.\nMCLU 510204.9\n\nType and size\n1 X 40 HC\n"
    source_label = {
        "documentPatch": {
            "containers": [{"containerNumber": "MCLU5102049", "typeDescription": "1 X 40 HC"}]
        }
    }
    target_label = {
        "documentPatch": {
            "containers": [
                {
                    "containerNumber": "MCLU2116658",
                    "sizeCategory": "TWENTY_FOOT_STANDARD_HEIGHT",
                    "typeCategory": "GENERAL_PURPOSE",
                }
            ]
        }
    }

    requirements = container_equipment_replacement_requirements(source, source_label, target_label)

    assert [(row.sourceLineIds, row.targetSurface) for row in requirements] == [
        (("L00005",), "20' STANDARD HEIGHT GENERAL PURPOSE")
    ]


def test_multi_container_equipment_binds_to_unique_adjacent_type_rows() -> None:
    source = "BEAU4253489\nSEAL A1\n1 x 40HC\n\nMEDU6168416\nSEAL B2\n20GP\n"
    source_label = {
        "documentPatch": {
            "containers": [
                {"containerNumber": "BEAU4253489", "typeDescription": "1 x 40HC"},
                {"containerNumber": "MEDU6168416", "typeDescription": "20GP"},
            ]
        }
    }
    target_label = {
        "documentPatch": {
            "containers": [
                {
                    "containerNumber": "BEAU9253725",
                    "sizeCategory": "TWENTY_FOOT_STANDARD_HEIGHT",
                    "typeCategory": "GENERAL_PURPOSE",
                },
                {
                    "containerNumber": "MEDU9629757",
                    "sizeCategory": "FORTY_FOOT_HIGH_CUBE",
                    "typeCategory": "GENERAL_PURPOSE",
                },
            ]
        }
    }

    requirements = container_equipment_replacement_requirements(source, source_label, target_label)

    assert [(row.sourceLineIds, row.sourceSurface, row.targetSurface) for row in requirements] == [
        (("L00003",), "1 x 40HC", "20' STANDARD HEIGHT GENERAL PURPOSE"),
        (("L00007",), "20GP", "40' HIGH CUBE GENERAL PURPOSE"),
    ]
    workspace = RewriteWorkspace(
        original_text=source,
        current_text=source,
        anchored_scalar_replacement_requirements=requirements,
    )
    apply_deterministic_prefills(workspace)
    assert workspace.current_text.splitlines()[2] == "20' STANDARD HEIGHT GENERAL PURPOSE"
    assert workspace.current_text.splitlines()[6] == "40' HIGH CUBE GENERAL PURPOSE"


def test_multi_container_equipment_allows_decorative_blank_inside_owned_record() -> None:
    source = "BEAU4253489\nSEAL A1\n\nQTY 1 40' HC\n\nMEDU6168416\nSEAL B2\n\nQTY 1 40' HC\n"
    source_label = {
        "documentPatch": {
            "containers": [
                {"containerNumber": "BEAU4253489", "typeDescription": "40' HC"},
                {"containerNumber": "MEDU6168416", "typeDescription": "40' HC"},
            ]
        }
    }
    target_label = {
        "documentPatch": {
            "containers": [
                {
                    "containerNumber": "BEAU9253725",
                    "sizeCategory": "TWENTY_FOOT_STANDARD_HEIGHT",
                    "typeCategory": "GENERAL_PURPOSE",
                },
                {
                    "containerNumber": "MEDU9629757",
                    "sizeCategory": "FORTY_FOOT_HIGH_CUBE",
                    "typeCategory": "GENERAL_PURPOSE",
                },
            ]
        }
    }

    requirements = container_equipment_replacement_requirements(source, source_label, target_label)

    assert [(row.sourceLineIds, row.sourceSurface) for row in requirements] == [
        (("L00004",), "40' HC")
    ]


def test_unresolved_partial_equipment_surface_is_replaced_on_its_container_row() -> None:
    source = "TGHU6144091|40|0491515\n"
    source_label = {
        "documentPatch": {
            "containers": [{"containerNumber": "TGHU6144091", "typeDescription": "40"}]
        }
    }
    target_label = {
        "documentPatch": {
            "containers": [
                {
                    "containerNumber": "TGHU9144098",
                    "sizeCategory": "FORTY_FOOT_HIGH_CUBE",
                    "typeCategory": "GENERAL_PURPOSE",
                }
            ]
        }
    }

    requirements = container_equipment_replacement_requirements(source, source_label, target_label)

    assert [(row.sourceLineIds, row.sourceSurface, row.targetSurface) for row in requirements] == [
        (("L00001",), "40", "40' HIGH CUBE GENERAL PURPOSE")
    ]


def test_single_container_recovers_one_unlabeled_printed_equipment_alias() -> None:
    source = "CAIU4204766 / 04070\n52 PACKAGES /FCL / FCL/40HQ/7060.000KGS/40.000M3\n"
    source_label = {
        "documentPatch": {"containers": [{"containerNumber": "CAIU4204766"}]}
    }
    target_label = {
        "documentPatch": {
            "containers": [
                {
                    "containerNumber": "CAIU6730332",
                    "sizeCategory": "TWENTY_FOOT_STANDARD_HEIGHT",
                    "typeCategory": "GENERAL_PURPOSE",
                }
            ]
        }
    }

    requirements = container_equipment_replacement_requirements(source, source_label, target_label)

    assert [(row.sourceLineIds, row.sourceSurface, row.targetSurface) for row in requirements] == [
        (("L00002",), "40HQ", "20' STANDARD HEIGHT GENERAL PURPOSE")
    ]


def test_positional_equipment_rows_bind_when_occurrence_count_matches_container_subset() -> None:
    source = (
        "FSCU5906804 / 2439296\nKKFU6721390 / A147072\nONEU9083430 / 2439382\n"
        "/FCL / FCL/40RQ/19886.321KGS/57.000M3\n"
        "/FCL / FCL/40RQ/19881.604KGS/57.000M3\n"
        "/FCL / FCL/40RQ/19886.321KGS/57.000M3\n"
    )
    source_label = {
        "documentPatch": {
            "containers": [
                {"containerNumber": "FSCU5906804", "typeDescription": "40RQ"},
                {"containerNumber": "KKFU6721390", "typeDescription": "40RQ"},
                {"containerNumber": "ONEU9083430", "typeDescription": "40RQ"},
            ]
        }
    }
    target_label = {
        "documentPatch": {
            "containers": [
                {
                    "containerNumber": f"ONEU90000{index}0",
                    "sizeCategory": "FORTY_FOOT_HIGH_CUBE",
                    "typeCategory": "GENERAL_PURPOSE",
                }
                for index in range(3)
            ]
        }
    }

    requirements = container_equipment_replacement_requirements(source, source_label, target_label)

    assert [row.sourceLineIds for row in requirements] == [
        ("L00004",),
        ("L00005",),
        ("L00006",),
    ]


def test_container_package_nouns_follow_allocation_owned_target_category() -> None:
    source = (
        "PONU1944248 40 DRY 8'6 27 PALLETS 5940.000 KGS\n"
        "MRKU0434926 40 DRY 8'6 17 PALLETS 3090.000 KGS\n"
        "--- PAGE 2 ---\n"
        "PONU1944248 40 DRY 8'6 27 PALLETS 5940.000 KGS\n"
        "MRKU0434926 40 DRY 8'6 17 PALLETS 3090.000 KGS\n"
    )
    source_numbers = ("PONU1944248", "MRKU0434926")
    target_numbers = ("PONU9987518", "MRKU3121421")
    source_label = {
        "documentPatch": {
            "containers": [
                {"containerNumber": number, "typeDescription": "40 DRY 8'6"}
                for number in source_numbers
            ],
            "cargoPackages": [
                {
                    "groupId": "g1",
                    "packageId": "p1",
                    "quantity": 44,
                    "typeCategory": "PACKAGE_PALLET",
                }
            ],
            "cargoAllocationGroups": [
                {
                    "groupId": "g1",
                    "packageIds": ["p1"],
                    "coverage": "single_package_level",
                    "allocations": [
                        {"containerNumber": source_numbers[0], "packageQuantity": 27},
                        {"containerNumber": source_numbers[1], "packageQuantity": 17},
                    ],
                }
            ],
        }
    }
    target_label = {
        "documentPatch": {
            "containers": [
                {
                    "containerNumber": number,
                    "sizeCategory": "FORTY_FOOT_STANDARD_HEIGHT",
                    "typeCategory": "GENERAL_PURPOSE",
                }
                for number in target_numbers
            ],
            "cargoPackages": [
                {
                    "groupId": "g1",
                    "packageId": "p1",
                    "quantity": 21,
                    "typeCategory": "PACKAGE_PACKAGE",
                }
            ],
            "cargoAllocationGroups": [
                {
                    "groupId": "g1",
                    "packageIds": ["p1"],
                    "coverage": "single_package_level",
                    "allocations": [
                        {"containerNumber": target_numbers[0], "packageQuantity": 13},
                        {"containerNumber": target_numbers[1], "packageQuantity": 8},
                    ],
                }
            ],
        }
    }

    requirements = container_package_type_replacement_requirements(
        source,
        source_label,
        target_label,
        _target_integrity_resources().packages,
    )

    assert len(requirements) == 4
    assert {row.sourceLineIds for row in requirements} == {
        ("L00001",),
        ("L00002",),
        ("L00004",),
        ("L00005",),
    }
    assert {row.targetSurface for row in requirements} == {"PACKAGES"}
    workspace = RewriteWorkspace(
        original_text=source,
        current_text=source,
        anchored_scalar_replacement_requirements=requirements,
    )
    apply_deterministic_prefills(workspace)
    assert "PALLET" not in workspace.current_text
    assert workspace.current_text.count("PACKAGES") == 4


def test_package_noun_renderer_does_not_upgrade_membership_only_allocations() -> None:
    source = "TGBU1028944 20 DRY 90 PIECES 1900.000 KGS\n"
    source_label = {
        "documentPatch": {
            "containers": [{"containerNumber": "TGBU1028944", "typeDescription": "20 DRY"}],
            "cargoPackages": [
                {
                    "groupId": "g1",
                    "packageId": "p1",
                    "quantity": 90,
                    "typeCategory": "PACKAGE_PIECE",
                }
            ],
            "cargoAllocationGroups": [
                {
                    "groupId": "g1",
                    "packageIds": [],
                    "coverage": "container_membership_only",
                    "allocations": [{"containerNumber": "TGBU1028944"}],
                }
            ],
        }
    }
    target_label = {
        "documentPatch": {
            "containers": [
                {
                    "containerNumber": "TGBU7153847",
                    "sizeCategory": "TWENTY_FOOT_STANDARD_HEIGHT",
                    "typeCategory": "GENERAL_PURPOSE",
                }
            ],
            "cargoPackages": [
                {
                    "groupId": "g1",
                    "packageId": "p1",
                    "quantity": 90,
                    "typeCategory": "PACKAGE_PIECE",
                }
            ],
            "cargoAllocationGroups": [
                {
                    "groupId": "g1",
                    "packageIds": [],
                    "coverage": "container_membership_only",
                    "allocations": [{"containerNumber": "TGBU7153847"}],
                }
            ],
        }
    }

    assert (
        container_package_type_replacement_requirements(
            source,
            source_label,
            target_label,
            _target_integrity_resources().packages,
        )
        == ()
    )


def test_package_noun_renderer_preserves_parenthesized_plural_envelope() -> None:
    source = "TGBU1028944 20 DRY 14 PACKAGE(S) 1900.000 KGS\n"
    source_label = {
        "documentPatch": {
            "containers": [{"containerNumber": "TGBU1028944", "typeDescription": "20 DRY"}],
            "cargoPackages": [
                {
                    "groupId": "g1",
                    "packageId": "p1",
                    "quantity": 14,
                    "typeCategory": "PACKAGE_PACKAGE",
                }
            ],
            "cargoAllocationGroups": [
                {
                    "groupId": "g1",
                    "packageIds": ["p1"],
                    "coverage": "single_package_level",
                    "allocations": [{"containerNumber": "TGBU1028944", "packageQuantity": 14}],
                }
            ],
        }
    }
    target_label = {
        "documentPatch": {
            "containers": [
                {
                    "containerNumber": "TGBU7153847",
                    "sizeCategory": "TWENTY_FOOT_STANDARD_HEIGHT",
                    "typeCategory": "GENERAL_PURPOSE",
                }
            ],
            "cargoPackages": [
                {
                    "groupId": "g1",
                    "packageId": "p1",
                    "quantity": 14,
                    "typeCategory": "PACKAGE_BUNDLE",
                }
            ],
            "cargoAllocationGroups": [
                {
                    "groupId": "g1",
                    "packageIds": ["p1"],
                    "coverage": "single_package_level",
                    "allocations": [{"containerNumber": "TGBU7153847", "packageQuantity": 14}],
                }
            ],
        }
    }

    requirements = container_package_type_replacement_requirements(
        source,
        source_label,
        target_label,
        _target_integrity_resources().packages,
    )
    workspace = RewriteWorkspace(
        original_text=source,
        current_text=source,
        anchored_scalar_replacement_requirements=requirements,
    )
    apply_deterministic_prefills(workspace)

    assert "14 BUNDLE(S)" in workspace.current_text
    assert "BUNDLES(S)" not in workspace.current_text


def test_cargo_package_noun_renderer_covers_totals_and_repeated_allocation_rows() -> None:
    source = (
        "TOTAL 1,878 CARTONS\n"
        "(TOTAL ONE THOUSAND EIGHT HUNDRED\n"
        "SEVENTY EIGHT CARTONS ONLY)\n"
        "1X472 CARTONS\n"
        "SOURCE GOODS\n"
        "1X472 CARTONS\n"
        "SOURCE GOODS\n"
        "1X467 CARTONS\n"
        "SOURCE GOODS\n"
        "1X467 CARTONS\n"
    )
    source_label = {
        "documentPatch": {
            "cargoGroups": [{"groupId": "g1", "description": "SOURCE GOODS"}],
            "cargoPackages": [
                {
                    "groupId": "g1",
                    "packageId": "p1",
                    "quantity": 1878,
                    "typeCategory": "PACKAGE_CARTON",
                }
            ],
            "cargoAllocationGroups": [
                {
                    "groupId": "g1",
                    "packageIds": ["p1"],
                    "coverage": "single_package_level",
                    "allocations": [
                        {"containerNumber": "CMAU7221965", "packageQuantity": 467},
                        {"containerNumber": "TCNU2842580", "packageQuantity": 472},
                        {"containerNumber": "CAAU6051350", "packageQuantity": 467},
                        {"containerNumber": "CAIU8394366", "packageQuantity": 472},
                    ],
                }
            ],
        }
    }
    target_label = {
        "documentPatch": {
            "cargoGroups": [{"groupId": "g1", "description": "TARGET GOODS"}],
            "cargoPackages": [
                {
                    "groupId": "g1",
                    "packageId": "p1",
                    "quantity": 991,
                    "typeCategory": "PACKAGE_PACKAGE",
                }
            ],
            "cargoAllocationGroups": [
                {
                    "groupId": "g1",
                    "packageIds": ["p1"],
                    "coverage": "single_package_level",
                    "allocations": [
                        {"containerNumber": "CMAU9109297", "packageQuantity": 247},
                        {"containerNumber": "TCNU3213134", "packageQuantity": 249},
                        {"containerNumber": "CAAU6283355", "packageQuantity": 246},
                        {"containerNumber": "CAIU5101377", "packageQuantity": 249},
                    ],
                }
            ],
        }
    }
    cargo = (
        CargoFlavorRewriteRequirement(
            requirementId="cargo-group-1-span-1",
            targetPath="documentPatch.cargoGroups[0].description",
            sourceLineIds=(
                "L00004",
                "L00005",
                "L00006",
                "L00007",
                "L00008",
                "L00009",
                "L00010",
            ),
            sourceSurfaces=(
                "1X472 CARTONS",
                "SOURCE GOODS",
                "1X472 CARTONS",
                "SOURCE GOODS",
                "1X467 CARTONS",
                "SOURCE GOODS",
                "1X467 CARTONS",
            ),
            targetDescription="TARGET GOODS",
        ),
    )

    requirements = cargo_package_type_replacement_requirements(
        source,
        source_label,
        target_label,
        _target_integrity_resources().packages,
        cargo,
    )
    assert {line for row in requirements for line in row.sourceLineIds} == {
        "L00001",
        "L00003",
        "L00004",
        "L00006",
        "L00008",
        "L00010",
    }
    workspace = RewriteWorkspace(
        original_text=source,
        current_text=source,
        anchored_scalar_replacement_requirements=requirements,
    )
    apply_deterministic_prefills(workspace)

    assert "CARTON" not in workspace.current_text
    assert workspace.current_text.count("PACKAGES") == 6

    quantity_requirements = cargo_package_quantity_replacement_requirements(
        source,
        source_label,
        target_label,
        cargo,
    )
    assert [row.sourceLineIds[0] for row in quantity_requirements] == [
        "L00001",
        "L00004",
        "L00006",
        "L00008",
        "L00010",
    ]
    assert [row.targetSurface for row in quantity_requirements] == [
        "991",
        "249",
        "249",
        "247",
        "246",
    ]

    guarded = bind_cargo_package_surface_guards(
        cargo,
        source_label,
        target_label,
        _target_integrity_resources().packages,
    )
    assert guarded[0].enforcePackageSurfaceGuard
    assert guarded[0].allowedPackageSurfaces == ("PACKAGE", "PACKAGES")


def test_cargo_package_guard_rejects_an_invented_non_target_package_noun() -> None:
    requirement = CargoFlavorRewriteRequirement(
        requirementId="cargo-group-1-span-1",
        targetPath="documentPatch.cargoGroups[0].description",
        targetDescription="SYNTHETIC MACHINE PARTS",
        sourceLineIds=("L00001", "L00002"),
        sourceSurfaces=("SOURCE MACHINE PARTS", "PACKED FOR EXPORT"),
        enforcePackageSurfaceGuard=True,
        allowedPackageSurfaces=("PACKAGE", "PACKAGES"),
    )

    failures = _cargo_flavor_rewrite_failures(
        "SYNTHETIC MACHINE PARTS\nPACKED IN EXPORT CARTONS\n",
        (requirement,),
    )

    assert failures[0]["unexpectedPackageSurfaces"] == [
        {
            "lineId": "L00002",
            "surface": "CARTONS",
            "allowedSurfaces": ["PACKAGE", "PACKAGES"],
        }
    ]


def test_cargo_package_guard_allows_package_surface_in_explicit_target_text() -> None:
    source_label = {
        "documentPatch": {
            "cargoGroups": [{"groupId": "g1", "description": "SOURCE GOODS"}],
            "cargoPackages": [
                {
                    "groupId": "g1",
                    "packageId": "p1",
                    "quantity": 665,
                    "typeCategory": "PACKAGE_CARTON",
                }
            ],
        }
    }
    target_label = {
        "documentPatch": {
            "cargoGroups": [
                {
                    "groupId": "g1",
                    "description": "HOT-ROLLED STEEL",
                    "additionalInformation": ["101 PACKAGES PACKED IN STEEL BUNDLES"],
                }
            ],
            "cargoPackages": [
                {
                    "groupId": "g1",
                    "packageId": "p1",
                    "quantity": 101,
                    "typeCategory": "PACKAGE_PACKAGE",
                }
            ],
        }
    }
    requirement = CargoFlavorRewriteRequirement(
        requirementId="cargo-group-1-span-1",
        targetPath="documentPatch.cargoGroups[0].description",
        targetDescription="HOT-ROLLED STEEL",
        sourceLineIds=("L00001",),
        sourceSurfaces=("SOURCE GOODS",),
    )

    guarded = bind_cargo_package_surface_guards(
        (requirement,),
        source_label,
        target_label,
        _target_integrity_resources().packages,
    )

    assert set(guarded[0].allowedPackageSurfaces) == {
        "PACKAGE",
        "PACKAGES",
        "STEEL BUNDLES",
    }
    assert not _cargo_flavor_rewrite_failures(
        "HOT-ROLLED STEEL\n101 PACKAGES PACKED IN STEEL BUNDLES\n",
        guarded,
    )


def test_compact_aggregate_equipment_summary_renders_every_target_family() -> None:
    source = "5 X 40HC\nContinued on Next Sheet\n"
    source_label = {
        "documentPatch": {
            "containers": [
                {"containerNumber": f"CMAU00000{index}0", "typeDescription": "40HC"}
                for index in range(5)
            ]
        }
    }
    target_label = {
        "documentPatch": {
            "containers": [
                {
                    "containerNumber": f"CMAU10000{index}0",
                    "sizeCategory": (
                        "FORTY_FOOT_HIGH_CUBE" if index < 3 else "TWENTY_FOOT_STANDARD_HEIGHT"
                    ),
                    "typeCategory": "GENERAL_PURPOSE",
                }
                for index in range(5)
            ]
        }
    }

    requirements = _aggregate_equipment_breakdown_requirements(source, source_label, target_label)

    assert len(requirements) == 5
    assert {row.targetSurface for row in requirements} == {
        "3 X 40' HIGH CUBE GENERAL PURPOSE + 2 X 20' STANDARD HEIGHT GENERAL PURPOSE"
    }
    workspace = RewriteWorkspace(
        original_text=source,
        current_text=source,
        surface_requirements=requirements,
    )
    apply_deterministic_prefills(workspace)
    assert workspace.current_text.splitlines()[0] == next(iter(requirements)).targetSurface


def test_parenthetical_aggregate_equipment_summary_renders_every_target_family() -> None:
    source = "SAY: EIGHT (20DRX8) CONTAINERS ONLY.\n"
    source_label = {
        "documentPatch": {
            "containers": [{"containerNumber": f"CMAU00000{index}0"} for index in range(8)]
        }
    }
    target_label = {
        "documentPatch": {
            "containers": [
                {
                    "containerNumber": f"CMAU10000{index}0",
                    "sizeCategory": (
                        "FORTY_FOOT_HIGH_CUBE" if index != 2 else "TWENTY_FOOT_STANDARD_HEIGHT"
                    ),
                    "typeCategory": "GENERAL_PURPOSE",
                }
                for index in range(8)
            ]
        }
    }

    requirements = _aggregate_equipment_breakdown_requirements(source, source_label, target_label)

    assert len(requirements) == 8
    assert {row.targetSurface for row in requirements} == {
        "SAY: EIGHT (40' HIGH CUBE GENERAL PURPOSEX7 + "
        "20' STANDARD HEIGHT GENERAL PURPOSEX1) CONTAINERS ONLY."
    }
    workspace = RewriteWorkspace(
        original_text=source,
        current_text=source,
        surface_requirements=requirements,
    )
    apply_deterministic_prefills(workspace)
    assert workspace.current_text.splitlines()[0] == next(iter(requirements)).targetSurface


def test_aggregate_summary_is_not_partially_rendered_after_topology_projection() -> None:
    source = "SAY: TWO (20DRX2) CONTAINERS ONLY.\n"
    source_label = {
        "documentPatch": {
            "containers": [
                {"containerNumber": "CMAU0000010"},
                {"containerNumber": "CMAU0000020"},
            ]
        }
    }
    target_label = {
        "documentPatch": {
            "containers": [
                {
                    "containerNumber": "CMAU1000010",
                    "sizeCategory": "FORTY_FOOT_HIGH_CUBE",
                    "typeCategory": "GENERAL_PURPOSE",
                },
                {"containerNumber": "CMAU1000020"},
            ]
        }
    }

    requirements = _aggregate_equipment_breakdown_requirements(
        source, source_label, target_label
    )

    assert requirements == ()


def test_mixed_parenthetical_aggregate_equipment_summary_is_reconciled() -> None:
    source = "SAY : TWO (20DRX1 & 40HCX1) CONTAINERS ONLY.\n"
    source_label = {
        "documentPatch": {
            "containers": [
                {"containerNumber": "FSCU3530390"},
                {"containerNumber": "MEDU4913783"},
            ]
        }
    }
    target_label = {
        "documentPatch": {
            "containers": [
                {
                    "containerNumber": "FSCU1127468",
                    "sizeCategory": "FORTY_FOOT_HIGH_CUBE",
                    "typeCategory": "GENERAL_PURPOSE",
                },
                {
                    "containerNumber": "MEDU2919397",
                    "sizeCategory": "FORTY_FOOT_HIGH_CUBE",
                    "typeCategory": "GENERAL_PURPOSE",
                },
            ]
        }
    }

    requirements = _aggregate_equipment_breakdown_requirements(source, source_label, target_label)

    assert len(requirements) == 2
    assert {row.targetSurface for row in requirements} == {
        "SAY : TWO (40' HIGH CUBE GENERAL PURPOSEX2) CONTAINERS ONLY."
    }


def test_counted_aggregate_equipment_preserves_container_clause() -> None:
    source = "2 x 40HR CONTAINER\n"
    source_label = {
        "documentPatch": {
            "containers": [
                {"containerNumber": "MNBU4006124", "typeDescription": "40HR - CONTAINER"},
                {"containerNumber": "MNBU9089862", "typeDescription": "40HR - CONTAINER"},
            ]
        }
    }
    target_label = {
        "documentPatch": {
            "containers": [
                {
                    "containerNumber": "MNBU1006127",
                    "sizeCategory": "FORTY_FOOT_HIGH_CUBE",
                    "typeCategory": "GENERAL_PURPOSE",
                },
                {
                    "containerNumber": "MNBU1089860",
                    "sizeCategory": "TWENTY_FOOT_STANDARD_HEIGHT",
                    "typeCategory": "GENERAL_PURPOSE",
                },
            ]
        }
    }

    requirements = _aggregate_equipment_breakdown_requirements(source, source_label, target_label)

    assert len(requirements) == 2
    assert {row.targetSurface for row in requirements} == {
        "1 x 40' HIGH CUBE GENERAL PURPOSE + 1 x 20' STANDARD HEIGHT GENERAL PURPOSE CONTAINER"
    }


def test_literal_aggregate_equipment_supports_unresolved_source_alias() -> None:
    source = "2x40' HW\n"
    source_label = {
        "documentPatch": {
            "containers": [
                {"containerNumber": "BORU7013618", "typeDescription": "40' HW"},
                {"containerNumber": "BORU7011256", "typeDescription": "40' HW"},
            ]
        }
    }
    target_label = {
        "documentPatch": {
            "containers": [
                {
                    "containerNumber": "BORU9013612",
                    "sizeCategory": "FORTY_FOOT_HIGH_CUBE",
                    "typeCategory": "GENERAL_PURPOSE",
                },
                {
                    "containerNumber": "BORU9011252",
                    "sizeCategory": "TWENTY_FOOT_STANDARD_HEIGHT",
                    "typeCategory": "GENERAL_PURPOSE",
                },
            ]
        }
    }

    requirements = _aggregate_equipment_breakdown_requirements(source, source_label, target_label)

    assert len(requirements) == 2
    assert {row.targetSurface for row in requirements} == {
        "1x40' HIGH CUBE GENERAL PURPOSE + 1x20' STANDARD HEIGHT GENERAL PURPOSE"
    }


def test_dense_container_table_reconciles_rows_tare_and_document_totals() -> None:
    config = load_synthesis_raw_text_rewrite_cycle_probe_config(
        Path("configs/synthesis/mpci_bl_raw_text_atomic10_luna_high.yaml")
    )
    limits = capacity_limits(config.target_integrity.transport_capacity)
    source_numbers = ("TLLU4654238", "CMAU7646960")
    target_numbers = ("TLLU1488350", "CMAU9583639")
    source = (
        "--- PAGE 1 ---\n"
        "GROSS WEIGHT\nCARGO\nKGS\nKGS\nCBM\n"
        f"{source_numbers[0]}\nSEAL A1\n1 x 40HC 20 PACKAGE(S)\n"
        "20000.000\n3700\n60.000\n\n"
        f"{source_numbers[1]}\nSEAL A2\n1 x 40HC 10 PACKAGE(S)\n"
        "10000.000\n2300\n40.000\n\n"
        "Continued From Previous Sheet Sheet 2 of 3 30000.000 6000 100.000\n"
    )
    source_label = {
        "documentPatch": {
            "containers": [
                {"containerNumber": number, "typeDescription": "40HC"} for number in source_numbers
            ],
            "cargoGroups": [
                {
                    "groupId": "g1",
                    "grossWeight": {"value": 30000, "unit": "kilogram"},
                    "volume": {"value": 100, "unit": "cubic_metre"},
                }
            ],
            "cargoPackages": [{"groupId": "g1", "packageId": "p1", "quantity": 30}],
            "cargoAllocationGroups": [
                {
                    "groupId": "g1",
                    "packageIds": ["p1"],
                    "coverage": "single_package_level",
                    "allocations": [
                        {"containerNumber": source_numbers[0], "packageQuantity": 20},
                        {"containerNumber": source_numbers[1], "packageQuantity": 10},
                    ],
                }
            ],
        }
    }
    target_label = {
        "documentPatch": {
            "containers": [
                {
                    "containerNumber": target_numbers[0],
                    "sizeCategory": "FORTY_FOOT_HIGH_CUBE",
                    "typeCategory": "GENERAL_PURPOSE",
                },
                {
                    "containerNumber": target_numbers[1],
                    "sizeCategory": "TWENTY_FOOT_STANDARD_HEIGHT",
                    "typeCategory": "GENERAL_PURPOSE",
                },
            ],
            "cargoGroups": [
                {
                    "groupId": "g1",
                    "grossWeight": {"value": 20000, "unit": "kilogram"},
                    "volume": {"value": 60, "unit": "cubic_metre"},
                }
            ],
            "cargoPackages": [{"groupId": "g1", "packageId": "p1", "quantity": 12}],
            "cargoAllocationGroups": [
                {
                    "groupId": "g1",
                    "packageIds": ["p1"],
                    "coverage": "single_package_level",
                    "allocations": [
                        {"containerNumber": target_numbers[0], "packageQuantity": 7},
                        {"containerNumber": target_numbers[1], "packageQuantity": 5},
                    ],
                }
            ],
        }
    }
    profiles = (
        EmpiricalOperationalProfile(
            document_id="profile-40",
            container_number="PROF4000000",
            equipment_family="forty_high_cube",
            gross_weight_kg=None,
            gross_utilization=None,
            volume_m3=None,
            volume_utilization=None,
            tare_weight_kg=Decimal("3850"),
        ),
        EmpiricalOperationalProfile(
            document_id="profile-20",
            container_number="PROF2000000",
            equipment_family="twenty_standard",
            gross_weight_kg=None,
            gross_utilization=None,
            volume_m3=None,
            volume_utilization=None,
            tare_weight_kg=Decimal("2250"),
        ),
    )

    operational = operational_flavor_requirements(
        source,
        source_label,
        target_label,
        source_document_id="source-document",
        scenario_id="scenario-dense",
        profiles=profiles,
        limits=limits,
    )
    aggregates = aggregate_operational_replacement_requirements(source, operational)
    equipment = container_equipment_replacement_requirements(source, source_label, target_label)
    workspace = RewriteWorkspace(
        original_text=source,
        current_text=source,
        operational_flavor_requirements=operational,
        anchored_scalar_replacement_requirements=(aggregates + equipment),
    )

    apply_deterministic_prefills(workspace)

    assert {row.kind for row in operational} == {
        "gross_weight_kg",
        "tare_weight_kg",
        "volume_m3",
        "package_quantity",
    }
    assert workspace.current_text.splitlines()[8:19] == [
        "1 x 40HC 7 PACKAGE(S)",
        "11666.667",
        "3850",
        "35.000",
        "",
        source_numbers[1],
        "SEAL A2",
        "1 x 20' STANDARD HEIGHT GENERAL PURPOSE 5 PACKAGE(S)",
        "8333.333",
        "2250",
        "25.000",
    ]
    assert workspace.current_text.splitlines()[-1] == (
        "Continued From Previous Sheet Sheet 2 of 3 20000.000 6100 60.000"
    )


def test_inline_operational_row_is_prefilled_right_to_left_without_column_drift() -> None:
    config = load_synthesis_raw_text_rewrite_cycle_probe_config(
        Path("configs/synthesis/mpci_bl_raw_text_atomic10_luna_high.yaml")
    )
    limits = capacity_limits(config.target_integrity.transport_capacity)
    source = "PONU1944248 40 DRY 8'6 27 PALLETS 5940.000 KGS 15.0000 CBM\n"
    source_label = {
        "documentPatch": {
            "containers": [{"containerNumber": "PONU1944248", "typeDescription": "40 DRY 8'6"}],
            "cargoGroups": [
                {
                    "groupId": "g1",
                    "grossWeight": {"value": 5940, "unit": "kilogram"},
                    "volume": {"value": 15, "unit": "cubic_metre"},
                }
            ],
            "cargoPackages": [
                {
                    "groupId": "g1",
                    "packageId": "p1",
                    "quantity": 27,
                    "typeCategory": "PACKAGE_PALLET",
                }
            ],
            "cargoAllocationGroups": [
                {
                    "groupId": "g1",
                    "packageIds": ["p1"],
                    "coverage": "single_package_level",
                    "allocations": [{"containerNumber": "PONU1944248", "packageQuantity": 27}],
                }
            ],
        }
    }
    target_label = {
        "documentPatch": {
            "containers": [
                {
                    "containerNumber": "PONU9987518",
                    "sizeCategory": "FORTY_FOOT_STANDARD_HEIGHT",
                    "typeCategory": "GENERAL_PURPOSE",
                }
            ],
            "cargoGroups": [
                {
                    "groupId": "g1",
                    "grossWeight": {"value": 3000, "unit": "kilogram"},
                    "volume": {"value": 10, "unit": "cubic_metre"},
                }
            ],
            "cargoPackages": [
                {
                    "groupId": "g1",
                    "packageId": "p1",
                    "quantity": 8,
                    "typeCategory": "PACKAGE_PACKAGE",
                }
            ],
            "cargoAllocationGroups": [
                {
                    "groupId": "g1",
                    "packageIds": ["p1"],
                    "coverage": "single_package_level",
                    "allocations": [{"containerNumber": "PONU9987518", "packageQuantity": 8}],
                }
            ],
        }
    }
    operational = operational_flavor_requirements(
        source,
        source_label,
        target_label,
        source_document_id="source-document",
        scenario_id="scenario-inline-row",
        profiles=(),
        limits=limits,
    )
    package_types = container_package_type_replacement_requirements(
        source,
        source_label,
        target_label,
        _target_integrity_resources().packages,
    )
    assert {row.kind for row in operational} == {
        "gross_weight_kg",
        "volume_m3",
        "package_quantity",
    }
    workspace = RewriteWorkspace(
        original_text=source,
        current_text=source,
        operational_flavor_requirements=operational,
        anchored_scalar_replacement_requirements=package_types,
    )

    apply_deterministic_prefills(workspace)

    assert workspace.current_text == (
        "PONU1944248 40 DRY 8'6 8 PACKAGES 3000.000 KGS 10.0000 CBM\n"
    )


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


def test_cargo_failure_diagnostic_names_only_unchanged_lines() -> None:
    requirement = CargoFlavorRewriteRequirement(
        requirementId="cargo-group-1-span-1",
        targetPath="documentPatch.cargoGroups[0].description",
        targetDescription="SYNTHETIC PAPER PRODUCTS",
        sourceLineIds=("L00002", "L00003", "L00004"),
        sourceSurfaces=("OLD PAPER", "OPEN DOT 6INCH", "OPEN DOT 3INCH"),
    )

    failures = _cargo_flavor_rewrite_failures(
        "--- PAGE 1 ---\nSYNTHETIC PAPER PRODUCTS\nOPEN DOT 6INCH\nNEW AUXILIARY DETAIL\n",
        (requirement,),
    )

    assert failures == (
        {
            "requirementId": "cargo-group-1-span-1",
            "targetPath": "documentPatch.cargoGroups[0].description",
            "targetDescriptionMissing": False,
            "missingLineIds": [],
            "unchangedSourceLines": [{"lineId": "L00003", "sourceSurface": "OPEN DOT 6INCH"}],
        },
    )


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


def test_labeled_net_weight_and_every_repeated_measurement_are_prefilled() -> None:
    source = (
        "--- PAGE 1 ---\n"
        "NET WEIGHT: 16,000 KGS\n"
        "Net Weight (KGS)\n"
        "16000.000\n"
        "G.W 16128.000 N.W 16000.000\n"
    )
    source_label = {
        "documentPatch": {
            "cargoGroups": [{"groupId": "g1", "netWeight": {"value": 16000, "unit": "kilogram"}}]
        }
    }
    target_label = {
        "documentPatch": {
            "cargoGroups": [{"groupId": "g1", "netWeight": {"value": 24715.9, "unit": "kilogram"}}]
        }
    }

    requirements = anchored_measurement_replacement_requirements(source, source_label, target_label)

    assert [(row.sourceLineIds, row.sourceSurface, row.targetSurface) for row in requirements] == [
        (("L00002",), "16,000", "24,715.9"),
        (("L00004", "L00005"), "16000.000", "24715.900"),
    ]
    workspace = RewriteWorkspace(
        original_text=source,
        current_text=source,
        anchored_scalar_replacement_requirements=requirements,
    )
    apply_deterministic_prefills(workspace)
    assert "NET WEIGHT: 24,715.9 KGS" in workspace.current_text
    assert "24715.900" in workspace.current_text
    assert "G.W 16128.000 N.W 24715.900" in workspace.current_text
    assert "16,000" not in workspace.current_text
    assert "16000.000" not in workspace.current_text


def test_package_quantity_prefill_covers_all_explicit_noun_bound_occurrences() -> None:
    source = (
        "--- PAGE 1 ---\n"
        "640 BAGS\n"
        "640.00BAGS\n"
        "SAY SIX HUNDRED FORTY PACKAGE(S)\n"
        "PHONE +60 640 9999\n"
        "60 DAYS FREE TIME\n"
    )
    source_label = {
        "documentPatch": {
            "cargoGroups": [{"groupId": "g1"}],
            "cargoPackages": [{"groupId": "g1", "packageId": "p1", "quantity": 640}],
            "cargoAllocationGroups": [
                {
                    "groupId": "g1",
                    "allocations": [{"containerNumber": "ABCD0000000", "packageQuantity": 640}],
                }
            ],
        }
    }
    target_label = {
        "documentPatch": {
            "cargoGroups": [{"groupId": "g1"}],
            "cargoPackages": [{"groupId": "g1", "packageId": "p1", "quantity": 512}],
            "cargoAllocationGroups": [
                {
                    "groupId": "g1",
                    "allocations": [{"containerNumber": "ABCD0000000", "packageQuantity": 512}],
                }
            ],
        }
    }
    leaves = rewrite_changed_leaves(source_label, target_label)

    requirements = anchored_package_quantity_replacement_requirements(source, leaves)

    assert [(row.sourceLineIds, row.sourceSurface, row.targetSurface) for row in requirements] == [
        (("L00002",), "640", "512"),
        (("L00003",), "640.00", "512.00"),
        (("L00004",), "SIX HUNDRED FORTY", "FIVE HUNDRED TWELVE"),
    ]
    workspace = RewriteWorkspace(
        original_text=source,
        current_text=source,
        anchored_scalar_replacement_requirements=requirements,
    )
    apply_deterministic_prefills(workspace)
    assert workspace.current_text == (
        "--- PAGE 1 ---\n"
        "512 BAGS\n"
        "512.00BAGS\n"
        "SAY FIVE HUNDRED TWELVE PACKAGE(S)\n"
        "PHONE +60 640 9999\n"
        "60 DAYS FREE TIME\n"
    )


def test_package_quantity_prefill_rejects_one_source_number_with_conflicting_targets() -> None:
    source = "10 BAGS\n10 CARTONS\n"
    leaves = (
        ChangedLeaf(
            path="documentPatch.cargoPackages[0].quantity",
            sourcePresent=True,
            targetPresent=True,
            sourceValue=10,
            targetValue=12,
            evidenceClass="printed_fact",
            requiresTextEdit=True,
        ),
        ChangedLeaf(
            path="documentPatch.cargoPackages[1].quantity",
            sourcePresent=True,
            targetPresent=True,
            sourceValue=10,
            targetValue=8,
            evidenceClass="printed_fact",
            requiresTextEdit=True,
        ),
    )

    assert anchored_package_quantity_replacement_requirements(source, leaves) == ()


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


def test_package_noun_prefill_allows_digit_adjacent_compact_surfaces() -> None:
    source = "88PLTS=1556CTNS=31064PCS\n2396CARTON(S)\n"
    requirements = (
        AnchoredScalarReplacementRequirement(
            targetPaths=("documentPatch.cargoPackages[0].typeCategory",),
            sourceLineIds=("L00001",),
            sourceSurface="CTNS",
            targetSurface="SACKS",
            surfaceKind="package_noun",
        ),
        AnchoredScalarReplacementRequirement(
            targetPaths=("documentPatch.cargoPackages[1].typeCategory",),
            sourceLineIds=("L00002",),
            sourceSurface="CARTON",
            targetSurface="PACKAGE",
            surfaceKind="package_noun",
        ),
    )
    workspace = RewriteWorkspace(
        original_text=source,
        current_text=source,
        anchored_scalar_replacement_requirements=requirements,
    )

    apply_deterministic_prefills(workspace)

    assert workspace.current_text == "88PLTS=1556SACKS=31064PCS\n2396PACKAGE(S)\n"


def test_repeated_equal_package_totals_remain_owned_by_their_cargo_groups() -> None:
    source = (
        "1 Container Said to Contain 1 PACKAGE\n"
        "FIRST MACHINE\n"
        "MAEU4092466 SEAL1 40 OPEN 9'6 1 PACKAGE 22000.000 KGS\n"
        "1 Container Said to Contain 1 PACKAGE\n"
        "SECOND MACHINE\n"
        "MAEU4198170 SEAL2 40 OPEN 9'6 1 PACKAGE 22100.000 KGS\n"
    )
    source_label = {
        "documentPatch": {
            "cargoGroups": [
                {"groupId": "g1", "description": "FIRST MACHINE"},
                {"groupId": "g2", "description": "SECOND MACHINE"},
            ],
            "cargoPackages": [
                {
                    "groupId": "g1",
                    "packageId": "p1",
                    "quantity": 1,
                    "typeCategory": "PACKAGE_PACKAGE",
                },
                {
                    "groupId": "g2",
                    "packageId": "p2",
                    "quantity": 1,
                    "typeCategory": "PACKAGE_PACKAGE",
                },
            ],
            "cargoAllocationGroups": [
                {
                    "groupId": "g1",
                    "packageIds": ["p1"],
                    "allocations": [
                        {"containerNumber": "MAEU4092466", "packageQuantity": 1}
                    ],
                },
                {
                    "groupId": "g2",
                    "packageIds": ["p2"],
                    "allocations": [
                        {"containerNumber": "MAEU4198170", "packageQuantity": 1}
                    ],
                },
            ],
        }
    }
    target_label = {
        "documentPatch": {
            "cargoGroups": [
                {"groupId": "g1", "description": "ELECTRIC GENERATOR"},
                {"groupId": "g2", "description": "OTHER SEATS"},
            ],
            "cargoPackages": [
                {
                    "groupId": "g1",
                    "packageId": "p1",
                    "quantity": 33,
                    "typeCategory": "PACKAGE_PACKAGE",
                },
                {
                    "groupId": "g2",
                    "packageId": "p2",
                    "quantity": 34,
                    "typeCategory": "PACKAGE_CARTON",
                },
            ],
            "cargoAllocationGroups": [
                {
                    "groupId": "g1",
                    "packageIds": ["p1"],
                    "allocations": [
                        {"containerNumber": "MAEU4105558", "packageQuantity": 33}
                    ],
                },
                {
                    "groupId": "g2",
                    "packageIds": ["p2"],
                    "allocations": [
                        {"containerNumber": "MAEU8030987", "packageQuantity": 34}
                    ],
                },
            ],
        }
    }
    cargo_requirements = (
        CargoFlavorRewriteRequirement(
            requirementId="cargo-group-1-span-1",
            targetPath="documentPatch.cargoGroups[0].description",
            targetDescription="ELECTRIC GENERATOR",
            sourceLineIds=("L00002",),
            sourceSurfaces=("FIRST MACHINE",),
        ),
        CargoFlavorRewriteRequirement(
            requirementId="cargo-group-2-span-1",
            targetPath="documentPatch.cargoGroups[1].description",
            targetDescription="OTHER SEATS",
            sourceLineIds=("L00005",),
            sourceSurfaces=("SECOND MACHINE",),
        ),
    )

    package_types = cargo_package_type_replacement_requirements(
        source,
        source_label,
        target_label,
        _target_integrity_resources().packages,
        cargo_requirements,
    )
    quantities = cargo_package_quantity_replacement_requirements(
        source, source_label, target_label, cargo_requirements
    )

    assert [(row.sourceLineIds, row.sourceSurface, row.targetSurface) for row in package_types] == [
        (("L00001",), "PACKAGE", "PACKAGES"),
        (("L00003",), "PACKAGE", "PACKAGES"),
        (("L00004",), "PACKAGE", "CARTONS"),
        (("L00006",), "PACKAGE", "CARTONS"),
    ]
    assert [(row.sourceLineIds, row.sourceSurface, row.targetSurface) for row in quantities] == [
        (("L00001",), "1", "33"),
        (("L00003",), "1", "33"),
        (("L00004",), "1", "34"),
        (("L00006",), "1", "34"),
    ]
    assert quantities[0].targetPaths == (
        "documentPatch.cargoAllocationGroups[0].allocations[0].packageQuantity",
        "documentPatch.cargoPackages[0].quantity",
    )
    assert quantities[2].targetPaths == (
        "documentPatch.cargoAllocationGroups[1].allocations[0].packageQuantity",
        "documentPatch.cargoPackages[1].quantity",
    )


def test_deterministic_prefill_applies_unambiguous_derived_surfaces() -> None:
    source = (
        "--- PAGE 1 ---\nHS CODE: 8714.10.00\nON BOARD: 05/05/2025\nCARRIER RECEIPT\n12 X 40'\n"
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


def test_deterministic_prefill_defers_one_hs_surface_with_multiple_targets() -> None:
    source = "HS CODE: 2401108590\n"
    first = SurfaceRenderingRequirement(
        kind="hs_code",
        targetPath="documentPatch.cargoGroups[0].hsCodes[0]",
        sourceSurface="2401108590",
        targetSurface="0902400010",
        sourceOccurrences=1,
        contextEvidence=source.strip(),
    )
    second = first.model_copy(
        update={
            "targetPath": "documentPatch.cargoGroups[1].hsCodes[0]",
            "targetSurface": "1209910010",
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


def test_deterministic_prefill_rejects_unowned_conflicting_date_projection() -> None:
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

    with pytest.raises(ValueError, match="cannot bind every conflicting date occurrence"):
        apply_deterministic_prefills(workspace)

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


def test_cargo_spans_do_not_merge_adjacent_exact_near_duplicate_products() -> None:
    source = (
        "USED MACHINE CATERPILLAR 966 F SERIAL NUMBER 3XJ01530\n"
        "MAEU4092466 ML-ES0121810 40 OPEN 9'6 1 PACKAGE 22000.000 KGS\n"
        "1 Container Said to Contain 1 PACKAGE\n"
        "USED MACHINE CATERPILLAR 966 D SERIAL NUMBER 94X02316\n"
        "MAEU4198170 ML-ES0121855 40 OPEN 9'6 1 PACKAGE 22100.000 KGS\n"
    )
    source_label = {
        "documentPatch": {
            "cargoGroups": [
                {"groupId": "g1", "description": "USED MACHINE CATERPILLAR 966 F"},
                {"groupId": "g2", "description": "USED MACHINE CATERPILLAR 966 D"},
            ]
        }
    }
    target_label = {
        "documentPatch": {
            "cargoGroups": [
                {"groupId": "g1", "description": "ELECTRIC GENERATOR"},
                {"groupId": "g2", "description": "OTHER SEATS"},
            ]
        }
    }

    requirements = cargo_flavor_rewrite_requirements(source, source_label, target_label)

    assert [(row.targetPath, row.sourceLineIds) for row in requirements] == [
        ("documentPatch.cargoGroups[0].description", ("L00001",)),
        ("documentPatch.cargoGroups[1].description", ("L00004",)),
    ]


def test_exact_single_line_cargo_description_is_compiled_without_model_rewriting() -> None:
    source = "Heading\nproduits de maintenance industrielle\nUN 3077 OLD SHIPPING NAME\n"
    source_label = {
        "documentPatch": {
            "cargoGroups": [
                {"groupId": "g1", "description": "produits de maintenance industrielle"}
            ]
        }
    }
    requirements = (
        CargoFlavorRewriteRequirement(
            requirementId="cargo-group-1-span-1",
            targetPath="documentPatch.cargoGroups[0].description",
            targetDescription="ammonium perchlorate",
            sourceLineIds=("L00002",),
            sourceSurfaces=("produits de maintenance industrielle",),
        ),
    )

    compiled = exact_cargo_line_replacement_requirements(source, source_label, requirements)

    assert compiled == (
        AnchoredScalarReplacementRequirement(
            targetPaths=("documentPatch.cargoGroups[0].description",),
            sourceLineIds=("L00002",),
            sourceSurface="produits de maintenance industrielle",
            targetSurface="ammonium perchlorate",
        ),
    )


def test_cargo_span_does_not_match_word_suffix_in_unrelated_header() -> None:
    source = (
        "--- PAGE 1 ---\n"
        "FREIGHT & CHARGES\n"
        "RATE\n"
        "\n"
        "80 Drum(s) of ASEPTIC WHITE GRAPE JUICE CONCENTRATE\n"
        "JUICE CONCENTRATE\n"
    )
    source_label = {
        "documentPatch": {
            "cargoGroups": [
                {
                    "groupId": "g1",
                    "description": "ASEPTIC WHITE GRAPE JUICE CONCENTRATE",
                }
            ]
        }
    }
    target_label = {
        "documentPatch": {"cargoGroups": [{"groupId": "g1", "description": "CHILLED SHARK FINS"}]}
    }

    requirements = cargo_flavor_rewrite_requirements(source, source_label, target_label)

    assert len(requirements) == 1
    assert requirements[0].sourceLineIds == ("L00005", "L00006")


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


def test_generic_signatory_role_is_not_invented_as_a_private_agent_identity() -> None:
    source_label = {"documentPatch": {"parties": {"carrier": {"name": "SOURCE LINE"}}}}
    target_label = {"documentPatch": {"parties": {"carrier": {"name": "TARGET LINE"}}}}
    source = (
        "By\n"
        "General Manager\n"
        "as agent for the Carrier SOURCE LINE\n"
    )

    assert raw_auxiliary_identity_requirements(source, source_label, target_label) == ()


def test_named_signatory_with_generic_role_remains_anonymization_owned() -> None:
    source_label = {"documentPatch": {"parties": {"carrier": {"name": "SOURCE LINE"}}}}
    target_label = {"documentPatch": {"parties": {"carrier": {"name": "TARGET LINE"}}}}
    source = (
        "A. MARIN\n"
        "General Manager\n"
        "as agent for the Carrier SOURCE LINE\n"
    )

    requirements = raw_auxiliary_identity_requirements(source, source_label, target_label)

    assert len(requirements) == 1
    assert requirements[0].sourceIdentity == "A. MARIN\nGeneral Manager"


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


def test_signed_on_behalf_agent_is_not_duplicated_by_generic_inline_parser() -> None:
    source_label = {"documentPatch": {"parties": {"carrier": {"name": "SOURCE OCEAN LINE"}}}}
    target_label = {"documentPatch": {"parties": {"carrier": {"name": "AURELIA HORIZON MARITIME"}}}}
    source = (
        "SIGNED on behalf of the Carrier SOURCE OCEAN LINE\n"
        "by Meridian Gateway Agencies N.V. As Agent For The Carrier\n"
        "\n"
        "--- PAGE 2 ---\n"
    )

    requirements = raw_auxiliary_identity_requirements(source, source_label, target_label)

    assert len(requirements) == 1
    assert requirements[0].sourceIdentity == "Meridian Gateway Agencies N.V."
    assert requirements[0].targetPrincipalName == "AURELIA HORIZON MARITIME"
    assert requirements[0].sourceEvidence == (
        "SIGNED on behalf of the Carrier SOURCE OCEAN LINE\n"
        "by Meridian Gateway Agencies N.V. As Agent For The Carrier"
    )


def test_signed_for_carrier_principal_is_not_absorbed_into_following_agent_identity() -> None:
    source_label = {"documentPatch": {"parties": {"carrier": {"name": "CMA CGM"}}}}
    target_label = {
        "documentPatch": {"parties": {"carrier": {"name": "NORTHGATE MARITIME LINES LLC"}}}
    }
    source = (
        "SIGNED FOR THE CARRIER CMA CGM S.A.\n"
        "BY CMA CGM XIAMEN\n"
        "as agents for the carrier CMA CGM S. A.\n"
    )

    requirements = raw_auxiliary_identity_requirements(source, source_label, target_label)

    assert len(requirements) == 1
    assert requirements[0].sourceIdentity == "BY CMA CGM XIAMEN"
    assert requirements[0].sourceEvidence == (
        "BY CMA CGM XIAMEN\nas agents for the carrier CMA CGM S. A."
    )


def test_abbreviated_agent_principal_still_anonymizes_the_explicit_agent() -> None:
    source_label = {
        "documentPatch": {
            "parties": {
                "carrier": {"name": "CMA CGM Société Anonyme au Capital de 234 988 330 Euros"}
            }
        }
    }
    target_label = {
        "documentPatch": {"parties": {"carrier": {"name": "HELVETIC CREST NAVIGATION AG"}}}
    }
    source = (
        "SIGNED FOR THE CARRIER CMA CGM S.A.\n"
        "BY CMA CGM Deutschland GmbH Shipping Agency as agents for the carrier CMA CGM S. A.\n"
    )

    requirements = raw_auxiliary_identity_requirements(source, source_label, target_label)

    assert len(requirements) == 1
    assert requirements[0].sourceIdentity == "BY CMA CGM Deutschland GmbH Shipping Agency"
    assert requirements[0].targetPrincipalName == "HELVETIC CREST NAVIGATION AG"


def test_signed_for_carrier_trailing_by_is_not_part_of_the_principal() -> None:
    source_label = {"documentPatch": {"parties": {"carrier": {"name": "CMA CGM SA"}}}}
    target_label = {
        "documentPatch": {"parties": {"carrier": {"name": "NORTHGATE MARITIME LINES LLC"}}}
    }
    source = (
        "Signed for the Carrier CMA CGM SA by\nCMA CGM (AMERICA) LLC as agent for the Carrier\n"
    )

    surfaces = surface_rendering_requirements(source, source_label, target_label)
    principal = [row for row in surfaces if row.kind == "carrier_principal_identity"]
    assert [(row.sourceSurface, row.sourceLineIds) for row in principal] == [
        ("CMA CGM SA", ("L00001",))
    ]
    workspace = RewriteWorkspace(
        original_text=source,
        current_text=source,
        current_target_label=target_label,
        surface_requirements=tuple(surfaces),
    )
    apply_deterministic_prefills(workspace)
    assert workspace.current_text.splitlines()[0] == (
        "Signed for the Carrier NORTHGATE MARITIME LINES LLC by"
    )
    auxiliary = raw_auxiliary_identity_requirements(source, source_label, target_label)
    assert len(auxiliary) == 1
    assert auxiliary[0].sourceIdentity == "CMA CGM (AMERICA) LLC"
    assert auxiliary[0].targetPrincipalName == "NORTHGATE MARITIME LINES LLC"


def test_wrapped_on_board_agent_is_anonymized_when_the_task_has_no_carrier() -> None:
    source = (
        "Shipped on Board EVER LEARNED 19-MAY-2023 CMA CGM CHINA SHIPPING\n"
        "CO. LTD As agents for the Carrier\n"
    )
    label = {"documentPatch": {"parties": {}}}

    requirements = raw_auxiliary_identity_requirements(source, label, label)

    assert len(requirements) == 1
    assert requirements[0].requirementId == "carrier-agent-L00001"
    assert requirements[0].sourceIdentity == "CMA CGM CHINA SHIPPING\nCO. LTD"
    assert requirements[0].sourceIdentityLineCount == 2
    assert requirements[0].targetPrincipalName == "THE CARRIER"


def test_inline_on_board_agent_excludes_vessel_and_date_from_its_identity() -> None:
    source = (
        "Shipped on Board APL SINGAPURA 20-MAY-2024 CMA CGM XIAMEN "
        "As agents for the Carrier\n\nWeight in Kgs Total: 5 CONTAINER(S)\n"
    )
    source_label = {"documentPatch": {"parties": {"carrier": {"name": "CMA CGM S.A."}}}}
    target_label = {"documentPatch": {"parties": {"carrier": {"name": "ALTURA MARITIMA S.A."}}}}

    requirements = raw_auxiliary_identity_requirements(source, source_label, target_label)

    assert len(requirements) == 1
    assert requirements[0].sourceIdentity == "CMA CGM XIAMEN"
    assert requirements[0].sourceIdentityLineCount == 1
    assert requirements[0].gapLineCount == 0
    assert requirements[0].targetPrincipalName == "THE CARRIER"


def test_inline_on_board_agent_accepts_a_unicode_fictional_identity() -> None:
    source = (
        "Shipped on Board APL SINGAPURA 20-MAY-2024 CMA CGM XIAMEN "
        "As agents for the Carrier\n\nWeight in Kgs Total: 5 CONTAINER(S)\n"
    )
    source_label = {"documentPatch": {"parties": {"carrier": {"name": "CMA CGM S.A."}}}}
    target_label = {
        "documentPatch": {"parties": {"carrier": {"name": "ALTURA MARITIMA S.A."}}}
    }
    workspace = RewriteWorkspace(
        original_text=source,
        current_text=source,
        current_target_label=target_label,
        raw_auxiliary_identity_requirements=raw_auxiliary_identity_requirements(
            source, source_label, target_label
        ),
    )

    commit = apply_line_range_replacements(
        workspace,
        (
            _edit(
                source,
                1,
                1,
                "Shipped on Board KANDA LOGGER 09-APR-2025 "
                "Servicios Logísticos Altamar S. de R.L. As agents for the Carrier",
            ),
        ),
    )

    assert len(commit.rawAuxiliaryIdentityRealizations) == 1
    assert commit.rawAuxiliaryIdentityRealizations[0].gapLineCount == 0
    assert deterministic_rewrite_audit(workspace, ()).rawAuxiliaryIdentitiesReplaced


def test_generic_agent_relation_stays_generic_when_target_carrier_exists() -> None:
    source = (
        "Shipped on Board SOURCE VESSEL 19-MAY-2023 OLD SHIPPING AGENCY\n"
        "PTE LTD As agents for the Carrier\n"
    )
    source_label = {"documentPatch": {"parties": {"carrier": {"name": "OLD OCEAN LINE"}}}}
    target_label = {"documentPatch": {"parties": {"carrier": {"name": "NEW MERIDIAN LINE"}}}}

    requirements = raw_auxiliary_identity_requirements(source, source_label, target_label)

    assert len(requirements) == 1
    assert requirements[0].targetPrincipalName == "THE CARRIER"


def test_shared_carrier_legal_identity_repairs_the_longer_party_target() -> None:
    raw = "FORWARDING AGENT\nTRANSGLORY S.A.\nSigned on behalf of the Carrier: TRANSGLORY\n"
    source = {
        "documentPatch": {
            "parties": {
                "carrier": {"name": "TRANSGLORY"},
                "forwardingAgent": {"name": "TRANSGLORY S.A."},
            }
        }
    }
    target = {
        "documentPatch": {
            "parties": {
                "carrier": {"name": "Aureline Oceanic Carriers"},
                "forwardingAgent": {"name": "Kestrel Meridian Forwarding S.A."},
            }
        }
    }

    changes = repair_overlapping_source_scalar_targets(raw, source, target)

    assert [row.reason for row in changes] == ["shared_source_party_identity_topology"]
    assert target["documentPatch"]["parties"]["carrier"]["name"] == ("Aureline Oceanic Carriers")
    assert target["documentPatch"]["parties"]["forwardingAgent"]["name"] == (
        "Aureline Oceanic Carriers S.A."
    )


def test_page_split_high_information_mark_is_anchored_atom_by_atom() -> None:
    source = (
        "--- PAGE 1 ---\nMARKS & NUMBERS\nQ40043311 -\n--- PAGE 2 ---\nMARKS & NUMBERS\nQ40043782\n"
    )
    leaves = (
        ChangedLeaf(
            path="documentPatch.cargoGroups[0].marksAndNumbers[0]",
            sourcePresent=True,
            targetPresent=True,
            sourceValue="Q40043311 - Q40043782",
            targetValue="RZ2401 - RZ2413",
            evidenceClass="printed_fact",
            requiresTextEdit=True,
        ),
    )

    requirements = anchored_scalar_replacement_requirements(source, leaves)

    assert [(row.sourceSurface, row.targetSurface, row.sourceLineIds) for row in requirements] == [
        ("Q40043311", "RZ2401", ("L00003",)),
        ("Q40043782", "RZ2413", ("L00006",)),
    ]
    assert target_literal_requirements(leaves, requirements) == ()
    workspace = RewriteWorkspace(
        original_text=source,
        current_text=source,
        anchored_scalar_replacement_requirements=requirements,
    )
    apply_deterministic_prefills(workspace)
    assert "RZ2401 -\n--- PAGE 2 ---" in workspace.current_text
    assert workspace.current_text.endswith("RZ2413\n")


def test_repeated_mark_surface_is_anchored_only_by_explicit_marks_prefix() -> None:
    source = (
        "CONSIGNEE\nEL GALLAD COMPANY FOR TRADING\n\n"
        "NOTIFY PARTY\nEL GALLAD COMPANY FOR TRADING\n\n"
        "MARKS EL GALLAD COMPANY FOR TRADING\n"
    )
    leaves = (
        ChangedLeaf(
            path="documentPatch.cargoGroups[0].marksAndNumbers[0]",
            sourcePresent=True,
            targetPresent=True,
            sourceValue="EL GALLAD COMPANY FOR TRADING",
            targetValue="NORTHSTAR CARGO 4827",
            evidenceClass="printed_fact",
            requiresTextEdit=True,
        ),
    )

    requirements = anchored_scalar_replacement_requirements(source, leaves)

    assert len(requirements) == 1
    assert requirements[0].sourceLineIds == ("L00007",)
    workspace = RewriteWorkspace(
        original_text=source,
        current_text=source,
        anchored_scalar_replacement_requirements=requirements,
    )
    apply_deterministic_prefills(workspace)
    assert workspace.current_text.count("EL GALLAD COMPANY FOR TRADING") == 2
    assert "MARKS NORTHSTAR CARGO 4827" in workspace.current_text


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
