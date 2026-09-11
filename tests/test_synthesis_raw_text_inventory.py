from __future__ import annotations

import json
from decimal import Decimal
from pathlib import Path
from types import SimpleNamespace

import pytest
from pydantic_ai.exceptions import ModelAPIError, ModelHTTPError

from document_ocr.hashing import canonical_json_bytes, sha256_bytes, sha256_file
from document_ocr.synthesis.config import (
    load_synthesis_raw_text_inventory_batch_config,
)
from document_ocr.synthesis.raw_text_hybrid_probe import (
    HybridWorkItem,
    _temperature_deactivation_line_numbers,
)
from document_ocr.synthesis.raw_text_inventory import (
    _SIGNED_CARRIER_IDENTITY,
    FullDocumentAudit,
    InventoryCandidate,
    RegressionCase,
    TemplateMutationCase,
    TemplateMutationLine,
    _carrier_receipt_count_matches_target,
    _is_shipment_aggregate_value_line,
    _numeric_lines,
    _raw_agent_identity_line_numbers,
    apply_deterministic_auxiliary_edits,
    audit_full_document,
    auxiliary_label_surfaces,
    build_mutable_inventory,
    locate_auxiliary_values,
    locate_embedded_long_identifiers,
    locate_standalone_opaque_identifiers,
    parse_regression_oracle,
    parse_template_mutation_profile,
)
from document_ocr.synthesis.raw_text_inventory_probe import (
    InventoryProbeCaseResult,
    _allocation_quantity_evidence_lines,
    _auxiliary_label_lock_requirements,
    _cargo_marks_sequence_evidence,
    _CompoundSlot,
    _editor_payload,
    _empty_usage,
    _handling_affix_requirements_by_line,
    _include_unique_linked_party_copies,
    _initial_request_batches,
    _is_explicit_package_quantity_line,
    _load_inventory_checkpoint,
    _locked_literal_requirements_by_line,
    _output_schema,
    _party_owned_surface_lines,
    _preflight_initial_model_contracts,
    _preview_repair_selection,
    _publish_inventory_checkpoint,
    _refine_cargo_group_evidence,
    _refine_equipment_evidence,
    _refine_foreign_party_evidence,
    _refine_numeric_evidence,
    _refine_party_evidence,
    _refine_party_occurrence_requirements,
    _refine_sibling_evidence,
    _repair_selection,
    _retry_delay_seconds,
    _Slot,
    _target_occurrence_repair,
    _transient_route_error,
    _validated_compound_realizations,
    _validated_output_replacements,
    raw_text_inventory_compiler_contract,
)
from document_ocr.synthesis.raw_text_rewrite_cycle_probe import (
    AnchoredScalarReplacementRequirement,
    AppliedDeterministicPrefill,
    AtomicRewriteCommit,
    CargoFlavorRewriteRequirement,
    CompoundPartyFlavorRequirement,
    RawAuxiliaryIdentityRequirement,
    RewriteWorkspace,
    SourceStatusPreservationRequirement,
    SurfaceRenderingRequirement,
    TargetLiteralRequirement,
    TargetValueOccurrenceRequirement,
    _party_scalar_occurrence_count,
    source_semantic_role_hints,
)
from document_ocr.synthesis.run_safety import StagedArtifactRun

_PROJECT_ROOT = Path(__file__).resolve().parents[1]


def test_inventory_compiler_contract_pins_container_semantics() -> None:
    dependencies = raw_text_inventory_compiler_contract()["dependencyImplementationSha256"]

    assert isinstance(dependencies, dict)
    assert dependencies["containerSemantics"] == sha256_file(
        _PROJECT_ROOT / "src/document_ocr/synthesis/container_semantics.py"
    )


def _labels() -> tuple[dict[str, object], dict[str, object]]:
    source = {
        "schemaVersion": "3.0.0-experimental",
        "documentPatch": {
            "cargoGroups": [
                {
                    "groupId": "g1",
                    "description": "GARLIC",
                    "grossWeight": {"unit": "kilogram", "value": 27560.0},
                    "volume": {"unit": "cubic_metre", "value": 55.0},
                }
            ],
            "cargoPackages": [
                {
                    "groupId": "g1",
                    "packageId": "p1",
                    "quantity": 2600,
                    "typeCategory": "PACKAGE_CARTON",
                }
            ],
            "containers": [
                {
                    "containerNumber": "FSCU5904400",
                    "temperatureSetpoint": {"unit": "celsius", "value": -3.0},
                }
            ],
        },
    }
    target = {
        "schemaVersion": "5.0.0-experimental",
        "documentPatch": {
            "cargoGroups": [
                {
                    "groupId": "g1",
                    "description": "STEEL",
                    "grossWeight": {"unit": "kilogram", "value": 2135.1},
                    "volume": {"unit": "cubic_metre", "value": 14.6},
                }
            ],
            "cargoPackages": [
                {"groupId": "g1", "packageId": "p1", "quantity": 325, "typeCategory": "PACKAGE"}
            ],
            "containers": [{"containerNumber": "FSCU4850130"}],
        },
    }
    return source, target


def test_locate_auxiliary_values_requires_explicit_value_syntax() -> None:
    text = (
        "ACID NO. 5086429732023040027\n"
        "SERVICE CONTRACT NO.\n"
        "TAON00243A\n"
        "Merchant must provide an ACID number and customs information.\n"
    )

    values = locate_auxiliary_values(text)

    assert {row.value for row in values} == {"5086429732023040027", "TAON00243A"}
    assert all("Merchant" not in row.value for row in values)


def test_source_only_contact_person_is_model_owned_not_shape_randomized() -> None:
    text = "CONTACT PERSON: HUANG JINLONG/ PANNI\n"

    located = locate_auxiliary_values(text)
    assert [(row.category, row.value, row.line_numbers) for row in located] == [
        ("contact person", "HUANG JINLONG/ PANNI", (1,))
    ]

    inventory = build_mutable_inventory(
        source_text=text,
        current_text=text,
        source_label={},
        target_label={},
        work_items=(),
        oracle_case=None,
    )

    assert len(inventory) == 1
    assert inventory[0].category == "source_only_contact_identity"
    assert inventory[0].disposition == "model_residual"
    assert inventory[0].sourceSurface == "HUANG JINLONG/ PANNI"


def test_contact_person_heading_does_not_make_generic_name_fields_writable() -> None:
    text = (
        "(BRAND NAME: INDUSTRIAL TECHNICAL GRADE)\n"
        "LOGISTICS CONTACT PERSON NAME, TEL. EMAIL\n"
        "Substance Name(Proper Shipping Name): TRINITROANISOLE\n"
    )

    assert locate_auxiliary_values(text) == ()


def test_auxiliary_label_surfaces_preserve_product_and_acid_headings() -> None:
    line = "PRODUCT CODE:100G*100BAGS/CARTON: 0402620044 ACID:1003633852023110047"

    assert auxiliary_label_surfaces(line) == ("PRODUCT CODE:", "ACID:")
    assert sorted(row.value for row in locate_auxiliary_values(line)) == [
        "0402620044",
        "1003633852023110047",
    ]


def test_package_owned_lot_is_not_also_locked_as_an_auxiliary_label() -> None:
    package_requirement = {
        "kind": "source_only_cargo_packaging",
        "paths": ["documentPatch.cargoGroups[0].description"],
        "source": ["lot"],
        "target": {
            "description": "TARGET GOODS",
            "allowedPackageSurfaces": ["CARTON", "CARTONS"],
        },
    }

    assert auxiliary_label_surfaces("One lot used machines and parts") == ("lot ",)
    assert (
        _auxiliary_label_lock_requirements(
            "One lot used machines and parts", (package_requirement,)
        )
        == ()
    )
    assert _auxiliary_label_lock_requirements("LOT: B7", ()) == (
        {
            "kind": "host_locked_source_literal",
            "paths": ["rawTemplate.fieldLabel"],
            "target": "LOT: ",
            "policy": "preserve_verbatim_on_this_line",
        },
    )


def test_embedded_long_identifier_excludes_phone_and_measurement() -> None:
    text = (
        "300G*40BAGS/CARTON: 0402620045 ACID:1003633852023110047\n"
        "Emergency Phone: 202-37609091\n"
        "Total Gross Weight: 19464100 KGS\n"
    )

    values = locate_embedded_long_identifiers(text)

    assert [(row.value, row.line_numbers) for row in values] == [
        ("0402620045", (1,)),
        ("1003633852023110047", (1,)),
    ]


def test_template_mutation_profile_grants_exact_line_attention_without_forcing_change() -> None:
    source_text = "SERVICE CONTRACT\nF14\n"
    profile_case = TemplateMutationCase(
        documentId="doc_" + "a" * 64,
        sourceTextSha256=sha256_bytes(source_text.encode()),
        lines=(
            TemplateMutationLine(
                lineId="L00002",
                sourceLineSha256=sha256_bytes(b"F14"),
                findingKinds=("source_only_private_or_auxiliary_fact",),
            ),
        ),
    )

    inventory = build_mutable_inventory(
        source_text=source_text,
        current_text=source_text,
        source_label={},
        target_label={},
        work_items=(),
        oracle_case=None,
        template_profile_case=profile_case,
    )
    profiled = [row for row in inventory if row.category == "template_profile_residual"]
    assert len(profiled) == 1
    assert profiled[0].lineIds == ("L00002",)

    audit = audit_full_document(
        document_id="doc_" + "a" * 64,
        source_text=source_text,
        output_text=source_text,
        source_label={},
        target_label={},
        inventory=inventory,
        deterministic_edits=(),
        oracle_case=None,
    )
    assert audit.passed is True


def test_parse_template_mutation_profile_rejects_duplicate_line_ownership() -> None:
    source_text = "A\nB\n"
    line = {
        "lineId": "L00001",
        "sourceLineSha256": sha256_bytes(b"A"),
        "findingKinds": ["target_fact_mismatch"],
    }
    value = {
        "schemaVersion": 1,
        "name": "duplicate-test",
        "sources": [
            {
                "path": "audit/_COMMIT.json",
                "sha256": "1" * 64,
                "kind": "certification_commit",
            }
        ],
        "cases": [
            {
                "documentId": "doc_" + "b" * 64,
                "sourceTextSha256": sha256_bytes(source_text.encode()),
                "lines": [line, line],
            }
        ],
    }

    with pytest.raises(ValueError, match="repeats a line ID"):
        parse_template_mutation_profile(value)


def test_template_mutation_profile_rejects_different_source_bytes() -> None:
    source_text = "SERVICE CONTRACT\nF14\n"
    profile_case = TemplateMutationCase(
        documentId="doc_" + "c" * 64,
        sourceTextSha256=sha256_bytes(b"different source"),
        lines=(
            TemplateMutationLine(
                lineId="L00002",
                sourceLineSha256=sha256_bytes(b"F14"),
                findingKinds=("source_only_private_or_auxiliary_fact",),
            ),
        ),
    )

    with pytest.raises(ValueError, match="source text SHA-256 differs"):
        build_mutable_inventory(
            source_text=source_text,
            current_text=source_text,
            source_label={},
            target_label={},
            work_items=(),
            oracle_case=None,
            template_profile_case=profile_case,
        )


def test_profile_context_is_read_only_and_cannot_be_returned_as_a_slot() -> None:
    slots = (
        _Slot(
            alias="s0",
            line_id="L00002",
            source_line="F14",
            requirements=(
                {
                    "kind": "template_profile_residual",
                    "policy": "reevaluate_exact_line_against_complete_synthetic_target",
                },
            ),
        ),
    )
    compiled = SimpleNamespace(
        workspace=RewriteWorkspace(
            original_text="SERVICE CONTRACT\nF14\nBOOKING REFERENCE\n",
            current_text="SERVICE CONTRACT\nF14\nBOOKING REFERENCE\n",
            current_target_label={"documentPatch": {"references": {"bookingNumber": "B77"}}},
        ),
        work_items=(),
        bundle=SimpleNamespace(result=SimpleNamespace(documentId="doc_" + "d" * 64)),
    )

    payload = _editor_payload(compiled, slots, profile_context_lines=1)
    assert payload["templateProfileReadOnlyContext"] == [
        {"lineId": "L00001", "sourceLine": "SERVICE CONTRACT"},
        {"lineId": "L00003", "sourceLine": "BOOKING REFERENCE"},
    ]
    _output_type, schema = _output_schema(slots)
    assert schema["required"] == ["s0"]
    assert isinstance(schema["properties"], dict)
    assert set(schema["properties"]) == {"s0"}


def test_source_only_phone_is_model_owned_after_task_phone_edit() -> None:
    source_text = "TEL:+202-37609091\nEmergency Phone: 202-37609091\n"
    current_text = "TEL:+65 6128 4739\nEmergency Phone: 202-37609091\n"
    source_label = {
        "documentPatch": {
            "parties": {"consignee": {"contactDetails": {"phoneNumbers": ["+202-37609091"]}}}
        }
    }
    target_label = {
        "documentPatch": {
            "parties": {"consignee": {"contactDetails": {"phoneNumbers": ["+65 6128 4739"]}}}
        }
    }
    work_item = HybridWorkItem(
        workItemId="W0001",
        targetPaths=("documentPatch.parties.consignee.contactDetails.phoneNumbers[0]",),
        action="replace",
        sourceValue="+202-37609091",
        targetValue="+65 6128 4739",
        state="deterministic_applied",
        evidenceLineIds=("L00001",),
        spanIds=(),
        locator="deterministic_requirement",
        rationale="test",
    )

    inventory = build_mutable_inventory(
        source_text=source_text,
        current_text=current_text,
        source_label=source_label,
        target_label=target_label,
        work_items=(work_item,),
        oracle_case=None,
    )
    model_sources = {row.sourceSurface for row in inventory if row.disposition == "model_residual"}
    output, edits = apply_deterministic_auxiliary_edits(
        text=current_text,
        document_id="doc_" + "a" * 64,
        scenario_id="syn-test",
        candidates=inventory,
    )

    assert "+202-37609091" not in model_sources
    assert "202-37609091" in model_sources
    assert output == current_text
    assert edits == ()


def test_locate_auxiliary_values_owns_value_after_trailing_cross_line_heading() -> None:
    text = (
        "(19,228.000KG/20.000M3/11PK)/ACID NUMBER:\n"
        "4649037182024040034\n"
        "Ordinary cargo prose mentioning an ACID number\n"
    )

    values = locate_auxiliary_values(text)

    assert [(row.category, row.value, row.line_numbers) for row in values] == [
        ("acid number", "4649037182024040034", (2,))
    ]


def test_locate_auxiliary_values_covers_unlabeled_email_and_explicit_operational_ids() -> None:
    text = (
        "Carrier Booking No.\n"
        "FHAM0177711\n"
        "SHIPPER ID: 231842862\n"
        "SCAC CODE: MAEU\n"
        "LOT NUMBER-21105117\n"
        "RMS NUMBER-21094106\n"
        "AS PER ORDER 541988\n"
        "SWIFT/BIC AGRIEGCXXXX\n"
        "SIGNED AGENT LTD ABN 42155-902-318\n"
        "ACCOUNTS@NAUTICLOG.COM\n"
    )

    values = locate_auxiliary_values(text)

    assert {row.value for row in values} == {
        "FHAM0177711",
        "231842862",
        "MAEU",
        "21105117",
        "21094106",
        "541988",
        "AGRIEGCXXXX",
        "42155-902-318",
        "ACCOUNTS@NAUTICLOG.COM",
    }


def test_locate_auxiliary_values_bounds_multiple_fields_on_one_line() -> None:
    text = (
        "INVOICE NO.6123110025 DATE FEBRUARY 2ND, 2024 "
        "ACID : 5452479342024010012 IMPORTER TAXATION NUMBER :545247934 "
        "EXPORTER IDENTIFICATION NUMBER :0205552008284\n"
        "CONTACT: WESSAM MOSTAFA; EMAIL: WMOSTAFA@LEHAA.NET; "
        "PH: +20-237-601-777\n"
    )

    values = locate_auxiliary_values(text)

    assert {(row.category, row.value) for row in values} == {
        ("invoice no", "6123110025"),
        ("acid", "5452479342024010012"),
        ("importer taxation number", "545247934"),
        ("exporter identification number", "0205552008284"),
        ("email", "WMOSTAFA@LEHAA.NET"),
        ("ph", "+20-237-601-777"),
    }


def test_inventory_does_not_anonymize_identifier_owned_by_task_field() -> None:
    source_text = "Batch : PR7XB93\nINVOICE NO. 1123200938 DATED: 27.11.2023\n"
    source_label = {
        "documentPatch": {
            "cargoGroups": [{"additionalInformation": ["Batch : PR7XB93"]}],
            "forwardingAndExportReferences": ["1123200938 27.11.2023"],
        }
    }
    target_label = {
        "documentPatch": {
            "cargoGroups": [{"additionalInformation": ["Batch : QL8KD41"]}],
            "forwardingAndExportReferences": ["6950488277 06.06.2024"],
        }
    }
    work_items = (
        HybridWorkItem(
            workItemId="W0001",
            targetPaths=("documentPatch.cargoGroups[0].additionalInformation[0]",),
            action="replace",
            sourceValue="Batch : PR7XB93",
            targetValue="Batch : QL8KD41",
            state="agent_residual",
            evidenceLineIds=("L00001",),
            spanIds=("S001",),
            locator="exact_literal",
            rationale="test",
        ),
        HybridWorkItem(
            workItemId="W0002",
            targetPaths=("documentPatch.forwardingAndExportReferences[0]",),
            action="replace",
            sourceValue="1123200938 27.11.2023",
            targetValue="6950488277 06.06.2024",
            state="agent_residual",
            evidenceLineIds=("L00002",),
            spanIds=("S002",),
            locator="forwarding_reference_context",
            rationale="test",
        ),
    )

    inventory = build_mutable_inventory(
        source_text=source_text,
        current_text=source_text,
        source_label=source_label,
        target_label=target_label,
        work_items=work_items,
        oracle_case=None,
    )

    deterministic_sources = {
        row.sourceSurface
        for row in inventory
        if row.disposition == "deterministic_shape_replacement"
    }
    assert deterministic_sources.isdisjoint({"PR7XB93", "1123200938"})


def test_locate_auxiliary_values_rejects_measurement_pseudo_tax_field() -> None:
    values = locate_auxiliary_values(
        "REAL CBM: 13.453\n"
        "TAX CBM: 25.630\n"
        "Egyptian New Customs Law No. 207 for the year 2020.\n"
        "Spot Booking Amendment Fee Prepaid PARTY 11311666\n"
    )

    assert values == ()


def test_locate_auxiliary_values_handles_references_purchase_orders_and_box_address() -> None:
    text = (
        "(5) Reference Nos.: 25090004790549 / TLSOE24014490\n"
        "PO#4502324244 & 4 PALLETS\n"
        "P.O. BOX 712 CAIRO\n"
        "P.O.Box:21537\n"
    )

    values = locate_auxiliary_values(text)

    assert {(row.category, row.value) for row in values} == {
        ("reference nos", "25090004790549"),
        ("reference nos", "TLSOE24014490"),
        ("po", "4502324244"),
    }


def test_purchase_order_does_not_claim_following_package_or_equipment_columns() -> None:
    values = locate_auxiliary_values(
        "CSNU8456635 /7025198 PURCH.ORDER SHISHA-PO-015590 96 CARTONS /FCL/FCL /40HQ/\n"
    )

    assert [(row.category, row.value) for row in values] == [("po", "015590")]


def test_locate_auxiliary_values_preserves_contact_and_identifier_surface_shapes() -> None:
    text = (
        "EGYPT VAT 712098623\n"
        "PHONE NUMBER: +20 121 1116 149, +20 121 1116 150\n"
        "Invoice No : VTXL- 23460\n"
    )

    values = locate_auxiliary_values(text)

    assert {(row.category, row.value) for row in values} == {
        ("vat", "712098623"),
        ("phone number", "+20 121 1116 149"),
        ("phone number", "+20 121 1116 150"),
        ("invoice no", "VTXL- 23460"),
    }


def test_locate_auxiliary_values_covers_consolidation_eori_and_cargo_x_id() -> None:
    text = (
        "CONSOLIDATION NUMBER\n"
        "HOURTM2431A1\n"
        "ROTTERDAM, THE NETHERLANDS EORI#\n"
        "NL809653035000\n"
        "*CARGO X ID:9E853AC1-7A97-4BEA-8F4D-\n"
    )

    values = locate_auxiliary_values(text)

    assert {(row.category, row.value, row.line_numbers) for row in values} == {
        ("consolidation number", "HOURTM2431A1", (2,)),
        ("eori", "NL809653035000", (4,)),
        ("cargo x id", "9E853AC1-7A97-4BEA-8F4D", (5,)),
    }


def test_locate_auxiliary_values_covers_repeated_maritime_registration_fields() -> None:
    text = (
        "VAT NUM: ESA60973906\n"
        "R.C.S.: 562 024 422\n"
        "Company Registration No: 91330621MA2JUD752G\n"
        "Carrier's Reference: 72930692 HLCUMTR230512758\n"
        "THERMOGRAPHS: THRM12345 THRM67890\n"
    )

    values = locate_auxiliary_values(text)

    assert {(row.category, row.value) for row in values} == {
        ("vat num", "ESA60973906"),
        ("r.c.s.", "562 024 422"),
        ("company registration no", "91330621MA2JUD752G"),
        ("carrier's reference", "72930692"),
        ("carrier's reference", "HLCUMTR230512758"),
        ("thermographs", "THRM12345"),
        ("thermographs", "THRM67890"),
    }


def test_locate_auxiliary_values_preserves_fmc_oti_caption() -> None:
    values = locate_auxiliary_values("FMC-OTI NO. 024004N\n")

    assert [(row.category, row.value, row.line_numbers) for row in values] == [
        ("fmc-oti no", "024004N", (1,))
    ]


def test_locate_auxiliary_values_covers_trailing_carrier_registration_labels() -> None:
    values = locate_auxiliary_values(
        "562 024 422 R.C.S. Marseille\n562 024 422 Reg. No. Martigny-Ville\n"
    )

    assert {(row.category, row.value, row.line_numbers) for row in values} == {
        ("r.c.s", "562 024 422", (1,)),
        ("reg. no", "562 024 422", (2,)),
    }


def test_standalone_opaque_identifier_locator_excludes_dates_and_measurements() -> None:
    values = locate_standalone_opaque_identifiers(
        "91330621MA2JUD752G\n"
        "IL-02-511574089\n"
        "C41528C68EBE\n"
        "X20240812580136\n"
        "27560.000KGS\n"
        "2024-03-29\n"
        "29-APR-2024\n"
    )

    assert {(row.category, row.value, row.line_numbers) for row in values} == {
        ("unlabelled opaque identifier", "91330621MA2JUD752G", (1,)),
        ("unlabelled opaque identifier", "IL-02-511574089", (2,)),
        ("unlabelled opaque identifier", "C41528C68EBE", (3,)),
        ("unlabelled opaque identifier", "X20240812580136", (4,)),
    }


def test_inventory_does_not_anonymize_standalone_identifier_required_by_target() -> None:
    text = "TARGETREF2024999\n"
    target = {"documentPatch": {"forwardingAndExportReferences": ["TARGETREF2024999"]}}

    inventory = build_mutable_inventory(
        source_text=text,
        current_text=text,
        source_label=target,
        target_label=target,
        work_items=(),
        oracle_case=None,
    )

    assert not any(row.sourceSurface == "TARGETREF2024999" for row in inventory)


def test_inventory_detects_stale_semantics_auxiliary_and_reefer() -> None:
    source, target = _labels()
    text = (
        "--- PAGE 1 ---\n"
        "2600 CARTONS\n"
        "55.000M3\n"
        "CARGO IS STOWED IN A REFRIGERATED CONTAINER AT -3 C\n"
        "ACID NO. 5086429732023040027\n"
    )
    quantity_item = HybridWorkItem(
        workItemId="W0001",
        targetPaths=("documentPatch.cargoPackages[0].quantity",),
        action="replace",
        sourceValue=2600,
        targetValue=325,
        state="agent_residual",
        evidenceLineIds=("L00002",),
        spanIds=("S001",),
        locator="numeric_surface",
        rationale="test path ownership",
    )

    inventory = build_mutable_inventory(
        source_text=text,
        current_text=text,
        source_label=source,
        target_label=target,
        work_items=(quantity_item,),
        oracle_case=None,
    )

    categories = {row.category for row in inventory}
    assert "changed_source_occurrence" in categories
    assert "source_only_auxiliary" in categories
    assert "shipment_dependent_reefer" in categories


def test_inventory_detects_changed_negative_temperature_surface() -> None:
    source, target = _labels()
    target_patch = target["documentPatch"]
    assert isinstance(target_patch, dict)
    target_patch["containers"] = [
        {
            "containerNumber": "FSCU4850130",
            "temperatureSetpoint": {"unit": "celsius", "value": -22.5},
        }
    ]
    text = "--- PAGE 1 ---\nAT THE CARRYING TEMPERATURE OF -3\nDEGREES CELSIUS.\n"
    work_item = HybridWorkItem(
        workItemId="W0001",
        targetPaths=("documentPatch.containers[0].temperatureSetpoint.value",),
        action="replace",
        sourceValue=-3.0,
        targetValue=-22.5,
        state="agent_residual",
        evidenceLineIds=("L00002",),
        spanIds=("S001",),
        locator="numeric_surface",
        rationale="test path ownership",
    )

    inventory = build_mutable_inventory(
        source_text=text,
        current_text=text,
        source_label=source,
        target_label=target,
        work_items=(work_item,),
        oracle_case=None,
    )

    changed = [row for row in inventory if row.category == "changed_source_occurrence"]
    assert len(changed) == 1
    assert changed[0].sourceSurface == "-3.0"
    assert changed[0].lineIds == ("L00002",)


def test_numeric_inventory_never_owns_page_markers_or_unrelated_equal_numbers() -> None:
    source, target = _labels()
    source_patch = source["documentPatch"]
    target_patch = target["documentPatch"]
    assert isinstance(source_patch, dict)
    assert isinstance(target_patch, dict)
    source_patch["containers"] = [
        {
            "containerNumber": "FSCU5904400",
            "temperatureSetpoint": {"unit": "celsius", "value": 1.0},
        }
    ]
    target_patch["containers"] = [
        {
            "containerNumber": "FSCU4850130",
            "temperatureSetpoint": {"unit": "celsius", "value": -21.0},
        }
    ]
    work_item = HybridWorkItem(
        workItemId="W0001",
        targetPaths=("documentPatch.containers[0].temperatureSetpoint.value",),
        action="replace",
        sourceValue=1.0,
        targetValue=-21.0,
        state="agent_residual",
        evidenceLineIds=("L00003",),
        spanIds=("S001",),
        locator="numeric_surface",
        rationale="test path ownership",
    )
    text = "--- PAGE 1 ---\n1 ORIGINAL BILL\nSET TEMPERATURE 1 C\n"

    inventory = build_mutable_inventory(
        source_text=text,
        current_text=text,
        source_label=source,
        target_label=target,
        work_items=(work_item,),
        oracle_case=None,
    )

    changed = [row for row in inventory if row.category == "changed_source_occurrence"]
    assert len(changed) == 1
    assert changed[0].lineIds == ("L00003",)


def test_deterministic_auxiliary_edit_preserves_shape_and_repetition() -> None:
    source, target = _labels()
    text = "--- PAGE 1 ---\nACID NO. 5086429732023040027\n** ACID:#5086429732023040027#\n"
    inventory = build_mutable_inventory(
        source_text=text,
        current_text=text,
        source_label=source,
        target_label=target,
        work_items=(),
        oracle_case=None,
    )

    output, edits = apply_deterministic_auxiliary_edits(
        text=text,
        document_id="doc_" + "1" * 64,
        scenario_id="scenario-1",
        candidates=inventory,
    )

    assert len(edits) == 1
    assert edits[0].sourceSurface == "5086429732023040027"
    assert edits[0].targetSurface != edits[0].sourceSurface
    assert edits[0].targetSurface.isdigit()
    assert output.count(edits[0].targetSurface) == 2
    assert "5086429732023040027" not in output
    assert len(output) == len(text)


def test_deterministic_auxiliary_edit_preserves_embedded_identifier_relationship() -> None:
    source, target = _labels()
    text = "--- PAGE 1 ---\nIMPORTER TAX ID: 508642973\nACID NO. 5086429732023040027\n"
    inventory = build_mutable_inventory(
        source_text=text,
        current_text=text,
        source_label=source,
        target_label=target,
        work_items=(),
        oracle_case=None,
    )

    output, edits = apply_deterministic_auxiliary_edits(
        text=text,
        document_id="doc_" + "3" * 64,
        scenario_id="scenario-linked",
        candidates=inventory,
    )

    replacements = {row.sourceSurface: row.targetSurface for row in edits}
    assert set(replacements) == {"508642973", "5086429732023040027"}
    assert replacements["5086429732023040027"].startswith(replacements["508642973"])
    assert "508642973" not in output
    assert len(output) == len(text)


def test_deterministic_auxiliary_edit_is_confined_to_inventoried_lines() -> None:
    source = "MSCU"
    text = "--- PAGE 1 ---\nSCAC CODE: MSCU\nCONTAINER: MSCU7477141\n"
    candidate = InventoryCandidate(
        candidateId="I0001",
        category="source_only_auxiliary",
        sourceSurface=source,
        lineIds=("L00002",),
        targetPaths=(),
        targetSemantics={"policy": "synthesize_distinct_value"},
        disposition="deterministic_shape_replacement",
        rationale="The SCAC value is source-only operational identity.",
    )

    output, edits = apply_deterministic_auxiliary_edits(
        text=text,
        document_id="doc_" + "4" * 64,
        scenario_id="scenario-line-authority",
        candidates=(candidate,),
    )

    assert len(edits) == 1
    assert edits[0].lineIds == ("L00002",)
    assert f"SCAC CODE: {edits[0].targetSurface}" in output
    assert "CONTAINER: MSCU7477141" in output


def test_full_document_audit_rejects_false_pass_and_accepts_reconciled_output() -> None:
    source, target = _labels()
    source_text = (
        "--- PAGE 1 ---\n"
        "2600 CARTONS\n"
        "CARGO IS STOWED IN A REFRIGERATED CONTAINER AT -3 C\n"
        "ACID NO. 5086429732023040027\n"
    )
    oracle = RegressionCase.model_validate(
        {
            "documentId": "doc_" + "2" * 64,
            "assertions": (
                {
                    "assertionId": "known-gap",
                    "category": "test",
                    "sourceSurfacesMustDisappear": ("2600 CARTONS",),
                    "instruction": "replace stale total",
                },
            ),
        },
        strict=True,
    )
    quantity_item = HybridWorkItem(
        workItemId="W0001",
        targetPaths=("documentPatch.cargoPackages[0].quantity",),
        action="replace",
        sourceValue=2600,
        targetValue=325,
        state="agent_residual",
        evidenceLineIds=("L00002",),
        spanIds=("S001",),
        locator="numeric_surface",
        rationale="test path ownership",
    )
    inventory = build_mutable_inventory(
        source_text=source_text,
        current_text=source_text,
        source_label=source,
        target_label=target,
        work_items=(quantity_item,),
        oracle_case=oracle,
    )
    failed = audit_full_document(
        document_id=oracle.documentId,
        source_text=source_text,
        output_text=source_text,
        source_label=source,
        target_label=target,
        inventory=inventory,
        deterministic_edits=(),
        oracle_case=oracle,
        work_items=(quantity_item,),
    )
    assert failed.passed is False
    assert {row.category for row in failed.findings} >= {
        "retained_changed_source_value",
        "retained_source_auxiliary",
        "reefer_state_contradiction",
        "regression_oracle_failure",
    }

    deterministic, edits = apply_deterministic_auxiliary_edits(
        text=source_text,
        document_id=oracle.documentId,
        scenario_id="scenario-2",
        candidates=inventory,
    )
    corrected = deterministic.replace("2600 CARTONS", "325 PACKAGES").replace(
        "CARGO IS STOWED IN A REFRIGERATED CONTAINER AT -3 C",
        "CARGO IS STOWED IN A STANDARD DRY CONTAINER AMBIENT",
    )
    passed = audit_full_document(
        document_id=oracle.documentId,
        source_text=source_text,
        output_text=corrected,
        source_label=source,
        target_label=target,
        inventory=inventory,
        deterministic_edits=edits,
        oracle_case=oracle,
        work_items=(quantity_item,),
    )
    assert passed.passed is True


def test_pinned_regression_oracle_is_strict_and_unique() -> None:
    path = Path("data/evaluation/mpci-bl-raw-text-rewrite-regression-v1.json")
    oracle = parse_regression_oracle(json.loads(path.read_text(encoding="utf-8")))

    assert len(oracle.cases) == 12
    assert len({row.documentId for row in oracle.cases}) == 12
    assert all(row.assertions for row in oracle.cases)


def test_historical_oracle_does_not_forbid_topology_preserved_reefer_text() -> None:
    source, target = _labels()
    target_patch = target["documentPatch"]
    assert isinstance(target_patch, dict)
    target_patch["containers"] = [
        {
            "containerNumber": "FSCU4850130",
            "temperatureSetpoint": {"unit": "celsius", "value": -22.5},
        }
    ]
    surface = "CARGO IS STOWED IN A REFRIGERATED"
    oracle = RegressionCase.model_validate(
        {
            "documentId": "doc_" + "b" * 64,
            "assertions": (
                {
                    "assertionId": "old-ambient-target",
                    "category": "test",
                    "sourceSurfacesMustDisappear": (surface,),
                    "instruction": "historical target removed refrigeration",
                },
            ),
        },
        strict=True,
    )
    inventory = build_mutable_inventory(
        source_text=surface,
        current_text=surface,
        source_label=source,
        target_label=target,
        work_items=(),
        oracle_case=oracle,
    )

    assert not any(row.category == "regression_oracle_gap" for row in inventory)


def test_historical_oracle_preserves_unlabeled_anonymous_equipment_topology() -> None:
    surface = "/40\N{RIGHT SINGLE QUOTATION MARK} HC Containers Said to Contain"
    label = {"documentPatch": {"containers": None}}
    oracle = RegressionCase.model_validate(
        {
            "documentId": "doc_" + "c" * 64,
            "assertions": (
                {
                    "assertionId": "old-label-only-target",
                    "category": "test",
                    "sourceSurfacesMustDisappear": (surface,),
                    "instruction": "historical target omitted anonymous equipment",
                },
            ),
        },
        strict=True,
    )
    inventory = build_mutable_inventory(
        source_text=surface,
        current_text=surface,
        source_label=label,
        target_label=label,
        work_items=(),
        oracle_case=oracle,
    )

    assert not any(row.category == "regression_oracle_gap" for row in inventory)


def test_inventory_does_not_reclassify_anonymous_equipment_as_package_total() -> None:
    source = {
        "documentPatch": {
            "cargoPackages": [
                {
                    "groupId": "g1",
                    "packageId": "p1",
                    "quantity": 30,
                    "typeCategory": "PACKAGE_PACKAGE",
                }
            ],
            "containers": None,
        }
    }
    target = {
        "documentPatch": {
            "cargoPackages": [
                {
                    "groupId": "g1",
                    "packageId": "p1",
                    "quantity": 9,
                    "typeCategory": "PACKAGE_SKID",
                }
            ],
            "containers": None,
        }
    }
    text = (
        "--- PAGE 1 ---\n"
        "2\n\nQuantity and Kind of Packages\n\n"
        "/40\N{RIGHT SINGLE QUOTATION MARK} HC Containers Said to Contain\n\n"
        "TOTAL : 30 PACKAGES\n"
    )

    inventory = build_mutable_inventory(
        source_text=text,
        current_text=text,
        source_label=source,
        target_label=target,
        work_items=(),
        oracle_case=None,
    )

    assert not any(
        row.category == "shipment_dependent_aggregate" and "40" in row.sourceSurface
        for row in inventory
    )


def test_inventory_preserves_compact_anonymous_cntr_summary_without_target_containers() -> None:
    source = {
        "documentPatch": {
            "cargoPackages": [
                {
                    "groupId": "g1",
                    "packageId": "p1",
                    "quantity": 10,
                    "typeCategory": "PACKAGE_CARTON",
                }
            ],
            "containers": None,
        }
    }
    target = {
        "documentPatch": {
            "cargoPackages": [
                {
                    "groupId": "g1",
                    "packageId": "p1",
                    "quantity": 5,
                    "typeCategory": "PACKAGE_CARTON",
                }
            ],
            "containers": None,
        }
    }
    text = "--- PAGE 1 ---\nTOTAL: 1X40'HC LCL CNTR(S)\nTOTAL: 10 CARTONS\n"

    inventory = build_mutable_inventory(
        source_text=text,
        current_text=text,
        source_label=source,
        target_label=target,
        work_items=(),
        oracle_case=None,
    )

    assert not any(
        row.category == "shipment_dependent_aggregate" and "CNTR(S)" in row.sourceSurface
        for row in inventory
    )


@pytest.mark.parametrize(
    ("line", "expected"),
    [
        ("TOTALS: 4", True),
        ("Total Gross Weight: 25574.100 Kgs.", True),
        ("1 Container Said to Contain 1040 BAGS", True),
        ("CARRIER'S RECEIPT (see Clause 14.1) 3 container(s)", True),
        (
            "Carrier's Receipt (see clause 1 and 14). Total number of containers or packages "
            "received by Carrier.",
            False,
        ),
        ("(16) Carrier's Receipt", False),
        ("(22) TOTAL NUMBER OF", False),
        ("MEASUREMENT (20)", False),
        (
            "8.6 The aggregate liability of the Freight Forwarder shall not exceed the "
            "limits of liability for total loss of the goods.",
            False,
        ),
        (
            "SHIPPED in apparent good order, the total number or quantity of Containers or "
            "packages indicated in the Carrier's Receipt subject to all terms.",
            False,
        ),
    ],
)
def test_shipment_aggregate_detector_requires_populated_field_grammar(
    line: str, expected: bool
) -> None:
    assert _is_shipment_aggregate_value_line(line) is expected


def test_attached_page_count_is_document_topology_not_shipment_aggregate() -> None:
    assert _is_shipment_aggregate_value_line("TOTAL NUMBER OF ATTACHED 1 PAGE") is False
    assert _is_shipment_aggregate_value_line("TOTAL NUMBER OF ATTACHED PAGES: 2") is False


def test_carrier_receipt_count_is_preserved_only_when_target_container_count_matches() -> None:
    line = "CARRIER'S RECEIPT (see Clause 14.1) 3 container(s)"
    target = {
        "documentPatch": {
            "containers": [
                {"containerNumber": "AAAA000001"},
                {"containerNumber": "AAAA000002"},
                {"containerNumber": "AAAA000003"},
            ]
        }
    }

    assert _carrier_receipt_count_matches_target(line, target) is True
    assert (
        _carrier_receipt_count_matches_target(
            line, {"documentPatch": {"containers": target["documentPatch"]["containers"][:2]}}
        )
        is False
    )
    assert (
        _carrier_receipt_count_matches_target(line, {"documentPatch": {"containers": None}})
        is False
    )


@pytest.mark.parametrize(
    ("line", "following"),
    [
        ("TOTAL NUMBER OF CONTAINERS RECEIVED BY THE CARRIER: 3", None),
        ("Total No. of Containers received by the Carrier: 3", None),
        ("Weight in Kgs Total: 3 Container(s)", None),
        ("Total number of containers or packages 3", "received by Carrier:"),
    ],
)
def test_carrier_receipt_count_recognizes_exact_observed_grammars(
    line: str, following: str | None
) -> None:
    target = {
        "documentPatch": {
            "containers": [
                {"containerNumber": "AAAA000001"},
                {"containerNumber": "AAAA000002"},
                {"containerNumber": "AAAA000003"},
            ]
        }
    }

    assert _carrier_receipt_count_matches_target(line, target, following_line=following) is True


def test_carrier_receipt_count_recognizes_acknowledged_container_package_grammar() -> None:
    line = (
        "TOTAL NO. OF CONTAINERS/PACKAGES RECEIVED & ACKNOWLEDGED BY CARRIER "
        "FOR THE PURPOSE OF CALCULATION OF PACKAGE LIMITATION (IF APPLICABLE): "
        "10 CONTAINER(S)/PACKAGE(S)"
    )
    target = {
        "documentPatch": {
            "containers": [{"containerNumber": f"AAAA{index:06d}"} for index in range(1, 11)]
        }
    }

    assert _carrier_receipt_count_matches_target(line, target) is True
    assert (
        _carrier_receipt_count_matches_target(
            line,
            {"documentPatch": {"containers": target["documentPatch"]["containers"][:-1]}},
        )
        is False
    )


def test_split_carrier_receipt_count_requires_its_continuation_heading() -> None:
    target = {"documentPatch": {"containers": [{"containerNumber": "AAAA000001"}]}}

    assert not _carrier_receipt_count_matches_target(
        "Total number of containers or packages 1",
        target,
        following_line="unrelated cargo text",
    )


@pytest.mark.parametrize(
    ("line", "target_weight", "expected_candidate"),
    [
        ("Cargo Gross Weight (4) 9097 KG", 9096.8, False),
        ("TOTAL GROSS WEIGHT: 57,072.40 KGS", 57072.4, False),
        ("TOTAL GROSS WEIGHT: 57,073.40 KGS", 57072.4, True),
        ("TOTAL GROSS WEIGHT: 57,072.40 KGS 944 CARTONS", 57072.4, True),
    ],
)
def test_inventory_only_rewrites_weight_aggregate_when_target_disagrees(
    line: str, target_weight: float, expected_candidate: bool
) -> None:
    source = {
        "documentPatch": {
            "cargoGroups": [
                {
                    "groupId": "g1",
                    "description": "SOURCE GOODS",
                    "grossWeight": {"unit": "kilogram", "value": 1000.0},
                }
            ]
        }
    }
    target = {
        "documentPatch": {
            "cargoGroups": [
                {
                    "groupId": "g1",
                    "description": "TARGET GOODS",
                    "grossWeight": {"unit": "kilogram", "value": target_weight},
                }
            ]
        }
    }

    inventory = build_mutable_inventory(
        source_text=line + "\n",
        current_text=line + "\n",
        source_label=source,
        target_label=target,
        work_items=(),
        oracle_case=None,
    )

    observed = any(row.category == "shipment_dependent_aggregate" for row in inventory)
    assert observed is expected_candidate


@pytest.mark.parametrize(
    ("line", "target_volume", "expected_candidate"),
    [
        ("14.600CBM", 14.6, False),
        ("MEASUREMENT: 14.600 CBM", 14.6, False),
        ("6.875M3", 14.6, True),
        ("MEASUREMENT: 6.875 CBM", 14.6, True),
    ],
)
def test_inventory_only_rewrites_volume_aggregate_when_target_disagrees(
    line: str, target_volume: float, expected_candidate: bool
) -> None:
    source = {
        "documentPatch": {
            "cargoGroups": [
                {
                    "groupId": "g1",
                    "description": "SOURCE GOODS",
                    "volume": {"unit": "cubic_metre", "value": 6.875},
                }
            ]
        }
    }
    target = {
        "documentPatch": {
            "cargoGroups": [
                {
                    "groupId": "g1",
                    "description": "TARGET GOODS",
                    "volume": {"unit": "cubic_metre", "value": target_volume},
                }
            ]
        }
    }

    inventory = build_mutable_inventory(
        source_text=line + "\n",
        current_text=line + "\n",
        source_label=source,
        target_label=target,
        work_items=(),
        oracle_case=None,
    )

    observed = any(row.category == "shipment_dependent_aggregate" for row in inventory)
    assert observed is expected_candidate


def test_inventory_does_not_reopen_a_target_rendered_receipt_aggregate() -> None:
    source = {
        "documentPatch": {"cargoPackages": [{"groupId": "g1", "packageId": "p1", "quantity": 12}]}
    }
    target = {
        "documentPatch": {"cargoPackages": [{"groupId": "g1", "packageId": "p1", "quantity": 5}]}
    }
    source_text = "Total number of containers or packages 12 received by Carrier:\n"
    current_text = "Total number of containers or packages 5 received by Carrier:\n"
    requirement = SurfaceRenderingRequirement(
        kind="carrier_receipt_count",
        targetPath="documentPatch.cargoPackages",
        sourceSurface="12",
        targetSurface="5",
        sourceOccurrences=1,
        contextEvidence=source_text.rstrip("\n"),
    )

    inventory = build_mutable_inventory(
        source_text=source_text,
        current_text=current_text,
        source_label=source,
        target_label=target,
        work_items=(),
        oracle_case=None,
        surface_requirements=(requirement,),
    )

    assert not any(row.category == "shipment_dependent_aggregate" for row in inventory)


def test_numeric_evidence_excludes_dates_and_legal_clause_numbers() -> None:
    text = (
        "Onward routing (see clause 1)\n"
        "SHIPPED ON BOARD 2024-01-07\n"
        "Carrier's Receipt (see clause 1 and 14)\n"
        "1 Container Said to Contain 1 PIECE\n"
    )
    item = HybridWorkItem(
        workItemId="W0001",
        targetPaths=("documentPatch.cargoPackages[0].quantity",),
        action="replace",
        sourceValue=1,
        targetValue=40,
        state="agent_residual",
        evidenceLineIds=("L00001", "L00002", "L00003", "L00004"),
        spanIds=("S001",),
        locator="numeric_surface",
        rationale="global numeric evidence",
    )
    label = {
        "documentPatch": {
            "cargoPackages": [{"groupId": "g1", "packageId": "p1", "quantity": 1}],
            "cargoGroups": [{"groupId": "g1", "description": "GOODS"}],
        }
    }
    bundle = SimpleNamespace(
        sourceLabel=label,
        targetLabel=label,
        cargoFlavorRewriteRequirements=(),
        operationalFlavorRequirements=(),
    )

    refined = _refine_numeric_evidence(text, (item,), bundle)

    assert refined[0].evidenceLineIds == ("L00004",)
    assert refined[0].locator == "relation_scoped_numeric_surface"


def test_numeric_refinement_never_reopens_a_deterministically_applied_quantity() -> None:
    item = HybridWorkItem(
        workItemId="W0001",
        targetPaths=("documentPatch.cargoAllocationGroups[0].allocations[0].packageQuantity",),
        action="replace",
        sourceValue=3,
        targetValue=53,
        state="deterministic_applied",
        evidenceLineIds=(),
        spanIds=(),
        locator="deterministic_requirement",
        rationale="already rendered by an anchored scalar requirement",
    )
    bundle = SimpleNamespace(
        sourceLabel={},
        targetLabel={},
        cargoFlavorRewriteRequirements=(),
        operationalFlavorRequirements=(),
    )

    refined = _refine_numeric_evidence("53 PALLETS\n", (item,), bundle)

    assert refined == (item,)


def test_model_slot_locks_flattened_equipment_role_surface() -> None:
    text = (
        "--- PAGE 1 ---\n2\n\nQuantity and Kind of Packages\n\n/40' HC Containers Said to Contain\n"
    )
    target = {"documentPatch": {"containers": None}}
    workspace = RewriteWorkspace(
        original_text=text,
        current_text=text,
        source_role_hints=source_semantic_role_hints(text, target),
    )

    locked = _locked_literal_requirements_by_line(workspace)

    assert locked["L00006"] == [
        {
            "kind": "host_locked_source_literal",
            "paths": [
                "documentPatch.cargoPackages[].typeCategory",
                "documentPatch.cargoGroups[].marksAndNumbers",
            ],
            "target": "/40' HC Containers Said to Contain",
            "policy": "preserve_exact_flattened_ocr_semantic_role_on_this_line",
        }
    ]


def test_inventory_batch_config_pins_cost_first_fallback_and_retry_schedule() -> None:
    config = load_synthesis_raw_text_inventory_batch_config(
        Path("configs/synthesis/mpci_bl_raw_text_inventory50_topology_glm53_native_v2.yaml")
    )

    assert config.workflow.documents == 50
    assert config.workflow.max_concurrent_documents == 8
    assert config.provider.provider_order == (
        "deepinfra/fp4",
        "fireworks",
        "nextbit/fp8",
    )
    assert config.workflow.max_successful_model_responses_per_document == 3
    first = _retry_delay_seconds(
        document_id="doc_" + "1" * 64,
        route_round=2,
        config=config,
    )
    assert 5.0 <= first <= 7.0
    assert first == _retry_delay_seconds(
        document_id="doc_" + "1" * 64,
        route_round=2,
        config=config,
    )
    assert (
        _retry_delay_seconds(
            document_id="doc_" + "1" * 64,
            route_round=1,
            config=config,
        )
        == 0.0
    )


def test_inventory_batch_config_allows_bounded_four_response_recovery() -> None:
    config = load_synthesis_raw_text_inventory_batch_config(
        Path(
            "configs/synthesis/"
            "mpci_bl_raw_text_inventory2_additional_maritime_v27_hard_recovery_glm53.yaml"
        )
    )

    assert config.workflow.max_successful_model_responses_per_document == 4


def test_inventory_route_retry_distinguishes_transient_from_static_incompatibility() -> None:
    overloaded = ModelHTTPError(429, "z-ai/glm-5.3-flash", {"message": "overloaded"})
    incompatible = ModelHTTPError(
        404,
        "z-ai/glm-5.3-flash",
        {
            "metadata": {
                "failed_routing_step": "Filter by Parameters",
            }
        },
    )

    assert _transient_route_error(overloaded) is True
    assert _transient_route_error(ModelAPIError("z-ai/glm-5.3-flash", "Connection error.")) is True
    assert _transient_route_error(incompatible) is False


def test_inventory_checkpoint_restores_commit_receipt_for_deterministic_replay(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    document_id = "doc_" + "a" * 64
    deterministic_text = "SOURCE AGENT\n"
    final_text = "FICTIONAL AGENT\n"
    deterministic_sha = sha256_bytes(deterministic_text.encode())
    final_sha = sha256_bytes(final_text.encode())
    target_sha = sha256_bytes(canonical_json_bytes({}))
    workspace = RewriteWorkspace(
        original_text=deterministic_text,
        current_text=final_text,
        current_target_label={},
    )
    commit = AtomicRewriteCommit(
        beforeTextSha256=deterministic_sha,
        afterTextSha256=final_sha,
        beforeTargetLabelSha256=target_sha,
        afterTargetLabelSha256=target_sha,
        compoundPartyFlavorRealizations=(),
        rawAuxiliaryIdentityRealizations=(),
        appliedReplacements=(),
    )
    workspace.commits.append(commit)
    compiled = SimpleNamespace(
        workspace=workspace,
        bundle=SimpleNamespace(result=SimpleNamespace(documentId=document_id)),
    )
    material = SimpleNamespace(
        compiled=compiled,
        deterministic_text_sha256=deterministic_sha,
    )
    empty_usage = _empty_usage()
    result = InventoryProbeCaseResult(
        documentId=document_id,
        status="needs_review",
        reason="test result",
        sourceLines=1,
        modelLines=1,
        compilerWorkItems=1,
        inventoryCandidates=1,
        deterministicInventoryEdits=0,
        hostRewriteAuditPassed=True,
        legacyResidualCandidates=0,
        fullDocumentAudit=FullDocumentAudit(
            documentId=document_id,
            sourceTextSha256=deterministic_sha,
            outputTextSha256=final_sha,
            inventorySha256="b" * 64,
            candidates=1,
            deterministicCandidates=0,
            residualCandidates=1,
            findings=(),
            passed=False,
        ),
        outputTextSha256=final_sha,
        usage=empty_usage,
    )
    staged = StagedArtifactRun(
        output_parent=tmp_path,
        run_name="checkpoint-receipt-test",
        transaction_sha256="c" * 64,
    )
    _publish_inventory_checkpoint(
        staged=staged,
        material=material,
        result=result,
        stages=(),
    )
    checkpoint = json.loads(
        (staged.stage_root / f"cases/{document_id}/checkpoint.json").read_text()
    )
    assert checkpoint["schemaVersion"] == 2
    assert checkpoint["commits"] == [commit.model_dump(mode="json")]

    workspace.current_text = deterministic_text
    workspace.commits.clear()

    def reproduced_result(**_: object) -> object:
        assert workspace.current_text == final_text
        assert workspace.commits == [commit]
        return result

    monkeypatch.setattr(
        "document_ocr.synthesis.raw_text_inventory_probe._inventory_case_result",
        reproduced_result,
    )
    loaded = _load_inventory_checkpoint(staged=staged, material=material)

    assert loaded == (result, ())
    assert result.usage.estimatedCostUsd == Decimal("0")


def test_dynamic_patch_schema_requires_every_slot_and_forbids_extra_keys() -> None:
    slots = (
        _Slot(alias="s0", line_id="L00001", source_line="A", requirements=()),
        _Slot(alias="s1", line_id="L00002", source_line="B", requirements=()),
    )
    _output_type, schema = _output_schema(slots)

    assert schema["required"] == ["s0", "s1"]
    assert schema["additionalProperties"] is False
    assert [
        row.newText for row in _validated_output_replacements(slots, {"s0": "X", "s1": "Y"})
    ] == [
        "X",
        "Y",
    ]
    with pytest.raises(ValueError, match="complete required slot set"):
        _validated_output_replacements(slots, {"s0": "X"})
    with pytest.raises(ValueError, match="complete required slot set"):
        _validated_output_replacements(slots, {"s0": "X", "s1": "Y", "s2": "Z"})
    with pytest.raises(ValueError, match="contains a newline"):
        _validated_output_replacements(slots, {"s0": "X\nY", "s1": "Y"})
    with pytest.raises(ValueError, match="degenerates a lexical line"):
        _validated_output_replacements(slots, {"s0": ".", "s1": "Y"})
    with pytest.raises(ValueError, match=r"not populated strings: s0, s1") as error:
        _validated_output_replacements(slots, {"s0": "", "s1": "   "})
    selected, compounds = _repair_selection(slots, (), error_message=str(error.value))
    assert tuple(row.alias for row in selected) == ("s0", "s1")
    assert compounds == ()


def test_repeated_identical_auxiliary_lines_require_one_consistent_realization() -> None:
    requirement = {
        "kind": "changed_source_auxiliary_copy",
        "candidateId": "I0042",
        "source": "MAERSK TAIWAN LTD - TAIPEI",
        "target": {
            "policy": "synthesize_distinct_context_compatible_auxiliary_value",
        },
    }
    slots = (
        _Slot(
            alias="s0",
            line_id="L00003",
            source_line="MAERSK TAIWAN LTD - TAIPEI",
            requirements=(requirement,),
        ),
        _Slot(
            alias="s1",
            line_id="L00019",
            source_line="MAERSK TAIWAN LTD - TAIPEI",
            requirements=(requirement,),
        ),
    )

    replacements = _validated_output_replacements(
        slots,
        {
            "s0": "FORMOSA MARITIME LTD - TAIPEI",
            "s1": "FORMOSA MARITIME LTD - TAIPEI",
        },
    )
    assert len(replacements) == 2
    with pytest.raises(ValueError, match="inconsistent fictional realizations"):
        _validated_output_replacements(
            slots,
            {
                "s0": "FORMOSA MARITIME LTD - TAIPEI",
                "s1": "PACIFIC AGENCY LTD - TAIPEI",
            },
        )


def test_initial_request_partition_keeps_party_dependencies_together() -> None:
    slots = (
        _Slot(
            alias="s0",
            line_id="L00001",
            source_line="OLD SHIPPER",
            requirements=(
                {
                    "kind": "task_label_delta",
                    "workItemId": "W0001",
                    "paths": ["documentPatch.parties.shipper.name"],
                },
            ),
        ),
        _Slot(
            alias="s1",
            line_id="L00002",
            source_line="OLD ADDRESS",
            requirements=(
                {
                    "kind": "task_label_delta",
                    "workItemId": "W0002",
                    "paths": ["documentPatch.parties.shipper.address"],
                },
            ),
        ),
        _Slot(
            alias="s2",
            line_id="L00005",
            source_line="OLD CARGO",
            requirements=(
                {
                    "kind": "task_label_delta",
                    "workItemId": "W0003",
                    "paths": ["documentPatch.cargoGroups[0].description"],
                },
            ),
        ),
        _Slot(alias="s3", line_id="L00009", source_line="AUX A", requirements=()),
        _Slot(alias="s4", line_id="L00010", source_line="AUX B", requirements=()),
    )
    compound = _CompoundSlot(
        alias="c0",
        requirement=CompoundPartyFlavorRequirement(
            targetPath="documentPatch.parties.shipper.name",
            relationships=("trading_as",),
            sourceLabelName="OLD SHIPPER",
            targetPrimaryName="NEW SHIPPER",
        ),
    )
    material = SimpleNamespace(
        slots=slots,
        compound_slots=(compound,),
        compiled=SimpleNamespace(
            work_items=(),
            workspace=SimpleNamespace(
                cargo_flavor_rewrite_requirements=(),
                raw_auxiliary_identity_requirements=(),
            ),
        ),
    )

    batches = _initial_request_batches(material, maximum_slots=3)

    assert [tuple(slot.alias for slot in batch.slots) for batch in batches] == [
        ("s0", "s1", "s2"),
        ("s3", "s4"),
    ]
    assert tuple(slot.alias for slot in batches[0].compound_slots) == ("c0",)
    assert batches[1].compound_slots == ()


def test_initial_request_partition_keeps_inventory_candidate_together() -> None:
    slots = (
        _Slot(
            alias="s0",
            line_id="L00001",
            source_line="AUXILIARY OFFICE",
            requirements=(
                {
                    "kind": "changed_source_auxiliary_copy",
                    "candidateId": "I0042",
                },
            ),
        ),
        _Slot(
            alias="s1",
            line_id="L00002",
            source_line="UNRELATED",
            requirements=(),
        ),
        _Slot(
            alias="s2",
            line_id="L00003",
            source_line="AUXILIARY OFFICE",
            requirements=(
                {
                    "kind": "changed_source_auxiliary_copy",
                    "candidateId": "I0042",
                },
            ),
        ),
    )
    material = SimpleNamespace(
        slots=slots,
        compound_slots=(),
        compiled=SimpleNamespace(
            work_items=(),
            workspace=SimpleNamespace(
                cargo_flavor_rewrite_requirements=(),
                raw_auxiliary_identity_requirements=(),
            ),
        ),
    )

    batches = _initial_request_batches(material, maximum_slots=2)

    assert [tuple(slot.alias for slot in batch.slots) for batch in batches] == [
        ("s0", "s2"),
        ("s1",),
    ]


def test_initial_request_partition_does_not_join_independent_fixed_path_occurrences() -> None:
    slots = (
        _Slot(
            alias="s0",
            line_id="L00001",
            source_line="FIRST COPY",
            requirements=(
                {
                    "kind": "task_label_delta",
                    "workItemId": "W0001",
                    "paths": ["documentPatch.cargoGroups[0].description"],
                },
            ),
        ),
        _Slot(
            alias="s1",
            line_id="L00002",
            source_line="SECOND COPY",
            requirements=(
                {
                    "kind": "task_label_delta",
                    "workItemId": "W0002",
                    "paths": ["documentPatch.cargoGroups[0].description"],
                },
            ),
        ),
    )
    material = SimpleNamespace(
        slots=slots,
        compound_slots=(),
        compiled=SimpleNamespace(
            work_items=(),
            workspace=SimpleNamespace(
                cargo_flavor_rewrite_requirements=(),
                raw_auxiliary_identity_requirements=(),
            ),
        ),
    )

    batches = _initial_request_batches(material, maximum_slots=1)

    assert [tuple(slot.alias for slot in batch.slots) for batch in batches] == [
        ("s0",),
        ("s1",),
    ]


def test_initial_request_partition_rejects_an_oversized_semantic_component() -> None:
    slots = tuple(
        _Slot(
            alias=f"s{index}",
            line_id=f"L{index + 1:05d}",
            source_line=f"PARTY {index}",
            requirements=(
                {
                    "kind": "task_label_delta",
                    "paths": ["documentPatch.parties.shipper.name"],
                },
            ),
        )
        for index in range(4)
    )
    material = SimpleNamespace(
        slots=slots,
        compound_slots=(),
        compiled=SimpleNamespace(
            work_items=(),
            workspace=SimpleNamespace(
                cargo_flavor_rewrite_requirements=(),
                raw_auxiliary_identity_requirements=(),
            ),
        ),
    )

    with pytest.raises(ValueError, match="semantic line component exceeds"):
        _initial_request_batches(material, maximum_slots=3)


def test_initial_request_preflight_rejects_partitions_above_response_budget() -> None:
    slots = tuple(
        _Slot(
            alias=f"s{index}",
            line_id=f"L{index + 1:05d}",
            source_line=f"LINE {index}",
            requirements=(),
        )
        for index in range(5)
    )
    material = SimpleNamespace(
        slots=slots,
        compound_slots=(),
        compiled=SimpleNamespace(
            bundle=SimpleNamespace(result=SimpleNamespace(documentId="doc_" + "a" * 64)),
            work_items=(),
            workspace=SimpleNamespace(
                cargo_flavor_rewrite_requirements=(),
                raw_auxiliary_identity_requirements=(),
            ),
        ),
    )

    with pytest.raises(
        ValueError,
        match="initial request partitions exceed the successful-response budget",
    ):
        _preflight_initial_model_contracts(
            (material,),
            maximum_slots=3,
            maximum_successful_responses=1,
            profile_context_lines=0,
        )


def test_repair_selection_is_line_local_but_expands_a_party_role_block() -> None:
    slots = (
        _Slot(
            alias="s0",
            line_id="L00002",
            source_line="SOURCE SHIPPER",
            requirements=(
                {
                    "kind": "task_label_delta",
                    "paths": ["documentPatch.parties.shipper.name"],
                },
            ),
        ),
        _Slot(
            alias="s1",
            line_id="L00003",
            source_line="SOURCE ADDRESS",
            requirements=(
                {
                    "kind": "task_label_delta",
                    "paths": ["documentPatch.parties.shipper.address"],
                },
            ),
        ),
        _Slot(
            alias="s2",
            line_id="L00009",
            source_line="OLD CARGO",
            requirements=(
                {
                    "kind": "task_label_delta",
                    "paths": ["documentPatch.cargoGroups[0].description"],
                },
            ),
        ),
    )
    audit = FullDocumentAudit(
        documentId="doc_" + "d" * 64,
        sourceTextSha256="a" * 64,
        outputTextSha256="b" * 64,
        inventorySha256="c" * 64,
        candidates=0,
        deterministicCandidates=0,
        residualCandidates=0,
        findings=(
            {
                "findingId": "F0001",
                "category": "semantic_slot_mismatch",
                "lineIds": ("L00002",),
                "sourceSurface": "TARGET SHIPPER",
                "targetPaths": ("documentPatch.parties.shipper.name",),
                "explanation": "missing exact target party scalar",
            },
        ),
        passed=False,
    )

    selected, compounds = _repair_selection(slots, (), full_audit=audit)

    assert tuple(row.alias for row in selected) == ("s0", "s1")
    assert compounds == ()


def test_repair_selection_keeps_nonparty_findings_line_local() -> None:
    slots = (
        _Slot(alias="s0", line_id="L00002", source_line="OLD", requirements=()),
        _Slot(alias="s1", line_id="L00003", source_line="OTHER", requirements=()),
    )
    audit = FullDocumentAudit(
        documentId="doc_" + "e" * 64,
        sourceTextSha256="a" * 64,
        outputTextSha256="b" * 64,
        inventorySha256="c" * 64,
        candidates=1,
        deterministicCandidates=0,
        residualCandidates=1,
        findings=(
            {
                "findingId": "F0001",
                "category": "inventory_candidate_unhandled",
                "lineIds": ("L00003",),
                "sourceSurface": "OTHER",
                "targetPaths": (),
                "explanation": "stale candidate",
            },
        ),
        passed=False,
    )

    selected, compounds = _repair_selection(slots, (), full_audit=audit)

    assert tuple(row.alias for row in selected) == ("s1",)
    assert compounds == ()


def test_repair_selection_expands_a_cargo_group_so_coexisting_values_survive() -> None:
    cargo_path = "documentPatch.cargoGroups[0].description"
    slots = (
        _Slot(
            alias="s0",
            line_id="L00002",
            source_line="TARGET DESCRIPTION",
            requirements=({"kind": "cargo", "paths": [cargo_path]},),
        ),
        _Slot(
            alias="s1",
            line_id="L00003",
            source_line="OPEN DOT 6INCH",
            requirements=({"kind": "cargo", "paths": [cargo_path]},),
        ),
        _Slot(
            alias="s2",
            line_id="L00004",
            source_line="NEW AUXILIARY DETAIL",
            requirements=({"kind": "cargo", "paths": [cargo_path]},),
        ),
    )
    error_message = (
        'cargo rewrite failed: {"targetPath": "documentPatch.cargoGroups[0].description", '
        '"unchangedSourceLines": [{"lineId": "L00003", '
        '"sourceSurface": "OPEN DOT 6INCH"}]}'
    )

    selected, compounds = _repair_selection(slots, (), error_message=error_message)

    assert tuple(row.alias for row in selected) == ("s0", "s1", "s2")
    assert compounds == ()


def test_preview_repair_selection_reports_independent_party_and_cargo_defects_together() -> None:
    slots = (
        _Slot(alias="s0", line_id="L00001", source_line="OLD PARTY", requirements=()),
        _Slot(
            alias="s1",
            line_id="L00002",
            source_line="OLD CARGO DETAIL",
            requirements=(
                {
                    "kind": "cargo",
                    "paths": ["documentPatch.cargoGroups[0].description"],
                },
            ),
        ),
    )
    workspace = RewriteWorkspace(
        original_text="OLD PARTY\nOLD CARGO DETAIL\n",
        current_text="OLD PARTY\nOLD CARGO DETAIL\n",
        cargo_flavor_rewrite_requirements=(
            CargoFlavorRewriteRequirement(
                requirementId="cargo-group-1-span-1",
                targetPath="documentPatch.cargoGroups[0].description",
                targetDescription="TARGET CARGO",
                sourceLineIds=("L00002",),
                sourceSurfaces=("OLD CARGO DETAIL",),
            ),
        ),
    )
    material = SimpleNamespace(
        compiled=SimpleNamespace(
            workspace=workspace,
            bundle=SimpleNamespace(
                result=SimpleNamespace(documentId="doc_" + "a" * 64),
                surfaceRenderingRequirements=(),
                anchoredScalarReplacementRequirements=(),
            ),
            work_items=(),
        ),
        slots=slots,
        compound_slots=(),
        inventory=(),
        deterministic_edits=(),
        oracle_case=None,
    )

    selected, compounds, diagnostics, preview_audit = _preview_repair_selection(
        material,
        {"s0": "WRONG PARTY", "s1": "OLD CARGO DETAIL"},
        error_message="party target is missing from s0",
    )

    assert tuple(row.alias for row in selected) == ("s0", "s1")
    assert compounds == ()
    assert any(row.get("kind") == "cargoFlavorRewrite" for row in diagnostics)
    assert preview_audit is not None


def test_preview_repair_selection_reports_missing_target_literal_behind_slot_error() -> None:
    target_path = "documentPatch.cargoGroups[0].additionalInformation[0]"
    slots = (
        _Slot(
            alias="s0",
            line_id="L00001",
            source_line="SOURCE ORIGIN",
            requirements=(
                {
                    "kind": "host_locked_target_literal",
                    "paths": ["documentPatch.cargoGroups[0].origin.name"],
                    "target": "TARGET ORIGIN",
                },
            ),
        ),
        _Slot(
            alias="s1",
            line_id="L00002",
            source_line="OLD PACKING",
            requirements=(
                {
                    "kind": "task_label_delta",
                    "paths": [target_path],
                    "target": "PACKED IN 234 WOODEN CASES",
                },
            ),
        ),
    )
    workspace = RewriteWorkspace(
        original_text="SOURCE ORIGIN\nOLD PACKING\n",
        current_text="SOURCE ORIGIN\nOLD PACKING\n",
        target_literal_requirements=(
            TargetLiteralRequirement(
                targetPath=target_path,
                targetValue="PACKED IN 234 WOODEN CASES",
                matchPolicy="semantic_literal",
            ),
        ),
    )
    material = SimpleNamespace(
        compiled=SimpleNamespace(
            workspace=workspace,
            bundle=SimpleNamespace(
                result=SimpleNamespace(documentId="doc_" + "f" * 64),
                surfaceRenderingRequirements=(),
                anchoredScalarReplacementRequirements=(),
            ),
            work_items=(),
        ),
        slots=slots,
        compound_slots=(),
        inventory=(),
        deterministic_edits=(),
        oracle_case=None,
    )

    selected, compounds, diagnostics, _preview_audit = _preview_repair_selection(
        material,
        {"s0": "WRONG ORIGIN", "s1": "REWRITTEN PACKING"},
        error_message="provider output violates host slot contract: s0",
    )

    assert tuple(row.alias for row in selected) == ("s0", "s1")
    assert compounds == ()
    assert any(row.get("kind") == "missingTargetLiterals" for row in diagnostics)


def test_preview_repair_selection_localizes_an_introduced_source_status_copy() -> None:
    slots = (
        _Slot(
            alias="s0",
            line_id="L00001",
            source_line="HS 123456",
            requirements=(
                {
                    "kind": "task_label_delta",
                    "paths": ["documentPatch.cargoGroups[0].hsCodes[0]"],
                    "target": "654321",
                },
            ),
        ),
        _Slot(
            alias="s1",
            line_id="L00002",
            source_line="OLD CARGO DETAIL",
            requirements=(
                {
                    "kind": "cargo",
                    "paths": ["documentPatch.cargoGroups[0].description"],
                },
            ),
        ),
    )
    workspace = RewriteWorkspace(
        original_text="HS 123456\nOLD CARGO DETAIL\nFREIGHT PREPAID\n",
        current_text="HS 123456\nOLD CARGO DETAIL\nFREIGHT PREPAID\n",
        source_status_requirements=(
            SourceStatusPreservationRequirement(
                sourceSurface="FREIGHT PREPAID",
                sourceOccurrences=1,
            ),
        ),
    )
    material = SimpleNamespace(
        compiled=SimpleNamespace(
            workspace=workspace,
            bundle=SimpleNamespace(
                result=SimpleNamespace(documentId="doc_" + "b" * 64),
                surfaceRenderingRequirements=(),
                anchoredScalarReplacementRequirements=(),
            ),
            work_items=(),
        ),
        slots=slots,
        compound_slots=(),
        inventory=(),
        deterministic_edits=(),
        oracle_case=None,
    )

    selected, compounds, diagnostics, _preview_audit = _preview_repair_selection(
        material,
        {"s0": "HS 654321", "s1": "FREIGHT PREPAID"},
        error_message=("candidate changes an unchanged source status surface: FREIGHT PREPAID"),
    )

    assert tuple(row.alias for row in selected) == ("s1",)
    assert compounds == ()
    assert any(row.get("kind") == "sourceStatusCardinality" for row in diagnostics)


def test_auxiliary_identity_consistency_is_declared_and_repaired_only_on_group_lines() -> None:
    requirement = RawAuxiliaryIdentityRequirement(
        requirementId="carrier-agent-L00002",
        relationship="agent_for_carrier",
        sourceIdentity="OLD AGENT",
        sourceIdentityLineCount=1,
        gapLineCount=0,
        consistencyGroupId="carrier-agent-group-1",
        targetPrincipalName="TARGET CARRIER",
        sourceEvidence="OLD AGENT as agent for TARGET CARRIER",
    )
    second = requirement.model_copy(update={"requirementId": "carrier-agent-L00004"})
    slots = (
        _Slot(alias="s0", line_id="L00001", source_line="UNRELATED", requirements=()),
        _Slot(alias="s1", line_id="L00002", source_line="OLD AGENT", requirements=()),
        _Slot(alias="s2", line_id="L00004", source_line="OLD AGENT", requirements=()),
    )
    workspace = RewriteWorkspace(
        original_text="UNRELATED\nOLD AGENT\nRELATION\nOLD AGENT\n",
        current_text="UNRELATED\nOLD AGENT\nRELATION\nOLD AGENT\n",
        raw_auxiliary_identity_requirements=(requirement, second),
    )
    compiled = SimpleNamespace(
        workspace=workspace,
        work_items=(),
        bundle=SimpleNamespace(
            result=SimpleNamespace(documentId="doc_" + "b" * 64),
            surfaceRenderingRequirements=(),
            anchoredScalarReplacementRequirements=(),
        ),
    )
    material = SimpleNamespace(
        compiled=compiled,
        slots=slots,
        compound_slots=(),
        inventory=(),
        deterministic_edits=(),
        oracle_case=None,
    )

    payload = _editor_payload(compiled, slots)
    assert payload["auxiliaryIdentityConsistencyGroups"] == [
        {
            "consistencyGroupId": "carrier-agent-group-1",
            "sourceIdentities": ["OLD AGENT"],
            "sourceOccurrenceSlotGroups": [["s1"], ["s2"]],
            "instruction": (
                "Render one distinct fictional auxiliary identity exactly once across each "
                "sourceOccurrenceSlotGroups entry, reflowing it across that occurrence's "
                "physical lines. Reuse the same identity across all occurrence groups while "
                "preserving surrounding legal wording and each referenced target principal."
            ),
        }
    ]

    selected, compounds, diagnostics, _preview_audit = _preview_repair_selection(
        material,
        {"s0": "CHANGED", "s1": "AGENT ONE", "s2": "AGENT TWO"},
        error_message=(
            "repeated raw auxiliary identity has inconsistent fictional realizations: "
            "carrier-agent-group-1"
        ),
    )

    assert tuple(row.alias for row in selected) == ("s1", "s2")
    assert compounds == ()
    assert any(row.get("kind") == "rawAuxiliaryIdentityConsistency" for row in diagnostics)


def test_any_auxiliary_identity_line_repair_closes_over_all_repeated_occurrences() -> None:
    first = RawAuxiliaryIdentityRequirement(
        requirementId="carrier-agent-L00002",
        relationship="agent_for_carrier",
        sourceIdentity="OLD AGENT\nSHIPPING AGENCY",
        sourceIdentityLineCount=2,
        gapLineCount=0,
        consistencyGroupId="carrier-agent-group-1",
        targetPrincipalName="THE CARRIER",
        sourceEvidence="OLD AGENT SHIPPING AGENCY as agents for the carrier",
    )
    repeated = first.model_copy(
        update={
            "requirementId": "carrier-agent-L00005",
            "sourceIdentity": "OLD AGENT",
            "sourceIdentityLineCount": 1,
        }
    )
    slots = (
        _Slot(alias="s0", line_id="L00001", source_line="UNRELATED", requirements=()),
        _Slot(alias="s1", line_id="L00002", source_line="OLD AGENT", requirements=()),
        _Slot(alias="s2", line_id="L00003", source_line="SHIPPING AGENCY", requirements=()),
        _Slot(alias="s3", line_id="L00005", source_line="OLD AGENT", requirements=()),
    )
    compiled = SimpleNamespace(
        workspace=RewriteWorkspace(
            original_text="UNRELATED\nOLD AGENT\nSHIPPING AGENCY\nRELATION\nOLD AGENT\n",
            current_text="UNRELATED\nOLD AGENT\nSHIPPING AGENCY\nRELATION\nOLD AGENT\n",
            raw_auxiliary_identity_requirements=(first, repeated),
        ),
        work_items=(),
        bundle=SimpleNamespace(
            result=SimpleNamespace(documentId="doc_" + "8" * 64),
            surfaceRenderingRequirements=(),
            anchoredScalarReplacementRequirements=(),
        ),
    )
    material = SimpleNamespace(
        compiled=compiled,
        slots=slots,
        compound_slots=(),
        inventory=(),
        deterministic_edits=(),
        oracle_case=None,
    )

    selected, _compounds, diagnostics, _audit = _preview_repair_selection(
        material,
        {"s0": "UNRELATED", "s1": "NEW AGENT", "s2": "AGENCY", "s3": "NEW AGENT"},
        error_message="slot changes host-locked legal affixes: s2",
    )

    assert tuple(row.alias for row in selected) == ("s1", "s2", "s3")
    assert any(row.get("kind") == "rawAuxiliaryIdentityRepairClosure" for row in diagnostics)


def test_auxiliary_identity_source_residual_repairs_every_line_in_the_identity_group() -> None:
    first = RawAuxiliaryIdentityRequirement(
        requirementId="carrier-agent-L00002",
        relationship="agent_for_carrier",
        sourceIdentity="OLD AGENT\nSHIPPING AGENCY",
        sourceIdentityLineCount=2,
        gapLineCount=0,
        consistencyGroupId="carrier-agent-group-1",
        targetPrincipalName="THE CARRIER",
        sourceEvidence="OLD AGENT SHIPPING AGENCY as agents for the carrier",
    )
    repeated = first.model_copy(
        update={
            "requirementId": "carrier-agent-L00005",
            "sourceIdentity": "OLD AGENT",
            "sourceIdentityLineCount": 1,
        }
    )
    slots = (
        _Slot(alias="s0", line_id="L00001", source_line="UNRELATED", requirements=()),
        _Slot(alias="s1", line_id="L00002", source_line="OLD AGENT", requirements=()),
        _Slot(alias="s2", line_id="L00003", source_line="SHIPPING AGENCY", requirements=()),
        _Slot(alias="s3", line_id="L00005", source_line="OLD AGENT", requirements=()),
    )
    compiled = SimpleNamespace(
        workspace=RewriteWorkspace(
            original_text="UNRELATED\nOLD AGENT\nSHIPPING AGENCY\nRELATION\nOLD AGENT\n",
            current_text="UNRELATED\nOLD AGENT\nSHIPPING AGENCY\nRELATION\nOLD AGENT\n",
            raw_auxiliary_identity_requirements=(first, repeated),
        ),
        work_items=(),
        bundle=SimpleNamespace(
            result=SimpleNamespace(documentId="doc_" + "c" * 64),
            surfaceRenderingRequirements=(),
            anchoredScalarReplacementRequirements=(),
        ),
    )
    material = SimpleNamespace(
        compiled=compiled,
        slots=slots,
        compound_slots=(),
        inventory=(),
        deterministic_edits=(),
        oracle_case=None,
    )

    selected, compounds, diagnostics, _preview_audit = _preview_repair_selection(
        material,
        {"s0": "UNRELATED", "s1": "NEW AGENT", "s2": "SHIPPING AGENCY", "s3": "NEW AGENT"},
        error_message=("raw auxiliary source identity remains after rewrite: carrier-agent-L00002"),
    )

    assert tuple(row.alias for row in selected) == ("s1", "s2", "s3")
    assert compounds == ()
    assert any(row.get("kind") == "rawAuxiliaryIdentityConsistency" for row in diagnostics)


def test_auxiliary_identity_consistency_accepts_line_wrap_variants() -> None:
    first = RawAuxiliaryIdentityRequirement(
        requirementId="carrier-agent-L00001",
        relationship="agent_for_carrier",
        sourceIdentity="Yang Ming Line\nBy (Thailand) Co., LTD.",
        sourceIdentityLineCount=2,
        gapLineCount=0,
        consistencyGroupId="carrier-agent-same-identity",
        targetPrincipalName="TARGET CARRIER",
        sourceEvidence="first rendering",
    )
    second = first.model_copy(
        update={
            "requirementId": "carrier-agent-L00004",
            "sourceIdentity": "By Yang Ming Line\n(Thailand) Co., LTD.",
            "targetPrincipalName": "THE CARRIER",
        }
    )
    slots = (
        _Slot(alias="s0", line_id="L00001", source_line="Yang Ming Line", requirements=()),
        _Slot(alias="s1", line_id="L00002", source_line="By (Thailand) Co., LTD.", requirements=()),
        _Slot(alias="s2", line_id="L00004", source_line="By Yang Ming Line", requirements=()),
        _Slot(alias="s3", line_id="L00005", source_line="(Thailand) Co., LTD.", requirements=()),
    )
    compiled = SimpleNamespace(
        workspace=RewriteWorkspace(
            original_text="Yang Ming Line\nBy (Thailand) Co., LTD.\nRELATION\n"
            "By Yang Ming Line\n(Thailand) Co., LTD.\n",
            current_text="Yang Ming Line\nBy (Thailand) Co., LTD.\nRELATION\n"
            "By Yang Ming Line\n(Thailand) Co., LTD.\n",
            raw_auxiliary_identity_requirements=(first, second),
        ),
        work_items=(),
        bundle=SimpleNamespace(result=SimpleNamespace(documentId="doc_" + "c" * 64)),
    )

    payload = _editor_payload(compiled, slots)

    assert payload["auxiliaryIdentityConsistencyGroups"] == [
        {
            "consistencyGroupId": "carrier-agent-same-identity",
            "sourceIdentities": [
                "By Yang Ming Line\n(Thailand) Co., LTD.",
                "Yang Ming Line\nBy (Thailand) Co., LTD.",
            ],
            "sourceOccurrenceSlotGroups": [["s0", "s1"], ["s2", "s3"]],
            "instruction": (
                "Render one distinct fictional auxiliary identity exactly once across each "
                "sourceOccurrenceSlotGroups entry, reflowing it across that occurrence's "
                "physical lines. Reuse the same identity across all occurrence groups while "
                "preserving surrounding legal wording and each referenced target principal."
            ),
        }
    ]


def test_editor_payload_keeps_all_changed_cargo_scalars_in_one_block() -> None:
    first_path = "documentPatch.cargoGroups[0].additionalInformation[0]"
    second_path = "documentPatch.cargoGroups[0].additionalInformation[1]"
    slots = (
        _Slot(
            alias="s0",
            line_id="L00001",
            source_line="OLD FIRST",
            requirements=({"kind": "task_label_delta", "paths": [first_path]},),
        ),
        _Slot(
            alias="s1",
            line_id="L00002",
            source_line="OLD SECOND",
            requirements=({"kind": "task_label_delta", "paths": [second_path]},),
        ),
    )
    compiled = SimpleNamespace(
        work_items=(),
        bundle=SimpleNamespace(result=SimpleNamespace(documentId="doc_" + "d" * 64)),
        workspace=RewriteWorkspace(
            original_text="OLD FIRST\nOLD SECOND\n",
            current_text="OLD FIRST\nOLD SECOND\n",
            current_target_label={
                "documentPatch": {
                    "cargoGroups": [
                        {
                            "groupId": "g1",
                            "additionalInformation": [
                                "2020 CARTONS CONTAINING 497 PIECES ONLY",
                                "PACKED IN 2020 CARTONS WITH PROTECTIVE WRAP",
                            ],
                        }
                    ]
                }
            },
        ),
    )

    payload = _editor_payload(compiled, slots)

    assert payload["cargoBlocks"] == [
        {
            "groupPath": "documentPatch.cargoGroups[0]",
            "slots": ["s0", "s1"],
            "requiredTargetScalars": [
                {"path": first_path, "value": "2020 CARTONS CONTAINING 497 PIECES ONLY"},
                {
                    "path": second_path,
                    "value": "PACKED IN 2020 CARTONS WITH PROTECTIVE WRAP",
                },
            ],
            "instruction": (
                "Render every required target scalar exactly once across these slots. "
                "All values must coexist; never replace one required value with another "
                "during a correction. "
                "Preserve the block's physical line count and OCR style. For repeated source "
                "product rows, repeat and reflow the authoritative target description instead "
                "of inventing a product variant or packaging fact."
            ),
        }
    ]


def test_editor_payload_omits_removed_optional_cargo_scalar_from_block() -> None:
    removed_path = "documentPatch.cargoGroups[0].dangerousGoods[0].packingGroupCategory"
    slot = _Slot(
        alias="s0",
        line_id="L00001",
        source_line="UN 1013 CLASS 2.2 PG III",
        requirements=({"kind": "task_label_delta", "paths": [removed_path]},),
    )
    compiled = SimpleNamespace(
        work_items=(),
        bundle=SimpleNamespace(result=SimpleNamespace(documentId="doc_" + "e" * 64)),
        workspace=RewriteWorkspace(
            original_text="UN 1013 CLASS 2.2 PG III\n",
            current_text="UN 1013 CLASS 2.2 PG III\n",
            current_target_label={
                "documentPatch": {
                    "cargoGroups": [
                        {
                            "groupId": "g1",
                            "dangerousGoods": [{"unNumber": "1013", "hazardCategory": "GASES"}],
                        }
                    ]
                }
            },
        ),
    )

    payload = _editor_payload(compiled, (slot,))

    assert payload.get("cargoBlocks") in (None, [])


def test_occurrence_repair_targets_only_surplus_auxiliary_carrier_line() -> None:
    target = "Asterline Maritime Carriers Ltd."
    slots = (
        _Slot(
            alias="s0",
            line_id="L00001",
            source_line="SIGNED FOR OLD CARRIER",
            requirements=(
                {
                    "kind": "task_label_delta",
                    "paths": ["documentPatch.parties.carrier.name"],
                },
            ),
        ),
        _Slot(
            alias="s1",
            line_id="L00002",
            source_line="OLD CARRIER agency tariff",
            requirements=(
                {
                    "kind": "carrier_dependent_branding",
                    "paths": ["documentPatch.parties.carrier.name"],
                },
            ),
        ),
    )
    workspace = RewriteWorkspace(
        original_text="SIGNED FOR OLD CARRIER\nOLD CARRIER agency tariff\n",
        current_text="SIGNED FOR OLD CARRIER\nOLD CARRIER agency tariff\n",
        target_value_occurrence_requirements=(
            TargetValueOccurrenceRequirement(
                targetPaths=("documentPatch.parties.carrier.name",),
                targetValue=target,
                requiredOccurrences=1,
            ),
        ),
    )
    material = SimpleNamespace(compiled=SimpleNamespace(workspace=workspace), slots=slots)

    selected, diagnostics = _target_occurrence_repair(
        material,
        {
            "s0": f"SIGNED FOR {target}",
            "s1": f"{target} agency tariff",
        },
    )

    assert tuple(row.alias for row in selected) == ("s1",)
    assert diagnostics[0]["requiredOccurrences"] == 1
    assert diagnostics[0]["observedOccurrences"] == 2
    assert diagnostics[0]["repairLineIds"] == ["L00002"]


def test_occurrence_repair_targets_every_line_of_surplus_wrapped_party_value() -> None:
    target = "Marudham Alloy Works Private Limited"
    slots = (
        _Slot(
            alias="s0",
            line_id="L00001",
            source_line="OLD SHIPPER",
            requirements=(
                {
                    "kind": "task_label_delta",
                    "paths": ["documentPatch.parties.shipper.name"],
                },
            ),
        ),
        _Slot(alias="s1", line_id="L00003", source_line="OLD CARGO A", requirements=()),
        _Slot(alias="s2", line_id="L00004", source_line="OLD CARGO B", requirements=()),
    )
    workspace = RewriteWorkspace(
        original_text="OLD SHIPPER\nDETAIL\nOLD CARGO A\nOLD CARGO B\n",
        current_text="OLD SHIPPER\nDETAIL\nOLD CARGO A\nOLD CARGO B\n",
        target_value_occurrence_requirements=(
            TargetValueOccurrenceRequirement(
                targetPaths=("documentPatch.parties.shipper.name",),
                targetValue=target,
                requiredOccurrences=1,
            ),
        ),
    )
    material = SimpleNamespace(compiled=SimpleNamespace(workspace=workspace), slots=slots)

    selected, diagnostics = _target_occurrence_repair(
        material,
        {
            "s0": target,
            "s1": "MARUDHAM ALLOY WORKS",
            "s2": "PRIVATE LIMITED",
        },
    )

    assert tuple(row.alias for row in selected) == ("s1", "s2")
    assert diagnostics[0]["observedOccurrences"] == 2
    assert diagnostics[0]["occurrenceLineIds"] == ["L00001", "L00003", "L00004"]
    assert diagnostics[0]["repairLineIds"] == ["L00003", "L00004"]


def test_sibling_evidence_is_limited_to_directly_located_same_object_lines() -> None:
    quantity = HybridWorkItem(
        workItemId="W0001",
        targetPaths=("documentPatch.cargoPackages[0].quantity",),
        action="replace",
        sourceValue=12,
        targetValue=18,
        state="agent_residual",
        evidenceLineIds=("L00010",),
        spanIds=("S001",),
        locator="numeric_surface",
        rationale="direct package evidence",
    )
    category = HybridWorkItem(
        workItemId="W0002",
        targetPaths=("documentPatch.cargoPackages[0].typeCategory",),
        action="replace",
        sourceValue="PACKAGE_BAG",
        targetValue="PACKAGE_CARTON",
        state="agent_residual",
        evidenceLineIds=("L00010", "L00020"),
        spanIds=("S002",),
        locator="sibling_object_evidence",
        rationale="broad package fallback",
    )
    unrelated = HybridWorkItem(
        workItemId="W0003",
        targetPaths=("documentPatch.cargoPackages[1].quantity",),
        action="replace",
        sourceValue=3,
        targetValue=4,
        state="agent_residual",
        evidenceLineIds=("L00030",),
        spanIds=("S003",),
        locator="numeric_surface",
        rationale="another package",
    )

    refined = _refine_sibling_evidence((quantity, category, unrelated))

    assert refined[0] == quantity
    assert refined[1].evidenceLineIds == ("L00010",)
    assert refined[2] == unrelated


def test_repeated_cargo_scalar_is_bound_to_its_indexed_description_block() -> None:
    text = (
        "FIRST GOODS\n"
        "MATERIAL 12345678\n"
        "BATCH 1111111111\n"
        "SECOND GOODS\n"
        "MATERIAL 12345678\n"
        "BATCH 2222222222\n"
    )
    first = HybridWorkItem(
        workItemId="W0001",
        targetPaths=("documentPatch.cargoGroups[0].additionalInformation[0]",),
        action="replace",
        sourceValue="MATERIAL 12345678",
        targetValue="MATERIAL 87654321",
        state="agent_residual",
        evidenceLineIds=("L00001", "L00004"),
        spanIds=("S001",),
        locator="sibling_object_evidence",
        rationale="ambiguous repeated scalar",
    )
    second = first.model_copy(
        update={
            "workItemId": "W0002",
            "targetPaths": ("documentPatch.cargoGroups[1].additionalInformation[0]",),
            "targetValue": "MATERIAL 24681357",
        }
    )
    bundle = SimpleNamespace(
        sourceLabel={"documentPatch": {"cargoGroups": []}},
        targetLabel={"documentPatch": {"cargoGroups": []}},
        cargoFlavorRewriteRequirements=(
            CargoFlavorRewriteRequirement(
                requirementId="cargo-g1",
                targetPath="documentPatch.cargoGroups[0].description",
                targetDescription="SYNTHETIC FIRST GOODS",
                sourceLineIds=("L00001",),
                sourceSurfaces=("FIRST GOODS",),
            ),
            CargoFlavorRewriteRequirement(
                requirementId="cargo-g2",
                targetPath="documentPatch.cargoGroups[1].description",
                targetDescription="SYNTHETIC SECOND GOODS",
                sourceLineIds=("L00004",),
                sourceSurfaces=("SECOND GOODS",),
            ),
        ),
    )

    refined = _refine_cargo_group_evidence(text, (first, second), bundle)

    assert refined[0].evidenceLineIds == ("L00002",)
    assert refined[1].evidenceLineIds == ("L00005",)
    assert all(row.locator == "cargo_group_relation_scope" for row in refined)


def test_complete_marks_sequences_bind_shared_values_to_indexed_cargo_groups() -> None:
    source = {
        "documentPatch": {
            "cargoGroups": [
                {"marksAndNumbers": ["SHARED PLANT", "LOT 100", "ROW A"]},
                {"marksAndNumbers": ["SHARED PLANT", "LOT 100", "ROW B"]},
            ]
        }
    }
    target = {
        "documentPatch": {
            "cargoGroups": [
                {"marksAndNumbers": ["TARGET A", "LOT 900", "NEW A"]},
                {"marksAndNumbers": ["TARGET B", "LOT 800", "NEW B"]},
            ]
        }
    }
    text = (
        "MARKS\n"
        "SHARED PLANT\nLOT 100\nNEW A\n"
        "\n"
        "SHARED PLANT\nLOT 100\nROW B\n"
        "MARKS COPY\n"
        "SHARED PLANT\nLOT 100\nROW B\n"
    )

    evidence = _cargo_marks_sequence_evidence(text, source, target)

    assert evidence["documentPatch.cargoGroups[0].marksAndNumbers[0]"] == frozenset({2})
    assert evidence["documentPatch.cargoGroups[0].marksAndNumbers[2]"] == frozenset({4})
    assert evidence["documentPatch.cargoGroups[1].marksAndNumbers[0]"] == frozenset({6, 10})
    assert evidence["documentPatch.cargoGroups[1].marksAndNumbers[2]"] == frozenset({8, 12})


def test_inventory_does_not_rebind_origin_embedded_in_owned_marks_surface() -> None:
    source = {
        "documentPatch": {
            "cargoGroups": [{"marksAndNumbers": ["Made in Taiwan"], "origin": {"name": "Taiwan"}}]
        }
    }
    target = {
        "documentPatch": {
            "cargoGroups": [{"marksAndNumbers": ["Made in Korea"], "origin": {"name": "Korea"}}]
        }
    }
    text = "Made in Taiwan\nORIGIN: Taiwan\n"
    items = (
        HybridWorkItem(
            workItemId="W0001",
            targetPaths=("documentPatch.cargoGroups[0].marksAndNumbers[0]",),
            action="replace",
            sourceValue="Made in Taiwan",
            targetValue="Made in Korea",
            state="agent_residual",
            evidenceLineIds=("L00001",),
            spanIds=("S001",),
            locator="relation_scoped_surface",
            rationale="complete marks surface",
        ),
        HybridWorkItem(
            workItemId="W0002",
            targetPaths=("documentPatch.cargoGroups[0].origin.name",),
            action="replace",
            sourceValue="Taiwan",
            targetValue="Korea",
            state="agent_residual",
            evidenceLineIds=("L00002",),
            spanIds=("S002",),
            locator="exact_literal",
            rationale="cargo origin",
        ),
    )

    inventory = build_mutable_inventory(
        source_text=text,
        current_text=text,
        source_label=source,
        target_label=target,
        work_items=items,
        oracle_case=None,
    )

    origin_rows = [
        row
        for row in inventory
        if row.sourceSurface.casefold() == "taiwan"
        and "documentPatch.cargoGroups[0].origin.name" in row.targetPaths
    ]
    assert [row.lineIds for row in origin_rows] == [("L00002",)]


def test_identical_marks_sequences_are_not_arbitrarily_assigned_to_cargo_groups() -> None:
    source = {
        "documentPatch": {
            "cargoGroups": [
                {"marksAndNumbers": ["SHARED", "LOT 100"]},
                {"marksAndNumbers": ["SHARED", "LOT 100"]},
            ]
        }
    }
    target = {
        "documentPatch": {
            "cargoGroups": [
                {"marksAndNumbers": ["TARGET A", "LOT 900"]},
                {"marksAndNumbers": ["TARGET B", "LOT 800"]},
            ]
        }
    }

    assert (
        _cargo_marks_sequence_evidence(
            "SHARED\nLOT 100\n",
            source,
            target,
        )
        == {}
    )


def test_party_evidence_excludes_decorated_heading_and_unrelated_role_lines() -> None:
    text = "CONSIGNEE (3) (NOT NEGOTIABLE UNLESS CONSIGNED TO ORDER)\nTO ORDER\nOTHER PARTY LINE\n"
    source = {"documentPatch": {"parties": {"consignee": {"name": "TO ORDER"}}}}
    item = HybridWorkItem(
        workItemId="W0001",
        targetPaths=("documentPatch.parties.consignee.name",),
        action="replace",
        sourceValue="TO ORDER",
        targetValue="Kestrel Meridian Trading Ltd.",
        state="agent_residual",
        evidenceLineIds=("L00001", "L00002", "L00003"),
        spanIds=("S001",),
        locator="party_role_block",
        rationale="broad party block",
    )

    refined = _refine_party_evidence(text, source, (item,))

    assert refined[0].evidenceLineIds == ("L00002",)
    assert refined[0].locator == "exact_literal"
    assert _party_scalar_occurrence_count(text, "TO ORDER") == 1


def test_party_evidence_excludes_deterministically_rendered_carrier_header() -> None:
    text = (
        "--- PAGE 1 ---\n"
        "MEDITERRANEAN SHIPPING COMPANY S.A.\n"
        "\n"
        "CARRIER\n"
        "MSC Mediterranean Shipping Company S.A.\n"
    )
    source = {
        "documentPatch": {
            "parties": {"carrier": {"name": "MSC Mediterranean Shipping Company S.A."}}
        }
    }
    item = HybridWorkItem(
        workItemId="W0001",
        targetPaths=("documentPatch.parties.carrier.name",),
        action="replace",
        sourceValue="MSC Mediterranean Shipping Company S.A.",
        targetValue="Asterwave Ocean Carriage S.A.",
        state="agent_residual",
        evidenceLineIds=("L00002", "L00005"),
        spanIds=("S001",),
        locator="party_role_block",
        rationale="carrier block and header",
    )
    requirement = SurfaceRenderingRequirement(
        kind="carrier_header_identity",
        targetPath="documentPatch.parties.carrier.name",
        sourceSurface="MEDITERRANEAN SHIPPING COMPANY S.A.",
        targetSurface="Asterwave Ocean Carriage S.A.",
        sourceOccurrences=1,
        contextEvidence="--- PAGE 1 ---\nMEDITERRANEAN SHIPPING COMPANY S.A.\n\nCARRIER",
    )

    refined = _refine_party_evidence(text, source, (item,), (requirement,))

    assert refined[0].evidenceLineIds == ("L00005",)
    assert refined[0].locator == "exact_literal"


def test_party_evidence_includes_punctuation_variant_inside_owned_role_block() -> None:
    text = "CARRIER\nARKAS CONTAINER TRANSPORT S.A.\nas agents of Arkas Container Transport, S.A.\n"
    source = {
        "documentPatch": {"parties": {"carrier": {"name": "Arkas Container Transport, S.A."}}}
    }
    item = HybridWorkItem(
        workItemId="W0001",
        targetPaths=("documentPatch.parties.carrier.name",),
        action="replace",
        sourceValue="Arkas Container Transport, S.A.",
        targetValue="Bluehaven Maritime Lines, Ltd.",
        state="agent_residual",
        evidenceLineIds=("L00002", "L00003"),
        spanIds=("S001",),
        locator="party_role_block",
        rationale="carrier block",
    )

    refined = _refine_party_evidence(text, source, (item,))

    assert refined[0].evidenceLineIds == ("L00002", "L00003")


def test_inventory_grants_authority_to_complete_party_name_punctuation_variant() -> None:
    source_text = "ARKAS CONTAINER TRANSPORT S.A.\nas agents of Arkas Container Transport, S.A.\n"
    source = {
        "documentPatch": {"parties": {"carrier": {"name": "Arkas Container Transport, S.A."}}}
    }
    target = {"documentPatch": {"parties": {"carrier": {"name": "Bluehaven Maritime Lines, Ltd."}}}}
    item = HybridWorkItem(
        workItemId="W0001",
        targetPaths=("documentPatch.parties.carrier.name",),
        action="replace",
        sourceValue="Arkas Container Transport, S.A.",
        targetValue="Bluehaven Maritime Lines, Ltd.",
        state="agent_residual",
        evidenceLineIds=("L00002",),
        spanIds=("S001",),
        locator="exact_literal",
        rationale="exact principal line",
    )

    inventory = build_mutable_inventory(
        source_text=source_text,
        current_text=source_text,
        source_label=source,
        target_label=target,
        work_items=(item,),
        oracle_case=None,
    )

    carrier_candidates = [
        row
        for row in inventory
        if row.category == "changed_source_occurrence"
        and row.targetPaths == ("documentPatch.parties.carrier.name",)
    ]
    assert {line for row in carrier_candidates for line in row.lineIds} == {
        "L00001",
        "L00002",
    }


def test_party_audit_accepts_city_interleaved_inside_complete_wrapped_address() -> None:
    source_text = (
        "NOTIFY PARTY\n"
        "OLD COMPANY\n"
        "182 SHIMOBUN, KINSEI-CHO, SHIKOKUCHUO-CITY,\n"
        "EHIME-PREF. 799-0111 JAPAN\n"
    )
    output_text = (
        "NOTIFY PARTY\n"
        "NEW COMPANY\n"
        "18 Al Mashtal Street, Shubra al Khaymah,\n"
        "Industrial District EGYPT\n"
    )
    source = {
        "documentPatch": {
            "parties": {
                "notifyParties": [
                    {
                        "name": "OLD COMPANY",
                        "address": "182 SHIMOBUN, KINSEI-CHO, EHIME-PREF. 799-0111",
                        "city": "SHIKOKUCHUO-CITY",
                        "country": "JAPAN",
                    }
                ]
            }
        }
    }
    target = {
        "documentPatch": {
            "parties": {
                "notifyParties": [
                    {
                        "name": "NEW COMPANY",
                        "address": "18 Al Mashtal Street, Industrial District",
                        "city": "Shubra al Khaymah",
                        "country": "EGYPT",
                    }
                ]
            }
        }
    }
    items = tuple(
        HybridWorkItem(
            workItemId=f"W{index:04d}",
            targetPaths=(f"documentPatch.parties.notifyParties[0].{field}",),
            action="replace",
            sourceValue=source["documentPatch"]["parties"]["notifyParties"][0][field],
            targetValue=target["documentPatch"]["parties"]["notifyParties"][0][field],
            state="agent_residual",
            evidenceLineIds=lines,
            spanIds=("S001",),
            locator="literal_in_party_role_block",
            rationale="role-owned party scalar",
        )
        for index, (field, lines) in enumerate(
            (
                ("name", ("L00002",)),
                ("address", ("L00003", "L00004")),
                ("city", ("L00003",)),
                ("country", ("L00004",)),
            ),
            start=1,
        )
    )

    audit = audit_full_document(
        document_id="doc_" + "b" * 64,
        source_text=source_text,
        output_text=output_text,
        source_label=source,
        target_label=target,
        inventory=(),
        deterministic_edits=(),
        oracle_case=None,
        work_items=items,
    )

    assert audit.passed is True


def test_changed_source_constituent_is_licensed_only_inside_rendered_reference() -> None:
    source_text = "CROP YEAR: 2023\nSB NO: 6446167 DT: 25-07-2023\n"
    output_text = "CROP YEAR: 2025\nSB NO: 6446167 DT: 25-07-2023\n"
    source = {
        "documentPatch": {
            "cargoGroups": [{"groupId": "g1", "additionalInformation": ["2023"]}],
            "forwardingAndExportReferences": ["6446167 25-07-2023"],
        }
    }
    target = {
        "documentPatch": {
            "cargoGroups": [{"groupId": "g1", "additionalInformation": ["2025"]}],
            "forwardingAndExportReferences": ["6446167 25-07-2023"],
        }
    }
    item = HybridWorkItem(
        workItemId="W0001",
        targetPaths=("documentPatch.cargoGroups[0].additionalInformation[0]",),
        action="replace",
        sourceValue="2023",
        targetValue="2025",
        state="agent_residual",
        evidenceLineIds=("L00001",),
        spanIds=("S001",),
        locator="exact_literal",
        rationale="exact crop-year line",
    )

    inventory = build_mutable_inventory(
        source_text=source_text,
        current_text=source_text,
        source_label=source,
        target_label=target,
        work_items=(item,),
        oracle_case=None,
    )
    changed_lines = {
        line_id for row in inventory if row.sourceSurface == "2023" for line_id in row.lineIds
    }
    assert changed_lines == {"L00001"}

    audit = audit_full_document(
        document_id="doc_" + "c" * 64,
        source_text=source_text,
        output_text=output_text,
        source_label=source,
        target_label=target,
        inventory=inventory,
        deterministic_edits=(),
        oracle_case=None,
        work_items=(item,),
    )

    assert audit.passed is True


def test_party_occurrence_refinement_counts_abbreviated_legal_principals() -> None:
    text = (
        "CARRIER\nCMA CGM Société Anonyme\n\n"
        "website: www.cma-cgm.com\n"
        "SIGNED FOR THE CARRIER CMA CGM S.A.\n"
    )
    item = HybridWorkItem(
        workItemId="W0001",
        targetPaths=("documentPatch.parties.carrier.name",),
        action="replace",
        sourceValue="CMA CGM Société Anonyme",
        targetValue="Northgate Maritime Lines LLC",
        state="agent_residual",
        evidenceLineIds=("L00002",),
        spanIds=("S001",),
        locator="exact_literal",
        rationale="role-owned carrier line",
    )
    requirement = TargetValueOccurrenceRequirement(
        targetPaths=("documentPatch.parties.carrier.name",),
        targetValue="Northgate Maritime Lines LLC",
        requiredOccurrences=3,
    )

    refined = _refine_party_occurrence_requirements(text, text, (item,), (requirement,))

    assert refined[0].requiredOccurrences == 2


def test_party_occurrence_refinement_excludes_agent_alias_carrier_tokens() -> None:
    text = (
        "CMA CGM\n"
        "CARRIER: CMA CGM Société Anonyme\n"
        "available from any CMA CGM agency\n"
        "Shipped APL SINGAPURA CMA CGM XIAMEN As agents for the Carrier\n"
        "SIGNED FOR THE CARRIER CMA CGM S.A.\n"
        "BY CMA CGM XIAMEN\n"
        "as agents for the carrier CMA CGM S. A.\n"
    )
    item = HybridWorkItem(
        workItemId="W0001",
        targetPaths=("documentPatch.parties.carrier.name",),
        action="replace",
        sourceValue="CMA CGM",
        targetValue="Northgate Maritime Lines LLC",
        state="agent_residual",
        evidenceLineIds=("L00001",),
        spanIds=("S001",),
        locator="exact_literal",
        rationale="carrier identities and agency relationships",
    )
    requirement = TargetValueOccurrenceRequirement(
        targetPaths=("documentPatch.parties.carrier.name",),
        targetValue="Northgate Maritime Lines LLC",
        requiredOccurrences=7,
    )

    refined = _refine_party_occurrence_requirements(text, text, (item,), (requirement,))

    assert refined[0].requiredOccurrences == 5


def test_party_occurrence_refinement_excludes_carrier_inside_labeled_vessel_name() -> None:
    text = "VESSEL\nCMA CGM MOLIERE\nAS AGENT FOR, THE CARRIER, CMA CGM\n"
    source_label = {
        "documentPatch": {
            "parties": {"carrier": {"name": "CMA CGM"}},
            "transport": {"vesselName": "CMA CGM MOLIERE"},
        }
    }
    item = HybridWorkItem(
        workItemId="W0001",
        targetPaths=("documentPatch.parties.carrier.name",),
        action="replace",
        sourceValue="CMA CGM",
        targetValue="Northgate Maritime Lines LLC",
        state="agent_residual",
        evidenceLineIds=("L00003",),
        spanIds=("S001",),
        locator="carrier_principal_template_slots",
        rationale="one legal carrier-principal slot",
    )
    requirement = TargetValueOccurrenceRequirement(
        targetPaths=("documentPatch.parties.carrier.name",),
        targetValue="Northgate Maritime Lines LLC",
        requiredOccurrences=2,
    )

    refined = _refine_party_occurrence_requirements(
        text,
        text,
        (item,),
        (requirement,),
        source_label,
    )

    assert refined[0].requiredOccurrences == 1


def test_party_occurrence_refinement_counts_prefilled_header_and_residual_signature() -> None:
    target = "Helviora Ocean Link AG"
    current_text = (
        f"{target}\n"
        "SIGNED on behalf of the Carrier MSC Mediterranean Shipping Company S.A.\n"
        "by Mediterranean Shipping Company (Aust) Pty Ltd as Agent\n"
    )
    item = HybridWorkItem(
        workItemId="W0001",
        targetPaths=("documentPatch.parties.carrier.name",),
        action="replace",
        sourceValue="MSC Mediterranean Shipping Company S.A.",
        targetValue=target,
        state="agent_residual",
        evidenceLineIds=("L00002",),
        spanIds=("S001",),
        locator="exact_literal",
        rationale="carrier signature remains after deterministic masthead rendering",
    )
    requirement = TargetValueOccurrenceRequirement(
        targetPaths=("documentPatch.parties.carrier.name",),
        targetValue=target,
        requiredOccurrences=1,
    )

    original_text = (
        "MSC Mediterranean Shipping Company S.A.\n"
        "SIGNED on behalf of the Carrier MSC Mediterranean Shipping Company S.A.\n"
        "by Mediterranean Shipping Company (Aust) Pty Ltd as Agent\n"
    )
    refined = _refine_party_occurrence_requirements(
        original_text, current_text, (item,), (requirement,)
    )

    assert refined[0].requiredOccurrences == 2


def test_party_occurrence_refinement_preserves_all_original_carrier_principal_slots() -> None:
    source = "Old Carrier Lines Ltd."
    target = "Helviora Ocean Link AG"
    original_text = f"{source}\nSIGNED FOR THE CARRIER {source}\nas agents for the carrier OCLL\n"
    current_text = f"{target}\nSIGNED FOR THE CARRIER {source}\nas agents for the carrier OCLL\n"
    item = HybridWorkItem(
        workItemId="W0001",
        targetPaths=("documentPatch.parties.carrier.name",),
        action="replace",
        sourceValue=source,
        targetValue=target,
        state="agent_residual",
        evidenceLineIds=("L00002", "L00003"),
        spanIds=("S001",),
        locator="carrier_principal_template_slots",
        rationale="three carrier-principal slots survive one deterministic prefill",
    )
    requirement = TargetValueOccurrenceRequirement(
        targetPaths=("documentPatch.parties.carrier.name",),
        targetValue=target,
        requiredOccurrences=1,
    )

    refined = _refine_party_occurrence_requirements(
        original_text, current_text, (item,), (requirement,)
    )

    assert refined[0].requiredOccurrences == 3


def test_equipment_evidence_requires_a_printed_equipment_anchor() -> None:
    current_text = (
        "2x40' HW\n"
        "BORU 949070-7 40' HW SAID TO CONTAIN 10,203.70 KGS\n"
        "BATCH 69314827 MATERIAL 0084618371 GRADE CMRDY9W\n"
        "BORU 790788-0 40' HW SAID TO CONTAIN 13,609.53 KGS\n"
    )
    item = HybridWorkItem(
        workItemId="W0001",
        targetPaths=("documentPatch.containers[0].printedEquipmentSurface",),
        action="replace_equipment_surface",
        sourceValue="40' HW",
        targetValue={
            "sizeCategory": "FORTY_FOOT_HIGH_CUBE",
            "typeCategory": "GENERAL_PURPOSE",
        },
        state="agent_residual",
        evidenceLineIds=("L00001", "L00002", "L00003", "L00004"),
        spanIds=("S001",),
        locator="sibling_object_evidence",
        rationale="generic pre-refinement evidence",
    )
    bundle = SimpleNamespace(
        sourceLabel={
            "documentPatch": {
                "containers": [
                    {"containerNumber": "BORU7013618"},
                    {"containerNumber": "BORU7011256"},
                ]
            }
        },
        targetLabel={
            "documentPatch": {
                "containers": [
                    {"containerNumber": "BORU9490707"},
                    {"containerNumber": "BORU7907880"},
                ]
            }
        },
        anchoredScalarReplacementRequirements=(
            AnchoredScalarReplacementRequirement(
                targetPaths=("documentPatch.containers[0].printedEquipmentSurface",),
                sourceLineIds=("L00002",),
                sourceSurface="40' HW",
                targetSurface="40' HIGH CUBE GENERAL PURPOSE",
            ),
        ),
    )

    refined = _refine_equipment_evidence(current_text, (item,), bundle)

    assert refined[0].evidenceLineIds == ("L00002",)
    assert refined[0].locator == "relation_scoped_surface"


def test_equipment_evidence_blocks_identifier_only_insertion() -> None:
    item = HybridWorkItem(
        workItemId="W0001",
        targetPaths=("documentPatch.containers[0].printedEquipmentSurface",),
        action="add_equipment_surface",
        sourceValue=None,
        targetValue={
            "sizeCategory": "FORTY_FOOT_HIGH_CUBE",
            "typeCategory": "GENERAL_PURPOSE",
        },
        state="agent_residual",
        evidenceLineIds=("L00001",),
        spanIds=("S001",),
        locator="sibling_object_evidence",
        rationale="container identifier only",
    )
    bundle = SimpleNamespace(anchoredScalarReplacementRequirements=())

    refined = _refine_equipment_evidence("MSCU1234567\n", (item,), bundle)

    assert refined[0].state == "blocked_unlocated"
    assert refined[0].evidenceLineIds == ()


def test_shared_target_scalar_is_licensed_only_outside_changed_route_lines() -> None:
    source_text = (
        "DELIVERY AGENT\n"
        "ASTERBRUG DELIVERY LOGISTICS\n"
        "ROTTERDAM NETHERLANDS\n"
        "PORT OF DISCHARGE\n"
        "ROTTERDAM NETHERLANDS\n"
    )
    output_text = (
        source_text.replace("ROTTERDAM NETHERLANDS\n", "ROTTERDAM NETHERLANDS\n", 1).rsplit(
            "ROTTERDAM NETHERLANDS", 1
        )[0]
        + "KENOSHA UNITED STATES\n"
    )
    source = {
        "documentPatch": {
            "parties": {
                "deliveryAgent": {
                    "name": "ASTERBRUG DELIVERY LOGISTICS",
                    "city": "ROTTERDAM",
                    "country": "NETHERLANDS",
                }
            },
            "route": {"portOfDischarge": {"name": "ROTTERDAM", "country": "NETHERLANDS"}},
        }
    }
    target = {
        "documentPatch": {
            "parties": {
                "deliveryAgent": {
                    "name": "ASTERBRUG DELIVERY LOGISTICS",
                    "city": "ROTTERDAM",
                    "country": "NETHERLANDS",
                }
            },
            "route": {"portOfDischarge": {"name": "KENOSHA", "country": "UNITED STATES"}},
        }
    }
    items = (
        HybridWorkItem(
            workItemId="W0001",
            targetPaths=("documentPatch.route.portOfDischarge.name",),
            action="replace",
            sourceValue="ROTTERDAM",
            targetValue="KENOSHA",
            state="agent_residual",
            evidenceLineIds=("L00005",),
            spanIds=("S001",),
            locator="location_role_surface",
            rationale="route-owned locality",
        ),
        HybridWorkItem(
            workItemId="W0002",
            targetPaths=("documentPatch.route.portOfDischarge.country",),
            action="replace",
            sourceValue="NETHERLANDS",
            targetValue="UNITED STATES",
            state="agent_residual",
            evidenceLineIds=("L00005",),
            spanIds=("S001",),
            locator="location_role_surface",
            rationale="route-owned country",
        ),
    )

    inventory = build_mutable_inventory(
        source_text=source_text,
        current_text=output_text,
        source_label=source,
        target_label=target,
        work_items=items,
        oracle_case=None,
    )
    audit = audit_full_document(
        document_id="doc_" + "f" * 64,
        source_text=source_text,
        output_text=output_text,
        source_label=source,
        target_label=target,
        inventory=inventory,
        deterministic_edits=(),
        oracle_case=None,
        work_items=items,
    )

    assert audit.passed is True


def test_nonparty_evidence_excludes_an_unchanged_party_role_copy() -> None:
    text = (
        "DELIVERY AGENT\n"
        "ASTERBRUG DELIVERY LOGISTICS\n"
        "ROTTERDAM NETHERLANDS\n\n"
        "PORT OF DISCHARGE\n"
        "ROTTERDAM NETHERLANDS\n"
    )
    source = {
        "documentPatch": {
            "parties": {
                "deliveryAgent": {
                    "name": "ASTERBRUG DELIVERY LOGISTICS",
                    "city": "ROTTERDAM",
                    "country": "NETHERLANDS",
                }
            },
            "route": {"portOfDischarge": {"name": "ROTTERDAM", "country": "NETHERLANDS"}},
        }
    }
    target = {
        "documentPatch": {
            "parties": {
                "deliveryAgent": {
                    "name": "ASTERBRUG DELIVERY LOGISTICS",
                    "city": "ROTTERDAM",
                    "country": "NETHERLANDS",
                }
            },
            "route": {"portOfDischarge": {"name": "KENOSHA", "country": "UNITED STATES"}},
        }
    }
    item = HybridWorkItem(
        workItemId="W0001",
        targetPaths=("documentPatch.route.portOfDischarge.country",),
        action="replace",
        sourceValue="NETHERLANDS",
        targetValue="UNITED STATES",
        state="agent_residual",
        evidenceLineIds=("L00003", "L00006"),
        spanIds=("S001",),
        locator="sibling_object_evidence",
        rationale="broad literal evidence",
    )

    refined = _refine_foreign_party_evidence(text, source, target, (item,))

    assert refined[0].evidenceLineIds == ("L00006",)
    assert refined[0].state == "agent_residual"


def test_nonparty_evidence_excludes_a_changed_party_role_copy() -> None:
    text = (
        "DELIVERY AGENT\n"
        "ASTERBRUG DELIVERY LOGISTICS\n"
        "ROTTERDAM THE NETHERLANDS\n\n"
        "PORT OF DISCHARGE\n"
        "ROTTERDAM NETHERLANDS\n"
    )
    source = {
        "documentPatch": {
            "parties": {
                "deliveryAgent": {
                    "name": "ASTERBRUG DELIVERY LOGISTICS",
                    "city": "ROTTERDAM",
                    "country": "THE NETHERLANDS",
                }
            },
            "route": {"portOfDischarge": {"name": "ROTTERDAM", "country": "NETHERLANDS"}},
        }
    }
    target = {
        "documentPatch": {
            "parties": {
                "deliveryAgent": {
                    "name": "ASTERBRUG DELIVERY LOGISTICS",
                    "city": "ALPHEN",
                    "country": "NETHERLANDS",
                }
            },
            "route": {"portOfDischarge": {"name": "KENOSHA", "country": "UNITED STATES"}},
        }
    }
    route_city = HybridWorkItem(
        workItemId="W0001",
        targetPaths=("documentPatch.route.portOfDischarge.name",),
        action="replace",
        sourceValue="ROTTERDAM",
        targetValue="KENOSHA",
        state="agent_residual",
        evidenceLineIds=("L00003", "L00006"),
        spanIds=("S001",),
        locator="sibling_object_evidence",
        rationale="broad literal evidence",
    )
    route_country = route_city.model_copy(
        update={
            "workItemId": "W0002",
            "targetPaths": ("documentPatch.route.portOfDischarge.country",),
            "sourceValue": "NETHERLANDS",
            "targetValue": "UNITED STATES",
        }
    )

    refined = _refine_foreign_party_evidence(text, source, target, (route_city, route_country))

    assert refined[0].evidenceLineIds == ("L00006",)
    assert refined[1].evidenceLineIds == ("L00006",)


def test_nonparty_country_evidence_excludes_party_shorthand_repeat() -> None:
    text = (
        "DESTINATION AGENT\n"
        "ASTERBRUG DELIVERY LOGISTICS\n"
        "ROTTERDAM, THE NETHERLANDS\n"
        "NL809653035000\n"
        "ROTTERDAM NETHERLANDS\n\n"
        "PORT OF DISCHARGE\n"
        "ROTTERDAM NETHERLANDS\n"
    )
    source = {
        "documentPatch": {
            "parties": {
                "deliveryAgent": {
                    "name": "ASTERBRUG DELIVERY LOGISTICS",
                    "city": "ROTTERDAM",
                    "country": "THE NETHERLANDS",
                }
            },
            "route": {"portOfDischarge": {"name": "ROTTERDAM", "country": "NETHERLANDS"}},
        }
    }
    target = {
        "documentPatch": {
            "parties": {
                "deliveryAgent": {
                    "name": "ASTERBRUG DELIVERY LOGISTICS",
                    "city": "ALPHEN AAN DEN RIJN",
                    "country": "NETHERLANDS",
                }
            },
            "route": {"portOfDischarge": {"name": "KENOSHA", "country": "UNITED STATES"}},
        }
    }
    route_country = HybridWorkItem(
        workItemId="W0001",
        targetPaths=("documentPatch.route.portOfDischarge.country",),
        action="replace",
        sourceValue="NETHERLANDS",
        targetValue="UNITED STATES",
        state="agent_residual",
        evidenceLineIds=("L00003", "L00005", "L00008"),
        spanIds=("S001",),
        locator="sibling_object_evidence",
        rationale="broad literal evidence",
    )

    refined = _refine_foreign_party_evidence(text, source, target, (route_country,))

    assert refined[0].evidenceLineIds == ("L00008",)


def test_party_address_constituent_protects_only_its_exact_line() -> None:
    text = (
        "FREIGHT AND CHARGES (indicate whether prepaid or where payable)\n"
        "\nMERSIN\n\nPLACE AND DATE OF ISSUE\nMERSIN 30.05.2025\n\n"
        "For the Carrier:\nBUTROS TRADING & TRANSPORT S.A.\nMERSIN - TURKEY\n"
    )
    source = {
        "documentPatch": {
            "freight": {"paymentPlace": {"name": "MERSIN"}},
            "placeOfIssue": {"name": "MERSIN"},
            "parties": {
                "carrier": {
                    "name": "BUTROS TRADING & TRANSPORT S.A.",
                    "address": "MERSIN - TURKEY",
                }
            },
        }
    }
    target = {
        "documentPatch": {
            "freight": {"paymentPlace": {"name": "METROTOWN"}},
            "placeOfIssue": {"name": "LARGO"},
            "parties": source["documentPatch"]["parties"],
        }
    }
    payment = HybridWorkItem(
        workItemId="W0001",
        targetPaths=("documentPatch.freight.paymentPlace.name",),
        action="replace",
        sourceValue="MERSIN",
        targetValue="METROTOWN",
        state="agent_residual",
        evidenceLineIds=("L00003", "L00006", "L00010"),
        spanIds=("S001",),
        locator="exact_literal",
        rationale="shared locality",
    )
    issue = payment.model_copy(
        update={
            "workItemId": "W0002",
            "targetPaths": ("documentPatch.placeOfIssue.name",),
            "targetValue": "LARGO",
        }
    )

    refined = _refine_foreign_party_evidence(text, source, target, (payment, issue))

    assert refined[0].evidenceLineIds == ("L00003",)
    assert refined[1].evidenceLineIds == ("L00006",)


def test_allocation_quantity_is_bound_to_its_container_package_rows() -> None:
    text = (
        "Total Items: 40\n"
        "CONT0000001\n"
        "40' HIGH CUBE\n"
        "20 Package(s) of MACHINERY\n"
        "20 PALLETS\n"
        "CONT0000002\n"
        "40' HIGH CUBE\n"
        "20 Package(s) of PARTS\n"
        "20 PALLETS\n"
    )
    source = {
        "documentPatch": {
            "containers": [
                {"containerNumber": "CONT0000001"},
                {"containerNumber": "CONT0000002"},
            ],
            "cargoAllocationGroups": [
                {
                    "groupId": "g1",
                    "allocations": [
                        {"containerNumber": "CONT0000001", "packageQuantity": 20},
                        {"containerNumber": "CONT0000002", "packageQuantity": 20},
                    ],
                }
            ],
        }
    }
    target = {
        "documentPatch": {
            "containers": [
                {"containerNumber": "CONT0000001"},
                {"containerNumber": "CONT0000002"},
            ],
            "cargoAllocationGroups": [
                {
                    "groupId": "g1",
                    "allocations": [
                        {"containerNumber": "CONT0000001", "packageQuantity": 68},
                        {"containerNumber": "CONT0000002", "packageQuantity": 67},
                    ],
                }
            ],
        }
    }
    first = HybridWorkItem(
        workItemId="W0001",
        targetPaths=("documentPatch.cargoAllocationGroups[0].allocations[0].packageQuantity",),
        action="replace",
        sourceValue=20,
        targetValue=68,
        state="agent_residual",
        evidenceLineIds=("L00003", "L00004", "L00005", "L00007", "L00008", "L00009"),
        spanIds=("S001",),
        locator="numeric_surface",
        rationale="broad numeric evidence",
    )
    second = first.model_copy(
        update={
            "workItemId": "W0002",
            "targetPaths": (
                "documentPatch.cargoAllocationGroups[0].allocations[1].packageQuantity",
            ),
            "targetValue": 67,
        }
    )
    bundle = SimpleNamespace(
        sourceLabel=source,
        targetLabel=target,
        cargoFlavorRewriteRequirements=(),
        operationalFlavorRequirements=(),
    )

    refined = _refine_numeric_evidence(text, (first, second), bundle)

    assert refined[0].evidenceLineIds == ("L00004", "L00005")
    assert refined[1].evidenceLineIds == ("L00008", "L00009")


def test_allocation_quantity_uses_relation_owned_column_cell_and_formatted_container() -> None:
    text = (
        "CONTAINER NOS.\nTYPE\nC. SEAL NO\nQTY\nP. TYPE\nGR. WT(KG)\n"
        "MCLU 510204.9\nHC40\nA1227883\n990\nBAGS\n25344.000\n"
    )
    source = {
        "documentPatch": {
            "containers": [{"containerNumber": "MCLU5102049"}],
            "cargoAllocationGroups": [
                {
                    "groupId": "g1",
                    "allocations": [{"containerNumber": "MCLU5102049", "packageQuantity": 990}],
                }
            ],
        }
    }
    target = {
        "documentPatch": {
            "containers": [{"containerNumber": "MCLU0899808"}],
            "cargoAllocationGroups": [
                {
                    "groupId": "g1",
                    "allocations": [{"containerNumber": "MCLU0899808", "packageQuantity": 327}],
                }
            ],
        }
    }

    assert _allocation_quantity_evidence_lines(
        text,
        path="documentPatch.cargoAllocationGroups[0].allocations[0].packageQuantity",
        source_value=990,
        source_label=source,
        target_label=target,
        candidate_line_numbers={10},
    ) == {10}


@pytest.mark.parametrize(
    ("surface", "quantity"),
    (("1 PLT", 1), ("248 CTNS", 248), ("10PL", 10), ("1 PK", 1)),
)
def test_package_quantity_recognizes_common_printed_abbreviations(
    surface: str, quantity: int
) -> None:
    assert _is_explicit_package_quantity_line(surface, quantity)


def test_allocation_quantities_follow_complete_column_order() -> None:
    text = (
        "CONTAINER NOS.\n"
        "FSCU5906804\n"
        "KKFU6721390\n"
        "ONEU9083430\n"
        "2184 BOXES\n"
        "2184 BOXES\n"
        "2184 BOXES\n"
    )
    source = {
        "documentPatch": {
            "containers": [
                {"containerNumber": "FSCU5906804"},
                {"containerNumber": "KKFU6721390"},
                {"containerNumber": "ONEU9083430"},
            ],
            "cargoAllocationGroups": [
                {
                    "groupId": f"g{index + 1}",
                    "allocations": [{"containerNumber": container, "packageQuantity": 2184}],
                }
                for index, container in enumerate(("FSCU5906804", "KKFU6721390", "ONEU9083430"))
            ],
        }
    }
    target = {
        "documentPatch": {
            **source["documentPatch"],
            "cargoAllocationGroups": [
                {
                    "groupId": f"g{index + 1}",
                    "allocations": [{"containerNumber": container, "packageQuantity": quantity}],
                }
                for index, (container, quantity) in enumerate(
                    zip(
                        ("FSCU5906804", "KKFU6721390", "ONEU9083430"),
                        (83, 461, 83),
                        strict=True,
                    )
                )
            ],
        }
    }
    items = tuple(
        HybridWorkItem(
            workItemId=f"W{index + 1:04d}",
            targetPaths=(
                f"documentPatch.cargoAllocationGroups[{index}].allocations[0].packageQuantity",
            ),
            action="replace",
            sourceValue=2184,
            targetValue=quantity,
            state="agent_residual",
            evidenceLineIds=(f"L{index + 2:05d}",),
            spanIds=(f"S{index + 1:03d}",),
            locator="sibling_object_evidence",
            rationale="columnar source evidence",
        )
        for index, quantity in enumerate((83, 461, 83))
    )
    bundle = SimpleNamespace(
        sourceLabel=source,
        targetLabel=target,
        cargoFlavorRewriteRequirements=(),
        operationalFlavorRequirements=(),
    )

    refined = _refine_numeric_evidence(text, items, bundle)

    assert [item.evidenceLineIds for item in refined] == [
        ("L00005",),
        ("L00006",),
        ("L00007",),
    ]


def test_allocation_quantities_follow_repeated_complete_cycles() -> None:
    text = (
        "MNBU4006124 MLBR0338050\n"
        "22 PLT 7103.882 KG\n"
        "MNBU9089862 MLBR0337964\n"
        "22 PLT 7100.788 KG\n"
        "SUMMARY\n"
        "22 PLT CY/CY\n"
        "22 PLT CY/CY\n"
    )
    source = {
        "documentPatch": {
            "containers": [
                {"containerNumber": "MNBU4006124"},
                {"containerNumber": "MNBU9089862"},
            ],
            "cargoAllocationGroups": [
                {
                    "groupId": "g1",
                    "allocations": [
                        {"containerNumber": "MNBU4006124", "packageQuantity": 22},
                        {"containerNumber": "MNBU9089862", "packageQuantity": 22},
                    ],
                }
            ],
        }
    }
    target = {
        "documentPatch": {
            **source["documentPatch"],
            "cargoAllocationGroups": [
                {
                    "groupId": "g1",
                    "allocations": [
                        {"containerNumber": "MNBU4006124", "packageQuantity": 4},
                        {"containerNumber": "MNBU9089862", "packageQuantity": 3},
                    ],
                }
            ],
        }
    }
    first = HybridWorkItem(
        workItemId="W0001",
        targetPaths=("documentPatch.cargoAllocationGroups[0].allocations[0].packageQuantity",),
        action="replace",
        sourceValue=22,
        targetValue=4,
        state="agent_residual",
        evidenceLineIds=("L00001",),
        spanIds=("S001",),
        locator="sibling_object_evidence",
        rationale="repeated source evidence",
    )
    second = first.model_copy(
        update={
            "workItemId": "W0002",
            "targetPaths": (
                "documentPatch.cargoAllocationGroups[0].allocations[1].packageQuantity",
            ),
            "targetValue": 3,
            "evidenceLineIds": ("L00003",),
            "spanIds": ("S002",),
        }
    )
    bundle = SimpleNamespace(
        sourceLabel=source,
        targetLabel=target,
        cargoFlavorRewriteRequirements=(),
        operationalFlavorRequirements=(),
    )

    refined = _refine_numeric_evidence(text, (first, second), bundle)

    assert refined[0].evidenceLineIds == ("L00002", "L00006")
    assert refined[1].evidenceLineIds == ("L00004", "L00007")


def test_shipment_package_total_excludes_same_valued_equipment_size() -> None:
    text = "Total Items: 40\nCONT0000001\n40' HIGH CUBE\n40 PALLETS\n"
    source = {
        "documentPatch": {"cargoPackages": [{"groupId": "g1", "packageId": "p1", "quantity": 40}]}
    }
    target = {
        "documentPatch": {"cargoPackages": [{"groupId": "g1", "packageId": "p1", "quantity": 135}]}
    }
    item = HybridWorkItem(
        workItemId="W0001",
        targetPaths=("documentPatch.cargoPackages[0].quantity",),
        action="replace",
        sourceValue=40,
        targetValue=135,
        state="agent_residual",
        evidenceLineIds=("L00001", "L00003", "L00004"),
        spanIds=("S001",),
        locator="numeric_surface",
        rationale="broad numeric evidence",
    )
    bundle = SimpleNamespace(
        sourceLabel=source,
        targetLabel=target,
        cargoFlavorRewriteRequirements=(),
        operationalFlavorRequirements=(),
    )

    refined = _refine_numeric_evidence(text, (item,), bundle)

    assert refined[0].evidenceLineIds == ("L00001",)
    assert refined[0].locator == "relation_scoped_numeric_surface"


def test_package_category_uses_relation_owned_allocation_rows() -> None:
    source = {
        "documentPatch": {
            "cargoPackages": [
                {"groupId": "g1", "packageId": "p1", "typeCategory": "PACKAGE_PALLET"}
            ],
            "cargoAllocationGroups": [
                {
                    "groupId": "g1",
                    "allocations": [{"containerNumber": "CONT0000001", "packageQuantity": 20}],
                }
            ],
        }
    }
    target = {
        "documentPatch": {
            "cargoPackages": [
                {"groupId": "g1", "packageId": "p1", "typeCategory": "PACKAGE_CARTON"}
            ],
            "cargoAllocationGroups": [
                {
                    "groupId": "g1",
                    "allocations": [{"containerNumber": "CONT0000001", "packageQuantity": 68}],
                }
            ],
        }
    }
    package_type = HybridWorkItem(
        workItemId="W0001",
        targetPaths=("documentPatch.cargoPackages[0].typeCategory",),
        action="replace",
        sourceValue="PACKAGE_PALLET",
        targetValue="PACKAGE_CARTON",
        state="agent_residual",
        evidenceLineIds=("L00001", "L00003"),
        spanIds=("S001",),
        locator="sibling_object_evidence",
        rationale="broad sibling evidence",
    )
    allocation = HybridWorkItem(
        workItemId="W0002",
        targetPaths=("documentPatch.cargoAllocationGroups[0].allocations[0].packageQuantity",),
        action="replace",
        sourceValue=20,
        targetValue=68,
        state="agent_residual",
        evidenceLineIds=("L00004", "L00005"),
        spanIds=("S002",),
        locator="relation_scoped_numeric_surface",
        rationale="container-owned package rows",
    )
    bundle = SimpleNamespace(sourceLabel=source, targetLabel=target)

    refined = _refine_sibling_evidence((package_type, allocation), bundle)

    assert refined[0].evidenceLineIds == ("L00004", "L00005")


def test_material_qualified_package_quantity_is_explicit() -> None:
    assert _is_explicit_package_quantity_line("47 WOODEN CASES", 47) is True
    assert _is_explicit_package_quantity_line("47 CORRUGATED CARTONS", 47) is True


def test_party_occurrence_refinement_resolves_uneven_repeated_role_blocks() -> None:
    text = (
        "--- PAGE 1 ---\nCONSIGNEE\nOLD COMPANY\n"
        "--- PAGE 2 ---\nCONSIGNEE\nOLD COMPANY\nNOTIFY PARTY\nOLD COMPANY\n"
    )
    items = (
        HybridWorkItem(
            workItemId="W0001",
            targetPaths=("documentPatch.parties.consignee.name",),
            action="replace",
            sourceValue="OLD COMPANY",
            targetValue="NEW CONSIGNEE LTD",
            state="agent_residual",
            evidenceLineIds=("L00003", "L00006"),
            spanIds=("S001", "S002"),
            locator="literal_in_party_role_block",
            rationale="two role-owned consignee copies",
        ),
        HybridWorkItem(
            workItemId="W0002",
            targetPaths=("documentPatch.parties.notifyParties[0].name",),
            action="replace",
            sourceValue="OLD COMPANY",
            targetValue="NEW NOTIFY LTD",
            state="agent_residual",
            evidenceLineIds=("L00008",),
            spanIds=("S003",),
            locator="literal_in_party_role_block",
            rationale="one role-owned notify copy",
        ),
    )
    requirements = (
        TargetValueOccurrenceRequirement(
            targetPaths=("documentPatch.parties.consignee.name",),
            targetValue="NEW CONSIGNEE LTD",
            requiredOccurrences=2,
        ),
        TargetValueOccurrenceRequirement(
            targetPaths=("documentPatch.parties.notifyParties[0].name",),
            targetValue="NEW NOTIFY LTD",
            requiredOccurrences=1,
        ),
    )

    refined = _refine_party_occurrence_requirements(text, text, items, requirements)

    assert {row.targetValue: row.requiredOccurrences for row in refined} == {
        "NEW CONSIGNEE LTD": 2,
        "NEW NOTIFY LTD": 1,
    }


def test_occurrence_repair_can_select_inventory_owned_changed_party_line() -> None:
    target = "Bluehaven Maritime Lines, Ltd."
    slots = (
        _Slot(
            alias="s0",
            line_id="L00001",
            source_line="as agents of OLD CARRIER",
            requirements=(
                {
                    "kind": "changed_source_occurrence",
                    "paths": ["documentPatch.parties.carrier.name"],
                },
            ),
        ),
    )
    workspace = RewriteWorkspace(
        original_text="as agents of OLD CARRIER\n",
        current_text="as agents of OLD CARRIER\n",
        target_value_occurrence_requirements=(
            TargetValueOccurrenceRequirement(
                targetPaths=("documentPatch.parties.carrier.name",),
                targetValue=target,
                requiredOccurrences=1,
            ),
        ),
    )
    material = SimpleNamespace(compiled=SimpleNamespace(workspace=workspace), slots=slots)

    selected, diagnostics = _target_occurrence_repair(
        material,
        {"s0": "as agents of OLD CARRIER"},
    )

    assert tuple(row.alias for row in selected) == ("s0",)
    assert diagnostics[0]["observedOccurrences"] == 0
    assert diagnostics[0]["roleOwnedLineIds"] == ["L00001"]


def test_occurrence_repair_expands_embedded_party_copy_to_enclosing_party_group() -> None:
    notify_target = "BRIGHTHARBOR TRADE SERVICES LLC"
    shipper_name = HybridWorkItem(
        workItemId="W0001",
        targetPaths=("documentPatch.parties.shipper.name",),
        action="replace",
        sourceValue="OLD SHIPPER",
        targetValue="NEW SHIPPER",
        state="agent_residual",
        evidenceLineIds=("L00001",),
        spanIds=("S001",),
        locator="exact_literal",
        rationale="shipper name",
    )
    shipper_address = HybridWorkItem(
        workItemId="W0002",
        targetPaths=("documentPatch.parties.shipper.address",),
        action="replace",
        sourceValue="C/O OLD NOTIFY OLD ADDRESS",
        targetValue="NEW ADDRESS",
        state="agent_residual",
        evidenceLineIds=("L00002", "L00003"),
        spanIds=("S002",),
        locator="party_role_block",
        rationale="shipper address",
    )
    notify_name = HybridWorkItem(
        workItemId="W0003",
        targetPaths=("documentPatch.parties.notifyParties[0].name",),
        action="replace",
        sourceValue="OLD NOTIFY",
        targetValue=notify_target,
        state="agent_residual",
        evidenceLineIds=("L00004",),
        spanIds=("S003",),
        locator="exact_literal",
        rationale="notify name",
    )
    slots = tuple(
        _Slot(
            alias=f"s{number - 1}",
            line_id=f"L{number:05d}",
            source_line=value,
            requirements=(
                {
                    "kind": "changed_source_occurrence",
                    "paths": ["documentPatch.parties.notifyParties[0].name"],
                },
            )
            if number in {2, 4, 5}
            else (),
        )
        for number, value in enumerate(
            ("OLD SHIPPER", "C/O OLD NOTIFY", "OLD ADDRESS", "OLD NOTIFY", "EXPORTER OLD NOTIFY"),
            start=1,
        )
    )
    workspace = RewriteWorkspace(
        original_text="\n".join(slot.source_line for slot in slots) + "\n",
        current_text="\n".join(slot.source_line for slot in slots) + "\n",
        target_value_occurrence_requirements=(
            TargetValueOccurrenceRequirement(
                targetPaths=("documentPatch.parties.notifyParties[0].name",),
                targetValue=notify_target,
                requiredOccurrences=3,
            ),
        ),
    )
    material = SimpleNamespace(
        compiled=SimpleNamespace(
            workspace=workspace,
            work_items=(shipper_name, shipper_address, notify_name),
        ),
        slots=slots,
    )

    selected, diagnostics = _target_occurrence_repair(
        material,
        {
            "s0": "NEW SHIPPER",
            "s1": "C/O NEW ADDRESS",
            "s2": "NEW ADDRESS",
            "s3": notify_target,
            "s4": f"EXPORTER {notify_target}",
        },
    )

    assert tuple(row.alias for row in selected) == ("s0", "s1", "s2", "s3", "s4")
    assert diagnostics[0]["repairLineIds"] == [
        "L00001",
        "L00002",
        "L00003",
        "L00004",
        "L00005",
    ]


def test_package_quantity_grammar_excludes_vat_ordinals_and_original_counts() -> None:
    lines = (
        "VAT No. EZ082575082 831652320155 / 475572 / 1273 / 3 / 6",
        "(3) NOTIFY ADDRESS:",
        "3*",
        "3 / Three",
        "3 pallets 719,000",
        "3* 719,000 KG",
    )

    assert [_is_explicit_package_quantity_line(line, 3) for line in lines] == [
        False,
        False,
        True,
        False,
        True,
        True,
    ]


def test_numeric_evidence_limits_temperature_to_thermal_context() -> None:
    text = (
        "CARRYING TEMPERATURE: -21.0 C\n"
        "TEMPERATURE TO BE SET AT +1,0 C\n"
        "HLBU 7872400 1 CONT. 40'X9'6\" REEFER CONTAINER SLAC*\n"
    )
    item = HybridWorkItem(
        workItemId="W0001",
        targetPaths=("documentPatch.containers[0].temperatureSetpoint.value",),
        action="replace",
        sourceValue=1.0,
        targetValue=-21.0,
        state="agent_residual",
        evidenceLineIds=("L00001", "L00002", "L00003"),
        spanIds=("S001",),
        locator="numeric_surface",
        rationale="global numeric evidence",
    )
    bundle = SimpleNamespace(
        sourceLabel={},
        targetLabel={},
        cargoFlavorRewriteRequirements=(),
        operationalFlavorRequirements=(),
    )

    refined = _refine_numeric_evidence(text, (item,), bundle)

    assert refined[0].evidenceLineIds == ("L00002",)
    assert refined[0].locator == "thermal_context_numeric_surface"


def test_party_occurrence_count_retains_real_name_containing_heading_word() -> None:
    assert (
        _party_scalar_occurrence_count("CONSIGNEE LOGISTICS LTD.\n", "CONSIGNEE LOGISTICS LTD.")
        == 1
    )


def test_full_audit_accepts_path_anchored_party_contact_outside_role_block() -> None:
    source = {
        "documentPatch": {
            "parties": {
                "deliveryAgent": {
                    "name": "MLH SHIPPING",
                    "contactDetails": {"emailAddresses": ["old@mlh-shipping.example"]},
                }
            }
        }
    }
    target = {
        "documentPatch": {
            "parties": {
                "deliveryAgent": {
                    "name": "MLH SHIPPING",
                    "contactDetails": {"emailAddresses": ["dispatch@lamgiang.example"]},
                }
            }
        }
    }
    source_text = (
        "Delivery Agent at place of delivery\nMLH SHIPPING\n\nEMAIL:old@mlh-shipping.example\n"
    )
    output_text = source_text.replace("old@mlh-shipping.example", "dispatch@lamgiang.example")
    anchor = AnchoredScalarReplacementRequirement(
        targetPaths=("documentPatch.parties.deliveryAgent.contactDetails.emailAddresses[0]",),
        sourceLineIds=("L00004",),
        sourceSurface="old@mlh-shipping.example",
        targetSurface="dispatch@lamgiang.example",
    )

    without_anchor = audit_full_document(
        document_id="doc_" + "8" * 64,
        source_text=source_text,
        output_text=output_text,
        source_label=source,
        target_label=target,
        inventory=(),
        deterministic_edits=(),
        oracle_case=None,
    )
    with_anchor = audit_full_document(
        document_id="doc_" + "8" * 64,
        source_text=source_text,
        output_text=output_text,
        source_label=source,
        target_label=target,
        inventory=(),
        deterministic_edits=(),
        oracle_case=None,
        anchored_replacements=(anchor,),
    )

    assert without_anchor.passed is False
    assert {row.category for row in without_anchor.findings} == {"semantic_slot_mismatch"}
    assert with_anchor.passed is True


def test_full_audit_accepts_party_name_fused_to_ocr_heading_in_owned_block() -> None:
    source = {"documentPatch": {"parties": {"notifyParties": [{"name": "OLD TRADER"}]}}}
    target = {"documentPatch": {"parties": {"notifyParties": [{"name": "Cedar Gate Ltd."}]}}}
    item = HybridWorkItem(
        workItemId="W0001",
        targetPaths=("documentPatch.parties.notifyParties[0].name",),
        action="replace",
        sourceValue="OLD TRADER",
        targetValue="Cedar Gate Ltd.",
        state="agent_residual",
        evidenceLineIds=("L00001",),
        spanIds=("S001",),
        locator="party_role_block",
        rationale="OCR concatenated heading and value",
    )

    audit = audit_full_document(
        document_id="doc_" + "a" * 64,
        source_text="Notify PartyOLD TRADER\n",
        output_text="Notify PartyCedar Gate Ltd.\n",
        source_label=source,
        target_label=target,
        inventory=(),
        deterministic_edits=(),
        oracle_case=None,
        work_items=(item,),
    )

    assert audit.passed is True


def test_full_audit_does_not_assign_carrier_header_name_to_address_slot() -> None:
    source = {
        "documentPatch": {
            "parties": {"carrier": {"name": "SOURCE LINE", "address": "12 SOURCE QUAY"}}
        }
    }
    target = {
        "documentPatch": {
            "parties": {"carrier": {"name": "TARGET LINE", "address": "88 TARGET WHARF"}}
        }
    }
    source_text = "SOURCE LINE\n12 SOURCE QUAY\n"
    output_text = "TARGET LINE\n88 TARGET WHARF\n"
    address_item = HybridWorkItem(
        workItemId="W0001",
        targetPaths=("documentPatch.parties.carrier.address",),
        action="replace",
        sourceValue="12 SOURCE QUAY",
        targetValue="88 TARGET WHARF",
        state="agent_residual",
        evidenceLineIds=("L00002",),
        spanIds=("S001",),
        locator="exact_literal",
        rationale="carrier address line",
    )
    header = SurfaceRenderingRequirement(
        kind="carrier_header_identity",
        targetPath="documentPatch.parties.carrier.name",
        sourceSurface="SOURCE LINE",
        targetSurface="TARGET LINE",
        sourceOccurrences=1,
        contextEvidence="SOURCE LINE\n12 SOURCE QUAY",
    )

    audit = audit_full_document(
        document_id="doc_" + "d" * 64,
        source_text=source_text,
        output_text=output_text,
        source_label=source,
        target_label=target,
        inventory=(),
        deterministic_edits=(),
        oracle_case=None,
        work_items=(address_item,),
        surface_requirements=(header,),
    )

    assert audit.passed is True


def test_full_audit_does_not_treat_party_value_inside_heading_as_stale_data() -> None:
    source = {"documentPatch": {"parties": {"consignee": {"name": "TO ORDER"}}}}
    target = {
        "documentPatch": {"parties": {"consignee": {"name": "Kestrel Meridian Trading Ltd."}}}
    }
    heading = "CONSIGNEE (3) (NOT NEGOTIABLE UNLESS CONSIGNED TO ORDER)"
    source_text = f"{heading}\nTO ORDER\n"
    output_text = f"{heading}\nKestrel Meridian Trading Ltd.\n"
    item = HybridWorkItem(
        workItemId="W0001",
        targetPaths=("documentPatch.parties.consignee.name",),
        action="replace",
        sourceValue="TO ORDER",
        targetValue="Kestrel Meridian Trading Ltd.",
        state="agent_residual",
        evidenceLineIds=("L00002",),
        spanIds=("S001",),
        locator="exact_literal",
        rationale="exact party value",
    )
    inventory = build_mutable_inventory(
        source_text=source_text,
        current_text=source_text,
        source_label=source,
        target_label=target,
        work_items=(item,),
        oracle_case=None,
    )

    audit = audit_full_document(
        document_id="doc_" + "9" * 64,
        source_text=source_text,
        output_text=output_text,
        source_label=source,
        target_label=target,
        inventory=inventory,
        deterministic_edits=(),
        oracle_case=None,
        work_items=(item,),
    )

    changed = [row for row in inventory if row.category == "changed_source_occurrence"]
    assert len(changed) == 1
    assert changed[0].lineIds == ("L00002",)
    assert audit.passed is True


def test_dynamic_patch_schema_requires_compound_realization_metadata() -> None:
    slots = (_Slot(alias="s0", line_id="L00001", source_line="OLD", requirements=()),)
    requirement = CompoundPartyFlavorRequirement(
        targetPath="documentPatch.parties.carrier.name",
        relationships=("trading_as",),
        sourceLabelName="Old Carrier trading as Old Brand",
        targetPrimaryName="New Carrier Ltd.",
    )
    compound = (_CompoundSlot(alias="c0", requirement=requirement),)
    _output_type, schema = _output_schema(slots, compound)

    assert schema["required"] == ["s0", "c0"]
    assert (
        _validated_compound_realizations(
            compound,
            {
                "s0": "NEW CARRIER LTD. TRADING AS NORTHSTAR LINE",
                "c0": "New Carrier Ltd. trading as Northstar Line",
            },
        )[0].targetPrimaryName
        == "New Carrier Ltd."
    )
    with pytest.raises(ValueError, match="compound slot is not a populated string"):
        _validated_compound_realizations(compound, {"s0": "NEW CARRIER LTD."})


def test_required_semantic_surface_rejects_truncated_cargo_description() -> None:
    target = "Other flat-rolled products of stainless steel, of a width of less than 600 mm"
    slot = _Slot(
        alias="s0",
        line_id="L00001",
        source_line="FRESH GARLIC",
        requirements=(
            {
                "kind": "required_semantic_surface",
                "paths": ["documentPatch.cargoGroups[0].description"],
                "target": target,
                "policy": "render_complete_target_modulo_casing_on_this_line",
            },
        ),
    )

    assert _validated_output_replacements((slot,), {"s0": target.upper()})
    with pytest.raises(ValueError, match="truncates a required semantic surface"):
        _validated_output_replacements(
            (slot,), {"s0": "OTHER FLAT-ROLLED PRODUCTS OF STAINLESS STEEL"}
        )


def test_temperature_deactivation_owns_only_thermal_numeric_line() -> None:
    text = (
        "CNTR. NOS. WISEAL NOS.\n"
        "FSCU4850130 / HF34458XZ\n"
        "AT THE CARRYING TEMPERATURE OF -3\n"
        "DEGREES CELSIUS.\n"
    )

    lines = _temperature_deactivation_line_numbers(text, {"value": -3.0, "unit": "celsius"})

    assert lines == {3}


def test_locked_prefill_literal_is_exposed_and_cannot_be_regenerated() -> None:
    prefill = AppliedDeterministicPrefill(
        lineId="L00001",
        targetPaths=("documentPatch.parties.shipper.contactDetails.phoneNumbers[0]",),
        sourceSurface="00971506714549",
        targetSurface="+62 31 8894 2716",
        beforeLine="TEL. NO 00971506714549",
        afterLine="TEL. NO +62 31 8894 2716",
    )
    workspace = RewriteWorkspace(
        original_text="TEL. NO 00971506714549\n",
        current_text="TEL. NO +62 31 8894 2716\n",
        deterministic_prefills=[prefill],
    )
    requirement = _locked_literal_requirements_by_line(workspace)["L00001"][0]
    slot = _Slot(
        alias="s0",
        line_id="L00001",
        source_line="TEL. NO +62 31 8894 2716",
        requirements=(requirement,),
    )

    assert requirement["kind"] == "host_locked_target_literal"
    assert _validated_output_replacements((slot,), {"s0": "TEL. NO +62 31 8894 2716 (EXPORT)"})[
        0
    ].newText.endswith("(EXPORT)")
    assert (
        _validated_output_replacements((slot,), {"s0": "TEL. NO +62 31 8894 2716 (export)"})[
            0
        ].newText
        == "TEL. NO +62 31 8894 2716 (export)"
    )
    with pytest.raises(ValueError, match="host-locked literals"):
        _validated_output_replacements((slot,), {"s0": "TEL. NO +62 31 7742 6189"})


def test_raw_auxiliary_identity_preserves_legal_prefix_and_suffix() -> None:
    requirement = RawAuxiliaryIdentityRequirement(
        requirementId="carrier-agent-L00001",
        relationship="agent_for_carrier",
        sourceIdentity="MSC KOREA LIMITED",
        sourceIdentityLineCount=1,
        gapLineCount=0,
        consistencyGroupId="carrier-agent-group-1",
        targetPrincipalName="AURELIA HORIZON MARITIME",
        sourceEvidence="by MSC KOREA LIMITED As Agent For The Carrier",
    )
    workspace = RewriteWorkspace(
        original_text="by MSC KOREA LIMITED As Agent For The Carrier\n",
        current_text="by MSC KOREA LIMITED As Agent For The Carrier\n",
        raw_auxiliary_identity_requirements=(requirement,),
    )
    requirements = _locked_literal_requirements_by_line(workspace)["L00001"]
    slot = _Slot(
        alias="s0",
        line_id="L00001",
        source_line="by MSC KOREA LIMITED As Agent For The Carrier",
        requirements=tuple(requirements),
    )

    assert _validated_output_replacements(
        (slot,),
        {"s0": "by FICTIONAL SHIPPING LTD As Agent For The Carrier"},
    )
    with pytest.raises(ValueError, match="host-locked legal affixes"):
        _validated_output_replacements(
            (slot,),
            {"s0": "for FICTIONAL SHIPPING LTD acting for Carrier"},
        )


def test_wrapped_on_board_agent_locks_only_the_static_heading_prefix() -> None:
    requirement = RawAuxiliaryIdentityRequirement(
        requirementId="carrier-agent-L00001",
        relationship="agent_for_carrier",
        sourceIdentity="OLD AGENCY\nPTE LTD",
        sourceIdentityLineCount=2,
        gapLineCount=0,
        consistencyGroupId="carrier-agent-group-1",
        targetPrincipalName="THE CARRIER",
        sourceEvidence=(
            "Shipped on Board OLD VESSEL 01-JAN-2024 OLD AGENCY\nPTE LTD as agents for the Carrier"
        ),
    )
    workspace = RewriteWorkspace(
        original_text=requirement.sourceEvidence + "\n",
        current_text=requirement.sourceEvidence + "\n",
        raw_auxiliary_identity_requirements=(requirement,),
    )

    locked = _locked_literal_requirements_by_line(workspace)

    prefixes = [
        row["target"] for row in locked["L00001"] if row["kind"] == "host_locked_source_prefix"
    ]
    assert prefixes == ["Shipped on Board "]


def test_numeric_inventory_ignores_date_components() -> None:
    assert _numeric_lines("15-Sep-2023\n15.000 cu. m.\n", 15.0) == {2}


def test_signed_carrier_principal_is_not_a_source_only_signing_identity() -> None:
    assert (
        _SIGNED_CARRIER_IDENTITY.fullmatch("Signed for the Carrier AURELIS OCEAN TRANSPORT S.A. by")
        is None
    )


def test_generic_raw_agent_identity_lines_are_recognized() -> None:
    text = (
        "Shipped on Board SOURCE VESSEL 19-MAY-2023 OLD SHIPPING AGENCY\n"
        "PTE LTD As agents for the Carrier\n"
    )

    assert _raw_agent_identity_line_numbers(
        text,
        source_carrier="OLD OCEAN LINE",
        target_carrier="NEW MERIDIAN LINE",
    ) == {1, 2}


def test_output_validation_reports_all_missing_host_locks_in_one_repair() -> None:
    slots = tuple(
        _Slot(
            alias=f"s{index}",
            line_id=f"L{index + 1:05d}",
            source_line=f"COUNTRY: OLD-{index}",
            requirements=(
                {
                    "kind": "host_locked_target_literal",
                    "paths": [f"documentPatch.route.field{index}"],
                    "target": target,
                    "policy": "preserve_verbatim_on_this_line",
                },
            ),
        )
        for index, target in enumerate(("SINGAPORE", "KOREA, REPUBLIC OF"))
    )

    with pytest.raises(ValueError) as error:
        _validated_output_replacements(
            slots,
            {"s0": "MALAYSIA", "s1": "JAPAN"},
        )

    assert "s0" in str(error.value)
    assert "s1" in str(error.value)

    corrected = _validated_output_replacements(
        slots,
        {"s0": "COUNTRY: Singapore", "s1": "COUNTRY: Korea, Republic of"},
    )
    assert [row.newText for row in corrected] == [
        "COUNTRY: SINGAPORE",
        "COUNTRY: KOREA, REPUBLIC OF",
    ]


def test_slot_local_party_error_expands_to_complete_role_group() -> None:
    party_slots = (
        _Slot(
            alias="s0",
            line_id="L00001",
            source_line="OLD ADDRESS",
            requirements=(
                {
                    "kind": "task_label_delta",
                    "paths": ["documentPatch.parties.consignee.address"],
                    "target": "NEW ADDRESS",
                },
            ),
        ),
        _Slot(
            alias="s1",
            line_id="L00002",
            source_line="OLD CITY",
            requirements=(
                {
                    "kind": "task_label_delta",
                    "paths": ["documentPatch.parties.consignee.city"],
                    "target": "NEW CITY",
                },
            ),
        ),
        _Slot(
            alias="s2",
            line_id="L00003",
            source_line="OLD PORT",
            requirements=(
                {
                    "kind": "task_label_delta",
                    "paths": ["documentPatch.route.portOfLoading.name"],
                    "target": "NEW PORT",
                },
            ),
        ),
    )

    selected, _compounds = _repair_selection(
        party_slots,
        (),
        error_message="provider output slot omits host-locked literals: s0",
    )

    assert [slot.alias for slot in selected] == ["s0", "s1"]


def test_handling_instruction_affix_is_locked_outside_changed_value() -> None:
    source = (
        "VENTILLATION REQUIRED BY SHIPPER TO BE SET AT 10 CBM PER HOUR. "
        "CARRIAGE PER CLAUSE 11 OF THIS B/L."
    )
    item = HybridWorkItem(
        workItemId="W0001",
        targetPaths=("documentPatch.cargoGroups[0].handlingInstructions[0]",),
        action="replace",
        sourceValue="VENTILLATION REQUIRED BY SHIPPER TO BE SET AT 10 CBM PER HOUR",
        targetValue="KEEP DRY AND PROTECTED FROM SURFACE DAMAGE.",
        state="agent_residual",
        evidenceLineIds=("L00001",),
        spanIds=("S001",),
        locator="exact_literal",
        rationale="test",
    )
    workspace = RewriteWorkspace(original_text=source + "\n", current_text=source + "\n")

    requirements = _handling_affix_requirements_by_line(workspace, (item,))["L00001"]
    slot = _Slot(
        alias="s0",
        line_id="L00001",
        source_line=source,
        requirements=tuple(requirements),
    )

    suffix = ". CARRIAGE PER CLAUSE 11 OF THIS B/L."
    assert requirements[0]["target"] == suffix
    assert _validated_output_replacements(
        (slot,),
        {"s0": "KEEP DRY AND PROTECTED FROM SURFACE DAMAGE." + suffix},
    )
    with pytest.raises(ValueError, match="host-locked literals"):
        _validated_output_replacements(
            (slot,), {"s0": "KEEP DRY AND PROTECTED FROM SURFACE DAMAGE."}
        )


def test_target_embedded_surface_is_licensed_only_on_the_rendered_target_line() -> None:
    source = {
        "documentPatch": {
            "parties": {"consignee": {"city": "ALEXANDRIA"}},
            "route": {"portOfDischarge": {"name": "PORT SAID"}},
        }
    }
    target = {
        "documentPatch": {
            "parties": {"consignee": {"city": "Jurong West"}},
            "route": {"portOfDischarge": {"name": "El Iskandariya (Alexandria)"}},
        }
    }
    source_text = "ALEXANDRIA\nPORT SAID\nALEXANDRIA\n"
    good = "Jurong West\nEl Iskandariya (Alexandria)\nGENERAL TERMS\n"
    bad = "Jurong West\nEl Iskandariya (Alexandria)\nALEXANDRIA\n"
    document_id = "doc_" + "4" * 64

    passed = audit_full_document(
        document_id=document_id,
        source_text=source_text,
        output_text=good,
        source_label=source,
        target_label=target,
        inventory=(),
        deterministic_edits=(),
        oracle_case=None,
    )
    failed = audit_full_document(
        document_id=document_id,
        source_text=source_text,
        output_text=bad,
        source_label=source,
        target_label=target,
        inventory=(),
        deterministic_edits=(),
        oracle_case=None,
    )

    assert passed.passed is True
    assert failed.passed is False
    assert failed.findings[0].category == "retained_changed_source_value"
    assert failed.findings[0].lineIds == ("L00003",)


def test_exact_shared_target_surface_does_not_license_the_changed_role_line() -> None:
    source = {
        "documentPatch": {
            "parties": {"consignee": {"city": "ALEXANDRIA"}},
            "route": {"portOfDischarge": {"name": "PORT SAID"}},
        }
    }
    target = {
        "documentPatch": {
            "parties": {"consignee": {"city": "JURONG WEST"}},
            "route": {"portOfDischarge": {"name": "ALEXANDRIA"}},
        }
    }
    work_item = HybridWorkItem(
        workItemId="W0001",
        targetPaths=("documentPatch.parties.consignee.city",),
        action="replace",
        sourceValue="ALEXANDRIA",
        targetValue="JURONG WEST",
        state="agent_residual",
        evidenceLineIds=("L00001",),
        spanIds=("S001",),
        locator="party_role_block",
        rationale="test path ownership",
    )

    audit = audit_full_document(
        document_id="doc_" + "7" * 64,
        source_text="ALEXANDRIA\nPORT SAID\n",
        output_text="ALEXANDRIA\nALEXANDRIA\n",
        source_label=source,
        target_label=target,
        inventory=(),
        deterministic_edits=(),
        oracle_case=None,
        work_items=(work_item,),
    )

    assert audit.passed is False
    assert audit.findings[0].category == "retained_changed_source_value"
    assert audit.findings[0].lineIds == ("L00001",)


def test_shared_surface_is_licensed_on_other_target_paths_proven_line_only() -> None:
    source = {
        "documentPatch": {
            "parties": {"deliveryAgent": {"country": "THE NETHERLANDS"}},
            "route": {
                "portOfDischarge": {"country": "NETHERLANDS"},
                "placeOfDelivery": {"country": "NETHERLANDS"},
            },
        }
    }
    target = {
        "documentPatch": {
            "parties": {"deliveryAgent": {"country": "Netherlands"}},
            "route": {
                "portOfDischarge": {"country": "United States"},
                "placeOfDelivery": {"country": "United States"},
            },
        }
    }
    work_items = (
        HybridWorkItem(
            workItemId="W0002",
            targetPaths=(
                "documentPatch.route.portOfDischarge.country",
                "documentPatch.route.placeOfDelivery.country",
            ),
            action="replace",
            sourceValue="NETHERLANDS",
            targetValue="United States",
            state="agent_residual",
            evidenceLineIds=("L00001", "L00002"),
            spanIds=("S002", "S003"),
            locator="exact_literal",
            rationale="test overlapping source evidence",
        ),
    )

    passed = audit_full_document(
        document_id="doc_" + "8" * 64,
        source_text="AGENT: THE NETHERLANDS\nPORT: NETHERLANDS\n",
        output_text="AGENT: Netherlands\nPORT: United States\n",
        source_label=source,
        target_label=target,
        inventory=(),
        deterministic_edits=(),
        oracle_case=None,
        deterministic_prefills=(
            AppliedDeterministicPrefill(
                lineId="L00001",
                targetPaths=("documentPatch.parties.deliveryAgent.country",),
                sourceSurface="THE NETHERLANDS",
                targetSurface="NETHERLANDS",
                beforeLine="AGENT: THE NETHERLANDS",
                afterLine="AGENT: NETHERLANDS",
            ),
        ),
        work_items=work_items,
    )
    failed = audit_full_document(
        document_id="doc_" + "9" * 64,
        source_text="AGENT: THE NETHERLANDS\nPORT: NETHERLANDS\n",
        output_text="AGENT: Netherlands\nPORT: NETHERLANDS\n",
        source_label=source,
        target_label=target,
        inventory=(),
        deterministic_edits=(),
        oracle_case=None,
        deterministic_prefills=(
            AppliedDeterministicPrefill(
                lineId="L00001",
                targetPaths=("documentPatch.parties.deliveryAgent.country",),
                sourceSurface="THE NETHERLANDS",
                targetSurface="NETHERLANDS",
                beforeLine="AGENT: THE NETHERLANDS",
                afterLine="AGENT: NETHERLANDS",
            ),
        ),
        work_items=work_items,
    )

    assert passed.passed is True
    assert failed.passed is False
    assert failed.findings[0].lineIds == ("L00002",)


def test_party_locality_ownership_covers_shorthand_repeat_inside_role_block() -> None:
    text = (
        "DESTINATION AGENT\n"
        "OLD DELIVERY BV\n"
        "ROTTERDAM, THE NETHERLANDS\n"
        "NL809653035000\n"
        "ROTTERDAM NETHERLANDS\n"
        "\n"
        "PORT OF DISCHARGE\n"
        "ROTTERDAM NETHERLANDS\n"
    )
    source_label = {
        "documentPatch": {
            "parties": {
                "deliveryAgent": {
                    "name": "OLD DELIVERY BV",
                    "city": "ROTTERDAM",
                    "country": "THE NETHERLANDS",
                }
            }
        }
    }

    assert _party_owned_surface_lines(text, source_label, "NETHERLANDS") == {3, 5}


def test_legal_relation_surface_is_protected_but_not_granted_model_authority() -> None:
    source = {
        "documentPatch": {
            "parties": {
                "shipper": {"name": "SOURCE SHIPPER"},
                "carrier": {"name": "SOURCE CARRIER"},
            }
        }
    }
    target = {
        "documentPatch": {
            "parties": {
                "shipper": {"name": "TARGET SHIPPER"},
                "carrier": {"name": "TARGET CARRIER"},
            }
        }
    }
    source_text = "SOURCE SHIPPER\nas agent for and on behalf of\nSOURCE CARRIER\n"
    output_text = "TARGET SHIPPER\nas agent for and on behalf of\nTARGET CARRIER\n"
    inventory = build_mutable_inventory(
        source_text=source_text,
        current_text=source_text,
        source_label=source,
        target_label=target,
        work_items=(),
        oracle_case=None,
    )

    audit = audit_full_document(
        document_id="doc_" + "5" * 64,
        source_text=source_text,
        output_text=output_text,
        source_label=source,
        target_label=target,
        inventory=inventory,
        deterministic_edits=(),
        oracle_case=None,
    )

    assert not any(row.sourceSurface == "as agent for and on behalf of" for row in inventory)
    assert audit.passed is True

    altered = audit_full_document(
        document_id="doc_" + "5" * 64,
        source_text=source_text,
        output_text="TARGET SHIPPER\nfor TARGET CARRIER\nTARGET CARRIER\n",
        source_label=source,
        target_label=target,
        inventory=inventory,
        deterministic_edits=(),
        oracle_case=None,
    )
    assert altered.passed is False
    assert any(row.category == "legal_relation_topology_mismatch" for row in altered.findings)


def test_inventory_covers_carrier_branding_and_source_only_free_time() -> None:
    source = {"documentPatch": {"parties": {"carrier": {"name": "OCEAN NETWORK"}}}}
    target = {"documentPatch": {"parties": {"carrier": {"name": "MERIDIAN TIDE"}}}}
    text = "website at www.one-line.com\n(ONE), AS CARRIER\n21 DAYS FREETIME OF DEMURRAGE\n"

    inventory = build_mutable_inventory(
        source_text=text,
        current_text=text,
        source_label=source,
        target_label=target,
        work_items=(),
        oracle_case=None,
    )

    surfaces = {(row.category, row.sourceSurface) for row in inventory}
    assert ("carrier_dependent_branding", "www.one-line.com") in surfaces
    assert ("carrier_dependent_branding", "(ONE)") in surfaces
    assert (
        "source_only_operational_scalar",
        "21 DAYS FREETIME OF DEMURRAGE",
    ) in surfaces


def test_inventory_does_not_reclassify_rendered_target_website_as_stale_branding() -> None:
    source = {
        "documentPatch": {
            "parties": {
                "carrier": {
                    "name": "MSC Mediterranean Shipping Company S.A.",
                    "contactDetails": {"websiteUrls": ["https://www.msc.com"]},
                }
            }
        }
    }
    target = {
        "documentPatch": {
            "parties": {
                "carrier": {
                    "name": "Helviora Ocean Link AG",
                    "contactDetails": {"websiteUrls": ["https://www.helviora-oceanlink.ch"]},
                }
            }
        }
    }
    text = "website: https://www.helviora-oceanlink.ch\n"

    inventory = build_mutable_inventory(
        source_text="website: https://www.msc.com\n",
        current_text=text,
        source_label=source,
        target_label=target,
        work_items=(),
        oracle_case=None,
    )

    assert not any(row.category == "carrier_dependent_branding" for row in inventory)


def test_inventory_exposes_carrier_signing_affiliate_as_distinct_flavor() -> None:
    source = {
        "documentPatch": {"parties": {"carrier": {"name": "Ocean Network Express Pte. Ltd."}}}
    }
    target = {"documentPatch": {"parties": {"carrier": {"name": "Meridian Tide Maritime Ltd."}}}}
    text = (
        "SIGNED OCEAN NETWORK EXPRESS (CHINA) LTD.\n"
        "BY:\n"
        "QINGDAO BRANCH\n"
        "\n"
        "as agent for and on behalf of\n"
        "\n"
        "Ocean Network Express Pte. Ltd.\n"
    )

    inventory = build_mutable_inventory(
        source_text=text,
        current_text=text,
        source_label=source,
        target_label=target,
        work_items=(),
        oracle_case=None,
    )

    signing = [row for row in inventory if row.category == "source_only_signing_identity"]
    assert len(signing) == 1
    assert signing[0].sourceSurface == "OCEAN NETWORK EXPRESS (CHINA) LTD."
    assert signing[0].lineIds == ("L00001",)
    assert signing[0].targetSemantics == {
        "policy": "synthesize_distinct_carrier_signing_agent_or_affiliate",
        "targetCarrierName": "Meridian Tide Maritime Ltd.",
        "mustDifferFromTargetCarrier": True,
        "preserveSignedPrefix": True,
    }


def test_inventory_exposes_multiline_signed_by_identity_without_target_carrier() -> None:
    source = {"documentPatch": {"parties": {}}}
    target = {"documentPatch": {"parties": {}}}
    text = "Signed by:\nCOSCO SHIPPING LINES (SPAIN)\nS.A.\n\nAS AGENT\n"

    inventory = build_mutable_inventory(
        source_text=text,
        current_text=text,
        source_label=source,
        target_label=target,
        work_items=(),
        oracle_case=None,
    )

    signing = [row for row in inventory if row.category == "source_only_signing_identity"]
    assert len(signing) == 1
    assert signing[0].sourceSurface == "COSCO SHIPPING LINES (SPAIN)\nS.A."
    assert signing[0].lineIds == ("L00002", "L00003")
    assert signing[0].targetPaths == ()
    assert signing[0].targetSemantics == {
        "policy": "synthesize_distinct_carrier_signing_agent_or_affiliate",
        "targetCarrierName": None,
        "mustDifferFromNamedTargetParties": True,
        "preserveSignedByHeading": True,
        "preserveOccupiedLineCount": True,
    }


def test_full_audit_accepts_path_owned_date_concatenated_to_on_label() -> None:
    source_text = "SHIPPED ON BOARD ON2024-03-16 AT PORT\n"
    output_text = "SHIPPED ON BOARD ON2024-01-07 AT PORT\n"
    requirement = SurfaceRenderingRequirement(
        kind="date",
        targetPath="documentPatch.shippedOnBoardDate",
        sourceSurface="2024-03-16",
        targetSurface="2024-01-07",
        sourceOccurrences=1,
        contextEvidence=source_text.rstrip("\n"),
    )

    audit = audit_full_document(
        document_id="doc_" + "d" * 64,
        source_text=source_text,
        output_text=output_text,
        source_label={"documentPatch": {"shippedOnBoardDate": "2024-03-16"}},
        target_label={"documentPatch": {"shippedOnBoardDate": "2024-01-07"}},
        inventory=(),
        deterministic_edits=(),
        oracle_case=None,
        surface_requirements=(requirement,),
    )

    assert audit.passed is True


def test_freight_option_caption_is_not_a_mutable_arrangement_occurrence() -> None:
    source = {"documentPatch": {"freight": {"paymentArrangement": "prepaid"}}}
    target = {"documentPatch": {"freight": {"paymentArrangement": "collect"}}}
    source_text = (
        "FREIGHT PREPAID\n"
        "Freight & Charges Revenue Tons Rate Per Amount Prepaid Collect Payable at /by\n"
        "Rate Unit Currency Prepaid Collect\n"
        "Charges Name Prepaid/Collect Invoice Party Customer Code\n"
        "Documentation Fee - Origin Prepaid EXPORTER ACCOUNT\n"
    )
    output_text = (
        "FREIGHT COLLECT\n"
        "Freight & Charges Revenue Tons Rate Per Amount Prepaid Collect Payable at /by\n"
        "Rate Unit Currency Prepaid Collect\n"
        "Charges Name Prepaid/Collect Invoice Party Customer Code\n"
        "Documentation Fee - Origin Prepaid EXPORTER ACCOUNT\n"
    )
    item = HybridWorkItem(
        workItemId="W0001",
        targetPaths=("documentPatch.freight.paymentArrangement",),
        action="replace",
        sourceValue="prepaid",
        targetValue="collect",
        state="agent_residual",
        evidenceLineIds=("L00001",),
        spanIds=("S001",),
        locator="freight_arrangement_surface",
        rationale="selected freight arrangement",
    )
    inventory = build_mutable_inventory(
        source_text=source_text,
        current_text=source_text,
        source_label=source,
        target_label=target,
        work_items=(item,),
        oracle_case=None,
    )
    audit = audit_full_document(
        document_id="doc_" + "e" * 64,
        source_text=source_text,
        output_text=output_text,
        source_label=source,
        target_label=target,
        inventory=inventory,
        deterministic_edits=(),
        oracle_case=None,
        work_items=(item,),
    )

    assert [row.lineIds for row in inventory if row.category == "changed_source_occurrence"] == [
        ("L00001",)
    ]
    assert audit.passed is True


def test_inventory_accepts_path_target_on_line_when_source_number_changes_role() -> None:
    source = {
        "documentPatch": {"cargoPackages": [{"groupId": "g1", "packageId": "p1", "quantity": 20}]}
    }
    target = {
        "documentPatch": {"cargoPackages": [{"groupId": "g1", "packageId": "p1", "quantity": 7}]}
    }
    source_text = "20 PALLETS / 40HQ\n"
    output_text = "7 BOXES / 20' STANDARD HEIGHT GENERAL PURPOSE\n"
    item = HybridWorkItem(
        workItemId="W0001",
        targetPaths=("documentPatch.cargoPackages[0].quantity",),
        action="replace",
        sourceValue=20,
        targetValue=7,
        state="agent_residual",
        evidenceLineIds=("L00001",),
        spanIds=("S001",),
        locator="relation_scoped_numeric_surface",
        rationale="package row",
    )
    inventory = build_mutable_inventory(
        source_text=source_text,
        current_text=source_text,
        source_label=source,
        target_label=target,
        work_items=(item,),
        oracle_case=None,
    )
    audit = audit_full_document(
        document_id="doc_" + "f" * 64,
        source_text=source_text,
        output_text=output_text,
        source_label=source,
        target_label=target,
        inventory=inventory,
        deterministic_edits=(),
        oracle_case=None,
        work_items=(item,),
    )

    assert any(row.category == "changed_source_occurrence" for row in inventory)
    assert audit.passed is True


def test_inventory_partitions_shared_source_surface_by_path_owned_lines() -> None:
    source = {
        "documentPatch": {
            "freight": {"paymentPlace": {"name": "MUMBAI"}},
            "placeOfIssue": {"name": "MUMBAI"},
        }
    }
    target = {
        "documentPatch": {
            "freight": {"paymentPlace": {"name": "Jarajus"}},
            "placeOfIssue": {"name": "Quanzhou"},
        }
    }
    text = "FREIGHT PAYABLE AT MUMBAI\nPLACE OF ISSUE MUMBAI\n"
    items = (
        HybridWorkItem(
            workItemId="W0001",
            targetPaths=("documentPatch.freight.paymentPlace.name",),
            action="replace",
            sourceValue="MUMBAI",
            targetValue="Jarajus",
            state="agent_residual",
            evidenceLineIds=("L00001",),
            spanIds=("S001",),
            locator="location_role_surface",
            rationale="freight role",
        ),
        HybridWorkItem(
            workItemId="W0002",
            targetPaths=("documentPatch.placeOfIssue.name",),
            action="replace",
            sourceValue="MUMBAI",
            targetValue="Quanzhou",
            state="agent_residual",
            evidenceLineIds=("L00002",),
            spanIds=("S002",),
            locator="location_role_surface",
            rationale="issue role",
        ),
    )

    inventory = build_mutable_inventory(
        source_text=text,
        current_text=text,
        source_label=source,
        target_label=target,
        work_items=items,
        oracle_case=None,
    )

    changed = [row for row in inventory if row.category == "changed_source_occurrence"]
    assert [(row.lineIds, row.targetPaths, row.targetSemantics) for row in changed] == [
        (
            ("L00001",),
            ("documentPatch.freight.paymentPlace.name",),
            [{"path": "documentPatch.freight.paymentPlace.name", "target": "Jarajus"}],
        ),
        (
            ("L00002",),
            ("documentPatch.placeOfIssue.name",),
            [{"path": "documentPatch.placeOfIssue.name", "target": "Quanzhou"}],
        ),
    ]


def test_inventory_maps_unowned_party_scalar_copy_to_its_unique_target() -> None:
    source = {"documentPatch": {"parties": {"carrier": {"address": "12 SOURCE QUAY"}}}}
    target = {"documentPatch": {"parties": {"carrier": {"address": "88 TARGET WHARF"}}}}
    text = "CARRIER ADDRESS: 12 SOURCE QUAY\nLOCAL OFFICE: 12 SOURCE QUAY\n"
    item = HybridWorkItem(
        workItemId="W0001",
        targetPaths=("documentPatch.parties.carrier.address",),
        action="replace",
        sourceValue="12 SOURCE QUAY",
        targetValue="88 TARGET WHARF",
        state="agent_residual",
        evidenceLineIds=("L00001",),
        spanIds=("S001",),
        locator="exact_literal",
        rationale="carrier address slot",
    )

    inventory = build_mutable_inventory(
        source_text=text,
        current_text=text,
        source_label=source,
        target_label=target,
        work_items=(item,),
        oracle_case=None,
    )

    assert [
        (row.category, row.lineIds) for row in inventory if "changed_source" in row.category
    ] == [
        ("changed_source_occurrence", ("L00001",)),
        ("changed_source_occurrence", ("L00002",)),
    ]
    repeated = next(row for row in inventory if row.lineIds == ("L00002",))
    assert repeated.targetSemantics == [
        {
            "path": "documentPatch.parties.carrier.address",
            "target": "88 TARGET WHARF",
        }
    ]

    requirements = _include_unique_linked_party_copies(
        text,
        (item,),
        inventory,
        (
            TargetValueOccurrenceRequirement(
                targetPaths=("documentPatch.parties.carrier.address",),
                targetValue="88 TARGET WHARF",
                requiredOccurrences=1,
            ),
        ),
        (),
    )
    assert requirements[0].requiredOccurrences == 2


def test_inventory_keeps_unique_scalar_auxiliary_on_multi_target_context_line() -> None:
    source = {
        "documentPatch": {
            "parties": {
                "notifyParties": [{"city": "TAIPEI", "country": "TAIWAN"}],
            },
            "route": {"portOfLoading": {"country": "TAIWAN"}},
        }
    }
    target = {
        "documentPatch": {
            "parties": {
                "notifyParties": [{"city": "Qina", "country": "EGYPT"}],
            },
            "route": {"portOfLoading": {"country": "KOREA"}},
        }
    }
    text = (
        "NOTIFY PARTY\n"
        "TAIPEI, TAIWAN\n"
        "Collection Business Unit Maersk Taiwan Ltd - Taipei\n"
        "PORT OF LOADING\n"
        "TAIWAN\n"
    )
    items = (
        HybridWorkItem(
            workItemId="W0001",
            targetPaths=("documentPatch.parties.notifyParties[0].city",),
            action="replace",
            sourceValue="TAIPEI",
            targetValue="Qina",
            state="agent_residual",
            evidenceLineIds=("L00002",),
            spanIds=("S001",),
            locator="party_role_block",
            rationale="notify locality",
        ),
        HybridWorkItem(
            workItemId="W0002",
            targetPaths=("documentPatch.parties.notifyParties[0].country",),
            action="replace",
            sourceValue="TAIWAN",
            targetValue="EGYPT",
            state="agent_residual",
            evidenceLineIds=("L00002",),
            spanIds=("S002",),
            locator="party_role_block",
            rationale="notify country",
        ),
        HybridWorkItem(
            workItemId="W0003",
            targetPaths=("documentPatch.route.portOfLoading.country",),
            action="replace",
            sourceValue="TAIWAN",
            targetValue="KOREA",
            state="agent_residual",
            evidenceLineIds=("L00005",),
            spanIds=("S003",),
            locator="location_role_surface",
            rationale="port country",
        ),
    )

    inventory = build_mutable_inventory(
        source_text=text,
        current_text=text,
        source_label=source,
        target_label=target,
        work_items=items,
        oracle_case=None,
    )

    collection_city = next(
        row for row in inventory if row.sourceSurface == "TAIPEI" and row.lineIds == ("L00003",)
    )
    assert collection_city.category == "changed_source_auxiliary_copy"
    assert collection_city.targetSemantics == {
        "policy": "synthesize_distinct_context_compatible_auxiliary_value",
        "mustDifferFromSource": True,
        "mustNotDuplicateAnyLabeledTarget": True,
    }


def test_inventory_does_not_count_care_of_or_foreign_role_as_notify_copy() -> None:
    source = {
        "documentPatch": {
            "parties": {
                "shipper": {
                    "name": "ACE PRIMA RESOURCES SDN BHD",
                    "address": "C/O CHEMIDEX FZCO 23080-001 A2 BUILDING",
                },
                "notifyParties": [{"name": "CHEMIDEX FZCO"}],
            }
        }
    }
    target = {
        "documentPatch": {
            "parties": {
                "shipper": {
                    "name": "VAJRA MERIDIAN METALS PRIVATE LIMITED",
                    "address": "PLOT 18 AUTONAGAR INDUSTRIAL ESTATE",
                },
                "notifyParties": [{"name": "BRIGHTHARBOR TRADE SERVICES LLC"}],
            }
        }
    }
    text = (
        "SHIPPER\nACE PRIMA RESOURCES SDN BHD\nC/O CHEMIDEX FZCO 23080-001 A2 BUILDING\n"
        "NOTIFY PARTY\nCHEMIDEX FZCO\nEXPORTER: CHEMIDEX FZCO\n"
    )
    shipper_address = HybridWorkItem(
        workItemId="W0001",
        targetPaths=("documentPatch.parties.shipper.address",),
        action="replace",
        sourceValue="C/O CHEMIDEX FZCO 23080-001 A2 BUILDING",
        targetValue="PLOT 18 AUTONAGAR INDUSTRIAL ESTATE",
        state="agent_residual",
        evidenceLineIds=("L00003",),
        spanIds=("S001",),
        locator="exact_literal",
        rationale="shipper address",
    )
    notify = HybridWorkItem(
        workItemId="W0002",
        targetPaths=("documentPatch.parties.notifyParties[0].name",),
        action="replace",
        sourceValue="CHEMIDEX FZCO",
        targetValue="BRIGHTHARBOR TRADE SERVICES LLC",
        state="agent_residual",
        evidenceLineIds=("L00005",),
        spanIds=("S002",),
        locator="literal_in_party_role_block",
        rationale="notify role",
    )

    inventory = build_mutable_inventory(
        source_text=text,
        current_text=text,
        source_label=source,
        target_label=target,
        work_items=(shipper_address, notify),
        oracle_case=None,
    )
    notify_rows = [
        row
        for row in inventory
        if row.sourceSurface == "CHEMIDEX FZCO" and "changed_source" in row.category
    ]

    assert [(row.category, row.lineIds) for row in notify_rows] == [
        ("changed_source_occurrence", ("L00005",)),
        ("changed_source_auxiliary_copy", ("L00006",)),
    ]
    requirements = _include_unique_linked_party_copies(
        text,
        (shipper_address, notify),
        inventory,
        (
            TargetValueOccurrenceRequirement(
                targetPaths=("documentPatch.parties.notifyParties[0].name",),
                targetValue="BRIGHTHARBOR TRADE SERVICES LLC",
                requiredOccurrences=1,
            ),
        ),
        (),
    )
    assert requirements[0].requiredOccurrences == 1


def test_inventory_does_not_rebind_party_name_that_is_owned_as_marks() -> None:
    source = {
        "documentPatch": {
            "parties": {"shipper": {"name": "TWIN DRAGON MARKETING, INC."}},
            "cargoGroups": [{"marksAndNumbers": ["TWIN DRAGON MARKETING INC."]}],
        }
    }
    target = {
        "documentPatch": {
            "parties": {"shipper": {"name": "Haiyuan Optoelectronic Commerce Co., Ltd."}},
            "cargoGroups": [{"marksAndNumbers": ["NOVA CIRCUIT"]}],
        }
    }
    text = "TWIN DRAGON MARKETING, INC.\nTWIN DRAGON MARKETING INC.\n"
    shipper = HybridWorkItem(
        workItemId="W0001",
        targetPaths=("documentPatch.parties.shipper.name",),
        action="replace",
        sourceValue="TWIN DRAGON MARKETING, INC.",
        targetValue="Haiyuan Optoelectronic Commerce Co., Ltd.",
        state="agent_residual",
        evidenceLineIds=("L00001",),
        spanIds=("S001",),
        locator="exact_literal",
        rationale="shipper name slot",
    )
    marks = HybridWorkItem(
        workItemId="W0002",
        targetPaths=("documentPatch.cargoGroups[0].marksAndNumbers[0]",),
        action="replace",
        sourceValue="TWIN DRAGON MARKETING INC.",
        targetValue="NOVA CIRCUIT",
        state="agent_residual",
        evidenceLineIds=("L00002",),
        spanIds=("S002",),
        locator="exact_literal",
        rationale="marks slot",
    )

    inventory = build_mutable_inventory(
        source_text=text,
        current_text=text,
        source_label=source,
        target_label=target,
        work_items=(shipper, marks),
        oracle_case=None,
    )

    assert not any(
        row.lineIds == ("L00002",) and "documentPatch.parties.shipper.name" in row.targetPaths
        for row in inventory
    )
    requirements = _include_unique_linked_party_copies(
        text,
        (shipper, marks),
        inventory,
        (
            TargetValueOccurrenceRequirement(
                targetPaths=("documentPatch.parties.shipper.name",),
                targetValue="Haiyuan Optoelectronic Commerce Co., Ltd.",
                requiredOccurrences=1,
            ),
        ),
        (),
    )
    assert requirements[0].requiredOccurrences == 1


def test_inventory_exact_marks_owner_beats_tolerant_party_match_without_line_work_item() -> None:
    source = {
        "documentPatch": {
            "parties": {"shipper": {"name": "TWIN DRAGON MARKETING, INC."}},
            "cargoGroups": [{"marksAndNumbers": ["TWIN DRAGON MARKETING INC."]}],
        }
    }
    target = {
        "documentPatch": {
            "parties": {"shipper": {"name": "Haiyuan Optoelectronic Commerce Co., Ltd."}},
            "cargoGroups": [{"marksAndNumbers": ["NOVA CIRCUIT"]}],
        }
    }
    text = "TWIN DRAGON MARKETING, INC.\nTWIN DRAGON MARKETING INC.\nGOODS DESCRIPTION\n"
    shipper = HybridWorkItem(
        workItemId="W0001",
        targetPaths=("documentPatch.parties.shipper.name",),
        action="replace",
        sourceValue="TWIN DRAGON MARKETING, INC.",
        targetValue="Haiyuan Optoelectronic Commerce Co., Ltd.",
        state="agent_residual",
        evidenceLineIds=("L00001",),
        spanIds=("S001",),
        locator="exact_literal",
        rationale="shipper name slot",
    )
    # Relation-scoped cargo evidence can be broader than the literal marks line.  Exact source
    # label ownership must still win without relying on this work item's evidence line.
    marks = HybridWorkItem(
        workItemId="W0002",
        targetPaths=("documentPatch.cargoGroups[0].marksAndNumbers[0]",),
        action="replace",
        sourceValue="TWIN DRAGON MARKETING INC.",
        targetValue="NOVA CIRCUIT",
        state="agent_residual",
        evidenceLineIds=("L00003",),
        spanIds=("S002",),
        locator="relation_scoped_surface",
        rationale="relation-scoped marks slot",
    )

    inventory = build_mutable_inventory(
        source_text=text,
        current_text=text,
        source_label=source,
        target_label=target,
        work_items=(shipper, marks),
        oracle_case=None,
    )

    conflicting = [
        row
        for row in inventory
        if row.lineIds == ("L00002",) and "documentPatch.parties.shipper.name" in row.targetPaths
    ]
    assert conflicting == []
    exact_marks = [
        row
        for row in inventory
        if row.lineIds == ("L00002",)
        and "documentPatch.cargoGroups[0].marksAndNumbers[0]" in row.targetPaths
    ]
    assert len(exact_marks) == 1


def test_inventory_does_not_count_legal_status_as_extra_party_copy() -> None:
    source = {"documentPatch": {"parties": {"consignee": {"name": "TO ORDER"}}}}
    target_name = "Kernfeld Energiekomponenten GmbH"
    target = {"documentPatch": {"parties": {"consignee": {"name": target_name}}}}
    heading = (
        "(3) Consignee (complete name and address)/(unless provided otherwise, "
        "a consignment 'To Order' means To Order of Shipper.)"
    )
    legal = "NOT NEGOTIABLE UNLESS CONSIGNED TO ORDER"
    text = f"{legal}\n{heading}\nTO ORDER\n"
    item = HybridWorkItem(
        workItemId="W0001",
        targetPaths=("documentPatch.parties.consignee.name",),
        action="replace",
        sourceValue="TO ORDER",
        targetValue=target_name,
        state="agent_residual",
        evidenceLineIds=("L00003",),
        spanIds=("S001",),
        locator="exact_literal",
        rationale="role-owned consignee value",
    )
    inventory = build_mutable_inventory(
        source_text=text,
        current_text=text,
        source_label=source,
        target_label=target,
        work_items=(item,),
        oracle_case=None,
    )
    requirements = _include_unique_linked_party_copies(
        text,
        (item,),
        inventory,
        (
            TargetValueOccurrenceRequirement(
                targetPaths=("documentPatch.parties.consignee.name",),
                targetValue=target_name,
                requiredOccurrences=1,
            ),
        ),
        (
            SourceStatusPreservationRequirement(sourceSurface=legal, sourceOccurrences=1),
            SourceStatusPreservationRequirement(sourceSurface=heading, sourceOccurrences=1),
        ),
    )

    assert requirements[0].requiredOccurrences == 1


def test_carrier_cardinality_includes_nested_forwarding_agent_target_occurrences() -> None:
    source = {
        "documentPatch": {
            "parties": {
                "carrier": {"name": "TRANSGLORY"},
                "forwardingAgent": {"name": "TRANSGLORY S.A."},
            }
        }
    }
    text = (
        "TRANSGLORY S.A.\n"
        "TRANSGLORY, S.A.\n"
        "TRANSGLORY, S.A.\n"
        "TRANSGLORY S.A.\n"
        "TRANSGLORY, S.A.\n"
        "TRANSGLORY, S.A.\n"
        "Signed on behalf of the Carrier: TRANSGLORY\n"
        "Signed on behalf of the Carrier: TRANSGLORY\n"
    )
    carrier_target = "Aureline Oceanic Carriers"
    forwarding_target = "Aureline Oceanic Carriers S.A."
    items = (
        HybridWorkItem(
            workItemId="W0001",
            targetPaths=("documentPatch.parties.carrier.name",),
            action="replace",
            sourceValue="TRANSGLORY",
            targetValue=carrier_target,
            state="agent_residual",
            evidenceLineIds=("L00007", "L00008"),
            spanIds=("S001",),
            locator="party_role_block",
            rationale="carrier principal",
        ),
        HybridWorkItem(
            workItemId="W0002",
            targetPaths=("documentPatch.parties.forwardingAgent.name",),
            action="replace",
            sourceValue="TRANSGLORY S.A.",
            targetValue=forwarding_target,
            state="agent_residual",
            evidenceLineIds=(
                "L00001",
                "L00002",
                "L00003",
                "L00004",
                "L00005",
                "L00006",
            ),
            spanIds=("S002",),
            locator="party_role_block",
            rationale="forwarding agent",
        ),
    )
    requirements = _refine_party_occurrence_requirements(
        text,
        text,
        items,
        (
            TargetValueOccurrenceRequirement(
                targetPaths=("documentPatch.parties.carrier.name",),
                targetValue=carrier_target,
                requiredOccurrences=1,
            ),
            TargetValueOccurrenceRequirement(
                targetPaths=("documentPatch.parties.forwardingAgent.name",),
                targetValue=forwarding_target,
                requiredOccurrences=1,
            ),
        ),
        source,
    )
    adjusted = _include_unique_linked_party_copies(
        text,
        items,
        (),
        requirements,
        (),
    )

    by_path = {row.targetPaths[0]: row.requiredOccurrences for row in adjusted}
    assert by_path["documentPatch.parties.carrier.name"] == 8
    assert by_path["documentPatch.parties.forwardingAgent.name"] == 6

    inventory = build_mutable_inventory(
        source_text=text,
        current_text=text,
        source_label=source,
        target_label={
            "documentPatch": {
                "parties": {
                    "carrier": {"name": carrier_target},
                    "forwardingAgent": {"name": forwarding_target},
                }
            }
        },
        work_items=items,
        oracle_case=None,
    )
    carrier_owned_lines = {
        line_id
        for row in inventory
        if "documentPatch.parties.carrier.name" in row.targetPaths
        for line_id in row.lineIds
    }
    forwarding_owned_lines = {
        line_id
        for row in inventory
        if "documentPatch.parties.forwardingAgent.name" in row.targetPaths
        for line_id in row.lineIds
    }
    assert carrier_owned_lines == {"L00007", "L00008"}
    assert forwarding_owned_lines == {
        "L00001",
        "L00002",
        "L00003",
        "L00004",
        "L00005",
        "L00006",
    }


def test_full_document_audit_rejects_punctuation_only_line() -> None:
    source_text = "DEGREES CELSIUS.\n"
    audit = audit_full_document(
        document_id="doc_" + "6" * 64,
        source_text=source_text,
        output_text=".\n",
        source_label={},
        target_label={},
        inventory=(),
        deterministic_edits=(),
        oracle_case=None,
    )

    assert audit.passed is False
    assert audit.findings[0].category == "lexical_line_degenerated"


def test_full_audit_requires_exact_party_value_in_its_own_role_block() -> None:
    source = {
        "documentPatch": {
            "parties": {
                "shipper": {"name": "SOURCE METALS S.A.C."},
                "carrier": {"name": "SOURCE CARRIER"},
            }
        }
    }
    target = {
        "documentPatch": {
            "parties": {
                "shipper": {"name": "ANDEAN METALS S.A.C."},
                "carrier": {"name": "TARGET CARRIER"},
            }
        }
    }
    source_text = (
        "SHIPPER\nSOURCE METALS S.A.C.\n\nSIGNED SOURCE CARRIER S.A.C.\n"
        "as agent for\nSOURCE CARRIER\n"
    )
    output_text = (
        "SHIPPER\nANDEAN METALS S.A.C\n\nSIGNED TARGET CARRIER S.A.C.\n"
        "as agent for\nTARGET CARRIER\n"
    )
    audit = audit_full_document(
        document_id="doc_" + "8" * 64,
        source_text=source_text,
        output_text=output_text,
        source_label=source,
        target_label=target,
        inventory=(),
        deterministic_edits=(),
        oracle_case=None,
    )

    assert audit.passed is False
    assert any(
        row.category == "semantic_slot_mismatch"
        and row.targetPaths == ("documentPatch.parties.shipper.name",)
        for row in audit.findings
    )


def test_full_audit_accepts_party_scalar_reflow_within_the_same_role_block() -> None:
    source = {
        "documentPatch": {
            "parties": {
                "consignee": {
                    "name": "SOURCE IMPORTS LTD.",
                    "address": "OLD STREET 7, ALEXANDRIA",
                    "city": "ALEXANDRIA",
                    "country": "EGYPT",
                }
            }
        }
    }
    target = {
        "documentPatch": {
            "parties": {
                "consignee": {
                    "name": "MERIDIAN IMPORTS LTD.",
                    "address": "NEW HARBOUR ROAD 18",
                    "city": "LIMASSOL",
                    "country": "CYPRUS",
                }
            }
        }
    }
    source_text = "CONSIGNEE\nSOURCE IMPORTS LTD.\nOLD STREET 7, ALEXANDRIA\nALEXANDRIA\nEGYPT\n"
    output_text = "CONSIGNEE\nMERIDIAN IMPORTS LTD.\nNEW HARBOUR\nROAD 18, LIMASSOL\nCYPRUS\n"
    items = tuple(
        HybridWorkItem(
            workItemId=f"W{index:04d}",
            targetPaths=(f"documentPatch.parties.consignee.{field}",),
            action="replace",
            sourceValue=source["documentPatch"]["parties"]["consignee"][field],
            targetValue=target["documentPatch"]["parties"]["consignee"][field],
            state="agent_residual",
            evidenceLineIds=(f"L{index + 1:05d}",),
            spanIds=(f"S{index:03d}",),
            locator="exact_literal",
            rationale="one original scalar line",
        )
        for index, field in enumerate(("name", "address", "city", "country"), start=1)
    )

    audit = audit_full_document(
        document_id="doc_" + "7" * 64,
        source_text=source_text,
        output_text=output_text,
        source_label=source,
        target_label=target,
        inventory=(),
        deterministic_edits=(),
        oracle_case=None,
        work_items=items,
    )

    assert audit.passed is True


def test_inventory_groups_case_equivalent_source_values_into_one_auxiliary_decision() -> None:
    source = {
        "documentPatch": {
            "parties": {"shipper": {"country": "TAIWAN"}},
            "cargoGroups": [{"groupId": "g1", "origin": {"name": "Taiwan"}}],
        }
    }
    target = {
        "documentPatch": {
            "parties": {"shipper": {"country": "Chile"}},
            "cargoGroups": [{"groupId": "g1", "origin": {"name": "Taiwan, Province of China"}}],
        }
    }
    text = "CARRIER OFFICE: MAERSK TAIWAN LTD - TAIPEI\n"

    inventory = build_mutable_inventory(
        source_text=text,
        current_text=text,
        source_label=source,
        target_label=target,
        work_items=(),
        oracle_case=None,
    )

    matching = [
        row
        for row in inventory
        if row.lineIds == ("L00001",)
        and row.category in {"changed_source_occurrence", "changed_source_auxiliary_copy"}
    ]
    assert len(matching) == 1
    assert matching[0].category == "changed_source_auxiliary_copy"
    assert set(matching[0].targetPaths) == {
        "documentPatch.parties.shipper.country",
        "documentPatch.cargoGroups[0].origin.name",
    }


def test_full_audit_requires_equipment_semantics_on_owned_line() -> None:
    item = HybridWorkItem(
        workItemId="W0001",
        targetPaths=("documentPatch.containers[0].printedEquipmentSurface",),
        action="replace_equipment_surface",
        sourceValue="40HR",
        targetValue={
            "sizeCategory": "FORTY_FOOT_HIGH_CUBE",
            "typeCategory": "GENERAL_PURPOSE",
        },
        state="agent_residual",
        evidenceLineIds=("L00001",),
        spanIds=("S001",),
        locator="exact_literal",
        rationale="test",
    )
    audit = audit_full_document(
        document_id="doc_" + "9" * 64,
        source_text="1X40HR CONTAINER(S) SAID TO\n",
        output_text="325 PACKAGES SAID TO\n",
        source_label={},
        target_label={},
        inventory=(),
        deterministic_edits=(),
        oracle_case=None,
        work_items=(item,),
    )

    assert audit.passed is False
    assert any(
        row.category == "semantic_slot_mismatch" and row.targetPaths == item.targetPaths
        for row in audit.findings
    )


def test_full_audit_accepts_compiler_owned_equipment_semantics() -> None:
    target_path = "documentPatch.containers[0].printedEquipmentSurface"
    item = HybridWorkItem(
        workItemId="W0001",
        targetPaths=(target_path,),
        action="replace_equipment_surface",
        sourceValue="1 X 40 HC",
        targetValue={
            "sizeCategory": "TWENTY_FOOT_STANDARD_HEIGHT",
            "typeCategory": "GENERAL_PURPOSE",
        },
        state="deterministic_applied",
        evidenceLineIds=(),
        spanIds=(),
        locator="deterministic_requirement",
        rationale="test compiler-owned equipment",
    )
    anchored = AnchoredScalarReplacementRequirement(
        targetPaths=(target_path,),
        sourceLineIds=("L00001",),
        sourceSurface="1 X 40 HC",
        targetSurface="1 X 20' STANDARD HEIGHT GENERAL PURPOSE",
    )
    audit = audit_full_document(
        document_id="doc_" + "8" * 64,
        source_text="1 X 40 HC\n",
        output_text="1 X 20' STANDARD HEIGHT GENERAL PURPOSE\n",
        source_label={},
        target_label={},
        inventory=(),
        deterministic_edits=(),
        oracle_case=None,
        work_items=(item,),
        anchored_replacements=(anchored,),
    )

    assert audit.passed is True


def test_inventory_does_not_treat_projected_v3_equipment_text_as_removed_fact() -> None:
    source_text = "--- PAGE 1 ---\n3 x 40' HIGH CUBE REEFER\n"
    source_label = {
        "documentPatch": {
            "containers": [
                {
                    "containerNumber": "MSCU1234566",
                    "typeDescription": "40' HIGH CUBE REEFER",
                }
            ]
        }
    }
    target_label = {
        "documentPatch": {
            "containers": [
                {
                    "containerNumber": "MSCU1234566",
                    "sizeCategory": "FORTY_FOOT_HIGH_CUBE",
                    "typeCategory": "REFRIGERATED",
                }
            ]
        }
    }
    work_item = HybridWorkItem(
        workItemId="W0001",
        targetPaths=("documentPatch.containers[0].printedEquipmentSurface",),
        action="preserve_equivalent_equipment_surface",
        sourceValue="40' HIGH CUBE REEFER",
        targetValue={
            "sizeCategory": "FORTY_FOOT_HIGH_CUBE",
            "typeCategory": "REFRIGERATED",
        },
        state="preserved_by_policy",
        evidenceLineIds=("L00002",),
        spanIds=("S001",),
        locator="exact_literal",
        rationale="v3 display text projects to v5 semantic categories",
    )

    inventory = build_mutable_inventory(
        source_text=source_text,
        current_text=source_text,
        source_label=source_label,
        target_label=target_label,
        work_items=(work_item,),
        oracle_case=None,
    )

    assert all(row.sourceSurface != "40' HIGH CUBE REEFER" for row in inventory)


def test_full_audit_accepts_host_derived_mixed_equipment_summary() -> None:
    source_text = "4X 40'CONTAINER SAID TO\nCONTAIN 1,886 CARTONS\n"
    target_surface = (
        "3X 40' HIGH CUBE GENERAL PURPOSE + "
        "1X 20' STANDARD HEIGHT GENERAL PURPOSE CONTAINERS SAID TO"
    )
    output_text = source_text.replace("4X 40'CONTAINER SAID TO", target_surface)
    items = tuple(
        HybridWorkItem(
            workItemId=f"W{index + 1:04d}",
            targetPaths=(f"documentPatch.containers[{index}].printedEquipmentSurface",),
            action="replace_equipment_surface",
            sourceValue="40'CONTAINER",
            targetValue={
                "sizeCategory": (
                    "FORTY_FOOT_HIGH_CUBE" if index < 3 else "TWENTY_FOOT_STANDARD_HEIGHT"
                ),
                "typeCategory": "GENERAL_PURPOSE",
            },
            state="deterministic_applied",
            evidenceLineIds=(),
            spanIds=(),
            locator="deterministic_requirement",
            rationale="host-derived aggregate equipment surface",
        )
        for index in range(4)
    )
    requirements = tuple(
        SurfaceRenderingRequirement(
            kind="aggregate_equipment_breakdown",
            targetPath=f"documentPatch.containers[{index}].printedEquipmentSurface",
            sourceSurface="4X 40'CONTAINER SAID TO",
            targetSurface=target_surface,
            sourceOccurrences=1,
            contextEvidence=source_text.strip(),
        )
        for index in range(4)
    )

    audit = audit_full_document(
        document_id="doc_" + "e" * 64,
        source_text=source_text,
        output_text=output_text,
        source_label={},
        target_label={},
        inventory=(),
        deterministic_edits=(),
        oracle_case=None,
        work_items=items,
        surface_requirements=requirements,
    )

    assert audit.passed is True


def test_full_audit_rejects_stale_carrier_header_even_if_target_occurs_elsewhere() -> None:
    requirement = SurfaceRenderingRequirement(
        kind="carrier_header_identity",
        targetPath="documentPatch.parties.carrier.name",
        sourceSurface="ARKAS",
        targetSurface="MAREVANTA",
        sourceOccurrences=1,
        contextEvidence="--- PAGE 1 ---\nARKAS\n\nSHIPPER",
    )
    audit = audit_full_document(
        document_id="doc_" + "a" * 64,
        source_text="--- PAGE 1 ---\nARKAS\n\nCARRIER\nMarevanta Ocean Lines, S.A.\n",
        output_text="--- PAGE 1 ---\nARKAS\n\nCARRIER\nMarevanta Ocean Lines, S.A.\n",
        source_label={},
        target_label={},
        inventory=(),
        deterministic_edits=(),
        oracle_case=None,
        surface_requirements=(requirement,),
    )

    assert audit.passed is False
    assert any(
        row.category == "semantic_slot_mismatch" and row.lineIds == ("L00002",)
        for row in audit.findings
    )


def test_full_audit_does_not_require_a_shared_date_format_in_the_wrong_role() -> None:
    requirement = SurfaceRenderingRequirement(
        kind="date",
        targetPath="documentPatch.issueDate",
        sourceSurface="07-MAY-2024",
        targetSurface="19-MAY-2025",
        sourceOccurrences=1,
        contextEvidence="Shipped on Board 07-MAY-2024",
    )
    audit = audit_full_document(
        document_id="doc_" + "f" * 64,
        source_text="Shipped on Board 07-MAY-2024\n",
        output_text="Shipped on Board 19-MAY-2025\n",
        source_label={},
        target_label={},
        inventory=(),
        deterministic_edits=(),
        oracle_case=None,
        surface_requirements=(requirement,),
    )

    assert audit.passed is True


def test_editor_payload_exposes_scalar_specific_party_line_topology() -> None:
    item = HybridWorkItem(
        workItemId="W0001",
        targetPaths=("documentPatch.parties.consignee.name",),
        action="replace",
        sourceValue="SOURCE COMPANY NAME",
        targetValue="Vinterhamn Provisions AB",
        state="agent_residual",
        evidenceLineIds=("L00001", "L00002", "L00003"),
        spanIds=("S001",),
        locator="party_role_block",
        rationale="wrapped consignee name",
    )
    slots = tuple(
        _Slot(
            alias=f"s{index}",
            line_id=f"L0000{index + 1}",
            source_line=value,
            requirements=(),
        )
        for index, value in enumerate(("SOURCE", "COMPANY", "NAME"))
    )
    compiled = SimpleNamespace(
        work_items=(item,),
        bundle=SimpleNamespace(result=SimpleNamespace(documentId="doc_" + "a" * 64)),
        workspace=RewriteWorkspace(
            original_text="SOURCE\nCOMPANY\nNAME\n",
            current_text="SOURCE\nCOMPANY\nNAME\n",
        ),
    )

    payload = _editor_payload(compiled, slots)

    assert payload["partyBlocks"] == [
        {
            "rolePath": "documentPatch.parties.consignee",
            "slots": ["s0", "s1", "s2"],
            "requiredTargetScalars": [
                {
                    "path": "documentPatch.parties.consignee.name",
                    "value": "Vinterhamn Provisions AB",
                    "sourceSlots": ["s0", "s1", "s2"],
                    "sourceSlotGroups": [["s0", "s1", "s2"]],
                }
            ],
            "instruction": (
                "Across these slots, render every required target scalar completely. Each "
                "scalar's sourceSlotGroups are independent printed copies. Render the complete "
                "target scalar once inside every group; reflow only within that group. "
                "sourceSlots is their union. No word, number, postal code, or terminal "
                "punctuation from a target scalar may be omitted."
            ),
        }
    ]


def test_editor_payload_keeps_repeated_party_scalar_copies_independent() -> None:
    item = HybridWorkItem(
        workItemId="W0001",
        targetPaths=("documentPatch.parties.consignee.address",),
        action="replace",
        sourceValue="9 OLD HARBOUR ROAD",
        targetValue="2-18-7 Shinmachi, Coastal Trade Building 4F",
        state="agent_residual",
        evidenceLineIds=("L00002", "L00006", "L00010"),
        spanIds=("S001", "S002", "S003"),
        locator="exact_literal",
        rationale="same consignee block printed on three pages",
    )
    source = (
        "CONSIGNEE\n9 OLD HARBOUR ROAD\n\n--- PAGE 2 ---\n"
        "CONSIGNEE\n9 OLD HARBOUR ROAD\n\n--- PAGE 3 ---\n"
        "CONSIGNEE\n9 OLD HARBOUR ROAD\n"
    )
    slots = tuple(
        _Slot(
            alias=f"s{index}",
            line_id=f"L{line_number:05d}",
            source_line="9 OLD HARBOUR ROAD",
            requirements=(),
        )
        for index, line_number in enumerate((2, 6, 10))
    )
    compiled = SimpleNamespace(
        work_items=(item,),
        bundle=SimpleNamespace(result=SimpleNamespace(documentId="doc_" + "b" * 64)),
        workspace=RewriteWorkspace(original_text=source, current_text=source),
    )

    payload = _editor_payload(compiled, slots)

    scalar = payload["partyBlocks"][0]["requiredTargetScalars"][0]
    assert scalar["sourceSlots"] == ["s0", "s1", "s2"]
    assert scalar["sourceSlotGroups"] == [["s0"], ["s1"], ["s2"]]


def test_editor_payload_does_not_treat_city_inside_company_name_as_city_slot() -> None:
    source = (
        "SHIPPER\n"
        "SHANGHAI BASF POLYURETHANE COMPANY\n"
        "LIMITED\n"
        "NO.25 CHU HUA ROAD, SHANGHAI\n"
        "CHEMICAL INDUSTRY PARK, SHANGHAI,\n"
        "201507, P.R. CHINA\n"
        "TAX ID: 45367422331225946H\n\n"
    )
    source_party = {
        "name": "SHANGHAI BASF POLYURETHANE COMPANY LIMITED",
        "address": "NO.25 CHU HUA ROAD, SHANGHAI CHEMICAL INDUSTRY PARK, 201507",
        "city": "SHANGHAI",
        "country": "P.R. CHINA",
    }
    target_party = {
        "name": "NORTHSTAR INDUSTRIAL SYSTEMS PRIVATE LIMITED",
        "address": "Plot 18, Sector 7, Firozpur Road Industrial Estate",
        "city": "Zira",
        "country": "India",
    }
    broad_lines = tuple(f"L{number:05d}" for number in range(2, 8))
    items = tuple(
        HybridWorkItem(
            workItemId=f"W{ordinal:04d}",
            targetPaths=(f"documentPatch.parties.shipper.{field}",),
            action="replace",
            sourceValue=source_party[field],
            targetValue=target_party[field],
            state="agent_residual",
            evidenceLineIds=broad_lines,
            spanIds=("S001",),
            locator="party_role_block",
            rationale="wrapped shipper block",
        )
        for ordinal, field in enumerate(("name", "address", "city", "country"), start=1)
    )
    slots = tuple(
        _Slot(
            alias=f"s{ordinal}",
            line_id=f"L{line_number:05d}",
            source_line=source.splitlines()[line_number - 1],
            requirements=(),
        )
        for ordinal, line_number in enumerate(range(2, 8))
    )
    compiled = SimpleNamespace(
        work_items=items,
        bundle=SimpleNamespace(result=SimpleNamespace(documentId="doc_" + "c" * 64)),
        workspace=RewriteWorkspace(
            original_text=source,
            current_text=source,
            source_label={"documentPatch": {"parties": {"shipper": source_party}}},
        ),
    )

    payload = _editor_payload(compiled, slots)

    scalars = {
        row["path"].rsplit(".", 1)[-1]: row
        for row in payload["partyBlocks"][0]["requiredTargetScalars"]
    }
    assert scalars["name"]["sourceSlotGroups"] == [["s0", "s1"]]
    assert scalars["address"]["sourceSlotGroups"] == [["s2", "s3", "s4"]]
    assert scalars["city"]["sourceSlotGroups"] == [["s3"]]
    assert scalars["country"]["sourceSlotGroups"] == [["s4"]]
