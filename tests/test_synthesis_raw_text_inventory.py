from __future__ import annotations

import json
from decimal import Decimal
from pathlib import Path
from types import SimpleNamespace

import pytest
from pydantic_ai.exceptions import ModelAPIError, ModelHTTPError

from document_ocr.hashing import canonical_json_bytes, sha256_bytes
from document_ocr.synthesis.config import (
    load_synthesis_raw_text_inventory_batch_config,
)
from document_ocr.synthesis.raw_text_hybrid_probe import (
    HybridWorkItem,
    _temperature_deactivation_line_numbers,
)
from document_ocr.synthesis.raw_text_inventory import (
    FullDocumentAudit,
    InventoryCandidate,
    RegressionCase,
    _carrier_receipt_count_matches_target,
    _is_shipment_aggregate_value_line,
    apply_deterministic_auxiliary_edits,
    audit_full_document,
    build_mutable_inventory,
    locate_auxiliary_values,
    parse_regression_oracle,
)
from document_ocr.synthesis.raw_text_inventory_probe import (
    InventoryProbeCaseResult,
    _CompoundSlot,
    _editor_payload,
    _empty_usage,
    _handling_affix_requirements_by_line,
    _is_explicit_package_quantity_line,
    _load_inventory_checkpoint,
    _locked_literal_requirements_by_line,
    _output_schema,
    _preview_repair_selection,
    _publish_inventory_checkpoint,
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
)
from document_ocr.synthesis.raw_text_rewrite_cycle_probe import (
    AnchoredScalarReplacementRequirement,
    AppliedDeterministicPrefill,
    AtomicRewriteCommit,
    CargoFlavorRewriteRequirement,
    CompoundPartyFlavorRequirement,
    RawAuxiliaryIdentityRequirement,
    RewriteWorkspace,
    SurfaceRenderingRequirement,
    TargetValueOccurrenceRequirement,
    _party_scalar_occurrence_count,
    source_semantic_role_hints,
)
from document_ocr.synthesis.run_safety import StagedArtifactRun


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


def test_deterministic_auxiliary_uses_observed_phone_punctuation_after_task_edit() -> None:
    source_text = (
        "TEL:+202-37609091\n"
        "Emergency Phone: 202-37609091\n"
    )
    current_text = (
        "TEL:+65 6128 4739\n"
        "Emergency Phone: 202-37609091\n"
    )
    source_label = {
        "documentPatch": {
            "parties": {
                "consignee": {
                    "contactDetails": {"phoneNumbers": ["+202-37609091"]}
                }
            }
        }
    }
    target_label = {
        "documentPatch": {
            "parties": {
                "consignee": {
                    "contactDetails": {"phoneNumbers": ["+65 6128 4739"]}
                }
            }
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
    deterministic_sources = {
        row.sourceSurface
        for row in inventory
        if row.disposition == "deterministic_shape_replacement"
    }
    output, edits = apply_deterministic_auxiliary_edits(
        text=current_text,
        document_id="doc_" + "a" * 64,
        scenario_id="syn-test",
        candidates=inventory,
    )

    assert "+202-37609091" not in deterministic_sources
    assert "202-37609091" in deterministic_sources
    assert "Emergency Phone: 202-37609091" not in output
    assert len(edits) == 1


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
    assert _carrier_receipt_count_matches_target(
        line, {"documentPatch": {"containers": target["documentPatch"]["containers"][:2]}}
    ) is False
    assert _carrier_receipt_count_matches_target(
        line, {"documentPatch": {"containers": None}}
    ) is False


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
    assert _transient_route_error(
        ModelAPIError("z-ai/glm-5.3-flash", "Connection error.")
    ) is True
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


def test_repair_selection_keeps_cargo_host_error_line_local() -> None:
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

    assert tuple(row.alias for row in selected) == ("s1",)
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
            "targetPrincipalName": "TARGET CARRIER",
            "slots": ["s1", "s2"],
            "instruction": (
                "Render one distinct fictional auxiliary identity and reuse that exact "
                "identity in every listed slot while preserving each line's surrounding legal "
                "relationship wording."
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
            "targetPrincipalName": "TARGET CARRIER",
            "slots": ["s0", "s1", "s2", "s3"],
            "instruction": (
                "Render one distinct fictional auxiliary identity and reuse that exact "
                "identity in every listed slot while preserving each line's surrounding legal "
                "relationship wording."
            ),
        }
    ]


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
        "documentPatch": {
            "parties": {"carrier": {"name": "Arkas Container Transport, S.A."}}
        }
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
    source_text = (
        "ARKAS CONTAINER TRANSPORT S.A.\n"
        "as agents of Arkas Container Transport, S.A.\n"
    )
    source = {
        "documentPatch": {
            "parties": {"carrier": {"name": "Arkas Container Transport, S.A."}}
        }
    }
    target = {
        "documentPatch": {
            "parties": {"carrier": {"name": "Bluehaven Maritime Lines, Ltd."}}
        }
    }
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

    refined = _refine_party_occurrence_requirements(text, (item,), (requirement,))

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

    refined = _refine_party_occurrence_requirements(text, (item,), (requirement,))

    assert refined[0].requiredOccurrences == 5


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

    refined = _refine_party_occurrence_requirements(
        current_text, (item,), (requirement,)
    )

    assert refined[0].requiredOccurrences == 2


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
            "parties": {
                "carrier": {"name": "SOURCE LINE", "address": "12 SOURCE QUAY"}
            }
        }
    }
    target = {
        "documentPatch": {
            "parties": {
                "carrier": {"name": "TARGET LINE", "address": "88 TARGET WHARF"}
            }
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
    with pytest.raises(ValueError, match="host-locked literals"):
        _validated_output_replacements((slot,), {"s0": "TEL. NO +62 31 7742 6189"})


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


def test_inventory_marks_unowned_party_scalar_copy_as_auxiliary_flavor() -> None:
    source = {
        "documentPatch": {
            "parties": {"carrier": {"address": "12 SOURCE QUAY"}}
        }
    }
    target = {
        "documentPatch": {
            "parties": {"carrier": {"address": "88 TARGET WHARF"}}
        }
    }
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
        ("changed_source_auxiliary_copy", ("L00002",)),
    ]


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
                    "FORTY_FOOT_HIGH_CUBE"
                    if index < 3
                    else "TWENTY_FOOT_STANDARD_HEIGHT"
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
