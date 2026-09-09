from __future__ import annotations

import json
from pathlib import Path

import pytest
from pydantic_ai.exceptions import ModelHTTPError

from document_ocr.hashing import canonical_json_bytes, sha256_bytes
from document_ocr.synthesis.config import load_synthesis_raw_text_certified_correction_config
from document_ocr.synthesis.raw_text_certification import SemanticAuditFinding
from document_ocr.synthesis.raw_text_certified_correction import (
    CorrectionStage,
    _apply_corrections,
    _attach_host_audit,
    _correction_lines,
    _correction_result,
    _deterministic_host_findings,
    _empty_usage,
    _findings_without_changed_evidence,
    _host_occurrence_findings,
    _load_case_checkpoint,
    _output_schema,
    _party_address_occurrence_line_sets,
    _payload,
    _publish_case_checkpoint,
    _refine_carrier_occurrence_contract,
    _select_repair_rows,
    _transient_route_error,
)
from document_ocr.synthesis.run_safety import StagedArtifactRun

_PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _finding(*line_ids: str) -> SemanticAuditFinding:
    fragments = {"L00002": "OLD 10", "L00004": "NEW 20"}
    return SemanticAuditFinding.model_validate(
        {
            "findingKind": "repeated_or_derived_fact_mismatch",
            "evidence": tuple(
                {"lineId": line_id, "currentFragment": fragments[line_id]} for line_id in line_ids
            ),
            "problem": "The old repeated total contradicts the target value.",
        }
    )


def _contract() -> dict[str, object]:
    return {
        "targetLiteralRequirements": [],
        "targetValueOccurrenceRequirements": [],
        "targetIntegrity": {
            "final_receipt": {"valid": True},
            "topology_matched": True,
        },
        "rawAuxiliaryIdentityRequirements": [],
    }


def test_parameter_filter_404_advances_to_next_correction_route() -> None:
    error = ModelHTTPError(
        404,
        "z-ai/glm-5.3-flash",
        {
            "message": "No endpoints found that can handle the requested parameters.",
            "metadata": {"failed_routing_step": "Filter by Parameters"},
        },
    )

    assert _transient_route_error(error)


def test_correction_changes_only_defect_line_and_echoes_comparison() -> None:
    source = "HEAD\nOLD 10\nKEEP\nNEW 20\n"
    current = source
    finding = _finding("L00002", "L00004")
    rows = _correction_lines(source=source, current=current, findings=(finding,))

    candidate, audit = _apply_corrections(
        source=source,
        current=current,
        rows=rows,
        findings=(finding,),
        output={"s0": "NEW 20", "s1": "NEW 20"},
        contract=_contract(),
    )

    assert candidate == "HEAD\nNEW 20\nKEEP\nNEW 20\n"
    assert audit.passed
    assert audit.citedLines == 2
    assert audit.changedLines == 1
    assert audit.resolvedFindings == 1


def test_correction_checkpoint_replays_exact_output_and_rejects_tampering(
    tmp_path: Path,
) -> None:
    document_id = "doc_" + "c" * 64
    source = "HEAD\nOLD 10\nKEEP\nNEW 20\n"
    current = source
    source_label = {"documentPatch": {"cargoGroups": [{"packages": 10}]}}
    target_label = {"documentPatch": {"cargoGroups": [{"packages": 20}]}}
    contract = _contract()
    finding = _finding("L00002", "L00004")
    findings = (finding,)
    rows = _correction_lines(source=source, current=current, findings=findings)
    output = {"s0": "NEW 20", "s1": "NEW 20"}
    final, audit = _apply_corrections(
        source=source,
        current=current,
        rows=rows,
        findings=findings,
        output=output,
        contract=contract,
    )
    _output_type, schema = _output_schema(rows)
    payload = _payload(
        document_id=document_id,
        source_label=source_label,
        target_label=target_label,
        rows=rows,
    )
    stage = _attach_host_audit(
        (
            CorrectionStage(
                routeProvider=None,
                semanticAttempt=1,
                routeRound=1,
                routeAttempt=1,
                retryDelayBeforeSeconds=0.0,
                inputPayloadSha256=sha256_bytes(canonical_json_bytes(payload)),
                outputSchemaSha256=sha256_bytes(canonical_json_bytes(schema)),
                startedAtUnixSeconds=1.0,
                completedAtUnixSeconds=2.0,
                usage=_empty_usage(),
                messages=[],
                modelOutput=output,
                hostAudit=None,
                errorType=None,
                errorMessage=None,
            ),
        ),
        audit,
    )
    result = _correction_result(
        document_id=document_id,
        status="correction_candidate",
        reason=(
            "Every exact-line local gate passed; independent recertification is required."
        ),
        source=source,
        current=current,
        final=final,
        rows=rows,
        findings=findings,
        stages=stage,
    )
    config = load_synthesis_raw_text_certified_correction_config(
        _PROJECT_ROOT
        / "configs/synthesis/"
        "mpci_bl_raw_text_certified_correction2_glm53_v22_wrapped_carrier_topology.yaml"
    )
    staged = StagedArtifactRun(
        output_parent=tmp_path,
        run_name="correction-checkpoint-test",
        transaction_sha256="d" * 64,
    )
    _publish_case_checkpoint(
        staged=staged,
        document_id=document_id,
        source=source,
        current=current,
        source_label=source_label,
        target_label=target_label,
        contract=contract,
        findings=findings,
        rows=rows,
        already_certified=False,
        stages=stage,
        final=final,
        result=result,
    )

    loaded = _load_case_checkpoint(
        staged=staged,
        document_id=document_id,
        source=source,
        current=current,
        source_label=source_label,
        target_label=target_label,
        contract=contract,
        findings=findings,
        rows=rows,
        already_certified=False,
        config=config,
    )

    assert loaded == (result, stage, final)
    checkpoint_path = staged.stage_root / f"cases/{document_id}/checkpoint.json"
    value = json.loads(checkpoint_path.read_text())
    value["finalText"] = final.replace("NEW 20", "TAMPERED", 1)
    checkpoint_path.write_bytes(canonical_json_bytes(value) + b"\n")
    with pytest.raises(ValueError, match="deterministic replay"):
        _load_case_checkpoint(
            staged=staged,
            document_id=document_id,
            source=source,
            current=current,
            source_label=source_label,
            target_label=target_label,
            contract=contract,
            findings=findings,
            rows=rows,
            already_certified=False,
            config=config,
        )


def test_correction_rejects_finding_when_all_evidence_is_unchanged() -> None:
    text = "HEAD\nOLD 10\n"
    finding = SemanticAuditFinding.model_validate(
        {
            "findingKind": "target_fact_mismatch",
            "evidence": ({"lineId": "L00002", "currentFragment": "OLD 10"},),
            "problem": "Target requires a different total.",
        }
    )
    rows = _correction_lines(source=text, current=text, findings=(finding,))

    _candidate, audit = _apply_corrections(
        source=text,
        current=text,
        rows=rows,
        findings=(finding,),
        output={"s0": "OLD 10"},
        contract=_contract(),
    )

    assert not audit.passed
    assert audit.unresolvedFindings == 1
    assert audit.findings == ("correction left 1 semantic finding(s) wholly unchanged",)


def test_correction_expands_exact_repeated_private_identifier_occurrences() -> None:
    text = (
        "CARRIER\n"
        "CO. REG. NO 196700080N\n"
        "TERMS\n"
        "CO. REG. NO 196700080N\n"
    )
    finding = SemanticAuditFinding.model_validate(
        {
            "findingKind": "party_or_legal_identity_mismatch",
            "evidence": (
                {"lineId": "L00002", "currentFragment": "CO. REG. NO 196700080N"},
            ),
            "problem": "The source carrier registration survives.",
        }
    )

    rows = _correction_lines(source=text, current=text, findings=(finding,))

    assert tuple(row.lineId for row in rows) == ("L00002", "L00004")


def test_correction_does_not_expand_ambiguous_numeric_target_evidence() -> None:
    text = "COUNT 10\nSECOND COUNT 10\n"
    finding = SemanticAuditFinding.model_validate(
        {
            "findingKind": "target_fact_mismatch",
            "evidence": ({"lineId": "L00001", "currentFragment": "COUNT 10"},),
            "problem": "Only the first count conflicts with its target row.",
        }
    )

    rows = _correction_lines(source=text, current=text, findings=(finding,))

    assert tuple(row.lineId for row in rows) == ("L00001",)


def test_correction_native_schema_rejects_whitespace_only_slots() -> None:
    text = "HEAD\nOLD 10\n"
    finding = SemanticAuditFinding.model_validate(
        {
            "findingKind": "target_fact_mismatch",
            "evidence": ({"lineId": "L00002", "currentFragment": "OLD 10"},),
            "problem": "Target requires a different total.",
        }
    )
    rows = _correction_lines(source=text, current=text, findings=(finding,))

    _output_type, schema = _output_schema(rows)

    assert schema["properties"]["s0"]["pattern"] == r".*\S.*"


def test_deterministic_host_finding_localizes_missing_target_to_source_role_line() -> None:
    source = "SHIPPER\nOLD SHIPPER LTD\n"
    current = "SHIPPER\nFictional auxiliary agent\n"
    contract = {
        **_contract(),
        "targetLiteralRequirements": [
            {
                "matchPolicy": "semantic_literal",
                "targetPath": "documentPatch.parties.shipper.name",
                "targetValue": "NEW SHIPPER LTD",
            }
        ],
        "changedLeaves": [
            {
                "path": "documentPatch.parties.shipper.name",
                "sourceValue": "OLD SHIPPER LTD",
            }
        ],
    }

    findings = _deterministic_host_findings(
        source=source,
        current=current,
        contract=contract,
    )

    assert len(findings) == 1
    assert tuple(row.lineId for row in findings[0].evidence) == ("L00002",)


def test_deterministic_host_finding_accepts_ordered_reference_literal() -> None:
    source = "EXPORT REFERENCE\n1123200938 DATE: 27.11.2023\n"
    current = "EXPORT REFERENCE\n785897 DATE: 07.07.2023\n"
    contract = {
        **_contract(),
        "targetLiteralRequirements": [
            {
                "matchPolicy": "ordered_semantic_atoms",
                "targetPath": "documentPatch.forwardingAndExportReferences[0]",
                "targetValue": "785897 07.07.2023",
            }
        ],
        "changedLeaves": [
            {
                "path": "documentPatch.forwardingAndExportReferences[0]",
                "sourceValue": "1123200938 27.11.2023",
            }
        ],
    }

    assert _deterministic_host_findings(source=source, current=current, contract=contract) == ()


def test_deterministic_host_finding_localizes_punctuation_variant_party_copies() -> None:
    source = (
        "CARRIER\n"
        "NO. 33 ALTUNIZADE / ISTANBUL / TURKEY\n"
        "SIGNED AT NO.33 ALTUNIZADE / ISTANBUL / TURKEY\n"
    )
    current = (
        "CARRIER\n"
        "FICTIONAL STREET / ZUERICH / SWITZERLAND\n"
        "SIGNED AT Hafenstrasse 47, 8005 / Zuerich / Switzerland\n"
    )
    contract = {
        **_contract(),
        "changedLeaves": [
            {
                "path": "documentPatch.parties.carrier.address",
                "sourceValue": "NO.33 ALTUNIZADE",
            }
        ],
        "targetValueOccurrenceRequirements": [
            {
                "targetPaths": ["documentPatch.parties.carrier.address"],
                "targetValue": "Hafenstrasse 47, 8005",
                "requiredOccurrences": 2,
            }
        ],
    }

    findings = _deterministic_host_findings(source=source, current=current, contract=contract)

    assert len(findings) == 1
    assert tuple(row.lineId for row in findings[0].evidence) == ("L00002", "L00003")


def test_party_address_locator_owns_omitted_street_line_but_not_party_aliases() -> None:
    source = (
        "TURKON LINE\n"
        "TURKON KONTEYNER TASIMACILIK\n"
        "TURKON CONTAINER TRANSPORTATION&SHIPPING INC.\n"
        "ORD. PROF. DR. FAHRETTIN KERIM GOKAY CAD.\n"
        "NO. 33 ALTUNIZADE / ISTANBUL / TURKEY\n"
        "\n"
        "Signed for the carrier (Turkon Container Transportation and Shipping Inc.) "
        "No.33 ALTUNIZADE / ISTANBUL / TURKEY\n"
    )
    label = {
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

    groups = _party_address_occurrence_line_sets(
        source,
        path="documentPatch.parties.carrier.address",
        source_label=label,
    )

    assert groups == (frozenset({4, 5}), frozenset({7}))


def test_cumulative_finding_resolution_survives_targeted_retry() -> None:
    findings = (
        SemanticAuditFinding.model_validate(
            {
                "findingKind": "target_fact_mismatch",
                "evidence": ({"lineId": "L00001", "currentFragment": "OLD A"},),
                "problem": "First fact is stale.",
            }
        ),
        SemanticAuditFinding.model_validate(
            {
                "findingKind": "target_fact_mismatch",
                "evidence": ({"lineId": "L00002", "currentFragment": "OLD B"},),
                "problem": "Second fact is stale.",
            }
        ),
    )

    unresolved = _findings_without_changed_evidence(
        baseline="OLD A\nOLD B\n",
        candidate="NEW A\nOLD B\n",
        findings=findings,
    )

    assert unresolved == (findings[1],)


def test_repair_selection_is_bounded_to_host_rejected_slot() -> None:
    source = "HEAD\nOLD 10\nKEEP\n"
    finding = SemanticAuditFinding.model_validate(
        {
            "findingKind": "target_fact_mismatch",
            "evidence": (
                {"lineId": "L00002", "currentFragment": "OLD 10"},
                {"lineId": "L00003", "currentFragment": "KEEP"},
            ),
            "problem": "Only one cited line is defective.",
        }
    )
    rows = _correction_lines(source=source, current=source, findings=(finding,))

    selected, repair_findings = _select_repair_rows(
        rows=rows,
        source=source,
        current=source,
        contract={**_contract(), "changedLeaves": []},
        host_error="ValueError: correction slot is not a non-empty string: s0",
    )

    assert tuple(row.slot for row in selected) == ("s0",)
    assert repair_findings == ()


def test_carrier_contract_counts_full_name_and_abbreviated_legal_principals() -> None:
    source = (
        "CARRIER: CMA CGM Société Anonyme\n"
        "SIGNED FOR THE CARRIER CMA CGM S.A.\n"
        "BY FICTIONAL AGENT\n"
        "as agents for the carrier CMA CGM S. A.\n"
    )
    contract = {
        **_contract(),
        "changedLeaves": [
            {
                "path": "documentPatch.parties.carrier.name",
                "sourceValue": "CMA CGM Société Anonyme",
            }
        ],
        "targetValueOccurrenceRequirements": [
            {
                "targetPaths": ["documentPatch.parties.carrier.name"],
                "targetValue": "Asterline Maritime Carriers Pte. Ltd.",
                "requiredOccurrences": 1,
            }
        ],
    }

    refined = _refine_carrier_occurrence_contract(source=source, contract=contract)

    assert refined["targetValueOccurrenceRequirements"][0]["requiredOccurrences"] == 3


def test_carrier_contract_does_not_double_count_wrapped_legal_principal() -> None:
    source = (
        "SIGNED FOR THE CARRIER Ocean Network Express Pte. Ltd.\n"
        "(ONE), AS CARRIER\n"
        "SIGNED FOR THE CARRIER Ocean Network Express Pte. Ltd.\n"
        "(ONE), AS CARRIER\n"
    )
    contract = {
        **_contract(),
        "changedLeaves": [
            {
                "path": "documentPatch.parties.carrier.name",
                "sourceValue": "Ocean Network Express Pte. Ltd. (ONE)",
            }
        ],
        "targetValueOccurrenceRequirements": [
            {
                "targetPaths": ["documentPatch.parties.carrier.name"],
                "targetValue": "Meridian Tidal Carriers Ltd.",
                "requiredOccurrences": 1,
            }
        ],
    }

    refined = _refine_carrier_occurrence_contract(source=source, contract=contract)

    assert refined["targetValueOccurrenceRequirements"][0]["requiredOccurrences"] == 2


def test_carrier_contract_replays_repeated_address_topology() -> None:
    source = (
        "CARRIER\n"
        "NO.33 ALTUNIZADE / ISTANBUL / TURKEY\n"
        "SIGNED FOR CARRIER AT NO.33 ALTUNIZADE / ISTANBUL / TURKEY\n"
    )
    contract = {
        **_contract(),
        "changedLeaves": [
            {
                "path": "documentPatch.parties.carrier.address",
                "sourceValue": "NO.33 ALTUNIZADE",
            }
        ],
        "targetValueOccurrenceRequirements": [
            {
                "targetPaths": ["documentPatch.parties.carrier.address"],
                "targetValue": "Hafenstrasse 47, 8005",
                "requiredOccurrences": 1,
            }
        ],
    }

    refined = _refine_carrier_occurrence_contract(source=source, contract=contract)

    assert refined["targetValueOccurrenceRequirements"][0]["requiredOccurrences"] == 2


def test_carrier_contract_ignores_uncontracted_split_address() -> None:
    source = (
        "Transpac Container System Pte. Ltd.\n"
        "d/b/a Blue Anchor Line\n"
        "5 Temasek Boulevard\n"
        "#06-01-03\n"
        "SuntecTower Five\n"
        "Singapore (038985)\n"
    )
    contract = {
        **_contract(),
        "changedLeaves": [
            {
                "path": "documentPatch.parties.carrier.name",
                "sourceValue": "Transpac Container System Pte. Ltd. d/b/a Blue Anchor Line",
            },
            {
                "path": "documentPatch.parties.carrier.address",
                "sourceValue": "5 Temasek Boulevard #06-01-03 SuntecTower Five (038985)",
            },
        ],
        "targetValueOccurrenceRequirements": [
            {
                "targetPaths": ["documentPatch.parties.carrier.name"],
                "targetValue": "Rivermark Transit AG",
                "requiredOccurrences": 1,
            }
        ],
    }

    refined = _refine_carrier_occurrence_contract(source=source, contract=contract)

    assert refined["targetValueOccurrenceRequirements"][0]["requiredOccurrences"] == 1


def test_carrier_contract_rejects_unlocatable_contracted_split_address() -> None:
    source = (
        "5 Temasek Boulevard\n"
        "#06-01-03\n"
        "SuntecTower Five\n"
        "Singapore (038985)\n"
    )
    contract = {
        **_contract(),
        "changedLeaves": [
            {
                "path": "documentPatch.parties.carrier.address",
                "sourceValue": "5 Temasek Boulevard #06-01-03 SuntecTower Five (038985)",
            }
        ],
        "targetValueOccurrenceRequirements": [
            {
                "targetPaths": ["documentPatch.parties.carrier.address"],
                "targetValue": "Industriestrasse 27",
                "requiredOccurrences": 1,
            }
        ],
    }

    with pytest.raises(ValueError, match="carrier address has no role-owned source template slot"):
        _refine_carrier_occurrence_contract(source=source, contract=contract)


def test_correction_rejects_line_edge_whitespace_drift() -> None:
    text = "HEAD\n  OLD 10  \n"
    finding = SemanticAuditFinding.model_validate(
        {
            "findingKind": "target_fact_mismatch",
            "evidence": ({"lineId": "L00002", "currentFragment": "OLD 10"},),
            "problem": "Target requires a different total.",
        }
    )
    rows = _correction_lines(source=text, current=text, findings=(finding,))

    with pytest.raises(ValueError, match="line-edge whitespace"):
        _apply_corrections(
            source=text,
            current=text,
            rows=rows,
            findings=(finding,),
            output={"s0": "NEW 20"},
            contract=_contract(),
        )


def test_correction_preserves_opaque_numeric_identifier_shape() -> None:
    text = "HEAD\n5848932722024070020\n"
    finding = SemanticAuditFinding.model_validate(
        {
            "findingKind": "source_only_private_or_auxiliary_fact",
            "evidence": ({"lineId": "L00002", "currentFragment": "5848932722024070020"},),
            "problem": "A source shipment identifier survives.",
        }
    )
    rows = _correction_lines(source=text, current=text, findings=(finding,))

    with pytest.raises(ValueError, match="numeric identifier shape"):
        _apply_corrections(
            source=text,
            current=text,
            rows=rows,
            findings=(finding,),
            output={"s0": "48273610559081234706"},
            contract=_contract(),
        )


def test_host_occurrence_findings_localize_surplus_inside_reflowed_role_block() -> None:
    source = (
        "CONSIGNEE\nOLD STREET\nOLD DISTRICT\nCITY\nNOTIFY PARTY\nOLD STREET\nOLD DISTRICT\nCITY\n"
    )
    current = (
        "CONSIGNEE\nNEW STREET\nFICTIONAL DISTRICT\nCITY\n"
        "NOTIFY PARTY\nNEW STREET\nNEW STREET\nCITY\n"
    )
    contract = {
        "targetValueOccurrenceRequirements": [
            {
                "targetPaths": [
                    "documentPatch.parties.consignee.address",
                    "documentPatch.parties.notifyParties[0].address",
                ],
                "targetValue": "NEW STREET",
                "requiredOccurrences": 2,
            }
        ],
        "changedLeaves": [
            {
                "path": "documentPatch.parties.consignee.address",
                "sourceValue": "OLD STREET OLD DISTRICT",
            },
            {
                "path": "documentPatch.parties.notifyParties[0].address",
                "sourceValue": "OLD STREET OLD DISTRICT",
            },
        ],
    }

    findings = _host_occurrence_findings(
        source=source,
        current=current,
        contract=contract,
    )

    assert len(findings) == 1
    assert tuple(row.lineId for row in findings[0].evidence) == ("L00007",)
    assert "remove or reflow that duplicate" in findings[0].problem
