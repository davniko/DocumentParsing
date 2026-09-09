from __future__ import annotations

import json
from pathlib import Path

import pytest
from pydantic_ai.exceptions import ModelHTTPError

from document_ocr.hashing import canonical_json_bytes, sha256_bytes
from document_ocr.synthesis.raw_text_certification import (
    CertificationCaseResult,
    SemanticAuditEvidence,
    SemanticAuditFinding,
    SemanticAuditOutput,
    _contains_bounded_interleaved_description,
    _empty_usage,
    _host_audit,
    _load_case_checkpoint,
    _payload,
    _publish_case_checkpoint,
    _target_literal_present,
    _transient_route_error,
    _validate_audit_output,
)
from document_ocr.synthesis.run_safety import StagedArtifactRun


def _contract() -> dict[str, object]:
    return {
        "targetLiteralRequirements": [
            {"targetPath": "documentPatch.billOfLadingNumber", "targetValue": "NEW-123"}
        ],
        "targetValueOccurrenceRequirements": [
            {
                "targetPaths": ["documentPatch.parties.carrier.name"],
                "targetValue": "NEW CARRIER",
                "requiredOccurrences": 2,
            }
        ],
        "targetIntegrity": {"final_receipt": {"valid": True}, "topology_matched": True},
        "rawAuxiliaryIdentityRequirements": [],
    }


def _finding(fragment: str = "OLD-123", line_id: str = "L00002") -> SemanticAuditFinding:
    return SemanticAuditFinding(
        findingKind="target_fact_mismatch",
        evidence=(SemanticAuditEvidence(lineId=line_id, currentFragment=fragment),),
        problem="The source bill number remains in the candidate.",
    )


def test_parameter_filter_404_advances_to_next_certification_route() -> None:
    error = ModelHTTPError(
        404,
        "z-ai/glm-5.3-flash",
        {
            "message": "No endpoints found that can handle the requested parameters.",
            "metadata": {"failed_routing_step": "Filter by Parameters"},
        },
    )

    assert _transient_route_error(error)


def test_read_only_audit_accepts_only_exact_candidate_evidence() -> None:
    current = "--- PAGE 1 ---\nOLD-123\nNEW CARRIER\n"
    finding = _finding()
    assert _validate_audit_output(SemanticAuditOutput(findings=(finding,)), current=current) == (
        finding,
    )

    for invalid, message in (
        (_finding("MISSING"), "not exact candidate text"),
        (_finding("--- PAGE 1 ---", "L00001"), "blank/page-marker"),
        (_finding("OLD-123", "L00005"), "outside the document"),
    ):
        with pytest.raises(ValueError, match=message):
            _validate_audit_output(SemanticAuditOutput(findings=(invalid,)), current=current)


def test_read_only_audit_rejects_duplicate_findings() -> None:
    finding = _finding()
    with pytest.raises(ValueError, match="repeats an identical finding"):
        _validate_audit_output(
            SemanticAuditOutput(findings=(finding, finding)),
            current="--- PAGE 1 ---\nOLD-123\n",
        )


def test_payload_exposes_each_source_candidate_line_once() -> None:
    payload = _payload(
        document_id="doc_" + "a" * 64,
        source="--- PAGE 1 ---\nOLD\nKEEP\n",
        current="--- PAGE 1 ---\nNEW\nKEEP\n",
        source_label={"documentPatch": {"billOfLadingNumber": "OLD"}},
        target_label={"documentPatch": {"billOfLadingNumber": "NEW"}},
        host_findings=("known host failure",),
    )
    assert payload["lineLedger"] == [
        {"lineId": "L00001", "sourceLine": "--- PAGE 1 ---", "currentLine": None},
        {"lineId": "L00002", "sourceLine": "OLD", "currentLine": "NEW"},
        {"lineId": "L00003", "sourceLine": "KEEP", "currentLine": None},
    ]
    assert payload["deterministicHostFindings"] == ["known host failure"]


def test_host_audit_requires_target_counts_and_rejects_new_artifacts() -> None:
    source = "--- PAGE 1 ---\nOLD-123\nOLD CARRIER\nOLD CARRIER\nN/A\n"
    valid = "--- PAGE 1 ---\nNEW-123\nNEW CARRIER\nNEW CARRIER\nN/A\n"
    assert _host_audit(source=source, output=valid, contract=_contract()).passed is True
    for corrupted, expected in (
        (valid.replace("NEW-123", "OLD-123"), "missing target literal"),
        (valid.replace("NEW CARRIER\n", "OLD CARRIER\n", 1), "occurrence count"),
        (
            valid.replace("N/A", "NEW CARRIER"),
            "occurrence count",
        ),
        (valid.replace("N/A", "N/A N/A"), "new placeholder"),
        (valid.replace("N/A", "PACKAGE_PALLET"), "internal schema category"),
        (valid.replace("N/A", "I must return a patch"), "model-control prose"),
        (valid.replace("NEW-123", " NEW-123"), "line-edge whitespace"),
    ):
        audit = _host_audit(source=source, output=corrupted, contract=_contract())
        assert audit.passed is False
        assert any(expected in row for row in audit.findings)


def test_host_audit_rejects_retained_source_auxiliary_values_and_identities() -> None:
    source = (
        "--- PAGE 1 ---\nB/L NO: OLD-123\nOLD CARRIER\nOLD CARRIER\n"
        "BOOKING NO: BK-729104\nSIGNED BY OLD SOURCE AGENT LTD\n"
    )
    contract = _contract()
    contract["rawAuxiliaryIdentityRequirements"] = [{"sourceIdentity": "OLD SOURCE AGENT LTD"}]
    output = (
        "--- PAGE 1 ---\nB/L NO: NEW-123\nNEW CARRIER\nNEW CARRIER\n"
        "BOOKING NO: BK-729104\nSIGNED BY OLD SOURCE AGENT LTD\n"
    )
    audit = _host_audit(source=source, output=output, contract=contract)
    assert audit.passed is False
    assert any("source-only auxiliary value survived" in row for row in audit.findings)
    assert any("source-only auxiliary identity survived" in row for row in audit.findings)


def test_host_audit_does_not_confuse_short_auxiliary_token_with_identifier_prefix() -> None:
    source = (
        "--- PAGE 1 ---\nB/L NO: OLD-123\nOLD CARRIER\nOLD CARRIER\n"
        "website: old.example SCAC Code: MSCU\nMSCU7477141\n"
    )
    output = (
        "--- PAGE 1 ---\nB/L NO: NEW-123\nNEW CARRIER\nNEW CARRIER\n"
        "website: new.example SCAC Code: DRZF\nMSCU5500026\n"
    )

    assert _host_audit(source=source, output=output, contract=_contract()).passed is True


def test_host_audit_accepts_bounded_ordered_cargo_interleaving() -> None:
    description = "POLYISOBUTYLENE, POLYPROPYLENE AND PROPYLENE COPOLYMERS IN PRIMARY FORMS"
    output = (
        "--- PAGE 1 ---\nPOLYISOBUTYLENE,\nPOLYPROPYLENE AND PROPYLENE\n"
        "HS CODE: 39022070\nCOPOLYMERS IN PRIMARY\nFORMS\n"
    )
    contract = {
        "targetLiteralRequirements": [
            {"targetPath": "documentPatch.cargoGroups[0].description", "targetValue": description}
        ],
        "targetValueOccurrenceRequirements": [],
        "targetIntegrity": {"final_receipt": {"valid": True}, "topology_matched": True},
        "rawAuxiliaryIdentityRequirements": [],
    }
    assert _contains_bounded_interleaved_description(output, description) is True
    assert _host_audit(source=output, output=output, contract=contract).passed is True
    assert not _contains_bounded_interleaved_description(
        output.replace("COPOLYMERS IN PRIMARY", "PRIMARY COPOLYMERS IN"), description
    )


def test_host_audit_accepts_format_preserving_container_identifier_surface() -> None:
    source = "--- PAGE 1 ---\nCONTAINER: OLDU 123456.0\n"
    output = "--- PAGE 1 ---\nCONTAINER: MCLU 278760.1\n"
    contract = {
        "targetLiteralRequirements": [
            {
                "targetPath": "documentPatch.containers[0].containerNumber",
                "targetValue": "MCLU2787601",
                "matchPolicy": "alphanumeric_identifier",
            }
        ],
        "targetValueOccurrenceRequirements": [],
        "targetIntegrity": {"final_receipt": {"valid": True}, "topology_matched": True},
        "rawAuxiliaryIdentityRequirements": [],
    }

    assert _host_audit(source=source, output=output, contract=contract).passed is True
    contract["targetLiteralRequirements"][0]["matchPolicy"] = "semantic_literal"
    assert _host_audit(source=source, output=output, contract=contract).passed is False


def test_party_target_literal_accepts_source_preserved_glued_heading() -> None:
    text = "--- PAGE 1 ---\nNotify PartyCedar Gate Commercial Services\n"

    assert _target_literal_present(
        text=text,
        target_path="documentPatch.parties.notifyParties[0].name",
        target_value="Cedar Gate Commercial Services",
    )
    assert not _target_literal_present(
        text=text,
        target_path="documentPatch.cargoGroups[0].description",
        target_value="Cedar Gate Commercial Services",
    )


def test_dangerous_goods_un_number_accepts_standard_un_prefix() -> None:
    assert _target_literal_present(
        text="UN1161, DIMETHYL CARBONATE",
        target_path="documentPatch.cargoGroups[0].dangerousGoods[0].unNumber",
        target_value="1161",
    )
    assert not _target_literal_present(
        text="UN11610, OTHER PRODUCT",
        target_path="documentPatch.cargoGroups[0].dangerousGoods[0].unNumber",
        target_value="1161",
    )


@pytest.mark.parametrize(
    "text",
    (
        "INVOICE NO. 785897 DATED: 07.07.2023\n",
        "EXPORT REFERENCE: 785897\nDATE: 07.07.2023\n",
    ),
)
def test_ordered_reference_literal_accepts_bounded_contextual_atoms(text: str) -> None:
    assert _target_literal_present(
        text=text,
        target_path="documentPatch.forwardingAndExportReferences[0]",
        target_value="785897 07.07.2023",
        match_policy="ordered_semantic_atoms",
    )


@pytest.mark.parametrize(
    "text",
    (
        "B/L NO. 785897\nUNRELATED DATE: 07.07.2023\n",
        "INVOICE NO. 07.07.2023 DATED: 785897\n",
        "INVOICE NO. 785897 ONE TWO THREE FOUR FIVE SIX SEVEN 07.07.2023\n",
        "INVOICE NO. 785897\nUNRELATED\nDATE: 07.07.2023\n",
    ),
)
def test_ordered_reference_literal_rejects_unowned_or_unbounded_atoms(text: str) -> None:
    assert not _target_literal_present(
        text=text,
        target_path="documentPatch.forwardingAndExportReferences[0]",
        target_value="785897 07.07.2023",
        match_policy="ordered_semantic_atoms",
    )


def test_host_audit_accepts_ordered_reference_literal_contract() -> None:
    source = "--- PAGE 1 ---\nINVOICE NO. 1123200938 DATED: 27.11.2023\n"
    output = "--- PAGE 1 ---\nINVOICE NO. 785897 DATED: 07.07.2023\n"
    contract = {
        "targetLiteralRequirements": [
            {
                "targetPath": "documentPatch.forwardingAndExportReferences[0]",
                "targetValue": "785897 07.07.2023",
                "matchPolicy": "ordered_semantic_atoms",
            }
        ],
        "targetValueOccurrenceRequirements": [],
        "targetIntegrity": {"final_receipt": {"valid": True}, "topology_matched": True},
        "rawAuxiliaryIdentityRequirements": [],
    }

    assert _host_audit(source=source, output=output, contract=contract).passed is True


def test_read_only_checkpoint_replays_candidate_identity(tmp_path: Path) -> None:
    document_id = "doc_" + "b" * 64
    source = "--- PAGE 1 ---\nOLD-123\nOLD CARRIER\nOLD CARRIER\n"
    candidate = "--- PAGE 1 ---\nNEW-123\nNEW CARRIER\nNEW CARRIER\n"
    contract = _contract()
    audit = _host_audit(source=source, output=candidate, contract=contract)
    result = CertificationCaseResult(
        documentId=document_id,
        status="call_failed",
        reason="No provider response.",
        auditPasses=0,
        semanticFindings=0,
        candidateImmutable=True,
        hostAudit=audit,
        sourceTextSha256=sha256_bytes(source.encode()),
        inputCandidateSha256=sha256_bytes(candidate.encode()),
        finalTextSha256=sha256_bytes(candidate.encode()),
        usage=_empty_usage(),
    )
    staged = StagedArtifactRun(
        output_parent=tmp_path,
        run_name="read-only-certification-checkpoint-test",
        transaction_sha256="c" * 64,
    )
    _publish_case_checkpoint(
        staged=staged,
        document_id=document_id,
        source=source,
        candidate=candidate,
        contract=contract,
        stages=(),
        result=result,
    )
    loaded = _load_case_checkpoint(
        staged=staged,
        document_id=document_id,
        source=source,
        candidate=candidate,
        contract=contract,
    )
    assert loaded == (result, ())
    value = json.loads((staged.stage_root / f"cases/{document_id}/checkpoint.json").read_text())
    assert value["sourceContractSha256"] == sha256_bytes(canonical_json_bytes(contract))
