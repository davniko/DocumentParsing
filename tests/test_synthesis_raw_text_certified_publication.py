from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import pytest

from document_ocr.hashing import canonical_json_bytes, sha256_bytes
from document_ocr.synthesis.config import SynthesisRawTextCertifiedPublicationConfig
from document_ocr.synthesis.raw_text_certification import (
    CertificationAuditReplay,
    CertificationCaseResult,
    _empty_usage,
    _host_audit,
)
from document_ocr.synthesis.raw_text_certification_artifacts import (
    ValidatedCertificationCase,
)
from document_ocr.synthesis.raw_text_certified_publication import _load_case


def _publication_config() -> dict[str, object]:
    return {
        "schema_version": 1,
        "task": "bill_of_lading_synthetic_raw_text_certified_publication_v1",
        "run": {"run_id": "certified-publication-test", "output_dir": "artifacts/out"},
        "certification_sources": [
            {
                "run": {
                    "path": "artifacts/certification-a",
                    "commit_sha256": "a" * 64,
                    "transaction_sha256": "b" * 64,
                },
                "certified_documents": 1,
                "certified_document_ids_sha256": "c" * 64,
            }
        ],
        "workflow": {
            "documents": 1,
            "required_audit_contract_version": 2,
            "require_every_case_certified": True,
            "require_unique_source_documents": True,
            "require_unique_scenarios": True,
            "replay_current_host_audit": True,
            "require_v5_canonical_target": True,
            "publish_training_records": True,
        },
    }


def _certified_case(tmp_path: Path) -> tuple[Path, ValidatedCertificationCase]:
    root = tmp_path / "certification"
    document_id = "doc_" + "1" * 64
    root.mkdir()
    source = b"--- PAGE 1 ---\nB/L OLD\n"
    final = b"--- PAGE 1 ---\nB/L NEW\n"
    source_label = {
        "schemaVersion": "3.0.0-experimental",
        "documentPatch": {"billOfLadingNumber": "OLD"},
    }
    target_label = {
        "schemaVersion": "5.0.0-experimental",
        "documentPatch": {"billOfLadingNumber": "NEW"},
    }
    scenario_id = "syn_" + "2" * 64
    contract = {
        "changedLeaves": [
            {
                "path": "documentPatch.billOfLadingNumber",
                "sourceValue": "OLD",
                "targetValue": "NEW",
            }
        ],
        "targetLiteralRequirements": [
            {
                "targetPath": "documentPatch.billOfLadingNumber",
                "targetValue": "NEW",
            }
        ],
        "targetValueOccurrenceRequirements": [],
        "targetIntegrity": {
            "topology_matched": True,
            "final_receipt": {"valid": True},
        },
        "rawAuxiliaryIdentityRequirements": [],
        "sourceLabel": source_label,
        "targetLabel": target_label,
        "result": {
            "documentId": document_id,
            "scenarioId": scenario_id,
            "sourceTextSha256": sha256_bytes(source),
            "sourceLabelSha256": sha256_bytes(canonical_json_bytes(source_label)),
            "targetLabelSha256": sha256_bytes(canonical_json_bytes(target_label)),
        },
    }
    audit = _host_audit(source=source.decode(), output=final.decode(), contract=contract)
    audit_plan = {"auditContractVersion": 2, "test": "already replayed"}
    audit_plan_sha256 = sha256_bytes(canonical_json_bytes(audit_plan))
    result = CertificationCaseResult(
        documentId=document_id,
        auditContractVersion=2,
        certificationMode="production",
        auditPlanSha256=audit_plan_sha256,
        status="certified",
        reason="Every required read-only semantic audit and deterministic host gate passed.",
        auditPasses=2,
        cleanAuditPasses=2,
        requiredAuditPasses=2,
        semanticFindings=0,
        candidateImmutable=True,
        hostAudit=audit,
        sourceTextSha256=sha256_bytes(source),
        inputCandidateSha256=sha256_bytes(final),
        finalTextSha256=sha256_bytes(final),
        usage=_empty_usage(),
    )
    replay = CertificationAuditReplay(
        outputs=(),
        findings=(),
        audit_passes=2,
        clean_audit_passes=2,
        audit_plan_sha256=audit_plan_sha256,
    )
    return root, ValidatedCertificationCase(
        document_id=document_id,
        source=source,
        candidate=final,
        source_label=source_label,
        target_label=target_label,
        contract=contract,
        audit_plan=audit_plan,
        result=result,
        stages=(),
        replay=replay,
    )


def test_publication_config_requires_exact_source_count() -> None:
    value = _publication_config()
    value["workflow"]["documents"] = 2  # type: ignore[index]

    with pytest.raises(ValueError, match="source counts"):
        SynthesisRawTextCertifiedPublicationConfig.model_validate(value, strict=True)


def test_load_case_replays_certification_and_current_host_contract(tmp_path: Path) -> None:
    root, certified = _certified_case(tmp_path)

    loaded = _load_case(root, certified, required_audit_contract_version=2)

    assert loaded.document_id == certified.document_id
    assert loaded.scenario_id == "syn_" + "2" * 64
    assert loaded.final_text == b"--- PAGE 1 ---\nB/L NEW\n"


def test_load_case_rejects_certification_that_changed_candidate_bytes(tmp_path: Path) -> None:
    root, certified = _certified_case(tmp_path)
    tampered = replace(certified, candidate=b"--- PAGE 1 ---\nB/L ALTERED\n")

    with pytest.raises(ValueError, match="complete certification contract"):
        _load_case(root, tampered, required_audit_contract_version=2)
