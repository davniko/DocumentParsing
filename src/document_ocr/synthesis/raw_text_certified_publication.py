"""Immutable publication of independently certified synthetic OCR/label pairs."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast

from pydantic import JsonValue

from document_ocr.atomic import read_regular_file_bytes
from document_ocr.hashing import canonical_json_bytes, sha256_bytes, sha256_file
from document_ocr.label_schemas.bill_of_lading_v5 import (
    BillOfLadingRelationExplicitV5Label,
)
from document_ocr.synthesis.config import (
    SynthesisRawTextCertifiedPublicationConfig,
    load_synthesis_raw_text_certified_publication_config,
)
from document_ocr.synthesis.raw_text_certification import CertificationCaseResult
from document_ocr.synthesis.raw_text_certification_artifacts import (
    ValidatedCertificationCase,
    ValidatedCertificationRun,
    load_validated_certification_run,
)
from document_ocr.synthesis.raw_text_certification_host import CertificationInvariantCase
from document_ocr.synthesis.raw_text_hybrid_probe import _artifact_inventory
from document_ocr.synthesis.raw_text_inventory_probe import _validate_reference_run
from document_ocr.synthesis.run_safety import StagedArtifactRun
from document_ocr.training.config import resolve_config_path

_IMPLEMENTATION_PATH = Path(__file__).resolve(strict=True)


@dataclass(frozen=True, slots=True)
class _CertifiedCase:
    source_run: Path
    document_id: str
    scenario_id: str
    source_text: bytes
    final_text: bytes
    source_label: dict[str, Any]
    target_label: dict[str, Any]
    contract: dict[str, Any]
    audit_plan: dict[str, Any]
    stages_sha256: str
    result: CertificationCaseResult
    invariant_case: CertificationInvariantCase | None


def _read_json_object(path: Path) -> dict[str, Any]:
    value = json.loads(read_regular_file_bytes(path))
    if not isinstance(value, dict):
        raise ValueError(f"artifact is not a JSON object: {path}")
    return cast(dict[str, Any], value)


def _load_case(
    root: Path,
    certified: ValidatedCertificationCase,
    *,
    required_audit_contract_version: int,
) -> _CertifiedCase:
    document_id = certified.document_id
    source = certified.source
    candidate = certified.candidate
    final = certified.candidate
    result = certified.result
    source_label = certified.source_label
    target_label = certified.target_label
    contract = certified.contract
    replay = certified.replay
    if (
        result.documentId != document_id
        or result.auditContractVersion != required_audit_contract_version
        or result.certificationMode != "production"
        or result.status != "certified"
        or result.auditPasses != 2
        or result.cleanAuditPasses != 2
        or result.requiredAuditPasses != 2
        or replay.audit_passes != 2
        or replay.findings
        or result.auditPlanSha256 != sha256_bytes(canonical_json_bytes(certified.audit_plan))
        or result.semanticFindings != 0
        or result.deterministicFindings != 0
        or not result.hostAudit.passed
        or result.hostAudit.findings
        or result.sourceTextSha256 != sha256_bytes(source)
        or result.inputCandidateSha256 != sha256_bytes(candidate)
        or result.finalTextSha256 != sha256_bytes(final)
    ):
        raise ValueError(f"case does not meet the complete certification contract: {document_id}")
    invariant = certified.invariant_case
    if required_audit_contract_version == 3:
        if (
            invariant is None
            or not invariant.audit.passed
            or invariant.audit.findings
            or result.invariantEnvelopeSha256
            != sha256_bytes(canonical_json_bytes(invariant.envelope.model_dump(mode="json")))
            or result.invariantAuditSha256
            != sha256_bytes(canonical_json_bytes(invariant.audit.model_dump(mode="json")))
        ):
            raise ValueError(
                f"case does not have replayed deterministic certification: {document_id}"
            )
    elif invariant is not None:
        raise ValueError(f"contract-v2 publication case carries v3 invariants: {document_id}")
    contract_result = contract.get("result")
    if not isinstance(contract_result, dict):
        raise ValueError(f"source contract lacks its scenario receipt: {document_id}")
    scenario_id = contract_result.get("scenarioId")
    if (
        not isinstance(scenario_id, str)
        or len(scenario_id) != 68
        or not scenario_id.startswith("syn_")
        or any(character not in "0123456789abcdef" for character in scenario_id[4:])
    ):
        raise ValueError(f"source contract has an invalid scenario ID: {document_id}")
    if (
        contract_result.get("documentId") != document_id
        or contract_result.get("sourceTextSha256") != sha256_bytes(source)
        or contract_result.get("sourceLabelSha256")
        != sha256_bytes(canonical_json_bytes(source_label))
        or contract_result.get("targetLabelSha256")
        != sha256_bytes(canonical_json_bytes(target_label))
    ):
        raise ValueError(f"source contract identity differs from case bytes: {document_id}")
    integrity = contract.get("targetIntegrity")
    if (
        not isinstance(integrity, dict)
        or integrity.get("topology_matched") is not True
        or not isinstance(integrity.get("final_receipt"), dict)
        or integrity["final_receipt"].get("valid") is not True
    ):
        raise ValueError(f"target integrity receipt is not valid: {document_id}")
    canonical_target = BillOfLadingRelationExplicitV5Label.model_validate_json(
        canonical_json_bytes(target_label), strict=True
    ).canonical_target()
    if canonical_target != target_label:
        raise ValueError(f"target is not canonical schema-v5 JSON: {document_id}")
    return _CertifiedCase(
        source_run=root,
        document_id=document_id,
        scenario_id=scenario_id,
        source_text=source,
        final_text=final,
        source_label=source_label,
        target_label=target_label,
        contract=contract,
        audit_plan=certified.audit_plan,
        stages_sha256=sha256_bytes(
            canonical_json_bytes([row.model_dump(mode="json") for row in certified.stages]) + b"\n"
        ),
        result=result,
        invariant_case=invariant,
    )


def _report(cases: tuple[_CertifiedCase, ...]) -> str:
    source_bytes = sum(len(row.source_text) for row in cases)
    output_bytes = sum(len(row.final_text) for row in cases)
    source_lines = sum(len(row.source_text.splitlines()) for row in cases)
    output_lines = sum(len(row.final_text.splitlines()) for row in cases)
    return "\n".join(
        (
            "# Certified synthetic raw-OCR publication",
            "",
            "## Outcome",
            "",
            f"- Independently certified document/label pairs: **{len(cases)} / {len(cases)}**.",
            "- Semantic findings at publication: **0**.",
            "- Deterministic invariant findings at publication: **0**.",
            "- Current deterministic host-audit failures: **0**.",
            "- Candidate bytes modified by certification/publication: **0**.",
            f"- Source / synthetic OCR bytes: **{source_bytes:,} / {output_bytes:,}**.",
            f"- Source / synthetic OCR lines: **{source_lines:,} / {output_lines:,}**.",
            "",
            "`dataset/records.jsonl` uses each immutable synthetic scenario ID as `documentId`, "
            "retains the original template document as both `sourceDocumentId` and "
            "`splitGroupId`, and contains the exact certified OCR plus canonical schema-v5 "
            "target. Descendants must be partitioned by `splitGroupId` when later combined with "
            "their real source templates.",
            "",
        )
    )


def run_raw_text_certified_publication(
    *,
    project_root: Path,
    config_path: Path,
    config: SynthesisRawTextCertifiedPublicationConfig,
) -> dict[str, JsonValue]:
    if config_path.is_symlink() or not config_path.is_file():
        raise ValueError("certified-publication configuration must be a regular file")
    loaded_config = load_synthesis_raw_text_certified_publication_config(config_path)
    if loaded_config != config:
        raise ValueError("certified-publication configuration object differs from config_path")
    if (
        "required_audit_contract_version" not in loaded_config.workflow.model_fields_set
        or config.workflow.required_audit_contract_version not in {2, 3}
    ):
        raise ValueError("new publication requires an explicit modern certification contract")
    required_contract = config.workflow.required_audit_contract_version
    source_roots: list[Path] = []
    selected_ids: dict[Path, tuple[str, ...]] = {}
    source_runs: dict[Path, ValidatedCertificationRun] = {}
    for source in config.certification_sources:
        root = _validate_reference_run(
            project_root,
            source.run.path,
            source.run.commit_sha256,
            source.run.transaction_sha256,
        )
        certification_run = load_validated_certification_run(
            root,
            project_root=project_root if required_contract == 3 else None,
        )
        certification_config = certification_run.config
        if (
            certification_config.audit_contract_version != required_contract
            or certification_config.workflow.evaluation_only
            or certification_config.workflow.semantic_audit_passes != 2
        ):
            raise ValueError(f"publication source is not the required production contract: {root}")
        ids = tuple(
            sorted(
                row.document_id
                for row in certification_run.cases
                if row.result.status == "certified"
            )
        )
        if len(ids) != source.certified_documents:
            raise ValueError(f"configured certified count differs for source: {root}")
        if sha256_bytes(canonical_json_bytes(list(ids))) != source.certified_document_ids_sha256:
            raise ValueError(f"configured certified document identity differs for source: {root}")
        source_roots.append(root)
        selected_ids[root] = ids
        source_runs[root] = certification_run

    cases = tuple(
        _load_case(
            root,
            source_runs[root].case_by_id()[document_id],
            required_audit_contract_version=required_contract,
        )
        for root in source_roots
        for document_id in selected_ids[root]
    )
    if len(cases) != config.workflow.documents:
        raise ValueError("loaded certified cases differ from configured publication scope")
    document_ids = tuple(row.document_id for row in cases)
    scenario_ids = tuple(row.scenario_id for row in cases)
    if len(document_ids) != len(set(document_ids)):
        raise ValueError("certified publication repeats a source document")
    if len(scenario_ids) != len(set(scenario_ids)):
        raise ValueError("certified publication repeats a synthetic scenario")

    transaction: dict[str, JsonValue] = {
        "schemaVersion": 3 if required_contract == 3 else 1,
        "runId": config.run.run_id,
        "configSha256": sha256_file(config_path),
        "resolvedConfigSha256": sha256_bytes(canonical_json_bytes(config.model_dump(mode="json"))),
        "requiredAuditContractVersion": config.workflow.required_audit_contract_version,
        "implementationSha256": sha256_file(_IMPLEMENTATION_PATH),
        "dependencyImplementationSha256": {
            "certification": sha256_file(Path(__file__).with_name("raw_text_certification.py")),
            "certificationArtifacts": sha256_file(
                Path(__file__).with_name("raw_text_certification_artifacts.py")
            ),
            "inventoryRunner": sha256_file(Path(__file__).with_name("raw_text_inventory_probe.py")),
            "targetSchema": sha256_file(
                Path(__file__).parents[1] / "label_schemas/bill_of_lading_v5.py"
            ),
            **(
                {
                    "certificationHost": sha256_file(
                        Path(__file__).with_name("raw_text_certification_host.py")
                    ),
                    "certificationInvariants": sha256_file(
                        Path(__file__).with_name("raw_text_certification_invariants.py")
                    ),
                    "certificationReferences": sha256_file(
                        Path(__file__).with_name("raw_text_certification_references.py")
                    ),
                }
                if required_contract == 3
                else {}
            ),
        },
        "sourceRuns": [
            {
                "path": source.run.path,
                "commitSha256": source.run.commit_sha256,
                "transactionSha256": source.run.transaction_sha256,
                "certifiedDocuments": source.certified_documents,
                "certifiedDocumentIdsSha256": source.certified_document_ids_sha256,
                "configSha256": sha256_file(source_runs[root].root / "config.yaml"),
                "resolvedConfigSha256": sha256_bytes(
                    canonical_json_bytes(source_runs[root].config.model_dump(mode="json"))
                ),
                "transactionArtifactSha256": sha256_file(
                    source_runs[root].root / "provenance/transaction.json"
                ),
                "summarySha256": sha256_file(source_runs[root].root / "summary.json"),
                **(
                    {
                        "invariantReferenceReceiptSha256": source_runs[root].transaction[
                            "invariantReferenceReceiptSha256"
                        ],
                        "capacityConfigSha256": source_runs[root].transaction[
                            "capacityConfigSha256"
                        ],
                    }
                    if required_contract == 3
                    else {}
                ),
            }
            for source, root in zip(config.certification_sources, source_roots, strict=True)
        ],
        "documentIdsSha256": sha256_bytes(canonical_json_bytes(sorted(document_ids))),
        "scenarioIdsSha256": sha256_bytes(canonical_json_bytes(sorted(scenario_ids))),
    }
    staged = StagedArtifactRun(
        output_parent=resolve_config_path(project_root, config.run.output_dir),
        run_name=config.run.run_id,
        transaction_sha256=sha256_bytes(canonical_json_bytes(transaction)),
    )
    if staged.completed:
        completed = cast(
            dict[str, JsonValue], _read_json_object(staged.final_root / "summary.json")
        )
        completed["artifactRoot"] = str(staged.final_root)
        completed["commitSha256"] = sha256_file(staged.final_root / "_COMMIT.json")
        return completed
    staged.recover_interrupted_temporary_files()
    staged.publish_bytes("config.yaml", read_regular_file_bytes(config_path))

    records: list[dict[str, Any]] = []
    lineage: list[dict[str, Any]] = []
    for row in sorted(cases, key=lambda item: item.scenario_id):
        source_relative = row.source_run.relative_to(project_root)
        record = {
            "documentId": row.scenario_id,
            "sourceDocumentId": row.document_id,
            "splitGroupId": row.document_id,
            "joinedRawText": row.final_text.decode("utf-8"),
            "joinedRawTextSha256": sha256_bytes(row.final_text),
            "target": row.target_label,
        }
        records.append(record)
        lineage.append(
            {
                "documentId": row.scenario_id,
                "sourceDocumentId": row.document_id,
                "splitGroupId": row.document_id,
                "certificationRun": str(source_relative),
                "sourceTextSha256": sha256_bytes(row.source_text),
                "joinedRawTextSha256": sha256_bytes(row.final_text),
                "sourceLabelSha256": sha256_bytes(canonical_json_bytes(row.source_label)),
                "targetSha256": sha256_bytes(canonical_json_bytes(row.target_label)),
                "certificationResultSha256": sha256_bytes(
                    canonical_json_bytes(row.result.model_dump(mode="json"))
                ),
                "certificationAuditPlanSha256": sha256_bytes(canonical_json_bytes(row.audit_plan)),
                "certificationStagesSha256": row.stages_sha256,
                "certificationInvariantEnvelopeSha256": (
                    sha256_bytes(
                        canonical_json_bytes(row.invariant_case.envelope.model_dump(mode="json"))
                    )
                    if row.invariant_case is not None
                    else None
                ),
                "certificationInvariantAuditSha256": (
                    sha256_bytes(
                        canonical_json_bytes(row.invariant_case.audit.model_dump(mode="json"))
                    )
                    if row.invariant_case is not None
                    else None
                ),
            }
        )
        prefix = f"cases/{row.scenario_id}"
        staged.publish_bytes(f"{prefix}/source.txt", row.source_text)
        staged.publish_bytes(f"{prefix}/final.txt", row.final_text)
        staged.publish_json(f"{prefix}/source-label.json", row.source_label)
        staged.publish_json(f"{prefix}/target-label.json", row.target_label)
        staged.publish_json(f"{prefix}/source-contract.json", row.contract)
        staged.publish_json(f"{prefix}/certification-audit-plan.json", row.audit_plan)
        staged.publish_json(
            f"{prefix}/certification-result.json", row.result.model_dump(mode="json")
        )
        staged.publish_json(f"{prefix}/lineage.json", lineage[-1])
        if row.invariant_case is not None:
            staged.publish_json(
                f"{prefix}/inventory.json",
                [value.model_dump(mode="json") for value in row.invariant_case.inventory],
            )
            staged.publish_json(
                f"{prefix}/deterministic-edits.json",
                [value.model_dump(mode="json") for value in row.invariant_case.deterministic_edits],
            )
            staged.publish_json(
                f"{prefix}/certification-invariant-envelope.json",
                row.invariant_case.envelope.model_dump(mode="json"),
            )
            staged.publish_json(
                f"{prefix}/certification-invariant-audit.json",
                row.invariant_case.audit.model_dump(mode="json"),
            )

    records_bytes = b"".join(canonical_json_bytes(row) + b"\n" for row in records)
    lineage_bytes = b"".join(canonical_json_bytes(row) + b"\n" for row in lineage)
    staged.publish_bytes("dataset/records.jsonl", records_bytes)
    staged.publish_bytes("dataset/lineage.jsonl", lineage_bytes)
    dataset_manifest = {
        "schemaVersion": 1,
        "records": len(records),
        "files": [
            {
                "kind": "training_records",
                "path": "records.jsonl",
                "records": len(records),
                "bytes": len(records_bytes),
                "sha256": sha256_bytes(records_bytes),
            },
            {
                "kind": "lineage",
                "path": "lineage.jsonl",
                "records": len(lineage),
                "bytes": len(lineage_bytes),
                "sha256": sha256_bytes(lineage_bytes),
            },
        ],
        "splitGroupField": "splitGroupId",
        "targetSchemaVersion": "5.0.0-experimental",
    }
    staged.publish_json("dataset/manifest.json", dataset_manifest)
    summary: dict[str, JsonValue] = {
        "schemaVersion": 3 if required_contract == 3 else 1,
        "runId": config.run.run_id,
        "status": "complete",
        "sourceCertificationRuns": len(source_roots),
        "documents": len(cases),
        "certifiedDocuments": len(cases),
        "semanticFindings": 0,
        "deterministicFindings": 0,
        "currentHostAuditFailures": 0,
        "candidateBytesModified": 0,
        "trainingRecordsPublished": True,
        "recordsSha256": sha256_bytes(records_bytes),
        "lineageSha256": sha256_bytes(lineage_bytes),
        "documentIdsSha256": transaction["documentIdsSha256"],
        "scenarioIdsSha256": transaction["scenarioIdsSha256"],
    }
    staged.publish_json("summary.json", summary)
    staged.publish_bytes("REPORT.md", _report(cases).encode())
    staged.publish_json("provenance/transaction.json", transaction)
    staged.commit(
        expected_artifacts=_artifact_inventory(staged.stage_root),
        metadata={
            "schemaVersion": 3 if required_contract == 3 else 1,
            "documents": len(cases),
            "certifiedDocuments": len(cases),
            "trainingRecordsPublished": True,
        },
    )
    output = dict(summary)
    output["artifactRoot"] = str(staged.final_root)
    output["commitSha256"] = sha256_file(staged.final_root / "_COMMIT.json")
    return output
