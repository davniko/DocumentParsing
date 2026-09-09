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
from document_ocr.synthesis.config import SynthesisRawTextCertifiedPublicationConfig
from document_ocr.synthesis.raw_text_certification import (
    CertificationCaseResult,
    _host_audit,
)
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
    result: CertificationCaseResult


def _read_json_object(path: Path) -> dict[str, Any]:
    value = json.loads(read_regular_file_bytes(path))
    if not isinstance(value, dict):
        raise ValueError(f"artifact is not a JSON object: {path}")
    return cast(dict[str, Any], value)


def _read_jsonl(path: Path) -> tuple[dict[str, Any], ...]:
    payload = read_regular_file_bytes(path)
    if payload and not payload.endswith(b"\n"):
        raise ValueError(f"JSONL artifact lacks a terminal newline: {path}")
    rows: list[dict[str, Any]] = []
    for line_number, line in enumerate(payload.splitlines(), start=1):
        if not line.strip():
            raise ValueError(f"JSONL artifact contains a blank row: {path}:{line_number}")
        value = json.loads(line)
        if not isinstance(value, dict):
            raise ValueError(f"JSONL row is not an object: {path}:{line_number}")
        rows.append(cast(dict[str, Any], value))
    return tuple(rows)


def _certified_ids(root: Path) -> tuple[str, ...]:
    rows = _read_jsonl(root / "generation" / "results.jsonl")
    ids = tuple(
        sorted(
            cast(str, row["documentId"])
            for row in rows
            if row.get("status") == "certified"
            and isinstance(row.get("documentId"), str)
        )
    )
    if len(ids) != len(set(ids)):
        raise ValueError(f"certification results repeat a certified document: {root}")
    return ids


def _load_case(root: Path, document_id: str) -> _CertifiedCase:
    case_root = root / "cases" / document_id
    required = (
        "source.txt",
        "input-candidate.txt",
        "final.txt",
        "source-label.json",
        "target-label.json",
        "source-contract.json",
        "stages.json",
        "result.json",
    )
    for name in required:
        path = case_root / name
        if path.is_symlink() or not path.is_file():
            raise ValueError(f"certified case artifact is missing or unsafe: {path}")
    source = read_regular_file_bytes(case_root / "source.txt")
    candidate = read_regular_file_bytes(case_root / "input-candidate.txt")
    final = read_regular_file_bytes(case_root / "final.txt")
    if candidate != final:
        raise ValueError(f"read-only certification modified candidate bytes: {document_id}")
    result = CertificationCaseResult.model_validate_json(
        read_regular_file_bytes(case_root / "result.json"), strict=True
    )
    if (
        result.documentId != document_id
        or result.status != "certified"
        or result.auditPasses != 1
        or result.semanticFindings != 0
        or not result.hostAudit.passed
        or result.hostAudit.findings
        or result.sourceTextSha256 != sha256_bytes(source)
        or result.inputCandidateSha256 != sha256_bytes(candidate)
        or result.finalTextSha256 != sha256_bytes(final)
    ):
        raise ValueError(f"case does not meet the complete certification contract: {document_id}")
    source_label = _read_json_object(case_root / "source-label.json")
    target_label = _read_json_object(case_root / "target-label.json")
    contract = _read_json_object(case_root / "source-contract.json")
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
    replay = _host_audit(
        source=source.decode("utf-8"),
        output=final.decode("utf-8"),
        contract=contract,
    )
    if not replay.passed or replay.findings:
        raise ValueError(
            f"current host audit rejects previously certified case {document_id}: "
            f"{replay.findings}"
        )
    return _CertifiedCase(
        source_run=root,
        document_id=document_id,
        scenario_id=scenario_id,
        source_text=source,
        final_text=final,
        source_label=source_label,
        target_label=target_label,
        contract=contract,
        result=result,
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
    source_roots: list[Path] = []
    selected_ids: dict[Path, tuple[str, ...]] = {}
    for source in config.certification_sources:
        root = _validate_reference_run(
            project_root,
            source.run.path,
            source.run.commit_sha256,
            source.run.transaction_sha256,
        )
        ids = _certified_ids(root)
        if len(ids) != source.certified_documents:
            raise ValueError(f"configured certified count differs for source: {root}")
        if sha256_bytes(canonical_json_bytes(list(ids))) != source.certified_document_ids_sha256:
            raise ValueError(f"configured certified document identity differs for source: {root}")
        source_roots.append(root)
        selected_ids[root] = ids

    cases = tuple(
        _load_case(root, document_id)
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
        "schemaVersion": 1,
        "runId": config.run.run_id,
        "configSha256": sha256_file(config_path),
        "implementationSha256": sha256_file(_IMPLEMENTATION_PATH),
        "dependencyImplementationSha256": {
            "certification": sha256_file(
                Path(__file__).with_name("raw_text_certification.py")
            ),
            "inventoryRunner": sha256_file(
                Path(__file__).with_name("raw_text_inventory_probe.py")
            ),
            "targetSchema": sha256_file(
                Path(__file__).parents[1] / "label_schemas/bill_of_lading_v5.py"
            ),
        },
        "sourceRuns": [
            {
                "path": source.run.path,
                "commitSha256": source.run.commit_sha256,
                "transactionSha256": source.run.transaction_sha256,
                "certifiedDocuments": source.certified_documents,
                "certifiedDocumentIdsSha256": source.certified_document_ids_sha256,
            }
            for source in config.certification_sources
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
        return cast(dict[str, JsonValue], _read_json_object(staged.final_root / "summary.json"))
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
            }
        )
        prefix = f"cases/{row.scenario_id}"
        staged.publish_bytes(f"{prefix}/source.txt", row.source_text)
        staged.publish_bytes(f"{prefix}/final.txt", row.final_text)
        staged.publish_json(f"{prefix}/source-label.json", row.source_label)
        staged.publish_json(f"{prefix}/target-label.json", row.target_label)
        staged.publish_json(f"{prefix}/source-contract.json", row.contract)
        staged.publish_json(
            f"{prefix}/certification-result.json", row.result.model_dump(mode="json")
        )
        staged.publish_json(
            f"{prefix}/lineage.json", lineage[-1]
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
        "schemaVersion": 1,
        "runId": config.run.run_id,
        "status": "complete",
        "sourceCertificationRuns": len(source_roots),
        "documents": len(cases),
        "certifiedDocuments": len(cases),
        "semanticFindings": 0,
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
            "schemaVersion": 1,
            "documents": len(cases),
            "certifiedDocuments": len(cases),
            "trainingRecordsPublished": True,
        },
    )
    output = dict(summary)
    output["artifactRoot"] = str(staged.final_root)
    output["commitSha256"] = sha256_file(staged.final_root / "_COMMIT.json")
    return output
