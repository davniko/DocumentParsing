"""Deterministic replay of one committed modern certification run.

The staged-run receipt proves byte immutability.  This module proves that those bytes form one
complete certification run: configuration, transaction, per-case transcripts, audit plans,
aggregate results, and summary must all agree.  Correction, publication, and the parent pipeline
share this boundary so none of them can accidentally re-authorize only a convenient subset of a
certification artifact.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from decimal import Decimal
from pathlib import Path
from typing import Any, cast

from pydantic import JsonValue

from document_ocr.atomic import read_regular_file_bytes
from document_ocr.hashing import canonical_json_bytes, sha256_bytes, sha256_file
from document_ocr.synthesis.config import (
    SynthesisRawTextCertificationConfig,
    load_synthesis_raw_text_certification_config,
)
from document_ocr.synthesis.linguistic_probe_runtime import LinguisticUsageReceipt
from document_ocr.synthesis.raw_text_certification import (
    CertificationAuditReplay,
    CertificationCaseResult,
    CertificationStage,
    _audit_plan,
    _certification_mode,
    _combined_usage,
    _configured_audit_efforts,
    _validate_case_contract_labels,
    validate_certification_case_result,
    validate_certification_run_stage_identities,
)
from document_ocr.synthesis.raw_text_certification_host import (
    CertificationInvariantCase,
    CertificationInvariantContext,
    load_certification_invariant_context,
    replay_certification_invariant_case,
)

_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_TRANSACTION_KEYS = {
    "schemaVersion",
    "runId",
    "configSha256",
    "resolvedConfigSha256",
    "inputRunCommitSha256",
    "inputRunTransactionSha256",
    "benchmarkPlan",
    "inputCaseContractFilename",
    "promptSha256",
    "implementationSha256",
    "dependencyImplementationSha256",
    "documentIds",
    "runtime",
}
_V3_TRANSACTION_KEYS = _TRANSACTION_KEYS | {
    "invariantReferenceReceipt",
    "invariantReferenceReceiptSha256",
    "capacityConfigSha256",
}
_RUNTIME_KEYS = {
    "pydanticAiVersion",
    "openaiVersion",
    "model",
    "providerOrder",
    "maxConcurrentDocuments",
    "semanticAuditPasses",
    "auditContractVersion",
    "certificationMode",
    "reasoningEfforts",
}
_DEPENDENCY_KEYS = {
    "containerSemantics",
    "providerRuntime",
    "hybridRuntime",
    "inventory",
    "inventoryRunner",
    "rewriteContract",
    "rewriteRuntime",
    "transportCapacity",
}
_V3_DEPENDENCY_KEYS = _DEPENDENCY_KEYS | {
    "certificationHost",
    "certificationInvariants",
    "certificationReferences",
}


@dataclass(frozen=True, slots=True)
class ValidatedCertificationCase:
    document_id: str
    source: bytes
    candidate: bytes
    source_label: dict[str, Any]
    target_label: dict[str, Any]
    contract: dict[str, Any]
    audit_plan: dict[str, Any]
    result: CertificationCaseResult
    stages: tuple[CertificationStage, ...]
    replay: CertificationAuditReplay
    invariant_case: CertificationInvariantCase | None = None


@dataclass(frozen=True, slots=True)
class ValidatedCertificationRun:
    root: Path
    config: SynthesisRawTextCertificationConfig
    cases: tuple[ValidatedCertificationCase, ...]
    transaction: dict[str, Any]
    summary: dict[str, Any]
    invariant_context: CertificationInvariantContext | None = None

    def case_by_id(self) -> dict[str, ValidatedCertificationCase]:
        return {row.document_id: row for row in self.cases}


def _read_json_object(path: Path) -> dict[str, Any]:
    value = json.loads(read_regular_file_bytes(path))
    if not isinstance(value, dict):
        raise ValueError(f"certification artifact is not a JSON object: {path}")
    return cast(dict[str, Any], value)


def _read_results(path: Path) -> tuple[CertificationCaseResult, ...]:
    payload = read_regular_file_bytes(path)
    if not payload or not payload.endswith(b"\n"):
        raise ValueError("certification aggregate results lack a terminal newline")
    rows: list[CertificationCaseResult] = []
    for line_number, line in enumerate(payload.splitlines(), start=1):
        if not line.strip():
            raise ValueError(f"certification aggregate results contain a blank row: {line_number}")
        rows.append(CertificationCaseResult.model_validate_json(line, strict=True))
    return tuple(rows)


def _require_hash(value: object, *, label: str) -> str:
    if not isinstance(value, str) or _SHA256.fullmatch(value) is None:
        raise ValueError(f"certification {label} is not a SHA-256")
    return value


def _validate_transaction(
    *,
    root: Path,
    config_path: Path,
    config: SynthesisRawTextCertificationConfig,
    transaction: dict[str, Any],
    invariant_reference_receipt: dict[str, Any] | None,
) -> None:
    expected_transaction_keys = (
        _V3_TRANSACTION_KEYS if config.audit_contract_version == 3 else _TRANSACTION_KEYS
    )
    if set(transaction) != expected_transaction_keys:
        raise ValueError("certification transaction has an unexpected shape")
    dependencies = transaction.get("dependencyImplementationSha256")
    expected_dependency_keys = (
        _V3_DEPENDENCY_KEYS if config.audit_contract_version == 3 else _DEPENDENCY_KEYS
    )
    if not isinstance(dependencies, dict) or set(dependencies) != expected_dependency_keys:
        raise ValueError("certification transaction dependency coverage differs")
    for name, value in dependencies.items():
        _require_hash(value, label=f"dependency implementation {name}")
    _require_hash(transaction.get("implementationSha256"), label="implementation identity")
    runtime = transaction.get("runtime")
    if not isinstance(runtime, dict) or set(runtime) != _RUNTIME_KEYS:
        raise ValueError("certification transaction runtime has an unexpected shape")
    if not (
        isinstance(runtime.get("pydanticAiVersion"), str)
        and runtime["pydanticAiVersion"].strip()
        and isinstance(runtime.get("openaiVersion"), str)
        and runtime["openaiVersion"].strip()
    ):
        raise ValueError("certification transaction lacks runtime versions")
    expected_provider_order = (
        list(config.provider.provider_order or ()) if config.provider.kind == "openrouter" else []
    )
    if (
        transaction.get("schemaVersion") != (3 if config.audit_contract_version == 3 else 2)
        or transaction.get("runId") != config.run.run_id
        or config.run.run_id != root.name
        or transaction.get("configSha256") != sha256_file(config_path)
        or transaction.get("resolvedConfigSha256")
        != sha256_bytes(canonical_json_bytes(config.model_dump(mode="json")))
        or transaction.get("inputRunCommitSha256") != config.input_run.commit_sha256
        or transaction.get("inputRunTransactionSha256") != config.input_run.transaction_sha256
        or transaction.get("benchmarkPlan")
        != (
            config.benchmark_plan.model_dump(mode="json")
            if config.benchmark_plan is not None
            else None
        )
        or transaction.get("inputCaseContractFilename") != config.input_case_contract_filename
        or transaction.get("promptSha256") != config.prompt.sha256
        or transaction.get("documentIds") != list(config.case_ids)
        or runtime.get("model") != config.provider.model
        or runtime.get("providerOrder") != expected_provider_order
        or runtime.get("maxConcurrentDocuments") != config.workflow.max_concurrent_documents
        or runtime.get("semanticAuditPasses") != config.workflow.semantic_audit_passes
        or runtime.get("auditContractVersion") != config.audit_contract_version
        or runtime.get("certificationMode") != _certification_mode(config)
        or runtime.get("reasoningEfforts") != list(_configured_audit_efforts(config))
    ):
        raise ValueError("certification transaction differs from its exact configuration")
    if config.audit_contract_version == 3:
        invariant_config = config.invariant_inputs
        if invariant_config is None or invariant_reference_receipt is None:
            raise ValueError("contract-v3 transaction replay lacks invariant references")
        if (
            transaction.get("invariantReferenceReceipt") != invariant_reference_receipt
            or transaction.get("invariantReferenceReceiptSha256")
            != sha256_bytes(canonical_json_bytes(invariant_reference_receipt))
            or transaction.get("capacityConfigSha256")
            != sha256_bytes(
                canonical_json_bytes(invariant_config.transport_capacity.model_dump(mode="json"))
            )
        ):
            raise ValueError("contract-v3 transaction invariant inputs differ")


def _validate_summary(
    *,
    config: SynthesisRawTextCertificationConfig,
    cases: tuple[ValidatedCertificationCase, ...],
    summary: dict[str, Any],
) -> None:
    results = tuple(row.result for row in cases)
    stages = tuple(stage for row in cases for stage in row.stages)
    usage: LinguisticUsageReceipt = _combined_usage(stages)
    provider_cost = (
        str(usage.providerReportedCostUsd) if usage.providerReportedCostUsd is not None else None
    )
    expected_counts = {
        "certifiedDocuments": sum(row.status == "certified" for row in results),
        "evaluatedCleanDocuments": sum(row.status == "evaluated_clean" for row in results),
        "needsReviewDocuments": sum(row.status == "needs_review" for row in results),
        "callFailedDocuments": sum(row.status == "call_failed" for row in results),
    }
    if not (
        summary.get("schemaVersion") == (3 if config.audit_contract_version == 3 else 2)
        and summary.get("runId") == config.run.run_id
        and summary.get("status") == "complete"
        and summary.get("model") == config.provider.model
        and summary.get("auditContractVersion") == config.audit_contract_version
        and summary.get("certificationMode") == _certification_mode(config)
        and summary.get("requiredAuditPasses") == config.workflow.semantic_audit_passes
        and summary.get("documents") == len(results)
        and all(summary.get(key) == value for key, value in expected_counts.items())
        and summary.get("semanticFindings") == sum(row.semanticFindings for row in results)
        and (
            config.audit_contract_version != 3
            or (
                summary.get("deterministicFindings")
                == sum(row.deterministicFindings for row in results)
                and summary.get("deterministicRejectedBeforeProvider")
                == sum(row.status == "needs_review" and row.auditPasses == 0 for row in results)
            )
        )
        and summary.get("requests") == usage.requests
        and summary.get("providerAttempts") == len(stages)
        and summary.get("failedProviderAttempts")
        == sum(row.errorType is not None for row in stages)
        and summary.get("hostRejectedProviderOutputs")
        == sum(row.hostError is not None for row in stages)
        and summary.get("inputTokens") == usage.inputTokens
        and summary.get("reasoningTokens") == usage.reasoningTokens
        and summary.get("visibleOutputTokens") == usage.visibleOutputTokens
        and summary.get("outputTokens") == usage.outputTokens
        and summary.get("estimatedCostUsd") == str(usage.estimatedCostUsd)
        and summary.get("providerReportedCostUsd") == provider_cost
        and summary.get("estimatedCostPerThousandUsd")
        == str(usage.estimatedCostUsd * Decimal(1000) / len(results))
        and summary.get("providerReportedCostPerThousandUsd")
        == (
            str(usage.providerReportedCostUsd * Decimal(1000) / len(results))
            if usage.providerReportedCostUsd is not None
            else None
        )
        and summary.get("candidateBytesModified") == 0
        and summary.get("trainingRecordsPublished") is False
    ):
        raise ValueError("certification summary differs from deterministic replay")
    checkpointed = summary.get("checkpointedDocumentsAtStart")
    processed = summary.get("processedDocumentsThisInvocation")
    if (
        not isinstance(checkpointed, int)
        or isinstance(checkpointed, bool)
        or not isinstance(processed, int)
        or isinstance(processed, bool)
        or checkpointed < 0
        or processed < 0
        or checkpointed + processed != len(results)
    ):
        raise ValueError("certification summary checkpoint counts differ")


def load_validated_certification_run(
    root: Path, *, project_root: Path | None = None
) -> ValidatedCertificationRun:
    """Replay every artifact in one already commit-validated v2 or v3 run."""

    if root.is_symlink() or not root.is_dir():
        raise ValueError(f"certification root is not a plain directory: {root}")
    config_path = root / "config.yaml"
    if config_path.is_symlink() or not config_path.is_file():
        raise ValueError("certification run lacks a regular configuration")
    config = load_synthesis_raw_text_certification_config(config_path)
    if (
        "audit_contract_version" not in config.model_fields_set
        or config.audit_contract_version not in {2, 3}
    ):
        raise ValueError("authoritative certification run is not an explicit modern contract")
    invariant_context = None
    invariant_reference_receipt: dict[str, Any] | None = None
    if config.audit_contract_version == 3:
        if project_root is None:
            raise ValueError("contract-v3 replay requires the exact project root")
        invariant_config = config.invariant_inputs
        if invariant_config is None:
            raise ValueError("contract-v3 replay lacks pinned invariant inputs")
        invariant_context = load_certification_invariant_context(
            project_root=project_root.resolve(strict=True),
            config=invariant_config,
        )
        receipt_path = root / "references/certification-reference-receipt.json"
        invariant_reference_receipt = _read_json_object(receipt_path)
        if invariant_reference_receipt != invariant_context.references.receipt.model_dump(
            mode="json"
        ):
            raise ValueError("contract-v3 reference receipt differs from pinned inputs")
    prompt_path = root / "prompts/read-only-auditor.md"
    if (
        prompt_path.is_symlink()
        or not prompt_path.is_file()
        or sha256_file(prompt_path) != config.prompt.sha256
    ):
        raise ValueError("certification run prompt differs from its configured pin")
    transaction = _read_json_object(root / "provenance/transaction.json")
    transaction_marker = _read_json_object(root / "_TRANSACTION.json")
    if (
        set(transaction_marker) != {"schemaVersion", "runName", "transactionSha256"}
        or transaction_marker.get("schemaVersion") != 1
        or transaction_marker.get("runName") != root.name
        or transaction_marker.get("transactionSha256")
        != sha256_bytes(canonical_json_bytes(transaction))
    ):
        raise ValueError("certification transaction is not bound to its staged-run marker")
    _validate_transaction(
        root=root,
        config_path=config_path,
        config=config,
        transaction=transaction,
        invariant_reference_receipt=invariant_reference_receipt,
    )
    aggregate_results = _read_results(root / "generation/results.jsonl")
    if tuple(row.documentId for row in aggregate_results) != tuple(config.case_ids):
        raise ValueError("certification aggregate results differ from configured scope")

    cases: list[ValidatedCertificationCase] = []
    stages_by_document: dict[str, tuple[CertificationStage, ...]] = {}
    for aggregate_result in aggregate_results:
        document_id = aggregate_result.documentId
        case_root = root / "cases" / document_id
        required = (
            "source.txt",
            "input-candidate.txt",
            "final.txt",
            "source-label.json",
            "target-label.json",
            "source-contract.json",
            "audit-plan.json",
            "stages.json",
            "result.json",
            *(
                (
                    "inventory.json",
                    "deterministic-edits.json",
                    "invariant-envelope.json",
                    "invariant-audit.json",
                )
                if config.audit_contract_version == 3
                else ()
            ),
        )
        for name in required:
            path = case_root / name
            if path.is_symlink() or not path.is_file():
                raise ValueError(f"certification case artifact is missing or unsafe: {path}")
        source = read_regular_file_bytes(case_root / "source.txt")
        candidate = read_regular_file_bytes(case_root / "input-candidate.txt")
        final = read_regular_file_bytes(case_root / "final.txt")
        if candidate != final:
            raise ValueError(f"read-only certification modified candidate bytes: {document_id}")
        source_label = _read_json_object(case_root / "source-label.json")
        target_label = _read_json_object(case_root / "target-label.json")
        contract = _read_json_object(case_root / "source-contract.json")
        _validate_case_contract_labels(
            document_id=document_id,
            contract=contract,
            source_label=source_label,
            target_label=target_label,
        )
        audit_plan = _read_json_object(case_root / "audit-plan.json")
        expected_plan = _audit_plan(
            contract,
            source=source.decode("utf-8"),
            current=final.decode("utf-8"),
        )
        if audit_plan != expected_plan:
            raise ValueError(f"certification audit plan fails replay: {document_id}")
        stages_value = json.loads(read_regular_file_bytes(case_root / "stages.json"))
        if not isinstance(stages_value, list):
            raise ValueError(f"certification stages are not a list: {document_id}")
        stages = tuple(
            CertificationStage.model_validate_json(canonical_json_bytes(row), strict=True)
            for row in cast(list[JsonValue], stages_value)
        )
        result = CertificationCaseResult.model_validate_json(
            read_regular_file_bytes(case_root / "result.json"), strict=True
        )
        invariant_case = None
        if config.audit_contract_version == 3:
            if invariant_context is None:
                raise RuntimeError("contract-v3 reference context was not loaded")
            invariant_case = replay_certification_invariant_case(
                document_id=document_id,
                source=source.decode("utf-8"),
                candidate=final.decode("utf-8"),
                contract=contract,
                inventory_value=json.loads(read_regular_file_bytes(case_root / "inventory.json")),
                deterministic_edits_value=json.loads(
                    read_regular_file_bytes(case_root / "deterministic-edits.json")
                ),
                envelope_value=_read_json_object(case_root / "invariant-envelope.json"),
                audit_value=_read_json_object(case_root / "invariant-audit.json"),
                context=invariant_context,
            )
        replay = validate_certification_case_result(
            result=result,
            stages=stages,
            source=source.decode("utf-8"),
            current=final.decode("utf-8"),
            contract=contract,
            config=config,
            invariant_case=invariant_case,
        )
        if (
            result != aggregate_result
            or result.documentId != document_id
            or result.auditPlanSha256 != sha256_bytes(canonical_json_bytes(audit_plan))
        ):
            raise ValueError(f"certification case and aggregate results differ: {document_id}")
        stages_by_document[document_id] = stages
        cases.append(
            ValidatedCertificationCase(
                document_id=document_id,
                source=source,
                candidate=final,
                source_label=source_label,
                target_label=target_label,
                contract=contract,
                audit_plan=audit_plan,
                result=result,
                stages=stages,
                replay=replay,
                invariant_case=invariant_case,
            )
        )
    validate_certification_run_stage_identities(stages_by_document)
    frozen_cases = tuple(cases)
    summary = _read_json_object(root / "summary.json")
    _validate_summary(config=config, cases=frozen_cases, summary=summary)
    return ValidatedCertificationRun(
        root=root,
        config=config,
        cases=frozen_cases,
        transaction=transaction,
        summary=summary,
        invariant_context=invariant_context,
    )
