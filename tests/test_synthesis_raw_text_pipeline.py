from __future__ import annotations

import json
from decimal import Decimal
from pathlib import Path
from typing import Any, Literal

import pytest
import yaml
from pydantic_ai.messages import (
    ModelRequest,
    ModelResponse,
    SystemPromptPart,
    TextPart,
    UserPromptPart,
)
from pydantic_ai.usage import RequestUsage

from document_ocr.hashing import canonical_json_bytes, sha256_bytes
from document_ocr.synthesis.config import (
    SynthesisRawTextCertificationConfig,
    SynthesisRawTextCertifiedCorrectionConfig,
    SynthesisRawTextInventoryBatchConfig,
    SynthesisRawTextPipelineConfig,
    load_synthesis_raw_text_certification_config,
    load_synthesis_raw_text_certified_correction_config,
    load_synthesis_raw_text_certified_publication_config,
    load_synthesis_raw_text_inventory_batch_config,
    load_synthesis_raw_text_pipeline_config,
)
from document_ocr.synthesis.linguistic_probe_runtime import (
    LinguisticUsageReceipt,
    model_messages,
    usage_receipt,
)
from document_ocr.synthesis.raw_text_certification import (
    CertificationAuditReplay,
    CertificationCaseResult,
    CertificationHostAudit,
    CertificationRouteError,
    CertificationStage,
    SemanticAuditEvidence,
    SemanticAuditFinding,
    SemanticAuditOutput,
    _audit_expected_coverage,
    _audit_plan,
    _certification_outcome,
    _combined_usage,
    _host_audit,
    _payload,
)
from document_ocr.synthesis.raw_text_certification_invariants import (
    CertificationInvariantAudit,
    DeterministicCertificationFinding,
    DeterministicLineRepair,
    InvariantCheck,
    InvariantEvidence,
)
from document_ocr.synthesis.raw_text_certified_correction import (
    CorrectionCaseResult,
    _refine_carrier_occurrence_contract,
)
from document_ocr.synthesis.raw_text_hybrid_probe import _artifact_inventory
from document_ocr.synthesis.raw_text_inventory_probe import InventoryProbeCaseResult
from document_ocr.synthesis.raw_text_pipeline import (
    RawTextPipelineBlockedError,
    _validate_pipeline_config_file,
    run_raw_text_pipeline,
)
from document_ocr.synthesis.run_safety import StagedArtifactRun

_PROJECT_ROOT = Path(__file__).resolve().parents[1]
_DOCUMENT_IDS = tuple(f"doc_{value * 64}" for value in ("1", "2", "3"))


def _empty_usage() -> LinguisticUsageReceipt:
    return LinguisticUsageReceipt(
        requests=0,
        providerResponseIds=(),
        finishReasons=(),
        inputTokens=0,
        cacheReadTokens=0,
        cacheWriteTokens=0,
        outputTokens=0,
        reasoningTokens=0,
        visibleOutputTokens=0,
        estimatedCostUsd=Decimal(0),
    )


def _write_pipeline_fixture(
    tmp_path: Path,
) -> tuple[Path, SynthesisRawTextPipelineConfig]:
    source_inventory = load_synthesis_raw_text_inventory_batch_config(
        _PROJECT_ROOT / "configs/synthesis/"
        "mpci_bl_raw_text_inventory100_additional_maritime_v55_pipeline_glm53.yaml"
    )
    inventory_value = source_inventory.model_dump(mode="python")
    inventory_value["run"] = {"run_id": "inventory-test", "output_dir": "runs"}
    inventory_value["selection"] = "explicit_pinned_document_ids"
    inventory_value["cases"] = [{"document_id": document_id} for document_id in _DOCUMENT_IDS]
    inventory_workflow = inventory_value["workflow"]
    inventory_workflow["documents"] = 3
    inventory_workflow["max_concurrent_documents"] = 3
    inventory = SynthesisRawTextInventoryBatchConfig.model_validate(inventory_value, strict=True)
    inventory_path = tmp_path / "inventory.yaml"
    inventory_path.write_text(
        yaml.safe_dump(inventory.model_dump(mode="python"), sort_keys=False),
        encoding="utf-8",
    )

    certification_prompt = tmp_path / "certification.md"
    correction_prompt = tmp_path / "correction.md"
    certification_prompt.write_text("read-only audit", encoding="utf-8")
    correction_prompt.write_text("exact-line correction", encoding="utf-8")

    source_pipeline = load_synthesis_raw_text_pipeline_config(
        _PROJECT_ROOT
        / "configs/synthesis/mpci_bl_raw_text_pipeline100_additional_maritime_v1_glm53.yaml"
    )
    pipeline_value = source_pipeline.model_dump(mode="python")
    pipeline_value["run"] = {"run_id": "pipeline-test", "output_dir": "runs"}
    pipeline_value["inventory_config"] = {
        "path": "inventory.yaml",
        "sha256": sha256_bytes(inventory_path.read_bytes()),
    }
    pipeline_value["certification"]["shard_size"] = 2
    pipeline_value["certification"]["audit_contract_version"] = 2
    pipeline_value["certification"]["prompt"] = {
        "path": "certification.md",
        "sha256": sha256_bytes(certification_prompt.read_bytes()),
    }
    pipeline_value["certification"]["workflow"]["documents"] = 2
    pipeline_value["certification"]["workflow"]["max_concurrent_documents"] = 2
    pipeline_value["certification"]["workflow"].update(
        {
            "semantic_audit_passes": 2,
            "confirmation_reasoning_effort": "high",
            "evaluation_only": False,
            "require_complete_dimension_coverage": True,
            "require_unanimous_clean": True,
        }
    )
    pipeline_value["certification"]["provider"]["reasoning_effort"] = "low"
    pipeline_value["certification"]["provider"]["max_output_tokens"] = 12288
    pipeline_value["correction"]["shard_size"] = 2
    pipeline_value["correction"]["prompt"] = {
        "path": "correction.md",
        "sha256": sha256_bytes(correction_prompt.read_bytes()),
    }
    pipeline_value["correction"]["workflow"]["documents"] = 2
    pipeline_value["correction"]["workflow"]["max_concurrent_documents"] = 2
    pipeline_value["correction"]["workflow"]["required_audit_contract_version"] = 2
    pipeline_value["publication"]["documents"] = 3
    pipeline_value["publication"]["required_audit_contract_version"] = 2
    pipeline_value["workflow"]["documents"] = 3
    pipeline = SynthesisRawTextPipelineConfig.model_validate(pipeline_value, strict=True)
    pipeline_path = tmp_path / "pipeline.yaml"
    pipeline_path.write_text(
        yaml.safe_dump(pipeline.model_dump(mode="python"), sort_keys=False),
        encoding="utf-8",
    )
    return pipeline_path, pipeline


def _contract_v3_pipeline_value(
    pipeline: SynthesisRawTextPipelineConfig,
) -> dict[str, Any]:
    value = pipeline.model_dump(mode="python")
    value["schema_version"] = 2
    value["task"] = "bill_of_lading_synthetic_raw_text_pipeline_v2"
    value["certification"]["audit_contract_version"] = 3
    value["certification"]["invariant_inputs"] = {
        "iso3166_snapshot": {"path": "iso.json", "sha256": "1" * 64},
        "geonames_registry": {
            "path": "geonames",
            "commit_sha256": "2" * 64,
            "transaction_sha256": "3" * 64,
        },
        "geonames_registry_receipt_sha256": "4" * 64,
        "phonenumberslite_version": "9.0.38",
        "package_registry": {"path": "packages.json", "sha256": "5" * 64},
        "package_registry_entries": 1,
        "transport_capacity": {
            "policy": "source_type_aware_maersk_upper_bounds_v1",
            "published_reference_margin_fraction": Decimal("0.05"),
            "twenty_standard_payload_kg": Decimal("28300"),
            "twenty_standard_volume_m3": Decimal("33.2"),
            "forty_standard_payload_kg": Decimal("28870"),
            "forty_standard_volume_m3": Decimal("67.7"),
            "forty_high_cube_payload_kg": Decimal("28690"),
            "forty_high_cube_volume_m3": Decimal("76.4"),
            "forty_five_high_cube_payload_kg": Decimal("27650"),
            "forty_five_high_cube_volume_m3": Decimal("86"),
            "out_of_gauge_payload_kg": Decimal("47300"),
            "unclassified_payload_kg": Decimal("47300"),
            "unclassified_volume_m3": Decimal("86"),
        },
    }
    value["correction"]["max_rounds_per_document"] = 1
    value["correction"]["max_attempts_per_round"] = 1
    value["correction"]["workflow"]["required_audit_contract_version"] = 3
    value["correction"]["workflow"]["max_provider_route_rounds"] = 1
    value["correction"]["workflow"]["max_successful_model_responses_per_document"] = 1
    value["publication"]["required_audit_contract_version"] = 3
    return value


def test_pipeline_v3_has_an_explicit_top_level_launch_contract(tmp_path: Path) -> None:
    _legacy_path, legacy = _write_pipeline_fixture(tmp_path)
    value = _contract_v3_pipeline_value(legacy)
    current = SynthesisRawTextPipelineConfig.model_validate(value, strict=True)
    current_path = tmp_path / "pipeline-v3.yaml"
    current_path.write_text(
        yaml.safe_dump(current.model_dump(mode="python"), sort_keys=False),
        encoding="utf-8",
    )

    _validate_pipeline_config_file(config_path=current_path, config=current)
    assert current.schema_version == 2
    assert current.certification.audit_contract_version == 3
    assert current.correction.workflow.required_audit_contract_version == 3
    assert current.publication.required_audit_contract_version == 3

    stale_schema = dict(value)
    stale_schema["schema_version"] = 1
    stale_schema["task"] = "bill_of_lading_synthetic_raw_text_pipeline_v1"
    with pytest.raises(ValueError, match="top-level schema/task v2"):
        SynthesisRawTextPipelineConfig.model_validate(stale_schema, strict=True)

    repeated_correction = _contract_v3_pipeline_value(legacy)
    repeated_correction["correction"]["max_rounds_per_document"] = 2
    with pytest.raises(ValueError, match="exactly one round and one attempt"):
        SynthesisRawTextPipelineConfig.model_validate(repeated_correction, strict=True)


def test_pipeline_v3_quarantines_semantic_and_unowned_findings_without_retry() -> None:
    import document_ocr.synthesis.raw_text_pipeline as pipeline_module

    source = "ORIGINAL\n"
    candidate = "SYNTHETIC\n"
    common_result = {
        "documentId": _DOCUMENT_IDS[0],
        "auditContractVersion": 3,
        "certificationMode": "production",
        "auditPlanSha256": "1" * 64,
        "invariantEnvelopeSha256": "2" * 64,
        "invariantAuditSha256": "3" * 64,
        "status": "needs_review",
        "reason": "independent audit found a defect",
        "requiredAuditPasses": 2,
        "candidateImmutable": True,
        "hostAudit": CertificationHostAudit(
            passed=True,
            findings=(),
            sourceLines=1,
            outputLines=1,
            targetLiteralRequirements=0,
            targetOccurrenceRequirements=0,
        ),
        "sourceTextSha256": sha256_bytes(source.encode()),
        "inputCandidateSha256": sha256_bytes(candidate.encode()),
        "finalTextSha256": sha256_bytes(candidate.encode()),
        "usage": _empty_usage(),
    }
    clean_invariant_audit = CertificationInvariantAudit(
        schemaVersion=1,
        envelopeSha256="4" * 64,
        passed=True,
        checks=(InvariantCheck(checkId="target_literals", findings=0, passed=True),),
        findings=(),
    )
    semantic_result = CertificationCaseResult.model_validate(
        {
            **common_result,
            "auditPasses": 1,
            "cleanAuditPasses": 0,
            "semanticFindings": 1,
            "deterministicFindings": 0,
        },
        strict=True,
    )

    assert (
        pipeline_module._v3_certification_terminal_reason(
            result=semantic_result,
            invariant_audit=clean_invariant_audit,
            correction_rounds=0,
            correction_round_limit=1,
        )
        == pipeline_module._V3_CERTIFICATION_QUARANTINE_REASON
    )

    repair = DeterministicLineRepair(
        method="replace_exact_fragment_v1",
        lineId="L00001",
        expectedCurrentLineSha256=sha256_bytes(candidate.rstrip("\n").encode()),
        oldFragment="SYNTHETIC",
        newFragment="TARGET",
    )
    unowned = DeterministicCertificationFinding(
        invariantId="D" + "5" * 16,
        findingKind="target_fact_mismatch",
        dimension="target_fact_fidelity",
        evidence=(InvariantEvidence(lineId="L00001", currentLine="SYNTHETIC"),),
        targetPaths=("documentPatch.billOfLadingNumber",),
        problem="Target value is absent.",
        repairs=(),
    )
    repairable = unowned.model_copy(update={"invariantId": "D" + "6" * 16, "repairs": (repair,)})

    def deterministic_audit(
        finding: DeterministicCertificationFinding,
    ) -> CertificationInvariantAudit:
        return CertificationInvariantAudit(
            schemaVersion=1,
            envelopeSha256="4" * 64,
            passed=False,
            checks=(InvariantCheck(checkId="target_literals", findings=1, passed=False),),
            findings=(finding,),
        )

    deterministic_result = CertificationCaseResult.model_validate(
        {
            **common_result,
            "auditPasses": 2,
            "cleanAuditPasses": 2,
            "semanticFindings": 0,
            "deterministicFindings": 1,
        },
        strict=True,
    )
    assert (
        pipeline_module._v3_certification_terminal_reason(
            result=deterministic_result,
            invariant_audit=deterministic_audit(unowned),
            correction_rounds=0,
            correction_round_limit=1,
        )
        == pipeline_module._V3_CERTIFICATION_QUARANTINE_REASON
    )
    assert (
        pipeline_module._v3_certification_terminal_reason(
            result=deterministic_result,
            invariant_audit=deterministic_audit(repairable),
            correction_rounds=0,
            correction_round_limit=1,
        )
        is None
    )
    assert (
        pipeline_module._v3_certification_terminal_reason(
            result=deterministic_result,
            invariant_audit=deterministic_audit(repairable),
            correction_rounds=1,
            correction_round_limit=1,
        )
        == pipeline_module._CORRECTION_ROUND_LIMIT_REASON
    )


def test_pipeline_v3_propagates_current_contract_through_publication(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import document_ocr.synthesis.raw_text_pipeline as pipeline_module

    _legacy_path, legacy = _write_pipeline_fixture(tmp_path)
    value = _contract_v3_pipeline_value(legacy)
    current = SynthesisRawTextPipelineConfig.model_validate(value, strict=True)
    current_path = tmp_path / "pipeline-v3-run.yaml"
    current_path.write_text(
        yaml.safe_dump(current.model_dump(mode="python"), sort_keys=False),
        encoding="utf-8",
    )
    certification_configs: list[SynthesisRawTextCertificationConfig] = []
    publication_configs = []
    correction_called = False

    monkeypatch.setattr(
        pipeline_module,
        "preflight_raw_text_pipeline",
        lambda **_kwargs: {"status": "preflight_complete"},
    )

    def run_inventory(*, project_root: Path, config: Any, **_kwargs: Any) -> dict[str, Any]:
        rows = tuple(_inventory_result(document_id) for document_id in _DOCUMENT_IDS)
        return _commit_fake_run(
            project_root=project_root,
            config=config,
            rows=rows,
            summary={"status": "complete", "documents": len(rows)},
            case_texts={
                document_id: (
                    _source_text(document_id),
                    _initial_candidate(document_id),
                    _initial_candidate(document_id),
                )
                for document_id in _DOCUMENT_IDS
            },
        )

    def run_certification(
        *, project_root: Path, config: SynthesisRawTextCertificationConfig, **_kwargs: Any
    ) -> dict[str, Any]:
        certification_configs.append(config)
        input_root = project_root / config.input_run.path
        rows: list[CertificationCaseResult] = []
        case_texts: dict[str, tuple[str, str, str]] = {}
        for document_id in config.case_ids:
            source = (input_root / "cases" / document_id / "source.txt").read_text()
            candidate = (input_root / "cases" / document_id / "final.txt").read_text()
            rows.append(
                CertificationCaseResult(
                    documentId=document_id,
                    auditContractVersion=3,
                    certificationMode="production",
                    auditPlanSha256="a" * 64,
                    invariantEnvelopeSha256="b" * 64,
                    invariantAuditSha256="c" * 64,
                    status="certified",
                    reason="test current-contract certification",
                    auditPasses=2,
                    cleanAuditPasses=2,
                    requiredAuditPasses=2,
                    semanticFindings=0,
                    deterministicFindings=0,
                    candidateImmutable=True,
                    hostAudit=CertificationHostAudit(
                        passed=True,
                        findings=(),
                        sourceLines=len(source.splitlines()),
                        outputLines=len(candidate.splitlines()),
                        targetLiteralRequirements=0,
                        targetOccurrenceRequirements=0,
                    ),
                    sourceTextSha256=sha256_bytes(source.encode()),
                    inputCandidateSha256=sha256_bytes(candidate.encode()),
                    finalTextSha256=sha256_bytes(candidate.encode()),
                    usage=_empty_usage(),
                )
            )
            case_texts[document_id] = (source, candidate, candidate)
        transaction_sha256 = sha256_bytes(
            canonical_json_bytes(
                {
                    "kind": "test-current-certification",
                    "runId": config.run.run_id,
                }
            )
        )
        staged = StagedArtifactRun(
            output_parent=project_root / config.run.output_dir,
            run_name=config.run.run_id,
            transaction_sha256=transaction_sha256,
        )
        staged.publish_bytes(
            "config.yaml",
            yaml.safe_dump(config.model_dump(mode="python"), sort_keys=False).encode(),
        )
        staged.publish_json("summary.json", {"schemaVersion": 3, "documents": len(rows)})
        staged.publish_bytes(
            "generation/results.jsonl",
            b"".join(canonical_json_bytes(row.model_dump(mode="json")) + b"\n" for row in rows),
        )
        for row in rows:
            source, candidate, final = case_texts[row.documentId]
            prefix = f"cases/{row.documentId}"
            staged.publish_bytes(f"{prefix}/source.txt", source.encode())
            staged.publish_bytes(f"{prefix}/input-candidate.txt", candidate.encode())
            staged.publish_bytes(f"{prefix}/final.txt", final.encode())
            staged.publish_json(f"{prefix}/result.json", row.model_dump(mode="json"))
        staged.commit(
            expected_artifacts=_artifact_inventory(staged.stage_root),
            metadata={"schemaVersion": 3, "documents": len(rows)},
        )
        return {"status": "complete", "artifactRoot": str(staged.final_root)}

    def run_correction(**_kwargs: Any) -> dict[str, Any]:
        nonlocal correction_called
        correction_called = True
        raise AssertionError("clean contract-v3 cases must not enter correction")

    def run_publication(*, project_root: Path, config: Any, **_kwargs: Any) -> dict[str, Any]:
        publication_configs.append(config)
        return _commit_fake_run(
            project_root=project_root,
            config=config,
            rows=(),
            summary={
                "status": "complete",
                "documents": len(_DOCUMENT_IDS),
                "certifiedDocuments": len(_DOCUMENT_IDS),
                "trainingRecordsPublished": True,
            },
        )

    monkeypatch.setattr(pipeline_module, "run_raw_text_inventory_batch", run_inventory)
    monkeypatch.setattr(pipeline_module, "run_raw_text_certification", run_certification)
    monkeypatch.setattr(pipeline_module, "run_raw_text_certified_correction", run_correction)
    monkeypatch.setattr(pipeline_module, "run_raw_text_certified_publication", run_publication)

    result = run_raw_text_pipeline(
        project_root=tmp_path,
        config_path=current_path,
        config=current,
    )

    assert result["schemaVersion"] == 2
    assert result["status"] == "complete"
    assert correction_called is False
    assert certification_configs
    assert all(row.schema_version == 3 for row in certification_configs)
    assert all(row.audit_contract_version == 3 for row in certification_configs)
    assert len(publication_configs) == 1
    assert publication_configs[0].workflow.required_audit_contract_version == 3
    transaction = json.loads(
        (Path(str(result["artifactRoot"])) / "provenance/transaction.json").read_text()
    )
    assert transaction["schemaVersion"] == 2


def _certification_contract(
    *, document_id: str, source_label: dict[str, Any], target_label: dict[str, Any]
) -> dict[str, Any]:
    return {
        "changedLeaves": [],
        "compactLabelChangeContract": [],
        "sourceStatusPreservationRequirements": [],
        "surfaceRenderingRequirements": [],
        "operationalFlavorRequirements": [],
        "targetLiteralRequirements": [],
        "targetValueOccurrenceRequirements": [],
        "targetIntegrity": {"final_receipt": {"valid": True}, "topology_matched": True},
        "rawAuxiliaryIdentityRequirements": [],
        "jurisdictionalSurfaceRequirements": [],
        "compoundPartyFlavorRequirements": [],
        "inlineSlotTopologyRequirements": [],
        "cargoFlavorRewriteRequirements": [],
        "anchoredScalarReplacementRequirements": [],
        "sourceSemanticRoleHints": [],
        "sourceLabel": source_label,
        "targetLabel": target_label,
        "result": {"documentId": document_id},
    }


def _semantic_output(
    *,
    contract: dict[str, Any],
    source: str,
    candidate: str,
    finding: SemanticAuditFinding | None,
) -> SemanticAuditOutput:
    coverage = _audit_expected_coverage(_audit_plan(contract, source=source, current=candidate))
    failed_dimension = next(iter(coverage)) if finding is not None else None
    failed_obligation_id = coverage[failed_dimension][0] if failed_dimension is not None else None
    return SemanticAuditOutput.model_validate(
        {
            "auditContractVersion": 2,
            "dimensionChecks": tuple(
                {
                    "dimension": dimension,
                    "assessment": f"Completed the {dimension} obligations.",
                }
                for dimension in coverage
            ),
            "findings": (
                (
                    {
                        "findingKind": finding.findingKind,
                        "evidence": tuple(
                            {"lineId": evidence.lineId} for evidence in finding.evidence
                        ),
                        "problem": finding.problem,
                        "obligationIds": (failed_obligation_id,),
                    },
                )
                if finding is not None
                else ()
            ),
        },
        strict=True,
    )


def _certification_stage(
    *,
    project_root: Path,
    config: SynthesisRawTextCertificationConfig,
    document_id: str,
    source: str,
    candidate: str,
    contract: dict[str, Any],
    audit_pass: Literal[1, 2],
    output: SemanticAuditOutput | None,
) -> CertificationStage:
    plan = _audit_plan(contract, source=source, current=candidate)
    host = _host_audit(source=source, output=candidate, contract=contract)
    payload = _payload(
        document_id=document_id,
        source=source,
        current=candidate,
        source_label=contract["sourceLabel"],
        target_label=contract["targetLabel"],
        host_findings=host.findings,
        audit_contract_version=2,
        audit_plan=plan,
    )
    assert config.provider.kind == "openrouter"
    assert config.provider.provider_order is not None
    route = config.provider.provider_order[0]
    identity = f"{document_id}-{audit_pass}"
    request = ModelRequest(
        parts=(
            SystemPromptPart((project_root / config.prompt.path).read_text()),
            UserPromptPart(json.dumps(payload, ensure_ascii=False, separators=(",", ":"))),
        ),
        run_id=f"run-{identity}",
        conversation_id=f"conversation-{identity}",
    )
    reasoning_effort: Literal["low", "high"] = "low" if audit_pass == 1 else "high"
    common: dict[str, Any] = {
        "auditContractVersion": 2,
        "auditPass": audit_pass,
        "reasoningEffort": reasoning_effort,
        "auditPlanSha256": sha256_bytes(canonical_json_bytes(plan)),
        "routeProvider": route,
        "routeRound": 1,
        "routeAttempt": 1,
        "retryDelayBeforeSeconds": 0.0,
        "inputPayloadSha256": sha256_bytes(canonical_json_bytes(payload)),
        "outputSchemaSha256": sha256_bytes(
            canonical_json_bytes(SemanticAuditOutput.model_json_schema())
        ),
        "startedAtUnixSeconds": float(audit_pass * 2),
        "completedAtUnixSeconds": float(audit_pass * 2 + 1),
        "hostError": None,
    }
    if output is None:
        route_error = CertificationRouteError(
            category="other",
            errorType="RuntimeError",
            message="terminal fake provider failure",
        )
        return CertificationStage.model_validate(
            {
                **common,
                "usage": _empty_usage(),
                "messages": model_messages((request,)),
                "modelOutput": None,
                "errorType": "RuntimeError",
                "errorMessage": route_error.message,
                "retryableRouteError": False,
                "routeError": route_error,
            },
            strict=True,
        )
    response = ModelResponse(
        parts=(TextPart(canonical_json_bytes(output.model_dump(mode="json")).decode()),),
        usage=RequestUsage(input_tokens=1, output_tokens=1),
        model_name=config.provider.model,
        provider_name="openrouter",
        provider_url="https://openrouter.ai/api/v1",
        provider_details={
            "downstream_provider": {
                "deepinfra/fp4": "DeepInfra",
                "coreweave/fp8": "CoreWeave",
                "fireworks": "Fireworks",
                "nextbit/fp8": "NextBit",
            }[route]
        },
        provider_response_id=f"response-{identity}",
        finish_reason="stop",
        run_id=request.run_id,
        conversation_id=request.conversation_id,
    )
    return CertificationStage.model_validate(
        {
            **common,
            "usage": usage_receipt((response,), config.provider.pricing),
            "messages": model_messages((request, response)),
            "modelOutput": output.model_dump(mode="json"),
            "errorType": None,
            "errorMessage": None,
        },
        strict=True,
    )


def _complete_certification_case(
    *,
    project_root: Path,
    config: SynthesisRawTextCertificationConfig,
    document_id: str,
    status: Literal["certified", "needs_review", "call_failed"],
    source: str,
    candidate: str,
) -> tuple[
    CertificationCaseResult,
    tuple[CertificationStage, ...],
    dict[str, Any],
    dict[str, Any],
    dict[str, Any],
]:
    input_case = project_root / config.input_run.path / "cases" / document_id
    source_label = json.loads((input_case / "source-label.json").read_text())
    target_label = json.loads((input_case / "target-label.json").read_text())
    contract = json.loads((input_case / config.input_case_contract_filename).read_text())
    finding = (
        SemanticAuditFinding(
            findingKind="target_fact_mismatch",
            evidence=(SemanticAuditEvidence(lineId="L00001", currentFragment=candidate),),
            problem="The candidate contains a test contradiction.",
        )
        if status == "needs_review"
        else None
    )
    output = _semantic_output(
        contract=contract,
        source=source,
        candidate=candidate,
        finding=finding,
    )
    stages = (
        (
            _certification_stage(
                project_root=project_root,
                config=config,
                document_id=document_id,
                source=source,
                candidate=candidate,
                contract=contract,
                audit_pass=1,
                output=None,
            ),
        )
        if status == "call_failed"
        else (
            _certification_stage(
                project_root=project_root,
                config=config,
                document_id=document_id,
                source=source,
                candidate=candidate,
                contract=contract,
                audit_pass=1,
                output=output,
            ),
            *(
                (
                    _certification_stage(
                        project_root=project_root,
                        config=config,
                        document_id=document_id,
                        source=source,
                        candidate=candidate,
                        contract=contract,
                        audit_pass=2,
                        output=output,
                    ),
                )
                if status == "certified"
                else ()
            ),
        )
    )
    replay = CertificationAuditReplay(
        outputs=tuple(output for stage in stages if stage.modelOutput is not None),
        findings=(finding,) if finding is not None else (),
        audit_passes=0 if status == "call_failed" else len(stages),
        clean_audit_passes=2 if status == "certified" else 0,
        audit_plan_sha256=sha256_bytes(
            canonical_json_bytes(_audit_plan(contract, source=source, current=candidate))
        ),
    )
    host = _host_audit(source=source, output=candidate, contract=contract)
    expected_status, reason = _certification_outcome(
        replay=replay,
        host_audit=host,
        mode="production",
        required_passes=2,
    )
    assert expected_status == status
    result = CertificationCaseResult(
        documentId=document_id,
        auditContractVersion=2,
        certificationMode="production",
        auditPlanSha256=replay.audit_plan_sha256,
        status=status,
        reason=reason,
        auditPasses=replay.audit_passes,
        cleanAuditPasses=replay.clean_audit_passes,
        requiredAuditPasses=2,
        semanticFindings=len(replay.findings),
        candidateImmutable=True,
        hostAudit=host,
        sourceTextSha256=sha256_bytes(source.encode()),
        inputCandidateSha256=sha256_bytes(candidate.encode()),
        finalTextSha256=sha256_bytes(candidate.encode()),
        usage=_combined_usage(stages),
    )
    return result, stages, source_label, target_label, contract


def _commit_fake_run(
    *,
    project_root: Path,
    config: Any,
    rows: tuple[Any, ...],
    summary: dict[str, Any],
    case_texts: dict[str, tuple[str, str, str]] | None = None,
) -> dict[str, Any]:
    inventory_rows = bool(rows) and all(isinstance(row, InventoryProbeCaseResult) for row in rows)
    certification_rows = isinstance(config, SynthesisRawTextCertificationConfig)
    correction_rows = isinstance(config, SynthesisRawTextCertifiedCorrectionConfig)
    certification_material: dict[
        str,
        tuple[
            tuple[CertificationStage, ...],
            dict[str, Any],
            dict[str, Any],
            dict[str, Any],
        ],
    ] = {}
    if certification_rows:
        if case_texts is None:
            raise AssertionError("fake certification requires exact case bytes")
        rebuilt: list[CertificationCaseResult] = []
        for requested in rows:
            source, _input_candidate, candidate = case_texts[requested.documentId]
            result, stages, source_label, target_label, contract = _complete_certification_case(
                project_root=project_root,
                config=config,
                document_id=requested.documentId,
                status=requested.status,
                source=source,
                candidate=candidate,
            )
            rebuilt.append(result)
            certification_material[requested.documentId] = (
                stages,
                source_label,
                target_label,
                contract,
            )
        rows = tuple(rebuilt)
    if inventory_rows:
        summary = {
            **summary,
            "schemaVersion": 2,
            "runId": config.run.run_id,
            "status": "complete",
            "liveDocuments": len(rows),
            "trainingReadyDocuments": sum(row.status == "training_ready" for row in rows),
            "needsReviewDocuments": sum(row.status == "needs_review" for row in rows),
            "callFailedDocuments": sum(row.status == "call_failed" for row in rows),
            "compilerBlockedDocuments": sum(row.status == "compiler_blocked" for row in rows),
        }
    elif certification_rows:
        all_stages = tuple(
            stage
            for document_id in config.case_ids
            for stage in certification_material[document_id][0]
        )
        total_usage = _combined_usage(all_stages)
        summary = {
            **summary,
            "schemaVersion": 2,
            "runId": config.run.run_id,
            "status": "complete",
            "model": config.provider.model,
            "auditContractVersion": 2,
            "certificationMode": "production",
            "requiredAuditPasses": 2,
            "documents": len(rows),
            "checkpointedDocumentsAtStart": 0,
            "processedDocumentsThisInvocation": len(rows),
            "certifiedDocuments": sum(row.status == "certified" for row in rows),
            "evaluatedCleanDocuments": 0,
            "needsReviewDocuments": sum(row.status == "needs_review" for row in rows),
            "callFailedDocuments": sum(row.status == "call_failed" for row in rows),
            "semanticFindings": sum(row.semanticFindings for row in rows),
            "requests": total_usage.requests,
            "providerAttempts": len(all_stages),
            "failedProviderAttempts": sum(row.errorType is not None for row in all_stages),
            "hostRejectedProviderOutputs": sum(row.hostError is not None for row in all_stages),
            "inputTokens": total_usage.inputTokens,
            "reasoningTokens": total_usage.reasoningTokens,
            "visibleOutputTokens": total_usage.visibleOutputTokens,
            "outputTokens": total_usage.outputTokens,
            "estimatedCostUsd": str(total_usage.estimatedCostUsd),
            "providerReportedCostUsd": (
                str(total_usage.providerReportedCostUsd)
                if total_usage.providerReportedCostUsd is not None
                else None
            ),
            "estimatedCostPerThousandUsd": str(
                total_usage.estimatedCostUsd * Decimal(1000) / len(rows)
            ),
            "providerReportedCostPerThousandUsd": (
                str(total_usage.providerReportedCostUsd * Decimal(1000) / len(rows))
                if total_usage.providerReportedCostUsd is not None
                else None
            ),
            "wallSeconds": 1.0,
            "throughputDocumentsPerHour": float(len(rows) * 3600),
            "candidateBytesModified": 0,
            "trainingRecordsPublished": False,
        }
    elif correction_rows:
        summary = {
            **summary,
            "schemaVersion": 1,
            "runId": config.run.run_id,
            "status": "complete",
            "documents": len(rows),
            "unchangedCertifiedDocuments": sum(row.status == "unchanged_certified" for row in rows),
            "correctionCandidateDocuments": sum(
                row.status == "correction_candidate" for row in rows
            ),
            "needsReviewDocuments": sum(row.status == "needs_review" for row in rows),
            "callFailedDocuments": sum(row.status == "call_failed" for row in rows),
            "sourceFindings": sum(row.sourceFindings for row in rows),
            "citedLines": sum(row.citedLines for row in rows),
            "changedLines": sum(row.changedLines for row in rows),
            "trainingRecordsPublished": False,
        }
    config_payload = yaml.safe_dump(
        config.model_dump(mode="python"),
        sort_keys=False,
    ).encode()
    if inventory_rows:
        import document_ocr.synthesis.raw_text_pipeline as pipeline_module

        stage_hashes = pipeline_module._stage_implementation_hashes()
        transaction = {
            "schemaVersion": 2,
            "runId": config.run.run_id,
            "configSha256": sha256_bytes(config_payload),
            "baseBatchConfigSha256": config.base_batch_config.sha256,
            "regressionOracleSha256": config.regression_oracle.sha256,
            "templateMutationProfileSha256": (
                config.template_mutation_profile.sha256
                if config.template_mutation_profile is not None
                else None
            ),
            "promptSha256": config.prompt.sha256,
            "referenceCommits": {
                "glm": config.reference_runs.glm.commit_sha256,
                "luna": config.reference_runs.luna.commit_sha256,
            },
            "implementationSha256": stage_hashes["inventory"],
            "inventoryImplementationSha256": stage_hashes["inventoryContract"],
            "hybridBatchImplementationSha256": stage_hashes["hybridBatch"],
            "hybridCompilerImplementationSha256": stage_hashes["hybridRuntime"],
            "containerSemanticsImplementationSha256": stage_hashes["containerSemantics"],
            "providerRuntimeImplementationSha256": stage_hashes["providerRuntime"],
            "rewriteContractImplementationSha256": stage_hashes["rewriteContract"],
            "linguisticPlanSha256": "a" * 64,
            "sourceCorpusSha256": "b" * 64,
            "targetSha256": "c" * 64,
            "liveDocumentIds": [row.documentId for row in rows],
            "runtime": {
                "maxConcurrentDocuments": config.workflow.max_concurrent_documents,
                "maxProviderRouteRounds": config.workflow.max_provider_route_rounds,
                "model": config.provider.model,
                "openaiVersion": "test",
                "outputMode": config.workflow.output_mode,
                "providerOrder": list(config.provider.provider_order),
                "pydanticAiVersion": "test",
            },
        }
        transaction_sha256 = sha256_bytes(canonical_json_bytes(transaction))
    elif certification_rows:
        import document_ocr.synthesis.raw_text_pipeline as pipeline_module

        stage_hashes = pipeline_module._stage_implementation_hashes()
        transaction = {
            "schemaVersion": 2,
            "runId": config.run.run_id,
            "configSha256": sha256_bytes(config_payload),
            "resolvedConfigSha256": sha256_bytes(
                canonical_json_bytes(config.model_dump(mode="json"))
            ),
            "inputRunCommitSha256": config.input_run.commit_sha256,
            "inputRunTransactionSha256": config.input_run.transaction_sha256,
            "benchmarkPlan": None,
            "inputCaseContractFilename": config.input_case_contract_filename,
            "promptSha256": config.prompt.sha256,
            "implementationSha256": stage_hashes["certification"],
            "dependencyImplementationSha256": {
                "containerSemantics": stage_hashes["containerSemantics"],
                "providerRuntime": stage_hashes["providerRuntime"],
                "hybridRuntime": stage_hashes["hybridRuntime"],
                "inventory": stage_hashes["inventoryContract"],
                "inventoryRunner": stage_hashes["inventory"],
                "rewriteContract": stage_hashes["rewriteContract"],
                "rewriteRuntime": stage_hashes["rewriteRuntime"],
                "transportCapacity": stage_hashes["transportCapacity"],
            },
            "documentIds": list(config.case_ids),
            "runtime": {
                "pydanticAiVersion": "test",
                "openaiVersion": "test",
                "model": config.provider.model,
                "providerOrder": list(config.provider.provider_order or ()),
                "maxConcurrentDocuments": config.workflow.max_concurrent_documents,
                "semanticAuditPasses": config.workflow.semantic_audit_passes,
                "auditContractVersion": config.audit_contract_version,
                "certificationMode": "production",
                "reasoningEfforts": ["low", "high"],
            },
        }
        transaction_sha256 = sha256_bytes(canonical_json_bytes(transaction))
    elif correction_rows:
        import document_ocr.synthesis.raw_text_pipeline as pipeline_module

        stage_hashes = pipeline_module._stage_implementation_hashes()
        certification_root = project_root / config.certification_run.path
        certification_config = load_synthesis_raw_text_certification_config(
            certification_root / "config.yaml"
        )
        transaction = {
            "schemaVersion": 2,
            "runId": config.run.run_id,
            "configSha256": sha256_bytes(config_payload),
            "resolvedConfigSha256": sha256_bytes(
                canonical_json_bytes(config.model_dump(mode="json"))
            ),
            "certificationRunCommitSha256": config.certification_run.commit_sha256,
            "certificationRunTransactionSha256": (config.certification_run.transaction_sha256),
            "certificationConfigSha256": sha256_bytes(
                (certification_root / "config.yaml").read_bytes()
            ),
            "certificationResolvedConfigSha256": sha256_bytes(
                canonical_json_bytes(certification_config.model_dump(mode="json"))
            ),
            "certificationTransactionArtifactSha256": sha256_bytes(
                (certification_root / "provenance/transaction.json").read_bytes()
            ),
            "certificationSummarySha256": sha256_bytes(
                (certification_root / "summary.json").read_bytes()
            ),
            "certificationAuditContractVersion": 2,
            "certificationMode": "production",
            "promptSha256": config.prompt.sha256,
            "implementationSha256": stage_hashes["correction"],
            "dependencyImplementationSha256": {
                "certification": stage_hashes["certification"],
                "certificationArtifacts": stage_hashes["certificationArtifacts"],
                "providerRuntime": stage_hashes["providerRuntime"],
                "hybridRuntime": stage_hashes["hybridRuntime"],
                "inventory": stage_hashes["inventoryContract"],
                "inventoryRunner": stage_hashes["inventory"],
                "rewriteContract": stage_hashes["rewriteContract"],
                "rewriteRuntime": stage_hashes["rewriteRuntime"],
            },
            "documentIds": list(config.case_ids),
            "runtime": {
                "pydanticAiVersion": "test",
                "openaiVersion": "test",
                "model": config.provider.model,
                "providerOrder": list(config.provider.provider_order or ()),
                "maxConcurrentDocuments": config.workflow.max_concurrent_documents,
                "maxSuccessfulModelResponsesPerDocument": (
                    config.workflow.max_successful_model_responses_per_document
                ),
            },
        }
        transaction_sha256 = sha256_bytes(canonical_json_bytes(transaction))
    else:
        transaction = None
        transaction_sha256 = sha256_bytes(canonical_json_bytes(config.model_dump(mode="json")))
    staged = StagedArtifactRun(
        output_parent=project_root / config.run.output_dir,
        run_name=config.run.run_id,
        transaction_sha256=transaction_sha256,
    )
    staged.publish_bytes("config.yaml", config_payload)
    if certification_rows:
        staged.publish_bytes(
            "prompts/read-only-auditor.md",
            (project_root / config.prompt.path).read_bytes(),
        )
    if transaction is not None:
        staged.publish_json("provenance/transaction.json", transaction)
    staged.publish_json("summary.json", summary)
    staged.publish_bytes(
        "generation/results.jsonl",
        b"".join(canonical_json_bytes(row.model_dump(mode="json")) + b"\n" for row in rows),
    )
    for document_id, (source, input_candidate, final) in (case_texts or {}).items():
        prefix = f"cases/{document_id}"
        staged.publish_bytes(f"{prefix}/source.txt", source.encode())
        staged.publish_bytes(f"{prefix}/input-candidate.txt", input_candidate.encode())
        staged.publish_bytes(f"{prefix}/final.txt", final.encode())
        if certification_rows:
            stages, source_label, target_label, fake_contract = certification_material[document_id]
        elif correction_rows:
            certification_case = (
                project_root / config.certification_run.path / "cases" / document_id
            )
            source_label = json.loads((certification_case / "source-label.json").read_text())
            target_label = json.loads((certification_case / "target-label.json").read_text())
            fake_contract = _refine_carrier_occurrence_contract(
                source=source,
                contract=json.loads((certification_case / "source-contract.json").read_text()),
            )
        else:
            source_label = {
                "artifact": "source-label.json",
                "documentId": document_id,
                "schemaVersion": "5.0.0-experimental",
                "documentPatch": {},
            }
            target_label = {
                "artifact": "target-label.json",
                "documentId": document_id,
                "schemaVersion": "5.0.0-experimental",
                "documentPatch": {},
            }
            fake_contract = _certification_contract(
                document_id=document_id,
                source_label=source_label,
                target_label=target_label,
            )
        staged.publish_json(f"{prefix}/contract.json", fake_contract)
        if certification_rows or correction_rows:
            staged.publish_json(f"{prefix}/source-contract.json", fake_contract)
        if certification_rows or correction_rows:
            staged.publish_json(f"{prefix}/source-label.json", source_label)
            staged.publish_json(f"{prefix}/target-label.json", target_label)
        if certification_rows:
            staged.publish_json(
                f"{prefix}/audit-plan.json",
                _audit_plan(fake_contract, source=source, current=final),
            )
            staged.publish_json(
                f"{prefix}/stages.json",
                [row.model_dump(mode="json") for row in stages],
            )
        if inventory_rows:
            staged.publish_json(f"{prefix}/source-label.json", source_label)
            staged.publish_json(f"{prefix}/target-label.json", target_label)
            for filename in (
                "inventory.json",
                "deterministic-edits.json",
                "model-slots.json",
                "compound-slots.json",
                "editor-payload.json",
            ):
                staged.publish_json(
                    f"{prefix}/{filename}",
                    {"artifact": filename, "documentId": document_id},
                )
    for row in rows:
        staged.publish_json(
            f"cases/{row.documentId}/result.json",
            row.model_dump(mode="json"),
        )
    staged.commit(
        expected_artifacts=_artifact_inventory(staged.stage_root),
        metadata={"schemaVersion": 1, "documents": len(rows)},
    )
    return {
        **summary,
        "artifactRoot": str(staged.final_root),
    }


def _inventory_result(document_id: str) -> InventoryProbeCaseResult:
    return InventoryProbeCaseResult(
        documentId=document_id,
        status="training_ready",
        reason="all inventory gates passed",
        sourceLines=1,
        modelLines=1,
        compilerWorkItems=1,
        inventoryCandidates=1,
        deterministicInventoryEdits=0,
        hostRewriteAuditPassed=True,
        legacyResidualCandidates=0,
        outputTextSha256=sha256_bytes(_initial_candidate(document_id).encode()),
        usage=_empty_usage(),
    )


def _source_text(document_id: str) -> str:
    return f"source:{document_id}"


def _initial_candidate(document_id: str) -> str:
    return f"inventory:{document_id}"


def _corrected_candidate(document_id: str) -> str:
    return f"corrected:{document_id}"


def _certification_result(
    document_id: str,
    *,
    status: Literal["certified", "needs_review", "call_failed"],
    source: str,
    candidate: str,
) -> CertificationCaseResult:
    actionable = status == "needs_review"
    return CertificationCaseResult(
        documentId=document_id,
        status=status,
        reason=f"test {status}",
        auditPasses=1 if status != "call_failed" else 0,
        semanticFindings=1 if actionable else 0,
        candidateImmutable=True,
        hostAudit=CertificationHostAudit(
            passed=True,
            findings=(),
            sourceLines=1,
            outputLines=1,
            targetLiteralRequirements=0,
            targetOccurrenceRequirements=0,
        ),
        sourceTextSha256=sha256_bytes(source.encode()),
        inputCandidateSha256=sha256_bytes(candidate.encode()),
        finalTextSha256=sha256_bytes(candidate.encode()),
        usage=_empty_usage(),
    )


def _correction_result(
    document_id: str, *, source: str, current: str, final: str
) -> CorrectionCaseResult:
    return CorrectionCaseResult(
        documentId=document_id,
        status="correction_candidate",
        reason="local gates passed",
        citedLines=1,
        changedLines=1,
        sourceFindings=1,
        locallyResolvedFindings=1,
        requiresRecertification=True,
        sourceTextSha256=sha256_bytes(source.encode()),
        inputCandidateSha256=sha256_bytes(current.encode()),
        finalTextSha256=sha256_bytes(final.encode()),
        usage=_empty_usage(),
    )


def test_pipeline_shards_retries_corrects_recertifies_and_publishes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import document_ocr.synthesis.raw_text_pipeline as pipeline_module

    config_path, config = _write_pipeline_fixture(tmp_path)
    certification_attempts: dict[str, int] = {}
    certification_configs = []
    correction_configs = []
    publication_configs = []

    monkeypatch.setattr(
        pipeline_module,
        "preflight_raw_text_pipeline",
        lambda **_kwargs: {"status": "preflight_complete"},
    )

    def run_inventory(*, project_root: Path, config: Any, **_kwargs: Any) -> dict[str, Any]:
        rows = tuple(_inventory_result(document_id) for document_id in _DOCUMENT_IDS)
        return _commit_fake_run(
            project_root=project_root,
            config=config,
            rows=rows,
            summary={
                "status": "complete",
                "documents": 3,
                "trainingReadyDocuments": 3,
            },
            case_texts={
                document_id: (
                    _source_text(document_id),
                    _initial_candidate(document_id),
                    _initial_candidate(document_id),
                )
                for document_id in _DOCUMENT_IDS
            },
        )

    def run_certification(*, project_root: Path, config: Any, **_kwargs: Any) -> dict[str, Any]:
        certification_configs.append(config)
        results = []
        case_texts: dict[str, tuple[str, str, str]] = {}
        input_root = project_root / config.input_run.path
        for document_id in config.case_ids:
            certification_attempts[document_id] = certification_attempts.get(document_id, 0) + 1
            status: Literal["certified", "needs_review", "call_failed"]
            if document_id == _DOCUMENT_IDS[1] and certification_attempts[document_id] == 1:
                status = "needs_review"
            elif document_id == _DOCUMENT_IDS[2] and certification_attempts[document_id] == 1:
                status = "call_failed"
            else:
                status = "certified"
            source = (input_root / "cases" / document_id / "source.txt").read_text()
            candidate = (input_root / "cases" / document_id / "final.txt").read_text()
            results.append(
                _certification_result(
                    document_id,
                    status=status,
                    source=source,
                    candidate=candidate,
                )
            )
            case_texts[document_id] = (source, candidate, candidate)
        return _commit_fake_run(
            project_root=project_root,
            config=config,
            rows=tuple(results),
            summary={"status": "complete", "documents": len(results)},
            case_texts=case_texts,
        )

    def run_correction(*, project_root: Path, config: Any, **_kwargs: Any) -> dict[str, Any]:
        correction_configs.append(config)
        input_root = project_root / config.certification_run.path
        case_texts: dict[str, tuple[str, str, str]] = {}
        rows = []
        for document_id in config.case_ids:
            source = (input_root / "cases" / document_id / "source.txt").read_text()
            current = (input_root / "cases" / document_id / "final.txt").read_text()
            final = _corrected_candidate(document_id)
            rows.append(
                _correction_result(
                    document_id,
                    source=source,
                    current=current,
                    final=final,
                )
            )
            case_texts[document_id] = (source, current, final)
        return _commit_fake_run(
            project_root=project_root,
            config=config,
            rows=tuple(rows),
            summary={"status": "complete", "documents": len(rows)},
            case_texts=case_texts,
        )

    def run_publication(*, project_root: Path, config: Any, **_kwargs: Any) -> dict[str, Any]:
        publication_configs.append(config)
        result = _commit_fake_run(
            project_root=project_root,
            config=config,
            rows=(),
            summary={
                "status": "complete",
                "documents": 3,
                "certifiedDocuments": 3,
                "trainingRecordsPublished": True,
            },
        )
        return result

    monkeypatch.setattr(pipeline_module, "run_raw_text_inventory_batch", run_inventory)
    monkeypatch.setattr(pipeline_module, "run_raw_text_certification", run_certification)
    monkeypatch.setattr(pipeline_module, "run_raw_text_certified_correction", run_correction)
    monkeypatch.setattr(pipeline_module, "run_raw_text_certified_publication", run_publication)

    result = run_raw_text_pipeline(
        project_root=tmp_path,
        config_path=config_path,
        config=config,
    )

    assert result["status"] == "complete"
    assert result["certifiedDocuments"] == 3
    assert result["trainingRecordsPublished"] is True
    assert result["correctionRounds"] == 1
    assert len(correction_configs) == 1
    assert correction_configs[0].case_ids == (_DOCUMENT_IDS[1],)
    corrected_certification = next(
        row for row in certification_configs if row.case_ids == (_DOCUMENT_IDS[1],)
    )
    assert corrected_certification.input_case_contract_filename == "source-contract.json"
    assert certification_attempts[_DOCUMENT_IDS[2]] == 2
    assert len(publication_configs) == 1
    assert (
        sum(source.certified_documents for source in publication_configs[0].certification_sources)
        == 3
    )
    lineage = (Path(str(result["artifactRoot"])) / "lineage/cases.jsonl").read_text()
    assert '"stage":"correction"' in lineage
    assert '"stage":"certification"' in lineage
    generated_root = Path(str(result["artifactRoot"])) / "generated-configs"
    generated_paths = tuple(sorted(generated_root.iterdir()))
    assert generated_paths
    assert all(path.suffix == ".yaml" for path in generated_paths)
    for path in generated_paths:
        if "-correct-" in path.name:
            load_synthesis_raw_text_certified_correction_config(path)
        elif path.name.endswith("-publication.yaml"):
            load_synthesis_raw_text_certified_publication_config(path)
        else:
            load_synthesis_raw_text_certification_config(path)


def test_pipeline_commits_blocked_lineage_and_never_publishes_partial_cohort(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import document_ocr.synthesis.raw_text_pipeline as pipeline_module

    config_path, config = _write_pipeline_fixture(tmp_path)
    publication_called = False

    monkeypatch.setattr(
        pipeline_module,
        "preflight_raw_text_pipeline",
        lambda **_kwargs: {"status": "preflight_complete"},
    )

    def run_inventory(*, project_root: Path, config: Any, **_kwargs: Any) -> dict[str, Any]:
        rows = tuple(_inventory_result(document_id) for document_id in _DOCUMENT_IDS)
        return _commit_fake_run(
            project_root=project_root,
            config=config,
            rows=rows,
            summary={"status": "complete", "documents": 3},
            case_texts={
                document_id: (
                    _source_text(document_id),
                    _initial_candidate(document_id),
                    _initial_candidate(document_id),
                )
                for document_id in _DOCUMENT_IDS
            },
        )

    def run_certification(*, project_root: Path, config: Any, **_kwargs: Any) -> dict[str, Any]:
        input_root = project_root / config.input_run.path
        case_texts: dict[str, tuple[str, str, str]] = {}
        rows = []
        for document_id in config.case_ids:
            source = (input_root / "cases" / document_id / "source.txt").read_text()
            candidate = (input_root / "cases" / document_id / "final.txt").read_text()
            rows.append(
                _certification_result(
                    document_id,
                    status="call_failed",
                    source=source,
                    candidate=candidate,
                )
            )
            case_texts[document_id] = (source, candidate, candidate)
        return _commit_fake_run(
            project_root=project_root,
            config=config,
            rows=tuple(rows),
            summary={"status": "complete", "documents": len(rows)},
            case_texts=case_texts,
        )

    def run_publication(**_kwargs: Any) -> dict[str, Any]:
        nonlocal publication_called
        publication_called = True
        raise AssertionError("partial cohorts must not reach publication")

    monkeypatch.setattr(pipeline_module, "run_raw_text_inventory_batch", run_inventory)
    monkeypatch.setattr(pipeline_module, "run_raw_text_certification", run_certification)
    monkeypatch.setattr(pipeline_module, "run_raw_text_certified_publication", run_publication)

    with pytest.raises(RawTextPipelineBlockedError, match="without complete publication"):
        run_raw_text_pipeline(
            project_root=tmp_path,
            config_path=config_path,
            config=config,
        )

    assert publication_called is False
    summary = json.loads((tmp_path / "runs/pipeline-test/summary.json").read_text())
    assert summary["status"] == "blocked"
    assert summary["certifiedDocuments"] == 0
    assert summary["blockedDocuments"] == 3
    assert summary["trainingRecordsPublished"] is False


def test_pipeline_retries_only_unready_inventory_cases_and_records_complete_lineage(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import document_ocr.synthesis.raw_text_pipeline as pipeline_module

    config_path, config = _write_pipeline_fixture(tmp_path)
    inventory_scopes: list[tuple[str, ...]] = []
    certification_called = False

    monkeypatch.setattr(
        pipeline_module,
        "preflight_raw_text_pipeline",
        lambda **_kwargs: {"status": "preflight_complete"},
    )
    monkeypatch.setattr(
        pipeline_module,
        "preflight_raw_text_inventory_batch",
        lambda **_kwargs: {"status": "preflight_complete"},
    )

    def run_inventory(*, project_root: Path, config: Any, **_kwargs: Any) -> dict[str, Any]:
        document_ids = (
            tuple(row.document_id for row in config.cases)
            if config.selection == "explicit_pinned_document_ids"
            else _DOCUMENT_IDS
        )
        inventory_scopes.append(document_ids)
        rows = tuple(
            _inventory_result(document_id).model_copy(
                update={
                    "status": (
                        "needs_review" if document_id == _DOCUMENT_IDS[2] else "training_ready"
                    ),
                    "reason": "test inventory outcome",
                }
            )
            for document_id in document_ids
        )
        return _commit_fake_run(
            project_root=project_root,
            config=config,
            rows=rows,
            summary={"status": "complete", "documents": len(rows)},
            case_texts={
                document_id: (
                    _source_text(document_id),
                    _initial_candidate(document_id),
                    _initial_candidate(document_id),
                )
                for document_id in document_ids
            },
        )

    def run_certification(**_kwargs: Any) -> dict[str, Any]:
        nonlocal certification_called
        certification_called = True
        raise AssertionError("an incomplete inventory cohort must not be certified")

    monkeypatch.setattr(pipeline_module, "run_raw_text_inventory_batch", run_inventory)
    monkeypatch.setattr(pipeline_module, "run_raw_text_certification", run_certification)

    with pytest.raises(RawTextPipelineBlockedError, match="without complete publication"):
        run_raw_text_pipeline(
            project_root=tmp_path,
            config_path=config_path,
            config=config,
        )

    assert inventory_scopes == [
        _DOCUMENT_IDS,
        (_DOCUMENT_IDS[2],),
        (_DOCUMENT_IDS[2],),
    ]
    assert certification_called is False
    root = tmp_path / "runs/pipeline-test"
    summary = json.loads((root / "summary.json").read_text())
    assert summary["inventoryReadyDocuments"] == 2
    assert summary["primaryBlockingDocuments"] == 1
    assert summary["blockedDocuments"] == 3
    lineage = tuple(
        json.loads(line) for line in (root / "lineage/cases.jsonl").read_text().splitlines()
    )
    assert tuple(row["documentId"] for row in lineage) == _DOCUMENT_IDS
    assert len(lineage[2]["history"]) == 3
    assert "inventory did not produce" in lineage[2]["blockingReason"]
    retry_configs = tuple(sorted((root / "generated-configs").glob("*inventory*.yaml")))
    assert len(retry_configs) == 2
    assert all(
        load_synthesis_raw_text_inventory_batch_config(path).workflow.documents == 1
        for path in retry_configs
    )


def test_pipeline_inventory_resume_restores_ready_cases_and_retries_only_missing_case(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import document_ocr.synthesis.raw_text_pipeline as pipeline_module

    real_pipeline_preflight = pipeline_module.preflight_raw_text_pipeline
    config_path, config = _write_pipeline_fixture(tmp_path)
    initial_inventory_scopes: list[tuple[str, ...]] = []

    monkeypatch.setattr(
        pipeline_module,
        "preflight_raw_text_pipeline",
        lambda **_kwargs: {"status": "preflight_complete"},
    )
    monkeypatch.setattr(
        pipeline_module,
        "preflight_raw_text_inventory_batch",
        lambda *, config, **_kwargs: {
            "status": "preflight_complete",
            "compiledDocuments": config.workflow.documents,
        },
    )

    def run_blocked_inventory(*, project_root: Path, config: Any, **_kwargs: Any) -> dict[str, Any]:
        document_ids = (
            tuple(row.document_id for row in config.cases)
            if config.selection == "explicit_pinned_document_ids"
            else _DOCUMENT_IDS
        )
        initial_inventory_scopes.append(document_ids)
        rows = tuple(
            _inventory_result(document_id).model_copy(
                update={
                    "status": (
                        "call_failed" if document_id == _DOCUMENT_IDS[2] else "training_ready"
                    ),
                    "reason": "test inventory outcome",
                }
            )
            for document_id in document_ids
        )
        return _commit_fake_run(
            project_root=project_root,
            config=config,
            rows=rows,
            summary={"status": "complete", "documents": len(rows)},
            case_texts={
                document_id: (
                    _source_text(document_id),
                    _initial_candidate(document_id),
                    _initial_candidate(document_id),
                )
                for document_id in document_ids
            },
        )

    monkeypatch.setattr(pipeline_module, "run_raw_text_inventory_batch", run_blocked_inventory)
    with pytest.raises(RawTextPipelineBlockedError, match="without complete publication"):
        run_raw_text_pipeline(
            project_root=tmp_path,
            config_path=config_path,
            config=config,
        )
    assert initial_inventory_scopes == [
        _DOCUMENT_IDS,
        (_DOCUMENT_IDS[2],),
        (_DOCUMENT_IDS[2],),
    ]

    parent_root = tmp_path / "runs/pipeline-test"
    parent_commit = json.loads((parent_root / "_COMMIT.json").read_text())
    inventory_value = load_synthesis_raw_text_inventory_batch_config(
        tmp_path / "inventory.yaml"
    ).model_dump(mode="python")
    inventory_value["run"] = {
        "run_id": "inventory-resume-test",
        "output_dir": "runs",
    }
    inventory_value["selection"] = "explicit_pinned_document_ids"
    inventory_value["cases"] = [{"document_id": _DOCUMENT_IDS[2]}]
    inventory_value["workflow"]["documents"] = 1
    inventory_value["workflow"]["max_concurrent_documents"] = 1
    resume_inventory = SynthesisRawTextInventoryBatchConfig.model_validate(
        inventory_value, strict=True
    )
    resume_inventory_path = tmp_path / "inventory-resume.yaml"
    resume_inventory_path.write_text(
        yaml.safe_dump(resume_inventory.model_dump(mode="python"), sort_keys=False),
        encoding="utf-8",
    )

    pipeline_value = config.model_dump(mode="python")
    pipeline_value["run"] = {"run_id": "pipeline-resumed", "output_dir": "runs"}
    pipeline_value["inventory_config"] = {
        "path": "inventory-resume.yaml",
        "sha256": sha256_bytes(resume_inventory_path.read_bytes()),
    }
    pipeline_value["inventory_resume_run"] = {
        "path": "runs/pipeline-test",
        "commit_sha256": sha256_bytes((parent_root / "_COMMIT.json").read_bytes()),
        "transaction_sha256": parent_commit["transaction_sha256"],
    }
    resume_config = SynthesisRawTextPipelineConfig.model_validate(pipeline_value, strict=True)
    resume_config_path = tmp_path / "pipeline-resumed.yaml"
    resume_config_path.write_text(
        yaml.safe_dump(resume_config.model_dump(mode="python"), sort_keys=False),
        encoding="utf-8",
    )
    monkeypatch.setattr(
        pipeline_module,
        "preflight_raw_text_pipeline",
        real_pipeline_preflight,
    )
    preflight = real_pipeline_preflight(
        project_root=tmp_path,
        config_path=resume_config_path,
        config=resume_config,
    )
    assert preflight["documents"] == 3
    assert preflight["resumedInventoryDocuments"] == 2
    assert preflight["compiledInventoryDocuments"] == 1
    assert preflight["providerSecretsLoaded"] is False
    assert preflight["providerRequests"] == 0

    resumed_inventory_scopes: list[tuple[str, ...]] = []
    certification_scopes: list[tuple[str, ...]] = []
    publication_configs = []

    def run_resumed_inventory(*, project_root: Path, config: Any, **_kwargs: Any) -> dict[str, Any]:
        document_ids = tuple(row.document_id for row in config.cases)
        resumed_inventory_scopes.append(document_ids)
        rows = tuple(_inventory_result(document_id) for document_id in document_ids)
        return _commit_fake_run(
            project_root=project_root,
            config=config,
            rows=rows,
            summary={"status": "complete", "documents": len(rows)},
            case_texts={
                document_id: (
                    _source_text(document_id),
                    _initial_candidate(document_id),
                    _initial_candidate(document_id),
                )
                for document_id in document_ids
            },
        )

    def run_certification(*, project_root: Path, config: Any, **_kwargs: Any) -> dict[str, Any]:
        certification_scopes.append(config.case_ids)
        input_root = project_root / config.input_run.path
        rows = []
        case_texts: dict[str, tuple[str, str, str]] = {}
        for document_id in config.case_ids:
            source = (input_root / "cases" / document_id / "source.txt").read_text()
            candidate = (input_root / "cases" / document_id / "final.txt").read_text()
            rows.append(
                _certification_result(
                    document_id,
                    status="certified",
                    source=source,
                    candidate=candidate,
                )
            )
            case_texts[document_id] = (source, candidate, candidate)
        return _commit_fake_run(
            project_root=project_root,
            config=config,
            rows=tuple(rows),
            summary={"status": "complete", "documents": len(rows)},
            case_texts=case_texts,
        )

    def run_publication(*, project_root: Path, config: Any, **_kwargs: Any) -> dict[str, Any]:
        publication_configs.append(config)
        return _commit_fake_run(
            project_root=project_root,
            config=config,
            rows=(),
            summary={
                "status": "complete",
                "documents": 3,
                "certifiedDocuments": 3,
                "trainingRecordsPublished": True,
            },
        )

    monkeypatch.setattr(pipeline_module, "run_raw_text_inventory_batch", run_resumed_inventory)
    monkeypatch.setattr(pipeline_module, "run_raw_text_certification", run_certification)
    monkeypatch.setattr(pipeline_module, "run_raw_text_certified_publication", run_publication)

    result = run_raw_text_pipeline(
        project_root=tmp_path,
        config_path=resume_config_path,
        config=resume_config,
    )

    assert result["status"] == "complete"
    assert result["inventoryReadyDocuments"] == 3
    assert result["certifiedDocuments"] == 3
    assert resumed_inventory_scopes == [(_DOCUMENT_IDS[2],)]
    assert sorted(certification_scopes) == [
        (_DOCUMENT_IDS[0], _DOCUMENT_IDS[1]),
        (_DOCUMENT_IDS[2],),
    ]
    assert len(publication_configs) == 1
    assert (
        sum(source.certified_documents for source in publication_configs[0].certification_sources)
        == 3
    )
    resume_root = Path(str(result["artifactRoot"]))
    resume_input = json.loads((resume_root / "inputs/inventory-resume-run.json").read_text())
    assert resume_input["path"] == "runs/pipeline-test"
    lineage = tuple(
        json.loads(line) for line in (resume_root / "lineage/cases.jsonl").read_text().splitlines()
    )
    assert [row["inventoryRound"] for row in lineage] == [1, 1, 4]

    stage_hashes = pipeline_module._stage_implementation_hashes()
    monkeypatch.setattr(
        pipeline_module,
        "_stage_implementation_hashes",
        lambda: {**stage_hashes, "inventory": "0" * 64},
    )
    with pytest.raises(ValueError, match="implementation differs for: inventory"):
        real_pipeline_preflight(
            project_root=tmp_path,
            config_path=resume_config_path,
            config=resume_config,
        )

    monkeypatch.setattr(
        pipeline_module,
        "_stage_implementation_hashes",
        lambda: stage_hashes,
    )
    child_root = tmp_path / lineage[0]["history"][0]["run"]["path"]
    (child_root / "cases" / _DOCUMENT_IDS[0] / "result.json").unlink()
    with pytest.raises(RuntimeError, match="committed artifact inventory differs"):
        real_pipeline_preflight(
            project_root=tmp_path,
            config_path=resume_config_path,
            config=resume_config,
        )


def test_inventory_resume_explicitly_rejects_compiler_blocked_children(
    tmp_path: Path,
) -> None:
    import document_ocr.synthesis.raw_text_pipeline as pipeline_module

    _pipeline_path, _pipeline = _write_pipeline_fixture(tmp_path)
    value = load_synthesis_raw_text_inventory_batch_config(tmp_path / "inventory.yaml").model_dump(
        mode="python"
    )
    value["run"] = {"run_id": "inventory-compiler-blocked", "output_dir": "runs"}
    value["cases"] = [{"document_id": _DOCUMENT_IDS[0]}]
    value["workflow"]["documents"] = 1
    value["workflow"]["max_concurrent_documents"] = 1
    inventory = SynthesisRawTextInventoryBatchConfig.model_validate(value, strict=True)
    result = _inventory_result(_DOCUMENT_IDS[0]).model_copy(
        update={
            "status": "compiler_blocked",
            "reason": "compiler rejected the source contract",
            "hostRewriteAuditPassed": False,
            "fullDocumentAudit": None,
            "compilerErrorType": "ValueError",
            "compilerErrorMessage": "unowned source fact",
        }
    )
    receipt = _commit_fake_run(
        project_root=tmp_path,
        config=inventory,
        rows=(result,),
        summary={"status": "complete", "documents": 1},
        case_texts={
            _DOCUMENT_IDS[0]: (
                _source_text(_DOCUMENT_IDS[0]),
                _initial_candidate(_DOCUMENT_IDS[0]),
                _source_text(_DOCUMENT_IDS[0]),
            )
        },
    )
    reference = pipeline_module._committed_run_reference(
        project_root=tmp_path,
        run_id=inventory.run.run_id,
        output_dir=inventory.run.output_dir,
    )

    with pytest.raises(ValueError, match="cannot reuse compiler-blocked cases"):
        pipeline_module._validate_inventory_child_run(
            project_root=tmp_path,
            reference=reference,
            document_ids=(_DOCUMENT_IDS[0],),
            inventory_contract_sha256=pipeline_module._inventory_contract_sha256(inventory),
            transaction_contract_sha256=None,
            stage_hashes=pipeline_module._stage_implementation_hashes(),
        )

    assert receipt["status"] == "complete"


def test_pipeline_resume_replays_exact_state_and_chains_only_unresolved_work(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import document_ocr.synthesis.raw_text_pipeline as pipeline_module

    real_pipeline_preflight = pipeline_module.preflight_raw_text_pipeline
    config_path, config = _write_pipeline_fixture(tmp_path)
    phase = "parent"
    inventory_calls = 0
    certification_scopes: list[tuple[str, tuple[str, ...]]] = []
    correction_scopes: list[tuple[str, tuple[str, ...]]] = []
    final_resume_correction_calls = 0
    publication_configs = []

    monkeypatch.setattr(
        pipeline_module,
        "preflight_raw_text_pipeline",
        lambda **_kwargs: {"status": "preflight_complete"},
    )

    def run_inventory(*, project_root: Path, config: Any, **_kwargs: Any) -> dict[str, Any]:
        nonlocal inventory_calls
        inventory_calls += 1
        rows = tuple(_inventory_result(document_id) for document_id in _DOCUMENT_IDS)
        return _commit_fake_run(
            project_root=project_root,
            config=config,
            rows=rows,
            summary={"status": "complete", "documents": 3},
            case_texts={
                document_id: (
                    _source_text(document_id),
                    _initial_candidate(document_id),
                    _initial_candidate(document_id),
                )
                for document_id in _DOCUMENT_IDS
            },
        )

    def run_certification(*, project_root: Path, config: Any, **_kwargs: Any) -> dict[str, Any]:
        certification_scopes.append((phase, config.case_ids))
        input_root = project_root / config.input_run.path
        results = []
        case_texts: dict[str, tuple[str, str, str]] = {}
        for document_id in config.case_ids:
            source = (input_root / "cases" / document_id / "source.txt").read_text()
            candidate = (input_root / "cases" / document_id / "final.txt").read_text()
            status: Literal["certified", "needs_review", "call_failed"]
            if (
                phase == "parent"
                and document_id != _DOCUMENT_IDS[0]
                and candidate.startswith("inventory:")
            ):
                status = "needs_review"
            else:
                status = "certified"
            results.append(
                _certification_result(
                    document_id,
                    status=status,
                    source=source,
                    candidate=candidate,
                )
            )
            case_texts[document_id] = (source, candidate, candidate)
        return _commit_fake_run(
            project_root=project_root,
            config=config,
            rows=tuple(results),
            summary={"status": "complete", "documents": len(results)},
            case_texts=case_texts,
        )

    def run_correction(*, project_root: Path, config: Any, **_kwargs: Any) -> dict[str, Any]:
        nonlocal final_resume_correction_calls
        correction_scopes.append((phase, config.case_ids))
        input_root = project_root / config.certification_run.path
        rows = []
        case_texts: dict[str, tuple[str, str, str]] = {}
        for document_id in config.case_ids:
            source = (input_root / "cases" / document_id / "source.txt").read_text()
            current = (input_root / "cases" / document_id / "final.txt").read_text()
            succeeds = phase == "parent" and document_id == _DOCUMENT_IDS[1]
            if phase == "resume-final":
                final_resume_correction_calls += 1
                succeeds = final_resume_correction_calls == 2
            final = _corrected_candidate(document_id) if succeeds else current
            result = _correction_result(
                document_id,
                source=source,
                current=current,
                final=final,
            )
            if not succeeds:
                result = result.model_copy(
                    update={
                        "status": "call_failed",
                        "reason": "test transient route exhaustion",
                        "changedLines": 0,
                        "locallyResolvedFindings": 0,
                        "requiresRecertification": False,
                    }
                )
            rows.append(result)
            case_texts[document_id] = (source, current, final)
        return _commit_fake_run(
            project_root=project_root,
            config=config,
            rows=tuple(rows),
            summary={"status": "complete", "documents": len(rows)},
            case_texts=case_texts,
        )

    def run_publication(*, project_root: Path, config: Any, **_kwargs: Any) -> dict[str, Any]:
        publication_configs.append(config)
        return _commit_fake_run(
            project_root=project_root,
            config=config,
            rows=(),
            summary={
                "status": "complete",
                "documents": 3,
                "certifiedDocuments": 3,
                "trainingRecordsPublished": True,
            },
        )

    monkeypatch.setattr(pipeline_module, "run_raw_text_inventory_batch", run_inventory)
    monkeypatch.setattr(pipeline_module, "run_raw_text_certification", run_certification)
    monkeypatch.setattr(pipeline_module, "run_raw_text_certified_correction", run_correction)
    monkeypatch.setattr(pipeline_module, "run_raw_text_certified_publication", run_publication)

    with pytest.raises(RawTextPipelineBlockedError, match="without complete publication"):
        run_raw_text_pipeline(
            project_root=tmp_path,
            config_path=config_path,
            config=config,
        )
    assert inventory_calls == 1
    parent_root = tmp_path / "runs/pipeline-test"
    parent_summary = json.loads((parent_root / "summary.json").read_text())
    assert parent_summary["certifiedDocuments"] == 2
    assert parent_summary["primaryBlockingDocuments"] == 1
    assert parent_summary["blockedDocuments"] == 1
    parent_lineage = {
        row["documentId"]: row
        for row in (
            json.loads(line)
            for line in (parent_root / "lineage/cases.jsonl").read_text().splitlines()
        )
    }
    assert parent_lineage[_DOCUMENT_IDS[1]]["status"] == "certified"
    assert parent_lineage[_DOCUMENT_IDS[2]]["status"] == "blocked"
    assert "configured attempt limit" in parent_lineage[_DOCUMENT_IDS[2]]["blockingReason"]

    def resume_config(
        *, parent_root: Path, run_id: str, correction_attempts: int
    ) -> tuple[Path, SynthesisRawTextPipelineConfig]:
        parent_receipt = json.loads((parent_root / "_COMMIT.json").read_text())
        value = config.model_dump(mode="python")
        value["run"] = {"run_id": run_id, "output_dir": "runs"}
        value["inventory_resume_run"] = None
        value["pipeline_resume_run"] = {
            "path": parent_root.relative_to(tmp_path).as_posix(),
            "commit_sha256": sha256_bytes((parent_root / "_COMMIT.json").read_bytes()),
            "transaction_sha256": parent_receipt["transaction_sha256"],
        }
        value["correction"]["max_attempts_per_round"] = correction_attempts
        resumed = SynthesisRawTextPipelineConfig.model_validate(value, strict=True)
        path = tmp_path / f"{run_id}.yaml"
        path.write_text(
            yaml.safe_dump(resumed.model_dump(mode="python"), sort_keys=False),
            encoding="utf-8",
        )
        return path, resumed

    phase = "resume-blocked"
    first_resume_path, first_resume = resume_config(
        parent_root=parent_root,
        run_id="pipeline-resume-one",
        correction_attempts=1,
    )
    monkeypatch.setattr(
        pipeline_module,
        "preflight_raw_text_pipeline",
        real_pipeline_preflight,
    )
    first_preflight = real_pipeline_preflight(
        project_root=tmp_path,
        config_path=first_resume_path,
        config=first_resume,
    )
    assert first_preflight["resumedPipelineDocuments"] == 3
    assert first_preflight["resumedCertifiedDocuments"] == 2
    assert first_preflight["unresolvedPipelineDocuments"] == 1
    assert first_preflight["compiledInventoryDocuments"] == 0
    assert first_preflight["providerRequests"] == 0
    with pytest.raises(RawTextPipelineBlockedError, match="without complete publication"):
        run_raw_text_pipeline(
            project_root=tmp_path,
            config_path=first_resume_path,
            config=first_resume,
        )
    assert inventory_calls == 1
    first_resume_root = tmp_path / "runs/pipeline-resume-one"

    phase = "resume-final"
    final_resume_path, final_resume = resume_config(
        parent_root=first_resume_root,
        run_id="pipeline-resume-two",
        correction_attempts=2,
    )
    final_preflight = real_pipeline_preflight(
        project_root=tmp_path,
        config_path=final_resume_path,
        config=final_resume,
    )
    assert final_preflight["resumedCertifiedDocuments"] == 2
    assert final_preflight["unresolvedPipelineDocuments"] == 1
    result = run_raw_text_pipeline(
        project_root=tmp_path,
        config_path=final_resume_path,
        config=final_resume,
    )

    assert result["status"] == "complete"
    assert result["certifiedDocuments"] == 3
    assert result["correctionRounds"] == 2
    assert result["trainingRecordsPublished"] is True
    assert inventory_calls == 1
    assert [scope for item_phase, scope in certification_scopes if item_phase != "parent"] == [
        (_DOCUMENT_IDS[2],),
    ]
    assert [scope for item_phase, scope in correction_scopes if item_phase == "resume-blocked"] == [
        (_DOCUMENT_IDS[2],)
    ]
    assert [scope for item_phase, scope in correction_scopes if item_phase == "resume-final"] == [
        (_DOCUMENT_IDS[2],),
        (_DOCUMENT_IDS[2],),
    ]
    assert len(publication_configs) == 1
    assert (
        sum(source.certified_documents for source in publication_configs[0].certification_sources)
        == 3
    )
    final_lineage = tuple(
        json.loads(line)
        for line in (Path(str(result["artifactRoot"])) / "lineage/cases.jsonl")
        .read_text()
        .splitlines()
    )
    assert final_lineage[0]["history"][-1]["stage"] == "certification"
    assert final_lineage[1]["history"][-1]["stage"] == "certification"
    assert [
        row["attempt"] for row in final_lineage[2]["history"] if row["stage"] == "correction"
    ] == [1, 2, 3, 4]

    parent_certification_root = next(
        tmp_path / row["run"]["path"]
        for row in json.loads((parent_root / "lineage/subruns.json").read_text())
        if row["stage"] == "certification"
    )
    (parent_certification_root / "cases" / _DOCUMENT_IDS[0] / "result.json").unlink()
    with pytest.raises(RuntimeError, match="committed artifact inventory differs"):
        real_pipeline_preflight(
            project_root=tmp_path,
            config_path=final_resume_path,
            config=final_resume,
        )


def test_pipeline_config_rejects_a_publication_count_mismatch(tmp_path: Path) -> None:
    _config_path, config = _write_pipeline_fixture(tmp_path)
    value = config.model_dump(mode="python")
    value["publication"]["documents"] = 2

    with pytest.raises(ValueError, match="publication count differs"):
        SynthesisRawTextPipelineConfig.model_validate(value, strict=True)


def test_pipeline_config_rejects_two_resume_authorities(tmp_path: Path) -> None:
    _config_path, config = _write_pipeline_fixture(tmp_path)
    value = config.model_dump(mode="python")
    reference = {
        "path": "runs/prior",
        "commit_sha256": "a" * 64,
        "transaction_sha256": "b" * 64,
    }
    value["inventory_resume_run"] = reference
    value["pipeline_resume_run"] = reference

    with pytest.raises(ValueError, match="mutually exclusive"):
        SynthesisRawTextPipelineConfig.model_validate(value, strict=True)
