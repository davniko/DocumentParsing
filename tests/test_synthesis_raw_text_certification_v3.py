from __future__ import annotations

import json
from datetime import date
from decimal import Decimal
from pathlib import Path
from typing import Any, cast

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

from document_ocr.hashing import canonical_json_bytes, sha256_bytes, sha256_file
from document_ocr.synthesis.config import (
    SynthesisRawTextCertificationConfig,
    SynthesisRawTextCertifiedCorrectionConfig,
    SynthesisRawTextCertifiedPublicationConfig,
)
from document_ocr.synthesis.linguistic_probe_runtime import model_messages, usage_receipt
from document_ocr.synthesis.raw_text_certification import (
    CertificationStage,
    SemanticAuditOutput,
    _audit_expected_coverage,
    _audit_plan,
    _payload,
    run_raw_text_certification,
)
from document_ocr.synthesis.raw_text_certification_artifacts import (
    load_validated_certification_run,
)
from document_ocr.synthesis.raw_text_certification_host import (
    CertificationInvariantCase,
    CertificationInvariantContext,
)
from document_ocr.synthesis.raw_text_certification_references import (
    CertificationReferenceIndex,
    CertificationReferenceReceipt,
)
from document_ocr.synthesis.raw_text_certified_correction import (
    run_raw_text_certified_correction,
)
from document_ocr.synthesis.raw_text_certified_publication import (
    run_raw_text_certified_publication,
)
from document_ocr.synthesis.transport_capacity import TransportCapacityLimits

_PROJECT_ROOT = Path(__file__).resolve().parents[1]


class _FakeAsyncOpenAI:
    def __init__(self, **_kwargs: Any) -> None:
        pass

    async def close(self) -> None:
        pass


def _reference_context() -> CertificationInvariantContext:
    body = {
        "schemaVersion": 2,
        "iso3166Sha256": "1" * 64,
        "geonamesRegistryReceiptSha256": "2" * 64,
        "geonamesRegistryContentSha256": "3" * 64,
        "geonamesJsonlSha256": "4" * 64,
        "geonamesRecords": 1,
        "localityProjectionSha256": "5" * 64,
        "phonenumbersDistribution": "phonenumberslite",
        "phonenumbersVersion": "9.0.38",
        "callingCodeRegionMapSha256": "6" * 64,
        "packageRegistrySha256": "7" * 64,
        "packageRegistryEntries": 1,
        "packageRegistryPayloadSha256": "8" * 64,
    }
    receipt = CertificationReferenceReceipt.model_validate(
        {**body, "contentSha256": sha256_bytes(canonical_json_bytes(body))},
        strict=True,
    )
    references = CertificationReferenceIndex(
        receipt=receipt,
        country_aliases={},
        locality_aliases={},
        locality_countries={},
        package_display_names={},
    )
    limits = TransportCapacityLimits(
        policy="source_type_aware_maersk_upper_bounds_v1",
        published_reference_margin_fraction=Decimal("0.05"),
        twenty_standard_payload_kg=Decimal("28300"),
        twenty_standard_volume_m3=Decimal("33.2"),
        forty_standard_payload_kg=Decimal("28870"),
        forty_standard_volume_m3=Decimal("67.7"),
        forty_high_cube_payload_kg=Decimal("28690"),
        forty_high_cube_volume_m3=Decimal("76.4"),
        forty_five_high_cube_payload_kg=Decimal("27650"),
        forty_five_high_cube_volume_m3=Decimal("86"),
        out_of_gauge_payload_kg=Decimal("47300"),
        unclassified_payload_kg=Decimal("47300"),
        unclassified_volume_m3=Decimal("86"),
    )
    return CertificationInvariantContext(references=references, capacity_limits=limits)


def _contract(document_id: str) -> dict[str, Any]:
    source = b"--- PAGE 1 ---\nB/L: OLD-123\n"
    source_label = {
        "schemaVersion": "3.0.0-experimental",
        "documentPatch": {"billOfLadingNumber": "OLD-123"},
    }
    target_label = {
        "schemaVersion": "5.0.0-experimental",
        "documentPatch": {"billOfLadingNumber": "NEW-456"},
    }
    return {
        "deterministicPrefills": [],
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
        "anchoredScalarReplacementRequirements": [
            {
                "targetPaths": ["documentPatch.billOfLadingNumber"],
                "sourceLineIds": ["L00002"],
                "sourceSurface": "OLD-123",
                "targetSurface": "NEW-456",
                "surfaceKind": "scalar",
            }
        ],
        "sourceSemanticRoleHints": [],
        "sourceLabel": source_label,
        "targetLabel": target_label,
        "result": {
            "documentId": document_id,
            "scenarioId": "syn_" + "b" * 64,
            "sourceTextSha256": sha256_bytes(source),
            "sourceLabelSha256": sha256_bytes(canonical_json_bytes(source_label)),
            "targetLabelSha256": sha256_bytes(canonical_json_bytes(target_label)),
        },
    }


def _fixture(
    tmp_path: Path,
    *,
    candidate: str,
    run_id: str,
) -> tuple[Path, Path, SynthesisRawTextCertificationConfig, CertificationInvariantContext]:
    document_id = "doc_" + "a" * 64
    prompt_path = tmp_path / "auditor.md"
    prompt_path.write_text("Audit the complete immutable candidate.", encoding="utf-8")
    input_root = tmp_path / "input"
    case_root = input_root / "cases" / document_id
    case_root.mkdir(parents=True)
    source = "--- PAGE 1 ---\nB/L: OLD-123\n"
    contract = _contract(document_id)
    (case_root / "source.txt").write_text(source, encoding="utf-8")
    (case_root / "final.txt").write_text(candidate, encoding="utf-8")
    for name, value in (
        ("source-label.json", contract["sourceLabel"]),
        ("target-label.json", contract["targetLabel"]),
        ("source-contract.json", contract),
        ("inventory.json", []),
        ("deterministic-edits.json", []),
    ):
        (case_root / name).write_bytes(canonical_json_bytes(value) + b"\n")
    config = SynthesisRawTextCertificationConfig.model_validate(
        {
            "schema_version": 3,
            "task": "bill_of_lading_synthetic_raw_text_certification_v3",
            "audit_contract_version": 3,
            "environment_file": ".env",
            "run": {"run_id": run_id, "output_dir": "outputs"},
            "input_run": {
                "path": "input",
                "commit_sha256": "a" * 64,
                "transaction_sha256": "b" * 64,
            },
            "invariant_inputs": {
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
                    "published_reference_margin_fraction": 0.05,
                    "twenty_standard_payload_kg": 28300,
                    "twenty_standard_volume_m3": 33.2,
                    "forty_standard_payload_kg": 28870,
                    "forty_standard_volume_m3": 67.7,
                    "forty_high_cube_payload_kg": 28690,
                    "forty_high_cube_volume_m3": 76.4,
                    "forty_five_high_cube_payload_kg": 27650,
                    "forty_five_high_cube_volume_m3": 86,
                    "out_of_gauge_payload_kg": 47300,
                    "unclassified_payload_kg": 47300,
                    "unclassified_volume_m3": 86,
                },
            },
            "input_case_contract_filename": "source-contract.json",
            "case_ids": [document_id],
            "prompt": {"path": "auditor.md", "sha256": sha256_file(prompt_path)},
            "provider": {
                "kind": "openrouter",
                "model": "z-ai/glm-5.3-flash",
                "api_key_env": "OPENROUTER_API_KEY",
                "reasoning_effort": "low",
                "request_timeout_seconds": 30,
                "transport_max_retries": 0,
                "max_output_tokens": 12288,
                "require_parameters": True,
                "data_collection": "deny",
                "allow_fallbacks": True,
                "provider_order": ["deepinfra/fp4", "coreweave/fp8"],
                "max_price": {"prompt": 0.075, "completion": 0.25},
                "pricing": {
                    "currency": "USD",
                    "effective_date": date(2026, 9, 4),
                    "source_url": "https://openrouter.ai/z-ai/glm-5.3-flash",
                    "input_usd_per_million": 0.075,
                    "cached_input_usd_per_million": 0.015,
                    "cache_write_multiplier": 1.0,
                    "output_usd_per_million": 0.25,
                },
            },
            "workflow": {
                "documents": 1,
                "max_concurrent_documents": 1,
                "semantic_audit_passes": 2,
                "confirmation_reasoning_effort": "high",
                "evaluation_only": False,
                "require_complete_dimension_coverage": True,
                "require_unanimous_clean": True,
                "max_provider_route_rounds": 1,
                "retry_initial_delay_seconds": 0,
                "retry_delay_multiplier": 1,
                "retry_max_delay_seconds": 0,
                "retry_jitter_seconds": 0,
                "provider_native_json_schema": True,
                "require_exact_finding_evidence": True,
                "require_input_candidate_immutable": True,
                "require_exact_line_and_page_topology": True,
                "require_prior_host_contract": True,
                "persist_every_prompt_response_and_receipt": True,
                "publish_training_records": False,
            },
        },
        strict=True,
    )
    config_path = tmp_path / f"{run_id}.yaml"
    config_path.write_text(
        yaml.safe_dump(config.model_dump(mode="python"), sort_keys=False),
        encoding="utf-8",
    )
    return config_path, input_root, config, _reference_context()


def _clean_output(*, contract: dict[str, Any], source: str, candidate: str) -> SemanticAuditOutput:
    coverage = _audit_expected_coverage(_audit_plan(contract, source=source, current=candidate))
    return SemanticAuditOutput.model_validate(
        {
            "auditContractVersion": 2,
            "dimensionChecks": tuple(
                {
                    "dimension": dimension,
                    "assessment": f"Completed every {dimension} obligation.",
                }
                for dimension in coverage
            ),
            "findings": (),
        },
        strict=True,
    )


def _clean_stage(
    *,
    document_id: str,
    audit_pass: int,
    reasoning_effort: str,
    source: str,
    candidate: str,
    contract: dict[str, Any],
    invariant_case: CertificationInvariantCase,
    config: SynthesisRawTextCertificationConfig,
    prompt: str,
) -> CertificationStage:
    plan = _audit_plan(contract, source=source, current=candidate)
    payload = _payload(
        document_id=document_id,
        source=source,
        current=candidate,
        source_label=cast(dict[str, Any], contract["sourceLabel"]),
        target_label=cast(dict[str, Any], contract["targetLabel"]),
        host_findings=(),
        audit_contract_version=3,
        audit_plan=plan,
        invariant_case=invariant_case,
    )
    output = _clean_output(contract=contract, source=source, candidate=candidate)
    identity = f"pass-{audit_pass}"
    request = ModelRequest(
        parts=(
            SystemPromptPart(prompt),
            UserPromptPart(json.dumps(payload, ensure_ascii=False, separators=(",", ":"))),
        ),
        run_id=f"run-{identity}",
        conversation_id=f"conversation-{identity}",
    )
    response = ModelResponse(
        parts=(TextPart(canonical_json_bytes(output.model_dump(mode="json")).decode()),),
        usage=RequestUsage(input_tokens=10, output_tokens=5),
        model_name=config.provider.model,
        provider_name="openrouter",
        provider_url="https://openrouter.ai/api/v1",
        provider_details={"downstream_provider": "DeepInfra"},
        provider_response_id=f"response-{identity}",
        finish_reason="stop",
        run_id=request.run_id,
        conversation_id=request.conversation_id,
    )
    messages = (request, response)
    return CertificationStage.model_validate(
        {
            "auditContractVersion": 3,
            "auditPass": audit_pass,
            "reasoningEffort": reasoning_effort,
            "auditPlanSha256": sha256_bytes(canonical_json_bytes(plan)),
            "routeProvider": "deepinfra/fp4",
            "routeRound": 1,
            "routeAttempt": 1,
            "retryDelayBeforeSeconds": 0.0,
            "inputPayloadSha256": sha256_bytes(canonical_json_bytes(payload)),
            "outputSchemaSha256": sha256_bytes(
                canonical_json_bytes(SemanticAuditOutput.model_json_schema())
            ),
            "startedAtUnixSeconds": float(audit_pass * 2),
            "completedAtUnixSeconds": float(audit_pass * 2 + 1),
            "usage": usage_receipt(messages[1:], config.provider.pricing),
            "messages": model_messages(messages),
            "modelOutput": output.model_dump(mode="json"),
            "hostError": None,
            "errorType": None,
            "errorMessage": None,
        },
        strict=True,
    )


def _patch_host_inputs(
    monkeypatch: pytest.MonkeyPatch,
    *,
    input_root: Path,
    context: CertificationInvariantContext,
) -> None:
    import document_ocr.synthesis.raw_text_certification as certification_module
    import document_ocr.synthesis.raw_text_certification_artifacts as artifact_module

    monkeypatch.setattr(
        certification_module,
        "_validate_reference_run",
        lambda *_args, **_kwargs: input_root,
    )
    monkeypatch.setattr(
        certification_module,
        "load_certification_invariant_context",
        lambda **_kwargs: context,
    )
    monkeypatch.setattr(
        artifact_module,
        "load_certification_invariant_context",
        lambda **_kwargs: context,
    )


def _correction_config(
    *,
    tmp_path: Path,
    certification_root: Path,
    certification: SynthesisRawTextCertificationConfig,
) -> tuple[Path, SynthesisRawTextCertifiedCorrectionConfig]:
    commit = json.loads((certification_root / "_COMMIT.json").read_text())
    provider = certification.provider.model_dump(mode="python")
    provider["max_output_tokens"] = 8192
    config = SynthesisRawTextCertifiedCorrectionConfig.model_validate(
        {
            "schema_version": 1,
            "task": "bill_of_lading_synthetic_raw_text_certified_correction_v1",
            "environment_file": ".env",
            "run": {"run_id": "v3-deterministic-repair", "output_dir": "outputs"},
            "certification_run": {
                "path": certification_root.relative_to(tmp_path).as_posix(),
                "commit_sha256": sha256_file(certification_root / "_COMMIT.json"),
                "transaction_sha256": commit["transaction_sha256"],
            },
            "case_ids": list(certification.case_ids),
            "prompt": certification.prompt.model_dump(mode="python"),
            "provider": provider,
            "workflow": {
                "documents": 1,
                "required_audit_contract_version": 3,
                "max_concurrent_documents": 1,
                "max_provider_route_rounds": 1,
                "max_successful_model_responses_per_document": 1,
                "retry_initial_delay_seconds": 0,
                "retry_delay_multiplier": 1,
                "retry_max_delay_seconds": 0,
                "retry_jitter_seconds": 0,
                "provider_native_json_schema": True,
                "require_exact_finding_evidence": True,
                "require_only_cited_lines_mutable": True,
                "require_exact_line_and_page_topology": True,
                "persist_every_prompt_response_and_receipt": True,
                "publish_training_records": False,
            },
        },
        strict=True,
    )
    path = tmp_path / "v3-deterministic-repair.yaml"
    path.write_text(
        yaml.safe_dump(config.model_dump(mode="python"), sort_keys=False),
        encoding="utf-8",
    )
    return path, config


def _publication_config(
    *,
    tmp_path: Path,
    certification_root: Path,
    certification: SynthesisRawTextCertificationConfig,
) -> tuple[Path, SynthesisRawTextCertifiedPublicationConfig]:
    commit = json.loads((certification_root / "_COMMIT.json").read_text())
    document_ids = sorted(certification.case_ids)
    config = SynthesisRawTextCertifiedPublicationConfig.model_validate(
        {
            "schema_version": 1,
            "task": "bill_of_lading_synthetic_raw_text_certified_publication_v1",
            "run": {"run_id": "v3-publication", "output_dir": "outputs"},
            "certification_sources": [
                {
                    "run": {
                        "path": certification_root.relative_to(tmp_path).as_posix(),
                        "commit_sha256": sha256_file(certification_root / "_COMMIT.json"),
                        "transaction_sha256": commit["transaction_sha256"],
                    },
                    "certified_documents": len(document_ids),
                    "certified_document_ids_sha256": sha256_bytes(
                        canonical_json_bytes(document_ids)
                    ),
                }
            ],
            "workflow": {
                "documents": len(document_ids),
                "required_audit_contract_version": 3,
                "require_every_case_certified": True,
                "require_unique_source_documents": True,
                "require_unique_scenarios": True,
                "replay_current_host_audit": True,
                "require_v5_canonical_target": True,
                "publish_training_records": True,
            },
        },
        strict=True,
    )
    path = tmp_path / "v3-publication.yaml"
    path.write_text(
        yaml.safe_dump(config.model_dump(mode="python"), sort_keys=False),
        encoding="utf-8",
    )
    return path, config


def test_contract_v3_host_rejection_commits_without_loading_a_provider_key(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import document_ocr.synthesis.raw_text_certification as certification_module

    candidate = "--- PAGE 1 ---\nB/L: OLD-123\n"
    config_path, input_root, config, context = _fixture(
        tmp_path,
        candidate=candidate,
        run_id="v3-host-reject",
    )
    _patch_host_inputs(monkeypatch, input_root=input_root, context=context)
    monkeypatch.setattr(
        certification_module,
        "load_provider_key",
        lambda *_args, **_kwargs: pytest.fail("host rejection must not load a credential"),
    )

    summary = run_raw_text_certification(
        project_root=tmp_path,
        config_path=config_path,
        config=config,
    )
    root = tmp_path / "outputs/v3-host-reject"
    replayed = load_validated_certification_run(root, project_root=tmp_path)

    assert summary["deterministicRejectedBeforeProvider"] == 1
    assert summary["requests"] == 0
    assert replayed.cases[0].result.status == "needs_review"
    assert replayed.cases[0].result.auditPasses == 0
    assert replayed.cases[0].invariant_case is not None
    assert not replayed.cases[0].invariant_case.audit.passed


def test_contract_v3_clean_candidate_replays_two_provider_passes_and_rejects_tampering(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import document_ocr.synthesis.raw_text_certification as certification_module

    candidate = "--- PAGE 1 ---\nB/L: NEW-456\n"
    config_path, input_root, config, context = _fixture(
        tmp_path,
        candidate=candidate,
        run_id="v3-clean",
    )
    _patch_host_inputs(monkeypatch, input_root=input_root, context=context)
    monkeypatch.setattr(certification_module, "load_provider_key", lambda *_args: "test-key")
    monkeypatch.setattr(certification_module, "AsyncOpenAI", _FakeAsyncOpenAI)
    monkeypatch.setattr(
        certification_module,
        "_model_pair",
        lambda **_kwargs: (cast(Any, object()), cast(Any, object())),
    )

    async def fake_call_audit(**kwargs: Any) -> tuple[Any, tuple[CertificationStage, ...]]:
        stages = tuple(
            _clean_stage(
                document_id=kwargs["document_id"],
                audit_pass=audit_pass,
                reasoning_effort=reasoning,
                source=kwargs["source"],
                candidate=kwargs["current"],
                contract=kwargs["contract"],
                invariant_case=kwargs["invariant_case"],
                config=kwargs["config"],
                prompt=kwargs["prompt"],
            )
            for audit_pass, reasoning in ((1, "low"), (2, "high"))
        )
        return (), stages

    monkeypatch.setattr(certification_module, "_call_audit", fake_call_audit)

    summary = run_raw_text_certification(
        project_root=tmp_path,
        config_path=config_path,
        config=config,
    )
    root = tmp_path / "outputs/v3-clean"
    replayed = load_validated_certification_run(root, project_root=tmp_path)

    assert summary["certifiedDocuments"] == 1
    assert summary["requests"] == 2
    assert replayed.cases[0].result.status == "certified"
    assert replayed.cases[0].invariant_case is not None
    assert replayed.cases[0].invariant_case.audit.passed

    publication_path, publication = _publication_config(
        tmp_path=tmp_path,
        certification_root=root,
        certification=config,
    )
    published = run_raw_text_certified_publication(
        project_root=tmp_path,
        config_path=publication_path,
        config=publication,
    )
    publication_root = tmp_path / "outputs/v3-publication"
    record = json.loads((publication_root / "dataset/records.jsonl").read_text())
    published_case = publication_root / ("cases/syn_" + "b" * 64)
    assert published["trainingRecordsPublished"] is True
    assert record["sourceDocumentId"] == config.case_ids[0]
    assert record["joinedRawText"] == candidate
    assert (published_case / "certification-invariant-audit.json").is_file()
    assert (
        run_raw_text_certified_publication(
            project_root=tmp_path,
            config_path=publication_path,
            config=publication,
        )["commitSha256"]
        == published["commitSha256"]
    )

    envelope_path = root / f"cases/{config.case_ids[0]}/invariant-envelope.json"
    envelope = json.loads(envelope_path.read_text())
    envelope["candidateTextSha256"] = "f" * 64
    envelope_path.write_bytes(canonical_json_bytes(envelope) + b"\n")
    with pytest.raises(ValueError, match="invariant artifacts fail deterministic replay"):
        load_validated_certification_run(root, project_root=tmp_path)


def test_contract_v3_correction_applies_only_complete_host_authored_repairs(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import document_ocr.synthesis.raw_text_certification as certification_module
    import document_ocr.synthesis.raw_text_certified_correction as correction_module

    candidate = "--- PAGE 1 ---\nB/L: OLD-123\n"
    certification_path, input_root, certification, context = _fixture(
        tmp_path,
        candidate=candidate,
        run_id="v3-repair-source",
    )
    _patch_host_inputs(monkeypatch, input_root=input_root, context=context)
    monkeypatch.setattr(
        certification_module,
        "load_provider_key",
        lambda *_args, **_kwargs: pytest.fail("deterministic rejection must not load a credential"),
    )
    run_raw_text_certification(
        project_root=tmp_path,
        config_path=certification_path,
        config=certification,
    )
    certification_root = tmp_path / "outputs/v3-repair-source"
    replayed = load_validated_certification_run(certification_root, project_root=tmp_path)
    invariant = replayed.cases[0].invariant_case
    assert invariant is not None
    assert [repair.oldFragment for row in invariant.audit.findings for repair in row.repairs] == [
        "OLD-123"
    ]

    correction_path, correction = _correction_config(
        tmp_path=tmp_path,
        certification_root=certification_root,
        certification=certification,
    )
    monkeypatch.setattr(
        correction_module,
        "load_provider_key",
        lambda *_args, **_kwargs: pytest.fail("contract-v3 correction must not load a credential"),
    )
    summary = run_raw_text_certified_correction(
        project_root=tmp_path,
        config_path=correction_path,
        config=correction,
    )
    correction_root = tmp_path / "outputs/v3-deterministic-repair"
    case_root = correction_root / "cases" / certification.case_ids[0]

    assert summary["correctionCandidateDocuments"] == 1
    assert summary["requests"] == 0
    assert summary["estimatedCostUsd"] == "0"
    assert (case_root / "final.txt").read_text() == "--- PAGE 1 ---\nB/L: NEW-456\n"
    assert json.loads((case_root / "invariant-audit.json").read_text())["passed"] is True
    assert json.loads((case_root / "stages.json").read_text()) == []
    assert (
        run_raw_text_certified_correction(
            project_root=tmp_path,
            config_path=correction_path,
            config=correction,
        )["commitSha256"]
        == summary["commitSha256"]
    )
