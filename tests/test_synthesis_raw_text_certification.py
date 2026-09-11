from __future__ import annotations

import asyncio
import json
from dataclasses import replace
from decimal import Decimal
from importlib.metadata import version
from pathlib import Path
from typing import Any, Literal, cast

import pytest
import yaml
from pydantic import ValidationError
from pydantic_ai.exceptions import ModelHTTPError, UnexpectedModelBehavior
from pydantic_ai.messages import (
    ModelMessagesTypeAdapter,
    ModelRequest,
    ModelResponse,
    SystemPromptPart,
    TextPart,
    ToolCallPart,
    UserPromptPart,
)
from pydantic_ai.usage import RequestUsage

from document_ocr.hashing import canonical_json_bytes, sha256_bytes, sha256_file
from document_ocr.synthesis.config import (
    CommittedArtifactDirectoryConfig,
    SynthesisRawTextCertificationConfig,
    load_synthesis_raw_text_certification_config,
)
from document_ocr.synthesis.linguistic_probe_runtime import model_messages, usage_receipt
from document_ocr.synthesis.raw_text_certification import (
    CertificationCaseResult,
    CertificationRouteError,
    CertificationStage,
    SemanticAuditEvidence,
    SemanticAuditFinding,
    SemanticAuditOutput,
    _audit_expected_coverage,
    _audit_plan,
    _call_audit,
    _contains_bounded_interleaved_description,
    _empty_usage,
    _execute_certification_cases,
    _host_audit,
    _load_case_checkpoint,
    _payload,
    _publish_case_checkpoint,
    _route_error_is_retryable_for_attempt,
    _route_error_receipt,
    _target_literal_present,
    _transient_route_error,
    _validate_audit_output,
    _validate_benchmark_plan_authorization,
    _validate_case_contract_labels,
    certification_implementation_contract,
    replay_certification_audit_stages,
    validate_certification_run_stage_identities,
)
from document_ocr.synthesis.run_safety import StagedArtifactRun

_PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _yaml_value(value: object) -> object:
    if isinstance(value, dict):
        return {key: _yaml_value(item) for key, item in value.items()}
    if isinstance(value, tuple):
        return [_yaml_value(item) for item in value]
    if isinstance(value, Decimal):
        return float(value)
    return value


def test_certification_implementation_contract_pins_invariant_runtime_dependencies() -> None:
    dependencies = certification_implementation_contract()["dependencyImplementationSha256"]

    assert isinstance(dependencies, dict)
    for name, filename in {
        "containerSemantics": "container_semantics.py",
        "transportCapacity": "transport_capacity.py",
    }.items():
        assert dependencies[name] == sha256_file(
            _PROJECT_ROOT / "src/document_ocr/synthesis" / filename
        )


def _contract() -> dict[str, object]:
    return {
        "compactLabelChangeContract": [],
        "sourceStatusPreservationRequirements": [],
        "surfaceRenderingRequirements": [],
        "operationalFlavorRequirements": [],
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
        "jurisdictionalSurfaceRequirements": [],
        "compoundPartyFlavorRequirements": [],
        "inlineSlotTopologyRequirements": [],
        "cargoFlavorRewriteRequirements": [],
        "anchoredScalarReplacementRequirements": [],
        "sourceSemanticRoleHints": [],
        "sourceLabel": {"documentPatch": {"billOfLadingNumber": "OLD-123"}},
        "targetLabel": {"documentPatch": {"billOfLadingNumber": "NEW-123"}},
    }


def _finding(fragment: str = "OLD-123", line_id: str = "L00002") -> SemanticAuditFinding:
    return SemanticAuditFinding(
        findingKind="target_fact_mismatch",
        evidence=(SemanticAuditEvidence(lineId=line_id, currentFragment=fragment),),
        problem="The source bill number remains in the candidate.",
    )


def _v2_output(
    *,
    contract: dict[str, object],
    source: str,
    current: str,
    findings: tuple[SemanticAuditFinding, ...] = (),
) -> SemanticAuditOutput:
    coverage = _audit_expected_coverage(_audit_plan(contract, source=source, current=current))
    first_target_id = coverage["target_fact_fidelity"][0]
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
            "findings": tuple(
                {
                    "findingKind": finding.findingKind,
                    "evidence": tuple({"lineId": evidence.lineId} for evidence in finding.evidence),
                    "problem": finding.problem,
                    "obligationIds": (first_target_id,),
                }
                for finding in findings
            ),
        },
        strict=True,
    )


def _v2_config(document_id: str) -> SynthesisRawTextCertificationConfig:
    legacy = load_synthesis_raw_text_certification_config(
        _PROJECT_ROOT
        / "configs/synthesis/mpci_bl_raw_text_certification5_glm53_v24_third_residuals.yaml"
    ).model_dump(mode="python")
    legacy["audit_contract_version"] = 2
    legacy["case_ids"] = [document_id]
    prompt_path = "prompts/synthesis/mpci_bl_raw_text_semantic_auditor_v3.md"
    legacy["prompt"] = {
        "path": prompt_path,
        "sha256": sha256_file(_PROJECT_ROOT / prompt_path),
    }
    legacy["provider"]["reasoning_effort"] = "low"
    legacy["provider"]["max_output_tokens"] = 12288
    legacy["workflow"].update(
        {
            "documents": 1,
            "max_concurrent_documents": 1,
            "semantic_audit_passes": 2,
            "confirmation_reasoning_effort": "high",
            "evaluation_only": False,
            "require_complete_dimension_coverage": True,
            "require_unanimous_clean": True,
        }
    )
    return SynthesisRawTextCertificationConfig.model_validate(legacy, strict=True)


def test_planned_case_execution_stops_before_the_next_call() -> None:
    called: list[str] = []

    async def run_case(document_id: str) -> bool:
        called.append(document_id)
        return document_id == "second"

    with pytest.raises(RuntimeError, match="second"):
        asyncio.run(
            _execute_certification_cases(
                document_ids=("first", "second", "must-not-run"),
                run_case=run_case,
                fail_fast=True,
            )
        )
    assert called == ["first", "second"]


def _v2_evaluation_config(
    document_id: str, *, reasoning_effort: Literal["low", "medium", "high"]
) -> SynthesisRawTextCertificationConfig:
    value = _v2_config(document_id).model_dump(mode="python")
    value["provider"].update(
        {
            "reasoning_effort": reasoning_effort,
            "allow_fallbacks": False,
            "provider_only": ["deepinfra/fp4"],
            "provider_order": None,
            "provider_sort": None,
        }
    )
    value["workflow"].update(
        {
            "semantic_audit_passes": 1,
            "confirmation_reasoning_effort": None,
            "evaluation_only": True,
            "require_unanimous_clean": False,
        }
    )
    return SynthesisRawTextCertificationConfig.model_validate(value, strict=True)


def _v2_stage(
    *,
    document_id: str,
    audit_pass: Literal[1, 2],
    reasoning_effort: Literal["low", "medium", "high"],
    response_id: str,
    source: str,
    current: str,
    contract: dict[str, object],
    output: SemanticAuditOutput,
    config: SynthesisRawTextCertificationConfig,
    route_provider: str | None = None,
    route_attempt: int = 1,
    started_at: float | None = None,
    attempt_identity: str | None = None,
) -> CertificationStage:
    plan = _audit_plan(contract, source=source, current=current)
    host = _host_audit(source=source, output=current, contract=contract)
    payload = _payload(
        document_id=document_id,
        source=source,
        current=current,
        source_label=contract["sourceLabel"],  # type: ignore[arg-type]
        target_label=contract["targetLabel"],  # type: ignore[arg-type]
        host_findings=host.findings,
        audit_contract_version=2,
        audit_plan=plan,
    )
    prompt = (_PROJECT_ROOT / config.prompt.path).read_text(encoding="utf-8")
    identity = response_id if attempt_identity is None else attempt_identity
    attempt_run_id = f"run-{identity}"
    conversation_id = f"conversation-{identity}"
    assert config.provider.kind == "openrouter"
    stage_route_provider = route_provider
    if (
        stage_route_provider is None
        and config.provider.allow_fallbacks
        and config.provider.provider_order is not None
        and len(config.provider.provider_order) >= 2
    ):
        stage_route_provider = config.provider.provider_order[0]
    requested_provider = stage_route_provider
    if requested_provider is None:
        provider_only = config.provider.provider_only or ()
        assert len(provider_only) == 1
        requested_provider = provider_only[0]
    downstream_provider = {
        "gmicloud/fp8": "GMICloud",
        "deepinfra/fp4": "DeepInfra",
        "coreweave/fp8": "CoreWeave",
        "fireworks": "Fireworks",
        "nextbit/fp8": "NextBit",
    }[requested_provider]
    messages = (
        ModelRequest(
            parts=(
                SystemPromptPart(prompt),
                UserPromptPart(json.dumps(payload, ensure_ascii=False, separators=(",", ":"))),
            ),
            run_id=attempt_run_id,
            conversation_id=conversation_id,
        ),
        ModelResponse(
            parts=(TextPart(canonical_json_bytes(output.model_dump(mode="json")).decode("utf-8")),),
            usage=RequestUsage(input_tokens=1, output_tokens=1),
            model_name=config.provider.model,
            provider_name="openrouter",
            provider_url="https://openrouter.ai/api/v1",
            provider_details={"downstream_provider": downstream_provider},
            provider_response_id=response_id,
            finish_reason="stop",
            run_id=attempt_run_id,
            conversation_id=conversation_id,
        ),
    )
    stage_started_at = float(audit_pass) if started_at is None else started_at
    return CertificationStage.model_validate(
        {
            "auditContractVersion": 2,
            "auditPass": audit_pass,
            "reasoningEffort": reasoning_effort,
            "auditPlanSha256": sha256_bytes(canonical_json_bytes(plan)),
            "routeProvider": stage_route_provider,
            "routeRound": 1,
            "routeAttempt": route_attempt,
            "retryDelayBeforeSeconds": 0.0,
            "inputPayloadSha256": sha256_bytes(canonical_json_bytes(payload)),
            "outputSchemaSha256": sha256_bytes(
                canonical_json_bytes(SemanticAuditOutput.model_json_schema())
            ),
            "startedAtUnixSeconds": stage_started_at,
            "completedAtUnixSeconds": stage_started_at + 0.5,
            "usage": usage_receipt(messages[1:], config.provider.pricing),
            "messages": model_messages(messages),
            "modelOutput": output.model_dump(mode="json"),
            "hostError": None,
            "errorType": None,
            "errorMessage": None,
        },
        strict=True,
    )


def _v2_error_stage(
    *,
    document_id: str,
    audit_pass: Literal[1, 2],
    reasoning_effort: Literal["low", "medium", "high"],
    source: str,
    current: str,
    contract: dict[str, object],
    config: SynthesisRawTextCertificationConfig,
    route_provider: str,
    route_attempt: int,
    retryable: bool,
) -> CertificationStage:
    base = _v2_stage(
        document_id=document_id,
        audit_pass=audit_pass,
        reasoning_effort=reasoning_effort,
        response_id=f"unused-{audit_pass}-{route_attempt}",
        source=source,
        current=current,
        contract=contract,
        output=_v2_output(contract=contract, source=source, current=current),
        config=config,
    ).model_dump(mode="python")
    messages = base["messages"]
    assert isinstance(messages, list)
    error = (
        UnexpectedModelBehavior("provider route failed")
        if retryable
        else RuntimeError("provider route failed")
    )
    route_error = _route_error_receipt(error)
    return CertificationStage.model_validate(
        {
            **base,
            "routeProvider": route_provider,
            "routeAttempt": route_attempt,
            "startedAtUnixSeconds": float(route_attempt),
            "completedAtUnixSeconds": float(route_attempt) + 0.5,
            "usage": _empty_usage(),
            "messages": messages[:1],
            "modelOutput": None,
            "errorType": type(error).__name__,
            "errorMessage": str(error),
            "retryableRouteError": retryable,
            "routeError": route_error,
        },
        strict=True,
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


def test_attempt_retry_requires_transient_zero_response_and_zero_cost() -> None:
    rate_limit = _route_error_receipt(
        ModelHTTPError(
            429,
            "z-ai/glm-5.3-flash",
            {"message": "temporarily rate-limited upstream"},
        )
    )
    empty = _empty_usage()

    assert _route_error_is_retryable_for_attempt(rate_limit, empty)
    assert not _route_error_is_retryable_for_attempt(
        rate_limit,
        empty.model_copy(update={"requests": 1, "providerResponseIds": ("paid-response",)}),
    )
    assert not _route_error_is_retryable_for_attempt(
        _route_error_receipt(UnexpectedModelBehavior("invalid structured output")),
        empty,
    )


@pytest.mark.parametrize("reasoning_effort", ("low", "medium", "high"))
def test_contract_v2_evaluation_uses_one_isolated_native_effort(
    reasoning_effort: Literal["low", "medium", "high"],
) -> None:
    config = _v2_evaluation_config(
        "doc_" + "e" * 64,
        reasoning_effort=reasoning_effort,
    )

    assert config.workflow.evaluation_only
    assert config.workflow.semantic_audit_passes == 1
    assert config.provider.reasoning_effort == reasoning_effort
    assert config.provider.kind == "openrouter"
    assert config.provider.provider_only == ("deepinfra/fp4",)
    assert config.provider.provider_order is None
    assert config.provider.max_output_tokens == 12288


def test_contract_v2_medium_stage_replays_on_the_real_validation_path() -> None:
    document_id = "doc_" + "d" * 64
    text = "--- PAGE 1 ---\nNEW-123\nNEW CARRIER\nNEW CARRIER\n"
    contract = _contract()
    config = _v2_evaluation_config(document_id, reasoning_effort="medium")
    output = _v2_output(contract=contract, source=text, current=text)
    stage = _v2_stage(
        document_id=document_id,
        audit_pass=1,
        reasoning_effort="medium",
        response_id="medium-receipt",
        source=text,
        current=text,
        contract=contract,
        output=output,
        config=config,
    )

    replay = replay_certification_audit_stages(
        document_id=document_id,
        stages=(stage,),
        source=text,
        current=text,
        contract=contract,
        audit_contract_version=2,
        config=config,
    )

    assert replay.audit_passes == 1
    assert replay.clean_audit_passes == 1
    assert replay.findings == ()


def _authorized_medium_config(
    *,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> SynthesisRawTextCertificationConfig:
    import document_ocr.synthesis.raw_text_certification as certification_module

    document_id = "doc_" + "a" * 64
    baseline = _v2_evaluation_config(document_id, reasoning_effort="low")
    baseline = baseline.model_copy(
        update={"run": baseline.run.model_copy(update={"run_id": "benchmark-baseline"})}
    )
    challenger = _v2_evaluation_config(document_id, reasoning_effort="medium")
    challenger = challenger.model_copy(
        update={"run": challenger.run.model_copy(update={"run_id": "benchmark-challenger"})}
    )
    plan_root = tmp_path / "artifacts/benchmark-plan"
    (plan_root / "authorization").mkdir(parents=True)
    (plan_root / "inputs").mkdir()
    template_rows = (("baseline", "low", baseline), ("challenger", "medium", challenger))
    authorization_arms: list[dict[str, object]] = []
    plan_arms: dict[str, object] = {}

    for role, effort, template in template_rows:
        payload = yaml.safe_dump(
            _yaml_value(template.model_dump(mode="python")),
            sort_keys=False,
        ).encode("utf-8")
        copied_path = plan_root / f"inputs/{role}-config.yaml"
        copied_path.write_bytes(payload)
        source_path = f"configs/{role}.yaml"
        authorization_arms.append(
            {
                "role": role,
                "effort": effort,
                "templateConfigPath": source_path,
                "templateConfigSha256": sha256_bytes(payload),
                "templateResolvedConfigSha256": sha256_bytes(
                    canonical_json_bytes(template.model_dump(mode="json"))
                ),
                "runId": template.run.run_id,
            }
        )
        plan_arms[role] = {
            "effort": effort,
            "config": {"path": source_path, "sha256": sha256_bytes(payload)},
        }
    (plan_root / "authorization/arm-configs.json").write_bytes(
        canonical_json_bytes(
            {
                "schemaVersion": 1,
                "task": "raw_text_certification_benchmark_plan_authorization_v1",
                "arms": authorization_arms,
            }
        )
        + b"\n"
    )
    (plan_root / "config.json").write_bytes(
        canonical_json_bytes(
            {
                "schemaVersion": 2,
                "task": "raw_text_certification_contract_v2_benchmark_plan_v2",
                "run": {
                    "runId": "benchmark-plan",
                    "outputDir": "artifacts",
                },
                "cohortRun": {
                    "path": "artifacts/cohort",
                    "commitSha256": "1" * 64,
                    "transactionSha256": "2" * 64,
                },
                **plan_arms,
            }
        )
        + b"\n"
    )
    (plan_root / "provenance").mkdir()
    (plan_root / "provenance/transaction.json").write_bytes(
        canonical_json_bytes(
            {
                "certificationImplementation": certification_implementation_contract(),
                "certificationRuntime": {
                    "pydanticAiVersion": version("pydantic-ai-slim"),
                    "openaiVersion": version("openai"),
                },
            }
        )
        + b"\n"
    )
    monkeypatch.setattr(
        certification_module,
        "_validate_reference_run",
        lambda *_args, **_kwargs: plan_root,
    )
    authority = CommittedArtifactDirectoryConfig(
        path="artifacts/benchmark-plan",
        commit_sha256="3" * 64,
        transaction_sha256="4" * 64,
    )
    return challenger.model_copy(update={"benchmark_plan": authority})


def test_benchmark_authority_admits_only_the_exact_medium_template(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    authorized = _authorized_medium_config(tmp_path=tmp_path, monkeypatch=monkeypatch)

    _validate_benchmark_plan_authorization(tmp_path, authorized)

    changed_run = authorized.run.model_copy(update={"run_id": "unplanned-medium"})
    with pytest.raises(ValueError, match="not an exact authorized benchmark arm"):
        _validate_benchmark_plan_authorization(
            tmp_path,
            authorized.model_copy(update={"run": changed_run}),
        )


def test_unauthorized_benchmark_config_fails_before_key_or_provider_construction(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import document_ocr.synthesis.raw_text_certification as certification_module

    authorized = _authorized_medium_config(tmp_path=tmp_path, monkeypatch=monkeypatch)
    config_path = tmp_path / "authorized-medium.yaml"
    config_path.write_text(
        yaml.safe_dump(_yaml_value(authorized.model_dump(mode="python")), sort_keys=False),
        encoding="utf-8",
    )
    key_loaded = False

    def reject_authority(*_args: object, **_kwargs: object) -> None:
        raise ValueError("not an exact authorized benchmark arm")

    def load_key(*_args: object, **_kwargs: object) -> str:
        nonlocal key_loaded
        key_loaded = True
        return "must-not-load"

    monkeypatch.setattr(
        certification_module,
        "_validate_benchmark_plan_authorization",
        reject_authority,
    )
    monkeypatch.setattr(certification_module, "load_provider_key", load_key)

    with pytest.raises(ValueError, match="not an exact authorized benchmark arm"):
        certification_module.run_raw_text_certification(
            project_root=tmp_path,
            config_path=config_path,
            config=authorized,
        )
    assert not key_loaded


def test_contract_v2_only_spends_high_reasoning_to_confirm_a_clean_screen(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import document_ocr.synthesis.raw_text_certification as certification_module

    document_id = "doc_" + "6" * 64
    source = "--- PAGE 1 ---\nOLD-123\nOLD CARRIER\nOLD CARRIER\n"
    current = "--- PAGE 1 ---\nNEW-123\nNEW CARRIER\nNEW CARRIER\n"
    contract = _contract()
    config = _v2_config(document_id)
    clean = _v2_output(contract=contract, source=source, current=current)
    rejecting = _v2_output(
        contract=contract,
        source=source,
        current=current,
        findings=(_finding("NEW-123"),),
    )
    supplied = clean
    efforts: list[str] = []

    async def fake_call_pass(
        **kwargs: Any,
    ) -> tuple[SemanticAuditOutput, tuple[Any, ...]]:
        efforts.append(kwargs["reasoning_effort"])
        return supplied, ()

    monkeypatch.setattr(certification_module, "_call_audit_pass", fake_call_pass)

    def invoke(*, host_findings: tuple[str, ...] = ()) -> None:
        asyncio.run(
            _call_audit(
                document_id=document_id,
                source=source,
                current=current,
                source_label=contract["sourceLabel"],  # type: ignore[arg-type]
                target_label=contract["targetLabel"],  # type: ignore[arg-type]
                contract=contract,
                host_findings=host_findings,
                model=cast(Any, object()),
                config=config,
                prompt="test prompt",
            )
        )

    invoke()
    assert efforts == ["low", "high"]
    efforts.clear()

    supplied = rejecting
    invoke()
    assert efforts == ["low"]
    efforts.clear()

    supplied = clean
    invoke(host_findings=("deterministic rejection",))
    assert efforts == ["low"]


def test_route_error_receipt_cannot_relabel_a_runtime_error_as_retryable() -> None:
    with pytest.raises(ValueError, match="category and concrete type disagree"):
        CertificationRouteError(
            category="model_api",
            errorType="RuntimeError",
            message="not a provider API error",
        )


def test_contract_v2_replay_requires_two_distinct_receipted_passes() -> None:
    document_id = "doc_" + "d" * 64
    source = "--- PAGE 1 ---\nOLD-123\nOLD CARRIER\nOLD CARRIER\n"
    current = "--- PAGE 1 ---\nNEW-123\nNEW CARRIER\nNEW CARRIER\n"
    contract = _contract()
    config = _v2_config(document_id)
    output = _v2_output(contract=contract, source=source, current=current)
    low = _v2_stage(
        document_id=document_id,
        audit_pass=1,
        reasoning_effort="low",
        response_id="response-low",
        source=source,
        current=current,
        contract=contract,
        output=output,
        config=config,
    )
    high = _v2_stage(
        document_id=document_id,
        audit_pass=2,
        reasoning_effort="high",
        response_id="response-high",
        source=source,
        current=current,
        contract=contract,
        output=output,
        config=config,
    )

    replay = replay_certification_audit_stages(
        document_id=document_id,
        stages=(low, high),
        source=source,
        current=current,
        contract=contract,
        audit_contract_version=2,
        config=config,
    )
    assert replay.audit_passes == replay.clean_audit_passes == 2
    assert replay.findings == ()

    with pytest.raises(ValueError, match="reuse a provider response"):
        duplicate_high = _v2_stage(
            document_id=document_id,
            audit_pass=2,
            reasoning_effort="high",
            response_id="response-low",
            source=source,
            current=current,
            contract=contract,
            output=output,
            config=config,
            attempt_identity="distinct-high-attempt",
        )
        replay_certification_audit_stages(
            document_id=document_id,
            stages=(low, duplicate_high),
            source=source,
            current=current,
            contract=contract,
            audit_contract_version=2,
            config=config,
        )

    with pytest.raises(ValueError, match="reuses a provider attempt"):
        validate_certification_run_stage_identities(
            {
                document_id: (low,),
                "doc_" + "c" * 64: (low,),
            }
        )


def test_contract_v2_replay_rejects_truncated_required_pass_and_route_attempts() -> None:
    document_id = "doc_" + "7" * 64
    source = "--- PAGE 1 ---\nOLD-123\nOLD CARRIER\nOLD CARRIER\n"
    current = "--- PAGE 1 ---\nNEW-123\nNEW CARRIER\nNEW CARRIER\n"
    contract = _contract()
    config = _v2_config(document_id)
    output = _v2_output(contract=contract, source=source, current=current)
    low = _v2_stage(
        document_id=document_id,
        audit_pass=1,
        reasoning_effort="low",
        response_id="response-low-only",
        source=source,
        current=current,
        contract=contract,
        output=output,
        config=config,
    )
    with pytest.raises(ValueError, match="required next pass"):
        replay_certification_audit_stages(
            document_id=document_id,
            stages=(low,),
            source=source,
            current=current,
            contract=contract,
            audit_contract_version=2,
            config=config,
        )

    retryable = _v2_error_stage(
        document_id=document_id,
        audit_pass=1,
        reasoning_effort="low",
        source=source,
        current=current,
        contract=contract,
        config=config,
        route_provider="deepinfra/fp4",
        route_attempt=1,
        retryable=True,
    )
    with pytest.raises(ValueError, match="exhausting active routes"):
        replay_certification_audit_stages(
            document_id=document_id,
            stages=(retryable,),
            source=source,
            current=current,
            contract=contract,
            audit_contract_version=2,
            config=config,
        )


def test_contract_v2_replay_binds_output_and_message_sequence() -> None:
    document_id = "doc_" + "8" * 64
    source = "--- PAGE 1 ---\nOLD-123\nOLD CARRIER\nOLD CARRIER\n"
    current = "--- PAGE 1 ---\nNEW-123\nNEW CARRIER\nNEW CARRIER\n"
    contract = _contract()
    config = _v2_config(document_id)
    clean = _v2_output(contract=contract, source=source, current=current)
    low = _v2_stage(
        document_id=document_id,
        audit_pass=1,
        reasoning_effort="low",
        response_id="response-bound",
        source=source,
        current=current,
        contract=contract,
        output=clean,
        config=config,
    )
    mismatched = _v2_output(
        contract=contract,
        source=source,
        current=current,
        findings=(_finding(),),
    )
    with pytest.raises(ValueError, match="differs from its native response"):
        replay_certification_audit_stages(
            document_id=document_id,
            stages=(low.model_copy(update={"modelOutput": mismatched.model_dump(mode="json")}),),
            source=source,
            current=current,
            contract=contract,
            audit_contract_version=2,
            config=config,
        )

    trailing_messages: Any = model_messages(
        (ModelRequest(parts=(UserPromptPart("impossible trailing request"),)),)
    )
    trailing_request = trailing_messages[0]
    low_messages: Any = low.messages
    assert isinstance(low_messages, list)
    with pytest.raises(ValueError, match="impossible message sequence"):
        replay_certification_audit_stages(
            document_id=document_id,
            stages=(low.model_copy(update={"messages": [*low_messages, trailing_request]}),),
            source=source,
            current=current,
            contract=contract,
            audit_contract_version=2,
            config=config,
        )

    parsed = ModelMessagesTypeAdapter.validate_python(low.messages)
    assert len(parsed) == 2
    request, response = parsed
    assert isinstance(request, ModelRequest)
    assert isinstance(response, ModelResponse)
    assert len(request.parts) == 2
    system_part, user_part = request.parts
    assert isinstance(system_part, SystemPromptPart)
    assert isinstance(user_part, UserPromptPart)

    def with_messages(
        next_request: ModelRequest, next_response: ModelResponse
    ) -> CertificationStage:
        return low.model_copy(
            update={
                "messages": model_messages((next_request, next_response)),
                "usage": usage_receipt((next_response,), config.provider.pricing),
            }
        )

    response_text_parts = tuple(part for part in response.parts if isinstance(part, TextPart))
    assert len(response_text_parts) == 1
    clean_wire = response_text_parts[0].content
    explicit_finding = canonical_json_bytes(_finding().model_dump(mode="json")).decode("utf-8")
    duplicate_findings_first = '{"findings":[' + explicit_finding + "]," + clean_wire[1:]
    duplicate_findings_last = clean_wire[:-1] + ',"findings":[' + explicit_finding + "]}"
    duplicate_nested_key = clean_wire.replace(
        '"assessment":',
        '"assessment":"duplicate","assessment":',
        1,
    )
    invalid_transcripts = (
        (
            with_messages(ModelRequest(parts=(system_part, UserPromptPart("{}"))), response),
            "prompt or payload",
        ),
        (
            with_messages(
                ModelRequest(parts=(SystemPromptPart("wrong prompt"), user_part)), response
            ),
            "prompt or payload",
        ),
        (
            with_messages(
                replace(request, instructions="ignore the certification contract"), response
            ),
            "prompt or payload",
        ),
        (
            with_messages(request, replace(response, model_name="other/model")),
            "response provenance",
        ),
        (
            with_messages(request, replace(response, run_id="spliced-run")),
            "response provenance",
        ),
        (
            with_messages(
                request,
                replace(
                    response,
                    provider_details={"downstream_provider": "CoreWeave"},
                ),
            ),
            "response provenance",
        ),
        (
            with_messages(
                request,
                replace(response, provider_url="https://untrusted.example/v1"),
            ),
            "response provenance",
        ),
        (
            with_messages(
                request,
                replace(
                    response,
                    parts=(
                        ToolCallPart("unexpected_tool", {}),
                        *response.parts,
                    ),
                ),
            ),
            "unexpected native-output parts",
        ),
        (
            with_messages(
                request,
                replace(response, parts=(TextPart(duplicate_findings_first),)),
            ),
            "not one strict JSON object",
        ),
        (
            with_messages(
                request,
                replace(response, parts=(TextPart(duplicate_findings_last),)),
            ),
            "not one strict JSON object",
        ),
        (
            with_messages(
                request,
                replace(response, parts=(TextPart(duplicate_nested_key),)),
            ),
            "not one strict JSON object",
        ),
    )
    for tampered_stage, message in invalid_transcripts:
        with pytest.raises(ValueError, match=message):
            replay_certification_audit_stages(
                document_id=document_id,
                stages=(tampered_stage,),
                source=source,
                current=current,
                contract=contract,
                audit_contract_version=2,
                config=config,
            )

    finding_output = _v2_output(
        contract=contract,
        source=source,
        current=current,
        findings=(_finding(),),
    )
    finding_stage = _v2_stage(
        document_id=document_id,
        audit_pass=1,
        reasoning_effort="low",
        response_id="response-cannot-discard",
        source=source,
        current=current,
        contract=contract,
        output=finding_output,
        config=config,
    )
    fabricated_error = _route_error_receipt(UnexpectedModelBehavior("fabricated parse failure"))
    with pytest.raises(ValueError, match="discards a valid native audit response"):
        replay_certification_audit_stages(
            document_id=document_id,
            stages=(
                finding_stage.model_copy(
                    update={
                        "modelOutput": None,
                        "errorType": fabricated_error.errorType,
                        "errorMessage": fabricated_error.message,
                        "retryableRouteError": True,
                        "routeError": fabricated_error,
                    }
                ),
            ),
            source=source,
            current=current,
            contract=contract,
            audit_contract_version=2,
            config=config,
        )

    with pytest.raises(ValueError, match="invents a host rejection"):
        replay_certification_audit_stages(
            document_id=document_id,
            stages=(low.model_copy(update={"hostError": "invented rejection"}),),
            source=source,
            current=current,
            contract=contract,
            audit_contract_version=2,
            config=config,
        )


def test_contract_v2_replay_accepts_exact_host_rejection_then_next_route() -> None:
    document_id = "doc_" + "6" * 64
    source = "--- PAGE 1 ---\nOLD-123\nOLD CARRIER\nOLD CARRIER\n"
    current = "--- PAGE 1 ---\nNEW-123\nNEW CARRIER\nNEW CARRIER\n"
    contract = _contract()
    config = _v2_config(document_id)
    rejected_output = _v2_output(
        contract=contract,
        source=source,
        current=current,
        findings=(_finding("--- PAGE 1 ---", "L00001"),),
    )
    with pytest.raises(ValueError) as caught:
        _validate_audit_output(
            rejected_output,
            current=current,
            expected_coverage=_audit_expected_coverage(
                _audit_plan(contract, source=source, current=current)
            ),
        )
    rejected = _v2_stage(
        document_id=document_id,
        audit_pass=1,
        reasoning_effort="low",
        response_id="response-host-rejected",
        source=source,
        current=current,
        contract=contract,
        output=rejected_output,
        config=config,
    ).model_copy(update={"hostError": str(caught.value)})
    clean_output = _v2_output(contract=contract, source=source, current=current)
    accepted_low = _v2_stage(
        document_id=document_id,
        audit_pass=1,
        reasoning_effort="low",
        response_id="response-low-after-host-reject",
        source=source,
        current=current,
        contract=contract,
        output=clean_output,
        config=config,
        route_provider="coreweave/fp8",
        route_attempt=2,
        started_at=2.0,
    )
    accepted_high = _v2_stage(
        document_id=document_id,
        audit_pass=2,
        reasoning_effort="high",
        response_id="response-high-after-host-reject",
        source=source,
        current=current,
        contract=contract,
        output=clean_output,
        config=config,
        started_at=3.0,
    )
    replay = replay_certification_audit_stages(
        document_id=document_id,
        stages=(rejected, accepted_low, accepted_high),
        source=source,
        current=current,
        contract=contract,
        audit_contract_version=2,
        config=config,
    )
    assert replay.audit_passes == replay.clean_audit_passes == 2


def test_legacy_result_rejects_contract_v2_outcome_metadata() -> None:
    source = "--- PAGE 1 ---\nOLD-123\nOLD CARRIER\nOLD CARRIER\n"
    current = "--- PAGE 1 ---\nNEW-123\nNEW CARRIER\nNEW CARRIER\n"
    with pytest.raises(ValueError, match="contract-v2 outcome metadata"):
        CertificationCaseResult(
            documentId="doc_" + "9" * 64,
            status="evaluated_clean",
            reason="Forged mixed-era result.",
            auditPasses=2,
            cleanAuditPasses=2,
            requiredAuditPasses=2,
            semanticFindings=0,
            candidateImmutable=True,
            hostAudit=_host_audit(source=source, output=current, contract=_contract()),
            sourceTextSha256=sha256_bytes(source.encode()),
            inputCandidateSha256=sha256_bytes(current.encode()),
            finalTextSha256=sha256_bytes(current.encode()),
            usage=_empty_usage(),
        )


def test_case_contract_labels_must_equal_the_payload_files() -> None:
    contract = _contract()
    source_label = contract["sourceLabel"]
    target_label = contract["targetLabel"]
    assert isinstance(source_label, dict)
    assert isinstance(target_label, dict)
    _validate_case_contract_labels(
        document_id="doc_" + "a" * 64,
        contract=contract,
        source_label=source_label,
        target_label=target_label,
    )
    with pytest.raises(ValueError, match="labels differ from its embedded contract"):
        _validate_case_contract_labels(
            document_id="doc_" + "a" * 64,
            contract=contract,
            source_label={"documentPatch": {"billOfLadingNumber": "TAMPERED"}},
            target_label=target_label,
        )


def test_contract_v2_replay_rejects_impossible_or_tampered_stage_topology() -> None:
    document_id = "doc_" + "e" * 64
    source = "--- PAGE 1 ---\nOLD-123\nOLD CARRIER\nOLD CARRIER\n"
    current = "--- PAGE 1 ---\nNEW-123\nNEW CARRIER\nNEW CARRIER\n"
    contract = _contract()
    config = _v2_config(document_id)
    output = _v2_output(contract=contract, source=source, current=current)
    low = _v2_stage(
        document_id=document_id,
        audit_pass=1,
        reasoning_effort="low",
        response_id="response-low",
        source=source,
        current=current,
        contract=contract,
        output=output,
        config=config,
    )
    high = _v2_stage(
        document_id=document_id,
        audit_pass=2,
        reasoning_effort="high",
        response_id="response-high",
        source=source,
        current=current,
        contract=contract,
        output=output,
        config=config,
    )
    second_route_failure = _v2_error_stage(
        document_id=document_id,
        audit_pass=1,
        reasoning_effort="low",
        source=source,
        current=current,
        contract=contract,
        config=config,
        route_provider="coreweave/fp8",
        route_attempt=2,
        retryable=False,
    )

    for stages, message in (
        ((), "no provider audit attempt"),
        ((low, second_route_failure, high), "not terminal"),
        (
            (low.model_copy(update={"inputPayloadSha256": "0" * 64}), high),
            "payload or schema identity differs",
        ),
        (
            (low.model_copy(update={"routeProvider": "unconfigured"}), high),
            "unevaluated provider route",
        ),
        (
            (low.model_copy(update={"retryDelayBeforeSeconds": 1.0}), high),
            "retry delay differs",
        ),
    ):
        with pytest.raises(ValueError, match=message):
            replay_certification_audit_stages(
                document_id=document_id,
                stages=stages,
                source=source,
                current=current,
                contract=contract,
                audit_contract_version=2,
                config=config,
            )


def test_read_only_audit_materializes_exact_candidate_evidence_from_line_ids() -> None:
    source = "--- PAGE 1 ---\nOLD-123\nNEW CARRIER\n"
    current = "--- PAGE 1 ---\nOLD-123\nNEW CARRIER\n"
    contract = _contract()
    coverage = _audit_expected_coverage(_audit_plan(contract, source=source, current=current))
    finding = _finding()
    assert _validate_audit_output(
        _v2_output(contract=contract, source=source, current=current, findings=(finding,)),
        current=current,
        expected_coverage=coverage,
    ) == (finding,)

    copied_wrong = _finding("MISSING")
    assert _validate_audit_output(
        _v2_output(
            contract=contract,
            source=source,
            current=current,
            findings=(copied_wrong,),
        ),
        current=current,
        expected_coverage=coverage,
    ) == (_finding(),)

    for invalid, message in (
        (_finding("--- PAGE 1 ---", "L00001"), "blank/page-marker"),
        (_finding("OLD-123", "L00005"), "outside the document"),
    ):
        with pytest.raises(ValueError, match=message):
            _validate_audit_output(
                _v2_output(
                    contract=contract,
                    source=source,
                    current=current,
                    findings=(invalid,),
                ),
                current=current,
                expected_coverage=coverage,
            )


def test_read_only_audit_normalizes_duplicate_findings() -> None:
    source = "--- PAGE 1 ---\nOLD-123\n"
    current = source
    contract = _contract()
    coverage = _audit_expected_coverage(_audit_plan(contract, source=source, current=current))
    finding = _finding()
    assert _validate_audit_output(
        _v2_output(
            contract=contract,
            source=source,
            current=current,
            findings=(finding, finding),
        ),
        current=current,
        expected_coverage=coverage,
    ) == (finding,)


def test_read_only_audit_requires_exact_dimension_attestations() -> None:
    text = "--- PAGE 1 ---\nNEW-123\nNEW CARRIER\nNEW CARRIER\n"
    contract = _contract()
    plan = _audit_plan(contract, source=text, current=text)
    coverage = _audit_expected_coverage(plan)
    output = _v2_output(contract=contract, source=text, current=text)

    assert (
        _validate_audit_output(
            output,
            current=text,
            expected_coverage=coverage,
        )
        == ()
    )

    duplicate = output.model_dump(mode="python")
    duplicate["dimensionChecks"][0]["dimension"] = duplicate["dimensionChecks"][1]["dimension"]
    with pytest.raises(ValueError, match="dimensions are missing, repeated, or out of order"):
        SemanticAuditOutput.model_validate(duplicate, strict=True)


def test_read_only_audit_schema_cannot_express_dangling_finding_links() -> None:
    text = "--- PAGE 1 ---\nNEW-123\nNEW CARRIER\nNEW CARRIER\n"
    output = _v2_output(contract=_contract(), source=text, current=text).model_dump(mode="python")
    output["findingLinks"] = ({"findingIndex": 0, "obligationIds": ("D0001",)},)

    with pytest.raises(ValidationError, match="findingLinks"):
        SemanticAuditOutput.model_validate(output, strict=True)

    schema = SemanticAuditOutput.model_json_schema()
    assert "findingLinks" not in schema["properties"]
    finding_schema = schema["$defs"]["ContractV2SemanticAuditFinding"]
    assert "obligationIds" in finding_schema["required"]
    evidence_schema = schema["$defs"]["ContractV2SemanticAuditEvidence"]
    assert evidence_schema["required"] == ["lineId"]
    assert "currentFragment" not in evidence_schema["properties"]


def test_read_only_audit_rejects_invented_linked_obligation_identity() -> None:
    text = "--- PAGE 1 ---\nOLD-123\nNEW CARRIER\nNEW CARRIER\n"
    contract = _contract()
    coverage = _audit_expected_coverage(_audit_plan(contract, source=text, current=text))
    output = _v2_output(
        contract=contract,
        source=text,
        current=text,
        findings=(_finding(),),
    ).model_dump(mode="python")
    output["findings"][0]["obligationIds"] = ("D9999",)

    with pytest.raises(ValueError, match="links an absent obligation"):
        _validate_audit_output(
            SemanticAuditOutput.model_validate(output, strict=True),
            current=text,
            expected_coverage=coverage,
        )


def test_read_only_audit_does_not_treat_model_category_as_host_proof() -> None:
    text = "--- PAGE 1 ---\nOLD-123\nNEW CARRIER\nNEW CARRIER\n"
    contract = _contract()
    coverage = _audit_expected_coverage(_audit_plan(contract, source=text, current=text))
    output = _v2_output(
        contract=contract,
        source=text,
        current=text,
        findings=(_finding(),),
    ).model_dump(mode="python")
    output["findings"][0]["obligationIds"] = (coverage["equipment_temperature_and_capacity"][0],)
    assert _validate_audit_output(
        SemanticAuditOutput.model_validate(output, strict=True),
        current=text,
        expected_coverage=coverage,
    ) == (_finding(),)


def test_read_only_audit_accepts_one_repair_linked_across_dimensions() -> None:
    text = "--- PAGE 1 ---\nOLD-123\nNEW CARRIER\nNEW CARRIER\n"
    contract = _contract()
    coverage = _audit_expected_coverage(_audit_plan(contract, source=text, current=text))
    output = _v2_output(
        contract=contract,
        source=text,
        current=text,
        findings=(_finding(),),
    ).model_dump(mode="python")
    route_id = coverage["route_jurisdiction_and_identifiers"][0]
    output["findings"][0]["obligationIds"] = (
        *output["findings"][0]["obligationIds"],
        route_id,
    )
    parsed = SemanticAuditOutput.model_validate(output, strict=True)

    assert _validate_audit_output(parsed, current=text, expected_coverage=coverage) == (_finding(),)


def test_payload_exposes_each_source_candidate_line_once() -> None:
    payload = _payload(
        document_id="doc_" + "a" * 64,
        source="--- PAGE 1 ---\nOLD\nKEEP\n",
        current="--- PAGE 1 ---\nNEW\nKEEP\n",
        source_label={"documentPatch": {"billOfLadingNumber": "OLD"}},
        target_label={"documentPatch": {"billOfLadingNumber": "NEW"}},
        host_findings=("known host failure",),
        audit_contract_version=1,
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
    contract: dict[str, Any] = {
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
    contract: dict[str, Any] = {
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
    config = load_synthesis_raw_text_certification_config(
        _PROJECT_ROOT
        / "configs/synthesis/mpci_bl_raw_text_certification5_glm53_v24_third_residuals.yaml"
    )
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
        config=config,
        stages=(),
        result=result,
    )
    loaded = _load_case_checkpoint(
        staged=staged,
        document_id=document_id,
        source=source,
        candidate=candidate,
        contract=contract,
        config=config,
    )
    assert loaded == (result, ())
    value = json.loads((staged.stage_root / f"cases/{document_id}/checkpoint.json").read_text())
    assert value["sourceContractSha256"] == sha256_bytes(canonical_json_bytes(contract))
