"""Read-only semantic certification for synthetic raw-OCR candidates.

The renderer/compiler owns every mutable byte.  This stage owns none: one independent,
provider-native structured call may only report exact evidence from the candidate.  The host
validates every cited fragment against the immutable candidate and combines those findings with
deterministic topology, target-surface, and source-leakage checks.  A model can therefore prevent
publication, but it can never alter a document or expand write authority.
"""

from __future__ import annotations

import asyncio
import json
import re
import time
from collections.abc import Awaitable, Callable, Mapping, Sequence
from dataclasses import dataclass
from decimal import Decimal
from importlib.metadata import version
from pathlib import Path, PurePosixPath
from typing import Annotated, Any, Literal, cast

from openai import AsyncOpenAI
from pydantic import BaseModel, ConfigDict, Field, JsonValue, StringConstraints, model_validator
from pydantic_ai import Agent, NativeOutput, capture_run_messages
from pydantic_ai.exceptions import ModelAPIError, ModelHTTPError, UnexpectedModelBehavior
from pydantic_ai.messages import (
    ModelMessagesTypeAdapter,
    ModelRequest,
    ModelResponse,
    SystemPromptPart,
    TextPart,
    ThinkingPart,
    UserPromptPart,
)
from pydantic_ai.models import Model
from pydantic_ai.usage import UsageLimits

from document_ocr.atomic import read_regular_file_bytes
from document_ocr.hashing import canonical_json_bytes, sha256_bytes, sha256_file
from document_ocr.synthesis.config import (
    SynthesisRawTextCertificationConfig,
    load_synthesis_raw_text_certification_config,
)
from document_ocr.synthesis.linguistic_probe_runtime import (
    LinguisticUsageReceipt,
    load_provider_key,
    model_messages,
    usage_receipt,
)
from document_ocr.synthesis.raw_text_certification_host import (
    CertificationInvariantCase,
    load_certification_invariant_case,
    load_certification_invariant_context,
)
from document_ocr.synthesis.raw_text_hybrid_probe import (
    _artifact_inventory,
    _model_pair,
    _provider_attempt_routes,
    _retryable_route_error,
)
from document_ocr.synthesis.raw_text_inventory import locate_auxiliary_values
from document_ocr.synthesis.raw_text_inventory_probe import _validate_reference_run
from document_ocr.synthesis.raw_text_rewrite_cycle_probe import (
    AnchoredScalarReplacementRequirement,
    CargoFlavorRewriteRequirement,
    CompactLabelChangeDirective,
    CompoundPartyFlavorRequirement,
    InlineSlotTopologyRequirement,
    JurisdictionalSurfaceRequirement,
    OperationalFlavorRequirement,
    RawAuxiliaryIdentityRequirement,
    SourceSemanticRoleHint,
    SourceStatusPreservationRequirement,
    SurfaceRenderingRequirement,
    TargetLiteralRequirement,
    TargetValueOccurrenceRequirement,
    _combine_usage,
    _missing_target_literals,
    _settings,
    _target_value_occurrence_count,
)
from document_ocr.synthesis.raw_text_rewrite_probe import _resolve_pinned_file, unified_text_diff
from document_ocr.synthesis.run_safety import StagedArtifactRun
from document_ocr.training.config import resolve_config_path

NonEmptyText = Annotated[str, StringConstraints(strip_whitespace=True, min_length=1)]
LineId = Annotated[str, StringConstraints(pattern=r"^L[0-9]{5}$")]
AuditObligationId = Annotated[str, StringConstraints(pattern=r"^[A-Z][0-9]{4}$")]
Sha256 = Annotated[str, StringConstraints(pattern=r"^[0-9a-f]{64}$")]
_STRICT = ConfigDict(extra="forbid", frozen=True, strict=True, allow_inf_nan=False)
_IMPLEMENTATION_PATH = Path(__file__).resolve(strict=True)
_DEPENDENCY_IMPLEMENTATION_FILES = {
    "certificationHost": "raw_text_certification_host.py",
    "certificationInvariants": "raw_text_certification_invariants.py",
    "certificationReferences": "raw_text_certification_references.py",
    "containerSemantics": "container_semantics.py",
    "providerRuntime": "linguistic_probe_runtime.py",
    "hybridRuntime": "raw_text_hybrid_probe.py",
    "inventory": "raw_text_inventory.py",
    "inventoryRunner": "raw_text_inventory_probe.py",
    "rewriteContract": "raw_text_rewrite_cycle_probe.py",
    "rewriteRuntime": "raw_text_rewrite_probe.py",
    "transportCapacity": "transport_capacity.py",
}
_OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"
_OPENROUTER_DOWNSTREAM_BY_ROUTE = {
    "gmicloud/fp8": "GMICloud",
    "deepinfra/fp4": "DeepInfra",
    "coreweave/fp8": "CoreWeave",
    "fireworks": "Fireworks",
    "nextbit/fp8": "NextBit",
}
_PAGE_MARKER = re.compile(r"^--- PAGE [1-9][0-9]* ---[ \t]*$")
_MODEL_META = re.compile(
    r"(?i)\b(?:previous turn|the user(?:'s)? (?:message|prompt)|host asks|correction patch|"
    r"re-submit|as an ai|i (?:must|need to|should) (?:return|output|submit)|"
    r"valid patch object|accidental trailing junk)\b"
)
_NEW_PLACEHOLDER = re.compile(r"(?i)(?<![A-Z0-9])(?:N/A|UNKNOWN|UNAVAILABLE|NULL)(?![A-Z0-9])")
_INTERNAL_CATEGORY = re.compile(
    r"\b(?:PACKAGE_(?:BAG|BALE|BOX|CARTON|CASE|CRATE|DRUM|PACKAGE|PALLET|PIECE|ROLL|SACK)|"
    r"FORTY_FOOT_HIGH_CUBE|FORTY_FOOT_STANDARD_HEIGHT|TWENTY_FOOT_STANDARD_HEIGHT|"
    r"GENERAL_PURPOSE)\b"
)
_EVIDENCE_TOKEN = re.compile(r"[^\W_]+", flags=re.UNICODE)
_CARGO_DESCRIPTION_PATH = re.compile(r"^documentPatch\.cargoGroups\[[0-9]+\]\.description$")
AuditDimension = Literal[
    "target_fact_fidelity",
    "repeated_and_derived_relations",
    "source_private_and_auxiliary_replacement",
    "party_legal_and_negotiability",
    "route_jurisdiction_and_identifiers",
    "equipment_temperature_and_capacity",
    "cargo_packages_and_dangerous_goods",
    "template_format_and_model_artifacts",
]
AuditReasoningEffort = Literal["none", "minimal", "low", "medium", "high", "xhigh", "max"]
_AUDIT_DIMENSIONS: tuple[AuditDimension, ...] = (
    "target_fact_fidelity",
    "repeated_and_derived_relations",
    "source_private_and_auxiliary_replacement",
    "party_legal_and_negotiability",
    "route_jurisdiction_and_identifiers",
    "equipment_temperature_and_capacity",
    "cargo_packages_and_dangerous_goods",
    "template_format_and_model_artifacts",
)
_FINDING_DIMENSIONS: dict[str, AuditDimension] = {
    "target_fact_mismatch": "target_fact_fidelity",
    "repeated_or_derived_fact_mismatch": "repeated_and_derived_relations",
    "source_only_private_or_auxiliary_fact": "source_private_and_auxiliary_replacement",
    "party_or_legal_identity_mismatch": "party_legal_and_negotiability",
    "route_or_jurisdiction_mismatch": "route_jurisdiction_and_identifiers",
    "equipment_or_temperature_mismatch": "equipment_temperature_and_capacity",
    "cargo_or_dangerous_goods_mismatch": "cargo_packages_and_dangerous_goods",
    "format_or_model_artifact": "template_format_and_model_artifacts",
}


def certification_implementation_contract() -> dict[str, JsonValue]:
    """Return the exact implementation identities that own certification behavior."""

    return {
        "implementationSha256": sha256_file(_IMPLEMENTATION_PATH),
        "dependencyImplementationSha256": {
            name: sha256_file(Path(__file__).with_name(filename))
            for name, filename in _DEPENDENCY_IMPLEMENTATION_FILES.items()
        },
    }


class CertificationBenchmarkPlanArmAuthorization(BaseModel):
    """One exact evaluation template admitted by a committed benchmark plan."""

    model_config = _STRICT

    role: Literal["baseline", "challenger"]
    effort: Literal["low", "medium", "high"]
    templateConfigPath: NonEmptyText
    templateConfigSha256: Sha256
    templateResolvedConfigSha256: Sha256
    runId: NonEmptyText

    @model_validator(mode="after")
    def template_path_is_safe(self) -> CertificationBenchmarkPlanArmAuthorization:
        path = PurePosixPath(self.templateConfigPath)
        if (
            path.is_absolute()
            or any(part in {"", ".", ".."} for part in path.parts)
            or "\x00" in self.templateConfigPath
        ):
            raise ValueError("benchmark authorization template path is unsafe")
        return self


class CertificationBenchmarkPlanAuthorization(BaseModel):
    """Plan-owned membership proof checked before secrets or provider construction."""

    model_config = _STRICT

    schemaVersion: Literal[1]
    task: Literal["raw_text_certification_benchmark_plan_authorization_v1"]
    arms: Annotated[
        tuple[CertificationBenchmarkPlanArmAuthorization, ...],
        Field(min_length=2, max_length=2),
    ]

    @model_validator(mode="after")
    def roles_and_efforts_are_ordered(self) -> CertificationBenchmarkPlanAuthorization:
        if tuple(row.role for row in self.arms) != ("baseline", "challenger"):
            raise ValueError("benchmark authorization arms are not baseline then challenger")
        effort_rank = {"low": 0, "medium": 1, "high": 2}
        if effort_rank[self.arms[0].effort] >= effort_rank[self.arms[1].effort]:
            raise ValueError("benchmark authorization challenger effort is not higher")
        if self.arms[0].runId == self.arms[1].runId:
            raise ValueError("benchmark authorization reuses an arm run identity")
        return self


def _validate_benchmark_plan_authorization(
    project_root: Path,
    config: SynthesisRawTextCertificationConfig,
) -> None:
    """Prove exact plan membership before loading a key or constructing a provider client."""

    authority = config.benchmark_plan
    if authority is None:
        return
    plan_root = _validate_reference_run(
        project_root,
        authority.path,
        authority.commit_sha256,
        authority.transaction_sha256,
    )
    authorization_path = plan_root / "authorization/arm-configs.json"
    plan_config_path = plan_root / "config.json"
    transaction_path = plan_root / "provenance/transaction.json"
    for path in (authorization_path, plan_config_path, transaction_path):
        if path.is_symlink() or not path.is_file():
            raise ValueError(f"benchmark plan lacks a regular authority artifact: {path}")
    authorization = CertificationBenchmarkPlanAuthorization.model_validate_json(
        read_regular_file_bytes(authorization_path),
        strict=True,
    )
    plan_config = _read_json_object(plan_config_path)
    transaction = _read_json_object(transaction_path)
    expected_runtime = {
        "pydanticAiVersion": version("pydantic-ai-slim"),
        "openaiVersion": version("openai"),
    }
    if (
        plan_config.get("schemaVersion") != 2
        or plan_config.get("task") != "raw_text_certification_contract_v2_benchmark_plan_v2"
        or transaction.get("certificationImplementation") != certification_implementation_contract()
        or transaction.get("certificationRuntime") != expected_runtime
    ):
        raise ValueError("benchmark plan authority differs from the current certification runtime")

    stripped_config = config.model_copy(update={"benchmark_plan": None})
    matching_roles: list[str] = []
    for arm in authorization.arms:
        arm_specification = plan_config.get(arm.role)
        expected_arm_specification = {
            "effort": arm.effort,
            "config": {
                "path": arm.templateConfigPath,
                "sha256": arm.templateConfigSha256,
            },
        }
        template_path = plan_root / f"inputs/{arm.role}-config.yaml"
        if (
            arm_specification != expected_arm_specification
            or template_path.is_symlink()
            or not template_path.is_file()
            or sha256_file(template_path) != arm.templateConfigSha256
        ):
            raise ValueError("benchmark plan arm authorization differs from its committed template")
        template = load_synthesis_raw_text_certification_config(template_path)
        template_resolved_sha256 = sha256_bytes(
            canonical_json_bytes(template.model_dump(mode="json"))
        )
        if (
            template.benchmark_plan is not None
            or template.provider.reasoning_effort != arm.effort
            or template.run.run_id != arm.runId
            or template_resolved_sha256 != arm.templateResolvedConfigSha256
        ):
            raise ValueError("benchmark plan template differs from its authorization receipt")
        if stripped_config == template:
            matching_roles.append(arm.role)
    if len(matching_roles) != 1:
        raise ValueError("certification configuration is not an exact authorized benchmark arm")


class SemanticAuditEvidence(BaseModel):
    """An exact candidate fragment supporting one semantic finding."""

    model_config = _STRICT

    lineId: LineId
    currentFragment: Annotated[str, StringConstraints(min_length=1)]


class SemanticAuditFinding(BaseModel):
    """One read-only semantic defect; it contains no replacement or write instruction."""

    model_config = _STRICT

    findingKind: Literal[
        "target_fact_mismatch",
        "repeated_or_derived_fact_mismatch",
        "source_only_private_or_auxiliary_fact",
        "party_or_legal_identity_mismatch",
        "route_or_jurisdiction_mismatch",
        "equipment_or_temperature_mismatch",
        "cargo_or_dangerous_goods_mismatch",
        "format_or_model_artifact",
    ]
    evidence: Annotated[tuple[SemanticAuditEvidence, ...], Field(min_length=1)]
    problem: NonEmptyText


class LegacySemanticAuditOutput(BaseModel):
    """Historical finding-only contract used by immutable certification artifacts."""

    model_config = _STRICT

    findings: Annotated[tuple[SemanticAuditFinding, ...], Field(max_length=120)]


class ContractV2SemanticAuditEvidence(BaseModel):
    """One immutable candidate line selected by the model and materialized by the host."""

    model_config = _STRICT

    lineId: LineId


class ContractV2SemanticAuditFinding(BaseModel):
    """One defect with citations and failed obligations embedded in one atomic row."""

    model_config = _STRICT

    findingKind: Literal[
        "target_fact_mismatch",
        "repeated_or_derived_fact_mismatch",
        "source_only_private_or_auxiliary_fact",
        "party_or_legal_identity_mismatch",
        "route_or_jurisdiction_mismatch",
        "equipment_or_temperature_mismatch",
        "cargo_or_dangerous_goods_mismatch",
        "format_or_model_artifact",
    ]
    evidence: Annotated[tuple[ContractV2SemanticAuditEvidence, ...], Field(min_length=1)]
    problem: NonEmptyText

    obligationIds: Annotated[tuple[AuditObligationId, ...], Field(min_length=1, max_length=1000)]


class SemanticAuditDimensionCheck(BaseModel):
    """One mandatory attestation over a disjoint semantic audit dimension."""

    model_config = _STRICT

    dimension: AuditDimension
    assessment: NonEmptyText


class SemanticAuditOutput(BaseModel):
    """Complete contract-v2 coverage receipt plus exact-evidence defects."""

    model_config = _STRICT

    auditContractVersion: Literal[2]
    dimensionChecks: Annotated[
        tuple[SemanticAuditDimensionCheck, ...],
        Field(min_length=len(_AUDIT_DIMENSIONS), max_length=len(_AUDIT_DIMENSIONS)),
    ]
    findings: Annotated[tuple[ContractV2SemanticAuditFinding, ...], Field(max_length=120)]

    @model_validator(mode="after")
    def dimensions_are_internally_complete(self) -> SemanticAuditOutput:
        if tuple(row.dimension for row in self.dimensionChecks) != _AUDIT_DIMENSIONS:
            raise ValueError("semantic audit dimensions are missing, repeated, or out of order")
        return self


AuditOutput = LegacySemanticAuditOutput | SemanticAuditOutput


@dataclass(frozen=True, slots=True)
class CertificationAuditReplay:
    outputs: tuple[AuditOutput, ...]
    findings: tuple[SemanticAuditFinding, ...]
    audit_passes: int
    clean_audit_passes: int
    audit_plan_sha256: str | None


class CertificationHostAudit(BaseModel):
    model_config = _STRICT

    passed: bool
    findings: tuple[NonEmptyText, ...]
    sourceLines: Annotated[int, Field(gt=0)]
    outputLines: Annotated[int, Field(gt=0)]
    targetLiteralRequirements: Annotated[int, Field(ge=0)]
    targetOccurrenceRequirements: Annotated[int, Field(ge=0)]

    @model_validator(mode="after")
    def verdict_matches_findings(self) -> CertificationHostAudit:
        if self.passed != (not self.findings):
            raise ValueError("certification host verdict and findings disagree")
        return self


class CertificationRouteError(BaseModel):
    model_config = _STRICT

    category: Literal["unexpected_model_behavior", "model_http", "model_api", "other"]
    errorType: NonEmptyText
    message: str
    statusCode: Annotated[int, Field(ge=100, le=599)] | None = None
    failedRoutingStep: str | None = None

    @model_validator(mode="after")
    def category_fields_are_consistent(self) -> CertificationRouteError:
        expected_types = {
            "unexpected_model_behavior": {
                "UnexpectedModelBehavior",
                "ContentFilterError",
                "IncompleteToolCall",
            },
            "model_http": {"ModelHTTPError"},
            "model_api": {"ModelAPIError"},
        }
        reserved_types = set().union(*expected_types.values())
        if self.category == "other":
            if self.errorType in reserved_types:
                raise ValueError("other certification error claims a reserved error type")
        elif self.errorType not in expected_types[self.category]:
            raise ValueError("certification error category and concrete type disagree")
        if self.category == "model_http":
            if self.statusCode is None:
                raise ValueError("HTTP certification error lacks its status code")
        elif self.statusCode is not None or self.failedRoutingStep is not None:
            raise ValueError("non-HTTP certification error has HTTP routing metadata")
        return self


class CertificationStage(BaseModel):
    model_config = _STRICT

    auditContractVersion: Literal[1, 2, 3] = 1
    auditPass: Literal[1, 2]
    reasoningEffort: AuditReasoningEffort | None = None
    auditPlanSha256: Sha256 | None = None
    routeProvider: str | None
    routeRound: Annotated[int, Field(ge=1)]
    routeAttempt: Annotated[int, Field(ge=1)]
    retryDelayBeforeSeconds: Annotated[float, Field(ge=0)]
    inputPayloadSha256: Sha256
    outputSchemaSha256: Sha256
    startedAtUnixSeconds: float
    completedAtUnixSeconds: float
    usage: LinguisticUsageReceipt
    messages: JsonValue
    modelOutput: JsonValue | None
    hostError: str | None
    errorType: str | None
    errorMessage: str | None
    retryableRouteError: bool | None = None
    routeError: CertificationRouteError | None = None

    @model_validator(mode="after")
    def outcome_and_timing_are_consistent(self) -> CertificationStage:
        if self.completedAtUnixSeconds < self.startedAtUnixSeconds:
            raise ValueError("certification stage completes before it starts")
        if self.auditContractVersion == 1:
            if self.auditPass != 1 or self.auditPlanSha256 is not None:
                raise ValueError("legacy certification stage has contract-v2 metadata")
        elif self.auditPlanSha256 is None or self.reasoningEffort not in {
            "low",
            "medium",
            "high",
        }:
            raise ValueError("contract-v2 certification stage lacks its plan or allowed effort")
        if self.hostError is not None:
            if (
                self.modelOutput is None
                or self.errorType is not None
                or self.errorMessage is not None
            ):
                raise ValueError("host-rejected certification stage has contradictory outcomes")
        elif self.modelOutput is not None:
            if self.errorType is not None or self.errorMessage is not None:
                raise ValueError("successful certification stage also records a provider error")
        elif self.errorType is None or self.errorMessage is None:
            raise ValueError("failed certification stage has an incomplete error receipt")
        if self.auditContractVersion in {2, 3}:
            if self.errorType is None:
                if self.retryableRouteError is not None or self.routeError is not None:
                    raise ValueError("non-provider-error stage has route-error metadata")
            elif (
                self.retryableRouteError is None
                or self.routeError is None
                or self.routeError.errorType != self.errorType
                or self.routeError.message != self.errorMessage
                or (
                    self.retryableRouteError
                    and not _route_error_receipt_is_retryable(self.routeError)
                )
            ):
                raise ValueError("provider-error stage has inconsistent route-error metadata")
        if (
            self.auditContractVersion in {2, 3}
            and self.modelOutput is not None
            and (
                self.usage.requests != 1
                or len(self.usage.providerResponseIds) != 1
                or not isinstance(self.messages, list)
                or not self.messages
            )
        ):
            raise ValueError("model-returned certification stage lacks one receipted response")
        return self


class CertificationCaseResult(BaseModel):
    model_config = _STRICT

    documentId: str
    auditContractVersion: Literal[1, 2, 3] = 1
    certificationMode: Literal["legacy", "evaluation", "production"] = "legacy"
    auditPlanSha256: Sha256 | None = None
    invariantEnvelopeSha256: Sha256 | None = None
    invariantAuditSha256: Sha256 | None = None
    status: Literal["certified", "evaluated_clean", "needs_review", "call_failed"]
    reason: NonEmptyText
    auditPasses: Annotated[int, Field(ge=0, le=2)]
    cleanAuditPasses: Annotated[int, Field(ge=0, le=2)] = 0
    requiredAuditPasses: Annotated[int, Field(ge=1, le=2)] = 1
    semanticFindings: Annotated[int, Field(ge=0)]
    deterministicFindings: Annotated[int, Field(ge=0)] = 0
    candidateImmutable: Literal[True]
    hostAudit: CertificationHostAudit
    sourceTextSha256: Sha256
    inputCandidateSha256: Sha256
    finalTextSha256: Sha256
    usage: LinguisticUsageReceipt

    @model_validator(mode="after")
    def immutable_candidate_and_v2_outcome_are_consistent(self) -> CertificationCaseResult:
        if self.inputCandidateSha256 != self.finalTextSha256:
            raise ValueError("read-only certification changed the candidate identity")
        if self.cleanAuditPasses > self.auditPasses:
            raise ValueError("clean certification passes exceed completed audit passes")
        if self.auditContractVersion == 1:
            if (
                self.certificationMode != "legacy"
                or self.auditPlanSha256 is not None
                or self.invariantEnvelopeSha256 is not None
                or self.invariantAuditSha256 is not None
                or self.deterministicFindings != 0
            ):
                raise ValueError("legacy certification result has contract-v2 metadata")
            if (
                self.status == "evaluated_clean"
                or self.auditPasses > 1
                or self.cleanAuditPasses != 0
                or self.requiredAuditPasses != 1
            ):
                raise ValueError("legacy certification result has contract-v2 outcome metadata")
            return self
        if self.auditContractVersion == 2 and (
            self.invariantEnvelopeSha256 is not None
            or self.invariantAuditSha256 is not None
            or self.deterministicFindings != 0
        ):
            raise ValueError("contract-v2 result has contract-v3 invariant metadata")
        if self.auditContractVersion == 3 and (
            self.invariantEnvelopeSha256 is None or self.invariantAuditSha256 is None
        ):
            raise ValueError("contract-v3 result lacks deterministic invariant identities")
        if self.auditPlanSha256 is None:
            raise ValueError("contract-v2 certification result lacks its audit-plan identity")
        if self.certificationMode == "legacy":
            raise ValueError("contract-v2 certification result cannot use legacy mode")
        expected_passes = 1 if self.certificationMode == "evaluation" else 2
        if self.requiredAuditPasses != expected_passes:
            raise ValueError("contract-v2 certification mode has the wrong pass requirement")
        if self.auditPasses > self.requiredAuditPasses:
            raise ValueError("completed audit passes exceed the configured pass requirement")
        expected_clean = self.auditPasses - (1 if self.semanticFindings else 0)
        if self.cleanAuditPasses != expected_clean:
            raise ValueError("clean audit pass count disagrees with semantic findings")
        if self.status == "certified":
            if not (
                self.certificationMode == "production"
                and self.auditPasses == 2
                and self.cleanAuditPasses == 2
                and self.semanticFindings == 0
                and self.deterministicFindings == 0
                and self.hostAudit.passed
                and not self.hostAudit.findings
            ):
                raise ValueError("certified result lacks complete unanimous v2 evidence")
        elif self.status == "evaluated_clean":
            if not (
                self.certificationMode == "evaluation"
                and self.auditPasses == 1
                and self.cleanAuditPasses == 1
                and self.semanticFindings == 0
                and self.deterministicFindings == 0
                and self.hostAudit.passed
                and not self.hostAudit.findings
            ):
                raise ValueError("evaluation-clean result lacks its single-pass evidence")
        elif self.status == "needs_review":
            if (
                self.semanticFindings == 0
                and self.deterministicFindings == 0
                and self.hostAudit.passed
            ):
                raise ValueError("needs-review result has no actionable defect")
        elif not (
            self.auditPasses < self.requiredAuditPasses
            and self.semanticFindings == 0
            and self.deterministicFindings == 0
            and self.hostAudit.passed
        ):
            raise ValueError("call-failed result does not represent an incomplete clean cascade")
        return self


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


def _combined_usage(stages: Sequence[CertificationStage]) -> LinguisticUsageReceipt:
    return _combine_usage([row.usage for row in stages]) if stages else _empty_usage()


def _line_number(value: str) -> int:
    if re.fullmatch(r"L[0-9]{5}", value) is None:
        raise ValueError(f"invalid line ID: {value!r}")
    return int(value[1:])


def _normalized(value: str) -> str:
    return " ".join(value.casefold().split())


def _count_surface(text: str, surface: str) -> int:
    pieces = tuple(piece for piece in re.split(r"\s+", surface.strip()) if piece)
    if not pieces:
        return 0
    stripped = surface.strip()
    prefix = r"(?<![A-Za-z0-9])" if stripped[0].isalnum() else ""
    suffix = r"(?![A-Za-z0-9])" if stripped[-1].isalnum() else ""
    pattern = re.compile(
        prefix + r"\s+".join(re.escape(piece) for piece in pieces) + suffix,
        re.IGNORECASE,
    )
    return sum(1 for _ in pattern.finditer(text))


def _contains_bounded_interleaved_description(
    text: str,
    description: str,
    *,
    max_interleaved_tokens: int = 12,
    max_line_span: int = 8,
) -> bool:
    """Recognize an ordered cargo phrase interrupted by nearby flattened table columns."""

    target_tokens = tuple(
        match.group(0).casefold() for match in _EVIDENCE_TOKEN.finditer(description)
    )
    if len(target_tokens) < 4:
        return False
    document_tokens = tuple(
        (match.group(0).casefold(), line_number)
        for line_number, line in enumerate(text.splitlines(), start=1)
        for match in _EVIDENCE_TOKEN.finditer(line)
    )
    for start, (token, start_line) in enumerate(document_tokens):
        if token != target_tokens[0]:
            continue
        target_index = 1
        interleaved = 0
        for candidate, line_number in document_tokens[start + 1 :]:
            if line_number - start_line > max_line_span:
                break
            if candidate == target_tokens[target_index]:
                target_index += 1
                if target_index == len(target_tokens):
                    return True
            else:
                interleaved += 1
                if interleaved > max_interleaved_tokens:
                    break
    return False


def _target_literal_present(
    *,
    text: str,
    target_path: object,
    target_value: str,
    match_policy: object = "semantic_literal",
) -> bool:
    if match_policy == "alphanumeric_identifier":
        compact_text = re.sub(r"[^A-Z0-9]", "", text.upper())
        compact_target = re.sub(r"[^A-Z0-9]", "", target_value.upper())
        return bool(compact_target) and compact_target in compact_text
    if match_policy == "ordered_semantic_atoms":
        if not isinstance(target_path, str):
            raise ValueError("ordered target literal requirement lacks a target path")
        requirement = TargetLiteralRequirement(
            targetPath=target_path,
            targetValue=target_value,
            matchPolicy="ordered_semantic_atoms",
        )
        return not _missing_target_literals(text, (requirement,))
    if match_policy != "semantic_literal":
        raise ValueError(f"unknown target literal match policy: {match_policy!r}")
    if _count_surface(text, target_value) >= 1:
        return True
    # Flattened OCR frequently glues a static party heading to its first value, for example
    # ``Notify PartyAcme Logistics``.  Requiring a word boundary before the target would reject
    # the exact target while also incentivizing the renderer to damage the source formatting by
    # inserting whitespace.  Limit the boundary-free fallback to party fields, where the whole
    # authoritative scalar (rather than a short category token) is being checked.
    if isinstance(target_path, str) and target_path.startswith("documentPatch.parties."):
        return _normalized(target_value) in _normalized(text)
    if isinstance(target_path, str) and target_path.endswith(".unNumber"):
        compact_target = re.sub(r"\D", "", target_value)
        return (
            bool(compact_target)
            and re.search(
                rf"(?<![0-9])(?:UN[ \t]*)?{re.escape(compact_target)}(?![0-9])",
                text,
                re.IGNORECASE,
            )
            is not None
        )
    return (
        isinstance(target_path, str)
        and _CARGO_DESCRIPTION_PATH.fullmatch(target_path) is not None
        and _contains_bounded_interleaved_description(text, target_value)
    )


def _page_markers(text: str) -> tuple[tuple[int, str], ...]:
    return tuple(
        (number, line)
        for number, line in enumerate(text.splitlines(), start=1)
        if _PAGE_MARKER.fullmatch(line) is not None
    )


def _host_audit(*, source: str, output: str, contract: Mapping[str, Any]) -> CertificationHostAudit:
    """Check topology, target obligations, and known source-only leakage deterministically."""

    findings: list[str] = []
    source_lines = source.splitlines()
    output_lines = output.splitlines()
    if len(source_lines) != len(output_lines):
        findings.append("OCR line count changed")
    if _page_markers(source) != _page_markers(output):
        findings.append("page markers or their line positions changed")
    source_kept = source.splitlines(keepends=True)
    output_kept = output.splitlines(keepends=True)
    if len(source_kept) == len(output_kept):
        newline_drift: list[int] = []
        blank_drift: list[int] = []
        edge_drift: list[int] = []
        for number, (source_raw, output_raw) in enumerate(
            zip(source_kept, output_kept, strict=True), start=1
        ):
            source_body = source_raw.rstrip("\r\n")
            output_body = output_raw.rstrip("\r\n")
            if source_raw[len(source_body) :] != output_raw[len(output_body) :]:
                newline_drift.append(number)
            if bool(source_body.strip()) != bool(output_body.strip()):
                blank_drift.append(number)
            source_leading = cast(re.Match[str], re.match(r"^[ \t]*", source_body)).group(0)
            source_trailing = cast(re.Match[str], re.search(r"[ \t]*$", source_body)).group(0)
            output_leading = cast(re.Match[str], re.match(r"^[ \t]*", output_body)).group(0)
            output_trailing = cast(re.Match[str], re.search(r"[ \t]*$", output_body)).group(0)
            if (source_leading, source_trailing) != (output_leading, output_trailing):
                edge_drift.append(number)
        if newline_drift:
            findings.append(f"line-ending topology changed on lines {newline_drift[:12]}")
        if blank_drift:
            findings.append(f"blank-line topology changed on lines {blank_drift[:12]}")
        if edge_drift:
            findings.append(f"line-edge whitespace changed on lines {edge_drift[:12]}")

    literal_rows = contract.get("targetLiteralRequirements")
    if not isinstance(literal_rows, list):
        raise ValueError("source-run contract lacks targetLiteralRequirements")
    for row in literal_rows:
        if not isinstance(row, Mapping) or not isinstance(row.get("targetValue"), str):
            raise ValueError("invalid target literal requirement")
        target = cast(str, row["targetValue"])
        if not _target_literal_present(
            text=output,
            target_path=row.get("targetPath"),
            target_value=target,
            match_policy=row.get("matchPolicy", "semantic_literal"),
        ):
            findings.append(f"missing target literal for {row.get('targetPath')}: {target!r}")

    occurrence_rows = contract.get("targetValueOccurrenceRequirements")
    if not isinstance(occurrence_rows, list):
        raise ValueError("source-run contract lacks targetValueOccurrenceRequirements")
    for row in occurrence_rows:
        if (
            not isinstance(row, Mapping)
            or not isinstance(row.get("targetValue"), str)
            or not isinstance(row.get("requiredOccurrences"), int)
        ):
            raise ValueError("invalid target occurrence requirement")
        requirement = TargetValueOccurrenceRequirement.model_validate_json(
            canonical_json_bytes(row), strict=True
        )
        target = requirement.targetValue
        required = requirement.requiredOccurrences
        observed = _target_value_occurrence_count(output, requirement)
        if observed != required:
            findings.append(
                f"target occurrence count differs: expected={required} observed={observed} "
                f"for {row.get('targetPaths')}: {target!r}"
            )

    target_integrity = contract.get("targetIntegrity")
    if not isinstance(target_integrity, Mapping):
        raise ValueError("source-run contract lacks targetIntegrity")
    receipt = target_integrity.get("final_receipt")
    if not isinstance(receipt, Mapping) or receipt.get("valid") is not True:
        findings.append("upstream target transport-capacity receipt is not valid")
    if target_integrity.get("topology_matched") is not True:
        findings.append("upstream source/target object topology is not matched")

    source_placeholders: dict[str, int] = {}
    output_placeholders: dict[str, int] = {}
    for match in _NEW_PLACEHOLDER.finditer(source):
        token = match.group(0).casefold()
        source_placeholders[token] = source_placeholders.get(token, 0) + 1
    for match in _NEW_PLACEHOLDER.finditer(output):
        token = match.group(0).casefold()
        output_placeholders[token] = output_placeholders.get(token, 0) + 1
    for token, count in sorted(output_placeholders.items()):
        if count > source_placeholders.get(token, 0):
            findings.append(
                f"new placeholder introduced: {token!r} count={count} "
                f"source_count={source_placeholders.get(token, 0)}"
            )

    for auxiliary in locate_auxiliary_values(source):
        if _count_surface(output, auxiliary.value) > 0:
            findings.append(
                "source-only auxiliary value survived anonymization: "
                f"{auxiliary.category!r} value={auxiliary.value!r}"
            )
    identity_rows = contract.get("rawAuxiliaryIdentityRequirements", [])
    if not isinstance(identity_rows, list):
        raise ValueError("invalid rawAuxiliaryIdentityRequirements")
    for row in identity_rows:
        if not isinstance(row, Mapping) or not isinstance(row.get("sourceIdentity"), str):
            raise ValueError("invalid raw auxiliary identity requirement")
        identity = cast(str, row["sourceIdentity"])
        if _count_surface(output, identity) > 0:
            findings.append(f"source-only auxiliary identity survived anonymization: {identity!r}")
    if model_match := _MODEL_META.search(output):
        findings.append(f"model-control prose leaked into OCR: {model_match.group(0)!r}")
    if internal_match := _INTERNAL_CATEGORY.search(output):
        findings.append(f"internal schema category leaked into OCR: {internal_match.group(0)!r}")

    return CertificationHostAudit(
        passed=not findings,
        findings=tuple(findings),
        sourceLines=len(source_lines),
        outputLines=len(output_lines),
        targetLiteralRequirements=len(literal_rows),
        targetOccurrenceRequirements=len(occurrence_rows),
    )


_AUDIT_PLAN_COLLECTIONS: tuple[tuple[str, str, AuditDimension, type[BaseModel]], ...] = (
    ("M", "compactLabelChangeContract", "target_fact_fidelity", CompactLabelChangeDirective),
    (
        "P",
        "sourceStatusPreservationRequirements",
        "template_format_and_model_artifacts",
        SourceStatusPreservationRequirement,
    ),
    (
        "S",
        "surfaceRenderingRequirements",
        "repeated_and_derived_relations",
        SurfaceRenderingRequirement,
    ),
    (
        "O",
        "operationalFlavorRequirements",
        "repeated_and_derived_relations",
        OperationalFlavorRequirement,
    ),
    (
        "A",
        "rawAuxiliaryIdentityRequirements",
        "source_private_and_auxiliary_replacement",
        RawAuxiliaryIdentityRequirement,
    ),
    (
        "J",
        "jurisdictionalSurfaceRequirements",
        "route_jurisdiction_and_identifiers",
        JurisdictionalSurfaceRequirement,
    ),
    (
        "R",
        "compoundPartyFlavorRequirements",
        "party_legal_and_negotiability",
        CompoundPartyFlavorRequirement,
    ),
    (
        "I",
        "inlineSlotTopologyRequirements",
        "template_format_and_model_artifacts",
        InlineSlotTopologyRequirement,
    ),
    (
        "G",
        "cargoFlavorRewriteRequirements",
        "cargo_packages_and_dangerous_goods",
        CargoFlavorRewriteRequirement,
    ),
    ("T", "targetLiteralRequirements", "target_fact_fidelity", TargetLiteralRequirement),
    (
        "N",
        "targetValueOccurrenceRequirements",
        "repeated_and_derived_relations",
        TargetValueOccurrenceRequirement,
    ),
    (
        "B",
        "anchoredScalarReplacementRequirements",
        "target_fact_fidelity",
        AnchoredScalarReplacementRequirement,
    ),
    (
        "E",
        "sourceSemanticRoleHints",
        "equipment_temperature_and_capacity",
        SourceSemanticRoleHint,
    ),
)


def _surface_requirement_dimension(
    *, default: AuditDimension, key: str, row: BaseModel
) -> AuditDimension:
    if key != "surfaceRenderingRequirements":
        return default
    kind = cast(SurfaceRenderingRequirement, row).kind
    if "equipment" in kind:
        return "equipment_temperature_and_capacity"
    if kind == "carrier_receipt_count":
        return "repeated_and_derived_relations"
    if kind in {"carrier_header_identity", "carrier_principal_identity"}:
        return "party_legal_and_negotiability"
    if kind in {
        "hs_code",
        "hs_code_block",
        "cargo_origin",
        "dangerous_goods_tuple",
        "dangerous_goods_proper_shipping_name",
    }:
        return "cargo_packages_and_dangerous_goods"
    return "target_fact_fidelity"


def _project_audit_claim(row: BaseModel) -> dict[str, JsonValue]:
    """Keep only fields needed to audit a row already backed by the full ledger/labels."""

    if isinstance(row, OperationalFlavorRequirement):
        return {
            "requirementId": row.requirementId,
            "kind": row.kind,
            "sourceLineId": row.sourceLineId,
            "sourceValueSurface": row.sourceValueSurface,
            "targetValueSurface": row.targetValueSurface,
            "sourceCanonicalValue": row.sourceCanonicalValue,
            "targetCanonicalValue": row.targetCanonicalValue,
            "consistencyGroupId": row.consistencyGroupId,
            "targetContainerNumber": row.targetContainerNumber,
            "targetEquipmentFamily": row.targetEquipmentFamily,
            "maximumValue": row.maximumValue,
            "sameLineFollowingContainerNumber": row.sameLineFollowingContainerNumber,
        }
    if isinstance(row, SurfaceRenderingRequirement):
        value: dict[str, JsonValue] = {
            "kind": row.kind,
            "targetPath": row.targetPath,
            "sourceSurface": row.sourceSurface,
            "targetSurface": row.targetSurface,
            "sourceOccurrences": row.sourceOccurrences,
            "sourceLineIds": cast(JsonValue, list(row.sourceLineIds)),
        }
        if not row.sourceLineIds:
            value["contextEvidence"] = row.contextEvidence
        return value
    if isinstance(row, RawAuxiliaryIdentityRequirement):
        return {
            "requirementId": row.requirementId,
            "relationship": row.relationship,
            "sourceIdentity": row.sourceIdentity,
            "consistencyGroupId": row.consistencyGroupId,
            "targetPrincipalName": row.targetPrincipalName,
        }
    if isinstance(row, JurisdictionalSurfaceRequirement):
        return {
            "requirementId": row.requirementId,
            "programId": row.programId,
            "tradeDirection": row.tradeDirection,
            "programJurisdictionCountryCode": row.programJurisdictionCountryCode,
            "targetRouteCountryCode": row.targetRouteCountryCode,
            "sourceLineIds": cast(JsonValue, list(row.sourceLineIds)),
            "sourceOccurrenceLineIds": cast(
                JsonValue, [list(line_ids) for line_ids in row.sourceOccurrenceLineIds]
            ),
            "sourceSurface": row.sourceSurface,
            "targetSurface": row.targetSurface,
            "alternativeTargetSurfaces": cast(JsonValue, list(row.alternativeTargetSurfaces)),
            "sourceOccurrences": row.sourceOccurrences,
            "authority": row.authority,
            "officialSourceUrl": row.officialSourceUrl,
            "rewriteBasis": row.rewriteBasis,
            "targetPartyRole": row.targetPartyRole,
            "sourcePartyCountryCode": row.sourcePartyCountryCode,
            "targetPartyCountryCode": row.targetPartyCountryCode,
        }
    if isinstance(row, SourceSemanticRoleHint):
        return {
            "role": row.role,
            "sourceLineId": row.sourceLineId,
            "sourceSurface": row.sourceSurface,
            "requiredOutputSurface": row.requiredOutputSurface,
            "forbiddenTargetPathPrefixes": cast(JsonValue, list(row.forbiddenTargetPathPrefixes)),
        }
    return cast(dict[str, JsonValue], row.model_dump(mode="json"))


def _mutation_domain(paths: Sequence[str]) -> AuditDimension | None:
    joined = " ".join(paths)
    if ".parties." in joined or any(
        token in joined for token in (".negotiability", ".numberOfOriginals")
    ):
        return "party_legal_and_negotiability"
    if any(
        token in joined
        for token in (
            ".route.",
            ".placeOfIssue",
            ".transport.vesselName",
            ".transport.voyageNumber",
        )
    ):
        return "route_jurisdiction_and_identifiers"
    if ".containers[" in joined or ".temperature" in joined:
        return "equipment_temperature_and_capacity"
    if any(token in joined for token in (".cargoGroups[", ".cargoPackages[", ".cargoAllocation")):
        return "cargo_packages_and_dangerous_goods"
    return None


def _audit_plan(contract: Mapping[str, Any], *, source: str, current: str) -> dict[str, JsonValue]:
    """Project the trusted rewrite contract into complete, bounded audit obligations.

    The rows are evidence/focus obligations, not assertions that the prior compiler was right.
    The auditor must reconcile each row with the complete source/target labels and line ledger.
    This keeps deterministic enumeration on the host while leaving semantic judgment read-only.
    """

    obligations: list[dict[str, JsonValue]] = []
    for index, dimension in enumerate(_AUDIT_DIMENSIONS, start=1):
        obligations.append(
            {
                "obligationId": f"D{index:04d}",
                "dimension": dimension,
                "contractSource": "whole_document_dimension",
                "claim": {
                    "instruction": (
                        "Inspect the complete labels and every candidate ledger row for this "
                        "dimension; the listed contract rows are focus aids, not an exhaustive "
                        "substitute for whole-document review."
                    )
                },
            }
        )
    mutations: list[tuple[str, CompactLabelChangeDirective]] = []
    for prefix, key, default_dimension, row_model in _AUDIT_PLAN_COLLECTIONS:
        rows = contract.get(key)
        if not isinstance(rows, list):
            raise ValueError(f"source-run contract lacks {key}")
        if len(rows) > 9999:
            raise ValueError(f"source-run contract has too many {key} rows")
        for index, row in enumerate(rows, start=1):
            if not isinstance(row, Mapping):
                raise ValueError(f"source-run contract has an invalid {key} row")
            parsed = row_model.model_validate_json(canonical_json_bytes(row), strict=True)
            dimension = _surface_requirement_dimension(
                default=default_dimension, key=key, row=parsed
            )
            obligations.append(
                {
                    "obligationId": f"{prefix}{index:04d}",
                    "dimension": dimension,
                    "contractSource": key,
                    "claim": cast(JsonValue, _project_audit_claim(parsed)),
                }
            )
            if isinstance(parsed, CompactLabelChangeDirective):
                mutations.append((f"M{index:04d}", parsed))
                if parsed.action in {
                    "replace",
                    "replace_equipment_surface",
                    "deactivate_extractable_fact_preserve_slot",
                }:
                    obligations.append(
                        {
                            "obligationId": f"X{index:04d}",
                            "dimension": "source_private_and_auxiliary_replacement",
                            "contractSource": "retireChangedSourceValue",
                            "claim": {
                                "mutationObligationId": f"M{index:04d}",
                                "scope": "path_and_role_owned_not_a_global_surface_ban",
                            },
                        }
                    )
            if isinstance(parsed, OperationalFlavorRequirement):
                obligations.append(
                    {
                        "obligationId": f"Y{index:04d}",
                        "dimension": "source_private_and_auxiliary_replacement",
                        "contractSource": "retireOperationalSourceValue",
                        "claim": {
                            "operationalObligationId": f"O{index:04d}",
                            "scope": "line_and_role_owned_not_a_global_surface_ban",
                        },
                    }
                )
    domain_mutations: dict[AuditDimension, list[tuple[str, CompactLabelChangeDirective]]] = {}
    for obligation_id, mutation in mutations:
        domain = _mutation_domain(mutation.paths)
        if domain is not None:
            domain_mutations.setdefault(domain, []).append((obligation_id, mutation))
    for index, dimension in enumerate(_AUDIT_DIMENSIONS, start=1):
        rows = domain_mutations.get(dimension)
        if not rows:
            continue
        obligations.append(
            {
                "obligationId": f"K{index:04d}",
                "dimension": dimension,
                "contractSource": "domainMutationIndex",
                "claim": {
                    "referencedMutationObligationIds": cast(
                        JsonValue, [obligation_id for obligation_id, _row in rows]
                    ),
                },
            }
        )
    integrity = contract.get("targetIntegrity")
    if not isinstance(integrity, Mapping):
        raise ValueError("source-run contract lacks targetIntegrity")
    final_receipt = integrity.get("final_receipt")
    if not isinstance(final_receipt, Mapping):
        raise ValueError("source-run contract lacks the final target-integrity receipt")
    obligations.append(
        {
            "obligationId": "Q0001",
            "dimension": "target_fact_fidelity",
            "contractSource": "targetIntegrity.final_receipt",
            "claim": {
                "scope": "target_label_only_capacity_receipt_not_raw_candidate_proof",
                "receipt": cast(JsonValue, dict(final_receipt)),
            },
        }
    )
    target_label = contract.get("targetLabel")
    if not isinstance(target_label, Mapping) or not isinstance(
        target_label.get("documentPatch"), Mapping
    ):
        raise ValueError("source-run contract lacks its structured target label")
    domain_paths: tuple[tuple[str, AuditDimension, tuple[str, ...]], ...] = (
        (
            "F0001",
            "party_legal_and_negotiability",
            ("documentPatch.negotiability", "documentPatch.parties"),
        ),
        (
            "W0001",
            "route_jurisdiction_and_identifiers",
            (
                "documentPatch.route",
                "documentPatch.placeOfIssue",
                "documentPatch.freight.paymentPlace",
                "documentPatch.parties.*.country",
            ),
        ),
        (
            "Z0001",
            "equipment_temperature_and_capacity",
            ("documentPatch.containers", "documentPatch.cargoGroups.*.temperature"),
        ),
        (
            "C0001",
            "cargo_packages_and_dangerous_goods",
            (
                "documentPatch.cargoGroups",
                "documentPatch.cargoPackages",
                "documentPatch.cargoAllocationGroups",
            ),
        ),
        (
            "L0001",
            "repeated_and_derived_relations",
            (
                "documentPatch.cargoGroups.*.(grossWeight|netWeight|volume)",
                "documentPatch.cargoPackages.*.quantity",
                "documentPatch.cargoAllocationGroups.*.allocations",
                "documentPatch.containers",
            ),
        ),
    )
    for obligation_id, dimension, target_paths in domain_paths:
        obligations.append(
            {
                "obligationId": obligation_id,
                "dimension": dimension,
                "contractSource": "targetDomainSnapshot",
                "claim": {
                    "targetPaths": cast(JsonValue, list(target_paths)),
                    "instruction": (
                        "Audit these paths even when source and target labels agree; reconcile "
                        "every candidate assertion, repeated copy, and explicit relation."
                    ),
                },
            }
        )
    source_lines = source.splitlines()
    current_lines = current.splitlines()
    if len(source_lines) != len(current_lines):
        raise ValueError("semantic audit plan requires stable source/current line topology")
    all_nonblank = [
        f"L{number:05d}"
        for number, line in enumerate(current_lines, start=1)
        if line.strip() and _PAGE_MARKER.fullmatch(line) is None
    ]
    unchanged_nonblank = [
        f"L{number:05d}"
        for number, (source_line, current_line) in enumerate(
            zip(source_lines, current_lines, strict=True), start=1
        )
        if source_line == current_line
        and current_line.strip()
        and _PAGE_MARKER.fullmatch(current_line) is None
    ]
    changed_nonblank = [
        f"L{number:05d}"
        for number, (source_line, current_line) in enumerate(
            zip(source_lines, current_lines, strict=True), start=1
        )
        if source_line != current_line
        and current_line.strip()
        and _PAGE_MARKER.fullmatch(current_line) is None
    ]
    obligations.extend(
        (
            {
                "obligationId": "U0001",
                "dimension": "source_private_and_auxiliary_replacement",
                "contractSource": "unchangedNonblankLedgerScope",
                "claim": {
                    "priorityUnchangedLineIds": cast(JsonValue, unchanged_nonblank),
                    "instruction": (
                        f"Inspect all {len(all_nonblank)} nonblank ledger rows for source-specific "
                        "residue; give "
                        "priority to the listed byte-unchanged source lines."
                    ),
                },
            },
            {
                "obligationId": "V0001",
                "dimension": "template_format_and_model_artifacts",
                "contractSource": "changedNonblankLedgerScope",
                "claim": {
                    "lineIds": cast(JsonValue, changed_nonblank),
                    "instruction": (
                        "Inspect every listed changed candidate line for damaged labels, roles, "
                        "clauses, duplication, or malformed text."
                    ),
                },
            },
        )
    )
    if len(obligations) > 1000:
        raise ValueError("semantic audit plan exceeds its bounded obligation contract")
    required_coverage = [
        {
            "dimension": dimension,
            "obligationIds": [
                row["obligationId"] for row in obligations if row["dimension"] == dimension
            ],
        }
        for dimension in _AUDIT_DIMENSIONS
    ]
    return {
        "auditContractVersion": 2,
        "requiredDimensions": cast(JsonValue, list(_AUDIT_DIMENSIONS)),
        "requiredCoverage": cast(JsonValue, required_coverage),
        "obligations": cast(JsonValue, obligations),
    }


def _audit_expected_coverage(
    plan: Mapping[str, JsonValue],
) -> dict[AuditDimension, tuple[str, ...]]:
    value = plan.get("requiredCoverage")
    if not isinstance(value, list):
        raise ValueError("semantic audit plan lacks requiredCoverage")
    coverage: dict[AuditDimension, tuple[str, ...]] = {}
    for row in value:
        if not isinstance(row, Mapping):
            raise ValueError("semantic audit plan has an invalid coverage row")
        dimension = row.get("dimension")
        obligation_ids = row.get("obligationIds")
        if dimension not in _AUDIT_DIMENSIONS or not isinstance(obligation_ids, list):
            raise ValueError("semantic audit plan has an invalid coverage dimension")
        if not obligation_ids or any(not isinstance(item, str) for item in obligation_ids):
            raise ValueError("semantic audit plan has invalid obligation identities")
        coverage[dimension] = tuple(cast(list[str], obligation_ids))
    if tuple(coverage) != _AUDIT_DIMENSIONS:
        raise ValueError("semantic audit plan coverage is missing or out of order")
    return coverage


def _validate_candidate_evidence(
    evidence_rows: Sequence[SemanticAuditEvidence], *, current_lines: Sequence[str]
) -> None:
    for evidence in evidence_rows:
        number = _line_number(evidence.lineId)
        if not 1 <= number <= len(current_lines):
            raise ValueError(f"semantic audit line is outside the document: {evidence.lineId}")
        line = current_lines[number - 1]
        if _PAGE_MARKER.fullmatch(line) is not None or not line.strip():
            raise ValueError(f"semantic audit cites a blank/page-marker line: {evidence.lineId}")
        if evidence.currentFragment not in line:
            raise ValueError(
                f"semantic audit evidence is not exact candidate text: {evidence.lineId}"
            )


def _materialize_v2_evidence(
    evidence_rows: Sequence[ContractV2SemanticAuditEvidence],
    *,
    current_lines: Sequence[str],
) -> tuple[SemanticAuditEvidence, ...]:
    materialized: list[SemanticAuditEvidence] = []
    for evidence in evidence_rows:
        number = _line_number(evidence.lineId)
        if not 1 <= number <= len(current_lines):
            raise ValueError(f"semantic audit line is outside the document: {evidence.lineId}")
        line = current_lines[number - 1]
        if _PAGE_MARKER.fullmatch(line) is not None or not line.strip():
            raise ValueError(f"semantic audit cites a blank/page-marker line: {evidence.lineId}")
        materialized.append(SemanticAuditEvidence(lineId=evidence.lineId, currentFragment=line))
    return tuple(materialized)


def _validate_audit_output(
    output: AuditOutput,
    *,
    current: str,
    expected_coverage: Mapping[AuditDimension, Sequence[str]] | None = None,
) -> tuple[SemanticAuditFinding, ...]:
    """Resolve every model citation to exact immutable candidate text."""

    current_lines = current.splitlines()
    if isinstance(output, SemanticAuditOutput):
        if expected_coverage is None:
            raise ValueError("contract-v2 semantic audit lacks expected obligation identities")
        obligation_dimension_by_id: dict[str, AuditDimension] = {}
        for dimension_check in output.dimensionChecks:
            dimension = dimension_check.dimension
            expected_ids = tuple(expected_coverage[dimension])
            for obligation_id in expected_ids:
                if obligation_id in obligation_dimension_by_id:
                    raise ValueError("semantic audit plan repeats an obligation identity")
                obligation_dimension_by_id[obligation_id] = dimension
        for linked_finding in output.findings:
            for obligation_id in linked_finding.obligationIds:
                if obligation_id not in obligation_dimension_by_id:
                    raise ValueError("semantic finding links an absent obligation")
    elif expected_coverage is not None:
        raise ValueError("legacy semantic audit cannot satisfy contract-v2 obligations")
    seen: set[str] = set()
    accepted: list[SemanticAuditFinding] = []
    for raw_finding in output.findings:
        evidence = (
            _materialize_v2_evidence(raw_finding.evidence, current_lines=current_lines)
            if isinstance(raw_finding, ContractV2SemanticAuditFinding)
            else raw_finding.evidence
        )
        accepted_finding = SemanticAuditFinding(
            findingKind=raw_finding.findingKind,
            evidence=evidence,
            problem=raw_finding.problem,
        )
        digest = sha256_bytes(canonical_json_bytes(accepted_finding.model_dump(mode="json")))
        if digest in seen:
            continue
        seen.add(digest)
        _validate_candidate_evidence(accepted_finding.evidence, current_lines=current_lines)
        accepted.append(accepted_finding)
    return tuple(accepted)


def _parse_native_audit_response(
    *,
    response: ModelResponse,
    route_provider: str | None,
    config: SynthesisRawTextCertificationConfig,
    output_model: type[BaseModel],
) -> AuditOutput:
    """Parse one provider-bound strict native-JSON response."""

    if response.model_name != config.provider.model:
        raise ValueError("certification response model differs from its configured model")
    provider_details = response.provider_details
    if (
        response.provider_name != "openrouter"
        or response.provider_url != _OPENROUTER_BASE_URL
        or not isinstance(provider_details, Mapping)
        or provider_details.get("downstream_provider")
        != _expected_openrouter_downstream(route_provider, config)
    ):
        raise ValueError("certification response provider differs from its requested route")
    text_parts = tuple(part for part in response.parts if isinstance(part, TextPart))
    if not (
        len(text_parts) == 1
        and all(isinstance(part, (TextPart, ThinkingPart)) for part in response.parts)
    ):
        raise UnexpectedModelBehavior("certification response has unexpected native-output parts")

    def reject_duplicate_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        value: dict[str, Any] = {}
        for key, item in pairs:
            if key in value:
                raise ValueError(f"duplicate JSON object key: {key}")
            value[key] = item
        return value

    def reject_non_finite_constant(value: str) -> None:
        raise ValueError(f"non-finite JSON constant: {value}")

    try:
        decoded = json.loads(
            text_parts[0].content,
            object_pairs_hook=reject_duplicate_keys,
            parse_constant=reject_non_finite_constant,
        )
        return cast(
            AuditOutput,
            output_model.model_validate_json(canonical_json_bytes(decoded), strict=True),
        )
    except (TypeError, ValueError) as error:
        raise UnexpectedModelBehavior(
            "certification native response is not one strict JSON object"
        ) from error


def _validate_native_audit_response(
    output: AuditOutput,
    *,
    response: ModelResponse,
    route_provider: str | None,
    config: SynthesisRawTextCertificationConfig,
) -> None:
    """Bind PydanticAI's parsed output to the exact provider response bytes."""

    output_model: type[BaseModel] = (
        SemanticAuditOutput
        if isinstance(output, SemanticAuditOutput)
        else LegacySemanticAuditOutput
    )
    wire_output = _parse_native_audit_response(
        response=response,
        route_provider=route_provider,
        config=config,
        output_model=output_model,
    )
    if wire_output != output:
        raise UnexpectedModelBehavior(
            "certification parsed output differs from its native response"
        )


def _payload(
    *,
    document_id: str,
    source: str,
    current: str,
    source_label: Mapping[str, Any],
    target_label: Mapping[str, Any],
    host_findings: Sequence[str],
    audit_contract_version: Literal[1, 2, 3],
    audit_plan: Mapping[str, JsonValue] | None = None,
    invariant_case: CertificationInvariantCase | None = None,
) -> dict[str, JsonValue]:
    source_lines = source.splitlines()
    current_lines = current.splitlines()
    if len(source_lines) != len(current_lines):
        raise ValueError("semantic audit requires stable source/current line topology")
    ledger: list[dict[str, JsonValue]] = []
    for number, (source_line, current_line) in enumerate(
        zip(source_lines, current_lines, strict=True), start=1
    ):
        ledger.append(
            {
                "lineId": f"L{number:05d}",
                "sourceLine": source_line,
                "currentLine": None if current_line == source_line else current_line,
            }
        )
    payload: dict[str, JsonValue] = {
        "documentId": document_id,
        "instruction": (
            "Audit the complete candidate from first principles. currentLine=null means the "
            "candidate still equals sourceLine. You have no edit authority."
        ),
        "deterministicHostFindings": cast(JsonValue, list(host_findings)),
        "sourceTaskLabel": cast(JsonValue, source_label),
        "syntheticTargetLabel": cast(JsonValue, target_label),
        "lineLedger": cast(JsonValue, ledger),
    }
    if audit_contract_version in {2, 3}:
        if audit_plan is None:
            raise ValueError("modern semantic payload lacks its host-generated audit plan")
        payload["auditPlan"] = cast(JsonValue, dict(audit_plan))
        if audit_contract_version == 3:
            if invariant_case is None or not invariant_case.audit.passed:
                raise ValueError(
                    "contract-v3 semantic payload requires a passing deterministic invariant audit"
                )
            payload["deterministicInvariantReceipt"] = {
                "envelope": cast(JsonValue, invariant_case.envelope.model_dump(mode="json")),
                "audit": cast(JsonValue, invariant_case.audit.model_dump(mode="json")),
            }
        elif invariant_case is not None:
            raise ValueError("contract-v2 semantic payload cannot carry contract-v3 invariants")
    elif audit_plan is not None:
        raise ValueError("legacy semantic payload cannot carry a contract-v2 audit plan")
    elif invariant_case is not None:
        raise ValueError("legacy semantic payload cannot carry contract-v3 invariants")
    return payload


def _transient_route_error(error: Exception) -> bool:
    # Keep the route policy centralized.  In particular, OpenRouter reports an endpoint that was
    # removed by parameter/tool filtering as HTTP 404; that is terminal for the pinned endpoint,
    # but retryable on the next configured implementation of the same model.  Re-filtering the
    # already-classified error by status here previously discarded that valid failover case.
    return _retryable_route_error(error)


def _route_error_receipt(error: Exception) -> CertificationRouteError:
    category: Literal["unexpected_model_behavior", "model_http", "model_api", "other"]
    if isinstance(error, UnexpectedModelBehavior):
        category = "unexpected_model_behavior"
        status_code = None
        failed_routing_step = None
    elif isinstance(error, ModelHTTPError):
        category = "model_http"
        status_code = error.status_code
        metadata = error.body.get("metadata") if isinstance(error.body, Mapping) else None
        routing_step = (
            metadata.get("failed_routing_step") if isinstance(metadata, Mapping) else None
        )
        failed_routing_step = routing_step if isinstance(routing_step, str) else None
    elif isinstance(error, ModelAPIError):
        category = "model_api"
        status_code = None
        failed_routing_step = None
    else:
        category = "other"
        status_code = None
        failed_routing_step = None
    receipt = CertificationRouteError(
        category=category,
        errorType=type(error).__name__,
        message=str(error),
        statusCode=status_code,
        failedRoutingStep=failed_routing_step,
    )
    if _route_error_receipt_is_retryable(receipt) != _retryable_route_error(error):
        raise RuntimeError("certification route-error receipt changed retry semantics")
    return receipt


def _route_error_receipt_is_retryable(error: CertificationRouteError) -> bool:
    if error.category in {"unexpected_model_behavior", "model_api"}:
        return True
    if error.category == "other":
        return False
    status_code = error.statusCode
    if status_code is None:
        raise ValueError("HTTP certification route error lacks its status")
    if status_code in {408, 425, 429} or status_code >= 500:
        return True
    return status_code == 404 and error.failedRoutingStep in {
        "Filter by Parameters",
        "Filter by Tool Compatibility",
    }


def _route_error_is_retryable_for_attempt(
    error: CertificationRouteError,
    usage: LinguisticUsageReceipt,
) -> bool:
    """Retry only a transient error that produced no billable model response."""

    if error.category == "unexpected_model_behavior" or not _route_error_receipt_is_retryable(
        error
    ):
        return False
    return (
        usage.requests == 0
        and usage.inputTokens == 0
        and usage.outputTokens == 0
        and usage.reasoningTokens == 0
        and usage.visibleOutputTokens == 0
        and usage.cacheReadTokens == 0
        and usage.cacheWriteTokens == 0
        and usage.estimatedCostUsd == 0
        and usage.providerReportedCostUsd in {None, Decimal(0)}
        and not usage.providerResponseIds
        and not usage.downstreamProviders
        and not usage.finishReasons
    )


def _expected_openrouter_downstream(
    route_provider: str | None,
    config: SynthesisRawTextCertificationConfig,
) -> str:
    if config.provider.kind != "openrouter":
        raise ValueError("contract-v2 route receipt requires OpenRouter")
    requested_route = route_provider
    if requested_route is None:
        provider_only = config.provider.provider_only or ()
        if len(provider_only) != 1:
            raise ValueError("isolated certification route lacks one pinned provider")
        requested_route = provider_only[0]
    try:
        return _OPENROUTER_DOWNSTREAM_BY_ROUTE[requested_route]
    except KeyError as error:
        raise ValueError("certification stage names an unevaluated provider route") from error


def _retry_delay(
    *,
    document_id: str,
    audit_pass: Literal[1, 2],
    route_round: int,
    config: SynthesisRawTextCertificationConfig,
) -> float:
    if route_round <= 1:
        return 0.0
    workflow = config.workflow
    retry_index = route_round - 2
    base = workflow.retry_initial_delay_seconds * (workflow.retry_delay_multiplier**retry_index)
    bounded = min(base, workflow.retry_max_delay_seconds)
    digest = bytes.fromhex(
        sha256_bytes(f"{document_id}\0{audit_pass}\0{route_round}\0audit-v2".encode())
    )
    fraction = int.from_bytes(digest[:8], "big") / float(2**64 - 1)
    return min(workflow.retry_max_delay_seconds, bounded + fraction * workflow.retry_jitter_seconds)


async def _call_audit_pass(
    *,
    document_id: str,
    audit_pass: Literal[1, 2],
    reasoning_effort: AuditReasoningEffort,
    payload: Mapping[str, JsonValue],
    audit_plan_sha256: str | None,
    current: str,
    expected_coverage: Mapping[AuditDimension, Sequence[str]] | None,
    model: Model,
    config: SynthesisRawTextCertificationConfig,
    prompt: str,
) -> tuple[AuditOutput | None, tuple[CertificationStage, ...]]:
    output_model: type[BaseModel] = (
        SemanticAuditOutput if config.audit_contract_version != 1 else LegacySemanticAuditOutput
    )
    output_spec = NativeOutput(
        output_model,
        name="synthetic_bl_read_only_semantic_audit",
        description=(
            "Complete obligation/dimension coverage with exact-evidence findings."
            if config.audit_contract_version != 1
            else "Exact-evidence findings, or an explicit empty whole-document audit."
        ),
        strict=True,
    )
    schema = output_model.model_json_schema()
    prompt_sha = sha256_bytes(prompt.encode())
    provider = config.provider.model_copy(update={"reasoning_effort": reasoning_effort})
    routes = _provider_attempt_routes(config.provider)
    active_routes = routes
    stages: list[CertificationStage] = []
    route_attempt = 0
    for route_round in range(1, config.workflow.max_provider_route_rounds + 1):
        delay = _retry_delay(
            document_id=document_id,
            audit_pass=audit_pass,
            route_round=route_round,
            config=config,
        )
        if delay:
            await asyncio.sleep(delay)
        transient_routes: list[str | None] = []
        for route_provider in active_routes:
            route_attempt += 1
            agent = Agent[None, Any](
                model,
                output_type=output_spec,
                system_prompt=prompt,
                model_settings=_settings(
                    provider,
                    stage="reviewer",
                    prompt_sha256=prompt_sha,
                    route_provider=route_provider,
                ),
                retries={"output": 0},
                name="synthetic-bl-read-only-semantic-auditor",
            )
            started = time.time()
            observed: AuditOutput | None = None
            output: AuditOutput | None = None
            error: Exception | None = None
            host_error: str | None = None
            with capture_run_messages() as captured:
                try:
                    result = await agent.run(
                        json.dumps(payload, ensure_ascii=False, separators=(",", ":")),
                        usage_limits=UsageLimits(
                            request_limit=1,
                            output_tokens_limit=config.provider.max_output_tokens,
                        ),
                    )
                    candidate_output = cast(AuditOutput, result.output)
                    _validate_native_audit_response(
                        candidate_output,
                        response=result.response,
                        route_provider=route_provider,
                        config=config,
                    )
                    observed = candidate_output
                    _validate_audit_output(
                        observed,
                        current=current,
                        expected_coverage=expected_coverage,
                    )
                    output = observed
                except Exception as caught:
                    if observed is not None:
                        host_error = str(caught)
                    else:
                        error = caught
            responses = tuple(row for row in captured if isinstance(row, ModelResponse))
            route_error = (
                _route_error_receipt(error)
                if error is not None and config.audit_contract_version != 1
                else None
            )
            attempt_usage = usage_receipt(
                responses,
                config.provider.pricing,
                require_provider_cost=False,
            )
            retryable_route_error = (
                _route_error_is_retryable_for_attempt(route_error, attempt_usage)
                if route_error is not None
                else (_transient_route_error(error) if error is not None else None)
            )
            stages.append(
                CertificationStage(
                    auditContractVersion=config.audit_contract_version,
                    auditPass=audit_pass,
                    reasoningEffort=reasoning_effort,
                    auditPlanSha256=audit_plan_sha256,
                    routeProvider=route_provider,
                    routeRound=route_round,
                    routeAttempt=route_attempt,
                    retryDelayBeforeSeconds=delay if route_provider == active_routes[0] else 0.0,
                    inputPayloadSha256=sha256_bytes(canonical_json_bytes(payload)),
                    outputSchemaSha256=sha256_bytes(canonical_json_bytes(schema)),
                    startedAtUnixSeconds=started,
                    completedAtUnixSeconds=time.time(),
                    usage=attempt_usage,
                    messages=model_messages(captured),
                    modelOutput=(
                        cast(JsonValue, observed.model_dump(mode="json"))
                        if observed is not None
                        else None
                    ),
                    hostError=host_error,
                    errorType=type(error).__name__ if error is not None else None,
                    errorMessage=str(error) if error is not None else None,
                    retryableRouteError=retryable_route_error,
                    routeError=route_error,
                )
            )
            if output is not None:
                return output, tuple(stages)
            if host_error is not None:
                continue
            if error is None or not retryable_route_error:
                return None, tuple(stages)
            transient_routes.append(route_provider)
        if not transient_routes:
            break
        active_routes = tuple(transient_routes)
    return None, tuple(stages)


async def _call_audit(
    *,
    document_id: str,
    source: str,
    current: str,
    source_label: Mapping[str, Any],
    target_label: Mapping[str, Any],
    contract: Mapping[str, Any],
    host_findings: Sequence[str],
    invariant_case: CertificationInvariantCase | None = None,
    model: Model,
    config: SynthesisRawTextCertificationConfig,
    prompt: str,
) -> tuple[tuple[AuditOutput, ...], tuple[CertificationStage, ...]]:
    audit_plan = (
        _audit_plan(contract, source=source, current=current)
        if config.audit_contract_version != 1
        else None
    )
    expected_coverage = _audit_expected_coverage(audit_plan) if audit_plan is not None else None
    audit_plan_sha256 = (
        sha256_bytes(canonical_json_bytes(audit_plan)) if audit_plan is not None else None
    )
    payload = _payload(
        document_id=document_id,
        source=source,
        current=current,
        source_label=source_label,
        target_label=target_label,
        host_findings=host_findings,
        audit_contract_version=config.audit_contract_version,
        audit_plan=audit_plan,
        invariant_case=invariant_case,
    )
    efforts = [config.provider.reasoning_effort]
    if config.workflow.semantic_audit_passes == 2:
        confirmation_effort = config.workflow.confirmation_reasoning_effort
        if confirmation_effort is None:
            raise ValueError("two-pass semantic audit lacks its confirmation effort")
        efforts.append(confirmation_effort)
    outputs: list[AuditOutput] = []
    stages: list[CertificationStage] = []
    for pass_index, reasoning_effort in enumerate(efforts, start=1):
        audit_pass = cast(Literal[1, 2], pass_index)
        output, pass_stages = await _call_audit_pass(
            document_id=document_id,
            audit_pass=audit_pass,
            reasoning_effort=reasoning_effort,
            payload=payload,
            audit_plan_sha256=audit_plan_sha256,
            current=current,
            expected_coverage=expected_coverage,
            model=model,
            config=config,
            prompt=prompt,
        )
        stages.extend(pass_stages)
        if output is None:
            break
        outputs.append(output)
        findings = _validate_audit_output(
            output,
            current=current,
            expected_coverage=expected_coverage,
        )
        # A deterministic or semantic rejection is already actionable.  The expensive high pass
        # is reserved for confirming otherwise-clean candidates and receives no prior-pass output.
        if findings or host_findings:
            break
    return tuple(outputs), tuple(stages)


def _configured_audit_efforts(
    config: SynthesisRawTextCertificationConfig,
) -> tuple[AuditReasoningEffort, ...]:
    efforts = [config.provider.reasoning_effort]
    if config.workflow.semantic_audit_passes == 2:
        if config.workflow.confirmation_reasoning_effort is None:
            raise ValueError("two-pass certification lacks its confirmation effort")
        efforts.append(config.workflow.confirmation_reasoning_effort)
    return tuple(efforts)


def validate_certification_run_stage_identities(
    stages_by_document: Mapping[str, Sequence[CertificationStage]],
) -> None:
    """Reject reuse of one provider attempt or response across documents."""

    run_ids: set[str] = set()
    conversation_ids: set[str] = set()
    response_ids: set[str] = set()
    for document_id, stages in stages_by_document.items():
        for stage in stages:
            if stage.auditContractVersion == 1:
                continue
            messages = ModelMessagesTypeAdapter.validate_python(stage.messages)
            requests = tuple(row for row in messages if isinstance(row, ModelRequest))
            responses = tuple(row for row in messages if isinstance(row, ModelResponse))
            if len(requests) != 1:
                raise ValueError(
                    f"certification case {document_id} stage lacks one request identity"
                )
            request = requests[0]
            if request.run_id is None or request.conversation_id is None:
                raise ValueError(
                    f"certification case {document_id} stage lacks attempt identifiers"
                )
            if request.run_id in run_ids or request.conversation_id in conversation_ids:
                raise ValueError("certification run reuses a provider attempt across documents")
            run_ids.add(request.run_id)
            conversation_ids.add(request.conversation_id)
            for response in responses:
                response_id = response.provider_response_id
                if response_id is None or not response_id.strip():
                    raise ValueError("certification response lacks its provider identity")
                if response_id in response_ids:
                    raise ValueError(
                        "certification run reuses a provider response across documents"
                    )
                response_ids.add(response_id)


def replay_certification_audit_stages(
    *,
    document_id: str,
    stages: Sequence[CertificationStage],
    source: str,
    current: str,
    contract: Mapping[str, Any],
    audit_contract_version: Literal[1, 2, 3],
    expected_reasoning_efforts: Sequence[str] | None = None,
    config: SynthesisRawTextCertificationConfig | None = None,
    invariant_case: CertificationInvariantCase | None = None,
) -> CertificationAuditReplay:
    """Replay every accepted logical audit pass and union its semantic findings."""

    plan = (
        _audit_plan(contract, source=source, current=current)
        if audit_contract_version != 1
        else None
    )
    coverage = _audit_expected_coverage(plan) if plan is not None else None
    plan_sha256 = sha256_bytes(canonical_json_bytes(plan)) if plan is not None else None
    input_payload_sha256: str | None = None
    expected_user_prompt: str | None = None
    output_schema_sha256: str | None = None
    if audit_contract_version in {2, 3}:
        if config is None or config.audit_contract_version != audit_contract_version:
            raise ValueError("modern certification replay requires its exact configuration")
        if audit_contract_version == 3:
            if invariant_case is None:
                raise ValueError("contract-v3 certification replay lacks deterministic invariants")
        elif invariant_case is not None:
            raise ValueError("contract-v2 certification replay has contract-v3 invariants")
        source_label = contract.get("sourceLabel")
        target_label = contract.get("targetLabel")
        if not isinstance(source_label, Mapping) or not isinstance(target_label, Mapping):
            raise ValueError("contract-v2 certification source contract lacks complete labels")
        host_audit = _host_audit(source=source, output=current, contract=contract)
        if (
            audit_contract_version == 3
            and invariant_case is not None
            and (not invariant_case.audit.passed or not host_audit.passed)
        ):
            if stages:
                raise ValueError(
                    "contract-v3 provider audit ran after a deterministic host rejection"
                )
            return CertificationAuditReplay(
                outputs=(),
                findings=(),
                audit_passes=0,
                clean_audit_passes=0,
                audit_plan_sha256=plan_sha256,
            )
        expected_payload = _payload(
            document_id=document_id,
            source=source,
            current=current,
            source_label=source_label,
            target_label=target_label,
            host_findings=host_audit.findings,
            audit_contract_version=audit_contract_version,
            audit_plan=plan,
            invariant_case=invariant_case,
        )
        input_payload_sha256 = sha256_bytes(canonical_json_bytes(expected_payload))
        expected_user_prompt = json.dumps(
            expected_payload,
            ensure_ascii=False,
            separators=(",", ":"),
        )
        output_schema_sha256 = sha256_bytes(
            canonical_json_bytes(SemanticAuditOutput.model_json_schema())
        )
        if expected_reasoning_efforts is None:
            expected_reasoning_efforts = _configured_audit_efforts(config)
        configured_routes = _provider_attempt_routes(config.provider)
        if not stages:
            raise ValueError("modern clean candidate has no provider audit attempt")
    else:
        configured_routes = ()
    accepted: dict[int, tuple[AuditOutput, tuple[SemanticAuditFinding, ...]]] = {}
    route_attempts: dict[int, int] = {}
    prior_pass = 1
    prior_round_by_pass: dict[int, int] = {}
    active_routes_by_pass: dict[int, tuple[str | None, ...]] = {}
    round_stages: dict[tuple[int, int], list[CertificationStage]] = {}
    accepted_response_ids: set[str] = set()
    receipted_response_ids: set[str] = set()
    attempt_run_ids: set[str] = set()
    attempt_conversation_ids: set[str] = set()
    prior_stage: CertificationStage | None = None
    for stage in stages:
        if prior_stage is not None:
            if stage.startedAtUnixSeconds < prior_stage.completedAtUnixSeconds:
                raise ValueError("certification stage chronology overlaps or moves backwards")
            if prior_stage.errorType is not None and prior_stage.retryableRouteError is False:
                raise ValueError("certification continued after a terminal provider error")
        if stage.auditContractVersion != audit_contract_version:
            raise ValueError("certification stages mix audit contract versions")
        if stage.auditPass in accepted:
            raise ValueError("accepted certification output was not terminal for its pass")
        if audit_contract_version in {2, 3} and stage.auditPlanSha256 != plan_sha256:
            raise ValueError("certification stage audit-plan identity differs")
        if audit_contract_version in {2, 3} and (
            stage.inputPayloadSha256 != input_payload_sha256
            or stage.outputSchemaSha256 != output_schema_sha256
        ):
            raise ValueError("certification stage payload or schema identity differs")
        if audit_contract_version in {2, 3}:
            if stage.errorType is None:
                if stage.retryableRouteError is not None or stage.routeError is not None:
                    raise ValueError("certification non-error stage has route-error metadata")
            elif (
                stage.routeError is None
                or stage.retryableRouteError is None
                or stage.routeError.errorType != stage.errorType
                or stage.routeError.message != stage.errorMessage
                or (
                    stage.retryableRouteError
                    and not _route_error_receipt_is_retryable(stage.routeError)
                )
            ):
                raise ValueError("certification stage route-error receipt differs")
            parsed_messages = ModelMessagesTypeAdapter.validate_python(stage.messages)
            if not (
                len(parsed_messages) in {1, 2}
                and isinstance(parsed_messages[0], ModelRequest)
                and (len(parsed_messages) == 1 or isinstance(parsed_messages[1], ModelResponse))
            ):
                raise ValueError("certification stage has an impossible message sequence")
            request = parsed_messages[0]
            if not (
                request.instructions is None
                and request.state == "complete"
                and isinstance(request.run_id, str)
                and bool(request.run_id.strip())
                and isinstance(request.conversation_id, str)
                and bool(request.conversation_id.strip())
                and len(request.parts) == 2
                and isinstance(request.parts[0], SystemPromptPart)
                and isinstance(request.parts[1], UserPromptPart)
                and sha256_bytes(request.parts[0].content.encode("utf-8"))
                == cast(SynthesisRawTextCertificationConfig, config).prompt.sha256
                and request.parts[1].content == expected_user_prompt
            ):
                raise ValueError(
                    "certification request transcript differs from its prompt or payload"
                )
            if (
                request.run_id in attempt_run_ids
                or request.conversation_id in attempt_conversation_ids
            ):
                raise ValueError("certification stages reuse an attempt identity")
            attempt_run_ids.add(request.run_id)
            attempt_conversation_ids.add(request.conversation_id)
            message_responses = tuple(
                row for row in parsed_messages if isinstance(row, ModelResponse)
            )
            if (
                usage_receipt(
                    message_responses,
                    cast(SynthesisRawTextCertificationConfig, config).provider.pricing,
                    require_provider_cost=False,
                )
                != stage.usage
            ):
                raise ValueError("certification stage usage differs from its message receipt")
            response_ids = tuple(stage.usage.providerResponseIds)
            if len(response_ids) != len(set(response_ids)) or receipted_response_ids.intersection(
                response_ids
            ):
                raise ValueError("certification stages reuse a provider response identity")
            receipted_response_ids.update(response_ids)
            wire_output: AuditOutput | None = None
            if message_responses:
                response = message_responses[0]
                response_details = response.provider_details
                if (
                    len(response_ids) != 1
                    or not response_ids[0].strip()
                    or response.provider_response_id != response_ids[0]
                    or response.run_id != request.run_id
                    or response.conversation_id != request.conversation_id
                    or response.model_name
                    != cast(SynthesisRawTextCertificationConfig, config).provider.model
                    or response.provider_name != "openrouter"
                    or response.provider_url != _OPENROUTER_BASE_URL
                    or not isinstance(response_details, Mapping)
                    or response_details.get("downstream_provider")
                    != _expected_openrouter_downstream(
                        stage.routeProvider,
                        cast(SynthesisRawTextCertificationConfig, config),
                    )
                ):
                    raise ValueError(
                        "certification response provenance differs from its request or route"
                    )
                try:
                    wire_output = _parse_native_audit_response(
                        response=response,
                        route_provider=stage.routeProvider,
                        config=cast(SynthesisRawTextCertificationConfig, config),
                        output_model=SemanticAuditOutput,
                    )
                except UnexpectedModelBehavior as error:
                    if stage.modelOutput is not None:
                        raise ValueError(str(error)) from error
                    if (
                        stage.routeError is None
                        or stage.routeError.category != "unexpected_model_behavior"
                    ):
                        raise ValueError(
                            "certification response parse failure lacks its exact error category"
                        ) from error
            if stage.modelOutput is None:
                if wire_output is not None:
                    raise ValueError("certification stage discards a valid native audit response")
            else:
                if wire_output is None:
                    raise ValueError(
                        "model-returned certification stage lacks one native JSON response"
                    )
                receipted_output = SemanticAuditOutput.model_validate_json(
                    canonical_json_bytes(stage.modelOutput),
                    strict=True,
                )
                if receipted_output != wire_output:
                    raise ValueError("certification parsed output differs from its native response")
                try:
                    _validate_audit_output(
                        receipted_output,
                        current=current,
                        expected_coverage=coverage,
                    )
                except ValueError as error:
                    if stage.hostError != str(error):
                        raise ValueError(
                            "certification host rejection differs from deterministic replay: "
                            f"{error}"
                        ) from error
                else:
                    if stage.hostError is not None:
                        raise ValueError(
                            "certification stage invents a host rejection for valid model output"
                        )
        if stage.auditPass < prior_pass or stage.auditPass > prior_pass + 1:
            raise ValueError("certification stage pass sequence is not contiguous")
        if stage.auditPass == 2 and 1 not in accepted:
            raise ValueError("confirmation audit began before an accepted screen")
        if stage.auditPass == 2 and accepted[1][1]:
            raise ValueError("confirmation audit followed a rejecting screen")
        if stage.auditPass == 2 and audit_contract_version in {2, 3} and not host_audit.passed:
            raise ValueError("confirmation audit followed a deterministic host rejection")
        prior_pass = stage.auditPass
        route_attempts[stage.auditPass] = route_attempts.get(stage.auditPass, 0) + 1
        if stage.routeAttempt != route_attempts[stage.auditPass]:
            raise ValueError("certification stage route-attempt sequence differs")
        prior_round = prior_round_by_pass.get(stage.auditPass)
        if prior_round is None:
            if stage.routeRound != 1:
                raise ValueError("certification pass did not begin at route round one")
            active_routes_by_pass[stage.auditPass] = configured_routes
        elif stage.routeRound not in {prior_round, prior_round + 1}:
            raise ValueError("certification stage route-round sequence is not contiguous")
        elif stage.routeRound == prior_round + 1 and audit_contract_version in {2, 3}:
            prior_rows = round_stages[(stage.auditPass, prior_round)]
            prior_active_routes = active_routes_by_pass[stage.auditPass]
            if len(prior_rows) != len(prior_active_routes):
                raise ValueError("certification retry round began before exhausting active routes")
            next_active_routes = tuple(
                row.routeProvider for row in prior_rows if row.retryableRouteError is True
            )
            if not next_active_routes:
                raise ValueError("certification retry round has no retryable routes")
            active_routes_by_pass[stage.auditPass] = next_active_routes
        prior_round_by_pass[stage.auditPass] = stage.routeRound
        if audit_contract_version in {2, 3}:
            if (
                stage.routeRound
                > cast(
                    SynthesisRawTextCertificationConfig, config
                ).workflow.max_provider_route_rounds
            ):
                raise ValueError("certification stage exceeds configured provider-route rounds")
            round_key = (stage.auditPass, stage.routeRound)
            rows_in_round = round_stages.setdefault(round_key, [])
            active_routes = active_routes_by_pass[stage.auditPass]
            route_position = len(rows_in_round)
            if (
                route_position >= len(active_routes)
                or stage.routeProvider != active_routes[route_position]
            ):
                raise ValueError("certification routes differ from the active provider order")
            expected_delay = (
                _retry_delay(
                    document_id=document_id,
                    audit_pass=stage.auditPass,
                    route_round=stage.routeRound,
                    config=cast(SynthesisRawTextCertificationConfig, config),
                )
                if not rows_in_round
                else 0.0
            )
            if abs(stage.retryDelayBeforeSeconds - expected_delay) > 1e-9:
                raise ValueError("certification retry delay differs from deterministic policy")
            rows_in_round.append(stage)
        if expected_reasoning_efforts is not None:
            if stage.auditPass > len(expected_reasoning_efforts):
                raise ValueError("certification stage exceeds configured audit passes")
            if audit_contract_version in {2, 3} and (
                stage.reasoningEffort != expected_reasoning_efforts[stage.auditPass - 1]
            ):
                raise ValueError("certification stage reasoning effort differs")
        valid_output = (
            stage.modelOutput is not None
            and stage.hostError is None
            and stage.errorType is None
            and stage.errorMessage is None
        )
        if not valid_output:
            prior_stage = stage
            continue
        if stage.auditPass in accepted:
            raise ValueError("certification pass has more than one accepted output")
        accepted_ids = set(stage.usage.providerResponseIds)
        if accepted_response_ids & accepted_ids:
            raise ValueError("certification passes reuse a provider response identity")
        accepted_response_ids.update(accepted_ids)
        output_model: type[BaseModel] = (
            SemanticAuditOutput if audit_contract_version != 1 else LegacySemanticAuditOutput
        )
        output = cast(
            AuditOutput,
            output_model.model_validate_json(canonical_json_bytes(stage.modelOutput), strict=True),
        )
        findings = _validate_audit_output(
            output,
            current=current,
            expected_coverage=coverage,
        )
        accepted[stage.auditPass] = (output, findings)
        prior_stage = stage
    if audit_contract_version in {2, 3} and stages:
        exact_config = cast(SynthesisRawTextCertificationConfig, config)
        final_stage = stages[-1]
        accepted_final = accepted.get(final_stage.auditPass)
        if accepted_final is not None:
            _output, final_findings = accepted_final
            if (
                final_stage.auditPass < exact_config.workflow.semantic_audit_passes
                and not final_findings
                and host_audit.passed
            ):
                raise ValueError("certification transcript stops before its required next pass")
        elif not (final_stage.errorType is not None and final_stage.retryableRouteError is False):
            final_round_rows = round_stages[(final_stage.auditPass, final_stage.routeRound)]
            final_active_routes = active_routes_by_pass[final_stage.auditPass]
            if len(final_round_rows) != len(final_active_routes):
                raise ValueError("certification transcript stops before exhausting active routes")
            has_retryable_route = any(row.retryableRouteError is True for row in final_round_rows)
            if (
                has_retryable_route
                and final_stage.routeRound < exact_config.workflow.max_provider_route_rounds
            ):
                raise ValueError("certification transcript stops before its required retry round")
    if tuple(accepted) != tuple(range(1, len(accepted) + 1)):
        raise ValueError("accepted certification passes are not contiguous")
    all_findings: list[SemanticAuditFinding] = []
    seen_findings: set[str] = set()
    for _output, findings in accepted.values():
        for finding in findings:
            digest = sha256_bytes(canonical_json_bytes(finding.model_dump(mode="json")))
            if digest in seen_findings:
                raise ValueError("certification passes repeat an identical semantic finding")
            seen_findings.add(digest)
            all_findings.append(finding)
    return CertificationAuditReplay(
        outputs=tuple(output for output, _findings in accepted.values()),
        findings=tuple(all_findings),
        audit_passes=len(accepted),
        clean_audit_passes=sum(not findings for _output, findings in accepted.values()),
        audit_plan_sha256=plan_sha256,
    )


def _certification_mode(
    config: SynthesisRawTextCertificationConfig,
) -> Literal["legacy", "evaluation", "production"]:
    if config.audit_contract_version == 1:
        return "legacy"
    return "evaluation" if config.workflow.evaluation_only else "production"


def _certification_outcome(
    *,
    replay: CertificationAuditReplay,
    host_audit: CertificationHostAudit,
    invariant_case: CertificationInvariantCase | None = None,
    mode: Literal["legacy", "evaluation", "production"],
    required_passes: int,
) -> tuple[Literal["certified", "evaluated_clean", "needs_review", "call_failed"], str]:
    if invariant_case is not None and not invariant_case.audit.passed:
        return (
            "needs_review",
            "The deterministic invariant audit proved one or more candidate defects before "
            "provider execution; the candidate must be regenerated or receive an exact "
            "host-authorized repair.",
        )
    if replay.findings or not host_audit.passed:
        return (
            "needs_review",
            "Read-only audit found semantic defects or a deterministic host gate failed; "
            "the candidate must return to the compiler.",
        )
    if replay.audit_passes < required_passes:
        return (
            "call_failed",
            "The configured semantic audit sequence did not return every required valid pass.",
        )
    if mode == "evaluation":
        return (
            "evaluated_clean",
            "The evaluation-only semantic audit found no defect; this is not publishable "
            "certification.",
        )
    return (
        "certified",
        "Every required read-only semantic audit and deterministic host gate passed.",
    )


def validate_certification_case_result(
    *,
    result: CertificationCaseResult,
    stages: Sequence[CertificationStage],
    source: str,
    current: str,
    contract: Mapping[str, Any],
    config: SynthesisRawTextCertificationConfig,
    invariant_case: CertificationInvariantCase | None = None,
) -> CertificationAuditReplay:
    """Recompute one case result from immutable bytes, configuration, and all audit stages."""

    host_audit = _host_audit(source=source, output=current, contract=contract)
    replay = replay_certification_audit_stages(
        document_id=result.documentId,
        stages=stages,
        source=source,
        current=current,
        contract=contract,
        audit_contract_version=config.audit_contract_version,
        expected_reasoning_efforts=(
            _configured_audit_efforts(config) if config.audit_contract_version != 1 else None
        ),
        config=config if config.audit_contract_version != 1 else None,
        invariant_case=invariant_case,
    )
    common_matches = (
        result.auditContractVersion == config.audit_contract_version
        and result.sourceTextSha256 == sha256_bytes(source.encode())
        and result.inputCandidateSha256 == sha256_bytes(current.encode())
        and result.finalTextSha256 == result.inputCandidateSha256
        and result.hostAudit == host_audit
        and result.usage == _combined_usage(stages)
        and result.semanticFindings == len(replay.findings)
        and result.auditPasses == replay.audit_passes
        and result.deterministicFindings
        == (len(invariant_case.audit.findings) if invariant_case is not None else 0)
    )
    if not common_matches:
        raise ValueError("certification case result fails deterministic replay")
    if config.audit_contract_version == 1:
        has_host_rejection = any(row.hostError is not None for row in stages)
        expected_status: str
        if replay.audit_passes == 0:
            expected_status = "needs_review" if has_host_rejection else "call_failed"
        elif replay.findings or not host_audit.passed:
            expected_status = "needs_review"
        else:
            expected_status = "certified"
        if result.certificationMode != "legacy" or result.status != expected_status:
            raise ValueError("legacy certification result status fails deterministic replay")
        return replay
    mode = _certification_mode(config)
    status, reason = _certification_outcome(
        replay=replay,
        host_audit=host_audit,
        invariant_case=invariant_case,
        mode=mode,
        required_passes=config.workflow.semantic_audit_passes,
    )
    expected_result = CertificationCaseResult(
        documentId=result.documentId,
        auditContractVersion=config.audit_contract_version,
        certificationMode=mode,
        auditPlanSha256=replay.audit_plan_sha256,
        invariantEnvelopeSha256=(
            sha256_bytes(canonical_json_bytes(invariant_case.envelope.model_dump(mode="json")))
            if invariant_case is not None
            else None
        ),
        invariantAuditSha256=(
            sha256_bytes(canonical_json_bytes(invariant_case.audit.model_dump(mode="json")))
            if invariant_case is not None
            else None
        ),
        status=status,
        reason=reason,
        auditPasses=replay.audit_passes,
        cleanAuditPasses=replay.clean_audit_passes,
        requiredAuditPasses=config.workflow.semantic_audit_passes,
        semanticFindings=len(replay.findings),
        deterministicFindings=(
            len(invariant_case.audit.findings) if invariant_case is not None else 0
        ),
        candidateImmutable=True,
        hostAudit=host_audit,
        sourceTextSha256=sha256_bytes(source.encode()),
        inputCandidateSha256=sha256_bytes(current.encode()),
        finalTextSha256=sha256_bytes(current.encode()),
        usage=_combined_usage(stages),
    )
    if result != expected_result:
        raise ValueError("modern certification result fails deterministic replay")
    return replay


def _read_json_object(path: Path) -> dict[str, Any]:
    value = json.loads(read_regular_file_bytes(path))
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path}")
    return value


def _validate_case_contract_labels(
    *,
    document_id: str,
    contract: Mapping[str, Any],
    source_label: Mapping[str, Any],
    target_label: Mapping[str, Any],
) -> None:
    if contract.get("sourceLabel") != source_label or contract.get("targetLabel") != target_label:
        raise ValueError(
            f"certification case {document_id} labels differ from its embedded contract"
        )


def _publish_case_checkpoint(
    *,
    staged: StagedArtifactRun,
    document_id: str,
    source: str,
    candidate: str,
    contract: Mapping[str, Any],
    config: SynthesisRawTextCertificationConfig,
    stages: Sequence[CertificationStage],
    result: CertificationCaseResult,
) -> None:
    staged.publish_json(
        f"cases/{document_id}/checkpoint.json",
        {
            "schemaVersion": 4,
            "documentId": document_id,
            "auditContractVersion": config.audit_contract_version,
            "certificationMode": _certification_mode(config),
            "requiredAuditPasses": config.workflow.semantic_audit_passes,
            "auditPlanSha256": result.auditPlanSha256,
            "invariantEnvelopeSha256": result.invariantEnvelopeSha256,
            "invariantAuditSha256": result.invariantAuditSha256,
            "sourceTextSha256": sha256_bytes(source.encode()),
            "inputCandidateSha256": sha256_bytes(candidate.encode()),
            "sourceContractSha256": sha256_bytes(canonical_json_bytes(contract)),
            "stages": [row.model_dump(mode="json") for row in stages],
            "result": result.model_dump(mode="json"),
        },
    )


def _load_case_checkpoint(
    *,
    staged: StagedArtifactRun,
    document_id: str,
    source: str,
    candidate: str,
    contract: Mapping[str, Any],
    config: SynthesisRawTextCertificationConfig,
    invariant_case: CertificationInvariantCase | None = None,
) -> tuple[CertificationCaseResult, tuple[CertificationStage, ...]] | None:
    path = staged.stage_root / f"cases/{document_id}/checkpoint.json"
    if not path.exists():
        return None
    if path.is_symlink() or not path.is_file():
        raise ValueError(f"certification checkpoint is not a regular file: {path}")
    value = json.loads(read_regular_file_bytes(path))
    expected = {
        "schemaVersion",
        "documentId",
        "auditContractVersion",
        "certificationMode",
        "requiredAuditPasses",
        "auditPlanSha256",
        "invariantEnvelopeSha256",
        "invariantAuditSha256",
        "sourceTextSha256",
        "inputCandidateSha256",
        "sourceContractSha256",
        "stages",
        "result",
    }
    if not isinstance(value, dict) or set(value) != expected:
        raise ValueError(f"certification checkpoint has an invalid shape: {path}")
    if (
        value["schemaVersion"] != 4
        or value["documentId"] != document_id
        or value["auditContractVersion"] != config.audit_contract_version
        or value["certificationMode"] != _certification_mode(config)
        or value["requiredAuditPasses"] != config.workflow.semantic_audit_passes
        or value["sourceTextSha256"] != sha256_bytes(source.encode())
        or value["inputCandidateSha256"] != sha256_bytes(candidate.encode())
        or value["sourceContractSha256"] != sha256_bytes(canonical_json_bytes(contract))
        or value["invariantEnvelopeSha256"]
        != (
            sha256_bytes(canonical_json_bytes(invariant_case.envelope.model_dump(mode="json")))
            if invariant_case is not None
            else None
        )
        or value["invariantAuditSha256"]
        != (
            sha256_bytes(canonical_json_bytes(invariant_case.audit.model_dump(mode="json")))
            if invariant_case is not None
            else None
        )
    ):
        raise ValueError(f"certification checkpoint identity differs: {path}")
    stages = tuple(
        CertificationStage.model_validate_json(canonical_json_bytes(row), strict=True)
        for row in cast(list[Any], value["stages"])
    )
    result = CertificationCaseResult.model_validate_json(
        canonical_json_bytes(value["result"]), strict=True
    )
    replay = validate_certification_case_result(
        result=result,
        stages=stages,
        source=source,
        current=candidate,
        contract=contract,
        config=config,
        invariant_case=invariant_case,
    )
    if result.documentId != document_id or value["auditPlanSha256"] != replay.audit_plan_sha256:
        raise ValueError(f"certification checkpoint fails deterministic replay: {path}")
    return result, stages


def _report(
    *, results: Sequence[CertificationCaseResult], stages: Sequence[CertificationStage]
) -> str:
    usage = _combined_usage(stages)
    count = max(1, len(results))
    estimated_rate = usage.estimatedCostUsd * Decimal(1000) / count
    provider_rate = (
        usage.providerReportedCostUsd * Decimal(1000) / count
        if usage.providerReportedCostUsd is not None
        else None
    )
    rows = [
        "# Read-only raw-text semantic certification",
        "",
        "## Outcome",
        "",
        f"- Certified: **{sum(row.status == 'certified' for row in results)}/{len(results)}**.",
        (
            "- Evaluation-only clean (never publishable): "
            f"**{sum(row.status == 'evaluated_clean' for row in results)}**."
        ),
        f"- Needs review: **{sum(row.status == 'needs_review' for row in results)}**.",
        f"- Call failed: **{sum(row.status == 'call_failed' for row in results)}**.",
        f"- Semantic findings: **{sum(row.semanticFindings for row in results)}**.",
        f"- Deterministic invariant findings: "
        f"**{sum(row.deterministicFindings for row in results)}**.",
        f"- Requests: **{usage.requests}**.",
        (
            f"- Input / reasoning / visible tokens: **{usage.inputTokens:,} / "
            f"{usage.reasoningTokens:,} / {usage.visibleOutputTokens:,}**."
        ),
        f"- Pinned-price estimate: **${usage.estimatedCostUsd}** (${estimated_rate:.3f}/1,000).",
        (
            "- Provider-reported cost: unavailable for at least one billed response."
            if provider_rate is None
            else f"- Provider-reported cost: **${usage.providerReportedCostUsd}** "
            f"(${provider_rate:.3f}/1,000)."
        ),
        "- Candidate bytes modified by certification: **0**.",
        "- Training records published: **0**.",
        "",
        "## Cases",
        "",
        "| Document | Mode | Status | Passes | Model findings | Invariant findings | "
        "Host findings | Cost |",
        "|---|---|---|---:|---:|---:|---:|---:|",
    ]
    for row in results:
        rows.append(
            f"| `{row.documentId}` | {row.certificationMode} | {row.status} | "
            f"{row.auditPasses}/{row.requiredAuditPasses} | {row.semanticFindings} | "
            f"{row.deterministicFindings} | {len(row.hostAudit.findings)} | "
            f"${row.usage.estimatedCostUsd} |"
        )
    rows.extend(
        (
            "",
            "A production-certified row has an unchanged input/output SHA-256, all deterministic "
            "host gates passing, complete obligation coverage, and independent low/high structured "
            "audits that both found no defect. Evaluation-only clean rows are deliberately "
            "non-publishable. A finding blocks publication and returns to the compiler; this stage "
            "never applies a repair.",
            "",
        )
    )
    return "\n".join(rows)


async def _execute_certification_cases(
    *,
    document_ids: Sequence[str],
    run_case: Callable[[str], Awaitable[bool]],
    fail_fast: bool,
) -> None:
    """Run planned evaluations serially so one invalid paid call stops the arm."""

    if fail_fast:
        for document_id in document_ids:
            if await run_case(document_id):
                raise RuntimeError(
                    "planned certification benchmark stopped after an invalid audit call: "
                    f"{document_id}"
                )
        return
    await asyncio.gather(*(run_case(document_id) for document_id in document_ids))


def run_raw_text_certification(
    *,
    project_root: Path,
    config_path: Path,
    config: SynthesisRawTextCertificationConfig,
) -> dict[str, JsonValue]:
    if config_path.is_symlink() or not config_path.is_file():
        raise ValueError("certification configuration must be a regular file")
    loaded_config = load_synthesis_raw_text_certification_config(config_path)
    if loaded_config != config:
        raise ValueError("certification configuration object differs from config_path")
    _validate_benchmark_plan_authorization(project_root, config)
    if (
        "audit_contract_version" not in loaded_config.model_fields_set
        or config.audit_contract_version != 3
    ):
        raise ValueError("new certification runs require explicit audit_contract_version: 3")
    input_root = _validate_reference_run(
        project_root,
        config.input_run.path,
        config.input_run.commit_sha256,
        config.input_run.transaction_sha256,
    )
    contract_name = config.input_case_contract_filename
    invariant_config = config.invariant_inputs
    if invariant_config is None:
        raise ValueError("contract-v3 certification lacks pinned invariant inputs")
    invariant_context = load_certification_invariant_context(
        project_root=project_root,
        config=invariant_config,
    )
    prompt_path = _resolve_pinned_file(
        project_root,
        config.prompt.path,
        config.prompt.sha256,
        label="read-only certification prompt",
    )
    prompt_bytes = read_regular_file_bytes(prompt_path)
    for document_id in config.case_ids:
        case_root = input_root / "cases" / document_id
        if not case_root.is_dir():
            raise ValueError(f"input run lacks certification case {document_id}")
        for name in (
            "source.txt",
            "final.txt",
            "source-label.json",
            "target-label.json",
            contract_name,
            "inventory.json",
            "deterministic-edits.json",
        ):
            path = case_root / name
            if path.is_symlink() or not path.is_file():
                raise ValueError(f"input certification artifact is not a regular file: {path}")
        _validate_case_contract_labels(
            document_id=document_id,
            contract=_read_json_object(case_root / contract_name),
            source_label=_read_json_object(case_root / "source-label.json"),
            target_label=_read_json_object(case_root / "target-label.json"),
        )

    implementation_contract = certification_implementation_contract()
    transaction = {
        "schemaVersion": 3,
        "runId": config.run.run_id,
        "configSha256": sha256_file(config_path),
        "resolvedConfigSha256": sha256_bytes(canonical_json_bytes(config.model_dump(mode="json"))),
        "inputRunCommitSha256": config.input_run.commit_sha256,
        "inputRunTransactionSha256": config.input_run.transaction_sha256,
        "benchmarkPlan": (
            config.benchmark_plan.model_dump(mode="json")
            if config.benchmark_plan is not None
            else None
        ),
        "inputCaseContractFilename": contract_name,
        "promptSha256": config.prompt.sha256,
        "invariantReferenceReceipt": invariant_context.references.receipt.model_dump(mode="json"),
        "invariantReferenceReceiptSha256": sha256_bytes(
            canonical_json_bytes(invariant_context.references.receipt.model_dump(mode="json"))
        ),
        "capacityConfigSha256": sha256_bytes(
            canonical_json_bytes(invariant_config.transport_capacity.model_dump(mode="json"))
        ),
        "implementationSha256": implementation_contract["implementationSha256"],
        "dependencyImplementationSha256": implementation_contract["dependencyImplementationSha256"],
        "documentIds": list(config.case_ids),
        "runtime": {
            "pydanticAiVersion": version("pydantic-ai-slim"),
            "openaiVersion": version("openai"),
            "model": config.provider.model,
            "providerOrder": (
                list(config.provider.provider_order or ())
                if config.provider.kind == "openrouter"
                else []
            ),
            "maxConcurrentDocuments": config.workflow.max_concurrent_documents,
            "semanticAuditPasses": config.workflow.semantic_audit_passes,
            "auditContractVersion": config.audit_contract_version,
            "certificationMode": _certification_mode(config),
            "reasoningEfforts": list(_configured_audit_efforts(config)),
        },
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
    staged.publish_bytes("prompts/read-only-auditor.md", prompt_bytes)
    staged.publish_json(
        "references/certification-reference-receipt.json",
        invariant_context.references.receipt.model_dump(mode="json"),
    )

    invariant_cases_by_id: dict[str, CertificationInvariantCase] = {}
    host_audits_by_id: dict[str, CertificationHostAudit] = {}
    for document_id in config.case_ids:
        case_root = input_root / "cases" / document_id
        source = read_regular_file_bytes(case_root / "source.txt").decode("utf-8")
        candidate = read_regular_file_bytes(case_root / "final.txt").decode("utf-8")
        contract = _read_json_object(case_root / contract_name)
        host_audits_by_id[document_id] = _host_audit(
            source=source,
            output=candidate,
            contract=contract,
        )
        invariant_cases_by_id[document_id] = load_certification_invariant_case(
            case_root=case_root,
            document_id=document_id,
            source=source,
            candidate=candidate,
            contract=contract,
            context=invariant_context,
        )

    results_by_id: dict[str, CertificationCaseResult] = {}
    stages_by_id: dict[str, tuple[CertificationStage, ...]] = {}
    for document_id in config.case_ids:
        case_root = input_root / "cases" / document_id
        source = read_regular_file_bytes(case_root / "source.txt").decode("utf-8")
        candidate = read_regular_file_bytes(case_root / "final.txt").decode("utf-8")
        contract = _read_json_object(case_root / contract_name)
        checkpoint = _load_case_checkpoint(
            staged=staged,
            document_id=document_id,
            source=source,
            candidate=candidate,
            contract=contract,
            config=config,
            invariant_case=invariant_cases_by_id[document_id],
        )
        if checkpoint is not None:
            results_by_id[document_id], stages_by_id[document_id] = checkpoint
    checkpointed_at_start = len(results_by_id)
    pending_provider_work = any(
        document_id not in results_by_id
        and host_audits_by_id[document_id].passed
        and invariant_cases_by_id[document_id].audit.passed
        for document_id in config.case_ids
    )
    client: AsyncOpenAI | None = None
    model: Model | None = None
    if pending_provider_work:
        key = load_provider_key(project_root, config.environment_file, config.provider.api_key_env)
        client_arguments: dict[str, Any] = {
            "api_key": key,
            "max_retries": config.provider.transport_max_retries,
            "timeout": config.provider.request_timeout_seconds,
        }
        if config.provider.kind == "openrouter":
            client_arguments.update(
                {
                    "base_url": _OPENROUTER_BASE_URL,
                    "default_headers": {"X-Title": "DocumentParsing"},
                }
            )
        client = AsyncOpenAI(**client_arguments)
        model, _ = _model_pair(
            client=client,
            editor_provider=config.provider,
            reviewer_provider=config.provider,
        )
    started = time.perf_counter()
    progress_lock = asyncio.Lock()
    limiter = asyncio.Semaphore(config.workflow.max_concurrent_documents)

    async def run_case(document_id: str) -> bool:
        case_root = input_root / "cases" / document_id
        source = read_regular_file_bytes(case_root / "source.txt").decode("utf-8")
        candidate = read_regular_file_bytes(case_root / "final.txt").decode("utf-8")
        source_label = _read_json_object(case_root / "source-label.json")
        target_label = _read_json_object(case_root / "target-label.json")
        contract = _read_json_object(case_root / contract_name)
        host_audit = host_audits_by_id[document_id]
        invariant_case = invariant_cases_by_id[document_id]
        stages: tuple[CertificationStage, ...] = ()
        if host_audit.passed and invariant_case.audit.passed:
            if model is None:
                raise RuntimeError("certification provider was not initialized for a clean case")
            async with limiter:
                _outputs, stages = await _call_audit(
                    document_id=document_id,
                    source=source,
                    current=candidate,
                    source_label=source_label,
                    target_label=target_label,
                    contract=contract,
                    host_findings=host_audit.findings,
                    invariant_case=invariant_case,
                    model=model,
                    config=config,
                    prompt=prompt_bytes.decode("utf-8"),
                )
        replay = replay_certification_audit_stages(
            document_id=document_id,
            stages=stages,
            source=source,
            current=candidate,
            contract=contract,
            audit_contract_version=config.audit_contract_version,
            expected_reasoning_efforts=_configured_audit_efforts(config),
            config=config,
            invariant_case=invariant_case,
        )
        mode = _certification_mode(config)
        status, reason = _certification_outcome(
            replay=replay,
            host_audit=host_audit,
            invariant_case=invariant_case,
            mode=mode,
            required_passes=config.workflow.semantic_audit_passes,
        )
        result = CertificationCaseResult(
            documentId=document_id,
            auditContractVersion=config.audit_contract_version,
            certificationMode=mode,
            auditPlanSha256=replay.audit_plan_sha256,
            invariantEnvelopeSha256=sha256_bytes(
                canonical_json_bytes(invariant_case.envelope.model_dump(mode="json"))
            ),
            invariantAuditSha256=sha256_bytes(
                canonical_json_bytes(invariant_case.audit.model_dump(mode="json"))
            ),
            status=status,
            reason=reason,
            auditPasses=replay.audit_passes,
            cleanAuditPasses=replay.clean_audit_passes,
            requiredAuditPasses=config.workflow.semantic_audit_passes,
            semanticFindings=len(replay.findings),
            deterministicFindings=len(invariant_case.audit.findings),
            candidateImmutable=True,
            hostAudit=host_audit,
            sourceTextSha256=sha256_bytes(source.encode()),
            inputCandidateSha256=sha256_bytes(candidate.encode()),
            finalTextSha256=sha256_bytes(candidate.encode()),
            usage=_combined_usage(stages),
        )
        async with progress_lock:
            _publish_case_checkpoint(
                staged=staged,
                document_id=document_id,
                source=source,
                candidate=candidate,
                contract=contract,
                config=config,
                stages=stages,
                result=result,
            )
            results_by_id[document_id] = result
            stages_by_id[document_id] = stages
            completed = len(results_by_id)
            elapsed = time.perf_counter() - started
            processed = completed - checkpointed_at_start
            rate = processed / elapsed if elapsed else 0.0
            print(
                json.dumps(
                    {
                        "command": "run-raw-text-certification",
                        "phase": "read_only_semantic_audit",
                        "processed_documents": completed,
                        "remaining_documents": len(config.case_ids) - completed,
                        "certified_documents": sum(
                            row.status == "certified" for row in results_by_id.values()
                        ),
                        "evaluated_clean_documents": sum(
                            row.status == "evaluated_clean" for row in results_by_id.values()
                        ),
                        "needs_review_documents": sum(
                            row.status == "needs_review" for row in results_by_id.values()
                        ),
                        "call_failed_documents": sum(
                            row.status == "call_failed" for row in results_by_id.values()
                        ),
                        "elapsed_seconds": round(elapsed, 3),
                        "throughput_documents_per_hour": round(rate * 3600, 3),
                        "eta_seconds": (
                            round((len(config.case_ids) - completed) / rate, 3) if rate else None
                        ),
                        "status": "progress",
                    },
                    allow_nan=False,
                    sort_keys=True,
                ),
                flush=True,
            )
        return result.status == "call_failed"

    async def execute() -> None:
        try:
            pending_ids = tuple(row for row in config.case_ids if row not in results_by_id)
            await _execute_certification_cases(
                document_ids=pending_ids,
                run_case=run_case,
                fail_fast=config.benchmark_plan is not None,
            )
        finally:
            if client is not None:
                await client.close()

    asyncio.run(execute())
    if set(results_by_id) != set(config.case_ids):
        raise RuntimeError("certification run returned an incomplete result set")
    results = tuple(results_by_id[row] for row in config.case_ids)
    all_stages = tuple(stage for row in config.case_ids for stage in stages_by_id[row])
    validate_certification_run_stage_identities(stages_by_id)
    usage = _combined_usage(all_stages)
    wall_seconds = time.perf_counter() - started
    processed = len(results) - checkpointed_at_start
    for document_id in config.case_ids:
        case_root = input_root / "cases" / document_id
        source = read_regular_file_bytes(case_root / "source.txt").decode("utf-8")
        candidate = read_regular_file_bytes(case_root / "final.txt").decode("utf-8")
        prefix = f"cases/{document_id}"
        staged.publish_bytes(f"{prefix}/source.txt", source.encode())
        staged.publish_bytes(f"{prefix}/input-candidate.txt", candidate.encode())
        staged.publish_bytes(f"{prefix}/final.txt", candidate.encode())
        staged.publish_bytes(
            f"{prefix}/diff-from-input.patch", unified_text_diff(candidate, candidate).encode()
        )
        staged.publish_bytes(
            f"{prefix}/diff-from-source.patch", unified_text_diff(source, candidate).encode()
        )
        staged.publish_json(
            f"{prefix}/source-label.json", _read_json_object(case_root / "source-label.json")
        )
        staged.publish_json(
            f"{prefix}/target-label.json", _read_json_object(case_root / "target-label.json")
        )
        staged.publish_json(
            f"{prefix}/source-contract.json", _read_json_object(case_root / contract_name)
        )
        contract = _read_json_object(case_root / contract_name)
        staged.publish_json(
            f"{prefix}/audit-plan.json",
            _audit_plan(contract, source=source, current=candidate),
        )
        invariant_case = invariant_cases_by_id[document_id]
        staged.publish_json(
            f"{prefix}/inventory.json",
            [row.model_dump(mode="json") for row in invariant_case.inventory],
        )
        staged.publish_json(
            f"{prefix}/deterministic-edits.json",
            [row.model_dump(mode="json") for row in invariant_case.deterministic_edits],
        )
        staged.publish_json(
            f"{prefix}/invariant-envelope.json",
            invariant_case.envelope.model_dump(mode="json"),
        )
        staged.publish_json(
            f"{prefix}/invariant-audit.json",
            invariant_case.audit.model_dump(mode="json"),
        )
        staged.publish_json(
            f"{prefix}/stages.json",
            [row.model_dump(mode="json") for row in stages_by_id[document_id]],
        )
        staged.publish_json(
            f"{prefix}/result.json", results_by_id[document_id].model_dump(mode="json")
        )
    summary: dict[str, JsonValue] = {
        "schemaVersion": config.schema_version,
        "runId": config.run.run_id,
        "status": "complete",
        "model": config.provider.model,
        "auditContractVersion": config.audit_contract_version,
        "certificationMode": _certification_mode(config),
        "requiredAuditPasses": config.workflow.semantic_audit_passes,
        "documents": len(results),
        "checkpointedDocumentsAtStart": checkpointed_at_start,
        "processedDocumentsThisInvocation": processed,
        "certifiedDocuments": sum(row.status == "certified" for row in results),
        "evaluatedCleanDocuments": sum(row.status == "evaluated_clean" for row in results),
        "needsReviewDocuments": sum(row.status == "needs_review" for row in results),
        "callFailedDocuments": sum(row.status == "call_failed" for row in results),
        "semanticFindings": sum(row.semanticFindings for row in results),
        "deterministicFindings": sum(row.deterministicFindings for row in results),
        "deterministicRejectedBeforeProvider": sum(
            row.status == "needs_review" and row.auditPasses == 0 for row in results
        ),
        "requests": usage.requests,
        "providerAttempts": len(all_stages),
        "failedProviderAttempts": sum(row.errorType is not None for row in all_stages),
        "hostRejectedProviderOutputs": sum(row.hostError is not None for row in all_stages),
        "inputTokens": usage.inputTokens,
        "reasoningTokens": usage.reasoningTokens,
        "visibleOutputTokens": usage.visibleOutputTokens,
        "outputTokens": usage.outputTokens,
        "estimatedCostUsd": str(usage.estimatedCostUsd),
        "providerReportedCostUsd": (
            str(usage.providerReportedCostUsd)
            if usage.providerReportedCostUsd is not None
            else None
        ),
        "estimatedCostPerThousandUsd": str(usage.estimatedCostUsd * Decimal(1000) / len(results)),
        "providerReportedCostPerThousandUsd": (
            str(usage.providerReportedCostUsd * Decimal(1000) / len(results))
            if usage.providerReportedCostUsd is not None
            else None
        ),
        "wallSeconds": round(wall_seconds, 6),
        "throughputDocumentsPerHour": (
            round(processed / wall_seconds * 3600, 6) if processed else None
        ),
        "candidateBytesModified": 0,
        "trainingRecordsPublished": False,
    }
    staged.publish_json("summary.json", summary)
    staged.publish_bytes(
        "generation/results.jsonl",
        b"".join(canonical_json_bytes(row.model_dump(mode="json")) + b"\n" for row in results),
    )

    staged.publish_bytes("REPORT.md", _report(results=results, stages=all_stages).encode())
    staged.publish_json("provenance/transaction.json", transaction)
    staged.commit(
        expected_artifacts=_artifact_inventory(staged.stage_root),
        metadata={
            "schemaVersion": 2,
            "documents": len(results),
            "certifiedDocuments": cast(int, summary["certifiedDocuments"]),
            "candidateBytesModified": 0,
            "trainingRecordsPublished": False,
        },
    )
    output = dict(summary)
    output["artifactRoot"] = str(staged.final_root)
    output["commitSha256"] = sha256_file(staged.final_root / "_COMMIT.json")
    return output
