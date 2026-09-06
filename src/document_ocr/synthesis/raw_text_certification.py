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
from collections.abc import Mapping, Sequence
from decimal import Decimal
from importlib.metadata import version
from pathlib import Path
from typing import Annotated, Any, Literal, cast

from openai import AsyncOpenAI
from pydantic import BaseModel, ConfigDict, Field, JsonValue, StringConstraints, model_validator
from pydantic_ai import Agent, NativeOutput, capture_run_messages
from pydantic_ai.messages import ModelResponse
from pydantic_ai.models import Model
from pydantic_ai.usage import UsageLimits

from document_ocr.atomic import read_regular_file_bytes
from document_ocr.hashing import canonical_json_bytes, sha256_bytes, sha256_file
from document_ocr.synthesis.config import SynthesisRawTextCertificationConfig
from document_ocr.synthesis.linguistic_probe_runtime import (
    LinguisticUsageReceipt,
    load_provider_key,
    model_messages,
    usage_receipt,
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
    TargetValueOccurrenceRequirement,
    _combine_usage,
    _settings,
    _target_value_occurrence_count,
)
from document_ocr.synthesis.raw_text_rewrite_probe import _resolve_pinned_file, unified_text_diff
from document_ocr.synthesis.run_safety import StagedArtifactRun
from document_ocr.training.config import resolve_config_path

NonEmptyText = Annotated[str, StringConstraints(strip_whitespace=True, min_length=1)]
LineId = Annotated[str, StringConstraints(pattern=r"^L[0-9]{5}$")]
_STRICT = ConfigDict(extra="forbid", frozen=True, strict=True, allow_inf_nan=False)
_IMPLEMENTATION_PATH = Path(__file__).resolve(strict=True)
_OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"
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


class SemanticAuditEvidence(BaseModel):
    """An exact candidate fragment supporting one semantic finding."""

    model_config = _STRICT

    lineId: LineId
    currentFragment: Annotated[str, StringConstraints(min_length=1, max_length=1200)]


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
    evidence: Annotated[tuple[SemanticAuditEvidence, ...], Field(min_length=1, max_length=12)]
    problem: Annotated[str, StringConstraints(strip_whitespace=True, min_length=1, max_length=600)]

    @model_validator(mode="after")
    def evidence_lines_are_unique(self) -> SemanticAuditFinding:
        line_ids = tuple(row.lineId for row in self.evidence)
        if len(line_ids) != len(set(line_ids)):
            raise ValueError("semantic finding repeats an evidence line")
        return self


class SemanticAuditOutput(BaseModel):
    """Empty findings are an explicit whole-document clean assertion."""

    model_config = _STRICT

    findings: Annotated[tuple[SemanticAuditFinding, ...], Field(max_length=120)]


class CertificationHostAudit(BaseModel):
    model_config = _STRICT

    passed: bool
    findings: tuple[NonEmptyText, ...]
    sourceLines: Annotated[int, Field(gt=0)]
    outputLines: Annotated[int, Field(gt=0)]
    targetLiteralRequirements: Annotated[int, Field(ge=0)]
    targetOccurrenceRequirements: Annotated[int, Field(ge=0)]


class CertificationStage(BaseModel):
    model_config = _STRICT

    auditPass: Literal[1]
    routeProvider: str | None
    routeRound: Annotated[int, Field(ge=1)]
    routeAttempt: Annotated[int, Field(ge=1)]
    retryDelayBeforeSeconds: Annotated[float, Field(ge=0)]
    inputPayloadSha256: str
    outputSchemaSha256: str
    startedAtUnixSeconds: float
    completedAtUnixSeconds: float
    usage: LinguisticUsageReceipt
    messages: JsonValue
    modelOutput: JsonValue | None
    hostError: str | None
    errorType: str | None
    errorMessage: str | None


class CertificationCaseResult(BaseModel):
    model_config = _STRICT

    documentId: str
    status: Literal["certified", "needs_review", "call_failed"]
    reason: NonEmptyText
    auditPasses: Annotated[int, Field(ge=0, le=1)]
    semanticFindings: Annotated[int, Field(ge=0)]
    candidateImmutable: Literal[True]
    hostAudit: CertificationHostAudit
    sourceTextSha256: str
    inputCandidateSha256: str
    finalTextSha256: str
    usage: LinguisticUsageReceipt


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
        return bool(compact_target) and re.search(
            rf"(?<![0-9])(?:UN[ \t]*)?{re.escape(compact_target)}(?![0-9])",
            text,
            re.IGNORECASE,
        ) is not None
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
        findings.append(
            f"internal schema category leaked into OCR: {internal_match.group(0)!r}"
        )

    return CertificationHostAudit(
        passed=not findings,
        findings=tuple(findings),
        sourceLines=len(source_lines),
        outputLines=len(output_lines),
        targetLiteralRequirements=len(literal_rows),
        targetOccurrenceRequirements=len(occurrence_rows),
    )


def _validate_audit_output(
    output: SemanticAuditOutput,
    *,
    current: str,
) -> tuple[SemanticAuditFinding, ...]:
    """Resolve every model citation to exact immutable candidate text."""

    current_lines = current.splitlines()
    seen: set[str] = set()
    accepted: list[SemanticAuditFinding] = []
    for finding in output.findings:
        digest = sha256_bytes(canonical_json_bytes(finding.model_dump(mode="json")))
        if digest in seen:
            raise ValueError("semantic audit repeats an identical finding")
        seen.add(digest)
        for evidence in finding.evidence:
            number = _line_number(evidence.lineId)
            if not 1 <= number <= len(current_lines):
                raise ValueError(f"semantic audit line is outside the document: {evidence.lineId}")
            line = current_lines[number - 1]
            if _PAGE_MARKER.fullmatch(line) is not None or not line.strip():
                raise ValueError(
                    f"semantic audit cites a blank/page-marker line: {evidence.lineId}"
                )
            if evidence.currentFragment not in line:
                raise ValueError(
                    f"semantic audit evidence is not exact candidate text: {evidence.lineId}"
                )
        accepted.append(finding)
    return tuple(accepted)


def _payload(
    *,
    document_id: str,
    source: str,
    current: str,
    source_label: Mapping[str, Any],
    target_label: Mapping[str, Any],
    host_findings: Sequence[str],
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
    return {
        "documentId": document_id,
        "instruction": (
            "Audit the complete candidate from first principles. currentLine=null means the "
            "candidate still equals sourceLine. Return findings only; you have no edit authority."
        ),
        "deterministicHostFindings": cast(JsonValue, list(host_findings)),
        "sourceTaskLabel": cast(JsonValue, source_label),
        "syntheticTargetLabel": cast(JsonValue, target_label),
        "lineLedger": cast(JsonValue, ledger),
    }


def _transient_route_error(error: Exception) -> bool:
    # Keep the route policy centralized.  In particular, OpenRouter reports an endpoint that was
    # removed by parameter/tool filtering as HTTP 404; that is terminal for the pinned endpoint,
    # but retryable on the next configured implementation of the same model.  Re-filtering the
    # already-classified error by status here previously discarded that valid failover case.
    return _retryable_route_error(error)


def _retry_delay(
    *, document_id: str, route_round: int, config: SynthesisRawTextCertificationConfig
) -> float:
    if route_round <= 1:
        return 0.0
    workflow = config.workflow
    retry_index = route_round - 2
    base = workflow.retry_initial_delay_seconds * (workflow.retry_delay_multiplier**retry_index)
    bounded = min(base, workflow.retry_max_delay_seconds)
    digest = bytes.fromhex(sha256_bytes(f"{document_id}\0{route_round}\0audit-v2".encode()))
    fraction = int.from_bytes(digest[:8], "big") / float(2**64 - 1)
    return min(workflow.retry_max_delay_seconds, bounded + fraction * workflow.retry_jitter_seconds)


async def _call_audit(
    *,
    document_id: str,
    source: str,
    current: str,
    source_label: Mapping[str, Any],
    target_label: Mapping[str, Any],
    host_findings: Sequence[str],
    model: Model,
    config: SynthesisRawTextCertificationConfig,
    prompt: str,
) -> tuple[SemanticAuditOutput | None, tuple[CertificationStage, ...]]:
    payload = _payload(
        document_id=document_id,
        source=source,
        current=current,
        source_label=source_label,
        target_label=target_label,
        host_findings=host_findings,
    )
    output_spec = NativeOutput(
        SemanticAuditOutput,
        name="synthetic_bl_read_only_semantic_audit",
        description="Exact-evidence findings, or an explicit empty whole-document audit.",
        strict=True,
    )
    schema = SemanticAuditOutput.model_json_schema()
    prompt_sha = sha256_bytes(prompt.encode())
    routes = _provider_attempt_routes(config.provider)
    active_routes = routes
    stages: list[CertificationStage] = []
    route_attempt = 0
    for route_round in range(1, config.workflow.max_provider_route_rounds + 1):
        delay = _retry_delay(document_id=document_id, route_round=route_round, config=config)
        if delay:
            await asyncio.sleep(delay)
        transient_routes: list[str | None] = []
        for route_provider in active_routes:
            route_attempt += 1
            agent = Agent[None, SemanticAuditOutput](
                model,
                output_type=output_spec,
                system_prompt=prompt,
                model_settings=_settings(
                    config.provider,
                    stage="reviewer",
                    prompt_sha256=prompt_sha,
                    route_provider=route_provider,
                ),
                retries={"output": 0},
                name="synthetic-bl-read-only-semantic-auditor",
            )
            started = time.time()
            observed: SemanticAuditOutput | None = None
            output: SemanticAuditOutput | None = None
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
                    observed = result.output
                    _validate_audit_output(
                        observed,
                        current=current,
                    )
                    output = observed
                except Exception as caught:
                    if observed is not None:
                        host_error = str(caught)
                    else:
                        error = caught
            responses = tuple(row for row in captured if isinstance(row, ModelResponse))
            stages.append(
                CertificationStage(
                    auditPass=1,
                    routeProvider=route_provider,
                    routeRound=route_round,
                    routeAttempt=route_attempt,
                    retryDelayBeforeSeconds=delay if route_provider == active_routes[0] else 0.0,
                    inputPayloadSha256=sha256_bytes(canonical_json_bytes(payload)),
                    outputSchemaSha256=sha256_bytes(canonical_json_bytes(schema)),
                    startedAtUnixSeconds=started,
                    completedAtUnixSeconds=time.time(),
                    usage=usage_receipt(
                        responses,
                        config.provider.pricing,
                        require_provider_cost=False,
                    ),
                    messages=model_messages(captured),
                    modelOutput=(
                        cast(JsonValue, observed.model_dump(mode="json"))
                        if observed is not None
                        else None
                    ),
                    hostError=host_error,
                    errorType=type(error).__name__ if error is not None else None,
                    errorMessage=str(error) if error is not None else None,
                )
            )
            if output is not None:
                return output, tuple(stages)
            if host_error is not None:
                continue
            if error is None or not _transient_route_error(error):
                return None, tuple(stages)
            transient_routes.append(route_provider)
        if not transient_routes:
            break
        active_routes = tuple(transient_routes)
    return None, tuple(stages)


def _read_json_object(path: Path) -> dict[str, Any]:
    value = json.loads(read_regular_file_bytes(path))
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path}")
    return value


def _publish_case_checkpoint(
    *,
    staged: StagedArtifactRun,
    document_id: str,
    source: str,
    candidate: str,
    contract: Mapping[str, Any],
    stages: Sequence[CertificationStage],
    result: CertificationCaseResult,
) -> None:
    staged.publish_json(
        f"cases/{document_id}/checkpoint.json",
        {
            "schemaVersion": 2,
            "documentId": document_id,
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
        "sourceTextSha256",
        "inputCandidateSha256",
        "sourceContractSha256",
        "stages",
        "result",
    }
    if not isinstance(value, dict) or set(value) != expected:
        raise ValueError(f"certification checkpoint has an invalid shape: {path}")
    if (
        value["schemaVersion"] != 2
        or value["documentId"] != document_id
        or value["sourceTextSha256"] != sha256_bytes(source.encode())
        or value["inputCandidateSha256"] != sha256_bytes(candidate.encode())
        or value["sourceContractSha256"] != sha256_bytes(canonical_json_bytes(contract))
    ):
        raise ValueError(f"certification checkpoint identity differs: {path}")
    stages = tuple(
        CertificationStage.model_validate_json(canonical_json_bytes(row), strict=True)
        for row in cast(list[Any], value["stages"])
    )
    result = CertificationCaseResult.model_validate_json(
        canonical_json_bytes(value["result"]), strict=True
    )
    audit = _host_audit(source=source, output=candidate, contract=contract)
    successful_outputs = [row.modelOutput for row in stages if row.modelOutput is not None]
    semantic_count = 0
    if successful_outputs:
        output = SemanticAuditOutput.model_validate_json(
            canonical_json_bytes(successful_outputs[-1]), strict=True
        )
        semantic_count = len(_validate_audit_output(output, current=candidate))
    if (
        result.documentId != document_id
        or result.sourceTextSha256 != sha256_bytes(source.encode())
        or result.inputCandidateSha256 != sha256_bytes(candidate.encode())
        or result.finalTextSha256 != result.inputCandidateSha256
        or result.hostAudit != audit
        or result.usage != _combined_usage(stages)
        or result.semanticFindings != semantic_count
        or result.auditPasses != (1 if successful_outputs else 0)
    ):
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
        f"- Needs review: **{sum(row.status == 'needs_review' for row in results)}**.",
        f"- Call failed: **{sum(row.status == 'call_failed' for row in results)}**.",
        f"- Semantic findings: **{sum(row.semanticFindings for row in results)}**.",
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
        "| Document | Status | Model findings | Host findings | Cost |",
        "|---|---|---:|---:|---:|",
    ]
    for row in results:
        rows.append(
            f"| `{row.documentId}` | {row.status} | {row.semanticFindings} | "
            f"{len(row.hostAudit.findings)} | ${row.usage.estimatedCostUsd} |"
        )
    rows.extend(
        (
            "",
            "A certified row has an unchanged input/output SHA-256, all deterministic host gates "
            "passing, and a provider-native structured audit with findings=[]. A non-empty "
            "finding blocks publication and must return to the template compiler; this stage "
            "never applies a repair.",
            "",
        )
    )
    return "\n".join(rows)


def run_raw_text_certification(
    *,
    project_root: Path,
    config_path: Path,
    config: SynthesisRawTextCertificationConfig,
) -> dict[str, JsonValue]:
    input_root = _validate_reference_run(
        project_root,
        config.input_run.path,
        config.input_run.commit_sha256,
        config.input_run.transaction_sha256,
    )
    contract_name = config.input_case_contract_filename
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
        ):
            path = case_root / name
            if path.is_symlink() or not path.is_file():
                raise ValueError(f"input certification artifact is not a regular file: {path}")

    transaction = {
        "schemaVersion": 2,
        "runId": config.run.run_id,
        "configSha256": sha256_file(config_path),
        "inputRunCommitSha256": config.input_run.commit_sha256,
        "inputRunTransactionSha256": config.input_run.transaction_sha256,
        "inputCaseContractFilename": contract_name,
        "promptSha256": config.prompt.sha256,
        "implementationSha256": sha256_file(_IMPLEMENTATION_PATH),
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
        },
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
    staged.publish_bytes("prompts/read-only-auditor.md", prompt_bytes)

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
        )
        if checkpoint is not None:
            results_by_id[document_id], stages_by_id[document_id] = checkpoint
    checkpointed_at_start = len(results_by_id)
    started = time.perf_counter()
    progress_lock = asyncio.Lock()
    limiter = asyncio.Semaphore(config.workflow.max_concurrent_documents)

    async def run_case(document_id: str) -> None:
        case_root = input_root / "cases" / document_id
        source = read_regular_file_bytes(case_root / "source.txt").decode("utf-8")
        candidate = read_regular_file_bytes(case_root / "final.txt").decode("utf-8")
        source_label = _read_json_object(case_root / "source-label.json")
        target_label = _read_json_object(case_root / "target-label.json")
        contract = _read_json_object(case_root / contract_name)
        host_audit = _host_audit(source=source, output=candidate, contract=contract)
        async with limiter:
            output, stages = await _call_audit(
                document_id=document_id,
                source=source,
                current=candidate,
                source_label=source_label,
                target_label=target_label,
                host_findings=host_audit.findings,
                model=model,
                config=config,
                prompt=prompt_bytes.decode("utf-8"),
            )
        if output is None:
            has_host_rejection = any(row.hostError is not None for row in stages)
            status: Literal["certified", "needs_review", "call_failed"] = (
                "needs_review" if has_host_rejection else "call_failed"
            )
            reason = (
                "Every structured audit response failed exact-evidence validation."
                if has_host_rejection
                else "Every configured provider route failed before a structured audit."
            )
            semantic_findings = 0
            audit_passes = 0
        else:
            findings = _validate_audit_output(
                output,
                current=candidate,
            )
            semantic_findings = len(findings)
            audit_passes = 1
            if not findings and host_audit.passed:
                status = "certified"
                reason = "Read-only semantic audit and every deterministic host gate passed."
            else:
                status = "needs_review"
                reason = (
                    "Read-only audit found semantic defects or a deterministic host gate failed; "
                    "the candidate must return to the compiler."
                )
        result = CertificationCaseResult(
            documentId=document_id,
            status=status,
            reason=reason,
            auditPasses=audit_passes,
            semanticFindings=semantic_findings,
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

    async def execute() -> None:
        try:
            await asyncio.gather(
                *(run_case(row) for row in config.case_ids if row not in results_by_id)
            )
        finally:
            await client.close()

    asyncio.run(execute())
    if set(results_by_id) != set(config.case_ids):
        raise RuntimeError("certification run returned an incomplete result set")
    results = tuple(results_by_id[row] for row in config.case_ids)
    all_stages = tuple(stage for row in config.case_ids for stage in stages_by_id[row])
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
        staged.publish_json(
            f"{prefix}/stages.json",
            [row.model_dump(mode="json") for row in stages_by_id[document_id]],
        )
        staged.publish_json(
            f"{prefix}/result.json", results_by_id[document_id].model_dump(mode="json")
        )
    summary: dict[str, JsonValue] = {
        "schemaVersion": 2,
        "runId": config.run.run_id,
        "status": "complete",
        "model": config.provider.model,
        "documents": len(results),
        "checkpointedDocumentsAtStart": checkpointed_at_start,
        "processedDocumentsThisInvocation": processed,
        "certifiedDocuments": sum(row.status == "certified" for row in results),
        "needsReviewDocuments": sum(row.status == "needs_review" for row in results),
        "callFailedDocuments": sum(row.status == "call_failed" for row in results),
        "semanticFindings": sum(row.semanticFindings for row in results),
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
