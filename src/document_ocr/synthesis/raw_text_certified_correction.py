"""Exact-line correction of synthetic OCR candidates from read-only audit evidence.

The semantic auditor remains read-only.  Its exact candidate citations define the complete
write boundary for this stage: one provider-native structured call may return a replacement for
each cited physical line, and no other byte can change.  The resulting candidate is deliberately
*not* declared training-ready here; it must pass a fresh independent certification run.
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
from pydantic import BaseModel, ConfigDict, Field, JsonValue, StringConstraints
from pydantic_ai import Agent, NativeOutput, StructuredDict, capture_run_messages
from pydantic_ai.messages import ModelResponse
from pydantic_ai.models import Model
from pydantic_ai.usage import UsageLimits

from document_ocr.atomic import read_regular_file_bytes
from document_ocr.hashing import canonical_json_bytes, sha256_bytes, sha256_file
from document_ocr.synthesis.config import SynthesisRawTextCertifiedCorrectionConfig
from document_ocr.synthesis.linguistic_probe_runtime import (
    LinguisticUsageReceipt,
    load_provider_key,
    model_messages,
    usage_receipt,
)
from document_ocr.synthesis.raw_text_certification import (
    CertificationCaseResult,
    CertificationStage,
    SemanticAuditFinding,
    SemanticAuditOutput,
    _host_audit,
    _target_literal_present,
    _validate_audit_output,
)
from document_ocr.synthesis.raw_text_hybrid_probe import (
    _artifact_inventory,
    _literal_occurrence_line_sets,
    _model_pair,
    _party_block_line_groups,
    _provider_attempt_routes,
    _retryable_route_error,
    _source_party,
)
from document_ocr.synthesis.raw_text_inventory import locate_auxiliary_values
from document_ocr.synthesis.raw_text_inventory_probe import _validate_reference_run
from document_ocr.synthesis.raw_text_rewrite_cycle_probe import (
    TargetValueOccurrenceRequirement,
    _combine_usage,
    _party_scalar_occurrence_count,
    _party_scalar_occurrence_line_sets,
    _settings,
    _target_value_occurrence_count,
    carrier_principal_template_slot_count,
    carrier_principal_template_slot_groups,
)
from document_ocr.synthesis.raw_text_rewrite_probe import unified_text_diff
from document_ocr.synthesis.run_safety import StagedArtifactRun
from document_ocr.training.config import resolve_config_path

NonEmptyText = Annotated[str, StringConstraints(strip_whitespace=True, min_length=1)]
LineId = Annotated[str, StringConstraints(pattern=r"^L[0-9]{5}$")]
_STRICT = ConfigDict(extra="forbid", frozen=True, strict=True, allow_inf_nan=False)
_IMPLEMENTATION_PATH = Path(__file__).resolve(strict=True)
_OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"
_PAGE_MARKER = re.compile(r"^--- PAGE [1-9][0-9]* ---[ \t]*$")
_OPAQUE_NUMERIC_LINE = re.compile(r"^[0-9]{5,}$")


class CorrectionLine(BaseModel):
    """One exact physical line made mutable by semantic-audit evidence."""

    model_config = _STRICT

    slot: NonEmptyText
    lineId: LineId
    sourceLine: str
    currentLine: Annotated[str, StringConstraints(min_length=1)]
    previousContext: tuple[str, ...]
    nextContext: tuple[str, ...]
    findings: tuple[SemanticAuditFinding, ...] = Field(min_length=1)


class CorrectionHostAudit(BaseModel):
    model_config = _STRICT

    passed: bool
    findings: tuple[NonEmptyText, ...]
    citedLines: Annotated[int, Field(gt=0)]
    changedLines: Annotated[int, Field(ge=0)]
    resolvedFindings: Annotated[int, Field(ge=0)]
    unresolvedFindings: Annotated[int, Field(ge=0)]


class CorrectionStage(BaseModel):
    model_config = _STRICT

    routeProvider: str | None
    semanticAttempt: Annotated[int, Field(ge=1)]
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
    hostAudit: CorrectionHostAudit | None
    errorType: str | None
    errorMessage: str | None


class CorrectionCaseResult(BaseModel):
    model_config = _STRICT

    documentId: str
    status: Literal[
        "unchanged_certified",
        "correction_candidate",
        "needs_review",
        "call_failed",
    ]
    reason: NonEmptyText
    citedLines: Annotated[int, Field(ge=0)]
    changedLines: Annotated[int, Field(ge=0)]
    sourceFindings: Annotated[int, Field(ge=0)]
    locallyResolvedFindings: Annotated[int, Field(ge=0)]
    requiresRecertification: bool
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


def _combined_usage(stages: Sequence[CorrectionStage]) -> LinguisticUsageReceipt:
    return _combine_usage([row.usage for row in stages]) if stages else _empty_usage()


def _read_json_object(path: Path) -> dict[str, Any]:
    value = json.loads(read_regular_file_bytes(path))
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path}")
    return value


def _line_number(line_id: str) -> int:
    if re.fullmatch(r"L[0-9]{5}", line_id) is None:
        raise ValueError(f"invalid line ID: {line_id!r}")
    return int(line_id[1:])


def _last_semantic_output(stages_path: Path, *, candidate: str) -> SemanticAuditOutput:
    value = json.loads(read_regular_file_bytes(stages_path))
    if not isinstance(value, list):
        raise ValueError(f"certification stages must be a list: {stages_path}")
    stages = tuple(
        CertificationStage.model_validate_json(canonical_json_bytes(row), strict=True)
        for row in value
    )
    outputs = [row.modelOutput for row in stages if row.modelOutput is not None]
    if not outputs:
        raise ValueError(f"certification case has no provider-valid semantic output: {stages_path}")
    output = SemanticAuditOutput.model_validate_json(canonical_json_bytes(outputs[-1]), strict=True)
    _validate_audit_output(output, current=candidate)
    return output


def _host_occurrence_findings(
    *, source: str, current: str, contract: Mapping[str, Any]
) -> tuple[SemanticAuditFinding, ...]:
    """Localize surplus target-party occurrences that a prior model audit omitted.

    The immutable rewrite contract records both the source/target scalar pair and its exact target
    cardinality.  Source occurrence spans identify each role-owned print block.  Within a source
    span, only the first target occurrence is the role scalar; additional copies are model-created
    duplicates.  Target occurrences outside every source role span are auxiliary copies of the
    primary identity and are likewise surplus.  This is a semantic-path rule, not a global
    first/last-line heuristic.
    """

    occurrence_rows = contract.get("targetValueOccurrenceRequirements")
    changed_leaves = contract.get("changedLeaves")
    if not isinstance(occurrence_rows, list) or not isinstance(changed_leaves, list):
        raise ValueError("source contract lacks occurrence requirements or changed leaves")
    leaves_by_path = {
        row["path"]: row
        for row in changed_leaves
        if isinstance(row, Mapping) and isinstance(row.get("path"), str)
    }
    current_lines = current.splitlines()
    source_label = contract.get("sourceLabel")
    findings: list[SemanticAuditFinding] = []
    for raw_requirement in occurrence_rows:
        if not isinstance(raw_requirement, Mapping):
            raise ValueError("invalid target occurrence requirement")
        requirement = TargetValueOccurrenceRequirement.model_validate_json(
            canonical_json_bytes(raw_requirement), strict=True
        )
        is_party_scalar = any(
            path.startswith("documentPatch.parties.")
            and path.rsplit(".", 1)[-1] in {"name", "address"}
            for path in requirement.targetPaths
        )
        if not is_party_scalar:
            continue
        observed = _target_value_occurrence_count(current, requirement)
        if observed <= requirement.requiredOccurrences:
            continue
        target_groups = _party_scalar_occurrence_line_sets(current, requirement.targetValue)
        source_groups: list[frozenset[int]] = []
        for path in requirement.targetPaths:
            leaf = leaves_by_path.get(path)
            source_value = leaf.get("sourceValue") if isinstance(leaf, Mapping) else None
            if not isinstance(source_value, str) or not source_value.strip():
                continue
            if (
                path.startswith("documentPatch.parties.")
                and path.endswith(".address")
                and isinstance(source_label, Mapping)
            ):
                source_groups.extend(
                    _party_address_occurrence_line_sets(
                        source,
                        path=path,
                        source_label=cast(Mapping[str, JsonValue], source_label),
                    )
                )
            elif path.startswith("documentPatch.parties."):
                source_groups.extend(_party_scalar_occurrence_line_sets(source, source_value))
            else:
                source_groups.extend(_literal_occurrence_line_sets(source, source_value))
        unique_source_groups = tuple(
            sorted(set(source_groups), key=lambda values: tuple(sorted(values)))
        )
        retained: set[frozenset[int]] = set()
        for source_group in unique_source_groups:
            candidates = sorted(
                (group for group in target_groups if group & source_group),
                key=lambda values: tuple(sorted(values)),
            )
            if candidates:
                retained.add(candidates[0])
        if len(retained) < requirement.requiredOccurrences:
            for group in target_groups:
                if group not in retained:
                    retained.add(group)
                if len(retained) == requirement.requiredOccurrences:
                    break
        surplus = tuple(group for group in target_groups if group not in retained)
        surplus_count = observed - requirement.requiredOccurrences
        if len(surplus) != surplus_count:
            # A wrapped occurrence can be represented by one line set while contributing more
            # than one normalized count. Without an exact one-to-one localization, do not grant
            # model write authority.
            continue
        evidence = tuple(
            {
                "lineId": f"L{number:05d}",
                "currentFragment": current_lines[number - 1][:1200],
            }
            for group in surplus
            for number in sorted(group)
        )
        if not evidence:
            continue
        address_requirement = all(path.endswith(".address") for path in requirement.targetPaths)
        resolution = (
            "the cited line duplicates the target address inside one role-owned address block; "
            "remove or reflow that duplicate while retaining the target address exactly once in "
            "the block and preserving its city and country"
            if address_requirement
            else (
                "the cited occurrence is an auxiliary duplicate and must become distinct "
                "fictional flavor of the same kind"
            )
        )
        findings.append(
            SemanticAuditFinding.model_validate(
                {
                    "findingKind": "party_or_legal_identity_mismatch",
                    "evidence": evidence,
                    "problem": (
                        f"The target party scalar {requirement.targetValue!r} occurs {observed} "
                        f"times but its role topology requires {requirement.requiredOccurrences}; "
                        f"{resolution}."
                    ),
                },
                strict=True,
            )
        )
    return tuple(findings)


def _evidence_for_lines(
    current: str, line_numbers: Sequence[int]
) -> tuple[dict[str, str], ...]:
    current_lines = current.splitlines()
    return tuple(
        {
            "lineId": f"L{number:05d}",
            "currentFragment": current_lines[number - 1][:1200],
        }
        for number in sorted(set(line_numbers))
        if 1 <= number <= len(current_lines)
        and current_lines[number - 1].strip()
        and _PAGE_MARKER.fullmatch(current_lines[number - 1]) is None
    )


def _changed_leaf_source_lines(
    *, source: str, contract: Mapping[str, Any], target_paths: Sequence[str]
) -> tuple[int, ...]:
    changed_leaves = contract.get("changedLeaves")
    if not isinstance(changed_leaves, list):
        raise ValueError("source contract lacks changedLeaves")
    wanted = set(target_paths)
    numbers: set[int] = set()
    for leaf in changed_leaves:
        if not isinstance(leaf, Mapping) or leaf.get("path") not in wanted:
            continue
        source_value = leaf.get("sourceValue")
        if not isinstance(source_value, str) or not source_value.strip():
            continue
        path = cast(str, leaf["path"])
        source_label = contract.get("sourceLabel")
        if path.startswith("documentPatch.parties.") and path.endswith(".address"):
            groups = (
                _party_address_occurrence_line_sets(
                    source,
                    path=path,
                    source_label=cast(Mapping[str, JsonValue], source_label),
                )
                if isinstance(source_label, Mapping)
                else _party_scalar_occurrence_line_sets(source, source_value)
            )
        elif path.startswith("documentPatch.parties.") and path.endswith(".name"):
            groups = _party_scalar_occurrence_line_sets(source, source_value)
        else:
            groups = _literal_occurrence_line_sets(source, source_value)
        for group in groups:
            numbers.update(group)
    return tuple(sorted(numbers))


def _party_address_occurrence_line_sets(
    text: str,
    *,
    path: str,
    source_label: Mapping[str, JsonValue],
) -> tuple[frozenset[int], ...]:
    """Resolve complete printed address sub-blocks without owning unrelated party lines.

    A normalized task address can represent several OCR lines and omit source-only street or
    building detail.  Starting at the matched party-name boundary and ending at the last exact
    address/city/country token captures that complete address realization.  The enclosing role
    block proves ownership; exact ordered-token matching proves both boundaries.  If either
    boundary cannot be proven, the function falls back to the literal scalar lines rather than
    broadening write authority.
    """

    resolved = _source_party(source_label, path)
    if resolved is None:
        return ()
    _role, party = resolved
    raw_address = party.get("address")
    if not isinstance(raw_address, str) or not raw_address.strip():
        return ()
    role_groups = _party_block_line_groups(text, path=path, source_label=source_label)
    address_groups = _party_scalar_occurrence_line_sets(text, raw_address)
    raw_name = party.get("name")
    name_groups = (
        _party_scalar_occurrence_line_sets(text, raw_name)
        if isinstance(raw_name, str) and raw_name.strip()
        else ()
    )
    locality_groups = tuple(
        occurrence
        for key in ("address", "city", "country")
        if isinstance((value := party.get(key)), str) and value.strip()
        for occurrence in _party_scalar_occurrence_line_sets(text, value)
    )
    expanded: list[frozenset[int]] = []
    for role_group in role_groups:
        role_addresses = tuple(group for group in address_groups if group <= role_group)
        if not role_addresses:
            continue
        role_localities = tuple(group for group in locality_groups if group <= role_group)
        role_names = tuple(group for group in name_groups if group <= role_group)
        for address_group in role_addresses:
            first_address = min(address_group)
            last_locality = max(
                (max(group) for group in role_localities if min(group) >= first_address),
                default=max(address_group),
            )
            preceding_names = tuple(
                group for group in role_names if max(group) <= first_address
            )
            first_line = (
                max(max(group) for group in preceding_names) + 1
                if preceding_names
                and max(max(group) for group in preceding_names) < first_address
                else first_address
            )
            owned = frozenset(
                number
                for number in range(first_line, last_locality + 1)
                if number in role_group
            )
            if owned:
                expanded.append(owned)
    if expanded:
        unique = {tuple(sorted(group)): group for group in expanded}
        return tuple(unique[key] for key in sorted(unique))
    return address_groups


def _refine_carrier_occurrence_contract(
    *, source: str, contract: Mapping[str, Any]
) -> dict[str, Any]:
    """Replay carrier name and address topology from the immutable source template."""

    refined = json.loads(canonical_json_bytes(contract))
    if not isinstance(refined, dict):
        raise ValueError("source contract is not a JSON object")
    occurrence_rows = refined.get("targetValueOccurrenceRequirements")
    changed_leaves = refined.get("changedLeaves")
    if not isinstance(occurrence_rows, list) or not isinstance(changed_leaves, list):
        raise ValueError("source contract lacks carrier occurrence inputs")
    source_carriers = {
        row.get("sourceValue")
        for row in changed_leaves
        if isinstance(row, Mapping)
        and row.get("path") == "documentPatch.parties.carrier.name"
        and isinstance(row.get("sourceValue"), str)
    }
    if len(source_carriers) > 1:
        raise ValueError("source contract has conflicting carrier-name source values")
    if source_carriers:
        source_carrier = cast(str, next(iter(source_carriers)))
        required = carrier_principal_template_slot_count(source, source_carrier)
        if required < 1:
            raise ValueError("carrier name has no role-owned source template slot")
        for row in occurrence_rows:
            if not isinstance(row, dict):
                raise ValueError("invalid target occurrence requirement")
            paths = row.get("targetPaths")
            if isinstance(paths, list) and "documentPatch.parties.carrier.name" in paths:
                previous = row.get("requiredOccurrences")
                if not isinstance(previous, int):
                    raise ValueError("carrier occurrence requirement lacks an integer cardinality")
                row["requiredOccurrences"] = max(previous, required)

    source_addresses = {
        row.get("sourceValue")
        for row in changed_leaves
        if isinstance(row, Mapping)
        and row.get("path") == "documentPatch.parties.carrier.address"
        and isinstance(row.get("sourceValue"), str)
    }
    if len(source_addresses) > 1:
        raise ValueError("source contract has conflicting carrier-address source values")
    if source_addresses:
        source_address = cast(str, next(iter(source_addresses)))
        address_required = _party_scalar_occurrence_count(source, source_address)
        if address_required < 1:
            raise ValueError("carrier address has no role-owned source template slot")
        for row in occurrence_rows:
            if not isinstance(row, dict):
                raise ValueError("invalid target occurrence requirement")
            paths = row.get("targetPaths")
            if isinstance(paths, list) and "documentPatch.parties.carrier.address" in paths:
                previous = row.get("requiredOccurrences")
                if not isinstance(previous, int):
                    raise ValueError("carrier address occurrence requirement lacks cardinality")
                row["requiredOccurrences"] = max(previous, address_required)
    return refined


def _deterministic_host_findings(
    *, source: str, current: str, contract: Mapping[str, Any]
) -> tuple[SemanticAuditFinding, ...]:
    """Convert deterministic host failures into exact, independently verifiable edit evidence.

    Certification reports host postconditions separately from model findings.  A corrector that
    ignores them cannot repair a missing target or surviving source-only identifier.  This
    adapter grants write authority only when the contract or the source parser resolves the
    failure to exact physical lines; unlocalizable topology failures remain hard errors.
    """

    host_messages = set(_host_audit(source=source, output=current, contract=contract).findings)
    findings: list[SemanticAuditFinding] = list(
        _host_occurrence_findings(source=source, current=current, contract=contract)
    )
    current_lines = current.splitlines()

    literal_rows = contract.get("targetLiteralRequirements")
    if not isinstance(literal_rows, list):
        raise ValueError("source contract lacks targetLiteralRequirements")
    for row in literal_rows:
        if not isinstance(row, Mapping) or not isinstance(row.get("targetValue"), str):
            raise ValueError("invalid target literal requirement")
        path = row.get("targetPath")
        target = cast(str, row["targetValue"])
        if _target_literal_present(
            text=current,
            target_path=path,
            target_value=target,
            match_policy=row.get("matchPolicy", "semantic_literal"),
        ):
            continue
        paths = (path,) if isinstance(path, str) else ()
        line_numbers = set(
            _changed_leaf_source_lines(
                source=source,
                contract=contract,
                target_paths=paths,
            )
        )
        normalized_target = " ".join(target.casefold().split())
        line_numbers.update(
            number
            for number, line in enumerate(current_lines, start=1)
            if normalized_target and normalized_target in " ".join(line.casefold().split())
        )
        evidence = _evidence_for_lines(current, tuple(line_numbers))
        if not evidence:
            raise ValueError(
                f"missing target literal cannot be localized safely: {path}: {target!r}"
            )
        findings.append(
            SemanticAuditFinding.model_validate(
                {
                    "findingKind": "target_fact_mismatch",
                    "evidence": evidence,
                    "problem": (
                        f"The exact target scalar {target!r} for {path} is missing; render it "
                        "only in the cited source-role line or lines."
                    ),
                },
                strict=True,
            )
        )

    occurrence_rows = contract.get("targetValueOccurrenceRequirements")
    if not isinstance(occurrence_rows, list):
        raise ValueError("source contract lacks targetValueOccurrenceRequirements")
    surplus_digests = {
        sha256_bytes(canonical_json_bytes(row.model_dump(mode="json"))) for row in findings
    }
    for raw_requirement in occurrence_rows:
        if not isinstance(raw_requirement, Mapping):
            raise ValueError("invalid target occurrence requirement")
        requirement = TargetValueOccurrenceRequirement.model_validate_json(
            canonical_json_bytes(raw_requirement), strict=True
        )
        observed = _target_value_occurrence_count(current, requirement)
        if observed == requirement.requiredOccurrences:
            continue
        if observed > requirement.requiredOccurrences and any(
            requirement.targetValue in row.problem
            and row.findingKind == "party_or_legal_identity_mismatch"
            for row in findings
        ):
            continue
        occurrence_line_numbers: set[int] = set()
        if observed > requirement.requiredOccurrences:
            groups = (
                _party_scalar_occurrence_line_sets(current, requirement.targetValue)
                if any(
                    path.startswith("documentPatch.parties.")
                    and path.rsplit(".", 1)[-1] in {"name", "address"}
                    for path in requirement.targetPaths
                )
                else _literal_occurrence_line_sets(current, requirement.targetValue)
            )
            for group in groups:
                occurrence_line_numbers.update(group)
            instruction = "remove only surplus auxiliary copies while preserving role-owned copies"
        else:
            occurrence_line_numbers.update(
                _changed_leaf_source_lines(
                    source=source,
                    contract=contract,
                    target_paths=requirement.targetPaths,
                )
            )
            if "documentPatch.parties.carrier.name" in requirement.targetPaths:
                changed_leaves = contract.get("changedLeaves")
                if not isinstance(changed_leaves, list):
                    raise ValueError("source contract lacks changedLeaves")
                source_values = {
                    row.get("sourceValue")
                    for row in changed_leaves
                    if isinstance(row, Mapping)
                    and row.get("path") == "documentPatch.parties.carrier.name"
                    and isinstance(row.get("sourceValue"), str)
                }
                if len(source_values) != 1:
                    raise ValueError("carrier occurrence repair lacks one source carrier value")
                for group in carrier_principal_template_slot_groups(
                    source, cast(str, next(iter(source_values)))
                ):
                    occurrence_line_numbers.update(group)
            instruction = (
                "restore the missing address exactly once in its role-owned address block; "
                "distribute address, city, and country across consecutive cited lines without "
                "duplicating the full target address"
                if all(path.endswith(".address") for path in requirement.targetPaths)
                else "restore the missing scalar only on its source-role line or lines"
            )
        evidence = _evidence_for_lines(current, tuple(occurrence_line_numbers))
        if not evidence:
            raise ValueError(
                "target occurrence mismatch cannot be localized safely: "
                f"{requirement.targetPaths}: {requirement.targetValue!r}"
            )
        finding = SemanticAuditFinding.model_validate(
            {
                "findingKind": "party_or_legal_identity_mismatch",
                "evidence": evidence,
                "problem": (
                    f"Target scalar {requirement.targetValue!r} occurs {observed} times but "
                    f"requires {requirement.requiredOccurrences}; {instruction}."
                ),
            },
            strict=True,
        )
        digest = sha256_bytes(canonical_json_bytes(finding.model_dump(mode="json")))
        if digest not in surplus_digests:
            findings.append(finding)
            surplus_digests.add(digest)

    for auxiliary in locate_auxiliary_values(source):
        expected_message = (
            "source-only auxiliary value survived anonymization: "
            f"{auxiliary.category!r} value={auxiliary.value!r}"
        )
        if expected_message not in host_messages:
            continue
        groups = _literal_occurrence_line_sets(current, auxiliary.value)
        if not groups:
            continue
        evidence = _evidence_for_lines(
            current, tuple(number for group in groups for number in group)
        )
        findings.append(
            SemanticAuditFinding.model_validate(
                {
                    "findingKind": "source_only_private_or_auxiliary_fact",
                    "evidence": evidence,
                    "problem": (
                        f"Source-only {auxiliary.category!r} value {auxiliary.value!r} survives; "
                        "replace it with a distinct fictional value of the same printed shape."
                    ),
                },
                strict=True,
            )
        )
    identity_rows = contract.get("rawAuxiliaryIdentityRequirements", [])
    if not isinstance(identity_rows, list):
        raise ValueError("invalid rawAuxiliaryIdentityRequirements")
    for row in identity_rows:
        identity = row.get("sourceIdentity") if isinstance(row, Mapping) else None
        if not isinstance(identity, str):
            raise ValueError("invalid raw auxiliary identity requirement")
        if (
            f"source-only auxiliary identity survived anonymization: {identity!r}"
            not in host_messages
        ):
            continue
        groups = _literal_occurrence_line_sets(current, identity)
        if not groups:
            continue
        evidence = _evidence_for_lines(
            current, tuple(number for group in groups for number in group)
        )
        findings.append(
            SemanticAuditFinding.model_validate(
                {
                    "findingKind": "source_only_private_or_auxiliary_fact",
                    "evidence": evidence,
                    "problem": (
                        f"Source-only auxiliary identity {identity!r} survives; replace it with "
                        "one coherent fictional identity while retaining the legal relationship."
                    ),
                },
                strict=True,
            )
        )

    unique: dict[str, SemanticAuditFinding] = {}
    for finding in findings:
        digest = sha256_bytes(canonical_json_bytes(finding.model_dump(mode="json")))
        unique[digest] = finding
    return tuple(unique[key] for key in sorted(unique))


def _correction_lines(
    *, source: str, current: str, findings: Sequence[SemanticAuditFinding], context_lines: int = 2
) -> tuple[CorrectionLine, ...]:
    source_lines = source.splitlines()
    current_lines = current.splitlines()
    if len(source_lines) != len(current_lines):
        raise ValueError("correction requires stable source/current line topology")
    findings_by_line: dict[str, list[SemanticAuditFinding]] = {}
    for finding in findings:
        for evidence in finding.evidence:
            findings_by_line.setdefault(evidence.lineId, []).append(finding)
            if finding.findingKind not in {
                "source_only_private_or_auxiliary_fact",
                "party_or_legal_identity_mismatch",
                "cargo_or_dangerous_goods_mismatch",
            }:
                continue
            fragment = evidence.currentFragment
            if len(fragment.strip()) < 8:
                continue
            for number, line in enumerate(current_lines, start=1):
                if fragment in line:
                    findings_by_line.setdefault(f"L{number:05d}", []).append(finding)
    rows: list[CorrectionLine] = []
    for ordinal, line_id in enumerate(sorted(findings_by_line, key=_line_number)):
        number = _line_number(line_id)
        if not 1 <= number <= len(current_lines):
            raise ValueError(f"correction line is outside the document: {line_id}")
        current_line = current_lines[number - 1]
        if not current_line.strip() or _PAGE_MARKER.fullmatch(current_line) is not None:
            raise ValueError(f"correction cannot own a blank/page-marker line: {line_id}")
        unique: dict[str, SemanticAuditFinding] = {}
        for finding in findings_by_line[line_id]:
            digest = sha256_bytes(canonical_json_bytes(finding.model_dump(mode="json")))
            unique[digest] = finding
        start = max(0, number - 1 - context_lines)
        stop = min(len(current_lines), number + context_lines)
        rows.append(
            CorrectionLine(
                slot=f"s{ordinal}",
                lineId=line_id,
                sourceLine=source_lines[number - 1],
                currentLine=current_line,
                previousContext=tuple(current_lines[start : number - 1]),
                nextContext=tuple(current_lines[number:stop]),
                findings=tuple(unique[key] for key in sorted(unique)),
            )
        )
    if findings and not rows:
        raise ValueError("semantic findings yielded no exact correction lines")
    return tuple(rows)


def _refresh_rows(rows: Sequence[CorrectionLine], current: str) -> tuple[CorrectionLine, ...]:
    """Rebind a stable correction slot to the latest uncommitted candidate text."""

    current_lines = current.splitlines()
    refreshed: list[CorrectionLine] = []
    for row in rows:
        number = _line_number(row.lineId)
        if not 1 <= number <= len(current_lines):
            raise ValueError(f"repair line is outside the candidate: {row.lineId}")
        start = max(0, number - 3)
        stop = min(len(current_lines), number + 2)
        refreshed.append(
            row.model_copy(
                update={
                    "currentLine": current_lines[number - 1],
                    "previousContext": tuple(current_lines[start : number - 1]),
                    "nextContext": tuple(current_lines[number:stop]),
                }
            )
        )
    return tuple(refreshed)


def _select_repair_rows(
    *,
    rows: Sequence[CorrectionLine],
    source: str,
    current: str,
    contract: Mapping[str, Any],
    host_error: str,
    unresolved_findings: Sequence[SemanticAuditFinding] = (),
) -> tuple[tuple[CorrectionLine, ...], tuple[SemanticAuditFinding, ...]]:
    """Select the smallest exact-line retry surface justified by a host rejection."""

    row_by_slot = {row.slot: row for row in rows}
    row_by_line = {row.lineId: row for row in rows}
    selected_slots = {
        match
        for match in re.findall(r"(?<![A-Za-z0-9])(s[0-9]+)(?![A-Za-z0-9])", host_error)
        if match in row_by_slot
    }
    selected_slots.update(
        row_by_line[evidence.lineId].slot
        for finding in unresolved_findings
        for evidence in finding.evidence
        if evidence.lineId in row_by_line
    )
    repair_findings = _deterministic_host_findings(
        source=source,
        current=current,
        contract=contract,
    )
    for finding in repair_findings:
        selected_slots.update(
            row_by_line[evidence.lineId].slot
            for evidence in finding.evidence
            if evidence.lineId in row_by_line
        )
    if not selected_slots:
        # A host error without an exact line citation cannot safely expand write authority.  The
        # retry may still regenerate the already-authorized set, but never any other line.
        selected_slots.update(row_by_slot)
    selected = tuple(row for row in rows if row.slot in selected_slots)
    return _refresh_rows(selected, current), repair_findings


def _failed_application_audit(
    *, rows: Sequence[CorrectionLine], message: str
) -> CorrectionHostAudit:
    return CorrectionHostAudit(
        passed=False,
        findings=(message,),
        citedLines=len(rows),
        changedLines=0,
        resolvedFindings=0,
        unresolvedFindings=sum(len(row.findings) for row in rows),
    )


def _output_schema(
    rows: Sequence[CorrectionLine],
) -> tuple[type[dict[str, Any]], dict[str, JsonValue]]:
    properties = {
        row.slot: {
            "type": "string",
            "minLength": 1,
            "maxLength": 10000,
            "pattern": r".*\S.*",
            "description": (
                f"Complete final text for {row.lineId}; echo currentLine exactly if it is "
                "comparison evidence rather than the defective line. No newline."
            ),
        }
        for row in rows
    }
    schema: dict[str, JsonValue] = {
        "type": "object",
        "properties": cast(JsonValue, properties),
        "required": cast(JsonValue, [row.slot for row in rows]),
        "additionalProperties": False,
    }
    return (
        StructuredDict(
            cast(Any, schema),
            name="CertifiedLineCorrections",
            description="Complete text for every exact line cited by the read-only auditor.",
        ),
        schema,
    )


def _payload(
    *,
    document_id: str,
    source_label: Mapping[str, Any],
    target_label: Mapping[str, Any],
    rows: Sequence[CorrectionLine],
    repair_context: Mapping[str, JsonValue] | None = None,
) -> dict[str, JsonValue]:
    payload: dict[str, JsonValue] = {
        "task": (
            "Resolve every independently audited defect using only the exact cited lines. "
            "Return complete final line strings; the host applies them atomically."
        ),
        "documentId": document_id,
        "rules": cast(
            JsonValue,
            [
                "The synthetic target label is authoritative for every labeled fact.",
                (
                    "A source-only shipment-specific value or identity must be replaced with a "
                    "distinct plausible fictional value of the same semantic kind and format."
                ),
                (
                    "Preserve all unrelated wording, field labels that remain jurisdictionally "
                    "valid, punctuation, casing style, spacing, units, and OCR line topology."
                ),
                (
                    "If a jurisdiction-specific field label no longer matches the target, make "
                    "the label jurisdiction-neutral while preserving its field function."
                ),
                (
                    "Recompute repeated counts, weights, volumes, equipment summaries, route "
                    "wording, reefer state, and dangerous-goods wording from the target."
                ),
                (
                    "Do not infer freight payment place from prepaid or collect; arrangement and "
                    "place are independent fields."
                ),
                (
                    "Do not validate a newly fictionalized auxiliary identifier against an "
                    "external registry; preserve its printed kind and surface shape."
                ),
                "Never insert N/A, UNKNOWN, UNAVAILABLE, NULL, commentary, or model-control prose.",
                (
                    "Some cited lines are comparison evidence. Return those currentLine strings "
                    "byte-for-byte unchanged unless they themselves contain the defect."
                ),
                "Return every required slot exactly once and no other key.",
            ],
        ),
        "sourceTaskLabel": cast(JsonValue, source_label),
        "syntheticTargetLabel": cast(JsonValue, target_label),
        "lineSlots": cast(
            JsonValue,
            [
                {
                    "slot": row.slot,
                    "lineId": row.lineId,
                    "sourceLine": row.sourceLine,
                    "currentLine": row.currentLine,
                    "previousContext": list(row.previousContext),
                    "nextContext": list(row.nextContext),
                    "findings": [item.model_dump(mode="json") for item in row.findings],
                }
                for row in rows
            ],
        ),
    }
    if repair_context is not None:
        payload["repairContext"] = cast(JsonValue, dict(repair_context))
    return payload


def _edge_whitespace(value: str) -> tuple[str, str]:
    leading = cast(re.Match[str], re.match(r"^[ \t]*", value)).group(0)
    trailing = cast(re.Match[str], re.search(r"[ \t]*$", value)).group(0)
    return leading, trailing


def _apply_corrections(
    *,
    source: str,
    current: str,
    rows: Sequence[CorrectionLine],
    findings: Sequence[SemanticAuditFinding],
    output: Mapping[str, Any],
    contract: Mapping[str, Any],
    baseline_current: str | None = None,
) -> tuple[str, CorrectionHostAudit]:
    expected = {row.slot for row in rows}
    if set(output) != expected:
        raise ValueError("correction output keys differ from the exact required slot set")
    lines = current.splitlines(keepends=True)
    line_by_id = {row.lineId: row for row in rows}
    replacement_by_line: dict[str, str] = {}
    for row in rows:
        replacement = output[row.slot]
        if not isinstance(replacement, str) or not replacement.strip():
            raise ValueError(f"correction slot is not a non-empty string: {row.slot}")
        if "\n" in replacement or "\r" in replacement:
            raise ValueError(f"correction slot contains a newline: {row.slot}")
        if _edge_whitespace(replacement) != _edge_whitespace(row.currentLine):
            raise ValueError(f"correction slot changes line-edge whitespace: {row.slot}")
        current_body = row.currentLine.strip()
        replacement_body = replacement.strip()
        if _OPAQUE_NUMERIC_LINE.fullmatch(current_body) is not None and (
            _OPAQUE_NUMERIC_LINE.fullmatch(replacement_body) is None
            or len(replacement_body) != len(current_body)
        ):
            raise ValueError(
                f"correction slot changes an opaque numeric identifier shape: {row.slot}"
            )
        replacement_by_line[row.lineId] = replacement
        index = _line_number(row.lineId) - 1
        raw = lines[index]
        ending = raw[len(raw.rstrip("\r\n")) :]
        lines[index] = replacement + ending
    candidate = "".join(lines)
    baseline_bodies = (baseline_current if baseline_current is not None else current).splitlines()
    candidate_bodies = candidate.splitlines()
    if len(baseline_bodies) != len(candidate_bodies):
        raise ValueError("correction baseline and candidate line topology differ")
    changed_ids = {
        f"L{number:05d}"
        for number, (before, after) in enumerate(
            zip(baseline_bodies, candidate_bodies, strict=True), start=1
        )
        if before != after
    }
    resolved = 0
    unresolved = 0
    for finding in findings:
        evidence_ids = {row.lineId for row in finding.evidence}
        if evidence_ids & changed_ids:
            resolved += 1
        else:
            unresolved += 1
    host = _host_audit(source=source, output=candidate, contract=contract)
    unresolved_message = (
        f"correction left {unresolved} semantic finding(s) wholly unchanged"
        if unresolved
        else None
    )
    audit = CorrectionHostAudit(
        # A finding may cite comparison lines, but at least one of its evidence lines must change.
        # Otherwise the response has made no attempt to resolve that audited defect and a bounded
        # targeted retry is cheaper and safer than another whole-document certification call.
        passed=host.passed and unresolved == 0,
        findings=host.findings + ((unresolved_message,) if unresolved_message else ()),
        citedLines=len(line_by_id),
        changedLines=len(changed_ids),
        resolvedFindings=resolved,
        unresolvedFindings=unresolved,
    )
    return candidate, audit


def _findings_without_changed_evidence(
    *,
    baseline: str,
    candidate: str,
    findings: Sequence[SemanticAuditFinding],
) -> tuple[SemanticAuditFinding, ...]:
    baseline_lines = baseline.splitlines()
    candidate_lines = candidate.splitlines()
    if len(baseline_lines) != len(candidate_lines):
        raise ValueError("correction baseline and candidate line topology differ")
    changed_ids = {
        f"L{number:05d}"
        for number, (before, after) in enumerate(
            zip(baseline_lines, candidate_lines, strict=True), start=1
        )
        if before != after
    }
    return tuple(
        finding
        for finding in findings
        if not ({row.lineId for row in finding.evidence} & changed_ids)
    )


def _transient_route_error(error: Exception) -> bool:
    # Keep the route policy centralized.  OpenRouter's parameter/tool-compatibility rejection is
    # an HTTP 404 for one pinned endpoint and must advance to the next configured endpoint.  A
    # second status filter here previously made the shared classifier's explicit 404 rule dead.
    return _retryable_route_error(error)


def _retry_delay(
    *, document_id: str, route_round: int, config: SynthesisRawTextCertifiedCorrectionConfig
) -> float:
    if route_round <= 1:
        return 0.0
    workflow = config.workflow
    retry_index = route_round - 2
    base = workflow.retry_initial_delay_seconds * (workflow.retry_delay_multiplier**retry_index)
    bounded = min(base, workflow.retry_max_delay_seconds)
    digest = bytes.fromhex(sha256_bytes(f"{document_id}\0{route_round}\0correction-v1".encode()))
    fraction = int.from_bytes(digest[:8], "big") / float(2**64 - 1)
    return min(workflow.retry_max_delay_seconds, bounded + fraction * workflow.retry_jitter_seconds)


async def _call_model(
    *,
    document_id: str,
    source_label: Mapping[str, Any],
    target_label: Mapping[str, Any],
    rows: Sequence[CorrectionLine],
    model: Model,
    config: SynthesisRawTextCertifiedCorrectionConfig,
    prompt: str,
    semantic_attempt: int,
    repair_context: Mapping[str, JsonValue] | None = None,
) -> tuple[Mapping[str, Any] | None, tuple[CorrectionStage, ...]]:
    payload = _payload(
        document_id=document_id,
        source_label=source_label,
        target_label=target_label,
        rows=rows,
        repair_context=repair_context,
    )
    output_type, schema = _output_schema(rows)
    output_spec = NativeOutput(
        output_type,
        name="certified_line_corrections",
        description="Complete final text for every exact cited OCR line.",
        strict=True,
    )
    routes = _provider_attempt_routes(config.provider)
    active_routes = routes
    stages: list[CorrectionStage] = []
    attempt = 0
    for route_round in range(1, config.workflow.max_provider_route_rounds + 1):
        delay = _retry_delay(document_id=document_id, route_round=route_round, config=config)
        if delay:
            await asyncio.sleep(delay)
        transient_routes: list[str | None] = []
        for route_provider in active_routes:
            attempt += 1
            agent = Agent[None, dict[str, Any]](
                model,
                output_type=output_spec,
                system_prompt=prompt,
                model_settings=cast(
                    Any,
                    _settings(
                        config.provider,
                        stage="editor",
                        prompt_sha256=config.prompt.sha256,
                        route_provider=route_provider,
                    ),
                ),
                retries={"output": 0},
                name="synthetic-bl-certified-line-corrector",
            )
            started = time.time()
            output: Mapping[str, Any] | None = None
            error: Exception | None = None
            with capture_run_messages() as captured:
                try:
                    result = await agent.run(
                        json.dumps(payload, ensure_ascii=False, separators=(",", ":")),
                        usage_limits=UsageLimits(
                            request_limit=1,
                            output_tokens_limit=config.provider.max_output_tokens,
                        ),
                    )
                    output = cast(Mapping[str, Any], result.output)
                except Exception as caught:
                    error = caught
            responses = tuple(row for row in captured if isinstance(row, ModelResponse))
            stages.append(
                CorrectionStage(
                    routeProvider=route_provider,
                    semanticAttempt=semantic_attempt,
                    routeRound=route_round,
                    routeAttempt=attempt,
                    retryDelayBeforeSeconds=delay if route_provider == active_routes[0] else 0.0,
                    inputPayloadSha256=sha256_bytes(canonical_json_bytes(payload)),
                    outputSchemaSha256=sha256_bytes(canonical_json_bytes(schema)),
                    startedAtUnixSeconds=started,
                    completedAtUnixSeconds=time.time(),
                    usage=usage_receipt(
                        responses, config.provider.pricing, require_provider_cost=False
                    ),
                    messages=model_messages(captured),
                    modelOutput=cast(JsonValue, dict(output)) if output is not None else None,
                    hostAudit=None,
                    errorType=type(error).__name__ if error is not None else None,
                    errorMessage=str(error) if error is not None else None,
                )
            )
            if output is not None:
                return output, tuple(stages)
            if error is None or not _transient_route_error(error):
                return None, tuple(stages)
            transient_routes.append(route_provider)
        if not transient_routes:
            break
        active_routes = tuple(transient_routes)
    return None, tuple(stages)


def _attach_host_audit(
    stages: Sequence[CorrectionStage], audit: CorrectionHostAudit
) -> tuple[CorrectionStage, ...]:
    updated = list(stages)
    for index in range(len(updated) - 1, -1, -1):
        if updated[index].modelOutput is not None:
            updated[index] = updated[index].model_copy(update={"hostAudit": audit})
            return tuple(updated)
    raise ValueError("correction host audit has no successful provider response")


def _report(results: Sequence[CorrectionCaseResult], stages: Sequence[CorrectionStage]) -> str:
    usage = _combined_usage(stages)
    count = max(1, len(results))
    unchanged = sum(row.status == "unchanged_certified" for row in results)
    candidates = sum(row.status == "correction_candidate" for row in results)
    needs_review = sum(row.status == "needs_review" for row in results)
    call_failed = sum(row.status == "call_failed" for row in results)
    cited_lines = sum(row.citedLines for row in results)
    changed_lines = sum(row.changedLines for row in results)
    estimated_rate = usage.estimatedCostUsd * Decimal(1000) / count
    return "\n".join(
        (
            "# Exact-evidence raw-text correction",
            "",
            "## Outcome",
            "",
            f"- Input documents: **{len(results)}**.",
            f"- Previously certified and copied unchanged: **{unchanged}**.",
            (f"- Correction candidates requiring independent recertification: **{candidates}**."),
            f"- Needs review after local host checks: **{needs_review}**.",
            f"- Call failed: **{call_failed}**.",
            f"- Exact cited / changed lines: **{cited_lines} / {changed_lines}**.",
            f"- Requests: **{usage.requests}**.",
            (
                "- Maximum successful model responses per corrected document: "
                f"**{max((row.semanticAttempt for row in stages), default=0)} observed**."
            ),
            (
                "- Input / reasoning / visible tokens: "
                f"**{usage.inputTokens:,} / {usage.reasoningTokens:,} / "
                f"{usage.visibleOutputTokens:,}**."
            ),
            (
                f"- Pinned-price estimate: **${usage.estimatedCostUsd}** "
                f"(${estimated_rate:.3f}/1,000)."
            ),
            (
                "- Provider-reported cost: unavailable for at least one successful response."
                if usage.providerReportedCostUsd is None
                else f"- Provider-reported cost: **${usage.providerReportedCostUsd}** "
                f"(${usage.providerReportedCostUsd * Decimal(1000) / count:.3f}/1,000)."
            ),
            "",
            (
                "Only auditor-cited physical lines were mutable. This stage does not certify or "
                "publish training data; every changed candidate must pass a fresh independent "
                "whole-document audit."
            ),
            "",
        )
    )


def run_raw_text_certified_correction(
    *,
    project_root: Path,
    config_path: Path,
    config: SynthesisRawTextCertifiedCorrectionConfig,
) -> dict[str, JsonValue]:
    certification_root = _validate_reference_run(
        project_root,
        config.certification_run.path,
        config.certification_run.commit_sha256,
        config.certification_run.transaction_sha256,
    )
    prompt_path = resolve_config_path(project_root, config.prompt.path)
    if (
        prompt_path.is_symlink()
        or not prompt_path.is_file()
        or sha256_file(prompt_path) != config.prompt.sha256
    ):
        raise ValueError("certified-correction prompt differs from its configured pin")
    prompt_bytes = read_regular_file_bytes(prompt_path)
    case_material: dict[str, dict[str, Any]] = {}
    for document_id in config.case_ids:
        case_root = certification_root / "cases" / document_id
        if not case_root.is_dir():
            raise ValueError(f"certification run lacks correction case {document_id}")
        required = (
            "source.txt",
            "final.txt",
            "source-label.json",
            "target-label.json",
            "source-contract.json",
            "stages.json",
            "result.json",
        )
        if any(
            (case_root / name).is_symlink() or not (case_root / name).is_file() for name in required
        ):
            raise ValueError(f"certification case is incomplete: {document_id}")
        source = read_regular_file_bytes(case_root / "source.txt").decode("utf-8")
        current = read_regular_file_bytes(case_root / "final.txt").decode("utf-8")
        result = CertificationCaseResult.model_validate_json(
            read_regular_file_bytes(case_root / "result.json"), strict=True
        )
        if result.documentId != document_id or result.inputCandidateSha256 != sha256_bytes(
            current.encode()
        ):
            raise ValueError(f"certification candidate identity differs: {document_id}")
        audit_output = _last_semantic_output(case_root / "stages.json", candidate=current)
        model_findings = _validate_audit_output(audit_output, current=current)
        contract = _refine_carrier_occurrence_contract(
            source=source,
            contract=_read_json_object(case_root / "source-contract.json"),
        )
        occurrence_findings = _host_occurrence_findings(
            source=source,
            current=current,
            contract=contract,
        )
        deterministic_findings = _deterministic_host_findings(
            source=source,
            current=current,
            contract=contract,
        )
        unique_findings: dict[str, SemanticAuditFinding] = {}
        for finding in (*model_findings, *occurrence_findings, *deterministic_findings):
            digest = sha256_bytes(canonical_json_bytes(finding.model_dump(mode="json")))
            unique_findings[digest] = finding
        findings = tuple(unique_findings[key] for key in sorted(unique_findings))
        current_host_audit = _host_audit(source=source, output=current, contract=contract)
        already_certified = not findings and current_host_audit.passed
        if result.status == "certified" and model_findings:
            raise ValueError(f"certified case carries semantic findings: {document_id}")
        if not already_certified and not findings:
            raise ValueError(
                f"non-certified case has no actionable semantic findings: {document_id}"
            )
        case_material[document_id] = {
            "root": case_root,
            "source": source,
            "current": current,
            "sourceLabel": _read_json_object(case_root / "source-label.json"),
            "targetLabel": _read_json_object(case_root / "target-label.json"),
            "contract": contract,
            "findings": findings,
            "rows": _correction_lines(source=source, current=current, findings=findings),
            "certificationResult": result,
            "alreadyCertified": already_certified,
        }

    transaction = {
        "schemaVersion": 1,
        "runId": config.run.run_id,
        "configSha256": sha256_file(config_path),
        "certificationRunCommitSha256": config.certification_run.commit_sha256,
        "certificationRunTransactionSha256": config.certification_run.transaction_sha256,
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
            "maxSuccessfulModelResponsesPerDocument": (
                config.workflow.max_successful_model_responses_per_document
            ),
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
    staged.publish_bytes("prompts/corrector.md", prompt_bytes)

    key = load_provider_key(project_root, config.environment_file, config.provider.api_key_env)
    client_arguments: dict[str, Any] = {
        "api_key": key,
        "max_retries": config.provider.transport_max_retries,
        "timeout": config.provider.request_timeout_seconds,
    }
    if config.provider.kind == "openrouter":
        client_arguments.update(
            {"base_url": _OPENROUTER_BASE_URL, "default_headers": {"X-Title": "DocumentParsing"}}
        )
    client = AsyncOpenAI(**client_arguments)
    model, _ = _model_pair(
        client=client, editor_provider=config.provider, reviewer_provider=config.provider
    )
    results: dict[str, CorrectionCaseResult] = {}
    stages_by_id: dict[str, tuple[CorrectionStage, ...]] = {}
    outputs: dict[str, str] = {}
    limiter = asyncio.Semaphore(config.workflow.max_concurrent_documents)
    progress_lock = asyncio.Lock()
    started = time.perf_counter()

    async def run_case(document_id: str) -> None:
        material = case_material[document_id]
        source = cast(str, material["source"])
        current = cast(str, material["current"])
        findings = cast(tuple[SemanticAuditFinding, ...], material["findings"])
        rows = cast(tuple[CorrectionLine, ...], material["rows"])
        already_certified = cast(bool, material["alreadyCertified"])
        stage_rows: tuple[CorrectionStage, ...] = ()
        final = current
        if already_certified:
            status: Literal[
                "unchanged_certified", "correction_candidate", "needs_review", "call_failed"
            ] = "unchanged_certified"
            reason = (
                "Input candidate was already independently certified and remained byte-identical."
            )
            changed = 0
            locally_resolved = 0
        else:
            all_stage_rows: list[CorrectionStage] = []
            working_current = current
            active_rows = rows
            repair_context: Mapping[str, JsonValue] | None = None
            status = "needs_review"
            reason = "The bounded correction responses were exhausted without a valid candidate."
            async with limiter:
                for semantic_attempt in range(
                    1, config.workflow.max_successful_model_responses_per_document + 1
                ):
                    output, attempt_stages = await _call_model(
                        document_id=document_id,
                        source_label=cast(Mapping[str, Any], material["sourceLabel"]),
                        target_label=cast(Mapping[str, Any], material["targetLabel"]),
                        rows=active_rows,
                        model=model,
                        config=config,
                        prompt=prompt_bytes.decode("utf-8"),
                        semantic_attempt=semantic_attempt,
                        repair_context=repair_context,
                    )
                    if output is None:
                        all_stage_rows.extend(attempt_stages)
                        status = "call_failed"
                        reason = (
                            "Every configured provider route failed before a structured correction."
                        )
                        break
                    try:
                        candidate, audit = _apply_corrections(
                            source=source,
                            current=working_current,
                            rows=active_rows,
                            findings=findings,
                            output=output,
                            contract=cast(Mapping[str, Any], material["contract"]),
                            baseline_current=current,
                        )
                    except Exception as error:
                        host_error = f"{type(error).__name__}: {error}"
                        failed_audit = _failed_application_audit(
                            rows=active_rows,
                            message=host_error,
                        )
                        all_stage_rows.extend(_attach_host_audit(attempt_stages, failed_audit))
                        if (
                            semantic_attempt
                            >= config.workflow.max_successful_model_responses_per_document
                        ):
                            reason = (
                                "Structured correction failed deterministic host application: "
                                f"{host_error}"
                            )
                            break
                        active_rows, repair_findings = _select_repair_rows(
                            rows=rows,
                            source=source,
                            current=working_current,
                            contract=cast(Mapping[str, Any], material["contract"]),
                            host_error=host_error,
                        )
                        repair_context = {
                            "semanticAttempt": semantic_attempt + 1,
                            "hostRejection": host_error,
                            "hostFindings": cast(
                                JsonValue,
                                [row.model_dump(mode="json") for row in repair_findings],
                            ),
                            "instruction": (
                                "Correct only the resubmitted slots. The other first-response "
                                "lines are retained unchanged by the host."
                            ),
                        }
                        continue
                    all_stage_rows.extend(_attach_host_audit(attempt_stages, audit))
                    working_current = candidate
                    if audit.passed:
                        status = "correction_candidate"
                        reason = (
                            "Every exact-line local gate passed; independent recertification is "
                            "required."
                        )
                        break
                    if (
                        semantic_attempt
                        >= config.workflow.max_successful_model_responses_per_document
                    ):
                        reason = (
                            "The bounded correction responses exhausted deterministic local "
                            "postconditions."
                        )
                        break
                    host_error = "; ".join(audit.findings)
                    active_rows, repair_findings = _select_repair_rows(
                        rows=rows,
                        source=source,
                        current=working_current,
                        contract=cast(Mapping[str, Any], material["contract"]),
                        host_error=host_error,
                        unresolved_findings=_findings_without_changed_evidence(
                            baseline=current,
                            candidate=working_current,
                            findings=findings,
                        ),
                    )
                    repair_context = {
                        "semanticAttempt": semantic_attempt + 1,
                        "hostRejection": host_error,
                        "hostFindings": cast(
                            JsonValue,
                            [row.model_dump(mode="json") for row in repair_findings],
                        ),
                        "instruction": (
                            "Correct only the resubmitted slots. The other first-response lines "
                            "are retained unchanged by the host."
                        ),
                    }
            stage_rows = tuple(all_stage_rows)
            final = working_current
            current_lines = current.splitlines()
            final_lines = final.splitlines()
            if len(current_lines) != len(final_lines):
                raise RuntimeError("correction loop changed physical line topology")
            changed_line_ids = {
                f"L{number:05d}"
                for number, (before, after) in enumerate(
                    zip(current_lines, final_lines, strict=True), start=1
                )
                if before != after
            }
            changed = len(changed_line_ids)
            locally_resolved = sum(
                bool({evidence.lineId for evidence in finding.evidence} & changed_line_ids)
                for finding in findings
            )
        result = CorrectionCaseResult(
            documentId=document_id,
            status=status,
            reason=reason,
            citedLines=len(rows),
            changedLines=changed,
            sourceFindings=len(findings),
            locallyResolvedFindings=locally_resolved,
            requiresRecertification=status == "correction_candidate",
            sourceTextSha256=sha256_bytes(source.encode()),
            inputCandidateSha256=sha256_bytes(current.encode()),
            finalTextSha256=sha256_bytes(final.encode()),
            usage=_combined_usage(stage_rows),
        )
        async with progress_lock:
            results[document_id] = result
            stages_by_id[document_id] = stage_rows
            outputs[document_id] = final
            elapsed = time.perf_counter() - started
            rate = len(results) / elapsed if elapsed else 0.0
            print(
                json.dumps(
                    {
                        "command": "run-raw-text-certified-correction",
                        "phase": "exact_line_correction",
                        "processed_documents": len(results),
                        "remaining_documents": len(config.case_ids) - len(results),
                        "correction_candidates": sum(
                            row.status == "correction_candidate" for row in results.values()
                        ),
                        "needs_review_documents": sum(
                            row.status == "needs_review" for row in results.values()
                        ),
                        "call_failed_documents": sum(
                            row.status == "call_failed" for row in results.values()
                        ),
                        "elapsed_seconds": round(elapsed, 3),
                        "throughput_documents_per_hour": round(rate * 3600, 3),
                        "eta_seconds": round((len(config.case_ids) - len(results)) / rate, 3)
                        if rate
                        else None,
                        "status": "progress",
                    },
                    allow_nan=False,
                    sort_keys=True,
                ),
                flush=True,
            )

    async def execute() -> None:
        try:
            await asyncio.gather(*(run_case(document_id) for document_id in config.case_ids))
        finally:
            await client.close()

    asyncio.run(execute())
    if set(results) != set(config.case_ids):
        raise RuntimeError("certified correction returned an incomplete result set")
    ordered_results = tuple(results[document_id] for document_id in config.case_ids)
    all_stages = tuple(
        stage for document_id in config.case_ids for stage in stages_by_id[document_id]
    )
    usage = _combined_usage(all_stages)
    wall_seconds = time.perf_counter() - started
    for document_id in config.case_ids:
        material = case_material[document_id]
        source = cast(str, material["source"])
        current = cast(str, material["current"])
        final = outputs[document_id]
        prefix = f"cases/{document_id}"
        staged.publish_bytes(f"{prefix}/source.txt", source.encode())
        staged.publish_bytes(f"{prefix}/input-candidate.txt", current.encode())
        staged.publish_bytes(f"{prefix}/final.txt", final.encode())
        staged.publish_bytes(
            f"{prefix}/diff-from-input.patch", unified_text_diff(current, final).encode()
        )
        staged.publish_bytes(
            f"{prefix}/diff-from-source.patch", unified_text_diff(source, final).encode()
        )
        source_case_root = cast(Path, material["root"])
        for source_name, target_name in (
            ("source-label.json", "source-label.json"),
            ("target-label.json", "target-label.json"),
        ):
            staged.publish_json(
                f"{prefix}/{target_name}",
                _read_json_object(source_case_root / source_name),
            )
        staged.publish_json(
            f"{prefix}/source-contract.json",
            cast(Mapping[str, Any], material["contract"]),
        )
        staged.publish_json(
            f"{prefix}/audit-findings.json",
            [
                row.model_dump(mode="json")
                for row in cast(tuple[SemanticAuditFinding, ...], material["findings"])
            ],
        )
        staged.publish_json(
            f"{prefix}/stages.json",
            [row.model_dump(mode="json") for row in stages_by_id[document_id]],
        )
        staged.publish_json(f"{prefix}/result.json", results[document_id].model_dump(mode="json"))
    summary: dict[str, JsonValue] = {
        "schemaVersion": 1,
        "runId": config.run.run_id,
        "status": "complete",
        "model": config.provider.model,
        "documents": len(ordered_results),
        "unchangedCertifiedDocuments": sum(
            row.status == "unchanged_certified" for row in ordered_results
        ),
        "correctionCandidateDocuments": sum(
            row.status == "correction_candidate" for row in ordered_results
        ),
        "needsReviewDocuments": sum(row.status == "needs_review" for row in ordered_results),
        "callFailedDocuments": sum(row.status == "call_failed" for row in ordered_results),
        "sourceFindings": sum(row.sourceFindings for row in ordered_results),
        "citedLines": sum(row.citedLines for row in ordered_results),
        "changedLines": sum(row.changedLines for row in ordered_results),
        "requests": usage.requests,
        "providerAttempts": len(all_stages),
        "failedProviderAttempts": sum(row.errorType is not None for row in all_stages),
        "inputTokens": usage.inputTokens,
        "reasoningTokens": usage.reasoningTokens,
        "visibleOutputTokens": usage.visibleOutputTokens,
        "outputTokens": usage.outputTokens,
        "estimatedCostUsd": str(usage.estimatedCostUsd),
        "providerReportedCostUsd": str(usage.providerReportedCostUsd)
        if usage.providerReportedCostUsd is not None
        else None,
        "estimatedCostPerThousandUsd": str(
            usage.estimatedCostUsd * Decimal(1000) / len(ordered_results)
        ),
        "providerReportedCostPerThousandUsd": (
            str(usage.providerReportedCostUsd * Decimal(1000) / len(ordered_results))
            if usage.providerReportedCostUsd is not None
            else None
        ),
        "wallSeconds": round(wall_seconds, 6),
        "throughputDocumentsPerHour": round(len(ordered_results) / wall_seconds * 3600, 6),
        "trainingRecordsPublished": False,
    }
    staged.publish_json("summary.json", summary)
    staged.publish_bytes(
        "generation/results.jsonl",
        b"".join(
            canonical_json_bytes(row.model_dump(mode="json")) + b"\n" for row in ordered_results
        ),
    )
    staged.publish_bytes("REPORT.md", _report(ordered_results, all_stages).encode())
    staged.publish_json("provenance/transaction.json", transaction)
    staged.commit(
        expected_artifacts=_artifact_inventory(staged.stage_root),
        metadata={
            "schemaVersion": 1,
            "documents": len(ordered_results),
            "correctionCandidateDocuments": cast(int, summary["correctionCandidateDocuments"]),
            "trainingRecordsPublished": False,
        },
    )
    output = dict(summary)
    output["artifactRoot"] = str(staged.final_root)
    output["commitSha256"] = sha256_file(staged.final_root / "_COMMIT.json")
    return output
