"""Compiler-first, fail-closed probe for synthetic raw-OCR rewriting.

This is deliberately an experiment rather than a training-data publisher.  It measures
deterministic rewrite coverage over fifty source/target pairs, compiles the richer contracts from
an earlier audited atomic run into host-owned residual spans, and permits at most two one-shot
editor/reviewer model pairs.  The provider never receives the complete OCR or complete labels.
"""

from __future__ import annotations

import asyncio
import copy
import json
import os
import re
import time
from collections import Counter, defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from decimal import Decimal
from importlib.metadata import version
from pathlib import Path
from typing import Annotated, Any, Literal, cast

from openai import AsyncOpenAI
from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    JsonValue,
    StringConstraints,
    create_model,
)
from pydantic_ai import (
    Agent,
    ModelProfile,
    ModelRetry,
    NativeOutput,
    capture_run_messages,
)
from pydantic_ai.exceptions import ModelAPIError, ModelHTTPError, UnexpectedModelBehavior
from pydantic_ai.messages import ModelResponse
from pydantic_ai.models import Model
from pydantic_ai.models.openai import OpenAIResponsesModel
from pydantic_ai.models.openrouter import OpenRouterModel
from pydantic_ai.profiles import merge_profile
from pydantic_ai.providers.openai import OpenAIProvider
from pydantic_ai.providers.openrouter import OpenRouterProvider
from pydantic_ai.usage import UsageLimits

from document_ocr.atomic import read_regular_file_bytes
from document_ocr.hashing import canonical_json_bytes, sha256_bytes, sha256_file
from document_ocr.synthesis.config import (
    CommittedArtifactDirectoryConfig,
    RawTextRewriteCycleProvidersConfig,
    RawTextRewriteProviderConfig,
    SynthesisRawTextHybridProbeConfig,
)
from document_ocr.synthesis.linguistic_probe_runtime import (
    LinguisticUsageReceipt,
    load_provider_key,
    model_messages,
    usage_receipt,
)
from document_ocr.synthesis.raw_text_rewrite_cycle_probe import (
    AppliedDeterministicPrefill,
    AtomicRewriteCommit,
    CargoFlavorRewriteRequirement,
    CompoundPartyFlavorRealization,
    DeterministicRewriteAudit,
    LineRangeReplacement,
    RewriteCycleBundle,
    RewriteWorkspace,
    SurfaceRenderingRequirement,
    _combine_usage,
    _date_context_path,
    _line_body,
    _line_ending,
    _line_id,
    _line_number,
    _literal_phrase_pattern,
    _numeric_span_overlaps_date,
    _parse_measurement_surface,
    _settings,
    apply_deterministic_prefills,
    apply_line_range_replacements,
    carrier_principal_template_slot_groups,
    deterministic_rewrite_audit,
)
from document_ocr.synthesis.raw_text_rewrite_probe import (
    ChangedLeaf,
    _load_jsonl,
    _resolve_pinned_file,
    changed_leaves,
    unified_text_diff,
)
from document_ocr.synthesis.run_safety import StagedArtifactRun
from document_ocr.synthesis.task_adapter import (
    BILL_OF_LADING_TASK_ADAPTER,
    BILL_OF_LADING_V5_TASK_ADAPTER,
)
from document_ocr.training.config import resolve_config_path

NonEmptyText = Annotated[str, StringConstraints(strip_whitespace=True, min_length=1)]
EvidenceText = Annotated[str, StringConstraints(min_length=1, max_length=600)]
Sha256 = Annotated[str, StringConstraints(pattern=r"^[0-9a-f]{64}$")]
_STRICT = ConfigDict(extra="forbid", frozen=True, strict=True, allow_inf_nan=False)
_IMPLEMENTATION_PATH = Path(__file__).resolve(strict=True)
_OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"
_PAGE_MARKER = re.compile(r"^--- PAGE [1-9][0-9]* ---[ \t]*$")
_SPACE_RUN = re.compile(r"\s+")
_NUMERIC_TOKEN = re.compile(
    r"(?<![0-9.,])[-+\N{MINUS SIGN}]?[0-9]+(?:[.,][0-9]+)*(?![0-9])"
)
_ATTACHED_NUMERIC_UNIT = re.compile(
    r"^(?:KGS?|KGM|KILOGRAMS?|CBM|M3|CUM|CUFT|MT|TONS?|"
    r"PKGS?|PACKAGES?|PLTS?|PALLETS?|CTNS?|CARTONS?|PCS?|PIECES?|"
    r"BAGS?|BALES?|BOX(?:ES)?|CRATES?|CASES?|DRUMS?|ROLLS?|BUNDLES?|"
    r"SETS?|LOTS?|UNITS?|SACKS?|JERRICANS?|TINS?|CANS?|BARRELS?|[CF])\b",
    re.IGNORECASE,
)
_TEMPERATURE_CONTEXT = re.compile(
    r"\b(?:TEMP(?:ERATURE)?|SET[ -]?POINT|CELSIUS|CENTIGRADE|FAHRENHEIT)\b",
    re.IGNORECASE,
)
_FORWARDING_REFERENCE_PATH = re.compile(
    r"^documentPatch\.forwardingAndExportReferences\[[0-9]+\]$"
)
_FORWARDING_REFERENCE_CONTEXT = re.compile(
    r"\b(?:EXPORT(?:[ \t]+REFERENCES?)?|INVOICE|SHIPPING[ \t]+BILL|"
    r"SB[ \t]*(?:NO\.?|NUMBER)|CUSTOMS|REFERENCE|REF(?:ERENCE)?[ \t]*NO|"
    r"F/?AGENT[^\r\n]{0,30}\bREF|SHIPPER'?S[ \t]+REF)\b",
    re.IGNORECASE,
)


class HybridWorkItem(BaseModel):
    """One semantic delta and the host-owned spans in which it may be realized."""

    model_config = _STRICT

    workItemId: Annotated[str, StringConstraints(pattern=r"^W[0-9]{4}$")]
    targetPaths: Annotated[tuple[NonEmptyText, ...], Field(min_length=1)]
    action: NonEmptyText
    sourceValue: JsonValue
    targetValue: JsonValue
    state: Literal[
        "deterministic_applied",
        "preserved_by_policy",
        "agent_residual",
        "blocked_unlocated",
    ]
    evidenceLineIds: tuple[Annotated[str, StringConstraints(pattern=r"^L[0-9]{5}$")], ...]
    spanIds: tuple[Annotated[str, StringConstraints(pattern=r"^S[0-9]{3}$")], ...]
    locator: Literal[
        "deterministic_requirement",
        "exact_literal",
        "flexible_whitespace_literal",
        "numeric_surface",
        "relation_scoped_numeric_surface",
        "thermal_context_numeric_surface",
        "freight_arrangement_surface",
        "forwarding_reference_context",
        "party_role_block",
        "literal_in_party_role_block",
        "rendered_surface_and_party_role_block",
        "carrier_principal_template_slots",
        "wrapped_carrier_signature_literal",
        "location_role_surface",
        "relation_scoped_surface",
        "sibling_object_evidence",
        "policy_projection",
        "unlocated",
    ]
    rationale: NonEmptyText


class ResidualSpan(BaseModel):
    """A non-overlapping OCR interval owned by the host, not selected by the model."""

    model_config = _STRICT

    spanId: Annotated[str, StringConstraints(pattern=r"^S[0-9]{3}$")]
    startLine: Annotated[int, Field(ge=1)]
    endLine: Annotated[int, Field(ge=1)]
    workItemIds: Annotated[tuple[NonEmptyText, ...], Field(min_length=1)]
    lines: tuple[str, ...]


class AppliedCompilerEdit(BaseModel):
    model_config = _STRICT

    requirementId: NonEmptyText
    kind: Literal["operational_surface", "jurisdiction_surface"]
    lineId: Annotated[str, StringConstraints(pattern=r"^L[0-9]{5}$")]
    sourceSurface: NonEmptyText
    targetSurface: NonEmptyText
    beforeLine: str
    afterLine: str


class HybridReviewFinding(BaseModel):
    model_config = _STRICT

    spanId: Annotated[str, StringConstraints(pattern=r"^S[0-9]{3}$")]
    workItemIds: Annotated[tuple[NonEmptyText, ...], Field(min_length=1, max_length=8)]
    evidence: EvidenceText
    correction: Annotated[
        str,
        StringConstraints(strip_whitespace=True, min_length=1, max_length=500),
    ]


class HybridReviewReceipt(BaseModel):
    """Compact independent review over residual spans only."""

    model_config = _STRICT

    verdict: Literal["pass", "revise"]
    semanticChanges: Literal["pass", "fail"]
    staleSourceFacts: Literal["pass", "fail"]
    formattingAndTopology: Literal["pass", "fail"]
    findings: Annotated[tuple[HybridReviewFinding, ...], Field(max_length=12)]


class HybridModelStage(BaseModel):
    model_config = _STRICT

    stage: Literal["residual_editor", "residual_reviewer"]
    providerModel: NonEmptyText
    routeProvider: NonEmptyText | None
    reasoningEffort: NonEmptyText
    inputPayloadSha256: Sha256
    startedAtUnixSeconds: float
    completedAtUnixSeconds: float
    usage: LinguisticUsageReceipt
    messages: JsonValue
    errorType: NonEmptyText | None
    errorMessage: NonEmptyText | None


def _provider_attempt_routes(provider: RawTextRewriteProviderConfig) -> tuple[str | None, ...]:
    """Return explicit, observable routes for one semantic model stage.

    OpenRouter's own fallback is retained for simple one-route configurations. When an ordered
    route contains multiple endpoints, each attempt is pinned independently so an upstream 429 or
    5xx cannot silently defeat failover and every failed transport attempt remains in the artifact.
    """

    if (
        provider.kind != "openrouter"
        or not provider.allow_fallbacks
        or provider.provider_order is None
        or len(provider.provider_order) < 2
    ):
        return (None,)
    return tuple(provider.provider_order)


def _retryable_route_error(error: Exception) -> bool:
    """Whether an endpoint-specific failure is safe to retry on the next configured route."""

    # A provider can accept the native-output request yet spend its complete output budget on
    # hidden reasoning without returning the required object. PydanticAI correctly exposes that
    # as unexpected model behaviour rather than an HTTP error. It is still endpoint-specific:
    # the next explicitly pinned implementation of the same model may satisfy the schema. Every
    # failed attempt is receipted, so continuing does not hide or duplicate a successful write.
    if isinstance(error, UnexpectedModelBehavior):
        return True
    if not isinstance(error, ModelAPIError):
        return False
    # PydanticAI wraps DNS, connection-reset, and timeout failures without an HTTP status in the
    # base ModelAPIError. They are endpoint-transient and should continue through the explicitly
    # configured routes. HTTP subclasses retain the narrower status policy below.
    if not isinstance(error, ModelHTTPError):
        return True
    if error.status_code in {408, 425, 429} or error.status_code >= 500:
        return True
    if error.status_code != 404 or not isinstance(error.body, Mapping):
        return False
    metadata = error.body.get("metadata")
    return isinstance(metadata, Mapping) and metadata.get("failed_routing_step") in {
        "Filter by Parameters",
        "Filter by Tool Compatibility",
    }


class HybridCaseResult(BaseModel):
    model_config = _STRICT

    documentId: NonEmptyText
    status: Literal["quality_validated", "needs_review", "call_failed", "compiler_blocked"]
    reason: NonEmptyText
    sourceLines: Annotated[int, Field(ge=1)]
    residualLines: Annotated[int, Field(ge=0)]
    residualSpans: Annotated[int, Field(ge=0)]
    workItems: Annotated[int, Field(ge=1)]
    deterministicWorkItems: Annotated[int, Field(ge=0)]
    residualWorkItems: Annotated[int, Field(ge=0)]
    blockedWorkItems: Annotated[int, Field(ge=0)]
    deterministicPrefills: Annotated[int, Field(ge=0)]
    compilerEdits: Annotated[int, Field(ge=0)]
    outputTextSha256: Sha256
    baselineOutputTextSha256: Sha256 | None
    exactBaselineMatch: bool | None
    baselineReferenceGapLines: tuple[Annotated[int, Field(ge=1)], ...] | None
    deterministicAudit: DeterministicRewriteAudit | None
    review: HybridReviewReceipt | None
    usage: LinguisticUsageReceipt


@dataclass(frozen=True, slots=True)
class _CompiledCase:
    bundle: RewriteCycleBundle
    workspace: RewriteWorkspace
    work_items: tuple[HybridWorkItem, ...]
    spans: tuple[ResidualSpan, ...]
    compiler_edits: tuple[AppliedCompilerEdit, ...]


@dataclass(slots=True)
class _ResidualCommitContext:
    workspace: RewriteWorkspace
    spans: dict[str, ResidualSpan]
    authorized_line_ids: frozenset[str]


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


def _validate_committed_run(
    project_root: Path, configured: CommittedArtifactDirectoryConfig
) -> Path:
    root = resolve_config_path(project_root, configured.path)
    commit = root / "_COMMIT.json"
    if root.is_symlink() or not root.is_dir() or commit.is_symlink() or not commit.is_file():
        raise ValueError(f"baseline atomic run is not a committed regular directory: {root}")
    if sha256_file(commit) != configured.commit_sha256:
        raise ValueError("baseline atomic run commit SHA-256 differs")
    staged = StagedArtifactRun(
        output_parent=root.parent,
        run_name=root.name,
        transaction_sha256=configured.transaction_sha256,
    )
    staged.validate_committed_run()
    return root.resolve(strict=True)


def _semantic_surface(value: Any) -> str:
    return " ".join(str(value).casefold().split())


def _party_match_surface(value: Any) -> str:
    """Canonicalize only layout-insensitive party punctuation for block matching.

    Party labels normalize punctuation that OCR often prints as separate tokens (``NO.33`` versus
    ``NO. 33``), and company names commonly interchange ``and`` with ``&``.  Party-block
    discovery needs those representations to compare equal, but it must not use fuzzy matching:
    every alphanumeric token still has to occur in the same order.
    """

    expanded = str(value).replace("&", " AND ").casefold()
    return " ".join(re.findall(r"[^\W_]+", expanded))


def _root_path(path: str) -> str:
    relative = path.removeprefix("documentPatch.")
    return relative.split(".", 1)[0].split("[", 1)[0]


def _literal_classification(
    raw_text: str,
    leaf: ChangedLeaf,
    targets_by_source: Mapping[str, frozenset[str]],
) -> str:
    source = leaf.sourceValue
    target = leaf.targetValue
    if not isinstance(source, str) or not isinstance(target, str):
        return "non_string_or_absent"
    if not source or not target:
        return "non_string_or_absent"
    exact = raw_text.count(source)
    if exact:
        suffix = "unique" if exact == 1 else "multiple"
        collision = len(targets_by_source.get(_semantic_surface(source), frozenset())) > 1
        return f"exact_{suffix}_{'conflicting' if collision else 'consistent'}"
    pattern = re.compile(re.escape(source).replace(r"\ ", r"\s+"), re.IGNORECASE)
    matches = tuple(pattern.finditer(raw_text))
    if matches:
        return "flexible_whitespace_unique" if len(matches) == 1 else "flexible_whitespace_multiple"
    return "not_literal"


def audit_literal_coverage(
    source_rows: Sequence[Mapping[str, Any]],
    target_rows: Sequence[Mapping[str, Any]],
    *,
    limit: int,
) -> tuple[dict[str, JsonValue], tuple[dict[str, JsonValue], ...]]:
    """Measure discoverability only; no candidate in this census is authorized to mutate text."""

    source_by_id = {cast(str, row["documentId"]): row for row in source_rows}
    details: list[dict[str, JsonValue]] = []
    overall = Counter[str]()
    roots: dict[str, Counter[str]] = defaultdict(Counter)
    total_leaves = 0
    direct_literal = 0
    for target_row in target_rows[:limit]:
        document_id = cast(str, target_row["baseDocumentId"])
        source_row = source_by_id.get(document_id)
        if source_row is None:
            raise ValueError(f"synthetic target has no source corpus row: {document_id}")
        raw_text = source_row.get("joinedRawText")
        if not isinstance(raw_text, str):
            raise ValueError(f"source corpus row has no joinedRawText: {document_id}")
        source_label = BILL_OF_LADING_TASK_ADAPTER.validate_target(
            document_id=document_id,
            target=cast(Mapping[str, Any], source_row["target"]),
        )
        target_label = BILL_OF_LADING_V5_TASK_ADAPTER.validate_target(
            document_id=cast(str, target_row["scenarioId"]),
            target=cast(Mapping[str, Any], target_row["target"]),
        )
        leaves = tuple(
            row
            for row in changed_leaves(source_label, target_label)
            if row.requiresTextEdit and row.sourceValue != row.targetValue
        )
        targets_by_source: dict[str, set[str]] = defaultdict(set)
        for leaf in leaves:
            if isinstance(leaf.sourceValue, str) and isinstance(leaf.targetValue, str):
                targets_by_source[_semantic_surface(leaf.sourceValue)].add(
                    _semantic_surface(leaf.targetValue)
                )
        frozen_targets = {key: frozenset(values) for key, values in targets_by_source.items()}
        counts = Counter(_literal_classification(raw_text, leaf, frozen_targets) for leaf in leaves)
        literal_count = sum(value for key, value in counts.items() if key.startswith("exact_"))
        total_leaves += len(leaves)
        direct_literal += literal_count
        overall.update(counts)
        for leaf in leaves:
            classification = _literal_classification(raw_text, leaf, frozen_targets)
            roots[_root_path(leaf.path)][classification] += 1
        details.append(
            {
                "documentId": document_id,
                "changedTextLeaves": len(leaves),
                "exactLiteralLeaves": literal_count,
                "exactLiteralFraction": literal_count / len(leaves) if leaves else 0.0,
                "classifications": dict(sorted(counts.items())),
            }
        )
    summary: dict[str, JsonValue] = {
        "documents": len(details),
        "changedTextLeaves": total_leaves,
        "exactLiteralCandidates": direct_literal,
        "exactLiteralCandidateFraction": direct_literal / total_leaves if total_leaves else 0.0,
        "classificationCounts": cast(JsonValue, dict(sorted(overall.items()))),
        "byRoot": cast(
            JsonValue,
            {
                root: {
                    "leaves": sum(counts.values()),
                    "exactLiteralCandidates": sum(
                        value for key, value in counts.items() if key.startswith("exact_")
                    ),
                    "classifications": dict(sorted(counts.items())),
                }
                for root, counts in sorted(roots.items())
            },
        ),
    }
    return summary, tuple(details)


def _workspace_from_bundle(bundle: RewriteCycleBundle) -> RewriteWorkspace:
    workspace = RewriteWorkspace(
        original_text=bundle.sourceText,
        current_text=bundle.sourceText,
        scenario_id=bundle.result.scenarioId,
        source_label=copy.deepcopy(bundle.sourceLabel),
        upstream_target_label=copy.deepcopy(bundle.upstreamTargetLabel),
        current_target_label=copy.deepcopy(bundle.targetLabel),
        surface_requirements=bundle.surfaceRenderingRequirements,
        target_literal_requirements=bundle.targetLiteralRequirements,
        target_value_occurrence_requirements=bundle.targetValueOccurrenceRequirements,
        anchored_scalar_replacement_requirements=(bundle.anchoredScalarReplacementRequirements),
        source_role_hints=bundle.sourceSemanticRoleHints,
        inline_slot_requirements=bundle.inlineSlotTopologyRequirements,
        source_status_requirements=bundle.sourceStatusPreservationRequirements,
        jurisdictional_requirements=bundle.jurisdictionalSurfaceRequirements,
        raw_auxiliary_identity_requirements=bundle.rawAuxiliaryIdentityRequirements,
        operational_flavor_requirements=bundle.operationalFlavorRequirements,
        cargo_flavor_rewrite_requirements=bundle.cargoFlavorRewriteRequirements,
    )
    apply_deterministic_prefills(workspace)
    _apply_party_role_country_prefills(workspace)
    return workspace


def _replace_one_line_surface(
    workspace: RewriteWorkspace,
    *,
    line_number: int,
    source_surface: str,
    target_surface: str,
) -> tuple[str, str] | None:
    lines = workspace.current_text.splitlines(keepends=True)
    if not 1 <= line_number <= len(lines):
        raise ValueError(f"compiler line is outside OCR: {line_number}")
    before = _line_body(lines[line_number - 1])
    if source_surface not in before:
        if target_surface in before:
            return None
        raise ValueError(f"compiler cannot locate {source_surface!r} on line {line_number}")
    if before.count(source_surface) != 1:
        raise ValueError(
            f"compiler source surface is not unique on line {line_number}: {source_surface!r}"
        )
    after = before.replace(source_surface, target_surface, 1)
    lines[line_number - 1] = after + _line_ending(lines[line_number - 1])
    workspace.current_text = "".join(lines)
    return before, after


def _render_route_neutral_customs_heading(
    match: re.Match[str],
    *,
    registered_source: str,
    generic_target: str,
) -> str:
    """Preserve the observed heading's case and delimiter while changing its jurisdiction."""

    observed = match.group(0)
    observed_letters = "".join(character for character in observed if character.isalpha())
    rendered = generic_target
    if observed_letters.islower():
        rendered = rendered.lower()
    elif observed_letters.istitle():
        rendered = rendered.title()
    if registered_source.rstrip().endswith(("-", ":", "#")):
        delimiter = re.search(r"[ \t]*[-:#]+[ \t]*$", observed)
        if delimiter is None:
            raise ValueError("registered customs delimiter was absent from its exact match")
        rendered += delimiter.group(0)
    return rendered


def _apply_compiler_requirements(
    workspace: RewriteWorkspace,
) -> tuple[AppliedCompilerEdit, ...]:
    """Apply only requirements with an explicit line or exact audited occurrence count."""

    edits: list[AppliedCompilerEdit] = []
    # Operational requirements are applied exactly once by ``apply_deterministic_prefills``.
    # That renderer re-resolves the named measurement group at its proven grammar/column. A
    # second scalar pass here used to search for the old value anywhere on the already-mutated
    # line; short values such as ``1`` could therefore corrupt a newly generated seal, phone
    # number, or legal clause. Jurisdictional requirements below are the only compiler-stage
    # edits because they are deliberately excluded from the deterministic prefill renderer.

    jurisdiction_line_owners = Counter(
        line_id
        for requirement in workspace.jurisdictional_requirements
        for line_id in set(requirement.sourceLineIds)
    )
    for requirement in workspace.jurisdictional_requirements:
        lines = workspace.current_text.splitlines(keepends=True)
        source_pattern = _literal_phrase_pattern(requirement.sourceSurface)
        target_pattern = _literal_phrase_pattern(requirement.targetSurface)
        planned: list[tuple[int, str, str, str]] = []
        deterministic = len(requirement.sourceLineIds) == requirement.sourceOccurrences
        for line_id in requirement.sourceLineIds:
            if jurisdiction_line_owners[line_id] != 1:
                # Several named-program selectors on one line form one legal/customs clause.
                # Replacing each token independently can retain a contradictory assertion.
                deterministic = False
                break
            line_number = _line_number(line_id)
            if not 1 <= line_number <= len(lines):
                deterministic = False
                break
            before = _line_body(lines[line_number - 1])
            matches = tuple(source_pattern.finditer(before))
            if not matches:
                if target_pattern.search(before) is not None:
                    continue
                deterministic = False
                break
            # Multiple references on one line are a legal/customs clause, not a scalar label.
            # It needs one scoped linguistic rewrite so the whole jurisdictional assertion—not
            # merely an acronym—is made route-neutral.
            if len(matches) != 1:
                deterministic = False
                break
            rendered_target = _render_route_neutral_customs_heading(
                matches[0],
                registered_source=requirement.sourceSurface,
                generic_target=requirement.targetSurface,
            )
            after = (
                before[: matches[0].start()]
                + rendered_target
                + before[matches[0].end() :]
            )
            planned.append((line_number, before, after, rendered_target))
        if not deterministic:
            continue
        for line_number, before, after, rendered_target in planned:
            ending = _line_ending(lines[line_number - 1])
            lines[line_number - 1] = after + ending
            workspace.current_text = "".join(lines)
            edits.append(
                AppliedCompilerEdit(
                    requirementId=requirement.requirementId,
                    kind="jurisdiction_surface",
                    lineId=f"L{line_number:05d}",
                    sourceSurface=requirement.sourceSurface,
                    targetSurface=rendered_target,
                    beforeLine=before,
                    afterLine=after,
                )
            )
    if len(workspace.current_text.splitlines()) != len(workspace.original_text.splitlines()):
        raise ValueError("compiler requirement edits changed OCR line count")
    return tuple(edits)


def _offset_line_numbers(text: str, start: int, end: int) -> set[int]:
    first = text.count("\n", 0, start) + 1
    last = text.count("\n", 0, max(start, end - 1)) + 1
    return set(range(first, last + 1))


def _literal_line_numbers(text: str, value: str) -> tuple[set[int], str]:
    lines: set[int] = set()
    cursor = 0
    while True:
        offset = text.find(value, cursor)
        if offset < 0:
            break
        lines.update(_offset_line_numbers(text, offset, offset + len(value)))
        cursor = offset + len(value)
    if lines:
        return lines, "exact_literal"
    pieces = [piece for piece in _SPACE_RUN.split(value.strip()) if piece]
    if not pieces:
        return set(), "unlocated"
    pattern = re.compile(r"\s+".join(re.escape(piece) for piece in pieces), re.IGNORECASE)
    for match in pattern.finditer(text):
        lines.update(_offset_line_numbers(text, match.start(), match.end()))
    if lines:
        return lines, "flexible_whitespace_literal"

    # OCR may insert a physical line break inside one lexical token (for example
    # ``250.0000`` becomes ``250.\n0000``).  Build an offset-preserving view with only
    # whitespace removed and accept an exact, case-insensitive match in that view.  This is
    # still a literal locator: punctuation and every non-whitespace character must match in
    # order.  It is deliberately not fuzzy matching.
    compact_text_chars: list[str] = []
    compact_offsets: list[int] = []
    for offset, character in enumerate(text):
        if character.isspace():
            continue
        compact_text_chars.append(character)
        compact_offsets.append(offset)
    compact_value = "".join(character for character in value if not character.isspace())
    if not compact_value:
        return set(), "unlocated"
    compact_text = "".join(compact_text_chars)
    cursor = 0
    while True:
        compact_offset = compact_text.casefold().find(compact_value.casefold(), cursor)
        if compact_offset < 0:
            break
        start = compact_offsets[compact_offset]
        end = compact_offsets[compact_offset + len(compact_value) - 1] + 1
        lines.update(_offset_line_numbers(text, start, end))
        cursor = compact_offset + len(compact_value)
    return (lines, "flexible_whitespace_literal") if lines else (set(), "unlocated")


def _container_identifier_line_numbers(text: str, value: str) -> set[int]:
    """Locate an ISO-style container identifier despite OCR separator noise.

    Container numbers are semantic identifiers rather than prose. OCR commonly renders the same
    eleven alphanumeric characters with spaces or a check-digit separator (for example
    ``MCLU 510204.9``). Removing non-alphanumeric separators from both the label value and one
    physical OCR line is therefore an exact identifier comparison, not fuzzy matching.
    """

    expected = "".join(character for character in value.upper() if character.isalnum())
    # Production ISO 6346 identifiers contain eleven characters. Unit fixtures and a small
    # number of source labels can omit the check digit, so retain exact comparison for the
    # ten-character stem as well; shorter strings would be too collision-prone.
    if len(expected) not in {10, 11}:
        return set()
    return {
        number
        for number, line in enumerate(text.splitlines(), start=1)
        if expected in "".join(character for character in line.upper() if character.isalnum())
    }


def _literal_occurrence_line_sets(text: str, value: str) -> tuple[frozenset[int], ...]:
    """Locate each exact or whitespace-normalized occurrence independently."""

    matches: list[frozenset[int]] = []
    cursor = 0
    while True:
        offset = text.find(value, cursor)
        if offset < 0:
            break
        matches.append(frozenset(_offset_line_numbers(text, offset, offset + len(value))))
        cursor = offset + len(value)
    if matches:
        return tuple(matches)
    pieces = [piece for piece in _SPACE_RUN.split(value.strip()) if piece]
    if not pieces:
        return ()
    pattern = re.compile(r"\s+".join(re.escape(piece) for piece in pieces), re.IGNORECASE)
    return tuple(
        frozenset(_offset_line_numbers(text, match.start(), match.end()))
        for match in pattern.finditer(text)
    )


def _party_path(path: str) -> tuple[str, int | None] | None:
    match = re.match(
        r"^documentPatch\.parties\.(notifyParties\[([0-9]+)\]|([A-Za-z]+))\.",
        path,
    )
    if match is None:
        return None
    return ("notifyParties", int(match.group(2))) if match.group(2) else (match.group(3), None)


def _source_party(
    source_label: Mapping[str, JsonValue], path: str
) -> tuple[str, Mapping[str, JsonValue]] | None:
    parsed = _party_path(path)
    if parsed is None:
        return None
    role, index = parsed
    patch = source_label.get("documentPatch")
    if not isinstance(patch, Mapping):
        return None
    parties = patch.get("parties")
    if not isinstance(parties, Mapping):
        return None
    raw_party = parties.get(role)
    if index is not None:
        if not isinstance(raw_party, Sequence) or isinstance(raw_party, (str, bytes)):
            return None
        if index >= len(raw_party):
            return None
        raw_party = raw_party[index]
    if not isinstance(raw_party, Mapping):
        return None
    semantic_role = "notify" if role == "notifyParties" else role
    return semantic_role, cast(Mapping[str, JsonValue], raw_party)


def _party_heading_roles(line: str) -> frozenset[str]:
    """Recognize structural party captions without classifying legal prose as a heading.

    Bill-of-Lading boilerplate mentions ``carrier``, ``shipper``, and ``consignee`` hundreds of
    times.  A substring classifier therefore turns signatures, liability clauses, and cargo
    captions into party-block boundaries.  Party captions instead have a small, stable grammar:
    the role occurs at the beginning of the line (optionally after a form-field number), or the
    line is an explicit ``name/address of <role>`` caption.  The returned role names deliberately
    match the task schema.
    """

    normalized = " ".join(re.findall(r"[a-z0-9]+", line.casefold()))
    normalized = re.sub(r"^(?:[0-9]+\s+)+", "", normalized)
    roles: set[str] = set()
    if re.match(
        r"^(?:(?:name|name and address|address) of )?"
        r"(?:shipper(?: exporter)?|exporter)(?:\b|$)",
        normalized,
    ):
        roles.add("shipper")
    if re.match(
        r"^(?:(?:name|name and address|address) of )?consignee(?:\b|$)", normalized
    ):
        roles.add("consignee")
    if re.match(
        r"^(?:(?:name|name and address|address) of )?notify(?: party| parties)?(?:\b|$)",
        normalized,
    ):
        roles.add("notify")
    if re.match(r"^(?:freight forwarder|forwarding agent)(?:\b|$)", normalized):
        roles.add("forwardingAgent")
    if re.match(
        r"^(?:delivery agent|agent at destination|agent address|agent for delivery|"
        r"delivery of goods apply to|for delivery apply to|"
        r"as (?:frt fwdrs )?delivery agent(?: only)?)(?:\b|$)",
        normalized,
    ):
        roles.add("deliveryAgent")

    stripped = re.sub(r"^[ \t]*(?:\([0-9]+\)|\[[0-9]+\]|[0-9]+[.)])[ \t]*", "", line)
    if (
        re.fullmatch(
            r"[ \t]*(?:(?:NAME|NAME[ \t]+AND[ \t]+ADDRESS|ADDRESS)[ \t]+OF[ \t]+)?"
            r"(?:OCEAN[ \t]+)?CARRIER(?:[ \t]+(?:NAME|ADDRESS))?[ \t:.-]*",
            stripped,
            re.IGNORECASE,
        )
        is not None
        or re.match(r"^[ \t]*(?:OCEAN[ \t]+)?CARRIER[ \t]*:", stripped, re.IGNORECASE)
        is not None
    ):
        roles.add("carrier")
    return frozenset(roles)


def _location_role_for_path(path: str) -> str | None:
    if path.startswith("documentPatch.placeOfIssue."):
        return "placeOfIssue"
    match = re.match(
        r"^documentPatch\.route\.(placeOfReceipt|portOfLoading|portOfDischarge|"
        r"placeOfDelivery|finalDestination|transshipmentPorts\[[0-9]+\])\.",
        path,
    )
    if match is not None:
        role = match.group(1)
        return "transshipmentPort" if role.startswith("transshipmentPorts") else role
    if path.startswith("documentPatch.freight.paymentPlace."):
        return "freightPaymentPlace"
    return None


def _location_heading_roles(line: str) -> frozenset[str]:
    normalized = " ".join(re.findall(r"[a-z0-9]+", line.casefold()))
    roles: set[str] = set()
    if (
        "place of receipt" in normalized
        or "precarried by" in normalized
        or "pre carriage by" in normalized
    ):
        roles.add("placeOfReceipt")
    if "port of loading" in normalized or "port loading" in normalized:
        roles.add("portOfLoading")
    if "port of discharge" in normalized or "port discharge" in normalized:
        roles.add("portOfDischarge")
    if "place of delivery" in normalized or "final destination" in normalized:
        roles.add("placeOfDelivery")
        roles.add("finalDestination")
    if (
        ("place" in normalized and "issue" in normalized)
        or "place and date of issuance" in normalized
        or "issued at" in normalized
    ):
        roles.add("placeOfIssue")
    if "transshipment" in normalized or "transhipment" in normalized:
        roles.add("transshipmentPort")
    if (
        ("freight" in normalized and ("payable" in normalized or "payment" in normalized))
        or "prepaid at" in normalized
        or "collect at" in normalized
    ):
        roles.add("freightPaymentPlace")
    return frozenset(roles)


def _contextual_location_line_numbers(text: str, path: str, value: str) -> set[int]:
    """Bind a repeated locality surface to its printed B/L semantic heading."""

    requested_role = _location_role_for_path(path)
    if requested_role is None:
        return set()
    bodies = text.splitlines()
    occurrences = _literal_occurrence_line_sets(text, value)
    if not occurrences:
        return set()
    role_matches: list[frozenset[int]] = []
    for occurrence in occurrences:
        first, last = min(occurrence), max(occurrence)
        paragraph_start = first
        while paragraph_start > 1:
            previous = bodies[paragraph_start - 2]
            if not previous.strip() or _PAGE_MARKER.fullmatch(previous) is not None:
                break
            paragraph_start -= 1
        context_roles: set[str] = set()
        for line in bodies[paragraph_start - 1 : last]:
            context_roles.update(_location_heading_roles(line))
        # Some carrier forms deliberately separate a heading from its populated value with one
        # or more blank layout rows. The closest preceding occupied line remains a valid heading
        # only when it is itself an unambiguous structural location caption. This does not cross a
        # page marker and does not use arbitrary proximity to infer a role.
        if requested_role not in context_roles and paragraph_start == first:
            previous_number = first - 1
            while previous_number >= 1 and not bodies[previous_number - 1].strip():
                previous_number -= 1
            if (
                previous_number >= 1
                and _PAGE_MARKER.fullmatch(bodies[previous_number - 1]) is None
            ):
                context_roles.update(_location_heading_roles(bodies[previous_number - 1]))
        if requested_role in context_roles:
            role_matches.append(occurrence)
    if role_matches:
        return set().union(*role_matches)
    return set(occurrences[0]) if len(occurrences) == 1 else set()


def _party_block_line_groups(
    text: str,
    *,
    path: str,
    source_label: Mapping[str, JsonValue],
) -> tuple[frozenset[int], ...]:
    """Resolve one party's printed block from its exact identity and semantic role.

    Labels commonly normalize an address that OCR splits across several lines or interleaves with
    the city.  Authorizing only the normalized literal strands necessary lines.  This locator
    instead anchors on the exact party name, bounds each candidate at blank/page/next-party
    boundaries, and selects only candidates whose printed role matches the schema path.  Repeated
    copies of the same role (for example duplicate pages) remain jointly owned.
    """

    resolved = _source_party(source_label, path)
    if resolved is None:
        return ()
    requested_role, party = resolved
    anchor_values = tuple(
        raw_value
        for raw_value in party.values()
        if isinstance(raw_value, str) and raw_value.strip()
    )
    if not anchor_values:
        return ()
    bodies = text.splitlines()
    candidates: dict[frozenset[int], tuple[int, int]] = {}
    occurrences = tuple(
        occurrence
        for value in anchor_values
        for occurrence in _literal_occurrence_line_sets(text, value)
    )
    for occurrence in occurrences:
        if not occurrence:
            continue
        first, last = min(occurrence), max(occurrence)
        paragraph_start = first
        while paragraph_start > 1:
            previous = bodies[paragraph_start - 2]
            if not previous.strip() or _PAGE_MARKER.fullmatch(previous) is not None:
                break
            paragraph_start -= 1
        # OCR commonly inserts one or more empty lines between a party heading and the
        # first identity line.  Treat the nearest non-empty line immediately before that
        # blank run as the heading, but never jump across intervening lexical content.  The
        # previous implementation stopped at the blank line, making headed CONSIGNEE and
        # NOTIFY blocks indistinguishable when both printed the same source party.
        preceding_heading_line: int | None = None
        probe = paragraph_start - 1
        while probe >= 1 and not bodies[probe - 1].strip():
            probe -= 1
        if probe >= 1 and _party_heading_roles(bodies[probe - 1]):
            preceding_heading_line = probe
        paragraph_end = last
        while paragraph_end < len(bodies):
            following = bodies[paragraph_end]
            if not following.strip() or _PAGE_MARKER.fullmatch(following) is not None:
                break
            following_roles = _party_heading_roles(following)
            if following_roles and requested_role not in following_roles:
                break
            paragraph_end += 1
        heading_roles: set[str] = set()
        role_heading_line: int | None = None
        if preceding_heading_line is not None:
            preceding_roles = _party_heading_roles(bodies[preceding_heading_line - 1])
            heading_roles.update(preceding_roles)
            if requested_role in preceding_roles:
                role_heading_line = preceding_heading_line
        # Some carriers print a formal NOTIFY PARTY caption and then qualify the named party as
        # ``(AS FRT FWDRS DELIVERY AGENT ONLY)`` on the following line.  The reviewed label is
        # correctly role-bound to deliveryAgent in that topology.  Scan the complete bounded
        # party paragraph for explicit caption grammar, not just lines preceding the name anchor;
        # `_party_heading_roles` deliberately rejects free-form legal prose.
        for line_number in range(paragraph_start, paragraph_end + 1):
            roles = _party_heading_roles(bodies[line_number - 1])
            heading_roles.update(roles)
            if requested_role in roles:
                role_heading_line = line_number
        if heading_roles and requested_role not in heading_roles:
            continue
        data_start = (
            role_heading_line + 1
            if role_heading_line is not None and role_heading_line < first
            else paragraph_start
        )
        while data_start < first and not bodies[data_start - 1].strip():
            data_start += 1
        block = frozenset(range(data_start, paragraph_end + 1))
        joined = _party_match_surface(" ".join(bodies[data_start - 1 : paragraph_end]))
        matched_fields = sum(
            1
            for key, raw_value in party.items()
            if key != "name"
            and isinstance(raw_value, str)
            and raw_value.strip()
            and _party_match_surface(raw_value) in joined
        )
        role_score = 1 if requested_role in heading_roles else 0
        # An explicit semantic heading outranks a more textually complete roleless copy (for
        # example a delivery-contact or signing block).  Field coverage only breaks ties
        # between blocks with equal role evidence.
        score = (role_score, matched_fields)
        if block not in candidates or score > candidates[block]:
            candidates[block] = score
    if not candidates:
        return ()
    best = max(candidates.values())
    return tuple(
        sorted(
            (lines for lines, score in candidates.items() if score == best),
            key=lambda values: (min(values), max(values)),
        )
    )


def _party_block_line_numbers(
    text: str,
    *,
    path: str,
    source_label: Mapping[str, JsonValue],
) -> set[int]:
    """Return the union of the independently preserved occurrences of one party role."""

    groups = _party_block_line_groups(text, path=path, source_label=source_label)
    return set().union(*groups) if groups else set()


def _case_preserving_surface(source: str, target: str) -> str:
    letters = [character for character in source if character.isalpha()]
    if letters and all(character.isupper() for character in letters):
        return target.upper()
    if letters and all(character.islower() for character in letters):
        return target.lower()
    return target


def _surface_tokens(value: str) -> tuple[str, ...]:
    return tuple(re.findall(r"[^\W_]+", value.upper(), flags=re.UNICODE))


def _country_is_embedded_in_other_party_scalar(
    *,
    line: str,
    source_country: str,
    source_party: Mapping[str, Any],
) -> bool:
    """Return whether a country occurrence belongs to another labeled party scalar.

    Party addresses can legitimately contain a country word as part of a street or company
    name (for example ``ORT ISRAEL STR.``). Replacing that token as country metadata corrupts
    the address and creates an impossible downstream lock. We use exact Unicode word tokens and
    require neighboring source-scalar context; no fuzzy alias or geographic inference is made.
    A conservative match leaves the complete scalar to the role-bound renderer.
    """

    country_tokens = _surface_tokens(source_country)
    line_tokens = _surface_tokens(line)
    if not country_tokens or not line_tokens:
        return False
    line_positions = tuple(
        index
        for index in range(len(line_tokens) - len(country_tokens) + 1)
        if line_tokens[index : index + len(country_tokens)] == country_tokens
    )
    if not line_positions:
        return False
    for field, scalar in source_party.items():
        if field == "country" or not isinstance(scalar, str) or not scalar.strip():
            continue
        scalar_tokens = _surface_tokens(scalar)
        if len(scalar_tokens) <= len(country_tokens):
            continue
        for scalar_index in range(len(scalar_tokens) - len(country_tokens) + 1):
            if scalar_tokens[scalar_index : scalar_index + len(country_tokens)] != country_tokens:
                continue
            expected_before = scalar_tokens[scalar_index - 1] if scalar_index > 0 else None
            after_index = scalar_index + len(country_tokens)
            expected_after = (
                scalar_tokens[after_index] if after_index < len(scalar_tokens) else None
            )
            for line_index in line_positions:
                observed_before = line_tokens[line_index - 1] if line_index > 0 else None
                line_after = line_index + len(country_tokens)
                observed_after = line_tokens[line_after] if line_after < len(line_tokens) else None
                before_matches = expected_before is not None and observed_before == expected_before
                after_matches = expected_after is not None and observed_after == expected_after
                if before_matches or after_matches:
                    return True
    return False


def _apply_party_role_country_prefills(
    workspace: RewriteWorkspace,
) -> tuple[AppliedDeterministicPrefill, ...]:
    """Render exact country occurrences inside their proven source party blocks.

    Some B/Ls repeat a party country in a detached customs-registration field.  The generic
    scalar locator can then bind the schema path to that more explicit heading while leaving the
    visibly printed country inside the actual shipper/consignee block as an apparent auxiliary
    copy.  A country that occurs exactly inside a role-proven block is not creative language: it
    is the same task scalar and can be replaced deterministically on that line.  Every exact
    non-name occurrence in each independent role block is updated because documents commonly
    print the country on both an address line and a locality line. Unresolved aliases remain
    model-owned and fail closed.
    """

    current_lines = workspace.current_text.splitlines(keepends=True)
    applied: list[AppliedDeterministicPrefill] = []
    for leaf in changed_leaves(workspace.source_label, workspace.current_target_label):
        if (
            not leaf.requiresTextEdit
            or not leaf.path.startswith("documentPatch.parties.")
            or not leaf.path.endswith(".country")
            or not isinstance(leaf.sourceValue, str)
            or not isinstance(leaf.targetValue, str)
            or leaf.sourceValue == leaf.targetValue
        ):
            continue
        role_groups = _party_block_line_groups(
            workspace.original_text,
            path=leaf.path,
            source_label=cast(Mapping[str, JsonValue], workspace.source_label),
        )
        if not role_groups:
            continue
        resolved = _source_party(cast(Mapping[str, JsonValue], workspace.source_label), leaf.path)
        source_name = resolved[1].get("name") if resolved is not None else None
        source_party = resolved[1] if resolved is not None else {}
        name_lines = (
            set().union(*_literal_occurrence_line_sets(workspace.original_text, source_name))
            if isinstance(source_name, str)
            and source_name.strip()
            and _literal_occurrence_line_sets(workspace.original_text, source_name)
            else set()
        )
        source_surfaces = [leaf.sourceValue]
        # Some forms print the same country twice inside one proven party block, once as the
        # label-normalized value (``THE NETHERLANDS``) and once without the optional leading
        # definite article (``NETHERLANDS``).  This is a grammatical surface variant, not a
        # country alias or geographic inference.  Own it only inside the same role-proven block
        # and project it to the exact target country like the complete source surface.
        article_match = re.fullmatch(r"\s*THE\s+(.+?)\s*", leaf.sourceValue, re.IGNORECASE)
        if article_match is not None and article_match.group(1) not in source_surfaces:
            source_surfaces.append(article_match.group(1))
        occurrences_by_surface = {
            source_surface: _literal_occurrence_line_sets(
                workspace.original_text, source_surface
            )
            for source_surface in source_surfaces
        }
        for role_group in role_groups:
            matches = [
                (source_surface, occurrence)
                for source_surface, occurrences in occurrences_by_surface.items()
                for occurrence in occurrences
                if len(occurrence) == 1
                and occurrence <= role_group
                and not occurrence <= name_lines
            ]
            if not matches:
                continue
            for source_surface, selected in sorted(
                matches,
                key=lambda value: (max(value[1]), min(value[1]), -len(value[0])),
            ):
                line_number = next(iter(selected))
                before = _line_body(current_lines[line_number - 1])
                original_line = workspace.original_text.splitlines()[line_number - 1]
                if _country_is_embedded_in_other_party_scalar(
                    line=original_line,
                    source_country=source_surface,
                    source_party=source_party,
                ):
                    continue
                if before.count(source_surface) != 1:
                    continue
                target_surface = _case_preserving_surface(source_surface, leaf.targetValue)
                after = before.replace(source_surface, target_surface, 1)
                current_lines[line_number - 1] = after + _line_ending(
                    current_lines[line_number - 1]
                )
                applied.append(
                    AppliedDeterministicPrefill(
                        lineId=f"L{line_number:05d}",
                        targetPaths=(leaf.path,),
                        sourceSurface=source_surface,
                        targetSurface=target_surface,
                        beforeLine=before,
                        afterLine=after,
                    )
                )
    if applied:
        workspace.current_text = "".join(current_lines)
        workspace.deterministic_prefills.extend(applied)
    return tuple(applied)


def _cargo_requirement_line_numbers(
    text: str,
    requirement: CargoFlavorRewriteRequirement,
    *,
    expand_repeated_surfaces: bool = True,
) -> set[int]:
    """Own every exact repeated rendering of an audited cargo-description source block."""

    lines = {_line_number(value) for value in requirement.sourceLineIds}
    if expand_repeated_surfaces:
        for surface in requirement.sourceSurfaces:
            located, _ = _literal_line_numbers(text, surface)
            lines.update(located)
    return lines


def _numeric_line_numbers(text: str, value: int | float) -> set[int]:
    expected = Decimal(str(value))
    matches: set[int] = set()
    for line_number, line in enumerate(text.splitlines(), start=1):
        if _PAGE_MARKER.fullmatch(line) is not None:
            continue
        for match in _NUMERIC_TOKEN.finditer(line):
            if _numeric_span_overlaps_date(line, match.start(), match.end()):
                continue
            if match.start() > 0 and line[match.start() - 1].isalnum():
                continue
            if (
                match.end() < len(line)
                and line[match.end()].isalpha()
                and _ATTACHED_NUMERIC_UNIT.match(line[match.end() :]) is None
            ):
                continue
            if expected in _numeric_surface_values(match.group(0)):
                matches.add(line_number)
    return matches


def _numeric_surface_values(surface: str) -> frozenset[Decimal]:
    """Return every locale-plausible value for one OCR numeric surface.

    A single separator followed by three digits is inherently ambiguous without a unit-specific
    capacity bound: ``3.741`` may mean 3,741 while ``20931.200`` may mean 20,931.200.  Residual
    evidence lookup has the exact source-label value available, so retaining both interpretations
    is safer and more precise than choosing the larger value globally.
    """

    value = surface.strip().replace("\N{MINUS SIGN}", "-")
    if re.fullmatch(r"[-+]?[0-9]+(?:[.,][0-9]+)*", value) is None:
        return frozenset()
    sign = Decimal(-1) if value.startswith("-") else Decimal(1)
    unsigned = value.lstrip("+-")
    candidates: set[Decimal] = set()
    parsed = _parse_measurement_surface(unsigned, maximum=None)
    if parsed is not None:
        candidates.add(sign * parsed.value)
    separators = {character for character in unsigned if character in ".,"}
    if not separators:
        candidates.add(sign * Decimal(unsigned))
    elif len(separators) == 1:
        separator = next(iter(separators))
        pieces = unsigned.split(separator)
        if len(pieces) == 2:
            integer, fraction = pieces
            candidates.add(sign * Decimal(f"{integer}.{fraction}"))
        if all(len(piece) == 3 for piece in pieces[1:]):
            candidates.add(sign * Decimal("".join(pieces)))
    else:
        decimal_separator = "." if unsigned.rfind(".") > unsigned.rfind(",") else ","
        grouping_separator = "," if decimal_separator == "." else "."
        integer, fraction = unsigned.rsplit(decimal_separator, 1)
        candidates.add(sign * Decimal(integer.replace(grouping_separator, "") + "." + fraction))
    return frozenset(row for row in candidates if row.is_finite())


def _object_scope(path: str) -> str:
    allocation = re.match(
        r"^(documentPatch\.cargoAllocationGroups\[[0-9]+\]\.allocations\[[0-9]+\])",
        path,
    )
    if allocation is not None:
        return allocation.group(1)
    match = re.match(
        r"^(documentPatch\.(?:cargoGroups|cargoPackages|cargoAllocationGroups|containers)\[[0-9]+\])",
        path,
    )
    if match is not None:
        return match.group(1)
    party = re.match(
        r"^(documentPatch\.parties\.(?:notifyParties\[[0-9]+\]|[A-Za-z]+))",
        path,
    )
    if party is not None:
        return party.group(1)
    # A root scalar or root-list element is its own semantic object.  Returning the bare
    # ``documentPatch`` parent would let an unlocated B/L reference, date, or document number
    # borrow write authority from every unrelated top-level field in the document.
    if path.startswith("documentPatch.") and "." not in path[len("documentPatch.") :]:
        return path
    return path.rsplit(".", 1)[0]


def _forwarding_reference_line_numbers(text: str, source_value: str) -> set[int]:
    """Locate a forwarding/export reference only under explicit reference grammar.

    These label values commonly combine an identifier and date while the document inserts
    labels such as ``INVOICE NO.`` and ``DATED:`` between them, sometimes across two lines.
    Searching either atom globally is unsafe because the same identifier can also be the B/L or
    booking number.  This locator accepts only one- or two-line windows containing every ordered
    semantic atom and an explicit forwarding/export reference heading.
    """

    atoms = tuple(
        atom
        for piece in source_value.split()
        if (atom := _semantic_surface(piece)) and len(atom) >= 3
    )
    if not atoms:
        return set()
    lines = text.splitlines()
    candidates: list[frozenset[int]] = []
    for start in range(len(lines)):
        for width in (1, 2):
            stop = start + width
            if stop > len(lines):
                continue
            window = "\n".join(lines[start:stop])
            context_start = max(0, start - 4)
            context = "\n".join(lines[context_start:stop])
            if _FORWARDING_REFERENCE_CONTEXT.search(context) is None:
                continue
            normalized = _semantic_surface(window)
            cursor = 0
            for atom in atoms:
                position = normalized.find(atom, cursor)
                if position < 0:
                    break
                cursor = position + len(atom)
            else:
                candidates.append(
                    frozenset(
                        start + offset + 1
                        for offset, line in enumerate(lines[start:stop])
                        if any(atom in _semantic_surface(line) for atom in atoms)
                    )
                )
    # A one-line reference can also appear in an overlapping two-line window alongside an
    # unrelated B/L or booking number.  Retain only locally minimal evidence windows while still
    # preserving genuinely repeated reference renderings elsewhere in the document.
    minimal = {
        candidate
        for candidate in candidates
        if candidate
        and not any(other < candidate for other in candidates if other)
    }
    return set().union(*minimal) if minimal else set()


def _paragraph_line_numbers(text: str, seeds: set[int]) -> set[int]:
    bodies = text.splitlines()
    output: set[int] = set()
    for seed in seeds:
        if not 1 <= seed <= len(bodies) or _PAGE_MARKER.fullmatch(bodies[seed - 1]) is not None:
            continue
        start = seed
        while start > 1:
            previous = bodies[start - 2]
            if not previous.strip() or _PAGE_MARKER.fullmatch(previous) is not None:
                break
            start -= 1
        end = seed
        while end < len(bodies):
            following = bodies[end]
            if not following.strip() or _PAGE_MARKER.fullmatch(following) is not None:
                break
            end += 1
        output.update(range(start, end + 1))
    return output


def _label_patch(label: Mapping[str, JsonValue]) -> Mapping[str, Any]:
    patch = label.get("documentPatch")
    return cast(Mapping[str, Any], patch) if isinstance(patch, Mapping) else {}


def _relation_scopes(
    text: str,
    *,
    source_label: Mapping[str, JsonValue],
    target_label: Mapping[str, JsonValue],
    cargo_lines_by_group: Mapping[str, set[int]],
) -> dict[str, set[int]]:
    """Build indexed cargo/equipment scopes from explicit schema relations."""

    source_patch = _label_patch(source_label)
    target_patch = _label_patch(target_label)
    output: dict[str, set[int]] = {}
    group_paragraphs = {
        group_id: _paragraph_line_numbers(text, lines)
        for group_id, lines in cargo_lines_by_group.items()
    }

    def objects(patch: Mapping[str, Any], key: str) -> Sequence[Any]:
        value = patch.get(key)
        return value if isinstance(value, Sequence) and not isinstance(value, (str, bytes)) else ()

    # Container allocations are stronger row ownership than repeated cargo text. Dense B/L
    # tables commonly repeat an identical goods/material/HS tuple under several containers;
    # locating by that tuple alone makes every indexed group appear to own every copy. When a
    # source allocation names a printed container, its blank-delimited paragraph is the exact
    # physical scope for that group and replaces the weaker description-derived scope.
    allocation_scopes: dict[str, set[int]] = defaultdict(set)
    target_allocation_groups = {
        cast(str, row["groupId"]): row
        for row in objects(target_patch, "cargoAllocationGroups")
        if isinstance(row, Mapping) and isinstance(row.get("groupId"), str)
    }
    for raw_group in objects(source_patch, "cargoAllocationGroups"):
        if not isinstance(raw_group, Mapping) or not isinstance(raw_group.get("groupId"), str):
            continue
        group_id = cast(str, raw_group["groupId"])
        allocation_sets = [raw_group.get("allocations")]
        target_group = target_allocation_groups.get(group_id)
        if target_group is not None:
            allocation_sets.append(target_group.get("allocations"))
        container_lines: set[int] = set()
        for allocations in allocation_sets:
            if not isinstance(allocations, Sequence) or isinstance(allocations, (str, bytes)):
                continue
            for allocation in allocations:
                number = (
                    allocation.get("containerNumber")
                    if isinstance(allocation, Mapping)
                    else None
                )
                if not isinstance(number, str) or not number:
                    continue
                located = _container_identifier_line_numbers(text, number)
                container_lines.update(located)
        if container_lines:
            allocation_scopes[group_id].update(_paragraph_line_numbers(text, container_lines))
    group_paragraphs.update(allocation_scopes)

    for family in ("cargoGroups", "cargoPackages", "cargoAllocationGroups"):
        for index, raw_object in enumerate(objects(source_patch, family)):
            if not isinstance(raw_object, Mapping):
                continue
            group_id = raw_object.get("groupId")
            if isinstance(group_id, str) and group_paragraphs.get(group_id):
                output[f"documentPatch.{family}[{index}]"] = set(group_paragraphs[group_id])

    source_containers = objects(source_patch, "containers")
    target_containers = objects(target_patch, "containers")
    container_scopes_by_number: dict[str, set[int]] = {}
    for index, raw_object in enumerate(source_containers):
        if not isinstance(raw_object, Mapping):
            continue
        numbers = [raw_object.get("containerNumber")]
        if index < len(target_containers) and isinstance(target_containers[index], Mapping):
            numbers.append(target_containers[index].get("containerNumber"))
        seeds: set[int] = set()
        for number in numbers:
            if isinstance(number, str) and number:
                located = _container_identifier_line_numbers(text, number)
                seeds.update(located)
        scope = _paragraph_line_numbers(text, seeds)
        if scope:
            output[f"documentPatch.containers[{index}]"] = scope
            for number in numbers:
                if isinstance(number, str) and number:
                    container_scopes_by_number[number] = scope

    target_alloc_groups = objects(target_patch, "cargoAllocationGroups")
    for group_index, raw_group in enumerate(objects(source_patch, "cargoAllocationGroups")):
        if not isinstance(raw_group, Mapping):
            continue
        target_group = (
            target_alloc_groups[group_index]
            if group_index < len(target_alloc_groups)
            and isinstance(target_alloc_groups[group_index], Mapping)
            else {}
        )
        source_allocations = raw_group.get("allocations")
        target_allocations = target_group.get("allocations")
        if not isinstance(source_allocations, Sequence) or isinstance(
            source_allocations, (str, bytes)
        ):
            continue
        target_allocations = (
            target_allocations
            if isinstance(target_allocations, Sequence)
            and not isinstance(target_allocations, (str, bytes))
            else ()
        )
        group_scope = output.get(f"documentPatch.cargoAllocationGroups[{group_index}]", set())
        for allocation_index, raw_allocation in enumerate(source_allocations):
            if not isinstance(raw_allocation, Mapping):
                continue
            numbers = [raw_allocation.get("containerNumber")]
            if allocation_index < len(target_allocations) and isinstance(
                target_allocations[allocation_index], Mapping
            ):
                numbers.append(target_allocations[allocation_index].get("containerNumber"))
            scope = set(group_scope)
            for number in numbers:
                if isinstance(number, str):
                    scope.update(container_scopes_by_number.get(number, set()))
            if scope:
                output[
                    f"documentPatch.cargoAllocationGroups[{group_index}].allocations["
                    f"{allocation_index}]"
                ] = scope
    return output


def _apply_relation_scoped_additional_information_prefills(
    workspace: RewriteWorkspace,
    relation_scopes: Mapping[str, set[int]],
) -> tuple[AppliedDeterministicPrefill, ...]:
    """Render exact cargo auxiliary values within their allocation-proven group scope.

    Additional-information rows such as material, batch, and grade are already generated by the
    linguistic stage. Their insertion has no creative degree of freedom, but dense carrier tables
    often repeat the same source value under several containers or print one heading above a
    column of values. Global literal ownership is therefore invalid. This renderer accepts only
    an exact inline source surface or an exact whole-line value under an exact heading inside the
    indexed cargo group's relation scope. Ambiguous or absent evidence remains agent-owned.
    """

    source_groups = _label_patch(workspace.source_label).get("cargoGroups")
    target_groups = _label_patch(workspace.current_target_label).get("cargoGroups")
    if not isinstance(source_groups, Sequence) or isinstance(source_groups, (str, bytes)):
        return ()
    if not isinstance(target_groups, Sequence) or isinstance(target_groups, (str, bytes)):
        return ()
    lines = workspace.current_text.splitlines(keepends=True)
    applied: list[AppliedDeterministicPrefill] = []
    for group_index, (source_group, target_group) in enumerate(
        zip(source_groups, target_groups, strict=False)
    ):
        if not isinstance(source_group, Mapping) or not isinstance(target_group, Mapping):
            continue
        scope = relation_scopes.get(f"documentPatch.cargoGroups[{group_index}]", set())
        if not scope:
            continue
        source_values = source_group.get("additionalInformation")
        target_values = target_group.get("additionalInformation")
        if not isinstance(source_values, Sequence) or isinstance(source_values, (str, bytes)):
            continue
        if not isinstance(target_values, Sequence) or isinstance(target_values, (str, bytes)):
            continue
        projections: dict[str, set[str]] = defaultdict(set)
        for source_value, target_value in zip(source_values, target_values, strict=False):
            if isinstance(source_value, str) and isinstance(target_value, str):
                projections[source_value].add(target_value)
        if any(len(targets) > 1 for targets in projections.values()):
            # Two schema rows with the same source surface but different targets have no
            # occurrence-level identity in the label. The agent must resolve their topology.
            continue
        for value_index, (source_value, target_value) in enumerate(
            zip(source_values, target_values, strict=False)
        ):
            if (
                not isinstance(source_value, str)
                or not isinstance(target_value, str)
                or source_value == target_value
            ):
                continue
            path = (
                f"documentPatch.cargoGroups[{group_index}].additionalInformation[{value_index}]"
            )
            scoped_numbers = tuple(
                number for number in sorted(scope) if 1 <= number <= len(lines)
            )
            exact_occurrences: list[tuple[int, int]] = []
            for number in scoped_numbers:
                body = _line_body(lines[number - 1])
                start = 0
                while (offset := body.find(source_value, start)) >= 0:
                    exact_occurrences.append((number, offset))
                    start = offset + len(source_value)
            if exact_occurrences:
                by_line: dict[int, list[int]] = defaultdict(list)
                for number, offset in exact_occurrences:
                    by_line[number].append(offset)
                for number, offsets in sorted(by_line.items()):
                    before = _line_body(lines[number - 1])
                    after = before
                    for offset in reversed(offsets):
                        after = (
                            after[:offset] + target_value + after[offset + len(source_value) :]
                        )
                    lines[number - 1] = after + _line_ending(lines[number - 1])
                    applied.append(
                        AppliedDeterministicPrefill(
                            lineId=f"L{number:05d}",
                            targetPaths=(path,),
                            sourceSurface=source_value,
                            targetSurface=target_value,
                            beforeLine=before,
                            afterLine=after,
                        )
                    )
                continue

            # Columnar forms print a heading once and each field value on its own line. Find the
            # longest common leading token prefix whose exact heading and exact value line both
            # occur inside this cargo scope. This deliberately does not search arbitrary
            # substrings or infer a field alias.
            source_parts = source_value.split()
            target_parts = target_value.split()
            rendered_column = False
            for prefix_length in range(min(len(source_parts), len(target_parts)) - 1, 0, -1):
                source_prefix = " ".join(source_parts[:prefix_length])
                target_prefix = " ".join(target_parts[:prefix_length])
                if _semantic_surface(source_prefix) != _semantic_surface(target_prefix):
                    continue
                source_suffix = " ".join(source_parts[prefix_length:])
                target_suffix = " ".join(target_parts[prefix_length:])
                heading_lines = tuple(
                    number
                    for number in scoped_numbers
                    if _semantic_surface(_line_body(lines[number - 1]).strip())
                    == _semantic_surface(source_prefix)
                )
                value_lines = tuple(
                    number
                    for number in scoped_numbers
                    if _line_body(lines[number - 1]).strip() == source_suffix
                    and any(heading < number for heading in heading_lines)
                )
                if not heading_lines or not value_lines:
                    continue
                for number in value_lines:
                    before = _line_body(lines[number - 1])
                    leading = before[: len(before) - len(before.lstrip())]
                    trailing = before[len(before.rstrip()) :]
                    after = leading + target_suffix + trailing
                    lines[number - 1] = after + _line_ending(lines[number - 1])
                    applied.append(
                        AppliedDeterministicPrefill(
                            lineId=f"L{number:05d}",
                            targetPaths=(path,),
                            sourceSurface=source_suffix,
                            targetSurface=target_suffix,
                            beforeLine=before,
                            afterLine=after,
                        )
                    )
                rendered_column = True
                break
            if rendered_column:
                continue
    if applied:
        workspace.current_text = "".join(lines)
        workspace.deterministic_prefills.extend(applied)
    return tuple(applied)


def _requirement_paths_already_prefilled(
    workspace: RewriteWorkspace,
) -> frozenset[str]:
    return frozenset(
        path for prefill in workspace.deterministic_prefills for path in prefill.targetPaths
    )


def _line_set_for_directive(text: str, path: str, source_value: JsonValue) -> tuple[set[int], str]:
    if (
        _FORWARDING_REFERENCE_PATH.fullmatch(path) is not None
        and isinstance(source_value, str)
        and source_value
    ):
        lines = _forwarding_reference_line_numbers(text, source_value)
        return lines, "forwarding_reference_context" if lines else "unlocated"
    if path == "documentPatch.freight.paymentArrangement" and source_value in {
        "prepaid",
        "collect",
    }:
        arrangement = source_value
        other = "collect" if arrangement == "prepaid" else "prepaid"
        explicit = re.compile(
            rf"\b(?:ocean[ \t]+)?freight\b[^\r\n]{{0,24}}\b{arrangement}\b|"
            rf"\b{arrangement}\b[^\r\n]{{0,24}}\bfreight\b",
            re.IGNORECASE,
        )
        lines = {
            line_number
            for line_number, line in enumerate(text.splitlines(), start=1)
            if explicit.search(line) is not None
            and re.search(rf"\b{other}\b", line, re.IGNORECASE) is None
        }
        if not lines:
            location = "destination" if arrangement == "collect" else "origin"
            payable = re.compile(
                rf"\bfreight[ \t]+payable[ \t]+at[ \t]+{location}\b",
                re.IGNORECASE,
            )
            lines = {
                line_number
                for line_number, line in enumerate(text.splitlines(), start=1)
                if payable.search(line) is not None
            }
        if lines:
            return lines, "freight_arrangement_surface"
    if isinstance(source_value, str) and source_value:
        lines, locator = _literal_line_numbers(text, source_value)
        if (
            not lines
            and path == "documentPatch.parties.carrier.name"
            and (wrapped := _wrapped_carrier_signature_lines(text, source_value))
        ):
            return wrapped, "wrapped_carrier_signature_literal"
        return lines, locator
    if isinstance(source_value, (int, float)) and not isinstance(source_value, bool):
        lines = _numeric_line_numbers(text, source_value)
        return lines, "numeric_surface" if lines else "unlocated"
    return set(), "unlocated"


def _wrapped_carrier_signature_lines(text: str, source_value: str) -> set[int]:
    """Locate a carrier name split by a form's ``SIGNED ... / BY: ...`` line break.

    ``BY:`` is structural form text, not part of the carrier identity.  This remains an exact
    locator: after removing only those two recognized labels, the joined semantic surface must
    equal the reviewed source carrier.  No fuzzy or partial-name matching is permitted.
    """

    lines = text.splitlines()
    expected = _semantic_surface(source_value)
    for index, line in enumerate(lines[:-1]):
        signed = re.fullmatch(r"[ \t]*SIGNED[ \t]+(?P<value>.+?)[ \t]*", line, re.I)
        following = re.fullmatch(
            r"[ \t]*BY[ \t]*:[ \t]*(?P<value>.+?)[ \t]*", lines[index + 1], re.I
        )
        if signed is None or following is None:
            continue
        observed = _semantic_surface(
            f"{signed.group('value')} {following.group('value')}"
        )
        if observed == expected:
            return {index + 1, index + 2}
    return set()


def _temperature_deactivation_line_numbers(text: str, source_value: JsonValue) -> set[int]:
    """Bind a removed setpoint to its printed numeric line, never a sibling container line.

    A null target has no literal target surface.  The generic sibling-object fallback previously
    borrowed container-number and header lines, granting the model authority it did not need.
    Temperature values are instead accepted only on a numeric line with explicit thermal context
    on that line or within the two immediately adjacent occupied lines.
    """

    if not isinstance(source_value, Mapping):
        return set()
    value = source_value.get("value")
    if not isinstance(value, (int, float)) or isinstance(value, bool):
        return set()
    numeric_lines = _numeric_line_numbers(text, value)
    bodies = text.splitlines()
    output: set[int] = set()
    for number in numeric_lines:
        raw_line = bodies[number - 1]
        if value < 0 and re.search(r"[-\N{MINUS SIGN}]\s*[0-9]", raw_line) is None:
            continue
        same_line_context = _TEMPERATURE_CONTEXT.search(raw_line) is not None
        bare_temperature_value = (
            re.fullmatch(
                r"[ \t]*[-+\N{MINUS SIGN}]?[0-9]+(?:[.,][0-9]+)*"
                r"(?:[ \t]*(?:\N{DEGREE SIGN}[ \t]*)?[CF])?[ \t]*",
                raw_line,
                re.IGNORECASE,
            )
            is not None
        )
        start = max(0, number - 3)
        end = min(len(bodies), number + 2)
        context = " ".join(line for line in bodies[start:end] if line.strip())
        if same_line_context or (
            bare_temperature_value and _TEMPERATURE_CONTEXT.search(context) is not None
        ):
            output.add(number)
    return output


def _context_bound_surface_lines(
    text: str,
    requirement: SurfaceRenderingRequirement,
) -> set[int]:
    """Locate a rendered surface through its exact audited context, not all global copies."""

    if requirement.sourceLineIds:
        bodies = text.splitlines()
        line_numbers = {int(value[1:]) for value in requirement.sourceLineIds}
        if any(not 1 <= number <= len(bodies) for number in line_numbers):
            return set()
        source_pattern = re.compile(re.escape(requirement.sourceSurface))
        target_pattern = re.compile(re.escape(requirement.targetSurface))
        if all(
            source_pattern.search(bodies[number - 1]) is not None
            or target_pattern.search(bodies[number - 1]) is not None
            for number in line_numbers
        ):
            return line_numbers
        return set()

    if requirement.kind == "date":
        # Identical printed dates can belong to different semantic fields.  Use the same
        # nearest-heading parser that created the rendering requirement; a blank-delimited block
        # is insufficient because ``DATED`` does not literally contain the token ``date`` and a
        # stored context excerpt can contain both occurrences.
        contextual: set[int] = set()
        for match in re.finditer(re.escape(requirement.sourceSurface), text):
            if _date_context_path(text, match.start()) == requirement.targetPath:
                contextual.update(_offset_line_numbers(text, match.start(), match.end()))
        return contextual
    if requirement.kind == "date_global":
        return {
            number
            for match in re.finditer(re.escape(requirement.sourceSurface), text)
            for number in _offset_line_numbers(text, match.start(), match.end())
        }

    context = requirement.contextEvidence
    if not context:
        return set()
    context_offsets = [match.start() for match in re.finditer(re.escape(context), text)]
    if len(context_offsets) != 1:
        return set()
    local_offsets = [
        match.start() for match in re.finditer(re.escape(requirement.sourceSurface), context)
    ]
    if not local_offsets:
        return set()
    context_start = context_offsets[0]
    lines: set[int] = set()
    for local_start in local_offsets:
        start = context_start + local_start
        lines.update(_offset_line_numbers(text, start, start + len(requirement.sourceSurface)))
    return lines


def _expanded_intervals(
    text: str,
    core_lines: set[int],
    *,
    context: int,
    merge_gap: int,
) -> tuple[tuple[int, int], ...]:
    bodies = text.splitlines()
    protected = {
        index
        for index, line in enumerate(bodies, start=1)
        if _PAGE_MARKER.fullmatch(line) is not None
    }
    expanded: set[int] = set()
    for line_number in core_lines:
        for candidate in range(
            max(1, line_number - context),
            min(len(bodies), line_number + context) + 1,
        ):
            if candidate not in protected:
                expanded.add(candidate)
    if not expanded:
        return ()
    ordered = sorted(expanded)
    intervals: list[tuple[int, int]] = []
    start = previous = ordered[0]
    for line_number in ordered[1:]:
        crosses_marker = any(marker in range(previous + 1, line_number + 1) for marker in protected)
        if not crosses_marker and line_number - previous - 1 <= merge_gap:
            previous = line_number
            continue
        intervals.append((start, previous))
        start = previous = line_number
    intervals.append((start, previous))
    # Context expansion can straddle a protected page marker. The marker splits that window and
    # may leave a context-only fragment on the adjacent page; only intervals retaining at least
    # one located work-item line are model-authorized spans.
    return tuple(
        interval for interval in intervals if core_lines & set(range(interval[0], interval[1] + 1))
    )


def _trim_context_blank_edges(
    bodies: Sequence[str],
    interval: tuple[int, int],
    core_lines: set[int],
) -> tuple[int, int]:
    """Remove only blank edge lines introduced as context, never semantic/core lines."""

    start, end = interval
    while start < end and start not in core_lines and not bodies[start - 1].strip():
        start += 1
    while end > start and end not in core_lines and not bodies[end - 1].strip():
        end -= 1
    return start, end


def compile_case(
    bundle: RewriteCycleBundle,
    *,
    context_lines: int,
    merge_gap_lines: int,
) -> _CompiledCase:
    workspace = _workspace_from_bundle(bundle)
    compiler_edits = _apply_compiler_requirements(workspace)
    cargo_requirements = {
        requirement.targetPath: requirement
        for requirement in bundle.cargoFlavorRewriteRequirements
        if requirement.lineRole == "label_grounded"
    }
    cargo_surface_paths: dict[str, set[str]] = defaultdict(set)
    for requirement in bundle.cargoFlavorRewriteRequirements:
        if requirement.lineRole != "label_grounded":
            continue
        for surface in requirement.sourceSurfaces:
            cargo_surface_paths[_semantic_surface(surface)].add(requirement.targetPath)

    def cargo_lines(requirement: CargoFlavorRewriteRequirement) -> set[int]:
        expand = all(
            len(cargo_surface_paths[_semantic_surface(surface)]) == 1
            for surface in requirement.sourceSurfaces
        )
        return _cargo_requirement_line_numbers(
            workspace.current_text,
            requirement,
            expand_repeated_surfaces=expand,
        )

    source_patch = _label_patch(bundle.sourceLabel)
    source_cargo_groups = source_patch.get("cargoGroups")
    source_cargo_groups = (
        source_cargo_groups
        if isinstance(source_cargo_groups, Sequence)
        and not isinstance(source_cargo_groups, (str, bytes))
        else ()
    )
    cargo_lines_by_group: dict[str, set[int]] = {}
    for requirement in bundle.cargoFlavorRewriteRequirements:
        if requirement.lineRole != "label_grounded":
            continue
        match = re.match(r"^documentPatch\.cargoGroups\[([0-9]+)\]", requirement.targetPath)
        if match is None:
            continue
        index = int(match.group(1))
        if index >= len(source_cargo_groups) or not isinstance(source_cargo_groups[index], Mapping):
            continue
        group_id = source_cargo_groups[index].get("groupId")
        if isinstance(group_id, str):
            cargo_lines_by_group.setdefault(group_id, set()).update(cargo_lines(requirement))
    relation_scopes = _relation_scopes(
        workspace.current_text,
        source_label=bundle.sourceLabel,
        target_label=bundle.targetLabel,
        cargo_lines_by_group=cargo_lines_by_group,
    )
    _apply_relation_scoped_additional_information_prefills(workspace, relation_scopes)
    prefilled_paths = _requirement_paths_already_prefilled(workspace)

    surface_lines_by_path: dict[str, set[int]] = defaultdict(set)
    for surface_requirement in bundle.surfaceRenderingRequirements:
        current_lines = _context_bound_surface_lines(
            workspace.original_text,
            surface_requirement,
        )
        if not current_lines:
            current_lines, _ = _literal_line_numbers(
                workspace.current_text, surface_requirement.sourceSurface
            )
        if not current_lines:
            current_lines, _ = _literal_line_numbers(
                workspace.current_text, surface_requirement.targetSurface
            )
        for path in surface_requirement.targetPath.split(";"):
            surface_lines_by_path[path].update(current_lines)

    specs: list[dict[str, Any]] = []
    line_sets: dict[str, set[int]] = {}
    object_lines: dict[str, set[int]] = defaultdict(set)
    for scope, lines in relation_scopes.items():
        object_lines[scope].update(lines)
    for prefill in workspace.deterministic_prefills:
        line_number = _line_number(prefill.lineId)
        for path in prefill.targetPaths:
            object_lines[_object_scope(path)].add(line_number)

    direct_evidence: dict[str, tuple[set[int], str]] = {}
    for directive in bundle.labelChangeContract:
        raw_auxiliary_identity_lines: set[int] = set()
        if isinstance(directive.sourceValue, str):
            source_pattern = _literal_phrase_pattern(directive.sourceValue)
            for requirement in bundle.rawAuxiliaryIdentityRequirements:
                if source_pattern.search(requirement.sourceIdentity) is None:
                    continue
                start = _line_number(
                    requirement.requirementId[requirement.requirementId.rfind("-L") + 1 :]
                )
                raw_auxiliary_identity_lines.update(
                    range(start, start + requirement.sourceIdentityLineCount)
                )
        cargo_requirement = cargo_requirements.get(directive.path)
        if cargo_requirement is not None:
            direct_lines = cargo_lines(cargo_requirement)
            locator = "exact_literal"
        elif surface_lines_by_path.get(directive.path):
            direct_lines = set(surface_lines_by_path[directive.path])
            locator = "exact_literal"
            party_lines = _party_block_line_numbers(
                workspace.current_text,
                path=directive.path,
                source_label=bundle.sourceLabel,
            )
            if party_lines:
                # Surface requirements predate semantic party-block ownership and can include
                # every global copy of a shared consignee/notify scalar.  Once the source party
                # role is proven, it is the authoritative boundary: intersect when the exact
                # surface is present in that role, otherwise retain the complete role block for
                # wrapped or normalized address values.
                direct_lines = (direct_lines & party_lines) or party_lines
                locator = "rendered_surface_and_party_role_block"
        else:
            # Literal evidence is the narrowest authority.  A party role block is used to
            # disambiguate repeated literals, or as a fallback only when OCR wrapping prevents
            # literal location.  Choosing the whole block first attached name deltas to city and
            # country lines and made otherwise valid rewrites fail occurrence checks.
            direct_lines, locator = _line_set_for_directive(
                workspace.current_text, directive.path, directive.sourceValue
            )
            party_lines = _party_block_line_numbers(
                workspace.current_text,
                path=directive.path,
                source_label=bundle.sourceLabel,
            )
            if party_lines:
                role_literal_lines = direct_lines & party_lines
                if role_literal_lines:
                    direct_lines = role_literal_lines
                    locator = "literal_in_party_role_block"
                elif not direct_lines:
                    direct_lines = party_lines
                    locator = "party_role_block"
        if isinstance(directive.sourceValue, str):
            contextual_location_lines = _contextual_location_line_numbers(
                workspace.current_text,
                directive.path,
                directive.sourceValue,
            )
            contextual_location_lines.difference_update(raw_auxiliary_identity_lines)
            if contextual_location_lines:
                # A globally repeated locality is not owned by every matching line. The
                # semantic route/place heading is stronger evidence than literal equality,
                # including when a derived surface requirement already found global copies.
                direct_lines = contextual_location_lines
                locator = "location_role_surface"
            else:
                direct_lines.difference_update(raw_auxiliary_identity_lines)
        if (
            directive.path == "documentPatch.parties.carrier.name"
            and isinstance(directive.sourceValue, str)
        ):
            principal_groups = carrier_principal_template_slot_groups(
                workspace.current_text, directive.sourceValue
            )
            principal_lines = {number for group in principal_groups for number in group}
            if principal_lines:
                direct_lines.update(principal_lines)
                locator = "carrier_principal_template_slots"
        relation_scope = relation_scopes.get(_object_scope(directive.path), set())
        narrowed = direct_lines & relation_scope
        if direct_lines and relation_scope and narrowed != direct_lines:
            direct_lines = narrowed
            locator = "relation_scoped_surface" if narrowed else "relation_scope_miss"
        direct_evidence[directive.path] = (direct_lines, locator)
        object_lines[_object_scope(directive.path)].update(direct_lines)

    handled_cargo_paths: set[str] = set()
    for ordinal, directive in enumerate(bundle.labelChangeContract, start=1):
        work_id = f"W{ordinal:04d}"
        paths = (directive.path,)
        action: str = directive.action
        source_value = directive.sourceValue
        target_value = directive.targetValue
        item_lines: set[int]
        cargo_requirement = cargo_requirements.get(directive.path)
        if cargo_requirement is not None:
            handled_cargo_paths.add(directive.path)
            action = "rewrite_cargo_flavor_block"
            source_value = cast(JsonValue, list(cargo_requirement.sourceSurfaces))
            target_value = cargo_requirement.targetDescription
            item_lines = cargo_lines(cargo_requirement)
            state = "agent_residual"
            locator = "exact_literal"
            rationale = "Reviewed cargo-block line ownership is explicit in the host contract."
        elif directive.action == "preserve_equivalent_equipment_surface":
            state = "preserved_by_policy"
            locator = "policy_projection"
            item_lines = set()
            rationale = "The reviewed source equipment surface already realizes target semantics."
        elif any(path in prefilled_paths for path in paths):
            state = "deterministic_applied"
            locator = "deterministic_requirement"
            item_lines = set()
            rationale = (
                "A line-bound or typed deterministic requirement already rendered this value."
            )
        elif (
            directive.action == "deactivate_extractable_fact_preserve_slot"
            and directive.path.endswith(".temperatureSetpoint")
        ):
            item_lines = _temperature_deactivation_line_numbers(
                workspace.current_text, directive.sourceValue
            )
            state = "agent_residual" if item_lines else "blocked_unlocated"
            locator = "numeric_surface" if item_lines else "unlocated"
            rationale = (
                "The removed setpoint is bound only to numeric evidence under explicit thermal "
                "context; related reefer statements are inventoried independently."
                if item_lines
                else "No numeric setpoint under explicit thermal context was proven."
            )
        else:
            item_lines, locator = direct_evidence[directive.path]
            state = "agent_residual" if item_lines else "blocked_unlocated"
            rationale = (
                "Source evidence was located without fuzzy matching and is delegated as a "
                "host-owned residual span."
                if item_lines
                else "No exact, whitespace-normalized, numeric, or sibling evidence was proven."
            )
        line_sets[work_id] = item_lines
        object_lines[_object_scope(directive.path)].update(item_lines)
        specs.append(
            {
                "workItemId": work_id,
                "targetPaths": paths,
                "action": action,
                "sourceValue": source_value,
                "targetValue": target_value,
                "state": state,
                "locator": locator,
                "rationale": rationale,
            }
        )

    # Category leaves and target additions often have no literal source representation.  They may
    # only borrow already-proven evidence from the same list object; no global/fuzzy search occurs.
    for spec in specs:
        if spec["state"] != "blocked_unlocated":
            continue
        if spec["action"] == "deactivate_extractable_fact_preserve_slot" and cast(
            tuple[str, ...], spec["targetPaths"]
        )[0].endswith(".temperatureSetpoint"):
            # Never turn a missing temperature locator into write authority over an unrelated
            # sibling such as a container number or seal.  The case must remain blocked.
            continue
        path = cast(str, spec["targetPaths"][0])
        siblings = object_lines.get(_object_scope(path), set())
        if siblings:
            line_sets[cast(str, spec["workItemId"])] = set(siblings)
            spec["state"] = "agent_residual"
            spec["locator"] = "sibling_object_evidence"
            spec["rationale"] = (
                "The non-literal categorical/addition delta is scoped to exact evidence for a "
                "sibling field in the same indexed object."
            )

    next_ordinal = len(specs) + 1

    def add_requirement(
        *,
        paths: tuple[str, ...],
        action: str,
        source: JsonValue,
        target: JsonValue,
        lines: set[int],
        rationale: str,
    ) -> None:
        nonlocal next_ordinal
        work_id = f"W{next_ordinal:04d}"
        next_ordinal += 1
        specs.append(
            {
                "workItemId": work_id,
                "targetPaths": paths,
                "action": action,
                "sourceValue": source,
                "targetValue": target,
                "state": "agent_residual" if lines else "blocked_unlocated",
                "locator": "exact_literal" if lines else "unlocated",
                "rationale": rationale,
            }
        )
        line_sets[work_id] = lines

    for cargo_requirement in bundle.cargoFlavorRewriteRequirements:
        if cargo_requirement.targetPath in handled_cargo_paths:
            continue
        add_requirement(
            paths=(cargo_requirement.targetPath,),
            action="rewrite_cargo_flavor_block",
            source=list(cargo_requirement.sourceSurfaces),
            target=cargo_requirement.targetDescription,
            lines=cargo_lines(cargo_requirement),
            rationale="Reviewed cargo-block line ownership is explicit in the baseline contract.",
        )
    for auxiliary_requirement in bundle.rawAuxiliaryIdentityRequirements:
        match = re.search(r"L[0-9]{5}", auxiliary_requirement.requirementId)
        if match is None:
            raise ValueError(
                "raw auxiliary identity requirement lacks its exact start line: "
                f"{auxiliary_requirement.requirementId}"
            )
        start = _line_number(match.group(0))
        lines = set(
            range(start, start + auxiliary_requirement.sourceIdentityLineCount)
        )
        if not lines or max(lines) > len(workspace.current_text.splitlines()):
            raise ValueError(
                "raw auxiliary identity requirement is outside the OCR: "
                f"{auxiliary_requirement.requirementId}"
            )
        add_requirement(
            paths=(f"rawAuxiliaryIdentity.{auxiliary_requirement.requirementId}",),
            action="anonymize_auxiliary_identity_preserve_relationship",
            source=auxiliary_requirement.sourceIdentity,
            target={
                "targetPrincipalName": auxiliary_requirement.targetPrincipalName,
                "replacementAuxiliaryIdentity": "synthesize_distinct_fictional_identity",
            },
            lines=lines,
            rationale="Raw-only agency identity must be fictionalized while retaining its role.",
        )
    for compound_requirement in bundle.compoundPartyFlavorRequirements:
        lines, _ = _literal_line_numbers(
            workspace.current_text, compound_requirement.sourceLabelName
        )
        add_requirement(
            paths=(compound_requirement.targetPath,),
            action="synthesize_compound_party_flavor",
            source=compound_requirement.sourceLabelName,
            target={
                "targetPrimaryName": compound_requirement.targetPrimaryName,
                "relationships": list(compound_requirement.relationships),
                "auxiliaryIdentity": "synthesize_distinct_fictional_identity",
            },
            lines=lines,
            rationale=(
                "Source legal-party topology requires a distinct fictional auxiliary identity."
            ),
        )
    applied_requirement_ids = {row.requirementId for row in compiler_edits}
    for operational_requirement in bundle.operationalFlavorRequirements:
        line_number = _line_number(operational_requirement.sourceLineId)
        work_id = f"W{next_ordinal:04d}"
        next_ordinal += 1
        applied = (
            operational_requirement.requirementId in applied_requirement_ids
            or f"rawOperational.{operational_requirement.requirementId}" in prefilled_paths
        )
        specs.append(
            {
                "workItemId": work_id,
                "targetPaths": (f"rawOperational.{operational_requirement.requirementId}",),
                "action": "render_operational_surface",
                "sourceValue": operational_requirement.sourceValueSurface,
                "targetValue": operational_requirement.targetValueSurface,
                "state": "deterministic_applied" if applied else "agent_residual",
                "locator": ("deterministic_requirement" if applied else "exact_literal"),
                "rationale": (
                    "Exact line-bound operational surface was compiled locally."
                    if applied
                    else (
                        "Operational grammar was line-bound but not unambiguous enough for a "
                        "local write."
                    )
                ),
            }
        )
        line_sets[work_id] = set() if applied else {line_number}
    for jurisdiction_requirement in bundle.jurisdictionalSurfaceRequirements:
        requirement_id = jurisdiction_requirement.requirementId
        work_id = f"W{next_ordinal:04d}"
        next_ordinal += 1
        applied = requirement_id in applied_requirement_ids
        lines = {
            _line_number(line_id)
            for line_id in jurisdiction_requirement.sourceLineIds
            if 1 <= _line_number(line_id) <= len(workspace.current_text.splitlines())
        }
        specs.append(
            {
                "workItemId": work_id,
                "targetPaths": (f"rawJurisdiction.{jurisdiction_requirement.programId}",),
                "action": "generalize_stale_jurisdiction_surface",
                "sourceValue": jurisdiction_requirement.sourceSurface,
                "targetValue": jurisdiction_requirement.targetSurface,
                "state": "deterministic_applied"
                if applied
                else ("agent_residual" if lines else "blocked_unlocated"),
                "locator": "deterministic_requirement"
                if applied
                else ("exact_literal" if lines else "unlocated"),
                "rationale": (
                    "Exact audited jurisdiction surface was compiled locally."
                    if applied
                    else "Jurisdiction surface retained for scoped residual rewriting."
                ),
            }
        )
        line_sets[work_id] = set() if applied else lines

    core_lines = (
        set().union(
            *(
                line_sets[cast(str, spec["workItemId"])]
                for spec in specs
                if spec["state"] == "agent_residual"
            )
        )
        if any(spec["state"] == "agent_residual" for spec in specs)
        else set()
    )
    intervals = _expanded_intervals(
        workspace.current_text,
        core_lines,
        context=context_lines,
        merge_gap=merge_gap_lines,
    )
    spans: list[ResidualSpan] = []
    bodies = workspace.current_text.splitlines()
    work_to_spans: dict[str, list[str]] = defaultdict(list)
    trimmed_intervals: list[tuple[int, int]] = []
    for interval in intervals:
        # Blank lines added only as surrounding context cannot carry a located work item, and the
        # legacy line-range API intentionally strips one trailing newline terminator.  Excluding
        # those blank edges removes that serialization ambiguity while preserving every blank line
        # inside a semantic span and every core evidence line.
        trimmed_intervals.append(_trim_context_blank_edges(bodies, interval, core_lines))
    for ordinal, (start, end) in enumerate(trimmed_intervals, start=1):
        span_id = f"S{ordinal:03d}"
        work_ids = tuple(
            cast(str, spec["workItemId"])
            for spec in specs
            if spec["state"] == "agent_residual"
            and line_sets[cast(str, spec["workItemId"])] & set(range(start, end + 1))
        )
        if not work_ids:
            raise ValueError("residual span has no owning work item")
        for work_id in work_ids:
            work_to_spans[work_id].append(span_id)
        spans.append(
            ResidualSpan(
                spanId=span_id,
                startLine=start,
                endLine=end,
                workItemIds=work_ids,
                lines=tuple(bodies[start - 1 : end]),
            )
        )
    work_items = tuple(
        HybridWorkItem(
            **spec,
            evidenceLineIds=tuple(
                _line_id(line_number, bodies[line_number - 1])
                for line_number in sorted(line_sets[cast(str, spec["workItemId"])])
            ),
            spanIds=tuple(work_to_spans.get(cast(str, spec["workItemId"]), ())),
        )
        for spec in specs
    )
    if any(item.state == "agent_residual" and not item.spanIds for item in work_items):
        raise ValueError("agent residual work item has no host-owned span")
    return _CompiledCase(
        bundle=bundle,
        workspace=workspace,
        work_items=work_items,
        spans=tuple(spans),
        compiler_edits=compiler_edits,
    )


def _native_residual_output_type(context: _ResidualCommitContext) -> type[BaseModel]:
    """Build a changed-line-only schema bounded to the host-owned span set.

    Context lines are input-only; echoing them made output size proportional to document length
    instead of edit size. The model returns only changed line IDs and replacements. The host
    applies them after provider-native schema validation, keeping mutation outside the model.
    """

    if not context.spans:
        raise ValueError("residual output requires at least one span")
    visible_line_ids = {
        _line_id(line_number, line)
        for span in context.spans.values()
        for line_number, line in zip(
            range(span.startLine, span.endLine + 1), span.lines, strict=True
        )
    }
    if not context.authorized_line_ids or not context.authorized_line_ids <= visible_line_ids:
        raise ValueError("authorized residual line IDs must be a non-empty subset of span lines")
    line_ids = tuple(sorted(context.authorized_line_ids, key=_line_number))
    if len(line_ids) != len(set(line_ids)):
        raise ValueError("residual spans contain duplicate line IDs")
    line_id_type = cast(Any, Literal).__getitem__(line_ids)
    suffix = sha256_bytes("\n".join(line_ids).encode("utf-8"))[:12]
    edit_model = create_model(
        f"HostOwnedResidualLineEdit_{suffix}",
        __config__=_STRICT,
        lineId=(line_id_type, Field(description="One exact authorized OCR line ID.")),
        newLine=(
            Annotated[str, StringConstraints(min_length=1, max_length=10_000)],
            Field(description="Complete replacement text for this line, without a newline."),
        ),
    )
    edits_type = cast(Any, Annotated).__class_getitem__(
        (
            list.__class_getitem__(edit_model),
            Field(min_length=1, max_length=len(line_ids)),
        )
    )
    flavor_type = Annotated[
        list[CompoundPartyFlavorRealization],
        Field(max_length=6),
    ]

    return create_model(
        f"NativeResidualRewrite_{suffix}",
        __config__=_STRICT,
        edits=(edits_type, ...),
        compoundPartyFlavorRealizations=(flavor_type, Field(default_factory=list)),
    )


def _apply_native_residual_output(
    context: _ResidualCommitContext,
    output: BaseModel,
) -> AtomicRewriteCommit:
    dumped = output.model_dump(mode="python")
    parsed = cast(list[dict[str, Any]], dumped["edits"])
    observed_ids = [cast(str, row["lineId"]) for row in parsed]
    if len(observed_ids) != len(set(observed_ids)):
        raise ValueError("native residual output repeats an authorized line ID")
    current_lines = context.workspace.current_text.splitlines()
    authorized_lines = {
        line_id: (
            _line_number(line_id),
            current_lines[_line_number(line_id) - 1],
        )
        for line_id in context.authorized_line_ids
    }
    if not set(observed_ids) <= set(authorized_lines):
        raise ValueError("native residual output references a line outside its host-owned spans")
    replacements: list[LineRangeReplacement] = []
    for row in sorted(parsed, key=lambda value: _line_number(cast(str, value["lineId"]))):
        line_id = cast(str, row["lineId"])
        _, before = authorized_lines[line_id]
        new_line = cast(str, row["newLine"])
        if "\n" in new_line or "\r" in new_line:
            raise ValueError("newLine must contain exactly one OCR line")
        if new_line == before:
            continue
        replacements.append(
            LineRangeReplacement(
                startLineId=line_id,
                endLineId=line_id,
                newText=new_line,
            )
        )
    if not replacements:
        raise ValueError("native residual output contains no changed line")
    flavor_realizations = [
        CompoundPartyFlavorRealization.model_validate(row, strict=True)
        for row in cast(list[dict[str, Any]], dumped["compoundPartyFlavorRealizations"])
    ]
    return apply_line_range_replacements(
        context.workspace,
        replacements,
        compound_party_flavor_realizations=flavor_realizations,
    )


def _compact_work_item(item: HybridWorkItem) -> dict[str, JsonValue]:
    return {
        "workItemId": item.workItemId,
        "paths": cast(JsonValue, list(item.targetPaths)),
        "action": item.action,
        "source": item.sourceValue,
        "target": item.targetValue,
        "evidenceLineIds": cast(JsonValue, list(item.evidenceLineIds)),
        "spanIds": cast(JsonValue, list(item.spanIds)),
    }


def _editor_payload(compiled: _CompiledCase) -> dict[str, JsonValue]:
    residual = tuple(item for item in compiled.work_items if item.state == "agent_residual")
    residual_paths = {path for item in residual for path in item.targetPaths}
    authorized_line_ids = {line_id for item in residual for line_id in item.evidenceLineIds}
    original_lines = compiled.workspace.original_text.splitlines()
    rendering_requirements: list[dict[str, JsonValue]] = []
    for requirement in compiled.bundle.surfaceRenderingRequirements:
        if requirement.targetPath not in residual_paths:
            continue
        lines = _context_bound_surface_lines(
            compiled.workspace.original_text,
            requirement,
        )
        rendering_requirements.append(
            {
                "path": requirement.targetPath,
                "sourceSurface": requirement.sourceSurface,
                "targetSurface": requirement.targetSurface,
                "lineIds": cast(
                    JsonValue,
                    [_line_id(line, original_lines[line - 1]) for line in sorted(lines)],
                ),
            }
        )
    return {
        "task": "Complete the residual semantic substitutions in the authorized OCR spans.",
        "documentId": compiled.bundle.result.documentId,
        "workItems": cast(JsonValue, [_compact_work_item(item) for item in residual]),
        "requiredRenderedSurfaces": cast(JsonValue, rendering_requirements),
        "protectedLineFragments": cast(
            JsonValue,
            [
                {
                    "lineId": prefill.lineId,
                    "value": prefill.targetSurface,
                    "paths": list(prefill.targetPaths),
                }
                for prefill in compiled.workspace.deterministic_prefills
                if prefill.lineId in authorized_line_ids
            ],
        ),
        "spans": cast(
            JsonValue,
            [
                {
                    "spanId": span.spanId,
                    "workItemIds": list(span.workItemIds),
                    "lines": [
                        {
                            "lineId": _line_id(line_number, line),
                            "text": line,
                        }
                        for line_number, line in zip(
                            range(span.startLine, span.endLine + 1),
                            span.lines,
                            strict=True,
                        )
                    ],
                }
                for span in compiled.spans
            ],
        ),
        "targetOccurrenceRequirements": cast(
            JsonValue,
            [
                {
                    "paths": list(row.targetPaths),
                    "value": row.targetValue,
                    "count": row.requiredOccurrences,
                }
                for row in compiled.workspace.target_value_occurrence_requirements
                if any(
                    path in {p for item in residual for p in item.targetPaths}
                    for path in row.targetPaths
                )
            ],
        ),
        "compoundPartyFlavorRequirements": cast(
            JsonValue,
            [
                row.model_dump(mode="json")
                for row in compiled.bundle.compoundPartyFlavorRequirements
            ],
        ),
        "rawAuxiliaryIdentityRequirements": cast(
            JsonValue,
            [
                row.model_dump(mode="json")
                for row in compiled.bundle.rawAuxiliaryIdentityRequirements
            ],
        ),
    }


def _review_payload(
    compiled: _CompiledCase,
    *,
    audit: DeterministicRewriteAudit,
) -> dict[str, JsonValue]:
    current_lines = compiled.workspace.current_text.splitlines()
    return {
        "task": "Independently verify only the compiled residual changes.",
        "documentId": compiled.bundle.result.documentId,
        "workItems": cast(
            JsonValue,
            [
                _compact_work_item(item)
                for item in compiled.work_items
                if item.state == "agent_residual"
            ],
        ),
        "spanDiffs": cast(
            JsonValue,
            [
                {
                    "spanId": span.spanId,
                    "before": list(span.lines),
                    "after": current_lines[span.startLine - 1 : span.endLine],
                }
                for span in compiled.spans
            ],
        ),
        "hostAudit": {
            "corePassed": audit.core_passed,
            "requiredTargetLiteralsRendered": audit.requiredTargetLiteralsRendered,
            "targetValueOccurrencesExact": audit.targetValueOccurrencesExact,
            "rawAuxiliaryIdentitiesReplaced": audit.rawAuxiliaryIdentitiesReplaced,
            "operationalFlavorRendered": audit.operationalFlavorRendered,
            "cargoFlavorRewritten": audit.cargoFlavorRewritten,
        },
    }


def _payload_sha256(payload: Mapping[str, JsonValue]) -> str:
    return sha256_bytes(canonical_json_bytes(payload))


def _baseline_reference_gap_lines(
    source_text: str,
    current_text: str,
    audited_reference_text: str,
) -> tuple[int, ...]:
    """Find unchanged source lines that a retained, audited full-context rewrite changed."""

    source = source_text.splitlines()
    current = current_text.splitlines()
    reference = audited_reference_text.splitlines()
    if not (len(source) == len(current) == len(reference)):
        raise ValueError("source, current, and audited reference line counts differ")
    return tuple(
        line_number
        for line_number, (source_line, current_line, reference_line) in enumerate(
            zip(source, current, reference, strict=True),
            start=1,
        )
        if current_line == source_line and reference_line != source_line
    )


async def _call_editor(
    *,
    compiled: _CompiledCase,
    model: Model,
    provider: RawTextRewriteProviderConfig,
    prompt: str,
    prompt_sha256: str,
) -> tuple[AtomicRewriteCommit | None, tuple[HybridModelStage, ...]]:
    payload = _editor_payload(compiled)
    context = _ResidualCommitContext(
        workspace=compiled.workspace,
        spans={span.spanId: span for span in compiled.spans},
        authorized_line_ids=frozenset(
            line_id
            for item in compiled.work_items
            if item.state == "agent_residual"
            for line_id in item.evidenceLineIds
        ),
    )
    output_type = _native_residual_output_type(context)
    routes = _provider_attempt_routes(provider)
    stages: list[HybridModelStage] = []
    for route_index, route_provider in enumerate(routes):
        output_spec = NativeOutput(
            output_type,
            name="synthetic_bl_residual_line_edits",
            description=(
                "Return only changed host-owned OCR lines, each as one complete replacement."
            ),
            strict=True,
        )
        agent = Agent[None, BaseModel](
            model,
            output_type=output_spec,
            system_prompt=prompt,
            model_settings=_settings(
                provider,
                stage="editor",
                prompt_sha256=prompt_sha256,
                route_provider=route_provider,
            ),
            retries={"output": 0},
            name="synthetic-bl-hybrid-residual-editor",
        )
        started = time.time()
        output: AtomicRewriteCommit | None = None
        error: Exception | None = None
        with capture_run_messages() as captured:
            try:
                result = await agent.run(
                    json.dumps(payload, ensure_ascii=False, separators=(",", ":")),
                    usage_limits=UsageLimits(
                        request_limit=1,
                        output_tokens_limit=provider.max_output_tokens,
                    ),
                )
                output = _apply_native_residual_output(context, result.output)
            except Exception as caught:
                error = caught
        responses = tuple(row for row in captured if isinstance(row, ModelResponse))
        stages.append(
            HybridModelStage(
                stage="residual_editor",
                providerModel=provider.model,
                routeProvider=route_provider,
                reasoningEffort=provider.reasoning_effort,
                inputPayloadSha256=_payload_sha256(payload),
                startedAtUnixSeconds=started,
                completedAtUnixSeconds=time.time(),
                usage=usage_receipt(
                    responses,
                    provider.pricing,
                    require_provider_cost=provider.kind == "openrouter",
                ),
                messages=model_messages(captured),
                errorType=type(error).__name__ if error is not None else None,
                errorMessage=str(error) if error is not None else None,
            )
        )
        if output is not None:
            return output, tuple(stages)
        if error is None or not _retryable_route_error(error) or route_index + 1 == len(routes):
            break
    return None, tuple(stages)


async def _call_reviewer(
    *,
    compiled: _CompiledCase,
    audit: DeterministicRewriteAudit,
    model: Model,
    provider: RawTextRewriteProviderConfig,
    prompt: str,
    prompt_sha256: str,
) -> tuple[HybridReviewReceipt | None, tuple[HybridModelStage, ...]]:
    payload = _review_payload(compiled, audit=audit)
    spans_by_id = {span.spanId: span for span in compiled.spans}
    current_lines = compiled.workspace.current_text.splitlines()
    work_items_by_span = {span.spanId: frozenset(span.workItemIds) for span in compiled.spans}

    def validate_review(output: HybridReviewReceipt) -> HybridReviewReceipt:
        checks = (output.semanticChanges, output.staleSourceFacts, output.formattingAndTopology)
        if output.verdict == "pass" and (
            output.findings or any(value != "pass" for value in checks)
        ):
            raise ModelRetry("A pass verdict requires all checks to pass and zero findings.")
        if output.verdict == "revise" and (
            not output.findings or all(value == "pass" for value in checks)
        ):
            raise ModelRetry("A revise verdict requires a failed check and a grounded finding.")
        for finding in output.findings:
            if finding.spanId not in spans_by_id:
                raise ModelRetry("Finding references an unknown span ID.")
            if not set(finding.workItemIds) <= work_items_by_span[finding.spanId]:
                raise ModelRetry("Finding references a work item not owned by its cited span.")
            span = spans_by_id[finding.spanId]
            span_after = "\n".join(current_lines[span.startLine - 1 : span.endLine])
            if _semantic_surface(finding.evidence) not in _semantic_surface(span_after):
                raise ModelRetry("Finding evidence does not occur in a rewritten residual span.")
        if output.verdict == "pass" and not audit.core_passed:
            raise ModelRetry("Reviewer cannot pass a failed host audit.")
        return output

    routes = _provider_attempt_routes(provider)
    stages: list[HybridModelStage] = []
    for route_index, route_provider in enumerate(routes):
        output_spec = NativeOutput(
            HybridReviewReceipt,
            name="synthetic_bl_residual_span_review",
            description=("Compact evidence-grounded verdict over only the residual span changes."),
            strict=True,
        )
        agent = Agent[None, HybridReviewReceipt](
            model,
            output_type=output_spec,
            system_prompt=prompt,
            model_settings=_settings(
                provider,
                stage="reviewer",
                prompt_sha256=prompt_sha256,
                route_provider=route_provider,
            ),
            retries={"output": 0},
            name="synthetic-bl-hybrid-residual-reviewer",
        )
        agent.output_validator(validate_review)
        started = time.time()
        output: HybridReviewReceipt | None = None
        error: Exception | None = None
        with capture_run_messages() as captured:
            try:
                result = await agent.run(
                    json.dumps(payload, ensure_ascii=False, separators=(",", ":")),
                    usage_limits=UsageLimits(
                        request_limit=1,
                        output_tokens_limit=provider.max_output_tokens,
                    ),
                )
                output = result.output
            except Exception as caught:
                error = caught
        responses = tuple(row for row in captured if isinstance(row, ModelResponse))
        stages.append(
            HybridModelStage(
                stage="residual_reviewer",
                providerModel=provider.model,
                routeProvider=route_provider,
                reasoningEffort=provider.reasoning_effort,
                inputPayloadSha256=_payload_sha256(payload),
                startedAtUnixSeconds=started,
                completedAtUnixSeconds=time.time(),
                usage=usage_receipt(
                    responses,
                    provider.pricing,
                    require_provider_cost=provider.kind == "openrouter",
                ),
                messages=model_messages(captured),
                errorType=type(error).__name__ if error is not None else None,
                errorMessage=str(error) if error is not None else None,
            )
        )
        if output is not None:
            return output, tuple(stages)
        if error is None or not _retryable_route_error(error) or route_index + 1 == len(routes):
            break
    return None, tuple(stages)


def _find_bundle(root: Path, document_id: str) -> RewriteCycleBundle:
    matches = tuple(root.glob(f"cases/*-{document_id}/bundle.json"))
    if len(matches) != 1:
        raise ValueError(
            f"expected exactly one audited baseline bundle for {document_id}, found {len(matches)}"
        )
    path = matches[0]
    if path.is_symlink() or not path.is_file():
        raise ValueError(f"baseline bundle is not a regular file: {path}")
    return RewriteCycleBundle.model_validate_json(read_regular_file_bytes(path), strict=True)


def _model_pair(
    *,
    client: AsyncOpenAI,
    editor_provider: RawTextRewriteProviderConfig,
    reviewer_provider: RawTextRewriteProviderConfig,
) -> tuple[Model, Model]:
    if editor_provider.kind == "openrouter":
        openrouter_provider = OpenRouterProvider(openai_client=client)
        # OpenRouter's gateway supports native ``response_format`` even when PydanticAI has no
        # vendor profile for a model prefix (currently true for ``z-ai/*``).  The request keeps
        # ``require_parameters=true``, so OpenRouter will only select downstream endpoints that
        # advertise response-format support rather than silently dropping the schema.
        native_schema_profile = ModelProfile(supports_json_schema_output=True)
        editor: Model = OpenRouterModel(
            editor_provider.model,
            provider=openrouter_provider,
            profile=merge_profile(
                openrouter_provider.model_profile(editor_provider.model),
                native_schema_profile,
            ),
        )
        reviewer: Model = OpenRouterModel(
            reviewer_provider.model,
            provider=openrouter_provider,
            profile=merge_profile(
                openrouter_provider.model_profile(reviewer_provider.model),
                native_schema_profile,
            ),
        )
    else:
        openai_provider = OpenAIProvider(openai_client=client)
        editor = OpenAIResponsesModel(editor_provider.model, provider=openai_provider)
        reviewer = OpenAIResponsesModel(reviewer_provider.model, provider=openai_provider)
    return editor, reviewer


async def _run_model_case(
    *,
    compiled: _CompiledCase,
    editor_model: Model,
    reviewer_model: Model,
    providers: RawTextRewriteCycleProvidersConfig,
    editor_prompt: str,
    editor_prompt_sha256: str,
    reviewer_prompt: str,
    reviewer_prompt_sha256: str,
    audited_reference_text: str | None,
) -> tuple[HybridCaseResult, tuple[HybridModelStage, ...]]:
    stages: list[HybridModelStage] = []
    blocked = sum(item.state == "blocked_unlocated" for item in compiled.work_items)
    deterministic = sum(
        item.state in {"deterministic_applied", "preserved_by_policy"}
        for item in compiled.work_items
    )
    residual = sum(item.state == "agent_residual" for item in compiled.work_items)
    source_lines = len(compiled.workspace.original_text.splitlines())
    residual_lines = sum(span.endLine - span.startLine + 1 for span in compiled.spans)
    review: HybridReviewReceipt | None = None
    audit: DeterministicRewriteAudit | None = None

    if blocked:
        status: Literal["quality_validated", "needs_review", "call_failed", "compiler_blocked"] = (
            "compiler_blocked"
        )
        reason = "One or more semantic deltas lack an exact host-owned evidence span."
    elif not compiled.spans:
        audit = deterministic_rewrite_audit(compiled.workspace, compiled.bundle.changedLeaves)
        status = "quality_validated" if audit.core_passed else "needs_review"
        reason = (
            "All work was deterministic and the host audit passed."
            if audit.core_passed
            else "Deterministic-only output failed the host audit."
        )
    else:
        commit, editor_stages = await _call_editor(
            compiled=compiled,
            model=editor_model,
            provider=providers.editor,
            prompt=editor_prompt,
            prompt_sha256=editor_prompt_sha256,
        )
        stages.extend(editor_stages)
        if commit is None:
            status = "call_failed"
            reason = "The one-shot residual editor call failed; no partial patch was committed."
        else:
            audit = deterministic_rewrite_audit(compiled.workspace, compiled.bundle.changedLeaves)
            if not audit.core_passed:
                status = "needs_review"
                reason = (
                    "The residual edit committed atomically but failed the host fidelity audit."
                )
            else:
                review, review_stages = await _call_reviewer(
                    compiled=compiled,
                    audit=audit,
                    model=reviewer_model,
                    provider=providers.reviewer,
                    prompt=reviewer_prompt,
                    prompt_sha256=reviewer_prompt_sha256,
                )
                stages.extend(review_stages)
                if review is None:
                    status = "call_failed"
                    reason = "The compact independent reviewer call failed."
                elif review.verdict == "pass":
                    status = "quality_validated"
                    reason = "Host audit and compact independent residual review both passed."
                else:
                    status = "needs_review"
                    reason = "Compact residual review found a semantic or formatting defect."
    total_usage = _combine_usage([stage.usage for stage in stages]) if stages else _empty_usage()
    reference_gap_lines = (
        _baseline_reference_gap_lines(
            compiled.workspace.original_text,
            compiled.workspace.current_text,
            audited_reference_text,
        )
        if audited_reference_text is not None
        else None
    )
    if status == "quality_validated" and reference_gap_lines:
        status = "needs_review"
        reason = (
            f"The declared contract passed, but {len(reference_gap_lines)} source lines changed "
            "by the audited full-context reference remain byte-identical; source-only or "
            "duplicated semantic slots are not yet modeled."
        )
    result = HybridCaseResult(
        documentId=compiled.bundle.result.documentId,
        status=status,
        reason=reason,
        sourceLines=source_lines,
        residualLines=residual_lines,
        residualSpans=len(compiled.spans),
        workItems=len(compiled.work_items),
        deterministicWorkItems=deterministic,
        residualWorkItems=residual,
        blockedWorkItems=blocked,
        deterministicPrefills=len(compiled.workspace.deterministic_prefills),
        compilerEdits=len(compiled.compiler_edits),
        outputTextSha256=sha256_bytes(compiled.workspace.current_text.encode("utf-8")),
        baselineOutputTextSha256=(
            sha256_bytes(audited_reference_text.encode("utf-8"))
            if audited_reference_text is not None
            else None
        ),
        exactBaselineMatch=(
            compiled.workspace.current_text == audited_reference_text
            if audited_reference_text is not None
            else None
        ),
        baselineReferenceGapLines=reference_gap_lines,
        deterministicAudit=audit,
        review=review,
        usage=total_usage,
    )
    return result, tuple(stages)


def _publish_jsonl(staged: StagedArtifactRun, path: str, rows: Sequence[Mapping[str, Any]]) -> None:
    payload = b"".join(canonical_json_bytes(row) + b"\n" for row in rows)
    staged.publish_bytes(path, payload)


def _plot_bytes(
    coverage: Mapping[str, Any],
    compiler_rows: Sequence[Mapping[str, Any]],
    results: Sequence[HybridCaseResult],
    baselines: Mapping[str, RewriteCycleBundle],
) -> dict[str, bytes]:
    import io

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import pandas as pd  # type: ignore[import-untyped]
    import seaborn as sns  # type: ignore[import-untyped]

    sns.set_theme(style="whitegrid", context="notebook")
    plots: dict[str, bytes] = {}

    def render(name: str, figure: Any) -> None:
        stream = io.BytesIO()
        figure.savefig(stream, format="png", dpi=180, bbox_inches="tight")
        plots[name] = stream.getvalue()
        plt.close(figure)

    root_rows = []
    for root, value in cast(Mapping[str, Mapping[str, Any]], coverage["byRoot"]).items():
        leaves = cast(int, value["leaves"])
        exact = cast(int, value["exactLiteralCandidates"])
        root_rows.append({"root": root, "leaves": leaves, "exact_fraction": exact / leaves})
    root_frame = pd.DataFrame(root_rows).sort_values("leaves", ascending=False)
    figure, axes = plt.subplots(1, 2, figsize=(15, 6))
    sns.barplot(data=root_frame, y="root", x="leaves", ax=axes[0], color="#4C78A8")
    sns.barplot(data=root_frame, y="root", x="exact_fraction", ax=axes[1], color="#59A14F")
    axes[0].set(title="Changed text leaves by schema root", xlabel="leaves", ylabel="")
    axes[1].set(title="Exact literal discoverability", xlabel="fraction", ylabel="")
    axes[1].set_xlim(0, 1)
    render("01_literal_coverage_by_root.png", figure)

    compiler_frame = pd.DataFrame(compiler_rows)
    figure, axes = plt.subplots(1, 2, figsize=(14, 5.5))
    sns.barplot(data=compiler_frame, x="caseNumber", y="deterministicFraction", ax=axes[0])
    sns.barplot(data=compiler_frame, x="caseNumber", y="residualLineFraction", ax=axes[1])
    axes[0].set(
        title="Compiler-resolved work-item fraction",
        xlabel="baseline case",
        ylabel="fraction",
    )
    axes[1].set(title="OCR retained in residual windows", xlabel="baseline case", ylabel="fraction")
    axes[0].set_ylim(0, 1)
    axes[1].set_ylim(0, 1)
    render("02_compiler_and_window_coverage.png", figure)

    comparison_rows: list[dict[str, Any]] = []
    for result in results:
        baseline = baselines[result.documentId].result.totalUsage
        for method, usage in (("baseline", baseline), ("hybrid", result.usage)):
            comparison_rows.extend(
                (
                    {
                        "document": result.documentId[:12],
                        "method": method,
                        "metric": "input",
                        "value": usage.inputTokens,
                    },
                    {
                        "document": result.documentId[:12],
                        "method": method,
                        "metric": "reasoning",
                        "value": usage.reasoningTokens,
                    },
                    {
                        "document": result.documentId[:12],
                        "method": method,
                        "metric": "visible_output",
                        "value": usage.visibleOutputTokens,
                    },
                )
            )
    if comparison_rows:
        frame = pd.DataFrame(comparison_rows)
        figure, axes = plt.subplots(1, 3, figsize=(16, 5))
        for axis, metric in zip(axes, ("input", "reasoning", "visible_output"), strict=True):
            sns.barplot(
                data=frame[frame.metric == metric],
                x="document",
                y="value",
                hue="method",
                ax=axis,
            )
            axis.set(title=f"{metric.replace('_', ' ').title()} tokens", xlabel="", ylabel="tokens")
        render("03_baseline_vs_hybrid_tokens.png", figure)
    return plots


def _artifact_inventory(root: Path) -> tuple[str, ...]:
    files: list[str] = []
    for directory, directory_names, filenames in os.walk(root, followlinks=False):
        directory_path = Path(directory)
        if any((directory_path / name).is_symlink() for name in directory_names):
            raise ValueError("hybrid artifact tree contains a symlink directory")
        for name in filenames:
            path = directory_path / name
            if path.is_symlink():
                raise ValueError("hybrid artifact tree contains a symlink file")
            relative = path.relative_to(root).as_posix()
            if relative not in {"_TRANSACTION.json", "_COMMIT.json"}:
                files.append(relative)
    return tuple(sorted(files))


def _money(value: Decimal | None) -> str:
    return "unavailable" if value is None else f"${value}"


def _report(
    *,
    summary: Mapping[str, Any],
    coverage: Mapping[str, Any],
    compiler_rows: Sequence[Mapping[str, Any]],
    results: Sequence[HybridCaseResult],
    baselines: Mapping[str, RewriteCycleBundle],
) -> str:
    lines = [
        "# Compiler-first synthetic raw-OCR rewrite probe",
        "",
        "## Outcome",
        "",
        f"- Zero-API corpus audit: **{coverage['documents']} documents / "
        f"{coverage['changedTextLeaves']} changed text leaves**.",
        f"- Exact literal discoverability: **{coverage['exactLiteralCandidates']} "
        f"({coverage['exactLiteralCandidateFraction']:.1%})**. This is discovery coverage, "
        "not permission to mutate.",
        f"- Enriched compiler audit: **{len(compiler_rows)} previously audited hard cases**.",
        f"- Live proof cases: **{summary['qualityValidatedCases']}/{summary['modelCases']} "
        "quality validated**.",
        f"- Live requests / input / reasoning / visible output: **{summary['requests']} / "
        f"{summary['inputTokens']} / {summary['reasoningTokens']} / "
        f"{summary['visibleOutputTokens']}**.",
        f"- Provider-reported live cost: **{_money(summary['providerReportedCostUsd'])}**.",
        "- Training records published: **no**.",
        "",
        "## What was tested",
        "",
        "### Branch A — materialized files and generic filesystem tools",
        "",
        "Every selected case is materialized as immutable `source.txt`, "
        "`after-deterministic.txt`, `work-items.json`, `residual-spans.json`, and `final.txt`. "
        "This improves auditability but does not itself reduce provider tokens: an external model "
        "must still receive bytes returned by `read`/`search` calls. PydanticAI's FileSystem "
        "capability offers read/edit/search and optimistic hashes, while CodeMode can keep "
        "intermediate tool results out of history. Neither removes the need to transmit selected "
        "OCR content, so the generic filesystem branch was not used for the live default.",
        "",
        "### Branch B — host-owned rewrite compiler",
        "",
        "The compiler separates discovery from write authority. Exact literals, flexible "
        "whitespace matches, numeric surfaces, and same-object sibling evidence may locate a "
        "candidate. Only pre-existing line-bound/typed requirements and exact audited "
        "operational/jurisdiction rules are applied without a model. Everything else becomes a "
        "stable work item tied to a non-overlapping host-owned span; anything unlocated blocks "
        "the call. Fuzzy matching never authorizes an edit.",
        "",
        "### Branch C — compact constrained residual model",
        "",
        "GLM receives only residual work items and authorized spans—not the complete OCR, source "
        "label, or target label. It emits one strict PydanticAI terminal output containing every "
        "span exactly once. The host maps opaque span IDs back to immutable line ranges and reuses "
        "the existing transactional validators. There is no search/read/edit loop and no graph.",
        "",
        "### Branch D — risk-scoped independent review",
        "",
        "The reviewer sees only work items plus before/after residual spans and the host-audit "
        "booleans. Its schema requires an evidence-grounded pass or revise verdict. This tests "
        "whether full-document review can be removed from the common path without removing "
        "independent semantic oversight.",
        "",
        "## Zero-API coverage by root",
        "",
        "| Root | Changed leaves | Exact literal candidates | Fraction |",
        "|---|---:|---:|---:|",
    ]
    for root, values in cast(Mapping[str, Mapping[str, Any]], coverage["byRoot"]).items():
        leaves = cast(int, values["leaves"])
        exact = cast(int, values["exactLiteralCandidates"])
        lines.append(f"| `{root}` | {leaves} | {exact} | {exact / leaves:.1%} |")
    lines.extend(
        [
            "",
            "## Enriched compiler audit",
            "",
            "| Case | Document | Work items | Deterministic/preserved | Residual | Blocked | "
            "Residual OCR lines |",
            "|---:|---|---:|---:|---:|---:|---:|",
        ]
    )
    for row in compiler_rows:
        lines.append(
            f"| {row['caseNumber']} | `{row['documentId']}` | {row['workItems']} | "
            f"{row['deterministicWorkItems']} | {row['residualWorkItems']} | "
            f"{row['blockedWorkItems']} | {row['residualLines']}/{row['sourceLines']} "
            f"({row['residualLineFraction']:.1%}) |"
        )
    lines.extend(
        [
            "",
            "## Live model proof",
            "",
            "| Document | Outcome | Residual lines | Reference gaps | "
            "Baseline requests/input/reasoning/visible | "
            "Hybrid requests/input/reasoning/visible | Baseline cost | Hybrid cost |",
            "|---|---|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for result in results:
        baseline = baselines[result.documentId].result.totalUsage
        lines.append(
            f"| `{result.documentId}` | {result.status} | {result.residualLines}/"
            f"{result.sourceLines} | {len(result.baselineReferenceGapLines or ())} | "
            f"{baseline.requests}/{baseline.inputTokens}/"
            f"{baseline.reasoningTokens}/{baseline.visibleOutputTokens} | "
            f"{result.usage.requests}/{result.usage.inputTokens}/"
            f"{result.usage.reasoningTokens}/{result.usage.visibleOutputTokens} | "
            f"{_money(baseline.providerReportedCostUsd)} | "
            f"{_money(result.usage.providerReportedCostUsd)} |"
        )
    lines.extend(
        [
            "",
            "## Interpretation and decision gates",
            "",
            "- A Markdown/file representation is retained for humans and reproducibility; it is "
            "not treated as a token-saving mechanism by itself.",
            "- The useful capability is a domain-specific, terminal, schema-constrained patch "
            "operation. Generic filesystem or code execution would add turns and a larger attack "
            "surface for this task.",
            "- The 50-document literal census is an upper-bound discovery result. Repeated or "
            "cross-role values stay residual until a role parser proves their span ownership.",
            "- A declared-contract pass is withheld when an unchanged source line differs from "
            "the retained audited full-context rewrite. This reference-gap guard exposes "
            "source-only identifiers and duplicated operational facts that the compact contract "
            "does not yet model.",
            "- This probe does not publish synthetic training rows. Production promotion requires "
            "coverage on a larger held-out set, explicit typed renderers for dates/measures/"
            "quantities, and zero unlocated work items.",
            "- Provider/model choice remains configuration-only. The live config uses OpenRouter "
            "GLM, while the same provider union still supports OpenAI/Luna.",
            "",
            "## Retained evidence",
            "",
            "- `analysis/literal-coverage.json` and `analysis/document-coverage.jsonl`: complete "
            "zero-API census.",
            "- `analysis/compiler-cases.jsonl`: ten enriched hard-case compiler measurements.",
            "- `cases/<document>/`: source, deterministic intermediate, spans, work items, exact "
            "model messages, usage receipts, final OCR, diff, audit, and review.",
            "- `plots/`: literal coverage, compiler/window coverage, and baseline-versus-hybrid "
            "token charts generated with matplotlib/seaborn.",
            "",
            "## References consulted",
            "",
            "- PydanticAI capabilities: https://pydantic.dev/docs/ai/capabilities/overview/",
            "- PydanticAI FileSystem: https://pydantic.dev/docs/ai/harness/filesystem/",
            "- PydanticAI CodeMode: https://pydantic.dev/docs/ai/harness/code-mode/",
            "- PydanticAI toolsets: https://pydantic.dev/docs/ai/tools-toolsets/toolsets/",
            "- PydanticAI output functions: https://pydantic.dev/docs/ai/core-concepts/output/",
            "- OpenRouter reasoning controls: "
            "https://openrouter.ai/docs/guides/best-practices/reasoning-tokens",
            "",
        ]
    )
    return "\n".join(lines)


def run_raw_text_hybrid_probe(
    *,
    project_root: Path,
    config_path: Path,
    config: SynthesisRawTextHybridProbeConfig,
) -> dict[str, JsonValue]:
    baseline_root = _validate_committed_run(project_root, config.inputs.baseline_atomic_run)
    source_path = _resolve_pinned_file(
        project_root,
        config.inputs.source_corpus.path,
        config.inputs.source_corpus.sha256,
        label="hybrid source corpus",
    )
    target_path = _resolve_pinned_file(
        project_root,
        config.inputs.synthetic_targets.path,
        config.inputs.synthetic_targets.sha256,
        label="hybrid synthetic targets",
    )
    editor_prompt_path = _resolve_pinned_file(
        project_root,
        config.prompts.editor.path,
        config.prompts.editor.sha256,
        label="hybrid editor prompt",
    )
    reviewer_prompt_path = _resolve_pinned_file(
        project_root,
        config.prompts.reviewer.path,
        config.prompts.reviewer.sha256,
        label="hybrid reviewer prompt",
    )
    source_rows = _load_jsonl(
        source_path,
        records=config.inputs.source_corpus.records,
        key="documentId",
        label="hybrid source corpus",
    )
    target_rows = _load_jsonl(
        target_path,
        records=config.inputs.synthetic_targets.records,
        key="baseDocumentId",
        label="hybrid synthetic targets",
    )
    coverage, coverage_rows = audit_literal_coverage(
        source_rows,
        target_rows,
        limit=config.workflow.audit_documents,
    )
    editor_prompt_bytes = read_regular_file_bytes(editor_prompt_path)
    reviewer_prompt_bytes = read_regular_file_bytes(reviewer_prompt_path)
    editor_prompt = editor_prompt_bytes.decode("utf-8")
    reviewer_prompt = reviewer_prompt_bytes.decode("utf-8")

    baseline_bundle_paths = sorted(baseline_root.glob("cases/*/bundle.json"))
    if not baseline_bundle_paths:
        raise ValueError("baseline atomic run contains no case bundles")
    baseline_bundles = tuple(
        RewriteCycleBundle.model_validate_json(read_regular_file_bytes(path), strict=True)
        for path in baseline_bundle_paths
    )
    baseline_by_id = {bundle.result.documentId: bundle for bundle in baseline_bundles}
    if len(baseline_by_id) != len(baseline_bundles):
        raise ValueError("baseline atomic run contains duplicate document bundles")
    compiled_baselines = tuple(
        compile_case(
            bundle,
            context_lines=config.workflow.context_lines_per_residual,
            merge_gap_lines=config.workflow.merge_residual_gap_lines,
        )
        for bundle in baseline_bundles
    )
    compiler_rows: list[dict[str, JsonValue]] = []
    for compiled in compiled_baselines:
        source_lines = len(compiled.workspace.original_text.splitlines())
        residual_lines = sum(span.endLine - span.startLine + 1 for span in compiled.spans)
        deterministic = sum(
            item.state in {"deterministic_applied", "preserved_by_policy"}
            for item in compiled.work_items
        )
        residual = sum(item.state == "agent_residual" for item in compiled.work_items)
        blocked = sum(item.state == "blocked_unlocated" for item in compiled.work_items)
        compiler_rows.append(
            {
                "caseNumber": compiled.bundle.result.caseNumber,
                "documentId": compiled.bundle.result.documentId,
                "sourceLines": source_lines,
                "residualLines": residual_lines,
                "residualLineFraction": residual_lines / source_lines,
                "residualSpans": len(compiled.spans),
                "workItems": len(compiled.work_items),
                "deterministicWorkItems": deterministic,
                "deterministicFraction": deterministic / len(compiled.work_items),
                "residualWorkItems": residual,
                "blockedWorkItems": blocked,
                "deterministicPrefills": len(compiled.workspace.deterministic_prefills),
                "compilerEdits": len(compiled.compiler_edits),
            }
        )
    selected_compiled: list[_CompiledCase] = []
    for case in config.cases:
        bundle = baseline_by_id.get(case.document_id)
        if bundle is None:
            raise ValueError(f"hybrid proof case is absent from baseline run: {case.document_id}")
        selected_compiled.append(
            compile_case(
                bundle,
                context_lines=config.workflow.context_lines_per_residual,
                merge_gap_lines=config.workflow.merge_residual_gap_lines,
            )
        )

    transaction = {
        "schemaVersion": 1,
        "runId": config.run.run_id,
        "configSha256": sha256_file(config_path),
        "baselineCommitSha256": config.inputs.baseline_atomic_run.commit_sha256,
        "baselineTransactionSha256": config.inputs.baseline_atomic_run.transaction_sha256,
        "sourceCorpusSha256": config.inputs.source_corpus.sha256,
        "syntheticTargetsSha256": config.inputs.synthetic_targets.sha256,
        "editorPromptSha256": config.prompts.editor.sha256,
        "reviewerPromptSha256": config.prompts.reviewer.sha256,
        "implementationSha256": sha256_file(_IMPLEMENTATION_PATH),
        "cases": [case.document_id for case in config.cases],
        "runtime": {
            "pydanticAiVersion": version("pydantic-ai-slim"),
            "openaiVersion": version("openai"),
            "editorModel": config.providers.editor.model,
            "reviewerModel": config.providers.reviewer.model,
        },
    }
    staged = StagedArtifactRun(
        output_parent=resolve_config_path(project_root, config.run.output_dir),
        run_name=config.run.run_id,
        transaction_sha256=sha256_bytes(canonical_json_bytes(transaction)),
    )
    if staged.completed:
        return cast(
            dict[str, JsonValue],
            json.loads(read_regular_file_bytes(staged.final_root / "summary.json")),
        )
    staged.recover_interrupted_temporary_files()
    staged.publish_bytes("config.yaml", read_regular_file_bytes(config_path))
    staged.publish_bytes("prompts/editor.md", editor_prompt_bytes)
    staged.publish_bytes("prompts/reviewer.md", reviewer_prompt_bytes)
    staged.publish_json("analysis/literal-coverage.json", coverage)
    _publish_jsonl(staged, "analysis/document-coverage.jsonl", coverage_rows)
    _publish_jsonl(staged, "analysis/compiler-cases.jsonl", compiler_rows)

    results: tuple[HybridCaseResult, ...] = ()
    stage_records: dict[str, tuple[HybridModelStage, ...]] = {}
    setup_error: str | None = None

    async def execute() -> tuple[HybridCaseResult, ...]:
        nonlocal setup_error
        editor_provider = config.providers.editor
        reviewer_provider = config.providers.reviewer
        try:
            key = load_provider_key(
                project_root, config.environment_file, editor_provider.api_key_env
            )
            client = AsyncOpenAI(
                api_key=key,
                base_url=(_OPENROUTER_BASE_URL if editor_provider.kind == "openrouter" else None),
                max_retries=max(
                    editor_provider.transport_max_retries,
                    reviewer_provider.transport_max_retries,
                ),
                timeout=max(
                    editor_provider.request_timeout_seconds,
                    reviewer_provider.request_timeout_seconds,
                ),
                default_headers=(
                    {"X-Title": "DocumentParsing"} if editor_provider.kind == "openrouter" else None
                ),
            )
            editor_model, reviewer_model = _model_pair(
                client=client,
                editor_provider=editor_provider,
                reviewer_provider=reviewer_provider,
            )
        except Exception as error:
            setup_error = f"{type(error).__name__}: {error}"
            return tuple(
                HybridCaseResult(
                    documentId=compiled.bundle.result.documentId,
                    status="call_failed",
                    reason="Provider setup failed after the zero-API compiler audit.",
                    sourceLines=len(compiled.workspace.original_text.splitlines()),
                    residualLines=sum(span.endLine - span.startLine + 1 for span in compiled.spans),
                    residualSpans=len(compiled.spans),
                    workItems=len(compiled.work_items),
                    deterministicWorkItems=sum(
                        item.state in {"deterministic_applied", "preserved_by_policy"}
                        for item in compiled.work_items
                    ),
                    residualWorkItems=sum(
                        item.state == "agent_residual" for item in compiled.work_items
                    ),
                    blockedWorkItems=sum(
                        item.state == "blocked_unlocated" for item in compiled.work_items
                    ),
                    deterministicPrefills=len(compiled.workspace.deterministic_prefills),
                    compilerEdits=len(compiled.compiler_edits),
                    outputTextSha256=sha256_bytes(compiled.workspace.current_text.encode("utf-8")),
                    baselineOutputTextSha256=sha256_bytes(
                        compiled.bundle.outputText.encode("utf-8")
                    ),
                    exactBaselineMatch=False,
                    baselineReferenceGapLines=_baseline_reference_gap_lines(
                        compiled.workspace.original_text,
                        compiled.workspace.current_text,
                        compiled.bundle.outputText,
                    ),
                    deterministicAudit=None,
                    review=None,
                    usage=_empty_usage(),
                )
                for compiled in selected_compiled
            )
        try:
            output: list[HybridCaseResult] = []
            for compiled in selected_compiled:
                result, stages = await _run_model_case(
                    compiled=compiled,
                    editor_model=editor_model,
                    reviewer_model=reviewer_model,
                    providers=config.providers,
                    editor_prompt=editor_prompt,
                    editor_prompt_sha256=config.prompts.editor.sha256,
                    reviewer_prompt=reviewer_prompt,
                    reviewer_prompt_sha256=config.prompts.reviewer.sha256,
                    audited_reference_text=compiled.bundle.outputText,
                )
                output.append(result)
                stage_records[result.documentId] = stages
            return tuple(output)
        finally:
            await client.close()

    results = asyncio.run(execute())
    for compiled, result in zip(selected_compiled, results, strict=True):
        prefix = f"cases/{result.documentId}"
        staged.publish_bytes(
            f"{prefix}/source.txt", compiled.workspace.original_text.encode("utf-8")
        )
        # Reconstruct the deterministic intermediate independently because the live workspace may
        # now contain model edits.
        deterministic_copy = compile_case(
            compiled.bundle,
            context_lines=config.workflow.context_lines_per_residual,
            merge_gap_lines=config.workflow.merge_residual_gap_lines,
        )
        staged.publish_bytes(
            f"{prefix}/after-deterministic.txt",
            deterministic_copy.workspace.current_text.encode("utf-8"),
        )
        staged.publish_json(
            f"{prefix}/work-items.json",
            [item.model_dump(mode="json") for item in deterministic_copy.work_items],
        )
        staged.publish_json(
            f"{prefix}/residual-spans.json",
            [span.model_dump(mode="json") for span in deterministic_copy.spans],
        )
        staged.publish_json(
            f"{prefix}/compiler-edits.json",
            [row.model_dump(mode="json") for row in deterministic_copy.compiler_edits],
        )
        staged.publish_json(f"{prefix}/editor-payload.json", _editor_payload(deterministic_copy))
        staged.publish_bytes(f"{prefix}/final.txt", compiled.workspace.current_text.encode("utf-8"))
        staged.publish_bytes(
            f"{prefix}/baseline-reference.txt", compiled.bundle.outputText.encode("utf-8")
        )
        staged.publish_bytes(
            f"{prefix}/diff.patch",
            unified_text_diff(
                compiled.workspace.original_text, compiled.workspace.current_text
            ).encode("utf-8"),
        )
        staged.publish_json(
            f"{prefix}/stages.json",
            [row.model_dump(mode="json") for row in stage_records.get(result.documentId, ())],
        )
        staged.publish_json(f"{prefix}/result.json", result.model_dump(mode="json"))

    total_usage = (
        _combine_usage([result.usage for result in results]) if results else _empty_usage()
    )
    summary: dict[str, JsonValue] = {
        "schemaVersion": 1,
        "runId": config.run.run_id,
        "status": "complete",
        "auditDocuments": coverage["documents"],
        "changedTextLeaves": coverage["changedTextLeaves"],
        "exactLiteralCandidates": coverage["exactLiteralCandidates"],
        "exactLiteralCandidateFraction": coverage["exactLiteralCandidateFraction"],
        "compilerCases": len(compiler_rows),
        "modelCases": len(results),
        "qualityValidatedCases": sum(result.status == "quality_validated" for result in results),
        "needsReviewCases": sum(result.status == "needs_review" for result in results),
        "callFailedCases": sum(result.status == "call_failed" for result in results),
        "compilerBlockedCases": sum(result.status == "compiler_blocked" for result in results),
        "providerSetupError": setup_error,
        "requests": total_usage.requests,
        "inputTokens": total_usage.inputTokens,
        "cacheReadTokens": total_usage.cacheReadTokens,
        "outputTokens": total_usage.outputTokens,
        "reasoningTokens": total_usage.reasoningTokens,
        "visibleOutputTokens": total_usage.visibleOutputTokens,
        "estimatedCostUsd": str(total_usage.estimatedCostUsd),
        "providerReportedCostUsd": (
            str(total_usage.providerReportedCostUsd)
            if total_usage.providerReportedCostUsd is not None
            else None
        ),
        "downstreamProviders": cast(JsonValue, list(total_usage.downstreamProviders)),
        "trainingRecordsPublished": False,
    }
    staged.publish_json("summary.json", summary)
    report = _report(
        summary=summary,
        coverage=coverage,
        compiler_rows=compiler_rows,
        results=results,
        baselines=baseline_by_id,
    )
    staged.publish_bytes("REPORT.md", report.encode("utf-8"))
    for name, payload in _plot_bytes(coverage, compiler_rows, results, baseline_by_id).items():
        staged.publish_bytes(f"plots/{name}", payload)
    artifacts = _artifact_inventory(staged.stage_root)
    staged.commit(
        expected_artifacts=artifacts,
        metadata={
            "schemaVersion": 1,
            "auditDocuments": cast(int, coverage["documents"]),
            "modelCases": len(results),
            "qualityValidatedCases": cast(int, summary["qualityValidatedCases"]),
            "trainingRecordsPublished": False,
        },
    )
    output = dict(summary)
    output["artifactRoot"] = str(staged.final_root)
    output["commitSha256"] = sha256_file(staged.final_root / "_COMMIT.json")
    return output
