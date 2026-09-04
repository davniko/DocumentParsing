"""Tool-mediated raw-OCR rewriting probe for synthetic Bill of Lading labels.

Each case owns an isolated mutable workspace.  The model can only change that
workspace through exact contextual edits, and a deterministic output validator
refuses completion until every printable changed label leaf is covered and the
final diff has been inspected.  This is deliberately an audited probe: it
publishes no training records.
"""

from __future__ import annotations

import asyncio
import difflib
import json
import re
import resource
import time
from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from datetime import UTC, datetime
from decimal import Decimal
from importlib.metadata import version
from itertools import pairwise
from pathlib import Path
from typing import Annotated, Any, Literal, cast

from openai import AsyncOpenAI
from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    JsonValue,
    StringConstraints,
    model_validator,
)
from pydantic_ai import (
    Agent,
    ModelRetry,
    NativeOutput,
    RunContext,
    Tool,
    capture_run_messages,
)
from pydantic_ai.concurrency import ConcurrencyLimiter
from pydantic_ai.messages import ModelResponse
from pydantic_ai.models.openai import OpenAIResponsesModel
from pydantic_ai.providers.openai import OpenAIProvider
from pydantic_ai.usage import UsageLimits

from document_ocr.atomic import json_artifact_bytes, read_regular_file_bytes
from document_ocr.hashing import canonical_json_bytes, sha256_bytes, sha256_file
from document_ocr.label_schemas import bill_of_lading_v3, bill_of_lading_v5
from document_ocr.synthesis.config import (
    SynthesisRawTextRewriteCycleProbeConfig,
    SynthesisRawTextRewriteProbeConfig,
)
from document_ocr.synthesis.linguistic_probe_runtime import (
    LinguisticUsageReceipt,
    load_openai_key,
    model_messages,
    openai_responses_settings,
    usage_receipt,
)
from document_ocr.synthesis.run_safety import StagedArtifactRun
from document_ocr.synthesis.task_adapter import (
    BILL_OF_LADING_TASK_ADAPTER,
    BILL_OF_LADING_V5_TASK_ADAPTER,
)
from document_ocr.training.config import resolve_config_path

NonEmptyText = Annotated[str, StringConstraints(strip_whitespace=True, min_length=1)]
Sha256 = Annotated[str, StringConstraints(pattern=r"^[0-9a-f]{64}$")]
_STRICT = ConfigDict(extra="forbid", frozen=True, strict=True, allow_inf_nan=False)
_IMPLEMENTATION_PATH = Path(__file__).resolve(strict=True)
_NEWLINE = re.compile(r"\r\n|\r|\n")
_PAGE_MARKER = re.compile(r"^--- PAGE [1-9][0-9]* ---[ \t]*$")
_RELATION_ONLY_NAMES = frozenset({"coverage", "groupId", "packageId", "packageIds"})


class RawTextRewriteProbeError(RuntimeError):
    """The probe could not produce a complete, auditable result set."""


class ChangedLeaf(BaseModel):
    model_config = _STRICT

    path: NonEmptyText
    sourcePresent: bool
    targetPresent: bool
    sourceValue: JsonValue
    targetValue: JsonValue
    evidenceClass: Literal["printed_fact", "relation_metadata", "schema_metadata"]
    requiresTextEdit: bool


class ExactTextEdit(BaseModel):
    """One exact contextual replacement requested by the model."""

    model_config = _STRICT

    oldText: Annotated[
        str,
        StringConstraints(min_length=1, max_length=50_000),
        Field(description="Exact text currently present in the mutable OCR workspace."),
    ]
    newText: Annotated[
        str,
        StringConstraints(max_length=50_000),
        Field(description="Replacement text with the same newline and edge-whitespace shape."),
    ]
    leftContext: Annotated[
        str,
        StringConstraints(max_length=2_000),
        Field(description="Literal text immediately before oldText, or empty when unique."),
    ]
    rightContext: Annotated[
        str,
        StringConstraints(max_length=2_000),
        Field(description="Literal text immediately after oldText, or empty when unique."),
    ]
    expectedMatches: Annotated[
        int,
        Field(ge=1, le=1_000, description="Exact contextual match count required atomically."),
    ]
    applyToAllMatches: Annotated[
        bool,
        Field(description="True only when every contextual occurrence has the same meaning."),
    ]
    category: Annotated[
        Literal["target_field", "auxiliary_personal_data"],
        Field(description="Whether this realizes target truth or anonymizes excluded flavor."),
    ]
    targetPaths: Annotated[
        tuple[NonEmptyText, ...],
        Field(
            description=(
                "Exact requiresTextEdit label paths realized by a target-field edit; always [] "
                "for auxiliary-personal-data edits."
            )
        ),
    ]
    auxiliaryKind: Annotated[
        NonEmptyText | None,
        Field(
            description=(
                "Specific flavor-data kind for auxiliary edits; null for target-field edits."
            )
        ),
    ]

    @model_validator(mode="after")
    def category_contract(self) -> ExactTextEdit:
        if self.oldText == self.newText:
            raise ValueError("an exact text edit must change text")
        if self.applyToAllMatches is False and self.expectedMatches != 1:
            raise ValueError("a single contextual edit must expect exactly one match")
        if len(self.targetPaths) != len(set(self.targetPaths)):
            raise ValueError("targetPaths must not contain duplicates")
        if self.category == "target_field":
            if not self.targetPaths or self.auxiliaryKind is not None:
                raise ValueError("target-field edits require paths and no auxiliary kind")
        else:
            if self.targetPaths:
                raise ValueError("auxiliary-personal-data edits must set targetPaths to []")
            if self.auxiliaryKind is None or not self.newText:
                raise ValueError(
                    "auxiliary-personal-data edits require a non-empty replacement and kind"
                )
        return self


class AppliedTextEdit(BaseModel):
    model_config = _STRICT

    batch: Annotated[int, Field(ge=1)]
    category: Literal["target_field", "auxiliary_personal_data"]
    targetPaths: tuple[NonEmptyText, ...]
    auxiliaryKind: NonEmptyText | None
    oldText: str
    newText: str
    leftContext: str
    rightContext: str
    matchedOccurrences: Annotated[int, Field(ge=1)]
    originalOffsets: tuple[Annotated[int, Field(ge=0)], ...]


class RewriteCompletionReceipt(BaseModel):
    """Small provider-constrained receipt; detailed facts come from tool state."""

    model_config = _STRICT

    status: Literal["complete", "blocked"]
    currentTextSha256: Sha256
    unresolvedChangedPaths: tuple[NonEmptyText, ...]
    summary: NonEmptyText

    @model_validator(mode="after")
    def status_matches_unresolved(self) -> RewriteCompletionReceipt:
        if self.status == "complete" and self.unresolvedChangedPaths:
            raise ValueError("complete status cannot carry unresolved paths")
        if self.status == "blocked" and not self.unresolvedChangedPaths:
            raise ValueError("blocked status requires at least one unresolved path")
        if len(self.unresolvedChangedPaths) != len(set(self.unresolvedChangedPaths)):
            raise ValueError("unresolved paths must be unique")
        return self


class RewriteCaseMetrics(BaseModel):
    model_config = _STRICT

    sourceCharacters: Annotated[int, Field(ge=0)]
    outputCharacters: Annotated[int, Field(ge=0)]
    lengthDelta: int
    sourceLines: Annotated[int, Field(ge=0)]
    outputLines: Annotated[int, Field(ge=0)]
    editBatches: Annotated[int, Field(ge=0)]
    editSpecifications: Annotated[int, Field(ge=0)]
    replacedOccurrences: Annotated[int, Field(ge=0)]
    sourceCharactersSelectedForReplacement: Annotated[int, Field(ge=0)]
    requiredChangedPaths: Annotated[int, Field(ge=0)]
    resolvedChangedPaths: Annotated[int, Field(ge=0)]
    auxiliaryAnonymizations: Annotated[int, Field(ge=0)]
    matchingBlockSourceFraction: Annotated[float, Field(ge=0, le=1)]
    durationMs: Annotated[float, Field(ge=0)]


class RewriteCaseResult(BaseModel):
    model_config = _STRICT

    schemaVersion: Literal[1]
    caseNumber: Annotated[int, Field(ge=1, le=10)]
    documentId: NonEmptyText
    scenarioId: NonEmptyText
    status: Literal["complete", "blocked", "call_failed"]
    sourceTextSha256: Sha256
    outputTextSha256: Sha256
    sourceLabelSha256: Sha256
    targetLabelSha256: Sha256
    changedLeavesSha256: Sha256
    promptPayloadSha256: Sha256
    outputSchemaSha256: Sha256
    startedAt: datetime
    completedAt: datetime
    finalReceipt: RewriteCompletionReceipt | None
    checks: dict[str, bool]
    metrics: RewriteCaseMetrics
    usage: LinguisticUsageReceipt
    errorType: NonEmptyText | None
    errorMessage: NonEmptyText | None


class RewriteCaseBundle(BaseModel):
    """Atomic resume boundary containing every model- or user-visible case byte."""

    model_config = _STRICT

    schemaVersion: Literal[1]
    result: RewriteCaseResult
    feature: dict[str, JsonValue]
    sourceLabel: dict[str, JsonValue]
    targetLabel: dict[str, JsonValue]
    changedLeaves: tuple[ChangedLeaf, ...]
    sourceText: str
    outputText: str
    appliedEdits: tuple[AppliedTextEdit, ...]
    diff: str
    promptPayload: dict[str, JsonValue]
    modelMessages: JsonValue


@dataclass(slots=True)
class RewriteWorkspace:
    document_id: str
    original_text: str
    current_text: str
    required_paths: frozenset[str]
    applied_edits: list[AppliedTextEdit] = field(default_factory=list)
    batch_count: int = 0
    inspection_count: int = 0
    inspected_sha256: str | None = None

    @property
    def current_sha256(self) -> str:
        return sha256_bytes(self.current_text.encode("utf-8"))

    @property
    def resolved_paths(self) -> frozenset[str]:
        return frozenset(
            path
            for edit in self.applied_edits
            if edit.category == "target_field"
            for path in edit.targetPaths
        )


def _newline_sequence(value: str) -> tuple[str, ...]:
    return tuple(match.group(0) for match in _NEWLINE.finditer(value))


def _page_markers(value: str) -> tuple[str, ...]:
    return tuple(line for line in value.splitlines() if _PAGE_MARKER.fullmatch(line))


def _edge_whitespace(value: str) -> tuple[str, str]:
    leading = re.match(r"^[ \t]*", value)
    trailing = re.search(r"[ \t]*$", value)
    return (
        leading.group(0) if leading is not None else "",
        trailing.group(0) if trailing is not None else "",
    )


def _all_occurrences(text: str, needle: str) -> tuple[int, ...]:
    offsets: list[int] = []
    cursor = 0
    while True:
        offset = text.find(needle, cursor)
        if offset < 0:
            return tuple(offsets)
        offsets.append(offset)
        cursor = offset + len(needle)


def _contextual_offsets(text: str, edit: ExactTextEdit) -> tuple[int, ...]:
    selected: list[int] = []
    for offset in _all_occurrences(text, edit.oldText):
        end = offset + len(edit.oldText)
        if (
            edit.leftContext
            and text[max(0, offset - len(edit.leftContext)) : offset] != edit.leftContext
        ):
            continue
        if edit.rightContext and text[end : end + len(edit.rightContext)] != edit.rightContext:
            continue
        selected.append(offset)
    return tuple(selected)


def _occurrence_diagnostic(text: str, needle: str) -> str:
    rows = []
    for offset in _all_occurrences(text, needle)[:5]:
        end = offset + len(needle)
        rows.append(
            {
                "line": text.count("\n", 0, offset) + 1,
                "immediateLeft": text[max(0, offset - 48) : offset],
                "immediateRight": text[end : end + 48],
            }
        )
    return json.dumps(rows, ensure_ascii=False, separators=(",", ":"))


def apply_edit_batch(
    workspace: RewriteWorkspace, edits: Sequence[ExactTextEdit]
) -> tuple[AppliedTextEdit, ...]:
    """Validate a batch against one pre-edit snapshot, then commit it atomically."""

    if not edits:
        raise ValueError("an edit batch must not be empty")
    snapshot = workspace.current_text
    operations: list[tuple[int, int, ExactTextEdit]] = []
    selected_by_edit: list[tuple[ExactTextEdit, tuple[int, ...]]] = []
    errors: list[str] = []
    for index, edit in enumerate(edits):
        prefix = f"edit[{index}] oldText={edit.oldText[:80]!r}"
        unknown = sorted(set(edit.targetPaths) - workspace.required_paths)
        if unknown:
            errors.append(f"{prefix}: unknown or non-printable target paths {unknown}")
            continue
        if _newline_sequence(edit.oldText) != _newline_sequence(edit.newText):
            errors.append(f"{prefix}: replacement changes the source newline sequence")
            continue
        if _edge_whitespace(edit.oldText) != _edge_whitespace(edit.newText):
            errors.append(
                f"{prefix}: replacement changes leading or trailing horizontal whitespace"
            )
            continue
        offsets = _contextual_offsets(snapshot, edit)
        if len(offsets) != edit.expectedMatches:
            global_count = len(_all_occurrences(snapshot, edit.oldText))
            context_diagnostic = _occurrence_diagnostic(snapshot, edit.oldText)
            errors.append(
                f"{prefix}: contextual match count is {len(offsets)}, expected "
                f"{edit.expectedMatches}; global match count is {global_count}; exact adjacent "
                f"contexts for up to five matches are {context_diagnostic}"
            )
            continue
        if not edit.applyToAllMatches and len(offsets) != 1:
            errors.append(f"{prefix}: non-global contextual edit is not uniquely matched")
            continue
        selected_by_edit.append((edit, offsets))
        operations.extend((offset, offset + len(edit.oldText), edit) for offset in offsets)
    if errors:
        raise ValueError("; ".join(errors))
    ordered = sorted(operations, key=lambda row: (row[0], row[1]))
    for previous, current in pairwise(ordered):
        if current[0] < previous[1]:
            raise ValueError("an edit batch contains overlapping replacements")
    updated = snapshot
    for start, end, edit in reversed(ordered):
        updated = updated[:start] + edit.newText + updated[end:]
    if _newline_sequence(updated) != _newline_sequence(workspace.original_text):
        raise ValueError("the updated document no longer preserves the original newline sequence")
    if _page_markers(updated) != _page_markers(workspace.original_text):
        raise ValueError("the updated document changes page markers or page order")
    batch = workspace.batch_count + 1
    receipts = tuple(
        AppliedTextEdit(
            batch=batch,
            category=edit.category,
            targetPaths=edit.targetPaths,
            auxiliaryKind=edit.auxiliaryKind,
            oldText=edit.oldText,
            newText=edit.newText,
            leftContext=edit.leftContext,
            rightContext=edit.rightContext,
            matchedOccurrences=len(offsets),
            originalOffsets=offsets,
        )
        for edit, offsets in selected_by_edit
    )
    workspace.current_text = updated
    workspace.batch_count = batch
    workspace.applied_edits.extend(receipts)
    workspace.inspected_sha256 = None
    return receipts


def unified_text_diff(source: str, target: str) -> str:
    lines = difflib.unified_diff(
        source.splitlines(keepends=True),
        target.splitlines(keepends=True),
        fromfile="raw-before.txt",
        tofile="raw-after.txt",
    )
    return "".join(lines)


def apply_text_edits(
    ctx: RunContext[RewriteWorkspace], edits: list[ExactTextEdit]
) -> dict[str, JsonValue]:
    """Apply one atomic batch of exact contextual replacements to the OCR text.

    Args:
        ctx: The isolated rewrite workspace.
        edits: Exact old/new values, immediate contexts, match counts, and label paths.
    """

    try:
        receipts = apply_edit_batch(ctx.deps, edits)
    except ValueError as error:
        raise ModelRetry(f"Exact edit rejected: {error}") from error
    return {
        "status": "applied",
        "batch": ctx.deps.batch_count,
        "editSpecifications": len(receipts),
        "replacedOccurrences": sum(row.matchedOccurrences for row in receipts),
        "currentTextSha256": ctx.deps.current_sha256,
        "remainingChangedPaths": cast(
            list[JsonValue], sorted(ctx.deps.required_paths - ctx.deps.resolved_paths)
        ),
    }


def inspect_current_diff(ctx: RunContext[RewriteWorkspace]) -> dict[str, JsonValue]:
    """Inspect the complete current unified diff after all intended edits."""

    workspace = ctx.deps
    difference = unified_text_diff(workspace.original_text, workspace.current_text)
    workspace.inspection_count += 1
    workspace.inspected_sha256 = workspace.current_sha256
    return {
        "status": "inspected",
        "currentTextSha256": workspace.current_sha256,
        "editBatches": workspace.batch_count,
        "remainingChangedPaths": cast(
            list[JsonValue], sorted(workspace.required_paths - workspace.resolved_paths)
        ),
        "unifiedDiff": difference,
    }


def validate_completion_receipt(
    workspace: RewriteWorkspace, receipt: RewriteCompletionReceipt
) -> RewriteCompletionReceipt:
    if receipt.currentTextSha256 != workspace.current_sha256:
        raise ValueError("receipt SHA-256 differs from the current tool workspace")
    if not workspace.applied_edits:
        raise ValueError("apply_text_edits was not used")
    if workspace.inspection_count == 0 or workspace.inspected_sha256 != workspace.current_sha256:
        raise ValueError("inspect_current_diff was not called on the final workspace state")
    unknown = workspace.resolved_paths - workspace.required_paths
    if unknown:
        raise ValueError(f"edits claim unknown changed paths: {sorted(unknown)}")
    missing = workspace.required_paths - workspace.resolved_paths
    if receipt.status == "complete":
        if missing:
            raise ValueError(f"changed paths remain unresolved: {sorted(missing)}")
    elif frozenset(receipt.unresolvedChangedPaths) != missing:
        raise ValueError("blocked receipt must name exactly the unresolved changed paths")
    return receipt


def _json_leaf_changes(
    source: JsonValue,
    target: JsonValue,
    *,
    path: str = "",
    source_present: bool = True,
    target_present: bool = True,
) -> list[ChangedLeaf]:
    if isinstance(source, dict) and isinstance(target, dict):
        rows: list[ChangedLeaf] = []
        for key in sorted(set(source) | set(target)):
            child_path = f"{path}.{key}" if path else key
            if key not in source:
                rows.extend(
                    _json_leaf_changes(
                        None,
                        target[key],
                        path=child_path,
                        source_present=False,
                    )
                )
            elif key not in target:
                rows.extend(
                    _json_leaf_changes(
                        source[key],
                        None,
                        path=child_path,
                        target_present=False,
                    )
                )
            else:
                rows.extend(
                    _json_leaf_changes(
                        source[key],
                        target[key],
                        path=child_path,
                    )
                )
        return rows
    if isinstance(source, list) and isinstance(target, list):
        rows = []
        for index in range(max(len(source), len(target))):
            child_path = f"{path}[{index}]"
            if index >= len(source):
                rows.extend(
                    _json_leaf_changes(
                        None,
                        target[index],
                        path=child_path,
                        source_present=False,
                    )
                )
            elif index >= len(target):
                rows.extend(
                    _json_leaf_changes(
                        source[index],
                        None,
                        path=child_path,
                        target_present=False,
                    )
                )
            else:
                rows.extend(
                    _json_leaf_changes(
                        source[index],
                        target[index],
                        path=child_path,
                    )
                )
        return rows
    if source_present and target_present and source == target:
        return []
    leaf_name = re.split(r"\.|\[", path)[-1].rstrip("]")
    if path == "schemaVersion":
        evidence_class: Literal["printed_fact", "relation_metadata", "schema_metadata"] = (
            "schema_metadata"
        )
    elif leaf_name in _RELATION_ONLY_NAMES:
        evidence_class = "relation_metadata"
    else:
        evidence_class = "printed_fact"
    return [
        ChangedLeaf(
            path=path,
            sourcePresent=source_present,
            targetPresent=target_present,
            sourceValue=source,
            targetValue=target,
            evidenceClass=evidence_class,
            requiresTextEdit=evidence_class == "printed_fact",
        )
    ]


def changed_leaves(
    source: Mapping[str, JsonValue], target: Mapping[str, JsonValue]
) -> tuple[ChangedLeaf, ...]:
    return tuple(_json_leaf_changes(cast(JsonValue, dict(source)), cast(JsonValue, dict(target))))


def _resolve_pinned_file(
    project_root: Path, configured_path: str, expected_sha256: str, *, label: str
) -> Path:
    path = resolve_config_path(project_root, configured_path)
    if path.is_symlink() or not path.is_file() or sha256_file(path) != expected_sha256:
        raise ValueError(f"{label} differs from its configured pin: {path}")
    return path.resolve(strict=True)


def _validate_linguistic_run(
    project_root: Path,
    config: SynthesisRawTextRewriteProbeConfig | SynthesisRawTextRewriteCycleProbeConfig,
) -> Path:
    configured = config.inputs.linguistic_completion_run
    root = resolve_config_path(project_root, configured.path)
    commit = root / "_COMMIT.json"
    if root.is_symlink() or not root.is_dir() or sha256_file(commit) != configured.commit_sha256:
        raise ValueError(f"linguistic completion run differs from its configured pin: {root}")
    StagedArtifactRun(
        output_parent=root.parent,
        run_name=root.name,
        transaction_sha256=configured.transaction_sha256,
    ).validate_committed_run()
    return root.resolve(strict=True)


def _load_jsonl(path: Path, *, records: int, key: str, label: str) -> tuple[dict[str, Any], ...]:
    rows: list[dict[str, Any]] = []
    identifiers: set[str] = set()
    with path.open("rb") as stream:
        for line_number, raw in enumerate(stream, start=1):
            if not raw.strip() or not raw.endswith(b"\n"):
                raise ValueError(f"{label} row {line_number} is blank or unterminated")
            try:
                value = json.loads(raw)
            except (UnicodeDecodeError, json.JSONDecodeError) as error:
                raise ValueError(f"{label} row {line_number} is invalid JSON") from error
            if not isinstance(value, dict) or not isinstance(value.get(key), str):
                raise ValueError(f"{label} row {line_number} lacks string key {key!r}")
            identifier = cast(str, value[key])
            if identifier in identifiers:
                raise ValueError(f"{label} repeats {identifier}")
            identifiers.add(identifier)
            rows.append(value)
    if len(rows) != records:
        raise ValueError(f"{label} count differs from configuration")
    return tuple(rows)


def _matching_fraction(source: str, target: str) -> float:
    if not source:
        return 1.0 if not target else 0.0
    matcher = difflib.SequenceMatcher(a=source, b=target, autojunk=False)
    return min(1.0, sum(row.size for row in matcher.get_matching_blocks()) / len(source))


def _case_metrics(
    workspace: RewriteWorkspace, *, duration_ms: float, required_paths: int
) -> RewriteCaseMetrics:
    source = workspace.original_text
    output = workspace.current_text
    return RewriteCaseMetrics(
        sourceCharacters=len(source),
        outputCharacters=len(output),
        lengthDelta=len(output) - len(source),
        sourceLines=len(source.splitlines()),
        outputLines=len(output.splitlines()),
        editBatches=workspace.batch_count,
        editSpecifications=len(workspace.applied_edits),
        replacedOccurrences=sum(row.matchedOccurrences for row in workspace.applied_edits),
        sourceCharactersSelectedForReplacement=sum(
            len(row.oldText) * row.matchedOccurrences for row in workspace.applied_edits
        ),
        requiredChangedPaths=required_paths,
        resolvedChangedPaths=len(workspace.resolved_paths),
        auxiliaryAnonymizations=sum(
            row.matchedOccurrences
            for row in workspace.applied_edits
            if row.category == "auxiliary_personal_data"
        ),
        matchingBlockSourceFraction=_matching_fraction(source, output),
        durationMs=duration_ms,
    )


def _case_checks(
    workspace: RewriteWorkspace,
    *,
    status: Literal["complete", "blocked", "call_failed"],
    receipt: RewriteCompletionReceipt | None,
) -> dict[str, bool]:
    required_covered = workspace.required_paths <= workspace.resolved_paths
    auxiliary_valid = all(
        row.category != "auxiliary_personal_data"
        or (
            bool(row.newText)
            and row.auxiliaryKind is not None
            and _newline_sequence(row.oldText) == _newline_sequence(row.newText)
            and _edge_whitespace(row.oldText) == _edge_whitespace(row.newText)
        )
        for row in workspace.applied_edits
    )
    return {
        "editToolUsed": bool(workspace.applied_edits),
        "diffInspectedAtFinalSha": (
            workspace.inspection_count > 0
            and workspace.inspected_sha256 == workspace.current_sha256
        ),
        "receiptShaMatches": (
            receipt is not None and receipt.currentTextSha256 == workspace.current_sha256
        ),
        "pageMarkersPreserved": _page_markers(workspace.current_text)
        == _page_markers(workspace.original_text),
        "newlineSequencePreserved": _newline_sequence(workspace.current_text)
        == _newline_sequence(workspace.original_text),
        "requiredPathCoverage": required_covered,
        "unknownPathFree": workspace.resolved_paths <= workspace.required_paths,
        "auxiliaryReplacementShapeValid": auxiliary_valid,
        "completeReceiptValidated": status == "complete" and receipt is not None,
    }


async def _run_case(
    *,
    case_number: int,
    source_row: Mapping[str, Any],
    target_row: Mapping[str, Any],
    feature: Mapping[str, JsonValue],
    system_prompt: str,
    model: OpenAIResponsesModel,
    limiter: ConcurrencyLimiter,
    config: SynthesisRawTextRewriteProbeConfig,
) -> RewriteCaseBundle:
    document_id = cast(str, source_row["documentId"])
    source_text = cast(str, source_row["joinedRawText"])
    source_label = BILL_OF_LADING_TASK_ADAPTER.validate_target(
        document_id=document_id,
        target=cast(Mapping[str, Any], source_row[config.inputs.source_target_field]),
    )
    target_label = BILL_OF_LADING_V5_TASK_ADAPTER.validate_target(
        document_id=cast(str, target_row["scenarioId"]),
        target=cast(Mapping[str, Any], target_row["target"]),
    )
    leaves = changed_leaves(
        cast(Mapping[str, JsonValue], source_label),
        cast(Mapping[str, JsonValue], target_label),
    )
    required_paths = frozenset(row.path for row in leaves if row.requiresTextEdit)
    if not required_paths:
        raise ValueError(f"rewrite case has no printable target changes: {document_id}")
    workspace = RewriteWorkspace(
        document_id=document_id,
        original_text=source_text,
        current_text=source_text,
        required_paths=required_paths,
    )
    payload: dict[str, JsonValue] = {
        "documentId": document_id,
        "task": "Rewrite the source OCR so it evidences the synthetic target label.",
        "sourceLabel": cast(JsonValue, source_label),
        "syntheticTargetLabel": cast(JsonValue, target_label),
        "changedLeaves": cast(JsonValue, [row.model_dump(mode="json") for row in leaves]),
        "rawOcrText": source_text,
    }
    user_prompt = json.dumps(payload, ensure_ascii=False, separators=(",", ":"))
    output_schema = RewriteCompletionReceipt.model_json_schema(mode="validation")
    tool_retries = config.workflow.tool_retries
    model_settings = openai_responses_settings(config.provider)
    model_settings["parallel_tool_calls"] = False
    edit_tool = Tool(
        apply_text_edits,
        max_retries=tool_retries,
        strict=True,
        name="apply_text_edits",
    )
    diff_tool = Tool(
        inspect_current_diff,
        max_retries=tool_retries,
        strict=True,
        name="inspect_current_diff",
    )
    agent = Agent[RewriteWorkspace, RewriteCompletionReceipt](
        model,
        deps_type=RewriteWorkspace,
        tools=(edit_tool, diff_tool),
        output_type=NativeOutput(
            RewriteCompletionReceipt,
            name="synthetic_bill_of_lading_raw_text_rewrite_receipt",
            description=(
                "Return the final SHA and blocked-path receipt only after exact edit and diff "
                "tools have established the complete rewritten OCR."
            ),
            strict=True,
        ),
        system_prompt=system_prompt,
        model_settings=model_settings,
        retries=tool_retries,
        max_concurrency=limiter,
        name="synthetic-bill-of-lading-raw-text-rewriter",
    )

    @agent.output_validator
    def validate_output(
        ctx: RunContext[RewriteWorkspace], output: RewriteCompletionReceipt
    ) -> RewriteCompletionReceipt:
        try:
            return validate_completion_receipt(ctx.deps, output)
        except ValueError as error:
            raise ModelRetry(f"Final rewrite receipt rejected: {error}") from error

    started_at = datetime.now(UTC)
    started = time.perf_counter()
    receipt: RewriteCompletionReceipt | None = None
    status: Literal["complete", "blocked", "call_failed"] = "call_failed"
    error_type: str | None = None
    error_message: str | None = None
    messages: JsonValue = []
    with capture_run_messages() as captured:
        try:
            result = await agent.run(
                user_prompt,
                deps=workspace,
                usage_limits=UsageLimits(
                    request_limit=config.workflow.max_model_requests_per_case,
                    output_tokens_limit=(
                        config.provider.max_output_tokens
                        * config.workflow.max_model_requests_per_case
                    ),
                ),
            )
            receipt = result.output
            status = receipt.status
        except Exception as error:
            error_type = type(error).__name__
            error_message = str(error)
        finally:
            messages = model_messages(captured)
    responses = tuple(message for message in captured if isinstance(message, ModelResponse))
    duration_ms = (time.perf_counter() - started) * 1000.0
    checks = _case_checks(workspace, status=status, receipt=receipt)
    if status == "complete" and not all(checks.values()):
        raise RuntimeError(f"validated complete rewrite has failing checks: {document_id}")
    source_label_json = cast(dict[str, JsonValue], source_label)
    target_label_json = cast(dict[str, JsonValue], target_label)
    changed_payload = [row.model_dump(mode="json") for row in leaves]
    result_record = RewriteCaseResult(
        schemaVersion=1,
        caseNumber=case_number,
        documentId=document_id,
        scenarioId=cast(str, target_row["scenarioId"]),
        status=status,
        sourceTextSha256=sha256_bytes(source_text.encode("utf-8")),
        outputTextSha256=workspace.current_sha256,
        sourceLabelSha256=sha256_bytes(canonical_json_bytes(source_label_json)),
        targetLabelSha256=sha256_bytes(canonical_json_bytes(target_label_json)),
        changedLeavesSha256=sha256_bytes(canonical_json_bytes(changed_payload)),
        promptPayloadSha256=sha256_bytes(user_prompt.encode("utf-8")),
        outputSchemaSha256=sha256_bytes(canonical_json_bytes(output_schema)),
        startedAt=started_at,
        completedAt=datetime.now(UTC),
        finalReceipt=receipt,
        checks=checks,
        metrics=_case_metrics(
            workspace, duration_ms=duration_ms, required_paths=len(required_paths)
        ),
        usage=usage_receipt(responses, config.provider.pricing),
        errorType=error_type,
        errorMessage=error_message,
    )
    return RewriteCaseBundle(
        schemaVersion=1,
        result=result_record,
        feature=dict(feature),
        sourceLabel=source_label_json,
        targetLabel=target_label_json,
        changedLeaves=leaves,
        sourceText=source_text,
        outputText=workspace.current_text,
        appliedEdits=tuple(workspace.applied_edits),
        diff=unified_text_diff(source_text, workspace.current_text),
        promptPayload=payload,
        modelMessages=messages,
    )


def _bundle_path(case_number: int, document_id: str) -> str:
    return f"cases/{case_number:02d}-{document_id}/bundle.json"


def _load_bundle(path: Path) -> RewriteCaseBundle | None:
    if not path.exists():
        return None
    if path.is_symlink() or not path.is_file():
        raise ValueError(f"rewrite bundle resume path is not a regular file: {path}")
    return RewriteCaseBundle.model_validate_json(read_regular_file_bytes(path), strict=True)


async def _run_cases(
    *,
    selected: Sequence[tuple[Mapping[str, Any], Mapping[str, Any], Mapping[str, JsonValue]]],
    system_prompt: str,
    staged: StagedArtifactRun,
    config: SynthesisRawTextRewriteProbeConfig,
    project_root: Path,
) -> tuple[RewriteCaseBundle, ...]:
    client = AsyncOpenAI(
        api_key=load_openai_key(project_root, config.environment_file),
        max_retries=config.provider.transport_max_retries,
        timeout=config.provider.request_timeout_seconds,
    )
    model = OpenAIResponsesModel(
        config.provider.model, provider=OpenAIProvider(openai_client=client)
    )
    limiter = ConcurrencyLimiter(
        config.workflow.max_concurrent_requests,
        name="synthetic-raw-text-rewrite-provider-requests",
    )
    case_limiter = asyncio.Semaphore(config.workflow.max_concurrent_cases)
    progress_lock = asyncio.Lock()
    started = time.perf_counter()
    processed = 0
    results: list[RewriteCaseBundle | None] = [None] * len(selected)

    async def run_one(index: int) -> None:
        nonlocal processed
        source_row, target_row, feature = selected[index]
        document_id = cast(str, source_row["documentId"])
        path = staged.stage_root / _bundle_path(index + 1, document_id)
        existing = _load_bundle(path)
        if existing is not None:
            results[index] = existing
        else:
            async with case_limiter:
                bundle = await _run_case(
                    case_number=index + 1,
                    source_row=source_row,
                    target_row=target_row,
                    feature=feature,
                    system_prompt=system_prompt,
                    model=model,
                    limiter=limiter,
                    config=config,
                )
                staged.publish_json(
                    _bundle_path(index + 1, document_id), bundle.model_dump(mode="json")
                )
                results[index] = bundle
        async with progress_lock:
            processed += 1
            elapsed = time.perf_counter() - started
            rate = processed / elapsed if elapsed else 0.0
            complete = sum(row is not None and row.result.status == "complete" for row in results)
            print(
                json.dumps(
                    {
                        "command": "run-raw-text-rewrite-probe",
                        "phase": "tool_mediated_rewrite",
                        "processed_cases": processed,
                        "remaining_cases": len(selected) - processed,
                        "complete_cases": complete,
                        "elapsed_seconds": round(elapsed, 3),
                        "throughput_cases_per_hour": round(rate * 3600.0, 3),
                        "eta_seconds": round((len(selected) - processed) / rate, 3)
                        if rate
                        else None,
                        "status": "progress",
                    },
                    allow_nan=False,
                    sort_keys=True,
                ),
                flush=True,
            )

    try:
        await asyncio.gather(*(run_one(index) for index in range(len(selected))))
    finally:
        await client.close()
    if any(row is None for row in results):
        raise RuntimeError("rewrite worker pool returned an incomplete result set")
    return tuple(cast(RewriteCaseBundle, row) for row in results)


def _unit_usage_for_document(run_root: Path, result: Mapping[str, Any]) -> dict[str, JsonValue]:
    paths = [*cast(list[str], result["partyUnitPaths"]), cast(str, result["cargoUnitPath"])]
    totals: Counter[str] = Counter()
    estimated_cost = Decimal(0)
    for relative in paths:
        unit = json.loads(read_regular_file_bytes(run_root / relative))
        for attempt in unit["attempts"]:
            usage = attempt["usage"]
            for key in (
                "requests",
                "inputTokens",
                "cacheReadTokens",
                "outputTokens",
                "reasoningTokens",
                "visibleOutputTokens",
            ):
                totals[key] += int(usage[key])
            estimated_cost += Decimal(str(usage["estimatedCostUsd"]))
    return {
        "linguisticUnits": len(paths),
        **dict(totals),
        "estimatedCostUsd": str(estimated_cost),
    }


def _plot_bytes(
    linguistic_rows: Sequence[Mapping[str, Any]], bundles: Sequence[RewriteCaseBundle]
) -> dict[str, bytes]:
    try:
        import io

        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import pandas as pd  # type: ignore[import-untyped]
        import seaborn as sns  # type: ignore[import-untyped]
    except ImportError as error:
        raise RuntimeError("rewrite analysis requires matplotlib, pandas, and seaborn") from error
    sns.set_theme(style="whitegrid", context="notebook")
    linguistic = pd.DataFrame(linguistic_rows)
    rewrite_rows = [
        {
            "case": row.result.caseNumber,
            "status": row.result.status,
            "input_tokens": row.result.usage.inputTokens,
            "output_tokens": row.result.usage.outputTokens,
            "reasoning_tokens": row.result.usage.reasoningTokens,
            "cost_usd": float(row.result.usage.estimatedCostUsd),
            "duration_seconds": row.result.metrics.durationMs / 1000.0,
            "edit_specs": row.result.metrics.editSpecifications,
            "occurrences": row.result.metrics.replacedOccurrences,
            "required_paths": row.result.metrics.requiredChangedPaths,
            "auxiliary": row.result.metrics.auxiliaryAnonymizations,
            "matching_fraction": row.result.metrics.matchingBlockSourceFraction,
            "length_delta": row.result.metrics.lengthDelta,
        }
        for row in bundles
    ]
    rewrite = pd.DataFrame(rewrite_rows)
    plots: dict[str, bytes] = {}

    def render(name: str, figure: Any) -> None:
        stream = io.BytesIO()
        figure.savefig(
            stream,
            format="png",
            dpi=170,
            bbox_inches="tight",
            metadata={"Software": "document-ocr synthetic raw-text rewrite probe"},
        )
        plots[name] = stream.getvalue()
        plt.close(figure)

    figure, axes = plt.subplots(1, 2, figsize=(13, 5))
    sns.countplot(data=linguistic, x="document_type", ax=axes[0], color="#2563EB")
    sns.countplot(data=linguistic, x="source_corpus", ax=axes[1], color="#0EA5E9")
    axes[0].set(title="50 synthetic targets by document type", xlabel="", ylabel="documents")
    axes[1].set(title="50 synthetic targets by source corpus", xlabel="", ylabel="documents")
    for axis in axes:
        axis.tick_params(axis="x", rotation=25)
    render("01_synthesis50_composition.png", figure)

    melted = linguistic.melt(
        id_vars="document_id",
        value_vars=("page_count", "container_count", "goods_group_count", "package_fact_count"),
        var_name="feature",
        value_name="count",
    )
    figure, axis = plt.subplots(figsize=(12, 6))
    sns.boxplot(data=melted, x="feature", y="count", ax=axis, color="#93C5FD")
    sns.stripplot(data=melted, x="feature", y="count", ax=axis, color="#1E3A8A", alpha=0.55)
    axis.set(
        title="Template complexity across the 50 synthesized targets", xlabel="", ylabel="count"
    )
    render("02_synthesis50_complexity.png", figure)

    special = pd.DataFrame(
        {
            "feature": [
                "multi-page",
                "multi-container",
                "multi-goods",
                "dangerous goods",
                "temperature",
            ],
            "documents": [
                int(linguistic["multi_page"].sum()),
                int(linguistic["multi_container"].sum()),
                int(linguistic["multi_goods"].sum()),
                int(linguistic["dangerous_goods_present"].sum()),
                int(linguistic["temperature_present"].sum()),
            ],
        }
    )
    figure, axis = plt.subplots(figsize=(11, 5))
    sns.barplot(data=special, x="documents", y="feature", ax=axis, color="#14B8A6")
    axis.set(
        title="Special-feature coverage in the 50-target baseline", xlabel="documents", ylabel=""
    )
    render("03_synthesis50_special_features.png", figure)

    figure, axes = plt.subplots(1, 2, figsize=(14, 5.5))
    sns.scatterplot(
        data=linguistic,
        x="inputTokens",
        y="outputTokens",
        size="linguisticUnits",
        hue="source_corpus",
        ax=axes[0],
    )
    sns.histplot(data=linguistic, x="estimatedCostUsdFloat", bins=12, ax=axes[1], color="#F59E0B")
    axes[0].set(title="Linguistic-generation tokens per document")
    axes[1].set(title="Linguistic-generation cost per document", xlabel="estimated USD")
    render("04_synthesis50_provider_usage.png", figure)

    figure, axes = plt.subplots(1, 2, figsize=(14, 5.5))
    sns.barplot(data=rewrite, x="case", y="cost_usd", hue="status", ax=axes[0], dodge=False)
    sns.barplot(data=rewrite, x="case", y="duration_seconds", hue="status", ax=axes[1], dodge=False)
    axes[0].set(title="Rewrite cost by case", xlabel="case", ylabel="estimated USD")
    axes[1].set(title="Rewrite wall time by case", xlabel="case", ylabel="seconds")
    for axis in axes:
        legend = axis.get_legend()
        if legend is not None:
            legend.set_title("")
    render("05_rewrite_cost_and_duration.png", figure)

    edit_melted = rewrite.melt(
        id_vars=("case", "status"),
        value_vars=("edit_specs", "occurrences", "required_paths", "auxiliary"),
        var_name="measure",
        value_name="count",
    )
    figure, axis = plt.subplots(figsize=(13, 6))
    sns.barplot(data=edit_melted, x="case", y="count", hue="measure", ax=axis)
    axis.set(title="Rewrite edit and changed-path volume", xlabel="case", ylabel="count")
    render("06_rewrite_edit_volume.png", figure)

    figure, axes = plt.subplots(1, 2, figsize=(14, 5.5))
    sns.barplot(
        data=rewrite, x="case", y="matching_fraction", hue="status", ax=axes[0], dodge=False
    )
    sns.barplot(data=rewrite, x="case", y="length_delta", hue="status", ax=axes[1], dodge=False)
    axes[0].set(ylim=(0, 1.01), title="Byte-order-preserving matching-block share", xlabel="case")
    axes[1].set(title="Raw-text character-count delta", xlabel="case")
    for axis in axes:
        legend = axis.get_legend()
        if legend is not None:
            legend.set_title("")
    render("07_rewrite_text_fidelity.png", figure)

    token_frame = rewrite.melt(
        id_vars=("case", "status"),
        value_vars=("input_tokens", "output_tokens", "reasoning_tokens"),
        var_name="token_kind",
        value_name="tokens",
    )
    figure, axis = plt.subplots(figsize=(13, 6))
    sns.barplot(data=token_frame, x="case", y="tokens", hue="token_kind", ax=axis)
    axis.set(title="Rewrite provider tokens by case", xlabel="case", ylabel="tokens")
    render("08_rewrite_provider_tokens.png", figure)
    return plots


def _markdown_fence(value: str, language: str) -> str:
    fence = "~~~~"
    while fence in value:
        fence += "~"
    return f"{fence}{language}\n{value}\n{fence}"


def _report(
    *,
    summary: Mapping[str, Any],
    bundles: Sequence[RewriteCaseBundle],
) -> str:
    case_outcome = (
        f"{summary['completeCases']} / {summary['blockedCases']} / {summary['callFailedCases']}"
    )
    token_outcome = (
        f"{summary['inputTokens']:,} / {summary['outputTokens']:,} / {summary['reasoningTokens']:,}"
    )
    lines = [
        "# Synthetic B/L raw-text rewrite probe",
        "",
        "## Outcome",
        "",
        f"- Upstream synthesized targets evaluated: **{summary['synthesisDocuments']:,}**",
        f"- Stand-alone rewrite cases: **{summary['rewriteCases']:,}**",
        f"- Complete / blocked / call failed: **{case_outcome}**",
        f"- Provider requests: **{summary['requests']:,}**",
        f"- Input / output / reasoning tokens: **{token_outcome}**",
        f"- Estimated rewrite cost: **${summary['estimatedCostUsd']}**",
        f"- Concurrent wall time: **{summary['wallSeconds']:.3f} seconds**",
        "",
        "Each case was isolated. The model received the source label, complete source OCR, "
        "synthetic",
        "target label, and exact changed-leaf inventory. It could mutate text only through exact",
        "contextual edit calls, and completion required inspection of the final unified diff. No",
        "training records were published by this probe.",
        "",
        "## Visual analysis",
        "",
        "![50-target composition](plots/01_synthesis50_composition.png)",
        "![50-target complexity](plots/02_synthesis50_complexity.png)",
        "![Special features](plots/03_synthesis50_special_features.png)",
        "![Upstream provider usage](plots/04_synthesis50_provider_usage.png)",
        "![Rewrite cost and duration](plots/05_rewrite_cost_and_duration.png)",
        "![Rewrite edit volume](plots/06_rewrite_edit_volume.png)",
        "![Rewrite text fidelity](plots/07_rewrite_text_fidelity.png)",
        "![Rewrite provider tokens](plots/08_rewrite_provider_tokens.png)",
        "",
        "## Case index",
        "",
        "| Case | Document | Pages | Containers | Goods | DG | Temperature | Status | Cost (USD) |",
        "|---:|---|---:|---:|---:|:---:|:---:|---|---:|",
    ]
    for bundle in bundles:
        result = bundle.result
        feature = bundle.feature
        lines.append(
            (
                "| {case} | `{doc}` | {pages} | {containers} | {goods} | {dg} | "
                "{temp} | {status} | {cost} |"
            ).format(
                case=result.caseNumber,
                doc=result.documentId,
                pages=feature["page_count"],
                containers=feature["container_count"],
                goods=feature["goods_group_count"],
                dg="yes" if feature["dangerous_goods_present"] else "no",
                temp="yes" if feature["temperature_present"] else "no",
                status=result.status,
                cost=result.usage.estimatedCostUsd,
            )
        )
    for bundle in bundles:
        result = bundle.result
        feature = bundle.feature
        source_line = (
            f"- Source: `{feature['source_corpus']}` / `{feature['document_type']}`; "
            f"pages={feature['page_count']}, containers={feature['container_count']}, "
            f"goods={feature['goods_group_count']}"
        )
        path_line = (
            "- Required/resolved changed paths: "
            f"**{result.metrics.requiredChangedPaths}/"
            f"{result.metrics.resolvedChangedPaths}**"
        )
        edit_line = (
            "- Edit batches/specifications/occurrences: "
            f"**{result.metrics.editBatches}/{result.metrics.editSpecifications}/"
            f"{result.metrics.replacedOccurrences}**"
        )
        auxiliary_line = (
            f"- Auxiliary personal-data replacements: **{result.metrics.auxiliaryAnonymizations}**"
        )
        match_line = (
            "- Matching-block source fraction: "
            f"**{result.metrics.matchingBlockSourceFraction:.6f}**"
        )
        usage_line = (
            "- Input/output/reasoning tokens: "
            f"**{result.usage.inputTokens}/{result.usage.outputTokens}/"
            f"{result.usage.reasoningTokens}**; cost "
            f"**${result.usage.estimatedCostUsd}**"
        )
        lines.extend(
            [
                "",
                f"## Case {result.caseNumber}: `{result.documentId}`",
                "",
                f"- Status: **{result.status}**",
                source_line,
                path_line,
                edit_line,
                auxiliary_line,
                match_line,
                usage_line,
                f"- Deterministic checks: `{json.dumps(result.checks, sort_keys=True)}`",
                "",
                "### Source label supplied to the agent",
                "",
                _markdown_fence(
                    json.dumps(bundle.sourceLabel, ensure_ascii=False, indent=2, sort_keys=True),
                    "json",
                ),
                "",
                "### Synthetic target label supplied to the agent",
                "",
                _markdown_fence(
                    json.dumps(bundle.targetLabel, ensure_ascii=False, indent=2, sort_keys=True),
                    "json",
                ),
                "",
                "### Source raw OCR supplied to the agent",
                "",
                _markdown_fence(bundle.sourceText, "text"),
                "",
                "### Rewritten raw OCR",
                "",
                _markdown_fence(bundle.outputText, "text"),
                "",
                "### Exact unified diff",
                "",
                _markdown_fence(bundle.diff or "(no diff)", "diff"),
                "",
                "### Applied edit receipt",
                "",
                _markdown_fence(
                    json.dumps(
                        [row.model_dump(mode="json") for row in bundle.appliedEdits],
                        ensure_ascii=False,
                        indent=2,
                        sort_keys=True,
                    ),
                    "json",
                ),
            ]
        )
    return "\n".join(lines) + "\n"


def _publish_case_files(staged: StagedArtifactRun, bundle: RewriteCaseBundle) -> list[str]:
    prefix = f"cases/{bundle.result.caseNumber:02d}-{bundle.result.documentId}"
    values: dict[str, bytes] = {
        f"{prefix}/source-label.json": json_artifact_bytes(bundle.sourceLabel),
        f"{prefix}/target-label.json": json_artifact_bytes(bundle.targetLabel),
        f"{prefix}/changed-leaves.json": json_artifact_bytes(
            [row.model_dump(mode="json") for row in bundle.changedLeaves]
        ),
        f"{prefix}/raw-before.txt": bundle.sourceText.encode("utf-8"),
        f"{prefix}/raw-after.txt": bundle.outputText.encode("utf-8"),
        f"{prefix}/edits.json": json_artifact_bytes(
            [row.model_dump(mode="json") for row in bundle.appliedEdits]
        ),
        f"{prefix}/diff.patch": bundle.diff.encode("utf-8"),
        f"{prefix}/input.json": json_artifact_bytes(bundle.promptPayload),
        f"{prefix}/messages.json": json_artifact_bytes(bundle.modelMessages),
        f"{prefix}/result.json": json_artifact_bytes(bundle.result.model_dump(mode="json")),
    }
    for relative, payload in values.items():
        staged.publish_bytes(relative, payload)
    return sorted(values)


def run_raw_text_rewrite_probe(
    *,
    project_root: Path,
    config_path: Path,
    config: SynthesisRawTextRewriteProbeConfig,
) -> dict[str, JsonValue]:
    linguistic_root = _validate_linguistic_run(project_root, config)
    results_path = _resolve_pinned_file(
        project_root,
        config.inputs.linguistic_results.path,
        config.inputs.linguistic_results.sha256,
        label="linguistic results",
    )
    targets_path = _resolve_pinned_file(
        project_root,
        config.inputs.synthetic_targets.path,
        config.inputs.synthetic_targets.sha256,
        label="linguistic targets",
    )
    summary_path = _resolve_pinned_file(
        project_root,
        config.inputs.linguistic_summary.path,
        config.inputs.linguistic_summary.sha256,
        label="linguistic summary",
    )
    for path in (results_path, targets_path, summary_path):
        if not path.is_relative_to(linguistic_root):
            raise ValueError("linguistic artifact is outside the pinned committed run")
    source_path = _resolve_pinned_file(
        project_root,
        config.inputs.source_corpus.path,
        config.inputs.source_corpus.sha256,
        label="source corpus",
    )
    features_path = _resolve_pinned_file(
        project_root,
        config.inputs.document_features.path,
        config.inputs.document_features.sha256,
        label="document features",
    )
    prompt_path = _resolve_pinned_file(
        project_root, config.prompt.path, config.prompt.sha256, label="rewrite prompt"
    )
    prompt_bytes = read_regular_file_bytes(prompt_path)
    try:
        system_prompt = prompt_bytes.decode("utf-8")
    except UnicodeDecodeError as error:
        raise ValueError("rewrite prompt is not valid UTF-8") from error

    results = _load_jsonl(
        results_path,
        records=config.inputs.linguistic_results.records,
        key="baseDocumentId",
        label="linguistic results",
    )
    targets = _load_jsonl(
        targets_path,
        records=config.inputs.synthetic_targets.records,
        key="baseDocumentId",
        label="linguistic targets",
    )
    source_rows = _load_jsonl(
        source_path,
        records=config.inputs.source_corpus.records,
        key="documentId",
        label="source corpus",
    )
    features = _load_jsonl(
        features_path,
        records=config.inputs.document_features.records,
        key="document_id",
        label="document features",
    )
    result_by_id = {row["baseDocumentId"]: row for row in results}
    target_by_id = {row["baseDocumentId"]: row for row in targets}
    source_by_id = {row["documentId"]: row for row in source_rows}
    feature_by_id = {row["document_id"]: row for row in features}
    if tuple(result_by_id) != tuple(target_by_id):
        raise ValueError("linguistic result and target order differs")
    for document_id, result in result_by_id.items():
        target = target_by_id[document_id]
        if result.get("status") != "success" or result.get("target") != target.get("target"):
            raise ValueError(f"linguistic result/target mismatch for {document_id}")
        target_sha = sha256_bytes(canonical_json_bytes(target["target"]))
        if target.get("targetSha256") != target_sha or result.get("targetSha256") != target_sha:
            raise ValueError(f"linguistic target SHA-256 mismatch for {document_id}")
    selected: list[tuple[Mapping[str, Any], Mapping[str, Any], Mapping[str, JsonValue]]] = []
    for case in config.cases:
        try:
            source = source_by_id[case.document_id]
            target = target_by_id[case.document_id]
            feature = feature_by_id[case.document_id]
        except KeyError as error:
            raise ValueError(
                f"rewrite case is absent from a pinned input: {case.document_id}"
            ) from error
        raw_text = source.get("joinedRawText")
        raw_sha = source.get("joinedRawTextSha256")
        if not isinstance(raw_text, str) or sha256_bytes(raw_text.encode("utf-8")) != raw_sha:
            raise ValueError(f"source raw text fails its SHA-256: {case.document_id}")
        selected.append((source, target, cast(Mapping[str, JsonValue], feature)))

    output_schema = RewriteCompletionReceipt.model_json_schema(mode="validation")
    transaction = {
        "schemaVersion": 1,
        "runId": config.run.run_id,
        "configSha256": sha256_file(config_path),
        "linguisticRunTransactionSha256": (
            config.inputs.linguistic_completion_run.transaction_sha256
        ),
        "linguisticResultsSha256": config.inputs.linguistic_results.sha256,
        "syntheticTargetsSha256": config.inputs.synthetic_targets.sha256,
        "sourceCorpusSha256": config.inputs.source_corpus.sha256,
        "documentFeaturesSha256": config.inputs.document_features.sha256,
        "promptSha256": config.prompt.sha256,
        "outputSchemaSha256": sha256_bytes(canonical_json_bytes(output_schema)),
        "caseDocumentIds": [case.document_id for case in config.cases],
        "implementationSha256": sha256_file(_IMPLEMENTATION_PATH),
        "sourceSchemaImplementationSha256": sha256_file(
            Path(bill_of_lading_v3.__file__).resolve(strict=True)
        ),
        "targetSchemaImplementationSha256": sha256_file(
            Path(bill_of_lading_v5.__file__).resolve(strict=True)
        ),
        "runtime": {
            "model": config.provider.model,
            "reasoningEffort": config.provider.reasoning_effort,
            "pydanticAiVersion": version("pydantic-ai-slim"),
            "openaiVersion": version("openai"),
        },
    }
    transaction_sha256 = sha256_bytes(canonical_json_bytes(transaction))
    staged = StagedArtifactRun(
        output_parent=resolve_config_path(project_root, config.run.output_dir),
        run_name=config.run.run_id,
        transaction_sha256=transaction_sha256,
    )
    case_expected: list[str] = []
    for index, case in enumerate(config.cases, start=1):
        prefix = f"cases/{index:02d}-{case.document_id}"
        case_expected.append(f"{prefix}/bundle.json")
        case_expected.extend(
            f"{prefix}/{name}"
            for name in (
                "changed-leaves.json",
                "diff.patch",
                "edits.json",
                "input.json",
                "messages.json",
                "raw-after.txt",
                "raw-before.txt",
                "result.json",
                "source-label.json",
                "target-label.json",
            )
        )
    # Plot names are explicit so an empty or renamed plot cannot silently enter publication.
    plot_expected = [
        "plots/01_synthesis50_composition.png",
        "plots/02_synthesis50_complexity.png",
        "plots/03_synthesis50_special_features.png",
        "plots/04_synthesis50_provider_usage.png",
        "plots/05_rewrite_cost_and_duration.png",
        "plots/06_rewrite_edit_volume.png",
        "plots/07_rewrite_text_fidelity.png",
        "plots/08_rewrite_provider_tokens.png",
    ]
    expected = sorted(
        [
            "REPORT.md",
            "analysis/linguistic50-metrics.jsonl",
            "analysis/summary.json",
            "config.yaml",
            "generation/results.jsonl",
            "generation/summary.json",
            "prompt/rewrite.md",
            "schema/rewrite-receipt.schema.json",
            *case_expected,
            *plot_expected,
        ]
    )
    if staged.completed:
        staged.commit(
            expected_artifacts=expected,
            metadata={"rewrite_cases": 10, "schema_version": 1},
        )
        return cast(
            dict[str, JsonValue],
            json.loads(read_regular_file_bytes(staged.final_root / "generation/summary.json")),
        )
    staged.recover_interrupted_temporary_files()
    staged.publish_bytes("config.yaml", read_regular_file_bytes(config_path))
    staged.publish_bytes("prompt/rewrite.md", prompt_bytes)
    staged.publish_bytes("schema/rewrite-receipt.schema.json", json_artifact_bytes(output_schema))
    wall_started = time.perf_counter()
    bundles = asyncio.run(
        _run_cases(
            selected=selected,
            system_prompt=system_prompt,
            staged=staged,
            config=config,
            project_root=project_root,
        )
    )
    wall_seconds = time.perf_counter() - wall_started
    for bundle in bundles:
        _publish_case_files(staged, bundle)

    linguistic_metrics: list[dict[str, JsonValue]] = []
    for result in results:
        document_id = cast(str, result["baseDocumentId"])
        feature = feature_by_id[document_id]
        usage = _unit_usage_for_document(linguistic_root, result)
        linguistic_metrics.append(
            {
                "document_id": document_id,
                "document_type": feature["document_type"],
                "source_corpus": feature["source_corpus"],
                "page_count": feature["page_count"],
                "container_count": feature["container_count"],
                "goods_group_count": feature["goods_group_count"],
                "package_fact_count": feature["package_fact_count"],
                "multi_page": feature["multi_page"],
                "multi_container": feature["multi_container"],
                "multi_goods": feature["multi_goods"],
                "dangerous_goods_present": feature["dangerous_goods_present"],
                "temperature_present": feature["temperature_present"],
                **usage,
                "estimatedCostUsdFloat": float(cast(str, usage["estimatedCostUsd"])),
            }
        )
    linguistic_payload = b"".join(canonical_json_bytes(row) + b"\n" for row in linguistic_metrics)
    staged.publish_bytes("analysis/linguistic50-metrics.jsonl", linguistic_payload)

    total_cost = sum((bundle.result.usage.estimatedCostUsd for bundle in bundles), Decimal(0))
    summary: dict[str, JsonValue] = {
        "schemaVersion": 1,
        "runId": config.run.run_id,
        "status": "complete",
        "synthesisDocuments": len(results),
        "rewriteCases": len(bundles),
        "completeCases": sum(row.result.status == "complete" for row in bundles),
        "blockedCases": sum(row.result.status == "blocked" for row in bundles),
        "callFailedCases": sum(row.result.status == "call_failed" for row in bundles),
        "requests": sum(row.result.usage.requests for row in bundles),
        "inputTokens": sum(row.result.usage.inputTokens for row in bundles),
        "cacheReadTokens": sum(row.result.usage.cacheReadTokens for row in bundles),
        "outputTokens": sum(row.result.usage.outputTokens for row in bundles),
        "reasoningTokens": sum(row.result.usage.reasoningTokens for row in bundles),
        "visibleOutputTokens": sum(row.result.usage.visibleOutputTokens for row in bundles),
        "estimatedCostUsd": str(total_cost),
        "wallSeconds": wall_seconds,
        "throughputCasesPerHour": len(bundles) / wall_seconds * 3600.0,
        "peakResidentMemoryMiB": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0,
        "trainingRecordsPublished": False,
    }
    results_payload = b"".join(
        canonical_json_bytes(bundle.result.model_dump(mode="json")) + b"\n" for bundle in bundles
    )
    staged.publish_bytes("generation/results.jsonl", results_payload)
    staged.publish_json("generation/summary.json", summary)
    staged.publish_json("analysis/summary.json", summary)
    plots = _plot_bytes(linguistic_metrics, bundles)
    if set(plots) != {Path(path).name for path in plot_expected}:
        raise RuntimeError("rewrite analysis produced an unexpected plot inventory")
    for name, payload in plots.items():
        staged.publish_bytes(f"plots/{name}", payload)
    staged.publish_bytes("REPORT.md", _report(summary=summary, bundles=bundles).encode("utf-8"))
    commit = staged.commit(
        expected_artifacts=expected,
        metadata={"rewrite_cases": 10, "schema_version": 1},
    )
    summary["commitContentSha256"] = commit.receipt.content_sha256
    return summary
