"""Inventory-complete, bounded-call synthetic raw-OCR rendering experiment.

This runner evaluates three claims independently:

1. A full-document guard rejects the known false passes from both prior model arms.
2. Source-only formal identifiers and contacts can be rendered locally at zero model cost.
3. The remaining host-owned lines can be returned in one provider-constrained dictionary, with
   every output key required and no model reviewer. A bounded corrective resubmission is allowed
   only after a deterministic host rejection.

It is intentionally an evaluation runner and never publishes training records.
"""

from __future__ import annotations

import asyncio
import copy
import json
import re
import time
from collections import Counter, defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, replace
from decimal import Decimal
from importlib.metadata import version
from itertools import pairwise
from pathlib import Path
from typing import Annotated, Any, Literal, cast

from openai import AsyncOpenAI
from pydantic import BaseModel, ConfigDict, Field, JsonValue
from pydantic_ai import (
    Agent,
    NativeOutput,
    StructuredDict,
    ToolOutput,
    capture_run_messages,
)
from pydantic_ai.exceptions import ModelAPIError, ModelHTTPError
from pydantic_ai.messages import ModelResponse
from pydantic_ai.models import Model
from pydantic_ai.usage import UsageLimits

from document_ocr.atomic import read_regular_file_bytes
from document_ocr.hashing import canonical_json_bytes, sha256_bytes, sha256_file
from document_ocr.label_schemas.bill_of_lading_v5 import (
    ContainerSizeCategory,
    ContainerTypeCategory,
)
from document_ocr.synthesis.config import (
    SynthesisRawTextInventoryBatchConfig,
    SynthesisRawTextInventoryProbeConfig,
    load_synthesis_raw_text_hybrid_batch_config,
)
from document_ocr.synthesis.container_semantics import canonical_equipment_surface
from document_ocr.synthesis.drafts import target_value
from document_ocr.synthesis.linguistic_probe_runtime import (
    LinguisticUsageReceipt,
    load_provider_key,
    model_messages,
    usage_receipt,
)
from document_ocr.synthesis.raw_text_hybrid_batch import _load_inputs
from document_ocr.synthesis.raw_text_hybrid_probe import (
    HybridWorkItem,
    _artifact_inventory,
    _CompiledCase,
    _container_identifier_line_numbers,
    _context_bound_surface_lines,
    _contextual_location_line_numbers,
    _literal_line_numbers,
    _literal_occurrence_line_sets,
    _model_pair,
    _numeric_surface_values,
    _object_scope,
    _party_block_line_groups,
    _provider_attempt_routes,
    _retryable_route_error,
    _source_party,
    _temperature_deactivation_line_numbers,
    compile_case,
)
from document_ocr.synthesis.raw_text_inventory import (
    DeterministicInventoryEdit,
    FullDocumentAudit,
    InventoryCandidate,
    RegressionCase,
    RegressionOracle,
    TemplateMutationCase,
    TemplateMutationProfile,
    _is_shipment_aggregate_value_line,
    apply_deterministic_auxiliary_edits,
    audit_full_document,
    auxiliary_label_surfaces,
    build_mutable_inventory,
    line_number,
    parse_regression_oracle,
    parse_template_mutation_profile,
)
from document_ocr.synthesis.raw_text_rewrite_cycle_probe import (
    _PARTY_HEADING_LINE,
    AtomicRewriteCommit,
    CompoundPartyFlavorRealization,
    CompoundPartyFlavorRequirement,
    DeterministicRewriteAudit,
    LineRangeReplacement,
    RewriteCycleBundle,
    RewriteWorkspace,
    SourceStatusPreservationRequirement,
    SurfaceRenderingRequirement,
    TargetValueOccurrenceRequirement,
    _auxiliary_identity_token_key,
    _cargo_flavor_rewrite_failures,
    _line_ending,
    _missing_target_literals,
    _party_scalar_occurrence_count,
    _party_scalar_occurrence_line_sets,
    _semantic_normalize,
    _settings,
    _target_value_occurrence_count,
    apply_line_range_replacements,
    build_rewrite_contract_bundle,
    build_target_integrity_resources,
    carrier_principal_occurrence_count,
    carrier_principal_template_slot_count,
    deterministic_rewrite_audit,
    prepare_rewrite_state,
    unexpected_cargo_package_surfaces,
)
from document_ocr.synthesis.raw_text_rewrite_probe import (
    _resolve_pinned_file,
    unified_text_diff,
)
from document_ocr.synthesis.run_safety import StagedArtifactRun
from document_ocr.training.config import resolve_config_path

_STRICT = ConfigDict(extra="forbid", frozen=True, strict=True, allow_inf_nan=False)
_IMPLEMENTATION_PATH = Path(__file__).resolve(strict=True)
_OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"


class InventoryHostEvaluation(BaseModel):
    """Deterministic disposition of one provider-valid patch candidate."""

    model_config = _STRICT

    passed: bool
    phase: Literal["atomic_apply", "full_document_audit"]
    errorType: str | None
    errorMessage: str | None
    requestedRepairSlots: tuple[str, ...]
    requestedRepairCompoundSlots: tuple[str, ...]
    hostRewriteAudit: DeterministicRewriteAudit | None
    fullDocumentAudit: FullDocumentAudit | None


class InventoryModelStage(BaseModel):
    model_config = _STRICT

    providerModel: str
    semanticAttempt: Annotated[int, Field(ge=1, le=4)]
    requestKind: Literal["initial_partition", "repair"]
    initialPartitionOrdinal: Annotated[int, Field(ge=1)] | None
    initialPartitionCount: Annotated[int, Field(ge=1)]
    routeProvider: str | None
    routeRound: Annotated[int, Field(ge=1)]
    routeAttempt: Annotated[int, Field(ge=1)]
    retryDelayBeforeSeconds: Annotated[float, Field(ge=0)]
    outputMode: Literal["native", "tool"]
    inputPayloadSha256: str
    outputSchemaSha256: str
    startedAtUnixSeconds: float
    completedAtUnixSeconds: float
    usage: LinguisticUsageReceipt
    messages: JsonValue
    modelOutput: JsonValue | None
    hostEvaluation: InventoryHostEvaluation | None
    errorType: str | None
    errorMessage: str | None


class InventoryProbeCaseResult(BaseModel):
    model_config = _STRICT

    documentId: str
    status: Literal["training_ready", "needs_review", "call_failed", "compiler_blocked"]
    reason: str
    sourceLines: int
    modelLines: int
    compilerWorkItems: int
    inventoryCandidates: int
    deterministicInventoryEdits: int
    hostRewriteAuditPassed: bool
    legacyResidualCandidates: Annotated[int, Field(ge=0)]
    fullDocumentAudit: FullDocumentAudit | None = None
    compilerErrorType: str | None = None
    compilerErrorMessage: str | None = None
    outputTextSha256: str
    usage: LinguisticUsageReceipt


@dataclass(frozen=True, slots=True)
class _Slot:
    alias: str
    line_id: str
    source_line: str
    requirements: tuple[dict[str, JsonValue], ...]


@dataclass(frozen=True, slots=True)
class _CompoundSlot:
    alias: str
    requirement: CompoundPartyFlavorRequirement


@dataclass(frozen=True, slots=True)
class _InitialRequestBatch:
    slots: tuple[_Slot, ...]
    compound_slots: tuple[_CompoundSlot, ...]


InventoryRunConfig = SynthesisRawTextInventoryProbeConfig | SynthesisRawTextInventoryBatchConfig


@dataclass(frozen=True, slots=True)
class _InventoryCaseMaterial:
    compiled: _CompiledCase
    oracle_case: RegressionCase | None
    inventory: tuple[InventoryCandidate, ...]
    deterministic_edits: tuple[DeterministicInventoryEdit, ...]
    slots: tuple[_Slot, ...]
    compound_slots: tuple[_CompoundSlot, ...]
    deterministic_text_sha256: str


@dataclass(frozen=True, slots=True)
class _CompilerBlock:
    document_id: str
    source_text: str
    source_label: Mapping[str, Any]
    target_label: Mapping[str, Any]
    error_type: str
    error_message: str


@dataclass(frozen=True, slots=True)
class _CandidateEvaluation:
    compiled: _CompiledCase
    host_audit: DeterministicRewriteAudit
    full_audit: FullDocumentAudit

    @property
    def passed(self) -> bool:
        return self.host_audit.core_passed and self.full_audit.passed


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


def _explicit_provider_order(config: InventoryRunConfig) -> tuple[str, ...]:
    provider = config.provider
    if provider.kind != "openrouter":
        return ()
    return provider.provider_order or ()


def _load_oracle(project_root: Path, config: InventoryRunConfig) -> RegressionOracle:
    path = _resolve_pinned_file(
        project_root,
        config.regression_oracle.path,
        config.regression_oracle.sha256,
        label="full-document rewrite regression oracle",
    )
    value = json.loads(read_regular_file_bytes(path))
    if not isinstance(value, Mapping):
        raise ValueError("full-document rewrite regression oracle root must be an object")
    oracle = parse_regression_oracle(cast(Mapping[str, Any], value))
    if len(oracle.cases) != config.workflow.regression_documents:
        raise ValueError("regression oracle document count differs from configured count")
    return oracle


def _load_template_mutation_profile(
    project_root: Path, config: InventoryRunConfig
) -> tuple[TemplateMutationProfile | None, Path | None]:
    if not isinstance(config, SynthesisRawTextInventoryBatchConfig):
        return None, None
    pinned = config.template_mutation_profile
    if pinned is None:
        return None, None
    path = _resolve_pinned_file(
        project_root,
        pinned.path,
        pinned.sha256,
        label="source-template mutation profile",
    )
    value = json.loads(read_regular_file_bytes(path))
    if not isinstance(value, Mapping):
        raise ValueError("source-template mutation profile root must be an object")
    return parse_template_mutation_profile(cast(Mapping[str, Any], value)), path


def _validate_reference_run(
    project_root: Path,
    path_value: str,
    commit_sha256: str,
    transaction_sha256: str,
) -> Path:
    root = resolve_config_path(project_root, path_value)
    commit = root / "_COMMIT.json"
    if root.is_symlink() or not root.is_dir() or commit.is_symlink() or not commit.is_file():
        raise ValueError(f"rewrite reference run is not a committed directory: {root}")
    if sha256_file(commit) != commit_sha256:
        raise ValueError(f"rewrite reference commit differs from configured pin: {root}")
    StagedArtifactRun(
        output_parent=root.parent,
        run_name=root.name,
        transaction_sha256=transaction_sha256,
    ).validate_committed_run()
    return root.resolve(strict=True)


def _read_hybrid_work_items(path: Path) -> tuple[HybridWorkItem, ...]:
    value = json.loads(read_regular_file_bytes(path))
    if not isinstance(value, list):
        raise ValueError(f"hybrid work-item artifact must be a list: {path}")
    return tuple(
        HybridWorkItem.model_validate_json(canonical_json_bytes(row), strict=True) for row in value
    )


def _negative_fixture_audits(
    *,
    reference_root: Path,
    oracle: RegressionOracle,
    source_by_id: Mapping[str, Mapping[str, Any]],
    target_by_id: Mapping[str, Mapping[str, Any]],
) -> tuple[dict[str, JsonValue], ...]:
    rows: list[dict[str, JsonValue]] = []
    for oracle_case in oracle.cases:
        document_id = oracle_case.documentId
        case_root = reference_root / "cases" / document_id
        source = source_by_id[document_id]
        target = target_by_id[document_id]
        final_path = case_root / "final.txt"
        work_items_path = case_root / "work-items.json"
        if not final_path.is_file() or not work_items_path.is_file():
            raise ValueError(f"rewrite reference lacks regression case: {document_id}")
        output_text = read_regular_file_bytes(final_path).decode("utf-8")
        work_items = _read_hybrid_work_items(work_items_path)
        inventory = build_mutable_inventory(
            source_text=cast(str, source["joinedRawText"]),
            current_text=output_text,
            source_label=cast(Mapping[str, Any], source["target"]),
            target_label=cast(Mapping[str, Any], target["target"]),
            work_items=work_items,
            oracle_case=oracle_case,
        )
        audit = audit_full_document(
            document_id=document_id,
            source_text=cast(str, source["joinedRawText"]),
            output_text=output_text,
            source_label=cast(Mapping[str, Any], source["target"]),
            target_label=cast(Mapping[str, Any], target["target"]),
            inventory=inventory,
            deterministic_edits=(),
            oracle_case=oracle_case,
            work_items=work_items,
        )
        rows.append(
            {
                "documentId": document_id,
                "referenceStatus": cast(
                    JsonValue,
                    json.loads(read_regular_file_bytes(case_root / "result.json")).get("status"),
                ),
                "newFullDocumentAuditPassed": audit.passed,
                "inventoryCandidates": len(inventory),
                "findingCount": len(audit.findings),
                "findingCategories": cast(
                    JsonValue, sorted({finding.category for finding in audit.findings})
                ),
                "audit": audit.model_dump(mode="json"),
            }
        )
    return tuple(rows)


def _work_item_requirement(item: HybridWorkItem) -> dict[str, JsonValue]:
    requirement: dict[str, JsonValue] = {
        "kind": "task_label_delta",
        "workItemId": item.workItemId,
        "action": item.action,
        "paths": cast(JsonValue, list(item.targetPaths)),
        "source": item.sourceValue,
        "target": item.targetValue,
    }
    if item.action in {"replace_equipment_surface", "add_equipment_surface"} and isinstance(
        item.targetValue, Mapping
    ):
        size = item.targetValue.get("sizeCategory")
        type_category = item.targetValue.get("typeCategory")
        if isinstance(size, str) and isinstance(type_category, str):
            requirement["unambiguousTargetPrintedSurface"] = canonical_equipment_surface(
                cast(ContainerSizeCategory, size),
                cast(ContainerTypeCategory, type_category),
            )
    return requirement


def _work_item_object_scopes(item: HybridWorkItem) -> frozenset[str]:
    return frozenset(_object_scope(path) for path in item.targetPaths)


_CARGO_PACKAGE_TYPE_PATH = re.compile(
    r"^documentPatch\.cargoPackages\[([0-9]+)\]\.typeCategory$"
)
_CARGO_GROUP_FIELD_PATH = re.compile(r"^documentPatch\.cargoGroups\[([0-9]+)\]\.")
_CARGO_MARKS_PATH = re.compile(
    r"^documentPatch\.cargoGroups\[([0-9]+)\]\.marksAndNumbers\[([0-9]+)\]$"
)
_PAGE_MARKER = re.compile(r"^--- PAGE [1-9][0-9]* ---[ \t]*$")


def _cargo_marks_values(label: Mapping[str, Any], group_index: int) -> tuple[str, ...] | None:
    groups = _sequence_objects(label, "cargoGroups")
    if group_index >= len(groups) or not isinstance(groups[group_index], Mapping):
        return None
    values = groups[group_index].get("marksAndNumbers")
    if not isinstance(values, Sequence) or isinstance(values, (str, bytes)):
        return None
    if not values or not all(isinstance(value, str) and value for value in values):
        return None
    return tuple(cast(str, value) for value in values)


def _cargo_marks_sequence_evidence(
    current_text: str,
    source_label: Mapping[str, Any],
    target_label: Mapping[str, Any],
) -> dict[str, frozenset[int]]:
    """Map complete ordered marks blocks to indexed cargo groups.

    Shared marks such as a plant code or country surface cannot identify one cargo group by
    themselves.  A complete multi-row sequence can: every row must contain the source or already
    rendered target value at the same schema position, and consecutive values must occupy
    consecutive non-blank lines without crossing a page marker.  Repeated copies of one complete
    group are retained.  A physical sequence matching more than one group is deliberately left
    unclaimed so the existing compiler remains fail-closed.
    """

    bodies = current_text.splitlines()
    occupied = tuple(
        number
        for number, body in enumerate(bodies, start=1)
        if body.strip()
    )
    occupied_position = {number: index for index, number in enumerate(occupied)}
    source_groups = _sequence_objects(source_label, "cargoGroups")
    target_groups = _sequence_objects(target_label, "cargoGroups")
    group_count = min(len(source_groups), len(target_groups))
    matches_by_group: dict[int, set[tuple[frozenset[int], ...]]] = defaultdict(set)

    for group_index in range(group_count):
        source_values = _cargo_marks_values(source_label, group_index)
        target_values = _cargo_marks_values(target_label, group_index)
        if (
            source_values is None
            or target_values is None
            or len(source_values) != len(target_values)
            or len(source_values) < 2
        ):
            continue
        alternatives = tuple(
            tuple(
                sorted(
                    set(_literal_occurrence_line_sets(current_text, source_value))
                    | set(_literal_occurrence_line_sets(current_text, target_value)),
                    key=lambda value: (min(value), len(value)),
                )
            )
            for source_value, target_value in zip(source_values, target_values, strict=True)
        )
        if any(not rows for rows in alternatives):
            continue

        def extend(
            position: int,
            selected: tuple[frozenset[int], ...],
            *,
            _alternatives: tuple[tuple[frozenset[int], ...], ...] = alternatives,
            _group_index: int = group_index,
        ) -> None:
            if position == len(_alternatives):
                matches_by_group[_group_index].add(selected)
                return
            for occurrence in _alternatives[position]:
                if not occurrence:
                    continue
                numbers = tuple(sorted(occurrence))
                if any(_PAGE_MARKER.fullmatch(bodies[number - 1]) for number in numbers):
                    continue
                if selected:
                    previous_end = max(selected[-1])
                    previous_position = occupied_position.get(previous_end)
                    if previous_position is None or previous_position + 1 >= len(occupied):
                        continue
                    if min(numbers) != occupied[previous_position + 1]:
                        continue
                extend(position + 1, (*selected, occurrence))

        extend(0, ())

    owners_by_sequence: dict[tuple[frozenset[int], ...], set[int]] = defaultdict(set)
    for group_index, sequences in matches_by_group.items():
        for sequence in sequences:
            owners_by_sequence[sequence].add(group_index)

    evidence: dict[str, set[int]] = defaultdict(set)
    for group_index, sequences in matches_by_group.items():
        for sequence in sequences:
            if owners_by_sequence[sequence] != {group_index}:
                continue
            for marks_index, occurrence in enumerate(sequence):
                evidence[
                    f"documentPatch.cargoGroups[{group_index}].marksAndNumbers[{marks_index}]"
                ].update(occurrence)
    return {path: frozenset(numbers) for path, numbers in evidence.items()}


def _refine_sibling_evidence(
    work_items: Sequence[HybridWorkItem],
    bundle: RewriteCycleBundle | None = None,
) -> tuple[HybridWorkItem, ...]:
    """Replace broad sibling-object fallback lines with proven same-object evidence.

    The generic hybrid compiler deliberately errs toward recall. For a package category whose
    schema token is not printed literally, that can include a whole cargo block—including email,
    customs, and legal lines. The inventory renderer needs the tighter boundary: use evidence
    from a directly located sibling (normally quantity, identifier, or description) in the same
    indexed object. If no such sibling exists, retain the original fail-closed evidence rather
    than guessing a new location.
    """

    refined: list[HybridWorkItem] = []
    for item in work_items:
        if item.locator != "sibling_object_evidence":
            refined.append(item)
            continue
        related_allocation_lines: set[str] = set()
        if bundle is not None:
            package_paths = tuple(
                path for path in item.targetPaths if _CARGO_PACKAGE_TYPE_PATH.fullmatch(path)
            )
            package_group_ids = frozenset().union(
                *(
                    _quantity_group_ids(
                        label,
                        path.removesuffix("typeCategory") + "quantity",
                    )
                    for label in (bundle.sourceLabel, bundle.targetLabel)
                    for path in package_paths
                )
            )
            if package_group_ids:
                related_allocation_lines.update(
                    line_id
                    for sibling in work_items
                    if sibling.workItemId != item.workItemId
                    and any(
                        _ALLOCATION_PACKAGE_QUANTITY_PATH.fullmatch(path) is not None
                        and (
                            _quantity_group_ids(bundle.sourceLabel, path)
                            | _quantity_group_ids(bundle.targetLabel, path)
                        )
                        & package_group_ids
                        for path in sibling.targetPaths
                    )
                    for line_id in sibling.evidenceLineIds
                )
        scopes = _work_item_object_scopes(item)
        direct_lines = {
            line_id
            for sibling in work_items
            if sibling.workItemId != item.workItemId
            and sibling.locator not in {"sibling_object_evidence", "policy_projection", "unlocated"}
            and scopes & _work_item_object_scopes(sibling)
            for line_id in sibling.evidenceLineIds
        }
        # A package category is printed on the per-container package rows when allocations are
        # explicit. Those relation-owned rows are stronger evidence than a shipment-total line,
        # which often contains only a number and previously caused category text to be inserted
        # into the wrong field.
        if related_allocation_lines:
            direct_lines = related_allocation_lines
        refined.append(
            item.model_copy(
                update={
                    "evidenceLineIds": tuple(sorted(direct_lines, key=line_number))
                    if direct_lines
                    else item.evidenceLineIds
                }
            )
        )
    return tuple(refined)


def _refine_cargo_group_evidence(
    current_text: str,
    work_items: Sequence[HybridWorkItem],
    bundle: RewriteCycleBundle,
) -> tuple[HybridWorkItem, ...]:
    """Bind repeated cargo scalars to their indexed, ordered OCR cargo block.

    Flattened carrier riders often repeat identical material, batch, grade, HS, or description
    values across many containers. Literal matching alone therefore gives one indexed target
    every physical occurrence, while sibling fallback can incorrectly place an additional-info
    value onto the description line. The linguistic cargo plan already proves each indexed
    description block and preserves source order. Its next description anchor is a deterministic
    upper boundary for the current group; exact source-scalar occurrences inside that interval
    are the only values owned by that indexed group.

    If an exact occurrence cannot be established inside the proven interval, this refinement does
    nothing. The existing compiler state remains fail-closed rather than guessing from proximity.
    """

    marks_evidence = _cargo_marks_sequence_evidence(
        current_text,
        bundle.sourceLabel,
        bundle.targetLabel,
    )
    sequence_refined: list[HybridWorkItem] = []
    for item in work_items:
        marks_paths = tuple(path for path in item.targetPaths if _CARGO_MARKS_PATH.fullmatch(path))
        if marks_paths and len(marks_paths) == len(item.targetPaths) and all(
            path in marks_evidence for path in marks_paths
        ):
            owned = set().union(*(marks_evidence[path] for path in marks_paths))
            sequence_refined.append(
                item.model_copy(
                    update={
                        "evidenceLineIds": tuple(
                            f"L{number:05d}" for number in sorted(owned)
                        ),
                        "locator": "cargo_group_relation_scope",
                        "rationale": (
                            "Exact position inside a complete, ordered, indexed marks-and-numbers "
                            "sequence."
                        ),
                    }
                )
            )
        else:
            sequence_refined.append(item)

    work_items = tuple(sequence_refined)
    anchors: dict[int, tuple[int, ...]] = {}
    for requirement in bundle.cargoFlavorRewriteRequirements:
        if requirement.lineRole != "label_grounded":
            continue
        match = re.fullmatch(
            r"documentPatch\.cargoGroups\[([0-9]+)\]\.description",
            requirement.targetPath,
        )
        if match is None or not requirement.sourceLineIds:
            continue
        index = int(match.group(1))
        numbers = tuple(sorted(line_number(value) for value in requirement.sourceLineIds))
        if index in anchors and anchors[index] != numbers:
            raise ValueError(f"cargo group {index} has conflicting description anchors")
        anchors[index] = numbers
    if not anchors:
        return tuple(work_items)

    ordered = sorted(anchors)
    if any(
        min(anchors[left]) >= min(anchors[right])
        for left, right in pairwise(ordered)
    ):
        # Ordered source-label groups are the required relation key. A non-monotonic plan cannot
        # safely partition repeated scalars, so preserve the original fail-closed compiler output.
        return tuple(work_items)
    line_count = len(current_text.splitlines())
    regions = {
        index: (
            min(anchors[index]),
            (min(anchors[ordered[position + 1]]) - 1)
            if position + 1 < len(ordered)
            else line_count,
        )
        for position, index in enumerate(ordered)
    }

    refined: list[HybridWorkItem] = []
    for item in work_items:
        if item.state != "agent_residual" or not isinstance(item.sourceValue, str):
            refined.append(item)
            continue
        group_indices = {
            int(match.group(1))
            for path in item.targetPaths
            if (match := _CARGO_GROUP_FIELD_PATH.match(path)) is not None
        }
        if not group_indices or any(index not in regions for index in group_indices):
            refined.append(item)
            continue
        exact_groups = _literal_occurrence_line_sets(current_text, item.sourceValue)
        owned = {
            number
            for index in group_indices
            for start, end in (regions[index],)
            for occurrence in exact_groups
            if occurrence and min(occurrence) >= start and max(occurrence) <= end
            for number in occurrence
        }
        if not owned:
            refined.append(item)
            continue
        refined.append(
            item.model_copy(
                update={
                    "evidenceLineIds": tuple(
                        f"L{number:05d}" for number in sorted(owned)
                    ),
                    "locator": "cargo_group_relation_scope",
                    "rationale": (
                        "Exact source scalar occurrence inside the indexed cargo block bounded "
                        "by the ordered description plan."
                    ),
                }
            )
        )
    return tuple(refined)


_PRINTED_EQUIPMENT_PATH = re.compile(
    r"^documentPatch\.containers\[([0-9]+)\]\.printedEquipmentSurface$"
)


def _refine_equipment_evidence(
    current_text: str,
    work_items: Sequence[HybridWorkItem],
    bundle: RewriteCycleBundle,
) -> tuple[HybridWorkItem, ...]:
    """Bind equipment-category edits only to a proven printed-equipment source line.

    A container identifier proves which container a row belongs to, but it does not prove that the
    row contains an equipment-type slot.  Adding a type phrase to an identifier-only line changes
    the template topology.  Exact anchored requirements are built from a same-line or uniquely
    adjacent source equipment surface; anything else remains fail-closed.
    """

    del current_text
    anchored_by_path: dict[str, set[str]] = defaultdict(set)
    for requirement in bundle.anchoredScalarReplacementRequirements:
        for path in requirement.targetPaths:
            if _PRINTED_EQUIPMENT_PATH.fullmatch(path) is not None:
                anchored_by_path[path].update(requirement.sourceLineIds)
    refined: list[HybridWorkItem] = []
    for item in work_items:
        if item.state != "agent_residual" or item.action not in {
            "add_equipment_surface",
            "replace_equipment_surface",
        }:
            refined.append(item)
            continue
        equipment_paths = tuple(
            path for path in item.targetPaths if _PRINTED_EQUIPMENT_PATH.fullmatch(path) is not None
        )
        owned_line_ids = {
            line_id for path in equipment_paths for line_id in anchored_by_path.get(path, ())
        }
        if not owned_line_ids:
            refined.append(
                item.model_copy(
                    update={
                        "state": "blocked_unlocated",
                        "evidenceLineIds": (),
                        "spanIds": (),
                        "locator": "unlocated",
                        "rationale": (
                            "No exact same-line or uniquely adjacent printed equipment surface "
                            "can own this categorical projection."
                        ),
                    }
                )
            )
            continue
        refined.append(
            item.model_copy(
                update={
                    "evidenceLineIds": tuple(sorted(owned_line_ids, key=line_number)),
                    "locator": "relation_scoped_surface",
                    "rationale": (
                        "The projected equipment surface is owned only by its exact printed "
                        "source-equipment line."
                    ),
                }
            )
        )
    return tuple(refined)


def _party_owned_surface_lines(
    current_text: str,
    source_label: Mapping[str, JsonValue],
    surface: str,
) -> set[int]:
    """Locate occurrences owned by a labeled source-party field.

    A route and a party frequently share the same locality. Literal matching therefore cannot
    give a route edit authority over occurrences inside a delivery-agent or other party block,
    regardless of whether that party itself changes in the synthetic target. The party's source
    label plus its bounded role block are sufficient ownership evidence; the target is irrelevant
    to that source-side question.
    """

    source_patch = source_label.get("documentPatch")
    source_parties = source_patch.get("parties") if isinstance(source_patch, Mapping) else None
    if not isinstance(source_parties, Mapping):
        return set()
    normalized_surface = _semantic_normalize(surface)
    protected: set[int] = set()
    for role, source_value in source_parties.items():
        source_rows = (
            source_value
            if role == "notifyParties"
            and isinstance(source_value, Sequence)
            and not isinstance(source_value, (str, bytes))
            else (source_value,)
        )
        for index, source_party in enumerate(source_rows):
            if not isinstance(source_party, Mapping):
                continue
            for field, source_scalar in source_party.items():
                if (
                    not isinstance(source_scalar, str)
                    or not normalized_surface
                    or re.search(
                        rf"(?:^| ){re.escape(normalized_surface)}(?: |$)",
                        _semantic_normalize(source_scalar),
                    )
                    is None
                ):
                    continue
                role_path = (
                    f"documentPatch.parties.notifyParties[{index}]"
                    if role == "notifyParties"
                    else f"documentPatch.parties.{role}"
                )
                if (
                    _semantic_normalize(source_scalar) == normalized_surface
                    or field in {"city", "country"}
                ):
                    for group in _party_block_line_groups(
                        current_text,
                        path=f"{role_path}.{field}",
                        source_label=source_label,
                    ):
                        protected.update(group)
                else:
                    # A token embedded in a larger name/address owns only that complete scalar's
                    # occurrence, not the entire party block. Broad name/address protection can
                    # otherwise swallow unrelated freight/place fields before the next heading.
                    for group in _literal_occurrence_line_sets(current_text, source_scalar):
                        protected.update(group)
    occurrence_lines = {
        number
        for group in _literal_occurrence_line_sets(current_text, surface)
        for number in group
    }
    return protected & occurrence_lines


def _refine_foreign_party_evidence(
    current_text: str,
    source_label: Mapping[str, JsonValue],
    target_label: Mapping[str, JsonValue],
    work_items: Sequence[HybridWorkItem],
) -> tuple[HybridWorkItem, ...]:
    """Prevent a non-party delta from overwriting an unchanged party-role scalar.

    Flattened OCR often repeats a country or city in both a route field and a delivery-agent
    block.  Literal/sibling expansion may attach both lines to the route work item.  An exact
    source-and-target party scalar is stronger role evidence: remove only that scalar's party
    occurrence, retaining every actual route occurrence.  A work item with no remaining evidence
    fails closed.
    """

    refined: list[HybridWorkItem] = []
    for item in work_items:
        if (
            item.state != "agent_residual"
            or any(_party_role_path(path) is not None for path in item.targetPaths)
            or not isinstance(item.sourceValue, str)
        ):
            refined.append(item)
            continue
        protected = _party_owned_surface_lines(
            current_text,
            source_label,
            item.sourceValue,
        )
        contextual = set().union(
            *(
                _contextual_location_line_numbers(current_text, path, item.sourceValue)
                for path in item.targetPaths
            )
        )
        contextual.intersection_update(line_number(value) for value in item.evidenceLineIds)
        # An explicit structural heading is stronger ownership evidence than a shared locality
        # token in another party field. In particular, freight payment and issue locations often
        # repeat the carrier's city. Preserve only their caption-bound occurrences instead of
        # allowing broad party-block discovery to consume those lines.
        if contextual:
            refined.append(
                item.model_copy(
                    update={
                        "evidenceLineIds": tuple(
                            f"L{number:05d}" for number in sorted(contextual)
                        ),
                        "locator": "location_role_surface",
                    }
                )
            )
            continue
        if not protected:
            refined.append(item)
            continue
        evidence = tuple(
            line_id for line_id in item.evidenceLineIds if line_number(line_id) not in protected
        )
        if evidence:
            refined.append(
                item.model_copy(
                    update={
                        "evidenceLineIds": evidence,
                        "rationale": (
                            item.rationale
                            + " Unchanged party-role copies of the same surface are excluded."
                        ),
                    }
                )
            )
        else:
            refined.append(
                item.model_copy(
                    update={
                        "state": "blocked_unlocated",
                        "evidenceLineIds": (),
                        "spanIds": (),
                        "locator": "unlocated",
                        "rationale": (
                            "Every candidate occurrence belongs to an unchanged party-role scalar; "
                            "no path-owned non-party evidence remains."
                        ),
                    }
                )
            )
    return tuple(refined)


_PACKAGE_QUANTITY_PATH = re.compile(
    r"^documentPatch\.(?:cargoPackages\[[0-9]+\]\.quantity|"
    r"cargoAllocationGroups\[[0-9]+\]\.allocations\[[0-9]+\]\.packageQuantity)$"
)
_CARGO_PACKAGE_QUANTITY_PATH = re.compile(
    r"^documentPatch\.cargoPackages\[([0-9]+)\]\.quantity$"
)
_ALLOCATION_PACKAGE_QUANTITY_PATH = re.compile(
    r"^documentPatch\.cargoAllocationGroups\[([0-9]+)\]\.allocations\[([0-9]+)\]"
    r"\.packageQuantity$"
)
_TEMPERATURE_VALUE_PATH = re.compile(
    r"^documentPatch\.containers\[[0-9]+\]\.temperatureSetpoint\.value$"
)
_PACKAGE_NOUN_AFTER_COUNT = re.compile(
    r"(?ix)^[ \t]*(?:X[ \t]+)?"
    r"(?:(?:WOODEN|WOOD|PLYWOOD|CARDBOARD|FIBREBOARD|FIBERBOARD|CORRUGATED|"
    r"PLASTIC|STEEL|METAL)[ \t]+)?(?:"
    r"BAGS?|BALES?|BARRELS?|BOX(?:ES)?|BUNDLES?|CANS?|CARTONS?|CTNS?|CASES?|"
    r"COILS?|CRATES?|CYLINDERS?|DRUMS?|PACKAGES?|PALLETS?|PIECES?|"
    r"PKGS?|PK|PCS?|PLTS?|PL|ROLLS?|SACKS?|SKIDS?|TANKS?|TINS?|UNITS?"
    r")\b"
)


def _is_explicit_package_quantity_line(raw_line: str, value: int | float) -> bool:
    """Recognize a quantity only when its printed grammar identifies package evidence.

    A naked small integer is not package evidence: B/L OCR contains field ordinals, VAT/reference
    components, page numbers, original-count spell-outs, and measurement components.  This
    recognizer grants authority only to a starred count or a count immediately bound to a package
    noun.  It classifies syntax, not package categories or observed-value aliases.
    """

    expected = Decimal(str(value))
    for match in re.finditer(r"(?<![0-9.,])[-+]?\d+(?:[.,]\d+)*(?![0-9.,])", raw_line):
        if expected not in _numeric_surface_values(match.group(0)):
            continue
        following = raw_line[match.end() :]
        if re.match(r"^[ \t]*\*", following) is not None:
            return True
        if _PACKAGE_NOUN_AFTER_COUNT.match(following) is not None:
            return True
    return False


def _sequence_objects(label: Mapping[str, Any], key: str) -> Sequence[Any]:
    patch = label.get("documentPatch")
    value = patch.get(key) if isinstance(patch, Mapping) else None
    return value if isinstance(value, Sequence) and not isinstance(value, (str, bytes)) else ()


def _quantity_group_ids(label: Mapping[str, Any], path: str) -> frozenset[str]:
    package = re.match(r"^documentPatch\.cargoPackages\[([0-9]+)\]\.quantity$", path)
    if package is not None:
        index = int(package.group(1))
        rows = _sequence_objects(label, "cargoPackages")
    else:
        allocation = re.match(
            r"^documentPatch\.cargoAllocationGroups\[([0-9]+)\]\.allocations\[[0-9]+\]"
            r"\.packageQuantity$",
            path,
        )
        if allocation is None:
            return frozenset()
        index = int(allocation.group(1))
        rows = _sequence_objects(label, "cargoAllocationGroups")
    if index >= len(rows) or not isinstance(rows[index], Mapping):
        return frozenset()
    value = rows[index].get("groupId")
    return frozenset((value,)) if isinstance(value, str) and value else frozenset()


def _cargo_group_id(label: Mapping[str, Any], path: str) -> str | None:
    match = re.match(r"^documentPatch\.cargoGroups\[([0-9]+)\]", path)
    if match is None:
        return None
    rows = _sequence_objects(label, "cargoGroups")
    index = int(match.group(1))
    if index >= len(rows) or not isinstance(rows[index], Mapping):
        return None
    value = rows[index].get("groupId")
    return value if isinstance(value, str) and value else None


def _allocation_container_numbers(
    label: Mapping[str, Any], group_ids: frozenset[str]
) -> frozenset[str]:
    output: set[str] = set()
    for group in _sequence_objects(label, "cargoAllocationGroups"):
        if not isinstance(group, Mapping) or group.get("groupId") not in group_ids:
            continue
        allocations = group.get("allocations")
        if not isinstance(allocations, Sequence) or isinstance(allocations, (str, bytes)):
            continue
        for allocation in allocations:
            value = allocation.get("containerNumber") if isinstance(allocation, Mapping) else None
            if isinstance(value, str) and value:
                output.add(value)
    return frozenset(output)


def _allocation_container_for_path(label: Mapping[str, Any], path: str) -> str | None:
    match = _ALLOCATION_PACKAGE_QUANTITY_PATH.fullmatch(path)
    if match is None:
        return None
    groups = _sequence_objects(label, "cargoAllocationGroups")
    group_index, allocation_index = (int(value) for value in match.groups())
    if group_index >= len(groups) or not isinstance(groups[group_index], Mapping):
        return None
    allocations = groups[group_index].get("allocations")
    if (
        not isinstance(allocations, Sequence)
        or isinstance(allocations, (str, bytes))
        or allocation_index >= len(allocations)
        or not isinstance(allocations[allocation_index], Mapping)
    ):
        return None
    value = allocations[allocation_index].get("containerNumber")
    return value if isinstance(value, str) and value else None


def _container_numbers(label: Mapping[str, Any]) -> tuple[str, ...]:
    return tuple(
        value
        for row in _sequence_objects(label, "containers")
        if isinstance(row, Mapping)
        and isinstance((value := row.get("containerNumber")), str)
        and value
    )


def _allocation_quantity_evidence_lines(
    current_text: str,
    *,
    path: str,
    source_value: int | float,
    source_label: Mapping[str, Any],
    target_label: Mapping[str, Any],
    candidate_line_numbers: set[int] | None = None,
) -> set[int]:
    """Bind one allocation quantity to the package rows following its own container.

    Container-size numerals and allocation counts often share values such as 20 or 40.  The
    container relation is the disambiguator: locate the source/target container identifier, bound
    its section by the next known container or page marker, then admit only explicit package-count
    grammar inside that section.  This derives authority from the label graph and printed syntax,
    not from proximity alone.
    """

    bodies = current_text.splitlines()

    def explicit_at(number: int, value: int | float) -> bool:
        if not 1 <= number <= len(bodies):
            return False
        if _is_explicit_package_quantity_line(bodies[number - 1], value):
            return True
        # Columnar OCR can split ``990 BAGS`` into two physical cells. The numeric cell is valid
        # only when it is already relation-owned and the immediately following occupied cell is a
        # package noun. This never promotes arbitrary naked numerals globally.
        tokens = re.findall(r"(?<![0-9.,])[-+]?\d+(?:[.,]\d+)*(?![0-9.,])", bodies[number - 1])
        if (
            len(tokens) != 1
            or Decimal(str(value)) not in _numeric_surface_values(tokens[0])
            or re.sub(r"[-+0-9.,\s]", "", bodies[number - 1])
        ):
            return False
        following = number + 1
        while following <= len(bodies) and not bodies[following - 1].strip():
            following += 1
        return following <= len(bodies) and _PACKAGE_NOUN_AFTER_COUNT.fullmatch(
            bodies[following - 1]
        ) is not None

    if candidate_line_numbers:
        direct = {
            number for number in candidate_line_numbers if explicit_at(number, source_value)
        }
        if len(direct) == 1:
            return direct

    owned_containers = {
        value
        for label in (source_label, target_label)
        if (value := _allocation_container_for_path(label, path)) is not None
    }
    if not owned_containers:
        return set()
    all_containers = set(_container_numbers(source_label)) | set(_container_numbers(target_label))
    all_anchor_lines = sorted(
        {
            number
            for container in all_containers
            for number in _container_identifier_line_numbers(current_text, container)
        }
    )
    owned_anchor_lines = sorted(
        {
            number
            for container in owned_containers
            for number in _container_identifier_line_numbers(current_text, container)
        }
    )
    evidence: set[int] = set()
    for anchor in owned_anchor_lines:
        later_anchors = (number for number in all_anchor_lines if number > anchor)
        boundary = next(later_anchors, len(bodies) + 1)
        for number in range(anchor, boundary):
            if number > anchor and re.fullmatch(r"--- PAGE [1-9][0-9]* ---", bodies[number - 1]):
                break
            if explicit_at(number, source_value):
                evidence.add(number)
    return evidence


def _allocation_quantity_path_rows(
    label: Mapping[str, Any],
) -> tuple[tuple[str, int | float, str], ...]:
    """Return allocation quantities and their containers in stable schema order."""

    output: list[tuple[str, int | float, str]] = []
    for group_index, group in enumerate(_sequence_objects(label, "cargoAllocationGroups")):
        if not isinstance(group, Mapping):
            continue
        allocations = group.get("allocations")
        if not isinstance(allocations, Sequence) or isinstance(allocations, (str, bytes)):
            continue
        for allocation_index, allocation in enumerate(allocations):
            if not isinstance(allocation, Mapping):
                continue
            quantity = allocation.get("packageQuantity")
            container = allocation.get("containerNumber")
            if (
                not isinstance(quantity, (int, float))
                or isinstance(quantity, bool)
                or not isinstance(container, str)
                or not container
            ):
                continue
            output.append(
                (
                    "documentPatch.cargoAllocationGroups"
                    f"[{group_index}].allocations[{allocation_index}].packageQuantity",
                    quantity,
                    container,
                )
            )
    return tuple(output)


def _ordered_allocation_quantity_evidence(
    current_text: str,
    *,
    source_label: Mapping[str, Any],
    target_label: Mapping[str, Any],
    active_paths: frozenset[str],
) -> dict[str, set[int]]:
    """Prove allocation quantity ownership across row-wise and column-wise OCR tables.

    Bills of lading render container allocations in two recurring topologies. Row-wise forms put
    each package count after its container; column-wise forms print every container first and the
    package-count column afterwards. A repeated continuation block can then repeat the counts in
    the same allocation order without repeating the container column. Literal proximity cannot
    represent all three cases.

    This resolver uses only schema relations and explicit package-count grammar. Every active
    allocation must have one current source-or-target container anchor. Equal source quantities
    are assigned either by non-overlapping container sections, when every section has the same
    number of copies, or by complete order-preserving table cycles. Anything incomplete or
    non-monotonic remains unresolved.
    """

    bodies = current_text.splitlines()
    source_rows = tuple(
        row for row in _allocation_quantity_path_rows(source_label) if row[0] in active_paths
    )
    if not source_rows:
        return {}

    target_containers = {
        path: _allocation_container_for_path(target_label, path)
        for path, _quantity, _ in source_rows
    }
    anchor_by_path: dict[str, int] = {}
    previous_anchor = 0
    for path, _quantity, source_container in source_rows:
        containers = {source_container}
        target_container = target_containers[path]
        if target_container is not None:
            containers.add(target_container)
        anchors = {
            number
            for container in containers
            for number in _container_identifier_line_numbers(current_text, container)
        }
        following_anchors = sorted(number for number in anchors if number > previous_anchor)
        if not following_anchors:
            return {}
        anchor_by_path[path] = following_anchors[0]
        previous_anchor = following_anchors[0]

    def explicit_at(number: int, value: int | float) -> bool:
        if not 1 <= number <= len(bodies):
            return False
        if _is_explicit_package_quantity_line(bodies[number - 1], value):
            return True
        tokens = re.findall(
            r"(?<![0-9.,])[-+]?\d+(?:[.,]\d+)*(?![0-9.,])", bodies[number - 1]
        )
        if (
            len(tokens) != 1
            or Decimal(str(value)) not in _numeric_surface_values(tokens[0])
            or re.sub(r"[-+0-9.,\s]", "", bodies[number - 1])
        ):
            return False
        following = number + 1
        while following <= len(bodies) and not bodies[following - 1].strip():
            following += 1
        return following <= len(bodies) and _PACKAGE_NOUN_AFTER_COUNT.fullmatch(
            bodies[following - 1]
        ) is not None

    output: dict[str, set[int]] = defaultdict(set)
    quantities = {quantity for _path, quantity, _container in source_rows}
    for quantity in quantities:
        matching_paths = [path for path, value, _container in source_rows if value == quantity]
        candidates = [
            number
            for number in range(1, len(bodies) + 1)
            if explicit_at(number, quantity)
            and (
                len(matching_paths) == 1
                or not _is_shipment_aggregate_value_line(bodies[number - 1])
            )
        ]
        if not candidates:
            return {}
        if len(matching_paths) == 1:
            output[matching_paths[0]].update(candidates)
            continue

        anchors = [anchor_by_path[path] for path in matching_paths]
        boundaries = [*anchors[1:], len(bodies) + 1]
        row_groups = [
            [number for number in candidates if anchor <= number < boundary]
            for anchor, boundary in zip(anchors, boundaries, strict=True)
        ]
        group_sizes = {len(group) for group in row_groups}
        if group_sizes and 0 not in group_sizes and len(group_sizes) == 1:
            for path, group in zip(matching_paths, row_groups, strict=True):
                output[path].update(group)
            continue

        width = len(matching_paths)
        if len(candidates) < width:
            return {}
        first_cycle = candidates[:width]
        if any(candidate < anchor for candidate, anchor in zip(first_cycle, anchors, strict=True)):
            return {}
        # A later continuation page can repeat only a subset of allocation rows. Map every
        # complete cycle here; path-specific relation evidence below owns any proven remainder.
        for offset in range(0, len(candidates) - width + 1, width):
            cycle = candidates[offset : offset + width]
            for path, number in zip(matching_paths, cycle, strict=True):
                output[path].add(number)
    return dict(output)


def _shipment_package_total_lines(current_text: str, value: int | float) -> set[int]:
    """Locate explicit shipment-level package totals with the requested numeric value."""

    expected = Decimal(str(value))
    output: set[int] = set()
    for number, body in enumerate(current_text.splitlines(), start=1):
        if not _is_shipment_aggregate_value_line(body):
            continue
        tokens = re.findall(r"(?<![0-9.,])[-+]?\d+(?:[.,]\d+)*(?![0-9])", body)
        if any(expected in _numeric_surface_values(token) for token in tokens):
            output.add(number)
    return output


def _refine_numeric_evidence(
    current_text: str,
    work_items: Sequence[HybridWorkItem],
    bundle: RewriteCycleBundle,
) -> tuple[HybridWorkItem, ...]:
    """Remove unrelated numeric collisions from quantity and temperature authority.

    Small package counts such as ``1`` and ``6`` occur throughout bills of lading as dates,
    field ordinals, and legal-clause references. Numeric equality does not make those lines cargo
    evidence. We retain only host-proven cargo-description lines, explicit package-count grammar,
    operational package rows, related container rows, and strict shipment-summary grammars.
    Temperature values are separately limited to lines with thermal context. If none can be
    proven the original evidence remains fail-closed rather than guessing a location.
    """

    bodies = current_text.splitlines()
    active_allocation_paths = frozenset(
        path
        for item in work_items
        if item.state == "agent_residual"
        for path in item.targetPaths
        if _ALLOCATION_PACKAGE_QUANTITY_PATH.fullmatch(path) is not None
    )
    ordered_allocation_evidence = _ordered_allocation_quantity_evidence(
        current_text,
        source_label=bundle.sourceLabel,
        target_label=bundle.targetLabel,
        active_paths=active_allocation_paths,
    )
    refined: list[HybridWorkItem] = []
    for item in work_items:
        quantity_paths = tuple(
            path for path in item.targetPaths if _PACKAGE_QUANTITY_PATH.match(path)
        )
        temperature_paths = tuple(
            path for path in item.targetPaths if _TEMPERATURE_VALUE_PATH.match(path)
        )
        if (
            item.state != "agent_residual"
            or (not quantity_paths and not temperature_paths)
            or not isinstance(item.sourceValue, (int, float))
            or isinstance(item.sourceValue, bool)
        ):
            refined.append(item)
            continue
        owned = {line_number(value) for value in item.evidenceLineIds}
        if temperature_paths:
            allowed_temperature = _temperature_deactivation_line_numbers(
                current_text, {"value": item.sourceValue}
            )
            narrowed_temperature = owned & allowed_temperature
            if not narrowed_temperature:
                refined.append(item)
                continue
            refined.append(
                item.model_copy(
                    update={
                        "evidenceLineIds": tuple(
                            f"L{number:05d}" for number in sorted(narrowed_temperature)
                        ),
                        "locator": "thermal_context_numeric_surface",
                    }
                )
            )
            continue
        allocation_paths = tuple(
            path for path in quantity_paths if _ALLOCATION_PACKAGE_QUANTITY_PATH.fullmatch(path)
        )
        if allocation_paths:
            narrowed_allocations: set[int] = set()
            for path in allocation_paths:
                ordered = ordered_allocation_evidence.get(path, set())
                relation_scoped = _allocation_quantity_evidence_lines(
                    current_text,
                    path=path,
                    source_value=item.sourceValue,
                    source_label=bundle.sourceLabel,
                    target_label=bundle.targetLabel,
                    candidate_line_numbers=owned,
                )
                if ordered:
                    claimed_by_other_paths = set().union(
                        *(
                            lines
                            for candidate_path, lines in ordered_allocation_evidence.items()
                            if candidate_path != path
                        )
                    )
                    relation_scoped.difference_update(claimed_by_other_paths)
                narrowed_allocations.update(ordered | relation_scoped)
            if narrowed_allocations:
                refined.append(
                    item.model_copy(
                        update={
                            "evidenceLineIds": tuple(
                                f"L{number:05d}" for number in sorted(narrowed_allocations)
                            ),
                            "locator": "relation_scoped_numeric_surface",
                        }
                    )
                )
            else:
                refined.append(
                    item.model_copy(
                        update={
                            "state": "blocked_unlocated",
                            "evidenceLineIds": (),
                            "spanIds": (),
                            "locator": "unlocated",
                            "rationale": (
                                "No explicit package quantity is printed inside the allocation's "
                                "container-owned section."
                            ),
                        }
                    )
                )
            continue
        package_paths = tuple(
            path for path in quantity_paths if _CARGO_PACKAGE_QUANTITY_PATH.fullmatch(path)
        )
        shipment_totals = (
            _shipment_package_total_lines(current_text, item.sourceValue)
            if package_paths
            else set()
        )
        if shipment_totals:
            refined.append(
                item.model_copy(
                    update={
                        "evidenceLineIds": tuple(
                            f"L{number:05d}" for number in sorted(shipment_totals)
                        ),
                        "locator": "relation_scoped_numeric_surface",
                    }
                )
            )
            continue
        group_ids = frozenset().union(
            *(
                _quantity_group_ids(bundle.sourceLabel, path)
                | _quantity_group_ids(bundle.targetLabel, path)
                for path in quantity_paths
            )
        )
        allowed: set[int] = set()
        for cargo_requirement in bundle.cargoFlavorRewriteRequirements:
            group_id = _cargo_group_id(
                bundle.sourceLabel, cargo_requirement.targetPath
            ) or _cargo_group_id(bundle.targetLabel, cargo_requirement.targetPath)
            if not group_ids or group_id in group_ids:
                allowed.update(line_number(value) for value in cargo_requirement.sourceLineIds)
        related_containers = _allocation_container_numbers(
            bundle.sourceLabel, group_ids
        ) | _allocation_container_numbers(bundle.targetLabel, group_ids)
        for container_number in related_containers:
            located, _ = _literal_line_numbers(current_text, container_number)
            allowed.update(located)
        for operational_requirement in bundle.operationalFlavorRequirements:
            if operational_requirement.kind != "package_quantity":
                continue
            if (
                not related_containers
                or operational_requirement.targetContainerNumber in related_containers
            ):
                allowed.add(line_number(operational_requirement.sourceLineId))
        allowed.update(
            number
            for number in owned
            if 1 <= number <= len(bodies)
            and _is_explicit_package_quantity_line(bodies[number - 1], item.sourceValue)
        )
        allowed.update(
            number
            for number in owned
            if 1 <= number <= len(bodies) and _is_shipment_aggregate_value_line(bodies[number - 1])
        )
        narrowed = owned & allowed
        if not narrowed:
            refined.append(item)
            continue
        refined.append(
            item.model_copy(
                update={
                    "evidenceLineIds": tuple(f"L{number:05d}" for number in sorted(narrowed)),
                    "locator": "relation_scoped_numeric_surface",
                }
            )
        )
    return tuple(refined)


def _party_scalar_source_line_groups(
    *,
    source_text: str,
    source_label: Mapping[str, JsonValue],
    path: str,
    work_item: HybridWorkItem,
) -> tuple[frozenset[int], ...]:
    """Locate each printed copy of one party scalar inside its proven role blocks.

    Party blocks are deliberately broad enough to recover wrapped OCR, but they are not scalar
    ownership.  A city token can occur inside the company name, an address can wrap over several
    lines, and the same party block can be repeated on several pages.  This helper turns that
    block-level evidence into exact, independent scalar groups and is shared by both the edit
    authority and the model-facing topology contract.
    """

    owned_numbers = {line_number(value) for value in work_item.evidenceLineIds}
    if not owned_numbers or not isinstance(work_item.sourceValue, str):
        return ()

    role_groups = _party_block_line_groups(
        source_text,
        path=path,
        source_label=source_label,
    )
    owned_role_groups = tuple(
        group & owned_numbers for group in role_groups if group & owned_numbers
    )
    source_lines = source_text.splitlines()
    heading_numbers = {
        number
        for number in owned_numbers
        if 1 <= number <= len(source_lines)
        and _PARTY_HEADING_LINE.fullmatch(source_lines[number - 1]) is not None
    }
    candidate_groups = [
        occurrence
        for occurrence in _literal_occurrence_line_sets(source_text, work_item.sourceValue)
        if occurrence <= owned_numbers
        and occurrence.isdisjoint(heading_numbers)
        and (not owned_role_groups or any(occurrence <= group for group in owned_role_groups))
    ]
    normalized_source = _semantic_normalize(work_item.sourceValue)
    for number, line in enumerate(source_text.splitlines(), start=1):
        occurrence = frozenset((number,))
        if (
            number in owned_numbers
            and number not in heading_numbers
            and normalized_source
            and normalized_source in _semantic_normalize(line)
            and occurrence not in candidate_groups
            and (
                not owned_role_groups
                or any(occurrence <= group for group in owned_role_groups)
            )
        ):
            candidate_groups.append(occurrence)

    field_name = path.rsplit(".", 1)[-1]
    resolved_party = _source_party(source_label, path)
    if candidate_groups and field_name in {"city", "country"} and resolved_party is not None:
        _role, party = resolved_party
        source_name = party.get("name")
        name_lines = (
            set().union(
                *(
                    occurrence
                    for occurrence in _literal_occurrence_line_sets(source_text, source_name)
                    if occurrence <= owned_numbers
                )
            )
            if isinstance(source_name, str) and source_name.strip()
            else set()
        )
        selected: list[frozenset[int]] = []
        for group in owned_role_groups or (frozenset(owned_numbers),):
            matches = [
                occurrence
                for occurrence in candidate_groups
                if occurrence <= group and not occurrence <= name_lines
            ]
            if matches:
                # Party locality is conventionally printed at the end of its block.  Selecting
                # the last non-name occurrence prevents a locality repeated inside an address
                # from being treated as another independent city/country copy.
                selected.append(max(matches, key=lambda value: (max(value), min(value))))
        if selected:
            candidate_groups = selected

    if not candidate_groups and field_name == "address":
        # Labels normalize wrapped addresses, while OCR can repeat a city token or attach the
        # postal code to the country line. Find the shortest contiguous span whose token multiset
        # covers the complete labeled source address. This is exact coverage, not fuzzy matching.
        source_tokens = Counter(_semantic_normalize(work_item.sourceValue).split())
        source_lines = source_text.splitlines()
        for group in owned_role_groups:
            role_numbers = sorted(group)
            best: tuple[int, int] | None = None
            for start_index, start in enumerate(role_numbers):
                observed: Counter[str] = Counter()
                previous = start - 1
                for end in role_numbers[start_index:]:
                    if end != previous + 1:
                        break
                    observed.update(_semantic_normalize(source_lines[end - 1]).split())
                    previous = end
                    if observed >= source_tokens:
                        candidate = (start, end)
                        if best is None or (end - start, start) < (
                            best[1] - best[0],
                            best[0],
                        ):
                            best = candidate
                        break
            if best is not None:
                candidate_groups.append(frozenset(range(best[0], best[1] + 1)))

    if not candidate_groups:
        candidate_groups.extend(
            (role_group & owned_numbers) - heading_numbers
            for role_group in owned_role_groups
            if (role_group & owned_numbers) - heading_numbers
        )
    if not candidate_groups:
        contiguous: list[int] = []
        for number in sorted(owned_numbers - heading_numbers):
            if contiguous and number != contiguous[-1] + 1:
                candidate_groups.append(frozenset(contiguous))
                contiguous = []
            contiguous.append(number)
        if contiguous:
            candidate_groups.append(frozenset(contiguous))

    return tuple(
        sorted(
            set(candidate_groups),
            key=lambda numbers: tuple(sorted(numbers)),
        )
    )


def _refine_party_evidence(
    source_text: str,
    source_label: Mapping[str, JsonValue],
    work_items: Sequence[HybridWorkItem],
    surface_requirements: Sequence[SurfaceRenderingRequirement] = (),
) -> tuple[HybridWorkItem, ...]:
    """Reduce broad party-block authority to scalar-specific printed line groups."""

    refined: list[HybridWorkItem] = []
    for item in work_items:
        party_paths = tuple(
            path for path in item.targetPaths if _party_role_path(path) is not None
        )
        if item.locator not in {
            "party_role_block",
            "rendered_surface_and_party_role_block",
        } or not isinstance(item.sourceValue, str) or not party_paths:
            refined.append(item)
            continue
        groups = tuple(
            group
            for path in party_paths
            for group in _party_scalar_source_line_groups(
                source_text=source_text,
                source_label=source_label,
                path=path,
                work_item=item,
            )
        )
        narrowed = set().union(*groups) if groups else set()
        if not narrowed:
            refined.append(item)
            continue
        _located, locator = _literal_line_numbers(source_text, item.sourceValue)
        refined.append(
            item.model_copy(
                update={
                    "evidenceLineIds": tuple(f"L{number:05d}" for number in sorted(narrowed)),
                    "locator": locator,
                }
            )
        )
    return tuple(refined)


def _iter_string_label_leaves(
    value: Any,
    *,
    path: str = "",
) -> Sequence[tuple[str, str]]:
    """Return string leaves with stable schema paths for source-side ownership checks."""

    rows: list[tuple[str, str]] = []
    if isinstance(value, Mapping):
        for key, child in value.items():
            child_path = f"{path}.{key}" if path else str(key)
            rows.extend(_iter_string_label_leaves(child, path=child_path))
    elif isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        for index, child in enumerate(value):
            rows.extend(_iter_string_label_leaves(child, path=f"{path}[{index}]"))
    elif isinstance(value, str):
        rows.append((path, value))
    return tuple(rows)


def _carrier_embedded_foreign_owner_groups(
    original_text: str,
    source_label: Mapping[str, Any] | None,
    carrier_surface: str,
) -> tuple[frozenset[int], ...]:
    """Locate a carrier token embedded in a different complete labeled source scalar.

    A vessel such as ``CMA CGM MOLIERE`` contains the carrier ``CMA CGM`` but is not another
    carrier-principal slot. The complete source scalar and its distinct schema path prove that
    ownership without relying on vessel-name regexes. Exact-equal values remain unresolved here
    because their individual occurrences require a stronger role locator.
    """

    if source_label is None:
        return ()
    normalized_carrier = _semantic_normalize(carrier_surface)
    if not normalized_carrier:
        return ()
    groups: set[frozenset[int]] = set()
    for path, value in _iter_string_label_leaves(source_label):
        if path.startswith("documentPatch.parties.carrier."):
            continue
        normalized_value = _semantic_normalize(value)
        if normalized_value == normalized_carrier or normalized_carrier not in normalized_value:
            continue
        groups.update(_party_scalar_occurrence_line_sets(original_text, value))
    return tuple(sorted(groups, key=lambda group: (min(group), len(group))))


def _refine_party_occurrence_requirements(
    original_text: str,
    current_text: str,
    work_items: Sequence[HybridWorkItem],
    requirements: Sequence[TargetValueOccurrenceRequirement],
    source_label: Mapping[str, Any] | None = None,
) -> tuple[TargetValueOccurrenceRequirement, ...]:
    """Count party identity copies only inside their proven semantic role blocks.

    The baseline occurrence guard is intentionally global because it predates the inventory
    compiler. Here every party work item already has role-owned evidence. Recomputing name and
    address cardinality within that evidence prevents websites, marks, signing affiliates, and
    legal carrier references from demanding extra copies of the task-label identity, while still
    preserving genuinely repeated role blocks across pages.
    """

    bodies = current_text.splitlines()
    refined: list[TargetValueOccurrenceRequirement] = []
    for requirement in requirements:
        party_paths = tuple(
            path
            for path in requirement.targetPaths
            if _party_role_path(path) is not None and path.rsplit(".", 1)[-1] in {"name", "address"}
        )
        if not party_paths:
            refined.append(requirement)
            continue
        contributions: list[int] = []
        for path in party_paths:
            matching = tuple(item for item in work_items if path in item.targetPaths)
            if not matching:
                contributions = []
                break
            count = 0
            for item in matching:
                if not isinstance(item.sourceValue, str):
                    continue
                line_numbers = sorted(
                    {
                        line_number(value)
                        for value in item.evidenceLineIds
                        if 1 <= line_number(value) <= len(bodies)
                    }
                )
                if path == "documentPatch.parties.carrier.name":
                    # Carrier principals legitimately occur in mastheads, tariff branding, and
                    # signatures outside one form block. Count both already rendered target
                    # principals and source principals still awaiting a residual rewrite. This
                    # matters when a short carrier masthead is deterministically expanded to the
                    # full target name before a separate signature line is delegated.
                    excluded_groups = _carrier_embedded_foreign_owner_groups(
                        original_text,
                        source_label,
                        item.sourceValue,
                    )
                    target_count = carrier_principal_occurrence_count(
                        current_text,
                        requirement.targetValue,
                        excluded_line_groups=excluded_groups,
                    )
                    source_count = carrier_principal_occurrence_count(
                        current_text,
                        item.sourceValue,
                        excluded_line_groups=excluded_groups,
                    )
                    source_inside_target = carrier_principal_occurrence_count(
                        requirement.targetValue, item.sourceValue
                    )
                    count = max(
                        count,
                        carrier_principal_template_slot_count(
                            original_text,
                            item.sourceValue,
                            excluded_line_groups=excluded_groups,
                        ),
                        target_count
                        + max(0, source_count - target_count * source_inside_target),
                    )
                else:
                    role_text = "\n".join(bodies[number - 1] for number in line_numbers)
                    target_count = _party_scalar_occurrence_count(
                        role_text, requirement.targetValue
                    )
                    source_count = _party_scalar_occurrence_count(role_text, item.sourceValue)
                    source_inside_target = _party_scalar_occurrence_count(
                        requirement.targetValue, item.sourceValue
                    )
                    count = max(
                        count,
                        target_count
                        + max(0, source_count - target_count * source_inside_target),
                    )
            if count < 1:
                contributions = []
                break
            contributions.append(count)
        if not contributions:
            refined.append(requirement)
            continue
        refined.append(requirement.model_copy(update={"requiredOccurrences": sum(contributions)}))
    return tuple(refined)


def _include_unique_linked_party_copies(
    current_text: str,
    work_items: Sequence[HybridWorkItem],
    inventory: Sequence[InventoryCandidate],
    requirements: Sequence[TargetValueOccurrenceRequirement],
    source_status_requirements: Sequence[SourceStatusPreservationRequirement],
) -> tuple[TargetValueOccurrenceRequirement, ...]:
    """Count exact party copies outside their primary form blocks when their target is unique.

    The role-block refinement deliberately excludes unowned lines. That is correct for explicit
    signing agents, which are parsed separately, but not for an exact shipper/consignee identity
    repeated in a customs or tax statement. The inventory has already removed explicit legal-agent
    lines and proven that a remaining copy has one semantic target. Add only those proven copies to
    the global target cardinality so the renderer cannot replace them with an unrelated company.
    """

    bodies = current_text.splitlines()
    protected_status_lines = {
        f"L{number:05d}"
        for number, body in enumerate(bodies, start=1)
        if any(body == requirement.sourceSurface for requirement in source_status_requirements)
    }
    work_items_by_line: dict[str, list[HybridWorkItem]] = defaultdict(list)
    for item in work_items:
        for line_id in item.evidenceLineIds:
            work_items_by_line[line_id].append(item)
    output: list[TargetValueOccurrenceRequirement] = []
    for requirement in requirements:
        party_paths = {
            path
            for path in requirement.targetPaths
            if _party_role_path(path) is not None
            and path.rsplit(".", 1)[-1] in {"name", "address"}
        }
        # Carrier principal cardinality already includes legal-syntax and unowned branding slots.
        if not party_paths or "documentPatch.parties.carrier.name" in party_paths:
            output.append(requirement)
            continue
        extra_occurrences = 0
        for candidate in inventory:
            if candidate.category != "changed_source_occurrence":
                continue
            target_rows = candidate.targetSemantics
            if not isinstance(target_rows, Sequence) or isinstance(target_rows, (str, bytes)):
                continue
            linked_paths = {
                row.get("path")
                for row in target_rows
                if isinstance(row, Mapping)
                and row.get("target") == requirement.targetValue
                and row.get("path") in party_paths
            }
            if not linked_paths:
                continue
            # A punctuation-equivalent party name can itself be the value of another labeled
            # field, such as marks and numbers.  The explicit path owner wins on that line; it is
            # not an extra party occurrence.  The inventory compiler normally suppresses this
            # collision, while this check keeps cardinality safe for externally supplied or
            # persisted inventories as well.
            candidate_key = _semantic_normalize(candidate.sourceSurface)
            foreign_owned_lines = {
                line_id
                for line_id in candidate.lineIds
                if any(
                    not linked_paths.intersection(item.targetPaths)
                    and isinstance(item.sourceValue, str)
                    and _semantic_normalize(item.sourceValue) == candidate_key
                    for item in work_items_by_line.get(line_id, ())
                )
            }
            owned_lines = {
                line_id
                for item in work_items
                if item.sourceValue == candidate.sourceSurface
                and linked_paths.intersection(item.targetPaths)
                for line_id in item.evidenceLineIds
            }
            unowned_lines = tuple(
                line_id
                for line_id in candidate.lineIds
                if line_id not in owned_lines
                and line_id not in foreign_owned_lines
                and line_id not in protected_status_lines
            )
            if not unowned_lines:
                continue
            selected = "\n".join(
                bodies[line_number(line_id) - 1] for line_id in unowned_lines
            )
            count = _party_scalar_occurrence_count(selected, candidate.sourceSurface)
            if count < 1:
                raise ValueError(
                    "linked party inventory candidate has no countable source occurrence: "
                    f"{candidate.candidateId}"
                )
            extra_occurrences += count
        output.append(
            requirement.model_copy(
                update={
                    "requiredOccurrences": requirement.requiredOccurrences + extra_occurrences
                }
            )
        )
    # A short carrier principal can be a proper substring of another labeled party identity,
    # such as ``TRANSGLORY`` inside forwarding agent ``TRANSGLORY S.A.``.  The carrier role
    # refinement correctly excludes those foreign-owner lines when counting principal slots, but
    # the global target counter necessarily sees the substring in the rendered forwarding name.
    # Add those structurally required nested occurrences to the carrier cardinality.
    nested_adjusted: list[TargetValueOccurrenceRequirement] = []
    for requirement in output:
        if "documentPatch.parties.carrier.name" not in requirement.targetPaths:
            nested_adjusted.append(requirement)
            continue
        nested_occurrences = sum(
            other.requiredOccurrences
            * _party_scalar_occurrence_count(other.targetValue, requirement.targetValue)
            for other in output
            if other is not requirement
            and any(
                path.startswith("documentPatch.parties.")
                and path.rsplit(".", 1)[-1] in {"name", "address"}
                for path in other.targetPaths
            )
        )
        nested_adjusted.append(
            requirement.model_copy(
                update={
                    "requiredOccurrences": requirement.requiredOccurrences + nested_occurrences
                }
            )
        )
    return tuple(nested_adjusted)


def _inventory_requirement(item: InventoryCandidate) -> dict[str, JsonValue]:
    return {
        "kind": item.category,
        "candidateId": item.candidateId,
        "paths": cast(JsonValue, list(item.targetPaths)),
        "source": item.sourceSurface,
        "target": item.targetSemantics,
        "rationale": item.rationale,
    }


def _locked_literal_requirements_by_line(
    workspace: RewriteWorkspace,
) -> dict[str, list[dict[str, JsonValue]]]:
    """Expose exact host-rendered literals that a residual line must not rewrite.

    Deterministic prefills run before the linguistic renderer.  A party or cargo line can still
    require reflow for another field, so omitting these literals from the model contract invites
    it to invent a second phone, seal, identifier, date, or measurement.  Locks add no write
    authority: they only constrain lines already selected by a genuine residual requirement.
    """

    locked: dict[str, list[dict[str, JsonValue]]] = defaultdict(list)
    for prefill in workspace.deterministic_prefills:
        locked[prefill.lineId].append(
            {
                "kind": "host_locked_target_literal",
                "paths": cast(JsonValue, list(prefill.targetPaths)),
                "target": prefill.targetSurface,
                "policy": "preserve_verbatim_on_this_line",
            }
        )
    for role_hint in workspace.source_role_hints:
        locked[role_hint.sourceLineId].append(
            {
                "kind": "host_locked_source_literal",
                "paths": cast(JsonValue, list(role_hint.forbiddenTargetPathPrefixes)),
                "target": role_hint.requiredOutputSurface,
                "policy": "preserve_exact_flattened_ocr_semantic_role_on_this_line",
            }
        )
    current_lines = workspace.current_text.splitlines()
    for requirement in workspace.raw_auxiliary_identity_requirements:
        match = re.search(r"L[0-9]{5}", requirement.requirementId)
        if match is None:
            raise ValueError(
                "raw auxiliary identity requirement lacks its source line ID: "
                f"{requirement.requirementId}"
            )
        start = line_number(match.group(0))
        source_fragments = requirement.sourceIdentity.splitlines()
        if len(source_fragments) != requirement.sourceIdentityLineCount:
            raise ValueError(
                "raw auxiliary identity source fragments differ from line-count contract: "
                f"{requirement.requirementId}"
            )
        for offset, source_fragment in enumerate(source_fragments):
            number = start + offset
            if not 1 <= number <= len(current_lines):
                raise ValueError(
                    "raw auxiliary identity line is outside OCR: "
                    f"{requirement.requirementId}"
                )
            current_line = current_lines[number - 1]
            if current_line.count(source_fragment) != 1:
                raise ValueError(
                    "raw auxiliary identity fragment is not unique on its exact line: "
                    f"{requirement.requirementId}, L{number:05d}"
                )
            column = current_line.index(source_fragment)
            prefix = current_line[:column]
            suffix = current_line[column + len(source_fragment) :]
            prefix_lock = prefix
            shipped_prefix = re.match(
                r"(?i)^[ \t]*SHIPPED[ \t]+ON[ \t]+BOARD(?:ED)?"
                r"(?:[ \t]*[:.-][ \t]*|[ \t]+)",
                prefix,
            )
            if shipped_prefix is not None:
                # Vessel and date values between this static heading and the raw agent are
                # independently mutable. Lock only the template heading, not those source facts.
                prefix_lock = shipped_prefix.group(0)
            if prefix_lock:
                locked[f"L{number:05d}"].append(
                    {
                        "kind": "host_locked_source_prefix",
                        "paths": cast(
                            JsonValue,
                            [f"rawAuxiliaryIdentity.{requirement.requirementId}"],
                        ),
                        "target": prefix_lock,
                        "policy": "preserve_verbatim_at_line_start",
                    }
                )
            if suffix:
                locked[f"L{number:05d}"].append(
                    {
                        "kind": "host_locked_source_suffix",
                        "paths": cast(
                            JsonValue,
                            [f"rawAuxiliaryIdentity.{requirement.requirementId}"],
                        ),
                        "target": suffix,
                        "policy": "preserve_verbatim_at_line_end",
                    }
                )
    return locked


def _party_scalar_source_slot_groups(
    compiled: _CompiledCase,
    *,
    path: str,
    work_item: HybridWorkItem,
    alias_by_line: Mapping[str, str],
) -> tuple[tuple[str, ...], ...]:
    """Map the shared scalar-specific line groups into provider slot aliases."""

    candidate_groups = _party_scalar_source_line_groups(
        source_text=compiled.workspace.original_text,
        source_label=compiled.workspace.source_label,
        path=path,
        work_item=work_item,
    )

    groups: set[tuple[str, ...]] = set()
    line_number_by_alias = {
        alias: line_number(line_id) for line_id, alias in alias_by_line.items()
    }
    for candidate_numbers in candidate_groups:
        line_ids = tuple(f"L{number:05d}" for number in sorted(candidate_numbers))
        # A repair request may intentionally expose only one surplus occurrence. Publishing a
        # partial multiline address as a complete source group tells the model to collapse the
        # full target address into that single line. Only complete groups are authoritative.
        if not all(line_id in alias_by_line for line_id in line_ids):
            continue
        aliases = tuple(alias_by_line[line_id] for line_id in line_ids)
        if aliases:
            groups.add(aliases)
    return tuple(
        sorted(
            groups,
            key=lambda aliases: tuple(line_number_by_alias[value] for value in aliases),
        )
    )


def _minimal_slot_groups(groups: set[tuple[str, ...]]) -> tuple[tuple[str, ...], ...]:
    """Discard role-block supersets when an exact scalar occurrence is also available."""

    ordered = sorted(
        groups,
        key=lambda values: (len(values), tuple(int(value.removeprefix("s")) for value in values)),
    )
    retained: list[tuple[str, ...]] = []
    for group in ordered:
        group_set = set(group)
        if any(set(candidate) < group_set for candidate in ordered):
            continue
        retained.append(group)
    return tuple(
        sorted(
            retained,
            key=lambda values: tuple(int(value.removeprefix("s")) for value in values),
        )
    )


def _handling_affix_requirements_by_line(
    workspace: RewriteWorkspace,
    work_items: Sequence[HybridWorkItem],
) -> dict[str, list[dict[str, JsonValue]]]:
    """Protect formatting/legal text adjacent to a changed handling instruction.

    A label-owned handling value may occupy only the beginning or middle of an OCR line.  The
    remaining prefix/suffix is template text, not creative write authority.  In particular, a
    legal suffix such as ``CARRIAGE PER CLAUSE 11`` must survive a handling-value substitution.
    """

    locked: dict[str, list[dict[str, JsonValue]]] = defaultdict(list)
    current_lines = workspace.current_text.splitlines()
    for item in work_items:
        if item.state != "agent_residual" or not isinstance(item.sourceValue, str):
            continue
        if not any(".handlingInstructions[" in path for path in item.targetPaths):
            continue
        for line_id in item.evidenceLineIds:
            number = line_number(line_id)
            if not 1 <= number <= len(current_lines):
                continue
            line = current_lines[number - 1]
            if line.count(item.sourceValue) != 1:
                continue
            start = line.index(item.sourceValue)
            end = start + len(item.sourceValue)
            for position, fragment in (("prefix", line[:start]), ("suffix", line[end:])):
                if not fragment:
                    continue
                locked[line_id].append(
                    {
                        "kind": "host_locked_source_literal",
                        "paths": cast(JsonValue, list(item.targetPaths)),
                        "target": fragment,
                        "position": position,
                        "policy": "preserve_verbatim_on_this_line",
                    }
                )
    return locked


def _model_slots(
    compiled: _CompiledCase,
    inventory: Sequence[InventoryCandidate],
    deterministic_edits: Sequence[DeterministicInventoryEdit] = (),
) -> tuple[_Slot, ...]:
    requirements_by_line: dict[str, list[dict[str, JsonValue]]] = defaultdict(list)
    current_lines = compiled.workspace.current_text.splitlines()
    for work_item in compiled.work_items:
        if work_item.state != "agent_residual":
            continue
        requirement = _work_item_requirement(work_item)
        for value in work_item.evidenceLineIds:
            requirements_by_line[value].append(requirement)
    for inventory_item in inventory:
        if inventory_item.disposition != "model_residual":
            continue
        requirement = _inventory_requirement(inventory_item)
        for value in inventory_item.lineIds:
            requirements_by_line[value].append(requirement)
    for surface_requirement in compiled.bundle.surfaceRenderingRequirements:
        surface_lines = _context_bound_surface_lines(
            compiled.workspace.original_text, surface_requirement
        )
        for number in sorted(surface_lines):
            line_id = f"L{number:05d}"
            if line_id not in requirements_by_line:
                continue
            requirements_by_line[line_id].append(
                {
                    "kind": "required_rendered_surface",
                    "paths": cast(JsonValue, [surface_requirement.targetPath]),
                    "source": surface_requirement.sourceSurface,
                    "target": surface_requirement.targetSurface,
                    "policy": "render_exactly_on_this_path_owned_line",
                }
            )
    for jurisdiction_requirement in compiled.bundle.jurisdictionalSurfaceRequirements:
        for line_id in jurisdiction_requirement.sourceLineIds:
            if line_id not in requirements_by_line:
                continue
            requirements_by_line[line_id].append(
                {
                    "kind": "jurisdictional_surface",
                    "paths": cast(
                        JsonValue,
                        [f"rawJurisdiction.{jurisdiction_requirement.programId}"],
                    ),
                    "source": jurisdiction_requirement.sourceSurface,
                    "target": jurisdiction_requirement.targetSurface,
                    "sourceJurisdictionCountryCode": (
                        jurisdiction_requirement.programJurisdictionCountryCode
                    ),
                    "targetRouteCountryCode": jurisdiction_requirement.targetRouteCountryCode,
                    "policy": (
                        "rewrite_every_program_specific_assertion_on_this_line_into_a_"
                        "route_neutral_customs_reference_while_preserving_unrelated_text"
                    ),
                }
            )
    for cargo_requirement in compiled.bundle.cargoFlavorRewriteRequirements:
        for line_id in cargo_requirement.sourceLineIds:
            number = line_number(line_id)
            if not 1 <= number <= len(current_lines):
                continue
            had_owned_slot = line_id in requirements_by_line
            unexpected_packages = unexpected_cargo_package_surfaces(
                current_lines[number - 1], cargo_requirement
            )
            if unexpected_packages:
                requirements_by_line[line_id].append(
                    {
                        "kind": "source_only_cargo_packaging",
                        "paths": cast(JsonValue, [cargo_requirement.targetPath]),
                        "source": cast(JsonValue, list(unexpected_packages)),
                        "target": {
                            "description": cargo_requirement.targetDescription,
                            "allowedPackageSurfaces": cast(
                                JsonValue,
                                list(cargo_requirement.allowedPackageSurfaces),
                            ),
                        },
                        "policy": (
                            "rewrite_as_a_realistic_target_cargo_detail_using_only_allowed_"
                            "package_surfaces_while_preserving_host_locked_literals"
                        ),
                    }
                )
            if had_owned_slot and len(cargo_requirement.sourceLineIds) == 1:
                requirements_by_line[line_id].append(
                    {
                        "kind": "required_semantic_surface",
                        "paths": cast(JsonValue, [cargo_requirement.targetPath]),
                        "target": cargo_requirement.targetDescription,
                        "policy": "render_complete_target_modulo_casing_on_this_line",
                    }
                )
    for edit in deterministic_edits:
        for line_id in edit.lineIds:
            if line_id not in requirements_by_line:
                continue
            requirements_by_line[line_id].append(
                {
                    "kind": "host_locked_target_literal",
                    "paths": cast(JsonValue, ["rawAuxiliary.deterministicEdit"]),
                    "target": edit.targetSurface,
                    "policy": "preserve_verbatim_on_this_line",
                }
            )
    for line_id in tuple(requirements_by_line):
        number = line_number(line_id)
        if not 1 <= number <= len(current_lines):
            continue
        for label_surface in auxiliary_label_surfaces(current_lines[number - 1]):
            requirements_by_line[line_id].append(
                {
                    "kind": "host_locked_source_literal",
                    "paths": cast(JsonValue, ["rawTemplate.fieldLabel"]),
                    "target": label_surface,
                    "policy": "preserve_verbatim_on_this_line",
                }
            )
    locked_by_line = _locked_literal_requirements_by_line(compiled.workspace)
    handling_locks_by_line = _handling_affix_requirements_by_line(
        compiled.workspace, compiled.work_items
    )
    for value in tuple(requirements_by_line):
        requirements_by_line[value].extend(locked_by_line.get(value, ()))
        requirements_by_line[value].extend(handling_locks_by_line.get(value, ()))
    slots: list[_Slot] = []
    for ordinal, (line_id, requirements) in enumerate(
        sorted(requirements_by_line.items(), key=lambda row: line_number(row[0]))
    ):
        number = line_number(line_id)
        if not 1 <= number <= len(current_lines):
            raise ValueError(f"model slot line is outside the OCR text: {line_id}")
        source_line = current_lines[number - 1]
        if not source_line.strip() or source_line.startswith("--- PAGE "):
            raise ValueError(
                "model slot cannot own a blank/page-marker line: "
                f"document={compiled.bundle.result.documentId} line={line_id} "
                f"requirements={requirements!r}"
            )
        unique: dict[str, dict[str, JsonValue]] = {}
        for requirement in requirements:
            digest = sha256_bytes(canonical_json_bytes(requirement))
            unique[digest] = requirement
        slots.append(
            _Slot(
                alias=f"s{ordinal}",
                line_id=line_id,
                source_line=source_line,
                requirements=tuple(unique[key] for key in sorted(unique)),
            )
        )
    if not slots:
        raise ValueError("inventory probe has no model residual slots")
    return tuple(slots)


def _compound_slots(compiled: _CompiledCase) -> tuple[_CompoundSlot, ...]:
    return tuple(
        _CompoundSlot(alias=f"c{ordinal}", requirement=requirement)
        for ordinal, requirement in enumerate(compiled.bundle.compoundPartyFlavorRequirements)
    )


def _initial_request_batches(
    material: _InventoryCaseMaterial,
    *,
    maximum_slots: int | None,
) -> tuple[_InitialRequestBatch, ...]:
    """Partition a large initial edit without splitting semantic dependencies.

    Every line that jointly realizes one changed field, party role, cargo phrase, or repeated
    auxiliary identity is connected before packing.  The returned batches are only transport
    partitions: their responses are assembled and then applied/audited once, atomically.
    """

    slots = material.slots
    compounds = material.compound_slots
    if maximum_slots is None or len(slots) <= maximum_slots:
        return (_InitialRequestBatch(slots=slots, compound_slots=compounds),)

    line_to_index = {slot.line_id: index for index, slot in enumerate(slots)}
    parent = list(range(len(slots)))

    def find(index: int) -> int:
        while parent[index] != index:
            parent[index] = parent[parent[index]]
            index = parent[index]
        return index

    def union(indices: Sequence[int]) -> None:
        if not indices:
            return
        root = find(indices[0])
        for index in indices[1:]:
            other = find(index)
            if root != other:
                parent[other] = root

    dependency_indices: dict[tuple[str, str], set[int]] = defaultdict(set)
    for index, slot in enumerate(slots):
        for requirement in slot.requirements:
            work_item_id = requirement.get("workItemId")
            if isinstance(work_item_id, str):
                dependency_indices[("work_item", work_item_id)].add(index)
            candidate_id = requirement.get("candidateId")
            if isinstance(candidate_id, str):
                dependency_indices[("inventory_candidate", candidate_id)].add(index)
            paths = requirement.get("paths")
            if isinstance(paths, list):
                for path in paths:
                    if not isinstance(path, str):
                        continue
                    if (role := _party_role_path(path)) is not None:
                        dependency_indices[("party_role", role)].add(index)
    for item in material.compiled.work_items:
        indices = [
            line_to_index[line_id]
            for line_id in item.evidenceLineIds
            if line_id in line_to_index
        ]
        union(indices)
    for requirement in material.compiled.workspace.cargo_flavor_rewrite_requirements:
        union(
            [
                line_to_index[line_id]
                for line_id in requirement.sourceLineIds
                if line_id in line_to_index
            ]
        )
    auxiliary_groups: dict[str, set[int]] = defaultdict(set)
    for requirement in material.compiled.workspace.raw_auxiliary_identity_requirements:
        match = re.search(r"L[0-9]{5}", requirement.requirementId)
        if match is None:
            raise ValueError(
                "raw auxiliary identity requirement lacks its source line ID: "
                f"{requirement.requirementId}"
            )
        start = line_number(match.group(0))
        for number in range(start, start + requirement.sourceIdentityLineCount):
            if (index := line_to_index.get(f"L{number:05d}")) is not None:
                auxiliary_groups[requirement.consistencyGroupId].add(index)
    for indices in (*dependency_indices.values(), *auxiliary_groups.values()):
        union(sorted(indices))

    compound_indices: dict[str, tuple[int, ...]] = {}
    for compound in compounds:
        target_path = compound.requirement.targetPath
        target_role = _party_role_path(target_path)
        indices = tuple(
            index
            for index, slot in enumerate(slots)
            if target_path in _slot_requirement_paths(slot)
            or (
                target_role is not None
                and any(
                    _party_role_path(path) == target_role
                    for path in _slot_requirement_paths(slot)
                )
            )
        )
        if not indices:
            raise ValueError(
                "compound-party realization has no model line in its target party role: "
                f"{target_path}"
            )
        union(indices)
        compound_indices[compound.alias] = indices

    components: dict[int, list[int]] = defaultdict(list)
    for index in range(len(slots)):
        components[find(index)].append(index)
    ordered_components = sorted(
        (tuple(indices) for indices in components.values()),
        key=lambda indices: min(line_number(slots[index].line_id) for index in indices),
    )
    oversized = [indices for indices in ordered_components if len(indices) > maximum_slots]
    if oversized:
        largest = max(len(indices) for indices in oversized)
        raise ValueError(
            "one semantic line component exceeds max_initial_slots_per_request: "
            f"component_slots={largest}, configured_maximum={maximum_slots}"
        )

    packed: list[list[int]] = []
    current: list[int] = []
    for component in ordered_components:
        if current and len(current) + len(component) > maximum_slots:
            packed.append(current)
            current = []
        current.extend(component)
    if current:
        packed.append(current)

    batches: list[_InitialRequestBatch] = []
    for indices in packed:
        selected = set(indices)
        selected_compounds = tuple(
            compound
            for compound in compounds
            if any(index in selected for index in compound_indices[compound.alias])
        )
        batches.append(
            _InitialRequestBatch(
                slots=tuple(slots[index] for index in sorted(indices)),
                compound_slots=selected_compounds,
            )
        )
    return tuple(batches)


def _output_schema(
    slots: Sequence[_Slot],
    compound_slots: Sequence[_CompoundSlot] = (),
) -> tuple[type[dict[str, Any]], dict[str, JsonValue]]:
    properties = {
        slot.alias: {
            "type": "string",
            "minLength": 1,
            "maxLength": 10000,
            "description": f"Complete replacement text for host-owned {slot.line_id}; no newline.",
        }
        for slot in slots
    }
    properties.update(
        {
            slot.alias: {
                "type": "string",
                "minLength": 5,
                "maxLength": 360,
                "description": (
                    "Exact compound-party substring beginning with the target primary name; it "
                    "must also occur across the submitted replacement lines."
                ),
            }
            for slot in compound_slots
        }
    )
    required = [slot.alias for slot in slots] + [slot.alias for slot in compound_slots]
    schema: dict[str, JsonValue] = {
        "type": "object",
        "properties": cast(JsonValue, properties),
        "required": cast(JsonValue, required),
        "additionalProperties": False,
    }
    return (
        StructuredDict(
            cast(Any, schema),
            name="TemplatePatch",
            description="Every required host-owned OCR line replacement, keyed by opaque slot.",
        ),
        schema,
    )


def _preflight_initial_model_contracts(
    materials: Sequence[_InventoryCaseMaterial],
    *,
    maximum_slots: int | None,
    maximum_successful_responses: int,
    profile_context_lines: int,
) -> None:
    """Materialize every initial provider contract before the first paid request.

    Dynamic work-item payloads and schemas are deterministic.  Validating them lazily inside
    concurrent provider tasks can otherwise spend on earlier documents before a later optional
    field exposes a host-side contract defect.  Repair payloads remain derived from an already
    validated initial contract plus host diagnostics.
    """

    for material in materials:
        document_id = material.compiled.bundle.result.documentId
        try:
            batches = _initial_request_batches(material, maximum_slots=maximum_slots)
            if len(batches) > maximum_successful_responses:
                raise ValueError(
                    "initial request partitions exceed the successful-response budget: "
                    f"batches={len(batches)}, budget={maximum_successful_responses}"
                )
            for batch in batches:
                payload = _editor_payload(
                    material.compiled,
                    batch.slots,
                    batch.compound_slots,
                    profile_context_lines=profile_context_lines,
                )
                _output_type, schema = _output_schema(batch.slots, batch.compound_slots)
                canonical_json_bytes(payload)
                canonical_json_bytes(schema)
        except Exception as error:
            raise ValueError(
                f"initial model contract preflight failed for {document_id}: "
                f"{type(error).__name__}: {error}"
            ) from error


def _editor_payload(
    compiled: _CompiledCase,
    slots: Sequence[_Slot],
    compound_slots: Sequence[_CompoundSlot] = (),
    *,
    profile_context_lines: int = 0,
) -> dict[str, JsonValue]:
    requirement_by_digest: dict[str, dict[str, JsonValue]] = {}
    for slot in slots:
        for requirement in slot.requirements:
            digest = sha256_bytes(canonical_json_bytes(requirement))
            requirement_by_digest[digest] = requirement
    reference_by_digest = {
        digest: f"r{ordinal}" for ordinal, digest in enumerate(sorted(requirement_by_digest))
    }
    party_blocks: dict[str, dict[str, Any]] = {}
    alias_by_line = {slot.line_id: slot.alias for slot in slots}
    auxiliary_consistency_groups: dict[str, dict[str, Any]] = {}
    for requirement in compiled.workspace.raw_auxiliary_identity_requirements:
        match = re.search(r"L[0-9]{5}", requirement.requirementId)
        if match is None:
            raise ValueError(
                "raw auxiliary identity requirement lacks its source line ID: "
                f"{requirement.requirementId}"
            )
        start = line_number(match.group(0))
        aliases = tuple(
            alias_by_line[line_id]
            for number in range(start, start + requirement.sourceIdentityLineCount)
            if (line_id := f"L{number:05d}") in alias_by_line
        )
        if not aliases:
            continue
        row = auxiliary_consistency_groups.setdefault(
            requirement.consistencyGroupId,
            {
                "consistencyGroupId": requirement.consistencyGroupId,
                "sourceIdentityKey": _auxiliary_identity_token_key(requirement.sourceIdentity),
                "sourceIdentities": {requirement.sourceIdentity},
                "sourceOccurrenceSlotGroups": set(),
            },
        )
        if row["sourceIdentityKey"] != _auxiliary_identity_token_key(
            requirement.sourceIdentity
        ):
            raise ValueError(
                "raw auxiliary identity consistency group has conflicting semantics: "
                f"{requirement.consistencyGroupId}"
            )
        cast(set[str], row["sourceIdentities"]).add(requirement.sourceIdentity)
        cast(set[tuple[str, ...]], row["sourceOccurrenceSlotGroups"]).add(aliases)
    for work_item in compiled.work_items:
        if work_item.state != "agent_residual":
            continue
        for path in work_item.targetPaths:
            match = re.match(
                r"^(documentPatch\.parties\.(?:notifyParties\[[0-9]+\]|[A-Za-z]+))\.",
                path,
            )
            if match is None or not isinstance(work_item.targetValue, str):
                continue
            role_path = match.group(1)
            owned_aliases = {
                alias_by_line[line_id]
                for line_id in work_item.evidenceLineIds
                if line_id in alias_by_line
            }
            if not owned_aliases:
                continue
            row = party_blocks.setdefault(
                role_path,
                {"rolePath": role_path, "slots": set(), "requiredTargetScalars": {}},
            )
            cast(set[str], row["slots"]).update(owned_aliases)
            scalar_rows = cast(dict[str, dict[str, Any]], row["requiredTargetScalars"])
            scalar_row = scalar_rows.setdefault(
                path,
                {
                    "value": work_item.targetValue,
                    "sourceSlots": set(),
                    "sourceSlotGroups": set(),
                },
            )
            if scalar_row["value"] != work_item.targetValue:
                raise ValueError(f"party scalar has conflicting targets: {path}")
            cast(set[str], scalar_row["sourceSlots"]).update(owned_aliases)
            cast(set[tuple[str, ...]], scalar_row["sourceSlotGroups"]).update(
                _party_scalar_source_slot_groups(
                    compiled,
                    path=path,
                    work_item=work_item,
                    alias_by_line=alias_by_line,
                )
            )

    cargo_blocks = _cargo_block_payload(compiled, slots)
    payload: dict[str, JsonValue] = {
        "task": (
            "Rewrite every host-owned OCR line so the full document represents the synthetic "
            "target and contains no source shipment identity or stale dependent fact."
        ),
        "documentId": compiled.bundle.result.documentId,
        "rules": cast(
            JsonValue,
            [
                "Return exactly one replacement string for every required slot key.",
                "Each string replaces one complete OCR line and must contain no newline.",
                "Preserve field labels, punctuation style, casing style, and line topology.",
                (
                    "Render every target party scalar exactly, including its punctuation; "
                    "only casing and line wrapping may follow the source block."
                ),
                "Never abbreviate or truncate a target cargo description or free-text scalar.",
                (
                    "Keep every lexical source line lexical and grammatical; punctuation-only "
                    "filler is invalid."
                ),
                "Never use placeholders such as N/A, UNKNOWN, or UNAVAILABLE.",
                (
                    "Synthesize realistic source-only flavor when its slot exists but is not a "
                    "label field."
                ),
                (
                    "A changed_source_auxiliary_copy is outside every labeled scalar slot; "
                    "fictionalize it without duplicating a labeled target value."
                ),
                (
                    "Reconcile repeated totals, measures, equipment, route, and reefer "
                    "statements from the target."
                ),
                (
                    "For equipment actions, reproduce unambiguousTargetPrintedSurface verbatim "
                    "on an owned equipment line; retain its surrounding heading/count syntax."
                ),
                (
                    "Do not copy a value across indexed containers or cargo rows unless the "
                    "target says so."
                ),
                (
                    "Preserve legal relationship wording; change its linked identities or "
                    "carrier alias instead."
                ),
                (
                    "An anonymous source equipment assertion is not a package total. If a line "
                    "is host-locked, preserve it exactly; render package quantity/type only in "
                    "the separately owned package line."
                ),
                (
                    "Every host_locked_target_literal must remain verbatim on its assigned line; "
                    "it was already rendered deterministically and is not creative flavor."
                ),
                (
                    "Every host_locked_source_prefix and host_locked_source_suffix must remain "
                    "verbatim at the start or end of its assigned line; only the identity "
                    "between those affixes is editable."
                ),
            ],
        ),
        "semanticReferences": cast(
            JsonValue,
            {
                reference_by_digest[digest]: requirement_by_digest[digest]
                for digest in sorted(requirement_by_digest)
            },
        ),
        "slots": cast(
            JsonValue,
            [
                {
                    "slot": slot.alias,
                    "sourceLine": slot.source_line,
                    "semanticRefs": cast(
                        JsonValue,
                        [
                            reference_by_digest[sha256_bytes(canonical_json_bytes(requirement))]
                            for requirement in slot.requirements
                        ],
                    ),
                }
                for slot in slots
            ],
        ),
        "partyBlocks": cast(
            JsonValue,
            [
                {
                    "rolePath": row["rolePath"],
                    "slots": sorted(cast(set[str], row["slots"])),
                    "requiredTargetScalars": cast(
                        JsonValue,
                        [
                            {
                                "path": path,
                                "value": scalar["value"],
                                "sourceSlots": sorted(
                                    {
                                        alias
                                        for group in _minimal_slot_groups(
                                            cast(
                                                set[tuple[str, ...]],
                                                scalar["sourceSlotGroups"],
                                            )
                                        )
                                        for alias in group
                                    }
                                    or cast(set[str], scalar["sourceSlots"]),
                                    key=lambda value: int(value.removeprefix("s")),
                                ),
                                "sourceSlotGroups": [
                                    list(group)
                                    for group in _minimal_slot_groups(
                                        cast(
                                            set[tuple[str, ...]],
                                            scalar["sourceSlotGroups"],
                                        )
                                    )
                                ],
                            }
                            for path, scalar in sorted(
                                cast(
                                    dict[str, dict[str, Any]],
                                    row["requiredTargetScalars"],
                                ).items()
                            )
                        ],
                    ),
                    "instruction": (
                        "Across these slots, render every required target scalar completely. "
                        "Each scalar's sourceSlotGroups are independent printed copies. Render "
                        "the complete target scalar once inside every group; reflow only within "
                        "that group. sourceSlots is their union. No word, number, postal code, or "
                        "terminal punctuation from a target scalar may be omitted."
                    ),
                }
                for _, row in sorted(party_blocks.items())
            ],
        ),
        "cargoBlocks": cast(JsonValue, cargo_blocks),
        "auxiliaryIdentityConsistencyGroups": cast(
            JsonValue,
            [
                {
                    "consistencyGroupId": group_id,
                    "sourceIdentities": sorted(cast(set[str], row["sourceIdentities"])),
                    "sourceOccurrenceSlotGroups": [
                        list(group)
                        for group in sorted(
                            cast(set[tuple[str, ...]], row["sourceOccurrenceSlotGroups"]),
                            key=lambda values: tuple(
                                int(value.removeprefix("s")) for value in values
                            ),
                        )
                    ],
                    "instruction": (
                        "Render one distinct fictional auxiliary identity exactly once across each "
                        "sourceOccurrenceSlotGroups entry, reflowing it across that occurrence's "
                        "physical lines. Reuse the same identity across all occurrence groups "
                        "while preserving surrounding legal wording and each referenced target "
                        "principal."
                    ),
                }
                for group_id, row in sorted(auxiliary_consistency_groups.items())
            ],
        ),
        "compoundRealizations": cast(
            JsonValue,
            [
                {
                    "slot": slot.alias,
                    "targetPath": slot.requirement.targetPath,
                    "targetPrimaryName": slot.requirement.targetPrimaryName,
                    "relationships": list(slot.requirement.relationships),
                    "instruction": (
                        "Return the exact substring beginning with targetPrimaryName and followed "
                        "by every required relationship plus a distinct realistic fictional "
                        "secondary identity. Render the same substring across the line slots."
                    ),
                }
                for slot in compound_slots
            ],
        ),
    }
    if any(
        requirement.get("kind") == "template_profile_residual"
        for slot in slots
        for requirement in slot.requirements
    ):
        payload["syntheticTargetLabel"] = cast(
            JsonValue, compiled.workspace.current_target_label
        )
        profile_line_numbers = {
            line_number(slot.line_id)
            for slot in slots
            if any(
                requirement.get("kind") == "template_profile_residual"
                for requirement in slot.requirements
            )
        }
        owned_line_numbers = {line_number(slot.line_id) for slot in slots}
        current_lines = compiled.workspace.current_text.splitlines()
        context_line_numbers = {
            candidate
            for number in profile_line_numbers
            for candidate in range(
                max(1, number - profile_context_lines),
                min(len(current_lines), number + profile_context_lines) + 1,
            )
            if candidate not in owned_line_numbers and current_lines[candidate - 1].strip()
        }
        payload["templateProfileReadOnlyContext"] = cast(
            JsonValue,
            [
                {
                    "lineId": f"L{number:05d}",
                    "sourceLine": current_lines[number - 1],
                }
                for number in sorted(context_line_numbers)
            ],
        )
    return payload


def _validated_output_replacements(
    slots: Sequence[_Slot],
    output: Mapping[str, Any],
    compound_slots: Sequence[_CompoundSlot] = (),
) -> tuple[LineRangeReplacement, ...]:
    expected = {slot.alias for slot in slots} | {slot.alias for slot in compound_slots}
    if set(output) != expected:
        raise ValueError("provider output keys differ from the complete required slot set")
    unpopulated = tuple(
        slot.alias
        for slot in slots
        if not isinstance(output.get(slot.alias), str)
        or not cast(str, output[slot.alias]).strip()
    )
    if unpopulated:
        if len(unpopulated) == 1:
            raise ValueError(
                f"provider output slot is not a populated string: {unpopulated[0]}"
            )
        raise ValueError(
            f"provider output slots are not populated strings: {', '.join(unpopulated)}"
        )
    replacements: list[LineRangeReplacement] = []
    validation_errors: list[str] = []
    repeated_auxiliary_outputs: dict[tuple[str, str], list[tuple[str, str]]] = defaultdict(
        list
    )
    for slot in slots:
        replacement = output.get(slot.alias)
        assert isinstance(replacement, str)
        if "\n" in replacement or "\r" in replacement:
            validation_errors.append(f"slot contains a newline: {slot.alias}")
            continue
        if any(character.isalnum() for character in slot.source_line) and not any(
            character.isalnum() for character in replacement
        ):
            validation_errors.append(
                f"slot degenerates a lexical line to punctuation: {slot.alias}"
            )
            continue
        # A model occasionally changes only the case of a host-prefilled literal while editing
        # neighboring flavor.  When there is exactly one punctuation-identical match, restore
        # the byte-exact lock deterministically instead of paying for a semantic repair call.
        for requirement in slot.requirements:
            target = requirement.get("target")
            if (
                requirement.get("kind")
                not in {"host_locked_target_literal", "host_locked_source_literal"}
                or not isinstance(target, str)
                or target in replacement
            ):
                continue
            matches = tuple(re.finditer(re.escape(target), replacement, re.IGNORECASE))
            if len(matches) == 1:
                match = matches[0]
                replacement = replacement[: match.start()] + target + replacement[match.end() :]
        missing_locks = [
            requirement["target"]
            for requirement in slot.requirements
            if requirement.get("kind")
            in {"host_locked_target_literal", "host_locked_source_literal"}
            and isinstance(requirement.get("target"), str)
            and cast(str, requirement["target"]) not in replacement
        ]
        if missing_locks:
            validation_errors.append(
                f"slot omits host-locked literals: {slot.alias}: {missing_locks}"
            )
        missing_prefixes = [
            requirement["target"]
            for requirement in slot.requirements
            if requirement.get("kind") == "host_locked_source_prefix"
            and isinstance(requirement.get("target"), str)
            and not replacement.startswith(cast(str, requirement["target"]))
        ]
        missing_suffixes = [
            requirement["target"]
            for requirement in slot.requirements
            if requirement.get("kind") == "host_locked_source_suffix"
            and isinstance(requirement.get("target"), str)
            and not replacement.endswith(cast(str, requirement["target"]))
        ]
        if missing_prefixes or missing_suffixes:
            validation_errors.append(
                "slot changes host-locked legal affixes: "
                f"{slot.alias}: prefixes={missing_prefixes}, suffixes={missing_suffixes}"
            )
        missing_required = [
            requirement["target"]
            for requirement in slot.requirements
            if requirement.get("kind") == "required_rendered_surface"
            and isinstance(requirement.get("target"), str)
            and cast(str, requirement["target"]) not in replacement
        ]
        if missing_required:
            validation_errors.append(
                f"slot omits path-owned rendered surfaces: {slot.alias}: "
                f"{missing_required}"
            )
        normalized_replacement = " ".join(replacement.casefold().split())
        missing_semantic = [
            requirement["target"]
            for requirement in slot.requirements
            if requirement.get("kind") == "required_semantic_surface"
            and isinstance(requirement.get("target"), str)
            and " ".join(cast(str, requirement["target"]).casefold().split())
            not in normalized_replacement
        ]
        if missing_semantic:
            validation_errors.append(
                f"slot truncates a required semantic surface: {slot.alias}: "
                f"{missing_semantic}"
            )
        for requirement in slot.requirements:
            candidate_id = requirement.get("candidateId")
            if (
                requirement.get("kind") == "changed_source_auxiliary_copy"
                and isinstance(candidate_id, str)
            ):
                repeated_auxiliary_outputs[(candidate_id, slot.source_line)].append(
                    (slot.alias, replacement)
                )
        if replacement == slot.source_line:
            continue
        replacements.append(
            LineRangeReplacement(
                startLineId=slot.line_id,
                endLineId=slot.line_id,
                newText=replacement,
            )
        )
    for (candidate_id, _source_line), realizations in repeated_auxiliary_outputs.items():
        if len(realizations) < 2:
            continue
        distinct = {replacement for _alias, replacement in realizations}
        if len(distinct) > 1:
            aliases = ", ".join(alias for alias, _replacement in realizations)
            validation_errors.append(
                "repeated identical auxiliary source line has inconsistent fictional "
                f"realizations: candidate={candidate_id}, slots={aliases}"
            )
    if validation_errors:
        raise ValueError(
            "provider output violates host slot contract: " + "; ".join(validation_errors)
        )
    if not replacements:
        raise ValueError("provider returned no changed OCR line")
    return tuple(replacements)


def _validated_compound_realizations(
    compound_slots: Sequence[_CompoundSlot], output: Mapping[str, Any]
) -> tuple[CompoundPartyFlavorRealization, ...]:
    unpopulated = tuple(
        slot.alias
        for slot in compound_slots
        if not isinstance(output.get(slot.alias), str)
        or not cast(str, output[slot.alias]).strip()
    )
    if unpopulated:
        if len(unpopulated) == 1:
            raise ValueError(
                f"provider compound slot is not a populated string: {unpopulated[0]}"
            )
        raise ValueError(
            "provider compound slots are not populated strings: " + ", ".join(unpopulated)
        )
    realizations: list[CompoundPartyFlavorRealization] = []
    for slot in compound_slots:
        rendered = output.get(slot.alias)
        assert isinstance(rendered, str)
        if "\n" in rendered or "\r" in rendered:
            raise ValueError(f"provider compound slot contains a newline: {slot.alias}")
        realizations.append(
            CompoundPartyFlavorRealization(
                targetPath=slot.requirement.targetPath,
                targetPrimaryName=slot.requirement.targetPrimaryName,
                renderedName=rendered,
            )
        )
    return tuple(realizations)


def _apply_output(
    compiled: _CompiledCase,
    slots: Sequence[_Slot],
    compound_slots: Sequence[_CompoundSlot],
    output: Mapping[str, Any],
) -> None:
    replacements = _validated_output_replacements(slots, output, compound_slots)
    realizations = _validated_compound_realizations(compound_slots, output)
    apply_line_range_replacements(
        compiled.workspace,
        replacements,
        compound_party_flavor_realizations=realizations,
    )


def _slot_requirement_paths(slot: _Slot) -> frozenset[str]:
    paths: set[str] = set()
    for requirement in slot.requirements:
        raw_paths = requirement.get("paths")
        if isinstance(raw_paths, list):
            paths.update(value for value in raw_paths if isinstance(value, str))
    return frozenset(paths)


def _party_role_path(path: str) -> str | None:
    match = re.match(
        r"^(documentPatch\.parties\.(?:notifyParties\[[0-9]+\]|[A-Za-z]+))(?:\.|$)",
        path,
    )
    return match.group(1) if match is not None else None


def _cargo_group_path(path: str) -> str | None:
    match = re.match(r"^(documentPatch\.cargoGroups\[[0-9]+\])(?:\.|$)", path)
    return match.group(1) if match is not None else None


def _cargo_block_payload(
    compiled: _CompiledCase,
    slots: Sequence[_Slot],
) -> list[dict[str, JsonValue]]:
    """Expose all changed free-text scalars that must coexist in one cargo block.

    Corrections are line-scoped, but two ``additionalInformation`` values commonly share the
    same two OCR lines.  Supplying only the value named by the latest rejection makes a model
    alternate between otherwise valid values.  This payload groups only changed, provider-owned
    target strings from the same cargo group; it neither grants new line authority nor exposes
    unrelated label fields.
    """

    paths_by_alias = {slot.alias: _slot_requirement_paths(slot) for slot in slots}
    package_guard_by_group = {
        requirement.targetPath.removesuffix(".description"): requirement
        for requirement in compiled.workspace.cargo_flavor_rewrite_requirements
        if requirement.enforcePackageSurfaceGuard
    }
    aliases_by_group: dict[str, set[str]] = defaultdict(set)
    paths_by_group: dict[str, set[str]] = defaultdict(set)
    for alias, paths in paths_by_alias.items():
        for path in paths:
            group_path = _cargo_group_path(path)
            if group_path is None:
                continue
            aliases_by_group[group_path].add(alias)
            paths_by_group[group_path].add(path)

    output: list[dict[str, JsonValue]] = []
    for group_path, aliases in sorted(aliases_by_group.items()):
        package_guard = package_guard_by_group.get(group_path)
        scalars: list[dict[str, JsonValue]] = []
        for path in sorted(paths_by_group[group_path]):
            try:
                value = target_value(compiled.workspace.current_target_label, path)
            except KeyError:
                # A changed optional source field can legitimately be absent from the synthetic
                # target (for example, a removed dangerous-goods packing group).  The line-level
                # requirement still tells the editor to remove it, but there is no target scalar
                # to co-render in ``cargoBlocks``.  Malformed paths and list topology errors keep
                # surfacing through their distinct exception types.
                continue
            if not isinstance(value, str) or not value.strip():
                continue
            scalars.append({"path": path, "value": value})
        if not scalars:
            continue
        block: dict[str, JsonValue] = {
                "groupPath": group_path,
                "slots": sorted(
                    aliases,
                    key=lambda value: int(value.removeprefix("s")),
                ),
                "requiredTargetScalars": cast(JsonValue, scalars),
                "instruction": (
                    "Render every required target scalar exactly once across these slots. All "
                    "values must coexist; never replace one required value with another during "
                    "a correction. Preserve the block's physical line count and OCR style. "
                    "For repeated source product rows, repeat and reflow the authoritative target "
                    "description instead of inventing a product variant or packaging fact."
                ),
            }
        if package_guard is not None:
            block["allowedPackageSurfaces"] = cast(
                JsonValue,
                list(package_guard.allowedPackageSurfaces),
            )
        output.append(block)
    return output


def _repair_selection(
    slots: Sequence[_Slot],
    compound_slots: Sequence[_CompoundSlot],
    *,
    error_message: str | None = None,
    full_audit: FullDocumentAudit | None = None,
) -> tuple[tuple[_Slot, ...], tuple[_CompoundSlot, ...]]:
    """Select the smallest safely independent correction surface.

    A provider repair must never regenerate dozens of already accepted lines. Findings carry
    exact line IDs and schema paths; party fields are the one exception because names and
    addresses legitimately wrap across the complete source-role block. Those blocks are retried
    as a unit. If a host error cannot be localized, the function deliberately returns the full
    contract rather than silently ignoring it.
    """

    slot_by_alias = {slot.alias: slot for slot in slots}
    compound_by_alias = {slot.alias: slot for slot in compound_slots}
    aliases_by_line = {slot.line_id: slot.alias for slot in slots}
    paths_by_alias = {slot.alias: _slot_requirement_paths(slot) for slot in slots}
    selected_aliases: set[str] = set()
    selected_compounds: set[str] = set()
    affected_paths: set[str] = set()
    line_localized_paths: set[str] = set()
    affected_party_roles: set[str] = set()
    affected_cargo_groups: set[str] = set()

    if error_message:
        slot_aliases = re.findall(r"(?<![A-Za-z0-9])(s[0-9]+)(?![A-Za-z0-9])", error_message)
        selected_aliases.update(value for value in slot_aliases if value in slot_by_alias)
        compound_aliases = re.findall(r"(?<![A-Za-z0-9])(c[0-9]+)(?![A-Za-z0-9])", error_message)
        selected_compounds.update(value for value in compound_aliases if value in compound_by_alias)
        for value in re.findall(r"L[0-9]{5}", error_message):
            alias = aliases_by_line.get(value)
            if alias is not None:
                selected_aliases.add(alias)
        for value in re.findall(r"(?i)\bline\s+([1-9][0-9]*)\b", error_message):
            alias = aliases_by_line.get(f"L{int(value):05d}")
            if alias is not None:
                selected_aliases.add(alias)
        direct_aliases = set(selected_aliases)
        affected_party_roles.update(
            role
            for alias in direct_aliases
            for role in (_party_role_path(path) for path in paths_by_alias.get(alias, ()))
            if role is not None
        )
        affected_cargo_groups.update(
            group
            for alias in direct_aliases
            for group in (_cargo_group_path(path) for path in paths_by_alias.get(alias, ()))
            if group is not None
        )
        for paths in paths_by_alias.values():
            if any(path in error_message for path in paths):
                affected_paths.update(path for path in paths if path in error_message)
        line_localized_paths.update(
            path
            for alias in direct_aliases
            for path in paths_by_alias.get(alias, ())
            if path in affected_paths
        )

    if full_audit is not None:
        for finding in full_audit.findings:
            finding_aliases: set[str] = set()
            for line_id in finding.lineIds:
                alias = aliases_by_line.get(line_id)
                if alias is not None:
                    selected_aliases.add(alias)
                    finding_aliases.add(alias)
            if finding_aliases:
                line_localized_paths.update(finding.targetPaths)
            else:
                affected_paths.update(finding.targetPaths)
            affected_party_roles.update(
                role
                for role in (_party_role_path(path) for path in finding.targetPaths)
                if role is not None
            )
            affected_cargo_groups.update(
                group
                for group in (_cargo_group_path(path) for path in finding.targetPaths)
                if group is not None
            )
            if finding.category == "legal_relation_topology_mismatch":
                selected_compounds.update(compound_by_alias)

    affected_party_roles.update(
        role for role in (_party_role_path(path) for path in affected_paths) if role is not None
    )
    if affected_party_roles:
        for alias, paths in paths_by_alias.items():
            if any(_party_role_path(path) in affected_party_roles for path in paths):
                selected_aliases.add(alias)
        for compound in compound_slots:
            if _party_role_path(compound.requirement.targetPath) in affected_party_roles:
                selected_compounds.add(compound.alias)

    affected_cargo_groups.update(
        group for group in (_cargo_group_path(path) for path in affected_paths) if group is not None
    )
    if affected_cargo_groups:
        for alias, paths in paths_by_alias.items():
            if any(_cargo_group_path(path) in affected_cargo_groups for path in paths):
                selected_aliases.add(alias)

    for alias, paths in paths_by_alias.items():
        if paths & (affected_paths - line_localized_paths):
            selected_aliases.add(alias)

    if not selected_aliases and not selected_compounds:
        return tuple(slots), tuple(compound_slots)
    if selected_compounds and not selected_aliases:
        compound_paths = {
            compound_by_alias[alias].requirement.targetPath for alias in selected_compounds
        }
        for alias, paths in paths_by_alias.items():
            if paths & compound_paths:
                selected_aliases.add(alias)
    if not selected_aliases:
        return tuple(slots), tuple(compound_slots)
    return (
        tuple(slot for slot in slots if slot.alias in selected_aliases),
        tuple(slot for slot in compound_slots if slot.alias in selected_compounds),
    )


def _evaluate_candidate(
    material: _InventoryCaseMaterial,
    output: Mapping[str, Any],
) -> _CandidateEvaluation:
    """Apply and audit a provider response on an isolated transaction workspace."""

    workspace = copy.deepcopy(material.compiled.workspace)
    compiled = replace(material.compiled, workspace=workspace)
    _apply_output(compiled, material.slots, material.compound_slots, output)
    host_audit = deterministic_rewrite_audit(workspace, compiled.bundle.changedLeaves)
    full_audit = audit_full_document(
        document_id=compiled.bundle.result.documentId,
        source_text=workspace.original_text,
        output_text=workspace.current_text,
        source_label=workspace.source_label,
        target_label=workspace.current_target_label,
        inventory=material.inventory,
        deterministic_edits=material.deterministic_edits,
        oracle_case=material.oracle_case,
        surface_requirements=compiled.bundle.surfaceRenderingRequirements,
        anchored_replacements=compiled.bundle.anchoredScalarReplacementRequirements,
        deterministic_prefills=workspace.deterministic_prefills,
        work_items=compiled.work_items,
    )
    return _CandidateEvaluation(
        compiled=compiled,
        host_audit=host_audit,
        full_audit=full_audit,
    )


def _commit_candidate(material: _InventoryCaseMaterial, candidate: _CandidateEvaluation) -> None:
    """Publish an already audited candidate into the live in-memory workspace."""

    if not candidate.passed:
        raise ValueError("cannot commit a rewrite candidate that failed deterministic audit")
    workspace = material.compiled.workspace
    if workspace.current_sha256 != material.deterministic_text_sha256 or workspace.commits:
        raise RuntimeError("live rewrite workspace changed before candidate commit")
    workspace.current_text = candidate.compiled.workspace.current_text
    workspace.commits.extend(candidate.compiled.workspace.commits)


def _candidate_text_preview(
    material: _InventoryCaseMaterial,
    output: Mapping[str, Any],
) -> str:
    """Materialize provider lines without bypassing the real transactional validator."""

    lines = material.compiled.workspace.current_text.splitlines(keepends=True)
    for slot in material.slots:
        rendered = output.get(slot.alias)
        if not isinstance(rendered, str) or "\n" in rendered or "\r" in rendered:
            raise ValueError(f"cannot preview invalid provider slot: {slot.alias}")
        index = line_number(slot.line_id) - 1
        lines[index] = rendered + _line_ending(lines[index])
    return "".join(lines)


def _target_occurrence_repair(
    material: _InventoryCaseMaterial,
    output: Mapping[str, Any],
) -> tuple[tuple[_Slot, ...], tuple[dict[str, JsonValue], ...]]:
    """Locate surplus/missing role-bound values after an atomic cardinality rejection."""

    preview = _candidate_text_preview(material, output)
    preview_lines = preview.splitlines()
    selected_aliases: set[str] = set()
    diagnostics: list[dict[str, JsonValue]] = []
    alias_by_line = {slot.line_id: slot.alias for slot in material.slots}
    for requirement in material.compiled.workspace.target_value_occurrence_requirements:
        observed = _target_value_occurrence_count(preview, requirement)
        if observed == requirement.requiredOccurrences:
            continue
        is_party_scalar = any(
            path.startswith("documentPatch.parties.")
            and path.rsplit(".", 1)[-1] in {"name", "address"}
            for path in requirement.targetPaths
        )
        occurrence_groups = (
            _party_scalar_occurrence_line_sets(preview, requirement.targetValue)
            if is_party_scalar
            else tuple(
                frozenset((number,))
                for number, line in enumerate(preview_lines, start=1)
                if _target_value_occurrence_count(line, requirement) > 0
            )
        )
        occurrence_lines = tuple(
            f"L{number:05d}"
            for number in sorted({number for group in occurrence_groups for number in group})
        )
        authorized_lines: set[str] = set()
        for slot in material.slots:
            if any(
                requirement_path in cast(list[str], row.get("paths", []))
                for row in slot.requirements
                if row.get("kind")
                in {
                    "task_label_delta",
                    "changed_source_occurrence",
                    "host_locked_target_literal",
                    "required_rendered_surface",
                }
                for requirement_path in requirement.targetPaths
            ):
                authorized_lines.add(slot.line_id)
        if observed > requirement.requiredOccurrences:
            surplus_groups = tuple(
                group
                for group in occurrence_groups
                if not {f"L{number:05d}" for number in group} <= authorized_lines
            )
            repair_lines = tuple(
                f"L{number:05d}"
                for number in sorted({number for group in surplus_groups for number in group})
            )
            if not repair_lines:
                repair_lines = occurrence_lines
            instruction = (
                "Remove the surplus exact value only from auxiliary/non-role lines; preserve "
                "each role-owned occurrence and render a coherent shorter fictional alias when "
                "the line is carrier branding."
            )
        else:
            repair_lines = tuple(sorted(authorized_lines, key=line_number))
            instruction = (
                "Render the exact target value on each missing role-owned line without adding it "
                "to unrelated auxiliary lines."
            )
        # A repeated party identity can be embedded inside another party's address (for example
        # ``C/O <notify party>``). Rewriting only that one line can destroy the enclosing role's
        # already-correct address, forcing repairs to oscillate between two valid targets. Expose
        # the complete host-owned party group that contains any selected line so both semantics
        # can be reflowed atomically in one bounded correction.
        repair_line_set = set(repair_lines)
        affected_party_roles = {
            role
            for item in getattr(material.compiled, "work_items", ())
            if repair_line_set.intersection(item.evidenceLineIds)
            for role in (_party_role_path(path) for path in item.targetPaths)
            if role is not None
        }
        if affected_party_roles:
            repair_line_set.update(
                line_id
                for item in material.compiled.work_items
                if any(
                    _party_role_path(path) in affected_party_roles for path in item.targetPaths
                )
                for line_id in item.evidenceLineIds
                if line_id in alias_by_line
            )
            repair_lines = tuple(sorted(repair_line_set, key=line_number))
        selected_aliases.update(
            alias_by_line[line_id] for line_id in repair_lines if line_id in alias_by_line
        )
        diagnostics.append(
            {
                "targetPaths": cast(JsonValue, list(requirement.targetPaths)),
                "targetValue": requirement.targetValue,
                "requiredOccurrences": requirement.requiredOccurrences,
                "observedOccurrences": observed,
                "occurrenceLineIds": cast(JsonValue, list(occurrence_lines)),
                "roleOwnedLineIds": cast(JsonValue, sorted(authorized_lines, key=line_number)),
                "repairLineIds": cast(JsonValue, list(repair_lines)),
                "instruction": instruction,
            }
        )
    selected = tuple(slot for slot in material.slots if slot.alias in selected_aliases)
    return selected, tuple(diagnostics)


def _preview_repair_selection(
    material: _InventoryCaseMaterial,
    output: Mapping[str, Any],
    *,
    error_message: str,
) -> tuple[
    tuple[_Slot, ...],
    tuple[_CompoundSlot, ...],
    tuple[dict[str, JsonValue], ...],
    FullDocumentAudit | None,
]:
    """Collect independent candidate defects before requesting another model response.

    ``apply_line_range_replacements`` deliberately fails on the first violated atomic
    postcondition.  Feeding only that first error back to the provider made independent defects
    serial: a party repair could consume the next response before four unchanged cargo lines were
    ever disclosed.  This helper materializes the still-uncommitted line dictionary, performs the
    read-only full-document audit, and unions every exact repair line it can prove.  It never
    grants authority outside compiler-owned slots and never commits the preview.
    """

    auxiliary_identity_error = "raw auxiliary" in error_message and "identity" in error_message
    occurrence_cardinality_error = (
        "duplicates or omits role-bound party values" in error_message
    )
    status_cardinality_error = "changes an unchanged source status surface" in error_message
    if auxiliary_identity_error or occurrence_cardinality_error or status_cardinality_error:
        base_slots: tuple[_Slot, ...] = ()
        base_compounds: tuple[_CompoundSlot, ...] = ()
    else:
        base_slots, base_compounds = _repair_selection(
            material.slots,
            material.compound_slots,
            error_message=error_message,
        )
    selected_aliases = {slot.alias for slot in base_slots}
    selected_compounds = {slot.alias for slot in base_compounds}
    diagnostics: list[dict[str, JsonValue]] = []

    if status_cardinality_error:
        status_aliases: set[str] = set()
        status_rows: list[dict[str, JsonValue]] = []
        for requirement in material.compiled.workspace.source_status_requirements:
            changed_aliases = tuple(
                slot.alias
                for slot in material.slots
                if slot.alias in output
                and isinstance(output[slot.alias], str)
                and (
                    (slot.source_line == requirement.sourceSurface)
                    != (output[slot.alias] == requirement.sourceSurface)
                )
            )
            if not changed_aliases:
                continue
            status_aliases.update(changed_aliases)
            status_rows.append(
                {
                    "sourceSurface": requirement.sourceSurface,
                    "sourceOccurrences": requirement.sourceOccurrences,
                    "repairSlots": cast(JsonValue, list(changed_aliases)),
                }
            )
        selected_aliases.update(status_aliases)
        if status_rows:
            diagnostics.append(
                {
                    "kind": "sourceStatusCardinality",
                    "requirements": cast(JsonValue, status_rows),
                }
            )
        else:
            # A status failure without a changed provider-owned line would contradict the
            # immutable workspace. Keep the run fail-closed and expose the full request rather
            # than silently losing a host invariant.
            selected_aliases.update(slot.alias for slot in material.slots)
            selected_compounds.update(slot.alias for slot in material.compound_slots)
    try:
        preview = _candidate_text_preview(material, output)
    except ValueError:
        return base_slots, base_compounds, (), None

    occurrence_slots, occurrence_diagnostics = _target_occurrence_repair(material, output)
    selected_aliases.update(slot.alias for slot in occurrence_slots)
    diagnostics.extend(occurrence_diagnostics)

    missing_literals = _missing_target_literals(
        preview, material.compiled.workspace.target_literal_requirements
    )
    if missing_literals:
        literal_error = (
            "candidate preview omits changed target text: "
            f"{[row.model_dump(mode='json') for row in missing_literals]}"
        )
        literal_slots, literal_compounds = _repair_selection(
            material.slots,
            material.compound_slots,
            error_message=literal_error,
        )
        selected_aliases.update(slot.alias for slot in literal_slots)
        selected_compounds.update(slot.alias for slot in literal_compounds)
        diagnostics.append(
            {
                "kind": "missingTargetLiterals",
                "requirements": cast(
                    JsonValue,
                    [row.model_dump(mode="json") for row in missing_literals],
                ),
            }
        )

    if auxiliary_identity_error:
        alias_by_line = {slot.line_id: slot.alias for slot in material.slots}
        requirements = material.compiled.workspace.raw_auxiliary_identity_requirements
        matched_groups = tuple(
            requirement.consistencyGroupId
            for requirement in requirements
            if requirement.consistencyGroupId in error_message
            or requirement.requirementId in error_message
        )
        auxiliary_lines: set[str] = set()
        for requirement in requirements:
            if requirement.consistencyGroupId not in matched_groups:
                continue
            match = re.search(r"L[0-9]{5}", requirement.requirementId)
            if match is None:
                continue
            start = line_number(match.group(0))
            auxiliary_lines.update(
                f"L{number:05d}"
                for number in range(start, start + requirement.sourceIdentityLineCount)
            )
        localized_auxiliary_aliases = {
            alias_by_line[line_id] for line_id in auxiliary_lines if line_id in alias_by_line
        }
        selected_aliases.update(localized_auxiliary_aliases)
        if matched_groups and localized_auxiliary_aliases:
            diagnostics.append(
                {
                    "kind": "rawAuxiliaryIdentityConsistency",
                    "consistencyGroupIds": cast(JsonValue, list(dict.fromkeys(matched_groups))),
                    "repairLineIds": cast(
                        JsonValue, sorted(auxiliary_lines, key=line_number)
                    ),
                }
            )
        elif not localized_auxiliary_aliases:
            # The error was not emitted by a known consistency group. Preserve the fail-closed
            # behavior rather than silently accepting an unlocalized host rejection.
            selected_aliases.update(slot.alias for slot in material.slots)
            selected_compounds.update(slot.alias for slot in material.compound_slots)

    cargo_failures = _cargo_flavor_rewrite_failures(
        preview,
        material.compiled.workspace.cargo_flavor_rewrite_requirements,
    )
    if cargo_failures:
        cargo_error = (
            "candidate preview leaves source cargo-description lines unchanged or omits the "
            f"target description: {json.dumps(cargo_failures, sort_keys=True)}"
        )
        cargo_slots, cargo_compounds = _repair_selection(
            material.slots,
            material.compound_slots,
            error_message=cargo_error,
        )
        selected_aliases.update(slot.alias for slot in cargo_slots)
        selected_compounds.update(slot.alias for slot in cargo_compounds)
        diagnostics.append(
            {
                "kind": "cargoFlavorRewrite",
                "failures": cast(JsonValue, list(cargo_failures)),
            }
        )

    workspace = copy.deepcopy(material.compiled.workspace)
    workspace.current_text = preview
    preview_audit = audit_full_document(
        document_id=material.compiled.bundle.result.documentId,
        source_text=workspace.original_text,
        output_text=preview,
        source_label=workspace.source_label,
        target_label=workspace.current_target_label,
        inventory=material.inventory,
        deterministic_edits=material.deterministic_edits,
        oracle_case=material.oracle_case,
        surface_requirements=material.compiled.bundle.surfaceRenderingRequirements,
        anchored_replacements=material.compiled.bundle.anchoredScalarReplacementRequirements,
        deterministic_prefills=workspace.deterministic_prefills,
        work_items=material.compiled.work_items,
    )
    if preview_audit.findings:
        audit_slots, audit_compounds = _repair_selection(
            material.slots,
            material.compound_slots,
            full_audit=preview_audit,
        )
        selected_aliases.update(slot.alias for slot in audit_slots)
        selected_compounds.update(slot.alias for slot in audit_compounds)
        diagnostics.append(
            {
                "kind": "fullDocumentPreview",
                "findings": cast(
                    JsonValue,
                    [
                        {
                            "category": row.category,
                            "lineIds": list(row.lineIds),
                            "sourceSurface": row.sourceSurface,
                            "targetPaths": list(row.targetPaths),
                            "explanation": row.explanation,
                        }
                        for row in preview_audit.findings
                    ],
                ),
            }
        )

    # Any correction touching one occurrence of a repeated raw auxiliary identity must expose
    # every physical line of every occurrence in that consistency group. This closure applies
    # regardless of which validator first noticed the defect (identity consistency, a locked
    # legal affix, or the full-document audit), preventing repairs from fixing one wrapped line
    # while invalidating its sibling or repeated copy.
    alias_by_line = {slot.line_id: slot.alias for slot in material.slots}
    selected_line_ids = {
        slot.line_id for slot in material.slots if slot.alias in selected_aliases
    }
    touched_groups: set[str] = set()
    requirement_lines: dict[str, set[str]] = defaultdict(set)
    for requirement in material.compiled.workspace.raw_auxiliary_identity_requirements:
        match = re.search(r"L[0-9]{5}", requirement.requirementId)
        if match is None:
            continue
        start = line_number(match.group(0))
        occupied = {
            f"L{number:05d}"
            for number in range(start, start + requirement.sourceIdentityLineCount)
        }
        requirement_lines[requirement.consistencyGroupId].update(occupied)
        if occupied & selected_line_ids:
            touched_groups.add(requirement.consistencyGroupId)
    if touched_groups:
        closed_lines = set().union(*(requirement_lines[group] for group in touched_groups))
        selected_aliases.update(
            alias_by_line[line_id] for line_id in closed_lines if line_id in alias_by_line
        )
        diagnostics.append(
            {
                "kind": "rawAuxiliaryIdentityRepairClosure",
                "consistencyGroupIds": cast(JsonValue, sorted(touched_groups)),
                "repairLineIds": cast(JsonValue, sorted(closed_lines, key=line_number)),
            }
        )

    return (
        tuple(slot for slot in material.slots if slot.alias in selected_aliases),
        tuple(
            slot for slot in material.compound_slots if slot.alias in selected_compounds
        ),
        tuple(diagnostics),
        preview_audit,
    )


async def _call_model(
    *,
    compiled: _CompiledCase,
    slots: Sequence[_Slot],
    compound_slots: Sequence[_CompoundSlot],
    model: Model,
    config: InventoryRunConfig,
    prompt: str,
    semantic_attempt: int,
    request_kind: Literal["initial_partition", "repair"],
    initial_partition_ordinal: int | None,
    initial_partition_count: int,
    repair_context: Mapping[str, JsonValue] | None = None,
) -> tuple[Mapping[str, Any] | None, tuple[InventoryModelStage, ...]]:
    if request_kind == "initial_partition" and initial_partition_ordinal is None:
        raise ValueError("initial partition request requires its partition ordinal")
    if request_kind == "repair" and initial_partition_ordinal is not None:
        raise ValueError("repair request cannot carry an initial partition ordinal")
    profile_context_lines = (
        config.workflow.template_profile_context_lines
        if isinstance(config, SynthesisRawTextInventoryBatchConfig)
        else 0
    )
    payload = _editor_payload(
        compiled,
        slots,
        compound_slots,
        profile_context_lines=profile_context_lines,
    )
    if repair_context is not None:
        payload["repair"] = cast(
            JsonValue,
            {
                "instruction": (
                    "The preceding candidate was rejected by deterministic host checks. Return "
                    "only the requested correction slots in this payload. Each previous value is "
                    "shown solely to diagnose the defect; correct it from the authoritative "
                    "semantic references."
                ),
                **repair_context,
            },
        )
    output_type, schema = _output_schema(slots, compound_slots)
    stages: list[InventoryModelStage] = []
    routes = _provider_attempt_routes(config.provider)
    if isinstance(config, SynthesisRawTextInventoryBatchConfig):
        maximum_rounds = config.workflow.max_provider_route_rounds
    else:
        maximum_rounds = 1
    eligible_routes = routes
    attempt_ordinal = 0
    for route_round in range(1, maximum_rounds + 1):
        delay = (
            _retry_delay_seconds(
                document_id=compiled.bundle.result.documentId,
                route_round=route_round,
                config=config,
            )
            if route_round > 1
            else 0.0
        )
        if delay:
            await asyncio.sleep(delay)
        transient_routes: list[str | None] = []
        for route_provider in eligible_routes:
            attempt_ordinal += 1
            if config.workflow.output_mode == "native":
                output_spec: Any = NativeOutput(
                    output_type,
                    name="template_patch",
                    description="Return the complete required opaque-slot patch.",
                    strict=True,
                )
            else:
                output_spec = ToolOutput(
                    output_type,
                    name="submit_template_patch",
                    description="Submit the complete required opaque-slot patch and finish.",
                    max_retries=0,
                    strict=True,
                    sequential=True,
                )
            agent = Agent[None, dict[str, Any]](
                model,
                output_type=output_spec,
                system_prompt=prompt,
                model_settings=cast(
                    Any,
                    # This existing helper is the repository's canonical mapping for OpenAI
                    # Responses and OpenRouter reasoning/routing parameters.
                    _settings(
                        config.provider,
                        stage="editor",
                        prompt_sha256=config.prompt.sha256,
                        route_provider=route_provider,
                    ),
                ),
                retries={"output": 0, "tools": 0},
                name="synthetic-bl-inventory-line-editor",
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
                InventoryModelStage(
                    providerModel=config.provider.model,
                    semanticAttempt=semantic_attempt,
                    requestKind=request_kind,
                    initialPartitionOrdinal=initial_partition_ordinal,
                    initialPartitionCount=initial_partition_count,
                    routeProvider=route_provider,
                    routeRound=route_round,
                    routeAttempt=attempt_ordinal,
                    retryDelayBeforeSeconds=delay if route_provider == eligible_routes[0] else 0.0,
                    outputMode=config.workflow.output_mode,
                    inputPayloadSha256=sha256_bytes(canonical_json_bytes(payload)),
                    outputSchemaSha256=sha256_bytes(canonical_json_bytes(schema)),
                    startedAtUnixSeconds=started,
                    completedAtUnixSeconds=time.time(),
                    usage=usage_receipt(
                        responses,
                        config.provider.pricing,
                        # OpenRouter token accounting is part of every successful response, but
                        # the optional gateway ``cost`` detail is not guaranteed by every routed
                        # provider.  The pinned-price estimate remains exact for the reported
                        # tokens; provider-cost coverage is published separately below.
                        require_provider_cost=False,
                    ),
                    messages=model_messages(captured),
                    modelOutput=cast(JsonValue, dict(output)) if output is not None else None,
                    hostEvaluation=None,
                    errorType=type(error).__name__ if error is not None else None,
                    errorMessage=str(error) if error is not None else None,
                ),
            )
            if output is not None:
                return output, tuple(stages)
            if error is None or not _retryable_route_error(error):
                return None, tuple(stages)
            if _transient_route_error(error):
                transient_routes.append(route_provider)
        if not transient_routes or route_round == maximum_rounds:
            break
        eligible_routes = tuple(transient_routes)
    return None, tuple(stages)


def _transient_route_error(error: Exception) -> bool:
    if not isinstance(error, Exception) or not _retryable_route_error(error):
        return False
    if isinstance(error, ModelAPIError) and not isinstance(error, ModelHTTPError):
        return True
    status_code = getattr(error, "status_code", None)
    return isinstance(status_code, int) and (status_code in {408, 425, 429} or status_code >= 500)


def _retry_delay_seconds(
    *,
    document_id: str,
    route_round: int,
    config: InventoryRunConfig,
) -> float:
    if route_round <= 1 or not isinstance(config, SynthesisRawTextInventoryBatchConfig):
        return 0.0
    workflow = config.workflow
    retry_index = route_round - 2
    base = workflow.retry_initial_delay_seconds * (workflow.retry_delay_multiplier**retry_index)
    bounded = min(base, workflow.retry_max_delay_seconds)
    digest = bytes.fromhex(
        sha256_bytes(f"{document_id}\0{route_round}\0route-retry-jitter-v1".encode())
    )
    fraction = int.from_bytes(digest[:8], "big") / float(2**64 - 1)
    return min(
        workflow.retry_max_delay_seconds,
        bounded + fraction * workflow.retry_jitter_seconds,
    )


def _combined_usage(stages: Sequence[InventoryModelStage]) -> LinguisticUsageReceipt:
    if not stages:
        return _empty_usage()
    from document_ocr.synthesis.raw_text_rewrite_cycle_probe import _combine_usage

    return _combine_usage([row.usage for row in stages])


def _attach_host_evaluation(
    stages: Sequence[InventoryModelStage],
    evaluation: InventoryHostEvaluation,
) -> tuple[InventoryModelStage, ...]:
    """Attach host disposition to the provider response it evaluated."""

    updated = list(stages)
    for index in range(len(updated) - 1, -1, -1):
        if updated[index].modelOutput is not None:
            if updated[index].hostEvaluation is not None:
                raise RuntimeError("provider response already has a host evaluation")
            updated[index] = updated[index].model_copy(update={"hostEvaluation": evaluation})
            return tuple(updated)
    raise ValueError("host evaluation has no successful provider response to attach to")


def _plot_bytes(
    reference_rows: Mapping[str, Sequence[Mapping[str, Any]]],
    results: Sequence[InventoryProbeCaseResult],
) -> dict[str, bytes]:
    import io

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import pandas as pd  # type: ignore[import-untyped]
    import seaborn as sns  # type: ignore[import-untyped]

    sns.set_theme(style="whitegrid", context="notebook")
    plots: dict[str, bytes] = {}

    def save(name: str, figure: Any) -> None:
        stream = io.BytesIO()
        figure.savefig(stream, format="png", dpi=180, bbox_inches="tight")
        plots[name] = stream.getvalue()
        plt.close(figure)

    fixture_frame = pd.DataFrame(
        [
            {
                "arm": arm,
                "document": cast(str, row["documentId"])[:12],
                "findings": row["findingCount"],
                "candidates": row["inventoryCandidates"],
            }
            for arm, rows in reference_rows.items()
            for row in rows
        ]
    )
    figure, axes = plt.subplots(1, 2, figsize=(14, 5.5))
    sns.boxplot(data=fixture_frame, x="arm", y="findings", ax=axes[0])
    sns.boxplot(data=fixture_frame, x="arm", y="candidates", ax=axes[1])
    axes[0].set(title="Known false-pass findings", xlabel="", ylabel="findings per document")
    axes[1].set(title="Detected mutable inventory", xlabel="", ylabel="candidates per document")
    save("01_negative_fixture_detection.png", figure)

    if results:
        live_frame = pd.DataFrame(
            [
                {
                    "document": row.documentId[:12],
                    "status": row.status,
                    "input": row.usage.inputTokens,
                    "reasoning": row.usage.reasoningTokens,
                    "visible": row.usage.visibleOutputTokens,
                    "cost": float(row.usage.estimatedCostUsd),
                    "model_lines": row.modelLines,
                }
                for row in results
            ]
        )
        token_frame = live_frame.melt(
            id_vars="document",
            value_vars=["input", "reasoning", "visible"],
            var_name="token_type",
            value_name="tokens",
        )
        figure, axes = plt.subplots(1, 2, figsize=(14, 5.5))
        sns.barplot(data=token_frame, x="document", y="tokens", hue="token_type", ax=axes[0])
        sns.scatterplot(
            data=live_frame,
            x="model_lines",
            y="cost",
            hue="status",
            s=90,
            ax=axes[1],
        )
        axes[0].set(title="Bounded-call token usage", xlabel="document", ylabel="tokens")
        axes[1].set(title="Residual lines versus cost", xlabel="model-owned lines", ylabel="USD")
        save("02_live_usage_and_cost.png", figure)
    return plots


def _report(
    *,
    config: InventoryRunConfig,
    reference_rows: Mapping[str, Sequence[Mapping[str, Any]]],
    results: Sequence[InventoryProbeCaseResult],
    stages_by_id: Mapping[str, Sequence[InventoryModelStage]],
    profile_documents_selected: int,
    profile_lines_selected: int,
    profile_model_lines: int,
) -> str:
    reference_counts = {
        arm: sum(not cast(bool, row["newFullDocumentAuditPassed"]) for row in rows)
        for arm, rows in reference_rows.items()
    }
    total_usage = _combined_usage([stage for stages in stages_by_id.values() for stage in stages])
    durations = {
        document_id: sum(row.completedAtUnixSeconds - row.startedAtUnixSeconds for row in stages)
        for document_id, stages in stages_by_id.items()
    }
    maximum_responses = (
        config.workflow.max_successful_model_responses_per_document
        if isinstance(config, SynthesisRawTextInventoryBatchConfig)
        else config.workflow.max_model_requests_per_document
    )
    lines = [
        "# Full-document inventory rewrite probe",
        "",
        "## Outcome",
        "",
        f"- Output mode: **{config.workflow.output_mode}**; model: `{config.provider.model}`.",
        f"- Old GLM false-pass fixtures rejected: **{reference_counts['glm']}/12**.",
        f"- Old Luna false-pass fixtures rejected: **{reference_counts['luna']}/12**.",
        (
            "- Live training-ready cases: "
            f"**{sum(row.status == 'training_ready' for row in results)}/{len(results)}**."
        ),
        (
            "- Zero-cost compiler blocks: "
            f"**{sum(row.status == 'compiler_blocked' for row in results)}**."
        ),
        (
            "- Model reviewer calls: **0**. A document permits at most "
            f"**{maximum_responses}** "
            "patch responses. Oversized initial patches may use dependency-preserving transport "
            "partitions; after initial assembly, another response requires a deterministic host "
            "rejection."
        ),
        (
            "- Independently reviewed template profile: "
            f"**{profile_documents_selected} selected documents / "
            f"{profile_lines_selected} evidence lines**; "
            f"**{profile_model_lines} exact OCR lines** reached model-owned slots after "
            "deterministic compilation."
        ),
        "- Training records published: **0**.",
        "",
        "## Live measurements",
        "",
        (
            "| Document | Status | Lines | Input | Reasoning | Visible | Cost | Time | "
            "Findings | Legacy residuals |"
        ),
        "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in results:
        lines.append(
            f"| `{row.documentId}` | {row.status} | {row.modelLines} | "
            f"{row.usage.inputTokens:,} | {row.usage.reasoningTokens:,} | "
            f"{row.usage.visibleOutputTokens:,} | ${row.usage.estimatedCostUsd} | "
            f"{durations.get(row.documentId, 0.0):.2f}s | "
            f"{len(row.fullDocumentAudit.findings) if row.fullDocumentAudit else 0} | "
            f"{row.legacyResidualCandidates} |"
        )
    compiler_blocks = tuple(row for row in results if row.status == "compiler_blocked")
    if compiler_blocks:
        lines.extend(("", "## Compiler blocks", ""))
        for row in compiler_blocks:
            lines.append(
                f"- `{row.documentId}` — `{row.compilerErrorType}`: "
                f"{row.compilerErrorMessage}"
            )
    per_thousand = total_usage.estimatedCostUsd * Decimal(1000) / max(1, len(results))
    provider_per_thousand = (
        total_usage.providerReportedCostUsd * Decimal(1000) / max(1, len(results))
        if total_usage.providerReportedCostUsd is not None
        else None
    )
    receipted_requests = sum(
        stage.usage.requests
        for stages in stages_by_id.values()
        for stage in stages
        if stage.usage.providerReportedCostUsd is not None
    )
    lines.extend(
        (
            "",
            f"Total live requests: **{total_usage.requests}**; input/reasoning/visible tokens: "
            f"**{total_usage.inputTokens:,} / {total_usage.reasoningTokens:,} / "
            f"{total_usage.visibleOutputTokens:,}**.",
            f"Pinned-price upper-bound cost: **${total_usage.estimatedCostUsd}**; same-shape "
            f"upper bound: **${per_thousand.quantize(Decimal('0.001'))}/1,000 documents**.",
            (
                "Provider-reported billed cost: **unavailable because receipt coverage was "
                f"{receipted_requests}/{total_usage.requests} requests**."
                if provider_per_thousand is None
                else f"Provider-reported billed cost: **${total_usage.providerReportedCostUsd}**; "
                f"same-shape billed rate: **${provider_per_thousand.quantize(Decimal('0.001'))}"
                "/1,000 documents**."
            ),
            "",
            "The per-thousand value is conditional on this exact difficult-case mix and provider "
            "price. It is not a production forecast.",
            (
                "Legacy residual counts are diagnostic only: that scanner is path-unaware and "
                "can flag a source token that is intentionally embedded in a different target "
                "field. Promotion is gated by its structural core plus the path-aware full-"
                "document audit and frozen regression oracle."
            ),
            "",
            "## Contract",
            "",
            "The host materializes changed-label occurrences, source-only identifiers/contacts, "
            "conditional reefer statements, repeated aggregates, legal-agent topology, and a "
            "manually adjudicated regression oracle. Shape-compatible identifiers are replaced "
            "locally. The model receives only the remaining exact lines plus their target "
            "semantics and returns a strict dictionary keyed by opaque slots. Every key is "
            "required. The host applies all changes atomically and performs both the existing "
            "rewrite audit and the independent full-document audit.",
            "",
        )
    )
    return "\n".join(lines)


def _inventory_case_result(
    *,
    material: _InventoryCaseMaterial,
    stages: Sequence[InventoryModelStage],
    model_output_committed: bool,
    failure: str | None,
) -> InventoryProbeCaseResult:
    compiled = material.compiled
    document_id = compiled.bundle.result.documentId
    host_audit = deterministic_rewrite_audit(compiled.workspace, compiled.bundle.changedLeaves)
    full_audit = audit_full_document(
        document_id=document_id,
        source_text=compiled.workspace.original_text,
        output_text=compiled.workspace.current_text,
        source_label=compiled.workspace.source_label,
        target_label=compiled.workspace.current_target_label,
        inventory=material.inventory,
        deterministic_edits=material.deterministic_edits,
        oracle_case=material.oracle_case,
        surface_requirements=compiled.bundle.surfaceRenderingRequirements,
        anchored_replacements=compiled.bundle.anchoredScalarReplacementRequirements,
        deterministic_prefills=compiled.workspace.deterministic_prefills,
        work_items=compiled.work_items,
    )
    if not model_output_committed:
        status: Literal["training_ready", "needs_review", "call_failed", "compiler_blocked"] = (
            "call_failed"
        )
        reason = failure or "The terminal model request failed."
    elif host_audit.core_passed and full_audit.passed:
        status = "training_ready"
        reason = "Atomic rewrite and every deterministic full-document postcondition passed."
    else:
        status = "needs_review"
        reason = (
            "The bounded patch committed, but a deterministic full-document postcondition failed."
        )
    return InventoryProbeCaseResult(
        documentId=document_id,
        status=status,
        reason=reason,
        sourceLines=len(compiled.workspace.original_text.splitlines()),
        modelLines=len(material.slots),
        compilerWorkItems=len(compiled.work_items),
        inventoryCandidates=len(material.inventory),
        deterministicInventoryEdits=len(material.deterministic_edits),
        hostRewriteAuditPassed=host_audit.core_passed,
        legacyResidualCandidates=len(host_audit.residualCandidates),
        fullDocumentAudit=full_audit,
        outputTextSha256=sha256_bytes(compiled.workspace.current_text.encode("utf-8")),
        usage=_combined_usage(stages),
    )


def _compiler_blocked_result(block: _CompilerBlock) -> InventoryProbeCaseResult:
    """Represent an expected fail-closed host preflight without spending provider tokens."""

    return InventoryProbeCaseResult(
        documentId=block.document_id,
        status="compiler_blocked",
        reason=(
            "The zero-API compiler preflight rejected this template/target pair before any "
            "provider request."
        ),
        sourceLines=len(block.source_text.splitlines()),
        modelLines=0,
        compilerWorkItems=0,
        inventoryCandidates=0,
        deterministicInventoryEdits=0,
        hostRewriteAuditPassed=False,
        legacyResidualCandidates=0,
        fullDocumentAudit=None,
        compilerErrorType=block.error_type,
        compilerErrorMessage=block.error_message,
        outputTextSha256=sha256_bytes(block.source_text.encode("utf-8")),
        usage=_empty_usage(),
    )


def _publish_inventory_checkpoint(
    *,
    staged: StagedArtifactRun,
    material: _InventoryCaseMaterial,
    result: InventoryProbeCaseResult,
    stages: Sequence[InventoryModelStage],
) -> None:
    compiled = material.compiled
    staged.publish_json(
        f"cases/{result.documentId}/checkpoint.json",
        {
            "schemaVersion": 2,
            "documentId": result.documentId,
            "deterministicTextSha256": material.deterministic_text_sha256,
            "finalText": compiled.workspace.current_text,
            "commits": [row.model_dump(mode="json") for row in compiled.workspace.commits],
            "result": result.model_dump(mode="json"),
            "stages": [row.model_dump(mode="json") for row in stages],
        },
    )


def _load_inventory_checkpoint(
    *,
    staged: StagedArtifactRun,
    material: _InventoryCaseMaterial,
) -> tuple[InventoryProbeCaseResult, tuple[InventoryModelStage, ...]] | None:
    document_id = material.compiled.bundle.result.documentId
    path = staged.stage_root / f"cases/{document_id}/checkpoint.json"
    if not path.exists():
        return None
    if path.is_symlink() or not path.is_file():
        raise ValueError(f"inventory checkpoint is not a regular file: {path}")
    value = json.loads(read_regular_file_bytes(path))
    expected = {
        "schemaVersion",
        "documentId",
        "deterministicTextSha256",
        "finalText",
        "commits",
        "result",
        "stages",
    }
    if not isinstance(value, dict) or set(value) != expected:
        raise ValueError(f"inventory checkpoint has an invalid shape: {path}")
    if (
        value["schemaVersion"] != 2
        or value["documentId"] != document_id
        or value["deterministicTextSha256"] != material.deterministic_text_sha256
    ):
        raise ValueError(f"inventory checkpoint identity differs: {path}")
    final_text = value["finalText"]
    if not isinstance(final_text, str):
        raise ValueError(f"inventory checkpoint final text is invalid: {path}")
    result = InventoryProbeCaseResult.model_validate_json(
        canonical_json_bytes(value["result"]), strict=True
    )
    stages_value = value["stages"]
    if not isinstance(stages_value, list):
        raise ValueError(f"inventory checkpoint stages are invalid: {path}")
    stages = tuple(
        InventoryModelStage.model_validate_json(canonical_json_bytes(row), strict=True)
        for row in stages_value
    )
    commits_value = value["commits"]
    if not isinstance(commits_value, list):
        raise ValueError(f"inventory checkpoint commits are invalid: {path}")
    commits = tuple(
        AtomicRewriteCommit.model_validate_json(canonical_json_bytes(row), strict=True)
        for row in commits_value
    )
    expected_commits = 0 if result.status == "call_failed" else 1
    if len(commits) != expected_commits:
        raise ValueError(f"inventory checkpoint commit count differs: {path}")
    if commits and (
        commits[0].beforeTextSha256 != material.deterministic_text_sha256
        or commits[-1].afterTextSha256 != sha256_bytes(final_text.encode("utf-8"))
        or commits[0].beforeTargetLabelSha256 != material.compiled.workspace.current_target_sha256
        or commits[-1].afterTargetLabelSha256 != material.compiled.workspace.current_target_sha256
    ):
        raise ValueError(f"inventory checkpoint commit chain differs: {path}")
    if result.documentId != document_id or result.outputTextSha256 != sha256_bytes(
        final_text.encode("utf-8")
    ):
        raise ValueError(f"inventory checkpoint output identity differs: {path}")
    if result.usage != _combined_usage(stages):
        raise ValueError(f"inventory checkpoint usage differs from its stage receipts: {path}")
    material.compiled.workspace.current_text = final_text
    material.compiled.workspace.commits.extend(commits)
    reproduced = _inventory_case_result(
        material=material,
        stages=stages,
        model_output_committed=result.status != "call_failed",
        failure=result.reason if result.status == "call_failed" else None,
    )
    if reproduced != result:
        raise ValueError(f"inventory checkpoint fails deterministic replay: {path}")
    return result, stages


def run_raw_text_inventory_probe(
    *,
    project_root: Path,
    config_path: Path,
    config: SynthesisRawTextInventoryProbeConfig,
) -> dict[str, JsonValue]:
    return _run_raw_text_inventory(
        project_root=project_root,
        config_path=config_path,
        config=config,
    )


def run_raw_text_inventory_batch(
    *,
    project_root: Path,
    config_path: Path,
    config: SynthesisRawTextInventoryBatchConfig,
) -> dict[str, JsonValue]:
    return _run_raw_text_inventory(
        project_root=project_root,
        config_path=config_path,
        config=config,
    )


def _run_raw_text_inventory(
    *,
    project_root: Path,
    config_path: Path,
    config: InventoryRunConfig,
) -> dict[str, JsonValue]:
    base_config_path = _resolve_pinned_file(
        project_root,
        config.base_batch_config.path,
        config.base_batch_config.sha256,
        label="inventory probe base batch config",
    )
    base_config = load_synthesis_raw_text_hybrid_batch_config(base_config_path)
    selected, source_rows, target_rows, _editor, _reviewer, plan_sha = _load_inputs(
        project_root, base_config
    )
    source_by_id = {cast(str, row["documentId"]): row for row in source_rows}
    target_by_id = {cast(str, row["baseDocumentId"]): row for row in target_rows}
    selected_by_id = {row.plan.baseDocumentId: row for row in selected}
    oracle = _load_oracle(project_root, config)
    oracle_by_id = {row.documentId: row for row in oracle.cases}
    template_profile, template_profile_path = _load_template_mutation_profile(
        project_root, config
    )
    template_profile_by_id: dict[str, TemplateMutationCase] = (
        {row.documentId: row for row in template_profile.cases}
        if template_profile is not None
        else {}
    )
    if isinstance(config, SynthesisRawTextInventoryBatchConfig):
        live_ids = (
            tuple(row.document_id for row in config.cases)
            if config.selection == "explicit_pinned_document_ids"
            else tuple(row.plan.baseDocumentId for row in selected)
        )
        if len(live_ids) != config.workflow.documents:
            raise ValueError("inventory batch input count differs from its fixed contract")
        max_concurrent_documents = config.workflow.max_concurrent_documents
    else:
        live_ids = tuple(row.document_id for row in config.cases)
        max_concurrent_documents = 1
    unknown_live_ids = tuple(
        document_id for document_id in live_ids if document_id not in selected_by_id
    )
    if unknown_live_ids:
        raise ValueError(
            "inventory probe contains a document outside the pinned target cohort: "
            f"{list(unknown_live_ids)}"
        )
    if any(document_id not in selected_by_id for document_id in oracle_by_id):
        raise ValueError("regression oracle contains a document outside the base 50 cohort")

    reference_roots = {
        "glm": _validate_reference_run(
            project_root,
            config.reference_runs.glm.path,
            config.reference_runs.glm.commit_sha256,
            config.reference_runs.glm.transaction_sha256,
        ),
        "luna": _validate_reference_run(
            project_root,
            config.reference_runs.luna.path,
            config.reference_runs.luna.commit_sha256,
            config.reference_runs.luna.transaction_sha256,
        ),
    }
    reference_rows = {
        arm: _negative_fixture_audits(
            reference_root=root,
            oracle=oracle,
            source_by_id=source_by_id,
            target_by_id=target_by_id,
        )
        for arm, root in reference_roots.items()
    }
    if any(
        cast(bool, row["newFullDocumentAuditPassed"])
        for rows in reference_rows.values()
        for row in rows
    ):
        raise RuntimeError("full-document guard failed to reject a pinned false-pass fixture")

    prompt_path = _resolve_pinned_file(
        project_root,
        config.prompt.path,
        config.prompt.sha256,
        label="inventory probe editor prompt",
    )
    prompt_bytes = read_regular_file_bytes(prompt_path)
    transaction = {
        "schemaVersion": config.schema_version,
        "runId": config.run.run_id,
        "configSha256": sha256_file(config_path),
        "baseBatchConfigSha256": config.base_batch_config.sha256,
        "regressionOracleSha256": config.regression_oracle.sha256,
        "templateMutationProfileSha256": (
            config.template_mutation_profile.sha256
            if isinstance(config, SynthesisRawTextInventoryBatchConfig)
            and config.template_mutation_profile is not None
            else None
        ),
        "referenceCommits": {
            "glm": config.reference_runs.glm.commit_sha256,
            "luna": config.reference_runs.luna.commit_sha256,
        },
        "sourceCorpusSha256": base_config.inputs.source_corpus.sha256,
        "targetSha256": base_config.inputs.synthetic_targets.sha256,
        "linguisticPlanSha256": plan_sha,
        "promptSha256": config.prompt.sha256,
        "implementationSha256": sha256_file(_IMPLEMENTATION_PATH),
        "inventoryImplementationSha256": sha256_file(
            Path(__file__).with_name("raw_text_inventory.py")
        ),
        "hybridCompilerImplementationSha256": sha256_file(
            Path(__file__).with_name("raw_text_hybrid_probe.py")
        ),
        "rewriteContractImplementationSha256": sha256_file(
            Path(__file__).with_name("raw_text_rewrite_cycle_probe.py")
        ),
        "providerRuntimeImplementationSha256": sha256_file(
            Path(__file__).with_name("linguistic_probe_runtime.py")
        ),
        "liveDocumentIds": list(live_ids),
        "runtime": {
            "pydanticAiVersion": version("pydantic-ai-slim"),
            "openaiVersion": version("openai"),
            "model": config.provider.model,
            "outputMode": config.workflow.output_mode,
            "providerOrder": list(_explicit_provider_order(config)),
            "maxConcurrentDocuments": max_concurrent_documents,
            "maxProviderRouteRounds": (
                config.workflow.max_provider_route_rounds
                if isinstance(config, SynthesisRawTextInventoryBatchConfig)
                else 1
            ),
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
    staged.publish_bytes("prompts/editor.md", prompt_bytes)
    staged.publish_json("regression/oracle.json", oracle.model_dump(mode="json"))
    if template_profile is not None and template_profile_path is not None:
        staged.publish_bytes(
            "regression/template-mutation-profile.json",
            read_regular_file_bytes(template_profile_path),
        )
    for arm, rows in reference_rows.items():
        staged.publish_bytes(
            f"regression/{arm}-negative-fixtures.jsonl",
            b"".join(canonical_json_bytes(row) + b"\n" for row in rows),
        )

    resources = build_target_integrity_resources(
        project_root=project_root,
        config=base_config,
        source_rows=source_rows,
        staged=staged,
    )
    compiled_cases: list[
        tuple[_CompiledCase, RegressionCase | None, TemplateMutationCase | None]
    ] = []
    compiler_blocks: dict[str, _CompilerBlock] = {}
    for case_number, document_id in enumerate(live_ids, start=1):
        selected_case = selected_by_id[document_id]
        try:
            state = prepare_rewrite_state(
                case_number=case_number,
                source_row=selected_case.source,
                target_row=selected_case.target,
                linguistic_plan=selected_case.plan,
                feature=selected_case.feature,
                config=base_config,
                target_integrity_resources=resources,
                defer_ambiguous_party_scalar_cardinality=True,
            )
            bundle = build_rewrite_contract_bundle(state)
            compiled = compile_case(
                bundle,
                context_lines=0,
                merge_gap_lines=0,
            )
            refined_work_items = _refine_numeric_evidence(
                compiled.workspace.current_text,
                compiled.work_items,
                compiled.bundle,
            )
            refined_work_items = _refine_sibling_evidence(
                refined_work_items,
                compiled.bundle,
            )
            refined_work_items = _refine_cargo_group_evidence(
                compiled.workspace.current_text,
                refined_work_items,
                compiled.bundle,
            )
            refined_work_items = _refine_equipment_evidence(
                compiled.workspace.current_text,
                refined_work_items,
                compiled.bundle,
            )
            refined_work_items = _refine_party_evidence(
                compiled.workspace.original_text,
                compiled.workspace.source_label,
                refined_work_items,
                compiled.bundle.surfaceRenderingRequirements,
            )
            refined_work_items = _refine_foreign_party_evidence(
                compiled.workspace.current_text,
                compiled.workspace.source_label,
                compiled.workspace.current_target_label,
                refined_work_items,
            )
            compiled = replace(compiled, work_items=refined_work_items)
            occurrence_requirements = _refine_party_occurrence_requirements(
                compiled.workspace.original_text,
                compiled.workspace.current_text,
                compiled.work_items,
                compiled.workspace.target_value_occurrence_requirements,
                compiled.workspace.source_label,
            )
            compiled.workspace.target_value_occurrence_requirements = occurrence_requirements
            compiled = replace(
                compiled,
                bundle=compiled.bundle.model_copy(
                    update={"targetValueOccurrenceRequirements": occurrence_requirements}
                ),
            )
        except ValueError as error:
            source_text = selected_case.source.get("joinedRawText")
            source_label = selected_case.source.get(base_config.inputs.source_target_field)
            target_label = selected_case.target.get("target")
            if (
                not isinstance(source_text, str)
                or not isinstance(source_label, Mapping)
                or not isinstance(target_label, Mapping)
            ):
                raise RuntimeError(
                    f"compiler-blocked case has invalid pinned inputs: {document_id}"
                ) from error
            compiler_blocks[document_id] = _CompilerBlock(
                document_id=document_id,
                source_text=source_text,
                source_label=source_label,
                target_label=target_label,
                error_type=type(error).__name__,
                error_message=str(error),
            )
            continue
        blocked = tuple(row for row in compiled.work_items if row.state == "blocked_unlocated")
        if blocked:
            compiler_blocks[document_id] = _CompilerBlock(
                document_id=document_id,
                source_text=compiled.workspace.original_text,
                source_label=compiled.workspace.source_label,
                target_label=compiled.workspace.current_target_label,
                error_type="UnlocatedCompilerWorkError",
                error_message=(
                    "semantic deltas lack exact host-owned evidence: "
                    + ", ".join(
                        path for row in blocked for path in row.targetPaths
                    )
                ),
            )
            continue
        profile_case = template_profile_by_id.get(document_id)
        if profile_case is not None:
            source_text_sha = sha256_bytes(compiled.workspace.original_text.encode("utf-8"))
            if profile_case.sourceTextSha256 != source_text_sha:
                raise ValueError(
                    "template mutation profile source text differs for " f"{document_id}"
                )
        compiled_cases.append((compiled, oracle_by_id.get(document_id), profile_case))

    if compiler_blocks:
        print(
            json.dumps(
                {
                    "command": "run-raw-text-inventory-batch",
                    "phase": "compiler_preflight",
                    "compiler_blocked_documents": len(compiler_blocks),
                    "blocks": [
                        {
                            "document_id": block.document_id,
                            "error_type": block.error_type,
                            "error_message": block.error_message,
                        }
                        for block in compiler_blocks.values()
                    ],
                    "status": "progress",
                },
                allow_nan=False,
                sort_keys=True,
            ),
            flush=True,
        )

    materials: list[_InventoryCaseMaterial] = []
    for compiled, oracle_case, profile_case in compiled_cases:
        document_id = compiled.bundle.result.documentId
        before_inventory = build_mutable_inventory(
            source_text=compiled.workspace.original_text,
            current_text=compiled.workspace.current_text,
            source_label=compiled.workspace.source_label,
            target_label=compiled.workspace.current_target_label,
            work_items=compiled.work_items,
            oracle_case=oracle_case,
            template_profile_case=profile_case,
            surface_requirements=compiled.bundle.surfaceRenderingRequirements,
            anchored_replacements=compiled.bundle.anchoredScalarReplacementRequirements,
        )
        deterministic_text, deterministic_edits = apply_deterministic_auxiliary_edits(
            text=compiled.workspace.current_text,
            document_id=document_id,
            scenario_id=compiled.workspace.scenario_id,
            candidates=before_inventory,
        )
        compiled.workspace.current_text = deterministic_text
        inventory = build_mutable_inventory(
            source_text=compiled.workspace.original_text,
            current_text=compiled.workspace.current_text,
            source_label=compiled.workspace.source_label,
            target_label=compiled.workspace.current_target_label,
            work_items=compiled.work_items,
            oracle_case=oracle_case,
            template_profile_case=profile_case,
            surface_requirements=compiled.bundle.surfaceRenderingRequirements,
            anchored_replacements=compiled.bundle.anchoredScalarReplacementRequirements,
        )
        occurrence_requirements = _include_unique_linked_party_copies(
            compiled.workspace.current_text,
            compiled.work_items,
            inventory,
            compiled.workspace.target_value_occurrence_requirements,
            compiled.workspace.source_status_requirements,
        )
        compiled.workspace.target_value_occurrence_requirements = occurrence_requirements
        compiled = replace(
            compiled,
            bundle=compiled.bundle.model_copy(
                update={"targetValueOccurrenceRequirements": occurrence_requirements}
            ),
        )
        materials.append(
            _InventoryCaseMaterial(
                compiled=compiled,
                oracle_case=oracle_case,
                inventory=inventory,
                deterministic_edits=deterministic_edits,
                slots=_model_slots(compiled, inventory, deterministic_edits),
                compound_slots=_compound_slots(compiled),
                deterministic_text_sha256=sha256_bytes(deterministic_text.encode("utf-8")),
            )
        )

    _preflight_initial_model_contracts(
        materials,
        maximum_slots=(
            config.workflow.max_initial_slots_per_request
            if isinstance(config, SynthesisRawTextInventoryBatchConfig)
            else None
        ),
        maximum_successful_responses=(
            config.workflow.max_successful_model_responses_per_document
            if isinstance(config, SynthesisRawTextInventoryBatchConfig)
            else config.workflow.max_model_requests_per_document
        ),
        profile_context_lines=(
            config.workflow.template_profile_context_lines
            if isinstance(config, SynthesisRawTextInventoryBatchConfig)
            else 0
        ),
    )

    key = load_provider_key(project_root, config.environment_file, config.provider.api_key_env)
    client = AsyncOpenAI(
        api_key=key,
        base_url=_OPENROUTER_BASE_URL if config.provider.kind == "openrouter" else None,
        max_retries=config.provider.transport_max_retries,
        timeout=config.provider.request_timeout_seconds,
        default_headers={"X-Title": "DocumentParsing"}
        if config.provider.kind == "openrouter"
        else None,
    )
    model, _ = _model_pair(
        client=client,
        editor_provider=config.provider,
        reviewer_provider=config.provider,
    )
    stages_by_id: dict[str, tuple[InventoryModelStage, ...]] = {}

    async def execute() -> tuple[tuple[InventoryProbeCaseResult, ...], int]:
        results: list[InventoryProbeCaseResult | None] = [None] * len(materials)
        for index, material in enumerate(materials):
            checkpoint = _load_inventory_checkpoint(staged=staged, material=material)
            if checkpoint is None:
                continue
            result, stages = checkpoint
            results[index] = result
            stages_by_id[result.documentId] = stages
        initially_completed = sum(row is not None for row in results)
        processed_this_invocation = 0
        limiter = asyncio.Semaphore(max_concurrent_documents)
        progress_lock = asyncio.Lock()
        started = time.perf_counter()

        async def run_one(index: int) -> None:
            nonlocal processed_this_invocation
            material = materials[index]
            compiled = material.compiled
            document_id = compiled.bundle.result.documentId
            async with limiter:
                if isinstance(config, SynthesisRawTextInventoryBatchConfig):
                    maximum_semantic_attempts = (
                        config.workflow.max_successful_model_responses_per_document
                    )
                else:
                    maximum_semantic_attempts = config.workflow.max_model_requests_per_document
                stage_rows: list[InventoryModelStage] = []
                failure: str | None = None
                committed = False
                repair_context: Mapping[str, JsonValue] | None = None
                candidate_output: dict[str, Any] = {}
                maximum_initial_slots = (
                    config.workflow.max_initial_slots_per_request
                    if isinstance(config, SynthesisRawTextInventoryBatchConfig)
                    else None
                )
                try:
                    initial_batches = _initial_request_batches(
                        material,
                        maximum_slots=maximum_initial_slots,
                    )
                except ValueError as error:
                    initial_batches = ()
                    failure = f"Initial request partition rejected the case: {error}"
                if len(initial_batches) > maximum_semantic_attempts:
                    failure = (
                        "Initial request partition exceeds the successful-response budget: "
                        f"batches={len(initial_batches)}, "
                        f"budget={maximum_semantic_attempts}"
                    )
                    initial_batches = ()
                active_slots = initial_batches[0].slots if initial_batches else ()
                active_compound_slots = (
                    initial_batches[0].compound_slots if initial_batches else ()
                )
                initial_batch_index = 0
                for semantic_attempt in (
                    range(1, maximum_semantic_attempts + 1) if initial_batches else ()
                ):
                    request_kind: Literal["initial_partition", "repair"] = (
                        "repair" if repair_context is not None else "initial_partition"
                    )
                    output, attempt_stages = await _call_model(
                        compiled=compiled,
                        slots=active_slots,
                        compound_slots=active_compound_slots,
                        model=model,
                        config=config,
                        prompt=prompt_bytes.decode("utf-8"),
                        semantic_attempt=semantic_attempt,
                        request_kind=request_kind,
                        initial_partition_ordinal=(
                            initial_batch_index + 1
                            if request_kind == "initial_partition"
                            else None
                        ),
                        initial_partition_count=len(initial_batches),
                        repair_context=repair_context,
                    )
                    if output is None:
                        stage_rows.extend(attempt_stages)
                        failure = "Every configured provider route failed before returning output."
                        break
                    candidate_output.update(output)
                    initial_batch_index += 1
                    if initial_batch_index < len(initial_batches):
                        stage_rows.extend(attempt_stages)
                        active_slots = initial_batches[initial_batch_index].slots
                        active_compound_slots = initial_batches[
                            initial_batch_index
                        ].compound_slots
                        repair_context = None
                        continue
                    try:
                        candidate = _evaluate_candidate(material, candidate_output)
                    except Exception as error:
                        (
                            active_slots,
                            active_compound_slots,
                            preview_diagnostics,
                            preview_audit,
                        ) = _preview_repair_selection(
                            material,
                            candidate_output,
                            error_message=str(error),
                        )
                        host_evaluation = InventoryHostEvaluation(
                            passed=False,
                            phase="atomic_apply",
                            errorType=type(error).__name__,
                            errorMessage=str(error),
                            requestedRepairSlots=tuple(row.alias for row in active_slots),
                            requestedRepairCompoundSlots=tuple(
                                row.alias for row in active_compound_slots
                            ),
                            hostRewriteAudit=None,
                            fullDocumentAudit=preview_audit,
                        )
                        stage_rows.extend(_attach_host_evaluation(attempt_stages, host_evaluation))
                        failure = (
                            "Host atomic apply rejected model output: "
                            f"{type(error).__name__}: {error}"
                        )
                        repair_context = {
                            "rejectionPhase": "atomic_apply",
                            "hostRejection": str(error),
                            "parallelHostDiagnostics": cast(
                                JsonValue, list(preview_diagnostics)
                            ),
                            "previousRejectedValues": cast(
                                JsonValue,
                                {
                                    row.alias: candidate_output.get(row.alias)
                                    for row in (*active_slots, *active_compound_slots)
                                },
                            ),
                        }
                        continue
                    if candidate.passed:
                        host_evaluation = InventoryHostEvaluation(
                            passed=True,
                            phase="full_document_audit",
                            errorType=None,
                            errorMessage=None,
                            requestedRepairSlots=(),
                            requestedRepairCompoundSlots=(),
                            hostRewriteAudit=candidate.host_audit,
                            fullDocumentAudit=candidate.full_audit,
                        )
                        stage_rows.extend(_attach_host_evaluation(attempt_stages, host_evaluation))
                        _commit_candidate(material, candidate)
                        committed = True
                        break
                    active_slots, active_compound_slots = _repair_selection(
                        material.slots,
                        material.compound_slots,
                        full_audit=candidate.full_audit,
                    )
                    host_evaluation = InventoryHostEvaluation(
                        passed=False,
                        phase="full_document_audit",
                        errorType="DeterministicPostconditionError",
                        errorMessage=(
                            f"{len(candidate.full_audit.findings)} full-document findings; "
                            f"core_passed={candidate.host_audit.core_passed}"
                        ),
                        requestedRepairSlots=tuple(row.alias for row in active_slots),
                        requestedRepairCompoundSlots=tuple(
                            row.alias for row in active_compound_slots
                        ),
                        hostRewriteAudit=candidate.host_audit,
                        fullDocumentAudit=candidate.full_audit,
                    )
                    stage_rows.extend(_attach_host_evaluation(attempt_stages, host_evaluation))
                    failure = (
                        "Deterministic full-document audit rejected model output: "
                        f"{len(candidate.full_audit.findings)} finding(s)."
                    )
                    repair_context = {
                        "rejectionPhase": "full_document_audit",
                        "hostFindings": cast(
                            JsonValue,
                            [
                                {
                                    "category": row.category,
                                    "lineIds": list(row.lineIds),
                                    "sourceSurface": row.sourceSurface,
                                    "targetPaths": list(row.targetPaths),
                                    "explanation": row.explanation,
                                }
                                for row in candidate.full_audit.findings
                            ],
                        ),
                        "previousRejectedValues": cast(
                            JsonValue,
                            {
                                row.alias: candidate_output.get(row.alias)
                                for row in (*active_slots, *active_compound_slots)
                            },
                        ),
                    }
                stages = tuple(stage_rows)
                result = _inventory_case_result(
                    material=material,
                    stages=stages,
                    model_output_committed=committed,
                    failure=failure,
                )
            async with progress_lock:
                _publish_inventory_checkpoint(
                    staged=staged,
                    material=material,
                    result=result,
                    stages=stages,
                )
                results[index] = result
                stages_by_id[document_id] = stages
                processed_this_invocation += 1
                completed = (
                    len(compiler_blocks) + initially_completed + processed_this_invocation
                )
                elapsed = time.perf_counter() - started
                rate = processed_this_invocation / elapsed if elapsed else 0.0
                print(
                    json.dumps(
                        {
                            "command": "run-raw-text-inventory-batch",
                            "phase": "render_and_audit",
                            "processed_documents": completed,
                            "remaining_documents": len(live_ids) - completed,
                            "training_ready_documents": sum(
                                row is not None and row.status == "training_ready"
                                for row in results
                            ),
                            "needs_review_documents": sum(
                                row is not None and row.status == "needs_review" for row in results
                            ),
                            "call_failed_documents": sum(
                                row is not None and row.status == "call_failed" for row in results
                            ),
                            "compiler_blocked_documents": len(compiler_blocks),
                            "elapsed_seconds": round(elapsed, 3),
                            "throughput_documents_per_hour": round(rate * 3600.0, 3),
                            "eta_seconds": (
                                round((len(live_ids) - completed) / rate, 3) if rate else None
                            ),
                            "status": "progress",
                        },
                        allow_nan=False,
                        sort_keys=True,
                    ),
                    flush=True,
                )

        pending = tuple(index for index, row in enumerate(results) if row is None)
        try:
            await asyncio.gather(*(run_one(index) for index in pending))
        finally:
            await client.close()
        if any(row is None for row in results):
            raise RuntimeError("inventory batch returned an incomplete result set")
        return tuple(cast(InventoryProbeCaseResult, row) for row in results), initially_completed

    execution_started = time.perf_counter()
    model_results, initially_completed = asyncio.run(execute())
    wall_seconds = time.perf_counter() - execution_started
    processed_this_invocation = len(model_results) - initially_completed
    model_result_by_id = {row.documentId: row for row in model_results}
    blocked_results = {
        document_id: _compiler_blocked_result(block)
        for document_id, block in compiler_blocks.items()
    }
    results = tuple(
        blocked_results.get(document_id) or model_result_by_id[document_id]
        for document_id in live_ids
    )
    material_by_id = {
        material.compiled.bundle.result.documentId: material for material in materials
    }
    for result in results:
        document_id = result.documentId
        material = material_by_id.get(document_id)
        if material is None:
            block = compiler_blocks[document_id]
            prefix = f"cases/{document_id}"
            staged.publish_bytes(f"{prefix}/source.txt", block.source_text.encode())
            staged.publish_json(f"{prefix}/source-label.json", block.source_label)
            staged.publish_json(f"{prefix}/target-label.json", block.target_label)
            staged.publish_json(
                f"{prefix}/compiler-block.json",
                {
                    "errorType": block.error_type,
                    "errorMessage": block.error_message,
                    "providerRequests": 0,
                },
            )
            staged.publish_bytes(f"{prefix}/final.txt", block.source_text.encode())
            staged.publish_bytes(f"{prefix}/diff.patch", b"")
            staged.publish_json(f"{prefix}/stages.json", [])
            staged.publish_json(f"{prefix}/result.json", result.model_dump(mode="json"))
            continue
        compiled = material.compiled
        prefix = f"cases/{document_id}"
        staged.publish_bytes(f"{prefix}/source.txt", compiled.workspace.original_text.encode())
        staged.publish_json(f"{prefix}/source-label.json", compiled.workspace.source_label)
        staged.publish_json(f"{prefix}/target-label.json", compiled.workspace.current_target_label)
        staged.publish_json(f"{prefix}/contract.json", compiled.bundle.model_dump(mode="json"))
        staged.publish_json(
            f"{prefix}/inventory.json",
            [row.model_dump(mode="json") for row in material.inventory],
        )
        staged.publish_json(
            f"{prefix}/deterministic-edits.json",
            [row.model_dump(mode="json") for row in material.deterministic_edits],
        )
        staged.publish_json(
            f"{prefix}/model-slots.json",
            [
                {
                    "alias": row.alias,
                    "lineId": row.line_id,
                    "sourceLine": row.source_line,
                    "requirements": list(row.requirements),
                }
                for row in material.slots
            ],
        )
        staged.publish_json(
            f"{prefix}/compound-slots.json",
            [
                {
                    "alias": row.alias,
                    "requirement": row.requirement.model_dump(mode="json"),
                }
                for row in material.compound_slots
            ],
        )
        staged.publish_json(
            f"{prefix}/editor-payload.json",
            _editor_payload(
                compiled,
                material.slots,
                material.compound_slots,
                profile_context_lines=(
                    config.workflow.template_profile_context_lines
                    if isinstance(config, SynthesisRawTextInventoryBatchConfig)
                    else 0
                ),
            ),
        )
        staged.publish_json(
            f"{prefix}/stages.json",
            [row.model_dump(mode="json") for row in stages_by_id.get(document_id, ())],
        )
        staged.publish_bytes(f"{prefix}/final.txt", compiled.workspace.current_text.encode())
        staged.publish_bytes(
            f"{prefix}/diff.patch",
            unified_text_diff(
                compiled.workspace.original_text, compiled.workspace.current_text
            ).encode(),
        )
        staged.publish_json(f"{prefix}/result.json", result.model_dump(mode="json"))

    total_usage = _combined_usage([stage for stages in stages_by_id.values() for stage in stages])
    profile_documents_selected = sum(
        document_id in template_profile_by_id for document_id in live_ids
    )
    profile_lines_selected = sum(
        len(template_profile_by_id[document_id].lines)
        for document_id in live_ids
        if document_id in template_profile_by_id
    )
    profile_model_lines = sum(
        sum(
            any(
                requirement.get("kind") == "template_profile_residual"
                for requirement in slot.requirements
            )
            for slot in material.slots
        )
        for material in materials
    )
    summary: dict[str, JsonValue] = {
        "schemaVersion": config.schema_version,
        "runId": config.run.run_id,
        "status": "complete",
        "model": config.provider.model,
        "outputMode": config.workflow.output_mode,
        "regressionDocuments": len(oracle.cases),
        "glmFalsePassesRejected": sum(
            not cast(bool, row["newFullDocumentAuditPassed"]) for row in reference_rows["glm"]
        ),
        "lunaFalsePassesRejected": sum(
            not cast(bool, row["newFullDocumentAuditPassed"]) for row in reference_rows["luna"]
        ),
        "liveDocuments": len(results),
        "trainingReadyDocuments": sum(row.status == "training_ready" for row in results),
        "needsReviewDocuments": sum(row.status == "needs_review" for row in results),
        "callFailedDocuments": sum(row.status == "call_failed" for row in results),
        "compilerBlockedDocuments": sum(row.status == "compiler_blocked" for row in results),
        "templateProfileDocumentsAvailable": len(template_profile_by_id),
        "templateProfileDocumentsSelected": profile_documents_selected,
        "templateProfileEvidenceLinesSelected": profile_lines_selected,
        "templateProfileModelLines": profile_model_lines,
        "requests": total_usage.requests,
        "providerAttempts": sum(len(stages) for stages in stages_by_id.values()),
        "failedProviderAttempts": sum(
            stage.errorType is not None for stages in stages_by_id.values() for stage in stages
        ),
        "providerCostReceiptedRequests": sum(
            stage.usage.requests
            for stages in stages_by_id.values()
            for stage in stages
            if stage.usage.providerReportedCostUsd is not None
        ),
        "providerCostReceiptCoverageFraction": (
            sum(
                stage.usage.requests
                for stages in stages_by_id.values()
                for stage in stages
                if stage.usage.providerReportedCostUsd is not None
            )
            / total_usage.requests
            if total_usage.requests
            else 1.0
        ),
        "providerRoutes": cast(JsonValue, list(_explicit_provider_order(config))),
        "downstreamProviderCounts": cast(
            JsonValue,
            {
                provider: total_usage.downstreamProviders.count(provider)
                for provider in sorted(set(total_usage.downstreamProviders))
            },
        ),
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
        "providerReportedCostPerThousandUsd": (
            str(total_usage.providerReportedCostUsd * Decimal(1000) / len(results))
            if total_usage.providerReportedCostUsd is not None
            else None
        ),
        "checkpointedDocumentsAtStart": initially_completed,
        "processedDocumentsThisInvocation": processed_this_invocation,
        "compilerBlockedBeforeProvider": len(compiler_blocks),
        "wallSeconds": round(wall_seconds, 6),
        "throughputDocumentsPerHour": round(
            processed_this_invocation / wall_seconds * 3600.0 if wall_seconds else 0.0,
            6,
        ),
        "trainingRecordsPublished": False,
    }
    staged.publish_json("summary.json", summary)
    staged.publish_bytes(
        "generation/results.jsonl",
        b"".join(canonical_json_bytes(row.model_dump(mode="json")) + b"\n" for row in results),
    )
    staged.publish_bytes(
        "REPORT.md",
        _report(
            config=config,
            reference_rows=reference_rows,
            results=results,
            stages_by_id=stages_by_id,
            profile_documents_selected=profile_documents_selected,
            profile_lines_selected=profile_lines_selected,
            profile_model_lines=profile_model_lines,
        ).encode(),
    )
    for name, payload in _plot_bytes(reference_rows, results).items():
        staged.publish_bytes(f"plots/{name}", payload)
    staged.publish_json("provenance/transaction.json", transaction)
    artifacts = _artifact_inventory(staged.stage_root)
    staged.commit(
        expected_artifacts=artifacts,
        metadata={
            "schemaVersion": 1,
            "regressionDocuments": len(oracle.cases),
            "liveDocuments": len(results),
            "trainingReadyDocuments": cast(int, summary["trainingReadyDocuments"]),
            "compilerBlockedDocuments": cast(int, summary["compilerBlockedDocuments"]),
            "trainingRecordsPublished": False,
        },
    )
    output = dict(summary)
    output["artifactRoot"] = str(staged.final_root)
    output["commitSha256"] = sha256_file(staged.final_root / "_COMMIT.json")
    return output
