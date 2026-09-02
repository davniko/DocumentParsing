"""Compose structured, route, and controlled B/L synthesis into one target.

Every upstream stage remains independently receipted, but this runner is the
single semantic hand-off to later text realization.  It requires an identical
source selection, rejects overlapping change ownership, validates the final
task schema and relational inverse, and publishes no training text.
"""

from __future__ import annotations

import json
import math
import re
import resource
import time
from collections import Counter
from collections.abc import Mapping, Sequence
from copy import deepcopy
from pathlib import Path
from typing import Annotated, Any, Literal, cast

from pydantic import BaseModel, ConfigDict, JsonValue, StringConstraints, model_validator

from document_ocr.atomic import read_regular_file_bytes
from document_ocr.hashing import canonical_json_bytes, sha256_bytes, sha256_file
from document_ocr.synthesis.anchors import leaf_items, normalized_role_path
from document_ocr.synthesis.config import SynthesisSemanticPlanConfig
from document_ocr.synthesis.run_context import ScenarioSourceRecord, SynthesisRunContext
from document_ocr.synthesis.run_safety import StagedArtifactRun
from document_ocr.synthesis.scenario_state import (
    ScenarioChange,
    ScenarioStageProposal,
    ScenarioState,
    StageProvenanceReceipt,
)
from document_ocr.synthesis.task_adapter import BILL_OF_LADING_TASK_ADAPTER

NonEmptyString = Annotated[str, StringConstraints(strip_whitespace=True, min_length=1)]
Sha256 = Annotated[str, StringConstraints(pattern=r"^[0-9a-f]{64}$")]
_PATH_TOKEN = re.compile(r"([A-Za-z_][A-Za-z0-9_]*)|\[([0-9]+)\]")


class SemanticPlanPipelineError(RuntimeError):
    """The pinned stages cannot be composed without changing their meaning."""


class _StrictFrozenModel(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, strict=True, allow_inf_nan=False)


class ComposedSemanticPlan(_StrictFrozenModel):
    """One complete semantic target awaiting linguistic/text realization."""

    schema_version: Literal[1]
    status: Literal["resolved_non_dg_semantic_plan_pending_text_and_dg"]
    scenario_id: NonEmptyString
    base_document_id: NonEmptyString
    template_id: NonEmptyString
    source_target_sha256: Sha256
    target: dict[str, JsonValue]
    target_sha256: Sha256
    state_sha256: Sha256
    scenario_state: ScenarioState
    remaining_blockers: tuple[NonEmptyString, ...]
    training_eligible: Literal[False]

    @model_validator(mode="after")
    def state_and_target_are_identical(self) -> ComposedSemanticPlan:
        if self.scenario_state.identity.scenario_id != self.scenario_id:
            raise ValueError("semantic plan scenario identity differs from its state")
        if self.scenario_state.identity.base_document_id != self.base_document_id:
            raise ValueError("semantic plan source identity differs from its state")
        if self.scenario_state.identity.template_id != self.template_id:
            raise ValueError("semantic plan template identity differs from its state")
        if self.scenario_state.source_target_sha256 != self.source_target_sha256:
            raise ValueError("semantic plan source hash differs from its state")
        if self.scenario_state.target != self.target:
            raise ValueError("semantic plan target differs from its state")
        if self.scenario_state.target_sha256 != self.target_sha256:
            raise ValueError("semantic plan target hash differs from its state")
        if self.scenario_state.content_sha256() != self.state_sha256:
            raise ValueError("semantic plan state hash differs from its payload")
        if self.remaining_blockers != tuple(sorted(set(self.remaining_blockers))):
            raise ValueError("semantic plan blockers must be unique and sorted")
        return self


def _resolve_file(project_root: Path, value: str, *, label: str) -> Path:
    candidate = Path(value)
    path = candidate if candidate.is_absolute() else project_root / candidate
    if path.is_symlink() or not path.resolve(strict=True).is_file():
        raise ValueError(f"{label} must be a regular file")
    return path.resolve(strict=True)


def _resolve_directory(project_root: Path, value: str, *, label: str) -> Path:
    candidate = Path(value)
    path = candidate if candidate.is_absolute() else project_root / candidate
    if path.is_symlink() or not path.resolve(strict=True).is_dir():
        raise ValueError(f"{label} must be a plain directory")
    return path.resolve(strict=True)


def _read_jsonl(
    path: Path, *, expected_sha256: str, expected_records: int, label: str
) -> list[dict[str, Any]]:
    if sha256_file(path) != expected_sha256:
        raise ValueError(f"{label} SHA-256 mismatch")
    rows: list[dict[str, Any]] = []
    with path.open("rb") as stream:
        for line_number, line in enumerate(stream, start=1):
            if not line.strip():
                raise ValueError(f"{label}:{line_number}: blank rows are forbidden")
            try:
                value = json.loads(line)
            except (UnicodeDecodeError, json.JSONDecodeError) as error:
                raise ValueError(f"{label}:{line_number}: invalid UTF-8 JSON") from error
            if not isinstance(value, dict):
                raise ValueError(f"{label}:{line_number}: row must be an object")
            rows.append(value)
    if len(rows) != expected_records:
        raise ValueError(f"{label} expected {expected_records} rows, found {len(rows)}")
    return rows


def _validate_committed_dependency(
    *, root: Path, commit_sha256: str, transaction_sha256: str, label: str
) -> None:
    commit = root / "_COMMIT.json"
    if sha256_file(commit) != commit_sha256:
        raise ValueError(f"{label} commit SHA-256 mismatch")
    run = StagedArtifactRun(
        output_parent=root.parent,
        run_name=root.name,
        transaction_sha256=transaction_sha256,
    )
    receipt = run.validate_committed_run()
    if receipt.transaction_sha256 != transaction_sha256:
        raise ValueError(f"{label} committed transaction differs from configuration")


def _rows_by_identity(
    rows: Sequence[Mapping[str, Any]], field: str, *, label: str
) -> dict[str, Mapping[str, Any]]:
    output: dict[str, Mapping[str, Any]] = {}
    for row_number, row in enumerate(rows, start=1):
        identity = row.get(field)
        if not isinstance(identity, str) or not identity:
            raise ValueError(f"{label} row {row_number} has no {field}")
        if identity in output:
            raise ValueError(f"{label} contains duplicate identity {identity}")
        output[identity] = row
    return output


def _leaf_map(target: Mapping[str, Any]) -> dict[str, JsonValue]:
    return cast(dict[str, JsonValue], dict(leaf_items(target)))


def _changed_paths(source: Mapping[str, Any], proposed: Mapping[str, Any]) -> tuple[str, ...]:
    source_leaves = _leaf_map(source)
    proposed_leaves = _leaf_map(proposed)
    return tuple(
        sorted(
            path
            for path in source_leaves.keys() | proposed_leaves.keys()
            if (path in source_leaves) != (path in proposed_leaves)
            or source_leaves.get(path) != proposed_leaves.get(path)
        )
    )


def _set_existing_leaf(target: dict[str, Any], path: str, value: JsonValue) -> None:
    tokens: list[str | int] = []
    consumed = ""
    for match in _PATH_TOKEN.finditer(path):
        if match.start() != len(consumed) + (1 if consumed and path[len(consumed)] == "." else 0):
            raise SemanticPlanPipelineError(f"cannot parse target path: {path}")
        field, index = match.groups()
        tokens.append(field if field is not None else int(cast(str, index)))
        consumed = path[: match.end()]
    if not tokens or consumed != path:
        raise SemanticPlanPipelineError(f"cannot parse target path: {path}")
    current: Any = target
    for token in tokens[:-1]:
        if isinstance(token, str):
            if not isinstance(current, dict) or token not in current:
                raise SemanticPlanPipelineError(f"target path is absent: {path}")
            current = current[token]
        else:
            if not isinstance(current, list) or not 0 <= token < len(current):
                raise SemanticPlanPipelineError(f"target list path is absent: {path}")
            current = current[token]
    final = tokens[-1]
    if isinstance(final, str):
        if not isinstance(current, dict) or final not in current:
            raise SemanticPlanPipelineError(f"target path is absent: {path}")
        current[final] = value
    else:
        if not isinstance(current, list) or not 0 <= final < len(current):
            raise SemanticPlanPipelineError(f"target list path is absent: {path}")
        current[final] = value


def _apply_leaf_change(
    target: dict[str, Any],
    path: str,
    *,
    present: bool,
    value: JsonValue,
) -> None:
    """Apply one exact scalar presence/value change without changing array topology."""

    tokens: list[str | int] = []
    consumed = ""
    for match in _PATH_TOKEN.finditer(path):
        if match.start() != len(consumed) + (1 if consumed and path[len(consumed)] == "." else 0):
            raise SemanticPlanPipelineError(f"cannot parse target path: {path}")
        field, index = match.groups()
        tokens.append(field if field is not None else int(cast(str, index)))
        consumed = path[: match.end()]
    if not tokens or consumed != path:
        raise SemanticPlanPipelineError(f"cannot parse target path: {path}")
    current: Any = target
    for token in tokens[:-1]:
        if isinstance(token, str):
            if not isinstance(current, dict) or token not in current:
                raise SemanticPlanPipelineError(f"target parent path is absent: {path}")
            current = current[token]
        else:
            if not isinstance(current, list) or not 0 <= token < len(current):
                raise SemanticPlanPipelineError(f"target parent list path is absent: {path}")
            current = current[token]
    final = tokens[-1]
    if isinstance(final, int):
        if not isinstance(current, list) or not 0 <= final < len(current):
            raise SemanticPlanPipelineError(f"target list path is absent: {path}")
        if not present:
            raise SemanticPlanPipelineError(
                f"leaf removal cannot change array cardinality: {path}"
            )
        current[final] = value
        return
    if not isinstance(current, dict):
        raise SemanticPlanPipelineError(f"target parent is not an object: {path}")
    if present:
        current[final] = value
    elif final in current:
        del current[final]
    else:
        raise SemanticPlanPipelineError(f"target leaf is already absent: {path}")


def _structured_change_contract(family: str) -> tuple[str, str]:
    if family in {
        "document_identifier",
        "voyage_identifier",
        "container_identifier",
        "seal_identifier",
    }:
        return "identifier", "exact_scalar"
    if family == "document_date":
        return "temporal", "formatted_date"
    if family in {"package_quantity", "cargo_measure", "allocation_quantity"}:
        return "numeric", "formatted_number"
    if family == "allocation_reference":
        return "relationship", "exact_scalar"
    raise SemanticPlanPipelineError(f"unknown structured semantic family: {family}")


def _controlled_change_contract(path: str) -> tuple[str, str, str, str]:
    role = normalized_role_path(path)
    if role.startswith("documentPatch.route.") or role.startswith("documentPatch.placeOfIssue."):
        return "geography", "exact_scalar", "route_conditioned_shipment_scenario_v1", "route"
    if role.startswith("documentPatch.parties.") and role.endswith((".city", ".country")):
        return "geography", "exact_scalar", "route_conditioned_party_locality_v1", "route"
    if role.startswith("documentPatch.freight."):
        return "categorical", "exact_scalar", "route_conditioned_freight_v1", "freight"
    if role == "documentPatch.transport.vesselName":
        return (
            "identifier",
            "exact_scalar",
            "public_cargo_vessel_registry_uniform_v1",
            "transport",
        )
    if role.endswith((".typeCategory", ".typeDescription")) and ".cargoPackages[]" in role:
        return "categorical", "exact_scalar", "role_conditioned_package_category_v2", "packages"
    if role.endswith(".typeDescription") and ".containers[]" in role:
        return "categorical", "exact_scalar", "registry_conditioned_equipment_type_v1", "equipment"
    if ".temperatureSetpoint." in role:
        return "numeric", "formatted_number", "reefer_conditioned_temperature_v1", "equipment"
    if ".cargoGroups[].hsCodes[]" in role:
        return "categorical", "exact_scalar", "registry_hs6_plus_national_extension_v1", "cargo_hs"
    if ".cargoGroups[].origin." in role:
        return "geography", "exact_scalar", "route_conditioned_cargo_origin_v2", "cargo_origin"
    raise SemanticPlanPipelineError(f"controlled stage changed an unowned path: {path}")


def _proposal(
    *,
    state: ScenarioState,
    stage_id: str,
    contract_id: str,
    target: dict[str, Any],
    change_metadata: Mapping[str, tuple[str, str, str, str]],
    provenance: Sequence[StageProvenanceReceipt],
) -> ScenarioStageProposal:
    old_leaves = _leaf_map(state.target)
    new_leaves = _leaf_map(target)
    paths = _changed_paths(state.target, target)
    changes = []
    for path in paths:
        try:
            kind, rendering, method, coupling = change_metadata[path]
        except KeyError as error:
            raise SemanticPlanPipelineError(f"stage has no metadata for path: {path}") from error
        changes.append(
            ScenarioChange.model_validate(
                {
                    "target_path": path,
                    "role_path": normalized_role_path(path),
                    "stage_id": stage_id,
                    "change_kind": kind,
                    "rendering_kind": rendering,
                    "old_present": path in old_leaves,
                    "old_value": old_leaves.get(path),
                    "new_present": path in new_leaves,
                    "new_value": new_leaves.get(path),
                    "method": method,
                    "coupling_group": coupling,
                },
                strict=True,
            )
        )
    return ScenarioStageProposal.model_validate(
        {
            "schema_version": 1,
            "stage_id": stage_id,
            "contract_id": contract_id,
            "input_state_sha256": state.content_sha256(),
            "target": target,
            "target_sha256": sha256_bytes(canonical_json_bytes(target)),
            "changes": tuple(row.model_dump(mode="python") for row in changes),
            "provenance": tuple(row.model_dump(mode="python") for row in provenance),
        },
        strict=True,
    )


def compose_semantic_targets(
    *,
    context: SynthesisRunContext,
    base_document_id: str,
    variant_index: int,
    seed: int,
    structured_target: Mapping[str, Any],
    structured_plan: Mapping[str, Any],
    controlled_target: Mapping[str, Any],
    structured_provenance: Sequence[StageProvenanceReceipt],
    controlled_provenance: Sequence[StageProvenanceReceipt],
) -> ScenarioState:
    """Compose two disjoint source-relative target mutations with exact ledgers."""

    initial = context.initial_state(
        base_document_id=base_document_id,
        seed=seed,
        variant_index=variant_index,
    )
    source = initial.target
    structured_paths = _changed_paths(source, structured_target)
    declared = structured_plan.get("changes")
    if not isinstance(declared, list):
        raise SemanticPlanPipelineError("structured plan has no change ledger")
    structured_rows = {cast(str, row["target_path"]): row for row in declared}
    if tuple(sorted(structured_rows)) != structured_paths:
        raise SemanticPlanPipelineError("structured target differs from its declared change ledger")
    structured_metadata: dict[str, tuple[str, str, str, str]] = {}
    for path in structured_paths:
        row = structured_rows[path]
        kind, rendering = _structured_change_contract(cast(str, row["family"]))
        structured_metadata[path] = (
            kind,
            rendering,
            cast(str, row["method"]),
            cast(str, row["coupling_group"]),
        )
    structured_proposal = _proposal(
        state=initial,
        stage_id="structured",
        contract_id="structured_numeric_identifier_temporal_v1",
        target=deepcopy(dict(structured_target)),
        change_metadata=structured_metadata,
        provenance=structured_provenance,
    )
    structured_state = context.advance(state=initial, proposal=structured_proposal).output_state

    controlled_paths = _changed_paths(source, controlled_target)
    overlap = sorted(set(structured_paths) & set(controlled_paths))
    if overlap:
        raise SemanticPlanPipelineError(f"semantic stages overlap change ownership: {overlap}")
    final_target = deepcopy(structured_state.target)
    controlled_leaves = _leaf_map(controlled_target)
    controlled_metadata: dict[str, tuple[str, str, str, str]] = {}
    for path in controlled_paths:
        _apply_leaf_change(
            final_target,
            path,
            present=path in controlled_leaves,
            value=controlled_leaves.get(path),
        )
        controlled_metadata[path] = _controlled_change_contract(path)
    controlled_proposal = _proposal(
        state=structured_state,
        stage_id="controlled-semantics",
        contract_id="route_and_controlled_semantics_v1",
        target=final_target,
        change_metadata=controlled_metadata,
        provenance=controlled_provenance,
    )
    return context.advance(
        state=structured_state,
        proposal=controlled_proposal,
    ).output_state


def _jsonl(rows: Sequence[Mapping[str, Any]]) -> bytes:
    return b"".join(canonical_json_bytes(dict(row)) + b"\n" for row in rows)


def _publish_runtime_once(
    stage: StagedArtifactRun, payload: Mapping[str, float]
) -> dict[str, float]:
    root = stage.final_root if stage.completed else stage.stage_root
    path = root / "runtime.json"
    if path.exists():
        value = json.loads(read_regular_file_bytes(path))
        if not isinstance(value, dict):
            raise SemanticPlanPipelineError("runtime receipt is malformed")
        return {field: float(value[field]) for field in ("elapsedSeconds", "peakRssMiB")}
    runtime = {field: float(value) for field, value in payload.items()}
    if any(not math.isfinite(value) or value < 0 for value in runtime.values()):
        raise SemanticPlanPipelineError("runtime receipt is invalid")
    stage.publish_json("runtime.json", runtime)
    return runtime


def run_semantic_plan_pipeline(
    *, project_root: Path, config_path: Path, config: SynthesisSemanticPlanConfig
) -> dict[str, Any]:
    started = time.perf_counter()
    structured_root = _resolve_directory(
        project_root, config.inputs.structured_run.path, label="structured run"
    )
    controlled_root = _resolve_directory(
        project_root, config.inputs.controlled_run.path, label="controlled run"
    )
    _validate_committed_dependency(
        root=structured_root,
        commit_sha256=config.inputs.structured_run.commit_sha256,
        transaction_sha256=config.inputs.structured_run.transaction_sha256,
        label="structured run",
    )
    _validate_committed_dependency(
        root=controlled_root,
        commit_sha256=config.inputs.controlled_run.commit_sha256,
        transaction_sha256=config.inputs.controlled_run.transaction_sha256,
        label="controlled run",
    )
    selection_rows = _read_jsonl(
        _resolve_file(
            project_root,
            config.inputs.structured_selection.path,
            label="structured selection",
        ),
        expected_sha256=config.inputs.structured_selection.sha256,
        expected_records=config.inputs.structured_selection.records,
        label="structured selection",
    )
    structured_targets = _read_jsonl(
        _resolve_file(
            project_root, config.inputs.structured_targets.path, label="structured targets"
        ),
        expected_sha256=config.inputs.structured_targets.sha256,
        expected_records=config.inputs.structured_targets.records,
        label="structured targets",
    )
    structured_plans = _read_jsonl(
        _resolve_file(project_root, config.inputs.structured_plans.path, label="structured plans"),
        expected_sha256=config.inputs.structured_plans.sha256,
        expected_records=config.inputs.structured_plans.records,
        label="structured plans",
    )
    controlled_scenarios = _read_jsonl(
        _resolve_file(
            project_root,
            config.inputs.controlled_scenarios.path,
            label="controlled scenarios",
        ),
        expected_sha256=config.inputs.controlled_scenarios.sha256,
        expected_records=config.inputs.controlled_scenarios.records,
        label="controlled scenarios",
    )
    controlled_targets = _read_jsonl(
        _resolve_file(
            project_root, config.inputs.controlled_targets.path, label="controlled targets"
        ),
        expected_sha256=config.inputs.controlled_targets.sha256,
        expected_records=config.inputs.controlled_targets.records,
        label="controlled targets",
    )

    source_path = _resolve_file(project_root, config.source.file.path, label="source corpus")
    source_rows = _read_jsonl(
        source_path,
        expected_sha256=config.source.file.sha256,
        expected_records=config.source.file.records,
        label="source corpus",
    )
    document_field = config.source.fields.document_id
    target_field = config.source.fields.target
    raw_hash_field = cast(str, config.source.fields.input_sha256)
    source_by_id = _rows_by_identity(source_rows, document_field, label="source corpus")
    ordered_ids = tuple(cast(str, row.get("document_id")) for row in selection_rows)
    if len(ordered_ids) != len(set(ordered_ids)) or any(not value for value in ordered_ids):
        raise ValueError("structured selection identities are invalid or duplicated")
    structured_target_by_id = _rows_by_identity(
        structured_targets, "baseDocumentId", label="structured targets"
    )
    structured_plan_by_id = _rows_by_identity(
        structured_plans, "base_document_id", label="structured plans"
    )
    controlled_scenario_by_id = _rows_by_identity(
        controlled_scenarios, "baseDocumentId", label="controlled scenarios"
    )
    controlled_target_by_id = _rows_by_identity(
        controlled_targets, "baseDocumentId", label="controlled targets"
    )
    expected_set = set(ordered_ids)
    for label, values in (
        ("structured targets", structured_target_by_id),
        ("structured plans", structured_plan_by_id),
        ("controlled scenarios", controlled_scenario_by_id),
        ("controlled targets", controlled_target_by_id),
    ):
        if set(values) != expected_set:
            raise SemanticPlanPipelineError(f"{label} selection differs from structured source")

    source_records = []
    for row in selection_rows:
        document_id = cast(str, row["document_id"])
        source = source_by_id[document_id]
        target = source.get(target_field)
        raw_hash = source.get(raw_hash_field)
        template_id = row.get("template_id")
        if not isinstance(target, dict) or not isinstance(raw_hash, str):
            raise ValueError(f"source row is incomplete: {document_id}")
        if not isinstance(template_id, str) or not template_id:
            raise ValueError(f"selection template is incomplete: {document_id}")
        source_records.append(
            ScenarioSourceRecord.from_target(
                document_id=document_id,
                template_id=template_id,
                target=target,
                source_raw_text_sha256=raw_hash,
            )
        )
    context = SynthesisRunContext(
        task_adapter=BILL_OF_LADING_TASK_ADAPTER,
        scenario_namespace=config.generation.scenario_namespace,
        source_records=source_records,
    )
    structured_provenance = (
        StageProvenanceReceipt(
            name="structured-commit",
            sha256=config.inputs.structured_run.commit_sha256,
        ),
        StageProvenanceReceipt(
            name="structured-transaction",
            sha256=config.inputs.structured_run.transaction_sha256,
        ),
    )
    controlled_provenance = (
        StageProvenanceReceipt(
            name="controlled-commit",
            sha256=config.inputs.controlled_run.commit_sha256,
        ),
        StageProvenanceReceipt(
            name="controlled-transaction",
            sha256=config.inputs.controlled_run.transaction_sha256,
        ),
    )

    plans: list[dict[str, Any]] = []
    stage_change_counts: Counter[str] = Counter()
    role_change_counts: Counter[str] = Counter()
    blocker_counts: Counter[str] = Counter()
    dangerous_documents = 0
    for position, document_id in enumerate(ordered_ids):
        structured_target_row = structured_target_by_id[document_id]
        structured_plan = structured_plan_by_id[document_id]
        controlled_target_row = controlled_target_by_id[document_id]
        controlled_scenario = controlled_scenario_by_id[document_id]
        structured_target = structured_target_row.get("target")
        controlled_target = controlled_target_row.get("target")
        if not isinstance(structured_target, dict) or not isinstance(controlled_target, dict):
            raise ValueError(f"semantic stage target is missing: {document_id}")
        if structured_target_row.get("targetSha256") != sha256_bytes(
            canonical_json_bytes(structured_target)
        ):
            raise SemanticPlanPipelineError(f"structured target hash differs: {document_id}")
        if controlled_target_row.get("draftTargetSha256") != sha256_bytes(
            canonical_json_bytes(controlled_target)
        ):
            raise SemanticPlanPipelineError(f"controlled target hash differs: {document_id}")
        if controlled_scenario.get("draftTargetSha256") != controlled_target_row.get(
            "draftTargetSha256"
        ):
            raise SemanticPlanPipelineError(f"controlled scenario hash differs: {document_id}")
        template_id = cast(str, selection_rows[position]["template_id"])
        stage_rows: tuple[Mapping[str, Any], ...] = (
            structured_target_row,
            structured_plan,
            controlled_target_row,
            controlled_scenario,
        )
        for stage_row in stage_rows:
            candidate_template = stage_row.get("templateId", stage_row.get("template_id"))
            if candidate_template != template_id:
                raise SemanticPlanPipelineError(f"stage template differs: {document_id}")
        state = compose_semantic_targets(
            context=context,
            base_document_id=document_id,
            variant_index=position,
            seed=config.generation.seed,
            structured_target=structured_target,
            structured_plan=structured_plan,
            controlled_target=controlled_target,
            structured_provenance=structured_provenance,
            controlled_provenance=controlled_provenance,
        )
        blockers = controlled_scenario.get("remainingBlockers")
        if not isinstance(blockers, list) or any(
            not isinstance(value, str) or not value for value in blockers
        ):
            raise ValueError(f"controlled scenario blockers are invalid: {document_id}")
        dangerous = controlled_scenario.get("dangerousGoods")
        if not isinstance(dangerous, dict):
            raise ValueError(f"controlled dangerous-goods status is absent: {document_id}")
        dangerous_documents += int(dangerous.get("sourceRecordCount", 0) > 0)
        blockers_tuple = tuple(sorted(set(blockers)))
        blocker_counts.update(blockers_tuple)
        for change in state.changes:
            stage_change_counts[change.stage_id] += 1
            role_change_counts[change.role_path] += 1
        plan = ComposedSemanticPlan.model_validate(
            {
                "schema_version": 1,
                "status": "resolved_non_dg_semantic_plan_pending_text_and_dg",
                "scenario_id": state.identity.scenario_id,
                "base_document_id": document_id,
                "template_id": template_id,
                "source_target_sha256": state.source_target_sha256,
                "target": state.target,
                "target_sha256": state.target_sha256,
                "state_sha256": state.content_sha256(),
                "scenario_state": state.model_dump(mode="python"),
                "remaining_blockers": blockers_tuple,
                "training_eligible": False,
            },
            strict=True,
        )
        plans.append(plan.model_dump(mode="json"))

    distribution = {
        "documents": len(plans),
        "distinctTemplates": len({row["template_id"] for row in plans}),
        "dangerousGoodsDocumentsDeferred": dangerous_documents,
        "stageChangeCounts": dict(sorted(stage_change_counts.items())),
        "roleChangeCounts": dict(sorted(role_change_counts.items())),
        "remainingBlockerCounts": dict(sorted(blocker_counts.items())),
    }
    validation = {
        "selectedDocuments": len(ordered_ids),
        "composedPlans": len(plans),
        "strictSchemaValidTargets": len(plans),
        "relationalInverseValidTargets": len(plans),
        "exactTwoStageStateChains": len(plans),
        "overlappingChangedPaths": 0,
        "trainingRecordsPublished": 0,
    }
    implementation = {
        Path(__file__).name: sha256_file(Path(__file__)),
        "scenario_state.py": sha256_file(Path(__file__).with_name("scenario_state.py")),
        "run_context.py": sha256_file(Path(__file__).with_name("run_context.py")),
    }
    transaction = sha256_bytes(
        canonical_json_bytes(
            {
                "contract": "mpci-bl-composed-semantic-plan-v1",
                "configSha256": sha256_file(config_path),
                "config": config.model_dump(mode="json"),
                "sourceSha256": config.source.file.sha256,
                "structuredCommitSha256": config.inputs.structured_run.commit_sha256,
                "controlledCommitSha256": config.inputs.controlled_run.commit_sha256,
                "implementation": implementation,
            }
        )
    )
    output_parent = Path(config.run.output_dir)
    if not output_parent.is_absolute():
        output_parent = project_root / output_parent
    stage = StagedArtifactRun(
        output_parent=output_parent,
        run_name=config.run.run_id,
        transaction_sha256=transaction,
    )
    if stage.completed:
        stage.validate_committed_run()
        return cast(dict[str, Any], json.loads((stage.final_root / "manifest.json").read_bytes()))
    stage.recover_interrupted_temporary_files()
    stage.publish_bytes("config.yaml", read_regular_file_bytes(config_path))
    stage.publish_json("source-context.json", context.receipt.model_dump(mode="json"))
    stage.publish_bytes("generation/semantic-plans.jsonl", _jsonl(plans))
    stage.publish_json("generation/distribution-summary.json", distribution)
    stage.publish_json("generation/validation-summary.json", validation)
    runtime = _publish_runtime_once(
        stage,
        {
            "elapsedSeconds": time.perf_counter() - started,
            "peakRssMiB": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024,
        },
    )
    manifest_status = "composed_non_dg_semantic_plan_complete_pending_text_and_dg"
    manifest = cast(
        dict[str, JsonValue],
        {
            "schemaVersion": 1,
            "runId": config.run.run_id,
            "status": manifest_status,
            "trainingEligible": False,
            "scenarioDocuments": len(plans),
            "trainingRecordsPublished": 0,
            "dangerousGoodsPolicy": config.generation.dangerous_goods_policy,
            "distribution": distribution,
            "validation": validation,
            "implementationSha256": implementation,
            "transactionSha256": transaction,
        },
    )
    stage.publish_json("manifest.json", manifest)
    commit = stage.commit(
        expected_artifacts=(
            "config.yaml",
            "generation/distribution-summary.json",
            "generation/semantic-plans.jsonl",
            "generation/validation-summary.json",
            "manifest.json",
            "runtime.json",
            "source-context.json",
        ),
        metadata={
            "status": manifest_status,
            "trainingEligible": False,
            "scenarioDocuments": len(plans),
        },
    )
    return {
        **manifest,
        "commitCreated": commit.created,
        "commitContentSha256": commit.receipt.content_sha256,
        "runtime": runtime,
    }
