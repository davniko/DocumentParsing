from __future__ import annotations

import asyncio
import json
import os
import re
import resource
import time
from collections import Counter
from collections.abc import Mapping, Sequence
from datetime import UTC, datetime
from decimal import Decimal
from pathlib import Path
from typing import Any, cast

import yaml
from document_ocr.atomic import json_artifact_bytes, read_regular_file_bytes
from document_ocr.hashing import canonical_json_bytes, sha256_bytes, sha256_file
from document_ocr.synthesis.run_safety import StagedArtifactRun

from .agents import AgentRuntime
from .host import (
    SpanDraft,
    anchor_drafts,
    anchor_summary,
    annotated_source,
    apply_anchor_overrides,
    apply_critic_patch,
    capability_contract,
    certify_template,
    line_spans,
    load_jsonl,
    masked_source,
    materialize_semantic_only_target_facts,
    normalize_compact_equipment_locality,
    normalize_country_code_locality,
    normalize_deterministic_draft_semantics,
    normalize_target_cobindings,
    numbered_source,
    required_target_cobindings,
    resolve_agent_proposals,
    risk_candidates,
    source_carrier,
    target_fact_components,
    target_path_relationship,
    template_summary,
    uncovered_risks,
    validate_agent_proposal_paths,
    validate_binding_realizations,
    validate_carrier_assessment,
    validate_repeated_binding_fact_topology,
    validate_target_binding_relationships,
    verify_file,
)
from .models import (
    AgentContractProtocol,
    AgentStageArtifact,
    CarrierAssessment,
    CertifiedSemanticTemplate,
    CompilerAgentOutput,
    CriticAgentOutput,
    DraftCheckpointRow,
    ExtractionCaseResult,
    ExtractionConfig,
    ExtractionStateCheckpoint,
    SelectionManifest,
    SemanticOnlyTargetFact,
)
from .selection import build_selection_manifest, carrier_resolution_classification


def project_root_from_config(config_path: Path) -> Path:
    current = config_path.resolve(strict=True).parent
    while current != current.parent:
        if (current / "pyproject.toml").is_file() and (current / "src/document_ocr").is_dir():
            return current
        current = current.parent
    raise ValueError("cannot locate repository root from extraction config")


def resolve_input(project_root: Path, configured: str) -> Path:
    path = (project_root / configured).resolve(strict=True)
    if path != project_root and project_root not in path.parents:
        raise ValueError(f"configured path escapes project root: {configured}")
    if path.is_symlink() or not path.is_file():
        raise ValueError(f"configured input is not a regular file: {configured}")
    return path


def load_config(path: Path) -> ExtractionConfig:
    payload = yaml.safe_load(read_regular_file_bytes(path))
    return ExtractionConfig.model_validate_json(
        json.dumps(payload, ensure_ascii=False, default=str)
    )


def _load_resume_run(
    *,
    project_root: Path,
    config_path: Path,
    config: ExtractionConfig,
    manifest: SelectionManifest,
) -> tuple[Path | None, dict[str, ExtractionCaseResult], bool, bool]:
    pinned = config.resume_from
    if pinned is None:
        return None, {}, False, False
    run_root = (project_root / pinned.path).resolve(strict=True)
    if project_root not in run_root.parents or run_root.is_symlink() or not run_root.is_dir():
        raise ValueError(f"resume run is not a regular project directory: {pinned.path}")
    commit_path = run_root / "_COMMIT.json"
    verify_file(commit_path, pinned.commit_sha256)
    commit = json.loads(read_regular_file_bytes(commit_path))
    if not isinstance(commit, Mapping) or not isinstance(commit.get("artifacts"), list):
        raise ValueError("resume run commit manifest is invalid")
    artifact_hashes: dict[str, str] = {}
    for item in commit["artifacts"]:
        if not isinstance(item, Mapping):
            raise ValueError("resume run commit contains a non-object artifact row")
        relative = item.get("relative_path")
        expected = item.get("sha256")
        if not isinstance(relative, str) or not isinstance(expected, str):
            raise ValueError("resume run commit artifact row is incomplete")
        artifact_path = (run_root / relative).resolve(strict=True)
        if run_root not in artifact_path.parents or artifact_path.is_symlink():
            raise ValueError(f"resume artifact escapes or aliases its run: {relative}")
        verify_file(artifact_path, expected)
        artifact_hashes[relative] = expected

    prior_config_path = run_root / "config.json"
    prior_selection_path = run_root / "selection-manifest.json"
    prior_results_path = run_root / "results.jsonl"
    for required in ("config.json", "selection-manifest.json", "results.jsonl"):
        if required not in artifact_hashes:
            raise ValueError(f"resume commit omits required artifact: {required}")
    prior_config = json.loads(read_regular_file_bytes(prior_config_path))
    if not isinstance(prior_config, Mapping):
        raise ValueError("resume config is not an object")
    for field in ("task", "phase", "inputs", "selection_seed"):
        current_value = config.model_dump(mode="json")[field]
        if prior_config.get(field) != current_value:
            raise ValueError(f"resume run differs in pinned extraction field: {field}")
    prompt_changed = prior_config.get("prompts") != config.model_dump(mode="json")["prompts"]
    if prompt_changed and not pinned.allow_prompt_change:
        raise ValueError(
            "resume run differs in pinned extraction field: prompts; set "
            "resume_from.allow_prompt_change=true only when prior outputs will be replayed "
            "under the current host and independently re-reviewed"
        )
    prior_contract_path = run_root / "resume-contract.json"
    contract_matched = False
    if "resume-contract.json" in artifact_hashes:
        prior_contract = json.loads(read_regular_file_bytes(prior_contract_path))
        if not isinstance(prior_contract, Mapping):
            raise ValueError("resume contract is not an object")
        contract_matched = prior_contract == _resume_contract(config_path)
    prior_manifest = SelectionManifest.model_validate_json(
        read_regular_file_bytes(prior_selection_path), strict=True
    )
    current_identity = tuple(
        (row.ordinal, row.document_id, row.source_sha256) for row in manifest.rows
    )
    prior_identity = tuple(
        (row.ordinal, row.document_id, row.source_sha256) for row in prior_manifest.rows
    )
    if prior_identity != current_identity:
        raise ValueError("resume selection identity differs from the current pinned selection")
    results = {
        row.document_id: row
        for row in (
            ExtractionCaseResult.model_validate_json(
                json.dumps(item, ensure_ascii=False, separators=(",", ":"))
            )
            for item in load_jsonl(prior_results_path, len(manifest.rows))
        )
    }
    expected_ids = {row.document_id for row in manifest.rows}
    if set(results) != expected_ids or len(results) != len(manifest.rows):
        raise ValueError("resume results do not cover the current selection exactly once")
    return run_root, results, prompt_changed, contract_matched


def _config_inputs(
    config: ExtractionConfig, project_root: Path
) -> tuple[
    list[dict[str, Any]],
    list[dict[str, Any]],
    list[dict[str, Any]],
    list[dict[str, Any]],
    Path,
    Path,
    Path,
    str,
    str,
]:
    source_path = resolve_input(project_root, config.inputs.source_corpus.path)
    feature_path = resolve_input(project_root, config.inputs.document_features.path)
    anchor_path = resolve_input(project_root, config.inputs.anchors.path)
    target_path = resolve_input(project_root, config.inputs.current_targets.path)
    findings_path = resolve_input(project_root, config.inputs.outcome_findings.path)
    invariants_path = resolve_input(project_root, config.inputs.outcome_invariants.path)
    manual_path = resolve_input(project_root, config.inputs.manual_audit.path)
    compiler_prompt_path = resolve_input(project_root, config.prompts.compiler.path)
    critic_prompt_path = resolve_input(project_root, config.prompts.critic.path)
    for path, expected in (
        (source_path, config.inputs.source_corpus.sha256),
        (feature_path, config.inputs.document_features.sha256),
        (anchor_path, config.inputs.anchors.sha256),
        (target_path, config.inputs.current_targets.sha256),
        (findings_path, config.inputs.outcome_findings.sha256),
        (invariants_path, config.inputs.outcome_invariants.sha256),
        (manual_path, config.inputs.manual_audit.sha256),
        (compiler_prompt_path, config.prompts.compiler.sha256),
        (critic_prompt_path, config.prompts.critic.sha256),
    ):
        verify_file(path, expected)
    return (
        load_jsonl(source_path, config.inputs.source_corpus.records),
        load_jsonl(feature_path, config.inputs.document_features.records),
        load_jsonl(anchor_path),
        load_jsonl(target_path, config.inputs.current_targets.records),
        findings_path,
        invariants_path,
        manual_path,
        compiler_prompt_path.read_text(encoding="utf-8"),
        critic_prompt_path.read_text(encoding="utf-8"),
    )


def _mark_host_rejected(stage: AgentStageArtifact, error: Exception) -> AgentStageArtifact:
    return stage.model_copy(
        update={
            "status": "host_rejected",
            "error_type": type(error).__name__,
            "error_message": str(error),
        }
    )


def _latest_compiler_revision_context(
    *,
    drafts: Sequence[SpanDraft] | None,
    prior_output: CompilerAgentOutput | None,
    compiler_stages: Sequence[AgentStageArtifact],
    resume_replay_warning: str | None,
) -> str | None:
    """Recover the latest actionable host defect, ignoring later provider failures.

    A provider/schema failure has no candidate output and therefore cannot supersede the most
    recent host rejection. Losing that rejection caused resumed runs to request another complete
    compilation instead of a bounded patch against the retained candidate.
    """

    if drafts is not None or prior_output is None:
        return None
    if resume_replay_warning is not None:
        return "Current host replay rejected the prior compiler candidate: " + resume_replay_warning
    prior_rejection = next(
        (
            stage
            for stage in reversed(compiler_stages)
            if stage.status == "host_rejected" and stage.error_message is not None
        ),
        None,
    )
    if prior_rejection is None:
        raise ValueError(
            "compiler lineage retains an invalid candidate without an actionable host rejection"
        )
    return "Host rejected the compiler output: " + prior_rejection.error_message


def _compiler_host_rejection(
    *,
    output: CompilerAgentOutput,
    raw: str,
    source_target: Mapping[str, Any],
    anchors: Sequence[SpanDraft],
    primary_error: Exception,
) -> ValueError:
    """Return one repair instruction containing every independently provable defect.

    The full materialization gate remains authoritative. These independent checks deliberately
    avoid changing or partially accepting candidate state; they only expose defects that can be
    proven even when the first full-state validation stopped at an unrelated error.
    """

    diagnostics: list[str] = []

    def add(scope: str, error: Exception) -> None:
        diagnostic = f"{scope}: {error}"
        if diagnostic not in diagnostics:
            diagnostics.append(diagnostic)

    add("complete candidate validation", primary_error)
    if not output.all_shipment_dependent_surfaces_accounted_for:
        add(
            "completion declaration",
            ValueError(
                "candidate declared itself incomplete; enumerate concrete unresolved items or "
                "submit an empty unresolved list for independent host and critic validation"
            ),
        )
    try:
        validate_agent_proposal_paths(proposals=output.bindings, source_target=source_target)
    except Exception as error:
        add("binding target paths", error)

    declared_semantic_only: tuple[SemanticOnlyTargetFact, ...] | None = None
    try:
        declared_semantic_only = materialize_semantic_only_target_facts(
            proposals=output.semantic_only_target_facts,
            source_target=source_target,
            provenance="compiler_audited_unprinted",
        )
    except Exception as error:
        add("semantic-only declarations", error)

    try:
        proposed = resolve_agent_proposals(raw=raw, proposals=output.bindings)
    except Exception as error:
        add("binding source occurrences", error)
    else:
        if declared_semantic_only is not None:
            try:
                apply_anchor_overrides(
                    raw=raw,
                    source_target=source_target,
                    anchors=anchors,
                    proposed=proposed,
                    overrides=output.anchor_overrides,
                    semantic_only_target_paths=tuple(
                        fact.target_path for fact in declared_semantic_only
                    ),
                )
            except Exception as error:
                add("anchor replacement integration", error)

        try:
            validate_repeated_binding_fact_topology(
                drafts=proposed,
                source_target=source_target,
            )
        except Exception as error:
            add("binding target topology", error)

        proposed_by_key: dict[str, list[SpanDraft]] = {}
        for draft in proposed:
            proposed_by_key.setdefault(draft.logical_key, []).append(draft)
        container_number_context = tuple(
            draft
            for draft in (*anchors, *proposed)
            if any(path.endswith(".containerNumber") for path in draft.target_paths)
        )
        for logical_key, logical_drafts in proposed_by_key.items():
            normalized_group: Sequence[SpanDraft] = logical_drafts
            if logical_drafts[0].value_kind == "equipment":
                original_spans = {(draft.char_start, draft.char_end) for draft in logical_drafts}
                context = {
                    draft.draft_id: draft for draft in (*container_number_context, *logical_drafts)
                }
                normalized_group = tuple(
                    draft
                    for draft in normalize_compact_equipment_locality(
                        raw=raw,
                        drafts=tuple(context.values()),
                        source_target=source_target,
                    )
                    if (draft.char_start, draft.char_end) in original_spans
                )
            normalized_group = normalize_country_code_locality(
                raw=raw,
                drafts=normalize_deterministic_draft_semantics(
                    drafts=normalized_group,
                    source_target=source_target,
                ),
            )
            normalized_by_key: dict[str, list[SpanDraft]] = {}
            for draft in normalized_group:
                normalized_by_key.setdefault(draft.logical_key, []).append(draft)
            try:
                for normalized_drafts in normalized_by_key.values():
                    validate_binding_realizations(
                        raw=raw,
                        drafts=normalized_drafts,
                        source_target=source_target,
                    )
            except Exception as error:
                add(f"binding realization for {logical_key}", error)

        declared_paths = {fact.target_path for fact in output.semantic_only_target_facts}
        proposed_paths = {
            path for draft in proposed for path in (*draft.target_paths, *draft.dependency_paths)
        }
        semantic_conflicts = sorted(declared_paths & proposed_paths)
        if semantic_conflicts:
            add(
                "semantic-only ownership",
                ValueError(
                    "declared semantic-only target paths also have proposed source ownership: "
                    + ", ".join(semantic_conflicts)
                ),
            )

    return ValueError(
        "Compiler candidate failed independent host checks; repair every listed defect in one "
        "patch:\n- " + "\n- ".join(diagnostics)
    )


def _critic_launch_decision(
    *,
    spent: Decimal,
    threshold: Decimal,
    has_unconfirmed_state: bool,
    post_threshold_confirmation_used: bool,
) -> tuple[bool, bool]:
    """Allow exactly one required state confirmation after the threshold.

    A host-valid compiler state has not been semantically audited until the first critic call.
    Likewise, an applied critic patch requires one independent confirmation. The budget guard may
    stop further discovery/repair calls, but it must not discard an otherwise valid state solely
    because compiler construction crossed the soft threshold.
    """

    if spent < threshold:
        return True, post_threshold_confirmation_used
    if has_unconfirmed_state and not post_threshold_confirmation_used:
        return True, True
    return False, post_threshold_confirmation_used


def _jsonl_bytes(rows: Sequence[Mapping[str, Any]]) -> bytes:
    return b"".join(canonical_json_bytes(row) + b"\n" for row in rows)


def _draft_inventory(
    raw: str,
    drafts: Sequence[SpanDraft],
    source_target: Mapping[str, Any],
) -> list[dict[str, Any]]:
    from .host import line_range_for_chars, line_spans

    lines = line_spans(raw)
    line_by_id = {line.line_id: line for line in lines}
    grouped: dict[str, list[SpanDraft]] = {}
    for draft in drafts:
        grouped.setdefault(draft.logical_key, []).append(draft)
    ordered = sorted(grouped.values(), key=lambda rows: min(row.char_start for row in rows))

    def occurrence_summary(row: SpanDraft) -> dict[str, Any]:
        line_start, line_end = line_range_for_chars(lines, row.char_start, row.char_end)
        range_start = line_by_id[line_start].char_start
        range_end = line_by_id[line_end].char_end
        region = raw[range_start:range_end]
        offsets: list[int] = []
        cursor = 0
        while True:
            found = region.find(row.source_text, cursor)
            if found < 0:
                break
            offsets.append(found)
            cursor = found + max(1, len(row.source_text))
        exact_offset = row.char_start - range_start
        if exact_offset not in offsets:
            raise ValueError(
                f"draft source span is not reproducible inside {line_start}-{line_end}: "
                f"{row.logical_key}"
            )
        selected_index = offsets.index(exact_offset)
        return {
            "sourceBindingId": row.draft_id,
            "lineStart": line_start,
            "lineEnd": line_end,
            "sourceText": row.source_text,
            "occurrenceIndex": selected_index,
            "exactMatchCount": len(offsets),
            "exactMatchCandidates": tuple(
                {
                    "occurrenceIndex": index,
                    "rangeRelativeCharStart": offset,
                    "rangeRelativeCharEnd": offset + len(row.source_text),
                    "leftContext": region[:offset],
                    "rightContext": region[offset + len(row.source_text) :],
                    "selected": index == selected_index,
                }
                for index, offset in enumerate(offsets)
            )
            if len(offsets) > 1
            else (),
        }

    output: list[dict[str, Any]] = []
    for rows in ordered:
        first = rows[0]
        output.append(
            {
                "sourceBindingIds": tuple(row.draft_id for row in rows),
                "logicalKey": first.logical_key,
                "renderMode": first.render_mode,
                "valueKind": first.value_kind,
                "groupKind": first.group_kind,
                "groupKey": first.group_key,
                "targetPaths": first.target_paths,
                "targetRelationship": target_path_relationship(source_target, first.target_paths),
                "independentTargetFactComponents": target_fact_components(
                    source_target, first.target_paths
                ),
                "derivation": first.derivation,
                "dependencyPaths": first.dependency_paths,
                "dependencyBindings": first.dependency_bindings,
                "occurrences": tuple(occurrence_summary(row) for row in rows),
            }
        )
    return output


def _host_semantic_only_target_facts(
    *, source_target: Mapping[str, Any], drafts: Sequence[SpanDraft]
) -> tuple[SemanticOnlyTargetFact, ...]:
    patch = source_target.get("documentPatch")
    if not isinstance(patch, Mapping) or "negotiability" not in patch:
        return ()
    path = "documentPatch.negotiability"
    if any(path in draft.target_paths for draft in drafts):
        return ()
    return (
        SemanticOnlyTargetFact.model_validate(
            {
                "target_path": path,
                "source_value": patch["negotiability"],
                "provenance": "host_document_semantic",
                "rationale": (
                    "This task label is a document-level semantic classification. It requires a "
                    "binding only when the OCR prints an unambiguous selected document status; "
                    "conditional boilerplate that discusses multiple alternatives remains literal."
                ),
            }
        ),
    )


def _merge_semantic_only_target_facts(
    *groups: Sequence[SemanticOnlyTargetFact],
) -> tuple[SemanticOnlyTargetFact, ...]:
    by_path: dict[str, SemanticOnlyTargetFact] = {}
    for fact in (fact for group in groups for fact in group):
        prior = by_path.setdefault(fact.target_path, fact)
        if prior.source_value != fact.source_value:
            raise ValueError(
                "semantic-only target path has inconsistent source values: " + fact.target_path
            )
    return tuple(by_path[path] for path in sorted(by_path))


def _effective_semantic_only_target_facts(
    *, facts: Sequence[SemanticOnlyTargetFact], drafts: Sequence[SpanDraft]
) -> tuple[SemanticOnlyTargetFact, ...]:
    owned = {path for draft in drafts for path in draft.target_paths}
    return tuple(fact for fact in facts if fact.target_path not in owned)


def _semantic_only_target_facts(
    *, source_target: Mapping[str, Any], drafts: Sequence[SpanDraft]
) -> list[dict[str, Any]]:
    return [
        {
            "targetPath": fact.target_path,
            "sourceValue": fact.source_value,
            "provenance": fact.provenance,
            "reason": fact.rationale,
        }
        for fact in _host_semantic_only_target_facts(source_target=source_target, drafts=drafts)
    ]


def _addressable_target_paths(source_target: Mapping[str, Any]) -> tuple[str, ...]:
    patch = source_target.get("documentPatch")
    if not isinstance(patch, Mapping):
        raise ValueError("source target lacks documentPatch")
    output: list[str] = []

    def visit(value: Any, path: str) -> None:
        if isinstance(value, Mapping):
            for key, child in value.items():
                child_path = f"{path}.{key}"
                output.append(child_path)
                visit(child, child_path)
        elif isinstance(value, list):
            for index, child in enumerate(value):
                child_path = f"{path}[{index}]"
                output.append(child_path)
                visit(child, child_path)

    visit(patch, "documentPatch")
    return tuple(output)


def _compiler_occurrence_candidates(
    *,
    raw: str,
    source_target: Mapping[str, Any],
    anchors: Sequence[SpanDraft],
    risks: Sequence[Any],
) -> tuple[dict[str, Any], ...]:
    """Build immutable exact-span handles from evidence already available to the host."""

    surfaces: set[str] = set()
    for anchor in anchors:
        surfaces.add(anchor.source_text)
    for risk in risks:
        surfaces.add(risk.source_text)

    def visit_target(value: Any, path: str) -> None:
        if isinstance(value, Mapping):
            for key, child in value.items():
                visit_target(child, f"{path}.{key}")
            return
        if isinstance(value, list):
            for index, child in enumerate(value):
                visit_target(child, f"{path}[{index}]")
            return
        if value is None or isinstance(value, bool):
            return
        surface = str(value)
        if len("".join(character for character in surface if character.isalnum())) >= 4:
            surfaces.add(surface)

    visit_target(source_target.get("documentPatch"), "documentPatch")
    lines = line_spans(raw)
    candidates: set[tuple[int, int, str]] = set()
    for surface in surfaces:
        if not surface:
            continue
        for match in re.finditer(re.escape(surface), raw):
            candidates.add((match.start(), match.end(), surface))

    output: list[dict[str, Any]] = []
    for index, (start, end, surface) in enumerate(sorted(candidates), start=1):
        start_line = next(line for line in lines if line.char_start <= start <= line.char_end)
        end_line = next(line for line in lines if line.char_start <= end - 1 <= line.char_end)
        range_text = raw[start_line.char_start : end_line.char_end]
        relative_start = start - start_line.char_start
        occurrence_index = sum(
            match.start() < relative_start for match in re.finditer(re.escape(surface), range_text)
        )
        output.append(
            {
                "occurrenceId": f"compiler_occurrence_{index:05d}",
                "lineStart": start_line.line_id,
                "lineEnd": end_line.line_id,
                "sourceText": surface,
                "occurrenceIndex": occurrence_index,
            }
        )
    return tuple(output)


def _compiler_payload(
    *,
    document_id: str,
    raw: str,
    source_target: Mapping[str, Any],
    feature: Mapping[str, Any],
    anchors: Sequence[SpanDraft],
    risks: Sequence[Any],
    prior_error: str | None,
    prior_output: CompilerAgentOutput | None,
    prior_drafts: Sequence[SpanDraft] | None,
    agent_contract_protocol: AgentContractProtocol,
) -> dict[str, Any]:
    uncovered_ids = {row.risk_id for row in uncovered_risks(raw, risks, anchors)}
    payload: dict[str, Any] = {
        "documentId": document_id,
        "expectedCarrierName": source_carrier(source_target),
        "sourceLabel": source_target,
        "allowedTargetPaths": _addressable_target_paths(source_target),
        "requiredTargetCoBindings": tuple(
            {
                "relationship": row.relationship,
                "targetPaths": row.target_paths,
            }
            for row in required_target_cobindings(source_target)
        ),
        "documentMetadata": {
            "documentType": feature["document_type"],
            "carrierFamily": feature["carrier_family"],
            "templateProxyId": feature["template_proxy_id"],
            "pages": feature["page_count"],
            "containers": feature["container_count"],
            "cargoGroups": feature["goods_group_count"],
            "packages": feature["package_fact_count"],
            "dangerousGoods": feature["dangerous_goods_count"],
            "temperatureSettings": feature["temperature_setting_count"],
        },
        "anchorBindings": anchor_summary(raw, anchors, source_target),
        "anchorBindingsRequiringSemanticReview": [
            row.draft_id
            for row in anchors
            if target_path_relationship(source_target, row.target_paths)
            == "composite_target_surface"
        ],
        "anchorBindingsWithSharedEqualityConstraint": [
            row.draft_id
            for row in anchors
            if target_path_relationship(source_target, row.target_paths) == "shared_value_equality"
        ],
        "anchorBindingsWithCrossFactEqualityAmbiguity": [
            row.draft_id
            for row in anchors
            if target_path_relationship(source_target, row.target_paths) == "shared_value_equality"
            and len(target_fact_components(source_target, row.target_paths)) > 1
        ],
        "semanticOnlyTargetFacts": _semantic_only_target_facts(
            source_target=source_target, drafts=anchors
        ),
        "riskCandidates": [
            {
                **row.model_dump(mode="json"),
                "currentlyOwnedByAcceptedAnchor": row.risk_id not in uncovered_ids,
            }
            for row in risks
        ],
        "numberedSource": numbered_source(raw),
    }
    if agent_contract_protocol == "reference_compact_v3":
        occurrence_candidates = _compiler_occurrence_candidates(
            raw=raw,
            source_target=source_target,
            anchors=anchors,
            risks=risks,
        )
        columns = (
            "occurrenceId",
            "lineStart",
            "lineEnd",
            "sourceText",
            "occurrenceIndex",
        )
        payload["occurrenceCandidates"] = {
            "columns": columns,
            "rows": tuple(
                tuple(row[column] for column in columns) for row in occurrence_candidates
            ),
        }
    if prior_error is not None:
        payload["requiredRevision"] = prior_error
        payload["repairInstruction"] = (
            "Return a complete replacement output that resolves every listed defect while "
            "preserving every unaffected prior binding exactly."
        )
    if prior_output is not None:
        payload["previousCandidateOutput"] = prior_output.model_dump(mode="json")
    if prior_drafts is not None:
        payload["previousCandidateBindingInventory"] = _draft_inventory(
            raw, prior_drafts, source_target
        )
        payload["previousCandidateMaskedTemplate"] = masked_source(raw, prior_drafts)
    return payload


def _preflight_summary(
    *,
    config: ExtractionConfig,
    manifest: SelectionManifest,
    source_rows: Sequence[Mapping[str, Any]],
    feature_rows: Sequence[Mapping[str, Any]],
    anchor_rows: Sequence[Mapping[str, Any]],
    compiler_prompt: str,
    critic_prompt: str,
) -> dict[str, Any]:
    sources = {str(row["documentId"]): row for row in source_rows}
    features = {str(row["document_id"]): row for row in feature_rows}
    anchors_by_id: dict[str, list[Mapping[str, Any]]] = {}
    for row in anchor_rows:
        anchors_by_id.setdefault(str(row["document_id"]), []).append(row)

    cases: list[dict[str, Any]] = []
    for selected in manifest.rows:
        document_id = selected.document_id
        source = sources[document_id]
        feature = features[document_id]
        raw = cast(str, source["joinedRawText"])
        source_sha = sha256_bytes(raw.encode("utf-8"))
        if source_sha != source["joinedRawTextSha256"] or source_sha != selected.source_sha256:
            raise ValueError(f"preflight source hash differs for {document_id}")
        source_target = cast(Mapping[str, Any], source["target"])
        carrier = source_carrier(source_target)
        anchors = normalize_target_cobindings(
            drafts=anchor_drafts(
                raw=raw,
                document_id=document_id,
                anchors=anchors_by_id.get(document_id, ()),
            ),
            source_target=source_target,
        )
        co_bindings = required_target_cobindings(source_target)
        carrier_anchors = tuple(row for row in anchors if row.render_mode == "carrier_static")
        shared_equality_anchors = tuple(
            row
            for row in anchors
            if target_path_relationship(source_target, row.target_paths) == "shared_value_equality"
        )
        cross_fact_equality_anchors = tuple(
            row
            for row in shared_equality_anchors
            if len(target_fact_components(source_target, row.target_paths)) > 1
        )
        semantic_review_anchors = tuple(
            row
            for row in anchors
            if target_path_relationship(source_target, row.target_paths)
            == "composite_target_surface"
        )
        risks = risk_candidates(raw, ())
        capability_contract(feature, source_target)
        compiler_payload = _compiler_payload(
            document_id=document_id,
            raw=raw,
            source_target=source_target,
            feature=feature,
            anchors=anchors,
            risks=risks,
            prior_error=None,
            prior_output=None,
            prior_drafts=None,
            agent_contract_protocol=config.workflow.agent_contract_protocol,
        )
        compiler_request_bytes = len(
            compiler_prompt.encode("utf-8")
            + json.dumps(compiler_payload, ensure_ascii=False, separators=(",", ":")).encode(
                "utf-8"
            )
        )
        cases.append(
            {
                "ordinal": selected.ordinal,
                "documentId": document_id,
                "carrierName": carrier,
                "sourceLabelCarrierPresent": carrier is not None,
                "sourceLabelCarrierOcrAnchored": bool(carrier_anchors),
                "carrierResolutionRequired": carrier is None or not carrier_anchors,
                "sourceSha256": source_sha,
                "sourceBytes": len(raw.encode("utf-8")),
                "sourceLines": len(raw.splitlines()),
                "acceptedAnchorBindings": len(anchors),
                "sharedEqualityAnchorBindings": len(shared_equality_anchors),
                "crossFactEqualityAnchorBindings": len(cross_fact_equality_anchors),
                "semanticReviewAnchorBindings": len(semantic_review_anchors),
                "carrierAnchorBindings": len(carrier_anchors),
                "requiredTargetCoBindings": len(co_bindings),
                "riskCandidates": len(risks),
                "compilerRequestBytes": compiler_request_bytes,
            }
        )
    if len(cases) != config.workflow.documents:
        raise ValueError("preflight case count differs from configured workflow")
    return {
        "schemaVersion": 1,
        "phase": config.phase,
        "documents": len(cases),
        "allSourceHashesValid": True,
        "allCarrierInputsClassified": True,
        "sourceLabelCarrierPresent": sum(row["sourceLabelCarrierPresent"] for row in cases),
        "sourceLabelCarrierOcrAnchored": sum(row["sourceLabelCarrierOcrAnchored"] for row in cases),
        "carrierResolutionRequired": sum(row["carrierResolutionRequired"] for row in cases),
        "compilerPromptSha256": sha256_bytes(compiler_prompt.encode("utf-8")),
        "criticPromptSha256": sha256_bytes(critic_prompt.encode("utf-8")),
        "totalSourceBytes": sum(row["sourceBytes"] for row in cases),
        "totalAcceptedAnchorBindings": sum(row["acceptedAnchorBindings"] for row in cases),
        "totalSharedEqualityAnchorBindings": sum(
            row["sharedEqualityAnchorBindings"] for row in cases
        ),
        "totalCrossFactEqualityAnchorBindings": sum(
            row["crossFactEqualityAnchorBindings"] for row in cases
        ),
        "totalSemanticReviewAnchorBindings": sum(
            row["semanticReviewAnchorBindings"] for row in cases
        ),
        "totalRiskCandidates": sum(row["riskCandidates"] for row in cases),
        "totalRequiredTargetCoBindings": sum(row["requiredTargetCoBindings"] for row in cases),
        "maxCompilerRequestBytes": max(row["compilerRequestBytes"] for row in cases),
        "providerLaunchAuthorized": config.workflow.provider_launch_authorized,
        "requireAllDocumentsCertified": config.workflow.require_all_documents_certified,
        "cases": cases,
    }


def _corpus_carrier_resolution_report(
    *,
    config: ExtractionConfig,
    source_rows: Sequence[Mapping[str, Any]],
    feature_rows: Sequence[Mapping[str, Any]],
    anchor_rows: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    sources = {str(row["documentId"]): row for row in source_rows}
    features = {str(row["document_id"]): row for row in feature_rows}
    if set(sources) != set(features):
        raise ValueError("source corpus and feature inventory document IDs differ")
    carrier_anchor_ids = {
        str(row["document_id"])
        for row in anchor_rows
        if row.get("patchable")
        and str(row.get("relation_target_path", "")).startswith("documentPatch.parties.carrier")
    }
    rows: list[dict[str, Any]] = []
    for document_id in sorted(sources):
        source = sources[document_id]
        feature = features[document_id]
        carrier = source_carrier(cast(Mapping[str, Any], source["target"]))
        feature_carrier = feature["carrier_name"]
        feature_carrier = None if feature_carrier == "<MISSING>" else feature_carrier
        if carrier != feature_carrier:
            raise ValueError(f"carrier name feature and source label disagree for {document_id}")
        classification = carrier_resolution_classification(source=source)
        if classification == "source_label_present":
            classification = (
                "source_label_ocr_anchored"
                if document_id in carrier_anchor_ids
                else "source_label_requires_agent_evidence"
            )
        rows.append(
            {
                "documentId": document_id,
                "classification": classification,
                "eligibleForCarrierBoundExtraction": classification
                in {"source_label_ocr_anchored", "source_label_requires_agent_evidence"},
                "sourceLabelCarrierName": carrier,
                "carrierFamily": (
                    None if feature["carrier_family"] == "<MISSING>" else feature["carrier_family"]
                ),
                "documentType": feature["document_type"],
                "templateProxyId": feature["template_proxy_id"],
                "pages": feature["page_count"],
            }
        )
    counts = Counter(row["classification"] for row in rows)
    type_counts = {
        classification: dict(
            sorted(
                Counter(
                    row["documentType"] for row in rows if row["classification"] == classification
                ).items()
            )
        )
        for classification in sorted(counts)
    }
    eligible = sum(row["eligibleForCarrierBoundExtraction"] for row in rows)
    return {
        "schemaVersion": 1,
        "sourceCorpusSha256": config.inputs.source_corpus.sha256,
        "documentFeaturesSha256": config.inputs.document_features.sha256,
        "anchorsSha256": config.inputs.anchors.sha256,
        "documents": len(rows),
        "allDocumentIdsClassified": True,
        "eligibleForCarrierBoundExtraction": eligible,
        "requiresExternalCarrierEnrichment": len(rows) - eligible,
        "classificationCounts": dict(sorted(counts.items())),
        "classificationDocumentTypeCounts": type_counts,
        "carrierEligibilityRule": (
            "carrier-bound extraction requires a non-empty pinned source-label carrier; an OCR "
            "master, vessel, agent, or generic carrier phrase cannot establish the legal carrier"
        ),
        "rows": rows,
    }


def audit_corpus_carrier_resolution(config_path: Path) -> dict[str, Any]:
    project_root = project_root_from_config(config_path)
    config = load_config(config_path)
    (
        source_rows,
        feature_rows,
        anchor_rows,
        _current_target_rows,
        _findings_path,
        _invariants_path,
        _manual_path,
        _compiler_prompt,
        _critic_prompt,
    ) = _config_inputs(config, project_root)
    return _corpus_carrier_resolution_report(
        config=config,
        source_rows=source_rows,
        feature_rows=feature_rows,
        anchor_rows=anchor_rows,
    )


def _critic_review_candidates(
    *, raw: str, drafts: Sequence[SpanDraft]
) -> tuple[dict[str, Any], ...]:
    """Expose deterministic high-recall review leads so one critic can audit them together.

    Candidates are not automatic findings. They make relationships that otherwise emerge only
    after serial masking revisions explicit, while the independent critic still decides whether
    each relationship is a defect.
    """

    def normalized(value: str) -> str:
        return "".join(character.casefold() for character in value if character.isalnum())

    def line_id(char_start: int) -> str:
        return f"L{raw.count(chr(10), 0, char_start) + 1:05d}"

    grouped: dict[str, list[SpanDraft]] = {}
    for draft in drafts:
        grouped.setdefault(draft.logical_key, []).append(draft)
    candidates: list[dict[str, Any]] = []

    # Existing bindings can hide another exact rendering of an already mutable value. Surface
    # every uncovered repeat; the critic decides whether its context is data or generic grammar.
    seen_repeat: set[tuple[str, int, int]] = set()
    for logical_key, rows in sorted(grouped.items()):
        for source_text in sorted({row.source_text for row in rows}):
            normalized_source = normalized(source_text)
            typed_short_integer = (
                rows[0].value_kind == "integer"
                and source_text.strip().isdigit()
                and bool(normalized_source)
            )
            if len(normalized_source) < 5 and not typed_short_integer:
                continue
            for match in re.finditer(re.escape(source_text), raw):
                if typed_short_integer and (
                    (match.start() > 0 and raw[match.start() - 1].isdigit())
                    or (match.end() < len(raw) and raw[match.end()].isdigit())
                ):
                    continue
                if any(
                    match.start() < draft.char_end and draft.char_start < match.end()
                    for draft in drafts
                ):
                    continue
                key = (source_text, match.start(), match.end())
                if key in seen_repeat:
                    continue
                seen_repeat.add(key)
                candidates.append(
                    {
                        "kind": "unowned_exact_repeat",
                        "logicalKeys": (logical_key,),
                        "lineIds": (line_id(match.start()),),
                        "sourceTexts": (source_text,),
                        "context": raw[max(0, match.start() - 48) : match.end() + 48],
                    }
                )

    # Several bindings in one semantic scope may be format variants of one fact. Present exact
    # equality and normalized containment without guessing whether they should be joined.
    scope_groups: dict[tuple[str, str, str], list[str]] = {}
    for logical_key, rows in grouped.items():
        first = rows[0]
        if first.render_mode in {"carrier_static", "literal_static"}:
            continue
        scope_groups.setdefault((first.group_kind, first.group_key, first.value_kind), []).append(
            logical_key
        )
    for scope, logical_keys in sorted(scope_groups.items()):
        for left_index, left_key in enumerate(sorted(logical_keys)):
            left_values = {row.source_text for row in grouped[left_key]}
            for right_key in sorted(logical_keys)[left_index + 1 :]:
                right_values = {row.source_text for row in grouped[right_key]}
                relationships = {
                    (left, right)
                    for left in left_values
                    for right in right_values
                    if min(len(normalized(left)), len(normalized(right))) >= 5
                    and (
                        normalized(left) in normalized(right)
                        or normalized(right) in normalized(left)
                    )
                }
                if not relationships:
                    continue
                candidates.append(
                    {
                        "kind": "same_scope_surface_relationship",
                        "logicalKeys": (left_key, right_key),
                        "lineIds": tuple(
                            sorted(
                                {
                                    line_id(row.char_start)
                                    for key in (left_key, right_key)
                                    for row in grouped[key]
                                }
                            )
                        ),
                        "sourceTexts": tuple(
                            sorted({value for pair in relationships for value in pair})
                        ),
                        "scope": {
                            "groupKind": scope[0],
                            "groupKey": scope[1],
                            "valueKind": scope[2],
                        },
                    }
                )

    for logical_key, rows in sorted(grouped.items()):
        surfaces = sorted({row.source_text for row in rows})
        normalized_surfaces = {normalized(value) for value in surfaces}
        if len(surfaces) > 1 and len(normalized_surfaces) == 1:
            candidates.append(
                {
                    "kind": "occurrence_boundary_or_format_variation",
                    "logicalKeys": (logical_key,),
                    "lineIds": tuple(sorted({line_id(row.char_start) for row in rows})),
                    "sourceTexts": tuple(surfaces),
                }
            )
        first = rows[0]
        if first.render_mode == "deterministic_derived":
            candidates.append(
                {
                    "kind": "derivation_contract_review",
                    "logicalKeys": (logical_key,),
                    "lineIds": tuple(sorted({line_id(row.char_start) for row in rows})),
                    "sourceTexts": tuple(surfaces),
                    "derivation": first.derivation,
                    "targetPaths": first.target_paths,
                    "dependencyPaths": first.dependency_paths,
                    "dependencyBindings": first.dependency_bindings,
                }
            )

    candidates.sort(
        key=lambda row: (
            row["kind"],
            row["logicalKeys"],
            row["lineIds"],
            row["sourceTexts"],
        )
    )
    return tuple(
        {"candidateId": f"review_candidate_{index:04d}", **row}
        for index, row in enumerate(candidates, start=1)
    )


_MASKED_BINDING_MARKER = re.compile(r"⟦[^⟧]+⟧")
_NUMBERED_MASKED_LINE = re.compile(r"^(?P<line_id>L[0-9]{5}) \| (?P<content>.*)$")
_PAGE_MARKER_LITERAL = re.compile(r"^--- PAGE [0-9]+ ---$")


def _literal_review_line_ids(masked_template: str) -> tuple[str, ...]:
    """Enumerate every source line that retains potentially meaningful literal text."""

    output: list[str] = []
    for row in masked_template.splitlines():
        match = _NUMBERED_MASKED_LINE.fullmatch(row)
        if match is None:
            raise ValueError("masked template contains a line outside the numbered-line contract")
        literal = _MASKED_BINDING_MARKER.sub("", match.group("content")).strip()
        if (
            any(character.isalnum() for character in literal)
            and _PAGE_MARKER_LITERAL.fullmatch(literal) is None
        ):
            output.append(match.group("line_id"))
    if not output:
        raise ValueError("masked template contains no literal lines for independent review")
    return tuple(output)


def _critic_payload(
    *,
    document_id: str,
    raw: str,
    source_target: Mapping[str, Any],
    feature: Mapping[str, Any],
    drafts: Sequence[SpanDraft],
    risks: Sequence[Any],
    prior_error: str | None,
    semantic_only_target_facts: Sequence[SemanticOnlyTargetFact] = (),
    review_history: Sequence[AgentStageArtifact] = (),
    agent_contract_protocol: AgentContractProtocol = "compact_discriminated_v2",
) -> dict[str, Any]:
    inventory = _draft_inventory(raw, drafts, source_target)
    effective_semantic_only = _effective_semantic_only_target_facts(
        facts=_merge_semantic_only_target_facts(
            _host_semantic_only_target_facts(source_target=source_target, drafts=drafts),
            semantic_only_target_facts,
        ),
        drafts=drafts,
    )
    current_masked_source = masked_source(raw, drafts)
    payload: dict[str, Any] = {
        "documentId": document_id,
        "expectedCarrierName": source_carrier(source_target),
        "sourceLabel": source_target,
        "allowedTargetPaths": _addressable_target_paths(source_target),
        "requiredTargetCoBindings": tuple(
            {
                "relationship": row.relationship,
                "targetPaths": row.target_paths,
            }
            for row in required_target_cobindings(source_target)
        ),
        "allowedRemovalLogicalKeys": tuple(row["logicalKey"] for row in inventory),
        "documentMetadata": {
            "documentType": feature["document_type"],
            "carrierFamily": feature["carrier_family"],
            "templateProxyId": feature["template_proxy_id"],
        },
        "bindingInventory": inventory,
        "semanticOnlyTargetFacts": [
            {
                "targetPath": fact.target_path,
                "sourceValue": fact.source_value,
                "provenance": fact.provenance,
                "reason": fact.rationale,
            }
            for fact in effective_semantic_only
        ],
        "priorReviewLedger": _critic_review_history(review_history),
        "literalLineReviewIds": _literal_review_line_ids(current_masked_source),
        "remainingRiskCandidates": [
            row.model_dump(mode="json") for row in uncovered_risks(raw, risks, drafts)
        ],
        "reviewCandidates": _critic_review_candidates(raw=raw, drafts=drafts),
        "annotatedSource": annotated_source(raw, drafts),
        "maskedTemplate": current_masked_source,
    }
    if agent_contract_protocol != "legacy_v1":
        occurrence_candidates = _compiler_occurrence_candidates(
            raw=raw,
            source_target=source_target,
            anchors=drafts,
            risks=risks,
        )
        columns = (
            "occurrenceId",
            "lineStart",
            "lineEnd",
            "sourceText",
            "occurrenceIndex",
        )
        payload["occurrenceCandidates"] = {
            "columns": columns,
            "rows": tuple(
                tuple(row[column] for column in columns) for row in occurrence_candidates
            ),
        }
    if prior_error is not None:
        payload["priorHostRejection"] = prior_error
        payload["repairInstruction"] = (
            "Re-audit the complete masked template and return corrected exact additions."
        )
    return payload


def _critic_review_history(
    stages: Sequence[AgentStageArtifact],
) -> dict[str, Any]:
    """Summarize audit progress without carrying stale semantic claims forward.

    Every critic request is an independent audit of the current host materialization.  A
    successful revision has already changed that materialization, while a host-rejected output
    was never established as semantically correct.  Replaying either output's findings therefore
    anchors the next critic to a superseded or untrusted claim.  The immutable stage artifacts
    retain the complete history; this payload contains progress metadata only.
    """

    status_counts: Counter[str] = Counter()
    successful_verdict_counts: Counter[str] = Counter()
    applied_revision_count = 0
    latest_stage: dict[str, Any] | None = None
    for stage in stages:
        status_counts[stage.status] += 1
        verdict: str | None = None
        if stage.output is not None:
            review = CriticAgentOutput.model_validate_json(
                json.dumps(stage.output, ensure_ascii=False, separators=(",", ":"))
            )
            verdict = review.verdict
            if stage.status == "success":
                successful_verdict_counts[review.verdict] += 1
                if review.verdict == "revise":
                    applied_revision_count += 1
        latest_stage = {
            "passNumber": stage.pass_number,
            "hostStatus": stage.status,
            "providerVerdict": verdict,
            "stateChanged": stage.status == "success" and verdict == "revise",
        }
    return {
        "completedStages": len(stages),
        "stageStatusCounts": dict(sorted(status_counts.items())),
        "successfulVerdictCounts": dict(sorted(successful_verdict_counts.items())),
        "appliedRevisionCount": applied_revision_count,
        "currentStateRevision": applied_revision_count,
        "latestStage": latest_stage,
        "carriedFindings": (),
        "interpretation": (
            "No prior finding is an active obligation. The current binding inventory, semantic-"
            "only facts, remaining risks, and annotated source are the complete authoritative "
            "state for a fresh audit. Successful revisions are already materialized; host-"
            "rejected findings were not validated. The separate priorHostRejection, when present, "
            "is mechanical feedback about the immediately preceding attempt only."
        ),
    }


def _critic_stagnation_reason(stages: Sequence[AgentStageArtifact]) -> str | None:
    if len(stages) < 2:
        return None
    last_two = stages[-2:]
    if all(stage.status == "host_rejected" for stage in last_two):
        errors = {stage.error_message for stage in last_two}
        if len(errors) == 1:
            return (
                "critic made no progress after two consecutive identical host rejections: "
                + str(last_two[-1].error_message)
            )
    return None


def _critic_feedback(output: CriticAgentOutput) -> str:
    return "Independent critic required a local binding patch:\n" + "\n".join(
        f"- {finding.finding_kind} at {','.join(finding.line_ids)}: {finding.explanation}"
        for finding in output.findings
    )


def _state_checkpoint(
    *,
    document_id: str,
    raw: str,
    source_target: Mapping[str, Any],
    drafts: Sequence[SpanDraft],
    assessment: CarrierAssessment,
    semantic_only_target_facts: Sequence[SemanticOnlyTargetFact],
    critic_outputs: Sequence[CriticAgentOutput],
) -> ExtractionStateCheckpoint:
    return ExtractionStateCheckpoint.model_validate(
        {
            "schema_version": 1,
            "document_id": document_id,
            "source_sha256": sha256_bytes(raw.encode("utf-8")),
            "source_label_sha256": sha256_bytes(canonical_json_bytes(source_target)),
            "drafts": tuple(
                DraftCheckpointRow.model_validate(
                    {
                        "draft_id": draft.draft_id,
                        "logical_key": draft.logical_key,
                        "render_mode": draft.render_mode,
                        "value_kind": draft.value_kind,
                        "group_kind": draft.group_kind,
                        "group_key": draft.group_key,
                        "target_paths": draft.target_paths,
                        "derivation": draft.derivation,
                        "dependency_paths": draft.dependency_paths,
                        "dependency_bindings": draft.dependency_bindings,
                        "char_start": draft.char_start,
                        "char_end": draft.char_end,
                        "source_text": draft.source_text,
                        "evidence_origin": draft.evidence_origin,
                        "render_policy": draft.render_policy,
                        "rationale": draft.rationale,
                    }
                )
                for draft in drafts
            ),
            "carrier_assessment": assessment,
            "semantic_only_target_facts": tuple(semantic_only_target_facts),
            "applied_critic_revisions": tuple(
                review for review in critic_outputs if review.verdict == "revise"
            ),
        }
    )


def _restore_state_checkpoint(
    *,
    checkpoint_payload: bytes,
    document_id: str,
    raw: str,
    source_target: Mapping[str, Any],
) -> tuple[
    tuple[SpanDraft, ...],
    CarrierAssessment,
    tuple[SemanticOnlyTargetFact, ...],
    list[CriticAgentOutput],
]:
    checkpoint = ExtractionStateCheckpoint.model_validate_json(checkpoint_payload, strict=True)
    if checkpoint.document_id != document_id:
        raise ValueError("state checkpoint document identity differs")
    if checkpoint.source_sha256 != sha256_bytes(raw.encode("utf-8")):
        raise ValueError("state checkpoint source hash differs")
    if checkpoint.source_label_sha256 != sha256_bytes(canonical_json_bytes(source_target)):
        raise ValueError("state checkpoint source-label hash differs")
    raw_drafts = tuple(SpanDraft(**row.model_dump(mode="python")) for row in checkpoint.drafts)
    for draft in raw_drafts:
        if raw[draft.char_start : draft.char_end] != draft.source_text:
            raise ValueError(
                "state checkpoint draft no longer matches pinned OCR: " + draft.draft_id
            )
    drafts = normalize_country_code_locality(
        raw=raw,
        drafts=normalize_deterministic_draft_semantics(
            drafts=normalize_compact_equipment_locality(
                raw=raw,
                drafts=normalize_target_cobindings(
                    drafts=raw_drafts,
                    source_target=source_target,
                ),
                source_target=source_target,
            ),
            source_target=source_target,
        ),
    )
    validate_target_binding_relationships(drafts=drafts, source_target=source_target)
    validate_binding_realizations(raw=raw, drafts=drafts, source_target=source_target)
    validate_carrier_assessment(
        assessment=checkpoint.carrier_assessment,
        expected=source_carrier(source_target),
        raw=raw,
        anchor_drafts_value=drafts,
    )
    semantic_only = _effective_semantic_only_target_facts(
        facts=checkpoint.semantic_only_target_facts,
        drafts=drafts,
    )
    return (
        drafts,
        checkpoint.carrier_assessment,
        semantic_only,
        list(checkpoint.applied_critic_revisions),
    )


def _validated_compiler_state(
    *,
    output: CompilerAgentOutput,
    raw: str,
    source_target: Mapping[str, Any],
    anchors: Sequence[SpanDraft],
) -> tuple[tuple[SpanDraft, ...], Any, tuple[SemanticOnlyTargetFact, ...]]:
    if not output.all_shipment_dependent_surfaces_accounted_for or output.unresolved:
        raise ValueError("compiler retained unresolved items: " + "; ".join(output.unresolved))
    validate_agent_proposal_paths(proposals=output.bindings, source_target=source_target)
    declared_semantic_only = materialize_semantic_only_target_facts(
        proposals=output.semantic_only_target_facts,
        source_target=source_target,
        provenance="compiler_audited_unprinted",
    )
    proposed = resolve_agent_proposals(raw=raw, proposals=output.bindings)
    drafts = normalize_country_code_locality(
        raw=raw,
        drafts=normalize_deterministic_draft_semantics(
            drafts=normalize_compact_equipment_locality(
                raw=raw,
                drafts=normalize_target_cobindings(
                    drafts=apply_anchor_overrides(
                        raw=raw,
                        source_target=source_target,
                        anchors=anchors,
                        proposed=proposed,
                        overrides=output.anchor_overrides,
                        semantic_only_target_paths=tuple(
                            fact.target_path for fact in declared_semantic_only
                        ),
                    ),
                    source_target=source_target,
                ),
                source_target=source_target,
            ),
            source_target=source_target,
        ),
    )
    validate_target_binding_relationships(drafts=drafts, source_target=source_target)
    validate_binding_realizations(raw=raw, drafts=drafts, source_target=source_target)
    declared_conflicts = sorted(
        {
            fact.target_path
            for fact in declared_semantic_only
            if any(fact.target_path in draft.target_paths for draft in drafts)
        }
    )
    if declared_conflicts:
        raise ValueError(
            "compiler semantic-only target facts still have source bindings: "
            + ", ".join(declared_conflicts)
        )
    expected_carrier = source_carrier(source_target)
    assessment = (
        output.carrier.model_copy(update={"aliases": ()})
        if expected_carrier is not None
        else output.carrier
    )
    validate_carrier_assessment(
        assessment=assessment,
        expected=expected_carrier,
        raw=raw,
        anchor_drafts_value=drafts,
    )
    semantic_only = _merge_semantic_only_target_facts(
        _host_semantic_only_target_facts(source_target=source_target, drafts=drafts),
        declared_semantic_only,
    )
    return drafts, assessment, semantic_only


def _replay_checkpoint_case(
    *,
    result: ExtractionCaseResult,
    raw: str,
    source_target: Mapping[str, Any],
    anchors: Sequence[SpanDraft],
    risks: Sequence[Any],
) -> tuple[
    tuple[SpanDraft, ...] | None,
    Any,
    tuple[SemanticOnlyTargetFact, ...],
    list[CriticAgentOutput],
    str | None,
]:
    replay_warnings: list[str] = []
    skipped_noop_passes: list[int] = []
    if tuple(result.risk_candidates) != tuple(risks):
        replay_warnings.append(
            "Current host deterministically changed the risk inventory; every current risk will "
            "be rechecked before fresh certification"
        )

    def warning(extra: str | None = None) -> str | None:
        noop_warning = (
            "Current host skipped functionally redundant prior critic passes: "
            + ",".join(str(value) for value in skipped_noop_passes)
            if skipped_noop_passes
            else None
        )
        parts = [
            *replay_warnings,
            *([noop_warning] if noop_warning is not None else []),
            *([extra] if extra is not None else []),
        ]
        return "; ".join(parts) if parts else None

    successful_compilers = tuple(
        stage for stage in result.compiler_stages if stage.status == "success"
    )
    if len(successful_compilers) > 1:
        raise ValueError("resume case contains multiple successful compiler checkpoints")
    if not successful_compilers:
        latest_candidate = next(
            (stage for stage in reversed(result.compiler_stages) if stage.output is not None),
            None,
        )
        if latest_candidate is None:
            return None, None, (), [], warning()
        candidate = CompilerAgentOutput.model_validate_json(
            json.dumps(
                latest_candidate.output,
                ensure_ascii=False,
                separators=(",", ":"),
            )
        )
        try:
            drafts, assessment, semantic_only = _validated_compiler_state(
                output=candidate,
                raw=raw,
                source_target=source_target,
                anchors=anchors,
            )
        except Exception as error:
            return (
                None,
                None,
                (),
                [],
                warning(
                    "Current host replay of the latest compiler candidate rejected it: "
                    + str(error)
                ),
            )
        return (
            drafts,
            assessment,
            semantic_only,
            [],
            warning(
                "Current host accepted the latest previously host-rejected compiler candidate "
                f"at pass {latest_candidate.pass_number} after deterministic contract "
                "correction; a fresh critic pass remains mandatory"
            ),
        )
    stage = successful_compilers[0]
    if stage.output is None:
        raise ValueError("successful resume compiler stage has no output")
    output = CompilerAgentOutput.model_validate_json(
        json.dumps(stage.output, ensure_ascii=False, separators=(",", ":"))
    )
    try:
        drafts, assessment, semantic_only = _validated_compiler_state(
            output=output,
            raw=raw,
            source_target=source_target,
            anchors=anchors,
        )
    except Exception as error:
        return (
            None,
            None,
            (),
            [],
            warning(
                "Current host rejected the prior successful compiler checkpoint: " + str(error)
            ),
        )
    critic_outputs: list[CriticAgentOutput] = []
    candidate_reviews: list[tuple[AgentStageArtifact, CriticAgentOutput]] = []
    for critic_stage in result.critic_stages:
        if critic_stage.output is None:
            if critic_stage.status == "success":
                raise ValueError("successful resume critic stage has no output")
            continue
        review = CriticAgentOutput.model_validate_json(
            json.dumps(critic_stage.output, ensure_ascii=False, separators=(",", ":"))
        )
        candidate_reviews.append((critic_stage, review))
    final_stage = result.critic_stages[-1] if result.critic_stages else None
    final_review = (
        CriticAgentOutput.model_validate_json(
            json.dumps(final_stage.output, ensure_ascii=False, separators=(",", ":"))
        )
        if final_stage is not None and final_stage.output is not None
        else None
    )
    final_is_successful_pass = bool(
        final_stage is not None
        and final_stage.status == "success"
        and final_review is not None
        and final_review.verdict == "pass"
    )
    if result.status == "certified" and not final_is_successful_pass:
        raise ValueError("certified resume case lacks a final successful critic pass")
    if result.status == "rejected" and final_is_successful_pass:
        raise ValueError("rejected resume case ends in a successful critic pass")
    newly_accepted_prior_rejections: list[int] = []
    for critic_stage, review in candidate_reviews:
        if review.verdict == "pass":
            # A prior pass is evidence only for the prior prompt and host contract. When this
            # function is used, the current run must obtain its own independent final pass.
            continue
        drafts_before_review = drafts
        try:
            review_semantic_only = materialize_semantic_only_target_facts(
                proposals=review.semantic_only_target_facts,
                source_target=source_target,
                provenance="critic_audited_unprinted",
            )
            drafts = normalize_country_code_locality(
                raw=raw,
                drafts=normalize_deterministic_draft_semantics(
                    drafts=normalize_compact_equipment_locality(
                        raw=raw,
                        drafts=apply_critic_patch(
                            raw=raw,
                            drafts=drafts,
                            findings=review.findings,
                            remove_inventory_binding_ids=review.remove_inventory_binding_ids,
                            additional_bindings=review.additional_bindings,
                            occurrence_removals=review.occurrence_removals,
                            semantic_only_target_paths=tuple(
                                fact.target_path for fact in review_semantic_only
                            ),
                            source_target=source_target,
                        ),
                        source_target=source_target,
                    ),
                    source_target=source_target,
                ),
            )
            validate_carrier_assessment(
                assessment=assessment,
                expected=source_carrier(source_target),
                raw=raw,
                anchor_drafts_value=drafts,
            )
            validate_binding_realizations(raw=raw, drafts=drafts, source_target=source_target)
        except Exception as error:
            if str(error).startswith(
                "critic transaction is a functional no-op after host canonicalization"
            ):
                skipped_noop_passes.append(critic_stage.pass_number)
                continue
            drafts = drafts_before_review
            replay_warnings.append(
                "Current host skipped incompatible prior critic pass "
                f"{critic_stage.pass_number}: {error}"
            )
            continue
        semantic_only = _effective_semantic_only_target_facts(
            facts=_merge_semantic_only_target_facts(
                semantic_only,
                review_semantic_only,
            ),
            drafts=drafts,
        )
        if critic_stage.status == "host_rejected":
            newly_accepted_prior_rejections.append(critic_stage.pass_number)
        critic_outputs.append(review)
    replay_note = (
        "Current host accepted formerly host-rejected critic transactions at passes "
        + ",".join(str(value) for value in newly_accepted_prior_rejections)
        if newly_accepted_prior_rejections
        else None
    )
    return drafts, assessment, semantic_only, critic_outputs, warning(replay_note)


def _restore_or_replay_resume_state(
    *,
    checkpoint_payload: bytes | None,
    result: ExtractionCaseResult,
    document_id: str,
    raw: str,
    source_target: Mapping[str, Any],
    anchors: Sequence[SpanDraft],
    risks: Sequence[Any],
    resume_contract_match: bool,
) -> tuple[
    tuple[SpanDraft, ...] | None,
    CarrierAssessment | None,
    tuple[SemanticOnlyTargetFact, ...],
    list[CriticAgentOutput],
    str | None,
]:
    checkpoint_error: Exception | None = None
    checkpoint_state: (
        tuple[
            tuple[SpanDraft, ...],
            CarrierAssessment,
            tuple[SemanticOnlyTargetFact, ...],
            list[CriticAgentOutput],
        ]
        | None
    ) = None
    if checkpoint_payload is not None:
        try:
            checkpoint_state = _restore_state_checkpoint(
                checkpoint_payload=checkpoint_payload,
                document_id=document_id,
                raw=raw,
                source_target=source_target,
            )
        except Exception as error:
            checkpoint_error = error
        if checkpoint_state is not None and resume_contract_match:
            drafts, assessment, semantic_only, critic_outputs = checkpoint_state
            return drafts, assessment, semantic_only, critic_outputs, None

    drafts, assessment, semantic_only, critic_outputs, replay_warning = _replay_checkpoint_case(
        result=result,
        raw=raw,
        source_target=source_target,
        anchors=anchors,
        risks=risks,
    )
    warnings = []
    if checkpoint_error is not None:
        warnings.append(
            "Current host rejected the prior exact state checkpoint and replayed its accepted "
            f"transactions instead: {checkpoint_error}"
        )
    if replay_warning is not None:
        warnings.append(replay_warning)
    if drafts is None and checkpoint_state is not None:
        drafts, assessment, semantic_only, critic_outputs = checkpoint_state
        warnings.append(
            "Current-host transaction replay did not recover a compiler state; retained the "
            "separately revalidated exact prior checkpoint"
        )
    elif checkpoint_state is not None:
        warnings.append(
            "Replayed prior transactions under the current host instead of trusting the older "
            "materialized checkpoint"
        )
    return (
        drafts,
        assessment,
        semantic_only,
        critic_outputs,
        "; ".join(warnings) if warnings else None,
    )


async def _extract_case(
    *,
    ordinal: int,
    document_id: str,
    source: Mapping[str, Any],
    feature: Mapping[str, Any],
    anchor_rows: Sequence[Mapping[str, Any]],
    runtime: AgentRuntime,
    compiler_prompt: str,
    critic_prompt: str,
    config: ExtractionConfig,
    staged: StagedArtifactRun,
    document_limiter: asyncio.Semaphore,
    resume_run_root: Path | None = None,
    resume_result: ExtractionCaseResult | None = None,
    resume_contract_match: bool = False,
) -> tuple[ExtractionCaseResult, CertifiedSemanticTemplate | None, str]:
    relative_root = f"cases/{document_id}"
    existing_result_path = staged.stage_root / relative_root / "result.json"
    if existing_result_path.is_file():
        result = ExtractionCaseResult.model_validate_json(
            read_regular_file_bytes(existing_result_path), strict=True
        )
        template: CertifiedSemanticTemplate | None = None
        masked = ""
        if result.status == "certified":
            template = CertifiedSemanticTemplate.model_validate_json(
                read_regular_file_bytes(staged.stage_root / relative_root / "template.json"),
                strict=True,
            )
            masked = (staged.stage_root / relative_root / "masked-template.txt").read_text(
                encoding="utf-8"
            )
        print(
            f"[{ordinal:03d}/{config.workflow.documents:03d}] resumed "
            f"{document_id} {result.status}",
            flush=True,
        )
        return result, template, masked

    async with document_limiter:
        started = time.perf_counter()
        raw = cast(str, source["joinedRawText"])
        if sha256_bytes(raw.encode("utf-8")) != source["joinedRawTextSha256"]:
            raise ValueError(f"source raw-text hash differs for {document_id}")
        source_target = cast(Mapping[str, Any], source["target"])
        document_anchors = [row for row in anchor_rows if row["document_id"] == document_id]
        anchors = normalize_target_cobindings(
            drafts=anchor_drafts(raw=raw, document_id=document_id, anchors=document_anchors),
            source_target=source_target,
        )
        risks = risk_candidates(raw, ())
        resumed_from_run = resume_run_root.name if resume_run_root is not None else None
        prior_elapsed_seconds = resume_result.elapsed_seconds if resume_result is not None else 0.0
        if resume_result is not None and resume_result.document_id != document_id:
            raise ValueError("resume case document identity differs")
        certified_resume_mode = (
            "exact_contract_reuse"
            if resume_contract_match
            else (
                "current_host_recertification"
                if config.workflow.certified_resume_policy == "current_host_recertify"
                else None
            )
        )
        if (
            resume_result is not None
            and resume_result.status == "certified"
            and certified_resume_mode is not None
        ):
            if resume_run_root is None:
                raise ValueError("certified resume case lacks its run root")
            prior_case_root = resume_run_root / relative_root
            prior_source = read_regular_file_bytes(prior_case_root / "source.txt")
            if prior_source != raw.encode("utf-8"):
                raise ValueError("certified resume source differs from the current pinned source")
            prior_label = json.loads(read_regular_file_bytes(prior_case_root / "source-label.json"))
            if prior_label != source_target:
                raise ValueError("certified resume label differs from the current pinned label")
            prior_template_payload = read_regular_file_bytes(prior_case_root / "template.json")
            prior_template = CertifiedSemanticTemplate.model_validate_json(
                prior_template_payload, strict=True
            )
            if (
                prior_template.document_id != document_id
                or prior_template.source_sha256 != source["joinedRawTextSha256"]
                or resume_result.template_sha256 != sha256_bytes(prior_template_payload)
            ):
                raise ValueError("certified resume template identity or hash differs")
            prior_checkpoint_payload = read_regular_file_bytes(
                prior_case_root / "state-checkpoint.json"
            )
            current_checkpoint: ExtractionStateCheckpoint | None = None
            if certified_resume_mode == "exact_contract_reuse":
                if tuple(resume_result.risk_candidates) != tuple(risks):
                    raise ValueError(
                        "certified resume risk inventory differs under current host code"
                    )
                _restore_state_checkpoint(
                    checkpoint_payload=prior_checkpoint_payload,
                    document_id=document_id,
                    raw=raw,
                    source_target=source_target,
                )
                template = prior_template
                template_payload = prior_template_payload
                masked_payload = read_regular_file_bytes(prior_case_root / "masked-template.txt")
                catalog_payload = read_regular_file_bytes(prior_case_root / "catalog-row.json")
                checkpoint_payload = prior_checkpoint_payload
                replay_note = None
            else:
                drafts, assessment, semantic_only, applied_revisions = _restore_state_checkpoint(
                    checkpoint_payload=prior_checkpoint_payload,
                    document_id=document_id,
                    raw=raw,
                    source_target=source_target,
                )
                final_stage = (
                    resume_result.critic_stages[-1] if resume_result.critic_stages else None
                )
                if (
                    final_stage is None
                    or final_stage.status != "success"
                    or final_stage.output is None
                ):
                    raise ValueError(
                        "current-host recertification requires a final successful critic receipt"
                    )
                final_review = CriticAgentOutput.model_validate_json(
                    json.dumps(
                        final_stage.output,
                        ensure_ascii=False,
                        separators=(",", ":"),
                    )
                )
                if final_review.verdict != "pass":
                    raise ValueError("current-host recertification requires a final critic pass")
                template = certify_template(
                    raw=raw,
                    document_id=document_id,
                    feature=feature,
                    source_target=source_target,
                    assessment=assessment,
                    drafts=drafts,
                    risks=risks,
                    critic_outputs=(*applied_revisions, final_review),
                    semantic_only_target_facts=semantic_only,
                )
                template_payload = json_artifact_bytes(template.model_dump(mode="json"))
                masked_payload = masked_source(raw, drafts).encode("utf-8")
                catalog_payload = json_artifact_bytes(template_summary(template))
                current_checkpoint = _state_checkpoint(
                    document_id=document_id,
                    raw=raw,
                    source_target=source_target,
                    drafts=drafts,
                    assessment=assessment,
                    semantic_only_target_facts=semantic_only,
                    critic_outputs=applied_revisions,
                )
                checkpoint_payload = json_artifact_bytes(current_checkpoint.model_dump(mode="json"))
                replay_note = (
                    "Recompiled and recertified the exact prior state and final critic receipt "
                    "under the current host contract without a provider request; the prior "
                    "compiler and critic prompts remain recorded in the resumed stage lineage"
                )
            reused_result = resume_result.model_copy(
                update={
                    "template_sha256": sha256_bytes(template_payload),
                    "risk_candidates": tuple(risks),
                    "elapsed_seconds": prior_elapsed_seconds + time.perf_counter() - started,
                    "resumed_from_run": resumed_from_run,
                    "resumed_prior_elapsed_seconds": prior_elapsed_seconds,
                    "resumed_prior_compiler_stages": len(resume_result.compiler_stages),
                    "resumed_prior_critic_stages": len(resume_result.critic_stages),
                    "resumed_prior_estimated_cost_usd": sum(
                        (
                            stage.usage.estimatedCostUsd
                            for stage in (
                                *resume_result.compiler_stages,
                                *resume_result.critic_stages,
                            )
                        ),
                        Decimal(0),
                    ),
                    "resume_contract_match": resume_contract_match,
                    "resume_replay_warning": replay_note,
                    "resume_mode": certified_resume_mode,
                }
            )
            staged.publish_bytes(f"{relative_root}/source.txt", prior_source)
            staged.publish_json(f"{relative_root}/source-label.json", source_target)
            staged.publish_bytes(f"{relative_root}/masked-template.txt", masked_payload)
            staged.publish_bytes(f"{relative_root}/template.json", template_payload)
            staged.publish_bytes(f"{relative_root}/catalog-row.json", catalog_payload)
            staged.publish_bytes(f"{relative_root}/state-checkpoint.json", checkpoint_payload)
            staged.publish_json(
                f"{relative_root}/result.json", reused_result.model_dump(mode="json")
            )
            print(
                f"[{ordinal:03d}/{config.workflow.documents:03d}] "
                f"{certified_resume_mode} "
                f"{document_id} from {resumed_from_run}",
                flush=True,
            )
            return reused_result, template, masked_payload.decode("utf-8")

        compiler_stages: list[AgentStageArtifact] = (
            list(resume_result.compiler_stages) if resume_result is not None else []
        )
        critic_stages: list[AgentStageArtifact] = (
            list(resume_result.critic_stages) if resume_result is not None else []
        )
        resumed_prior_compiler_stages = len(compiler_stages)
        resumed_prior_critic_stages = len(critic_stages)
        resumed_prior_estimated_cost = sum(
            (stage.usage.estimatedCostUsd for stage in (*compiler_stages, *critic_stages)),
            Decimal(0),
        )
        if resume_result is not None:
            if resume_run_root is None:
                raise ValueError("resume case lacks its run root")
            checkpoint_path = resume_run_root / relative_root / "state-checkpoint.json"
            (
                drafts,
                assessment,
                semantic_only_target_facts,
                critic_outputs,
                resume_replay_warning,
            ) = _restore_or_replay_resume_state(
                checkpoint_payload=(
                    read_regular_file_bytes(checkpoint_path) if checkpoint_path.is_file() else None
                ),
                result=resume_result,
                document_id=document_id,
                raw=raw,
                source_target=source_target,
                anchors=anchors,
                risks=risks,
                resume_contract_match=resume_contract_match,
            )
        else:
            drafts, assessment = None, None
            semantic_only_target_facts = ()
            critic_outputs = []
            resume_replay_warning = None
        reasons: list[str] = []
        template: CertifiedSemanticTemplate | None = None
        prior_compiler_output: CompilerAgentOutput | None = next(
            (
                CompilerAgentOutput.model_validate_json(
                    json.dumps(stage.output, ensure_ascii=False, separators=(",", ":"))
                )
                for stage in reversed(compiler_stages)
                if stage.output is not None
            ),
            None,
        )
        prior_compiler_drafts: tuple[SpanDraft, ...] | None = drafts
        revision_context = _latest_compiler_revision_context(
            drafts=drafts,
            prior_output=prior_compiler_output,
            compiler_stages=compiler_stages,
            resume_replay_warning=resume_replay_warning,
        )
        critic_revision_context: str | None = (
            f"Host rejected critic patch: {critic_stages[-1].error_message}"
            if resume_contract_match
            and critic_stages
            and critic_stages[-1].status == "host_rejected"
            else None
        )
        critic_pass = max((stage.pass_number for stage in critic_stages), default=0)
        critic_attempt_calls = len(critic_stages) - resumed_prior_critic_stages
        post_threshold_confirmation_used = False

        def spent() -> Decimal:
            return sum(
                (stage.usage.estimatedCostUsd for stage in (*compiler_stages, *critic_stages)),
                Decimal(0),
            )

        def current_attempt_spent() -> Decimal:
            return spent() - resumed_prior_estimated_cost

        next_compiler_pass = max((stage.pass_number for stage in compiler_stages), default=0) + 1
        compiler_attempt_calls = len(compiler_stages) - resumed_prior_compiler_stages
        remaining_compiler_calls = max(
            0, config.workflow.max_compiler_passes - compiler_attempt_calls
        )
        compiler_passes: tuple[int | None, ...] = (
            (None,)
            if drafts is not None
            else tuple(range(next_compiler_pass, next_compiler_pass + remaining_compiler_calls))
        )
        if not compiler_passes:
            reasons.append("compiler checkpoint is absent and configured passes are exhausted")

        for compiler_pass in compiler_passes:
            if compiler_pass is not None:
                if (
                    compiler_stages
                    and current_attempt_spent()
                    >= config.workflow.additional_call_launch_threshold_usd_per_document
                ):
                    reasons.append(
                        "current-attempt additional-call launch threshold reached before another "
                        f"compiler call: ${current_attempt_spent()} >= "
                        f"${config.workflow.additional_call_launch_threshold_usd_per_document}"
                    )
                    break
                output, stage = await runtime.compiler(
                    pass_number=compiler_pass,
                    system_prompt=compiler_prompt,
                    payload=_compiler_payload(
                        document_id=document_id,
                        raw=raw,
                        source_target=source_target,
                        feature=feature,
                        anchors=anchors,
                        risks=risks,
                        prior_error=revision_context,
                        prior_output=prior_compiler_output,
                        prior_drafts=prior_compiler_drafts,
                        agent_contract_protocol=config.workflow.agent_contract_protocol,
                    ),
                    retries=config.workflow.compiler_output_retries,
                )
                if output is None:
                    compiler_stages.append(stage)
                    reasons.append(stage.error_message or "compiler provider failed")
                    break
                prior_compiler_output = output
                try:
                    drafts, assessment, semantic_only_target_facts = _validated_compiler_state(
                        output=output,
                        raw=raw,
                        source_target=source_target,
                        anchors=anchors,
                    )
                    prior_compiler_drafts = drafts
                    compiler_stages.append(stage)
                    critic_revision_context = None
                except Exception as error:
                    rejection = _compiler_host_rejection(
                        output=output,
                        raw=raw,
                        source_target=source_target,
                        anchors=anchors,
                        primary_error=error,
                    )
                    compiler_stages.append(_mark_host_rejected(stage, rejection))
                    revision_context = str(rejection)
                    compiler_attempt_calls += 1
                    if compiler_attempt_calls == config.workflow.max_compiler_passes:
                        reasons.append(f"compiler host rejection: {rejection}")
                    continue

            if drafts is None or assessment is None:
                raise AssertionError("critic phase entered without a validated compiler checkpoint")

            while critic_attempt_calls < config.workflow.max_critic_passes:
                stagnation_reason = _critic_stagnation_reason(
                    critic_stages[resumed_prior_critic_stages:]
                )
                if stagnation_reason is not None:
                    reasons.append(stagnation_reason)
                    break
                launch_critic, post_threshold_confirmation_used = _critic_launch_decision(
                    spent=current_attempt_spent(),
                    threshold=(config.workflow.additional_call_launch_threshold_usd_per_document),
                    has_unconfirmed_state=(
                        not critic_stages[resumed_prior_critic_stages:]
                        or bool(critic_outputs and critic_outputs[-1].verdict == "revise")
                    ),
                    post_threshold_confirmation_used=post_threshold_confirmation_used,
                )
                if not launch_critic:
                    reasons.append(
                        "current-attempt additional-call launch threshold reached before another "
                        f"critic call: ${current_attempt_spent()} >= "
                        f"${config.workflow.additional_call_launch_threshold_usd_per_document}"
                    )
                    break
                critic_pass += 1
                critic_attempt_calls += 1
                review, critic_stage = await runtime.critic(
                    pass_number=critic_pass,
                    system_prompt=critic_prompt,
                    payload=_critic_payload(
                        document_id=document_id,
                        raw=raw,
                        source_target=source_target,
                        feature=feature,
                        drafts=drafts,
                        risks=risks,
                        prior_error=critic_revision_context,
                        semantic_only_target_facts=semantic_only_target_facts,
                        review_history=critic_stages,
                        agent_contract_protocol=config.workflow.agent_contract_protocol,
                    ),
                    retries=config.workflow.critic_output_retries,
                )
                if review is None:
                    critic_stages.append(critic_stage)
                    reasons.append(critic_stage.error_message or "critic provider failed")
                    break
                if review.verdict == "revise":
                    try:
                        review_semantic_only = materialize_semantic_only_target_facts(
                            proposals=review.semantic_only_target_facts,
                            source_target=source_target,
                            provenance="critic_audited_unprinted",
                        )
                        drafts = normalize_country_code_locality(
                            raw=raw,
                            drafts=normalize_deterministic_draft_semantics(
                                drafts=normalize_compact_equipment_locality(
                                    raw=raw,
                                    drafts=apply_critic_patch(
                                        raw=raw,
                                        drafts=drafts,
                                        findings=review.findings,
                                        remove_inventory_binding_ids=(
                                            review.remove_inventory_binding_ids
                                        ),
                                        additional_bindings=review.additional_bindings,
                                        occurrence_removals=review.occurrence_removals,
                                        semantic_only_target_paths=tuple(
                                            fact.target_path for fact in review_semantic_only
                                        ),
                                        source_target=source_target,
                                    ),
                                    source_target=source_target,
                                ),
                                source_target=source_target,
                            ),
                        )
                        validate_carrier_assessment(
                            assessment=assessment,
                            expected=source_carrier(source_target),
                            raw=raw,
                            anchor_drafts_value=drafts,
                        )
                        validate_binding_realizations(
                            raw=raw, drafts=drafts, source_target=source_target
                        )
                        semantic_only_target_facts = _effective_semantic_only_target_facts(
                            facts=_merge_semantic_only_target_facts(
                                semantic_only_target_facts,
                                review_semantic_only,
                            ),
                            drafts=drafts,
                        )
                    except Exception as error:
                        critic_stages.append(_mark_host_rejected(critic_stage, error))
                        critic_revision_context = f"Host rejected critic patch: {error}"
                        if critic_attempt_calls == config.workflow.max_critic_passes:
                            reasons.append(
                                f"critic exhausted passes after host-rejected patch: {error}"
                            )
                            break
                        continue
                    critic_outputs.append(review)
                    critic_stages.append(critic_stage)
                    critic_revision_context = None
                    if critic_attempt_calls == config.workflow.max_critic_passes:
                        reasons.append("critic exhausted passes after local binding patch")
                    continue
                remaining_risks = uncovered_risks(raw, risks, drafts)
                if remaining_risks:
                    details = "; ".join(
                        f"{row.risk_id} {row.line_id}={row.source_text!r}"
                        for row in remaining_risks
                    )
                    error = ValueError(
                        "critic passed while deterministic risk candidates remained unowned: "
                        + details
                    )
                    critic_stages.append(_mark_host_rejected(critic_stage, error))
                    critic_revision_context = f"Host rejected critic pass: {error}"
                    if critic_attempt_calls == config.workflow.max_critic_passes:
                        reasons.append(
                            "critic exhausted passes after false pass with unowned risks: "
                            + details
                        )
                        break
                    continue
                try:
                    critic_outputs.append(review)
                    template = certify_template(
                        raw=raw,
                        document_id=document_id,
                        feature=feature,
                        source_target=source_target,
                        assessment=assessment,
                        drafts=drafts,
                        risks=risks,
                        critic_outputs=critic_outputs,
                        semantic_only_target_facts=semantic_only_target_facts,
                    )
                    critic_stages.append(critic_stage)
                    break
                except Exception as error:
                    critic_stages.append(_mark_host_rejected(critic_stage, error))
                    critic_revision_context = f"Host rejected critic pass or certification: {error}"
                    if critic_attempt_calls == config.workflow.max_critic_passes:
                        reasons.append(
                            f"critic exhausted passes after failed certification: {error}"
                        )
                        break
                    continue

            if (
                template is None
                and not reasons
                and critic_attempt_calls == config.workflow.max_critic_passes
            ):
                reasons.append("template did not reach a final critic pass")
            break

        if template is None and not reasons:
            reasons.append("template did not reach a final critic pass")

        masked = masked_source(raw, drafts or anchors)
        template_payload = (
            json_artifact_bytes(template.model_dump(mode="json")) if template else None
        )
        template_sha = sha256_bytes(template_payload) if template_payload else None
        result = ExtractionCaseResult.model_validate(
            {
                "document_id": document_id,
                "status": "certified" if template is not None else "rejected",
                "rejection_reasons": tuple(reasons),
                "template_sha256": template_sha,
                "compiler_stages": tuple(compiler_stages),
                "critic_stages": tuple(critic_stages),
                "risk_candidates": risks,
                "elapsed_seconds": prior_elapsed_seconds + time.perf_counter() - started,
                "resumed_from_run": resumed_from_run,
                "resumed_prior_elapsed_seconds": prior_elapsed_seconds,
                "resumed_prior_compiler_stages": resumed_prior_compiler_stages,
                "resumed_prior_critic_stages": resumed_prior_critic_stages,
                "resumed_prior_estimated_cost_usd": resumed_prior_estimated_cost,
                "resume_contract_match": (
                    resume_contract_match if resume_result is not None else None
                ),
                "resume_replay_warning": resume_replay_warning,
                "resume_mode": ("continued_with_provider" if resume_result is not None else "none"),
            }
        )
        staged.publish_bytes(f"{relative_root}/source.txt", raw.encode("utf-8"))
        staged.publish_json(f"{relative_root}/source-label.json", source_target)
        staged.publish_bytes(f"{relative_root}/masked-template.txt", masked.encode("utf-8"))
        if drafts is not None and assessment is not None:
            checkpoint = _state_checkpoint(
                document_id=document_id,
                raw=raw,
                source_target=source_target,
                drafts=drafts,
                assessment=assessment,
                semantic_only_target_facts=semantic_only_target_facts,
                critic_outputs=critic_outputs,
            )
            staged.publish_json(
                f"{relative_root}/state-checkpoint.json",
                checkpoint.model_dump(mode="json"),
            )
        if template_payload is not None:
            staged.publish_bytes(f"{relative_root}/template.json", template_payload)
            staged.publish_json(f"{relative_root}/catalog-row.json", template_summary(template))
        staged.publish_json(f"{relative_root}/result.json", result.model_dump(mode="json"))
        case_cost = sum(
            (stage.usage.estimatedCostUsd for stage in (*compiler_stages, *critic_stages)),
            Decimal(0),
        )
        marginal_case_cost = case_cost - resumed_prior_estimated_cost
        print(
            f"[{ordinal:03d}/{config.workflow.documents:03d}] {result.status} {document_id} "
            f"compiler_calls={sum(stage.usage.requests for stage in compiler_stages)} "
            f"critic_calls={sum(stage.usage.requests for stage in critic_stages)} "
            f"cumulative_cost=${case_cost} marginal_cost=${marginal_case_cost}",
            flush=True,
        )
        return result, template, masked


def _usage_summary(
    results: Sequence[ExtractionCaseResult], *, marginal_only: bool = False
) -> dict[str, Any]:
    stages = [
        stage
        for result in results
        for stage in (
            *(
                result.compiler_stages[result.resumed_prior_compiler_stages :]
                if marginal_only
                else result.compiler_stages
            ),
            *(
                result.critic_stages[result.resumed_prior_critic_stages :]
                if marginal_only
                else result.critic_stages
            ),
        )
    ]
    compiler = [stage for stage in stages if stage.role == "compiler"]
    critic = [stage for stage in stages if stage.role == "critic"]

    def total(rows: Sequence[AgentStageArtifact]) -> dict[str, Any]:
        return {
            "stages": len(rows),
            "requests": sum(row.usage.requests for row in rows),
            "inputTokens": sum(row.usage.inputTokens for row in rows),
            "cacheReadTokens": sum(row.usage.cacheReadTokens for row in rows),
            "outputTokens": sum(row.usage.outputTokens for row in rows),
            "reasoningTokens": sum(row.usage.reasoningTokens for row in rows),
            "visibleOutputTokens": sum(row.usage.visibleOutputTokens for row in rows),
            "estimatedCostUsd": str(sum((row.usage.estimatedCostUsd for row in rows), Decimal(0))),
            "providerReportedCostUsd": (
                str(
                    sum(
                        (
                            row.usage.providerReportedCostUsd
                            for row in rows
                            if row.usage.providerReportedCostUsd is not None
                        ),
                        Decimal(0),
                    )
                )
                if rows
                and all(
                    row.usage.providerReportedCostUsd is not None
                    for row in rows
                    if row.usage.requests
                )
                else None
            ),
        }

    return {"compiler": total(compiler), "critic": total(critic), "all": total(stages)}


def _report(config: ExtractionConfig, summary: Mapping[str, Any]) -> str:
    usage = cast(Mapping[str, Any], summary["usage"])
    all_usage = cast(Mapping[str, Any], usage["all"])
    counts = cast(Mapping[str, Any], summary["statusCounts"])
    return "\n".join(
        [
            f"# Carrier-bound template extraction: {config.phase}",
            "",
            "## Result",
            "",
            f"- Selected documents: **{summary['documents']}**.",
            f"- Certified templates: **{counts.get('certified', 0)}**.",
            f"- Rejected templates: **{counts.get('rejected', 0)}**.",
            f"- Resumed cases: **{summary['resumedCases']}**"
            + (
                f" from **{summary['resumeFromRun']}**."
                if summary["resumeFromRun"] is not None
                else "."
            ),
            f"- Resume replay warnings: **{summary['resumeReplayWarnings']}**.",
            f"- Resume modes: **{summary['resumeModeCounts']}**.",
            f"- Certified resume policy: **{summary['certifiedResumePolicy']}**.",
            "- Resume prompt changed / host contract matched / certified reuse compatible: "
            f"**{summary['resumePromptChanged']} / "
            f"{summary['resumeHostContractMatched']} / "
            f"{summary['resumeCertifiedReuseCompatible']}**.",
            f"- All-document acceptance gate: **{summary['acceptanceGatePassed']}**.",
            f"- Provider requests: **{all_usage['requests']}**.",
            f"- Input / output / reasoning tokens: **{all_usage['inputTokens']} / "
            f"{all_usage['outputTokens']} / {all_usage['reasoningTokens']}**.",
            f"- Estimated provider cost: **${all_usage['estimatedCostUsd']}**.",
            "- Marginal provider cost in this run: "
            f"**${summary['marginalUsage']['all']['estimatedCostUsd']}**.",
            "- Additional-call launch threshold per document: "
            f"**${summary['additionalCallLaunchThresholdUsdPerDocument']}** "
            "(applies to new calls in this execution, excluding pinned resume spend; checked "
            "before a request; one final critic confirmation may launch after an accepted patch; "
            "not a hard post-request ceiling).",
            f"- Deterministic / agent-assisted bindings: "
            f"**{summary['deterministicBindings']} / {summary['agentAssistedBindings']}**.",
            f"- Audited semantic-only target facts: **{summary['semanticOnlyTargetFacts']}**.",
            f"- Peak process RSS: **{summary['peakRssMiB']:.3f} MiB**.",
            f"- Wall time: **{summary['wallSeconds']:.3f} seconds**.",
            "- Training records published: **false**.",
            "",
            "## Contract",
            "",
            "Every certified template passed exact source hashing and round trip, disjoint UTF-8 "
            "spans, literal-byte and page-marker preservation, all-slot sentinel isolation, "
            "exact carrier evidence (plus source-label equality when a label exists), "
            "deterministic risk ownership, and a final independent literal-remainder critic pass. "
            "Every binding also has a host-validated realization plan; mappings the host cannot "
            "prove deterministic are explicitly marked agent-required. A rejected case contributes "
            "no catalog template.",
            "",
            "## Artifact map",
            "",
            "- `selection-manifest.json`: frozen document selection and strata.",
            "- `cases/<documentId>/template.json`: certified semantic template.",
            "- `cases/<documentId>/masked-template.txt`: human-readable literal remainder.",
            "- `cases/<documentId>/result.json`: all compiler/critic attempts and usage.",
            "- `catalog.jsonl` and `results.jsonl`: ordered machine-readable aggregates.",
            "- No artifact in this run is a training-data publication.",
            "",
        ]
    )


async def run_extraction(config_path: Path) -> Path:
    project_root = project_root_from_config(config_path)
    config = load_config(config_path)
    if not config.workflow.provider_launch_authorized:
        raise ValueError(
            f"provider launch is not authorized for {config.phase}; pass the preceding quality "
            "gate and explicitly update the pinned config first"
        )
    (
        source_rows,
        feature_rows,
        anchor_rows,
        current_target_rows,
        findings_path,
        invariants_path,
        manual_path,
        compiler_prompt,
        critic_prompt,
    ) = _config_inputs(config, project_root)
    manifest = build_selection_manifest(
        config=config,
        source_rows=source_rows,
        feature_rows=feature_rows,
        anchor_rows=anchor_rows,
        current_target_rows=current_target_rows,
        findings_path=findings_path,
        invariants_path=invariants_path,
        manual_path=manual_path,
    )
    resume_run_root, resume_results, resume_prompt_changed, resume_host_contract_matched = (
        _load_resume_run(
            project_root=project_root,
            config_path=config_path,
            config=config,
            manifest=manifest,
        )
    )
    resume_contract_match = resume_host_contract_matched and not resume_prompt_changed
    preflight = _preflight_summary(
        config=config,
        manifest=manifest,
        source_rows=source_rows,
        feature_rows=feature_rows,
        anchor_rows=anchor_rows,
        compiler_prompt=compiler_prompt,
        critic_prompt=critic_prompt,
    )
    preflight["resumeFromRun"] = resume_run_root.name if resume_run_root else None
    preflight["resumeCases"] = len(resume_results)
    preflight["resumePromptChanged"] = resume_prompt_changed
    preflight["resumeHostContractMatched"] = resume_host_contract_matched
    preflight["resumeCertifiedReuseCompatible"] = resume_contract_match
    preflight["resumeStatusCounts"] = dict(
        sorted(Counter(row.status for row in resume_results.values()).items())
    )
    preflight["certifiedResumePolicy"] = config.workflow.certified_resume_policy
    transaction_payload = {
        "config": config.model_dump(mode="json"),
        "selection": manifest.model_dump(mode="json"),
        "experimentSourceSha256": _experiment_source_hash(config_path),
    }
    transaction_sha = sha256_bytes(canonical_json_bytes(transaction_payload))
    output_parent = (project_root / config.output_dir).resolve()
    staged = StagedArtifactRun(
        output_parent=output_parent,
        run_name=config.run_name,
        transaction_sha256=transaction_sha,
    )
    if staged.completed:
        print(staged.final_root)
        return staged.final_root
    staged.recover_interrupted_temporary_files()
    staged.publish_json("config.json", config.model_dump(mode="json"))
    staged.publish_json("preflight.json", preflight)
    staged.publish_json("selection-manifest.json", manifest.model_dump(mode="json"))
    staged.publish_json("resume-contract.json", _resume_contract(config_path))
    staged.publish_bytes("prompts/compiler.md", compiler_prompt.encode("utf-8"))
    staged.publish_bytes("prompts/critic.md", critic_prompt.encode("utf-8"))

    sources = {str(row["documentId"]): row for row in source_rows}
    features = {str(row["document_id"]): row for row in feature_rows}
    anchors_by_id: dict[str, list[Mapping[str, Any]]] = {}
    for row in anchor_rows:
        anchors_by_id.setdefault(str(row["document_id"]), []).append(row)
    runtime = AgentRuntime(
        project_root=project_root,
        environment_file=config.environment_file,
        compiler_provider=config.compiler_provider,
        critic_provider=config.critic_provider,
        agent_contract_protocol=config.workflow.agent_contract_protocol,
        max_concurrent_requests=config.workflow.max_concurrent_requests,
    )
    document_limiter = asyncio.Semaphore(config.workflow.max_concurrent_documents)
    started = time.perf_counter()
    tasks = [
        asyncio.create_task(
            _extract_case(
                ordinal=row.ordinal,
                document_id=row.document_id,
                source=sources[row.document_id],
                feature=features[row.document_id],
                anchor_rows=anchors_by_id[row.document_id],
                runtime=runtime,
                compiler_prompt=compiler_prompt,
                critic_prompt=critic_prompt,
                config=config,
                staged=staged,
                document_limiter=document_limiter,
                resume_run_root=resume_run_root,
                resume_result=resume_results.get(row.document_id),
                resume_contract_match=resume_contract_match,
            )
        )
        for row in manifest.rows
    ]
    bundles = await asyncio.gather(*tasks)
    results = [bundle[0] for bundle in bundles]
    templates = [bundle[1] for bundle in bundles if bundle[1] is not None]
    wall = time.perf_counter() - started
    status_counts = Counter(row.status for row in results)
    usage = _usage_summary(results)
    marginal_usage = _usage_summary(results, marginal_only=True)
    mode_counts: Counter[str] = Counter(
        binding.render_mode for template in templates for binding in template.bindings
    )
    realization_counts: Counter[str] = Counter(
        binding.realization.mode for template in templates for binding in template.bindings
    )
    summary = {
        "schemaVersion": 1,
        "phase": config.phase,
        "documents": len(results),
        "statusCounts": dict(sorted(status_counts.items())),
        "certifiedFraction": len(templates) / len(results),
        "acceptanceGatePassed": len(templates) == len(results),
        "bindings": sum(len(template.bindings) for template in templates),
        "occurrences": sum(len(template.byte_template.slots) for template in templates),
        "renderModeCounts": dict(sorted(mode_counts.items())),
        "realizationModeCounts": dict(sorted(realization_counts.items())),
        "agentResidualBindings": mode_counts.get("agent_residual", 0),
        "agentAssistedBindings": realization_counts.get("agent_required", 0),
        "semanticOnlyTargetFacts": sum(
            len(template.semantic_only_target_facts) for template in templates
        ),
        "deterministicBindings": sum(realization_counts.values())
        - realization_counts.get("agent_required", 0),
        "usage": usage,
        "marginalUsage": marginal_usage,
        "wallSeconds": wall,
        "peakRssMiB": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024,
        "completedAt": datetime.now(UTC).isoformat(),
        "publishTrainingRecords": False,
        "resumeFromRun": resume_run_root.name if resume_run_root else None,
        "resumedCases": sum(result.resumed_from_run is not None for result in results),
        "resumeReplayWarnings": sum(result.resume_replay_warning is not None for result in results),
        "resumeModeCounts": dict(sorted(Counter(result.resume_mode for result in results).items())),
        "resumePromptChanged": resume_prompt_changed,
        "resumeHostContractMatched": resume_host_contract_matched,
        "resumeCertifiedReuseCompatible": resume_contract_match,
        "certifiedResumePolicy": config.workflow.certified_resume_policy,
        "additionalCallLaunchThresholdUsdPerDocument": str(
            config.workflow.additional_call_launch_threshold_usd_per_document
        ),
    }
    staged.publish_bytes(
        "results.jsonl",
        _jsonl_bytes([row.model_dump(mode="json") for row in results]),
    )
    staged.publish_bytes(
        "catalog.jsonl",
        _jsonl_bytes([template_summary(row) for row in templates]),
    )
    staged.publish_json("summary.json", summary)
    staged.publish_bytes("REPORT.md", _report(config, summary).encode("utf-8"))

    expected = [
        "REPORT.md",
        "catalog.jsonl",
        "config.json",
        "prompts/compiler.md",
        "prompts/critic.md",
        "preflight.json",
        "resume-contract.json",
        "results.jsonl",
        "selection-manifest.json",
        "summary.json",
    ]
    for result in results:
        root = f"cases/{result.document_id}"
        expected.extend(
            [
                f"{root}/masked-template.txt",
                f"{root}/result.json",
                f"{root}/source-label.json",
                f"{root}/source.txt",
            ]
        )
        checkpoint_path = staged.stage_root / root / "state-checkpoint.json"
        if checkpoint_path.is_file():
            expected.append(f"{root}/state-checkpoint.json")
        if result.status == "certified":
            expected.extend([f"{root}/catalog-row.json", f"{root}/template.json"])
    staged.commit(
        expected_artifacts=expected,
        metadata={
            "schemaVersion": 1,
            "phase": config.phase,
            "documents": len(results),
            "certified": len(templates),
            "publishTrainingRecords": False,
        },
    )
    print(staged.final_root)
    return staged.final_root


def prepare_selection(config_path: Path) -> SelectionManifest:
    project_root = project_root_from_config(config_path)
    config = load_config(config_path)
    (
        source_rows,
        feature_rows,
        anchor_rows,
        current_target_rows,
        findings_path,
        invariants_path,
        manual_path,
        _compiler_prompt,
        _critic_prompt,
    ) = _config_inputs(config, project_root)
    return build_selection_manifest(
        config=config,
        source_rows=source_rows,
        feature_rows=feature_rows,
        anchor_rows=anchor_rows,
        current_target_rows=current_target_rows,
        findings_path=findings_path,
        invariants_path=invariants_path,
        manual_path=manual_path,
    )


def preflight_extraction(config_path: Path) -> dict[str, Any]:
    project_root = project_root_from_config(config_path)
    config = load_config(config_path)
    (
        source_rows,
        feature_rows,
        anchor_rows,
        current_target_rows,
        findings_path,
        invariants_path,
        manual_path,
        compiler_prompt,
        critic_prompt,
    ) = _config_inputs(config, project_root)
    manifest = build_selection_manifest(
        config=config,
        source_rows=source_rows,
        feature_rows=feature_rows,
        anchor_rows=anchor_rows,
        current_target_rows=current_target_rows,
        findings_path=findings_path,
        invariants_path=invariants_path,
        manual_path=manual_path,
    )
    resume_run_root, resume_results, resume_prompt_changed, resume_host_contract_matched = (
        _load_resume_run(
            project_root=project_root,
            config_path=config_path,
            config=config,
            manifest=manifest,
        )
    )
    summary = _preflight_summary(
        config=config,
        manifest=manifest,
        source_rows=source_rows,
        feature_rows=feature_rows,
        anchor_rows=anchor_rows,
        compiler_prompt=compiler_prompt,
        critic_prompt=critic_prompt,
    )
    summary["resumeFromRun"] = resume_run_root.name if resume_run_root else None
    summary["resumeCases"] = len(resume_results)
    summary["resumePromptChanged"] = resume_prompt_changed
    summary["resumeHostContractMatched"] = resume_host_contract_matched
    summary["resumeCertifiedReuseCompatible"] = (
        resume_host_contract_matched and not resume_prompt_changed
    )
    summary["resumeStatusCounts"] = dict(
        sorted(Counter(row.status for row in resume_results.values()).items())
    )
    summary["certifiedResumePolicy"] = config.workflow.certified_resume_policy
    return summary


def publish_offline_audit(
    *, development_config_path: Path, transfer_config_path: Path, run_name: str
) -> Path:
    project_root = project_root_from_config(development_config_path)
    if project_root_from_config(transfer_config_path) != project_root:
        raise ValueError("development and transfer configs resolve to different repositories")
    development_config = load_config(development_config_path)
    transfer_config = load_config(transfer_config_path)
    if development_config.phase != "development30" or transfer_config.phase != "transfer200":
        raise ValueError("offline audit requires development30 and transfer200 configs")
    if development_config.inputs != transfer_config.inputs:
        raise ValueError("development and transfer input pins differ")
    if development_config.prompts != transfer_config.prompts:
        raise ValueError("development and transfer prompt pins differ")
    if development_config.output_dir != transfer_config.output_dir:
        raise ValueError("development and transfer output directories differ")

    transaction_payload = {
        "developmentConfig": development_config.model_dump(mode="json"),
        "transferConfig": transfer_config.model_dump(mode="json"),
        "experimentSourceSha256": _experiment_source_hash(development_config_path),
    }
    transaction_sha = sha256_bytes(canonical_json_bytes(transaction_payload))
    staged = StagedArtifactRun(
        output_parent=(project_root / development_config.output_dir).resolve(),
        run_name=run_name,
        transaction_sha256=transaction_sha,
    )
    if staged.completed:
        return staged.final_root
    staged.recover_interrupted_temporary_files()

    (
        source_rows,
        feature_rows,
        anchor_rows,
        current_target_rows,
        findings_path,
        invariants_path,
        manual_path,
        compiler_prompt,
        critic_prompt,
    ) = _config_inputs(development_config, project_root)
    carrier_audit = _corpus_carrier_resolution_report(
        config=development_config,
        source_rows=source_rows,
        feature_rows=feature_rows,
        anchor_rows=anchor_rows,
    )
    development_selection = build_selection_manifest(
        config=development_config,
        source_rows=source_rows,
        feature_rows=feature_rows,
        anchor_rows=anchor_rows,
        current_target_rows=current_target_rows,
        findings_path=findings_path,
        invariants_path=invariants_path,
        manual_path=manual_path,
    )
    development_preflight = _preflight_summary(
        config=development_config,
        manifest=development_selection,
        source_rows=source_rows,
        feature_rows=feature_rows,
        anchor_rows=anchor_rows,
        compiler_prompt=compiler_prompt,
        critic_prompt=critic_prompt,
    )
    transfer_selection = build_selection_manifest(
        config=transfer_config,
        source_rows=source_rows,
        feature_rows=feature_rows,
        anchor_rows=anchor_rows,
        current_target_rows=current_target_rows,
        findings_path=findings_path,
        invariants_path=invariants_path,
        manual_path=manual_path,
    )
    transfer_preflight = _preflight_summary(
        config=transfer_config,
        manifest=transfer_selection,
        source_rows=source_rows,
        feature_rows=feature_rows,
        anchor_rows=anchor_rows,
        compiler_prompt=compiler_prompt,
        critic_prompt=critic_prompt,
    )
    development_ids = {row.document_id for row in development_selection.rows}
    transfer_ids = {row.document_id for row in transfer_selection.rows}
    overlap = sorted(development_ids & transfer_ids)
    if overlap:
        raise ValueError("development and transfer selections overlap: " + ", ".join(overlap))
    if set(transfer_config.excluded_document_ids) != development_ids:
        raise ValueError("transfer exclusions do not exactly equal the development selection")

    summary = {
        "schemaVersion": 2,
        "status": "offline_preflight_provider_disabled",
        "providerRequests": 0,
        "providerCostUsd": "0",
        "developmentDocuments": len(development_ids),
        "transferDocuments": len(transfer_ids),
        "combinedDocuments": len(development_ids | transfer_ids),
        "developmentTransferOverlap": 0,
        "developmentMissingSourceCarrier": sum(
            row.carrier_name is None for row in development_selection.rows
        ),
        "transferMissingSourceCarrier": sum(
            row.carrier_name is None for row in transfer_selection.rows
        ),
        "corpusDocuments": carrier_audit["documents"],
        "carrierBoundEligibleDocuments": carrier_audit["eligibleForCarrierBoundExtraction"],
        "externalCarrierEnrichmentRequired": carrier_audit["requiresExternalCarrierEnrichment"],
        "developmentExtractionAuthorized": (development_config.workflow.provider_launch_authorized),
        "transferExtractionAuthorized": transfer_config.workflow.provider_launch_authorized,
        "developmentLaunchRequirement": (
            "workflow.provider_launch_authorized must be explicitly set true"
        ),
        "transferLaunchRequirement": (
            "workflow.provider_launch_authorized must be explicitly set true"
        ),
        "publishTrainingRecords": False,
    }
    report = "\n".join(
        [
            "# Carrier-bound template v2 offline audit",
            "",
            "## Result",
            "",
            f"- Corpus rows classified: **{summary['corpusDocuments']}**.",
            "- Immediately carrier-bound eligible: "
            f"**{summary['carrierBoundEligibleDocuments']}**.",
            "- Requires external carrier enrichment: "
            f"**{summary['externalCarrierEnrichmentRequired']}**.",
            f"- Development / transfer selections: **{summary['developmentDocuments']} / "
            f"{summary['transferDocuments']}**, with zero overlap.",
            "- Provider requests and cost: **0 / $0** (offline only).",
            "- Provider launch flags: development "
            f"**{str(summary['developmentExtractionAuthorized']).lower()}**, transfer "
            f"**{str(summary['transferExtractionAuthorized']).lower()}**.",
            "- This receipt validates current inputs and selections only; it does not infer "
            "historical live-gate status.",
            "- Training records published: **false**.",
            "",
        ]
    )
    artifacts = {
        "REPORT.md": report.encode("utf-8"),
        "carrier-resolution-audit.json": json_artifact_bytes(carrier_audit),
        "development30-config.json": json_artifact_bytes(
            development_config.model_dump(mode="json")
        ),
        "development30-preflight.json": json_artifact_bytes(development_preflight),
        "development30-selection.json": json_artifact_bytes(
            development_selection.model_dump(mode="json")
        ),
        "summary.json": json_artifact_bytes(summary),
        "transfer200-config.json": json_artifact_bytes(transfer_config.model_dump(mode="json")),
        "transfer200-preflight.json": json_artifact_bytes(transfer_preflight),
        "transfer200-selection.json": json_artifact_bytes(
            transfer_selection.model_dump(mode="json")
        ),
    }
    for relative_path, payload in artifacts.items():
        staged.publish_bytes(relative_path, payload)
    staged.commit(
        expected_artifacts=artifacts,
        metadata={
            "schemaVersion": 2,
            "status": "offline_preflight_provider_disabled",
            "documents": summary["combinedDocuments"],
            "providerRequests": 0,
            "publishTrainingRecords": False,
        },
    )
    return staged.final_root


def _resume_contract(config_path: Path) -> dict[str, Any]:
    """Hash runtime experiment code separately from per-run configs and prompts."""

    experiment_root = config_path.resolve(strict=True).parent.parent
    candidates = [experiment_root / "pyproject.toml", experiment_root / "uv.lock"]
    candidates.extend(sorted((experiment_root / "src").rglob("*.py")))
    paths = tuple(path for path in candidates if path.is_file() and not path.is_symlink())
    files = tuple(
        {
            "path": path.relative_to(experiment_root).as_posix(),
            "sha256": sha256_file(path),
        }
        for path in paths
    )
    if len(files) != len(candidates):
        raise ValueError("resume contract contains a missing or symbolic runtime source file")
    return {
        "schemaVersion": 1,
        "hostImplementationSha256": sha256_bytes(canonical_json_bytes(files)),
        "files": files,
    }


def _experiment_source_hash(config_path: Path) -> str:
    experiment_root = config_path.resolve(strict=True).parent.parent
    ignored_directories = {".pytest_cache", ".ruff_cache", ".venv", "__pycache__"}
    paths: list[Path] = []
    for directory, directory_names, filenames in os.walk(experiment_root, followlinks=False):
        directory_names[:] = sorted(
            name
            for name in directory_names
            if name not in ignored_directories and not (Path(directory) / name).is_symlink()
        )
        for name in sorted(filenames):
            path = Path(directory) / name
            if path.is_file() and not path.is_symlink():
                paths.append(path)
    paths.sort()
    payload = [
        {
            "path": path.relative_to(experiment_root).as_posix(),
            "sha256": sha256_file(path),
        }
        for path in paths
    ]
    return sha256_bytes(canonical_json_bytes(payload))
