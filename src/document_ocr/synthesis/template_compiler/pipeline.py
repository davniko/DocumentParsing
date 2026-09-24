from __future__ import annotations

import asyncio
import json
import re
import resource
import time
from collections import Counter, defaultdict
from collections.abc import Mapping, Sequence
from contextlib import suppress
from dataclasses import asdict
from datetime import UTC, datetime
from decimal import Decimal
from functools import partial
from pathlib import Path
from typing import Any, cast

import yaml

from document_ocr.atomic import json_artifact_bytes, read_regular_file_bytes
from document_ocr.hashing import canonical_json_bytes, sha256_bytes, sha256_file
from document_ocr.synthesis.raw_text_rewrite_cycle_probe import CustomsProgramRegistry
from document_ocr.synthesis.run_safety import StagedArtifactRun
from document_ocr.synthesis.template_integrity import source_template_integrity_issues

from .agents import AgentRuntime
from .coherence import (
    CoherenceReviewRequired,
    coherence_review_candidates,
    materialize_coherence_decisions,
    reconcile_coherence_constraints_after_binding_revision,
    validate_coherence_contracts,
)
from .customs_presentation import normalize_registered_caption_ownership
from .host import (
    SpanDraft,
    all_risk_candidates,
    anchor_drafts,
    anchor_summary,
    annotated_source,
    apply_anchor_overrides,
    apply_critic_patch,
    binding_contract_signature,
    capability_contract,
    certify_template,
    is_token_bounded_surface_span,
    line_spans,
    load_jsonl,
    masked_source,
    materialize_semantic_only_target_facts,
    normalize_competing_auxiliary_outliers,
    normalize_deterministic_draft_semantics,
    normalize_pinned_carrier_assessment,
    normalize_relational_anchor_locality,
    normalize_source_boundaries,
    normalize_structured_row_locality,
    normalize_target_cobindings,
    numbered_source,
    reconcile_draft_overlaps,
    refuted_semantic_only_target_paths,
    required_target_cobindings,
    resolve_agent_proposal_inventory,
    resolve_agent_proposals,
    source_carrier,
    target_binding_surface_analysis,
    target_fact_components,
    target_path_relationship,
    template_summary,
    uncovered_risks,
    validate_agent_proposal_paths,
    validate_binding_realizations,
    validate_carrier_assessment,
    validate_draft_source_alignment,
    validate_mutable_token_boundaries,
    validate_repeated_binding_fact_topology,
    validate_target_binding_relationships,
    verify_file,
)
from .models import (
    AgentContractProtocol,
    AgentStageArtifact,
    AnchorOverride,
    CarrierAssessment,
    CertifiedSemanticTemplate,
    CoherenceConstraint,
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

_RELATIONAL_ANCHOR_PROTOCOLS = {
    "relational_reference_compact_v9",
    "hybrid_reference_partitioned_v10",
}


def project_root_from_config(config_path: Path) -> Path:
    current = config_path.resolve(strict=True).parent
    while current != current.parent:
        if (current / "pyproject.toml").is_file() and (current / "src/document_ocr").is_dir():
            return current
        current = current.parent
    raise ValueError("cannot locate repository root from extraction config")


def _validated_project_root(*, project_root: Path, config_path: Path) -> Path:
    """Bind an extraction invocation to the repository named by its caller."""

    if project_root.is_symlink():
        raise ValueError("project root must not be a symbolic link")
    resolved = project_root.resolve(strict=True)
    if not resolved.is_dir():
        raise ValueError("project root must be a directory")
    detected = project_root_from_config(config_path)
    if resolved != detected:
        raise ValueError(
            f"configured project root differs from the config repository: {resolved} != {detected}"
        )
    return resolved


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


def _require_exact_resume_prefix(
    *,
    prior_identity: tuple[tuple[int, str, str], ...],
    current_identity: tuple[tuple[int, str, str], ...],
) -> None:
    """Require a non-empty prior selection to be the exact current prefix."""

    if (
        not prior_identity
        or len(prior_identity) > len(current_identity)
        or prior_identity != current_identity[: len(prior_identity)]
    ):
        raise ValueError(
            "resume selection identity is not an exact prefix of the current pinned selection"
        )


def _load_resume_run(
    *,
    project_root: Path,
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
    for field in ("task", "phase", "inputs", "selection_seed", "customs_program_registry"):
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
        contract_matched = prior_contract == _resume_contract(project_root)
    prior_manifest = SelectionManifest.model_validate_json(
        read_regular_file_bytes(prior_selection_path), strict=True
    )
    current_identity = tuple(
        (row.ordinal, row.document_id, row.source_sha256) for row in manifest.rows
    )
    prior_identity = tuple(
        (row.ordinal, row.document_id, row.source_sha256) for row in prior_manifest.rows
    )
    _require_exact_resume_prefix(
        prior_identity=prior_identity,
        current_identity=current_identity,
    )
    results = {
        row.document_id: row
        for row in (
            ExtractionCaseResult.model_validate_json(
                json.dumps(item, ensure_ascii=False, separators=(",", ":"))
            )
            for item in load_jsonl(prior_results_path, len(prior_manifest.rows))
        )
    }
    expected_ids = {row.document_id for row in prior_manifest.rows}
    if set(results) != expected_ids or len(results) != len(prior_manifest.rows):
        raise ValueError("resume results do not cover the prior selection exactly once")
    return run_root, results, prompt_changed, contract_matched


def _config_inputs(
    config: ExtractionConfig, project_root: Path
) -> tuple[
    list[dict[str, Any]],
    list[dict[str, Any]],
    list[dict[str, Any]],
    str,
    str,
    str | None,
    str | None,
    str | None,
]:
    source_path = resolve_input(project_root, config.inputs.source_corpus.path)
    feature_path = resolve_input(project_root, config.inputs.document_features.path)
    anchor_path = resolve_input(project_root, config.inputs.anchors.path)
    compiler_prompt_path = resolve_input(project_root, config.prompts.compiler.path)
    critic_prompt_path = resolve_input(project_root, config.prompts.critic.path)
    required_files = [
        (source_path, config.inputs.source_corpus.sha256),
        (feature_path, config.inputs.document_features.sha256),
        (anchor_path, config.inputs.anchors.sha256),
        (compiler_prompt_path, config.prompts.compiler.sha256),
        (critic_prompt_path, config.prompts.critic.sha256),
    ]
    if config.customs_program_registry is not None:
        required_files.append(
            (
                resolve_input(project_root, config.customs_program_registry.path),
                config.customs_program_registry.sha256,
            )
        )
    staged_prompt_paths: list[Path | None] = []
    for configured in (
        config.prompts.compiler_repair,
        config.prompts.critic_audit,
        config.prompts.critic_plan,
    ):
        if configured is None:
            staged_prompt_paths.append(None)
            continue
        path = resolve_input(project_root, configured.path)
        required_files.append((path, configured.sha256))
        staged_prompt_paths.append(path)
    for path, expected in required_files:
        verify_file(path, expected)
    return (
        load_jsonl(source_path, config.inputs.source_corpus.records),
        load_jsonl(feature_path, config.inputs.document_features.records),
        load_jsonl(anchor_path),
        compiler_prompt_path.read_text(encoding="utf-8"),
        critic_prompt_path.read_text(encoding="utf-8"),
        (
            staged_prompt_paths[0].read_text(encoding="utf-8")
            if staged_prompt_paths[0] is not None
            else None
        ),
        (
            staged_prompt_paths[1].read_text(encoding="utf-8")
            if staged_prompt_paths[1] is not None
            else None
        ),
        (
            staged_prompt_paths[2].read_text(encoding="utf-8")
            if staged_prompt_paths[2] is not None
            else None
        ),
    )


def _mark_host_rejected(stage: AgentStageArtifact, error: Exception) -> AgentStageArtifact:
    return stage.model_copy(
        update={
            "status": "host_rejected",
            "error_type": type(error).__name__,
            "error_message": str(error),
        }
    )


def _apply_validated_critic_review(
    *,
    review: CriticAgentOutput,
    raw: str,
    drafts: Sequence[SpanDraft],
    source_target: Mapping[str, Any],
    assessment: CarrierAssessment,
    semantic_only_target_facts: Sequence[SemanticOnlyTargetFact],
    coherence_constraints: Sequence[CoherenceConstraint] = (),
    review_candidates: Sequence[Mapping[str, Any]] | None = None,
) -> tuple[
    tuple[SpanDraft, ...],
    tuple[SemanticOnlyTargetFact, ...],
    tuple[CoherenceConstraint, ...],
]:
    if review.verdict != "revise":
        raise ValueError("critic transaction preview requires a revise decision")
    candidate_rows = (
        tuple(review_candidates)
        if review_candidates is not None
        else (
            _critic_review_candidates(
                raw=raw,
                drafts=drafts,
                source_target=source_target,
                semantic_only_target_facts=semantic_only_target_facts,
                coherence_constraints=coherence_constraints,
            )
        )
    )
    revised_constraints = materialize_coherence_decisions(
        candidates=candidate_rows,
        constraints=coherence_constraints,
        decisions=review.coherence_decisions,
    )
    review_semantic_only = materialize_semantic_only_target_facts(
        proposals=review.semantic_only_target_facts,
        source_target=source_target,
        provenance="critic_audited_unprinted",
    )
    revised = normalize_source_boundaries(
        raw=raw,
        drafts=normalize_deterministic_draft_semantics(
            raw=raw,
            drafts=normalize_structured_row_locality(
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
    validate_draft_source_alignment(raw=raw, drafts=revised)
    validate_carrier_assessment(
        assessment=assessment,
        expected=source_carrier(source_target),
        raw=raw,
        anchor_drafts_value=revised,
    )
    validate_binding_realizations(raw=raw, drafts=revised, source_target=source_target)
    effective_semantic_only = _effective_semantic_only_target_facts(
        facts=_merge_semantic_only_target_facts(
            semantic_only_target_facts,
            review_semantic_only,
        ),
        drafts=revised,
    )
    revised_constraints = reconcile_coherence_constraints_after_binding_revision(
        raw=raw,
        bindings=revised,
        source_target=source_target,
        constraints=revised_constraints,
    )
    with suppress(CoherenceReviewRequired):
        validate_coherence_contracts(
            raw=raw,
            bindings=revised,
            source_target=source_target,
            constraints=revised_constraints,
            require_complete=False,
        )
    if (
        binding_contract_signature(revised) == binding_contract_signature(drafts)
        and tuple(effective_semantic_only) == tuple(semantic_only_target_facts)
        and revised_constraints == tuple(coherence_constraints)
    ):
        raise ValueError(
            "critic transaction is a functional no-op after complete host normalization; "
            "do not repeat the finding or patch unless the rendering contract actually changes"
        )
    return revised, effective_semantic_only, revised_constraints


def _preview_critic_review(
    review: CriticAgentOutput,
    *,
    raw: str,
    drafts: Sequence[SpanDraft],
    source_target: Mapping[str, Any],
    assessment: CarrierAssessment,
    semantic_only_target_facts: Sequence[SemanticOnlyTargetFact],
    coherence_constraints: Sequence[CoherenceConstraint] = (),
) -> None:
    revised, _semantic_only, _constraints = _apply_validated_critic_review(
        review=review,
        raw=raw,
        drafts=drafts,
        source_target=source_target,
        assessment=assessment,
        semantic_only_target_facts=semantic_only_target_facts,
        coherence_constraints=coherence_constraints,
    )
    remaining = uncovered_risks(raw, all_risk_candidates(raw, (), source_target), revised)
    if remaining:
        details = "; ".join(f"{row.risk_id} {row.line_id}={row.source_text!r}" for row in remaining)
        raise ValueError(
            "critic transaction leaves deterministic risk candidates unowned: " + details
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
    assert prior_rejection.error_message is not None
    return "Host rejected the compiler output: " + prior_rejection.error_message


def _materialize_compiler_drafts(
    *,
    raw: str,
    source_target: Mapping[str, Any],
    anchors: Sequence[SpanDraft],
    proposed: Sequence[SpanDraft],
    anchor_overrides: Sequence[AnchorOverride],
    semantic_only_target_paths: Sequence[str],
) -> tuple[SpanDraft, ...]:
    """Apply one compiler transaction through the canonical host normalization path."""

    materialized = normalize_source_boundaries(
        raw=raw,
        drafts=normalize_deterministic_draft_semantics(
            raw=raw,
            drafts=normalize_structured_row_locality(
                raw=raw,
                drafts=normalize_target_cobindings(
                    drafts=apply_anchor_overrides(
                        raw=raw,
                        source_target=source_target,
                        anchors=anchors,
                        proposed=proposed,
                        overrides=anchor_overrides,
                        semantic_only_target_paths=semantic_only_target_paths,
                    ),
                    source_target=source_target,
                ),
                source_target=source_target,
            ),
            source_target=source_target,
        ),
    )
    validate_draft_source_alignment(raw=raw, drafts=materialized)
    return materialized


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
    diagnostic_body_scopes: dict[str, str] = {}

    def add(scope: str, error: Exception) -> None:
        body = str(error)
        first_scope = diagnostic_body_scopes.setdefault(body, scope)
        diagnostic = (
            f"{scope}: {body}" if first_scope == scope else f"{scope}: same defect as {first_scope}"
        )
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
        materialized: tuple[SpanDraft, ...] | None = None
        if declared_semantic_only is not None:
            try:
                materialized = _materialize_compiler_drafts(
                    raw=raw,
                    source_target=source_target,
                    anchors=anchors,
                    proposed=proposed,
                    anchor_overrides=output.anchor_overrides,
                    semantic_only_target_paths=tuple(
                        fact.target_path for fact in declared_semantic_only
                    ),
                )
            except Exception as error:
                add("anchor replacement integration", error)
        if materialized is not None:
            try:
                validate_mutable_token_boundaries(raw=raw, drafts=materialized)
            except Exception as error:
                add("binding token boundaries", error)

        proposal_ids = {draft.draft_id for draft in proposed}
        diagnostic_proposals = tuple(
            draft
            for draft in normalize_competing_auxiliary_outliers((*anchors, *proposed))
            if draft.draft_id in proposal_ids
        )
        proposed_by_key: dict[str, list[SpanDraft]] = {}
        for draft in diagnostic_proposals:
            proposed_by_key.setdefault(draft.logical_key, []).append(draft)
        normalized_proposed: list[SpanDraft] = []
        container_number_context = tuple(
            draft
            for draft in (*anchors, *diagnostic_proposals)
            if any(path.endswith(".containerNumber") for path in draft.target_paths)
        )
        for logical_key, logical_drafts in proposed_by_key.items():
            normalized_group: Sequence[SpanDraft] = logical_drafts
            try:
                normalized_group = reconcile_draft_overlaps(
                    raw=raw,
                    drafts=normalized_group,
                    source_target=source_target,
                )
            except Exception as error:
                add(f"binding overlap normalization for {logical_key}", error)
                normalized_group = logical_drafts
            if logical_drafts[0].value_kind == "equipment":
                original_spans = {(draft.char_start, draft.char_end) for draft in logical_drafts}
                context = {
                    draft.draft_id: draft for draft in (*container_number_context, *logical_drafts)
                }
                try:
                    normalized_group = tuple(
                        draft
                        for draft in normalize_structured_row_locality(
                            raw=raw,
                            drafts=tuple(context.values()),
                            source_target=source_target,
                        )
                        if (draft.char_start, draft.char_end) in original_spans
                    )
                except Exception as error:
                    add(f"binding locality normalization for {logical_key}", error)
                    normalized_group = logical_drafts
            try:
                normalized_group = normalize_source_boundaries(
                    raw=raw,
                    drafts=normalize_deterministic_draft_semantics(
                        raw=raw,
                        drafts=normalized_group,
                        source_target=source_target,
                    ),
                )
            except Exception as error:
                add(f"binding semantic normalization for {logical_key}", error)
                normalized_group = logical_drafts
            normalized_proposed.extend(normalized_group)
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

        try:
            validate_repeated_binding_fact_topology(
                drafts=normalized_proposed,
                source_target=source_target,
                raw=raw,
            )
        except Exception as error:
            add("binding target topology", error)

        declared_paths = {fact.target_path for fact in output.semantic_only_target_facts}
        proposed_paths = {
            path
            for draft in diagnostic_proposals
            for path in (*draft.target_paths, *draft.dependency_paths)
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


def _overlapping_literal_offsets(value: str, surface: str) -> tuple[int, ...]:
    if not surface:
        raise ValueError("literal occurrence surface must not be empty")
    offsets: list[int] = []
    cursor = 0
    while True:
        found = value.find(surface, cursor)
        if found < 0:
            return tuple(offsets)
        offsets.append(found)
        cursor = found + 1


def _draft_inventory(
    raw: str,
    drafts: Sequence[SpanDraft],
    source_target: Mapping[str, Any],
    *,
    rejected_candidate: bool = False,
) -> list[dict[str, Any]]:
    from .host import line_range_for_chars, line_spans

    validate_draft_source_alignment(raw=raw, drafts=drafts)
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
        # Match the compiler occurrence resolver exactly: literal occurrences can overlap. For
        # example, ``C.C.`` occurs twice in ``C.C.C.`` and either span can be an edit handle.
        offsets = _overlapping_literal_offsets(region, row.source_text)
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
        target_path_error: str | None = None
        try:
            target_relationship = target_path_relationship(source_target, first.target_paths)
            fact_components = target_fact_components(source_target, first.target_paths)
        except ValueError as error:
            if not rejected_candidate:
                raise
            target_relationship = "invalid_target_path"
            fact_components = tuple((path,) for path in first.target_paths)
            target_path_error = str(error)
        inventory_row = {
            "sourceBindingIds": tuple(row.draft_id for row in rows),
            "logicalKey": first.logical_key,
            "renderMode": first.render_mode,
            "valueKind": first.value_kind,
            "groupKind": first.group_kind,
            "groupKey": first.group_key,
            "targetPaths": first.target_paths,
            "targetRelationship": target_relationship,
            "independentTargetFactComponents": fact_components,
            "derivation": first.derivation,
            "dependencyPaths": first.dependency_paths,
            "dependencyBindings": first.dependency_bindings,
            "occurrences": tuple(occurrence_summary(row) for row in rows),
        }
        if target_path_error is not None:
            inventory_row["targetPathValidationError"] = target_path_error
        output.append(inventory_row)
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


def _critic_addressable_target_paths(
    *, source_target: Mapping[str, Any], drafts: Sequence[SpanDraft]
) -> tuple[str, ...]:
    """Return the semantic target vocabulary needed by an independent critic.

    Object and nested-list summaries are compiler construction context, not printable facts. The
    critic retains every scalar leaf, every path used by the current template, and the two
    collection roots supported by count derivations. This removes no value the critic can bind or
    use as a dependency while avoiding repeated structural rows for every indexed object.
    """

    patch = source_target.get("documentPatch")
    if not isinstance(patch, Mapping):
        raise ValueError("source target lacks documentPatch")
    scalar_paths: set[str] = set()

    def visit(value: Any, path: str) -> None:
        if isinstance(value, Mapping):
            for key, child in value.items():
                visit(child, f"{path}.{key}")
        elif isinstance(value, list):
            for index, child in enumerate(value):
                visit(child, f"{path}[{index}]")
        else:
            scalar_paths.add(path)

    visit(patch, "documentPatch")
    all_paths = _addressable_target_paths(source_target)
    retained = scalar_paths | {
        path for draft in drafts for path in (*draft.target_paths, *draft.dependency_paths)
    }
    retained.update(
        path
        for path in ("documentPatch.containers", "documentPatch.cargoPackages")
        if path in all_paths
    )
    return tuple(path for path in all_paths if path in retained)


def _compiler_occurrence_candidates(
    *,
    raw: str,
    source_target: Mapping[str, Any],
    anchors: Sequence[SpanDraft],
    risks: Sequence[Any],
    authorized_anchor_overlap_ids: frozenset[str] | None = None,
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
        for start in _overlapping_literal_offsets(raw, surface):
            end = start + len(surface)
            if not is_token_bounded_surface_span(raw, start, end):
                continue
            candidates.add((start, end, surface))

    output: list[dict[str, Any]] = []
    for index, (start, end, surface) in enumerate(sorted(candidates), start=1):
        if authorized_anchor_overlap_ids is not None and any(
            start < anchor.char_end
            and anchor.char_start < end
            and (start, end) != (anchor.char_start, anchor.char_end)
            and anchor.draft_id not in authorized_anchor_overlap_ids
            for anchor in anchors
        ):
            # A partial scalar match inside a retained anchor is not an independent source
            # occurrence. Keeping it in the provider vocabulary invited values such as package
            # quantity ``96`` to be selected from inside container number ``TGBU9219649``.
            continue
        start_line = next(line for line in lines if line.char_start <= start <= line.char_end)
        end_line = next(line for line in lines if line.char_start <= end - 1 <= line.char_end)
        range_text = raw[start_line.char_start : end_line.char_end]
        relative_start = start - start_line.char_start
        occurrence_index = sum(
            offset < relative_start for offset in _overlapping_literal_offsets(range_text, surface)
        )
        output.append(
            {
                # This hot loop's percent formatter benchmarks materially faster than an f-string.
                "occurrenceId": "cand_%s_%05d"  # noqa: UP031
                % (start_line.line_id, index),
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
    semantic_review_ids = tuple(
        row.draft_id
        for row in anchors
        if target_path_relationship(source_target, row.target_paths) == "composite_target_surface"
    )
    shared_equality_ids = tuple(
        row.draft_id
        for row in anchors
        if target_path_relationship(source_target, row.target_paths) == "shared_value_equality"
    )
    cross_fact_equality_ids = tuple(
        row.draft_id
        for row in anchors
        if target_path_relationship(source_target, row.target_paths) == "shared_value_equality"
        and len(target_fact_components(source_target, row.target_paths)) > 1
    )
    initial_review_ids = tuple(dict.fromkeys((*semantic_review_ids, *cross_fact_equality_ids)))
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
        "anchorBindingsRequiringSemanticReview": list(semantic_review_ids),
        "anchorBindingsWithSharedEqualityConstraint": list(shared_equality_ids),
        "anchorBindingsWithCrossFactEqualityAmbiguity": list(cross_fact_equality_ids),
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
    if agent_contract_protocol in {
        "reference_compact_v3",
        "relational_reference_compact_v9",
        "hybrid_reference_partitioned_v10",
        "staged_local_v4",
        "faceted_staged_local_v5",
        "partitioned_staged_local_v6",
        "candidate_first_staged_local_v7",
        "candidate_first_staged_local_v8",
    }:
        if agent_contract_protocol == "candidate_first_staged_local_v8":
            payload["anchorBindingsAuthorizedForInitialReview"] = initial_review_ids
        occurrence_candidates = _compiler_occurrence_candidates(
            raw=raw,
            source_target=source_target,
            anchors=anchors,
            risks=risks,
            authorized_anchor_overlap_ids=(
                frozenset(initial_review_ids)
                if agent_contract_protocol == "candidate_first_staged_local_v8"
                else None
            ),
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
            raw,
            prior_drafts,
            source_target,
            rejected_candidate=prior_error is not None,
        )
        if agent_contract_protocol not in {
            "staged_local_v4",
            "faceted_staged_local_v5",
            "partitioned_staged_local_v6",
            "candidate_first_staged_local_v7",
            "candidate_first_staged_local_v8",
        }:
            payload["previousCandidateMaskedTemplate"] = masked_source(raw, prior_drafts)
    if (
        agent_contract_protocol
        in {
            "partitioned_staged_local_v6",
            "candidate_first_staged_local_v7",
            "candidate_first_staged_local_v8",
        }
        and prior_output is None
        and prior_error is None
    ):
        return _compact_initial_compiler_payload(payload)
    return payload


def _compact_initial_compiler_payload(payload: Mapping[str, Any]) -> dict[str, Any]:
    """Project the immutable first-pass compiler context into indexed record tables.

    The source label remains nested because it is the easiest semantic view for the compiler. The
    duplicated full target paths, anchor object keys, occurrence keys, and risk offsets do not.
    Local repair requests deliberately retain their verbose host slice and use a different prompt.
    """

    target_paths = payload.get("allowedTargetPaths")
    anchor_bindings = payload.get("anchorBindings")
    risks = payload.get("riskCandidates")
    co_bindings = payload.get("requiredTargetCoBindings")
    if not all(
        isinstance(value, (tuple, list))
        for value in (
            target_paths,
            anchor_bindings,
            risks,
            co_bindings,
        )
    ):
        raise ValueError("initial compiler payload lacks compactable inventories")
    target_paths = cast(Sequence[str], target_paths)
    if len(set(target_paths)) != len(target_paths):
        raise ValueError("initial compiler target paths are not unique")
    path_indexes = {path: index for index, path in enumerate(target_paths)}

    anchor_occurrence_rows: list[tuple[Any, ...]] = []
    anchor_binding_rows: list[tuple[Any, ...]] = []
    for binding_index, raw_binding in enumerate(cast(Sequence[Any], anchor_bindings)):
        if not isinstance(raw_binding, Mapping):
            raise ValueError("initial compiler anchor binding is not an object")
        occurrence_indexes: list[int] = []
        occurrences = raw_binding.get("occurrences")
        if not isinstance(occurrences, (tuple, list)):
            raise ValueError("initial compiler anchor occurrences are not a sequence")
        for occurrence in occurrences:
            if not isinstance(occurrence, Mapping):
                raise ValueError("initial compiler anchor occurrence is not an object")
            occurrence_indexes.append(len(anchor_occurrence_rows))
            anchor_occurrence_rows.append(
                (
                    occurrence["anchorBindingId"],
                    occurrence["lineStart"],
                    occurrence["lineEnd"],
                    occurrence["sourceText"],
                )
            )
        try:
            target_path_indexes = tuple(
                path_indexes[path] for path in cast(Sequence[str], raw_binding["targetPaths"])
            )
            component_indexes = tuple(
                tuple(path_indexes[path] for path in component)
                for component in cast(
                    Sequence[Sequence[str]], raw_binding["independentTargetFactComponents"]
                )
            )
        except KeyError as error:
            raise ValueError("anchor binding references an unknown target path") from error
        anchor_binding_rows.append(
            (
                binding_index,
                raw_binding["logicalKey"],
                raw_binding["renderMode"],
                raw_binding["valueKind"],
                raw_binding["groupKind"],
                raw_binding["groupKey"],
                target_path_indexes,
                raw_binding["targetRelationship"],
                component_indexes,
                raw_binding["renderPolicy"],
                tuple(occurrence_indexes),
            )
        )

    co_binding_rows: list[tuple[Any, ...]] = []
    for index, row in enumerate(cast(Sequence[Any], co_bindings)):
        if not isinstance(row, Mapping):
            raise ValueError("initial compiler co-binding is not an object")
        try:
            indexes = tuple(path_indexes[path] for path in cast(Sequence[str], row["targetPaths"]))
        except KeyError as error:
            raise ValueError("co-binding references an unknown target path") from error
        co_binding_rows.append((index, row["relationship"], indexes))

    risk_rows: list[tuple[Any, ...]] = []
    for row in cast(Sequence[Any], risks):
        if not isinstance(row, Mapping):
            raise ValueError("initial compiler risk candidate is not an object")
        risk_rows.append(
            (
                row["risk_id"],
                row["kind"],
                row["line_id"],
                row["source_text"],
                row["currentlyOwnedByAcceptedAnchor"],
            )
        )

    projected = {
        key: value
        for key, value in payload.items()
        if key
        not in {
            "allowedTargetPaths",
            "anchorBindings",
            "requiredTargetCoBindings",
            "riskCandidates",
        }
    }
    projected.update(
        {
            "compilerCompactContract": {
                "schemaVersion": 6,
                "allExplicitIndexesAreZeroBased": True,
            },
            "targetPathTable": {
                "columns": ("targetPathIndex", "targetPath"),
                "rows": tuple((index, path) for index, path in enumerate(target_paths)),
            },
            "coBindingTable": {
                "columns": ("coBindingIndex", "relationship", "targetPathIndexes"),
                "rows": tuple(co_binding_rows),
            },
            "anchorBindingTable": {
                "columns": (
                    "anchorIndex",
                    "logicalKey",
                    "renderMode",
                    "valueKind",
                    "groupKind",
                    "groupKey",
                    "targetPathIndexes",
                    "targetRelationship",
                    "independentTargetFactComponents",
                    "renderPolicy",
                    "occurrenceIndexes",
                ),
                "rows": tuple(anchor_binding_rows),
            },
            "anchorOccurrenceTable": {
                "columns": ("anchorBindingId", "lineStart", "lineEnd", "sourceText"),
                "rows": tuple(anchor_occurrence_rows),
            },
            "riskCandidateTable": {
                "columns": (
                    "riskId",
                    "kind",
                    "lineId",
                    "sourceText",
                    "currentlyOwnedByAcceptedAnchor",
                ),
                "rows": tuple(risk_rows),
            },
        }
    )
    return projected


def _prepare_anchor_rows(
    *,
    raw: str,
    document_id: str,
    anchor_rows: Sequence[Mapping[str, Any]],
    source_target: Mapping[str, Any],
    agent_contract_protocol: AgentContractProtocol,
) -> tuple[tuple[Mapping[str, Any], ...], dict[str, Any] | None]:
    rows = tuple(anchor_rows)
    if agent_contract_protocol not in _RELATIONAL_ANCHOR_PROTOCOLS:
        return rows, None
    normalized, report = normalize_relational_anchor_locality(
        raw=raw,
        document_id=document_id,
        anchors=rows,
        source_target=source_target,
    )
    return normalized, asdict(report)


def _preflight_summary(
    *,
    config: ExtractionConfig,
    manifest: SelectionManifest,
    source_rows: Sequence[Mapping[str, Any]],
    feature_rows: Sequence[Mapping[str, Any]],
    anchor_rows: Sequence[Mapping[str, Any]],
    compiler_prompt: str,
    critic_prompt: str,
    compiler_repair_prompt: str | None = None,
    critic_audit_prompt: str | None = None,
    critic_plan_prompt: str | None = None,
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
        integrity_issues = source_template_integrity_issues(raw, source_target)
        carrier = source_carrier(source_target)
        prepared_anchor_rows, locality_report = _prepare_anchor_rows(
            raw=raw,
            document_id=document_id,
            anchor_rows=anchors_by_id.get(document_id, ()),
            source_target=source_target,
            agent_contract_protocol=config.workflow.agent_contract_protocol,
        )
        anchors = normalize_target_cobindings(
            drafts=anchor_drafts(
                raw=raw,
                document_id=document_id,
                anchors=prepared_anchor_rows,
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
        risks = all_risk_candidates(raw, (), source_target)
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
                "sourceTemplateIntegrityIssues": integrity_issues,
                "providerEligible": not integrity_issues,
                "relationalAnchorLocality": locality_report,
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
        "sourceTemplateIntegrityReviewRequired": sum(not row["providerEligible"] for row in cases),
        "providerEligibleDocuments": sum(row["providerEligible"] for row in cases),
        "compilerPromptSha256": sha256_bytes(compiler_prompt.encode("utf-8")),
        "criticPromptSha256": sha256_bytes(critic_prompt.encode("utf-8")),
        "stagedPromptSha256s": {
            name: sha256_bytes(prompt.encode("utf-8")) if prompt is not None else None
            for name, prompt in (
                ("compilerRepair", compiler_repair_prompt),
                ("criticAudit", critic_audit_prompt),
                ("criticPlan", critic_plan_prompt),
            )
        },
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
        "totalRelocatedAnchors": sum(
            len(row["relationalAnchorLocality"]["relocated_anchor_ids"])
            for row in cases
            if row["relationalAnchorLocality"] is not None
        ),
        "totalExpandedAnchors": sum(
            len(row["relationalAnchorLocality"]["expanded_anchor_ids"])
            for row in cases
            if row["relationalAnchorLocality"] is not None
        ),
        "totalRecoveredTopologyAnchors": sum(
            len(row["relationalAnchorLocality"]["recovered_topology_anchor_ids"])
            for row in cases
            if row["relationalAnchorLocality"] is not None
        ),
        "totalSuppressedAnchors": sum(
            len(row["relationalAnchorLocality"]["suppressed_anchor_ids"])
            for row in cases
            if row["relationalAnchorLocality"] is not None
        ),
        "totalDeduplicatedAnchors": sum(
            len(row["relationalAnchorLocality"]["deduplicated_anchor_ids"])
            for row in cases
            if row["relationalAnchorLocality"] is not None
        ),
        "maxCompilerRequestBytes": max(row["compilerRequestBytes"] for row in cases),
        "providerLaunchAuthorized": config.workflow.provider_launch_authorized,
        "requireNoRejectedDocuments": config.workflow.require_no_rejected_documents,
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
        _compiler_prompt,
        _critic_prompt,
        *_staged_prompts,
    ) = _config_inputs(config, project_root)
    return _corpus_carrier_resolution_report(
        config=config,
        source_rows=source_rows,
        feature_rows=feature_rows,
        anchor_rows=anchor_rows,
    )


def _critic_review_candidates(
    *,
    raw: str,
    drafts: Sequence[SpanDraft],
    source_target: Mapping[str, Any] | None = None,
    semantic_only_target_facts: Sequence[SemanticOnlyTargetFact] = (),
    coherence_constraints: Sequence[CoherenceConstraint] = (),
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

    def current_and_previous_line(char_start: int) -> str:
        current_start = raw.rfind("\n", 0, char_start) + 1
        current_end = raw.find("\n", char_start)
        if current_end < 0:
            current_end = len(raw)
        previous_end = max(0, current_start - 1)
        previous_start = raw.rfind("\n", 0, previous_end) + 1
        return raw[previous_start:current_end]

    def current_line(char_start: int) -> str:
        start = raw.rfind("\n", 0, char_start) + 1
        end = raw.find("\n", char_start)
        return raw[start : len(raw) if end < 0 else end]

    def conditional_negotiability_line(value: str) -> bool:
        return (
            re.search(
                r"(?i)\bIF\s+THIS\s+IS\s+(?:A\s+)?(?:NON[- ]?)?NEGOTIABLE\b",
                value,
            )
            is not None
        )

    def original_bill_form_title_line(value: str) -> bool:
        return (
            re.fullmatch(
                r"(?i)\s*ORIGINAL\s+BILL\s+OF\s+LADING(?:\s*\([^)]*\))?\s*",
                value,
            )
            is not None
        )

    grouped: dict[str, list[SpanDraft]] = {}
    for draft in drafts:
        grouped.setdefault(draft.logical_key, []).append(draft)
    candidates: list[dict[str, Any]] = []

    def binding_candidate(
        *,
        kind: str,
        logical_key: str,
        rows: Sequence[SpanDraft],
        suggested_derivations: Sequence[str],
        relationship: str,
    ) -> dict[str, Any]:
        return {
            "kind": kind,
            "logicalKeys": (logical_key,),
            "lineIds": tuple(sorted({line_id(row.char_start) for row in rows})),
            "sourceTexts": tuple(sorted({row.source_text for row in rows})),
            "details": {
                "suggestedDerivations": tuple(suggested_derivations),
                "relationshipToVerify": relationship,
            },
        }

    # Existing bindings can hide another exact rendering of an already mutable value. Surface
    # every uncovered repeat; the critic decides whether its context is data or generic grammar.
    surface_owners: dict[str, set[str]] = {}
    surface_owner_rows: dict[tuple[str, str], list[SpanDraft]] = {}
    for logical_key, rows in sorted(grouped.items()):
        for source_text in sorted({row.source_text for row in rows}):
            matching_rows = [row for row in rows if row.source_text == source_text]
            surface_owners.setdefault(source_text, set()).add(logical_key)
            surface_owner_rows[(logical_key, source_text)] = matching_rows
    for source_text, logical_keys in sorted(surface_owners.items()):
        normalized_source = normalized(source_text)
        owner_rows = [
            row
            for logical_key in logical_keys
            for row in surface_owner_rows[(logical_key, source_text)]
        ]
        typed_integer = any(row.value_kind == "integer" for row in owner_rows) and (
            source_text.strip().isdigit() and bool(normalized_source)
        )
        compact_equipment_code = (
            any(
                row.group_kind == "equipment" and row.value_kind == "equipment"
                for row in owner_rows
            )
            and 3 <= len(normalized_source) <= 12
            and any(character.isalpha() for character in normalized_source)
            and any(character.isdigit() for character in normalized_source)
        )
        # One- and two-digit values are not self-identifying repeat candidates. They routinely
        # recur as clause, page, and row numbers, while their genuine package/container meaning is
        # already guarded by row-local host normalization. Longer typed quantities retain the
        # repeat lead because their accidental-collision rate is materially lower.
        typed_quantity_candidate = typed_integer and len(normalized_source) >= 3
        if len(normalized_source) < 5 and not (typed_quantity_candidate or compact_equipment_code):
            continue
        negotiability_owner = any(
            path.endswith(".negotiability") for row in owner_rows for path in row.target_paths
        )
        occurrences: list[tuple[str, str]] = []
        for start in _overlapping_literal_offsets(raw, source_text):
            end = start + len(source_text)
            if (source_text[0].isalnum() and start > 0 and raw[start - 1].isalnum()) or (
                source_text[-1].isalnum() and end < len(raw) and raw[end].isalnum()
            ):
                continue
            if any(start < draft.char_end and draft.char_start < end for draft in drafts):
                continue
            if negotiability_owner and conditional_negotiability_line(current_line(start)):
                continue
            if normalized_source == "original" and original_bill_form_title_line(
                current_line(start)
            ):
                continue
            occurrences.append(
                (
                    line_id(start),
                    raw[max(0, start - 48) : end + 48],
                )
            )
        if not occurrences:
            continue
        candidates.append(
            {
                "kind": "unowned_exact_repeat",
                "logicalKeys": tuple(sorted(logical_keys)),
                "lineIds": tuple(dict.fromkeys(line for line, _context in occurrences)),
                "sourceTexts": (source_text,),
                "context": tuple(dict.fromkeys(context for _line, context in occurrences)),
                "details": {
                    "possibleOwnerContexts": tuple(
                        {
                            "logicalKey": logical_key,
                            "groupKind": rows[0].group_kind,
                            "groupKey": rows[0].group_key,
                            "valueKind": rows[0].value_kind,
                            "targetPaths": rows[0].target_paths,
                            "ownedLineIds": tuple(
                                sorted({line_id(row.char_start) for row in rows})
                            ),
                        }
                        for logical_key in sorted(logical_keys)
                        for rows in (surface_owner_rows[(logical_key, source_text)],)
                    ),
                    "relationshipToVerify": (
                        "Decide whether each equal surface is a genuine repeat of one listed "
                        "owner or an independently mutable semantic role. Text equality alone is "
                        "not an ownership contract; use row context to select an owner."
                    ),
                },
            }
        )

    # Surface exact repeated literal phrases that have no current owner.  This is an audit lead,
    # not an automatic binding: repeated captions are expected and the critic can mark them valid.
    # The explicit ledger prevents a long first response from overlooking a later role-specific
    # value (for example, the same city printed separately for consignee and notify party).
    source_lines = line_spans(raw)
    repeated_literal_rows: dict[str, list[tuple[str, str, str]]] = defaultdict(list)
    for source_line in source_lines:
        owned_intervals = sorted(
            (
                max(source_line.char_start, draft.char_start),
                min(source_line.char_end, draft.char_end),
            )
            for draft in drafts
            if source_line.char_start < draft.char_end and draft.char_start < source_line.char_end
        )
        cursor = source_line.char_start
        unowned_intervals: list[tuple[int, int]] = []
        for start, end in owned_intervals:
            if cursor < start:
                unowned_intervals.append((cursor, start))
            cursor = max(cursor, end)
        if cursor < source_line.char_end:
            unowned_intervals.append((cursor, source_line.char_end))
        for start, end in unowned_intervals:
            surface = raw[start:end].strip(" \t|/,:;-")
            normalized_surface = normalized(surface)
            if (
                len(normalized_surface) < 5
                or not any(character.isalpha() for character in surface)
                or len(re.findall(r"[A-Za-z0-9]+", surface)) < 2
            ):
                continue
            repeated_literal_rows.setdefault(normalized_surface, []).append(
                (source_line.line_id, surface, current_and_previous_line(start))
            )

    for normalized_surface, literal_rows in sorted(repeated_literal_rows.items()):
        literal_line_ids = tuple(dict.fromkeys(line for line, _surface, _context in literal_rows))
        if len(literal_line_ids) < 2:
            continue
        existing_exact_repeat = next(
            (
                candidate
                for candidate in candidates
                if candidate["kind"] == "unowned_exact_repeat"
                and any(
                    normalized(str(surface)) == normalized_surface
                    for surface in candidate["sourceTexts"]
                )
            ),
            None,
        )
        if existing_exact_repeat is not None:
            existing_exact_repeat["lineIds"] = tuple(
                dict.fromkeys((*existing_exact_repeat["lineIds"], *literal_line_ids))
            )
            existing_exact_repeat["sourceTexts"] = tuple(
                dict.fromkeys(
                    (
                        *existing_exact_repeat["sourceTexts"],
                        *(surface for _line, surface, _context in literal_rows),
                    )
                )
            )
            existing_exact_repeat["context"] = tuple(
                dict.fromkeys(
                    (
                        *existing_exact_repeat["context"],
                        *(context for _line, _surface, context in literal_rows),
                    )
                )
            )
            continue
        exact_repeat_covered_lines = {
            f"L{int(line_id_value[1:]) + offset:05d}"
            for candidate in candidates
            if candidate["kind"] == "unowned_exact_repeat"
            for line_id_value in candidate["lineIds"]
            for offset in range(
                max(str(surface).count("\n") for surface in candidate["sourceTexts"]) + 1
            )
        }
        if set(literal_line_ids) <= exact_repeat_covered_lines:
            continue
        normalized_owner_keys = tuple(
            sorted(
                {
                    logical_key
                    for logical_key, owned_surface in surface_owner_rows
                    if normalized(owned_surface) == normalized_surface
                }
            )
        )
        normalized_owner_contexts = tuple(
            {
                "logicalKey": logical_key,
                "groupKind": owner_rows[0].group_kind,
                "groupKey": owner_rows[0].group_key,
                "valueKind": owner_rows[0].value_kind,
                "targetPaths": owner_rows[0].target_paths,
                "ownedLineIds": tuple(sorted({line_id(owner.char_start) for owner in owner_rows})),
            }
            for logical_key in normalized_owner_keys
            for matching_surfaces in (
                tuple(
                    surface
                    for owner_key, surface in surface_owner_rows
                    if owner_key == logical_key and normalized(surface) == normalized_surface
                ),
            )
            for owner_rows in (
                tuple(
                    owner
                    for surface in matching_surfaces
                    for owner in surface_owner_rows[(logical_key, surface)]
                ),
            )
        )
        candidates.append(
            {
                "kind": "unowned_repeated_literal_surface",
                "logicalKeys": normalized_owner_keys,
                "lineIds": literal_line_ids,
                "sourceTexts": tuple(
                    dict.fromkeys(surface for _line, surface, _context in literal_rows)
                ),
                "context": tuple(
                    dict.fromkeys(context for _line, _surface, context in literal_rows)
                ),
                "details": {
                    "possibleNormalizedOwnerContexts": normalized_owner_contexts,
                    "relationshipToVerify": (
                        "Decide whether these normalized-equal unowned phrases are selected "
                        "role-specific data, a formatting projection of a supplied owner, or "
                        "repeated fixed form grammar."
                    ),
                },
            }
        )

    # Standalone copy/release/status values are selected document facts often omitted by generic
    # token risk detectors. Titles containing these words are intentionally excluded: the critic
    # still decides whether each standalone line is selected data or fixed form grammar.
    standalone_statuses: dict[str, list[tuple[str, str]]] = {}
    offset = 0
    for raw_line in raw.splitlines(keepends=True):
        line_text = raw_line.rstrip("\r\n")
        stripped = line_text.strip()
        relative_start = line_text.find(stripped) if stripped else 0
        start = offset + relative_start
        end = start + len(stripped)
        if (
            stripped
            and re.fullmatch(
                r"(?i)(?:ORIGINALS?|COPY|COPIES|DUPLICATE|NEGOTIABLE|NON[- ]NEGOTIABLE|"
                r"SURRENDERED|EXPRESS RELEASE|TELEX RELEASE)[.!]?",
                stripped,
            )
            is not None
            and not any(start < draft.char_end and draft.char_start < end for draft in drafts)
        ):
            standalone_statuses.setdefault(stripped, []).append((line_id(start), line_text))
        offset += len(raw_line)
    candidates.extend(
        {
            "kind": "unowned_standalone_document_status",
            "logicalKeys": (),
            "lineIds": tuple(dict.fromkeys(line for line, _context in occurrences)),
            "sourceTexts": (source_text,),
            "context": tuple(dict.fromkeys(context for _line, context in occurrences)),
            "details": {
                "relationshipToVerify": (
                    "Decide whether this standalone surface is a selected copy, negotiability, "
                    "or release status; a form title or unselected option remains static grammar."
                )
            },
        }
        for source_text, occurrences in sorted(standalone_statuses.items())
    )

    # Semantic-only declarations are easy to overlook when their structured scalar is a semantic
    # classification rather than a literal OCR substring. Retrieve narrow, domain-grounded source
    # evidence and ask the critic to decide whether it prints the selected state. These are review
    # leads, never automatic ownership decisions.
    semantic_status_patterns = (
        (
            re.compile(r"(?:^|\.)negotiability$", re.IGNORECASE),
            re.compile(
                r"(?i)(?:\bnon[- ]?negotiable\b|\bnegotiable\b|"
                r"\boriginal bills? of lading\b|\bone of which being accomplished\b|"
                r"\bthe other\(s\) to be void\b)"
            ),
            "negotiability or original-bill status",
        ),
        (
            re.compile(r"(?:copyType|copyStatus|documentCopy)$", re.IGNORECASE),
            re.compile(r"(?i)\b(?:original|copy|copies|duplicate)\b"),
            "selected document-copy status",
        ),
        (
            re.compile(r"(?:releaseType|releaseStatus)$", re.IGNORECASE),
            re.compile(r"(?i)\b(?:surrendered|telex release|express release|sea waybill)\b"),
            "selected cargo-release status",
        ),
    )
    semantic_lines = line_spans(raw)
    for semantic_only_index, fact in enumerate(semantic_only_target_facts):
        matches: list[tuple[str, str, str]] = []
        source_surface = str(fact.source_value) if isinstance(fact.source_value, (str, int)) else ""
        patterns: list[tuple[re.Pattern[str], str]] = []
        if len(normalized(source_surface)) >= 5:
            patterns.append((re.compile(re.escape(source_surface), re.IGNORECASE), "exact value"))
        patterns.extend(
            (pattern, description)
            for path_pattern, pattern, description in semantic_status_patterns
            if path_pattern.search(fact.target_path) is not None
        )
        for source_line in semantic_lines:
            for pattern, match_kind in patterns:
                match = pattern.search(source_line.text)
                if match is None:
                    continue
                if fact.target_path.endswith(".negotiability") and conditional_negotiability_line(
                    source_line.text
                ):
                    continue
                absolute_start = source_line.char_start + match.start()
                absolute_end = source_line.char_start + match.end()
                if any(
                    absolute_start < draft.char_end and draft.char_start < absolute_end
                    for draft in drafts
                ):
                    continue
                matches.append((source_line.line_id, source_line.text, match_kind))
                break
        if matches:
            candidates.append(
                {
                    "kind": "semantic_only_evidence_review",
                    "logicalKeys": (),
                    "lineIds": tuple(dict.fromkeys(line for line, _text, _kind in matches)),
                    "sourceTexts": tuple(dict.fromkeys(text for _line, text, _kind in matches)),
                    "context": tuple(
                        dict.fromkeys(
                            current_and_previous_line(source_line.char_start)
                            for source_line in semantic_lines
                            if source_line.line_id in {line for line, _text, _kind in matches}
                        )
                    ),
                    "details": {
                        "semanticOnlyIndex": semantic_only_index,
                        "targetPath": fact.target_path,
                        "sourceValue": fact.source_value,
                        "provenance": fact.provenance,
                        "currentReason": fact.rationale,
                        "evidenceMatchKinds": tuple(
                            dict.fromkeys(kind for _line, _text, kind in matches)
                        ),
                        "relationshipToVerify": (
                            "Decide whether the cited literal OCR unambiguously prints this "
                            "selected semantic state. Retain semantic-only only when it does not."
                        ),
                    },
                }
            )

    # Compact operational qualifiers, explicit identifier-type selections, and measurement units
    # are shipment data when printed beside selected values, but generic token heuristics often
    # rediscover them only after another edit changes masking. These remain optional semantic
    # leads: definitions and already owned occurrences are excluded, and the critic decides if a
    # surviving surface is selected data or fixed form vocabulary.
    domain_patterns = (
        (
            "operational_status",
            re.compile(r"(?i)(?<![A-Z0-9])(?:SHIPPED|LADEN)\s+ON\s+BOARD(?![A-Z0-9])"),
            (
                "Determine whether this phrase is a selected shipment status or an unselected "
                "form caption; only a selected status requires ownership."
            ),
        ),
        (
            "operational_qualifier",
            re.compile(r"(?i)(?<![A-Z0-9])SLAC\*(?![A-Z0-9])"),
            (
                "Determine whether this is selected cargo data or static definition/form "
                "grammar; only selected data requires ownership."
            ),
        ),
        (
            "identifier_type",
            re.compile(
                r"(?i)(?<![A-Z0-9])TYPE\s*:\s*(?:VAT|TAX|REGISTRATION)\s+"
                r"(?:NUMBER|ID)(?![A-Z0-9])"
            ),
            (
                "Determine whether the identifier type is a selected value or only section/form "
                "grammar; only a selected value requires ownership."
            ),
        ),
        (
            "measurement_unit",
            re.compile(r"(?i)(?<![A-Z0-9])(?:KGM|KGS?|LBS?|MTS?|CBM|M3)(?![A-Z0-9])"),
            (
                "Determine whether this unit belongs to a selected measurement or only a generic "
                "column title; only the selected unit requires ownership."
            ),
        ),
    )
    domain_surfaces: dict[tuple[str, str], list[tuple[str, str]]] = {}
    for semantic_hint, pattern, _relationship in domain_patterns:
        for match in pattern.finditer(raw):
            if semantic_hint == "operational_qualifier" and re.match(r"\s*=", raw[match.end() :]):
                continue
            if semantic_hint == "measurement_unit" and not any(
                character.isdigit() for character in current_and_previous_line(match.start())
            ):
                continue
            if any(
                match.start() < draft.char_end and draft.char_start < match.end()
                for draft in drafts
            ):
                continue
            domain_surfaces.setdefault((semantic_hint, match.group()), []).append(
                (line_id(match.start()), current_and_previous_line(match.start()))
            )
    candidates.extend(
        {
            "kind": "domain_vocabulary_surface_review",
            "logicalKeys": (),
            "lineIds": tuple(dict.fromkeys(line for line, _context in occurrences)),
            "sourceTexts": (source_text,),
            "context": tuple(dict.fromkeys(context for _line, context in occurrences)),
            "details": {
                "semanticHint": semantic_hint,
                "relationshipToVerify": next(
                    relationship
                    for hint, _pattern, relationship in domain_patterns
                    if hint == semantic_hint
                ),
            },
        }
        for (semantic_hint, source_text), occurrences in sorted(domain_surfaces.items())
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
    for scope, scope_logical_keys in sorted(scope_groups.items()):
        for left_index, left_key in enumerate(sorted(scope_logical_keys)):
            left_values = {row.source_text for row in grouped[left_key]}
            for right_key in sorted(scope_logical_keys)[left_index + 1 :]:
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
        if first.render_mode == "carrier_static":
            candidates.append(
                {
                    "kind": "carrier_static_contract_review",
                    "logicalKeys": (logical_key,),
                    "lineIds": tuple(sorted({line_id(row.char_start) for row in rows})),
                    "sourceTexts": tuple(surfaces),
                    "targetPaths": first.target_paths,
                    "details": {
                        "occurrenceContexts": tuple(
                            dict.fromkeys(current_and_previous_line(row.char_start) for row in rows)
                        ),
                        "relationshipToVerify": (
                            "Verify that this is reusable public identity of the pinned carrier, "
                            "not a shipment-appointed local issuing or signing agent. Repeated "
                            "fixed SIGNED blocks are carrier relationships unless adjacent text "
                            "explicitly proves appointment."
                        ),
                    },
                }
            )
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
            continue

        # These are deliberately high-recall audit leads, not host-side semantic rewrites. The
        # complete-state auditor must either confirm the current contract or report the missing
        # derivation in its first pass. This prevents calculated relationships from being found
        # only after an unrelated revision changes the masking topology.
        key_tokens = set(re.findall(r"[a-z0-9]+", logical_key.casefold()))
        stripped_surfaces = {surface.strip() for surface in surfaces}
        if (
            first.render_mode == "deterministic_auxiliary"
            and not first.target_paths
            and {"country", "code"} <= key_tokens
            and stripped_surfaces
            and all(re.fullmatch(r"[A-Za-z]{2}", surface) for surface in stripped_surfaces)
        ):
            possible_dependency_keys = tuple(
                key
                for key, peer_rows in sorted(grouped.items())
                if key != logical_key
                and peer_rows[0].group_kind == first.group_kind
                and peer_rows[0].group_key == first.group_key
                and peer_rows[0].value_kind == "location"
                and any(len(normalized(peer.source_text)) > 2 for peer in peer_rows)
            )
            candidate = binding_candidate(
                kind="potential_country_code_derivation",
                logical_key=logical_key,
                rows=rows,
                suggested_derivations=("country_code",),
                relationship=(
                    "Verify whether this two-letter source-only code is deterministically "
                    "projected from a country binding in the same semantic group."
                ),
            )
            if possible_dependency_keys:
                dependency_rows = tuple(
                    peer for key in possible_dependency_keys for peer in grouped[key]
                )
                candidate["logicalKeys"] = (logical_key, *possible_dependency_keys)
                candidate["lineIds"] = tuple(
                    sorted(
                        {
                            *(line_id(row.char_start) for row in rows),
                            *(line_id(row.char_start) for row in dependency_rows),
                        }
                    )
                )
                candidate["sourceTexts"] = tuple(
                    sorted(
                        {
                            *(row.source_text for row in rows),
                            *(row.source_text for row in dependency_rows),
                        }
                    )
                )
                candidate["details"]["possibleDependencyLogicalKeys"] = possible_dependency_keys
            candidates.append(candidate)

        numeric_package_surface = first.value_kind == "package" and all(
            re.match(r"^\s*[0-9]", surface) is not None for surface in surfaces
        )
        numeric_total = numeric_package_surface or (
            first.value_kind in {"integer", "decimal_measurement"}
            and (
                "total" in key_tokens
                or any(
                    re.search(r"(?i)\btotal\b", current_and_previous_line(row.char_start))
                    is not None
                    for row in rows
                )
            )
        )
        if first.render_mode == "deterministic_auxiliary" and numeric_total:
            candidates.append(
                binding_candidate(
                    kind="potential_calculated_total_derivation",
                    logical_key=logical_key,
                    rows=rows,
                    suggested_derivations=(
                        "sum_package_quantity",
                        "sum_gross_weight",
                        "sum_net_weight",
                        "sum_tare_weight",
                        "sum_volume",
                        "sum_monetary_amounts",
                        "sum_decimal_values",
                        "container_count",
                        "package_count",
                        "container_package_count",
                    ),
                    relationship=(
                        "Verify whether this numeric or package surface is a detail assertion or "
                        "a total calculated from other owned facts; a direct structured scalar "
                        "remains a valid target binding."
                    ),
                )
            )

        equipment_receipt_surface = bool(surfaces) and all(
            re.match(r"(?i)^\s*[0-9]+\s*[x\u00d7](?=\s|[0-9]|$)", surface) is not None
            for surface in surfaces
        )
        if (
            first.render_mode in {"target_binding", "deterministic_auxiliary"}
            and first.group_kind == "equipment"
            and equipment_receipt_surface
            and (
                first.render_mode == "deterministic_auxiliary"
                or not first.target_paths
                or any(not path.endswith(".typeDescription") for path in first.target_paths)
            )
        ):
            candidates.append(
                binding_candidate(
                    kind="potential_equipment_receipt_derivation",
                    logical_key=logical_key,
                    rows=rows,
                    suggested_derivations=("equipment_receipt",),
                    relationship=(
                        "Verify whether the count-times-equipment surface is a receipt composed "
                        "from container facts rather than one directly stored scalar."
                    ),
                )
            )

    if source_target is not None:
        source_lines = line_spans(raw)
        nearby_unowned_cache: dict[str, tuple[str, tuple[tuple[str, str], ...]]] = {}

        def nearby_unowned_window(row: SpanDraft) -> tuple[str, tuple[tuple[str, str], ...]]:
            cached = nearby_unowned_cache.get(row.draft_id)
            if cached is not None:
                return cached
            row_first_line_index = next(
                index
                for index, source_line in enumerate(source_lines)
                if source_line.char_start <= row.char_start <= source_line.char_end
            )
            row_last_character = max(row.char_start, row.char_end - 1)
            row_last_line_index = next(
                index
                for index, source_line in enumerate(source_lines)
                if source_line.char_start <= row_last_character <= source_line.char_end
            )
            first_line = max(0, row_first_line_index - 2)
            last_line = min(len(source_lines) - 1, row_last_line_index + 2)
            window_start = source_lines[first_line].char_start
            window_end = source_lines[last_line].char_end
            characters = list(raw[window_start:window_end])
            for owned in drafts:
                overlap_start = max(window_start, owned.char_start)
                overlap_end = min(window_end, owned.char_end)
                if overlap_start < overlap_end:
                    characters[overlap_start - window_start : overlap_end - window_start] = " " * (
                        overlap_end - overlap_start
                    )
            text = "".join(characters)
            per_line = tuple(
                (
                    source_line.line_id,
                    text[
                        source_line.char_start - window_start : source_line.char_end - window_start
                    ],
                )
                for source_line in source_lines[first_line : last_line + 1]
            )
            result = (text, per_line)
            nearby_unowned_cache[row.draft_id] = result
            return result

        def nearby_unowned_text(row: SpanDraft) -> str:
            return nearby_unowned_window(row)[0]

        def omitted_evidence_line_ids(
            row: SpanDraft, omitted_tokens: Sequence[str]
        ) -> tuple[str, ...]:
            omitted_surface = normalized(" ".join(omitted_tokens))
            if not omitted_surface:
                return ()
            return tuple(
                line_id_value
                for line_id_value, unowned_line in nearby_unowned_window(row)[1]
                if omitted_surface in normalized(unowned_line)
            )

        for logical_key, rows in sorted(grouped.items()):
            frame_rows: list[SpanDraft] = []
            frame_details: list[dict[str, Any]] = []
            projection_rows: list[SpanDraft] = []
            projection_details: list[dict[str, Any]] = []
            if rows[0].render_mode == "agent_residual":
                joint_original_bill_status = (
                    rows[0].target_paths == ("documentPatch.negotiability",)
                    and len(rows) >= 2
                    and any(
                        "numberoforiginal" in normalized(current_and_previous_line(row.char_start))
                        and (
                            "billsoflading" in normalized(current_and_previous_line(row.char_start))
                            or "fbl" in normalized(current_and_previous_line(row.char_start))
                        )
                        for row in rows
                    )
                )
                relationship = (
                    "Verify this as one joint document-status contract: the populated original-"
                    "bill value and the separate operative negotiability surface are intentionally "
                    "heterogeneous occurrences that must remain coherent. Do not require either "
                    "occurrence to encode the complete target alone or split the count into an "
                    "independently generated auxiliary."
                    if joint_original_bill_status
                    else (
                        "Verify that every declared target component is represented by each "
                        "complete residual occurrence and that its boundaries exclude unrelated "
                        "markers, punctuation, captions, and adjacent facts."
                    )
                )
                candidates.append(
                    {
                        "kind": "agent_residual_contract_review",
                        "logicalKeys": (logical_key,),
                        "lineIds": tuple(sorted({line_id(row.char_start) for row in rows})),
                        "sourceTexts": tuple(sorted({row.source_text for row in rows})),
                        "targetPaths": rows[0].target_paths,
                        "details": {
                            "occurrences": tuple(
                                {
                                    "lineId": line_id(row.char_start),
                                    "nearbyUnownedText": nearby_unowned_text(row).strip(),
                                }
                                for row in rows
                            ),
                            "relationshipToVerify": relationship,
                        },
                    }
                )
            for row in rows:
                analysis = target_binding_surface_analysis(
                    draft=row,
                    source_target=source_target,
                )
                if analysis is None:
                    continue
                if any(
                    character.isalnum()
                    for character in analysis.literal_prefix + analysis.literal_suffix
                ):
                    frame_rows.append(row)
                    frame_details.append(
                        {
                            "lineId": line_id(row.char_start),
                            "literalPrefix": analysis.literal_prefix,
                            "literalSuffix": analysis.literal_suffix,
                            "nearbyUnownedText": nearby_unowned_text(row).strip(),
                        }
                    )
                omitted_tokens = (
                    analysis.omitted_target_prefix_tokens + analysis.omitted_target_suffix_tokens
                )
                omitted_line_ids = omitted_evidence_line_ids(row, omitted_tokens)
                if omitted_line_ids:
                    projection_rows.append(row)
                    projection_details.append(
                        {
                            "lineId": line_id(row.char_start),
                            "unownedLineIds": omitted_line_ids,
                            "omittedTargetPrefixTokens": analysis.omitted_target_prefix_tokens,
                            "omittedTargetSuffixTokens": analysis.omitted_target_suffix_tokens,
                            "nearbyContext": current_and_previous_line(row.char_start),
                            "nearbyUnownedText": nearby_unowned_text(row).strip(),
                        }
                    )
            if frame_rows:
                candidates.append(
                    {
                        "kind": "target_binding_alphanumeric_frame",
                        "logicalKeys": (logical_key,),
                        "lineIds": tuple(sorted({line_id(row.char_start) for row in frame_rows})),
                        "sourceTexts": tuple(sorted({row.source_text for row in frame_rows})),
                        "targetPaths": rows[0].target_paths,
                        "details": {
                            "occurrences": tuple(frame_details),
                            "relationshipToVerify": (
                                "The host proved the structured scalar inside this larger source "
                                "surface. Decide whether each alphanumeric frame is selected data "
                                "that belongs inside the binding or generic caption/format text."
                            ),
                        },
                    }
                )
            if projection_rows:
                candidates.append(
                    {
                        "kind": "nearby_omitted_target_tokens",
                        "logicalKeys": (logical_key,),
                        "lineIds": tuple(
                            sorted(
                                {
                                    line_id_value
                                    for details in projection_details
                                    for line_id_value in (
                                        details["lineId"],
                                        *cast(Sequence[str], details["unownedLineIds"]),
                                    )
                                }
                            )
                        ),
                        "sourceTexts": tuple(sorted({row.source_text for row in projection_rows})),
                        "targetPaths": rows[0].target_paths,
                        "details": {
                            "occurrences": tuple(projection_details),
                            "relationshipToVerify": (
                                "The owned surface is a token projection of its target value and "
                                "the omitted target tokens also appear unowned within one adjacent "
                                "line. Decide whether they complete this physical value surface."
                            ),
                        },
                    }
                )

            first = rows[0]
            contexts = tuple(
                dict.fromkeys(current_and_previous_line(row.char_start) for row in rows)
            )
            if (
                len(rows) > 1
                and len(contexts) > 1
                and first.render_mode not in {"carrier_static", "literal_static"}
                and first.value_kind
                in {"date", "location", "commercial_text", "legal_text", "operational_text"}
            ):
                candidates.append(
                    {
                        "kind": "repeated_binding_context_review",
                        "logicalKeys": (logical_key,),
                        "lineIds": tuple(sorted({line_id(row.char_start) for row in rows})),
                        "sourceTexts": tuple(sorted({row.source_text for row in rows})),
                        "targetPaths": first.target_paths,
                        "context": contexts,
                        "details": {
                            "relationshipToVerify": (
                                "Compare the semantic role of every occurrence. Equal date, "
                                "location, commercial, legal, or operational text in different "
                                "contexts must not share one owner unless it is the same fact."
                            )
                        },
                    }
                )

    if source_target is not None:
        candidates.extend(
            coherence_review_candidates(
                raw=raw,
                bindings=drafts,
                source_target=source_target,
                constraints=coherence_constraints,
            )
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
    coherence_constraints: Sequence[CoherenceConstraint] = (),
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
        "allowedTargetPaths": _critic_addressable_target_paths(
            source_target=source_target,
            drafts=drafts,
        ),
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
        "reviewCandidates": _critic_review_candidates(
            raw=raw,
            drafts=drafts,
            source_target=source_target,
            semantic_only_target_facts=effective_semantic_only,
            coherence_constraints=coherence_constraints,
        ),
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


def _compiler_stagnation_reason(stages: Sequence[AgentStageArtifact]) -> str | None:
    """Stop spending when a local compiler repair leaves the same host defect unchanged."""

    if len(stages) < 2:
        return None
    last_two = stages[-2:]
    if all(stage.status == "host_rejected" for stage in last_two):
        errors = {stage.error_message for stage in last_two}
        if len(errors) == 1:
            return (
                "compiler made no progress after two consecutive identical host rejections: "
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
    coherence_constraints: Sequence[CoherenceConstraint],
    critic_outputs: Sequence[CriticAgentOutput],
) -> ExtractionStateCheckpoint:
    validate_draft_source_alignment(raw=raw, drafts=drafts)
    return ExtractionStateCheckpoint.model_validate(
        {
            "schema_version": 2,
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
            "coherence_constraints": tuple(coherence_constraints),
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
    tuple[CoherenceConstraint, ...],
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
    validate_draft_source_alignment(raw=raw, drafts=raw_drafts)
    drafts = normalize_source_boundaries(
        raw=raw,
        drafts=normalize_deterministic_draft_semantics(
            raw=raw,
            drafts=normalize_structured_row_locality(
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
    validate_draft_source_alignment(raw=raw, drafts=drafts)
    validate_target_binding_relationships(drafts=drafts, source_target=source_target, raw=raw)
    validate_binding_realizations(raw=raw, drafts=drafts, source_target=source_target)
    with suppress(CoherenceReviewRequired):
        validate_coherence_contracts(
            raw=raw,
            bindings=drafts,
            source_target=source_target,
            constraints=checkpoint.coherence_constraints,
            require_complete=False,
        )
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
        checkpoint.coherence_constraints,
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
    drafts = _materialize_compiler_drafts(
        raw=raw,
        source_target=source_target,
        anchors=anchors,
        proposed=proposed,
        anchor_overrides=output.anchor_overrides,
        semantic_only_target_paths=tuple(fact.target_path for fact in declared_semantic_only),
    )
    # Overrides are transactional edits to accepted anchors. Boundary validation must inspect the
    # materialized transaction, not the pre-edit union; otherwise a correctly removed substring
    # anchor still rejects the replacement that made it safe.
    validate_mutable_token_boundaries(raw=raw, drafts=drafts)
    validate_target_binding_relationships(drafts=drafts, source_target=source_target, raw=raw)
    validate_binding_realizations(raw=raw, drafts=drafts, source_target=source_target)
    host_refuted_semantic_only_paths = refuted_semantic_only_target_paths(drafts)
    declared_semantic_only = tuple(
        fact
        for fact in declared_semantic_only
        if fact.target_path not in host_refuted_semantic_only_paths
    )
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
    assessment = normalize_pinned_carrier_assessment(
        assessment=output.carrier,
        expected=expected_carrier,
        raw=raw,
        drafts=drafts,
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
    tuple[CoherenceConstraint, ...],
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
            return None, None, (), (), [], warning()
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
            (),
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
            (),
            [],
            warning(
                "Current host rejected the prior successful compiler checkpoint: " + str(error)
            ),
        )
    coherence_constraints: tuple[CoherenceConstraint, ...] = ()
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
    if result.status == "review_required" and not final_is_successful_pass:
        raise ValueError("review-required resume case lacks its final successful critic pass")
    newly_accepted_prior_rejections: list[int] = []
    for critic_stage, review in candidate_reviews:
        if review.verdict == "pass":
            if review.coherence_decisions:
                candidates = _critic_review_candidates(
                    raw=raw,
                    drafts=drafts,
                    source_target=source_target,
                    semantic_only_target_facts=semantic_only,
                    coherence_constraints=coherence_constraints,
                )
                coherence_constraints = materialize_coherence_decisions(
                    candidates=candidates,
                    constraints=coherence_constraints,
                    decisions=review.coherence_decisions,
                )
            # A prior pass is evidence only for the prior prompt and host contract. When this
            # function is used, the current run must obtain its own independent final pass.
            continue
        drafts_before_review = drafts
        semantic_before_review = semantic_only
        constraints_before_review = coherence_constraints
        try:
            candidates = _critic_review_candidates(
                raw=raw,
                drafts=drafts,
                source_target=source_target,
                semantic_only_target_facts=semantic_only,
                coherence_constraints=coherence_constraints,
            )
            drafts, semantic_only, coherence_constraints = _apply_validated_critic_review(
                review=review,
                raw=raw,
                drafts=drafts,
                source_target=source_target,
                assessment=assessment,
                semantic_only_target_facts=semantic_only,
                coherence_constraints=coherence_constraints,
                review_candidates=candidates,
            )
        except Exception as error:
            if str(error).startswith(
                "critic transaction is a functional no-op after host canonicalization"
            ):
                skipped_noop_passes.append(critic_stage.pass_number)
                continue
            drafts = drafts_before_review
            semantic_only = semantic_before_review
            coherence_constraints = constraints_before_review
            replay_warnings.append(
                "Current host skipped incompatible prior critic pass "
                f"{critic_stage.pass_number}: {error}"
            )
            continue
        if critic_stage.status == "host_rejected":
            newly_accepted_prior_rejections.append(critic_stage.pass_number)
        critic_outputs.append(review)
    replay_note = (
        "Current host accepted formerly host-rejected critic transactions at passes "
        + ",".join(str(value) for value in newly_accepted_prior_rejections)
        if newly_accepted_prior_rejections
        else None
    )
    return (
        drafts,
        assessment,
        semantic_only,
        coherence_constraints,
        critic_outputs,
        warning(replay_note),
    )


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
    tuple[CoherenceConstraint, ...],
    list[CriticAgentOutput],
    str | None,
]:
    checkpoint_error: Exception | None = None
    checkpoint_state: (
        tuple[
            tuple[SpanDraft, ...],
            CarrierAssessment,
            tuple[SemanticOnlyTargetFact, ...],
            tuple[CoherenceConstraint, ...],
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
            (
                checkpoint_drafts,
                checkpoint_assessment,
                checkpoint_semantic_only,
                checkpoint_constraints,
                checkpoint_critic_outputs,
            ) = checkpoint_state
            return (
                checkpoint_drafts,
                checkpoint_assessment,
                checkpoint_semantic_only,
                checkpoint_constraints,
                checkpoint_critic_outputs,
                None,
            )

    (
        drafts,
        assessment,
        semantic_only,
        constraints,
        critic_outputs,
        replay_warning,
    ) = _replay_checkpoint_case(
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
        drafts, assessment, semantic_only, constraints, critic_outputs = checkpoint_state
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
        constraints,
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
    compiler_repair_prompt: str | None = None,
    critic_audit_prompt: str | None = None,
    critic_plan_prompt: str | None = None,
    config: ExtractionConfig,
    staged: StagedArtifactRun,
    document_limiter: asyncio.Semaphore,
    resume_run_root: Path | None = None,
    resume_result: ExtractionCaseResult | None = None,
    resume_contract_match: bool = False,
    customs_registry: CustomsProgramRegistry | None = None,
) -> tuple[ExtractionCaseResult, CertifiedSemanticTemplate | None, str]:
    if (config.customs_program_registry is None) != (customs_registry is None):
        raise ValueError("configured customs compilation registry was not supplied to the case")
    relative_root = f"cases/{document_id}"
    existing_result_path = staged.stage_root / relative_root / "result.json"
    if existing_result_path.is_file():
        result = ExtractionCaseResult.model_validate_json(
            read_regular_file_bytes(existing_result_path), strict=True
        )
        existing_template: CertifiedSemanticTemplate | None = None
        masked = ""
        if result.status == "certified":
            existing_template = CertifiedSemanticTemplate.model_validate_json(
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
        return result, existing_template, masked

    async with document_limiter:
        started = time.perf_counter()
        raw = cast(str, source["joinedRawText"])
        if sha256_bytes(raw.encode("utf-8")) != source["joinedRawTextSha256"]:
            raise ValueError(f"source raw-text hash differs for {document_id}")
        source_target = cast(Mapping[str, Any], source["target"])
        resumed_from_run = resume_run_root.name if resume_run_root is not None else None
        prior_elapsed_seconds = resume_result.elapsed_seconds if resume_result is not None else 0.0
        if resume_result is not None and resume_result.document_id != document_id:
            raise ValueError("resume case document identity differs")
        integrity_issues = source_template_integrity_issues(raw, source_target)
        if integrity_issues:
            prior_compiler_stages = (
                tuple(resume_result.compiler_stages) if resume_result is not None else ()
            )
            prior_critic_stages = (
                tuple(resume_result.critic_stages) if resume_result is not None else ()
            )
            prior_cost = sum(
                (
                    stage.usage.estimatedCostUsd
                    for stage in (*prior_compiler_stages, *prior_critic_stages)
                ),
                Decimal(0),
            )
            risks = all_risk_candidates(raw, (), source_target)
            result = ExtractionCaseResult.model_validate(
                {
                    "document_id": document_id,
                    "status": "review_required",
                    "rejection_reasons": tuple(
                        "source template integrity requires review: " + issue
                        for issue in integrity_issues
                    ),
                    "template_sha256": None,
                    "compiler_stages": prior_compiler_stages,
                    "critic_stages": prior_critic_stages,
                    "risk_candidates": risks,
                    "elapsed_seconds": prior_elapsed_seconds + time.perf_counter() - started,
                    "resumed_from_run": resumed_from_run,
                    "resumed_prior_elapsed_seconds": prior_elapsed_seconds,
                    "resumed_prior_compiler_stages": len(prior_compiler_stages),
                    "resumed_prior_critic_stages": len(prior_critic_stages),
                    "resumed_prior_estimated_cost_usd": prior_cost,
                    "resume_contract_match": (
                        resume_contract_match if resume_result is not None else None
                    ),
                    "resume_replay_warning": None,
                    "resume_mode": "source_integrity_review",
                }
            )
            staged.publish_bytes(f"{relative_root}/source.txt", raw.encode("utf-8"))
            staged.publish_json(f"{relative_root}/source-label.json", source_target)
            staged.publish_bytes(f"{relative_root}/masked-template.txt", raw.encode("utf-8"))
            staged.publish_json(f"{relative_root}/result.json", result.model_dump(mode="json"))
            print(
                f"[{ordinal:03d}/{config.workflow.documents:03d}] source_integrity_review "
                f"{document_id} new_provider_calls=0",
                flush=True,
            )
            return result, None, raw
        document_anchors = [row for row in anchor_rows if row["document_id"] == document_id]
        prepared_anchor_rows, locality_report = _prepare_anchor_rows(
            raw=raw,
            document_id=document_id,
            anchor_rows=document_anchors,
            source_target=source_target,
            agent_contract_protocol=config.workflow.agent_contract_protocol,
        )
        if locality_report is not None:
            staged.publish_json(f"{relative_root}/anchor-locality.json", locality_report)
        anchors = normalize_target_cobindings(
            drafts=anchor_drafts(
                raw=raw,
                document_id=document_id,
                anchors=prepared_anchor_rows,
            ),
            source_target=source_target,
        )
        risks = all_risk_candidates(raw, (), source_target)
        certified_resume_mode = (
            "exact_contract_reuse"
            if resume_contract_match
            else (
                "current_host_recertification"
                if config.workflow.certified_resume_policy == "current_host_recertify"
                else None
            )
        )
        if customs_registry is not None and not resume_contract_match:
            certified_resume_mode = None  # Changed caption contracts require a fresh critic pass.
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
                resumed_template = prior_template
                resumed_template_payload = prior_template_payload
                masked_payload = read_regular_file_bytes(prior_case_root / "masked-template.txt")
                catalog_payload = read_regular_file_bytes(prior_case_root / "catalog-row.json")
                checkpoint_payload = prior_checkpoint_payload
                replay_note = None
            else:
                (
                    certified_drafts,
                    certified_assessment,
                    certified_semantic_only,
                    certified_coherence_constraints,
                    certified_applied_revisions,
                ) = _restore_state_checkpoint(
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
                certified_template = certify_template(
                    raw=raw,
                    document_id=document_id,
                    feature=feature,
                    source_target=source_target,
                    assessment=certified_assessment,
                    drafts=certified_drafts,
                    risks=risks,
                    critic_outputs=(*certified_applied_revisions, final_review),
                    semantic_only_target_facts=certified_semantic_only,
                    coherence_constraints=certified_coherence_constraints,
                )
                resumed_template_payload = json_artifact_bytes(
                    certified_template.model_dump(mode="json")
                )
                masked_payload = masked_source(raw, certified_drafts).encode("utf-8")
                catalog_payload = json_artifact_bytes(template_summary(certified_template))
                current_checkpoint = _state_checkpoint(
                    document_id=document_id,
                    raw=raw,
                    source_target=source_target,
                    drafts=certified_drafts,
                    assessment=certified_assessment,
                    semantic_only_target_facts=certified_semantic_only,
                    coherence_constraints=certified_coherence_constraints,
                    critic_outputs=certified_applied_revisions,
                )
                checkpoint_payload = json_artifact_bytes(current_checkpoint.model_dump(mode="json"))
                resumed_template = certified_template
                replay_note = (
                    "Recompiled and recertified the exact prior state and final critic receipt "
                    "under the current host contract without a provider request; the prior "
                    "compiler and critic prompts remain recorded in the resumed stage lineage"
                )
            reused_result = resume_result.model_copy(
                update={
                    "template_sha256": sha256_bytes(resumed_template_payload),
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
            staged.publish_bytes(f"{relative_root}/template.json", resumed_template_payload)
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
            return reused_result, resumed_template, masked_payload.decode("utf-8")

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
        drafts: tuple[SpanDraft, ...] | None
        assessment: CarrierAssessment | None
        semantic_only_target_facts: tuple[SemanticOnlyTargetFact, ...]
        coherence_constraints: tuple[CoherenceConstraint, ...]
        critic_outputs: list[CriticAgentOutput]
        resume_replay_warning: str | None
        if resume_result is not None:
            if resume_run_root is None:
                raise ValueError("resume case lacks its run root")
            checkpoint_path = resume_run_root / relative_root / "state-checkpoint.json"
            (
                drafts,
                assessment,
                semantic_only_target_facts,
                coherence_constraints,
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
            coherence_constraints = ()
            critic_outputs = []
            resume_replay_warning = None
        reasons: list[str] = []
        review_required = False
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
        terminal_confirmation_pending = False
        post_threshold_confirmation_used = False
        candidate_prepass_pending = (
            config.workflow.agent_contract_protocol
            in {"candidate_first_staged_local_v7", "candidate_first_staged_local_v8"}
            and not critic_stages
        )
        full_critic_seen = bool(critic_stages)

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
                stagnation_reason = _compiler_stagnation_reason(
                    compiler_stages[resumed_prior_compiler_stages:]
                )
                if stagnation_reason is not None:
                    reasons.append(stagnation_reason)
                    break
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

                def preview_compiler_repair(candidate: CompilerAgentOutput) -> None:
                    try:
                        _validated_compiler_state(
                            output=candidate,
                            raw=raw,
                            source_target=source_target,
                            anchors=anchors,
                        )
                    except Exception as preview_error:
                        raise _compiler_host_rejection(
                            output=candidate,
                            raw=raw,
                            source_target=source_target,
                            anchors=anchors,
                            primary_error=preview_error,
                        ) from preview_error

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
                    repair_prompt=compiler_repair_prompt,
                    preview_output=preview_compiler_repair,
                )
                compiler_attempt_calls += 1
                if output is None:
                    compiler_stages.append(stage)
                    if compiler_attempt_calls >= config.workflow.max_compiler_passes:
                        reasons.append(stage.error_message or "compiler provider failed")
                    continue
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
                    try:
                        prior_compiler_drafts = resolve_agent_proposals(
                            raw=raw,
                            proposals=output.bindings,
                        )
                    except ValueError as inventory_error:
                        partial_inventory, rejected_occurrences = resolve_agent_proposal_inventory(
                            raw=raw,
                            proposals=output.bindings,
                        )
                        prior_compiler_drafts = partial_inventory or None
                        if partial_inventory:
                            revision_context += (
                                "\n- local repair inventory excludes explicitly unresolved "
                                f"occurrence(s): {', '.join(rejected_occurrences)}"
                            )
                        else:
                            revision_context += (
                                "\n- local repair candidate inventory unavailable: "
                                + str(inventory_error)
                            )
                    if compiler_attempt_calls == config.workflow.max_compiler_passes:
                        reasons.append(f"compiler host rejection: {rejection}")
                    continue

            if drafts is None or assessment is None:
                raise AssertionError("critic phase entered without a validated compiler checkpoint")

            while (
                critic_attempt_calls < config.workflow.max_critic_passes
                or terminal_confirmation_pending
            ):
                if customs_registry is not None:
                    try:
                        drafts = normalize_registered_caption_ownership(
                            raw=raw, drafts=tuple(drafts), registry=customs_registry
                        )
                        validate_draft_source_alignment(raw=raw, drafts=drafts)
                        validate_binding_realizations(
                            raw=raw, drafts=drafts, source_target=source_target
                        )
                    except ValueError as error:
                        reasons.append(
                            "registered customs ownership requires review: " + str(error)
                        )
                        review_required = True
                        break
                is_terminal_confirmation = terminal_confirmation_pending
                terminal_confirmation_pending = False
                stagnation_reason = _critic_stagnation_reason(
                    critic_stages[resumed_prior_critic_stages:]
                )
                if stagnation_reason is not None:
                    reasons.append(stagnation_reason)
                    break
                critic_payload = _critic_payload(
                    document_id=document_id,
                    raw=raw,
                    source_target=source_target,
                    feature=feature,
                    drafts=drafts,
                    risks=risks,
                    prior_error=critic_revision_context,
                    semantic_only_target_facts=semantic_only_target_facts,
                    coherence_constraints=coherence_constraints,
                    review_history=(
                        ()
                        if config.workflow.agent_contract_protocol
                        in {"candidate_first_staged_local_v7", "candidate_first_staged_local_v8"}
                        else critic_stages
                    ),
                    agent_contract_protocol=config.workflow.agent_contract_protocol,
                )
                candidate_only = candidate_prepass_pending
                if candidate_only and not (
                    critic_payload["reviewCandidates"] or critic_payload["remainingRiskCandidates"]
                ):
                    candidate_prepass_pending = False
                    continue
                launch_critic, post_threshold_confirmation_used = _critic_launch_decision(
                    spent=current_attempt_spent(),
                    threshold=(config.workflow.additional_call_launch_threshold_usd_per_document),
                    has_unconfirmed_state=(
                        not full_critic_seen
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
                candidate_prepass_pending = False
                if not candidate_only:
                    full_critic_seen = True
                review, critic_stage = await runtime.critic(
                    pass_number=critic_pass,
                    system_prompt=critic_prompt,
                    payload=critic_payload,
                    retries=config.workflow.critic_output_retries,
                    audit_prompt=critic_audit_prompt,
                    plan_prompt=critic_plan_prompt,
                    candidate_only=candidate_only,
                    preview_review=(
                        partial(
                            _preview_critic_review,
                            raw=raw,
                            drafts=tuple(cast(Sequence[SpanDraft], drafts)),
                            source_target=source_target,
                            assessment=assessment,
                            semantic_only_target_facts=tuple(semantic_only_target_facts),
                            coherence_constraints=tuple(coherence_constraints),
                        )
                    )
                    if config.workflow.agent_contract_protocol
                    in {
                        "staged_local_v4",
                        "faceted_staged_local_v5",
                        "partitioned_staged_local_v6",
                        "candidate_first_staged_local_v7",
                        "candidate_first_staged_local_v8",
                        "hybrid_reference_partitioned_v10",
                    }
                    else None,
                )
                if review is None:
                    critic_stages.append(critic_stage)
                    if (
                        is_terminal_confirmation
                        or critic_attempt_calls >= config.workflow.max_critic_passes
                    ):
                        reasons.append(critic_stage.error_message or "critic provider failed")
                    continue
                if review.verdict == "revise":
                    try:
                        (
                            drafts,
                            semantic_only_target_facts,
                            coherence_constraints,
                        ) = _apply_validated_critic_review(
                            review=review,
                            raw=raw,
                            drafts=drafts,
                            source_target=source_target,
                            assessment=assessment,
                            semantic_only_target_facts=semantic_only_target_facts,
                            coherence_constraints=coherence_constraints,
                            review_candidates=cast(
                                Sequence[Mapping[str, Any]],
                                critic_payload["reviewCandidates"],
                            ),
                        )
                    except Exception as error:
                        critic_stages.append(_mark_host_rejected(critic_stage, error))
                        critic_revision_context = f"Host rejected critic patch: {error}"
                        if (
                            is_terminal_confirmation
                            or critic_attempt_calls >= config.workflow.max_critic_passes
                        ):
                            reasons.append(
                                f"critic exhausted passes after host-rejected patch: {error}"
                            )
                            break
                        continue
                    critic_outputs.append(review)
                    critic_stages.append(critic_stage)
                    critic_revision_context = None
                    if is_terminal_confirmation:
                        reasons.append(
                            "terminal critic confirmation revised the state; no independently "
                            "confirmed template was published"
                        )
                    elif critic_attempt_calls == config.workflow.max_critic_passes:
                        terminal_confirmation_pending = True
                    continue
                if candidate_only:
                    critic_stages.append(critic_stage)
                    critic_revision_context = None
                    if critic_attempt_calls == config.workflow.max_critic_passes:
                        terminal_confirmation_pending = True
                    continue
                try:
                    prospective_coherence_constraints = materialize_coherence_decisions(
                        candidates=cast(
                            Sequence[Mapping[str, Any]],
                            critic_payload["reviewCandidates"],
                        ),
                        constraints=coherence_constraints,
                        decisions=review.coherence_decisions,
                    )
                except Exception as error:
                    critic_stages.append(_mark_host_rejected(critic_stage, error))
                    critic_revision_context = f"Host rejected critic coherence decisions: {error}"
                    if (
                        is_terminal_confirmation
                        or critic_attempt_calls >= config.workflow.max_critic_passes
                    ):
                        reasons.append(
                            "critic exhausted passes after invalid coherence decisions: "
                            + str(error)
                        )
                        break
                    continue
                remaining_risks = uncovered_risks(raw, risks, drafts)
                if remaining_risks:
                    details = "; ".join(
                        f"{row.risk_id} {row.line_id}={row.source_text!r}"
                        for row in remaining_risks
                    )
                    false_pass_error = ValueError(
                        "critic passed while deterministic risk candidates remained unowned: "
                        + details
                    )
                    critic_stages.append(_mark_host_rejected(critic_stage, false_pass_error))
                    critic_revision_context = f"Host rejected critic pass: {false_pass_error}"
                    if (
                        is_terminal_confirmation
                        or critic_attempt_calls >= config.workflow.max_critic_passes
                    ):
                        reasons.append(
                            "critic exhausted passes after false pass with unowned risks: "
                            + details
                        )
                        break
                    continue
                try:
                    prospective_critic_outputs = (*critic_outputs, review)
                    template = certify_template(
                        raw=raw,
                        document_id=document_id,
                        feature=feature,
                        source_target=source_target,
                        assessment=assessment,
                        drafts=drafts,
                        risks=risks,
                        critic_outputs=prospective_critic_outputs,
                        semantic_only_target_facts=semantic_only_target_facts,
                        coherence_constraints=prospective_coherence_constraints,
                    )
                    coherence_constraints = prospective_coherence_constraints
                    critic_outputs.append(review)
                    critic_stages.append(critic_stage)
                    break
                except CoherenceReviewRequired as error:
                    coherence_constraints = prospective_coherence_constraints
                    critic_outputs.append(review)
                    critic_stages.append(critic_stage)
                    reasons.append(f"semantic coherence requires review: {error}")
                    review_required = True
                    break
                except Exception as error:
                    critic_stages.append(_mark_host_rejected(critic_stage, error))
                    critic_revision_context = f"Host rejected critic pass or certification: {error}"
                    if (
                        is_terminal_confirmation
                        or critic_attempt_calls >= config.workflow.max_critic_passes
                    ):
                        reasons.append(
                            f"critic exhausted passes after failed certification: {error}"
                        )
                        break
                    continue

            if (
                template is None
                and not reasons
                and critic_attempt_calls >= config.workflow.max_critic_passes
                and not terminal_confirmation_pending
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
                "status": (
                    "certified"
                    if template is not None
                    else "review_required"
                    if review_required
                    else "rejected"
                ),
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
                coherence_constraints=coherence_constraints,
                critic_outputs=critic_outputs,
            )
            staged.publish_json(
                f"{relative_root}/state-checkpoint.json",
                checkpoint.model_dump(mode="json"),
            )
        if template is not None and template_payload is not None:
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


def _compilation_acceptance_gate_passed(
    results: Sequence[ExtractionCaseResult],
) -> bool:
    """Accept certified templates and explicit review quarantines, never rejections."""

    return all(result.status in {"certified", "review_required"} for result in results)


def _expected_extraction_artifacts(
    *,
    stage_root: Path,
    results: Sequence[ExtractionCaseResult],
    compiler_repair_prompt: str | None,
    critic_audit_prompt: str | None,
    critic_plan_prompt: str | None,
) -> tuple[str, ...]:
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
    expected.extend(
        f"prompts/{name}"
        for name, prompt in (
            ("compiler-repair.md", compiler_repair_prompt),
            ("critic-audit.md", critic_audit_prompt),
            ("critic-plan.md", critic_plan_prompt),
        )
        if prompt is not None
    )
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
        for optional_name in ("anchor-locality.json", "state-checkpoint.json"):
            if (stage_root / root / optional_name).is_file():
                expected.append(f"{root}/{optional_name}")
        if result.status == "certified":
            expected.extend([f"{root}/catalog-row.json", f"{root}/template.json"])
    return tuple(expected)


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
            f"- Review-required templates: **{counts.get('review_required', 0)}**.",
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
            f"- No-rejection acceptance gate: **{summary['acceptanceGatePassed']}**.",
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
            "prove deterministic are explicitly marked agent-required. Ambiguous arithmetic "
            "relationships are quarantined as review-required. Neither a rejected nor a "
            "review-required case contributes a catalog template.",
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


async def run_extraction(*, project_root: Path, config_path: Path) -> Path:
    project_root = _validated_project_root(project_root=project_root, config_path=config_path)
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
        compiler_prompt,
        critic_prompt,
        compiler_repair_prompt,
        critic_audit_prompt,
        critic_plan_prompt,
    ) = _config_inputs(config, project_root)
    manifest = build_selection_manifest(
        config=config,
        source_rows=source_rows,
        feature_rows=feature_rows,
        anchor_rows=anchor_rows,
    )
    resume_run_root, resume_results, resume_prompt_changed, resume_host_contract_matched = (
        _load_resume_run(
            project_root=project_root,
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
        compiler_repair_prompt=compiler_repair_prompt,
        critic_audit_prompt=critic_audit_prompt,
        critic_plan_prompt=critic_plan_prompt,
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
        "productionSourceSha256": _production_source_hash(project_root),
    }
    transaction_sha = sha256_bytes(canonical_json_bytes(transaction_payload))
    output_parent = (project_root / config.output_dir).resolve()
    staged = StagedArtifactRun(
        output_parent=output_parent,
        run_name=config.run_name,
        transaction_sha256=transaction_sha,
    )
    if staged.completed:
        return staged.final_root
    staged.recover_interrupted_temporary_files()
    staged.publish_json("config.json", config.model_dump(mode="json"))
    staged.publish_json("preflight.json", preflight)
    staged.publish_json("selection-manifest.json", manifest.model_dump(mode="json"))
    staged.publish_json("resume-contract.json", _resume_contract(project_root))
    staged.publish_bytes("prompts/compiler.md", compiler_prompt.encode("utf-8"))
    staged.publish_bytes("prompts/critic.md", critic_prompt.encode("utf-8"))
    for name, prompt in (
        ("compiler-repair.md", compiler_repair_prompt),
        ("critic-audit.md", critic_audit_prompt),
        ("critic-plan.md", critic_plan_prompt),
    ):
        if prompt is not None:
            staged.publish_bytes(f"prompts/{name}", prompt.encode("utf-8"))

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
        partition_critic_above_request_bytes=(config.workflow.partition_critic_above_request_bytes),
    )
    document_limiter = asyncio.Semaphore(config.workflow.max_concurrent_documents)
    customs_registry = (
        CustomsProgramRegistry.model_validate_json(
            resolve_input(project_root, config.customs_program_registry.path).read_bytes()
        )
        if config.customs_program_registry is not None
        else None
    )
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
                compiler_repair_prompt=compiler_repair_prompt,
                critic_audit_prompt=critic_audit_prompt,
                critic_plan_prompt=critic_plan_prompt,
                config=config,
                staged=staged,
                document_limiter=document_limiter,
                resume_run_root=resume_run_root,
                resume_result=resume_results.get(row.document_id),
                resume_contract_match=resume_contract_match,
                customs_registry=customs_registry,
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
        "acceptanceGatePassed": _compilation_acceptance_gate_passed(results),
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

    staged.commit(
        expected_artifacts=_expected_extraction_artifacts(
            stage_root=staged.stage_root,
            results=results,
            compiler_repair_prompt=compiler_repair_prompt,
            critic_audit_prompt=critic_audit_prompt,
            critic_plan_prompt=critic_plan_prompt,
        ),
        metadata={
            "schemaVersion": 1,
            "phase": config.phase,
            "documents": len(results),
            "certified": len(templates),
            "publishTrainingRecords": False,
        },
    )
    return staged.final_root


def prepare_selection(config_path: Path) -> SelectionManifest:
    project_root = project_root_from_config(config_path)
    config = load_config(config_path)
    (
        source_rows,
        feature_rows,
        anchor_rows,
        _compiler_prompt,
        _critic_prompt,
        *_staged_prompts,
    ) = _config_inputs(config, project_root)
    return build_selection_manifest(
        config=config,
        source_rows=source_rows,
        feature_rows=feature_rows,
        anchor_rows=anchor_rows,
    )


def preflight_extraction(*, project_root: Path, config_path: Path) -> dict[str, Any]:
    project_root = _validated_project_root(project_root=project_root, config_path=config_path)
    config = load_config(config_path)
    (
        source_rows,
        feature_rows,
        anchor_rows,
        compiler_prompt,
        critic_prompt,
        compiler_repair_prompt,
        critic_audit_prompt,
        critic_plan_prompt,
    ) = _config_inputs(config, project_root)
    manifest = build_selection_manifest(
        config=config,
        source_rows=source_rows,
        feature_rows=feature_rows,
        anchor_rows=anchor_rows,
    )
    resume_run_root, resume_results, resume_prompt_changed, resume_host_contract_matched = (
        _load_resume_run(
            project_root=project_root,
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
        compiler_repair_prompt=compiler_repair_prompt,
        critic_audit_prompt=critic_audit_prompt,
        critic_plan_prompt=critic_plan_prompt,
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


def _runtime_source_files(project_root: Path) -> tuple[Path, ...]:
    """Resolve the complete production Python runtime and dependency contract."""

    candidates = [project_root / "pyproject.toml"]
    lock_path = project_root / "uv.lock"
    if lock_path.exists():
        candidates.append(lock_path)
    source_root = project_root / "src" / "document_ocr"
    candidates.extend(sorted(source_root.rglob("*.py")))
    if not source_root.is_dir() or len(candidates) < 2:
        raise ValueError("production runtime source contract is empty")
    for path in candidates:
        if path.is_symlink() or not path.is_file():
            raise ValueError(
                f"resume contract contains a missing or symbolic runtime source file: {path}"
            )
    return tuple(candidates)


def _resume_contract(project_root: Path) -> dict[str, Any]:
    """Hash production runtime code separately from per-run configs and prompts."""

    paths = _runtime_source_files(project_root)
    files = [
        {
            "path": path.relative_to(project_root).as_posix(),
            "sha256": sha256_file(path),
        }
        for path in paths
    ]
    return {
        "schemaVersion": 1,
        "hostImplementationSha256": sha256_bytes(canonical_json_bytes(files)),
        "files": files,
    }


def _production_source_hash(project_root: Path) -> str:
    return str(_resume_contract(project_root)["hostImplementationSha256"])
