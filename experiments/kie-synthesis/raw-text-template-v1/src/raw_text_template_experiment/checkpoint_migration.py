"""One-time, zero-provider migration of paid v1 checkpoints to the production v2 contract.

This module deliberately lives in the experiment. Production accepts only the current checkpoint
schema; legacy interpretation is isolated here and produces an ordinary, hash-sealed current run.
"""

from __future__ import annotations

import json
import resource
import time
from collections import Counter
from collections.abc import Mapping, Sequence
from datetime import UTC, datetime
from decimal import Decimal
from pathlib import Path
from typing import Annotated, Any, Literal, cast

from document_ocr.atomic import json_artifact_bytes, read_regular_file_bytes
from document_ocr.hashing import canonical_json_bytes, sha256_bytes, sha256_file
from document_ocr.synthesis.run_safety import StagedArtifactRun, StagedCommitReceipt
from document_ocr.synthesis.template_compiler.host import (
    all_risk_candidates,
    certify_template,
    masked_source,
    template_summary,
    verify_file,
)
from document_ocr.synthesis.template_compiler.models import (
    CarrierAssessment,
    CriticAgentOutput,
    DraftCheckpointRow,
    ExtractionCaseResult,
    ExtractionConfig,
    ExtractionStateCheckpoint,
    NonEmptyText,
    PinnedRun,
    SelectionManifest,
    SelectionRow,
    SemanticOnlyTargetFact,
)
from document_ocr.synthesis.template_compiler.pipeline import (
    _config_inputs,
    _preflight_summary,
    _production_source_hash,
    _restore_state_checkpoint,
    _resume_contract,
    _state_checkpoint,
    _validated_project_root,
    load_config,
    project_root_from_config,
)
from document_ocr.synthesis.template_compiler.selection import build_selection_manifest
from document_ocr.synthesis.template_integrity import source_template_integrity_issues
from pydantic import BaseModel, ConfigDict, Field, model_validator

_STRICT = ConfigDict(extra="forbid", frozen=True, strict=True, allow_inf_nan=False)
_LEGACY_TASK = "carrier_bound_raw_text_template_extraction"
_LEGACY_PHASES = frozenset({"development30", "transfer200"})


class LegacyExtractionStateCheckpointV1(BaseModel):
    """The exact historical checkpoint shape accepted by this one-time migration."""

    model_config = _STRICT

    schema_version: Literal[1]
    document_id: NonEmptyText
    source_sha256: str
    source_label_sha256: str
    drafts: Annotated[tuple[DraftCheckpointRow, ...], Field(min_length=1)]
    carrier_assessment: CarrierAssessment
    semantic_only_target_facts: tuple[SemanticOnlyTargetFact, ...]
    applied_critic_revisions: tuple[CriticAgentOutput, ...]

    @model_validator(mode="after")
    def revisions_are_applied_transactions(self) -> LegacyExtractionStateCheckpointV1:
        if any(review.verdict != "revise" for review in self.applied_critic_revisions):
            raise ValueError("legacy checkpoint may retain only applied critic revisions")
        if len(self.source_sha256) != 64 or len(self.source_label_sha256) != 64:
            raise ValueError("legacy checkpoint hashes must be SHA-256 values")
        return self


class LegacySelectionManifestV1(BaseModel):
    model_config = _STRICT

    schema_version: Literal[1]
    phase: Literal["development30", "transfer200"]
    selection_seed: int
    source_corpus_sha256: str
    document_features_sha256: str
    rows: tuple[SelectionRow, ...]

    @model_validator(mode="after")
    def ordinals_and_ids_are_unique(self) -> LegacySelectionManifestV1:
        if tuple(row.ordinal for row in self.rows) != tuple(range(1, len(self.rows) + 1)):
            raise ValueError("legacy selection ordinals must be contiguous from one")
        ids = tuple(row.document_id for row in self.rows)
        if len(set(ids)) != len(ids):
            raise ValueError("legacy selection document IDs must be unique")
        return self


MigrationClassification = Literal[
    "current_host_certified",
    "critic_review_required",
    "fresh_compilation_required",
    "source_integrity_review",
]


class MigrationCaseReceipt(BaseModel):
    model_config = _STRICT

    schema_version: Literal[1]
    document_id: NonEmptyText
    classification: MigrationClassification
    provider_requests: Literal[0]
    legacy_run: NonEmptyText
    legacy_commit_sha256: str
    legacy_result_sha256: str
    legacy_checkpoint_sha256: str | None
    current_checkpoint_sha256: str | None
    current_template_sha256: str | None
    current_risk_candidates: Annotated[int, Field(ge=0)]
    retained_final_critic_receipt: bool
    historical_requests: Annotated[int, Field(ge=0)]
    historical_estimated_cost_usd: Annotated[Decimal, Field(ge=0)]
    diagnostic: str | None


def _translate_legacy_checkpoint(
    checkpoint: LegacyExtractionStateCheckpointV1,
) -> ExtractionStateCheckpoint:
    """Translate structure only; current-host restoration performs semantic normalization."""

    return ExtractionStateCheckpoint(
        schema_version=2,
        document_id=checkpoint.document_id,
        source_sha256=checkpoint.source_sha256,
        source_label_sha256=checkpoint.source_label_sha256,
        drafts=checkpoint.drafts,
        coherence_constraints=(),
        carrier_assessment=checkpoint.carrier_assessment,
        semantic_only_target_facts=checkpoint.semantic_only_target_facts,
        applied_critic_revisions=checkpoint.applied_critic_revisions,
    )


def _jsonl_bytes(rows: Sequence[Mapping[str, Any]]) -> bytes:
    return b"".join(canonical_json_bytes(row) + b"\n" for row in rows)


def _legacy_run_root(
    *, project_root: Path, pin: PinnedRun
) -> tuple[Path, StagedCommitReceipt, dict[str, str]]:
    supplied = project_root / pin.path
    if supplied.is_symlink():
        raise ValueError("legacy run must not be a symbolic link")
    root = supplied.resolve(strict=True)
    if not root.is_dir() or project_root not in root.parents:
        raise ValueError("legacy run must be a project-local directory")
    verify_file(root / "_COMMIT.json", pin.commit_sha256)
    receipt = StagedCommitReceipt.model_validate_json(
        read_regular_file_bytes(root / "_COMMIT.json"), strict=True
    )
    validated = StagedArtifactRun(
        output_parent=root.parent,
        run_name=root.name,
        transaction_sha256=receipt.transaction_sha256,
    ).validate_committed_run()
    if validated != receipt:
        raise ValueError("legacy run changed while its commit receipt was validated")
    artifacts = {row.relative_path: row.sha256 for row in receipt.artifacts}
    for required in ("config.json", "results.jsonl", "selection-manifest.json"):
        if required not in artifacts:
            raise ValueError(f"legacy commit omits required artifact: {required}")
    return root, receipt, artifacts


def _validate_legacy_lineage(
    *,
    legacy_root: Path,
    config: ExtractionConfig,
    manifest: SelectionManifest,
) -> LegacySelectionManifestV1:
    legacy_config = json.loads(read_regular_file_bytes(legacy_root / "config.json"))
    if not isinstance(legacy_config, Mapping):
        raise ValueError("legacy config is not an object")
    legacy_phase = legacy_config.get("phase")
    if legacy_config.get("task") != _LEGACY_TASK or legacy_phase not in _LEGACY_PHASES:
        raise ValueError("legacy run is not a supported pinned compilation lineage")
    if legacy_config.get("selection_seed") != config.selection_seed:
        raise ValueError("legacy selection seed differs from the current compilation")
    legacy_inputs = legacy_config.get("inputs")
    if not isinstance(legacy_inputs, Mapping):
        raise ValueError("legacy config inputs are not an object")
    current_inputs = config.inputs.model_dump(mode="json")
    for name in ("source_corpus", "document_features", "anchors"):
        if legacy_inputs.get(name) != current_inputs[name]:
            raise ValueError(f"legacy run differs in pinned input: {name}")

    legacy_manifest = LegacySelectionManifestV1.model_validate_json(
        read_regular_file_bytes(legacy_root / "selection-manifest.json"), strict=True
    )
    if legacy_manifest.phase != legacy_phase:
        raise ValueError("legacy config and selection manifest phases differ")
    if legacy_manifest.selection_seed != manifest.selection_seed:
        raise ValueError("legacy selection manifest seed differs")
    if legacy_manifest.source_corpus_sha256 != manifest.source_corpus_sha256:
        raise ValueError("legacy selection source-corpus hash differs")
    if legacy_manifest.document_features_sha256 != manifest.document_features_sha256:
        raise ValueError("legacy selection feature hash differs")
    legacy_identity = tuple(
        (row.ordinal, row.document_id, row.source_sha256) for row in legacy_manifest.rows
    )
    current_identity = tuple(
        (row.ordinal, row.document_id, row.source_sha256) for row in manifest.rows
    )
    if legacy_identity != current_identity:
        raise ValueError("legacy and current selections do not have identical ordered identity")
    return legacy_manifest


def _legacy_results(
    *, legacy_root: Path, manifest: SelectionManifest
) -> dict[str, ExtractionCaseResult]:
    payload = read_regular_file_bytes(legacy_root / "results.jsonl")
    lines = payload.splitlines()
    if not lines or any(not line for line in lines):
        raise ValueError("legacy results JSONL is empty or contains blank records")
    parsed = tuple(ExtractionCaseResult.model_validate_json(line, strict=True) for line in lines)
    expected = tuple(row.document_id for row in manifest.rows)
    actual = tuple(row.document_id for row in parsed)
    if actual != expected:
        raise ValueError("legacy results do not match the exact ordered current selection")
    return {row.document_id: row for row in parsed}


def _migrated_config(config: ExtractionConfig, *, run_name: str) -> ExtractionConfig:
    payload = config.model_dump(mode="json")
    payload["run_name"] = run_name
    payload["workflow"]["provider_launch_authorized"] = False
    payload["resume_from"] = None
    return ExtractionConfig.model_validate_json(
        json.dumps(payload, ensure_ascii=False, separators=(",", ":"))
    )


def _migration_result(
    *,
    document_id: str,
    status: Literal["certified", "review_required", "rejected"],
    reasons: Sequence[str],
    template_sha256: str | None,
    risk_candidates: Sequence[Any],
    elapsed_seconds: float,
    legacy_run: str,
    final_critic_stage: Any | None = None,
    source_integrity_review: bool = False,
) -> ExtractionCaseResult:
    retained = () if final_critic_stage is None else (final_critic_stage,)
    retained_cost = sum((stage.usage.estimatedCostUsd for stage in retained), Decimal(0))
    return ExtractionCaseResult.model_validate(
        {
            "document_id": document_id,
            "status": status,
            "rejection_reasons": tuple(reasons),
            "template_sha256": template_sha256,
            "compiler_stages": (),
            "critic_stages": retained,
            "risk_candidates": tuple(risk_candidates),
            "elapsed_seconds": elapsed_seconds,
            "resumed_from_run": legacy_run,
            "resumed_prior_elapsed_seconds": 0.0,
            "resumed_prior_compiler_stages": 0,
            "resumed_prior_critic_stages": len(retained),
            "resumed_prior_estimated_cost_usd": retained_cost,
            "resume_contract_match": False,
            "resume_replay_warning": (
                "Legacy checkpoint was explicitly migrated and revalidated by the current "
                "production host without a provider request"
            ),
            "resume_mode": (
                "source_integrity_review"
                if source_integrity_review
                else "current_host_recertification"
            ),
        }
    )


def _expected_artifacts(
    *,
    results: Sequence[ExtractionCaseResult],
    checkpoint_ids: set[str],
    has_compiler_repair_prompt: bool,
) -> tuple[str, ...]:
    expected = [
        "REPORT.md",
        "catalog.jsonl",
        "config.json",
        "migration.json",
        "preflight.json",
        "prompts/compiler.md",
        "prompts/critic-audit.md",
        "prompts/critic-plan.md",
        "prompts/critic.md",
        "results.jsonl",
        "resume-contract.json",
        "selection-manifest.json",
        "summary.json",
    ]
    if has_compiler_repair_prompt:
        expected.append("prompts/compiler-repair.md")
    for result in results:
        root = f"cases/{result.document_id}"
        expected.extend(
            (
                f"{root}/masked-template.txt",
                f"{root}/migration-receipt.json",
                f"{root}/result.json",
                f"{root}/source-label.json",
                f"{root}/source.txt",
            )
        )
        if result.document_id in checkpoint_ids:
            expected.append(f"{root}/state-checkpoint.json")
        if result.status == "certified":
            expected.extend((f"{root}/catalog-row.json", f"{root}/template.json"))
    return tuple(expected)


def _report(summary: Mapping[str, Any]) -> str:
    counts = cast(Mapping[str, int], summary["classificationCounts"])
    return "\n".join(
        (
            "# Current-schema checkpoint migration",
            "",
            "## Result",
            "",
            f"- Documents: **{summary['documents']}**.",
            f"- Current-host certified without provider calls: "
            f"**{counts.get('current_host_certified', 0)}**.",
            f"- Restored checkpoints requiring a current critic: "
            f"**{counts.get('critic_review_required', 0)}**.",
            f"- Fresh compilation required: **{counts.get('fresh_compilation_required', 0)}**.",
            f"- Source-integrity review: **{counts.get('source_integrity_review', 0)}**.",
            "- Provider requests made by migration: **0**.",
            f"- Wall time: **{summary['wallSeconds']:.3f} seconds**.",
            f"- Peak process RSS: **{summary['peakRssMiB']:.3f} MiB**.",
            "",
            "## Contract",
            "",
            "The legacy committed run, selected document identity, source bytes, labels, and v1 "
            "checkpoints were hash-verified. Each accepted checkpoint was translated structurally "
            "to schema v2, normalized and validated by the current production host, and emitted "
            "as a new atomic committed run. Production itself retains no legacy checkpoint path.",
            "",
        )
    )


def migrate_legacy_checkpoints(
    *,
    config_path: Path,
    legacy_run: Path,
    legacy_commit_sha256: str,
    run_name: str,
) -> Path:
    """Create a main-pipeline-compatible current-schema seed with zero provider calls."""

    project_root = project_root_from_config(config_path)
    project_root = _validated_project_root(project_root=project_root, config_path=config_path)
    config = load_config(config_path)
    if config.workflow.provider_launch_authorized:
        raise ValueError("checkpoint migration requires a provider-locked compilation config")
    migrated_config = _migrated_config(config, run_name=run_name)
    (
        source_rows,
        feature_rows,
        anchor_rows,
        compiler_prompt,
        critic_prompt,
        compiler_repair_prompt,
        critic_audit_prompt,
        critic_plan_prompt,
    ) = _config_inputs(migrated_config, project_root)
    if critic_audit_prompt is None or critic_plan_prompt is None:
        raise ValueError("current migration target lacks required staged critic prompts")
    manifest = build_selection_manifest(
        config=migrated_config,
        source_rows=source_rows,
        feature_rows=feature_rows,
        anchor_rows=anchor_rows,
    )
    pin = PinnedRun(
        path=legacy_run.as_posix(),
        commit_sha256=legacy_commit_sha256,
        allow_prompt_change=True,
    )
    legacy_root, legacy_commit, legacy_artifacts = _legacy_run_root(
        project_root=project_root, pin=pin
    )
    _validate_legacy_lineage(
        legacy_root=legacy_root,
        config=migrated_config,
        manifest=manifest,
    )
    prior_results = _legacy_results(legacy_root=legacy_root, manifest=manifest)
    sources = {str(row["documentId"]): row for row in source_rows}
    features = {str(row["document_id"]): row for row in feature_rows}
    preflight = _preflight_summary(
        config=migrated_config,
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

    implementation_path = Path(__file__).resolve(strict=True)
    transaction_payload = {
        "config": migrated_config.model_dump(mode="json"),
        "selection": manifest.model_dump(mode="json"),
        "productionSourceSha256": _production_source_hash(project_root),
        "migrationImplementationSha256": sha256_file(implementation_path),
        "legacyCommitSha256": legacy_commit_sha256,
    }
    transaction_sha = sha256_bytes(canonical_json_bytes(transaction_payload))
    output_parent = (project_root / migrated_config.output_dir).resolve()
    if output_parent != project_root and project_root not in output_parent.parents:
        raise ValueError("migration output directory escapes the project")
    staged = StagedArtifactRun(
        output_parent=output_parent,
        run_name=migrated_config.run_name,
        transaction_sha256=transaction_sha,
    )
    if staged.completed:
        return staged.final_root
    staged.recover_interrupted_temporary_files()
    staged.publish_json("config.json", migrated_config.model_dump(mode="json"))
    staged.publish_json("preflight.json", preflight)
    staged.publish_json("selection-manifest.json", manifest.model_dump(mode="json"))
    staged.publish_json("resume-contract.json", _resume_contract(project_root))
    staged.publish_bytes("prompts/compiler.md", compiler_prompt.encode("utf-8"))
    staged.publish_bytes("prompts/critic.md", critic_prompt.encode("utf-8"))
    staged.publish_bytes("prompts/critic-audit.md", critic_audit_prompt.encode("utf-8"))
    staged.publish_bytes("prompts/critic-plan.md", critic_plan_prompt.encode("utf-8"))
    if compiler_repair_prompt is not None:
        staged.publish_bytes("prompts/compiler-repair.md", compiler_repair_prompt.encode("utf-8"))
    migration_metadata = {
        "schemaVersion": 1,
        "kind": "explicit_legacy_checkpoint_migration",
        "providerRequests": 0,
        "legacyRun": str(legacy_root.relative_to(project_root)),
        "legacyCommitSha256": legacy_commit_sha256,
        "legacyTransactionSha256": legacy_commit.transaction_sha256,
        "migrationImplementation": str(implementation_path.relative_to(project_root)),
        "migrationImplementationSha256": sha256_file(implementation_path),
        "productionSourceSha256": _production_source_hash(project_root),
    }
    staged.publish_json("migration.json", migration_metadata)

    started = time.perf_counter()
    results: list[ExtractionCaseResult] = []
    templates = []
    receipts: list[MigrationCaseReceipt] = []
    checkpoint_ids: set[str] = set()
    for selection in manifest.rows:
        case_started = time.perf_counter()
        document_id = selection.document_id
        source = sources[document_id]
        raw = cast(str, source["joinedRawText"])
        source_bytes = raw.encode("utf-8")
        source_target = cast(Mapping[str, Any], source["target"])
        legacy_case = legacy_root / "cases" / document_id
        required_case_artifacts = (
            f"cases/{document_id}/result.json",
            f"cases/{document_id}/source-label.json",
            f"cases/{document_id}/source.txt",
        )
        if any(relative not in legacy_artifacts for relative in required_case_artifacts):
            raise ValueError(f"legacy commit omits required case artifacts for {document_id}")
        if read_regular_file_bytes(legacy_case / "source.txt") != source_bytes:
            raise ValueError(f"legacy source bytes differ for {document_id}")
        if json.loads(read_regular_file_bytes(legacy_case / "source-label.json")) != source_target:
            raise ValueError(f"legacy source label differs for {document_id}")
        prior = prior_results[document_id]
        historical_stages = (*prior.compiler_stages, *prior.critic_stages)
        historical_requests = sum(stage.usage.requests for stage in historical_stages)
        historical_cost = sum(
            (stage.usage.estimatedCostUsd for stage in historical_stages), Decimal(0)
        )
        result_sha = legacy_artifacts[f"cases/{document_id}/result.json"]
        risks = all_risk_candidates(raw, (), source_target)
        classification: MigrationClassification
        diagnostic: str | None = None
        legacy_checkpoint_sha: str | None = None
        current_checkpoint_sha: str | None = None
        current_template_sha: str | None = None
        retained_final = False
        masked = raw

        integrity_issues = source_template_integrity_issues(raw, source_target)
        if integrity_issues:
            classification = "source_integrity_review"
            diagnostic = "; ".join(integrity_issues)
            result = _migration_result(
                document_id=document_id,
                status="review_required",
                reasons=tuple(
                    "source template integrity requires review: " + issue
                    for issue in integrity_issues
                ),
                template_sha256=None,
                risk_candidates=risks,
                elapsed_seconds=time.perf_counter() - case_started,
                legacy_run=legacy_root.name,
                source_integrity_review=True,
            )
        else:
            checkpoint_relative = f"cases/{document_id}/state-checkpoint.json"
            try:
                if checkpoint_relative not in legacy_artifacts:
                    raise ValueError("legacy committed case has no state checkpoint")
                legacy_checkpoint_sha = legacy_artifacts[checkpoint_relative]
                legacy_checkpoint = LegacyExtractionStateCheckpointV1.model_validate_json(
                    read_regular_file_bytes(legacy_case / "state-checkpoint.json"), strict=True
                )
                translated = _translate_legacy_checkpoint(legacy_checkpoint)
                (
                    drafts,
                    assessment,
                    semantic_only,
                    coherence_constraints,
                    revisions,
                ) = _restore_state_checkpoint(
                    checkpoint_payload=json_artifact_bytes(translated.model_dump(mode="json")),
                    document_id=document_id,
                    raw=raw,
                    source_target=source_target,
                )
            except Exception as error:
                classification = "fresh_compilation_required"
                diagnostic = f"{type(error).__name__}: {error}"
                result = _migration_result(
                    document_id=document_id,
                    status="rejected",
                    reasons=(
                        "fresh compilation required after strict v1 migration: " + diagnostic,
                    ),
                    template_sha256=None,
                    risk_candidates=risks,
                    elapsed_seconds=time.perf_counter() - case_started,
                    legacy_run=legacy_root.name,
                )
            else:
                current_checkpoint = _state_checkpoint(
                    document_id=document_id,
                    raw=raw,
                    source_target=source_target,
                    drafts=drafts,
                    assessment=assessment,
                    semantic_only_target_facts=semantic_only,
                    coherence_constraints=coherence_constraints,
                    critic_outputs=revisions,
                )
                checkpoint_payload = json_artifact_bytes(current_checkpoint.model_dump(mode="json"))
                current_checkpoint_sha = sha256_bytes(checkpoint_payload)
                checkpoint_ids.add(document_id)
                staged.publish_bytes(
                    f"cases/{document_id}/state-checkpoint.json", checkpoint_payload
                )
                masked = masked_source(raw, drafts)
                candidate_stage = prior.critic_stages[-1] if prior.critic_stages else None
                try:
                    if (
                        prior.status != "certified"
                        or candidate_stage is None
                        or candidate_stage.status != "success"
                        or candidate_stage.output is None
                    ):
                        raise ValueError("legacy case lacks a final successful critic receipt")
                    final_review = CriticAgentOutput.model_validate_json(
                        json.dumps(
                            candidate_stage.output,
                            ensure_ascii=False,
                            separators=(",", ":"),
                        )
                    )
                    if final_review.verdict != "pass":
                        raise ValueError("legacy case does not end in a critic pass")
                    template = certify_template(
                        raw=raw,
                        document_id=document_id,
                        feature=features[document_id],
                        source_target=source_target,
                        assessment=assessment,
                        drafts=drafts,
                        risks=risks,
                        critic_outputs=(*revisions, final_review),
                        semantic_only_target_facts=semantic_only,
                        coherence_constraints=coherence_constraints,
                    )
                except Exception as error:
                    classification = "critic_review_required"
                    diagnostic = f"{type(error).__name__}: {error}"
                    result = _migration_result(
                        document_id=document_id,
                        status="rejected",
                        reasons=("current independent critic review required: " + diagnostic,),
                        template_sha256=None,
                        risk_candidates=risks,
                        elapsed_seconds=time.perf_counter() - case_started,
                        legacy_run=legacy_root.name,
                    )
                else:
                    classification = "current_host_certified"
                    template_payload = json_artifact_bytes(template.model_dump(mode="json"))
                    current_template_sha = sha256_bytes(template_payload)
                    retained_final = True
                    result = _migration_result(
                        document_id=document_id,
                        status="certified",
                        reasons=(),
                        template_sha256=current_template_sha,
                        risk_candidates=risks,
                        elapsed_seconds=time.perf_counter() - case_started,
                        legacy_run=legacy_root.name,
                        final_critic_stage=candidate_stage,
                    )
                    templates.append(template)
                    staged.publish_bytes(f"cases/{document_id}/template.json", template_payload)
                    staged.publish_json(
                        f"cases/{document_id}/catalog-row.json", template_summary(template)
                    )

        receipt = MigrationCaseReceipt(
            schema_version=1,
            document_id=document_id,
            classification=classification,
            provider_requests=0,
            legacy_run=legacy_root.name,
            legacy_commit_sha256=legacy_commit_sha256,
            legacy_result_sha256=result_sha,
            legacy_checkpoint_sha256=legacy_checkpoint_sha,
            current_checkpoint_sha256=current_checkpoint_sha,
            current_template_sha256=current_template_sha,
            current_risk_candidates=len(risks),
            retained_final_critic_receipt=retained_final,
            historical_requests=historical_requests,
            historical_estimated_cost_usd=historical_cost,
            diagnostic=diagnostic,
        )
        staged.publish_bytes(f"cases/{document_id}/source.txt", source_bytes)
        staged.publish_json(f"cases/{document_id}/source-label.json", source_target)
        staged.publish_bytes(f"cases/{document_id}/masked-template.txt", masked.encode("utf-8"))
        staged.publish_json(f"cases/{document_id}/result.json", result.model_dump(mode="json"))
        staged.publish_json(
            f"cases/{document_id}/migration-receipt.json", receipt.model_dump(mode="json")
        )
        results.append(result)
        receipts.append(receipt)

    classification_counts = Counter(row.classification for row in receipts)
    status_counts = Counter(row.status for row in results)
    summary = {
        "schemaVersion": 1,
        "phase": "explicit_legacy_checkpoint_migration",
        "documents": len(results),
        "classificationCounts": dict(sorted(classification_counts.items())),
        "statusCounts": dict(sorted(status_counts.items())),
        "currentSchemaCheckpoints": len(checkpoint_ids),
        "certifiedTemplates": len(templates),
        "providerRequests": 0,
        "historicalRequests": sum(row.historical_requests for row in receipts),
        "historicalEstimatedCostUsd": str(
            sum((row.historical_estimated_cost_usd for row in receipts), Decimal(0))
        ),
        "retainedFinalCriticReceipts": sum(row.retained_final_critic_receipt for row in receipts),
        "wallSeconds": time.perf_counter() - started,
        "peakRssMiB": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024,
        "completedAt": datetime.now(UTC).isoformat(),
    }
    staged.publish_bytes(
        "results.jsonl",
        _jsonl_bytes([row.model_dump(mode="json") for row in results]),
    )
    staged.publish_bytes(
        "catalog.jsonl", _jsonl_bytes([template_summary(row) for row in templates])
    )
    staged.publish_json("summary.json", summary)
    staged.publish_bytes("REPORT.md", _report(summary).encode("utf-8"))
    staged.commit(
        expected_artifacts=_expected_artifacts(
            results=results,
            checkpoint_ids=checkpoint_ids,
            has_compiler_repair_prompt=compiler_repair_prompt is not None,
        ),
        metadata={
            "schemaVersion": 1,
            "phase": "explicit_legacy_checkpoint_migration",
            "documents": len(results),
            "certified": len(templates),
            "providerRequests": 0,
        },
    )
    return staged.final_root
