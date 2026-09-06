"""Fifty-case compiler-first raw-OCR rewrite benchmark.

This experiment prepares every semantic rewrite contract locally, delegates only bounded residual
spans to one editor call, and performs one independent compact review.  It deliberately withholds
training publication: declared-contract success still requires the subsequent full-text manual
audit because the current semantic plan does not yet enumerate every raw-only auxiliary slot.
"""

from __future__ import annotations

import asyncio
import json
import time
from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from decimal import Decimal
from importlib.metadata import version
from pathlib import Path
from typing import Any, cast

from openai import AsyncOpenAI
from pydantic import JsonValue

from document_ocr.atomic import read_regular_file_bytes
from document_ocr.hashing import canonical_json_bytes, sha256_bytes, sha256_file
from document_ocr.synthesis.config import SynthesisRawTextHybridBatchConfig
from document_ocr.synthesis.linguistic_completion_pipeline import DocumentLinguisticPlan
from document_ocr.synthesis.linguistic_probe_runtime import load_provider_key
from document_ocr.synthesis.raw_text_hybrid_probe import (
    HybridCaseResult,
    HybridModelStage,
    _artifact_inventory,
    _CompiledCase,
    _empty_usage,
    _model_pair,
    _run_model_case,
    audit_literal_coverage,
    compile_case,
)
from document_ocr.synthesis.raw_text_rewrite_cycle_probe import (
    RewriteCycleBundle,
    _combine_usage,
    build_rewrite_contract_bundle,
    build_target_integrity_resources,
    prepare_rewrite_state,
)
from document_ocr.synthesis.raw_text_rewrite_probe import (
    _load_jsonl,
    _resolve_pinned_file,
    unified_text_diff,
)
from document_ocr.synthesis.run_safety import StagedArtifactRun
from document_ocr.training.config import resolve_config_path

_IMPLEMENTATION_PATH = Path(__file__).resolve(strict=True)
_OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"


@dataclass(frozen=True, slots=True)
class _SelectedCase:
    source: Mapping[str, Any]
    target: Mapping[str, Any]
    plan: DocumentLinguisticPlan
    feature: Mapping[str, JsonValue]


def _validate_linguistic_run(project_root: Path, config: SynthesisRawTextHybridBatchConfig) -> Path:
    configured = config.inputs.linguistic_completion_run
    root = resolve_config_path(project_root, configured.path)
    commit = root / "_COMMIT.json"
    if root.is_symlink() or not root.is_dir() or commit.is_symlink() or not commit.is_file():
        raise ValueError(f"linguistic completion run is not a committed directory: {root}")
    if sha256_file(commit) != configured.commit_sha256:
        raise ValueError("linguistic completion run commit SHA-256 differs")
    StagedArtifactRun(
        output_parent=root.parent,
        run_name=root.name,
        transaction_sha256=configured.transaction_sha256,
    ).validate_committed_run()
    return root.resolve(strict=True)


def _load_inputs(
    project_root: Path, config: SynthesisRawTextHybridBatchConfig
) -> tuple[
    tuple[_SelectedCase, ...],
    tuple[dict[str, Any], ...],
    tuple[dict[str, Any], ...],
    bytes,
    bytes,
    str,
]:
    linguistic_root = _validate_linguistic_run(project_root, config)
    results_path = _resolve_pinned_file(
        project_root,
        config.inputs.linguistic_results.path,
        config.inputs.linguistic_results.sha256,
        label="hybrid batch linguistic results",
    )
    targets_path = _resolve_pinned_file(
        project_root,
        config.inputs.synthetic_targets.path,
        config.inputs.synthetic_targets.sha256,
        label="hybrid batch synthetic targets",
    )
    summary_path = _resolve_pinned_file(
        project_root,
        config.inputs.linguistic_summary.path,
        config.inputs.linguistic_summary.sha256,
        label="hybrid batch linguistic summary",
    )
    plan_path = linguistic_root / "planning/document-plans.jsonl"
    if plan_path.is_symlink() or not plan_path.is_file():
        raise ValueError(f"linguistic completion run has no regular document plan: {plan_path}")
    plan_path = plan_path.resolve(strict=True)
    for path in (results_path, targets_path, summary_path, plan_path):
        if not path.is_relative_to(linguistic_root):
            raise ValueError("linguistic input is outside the pinned committed run")
    source_path = _resolve_pinned_file(
        project_root,
        config.inputs.source_corpus.path,
        config.inputs.source_corpus.sha256,
        label="hybrid batch source corpus",
    )
    feature_path = _resolve_pinned_file(
        project_root,
        config.inputs.document_features.path,
        config.inputs.document_features.sha256,
        label="hybrid batch document features",
    )
    editor_prompt_path = _resolve_pinned_file(
        project_root,
        config.prompts.editor.path,
        config.prompts.editor.sha256,
        label="hybrid batch editor prompt",
    )
    reviewer_prompt_path = _resolve_pinned_file(
        project_root,
        config.prompts.reviewer.path,
        config.prompts.reviewer.sha256,
        label="hybrid batch reviewer prompt",
    )
    results = _load_jsonl(
        results_path,
        records=config.inputs.linguistic_results.records,
        key="baseDocumentId",
        label="hybrid batch linguistic results",
    )
    targets = _load_jsonl(
        targets_path,
        records=config.inputs.synthetic_targets.records,
        key="baseDocumentId",
        label="hybrid batch synthetic targets",
    )
    raw_plans = _load_jsonl(
        plan_path,
        records=config.inputs.linguistic_results.records,
        key="baseDocumentId",
        label="hybrid batch linguistic plans",
    )
    plans = tuple(
        DocumentLinguisticPlan.model_validate_json(canonical_json_bytes(row), strict=True)
        for row in raw_plans
    )
    sources = _load_jsonl(
        source_path,
        records=config.inputs.source_corpus.records,
        key="documentId",
        label="hybrid batch source corpus",
    )
    features = _load_jsonl(
        feature_path,
        records=config.inputs.document_features.records,
        key="document_id",
        label="hybrid batch document features",
    )
    result_by_id = {cast(str, row["baseDocumentId"]): row for row in results}
    target_by_id = {cast(str, row["baseDocumentId"]): row for row in targets}
    plan_by_id = {row.baseDocumentId: row for row in plans}
    source_by_id = {cast(str, row["documentId"]): row for row in sources}
    feature_by_id = {cast(str, row["document_id"]): row for row in features}
    if tuple(result_by_id) != tuple(target_by_id) or tuple(result_by_id) != tuple(plan_by_id):
        raise ValueError("linguistic result, target, and plan order differs")
    for document_id, result in result_by_id.items():
        target = target_by_id[document_id]
        plan = plan_by_id[document_id]
        if result.get("status") != "success" or result.get("target") != target.get("target"):
            raise ValueError(f"linguistic result/target mismatch for {document_id}")
        if target.get("scenarioId") != plan.scenarioId:
            raise ValueError(f"linguistic plan/target scenario mismatch for {document_id}")
        target_sha = sha256_bytes(canonical_json_bytes(target["target"]))
        if target.get("targetSha256") != target_sha or result.get("targetSha256") != target_sha:
            raise ValueError(f"linguistic target SHA-256 mismatch for {document_id}")
    selected: list[_SelectedCase] = []
    for document_id in target_by_id:
        try:
            source = source_by_id[document_id]
            target = target_by_id[document_id]
            plan = plan_by_id[document_id]
            feature = feature_by_id[document_id]
        except KeyError as error:
            raise ValueError(f"hybrid case is absent from pinned inputs: {document_id}") from error
        raw_text = source.get("joinedRawText")
        if not isinstance(raw_text, str) or source.get("joinedRawTextSha256") != sha256_bytes(
            raw_text.encode("utf-8")
        ):
            raise ValueError(f"source raw text fails SHA-256: {document_id}")
        selected.append(
            _SelectedCase(
                source=source,
                target=target,
                plan=plan,
                feature=cast(Mapping[str, JsonValue], feature),
            )
        )
    return (
        tuple(selected),
        sources,
        targets,
        read_regular_file_bytes(editor_prompt_path),
        read_regular_file_bytes(reviewer_prompt_path),
        sha256_file(plan_path),
    )


def _publish_jsonl(
    staged: StagedArtifactRun, path: str, rows: Sequence[Mapping[str, JsonValue]]
) -> None:
    staged.publish_bytes(
        path,
        b"".join(canonical_json_bytes(row) + b"\n" for row in rows),
    )


def _publish_case_checkpoint(
    *,
    staged: StagedArtifactRun,
    compiled: _CompiledCase,
    result: HybridCaseResult,
    stages: Sequence[HybridModelStage],
) -> None:
    """Persist one completed provider transaction atomically before reporting progress."""

    final_text = compiled.workspace.current_text
    payload = {
        "schemaVersion": 1,
        "documentId": result.documentId,
        "finalText": final_text,
        "diff": unified_text_diff(compiled.workspace.original_text, final_text),
        "result": result.model_dump(mode="json"),
        "stages": [row.model_dump(mode="json") for row in stages],
    }
    staged.publish_bytes(
        f"cases/{result.documentId}/checkpoint.json",
        canonical_json_bytes(payload),
    )


def _load_case_checkpoint(
    *,
    staged: StagedArtifactRun,
    compiled: _CompiledCase,
) -> tuple[HybridCaseResult, tuple[HybridModelStage, ...]] | None:
    path = staged.stage_root / f"cases/{compiled.bundle.result.documentId}/checkpoint.json"
    if not path.exists():
        return None
    if path.is_symlink() or not path.is_file():
        raise ValueError(f"hybrid case checkpoint is not a regular file: {path}")
    value = json.loads(read_regular_file_bytes(path))
    if not isinstance(value, dict) or set(value) != {
        "schemaVersion",
        "documentId",
        "finalText",
        "diff",
        "result",
        "stages",
    }:
        raise ValueError(f"hybrid case checkpoint has an invalid shape: {path}")
    document_id = compiled.bundle.result.documentId
    if value["schemaVersion"] != 1 or value["documentId"] != document_id:
        raise ValueError(f"hybrid case checkpoint identity differs: {path}")
    final_text = value["finalText"]
    if not isinstance(final_text, str):
        raise ValueError(f"hybrid case checkpoint final text is invalid: {path}")
    if value["diff"] != unified_text_diff(compiled.workspace.original_text, final_text):
        raise ValueError(f"hybrid case checkpoint diff is invalid: {path}")
    stages_value = value["stages"]
    if not isinstance(stages_value, list):
        raise ValueError(f"hybrid case checkpoint stages are invalid: {path}")
    result, stages = _decode_checkpoint_models(value["result"], stages_value)
    if result.documentId != document_id:
        raise ValueError(f"hybrid case checkpoint result identity differs: {path}")
    if result.outputTextSha256 != sha256_bytes(final_text.encode("utf-8")):
        raise ValueError(f"hybrid case checkpoint final-text hash differs: {path}")
    compiled.workspace.current_text = final_text
    return result, stages


def _decode_checkpoint_models(
    result_value: Any,
    stages_value: Sequence[Any],
) -> tuple[HybridCaseResult, tuple[HybridModelStage, ...]]:
    """Restore strict models through their JSON boundary.

    Checkpoints are persisted with ``mode="json"``.  Revalidating the parsed Python mapping
    directly under strict mode rejects valid JSON arrays for tuple fields and JSON strings for
    Decimal fields.  ``model_validate_json`` applies the model's strict JSON semantics instead.
    """

    result = HybridCaseResult.model_validate_json(
        canonical_json_bytes(result_value),
        strict=True,
    )
    stages = tuple(
        HybridModelStage.model_validate_json(canonical_json_bytes(row), strict=True)
        for row in stages_value
    )
    return result, stages


def _plot_bytes(
    results: Sequence[HybridCaseResult],
    stages_by_document: Mapping[str, Sequence[HybridModelStage]],
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

    case_rows = [
        {
            "document": row.documentId[:12],
            "status": row.status,
            "input": row.usage.inputTokens,
            "reasoning": row.usage.reasoningTokens,
            "visible": row.usage.visibleOutputTokens,
            "cost": float(row.usage.estimatedCostUsd),
            "residual_fraction": row.residualLines / row.sourceLines,
        }
        for row in results
    ]
    frame = pd.DataFrame(case_rows)
    figure, axes = plt.subplots(1, 2, figsize=(14, 5.5))
    order = list(frame.status.value_counts().index)
    sns.countplot(data=frame, x="status", order=order, ax=axes[0], color="#4C78A8")
    sns.histplot(data=frame, x="residual_fraction", bins=12, ax=axes[1], color="#F28E2B")
    axes[0].set(title="Declared-contract outcomes", xlabel="", ylabel="documents")
    axes[0].tick_params(axis="x", rotation=20)
    axes[1].set(title="Residual OCR fraction", xlabel="residual lines / source lines")
    render("01_outcomes_and_residual_coverage.png", figure)

    token_frame = frame.melt(
        id_vars="document",
        value_vars=["input", "reasoning", "visible"],
        var_name="token_type",
        value_name="tokens",
    )
    figure, axis = plt.subplots(figsize=(15, 6))
    sns.boxplot(data=token_frame, x="token_type", y="tokens", ax=axis, color="#59A14F")
    sns.stripplot(data=token_frame, x="token_type", y="tokens", ax=axis, color="black", alpha=0.35)
    axis.set(title="Per-document provider token distributions", xlabel="", ylabel="tokens")
    render("02_token_distributions.png", figure)

    stage_rows = [
        {
            "document": document_id[:12],
            "stage": stage.stage,
            "duration_seconds": stage.completedAtUnixSeconds - stage.startedAtUnixSeconds,
            "cost": float(stage.usage.estimatedCostUsd),
        }
        for document_id, stages in stages_by_document.items()
        for stage in stages
    ]
    if stage_rows:
        stage_frame = pd.DataFrame(stage_rows)
        figure, axes = plt.subplots(1, 2, figsize=(14, 5.5))
        sns.boxplot(data=stage_frame, x="stage", y="duration_seconds", ax=axes[0])
        sns.boxplot(data=stage_frame, x="stage", y="cost", ax=axes[1])
        axes[0].set(title="Stage latency", xlabel="", ylabel="seconds")
        axes[1].set(title="Stage estimated cost", xlabel="", ylabel="USD")
        for axis in axes:
            axis.tick_params(axis="x", rotation=15)
        render("03_stage_latency_and_cost.png", figure)

    figure, axis = plt.subplots(figsize=(8, 6))
    sns.scatterplot(
        data=frame,
        x="residual_fraction",
        y="cost",
        hue="status",
        ax=axis,
        s=65,
    )
    axis.set(
        title="Residual coverage versus estimated cost",
        xlabel="residual lines / source lines",
        ylabel="USD per document",
    )
    render("04_residual_fraction_vs_cost.png", figure)
    return plots


def _report(
    *,
    summary: Mapping[str, JsonValue],
    results: Sequence[HybridCaseResult],
    stages_by_document: Mapping[str, Sequence[HybridModelStage]],
) -> str:
    expensive = sorted(results, key=lambda row: row.usage.estimatedCostUsd, reverse=True)[:10]
    durations = {
        document_id: sum(row.completedAtUnixSeconds - row.startedAtUnixSeconds for row in stages)
        for document_id, stages in stages_by_document.items()
    }
    lines = [
        "# Compiler-first raw-OCR 50-document model arm",
        "",
        "## Scope and safety boundary",
        "",
        f"- Model: `{summary['model']}`.",
        f"- Documents: **{summary['cases']}**.",
        f"- Declared-contract passes: **{summary['declaredContractValidatedCases']}**.",
        f"- Compiler-blocked: **{summary['compilerBlockedCases']}**; model-call failures: "
        f"**{summary['callFailedCases']}**.",
        "- Training records published: **0**.",
        "- Every declared-contract pass remains pending a full-text manual audit because raw-only "
        "auxiliary-slot coverage is not yet complete.",
        "",
        "## Usage",
        "",
        f"- Requests: **{summary['requests']}**.",
        f"- Provider attempts / failed attempts: **{summary['providerAttempts']} / "
        f"{summary['failedProviderAttempts']}**.",
        f"- Input/cache-read tokens: **{summary['inputTokens']} / {summary['cacheReadTokens']}**.",
        f"- Reasoning/visible output tokens: **{summary['reasoningTokens']} / "
        f"{summary['visibleOutputTokens']}**.",
        f"- Estimated total cost: **${summary['estimatedCostUsd']}**; estimated attempted cost per "
        f"1,000: **${summary['estimatedCostPerThousandUsd']}**.",
        f"- Provider-reported billed total / attempted cost per 1,000: "
        f"**${summary['providerReportedCostUsd']} / "
        f"${summary['providerReportedCostPerThousandUsd']}**.",
        f"- Wall time: **{summary['wallSeconds']} s**; throughput: "
        f"**{summary['throughputCasesPerHour']} documents/hour**.",
        "",
        "## Highest-cost documents",
        "",
        "| Document | Status | Input | Reasoning | Visible | Cost | Model-call time |",
        "|---|---|---:|---:|---:|---:|---:|",
    ]
    for row in expensive:
        lines.append(
            f"| `{row.documentId}` | {row.status} | {row.usage.inputTokens:,} | "
            f"{row.usage.reasoningTokens:,} | {row.usage.visibleOutputTokens:,} | "
            f"${row.usage.estimatedCostUsd} | {durations.get(row.documentId, 0.0):.2f}s |"
        )
    lines.extend(
        (
            "",
            "## Artifact guide",
            "",
            "Each `cases/<document>/` directory contains the source text, locally prepared "
            "contract, deterministic intermediate, residual work items and spans, exact "
            "provider-visible messages, final text, diff, host audit, and compact review. "
            "`generation/results.jsonl` is the machine-readable arm output.",
            "",
            "This artifact is an evaluation arm, not a training dataset. A later paired analysis "
            "must compare GLM and Luna outputs and record the manual full-text decisions.",
            "",
        )
    )
    return "\n".join(lines)


def run_raw_text_hybrid_batch(
    *,
    project_root: Path,
    config_path: Path,
    config: SynthesisRawTextHybridBatchConfig,
) -> dict[str, JsonValue]:
    (
        selected,
        sources,
        targets,
        editor_prompt_bytes,
        reviewer_prompt_bytes,
        plan_sha256,
    ) = _load_inputs(project_root, config)
    editor_prompt = editor_prompt_bytes.decode("utf-8")
    reviewer_prompt = reviewer_prompt_bytes.decode("utf-8")
    coverage, coverage_rows = audit_literal_coverage(
        sources,
        targets,
        limit=config.workflow.audit_documents,
    )
    transaction = {
        "schemaVersion": 2,
        "runId": config.run.run_id,
        "configSha256": sha256_file(config_path),
        "linguisticRunTransactionSha256": (
            config.inputs.linguistic_completion_run.transaction_sha256
        ),
        "linguisticResultsSha256": config.inputs.linguistic_results.sha256,
        "linguisticDocumentPlansSha256": plan_sha256,
        "syntheticTargetsSha256": config.inputs.synthetic_targets.sha256,
        "sourceCorpusSha256": config.inputs.source_corpus.sha256,
        "documentFeaturesSha256": config.inputs.document_features.sha256,
        "editorPromptSha256": config.prompts.editor.sha256,
        "reviewerPromptSha256": config.prompts.reviewer.sha256,
        "caseDocumentIds": [row.plan.baseDocumentId for row in selected],
        "implementationSha256": sha256_file(_IMPLEMENTATION_PATH),
        "compilerImplementationSha256": sha256_file(
            Path(__file__).with_name("raw_text_hybrid_probe.py")
        ),
        "contractImplementationSha256": sha256_file(
            Path(__file__).with_name("raw_text_rewrite_cycle_probe.py")
        ),
        "runtime": {
            "editorProviderKind": config.providers.editor.kind,
            "editorModel": config.providers.editor.model,
            "editorReasoningEffort": config.providers.editor.reasoning_effort,
            "reviewerProviderKind": config.providers.reviewer.kind,
            "reviewerModel": config.providers.reviewer.model,
            "reviewerReasoningEffort": config.providers.reviewer.reasoning_effort,
            "pydanticAiVersion": version("pydantic-ai-slim"),
            "openaiVersion": version("openai"),
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
    resources = build_target_integrity_resources(
        project_root=project_root,
        config=config,
        source_rows=sources,
        staged=staged,
    )
    contracts: list[RewriteCycleBundle] = []
    compiled_cases = []
    for case_number, row in enumerate(selected, start=1):
        state = prepare_rewrite_state(
            case_number=case_number,
            source_row=row.source,
            target_row=row.target,
            linguistic_plan=row.plan,
            feature=row.feature,
            config=config,
            target_integrity_resources=resources,
        )
        contract = build_rewrite_contract_bundle(state)
        contracts.append(contract)
        compiled_cases.append(
            compile_case(
                contract,
                context_lines=config.workflow.context_lines_per_residual,
                merge_gap_lines=config.workflow.merge_residual_gap_lines,
            )
        )

    for contract, compiled in zip(contracts, compiled_cases, strict=True):
        prefix = f"cases/{compiled.bundle.result.documentId}"
        staged.publish_json(f"{prefix}/contract.json", contract.model_dump(mode="json"))
        staged.publish_bytes(f"{prefix}/source.txt", compiled.workspace.original_text.encode())
        staged.publish_bytes(
            f"{prefix}/after-deterministic.txt", compiled.workspace.current_text.encode()
        )
        staged.publish_json(
            f"{prefix}/work-items.json",
            [row.model_dump(mode="json") for row in compiled.work_items],
        )
        staged.publish_json(
            f"{prefix}/residual-spans.json",
            [row.model_dump(mode="json") for row in compiled.spans],
        )
        staged.publish_json(
            f"{prefix}/compiler-edits.json",
            [row.model_dump(mode="json") for row in compiled.compiler_edits],
        )

    stages_by_document: dict[str, tuple[HybridModelStage, ...]] = {}

    async def execute() -> tuple[HybridCaseResult, ...]:
        output: list[HybridCaseResult | None] = [None] * len(compiled_cases)
        for index, compiled in enumerate(compiled_cases):
            checkpoint = _load_case_checkpoint(staged=staged, compiled=compiled)
            if checkpoint is None:
                continue
            result, stages = checkpoint
            output[index] = result
            stages_by_document[result.documentId] = stages
        processed = sum(row is not None for row in output)
        pending_indices = tuple(index for index, row in enumerate(output) if row is None)
        if not pending_indices:
            return tuple(cast(HybridCaseResult, row) for row in output)
        editor_provider = config.providers.editor
        reviewer_provider = config.providers.reviewer
        key = load_provider_key(project_root, config.environment_file, editor_provider.api_key_env)
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
        case_limiter = asyncio.Semaphore(config.workflow.max_concurrent_cases)
        progress_lock = asyncio.Lock()
        started = time.perf_counter()

        async def run_one(index: int) -> None:
            nonlocal processed
            compiled = compiled_cases[index]
            async with case_limiter:
                result, stages = await _run_model_case(
                    compiled=compiled,
                    editor_model=editor_model,
                    reviewer_model=reviewer_model,
                    providers=config.providers,
                    editor_prompt=editor_prompt,
                    editor_prompt_sha256=config.prompts.editor.sha256,
                    reviewer_prompt=reviewer_prompt,
                    reviewer_prompt_sha256=config.prompts.reviewer.sha256,
                    audited_reference_text=None,
                )
            async with progress_lock:
                _publish_case_checkpoint(
                    staged=staged,
                    compiled=compiled,
                    result=result,
                    stages=stages,
                )
                output[index] = result
                stages_by_document[result.documentId] = stages
                processed += 1
                elapsed = time.perf_counter() - started
                rate = processed / elapsed if elapsed else 0.0
                print(
                    json.dumps(
                        {
                            "command": "run-raw-text-hybrid-batch",
                            "phase": "compile_edit_review",
                            "processed_cases": processed,
                            "remaining_cases": len(compiled_cases) - processed,
                            "declared_contract_passes": sum(
                                row is not None and row.status == "quality_validated"
                                for row in output
                            ),
                            "elapsed_seconds": round(elapsed, 3),
                            "throughput_cases_per_hour": round(rate * 3600.0, 3),
                            "eta_seconds": (
                                round((len(compiled_cases) - processed) / rate, 3) if rate else None
                            ),
                            "status": "progress",
                        },
                        allow_nan=False,
                        sort_keys=True,
                    ),
                    flush=True,
                )

        try:
            await asyncio.gather(*(run_one(index) for index in pending_indices))
        finally:
            await client.close()
        if any(row is None for row in output):
            raise RuntimeError("hybrid worker pool returned an incomplete result set")
        return tuple(cast(HybridCaseResult, row) for row in output)

    started = time.perf_counter()
    results = asyncio.run(execute())
    wall_seconds = time.perf_counter() - started
    for compiled, result in zip(compiled_cases, results, strict=True):
        prefix = f"cases/{result.documentId}"
        staged.publish_bytes(f"{prefix}/final.txt", compiled.workspace.current_text.encode())
        staged.publish_bytes(
            f"{prefix}/diff.patch",
            unified_text_diff(
                compiled.workspace.original_text, compiled.workspace.current_text
            ).encode(),
        )
        staged.publish_json(
            f"{prefix}/stages.json",
            [row.model_dump(mode="json") for row in stages_by_document[result.documentId]],
        )
        staged.publish_json(f"{prefix}/result.json", result.model_dump(mode="json"))
    _publish_jsonl(
        staged,
        "generation/results.jsonl",
        [
            cast(
                Mapping[str, JsonValue],
                {
                    **row.model_dump(mode="json"),
                    "fullTextManualAuditStatus": "pending"
                    if row.status == "quality_validated"
                    else "not_applicable_until_contract_passes",
                    "trainingEligible": False,
                },
            )
            for row in results
        ],
    )
    total_usage = _combine_usage([row.usage for row in results]) if results else _empty_usage()
    declared_passes = sum(row.status == "quality_validated" for row in results)
    rate = len(results) / wall_seconds if wall_seconds else 0.0
    estimated_per_thousand = total_usage.estimatedCostUsd / Decimal(len(results)) * 1000
    provider_per_thousand = (
        total_usage.providerReportedCostUsd / Decimal(len(results)) * 1000
        if total_usage.providerReportedCostUsd is not None
        else None
    )
    downstream = Counter(total_usage.downstreamProviders)
    summary: dict[str, JsonValue] = {
        "schemaVersion": 2,
        "runId": config.run.run_id,
        "status": "complete",
        "model": config.providers.editor.model,
        "cases": len(results),
        "declaredContractValidatedCases": declared_passes,
        "manualFullTextAuditPendingCases": declared_passes,
        "needsReviewCases": sum(row.status == "needs_review" for row in results),
        "compilerBlockedCases": sum(row.status == "compiler_blocked" for row in results),
        "callFailedCases": sum(row.status == "call_failed" for row in results),
        "requests": total_usage.requests,
        "providerAttempts": sum(len(rows) for rows in stages_by_document.values()),
        "failedProviderAttempts": sum(
            row.errorType is not None for rows in stages_by_document.values() for row in rows
        ),
        "inputTokens": total_usage.inputTokens,
        "cacheReadTokens": total_usage.cacheReadTokens,
        "cacheWriteTokens": total_usage.cacheWriteTokens,
        "outputTokens": total_usage.outputTokens,
        "reasoningTokens": total_usage.reasoningTokens,
        "visibleOutputTokens": total_usage.visibleOutputTokens,
        "estimatedCostUsd": str(total_usage.estimatedCostUsd),
        "estimatedCostPerThousandUsd": str(estimated_per_thousand),
        "providerReportedCostUsd": (
            str(total_usage.providerReportedCostUsd)
            if total_usage.providerReportedCostUsd is not None
            else None
        ),
        "providerReportedCostPerThousandUsd": (
            str(provider_per_thousand) if provider_per_thousand is not None else None
        ),
        "downstreamProviderCounts": cast(JsonValue, dict(sorted(downstream.items()))),
        "wallSeconds": round(wall_seconds, 6),
        "throughputCasesPerHour": round(rate * 3600.0, 6),
        "trainingRecordsPublished": False,
    }
    staged.publish_json("summary.json", summary)
    staged.publish_bytes(
        "REPORT.md",
        _report(
            summary=summary,
            results=results,
            stages_by_document=stages_by_document,
        ).encode(),
    )
    for name, payload in _plot_bytes(results, stages_by_document).items():
        staged.publish_bytes(f"plots/{name}", payload)
    artifacts = _artifact_inventory(staged.stage_root)
    staged.commit(
        expected_artifacts=artifacts,
        metadata={
            "schemaVersion": 2,
            "cases": len(results),
            "declaredContractValidatedCases": declared_passes,
            "manualFullTextAuditPendingCases": declared_passes,
            "trainingRecordsPublished": False,
        },
    )
    output = dict(summary)
    output["artifactRoot"] = str(staged.final_root)
    output["commitSha256"] = sha256_file(staged.final_root / "_COMMIT.json")
    return output
