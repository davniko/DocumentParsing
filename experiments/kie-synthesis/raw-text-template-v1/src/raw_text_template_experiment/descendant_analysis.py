from __future__ import annotations

import io
import json
from collections import Counter
from collections.abc import Mapping
from pathlib import Path
from typing import Any, cast

import pandas as pd  # type: ignore[import-untyped]
import seaborn as sns  # type: ignore[import-untyped]
import yaml
from document_ocr.atomic import read_regular_file_bytes
from document_ocr.hashing import canonical_json_bytes, sha256_bytes, sha256_file
from document_ocr.synthesis.run_safety import StagedArtifactRun
from matplotlib import pyplot as plt
from matplotlib.figure import Figure

plt.switch_backend("Agg")


_IMPLEMENTATION_PATH = Path(__file__).resolve(strict=True)
_PALETTE = {
    "deterministic": "#1976D2",
    "agent": "#EF6C00",
    "existing_linguistic_target_carrier_restored": "#00897B",
    "controlled_source_variant": "#8E24AA",
    "passed": "#2E7D32",
    "concern": "#C62828",
    "pass_with_source_limitation": "#F9A825",
}
_ORIGIN_LABELS = {
    "existing_linguistic_target_carrier_restored": "Existing synthetic target",
    "controlled_source_variant": "Controlled source variant",
}


def _json(path: Path) -> dict[str, Any]:
    value = json.loads(read_regular_file_bytes(path))
    if not isinstance(value, dict):
        raise ValueError(f"JSON artifact is not an object: {path}")
    return value


def _jsonl(path: Path) -> tuple[dict[str, Any], ...]:
    rows = []
    for line_number, line in enumerate(read_regular_file_bytes(path).splitlines(), start=1):
        value = json.loads(line)
        if not isinstance(value, dict):
            raise ValueError(f"JSONL row {line_number} is not an object: {path}")
        rows.append(value)
    return tuple(rows)


def _validate_committed(root: Path) -> tuple[Path, dict[str, Any], str]:
    resolved = root.resolve(strict=True)
    if resolved.is_symlink() or not resolved.is_dir():
        raise ValueError(f"analysis input is not a regular run directory: {root}")
    commit_path = resolved / "_COMMIT.json"
    commit = _json(commit_path)
    transaction_sha256 = commit.get("transaction_sha256")
    if not isinstance(transaction_sha256, str):
        raise ValueError(f"committed run lacks a transaction digest: {root}")
    StagedArtifactRun(
        output_parent=resolved.parent,
        run_name=resolved.name,
        transaction_sha256=transaction_sha256,
    ).validate_committed_run()
    return resolved, commit, sha256_file(commit_path)


def _binding_category(logical_key: str) -> str:
    key = logical_key.casefold()
    if "parties" in key or "party" in key or "shipper" in key or "exporter" in key:
        return "party"
    if "cargo" in key or "package" in key or "hazard" in key or "temperature" in key:
        return "cargo"
    if "container" in key or "seal" in key or "equipment" in key:
        return "equipment"
    if "date" in key or "issue" in key or "route" in key or "port" in key:
        return "route/date"
    if "number" in key or "reference" in key or "acid" in key or "tax" in key:
        return "identifier"
    return "other"


def _figure_bytes(figure: Figure) -> bytes:
    buffer = io.BytesIO()
    figure.savefig(buffer, format="png", dpi=170, bbox_inches="tight", facecolor="white")
    plt.close(figure)
    return buffer.getvalue()


def _csv_bytes(frame: pd.DataFrame) -> bytes:
    return str(frame.to_csv(index=False, lineterminator="\n")).encode("utf-8")


def _describe(series: pd.Series) -> dict[str, float]:
    return {
        "minimum": float(series.min()),
        "median": float(series.median()),
        "p90": float(series.quantile(0.90)),
        "p95": float(series.quantile(0.95)),
        "maximum": float(series.max()),
        "mean": float(series.mean()),
    }


def _load_manual_review(
    path: Path, *, run_commit_sha256: str, run_transaction_sha256: str
) -> tuple[dict[str, Any], pd.DataFrame]:
    payload = yaml.safe_load(read_regular_file_bytes(path))
    if not isinstance(payload, dict) or payload.get("schema_version") != 1:
        raise ValueError("manual review has an invalid schema")
    pinned = payload.get("run")
    if not isinstance(pinned, dict):
        raise ValueError("manual review lacks its run pin")
    if pinned.get("commit_sha256") != run_commit_sha256:
        raise ValueError("manual review pins a different descendant commit")
    if pinned.get("transaction_sha256") != run_transaction_sha256:
        raise ValueError("manual review pins a different descendant transaction")
    reviews = payload.get("reviews")
    if not isinstance(reviews, list) or not reviews:
        raise ValueError("manual review has no reviewed cases")
    frame = pd.DataFrame(reviews)
    if frame["document_id"].duplicated().any():
        raise ValueError("manual review repeats a document")
    required = {
        "document_id",
        "target_origin",
        "rendering_fidelity",
        "carrier_preservation",
        "layout_preservation",
        "whole_document_coherence",
        "primary_surface",
        "observation",
    }
    if not required.issubset(frame.columns):
        raise ValueError("manual review lacks required dimensions")
    return payload, frame


def _document_frame(run_root: Path, result_rows: tuple[dict[str, Any], ...]) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for result in result_rows:
        document_id = cast(str, result["document_id"])
        prefix = run_root / "cases" / document_id
        stage = _json(prefix / "source-agent-stage.json")
        replay = _json(prefix / "replay-receipt.json")
        source = read_regular_file_bytes(prefix / "source.txt")
        rendered = read_regular_file_bytes(prefix / "rendered.txt")
        row = dict(result)
        row.update(
            {
                "source_lines": len(source.splitlines()),
                "rendered_lines": len(rendered.splitlines()),
                "pages": source.count(b"--- PAGE "),
                "agent_slot_fraction": result["agent_slot_count"] / result["template_slot_count"],
                "changed_slot_fraction": result["changed_slot_count"]
                / result["template_slot_count"],
                "visible_output_tokens": result["output_tokens"] - result["reasoning_tokens"],
                "reasoning_output_fraction": (
                    result["reasoning_tokens"] / result["output_tokens"]
                    if result["output_tokens"]
                    else 0.0
                ),
                "estimated_cost_usd": float(result["estimated_cost_usd"]),
                "provider_duration_seconds": float(stage["duration_seconds"]),
                "source_residual_slots": replay["source_output_slot_count"],
                "replayed_residual_slots": replay["replayed_output_slot_count"],
                "dropped_residual_slots": replay["dropped_output_slot_count"],
                "new_provider_requests": replay["new_provider_requests"],
            }
        )
        rows.append(row)
    frame = pd.DataFrame(rows).sort_values("document_id").reset_index(drop=True)
    if len(frame) != 30 or frame["document_id"].nunique() != 30:
        raise ValueError("descendant EDA requires exactly 30 unique documents")
    return frame


def _route_frame(run_root: Path, document_ids: pd.Series) -> pd.DataFrame:
    rows = []
    for document_id in document_ids:
        routes = json.loads(
            read_regular_file_bytes(run_root / "cases" / document_id / "route-plan.json")
        )
        if not isinstance(routes, list):
            raise ValueError(f"route plan is not a list: {document_id}")
        for route in routes:
            logical_key = route["logical_key"]
            rows.append(
                {
                    "document_id": document_id,
                    "binding_id": route["binding_id"],
                    "logical_key": logical_key,
                    "binding_origin": "anchor" if logical_key.startswith("anchor:") else "agent",
                    "category": _binding_category(logical_key),
                    "compiled_mode": route["compiled_mode"],
                    "runtime_route": route["runtime_route"],
                    "route_reason": route["route_reason"],
                    "slot_count": len(route["slot_ids"]),
                }
            )
    return pd.DataFrame(rows)


def _baseline_metrics(baseline_root: Path) -> dict[str, Any]:
    lineage = pd.read_csv(baseline_root / "lineage-costs.csv")
    cases = pd.read_csv(baseline_root / "case-metrics.csv")
    certified = _json(baseline_root / "machine-certified" / "manifest.json")
    confirmed = _json(baseline_root / "confirmed" / "manifest.json")
    documents = len(cases)
    if documents != 100:
        raise ValueError("comparison baseline is not the expected 100-document cohort")
    return {
        "documents": documents,
        "successfulStructuredResponses": int(cases["model_requests"].sum()),
        "providerRouteAttempts": int(cases["provider_attempts"].sum()),
        "failedProviderAttempts": int(cases["failed_provider_attempts"].sum()),
        "providerCostUsd": float(lineage["provider_reported_cost_usd"].sum()),
        "machineCertifiedDocuments": int(certified["documents"]),
        "manuallyConfirmedDocuments": int(confirmed["documents"]),
    }


def _plots(
    *,
    documents: pd.DataFrame,
    routes: pd.DataFrame,
    manual: pd.DataFrame,
    baseline: Mapping[str, Any],
    summary: Mapping[str, Any],
) -> dict[str, bytes]:
    sns.set_theme(style="whitegrid", context="notebook")
    output: dict[str, bytes] = {}
    plotting = documents.assign(target_origin_label=documents["target_origin"].map(_ORIGIN_LABELS))
    origin_palette = {
        _ORIGIN_LABELS[key]: _PALETTE[key] for key in documents["target_origin"].unique()
    }

    figure, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    status = documents["status"].value_counts()
    axes[0].bar(
        status.index,
        status.values,
        color=[_PALETTE.get(x, "#546E7A") for x in status.index],
    )
    axes[0].set(title="Machine acceptance", ylabel="Documents", ylim=(0, 32))
    axes[0].bar_label(axes[0].containers[0])
    origin = plotting.groupby(["target_origin_label", "status"]).size().unstack(fill_value=0)
    origin.plot(kind="bar", stacked=True, ax=axes[1], color=[_PALETTE["passed"]], legend=False)
    axes[1].set(title="Acceptance by target origin", xlabel="", ylabel="Documents")
    axes[1].tick_params(axis="x", rotation=18)
    figure.suptitle("Compiled-template descendant outcome: 30/30 machine accepted")
    output["plots/01_acceptance_and_origin.png"] = _figure_bytes(figure)

    ordered = documents.sort_values("template_slot_count").copy()
    figure, axes = plt.subplots(1, 2, figsize=(13, 7))
    y = range(len(ordered))
    axes[0].barh(y, ordered["deterministic_slot_count"], color=_PALETTE["deterministic"])
    axes[0].barh(
        y,
        ordered["agent_slot_count"],
        left=ordered["deterministic_slot_count"],
        color=_PALETTE["agent"],
    )
    axes[0].set(title="Runtime slot routing per document", xlabel="Template slots", yticks=[])
    totals = [int(summary["deterministicSlots"]), int(summary["agentSlots"])]
    axes[1].pie(
        totals,
        labels=["Deterministic", "Residual agent"],
        autopct="%1.1f%%",
        colors=[_PALETTE["deterministic"], _PALETTE["agent"]],
        startangle=90,
    )
    axes[1].set_title(f"All {sum(totals):,} slots")
    figure.suptitle("Most compiled slots are materialized without an LLM")
    output["plots/02_slot_routing.png"] = _figure_bytes(figure)

    figure, axes = plt.subplots(1, 2, figsize=(12, 4.8))
    sns.histplot(
        data=plotting,
        x="changed_slot_fraction",
        hue="target_origin_label",
        bins=10,
        multiple="layer",
        ax=axes[0],
        palette=origin_palette,
    )
    axes[0].set(title="Changed-slot fraction", xlabel="Changed slots / template slots")
    sns.boxplot(
        data=plotting,
        x="target_origin_label",
        y="changed_slot_fraction",
        ax=axes[1],
        palette=origin_palette,
        hue="target_origin_label",
        legend=False,
    )
    axes[1].set(title="Change scope by target origin", xlabel="", ylabel="Changed-slot fraction")
    axes[1].tick_params(axis="x", rotation=18)
    figure.suptitle("Existing linguistic targets drive broader document edits")
    output["plots/03_change_scope.png"] = _figure_bytes(figure)

    figure, axes = plt.subplots(1, 3, figsize=(15, 4.6))
    metrics = (
        ("input_tokens", "Input tokens"),
        ("estimated_cost_usd", "Upstream cost (USD)"),
        ("provider_duration_seconds", "Provider duration (s)"),
    )
    for axis, (column, label) in zip(axes, metrics, strict=True):
        sns.scatterplot(
            data=plotting,
            x="agent_slot_count",
            y=column,
            hue="target_origin_label",
            palette=origin_palette,
            ax=axis,
            legend=axis is axes[0],
        )
        axis.set(xlabel="Residual-agent slots", ylabel=label)
    axes[0].legend(title="Target origin", fontsize=8)
    figure.suptitle("Residual scope is the remaining cost and latency driver")
    output["plots/04_residual_cost_tokens_latency.png"] = _figure_bytes(figure)

    current_requests = float(summary["responseRequestsUsed"]) / int(summary["documents"])
    current_cost = float(summary["upstreamEstimatedCostUsd"]) / int(summary["documents"])
    baseline_requests = baseline["successfulStructuredResponses"] / baseline["documents"]
    baseline_cost = baseline["providerCostUsd"] / baseline["documents"]
    figure, axes = plt.subplots(1, 2, figsize=(10, 4.8))
    labels = ["Prior 100-doc\ncalibration lineage", "Compiled-template\nresidual stage"]
    axes[0].bar(labels, [baseline_requests, current_requests], color=["#78909C", _PALETTE["agent"]])
    axes[0].set(title="Structured responses used per document", ylabel="Responses/document")
    axes[0].bar_label(axes[0].containers[0], fmt="%.2f")
    axes[1].bar(labels, [baseline_cost, current_cost], color=["#78909C", _PALETTE["agent"]])
    axes[1].set(title="Provider cost per document", ylabel="USD/document")
    axes[1].bar_label(axes[1].containers[0], fmt="$%.4f")
    figure.suptitle("Directional comparison only: compilation and target synthesis are excluded")
    output["plots/05_prior_lineage_comparison.png"] = _figure_bytes(figure)

    agent_routes = routes[routes["runtime_route"] == "agent"]
    grouped = agent_routes.groupby("category").agg(
        bindings=("binding_id", "count"), slots=("slot_count", "sum")
    )
    grouped = grouped.sort_values("slots")
    figure, axis = plt.subplots(figsize=(9, 5))
    grouped[["bindings", "slots"]].plot(kind="barh", ax=axis, color=["#FFB74D", _PALETTE["agent"]])
    axis.set(title="Residual-agent work by semantic surface", xlabel="Count", ylabel="")
    output["plots/06_residual_surface_categories.png"] = _figure_bytes(figure)

    source_slots = int(documents["source_residual_slots"].sum())
    replayed_slots = int(documents["replayed_residual_slots"].sum())
    figure, axis = plt.subplots(figsize=(8, 4.8))
    bars = axis.bar(
        ["Paid v1 route", "Hardened host replay"],
        [source_slots, replayed_slots],
        color=["#B0BEC5", _PALETTE["agent"]],
    )
    axis.bar_label(bars)
    axis.set(
        title=f"Host hardening retired {source_slots - replayed_slots} residual slots",
        ylabel="Agent-routed slots",
    )
    output["plots/07_offline_replay_hardening.png"] = _figure_bytes(figure)

    figure, axes = plt.subplots(1, 2, figsize=(11, 4.8))
    render_counts = manual["rendering_fidelity"].value_counts()
    axes[0].bar(render_counts.index, render_counts.values, color=_PALETTE["passed"])
    axes[0].bar_label(axes[0].containers[0])
    axes[0].set(title="Target-to-template rendering fidelity", ylabel="Reviewed documents")
    coherence = manual["whole_document_coherence"].value_counts()
    coherence_labels = [
        "Source limitation" if value == "pass_with_source_limitation" else value.title()
        for value in coherence.index
    ]
    axes[1].bar(
        coherence_labels,
        coherence.values,
        color=[_PALETTE.get(value, "#546E7A") for value in coherence.index],
    )
    axes[1].bar_label(axes[1].containers[0])
    axes[1].set(title="Whole-document semantic coherence", ylabel="Reviewed documents")
    axes[1].tick_params(axis="x", rotation=18)
    figure.suptitle("Purposive manual stress review (n=6; not a population estimate)")
    output["plots/08_manual_stress_review.png"] = _figure_bytes(figure)
    return output


def _report(summary: Mapping[str, Any]) -> str:
    manual = cast(Mapping[str, Any], summary["manualReview"])
    comparison = cast(Mapping[str, Any], summary["priorLineageComparison"])
    return "\n".join(
        (
            "# Compiled-template descendant 30-document EDA",
            "",
            "## Outcome",
            "",
            f"- Machine acceptance: **{summary['passedDocuments']}/{summary['documents']}**.",
            (
                "- Exact carrier, topology, literal-region, page-marker, line-ending, format, "
                "target-semantic, and source-relationship checks: **30/30 each**."
            ),
            (
                f"- Runtime routing: **{summary['deterministicSlots']}/"
                f"{summary['templateSlots']} slots "
                f"({summary['deterministicSlotFraction']:.1%}) deterministic**; "
                f"**{summary['agentSlots']} ({summary['agentSlotFraction']:.1%}) "
                "residual-agent**."
            ),
            (
                f"- Provider path: **{summary['responseRequestsUsed']} upstream responses "
                f"reused**, **{summary['newProviderRequests']} new calls in replay**."
            ),
            (
                f"- Paid v1 residual stage: **${summary['upstreamEstimatedCostUsd']} total**, "
                f"**${summary['upstreamCostPerDocumentUsd']:.6f}/document**. "
                "Offline replay added **$0**."
            ),
            "",
            "## Interpretation",
            "",
            (
                "The compiled-template insertion mechanism is technically reliable on this "
                "cohort: all target-bound facts passed exact host checks, repeated identifier "
                "relationships were reconciled, and every byte outside owned slots remained "
                "exact. The result does **not** yet establish whole-document semantic quality."
            ),
            "",
            (
                "The purposive six-document manual stress review found target-to-template "
                f"rendering fidelity in **{manual['renderingFidelityPasses']}/6**, but "
                "whole-document coherence concerns in "
                f"**{manual['coherenceConcerns']}/6**. Those concerns concentrate in "
                "source-only residual fields that were not grouped with their party or "
                "commercial context (for example, exporter country versus shipper country "
                "and independently generated postal codes). One additional case retained an "
                "already-truncated OCR date surface."
            ),
            "",
            (
                "Accordingly, the correct decision is: **the compiled insertion core is "
                "validated; the current residual/template metadata contract is not yet ready "
                "for training publication or scale-up**. The next pass should improve compiler "
                "grouping and dependency metadata, then rerun this same audit without "
                "increasing the per-document call ceiling."
            ),
            "",
            "## Directional efficiency comparison",
            "",
            (
                "- Successful structured responses: **2.24/document -> 1.00/document "
                f"({comparison['responseReductionPercent']:.1f}% lower)**."
            ),
            (
                "- Provider cost: **$0.005054/document -> "
                f"${summary['upstreamCostPerDocumentUsd']:.6f}/document "
                f"({comparison['costReductionPercent']:.1f}% lower)**."
            ),
            (
                "- This is not a like-for-like total-pipeline comparison: the prior figure is "
                "the complete 100-document calibration/continuation lineage, while the new "
                "figure is only post-compilation residual rendering and excludes target "
                "synthesis plus amortized template compilation."
            ),
            "",
            "## Artifact map",
            "",
            (
                "- `documents.csv`: one row per rendered document, including routing, usage, "
                "latency, and invariants."
            ),
            "- `routes.csv`: every binding and its runtime route/reason.",
            (
                "- `manual-review.csv` and `inputs/manual-review.yaml`: the purposive review "
                "and exact observations."
            ),
            "- `plots/`: eight deterministic PNG figures.",
            "- No training records were published.",
            "",
        )
    )


def analyze_descendants(
    *,
    run_dir: Path,
    baseline_dir: Path,
    manual_review_path: Path,
    output_parent: Path,
    run_name: str,
) -> Path:
    run_root, run_commit, run_commit_sha = _validate_committed(run_dir)
    baseline_root, baseline_commit, baseline_commit_sha = _validate_committed(baseline_dir)
    run_transaction = cast(str, run_commit["transaction_sha256"])
    manual_payload, manual = _load_manual_review(
        manual_review_path.resolve(strict=True),
        run_commit_sha256=run_commit_sha,
        run_transaction_sha256=run_transaction,
    )
    result_rows = _jsonl(run_root / "results.jsonl")
    documents = _document_frame(run_root, result_rows)
    routes = _route_frame(run_root, documents["document_id"])
    run_summary = _json(run_root / "summary.json")
    if run_summary.get("status") != "passed" or int(run_summary.get("passedDocuments", 0)) != 30:
        raise ValueError("descendant run is not a complete 30-document pass")
    if set(manual["document_id"]) - set(documents["document_id"]):
        raise ValueError("manual review contains documents outside the descendant run")
    baseline = _baseline_metrics(baseline_root)

    request_reduction = 100 * (
        1
        - (run_summary["responseRequestsUsed"] / run_summary["documents"])
        / (baseline["successfulStructuredResponses"] / baseline["documents"])
    )
    current_cost_per_document = float(run_summary["upstreamEstimatedCostUsd"]) / int(
        run_summary["documents"]
    )
    baseline_cost_per_document = baseline["providerCostUsd"] / baseline["documents"]
    cost_reduction = 100 * (1 - current_cost_per_document / baseline_cost_per_document)
    manual_counts = Counter(manual["whole_document_coherence"])
    summary = {
        "schemaVersion": 1,
        "status": "complete",
        "decision": "insertion_core_validated_residual_contract_requires_hardening",
        "documents": 30,
        "passedDocuments": int((documents["status"] == "passed").sum()),
        "existingTargetDocuments": int(
            (documents["target_origin"] == "existing_linguistic_target_carrier_restored").sum()
        ),
        "controlledTargetDocuments": int(
            (documents["target_origin"] == "controlled_source_variant").sum()
        ),
        "templateBindings": int(documents["template_binding_count"].sum()),
        "templateSlots": int(documents["template_slot_count"].sum()),
        "deterministicBindings": int(documents["deterministic_binding_count"].sum()),
        "agentBindings": int(documents["agent_binding_count"].sum()),
        "deterministicSlots": int(documents["deterministic_slot_count"].sum()),
        "agentSlots": int(documents["agent_slot_count"].sum()),
        "deterministicSlotFraction": float(
            documents["deterministic_slot_count"].sum() / documents["template_slot_count"].sum()
        ),
        "agentSlotFraction": float(
            documents["agent_slot_count"].sum() / documents["template_slot_count"].sum()
        ),
        "changedSlots": int(documents["changed_slot_count"].sum()),
        "responseRequestsUsed": int(run_summary["responseRequestsUsed"]),
        "newProviderRequests": int(run_summary["newProviderRequests"]),
        "upstreamEstimatedCostUsd": run_summary["upstreamEstimatedCostUsd"],
        "upstreamCostPerDocumentUsd": current_cost_per_document,
        "inputTokens": int(documents["input_tokens"].sum()),
        "reasoningTokens": int(documents["reasoning_tokens"].sum()),
        "visibleOutputTokens": int(documents["visible_output_tokens"].sum()),
        "sourceResidualSlots": int(documents["source_residual_slots"].sum()),
        "replayedResidualSlots": int(documents["replayed_residual_slots"].sum()),
        "droppedResidualSlots": int(documents["dropped_residual_slots"].sum()),
        "distributions": {
            "templateSlots": _describe(documents["template_slot_count"]),
            "agentSlots": _describe(documents["agent_slot_count"]),
            "changedSlotFraction": _describe(documents["changed_slot_fraction"]),
            "inputTokens": _describe(documents["input_tokens"]),
            "upstreamCostUsd": _describe(documents["estimated_cost_usd"]),
            "providerDurationSeconds": _describe(documents["provider_duration_seconds"]),
        },
        "manualReview": {
            "documents": len(manual),
            "selectionMethod": manual_payload["selection_method"],
            "renderingFidelityPasses": int((manual["rendering_fidelity"] == "pass").sum()),
            "carrierPreservationPasses": int((manual["carrier_preservation"] == "pass").sum()),
            "coherenceConcerns": int(manual_counts["concern"]),
            "coherencePassWithSourceLimitation": int(manual_counts["pass_with_source_limitation"]),
        },
        "priorLineageComparison": {
            **baseline,
            "successfulResponsesPerDocument": baseline["successfulStructuredResponses"]
            / baseline["documents"],
            "currentResponsesPerDocument": run_summary["responseRequestsUsed"]
            / run_summary["documents"],
            "baselineCostPerDocumentUsd": baseline_cost_per_document,
            "currentResidualCostPerDocumentUsd": current_cost_per_document,
            "responseReductionPercent": request_reduction,
            "costReductionPercent": cost_reduction,
            "scopeCaveat": (
                "Prior metric is full calibration/continuation lineage; current metric is only "
                "post-compilation residual rendering and excludes target synthesis and template "
                "compilation."
            ),
        },
        "allAcceptanceInvariantsPassed": bool(
            documents[
                [
                    "carrier_unchanged",
                    "exact_topology",
                    "every_slot_bound_once",
                    "exact_literal_regions",
                    "page_markers_unchanged",
                    "line_endings_preserved",
                    "format_envelopes_valid",
                    "target_binding_semantics_valid",
                    "source_relationships_valid",
                ]
            ].all(axis=None)
        ),
        "trainingRecordsPublished": False,
    }
    transaction = {
        "schemaVersion": 1,
        "kind": "compiled_template_descendant_eda",
        "runName": run_name,
        "descendantRunCommitSha256": run_commit_sha,
        "descendantRunTransactionSha256": run_transaction,
        "baselineRunCommitSha256": baseline_commit_sha,
        "baselineRunTransactionSha256": baseline_commit["transaction_sha256"],
        "manualReviewSha256": sha256_file(manual_review_path.resolve(strict=True)),
        "implementationSha256": sha256_file(_IMPLEMENTATION_PATH),
    }
    output_root = output_parent.resolve()
    stage = StagedArtifactRun(
        output_parent=output_root,
        run_name=run_name,
        transaction_sha256=sha256_bytes(canonical_json_bytes(transaction)),
    )
    if stage.completed:
        stage.validate_committed_run()
        return stage.final_root
    stage.recover_interrupted_temporary_files()
    stage.publish_json("transaction.json", transaction)
    stage.publish_json("summary.json", summary)
    stage.publish_bytes("documents.csv", _csv_bytes(documents))
    stage.publish_bytes("routes.csv", _csv_bytes(routes))
    stage.publish_bytes("manual-review.csv", _csv_bytes(manual))
    stage.publish_bytes(
        "inputs/manual-review.yaml",
        read_regular_file_bytes(manual_review_path.resolve(strict=True)),
    )
    for relative, payload in _plots(
        documents=documents,
        routes=routes,
        manual=manual,
        baseline=baseline,
        summary=summary,
    ).items():
        stage.publish_bytes(relative, payload)
    stage.publish_bytes("REPORT.md", _report(summary).encode("utf-8"))
    expected = tuple(
        path.relative_to(stage.stage_root).as_posix()
        for path in sorted(stage.stage_root.rglob("*"))
        if path.is_file() and path.name not in {"_TRANSACTION.json", "_COMMIT.json"}
    )
    commit = stage.commit(
        expected_artifacts=expected,
        metadata={
            "kind": "compiled_template_descendant_eda",
            "documents": 30,
            "passedDocuments": summary["passedDocuments"],
            "decision": summary["decision"],
            "trainingRecordsPublished": False,
        },
    )
    if not commit.created:
        stage.validate_committed_run()
    return stage.final_root
