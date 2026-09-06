#!/usr/bin/env python3
"""Analyze one committed raw-text inventory batch and its pinned manual audit."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from collections import Counter
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd  # type: ignore[import-untyped]
import seaborn as sns  # type: ignore[import-untyped]


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path}")
    return value


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _failure_category(reason: str) -> str:
    mappings = (
        ("omits changed target text", "target_value_omitted"),
        ("duplicates or omits role-bound party values", "party_occurrence_count"),
        ("omits host-locked literals", "host_locked_literal_omitted"),
        ("omits path-owned rendered surfaces", "rendered_surface_omitted"),
        ("duplicates unchanged source lines", "source_line_duplicated"),
        ("stale named customs program", "jurisdictional_surface"),
        ("inline labeled slot", "inline_slot_topology"),
    )
    for needle, category in mappings:
        if needle in reason:
            return category
    return "other_atomic_rejection"


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _active_stage_seconds(stage_rows: list[dict[str, Any]]) -> float:
    intervals = sorted(
        (float(row["startedAtUnixSeconds"]), float(row["completedAtUnixSeconds"]))
        for row in stage_rows
    )
    clusters: list[tuple[float, float]] = []
    for start, end in intervals:
        if not clusters or start > clusters[-1][1] + 120:
            clusters.append((start, end))
        else:
            clusters[-1] = (clusters[-1][0], max(clusters[-1][1], end))
    return sum(end - start for start, end in clusters)


def _save(path: Path) -> None:
    plt.savefig(path, dpi=180, bbox_inches="tight")
    plt.close()


def _plots(
    *,
    cases: pd.DataFrame,
    failures: Counter[str],
    findings: Counter[str],
    manual: list[dict[str, Any]],
    stage_rows: list[dict[str, Any]],
    output: Path,
) -> None:
    plot_root = output / "plots"
    plot_root.mkdir(parents=True, exist_ok=True)
    sns.set_theme(style="whitegrid", context="notebook")
    status_order = ["training_ready", "needs_review", "call_failed"]

    plt.figure(figsize=(9, 5.5))
    sns.countplot(data=cases, x="status", order=status_order, hue="status", legend=False)
    plt.title("Host outcome over 50 attempted rewrites")
    plt.xlabel("")
    plt.ylabel("documents")
    _save(plot_root / "01_host_outcomes.png")

    failure_frame = pd.DataFrame(
        [{"category": key, "documents": value} for key, value in failures.most_common()]
    )
    plt.figure(figsize=(11, 6))
    sns.barplot(data=failure_frame, y="category", x="documents", hue="category", legend=False)
    plt.title("Atomic rejection taxonomy")
    plt.xlabel("documents")
    plt.ylabel("")
    _save(plot_root / "02_atomic_rejections.png")

    finding_frame = pd.DataFrame(
        [{"category": key, "findings": value} for key, value in findings.most_common()]
    )
    plt.figure(figsize=(11, 6))
    sns.barplot(data=finding_frame, y="category", x="findings", hue="category", legend=False)
    plt.title("Postcondition findings among committed needs-review cases")
    plt.xlabel("findings")
    plt.ylabel("")
    _save(plot_root / "03_postcondition_findings.png")

    token_frame = cases.melt(
        id_vars=["document_id", "status"],
        value_vars=["input_tokens", "reasoning_tokens", "visible_tokens"],
        var_name="token_class",
        value_name="tokens",
    )
    plt.figure(figsize=(11, 6))
    sns.barplot(
        data=token_frame,
        x="status",
        y="tokens",
        hue="token_class",
        order=status_order,
        estimator="mean",
    )
    plt.title("Mean token consumption by outcome")
    plt.xlabel("")
    plt.ylabel("tokens per document")
    _save(plot_root / "04_tokens_by_outcome.png")

    plt.figure(figsize=(11, 6))
    sns.boxplot(data=cases, x="status", y="cost_usd", order=status_order)
    sns.stripplot(data=cases, x="status", y="cost_usd", order=status_order, color="black")
    plt.title("Provider-reported document cost")
    plt.xlabel("")
    plt.ylabel("USD")
    _save(plot_root / "05_cost_by_outcome.png")

    plt.figure(figsize=(11, 6))
    sns.scatterplot(
        data=cases,
        x="model_lines",
        y="cost_usd",
        hue="status",
        hue_order=status_order,
        s=75,
    )
    plt.title("Residual line count versus provider-reported cost")
    plt.xlabel("model-owned OCR lines")
    plt.ylabel("USD")
    _save(plot_root / "06_line_scope_vs_cost.png")

    provider_frame = pd.DataFrame(stage_rows)
    provider_frame["outcome"] = provider_frame["errorType"].map(
        lambda value: "transport failure" if isinstance(value, str) else "response"
    )
    plt.figure(figsize=(10, 5.5))
    sns.countplot(data=provider_frame, x="routeProvider", hue="outcome")
    plt.title("Provider-route attempts and fallback recovery")
    plt.xlabel("configured route")
    plt.ylabel("attempts")
    _save(plot_root / "07_provider_routing.png")

    manual_frame = pd.DataFrame(manual)
    plt.figure(figsize=(8, 5.5))
    sns.countplot(data=manual_frame, x="trainingUsable", hue="trainingUsable", legend=False)
    plt.title("Manual audit of the six apparent host passes")
    plt.xlabel("training usable after full-text review")
    plt.ylabel("documents")
    _save(plot_root / "08_manual_ready_audit.png")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--manual-audit", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    run_root = args.run_root.resolve(strict=True)
    if not (run_root / "_COMMIT.json").is_file():
        raise ValueError(f"run is not committed: {run_root}")
    summary = _read_json(run_root / "summary.json")
    audit = _read_json(args.manual_audit.resolve(strict=True))
    if audit.get("runId") != summary.get("runId"):
        raise ValueError("manual audit run ID differs from the committed run")

    result_lines = (run_root / "generation/results.jsonl").read_text().splitlines()
    results = [json.loads(line) for line in result_lines]
    if len(results) != summary["liveDocuments"]:
        raise ValueError("result count differs from summary")
    ready_ids = {row["documentId"] for row in results if row["status"] == "training_ready"}
    manual = audit.get("cases")
    if not isinstance(manual, list) or {row["documentId"] for row in manual} != ready_ids:
        raise ValueError("manual audit must cover every and only host training-ready case")

    case_rows: list[dict[str, Any]] = []
    stage_rows: list[dict[str, Any]] = []
    failures: Counter[str] = Counter()
    findings: Counter[str] = Counter()
    for result in results:
        document_id = result["documentId"]
        stages = json.loads((run_root / "cases" / document_id / "stages.json").read_text())
        stage_rows.extend(stages)
        usage = result["usage"]
        case_rows.append(
            {
                "document_id": document_id,
                "status": result["status"],
                "model_lines": result["modelLines"],
                "inventory_candidates": result["inventoryCandidates"],
                "input_tokens": usage["inputTokens"],
                "reasoning_tokens": usage["reasoningTokens"],
                "visible_tokens": usage["visibleOutputTokens"],
                "cost_usd": float(usage["providerReportedCostUsd"]),
                "latency_seconds": sum(
                    row["completedAtUnixSeconds"] - row["startedAtUnixSeconds"] for row in stages
                ),
                "provider_attempts": len(stages),
                "successful_provider": next(
                    (
                        row["usage"]["downstreamProviders"][-1]
                        for row in reversed(stages)
                        if row["errorType"] is None
                    ),
                    None,
                ),
            }
        )
        if result["status"] == "call_failed":
            failures[_failure_category(result["reason"])] += 1
        elif result["status"] == "needs_review":
            findings.update(
                finding["category"] for finding in result["fullDocumentAudit"]["findings"]
            )

    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)
    _write_csv(output / "cases.csv", case_rows)
    _write_csv(
        output / "atomic-rejection-taxonomy.csv",
        [{"category": key, "documents": value} for key, value in failures.most_common()],
    )
    _write_csv(
        output / "postcondition-finding-taxonomy.csv",
        [{"category": key, "findings": value} for key, value in findings.most_common()],
    )
    _write_csv(
        output / "manual-ready-audit.csv",
        [
            {
                **row,
                "issueCategories": ";".join(row["issueCategories"]),
                "evidenceLines": ";".join(row["evidenceLines"]),
            }
            for row in manual
        ],
    )
    frame = pd.DataFrame(case_rows)
    active_seconds = _active_stage_seconds(stage_rows)
    manual_usable = sum(row["trainingUsable"] for row in manual)
    analysis = {
        "schemaVersion": 1,
        "runId": summary["runId"],
        "runCommitSha256": _sha256(run_root / "_COMMIT.json"),
        "manualAuditSha256": _sha256(args.manual_audit),
        "documents": len(case_rows),
        "hostStatusCounts": dict(Counter(frame["status"])),
        "manuallyAuditedHostPasses": len(manual),
        "manuallyTrainingUsable": manual_usable,
        "manualTrainingUsableFractionOfAll": manual_usable / len(case_rows),
        "providerReportedCostUsd": float(frame["cost_usd"].sum()),
        "providerReportedCostPerThousandUsd": float(frame["cost_usd"].mean() * 1000),
        "meanInputTokens": float(frame["input_tokens"].mean()),
        "meanReasoningTokens": float(frame["reasoning_tokens"].mean()),
        "meanVisibleTokens": float(frame["visible_tokens"].mean()),
        "activeProviderStageSeconds": active_seconds,
        "activeProviderThroughputDocumentsPerHour": len(case_rows) / active_seconds * 3600,
        "transportFailuresRecovered": sum(row["errorType"] is not None for row in stage_rows),
        "successfulProviderCounts": dict(Counter(frame["successful_provider"])),
        "atomicRejectionTaxonomy": dict(failures),
        "postconditionFindingTaxonomy": dict(findings),
        "trainingRecordsPublished": False,
    }
    (output / "summary.json").write_text(
        json.dumps(analysis, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    _plots(
        cases=frame,
        failures=failures,
        findings=findings,
        manual=manual,
        stage_rows=stage_rows,
        output=output,
    )

    report = [
        "# GLM 5.3 Flash inventory rewrite: 50-document audit",
        "",
        "## Outcome",
        "",
        f"- Provider-reported cost: **${analysis['providerReportedCostUsd']:.8f}** "
        f"(**${analysis['providerReportedCostPerThousandUsd']:.3f}/1,000 attempted documents**).",
        f"- Host outcomes: **{summary['trainingReadyDocuments']} training-ready**, "
        f"**{summary['needsReviewDocuments']} needs-review**, and "
        f"**{summary['callFailedDocuments']} atomically rejected**.",
        f"- Manual full-text audit: **{manual_usable}/{len(manual)}** apparent passes "
        f"were actually training-usable (**{manual_usable}/{len(case_rows)} overall**).",
        f"- Recovered transport failures: **{analysis['transportFailuresRecovered']}**; "
        f"successful providers: **{analysis['successfulProviderCounts']}**.",
        f"- Active provider-stage throughput across the two invocations: "
        f"**{analysis['activeProviderThroughputDocumentsPerHour']:.1f} documents/hour**.",
        "- Training records published: **0**.",
        "",
        "The committed run summary's throughput divides all 50 documents by only the resumed "
        "invocation time. The value above reconstructs both active invocation windows from "
        "provider-stage timestamps.",
        "",
        "## Manual audit of apparent passes",
        "",
        "| Document | Usable | Finding | Evidence |",
        "|---|---:|---|---|",
    ]
    for row in manual:
        report.append(
            f"| `{row['documentId']}` | {'yes' if row['trainingUsable'] else 'no'} | "
            f"{row['notes']} | {', '.join(row['evidenceLines']) or '-'} |"
        )
    report.extend(
        (
            "",
            "## Interpretation",
            "",
            "Provider routing and cost control worked, but the one-shot line renderer is not "
            "quality-proven on this cohort. Most atomic rejections are missing target values; "
            "postcondition holds combine genuine model omissions with false positives from "
            "path-insensitive numeric and boilerplate matching. The manual audit also found "
            "four false passes not represented by the current guard.",
            "",
            "This run is evaluation-only and must not be promoted into training data.",
            "",
        )
    )
    (output / "REPORT.md").write_text("\n".join(report), encoding="utf-8")


if __name__ == "__main__":
    main()
