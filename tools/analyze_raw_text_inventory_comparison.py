#!/usr/bin/env python3
"""Publish a reproducible comparison of committed raw-text inventory probe arms."""

from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd  # type: ignore[import-untyped]
import seaborn as sns  # type: ignore[import-untyped]


def _json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected object: {path}")
    return value


def _cost(usage: dict[str, Any]) -> tuple[float, str]:
    reported = usage.get("providerReportedCostUsd")
    if reported is not None:
        return float(reported), "provider_reported"
    return float(usage["estimatedCostUsd"]), "configured_price_estimate"


def _stage_latency(stages: list[dict[str, Any]]) -> float:
    return sum(
        float(stage["completedAtUnixSeconds"]) - float(stage["startedAtUnixSeconds"])
        for stage in stages
    )


def _load_cases(project_root: Path, plan: dict[str, Any]) -> list[dict[str, Any]]:
    audits = {
        (row["arm"], row["documentId"]): row for row in plan["manualAudits"]
    }
    rows: list[dict[str, Any]] = []
    for arm in plan["arms"]:
        arm_name = arm["name"]
        for case in arm["cases"]:
            document_id = case["documentId"]
            run_root = project_root / case["runPath"]
            if not (run_root / "_COMMIT.json").is_file():
                raise ValueError(f"run is not committed: {run_root}")
            case_root = run_root / "cases" / document_id
            result = _json(case_root / "result.json")
            stages_value = json.loads((case_root / "stages.json").read_text(encoding="utf-8"))
            if not isinstance(stages_value, list):
                raise ValueError(f"expected stage array: {case_root / 'stages.json'}")
            stages = [stage for stage in stages_value if isinstance(stage, dict)]
            summary = _json(run_root / "summary.json")
            usage = result["usage"]
            cost_usd, cost_basis = _cost(usage)
            audit = audits[(arm_name, document_id)]
            rows.append(
                {
                    "arm": arm_name,
                    "document_id": document_id,
                    "run_path": case["runPath"],
                    "status": result["status"],
                    "reason": result["reason"],
                    "host_pass": result["status"] == "training_ready",
                    "manual_semantic_pass": audit["semanticPass"],
                    "manual_training_usable": audit["trainingUsable"],
                    "manual_notes": audit["notes"],
                    "requests": usage["requests"],
                    "input_tokens": usage["inputTokens"],
                    "output_tokens": usage["outputTokens"],
                    "reasoning_tokens": usage["reasoningTokens"],
                    "visible_output_tokens": usage["visibleOutputTokens"],
                    "reasoning_share_of_output": (
                        usage["reasoningTokens"] / usage["outputTokens"]
                        if usage["outputTokens"]
                        else 0.0
                    ),
                    "cost_usd": cost_usd,
                    "cost_basis": cost_basis,
                    "model_stage_seconds": _stage_latency(stages),
                    "glm_false_passes_rejected": summary["glmFalsePassesRejected"],
                    "luna_false_passes_rejected": summary["lunaFalsePassesRejected"],
                    "source_path": str(case_root / "source.txt"),
                    "target_label_path": str(case_root / "target-label.json"),
                    "final_path": str(case_root / "final.txt"),
                    "diff_path": str(case_root / "diff.patch"),
                    "result_path": str(case_root / "result.json"),
                    "stages_path": str(case_root / "stages.json"),
                }
            )
    return rows


def _load_transport_events(project_root: Path, plan: dict[str, Any]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for event in plan.get("transportEvents", []):
        case_root = project_root / event["runPath"] / "cases" / event["documentId"]
        stages = json.loads((case_root / "stages.json").read_text(encoding="utf-8"))
        for stage in stages:
            if stage.get("errorType") is None:
                continue
            usage = stage["usage"]
            cost_usd, cost_basis = _cost(usage)
            rows.append(
                {
                    "arm": event["arm"],
                    "document_id": event["documentId"],
                    "route_provider": stage.get("routeProvider"),
                    "error_type": stage["errorType"],
                    "error_message": stage["errorMessage"],
                    "latency_seconds": (
                        float(stage["completedAtUnixSeconds"])
                        - float(stage["startedAtUnixSeconds"])
                    ),
                    "input_tokens": usage["inputTokens"],
                    "output_tokens": usage["outputTokens"],
                    "cost_usd": cost_usd,
                    "cost_basis": cost_basis,
                    "note": event["note"],
                }
            )
    return rows


def _load_baselines(project_root: Path, plan: dict[str, Any]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for arm in plan.get("baselines", []):
        for case in arm["cases"]:
            case_root = project_root / case["runPath"] / "cases" / case["documentId"]
            result = _json(case_root / "result.json")
            stages = json.loads((case_root / "stages.json").read_text(encoding="utf-8"))
            usage = result["usage"]
            cost_usd, cost_basis = _cost(usage)
            rows.append(
                {
                    "arm": arm["name"],
                    "document_id": case["documentId"],
                    "status": result["status"],
                    "requests": usage["requests"],
                    "input_tokens": usage["inputTokens"],
                    "output_tokens": usage["outputTokens"],
                    "reasoning_tokens": usage["reasoningTokens"],
                    "visible_output_tokens": usage["visibleOutputTokens"],
                    "cost_usd": cost_usd,
                    "cost_basis": cost_basis,
                    "model_stage_seconds": _stage_latency(stages),
                }
            )
    return rows


def _baseline_summaries(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[row["arm"]].append(row)
    return [
        {
            "arm": arm,
            "documents": len(values),
            "statuses": [row["status"] for row in values],
            "requests": sum(row["requests"] for row in values),
            "input_tokens": sum(row["input_tokens"] for row in values),
            "output_tokens": sum(row["output_tokens"] for row in values),
            "reasoning_tokens": sum(row["reasoning_tokens"] for row in values),
            "visible_output_tokens": sum(row["visible_output_tokens"] for row in values),
            "total_cost_usd": sum(row["cost_usd"] for row in values),
            "mean_model_stage_seconds": sum(row["model_stage_seconds"] for row in values)
            / len(values),
            "cost_basis": ", ".join(sorted({row["cost_basis"] for row in values})),
        }
        for arm, values in grouped.items()
    ]


def _summaries(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[row["arm"]].append(row)
    summaries: list[dict[str, Any]] = []
    for arm, values in grouped.items():
        count = len(values)
        total_cost = sum(row["cost_usd"] for row in values)
        summaries.append(
            {
                "arm": arm,
                "documents": count,
                "host_passes": sum(row["host_pass"] for row in values),
                "manual_semantic_passes": sum(row["manual_semantic_pass"] for row in values),
                "training_usable": sum(row["manual_training_usable"] for row in values),
                "input_tokens": sum(row["input_tokens"] for row in values),
                "output_tokens": sum(row["output_tokens"] for row in values),
                "reasoning_tokens": sum(row["reasoning_tokens"] for row in values),
                "visible_output_tokens": sum(row["visible_output_tokens"] for row in values),
                "total_cost_usd": total_cost,
                "mean_cost_usd": total_cost / count,
                "projected_cost_per_1000_usd": total_cost / count * 1000,
                "mean_model_stage_seconds": sum(
                    row["model_stage_seconds"] for row in values
                )
                / count,
                "cost_basis": ", ".join(sorted({row["cost_basis"] for row in values})),
                "glm_false_passes_rejected": min(
                    row["glm_false_passes_rejected"] for row in values
                ),
                "luna_false_passes_rejected": min(
                    row["luna_false_passes_rejected"] for row in values
                ),
            }
        )
    return summaries


def _save(path: Path) -> None:
    plt.savefig(path, dpi=180, bbox_inches="tight")
    plt.close()


def _plots(
    rows: list[dict[str, Any]],
    summaries: list[dict[str, Any]],
    baselines: list[dict[str, Any]],
    root: Path,
) -> None:
    plot_root = root / "plots"
    plot_root.mkdir(parents=True, exist_ok=True)
    cases = pd.DataFrame(rows)
    arms = [summary["arm"] for summary in summaries]
    palette = dict(zip(arms, sns.color_palette("colorblind", len(arms)), strict=True))
    sns.set_theme(style="whitegrid", context="talk")

    pass_frame = pd.DataFrame(
        [
            {
                "arm": summary["arm"],
                "measure": measure,
                "rate": summary[field] / summary["documents"],
            }
            for summary in summaries
            for measure, field in (
                ("host audit", "host_passes"),
                ("manual semantic", "manual_semantic_passes"),
                ("manual training usable", "training_usable"),
            )
        ]
    )
    plt.figure(figsize=(13, 6))
    sns.barplot(data=pass_frame, x="arm", y="rate", hue="measure")
    plt.ylim(0, 1.05)
    plt.ylabel("pass fraction")
    plt.xlabel("")
    plt.title("Hard-case pass rate (n=2 per arm)")
    plt.xticks(rotation=15, ha="right")
    _save(plot_root / "01_hard_case_pass_rate.png")

    token_frame = cases.melt(
        id_vars=["arm", "document_id"],
        value_vars=["input_tokens", "reasoning_tokens", "visible_output_tokens"],
        var_name="token_class",
        value_name="tokens",
    )
    plt.figure(figsize=(13, 7))
    sns.barplot(data=token_frame, x="arm", y="tokens", hue="token_class", estimator="mean")
    plt.ylabel("mean tokens per document")
    plt.xlabel("")
    plt.title("Token consumption by operating point")
    plt.xticks(rotation=15, ha="right")
    _save(plot_root / "02_token_consumption.png")

    summary_frame = pd.DataFrame(summaries)
    plt.figure(figsize=(13, 6))
    sns.barplot(
        data=summary_frame,
        x="arm",
        y="projected_cost_per_1000_usd",
        hue="arm",
        palette=palette,
        legend=False,
    )
    plt.ylabel("USD per 1,000 attempted documents")
    plt.xlabel("")
    plt.title("Linear cost projection from two deliberately difficult documents")
    plt.xticks(rotation=15, ha="right")
    _save(plot_root / "03_cost_projection.png")

    plt.figure(figsize=(13, 6))
    sns.barplot(data=cases, x="arm", y="model_stage_seconds", hue="arm", palette=palette)
    sns.stripplot(data=cases, x="arm", y="model_stage_seconds", color="black", size=7)
    plt.ylabel("model-stage seconds")
    plt.xlabel("")
    plt.title("Per-document model-stage latency")
    plt.xticks(rotation=15, ha="right")
    _save(plot_root / "04_model_stage_latency.png")

    plt.figure(figsize=(13, 6))
    sns.barplot(
        data=cases,
        x="arm",
        y="reasoning_share_of_output",
        hue="arm",
        palette=palette,
    )
    plt.ylim(0, 1.05)
    plt.ylabel("reasoning / total output tokens")
    plt.xlabel("")
    plt.title("Hidden reasoning share")
    plt.xticks(rotation=15, ha="right")
    _save(plot_root / "05_reasoning_share.png")

    new_by_arm = {row["arm"]: row for row in summaries}
    comparison_rows = [
        {
            "family": "GLM",
            "architecture": "previous editor+reviewer",
            "cost_per_1000": baselines[0]["total_cost_usd"] / 2 * 1000,
            "requests_per_document": baselines[0]["requests"] / 2,
        },
        {
            "family": "GLM",
            "architecture": "complete inventory one-call",
            "cost_per_1000": new_by_arm["GLM-5.3-Flash minimal"][
                "projected_cost_per_1000_usd"
            ],
            "requests_per_document": 1,
        },
        {
            "family": "Luna Low",
            "architecture": "previous editor+reviewer",
            "cost_per_1000": baselines[1]["total_cost_usd"] / 2 * 1000,
            "requests_per_document": baselines[1]["requests"] / 2,
        },
        {
            "family": "Luna Low",
            "architecture": "complete inventory one-call",
            "cost_per_1000": new_by_arm["Luna Low"]["projected_cost_per_1000_usd"],
            "requests_per_document": 1,
        },
    ]
    comparison = pd.DataFrame(comparison_rows)
    figure, axes = plt.subplots(1, 2, figsize=(16, 6))
    sns.barplot(
        data=comparison,
        x="family",
        y="cost_per_1000",
        hue="architecture",
        ax=axes[0],
    )
    sns.barplot(
        data=comparison,
        x="family",
        y="requests_per_document",
        hue="architecture",
        ax=axes[1],
    )
    axes[0].set(title="Exact-pair cost", xlabel="", ylabel="USD per 1,000")
    axes[1].set(title="Exact-pair request count", xlabel="", ylabel="requests / document")
    axes[1].legend_.remove()
    figure.suptitle("Previous restricted-span flow versus complete-inventory flow")
    figure.tight_layout()
    _save(plot_root / "06_exact_pair_architecture_comparison.png")


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _link(project_root: Path, value: str) -> str:
    return str((project_root / value).resolve())


def _report(
    project_root: Path,
    root: Path,
    rows: list[dict[str, Any]],
    summaries: list[dict[str, Any]],
    baselines: list[dict[str, Any]],
    transport_events: list[dict[str, Any]],
) -> None:
    lines = [
        "# Raw-text inventory rewrite hard-case comparison",
        "",
        "## Decision",
        "",
        (
            "The one-call complete-inventory architecture is validated as a hard-case proof, "
            "not yet as a production estimate. GLM minimal produced the cleanest two manually "
            "usable outputs. Luna Medium also produced two semantically correct outputs, but one "
            "was held from training for presentation defects. It cost more. Luna Low was "
            "under-capable; Luna High was wasteful and hit its output ceiling on the longer case."
        ),
        "",
        (
            "No training records were published. Every arm rejected all 12 historical GLM "
            "false passes and all 12 historical Luna false passes."
        ),
        "",
        "## Aggregate measurements",
        "",
        (
            "| Arm | Host pass | Manual semantic pass | Training usable | Input | Reasoning | "
            "Visible | Cost (2 docs) | Projected / 1k | Mean model latency | Cost basis |"
        ),
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|",
    ]
    for row in summaries:
        lines.append(
            f"| {row['arm']} | {row['host_passes']}/{row['documents']} | "
            f"{row['manual_semantic_passes']}/{row['documents']} | "
            f"{row['training_usable']}/{row['documents']} | {row['input_tokens']:,} | "
            f"{row['reasoning_tokens']:,} | {row['visible_output_tokens']:,} | "
            f"${row['total_cost_usd']:.6f} | ${row['projected_cost_per_1000_usd']:.2f} | "
            f"{row['mean_model_stage_seconds']:.1f}s | {row['cost_basis']} |"
        )
    lines.extend(
        [
            "",
            (
                "The projections are deliberately conservative and statistically weak: both "
                "documents were selected because older flows failed them, and n=2 cannot "
                "estimate a 50-document population. GLM cost is provider-reported; Luna cost "
                "uses the pinned configured rates because its API response did not report cost."
            ),
            "",
            "## Exact-pair comparison with the previous flow",
            "",
            "| Architecture | Outcomes | Requests | Input | Reasoning | Visible | Cost |",
            "|---|---|---:|---:|---:|---:|---:|",
        ]
    )
    for baseline in baselines:
        lines.append(
            f"| {baseline['arm']} | {', '.join(baseline['statuses'])} | "
            f"{baseline['requests']} | {baseline['input_tokens']:,} | "
            f"{baseline['reasoning_tokens']:,} | {baseline['visible_output_tokens']:,} | "
            f"${baseline['total_cost_usd']:.6f} |"
        )
    lines.extend(
        [
            "",
            (
                "On the exact GLM pair, the new architecture cut requests from 4 to 2, input "
                "tokens by 7.3%, visible output by 35.3%, and provider-reported cost by 9.4%. "
                "More importantly, it changed one old hold plus one old false pass into two "
                "independently usable outputs. The cost delta is modest because the old GLM "
                "reviewer was already cheap; the quality guard is the principal gain."
            ),
            "",
            "## Document-level audit",
            "",
            "| Arm | Document | Automated outcome | Manual outcome | Cost | Latency | Evidence |",
            "|---|---|---|---|---:|---:|---|",
        ]
    )
    for row in rows:
        doc = row["document_id"][:12]
        evidence = (
            f"[source]({_link(project_root, row['source_path'])}) / "
            f"[target]({_link(project_root, row['target_label_path'])}) / "
            f"[final]({_link(project_root, row['final_path'])}) / "
            f"[diff]({_link(project_root, row['diff_path'])}) / "
            f"[result]({_link(project_root, row['result_path'])}) / "
            f"[retained call]({_link(project_root, row['stages_path'])})"
        )
        lines.append(
            f"| {row['arm']} | `{doc}…` | {row['status']} | "
            f"{'pass' if row['manual_semantic_pass'] else 'fail'}"
            f"{' / held' if not row['manual_training_usable'] else ''} | "
            f"${row['cost_usd']:.6f} | "
            f"{row['model_stage_seconds']:.1f}s | {evidence} |"
        )
        lines.extend(["", f"- **{row['arm']} / `{doc}…`:** {row['manual_notes']}", ""])
    lines.extend(
        [
            "## Transport behavior",
            "",
            (
                f"GLM encountered {len(transport_events)} retained provider-route errors before "
                "a successful retry. These "
                "errors used zero reported tokens and $0.00. They are availability failures, "
                "not semantic failures; production execution still needs bounded retry/backoff "
                "across native-schema-capable endpoints."
            ),
            "",
            "## What the experiment proved",
            "",
            (
                "- The complete inventory plus deterministic compiler removed the old "
                "restricted-span blind spot: stale identifiers, role-specific party values, "
                "equipment semantics, repeated cargo facts, reefer state, and legal/operational "
                "flavor are audited over the full document."
            ),
            (
                "- A single provider-native structured response is sufficient on both hard "
                "cases; no editor/reviewer loop is required."
            ),
            (
                "- The host catches bad structured responses after schema validation. Luna Low "
                "returned valid JSON but was rejected for a blank lexical line and a wrong legal "
                "principal."
            ),
            (
                "- More reasoning is not monotonically better. Luna Medium was viable; Luna High "
                "spent 12,558 reasoning tokens across two cases and failed the longer one at the "
                "token ceiling."
            ),
            (
                "- GLM minimal is the candidate default for the 50-document validation gate. "
                "Luna Medium is the useful reference arm; Low and High should not be scaled."
            ),
            "",
            "## Remaining gate",
            "",
            (
                "This two-document proof cannot establish population reliability or production "
                "throughput. The next gate is the already planned paired 50-document run with "
                "concurrency, the same full-document audit, independent manual sampling, "
                "provider-error accounting, and distribution plots. Do not publish synthetic "
                "training rows until that gate confirms the false-pass rate is acceptably close "
                "to zero."
            ),
            "",
            "## Plots",
            "",
            "- [Hard-case pass rate](plots/01_hard_case_pass_rate.png)",
            "- [Token consumption](plots/02_token_consumption.png)",
            "- [Cost projection](plots/03_cost_projection.png)",
            "- [Model-stage latency](plots/04_model_stage_latency.png)",
            "- [Reasoning share](plots/05_reasoning_share.png)",
            (
                "- [Exact-pair architecture comparison]"
                "(plots/06_exact_pair_architecture_comparison.png)"
            ),
        ]
    )
    (root / "REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project-root", type=Path, default=Path("."))
    parser.add_argument("--plan", type=Path, required=True)
    args = parser.parse_args()
    project_root = args.project_root.resolve(strict=True)
    plan_path = (project_root / args.plan).resolve(strict=True)
    output_root = plan_path.parent
    plan = _json(plan_path)
    rows = _load_cases(project_root, plan)
    summaries = _summaries(rows)
    baseline_rows = _load_baselines(project_root, plan)
    baselines = _baseline_summaries(baseline_rows)
    transport_events = _load_transport_events(project_root, plan)
    _write_csv(output_root / "cases.csv", rows)
    _write_csv(output_root / "transport-events.csv", transport_events)
    with (output_root / "manual-audit.jsonl").open("w", encoding="utf-8") as handle:
        for audit in plan["manualAudits"]:
            handle.write(json.dumps(audit, ensure_ascii=False, sort_keys=True) + "\n")
    (output_root / "comparison.json").write_text(
        json.dumps(
            {
                "schemaVersion": 1,
                "summaries": summaries,
                "baselines": baselines,
                "transportEvents": transport_events,
                "trainingRecordsPublished": False,
            },
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    _plots(rows, summaries, baselines, output_root)
    _report(project_root, output_root, rows, summaries, baselines, transport_events)
    print(json.dumps({"cases": len(rows), "outputRoot": str(output_root), "status": "complete"}))


if __name__ == "__main__":
    main()
