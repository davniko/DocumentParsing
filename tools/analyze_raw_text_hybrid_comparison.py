#!/usr/bin/env python3
"""Compare two committed compiler-first raw-text rewrite benchmark runs."""

from __future__ import annotations

import argparse
import difflib
import hashlib
import json
import re
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

OUTCOME_ORDER = ["quality_validated", "needs_review", "call_failed", "compiler_blocked"]
PALETTE = {
    "GLM-5.3-Flash": "#2878b5",
    "Luna Low": "#e07a2f",
}
IDENTIFIER_PATTERNS = {
    "email": re.compile(r"(?i)\b[A-Z0-9._%+-]+@[A-Z0-9.-]+\.[A-Z]{2,}\b"),
    "url": re.compile(r"(?i)\b(?:https?://|www\.)[^\s<>]+"),
    "mixed_identifier": re.compile(
        r"\b(?=[A-Z0-9-]{8,}\b)(?=[A-Z0-9-]*[A-Z])(?=[A-Z0-9-]*\d)[A-Z0-9-]+\b",
        re.I,
    ),
    "long_number": re.compile(r"(?<!\d)\d{8,}(?!\d)"),
}


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected a JSON object: {path}")
    return value


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _failure_category(status: str, stage: dict[str, Any] | None) -> str:
    if status == "quality_validated":
        return "declared contract passed"
    if status == "needs_review":
        return "reviewer requested revision"
    if status == "compiler_blocked":
        return "compiler blocked"
    if stage is None:
        return "unclassified call failure"
    error_type = str(stage.get("errorType") or "")
    message = str(stage.get("errorMessage") or "")
    if error_type == "ModelHTTPError":
        return "provider transport"
    if error_type == "UnexpectedModelBehavior":
        return "native structured output"
    patterns = (
        ("duplicates or omits role-bound party", "party occurrence mismatch"),
        ("omits changed target text", "target value omission"),
        ("moves or omits an exact changed label scalar", "anchored scalar mismatch"),
        ("repeats an authorized line ID", "duplicate line edit"),
        ("named customs program", "customs surface mismatch"),
        ("unavailable/unknown placeholder", "placeholder introduced"),
        ("deterministic date/HS rendering", "date or HS surface mismatch"),
        ("raw auxiliary identity must retain", "auxiliary principal mismatch"),
        ("inconsistent fictional realizations", "auxiliary identity inconsistency"),
        ("compound-party flavor realizations", "compound-party realization mismatch"),
        ("punctuation or whitespace outside", "format-only mutation"),
    )
    for needle, category in patterns:
        if needle in message:
            return category
    return "other host postcondition"


def _line_change_counts(source: str, final: str) -> tuple[int, int, int]:
    source_lines = source.splitlines()
    final_lines = final.splitlines()
    replaced = inserted = deleted = 0
    matcher = difflib.SequenceMatcher(a=source_lines, b=final_lines, autojunk=False)
    for tag, i1, i2, j1, j2 in matcher.get_opcodes():
        if tag == "replace":
            replaced += max(i2 - i1, j2 - j1)
        elif tag == "insert":
            inserted += j2 - j1
        elif tag == "delete":
            deleted += i2 - i1
    return replaced, inserted, deleted


def _retained_identifiers(
    source: str,
    final: str,
    target_label: dict[str, Any],
) -> list[tuple[str, str]]:
    """Return high-entropy source surfaces retained outside the target label.

    This is deliberately a review queue, not an automatic privacy verdict. A retained value can
    be legitimate boilerplate or an intentionally unchanged fact, but every returned value needs
    semantic review before a synthetic record can be called anonymized.
    """

    target_text = json.dumps(target_label, ensure_ascii=False, sort_keys=True).casefold()
    retained: set[tuple[str, str]] = set()
    for kind, pattern in IDENTIFIER_PATTERNS.items():
        for match in pattern.finditer(source):
            value = match.group(0).rstrip(".,;:")
            if value.casefold() in target_text:
                continue
            if value in final:
                retained.add((kind, value))
    return sorted(retained)


def _load_arm(root: Path, arm: str) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    if not (root / "_COMMIT.json").is_file():
        raise ValueError(f"run is not committed: {root}")
    case_dirs = sorted(path for path in (root / "cases").iterdir() if path.is_dir())
    cases: list[dict[str, Any]] = []
    stages_out: list[dict[str, Any]] = []
    for case_dir in case_dirs:
        result = _read_json(case_dir / "result.json")
        contract = _read_json(case_dir / "contract.json")
        stages = json.loads((case_dir / "stages.json").read_text(encoding="utf-8"))
        if not isinstance(stages, list):
            raise ValueError(f"invalid stages: {case_dir}")
        source = (case_dir / "source.txt").read_text(encoding="utf-8")
        final = (case_dir / "final.txt").read_text(encoding="utf-8")
        failing_stage = next(
            (stage for stage in reversed(stages) if stage.get("errorType") is not None),
            None,
        )
        usage = result["usage"]
        exact_cost = usage.get("providerReportedCostUsd")
        if exact_cost is None:
            exact_cost = usage["estimatedCostUsd"]
        started = [float(stage["startedAtUnixSeconds"]) for stage in stages]
        completed = [float(stage["completedAtUnixSeconds"]) for stage in stages]
        replaced, inserted, deleted = _line_change_counts(source, final)
        target_label = contract["targetLabel"]
        retained = _retained_identifiers(source, final, target_label)
        row = {
            "arm": arm,
            "document_id": result["documentId"],
            "status": result["status"],
            "reason": result["reason"],
            "failure_category": _failure_category(result["status"], failing_stage),
            "failure_stage": failing_stage.get("stage") if failing_stage else None,
            "error_type": failing_stage.get("errorType") if failing_stage else None,
            "error_message": failing_stage.get("errorMessage") if failing_stage else None,
            "source_lines": result["sourceLines"],
            "residual_lines": result["residualLines"],
            "residual_fraction": result["residualLines"] / result["sourceLines"],
            "work_items": result["workItems"],
            "residual_work_items": result["residualWorkItems"],
            "requests": usage["requests"],
            "input_tokens": usage["inputTokens"],
            "output_tokens": usage["outputTokens"],
            "reasoning_tokens": usage["reasoningTokens"],
            "visible_output_tokens": usage["visibleOutputTokens"],
            "reasoning_fraction": (
                usage["reasoningTokens"] / usage["outputTokens"]
                if usage["outputTokens"]
                else 0.0
            ),
            "cost_usd": float(exact_cost),
            "case_wall_seconds": max(completed) - min(started) if stages else 0.0,
            "source_final_similarity": difflib.SequenceMatcher(
                a=source,
                b=final,
                autojunk=False,
            ).ratio(),
            "replaced_lines": replaced,
            "inserted_lines": inserted,
            "deleted_lines": deleted,
            "retained_high_entropy_identifiers": len(retained),
            "retained_identifier_values": " | ".join(value for _, value in retained),
        }
        cases.append(row)
        for index, stage in enumerate(stages, start=1):
            stage_usage = stage["usage"]
            stage_cost = stage_usage.get("providerReportedCostUsd")
            if stage_cost is None:
                stage_cost = stage_usage["estimatedCostUsd"]
            stages_out.append(
                {
                    "arm": arm,
                    "document_id": result["documentId"],
                    "attempt": index,
                    "stage": stage["stage"],
                    "route_provider": stage.get("routeProvider") or "direct",
                    "reasoning_effort": stage["reasoningEffort"],
                    "succeeded": stage.get("errorType") is None,
                    "error_type": stage.get("errorType"),
                    "latency_seconds": (
                        float(stage["completedAtUnixSeconds"])
                        - float(stage["startedAtUnixSeconds"])
                    ),
                    "input_tokens": stage_usage["inputTokens"],
                    "reasoning_tokens": stage_usage["reasoningTokens"],
                    "visible_output_tokens": stage_usage["visibleOutputTokens"],
                    "cost_usd": float(stage_cost),
                }
            )
    return cases, stages_out


def _save_figure(path: Path) -> None:
    plt.savefig(path, dpi=180, bbox_inches="tight")
    plt.close()


def _plots(cases: pd.DataFrame, stages: pd.DataFrame, paired: pd.DataFrame, root: Path) -> None:
    plot_root = root / "plots"
    plot_root.mkdir(parents=True, exist_ok=True)
    sns.set_theme(style="whitegrid", context="talk")

    plt.figure(figsize=(11, 6))
    counts = cases.groupby(["arm", "status"], observed=False).size().reset_index(name="cases")
    sns.barplot(data=counts, x="status", y="cases", hue="arm", palette=PALETTE, order=OUTCOME_ORDER)
    plt.title("Declared-contract outcomes (50 identical documents per arm)")
    plt.xlabel("")
    _save_figure(plot_root / "01_outcome_counts.png")

    matrix = pd.crosstab(paired["glm_status"], paired["luna_status"]).reindex(
        index=OUTCOME_ORDER, columns=OUTCOME_ORDER, fill_value=0
    )
    plt.figure(figsize=(9, 7))
    sns.heatmap(matrix, annot=True, fmt="g", cmap="Blues", cbar=False)
    plt.title("Paired outcome matrix")
    plt.xlabel("Luna Low")
    plt.ylabel("GLM-5.3-Flash")
    _save_figure(plot_root / "02_paired_outcome_matrix.png")

    token_long = cases.melt(
        id_vars=["arm", "document_id"],
        value_vars=["input_tokens", "reasoning_tokens", "visible_output_tokens"],
        var_name="token_type",
        value_name="tokens",
    )
    plt.figure(figsize=(13, 6))
    sns.barplot(data=token_long, x="arm", y="tokens", hue="token_type", estimator="mean")
    plt.title("Mean tokens per attempted document")
    plt.xlabel("")
    plt.legend(title="token type", bbox_to_anchor=(1.02, 1), loc="upper left")
    _save_figure(plot_root / "03_mean_token_composition.png")

    plt.figure(figsize=(11, 6))
    sns.boxplot(data=cases, x="arm", y="cost_usd", hue="status", hue_order=OUTCOME_ORDER)
    sns.stripplot(data=cases, x="arm", y="cost_usd", color="black", alpha=0.35, size=3)
    plt.title("Per-document cost distribution")
    plt.xlabel("")
    _save_figure(plot_root / "04_cost_distribution.png")

    plt.figure(figsize=(11, 6))
    sns.ecdfplot(data=cases, x="case_wall_seconds", hue="arm", palette=PALETTE)
    plt.title("Case wall-time empirical CDF")
    plt.xlabel("seconds")
    _save_figure(plot_root / "05_latency_ecdf.png")

    failures = (
        cases[cases["status"] != "quality_validated"]
        .groupby(["failure_category", "arm"])
        .size()
        .reset_index(name="cases")
    )
    plt.figure(figsize=(13, 8))
    sns.barplot(data=failures, y="failure_category", x="cases", hue="arm", palette=PALETTE)
    plt.title("Failure and review taxonomy")
    plt.ylabel("")
    _save_figure(plot_root / "06_failure_taxonomy.png")

    plt.figure(figsize=(11, 7))
    sns.scatterplot(
        data=cases,
        x="residual_work_items",
        y="cost_usd",
        hue="arm",
        style="status",
        palette=PALETTE,
        s=90,
    )
    plt.title("Residual complexity versus cost")
    _save_figure(plot_root / "07_complexity_vs_cost.png")

    paired_tokens = paired.melt(
        id_vars="document_id",
        value_vars=["glm_cost_usd", "luna_cost_usd"],
        var_name="arm",
        value_name="cost_usd",
    )
    paired_tokens["arm"] = paired_tokens["arm"].map(
        {"glm_cost_usd": "GLM-5.3-Flash", "luna_cost_usd": "Luna Low"}
    )
    paired_tokens["case_order"] = paired_tokens.groupby("arm").cumcount() + 1
    paired_tokens["cumulative_cost_usd"] = paired_tokens.groupby("arm")["cost_usd"].cumsum()
    plt.figure(figsize=(11, 6))
    sns.lineplot(
        data=paired_tokens,
        x="case_order",
        y="cumulative_cost_usd",
        hue="arm",
        palette=PALETTE,
        marker="o",
        markersize=3,
    )
    plt.title("Cumulative cost across the fixed 50-document order")
    plt.xlabel("documents attempted")
    _save_figure(plot_root / "08_cumulative_cost.png")

    glm_stages = stages[stages["arm"] == "GLM-5.3-Flash"].copy()
    glm_stages["attempt_outcome"] = glm_stages["succeeded"].map(
        {True: "success", False: "failure"}
    )
    plt.figure(figsize=(10, 6))
    provider_counts = (
        glm_stages.groupby(["route_provider", "attempt_outcome"])
        .size()
        .reset_index(name="attempts")
    )
    sns.barplot(
        data=provider_counts,
        x="route_provider",
        y="attempts",
        hue="attempt_outcome",
    )
    plt.title("GLM application-level provider attempts")
    plt.xlabel("")
    _save_figure(plot_root / "09_glm_provider_attempts.png")

    successful = cases[cases["status"] == "quality_validated"]
    plt.figure(figsize=(11, 6))
    sns.histplot(
        data=successful,
        x="retained_high_entropy_identifiers",
        hue="arm",
        palette=PALETTE,
        multiple="dodge",
        discrete=True,
        shrink=0.8,
    )
    plt.title("Manual-review queue: retained source identifiers in declared passes")
    plt.xlabel("candidate retained identifiers")
    _save_figure(plot_root / "10_retained_identifier_candidates.png")


def _report(
    cases: pd.DataFrame,
    stages: pd.DataFrame,
    paired: pd.DataFrame,
    roots: dict[str, Path],
) -> str:
    lines = [
        "# GLM-5.3-Flash vs Luna Low: 50-document hybrid rewrite audit",
        "",
        "## Scope and interpretation",
        "",
        "Both arms processed the same 50 pinned source/target/template triples. A status of "
        "`quality_validated` means the current compiler postconditions and compact model review "
        "passed; it is not yet a full-text anonymization or training-readiness verdict.",
        "",
        "## Run provenance",
        "",
        "| Arm | Run | Commit SHA-256 | Summary SHA-256 |",
        "|---|---|---|---|",
    ]
    for arm, root in roots.items():
        commit = _read_json(root / "_COMMIT.json")
        commit_sha256 = commit.get("commitSha256", _sha256(root / "_COMMIT.json"))
        lines.append(
            f"| {arm} | `{root.name}` | `{commit_sha256}` "
            f"| `{_sha256(root / 'summary.json')}` |"
        )
    lines.extend(["", "## Headline measurements", ""])
    lines.append(
        "| Arm | Passed | Review | Call failed | Wall s | Docs/h | Requests | Input | Reasoning "
        "| Visible | Cost | Cost/1k attempts | Cost/declared pass |"
    )
    lines.append("|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|")
    for arm, root in roots.items():
        summary = _read_json(root / "summary.json")
        arm_cases = cases[cases["arm"] == arm]
        counts = arm_cases["status"].value_counts()
        passed = int(counts.get("quality_validated", 0))
        total_cost = float(arm_cases["cost_usd"].sum())
        lines.append(
            f"| {arm} | {passed} | {int(counts.get('needs_review', 0))} | "
            f"{int(counts.get('call_failed', 0))} | {summary['wallSeconds']:.1f} | "
            f"{summary['throughputCasesPerHour']:.1f} | {int(arm_cases['requests'].sum())} | "
            f"{int(arm_cases['input_tokens'].sum()):,} | "
            f"{int(arm_cases['reasoning_tokens'].sum()):,} | "
            f"{int(arm_cases['visible_output_tokens'].sum()):,} | ${total_cost:.6f} | "
            f"${total_cost / len(arm_cases) * 1000:.3f} | "
            f"${total_cost / passed:.6f} |"
        )
    pair_counts = (
        paired.groupby(["glm_status", "luna_status"]).size().sort_values(ascending=False)
    )
    lines.extend(["", "## Paired outcomes", ""])
    lines.extend(
        f"- GLM `{glm}`, Luna `{luna}`: **{count}** documents."
        for (glm, luna), count in pair_counts.items()
    )
    lines.extend(["", "## Failure taxonomy", ""])
    taxonomy = (
        cases.groupby(["arm", "failure_category"]).size().reset_index(name="documents")
    )
    lines.extend(["| Arm | Category | Documents |", "|---|---|---:|"])
    for row in taxonomy.sort_values(["arm", "documents"], ascending=[True, False]).itertuples():
        lines.append(f"| {row.arm} | {row.failure_category} | {row.documents} |")
    lines.extend(
        [
            "",
            "## Cost and token interpretation",
            "",
            "Reasoning tokens are included in provider output-token billing. Failed calls "
            "therefore still incur cost. Input counts also repeat document-scoped context "
            "independently for the "
            "editor and reviewer; no stage uses hidden prior-response state.",
            "",
            "The retained-identifier scan is intentionally conservative. It flags exact "
            "high-entropy source values that remain in final text but are absent from the target "
            "label. Its rows are a manual-review queue, not automatic proof of a privacy failure.",
            "",
            "## Artifacts",
            "",
            "- `data/cases.csv`: one row per arm/document.",
            "- `data/stages.csv`: one row per provider attempt.",
            "- `data/paired.csv`: paired statuses, costs, tokens, and output similarity.",
            "- `data/retained-identifier-review.csv`: candidate source-only identifier retention.",
            "- `plots/`: ten Matplotlib/Seaborn figures.",
        ]
    )
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--glm-run", type=Path, required=True)
    parser.add_argument("--luna-run", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    roots = {
        "GLM-5.3-Flash": args.glm_run.resolve(strict=True),
        "Luna Low": args.luna_run.resolve(strict=True),
    }
    all_cases: list[dict[str, Any]] = []
    all_stages: list[dict[str, Any]] = []
    for arm, root in roots.items():
        cases, stages = _load_arm(root, arm)
        all_cases.extend(cases)
        all_stages.extend(stages)
    cases_df = pd.DataFrame(all_cases)
    stages_df = pd.DataFrame(all_stages)
    if cases_df.groupby("arm")["document_id"].nunique().to_dict() != {
        "GLM-5.3-Flash": 50,
        "Luna Low": 50,
    }:
        raise ValueError("each comparison arm must contain exactly 50 unique documents")
    glm = cases_df[cases_df["arm"] == "GLM-5.3-Flash"].set_index("document_id")
    luna = cases_df[cases_df["arm"] == "Luna Low"].set_index("document_id")
    if set(glm.index) != set(luna.index):
        raise ValueError("comparison arms contain different document IDs")
    paired = pd.DataFrame(index=sorted(glm.index))
    paired.index.name = "document_id"
    for prefix, frame in (("glm", glm), ("luna", luna)):
        for column in (
            "status",
            "failure_category",
            "cost_usd",
            "input_tokens",
            "reasoning_tokens",
            "visible_output_tokens",
            "case_wall_seconds",
            "retained_high_entropy_identifiers",
        ):
            paired[f"{prefix}_{column}"] = frame[column]
    similarities = []
    for document_id in paired.index:
        glm_text = (roots["GLM-5.3-Flash"] / "cases" / document_id / "final.txt").read_text(
            encoding="utf-8"
        )
        luna_text = (roots["Luna Low"] / "cases" / document_id / "final.txt").read_text(
            encoding="utf-8"
        )
        similarities.append(
            difflib.SequenceMatcher(a=glm_text, b=luna_text, autojunk=False).ratio()
        )
    paired["intermodel_output_similarity"] = similarities
    paired = paired.reset_index()

    output = args.output_dir.resolve()
    data_root = output / "data"
    data_root.mkdir(parents=True, exist_ok=False)
    cases_df.to_csv(data_root / "cases.csv", index=False)
    stages_df.to_csv(data_root / "stages.csv", index=False)
    paired.to_csv(data_root / "paired.csv", index=False)
    cases_df[cases_df["retained_high_entropy_identifiers"] > 0][
        [
            "arm",
            "document_id",
            "status",
            "retained_high_entropy_identifiers",
            "retained_identifier_values",
        ]
    ].to_csv(data_root / "retained-identifier-review.csv", index=False)
    _plots(cases_df, stages_df, paired, output)
    (output / "REPORT.md").write_text(
        _report(cases_df, stages_df, paired, roots),
        encoding="utf-8",
    )
    manifest = {
        "schemaVersion": 1,
        "runs": {arm: str(root) for arm, root in roots.items()},
        "documentsPerArm": 50,
        "files": {
            str(path.relative_to(output)): {
                "bytes": path.stat().st_size,
                "sha256": _sha256(path),
            }
            for path in sorted(output.rglob("*"))
            if path.is_file()
        },
    }
    (output / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
