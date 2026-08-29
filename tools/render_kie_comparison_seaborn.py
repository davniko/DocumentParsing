#!/usr/bin/env python3
"""Render an untruncated Matplotlib/Seaborn supplement for a paired KIE audit."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import textwrap
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

_BASELINE = "Minimal JSON target"
_RELATION = "Semantic + relation target"
_COLORS = {_BASELINE: "#4C78A8", _RELATION: "#F58518"}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _save(figure: plt.Figure, path: Path) -> None:
    figure.savefig(path, dpi=180, bbox_inches="tight", facecolor="white")
    plt.close(figure)


def _line_plot(history: pd.DataFrame, output: Path) -> None:
    labels = {"semantic-v2": _BASELINE, "relation-explicit-v3": _RELATION}
    data = history[history["epoch"] > 0].copy()
    data["run_label"] = data["run"].map(labels)
    figure, axes = plt.subplots(2, 2, figsize=(14, 10), constrained_layout=True)
    specifications = (
        ("field_value_f1", "Exact field/value F1", axes[0, 0]),
        ("precision", "Exact field/value precision", axes[0, 1]),
        ("recall", "Exact field/value recall", axes[1, 0]),
        ("eval_loss", "Teacher-forced evaluation loss", axes[1, 1]),
    )
    for field, title, axis in specifications:
        sns.lineplot(
            data=data,
            x="epoch",
            y=field,
            hue="run_label",
            marker="o",
            palette=_COLORS,
            ax=axis,
        )
        axis.set_title(title)
        axis.set_xlabel("Epoch")
        axis.grid(alpha=0.25)
    figure.suptitle("Training and generated-evaluation progression", fontsize=16)
    _save(figure, output / "01_progression_overview.png")


def _validity_and_runtime(history: pd.DataFrame, output: Path) -> None:
    labels = {"semantic-v2": _BASELINE, "relation-explicit-v3": _RELATION}
    data = history[history["epoch"] > 0].copy()
    data["run_label"] = data["run"].map(labels)
    long = data.melt(
        id_vars=["epoch", "run_label"],
        value_vars=["json_valid", "schema_valid", "eos_fraction"],
        var_name="metric",
        value_name="fraction",
    )
    long["metric"] = long["metric"].map(
        {
            "json_valid": "JSON valid",
            "schema_valid": "Schema valid",
            "eos_fraction": "EOS reached",
        }
    )
    figure, axes = plt.subplots(1, 2, figsize=(15, 6), constrained_layout=True)
    sns.lineplot(
        data=long,
        x="epoch",
        y="fraction",
        hue="run_label",
        style="metric",
        markers=True,
        dashes=False,
        palette=_COLORS,
        ax=axes[0],
    )
    axes[0].set_ylim(0, 1.05)
    axes[0].set_title("Structured-generation validity")
    axes[0].set_ylabel("Fraction of 60 validation documents")
    sns.lineplot(
        data=data,
        x="epoch",
        y="eval_runtime_seconds",
        hue="run_label",
        marker="o",
        palette=_COLORS,
        ax=axes[1],
    )
    axes[1].set_title("Generated evaluation runtime")
    axes[1].set_ylabel("Seconds")
    _save(figure, output / "02_validity_and_eval_runtime.png")


def _component_plot(components: pd.DataFrame, output: Path) -> None:
    data = components[["component", "previous_f1", "current_f1"]].melt(
        id_vars="component", var_name="run", value_name="f1"
    )
    data["run"] = data["run"].map(
        {"previous_f1": _BASELINE, "current_f1": _RELATION}
    )
    data["component"] = data["component"].str.replace("_", " ").str.title()
    figure, axis = plt.subplots(figsize=(13, 7), constrained_layout=True)
    sns.barplot(
        data=data,
        y="component",
        x="f1",
        hue="run",
        palette=_COLORS,
        orient="h",
        ax=axis,
    )
    axis.set_xlim(0, 1)
    axis.set_title("All target components (no top-N truncation)")
    axis.set_xlabel("Exact field/value F1")
    axis.set_ylabel("")
    _save(figure, output / "03_all_component_f1.png")


def _paged_field_deltas(data: pd.DataFrame, output: Path, prefix: str, title: str) -> list[Path]:
    ordered = data.sort_values(["f1_delta", "concept"], ascending=[True, True]).reset_index(
        drop=True
    )
    page_size = 24
    paths: list[Path] = []
    pages = math.ceil(len(ordered) / page_size)
    for page in range(pages):
        frame = ordered.iloc[page * page_size : (page + 1) * page_size].copy()
        frame["label"] = frame["concept"].map(lambda value: textwrap.fill(str(value), 42))
        figure, axis = plt.subplots(
            figsize=(14, max(7, 0.44 * len(frame))), constrained_layout=True
        )
        colors = frame["f1_delta"].map(lambda value: "#2A9D8F" if value >= 0 else "#E76F51")
        axis.barh(frame["label"], frame["f1_delta"], color=colors)
        axis.axvline(0, color="#333333", linewidth=1)
        axis.set_xlabel("F1 delta (semantic/relation minus minimal JSON)")
        axis.set_ylabel("")
        axis.set_title(f"{title} — page {page + 1}/{pages}")
        axis.grid(axis="x", alpha=0.2)
        path = output / f"{prefix}_{page + 1:02d}_of_{pages:02d}.png"
        _save(figure, path)
        paths.append(path)
    return paths


def _paired_documents(paired: pd.DataFrame, output: Path) -> None:
    figure, axes = plt.subplots(1, 2, figsize=(15, 6), constrained_layout=True)
    sns.histplot(
        data=paired,
        x="document_f1_delta",
        bins=20,
        kde=True,
        color="#6C5CE7",
        ax=axes[0],
    )
    axes[0].axvline(0, color="#333333", linewidth=1)
    axes[0].set_title("Paired document-level F1 deltas")
    sns.scatterplot(
        data=paired,
        x="previous_f1",
        y="current_f1",
        hue="page_count",
        size="cargo_groups",
        palette="viridis",
        sizes=(35, 180),
        ax=axes[1],
    )
    axes[1].plot([0, 1], [0, 1], linestyle="--", color="#333333")
    axes[1].set_xlim(0, 1)
    axes[1].set_ylim(0, 1)
    axes[1].set_title("Same-document accuracy")
    axes[1].set_xlabel(_BASELINE)
    axes[1].set_ylabel(_RELATION)
    _save(figure, output / "12_paired_document_deltas.png")


def _complexity(paired: pd.DataFrame, output: Path) -> None:
    frames: list[pd.DataFrame] = []
    for field, label in (
        ("page_count", "Page count"),
        ("containers", "Container count"),
        ("cargo_groups", "Cargo-group count"),
        ("cargo_packages", "Package-fact count"),
    ):
        grouped = paired.groupby(field, as_index=False).agg(
            documents=("document_id", "count"),
            mean_delta=("document_f1_delta", "mean"),
        )
        grouped["dimension"] = label
        grouped["level"] = grouped[field].astype(str)
        frames.append(grouped[["dimension", "level", "documents", "mean_delta"]])
    data = pd.concat(frames, ignore_index=True)
    facets = sns.catplot(
        data=data,
        x="level",
        y="mean_delta",
        col="dimension",
        col_wrap=2,
        kind="bar",
        sharex=False,
        sharey=True,
        color="#6C5CE7",
        height=4.2,
        aspect=1.45,
    )
    facets.set_axis_labels("Observed complexity level", "Mean paired F1 delta")
    facets.set_titles("{col_name}")
    for axis in facets.axes.flat:
        axis.axhline(0, color="#333333", linewidth=1)
        axis.tick_params(axis="x", rotation=45)
    facets.figure.suptitle("Performance delta by document complexity", y=1.02, fontsize=16)
    _save(facets.figure, output / "13_complexity_strata.png")


def _bootstrap(bootstrap: pd.DataFrame, output: Path) -> None:
    data = bootstrap.copy()
    data["label"] = data["metric"].str.replace("_", " ").map(
        lambda value: textwrap.fill(value.title(), 36)
    )
    positions = list(range(len(data)))
    figure, axis = plt.subplots(
        figsize=(13, max(6, 0.75 * len(data))), constrained_layout=True
    )
    lower = data["point_estimate"] - data["ci95_lower"]
    upper = data["ci95_upper"] - data["point_estimate"]
    axis.errorbar(
        data["point_estimate"],
        positions,
        xerr=[lower, upper],
        fmt="o",
        color="#6C5CE7",
        ecolor="#6C5CE7",
        capsize=4,
    )
    axis.axvline(0, color="#333333", linewidth=1)
    axis.set_yticks(positions, data["label"])
    axis.set_xlabel("Paired delta with document-bootstrap 95% CI")
    axis.set_title("Uncertainty in measured improvement")
    _save(figure, output / "14_bootstrap_confidence_intervals.png")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--audit-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    arguments = parser.parse_args()
    audit_root = arguments.audit_root.resolve(strict=True)
    table_root = (audit_root / "tables").resolve(strict=True)
    output = arguments.output_root.resolve()
    output.mkdir(parents=True, exist_ok=False)
    sns.set_theme(style="whitegrid", context="notebook")

    history = pd.read_csv(table_root / "eval_history_comparison.csv")
    components = pd.read_csv(table_root / "component_comparison.csv")
    fields = pd.read_csv(table_root / "field_comparison.csv")
    stable = pd.read_csv(table_root / "stable_valid_field_comparison.csv")
    paired = pd.read_csv(table_root / "paired_document_metrics.csv")
    bootstrap = pd.read_csv(table_root / "bootstrap.csv")

    _line_plot(history, output)
    _validity_and_runtime(history, output)
    _component_plot(components, output)
    field_paths = _paged_field_deltas(
        fields, output, "04_all_field_deltas", "All field-level F1 deltas"
    )
    stable_paths = _paged_field_deltas(
        stable,
        output,
        "08_stable_valid_field_deltas",
        "All field deltas on documents valid in both runs",
    )
    _paired_documents(paired, output)
    _complexity(paired, output)
    _bootstrap(bootstrap, output)

    plot_paths = sorted(output.glob("*.png"))
    report = output / "VISUAL_SUPPLEMENT.md"
    report.write_text(
        "\n".join(
            [
                "# Matplotlib/Seaborn paired-run visual supplement",
                "",
                "This supplement re-renders the exact existing paired audit using Matplotlib "
                "and Seaborn. Field plots are paginated rather than top-N filtered, and every "
                "figure uses tight/constrained layout so long labels are not silently clipped.",
                "",
                f"- Source audit: `{audit_root}`",
                f"- Field rows rendered: {len(fields)} across {len(field_paths)} pages",
                (
                    f"- Stable-valid field rows rendered: {len(stable)} across "
                    f"{len(stable_paths)} pages"
                ),
                f"- Paired validation documents: {len(paired)}",
                f"- Plot files: {len(plot_paths)}",
                "",
                "The source audit REPORT.md remains the authoritative statistical and semantic "
                "interpretation; this directory is its untruncated visual companion.",
                "",
            ]
        ),
        encoding="utf-8",
    )
    manifest = {
        "schemaVersion": 1,
        "sourceAudit": str(audit_root),
        "sourceReportSha256": _sha256(audit_root / "REPORT.md"),
        "renderer": {"matplotlib": matplotlib.__version__, "seaborn": sns.__version__},
        "fieldRows": len(fields),
        "stableValidFieldRows": len(stable),
        "pairedDocuments": len(paired),
        "files": [
            {
                "path": path.name,
                "bytes": path.stat().st_size,
                "sha256": _sha256(path),
            }
            for path in [*plot_paths, report]
        ],
    }
    (output / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps({"output": str(output), "plots": len(plot_paths)}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
