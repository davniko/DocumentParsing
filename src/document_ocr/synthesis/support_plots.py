"""Matplotlib/seaborn visualizations for synthesis-readiness artifacts."""

from __future__ import annotations

import io
from collections.abc import Mapping, Sequence
from typing import Any


def _imports() -> tuple[Any, Any, Any]:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import pandas as pd  # type: ignore[import-untyped]
        import seaborn as sns  # type: ignore[import-untyped]
    except ImportError as error:
        raise RuntimeError(
            "synthesis readiness plots require matplotlib, pandas, and seaborn"
        ) from error
    sns.set_theme(style="whitegrid", context="notebook")
    return plt, pd, sns


def _png(figure: Any) -> bytes:
    stream = io.BytesIO()
    figure.savefig(
        stream,
        format="png",
        dpi=160,
        bbox_inches="tight",
        metadata={"Software": "document-ocr synthesis readiness"},
    )
    figure.clear()
    return stream.getvalue()


def readiness_plots(
    *,
    table_counts: Mapping[str, int],
    anchor_status_counts: Mapping[str, int],
    document_anchor_rows: Sequence[Mapping[str, Any]],
    cohort_rows: Sequence[Mapping[str, Any]],
    family_rows: Sequence[Mapping[str, Any]],
    template_cohort_rows: Sequence[Mapping[str, Any]],
) -> dict[str, bytes]:
    plt, pd, sns = _imports()
    plots: dict[str, bytes] = {}

    table_frame = pd.DataFrame(
        sorted(table_counts.items(), key=lambda row: row[1]), columns=["table", "rows"]
    )
    figure, axis = plt.subplots(figsize=(11, 8))
    sns.barplot(data=table_frame, x="rows", y="table", ax=axis, color="#2563EB")
    axis.set(title="B/L domain-table row counts", xlabel="rows (log scale)", ylabel="")
    axis.set_xscale("log")
    plots["01_domain_table_rows.png"] = _png(figure)
    plt.close(figure)

    status_frame = pd.DataFrame(
        sorted(anchor_status_counts.items(), key=lambda row: row[1], reverse=True),
        columns=["status", "anchors"],
    )
    figure, axis = plt.subplots(figsize=(10, 5))
    sns.barplot(data=status_frame, x="anchors", y="status", ax=axis, color="#0EA5E9")
    axis.set(title="Audited OCR-anchor location outcomes", xlabel="anchor rows", ylabel="")
    plots["02_anchor_location_status.png"] = _png(figure)
    plt.close(figure)

    coverage_frame = pd.DataFrame(document_anchor_rows)[
        ["evidence_coverage", "patchable_coverage"]
    ].melt(var_name="coverage", value_name="fraction")
    figure, axis = plt.subplots(figsize=(10, 5.5))
    sns.histplot(
        data=coverage_frame,
        x="fraction",
        hue="coverage",
        bins=20,
        multiple="layer",
        element="step",
        stat="count",
        common_norm=False,
        ax=axis,
    )
    axis.set(
        title="Per-document evidence and uniquely patchable coverage",
        xlabel="fraction of source-fact leaves",
        ylabel="documents",
        xlim=(0, 1.01),
    )
    plots["03_document_anchor_coverage.png"] = _png(figure)
    plt.close(figure)

    cohort_frame = pd.DataFrame(cohort_rows).melt(
        id_vars="cohort",
        value_vars=("source_documents", "anchor_ready_documents", "generator_ready_documents"),
        var_name="readiness",
        value_name="documents",
    )
    figure, axis = plt.subplots(figsize=(13, 7))
    sns.barplot(data=cohort_frame, x="documents", y="cohort", hue="readiness", ax=axis)
    axis.set(title="Cohort support gates", xlabel="documents", ylabel="")
    plots["04_cohort_readiness.png"] = _png(figure)
    plt.close(figure)

    family_frame = pd.DataFrame(family_rows).melt(
        id_vars="mutation_family",
        value_vars=("documents_with_values", "fully_patchable_documents"),
        var_name="readiness",
        value_name="documents",
    )
    figure, axis = plt.subplots(figsize=(12, 6))
    sns.barplot(data=family_frame, x="documents", y="mutation_family", hue="readiness", ax=axis)
    axis.set(title="Deterministic mutation-family anchor support", xlabel="documents", ylabel="")
    plots["05_mutation_family_support.png"] = _png(figure)
    plt.close(figure)

    template_frame = pd.DataFrame(template_cohort_rows)
    if not template_frame.empty:
        template_totals = (
            template_frame.groupby("template_id", as_index=False)["template_documents"]
            .max()
            .sort_values(["template_documents", "template_id"], ascending=[False, True])
            .head(25)
        )
        chosen = set(template_totals["template_id"])
        matrix = (
            template_frame[template_frame["template_id"].isin(chosen)]
            .pivot(index="template_id", columns="cohort", values="anchor_ready_fraction")
            .reindex(template_totals["template_id"])
            .fillna(0.0)
        )
        figure, axis = plt.subplots(
            figsize=(max(11, 1.2 * len(matrix.columns)), max(7, 0.35 * len(matrix)))
        )
        sns.heatmap(matrix, cmap="viridis", vmin=0, vmax=1, ax=axis, cbar_kws={"label": "fraction"})
        axis.set(
            title="Top-25 template proxies: anchor-ready share by cohort",
            xlabel="cohort",
            ylabel="template proxy",
        )
        plots["06_template_cohort_readiness.png"] = _png(figure)
        plt.close(figure)
    return plots
