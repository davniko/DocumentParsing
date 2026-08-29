#!/usr/bin/env python3
# ruff: noqa: E501
"""Audit data-scaling evidence between the 483-record and 1,157-record KIE runs."""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
import shutil
import tempfile
import time
from collections import Counter
from collections.abc import Iterable, Mapping, Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, cast

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from document_ocr.atomic import atomic_write_json
from document_ocr.hashing import sha256_file
from document_ocr.training.config import load_training_config
from document_ocr.training.metrics import PredictionAssessment, assess_prediction
from document_ocr.training.tasks import load_training_task

OLD_DATASET = Path(
    "artifacts/kie-training/datasets/mpci-bl-semantic-v3-relation-explicit-table-view-v1"
)
NEW_DATASET = Path(
    "artifacts/kie-training/datasets/"
    "mpci-bl-combined1157-task-facing-package-categories-v2/records.jsonl"
)
OLD_RUN = Path("artifacts/kie-training/t5gemma2-270m-lora-mpci-bl-relation-explicit-v3-v1")
NEW_RUN = Path("artifacts/kie-training/t5gemma2-270m-lora-mpci-bl-combined1157-task-facing-v2")
OLD_ANALYSIS = Path(
    "artifacts/kie-training/analysis/"
    "t5gemma2-270m-lora-mpci-bl-relation-explicit-v3-v1-completed-run-v2"
)
NEW_ANALYSIS = Path(
    "artifacts/kie-training/analysis/"
    "t5gemma2-270m-lora-mpci-bl-combined1157-task-facing-v2-deep-audit-v3"
)
NEW_EVALUATION = Path(
    "artifacts/kie-training/analysis/"
    "t5gemma2-270m-lora-mpci-bl-combined1157-task-facing-v2-"
    "checkpoint900-eval-v1"
)

INDEX = re.compile(r"\[[0-9]+\]")


def _json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def _csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    if not rows:
        raise ValueError(f"refusing to publish an empty table: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = list(rows[0])
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def _canonical(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def normalize_path(path: str) -> str:
    """Erase list indices so support can be compared across documents."""

    return INDEX.sub("[]", path)


def _flatten(value: Any, path: str = "$") -> dict[str, Any]:
    if isinstance(value, dict):
        result: dict[str, Any] = {}
        for key, child in value.items():
            result.update(_flatten(child, f"{path}.{key}"))
        return result
    if isinstance(value, list):
        result = {}
        for index, child in enumerate(value):
            result.update(_flatten(child, f"{path}[{index}]"))
        return result
    return {path: value}


def _normalized_leaf_paths(value: Any) -> list[str]:
    return [normalize_path(path) for path in _flatten(value, "$.documentPatch")]


def _micro_counts(items: Iterable[PredictionAssessment]) -> tuple[int, int, int]:
    assessments = list(items)
    true_positive = sum(
        len(item.predicted_field_values & item.reference_field_values) for item in assessments
    )
    predicted = sum(len(item.predicted_field_values) for item in assessments)
    reference = sum(len(item.reference_field_values) for item in assessments)
    return true_positive, predicted, reference


def _prf(counts: tuple[int, int, int]) -> tuple[float, float, float]:
    true_positive, predicted, reference = counts
    precision = true_positive / predicted if predicted else 0.0
    recall = true_positive / reference if reference else 0.0
    f1 = 2 * true_positive / (predicted + reference) if predicted + reference else 1.0
    return precision, recall, f1


def _paired_bootstrap(
    old: Sequence[PredictionAssessment],
    new: Sequence[PredictionAssessment],
    *,
    seed: int = 424,
    replicates: int = 100_000,
) -> dict[str, float | int]:
    if len(old) != len(new) or not old:
        raise ValueError("paired bootstrap requires equal non-empty assessment lists")
    old_counts = np.asarray([_micro_counts([item]) for item in old], dtype=np.int64)
    new_counts = np.asarray([_micro_counts([item]) for item in new], dtype=np.int64)
    rng = np.random.default_rng(seed)
    deltas = np.empty(replicates, dtype=np.float64)
    for start in range(0, replicates, 10_000):
        stop = min(start + 10_000, replicates)
        indices = rng.integers(0, len(old), size=(stop - start, len(old)))
        old_sample = old_counts[indices].sum(axis=1)
        new_sample = new_counts[indices].sum(axis=1)
        old_f1 = 2 * old_sample[:, 0] / (old_sample[:, 1] + old_sample[:, 2])
        new_f1 = 2 * new_sample[:, 0] / (new_sample[:, 1] + new_sample[:, 2])
        deltas[start:stop] = new_f1 - old_f1
    return {
        "replicates": replicates,
        "seed": seed,
        "lower_95": float(np.quantile(deltas, 0.025)),
        "upper_95": float(np.quantile(deltas, 0.975)),
        "probability_delta_above_zero": float(np.mean(deltas > 0)),
    }


def _macro_f1(items: Sequence[PredictionAssessment]) -> float:
    return float(np.mean([_prf(_micro_counts([item]))[2] for item in items]))


def _support(records: Sequence[Mapping[str, Any]]) -> tuple[Counter[str], Counter[str]]:
    documents: Counter[str] = Counter()
    values: Counter[str] = Counter()
    for record in records:
        paths = _normalized_leaf_paths(record["target"]["documentPatch"])
        values.update(paths)
        documents.update(set(paths))
    return documents, values


def _f(value: str | int | float) -> float:
    return float(value)


def _i(value: str | int) -> int:
    return int(value)


def _metric_by_name(rows: Sequence[Mapping[str, Any]], name: str) -> Mapping[str, Any]:
    for row in rows:
        if row.get("metric") == name:
            return row
    raise KeyError(name)


def _setup_style() -> None:
    sns.set_theme(style="whitegrid", context="talk")
    plt.rcParams.update(
        {
            "figure.dpi": 120,
            "savefig.dpi": 160,
            "axes.titleweight": "bold",
            "axes.spines.top": False,
            "axes.spines.right": False,
        }
    )


def _save(fig: plt.Figure, path: Path) -> None:
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


def _plot_headline(rows: Sequence[Mapping[str, Any]], path: Path) -> None:
    frame = pd.DataFrame(rows)
    metrics = ["precision", "recall", "f1", "json_valid", "schema_valid"]
    plot = frame.melt(id_vars="run", value_vars=metrics, var_name="metric", value_name="value")
    fig, axis = plt.subplots(figsize=(13, 7))
    sns.barplot(data=plot, x="metric", y="value", hue="run", ax=axis)
    axis.axhline(0.9, color="#b22222", linestyle="--", linewidth=1.5, label="0.90 target")
    axis.set_ylim(0, 1.02)
    axis.set_title("Headline metrics (directional only: validation sets differ)")
    axis.set_xlabel("")
    axis.set_ylabel("score")
    _save(fig, path)


def _plot_epoch(rows: Sequence[Mapping[str, Any]], path: Path) -> None:
    frame = pd.DataFrame(rows)
    fig, axis = plt.subplots(figsize=(12, 7))
    sns.lineplot(data=frame, x="epoch", y="f1", hue="run", marker="o", linewidth=2.5, ax=axis)
    axis.axhline(0.9, color="#b22222", linestyle="--", linewidth=1.5)
    axis.set_ylim(0.6, 0.92)
    axis.set_title("Generated exact field/value F1 by epoch")
    axis.set_ylabel("F1")
    _save(fig, path)


def _plot_precision_recall(rows: Sequence[Mapping[str, Any]], path: Path) -> None:
    frame = pd.DataFrame(rows)
    fig, axes = plt.subplots(1, 2, figsize=(15, 6), sharex=True, sharey=True)
    for axis, metric in zip(axes, ["precision", "recall"], strict=True):
        sns.lineplot(data=frame, x="epoch", y=metric, hue="run", marker="o", ax=axis)
        axis.set_title(metric.replace("_", " ").title())
        axis.set_ylim(0.5, 0.9)
    fig.suptitle("The current run's directional gain is primarily recall")
    _save(fig, path)


def _plot_paired(rows: Sequence[Mapping[str, Any]], path: Path) -> None:
    frame = pd.DataFrame(rows).sort_values("old_f1")
    fig, axis = plt.subplots(figsize=(12, 7))
    y = np.arange(len(frame))
    axis.hlines(y, frame["old_f1"], frame["new_f1"], color="#9aa0a6", linewidth=2)
    axis.scatter(frame["old_f1"], y, label="prior", s=90, color="#4c72b0")
    axis.scatter(frame["new_f1"], y, label="current", s=90, color="#dd8452")
    axis.set_yticks(y, [value[:16] + "…" for value in frame["document_id"]])
    axis.set_xlim(0.65, 1.0)
    axis.set_xlabel("document exact field/value F1")
    axis.set_title("Only six documents were held out in both runs")
    axis.legend()
    _save(fig, path)


def _plot_delta(rows: Sequence[Mapping[str, Any]], key: str, path: Path, title: str) -> None:
    frame = pd.DataFrame(rows).sort_values("f1_delta")
    fig, axis = plt.subplots(figsize=(12, max(6, 0.45 * len(frame))))
    colors = np.where(frame["f1_delta"] >= 0, "#55a868", "#c44e52")
    axis.barh(frame[key], frame["f1_delta"], color=colors)
    axis.axvline(0, color="black", linewidth=1)
    axis.set_xlabel("current F1 - prior F1 (different validation sets)")
    axis.set_title(title)
    _save(fig, path)


def _plot_support(rows: Sequence[Mapping[str, Any]], path: Path) -> None:
    frame = pd.DataFrame(rows)
    frame = frame[(frame["old_validation_values"] >= 5) & (frame["new_validation_values"] >= 5)]
    fig, axis = plt.subplots(figsize=(12, 8))
    sns.scatterplot(
        data=frame,
        x="train_document_growth",
        y="f1_delta",
        size="new_validation_values",
        hue="section",
        sizes=(25, 250),
        alpha=0.8,
        ax=axis,
    )
    axis.axhline(0, color="black", linewidth=1)
    axis.axvline(1, color="black", linewidth=1, linestyle=":")
    axis.set_xscale("log")
    axis.set_xlabel("training-document support growth (x)")
    axis.set_ylabel("directional F1 delta")
    axis.set_title("More support helped many fields, but not every relationship")
    axis.legend(bbox_to_anchor=(1.02, 1), loc="upper left", fontsize=9)
    _save(fig, path)


def _plot_overlap(rows: Sequence[Mapping[str, Any]], path: Path) -> None:
    frame = pd.DataFrame(rows)
    matrix = frame.pivot(index="prior_split", columns="current_split", values="documents")
    fig, axis = plt.subplots(figsize=(8, 6))
    sns.heatmap(matrix, annot=True, fmt="g", cmap="Blues", cbar=False, ax=axis)
    axis.set_title("Cross-run split migration")
    axis.set_xlabel("current split")
    axis.set_ylabel("prior split")
    _save(fig, path)


def _plot_drift(rows: Sequence[Mapping[str, Any]], path: Path) -> None:
    frame = pd.DataFrame(rows).head(15).sort_values("changed_leaves")
    fig, axis = plt.subplots(figsize=(12, 8))
    axis.barh(frame["field_path"], frame["changed_leaves"], color="#8172b3")
    axis.set_title("Shared-target revisions are concentrated in package topology")
    axis.set_xlabel("changed leaves across 483 shared documents")
    _save(fig, path)


def _plot_current_error_budget(rows: Sequence[Mapping[str, Any]], path: Path) -> None:
    frame = pd.DataFrame(rows).sort_values("max_f1_if_section_perfect")
    fig, axis = plt.subplots(figsize=(12, 8))
    axis.barh(frame["section"], frame["max_f1_if_section_perfect"], color="#64b5cd")
    axis.axvline(0.9, color="#b22222", linestyle="--", linewidth=1.5)
    axis.axvline(0.8324450043608878, color="black", linestyle=":", linewidth=1.5)
    axis.set_xlim(0.83, 0.91)
    axis.set_xlabel("overall F1 after a perfect replacement of that one section")
    axis.set_title("No single section can close the 0.90 gap")
    _save(fig, path)


def _plot_distributions(old_path: Path, new_path: Path, path: Path) -> None:
    old = pd.read_csv(old_path)[["field_value_f1"]].rename(columns={"field_value_f1": "f1"})
    old["run"] = "prior (n=60)"
    new = pd.read_csv(new_path)[["field_f1"]].rename(columns={"field_f1": "f1"})
    new["run"] = "current (n=100)"
    frame = pd.concat([old, new], ignore_index=True)
    fig, axis = plt.subplots(figsize=(11, 7))
    sns.violinplot(data=frame, x="run", y="f1", inner="quart", cut=0, ax=axis)
    sns.stripplot(data=frame, x="run", y="f1", color="black", alpha=0.35, size=3, ax=axis)
    axis.axhline(0.9, color="#b22222", linestyle="--", linewidth=1.5)
    axis.set_title("Document-level F1 distributions (not paired)")
    axis.set_xlabel("")
    _save(fig, path)


def _plot_scenarios(rows: Sequence[Mapping[str, Any]], path: Path) -> None:
    frame = pd.DataFrame(rows)
    fig, axis = plt.subplots(figsize=(11, 7))
    sns.barplot(data=frame, x="target_f1", y="estimated_train_documents", hue="scenario", ax=axis)
    axis.set_yscale("log")
    axis.set_ylabel("illustrative training documents (log scale)")
    axis.set_xlabel("target F1")
    axis.set_title("Two confounded points cannot identify a scaling law")
    _save(fig, path)


def _plot_experiment(rows: Sequence[Mapping[str, Any]], path: Path) -> None:
    frame = pd.DataFrame(rows)
    fig, axis = plt.subplots(figsize=(11, 7))
    axis.plot(frame["train_documents"], frame["optimizer_steps_25_epochs"], marker="o")
    for _, row in frame.iterrows():
        axis.annotate(
            row["stage"],
            (row["train_documents"], row["optimizer_steps_25_epochs"]),
            xytext=(6, 6),
            textcoords="offset points",
        )
    axis.set_xlabel("nested training documents")
    axis.set_ylabel("optimizer steps at 25 epochs, effective batch 24")
    axis.set_title("Proposed controlled learning-curve sweep")
    _save(fig, path)


def _manifest(directory: Path, started: float) -> dict[str, Any]:
    artifacts = []
    for path in sorted(directory.rglob("*")):
        if path.is_file() and path.name != "analysis_manifest.json":
            artifacts.append(
                {
                    "path": str(path.relative_to(directory)),
                    "bytes": path.stat().st_size,
                    "sha256": sha256_file(path),
                }
            )
    return {
        "schema_version": 1,
        "status": "complete",
        "created_at": datetime.now(UTC).isoformat(),
        "analysis_seconds": time.perf_counter() - started,
        "artifacts": artifacts,
        "artifact_count": len(artifacts),
        "verified_bytes": sum(item["bytes"] for item in artifacts),
    }


def analyze(project_root: Path, output_dir: Path) -> dict[str, Any]:
    started = time.perf_counter()
    root = project_root.resolve(strict=True)
    destination = output_dir.resolve()
    if destination.exists():
        raise ValueError(f"output directory already exists: {destination}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix=f".{destination.name}.", dir=destination.parent))
    tables = temporary / "tables"
    plots = temporary / "plots"
    tables.mkdir()
    plots.mkdir()
    try:
        old_train = _jsonl(root / OLD_DATASET / "train.jsonl")
        old_validation = _jsonl(root / OLD_DATASET / "validation.jsonl")
        new_records = _jsonl(root / NEW_DATASET)
        new_by_id = {row["documentId"]: row for row in new_records}
        old_by_id = {row["documentId"]: row for row in old_train + old_validation}
        new_report = _json(root / NEW_RUN / "dataset-report.json")
        partition = new_report["inspection"]["partition"]["outputs"]
        new_train_ids = set(partition["train"]["document_ids"])
        new_validation_ids = set(partition["validation"]["document_ids"])
        old_train_ids = {row["documentId"] for row in old_train}
        old_validation_ids = {row["documentId"] for row in old_validation}
        old_ids = old_train_ids | old_validation_ids

        overlap_rows = [
            {
                "prior_split": old_split,
                "current_split": new_split,
                "documents": len(old_values & new_values),
            }
            for old_split, old_values in (
                ("train", old_train_ids),
                ("validation", old_validation_ids),
            )
            for new_split, new_values in (
                ("train", new_train_ids),
                ("validation", new_validation_ids),
            )
        ]
        _write_csv(tables / "split_overlap.csv", overlap_rows)

        shared = sorted(old_ids & set(new_by_id))
        same_input = sum(
            old_by_id[item]["joinedRawTextSha256"] == new_by_id[item]["joinedRawTextSha256"]
            for item in shared
        )
        same_target = sum(
            _canonical(old_by_id[item]["target"]) == _canonical(new_by_id[item]["target"])
            for item in shared
        )
        mutually_held_out = sorted(old_validation_ids & new_validation_ids)
        mutually_unseen = sorted(new_validation_ids - old_train_ids)
        (tables / "mutually_held_out_6.txt").write_text(
            "\n".join(mutually_held_out) + "\n", encoding="utf-8"
        )
        (tables / "mutually_unseen_66.txt").write_text(
            "\n".join(mutually_unseen) + "\n", encoding="utf-8"
        )

        config = load_training_config(root / NEW_RUN / "config.yaml")
        task = load_training_task(root, config)
        old_predictions = {
            row["document_id"]: row
            for row in _jsonl(root / OLD_ANALYSIS / "recovered-predictions" / "validation.jsonl")
        }
        new_predictions = {
            row["document_id"]: row
            for row in _jsonl(root / NEW_EVALUATION / "predictions" / "validation.jsonl")
        }
        old_assessments: list[PredictionAssessment] = []
        new_assessments: list[PredictionAssessment] = []
        paired_rows = []
        for document_id in mutually_held_out:
            old_prediction = old_predictions[document_id]
            new_prediction = new_predictions[document_id]
            old_item = assess_prediction(
                old_prediction["generated_text"], old_prediction["reference_text"], task
            )
            new_item = assess_prediction(
                new_prediction["generated_text"], new_prediction["reference_text"], task
            )
            old_assessments.append(old_item)
            new_assessments.append(new_item)
            old_f1 = _prf(_micro_counts([old_item]))[2]
            new_f1 = _prf(_micro_counts([new_item]))[2]
            paired_rows.append(
                {
                    "document_id": document_id,
                    "old_f1": old_f1,
                    "new_f1": new_f1,
                    "delta": new_f1 - old_f1,
                    "old_json_valid": int(old_item.json_valid),
                    "new_json_valid": int(new_item.json_valid),
                    "old_schema_valid": int(old_item.schema_valid),
                    "new_schema_valid": int(new_item.schema_valid),
                    "source_equal": int(
                        old_by_id[document_id]["joinedRawTextSha256"]
                        == new_by_id[document_id]["joinedRawTextSha256"]
                    ),
                    "target_equal": int(
                        _canonical(old_by_id[document_id]["target"])
                        == _canonical(new_by_id[document_id]["target"])
                    ),
                }
            )
        _write_csv(tables / "paired_documents.csv", paired_rows)
        old_p, old_r, old_f1 = _prf(_micro_counts(old_assessments))
        new_p, new_r, new_f1 = _prf(_micro_counts(new_assessments))
        paired_bootstrap = _paired_bootstrap(old_assessments, new_assessments)
        paired_summary = [
            {
                "run": "prior",
                "documents": len(old_assessments),
                "true_positive": _micro_counts(old_assessments)[0],
                "predicted": _micro_counts(old_assessments)[1],
                "reference": _micro_counts(old_assessments)[2],
                "precision": old_p,
                "recall": old_r,
                "micro_f1": old_f1,
                "macro_document_f1": _macro_f1(old_assessments),
                "json_valid_documents": sum(item.json_valid for item in old_assessments),
                "schema_valid_documents": sum(item.schema_valid for item in old_assessments),
            },
            {
                "run": "current",
                "documents": len(new_assessments),
                "true_positive": _micro_counts(new_assessments)[0],
                "predicted": _micro_counts(new_assessments)[1],
                "reference": _micro_counts(new_assessments)[2],
                "precision": new_p,
                "recall": new_r,
                "micro_f1": new_f1,
                "macro_document_f1": _macro_f1(new_assessments),
                "json_valid_documents": sum(item.json_valid for item in new_assessments),
                "schema_valid_documents": sum(item.schema_valid for item in new_assessments),
            },
        ]
        _write_csv(tables / "paired_holdout_metrics.csv", paired_summary)

        old_summary = _json(root / OLD_ANALYSIS / "summary.json")
        new_summary = _json(root / NEW_ANALYSIS / "summary.json")
        old_bootstrap = _metric_by_name(old_summary["bootstrap"], "field_value_f1")
        headline_rows = [
            {
                "run": "prior 423/60",
                "train_documents": 423,
                "validation_documents": 60,
                "precision": old_summary["field_value_counts"]["true_positive"]
                / old_summary["field_value_counts"]["predicted"],
                "recall": old_summary["field_value_counts"]["true_positive"]
                / old_summary["field_value_counts"]["reference"],
                "f1": old_summary["bootstrap"][2]["point_estimate"],
                "f1_ci_lower": old_bootstrap["ci95_lower"],
                "f1_ci_upper": old_bootstrap["ci95_upper"],
                "json_valid": old_summary["document_validity_counts"]["json_valid"] / 60,
                "schema_valid": old_summary["document_validity_counts"]["schema_valid"] / 60,
            },
            {
                "run": "current 1057/100",
                "train_documents": 1057,
                "validation_documents": 100,
                "precision": new_summary["metrics"]["precision"],
                "recall": new_summary["metrics"]["recall"],
                "f1": new_summary["metrics"]["f1"],
                "f1_ci_lower": new_summary["bootstrap"]["lower_95"],
                "f1_ci_upper": new_summary["bootstrap"]["upper_95"],
                "json_valid": new_summary["metrics"]["json_valid"],
                "schema_valid": new_summary["metrics"]["schema_valid"],
            },
        ]
        _write_csv(tables / "headline_metrics.csv", headline_rows)

        old_history = _csv(root / OLD_ANALYSIS / "tables" / "eval_history.csv")
        new_history = _csv(root / NEW_ANALYSIS / "tables" / "eval_history.csv")
        history_rows = []
        for run_name, rows in (("prior", old_history), ("current", new_history)):
            for row in rows:
                history_rows.append(
                    {
                        "run": run_name,
                        "epoch": _f(row["epoch"]),
                        "step": _i(row["step"]),
                        "precision": _f(row["eval_field_value_precision"]),
                        "recall": _f(row["eval_field_value_recall"]),
                        "f1": _f(row["eval_field_value_f1"]),
                        "json_valid": _f(row["eval_json_valid"]),
                        "schema_valid": _f(row["eval_schema_valid"]),
                        "eval_loss": _f(row["eval_loss"]),
                        "cargo_relation_f1": _f(row["eval_cargo_relation_f1"]),
                        "category_f1": _f(row["eval_category_value_f1"]),
                    }
                )
        _write_csv(tables / "epoch_progression.csv", history_rows)

        old_sections = {
            row["section"]: row
            for row in _csv(root / OLD_ANALYSIS / "tables" / "section_metrics.csv")
        }
        new_sections = {
            row["section"]: row
            for row in _csv(root / NEW_ANALYSIS / "tables" / "section_metrics.csv")
        }
        section_rows = [
            {
                "section": section,
                "old_f1": _f(old_sections[section]["f1"]),
                "new_f1": _f(new_sections[section]["f1"]),
                "f1_delta": _f(new_sections[section]["f1"]) - _f(old_sections[section]["f1"]),
                "old_reference_values": _i(old_sections[section]["reference_values"]),
                "new_reference_values": _i(new_sections[section]["reference_values"]),
            }
            for section in sorted(set(old_sections) & set(new_sections))
        ]
        _write_csv(tables / "section_directional_comparison.csv", section_rows)

        old_relations = {
            row["relation_fact"]: row
            for row in _csv(
                root / "artifacts/kie-training/analysis/"
                "t5gemma2-270m-lora-mpci-bl-relation-explicit-v3-v1-"
                "vs-combined487-v2-paired-audit-v1/tables/relation_facts.csv"
            )
        }
        new_relations = {
            row["fact_type"]: row
            for row in _csv(root / NEW_ANALYSIS / "tables" / "relation_metrics.csv")
        }
        relation_rows = [
            {
                "fact_type": fact,
                "old_f1": _f(old_relations[fact]["validation_f1"]),
                "new_f1": _f(new_relations[fact]["f1"]),
                "f1_delta": _f(new_relations[fact]["f1"])
                - _f(old_relations[fact]["validation_f1"]),
                "old_train_documents": _i(old_relations[fact]["train_documents"]),
                "old_train_facts": _i(old_relations[fact]["train_facts"]),
                "new_validation_reference": _i(new_relations[fact]["reference_facts"]),
            }
            for fact in sorted(set(old_relations) & set(new_relations))
        ]
        _write_csv(tables / "relation_directional_comparison.csv", relation_rows)

        old_support_docs, old_support_values = _support(old_train)
        new_train = [new_by_id[item] for item in new_train_ids]
        new_support_docs, new_support_values = _support(new_train)
        old_fields = {
            row["field_path"]: row
            for row in _csv(root / OLD_ANALYSIS / "tables" / "field_metrics.csv")
        }
        new_fields = {
            row["field_path"]: row
            for row in _csv(root / NEW_ANALYSIS / "tables" / "field_metrics.csv")
        }
        field_rows = []
        for field in sorted(set(old_fields) & set(new_fields)):
            old_documents = old_support_docs[field]
            new_documents = new_support_docs[field]
            field_rows.append(
                {
                    "field_path": field,
                    "section": field.split(".documentPatch.", 1)[-1].split(".", 1)[0],
                    "old_train_documents": old_documents,
                    "new_train_documents": new_documents,
                    "train_document_growth": new_documents / old_documents
                    if old_documents
                    else math.nan,
                    "old_train_values": old_support_values[field],
                    "new_train_values": new_support_values[field],
                    "old_validation_values": _i(old_fields[field]["reference_values"]),
                    "new_validation_values": _i(new_fields[field]["validation_values"]),
                    "old_f1": _f(old_fields[field]["f1"]),
                    "new_f1": _f(new_fields[field]["f1"]),
                    "f1_delta": _f(new_fields[field]["f1"]) - _f(old_fields[field]["f1"]),
                }
            )
        _write_csv(tables / "field_support_directional_comparison.csv", field_rows)

        drift_counter: Counter[str] = Counter()
        changed_documents = 0
        for document_id in shared:
            old_flat = _flatten(old_by_id[document_id]["target"])
            new_flat = _flatten(new_by_id[document_id]["target"])
            changed = False
            for path in set(old_flat) | set(new_flat):
                if old_flat.get(path, object()) != new_flat.get(path, object()):
                    drift_counter[normalize_path(path)] += 1
                    changed = True
            changed_documents += int(changed)
        drift_rows = [
            {"field_path": field, "changed_leaves": count}
            for field, count in drift_counter.most_common()
        ]
        _write_csv(tables / "target_drift_paths.csv", drift_rows)

        base_counts = (
            _i(new_summary["counts"]["true_positive"]),
            _i(new_summary["counts"]["predicted"]),
            _i(new_summary["counts"]["reference"]),
        )
        current_error_budget = []
        for section, row in new_sections.items():
            section_tp = _i(row["true_positive_values"])
            section_predicted = _i(row["predicted_values"])
            section_reference = _i(row["reference_values"])
            counterfactual = (
                base_counts[0] - section_tp + section_reference,
                base_counts[1] - section_predicted + section_reference,
                base_counts[2],
            )
            current_error_budget.append(
                {
                    "section": section,
                    "current_f1": _prf(base_counts)[2],
                    "max_f1_if_section_perfect": _prf(counterfactual)[2],
                    "maximum_delta": _prf(counterfactual)[2] - _prf(base_counts)[2],
                }
            )
        _write_csv(tables / "current_section_error_budget.csv", current_error_budget)

        n1, n2 = 423, 1057
        f1_1, f1_2 = headline_rows[0]["f1"], headline_rows[1]["f1"]
        error_exponent = -math.log((1 - f1_2) / (1 - f1_1)) / math.log(n2 / n1)
        log_slope = (f1_2 - f1_1) / math.log(n2 / n1)
        scenario_rows = []
        for target in (0.90, 0.95):
            scenario_rows.extend(
                [
                    {
                        "scenario": "error power law",
                        "target_f1": target,
                        "estimated_train_documents": n2
                        * ((1 - f1_2) / (1 - target)) ** (1 / error_exponent),
                        "planning_status": "illustrative_only_confounded_points",
                    },
                    {
                        "scenario": "log-linear F1",
                        "target_f1": target,
                        "estimated_train_documents": n2 * math.exp((target - f1_2) / log_slope),
                        "planning_status": "illustrative_only_confounded_points",
                    },
                ]
            )
        _write_csv(tables / "scaling_scenario_sensitivity.csv", scenario_rows)

        experiment_sizes = (211, 423, 634, 846, 1057)
        experiment_rows = [
            {
                "stage": f"{round(size / 1057 * 100)}%",
                "train_documents": size,
                "effective_batch": 24,
                "optimizer_steps_25_epochs": math.ceil(size / 24) * 25,
                "fixed_development_documents": 100,
                "seeds_screening": 1,
                "seeds_confirmation": 3 if size in {423, 1057} else 0,
            }
            for size in experiment_sizes
        ]
        _write_csv(tables / "controlled_learning_curve_plan.csv", experiment_rows)

        contract_rows = [
            {"property": "prior train records", "value": 423},
            {"property": "prior validation records", "value": 60},
            {"property": "current train records", "value": 1057},
            {"property": "current validation records", "value": 100},
            {"property": "shared records", "value": len(shared)},
            {"property": "shared raw OCR identical", "value": same_input},
            {"property": "shared targets identical", "value": same_target},
            {"property": "shared targets revised", "value": changed_documents},
            {"property": "mutually held out", "value": len(mutually_held_out)},
            {"property": "mutually unseen current validation", "value": len(mutually_unseen)},
            {
                "property": "prior val moved to current train",
                "value": len(old_validation_ids & new_train_ids),
            },
            {
                "property": "current val seen in prior train",
                "value": len(new_validation_ids & old_train_ids),
            },
            {
                "property": "prior prompt sha256",
                "value": sha256_file(root / OLD_RUN / "prompt.txt"),
            },
            {
                "property": "current prompt sha256",
                "value": sha256_file(root / NEW_RUN / "prompt.txt"),
            },
        ]
        _write_csv(tables / "run_contract_comparison.csv", contract_rows)

        _setup_style()
        _plot_headline(headline_rows, plots / "01_headline_directional_metrics.png")
        _plot_epoch(history_rows, plots / "02_epoch_f1_progression.png")
        _plot_precision_recall(history_rows, plots / "03_precision_recall_progression.png")
        _plot_paired(paired_rows, plots / "04_paired_six_documents.png")
        _plot_delta(
            section_rows,
            "section",
            plots / "05_section_directional_deltas.png",
            "Section F1 changes (directional, not paired)",
        )
        _plot_delta(
            relation_rows,
            "fact_type",
            plots / "06_relation_directional_deltas.png",
            "Relation F1 changes: local links improve, package allocation does not",
        )
        _plot_support(field_rows, plots / "07_support_growth_vs_f1_delta.png")
        _plot_overlap(overlap_rows, plots / "08_split_overlap.png")
        _plot_drift(drift_rows, plots / "09_target_drift.png")
        _plot_current_error_budget(current_error_budget, plots / "10_current_error_budget.png")
        _plot_distributions(
            root / OLD_ANALYSIS / "tables" / "document_metrics.csv",
            root / NEW_ANALYSIS / "tables" / "document_metrics.csv",
            plots / "11_document_f1_distributions.png",
        )
        _plot_scenarios(scenario_rows, plots / "12_scaling_scenario_sensitivity.png")
        _plot_experiment(experiment_rows, plots / "13_controlled_learning_curve_plan.png")

        summary = {
            "schema_version": 1,
            "status": "complete",
            "created_at": datetime.now(UTC).isoformat(),
            "headline": {
                "prior_f1": f1_1,
                "current_f1": f1_2,
                "directional_delta": f1_2 - f1_1,
                "directly_comparable": False,
                "reason": "split migration, target revisions, schema expansion, and validation identity drift",
            },
            "overlap": {
                "shared_documents": len(shared),
                "raw_ocr_identical": same_input,
                "targets_identical": same_target,
                "targets_revised": changed_documents,
                "mutually_held_out": len(mutually_held_out),
                "mutually_unseen_current_validation": len(mutually_unseen),
                "prior_validation_moved_to_current_train": len(old_validation_ids & new_train_ids),
                "current_validation_seen_in_prior_train": len(new_validation_ids & old_train_ids),
            },
            "paired_six": {
                "prior": paired_summary[0],
                "current": paired_summary[1],
                "micro_f1_delta": new_f1 - old_f1,
                "macro_document_f1_delta": _macro_f1(new_assessments) - _macro_f1(old_assessments),
                "documents_improved": sum(row["delta"] > 0 for row in paired_rows),
                "documents_regressed": sum(row["delta"] < 0 for row in paired_rows),
                "bootstrap": paired_bootstrap,
                "generalizable": False,
                "reason": "n=6",
            },
            "scaling_scenarios": scenario_rows,
            "controlled_learning_curve": experiment_rows,
            "current_gap": {
                "f1": new_summary["metrics"]["f1"],
                "additional_true_positives_at_fixed_totals_for_090": new_summary["gap_to_090"][
                    "additional_true_positives"
                ],
                "json_valid": new_summary["metrics"]["json_valid"],
                "schema_valid": new_summary["metrics"]["schema_valid"],
                "cargo_relation_f1": new_summary["metrics"]["cargo_relation_f1"],
                "category_f1": new_summary["metrics"]["category_value_f1"],
            },
            "evaluation_policy": {
                "decoding": "unconstrained_greedy",
                "constrained_decoding_role": "serving_guardrail_only",
            },
        }
        atomic_write_json(temporary / "summary.json", summary)

        report = f"""# Data-scaling and reliability diagnosis for MPCI B/L KIE

Generated at `{summary["created_at"]}` from the completed 483-record relation-explicit run and the retained checkpoint-900 evaluation of the 1,157-record task-facing run.

## Executive answer

The current system is better on its own benchmark: exact field/value F1 rose from **{f1_1:.4f}** to **{f1_2:.4f}** (+{f1_2 - f1_1:.4f}), recall rose strongly, JSON validity rose from **{headline_rows[0]["json_valid"]:.1%}** to **{headline_rows[1]["json_valid"]:.1%}**, and several ordinary extraction sections improved. That is useful product evidence.

It is **not evidence that doubling the data caused the gain**. All 483 prior documents are in the current corpus; 54/60 prior validation documents moved into current training; 34/100 current validation documents were in prior training; 43 shared targets were revised; and the injected schema expanded. The two headline validation scores are therefore neither paired nor independent holdout estimates.

The only already-generated clean overlap contains six documents held out in both runs, with identical OCR and targets. On those six, micro-F1 changes **{old_f1:.4f} → {new_f1:.4f}** ({new_f1 - old_f1:+.4f}); five regress and one improves. The paired bootstrap interval is **[{paired_bootstrap["lower_95"]:.4f}, {paired_bootstrap["upper_95"]:.4f}]**, but six documents are too few to generalize. The correct conclusion is **mixed evidence, not saturation and not a proven scaling gain**.

Constrained decoding is intentionally excluded from this evaluation program. Evaluation must continue with unconstrained greedy generation so JSON/schema validity remains a learned capability. Grammar/schema constraints belong only to live serving as a guardrail and must be reported separately.

## What did improve directionally

At common epoch checkpoints, the prior/current headline curves are:

| epoch | prior F1 | current F1 | prior recall | current recall |
|---:|---:|---:|---:|---:|
| 5 | 0.6791 | 0.6488 | 0.5875 | 0.5686 |
| 10 | 0.7734 | 0.7783 | 0.7079 | 0.7391 |
| 15 | 0.7798 | 0.8103 | 0.7294 | 0.7821 |
| 20 | 0.7953 | 0.8294 | 0.7485 | 0.8243 |

The larger run learns more slowly at first, then reaches substantially higher recall. This is consistent with useful additional supervision, but different validation sets prevent a causal attribution.

Directional section gains are strongest for containers, cargo packages/groups, and several party/location fields. Relation behavior is more revealing: `group_has_package` and `group_uses_container` improve, while `allocation_coverage`, `allocation_covers_package`, `container_has_package`, and allocation quantity remain flat or regress. Added volume helps local extraction and simple links more than the long-range package/allocation topology.

## Why this is not ordinary model saturation

The current curve still rises **0.6488 → 0.7783 → 0.8103 → 0.8294** at epochs 5/10/15/20, so the model has not reached a flat optimization ceiling. However, teacher-forced eval loss bottoms at epoch 10 and rises sharply while training loss approaches zero. More epochs on the same examples are overfitting, not a route to 0.90.

The limiting evidence is structural and distributional:

- Current cargo-relation F1 is **{new_summary["metrics"]["cargo_relation_f1"]:.4f}**, versus overall **{f1_2:.4f}**.
- Multi-cargo documents score about 0.70 while one-cargo documents score about 0.86.
- Schema-valid documents still score only about 0.868 micro-F1; syntax/schema learning is necessary but not sufficient.
- 40.9% of validation leaves contain values unseen under the same field path. Copying and grouping must generalize across templates rather than memorize values.
- The 0.90 target requires **{new_summary["gap_to_090"]["additional_true_positives"]}** additional exact matches at fixed prediction/reference totals.
- Perfecting any single section cannot reach 0.90. Cargo groups, packages, allocations, and parties must improve together.

This is best described as **representation/data-coverage saturation under the current training recipe**, not proof that the 270M-class architecture is saturated.

## Can sample count alone get us to 0.90 or 0.95?

Not yet knowably. Fitting two confounded points produces incompatible answers:

| illustrative fit | 0.90 | 0.95 |
|---|---:|---:|
| error power law | {scenario_rows[0]["estimated_train_documents"]:,.0f} | {scenario_rows[2]["estimated_train_documents"]:,.0f} |
| log-linear F1 | {scenario_rows[1]["estimated_train_documents"]:,.0f} | {scenario_rows[3]["estimated_train_documents"]:,.0f} |

These are sensitivity calculations, **not forecasts**. Their disagreement is the evidence: two changing-contract runs cannot identify a scaling curve. Do not use either number as a data-budget commitment.

The present evidence supports a practical hypothesis: a diverse, quality-controlled corpus in the high thousands may be enough to cross 0.90 **after** cargo representation and curriculum improvements shift the curve. Nothing here supports promising 0.95 from raw duplication or ordinary extra epochs. The planned ~10k augmented corpus is a reasonable upper validation stage, not the next blind run.

## Decisive experiment: controlled learning curve

Freeze the current task-facing target, semantic prompt, unconstrained greedy evaluator, 100-document development set, optimizer, LoRA topology, and checkpoint-selection rule. Train deterministic nested subsets of **211, 423, 634, 846, and 1,057** documents. Use one seed for screening, then repeat the 423- and 1,057-document endpoints with three seeds. Evaluate every 1-2 epochs and select on development F1 with schema validity as a co-gate.

This answers three different questions:

1. **Positive slope at 1,057:** add real, template-diverse data and extend the curve to ~2k, ~4k, then ~8-10k.
2. **Flat slope but training improves:** capacity/adaptation or target topology is limiting; test PiSSA/rank and compact cargo structure before labeling thousands more.
3. **Training and development both flat:** the representation or input lacks the grouping signal; prioritize a compact local cargo tree and GLM-OCR table view.

For a true reliability claim, create a separate carrier/template-grouped test set that is never used for prompt, schema, checkpoint, or augmentation decisions. The current 100-document validation has now served as development data.

## Highest-value improvement sequence

1. **Freeze a clean benchmark and run the 66-document cross-evaluation.** Current validation minus prior training gives 66 documents unseen by both adapters (60 new + six common validation). Evaluate both adapters afresh under the current prompt/schema/targets and identical unconstrained decoding. This compares deployed-system fitness but still bundles data plus target revisions.
2. **Run the controlled nested learning curve.** This is the only reliable way to estimate marginal value per additional audited document and fit a sample-scaling curve.
3. **A/B a compact, source-ordered cargo tree.** Keep package facts and allocations local to their cargo group, then deterministically project to MPCI arrays. Text2Event's sequence-to-structure work supports compact representations and substructure-to-full-structure curricula: https://aclanthology.org/2021.acl-long.217/.
4. **Add a real curriculum.** Train scalar/party fields and single-cargo structures before multi-cargo/multi-container documents, finishing on full-document targets. This adds supervised structural practice without changing one-pass serving.
5. **A/B concise field/relation semantics against the full schema dump.** UIE's structural schema instructor is the relevant precedent: https://aclanthology.org/2022.acl-long.395/.
6. **Add GLM-OCR table text as an input ablation.** The weak fields are precisely row grouping, weights, descriptions, packages, and allocations. LMDX and VRDU identify layout, hierarchical entities, tables, and unseen templates as central document-IE difficulties: https://research.google/pubs/lmdx-language-model-based-document-information-extraction-and-localization/ and https://arxiv.org/abs/2211.15421.
7. **Scale diversity, not duplicates.** Prioritize underrepresented carrier/templates and multi-cargo/multi-container/package-allocation signatures. IEPile provides supporting evidence that cleaned, standardized, schema-conditioned IE supervision improves IE and generalization: https://aclanthology.org/2024.acl-short.13/.
8. **Then run PEFT capacity diagnostics.** Current rank-32 rsLoRA already covers q/k/v/o and gate/up/down in encoder and decoder. Compare rank 64, PiSSA initialization, and one full-text-model fine-tune diagnostic at fixed data. Serving latency is unchanged after adapter merge; this test isolates adaptation capacity. PiSSA reports stronger convergence/performance than ordinary LoRA across its evaluated tasks: https://arxiv.org/abs/2404.02948.

T5Gemma 2 270M-270M is a **pretrained**, not instruction-tuned, checkpoint. Learning the task contract from supervised examples is therefore expected; its report describes the released checkpoints and post-training setting: https://arxiv.org/abs/2512.14856.

## Acceptance gates toward a reliable system

- Unconstrained JSON validity ≥ 0.995 and schema validity ≥ 0.98 on development and frozen test.
- Exact field/value micro-F1 ≥ 0.90 first; then ≥ 0.95 as a stretch target.
- Cargo relation F1 ≥ 0.90, not hidden by strong route/date fields.
- Report carrier/template-seen and carrier/template-unseen results separately.
- Report one-cargo and multi-cargo/multi-container strata separately.
- Use at least three training seeds for the chosen recipe and bootstrap by document/template.
- Keep constrained serving results as an additional operational metric, never as the model-learning score.

## Artifact map

- `summary.json`: machine-readable conclusions and controlled experiment plan.
- `tables/run_contract_comparison.csv`: exact contract and overlap facts.
- `tables/split_overlap.csv`: cross-run split migration.
- `tables/paired_holdout_metrics.csv` and `paired_documents.csv`: the six clean paired examples.
- `tables/mutually_unseen_66.txt`: benchmark IDs for the next cross-evaluation.
- `tables/field_support_directional_comparison.csv`: training support and field F1 changes.
- `tables/section_directional_comparison.csv` and `relation_directional_comparison.csv`: structural gains and regressions.
- `tables/scaling_scenario_sensitivity.csv`: deliberately non-authoritative two-point extrapolations.
- `tables/controlled_learning_curve_plan.csv`: the experiment needed for a real sample-count estimate.
- `plots/`: 13 matplotlib/seaborn diagnostics.
"""
        (temporary / "REPORT.md").write_text(report, encoding="utf-8")
        manifest = _manifest(temporary, started)
        atomic_write_json(temporary / "analysis_manifest.json", manifest)
        temporary.rename(destination)
        return cast(dict[str, Any], summary)
    except BaseException:
        shutil.rmtree(temporary, ignore_errors=True)
        raise


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    arguments = _parse_args()
    result = analyze(arguments.project_root, arguments.output_dir)
    print(json.dumps(result, ensure_ascii=False, sort_keys=True, separators=(",", ":")))


if __name__ == "__main__":
    main()
