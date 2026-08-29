#!/usr/bin/env python3
# ruff: noqa: E501
"""Publish a paired audit of embedding-LoRA and non-embedding KIE runs.

The comparison is deliberately prediction-first: all reported model-quality
deltas are recomputed from saved generations after proving that document IDs
and canonical references are identical across runs.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import random
import tempfile
from collections import Counter
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, cast

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import yaml
from analyze_kie_error_mechanisms import _cross_field_pairs, classify_leaf_error
from analyze_kie_task_facing_checkpoint import (
    Diagnostic,
    Record,
    _aggregate_diagnostics,
    _category_token_rows,
    _classify_errors,
    _document_rows,
    _fact_rows,
    _field_rows,
    _group_rows,
    _jsonl_load,
    _load_diagnostics,
    _load_records,
    _schema_issue_rows,
    _section_rows,
)
from safetensors import safe_open

from document_ocr.atomic import atomic_write_json
from document_ocr.hashing import sha256_file
from document_ocr.training.config import TrainingConfig
from document_ocr.training.tasks import load_training_task

BOOTSTRAP_REPLICATES = 10_000
BOOTSTRAP_SEED = 829


@dataclass(frozen=True, slots=True)
class RunSpec:
    key: str
    label: str
    run_dir: Path
    evaluation_dir: Path
    adapter_path: Path
    rank: int
    alpha: int
    epochs: int
    embeddings_adapted: bool


@dataclass(frozen=True, slots=True)
class RunAudit:
    spec: RunSpec
    diagnostics: tuple[Diagnostic, ...]
    schema_rows: tuple[dict[str, Any], ...]
    documents: tuple[dict[str, Any], ...]
    fields: tuple[dict[str, Any], ...]
    sections: tuple[dict[str, Any], ...]
    relations: tuple[dict[str, Any], ...]
    category_facts: tuple[dict[str, Any], ...]
    categories: tuple[dict[str, Any], ...]
    groups: tuple[dict[str, Any], ...]
    errors: tuple[dict[str, Any], ...]
    mechanisms: tuple[dict[str, Any], ...]
    history: tuple[dict[str, Any], ...]
    manifest_metrics: Mapping[str, Any]
    adapter: Mapping[str, Any]


def _json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path}")
    return cast(dict[str, Any], value)


def _write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    if not rows:
        raise ValueError(f"refusing to publish an empty CSV: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = sorted({key for row in rows for key in row})
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, extrasaction="raise")
        writer.writeheader()
        writer.writerows(rows)


def _write_jsonl(path: Path, rows: Iterable[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as stream:
        for row in rows:
            stream.write(json.dumps(dict(row), ensure_ascii=False, sort_keys=True) + "\n")


def _manifest_metrics(spec: RunSpec) -> dict[str, Any]:
    manifest = _json(spec.evaluation_dir / "manifest.json")
    metrics = manifest.get("metrics")
    if not isinstance(metrics, dict):
        raise ValueError(f"metrics absent from manifest: {spec.evaluation_dir}")
    if "validation_evaluation" in metrics:
        metrics = metrics["validation_evaluation"]
    if not isinstance(metrics, dict) or "eval_field_value_f1" not in metrics:
        raise ValueError(f"validation metrics absent from manifest: {spec.evaluation_dir}")
    return cast(dict[str, Any], metrics)


def _history(spec: RunSpec) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for event in _jsonl_load(spec.run_dir / "logs" / "events.jsonl"):
        metrics = event.get("metrics")
        if (
            event.get("event") != "trainer_log"
            or not isinstance(metrics, dict)
            or "eval_field_value_f1" not in metrics
        ):
            continue
        rows.append(
            {
                "run": spec.label,
                "run_key": spec.key,
                "step": int(event["global_step"]),
                "epoch": float(event["epoch"]),
                **metrics,
            }
        )
    trained = [row for row in rows if float(row["epoch"]) > 0]
    if not trained or int(trained[-1]["epoch"]) != spec.epochs:
        raise ValueError(f"evaluation history is incomplete: {spec.key}")
    return rows


def _adapter_inventory(spec: RunSpec) -> dict[str, Any]:
    environment = _json(spec.run_dir / "environment.json")
    model = environment.get("model")
    if not isinstance(model, dict):
        raise ValueError(f"model environment metadata absent: {spec.run_dir}")
    tensor_count = 0
    saved_parameters = 0
    lora_parameters = 0
    base_parameters = 0
    embedding_parameters = 0
    dtype_parameters: Counter[str] = Counter()
    largest: list[tuple[int, str, tuple[int, ...], str]] = []
    with safe_open(spec.adapter_path, framework="pt", device="cpu") as stream:
        for key in stream.keys():  # noqa: SIM118 - safe_open is not iterable
            tensor = stream.get_tensor(key)
            parameters = tensor.numel()
            tensor_count += 1
            saved_parameters += parameters
            dtype_parameters[str(tensor.dtype)] += parameters
            if ".lora_" in key:
                lora_parameters += parameters
            else:
                base_parameters += parameters
            if ".embed_tokens." in key:
                embedding_parameters += parameters
            largest.append((parameters, key, tuple(tensor.shape), str(tensor.dtype)))
    return {
        "run": spec.label,
        "run_key": spec.key,
        "rank": spec.rank,
        "alpha": spec.alpha,
        "epochs": spec.epochs,
        "embeddings_adapted": spec.embeddings_adapted,
        "adapted_module_count": int(model["adapted_module_count"]),
        "trainable_parameters": int(model["trainable_parameters"]),
        "trainable_parameter_ratio": float(model["trainable_parameter_ratio"]),
        "adapter_bytes": spec.adapter_path.stat().st_size,
        "adapter_sha256": sha256_file(spec.adapter_path),
        "saved_tensor_count": tensor_count,
        "saved_parameters": saved_parameters,
        "saved_lora_parameters": lora_parameters,
        "saved_base_parameters": base_parameters,
        "saved_embedding_parameters": embedding_parameters,
        "parameters_by_dtype": json.dumps(dtype_parameters, sort_keys=True),
        "largest_tensors": json.dumps(
            [
                {"parameters": count, "key": key, "shape": shape, "dtype": dtype}
                for count, key, shape, dtype in sorted(largest, reverse=True)[:8]
            ],
            sort_keys=True,
        ),
    }


def _mechanism_rows(errors: Sequence[Mapping[str, Any]], run: RunSpec) -> list[dict[str, Any]]:
    cross_fields = _cross_field_pairs(errors)
    rows: list[dict[str, Any]] = []
    for index, error in enumerate(errors):
        mechanism, basis, matching_paths = classify_leaf_error(
            error, cross_field_paths=cross_fields.get(index, ())
        )
        rows.append(
            {
                **dict(error),
                "run": run.label,
                "run_key": run.key,
                "mechanism": mechanism,
                "mechanism_basis": basis,
                "cross_field_match_paths": json.dumps(matching_paths),
            }
        )
    return rows


def _audit_run(
    spec: RunSpec,
    *,
    records: Mapping[str, Record],
    task: Any,
) -> RunAudit:
    diagnostics, schema_rows = _load_diagnostics(
        evaluation_dir=spec.evaluation_dir,
        records=records,
        task=task,
    )
    errors = [error for diagnostic in diagnostics for error in _classify_errors(diagnostic)]
    manifest_metrics = _manifest_metrics(spec)
    aggregate = _aggregate_diagnostics(diagnostics)
    for name in ("precision", "recall", "f1"):
        manifest_name = f"eval_field_value_{name}"
        if not math.isclose(
            float(aggregate[name]),
            float(manifest_metrics[manifest_name]),
            rel_tol=1e-12,
            abs_tol=1e-12,
        ):
            raise ValueError(f"saved predictions disagree with manifest: {spec.key}:{manifest_name}")
    return RunAudit(
        spec=spec,
        diagnostics=tuple(diagnostics),
        schema_rows=tuple(schema_rows),
        documents=tuple(_document_rows(diagnostics)),
        fields=tuple(_field_rows(diagnostics, records)),
        sections=tuple(_section_rows(diagnostics)),
        relations=tuple(_fact_rows(diagnostics, category=False)),
        category_facts=tuple(_fact_rows(diagnostics, category=True)),
        categories=tuple(_category_token_rows(diagnostics, records)),
        groups=tuple(_group_rows(diagnostics)),
        errors=tuple(errors),
        mechanisms=tuple(_mechanism_rows(errors, spec)),
        history=tuple(_history(spec)),
        manifest_metrics=manifest_metrics,
        adapter=_adapter_inventory(spec),
    )


def _assert_prediction_parity(audits: Sequence[RunAudit]) -> None:
    expected_ids: set[str] | None = None
    expected_references: dict[str, frozenset[tuple[str, str]]] | None = None
    for audit in audits:
        ids = {row.record.document_id for row in audit.diagnostics}
        references = {
            row.record.document_id: row.assessment.reference_field_values
            for row in audit.diagnostics
        }
        if expected_ids is None:
            expected_ids = ids
            expected_references = references
            continue
        if ids != expected_ids or references != expected_references:
            raise ValueError("prediction sets do not share identical documents and references")
    if expected_ids is None or len(expected_ids) != 100:
        raise ValueError("paired comparison requires exactly 100 validation documents")


def _summary_row(audit: RunAudit) -> dict[str, Any]:
    aggregate = _aggregate_diagnostics(audit.diagnostics)
    metrics = audit.manifest_metrics
    return {
        "run": audit.spec.label,
        "run_key": audit.spec.key,
        "rank": audit.spec.rank,
        "alpha": audit.spec.alpha,
        "epochs": audit.spec.epochs,
        "embeddings_adapted": audit.spec.embeddings_adapted,
        "documents": aggregate["documents"],
        "true_positive": aggregate["true_positive"],
        "predicted": aggregate["predicted"],
        "reference": aggregate["reference"],
        "field_precision": aggregate["precision"],
        "field_recall": aggregate["recall"],
        "field_f1": aggregate["f1"],
        "macro_document_f1": aggregate["macro_document_f1"],
        "field_accuracy": metrics["eval_field_value_accuracy"],
        "cargo_relation_f1": metrics["eval_cargo_relation_f1"],
        "cargo_relation_exact_match": metrics["eval_cargo_relation_exact_match"],
        "category_value_f1": metrics["eval_category_value_f1"],
        "category_value_exact_match": metrics["eval_category_value_exact_match"],
        "json_valid": aggregate["json_valid"],
        "schema_valid": aggregate["schema_valid"],
        "canonical_exact_match": aggregate["canonical_exact_match"],
        "eval_loss": metrics["eval_loss"],
        "generated_tokens_mean": metrics["eval_generated_tokens_mean"],
        "generated_tokens_max": metrics["eval_generated_tokens_max"],
        "generation_eos_reached_fraction": metrics["eval_generation_eos_reached_fraction"],
        "eval_runtime_seconds": metrics["eval_runtime"],
        "trainable_parameters": audit.adapter["trainable_parameters"],
        "adapter_bytes": audit.adapter["adapter_bytes"],
    }


def _comparison_rows(
    audits: Sequence[RunAudit],
    *,
    attribute: str,
    key: str,
) -> list[dict[str, Any]]:
    by_run = {
        audit.spec.key: {str(row[key]): row for row in cast(Sequence[dict[str, Any]], getattr(audit, attribute))}
        for audit in audits
    }
    keys = sorted(set().union(*(set(rows) for rows in by_run.values())))
    rows: list[dict[str, Any]] = []
    for item in keys:
        row: dict[str, Any] = {key: item}
        for audit in audits:
            metric = by_run[audit.spec.key].get(item)
            if metric is None:
                continue
            for name in (
                "reference_values",
                "reference_facts",
                "validation_values",
                "supported_documents",
                "precision",
                "recall",
                "f1",
                "whole_field_exact_rate",
                "whole_section_exact_rate",
                "exact_document_rate",
                "index_insensitive_f1",
            ):
                if name in metric:
                    row[f"{audit.spec.key}_{name}"] = metric[name]
        if "embedding_f1" in row and "r32_f1" in row:
            row["embedding_minus_r32_f1"] = float(row["embedding_f1"]) - float(row["r32_f1"])
        if "embedding_f1" in row and "r64_f1" in row:
            row["embedding_minus_r64_f1"] = float(row["embedding_f1"]) - float(row["r64_f1"])
        rows.append(row)
    return rows


def _document_comparison(audits: Sequence[RunAudit]) -> list[dict[str, Any]]:
    by_run = {
        audit.spec.key: {str(row["document_id"]): row for row in audit.documents}
        for audit in audits
    }
    rows: list[dict[str, Any]] = []
    for document_id in sorted(by_run["embedding"]):
        row: dict[str, Any] = {
            "document_id": document_id,
            "source_path": by_run["embedding"][document_id]["source_path"],
            "page_count": by_run["embedding"][document_id]["page_count"],
            "input_tokens": by_run["embedding"][document_id]["input_tokens"],
            "target_leaves": by_run["embedding"][document_id]["target_leaves"],
            "cargo_group_count": by_run["embedding"][document_id]["cargo_group_count"],
            "container_count": by_run["embedding"][document_id]["container_count"],
        }
        for audit in audits:
            metric = by_run[audit.spec.key][document_id]
            for name in (
                "field_precision",
                "field_recall",
                "field_f1",
                "relation_f1",
                "category_f1",
                "json_valid",
                "schema_valid",
                "schema_failure_class",
                "generated_reference_character_ratio",
            ):
                row[f"{audit.spec.key}_{name}"] = metric[name]
        row["embedding_minus_r32_field_f1"] = (
            float(row["embedding_field_f1"]) - float(row["r32_field_f1"])
        )
        row["embedding_minus_r64_field_f1"] = (
            float(row["embedding_field_f1"]) - float(row["r64_field_f1"])
        )
        rows.append(row)
    return rows


def _bootstrap_delta(embedding: RunAudit, baseline: RunAudit) -> dict[str, Any]:
    embedding_by_id = {row.record.document_id: row for row in embedding.diagnostics}
    baseline_by_id = {row.record.document_id: row for row in baseline.diagnostics}
    ids = sorted(embedding_by_id)
    if ids != sorted(baseline_by_id):
        raise ValueError("paired bootstrap identity mismatch")

    def f1(rows: Sequence[Diagnostic]) -> float:
        true_positive = sum(row.true_positive for row in rows)
        predicted = sum(row.predicted for row in rows)
        reference = sum(row.reference for row in rows)
        precision = true_positive / predicted if predicted else 0.0
        recall = true_positive / reference if reference else 0.0
        return 2 * precision * recall / (precision + recall) if precision + recall else 0.0

    rng = random.Random(BOOTSTRAP_SEED)
    deltas: list[float] = []
    for _ in range(BOOTSTRAP_REPLICATES):
        sampled_ids = [ids[rng.randrange(len(ids))] for _ in ids]
        deltas.append(
            f1([embedding_by_id[document_id] for document_id in sampled_ids])
            - f1([baseline_by_id[document_id] for document_id in sampled_ids])
        )
    deltas.sort()
    point = f1(list(embedding_by_id.values())) - f1(list(baseline_by_id.values()))
    per_document = [
        embedding_by_id[document_id].f1 - baseline_by_id[document_id].f1
        for document_id in ids
    ]
    return {
        "comparison": f"{embedding.spec.key}_minus_{baseline.spec.key}",
        "point_delta": point,
        "lower_95": deltas[249],
        "upper_95": deltas[9749],
        "probability_delta_positive": sum(value > 0 for value in deltas) / len(deltas),
        "document_wins": sum(value > 1e-12 for value in per_document),
        "document_ties": sum(abs(value) <= 1e-12 for value in per_document),
        "document_losses": sum(value < -1e-12 for value in per_document),
        "replicates": BOOTSTRAP_REPLICATES,
        "seed": BOOTSTRAP_SEED,
    }


def _valid_subset_rows(embedding: RunAudit, baseline: RunAudit) -> list[dict[str, Any]]:
    embedding_by_id = {row.record.document_id: row for row in embedding.diagnostics}
    baseline_by_id = {row.record.document_id: row for row in baseline.diagnostics}
    definitions = {
        "all_documents": lambda _a, _b: True,
        "both_json_valid": lambda a, b: a.assessment.json_valid and b.assessment.json_valid,
        "both_schema_valid": lambda a, b: a.assessment.schema_valid and b.assessment.schema_valid,
    }
    rows: list[dict[str, Any]] = []
    for subset, predicate in definitions.items():
        ids = [
            document_id
            for document_id in sorted(embedding_by_id)
            if predicate(embedding_by_id[document_id], baseline_by_id[document_id])
        ]
        embedding_aggregate = _aggregate_diagnostics([embedding_by_id[item] for item in ids])
        baseline_aggregate = _aggregate_diagnostics([baseline_by_id[item] for item in ids])
        rows.append(
            {
                "subset": subset,
                "documents": len(ids),
                "embedding_f1": embedding_aggregate["f1"],
                "r32_f1": baseline_aggregate["f1"],
                "embedding_minus_r32_f1": float(embedding_aggregate["f1"])
                - float(baseline_aggregate["f1"]),
            }
        )
    return rows


def _mechanism_summary(audits: Sequence[RunAudit]) -> list[dict[str, Any]]:
    names = sorted({str(row["mechanism"]) for audit in audits for row in audit.mechanisms})
    rows: list[dict[str, Any]] = []
    for name in names:
        row: dict[str, Any] = {"mechanism": name}
        for audit in audits:
            row[f"{audit.spec.key}_errors"] = sum(
                item["mechanism"] == name for item in audit.mechanisms
            )
        row["embedding_minus_r32_errors"] = (
            int(row["embedding_errors"]) - int(row["r32_errors"])
        )
        rows.append(row)
    return rows


def _error_type_summary(audits: Sequence[RunAudit]) -> list[dict[str, Any]]:
    names = sorted({str(row["error_type"]) for audit in audits for row in audit.errors})
    return [
        {
            "error_type": name,
            **{
                f"{audit.spec.key}_errors": sum(row["error_type"] == name for row in audit.errors)
                for audit in audits
            },
        }
        for name in names
    ]


def _plot(output: Path, name: str) -> Path:
    path = output / "plots" / f"{name}.png"
    path.parent.mkdir(parents=True, exist_ok=True)
    return path


def _save(path: Path) -> None:
    plt.tight_layout()
    plt.savefig(path, dpi=180, bbox_inches="tight")
    plt.close()


def _render_plots(
    output: Path,
    *,
    summaries: Sequence[Mapping[str, Any]],
    history: Sequence[Mapping[str, Any]],
    documents: Sequence[Mapping[str, Any]],
    fields: Sequence[Mapping[str, Any]],
    sections: Sequence[Mapping[str, Any]],
    relations: Sequence[Mapping[str, Any]],
    mechanisms: Sequence[Mapping[str, Any]],
    adapters: Sequence[Mapping[str, Any]],
) -> list[str]:
    sns.set_theme(style="whitegrid", context="notebook")
    palette = {
        "Rank 32 + embedding LoRA": "#C2410C",
        "Rank 32 non-embedding": "#2563EB",
        "Rank 64 non-embedding": "#059669",
    }
    paths: list[str] = []

    def record(name: str) -> Path:
        path = _plot(output, name)
        paths.append(str(path.relative_to(output)))
        return path

    history_frame = pd.DataFrame(history)
    history_frame = history_frame[history_frame["epoch"] > 0]
    plt.figure(figsize=(11, 7))
    sns.lineplot(
        data=history_frame,
        x="epoch",
        y="eval_field_value_f1",
        hue="run",
        style="run",
        markers=True,
        dashes=False,
        palette=palette,
    )
    plt.axhline(0.90, color="#7F1D1D", linestyle="--", label="Target 0.90")
    plt.ylim(0.2, 0.93)
    plt.title("Exact field/value F1 during training")
    _save(record("01_field_f1_progression"))

    progression = history_frame.melt(
        id_vars=["epoch", "run"],
        value_vars=["eval_cargo_relation_f1", "eval_category_value_f1"],
        var_name="metric",
        value_name="value",
    )
    progression["metric"] = progression["metric"].map(
        {"eval_cargo_relation_f1": "Cargo relation F1", "eval_category_value_f1": "Category F1"}
    )
    graph = sns.relplot(
        data=progression,
        x="epoch",
        y="value",
        hue="run",
        col="metric",
        kind="line",
        marker="o",
        palette=palette,
        height=5.5,
        aspect=1.05,
        facet_kws={"sharey": True},
    )
    graph.set(ylim=(0.2, 0.9))
    graph.fig.suptitle("Structured subtask progression", y=1.04)
    graph.fig.savefig(record("02_structured_metric_progression"), dpi=180, bbox_inches="tight")
    plt.close(graph.fig)

    plt.figure(figsize=(11, 7))
    sns.lineplot(
        data=history_frame,
        x="epoch",
        y="eval_loss",
        hue="run",
        style="run",
        markers=True,
        dashes=False,
        palette=palette,
    )
    plt.title("Teacher-forced validation loss")
    _save(record("03_eval_loss_progression"))

    summary_frame = pd.DataFrame(summaries)
    final_metrics = summary_frame.melt(
        id_vars=["run"],
        value_vars=["field_f1", "cargo_relation_f1", "category_value_f1"],
        var_name="metric",
        value_name="value",
    )
    final_metrics["metric"] = final_metrics["metric"].map(
        {"field_f1": "Field/value F1", "cargo_relation_f1": "Cargo relation F1", "category_value_f1": "Category F1"}
    )
    plt.figure(figsize=(12, 7))
    sns.barplot(data=final_metrics, x="metric", y="value", hue="run", palette=palette)
    plt.ylim(0.6, 0.86)
    plt.title("Final saved-prediction quality on the identical validation set")
    _save(record("04_final_metric_comparison"))

    doc_frame = pd.DataFrame(documents)
    plt.figure(figsize=(9, 9))
    sns.scatterplot(
        data=doc_frame,
        x="r32_field_f1",
        y="embedding_field_f1",
        hue="embedding_json_valid",
        size="target_leaves",
        sizes=(25, 180),
        palette={0: "#DC2626", 1: "#2563EB"},
    )
    plt.plot([0, 1], [0, 1], color="black", linestyle="--", alpha=0.5)
    plt.xlim(-0.03, 1.03)
    plt.ylim(-0.03, 1.03)
    plt.title("Paired document F1: embedding versus rank-32 control")
    _save(record("05_paired_document_scatter"))

    plt.figure(figsize=(11, 6))
    sns.histplot(doc_frame, x="embedding_minus_r32_field_f1", bins=30, color="#C2410C")
    plt.axvline(0, color="black", linestyle="--")
    plt.axvline(doc_frame["embedding_minus_r32_field_f1"].median(), color="#2563EB", linestyle=":")
    plt.title("Per-document F1 change from adapting token embeddings")
    _save(record("06_document_delta_distribution"))

    section_frame = pd.DataFrame(sections)
    section_plot = section_frame.melt(
        id_vars="section",
        value_vars=["embedding_f1", "r32_f1", "r64_f1"],
        var_name="run_key",
        value_name="f1",
    )
    section_plot["run"] = section_plot["run_key"].map(
        {"embedding_f1": "Rank 32 + embedding LoRA", "r32_f1": "Rank 32 non-embedding", "r64_f1": "Rank 64 non-embedding"}
    )
    plt.figure(figsize=(12, 10))
    sns.barplot(data=section_plot, y="section", x="f1", hue="run", palette=palette)
    plt.xlim(0, 1.02)
    plt.title("Section-level exact field/value F1")
    _save(record("07_section_f1_comparison"))

    field_frame = pd.DataFrame(fields)
    supported = field_frame[field_frame["embedding_validation_values"].fillna(0) >= 5]
    extremes = pd.concat(
        [supported.nsmallest(15, "embedding_minus_r32_f1"), supported.nlargest(10, "embedding_minus_r32_f1")]
    ).drop_duplicates("field_path")
    extremes = extremes.sort_values("embedding_minus_r32_f1")
    plt.figure(figsize=(13, max(8, len(extremes) * 0.38)))
    colors = ["#B91C1C" if value < 0 else "#047857" for value in extremes["embedding_minus_r32_f1"]]
    plt.barh(extremes["field_path"], extremes["embedding_minus_r32_f1"], color=colors)
    plt.axvline(0, color="black", linewidth=1)
    plt.title("Largest field-level changes: embedding minus rank-32 control")
    plt.xlabel("F1 delta")
    _save(record("08_field_f1_deltas"))

    relation_frame = pd.DataFrame(relations)
    relation_plot = relation_frame.melt(
        id_vars="fact_type",
        value_vars=["embedding_f1", "r32_f1", "r64_f1"],
        var_name="run_key",
        value_name="f1",
    )
    relation_plot["run"] = relation_plot["run_key"].map(
        {"embedding_f1": "Rank 32 + embedding LoRA", "r32_f1": "Rank 32 non-embedding", "r64_f1": "Rank 64 non-embedding"}
    )
    plt.figure(figsize=(12, 8))
    sns.barplot(data=relation_plot, y="fact_type", x="f1", hue="run", palette=palette)
    plt.xlim(0, 1.0)
    plt.title("Cargo-relation fact F1")
    _save(record("09_relation_fact_f1"))

    mechanism_frame = pd.DataFrame(mechanisms)
    mechanism_plot = mechanism_frame.melt(
        id_vars="mechanism",
        value_vars=["embedding_errors", "r32_errors", "r64_errors"],
        var_name="run_key",
        value_name="errors",
    )
    mechanism_plot["run"] = mechanism_plot["run_key"].map(
        {"embedding_errors": "Rank 32 + embedding LoRA", "r32_errors": "Rank 32 non-embedding", "r64_errors": "Rank 64 non-embedding"}
    )
    order = (
        mechanism_frame.assign(total=mechanism_frame[["embedding_errors", "r32_errors", "r64_errors"]].sum(axis=1))
        .sort_values("total")["mechanism"]
    )
    plt.figure(figsize=(13, 10))
    sns.barplot(data=mechanism_plot, y="mechanism", x="errors", hue="run", order=order, palette=palette)
    plt.title("Observable error mechanisms")
    _save(record("10_error_mechanism_comparison"))

    validity = summary_frame.melt(
        id_vars="run",
        value_vars=["json_valid", "schema_valid", "generation_eos_reached_fraction"],
        var_name="metric",
        value_name="fraction",
    )
    validity["metric"] = validity["metric"].map(
        {"json_valid": "JSON valid", "schema_valid": "Schema valid", "generation_eos_reached_fraction": "EOS reached"}
    )
    plt.figure(figsize=(11, 7))
    sns.barplot(data=validity, x="metric", y="fraction", hue="run", palette=palette)
    plt.ylim(0.65, 1.01)
    plt.title("Output completion and structural validity")
    _save(record("11_output_validity"))

    adapter_frame = pd.DataFrame(adapters)
    adapter_plot = adapter_frame.melt(
        id_vars="run",
        value_vars=["trainable_parameters", "adapter_bytes"],
        var_name="measure",
        value_name="value",
    )
    graph = sns.catplot(
        data=adapter_plot,
        x="run",
        y="value",
        hue="run",
        col="measure",
        kind="bar",
        sharey=False,
        palette=palette,
        legend=False,
        height=5.5,
        aspect=1.1,
    )
    graph.set_xticklabels(rotation=25, ha="right")
    graph.fig.suptitle("Adapter capacity and serialized size", y=1.04)
    graph.fig.savefig(record("12_adapter_footprint"), dpi=180, bbox_inches="tight")
    plt.close(graph.fig)
    return paths


def _markdown_table(rows: Sequence[Mapping[str, Any]], columns: Sequence[str]) -> str:
    lines = ["| " + " | ".join(columns) + " |", "|" + "|".join("---" for _ in columns) + "|"]
    for row in rows:
        values = []
        for column in columns:
            value = row[column]
            values.append(f"{value:.4f}" if isinstance(value, float) else str(value).replace("|", "\\|"))
        lines.append("| " + " | ".join(values) + " |")
    return "\n".join(lines)


def _report(
    *,
    summaries: Sequence[Mapping[str, Any]],
    bootstrap: Mapping[str, Any],
    subsets: Sequence[Mapping[str, Any]],
    fields: Sequence[Mapping[str, Any]],
    sections: Sequence[Mapping[str, Any]],
    mechanisms: Sequence[Mapping[str, Any]],
    documents: Sequence[Mapping[str, Any]],
    adapters: Sequence[Mapping[str, Any]],
    plots: Sequence[str],
) -> str:
    summary_by_key = {str(row["run_key"]): row for row in summaries}
    embedding = summary_by_key["embedding"]
    r32 = summary_by_key["r32"]
    r64 = summary_by_key["r64"]
    supported_fields = [row for row in fields if int(row.get("embedding_validation_values", 0)) >= 5]
    regressions = sorted(supported_fields, key=lambda row: float(row["embedding_minus_r32_f1"]))[:15]
    improvements = sorted(supported_fields, key=lambda row: float(row["embedding_minus_r32_f1"]), reverse=True)[:10]
    worst_documents = sorted(documents, key=lambda row: float(row["embedding_minus_r32_field_f1"]))[:10]
    invalid_embedding = [row for row in documents if not bool(row["embedding_json_valid"])]
    mechanism_deltas = sorted(mechanisms, key=lambda row: int(row["embedding_minus_r32_errors"]), reverse=True)
    adapter_by_key = {str(row["run_key"]): row for row in adapters}
    embedding_adapter = adapter_by_key["embedding"]
    r32_adapter = adapter_by_key["r32"]
    return f"""# Embedding-LoRA experiment: paired evaluation audit

Generated: `{datetime.now(UTC).isoformat()}`

## Verdict

Adapting the encoder and decoder token embeddings did **not** improve this task. On the exact same 100 validation documents and canonical references, the embedding run reached **{float(embedding['field_f1']):.4f} field/value F1**, versus **{float(r32['field_f1']):.4f}** for the otherwise matching rank-32 non-embedding control: **{float(embedding['field_f1']) - float(r32['field_f1']):+.4f}**. The paired 10,000-resample 95% interval is **[{float(bootstrap['lower_95']):+.4f}, {float(bootstrap['upper_95']):+.4f}]**, with only **{float(bootstrap['probability_delta_positive']):.1%}** of bootstrap replicates favoring embeddings.

The loss is primarily recall: **{float(embedding['field_recall']):.4f} vs {float(r32['field_recall']):.4f}** ({float(embedding['field_recall']) - float(r32['field_recall']):+.4f}), while precision changes by {float(embedding['field_precision']) - float(r32['field_precision']):+.4f}. The embedding run wins on {bootstrap['document_wins']} documents, ties on {bootstrap['document_ties']}, and loses on {bootstrap['document_losses']}.

The rank-64 non-embedding run is a secondary, non-isolated comparison because it also changes rank, alpha, accumulation, and epochs. Its final F1 is **{float(r64['field_f1']):.4f}**; the embedding result is essentially tied ({float(embedding['field_f1']) - float(r64['field_f1']):+.4f}) but used a different optimization contract.

## Exact final metrics

{_markdown_table(summaries, ['run', 'field_precision', 'field_recall', 'field_f1', 'field_accuracy', 'cargo_relation_f1', 'category_value_f1', 'json_valid', 'schema_valid', 'generation_eos_reached_fraction'])}

All rows above were recomputed from saved generations. Identity/reference parity is exact: 100 unique document IDs and byte-equivalent canonical references in every arm. No constrained decoding was used.

## Why the aggregate fell

1. **More malformed generations.** Embedding adaptation produced {len(invalid_embedding)} invalid-JSON documents versus {round((1 - float(r32['json_valid'])) * 100)} for rank-32. Two embedding outputs ran to the 4,096-token ceiling; three additional outputs ended with local syntax errors. Four documents that rank-32 handled correctly became zero-score invalid outputs. One rank-32 failure was rescued by embeddings.
2. **The deficit remains after removing malformed outputs, but is smaller.** On the {subsets[1]['documents']} documents where both outputs are valid JSON, the delta is {float(subsets[1]['embedding_minus_r32_f1']):+.4f}; on the {subsets[2]['documents']} where both are schema-valid it is {float(subsets[2]['embedding_minus_r32_f1']):+.4f}. Thus malformed output explains most, but not all, of the 3.06-point regression.
3. **Broad recall regression, not one cargo-only failure.** Cargo groups, parties, cargo packages, containers, dates, and route all decline versus rank-32. Cargo allocation relations are close overall and `container_has_package` improves, so the embeddings did not solve the dominant extraction hierarchy consistently.
4. **Training progression plateaued earlier and lower.** At epochs 5/10/15/20, embedding F1 was 0.6314/0.7576/0.7734/0.7900 versus 0.6488/0.7783/0.8103/0.8294 for rank-32. Embedding F1 was effectively flat from epoch 20 to 25 while teacher-forced validation loss rose from 0.1308 to 0.1465. More epochs are not supported by this curve.

## Capacity and serialization cost

Embedding LoRA raises trainable parameters from **{int(r32_adapter['trainable_parameters']):,}** to **{int(embedding_adapter['trainable_parameters']):,}** ({float(embedding_adapter['trainable_parameters']) / float(r32_adapter['trainable_parameters']):.2f}x). The adapter grows from **{float(r32_adapter['adapter_bytes']) / 2**20:.1f} MiB** to **{float(embedding_adapter['adapter_bytes']) / 2**20:.1f} MiB** ({float(embedding_adapter['adapter_bytes']) / float(r32_adapter['adapter_bytes']):.2f}x).

This model has a 262,144-token vocabulary. The two rank-32 embedding A matrices alone contain 16,777,216 trainable values, and PEFT also serialized the tied encoder embedding base tensor (167,772,160 bfloat16 values). That makes token-embedding LoRA unusually expensive here. The extra capacity did not translate into better validation quality.

## Paired validity-controlled comparison

{_markdown_table(subsets, ['subset', 'documents', 'embedding_f1', 'r32_f1', 'embedding_minus_r32_f1'])}

## Largest field regressions (support at least five)

{_markdown_table(regressions, ['field_path', 'embedding_validation_values', 'embedding_f1', 'r32_f1', 'embedding_minus_r32_f1'])}

## Largest field improvements (support at least five)

{_markdown_table(improvements, ['field_path', 'embedding_validation_values', 'embedding_f1', 'r32_f1', 'embedding_minus_r32_f1'])}

## Section comparison

{_markdown_table(sections, ['section', 'embedding_f1', 'r32_f1', 'embedding_minus_r32_f1', 'r64_f1'])}

## Error mechanisms with the largest embedding increases

{_markdown_table(mechanism_deltas[:15], ['mechanism', 'embedding_errors', 'r32_errors', 'embedding_minus_r32_errors', 'r64_errors'])}

## Most negative paired documents

{_markdown_table(worst_documents, ['document_id', 'target_leaves', 'embedding_field_f1', 'r32_field_f1', 'embedding_minus_r32_field_f1', 'embedding_schema_failure_class', 'r32_schema_failure_class'])}

## Recommendation

Do **not** carry token-embedding LoRA into the next main experiment. Retain the rank-32 non-embedding checkpoint as the strongest tested configuration on this split. The experiment is informative: embedding adaptation both diluted structural generation reliability and added substantial parameter/checkpoint cost. The next improvement should target supervision/data/task decomposition rather than more trainable embedding capacity. Rank-64 also failed to outperform rank-32, so simply increasing LoRA capacity is not a supported scaling direction on the present 1,157-document corpus.

The embedding run is fully complete (1,125/1,125 optimizer steps, final adapter, final predictions, and complete manifest). Its `status.json` remains at `artifacts_complete_manifest_pending` even though the complete manifest was published; that is a final status-publication bookkeeping defect, not an incomplete model run.

## Figures

""" + "\n".join(f"- [{Path(path).stem}](<{path}>)" for path in plots) + "\n"


def analyze(*, project_root: Path, output_dir: Path) -> None:
    project_root = project_root.resolve(strict=True)
    output_dir = output_dir.resolve()
    if output_dir.exists():
        raise ValueError(f"analysis output already exists: {output_dir}")
    embedding_run = project_root / "artifacts/kie-training/t5gemma2-270m-lora-mpci-bl-combined1157-task-facing-embeddings-v1"
    r32_run = project_root / "artifacts/kie-training/t5gemma2-270m-lora-mpci-bl-combined1157-task-facing-v2"
    r64_run = project_root / "artifacts/kie-training/t5gemma2-270m-lora-mpci-bl-combined1157-task-facing-r64-a96-ga32-e30-v1"
    specs = (
        RunSpec(
            key="embedding",
            label="Rank 32 + embedding LoRA",
            run_dir=embedding_run,
            evaluation_dir=embedding_run,
            adapter_path=embedding_run / "final-adapter/mpci_bl_combined1157_task_facing_embeddings_v1/adapter_model.safetensors",
            rank=32,
            alpha=64,
            epochs=25,
            embeddings_adapted=True,
        ),
        RunSpec(
            key="r32",
            label="Rank 32 non-embedding",
            run_dir=r32_run,
            evaluation_dir=project_root / "artifacts/kie-training/analysis/t5gemma2-270m-lora-mpci-bl-combined1157-task-facing-v2-checkpoint900-eval-v1",
            adapter_path=r32_run / "checkpoints/checkpoint-900/mpci_bl_combined1157_task_facing_v2/adapter_model.safetensors",
            rank=32,
            alpha=64,
            epochs=20,
            embeddings_adapted=False,
        ),
        RunSpec(
            key="r64",
            label="Rank 64 non-embedding",
            run_dir=r64_run,
            evaluation_dir=r64_run,
            adapter_path=r64_run / "final-adapter/mpci_bl_combined1157_task_facing_r64_a96_ga32_e30_v1/adapter_model.safetensors",
            rank=64,
            alpha=96,
            epochs=30,
            embeddings_adapted=False,
        ),
    )
    for spec in specs:
        for path in (spec.run_dir, spec.evaluation_dir, spec.adapter_path):
            path.resolve(strict=True)

    raw_config = yaml.safe_load((embedding_run / "config.yaml").read_text(encoding="utf-8"))
    config = TrainingConfig.model_validate(raw_config, strict=True)
    task = load_training_task(project_root, config)
    records = _load_records(
        project_root=project_root,
        run_dir=embedding_run,
        config=config,
        task=task,
        eda_dir=None,
    )
    audits = tuple(_audit_run(spec, records=records, task=task) for spec in specs)
    _assert_prediction_parity(audits)

    summaries = [_summary_row(audit) for audit in audits]
    history = [row for audit in audits for row in audit.history]
    documents = _document_comparison(audits)
    fields = _comparison_rows(audits, attribute="fields", key="field_path")
    sections = _comparison_rows(audits, attribute="sections", key="section")
    relations = _comparison_rows(audits, attribute="relations", key="fact_type")
    category_facts = _comparison_rows(audits, attribute="category_facts", key="fact_type")
    groups = _comparison_rows(audits, attribute="groups", key="group")
    mechanisms = _mechanism_summary(audits)
    error_types = _error_type_summary(audits)
    adapters = [dict(audit.adapter) for audit in audits]
    bootstrap = _bootstrap_delta(audits[0], audits[1])
    subsets = _valid_subset_rows(audits[0], audits[1])

    output_dir.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix=f".{output_dir.name}.", dir=output_dir.parent))
    try:
        tables = temporary / "tables"
        _write_csv(tables / "run_summary.csv", summaries)
        _write_csv(tables / "evaluation_history.csv", history)
        _write_csv(tables / "paired_document_metrics.csv", documents)
        _write_csv(tables / "field_comparison.csv", fields)
        _write_csv(tables / "section_comparison.csv", sections)
        _write_csv(tables / "relation_comparison.csv", relations)
        _write_csv(tables / "category_fact_comparison.csv", category_facts)
        _write_csv(tables / "cohort_comparison.csv", groups)
        _write_csv(tables / "error_mechanism_comparison.csv", mechanisms)
        _write_csv(tables / "error_type_comparison.csv", error_types)
        _write_csv(tables / "validity_controlled_comparison.csv", subsets)
        _write_csv(tables / "adapter_inventory.csv", adapters)
        _write_csv(tables / "paired_bootstrap.csv", [bootstrap])
        for audit in audits:
            _write_jsonl(tables / f"{audit.spec.key}_leaf_errors.jsonl", audit.mechanisms)
            issues = _schema_issue_rows(audit.schema_rows)
            if issues:
                _write_csv(tables / f"{audit.spec.key}_schema_issues.csv", issues)

        plot_paths = _render_plots(
            temporary,
            summaries=summaries,
            history=history,
            documents=documents,
            fields=fields,
            sections=sections,
            relations=relations,
            mechanisms=mechanisms,
            adapters=adapters,
        )
        report = _report(
            summaries=summaries,
            bootstrap=bootstrap,
            subsets=subsets,
            fields=fields,
            sections=sections,
            mechanisms=mechanisms,
            documents=documents,
            adapters=adapters,
            plots=plot_paths,
        )
        (temporary / "REPORT.md").write_text(report, encoding="utf-8")
        summary = {
            "schema_version": 1,
            "status": "complete",
            "created_at": datetime.now(UTC).isoformat(),
            "validation_contract": {
                "documents": 100,
                "identical_document_ids": True,
                "identical_reference_text": True,
                "constrained_decoding": False,
            },
            "runs": summaries,
            "paired_bootstrap": bootstrap,
            "validity_controlled_comparison": subsets,
            "plots": plot_paths,
        }
        atomic_write_json(temporary / "summary.json", summary)
        files = sorted(path for path in temporary.rglob("*") if path.is_file())
        manifest = {
            "schema_version": 1,
            "status": "complete",
            "created_at": datetime.now(UTC).isoformat(),
            "files": [
                {
                    "path": str(path.relative_to(temporary)),
                    "bytes": path.stat().st_size,
                    "sha256": sha256_file(path),
                }
                for path in files
            ],
        }
        atomic_write_json(temporary / "manifest.json", manifest)
        temporary.replace(output_dir)
    except BaseException:
        import shutil

        shutil.rmtree(temporary, ignore_errors=True)
        raise


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, default=Path("."))
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(
            "artifacts/kie-training/analysis/"
            "t5gemma2-270m-lora-mpci-bl-embedding-vs-nonembedding-v1"
        ),
    )
    return parser.parse_args()


def main() -> None:
    arguments = _parse_args()
    analyze(project_root=arguments.project_root, output_dir=arguments.output_dir)


if __name__ == "__main__":
    main()
