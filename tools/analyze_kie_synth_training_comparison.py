#!/usr/bin/env python3
"""Audit the 100-ID synthetic-augmentation training result against real-only runs.

This is an artifact-only analysis. It never loads model weights or mutates a run.
"""

from __future__ import annotations

import argparse
import csv
import json
import random
import re
from collections import Counter, defaultdict
from collections.abc import Hashable
from pathlib import Path
from types import SimpleNamespace
from typing import Any, cast

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pydantic import ValidationError

from document_ocr.training.config import load_training_config
from document_ocr.training.metrics import (
    _relation_explicit_facts,
    assess_prediction,
    structured_metrics,
)
from document_ocr.training.tasks import load_training_task
from tools.analyze_kie_error_mechanisms import _cross_field_pairs, classify_leaf_error
from tools.analyze_kie_task_facing_checkpoint import _classify_errors


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
    if not rows or any(not isinstance(row, dict) for row in rows):
        raise ValueError(f"empty or malformed JSONL: {path}")
    return rows


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    fields = sorted({key for row in rows for key in row})
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def norm(path: str) -> str:
    return re.sub(r"\[\d+\]", "[]", path)


def section(path: str) -> str:
    return path.removeprefix("$.documentPatch.").split(".")[0].split("[")[0]


def leaves(obj: Any, path: str = "$.documentPatch") -> set[tuple[str, str]]:
    if isinstance(obj, dict):
        return {leaf for key, value in obj.items() for leaf in leaves(value, f"{path}.{key}")}
    if isinstance(obj, list):
        return {
            leaf for index, value in enumerate(obj) for leaf in leaves(value, f"{path}[{index}]")
        }
    return {(path, json.dumps(obj, ensure_ascii=False, sort_keys=True, separators=(",", ":")))}


def score[Fact: Hashable](
    predicted: list[set[Fact]], reference: list[set[Fact]]
) -> dict[str, float]:
    tp = sum(len(p & r) for p, r in zip(predicted, reference, strict=True))
    p = sum(map(len, predicted))
    r = sum(map(len, reference))
    precision = tp / p if p else 0.0
    recall = tp / r if r else 0.0
    return {
        "tp": tp,
        "predicted": p,
        "reference": r,
        "precision": precision,
        "recall": recall,
        "f1": 2 * precision * recall / (precision + recall) if precision + recall else 0.0,
    }


def paired_bootstrap[Fact: Hashable](
    first: list[set[Fact]],
    second: list[set[Fact]],
    reference: list[set[Fact]],
    *,
    replicates: int = 3000,
) -> dict[str, float]:
    if len(first) != len(second) or len(first) != len(reference):
        raise ValueError("paired bootstrap length mismatch")
    rng = random.Random(412)
    deltas = []
    for _ in range(replicates):
        indices = [rng.randrange(len(reference)) for _ in reference]
        lhs = score([first[i] for i in indices], [reference[i] for i in indices])["f1"]
        rhs = score([second[i] for i in indices], [reference[i] for i in indices])["f1"]
        deltas.append(lhs - rhs)
    deltas.sort()
    return {
        "delta_f1": score(first, reference)["f1"] - score(second, reference)["f1"],
        "bootstrap_low_95": deltas[int(0.025 * replicates)],
        "bootstrap_high_95": deltas[int(0.975 * replicates)],
    }


def schema_issue(text: str, task: Any) -> str:
    try:
        value = json.loads(text)
    except json.JSONDecodeError as exc:
        return f"invalid_json: {exc.msg}"
    if not isinstance(value, dict):
        return f"json_non_object: {type(value).__name__}"
    try:
        task.canonicalize(value)
    except ValidationError as exc:
        return "; ".join(
            sorted({f"{'.'.join(map(str, item['loc']))}: {item['type']}" for item in exc.errors()})
        )
    except ValueError as exc:
        return str(exc)
    return ""


def load_predictions(
    project: Path, run: Path, prediction_dir: Path
) -> tuple[Any, dict[str, dict[str, Any]], dict[str, Any]]:
    config = load_training_config(run / "config.yaml")
    task = load_training_task(project, config)
    rows = read_jsonl(prediction_dir / "predictions" / "validation.jsonl")
    by_id = {str(row["document_id"]): row for row in rows}
    if len(rows) != 100 or len(by_id) != 100:
        raise ValueError(f"expected 100 unique validation predictions: {prediction_dir}")
    generated = [str(row["generated_text"]) for row in rows]
    reference = [str(row["reference_text"]) for row in rows]
    metrics, assessments = structured_metrics(generated, reference, task)
    metrics_path = prediction_dir / "metrics.json"
    if not metrics_path.is_file():
        metrics_path = run / "checkpoints" / "eval_results.json"
    recorded_metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
    for key in (
        "field_value_f1",
        "field_value_precision",
        "field_value_recall",
        "json_valid",
        "schema_valid",
    ):
        if abs(metrics[key] - recorded_metrics[f"eval_{key}"]) > 1e-12:
            raise ValueError(f"saved evaluation metric differs from replay: {prediction_dir}:{key}")
    for row, assessment in zip(rows, assessments, strict=True):
        for key in ("json_valid", "schema_valid", "canonical_exact_match"):
            if bool(row[key]) != getattr(assessment, key):
                raise ValueError(f"persisted flag mismatch: {row['document_id']}:{key}")
    return task, by_id, metrics


def raw_prediction_leaves(row: dict[str, Any]) -> set[tuple[str, str]]:
    try:
        value = json.loads(row["generated_text"])
    except json.JSONDecodeError:
        return set()
    if not isinstance(value, dict) or not isinstance(value.get("documentPatch"), dict):
        return set()
    return leaves(value["documentPatch"])


def cargo_relations(row: dict[str, Any], key: str) -> set[tuple[str, ...]]:
    try:
        value = json.loads(row[key])
    except json.JSONDecodeError:
        return set()
    if not isinstance(value, dict) or not isinstance(value.get("documentPatch"), dict):
        return set()
    relations, _ = _relation_explicit_facts(value["documentPatch"])
    return set(relations)


def common_fields(
    runs: dict[str, dict[str, dict[str, Any]]], current: str
) -> tuple[set[str], dict[str, Any]]:
    names = list(runs)
    ids = set(runs[current])
    if any(set(rows) != ids for rows in runs.values()):
        raise ValueError("runs do not cover identical validation document IDs")
    ref_sets = {
        name: {doc: leaves(json.loads(rows[doc]["reference_text"])["documentPatch"]) for doc in ids}
        for name, rows in runs.items()
    }
    families = [{norm(path) for doc in ids for path, _ in docs[doc]} for docs in ref_sets.values()]
    candidates = set.intersection(*families)
    stable = {
        family
        for family in candidates
        if all(
            {item for item in ref_sets[current][doc] if norm(item[0]) == family}
            == {item for item in ref_sets[name][doc] if norm(item[0]) == family}
            for name in names
            for doc in ids
        )
    }
    return stable, {
        "candidate_families": len(candidates),
        "stable_families": len(stable),
        "excluded_families": sorted(candidates - stable),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project-root", type=Path, required=True)
    parser.add_argument("--current-run", type=Path, required=True)
    parser.add_argument("--best-eval", type=Path, required=True)
    parser.add_argument(
        "--baseline", action="append", nargs=2, metavar=("NAME", "RUN"), required=True
    )
    parser.add_argument(
        "--baseline-eval", action="append", nargs=3, metavar=("NAME", "RUN", "EVAL")
    )
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    project = args.project_root.resolve()
    current_run = args.current_run.resolve()
    output = args.output.resolve()
    if output.exists() and any(output.iterdir()):
        raise ValueError(f"refusing to overwrite a nonempty analysis directory: {output}")
    names = {"best": (current_run, args.best_eval.resolve()), "last": (current_run, current_run)}
    for name, run in args.baseline:
        if name in names:
            raise ValueError(f"duplicate run name: {name}")
        names[name] = (Path(run).resolve(), Path(run).resolve())
    for name, run, evaluation in args.baseline_eval or []:
        if name in names:
            raise ValueError(f"duplicate run name: {name}")
        names[name] = (Path(run).resolve(), Path(evaluation).resolve())
    loaded = {name: load_predictions(project, *paths) for name, paths in names.items()}
    runs = {name: item[1] for name, item in loaded.items()}
    stable, drift = common_fields(runs, "best")
    ids = sorted(runs["best"])
    current_reference = {
        doc: leaves(json.loads(runs["best"][doc]["reference_text"])["documentPatch"]) for doc in ids
    }
    shared_ref = [
        {item for item in current_reference[doc] if norm(item[0]) in stable} for doc in ids
    ]
    relation_ref = [cargo_relations(runs["best"][doc], "reference_text") for doc in ids]
    if any(
        cargo_relations(rows[doc], "reference_text") != relation_ref[index]
        for rows in runs.values()
        for index, doc in enumerate(ids)
    ):
        raise ValueError("cargo relation references differ between the paired runs")
    comparison = []
    relation_type_counts: dict[tuple[str, str], Counter[str]] = defaultdict(Counter)
    common_field_counts: dict[tuple[str, str], Counter[str]] = defaultdict(Counter)
    common_predictions: dict[str, list[set[tuple[str, str]]]] = {}
    relation_predictions: dict[str, list[set[tuple[str, ...]]]] = {}
    for name, (_, rows, native_metrics) in loaded.items():
        common_pred = [
            {item for item in raw_prediction_leaves(rows[doc]) if norm(item[0]) in stable}
            for doc in ids
        ]
        common_predictions[name] = common_pred
        common = score(common_pred, shared_ref)
        for predicted_leaves, reference_leaves in zip(common_pred, shared_ref, strict=True):
            for field_facts, key in (
                (predicted_leaves, "predicted"),
                (reference_leaves, "reference"),
                (predicted_leaves & reference_leaves, "tp"),
            ):
                for path, _ in field_facts:
                    common_field_counts[(name, norm(path))][key] += 1
        relation_pred = [cargo_relations(rows[doc], "generated_text") for doc in ids]
        relation_predictions[name] = relation_pred
        relations = score(relation_pred, relation_ref)
        for predicted_facts, reference_facts in zip(relation_pred, relation_ref, strict=True):
            for relation_facts, key in (
                (predicted_facts, "predicted"),
                (reference_facts, "reference"),
                (predicted_facts & reference_facts, "tp"),
            ):
                for fact in relation_facts:
                    relation_type_counts[(name, fact[0])][key] += 1
        comparison.append(
            {
                "run": name,
                "native_f1": native_metrics["field_value_f1"],
                "native_json_valid": native_metrics["json_valid"],
                "native_schema_valid": native_metrics["schema_valid"],
                **{f"common_{k}": v for k, v in common.items()},
                **{f"relation_{k}": v for k, v in relations.items()},
            }
        )
    paired_rows = []
    for name in names:
        if name == "best":
            continue
        best_doc = [
            score([pred], [ref])["f1"]
            for pred, ref in zip(common_predictions["best"], shared_ref, strict=True)
        ]
        other_doc = [
            score([pred], [ref])["f1"]
            for pred, ref in zip(common_predictions[name], shared_ref, strict=True)
        ]
        paired_rows.append(
            {
                "comparison": f"best_minus_{name}",
                "documents_better": sum(a > b for a, b in zip(best_doc, other_doc, strict=True)),
                "documents_worse": sum(a < b for a, b in zip(best_doc, other_doc, strict=True)),
                "documents_tied": sum(a == b for a, b in zip(best_doc, other_doc, strict=True)),
                **paired_bootstrap(
                    common_predictions["best"], common_predictions[name], shared_ref
                ),
                **{
                    f"relation_{key}": value
                    for key, value in paired_bootstrap(
                        relation_predictions["best"], relation_predictions[name], relation_ref
                    ).items()
                },
            }
        )
    task = loaded["best"][0]
    source_rows = read_jsonl(
        project
        / "artifacts/kie-training/datasets/mpci-bl-real1157-v5-shared-validation100-v1"
        / "validation.jsonl"
    )
    source = {str(row["documentId"]): row for row in source_rows}
    if set(source) != set(ids):
        raise ValueError("validation source identity set differs from predictions")
    v3_source = {
        str(row["documentId"]): row
        for row in read_jsonl(
            project
            / "artifacts/kie-training/datasets"
            / "mpci-bl-combined1157-task-facing-package-categories-v2"
            / "records.jsonl"
        )
    }
    for doc in ids:
        if source[doc]["target"] != json.loads(runs["best"][doc]["reference_text"]):
            raise ValueError(f"validation reference differs from source: {doc}")
        if doc not in v3_source or v3_source[doc]["joinedRawText"] != source[doc]["joinedRawText"]:
            raise ValueError(f"real-only and synthetic-run OCR text differ: {doc}")
    assessments = {
        doc: assess_prediction(
            runs["best"][doc]["generated_text"], runs["best"][doc]["reference_text"], task
        )
        for doc in ids
    }
    fields: dict[str, Counter[str]] = defaultdict(Counter)
    sections: dict[str, Counter[str]] = defaultdict(Counter)
    document_rows: list[dict[str, Any]] = []
    schema_rows: list[dict[str, Any]] = []
    leaf_rows: list[dict[str, Any]] = []
    for doc in ids:
        assessment = assessments[doc]
        predicted_leaves = assessment.predicted_field_values
        reference_leaves = assessment.reference_field_values
        true_positive_leaves = predicted_leaves & reference_leaves
        for field_facts, key in (
            (predicted_leaves, "predicted"),
            (reference_leaves, "reference"),
            (true_positive_leaves, "tp"),
        ):
            for path, _ in field_facts:
                fields[norm(path)][key] += 1
                sections[section(path)][key] += 1
        ref_by_field: dict[str, set[tuple[str, str]]] = defaultdict(set)
        pred_by_field: dict[str, set[tuple[str, str]]] = defaultdict(set)
        for item in reference_leaves:
            ref_by_field[norm(item[0])].add(item)
        for item in predicted_leaves:
            pred_by_field[norm(item[0])].add(item)
        for field, values in ref_by_field.items():
            fields[field]["supported_docs"] += 1
            fields[field]["exact_docs"] += pred_by_field[field] == values
            reference_values = Counter(value for _, value in values)
            predicted_values = Counter(value for _, value in pred_by_field[field])
            fields[field]["unordered_tp"] += sum((reference_values & predicted_values).values())
            fields[field]["unordered_exact_docs"] += predicted_values == reference_values
        issue = schema_issue(runs["best"][doc]["generated_text"], task)
        if bool(issue) == assessment.schema_valid:
            raise ValueError(f"schema diagnosis mismatch: {doc}")
        if issue:
            schema_rows.append({"document_id": doc, "issue": issue})
        raw = source[doc]["joinedRawText"]
        diagnostic = SimpleNamespace(
            record=SimpleNamespace(document_id=doc, raw_text=raw), assessment=assessment
        )
        leaf_rows.extend(_classify_errors(cast(Any, diagnostic)))
        doc_score = score([set(predicted_leaves)], [set(reference_leaves)])
        patch = source[doc]["target"]["documentPatch"]
        document_rows.append(
            {
                "document_id": doc,
                "f1": doc_score["f1"],
                "tp": doc_score["tp"],
                "predicted": doc_score["predicted"],
                "reference": doc_score["reference"],
                "json_valid": assessment.json_valid,
                "schema_valid": assessment.schema_valid,
                "source_characters": len(raw),
                "target_leaves": len(reference_leaves),
                "containers": len(patch.get("containers", [])),
                "cargo_groups": len(patch.get("cargoGroups", [])),
                "cargo_packages": len(patch.get("cargoPackages", [])),
                "dangerous_goods": any("dangerousGoods" in path for path, _ in reference_leaves),
                "temperature_setpoint": any(
                    "temperatureSetpoint" in path for path, _ in reference_leaves
                ),
            }
        )
    cross = _cross_field_pairs(leaf_rows)
    for index, row in enumerate(leaf_rows):
        mechanism, basis, linked = classify_leaf_error(row, cross_field_paths=cross.get(index, ()))
        if (
            row["field_path"].endswith(".sizeCategory")
            and mechanism == "unsupported_text_candidate"
        ):
            mechanism = "categorical_semantic_error"
            basis = "normalized container-size category is not a literal OCR span"
        if (
            mechanism == "unsupported_or_wrong_numeric_value"
            and row["predicted_ocr_grounded"] is None
        ):
            mechanism = "numeric_grounding_unresolved"
            basis = "the literal grounding probe is inapplicable to this short or missing number"
        row.update(mechanism=mechanism, basis=basis, linked_paths=";".join(linked))

    def metric_rows(values: dict[str, Counter[str]], key: str) -> list[dict[str, Any]]:
        result = []
        for name, count in values.items():
            precision = count["tp"] / count["predicted"] if count["predicted"] else 0.0
            recall = count["tp"] / count["reference"] if count["reference"] else 0.0
            result.append(
                {
                    key: name,
                    "reference": count["reference"],
                    "predicted": count["predicted"],
                    "true_positive": count["tp"],
                    "false_negative": count["reference"] - count["tp"],
                    "false_positive": count["predicted"] - count["tp"],
                    "precision": precision,
                    "recall": recall,
                    "f1": 2 * precision * recall / (precision + recall)
                    if precision + recall
                    else 0.0,
                    "supported_documents": count["supported_docs"],
                    "whole_field_exact_rate": count["exact_docs"] / count["supported_docs"]
                    if count["supported_docs"]
                    else None,
                    "unordered_true_positive": count["unordered_tp"],
                    "unordered_whole_field_exact_rate": count["unordered_exact_docs"]
                    / count["supported_docs"]
                    if count["supported_docs"]
                    else None,
                }
            )
        return sorted(result, key=lambda row: (-row["reference"], row[key]))

    field_rows = metric_rows(fields, "field_path")
    section_rows = metric_rows(sections, "section")
    replay = score(
        [set(assessments[doc].predicted_field_values) for doc in ids],
        [set(assessments[doc].reference_field_values) for doc in ids],
    )
    if abs(replay["f1"] - loaded["best"][2]["field_value_f1"]) > 1e-12:
        raise ValueError("best-checkpoint metric replay differs from evaluation")
    output.mkdir(parents=True, exist_ok=True)
    write_csv(output / "run_comparison.csv", comparison)
    common_field_rows = []
    for (name, field), count in sorted(common_field_counts.items()):
        precision = count["tp"] / count["predicted"] if count["predicted"] else 0.0
        recall = count["tp"] / count["reference"] if count["reference"] else 0.0
        common_field_rows.append(
            {
                "run": name,
                "field_path": field,
                "reference": count["reference"],
                "predicted": count["predicted"],
                "true_positive": count["tp"],
                "precision": precision,
                "recall": recall,
                "f1": 2 * precision * recall / (precision + recall) if precision + recall else 0.0,
            }
        )
    write_csv(output / "common_fields_by_run.csv", common_field_rows)
    relation_type_rows = []
    for (name, fact_type), count in sorted(relation_type_counts.items()):
        precision = count["tp"] / count["predicted"] if count["predicted"] else 0.0
        recall = count["tp"] / count["reference"] if count["reference"] else 0.0
        relation_type_rows.append(
            {
                "run": name,
                "relation_type": fact_type,
                "reference": count["reference"],
                "predicted": count["predicted"],
                "true_positive": count["tp"],
                "precision": precision,
                "recall": recall,
                "f1": 2 * precision * recall / (precision + recall) if precision + recall else 0.0,
            }
        )
    write_csv(output / "relation_types.csv", relation_type_rows)
    write_csv(output / "paired_comparison.csv", paired_rows)
    write_csv(output / "fields.csv", field_rows)
    write_csv(output / "sections.csv", section_rows)
    write_csv(output / "documents.csv", document_rows)
    write_csv(output / "schema_issues.csv", schema_rows)
    write_csv(output / "leaf_errors.csv", leaf_rows)
    history = json.loads(
        (current_run / "checkpoints/trainer_state.json").read_text(encoding="utf-8")
    )
    progression = [
        {
            "step": row["step"],
            "epoch": row["epoch"],
            "f1": row["eval_field_value_f1"],
            "schema_valid": row["eval_schema_valid"],
            "json_valid": row["eval_json_valid"],
            "eval_loss": row["eval_loss"],
        }
        for row in history["log_history"]
        if "eval_field_value_f1" in row
    ]
    if len(progression) != 10 or max(progression, key=lambda row: row["f1"])["step"] != 1211:
        raise ValueError("unexpected training progression or best step")
    write_csv(output / "training_progression.csv", progression)
    slice_rows = []
    source_midpoint = sorted(row["source_characters"] for row in document_rows)[len(ids) // 2]
    leaf_midpoint = sorted(row["target_leaves"] for row in document_rows)[len(ids) // 2]
    for label, subset in (
        ("dangerous_goods", [row for row in document_rows if row["dangerous_goods"]]),
        ("no_dangerous_goods", [row for row in document_rows if not row["dangerous_goods"]]),
        ("temperature_setpoint", [row for row in document_rows if row["temperature_setpoint"]]),
        (
            "no_temperature_setpoint",
            [row for row in document_rows if not row["temperature_setpoint"]],
        ),
        ("schema_valid", [row for row in document_rows if row["schema_valid"]]),
        ("schema_invalid", [row for row in document_rows if not row["schema_valid"]]),
        (
            "source_shorter_half",
            [row for row in document_rows if row["source_characters"] < source_midpoint],
        ),
        (
            "source_longer_half",
            [row for row in document_rows if row["source_characters"] >= source_midpoint],
        ),
        (
            "target_fewer_leaves",
            [row for row in document_rows if row["target_leaves"] < leaf_midpoint],
        ),
        (
            "target_more_leaves",
            [row for row in document_rows if row["target_leaves"] >= leaf_midpoint],
        ),
        ("single_container", [row for row in document_rows if row["containers"] == 1]),
        ("multiple_containers", [row for row in document_rows if row["containers"] > 1]),
    ):
        subset_ids = [row["document_id"] for row in subset]
        slice_score = score(
            [set(assessments[doc].predicted_field_values) for doc in subset_ids],
            [set(assessments[doc].reference_field_values) for doc in subset_ids],
        )
        slice_rows.append({"slice": label, "documents": len(subset_ids), **slice_score})
    write_csv(output / "slices.csv", slice_rows)
    summary = {
        "validation_documents": len(ids),
        "shared_reference": drift,
        "common_reference_leaves": sum(map(len, shared_ref)),
        "best_metrics": loaded["best"][2],
        "last_metrics": loaded["last"][2],
        "schema_issues": Counter(row["issue"] for row in schema_rows),
        "mechanisms": Counter(row["mechanism"] for row in leaf_rows),
        "error_types": Counter(row["error_type"] for row in leaf_rows),
        "paired_comparison": paired_rows,
        "slices": slice_rows,
    }
    (output / "summary.json").write_text(
        json.dumps(summary, indent=2, default=dict) + "\n", encoding="utf-8"
    )
    fig, ax = plt.subplots(figsize=(10, 5))
    ordered = sorted(section_rows, key=lambda row: row["reference"], reverse=True)
    ax.barh([row["section"] for row in ordered][::-1], [row["f1"] for row in ordered][::-1])
    ax.set(
        xlim=(0, 1),
        xlabel="Exact field-value F1",
        title="Best checkpoint: validation F1 by section",
    )
    fig.tight_layout()
    fig.savefig(output / "section_f1.png", dpi=160)
    plt.close(fig)
    supported_fields = sorted(
        (row for row in field_rows if row["reference"] >= 30), key=lambda row: row["f1"]
    )[:20]
    fig, ax = plt.subplots(figsize=(12, 8))
    ax.barh(
        [row["field_path"].removeprefix("$.documentPatch.") for row in supported_fields][::-1],
        [row["f1"] for row in supported_fields][::-1],
    )
    ax.set(
        xlim=(0, 1),
        xlabel="Exact field-value F1",
        title="Lowest-scoring supported fields (at least 30 references)",
    )
    fig.tight_layout()
    fig.savefig(output / "lowest_supported_fields.png", dpi=160)
    plt.close(fig)
    mechanisms = Counter(row["mechanism"] for row in leaf_rows)
    fig, ax = plt.subplots(figsize=(11, 7))
    ordered_mechanisms = mechanisms.most_common()
    ax.barh(
        [key for key, _ in ordered_mechanisms][::-1],
        [count for _, count in ordered_mechanisms][::-1],
    )
    ax.set(xlabel="Leaf error rows", title="Best checkpoint: observable error mechanisms")
    fig.tight_layout()
    fig.savefig(output / "error_mechanisms.png", dpi=160)
    plt.close(fig)
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.plot(
        [row["step"] for row in progression],
        [row["f1"] for row in progression],
        marker="o",
        label="Field-value F1",
    )
    ax.plot(
        [row["step"] for row in progression],
        [row["schema_valid"] for row in progression],
        marker="s",
        label="Schema-valid fraction",
    )
    ax.axvline(1211, color="black", linestyle="--", alpha=0.5, label="Selected best step")
    ax.set(
        xlabel="Optimizer step",
        ylabel="Validation metric",
        ylim=(0, 1),
        title="Training progression: 100 real validation documents",
    )
    ax.legend()
    fig.tight_layout()
    fig.savefig(output / "training_progression.png", dpi=160)
    plt.close(fig)
    print(
        json.dumps(
            {"output": str(output), "comparison": comparison, "summary": summary},
            indent=2,
            default=dict,
        )
    )


if __name__ == "__main__":
    main()
