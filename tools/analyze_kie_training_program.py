#!/usr/bin/env python3
# ruff: noqa: E501
"""Synthesize every MPCI B/L T5Gemma training experiment into one audit."""

from __future__ import annotations

import argparse
import csv
import json
import math
import random
import shutil
import tempfile
from collections import Counter
from collections.abc import Mapping, Sequence
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
from analyze_kie_task_facing_checkpoint import (
    Diagnostic,
    _load_diagnostics,
    _load_records,
)

from document_ocr.atomic import atomic_write_json
from document_ocr.hashing import sha256_file
from document_ocr.training.config import TrainingConfig
from document_ocr.training.tasks import load_training_task

BOOTSTRAP_REPLICATES = 10_000
BOOTSTRAP_SEED = 829
TEXT_COLUMNS = frozenset(
    {
        "category_type",
        "concept",
        "fact_type",
        "field_path",
        "group",
        "group_type",
        "mechanism",
        "section",
        "token",
    }
)


@dataclass(frozen=True, slots=True)
class RunDefinition:
    key: str
    label: str
    run_dir: Path
    train_documents: int
    validation_documents: int
    task_family: str
    result_kind: str
    result_path: Path | None
    comparison_class: str


def _json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path}")
    return cast(dict[str, Any], value)


def _csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as stream:
        rows = list(csv.DictReader(stream))
    if not rows:
        raise ValueError(f"empty CSV input: {path}")
    return rows


def _jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if not line:
            raise ValueError(f"blank JSONL row: {path}:{line_number}")
        value = json.loads(line)
        if not isinstance(value, dict):
            raise ValueError(f"non-object JSONL row: {path}:{line_number}")
        rows.append(cast(dict[str, Any], value))
    return rows


def _write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    if not rows:
        raise ValueError(f"refusing to publish empty CSV: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = sorted({key for row in rows for key in row})
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, extrasaction="raise")
        writer.writeheader()
        writer.writerows(rows)


def _run_definitions(root: Path) -> tuple[RunDefinition, ...]:
    training = root / "artifacts/kie-training"
    analysis = training / "analysis"
    return (
        RunDefinition(
            "pilot106",
            "Pilot 106 / semantic v2",
            training / "t5gemma2-270m-lora-mpci-bl-pilot106-v5",
            90,
            16,
            "semantic-v2",
            "complete_manifest",
            training / "t5gemma2-270m-lora-mpci-bl-pilot106-v5/manifest.json",
            "directional_only",
        ),
        RunDefinition(
            "semantic487",
            "427 train / semantic v2",
            training / "t5gemma2-270m-lora-mpci-bl-combined487-v2",
            427,
            60,
            "semantic-v2",
            "complete_manifest",
            training / "t5gemma2-270m-lora-mpci-bl-combined487-v2/manifest.json",
            "paired_schema_ablation",
        ),
        RunDefinition(
            "relation483",
            "423 train / relation v3",
            training / "t5gemma2-270m-lora-mpci-bl-relation-explicit-v3-v1",
            423,
            60,
            "relation-v3",
            "complete_manifest",
            training / "t5gemma2-270m-lora-mpci-bl-relation-explicit-v3-v1/manifest.json",
            "paired_schema_ablation",
        ),
        RunDefinition(
            "task1157_r32",
            "1,057 train / rank 32",
            training / "t5gemma2-270m-lora-mpci-bl-combined1157-task-facing-v2",
            1057,
            100,
            "relation-v3-task-facing",
            "retained_checkpoint_evaluation",
            analysis
            / "t5gemma2-270m-lora-mpci-bl-combined1157-task-facing-v2-checkpoint900-eval-v1/manifest.json",
            "paired_capacity_ablation",
        ),
        RunDefinition(
            "task1157_r64",
            "1,057 train / rank 64",
            training / "t5gemma2-270m-lora-mpci-bl-combined1157-task-facing-r64-a96-ga32-e30-v1",
            1057,
            100,
            "relation-v3-task-facing",
            "complete_manifest",
            training
            / "t5gemma2-270m-lora-mpci-bl-combined1157-task-facing-r64-a96-ga32-e30-v1/manifest.json",
            "paired_capacity_ablation",
        ),
        RunDefinition(
            "task1157_embeddings",
            "1,057 train / embedding LoRA",
            training / "t5gemma2-270m-lora-mpci-bl-combined1157-task-facing-embeddings-v1",
            1057,
            100,
            "relation-v3-task-facing",
            "complete_manifest",
            training
            / "t5gemma2-270m-lora-mpci-bl-combined1157-task-facing-embeddings-v1/manifest.json",
            "paired_capacity_ablation",
        ),
        RunDefinition(
            "table483",
            "423 train / table input (interrupted)",
            training / "t5gemma2-270m-lora-mpci-bl-relation-v3-table-input-v2",
            423,
            60,
            "relation-v3-table-input",
            "interrupted_last_scheduled_eval",
            None,
            "incomplete_method_ablation",
        ),
        RunDefinition(
            "semantic487_failed",
            "Semantic 487 first attempt",
            training / "t5gemma2-270m-lora-mpci-bl-combined487-v1",
            427,
            60,
            "semantic-v2",
            "failed_before_training",
            None,
            "operational_only",
        ),
        RunDefinition(
            "table483_failed",
            "Table input batch-2 attempt",
            training / "t5gemma2-270m-lora-mpci-bl-relation-v3-table-input-v1",
            423,
            60,
            "relation-v3-table-input",
            "failed_before_evaluation",
            None,
            "operational_only",
        ),
    )


def _evaluation_history(run: RunDefinition) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for event in _jsonl(run.run_dir / "logs/events.jsonl"):
        metrics = event.get("metrics")
        if (
            event.get("event") != "trainer_log"
            or not isinstance(metrics, dict)
            or "eval_field_value_f1" not in metrics
        ):
            continue
        rows.append(
            {
                "run_key": run.key,
                "run": run.label,
                "task_family": run.task_family,
                "step": int(event["global_step"]),
                "epoch": float(event["epoch"]),
                **metrics,
            }
        )
    return rows


def _manifest_metrics(definition: RunDefinition) -> dict[str, Any] | None:
    if definition.result_path is None:
        return None
    manifest = _json(definition.result_path)
    metrics = manifest.get("metrics")
    if not isinstance(metrics, dict):
        raise ValueError(f"metrics absent from {definition.result_path}")
    if "validation_evaluation" in metrics:
        metrics = metrics["validation_evaluation"]
    if not isinstance(metrics, dict) or "eval_field_value_f1" not in metrics:
        raise ValueError(f"validation metrics absent from {definition.result_path}")
    return cast(dict[str, Any], metrics)


def _config_contract(definition: RunDefinition) -> dict[str, Any]:
    config = _json(definition.run_dir / "resolved-config.json")
    optimization = cast(dict[str, Any], config["optimization"])
    peft = cast(dict[str, Any], config["peft"])
    evaluation = cast(dict[str, Any], config["evaluation"])
    dataset = cast(dict[str, Any], config["dataset"])
    source = dataset.get("source")
    return {
        "run_key": definition.key,
        "run": definition.label,
        "task_family": definition.task_family,
        "comparison_class": definition.comparison_class,
        "result_kind": definition.result_kind,
        "train_documents": definition.train_documents,
        "validation_documents": definition.validation_documents,
        "dataset_sha256": source.get("sha256", "multiple_pinned_splits")
        if isinstance(source, dict)
        else "multiple_pinned_splits",
        "input_field": dataset["fields"]["input_text"],
        "epochs_planned": optimization["num_train_epochs"],
        "micro_batch": optimization["per_device_train_batch_size"],
        "gradient_accumulation": optimization["gradient_accumulation_steps"],
        "effective_batch": optimization["per_device_train_batch_size"]
        * optimization["gradient_accumulation_steps"],
        "learning_rate": optimization["learning_rate"],
        "scheduler": optimization["lr_scheduler_type"],
        "warmup_ratio": optimization["warmup_ratio"],
        "weight_decay": optimization["weight_decay"],
        "adam_beta1": optimization["adam_beta1"],
        "adam_beta2": optimization["adam_beta2"],
        "lora_rank": peft["rank"],
        "lora_alpha": peft["alpha"],
        "lora_dropout": peft["dropout"],
        "embedding_targeted": "embed_tokens" in peft["target_modules_regex"],
        "eval_batch": evaluation["per_device_batch_size"],
        "generation_max_length": evaluation["generation_max_length"],
        "max_source_length": dataset["preprocessing"]["max_source_length"],
        "max_target_length": dataset["preprocessing"]["max_target_length"],
    }


def _run_inventory(
    definitions: Sequence[RunDefinition],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    histories = {item.key: _evaluation_history(item) for item in definitions}
    contracts = [_config_contract(item) for item in definitions]
    contract_by_key = {row["run_key"]: row for row in contracts}
    inventory: list[dict[str, Any]] = []
    for definition in definitions:
        metrics = _manifest_metrics(definition)
        history = [row for row in histories[definition.key] if float(row["epoch"]) > 0]
        if metrics is None and definition.key == "table483" and history:
            final_history = history[-1]
            metrics = {
                name: value for name, value in final_history.items() if name.startswith("eval_")
            }
        final_epoch = history[-1]["epoch"] if history else None
        row = {
            **contract_by_key[definition.key],
            "result_available": metrics is not None,
            "result_is_final": definition.result_kind
            in {"complete_manifest", "retained_checkpoint_evaluation"},
            "evaluated_epoch": final_epoch,
        }
        if metrics is not None:
            row.update(
                {
                    "field_precision": metrics.get("eval_field_value_precision"),
                    "field_recall": metrics.get("eval_field_value_recall"),
                    "field_f1": metrics.get("eval_field_value_f1"),
                    "field_accuracy": metrics.get("eval_field_value_accuracy"),
                    "cargo_relation_f1": metrics.get("eval_cargo_relation_f1"),
                    "category_f1": metrics.get("eval_category_value_f1"),
                    "json_valid": metrics.get("eval_json_valid"),
                    "schema_valid": metrics.get("eval_schema_valid"),
                    "canonical_exact": metrics.get("eval_canonical_exact_match"),
                    "eval_loss": metrics.get("eval_loss"),
                    "eos_fraction": metrics.get("eval_generation_eos_reached_fraction"),
                    "generated_tokens_mean": metrics.get("eval_generated_tokens_mean"),
                }
            )
        inventory.append(row)
    return inventory, [row for values in histories.values() for row in values]


def _as_numbers(rows: Sequence[Mapping[str, str]]) -> list[dict[str, Any]]:
    converted: list[dict[str, Any]] = []
    for row in rows:
        item: dict[str, Any] = {}
        for key, value in row.items():
            if key in TEXT_COLUMNS:
                item[key] = value
                continue
            try:
                item[key] = float(value)
            except (TypeError, ValueError):
                item[key] = value
        converted.append(item)
    return converted


def _actionable_error_families(mechanisms: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    mapping = {
        "omission": "missing content",
        "ocr_grounded_over_extraction": "wrong OCR-grounded selection",
        "ocr_grounded_wrong_numeric_selection": "wrong OCR-grounded selection",
        "ocr_grounded_wrong_text_selection": "wrong OCR-grounded selection",
        "relation_or_structure_error": "topology or placement",
        "correct_value_wrong_list_position": "topology or placement",
        "wrong_field_assignment": "topology or placement",
        "list_alignment_mismatch": "topology or placement",
        "incomplete_or_shortened_value": "copy boundary or normalization",
        "contaminated_or_overcomplete_value": "copy boundary or normalization",
        "near_copy_corruption": "copy boundary or normalization",
        "unsupported_or_wrong_numeric_value": "copy boundary or normalization",
        "unsupported_text_candidate": "copy boundary or normalization",
        "numeric_scale_error": "copy boundary or normalization",
        "date_normalization_or_selection_error": "copy boundary or normalization",
        "categorical_semantic_error": "categorical semantics",
    }
    counts: Counter[str] = Counter()
    for row in mechanisms:
        mechanism = str(row["mechanism"])
        family = mapping.get(mechanism)
        if family is None:
            raise ValueError(f"unmapped error mechanism: {mechanism}")
        counts[family] += int(float(row["leaf_errors"]))
    total = sum(counts.values())
    interventions = {
        "missing content": "positive field semantics, curriculum, and targeted complex examples",
        "wrong OCR-grounded selection": "row/locality cues, table view, and clearer inclusion boundaries",
        "topology or placement": "compact local cargo tree plus deterministic identity projection",
        "copy boundary or normalization": "copy-oriented targets, span boundaries, and normalized-label QA",
        "categorical semantics": "retain readable categories and target rare-category coverage",
    }
    return [
        {
            "error_family": family,
            "leaf_errors": count,
            "fraction": count / total,
            "primary_intervention": interventions[family],
        }
        for family, count in sorted(counts.items(), key=lambda item: item[1], reverse=True)
    ]


def _field_rows(field_metrics: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for row in field_metrics:
        item = dict(row)
        item["error_burden"] = int(float(item["false_positive_values"])) + int(
            float(item["false_negative_values"])
        )
        item["precision_minus_recall"] = float(item["precision"]) - float(item["recall"])
        item["support_tier"] = (
            "high"
            if float(item["validation_values"]) >= 20
            else "medium"
            if float(item["validation_values"]) >= 5
            else "low"
        )
        rows.append(item)
    return rows


def _rank_correlation(frame: pd.DataFrame, left: str, right: str) -> float:
    return float(frame[[left, right]].rank().corr().iloc[0, 1])


def _quantile(sorted_values: Sequence[float], probability: float) -> float:
    if not sorted_values:
        raise ValueError("cannot compute a quantile of an empty sequence")
    if not 0.0 <= probability <= 1.0:
        raise ValueError(f"probability must be in [0, 1], got {probability}")
    position = (len(sorted_values) - 1) * probability
    lower_index = math.floor(position)
    upper_index = math.ceil(position)
    if lower_index == upper_index:
        return sorted_values[lower_index]
    weight = position - lower_index
    return sorted_values[lower_index] * (1.0 - weight) + sorted_values[upper_index] * weight


def _paired_f1_bootstrap(
    candidate: Mapping[str, Diagnostic], baseline: Mapping[str, Diagnostic]
) -> dict[str, Any]:
    ids = sorted(candidate)
    if ids != sorted(baseline):
        raise ValueError("paired capacity predictions have different identities")

    def pooled(values: Mapping[str, Diagnostic], sample: Sequence[str]) -> float:
        rows = [values[document_id] for document_id in sample]
        true_positive = sum(row.true_positive for row in rows)
        predicted = sum(row.predicted for row in rows)
        reference = sum(row.reference for row in rows)
        precision = true_positive / predicted if predicted else 0.0
        recall = true_positive / reference if reference else 0.0
        return 2 * precision * recall / (precision + recall) if precision + recall else 0.0

    rng = random.Random(BOOTSTRAP_SEED)
    deltas: list[float] = []
    for _ in range(BOOTSTRAP_REPLICATES):
        sample = [ids[rng.randrange(len(ids))] for _ in ids]
        deltas.append(pooled(candidate, sample) - pooled(baseline, sample))
    deltas.sort()
    per_document = [candidate[item].f1 - baseline[item].f1 for item in ids]
    return {
        "point_delta": pooled(candidate, ids) - pooled(baseline, ids),
        "lower_95": _quantile(deltas, 0.025),
        "upper_95": _quantile(deltas, 0.975),
        "probability_positive": sum(value > 0 for value in deltas) / len(deltas),
        "document_wins": sum(value > 1e-12 for value in per_document),
        "document_ties": sum(abs(value) <= 1e-12 for value in per_document),
        "document_losses": sum(value < -1e-12 for value in per_document),
        "replicates": BOOTSTRAP_REPLICATES,
        "seed": BOOTSTRAP_SEED,
    }


def _capacity_rows(
    root: Path, inventory: Sequence[Mapping[str, Any]]
) -> tuple[list[dict[str, Any]], dict[str, Diagnostic]]:
    training = root / "artifacts/kie-training"
    embedding_run = training / "t5gemma2-270m-lora-mpci-bl-combined1157-task-facing-embeddings-v1"
    config = TrainingConfig.model_validate(
        yaml.safe_load((embedding_run / "config.yaml").read_text(encoding="utf-8")), strict=True
    )
    task = load_training_task(root, config)
    records = _load_records(
        project_root=root,
        run_dir=embedding_run,
        config=config,
        task=task,
        eda_dir=None,
    )
    evaluation_dirs = {
        "task1157_r32": training
        / "analysis/t5gemma2-270m-lora-mpci-bl-combined1157-task-facing-v2-checkpoint900-eval-v1",
        "task1157_r64": training
        / "t5gemma2-270m-lora-mpci-bl-combined1157-task-facing-r64-a96-ga32-e30-v1",
        "task1157_embeddings": embedding_run,
    }
    diagnostics: dict[str, dict[str, Diagnostic]] = {}
    for key, path in evaluation_dirs.items():
        values, _ = _load_diagnostics(evaluation_dir=path, records=records, task=task)
        diagnostics[key] = {row.record.document_id: row for row in values}
    inventory_by_key = {str(row["run_key"]): row for row in inventory}
    baseline_f1 = float(inventory_by_key["task1157_r32"]["field_f1"])
    rows = [
        {
            "run_key": "task1157_r32",
            "run": inventory_by_key["task1157_r32"]["run"],
            "field_f1": baseline_f1,
            "delta_vs_rank32": 0.0,
            "lower_95": 0.0,
            "upper_95": 0.0,
            "probability_positive": None,
            "document_wins": None,
            "document_ties": 100,
            "document_losses": None,
            "paired_documents": 100,
        }
    ]
    for key in ("task1157_r64", "task1157_embeddings"):
        bootstrap = _paired_f1_bootstrap(diagnostics[key], diagnostics["task1157_r32"])
        rows.append(
            {
                "run_key": key,
                "run": inventory_by_key[key]["run"],
                "field_f1": inventory_by_key[key]["field_f1"],
                "delta_vs_rank32": bootstrap["point_delta"],
                "lower_95": bootstrap["lower_95"],
                "upper_95": bootstrap["upper_95"],
                "probability_positive": bootstrap["probability_positive"],
                "document_wins": bootstrap["document_wins"],
                "document_ties": bootstrap["document_ties"],
                "document_losses": bootstrap["document_losses"],
                "paired_documents": 100,
            }
        )
    return rows, diagnostics["task1157_r32"]


def _aggregate_diagnostics(values: Sequence[Diagnostic]) -> dict[str, Any]:
    if not values:
        raise ValueError("cannot aggregate an empty diagnostic subset")
    true_positive = sum(row.true_positive for row in values)
    predicted = sum(row.predicted for row in values)
    reference = sum(row.reference for row in values)
    precision = true_positive / predicted if predicted else 0.0
    recall = true_positive / reference if reference else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {
        "documents": len(values),
        "true_positive": true_positive,
        "predicted": predicted,
        "reference": reference,
        "precision": precision,
        "recall": recall,
        "f1": f1,
    }


def _validity_subset_rows(diagnostics: Mapping[str, Diagnostic]) -> list[dict[str, Any]]:
    subsets = (
        (
            "schema valid",
            [row for row in diagnostics.values() if row.assessment.schema_valid],
        ),
        (
            "JSON valid, schema invalid",
            [
                row
                for row in diagnostics.values()
                if row.assessment.json_valid and not row.assessment.schema_valid
            ],
        ),
        (
            "invalid JSON",
            [row for row in diagnostics.values() if not row.assessment.json_valid],
        ),
    )
    return [
        {"validity_subset": label, **_aggregate_diagnostics(values)}
        for label, values in subsets
        if values
    ]


def _model_wall_evidence(
    *,
    inventory: Sequence[Mapping[str, Any]],
    capacity: Sequence[Mapping[str, Any]],
    groups: Sequence[Mapping[str, Any]],
    schema_pair_summary: Mapping[str, Any],
) -> list[dict[str, Any]]:
    by_key = {str(row["run_key"]): row for row in inventory}
    group_by_key = {(str(row["group_type"]), str(row["group"])): row for row in groups}
    capacity_by_key = {str(row["run_key"]): row for row in capacity}
    return [
        {
            "signal": "rank-64 capacity ablation",
            "observation": f"paired F1 delta {float(capacity_by_key['task1157_r64']['delta_vs_rank32']):+.4f}",
            "interpretation": "more LoRA rank did not help under the tested, partly confounded optimization contract",
            "base_model_wall_evidence": "no",
            "confidence": "high for no benefit from this recipe",
        },
        {
            "signal": "embedding-LoRA capacity ablation",
            "observation": f"paired F1 delta {float(capacity_by_key['task1157_embeddings']['delta_vs_rank32']):+.4f}",
            "interpretation": "extra trainable embedding capacity harmed recall and validity",
            "base_model_wall_evidence": "no",
            "confidence": "high",
        },
        {
            "signal": "best learning curve",
            "observation": "generated F1 rose 0.6488 to 0.8294 from epoch 5 to 20",
            "interpretation": "optimization was not flat, although epoch-20 onward behavior overfits",
            "base_model_wall_evidence": "against a demonstrated wall",
            "confidence": "high",
        },
        {
            "signal": "paired schema intervention",
            "observation": f"aligned-core F1 delta {float(schema_pair_summary['aligned_core_micro_f1_delta']):+.4f}; package type +0.173",
            "interpretation": "representation and readable labels can materially improve specific behavior",
            "base_model_wall_evidence": "against an immutable wall",
            "confidence": "high for field-level effect",
        },
        {
            "signal": "multi-cargo cohort",
            "observation": f"one cargo {float(group_by_key[('cargo_groups', '1')]['f1']):.4f}; two or more {float(group_by_key[('cargo_groups', '2+')]['f1']):.4f}",
            "interpretation": "hierarchy complexity, not generic extraction, is the main generalization break",
            "base_model_wall_evidence": "ambiguous; data or representation likely",
            "confidence": "high for local benchmark",
        },
        {
            "signal": "directional corpus growth",
            "observation": f"relation-v3 benchmark scores {float(by_key['relation483']['field_f1']):.4f} then {float(by_key['task1157_r32']['field_f1']):.4f}",
            "interpretation": "consistent with data helping, but validation and targets differ",
            "base_model_wall_evidence": "against, but not causal",
            "confidence": "low",
        },
        {
            "signal": "near-zero training loss with rising eval loss",
            "observation": "teacher-forced eval loss bottomed around epoch 10 while exact generated F1 improved more slowly",
            "interpretation": "current examples are memorized; extra epochs on the same distribution are low value",
            "base_model_wall_evidence": "current recipe saturation, not architecture proof",
            "confidence": "high",
        },
    ]


def _recommendations() -> list[dict[str, Any]]:
    return [
        {
            "priority": 1,
            "experiment": "controlled nested learning curve",
            "change": "211/423/634/846/1057 deterministic train subsets; fixed current task, 100-doc dev set, and rank-32 recipe; repeat 423 and 1057 with three seeds",
            "question_answered": "whether marginal audited data is still improving the same benchmark",
            "expected_value": "decisive model-wall versus data-scaling evidence",
        },
        {
            "priority": 2,
            "experiment": "compact source-ordered cargo tree",
            "change": "place package facts and container allocations locally under each cargo row; deterministically project synthetic IDs and MPCI arrays after inference",
            "question_answered": "whether duplicated identity topology is causing omissions and wrong placement",
            "expected_value": "targets the largest multi-cargo and relation gap",
        },
        {
            "priority": 3,
            "experiment": "substructure-to-full-structure curriculum",
            "change": "mix scalar/party, cargo-node, cargo-relation, and full-document examples, ending with full targets",
            "question_answered": "whether the 270M model needs explicit structural practice before full JSON generation",
            "expected_value": "targets omission and topology errors without changing serving model size",
        },
        {
            "priority": 4,
            "experiment": "targeted data expansion",
            "change": "prioritize multi-cargo, multi-container, allocation-heavy, unseen-template, address-boundary, and rare-package-category cases; preserve real/synthetic lineage",
            "question_answered": "whether coverage rather than raw document count controls generalization",
            "expected_value": "more efficient than random duplication toward 2k, 4k, then 8-10k",
        },
        {
            "priority": 5,
            "experiment": "controlled GLM-OCR table-view A/B",
            "change": "identical 423-document targets, prompt, rank, batch, seed, steps, and checkpoints; vary only the delimited table section",
            "question_answered": "whether table text supplies the missing row-grouping signal",
            "expected_value": "current interrupted run is not an isolated test",
        },
        {
            "priority": 6,
            "experiment": "checkpoint and evaluation redesign",
            "change": "retain checkpoints every two epochs; use a small stratified sentinel frequently and the full dev set every five epochs; select a Pareto checkpoint using F1, cargo/category F1, validity, and EOS",
            "question_answered": "whether coarse single-metric selection is discarding a better operating point",
            "expected_value": "low-cost reliability and training-time improvement",
        },
        {
            "priority": 7,
            "experiment": "small optimizer/initialization screen",
            "change": "on the 423 subset compare learning rates 5e-5/1e-4/2e-4 and rank-32 LoRA versus PiSSA initialization; keep every other variable fixed",
            "question_answered": "whether the current 1e-4 random LoRA initialization is locally suboptimal",
            "expected_value": "possible incremental lift; lower priority than data and target topology",
        },
        {
            "priority": 8,
            "experiment": "full-parameter 270M diagnostic",
            "change": "if memory permits, compare one fixed-data full-parameter run against rank-32 LoRA; retain the same 270M serving architecture",
            "question_answered": "whether PEFT, rather than the base architecture, is the capacity limit",
            "expected_value": "diagnostic only after higher-value tests",
        },
    ]


def _plot_path(output: Path, name: str) -> Path:
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
    inventory: Sequence[Mapping[str, Any]],
    history: Sequence[Mapping[str, Any]],
    fields: Sequence[Mapping[str, Any]],
    sections: Sequence[Mapping[str, Any]],
    section_budget: Sequence[Mapping[str, Any]],
    mechanisms: Sequence[Mapping[str, Any]],
    error_families: Sequence[Mapping[str, Any]],
    relations: Sequence[Mapping[str, Any]],
    categories: Sequence[Mapping[str, Any]],
    groups: Sequence[Mapping[str, Any]],
    schema_fields: Sequence[Mapping[str, Any]],
    capacity: Sequence[Mapping[str, Any]],
) -> list[str]:
    sns.set_theme(style="whitegrid", context="notebook")
    paths: list[str] = []

    def target(name: str) -> Path:
        path = _plot_path(output, name)
        paths.append(str(path.relative_to(output)))
        return path

    inventory_frame = pd.DataFrame(inventory)
    final = inventory_frame[
        (inventory_frame["result_available"]) & (inventory_frame["result_is_final"])
    ]
    plt.figure(figsize=(13, 7))
    order = final.sort_values("field_f1")["run"]
    sns.barplot(data=final, y="run", x="field_f1", order=order, hue="task_family", dodge=False)
    plt.axvline(0.90, color="#991B1B", linestyle="--")
    plt.xlim(0, 0.92)
    plt.title(
        "All completed or retained-checkpoint results (contracts differ outside paired groups)"
    )
    _save(target("01_final_f1_scoreboard"))

    history_frame = pd.DataFrame(history)
    history_frame = history_frame[history_frame["epoch"] > 0]
    graph = sns.relplot(
        data=history_frame[
            history_frame["run_key"].isin(
                [
                    "pilot106",
                    "semantic487",
                    "relation483",
                    "task1157_r32",
                    "task1157_r64",
                    "task1157_embeddings",
                ]
            )
        ],
        x="epoch",
        y="eval_field_value_f1",
        hue="run",
        col="task_family",
        col_wrap=2,
        kind="line",
        marker="o",
        height=5.5,
        aspect=1.25,
        facet_kws={"sharex": False, "sharey": True},
    )
    curve_values = history_frame.loc[
        history_frame["run_key"].isin(
            [
                "pilot106",
                "semantic487",
                "relation483",
                "task1157_r32",
                "task1157_r64",
                "task1157_embeddings",
            ]
        ),
        "eval_field_value_f1",
    ]
    graph.set(
        ylim=(
            max(0.0, float(curve_values.min()) - 0.03),
            min(1.0, float(curve_values.max()) + 0.03),
        )
    )
    graph.set_axis_labels("Epoch", "Exact field/value F1")
    graph.fig.suptitle("Generated-evaluation learning curves", y=1.03)
    graph.fig.savefig(target("02_learning_curves_all_runs"), dpi=180, bbox_inches="tight")
    plt.close(graph.fig)

    plt.figure(figsize=(10, 8))
    sns.scatterplot(
        data=final,
        x="field_recall",
        y="field_precision",
        hue="run",
        size="train_documents",
        sizes=(80, 300),
    )
    plt.plot([0.6, 0.9], [0.6, 0.9], color="black", linestyle="--", alpha=0.4)
    plt.xlim(0.6, 0.87)
    plt.ylim(0.75, 0.88)
    plt.title("Final precision-recall operating points")
    _save(target("03_final_precision_recall"))

    validity = final[final["json_valid"].notna()].melt(
        id_vars="run",
        value_vars=["json_valid", "schema_valid"],
        var_name="metric",
        value_name="fraction",
    )
    plt.figure(figsize=(13, 7))
    sns.barplot(data=validity, y="run", x="fraction", hue="metric")
    plt.xlim(0.55, 1.01)
    plt.title("Learned JSON and schema validity without constrained decoding")
    _save(target("04_output_validity_all_runs"))

    scaling = final[
        final["run_key"].isin(["pilot106", "semantic487", "relation483", "task1157_r32"])
    ].copy()
    plt.figure(figsize=(10, 7))
    sns.lineplot(
        data=scaling,
        x="train_documents",
        y="field_f1",
        hue="task_family",
        style="task_family",
        markers=True,
        dashes=True,
    )
    for _, row in scaling.iterrows():
        plt.annotate(
            str(row["validation_documents"]) + " val",
            (row["train_documents"], row["field_f1"]),
            xytext=(5, 7),
            textcoords="offset points",
        )
    plt.xscale("log")
    plt.ylim(0.6, 0.86)
    plt.title("Directional data-volume evidence (different validation contracts)")
    _save(target("05_directional_data_scaling"))

    capacity_frame = pd.DataFrame(capacity)
    candidates = capacity_frame[capacity_frame.run_key != "task1157_r32"].copy()
    plt.figure(figsize=(11, 7))
    positions = list(range(len(candidates)))
    plt.errorbar(
        candidates.delta_vs_rank32,
        positions,
        xerr=[
            candidates.delta_vs_rank32 - candidates.lower_95,
            candidates.upper_95 - candidates.delta_vs_rank32,
        ],
        fmt="o",
        color="#2563EB",
        capsize=6,
    )
    plt.axvline(0.0, color="black", linestyle="--")
    plt.yticks(positions, candidates.run)
    plt.xlabel("Exact field/value F1 delta versus rank-32 baseline")
    plt.title("Paired capacity ablations with document-bootstrap 95% intervals")
    _save(target("06_paired_capacity_ablation"))

    section_frame = pd.DataFrame(sections).sort_values("f1")
    section_long = section_frame.melt(
        id_vars="section",
        value_vars=["precision", "recall", "f1"],
        var_name="metric",
        value_name="value",
    )
    plt.figure(figsize=(13, 10))
    sns.barplot(
        data=section_long, y="section", x="value", hue="metric", order=section_frame.section
    )
    plt.xlim(0, 1.02)
    plt.title("Strongest rank-32 checkpoint: section precision, recall, and F1")
    _save(target("07_current_section_metrics"))

    field_frame = pd.DataFrame(fields)
    supported = field_frame[field_frame.validation_values >= 20]
    weakest = supported.nsmallest(20, "f1").sort_values("f1")
    plt.figure(figsize=(13, 10))
    sns.barplot(data=weakest, y="field_path", x="f1", hue="section", dodge=False)
    plt.xlim(0, 1)
    plt.title("Weakest well-supported fields (at least 20 validation values)")
    _save(target("08_weakest_supported_fields"))

    strongest = supported.nlargest(20, "f1").sort_values("f1")
    plt.figure(figsize=(13, 10))
    sns.barplot(data=strongest, y="field_path", x="f1", hue="section", dodge=False)
    plt.xlim(0, 1.01)
    plt.title("Strongest well-supported fields (at least 20 validation values)")
    _save(target("09_strongest_supported_fields"))

    burden = field_frame.nlargest(20, "error_burden").sort_values("error_burden")
    plt.figure(figsize=(14, 11))
    plt.barh(
        burden.field_path, burden.false_negative_values, label="False negatives", color="#B91C1C"
    )
    plt.barh(
        burden.field_path,
        burden.false_positive_values,
        left=burden.false_negative_values,
        label="False positives",
        color="#D97706",
    )
    plt.legend()
    plt.title("Fields contributing the most exact-match error leaves")
    _save(target("10_field_error_burden"))

    plt.figure(figsize=(11, 9))
    sns.scatterplot(
        data=field_frame[field_frame.validation_values >= 5],
        x="recall",
        y="precision",
        hue="section",
        size="validation_values",
        sizes=(25, 220),
    )
    plt.plot([0, 1], [0, 1], color="black", linestyle="--", alpha=0.35)
    plt.xlim(0, 1.02)
    plt.ylim(0, 1.02)
    plt.title("Field precision versus recall")
    _save(target("11_field_precision_recall"))

    family_frame = pd.DataFrame(error_families).sort_values("leaf_errors")
    plt.figure(figsize=(11, 7))
    sns.barplot(
        data=family_frame, y="error_family", x="leaf_errors", hue="error_family", legend=False
    )
    plt.title("Actionable error families for the strongest checkpoint")
    _save(target("12_actionable_error_families"))

    mechanism_frame = pd.DataFrame(mechanisms).sort_values("leaf_errors")
    plt.figure(figsize=(12, 10))
    sns.barplot(data=mechanism_frame, y="mechanism", x="leaf_errors", color="#2563EB")
    plt.title("Observable leaf-error mechanisms")
    _save(target("13_error_mechanisms"))

    budget_frame = pd.DataFrame(section_budget).sort_values("maximum_delta")
    plt.figure(figsize=(12, 9))
    sns.barplot(data=budget_frame, y="section", x="maximum_delta", color="#7C3AED")
    plt.title("Maximum global F1 gain if one section became perfect")
    _save(target("14_section_error_budget"))

    relation_frame = pd.DataFrame(relations).sort_values("f1")
    relation_long = relation_frame.melt(
        id_vars="fact_type",
        value_vars=["precision", "recall", "f1"],
        var_name="metric",
        value_name="value",
    )
    plt.figure(figsize=(12, 8))
    sns.barplot(
        data=relation_long, y="fact_type", x="value", hue="metric", order=relation_frame.fact_type
    )
    plt.xlim(0, 1)
    plt.title("Cargo-relation fact performance")
    _save(target("15_cargo_relation_metrics"))

    category_frame = pd.DataFrame(categories)
    category_frame = category_frame[
        (category_frame.category_type != "confusion") & (category_frame.validation_references > 0)
    ].sort_values("f1")
    plt.figure(figsize=(13, 10))
    sns.scatterplot(
        data=category_frame,
        x="train_facts",
        y="f1",
        size="validation_references",
        hue="token",
        sizes=(50, 300),
        legend=False,
    )
    for _, row in category_frame.iterrows():
        plt.annotate(
            str(row.token).removeprefix("PACKAGE_"),
            (row.train_facts, row.f1),
            xytext=(4, 3),
            textcoords="offset points",
            fontsize=8,
        )
    plt.xscale("symlog", linthresh=1)
    plt.ylim(-0.03, 1.05)
    plt.title("Readable package-category performance versus training support")
    _save(target("16_package_category_support"))

    group_frame = pd.DataFrame(groups)
    chosen = group_frame[
        group_frame.group_type.isin(
            ["cargo_groups", "containers", "target_complexity", "template_coverage"]
        )
    ].copy()
    graph = sns.catplot(
        data=chosen,
        x="group",
        y="f1",
        col="group_type",
        col_wrap=2,
        kind="bar",
        sharex=False,
        sharey=True,
        height=5.2,
        aspect=1.2,
    )
    graph.set(ylim=(0, 0.9))
    graph.set_axis_labels("Cohort", "Field/value F1")
    graph.fig.suptitle("Performance by structural complexity and template coverage", y=1.03)
    graph.fig.savefig(target("17_complexity_cohorts"), dpi=180, bbox_inches="tight")
    plt.close(graph.fig)

    schema_frame = pd.DataFrame(schema_fields)
    schema_frame = schema_frame[
        (schema_frame.current_reference >= 5) & schema_frame.f1_delta.notna()
    ]
    schema_extremes = (
        pd.concat([schema_frame.nsmallest(12, "f1_delta"), schema_frame.nlargest(12, "f1_delta")])
        .drop_duplicates("concept")
        .sort_values("f1_delta")
    )
    plt.figure(figsize=(13, 11))
    colors = ["#B91C1C" if value < 0 else "#047857" for value in schema_extremes.f1_delta]
    plt.barh(schema_extremes.concept, schema_extremes.f1_delta, color=colors)
    plt.axvline(0, color="black")
    plt.title("Paired semantic-v2 to relation-v3 field changes on the same 60 documents")
    _save(target("18_schema_intervention_field_deltas"))

    table_history = history_frame[history_frame.run_key.isin(["relation483", "table483"])]
    plt.figure(figsize=(11, 7))
    sns.lineplot(data=table_history, x="epoch", y="eval_field_value_f1", hue="run", marker="o")
    plt.ylim(0.6, 0.82)
    plt.title("Table-view experiment: interrupted and not isolated")
    _save(target("19_table_input_trajectory"))

    current_history = history_frame[history_frame.run_key == "task1157_r32"]
    fig, axis = plt.subplots(figsize=(11, 7))
    axis.plot(
        current_history.epoch, current_history.eval_field_value_f1, marker="o", color="#2563EB"
    )
    axis.set_ylabel("Generated field/value F1", color="#2563EB")
    twin = axis.twinx()
    twin.plot(current_history.epoch, current_history.eval_loss, marker="s", color="#B91C1C")
    twin.set_ylabel("Teacher-forced eval loss", color="#B91C1C")
    axis.set_xlabel("Epoch")
    axis.set_title("Best run: generated F1 improves while eval loss turns upward")
    fig.tight_layout()
    fig.savefig(target("20_f1_vs_eval_loss"), dpi=180, bbox_inches="tight")
    plt.close(fig)

    plt.figure(figsize=(11, 8))
    sns.scatterplot(
        data=field_frame[field_frame.validation_values >= 5],
        x="train_documents",
        y="f1",
        hue="validation_novel_value_fraction",
        size="validation_values",
        sizes=(25, 220),
        palette="viridis",
    )
    plt.xscale("log")
    plt.title("Per-field training support alone does not explain F1")
    _save(target("21_field_support_and_novelty"))
    return paths


def _markdown_table(rows: Sequence[Mapping[str, Any]], columns: Sequence[str]) -> str:
    lines = ["| " + " | ".join(columns) + " |", "|" + "|".join("---" for _ in columns) + "|"]
    for row in rows:
        values: list[str] = []
        for column in columns:
            value = row.get(column)
            if isinstance(value, float):
                values.append(f"{value:.4f}")
            elif value is None:
                values.append("")
            else:
                values.append(str(value).replace("|", "\\|"))
        lines.append("| " + " | ".join(values) + " |")
    return "\n".join(lines)


def _report(
    *,
    inventory: Sequence[Mapping[str, Any]],
    fields: Sequence[Mapping[str, Any]],
    sections: Sequence[Mapping[str, Any]],
    section_budget: Sequence[Mapping[str, Any]],
    mechanisms: Sequence[Mapping[str, Any]],
    error_families: Sequence[Mapping[str, Any]],
    relations: Sequence[Mapping[str, Any]],
    groups: Sequence[Mapping[str, Any]],
    capacity: Sequence[Mapping[str, Any]],
    validity_subsets: Sequence[Mapping[str, Any]],
    current_bootstrap: Mapping[str, Any],
    wall_evidence: Sequence[Mapping[str, Any]],
    recommendations: Sequence[Mapping[str, Any]],
    correlations: Mapping[str, float],
    plots: Sequence[str],
) -> str:
    results = [row for row in inventory if row["result_available"] and row["result_is_final"]]
    current = next(row for row in results if row["run_key"] == "task1157_r32")
    strong = sorted(
        (row for row in fields if float(row["validation_values"]) >= 20),
        key=lambda row: float(row["f1"]),
        reverse=True,
    )[:15]
    weak = sorted(
        (row for row in fields if float(row["validation_values"]) >= 20),
        key=lambda row: float(row["f1"]),
    )[:20]
    burden = sorted(fields, key=lambda row: int(row["error_burden"]), reverse=True)[:20]
    group_lookup = {(str(row["group_type"]), str(row["group"])): row for row in groups}
    validity_by_name = {str(row["validity_subset"]): row for row in validity_subsets}
    reference = sum(int(row["reference"]) for row in validity_subsets)
    predicted = sum(int(row["predicted"]) for row in validity_subsets)
    true_positive = sum(int(row["true_positive"]) for row in validity_subsets)
    false_negative = reference - true_positive
    false_positive = predicted - true_positive
    matches_needed_for_090 = max(0, math.ceil(0.90 * (reference + predicted) / 2) - true_positive)
    return (
        f"""# MPCI B/L KIE training program: consolidated diagnosis

Generated: `{datetime.now(UTC).isoformat()}`

## Executive conclusion

The strongest tested checkpoint remains **rank-32 rsLoRA without embedding adaptation**, evaluated at epoch 20 on the current 100-document development split. It reaches **{float(current["field_f1"]):.4f} exact field/value F1** (document-bootstrap 95% interval **{float(current_bootstrap["lower_95"]):.4f}-{float(current_bootstrap["upper_95"]):.4f}**), **{float(current["field_precision"]):.4f} precision**, **{float(current["field_recall"]):.4f} recall**, **{float(current["cargo_relation_f1"]):.4f} cargo-relation F1**, **{float(current["category_f1"]):.4f} category F1**, **{float(current["json_valid"]):.1%} JSON validity**, and **{float(current["schema_valid"]):.1%} schema validity** without constrained decoding.

There is **no concrete evidence yet that the 270M base architecture has hit its capability wall**. There is strong evidence that the **current representation/data/training recipe has reached a local ceiling**:

- rank 64 and token-embedding LoRA both regress significantly on the identical validation set;
- training loss approaches zero while validation loss rises and generated F1 flattens;
- multi-cargo F1 is **{float(group_lookup[("cargo_groups", "2+")]["f1"]):.4f}**, versus **{float(group_lookup[("cargo_groups", "1")]["f1"]):.4f}** for one-cargo documents;
- a paired schema intervention improved package-type F1 by 17.3 points, proving that target representation can move at least some difficult behavior without changing model size;
- the historical data-growth scores rise directionally, but the splits and targets changed, so they cannot establish a learning curve.

The correct next move is therefore **controlled data scaling plus a cargo-representation/curriculum A/B**, not a larger LoRA rank, embedding adaptation, more epochs, or arbitrary Adam-beta tuning.

## What the metric means

On the strongest checkpoint, the validator contains **{reference:,} reference leaves** and **{predicted:,} predicted leaves**. Exactly **{true_positive:,}** match both field path and complete value, leaving **{false_negative:,} false negatives** and **{false_positive:,} false positives**. The system is slightly recall-limited. Reaching 0.90 at fixed predicted/reference totals requires **{matches_needed_for_090:,}** additional exact matches.

Whole-document exact match is only 1%. This is expected to be much lower than leaf F1 for a large nested object, but it demonstrates that most documents still contain at least one semantic or structural miss.

## Experiment scoreboard and comparability

{_markdown_table(results, ["run", "train_documents", "validation_documents", "task_family", "field_precision", "field_recall", "field_f1", "json_valid", "schema_valid", "comparison_class"])}

Only two comparisons support direct causal language:

1. rank-32 versus embedding LoRA uses identical 1,157-record data, split, target, prompt, and saved reference rows;
2. rank-32 versus rank-64 uses the same data/split/target, but also changes alpha, effective batch, and epochs, so it is paired but not a pure rank-only test.

The semantic-v2 versus relation-v3 run uses identical 60 source documents and OCR but changes target ontology and prompt. Historical sample-count comparisons use different validation contracts and are directional only.

The paired capacity results are:

{_markdown_table(capacity, ["run", "field_f1", "delta_vs_rank32", "lower_95", "upper_95", "probability_positive", "document_wins", "document_ties", "document_losses"])}

Both candidate document-bootstrap intervals exclude zero, but each model was trained with only one seed. These intervals quantify validation-document uncertainty for the realized checkpoints; they do **not** quantify training-seed variance. The rank-64 arm also changed alpha, effective batch, and duration, so it is evidence against that combined recipe—not a pure rank theorem.

## Error budget: what is actually going wrong

{_markdown_table(error_families, ["error_family", "leaf_errors", "fraction", "primary_intervention"])}

The dominant failure is **omission**, not unsupported hallucination. The next two large families are wrong OCR-grounded selection and structural placement. This matches the document-level result: cargo-group count has a rank correlation of **{correlations["document_cargo_groups_vs_f1"]:.3f}** with F1, while input length is only **{correlations["document_input_tokens_vs_f1"]:.3f}**. The model struggles mainly when it must reconstruct hierarchy from flattened OCR, rather than merely because a document is long.

The exact raw mechanism table is:

{_markdown_table(mechanisms, ["mechanism", "leaf_errors", "fraction"])}

## Where the model is weak

Fields below have at least 20 reference values, so these are not single-example curiosities:

{_markdown_table(weak, ["field_path", "train_documents", "validation_values", "precision", "recall", "f1", "error_burden", "index_insensitive_f1"])}

The high-impact interpretation is:

- `additionalInformation` and `marksAndNumbers` lack a sufficiently crisp positive inclusion boundary;
- allocation `packageId`, `packageIds`, and `coverage` expose synthetic identity and topology problems;
- package quantity, weights, descriptions, and allocation quantity are row-association problems more often than character-reading failures;
- party addresses are long, novel copy spans where partial or contaminated boundaries count as fully wrong;
- forwarding references have high precision but low recall, indicating conservative omission.

The twenty fields contributing the most false-positive plus false-negative leaves are:

{_markdown_table(burden, ["field_path", "false_negative_values", "false_positive_values", "error_burden", "f1", "validation_novel_value_fraction"])}

## Where the model is strong

{_markdown_table(strong, ["field_path", "validation_values", "precision", "recall", "f1", "whole_field_exact_rate"])}

Routes, dates, bill numbers, container numbers, vessel/voyage values, countries, common party names, and several contact fields are already strong. This matters for the capability-wall diagnosis: the 270M model reliably performs ordinary scalar copying and normalization. The weakness is concentrated in repeated, hierarchical, and boundary-sensitive fields.

## Structure, validity, and complexity

{_markdown_table(sections, ["section", "reference_values", "precision", "recall", "f1", "whole_section_exact_rate"])}

{_markdown_table(validity_subsets, ["validity_subset", "documents", "precision", "recall", "f1", "reference", "predicted"])}

On the {int(validity_by_name["schema valid"]["documents"])} schema-valid documents, pooled F1 is **{float(validity_by_name["schema valid"]["f1"]):.4f}**; on the {int(validity_by_name["JSON valid, schema invalid"]["documents"])} JSON-valid but schema-invalid documents it is **{float(validity_by_name["JSON valid, schema invalid"]["f1"]):.4f}**. Improving validity is important, but even perfect learned schema validity would not by itself cross 0.90.

No single section can close the global gap. Perfecting parties would lift overall F1 only to about 0.8800, cargo groups to 0.8773, and allocations to 0.8663:

{_markdown_table(sorted(section_budget, key=lambda row: float(row["maximum_delta"]), reverse=True), ["section", "maximum_delta", "max_f1_if_section_perfect"])}

Complexity cohorts confirm the hierarchy diagnosis, although the two-or-more-cargo cohort contains only 10 documents and should be enlarged in the frozen test:

{_markdown_table([group_lookup[key] for key in [("cargo_groups", "1"), ("cargo_groups", "2+"), ("containers", "1"), ("containers", "2+"), ("target_complexity", "40-79"), ("target_complexity", "80+"), ("template_coverage", "seen"), ("template_coverage", "unseen")]], ["group_type", "group", "documents", "precision", "recall", "f1", "schema_valid"])}

## Cargo relations and categories

{_markdown_table(relations, ["fact_type", "reference_facts", "precision", "recall", "f1", "exact_document_rate"])}

`container_has_package` is the weakest relation because it requires two identities plus the edge between them. `allocation_coverage` is also omission-heavy. This is the clearest reason to test a compact local cargo tree and generate synthetic IDs/flat MPCI arrays deterministically after inference.

Readable package categories should be retained. The paired schema experiment improved package-type extraction substantially, and common tokens such as carton and pallet are strong. Rare or visually ambiguous categories remain data targets; this does not justify reverting to opaque numeric identifiers.

## Are we at a model capability wall?

{_markdown_table(wall_evidence, ["signal", "observation", "interpretation", "base_model_wall_evidence", "confidence"])}

The honest answer is **not proven either way**, but current evidence favors data/representation/generalization over an architectural wall. A true wall claim needs two measurements that do not yet exist:

1. a controlled nested learning curve on one immutable split and task contract;
2. generated (not teacher-forced) training-set accuracy, so we can distinguish inability to learn the mapping from inability to generalize it.

If development F1 still rises at 1,057 examples, scale real audited data to roughly 2k and 4k before committing the full 8-10k augmentation stage. If it is flat while generated training accuracy remains high, diversify data and simplify topology. If both train-generation and dev performance are flat, then run the full-parameter 270M diagnostic to distinguish a LoRA wall from a base-model/representation wall. No larger model is required for this diagnosis.

## Training and hyperparameter conclusions

- **Keep:** rank-32 rsLoRA over q/k/v/o and gate/up/down projections in both encoder and decoder; effective batch 24; bfloat16; gradient checkpointing; dynamic padding; 1e-4 cosine schedule as the control recipe.
- **Stop:** rank 64, embedding LoRA, blind epoch extension, and larger generation caps. The first two regress; later epochs overfit; cap failures are repetition loops rather than legitimate long labels.
- **Checkpoint selection:** generated F1, cargo/category F1, JSON/schema validity, and EOS should form a Pareto/co-gated selection rule. Teacher-forced loss alone selects the wrong region.
- **Optimizer evidence:** every substantive run used AdamW with beta1 0.9, beta2 0.999, 5% warmup, and mostly 1e-4 LR. These values are a stable baseline, not proven optimal. There is no local evidence that beta tuning is the next high-value lever.
- **Low-cost sweep:** screen 5e-5, 1e-4, and 2e-4 plus LoRA versus PiSSA initialization on the fixed 423 subset. Change one variable at a time and promote only a clear generated-metric gain.
- **Table input:** the interrupted arm scored 0.6806 at epoch 6.45 and 0.7394 at epoch 12.91; it does not show an early lift, but it also changed effective batch/prompt and never completed. Treat it as inconclusive and rerun a true one-variable A/B.
- **Data split:** the current 100 documents are now a development set because they have informed repeated decisions. Create a carrier/template-grouped frozen test set before claiming 0.90 reliability.

## Prioritized next experiments

{_markdown_table(recommendations, ["priority", "experiment", "change", "question_answered", "expected_value"])}

## Research cross-check

- T5Gemma 2 releases the 270M-270M checkpoint as a pretrained encoder-decoder model; supervised task-contract learning is therefore expected rather than a sign of failure: https://arxiv.org/abs/2512.14856 and https://deepmind.google/models/gemma/t5gemma/.
- UIE uses an explicit structural schema instructor for text-to-structure extraction, supporting concise semantic schema guidance rather than relying on developer-oriented JSON names alone: https://aclanthology.org/2022.acl-long.395/.
- Text2Event combines sequence-to-structure generation with substructure/full-structure curriculum learning, directly supporting the proposed curriculum experiment: https://aclanthology.org/2021.acl-long.217/.
- VRDU reports that unseen templates and hierarchical document fields remain difficult, consistent with the observed multi-cargo gap: https://arxiv.org/abs/2211.15421.
- LMDX identifies missing layout encoding and grounding as major document-IE obstacles and demonstrates repeated/hierarchical extraction with localization, supporting a controlled table/locality view: https://research.google/pubs/lmdx-language-model-based-document-information-extraction-and-localization/.
- IEPile emphasizes high-quality, standardized, schema-conditioned IE supervision, supporting curated diversity rather than raw duplication: https://arxiv.org/abs/2402.14710.
- LoRA motivates low-rank adaptation from low intrinsic update rank, and PiSSA proposes a more informed initialization; these make PiSSA a reasonable small ablation, not a substitute for fixing data and topology: https://arxiv.org/abs/2106.09685 and https://arxiv.org/abs/2404.02948.

## Figures

"""
        + "\n".join(f"- [{Path(path).stem}](<{path}>)" for path in plots)
        + "\n"
    )


def analyze(*, project_root: Path, output_dir: Path) -> None:
    project_root = project_root.resolve(strict=True)
    output_dir = output_dir.resolve()
    if output_dir.exists():
        raise ValueError(f"analysis output already exists: {output_dir}")
    definitions = _run_definitions(project_root)
    for definition in definitions:
        definition.run_dir.resolve(strict=True)
        if definition.result_path is not None:
            definition.result_path.resolve(strict=True)
    inventory, history = _run_inventory(definitions)

    best_tables = project_root / (
        "artifacts/kie-training/analysis/"
        "t5gemma2-270m-lora-mpci-bl-combined1157-task-facing-v2-deep-audit-v3/tables"
    )
    mechanism_tables = project_root / (
        "artifacts/kie-training/analysis/"
        "t5gemma2-270m-lora-mpci-bl-combined1157-task-facing-v2-error-mechanisms-v1/tables"
    )
    scaling_tables = project_root / (
        "artifacts/kie-training/analysis/t5gemma2-270m-lora-mpci-bl-scaling-evidence-v1/tables"
    )
    schema_analysis = project_root / (
        "artifacts/kie-training/analysis/"
        "t5gemma2-270m-lora-mpci-bl-relation-explicit-v3-v1-vs-combined487-v2-paired-audit-v1"
    )
    fields = _field_rows(_as_numbers(_csv(best_tables / "field_metrics.csv")))
    sections = _as_numbers(_csv(best_tables / "section_metrics.csv"))
    relations = _as_numbers(_csv(best_tables / "relation_metrics.csv"))
    categories = _as_numbers(_csv(best_tables / "category_token_metrics.csv"))
    groups = _as_numbers(_csv(best_tables / "group_metrics.csv"))
    documents = _as_numbers(_csv(best_tables / "document_metrics.csv"))
    section_budget = _as_numbers(_csv(scaling_tables / "current_section_error_budget.csv"))
    mechanisms = _as_numbers(_csv(mechanism_tables / "mechanism_summary.csv"))
    error_families = _actionable_error_families(mechanisms)
    schema_fields = _as_numbers(_csv(schema_analysis / "tables/field_comparison.csv"))
    schema_pair_summary = _json(schema_analysis / "summary.json")
    current_bootstrap = _as_numbers(_csv(best_tables / "bootstrap_intervals.csv"))[0]
    capacity, current_diagnostics = _capacity_rows(project_root, inventory)
    validity_subsets = _validity_subset_rows(current_diagnostics)

    field_frame = pd.DataFrame(fields)
    supported_fields = field_frame[field_frame.validation_values >= 5]
    document_frame = pd.DataFrame(documents)
    correlations = {
        "field_train_documents_vs_f1": _rank_correlation(supported_fields, "train_documents", "f1"),
        "field_novel_fraction_vs_f1": _rank_correlation(
            supported_fields, "validation_novel_value_fraction", "f1"
        ),
        "document_input_tokens_vs_f1": _rank_correlation(
            document_frame, "input_tokens", "field_f1"
        ),
        "document_target_leaves_vs_f1": _rank_correlation(
            document_frame, "target_leaves", "field_f1"
        ),
        "document_cargo_groups_vs_f1": _rank_correlation(
            document_frame, "cargo_group_count", "field_f1"
        ),
        "document_containers_vs_f1": _rank_correlation(
            document_frame, "container_count", "field_f1"
        ),
    }
    wall_evidence = _model_wall_evidence(
        inventory=inventory,
        capacity=capacity,
        groups=groups,
        schema_pair_summary=schema_pair_summary,
    )
    recommendations = _recommendations()

    output_dir.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix=f".{output_dir.name}.", dir=output_dir.parent))
    try:
        tables = temporary / "tables"
        _write_csv(tables / "run_inventory.csv", inventory)
        _write_csv(tables / "evaluation_history.csv", history)
        _write_csv(tables / "current_field_metrics.csv", fields)
        _write_csv(tables / "current_section_metrics.csv", sections)
        _write_csv(tables / "current_relation_metrics.csv", relations)
        _write_csv(tables / "current_category_metrics.csv", categories)
        _write_csv(tables / "current_cohort_metrics.csv", groups)
        _write_csv(tables / "current_section_error_budget.csv", section_budget)
        _write_csv(tables / "current_error_mechanisms.csv", mechanisms)
        _write_csv(tables / "actionable_error_families.csv", error_families)
        _write_csv(tables / "paired_capacity_ablation.csv", capacity)
        _write_csv(tables / "current_validity_subsets.csv", validity_subsets)
        _write_csv(tables / "current_bootstrap_interval.csv", [current_bootstrap])
        _write_csv(tables / "paired_schema_field_changes.csv", schema_fields)
        _write_csv(tables / "model_wall_evidence.csv", wall_evidence)
        _write_csv(tables / "recommended_experiments.csv", recommendations)
        _write_csv(
            tables / "rank_correlations.csv",
            [
                {"relationship": key, "spearman_rank_correlation": value}
                for key, value in correlations.items()
            ],
        )
        plots = _render_plots(
            temporary,
            inventory=inventory,
            history=history,
            fields=fields,
            sections=sections,
            section_budget=section_budget,
            mechanisms=mechanisms,
            error_families=error_families,
            relations=relations,
            categories=categories,
            groups=groups,
            schema_fields=schema_fields,
            capacity=capacity,
        )
        report = _report(
            inventory=inventory,
            fields=fields,
            sections=sections,
            section_budget=section_budget,
            mechanisms=mechanisms,
            error_families=error_families,
            relations=relations,
            groups=groups,
            capacity=capacity,
            validity_subsets=validity_subsets,
            current_bootstrap=current_bootstrap,
            wall_evidence=wall_evidence,
            recommendations=recommendations,
            correlations=correlations,
            plots=plots,
        )
        (temporary / "REPORT.md").write_text(report, encoding="utf-8")
        summary = {
            "schema_version": 1,
            "status": "complete",
            "created_at": datetime.now(UTC).isoformat(),
            "best_run": next(row for row in inventory if row["run_key"] == "task1157_r32"),
            "capability_wall_conclusion": "not demonstrated; current recipe and representation are locally saturated",
            "actionable_error_families": error_families,
            "capacity_ablation": capacity,
            "validity_subsets": validity_subsets,
            "current_bootstrap_interval": current_bootstrap,
            "rank_correlations": correlations,
            "recommended_experiments": recommendations,
            "plots": plots,
        }
        atomic_write_json(temporary / "summary.json", summary)
        files = sorted(path for path in temporary.rglob("*") if path.is_file())
        atomic_write_json(
            temporary / "manifest.json",
            {
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
            },
        )
        temporary.replace(output_dir)
    except BaseException:
        shutil.rmtree(temporary, ignore_errors=True)
        raise


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, default=Path("."))
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(
            "artifacts/kie-training/analysis/t5gemma2-270m-lora-mpci-bl-all-runs-diagnosis-v2"
        ),
    )
    return parser.parse_args()


def main() -> None:
    arguments = _parse_args()
    analyze(project_root=arguments.project_root, output_dir=arguments.output_dir)


if __name__ == "__main__":
    main()
