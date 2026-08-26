#!/usr/bin/env python3
# ruff: noqa: E501
"""Publish a paired semantic-v2 versus relation-explicit-v3 KIE run audit.

The comparison is artifact-only: it verifies completed analysis publications, reloads
the pinned datasets and saved predictions, and never loads model weights or touches a
GPU.  The 60 validation documents must be identical by document ID and raw-OCR hash.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import itertools
import json
import math
import random
import re
import statistics
import sys
from collections import Counter, defaultdict
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, cast

import yaml

TOOLS_DIR = Path(__file__).resolve().parent
if str(TOOLS_DIR) not in sys.path:
    sys.path.insert(0, str(TOOLS_DIR))

from analyze_kie_training_run import (  # noqa: E402
    PALETTE,
    DatasetRecord,
    _grouped_bar_chart,
    _histogram,
    _horizontal_bar_chart,
    _line_chart,
    _load_dataset_records,
    _scatter_chart,
)
from audit_kie_reliability import (  # noqa: E402
    _flatten_ordered,
    _parse_patch,
    _render_heatmap,
    raw_grounding,
    value_similarity,
)

from document_ocr.training.config import TrainingConfig  # noqa: E402
from document_ocr.training.metrics import (  # noqa: E402
    PredictionAssessment,
    _relation_explicit_facts,
    assess_prediction,
)
from document_ocr.training.tasks import (  # noqa: E402
    TrainingTask,
    canonical_json,
    load_training_task,
)

COMPARISON_SCHEMA_VERSION = 1
BOOTSTRAP_REPLICATES = 10_000
BOOTSTRAP_SEED = 424
INDEX_PATTERN = re.compile(r"\[\d+\]")
TOKEN_PATTERN = re.compile(r"\w+|[^\w\s]", re.UNICODE)
TARGET_F1 = 0.90

RELATION_SCAFFOLD_FIELDS = frozenset(
    {
        "$.documentPatch.cargoGroups[].groupId",
        "$.documentPatch.cargoPackages[].groupId",
        "$.documentPatch.cargoPackages[].packageId",
        "$.documentPatch.cargoAllocationGroups[].groupId",
        "$.documentPatch.cargoAllocationGroups[].coverage",
        "$.documentPatch.cargoAllocationGroups[].packageIds[]",
        "$.documentPatch.cargoAllocationGroups[].allocations[].packageId",
    }
)


@dataclass(frozen=True, slots=True)
class Counts:
    true_positive: int
    predicted: int
    reference: int

    @property
    def precision(self) -> float:
        return self.true_positive / self.predicted if self.predicted else 0.0

    @property
    def recall(self) -> float:
        return self.true_positive / self.reference if self.reference else 0.0

    @property
    def f1(self) -> float:
        denominator = self.predicted + self.reference
        return 2 * self.true_positive / denominator if denominator else 0.0


@dataclass(frozen=True, slots=True)
class FieldConcept:
    name: str
    component: str
    comparison_kind: str
    previous_paths: tuple[str, ...]
    current_paths: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class RunBundle:
    label: str
    run_dir: Path
    analysis_dir: Path
    config: dict[str, Any]
    task: TrainingTask
    records: dict[str, DatasetRecord]
    predictions: dict[str, PredictionAssessment]
    prediction_rows: dict[str, dict[str, Any]]
    document_rows: dict[str, dict[str, str]]
    field_rows: dict[str, dict[str, str]]
    eval_rows: list[dict[str, str]]
    summary: dict[str, Any]
    manifest: dict[str, Any]
    schema_errors: list[dict[str, Any]]
    mlflow_run: dict[str, Any]


def _normalize_path(path: str) -> str:
    return INDEX_PATTERN.sub("[]", path)


def _safe_divide(numerator: float, denominator: float) -> float:
    return numerator / denominator if denominator else 0.0


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _json_load(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _jsonl_load(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                raise ValueError(f"blank JSONL row at {path}:{line_number}")
            value = json.loads(line)
            if not isinstance(value, dict):
                raise ValueError(f"JSONL row is not an object at {path}:{line_number}")
            rows.append(cast(dict[str, Any], value))
    return rows


def _csv_load(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return [dict(row) for row in csv.DictReader(handle)]


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2) + "\n",
        encoding="utf-8",
    )


def _write_jsonl(path: Path, rows: Iterable[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(
                json.dumps(dict(row), ensure_ascii=False, sort_keys=True, separators=(",", ":"))
                + "\n"
            )


def _write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    if not rows:
        raise ValueError(f"cannot publish an empty CSV: {path}")
    fieldnames = sorted({key for row in rows for key in row})
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="raise")
        writer.writeheader()
        writer.writerows(rows)


def _verify_analysis_manifest(path: Path) -> dict[str, Any]:
    manifest = cast(dict[str, Any], _json_load(path / "analysis_manifest.json"))
    entries = manifest.get("artifacts")
    if not isinstance(entries, list):
        entries = manifest.get("files")
    if not isinstance(entries, list) or not entries:
        raise ValueError(f"analysis manifest has no artifact inventory: {path}")
    mismatches: list[str] = []
    for item in entries:
        if not isinstance(item, dict):
            raise ValueError(f"malformed analysis manifest entry: {path}")
        artifact = path / str(item["path"])
        if (
            not artifact.is_file()
            or artifact.stat().st_size != int(item["bytes"])
            or _sha256(artifact) != str(item["sha256"])
        ):
            mismatches.append(str(item["path"]))
    if mismatches:
        raise ValueError(f"analysis artifact integrity failed at {path}: {mismatches[:10]}")
    return {"artifacts": len(entries), "integrity_passed": True}


def _load_bundle(
    project_root: Path,
    label: str,
    run_dir: Path,
    analysis_dir: Path,
) -> RunBundle:
    _verify_analysis_manifest(analysis_dir)
    config = cast(
        dict[str, Any], yaml.safe_load((run_dir / "config.yaml").read_text(encoding="utf-8"))
    )
    validated = TrainingConfig.model_validate(config)
    task = load_training_task(project_root, validated)
    records, _ = _load_dataset_records(project_root, config)
    validation_ids = {
        record.document_id for record in records.values() if record.split == "validation"
    }
    prediction_rows = {
        str(row["document_id"]): row
        for row in _jsonl_load(analysis_dir / "recovered-predictions" / "validation.jsonl")
    }
    if set(prediction_rows) != validation_ids:
        raise ValueError(f"{label} recovered prediction IDs do not match the validation split")
    predictions: dict[str, PredictionAssessment] = {}
    for document_id, row in prediction_rows.items():
        assessment = assess_prediction(
            str(row["generated_text"]), str(row["reference_text"]), task
        )
        expected = canonical_json(task.canonicalize(records[document_id].target))
        if assessment.reference_text != expected:
            raise ValueError(f"{label} reference identity mismatch: {document_id}")
        predictions[document_id] = assessment
    document_rows_list = _csv_load(analysis_dir / "tables" / "document_metrics.csv")
    field_rows_list = _csv_load(analysis_dir / "tables" / "field_metrics.csv")
    document_rows = {row["document_id"]: row for row in document_rows_list}
    field_rows = {row["field_path"]: row for row in field_rows_list}
    if set(document_rows) != validation_ids:
        raise ValueError(f"{label} document metric IDs do not match the validation split")
    return RunBundle(
        label=label,
        run_dir=run_dir,
        analysis_dir=analysis_dir,
        config=config,
        task=task,
        records=records,
        predictions=predictions,
        prediction_rows=prediction_rows,
        document_rows=document_rows,
        field_rows=field_rows,
        eval_rows=_csv_load(analysis_dir / "tables" / "eval_history.csv"),
        summary=cast(dict[str, Any], _json_load(analysis_dir / "summary.json")),
        manifest=cast(dict[str, Any], _json_load(run_dir / "manifest.json")),
        schema_errors=_jsonl_load(analysis_dir / "tables" / "schema_errors.jsonl"),
        mlflow_run=cast(dict[str, Any], _json_load(analysis_dir / "mlflow_run.json")),
    )


def _assert_paired_validation(previous: RunBundle, current: RunBundle) -> list[str]:
    previous_ids = {
        record.document_id
        for record in previous.records.values()
        if record.split == "validation"
    }
    current_ids = {
        record.document_id
        for record in current.records.values()
        if record.split == "validation"
    }
    if previous_ids != current_ids:
        raise ValueError("validation document IDs differ between the compared runs")
    for document_id in previous_ids:
        left = previous.records[document_id]
        right = current.records[document_id]
        if left.raw_text_sha256 != right.raw_text_sha256 or left.raw_text != right.raw_text:
            raise ValueError(f"validation raw OCR differs between runs: {document_id}")
    return sorted(previous_ids)


def _counts_from_pairs(
    predicted: Iterable[tuple[str, str]], reference: Iterable[tuple[str, str]]
) -> Counts:
    predicted_set = set(predicted)
    reference_set = set(reference)
    return Counts(
        true_positive=len(predicted_set & reference_set),
        predicted=len(predicted_set),
        reference=len(reference_set),
    )


def _counts_for_assessment(
    assessment: PredictionAssessment,
    *,
    include_fields: set[str] | None = None,
    exclude_fields: set[str] | frozenset[str] = frozenset(),
) -> Counts:
    def selected(item: tuple[str, str]) -> bool:
        field = _normalize_path(item[0])
        return field not in exclude_fields and (
            include_fields is None or field in include_fields
        )

    return _counts_from_pairs(
        (item for item in assessment.predicted_field_values if selected(item)),
        (item for item in assessment.reference_field_values if selected(item)),
    )


def _sum_counts(values: Iterable[Counts]) -> Counts:
    rows = list(values)
    return Counts(
        true_positive=sum(row.true_positive for row in rows),
        predicted=sum(row.predicted for row in rows),
        reference=sum(row.reference for row in rows),
    )


def _counts_row(prefix: str, value: Counts) -> dict[str, Any]:
    return {
        f"{prefix}_true_positive": value.true_positive,
        f"{prefix}_predicted": value.predicted,
        f"{prefix}_reference": value.reference,
        f"{prefix}_precision": value.precision,
        f"{prefix}_recall": value.recall,
        f"{prefix}_f1": value.f1,
    }


def _field_metric_counts(bundle: RunBundle, paths: Sequence[str]) -> Counts:
    rows = [bundle.field_rows[path] for path in paths]
    return Counts(
        true_positive=sum(int(row["true_positive_values"]) for row in rows),
        predicted=sum(int(row["predicted_values"]) for row in rows),
        reference=sum(int(row["reference_values"]) for row in rows),
    )


def _field_concepts(previous: RunBundle, current: RunBundle) -> list[FieldConcept]:
    concepts: list[FieldConcept] = []
    common = sorted(set(previous.field_rows) & set(current.field_rows))
    for path in common:
        if path.startswith("$.documentPatch.goodsItems") or path.startswith(
            "$.documentPatch.cargo"
        ):
            continue
        if path == "$.documentPatch.containers[].typeDescription":
            continue
        concepts.append(
            FieldConcept(
                name=path.removeprefix("$.documentPatch."),
                component="shared_document_fields",
                comparison_kind="same_field_and_representation",
                previous_paths=(path,),
                current_paths=(path,),
            )
        )
    cargo_suffixes = (
        "additionalInformation[]",
        "dangerousGoods[].unNumber",
        "description",
        "grossWeight.unit",
        "grossWeight.value",
        "handlingInstructions[]",
        "hsCodes[]",
        "marksAndNumbers[]",
        "netWeight.unit",
        "netWeight.value",
        "origin.identifier",
        "origin.name",
        "volume.unit",
        "volume.value",
    )
    for suffix in cargo_suffixes:
        concepts.append(
            FieldConcept(
                name=f"cargo.{suffix}",
                component="cargo_content",
                comparison_kind="same_value_policy_relocated_field",
                previous_paths=(f"$.documentPatch.goodsItems[].{suffix}",),
                current_paths=(f"$.documentPatch.cargoGroups[].{suffix}",),
            )
        )
    concepts.extend(
        [
            FieldConcept(
                name="cargo.package.quantity",
                component="package_quantity",
                comparison_kind="same_value_policy_relocated_field",
                previous_paths=("$.documentPatch.goodsItems[].packages[].quantity",),
                current_paths=("$.documentPatch.cargoPackages[].quantity",),
            ),
            FieldConcept(
                name="cargo.package.type",
                component="package_type_representation",
                comparison_kind="changed_to_readable_category_or_verbatim_fallback",
                previous_paths=("$.documentPatch.goodsItems[].packages[].type",),
                current_paths=(
                    "$.documentPatch.cargoPackages[].typeCategory",
                    "$.documentPatch.cargoPackages[].typeDescription",
                ),
            ),
            FieldConcept(
                name="cargo.allocation.containerNumber",
                component="allocation_values",
                comparison_kind="same_value_policy_relocated_field",
                previous_paths=(
                    "$.documentPatch.goodsItems[].containerAllocations[].containerNumber",
                ),
                current_paths=(
                    "$.documentPatch.cargoAllocationGroups[].allocations[].containerNumber",
                ),
            ),
            FieldConcept(
                name="cargo.allocation.packageQuantity",
                component="allocation_values",
                comparison_kind="same_value_policy_relocated_field",
                previous_paths=(
                    "$.documentPatch.goodsItems[].containerAllocations[].packageQuantity",
                ),
                current_paths=(
                    "$.documentPatch.cargoAllocationGroups[].allocations[].packageQuantity",
                ),
            ),
            FieldConcept(
                name="cargo.dangerousGoods.hazard",
                component="dangerous_goods_category",
                comparison_kind="changed_to_readable_category",
                previous_paths=(
                    "$.documentPatch.goodsItems[].dangerousGoods[].hazardClass",
                ),
                current_paths=(
                    "$.documentPatch.cargoGroups[].dangerousGoods[].hazardCategory",
                ),
            ),
            FieldConcept(
                name="container.type",
                component="container_type_representation",
                comparison_kind="code_and_description_collapsed_to_verbatim_description",
                previous_paths=(
                    "$.documentPatch.containers[].typeCode",
                    "$.documentPatch.containers[].typeDescription",
                ),
                current_paths=("$.documentPatch.containers[].typeDescription",),
            ),
        ]
    )
    for concept in concepts:
        missing_previous = set(concept.previous_paths) - set(previous.field_rows)
        missing_current = set(concept.current_paths) - set(current.field_rows)
        if missing_previous or missing_current:
            raise ValueError(
                f"field correspondence is not present in both analyses: {concept.name}: "
                f"previous={sorted(missing_previous)}, current={sorted(missing_current)}"
            )
    return concepts


def _field_comparison_rows(
    previous: RunBundle, current: RunBundle, concepts: Sequence[FieldConcept]
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for concept in concepts:
        left = _field_metric_counts(previous, concept.previous_paths)
        right = _field_metric_counts(current, concept.current_paths)
        rows.append(
            {
                "concept": concept.name,
                "component": concept.component,
                "comparison_kind": concept.comparison_kind,
                "previous_paths": " | ".join(concept.previous_paths),
                "current_paths": " | ".join(concept.current_paths),
                **_counts_row("previous", left),
                **_counts_row("current", right),
                "precision_delta": right.precision - left.precision,
                "recall_delta": right.recall - left.recall,
                "f1_delta": right.f1 - left.f1,
            }
        )
    return rows


def _component_rows(
    previous: RunBundle, current: RunBundle, concepts: Sequence[FieldConcept]
) -> list[dict[str, Any]]:
    grouped: dict[str, list[FieldConcept]] = defaultdict(list)
    for concept in concepts:
        grouped[concept.component].append(concept)
    rows: list[dict[str, Any]] = []
    for component, values in sorted(grouped.items()):
        left = _sum_counts(
            _field_metric_counts(previous, concept.previous_paths) for concept in values
        )
        right = _sum_counts(
            _field_metric_counts(current, concept.current_paths) for concept in values
        )
        rows.append(
            {
                "component": component,
                "concepts": len(values),
                **_counts_row("previous", left),
                **_counts_row("current", right),
                "precision_delta": right.precision - left.precision,
                "recall_delta": right.recall - left.recall,
                "f1_delta": right.f1 - left.f1,
            }
        )
    return rows


def _fact_counts(
    predicted: Iterable[tuple[str, ...]], reference: Iterable[tuple[str, ...]]
) -> Counts:
    predicted_set = set(predicted)
    reference_set = set(reference)
    return Counts(
        true_positive=len(predicted_set & reference_set),
        predicted=len(predicted_set),
        reference=len(reference_set),
    )


def _counter_metric(predicted: Counter[str], reference: Counter[str]) -> Counts:
    return Counts(
        true_positive=sum((predicted & reference).values()),
        predicted=sum(predicted.values()),
        reference=sum(reference.values()),
    )


def _repeated_ngram_fraction(text: str, size: int = 8) -> float:
    tokens = TOKEN_PATTERN.findall(text)
    ngrams = [tuple(tokens[index : index + size]) for index in range(len(tokens) - size + 1)]
    return 1 - _safe_divide(len(set(ngrams)), len(ngrams)) if ngrams else 0.0


def _maximum_character_run(text: str) -> int:
    return max((sum(1 for _ in group) for _, group in itertools.groupby(text)), default=0)


def _relation_structure(reference_text: str) -> dict[str, int]:
    value = json.loads(reference_text)
    patch = cast(dict[str, Any], value["documentPatch"])
    groups = patch.get("cargoGroups")
    packages = patch.get("cargoPackages")
    allocations = patch.get("cargoAllocationGroups")
    containers = patch.get("containers")
    allocation_rows = 0
    if isinstance(allocations, list):
        allocation_rows = sum(
            len(item.get("allocations", []))
            for item in allocations
            if isinstance(item, dict) and isinstance(item.get("allocations"), list)
        )
    return {
        "cargo_groups": len(groups) if isinstance(groups, list) else 0,
        "cargo_packages": len(packages) if isinstance(packages, list) else 0,
        "allocation_groups": len(allocations) if isinstance(allocations, list) else 0,
        "allocation_rows": allocation_rows,
        "containers": len(containers) if isinstance(containers, list) else 0,
    }


def _paired_document_rows(
    previous: RunBundle, current: RunBundle, document_ids: Sequence[str]
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for document_id in document_ids:
        left = previous.predictions[document_id]
        right = current.predictions[document_id]
        left_counts = _counts_for_assessment(left)
        right_counts = _counts_for_assessment(right)
        current_core = _counts_for_assessment(
            right, exclude_fields=RELATION_SCAFFOLD_FIELDS
        )
        relation = _fact_counts(
            right.predicted_cargo_relation_facts, right.reference_cargo_relation_facts
        )
        category = _fact_counts(
            right.predicted_category_values, right.reference_category_values
        )
        predicted_tokens = Counter(item[-1] for item in right.predicted_category_values)
        reference_tokens = Counter(item[-1] for item in right.reference_category_values)
        category_tokens = _counter_metric(predicted_tokens, reference_tokens)
        left_document = previous.document_rows[document_id]
        right_document = current.document_rows[document_id]
        generated = right.generated_text
        structure = _relation_structure(right.reference_text)
        rows.append(
            {
                "document_id": document_id,
                "cohort": left_document["cohort"],
                "page_count": int(right_document["page_count"]),
                "previous_json_valid": int(left.json_valid),
                "current_json_valid": int(right.json_valid),
                "json_validity_transition": f"{int(left.json_valid)}->{int(right.json_valid)}",
                "previous_schema_valid": int(left.schema_valid),
                "current_schema_valid": int(right.schema_valid),
                "schema_validity_transition": f"{int(left.schema_valid)}->{int(right.schema_valid)}",
                "previous_exact_document": int(left.canonical_exact_match),
                "current_exact_document": int(right.canonical_exact_match),
                **_counts_row("previous", left_counts),
                **_counts_row("current", right_counts),
                **_counts_row("current_aligned_core", current_core),
                "document_f1_delta": right_counts.f1 - left_counts.f1,
                **_counts_row("relation", relation),
                "relation_exact": int(
                    right.predicted_cargo_relation_facts
                    == right.reference_cargo_relation_facts
                ),
                **_counts_row("category_identity", category),
                **_counts_row("category_token", category_tokens),
                "category_exact": int(
                    right.predicted_category_values == right.reference_category_values
                ),
                "previous_input_tokens": int(left_document["input_tokens"]),
                "current_input_tokens": int(right_document["input_tokens"]),
                "previous_target_tokens": int(left_document["target_tokens"]),
                "current_target_tokens": int(right_document["target_tokens"]),
                "previous_generated_tokens": int(left_document["generated_tokens"]),
                "current_generated_tokens": int(right_document["generated_tokens"]),
                "current_generation_cap_hit": int(right_document["likely_generation_cap"]),
                "current_generated_to_reference_character_ratio": _safe_divide(
                    len(generated), len(right.reference_text)
                ),
                "current_repeated_8gram_fraction": _repeated_ngram_fraction(generated),
                "current_maximum_character_run": _maximum_character_run(generated),
                **structure,
            }
        )
    return rows


def _aggregate_document_counts(
    rows: Sequence[Mapping[str, Any]], prefix: str, indices: Sequence[int]
) -> Counts:
    return Counts(
        true_positive=sum(int(rows[index][f"{prefix}_true_positive"]) for index in indices),
        predicted=sum(int(rows[index][f"{prefix}_predicted"]) for index in indices),
        reference=sum(int(rows[index][f"{prefix}_reference"]) for index in indices),
    )


def _percentile(values: Sequence[float], quantile: float) -> float:
    if not values:
        raise ValueError("percentile requires values")
    ordered = sorted(values)
    index = quantile * (len(ordered) - 1)
    lower = math.floor(index)
    upper = math.ceil(index)
    if lower == upper:
        return ordered[lower]
    weight = index - lower
    return ordered[lower] * (1 - weight) + ordered[upper] * weight


def _bootstrap_rows(paired_rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    randomizer = random.Random(BOOTSTRAP_SEED)
    document_count = len(paired_rows)
    sampled: dict[str, list[float]] = defaultdict(list)
    for _ in range(BOOTSTRAP_REPLICATES):
        indices = [randomizer.randrange(document_count) for _ in range(document_count)]
        old_full = _aggregate_document_counts(paired_rows, "previous", indices)
        new_full = _aggregate_document_counts(paired_rows, "current", indices)
        new_core = _aggregate_document_counts(paired_rows, "current_aligned_core", indices)
        sampled["full_micro_f1_delta"].append(new_full.f1 - old_full.f1)
        sampled["aligned_core_micro_f1_delta"].append(new_core.f1 - old_full.f1)
        sampled["macro_document_f1_delta"].append(
            statistics.fmean(float(paired_rows[index]["document_f1_delta"]) for index in indices)
        )
        sampled["json_validity_delta"].append(
            statistics.fmean(
                int(paired_rows[index]["current_json_valid"])
                - int(paired_rows[index]["previous_json_valid"])
                for index in indices
            )
        )
        sampled["schema_validity_delta"].append(
            statistics.fmean(
                int(paired_rows[index]["current_schema_valid"])
                - int(paired_rows[index]["previous_schema_valid"])
                for index in indices
            )
        )

    all_indices = list(range(document_count))
    old_full = _aggregate_document_counts(paired_rows, "previous", all_indices)
    new_full = _aggregate_document_counts(paired_rows, "current", all_indices)
    new_core = _aggregate_document_counts(paired_rows, "current_aligned_core", all_indices)
    points = {
        "full_micro_f1_delta": new_full.f1 - old_full.f1,
        "aligned_core_micro_f1_delta": new_core.f1 - old_full.f1,
        "macro_document_f1_delta": statistics.fmean(
            float(row["document_f1_delta"]) for row in paired_rows
        ),
        "json_validity_delta": statistics.fmean(
            int(row["current_json_valid"]) - int(row["previous_json_valid"])
            for row in paired_rows
        ),
        "schema_validity_delta": statistics.fmean(
            int(row["current_schema_valid"]) - int(row["previous_schema_valid"])
            for row in paired_rows
        ),
    }
    return [
        {
            "metric": name,
            "point_estimate": points[name],
            "ci95_lower": _percentile(values, 0.025),
            "ci95_upper": _percentile(values, 0.975),
            "bootstrap_probability_positive": statistics.fmean(value > 0 for value in values),
            "bootstrap_replicates": BOOTSTRAP_REPLICATES,
            "bootstrap_seed": BOOTSTRAP_SEED,
            "resampling_unit": "paired_validation_document",
        }
        for name, values in sampled.items()
    ]


def _relation_rows(current: RunBundle) -> list[dict[str, Any]]:
    training_facts: dict[str, set[tuple[str, tuple[str, ...]]]] = defaultdict(set)
    training_documents: dict[str, set[str]] = defaultdict(set)
    for record in current.records.values():
        if record.split != "train":
            continue
        relations, _ = _relation_explicit_facts(record.target["documentPatch"])
        for fact in relations:
            training_facts[fact[0]].add((record.document_id, fact))
            training_documents[fact[0]].add(record.document_id)

    predicted: dict[str, set[tuple[str, tuple[str, ...]]]] = defaultdict(set)
    reference: dict[str, set[tuple[str, tuple[str, ...]]]] = defaultdict(set)
    reference_documents: dict[str, set[str]] = defaultdict(set)
    for document_id, assessment in current.predictions.items():
        for fact in assessment.predicted_cargo_relation_facts:
            predicted[fact[0]].add((document_id, fact))
        for fact in assessment.reference_cargo_relation_facts:
            reference[fact[0]].add((document_id, fact))
            reference_documents[fact[0]].add(document_id)

    rows: list[dict[str, Any]] = []
    for kind in sorted(set(predicted) | set(reference)):
        counts = Counts(
            true_positive=len(predicted[kind] & reference[kind]),
            predicted=len(predicted[kind]),
            reference=len(reference[kind]),
        )
        rows.append(
            {
                "relation_fact": kind,
                "train_documents": len(training_documents[kind]),
                "train_facts": len(training_facts[kind]),
                "validation_documents": len(reference_documents[kind]),
                **_counts_row("validation", counts),
            }
        )
    return rows


def _coverage_rows(current: RunBundle) -> tuple[list[dict[str, Any]], Counter[tuple[str, str]]]:
    reference = Counter[str]()
    predicted = Counter[str]()
    token_true_positive = Counter[str]()
    identity_pairs: Counter[tuple[str, str]] = Counter()
    for assessment in current.predictions.values():
        predicted_map = {
            fact[1]: fact[2]
            for fact in assessment.predicted_cargo_relation_facts
            if fact[0] == "allocation_coverage"
        }
        reference_map = {
            fact[1]: fact[2]
            for fact in assessment.reference_cargo_relation_facts
            if fact[0] == "allocation_coverage"
        }
        predicted_counter = Counter(predicted_map.values())
        reference_counter = Counter(reference_map.values())
        predicted.update(predicted_counter)
        reference.update(reference_counter)
        token_true_positive.update(predicted_counter & reference_counter)
        for group_id, value in reference_map.items():
            identity_pairs[(value, predicted_map.get(group_id, "<absent>"))] += 1
    rows = []
    for value in sorted(set(reference) | set(predicted)):
        counts = Counts(
            true_positive=token_true_positive[value],
            predicted=predicted[value],
            reference=reference[value],
        )
        rows.append({"coverage": value, **_counts_row("token_multiset", counts)})
    return rows, identity_pairs


def _category_rows(
    current: RunBundle,
) -> tuple[list[dict[str, Any]], Counter[tuple[str, str]], Counts]:
    train_counts = Counter[str]()
    train_documents: dict[str, set[str]] = defaultdict(set)
    for record in current.records.values():
        if record.split != "train":
            continue
        _, categories = _relation_explicit_facts(record.target["documentPatch"])
        for fact in categories:
            token = fact[-1]
            train_counts[token] += 1
            train_documents[token].add(record.document_id)

    predicted = Counter[str]()
    reference = Counter[str]()
    true_positive = Counter[str]()
    validation_documents: dict[str, set[str]] = defaultdict(set)
    confusion: Counter[tuple[str, str]] = Counter()
    total = Counts(0, 0, 0)
    token_counts: list[Counts] = []
    for document_id, assessment in current.predictions.items():
        predicted_counter = Counter(fact[-1] for fact in assessment.predicted_category_values)
        reference_counter = Counter(fact[-1] for fact in assessment.reference_category_values)
        predicted.update(predicted_counter)
        reference.update(reference_counter)
        true_positive.update(predicted_counter & reference_counter)
        token_counts.append(_counter_metric(predicted_counter, reference_counter))
        for token in reference_counter:
            validation_documents[token].add(document_id)
        predicted_map = {
            (fact[1], fact[2]): fact[3] for fact in assessment.predicted_category_values
        }
        reference_map = {
            (fact[1], fact[2]): fact[3] for fact in assessment.reference_category_values
        }
        for identity, token in reference_map.items():
            if identity in predicted_map and predicted_map[identity] != token:
                confusion[(token, predicted_map[identity])] += 1
    total = _sum_counts(token_counts)
    rows = []
    for token in sorted(set(reference) | set(predicted), key=lambda item: (-reference[item], item)):
        counts = Counts(
            true_positive=true_positive[token],
            predicted=predicted[token],
            reference=reference[token],
        )
        rows.append(
            {
                "category_token": token,
                "train_occurrences": train_counts[token],
                "train_documents": len(train_documents[token]),
                "validation_documents": len(validation_documents[token]),
                "seen_in_training": int(train_counts[token] > 0),
                **_counts_row("validation_token_multiset", counts),
            }
        )
    return rows, confusion, total


def _values_for_paths(
    assessment: PredictionAssessment,
    paths: Sequence[str],
    *,
    predicted: bool,
) -> list[dict[str, Any]]:
    text = assessment.generated_text if predicted else assessment.reference_text
    patch = _parse_patch(text)
    if patch is None:
        return []
    selected = set(paths)
    rows: list[dict[str, Any]] = []
    for path, scalar_json, value in _flatten_ordered(patch):
        if _normalize_path(path) in selected:
            rows.append({"path": path, "scalar_json": scalar_json, "value": value})
    return rows


def _one_to_one_value_errors(
    assessment: PredictionAssessment,
    paths: Sequence[str],
    raw_text: str,
) -> list[dict[str, Any]]:
    """Classify concept-level misses without reusing a prediction.

    Exact indexed matches are removed first, followed by equal values at different
    indices. Remaining reference/prediction values are paired by descending textual
    similarity solely to measure error distance; the pairing never changes the strict
    production metric and unmatched values remain omissions/additions.
    """

    predicted = _values_for_paths(assessment, paths, predicted=True)
    reference = _values_for_paths(assessment, paths, predicted=False)
    used_predicted: set[int] = set()
    used_reference: set[int] = set()
    rows: list[dict[str, Any]] = []

    def append_pair(kind: str, reference_index: int, predicted_index: int) -> None:
        ref = reference[reference_index]
        pred = predicted[predicted_index]
        similarity = value_similarity(ref["value"], pred["value"])
        rows.append(
            {
                "error_kind": kind,
                "reference_path": ref["path"],
                "predicted_path": pred["path"],
                "reference_value": ref["scalar_json"],
                "predicted_value": pred["scalar_json"],
                "predicted_grounding": raw_grounding(pred["value"], raw_text),
                **similarity,
            }
        )
        used_reference.add(reference_index)
        used_predicted.add(predicted_index)

    for reference_index, ref in enumerate(reference):
        for predicted_index, pred in enumerate(predicted):
            if predicted_index in used_predicted:
                continue
            if (ref["path"], ref["scalar_json"]) == (pred["path"], pred["scalar_json"]):
                append_pair("strict_exact", reference_index, predicted_index)
                break

    for reference_index, ref in enumerate(reference):
        if reference_index in used_reference:
            continue
        for predicted_index, pred in enumerate(predicted):
            if predicted_index in used_predicted:
                continue
            if ref["scalar_json"] == pred["scalar_json"]:
                append_pair("correct_value_wrong_index", reference_index, predicted_index)
                break

    candidates: list[tuple[float, int, int]] = []
    for reference_index, ref in enumerate(reference):
        if reference_index in used_reference:
            continue
        for predicted_index, pred in enumerate(predicted):
            if predicted_index in used_predicted:
                continue
            similarity = value_similarity(ref["value"], pred["value"])
            score = max(
                float(similarity["sequence_similarity"]),
                float(similarity["token_f1"]),
            )
            candidates.append((score, reference_index, predicted_index))
    for _score, reference_index, predicted_index in sorted(candidates, reverse=True):
        if reference_index in used_reference or predicted_index in used_predicted:
            continue
        append_pair("wrong_value", reference_index, predicted_index)

    for reference_index, ref in enumerate(reference):
        if reference_index not in used_reference:
            rows.append(
                {
                    "error_kind": "omission",
                    "reference_path": ref["path"],
                    "predicted_path": "",
                    "reference_value": ref["scalar_json"],
                    "predicted_value": "",
                    "predicted_grounding": "not_applicable",
                    "space_normalized_exact": False,
                    "alnum_normalized_exact": False,
                    "sequence_similarity": 0.0,
                    "token_f1": 0.0,
                    "reference_contains_prediction": False,
                    "prediction_contains_reference": False,
                    "character_length_delta": None,
                    "numeric_signed_delta": None,
                    "numeric_absolute_delta": None,
                    "numeric_relative_absolute_error": None,
                }
            )
    for predicted_index, pred in enumerate(predicted):
        if predicted_index not in used_predicted:
            rows.append(
                {
                    "error_kind": "addition",
                    "reference_path": "",
                    "predicted_path": pred["path"],
                    "reference_value": "",
                    "predicted_value": pred["scalar_json"],
                    "predicted_grounding": raw_grounding(pred["value"], raw_text),
                    "space_normalized_exact": False,
                    "alnum_normalized_exact": False,
                    "sequence_similarity": 0.0,
                    "token_f1": 0.0,
                    "reference_contains_prediction": False,
                    "prediction_contains_reference": False,
                    "character_length_delta": None,
                    "numeric_signed_delta": None,
                    "numeric_absolute_delta": None,
                    "numeric_relative_absolute_error": None,
                }
            )
    return rows


def _error_distance_rows(
    bundle: RunBundle,
    concepts: Mapping[str, Sequence[str]],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for document_id, assessment in bundle.predictions.items():
        raw_text = bundle.records[document_id].raw_text
        for concept, paths in concepts.items():
            for row in _one_to_one_value_errors(assessment, paths, raw_text):
                rows.append(
                    {
                        "run": bundle.label,
                        "document_id": document_id,
                        "concept": concept,
                        **row,
                    }
                )
    return rows


def _error_summary_rows(rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, str], list[Mapping[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[(str(row["run"]), str(row["concept"]))].append(row)
    output: list[dict[str, Any]] = []
    for (run, concept), values in sorted(grouped.items()):
        counts = Counter(str(row["error_kind"]) for row in values)
        wrong_values = [row for row in values if row["error_kind"] == "wrong_value"]
        output.append(
            {
                "run": run,
                "concept": concept,
                "reference_values": sum(
                    counts[kind]
                    for kind in ("strict_exact", "correct_value_wrong_index", "wrong_value", "omission")
                ),
                "strict_exact": counts["strict_exact"],
                "correct_value_wrong_index": counts["correct_value_wrong_index"],
                "wrong_value": counts["wrong_value"],
                "omission": counts["omission"],
                "addition": counts["addition"],
                "wrong_value_grounded": sum(
                    row["predicted_grounding"] != "not_grounded" for row in wrong_values
                ),
                "wrong_value_mean_sequence_similarity": (
                    statistics.fmean(float(row["sequence_similarity"]) for row in wrong_values)
                    if wrong_values
                    else 0.0
                ),
                "wrong_value_mean_token_f1": (
                    statistics.fmean(float(row["token_f1"]) for row in wrong_values)
                    if wrong_values
                    else 0.0
                ),
            }
        )
    return output


def _schema_failure_rows(bundle: RunBundle) -> list[dict[str, Any]]:
    counts: Counter[tuple[str, str, str]] = Counter()
    documents: dict[tuple[str, str, str], set[str]] = defaultdict(set)
    for row in bundle.schema_errors:
        document_id = str(row["document_id"])
        failure_class = str(row["failure_class"])
        errors = row.get("validation_errors")
        if failure_class == "invalid_json" or not isinstance(errors, list) or not errors:
            key = (failure_class, "<whole-output>", failure_class)
            counts[key] += 1
            documents[key].add(document_id)
            continue
        for error in errors:
            location = re.sub(r"\.\d+(?=\.|$)", "[]", str(error["location"]))
            key = (failure_class, location, str(error["type"]))
            counts[key] += 1
            documents[key].add(document_id)
    return [
        {
            "run": bundle.label,
            "failure_class": key[0],
            "location": key[1],
            "error_type": key[2],
            "instances": count,
            "documents": len(documents[key]),
        }
        for key, count in sorted(counts.items(), key=lambda item: (-item[1], item[0]))
    ]


def _generation_failure_rows(bundle: RunBundle) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for document_id, assessment in bundle.predictions.items():
        document = bundle.document_rows[document_id]
        generated = assessment.generated_text
        if assessment.json_valid and assessment.schema_valid and not int(document["likely_generation_cap"]):
            continue
        rows.append(
            {
                "run": bundle.label,
                "document_id": document_id,
                "json_valid": int(assessment.json_valid),
                "schema_valid": int(assessment.schema_valid),
                "generation_cap_hit": int(document["likely_generation_cap"]),
                "generated_tokens": int(document["generated_tokens"]),
                "generated_characters": len(generated),
                "generated_to_reference_character_ratio": _safe_divide(
                    len(generated), len(assessment.reference_text)
                ),
                "repeated_8gram_fraction": _repeated_ngram_fraction(generated),
                "maximum_character_run": _maximum_character_run(generated),
            }
        )
    return rows


def _eval_comparison_rows(previous: RunBundle, current: RunBundle) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for bundle in (previous, current):
        for row in bundle.eval_rows:
            rows.append(
                {
                    "run": bundle.label,
                    "epoch": float(row["epoch"]),
                    "step": int(float(row["step"])),
                    "field_value_f1": float(row["eval_field_value_f1"]),
                    "precision": float(row["eval_field_value_precision"]),
                    "recall": float(row["eval_field_value_recall"]),
                    "json_valid": float(row["eval_json_valid"]),
                    "schema_valid": float(row["eval_schema_valid"]),
                    "eval_loss": float(row["eval_loss"]),
                    "eval_runtime_seconds": float(row["eval_runtime"]),
                    "generated_tokens_mean": float(row["eval_generated_tokens_mean"]),
                    "generated_tokens_max": int(float(row["eval_generated_tokens_max"])),
                    "eos_fraction": float(row["eval_generation_eos_reached_fraction"]),
                    "relation_f1": float(row.get("eval_cargo_relation_f1") or 0.0),
                    "category_f1": float(row.get("eval_category_value_f1") or 0.0),
                    "is_certified_base_model": int(float(row.get("eval_is_base_model") or 0.0)),
                }
            )
    return rows


def _runtime_rows(previous: RunBundle, current: RunBundle) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for bundle in (previous, current):
        train = cast(Mapping[str, Any], cast(Mapping[str, Any], bundle.manifest["metrics"])["train"])
        start = int(bundle.mlflow_run["start_time_ms"])
        end = int(bundle.mlflow_run["end_time_ms"])
        config = bundle.config
        optimization = cast(Mapping[str, Any], config["optimization"])
        evaluation = cast(Mapping[str, Any], config["evaluation"])
        phase = {
            str(item["phase"]): item
            for item in cast(Sequence[Mapping[str, Any]], bundle.summary["phase_system_metrics"])
        }
        rows.append(
            {
                "run": bundle.label,
                "wall_seconds": (end - start) / 1000,
                "train_runtime_seconds": float(train["train_runtime"]),
                "evaluation_window_seconds": float(bundle.summary["evaluation_window_seconds"]),
                "train_samples_per_second": float(train["train_samples_per_second"]),
                "train_steps_per_second": float(train["train_steps_per_second"]),
                "num_input_tokens_seen": int(train["num_input_tokens_seen"]),
                "input_tokens_per_train_second": int(train["num_input_tokens_seen"])
                / float(train["train_runtime"]),
                "train_micro_batch": int(optimization["per_device_train_batch_size"]),
                "gradient_accumulation_steps": int(optimization["gradient_accumulation_steps"]),
                "eval_batch": int(evaluation["per_device_batch_size"]),
                "max_source_length": int(cast(Mapping[str, Any], cast(Mapping[str, Any], config["dataset"])["preprocessing"])["max_source_length"]),
                "max_target_length": int(cast(Mapping[str, Any], cast(Mapping[str, Any], config["dataset"])["preprocessing"])["max_target_length"]),
                "generation_max_length": int(evaluation["generation_max_length"]),
                "non_eval_gpu_utilization_mean": float(phase["non_evaluation"]["gpu_0_utilization_percentage_mean"]),
                "non_eval_gpu_memory_mib_mean": float(phase["non_evaluation"]["gpu_0_memory_usage_megabytes_mean"]),
                "scheduled_eval_gpu_utilization_mean": float(phase["scheduled_evaluation"]["gpu_0_utilization_percentage_mean"]),
            }
        )
    return rows


def _counts_for_paths_in_documents(
    bundle: RunBundle,
    paths: Sequence[str],
    document_ids: Iterable[str],
) -> Counts:
    include = set(paths)
    return _sum_counts(
        _counts_for_assessment(bundle.predictions[document_id], include_fields=include)
        for document_id in document_ids
    )


def _stable_field_rows(
    previous: RunBundle,
    current: RunBundle,
    concepts: Sequence[FieldConcept],
    paired_rows: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    stable_ids = [
        str(row["document_id"])
        for row in paired_rows
        if int(row["previous_json_valid"]) and int(row["current_json_valid"])
    ]
    rows: list[dict[str, Any]] = []
    for concept in concepts:
        left = _counts_for_paths_in_documents(previous, concept.previous_paths, stable_ids)
        right = _counts_for_paths_in_documents(current, concept.current_paths, stable_ids)
        rows.append(
            {
                "concept": concept.name,
                "documents": len(stable_ids),
                **_counts_row("previous", left),
                **_counts_row("current", right),
                "f1_delta": right.f1 - left.f1,
            }
        )
    return rows


def _relation_complexity_rows(
    paired_rows: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    buckets: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for row in paired_rows:
        relations = int(row["cargo_groups"]) + int(row["cargo_packages"]) + int(
            row["allocation_rows"]
        )
        if relations == 0:
            bucket = "no relation facts"
        elif relations <= 4:
            bucket = "1-4 relation nodes"
        elif relations <= 8:
            bucket = "5-8 relation nodes"
        else:
            bucket = "9+ relation nodes"
        buckets[bucket].append(row)
    order = (
        "no relation facts",
        "1-4 relation nodes",
        "5-8 relation nodes",
        "9+ relation nodes",
    )
    output = []
    for bucket in order:
        values = buckets.get(bucket, [])
        if not values:
            continue
        output.append(
            {
                "complexity": bucket,
                "documents": len(values),
                "relation_f1": _sum_counts(
                    Counts(
                        int(row["relation_true_positive"]),
                        int(row["relation_predicted"]),
                        int(row["relation_reference"]),
                    )
                    for row in values
                ).f1,
                "relation_exact": statistics.fmean(int(row["relation_exact"]) for row in values),
                "document_f1": _sum_counts(
                    Counts(
                        int(row["current_true_positive"]),
                        int(row["current_predicted"]),
                        int(row["current_reference"]),
                    )
                    for row in values
                ).f1,
            }
        )
    return output


def _eval_by_run(
    rows: Sequence[Mapping[str, Any]], run: str, *, exclude_epoch_zero: bool = False
) -> list[Mapping[str, Any]]:
    return [
        row
        for row in rows
        if row["run"] == run and (not exclude_epoch_zero or float(row["epoch"]) > 0)
    ]


def _plot_path(output_dir: Path, paths: list[str], name: str) -> Path:
    relative = f"plots/{name}"
    paths.append(relative)
    return output_dir / relative


def _render_plots(
    output_dir: Path,
    previous: RunBundle,
    current: RunBundle,
    paired_rows: Sequence[Mapping[str, Any]],
    bootstrap_rows: Sequence[Mapping[str, Any]],
    field_rows: Sequence[Mapping[str, Any]],
    component_rows: Sequence[Mapping[str, Any]],
    stable_rows: Sequence[Mapping[str, Any]],
    relation_rows: Sequence[Mapping[str, Any]],
    coverage_confusion: Mapping[tuple[str, str], int],
    category_rows: Sequence[Mapping[str, Any]],
    category_confusion: Mapping[tuple[str, str], int],
    category_token_counts: Counts,
    error_summary_rows: Sequence[Mapping[str, Any]],
    schema_rows: Sequence[Mapping[str, Any]],
    generation_rows: Sequence[Mapping[str, Any]],
    eval_rows: Sequence[Mapping[str, Any]],
    runtime_rows: Sequence[Mapping[str, Any]],
    complexity_rows: Sequence[Mapping[str, Any]],
) -> list[str]:
    plot_dir = output_dir / "plots"
    plot_dir.mkdir(parents=True, exist_ok=False)
    paths: list[str] = []
    colors = [PALETTE[1], PALETTE[0]]

    final_previous = cast(Mapping[str, Any], cast(Mapping[str, Any], previous.manifest["metrics"])["validation_evaluation"])
    final_current = cast(Mapping[str, Any], cast(Mapping[str, Any], current.manifest["metrics"])["validation_evaluation"])
    _grouped_bar_chart(
        _plot_path(output_dir, paths, "01_headline_validation_metrics.png"),
        "End-to-end validation metrics",
        "Same 60 raw-OCR documents; exact indexed field/value scoring",
        ["precision", "recall", "F1", "JSON valid", "schema valid"],
        [
            (
                previous.label,
                [
                    float(final_previous["eval_field_value_precision"]),
                    float(final_previous["eval_field_value_recall"]),
                    float(final_previous["eval_field_value_f1"]),
                    float(final_previous["eval_json_valid"]),
                    float(final_previous["eval_schema_valid"]),
                ],
                colors[0],
            ),
            (
                current.label,
                [
                    float(final_current["eval_field_value_precision"]),
                    float(final_current["eval_field_value_recall"]),
                    float(final_current["eval_field_value_f1"]),
                    float(final_current["eval_json_valid"]),
                    float(final_current["eval_schema_valid"]),
                ],
                colors[1],
            ),
        ],
        "Rate",
        y_max=1.0,
    )

    previous_eval = _eval_by_run(eval_rows, previous.label, exclude_epoch_zero=True)
    current_eval = _eval_by_run(eval_rows, current.label, exclude_epoch_zero=True)
    _line_chart(
        _plot_path(output_dir, paths, "02_field_value_f1_trajectory.png"),
        "Generated field/value F1 through training",
        "Scheduled evaluations only; final prediction publication is reported separately",
        [
            (previous.label, [float(row["epoch"]) for row in previous_eval], [float(row["field_value_f1"]) for row in previous_eval], colors[0]),
            (current.label, [float(row["epoch"]) for row in current_eval], [float(row["field_value_f1"]) for row in current_eval], colors[1]),
        ],
        "Epoch",
        "Micro-F1",
        y_range=(0.6, 0.83),
    )
    _line_chart(
        _plot_path(output_dir, paths, "03_precision_recall_trajectory.png"),
        "Precision and recall trajectory",
        "The newer run remains precision-dominant throughout",
        [
            (f"{previous.label} precision", [float(row["epoch"]) for row in previous_eval], [float(row["precision"]) for row in previous_eval], PALETTE[1]),
            (f"{previous.label} recall", [float(row["epoch"]) for row in previous_eval], [float(row["recall"]) for row in previous_eval], PALETTE[3]),
            (f"{current.label} precision", [float(row["epoch"]) for row in current_eval], [float(row["precision"]) for row in current_eval], PALETTE[0]),
            (f"{current.label} recall", [float(row["epoch"]) for row in current_eval], [float(row["recall"]) for row in current_eval], PALETTE[2]),
        ],
        "Epoch",
        "Rate",
        y_range=(0.5, 0.95),
    )
    _line_chart(
        _plot_path(output_dir, paths, "04_validity_trajectory.png"),
        "Output-contract validity through training",
        "JSON and Pydantic schema validity on the same validation split",
        [
            (f"{previous.label} JSON", [float(row["epoch"]) for row in previous_eval], [float(row["json_valid"]) for row in previous_eval], PALETTE[1]),
            (f"{previous.label} schema", [float(row["epoch"]) for row in previous_eval], [float(row["schema_valid"]) for row in previous_eval], PALETTE[3]),
            (f"{current.label} JSON", [float(row["epoch"]) for row in current_eval], [float(row["json_valid"]) for row in current_eval], PALETTE[0]),
            (f"{current.label} schema", [float(row["epoch"]) for row in current_eval], [float(row["schema_valid"]) for row in current_eval], PALETTE[2]),
        ],
        "Epoch",
        "Valid fraction",
        y_range=(0.4, 1.0),
    )
    _line_chart(
        _plot_path(output_dir, paths, "05_eval_loss_divergence.png"),
        "Teacher-forced validation loss",
        "Loss and generated structured F1 select different checkpoints",
        [
            (previous.label, [float(row["epoch"]) for row in previous_eval], [float(row["eval_loss"]) for row in previous_eval], colors[0]),
            (current.label, [float(row["epoch"]) for row in current_eval], [float(row["eval_loss"]) for row in current_eval], colors[1]),
        ],
        "Epoch",
        "Evaluation loss",
    )
    _line_chart(
        _plot_path(output_dir, paths, "06_current_relation_and_category_trajectory.png"),
        "Relation and readable-category learning",
        "Metrics introduced by the relation-explicit v3 target",
        [
            ("relation fact F1", [float(row["epoch"]) for row in current_eval], [float(row["relation_f1"]) for row in current_eval], PALETTE[0]),
            ("category identity F1", [float(row["epoch"]) for row in current_eval], [float(row["category_f1"]) for row in current_eval], PALETTE[2]),
            ("overall field/value F1", [float(row["epoch"]) for row in current_eval], [float(row["field_value_f1"]) for row in current_eval], PALETTE[1]),
        ],
        "Epoch",
        "F1",
        y_range=(0.45, 0.83),
    )
    _line_chart(
        _plot_path(output_dir, paths, "07_generation_length_trajectory.png"),
        "Generated output length through training",
        "Longer v3 schema and relation scaffolding materially increase decoding cost",
        [
            (previous.label, [float(row["epoch"]) for row in previous_eval], [float(row["generated_tokens_mean"]) for row in previous_eval], colors[0]),
            (current.label, [float(row["epoch"]) for row in current_eval], [float(row["generated_tokens_mean"]) for row in current_eval], colors[1]),
        ],
        "Epoch",
        "Mean generated tokens",
    )
    _line_chart(
        _plot_path(output_dir, paths, "08_evaluation_runtime_trajectory.png"),
        "Generation-based evaluation runtime",
        "60 documents per evaluation; batch and sequence limits differ by run",
        [
            (previous.label, [float(row["epoch"]) for row in previous_eval], [float(row["eval_runtime_seconds"]) for row in previous_eval], colors[0]),
            (current.label, [float(row["epoch"]) for row in current_eval], [float(row["eval_runtime_seconds"]) for row in current_eval], colors[1]),
        ],
        "Epoch",
        "Seconds",
    )

    deltas = [float(row["document_f1_delta"]) for row in paired_rows]
    _histogram(
        _plot_path(output_dir, paths, "09_paired_document_f1_delta.png"),
        "Per-document F1 change",
        "Positive values favor relation-explicit v3; every point uses the same document and OCR",
        deltas,
        14,
        "Current minus previous document F1",
    )
    transition_groups: dict[str, tuple[list[float], list[float]]] = {}
    for transition in ("1->1", "0->1", "1->0", "0->0"):
        subset = [row for row in paired_rows if row["json_validity_transition"] == transition]
        if subset:
            transition_groups[transition] = (
                [float(row["previous_f1"]) for row in subset],
                [float(row["current_f1"]) for row in subset],
            )
    _scatter_chart(
        _plot_path(output_dir, paths, "10_paired_document_f1_scatter.png"),
        "Previous versus current document F1",
        "Color denotes JSON-validity transition; diagonal agreement is visually implicit",
        [
            (transition, values[0], values[1], PALETTE[index % len(PALETTE)])
            for index, (transition, values) in enumerate(transition_groups.items())
        ],
        "Previous document F1",
        "Current document F1",
        y_range=(0.0, 1.0),
    )
    transition_counts = Counter(str(row["json_validity_transition"]) for row in paired_rows)
    _horizontal_bar_chart(
        _plot_path(output_dir, paths, "11_json_validity_transitions.png"),
        "Paired JSON-validity transitions",
        "Most documents are stable-valid, but gains and regressions occur on different samples",
        ["valid->valid", "invalid->valid", "valid->invalid", "invalid->invalid"],
        [transition_counts[value] for value in ("1->1", "0->1", "1->0", "0->0")],
        x_label="Documents",
        colors=[PALETTE[2], PALETTE[0], PALETTE[4], PALETTE[3]],
        x_max=60,
    )

    plot_components = sorted(component_rows, key=lambda row: float(row["current_reference"]), reverse=True)
    _grouped_bar_chart(
        _plot_path(output_dir, paths, "12_component_f1_comparison.png"),
        "Aligned schema-component F1",
        "Only explicitly mapped concepts are compared across schema versions",
        [str(row["component"]) for row in plot_components],
        [
            (previous.label, [float(row["previous_f1"]) for row in plot_components], colors[0]),
            (current.label, [float(row["current_f1"]) for row in plot_components], colors[1]),
        ],
        "F1",
        y_max=1.0,
    )
    key_names = {
        "cargo.package.type",
        "cargo.package.quantity",
        "cargo.description",
        "cargo.marksAndNumbers[]",
        "cargo.additionalInformation[]",
        "cargo.allocation.containerNumber",
        "cargo.allocation.packageQuantity",
        "containers[].containerNumber",
    }
    key_fields = [row for row in field_rows if row["concept"] in key_names]
    _grouped_bar_chart(
        _plot_path(output_dir, paths, "13_cargo_field_f1_full_set.png"),
        "Cargo and container field F1 — full validation set",
        "Strict indexed scoring; category+fallback type is compared with the prior raw type target",
        [str(row["concept"]).replace("cargo.", "") for row in key_fields],
        [
            (previous.label, [float(row["previous_f1"]) for row in key_fields], colors[0]),
            (current.label, [float(row["current_f1"]) for row in key_fields], colors[1]),
        ],
        "F1",
        y_max=1.0,
    )
    key_stable = [row for row in stable_rows if row["concept"] in key_names]
    _grouped_bar_chart(
        _plot_path(output_dir, paths, "14_cargo_field_f1_stable_valid.png"),
        "Cargo and container F1 — 50 outputs valid in both runs",
        "This removes most gain/loss caused solely by syntax-validity churn",
        [str(row["concept"]).replace("cargo.", "") for row in key_stable],
        [
            (previous.label, [float(row["previous_f1"]) for row in key_stable], colors[0]),
            (current.label, [float(row["current_f1"]) for row in key_stable], colors[1]),
        ],
        "F1",
        y_max=1.0,
    )

    _grouped_bar_chart(
        _plot_path(output_dir, paths, "15_relation_fact_precision_recall_f1.png"),
        "Relation-fact performance by ontology edge",
        "Identity-aware facts; group/package linkage remains the hardest relation",
        [str(row["relation_fact"]) for row in relation_rows],
        [
            ("precision", [float(row["validation_precision"]) for row in relation_rows], PALETTE[0]),
            ("recall", [float(row["validation_recall"]) for row in relation_rows], PALETTE[2]),
            ("F1", [float(row["validation_f1"]) for row in relation_rows], PALETTE[1]),
        ],
        "Rate",
        y_max=1.0,
    )
    _grouped_bar_chart(
        _plot_path(output_dir, paths, "16_relation_complexity.png"),
        "Performance by cargo-relation complexity",
        "Complexity counts groups, packages, and allocation rows in the reference",
        [str(row["complexity"]) for row in complexity_rows],
        [
            ("relation F1", [float(row["relation_f1"]) for row in complexity_rows], PALETTE[0]),
            ("relation exact", [float(row["relation_exact"]) for row in complexity_rows], PALETTE[2]),
            ("document F1", [float(row["document_f1"]) for row in complexity_rows], PALETTE[1]),
        ],
        "Rate",
        y_max=1.0,
    )
    coverage_labels = sorted({key for pair in coverage_confusion for key in pair})
    _render_heatmap(
        _plot_path(output_dir, paths, "17_allocation_coverage_confusion.png"),
        "Allocation-coverage identity confusion",
        "Rows are reference policies; columns are predicted policies or absence",
        coverage_labels,
        coverage_labels,
        coverage_confusion,
    )

    category_plot = [row for row in category_rows if int(row["validation_token_multiset_reference"]) > 0]
    _horizontal_bar_chart(
        _plot_path(output_dir, paths, "18_package_category_recall.png"),
        "Readable package-category recall",
        f"Token-multiset F1 across all categories is {category_token_counts.f1:.3f}",
        [str(row["category_token"]) for row in category_plot],
        [float(row["validation_token_multiset_recall"]) for row in category_plot],
        x_label="Recall",
        colors=[PALETTE[index % len(PALETTE)] for index in range(len(category_plot))],
        x_max=1.0,
    )
    seen = [row for row in category_plot if int(row["seen_in_training"])]
    unseen = [row for row in category_plot if not int(row["seen_in_training"])]
    groups = []
    if seen:
        groups.append(("seen in training", [float(row["train_documents"]) for row in seen], [float(row["validation_token_multiset_recall"]) for row in seen], PALETTE[0]))
    if unseen:
        groups.append(("unseen in training", [0.0 for _ in unseen], [float(row["validation_token_multiset_recall"]) for row in unseen], PALETTE[4]))
    _scatter_chart(
        _plot_path(output_dir, paths, "19_category_support_vs_recall.png"),
        "Category training support versus recall",
        "Document support is strongly confounded by a few cap-loop documents",
        groups,
        "Training documents containing category",
        "Validation recall",
        y_range=(0.0, 1.0),
    )
    category_labels = sorted({key for pair in category_confusion for key in pair}) or ["no direct confusion"]
    _render_heatmap(
        _plot_path(output_dir, paths, "20_category_identity_confusion.png"),
        "Category substitutions on matched package identities",
        "Omissions are not shown; only direct category-to-category substitutions",
        category_labels,
        category_labels,
        category_confusion,
    )

    error_concepts = [
        "package_type",
        "package_quantity",
        "description",
        "marks",
        "additional_information",
    ]
    current_errors = {
        str(row["concept"]): row
        for row in error_summary_rows
        if row["run"] == current.label and row["concept"] in error_concepts
    }
    _grouped_bar_chart(
        _plot_path(output_dir, paths, "21_current_cargo_error_taxonomy.png"),
        "Current cargo error taxonomy",
        "One-to-one diagnostic matching; strict metric remains unchanged",
        [name.replace("_", " ") for name in error_concepts],
        [
            ("wrong index", [float(current_errors[name]["correct_value_wrong_index"]) for name in error_concepts], PALETTE[1]),
            ("wrong value", [float(current_errors[name]["wrong_value"]) for name in error_concepts], PALETTE[4]),
            ("omission", [float(current_errors[name]["omission"]) for name in error_concepts], PALETTE[2]),
            ("addition", [float(current_errors[name]["addition"]) for name in error_concepts], PALETTE[3]),
        ],
        "Values",
        y_max=max(1.0, max(float(current_errors[name][key]) for name in error_concepts for key in ("correct_value_wrong_index", "wrong_value", "omission", "addition")) * 1.15),
    )
    schema_top = sorted(
        [row for row in schema_rows if row["run"] == current.label],
        key=lambda row: int(row["instances"]),
        reverse=True,
    )[:12]
    _horizontal_bar_chart(
        _plot_path(output_dir, paths, "22_schema_failure_locations.png"),
        "Current output-contract failure locations",
        "Repeated locations reflect the same copied value appearing in several linked structures",
        [f"{row['location']} ({row['error_type']})" for row in schema_top],
        [int(row["instances"]) for row in schema_top],
        x_label="Validation error instances",
        colors=[PALETTE[index % len(PALETTE)] for index in range(len(schema_top))],
    )
    invalid = [row for row in generation_rows if row["run"] == current.label and not int(row["json_valid"])]
    _scatter_chart(
        _plot_path(output_dir, paths, "23_invalid_generation_repetition.png"),
        "Invalid generations: length versus repetition",
        "Three 4,096-token failures are obvious repetition/copy loops, not legitimate long targets",
        [
            (
                "cap hit" if cap else "below cap",
                [float(row["generated_to_reference_character_ratio"]) for row in invalid if int(row["generation_cap_hit"]) == cap],
                [float(row["repeated_8gram_fraction"]) for row in invalid if int(row["generation_cap_hit"]) == cap],
                PALETTE[4] if cap else PALETTE[0],
            )
            for cap in (1, 0)
            if any(int(row["generation_cap_hit"]) == cap for row in invalid)
        ],
        "Generated/reference character ratio",
        "Repeated 8-gram fraction",
        y_range=(0.0, 1.0),
    )

    _grouped_bar_chart(
        _plot_path(output_dir, paths, "24_runtime_comparison.png"),
        "Training and evaluation time",
        "The v3 experiment nearly doubled elapsed cost",
        ["training runtime (h)", "wall time (h)", "eval windows (h)"],
        [
            (previous.label, [float(runtime_rows[0]["train_runtime_seconds"]) / 3600, float(runtime_rows[0]["wall_seconds"]) / 3600, float(runtime_rows[0]["evaluation_window_seconds"]) / 3600], colors[0]),
            (current.label, [float(runtime_rows[1]["train_runtime_seconds"]) / 3600, float(runtime_rows[1]["wall_seconds"]) / 3600, float(runtime_rows[1]["evaluation_window_seconds"]) / 3600], colors[1]),
        ],
        "Hours",
        y_max=max(float(row["wall_seconds"]) for row in runtime_rows) / 3600 * 1.12,
    )
    _grouped_bar_chart(
        _plot_path(output_dir, paths, "25_training_throughput.png"),
        "Training throughput",
        "Longer sequences plus batch 1/GA24 reduce hardware efficiency",
        ["samples/s", "steps/s x10", "input tokens/s /1000"],
        [
            (previous.label, [float(runtime_rows[0]["train_samples_per_second"]), float(runtime_rows[0]["train_steps_per_second"]) * 10, float(runtime_rows[0]["input_tokens_per_train_second"]) / 1000], colors[0]),
            (current.label, [float(runtime_rows[1]["train_samples_per_second"]), float(runtime_rows[1]["train_steps_per_second"]) * 10, float(runtime_rows[1]["input_tokens_per_train_second"]) / 1000], colors[1]),
        ],
        "Normalized throughput",
        y_max=max(float(row["input_tokens_per_train_second"]) / 1000 for row in runtime_rows) * 1.15,
    )
    _grouped_bar_chart(
        _plot_path(output_dir, paths, "26_sequence_length_inflation.png"),
        "Sequence-length inflation",
        "Means over the identical 60 validation documents",
        ["input tokens", "target tokens", "generated tokens"],
        [
            (previous.label, [statistics.fmean(float(row["previous_input_tokens"]) for row in paired_rows), statistics.fmean(float(row["previous_target_tokens"]) for row in paired_rows), statistics.fmean(float(row["previous_generated_tokens"]) for row in paired_rows)], colors[0]),
            (current.label, [statistics.fmean(float(row["current_input_tokens"]) for row in paired_rows), statistics.fmean(float(row["current_target_tokens"]) for row in paired_rows), statistics.fmean(float(row["current_generated_tokens"]) for row in paired_rows)], colors[1]),
        ],
        "Tokens per document",
        y_max=max(statistics.fmean(float(row["current_input_tokens"]) for row in paired_rows), statistics.fmean(float(row["previous_input_tokens"]) for row in paired_rows)) * 1.12,
    )
    _grouped_bar_chart(
        _plot_path(output_dir, paths, "27_gpu_efficiency_and_memory.png"),
        "Non-evaluation GPU utilization and memory",
        "Memory pressure fell, but utilization and throughput also fell",
        ["GPU util /100", "mean memory /32GiB"],
        [
            (previous.label, [float(runtime_rows[0]["non_eval_gpu_utilization_mean"]) / 100, float(runtime_rows[0]["non_eval_gpu_memory_mib_mean"]) / 32768], colors[0]),
            (current.label, [float(runtime_rows[1]["non_eval_gpu_utilization_mean"]) / 100, float(runtime_rows[1]["non_eval_gpu_memory_mib_mean"]) / 32768], colors[1]),
        ],
        "Normalized fraction",
        y_max=1.0,
    )
    _horizontal_bar_chart(
        _plot_path(output_dir, paths, "28_bootstrap_probability_of_improvement.png"),
        "Paired-bootstrap probability that v3 improved",
        "10,000 document-level resamples; this is not a posterior probability",
        [str(row["metric"]) for row in bootstrap_rows],
        [float(row["bootstrap_probability_positive"]) for row in bootstrap_rows],
        x_label="Fraction of bootstrap replicates above zero",
        colors=[PALETTE[index % len(PALETTE)] for index in range(len(bootstrap_rows))],
        x_max=1.0,
    )

    final_epoch_rows = {float(row["epoch"]): row for row in current_eval}
    if 20.0 in final_epoch_rows and 25.0 in final_epoch_rows:
        _grouped_bar_chart(
            _plot_path(output_dir, paths, "29_epoch20_vs_epoch25_multiobjective.png"),
            "Current run: epoch 20 versus epoch 25",
            "Epoch 20 is the better holistic operating point despite 0.0011 less overall F1",
            ["overall F1", "relation F1", "category F1", "JSON valid", "schema valid", "EOS reached"],
            [
                ("epoch 20", [float(final_epoch_rows[20.0][key]) for key in ("field_value_f1", "relation_f1", "category_f1", "json_valid", "schema_valid", "eos_fraction")], PALETTE[0]),
                ("epoch 25", [float(final_epoch_rows[25.0][key]) for key in ("field_value_f1", "relation_f1", "category_f1", "json_valid", "schema_valid", "eos_fraction")], PALETTE[1]),
            ],
            "Rate",
            y_max=1.0,
        )

    target_previous = sum(int(row["previous_reference"]) for row in component_rows)
    target_current = sum(int(row["current_reference"]) for row in component_rows)
    _grouped_bar_chart(
        _plot_path(output_dir, paths, "30_target_and_prediction_leaf_counts.png"),
        "Target and prediction leaf counts",
        "v3 includes 431 relation-scaffolding reference leaves in addition to aligned content",
        ["all reference leaves", "all predicted leaves", "aligned mapped reference leaves"],
        [
            (previous.label, [sum(int(row["previous_reference"]) for row in paired_rows), sum(int(row["previous_predicted"]) for row in paired_rows), target_previous], colors[0]),
            (current.label, [sum(int(row["current_reference"]) for row in paired_rows), sum(int(row["current_predicted"]) for row in paired_rows), target_current], colors[1]),
        ],
        "Scalar leaves",
        y_max=max(sum(int(row["current_reference"]) for row in paired_rows), sum(int(row["previous_reference"]) for row in paired_rows)) * 1.12,
    )
    return paths


def _markdown_table(headers: Sequence[str], rows: Sequence[Sequence[Any]]) -> str:
    def text(value: Any) -> str:
        if isinstance(value, float):
            return f"{value:.4f}"
        return str(value).replace("|", "\\|")

    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join("---" for _ in headers) + " |",
    ]
    lines.extend("| " + " | ".join(text(value) for value in row) + " |" for row in rows)
    return "\n".join(lines)


def _report(
    output_dir: Path,
    previous: RunBundle,
    current: RunBundle,
    paired_rows: Sequence[Mapping[str, Any]],
    bootstrap_rows: Sequence[Mapping[str, Any]],
    field_rows: Sequence[Mapping[str, Any]],
    component_rows: Sequence[Mapping[str, Any]],
    stable_rows: Sequence[Mapping[str, Any]],
    relation_rows: Sequence[Mapping[str, Any]],
    coverage_rows: Sequence[Mapping[str, Any]],
    coverage_confusion: Mapping[tuple[str, str], int],
    category_rows: Sequence[Mapping[str, Any]],
    category_token_counts: Counts,
    error_summary_rows: Sequence[Mapping[str, Any]],
    schema_rows: Sequence[Mapping[str, Any]],
    generation_rows: Sequence[Mapping[str, Any]],
    eval_rows: Sequence[Mapping[str, Any]],
    runtime_rows: Sequence[Mapping[str, Any]],
    complexity_rows: Sequence[Mapping[str, Any]],
    plot_paths: Sequence[str],
) -> str:
    previous_final = cast(Mapping[str, Any], cast(Mapping[str, Any], previous.manifest["metrics"])["validation_evaluation"])
    current_final = cast(Mapping[str, Any], cast(Mapping[str, Any], current.manifest["metrics"])["validation_evaluation"])
    bootstrap = {str(row["metric"]): row for row in bootstrap_rows}
    full_ci = bootstrap["full_micro_f1_delta"]
    core_ci = bootstrap["aligned_core_micro_f1_delta"]
    all_indices = list(range(len(paired_rows)))
    previous_counts = _aggregate_document_counts(paired_rows, "previous", all_indices)
    current_counts = _aggregate_document_counts(paired_rows, "current", all_indices)
    current_core = _aggregate_document_counts(paired_rows, "current_aligned_core", all_indices)
    improved = sum(float(row["document_f1_delta"]) > 0 for row in paired_rows)
    worsened = sum(float(row["document_f1_delta"]) < 0 for row in paired_rows)
    unchanged = len(paired_rows) - improved - worsened
    stable_ids = {
        str(row["document_id"])
        for row in paired_rows
        if int(row["previous_json_valid"]) and int(row["current_json_valid"])
    }
    stable_previous = _sum_counts(
        _counts_for_assessment(previous.predictions[document_id]) for document_id in stable_ids
    )
    stable_current = _sum_counts(
        _counts_for_assessment(current.predictions[document_id]) for document_id in stable_ids
    )
    json_transitions = Counter(str(row["json_validity_transition"]) for row in paired_rows)
    schema_transitions = Counter(str(row["schema_validity_transition"]) for row in paired_rows)
    previous_reference = sum(int(row["previous_reference"]) for row in paired_rows)
    current_reference = sum(int(row["current_reference"]) for row in paired_rows)
    relation_scaffold_reference = sum(
        _counts_for_assessment(
            assessment, include_fields=set(RELATION_SCAFFOLD_FIELDS)
        ).reference
        for assessment in current.predictions.values()
    )
    relation_supported = [row for row in paired_rows if int(row["relation_reference"]) > 0]
    category_supported = [row for row in paired_rows if int(row["category_identity_reference"]) > 0]
    current_invalid = [
        row for row in generation_rows if row["run"] == current.label and not int(row["json_valid"])
    ]
    cap_loops = [
        row
        for row in current_invalid
        if int(row["generation_cap_hit"])
        and (
            float(row["repeated_8gram_fraction"]) >= 0.3
            or int(row["maximum_character_run"]) >= 32
        )
    ]
    current_schema = [row for row in schema_rows if row["run"] == current.label]
    current_eval = _eval_by_run(eval_rows, current.label, exclude_epoch_zero=True)
    previous_eval = _eval_by_run(eval_rows, previous.label, exclude_epoch_zero=True)
    current_epochs = {float(row["epoch"]): row for row in current_eval}
    previous_epochs = {float(row["epoch"]): row for row in previous_eval}
    epoch_rows = []
    for epoch in sorted(set(previous_epochs) | set(current_epochs)):
        left = previous_epochs.get(epoch)
        right = current_epochs.get(epoch)
        epoch_rows.append(
            [
                int(epoch),
                float(left["field_value_f1"]) if left else "—",
                float(right["field_value_f1"]) if right else "—",
                (float(right["field_value_f1"]) - float(left["field_value_f1"]))
                if left and right
                else "—",
                float(right["relation_f1"]) if right else "—",
                float(right["category_f1"]) if right else "—",
                float(right["json_valid"]) if right else "—",
                float(right["schema_valid"]) if right else "—",
            ]
        )

    key_names = (
        "cargo.package.type",
        "cargo.package.quantity",
        "cargo.description",
        "cargo.marksAndNumbers[]",
        "cargo.additionalInformation[]",
        "cargo.allocation.containerNumber",
        "cargo.allocation.packageQuantity",
        "containers[].containerNumber",
    )
    fields = {str(row["concept"]): row for row in field_rows}
    stable = {str(row["concept"]): row for row in stable_rows}
    field_table = [
        [
            name,
            float(fields[name]["previous_f1"]),
            float(fields[name]["current_f1"]),
            float(fields[name]["f1_delta"]),
            float(stable[name]["previous_f1"]),
            float(stable[name]["current_f1"]),
            float(stable[name]["f1_delta"]),
        ]
        for name in key_names
    ]
    relation_table = [
        [
            row["relation_fact"],
            row["train_documents"],
            row["train_facts"],
            row["validation_reference"],
            row["validation_precision"],
            row["validation_recall"],
            row["validation_f1"],
        ]
        for row in relation_rows
    ]
    category_table = [
        [
            row["category_token"],
            row["train_documents"],
            row["validation_token_multiset_reference"],
            row["validation_token_multiset_precision"],
            row["validation_token_multiset_recall"],
            row["validation_token_multiset_f1"],
        ]
        for row in category_rows
        if int(row["validation_token_multiset_reference"]) > 0
    ]
    current_error_summary = [
        row for row in error_summary_rows if row["run"] == current.label
    ]
    error_table = [
        [
            row["concept"],
            row["reference_values"],
            row["strict_exact"],
            row["correct_value_wrong_index"],
            row["wrong_value"],
            row["omission"],
            row["addition"],
            row["wrong_value_grounded"],
        ]
        for row in current_error_summary
    ]
    top_improvements = sorted(paired_rows, key=lambda row: float(row["document_f1_delta"]), reverse=True)[:8]
    top_regressions = sorted(paired_rows, key=lambda row: float(row["document_f1_delta"]))[:8]
    # Run directories are under artifacts/kie-training; resolve prompt paths against the project root.
    project_root = current.run_dir.parents[2]
    prompt_previous = project_root / str(cast(Mapping[str, Any], previous.config["prompt"])["path"])
    prompt_current = project_root / str(cast(Mapping[str, Any], current.config["prompt"])["path"])
    previous_runtime, current_runtime = runtime_rows
    run_slowdown = float(current_runtime["train_runtime_seconds"]) / float(previous_runtime["train_runtime_seconds"]) - 1
    eval_slowdown = float(current_runtime["evaluation_window_seconds"]) / float(previous_runtime["evaluation_window_seconds"]) - 1
    input_inflation = statistics.fmean(float(row["current_input_tokens"]) for row in paired_rows) / statistics.fmean(float(row["previous_input_tokens"]) for row in paired_rows) - 1
    target_inflation = statistics.fmean(float(row["current_target_tokens"]) for row in paired_rows) / statistics.fmean(float(row["previous_target_tokens"]) for row in paired_rows) - 1
    generated_inflation = statistics.fmean(float(row["current_generated_tokens"]) for row in paired_rows) / statistics.fmean(float(row["previous_generated_tokens"]) for row in paired_rows) - 1
    identity_wrong = int(cast(Mapping[str, Any], current.summary["prediction_identity"])["published_ids_incorrect"])
    epoch20 = current_epochs.get(20.0)
    epoch25 = current_epochs.get(25.0)
    epoch20_text = ""
    if epoch20 and epoch25:
        epoch20_text = (
            f"Epoch 20 is the better multi-objective checkpoint: it gives up only "
            f"`{float(epoch25['field_value_f1']) - float(epoch20['field_value_f1']):.4f}` overall F1, "
            f"while category F1 is `{float(epoch20['category_f1']) - float(epoch25['category_f1']):+.4f}`, "
            f"JSON/schema validity are each `{float(epoch20['json_valid']) - float(epoch25['json_valid']):+.4f}` / "
            f"`{float(epoch20['schema_valid']) - float(epoch25['schema_valid']):+.4f}`, mean generation is "
            f"`{float(epoch25['generated_tokens_mean']) - float(epoch20['generated_tokens_mean']):.1f}` tokens shorter, and "
            f"evaluation is `{float(epoch25['eval_runtime_seconds']) - float(epoch20['eval_runtime_seconds']):.1f}` seconds faster. "
            "The current single-metric selector chose epoch 25 because it optimizes only overall field/value F1."
        )
    lines = [
        f"# Paired KIE training-run audit: `{current.manifest['run_id']}` versus `{previous.manifest['run_id']}`",
        "",
        f"Generated at `{datetime.now(UTC).isoformat()}` from immutable completed-run artifacts. No model weights were loaded, no GPU inference was performed, and the GLM-OCR service was not used by this analysis.",
        "",
        "## Executive conclusion",
        "",
        f"The new relation-explicit/category experiment is **not a statistically demonstrated end-to-end improvement**. Exact field/value micro-F1 moved from `{previous_counts.f1:.4f}` to `{current_counts.f1:.4f}` (`{current_counts.f1 - previous_counts.f1:+.4f}`), but the paired 10,000-resample 95% interval is `{float(full_ci['ci95_lower']):+.4f}` to `{float(full_ci['ci95_upper']):+.4f}` and includes zero. On the aligned non-scaffolding core, F1 is `{current_core.f1:.4f}` (`{current_core.f1 - previous_counts.f1:+.4f}` versus v2), with interval `{float(core_ci['ci95_lower']):+.4f}` to `{float(core_ci['ci95_upper']):+.4f}`. This remains well below the `0.90` reliability target.",
        "",
        f"The intervention is nevertheless informative rather than a failure. The readable package-category representation is a clear win: full-set package-type F1 rises from `{float(fields['cargo.package.type']['previous_f1']):.3f}` to `{float(fields['cargo.package.type']['current_f1']):.3f}`, and from `{float(stable['cargo.package.type']['previous_f1']):.3f}` to `{float(stable['cargo.package.type']['current_f1']):.3f}` on the {len(stable_ids)} documents whose outputs are JSON-valid in both runs. Package quantity also improves. In contrast, description/marks are essentially flat after controlling for validity, additional information regresses to zero, container identifiers regress slightly, and the new identity-aware relation facts reach only `{float(current_final['eval_cargo_relation_f1']):.3f}` F1.",
        "",
        f"The newer target is also materially more expensive: reference leaves increase from `{previous_reference:,}` to `{current_reference:,}` (`{current_reference / previous_reference - 1:+.1%}`), including `{relation_scaffold_reference:,}` relation-scaffolding leaves; mean validation input/target/generated lengths rise `{input_inflation:+.1%}` / `{target_inflation:+.1%}` / `{generated_inflation:+.1%}`; training runtime rises `{run_slowdown:+.1%}` and evaluation-window time `{eval_slowdown:+.1%}`. This cost bought a targeted category improvement, not a broad lift.",
        "",
        f"Output validity remains a hard gate. Current predictions contain `{len(current_invalid)}` invalid JSON outputs; `{len(cap_loops)}` are 4,096-token repetition/copy loops. JSON-valid but schema-invalid outputs are dominated by copied container identifiers that fail ISO 6346 validation, allocation-total mismatches, and one HS-code pattern error. These failures explain much of the document-level churn and the apparent container regression.",
        "",
        "## Audit scope and comparability",
        "",
        f"- Both runs use exactly the same `{len(paired_rows)}` validation document IDs, in the same order.",
        "- Each paired raw OCR string and SHA-256 digest is identical. Differences therefore come from the task/schema/prompt/training setup, not a changed evaluation corpus.",
        f"- The prior v2 target has `{previous_reference:,}` scalar reference leaves; v3 has `{current_reference:,}`. Raw micro-F1 is therefore not a perfectly identical ontology metric. The aligned-core diagnostic removes the known relation-scaffolding paths but does not pretend the schemas are otherwise identical.",
        f"- The v3 dataset carries an auxiliary table view, but config input field is `{cast(Mapping[str, Any], cast(Mapping[str, Any], current.config['dataset'])['fields'])['input_text']}` and the template does not render the auxiliary view. This run tests semantic instructions, readable categories, and relation-explicit labels—not table-augmented input.",
        "- The current on-start row explicitly records `eval_is_base_model=1`; it is a certified adapter-disabled base-model evaluation. It produced zero F1/validity and hit 4,096 tokens on all outputs. The previous run lacks that explicit marker, so its epoch-zero row is not used as certified base-model evidence.",
        f"- `{identity_wrong}/{len(paired_rows)}` current published prediction rows have the wrong `document_id` because prediction outputs were zipped against original dataset order after length-grouped sampling. Generated/reference pairing and aggregate metrics are correct; all identities were deterministically recovered from unique canonical references. The publication bug remains in runtime code and must be fixed separately before future run artifacts are trusted by ID.",
        "",
        "## Headline paired results",
        "",
        _markdown_table(
            ["Metric", "Previous", "Current", "Delta"],
            [
                ["Exact field/value precision", previous_counts.precision, current_counts.precision, current_counts.precision - previous_counts.precision],
                ["Exact field/value recall", previous_counts.recall, current_counts.recall, current_counts.recall - previous_counts.recall],
                ["Exact field/value F1", previous_counts.f1, current_counts.f1, current_counts.f1 - previous_counts.f1],
                ["Aligned-core current F1", previous_counts.f1, current_core.f1, current_core.f1 - previous_counts.f1],
                ["JSON validity", float(previous_final["eval_json_valid"]), float(current_final["eval_json_valid"]), float(current_final["eval_json_valid"]) - float(previous_final["eval_json_valid"])],
                ["Schema validity", float(previous_final["eval_schema_valid"]), float(current_final["eval_schema_valid"]), float(current_final["eval_schema_valid"]) - float(previous_final["eval_schema_valid"])],
                ["Whole-document exact", float(previous_final["eval_canonical_exact_match"]), float(current_final["eval_canonical_exact_match"]), float(current_final["eval_canonical_exact_match"]) - float(previous_final["eval_canonical_exact_match"])],
            ],
        ),
        "",
        f"At document level, `{improved}` samples improve, `{worsened}` worsen, and `{unchanged}` are unchanged. The median F1 delta is `{statistics.median(float(row['document_f1_delta']) for row in paired_rows):+.4f}`. Among the `{len(stable_ids)}` outputs valid in both runs, pooled F1 changes only `{stable_current.f1 - stable_previous.f1:+.4f}`. Thus much of the positive macro shift is explained by six invalid→valid documents, partly offset by four valid→invalid regressions, rather than consistent improvement across stable outputs.",
        "",
        _markdown_table(
            ["Transition", "JSON documents", "Schema documents"],
            [
                ["valid -> valid", json_transitions["1->1"], schema_transitions["1->1"]],
                ["invalid -> valid", json_transitions["0->1"], schema_transitions["0->1"]],
                ["valid -> invalid", json_transitions["1->0"], schema_transitions["1->0"]],
                ["invalid -> invalid", json_transitions["0->0"], schema_transitions["0->0"]],
            ],
        ),
        "",
        "### Paired bootstrap",
        "",
        _markdown_table(
            ["Metric", "Point delta", "95% lower", "95% upper", "P(delta > 0)"],
            [[row["metric"], row["point_estimate"], row["ci95_lower"], row["ci95_upper"], row["bootstrap_probability_positive"]] for row in bootstrap_rows],
        ),
        "",
        "The probability column is the fraction of paired bootstrap resamples above zero, not a Bayesian posterior. Every interval uses the document as the resampling unit, preserving within-document leaf dependence.",
        "",
        "## Training progression and checkpoint choice",
        "",
        _markdown_table(
            ["Epoch", "v2 F1", "v3 F1", "Delta", "v3 relation F1", "v3 category F1", "v3 JSON", "v3 schema"],
            epoch_rows,
        ),
        "",
        f"The v3 run learns more slowly: it trails v2 by `{float(current_epochs[5.0]['field_value_f1']) - float(previous_epochs[5.0]['field_value_f1']):+.4f}` at epoch 5, overtakes it at epoch 10, and finishes only `{float(current_epochs[25.0]['field_value_f1']) - float(previous_epochs[25.0]['field_value_f1']):+.4f}` ahead at the scheduled epoch-25 evaluation. Its relation F1 peaks at epoch 15 (`{max(float(row['relation_f1']) for row in current_eval):.4f}`), category F1 peaks at epoch 20 (`{max(float(row['category_f1']) for row in current_eval):.4f}`), and overall F1 nearly saturates by epoch 20.",
        "",
        epoch20_text,
        "",
        f"Teacher-forced eval loss is lowest at epoch `{int(min(current_eval, key=lambda row: float(row['eval_loss']))['epoch'])}` (`{min(float(row['eval_loss']) for row in current_eval):.4f}`), but generated structured F1 continues rising. This confirms the earlier diagnosis: cross-entropy loss is not an adequate model-selection metric for exact JSON extraction.",
        "",
        "## Schema intervention: what improved and what did not",
        "",
        _markdown_table(
            ["Aligned concept", "v2 full F1", "v3 full F1", "Delta", "v2 stable F1", "v3 stable F1", "Stable delta"],
            field_table,
        ),
        "",
        "The package-category change is the strongest causal lead. It converts a carrier-specific/verbatim surface target into readable semantic tokens while retaining a verbatim fallback. Token-only category F1 is higher than identity-aware category F1, which means choosing the category token is easier than attaching it to the correct package node. This validates readable categorical supervision but also confirms that package identity/linking remains the bottleneck.",
        "",
        "Package quantities also improve, including on stable-valid outputs. Cargo description changes little once validity churn is removed; marks remain weak; additional information is completely missed despite substantial train support. The semantic prompt is still underspecified about positive inclusion boundaries for `marksAndNumbers` versus `additionalInformation`: it says what boilerplate to exclude but not enough about what belongs in each target when present.",
        "",
        "Container-number regression is concentrated in the four new invalid JSON loops and several near-copy/check-digit schema failures. On stable-valid outputs it is much smaller, so the result is not evidence that the model broadly forgot container reading—but it is evidence that the more complex target increases failure amplification because a single identifier is repeated in containers and allocation structures.",
        "",
        "## Relation-explicit cargo ontology",
        "",
        f"Identity-aware cargo relation F1 is `{float(current_final['eval_cargo_relation_f1']):.4f}` (precision `{float(current_final['eval_cargo_relation_precision']):.4f}`, recall `{float(current_final['eval_cargo_relation_recall']):.4f}`). Only `{sum(int(row['relation_exact']) for row in relation_supported)}/{len(relation_supported)}` supported documents reproduce all relation facts exactly. Of the `{len(paired_rows) - len(relation_supported)}` documents with no reference relation facts, only `{sum(int(row['relation_exact']) for row in paired_rows if not int(row['relation_reference']))}` correctly stay empty.",
        "",
        _markdown_table(
            ["Relation fact", "Train docs", "Train facts", "Val refs", "Precision", "Recall", "F1"],
            relation_table,
        ),
        "",
        "`container_has_package` is the hardest edge because it requires the model to preserve two identities and their link simultaneously. Allocation-coverage errors are dominated by omission: one-to-one is the most common policy but many reference groups are absent rather than mapped to a wrong policy. The explicit ontology reduces list-index ambiguity, but it replaces some wrong-index errors with missing nodes/edges; it has not yet solved hierarchy reconstruction.",
        "",
        _markdown_table(
            ["Coverage token", "Reference", "Predicted", "Precision", "Recall", "F1"],
            [[row["coverage"], row["token_multiset_reference"], row["token_multiset_predicted"], row["token_multiset_precision"], row["token_multiset_recall"], row["token_multiset_f1"]] for row in coverage_rows],
        ),
        "",
        "## Readable categorical targets",
        "",
        f"Across package-category tokens, multiset F1 is `{category_token_counts.f1:.4f}` (precision `{category_token_counts.precision:.4f}`, recall `{category_token_counts.recall:.4f}`), versus `{float(current_final['eval_category_value_f1']):.4f}` identity-aware F1. Exact category identity sets are correct in `{sum(int(row['category_exact']) for row in category_supported)}/{len(category_supported)}` supported documents.",
        "",
        _markdown_table(
            ["Category", "Train docs", "Val refs", "Precision", "Recall", "F1"],
            category_table,
        ),
        "",
        "Common semantic tokens generalize well (`PACKAGE_PACKAGE`, `PALLET`, `CARTON`, `BAG`). Rare categories are unstable: `ROLL`, `DRUM_PLASTIC`, and `SACK` have low recall. `VEHICLE` is correctly inferred despite zero training occurrences, which is evidence that readable labels can exploit pretrained semantics. All eight `FLEXIBAG` validation values occur in one cap-loop document, so its zero recall cannot be attributed cleanly to categorical representation.",
        "",
        "## Error distance: how wrong are the cargo values?",
        "",
        _markdown_table(
            ["Concept", "Refs", "Exact", "Right value/wrong index", "Wrong value", "Omitted", "Added", "Grounded wrong values"],
            error_table,
        ),
        "",
        "The diagnostic performs one-to-one matching inside each document/concept and never reuses a prediction. Exact indexed matches are removed first, then equal values at a different list index, then remaining values are paired by maximum text similarity only to describe distance. It does not weaken or replace the strict metric.",
        "",
        "The dominant cargo failure is still hierarchy reconstruction, not free-form fabrication. Package type/quantity errors are largely wrong-node or omitted values; description misses are mostly shifts, omissions, or partial/contaminated copies. Most wrong values are visibly grounded in raw OCR. Marks/additional-information ambiguity is the notable exception because the model lacks a sufficiently sharp positive semantic boundary.",
        "",
        "## Output failures and schema validity",
        "",
        f"Current final publication has `{sum(not assessment.json_valid for assessment in current.predictions.values())}` invalid JSON outputs and `{sum(assessment.json_valid and not assessment.schema_valid for assessment in current.predictions.values())}` additional JSON-valid/schema-invalid outputs. `{len(cap_loops)}` invalid outputs are high-repetition cap loops, not legitimately long labels. The remaining malformed JSON output contains an ungrounded `EMAIL_PROFILE_18` fragment near an empty email value.",
        "",
        _markdown_table(
            ["Failure class", "Location", "Type", "Instances", "Documents"],
            [[row["failure_class"], row["location"], row["error_type"], row["instances"], row["documents"]] for row in current_schema[:16]],
        ),
        "",
        "A copied near-match container ID can produce multiple validator errors because it appears in the container list and again in allocation relations. This schema amplification is useful for rejecting invalid outputs, but the label topology makes a single generation error more costly. Allocation-total checks also catch relational inconsistency that leaf F1 alone would understate.",
        "",
        "## Runtime and resource cost",
        "",
        _markdown_table(
            ["Metric", previous.label, current.label, "Relative change"],
            [
                ["Training runtime (hours)", float(previous_runtime["train_runtime_seconds"]) / 3600, float(current_runtime["train_runtime_seconds"]) / 3600, run_slowdown],
                ["MLflow wall time (hours)", float(previous_runtime["wall_seconds"]) / 3600, float(current_runtime["wall_seconds"]) / 3600, float(current_runtime["wall_seconds"]) / float(previous_runtime["wall_seconds"]) - 1],
                ["Evaluation windows (hours)", float(previous_runtime["evaluation_window_seconds"]) / 3600, float(current_runtime["evaluation_window_seconds"]) / 3600, eval_slowdown],
                ["Train samples/second", previous_runtime["train_samples_per_second"], current_runtime["train_samples_per_second"], float(current_runtime["train_samples_per_second"]) / float(previous_runtime["train_samples_per_second"]) - 1],
                ["Input tokens/second", previous_runtime["input_tokens_per_train_second"], current_runtime["input_tokens_per_train_second"], float(current_runtime["input_tokens_per_train_second"]) / float(previous_runtime["input_tokens_per_train_second"]) - 1],
                ["Non-eval GPU utilization", float(previous_runtime["non_eval_gpu_utilization_mean"]) / 100, float(current_runtime["non_eval_gpu_utilization_mean"]) / 100, float(current_runtime["non_eval_gpu_utilization_mean"]) / float(previous_runtime["non_eval_gpu_utilization_mean"]) - 1],
            ],
        ),
        "",
        f"The prompt template grows from `{prompt_previous.stat().st_size:,}` to `{prompt_current.stat().st_size:,}` bytes before schema injection. Mean input tokens rise from `{statistics.fmean(float(row['previous_input_tokens']) for row in paired_rows):.1f}` to `{statistics.fmean(float(row['current_input_tokens']) for row in paired_rows):.1f}`. The current run also changed train micro-batch/accumulation from `{previous_runtime['train_micro_batch']}/{previous_runtime['gradient_accumulation_steps']}` to `{current_runtime['train_micro_batch']}/{current_runtime['gradient_accumulation_steps']}` and eval batch from `{previous_runtime['eval_batch']}` to `{current_runtime['eval_batch']}`. Peak memory pressure fell, but samples/second nearly halved and non-evaluation GPU utilization fell. The speed regression is therefore consistent with longer sequences plus smaller batches, not an unexplained stall.",
        "",
        "## Highest paired document changes",
        "",
        "### Improvements",
        "",
        _markdown_table(
            ["Document", "v2 F1", "v3 F1", "Delta", "JSON transition", "Relation F1"],
            [[row["document_id"], row["previous_f1"], row["current_f1"], row["document_f1_delta"], row["json_validity_transition"], row["relation_f1"]] for row in top_improvements],
        ),
        "",
        "### Regressions",
        "",
        _markdown_table(
            ["Document", "v2 F1", "v3 F1", "Delta", "JSON transition", "Relation F1"],
            [[row["document_id"], row["previous_f1"], row["current_f1"], row["document_f1_delta"], row["json_validity_transition"], row["relation_f1"]] for row in top_regressions],
        ),
        "",
        "## Decision-relevant findings",
        "",
        "1. Keep the 270M model and readable semantic categories: this experiment validates that direction. Do not revert package categories to opaque numeric or carrier-specific codes.",
        "2. Do not interpret the +0.003 headline F1 as proof that relation-explicit v3 broadly improved extraction. Its interval spans meaningful regression and improvement, and stable-valid content is nearly flat.",
        "3. Retain relation-explicit labels as an experimental auxiliary target, but address identity/link omission before scaling them unchanged. A compact two-stage or constrained relation decoder remains a strong next test.",
        "4. Add explicit positive semantics and examples for marks versus additional information. The current instruction improves exclusions but leaves inclusion ambiguous.",
        "5. Add repetition controls and/or structured decoding safeguards. Three cap failures are pathological loops; increasing max tokens would waste more time and worsen them.",
        "6. Select checkpoints with a multi-objective score or Pareto rule across overall F1, relation/category F1, validity, EOS completion, and runtime. For this run, epoch 20 is more attractive than epoch 25.",
        "7. Fix prediction identity publication before the next run. The deterministic recovery used here is safe only because all 60 canonical references are unique.",
        "8. The auxiliary table view was not tested in this run. It remains a clean next ablation once table OCR is available: same labels and prompt, with versus without a delimited table section.",
        "",
        "## Artifact map",
        "",
        f"- Current completed-run analysis: `{current.analysis_dir}`",
        f"- Previous completed-run analysis: `{previous.analysis_dir}`",
        "- `tables/paired_document_metrics.csv`: every paired document and validity transition.",
        "- `tables/field_comparison.csv`: all explicit cross-schema field correspondences.",
        "- `tables/error_distance.jsonl`: per-value exact/wrong-index/wrong-value/omission/addition diagnostics with raw-OCR grounding.",
        "- `tables/relation_facts.csv`, `category_values.csv`, and confusion tables: relation/category audit inputs.",
        "- `tables/bootstrap.csv`: deterministic paired resampling results.",
        "- `summary.json` and `analysis_manifest.json`: machine-readable conclusions and artifact hashes.",
        "",
        "## Plot gallery",
        "",
    ]
    for plot in plot_paths:
        title = Path(plot).stem.replace("_", " ").removeprefix("01 ").title()
        lines.extend([f"### {title}", "", f"![{title}]({plot})", ""])
    return "\n".join(lines).rstrip() + "\n"


def _publish_comparison(
    project_root: Path,
    previous_run_dir: Path,
    previous_analysis_dir: Path,
    current_run_dir: Path,
    current_analysis_dir: Path,
    output_dir: Path,
) -> None:
    if any(run_dir == output_dir or run_dir in output_dir.parents for run_dir in (previous_run_dir, current_run_dir)):
        raise ValueError("comparison output must be outside immutable training-run directories")
    output_dir.mkdir(parents=True, exist_ok=True)
    if any(output_dir.iterdir()):
        raise ValueError(f"comparison output directory is not empty: {output_dir}")
    previous = _load_bundle(project_root, "semantic-v2", previous_run_dir, previous_analysis_dir)
    current = _load_bundle(project_root, "relation-explicit-v3", current_run_dir, current_analysis_dir)
    document_ids = _assert_paired_validation(previous, current)
    concepts = _field_concepts(previous, current)
    paired_rows = _paired_document_rows(previous, current, document_ids)
    bootstrap_rows = _bootstrap_rows(paired_rows)
    field_rows = _field_comparison_rows(previous, current, concepts)
    component_rows = _component_rows(previous, current, concepts)
    stable_rows = _stable_field_rows(previous, current, concepts, paired_rows)
    relation_rows = _relation_rows(current)
    coverage_rows, coverage_confusion = _coverage_rows(current)
    category_rows, category_confusion, category_token_counts = _category_rows(current)
    complexity_rows = _relation_complexity_rows(paired_rows)
    previous_error_paths = {
        "package_type": ("$.documentPatch.goodsItems[].packages[].type",),
        "package_quantity": ("$.documentPatch.goodsItems[].packages[].quantity",),
        "description": ("$.documentPatch.goodsItems[].description",),
        "marks": ("$.documentPatch.goodsItems[].marksAndNumbers[]",),
        "additional_information": ("$.documentPatch.goodsItems[].additionalInformation[]",),
        "container_number": ("$.documentPatch.containers[].containerNumber",),
    }
    current_error_paths = {
        "package_type": (
            "$.documentPatch.cargoPackages[].typeCategory",
            "$.documentPatch.cargoPackages[].typeDescription",
        ),
        "package_quantity": ("$.documentPatch.cargoPackages[].quantity",),
        "description": ("$.documentPatch.cargoGroups[].description",),
        "marks": ("$.documentPatch.cargoGroups[].marksAndNumbers[]",),
        "additional_information": ("$.documentPatch.cargoGroups[].additionalInformation[]",),
        "container_number": ("$.documentPatch.containers[].containerNumber",),
    }
    error_rows = _error_distance_rows(previous, previous_error_paths) + _error_distance_rows(current, current_error_paths)
    error_summary_rows = _error_summary_rows(error_rows)
    schema_rows = _schema_failure_rows(previous) + _schema_failure_rows(current)
    generation_rows = _generation_failure_rows(previous) + _generation_failure_rows(current)
    eval_rows = _eval_comparison_rows(previous, current)
    runtime_rows = _runtime_rows(previous, current)
    table_dir = output_dir / "tables"
    _write_csv(table_dir / "paired_document_metrics.csv", paired_rows)
    _write_csv(table_dir / "bootstrap.csv", bootstrap_rows)
    _write_csv(table_dir / "field_comparison.csv", field_rows)
    _write_csv(table_dir / "component_comparison.csv", component_rows)
    _write_csv(table_dir / "stable_valid_field_comparison.csv", stable_rows)
    _write_csv(table_dir / "relation_facts.csv", relation_rows)
    _write_csv(table_dir / "allocation_coverage.csv", coverage_rows)
    _write_csv(table_dir / "category_values.csv", category_rows)
    _write_csv(table_dir / "relation_complexity.csv", complexity_rows)
    _write_jsonl(table_dir / "error_distance.jsonl", error_rows)
    _write_csv(table_dir / "error_distance_summary.csv", error_summary_rows)
    _write_csv(table_dir / "schema_failures.csv", schema_rows)
    _write_csv(table_dir / "generation_failures.csv", generation_rows)
    _write_csv(table_dir / "eval_history_comparison.csv", eval_rows)
    _write_csv(table_dir / "runtime_comparison.csv", runtime_rows)
    _write_csv(
        table_dir / "allocation_coverage_confusion.csv",
        [
            {"reference": key[0], "predicted": key[1], "documents": value}
            for key, value in sorted(coverage_confusion.items())
        ],
    )
    _write_csv(
        table_dir / "category_confusion.csv",
        [
            {"reference": key[0], "predicted": key[1], "instances": value}
            for key, value in sorted(category_confusion.items())
        ] or [{"reference": "<none>", "predicted": "<none>", "instances": 0}],
    )
    plot_paths = _render_plots(
        output_dir,
        previous,
        current,
        paired_rows,
        bootstrap_rows,
        field_rows,
        component_rows,
        stable_rows,
        relation_rows,
        coverage_confusion,
        category_rows,
        category_confusion,
        category_token_counts,
        error_summary_rows,
        schema_rows,
        generation_rows,
        eval_rows,
        runtime_rows,
        complexity_rows,
    )
    report = _report(
        output_dir,
        previous,
        current,
        paired_rows,
        bootstrap_rows,
        field_rows,
        component_rows,
        stable_rows,
        relation_rows,
        coverage_rows,
        coverage_confusion,
        category_rows,
        category_token_counts,
        error_summary_rows,
        schema_rows,
        generation_rows,
        eval_rows,
        runtime_rows,
        complexity_rows,
        plot_paths,
    )
    (output_dir / "REPORT.md").write_text(report, encoding="utf-8")
    all_indices = list(range(len(paired_rows)))
    previous_counts = _aggregate_document_counts(paired_rows, "previous", all_indices)
    current_counts = _aggregate_document_counts(paired_rows, "current", all_indices)
    current_core = _aggregate_document_counts(paired_rows, "current_aligned_core", all_indices)
    summary = {
        "schema_version": COMPARISON_SCHEMA_VERSION,
        "status": "complete",
        "created_at": datetime.now(UTC).isoformat(),
        "previous_run": previous.manifest["run_id"],
        "current_run": current.manifest["run_id"],
        "paired_validation_documents": len(document_ids),
        "raw_ocr_identity_verified": True,
        "previous_field_value": _counts_row("", previous_counts),
        "current_field_value": _counts_row("", current_counts),
        "current_aligned_core": _counts_row("", current_core),
        "full_micro_f1_delta": current_counts.f1 - previous_counts.f1,
        "aligned_core_micro_f1_delta": current_core.f1 - previous_counts.f1,
        "target_f1": TARGET_F1,
        "target_gap": TARGET_F1 - current_counts.f1,
        "bootstrap": bootstrap_rows,
        "category_token_multiset": _counts_row("", category_token_counts),
        "relation_supported_documents": sum(int(row["relation_reference"]) > 0 for row in paired_rows),
        "relation_exact_documents": sum(int(row["relation_exact"]) for row in paired_rows),
        "plots": plot_paths,
        "source_analysis_manifests_verified": True,
    }
    _write_json(output_dir / "summary.json", summary)
    artifacts = sorted(path for path in output_dir.rglob("*") if path.is_file())
    _write_json(
        output_dir / "analysis_manifest.json",
        {
            "schema_version": COMPARISON_SCHEMA_VERSION,
            "status": "complete",
            "created_at": datetime.now(UTC).isoformat(),
            "previous_run": previous.manifest["run_id"],
            "current_run": current.manifest["run_id"],
            "artifacts": [
                {
                    "path": str(path.relative_to(output_dir)),
                    "bytes": path.stat().st_size,
                    "sha256": _sha256(path),
                }
                for path in artifacts
            ],
        },
    )


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", required=True, type=Path)
    parser.add_argument("--previous-run-dir", required=True, type=Path)
    parser.add_argument("--previous-analysis-dir", required=True, type=Path)
    parser.add_argument("--current-run-dir", required=True, type=Path)
    parser.add_argument("--current-analysis-dir", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    _publish_comparison(
        args.project_root.resolve(),
        args.previous_run_dir.resolve(),
        args.previous_analysis_dir.resolve(),
        args.current_run_dir.resolve(),
        args.current_analysis_dir.resolve(),
        args.output_dir.resolve(),
    )
    print(
        json.dumps(
            {"status": "complete", "output_dir": str(args.output_dir.resolve())},
            sort_keys=True,
            separators=(",", ":"),
        )
    )


if __name__ == "__main__":
    main()
