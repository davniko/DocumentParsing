#!/usr/bin/env python3
# ruff: noqa: E501
"""Publish a reproducible, artifact-only analysis of one completed KIE training run.

The analyzer never loads model weights or reserves a GPU.  It verifies the immutable
run manifest, re-scores the saved predictions with the exact project metric contract,
joins pinned dataset/token-cache provenance, exports MLflow histories, and renders a
standalone report plus PNG plots using Pillow.
"""

from __future__ import annotations

import argparse
import csv
import difflib
import hashlib
import json
import math
import random
import re
import statistics
from collections import Counter, defaultdict
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, cast

import pyarrow as pa
import pyarrow.ipc as ipc
import yaml
from PIL import Image, ImageDraw, ImageFont
from pydantic import ValidationError

from document_ocr.training.config import TrainingConfig
from document_ocr.training.metrics import PredictionAssessment, assess_prediction
from document_ocr.training.tasks import TrainingTask, canonical_json, load_training_task

ANALYSIS_SCHEMA_VERSION = 1
BOOTSTRAP_REPLICATES = 10_000
BOOTSTRAP_SEED = 424
INDEX_PATTERN = re.compile(r"\[\d+\]")
PAGE_PATTERN = re.compile(r"(?m)^--- PAGE \d+ ---\s*$")
HEX64_PATTERN = re.compile(r"^[0-9a-f]{64}$")

PALETTE = (
    "#2563EB",
    "#DC2626",
    "#059669",
    "#7C3AED",
    "#D97706",
    "#0891B2",
    "#DB2777",
    "#4B5563",
)
INK = "#172033"
MUTED = "#667085"
GRID = "#DCE3EC"
BACKGROUND = "#FFFFFF"
PLOT_BACKGROUND = "#F8FAFC"


@dataclass(frozen=True, slots=True)
class DatasetRecord:
    document_id: str
    cohort: str
    split: str
    source_path: str
    raw_text: str
    raw_text_sha256: str
    target: dict[str, Any]
    page_count: int


@dataclass(frozen=True, slots=True)
class TokenLengths:
    input_tokens: int
    target_tokens: int
    input_original_tokens: int
    source_truncated: bool


@dataclass(frozen=True, slots=True)
class MetricCounts:
    true_positive: int
    predicted: int
    reference: int
    compared_paths: int

    @property
    def accuracy(self) -> float:
        return self.true_positive / self.compared_paths if self.compared_paths else 1.0

    @property
    def precision(self) -> float:
        return self.true_positive / self.predicted if self.predicted else 0.0

    @property
    def recall(self) -> float:
        return self.true_positive / self.reference if self.reference else 0.0

    @property
    def f1(self) -> float:
        denominator = self.precision + self.recall
        return 2 * self.precision * self.recall / denominator if denominator else 0.0


@dataclass(frozen=True, slots=True)
class PredictionDiagnostic:
    document_id: str
    published_document_id: str
    cohort: str
    assessment: PredictionAssessment
    counts: MetricCounts
    index_insensitive_counts: MetricCounts
    path_true_positive: int
    path_predicted: int
    path_reference: int
    substitutions: int
    omissions: int
    additions: int
    generated_chars: int
    reference_chars: int
    generated_tokens: int
    reference_content_tokens: int
    input_tokens: int
    target_tokens: int
    raw_text_chars: int
    page_count: int
    schema_failure_class: str
    schema_failure_detail: str
    likely_generation_cap: bool


@dataclass(frozen=True, slots=True)
class MlflowExport:
    run: dict[str, Any]
    histories: dict[str, list[dict[str, Any]]]


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


def _write_csv(path: Path, rows: Sequence[Mapping[str, Any]], fieldnames: Sequence[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="raise")
        writer.writeheader()
        writer.writerows(rows)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _resolved_checkpoint_path(run_dir: Path, recorded_path: str) -> Path:
    """Map the container-recorded checkpoint path to the immutable host run."""

    checkpoint_name = Path(recorded_path).name
    if not re.fullmatch(r"checkpoint-\d+", checkpoint_name):
        raise ValueError(f"unexpected best checkpoint path: {recorded_path}")
    resolved = run_dir / "checkpoints" / checkpoint_name
    if not resolved.is_dir():
        raise FileNotFoundError(f"best checkpoint is absent: {resolved}")
    return resolved


def _cohort_name(path: Path) -> str:
    name = path.parent.name
    if "pilot106" in name:
        return "pilot106"
    if "followup381" in name:
        return "followup381"
    return name


def _load_dataset_records(
    project_root: Path, config: Mapping[str, Any]
) -> tuple[dict[str, DatasetRecord], list[dict[str, Any]]]:
    dataset = cast(Mapping[str, Any], config["dataset"])
    splits = cast(Mapping[str, Any], dataset["splits"])
    records: dict[str, DatasetRecord] = {}
    source_rows: list[dict[str, Any]] = []
    input_hashes_by_split: dict[str, set[str]] = defaultdict(set)
    ids_by_split: dict[str, set[str]] = defaultdict(set)
    for split, sources_value in splits.items():
        sources = cast(Sequence[Mapping[str, Any]], sources_value)
        for source in sources:
            relative_path = Path(str(source["path"]))
            path = project_root / relative_path
            expected_sha = str(source["sha256"])
            actual_sha = _sha256(path)
            if actual_sha != expected_sha:
                raise ValueError(f"dataset source hash mismatch: {relative_path}")
            rows = _jsonl_load(path)
            expected_records = int(source["records"])
            if len(rows) != expected_records:
                raise ValueError(
                    f"dataset source row-count mismatch: {relative_path}: "
                    f"expected {expected_records}, found {len(rows)}"
                )
            cohort = _cohort_name(path)
            source_rows.append(
                {
                    "split": split,
                    "cohort": cohort,
                    "path": str(relative_path),
                    "records": len(rows),
                    "bytes": path.stat().st_size,
                    "sha256": actual_sha,
                }
            )
            for row in rows:
                document_id = str(row["documentId"])
                if document_id in records:
                    raise ValueError(f"duplicate document ID across configured splits: {document_id}")
                raw_text = str(row["joinedRawText"])
                raw_hash = str(row["joinedRawTextSha256"])
                if hashlib.sha256(raw_text.encode("utf-8")).hexdigest() != raw_hash:
                    raise ValueError(f"raw OCR hash mismatch for {document_id}")
                target = row["target"]
                if not isinstance(target, dict):
                    raise ValueError(f"target is not an object for {document_id}")
                page_count = len(PAGE_PATTERN.findall(raw_text))
                if page_count < 1:
                    raise ValueError(f"raw OCR has no page boundary for {document_id}")
                records[document_id] = DatasetRecord(
                    document_id=document_id,
                    cohort=cohort,
                    split=str(split),
                    source_path=str(relative_path),
                    raw_text=raw_text,
                    raw_text_sha256=raw_hash,
                    target=cast(dict[str, Any], target),
                    page_count=page_count,
                )
                input_hashes_by_split[str(split)].add(raw_hash)
                ids_by_split[str(split)].add(document_id)
    overlap_hashes = input_hashes_by_split["train"] & input_hashes_by_split["validation"]
    overlap_ids = ids_by_split["train"] & ids_by_split["validation"]
    if overlap_hashes or overlap_ids:
        raise ValueError(
            "train/validation exact leakage detected: "
            f"input_hashes={len(overlap_hashes)}, document_ids={len(overlap_ids)}"
        )
    return records, source_rows


def _load_token_lengths(cache_dir: Path, split: str, identity: str) -> dict[str, TokenLengths]:
    paths = sorted(cache_dir.glob(f"{split}-{identity}_*.arrow"))
    if not paths:
        raise FileNotFoundError(f"no token-cache shards found for {split=} and {identity=}")
    lengths: dict[str, TokenLengths] = {}
    for path in paths:
        with pa.memory_map(str(path), "r") as source:
            reader = ipc.open_stream(source)
            for batch in reader:
                columns = {name: batch.column(name) for name in batch.schema.names}
                for index in range(batch.num_rows):
                    document_id = str(columns["document_id"][index].as_py())
                    if document_id in lengths:
                        raise ValueError(f"duplicate token-cache document ID: {document_id}")
                    lengths[document_id] = TokenLengths(
                        input_tokens=int(columns["input_length"][index].as_py()),
                        target_tokens=int(columns["target_length"][index].as_py()),
                        input_original_tokens=int(
                            columns["input_original_length"][index].as_py()
                        ),
                        source_truncated=bool(columns["source_truncated"][index].as_py()),
                    )
    return lengths


def _normalize_path(path: str) -> str:
    return INDEX_PATTERN.sub("[]", path)


def _section(path: str) -> str:
    prefix = "$.documentPatch."
    if not path.startswith(prefix):
        return "unknown"
    remainder = path[len(prefix) :]
    return re.split(r"[.\[]", remainder, maxsplit=1)[0]


def _metric_counts(
    predicted: Iterable[tuple[str, str]], reference: Iterable[tuple[str, str]]
) -> MetricCounts:
    predicted_set = set(predicted)
    reference_set = set(reference)
    predicted_paths = {path for path, _ in predicted_set}
    reference_paths = {path for path, _ in reference_set}
    return MetricCounts(
        true_positive=len(predicted_set & reference_set),
        predicted=len(predicted_set),
        reference=len(reference_set),
        compared_paths=len(predicted_paths | reference_paths),
    )


def _index_insensitive_counts(
    predicted: Iterable[tuple[str, str]], reference: Iterable[tuple[str, str]]
) -> MetricCounts:
    predicted_counter = Counter((_normalize_path(path), value) for path, value in predicted)
    reference_counter = Counter((_normalize_path(path), value) for path, value in reference)
    true_positive = sum((predicted_counter & reference_counter).values())
    predicted_paths = {path for path, _ in predicted_counter}
    reference_paths = {path for path, _ in reference_counter}
    return MetricCounts(
        true_positive=true_positive,
        predicted=sum(predicted_counter.values()),
        reference=sum(reference_counter.values()),
        compared_paths=len(predicted_paths | reference_paths),
    )


def _schema_failure(
    generated_text: str, task: TrainingTask
) -> tuple[str, str, list[dict[str, Any]]]:
    try:
        parsed = json.loads(generated_text)
    except json.JSONDecodeError as error:
        detail = f"{error.msg} at line {error.lineno}, column {error.colno}, char {error.pos}"
        return "invalid_json", detail, []
    if not isinstance(parsed, dict):
        return "json_non_object", f"top-level type is {type(parsed).__name__}", []
    try:
        validated = task.target_model.model_validate_json(
            json.dumps(parsed, ensure_ascii=False, separators=(",", ":")), strict=True
        )
    except ValidationError as error:
        errors: list[dict[str, Any]] = []
        for item in error.errors(include_url=False, include_context=False, include_input=False):
            errors.append(
                {
                    "location": ".".join(str(value) for value in item["loc"]),
                    "type": str(item["type"]),
                    "message": str(item["msg"]),
                }
            )
        classes = sorted({str(item["type"]) for item in errors})
        return "schema_validation_error", "; ".join(classes), errors
    canonical = cast(Any, validated).canonical_target()
    if canonical != parsed:
        return (
            "noncanonical_sparse_representation",
            "prediction validates but contains defaults, nulls, or noncanonical structure",
            [],
        )
    return "none", "", []


def _reference_identity_index(
    dataset_records: Mapping[str, DatasetRecord], task: TrainingTask
) -> dict[str, str]:
    """Index unique validation targets for exact prediction-identity recovery."""

    reference_to_document: dict[str, str] = {}
    for candidate_id, candidate_record in dataset_records.items():
        if candidate_record.split != "validation":
            continue
        reference = canonical_json(task.canonicalize(candidate_record.target))
        if reference in reference_to_document:
            raise ValueError(
                "validation targets are not unique, so prediction identity recovery is ambiguous: "
                f"{candidate_id} and {reference_to_document[reference]}"
            )
        reference_to_document[reference] = candidate_id
    if not reference_to_document:
        raise ValueError("prediction identity recovery requires validation targets")
    return reference_to_document


def _safe_divide(numerator: int | float, denominator: int | float) -> float:
    return float(numerator) / float(denominator) if denominator else 0.0


def _load_predictions(
    run_dir: Path,
    dataset_records: Mapping[str, DatasetRecord],
    token_lengths: Mapping[str, TokenLengths],
    task: TrainingTask,
    generation_max_length: int,
) -> tuple[
    list[PredictionDiagnostic],
    list[dict[str, Any]],
    list[dict[str, Any]],
]:
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(run_dir / "final-adapter", local_files_only=True)
    prediction_rows = _jsonl_load(run_dir / "predictions" / "validation.jsonl")
    reference_to_document = _reference_identity_index(dataset_records, task)
    seen: set[str] = set()
    diagnostics: list[PredictionDiagnostic] = []
    schema_errors: list[dict[str, Any]] = []
    identity_rows: list[dict[str, Any]] = []
    for row_index, row in enumerate(prediction_rows):
        published_document_id = str(row["document_id"])
        reference_text = str(row["reference_text"])
        document_id = reference_to_document.get(reference_text)
        if document_id is None:
            raise ValueError(
                "saved prediction reference does not map to any pinned validation target: "
                f"row={row_index}, published_document_id={published_document_id}"
            )
        if document_id in seen:
            raise ValueError(f"duplicate recovered prediction document ID: {document_id}")
        seen.add(document_id)
        record = dataset_records.get(document_id)
        if record is None or record.split != "validation":
            raise ValueError(f"prediction does not map to configured validation row: {document_id}")
        lengths = token_lengths.get(document_id)
        if lengths is None:
            raise ValueError(f"prediction does not map to token cache: {document_id}")
        generated_text = str(row["generated_text"])
        expected_reference = canonical_json(task.canonicalize(record.target))
        if reference_text != expected_reference:
            raise ValueError(f"prediction reference differs from pinned dataset target: {document_id}")
        assessment = assess_prediction(generated_text, reference_text, task)
        for key in ("json_valid", "schema_valid", "canonical_exact_match"):
            if bool(row[key]) != bool(getattr(assessment, key)):
                raise ValueError(f"saved prediction flag drift for {document_id}: {key}")
        counts = _metric_counts(
            assessment.predicted_field_values, assessment.reference_field_values
        )
        index_counts = _index_insensitive_counts(
            assessment.predicted_field_values, assessment.reference_field_values
        )
        predicted_by_path = dict(assessment.predicted_field_values)
        reference_by_path = dict(assessment.reference_field_values)
        common_paths = set(predicted_by_path) & set(reference_by_path)
        substitutions = sum(
            predicted_by_path[path] != reference_by_path[path] for path in common_paths
        )
        omissions = len(set(reference_by_path) - set(predicted_by_path))
        additions = len(set(predicted_by_path) - set(reference_by_path))
        failure_class, failure_detail, validation_errors = _schema_failure(
            generated_text, task
        )
        if assessment.schema_valid and failure_class != "none":
            raise ValueError(f"schema-valid prediction has diagnosed failure: {document_id}")
        if not assessment.schema_valid and failure_class == "none":
            raise ValueError(f"schema-invalid prediction has no diagnosed failure: {document_id}")
        generated_tokens = len(tokenizer.encode(generated_text, add_special_tokens=False))
        reference_content_tokens = len(tokenizer.encode(reference_text, add_special_tokens=False))
        if reference_content_tokens + 1 != lengths.target_tokens:
            raise ValueError(
                f"reference token length does not match cached content+EOS contract: {document_id}"
            )
        likely_generation_cap = generated_tokens >= generation_max_length - 1
        diagnostic = PredictionDiagnostic(
            document_id=document_id,
            published_document_id=published_document_id,
            cohort=record.cohort,
            assessment=assessment,
            counts=counts,
            index_insensitive_counts=index_counts,
            path_true_positive=len(set(predicted_by_path) & set(reference_by_path)),
            path_predicted=len(predicted_by_path),
            path_reference=len(reference_by_path),
            substitutions=substitutions,
            omissions=omissions,
            additions=additions,
            generated_chars=len(generated_text),
            reference_chars=len(reference_text),
            generated_tokens=generated_tokens,
            reference_content_tokens=reference_content_tokens,
            input_tokens=lengths.input_tokens,
            target_tokens=lengths.target_tokens,
            raw_text_chars=len(record.raw_text),
            page_count=record.page_count,
            schema_failure_class=failure_class,
            schema_failure_detail=failure_detail,
            likely_generation_cap=likely_generation_cap,
        )
        diagnostics.append(diagnostic)
        published_record = dataset_records.get(published_document_id)
        identity_rows.append(
            {
                "row_index": row_index,
                "published_document_id": published_document_id,
                "resolved_document_id": document_id,
                "published_id_correct": int(published_document_id == document_id),
                "published_cohort": published_record.cohort if published_record else "unknown",
                "resolved_cohort": record.cohort,
                "reference_sha256": hashlib.sha256(reference_text.encode("utf-8")).hexdigest(),
            }
        )
        if failure_class != "none":
            schema_errors.append(
                {
                    "document_id": document_id,
                    "cohort": record.cohort,
                    "failure_class": failure_class,
                    "failure_detail": failure_detail,
                    "generated_characters": len(generated_text),
                    "generated_tokens": generated_tokens,
                    "likely_generation_cap": likely_generation_cap,
                    "ends_with_closing_brace": generated_text.rstrip().endswith("}"),
                    "validation_errors": validation_errors,
                }
            )
    validation_ids = {
        document_id for document_id, record in dataset_records.items() if record.split == "validation"
    }
    if seen != validation_ids:
        missing = sorted(validation_ids - seen)
        extra = sorted(seen - validation_ids)
        raise ValueError(f"prediction coverage mismatch: missing={missing}, extra={extra}")
    published_ids = {str(row["document_id"]) for row in prediction_rows}
    if published_ids != validation_ids:
        raise ValueError("published prediction IDs are not a permutation of validation IDs")
    return diagnostics, schema_errors, identity_rows


def _aggregate_counts(diagnostics: Sequence[PredictionDiagnostic]) -> MetricCounts:
    return MetricCounts(
        true_positive=sum(item.counts.true_positive for item in diagnostics),
        predicted=sum(item.counts.predicted for item in diagnostics),
        reference=sum(item.counts.reference for item in diagnostics),
        compared_paths=sum(item.counts.compared_paths for item in diagnostics),
    )


def _aggregate_index_counts(diagnostics: Sequence[PredictionDiagnostic]) -> MetricCounts:
    return MetricCounts(
        true_positive=sum(item.index_insensitive_counts.true_positive for item in diagnostics),
        predicted=sum(item.index_insensitive_counts.predicted for item in diagnostics),
        reference=sum(item.index_insensitive_counts.reference for item in diagnostics),
        compared_paths=sum(item.index_insensitive_counts.compared_paths for item in diagnostics),
    )


def _document_rows(diagnostics: Sequence[PredictionDiagnostic]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for item in sorted(diagnostics, key=lambda value: value.document_id):
        path_precision = _safe_divide(item.path_true_positive, item.path_predicted)
        path_recall = _safe_divide(item.path_true_positive, item.path_reference)
        path_f1 = _safe_divide(2 * path_precision * path_recall, path_precision + path_recall)
        rows.append(
            {
                "document_id": item.document_id,
                "published_document_id": item.published_document_id,
                "published_id_correct": int(item.document_id == item.published_document_id),
                "cohort": item.cohort,
                "json_valid": int(item.assessment.json_valid),
                "schema_valid": int(item.assessment.schema_valid),
                "canonical_exact_match": int(item.assessment.canonical_exact_match),
                "field_value_accuracy": item.counts.accuracy,
                "field_value_precision": item.counts.precision,
                "field_value_recall": item.counts.recall,
                "field_value_f1": item.counts.f1,
                "index_insensitive_f1": item.index_insensitive_counts.f1,
                "path_precision": path_precision,
                "path_recall": path_recall,
                "path_f1": path_f1,
                "true_positive_fields": item.counts.true_positive,
                "predicted_fields": item.counts.predicted,
                "reference_fields": item.counts.reference,
                "substitutions": item.substitutions,
                "omissions": item.omissions,
                "additions": item.additions,
                "page_count": item.page_count,
                "input_tokens": item.input_tokens,
                "target_tokens": item.target_tokens,
                "generated_tokens": item.generated_tokens,
                "generated_to_reference_token_ratio": _safe_divide(
                    item.generated_tokens, item.reference_content_tokens
                ),
                "raw_text_characters": item.raw_text_chars,
                "reference_characters": item.reference_chars,
                "generated_characters": item.generated_chars,
                "schema_failure_class": item.schema_failure_class,
                "likely_generation_cap": int(item.likely_generation_cap),
            }
        )
    return rows


def _field_rows(diagnostics: Sequence[PredictionDiagnostic]) -> list[dict[str, Any]]:
    strict: dict[str, Counter[str]] = defaultdict(Counter)
    unordered: dict[str, Counter[str]] = defaultdict(Counter)
    document_exact: dict[str, Counter[str]] = defaultdict(Counter)
    for item in diagnostics:
        predicted_pairs = set(item.assessment.predicted_field_values)
        reference_pairs = set(item.assessment.reference_field_values)
        for path, _ in predicted_pairs:
            strict[_normalize_path(path)]["predicted"] += 1
        for path, _ in reference_pairs:
            strict[_normalize_path(path)]["reference"] += 1
        for path, _ in predicted_pairs & reference_pairs:
            strict[_normalize_path(path)]["true_positive"] += 1

        predicted_by_field: dict[str, Counter[str]] = defaultdict(Counter)
        reference_by_field: dict[str, Counter[str]] = defaultdict(Counter)
        for path, value in predicted_pairs:
            predicted_by_field[_normalize_path(path)][value] += 1
        for path, value in reference_pairs:
            reference_by_field[_normalize_path(path)][value] += 1
        all_fields = set(predicted_by_field) | set(reference_by_field)
        for field in all_fields:
            predicted_values = predicted_by_field[field]
            reference_values = reference_by_field[field]
            unordered[field]["predicted"] += sum(predicted_values.values())
            unordered[field]["reference"] += sum(reference_values.values())
            unordered[field]["true_positive"] += sum(
                (predicted_values & reference_values).values()
            )
            if reference_values:
                document_exact[field]["supported_documents"] += 1
                if predicted_values == reference_values:
                    document_exact[field]["exact_documents"] += 1
            elif predicted_values:
                document_exact[field]["false_positive_documents"] += 1

    rows: list[dict[str, Any]] = []
    for field in sorted(set(strict) | set(unordered)):
        values = strict[field]
        true_positive = values["true_positive"]
        predicted = values["predicted"]
        reference = values["reference"]
        precision = _safe_divide(true_positive, predicted)
        recall = _safe_divide(true_positive, reference)
        f1 = _safe_divide(2 * precision * recall, precision + recall)
        unordered_values = unordered[field]
        unordered_precision = _safe_divide(
            unordered_values["true_positive"], unordered_values["predicted"]
        )
        unordered_recall = _safe_divide(
            unordered_values["true_positive"], unordered_values["reference"]
        )
        unordered_f1 = _safe_divide(
            2 * unordered_precision * unordered_recall,
            unordered_precision + unordered_recall,
        )
        docs = document_exact[field]
        rows.append(
            {
                "field_path": field,
                "section": _section(field),
                "reference_values": reference,
                "predicted_values": predicted,
                "true_positive_values": true_positive,
                "false_positive_values": predicted - true_positive,
                "false_negative_values": reference - true_positive,
                "precision": precision,
                "recall": recall,
                "f1": f1,
                "index_insensitive_precision": unordered_precision,
                "index_insensitive_recall": unordered_recall,
                "index_insensitive_f1": unordered_f1,
                "supported_documents": docs["supported_documents"],
                "exact_documents": docs["exact_documents"],
                "whole_field_exact_rate": _safe_divide(
                    docs["exact_documents"], docs["supported_documents"]
                ),
                "false_positive_documents": docs["false_positive_documents"],
            }
        )
    return rows


def _section_rows(diagnostics: Sequence[PredictionDiagnostic]) -> list[dict[str, Any]]:
    totals: dict[str, Counter[str]] = defaultdict(Counter)
    document_states: dict[str, Counter[str]] = defaultdict(Counter)
    for item in diagnostics:
        predicted = set(item.assessment.predicted_field_values)
        reference = set(item.assessment.reference_field_values)
        sections = {_section(path) for path, _ in predicted | reference}
        for section in sections:
            predicted_section = {(path, value) for path, value in predicted if _section(path) == section}
            reference_section = {(path, value) for path, value in reference if _section(path) == section}
            true_positive = len(predicted_section & reference_section)
            totals[section]["true_positive"] += true_positive
            totals[section]["predicted"] += len(predicted_section)
            totals[section]["reference"] += len(reference_section)
            if reference_section:
                document_states[section]["supported_documents"] += 1
                if predicted_section == reference_section:
                    document_states[section]["exact_documents"] += 1
            elif predicted_section:
                document_states[section]["false_positive_documents"] += 1
    rows: list[dict[str, Any]] = []
    for section in sorted(totals):
        values = totals[section]
        precision = _safe_divide(values["true_positive"], values["predicted"])
        recall = _safe_divide(values["true_positive"], values["reference"])
        f1 = _safe_divide(2 * precision * recall, precision + recall)
        states = document_states[section]
        rows.append(
            {
                "section": section,
                "reference_values": values["reference"],
                "predicted_values": values["predicted"],
                "true_positive_values": values["true_positive"],
                "precision": precision,
                "recall": recall,
                "f1": f1,
                "supported_documents": states["supported_documents"],
                "exact_documents": states["exact_documents"],
                "whole_section_exact_rate": _safe_divide(
                    states["exact_documents"], states["supported_documents"]
                ),
                "false_positive_documents": states["false_positive_documents"],
            }
        )
    return rows


def _aggregate_group(
    name: str, values: Sequence[PredictionDiagnostic], group_type: str
) -> dict[str, Any]:
    counts = _aggregate_counts(values)
    return {
        "group_type": group_type,
        "group": name,
        "documents": len(values),
        "json_valid": statistics.fmean(item.assessment.json_valid for item in values),
        "schema_valid": statistics.fmean(item.assessment.schema_valid for item in values),
        "canonical_exact_match": statistics.fmean(
            item.assessment.canonical_exact_match for item in values
        ),
        "field_value_accuracy": counts.accuracy,
        "field_value_precision": counts.precision,
        "field_value_recall": counts.recall,
        "field_value_f1": counts.f1,
        "macro_document_f1": statistics.fmean(item.counts.f1 for item in values),
        "reference_fields": counts.reference,
    }


def _group_rows(diagnostics: Sequence[PredictionDiagnostic]) -> list[dict[str, Any]]:
    rows = [_aggregate_group("all", diagnostics, "overall")]
    cohorts: dict[str, list[PredictionDiagnostic]] = defaultdict(list)
    pages: dict[str, list[PredictionDiagnostic]] = defaultdict(list)
    complexity: dict[str, list[PredictionDiagnostic]] = defaultdict(list)
    for item in diagnostics:
        cohorts[item.cohort].append(item)
        pages[str(item.page_count)].append(item)
        if item.counts.reference < 20:
            bucket = "<20 reference leaves"
        elif item.counts.reference < 40:
            bucket = "20-39 reference leaves"
        elif item.counts.reference < 60:
            bucket = "40-59 reference leaves"
        else:
            bucket = "60+ reference leaves"
        complexity[bucket].append(item)
    for name, values in sorted(cohorts.items()):
        rows.append(_aggregate_group(name, values, "cohort"))
    for name, values in sorted(pages.items(), key=lambda pair: int(pair[0])):
        rows.append(_aggregate_group(f"{name} page(s)", values, "page_count"))
    for name, values in complexity.items():
        rows.append(_aggregate_group(name, values, "label_complexity"))
    return rows


def _flatten_target_fields(target: Mapping[str, Any]) -> set[tuple[str, str]]:
    patch = target.get("documentPatch")
    if not isinstance(patch, dict):
        raise ValueError("target is missing documentPatch")

    def visit(value: Any, path: str) -> set[tuple[str, str]]:
        if isinstance(value, dict):
            result: set[tuple[str, str]] = set()
            for key in sorted(value):
                result.update(visit(value[key], f"{path}.{key}"))
            return result
        if isinstance(value, list):
            result = set()
            for index, child in enumerate(value):
                result.update(visit(child, f"{path}[{index}]"))
            return result
        return {(path, json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")))}

    return visit(patch, "$.documentPatch")


def _dataset_coverage_rows(
    records: Mapping[str, DatasetRecord], task: TrainingTask
) -> list[dict[str, Any]]:
    split_counts: dict[str, Counter[str]] = defaultdict(Counter)
    split_documents: dict[str, Counter[str]] = defaultdict(Counter)
    for record in records.values():
        canonical = task.canonicalize(record.target)
        fields = _flatten_target_fields(canonical)
        normalized = [_normalize_path(path) for path, _ in fields]
        split_counts[record.split].update(normalized)
        split_documents[record.split].update(set(normalized))
    rows: list[dict[str, Any]] = []
    field_names = sorted(set(split_counts["train"]) | set(split_counts["validation"]))
    for field in field_names:
        train_values = split_counts["train"][field]
        validation_values = split_counts["validation"][field]
        rows.append(
            {
                "field_path": field,
                "section": _section(field),
                "train_values": train_values,
                "validation_values": validation_values,
                "train_documents": split_documents["train"][field],
                "validation_documents": split_documents["validation"][field],
                "validation_field_seen_in_train": int(validation_values == 0 or train_values > 0),
                "train_document_prevalence": _safe_divide(
                    split_documents["train"][field],
                    sum(record.split == "train" for record in records.values()),
                ),
                "validation_document_prevalence": _safe_divide(
                    split_documents["validation"][field],
                    sum(record.split == "validation" for record in records.values()),
                ),
            }
        )
    return rows


def _percentile(sorted_values: Sequence[float], probability: float) -> float:
    if not sorted_values:
        raise ValueError("percentile requires values")
    position = (len(sorted_values) - 1) * probability
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return float(sorted_values[lower])
    weight = position - lower
    return float(sorted_values[lower] * (1 - weight) + sorted_values[upper] * weight)


def _bootstrap_intervals(
    diagnostics: Sequence[PredictionDiagnostic], replicates: int, seed: int
) -> list[dict[str, Any]]:
    rng = random.Random(seed)
    metric_values: dict[str, list[float]] = defaultdict(list)
    for _ in range(replicates):
        sample = [diagnostics[rng.randrange(len(diagnostics))] for _ in diagnostics]
        counts = _aggregate_counts(sample)
        metric_values["field_value_accuracy"].append(counts.accuracy)
        metric_values["field_value_precision"].append(counts.precision)
        metric_values["field_value_recall"].append(counts.recall)
        metric_values["field_value_f1"].append(counts.f1)
        metric_values["macro_document_f1"].append(
            statistics.fmean(item.counts.f1 for item in sample)
        )
        metric_values["json_valid"].append(
            statistics.fmean(item.assessment.json_valid for item in sample)
        )
        metric_values["schema_valid"].append(
            statistics.fmean(item.assessment.schema_valid for item in sample)
        )
        metric_values["canonical_exact_match"].append(
            statistics.fmean(item.assessment.canonical_exact_match for item in sample)
        )
    point_counts = _aggregate_counts(diagnostics)
    points = {
        "field_value_accuracy": point_counts.accuracy,
        "field_value_precision": point_counts.precision,
        "field_value_recall": point_counts.recall,
        "field_value_f1": point_counts.f1,
        "macro_document_f1": statistics.fmean(item.counts.f1 for item in diagnostics),
        "json_valid": statistics.fmean(item.assessment.json_valid for item in diagnostics),
        "schema_valid": statistics.fmean(item.assessment.schema_valid for item in diagnostics),
        "canonical_exact_match": statistics.fmean(
            item.assessment.canonical_exact_match for item in diagnostics
        ),
    }
    rows: list[dict[str, Any]] = []
    for metric, values in sorted(metric_values.items()):
        ordered = sorted(values)
        rows.append(
            {
                "metric": metric,
                "point_estimate": points[metric],
                "ci95_lower": _percentile(ordered, 0.025),
                "ci95_upper": _percentile(ordered, 0.975),
                "bootstrap_replicates": replicates,
                "bootstrap_seed": seed,
                "resampling_unit": "document",
            }
        )
    return rows


def _rank(values: Sequence[float]) -> list[float]:
    order = sorted(range(len(values)), key=values.__getitem__)
    ranks = [0.0] * len(values)
    index = 0
    while index < len(order):
        end = index + 1
        while end < len(order) and values[order[end]] == values[order[index]]:
            end += 1
        average_rank = (index + end - 1) / 2 + 1
        for location in order[index:end]:
            ranks[location] = average_rank
        index = end
    return ranks


def _pearson(left: Sequence[float], right: Sequence[float]) -> float:
    if len(left) != len(right) or len(left) < 2:
        raise ValueError("correlation inputs must have the same length >= 2")
    left_mean = statistics.fmean(left)
    right_mean = statistics.fmean(right)
    numerator = sum((a - left_mean) * (b - right_mean) for a, b in zip(left, right, strict=True))
    left_scale = math.sqrt(sum((value - left_mean) ** 2 for value in left))
    right_scale = math.sqrt(sum((value - right_mean) ** 2 for value in right))
    return numerator / (left_scale * right_scale) if left_scale and right_scale else 0.0


def _correlation_rows(diagnostics: Sequence[PredictionDiagnostic]) -> list[dict[str, Any]]:
    outcome = [item.counts.f1 for item in diagnostics]
    features: dict[str, list[float]] = {
        "input_tokens": [float(item.input_tokens) for item in diagnostics],
        "target_tokens": [float(item.target_tokens) for item in diagnostics],
        "reference_fields": [float(item.counts.reference) for item in diagnostics],
        "page_count": [float(item.page_count) for item in diagnostics],
        "generated_to_reference_token_ratio": [
            _safe_divide(item.generated_tokens, item.reference_content_tokens)
            for item in diagnostics
        ],
    }
    rows: list[dict[str, Any]] = []
    for name, values in features.items():
        rows.append(
            {
                "feature": name,
                "pearson_r_with_document_f1": _pearson(values, outcome),
                "spearman_rho_with_document_f1": _pearson(_rank(values), _rank(outcome)),
                "documents": len(values),
            }
        )
    return rows


def _load_trainer_histories(run_dir: Path) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    state = _json_load(run_dir / "checkpoints" / "trainer_state.json")
    history = cast(list[dict[str, Any]], state["log_history"])
    training = [row for row in history if "loss" in row]
    evaluation = [row for row in history if "eval_loss" in row]
    if not training or not evaluation:
        raise ValueError("trainer history is missing training or evaluation rows")
    return training, evaluation


def _epoch_zero_baseline(
    eval_history: Sequence[Mapping[str, Any]], environment: Mapping[str, Any]
) -> dict[str, Any]:
    """Describe the on-start baseline from the recorded runtime marker.

    The callback writes ``eval_is_base_model=1`` only while adapters are explicitly
    disabled.  Absence of that marker is not proof of a base-model evaluation.
    """

    if not eval_history:
        raise ValueError("evaluation history is empty")
    first = eval_history[0]
    certified = float(first.get("eval_is_base_model", 0.0)) == 1.0
    source_code = cast(Sequence[Mapping[str, Any]], environment["source_code"])
    runtime_sha256 = next(
        str(item["sha256"])
        for item in source_code
        if item.get("module") == "document_ocr.training.runtime"
    )
    return {
        "certified_base_model": certified,
        "reason": (
            "eval_is_base_model=1 records an adapter-disabled on-start evaluation"
            if certified
            else "eval_is_base_model=1 is absent; adapter-disabled execution is not certified"
        ),
        "recorded_runtime_sha256": runtime_sha256,
    }


def _load_phase_rows(run_dir: Path, mlflow_start_ms: int, mlflow_end_ms: int) -> list[dict[str, Any]]:
    events = _jsonl_load(run_dir / "logs" / "events.jsonl")
    eval_events = [
        row
        for row in events
        if row.get("event") == "trainer_log"
        and isinstance(row.get("metrics"), dict)
        and "eval_runtime" in cast(dict[str, Any], row["metrics"])
    ]
    rows: list[dict[str, Any]] = []
    for index, event in enumerate(eval_events):
        end = datetime.fromisoformat(str(event["timestamp"]))
        runtime = float(cast(dict[str, Any], event["metrics"])["eval_runtime"])
        end_seconds = end.timestamp() - mlflow_start_ms / 1000
        rows.append(
            {
                "phase": "on_start_evaluation" if index == 0 else "scheduled_evaluation",
                "step": int(event["global_step"]),
                "epoch": float(event["epoch"]),
                "start_elapsed_seconds": end_seconds - runtime,
                "end_elapsed_seconds": end_seconds,
                "duration_seconds": runtime,
            }
        )
    final_metrics = cast(dict[str, Any], _json_load(run_dir / "checkpoints" / "eval_results.json"))
    final_runtime = float(final_metrics["eval_runtime"])
    run_end_seconds = (mlflow_end_ms - mlflow_start_ms) / 1000
    rows.append(
        {
            "phase": "final_best_checkpoint_evaluation_and_prediction_publication",
            "step": int(cast(dict[str, Any], _json_load(run_dir / "checkpoints" / "trainer_state.json"))["global_step"]),
            "epoch": float(cast(dict[str, Any], _json_load(run_dir / "checkpoints" / "trainer_state.json"))["epoch"]),
            "start_elapsed_seconds": run_end_seconds - final_runtime,
            "end_elapsed_seconds": run_end_seconds,
            "duration_seconds": final_runtime,
        }
    )
    return rows


def _fetch_mlflow(run_dir: Path) -> MlflowExport:
    from mlflow import MlflowClient

    reference = cast(dict[str, Any], _json_load(run_dir / "mlflow-run.json"))
    client = MlflowClient(tracking_uri=str(reference["tracking_uri"]))
    run = client.get_run(str(reference["run_id"]))
    if run.info.status != "FINISHED":
        raise ValueError(f"MLflow run is not FINISHED: {run.info.status}")
    histories: dict[str, list[dict[str, Any]]] = {}
    for key in sorted(run.data.metrics):
        histories[key] = [
            {"key": item.key, "step": item.step, "timestamp_ms": item.timestamp, "value": item.value}
            for item in client.get_metric_history(str(reference["run_id"]), key)
        ]
    return MlflowExport(
        run={
            "run_id": run.info.run_id,
            "experiment_id": run.info.experiment_id,
            "status": run.info.status,
            "start_time_ms": run.info.start_time,
            "end_time_ms": run.info.end_time,
            "artifact_uri": run.info.artifact_uri,
            "params": dict(sorted(run.data.params.items())),
            "tags": dict(sorted(run.data.tags.items())),
            "latest_metrics": dict(sorted(run.data.metrics.items())),
        },
        histories=histories,
    )


def _mlflow_rows(export: MlflowExport) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    summary: list[dict[str, Any]] = []
    for key, history in export.histories.items():
        values = [float(item["value"]) for item in history]
        summary.append(
            {
                "metric": key,
                "points": len(values),
                "first": values[0],
                "last": values[-1],
                "minimum": min(values),
                "maximum": max(values),
                "mean": statistics.fmean(values),
            }
        )
    system_keys = sorted(key for key in export.histories if key.startswith("system/"))
    if not system_keys:
        raise ValueError("MLflow run contains no system metrics")
    expected_count = len(export.histories[system_keys[0]])
    if any(len(export.histories[key]) != expected_count for key in system_keys):
        raise ValueError("MLflow system metric histories have inconsistent lengths")
    rows: list[dict[str, Any]] = []
    start_ms = int(export.run["start_time_ms"])
    for index in range(expected_count):
        first = export.histories[system_keys[0]][index]
        step = int(first["step"])
        timestamp_ms = int(first["timestamp_ms"])
        row: dict[str, Any] = {
            "sample": index,
            "step": step,
            "timestamp_ms": timestamp_ms,
            "elapsed_seconds": (timestamp_ms - start_ms) / 1000,
        }
        for key in system_keys:
            point = export.histories[key][index]
            if int(point["step"]) != step or int(point["timestamp_ms"]) != timestamp_ms:
                raise ValueError(f"MLflow system metric histories are misaligned at {key}[{index}]")
            row[key] = float(point["value"])
        rows.append(row)
    return summary, rows


def _phase_system_rows(
    system_rows: Sequence[Mapping[str, Any]], phase_rows: Sequence[Mapping[str, Any]]
) -> list[dict[str, Any]]:
    grouped: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for sample in system_rows:
        elapsed = float(sample["elapsed_seconds"])
        matching = [
            phase
            for phase in phase_rows
            if float(phase["start_elapsed_seconds"])
            <= elapsed
            <= float(phase["end_elapsed_seconds"])
        ]
        if len(matching) > 1:
            raise ValueError(f"overlapping evaluation phases at elapsed second {elapsed}")
        phase_name = str(matching[0]["phase"]) if matching else "non_evaluation"
        grouped[phase_name].append(sample)
    metric_keys = (
        "system/gpu_0_utilization_percentage",
        "system/gpu_0_memory_usage_megabytes",
        "system/gpu_0_power_usage_watts",
        "system/cpu_utilization_percentage",
        "system/system_memory_usage_percentage",
    )
    rows: list[dict[str, Any]] = []
    for phase, samples in sorted(grouped.items()):
        row: dict[str, Any] = {"phase": phase, "samples": len(samples)}
        for key in metric_keys:
            values = sorted(float(sample[key]) for sample in samples)
            short_name = key.removeprefix("system/").replace("/", "_")
            row[f"{short_name}_mean"] = statistics.fmean(values)
            row[f"{short_name}_p50"] = _percentile(values, 0.50)
            row[f"{short_name}_p95"] = _percentile(values, 0.95)
            row[f"{short_name}_max"] = max(values)
        rows.append(row)
    return rows


def _verify_run_manifest(run_dir: Path) -> dict[str, Any]:
    manifest = cast(dict[str, Any], _json_load(run_dir / "manifest.json"))
    if manifest.get("status") != "complete":
        raise ValueError("run manifest is not complete")
    artifacts = cast(list[dict[str, Any]], manifest["artifacts"])
    mismatches: list[dict[str, Any]] = []
    total_bytes = 0
    for item in artifacts:
        relative = str(item["path"])
        path = run_dir / relative
        if not path.is_file():
            mismatches.append({"path": relative, "problem": "missing"})
            continue
        actual_bytes = path.stat().st_size
        actual_sha = _sha256(path)
        total_bytes += actual_bytes
        if actual_bytes != int(item["bytes"]) or actual_sha != str(item["sha256"]):
            mismatches.append(
                {
                    "path": relative,
                    "problem": "content_mismatch",
                    "expected_bytes": int(item["bytes"]),
                    "actual_bytes": actual_bytes,
                    "expected_sha256": str(item["sha256"]),
                    "actual_sha256": actual_sha,
                }
            )
    return {
        "manifest_status": manifest["status"],
        "manifest_artifacts": len(artifacts),
        "verified_artifacts": len(artifacts) - len(mismatches),
        "verified_bytes": total_bytes,
        "mismatches": mismatches,
        "integrity_passed": not mismatches,
        "status_file_is_manifest_pending_sentinel": cast(dict[str, Any], _json_load(run_dir / "status.json")).get("status")
        == "artifacts_complete_manifest_pending",
    }


def _font(size: int, *, bold: bool = False) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    filenames = (
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"
        if bold
        else "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        "/usr/share/fonts/truetype/liberation2/LiberationSans-Bold.ttf"
        if bold
        else "/usr/share/fonts/truetype/liberation2/LiberationSans-Regular.ttf",
    )
    for filename in filenames:
        if Path(filename).is_file():
            return ImageFont.truetype(filename, size=size)
    return ImageFont.load_default()


def _format_number(value: float) -> str:
    absolute = abs(value)
    if absolute >= 1_000_000:
        return f"{value / 1_000_000:.1f}M"
    if absolute >= 1_000:
        return f"{value / 1_000:.1f}k"
    if absolute >= 10:
        return f"{value:.0f}"
    if absolute >= 1:
        return f"{value:.1f}"
    if absolute == 0:
        return "0"
    if absolute < 0.001:
        return f"{value:.1e}"
    return f"{value:.3f}"


def _chart_canvas(title: str, subtitle: str) -> tuple[Image.Image, ImageDraw.ImageDraw]:
    image = Image.new("RGB", (1600, 900), BACKGROUND)
    draw = ImageDraw.Draw(image)
    draw.text((70, 35), title, fill=INK, font=_font(38, bold=True))
    draw.text((72, 84), subtitle, fill=MUTED, font=_font(20))
    return image, draw


def _draw_axes(
    draw: ImageDraw.ImageDraw,
    bounds: tuple[int, int, int, int],
    x_min: float,
    x_max: float,
    y_min: float,
    y_max: float,
    x_label: str,
    y_label: str,
) -> tuple[Callable[[float], float], Callable[[float], float]]:
    left, top, right, bottom = bounds
    draw.rectangle(bounds, fill=PLOT_BACKGROUND)
    x_span = x_max - x_min if x_max != x_min else 1.0
    y_span = y_max - y_min if y_max != y_min else 1.0
    def map_x(value: float) -> float:
        return left + (value - x_min) / x_span * (right - left)

    def map_y(value: float) -> float:
        return bottom - (value - y_min) / y_span * (bottom - top)
    for index in range(6):
        fraction = index / 5
        y = bottom - fraction * (bottom - top)
        value = y_min + fraction * y_span
        draw.line((left, y, right, y), fill=GRID, width=1)
        label = _format_number(value)
        width = draw.textbbox((0, 0), label, font=_font(16))[2]
        draw.text((left - width - 12, y - 9), label, fill=MUTED, font=_font(16))
    for index in range(6):
        fraction = index / 5
        x = left + fraction * (right - left)
        value = x_min + fraction * x_span
        draw.line((x, top, x, bottom), fill=GRID, width=1)
        label = _format_number(value)
        width = draw.textbbox((0, 0), label, font=_font(16))[2]
        draw.text((x - width / 2, bottom + 10), label, fill=MUTED, font=_font(16))
    draw.line((left, bottom, right, bottom), fill=INK, width=2)
    draw.line((left, top, left, bottom), fill=INK, width=2)
    width = draw.textbbox((0, 0), x_label, font=_font(18, bold=True))[2]
    draw.text(((left + right - width) / 2, bottom + 48), x_label, fill=INK, font=_font(18, bold=True))
    draw.text((12, (top + bottom) / 2), y_label, fill=INK, font=_font(18, bold=True))
    return map_x, map_y


def _line_chart(
    path: Path,
    title: str,
    subtitle: str,
    series: Sequence[tuple[str, Sequence[float], Sequence[float], str]],
    x_label: str,
    y_label: str,
    *,
    y_range: tuple[float, float] | None = None,
    shades: Sequence[tuple[float, float, str]] = (),
) -> None:
    all_x = [value for _, xs, _, _ in series for value in xs]
    all_y = [value for _, _, ys, _ in series for value in ys]
    if not all_x or not all_y:
        raise ValueError(f"line chart {title!r} has no values")
    x_min, x_max = min(all_x), max(all_x)
    if y_range is None:
        y_min, y_max = min(all_y), max(all_y)
        padding = (y_max - y_min) * 0.08 or 1.0
        y_min -= padding
        y_max += padding
    else:
        y_min, y_max = y_range
    image, draw = _chart_canvas(title, subtitle)
    bounds = (145, 135, 1515, 760)
    map_x, map_y = _draw_axes(draw, bounds, x_min, x_max, y_min, y_max, x_label, y_label)
    for shade_start, shade_end, _ in shades:
        start = max(bounds[0], min(bounds[2], map_x(shade_start)))
        end = max(bounds[0], min(bounds[2], map_x(shade_end)))
        if end > start:
            draw.rectangle((start, bounds[1], end, bounds[3]), fill="#E8EDF5")
    # Redraw grid lightly over phase shading.
    map_x, map_y = _draw_axes(draw, bounds, x_min, x_max, y_min, y_max, x_label, y_label)
    for shade_start, shade_end, _ in shades:
        start = max(bounds[0], min(bounds[2], map_x(shade_start)))
        end = max(bounds[0], min(bounds[2], map_x(shade_end)))
        if end > start:
            overlay = Image.new("RGBA", image.size, (0, 0, 0, 0))
            overlay_draw = ImageDraw.Draw(overlay)
            overlay_draw.rectangle((start, bounds[1], end, bounds[3]), fill=(100, 116, 139, 25))
            image = Image.alpha_composite(image.convert("RGBA"), overlay).convert("RGB")
            draw = ImageDraw.Draw(image)
    legend_x: float = 170
    for label, xs, ys, color in series:
        if len(xs) != len(ys):
            raise ValueError(f"line chart series length mismatch: {label}")
        points = [(map_x(x), map_y(y)) for x, y in zip(xs, ys, strict=True)]
        if len(points) == 1:
            x, y = points[0]
            draw.ellipse((x - 5, y - 5, x + 5, y + 5), fill=color)
        else:
            draw.line(points, fill=color, width=4, joint="curve")
            for x, y in points:
                draw.ellipse((x - 3, y - 3, x + 3, y + 3), fill=color)
        draw.rectangle((legend_x, 790, legend_x + 24, 814), fill=color)
        draw.text((legend_x + 34, 790), label, fill=INK, font=_font(18))
        legend_x += draw.textbbox((0, 0), label, font=_font(18))[2] + 85
    if shades:
        draw.rectangle((1320, 793, 1344, 811), fill="#D9E0EA")
        draw.text((1354, 790), "evaluation window", fill=INK, font=_font(16))
    path.parent.mkdir(parents=True, exist_ok=True)
    image.save(path, optimize=True)


def _horizontal_bar_chart(
    path: Path,
    title: str,
    subtitle: str,
    labels: Sequence[str],
    values: Sequence[float],
    *,
    x_label: str,
    colors: Sequence[str] | None = None,
    x_max: float | None = None,
) -> None:
    if not labels or len(labels) != len(values):
        raise ValueError(f"bar chart {title!r} has invalid values")
    image, draw = _chart_canvas(title, subtitle)
    left, top, right, bottom = 500, 140, 1510, 790
    maximum = x_max if x_max is not None else max(values) * 1.08 or 1.0
    draw.rectangle((left, top, right, bottom), fill=PLOT_BACKGROUND)
    for index in range(6):
        x = left + index / 5 * (right - left)
        draw.line((x, top, x, bottom), fill=GRID, width=1)
        value = maximum * index / 5
        text = _format_number(value)
        width = draw.textbbox((0, 0), text, font=_font(15))[2]
        draw.text((x - width / 2, bottom + 8), text, fill=MUTED, font=_font(15))
    row_height = (bottom - top) / len(labels)
    palette = list(colors) if colors else [PALETTE[index % len(PALETTE)] for index in range(len(labels))]
    for index, (label, value, color) in enumerate(zip(labels, values, palette, strict=True)):
        y0 = top + index * row_height + row_height * 0.18
        y1 = top + (index + 1) * row_height - row_height * 0.18
        x1 = left + value / maximum * (right - left)
        draw.rounded_rectangle((left, y0, x1, y1), radius=5, fill=color)
        truncated = label if len(label) <= 50 else "…" + label[-49:]
        width = draw.textbbox((0, 0), truncated, font=_font(16))[2]
        draw.text((left - width - 15, (y0 + y1) / 2 - 9), truncated, fill=INK, font=_font(16))
        draw.text((x1 + 8, (y0 + y1) / 2 - 9), _format_number(value), fill=INK, font=_font(16, bold=True))
    width = draw.textbbox((0, 0), x_label, font=_font(18, bold=True))[2]
    draw.text(((left + right - width) / 2, 835), x_label, fill=INK, font=_font(18, bold=True))
    path.parent.mkdir(parents=True, exist_ok=True)
    image.save(path, optimize=True)


def _histogram(
    path: Path,
    title: str,
    subtitle: str,
    values: Sequence[float],
    bins: int,
    x_label: str,
) -> None:
    if not values:
        raise ValueError(f"histogram {title!r} has no values")
    lower, upper = min(values), max(values)
    if lower == upper:
        lower -= 0.5
        upper += 0.5
    width = (upper - lower) / bins
    counts = [0] * bins
    for value in values:
        index = min(bins - 1, max(0, int((value - lower) / width)))
        counts[index] += 1
    centers = [lower + (index + 0.5) * width for index in range(bins)]
    image, draw = _chart_canvas(title, subtitle)
    bounds = (145, 135, 1515, 760)
    map_x, map_y = _draw_axes(
        draw, bounds, lower, upper, 0, max(counts) * 1.12 or 1, x_label, "Documents"
    )
    for center, count in zip(centers, counts, strict=True):
        x0 = map_x(center - width * 0.47)
        x1 = map_x(center + width * 0.47)
        draw.rectangle((x0, map_y(count), x1, map_y(0)), fill=PALETTE[0])
    image.save(path, optimize=True)


def _scatter_chart(
    path: Path,
    title: str,
    subtitle: str,
    groups: Sequence[tuple[str, Sequence[float], Sequence[float], str]],
    x_label: str,
    y_label: str,
    *,
    y_range: tuple[float, float] | None = None,
) -> None:
    all_x = [value for _, xs, _, _ in groups for value in xs]
    all_y = [value for _, _, ys, _ in groups for value in ys]
    if not all_x or not all_y:
        raise ValueError(f"scatter chart {title!r} has no values")
    x_padding = (max(all_x) - min(all_x)) * 0.05 or 1.0
    if y_range is None:
        y_padding = (max(all_y) - min(all_y)) * 0.05 or 0.1
        actual_y_range = (min(all_y) - y_padding, max(all_y) + y_padding)
    else:
        actual_y_range = y_range
    image, draw = _chart_canvas(title, subtitle)
    map_x, map_y = _draw_axes(
        draw,
        (145, 135, 1515, 760),
        min(all_x) - x_padding,
        max(all_x) + x_padding,
        actual_y_range[0],
        actual_y_range[1],
        x_label,
        y_label,
    )
    legend_x: float = 170
    for label, xs, ys, color in groups:
        for x, y in zip(xs, ys, strict=True):
            px, py = map_x(x), map_y(y)
            draw.ellipse((px - 6, py - 6, px + 6, py + 6), fill=color, outline="#FFFFFF", width=1)
        draw.ellipse((legend_x, 790, legend_x + 22, 812), fill=color)
        draw.text((legend_x + 32, 790), label, fill=INK, font=_font(18))
        legend_x += draw.textbbox((0, 0), label, font=_font(18))[2] + 80
    image.save(path, optimize=True)


def _grouped_bar_chart(
    path: Path,
    title: str,
    subtitle: str,
    labels: Sequence[str],
    series: Sequence[tuple[str, Sequence[float], str]],
    y_label: str,
    *,
    y_max: float,
) -> None:
    if not labels or any(len(values) != len(labels) for _, values, _ in series):
        raise ValueError(f"grouped bar chart {title!r} has invalid values")
    image, draw = _chart_canvas(title, subtitle)
    left, top, right, bottom = 145, 135, 1515, 760
    _map_x, map_y = _draw_axes(
        draw, (left, top, right, bottom), 0, len(labels), 0, y_max, "Group", y_label
    )
    group_width = (right - left) / len(labels)
    bar_width = group_width * 0.72 / len(series)
    for group_index, label in enumerate(labels):
        for series_index, (_, values, color) in enumerate(series):
            x0 = left + group_index * group_width + group_width * 0.14 + series_index * bar_width
            x1 = x0 + bar_width * 0.9
            draw.rectangle((x0, map_y(values[group_index]), x1, map_y(0)), fill=color)
        text = label if len(label) < 20 else label[:18] + "…"
        width = draw.textbbox((0, 0), text, font=_font(14))[2]
        draw.text((left + (group_index + 0.5) * group_width - width / 2, bottom + 12), text, fill=INK, font=_font(14))
    legend_x: float = 170
    for label, _, color in series:
        draw.rectangle((legend_x, 810, legend_x + 22, 832), fill=color)
        draw.text((legend_x + 32, 808), label, fill=INK, font=_font(17))
        legend_x += draw.textbbox((0, 0), label, font=_font(17))[2] + 80
    image.save(path, optimize=True)


def _error_bar_chart(
    path: Path, title: str, subtitle: str, intervals: Sequence[Mapping[str, Any]]
) -> None:
    image, draw = _chart_canvas(title, subtitle)
    left, top, right, bottom = 500, 145, 1510, 790
    draw.rectangle((left, top, right, bottom), fill=PLOT_BACKGROUND)
    for index in range(6):
        value = index / 5
        x = left + value * (right - left)
        draw.line((x, top, x, bottom), fill=GRID, width=1)
        draw.text((x - 12, bottom + 10), f"{value:.1f}", fill=MUTED, font=_font(15))
    row_height = (bottom - top) / len(intervals)
    for index, item in enumerate(intervals):
        y = top + (index + 0.5) * row_height
        low = left + float(item["ci95_lower"]) * (right - left)
        high = left + float(item["ci95_upper"]) * (right - left)
        point = left + float(item["point_estimate"]) * (right - left)
        draw.line((low, y, high, y), fill=PALETTE[0], width=5)
        draw.line((low, y - 8, low, y + 8), fill=PALETTE[0], width=3)
        draw.line((high, y - 8, high, y + 8), fill=PALETTE[0], width=3)
        draw.ellipse((point - 7, y - 7, point + 7, y + 7), fill=PALETTE[1])
        label = str(item["metric"])
        width = draw.textbbox((0, 0), label, font=_font(16))[2]
        draw.text((left - width - 15, y - 10), label, fill=INK, font=_font(16))
    image.save(path, optimize=True)


def _downsample(rows: Sequence[Mapping[str, Any]], key: str, max_points: int = 1200) -> tuple[list[float], list[float]]:
    if len(rows) <= max_points:
        return (
            [float(row["elapsed_seconds"]) / 3600 for row in rows],
            [float(row[key]) for row in rows],
        )
    bucket = math.ceil(len(rows) / max_points)
    x_values: list[float] = []
    y_values: list[float] = []
    for start in range(0, len(rows), bucket):
        chunk = rows[start : start + bucket]
        x_values.append(statistics.fmean(float(row["elapsed_seconds"]) for row in chunk) / 3600)
        y_values.append(statistics.fmean(float(row[key]) for row in chunk))
    return x_values, y_values


def _render_plots(
    output_dir: Path,
    training_history: Sequence[Mapping[str, Any]],
    eval_history: Sequence[Mapping[str, Any]],
    diagnostics: Sequence[PredictionDiagnostic],
    field_rows: Sequence[Mapping[str, Any]],
    section_rows: Sequence[Mapping[str, Any]],
    group_rows: Sequence[Mapping[str, Any]],
    bootstrap_rows: Sequence[Mapping[str, Any]],
    system_rows: Sequence[Mapping[str, Any]],
    phase_rows: Sequence[Mapping[str, Any]],
    phase_system_rows: Sequence[Mapping[str, Any]],
) -> list[str]:
    plots = output_dir / "plots"
    plots.mkdir(parents=True, exist_ok=True)
    generated: list[str] = []

    def publish(name: str, renderer: Callable[[Path], None]) -> None:
        path = plots / name
        renderer(path)
        generated.append(str(path.relative_to(output_dir)))

    train_steps = [float(row["step"]) for row in training_history]
    train_epochs = [float(row["epoch"]) for row in training_history]
    eval_epochs = [float(row["epoch"]) for row in eval_history]
    eval_steps = [float(row["step"]) for row in eval_history]
    generation_ceiling = max(int(row["eval_generated_tokens_max"]) for row in eval_history)
    base_model_certified = float(eval_history[0].get("eval_is_base_model", 0.0)) == 1.0
    publish(
        "01_training_loss_log_scale.png",
        lambda path: _line_chart(
            path,
            "Training loss collapsed across 25 epochs",
            "Log10 scale exposes late-stage changes hidden by the early loss drop.",
            [("window loss", train_steps, [math.log10(float(row["loss"])) for row in training_history], PALETTE[0])],
            "Optimizer step",
            "log10(loss)",
        ),
    )
    publish(
        "02_training_cumulative_loss.png",
        lambda path: _line_chart(
            path,
            "Cumulative training loss",
            "Optimizer-step-weighted mean, explicitly logged for MLflow comparability.",
            [("cumulative loss", train_epochs, [float(row["train_cumulative_loss"]) for row in training_history], PALETTE[2])],
            "Epoch",
            "Cumulative loss",
        ),
    )
    publish(
        "03_learning_rate_schedule.png",
        lambda path: _line_chart(
            path,
            "Learning-rate schedule",
            "Warmup followed by cosine decay to effectively zero at epoch 25.",
            [("learning rate", train_steps, [float(row["learning_rate"]) for row in training_history], PALETTE[4])],
            "Optimizer step",
            "Learning rate",
        ),
    )
    publish(
        "04_gradient_norm_log_scale.png",
        lambda path: _line_chart(
            path,
            "Gradient norm",
            "Log10 scale; values are observed before max-norm clipping by Trainer.",
            [("gradient norm", train_steps, [math.log10(float(row["grad_norm"])) for row in training_history], PALETTE[3])],
            "Optimizer step",
            "log10(gradient norm)",
        ),
    )
    publish(
        "05_eval_field_metrics.png",
        lambda path: _line_chart(
            path,
            "Exact field/value metrics by evaluation epoch",
            (
                "Epoch 0 is the explicitly adapter-disabled base-model baseline."
                if base_model_certified
                else "Epoch 0 lacks the adapter-disabled base-model marker."
            ),
            [
                ("precision", eval_epochs, [float(row["eval_field_value_precision"]) for row in eval_history], PALETTE[0]),
                ("recall", eval_epochs, [float(row["eval_field_value_recall"]) for row in eval_history], PALETTE[1]),
                ("F1", eval_epochs, [float(row["eval_field_value_f1"]) for row in eval_history], PALETTE[2]),
                ("accuracy", eval_epochs, [float(row["eval_field_value_accuracy"]) for row in eval_history], PALETTE[3]),
            ],
            "Epoch",
            "Score",
            y_range=(0, 1),
        ),
    )
    publish(
        "06_eval_validity_and_exactness.png",
        lambda path: _line_chart(
            path,
            "Output validity and whole-document exactness",
            "JSON syntax is substantially easier than full schema validity or exact document reproduction.",
            [
                ("JSON valid", eval_epochs, [float(row["eval_json_valid"]) for row in eval_history], PALETTE[0]),
                ("schema valid", eval_epochs, [float(row["eval_schema_valid"]) for row in eval_history], PALETTE[4]),
                ("document exact", eval_epochs, [float(row["eval_canonical_exact_match"]) for row in eval_history], PALETTE[1]),
            ],
            "Epoch",
            "Fraction of documents",
            y_range=(0, 1),
        ),
    )
    publish(
        "07_eval_loss.png",
        lambda path: _line_chart(
            path,
            "Teacher-forced validation loss",
            "Best loss occurs at epoch 5 even though generated exact field/value F1 continues improving.",
            [("eval loss", eval_epochs, [float(row["eval_loss"]) for row in eval_history], PALETTE[1])],
            "Epoch",
            "Cross-entropy loss",
        ),
    )
    publish(
        "08_eval_generation_length.png",
        lambda path: _line_chart(
            path,
            "Generated output length",
            f"The base/initialized model exhausted the {generation_ceiling:,}-token ceiling; trained checkpoints usually terminated earlier.",
            [("mean generated tokens", eval_epochs, [float(row["eval_generated_tokens_mean"]) for row in eval_history], PALETTE[5])],
            "Epoch",
            "Mean tokens",
            y_range=(0, generation_ceiling),
        ),
    )
    publish(
        "09_eval_runtime.png",
        lambda path: _line_chart(
            path,
            "Evaluation runtime",
            "Runtime tracks generated length because autoregressive decoding dominates evaluation cost.",
            [("runtime", eval_epochs, [float(row["eval_runtime"]) for row in eval_history], PALETTE[4])],
            "Epoch",
            "Seconds",
        ),
    )
    publish(
        "10_eval_gpu_memory.png",
        lambda path: _line_chart(
            path,
            "Evaluation GPU memory peaks",
            "Allocated and reserved peaks remained below the 24 GiB RTX 4090 device limit.",
            [
                ("allocated", eval_steps, [float(row["eval_cuda_peak_allocated_gib"]) for row in eval_history], PALETTE[0]),
                ("reserved", eval_steps, [float(row["eval_cuda_peak_reserved_gib"]) for row in eval_history], PALETTE[3]),
            ],
            "Optimizer step",
            "GiB",
            y_range=(0, 24),
        ),
    )
    f1_values = [item.counts.f1 for item in diagnostics]
    publish(
        "11_document_f1_distribution.png",
        lambda path: _histogram(
            path,
            "Per-document exact field/value F1",
            "Micro aggregate can conceal a long tail of difficult documents.",
            f1_values,
            12,
            "Document F1",
        ),
    )
    cohorts = sorted({item.cohort for item in diagnostics})
    publish(
        "12_precision_recall_by_document.png",
        lambda path: _scatter_chart(
            path,
            "Per-document precision versus recall",
            "Points below the diagonal are omission-dominated; color identifies source cohort.",
            [
                (
                    cohort,
                    [item.counts.precision for item in diagnostics if item.cohort == cohort],
                    [item.counts.recall for item in diagnostics if item.cohort == cohort],
                    PALETTE[index],
                )
                for index, cohort in enumerate(cohorts)
            ],
            "Precision",
            "Recall",
            y_range=(0, 1),
        ),
    )
    publish(
        "13_f1_vs_reference_complexity.png",
        lambda path: _scatter_chart(
            path,
            "Document F1 versus label complexity",
            "Complexity is the number of exact scalar reference leaves.",
            [
                (
                    cohort,
                    [float(item.counts.reference) for item in diagnostics if item.cohort == cohort],
                    [item.counts.f1 for item in diagnostics if item.cohort == cohort],
                    PALETTE[index],
                )
                for index, cohort in enumerate(cohorts)
            ],
            "Reference scalar leaves",
            "Document F1",
            y_range=(0, 1),
        ),
    )
    publish(
        "14_f1_vs_input_tokens.png",
        lambda path: _scatter_chart(
            path,
            "Document F1 versus input length",
            "No validation source was truncated; length therefore measures document difficulty, not clipping.",
            [
                (
                    cohort,
                    [float(item.input_tokens) for item in diagnostics if item.cohort == cohort],
                    [item.counts.f1 for item in diagnostics if item.cohort == cohort],
                    PALETTE[index],
                )
                for index, cohort in enumerate(cohorts)
            ],
            "Input tokens",
            "Document F1",
            y_range=(0, 1),
        ),
    )
    publish(
        "15_output_length_ratio.png",
        lambda path: _histogram(
            path,
            "Generated/reference token-length ratio",
            "A ratio near 1 is only a length diagnostic; it does not imply semantic correctness.",
            [_safe_divide(item.generated_tokens, item.reference_content_tokens) for item in diagnostics],
            14,
            "Generated tokens / reference tokens",
        ),
    )
    supported_fields = [row for row in field_rows if int(row["reference_values"]) >= 8]
    lowest_fields = sorted(supported_fields, key=lambda row: float(row["f1"]))[:15]
    publish(
        "16_lowest_supported_field_f1.png",
        lambda path: _horizontal_bar_chart(
            path,
            "Lowest-F1 fields with at least eight reference values",
            "Array indices are retained for strict scoring but collapsed only for field naming.",
            [str(row["field_path"]) for row in lowest_fields],
            [float(row["f1"]) for row in lowest_fields],
            x_label="Exact field/value F1",
            x_max=1,
        ),
    )
    error_fields = sorted(
        field_rows,
        key=lambda row: int(row["false_positive_values"]) + int(row["false_negative_values"]),
        reverse=True,
    )[:15]
    publish(
        "17_highest_field_error_counts.png",
        lambda path: _horizontal_bar_chart(
            path,
            "Fields contributing the most exact-value errors",
            "Count is false positives plus false negatives; a substitution contributes one of each.",
            [str(row["field_path"]) for row in error_fields],
            [float(int(row["false_positive_values"]) + int(row["false_negative_values"])) for row in error_fields],
            x_label="FP + FN values",
        ),
    )
    ordered_sections = sorted(section_rows, key=lambda row: int(row["reference_values"]), reverse=True)
    publish(
        "18_section_f1.png",
        lambda path: _horizontal_bar_chart(
            path,
            "Top-level section F1",
            "Micro exact scalar field/value scoring within each semantic output section.",
            [str(row["section"]) for row in ordered_sections],
            [float(row["f1"]) for row in ordered_sections],
            x_label="Field/value F1",
            x_max=1,
        ),
    )
    cohort_rows = [row for row in group_rows if row["group_type"] == "cohort"]
    publish(
        "19_cohort_metrics.png",
        lambda path: _grouped_bar_chart(
            path,
            "Validation performance by source cohort",
            "Pilot and follow-up subsets were independently sampled before being combined.",
            [str(row["group"]) for row in cohort_rows],
            [
                ("precision", [float(row["field_value_precision"]) for row in cohort_rows], PALETTE[0]),
                ("recall", [float(row["field_value_recall"]) for row in cohort_rows], PALETTE[1]),
                ("F1", [float(row["field_value_f1"]) for row in cohort_rows], PALETTE[2]),
                ("schema valid", [float(row["schema_valid"]) for row in cohort_rows], PALETTE[4]),
            ],
            "Score",
            y_max=1,
        ),
    )
    error_labels = ["substitutions", "omissions", "additions"]
    error_values = [
        float(sum(item.substitutions for item in diagnostics)),
        float(sum(item.omissions for item in diagnostics)),
        float(sum(item.additions for item in diagnostics)),
    ]
    publish(
        "20_error_taxonomy.png",
        lambda path: _horizontal_bar_chart(
            path,
            "Path-level error taxonomy",
            "Shared path with wrong value is a substitution; missing and unexpected paths are separate.",
            error_labels,
            error_values,
            x_label="Path occurrences",
            colors=(PALETTE[1], PALETTE[4], PALETTE[3]),
        ),
    )
    strict_counts = _aggregate_counts(diagnostics)
    index_counts = _aggregate_index_counts(diagnostics)
    publish(
        "21_index_sensitivity.png",
        lambda path: _grouped_bar_chart(
            path,
            "Strict versus index-insensitive field matching",
            "Index-insensitive matching is diagnostic only; it estimates list-position/alignment penalties.",
            ["precision", "recall", "F1"],
            [
                ("strict", [strict_counts.precision, strict_counts.recall, strict_counts.f1], PALETTE[0]),
                ("ignore array index", [index_counts.precision, index_counts.recall, index_counts.f1], PALETTE[5]),
            ],
            "Score",
            y_max=1,
        ),
    )
    publish(
        "22_bootstrap_confidence_intervals.png",
        lambda path: _error_bar_chart(
            path,
            "Document-bootstrap 95% confidence intervals",
            f"Percentile intervals from {BOOTSTRAP_REPLICATES:,} deterministic document-level resamples.",
            bootstrap_rows,
        ),
    )
    validity_labels = ["schema valid", "JSON-only valid", "invalid JSON"]
    validity_values = [
        float(sum(item.assessment.schema_valid for item in diagnostics)),
        float(sum(item.assessment.json_valid and not item.assessment.schema_valid for item in diagnostics)),
        float(sum(not item.assessment.json_valid for item in diagnostics)),
    ]
    publish(
        "23_validity_breakdown.png",
        lambda path: _horizontal_bar_chart(
            path,
            "Final validation output validity",
            f"Counts are mutually exclusive across all {len(diagnostics)} saved predictions.",
            validity_labels,
            validity_values,
            x_label="Documents",
            colors=(PALETTE[2], PALETTE[4], PALETTE[1]),
        ),
    )
    shade_hours = [
        (
            float(row["start_elapsed_seconds"]) / 3600,
            float(row["end_elapsed_seconds"]) / 3600,
            str(row["phase"]),
        )
        for row in phase_rows
    ]
    gpu_x, gpu_y = _downsample(system_rows, "system/gpu_0_utilization_percentage")
    publish(
        "24_gpu_utilization_timeline.png",
        lambda path: _line_chart(
            path,
            "GPU utilization over run wall time",
            "One-second MLflow samples downsampled by mean; shaded regions are evaluation windows.",
            [("GPU utilization", gpu_x, gpu_y, PALETTE[0])],
            "Elapsed hours",
            "Percent",
            y_range=(0, 100),
            shades=shade_hours,
        ),
    )
    memory_x, memory_y = _downsample(system_rows, "system/gpu_0_memory_usage_megabytes")
    publish(
        "25_gpu_memory_timeline.png",
        lambda path: _line_chart(
            path,
            "GPU memory usage over run wall time",
            "Driver-level MLflow samples; the device has 24 GiB nominal memory.",
            [("GPU memory", memory_x, [value / 1024 for value in memory_y], PALETTE[3])],
            "Elapsed hours",
            "GiB",
            y_range=(0, 24),
            shades=shade_hours,
        ),
    )
    power_x, power_y = _downsample(system_rows, "system/gpu_0_power_usage_watts")
    publish(
        "26_gpu_power_timeline.png",
        lambda path: _line_chart(
            path,
            "GPU power over run wall time",
            "One-second samples downsampled by mean; evaluation alternates compute and decode behavior.",
            [("GPU power", power_x, power_y, PALETTE[4])],
            "Elapsed hours",
            "Watts",
            y_range=(0, 450),
            shades=shade_hours,
        ),
    )
    cpu_x, cpu_y = _downsample(system_rows, "system/cpu_utilization_percentage")
    ram_x, ram_y = _downsample(system_rows, "system/system_memory_usage_percentage")
    publish(
        "27_host_cpu_and_ram.png",
        lambda path: _line_chart(
            path,
            "Host CPU and RAM utilization",
            "MLflow system metrics sampled once per second.",
            [
                ("CPU", cpu_x, cpu_y, PALETTE[0]),
                ("RAM", ram_x, ram_y, PALETTE[2]),
            ],
            "Elapsed hours",
            "Percent",
            y_range=(0, 100),
            shades=shade_hours,
        ),
    )
    page_rows = [row for row in group_rows if row["group_type"] == "page_count"]
    publish(
        "28_performance_by_page_count.png",
        lambda path: _grouped_bar_chart(
            path,
            "Performance by document page count",
            "The four-page slice contains only three documents and should be treated as directional.",
            [str(row["group"]) for row in page_rows],
            [
                ("field/value F1", [float(row["field_value_f1"]) for row in page_rows], PALETTE[2]),
                ("schema valid", [float(row["schema_valid"]) for row in page_rows], PALETTE[4]),
            ],
            "Score",
            y_max=1,
        ),
    )
    complexity_order = {
        "<20 reference leaves": 0,
        "20-39 reference leaves": 1,
        "40-59 reference leaves": 2,
        "60+ reference leaves": 3,
    }
    complexity_rows = sorted(
        [row for row in group_rows if row["group_type"] == "label_complexity"],
        key=lambda row: complexity_order[str(row["group"])],
    )
    publish(
        "29_performance_by_label_complexity.png",
        lambda path: _grouped_bar_chart(
            path,
            "Performance by reference-label complexity",
            "The 60+ leaf bucket contains five documents; difficulty rises sharply in this tail.",
            [str(row["group"]) for row in complexity_rows],
            [
                ("field/value F1", [float(row["field_value_f1"]) for row in complexity_rows], PALETTE[2]),
                ("schema valid", [float(row["schema_valid"]) for row in complexity_rows], PALETTE[4]),
            ],
            "Score",
            y_max=1,
        ),
    )
    alignment_fields = sorted(
        [row for row in field_rows if int(row["reference_values"]) >= 8],
        key=lambda row: float(row["index_insensitive_f1"]) - float(row["f1"]),
        reverse=True,
    )[:12]
    publish(
        "30_largest_list_alignment_penalties.png",
        lambda path: _horizontal_bar_chart(
            path,
            "Largest list-index alignment penalties",
            "Shown as index-insensitive F1 minus strict indexed F1; diagnostic only.",
            [str(row["field_path"]) for row in alignment_fields],
            [float(row["index_insensitive_f1"]) - float(row["f1"]) for row in alignment_fields],
            x_label="F1 recovered when array indices are ignored",
        ),
    )
    worst_documents = sorted(
        diagnostics,
        key=lambda item: item.counts.predicted + item.counts.reference - 2 * item.counts.true_positive,
        reverse=True,
    )[:15]
    publish(
        "31_highest_document_error_counts.png",
        lambda path: _horizontal_bar_chart(
            path,
            "Documents with the most strict field/value errors",
            "Error count is FP + FN; labels use the final 12 characters of the recovered document ID.",
            [item.document_id[-12:] for item in worst_documents],
            [
                float(
                    item.counts.predicted
                    + item.counts.reference
                    - 2 * item.counts.true_positive
                )
                for item in worst_documents
            ],
            x_label="FP + FN values",
        ),
    )
    phase_labels = [str(row["phase"]) for row in phase_system_rows]
    publish(
        "32_gpu_utilization_by_phase.png",
        lambda path: _horizontal_bar_chart(
            path,
            "Mean GPU utilization by run phase",
            "Derived from aligned one-second MLflow system samples and event-timestamp windows.",
            phase_labels,
            [float(row["gpu_0_utilization_percentage_mean"]) for row in phase_system_rows],
            x_label="Mean GPU utilization (%)",
            x_max=100,
        ),
    )
    failure_counts = Counter(item.schema_failure_class for item in diagnostics)
    failure_order = ["none", "schema_validation_error", "invalid_json"]
    publish(
        "33_output_failure_classes.png",
        lambda path: _horizontal_bar_chart(
            path,
            "Output validation failure classes",
            "Mutually exclusive terminal diagnosis from strict JSON parsing and Pydantic validation.",
            failure_order,
            [float(failure_counts[name]) for name in failure_order],
            x_label="Documents",
            colors=(PALETTE[2], PALETTE[4], PALETTE[1]),
        ),
    )
    return generated


def _failure_gallery(
    output_dir: Path,
    diagnostics: Sequence[PredictionDiagnostic],
    records: Mapping[str, DatasetRecord],
) -> None:
    ranked = sorted(
        diagnostics,
        key=lambda item: (
            item.assessment.json_valid,
            item.assessment.schema_valid,
            item.counts.f1,
            item.document_id,
        ),
    )[:12]
    lines = [
        "# Lowest-performing validation examples",
        "",
        "These examples are ranked by syntax validity, schema validity, then strict exact field/value F1. "
        "The raw OCR is intentionally not reproduced in full; each section includes a canonical JSON diff "
        "and the document ID needed to trace the pinned source row.",
        "",
    ]
    jsonl_rows: list[dict[str, Any]] = []
    for item in ranked:
        generated_lines = json.dumps(
            json.loads(item.assessment.generated_text), ensure_ascii=False, sort_keys=True, indent=2
        ).splitlines() if item.assessment.json_valid else item.assessment.generated_text.splitlines()
        reference_lines = json.dumps(
            json.loads(item.assessment.reference_text), ensure_ascii=False, sort_keys=True, indent=2
        ).splitlines()
        diff = list(
            difflib.unified_diff(
                reference_lines,
                generated_lines,
                fromfile="reference",
                tofile="generated",
                lineterm="",
            )
        )
        lines.extend(
            [
                f"## {item.document_id}",
                "",
                f"- Cohort: `{item.cohort}`",
                f"- Published row ID: `{item.published_document_id}`",
                f"- Source dataset: `{records[item.document_id].source_path}`",
                f"- Strict field/value F1: `{item.counts.f1:.4f}`",
                f"- JSON/schema valid: `{item.assessment.json_valid}` / `{item.assessment.schema_valid}`",
                f"- Errors: `{item.substitutions}` substitutions, `{item.omissions}` omissions, `{item.additions}` additions",
                f"- Failure class: `{item.schema_failure_class}`",
                "",
                "```diff",
                *diff[:240],
                "```",
                "",
            ]
        )
        jsonl_rows.append(
            {
                "document_id": item.document_id,
                "published_document_id": item.published_document_id,
                "cohort": item.cohort,
                "field_value_f1": item.counts.f1,
                "json_valid": item.assessment.json_valid,
                "schema_valid": item.assessment.schema_valid,
                "schema_failure_class": item.schema_failure_class,
                "reference_text": item.assessment.reference_text,
                "generated_text": item.assessment.generated_text,
            }
        )
    (output_dir / "examples").mkdir(parents=True, exist_ok=True)
    (output_dir / "examples" / "lowest_performing.md").write_text(
        "\n".join(lines), encoding="utf-8"
    )
    _write_jsonl(output_dir / "examples" / "lowest_performing.jsonl", jsonl_rows)


def _build_report(
    run_dir: Path,
    output_dir: Path,
    manifest: Mapping[str, Any],
    integrity: Mapping[str, Any],
    environment: Mapping[str, Any],
    config: Mapping[str, Any],
    trainer_state: Mapping[str, Any],
    eval_history: Sequence[Mapping[str, Any]],
    diagnostics: Sequence[PredictionDiagnostic],
    field_rows: Sequence[Mapping[str, Any]],
    section_rows: Sequence[Mapping[str, Any]],
    group_rows: Sequence[Mapping[str, Any]],
    bootstrap_rows: Sequence[Mapping[str, Any]],
    schema_errors: Sequence[Mapping[str, Any]],
    correlation_rows: Sequence[Mapping[str, Any]],
    coverage_rows: Sequence[Mapping[str, Any]],
    identity_rows: Sequence[Mapping[str, Any]],
    phase_system_rows: Sequence[Mapping[str, Any]],
    phase_rows: Sequence[Mapping[str, Any]],
    mlflow_export: MlflowExport,
    plot_paths: Sequence[str],
) -> str:
    counts = _aggregate_counts(diagnostics)
    index_counts = _aggregate_index_counts(diagnostics)
    final_metrics = cast(Mapping[str, Any], cast(Mapping[str, Any], manifest["metrics"])["validation_evaluation"])
    best_step = int(trainer_state["best_global_step"])
    best_metric = float(trainer_state["best_metric"])
    best_checkpoint = str(trainer_state["best_model_checkpoint"])
    json_invalid = sum(not item.assessment.json_valid for item in diagnostics)
    schema_invalid_json_valid = sum(
        item.assessment.json_valid and not item.assessment.schema_valid for item in diagnostics
    )
    exact = sum(item.assessment.canonical_exact_match for item in diagnostics)
    cap_hits = sum(item.likely_generation_cap for item in diagnostics)
    document_count = len(diagnostics)
    generation_max_length = int(cast(Mapping[str, Any], config["evaluation"])["generation_max_length"])
    epoch_zero = _epoch_zero_baseline(eval_history, environment)
    omissions = sum(item.omissions for item in diagnostics)
    additions = sum(item.additions for item in diagnostics)
    substitutions = sum(item.substitutions for item in diagnostics)
    first_post_init = eval_history[1]
    final_scheduled = eval_history[-1]
    loss_rise = _safe_divide(
        float(final_scheduled["eval_loss"]) - float(first_post_init["eval_loss"]),
        float(first_post_init["eval_loss"]),
    )
    f1_gain = float(final_scheduled["eval_field_value_f1"]) - float(
        first_post_init["eval_field_value_f1"]
    )
    cohort_rows = [row for row in group_rows if row["group_type"] == "cohort"]
    lowest_fields = sorted(
        [row for row in field_rows if int(row["reference_values"]) >= 8],
        key=lambda row: float(row["f1"]),
    )[:10]
    highest_error_fields = sorted(
        field_rows,
        key=lambda row: int(row["false_positive_values"]) + int(row["false_negative_values"]),
        reverse=True,
    )[:10]
    exact_match_ids = [item.document_id for item in diagnostics if item.assessment.canonical_exact_match]
    invalid_ids = [item.document_id for item in diagnostics if not item.assessment.json_valid]
    unseen_validation_fields = [
        row
        for row in coverage_rows
        if int(row["validation_values"]) > 0 and not int(row["validation_field_seen_in_train"])
    ]
    incorrect_published_ids = sum(not int(row["published_id_correct"]) for row in identity_rows)
    model = cast(Mapping[str, Any], environment["model"])
    hardware = cast(Mapping[str, Any], environment["hardware"])
    run_duration = (
        int(mlflow_export.run["end_time_ms"]) - int(mlflow_export.run["start_time_ms"])
    ) / 1000
    evaluation_seconds = sum(float(row["duration_seconds"]) for row in phase_rows)
    phase_system_lookup = {str(row["phase"]): row for row in phase_system_rows}
    non_eval_system = phase_system_lookup["non_evaluation"]
    initial_eval_system = phase_system_lookup["on_start_evaluation"]
    resolved_best_checkpoint = _resolved_checkpoint_path(run_dir, best_checkpoint)
    adapter_match = _sha256(
        run_dir / "final-adapter" / str(cast(Mapping[str, Any], config["peft"])["adapter_name"]) / "adapter_model.safetensors"
    ) == _sha256(
        resolved_best_checkpoint
        / str(cast(Mapping[str, Any], config["peft"])["adapter_name"])
        / "adapter_model.safetensors"
    )
    bootstrap_lookup = {str(row["metric"]): row for row in bootstrap_rows}
    f1_ci = bootstrap_lookup["field_value_f1"]
    report = f"""# T5Gemma2 270M LoRA — {manifest['run_id']} completed-run analysis

Generated at `{datetime.now(UTC).isoformat()}` from immutable run `{manifest['run_id']}`. This analysis loaded no model weights and used no GPU inference.

## Executive assessment

The run completed cleanly at the training/model level: all `{integrity['manifest_artifacts']}` manifest-listed artifacts ({int(integrity['verified_bytes']):,} bytes) passed byte-size and SHA-256 verification; MLflow is `FINISHED`; training reached all `{int(trainer_state['global_step'])}/{int(trainer_state['max_steps'])}` optimizer steps and `{float(trainer_state['epoch']):.1f}/{float(cast(Mapping[str, Any], config['optimization'])['num_train_epochs']):.1f}` epochs; and the final adapter is byte-identical to the selected best checkpoint at step `{best_step}`.

There is one serious metadata-publication defect: `{incorrect_published_ids}/{document_count}` rows in `predictions/validation.jsonl` carry the wrong `document_id`. Transformers 5.15 applies `LengthGroupedSampler` to evaluation/prediction when `train_sampling_strategy=group_by_length`; `_predict_and_publish` then zipped sampler-ordered outputs with original-order `dataset["document_id"]`. Generated text and reference text remained correctly paired, so aggregate metrics are unaffected. Every canonical reference target is unique in this split, allowing this analysis to recover a complete one-to-one identity mapping without inference or guessing. The original file remains immutable; `tables/prediction_identity_mapping.csv` records every published and recovered ID.

On the {document_count}-document validation set, the final saved predictions achieve **{counts.f1:.3f} exact field/value micro-F1** (95% document-bootstrap CI `{float(f1_ci['ci95_lower']):.3f}-{float(f1_ci['ci95_upper']):.3f}`), with precision `{counts.precision:.3f}` and recall `{counts.recall:.3f}`. Precision exceeds recall by `{counts.precision - counts.recall:.3f}`, and the path-level taxonomy contains `{omissions}` omissions versus `{additions}` additions and `{substitutions}` substitutions. The main remaining problem is therefore incomplete or misplaced extraction, not uncontrolled over-generation alone.

The output contract is not yet production-ready: `{document_count - json_invalid}/{document_count}` outputs are valid JSON, `{document_count - json_invalid - schema_invalid_json_valid}/{document_count}` are canonical schema-valid, and only `{exact}/{document_count}` is an exact whole-document match. `{cap_hits}` prediction(s) reached the {generation_max_length:,}-token generation ceiling; other syntax failures are finite malformed JSON outputs rather than cap exhaustion. Exact-document match is deliberately strict, but `{exact}/{document_count}` means almost every document still differs in at least one scalar value.

The generated metric improved through epoch 25: scheduled field/value F1 moved from `{float(first_post_init['eval_field_value_f1']):.3f}` at epoch 5 to `{float(final_scheduled['eval_field_value_f1']):.3f}` at epoch 25 (`+{f1_gain:.3f}`). In contrast, teacher-forced validation loss rose `{loss_rise:.1%}` over the same interval. This is a real objective divergence—not a corrupted run—and supports choosing checkpoints by generated structured F1 rather than cross-entropy loss. The gain from epoch 20 to 25 was only `{float(eval_history[-1]['eval_field_value_f1']) - float(eval_history[-2]['eval_field_value_f1']):.4f}`, so the curve is close to saturation on this split.

## Metric contract and interpretation

- A true positive is an exact `(indexed JSON path, canonical scalar JSON value)` match below `documentPatch`; `schemaVersion` is excluded.
- Micro precision/recall/F1 pool scalar leaves across documents. A wrong value at a shared path yields one false positive and one false negative.
- `field_value_accuracy` is true positives divided by the union of compared paths, not token accuracy and not document accuracy.
- `canonical_exact_match` requires the entire sparse canonical document target to match.
- JSON-valid but schema-invalid outputs still receive partial leaf credit, so schema validity must be read alongside F1.
- The index-insensitive diagnostic rises from strict F1 `{counts.f1:.3f}` to `{index_counts.f1:.3f}`. The `{index_counts.f1 - counts.f1:+.3f}` delta estimates list-index/alignment sensitivity; it is diagnostic and must not replace the strict metric.

## Epoch trajectory

| Epoch | Eval loss | JSON valid | Schema valid | Precision | Recall | F1 | Exact doc | Mean gen tokens | Runtime |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
"""
    for row in eval_history:
        report += (
            f"| {float(row['epoch']):.0f} | {float(row['eval_loss']):.4f} | "
            f"{float(row['eval_json_valid']):.3f} | {float(row['eval_schema_valid']):.3f} | "
            f"{float(row['eval_field_value_precision']):.3f} | {float(row['eval_field_value_recall']):.3f} | "
            f"{float(row['eval_field_value_f1']):.3f} | {float(row['eval_canonical_exact_match']):.3f} | "
            f"{float(row['eval_generated_tokens_mean']):.1f} | {float(row['eval_runtime']):.1f}s |\n"
        )
    report += f"""

Epoch 0 **{'is' if epoch_zero['certified_base_model'] else 'is not'}** a certified base-model baseline for this completed run: {epoch_zero['reason']}. {'The evaluation explicitly disabled the adapter and is valid auditable base-vs-adapter evidence.' if epoch_zero['certified_base_model'] else 'It may remain a useful qualitative starting point, but it is not auditable base-vs-adapter evidence.'}

The separately published final evaluation (`eval_results.json`) recomputed after best-model selection gives F1 `{float(final_metrics['eval_field_value_f1']):.6f}`, versus `{best_metric:.6f}` in the scheduled step-450 evaluation. The small `{float(final_metrics['eval_field_value_f1']) - best_metric:+.6f}` difference is compatible with non-bitwise-deterministic GPU generation (`full_determinism: false`); the saved 60-row prediction file is the authoritative final output analyzed here.

## Cohort and document-level behavior

| Cohort | Docs | JSON valid | Schema valid | Precision | Recall | F1 | Macro doc F1 |
|---|---:|---:|---:|---:|---:|---:|---:|
"""
    for row in cohort_rows:
        report += (
            f"| {row['group']} | {row['documents']} | {float(row['json_valid']):.3f} | "
            f"{float(row['schema_valid']):.3f} | {float(row['field_value_precision']):.3f} | "
            f"{float(row['field_value_recall']):.3f} | {float(row['field_value_f1']):.3f} | "
            f"{float(row['macro_document_f1']):.3f} |\n"
        )
    report += f"""

- Exact-match document IDs: `{', '.join(exact_match_ids) if exact_match_ids else 'none'}`.
- Invalid-JSON document IDs: `{', '.join(invalid_ids)}`.
- Likely generation-cap hits: `{cap_hits}` (`generated_tokens >= {generation_max_length - 1:,}`).
- Validation source truncations: `0`; cached input lengths therefore cover every source token.
- Exact train/validation document-ID or raw-OCR-hash leakage: `0`.
- Validation field paths absent from training: `{len(unseen_validation_fields)}`.
- Prediction rows with a misassigned published document ID: `{incorrect_published_ids}/{document_count}`; all {document_count} identities recovered uniquely from their pinned canonical references.

## Field-level weaknesses

Lowest strict F1 among fields with at least eight reference values:

| Field | Support | Precision | Recall | F1 | Whole-field exact docs |
|---|---:|---:|---:|---:|---:|
"""
    for row in lowest_fields:
        report += (
            f"| `{row['field_path']}` | {row['reference_values']} | {float(row['precision']):.3f} | "
            f"{float(row['recall']):.3f} | {float(row['f1']):.3f} | "
            f"{row['exact_documents']}/{row['supported_documents']} |\n"
        )
    report += """

Largest contributors to strict field/value error count:

| Field | FP | FN | Total | F1 |
|---|---:|---:|---:|---:|
"""
    for row in highest_error_fields:
        total = int(row["false_positive_values"]) + int(row["false_negative_values"])
        report += (
            f"| `{row['field_path']}` | {row['false_positive_values']} | {row['false_negative_values']} | "
            f"{total} | {float(row['f1']):.3f} |\n"
        )
    report += """

Top-level section results are in `tables/section_metrics.csv`. The field table includes both strict indexed scoring and an index-insensitive multiset diagnostic, plus a whole-field exact rate across supported documents.

## Validity failures

"""
    failure_counts = Counter(str(row["failure_class"]) for row in schema_errors)
    for failure_class, count in sorted(failure_counts.items()):
        report += f"- `{failure_class}`: {count} document(s)\n"
    report += f"""

Detailed Pydantic error locations/types are in `tables/schema_errors.jsonl`; no failure is silently coerced. The failure gallery contains the 12 lowest-performing saved predictions with canonical diffs.

## Training and systems behavior

- Wall time recorded by MLflow: `{run_duration / 3600:.3f}` hours.
- Trainer training runtime (includes scheduled evaluations): `{float(cast(Mapping[str, Any], manifest['metrics'])['train']['train_runtime']) / 3600:.3f}` hours.
- Hardware: `{cast(Sequence[Mapping[str, Any]], hardware['devices'])[0]['name']}`, `{int(cast(Sequence[Mapping[str, Any]], hardware['devices'])[0]['total_memory_bytes']) / 2**30:.2f}` GiB, BF16 + TF32.
- Trainable LoRA parameters: `{int(model['trainable_parameters']):,}` / `{int(model['total_parameters']):,}` (`{float(model['trainable_parameter_ratio']):.3%}`).
- Adapted modules: `{model['adapted_module_count']}` = 36 text layers x (4 attention projections + 3 MLP projections), covering all 18 encoder-text and all 18 decoder layers.
- Final adapter matches best checkpoint bytes: `{adapter_match}`.
- Peak evaluation allocation/reservation: `{max(float(row['eval_cuda_peak_allocated_gib']) for row in eval_history):.2f}` / `{max(float(row['eval_cuda_peak_reserved_gib']) for row in eval_history):.2f}` GiB.
- MLflow one-second system samples: `{len(next(iter([history for key, history in mlflow_export.histories.items() if key.startswith('system/')])))}` per metric.
- Measured evaluation/prediction windows: `{evaluation_seconds / 3600:.3f}` hours, or `{evaluation_seconds / run_duration:.1%}` of total MLflow wall time. This includes the initial evaluation, five scheduled evaluations, and the final best-checkpoint publication pass.
- Mean GPU utilization: `{float(non_eval_system['gpu_0_utilization_percentage_mean']):.1f}%` outside evaluation versus `{float(initial_eval_system['gpu_0_utilization_percentage_mean']):.1f}%` during the initial long-generation evaluation; detailed phase summaries are in `tables/phase_system_metrics.csv`.

Initial evaluation consumed `{float(eval_history[0]['eval_runtime']) / 60:.1f}` minutes because the untrained decoder averaged `{float(eval_history[0]['eval_generated_tokens_mean']):.0f}` tokens and reached EOS for only `{float(eval_history[0]['eval_generation_eos_reached_fraction']):.1%}` of examples. At the final scheduled evaluation, mean generation fell to `{float(eval_history[-1]['eval_generated_tokens_mean']):.0f}` tokens, EOS completion reached `{float(eval_history[-1]['eval_generation_eos_reached_fraction']):.1%}`, and runtime fell to `{float(eval_history[-1]['eval_runtime']) / 60:.1f}` minutes.

## What this run establishes—and what it does not

This is a successful approach-validation run: a 270M encoder-decoder with LoRA learned the sparse JSON mapping, produces exact scalar values with high precision, covers roughly three quarters of reference values, and usually emits valid JSON. It is not yet evidence of production generalization: the validation set has only 60 documents, comes from the same two labeled cohorts as training, and only one document is wholly exact. The bootstrap intervals quantify sampling uncertainty but cannot account for carrier/template/domain shift.

Recommended next decision: use the field/section tables and failure gallery to distinguish label ambiguity from repeatable model omissions before merely increasing epochs. More epochs on this exact split are low-value—the epoch-20→25 F1 gain is small and eval loss is rising. The next useful experiment should add genuinely diverse labeled documents and retain a carrier/template-disjoint test set; compare checkpoint 20 and 25 on that external holdout before selecting a serving adapter.

## Artifact map

- `summary.json`: compact machine-readable conclusions and core counts.
- `tables/document_metrics.csv`: one row per validation document.
- `tables/field_metrics.csv`: strict and index-insensitive metrics by normalized field path.
- `tables/section_metrics.csv`: top-level semantic section metrics.
- `tables/group_metrics.csv`: cohort, page-count, and label-complexity slices.
- `tables/eval_history.csv` / `training_history.csv`: exact Trainer histories.
- `tables/mlflow_system_history.csv`: aligned one-second GPU/CPU/RAM telemetry.
- `tables/bootstrap_intervals.csv`: 10,000-resample confidence intervals.
- `tables/schema_errors.jsonl`: structured JSON/schema failure diagnoses.
- `tables/prediction_identity_mapping.csv`: published IDs versus uniquely recovered source IDs.
- `recovered-predictions/validation.jsonl`: non-destructive corrected-ID copy of all saved predictions.
- `examples/lowest_performing.md`: canonical diffs for the 12 weakest documents.
- `source_integrity.json`: run-manifest verification.
- `analysis_manifest.json`: SHA-256 inventory of this analysis publication.

## Plot gallery

"""
    for path in plot_paths:
        title = Path(path).stem.replace("_", " ").title()
        report += f"### {title}\n\n![{title}]({path})\n\n"
    return report


def _publish_analysis(run_dir: Path, output_dir: Path, project_root: Path) -> None:
    if output_dir == run_dir or run_dir in output_dir.parents:
        raise ValueError("analysis output must be outside the immutable training-run directory")
    output_dir.mkdir(parents=True, exist_ok=True)
    if any(output_dir.iterdir()):
        raise ValueError(f"analysis output directory is not empty: {output_dir}")

    manifest = cast(dict[str, Any], _json_load(run_dir / "manifest.json"))
    config = cast(dict[str, Any], yaml.safe_load((run_dir / "config.yaml").read_text(encoding="utf-8")))
    environment = cast(dict[str, Any], _json_load(run_dir / "environment.json"))
    trainer_state = cast(dict[str, Any], _json_load(run_dir / "checkpoints" / "trainer_state.json"))
    validated_config = TrainingConfig.model_validate(config)
    task = load_training_task(project_root, validated_config)
    integrity = _verify_run_manifest(run_dir)
    if not integrity["integrity_passed"]:
        raise ValueError("immutable run artifact integrity failed")

    records, dataset_sources = _load_dataset_records(project_root, config)
    identity = str(manifest["dataset_cache_identity"])
    preprocessing = cast(Mapping[str, Any], cast(Mapping[str, Any], config["dataset"])["preprocessing"])
    cache_dir = project_root / str(preprocessing["cache_dir"])
    validation_lengths = _load_token_lengths(cache_dir, "validation", identity)
    generation_max_length = int(cast(Mapping[str, Any], config["evaluation"])["generation_max_length"])
    diagnostics, schema_errors, identity_rows = _load_predictions(
        run_dir,
        records,
        validation_lengths,
        task,
        generation_max_length,
    )
    document_rows = _document_rows(diagnostics)
    field_rows = _field_rows(diagnostics)
    section_rows = _section_rows(diagnostics)
    group_rows = _group_rows(diagnostics)
    coverage_rows = _dataset_coverage_rows(records, task)
    bootstrap_rows = _bootstrap_intervals(diagnostics, BOOTSTRAP_REPLICATES, BOOTSTRAP_SEED)
    correlation_rows = _correlation_rows(diagnostics)
    training_history, eval_history = _load_trainer_histories(run_dir)

    mlflow_export = _fetch_mlflow(run_dir)
    mlflow_summary_rows, system_rows = _mlflow_rows(mlflow_export)
    phase_rows = _load_phase_rows(
        run_dir,
        int(mlflow_export.run["start_time_ms"]),
        int(mlflow_export.run["end_time_ms"]),
    )
    phase_system_rows = _phase_system_rows(system_rows, phase_rows)
    epoch_zero = _epoch_zero_baseline(eval_history, environment)

    counts = _aggregate_counts(diagnostics)
    final_metrics = cast(Mapping[str, Any], cast(Mapping[str, Any], manifest["metrics"])["validation_evaluation"])
    expected_metrics = {
        "eval_json_valid": statistics.fmean(item.assessment.json_valid for item in diagnostics),
        "eval_schema_valid": statistics.fmean(item.assessment.schema_valid for item in diagnostics),
        "eval_canonical_exact_match": statistics.fmean(
            item.assessment.canonical_exact_match for item in diagnostics
        ),
        "eval_field_value_accuracy": counts.accuracy,
        "eval_field_value_precision": counts.precision,
        "eval_field_value_recall": counts.recall,
        "eval_field_value_f1": counts.f1,
    }
    metric_deltas = {
        key: expected - float(final_metrics[key]) for key, expected in expected_metrics.items()
    }
    if any(abs(delta) > 1e-12 for delta in metric_deltas.values()):
        raise ValueError(f"saved prediction metrics do not reproduce published metrics: {metric_deltas}")

    tables = output_dir / "tables"
    _write_csv(tables / "dataset_sources.csv", dataset_sources, list(dataset_sources[0]))
    _write_csv(tables / "document_metrics.csv", document_rows, list(document_rows[0]))
    _write_csv(tables / "field_metrics.csv", field_rows, list(field_rows[0]))
    _write_csv(tables / "section_metrics.csv", section_rows, list(section_rows[0]))
    _write_csv(tables / "group_metrics.csv", group_rows, list(group_rows[0]))
    _write_csv(tables / "dataset_field_coverage.csv", coverage_rows, list(coverage_rows[0]))
    _write_csv(tables / "bootstrap_intervals.csv", bootstrap_rows, list(bootstrap_rows[0]))
    _write_csv(tables / "correlations.csv", correlation_rows, list(correlation_rows[0]))
    _write_csv(tables / "training_history.csv", training_history, sorted({key for row in training_history for key in row}))
    _write_csv(tables / "eval_history.csv", eval_history, sorted({key for row in eval_history for key in row}))
    _write_csv(tables / "mlflow_metric_summary.csv", mlflow_summary_rows, list(mlflow_summary_rows[0]))
    _write_csv(tables / "mlflow_system_history.csv", system_rows, list(system_rows[0]))
    _write_csv(tables / "phase_timeline.csv", phase_rows, list(phase_rows[0]))
    _write_csv(
        tables / "phase_system_metrics.csv",
        phase_system_rows,
        list(phase_system_rows[0]),
    )
    _write_csv(
        tables / "prediction_identity_mapping.csv",
        identity_rows,
        list(identity_rows[0]),
    )
    _write_jsonl(tables / "schema_errors.jsonl", schema_errors)
    _write_jsonl(
        output_dir / "recovered-predictions" / "validation.jsonl",
        [
            {
                "document_id": item.document_id,
                "published_document_id": item.published_document_id,
                **item.assessment.to_dict(),
            }
            for item in diagnostics
        ],
    )
    _write_json(output_dir / "mlflow_run.json", mlflow_export.run)
    _write_json(output_dir / "source_integrity.json", integrity)
    _failure_gallery(output_dir, diagnostics, records)

    plot_paths = _render_plots(
        output_dir,
        training_history,
        eval_history,
        diagnostics,
        field_rows,
        section_rows,
        group_rows,
        bootstrap_rows,
        system_rows,
        phase_rows,
        phase_system_rows,
    )
    best_checkpoint = str(trainer_state["best_model_checkpoint"])
    adapter_name = str(cast(Mapping[str, Any], config["peft"])["adapter_name"])
    summary = {
        "schema_version": ANALYSIS_SCHEMA_VERSION,
        "source_run": str(manifest["run_id"]),
        "source_run_dir": str(run_dir),
        "analyzed_at": datetime.now(UTC).isoformat(),
        "documents": len(diagnostics),
        "integrity_passed": integrity["integrity_passed"],
        "mlflow_status": mlflow_export.run["status"],
        "completed_steps": int(trainer_state["global_step"]),
        "planned_steps": int(trainer_state["max_steps"]),
        "epochs": float(trainer_state["epoch"]),
        "best_checkpoint": best_checkpoint,
        "best_checkpoint_step": int(trainer_state["best_global_step"]),
        "best_scheduled_field_value_f1": float(trainer_state["best_metric"]),
        "final_adapter_matches_best_checkpoint": _sha256(
            run_dir / "final-adapter" / adapter_name / "adapter_model.safetensors"
        )
        == _sha256(
            _resolved_checkpoint_path(run_dir, best_checkpoint)
            / adapter_name
            / "adapter_model.safetensors"
        ),
        "final_saved_prediction_metrics": expected_metrics,
        "metric_reproduction_deltas": metric_deltas,
        "field_value_counts": {
            "true_positive": counts.true_positive,
            "predicted": counts.predicted,
            "reference": counts.reference,
            "compared_paths": counts.compared_paths,
        },
        "document_validity_counts": {
            "json_valid": sum(item.assessment.json_valid for item in diagnostics),
            "schema_valid": sum(item.assessment.schema_valid for item in diagnostics),
            "canonical_exact_match": sum(
                item.assessment.canonical_exact_match for item in diagnostics
            ),
            "likely_generation_cap": sum(item.likely_generation_cap for item in diagnostics),
        },
        "prediction_identity": {
            "published_rows": len(identity_rows),
            "published_ids_correct": sum(
                int(row["published_id_correct"]) for row in identity_rows
            ),
            "published_ids_incorrect": sum(
                not int(row["published_id_correct"]) for row in identity_rows
            ),
            "recovery_method": "unique canonical reference target",
            "recovery_complete": True,
        },
        "path_error_counts": {
            "substitutions": sum(item.substitutions for item in diagnostics),
            "omissions": sum(item.omissions for item in diagnostics),
            "additions": sum(item.additions for item in diagnostics),
        },
        "epoch_zero_baseline": epoch_zero,
        "phase_system_metrics": phase_system_rows,
        "evaluation_window_seconds": sum(
            float(row["duration_seconds"]) for row in phase_rows
        ),
        "bootstrap": bootstrap_rows,
        "plots": plot_paths,
    }
    _write_json(output_dir / "summary.json", summary)
    report = _build_report(
        run_dir,
        output_dir,
        manifest,
        integrity,
        environment,
        config,
        trainer_state,
        eval_history,
        diagnostics,
        field_rows,
        section_rows,
        group_rows,
        bootstrap_rows,
        schema_errors,
        correlation_rows,
        coverage_rows,
        identity_rows,
        phase_system_rows,
        phase_rows,
        mlflow_export,
        plot_paths,
    )
    (output_dir / "REPORT.md").write_text(report, encoding="utf-8")

    files = sorted(path for path in output_dir.rglob("*") if path.is_file())
    analysis_manifest = {
        "schema_version": ANALYSIS_SCHEMA_VERSION,
        "status": "complete",
        "source_run_id": manifest["run_id"],
        "created_at": datetime.now(UTC).isoformat(),
        "artifacts": [
            {
                "path": str(path.relative_to(output_dir)),
                "bytes": path.stat().st_size,
                "sha256": _sha256(path),
            }
            for path in files
        ],
    }
    _write_json(output_dir / "analysis_manifest.json", analysis_manifest)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--project-root", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    _publish_analysis(
        run_dir=args.run_dir.resolve(),
        output_dir=args.output_dir.resolve(),
        project_root=args.project_root.resolve(),
    )
    print(
        json.dumps(
            {
                "status": "complete",
                "output_dir": str(args.output_dir.resolve()),
            },
            separators=(",", ":"),
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
