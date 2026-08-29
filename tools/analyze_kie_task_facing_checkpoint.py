#!/usr/bin/env python3
# ruff: noqa: E501
"""Publish a granular audit of task-facing KIE checkpoint predictions."""

from __future__ import annotations

import argparse
import csv
import difflib
import hashlib
import json
import math
import random
import re
import shutil
import statistics
import tempfile
from collections import Counter, defaultdict
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, cast

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import pyarrow as pa
import pyarrow.ipc as ipc
import seaborn as sns
import yaml
from pydantic import ValidationError
from safetensors import safe_open

from document_ocr.atomic import atomic_write_json
from document_ocr.hashing import sha256_file
from document_ocr.training.config import TrainingConfig
from document_ocr.training.metrics import PredictionAssessment, assess_prediction
from document_ocr.training.tasks import TrainingTask, canonical_json, load_training_task

INDEX_PATTERN = re.compile(r"\[\d+\]")
PAGE_PATTERN = re.compile(r"(?m)^--- PAGE \d+ ---\s*$")
BOOTSTRAP_REPLICATES = 10_000
BOOTSTRAP_SEED = 424


@dataclass(frozen=True, slots=True)
class Record:
    document_id: str
    split: str
    raw_text: str
    target: dict[str, Any]
    source_corpus: str
    source_path: str
    page_count: int
    input_tokens: int
    target_tokens: int
    carrier_name: str
    carrier_family: str
    template_id: str
    template_seen_in_train: str
    container_count: int
    cargo_group_count: int
    package_count: int
    allocation_group_count: int
    target_leaves: int


@dataclass(frozen=True, slots=True)
class Diagnostic:
    record: Record
    assessment: PredictionAssessment
    true_positive: int
    predicted: int
    reference: int
    compared_paths: int
    index_true_positive: int
    schema_failure_class: str
    schema_failure_detail: str
    generated_characters: int
    reference_characters: int

    @property
    def precision(self) -> float:
        return self.true_positive / self.predicted if self.predicted else 0.0

    @property
    def recall(self) -> float:
        return self.true_positive / self.reference if self.reference else 0.0

    @property
    def f1(self) -> float:
        return _f1(self.precision, self.recall)

    @property
    def index_f1(self) -> float:
        precision = self.index_true_positive / self.predicted if self.predicted else 0.0
        recall = self.index_true_positive / self.reference if self.reference else 0.0
        return _f1(precision, recall)


def _f1(precision: float, recall: float) -> float:
    return 2 * precision * recall / (precision + recall) if precision + recall else 0.0


def _json_load(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _jsonl_load(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, start=1):
            if not line.strip():
                raise ValueError(f"blank JSONL row: {path}:{line_number}")
            value = json.loads(line)
            if not isinstance(value, dict):
                raise ValueError(f"non-object JSONL row: {path}:{line_number}")
            rows.append(cast(dict[str, Any], value))
    return rows


def _write_jsonl(path: Path, rows: Iterable[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as stream:
        for row in rows:
            stream.write(
                json.dumps(dict(row), ensure_ascii=False, sort_keys=True, separators=(",", ":"))
                + "\n"
            )


def _write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    if not rows:
        raise ValueError(f"cannot publish empty CSV: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = sorted({key for row in rows for key in row})
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, extrasaction="raise")
        writer.writeheader()
        writer.writerows(rows)


def _normalize_path(path: str) -> str:
    return INDEX_PATTERN.sub("[]", path)


def _section(path: str) -> str:
    prefix = "$.documentPatch."
    remainder = path[len(prefix) :] if path.startswith(prefix) else path
    return re.split(r"[.\[]", remainder, maxsplit=1)[0]


def _flatten(value: Any, path: str = "$.documentPatch") -> set[tuple[str, str]]:
    if isinstance(value, dict):
        result: set[tuple[str, str]] = set()
        for key in sorted(value):
            result.update(_flatten(value[key], f"{path}.{key}"))
        return result
    if isinstance(value, list):
        result = set()
        for index, child in enumerate(value):
            result.update(_flatten(child, f"{path}[{index}]"))
        return result
    return {(path, json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")))}


def _source_corpus(value: Any) -> str | None:
    if isinstance(value, dict):
        candidate = value.get("sourceCorpus")
        if isinstance(candidate, str) and candidate:
            return candidate
        for child in value.values():
            found = _source_corpus(child)
            if found is not None:
                return found
    elif isinstance(value, list):
        for child in value:
            found = _source_corpus(child)
            if found is not None:
                return found
    return None


def _load_token_lengths(cache_dir: Path, identity: str) -> dict[str, tuple[int, int]]:
    lengths: dict[str, tuple[int, int]] = {}
    for split in ("train", "validation"):
        paths = sorted(cache_dir.glob(f"{split}-{identity}_*.arrow"))
        if not paths:
            raise FileNotFoundError(f"token cache is absent for {split}: {identity}")
        for path in paths:
            with pa.memory_map(str(path), "r") as source:
                for batch in ipc.open_stream(source):
                    ids = batch.column("document_id")
                    inputs = batch.column("input_length")
                    targets = batch.column("target_length")
                    for index in range(batch.num_rows):
                        document_id = str(ids[index].as_py())
                        if document_id in lengths:
                            raise ValueError(f"duplicate token-cache row: {document_id}")
                        lengths[document_id] = (
                            int(inputs[index].as_py()),
                            int(targets[index].as_py()),
                        )
    return lengths


def _adapter_inventory(path: Path) -> dict[str, Any]:
    tensor_count = 0
    parameter_count = 0
    encoder_parameters = 0
    decoder_parameters = 0
    dtypes: Counter[str] = Counter()
    with safe_open(path, framework="pt", device="cpu") as stream:
        for key in stream.keys():  # noqa: SIM118 - safe_open is not iterable
            tensor = stream.get_tensor(key)
            parameters = tensor.numel()
            tensor_count += 1
            parameter_count += parameters
            dtypes[str(tensor.dtype)] += parameters
            if ".encoder." in key:
                encoder_parameters += parameters
            elif ".decoder." in key:
                decoder_parameters += parameters
    return {
        "path": str(path),
        "sha256": sha256_file(path),
        "bytes": path.stat().st_size,
        "tensor_count": tensor_count,
        "adapted_module_count": tensor_count // 2,
        "parameter_count": parameter_count,
        "encoder_parameters": encoder_parameters,
        "decoder_parameters": decoder_parameters,
        "parameters_by_dtype": dict(dtypes),
    }


def _load_eda_features(eda_dir: Path | None) -> dict[str, dict[str, str]]:
    if eda_dir is None:
        return {}
    path = eda_dir / "data" / "document-features.csv"
    with path.open(encoding="utf-8", newline="") as stream:
        return {row["document_id"]: row for row in csv.DictReader(stream)}


def _patch_counts(target: Mapping[str, Any]) -> tuple[int, int, int, int, str]:
    patch = cast(Mapping[str, Any], target["documentPatch"])
    containers = patch.get("containers", [])
    groups = patch.get("cargoGroups", [])
    packages = patch.get("cargoPackages", [])
    allocations = patch.get("cargoAllocationGroups", [])
    parties = patch.get("parties", {})
    carrier = parties.get("carrier", {}) if isinstance(parties, dict) else {}
    carrier_name = carrier.get("name", "") if isinstance(carrier, dict) else ""
    return (
        len(containers) if isinstance(containers, list) else 0,
        len(groups) if isinstance(groups, list) else 0,
        len(packages) if isinstance(packages, list) else 0,
        len(allocations) if isinstance(allocations, list) else 0,
        str(carrier_name),
    )


def _load_records(
    *,
    project_root: Path,
    run_dir: Path,
    config: TrainingConfig,
    task: TrainingTask,
    eda_dir: Path | None,
) -> dict[str, Record]:
    if config.dataset.source is None or config.dataset.partition is None:
        raise ValueError("task-facing analysis requires a runtime-partition dataset")
    source_path = project_root / config.dataset.source.path
    if sha256_file(source_path) != config.dataset.source.sha256:
        raise ValueError("configured dataset source SHA-256 differs from disk")
    rows = _jsonl_load(source_path)
    if len(rows) != config.dataset.source.records:
        raise ValueError("configured dataset record count differs from disk")
    report = cast(dict[str, Any], _json_load(run_dir / "dataset-report.json"))
    outputs = report["inspection"]["partition"]["outputs"]
    train_ids = set(outputs["train"]["document_ids"])
    validation_ids = set(outputs["validation"]["document_ids"])
    if train_ids & validation_ids or len(train_ids) != 1057 or len(validation_ids) != 100:
        raise ValueError("runtime partition identities are inconsistent")
    identity = str(report["cache_identity"])
    cache_dir = project_root / config.dataset.preprocessing.cache_dir
    token_lengths = _load_token_lengths(cache_dir, identity)
    features = _load_eda_features(eda_dir)

    lineage_path = source_path.parent / "lineage.jsonl"
    lineage = {str(row["documentId"]): row for row in _jsonl_load(lineage_path)}
    template_train = {
        features[document_id].get("template_proxy_id", "")
        for document_id in train_ids
        if document_id in features
        and features[document_id].get("template_proxy_id", "") not in {"", "<MISSING>"}
    }
    records: dict[str, Record] = {}
    for row in rows:
        document_id = str(row["documentId"])
        if document_id not in train_ids | validation_ids:
            raise ValueError(f"dataset row is absent from partition: {document_id}")
        raw_text = str(row["joinedRawText"])
        if hashlib.sha256(raw_text.encode()).hexdigest() != row["joinedRawTextSha256"]:
            raise ValueError(f"raw OCR hash mismatch: {document_id}")
        target = cast(dict[str, Any], row["target"])
        canonical = task.canonicalize(target)
        if canonical != target:
            raise ValueError(f"target is noncanonical: {document_id}")
        patch = cast(dict[str, Any], target["documentPatch"])
        leaf_count = len(_flatten(patch))
        container_count, group_count, package_count, allocation_count, carrier_name = _patch_counts(
            target
        )
        feature = features.get(document_id, {})
        template_id = feature.get("template_proxy_id", "<UNAVAILABLE>") or "<UNAVAILABLE>"
        if template_id in {"<MISSING>", "<UNAVAILABLE>"}:
            template_seen = "unknown"
        else:
            template_seen = "seen" if template_id in template_train else "unseen"
        corpus = feature.get("source_corpus", "") or _source_corpus(lineage[document_id])
        if corpus is None:
            raise ValueError(f"source corpus is absent from lineage: {document_id}")
        input_tokens, target_tokens = token_lengths[document_id]
        records[document_id] = Record(
            document_id=document_id,
            split="validation" if document_id in validation_ids else "train",
            raw_text=raw_text,
            target=target,
            source_corpus=corpus,
            source_path=feature.get("source_path", "<UNAVAILABLE>"),
            page_count=len(PAGE_PATTERN.findall(raw_text)),
            input_tokens=input_tokens,
            target_tokens=target_tokens,
            carrier_name=carrier_name or "<MISSING>",
            carrier_family=feature.get("carrier_family", "<UNAVAILABLE>"),
            template_id=template_id,
            template_seen_in_train=template_seen,
            container_count=container_count,
            cargo_group_count=group_count,
            package_count=package_count,
            allocation_group_count=allocation_count,
            target_leaves=leaf_count,
        )
    if set(records) != train_ids | validation_ids or set(token_lengths) != set(records):
        raise ValueError("dataset, partition, and token-cache identity sets differ")
    return records


def _schema_failure(text: str, task: TrainingTask) -> tuple[str, str, list[dict[str, str]]]:
    try:
        value = json.loads(text)
    except json.JSONDecodeError as error:
        return "invalid_json", f"{error.msg} at character {error.pos}", []
    if not isinstance(value, dict):
        return "json_non_object", type(value).__name__, []
    try:
        task.canonicalize(cast(dict[str, Any], value))
    except ValidationError as error:
        details = [
            {
                "location": ".".join(str(part) for part in item["loc"]),
                "type": str(item["type"]),
                "message": str(item["msg"]),
            }
            for item in error.errors(include_input=False, include_context=False, include_url=False)
        ]
        return (
            "schema_validation_error",
            "; ".join(sorted({row["type"] for row in details})),
            details,
        )
    except ValueError as error:
        return "schema_validation_error", str(error), []
    return "none", "", []


def _set_counts(
    predicted: Iterable[tuple[Any, ...]], reference: Iterable[tuple[Any, ...]]
) -> tuple[int, int, int, float, float, float]:
    predicted_set = set(predicted)
    reference_set = set(reference)
    true_positive = len(predicted_set & reference_set)
    precision = true_positive / len(predicted_set) if predicted_set else 0.0
    recall = true_positive / len(reference_set) if reference_set else 0.0
    return (
        true_positive,
        len(predicted_set),
        len(reference_set),
        precision,
        recall,
        _f1(precision, recall),
    )


def _index_true_positive(
    predicted: Iterable[tuple[str, str]], reference: Iterable[tuple[str, str]]
) -> int:
    predicted_counter = Counter((_normalize_path(path), value) for path, value in predicted)
    reference_counter = Counter((_normalize_path(path), value) for path, value in reference)
    return sum((predicted_counter & reference_counter).values())


def _load_diagnostics(
    *,
    evaluation_dir: Path,
    records: Mapping[str, Record],
    task: TrainingTask,
) -> tuple[list[Diagnostic], list[dict[str, Any]]]:
    prediction_rows = _jsonl_load(evaluation_dir / "predictions" / "validation.jsonl")
    diagnostics: list[Diagnostic] = []
    schema_rows: list[dict[str, Any]] = []
    seen: set[str] = set()
    for row in prediction_rows:
        document_id = str(row["document_id"])
        record = records.get(document_id)
        if record is None or record.split != "validation" or document_id in seen:
            raise ValueError(f"prediction identity is invalid: {document_id}")
        seen.add(document_id)
        expected_reference = canonical_json(task.canonicalize(record.target))
        if row["reference_text"] != expected_reference:
            raise ValueError(f"prediction reference drift: {document_id}")
        assessment = assess_prediction(str(row["generated_text"]), expected_reference, task)
        for key in ("json_valid", "schema_valid", "canonical_exact_match"):
            if bool(row[key]) != bool(getattr(assessment, key)):
                raise ValueError(f"persisted prediction flag drift: {document_id}:{key}")
        predicted_paths = {path for path, _ in assessment.predicted_field_values}
        reference_paths = {path for path, _ in assessment.reference_field_values}
        failure_class, failure_detail, errors = _schema_failure(assessment.generated_text, task)
        if assessment.schema_valid != (failure_class == "none"):
            raise ValueError(f"schema diagnosis disagrees with metric: {document_id}")
        if failure_class != "none":
            schema_rows.append(
                {
                    "document_id": document_id,
                    "failure_class": failure_class,
                    "failure_detail": failure_detail,
                    "errors": errors,
                }
            )
        diagnostics.append(
            Diagnostic(
                record=record,
                assessment=assessment,
                true_positive=len(
                    assessment.predicted_field_values & assessment.reference_field_values
                ),
                predicted=len(assessment.predicted_field_values),
                reference=len(assessment.reference_field_values),
                compared_paths=len(predicted_paths | reference_paths),
                index_true_positive=_index_true_positive(
                    assessment.predicted_field_values,
                    assessment.reference_field_values,
                ),
                schema_failure_class=failure_class,
                schema_failure_detail=failure_detail,
                generated_characters=len(assessment.generated_text),
                reference_characters=len(expected_reference),
            )
        )
    expected_ids = {
        document_id for document_id, record in records.items() if record.split == "validation"
    }
    if seen != expected_ids:
        raise ValueError("prediction coverage differs from the validation partition")
    return diagnostics, schema_rows


def _grounding_applicable(path: str, value: str | None) -> bool:
    if value is None or path.endswith(
        (".groupId", ".packageId", ".typeCategory", ".unit", ".coverage")
    ):
        return False
    return not path.endswith((".paymentArrangement", ".negotiability"))


def _decoded_scalar(value: str | None) -> Any:
    return json.loads(value) if value is not None else None


def _normalized_grounding(value: str | None, raw_text: str) -> bool | None:
    if value is None:
        return None
    decoded = _decoded_scalar(value)
    if isinstance(decoded, bool) or decoded is None:
        return None
    rendered = str(decoded).casefold()
    needle = re.sub(r"[^a-z0-9]+", "", rendered)
    haystack = re.sub(r"[^a-z0-9]+", "", raw_text.casefold())
    if len(needle) < 2:
        return None
    return needle in haystack


def _similarity(reference: str | None, predicted: str | None) -> float | None:
    if reference is None or predicted is None:
        return None
    left = _decoded_scalar(reference)
    right = _decoded_scalar(predicted)
    if (
        isinstance(left, (int, float))
        and not isinstance(left, bool)
        and isinstance(right, (int, float))
        and not isinstance(right, bool)
    ):
        denominator = max(abs(float(left)), abs(float(right)), 1.0)
        return max(0.0, 1.0 - abs(float(left) - float(right)) / denominator)
    return difflib.SequenceMatcher(None, str(left).casefold(), str(right).casefold()).ratio()


def _error_row(
    *,
    diagnostic: Diagnostic,
    error_type: str,
    reference_path: str | None,
    reference_value: str | None,
    predicted_path: str | None,
    predicted_value: str | None,
) -> dict[str, Any]:
    path = reference_path or predicted_path
    assert path is not None
    applicable = _grounding_applicable(path, predicted_value)
    return {
        "document_id": diagnostic.record.document_id,
        "error_type": error_type,
        "section": _section(path),
        "field_path": _normalize_path(path),
        "reference_path": reference_path or "",
        "reference_value": reference_value or "",
        "predicted_path": predicted_path or "",
        "predicted_value": predicted_value or "",
        "value_similarity": _similarity(reference_value, predicted_value),
        "predicted_ocr_grounded": (
            _normalized_grounding(predicted_value, diagnostic.record.raw_text)
            if applicable
            else None
        ),
        "reference_ocr_grounded": (
            _normalized_grounding(reference_value, diagnostic.record.raw_text)
            if _grounding_applicable(path, reference_value)
            else None
        ),
    }


def _classify_errors(diagnostic: Diagnostic) -> list[dict[str, Any]]:
    predicted = dict(diagnostic.assessment.predicted_field_values)
    reference = dict(diagnostic.assessment.reference_field_values)
    exact_paths = {
        path for path in predicted.keys() & reference.keys() if predicted[path] == reference[path]
    }
    remaining_predicted = {
        path: value for path, value in predicted.items() if path not in exact_paths
    }
    remaining_reference = {
        path: value for path, value in reference.items() if path not in exact_paths
    }
    rows: list[dict[str, Any]] = []

    for reference_path in sorted(tuple(remaining_reference)):
        reference_value = remaining_reference[reference_path]
        candidates = [
            predicted_path
            for predicted_path, predicted_value in remaining_predicted.items()
            if predicted_path != reference_path
            and _normalize_path(predicted_path) == _normalize_path(reference_path)
            and predicted_value == reference_value
        ]
        if candidates:
            predicted_path = sorted(candidates)[0]
            rows.append(
                _error_row(
                    diagnostic=diagnostic,
                    error_type="right_value_wrong_index",
                    reference_path=reference_path,
                    reference_value=reference_value,
                    predicted_path=predicted_path,
                    predicted_value=remaining_predicted[predicted_path],
                )
            )
            del remaining_reference[reference_path]
            del remaining_predicted[predicted_path]

    for path in sorted(remaining_reference.keys() & remaining_predicted.keys()):
        rows.append(
            _error_row(
                diagnostic=diagnostic,
                error_type="substitution",
                reference_path=path,
                reference_value=remaining_reference[path],
                predicted_path=path,
                predicted_value=remaining_predicted[path],
            )
        )
        del remaining_reference[path]
        del remaining_predicted[path]

    for reference_path in sorted(tuple(remaining_reference)):
        candidates = [
            predicted_path
            for predicted_path in remaining_predicted
            if _normalize_path(predicted_path) == _normalize_path(reference_path)
        ]
        if candidates:
            reference_value = remaining_reference[reference_path]
            predicted_path = max(
                candidates,
                key=lambda value: _similarity(reference_value, remaining_predicted[value]) or 0.0,
            )
            rows.append(
                _error_row(
                    diagnostic=diagnostic,
                    error_type="list_alignment_or_value_mismatch",
                    reference_path=reference_path,
                    reference_value=reference_value,
                    predicted_path=predicted_path,
                    predicted_value=remaining_predicted[predicted_path],
                )
            )
            del remaining_reference[reference_path]
            del remaining_predicted[predicted_path]

    for path, value in sorted(remaining_reference.items()):
        rows.append(
            _error_row(
                diagnostic=diagnostic,
                error_type="omission",
                reference_path=path,
                reference_value=value,
                predicted_path=None,
                predicted_value=None,
            )
        )
    for path, value in sorted(remaining_predicted.items()):
        rows.append(
            _error_row(
                diagnostic=diagnostic,
                error_type="addition",
                reference_path=None,
                reference_value=None,
                predicted_path=path,
                predicted_value=value,
            )
        )
    return rows


def _aggregate_diagnostics(values: Sequence[Diagnostic]) -> dict[str, float | int]:
    true_positive = sum(row.true_positive for row in values)
    predicted = sum(row.predicted for row in values)
    reference = sum(row.reference for row in values)
    precision = true_positive / predicted if predicted else 0.0
    recall = true_positive / reference if reference else 0.0
    return {
        "documents": len(values),
        "true_positive": true_positive,
        "predicted": predicted,
        "reference": reference,
        "precision": precision,
        "recall": recall,
        "f1": _f1(precision, recall),
        "macro_document_f1": statistics.fmean(row.f1 for row in values),
        "json_valid": statistics.fmean(row.assessment.json_valid for row in values),
        "schema_valid": statistics.fmean(row.assessment.schema_valid for row in values),
        "canonical_exact_match": statistics.fmean(
            row.assessment.canonical_exact_match for row in values
        ),
    }


def _document_rows(diagnostics: Sequence[Diagnostic]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for item in diagnostics:
        relation = _set_counts(
            item.assessment.predicted_cargo_relation_facts,
            item.assessment.reference_cargo_relation_facts,
        )
        category = _set_counts(
            item.assessment.predicted_category_values,
            item.assessment.reference_category_values,
        )
        rows.append(
            {
                "document_id": item.record.document_id,
                "source_corpus": item.record.source_corpus,
                "source_path": item.record.source_path,
                "carrier_name": item.record.carrier_name,
                "carrier_family": item.record.carrier_family,
                "template_id": item.record.template_id,
                "template_seen_in_train": item.record.template_seen_in_train,
                "page_count": item.record.page_count,
                "input_tokens": item.record.input_tokens,
                "target_tokens": item.record.target_tokens,
                "target_leaves": item.record.target_leaves,
                "container_count": item.record.container_count,
                "cargo_group_count": item.record.cargo_group_count,
                "package_count": item.record.package_count,
                "allocation_group_count": item.record.allocation_group_count,
                "json_valid": int(item.assessment.json_valid),
                "schema_valid": int(item.assessment.schema_valid),
                "canonical_exact_match": int(item.assessment.canonical_exact_match),
                "field_precision": item.precision,
                "field_recall": item.recall,
                "field_f1": item.f1,
                "index_insensitive_f1": item.index_f1,
                "relation_f1": relation[-1],
                "category_f1": category[-1],
                "generated_characters": item.generated_characters,
                "reference_characters": item.reference_characters,
                "generated_reference_character_ratio": (
                    item.generated_characters / item.reference_characters
                    if item.reference_characters
                    else 0.0
                ),
                "schema_failure_class": item.schema_failure_class,
            }
        )
    return sorted(rows, key=lambda row: str(row["document_id"]))


def _field_rows(
    diagnostics: Sequence[Diagnostic], records: Mapping[str, Record]
) -> list[dict[str, Any]]:
    strict: dict[str, Counter[str]] = defaultdict(Counter)
    unordered: dict[str, Counter[str]] = defaultdict(Counter)
    document_states: dict[str, Counter[str]] = defaultdict(Counter)
    train_values: dict[str, Counter[str]] = defaultdict(Counter)
    train_documents: Counter[str] = Counter()
    validation_novel_values: Counter[str] = Counter()
    normalized_by_document: dict[str, list[tuple[str, str]]] = {}
    for record in records.values():
        fields = _flatten(cast(dict[str, Any], record.target["documentPatch"]))
        normalized = [(_normalize_path(path), value) for path, value in fields]
        normalized_by_document[record.document_id] = normalized
        if record.split == "train":
            for field, value in normalized:
                train_values[field][value] += 1
            train_documents.update({field for field, _ in normalized})
    for record in records.values():
        if record.split != "validation":
            continue
        for field, value in normalized_by_document[record.document_id]:
            if not train_values[field][value]:
                validation_novel_values[field] += 1
    for item in diagnostics:
        predicted = item.assessment.predicted_field_values
        reference = item.assessment.reference_field_values
        for path, _ in predicted:
            strict[_normalize_path(path)]["predicted"] += 1
        for path, _ in reference:
            strict[_normalize_path(path)]["reference"] += 1
        for path, _ in predicted & reference:
            strict[_normalize_path(path)]["true_positive"] += 1
        predicted_counter = Counter((_normalize_path(path), value) for path, value in predicted)
        reference_counter = Counter((_normalize_path(path), value) for path, value in reference)
        intersection = predicted_counter & reference_counter
        for (field, _), count in predicted_counter.items():
            unordered[field]["predicted"] += count
        for (field, _), count in reference_counter.items():
            unordered[field]["reference"] += count
        for (field, _), count in intersection.items():
            unordered[field]["true_positive"] += count
        for field in {name for name, _ in reference_counter}:
            ref_values = Counter(
                {
                    value: count
                    for (name, value), count in reference_counter.items()
                    if name == field
                }
            )
            pred_values = Counter(
                {
                    value: count
                    for (name, value), count in predicted_counter.items()
                    if name == field
                }
            )
            document_states[field]["supported"] += 1
            if ref_values == pred_values:
                document_states[field]["exact"] += 1
    rows: list[dict[str, Any]] = []
    for field in sorted(strict):
        values = strict[field]
        precision = values["true_positive"] / values["predicted"] if values["predicted"] else 0
        recall = values["true_positive"] / values["reference"] if values["reference"] else 0
        index_precision = (
            unordered[field]["true_positive"] / unordered[field]["predicted"]
            if unordered[field]["predicted"]
            else 0
        )
        index_recall = (
            unordered[field]["true_positive"] / unordered[field]["reference"]
            if unordered[field]["reference"]
            else 0
        )
        rows.append(
            {
                "field_path": field,
                "section": _section(field),
                "train_documents": train_documents[field],
                "train_values": sum(train_values[field].values()),
                "train_unique_values": len(train_values[field]),
                "validation_values": values["reference"],
                "validation_novel_values": validation_novel_values[field],
                "validation_novel_value_fraction": (
                    validation_novel_values[field] / values["reference"]
                    if values["reference"]
                    else 0
                ),
                "predicted_values": values["predicted"],
                "true_positive_values": values["true_positive"],
                "false_positive_values": values["predicted"] - values["true_positive"],
                "false_negative_values": values["reference"] - values["true_positive"],
                "precision": precision,
                "recall": recall,
                "f1": _f1(precision, recall),
                "index_insensitive_f1": _f1(index_precision, index_recall),
                "index_penalty": _f1(index_precision, index_recall) - _f1(precision, recall),
                "supported_documents": document_states[field]["supported"],
                "whole_field_exact_rate": (
                    document_states[field]["exact"] / document_states[field]["supported"]
                    if document_states[field]["supported"]
                    else 0
                ),
            }
        )
    return rows


def _field_error_profile_rows(
    errors: Sequence[Mapping[str, Any]], fields: Sequence[Mapping[str, Any]]
) -> list[dict[str, Any]]:
    grouped: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for row in errors:
        grouped[str(row["field_path"])].append(row)
    field_by_path = {str(row["field_path"]): row for row in fields}
    profiles: list[dict[str, Any]] = []
    for field_path, rows in sorted(grouped.items()):
        error_types = Counter(str(row["error_type"]) for row in rows)
        similarities = [
            float(row["value_similarity"]) for row in rows if row["value_similarity"] is not None
        ]
        metric = field_by_path[field_path]
        profiles.append(
            {
                "field_path": field_path,
                "section": metric["section"],
                "validation_values": metric["validation_values"],
                "f1": metric["f1"],
                "index_insensitive_f1": metric["index_insensitive_f1"],
                "errors": len(rows),
                "omissions": error_types["omission"],
                "additions": error_types["addition"],
                "substitutions": error_types["substitution"],
                "right_value_wrong_index": error_types["right_value_wrong_index"],
                "list_alignment_or_value_mismatch": error_types["list_alignment_or_value_mismatch"],
                "mean_value_similarity": (statistics.fmean(similarities) if similarities else None),
                "median_value_similarity": (
                    statistics.median(similarities) if similarities else None
                ),
                "near_match_errors": sum(value >= 0.85 for value in similarities),
                "predicted_ocr_substring_matches": sum(
                    row["predicted_ocr_grounded"] is True for row in rows
                ),
                "reference_ocr_substring_matches": sum(
                    row["reference_ocr_grounded"] is True for row in rows
                ),
            }
        )
    return profiles


def _section_rows(diagnostics: Sequence[Diagnostic]) -> list[dict[str, Any]]:
    grouped: dict[str, list[tuple[set[tuple[str, str]], set[tuple[str, str]]]]] = defaultdict(list)
    for item in diagnostics:
        predicted = set(item.assessment.predicted_field_values)
        reference = set(item.assessment.reference_field_values)
        for section in {_section(path) for path, _ in predicted | reference}:
            grouped[section].append(
                (
                    {(path, value) for path, value in predicted if _section(path) == section},
                    {(path, value) for path, value in reference if _section(path) == section},
                )
            )
    rows: list[dict[str, Any]] = []
    for section, pairs in sorted(grouped.items()):
        true_positive = sum(len(predicted & reference) for predicted, reference in pairs)
        predicted_total = sum(len(predicted) for predicted, _ in pairs)
        reference_total = sum(len(reference) for _, reference in pairs)
        precision = true_positive / predicted_total if predicted_total else 0
        recall = true_positive / reference_total if reference_total else 0
        rows.append(
            {
                "section": section,
                "documents": len(pairs),
                "reference_values": reference_total,
                "predicted_values": predicted_total,
                "true_positive_values": true_positive,
                "precision": precision,
                "recall": recall,
                "f1": _f1(precision, recall),
                "whole_section_exact_rate": statistics.fmean(
                    predicted == reference for predicted, reference in pairs
                ),
            }
        )
    return rows


def _schema_issue_rows(schema_rows: Sequence[Mapping[str, Any]]) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    for schema_row in schema_rows:
        errors = schema_row["errors"]
        if errors:
            for error in errors:
                rows.append(
                    {
                        "document_id": str(schema_row["document_id"]),
                        "failure_class": str(schema_row["failure_class"]),
                        "failure_detail": str(schema_row["failure_detail"]),
                        "location": str(error["location"]),
                        "issue_type": str(error["type"]),
                        "message": str(error["message"]),
                    }
                )
        else:
            rows.append(
                {
                    "document_id": str(schema_row["document_id"]),
                    "failure_class": str(schema_row["failure_class"]),
                    "failure_detail": str(schema_row["failure_detail"]),
                    "location": "",
                    "issue_type": str(schema_row["failure_class"]),
                    "message": str(schema_row["failure_detail"]),
                }
            )
    return rows


def _fact_rows(diagnostics: Sequence[Diagnostic], *, category: bool) -> list[dict[str, Any]]:
    names = set()
    for item in diagnostics:
        predicted = (
            item.assessment.predicted_category_values
            if category
            else item.assessment.predicted_cargo_relation_facts
        )
        reference = (
            item.assessment.reference_category_values
            if category
            else item.assessment.reference_cargo_relation_facts
        )
        names.update(fact[0] for fact in predicted | reference)
    rows: list[dict[str, Any]] = []
    for name in sorted(names):
        true_positive = predicted_total = reference_total = supported = exact = 0
        for item in diagnostics:
            predicted_source = (
                item.assessment.predicted_category_values
                if category
                else item.assessment.predicted_cargo_relation_facts
            )
            reference_source = (
                item.assessment.reference_category_values
                if category
                else item.assessment.reference_cargo_relation_facts
            )
            predicted = {fact for fact in predicted_source if fact[0] == name}
            reference = {fact for fact in reference_source if fact[0] == name}
            true_positive += len(predicted & reference)
            predicted_total += len(predicted)
            reference_total += len(reference)
            if reference:
                supported += 1
                exact += int(predicted == reference)
        precision = true_positive / predicted_total if predicted_total else 0
        recall = true_positive / reference_total if reference_total else 0
        rows.append(
            {
                "fact_type": name,
                "reference_facts": reference_total,
                "predicted_facts": predicted_total,
                "true_positive_facts": true_positive,
                "precision": precision,
                "recall": recall,
                "f1": _f1(precision, recall),
                "supported_documents": supported,
                "exact_document_rate": exact / supported if supported else 0,
            }
        )
    return rows


def _category_token_rows(
    diagnostics: Sequence[Diagnostic], records: Mapping[str, Record]
) -> list[dict[str, Any]]:
    train_counter: Counter[tuple[str, str]] = Counter()
    for record in records.values():
        if record.split != "train":
            continue
        patch = cast(dict[str, Any], record.target["documentPatch"])
        for container in patch.get("containers", []):
            if "typeCategory" in container:
                train_counter[("container_type", container["typeCategory"])] += 1
        for package in patch.get("cargoPackages", []):
            if "typeCategory" in package:
                train_counter[("package_type", package["typeCategory"])] += 1
    counts: dict[tuple[str, str], Counter[str]] = defaultdict(Counter)
    confusions: list[dict[str, str]] = []
    for item in diagnostics:
        predicted_by_anchor = {
            fact[:-1]: fact[-1] for fact in item.assessment.predicted_category_values
        }
        reference_by_anchor = {
            fact[:-1]: fact[-1] for fact in item.assessment.reference_category_values
        }
        for anchor, token in reference_by_anchor.items():
            key = (anchor[0], token)
            counts[key]["reference"] += 1
            predicted = predicted_by_anchor.get(anchor)
            if predicted == token:
                counts[key]["true_positive"] += 1
            elif predicted is not None:
                confusions.append(
                    {
                        "document_id": item.record.document_id,
                        "category_type": anchor[0],
                        "reference_token": token,
                        "predicted_token": predicted,
                    }
                )
        for anchor, token in predicted_by_anchor.items():
            counts[(anchor[0], token)]["predicted"] += 1
    rows: list[dict[str, Any]] = []
    for (category_type, token), values in sorted(counts.items()):
        precision = values["true_positive"] / values["predicted"] if values["predicted"] else 0
        recall = values["true_positive"] / values["reference"] if values["reference"] else 0
        rows.append(
            {
                "category_type": category_type,
                "token": token,
                "train_facts": train_counter[(category_type, token)],
                "validation_references": values["reference"],
                "predictions": values["predicted"],
                "true_positives": values["true_positive"],
                "precision": precision,
                "recall": recall,
                "f1": _f1(precision, recall),
            }
        )
    return rows + [
        {
            "category_type": "confusion",
            "token": f"{row['reference_token']} -> {row['predicted_token']}",
            "train_facts": 0,
            "validation_references": 1,
            "predictions": 1,
            "true_positives": 0,
            "precision": 0,
            "recall": 0,
            "f1": 0,
        }
        for row in confusions
    ]


def _group_rows(diagnostics: Sequence[Diagnostic]) -> list[dict[str, Any]]:
    definitions: dict[str, dict[str, list[Diagnostic]]] = defaultdict(lambda: defaultdict(list))
    for item in diagnostics:
        record = item.record
        definitions["source_corpus"][record.source_corpus].append(item)
        definitions["template_coverage"][record.template_seen_in_train].append(item)
        definitions["page_count"][
            "1"
            if record.page_count == 1
            else "2"
            if record.page_count == 2
            else "3-4"
            if record.page_count <= 4
            else "5+"
        ].append(item)
        definitions["target_complexity"][
            "<40" if record.target_leaves < 40 else "40-79" if record.target_leaves < 80 else "80+"
        ].append(item)
        definitions["containers"][
            "0" if record.container_count == 0 else "1" if record.container_count == 1 else "2+"
        ].append(item)
        definitions["cargo_groups"][
            "0" if record.cargo_group_count == 0 else "1" if record.cargo_group_count == 1 else "2+"
        ].append(item)
        definitions["allocations"]["present" if record.allocation_group_count else "absent"].append(
            item
        )
    rows: list[dict[str, Any]] = []
    for group_type, values in sorted(definitions.items()):
        for group, members in sorted(values.items()):
            rows.append(
                {"group_type": group_type, "group": group, **_aggregate_diagnostics(members)}
            )
    return rows


def _dataset_complexity_rows(records: Mapping[str, Record]) -> list[dict[str, Any]]:
    predicates = {
        "5+ pages": lambda row: row.page_count >= 5,
        "multiple cargo groups": lambda row: row.cargo_group_count > 1,
        "multiple containers": lambda row: row.container_count > 1,
        "multiple package facts": lambda row: row.package_count > 1,
        ">100 target leaves": lambda row: row.target_leaves > 100,
        "multi-goods + multi-container": lambda row: (
            row.cargo_group_count > 1 and row.container_count > 1
        ),
        "multi-goods + allocations": lambda row: (
            row.cargo_group_count > 1 and row.allocation_group_count > 0
        ),
        "multi-goods + multi-package": lambda row: (
            row.cargo_group_count > 1 and row.package_count > 1
        ),
    }
    rows: list[dict[str, Any]] = []
    for split in ("train", "validation"):
        members = [row for row in records.values() if row.split == split]
        for feature, predicate in predicates.items():
            matching = sum(predicate(row) for row in members)
            rows.append(
                {
                    "split": split,
                    "feature": feature,
                    "matching_documents": matching,
                    "documents": len(members),
                    "fraction": matching / len(members),
                }
            )
    return rows


def _cargo_signature_rows(records: Mapping[str, Record]) -> list[dict[str, Any]]:
    train_signatures = Counter(
        (
            row.cargo_group_count,
            row.package_count,
            row.container_count,
            row.allocation_group_count,
        )
        for row in records.values()
        if row.split == "train"
    )
    validation_signatures = Counter(
        (
            row.cargo_group_count,
            row.package_count,
            row.container_count,
            row.allocation_group_count,
        )
        for row in records.values()
        if row.split == "validation"
    )
    signatures = sorted(train_signatures.keys() | validation_signatures.keys())
    return [
        {
            "signature": f"g{group_count}/p{package_count}/c{container_count}/a{allocation_count}",
            "cargo_groups": group_count,
            "packages": package_count,
            "containers": container_count,
            "allocations": allocation_count,
            "train_documents": train_signatures[signature],
            "validation_documents": validation_signatures[signature],
            "seen_in_train": bool(train_signatures[signature]),
        }
        for signature in signatures
        for group_count, package_count, container_count, allocation_count in [signature]
    ]


def _extended_cargo_signature_rows(records: Mapping[str, Record]) -> list[dict[str, Any]]:
    def signature(record: Record) -> tuple[int, int, int, int, int, tuple[str, ...]]:
        patch = cast(dict[str, Any], record.target["documentPatch"])
        allocation_groups = patch.get("cargoAllocationGroups", [])
        allocation_rows = sum(
            len(group.get("allocations", []))
            for group in allocation_groups
            if isinstance(group, dict) and isinstance(group.get("allocations", []), list)
        )
        coverage_modes = tuple(
            sorted(
                str(group.get("coverage", "<MISSING>"))
                for group in allocation_groups
                if isinstance(group, dict)
            )
        )
        return (
            record.cargo_group_count,
            record.package_count,
            record.container_count,
            record.allocation_group_count,
            allocation_rows,
            coverage_modes,
        )

    train_signatures = Counter(signature(row) for row in records.values() if row.split == "train")
    validation_signatures = Counter(
        signature(row) for row in records.values() if row.split == "validation"
    )
    signatures = sorted(train_signatures.keys() | validation_signatures.keys())
    return [
        {
            "signature": (
                f"g{group_count}/p{package_count}/c{container_count}/a{allocation_count}"
                f"/rows{allocation_rows}/coverage:{'|'.join(coverage_modes)}"
            ),
            "cargo_groups": group_count,
            "packages": package_count,
            "containers": container_count,
            "allocation_groups": allocation_count,
            "allocation_rows": allocation_rows,
            "coverage_modes": "|".join(coverage_modes),
            "train_documents": train_signatures[item],
            "validation_documents": validation_signatures[item],
            "seen_in_train": bool(train_signatures[item]),
        }
        for item in signatures
        for group_count, package_count, container_count, allocation_count, allocation_rows, coverage_modes in [
            item
        ]
    ]


def _history_rows(run_dir: Path) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    events = _jsonl_load(run_dir / "logs" / "events.jsonl")
    training: list[dict[str, Any]] = []
    evaluation: list[dict[str, Any]] = []
    for event in events:
        if event.get("event") != "trainer_log" or not isinstance(event.get("metrics"), dict):
            continue
        metrics = cast(dict[str, Any], event["metrics"])
        row = {
            "step": int(event["global_step"]),
            "epoch": float(event["epoch"]),
            **metrics,
        }
        if "eval_field_value_f1" in metrics:
            evaluation.append(row)
        elif "loss" in metrics:
            training.append(row)
    if len(evaluation) != 5 or training[-1]["step"] != 1125:
        raise ValueError("training history does not match the interrupted run contract")
    return training, evaluation


def _bootstrap(diagnostics: Sequence[Diagnostic]) -> list[dict[str, Any]]:
    rng = random.Random(BOOTSTRAP_SEED)
    values: list[float] = []
    for _ in range(BOOTSTRAP_REPLICATES):
        sample = [diagnostics[rng.randrange(len(diagnostics))] for _ in diagnostics]
        values.append(float(_aggregate_diagnostics(sample)["f1"]))
    values.sort()
    return [
        {
            "metric": "field_value_f1",
            "point": _aggregate_diagnostics(diagnostics)["f1"],
            "lower_95": values[249],
            "upper_95": values[9749],
            "replicates": BOOTSTRAP_REPLICATES,
            "seed": BOOTSTRAP_SEED,
        }
    ]


def _plot_bar(
    frame: pd.DataFrame,
    *,
    x: str,
    y: str,
    title: str,
    path: Path,
    horizontal: bool = False,
    hue: str | None = None,
) -> None:
    plt.figure(figsize=(12, max(6, len(frame) * 0.32) if horizontal else 7))
    if horizontal:
        sns.barplot(data=frame, x=y, y=x, hue=hue, errorbar=None)
    else:
        sns.barplot(data=frame, x=x, y=y, hue=hue, errorbar=None)
        plt.xticks(rotation=35, ha="right")
    plt.title(title)
    plt.grid(axis="x" if horizontal else "y", alpha=0.25)
    plt.tight_layout()
    plt.savefig(path, dpi=170)
    plt.close()


def _render_plots(
    *,
    output_dir: Path,
    training: Sequence[Mapping[str, Any]],
    evaluation: Sequence[Mapping[str, Any]],
    documents: Sequence[Mapping[str, Any]],
    fields: Sequence[Mapping[str, Any]],
    sections: Sequence[Mapping[str, Any]],
    relations: Sequence[Mapping[str, Any]],
    categories: Sequence[Mapping[str, Any]],
    errors: Sequence[Mapping[str, Any]],
    groups: Sequence[Mapping[str, Any]],
    dataset_complexity: Sequence[Mapping[str, Any]],
    bootstrap: Sequence[Mapping[str, Any]],
    prior_eval_path: Path | None,
) -> list[str]:
    plots = output_dir / "plots"
    plots.mkdir(parents=True)
    sns.set_theme(style="whitegrid", context="notebook")
    training_df = pd.DataFrame(training)
    eval_df = pd.DataFrame(evaluation)
    document_df = pd.DataFrame(documents)
    field_df = pd.DataFrame(fields)
    section_df = pd.DataFrame(sections)
    relation_df = pd.DataFrame(relations)
    category_df = pd.DataFrame(categories)
    error_df = pd.DataFrame(errors)
    group_df = pd.DataFrame(groups)
    complexity_df = pd.DataFrame(dataset_complexity)
    paths: list[str] = []

    def save(name: str) -> Path:
        paths.append(f"plots/{name}.png")
        return plots / f"{name}.png"

    plt.figure(figsize=(12, 7))
    sns.lineplot(data=training_df, x="step", y="loss", label="window loss")
    sns.lineplot(data=training_df, x="step", y="train_cumulative_loss", label="cumulative loss")
    plt.yscale("log")
    plt.title("Training loss: memorization continues after generated metrics slow")
    plt.tight_layout()
    plt.savefig(save("01_training_loss_log"), dpi=170)
    plt.close()

    metric_columns = [
        "eval_field_value_precision",
        "eval_field_value_recall",
        "eval_field_value_f1",
    ]
    melted = eval_df.melt(
        id_vars=["epoch"], value_vars=metric_columns, var_name="metric", value_name="value"
    )
    plt.figure(figsize=(11, 7))
    sns.lineplot(data=melted, x="epoch", y="value", hue="metric", marker="o")
    plt.ylim(0, 1)
    plt.title("Exact field/value progression")
    plt.tight_layout()
    plt.savefig(save("02_eval_field_progression"), dpi=170)
    plt.close()

    melted = eval_df.melt(
        id_vars=["epoch"],
        value_vars=["eval_cargo_relation_f1", "eval_category_value_f1", "eval_field_value_f1"],
        var_name="metric",
        value_name="value",
    )
    plt.figure(figsize=(11, 7))
    sns.lineplot(data=melted, x="epoch", y="value", hue="metric", marker="o")
    plt.ylim(0, 1)
    plt.title("Cargo relations and readable categories")
    plt.tight_layout()
    plt.savefig(save("03_relation_category_progression"), dpi=170)
    plt.close()

    melted = eval_df.melt(
        id_vars=["epoch"],
        value_vars=["eval_json_valid", "eval_schema_valid", "eval_canonical_exact_match"],
        var_name="metric",
        value_name="value",
    )
    plt.figure(figsize=(11, 7))
    sns.lineplot(data=melted, x="epoch", y="value", hue="metric", marker="o")
    plt.ylim(0, 1)
    plt.title("Output validity and whole-document exactness")
    plt.tight_layout()
    plt.savefig(save("04_validity_progression"), dpi=170)
    plt.close()

    fig, axis = plt.subplots(figsize=(11, 7))
    axis.plot(eval_df["epoch"], eval_df["eval_loss"], marker="o", color="#DC2626")
    axis.set_ylabel("Teacher-forced eval loss", color="#DC2626")
    twin = axis.twinx()
    twin.plot(eval_df["epoch"], eval_df["eval_field_value_f1"], marker="o", color="#2563EB")
    twin.set_ylabel("Generated field F1", color="#2563EB")
    axis.set_title("Objective divergence: eval loss rises while generated F1 improves")
    fig.tight_layout()
    fig.savefig(save("05_eval_loss_vs_generated_f1"), dpi=170)
    plt.close(fig)

    plt.figure(figsize=(11, 7))
    sns.histplot(document_df, x="field_f1", bins=20, kde=True)
    plt.axvline(document_df["field_f1"].median(), color="#DC2626", linestyle="--", label="median")
    plt.legend()
    plt.title("Validation document F1 distribution")
    plt.tight_layout()
    plt.savefig(save("06_document_f1_distribution"), dpi=170)
    plt.close()

    plt.figure(figsize=(9, 8))
    sns.scatterplot(
        data=document_df,
        x="field_recall",
        y="field_precision",
        hue="schema_valid",
        size="target_leaves",
        sizes=(25, 180),
    )
    plt.xlim(0, 1.02)
    plt.ylim(0, 1.02)
    plt.title("Document precision/recall and schema validity")
    plt.tight_layout()
    plt.savefig(save("07_document_precision_recall"), dpi=170)
    plt.close()

    _plot_bar(
        section_df.sort_values("f1"),
        x="section",
        y="f1",
        title="F1 by top-level section",
        path=save("08_section_f1"),
        horizontal=True,
    )
    supported = field_df[field_df["validation_values"] >= 5].nsmallest(20, "f1")
    _plot_bar(
        supported,
        x="field_path",
        y="f1",
        title="Lowest-F1 fields with at least five validation values",
        path=save("09_lowest_field_f1"),
        horizontal=True,
    )
    field_errors = field_df.assign(
        error_count=field_df["false_positive_values"] + field_df["false_negative_values"]
    ).nlargest(20, "error_count")
    _plot_bar(
        field_errors.sort_values("error_count"),
        x="field_path",
        y="error_count",
        title="Largest exact field/value error contributors",
        path=save("10_field_error_contribution"),
        horizontal=True,
    )

    penalty = field_df[field_df["index_penalty"] > 0].nlargest(20, "index_penalty")
    if not penalty.empty:
        melted = penalty.melt(
            id_vars="field_path",
            value_vars=["f1", "index_insensitive_f1"],
            var_name="metric",
            value_name="value",
        )
        _plot_bar(
            melted,
            x="field_path",
            y="value",
            hue="metric",
            title="Strict versus index-insensitive repeated-field F1",
            path=save("11_index_sensitivity"),
            horizontal=True,
        )

    error_counts = (
        error_df.groupby("error_type", as_index=False)
        .size()
        .rename(columns={"size": "count"})
        .sort_values("count")
    )
    _plot_bar(
        error_counts,
        x="error_type",
        y="count",
        title="One-to-one leaf error taxonomy",
        path=save("12_error_taxonomy"),
        horizontal=True,
    )
    _plot_bar(
        relation_df.sort_values("f1"),
        x="fact_type",
        y="f1",
        title="Identity-aware cargo relation F1",
        path=save("13_relation_fact_f1"),
        horizontal=True,
    )
    clean_categories = category_df[category_df["category_type"] != "confusion"]
    supported_categories = clean_categories[
        clean_categories["validation_references"] > 0
    ].sort_values("f1")
    _plot_bar(
        supported_categories,
        x="token",
        y="f1",
        hue="category_type",
        title="Readable category-token F1",
        path=save("14_category_token_f1"),
        horizontal=True,
    )

    plt.figure(figsize=(11, 8))
    sns.scatterplot(
        data=field_df,
        x="train_documents",
        y="f1",
        hue="section",
        size="validation_values",
        sizes=(20, 180),
    )
    plt.xscale("log")
    plt.title("Training support versus validation F1")
    plt.tight_layout()
    plt.savefig(save("15_train_support_vs_field_f1"), dpi=170)
    plt.close()

    for index, (group_type, title) in enumerate(
        (
            ("page_count", "F1 by page count"),
            ("target_complexity", "F1 by target complexity"),
            ("template_coverage", "F1 by template coverage"),
            ("cargo_groups", "F1 by cargo-group count"),
            ("containers", "F1 by container count"),
            ("allocations", "F1 with and without allocations"),
        ),
        start=16,
    ):
        subset = group_df[group_df["group_type"] == group_type]
        _plot_bar(subset, x="group", y="f1", title=title, path=save(f"{index:02d}_{group_type}_f1"))

    schema_counts = (
        document_df.groupby("schema_failure_class", as_index=False)
        .size()
        .rename(columns={"size": "documents"})
    )
    _plot_bar(
        schema_counts,
        x="schema_failure_class",
        y="documents",
        title="Schema failure classes",
        path=save("22_schema_failure_classes"),
    )

    plt.figure(figsize=(11, 7))
    sns.histplot(
        document_df,
        x="generated_reference_character_ratio",
        bins=25,
        hue="schema_valid",
        multiple="stack",
    )
    plt.axvline(1.0, color="black", linestyle="--")
    plt.title("Generated/reference output-length ratio")
    plt.tight_layout()
    plt.savefig(save("23_output_length_ratio"), dpi=170)
    plt.close()

    point = float(bootstrap[0]["point"])
    lower = float(bootstrap[0]["lower_95"])
    upper = float(bootstrap[0]["upper_95"])
    plt.figure(figsize=(9, 4))
    plt.errorbar([point], [0], xerr=[[point - lower], [upper - point]], fmt="o", capsize=8)
    plt.axvline(0.9, color="#DC2626", linestyle="--", label="target 0.90")
    plt.yticks([])
    plt.xlim(max(0, lower - 0.05), 0.93)
    plt.title("Document-bootstrap interval for exact field/value F1")
    plt.legend()
    plt.tight_layout()
    plt.savefig(save("24_bootstrap_field_f1"), dpi=170)
    plt.close()

    if prior_eval_path is not None:
        prior = pd.read_csv(prior_eval_path)
        current = eval_df[eval_df["epoch"] > 0][["epoch", "eval_field_value_f1"]].copy()
        current["run"] = "combined1157 task-facing"
        prior = prior[prior["epoch"] > 0][["epoch", "eval_field_value_f1"]].copy()
        prior["run"] = "prior relation-explicit 487"
        comparison = pd.concat([current, prior], ignore_index=True)
        plt.figure(figsize=(11, 7))
        sns.lineplot(data=comparison, x="epoch", y="eval_field_value_f1", hue="run", marker="o")
        plt.ylim(0.6, 0.86)
        plt.title("Directional progression comparison (different validation splits)")
        plt.tight_layout()
        plt.savefig(save("25_prior_run_progression"), dpi=170)
        plt.close()

    _plot_bar(
        complexity_df,
        x="feature",
        y="fraction",
        hue="split",
        title="Training versus validation structural complexity",
        path=save("26_split_complexity"),
        horizontal=True,
    )

    plt.figure(figsize=(11, 7))
    sns.lineplot(data=training_df, x="step", y="learning_rate")
    plt.title("Learning-rate schedule")
    plt.tight_layout()
    plt.savefig(save("27_learning_rate"), dpi=170)
    plt.close()

    plt.figure(figsize=(11, 7))
    sns.lineplot(data=training_df, x="step", y="grad_norm")
    plt.yscale("log")
    plt.title("Pre-clipping gradient norm")
    plt.tight_layout()
    plt.savefig(save("28_gradient_norm_log"), dpi=170)
    plt.close()

    fig, axis = plt.subplots(figsize=(11, 7))
    axis.plot(
        eval_df["epoch"],
        eval_df["eval_generated_tokens_mean"],
        marker="o",
        color="#7C3AED",
    )
    axis.set_ylabel("Mean generated tokens", color="#7C3AED")
    twin = axis.twinx()
    twin.plot(
        eval_df["epoch"],
        eval_df["eval_generation_eos_reached_fraction"],
        marker="o",
        color="#059669",
    )
    twin.set_ylabel("EOS reached fraction", color="#059669")
    twin.set_ylim(0, 1.05)
    axis.set_title("Generation completion across checkpoints")
    fig.tight_layout()
    fig.savefig(save("29_generation_length_and_eos"), dpi=170)
    plt.close(fig)

    plt.figure(figsize=(11, 7))
    sns.lineplot(data=eval_df, x="epoch", y="eval_runtime", marker="o")
    plt.title("Generated evaluation runtime")
    plt.tight_layout()
    plt.savefig(save("30_eval_runtime"), dpi=170)
    plt.close()

    for index, (x, title) in enumerate(
        (
            ("input_tokens", "Document F1 versus input length"),
            ("target_leaves", "Document F1 versus target complexity"),
        ),
        start=31,
    ):
        plt.figure(figsize=(11, 8))
        sns.scatterplot(
            data=document_df,
            x=x,
            y="field_f1",
            hue="template_seen_in_train",
            size="page_count",
            sizes=(25, 180),
        )
        plt.title(title)
        plt.tight_layout()
        plt.savefig(save(f"{index:02d}_field_f1_vs_{x}"), dpi=170)
        plt.close()

    source_groups = group_df[group_df["group_type"] == "source_corpus"].sort_values("f1")
    _plot_bar(
        source_groups,
        x="group",
        y="f1",
        title="F1 by source corpus",
        path=save("33_source_corpus_f1"),
        horizontal=True,
    )

    plt.figure(figsize=(10, 9))
    sns.scatterplot(
        data=field_df[field_df["validation_values"] > 0],
        x="recall",
        y="precision",
        hue="section",
        size="validation_values",
        sizes=(25, 200),
    )
    plt.plot([0, 1], [0, 1], linestyle="--", color="black", alpha=0.4)
    plt.xlim(0, 1.02)
    plt.ylim(0, 1.02)
    plt.title("Field precision versus recall")
    plt.tight_layout()
    plt.savefig(save("34_field_precision_recall"), dpi=170)
    plt.close()

    plt.figure(figsize=(11, 8))
    sns.scatterplot(
        data=field_df[field_df["validation_values"] > 0],
        x="validation_novel_value_fraction",
        y="f1",
        hue="section",
        size="validation_values",
        sizes=(25, 200),
    )
    plt.title("Novel-value exposure versus field F1")
    plt.tight_layout()
    plt.savefig(save("35_novel_value_fraction_vs_f1"), dpi=170)
    plt.close()

    confusion = category_df[category_df["category_type"] == "confusion"]
    if not confusion.empty:
        confusion = (
            confusion.groupby("token", as_index=False)
            .size()
            .rename(columns={"size": "count"})
            .nlargest(20, "count")
            .sort_values("count")
        )
        _plot_bar(
            confusion,
            x="token",
            y="count",
            title="Most frequent readable-category confusions",
            path=save("36_category_confusions"),
            horizontal=True,
        )
    return paths


def _markdown_table(rows: Sequence[Mapping[str, Any]], columns: Sequence[str]) -> str:
    if not rows:
        return "_No rows._"
    lines = ["| " + " | ".join(columns) + " |", "|" + "|".join("---" for _ in columns) + "|"]
    for row in rows:
        values = []
        for column in columns:
            value = row[column]
            if isinstance(value, float):
                values.append(f"{value:.4f}")
            else:
                values.append(str(value).replace("|", "\\|"))
        lines.append("| " + " | ".join(values) + " |")
    return "\n".join(lines)


def _report(
    *,
    summary: Mapping[str, Any],
    evaluation: Sequence[Mapping[str, Any]],
    fields: Sequence[Mapping[str, Any]],
    sections: Sequence[Mapping[str, Any]],
    relations: Sequence[Mapping[str, Any]],
    categories: Sequence[Mapping[str, Any]],
    errors: Sequence[Mapping[str, Any]],
    groups: Sequence[Mapping[str, Any]],
    dataset_complexity: Sequence[Mapping[str, Any]],
    schema_issues: Sequence[Mapping[str, Any]],
    field_error_profiles: Sequence[Mapping[str, Any]],
    plot_paths: Sequence[str],
) -> str:
    prior_comparison = summary.get("prior_run_comparison")
    if isinstance(prior_comparison, Mapping):
        prior_sentence = (
            f"The recovered score is {float(prior_comparison['absolute_f1_delta']):+.4f} "
            f"versus the prior run ({float(prior_comparison['prior_field_value_f1']):.4f}), "
            "but the validation identities differ, so this is directional rather than paired evidence."
        )
    else:
        prior_sentence = "No prior-run comparison was requested."
    field_errors = sorted(
        fields,
        key=lambda row: int(row["false_positive_values"]) + int(row["false_negative_values"]),
        reverse=True,
    )[:15]
    low_fields = sorted(
        (row for row in fields if int(row["validation_values"]) >= 5),
        key=lambda row: float(row["f1"]),
    )[:15]
    high_fields = sorted(
        (
            row
            for row in fields
            if int(row["validation_values"]) >= 5
            and not str(row["field_path"]).endswith((".groupId", ".packageId"))
        ),
        key=lambda row: (float(row["f1"]), int(row["validation_values"])),
        reverse=True,
    )[:15]
    error_counts = Counter(str(row["error_type"]) for row in errors)
    grounded_wrong = sum(
        row["error_type"] not in {"omission", "right_value_wrong_index"}
        and row["predicted_ocr_grounded"] is True
        for row in errors
    )
    cohort_rows = sorted(
        (row for row in groups if int(row["documents"]) >= 5),
        key=lambda row: (str(row["group_type"]), float(row["f1"])),
    )
    schema_issue_counts = Counter(str(row["message"]) for row in schema_issues)
    schema_issue_table = [
        {"issue": issue, "occurrences": count}
        for issue, count in schema_issue_counts.most_common(15)
    ]
    profile_by_field = {str(row["field_path"]): row for row in field_error_profiles}
    profiled_fields = [
        profile_by_field[field]
        for field in (
            "$.documentPatch.cargoPackages[].quantity",
            "$.documentPatch.cargoPackages[].typeCategory",
            "$.documentPatch.cargoGroups[].description",
            "$.documentPatch.cargoGroups[].grossWeight.value",
            "$.documentPatch.cargoGroups[].marksAndNumbers[]",
            "$.documentPatch.parties.shipper.address",
        )
    ]
    eval_table = [
        {
            "epoch": row["epoch"],
            "loss": row["eval_loss"],
            "precision": row["eval_field_value_precision"],
            "recall": row["eval_field_value_recall"],
            "f1": row["eval_field_value_f1"],
            "cargo_f1": row["eval_cargo_relation_f1"],
            "category_f1": row["eval_category_value_f1"],
            "json": row["eval_json_valid"],
            "schema": row["eval_schema_valid"],
        }
        for row in evaluation
    ]
    report = f"""# T5Gemma2 270M task-facing checkpoint audit

Generated at `{summary["created_at"]}` from the retained step-900 adapter of `{summary["source_run_id"]}`.

## Executive conclusion

The best retained checkpoint reaches **{summary["metrics"]["field_value_f1"]:.4f} exact field/value F1** on 100 documents (95% document-bootstrap interval **{summary["bootstrap"]["lower_95"]:.4f}-{summary["bootstrap"]["upper_95"]:.4f}**). {prior_sentence} It is not yet a reliable 0.90 system: schema validity is **{summary["metrics"]["schema_valid"]:.1%}**, whole-document exact match is **{summary["metrics"]["canonical_exact_match"]:.1%}**, and cargo-relation F1 is **{summary["metrics"]["cargo_relation_f1"]:.4f}**.

The optimization loop reached all 1,125 steps, but the run did **not** complete. It disappeared during the final generated evaluation: no step-1125 checkpoint, final adapter, predictions, result files, or manifest exists, while local status and MLflow remain `RUNNING`. This audit therefore evaluates checkpoint 900 without changing weights. It must not be described as the epoch-25 model.

The curve is diminishing rather than flat: field F1 rises 0.6488 → 0.7783 → 0.8103 → 0.8294 at epochs 5/10/15/20. However, teacher-forced eval loss bottoms at epoch 10 and rises to 0.1227 at epoch 20 while training loss approaches zero. The model is memorizing the training serialization while generated exactness improves more slowly. Ordinary extra epochs or small Adam-beta changes are therefore low-value next moves.

The exact leaf metric is positional. Repeated values at a different list index recover **{summary["metrics"]["index_insensitive_f1"] - summary["metrics"]["field_value_f1"]:.4f} F1** under an index-insensitive multiset diagnostic. Strict F1 remains the serving-contract metric, but this delta separates ordering errors from genuine extraction errors.

The adapter is not narrowly targeted: its rank-32 rsLoRA contains **{summary["adapter"]["parameter_count"]:,} parameters** across **{summary["adapter"]["adapted_module_count"]} modules**, split evenly across the text encoder and decoder. It covers every configured q/k/v/o attention projection and gate/up/down MLP projection. Capacity experiments remain useful, but a missing major projection family is not the diagnosed bottleneck.

At document level, median F1 is **{summary["document_distribution"]["median_f1"]:.4f}**; {summary["document_distribution"]["documents_at_least_090"]}/100 documents reach at least 0.90, while {summary["document_distribution"]["documents_below_070"]}/100 fall below 0.70. The two zero-F1 outliers are the two malformed-JSON generations, so constrained syntax would remove catastrophic parse loss even though it would not make their semantic content correct.

## Checkpoint progression

{_markdown_table(eval_table, ["epoch", "loss", "precision", "recall", "f1", "cargo_f1", "category_f1", "json", "schema"])}

Checkpoint 900 is the best available checkpoint. Checkpoint 675 has the strongest category F1 ({evaluation[3]["eval_category_value_f1"]:.4f}), but checkpoint 900 has higher global F1, recall, cargo F1, JSON validity, and schema validity.

## What prevents 0.90

- Current pooled counts are {summary["counts"]["true_positive"]:,} correct, {summary["counts"]["predicted"]:,} predicted, and {summary["counts"]["reference"]:,} reference leaves. Holding prediction/reference totals fixed, reaching 0.90 requires **{summary["gap_to_090"]["additional_true_positives"]:,}** additional exact matches.
- Cargo, package, allocation, and container leaves are {summary["dataset"]["validation_cargo_leaf_fraction"]:.1%} of validation supervision; parties add {summary["dataset"]["validation_party_leaf_fraction"]:.1%}. Cargo structure is therefore not a small edge case.
- Validation is harder than training: 5+ page, multi-container, multi-goods, multi-package, and high-leaf documents are all more prevalent. {summary["dataset"]["unseen_template_documents"]} of {summary["dataset"]["template_covered_validation_documents"]} EDA-covered validation documents have no training template match.
- Three validation documents have cargo/package/container/allocation count shapes absent from training; {summary["dataset"]["unseen_extended_cargo_pattern_documents"]} have an unseen extended pattern after allocation-row counts and coverage modes are included.
- {summary["dataset"]["novel_validation_leaves"]:,}/{summary["counts"]["reference"]:,} validation values ({summary["dataset"]["novel_validation_leaf_fraction"]:.1%}) are novel under their field path. Bill numbers, container numbers, seals, descriptions, and allocation anchors must be copied/generalized, not memorized.
- Only {summary["dataset"]["rare_support_validation_leaves"]:,} validation leaves ({summary["dataset"]["rare_support_validation_leaf_fraction"]:.1%}) belong to fields seen in fewer than 100 training documents. Rare field paths alone cannot explain the gap.
- Error taxonomy: {dict(error_counts)}. Wrong non-omitted predictions matching a normalized OCR substring: {grounded_wrong}. This is a diagnostic grounding heuristic, not a hallucination classifier; the dominant distinction is omitted/mis-grouped data versus unsupported values.

Structural prevalence by split:

{_markdown_table(dataset_complexity, ["feature", "split", "matching_documents", "documents", "fraction"])}

## Weakest fields

Lowest F1 among fields with at least five validation values:

{_markdown_table(low_fields, ["field_path", "train_documents", "validation_values", "precision", "recall", "f1", "index_insensitive_f1"])}

Largest strict error contributors:

{_markdown_table(field_errors, ["field_path", "false_positive_values", "false_negative_values", "f1", "train_documents", "validation_novel_value_fraction"])}

## Strongest fields

Highest F1 among non-synthetic-link fields with at least five validation values:

{_markdown_table(high_fields, ["field_path", "train_documents", "validation_values", "precision", "recall", "f1", "whole_field_exact_rate"])}

## Why schema validity is only 76%

There are {summary["schema_failures"]["documents"]} schema-invalid documents: {summary["schema_failures"]["invalid_json_documents"]} are malformed JSON and {summary["schema_failures"]["schema_validation_documents"]} parse as objects but violate the semantic target contract. The most frequent violations are:

{_markdown_table(schema_issue_table, ["issue", "occurrences"])}

JSON grammar constraints can remove the two syntax failures and many empty-list/enum errors. They cannot alone enforce allocation totals, foreign package references, or ISO container checks; those require a simpler single-source target plus deterministic projection/validation.

## How far wrong are the main weak fields?

{_markdown_table(profiled_fields, ["field_path", "validation_values", "f1", "index_insensitive_f1", "omissions", "additions", "substitutions", "right_value_wrong_index", "median_value_similarity"])}

- Package quantity errors mix 12 omissions, 17 additions, 11 value substitutions, and five correct quantities assigned to the wrong list index; this is not one numeric-format bug.
- Goods-description substitutions have median similarity around 0.65 and commonly add neighboring marks/weights or collapse several source rows, indicating boundary/grouping errors.
- Gross-weight substitutions are often wrong row totals or scale errors (for example, decimal-place shifts), while marks recover strongly when list position is ignored.
- Shipper-address exact F1 is low even though substitution similarity is usually high; punctuation, partial lines, and neighboring locality text make exact-string assembly brittle.

## Section performance

{_markdown_table(sorted(sections, key=lambda row: float(row["f1"])), ["section", "reference_values", "precision", "recall", "f1", "whole_section_exact_rate"])}

## Validation cohorts

{_markdown_table(cohort_rows, ["group_type", "group", "documents", "precision", "recall", "f1", "schema_valid"])}

## Cargo relations

{_markdown_table(relations, ["fact_type", "reference_facts", "precision", "recall", "f1", "exact_document_rate"])}

The separated `cargoGroups` / `cargoPackages` / `cargoAllocationGroups` target forces distant `gN` and `pN` references. The relation metric is position-independent and still plateaus near 0.71, so cargo hierarchy—not just array order—is the primary structural bottleneck.

## Readable categories

{_markdown_table([row for row in categories if row["category_type"] != "confusion" and int(row["validation_references"]) > 0], ["category_type", "token", "train_facts", "validation_references", "precision", "recall", "f1"])}

Readable tokens remain the right direction, but epoch-20 category F1 regressed from the epoch-15 peak. Rare category/structure combinations should be oversampled through real or validated synthetic examples rather than replaced by opaque MPCI codes.

## Evidence-backed improvement program

1. **Fix the target topology first.** Test one compact, source-ordered cargo tree with packages and allocations local to each cargo group, followed by deterministic projection into the MPCI arrays. Text2Event argues for reversible, compact, source-ordered labeled trees and reports gains from substructure-to-full-structure curricula: https://aclanthology.org/2021.acl-long.217/.
2. **Add a curriculum without adding an inference pass.** Derive scalar/party, single cargo-group, single allocation, and then full-document training examples from each audited label; finish on full documents. This increases supervision for the exact structures that plateau while serving remains one call.
3. **Use a compact schema instructor.** A/B test the current full developer JSON Schema against concise field semantics and allowed relations. UIE's structural schema instructor is the relevant precedent: https://aclanthology.org/2022.acl-long.395/.
4. **Add constrained greedy decoding for reliability.** Constrain JSON structure, legal keys, enums, and genuinely verbatim OCR spans. This should primarily improve validity/grounding; Text2Event's results do not support expecting constraints alone to close a seven-point semantic gap.
5. **Run capacity ablations only after topology/curriculum.** Current rank-32 rsLoRA already targets all q/k/v/o and gate/up/down projections across encoder text and decoder. Compare the same LoRA against PiSSA initialization and one full text-model fine-tune while keeping the 417M vision encoder frozen; do not jump to the 1B model. PEFT reference: https://huggingface.co/docs/peft/main/package_reference/lora; PiSSA: https://arxiv.org/abs/2404.02948.
6. **Scale diversity, not duplicates.** Build learning curves and add carrier/template-disjoint evaluation. VRDU shows repeated/hierarchical fields and unseen templates require different matching and remain substantially harder: https://arxiv.org/abs/2211.15421.
7. **Test the table input as a controlled ablation.** Compare identical labels/configuration with raw OCR versus raw OCR plus a delimited GLM-OCR table. Layout-aware document IE literature treats missing layout/group structure as a core obstacle: https://research.google/pubs/lmdx-language-model-based-document-information-extraction-and-localization/.
8. **Evaluate every 1-2 epochs in focused runs.** Five-epoch gaps are too coarse around the epoch-15/20 category/global tradeoff. Select a Pareto checkpoint across global F1, cargo F1, category F1, schema validity, and EOS rather than one scalar alone.

## Artifact map

- `tables/document_metrics.csv`: one row per validation document.
- `tables/field_metrics.csv`: strict/index-insensitive performance and train support.
- `tables/leaf_errors.jsonl`: one-to-one omission, addition, substitution, and list-alignment diagnoses.
- `tables/relation_metrics.csv` / `category_token_metrics.csv`: structural and categorical performance.
- `tables/group_metrics.csv`: source, template, length, and cargo-complexity slices.
- `tables/schema_failures.jsonl`: precise invalid-output diagnoses.
- `examples/lowest_performing.md`: the lowest-F1 documents and their largest discrepancies.
- `summary.json` and `analysis_manifest.json`: machine-readable conclusions and hashes.

## Plot gallery

"""
    for path in plot_paths:
        title = Path(path).stem.replace("_", " ").title()
        report += f"### {title}\n\n![{title}]({path})\n\n"
    return report


def _failure_gallery(
    path: Path, diagnostics: Sequence[Diagnostic], errors: Sequence[Mapping[str, Any]]
) -> None:
    grouped: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for row in errors:
        grouped[str(row["document_id"])].append(row)
    lowest = sorted(diagnostics, key=lambda row: row.f1)[:15]
    lines = ["# Lowest-performing validation documents", ""]
    for item in lowest:
        record = item.record
        lines.extend(
            [
                f"## {record.document_id}",
                "",
                f"- Source: `{record.source_path}`",
                f"- F1: `{item.f1:.4f}`; JSON/schema valid: `{item.assessment.json_valid}` / `{item.assessment.schema_valid}`",
                f"- Pages/containers/cargo groups/packages/allocations: `{record.page_count}` / `{record.container_count}` / `{record.cargo_group_count}` / `{record.package_count}` / `{record.allocation_group_count}`",
                f"- Template coverage: `{record.template_seen_in_train}`",
                "",
                "| Error | Field | Reference | Prediction | Similarity |",
                "|---|---|---|---|---:|",
            ]
        )
        ranked = sorted(
            grouped[record.document_id],
            key=lambda row: (row["error_type"] == "addition", row["field_path"]),
        )[:20]
        for row in ranked:
            lines.append(
                f"| {row['error_type']} | `{row['field_path']}` | `{str(row['reference_value'])[:180]}` | `{str(row['predicted_value'])[:180]}` | {row['value_similarity'] if row['value_similarity'] is not None else ''} |"
            )
        lines.append("")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


def publish_analysis(
    *,
    project_root: Path,
    run_dir: Path,
    evaluation_dir: Path,
    output_dir: Path,
    eda_dir: Path | None,
    prior_analysis_dir: Path | None,
) -> dict[str, Any]:
    project_root = project_root.resolve(strict=True)
    run_dir = run_dir.resolve(strict=True)
    evaluation_dir = evaluation_dir.resolve(strict=True)
    output_dir = output_dir.resolve()
    if output_dir.exists():
        raise ValueError(f"analysis output already exists: {output_dir}")
    if run_dir in output_dir.parents:
        raise ValueError("analysis output must be outside the immutable run directory")
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix=f".{output_dir.name}.", dir=output_dir.parent))
    try:
        raw_config = cast(
            dict[str, Any], yaml.safe_load((run_dir / "config.yaml").read_text(encoding="utf-8"))
        )
        config = TrainingConfig.model_validate(raw_config, strict=True)
        task = load_training_task(project_root, config)
        records = _load_records(
            project_root=project_root,
            run_dir=run_dir,
            config=config,
            task=task,
            eda_dir=eda_dir.resolve(strict=True) if eda_dir else None,
        )
        diagnostics, schema_rows = _load_diagnostics(
            evaluation_dir=evaluation_dir, records=records, task=task
        )
        evaluation_manifest = cast(dict[str, Any], _json_load(evaluation_dir / "manifest.json"))
        if (
            evaluation_manifest["checkpoint_step"] != 900
            or not evaluation_manifest["checkpoint_is_recorded_best"]
        ):
            raise ValueError("analysis prediction source is not retained best checkpoint 900")
        adapter_inventory = _adapter_inventory(
            run_dir
            / "checkpoints"
            / "checkpoint-900"
            / config.peft.adapter_name
            / "adapter_model.safetensors"
        )
        if adapter_inventory["sha256"] != evaluation_manifest["adapter"]["adapter_weights_sha256"]:
            raise ValueError("analyzed adapter weights differ from evaluated checkpoint")
        aggregate = _aggregate_diagnostics(diagnostics)
        recorded_metrics = evaluation_manifest["metrics"]
        for metric in ("field_value_precision", "field_value_recall", "field_value_f1"):
            if not math.isclose(
                float(aggregate[metric.removeprefix("field_value_")]),
                float(recorded_metrics[f"eval_{metric}"]),
                rel_tol=1e-12,
                abs_tol=1e-12,
            ):
                raise ValueError(f"recomputed prediction metric differs: {metric}")

        document_rows = _document_rows(diagnostics)
        error_rows = [row for item in diagnostics for row in _classify_errors(item)]
        field_rows = _field_rows(diagnostics, records)
        field_error_profile_rows = _field_error_profile_rows(error_rows, field_rows)
        section_rows = _section_rows(diagnostics)
        relation_rows = _fact_rows(diagnostics, category=False)
        category_fact_rows = _fact_rows(diagnostics, category=True)
        category_rows = _category_token_rows(diagnostics, records)
        group_rows = _group_rows(diagnostics)
        dataset_complexity_rows = _dataset_complexity_rows(records)
        cargo_signature_rows = _cargo_signature_rows(records)
        extended_cargo_signature_rows = _extended_cargo_signature_rows(records)
        training_rows, evaluation_rows = _history_rows(run_dir)
        bootstrap_rows = _bootstrap(diagnostics)
        schema_issue_rows = _schema_issue_rows(schema_rows)

        index_tp = sum(row.index_true_positive for row in diagnostics)
        index_precision = index_tp / int(aggregate["predicted"])
        index_recall = index_tp / int(aggregate["reference"])
        index_f1 = _f1(index_precision, index_recall)
        target_true_positive = math.ceil(
            0.9 * (int(aggregate["predicted"]) + int(aggregate["reference"])) / 2
        )
        validation_field_rows = [row for row in field_rows if int(row["validation_values"]) > 0]
        rare_leaves = sum(
            int(row["validation_values"])
            for row in validation_field_rows
            if int(row["train_documents"]) < 100
        )
        novel_leaves = sum(int(row["validation_novel_values"]) for row in validation_field_rows)
        validation_records = [row for row in records.values() if row.split == "validation"]
        validation_leaves = sum(row.target_leaves for row in validation_records)
        cargo_leaves = sum(
            len(
                {
                    pair
                    for pair in _flatten(cast(dict[str, Any], record.target["documentPatch"]))
                    if _section(pair[0])
                    in {"cargoGroups", "cargoPackages", "cargoAllocationGroups", "containers"}
                }
            )
            for record in validation_records
        )
        party_leaves = sum(
            len(
                {
                    pair
                    for pair in _flatten(cast(dict[str, Any], record.target["documentPatch"]))
                    if _section(pair[0]) == "parties"
                }
            )
            for record in validation_records
        )
        template_covered = [
            row for row in validation_records if row.template_seen_in_train != "unknown"
        ]
        unseen_cargo_signatures = sum(
            int(row["validation_documents"])
            for row in cargo_signature_rows
            if not bool(row["seen_in_train"])
        )
        unseen_extended_cargo_patterns = sum(
            int(row["validation_documents"])
            for row in extended_cargo_signature_rows
            if not bool(row["seen_in_train"])
        )
        metrics = {
            **{key: value for key, value in aggregate.items() if key not in {"documents"}},
            "field_value_accuracy": sum(row.true_positive for row in diagnostics)
            / sum(row.compared_paths for row in diagnostics),
            "field_value_precision": aggregate["precision"],
            "field_value_recall": aggregate["recall"],
            "field_value_f1": aggregate["f1"],
            "index_insensitive_f1": index_f1,
            "cargo_relation_f1": recorded_metrics["eval_cargo_relation_f1"],
            "category_value_f1": recorded_metrics["eval_category_value_f1"],
        }
        document_f1_values = sorted(row.f1 for row in diagnostics)
        prior_comparison: dict[str, Any] | None = None
        if prior_analysis_dir is not None:
            prior_summary = cast(
                dict[str, Any],
                _json_load(prior_analysis_dir.resolve(strict=True) / "summary.json"),
            )
            prior_f1 = float(prior_summary["final_saved_prediction_metrics"]["eval_field_value_f1"])
            prior_comparison = {
                "prior_analysis_dir": str(prior_analysis_dir.resolve(strict=True)),
                "prior_field_value_f1": prior_f1,
                "current_field_value_f1": float(metrics["field_value_f1"]),
                "absolute_f1_delta": float(metrics["field_value_f1"]) - prior_f1,
                "paired_comparison": False,
                "reason": "validation identities and dataset versions differ",
            }
        summary = {
            "schema_version": 1,
            "status": "complete",
            "created_at": datetime.now(UTC).isoformat(),
            "source_run_id": config.run.run_id,
            "source_run_completion": {
                "optimizer_steps_completed": 1125,
                "planned_optimizer_steps": 1125,
                "finalization_completed": False,
                "local_status": _json_load(run_dir / "status.json")["status"],
                "missing_artifacts": [
                    "checkpoint-1125",
                    "final-adapter",
                    "predictions",
                    "train_results.json",
                    "eval_results.json",
                    "manifest.json",
                ],
            },
            "checkpoint_evaluation": evaluation_manifest,
            "adapter": adapter_inventory,
            "prior_run_comparison": prior_comparison,
            "metrics": metrics,
            "document_distribution": {
                "mean_f1": statistics.fmean(document_f1_values),
                "median_f1": statistics.median(document_f1_values),
                "p10_f1": document_f1_values[9],
                "p25_f1": document_f1_values[24],
                "p75_f1": document_f1_values[74],
                "p90_f1": document_f1_values[89],
                "minimum_f1": document_f1_values[0],
                "maximum_f1": document_f1_values[-1],
                "documents_below_070": sum(value < 0.70 for value in document_f1_values),
                "documents_at_least_090": sum(value >= 0.90 for value in document_f1_values),
            },
            "counts": {
                "true_positive": aggregate["true_positive"],
                "predicted": aggregate["predicted"],
                "reference": aggregate["reference"],
            },
            "gap_to_090": {
                "target_true_positives_at_fixed_totals": target_true_positive,
                "additional_true_positives": max(
                    0, target_true_positive - int(aggregate["true_positive"])
                ),
            },
            "bootstrap": bootstrap_rows[0],
            "dataset": {
                "train_documents": 1057,
                "validation_documents": 100,
                "validation_leaves": validation_leaves,
                "validation_cargo_leaf_fraction": cargo_leaves / validation_leaves,
                "validation_party_leaf_fraction": party_leaves / validation_leaves,
                "novel_validation_leaves": novel_leaves,
                "novel_validation_leaf_fraction": novel_leaves / validation_leaves,
                "rare_support_validation_leaves": rare_leaves,
                "rare_support_validation_leaf_fraction": rare_leaves / validation_leaves,
                "template_covered_validation_documents": len(template_covered),
                "unseen_template_documents": sum(
                    row.template_seen_in_train == "unseen" for row in template_covered
                ),
                "unseen_cargo_signature_documents": unseen_cargo_signatures,
                "unseen_extended_cargo_pattern_documents": unseen_extended_cargo_patterns,
            },
            "error_taxonomy": dict(Counter(str(row["error_type"]) for row in error_rows)),
            "schema_failures": {
                "documents": len(schema_rows),
                "invalid_json_documents": sum(
                    row["failure_class"] == "invalid_json" for row in schema_rows
                ),
                "schema_validation_documents": sum(
                    row["failure_class"] == "schema_validation_error" for row in schema_rows
                ),
                "issue_types": dict(Counter(str(row["issue_type"]) for row in schema_issue_rows)),
            },
            "plots": [],
        }

        tables = temporary / "tables"
        _write_csv(tables / "training_history.csv", training_rows)
        _write_csv(tables / "eval_history.csv", evaluation_rows)
        _write_csv(tables / "document_metrics.csv", document_rows)
        _write_csv(tables / "field_metrics.csv", field_rows)
        _write_csv(tables / "field_error_profiles.csv", field_error_profile_rows)
        _write_csv(tables / "section_metrics.csv", section_rows)
        _write_csv(tables / "relation_metrics.csv", relation_rows)
        _write_csv(tables / "category_fact_metrics.csv", category_fact_rows)
        _write_csv(tables / "category_token_metrics.csv", category_rows)
        _write_csv(tables / "group_metrics.csv", group_rows)
        _write_csv(tables / "dataset_split_complexity.csv", dataset_complexity_rows)
        _write_csv(tables / "cargo_signatures.csv", cargo_signature_rows)
        _write_csv(tables / "extended_cargo_signatures.csv", extended_cargo_signature_rows)
        _write_csv(tables / "bootstrap_intervals.csv", bootstrap_rows)
        _write_jsonl(tables / "leaf_errors.jsonl", error_rows)
        _write_jsonl(tables / "schema_failures.jsonl", schema_rows)
        _write_csv(tables / "schema_validation_issues.csv", schema_issue_rows)
        _failure_gallery(temporary / "examples" / "lowest_performing.md", diagnostics, error_rows)

        prior_eval_path = (
            prior_analysis_dir.resolve(strict=True) / "tables" / "eval_history.csv"
            if prior_analysis_dir
            else None
        )
        plot_paths = _render_plots(
            output_dir=temporary,
            training=training_rows,
            evaluation=evaluation_rows,
            documents=document_rows,
            fields=field_rows,
            sections=section_rows,
            relations=relation_rows,
            categories=category_rows,
            errors=error_rows,
            groups=group_rows,
            dataset_complexity=dataset_complexity_rows,
            bootstrap=bootstrap_rows,
            prior_eval_path=prior_eval_path,
        )
        summary["plots"] = plot_paths
        atomic_write_json(temporary / "summary.json", summary)
        (temporary / "REPORT.md").write_text(
            _report(
                summary=summary,
                evaluation=evaluation_rows,
                fields=field_rows,
                sections=section_rows,
                relations=relation_rows,
                categories=category_rows,
                errors=error_rows,
                groups=group_rows,
                dataset_complexity=dataset_complexity_rows,
                schema_issues=schema_issue_rows,
                field_error_profiles=field_error_profile_rows,
                plot_paths=plot_paths,
            ),
            encoding="utf-8",
        )
        artifacts = [path for path in sorted(temporary.rglob("*")) if path.is_file()]
        manifest = {
            "schema_version": 1,
            "status": "complete",
            "source_run_id": config.run.run_id,
            "checkpoint_evaluation_manifest_sha256": sha256_file(evaluation_dir / "manifest.json"),
            "artifacts": [
                {
                    "path": str(path.relative_to(temporary)),
                    "bytes": path.stat().st_size,
                    "sha256": sha256_file(path),
                }
                for path in artifacts
            ],
        }
        atomic_write_json(temporary / "analysis_manifest.json", manifest)
        temporary.rename(output_dir)
    except BaseException:
        shutil.rmtree(temporary, ignore_errors=True)
        raise
    return summary


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, required=True)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--evaluation-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--eda-dir", type=Path)
    parser.add_argument("--prior-analysis-dir", type=Path)
    return parser.parse_args()


def main() -> None:
    arguments = _parse_args()
    summary = publish_analysis(
        project_root=arguments.project_root,
        run_dir=arguments.run_dir,
        evaluation_dir=arguments.evaluation_dir,
        output_dir=arguments.output_dir,
        eda_dir=arguments.eda_dir,
        prior_analysis_dir=arguments.prior_analysis_dir,
    )
    print(
        json.dumps(
            {
                "status": summary["status"],
                "output_dir": str(arguments.output_dir.resolve()),
                "field_value_f1": summary["metrics"]["field_value_f1"],
            },
            separators=(",", ":"),
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
