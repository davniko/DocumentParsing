#!/usr/bin/env python3
# ruff: noqa: E501
"""Audit one completed OCR-conditioned KIE run for reliability bottlenecks.

This is an artifact-only diagnostic.  It never loads model weights.  It joins the
immutable saved predictions to the pinned train/validation records and reviewed label
evidence, then measures support, novelty, copy grounding, semantic distance, list
alignment, template similarity, and counterfactual error-budget ceilings.
"""

from __future__ import annotations

import argparse
import csv
import difflib
import hashlib
import json
import math
import re
import statistics
import sys
import unicodedata
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
    MetricCounts,
    _chart_canvas,
    _font,
    _grouped_bar_chart,
    _histogram,
    _horizontal_bar_chart,
    _load_dataset_records,
    _normalize_path,
    _scatter_chart,
    _section,
)

from document_ocr.training.config import TrainingConfig  # noqa: E402
from document_ocr.training.metrics import PredictionAssessment, assess_prediction  # noqa: E402
from document_ocr.training.tasks import (  # noqa: E402
    TrainingTask,
    canonical_json,
    load_training_task,
)

AUDIT_SCHEMA_VERSION = 1
TARGET_F1 = 0.90
TOKEN_PATTERN = re.compile(r"[^\W_]+", re.UNICODE)
DIGIT_TOKEN_PATTERN = re.compile(r"\S*\d\S*")
SPACE_PATTERN = re.compile(r"\s+")
NON_ALNUM_PATTERN = re.compile(r"[^\w]+", re.UNICODE)

PACKAGE_TYPE_FIELD = "$.documentPatch.goodsItems[].packages[].type"
PACKAGE_QUANTITY_FIELD = "$.documentPatch.goodsItems[].packages[].quantity"
DESCRIPTION_FIELD = "$.documentPatch.goodsItems[].description"
MARKS_FIELD = "$.documentPatch.goodsItems[].marksAndNumbers[]"
ADDITIONAL_INFORMATION_FIELD = "$.documentPatch.goodsItems[].additionalInformation[]"

FOCUS_FIELDS = (
    PACKAGE_TYPE_FIELD,
    PACKAGE_QUANTITY_FIELD,
    DESCRIPTION_FIELD,
    MARKS_FIELD,
    ADDITIONAL_INFORMATION_FIELD,
    "$.documentPatch.goodsItems[].grossWeight.value",
    "$.documentPatch.goodsItems[].containerAllocations[].containerNumber",
    "$.documentPatch.goodsItems[].containerAllocations[].packageQuantity",
)

RESEARCH_SOURCES = (
    (
        "T5Gemma 2 technical report",
        "https://arxiv.org/abs/2512.14856",
        "The selected family is explicitly an encoder-decoder family intended for post-training and long-context modeling; the paper also exposes 270M, 1B, and 4B scales, making model-capacity ablation legitimate.",
    ),
    (
        "Official T5Gemma 2 270M-270M model card",
        "https://huggingface.co/google/t5gemma-2-270m-270m",
        "The checkpoint is a pretrained (not instruction-tuned) UL2-adapted encoder-decoder. Task behavior therefore has to come from supervised examples and the task prompt.",
    ),
    (
        "Google Gemma fine-tuning guidance",
        "https://ai.google.dev/gemma/docs/tune",
        "Google recommends task-specific input/output pairs with enough variation, testing on unseen tasks, and treats LoRA as a supported PEFT path rather than an inherently incorrect tuning method.",
    ),
    (
        "Unified Structure Generation for Universal Information Extraction",
        "https://aclanthology.org/2022.acl-long.395/",
        "UIE uses an explicit structural schema instructor and text-to-structure pretraining; this supports testing a semantic schema instruction rather than supplying field types alone.",
    ),
    (
        "PARSE: LLM Driven Schema Optimization for Reliable Entity Extraction",
        "https://aclanthology.org/2025.emnlp-industry.184/",
        "PARSE identifies ambiguous or incomplete developer-oriented JSON schemas as a structured-extraction reliability problem; this directly matches the current type-only prompt contract.",
    ),
    (
        "SchemaRAG: Dynamic Large Schema Reduction",
        "https://aclanthology.org/2026.acl-industry.78/",
        "SchemaRAG reports that pruning large schemas with metadata/examples can improve micro-F1 while reducing latency and tokens; this supports a section-conditioned schema ablation for the 104-field target surface.",
    ),
    (
        "Grammar-Constrained Decoding for Structured NLP Tasks",
        "https://arxiv.org/abs/2305.13971",
        "Grammar constraints can guarantee output structure and improve structured NLP tasks, but they do not by themselves resolve which OCR-grounded value belongs to which semantic field.",
    ),
    (
        "PICARD",
        "https://arxiv.org/abs/2109.05093",
        "Incremental constrained decoding materially improved fine-tuned T5 on a formal structured task by rejecting inadmissible continuations.",
    ),
    (
        "GenIE",
        "https://arxiv.org/abs/2112.08340",
        "Generative IE benefits from constraints at both the output-structure and schema/value levels; valid structure and grounded semantic selection are separate controls.",
    ),
    (
        "STAR low-resource IE augmentation",
        "https://ojs.aaai.org/index.php/AAAI/article/view/29839",
        "Structure-first synthetic generation with explicit instructions, self-reflection, and refinement improved low-resource IE; synthetic data is useful only with aggressive grounding and quality gates.",
    ),
    (
        "LoRA",
        "https://arxiv.org/abs/2106.09685",
        "The original LoRA evidence includes quality comparable to full fine-tuning on multiple language tasks. A full-tune or higher-rank branch should be a controlled capacity diagnostic, not the default explanation for this run.",
    ),
)


@dataclass(frozen=True, slots=True)
class PredictionRow:
    document_id: str
    generated_text: str
    reference_text: str
    assessment: PredictionAssessment


@dataclass(frozen=True, slots=True)
class EvidenceRecord:
    target_path: str
    evidence_kind: str
    normalization_rule: str
    raw_values: tuple[str, ...]
    excerpts: tuple[str, ...]
    page_numbers: tuple[int, ...]


@dataclass(frozen=True, slots=True)
class LabelSidecar:
    path: str
    evidence: Mapping[str, EvidenceRecord]


def _json_load(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _jsonl_load(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                raise ValueError(f"blank JSONL row at {path}:{line_number}")
            row = json.loads(line)
            if not isinstance(row, dict):
                raise ValueError(f"JSONL row is not an object at {path}:{line_number}")
            rows.append(cast(dict[str, Any], row))
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


def _safe_divide(numerator: float, denominator: float) -> float:
    return numerator / denominator if denominator else 0.0


def _schema_annotation_count(value: Any) -> int:
    """Count JSON Schema descriptions/titles without confusing same-named properties."""

    if isinstance(value, list):
        return sum(_schema_annotation_count(item) for item in value)
    if not isinstance(value, dict):
        return 0
    count = 0
    for key, item in value.items():
        if key in {"description", "title"}:
            count += 1
            continue
        if key in {"$defs", "definitions", "properties", "patternProperties"} and isinstance(
            item, dict
        ):
            count += sum(_schema_annotation_count(child) for child in item.values())
        else:
            count += _schema_annotation_count(item)
    return count


def _f1(precision: float, recall: float) -> float:
    return 2 * precision * recall / (precision + recall) if precision + recall else 0.0


def _metric_counts(predicted: int, reference: int, true_positive: int) -> MetricCounts:
    return MetricCounts(
        true_positive=true_positive,
        predicted=predicted,
        reference=reference,
        compared_paths=predicted + reference - true_positive,
    )


def _flatten_ordered(value: Any, path: str = "$.documentPatch") -> list[tuple[str, str, Any]]:
    if isinstance(value, dict):
        rows: list[tuple[str, str, Any]] = []
        for key in sorted(value):
            rows.extend(_flatten_ordered(value[key], f"{path}.{key}"))
        return rows
    if isinstance(value, list):
        rows = []
        for index, child in enumerate(value):
            rows.extend(_flatten_ordered(child, f"{path}[{index}]"))
        return rows
    scalar_json = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return [(path, scalar_json, value)]


def _parse_patch(text: str) -> dict[str, Any] | None:
    try:
        value = json.loads(text)
    except json.JSONDecodeError:
        return None
    if not isinstance(value, dict):
        return None
    patch = value.get("documentPatch")
    return cast(dict[str, Any], patch) if isinstance(patch, dict) else None


def _plain_text(value: Any) -> str:
    if isinstance(value, str):
        return value
    if isinstance(value, bool):
        return "true" if value else "false"
    if value is None:
        return "null"
    return str(value)


def _space_normalize(value: str) -> str:
    return SPACE_PATTERN.sub(" ", unicodedata.normalize("NFKC", value).casefold()).strip()


def _alnum_normalize(value: str) -> str:
    return NON_ALNUM_PATTERN.sub("", _space_normalize(value))


def _tokens(value: str) -> Counter[str]:
    return Counter(TOKEN_PATTERN.findall(_space_normalize(value)))


def _token_f1(left: str, right: str) -> float:
    left_tokens = _tokens(left)
    right_tokens = _tokens(right)
    overlap = sum((left_tokens & right_tokens).values())
    precision = _safe_divide(overlap, sum(left_tokens.values()))
    recall = _safe_divide(overlap, sum(right_tokens.values()))
    return _f1(precision, recall)


def value_similarity(reference: Any, predicted: Any) -> dict[str, Any]:
    """Return diagnostic similarity without weakening the exact production metric."""

    reference_text = _plain_text(reference)
    predicted_text = _plain_text(predicted)
    ref_space = _space_normalize(reference_text)
    pred_space = _space_normalize(predicted_text)
    ref_alnum = _alnum_normalize(reference_text)
    pred_alnum = _alnum_normalize(predicted_text)
    sequence_ratio = difflib.SequenceMatcher(a=ref_space, b=pred_space, autojunk=False).ratio()
    result: dict[str, Any] = {
        "space_normalized_exact": ref_space == pred_space,
        "alnum_normalized_exact": bool(ref_alnum) and ref_alnum == pred_alnum,
        "sequence_similarity": sequence_ratio,
        "token_f1": _token_f1(reference_text, predicted_text),
        "reference_contains_prediction": bool(pred_space) and pred_space in ref_space,
        "prediction_contains_reference": bool(ref_space) and ref_space in pred_space,
        "character_length_delta": len(predicted_text) - len(reference_text),
    }
    if (
        isinstance(reference, (int, float))
        and not isinstance(reference, bool)
        and isinstance(predicted, (int, float))
        and not isinstance(predicted, bool)
    ):
        absolute_delta = float(predicted) - float(reference)
        result["numeric_signed_delta"] = absolute_delta
        result["numeric_absolute_delta"] = abs(absolute_delta)
        result["numeric_relative_absolute_error"] = _safe_divide(
            abs(absolute_delta), abs(float(reference))
        )
    else:
        result["numeric_signed_delta"] = None
        result["numeric_absolute_delta"] = None
        result["numeric_relative_absolute_error"] = None
    return result


def raw_grounding(value: Any, raw_text: str) -> str:
    """Classify whether a predicted scalar is visibly grounded in raw OCR text."""

    text = _plain_text(value)
    if not text:
        return "not_grounded"
    if text in raw_text:
        return "verbatim"
    normalized_value = _space_normalize(text)
    normalized_raw = _space_normalize(raw_text)
    if normalized_value and normalized_value in normalized_raw:
        return "case_or_whitespace"
    alnum_value = _alnum_normalize(text)
    alnum_raw = _alnum_normalize(raw_text)
    if len(alnum_value) >= 3 and alnum_value in alnum_raw:
        return "punctuation_normalized"
    value_tokens = TOKEN_PATTERN.findall(normalized_value)
    raw_tokens = TOKEN_PATTERN.findall(normalized_raw)
    if len(value_tokens) >= 3:
        positions: list[int] = []
        cursor = 0
        for token in value_tokens:
            try:
                position = raw_tokens.index(token, cursor)
            except ValueError:
                positions = []
                break
            positions.append(position)
            cursor = position + 1
        if positions and positions[-1] - positions[0] + 1 - len(positions) <= max(
            24, 2 * len(positions)
        ):
            return "ordered_token_subsequence"
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        canonical_number = format(float(value), ".15g")
        raw_numbers = {
            candidate.replace(",", "")
            for candidate in re.findall(r"(?<!\w)[+-]?(?:\d[\d,]*)(?:\.\d+)?(?!\w)", raw_text)
        }
        if canonical_number in raw_numbers or (
            float(value).is_integer() and str(int(value)) in raw_numbers
        ):
            return "numeric_normalized"
    return "not_grounded"


def _find_context(value: Any, raw_text: str, radius: int = 180) -> str:
    text = _plain_text(value)
    start = raw_text.casefold().find(text.casefold())
    if start < 0:
        candidates = [token for token in TOKEN_PATTERN.findall(text) if len(token) >= 4]
        for token in sorted(candidates, key=len, reverse=True):
            start = raw_text.casefold().find(token.casefold())
            if start >= 0:
                break
    if start < 0:
        return ""
    left = max(0, start - radius)
    right = min(len(raw_text), start + len(text) + radius)
    return SPACE_PATTERN.sub(" ", raw_text[left:right]).strip()


def _value_match_rank(reference: Any, predicted: Any) -> tuple[float, float, float]:
    similarity = value_similarity(reference, predicted)
    return (
        float(similarity["space_normalized_exact"]),
        float(similarity["token_f1"]),
        float(similarity["sequence_similarity"]),
    )


def classify_reference_match(
    path: str,
    scalar_json: str,
    value: Any,
    predicted_fields: Sequence[tuple[str, str, Any]],
) -> tuple[str, str, Any | None]:
    """Classify one reference occurrence against raw parsed prediction leaves."""

    exact_path = [
        (candidate_path, candidate_json, candidate)
        for candidate_path, candidate_json, candidate in predicted_fields
        if candidate_path == path
    ]
    if any(candidate_json == scalar_json for _, candidate_json, _ in exact_path):
        return "exact", path, value
    normalized_path = _normalize_path(path)
    same_field = [item for item in predicted_fields if _normalize_path(item[0]) == normalized_path]
    exact_elsewhere = [item for item in same_field if item[1] == scalar_json]
    if exact_elsewhere:
        candidate = min(exact_elsewhere, key=lambda item: item[0])
        return "exact_value_other_index", candidate[0], candidate[2]
    if exact_path:
        candidate = max(exact_path, key=lambda item: _value_match_rank(value, item[2]))
        return "same_path_wrong_value", candidate[0], candidate[2]
    if same_field:
        candidate = max(same_field, key=lambda item: _value_match_rank(value, item[2]))
        return "same_field_other_value", candidate[0], candidate[2]
    return "field_absent", "", None


def _target_evidence_relation(target: Any, evidence: EvidenceRecord | None) -> str:
    if evidence is None:
        return "sidecar_missing"
    target_text = _plain_text(target)
    if target_text in evidence.raw_values:
        return "verbatim_raw_value"
    target_space = _space_normalize(target_text)
    if any(target_space == _space_normalize(value) for value in evidence.raw_values):
        return "case_or_whitespace_join"
    target_alnum = _alnum_normalize(target_text)
    if target_alnum and any(
        target_alnum == _alnum_normalize(value) for value in evidence.raw_values
    ):
        return "punctuation_normalized"
    return "semantic_or_numeric_normalization"


def _evidence_position(record: DatasetRecord, evidence: EvidenceRecord | None) -> float | None:
    if evidence is None:
        return None
    positions = [record.raw_text.find(excerpt) for excerpt in evidence.excerpts]
    valid = [position for position in positions if position >= 0]
    if not valid:
        return None
    return min(valid) / max(1, len(record.raw_text) - 1)


def _load_predictions(
    path: Path,
    records: Mapping[str, DatasetRecord],
    task: TrainingTask,
) -> dict[str, PredictionRow]:
    rows: dict[str, PredictionRow] = {}
    validation_ids = {
        record.document_id for record in records.values() if record.split == "validation"
    }
    for row in _jsonl_load(path):
        document_id = str(row["document_id"])
        if document_id in rows:
            raise ValueError(f"duplicate prediction document ID: {document_id}")
        if document_id not in validation_ids:
            raise ValueError(
                f"prediction does not map to configured validation document: {document_id}"
            )
        generated_text = str(row["generated_text"])
        reference_text = str(row["reference_text"])
        expected_reference = canonical_json(task.canonicalize(records[document_id].target))
        if reference_text != expected_reference:
            raise ValueError(f"prediction/reference identity mismatch for {document_id}")
        assessment = assess_prediction(generated_text, reference_text, task)
        rows[document_id] = PredictionRow(
            document_id=document_id,
            generated_text=generated_text,
            reference_text=reference_text,
            assessment=assessment,
        )
    if set(rows) != validation_ids:
        missing = sorted(validation_ids - set(rows))
        raise ValueError(f"saved prediction set is incomplete: missing={missing}")
    return rows


def _load_label_sidecars(
    project_root: Path,
    records: Mapping[str, DatasetRecord],
    task: TrainingTask,
) -> tuple[dict[str, LabelSidecar], list[dict[str, Any]]]:
    roots = sorted((project_root / "artifacts" / "kie-labels").glob("*/validated"))
    sidecars: dict[str, LabelSidecar] = {}
    provenance_rows: list[dict[str, Any]] = []
    for record in records.values():
        expected_target = canonical_json(task.canonicalize(record.target))
        matches: list[tuple[Path, dict[str, Any], str, tuple[str, ...], bool]] = []
        mismatches: list[str] = []
        expected_leaves = {
            path: scalar_json
            for path, scalar_json, _ in _flatten_ordered(record.target["documentPatch"])
        }
        for root in roots:
            candidate = root / f"{record.document_id}.json"
            if not candidate.is_file():
                continue
            value = _json_load(candidate)
            if not isinstance(value, dict) or not isinstance(value.get("label"), dict):
                mismatches.append(str(candidate.relative_to(project_root)))
                continue
            try:
                actual_target = canonical_json(
                    task.canonicalize(cast(dict[str, Any], value["label"]))
                )
            except ValueError:
                mismatches.append(str(candidate.relative_to(project_root)))
                continue
            candidate_label = cast(dict[str, Any], value["label"])
            if actual_target == expected_target:
                matches.append((candidate, cast(dict[str, Any], value), "exact_target", (), True))
                continue
            actual_leaves = {
                path: scalar_json
                for path, scalar_json, _ in _flatten_ordered(candidate_label["documentPatch"])
            }
            if set(actual_leaves) != set(expected_leaves):
                mismatches.append(str(candidate.relative_to(project_root)))
                continue
            changed_paths = tuple(
                sorted(
                    path for path in expected_leaves if expected_leaves[path] != actual_leaves[path]
                )
            )
            changed_values_grounded = all(
                raw_grounding(json.loads(expected_leaves[path]), record.raw_text) != "not_grounded"
                for path in changed_paths
            )
            if not changed_values_grounded:
                mismatches.append(str(candidate.relative_to(project_root)))
                continue
            matches.append(
                (
                    candidate,
                    cast(dict[str, Any], value),
                    "path_compatible_raw_grounded_target_revision",
                    changed_paths,
                    changed_values_grounded,
                )
            )
        if not matches:
            provenance_rows.append(
                {
                    "document_id": record.document_id,
                    "split": record.split,
                    "matching_sidecars": 0,
                    "selected_sidecar": "",
                    "selected_match_type": "none",
                    "changed_leaf_count": 0,
                    "changed_paths": "",
                    "changed_values_grounded_in_current_raw": 0,
                    "nonmatching_candidates": len(mismatches),
                }
            )
            continue
        # Prefer the large final v2 publications over calibration artifacts, then use path order.
        matches.sort(
            key=lambda item: (
                item[2] != "exact_target",
                "followup420-r3" not in str(item[0]) and "pilot150-r1" not in str(item[0]),
                str(item[0]),
            )
        )
        selected_path, selected, match_type, changed_paths, changed_values_grounded = matches[0]
        evidence_items = selected.get("evidence")
        if not isinstance(evidence_items, list):
            raise ValueError(f"label sidecar has no evidence list: {selected_path}")
        evidence: dict[str, EvidenceRecord] = {}
        for item in evidence_items:
            if not isinstance(item, dict):
                raise ValueError(f"malformed evidence entry: {selected_path}")
            target_path = "$." + str(item["targetPath"])
            raw_items = item.get("rawOcrEvidence")
            if not isinstance(raw_items, list) or not raw_items:
                raise ValueError(f"evidence has no raw OCR entries: {selected_path}:{target_path}")
            evidence[target_path] = EvidenceRecord(
                target_path=target_path,
                evidence_kind=str(item["evidenceKind"]),
                normalization_rule=str(item.get("normalizationRule", "")),
                raw_values=tuple(str(raw_item["rawValue"]) for raw_item in raw_items),
                excerpts=tuple(str(raw_item["ocrExcerpt"]) for raw_item in raw_items),
                page_numbers=tuple(int(raw_item["pageNumber"]) for raw_item in raw_items),
            )
        expected_paths = set(expected_leaves)
        if set(evidence) != expected_paths:
            missing = sorted(expected_paths - set(evidence))
            unexpected = sorted(set(evidence) - expected_paths)
            raise ValueError(
                f"sidecar evidence differs from target leaves for {record.document_id}: missing={missing}, unexpected={unexpected}"
            )
        sidecars[record.document_id] = LabelSidecar(
            path=str(selected_path.relative_to(project_root)),
            evidence=evidence,
        )
        provenance_rows.append(
            {
                "document_id": record.document_id,
                "split": record.split,
                "matching_sidecars": len(matches),
                "selected_sidecar": str(selected_path.relative_to(project_root)),
                "selected_match_type": match_type,
                "changed_leaf_count": len(changed_paths),
                "changed_paths": " | ".join(changed_paths),
                "changed_values_grounded_in_current_raw": int(changed_values_grounded),
                "nonmatching_candidates": len(mismatches),
            }
        )
    return sidecars, provenance_rows


def _make_shingles(raw_text: str, width: int = 5) -> set[int]:
    tokens = []
    for token in TOKEN_PATTERN.findall(_space_normalize(raw_text)):
        tokens.append("<id>" if DIGIT_TOKEN_PATTERN.fullmatch(token) else token)
    if len(tokens) < width:
        return {int.from_bytes(hashlib.blake2b(" ".join(tokens).encode(), digest_size=8).digest())}
    return {
        int.from_bytes(
            hashlib.blake2b(
                " ".join(tokens[index : index + width]).encode(), digest_size=8
            ).digest()
        )
        for index in range(len(tokens) - width + 1)
    }


def _jaccard(left: set[int], right: set[int]) -> float:
    return _safe_divide(len(left & right), len(left | right))


def _carrier(target: Mapping[str, Any]) -> str:
    patch = target.get("documentPatch")
    if not isinstance(patch, dict):
        return "<absent>"
    parties = patch.get("parties")
    if not isinstance(parties, dict):
        return "<absent>"
    carrier = parties.get("carrier")
    if not isinstance(carrier, dict):
        return "<absent>"
    name = carrier.get("name")
    return str(name) if isinstance(name, str) else "<absent>"


def _structure_features(record: DatasetRecord) -> set[str]:
    patch = cast(Mapping[str, Any], record.target["documentPatch"])
    goods = patch.get("goodsItems")
    goods_items = cast(Sequence[Mapping[str, Any]], goods) if isinstance(goods, list) else ()
    containers = patch.get("containers")
    container_items = (
        cast(Sequence[Mapping[str, Any]], containers) if isinstance(containers, list) else ()
    )
    package_counts = [
        len(item["packages"]) for item in goods_items if isinstance(item.get("packages"), list)
    ]
    allocation_counts = [
        len(item["containerAllocations"])
        for item in goods_items
        if isinstance(item.get("containerAllocations"), list)
    ]
    features = {"all_documents"}
    features.add("multiple_goods_items" if len(goods_items) > 1 else "zero_or_one_goods_item")
    features.add(
        "nested_package_levels"
        if any(count > 1 for count in package_counts)
        else "at_most_one_package_level_per_item"
    )
    features.add("multiple_containers" if len(container_items) > 1 else "zero_or_one_container")
    if allocation_counts:
        features.add("has_container_allocations")
    if any(count > 1 for count in allocation_counts):
        features.add("multiple_allocations_in_one_goods_item")
    if any(isinstance(item.get("marksAndNumbers"), list) for item in goods_items):
        features.add("has_marks_and_numbers")
    if any(isinstance(item.get("additionalInformation"), list) for item in goods_items):
        features.add("has_additional_information")
    if record.page_count >= 3:
        features.add("three_or_more_pages")
    if (
        len(goods_items) > 1
        or any(count > 1 for count in package_counts)
        or any(count > 1 for count in allocation_counts)
    ):
        features.add("complex_goods_structure")
    else:
        features.add("simple_goods_structure")
    return features


def _structure_rows(
    records: Mapping[str, DatasetRecord], predictions: Mapping[str, PredictionRow]
) -> list[dict[str, Any]]:
    train_documents: Counter[str] = Counter()
    validation_documents: dict[str, list[str]] = defaultdict(list)
    for record in records.values():
        for feature in _structure_features(record):
            if record.split == "train":
                train_documents[feature] += 1
            elif record.split == "validation":
                validation_documents[feature].append(record.document_id)
    rows = []
    for feature in sorted(set(train_documents) | set(validation_documents)):
        document_ids = validation_documents[feature]
        counts = (
            _counts_for_documents(document_ids, predictions)
            if document_ids
            else _metric_counts(0, 0, 0)
        )
        rows.append(
            {
                "feature": feature,
                "train_documents": train_documents[feature],
                "validation_documents": len(document_ids),
                "precision": counts.precision,
                "recall": counts.recall,
                "f1": counts.f1,
                "reference_values": counts.reference,
            }
        )
    return rows


def _counts_for_documents(
    document_ids: Iterable[str], predictions: Mapping[str, PredictionRow]
) -> MetricCounts:
    true_positive = predicted = reference = 0
    for document_id in document_ids:
        assessment = predictions[document_id].assessment
        true_positive += len(assessment.predicted_field_values & assessment.reference_field_values)
        predicted += len(assessment.predicted_field_values)
        reference += len(assessment.reference_field_values)
    return _metric_counts(predicted, reference, true_positive)


def _counterfactual_fix(
    predictions: Mapping[str, PredictionRow],
    *,
    document_filter: set[str] | None = None,
    field_filter: str | None = None,
    section_filter: str | None = None,
) -> MetricCounts:
    predicted_total = reference_total = true_positive = 0
    for document_id, row in predictions.items():
        predicted = set(row.assessment.predicted_field_values)
        reference = set(row.assessment.reference_field_values)
        if document_filter is not None and document_id in document_filter:
            predicted = set(reference)
        elif field_filter is not None:
            predicted = {item for item in predicted if _normalize_path(item[0]) != field_filter}
            predicted.update(item for item in reference if _normalize_path(item[0]) == field_filter)
        elif section_filter is not None:
            predicted = {
                item for item in predicted if _section(_normalize_path(item[0])) != section_filter
            }
            predicted.update(
                item for item in reference if _section(_normalize_path(item[0])) == section_filter
            )
        true_positive += len(predicted & reference)
        predicted_total += len(predicted)
        reference_total += len(reference)
    return _metric_counts(predicted_total, reference_total, true_positive)


def _pearson(left: Sequence[float], right: Sequence[float]) -> float:
    if len(left) != len(right) or len(left) < 2:
        return 0.0
    left_mean = statistics.fmean(left)
    right_mean = statistics.fmean(right)
    numerator = sum((x - left_mean) * (y - right_mean) for x, y in zip(left, right, strict=True))
    denominator = math.sqrt(
        sum((x - left_mean) ** 2 for x in left) * sum((y - right_mean) ** 2 for y in right)
    )
    return _safe_divide(numerator, denominator)


def _rank(values: Sequence[float]) -> list[float]:
    indexed = sorted(enumerate(values), key=lambda item: item[1])
    ranks = [0.0] * len(values)
    index = 0
    while index < len(indexed):
        end = index + 1
        while end < len(indexed) and indexed[end][1] == indexed[index][1]:
            end += 1
        rank = (index + end - 1) / 2 + 1
        for original_index, _ in indexed[index:end]:
            ranks[original_index] = rank
        index = end
    return ranks


def _spearman(left: Sequence[float], right: Sequence[float]) -> float:
    return _pearson(_rank(left), _rank(right))


def _field_stats(
    records: Mapping[str, DatasetRecord],
    predictions: Mapping[str, PredictionRow],
    sidecars: Mapping[str, LabelSidecar],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    train_values: dict[str, Counter[str]] = defaultdict(Counter)
    train_documents: dict[str, set[str]] = defaultdict(set)
    train_evidence_kinds: dict[str, Counter[str]] = defaultdict(Counter)
    train_evidence_relations: dict[str, Counter[str]] = defaultdict(Counter)
    validation_documents: dict[str, set[str]] = defaultdict(set)
    occurrence_rows: list[dict[str, Any]] = []
    addition_rows: list[dict[str, Any]] = []

    for record in records.values():
        canonical_patch = cast(dict[str, Any], record.target["documentPatch"])
        leaves = _flatten_ordered(canonical_patch)
        if record.split == "train":
            for path, scalar_json, value in leaves:
                field = _normalize_path(path)
                train_values[field][scalar_json] += 1
                train_documents[field].add(record.document_id)
                sidecar = sidecars.get(record.document_id)
                evidence = sidecar.evidence.get(path) if sidecar is not None else None
                if evidence is not None:
                    train_evidence_kinds[field][evidence.evidence_kind] += 1
                train_evidence_relations[field][_target_evidence_relation(value, evidence)] += 1
            continue

        row = predictions[record.document_id]
        predicted_fields = (
            _flatten_ordered(_parse_patch(row.generated_text))
            if _parse_patch(row.generated_text) is not None
            else []
        )
        sidecar = sidecars.get(record.document_id)
        reference_pairs = {(path, scalar_json) for path, scalar_json, _ in leaves}
        predicted_pairs = {(path, scalar_json) for path, scalar_json, _ in predicted_fields}
        target_leaf_count = len(leaves)
        for target_ordinal, (path, scalar_json, value) in enumerate(leaves):
            field = _normalize_path(path)
            validation_documents[field].add(record.document_id)
            match_class, predicted_path, predicted_value = classify_reference_match(
                path, scalar_json, value, predicted_fields
            )
            similarity = (
                value_similarity(value, predicted_value)
                if predicted_value is not None
                else {
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
            evidence = sidecar.evidence.get(path) if sidecar is not None else None
            occurrence_rows.append(
                {
                    "document_id": record.document_id,
                    "cohort": record.cohort,
                    "page_count": record.page_count,
                    "field_path": field,
                    "indexed_reference_path": path,
                    "indexed_predicted_path": predicted_path,
                    "reference_value": value,
                    "predicted_value": predicted_value,
                    "match_class": match_class,
                    "strict_correct": int((path, scalar_json) in predicted_pairs),
                    "exact_value_seen_in_train": int(train_values[field][scalar_json] > 0),
                    "exact_train_value_frequency": train_values[field][scalar_json],
                    "reference_evidence_kind": evidence.evidence_kind if evidence else "",
                    "reference_evidence_relation": _target_evidence_relation(value, evidence),
                    "reference_source_position": _evidence_position(record, evidence),
                    "target_sequence_position": target_ordinal / max(1, target_leaf_count - 1),
                    "prediction_raw_grounding": raw_grounding(predicted_value, record.raw_text)
                    if predicted_value is not None
                    else "absent",
                    "reference_context": evidence.excerpts[0]
                    if evidence
                    else _find_context(value, record.raw_text),
                    "prediction_context": _find_context(predicted_value, record.raw_text)
                    if predicted_value is not None
                    else "",
                    **similarity,
                }
            )
        for path, scalar_json, value in predicted_fields:
            if (path, scalar_json) in reference_pairs:
                continue
            field = _normalize_path(path)
            same_field_references = [item for item in leaves if _normalize_path(item[0]) == field]
            exact_elsewhere = any(
                reference_json == scalar_json for _, reference_json, _ in same_field_references
            )
            same_path = any(
                reference_path == path for reference_path, _, _ in same_field_references
            )
            addition_rows.append(
                {
                    "document_id": record.document_id,
                    "cohort": record.cohort,
                    "field_path": field,
                    "indexed_predicted_path": path,
                    "predicted_value": value,
                    "addition_class": (
                        "exact_value_other_index"
                        if exact_elsewhere
                        else "wrong_value_at_reference_path"
                        if same_path
                        else "extra_field_or_list_item"
                    ),
                    "prediction_raw_grounding": raw_grounding(value, record.raw_text),
                    "prediction_context": _find_context(value, record.raw_text),
                }
            )

    all_fields = sorted(
        set(train_values)
        | {str(row["field_path"]) for row in occurrence_rows}
        | {str(row["field_path"]) for row in addition_rows}
    )
    field_rows: list[dict[str, Any]] = []
    for field in all_fields:
        references = [row for row in occurrence_rows if row["field_path"] == field]
        additions = [row for row in addition_rows if row["field_path"] == field]
        true_positive = sum(int(row["strict_correct"]) for row in references)
        reference_count = len(references)
        predicted_count = true_positive + len(additions)
        counts = _metric_counts(predicted_count, reference_count, true_positive)
        seen = [row for row in references if row["exact_value_seen_in_train"]]
        unseen = [row for row in references if not row["exact_value_seen_in_train"]]
        evidence_total = sum(train_evidence_kinds[field].values())
        direct_copy = (
            train_evidence_relations[field]["verbatim_raw_value"]
            + train_evidence_relations[field]["case_or_whitespace_join"]
            + train_evidence_relations[field]["punctuation_normalized"]
        )
        grounded_additions = sum(
            row["prediction_raw_grounding"] != "not_grounded" for row in additions
        )
        match_counter = Counter(str(row["match_class"]) for row in references)
        field_rows.append(
            {
                "field_path": field,
                "section": _section(field),
                "train_documents": len(train_documents[field]),
                "train_document_prevalence": _safe_divide(
                    len(train_documents[field]),
                    sum(record.split == "train" for record in records.values()),
                ),
                "train_values": sum(train_values[field].values()),
                "train_unique_values": len(train_values[field]),
                "train_singleton_values": sum(count == 1 for count in train_values[field].values()),
                "train_unique_value_ratio": _safe_divide(
                    len(train_values[field]), sum(train_values[field].values())
                ),
                "train_direct_copy_evidence_rate": _safe_divide(direct_copy, evidence_total),
                "train_normalized_or_contextual_evidence_rate": _safe_divide(
                    evidence_total - train_evidence_kinds[field]["verbatim"], evidence_total
                ),
                "validation_documents": len(validation_documents[field]),
                "validation_reference_values": reference_count,
                "validation_predicted_values": predicted_count,
                "validation_exact_seen_values": len(seen),
                "validation_exact_seen_rate": _safe_divide(len(seen), reference_count),
                "seen_value_recall": _safe_divide(
                    sum(int(row["strict_correct"]) for row in seen), len(seen)
                ),
                "unseen_value_recall": _safe_divide(
                    sum(int(row["strict_correct"]) for row in unseen), len(unseen)
                ),
                "true_positive": true_positive,
                "false_positive": counts.predicted - counts.true_positive,
                "false_negative": counts.reference - counts.true_positive,
                "precision": counts.precision,
                "recall": counts.recall,
                "f1": counts.f1,
                "exact_value_other_index": match_counter["exact_value_other_index"],
                "same_path_wrong_value": match_counter["same_path_wrong_value"],
                "same_field_other_value": match_counter["same_field_other_value"],
                "field_absent": match_counter["field_absent"],
                "grounded_false_positive_rate": _safe_divide(grounded_additions, len(additions)),
            }
        )
    return field_rows, occurrence_rows, addition_rows


def _template_rows(
    records: Mapping[str, DatasetRecord], predictions: Mapping[str, PredictionRow]
) -> list[dict[str, Any]]:
    train_records = [record for record in records.values() if record.split == "train"]
    validation_records = [record for record in records.values() if record.split == "validation"]
    train_shingles = {
        record.document_id: _make_shingles(record.raw_text) for record in train_records
    }
    rows: list[dict[str, Any]] = []
    for record in validation_records:
        validation_shingles = _make_shingles(record.raw_text)
        nearest_id, similarity = max(
            (
                (
                    train_record.document_id,
                    _jaccard(validation_shingles, train_shingles[train_record.document_id]),
                )
                for train_record in train_records
            ),
            key=lambda item: item[1],
        )
        counts = _counts_for_documents([record.document_id], predictions)
        rows.append(
            {
                "document_id": record.document_id,
                "cohort": record.cohort,
                "carrier": _carrier(record.target),
                "nearest_train_document_id": nearest_id,
                "nearest_train_template_jaccard": similarity,
                "nearest_train_carrier": _carrier(records[nearest_id].target),
                "same_carrier": int(
                    _carrier(record.target) == _carrier(records[nearest_id].target)
                ),
                "field_value_precision": counts.precision,
                "field_value_recall": counts.recall,
                "field_value_f1": counts.f1,
            }
        )
    return rows


def _carrier_rows(
    records: Mapping[str, DatasetRecord], predictions: Mapping[str, PredictionRow]
) -> list[dict[str, Any]]:
    train_carriers: Counter[str] = Counter(
        _carrier(record.target) for record in records.values() if record.split == "train"
    )
    validation: dict[str, list[str]] = defaultdict(list)
    for record in records.values():
        if record.split == "validation":
            validation[_carrier(record.target)].append(record.document_id)
    rows = []
    for carrier, document_ids in sorted(
        validation.items(), key=lambda item: (-len(item[1]), item[0])
    ):
        counts = _counts_for_documents(document_ids, predictions)
        rows.append(
            {
                "carrier": carrier,
                "train_documents": train_carriers[carrier],
                "validation_documents": len(document_ids),
                "precision": counts.precision,
                "recall": counts.recall,
                "f1": counts.f1,
                "reference_values": counts.reference,
            }
        )
    return rows


def _counterfactual_rows(
    predictions: Mapping[str, PredictionRow], field_rows: Sequence[Mapping[str, Any]]
) -> list[dict[str, Any]]:
    baseline = _counterfactual_fix(predictions)
    rows: list[dict[str, Any]] = [
        {
            "intervention": "observed_baseline",
            "scope": "all",
            "precision": baseline.precision,
            "recall": baseline.recall,
            "f1": baseline.f1,
            "absolute_f1_gain": 0.0,
            "interpretation": "Actual saved prediction pairs.",
        }
    ]
    invalid_json = {
        document_id for document_id, row in predictions.items() if not row.assessment.json_valid
    }
    invalid_schema = {
        document_id for document_id, row in predictions.items() if not row.assessment.schema_valid
    }
    for name, documents, interpretation in (
        (
            "perfectly_replace_invalid_json_documents",
            invalid_json,
            "Optimistic ceiling for any syntax-constraining strategy; assumes every currently invalid document becomes semantically perfect.",
        ),
        (
            "perfectly_replace_all_schema_invalid_documents",
            invalid_schema,
            "Even more optimistic ceiling for perfect schema enforcement plus perfect semantic repair on every invalid document.",
        ),
    ):
        counts = _counterfactual_fix(predictions, document_filter=documents)
        rows.append(
            {
                "intervention": name,
                "scope": f"{len(documents)} documents",
                "precision": counts.precision,
                "recall": counts.recall,
                "f1": counts.f1,
                "absolute_f1_gain": counts.f1 - baseline.f1,
                "interpretation": interpretation,
            }
        )
    sections = sorted({str(row["section"]) for row in field_rows})
    for section in sections:
        counts = _counterfactual_fix(predictions, section_filter=section)
        rows.append(
            {
                "intervention": "perfect_section",
                "scope": section,
                "precision": counts.precision,
                "recall": counts.recall,
                "f1": counts.f1,
                "absolute_f1_gain": counts.f1 - baseline.f1,
                "interpretation": "Non-additive oracle: replace every prediction leaf in this section with its reference leaves.",
            }
        )
    for field in sorted(
        field_rows,
        key=lambda row: int(row["false_positive"]) + int(row["false_negative"]),
        reverse=True,
    )[:20]:
        field_path = str(field["field_path"])
        counts = _counterfactual_fix(predictions, field_filter=field_path)
        rows.append(
            {
                "intervention": "perfect_field",
                "scope": field_path,
                "precision": counts.precision,
                "recall": counts.recall,
                "f1": counts.f1,
                "absolute_f1_gain": counts.f1 - baseline.f1,
                "interpretation": "Non-additive oracle: replace this normalized field with its reference values.",
            }
        )
    return rows


def _prior_run_rows(project_root: Path) -> list[dict[str, Any]]:
    rows = []
    for run_dir in sorted(
        (project_root / "artifacts" / "kie-training").glob("t5gemma2-270m-lora-*")
    ):
        config_path = run_dir / "resolved-config.json"
        eval_path = run_dir / "checkpoints" / "eval_results.json"
        if not config_path.is_file() or not eval_path.is_file():
            continue
        config = _json_load(config_path)
        metrics = _json_load(eval_path)
        train_sources = config["dataset"]["splits"]["train"]
        validation_sources = config["dataset"]["splits"]["validation"]
        rows.append(
            {
                "run_id": run_dir.name,
                "train_documents": sum(int(source["records"]) for source in train_sources),
                "validation_documents": sum(
                    int(source["records"]) for source in validation_sources
                ),
                "epochs": float(config["optimization"]["num_train_epochs"]),
                "generation_max_length": int(config["evaluation"]["generation_max_length"]),
                "json_valid": float(metrics.get("eval_json_valid", 0.0)),
                "schema_valid": float(metrics.get("eval_schema_valid", 0.0)),
                "precision": float(metrics.get("eval_field_value_precision", 0.0)),
                "recall": float(metrics.get("eval_field_value_recall", 0.0)),
                "f1": float(metrics.get("eval_field_value_f1", 0.0)),
                "eos_reached": float(metrics.get("eval_generation_eos_reached_fraction", 0.0)),
                "comparable_note": "Different splits/configurations; trend only, not a controlled scaling curve.",
            }
        )
    return rows


def _category_rows(
    occurrence_rows: Sequence[Mapping[str, Any]], field: str
) -> list[dict[str, Any]]:
    rows = [
        dict(row)
        for row in occurrence_rows
        if row["field_path"] == field and not row["strict_correct"]
    ]
    for row in rows:
        if row["predicted_value"] is None:
            distance_class = "absent"
        elif row["match_class"] == "exact_value_other_index":
            distance_class = "correct_value_wrong_index"
        elif row["space_normalized_exact"] or row["alnum_normalized_exact"]:
            distance_class = "format_only"
        elif row["prediction_contains_reference"] or row["reference_contains_prediction"]:
            distance_class = "partial_or_contaminated_copy"
        elif float(row["sequence_similarity"]) >= 0.9:
            distance_class = "near_copy"
        elif float(row["sequence_similarity"]) >= 0.6:
            distance_class = "related_copy"
        else:
            distance_class = "far_or_wrong_selection"
        row["distance_class"] = distance_class
    return rows


def _numeric_error_class(row: Mapping[str, Any]) -> str:
    if row["predicted_value"] is None:
        return "absent"
    if row["match_class"] == "exact_value_other_index":
        return "correct_value_wrong_index"
    absolute = row["numeric_absolute_delta"]
    relative = row["numeric_relative_absolute_error"]
    if absolute is None or relative is None:
        return "non_numeric_prediction"
    if float(absolute) <= 1:
        return "off_by_at_most_1"
    if float(relative) <= 0.05:
        return "within_5_percent"
    if row["prediction_raw_grounding"] != "not_grounded":
        return "wrong_grounded_number"
    return "ungrounded_number"


def _position_rows(
    occurrence_rows: Sequence[Mapping[str, Any]], key: str, label: str
) -> list[dict[str, Any]]:
    buckets: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for row in occurrence_rows:
        value = row[key]
        if value is None:
            continue
        number = float(value)
        index = min(3, int(number * 4))
        buckets[("Q1", "Q2", "Q3", "Q4")[index]].append(row)
    output = []
    for bucket in ("Q1", "Q2", "Q3", "Q4"):
        values = buckets[bucket]
        output.append(
            {
                "position_type": label,
                "quartile": bucket,
                "reference_values": len(values),
                "exact_values": sum(int(row["strict_correct"]) for row in values),
                "recall": _safe_divide(
                    sum(int(row["strict_correct"]) for row in values), len(values)
                ),
            }
        )
    return output


def _render_heatmap(
    path: Path,
    title: str,
    subtitle: str,
    rows: Sequence[str],
    columns: Sequence[str],
    values: Mapping[tuple[str, str], int],
) -> None:
    image, draw = _chart_canvas(title, subtitle)
    left, top, right, bottom = 510, 160, 1510, 790
    maximum = max(values.values(), default=1)
    cell_width = (right - left) / max(1, len(columns))
    cell_height = (bottom - top) / max(1, len(rows))
    for row_index, row in enumerate(rows):
        label = row if len(row) <= 24 else row[:23] + "…"
        width = draw.textbbox((0, 0), label, font=_font(14))[2]
        draw.text(
            (left - width - 12, top + (row_index + 0.5) * cell_height - 8),
            label,
            fill="#172033",
            font=_font(14),
        )
        for column_index, column in enumerate(columns):
            count = values.get((row, column), 0)
            intensity = count / maximum
            color = (
                int(245 - 180 * intensity),
                int(248 - 130 * intensity),
                int(255 - 40 * intensity),
            )
            x0 = left + column_index * cell_width
            y0 = top + row_index * cell_height
            draw.rectangle((x0, y0, x0 + cell_width - 2, y0 + cell_height - 2), fill=color)
            if count:
                text = str(count)
                text_width = draw.textbbox((0, 0), text, font=_font(14, bold=True))[2]
                draw.text(
                    (x0 + (cell_width - text_width) / 2, y0 + cell_height / 2 - 8),
                    text,
                    fill="#172033",
                    font=_font(14, bold=True),
                )
    for column_index, column in enumerate(columns):
        label = column if len(column) <= 18 else column[:17] + "…"
        width = draw.textbbox((0, 0), label, font=_font(13))[2]
        draw.text(
            (left + (column_index + 0.5) * cell_width - width / 2, bottom + 12),
            label,
            fill="#172033",
            font=_font(13),
        )
    path.parent.mkdir(parents=True, exist_ok=True)
    image.save(path, optimize=True)


def _render_plots(
    output_dir: Path,
    field_rows: Sequence[Mapping[str, Any]],
    occurrence_rows: Sequence[Mapping[str, Any]],
    addition_rows: Sequence[Mapping[str, Any]],
    counterfactual_rows: Sequence[Mapping[str, Any]],
    template_rows: Sequence[Mapping[str, Any]],
    prior_runs: Sequence[Mapping[str, Any]],
    position_rows: Sequence[Mapping[str, Any]],
    structure_rows: Sequence[Mapping[str, Any]],
) -> list[str]:
    plot_dir = output_dir / "plots"
    plot_dir.mkdir(parents=True, exist_ok=False)
    paths: list[str] = []

    supported = [row for row in field_rows if int(row["validation_reference_values"]) >= 8]
    _scatter_chart(
        plot_dir / "01_field_f1_vs_train_support.png",
        "Field F1 versus training-document support",
        "Fields with at least 8 validation values; support alone does not encode semantic/list difficulty",
        [
            (
                "fields",
                [math.log10(int(row["train_documents"]) + 1) for row in supported],
                [float(row["f1"]) for row in supported],
                PALETTE[0],
            )
        ],
        "log10(train documents + 1)",
        "Exact field/value F1",
        y_range=(0.0, 1.0),
    )
    paths.append("plots/01_field_f1_vs_train_support.png")

    weak = sorted(supported, key=lambda row: float(row["f1"]))[:15]
    _horizontal_bar_chart(
        plot_dir / "02_weak_field_training_support.png",
        "Training support behind the weakest evaluated fields",
        "Document counts, not scalar counts; one complex document may contribute many repeated leaves",
        [str(row["field_path"]) for row in weak],
        [float(row["train_documents"]) for row in weak],
        x_label="Training documents containing field",
    )
    paths.append("plots/02_weak_field_training_support.png")

    seen_fields = [
        row
        for row in supported
        if int(row["validation_exact_seen_values"])
        and int(row["validation_exact_seen_values"]) < int(row["validation_reference_values"])
    ]
    seen_fields = sorted(
        seen_fields, key=lambda row: int(row["validation_reference_values"]), reverse=True
    )[:12]
    if seen_fields:
        _grouped_bar_chart(
            plot_dir / "03_seen_vs_unseen_value_recall.png",
            "Exact-value recall: seen versus unseen values",
            "A value is seen only when the same canonical scalar occurred under the same normalized field in training",
            [str(row["field_path"]).split(".")[-1] for row in seen_fields],
            [
                ("seen", [float(row["seen_value_recall"]) for row in seen_fields], PALETTE[2]),
                ("unseen", [float(row["unseen_value_recall"]) for row in seen_fields], PALETTE[1]),
            ],
            "Recall",
            y_max=1.0,
        )
        paths.append("plots/03_seen_vs_unseen_value_recall.png")

    focus = [row for row in field_rows if row["field_path"] in FOCUS_FIELDS]
    _grouped_bar_chart(
        plot_dir / "04_focus_field_precision_recall_f1.png",
        "Cargo-field precision, recall, and F1",
        "The requested package/quantity/description fields plus their structural neighbors",
        [str(row["field_path"]).replace("$.documentPatch.goodsItems[].", "") for row in focus],
        [
            ("precision", [float(row["precision"]) for row in focus], PALETTE[0]),
            ("recall", [float(row["recall"]) for row in focus], PALETTE[1]),
            ("F1", [float(row["f1"]) for row in focus], PALETTE[2]),
        ],
        "Metric",
        y_max=1.0,
    )
    paths.append("plots/04_focus_field_precision_recall_f1.png")

    error_classes = (
        "exact_value_other_index",
        "same_path_wrong_value",
        "same_field_other_value",
        "field_absent",
    )
    _grouped_bar_chart(
        plot_dir / "05_focus_field_error_modes.png",
        "How cargo-field reference values fail",
        "Each reference error is assigned to one mutually exclusive diagnostic class",
        [str(row["field_path"]).replace("$.documentPatch.goodsItems[].", "") for row in focus],
        [
            (name.replace("_", " "), [float(row[name]) for row in focus], PALETTE[index])
            for index, name in enumerate(error_classes)
        ],
        "Reference errors",
        y_max=max(1.0, max(sum(float(row[name]) for name in error_classes) for row in focus) * 1.1),
    )
    paths.append("plots/05_focus_field_error_modes.png")

    package_type_errors = _category_rows(occurrence_rows, PACKAGE_TYPE_FIELD)
    package_type_counts = Counter(str(row["distance_class"]) for row in package_type_errors)
    _horizontal_bar_chart(
        plot_dir / "06_package_type_error_distance.png",
        "Package-type error distance",
        "Format-only and wrong-index errors are separated from genuinely wrong semantic selections",
        list(package_type_counts),
        [float(package_type_counts[key]) for key in package_type_counts],
        x_label="Reference errors",
    )
    paths.append("plots/06_package_type_error_distance.png")

    quantity_errors = [
        dict(row)
        for row in occurrence_rows
        if row["field_path"] == PACKAGE_QUANTITY_FIELD and not row["strict_correct"]
    ]
    quantity_counts = Counter(_numeric_error_class(row) for row in quantity_errors)
    _horizontal_bar_chart(
        plot_dir / "07_package_quantity_error_distance.png",
        "Package-quantity error distance",
        "Wrong grounded numbers are OCR values assigned to the wrong semantic row/level, not invented digits",
        list(quantity_counts),
        [float(quantity_counts[key]) for key in quantity_counts],
        x_label="Reference errors",
    )
    paths.append("plots/07_package_quantity_error_distance.png")

    numeric_relative = [
        min(2.0, float(row["numeric_relative_absolute_error"]))
        for row in quantity_errors
        if row["numeric_relative_absolute_error"] is not None
        and row["match_class"] != "exact_value_other_index"
    ]
    if numeric_relative:
        _histogram(
            plot_dir / "08_package_quantity_relative_error.png",
            "Relative magnitude of wrong package quantities",
            "Values above 200% are clipped to 200% for display",
            numeric_relative,
            12,
            "Relative absolute error (clipped)",
        )
        paths.append("plots/08_package_quantity_relative_error.png")

    description_errors = _category_rows(occurrence_rows, DESCRIPTION_FIELD)
    similarities = [
        float(row["sequence_similarity"])
        for row in description_errors
        if row["predicted_value"] is not None
    ]
    if similarities:
        _histogram(
            plot_dir / "09_description_similarity.png",
            "Goods-description similarity when exact match fails",
            "Sequence similarity is diagnostic only; the production metric remains exact",
            similarities,
            10,
            "Case/whitespace-normalized sequence similarity",
        )
        paths.append("plots/09_description_similarity.png")
    description_counts = Counter(str(row["distance_class"]) for row in description_errors)
    _horizontal_bar_chart(
        plot_dir / "10_description_error_distance.png",
        "Goods-description error classes",
        "Separates missing items, partial/contaminated copies, row shifts, and far selections",
        list(description_counts),
        [float(description_counts[key]) for key in description_counts],
        x_label="Reference errors",
    )
    paths.append("plots/10_description_error_distance.png")

    focus_additions = [row for row in addition_rows if row["field_path"] in FOCUS_FIELDS]
    grounding_counts = Counter(str(row["prediction_raw_grounding"]) for row in focus_additions)
    _horizontal_bar_chart(
        plot_dir / "11_focus_false_positive_grounding.png",
        "Are extra cargo values present in raw OCR?",
        "Grounded extras usually indicate semantic selection or row-association errors rather than free hallucination",
        list(grounding_counts),
        [float(grounding_counts[key]) for key in grounding_counts],
        x_label="Unmatched predicted leaves",
    )
    paths.append("plots/11_focus_false_positive_grounding.png")

    section_oracles = sorted(
        [row for row in counterfactual_rows if row["intervention"] == "perfect_section"],
        key=lambda row: float(row["absolute_f1_gain"]),
        reverse=True,
    )[:10]
    _horizontal_bar_chart(
        plot_dir / "12_oracle_section_f1_gain.png",
        "Global F1 headroom by perfectly fixing one section",
        "Non-additive oracle analysis; this ranks error-budget concentration, not deployable interventions",
        [str(row["scope"]) for row in section_oracles],
        [float(row["absolute_f1_gain"]) for row in section_oracles],
        x_label="Absolute global F1 gain",
    )
    paths.append("plots/12_oracle_section_f1_gain.png")

    validity_oracles = [
        row
        for row in counterfactual_rows
        if row["intervention"]
        in {
            "observed_baseline",
            "perfectly_replace_invalid_json_documents",
            "perfectly_replace_all_schema_invalid_documents",
        }
    ]
    _horizontal_bar_chart(
        plot_dir / "13_validity_oracle_ceiling.png",
        "Even optimistic validity repair has a semantic ceiling",
        "Invalid outputs are replaced with perfect references; real constrained decoding cannot guarantee this semantic gain",
        [str(row["intervention"]) for row in validity_oracles],
        [float(row["f1"]) for row in validity_oracles],
        x_label="Global field/value F1",
        x_max=1.0,
    )
    paths.append("plots/13_validity_oracle_ceiling.png")

    _scatter_chart(
        plot_dir / "14_template_similarity_vs_f1.png",
        "Validation performance versus nearest training template",
        "Five-token Jaccard after replacing digit-bearing tokens; high similarity means the random split is template-friendly",
        [
            (
                "validation documents",
                [float(row["nearest_train_template_jaccard"]) for row in template_rows],
                [float(row["field_value_f1"]) for row in template_rows],
                PALETTE[0],
            )
        ],
        "Nearest-train template Jaccard",
        "Document F1",
        y_range=(0.0, 1.0),
    )
    paths.append("plots/14_template_similarity_vs_f1.png")

    source_position = [row for row in position_rows if row["position_type"] == "source_evidence"]
    target_position = [row for row in position_rows if row["position_type"] == "target_sequence"]
    _grouped_bar_chart(
        plot_dir / "15_recall_by_source_and_target_position.png",
        "Recall by source-evidence and target-sequence position",
        "A position gradient would indicate long-context retrieval or autoregressive tail degradation",
        [str(row["quartile"]) for row in source_position],
        [
            ("source evidence", [float(row["recall"]) for row in source_position], PALETTE[0]),
            ("target sequence", [float(row["recall"]) for row in target_position], PALETTE[1]),
        ],
        "Exact-value recall",
        y_max=1.0,
    )
    paths.append("plots/15_recall_by_source_and_target_position.png")

    comparable_runs = [row for row in prior_runs if float(row["f1"]) > 0]
    if comparable_runs:
        _scatter_chart(
            plot_dir / "16_observed_run_scale_trend.png",
            "Observed pilot-to-combined trend",
            "Configurations and validation splits differ; this is evidence of direction, not a fitted learning curve",
            [
                (
                    "runs",
                    [float(row["train_documents"]) for row in comparable_runs],
                    [float(row["f1"]) for row in comparable_runs],
                    PALETTE[2],
                )
            ],
            "Training documents",
            "Final saved field/value F1",
            y_range=(0.0, 1.0),
        )
        paths.append("plots/16_observed_run_scale_trend.png")

    package_pairs: Counter[tuple[str, str]] = Counter()
    for row in package_type_errors:
        reference = str(row["reference_value"])
        predicted = "<absent>" if row["predicted_value"] is None else str(row["predicted_value"])
        package_pairs[(reference, predicted)] += 1
    top_references = [
        key
        for key, _ in Counter(reference for reference, _ in package_pairs.elements()).most_common(
            12
        )
    ]
    top_predictions = [
        key
        for key, _ in Counter(predicted for _, predicted in package_pairs.elements()).most_common(
            12
        )
    ]
    if top_references and top_predictions:
        _render_heatmap(
            plot_dir / "17_package_type_confusion.png",
            "Package-type reference versus best predicted value",
            "Best same-field candidate; absent means no package-type value was generated for the document",
            top_references,
            top_predictions,
            package_pairs,
        )
        paths.append("plots/17_package_type_confusion.png")
    structure_order = (
        "multiple_goods_items",
        "nested_package_levels",
        "multiple_containers",
        "has_container_allocations",
        "multiple_allocations_in_one_goods_item",
        "has_marks_and_numbers",
        "has_additional_information",
        "three_or_more_pages",
        "simple_goods_structure",
        "complex_goods_structure",
    )
    structure_by_name = {str(row["feature"]): row for row in structure_rows}
    selected_structures = [
        structure_by_name[name] for name in structure_order if name in structure_by_name
    ]
    _horizontal_bar_chart(
        plot_dir / "18_structural_cohort_training_support.png",
        "Training documents by structural cargo cohort",
        "Cohorts overlap; counts expose whether complex table shapes are actually demonstrated",
        [str(row["feature"]) for row in selected_structures],
        [float(row["train_documents"]) for row in selected_structures],
        x_label="Training documents",
    )
    paths.append("plots/18_structural_cohort_training_support.png")
    evaluated_structures = [
        row for row in selected_structures if int(row["validation_documents"]) >= 2
    ]
    _horizontal_bar_chart(
        plot_dir / "19_structural_cohort_validation_f1.png",
        "Validation F1 by structural cargo cohort",
        "Overlapping diagnostic cohorts; small cohorts must not be interpreted as independent estimates",
        [str(row["feature"]) for row in evaluated_structures],
        [float(row["f1"]) for row in evaluated_structures],
        x_label="Exact field/value F1",
        x_max=1.0,
    )
    paths.append("plots/19_structural_cohort_validation_f1.png")
    return paths


def _report(
    output_dir: Path,
    run_id: str,
    summary: Mapping[str, Any],
    field_rows: Sequence[Mapping[str, Any]],
    counterfactual_rows: Sequence[Mapping[str, Any]],
    template_rows: Sequence[Mapping[str, Any]],
    position_rows: Sequence[Mapping[str, Any]],
    structure_rows: Sequence[Mapping[str, Any]],
    plot_paths: Sequence[str],
) -> None:
    focus_by_name = {str(row["field_path"]): row for row in field_rows}
    supported = [row for row in field_rows if int(row["validation_reference_values"]) >= 8]
    log_support = [math.log10(int(row["train_documents"]) + 1) for row in supported]
    f1_values = [float(row["f1"]) for row in supported]
    validity_oracles = {str(row["intervention"]): row for row in counterfactual_rows}
    section_oracles = sorted(
        [row for row in counterfactual_rows if row["intervention"] == "perfect_section"],
        key=lambda row: float(row["absolute_f1_gain"]),
        reverse=True,
    )
    nearest_values = sorted(float(row["nearest_train_template_jaccard"]) for row in template_rows)
    source_positions = {
        str(row["quartile"]): row
        for row in position_rows
        if row["position_type"] == "source_evidence"
    }
    target_positions = {
        str(row["quartile"]): row
        for row in position_rows
        if row["position_type"] == "target_sequence"
    }
    structures = {str(row["feature"]): row for row in structure_rows}

    package_type = focus_by_name[PACKAGE_TYPE_FIELD]
    package_quantity = focus_by_name[PACKAGE_QUANTITY_FIELD]
    description = focus_by_name[DESCRIPTION_FIELD]
    marks = focus_by_name[MARKS_FIELD]
    additional = focus_by_name[ADDITIONAL_INFORMATION_FIELD]
    type_errors = cast(Mapping[str, Any], summary["focus_error_distance"])["package_type"]
    quantity_errors = cast(Mapping[str, Any], summary["focus_error_distance"])["package_quantity"]
    description_errors = cast(Mapping[str, Any], summary["focus_error_distance"])[
        "goods_description"
    ]
    focus_additions = cast(Mapping[str, Any], summary["focus_error_distance"])[
        "unmatched_focus_predictions"
    ]

    lines = [
        f"# Reliability audit — {run_id}",
        "",
        f"Generated at `{summary['generated_at']}` without loading model weights or using the GPU. The completed run and the earlier analysis remain immutable; this is a separate diagnostic publication.",
        "",
        "## Executive verdict",
        "",
        f"The approach is learning the task, but the present evidence does **not** support production reliability or a claim near 0.90. The saved run reaches **{summary['baseline']['f1']:.3f} micro-F1** (precision `{summary['baseline']['precision']:.3f}`, recall `{summary['baseline']['recall']:.3f}`) and **{summary['baseline']['accuracy']:.3f} exact path-union accuracy**, while JSON/schema validity are `{summary['baseline']['json_valid']:.3f}` / `{summary['baseline']['schema_valid']:.3f}`. The gap is not primarily a sampling-hyperparameter problem: the epoch curve was nearly saturated, all text LoRA attention/MLP projections were adapted, and the main deficit is semantic recall plus cargo-row/list construction.",
        "",
        f"To reach F1 `{TARGET_F1:.2f}` from the observed counts while adding correct values with no new false positives would require recovering **{summary['target_gap']['perfect_new_true_positives_needed']}** of the current `{summary['target_gap']['false_negatives']}` missing reference leaves. At the current precision, recall would have to rise to `{summary['target_gap']['recall_needed_at_current_precision']:.3f}`; improving precision alone cannot reach 0.90 at the current recall. This makes omission/coverage work mandatory.",
        "",
        f"The project metric named `field_value_accuracy` is TP divided by the union of compared indexed paths, not ordinary classification accuracy. Reaching 0.90 on that stricter metric requires **{summary['target_gap']['true_positives_needed_for_accuracy_with_current_path_union']}** more exact paths if the current path union is unchanged, or **{summary['target_gap']['true_positives_needed_for_accuracy_after_removing_all_extra_paths']}** more after first removing every extra predicted path. Both the accuracy and F1 gates therefore require semantic corrections, not just valid JSON.",
        "",
        "## What the requested cargo fields actually look like",
        "",
        "| Field | Train docs | Train values | Val values | P | R | F1 | Correct value, wrong index | Field absent |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in (package_type, package_quantity, description, marks, additional):
        lines.append(
            f"| `{row['field_path']}` | {row['train_documents']} | {row['train_values']} | {row['validation_reference_values']} | {float(row['precision']):.3f} | {float(row['recall']):.3f} | {float(row['f1']):.3f} | {row['exact_value_other_index']} | {row['field_absent']} |"
        )
    lines.extend(
        [
            "",
            f"The error distance is unusually actionable. Package type has `{type_errors['reference_errors']}` reference misses: `{type_errors['categories'].get('correct_value_wrong_index', 0)}` are the right value at another list index, `{type_errors['categories'].get('absent', 0)}` are omissions, and only `{type_errors['reference_errors'] - type_errors['categories'].get('correct_value_wrong_index', 0) - type_errors['categories'].get('absent', 0)}` are other value selections. Every one of its `{type_errors['non_absent_best_predictions']}` non-absent best candidates is in raw OCR (`{type_errors['non_absent_grounded_in_raw']}/{type_errors['non_absent_best_predictions']}`).",
            "",
            f"Package quantity has `{quantity_errors['reference_errors']}` misses: `{quantity_errors['categories'].get('correct_value_wrong_index', 0)}` wrong-index, `{quantity_errors['categories'].get('absent', 0)}` absent, `{quantity_errors['categories'].get('wrong_grounded_number', 0)}` wrong-but-grounded numbers, and `{quantity_errors['categories'].get('within_5_percent', 0)}` near numeric error. All `{quantity_errors['non_absent_best_predictions']}` non-absent candidates are OCR-grounded. Goods-description misses are `{description_errors['categories'].get('partial_or_contaminated_copy', 0)}` partial/contaminated copies, `{description_errors['categories'].get('correct_value_wrong_index', 0)}` row shifts, and `{description_errors['categories'].get('absent', 0)}` omissions; `{description_errors['non_absent_grounded_in_raw']}/{description_errors['non_absent_best_predictions']}` non-absent best candidates are grounded.",
            "",
            f"Across all unmatched predictions in the focus cargo fields, `{focus_additions['grounded_in_raw']}/{focus_additions['total']}` are present in raw OCR. This is strong evidence that the main failure is **semantic selection and row association**, not unconstrained hallucination of text or numbers.",
            "",
            f"Package type and quantity are **not rare labels**: they occur in `{package_type['train_documents']}` and `{package_quantity['train_documents']}` of 427 training documents. Their F1 values (`{float(package_type['f1']):.3f}` and `{float(package_quantity['f1']):.3f}`) therefore cannot be explained by simple path absence. The detailed rows show whether the model selected a different OCR-grounded aggregate/nested level, shifted an array row, emitted a near variant, or omitted the field.",
            "",
            f"Lexical novelty also differs by field. Package type has `{package_type['train_unique_values']}` exact variants in training; seen validation types recall at `{float(package_type['seen_value_recall']):.3f}` versus `{float(package_type['unseen_value_recall']):.3f}` for unseen source strings. Quantity does not show the same memorization pattern (`{float(package_quantity['seen_value_recall']):.3f}` seen vs `{float(package_quantity['unseen_value_recall']):.3f}` unseen), confirming that number identity is not the key problem. Goods description appears in `{description['train_documents']}` training documents but has `{description['train_unique_values']}` distinct values; its `{float(description['f1']):.3f}` F1 is driven by structural and partial-copy errors. Additional information is both semantically discretionary and less consistently supported (`{additional['train_documents']}` train documents), matching its very low `{float(additional['f1']):.3f}` F1.",
            "",
            "The granular answer to “how far off” is in `tables/package_type_errors.csv`, `package_quantity_errors.csv`, and `goods_description_errors.csv`. Those retain the exact reference/prediction, best same-field match, numeric deltas, string/token similarity, OCR grounding class, and raw-OCR context. Diagnostic fuzzy similarities never replace strict exact scoring.",
            "",
            "## Root-cause decomposition",
            "",
            "### 1. Valid syntax is necessary, but not sufficient",
            "",
            f"Six outputs are invalid JSON and twelve are schema-invalid. Yet the deliberately impossible oracle that replaces every invalid-JSON document with its **perfect** label reaches only `{float(validity_oracles['perfectly_replace_invalid_json_documents']['f1']):.3f}` F1; replacing every schema-invalid document perfectly reaches `{float(validity_oracles['perfectly_replace_all_schema_invalid_documents']['f1']):.3f}`. Grammar/schema-constrained decoding should therefore be implemented for a reliable system, but it cannot close the semantic gap by itself.",
            "",
            "### 2. The runtime prompt communicates shape, not labeling semantics",
            "",
            "The exact prompt supplies a compact JSON Schema whose builder intentionally removes every `description` and `title`. It tells the model that `packages[].type` is a string, but not whether an aggregate `PKGS` total, a nested `CARTONS` level, or both belong in the sparse label. It does not state the frozen rules for marks, additional information, package hierarchy, container allocations, or role boundaries. Annotators followed a detailed semantic reference; the model receives those rules only indirectly through 427 demonstrations. This mismatch is directly aligned with the observed weak fields.",
            "",
            "A better branch should keep machine-valid shape constraints **and** add a concise, versioned semantic schema instructor covering the ambiguous high-error rules. Train and inference must use the same prompt. The full labeling manual should not simply be pasted into every example; create a compact task contract and measure its token cost and ablation effect.",
            "",
            "### 3. Goods are one coupled table reconstruction task",
            "",
            f"The largest non-additive error budget is the `{section_oracles[0]['scope']}` section: making that section perfect would add `{float(section_oracles[0]['absolute_f1_gain']):.3f}` global F1. Package rows, descriptions, weights, container allocations, marks, and additional information share the same inferred goods-item indices. One missed or merged item shifts many downstream paths. The earlier index-insensitive score gain confirms some alignment cost, but the detailed counterfactual shows this is not only index formatting; semantic row coverage remains missing.",
            "",
            f"The train set contains `{structures['multiple_goods_items']['train_documents']}` multi-goods documents, `{structures['nested_package_levels']['train_documents']}` documents with multiple package levels in one goods item, and `{structures['multiple_allocations_in_one_goods_item']['train_documents']}` with multiple allocations in one goods item. The overlapping `complex_goods_structure` cohort has `{structures['complex_goods_structure']['train_documents']}` train / `{structures['complex_goods_structure']['validation_documents']}` validation documents and validation F1 `{float(structures['complex_goods_structure']['f1']):.3f}`, versus `{float(structures['simple_goods_structure']['f1']):.3f}` for the simple cohort. These are the structures to oversample, not merely the fields to count.",
            "",
            "### 4. More data is justified, but random bulk is not enough",
            "",
            f"Across supported fields, Pearson/Spearman correlation between log training-document support and F1 is `{_pearson(log_support, f1_values):.3f}` / `{_spearman(log_support, f1_values):.3f}`. Support helps, as the pilots-to-combined trend suggests, but high-support cargo fields still fail because examples must cover package hierarchy and table shapes, not just more ordinary one-item bills.",
            "",
            f"Of the `{summary['dataset']['fields_observed_in_training']}` observed normalized leaf paths, `{summary['dataset']['fields_below_document_support']['25']}` occur in fewer than 25 training documents, `{summary['dataset']['fields_below_document_support']['50']}` in fewer than 50, and `{summary['dataset']['fields_below_document_support']['100']}` in fewer than 100. `field_coverage_targets.csv` makes those deficits explicit. Rare optional fields cannot be certified at 0.90 without either deliberate acquisition, a separate task, or a declared non-critical/abstaining policy.",
            "",
            f"The current 60-document validation split is also template-friendly: median nearest-training normalized five-token Jaccard is `{statistics.median(nearest_values):.3f}` (90th percentile `{nearest_values[math.floor(0.9 * (len(nearest_values) - 1))]:.3f}`). This is not leakage—the prior audit found no exact raw hash overlap—but it means random same-corpus validation is weaker than a carrier/template-disjoint production test. Performance under actual template shift is still unknown and should be expected to be no better than this result until measured.",
            "",
            "### 5. Five labels were corrected after their reviewed sidecars were published",
            "",
            f"`{summary['dataset']['path_compatible_postpublication_target_revisions']}` configured documents (`{summary['dataset']['postpublication_revised_leaves']}` leaves) no longer byte-match their reviewed label JSON. The paths are unchanged and every revised value is grounded in the current pinned raw OCR; the changes restore Latin diacritics or split a contact value. This audit joins them only after proving path compatibility and current-raw grounding, and records every case in `label_sidecar_provenance.csv`. The training data are usable, but the annotation publication should be regenerated so target and evidence are once again one immutable object.",
            "",
            "### 6. Position effects are measurable but not the dominant explanation",
            "",
            f"Reference-leaf recall from source-evidence Q1→Q4 is `{float(source_positions['Q1']['recall']):.3f}`, `{float(source_positions['Q2']['recall']):.3f}`, `{float(source_positions['Q3']['recall']):.3f}`, `{float(source_positions['Q4']['recall']):.3f}`. Target-sequence Q1→Q4 recall is `{float(target_positions['Q1']['recall']):.3f}`, `{float(target_positions['Q2']['recall']):.3f}`, `{float(target_positions['Q3']['recall']):.3f}`, `{float(target_positions['Q4']['recall']):.3f}`. These diagnostics distinguish encoder retrieval position from decoder-tail degradation; neither should be inferred from document length alone.",
            "",
            "## Recommended improvement program",
            "",
            "Run these as registered, controlled branches against one frozen evaluation contract. Do not tune by repeatedly inspecting the final external test.",
            "",
            "1. **Repair the evaluation/runtime contract first.** Fix prediction identity publication; add JSON-schema/grammar-constrained generation; retain strict field F1, schema validity, and exact-document match. Re-score outputs in original document order. Constraints are a reliability floor, not the semantic solution.",
            "2. **Create a frozen test hierarchy.** Keep a development validation set for model selection, plus at least one carrier/template-disjoint test and a complexity-stratified challenge set. Report micro/macro F1, per-critical-field P/R/F1, JSON/schema validity, and bootstrap CIs. A 0.90 point estimate without a lower confidence bound is not a reliability claim.",
            "3. **Run a prompt-contract ablation.** Compare the current type-only JSON Schema with a concise semantic schema instructor and with section-specific prompts. Include explicit aggregate-vs-nested package, goods-item grouping, marks/additional-info, allocation, and role rules. Keep the output target unchanged in the first ablation so the causal variable is the prompt.",
            "4. **Add targeted real labels before broad synthesis.** Select the next 500-1,000 documents by underrepresented field *and structure*: multiple goods items, multiple package levels, per-container rows, marks, additional cargo text, 3-4 pages, and uncommon carriers/templates. The first milestone should be +500 adjudicated diverse documents, then plot learning curves at +125/+250/+500. Continue to +1,000 only if the held-out curve is still improving materially.",
            "5. **Treat goods extraction as its own experiment.** Compare one-shot full-document JSON with two or three section-conditioned decodes (document/route, parties, goods/containers) and deterministic merge. A narrower goods decode reduces target length and field competition while preserving the two-model architecture. Batch the section calls to control throughput.",
            "6. **Run a capacity/adaptation matrix only after 1-4.** On the same frozen data/split, compare 270M-270M LoRA r32, a higher-rank or selective-full-tune branch, and T5Gemma 2 1B-1B LoRA. The current run does not isolate LoRA as the bottleneck; changing Adam betas or adding epochs is lower-value than these causal tests.",
            "7. **Use synthetic data as a coverage tool, not a substitute for truth.** Generate from target structures so rare combinations are deliberate, then require OCR-grounded rendering, schema validation, deterministic consistency checks, deduplication, and a real-only external test. Never let synthetic examples determine the acceptance result.",
            "8. **Add field-level confidence and abstention.** Capture token/field log-probabilities or agreement across independently trained seeds, calibrate on development data, and route low-confidence/structurally inconsistent documents to review. Report accuracy/F1 versus retained coverage. A highly reliable deployed system needs a fail-closed path even after aggregate F1 exceeds 0.90.",
            "",
            "### Acceptance gates",
            "",
            "- JSON valid = 1.000 and schema valid = 1.000 on every acceptance set (enforced decoding plus post-parse validation).",
            "- Overall exact field/value micro-F1 **and** exact path-union accuracy ≥ 0.90, with document-bootstrap 95% CI lower bounds ≥ 0.90 on the carrier/template-disjoint test.",
            "- Package type, package quantity, goods description, container number/allocation, B/L number, parties, route, weights, and dates each meet predeclared P/R/F1 floors; no aggregate score may hide a critical weak field.",
            "- Report results by page count, goods-item count, package-level count, carrier/template novelty, and OCR quality cohort.",
            "- Freeze and audit the external test; tune only on train/development sets. Re-run acceptance only for registered candidate configurations.",
            "",
            "## Research cross-reference",
            "",
        ]
    )
    for title, url, relevance in RESEARCH_SOURCES:
        lines.append(f"- [{title}]({url}): {relevance}")
    lines.extend(
        [
            "",
            "## Artifact map",
            "",
            "- `summary.json`: audit headline counts and target-gap arithmetic.",
            "- `tables/field_support_performance.csv`: train support, value novelty, evidence transformations, and final metrics for every normalized field.",
            "- `tables/reference_occurrences.jsonl`: every validation reference leaf with exact match class, similarity, OCR evidence, and positions.",
            "- `tables/prediction_additions.jsonl`: every unmatched predicted leaf with raw-OCR grounding.",
            "- `tables/package_type_errors.csv`, `package_quantity_errors.csv`, `goods_description_errors.csv`: requested error-distance audits.",
            "- `tables/counterfactual_error_budget.csv`: deliberately optimistic, non-additive oracle ceilings.",
            "- `tables/template_nearest_train.csv` and `carrier_performance.csv`: in-domain/template coverage diagnostics.",
            "- `tables/structural_cohort_performance.csv`: sample support and validation performance for complex goods/package/allocation shapes.",
            "- `tables/field_coverage_targets.csv`: additional-document deficits to descriptive 25/50/100/200 support thresholds.",
            "- `examples/focus_error_gallery.md`: human-readable source/reference/prediction examples.",
            "- `analysis_manifest.json`: hash/size inventory of this publication.",
            "",
            "## Plot gallery",
            "",
        ]
    )
    for path in plot_paths:
        title = Path(path).stem[3:].replace("_", " ").title()
        lines.extend([f"### {title}", "", f"![{title}]({path})", ""])
    (output_dir / "REPORT.md").write_text("\n".join(lines).rstrip() + "\n", encoding="utf-8")


def _error_gallery(
    output_dir: Path,
    occurrence_rows: Sequence[Mapping[str, Any]],
) -> None:
    focus = [
        row
        for row in occurrence_rows
        if row["field_path"] in {PACKAGE_TYPE_FIELD, PACKAGE_QUANTITY_FIELD, DESCRIPTION_FIELD}
        and not row["strict_correct"]
    ]
    focus.sort(
        key=lambda row: (
            row["field_path"],
            row["match_class"] == "field_absent",
            -float(row["sequence_similarity"]),
            row["document_id"],
        )
    )
    selected: list[Mapping[str, Any]] = []
    per_field: Counter[str] = Counter()
    seen_documents: set[tuple[str, str]] = set()
    for row in focus:
        key = (str(row["field_path"]), str(row["document_id"]))
        if key in seen_documents or per_field[str(row["field_path"])] >= 8:
            continue
        selected.append(row)
        seen_documents.add(key)
        per_field[str(row["field_path"])] += 1
    lines = [
        "# Focus-field error gallery",
        "",
        "These are representative exact-match failures selected deterministically from saved predictions. Context is raw OCR; similarity is diagnostic only.",
        "",
    ]
    for row in selected:
        lines.extend(
            [
                f"## `{row['document_id']}` — `{row['field_path']}`",
                "",
                f"- Match class: `{row['match_class']}`",
                f"- Reference: `{json.dumps(row['reference_value'], ensure_ascii=False)}`",
                f"- Best prediction: `{json.dumps(row['predicted_value'], ensure_ascii=False)}`",
                f"- Sequence similarity / token F1: `{float(row['sequence_similarity']):.3f}` / `{float(row['token_f1']):.3f}`",
                f"- Prediction OCR grounding: `{row['prediction_raw_grounding']}`",
                "",
                "Reference evidence/context:",
                "",
                f"> {str(row['reference_context']).replace(chr(10), ' ')}",
                "",
                "Prediction context:",
                "",
                f"> {str(row['prediction_context']).replace(chr(10), ' ') or '<not located>'}",
                "",
            ]
        )
    path = output_dir / "examples" / "focus_error_gallery.md"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines).rstrip() + "\n", encoding="utf-8")


def _manifest(output_dir: Path) -> None:
    files = []
    for path in sorted(output_dir.rglob("*")):
        if not path.is_file() or path.name == "analysis_manifest.json":
            continue
        files.append(
            {
                "path": str(path.relative_to(output_dir)),
                "bytes": path.stat().st_size,
                "sha256": _sha256(path),
            }
        )
    _write_json(
        output_dir / "analysis_manifest.json",
        {
            "schema_version": AUDIT_SCHEMA_VERSION,
            "files": files,
            "file_count": len(files),
            "total_bytes": sum(cast(int, item["bytes"]) for item in files),
        },
    )


def run_audit(
    project_root: Path, run_dir: Path, prior_analysis_dir: Path, output_dir: Path
) -> None:
    if output_dir.exists():
        raise FileExistsError(f"refusing to overwrite reliability audit: {output_dir}")
    config = cast(
        dict[str, Any], yaml.safe_load((run_dir / "config.yaml").read_text(encoding="utf-8"))
    )
    validated_config = TrainingConfig.model_validate(config)
    task = load_training_task(project_root, validated_config)
    records, _ = _load_dataset_records(project_root, config)
    prediction_path = prior_analysis_dir / "recovered-predictions" / "validation.jsonl"
    predictions = _load_predictions(prediction_path, records, task)
    sidecars, provenance_rows = _load_label_sidecars(project_root, records, task)
    if len(sidecars) != len(records):
        missing = sorted(set(records) - set(sidecars))
        raise ValueError(
            f"reviewed label sidecars are missing for {len(missing)} configured records: {missing[:10]}"
        )

    field_rows, occurrence_rows, addition_rows = _field_stats(records, predictions, sidecars)
    template_rows = _template_rows(records, predictions)
    carrier_rows = _carrier_rows(records, predictions)
    counterfactual_rows = _counterfactual_rows(predictions, field_rows)
    prior_runs = _prior_run_rows(project_root)
    position_rows = _position_rows(
        occurrence_rows, "reference_source_position", "source_evidence"
    ) + _position_rows(occurrence_rows, "target_sequence_position", "target_sequence")
    structure_rows = _structure_rows(records, predictions)
    package_type_errors = _category_rows(occurrence_rows, PACKAGE_TYPE_FIELD)
    quantity_errors = [
        dict(row, numeric_error_class=_numeric_error_class(row))
        for row in occurrence_rows
        if row["field_path"] == PACKAGE_QUANTITY_FIELD and not row["strict_correct"]
    ]
    description_errors = _category_rows(occurrence_rows, DESCRIPTION_FIELD)
    focus_additions = [row for row in addition_rows if row["field_path"] in FOCUS_FIELDS]

    def error_distance_summary(
        rows: Sequence[Mapping[str, Any]], category_key: str
    ) -> dict[str, Any]:
        non_absent = [row for row in rows if row["predicted_value"] is not None]
        return {
            "reference_errors": len(rows),
            "categories": dict(sorted(Counter(str(row[category_key]) for row in rows).items())),
            "non_absent_best_predictions": len(non_absent),
            "non_absent_grounded_in_raw": sum(
                row["prediction_raw_grounding"] != "not_grounded" for row in non_absent
            ),
        }

    baseline = _counterfactual_fix(predictions)
    compared_paths = sum(
        len(
            {path for path, _ in row.assessment.predicted_field_values}
            | {path for path, _ in row.assessment.reference_field_values}
        )
        for row in predictions.values()
    )
    field_value_accuracy = _safe_divide(baseline.true_positive, compared_paths)
    json_valid = statistics.fmean(row.assessment.json_valid for row in predictions.values())
    schema_valid = statistics.fmean(row.assessment.schema_valid for row in predictions.values())
    numerator = TARGET_F1 * (baseline.predicted + baseline.reference) - 2 * baseline.true_positive
    perfect_new_true_positives_needed = max(0, math.ceil(numerator / (2 - TARGET_F1)))
    recall_needed = TARGET_F1 * baseline.precision / (2 * baseline.precision - TARGET_F1)
    precision_needed_denominator = 2 * baseline.recall - TARGET_F1
    precision_needed = (
        TARGET_F1 * baseline.recall / precision_needed_denominator
        if precision_needed_denominator > 0
        else math.inf
    )
    true_positives_needed_for_accuracy = max(
        0, math.ceil(TARGET_F1 * compared_paths - baseline.true_positive)
    )
    true_positives_needed_for_accuracy_after_removing_extra_paths = max(
        0, math.ceil(TARGET_F1 * baseline.reference - baseline.true_positive)
    )
    train_count = sum(record.split == "train" for record in records.values())
    validation_count = sum(record.split == "validation" for record in records.values())
    under_support = {
        str(threshold): sum(0 < int(row["train_documents"]) < threshold for row in field_rows)
        for threshold in (10, 25, 50, 100)
    }
    summary = {
        "schema_version": AUDIT_SCHEMA_VERSION,
        "generated_at": datetime.now(UTC).isoformat(),
        "run_id": run_dir.name,
        "source_run_immutable": True,
        "model_weights_loaded": False,
        "gpu_used": False,
        "dataset": {
            "train_documents": train_count,
            "validation_documents": validation_count,
            "reviewed_sidecars_matched": len(sidecars),
            "path_compatible_postpublication_target_revisions": sum(
                row["selected_match_type"] == "path_compatible_raw_grounded_target_revision"
                for row in provenance_rows
            ),
            "postpublication_revised_leaves": sum(
                int(row["changed_leaf_count"]) for row in provenance_rows
            ),
            "fields_observed_in_training": sum(
                int(row["train_documents"]) > 0 for row in field_rows
            ),
            "fields_below_document_support": under_support,
        },
        "baseline": {
            "true_positive": baseline.true_positive,
            "compared_paths": compared_paths,
            "accuracy": field_value_accuracy,
            "predicted": baseline.predicted,
            "reference": baseline.reference,
            "false_positive": baseline.predicted - baseline.true_positive,
            "false_negative": baseline.reference - baseline.true_positive,
            "precision": baseline.precision,
            "recall": baseline.recall,
            "f1": baseline.f1,
            "json_valid": json_valid,
            "schema_valid": schema_valid,
        },
        "target_gap": {
            "target_f1": TARGET_F1,
            "false_negatives": baseline.reference - baseline.true_positive,
            "false_positives": baseline.predicted - baseline.true_positive,
            "perfect_new_true_positives_needed": perfect_new_true_positives_needed,
            "recall_needed_at_current_precision": recall_needed,
            "precision_needed_at_current_recall": precision_needed
            if math.isfinite(precision_needed)
            else None,
            "true_positives_needed_for_accuracy_with_current_path_union": true_positives_needed_for_accuracy,
            "true_positives_needed_for_accuracy_after_removing_all_extra_paths": true_positives_needed_for_accuracy_after_removing_extra_paths,
        },
        "prompt_contract": {
            "runtime_prompt_bytes": (run_dir / "prompt.txt").stat().st_size,
            "schema_annotation_count": _schema_annotation_count(
                json.loads(task.prompt_schema_json())
            ),
            "semantic_labeling_reference_included": "aggregate package"
            in (run_dir / "prompt.txt").read_text(encoding="utf-8").casefold(),
        },
        "focus_error_distance": {
            "package_type": error_distance_summary(package_type_errors, "distance_class"),
            "package_quantity": error_distance_summary(quantity_errors, "numeric_error_class"),
            "goods_description": error_distance_summary(description_errors, "distance_class"),
            "unmatched_focus_predictions": {
                "total": len(focus_additions),
                "grounded_in_raw": sum(
                    row["prediction_raw_grounding"] != "not_grounded" for row in focus_additions
                ),
            },
        },
    }

    output_dir.mkdir(parents=True)
    tables = output_dir / "tables"
    _write_json(output_dir / "summary.json", summary)
    _write_csv(tables / "label_sidecar_provenance.csv", provenance_rows, tuple(provenance_rows[0]))
    _write_csv(tables / "field_support_performance.csv", field_rows, tuple(field_rows[0]))
    _write_jsonl(tables / "reference_occurrences.jsonl", occurrence_rows)
    _write_jsonl(tables / "prediction_additions.jsonl", addition_rows)
    _write_csv(
        tables / "counterfactual_error_budget.csv",
        counterfactual_rows,
        tuple(counterfactual_rows[0]),
    )
    _write_csv(tables / "template_nearest_train.csv", template_rows, tuple(template_rows[0]))
    _write_csv(tables / "carrier_performance.csv", carrier_rows, tuple(carrier_rows[0]))
    _write_csv(tables / "prior_run_trend.csv", prior_runs, tuple(prior_runs[0]))
    _write_csv(tables / "position_recall.csv", position_rows, tuple(position_rows[0]))
    _write_csv(
        tables / "structural_cohort_performance.csv", structure_rows, tuple(structure_rows[0])
    )

    coverage_target_rows = [
        {
            "field_path": row["field_path"],
            "current_train_documents": row["train_documents"],
            "additional_to_25": max(0, 25 - int(row["train_documents"])),
            "additional_to_50": max(0, 50 - int(row["train_documents"])),
            "additional_to_100": max(0, 100 - int(row["train_documents"])),
            "additional_to_200": max(0, 200 - int(row["train_documents"])),
            "validation_reference_values": row["validation_reference_values"],
            "validation_f1": row["f1"],
        }
        for row in field_rows
    ]
    _write_csv(
        tables / "field_coverage_targets.csv",
        coverage_target_rows,
        tuple(coverage_target_rows[0]),
    )

    if package_type_errors:
        _write_csv(
            tables / "package_type_errors.csv", package_type_errors, tuple(package_type_errors[0])
        )
    if quantity_errors:
        _write_csv(
            tables / "package_quantity_errors.csv", quantity_errors, tuple(quantity_errors[0])
        )
    if description_errors:
        _write_csv(
            tables / "goods_description_errors.csv",
            description_errors,
            tuple(description_errors[0]),
        )

    package_distribution: Counter[str] = Counter()
    for record in records.values():
        if record.split != "train":
            continue
        for path, _, value in _flatten_ordered(record.target["documentPatch"]):
            if _normalize_path(path) == PACKAGE_TYPE_FIELD:
                package_distribution[str(value)] += 1
    package_distribution_rows = [
        {"package_type": value, "train_occurrences": count}
        for value, count in package_distribution.most_common()
    ]
    _write_csv(
        tables / "package_type_training_distribution.csv",
        package_distribution_rows,
        tuple(package_distribution_rows[0]),
    )

    plot_paths = _render_plots(
        output_dir,
        field_rows,
        occurrence_rows,
        addition_rows,
        counterfactual_rows,
        template_rows,
        prior_runs,
        position_rows,
        structure_rows,
    )
    _error_gallery(output_dir, occurrence_rows)
    _report(
        output_dir,
        run_dir.name,
        summary,
        field_rows,
        counterfactual_rows,
        template_rows,
        position_rows,
        structure_rows,
        plot_paths,
    )
    _manifest(output_dir)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, default=Path.cwd())
    parser.add_argument(
        "--run-dir",
        type=Path,
        default=Path("artifacts/kie-training/t5gemma2-270m-lora-mpci-bl-combined487-v2"),
    )
    parser.add_argument(
        "--prior-analysis-dir",
        type=Path,
        default=Path("artifacts/kie-training/analysis/t5gemma2-270m-lora-mpci-bl-combined487-v2"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(
            "artifacts/kie-training/analysis/t5gemma2-270m-lora-mpci-bl-combined487-v2-reliability-audit-v1"
        ),
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    project_root = args.project_root.resolve()
    run_dir = (
        (project_root / args.run_dir).resolve()
        if not args.run_dir.is_absolute()
        else args.run_dir.resolve()
    )
    prior_analysis_dir = (
        (project_root / args.prior_analysis_dir).resolve()
        if not args.prior_analysis_dir.is_absolute()
        else args.prior_analysis_dir.resolve()
    )
    output_dir = (
        (project_root / args.output_dir).resolve()
        if not args.output_dir.is_absolute()
        else args.output_dir.resolve()
    )
    run_audit(project_root, run_dir, prior_analysis_dir, output_dir)
    print(
        json.dumps(
            {
                "status": "complete",
                "run_id": run_dir.name,
                "output_dir": str(output_dir),
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
