#!/usr/bin/env python3
"""Classify exact KIE validation misses by observable failure mechanism.

The analysis is deliberately conservative: a value that does not match the raw
OCR under a normalized-substring probe is an *unsupported candidate*, not proven
hallucination.  The published tables retain the source leaf error and the basis
for every classification so the result remains auditable.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
from collections import Counter, defaultdict
from collections.abc import Mapping, Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd  # type: ignore[import-untyped]
import seaborn as sns  # type: ignore[import-untyped]
from matplotlib.patches import Rectangle

from document_ocr.atomic import atomic_write_json
from document_ocr.hashing import sha256_file

INDEX = re.compile(r"\[[0-9]+\]")
NON_ALNUM = re.compile(r"[^a-z0-9]+")

DEFAULT_AUDIT = Path(
    "artifacts/kie-training/analysis/"
    "t5gemma2-270m-lora-mpci-bl-combined1157-task-facing-v2-deep-audit-v3"
)
DEFAULT_PREDICTIONS = Path(
    "artifacts/kie-training/analysis/"
    "t5gemma2-270m-lora-mpci-bl-combined1157-task-facing-v2-"
    "checkpoint900-eval-v1/predictions/validation.jsonl"
)
DEFAULT_OUTPUT = Path(
    "artifacts/kie-training/analysis/"
    "t5gemma2-270m-lora-mpci-bl-combined1157-task-facing-v2-"
    "error-mechanisms-v1"
)


def _jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if not line:
            raise ValueError(f"blank JSONL line at {path}:{line_number}")
        value = json.loads(line)
        if not isinstance(value, dict):
            raise ValueError(f"JSONL row is not an object at {path}:{line_number}")
        rows.append(value)
    if not rows:
        raise ValueError(f"empty JSONL input: {path}")
    return rows


def _csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as stream:
        rows = list(csv.DictReader(stream))
    if not rows:
        raise ValueError(f"empty CSV input: {path}")
    return rows


def _write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    if not rows:
        raise ValueError(f"refusing to publish an empty CSV: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = sorted({key for row in rows for key in row})
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def _write_jsonl(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    if not rows:
        raise ValueError(f"refusing to publish an empty JSONL: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as stream:
        for row in rows:
            stream.write(json.dumps(dict(row), ensure_ascii=False, sort_keys=True) + "\n")


def _decode(value: str) -> Any:
    return json.loads(value) if value else None


def _normalize_path(path: str) -> str:
    return INDEX.sub("[]", path)


def _normalized_text(value: str) -> str:
    return NON_ALNUM.sub("", value.casefold())


def _is_relation_or_structure_path(path: str) -> bool:
    return path.endswith(
        (
            ".groupId",
            ".packageId",
            ".packageIds[]",
            ".allocations[].packageId",
            ".coverage",
        )
    )


def _is_categorical_path(path: str) -> bool:
    return path.endswith(
        (
            ".typeCategory",
            ".hazardCategory",
            ".paymentArrangement",
            ".negotiability",
            ".unit",
        )
    )


def _is_date_path(path: str) -> bool:
    return path.endswith("Date")


def _is_scale_error(reference: Any, predicted: Any) -> bool:
    if (
        isinstance(reference, bool)
        or isinstance(predicted, bool)
        or not isinstance(reference, (int, float))
        or not isinstance(predicted, (int, float))
        or float(reference) == 0.0
        or float(predicted) == 0.0
    ):
        return False
    ratio = abs(float(predicted) / float(reference))
    return any(
        math.isclose(ratio, factor, rel_tol=1e-9) for factor in (0.001, 0.01, 0.1, 10, 100, 1000)
    )


def _cross_field_pairs(errors: Sequence[Mapping[str, Any]]) -> dict[int, tuple[str, ...]]:
    """Find unambiguous one-addition/one-omission value moves within a document."""

    grouped: dict[tuple[str, str], dict[str, list[int]]] = defaultdict(
        lambda: {"addition": [], "omission": []}
    )
    for index, row in enumerate(errors):
        error_type = str(row["error_type"])
        if error_type not in {"addition", "omission"}:
            continue
        value = str(
            row.get("predicted_value") if error_type == "addition" else row.get("reference_value")
        )
        if value:
            grouped[(str(row["document_id"]), value)][error_type].append(index)

    matches: dict[int, tuple[str, ...]] = {}
    for group in grouped.values():
        if len(group["addition"]) != 1 or len(group["omission"]) != 1:
            continue
        addition_index = group["addition"][0]
        omission_index = group["omission"][0]
        predicted_path = str(errors[addition_index]["predicted_path"])
        reference_path = str(errors[omission_index]["reference_path"])
        if _normalize_path(predicted_path) == _normalize_path(reference_path):
            continue
        matches[addition_index] = (reference_path,)
        matches[omission_index] = (predicted_path,)
    return matches


def classify_leaf_error(
    row: Mapping[str, Any],
    *,
    cross_field_paths: tuple[str, ...] = (),
) -> tuple[str, str, tuple[str, ...]]:
    """Return mechanism, conservative basis, and any cross-field match paths."""

    error_type = str(row["error_type"])
    predicted_path = str(row.get("predicted_path") or "")
    reference_path = str(row.get("reference_path") or "")
    predicted_value = str(row.get("predicted_value") or "")
    reference_value = str(row.get("reference_value") or "")
    path = reference_path or predicted_path

    if cross_field_paths:
        return (
            "wrong_field_assignment",
            ("one added and one omitted leaf share this exact scalar under different field paths"),
            cross_field_paths,
        )

    if error_type == "omission":
        return "omission", "reference leaf is absent from the generated object", ()
    if error_type == "right_value_wrong_index":
        return (
            "correct_value_wrong_list_position",
            "exact scalar appears under the same normalized field path at another list index",
            (),
        )
    if error_type == "list_alignment_or_value_mismatch":
        return (
            "list_alignment_mismatch",
            "remaining list elements could not be aligned by exact value and position",
            (),
        )
    if _is_relation_or_structure_path(_normalize_path(path)):
        return (
            "relation_or_structure_error",
            "field is deterministic cargo relationship or synthetic identity bookkeeping",
            (),
        )
    if _is_categorical_path(_normalize_path(path)):
        return (
            "categorical_semantic_error",
            "field is a normalized categorical decision rather than a copied OCR span",
            (),
        )
    if _is_date_path(_normalize_path(path)):
        return (
            "date_normalization_or_selection_error",
            "ISO date output is normalized and cannot be judged by literal OCR substring alone",
            (),
        )

    predicted = _decode(predicted_value)
    reference = _decode(reference_value)
    if isinstance(predicted, (int, float)) and not isinstance(predicted, bool):
        if error_type == "substitution" and _is_scale_error(reference, predicted):
            return (
                "numeric_scale_error",
                "prediction differs from the target by an exact decimal power-of-ten factor",
                (),
            )
        if row.get("predicted_ocr_grounded") is True:
            return (
                "ocr_grounded_wrong_numeric_selection",
                "predicted number occurs in OCR but is not the labeled number for this field",
                (),
            )
        return (
            "unsupported_or_wrong_numeric_value",
            "number is not an exact normalized OCR substring for the labeled field",
            (),
        )

    if error_type == "substitution" and isinstance(predicted, str) and isinstance(reference, str):
        predicted_normalized = _normalized_text(predicted)
        reference_normalized = _normalized_text(reference)
        if predicted_normalized and predicted_normalized in reference_normalized:
            return (
                "incomplete_or_shortened_value",
                "normalized prediction is a proper substring of the target value",
                (),
            )
        if reference_normalized and reference_normalized in predicted_normalized:
            return (
                "contaminated_or_overcomplete_value",
                "normalized target is a proper substring of the prediction",
                (),
            )
        similarity = row.get("value_similarity")
        if isinstance(similarity, (int, float)) and similarity >= 0.85:
            return (
                "near_copy_corruption",
                "string differs from target but character similarity is at least 0.85",
                (),
            )
        if row.get("predicted_ocr_grounded") is True:
            return (
                "ocr_grounded_wrong_text_selection",
                "predicted text occurs in OCR but is not the labeled text for this field",
                (),
            )
        if row.get("predicted_ocr_grounded") is False:
            return (
                "unsupported_text_candidate",
                "predicted text is neither a normalized OCR substring nor a close/boundary copy",
                (),
            )
        return (
            "semantic_text_substitution",
            "text differs without a literal grounding or boundary diagnosis",
            (),
        )

    if error_type == "addition":
        if row.get("predicted_ocr_grounded") is True:
            return (
                "ocr_grounded_over_extraction",
                "extra value occurs in OCR but is not present in the target field",
                (),
            )
        if row.get("predicted_ocr_grounded") is False:
            return (
                "unsupported_text_candidate",
                "extra text does not match a normalized OCR substring",
                (),
            )
        return (
            "semantic_or_structural_addition",
            "extra value is not governed by literal OCR grounding",
            (),
        )
    raise ValueError(f"unsupported leaf error type: {error_type}")


def _output_failure_rows(
    schema_issues: Sequence[Mapping[str, str]], predictions: Sequence[Mapping[str, Any]]
) -> list[dict[str, Any]]:
    failures: Counter[str] = Counter()
    failures["invalid_json"] = sum(not bool(row.get("json_valid")) for row in predictions)
    failures["wrong_or_extra_key"] = sum(
        row.get("issue_type") == "extra_forbidden" for row in schema_issues
    )
    failures["missing_required_key"] = sum(
        row.get("issue_type") == "missing" for row in schema_issues
    )
    failures["invalid_value_or_relationship"] = sum(
        row.get("issue_type") not in {"extra_forbidden", "invalid_json", "missing"}
        for row in schema_issues
    )
    return [{"failure_type": key, "occurrences": failures[key]} for key in failures]


def _plots(
    *,
    rows: Sequence[Mapping[str, Any]],
    output_failures: Sequence[Mapping[str, Any]],
    output: Path,
) -> None:
    plot_dir = output / "plots"
    plot_dir.mkdir(parents=True, exist_ok=True)
    sns.set_theme(style="whitegrid", context="notebook")
    frame = pd.DataFrame(rows)

    mechanism = frame["mechanism"].value_counts().sort_values()
    fig, axis = plt.subplots(figsize=(11, 8))
    sns.barplot(x=mechanism.values, y=mechanism.index, ax=axis, color="#4472C4")
    axis.set(
        title="Exact field/value errors by observable mechanism", xlabel="Leaf errors", ylabel=""
    )
    for patch, value in zip(axis.patches, mechanism.values, strict=True):
        if not isinstance(patch, Rectangle):
            raise TypeError("seaborn barplot produced a non-rectangular patch")
        axis.text(
            value + max(mechanism.max() * 0.008, 0.5),
            patch.get_y() + patch.get_height() / 2,
            str(value),
            va="center",
        )
    fig.tight_layout()
    fig.savefig(plot_dir / "01_error_mechanisms.png", dpi=180)
    plt.close(fig)

    pivot = pd.crosstab(frame["section"], frame["mechanism"])
    ordered_sections = pivot.sum(axis=1).sort_values(ascending=False).index
    ordered_mechanisms = pivot.sum(axis=0).sort_values(ascending=False).index
    fig, axis = plt.subplots(figsize=(16, 9))
    sns.heatmap(
        pivot.loc[ordered_sections, ordered_mechanisms], cmap="Blues", annot=True, fmt="g", ax=axis
    )
    axis.set(title="Mechanism by output section", xlabel="Mechanism", ylabel="Section")
    axis.tick_params(axis="x", rotation=45)
    fig.tight_layout()
    fig.savefig(plot_dir / "02_mechanism_by_section.png", dpi=180)
    plt.close(fig)

    substitution = frame[
        (frame["error_type"] == "substitution") & frame["value_similarity"].notna()
    ].copy()
    fig, axis = plt.subplots(figsize=(11, 6))
    sns.histplot(
        data=substitution, x="value_similarity", bins=20, hue="mechanism", multiple="stack", ax=axis
    )
    axis.set(
        title="Character/numeric similarity of substitution errors",
        xlabel="Target-prediction similarity",
        ylabel="Errors",
    )
    fig.tight_layout()
    fig.savefig(plot_dir / "03_substitution_similarity.png", dpi=180)
    plt.close(fig)

    eligible = frame[frame["predicted_ocr_grounded"].notna()].copy()
    eligible["OCR substring"] = eligible["predicted_ocr_grounded"].map(
        {True: "matched", False: "not matched"}
    )
    grounding = pd.crosstab(eligible["error_type"], eligible["OCR substring"])
    grounding.plot(kind="bar", stacked=True, color=["#C55A11", "#70AD47"], figsize=(10, 6))
    plt.title("Literal normalized-OCR support for wrong predicted leaves")
    plt.xlabel("Leaf error type")
    plt.ylabel("Errors")
    plt.xticks(rotation=20, ha="right")
    plt.tight_layout()
    plt.savefig(plot_dir / "04_ocr_grounding_by_error_type.png", dpi=180)
    plt.close()

    top_fields = frame["field_path"].value_counts().head(20).index
    field_pivot = pd.crosstab(
        frame[frame["field_path"].isin(top_fields)]["field_path"],
        frame[frame["field_path"].isin(top_fields)]["mechanism"],
    )
    field_pivot = field_pivot.loc[
        top_fields, field_pivot.sum(axis=0).sort_values(ascending=False).index
    ]
    fig, axis = plt.subplots(figsize=(17, 11))
    sns.heatmap(field_pivot, cmap="mako", annot=True, fmt="g", ax=axis)
    axis.set(
        title="Mechanisms for the 20 largest field error contributors",
        xlabel="Mechanism",
        ylabel="Field",
    )
    axis.tick_params(axis="x", rotation=45)
    fig.tight_layout()
    fig.savefig(plot_dir / "05_top_field_mechanisms.png", dpi=180)
    plt.close(fig)

    failures = pd.DataFrame(output_failures)
    fig, axis = plt.subplots(figsize=(9, 5))
    sns.barplot(data=failures, x="occurrences", y="failure_type", color="#A5A5A5", ax=axis)
    axis.set(
        title="Output-contract failures (separate from leaf semantics)",
        xlabel="Occurrences",
        ylabel="",
    )
    fig.tight_layout()
    fig.savefig(plot_dir / "06_output_contract_failures.png", dpi=180)
    plt.close(fig)

    raw_types = pd.crosstab(frame["section"], frame["error_type"])
    raw_types = raw_types.loc[raw_types.sum(axis=1).sort_values(ascending=False).index]
    raw_types.plot(kind="bar", stacked=True, figsize=(13, 7), colormap="tab20c")
    plt.title("Original exact-match error types by section")
    plt.xlabel("Section")
    plt.ylabel("Leaf errors")
    plt.xticks(rotation=35, ha="right")
    plt.tight_layout()
    plt.savefig(plot_dir / "07_raw_error_types_by_section.png", dpi=180)
    plt.close()

    unsupported = frame[
        frame["mechanism"].isin(
            {"unsupported_text_candidate", "unsupported_or_wrong_numeric_value"}
        )
    ]
    unsupported_fields = unsupported["field_path"].value_counts().head(20).sort_values()
    fig, axis = plt.subplots(figsize=(11, 7))
    sns.barplot(x=unsupported_fields.values, y=unsupported_fields.index, color="#C55A11", ax=axis)
    axis.set(
        title="Conservative unsupported-value candidates by field",
        xlabel="Candidate leaf errors",
        ylabel="",
    )
    fig.tight_layout()
    fig.savefig(plot_dir / "08_unsupported_candidate_fields.png", dpi=180)
    plt.close(fig)


def _report(
    *,
    rows: Sequence[Mapping[str, Any]],
    output_failures: Sequence[Mapping[str, Any]],
    output: Path,
) -> str:
    counts = Counter(str(row["mechanism"]) for row in rows)
    raw_counts = Counter(str(row["error_type"]) for row in rows)
    unsupported = (
        counts["unsupported_text_candidate"] + counts["unsupported_or_wrong_numeric_value"]
    )
    cross_field_leaf_errors = counts["wrong_field_assignment"]
    cross_field_leaf_groups = len(
        {
            (
                str(row["document_id"]),
                str(row["predicted_value"] or row["reference_value"]),
                tuple(row["elsewhere_paths"]),
            )
            for row in rows
            if row["mechanism"] == "wrong_field_assignment"
        }
    )
    if cross_field_leaf_groups % 2:
        raise ValueError("one-to-one cross-field leaf groups must occur in pairs")
    cross_field_pairs = cross_field_leaf_groups // 2
    failures = {str(row["failure_type"]): int(row["occurrences"]) for row in output_failures}
    total = len(rows)
    lines = [
        "# T5Gemma2 270M validation error mechanisms",
        "",
        (
            f"Generated at `{datetime.now(UTC).isoformat()}` from the checkpoint-900 "
            "unconstrained validation predictions."
        ),
        "",
        "## Answer",
        "",
        (
            f"There are **{total:,} exact leaf errors**. The largest raw class is "
            f"missing output: **{raw_counts['omission']:,} omissions "
            f"({raw_counts['omission'] / total:.1%})**. The model is therefore not "
            "primarily failing by inventing arbitrary text or misspelling every value."
        ),
        "",
        (
            f"Only **{failures.get('wrong_or_extra_key', 0)}** schema issue is a "
            "genuinely wrong/extra JSON key, alongside "
            f"**{failures.get('invalid_json', 0)}** malformed JSON documents. Most "
            "failures use valid field names but omit, over-extract, mis-group, "
            "normalize, or assign the wrong value."
        ),
        "",
        (
            "The conservative unsupported-value bucket contains "
            f"**{unsupported:,} leaf errors ({unsupported / total:.1%})**. It is an "
            "upper bound on hallucination, not proof: literal OCR matching misses "
            "punctuation normalization, OCR variants, unit conversion, and label "
            "defects. Every candidate is retained in "
            "`tables/unsupported_candidates.csv` for inspection."
        ),
        "",
        (
            "Exact values placed under another semantic field account for "
            f"**{cross_field_leaf_errors} leaf errors across {cross_field_pairs} observed "
            "value/path groups**. Correct values at only the wrong list position account "
            f"for **{counts['correct_value_wrong_list_position']}** more errors."
        ),
        "",
        "## Mechanism counts",
        "",
        "| mechanism | leaf errors | fraction |",
        "|---|---:|---:|",
    ]
    for mechanism, count in counts.most_common():
        lines.append(f"| `{mechanism}` | {count:,} | {count / total:.1%} |")
    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            "- `omission` means the labeled leaf never appeared in the generated object.",
            (
                "- `incomplete_or_shortened_value` and "
                "`contaminated_or_overcomplete_value` are boundary errors: the model "
                "copied only part of a value or included neighboring text/rows."
            ),
            (
                "- `near_copy_corruption` is the closest match to a misspelling/copy "
                "error (character similarity at least 0.85 after other diagnoses)."
            ),
            (
                "- `ocr_grounded_wrong_*` and `ocr_grounded_over_extraction` are not "
                "hallucinations: the value exists in OCR, but the model selected the "
                "wrong row, total, role, or field scope."
            ),
            (
                "- `relation_or_structure_error` covers synthetic IDs and cargo-link "
                "decisions that cannot be copied literally from OCR."
            ),
            (
                "- `unsupported_*` is deliberately named as a candidate bucket. "
                "Visual/source review would be required before calling any individual "
                "item hallucinated."
            ),
            "",
            "## Artifacts",
            "",
            "- `tables/mechanism_errors.jsonl`: one auditable classification per exact leaf error.",
            "- `tables/mechanism_summary.csv`: mechanism totals and fractions.",
            "- `tables/mechanism_by_field.csv`: field-level mechanism matrix.",
            "- `tables/unsupported_candidates.csv`: conservative hallucination-review queue.",
            (
                "- `tables/output_contract_failures.csv`: wrong keys, malformed JSON, "
                "and other schema failures."
            ),
            "- `plots/`: eight matplotlib/seaborn figures.",
            "",
            "## Methodological boundary",
            "",
            (
                "This is an error-distance audit against one labeled validation set. "
                "Exact-match penalties can expose label inconsistencies as well as model "
                "mistakes. Literal OCR grounding uses normalized alphanumeric substring "
                "matching; it is useful evidence, but neither a full semantic grounding "
                "test nor a hallucination detector."
            ),
        ]
    )
    report = "\n".join(lines) + "\n"
    (output / "REPORT.md").write_text(report, encoding="utf-8")
    return report


def run(*, audit: Path, predictions_path: Path, output: Path) -> dict[str, Any]:
    errors_path = audit / "tables" / "leaf_errors.jsonl"
    schema_path = audit / "tables" / "schema_validation_issues.csv"
    errors = _jsonl(errors_path)
    predictions = _jsonl(predictions_path)
    schema_issues = _csv(schema_path)
    prediction_by_id = {str(row["document_id"]): row for row in predictions}
    if len(prediction_by_id) != len(predictions):
        raise ValueError("prediction document IDs are not unique")

    classified: list[dict[str, Any]] = []
    cross_field_matches = _cross_field_pairs(errors)
    for error_index, row in enumerate(errors):
        document_id = str(row["document_id"])
        prediction = prediction_by_id.get(document_id)
        if prediction is None:
            raise ValueError(f"leaf error has no prediction row: {document_id}")
        mechanism, basis, elsewhere = classify_leaf_error(
            row,
            cross_field_paths=cross_field_matches.get(error_index, ()),
        )
        classified.append(
            {
                **row,
                "mechanism": mechanism,
                "mechanism_basis": basis,
                "elsewhere_paths": list(elsewhere),
            }
        )
    if len(classified) != len(errors):
        raise AssertionError("classified leaf error count changed")

    counts = Counter(str(row["mechanism"]) for row in classified)
    summary_rows = [
        {
            "mechanism": mechanism,
            "leaf_errors": count,
            "fraction": count / len(classified),
        }
        for mechanism, count in counts.most_common()
    ]
    field_counts: dict[str, Counter[str]] = defaultdict(Counter)
    for row in classified:
        field_counts[str(row["field_path"])][str(row["mechanism"])] += 1
    mechanism_names = sorted(counts)
    field_rows = [
        {
            "field_path": field,
            "total_leaf_errors": sum(values.values()),
            **{mechanism: values[mechanism] for mechanism in mechanism_names},
        }
        for field, values in sorted(
            field_counts.items(), key=lambda item: (-sum(item[1].values()), item[0])
        )
    ]
    unsupported_rows = [
        row
        for row in classified
        if row["mechanism"] in {"unsupported_text_candidate", "unsupported_or_wrong_numeric_value"}
    ]
    output_failures = _output_failure_rows(schema_issues, predictions)

    if output.exists():
        raise ValueError(f"analysis output already exists: {output}")
    (output / "tables").mkdir(parents=True)
    _write_jsonl(output / "tables" / "mechanism_errors.jsonl", classified)
    _write_csv(output / "tables" / "mechanism_summary.csv", summary_rows)
    _write_csv(output / "tables" / "mechanism_by_field.csv", field_rows)
    _write_csv(output / "tables" / "unsupported_candidates.csv", unsupported_rows)
    _write_csv(output / "tables" / "output_contract_failures.csv", output_failures)
    _plots(rows=classified, output_failures=output_failures, output=output)
    _report(rows=classified, output_failures=output_failures, output=output)

    manifest = {
        "schema_version": 1,
        "generated_at": datetime.now(UTC).isoformat(),
        "inputs": {
            "leaf_errors": {"path": str(errors_path), "sha256": sha256_file(errors_path)},
            "schema_issues": {"path": str(schema_path), "sha256": sha256_file(schema_path)},
            "predictions": {"path": str(predictions_path), "sha256": sha256_file(predictions_path)},
        },
        "leaf_errors": len(classified),
        "mechanisms": dict(counts),
        "plots": 8,
    }
    atomic_write_json(output / "manifest.json", manifest)
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--audit", type=Path, default=DEFAULT_AUDIT)
    parser.add_argument("--predictions", type=Path, default=DEFAULT_PREDICTIONS)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    print(
        json.dumps(
            run(audit=args.audit, predictions_path=args.predictions, output=args.output),
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
